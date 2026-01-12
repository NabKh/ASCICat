"""
ascicat/calculator.py
Main ASCICalculator Class - The Orchestrator

Central calculation engine that coordinates all ASCICat components:
- Data loading and validation (via DataLoader)
- Reaction configuration management
- Activity/Stability/Cost scoring (via scoring.py)
- ASCI calculation and ranking
- Result storage and export
- Statistical analysis and reporting

This is the primary user interface to the ASCICat framework.

Author: N. Khossossi
Institution: DIFFER (Dutch Institute for Fundamental Energy Research)

Usage:
    >>> from ascicat import ASCICalculator
    >>> calc = ASCICalculator(reaction='HER')
    >>> calc.load_data('data/HER_clean.csv')
    >>> results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)
    >>> top10 = calc.get_top_catalysts(n=10)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import warnings
from datetime import datetime
from tqdm import tqdm

# Import ASCICat components
from .config import (
    ReactionConfig, 
    ASCIConstants,
    get_reaction_config,
    validate_weights,
    normalize_weights
)
from .data_loader import DataLoader
from .scoring import (
    score_activity,
    score_stability,
    score_cost,
    calculate_asci,
    ScoringFunctions
)
from .utils import (
    format_catalyst_name,
    save_to_json,
    create_metadata
)


# ============================================================================
# MAIN CALCULATOR CLASS
# ============================================================================

class ASCICalculator:
    """
    Main calculator for Activity-Stability-Cost Index (ASCI).
    
    This class orchestrates the entire ASCI calculation workflow:
    1. Load and validate catalyst data
    2. Configure reaction-specific parameters
    3. Score individual descriptors (activity, stability, cost)
    4. Calculate unified ASCI metric
    5. Rank and analyze results
    6. Export data and statistics
    
    The calculator automatically handles:
    - HER (Hydrogen Evolution Reaction)
    - CO2RR-CO (CO2 Reduction to CO)
    - CO2RR-CHO (CO2 Reduction via CHO pathway)
    - CO2RR-COCOH (CO2 Reduction via COOH pathway)
    
    Attributes
    ----------
    reaction : str
        Reaction type ('HER' or 'CO2RR')
    pathway : str
        Reaction pathway (e.g., 'CO', 'CHO', 'COCOH' for CO2RR)
    config : ReactionConfig
        Reaction configuration with optimal energies and parameters
    data : pd.DataFrame or None
        Loaded catalyst data
    results : pd.DataFrame or None
        Calculated ASCI results
    scoring_method : str
        Activity scoring method ('linear' or 'gaussian')
    verbose : bool
        Print detailed progress information
    
    Examples
    --------
    Basic HER calculation:
    
    >>> from ascicat import ASCICalculator
    >>> 
    >>> # Initialize for HER
    >>> calc = ASCICalculator(reaction='HER', verbose=True)
    >>> 
    >>> # Load data
    >>> calc.load_data('data/HER_clean.csv')
    >>> 
    >>> # Calculate ASCI with equal weights
    >>> results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)
    >>> 
    >>> # Get top 10 catalysts
    >>> top10 = calc.get_top_catalysts(n=10)
    >>> print(top10[['symbol', 'ASCI', 'activity_score', 'stability_score', 'cost_score']])
    
    CO2RR calculation with specific pathway:
    
    >>> # Initialize for CO2RR-CO pathway
    >>> calc = ASCICalculator(reaction='CO2RR', pathway='CO', verbose=True)
    >>> calc.load_data('data/CO2RR_CO_clean.csv')
    >>> 
    >>> # Calculate with activity-focused weights
    >>> results = calc.calculate_asci(w_a=0.5, w_s=0.3, w_c=0.2, method='linear')
    >>> 
    >>> # Print summary
    >>> calc.print_summary(n_top=20)
    
    Comparing multiple pathways:
    
    >>> # Compare different CO2RR pathways
    >>> pathways = ['CO', 'CHO', 'COCOH']
    >>> for pathway in pathways:
    ...     calc = ASCICalculator('CO2RR', pathway=pathway)
    ...     calc.load_data(f'data/CO2RR_{pathway}_clean.csv')
    ...     results = calc.calculate_asci()
    ...     print(f"{pathway}: Best ASCI = {results['ASCI'].max():.3f}")
    """
    
    def __init__(self,
                 reaction: str,
                 pathway: Optional[str] = None,
                 scoring_method: str = 'linear',
                 verbose: bool = True):
        """
        Initialize ASCICalculator.
        
        Parameters
        ----------
        reaction : str
            Reaction type - must be 'HER' or 'CO2RR'
            - 'HER': Hydrogen Evolution Reaction (2H⁺ + 2e⁻ → H₂)
            - 'CO2RR': CO₂ Reduction Reaction (various pathways)
        
        pathway : str, optional
            Specific reaction pathway (required for CO2RR):
            - 'CO': CO₂ → CO (ΔE_opt = -0.67 eV)
            - 'CHO': CO₂ → CH₃OH via CHO (ΔE_opt = -0.48 eV)
            - 'COCOH': CO₂ → HCOOH via COOH (ΔE_opt = -0.32 eV)
            For HER, pathway is automatically set to 'H_adsorption'
        
        scoring_method : {'linear', 'gaussian'}, optional
            Activity scoring method (default: 'linear')
            - 'linear': Linear decay from optimum, computationally efficient
            - 'gaussian': Exponential decay, sharper discrimination
        
        verbose : bool, optional
            Print detailed information during calculations (default: True)
        
        Raises
        ------
        ValueError
            If reaction is not 'HER' or 'CO2RR'
            If pathway is required but not provided
            If scoring_method is invalid
        
        Examples
        --------
        >>> # HER calculation (pathway automatic)
        >>> calc_her = ASCICalculator(reaction='HER')
        >>> 
        >>> # CO2RR-CO calculation (pathway required)
        >>> calc_co = ASCICalculator(reaction='CO2RR', pathway='CO')
        >>> 
        >>> # With Gaussian scoring and quiet mode
        >>> calc = ASCICalculator('HER', scoring_method='gaussian', verbose=False)
        """
        # Validate reaction type
        reaction = reaction.upper()
        if reaction not in ['HER', 'CO2RR']:
            raise ValueError(
                f"Reaction must be 'HER' or 'CO2RR', got '{reaction}'"
            )
        
        # Special handling for HER vs CO2RR
        if reaction == 'HER':
            # HER has only one pathway (H adsorption)
            if pathway is not None and pathway not in ['H_adsorption', 'default']:
                warnings.warn(
                    f"HER has only one pathway. Ignoring pathway='{pathway}'",
                    category=UserWarning
                )
            pathway = 'H_adsorption'
            
        elif reaction == 'CO2RR':
            # CO2RR requires pathway specification
            if pathway is None:
                warnings.warn(
                    "No pathway specified for CO2RR. Using default 'CO' pathway. "
                    "Available pathways: CO, CHO, COCOH",
                    category=UserWarning
                )
                pathway = 'CO'
            
            # Validate CO2RR pathway
            valid_pathways = ['CO', 'CHO', 'COCOH', 'default']
            if pathway not in valid_pathways:
                raise ValueError(
                    f"Invalid CO2RR pathway: '{pathway}'. "
                    f"Must be one of: {', '.join(valid_pathways)}"
                )
        
        # Validate scoring method
        scoring_method = scoring_method.lower()
        if scoring_method not in ['linear', 'gaussian']:
            raise ValueError(
                f"Scoring method must be 'linear' or 'gaussian', "
                f"got '{scoring_method}'"
            )
        
        # Store configuration
        self.reaction = reaction
        self.pathway = pathway
        self.scoring_method = scoring_method
        self.verbose = verbose
        
        # Load reaction configuration
        self.config = get_reaction_config(reaction, pathway)
        
        # Initialize data storage
        self.data = None              # Raw loaded data
        self.results = None           # Calculated ASCI results
        self.statistics = None        # Statistical summary
        
        # Initialize data loader
        self.data_loader = DataLoader(self.config, verbose=self.verbose)
        
        # Print initialization info
        if self.verbose:
            self._print_initialization_info()
    
    def _print_initialization_info(self):
        """Print calculator initialization information."""
        print("\n" + "="*80)
        print("ASCICat Calculator Initialized")
        print("="*80)
        print(f"\n🔬 Reaction: {self.reaction}")
        
        # Pathway info with optimal energy
        if self.reaction == 'HER':
            print(f"   Pathway: H adsorption")
            print(f"   Optimal Energy: ΔE_opt = {self.config.optimal_energy:.3f} eV")
            print(f"   Description: Hydrogen Evolution Reaction (2H⁺ + 2e⁻ → H₂)")
        
        elif self.reaction == 'CO2RR':
            print(f"   Pathway: {self.pathway}")
            print(f"   Optimal Energy: ΔE_opt = {self.config.optimal_energy:.3f} eV")
            
            # Pathway-specific descriptions
            if self.pathway == 'CO':
                print(f"   Description: CO₂ → CO (CO₂ + 2H⁺ + 2e⁻ → CO + H₂O)")
            elif self.pathway == 'CHO':
                print(f"   Description: CO₂ → CH₃OH via CHO intermediate")
            elif self.pathway == 'COCOH':
                print(f"   Description: CO₂ → HCOOH via COOH intermediate")
        
        print(f"\n⚙️  Configuration:")
        print(f"   Activity Width: σ_a = {self.config.activity_width:.3f} eV")
        print(f"   Scoring Method: {self.scoring_method.upper()}")
        print(f"   Default Weights: w_a={self.config.default_weights[0]:.2f}, "
              f"w_s={self.config.default_weights[1]:.2f}, "
              f"w_c={self.config.default_weights[2]:.2f} (EQUAL)")
        print("="*80 + "\n")
    
    def load_data(self, 
                  file_path: str,
                  validate: bool = True) -> pd.DataFrame:
        """
        Load catalyst data from CSV file.
        
        Loads and validates catalyst database containing DFT descriptors
        and material properties. Automatically handles both HER and CO2RR
        data formats with appropriate validation.
        
        Parameters
        ----------
        file_path : str
            Path to CSV file containing catalyst data
            Required columns:
            - 'DFT_ads_E': Adsorption energy (eV) - activity descriptor
            - 'surface_energy': Surface energy (J/m²) - stability descriptor
            - 'Cost': Material cost ($/kg) - economic descriptor
            - 'symbol': Catalyst identifier
            - 'AandB': Detailed catalyst composition
            - 'reaction_type': Reaction name
            - 'optimal_energy': Sabatier optimum for this reaction
            - 'activity_width': Scoring width parameter
        
        validate : bool, optional
            Perform data quality validation (default: True)
            Checks for:
            - Missing values
            - Out-of-range descriptors
            - Duplicate entries
            - Physical validity
        
        Returns
        -------
        pd.DataFrame
            Loaded and validated catalyst data
        
        Raises
        ------
        FileNotFoundError
            If data file doesn't exist
        ValueError
            If required columns are missing
            If data validation fails
        
        Notes
        -----
        Data Quality Checks:
        - Adsorption energies: |ΔE| < 10 eV (sanity check)
        - Surface energies: 0 < γ < 15 J/m² (physical bounds)
        - Costs: $0.01 < C < $1M per kg (reasonable range)
        - No missing values in critical columns
        - No duplicate catalysts
        
        For HER data:
        Expected format includes bimetallic catalysts with various
        surface facets (111, 100, 110, etc.)
        
        For CO2RR data:
        Expected format includes pathway-specific intermediates
        (CO*, CHO*, COOH*) with corresponding binding energies
        
        Examples
        --------
        >>> # Load HER data
        >>> calc = ASCICalculator('HER')
        >>> data = calc.load_data('data/HER_clean.csv')
        >>> print(f"Loaded {len(data)} HER catalysts")
        
        >>> # Load CO2RR-CO data
        >>> calc = ASCICalculator('CO2RR', pathway='CO')
        >>> data = calc.load_data('data/CO2RR_CO_clean.csv')
        >>> print(f"Loaded {len(data)} CO2RR-CO catalysts")
        
        >>> # Load without validation (faster, but risky)
        >>> data = calc.load_data('data/HER_clean.csv', validate=False)
        """
        if self.verbose:
            print(f"\n📂 Loading {self.reaction} data from: {file_path}")
            if self.reaction == 'CO2RR':
                print(f"   Pathway: {self.pathway}")
        
        # Use DataLoader for validation and loading
        self.data = self.data_loader.load(file_path, validate=validate)
        
        # Store original data size for reference
        self._original_data_size = len(self.data)
        
        # Print data summary if verbose
        if self.verbose:
            self._print_data_summary()
        
        return self.data
    
    def _print_data_summary(self):
        """Print summary of loaded data."""
        if self.data is None:
            print("⚠️  No data loaded")
            return
        
        print(f"\n📊 Data Summary:")
        print(f"   Total catalysts: {len(self.data):,}")
        
        # Descriptor statistics
        print(f"\n   Activity Descriptor (ΔE):")
        print(f"      Range: [{self.data['DFT_ads_E'].min():+.3f}, "
              f"{self.data['DFT_ads_E'].max():+.3f}] eV")
        print(f"      Mean:  {self.data['DFT_ads_E'].mean():+.3f} eV")
        print(f"      Optimal for {self.reaction}: {self.config.optimal_energy:+.3f} eV")
        
        print(f"\n   Stability Descriptor (γ):")
        print(f"      Range: [{self.data['surface_energy'].min():.3f}, "
              f"{self.data['surface_energy'].max():.3f}] J/m²")
        print(f"      Mean:  {self.data['surface_energy'].mean():.3f} J/m²")
        
        print(f"\n   Cost Descriptor:")
        print(f"      Range: [${self.data['Cost'].min():.2f}, "
              f"${self.data['Cost'].max():,.0f}] per kg")
        print(f"      Median: ${self.data['Cost'].median():,.0f} per kg")
        
        # Composition info if available
        if 'Ametal' in self.data.columns:
            n_elements = self.data['Ametal'].nunique()
            print(f"\n   Unique primary elements: {n_elements}")
            
            if n_elements <= 10:
                elements = sorted(self.data['Ametal'].unique())
                print(f"      Elements: {', '.join(map(str, elements))}")
    
    def calculate_asci(self,
                      w_a: float = 0.33,
                      w_s: float = 0.33,
                      w_c: float = 0.34,
                      method: Optional[str] = None,
                      show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate Activity-Stability-Cost Index for all catalysts.
        
        This is the MAIN CALCULATION METHOD that orchestrates:
        1. Activity scoring (Sabatier principle)
        2. Stability scoring (surface energy)
        3. Cost scoring (economic viability)
        4. Weighted integration into unified ASCI metric
        
        Parameters
        ----------
        w_a : float, optional
            Activity weight [0, 1] (default: 0.33)
            Higher values prioritize catalytic performance
        
        w_s : float, optional
            Stability weight [0, 1] (default: 0.33)
            Higher values prioritize electrochemical durability
        
        w_c : float, optional
            Cost weight [0, 1] (default: 0.34)
            Higher values prioritize economic viability
            
            Constraint: w_a + w_s + w_c must equal 1.0
        
        method : {'linear', 'gaussian'}, optional
            Activity scoring method (default: use initialized method)
            Override the method specified during initialization
        
        show_progress : bool, optional
            Display progress bar during calculation (default: True)
        
        Returns
        -------
        pd.DataFrame
            Results DataFrame with columns:
            - All original columns from input data
            - 'activity_score': Activity scores [0, 1]
            - 'stability_score': Stability scores [0, 1]
            - 'cost_score': Cost scores [0, 1]
            - 'ASCI': Combined ASCI scores [0, 1]
            - 'rank': Ranking (1 = best)
            Sorted by ASCI in descending order
        
        Raises
        ------
        ValueError
            If data not loaded (call load_data first)
            If weights invalid or don't sum to 1
        
        Notes
        -----
        Weight Selection Guidelines:
        
        1. Equal Weights (0.33, 0.33, 0.34) - DEFAULT
           - Unbiased exploratory screening
           - Recommended for initial surveys
           - Treats all objectives equally
        
        2. Activity-Focused (0.5, 0.3, 0.2)
           - Performance-critical applications
           - Research/fundamental studies
           - When activity is limiting factor
        
        3. Stability-Focused (0.3, 0.5, 0.2)
           - Long-term operation required
           - Harsh conditions (acidic/oxidizing)
           - Industrial durability emphasis
        
        4. Cost-Focused (0.3, 0.2, 0.5)
           - Large-scale deployment
           - Earth-abundant materials priority
           - Economic viability critical
        
        Scoring Details:
        
        For HER (ΔE_opt = -0.27 eV):
        - Perfect binding at -0.27 eV → S_a = 1.0
        - Too weak (ΔE > -0.12 eV) → S_a approaches 0
        - Too strong (ΔE < -0.42 eV) → S_a approaches 0
        
        For CO2RR-CO (ΔE_opt = -0.67 eV):
        - Perfect CO binding at -0.67 eV → S_a = 1.0
        - Ag/Au typically near optimum
        
        For CO2RR-CHO (ΔE_opt = -0.48 eV):
        - Optimal for methanol production
        - Cu-based catalysts often favorable
        
        For CO2RR-COCOH (ΔE_opt = -0.32 eV):
        - Optimal for formic acid production
        - Sn/Pb-based catalysts perform well
        
        Examples
        --------
        >>> # Basic HER calculation with equal weights
        >>> calc = ASCICalculator('HER')
        >>> calc.load_data('data/HER_clean.csv')
        >>> results = calc.calculate_asci()  # Uses defaults
        >>> print(f"Best ASCI: {results['ASCI'].max():.3f}")
        
        >>> # CO2RR-CO with activity focus
        >>> calc = ASCICalculator('CO2RR', pathway='CO')
        >>> calc.load_data('data/CO2RR_CO_clean.csv')
        >>> results = calc.calculate_asci(w_a=0.5, w_s=0.3, w_c=0.2)
        >>> 
        >>> # Get top catalyst
        >>> top = results.iloc[0]
        >>> print(f"Top catalyst: {top['symbol']}")
        >>> print(f"  ASCI: {top['ASCI']:.3f}")
        >>> print(f"  Activity: {top['activity_score']:.3f}")
        >>> print(f"  Stability: {top['stability_score']:.3f}")
        >>> print(f"  Cost: {top['cost_score']:.3f}")
        
        >>> # Compare different weight scenarios
        >>> scenarios = [
        ...     (0.33, 0.33, 0.34, 'Equal'),
        ...     (0.5, 0.3, 0.2, 'Activity'),
        ...     (0.3, 0.2, 0.5, 'Cost')
        ... ]
        >>> for w_a, w_s, w_c, name in scenarios:
        ...     results = calc.calculate_asci(w_a, w_s, w_c)
        ...     top_cat = results.iloc[0]['symbol']
        ...     print(f"{name:10s}: {top_cat}")
        """
        # Check if data is loaded
        if self.data is None:
            raise ValueError(
                "No data loaded. Call load_data() before calculate_asci()"
            )
        
        # Validate and normalize weights
        validate_weights(w_a, w_s, w_c)
        w_a, w_s, w_c = normalize_weights(w_a, w_s, w_c)  # Ensure exact sum
        
        # Determine scoring method
        if method is None:
            method = self.scoring_method
        method = method.lower()
        
        # Print calculation info
        if self.verbose:
            self._print_calculation_info(w_a, w_s, w_c, method)
        
        # Create copy of data for results
        results = self.data.copy()
        n_catalysts = len(results)
        
        # Initialize progress tracking
        if show_progress and self.verbose:
            pbar = tqdm(total=4, desc="Calculating ASCI", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        
        # ====================================================================
        # STEP 1: ACTIVITY SCORING
        # ====================================================================
        if show_progress and self.verbose:
            pbar.set_description("Scoring Activity")
        
        # Get adsorption energies
        delta_E = results['DFT_ads_E'].values
        
        # Score activity based on method and reaction
        if self.reaction == 'HER':
            # HER: Optimal at -0.27 eV (thermoneutral H binding)
            activity_scores = score_activity(
                delta_E,
                optimal_E=self.config.optimal_energy,  # -0.27 eV
                width=self.config.activity_width,      # 0.15 eV
                method=method
            )
        
        elif self.reaction == 'CO2RR':
            # CO2RR: Pathway-specific optimal energies from config
            # Uses self.config which is set based on pathway during __init__
            activity_scores = score_activity(
                delta_E,
                optimal_E=self.config.optimal_energy,
                width=self.config.activity_width,
                method=method
            )
        
        results['activity_score'] = activity_scores
        
        if show_progress and self.verbose:
            pbar.update(1)
        
        # ====================================================================
        # STEP 2: STABILITY SCORING
        # ====================================================================
        if show_progress and self.verbose:
            pbar.set_description("Scoring Stability")

        # Get surface energies
        gamma = results['surface_energy'].values

        # Score stability (inverse: lower γ → higher stability)
        # Uses data-driven normalization (min/max from actual data)
        stability_scores = score_stability(gamma)

        results['stability_score'] = stability_scores

        if show_progress and self.verbose:
            pbar.update(1)

        # ====================================================================
        # STEP 3: COST SCORING
        # ====================================================================
        if show_progress and self.verbose:
            pbar.set_description("Scoring Cost")

        # Get material costs
        costs = results['Cost'].values

        # Score cost (logarithmic: lower cost → higher score)
        # Uses data-driven normalization (min/max from actual data)
        cost_scores = score_cost(costs)

        results['cost_score'] = cost_scores
        
        if show_progress and self.verbose:
            pbar.update(1)
        
        # ====================================================================
        # STEP 4: COMBINED ASCI CALCULATION
        # ====================================================================
        if show_progress and self.verbose:
            pbar.set_description("Calculating ASCI")
        
        # Calculate weighted combination
        asci_scores = calculate_asci(
            activity_scores,
            stability_scores,
            cost_scores,
            w_a, w_s, w_c
        )
        
        results['ASCI'] = asci_scores
        
        if show_progress and self.verbose:
            pbar.update(1)
            pbar.close()
        
        # ====================================================================
        # FINALIZATION
        # ====================================================================
        
        # Sort by ASCI (descending)
        results = results.sort_values('ASCI', ascending=False).reset_index(drop=True)
        
        # Add ranking
        results['rank'] = range(1, len(results) + 1)
        
        # Store results and weights
        self.results = results
        self._last_weights = (w_a, w_s, w_c)
        self._last_method = method
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Print completion message
        if self.verbose:
            print(f"\n✓ ASCI calculation complete!")
            print(f"   Processed: {n_catalysts:,} catalysts")
            print(f"   Best ASCI: {results['ASCI'].max():.4f}")
            print(f"   Mean ASCI: {results['ASCI'].mean():.4f}")
            print(f"   Top catalyst: {results.iloc[0]['symbol']}\n")
        
        return results
    
    def _print_calculation_info(self, w_a: float, w_s: float, w_c: float, method: str):
        """Print calculation configuration."""
        print("\n" + "="*80)
        print(f"Calculating ASCI for {self.reaction}")
        if self.reaction == 'CO2RR':
            print(f"Pathway: {self.pathway} (ΔE_opt = {self.config.optimal_energy:.3f} eV)")
        print("="*80)
        
        print(f"\n⚙️  Configuration:")
        print(f"   Weights: w_a={w_a:.2f}, w_s={w_s:.2f}, w_c={w_c:.2f}")
        
        # Interpret weights
        if abs(w_a - w_s) < 0.05 and abs(w_a - w_c) < 0.05:
            print(f"   Strategy: EQUAL weights (unbiased screening)")
        elif w_a > w_s and w_a > w_c:
            print(f"   Strategy: ACTIVITY-FOCUSED")
        elif w_s > w_a and w_s > w_c:
            print(f"   Strategy: STABILITY-FOCUSED")
        elif w_c > w_a and w_c > w_s:
            print(f"   Strategy: COST-FOCUSED")
        
        print(f"   Activity Scoring: {method.upper()}")
        print(f"   Catalysts: {len(self.data):,}")
        print("="*80)
    
    def _calculate_statistics(self):
        """Calculate statistical summary of results."""
        if self.results is None:
            return
        
        self.statistics = {
            'n_catalysts': len(self.results),
            'weights': {
                'activity': self._last_weights[0],
                'stability': self._last_weights[1],
                'cost': self._last_weights[2]
            },
            'method': self._last_method,
            'asci': {
                'mean': float(self.results['ASCI'].mean()),
                'std': float(self.results['ASCI'].std()),
                'min': float(self.results['ASCI'].min()),
                'max': float(self.results['ASCI'].max()),
                'median': float(self.results['ASCI'].median()),
            },
            'activity': {
                'mean': float(self.results['activity_score'].mean()),
                'std': float(self.results['activity_score'].std()),
            },
            'stability': {
                'mean': float(self.results['stability_score'].mean()),
                'std': float(self.results['stability_score'].std()),
            },
            'cost': {
                'mean': float(self.results['cost_score'].mean()),
                'std': float(self.results['cost_score'].std()),
            }
        }
    
    def get_top_catalysts(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N catalysts by ASCI score.
        
        Parameters
        ----------
        n : int, optional
            Number of top catalysts to return (default: 10)
        
        Returns
        -------
        pd.DataFrame
            Top N catalysts sorted by ASCI score
        
        Raises
        ------
        ValueError
            If results not calculated (call calculate_asci first)
        
        Examples
        --------
        >>> calc = ASCICalculator('HER')
        >>> calc.load_data('data/HER_clean.csv')
        >>> calc.calculate_asci()
        >>> top10 = calc.get_top_catalysts(n=10)
        >>> print(top10[['rank', 'symbol', 'ASCI']])
        """
        if self.results is None:
            raise ValueError(
                "No results available. Run calculate_asci() first."
            )
        
        return self.results.head(n).copy()
    
    def get_statistics(self) -> Dict:
        """
        Get statistical summary of results.
        
        Returns
        -------
        dict
            Statistics dictionary containing:
            - n_catalysts: Total number of catalysts
            - weights: Applied weights (w_a, w_s, w_c)
            - method: Scoring method used
            - asci: ASCI statistics (mean, std, min, max, median)
            - activity: Activity score statistics
            - stability: Stability score statistics
            - cost: Cost score statistics
        
        Examples
        --------
        >>> stats = calc.get_statistics()
        >>> print(f"Mean ASCI: {stats['asci']['mean']:.3f}")
        >>> print(f"Best ASCI: {stats['asci']['max']:.3f}")
        """
        if self.statistics is None:
            raise ValueError(
                "No statistics available. Run calculate_asci() first."
            )
        
        return self.statistics.copy()
    
    def print_summary(self, n_top: int = 10):
        """
        Print comprehensive summary of results.
        
        Parameters
        ----------
        n_top : int, optional
            Number of top catalysts to display (default: 10)
        
        Examples
        --------
        >>> calc.calculate_asci()
        >>> calc.print_summary(n_top=20)
        """
        if self.results is None:
            print("⚠️  No results to summarize. Run calculate_asci() first.")
            return
        
        print("\n" + "="*80)
        print(f"ASCI Results Summary - {self.reaction}")
        if self.reaction == 'CO2RR':
            print(f"Pathway: {self.pathway}")
        print("="*80)
        
        # Statistics
        stats = self.statistics
        print(f"\n📊 Statistics:")
        print(f"   Total catalysts: {stats['n_catalysts']:,}")
        print(f"   ASCI range: [{stats['asci']['min']:.4f}, {stats['asci']['max']:.4f}]")
        print(f"   ASCI mean: {stats['asci']['mean']:.4f} ± {stats['asci']['std']:.4f}")
        print(f"   ASCI median: {stats['asci']['median']:.4f}")
        
        # Weights used
        print(f"\n⚖️  Weights Applied:")
        print(f"   Activity:  {stats['weights']['activity']:.2f}")
        print(f"   Stability: {stats['weights']['stability']:.2f}")
        print(f"   Cost:      {stats['weights']['cost']:.2f}")
        
        # Top catalysts
        print(f"\n🏆 Top {n_top} Catalysts:")
        print("-" * 80)
        
        top = self.get_top_catalysts(n_top)
        
        # Format display
        display_cols = ['rank', 'symbol', 'ASCI', 'activity_score', 
                       'stability_score', 'cost_score']
        
        # Check which columns exist
        display_cols = [col for col in display_cols if col in top.columns]
        
        print(f"{'Rank':>5} {'Catalyst':>15} {'ASCI':>8} {'Activity':>10} "
              f"{'Stability':>10} {'Cost':>8}")
        print("-" * 80)
        
        for idx, row in top.iterrows():
            print(f"{row['rank']:>5} {row['symbol']:>15} {row['ASCI']:>8.4f} "
                  f"{row['activity_score']:>10.4f} {row['stability_score']:>10.4f} "
                  f"{row['cost_score']:>8.4f}")
        
        print("="*80 + "\n")
    
    def save_results(self, 
                    file_path: str,
                    include_metadata: bool = True):
        """
        Save results to CSV file.
        
        Parameters
        ----------
        file_path : str
            Output file path
        include_metadata : bool, optional
            Save metadata JSON alongside CSV (default: True)
        
        Examples
        --------
        >>> calc.calculate_asci()
        >>> calc.save_results('results/HER_results.csv')
        """
        if self.results is None:
            raise ValueError(
                "No results to save. Run calculate_asci() first."
            )
        
        # Create output directory
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        self.results.to_csv(file_path, index=False)
        
        if self.verbose:
            print(f"✓ Results saved to: {file_path}")
        
        # Save metadata
        if include_metadata:
            metadata_path = file_path.with_suffix('.json')
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'reaction': self.reaction,
                'pathway': self.pathway,
                'n_catalysts': len(self.results),
                'weights': self._last_weights,
                'method': self._last_method,
                'statistics': self.statistics
            }
            save_to_json(metadata, str(metadata_path))
            
            if self.verbose:
                print(f"✓ Metadata saved to: {metadata_path}")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def quick_asci(reaction: str,
               data_file: str,
               pathway: Optional[str] = None,
               w_a: float = 0.33,
               w_s: float = 0.33,
               w_c: float = 0.34,
               method: str = 'linear',
               verbose: bool = True) -> pd.DataFrame:
    """
    Quick one-line ASCI calculation.
    
    Convenience function for rapid screening without explicit class usage.
    
    Parameters
    ----------
    reaction : str
        'HER' or 'CO2RR'
    data_file : str
        Path to data CSV
    pathway : str, optional
        Reaction pathway (required for CO2RR)
    w_a, w_s, w_c : float, optional
        Weights (default: equal)
    method : str, optional
        'linear' or 'gaussian' (default: 'linear')
    verbose : bool, optional
        Print information (default: True)
    
    Returns
    -------
    pd.DataFrame
        ASCI results
    
    Examples
    --------
    >>> from ascicat import quick_asci
    >>> results = quick_asci('HER', 'data/HER_clean.csv')
    >>> print(results.head())
    """
    calc = ASCICalculator(reaction, pathway=pathway, 
                         scoring_method=method, verbose=verbose)
    calc.load_data(data_file)
    results = calc.calculate_asci(w_a, w_s, w_c, method=method)
    return results


# ============================================================================
# MODULE-LEVEL TESTS
# ============================================================================

if __name__ == '__main__':
    print("Testing ASCICalculator")
    print("="*80)
    
    # Note: Requires actual data files to run
    print("\nThis module requires data files to test.")
    print("Run examples/example_1_basic_HER.py for full demonstration.")
    
    print("\n✓ Module loaded successfully")
    print("="*80)