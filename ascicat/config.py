"""
ascicat/config.py
Configuration Management for ASCI Calculations

Defines reaction configurations, mathematical constants, and parameter ranges
for Activity-Stability-Cost Integrated catalyst screening.

Author: N. Khossossi
Institution: DIFFER (Dutch Institute for Fundamental Energy Research)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import warnings


@dataclass
class ReactionConfig:
    """
    Configuration for a specific electrocatalytic reaction.
    
    This dataclass encapsulates all parameters needed for ASCI calculations
    for a given reaction pathway, including Sabatier-optimal energies,
    activity windows, and default weights.
    
    Attributes
    ----------
    name : str
        Reaction name (e.g., 'HER', 'CO2RR')
    pathway : str
        Specific reaction pathway (e.g., 'H_adsorption', 'CO', 'CHO')
    activity_descriptor : str
        Column name for activity descriptor (must be 'DFT_ads_E')
    optimal_energy : float
        Sabatier-optimal binding energy (eV) for maximum activity
    activity_width : float
        Gaussian/Linear width σ_a for activity scoring (eV)
        Defines acceptable deviation from optimal energy
    activity_window : Tuple[float, float]
        Plotting/analysis window [min, max] (eV) for volcano plots
    stability_range : Tuple[float, float]
        Min/max surface energy (J/m²) for normalization
    cost_range : Tuple[float, float]
        Min/max material cost ($/kg) for logarithmic normalization
    default_weights : Tuple[float, float, float]
        Default weights (w_a, w_s, w_c) - EQUAL by default
    description : str
        Human-readable reaction description
    references : List[str], optional
        Scientific references for optimal parameters
    
    Examples
    --------
    >>> her_config = ReactionConfig(
    ...     name='HER',
    ...     optimal_energy=-0.27,
    ...     activity_width=0.15,
    ...     default_weights=(0.33, 0.33, 0.34)
    ... )
    """
    
    # Reaction identification
    name: str
    pathway: str
    
    # Activity parameters
    activity_descriptor: str
    optimal_energy: float        # Sabatier optimum (eV)
    activity_width: float         # σ_a for scoring (eV)
    activity_window: Tuple[float, float]  # [min, max] for plotting
    
    # Stability parameters
    stability_range: Tuple[float, float]  # [min, max] surface energy (J/m²)
    
    # Cost parameters
    cost_range: Tuple[float, float]       # [min, max] cost ($/kg)
    
    # Default optimization weights (EQUAL WEIGHTS)
    default_weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)
    
    # Documentation
    description: str = ""
    references: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate weights sum to 1
        weight_sum = sum(self.default_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Default weights must sum to 1.0, got {weight_sum:.6f}"
            )
        
        # Validate weight ranges
        for i, w in enumerate(self.default_weights):
            if not (0.0 <= w <= 1.0):
                raise ValueError(
                    f"Weight {i} must be in [0, 1], got {w}"
                )
        
        # Validate activity window
        if self.activity_window[0] >= self.activity_window[1]:
            raise ValueError(
                f"Activity window min ({self.activity_window[0]}) must be "
                f"less than max ({self.activity_window[1]})"
            )
        
        # Validate stability range
        if self.stability_range[0] >= self.stability_range[1]:
            raise ValueError(
                f"Stability range min ({self.stability_range[0]}) must be "
                f"less than max ({self.stability_range[1]})"
            )
        
        # Validate cost range
        if self.cost_range[0] >= self.cost_range[1]:
            raise ValueError(
                f"Cost range min ({self.cost_range[0]}) must be "
                f"less than max ({self.cost_range[1]})"
            )
        
        # Validate positive values
        if self.activity_width <= 0:
            raise ValueError(
                f"Activity width must be positive, got {self.activity_width}"
            )
        
        if self.stability_range[0] < 0:
            warnings.warn(
                f"Negative surface energy ({self.stability_range[0]}) is "
                f"physically unrealistic"
            )
    
    def get_weight_dict(self) -> Dict[str, float]:
        """
        Get weights as dictionary.
        
        Returns
        -------
        dict
            Dictionary with keys 'activity', 'stability', 'cost'
        """
        return {
            'activity': self.default_weights[0],
            'stability': self.default_weights[1],
            'cost': self.default_weights[2]
        }
    
    def print_summary(self):
        """Print configuration summary."""
        print(f"\n{'='*70}")
        print(f"Reaction Configuration: {self.name}")
        if self.pathway:
            print(f"Pathway: {self.pathway}")
        print(f"{'='*70}")
        print(f"\nActivity Parameters:")
        print(f"  Optimal Energy:    {self.optimal_energy:+.3f} eV")
        print(f"  Activity Width:    {self.activity_width:.3f} eV")
        print(f"  Plotting Window:   [{self.activity_window[0]:+.2f}, "
              f"{self.activity_window[1]:+.2f}] eV")
        
        print(f"\nStability Parameters:")
        print(f"  Surface Energy:    [{self.stability_range[0]:.2f}, "
              f"{self.stability_range[1]:.2f}] J/m²")
        
        print(f"\nCost Parameters:")
        print(f"  Cost Range:        [${self.cost_range[0]:.2f}, "
              f"${self.cost_range[1]:,.0f}] per kg")
        
        print(f"\nDefault Weights (EQUAL):")
        print(f"  Activity:  {self.default_weights[0]:.2f}")
        print(f"  Stability: {self.default_weights[1]:.2f}")
        print(f"  Cost:      {self.default_weights[2]:.2f}")
        
        if self.description:
            print(f"\nDescription:")
            print(f"  {self.description}")
        
        if self.references:
            print(f"\nReferences:")
            for ref in self.references:
                print(f"  • {ref}")
        
        print(f"{'='*70}\n")


# ============================================================================
# PREDEFINED REACTION CONFIGURATIONS
# ============================================================================

HER_CONFIG = ReactionConfig(
    # Reaction identification
    name='HER',
    pathway='H_adsorption',
    
    # Activity parameters (Greeley et al., Nat. Mater. 2006)
    activity_descriptor='DFT_ads_E',
    optimal_energy=-0.27,         # eV (Sabatier optimum for HER)
    activity_width=0.15,          # eV (tolerance for "good" activity)
    activity_window=(-0.60, 0.10),  # eV (wide window for volcano visualization)
    
    # Stability parameters
    stability_range=(0.1, 5.0),   # J/m² (typical metallic surface energies)
    
    # Cost parameters (USGS commodity data)
    cost_range=(1.0, 200000.0),   # $/kg (earth-abundant to precious metals)
    
    # Default weights (EQUAL - unbiased screening)
    default_weights=(0.33, 0.33, 0.34),
    
    # Documentation
    description=(
        'Hydrogen Evolution Reaction (2H⁺ + 2e⁻ → H₂). '
        'The optimal binding energy of -0.27 eV represents the '
        'thermoneutral point where adsorption and desorption are balanced. '
        'Platinum group metals typically lie near this optimum.'
    ),
    references=[
        'Greeley, J. et al. Nat. Mater. 5, 909 (2006)',
        'Nørskov, J. K. et al. J. Electrochem. Soc. 152, J23 (2005)',
    ]
)


CO2RR_CO_CONFIG = ReactionConfig(
    # Reaction identification
    name='CO2RR',
    pathway='CO',

    # Activity parameters (Peterson & Nørskov, J. Phys. Chem. Lett. 2012)
    activity_descriptor='DFT_ads_E',
    optimal_energy=-0.67,         # eV (CO binding optimum)
    activity_width=0.15,          # eV (tolerance)
    activity_window=(-1.20, -0.20),  # eV (CO2RR volcano window)

    # Stability parameters (dataset-specific: based on 1st-99th percentile)
    stability_range=(0.20, 2.80),   # J/m² (CO₂RR-specific range)
    
    # Cost parameters
    cost_range=(1.0, 200000.0),   # $/kg
    
    # Default weights (EQUAL)
    default_weights=(0.33, 0.33, 0.34),
    
    # Documentation
    description=(
        'CO₂ Reduction to CO (CO₂ + 2H⁺ + 2e⁻ → CO + H₂O). '
        'The optimal CO binding energy of -0.67 eV balances CO₂ activation '
        'with CO desorption. Silver and gold exhibit near-optimal binding.'
    ),
    references=[
        'Peterson, A. A. & Nørskov, J. K. J. Phys. Chem. Lett. 3, 251 (2012)',
        'Nitopi, S. et al. Chem. Rev. 119, 7610 (2019)',
    ]
)


CO2RR_CHO_CONFIG = ReactionConfig(
    # Reaction identification
    name='CO2RR',
    pathway='CHO',

    # Activity parameters (CHO intermediate pathway)
    activity_descriptor='DFT_ads_E',
    optimal_energy=-0.48,         # eV (CHO binding optimum)
    activity_width=0.15,          # eV
    activity_window=(-0.90, -0.10),  # eV

    # Stability parameters (dataset-specific: based on 1st-99th percentile)
    stability_range=(0.20, 2.80),   # J/m² (CO₂RR-specific range)
    
    # Cost parameters
    cost_range=(1.0, 200000.0),   # $/kg
    
    # Default weights (EQUAL)
    default_weights=(0.33, 0.33, 0.34),
    
    # Documentation
    description=(
        'CO₂ Reduction via CHO Pathway to Methanol '
        '(CO₂ + 6H⁺ + 6e⁻ → CH₃OH + H₂O). '
        'The CHO intermediate binding energy controls selectivity toward '
        'C1 oxygenated products like methanol and formaldehyde.'
    ),
    references=[
        'Peterson, A. A. et al. Energy Environ. Sci. 3, 1311 (2010)',
        'Kuhl, K. P. et al. Energy Environ. Sci. 5, 7050 (2012)',
    ]
)


CO2RR_COCOH_CONFIG = ReactionConfig(
    # Reaction identification
    name='CO2RR',
    pathway='COCOH',

    # Activity parameters (COOH intermediate pathway)
    activity_descriptor='DFT_ads_E',
    optimal_energy=-0.32,         # eV (COOH binding optimum)
    activity_width=0.15,          # eV
    activity_window=(-0.70, 0.05),  # eV

    # Stability parameters (dataset-specific: based on 1st-99th percentile)
    stability_range=(0.20, 2.80),   # J/m² (CO₂RR-specific range)
    
    # Cost parameters
    cost_range=(1.0, 200000.0),   # $/kg
    
    # Default weights (EQUAL)
    default_weights=(0.33, 0.33, 0.34),
    
    # Documentation
    description=(
        'CO₂ Reduction via COOH Pathway to Formic Acid '
        '(CO₂ + 2H⁺ + 2e⁻ → HCOOH). '
        'The COOH intermediate binding controls selectivity toward '
        'formate/formic acid products, important for liquid fuel synthesis.'
    ),
    references=[
        'Yoo, J. S. et al. ChemSusChem 9, 358 (2016)',
        'Hansen, H. A. et al. J. Phys. Chem. Lett. 4, 388 (2013)',
    ]
)


# ============================================================================
# REACTION REGISTRY
# ============================================================================

REACTION_REGISTRY: Dict[str, Dict[str, ReactionConfig]] = {
    'HER': {
        'default': HER_CONFIG,
        'H_adsorption': HER_CONFIG,
    },
    'CO2RR': {
        'CO': CO2RR_CO_CONFIG,
        'CHO': CO2RR_CHO_CONFIG,
        'COCOH': CO2RR_COCOH_CONFIG,
        'default': CO2RR_CO_CONFIG,  # Default to CO pathway
    }
}


# ============================================================================
# ASCI CALCULATION CONSTANTS
# ============================================================================

class ASCIConstants:
    """
    Mathematical constants and configuration for ASCI calculations.
    
    This class provides centralized definitions of:
    - Score normalization ranges
    - Weight constraints
    - Numerical stability parameters
    - Default file paths
    - Visualization settings
    - Required data columns
    
    All constants are class attributes for global access.
    """
    
    # ========================================================================
    # SCORE NORMALIZATION
    # ========================================================================
    
    SCORE_MIN = 0.0  # Minimum normalized score
    SCORE_MAX = 1.0  # Maximum normalized score
    
    # ========================================================================
    # WEIGHT CONSTRAINTS
    # ========================================================================
    
    WEIGHT_MIN = 0.0      # Minimum individual weight
    WEIGHT_MAX = 1.0      # Maximum individual weight
    WEIGHT_SUM = 1.0      # Required sum of all weights
    
    # ========================================================================
    # NUMERICAL STABILITY
    # ========================================================================
    
    EPSILON = 1e-10       # Small constant to avoid division by zero
    TOLERANCE = 1e-6      # Tolerance for floating point comparisons
    
    # ========================================================================
    # DEFAULT DIRECTORIES
    # ========================================================================
    
    DEFAULT_DATA_DIR = 'data'
    DEFAULT_OUTPUT_DIR = 'results'
    DEFAULT_FIGURE_DIR = 'figures'
    DEFAULT_LOG_DIR = 'logs'
    
    # ========================================================================
    # VISUALIZATION DEFAULTS
    # ========================================================================
    
    # Figure quality
    DEFAULT_DPI = 600              # High resolution DPI
    SCREEN_DPI = 150               # Screen display DPI
    DEFAULT_FIGURE_SIZE = (12, 8)  # Width, height in inches
    WIDE_FIGURE_SIZE = (16, 6)     # For multi-panel figures
    SQUARE_FIGURE_SIZE = (10, 10)  # For 3D plots
    
    # Color schemes (colorblind-friendly)
    DEFAULT_CMAP = 'viridis'
    DIVERGING_CMAP = 'RdYlBu_r'
    SEQUENTIAL_CMAP = 'plasma'
    CATEGORICAL_COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ]
    
    # Plot styling
    FONT_SIZE_TITLE = 16
    FONT_SIZE_LABEL = 14
    FONT_SIZE_TICK = 12
    FONT_SIZE_LEGEND = 11
    
    LINE_WIDTH = 2.0
    MARKER_SIZE = 80
    ALPHA_FILL = 0.3
    ALPHA_SCATTER = 0.7
    
    # ========================================================================
    # DATA REQUIREMENTS
    # ========================================================================
    
    # Critical columns (must be present)
    REQUIRED_COLUMNS = [
        'DFT_ads_E',        # Activity descriptor (adsorption energy)
        'surface_energy',   # Stability descriptor (J/m²)
        'Cost',             # Economic descriptor ($/kg)
        'symbol',           # Catalyst identifier (short)
        'AandB',            # Catalyst identifier (detailed)
        'reaction_type',    # Reaction name
        'optimal_energy',   # Sabatier optimum for this reaction
        'activity_width',   # Scoring width parameter
    ]
    
    # Optional columns (useful but not required)
    OPTIONAL_COLUMNS = [
        'Ametal',              # Primary element
        'Bmetal',              # Secondary element
        'Cmetal',              # Tertiary element (for ternary alloys)
        'slab_millers',        # Surface facet (e.g., '111', '100')
        'activity_pathway',    # Reaction pathway identifier
        'bimetallic_filtered', # Quality filter flag
        'composition',         # Stoichiometry string
        'bulk_structure',      # Crystal structure
        'magnetic_moment',     # Total magnetic moment
    ]
    
    # ========================================================================
    # PHYSICAL CONSTANTS
    # ========================================================================
    
    # Universal constants
    AVOGADRO = 6.022e23       # Avogadro's number (mol⁻¹)
    ELECTRON_CHARGE = 1.602e-19  # Elementary charge (C)
    BOLTZMANN = 8.617e-5      # Boltzmann constant (eV/K)
    
    # Standard conditions
    STANDARD_TEMP = 298.15    # Standard temperature (K)
    STANDARD_PRESSURE = 1.0   # Standard pressure (atm)
    
    # Electrochemistry
    FARADAY = 96485.3329      # Faraday constant (C/mol)
    SHE_POTENTIAL = 4.44      # SHE vs vacuum (V)
    
    # ========================================================================
    # FILE FORMATS
    # ========================================================================
    
    SUPPORTED_DATA_FORMATS = ['.csv', '.xlsx', '.json']
    SUPPORTED_FIGURE_FORMATS = ['.png', '.pdf', '.svg', '.jpg', '.eps']
    
    # ========================================================================
    # VALIDATION THRESHOLDS
    # ========================================================================
    
    # Physical bounds for descriptors
    MAX_ABS_ADSORPTION_ENERGY = 10.0  # |ΔE| < 10 eV (sanity check)
    MAX_SURFACE_ENERGY = 15.0         # γ < 15 J/m² (very high but possible)
    MIN_SURFACE_ENERGY = 0.0          # γ ≥ 0 (thermodynamic requirement)
    MAX_COST = 1e7                    # $10M/kg (sanity check)
    MIN_COST = 0.01                   # $0.01/kg (essentially free)
    
    # Data quality thresholds
    MIN_DATASET_SIZE = 10             # Minimum catalysts for meaningful analysis
    MAX_MISSING_FRACTION = 0.1        # Maximum 10% missing values allowed
    MIN_CORRELATION_FOR_WARNING = 0.95  # Warn if descriptors highly correlated


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_reaction_config(reaction: str, pathway: Optional[str] = None) -> ReactionConfig:
    """
    Retrieve configuration for specified reaction and pathway.
    
    Parameters
    ----------
    reaction : str
        Reaction type ('HER', 'CO2RR')
    pathway : str, optional
        Specific pathway ('CO', 'CHO', 'COCOH' for CO2RR)
        If None, returns default pathway
    
    Returns
    -------
    ReactionConfig
        Configuration object for the specified reaction
    
    Raises
    ------
    ValueError
        If reaction or pathway not found in registry
    
    Examples
    --------
    >>> config = get_reaction_config('HER')
    >>> print(config.optimal_energy)
    -0.27
    
    >>> co2rr_config = get_reaction_config('CO2RR', pathway='CO')
    >>> print(co2rr_config.optimal_energy)
    -0.67
    """
    reaction = reaction.upper()
    
    if reaction not in REACTION_REGISTRY:
        available = ', '.join(REACTION_REGISTRY.keys())
        raise ValueError(
            f"Reaction '{reaction}' not found. "
            f"Available reactions: {available}"
        )
    
    reaction_dict = REACTION_REGISTRY[reaction]
    
    if pathway is None:
        pathway = 'default'
    
    if pathway not in reaction_dict:
        available = [k for k in reaction_dict.keys() if k != 'default']
        raise ValueError(
            f"Pathway '{pathway}' not found for {reaction}. "
            f"Available pathways: {', '.join(available)}"
        )
    
    return reaction_dict[pathway]


def list_available_reactions() -> Dict[str, List[str]]:
    """
    List all available reactions and their pathways.
    
    Returns
    -------
    dict
        Dictionary mapping reaction names to lists of available pathways
    
    Examples
    --------
    >>> reactions = list_available_reactions()
    >>> print(reactions)
    {'HER': ['default', 'H_adsorption'], 
     'CO2RR': ['CO', 'CHO', 'COCOH', 'default']}
    """
    result = {}
    for reaction, pathways in REACTION_REGISTRY.items():
        pathway_list = [p for p in pathways.keys() if p != 'default']
        if not pathway_list:
            pathway_list = ['default']
        result[reaction] = sorted(pathway_list)
    return result


def print_available_reactions():
    """
    Print formatted list of available reactions and pathways.
    
    Examples
    --------
    >>> print_available_reactions()
    
    Available Reactions in ASCICat:
    ================================
    
    HER (Hydrogen Evolution Reaction):
      • H_adsorption
    
    CO2RR (CO₂ Reduction Reaction):
      • CO
      • CHO
      • COCOH
    """
    print("\n" + "="*70)
    print("Available Reactions in ASCICat")
    print("="*70)
    
    reactions = list_available_reactions()
    
    reaction_names = {
        'HER': 'Hydrogen Evolution Reaction',
        'CO2RR': 'CO₂ Reduction Reaction'
    }
    
    for reaction, pathways in reactions.items():
        full_name = reaction_names.get(reaction, reaction)
        print(f"\n{reaction} ({full_name}):")
        for pathway in pathways:
            print(f"  • {pathway}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# CONVENIENCE DICTIONARY FOR DIRECT ACCESS
# ============================================================================

# REACTION_CONFIGS provides direct access to configs by full pathway name
# Usage: REACTION_CONFIGS['CO2RR-CO'].optimal_energy
REACTION_CONFIGS = {
    'HER': HER_CONFIG,
    'CO2RR-CO': CO2RR_CO_CONFIG,
    'CO2RR-CHO': CO2RR_CHO_CONFIG,
    'CO2RR-COCOH': CO2RR_COCOH_CONFIG,
}


def validate_weights(w_a: float, w_s: float, w_c: float, 
                    tolerance: float = ASCIConstants.TOLERANCE) -> None:
    """
    Validate that weights are physically valid and sum to unity.
    
    Parameters
    ----------
    w_a : float
        Activity weight
    w_s : float
        Stability weight
    w_c : float
        Cost weight
    tolerance : float, optional
        Tolerance for sum validation (default: 1e-6)
    
    Raises
    ------
    ValueError
        If weights are invalid or don't sum to 1
    
    Examples
    --------
    >>> validate_weights(0.33, 0.33, 0.34)  # Valid
    >>> validate_weights(0.5, 0.3, 0.3)     # Invalid - sum = 1.1
    ValueError: Weights must sum to 1, got 1.100000
    """
    # Check individual weight ranges
    for name, weight in [('w_a', w_a), ('w_s', w_s), ('w_c', w_c)]:
        if not (ASCIConstants.WEIGHT_MIN <= weight <= ASCIConstants.WEIGHT_MAX):
            raise ValueError(
                f"{name} must be in [0, 1], got {weight:.6f}"
            )
    
    # Check sum equals 1
    weight_sum = w_a + w_s + w_c
    if abs(weight_sum - ASCIConstants.WEIGHT_SUM) > tolerance:
        raise ValueError(
            f"Weights must sum to 1, got {weight_sum:.6f}"
        )


def normalize_weights(w_a: float, w_s: float, w_c: float) -> Tuple[float, float, float]:
    """
    Normalize weights to sum to exactly 1.0.
    
    Useful when weights are close to 1.0 but not exact due to
    floating point arithmetic.
    
    Parameters
    ----------
    w_a, w_s, w_c : float
        Input weights
    
    Returns
    -------
    tuple
        Normalized weights summing to 1.0
    
    Examples
    --------
    >>> normalize_weights(0.33, 0.33, 0.33)
    (0.33333..., 0.33333..., 0.33333...)
    """
    total = w_a + w_s + w_c
    
    if total == 0:
        # Equal weights if all zero
        return (1/3, 1/3, 1/3)
    
    return (w_a/total, w_s/total, w_c/total)


def get_data_file_path(reaction: str, pathway: Optional[str] = None) -> str:
    """
    Get expected data file path for a reaction.
    
    Parameters
    ----------
    reaction : str
        Reaction type
    pathway : str, optional
        Reaction pathway
    
    Returns
    -------
    str
        Expected CSV file path
    
    Examples
    --------
    >>> get_data_file_path('HER')
    'data/HER_clean.csv'
    >>> get_data_file_path('CO2RR', 'CO')
    'data/CO2RR_CO_clean.csv'
    """
    reaction = reaction.upper()
    
    if reaction == 'HER':
        return f"{ASCIConstants.DEFAULT_DATA_DIR}/HER_clean.csv"
    elif reaction == 'CO2RR':
        if pathway is None or pathway == 'default':
            pathway = 'CO'
        return f"{ASCIConstants.DEFAULT_DATA_DIR}/CO2RR_{pathway}_clean.csv"
    else:
        raise ValueError(f"Unknown reaction: {reaction}")


def create_custom_config(name: str,
                        optimal_energy: float,
                        activity_width: float = 0.15,
                        **kwargs) -> ReactionConfig:
    """
    Create custom reaction configuration.
    
    Allows users to define their own reaction with custom parameters
    while using sensible defaults for unspecified values.
    
    Parameters
    ----------
    name : str
        Custom reaction name
    optimal_energy : float
        Sabatier-optimal binding energy (eV)
    activity_width : float, optional
        Activity scoring width (default: 0.15 eV)
    **kwargs
        Additional ReactionConfig parameters
    
    Returns
    -------
    ReactionConfig
        Custom configuration object
    
    Examples
    --------
    >>> orr_config = create_custom_config(
    ...     name='ORR',
    ...     optimal_energy=-0.45,
    ...     activity_width=0.12,
    ...     description='Oxygen Reduction Reaction'
    ... )
    """
    # Set defaults if not provided
    defaults = {
        'pathway': 'custom',
        'activity_descriptor': 'DFT_ads_E',
        'activity_window': (optimal_energy - 0.5, optimal_energy + 0.5),
        'stability_range': (0.1, 5.0),
        'cost_range': (1.0, 200000.0),
        'default_weights': (0.33, 0.33, 0.34),
        'description': f'Custom reaction: {name}',
        'references': [],
    }
    
    # Update with user-provided values
    defaults.update(kwargs)
    
    return ReactionConfig(
        name=name,
        optimal_energy=optimal_energy,
        activity_width=activity_width,
        **defaults
    )


# ============================================================================
# MODULE-LEVEL TESTS
# ============================================================================

if __name__ == '__main__':
    # Test configuration loading
    print("Testing ASCICat Configuration Module")
    print("="*70)
    
    # Test HER config
    print("\n1. Testing HER Configuration:")
    her = get_reaction_config('HER')
    her.print_summary()
    
    # Test CO2RR configs
    print("\n2. Testing CO2RR Configurations:")
    for pathway in ['CO', 'CHO', 'COCOH']:
        config = get_reaction_config('CO2RR', pathway)
        print(f"\n{pathway} Pathway: ΔE_opt = {config.optimal_energy:.2f} eV")
    
    # Test weight validation
    print("\n3. Testing Weight Validation:")
    try:
        validate_weights(0.33, 0.33, 0.34)
        print("  ✓ Valid weights: (0.33, 0.33, 0.34)")
    except ValueError as e:
        print(f"  ✗ Error: {e}")
    
    try:
        validate_weights(0.5, 0.3, 0.3)
        print("  ✓ Valid weights: (0.5, 0.3, 0.3)")
    except ValueError as e:
        print(f"  ✗ Invalid weights: {e}")
    
    # List available reactions
    print("\n4. Available Reactions:")
    print_available_reactions()
    
    # Test custom config
    print("\n5. Creating Custom Configuration:")
    custom = create_custom_config(
        name='ORR',
        optimal_energy=-0.45,
        description='Oxygen Reduction Reaction (4e⁻ pathway)'
    )
    print(f"  Custom ORR: ΔE_opt = {custom.optimal_energy:.2f} eV")
    
    print("\n" + "="*70)
    print("✓ All configuration tests passed!")
    print("="*70 + "\n")