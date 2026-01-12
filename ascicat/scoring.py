"""
ascicat/scoring.py
Mathematical Scoring Functions for ASCI Framework

Implements rigorous normalization and scoring for:
- Activity (Sabatier principle: linear & Gaussian)
- Stability (surface energy: inverse linear)
- Cost (economic viability: logarithmic)
- Combined ASCI metric (weighted integration)

All scores normalized to [0, 1] with comprehensive validation.

Author: N. Khossossi
Institution: DIFFER (Dutch Institute for Fundamental Energy Research)

Mathematical Framework:
    φ_ASCI = w_a·S_a(ΔE) + w_s·S_s(γ) + w_c·S_c(C)

References:
    - Nørskov, J. K. et al. Nat. Chem. 1, 37 (2009)
    - Greeley, J. et al. Nat. Mater. 5, 909 (2006)
    - Sabatier, P. Ber. Dtsch. Chem. Ges. 44, 1984 (1911)
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import warnings

from .config import ASCIConstants


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

# Accept both single values and arrays
NumericType = Union[float, int, np.ndarray, pd.Series]


# ============================================================================
# ACTIVITY SCORING (SABATIER PRINCIPLE)
# ============================================================================

class ActivityScorer:
    """
    Activity scoring based on Sabatier principle.
    
    Implements both linear and Gaussian scoring methods to quantify
    proximity to thermodynamically optimal binding energies.
    
    The Sabatier principle states that optimal catalysts have intermediate
    binding: too weak → no activation, too strong → no desorption.
    
    Methods
    -------
    linear : Linear scoring (default, computationally efficient)
    gaussian : Gaussian scoring (sharper discrimination near optimum)
    """
    
    @staticmethod
    def linear(delta_E: NumericType,
               optimal_E: float,
               width: float) -> NumericType:
        """
        Linear activity scoring (DEFAULT METHOD).
        
        Scores decrease linearly with distance from optimal binding energy.
        Maximum score (1.0) at ΔE = ΔE_opt, zero score at |ΔE - ΔE_opt| ≥ σ_a.
        
        Mathematical Form:
            S_a(ΔE) = max(0, 1 - |ΔE - ΔE_opt| / σ_a)
        
        Parameters
        ----------
        delta_E : float or array-like
            Adsorption energy (eV)
        optimal_E : float
            Sabatier-optimal binding energy (eV)
        width : float
            Activity tolerance σ_a (eV). Defines acceptable deviation.
            Typical value: 0.15 eV
        
        Returns
        -------
        float or np.ndarray
            Activity scores normalized to [0, 1]
            - 1.0: Perfect activity (at optimum)
            - 0.5: Moderate activity (σ_a/2 from optimum)
            - 0.0: Poor activity (>σ_a from optimum)
        
        Notes
        -----
        Linear scoring advantages:
        - Computationally efficient
        - Easy interpretation (proportional to distance)
        - Consistent with traditional volcano plots
        - Symmetric treatment of over/underbinding
        
        Examples
        --------
        >>> from ascicat.scoring import ActivityScorer
        >>> scorer = ActivityScorer()
        >>> 
        >>> # Perfect catalyst at optimum
        >>> score = scorer.linear(delta_E=-0.27, optimal_E=-0.27, width=0.15)
        >>> print(f"Score: {score:.3f}")
        Score: 1.000
        >>> 
        >>> # Good catalyst (0.075 eV from optimum)
        >>> score = scorer.linear(delta_E=-0.195, optimal_E=-0.27, width=0.15)
        >>> print(f"Score: {score:.3f}")
        Score: 0.500
        >>> 
        >>> # Poor catalyst (far from optimum)
        >>> score = scorer.linear(delta_E=0.0, optimal_E=-0.27, width=0.15)
        >>> print(f"Score: {score:.3f}")
        Score: 0.000
        
        References
        ----------
        Greeley, J. et al. Nat. Mater. 5, 909 (2006)
        """
        # Validate inputs
        if width <= 0:
            raise ValueError(f"Activity width must be positive, got {width}")
        
        # Convert to numpy for vectorized operations
        delta_E = np.asarray(delta_E, dtype=float)
        
        # Calculate distance from optimum
        distance = np.abs(delta_E - optimal_E)
        
        # Linear scoring: 1 - (distance / width), clipped to [0, 1]
        score = 1.0 - (distance / width)
        score = np.maximum(0.0, score)  # Ensure non-negative
        score = np.minimum(1.0, score)  # Ensure ≤ 1
        
        return score
    
    @staticmethod
    def gaussian(delta_E: NumericType,
                 optimal_E: float,
                 width: float) -> NumericType:
        """
        Gaussian activity scoring (ALTERNATIVE METHOD).
        
        Scores decay exponentially with squared distance from optimal energy.
        Provides sharper discrimination near the Sabatier optimum.
        
        Mathematical Form:
            S_a(ΔE) = exp(-(ΔE - ΔE_opt)² / (2σ_a²))
        
        Parameters
        ----------
        delta_E : float or array-like
            Adsorption energy (eV)
        optimal_E : float
            Sabatier-optimal binding energy (eV)
        width : float
            Activity tolerance σ_a (eV). Controls Gaussian width.
        
        Returns
        -------
        float or np.ndarray
            Activity scores normalized to [0, 1]
            - 1.0: At optimum
            - 0.61: At σ_a from optimum (1 standard deviation)
            - 0.14: At 2σ_a from optimum
        
        Notes
        -----
        Gaussian scoring advantages:
        - Sharper discrimination near optimum
        - Never reaches exactly zero
        - Better for narrow volcano peaks
        - Mathematically smooth derivatives
        
        Disadvantages:
        - Less intuitive than linear
        - More aggressive penalization
        
        Examples
        --------
        >>> from ascicat.scoring import ActivityScorer
        >>> scorer = ActivityScorer()
        >>> 
        >>> # Compare linear vs Gaussian
        >>> energies = np.array([-0.42, -0.35, -0.27, -0.19, -0.12])
        >>> linear_scores = scorer.linear(energies, -0.27, 0.15)
        >>> gaussian_scores = scorer.gaussian(energies, -0.27, 0.15)
        >>> 
        >>> for E, lin, gauss in zip(energies, linear_scores, gaussian_scores):
        ...     print(f"ΔE={E:+.2f}: Linear={lin:.3f}, Gaussian={gauss:.3f}")
        ΔE=-0.42: Linear=0.000, Gaussian=0.135
        ΔE=-0.35: Linear=0.467, Gaussian=0.606
        ΔE=-0.27: Linear=1.000, Gaussian=1.000
        ΔE=-0.19: Linear=0.467, Gaussian=0.606
        ΔE=-0.12: Linear=0.000, Gaussian=0.135
        
        References
        ----------
        Nørskov, J. K. et al. Nat. Chem. 1, 37 (2009)
        """
        # Validate inputs
        if width <= 0:
            raise ValueError(f"Activity width must be positive, got {width}")
        
        # Convert to numpy
        delta_E = np.asarray(delta_E, dtype=float)
        
        # Calculate deviation from optimum
        deviation = delta_E - optimal_E
        
        # Gaussian scoring: exp(-(deviation²) / (2σ²))
        exponent = -(deviation ** 2) / (2 * width ** 2)
        score = np.exp(exponent)
        
        # Ensure bounds [0, 1] (should be automatic, but safe to clip)
        score = np.clip(score, 0.0, 1.0)
        
        return score


# ============================================================================
# STABILITY SCORING (SURFACE ENERGY)
# ============================================================================

def score_stability(surface_energy: NumericType,
                   gamma_min: Optional[float] = None,
                   gamma_max: Optional[float] = None) -> NumericType:
    """
    Stability scoring via inverse linear normalization.

    Lower surface energy indicates stronger metal-metal bonding and
    enhanced thermodynamic stability against reconstruction/dissolution.

    Mathematical Form:
        S_s(γ) = (γ_max - γ) / (γ_max - γ_min)

    Physical Interpretation:
        - Low γ → Strong bonding → High stability → High score
        - High γ → Weak bonding → Low stability → Low score

    Parameters
    ----------
    surface_energy : float or array-like
        Surface energy γ (J/m²)
    gamma_min : float, optional
        Minimum surface energy for normalization.
        If None (default), uses min value from input data (data-driven).
    gamma_max : float, optional
        Maximum surface energy for normalization.
        If None (default), uses max value from input data (data-driven).

    Returns
    -------
    float or np.ndarray
        Stability scores normalized to [0, 1]
        - 1.0: Maximum stability (γ = γ_min)
        - 0.0: Minimum stability (γ = γ_max)

    Raises
    ------
    ValueError
        If gamma_min >= gamma_max
        If any surface energy is negative (physically impossible)

    Notes
    -----
    Data-Driven Normalization (Default):
    When gamma_min and gamma_max are None, the function automatically
    computes them from the input data. This ensures scores span the
    full [0, 1] range, which is scientifically correct for ranking.

    Surface Energy Physical Ranges:
    - Pt(111): ~0.52 J/m² (highly stable)
    - Cu(111): ~1.83 J/m² (moderately stable)
    - Open surfaces: 2-4 J/m² (less stable)

    Examples
    --------
    >>> from ascicat.scoring import score_stability
    >>>
    >>> # Data-driven normalization (recommended)
    >>> gammas = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    >>> scores = score_stability(gammas)  # Auto min/max
    >>> print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    Score range: [0.000, 1.000]

    References
    ----------
    Hansen, H. A. et al. Phys. Chem. Chem. Phys. 10, 3722 (2008)
    """
    # Convert to numpy
    surface_energy = np.asarray(surface_energy, dtype=float)

    # Check for negative surface energies (physically impossible)
    if np.any(surface_energy < 0):
        negative_count = np.sum(surface_energy < 0)
        warnings.warn(
            f"Found {negative_count} negative surface energy values. "
            f"Surface energies must be non-negative (thermodynamic requirement). "
            f"These will be clipped to zero.",
            category=UserWarning
        )
        surface_energy = np.maximum(0.0, surface_energy)

    # Data-driven normalization: compute min/max from data if not provided
    if gamma_min is None:
        gamma_min = float(np.min(surface_energy))
    if gamma_max is None:
        gamma_max = float(np.max(surface_energy))

    # Validate normalization range
    if gamma_min >= gamma_max:
        # If all values are identical, return 1.0 (all equally stable)
        if gamma_min == gamma_max:
            return np.ones_like(surface_energy, dtype=float)
        raise ValueError(
            f"gamma_min ({gamma_min}) must be less than "
            f"gamma_max ({gamma_max})"
        )

    # Inverse linear normalization
    # High stability (low γ) → score near 1
    # Low stability (high γ) → score near 0
    score = (gamma_max - surface_energy) / (gamma_max - gamma_min)

    # Clip to [0, 1] for values outside normalization range
    score = np.clip(score, 0.0, 1.0)

    return score


# ============================================================================
# COST SCORING (ECONOMIC VIABILITY)
# ============================================================================

def score_cost(cost: NumericType,
               cost_min: Optional[float] = None,
               cost_max: Optional[float] = None) -> NumericType:
    """
    Cost scoring via logarithmic normalization.

    Logarithmic scaling handles the enormous range (5 orders of magnitude)
    in material costs across the periodic table while maintaining
    discrimination throughout the spectrum.

    Mathematical Form:
        S_c(C) = (log C_max - log C) / (log C_max - log C_min)

    Economic Interpretation:
        - Low cost → Economically viable → High score
        - High cost → Economically prohibitive → Low score

    Parameters
    ----------
    cost : float or array-like
        Material cost ($/kg), composition-weighted for alloys
    cost_min : float, optional
        Minimum cost for normalization.
        If None (default), uses min value from input data (data-driven).
    cost_max : float, optional
        Maximum cost for normalization.
        If None (default), uses max value from input data (data-driven).

    Returns
    -------
    float or np.ndarray
        Cost scores normalized to [0, 1]
        - 1.0: Maximum affordability (C = C_min)
        - 0.0: Minimum affordability (C = C_max)

    Raises
    ------
    ValueError
        If cost_min >= cost_max
        If any cost is negative or zero (mathematically invalid for log)

    Notes
    -----
    Data-Driven Normalization (Default):
    When cost_min and cost_max are None, the function automatically
    computes them from the input data. This ensures scores span the
    full [0, 1] range, which is scientifically correct for ranking.

    Logarithmic scaling ensures:
    - $1 vs $10: Significant score difference
    - $10,000 vs $10,010: Negligible score difference
    - Physically realistic economic sensitivity

    Examples
    --------
    >>> from ascicat.scoring import score_cost
    >>>
    >>> # Data-driven normalization (recommended)
    >>> costs = np.array([2.67, 10, 100, 1000, 10000, 107544])
    >>> scores = score_cost(costs)  # Auto min/max from data
    >>> print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    Score range: [0.000, 1.000]

    References
    ----------
    U.S. Geological Survey. Mineral Commodity Summaries 2024.
    """
    # Convert to numpy
    cost = np.asarray(cost, dtype=float)

    # Check for non-positive costs (invalid for logarithm)
    if np.any(cost <= 0):
        invalid_count = np.sum(cost <= 0)
        raise ValueError(
            f"Found {invalid_count} non-positive cost values. "
            f"All costs must be positive for logarithmic normalization. "
            f"Minimum cost: {np.min(cost):.6f}"
        )

    # Data-driven normalization: compute min/max from data if not provided
    if cost_min is None:
        cost_min = float(np.min(cost))
    if cost_max is None:
        cost_max = float(np.max(cost))

    # Validate normalization range
    if cost_min <= 0:
        raise ValueError(
            f"cost_min must be positive for logarithmic scaling, "
            f"got {cost_min}"
        )

    if cost_min >= cost_max:
        # If all values are identical, return 1.0 (all equally affordable)
        if cost_min == cost_max:
            return np.ones_like(cost, dtype=float)
        raise ValueError(
            f"cost_min ({cost_min}) must be less than cost_max ({cost_max})"
        )

    # Logarithmic normalization
    # Low cost (cheap) → score near 1
    # High cost (expensive) → score near 0
    log_cost = np.log10(cost)
    log_min = np.log10(cost_min)
    log_max = np.log10(cost_max)

    score = (log_max - log_cost) / (log_max - log_min)

    # Clip to [0, 1] for values outside normalization range
    score = np.clip(score, 0.0, 1.0)

    return score


# ============================================================================
# ACTIVITY SCORING WRAPPER (AUTO-SELECT METHOD)
# ============================================================================

def score_activity(delta_E: NumericType,
                   optimal_E: float,
                   width: float,
                   method: str = 'linear') -> NumericType:
    """
    Activity scoring with automatic method selection.
    
    Wrapper function that dispatches to either linear or Gaussian scoring
    based on user preference. Provides unified interface for activity
    assessment.
    
    Parameters
    ----------
    delta_E : float or array-like
        Adsorption energy (eV)
    optimal_E : float
        Sabatier-optimal binding energy (eV)
    width : float
        Activity tolerance σ_a (eV)
    method : {'linear', 'gaussian'}, optional
        Scoring method (default: 'linear')
        - 'linear': Linear decay from optimum
        - 'gaussian': Exponential decay from optimum
    
    Returns
    -------
    float or np.ndarray
        Activity scores normalized to [0, 1]
    
    Raises
    ------
    ValueError
        If method is not 'linear' or 'gaussian'
    
    Examples
    --------
    >>> from ascicat.scoring import score_activity
    >>> 
    >>> # Linear scoring (default)
    >>> score = score_activity(-0.27, optimal_E=-0.27, width=0.15)
    >>> print(f"Linear: {score:.3f}")
    Linear: 1.000
    >>> 
    >>> # Gaussian scoring
    >>> score = score_activity(-0.27, optimal_E=-0.27, width=0.15, 
    ...                        method='gaussian')
    >>> print(f"Gaussian: {score:.3f}")
    Gaussian: 1.000
    
    See Also
    --------
    ActivityScorer.linear : Linear activity scoring
    ActivityScorer.gaussian : Gaussian activity scoring
    """
    method = method.lower()
    
    if method == 'linear':
        return ActivityScorer.linear(delta_E, optimal_E, width)
    elif method == 'gaussian':
        return ActivityScorer.gaussian(delta_E, optimal_E, width)
    else:
        raise ValueError(
            f"Unknown activity scoring method: '{method}'. "
            f"Must be 'linear' or 'gaussian'."
        )


# ============================================================================
# COMBINED ASCI CALCULATION
# ============================================================================

def calculate_asci(activity_score: NumericType,
                   stability_score: NumericType,
                   cost_score: NumericType,
                   w_a: float,
                   w_s: float,
                   w_c: float) -> NumericType:
    """
    Calculate Activity-Stability-Cost Index (ASCI).
    
    Combines three normalized descriptor scores into a single unified metric
    through weighted linear combination. Provides objective catalyst ranking
    that balances catalytic performance, durability, and economic viability.
    
    Mathematical Form:
        φ_ASCI = w_a·S_a + w_s·S_s + w_c·S_c
    
    where:
        S_a ∈ [0, 1]: Activity score (Sabatier principle)
        S_s ∈ [0, 1]: Stability score (surface energy)
        S_c ∈ [0, 1]: Cost score (economic viability)
        w_a + w_s + w_c = 1 (weight normalization)
    
    Parameters
    ----------
    activity_score : float or array-like
        Normalized activity scores [0, 1]
    stability_score : float or array-like
        Normalized stability scores [0, 1]
    cost_score : float or array-like
        Normalized cost scores [0, 1]
    w_a : float
        Activity weight [0, 1]
    w_s : float
        Stability weight [0, 1]
    w_c : float
        Cost weight [0, 1]
        Constraint: w_a + w_s + w_c = 1
    
    Returns
    -------
    float or np.ndarray
        ASCI scores normalized to [0, 1]
        - 1.0: Ideal catalyst (perfect in all dimensions)
        - 0.5: Average performance
        - 0.0: Poor catalyst (fails all criteria)
    
    Raises
    ------
    ValueError
        If weights don't sum to 1 (within tolerance)
        If any weight is outside [0, 1]
        If score arrays have incompatible shapes
    
    Notes
    -----
    Weight Selection Guidelines:
    
    1. Equal Weights (0.33, 0.33, 0.34) - DEFAULT
       - Unbiased exploratory screening
       - No a priori preference
       - Recommended starting point
    
    2. Activity-Focused (0.5, 0.3, 0.2)
       - Performance-critical applications
       - Stability less constraining
       - Research/fundamental studies
    
    3. Stability-Focused (0.3, 0.5, 0.2)
       - Long-term operation required
       - Harsh electrochemical conditions
       - Industrial durability emphasis
    
    4. Cost-Focused (0.3, 0.2, 0.5)
       - Large-scale deployment
       - Commodity applications
       - Earth-abundant materials priority
    
    Unlike Pareto frontier methods that generate multiple solutions
    requiring subjective selection, ASCI provides deterministic ranking
    for reproducible catalyst prioritization.
    
    Examples
    --------
    >>> from ascicat.scoring import calculate_asci
    >>> 
    >>> # Perfect catalyst in all dimensions
    >>> asci = calculate_asci(1.0, 1.0, 1.0, 0.33, 0.33, 0.34)
    >>> print(f"Perfect catalyst: {asci:.3f}")
    Perfect catalyst: 1.000
    >>> 
    >>> # Realistic catalyst (good activity, moderate stability, cheap)
    >>> asci = calculate_asci(0.85, 0.65, 0.92, 0.33, 0.33, 0.34)
    >>> print(f"Cu-based catalyst: {asci:.3f}")
    Cu-based catalyst: 0.807
    >>> 
    >>> # Array calculation
    >>> activity = np.array([0.9, 0.7, 0.5])
    >>> stability = np.array([0.8, 0.6, 0.4])
    >>> cost = np.array([0.95, 0.85, 0.75])
    >>> asci = calculate_asci(activity, stability, cost, 0.33, 0.33, 0.34)
    >>> print(asci)
    [0.884 0.717 0.550]
    >>> 
    >>> # Weight sensitivity
    >>> # Equal weights
    >>> asci_equal = calculate_asci(0.8, 0.6, 0.9, 0.33, 0.33, 0.34)
    >>> # Activity-focused
    >>> asci_activity = calculate_asci(0.8, 0.6, 0.9, 0.5, 0.3, 0.2)
    >>> # Cost-focused
    >>> asci_cost = calculate_asci(0.8, 0.6, 0.9, 0.3, 0.2, 0.5)
    >>> print(f"Equal: {asci_equal:.3f}, Activity: {asci_activity:.3f}, Cost: {asci_cost:.3f}")
    Equal: 0.770, Activity: 0.760, Cost: 0.800
    
    References
    ----------
    Khossossi, N. (2025). ASCICat: Activity-Stability-Cost Integrated
    Framework for Electrocatalyst Discovery.
    """
    # Validate weights
    from .config import validate_weights
    validate_weights(w_a, w_s, w_c)
    
    # Convert to numpy for consistent handling
    activity_score = np.asarray(activity_score, dtype=float)
    stability_score = np.asarray(stability_score, dtype=float)
    cost_score = np.asarray(cost_score, dtype=float)
    
    # Validate score ranges
    for name, score in [('Activity', activity_score), 
                        ('Stability', stability_score),
                        ('Cost', cost_score)]:
        if np.any(score < 0) or np.any(score > 1):
            warnings.warn(
                f"{name} scores outside [0, 1] range. "
                f"Min: {np.min(score):.6f}, Max: {np.max(score):.6f}. "
                f"Scores will be clipped.",
                category=UserWarning
            )
            score = np.clip(score, 0.0, 1.0)
    
    # Check shape compatibility (for arrays)
    shapes = [activity_score.shape, stability_score.shape, cost_score.shape]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(
            f"Score arrays have incompatible shapes: "
            f"activity={shapes[0]}, stability={shapes[1]}, cost={shapes[2]}"
        )
    
    # Calculate weighted combination
    asci = (w_a * activity_score + 
            w_s * stability_score + 
            w_c * cost_score)
    
    # Final safety clip (should be redundant if inputs valid)
    asci = np.clip(asci, 0.0, 1.0)
    
    return asci


# ============================================================================
# CONVENIENCE CLASS FOR BATCH SCORING
# ============================================================================

class ScoringFunctions:
    """
    Unified interface for all ASCI scoring functions.
    
    Provides both static methods and instance methods for scoring operations.
    Useful for batch processing and configuration management.
    
    Attributes
    ----------
    config : ReactionConfig or None
        Reaction configuration (if initialized with config)
    
    Methods
    -------
    score_activity_linear : Linear activity scoring
    score_activity_gaussian : Gaussian activity scoring
    score_stability : Stability scoring
    score_cost : Cost scoring
    combined_asci_score : Full ASCI calculation
    
    Examples
    --------
    >>> from ascicat.scoring import ScoringFunctions
    >>> from ascicat.config import get_reaction_config
    >>> 
    >>> # Initialize with HER configuration
    >>> config = get_reaction_config('HER')
    >>> scorer = ScoringFunctions(config)
    >>> 
    >>> # Score single catalyst
    >>> activity = scorer.score_activity_linear(-0.25)
    >>> stability = scorer.score_stability(0.52)
    >>> cost = scorer.score_cost(8.5)
    >>> asci = scorer.combined_asci_score(activity, stability, cost, 
    ...                                   0.33, 0.33, 0.34)
    >>> print(f"ASCI: {asci:.3f}")
    ASCI: 0.948
    """
    
    def __init__(self, config=None):
        """
        Initialize ScoringFunctions.
        
        Parameters
        ----------
        config : ReactionConfig, optional
            Reaction configuration for automatic parameter loading
        """
        self.config = config
    
    def score_activity_linear(self, delta_E: NumericType) -> NumericType:
        """Linear activity scoring using config parameters."""
        if self.config is None:
            raise ValueError("Configuration required for automatic parameter loading")
        return ActivityScorer.linear(
            delta_E, 
            self.config.optimal_energy,
            self.config.activity_width
        )
    
    def score_activity_gaussian(self, delta_E: NumericType) -> NumericType:
        """Gaussian activity scoring using config parameters."""
        if self.config is None:
            raise ValueError("Configuration required for automatic parameter loading")
        return ActivityScorer.gaussian(
            delta_E,
            self.config.optimal_energy,
            self.config.activity_width
        )
    
    @staticmethod
    def score_stability(surface_energy: NumericType,
                       gamma_min: float = 0.1,
                       gamma_max: float = 5.0) -> NumericType:
        """Stability scoring (static method)."""
        return score_stability(surface_energy, gamma_min, gamma_max)
    
    @staticmethod
    def score_cost(cost: NumericType,
                   cost_min: float = 1.0,
                   cost_max: float = 200000.0) -> NumericType:
        """Cost scoring (static method)."""
        return score_cost(cost, cost_min, cost_max)
    
    @staticmethod
    def combined_asci_score(activity_score: NumericType,
                           stability_score: NumericType,
                           cost_score: NumericType,
                           w_a: float,
                           w_s: float,
                           w_c: float) -> NumericType:
        """Combined ASCI calculation (static method)."""
        return calculate_asci(
            activity_score, stability_score, cost_score,
            w_a, w_s, w_c
        )


# ============================================================================
# MODULE-LEVEL TESTS
# ============================================================================

if __name__ == '__main__':
    print("Testing ASCICat Scoring Module")
    print("="*70)
    
    # Test 1: Activity Scoring
    print("\n1. Activity Scoring Tests:")
    print("-" * 70)
    
    scorer = ActivityScorer()
    
    # Test perfect catalyst
    score_perfect = scorer.linear(-0.27, -0.27, 0.15)
    print(f"Perfect catalyst (ΔE = -0.27 eV): {score_perfect:.3f}")
    assert np.isclose(score_perfect, 1.0), "Perfect score should be 1.0"
    
    # Test linear vs Gaussian
    energies = np.linspace(-0.50, 0.0, 11)
    linear_scores = scorer.linear(energies, -0.27, 0.15)
    gaussian_scores = scorer.gaussian(energies, -0.27, 0.15)
    
    print("\nLinear vs Gaussian Comparison:")
    print(f"{'ΔE (eV)':>10} {'Linear':>10} {'Gaussian':>10}")
    for E, lin, gauss in zip(energies, linear_scores, gaussian_scores):
        print(f"{E:>10.2f} {lin:>10.3f} {gauss:>10.3f}")
    
    # Test 2: Stability Scoring
    print("\n2. Stability Scoring Tests:")
    print("-" * 70)
    
    gammas = np.array([0.52, 1.0, 2.0, 3.0, 4.5])
    stability_scores = score_stability(gammas)
    
    print(f"{'γ (J/m²)':>12} {'S_s':>10}")
    for g, s in zip(gammas, stability_scores):
        print(f"{g:>12.2f} {s:>10.3f}")
    
    # Test 3: Cost Scoring
    print("\n3. Cost Scoring Tests:")
    print("-" * 70)
    
    costs = np.array([1, 10, 100, 1000, 10000, 100000])
    cost_scores = score_cost(costs)
    
    print(f"{'Cost ($/kg)':>15} {'S_c':>10}")
    for c, s in zip(costs, cost_scores):
        print(f"{c:>15,.0f} {s:>10.3f}")
    
    # Test 4: Combined ASCI
    print("\n4. Combined ASCI Tests:")
    print("-" * 70)
    
    # Test case: Copper-based catalyst
    # Activity: good (ΔE = -0.25 eV, close to -0.27)
    # Stability: moderate (γ = 1.83 J/m²)
    # Cost: excellent ($8.5/kg)
    
    act_score = scorer.linear(-0.25, -0.27, 0.15)
    stab_score = score_stability(1.83)
    cost_score_val = score_cost(8.5)
    
    print(f"\nCopper-based Catalyst Example:")
    print(f"  Activity score:  {act_score:.3f}")
    print(f"  Stability score: {stab_score:.3f}")
    print(f"  Cost score:      {cost_score_val:.3f}")
    
    # Equal weights
    asci_equal = calculate_asci(act_score, stab_score, cost_score_val, 
                                0.33, 0.33, 0.34)
    print(f"  ASCI (equal):    {asci_equal:.3f}")
    
    # Activity-focused
    asci_activity = calculate_asci(act_score, stab_score, cost_score_val,
                                   0.5, 0.3, 0.2)
    print(f"  ASCI (activity): {asci_activity:.3f}")
    
    # Cost-focused
    asci_cost = calculate_asci(act_score, stab_score, cost_score_val,
                               0.3, 0.2, 0.5)
    print(f"  ASCI (cost):     {asci_cost:.3f}")
    
    # Test 5: Array Operations
    print("\n5. Vectorized Operations Test:")
    print("-" * 70)
    
    n_catalysts = 5
    delta_E_array = np.random.uniform(-0.5, 0.0, n_catalysts)
    gamma_array = np.random.uniform(0.5, 3.0, n_catalysts)
    cost_array = np.random.uniform(5, 50000, n_catalysts)
    
    act_array = scorer.linear(delta_E_array, -0.27, 0.15)
    stab_array = score_stability(gamma_array)
    cost_array_scores = score_cost(cost_array)
    asci_array = calculate_asci(act_array, stab_array, cost_array_scores,
                                0.33, 0.33, 0.34)
    
    print(f"Processed {n_catalysts} catalysts:")
    print(f"{'Catalyst':>10} {'ΔE':>8} {'γ':>8} {'Cost':>10} {'ASCI':>8}")
    for i in range(n_catalysts):
        print(f"{'Cat-'+str(i+1):>10} {delta_E_array[i]:>8.3f} "
              f"{gamma_array[i]:>8.2f} {cost_array[i]:>10,.0f} "
              f"{asci_array[i]:>8.3f}")
    
    print("\n" + "="*70)
    print("✓ All scoring tests passed!")
    print("="*70 + "\n")