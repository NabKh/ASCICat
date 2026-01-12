"""
ASCICat: Activity-Stability-Cost Integrated Catalyst Discovery
================================================================

A comprehensive Python framework for multi-objective electrocatalyst screening
that extends Nørskov's single-descriptor volcano plot approach to simultaneous
optimization of catalytic activity, electrochemical stability, and economic viability.

Main Components:
---------------
- ASCICalculator: Main calculation engine for ASCI scores
- Visualizer: Figure generation
- Analyzer: Advanced analysis tools (Pareto fronts, sensitivity)
- ReactionConfig: Reaction configuration management
- ScoringFunctions: Mathematical scoring implementations

Quick Start:
-----------
>>> from ascicat import ASCICalculator
>>> calc = ASCICalculator(reaction='HER')
>>> calc.load_data('data/HER_clean.csv')
>>> results = calc.calculate_asci(w_a=0.4, w_s=0.3, w_c=0.3)
>>> top10 = calc.get_top_catalysts(n=10)

Scientific Background:
--------------------
φ_ASCI = w_a·S_a(ΔE) + w_s·S_s(γ) + w_c·S_c(C)

where:
- S_a: Activity score (Sabatier principle)
- S_s: Stability score (surface energy)
- S_c: Cost score (economic viability)
- w_a, w_s, w_c: Customizable weights (sum to 1)

References:
----------
- Nørskov, J. K. et al. Nat. Chem. 1, 37 (2009)
- Greeley, J. et al. Nat. Mater. 5, 909 (2006)

Author: N. Khossossi
Institution: Dutch Institute for Fundamental Energy Research (DIFFER)
License: MIT
"""

# Version information
from .version import (
    __version__,
    __version_info__,
    __author__,
    __email__,
    __institution__,
    __url__,
    __license__,
    __description__,
    __citation__,
    print_version_info,
    get_version_dict,
)

# Core calculator
from .calculator import ASCICalculator, quick_asci

# Configuration
from .config import (
    ReactionConfig,
    ASCIConstants,
    get_reaction_config,
    list_available_reactions,
    validate_weights,
    get_data_file_path,
    # Predefined configurations
    HER_CONFIG,
    CO2RR_CO_CONFIG,
    CO2RR_CHO_CONFIG,
    CO2RR_COCOH_CONFIG,
    # Convenience dictionary
    REACTION_CONFIGS,
)

# Scoring functions
from .scoring import (
    ScoringFunctions,
    score_activity,
    score_stability,
    score_cost,
    calculate_asci,
)

# Data loading
from .data_loader import DataLoader  # REMOVED validate_data_format - it doesn't exist

# Visualization
from .visualizer import Visualizer

# Analysis tools
from .analyzer import (
    Analyzer,
    # REMOVED WeightSensitivityAnalyzer, ParetoFrontAnalyzer, StatisticalAnalyzer - they don't exist yet
)

# Sensitivity Analysis (with ternary diagrams, bootstrap CIs)
from .sensitivity import (
    SensitivityAnalyzer,
    SensitivityVisualizer,
    run_enhanced_sensitivity_analysis,
)

# Sampling for large datasets
from .sampling import (
    sample_for_visualization,
    sample_diverse_3d,
    get_representative_sample,
    get_relevant_window,
)

# Utility functions
from .utils import (
    load_catalyst_data,
    save_results,
    calculate_element_cost,
    format_catalyst_name,
    get_periodic_table_data,
)

# Package metadata for easy access
__all__ = [
    # Version
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__citation__',
    'print_version_info',
    'get_version_dict',
    
    # Core Classes
    'ASCICalculator',
    'Visualizer',
    'Analyzer',
    'DataLoader',
    'ReactionConfig',
    'ScoringFunctions',
    
    # Configuration
    'get_reaction_config',
    'list_available_reactions',
    'validate_weights',
    'get_data_file_path',
    'ASCIConstants',
    
    # Predefined Configs
    'HER_CONFIG',
    'CO2RR_CO_CONFIG',
    'CO2RR_CHO_CONFIG',
    'CO2RR_COCOH_CONFIG',
    'REACTION_CONFIGS',

    # Scoring Functions
    'score_activity',
    'score_stability',
    'score_cost',
    'calculate_asci',
    
    # Visualization Class
    'Visualizer',
    
    # Utilities
    'load_catalyst_data',
    'save_results',
    'calculate_element_cost',
    'format_catalyst_name',
    'get_periodic_table_data',
    
    # Convenience
    'quick_asci',

    # Sensitivity Analysis
    'SensitivityAnalyzer',
    'SensitivityVisualizer',
    'run_enhanced_sensitivity_analysis',

    # Sampling
    'sample_for_visualization',
    'sample_diverse_3d',
    'get_representative_sample',
    'get_relevant_window',
]

# ASCII Art Logo
_LOGO = r"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     █████╗ ███████╗ ██████╗██╗ ██████╗ █████╗ ████████╗               ║
║    ██╔══██╗██╔════╝██╔════╝██║██╔════╝██╔══██╗╚══██╔══╝               ║
║    ███████║███████╗██║     ██║██║     ███████║   ██║                  ║
║    ██╔══██║╚════██║██║     ██║██║     ██╔══██║   ██║                  ║
║    ██║  ██║███████║╚██████╗██║╚██████╗██║  ██║   ██║                  ║
║    ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝                  ║
║                                                                       ║
║    Activity-Stability-Cost Integrated Catalyst Discovery              ║
║    ═══════════════════════════════════════════════════════            ║
║                                                                       ║
║    Multi-Objective Electrocatalyst Screening Toolkit                  ║
║                                                                       ║
║       φ_ASCI = wₐ·Sₐ + wₛ·Sₛ + wc·Sc                                 ║
║                                                                       ║
║       ▸ Rigorous multi-objective catalyst ranking                     ║
║       ▸ Integrates DFT and ML descriptors                             ║
║       ▸ Reproducible, weighted, transparent decisions                 ║
║                                                                       ║
║    "Extending volcano plots beyond activity to practical discovery"   ║
║                                                                       ║
║    Version: 1.0.0                                                     ║
║    Author: N. Khossossi                                               ║
║    Institution: DIFFER                                                ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


def print_logo():
    """Print ASCICat ASCII art logo."""
    print(_LOGO)


def print_citation():
    """Print citation information."""
    print(__citation__)


# Configuration for package behavior
import warnings
import sys

# Set up warning filters
warnings.filterwarnings('default', category=DeprecationWarning, module='ascicat')

# Check Python version
if sys.version_info < (3, 8):
    raise ImportError(
        f"ASCICat requires Python 3.8 or later. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# Optional: Print welcome message on import (can be disabled)
_SHOW_WELCOME = False
if _SHOW_WELCOME:
    print(f"\n✅ ASCICat v{__version__} loaded successfully!")
    print(f"📚 Documentation: {__url__}")
    print(f"💡 Quick start: from ascicat import ASCICalculator\n")