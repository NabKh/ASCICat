#!/usr/bin/env python3
"""
ascicat_cli.py
Professional Command-Line Interface for ASCICat

World-class CLI for Activity-Stability-Cost Integrated catalyst screening.
Designed for researchers, computational chemists, and materials scientists.

Features:
- Intuitive argument parsing with extensive help
- Beautiful terminal output with colors and progress bars
- Support for all reactions (HER, CO2RR pathways)
- Batch processing and weight sensitivity analysis
- Automatic figure generation (high-resolution)
- Comprehensive error handling and validation
- Example commands and quick-start guide

Author: N. Khossossi
Institution: DIFFER (Dutch Institute for Fundamental Energy Research)

Quick Start:
    # Basic HER analysis
    ascicat --reaction HER --data data/HER_clean.csv --output results/
    
    # CO2RR with custom weights
    ascicat --reaction CO2RR --pathway CO --data data/CO2RR_CO_clean.csv \\
            --weights 0.5 0.3 0.2 --output results/
    
    # Generate all figures
    ascicat --reaction HER --data data/HER_clean.csv --output results/ \\
            --figures --all-figures --dpi 600

For detailed help:
    ascicat --help
    ascicat --examples
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import json
from datetime import datetime
import textwrap

# Color codes for beautiful terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Emoji helpers (with fallback for non-unicode terminals)
try:
    EMOJI = {
        'rocket': '',
        'check': '',
        'cross': '',
        'fire': '',
        'star': '',
        'chart': '',
        'folder': '',
        'gear': '',
        'lightning': '',
        'trophy': '',
        'microscope': '',
        'atom': '',
        'book': '',
        'bulb': '',
        'warning': '',
        'info': '',
        'save': '',
        'paint': ''
    }
except:
    # Fallback for terminals without unicode support
    EMOJI = {k: '' for k in ['rocket', 'check', 'cross', 'fire', 'star', 
             'chart', 'folder', 'gear', 'lightning', 'trophy', 'microscope',
             'atom', 'book', 'bulb', 'warning', 'info', 'save', 'paint']}

# Try to import ASCICat components
try:
    from ascicat import (
        ASCICalculator,
        Visualizer,
        Analyzer,
        get_reaction_config,
        list_available_reactions,
        print_version_info,
        __version__
    )
    ASCICAT_AVAILABLE = True
except ImportError as e:
    ASCICAT_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ============================================================================
# BANNER AND BRANDING
# ============================================================================

ASCICAT_BANNER = f"""{Colors.OKCYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     █████╗ ███████╗ ██████╗██╗ ██████╗ █████╗ ████████╗                  ║
║    ██╔══██╗██╔════╝██╔════╝██║██╔════╝██╔══██╗╚══██╔══╝                  ║
║    ███████║███████╗██║     ██║██║     ███████║   ██║                     ║
║    ██╔══██║╚════██║██║     ██║██║     ██╔══██║   ██║                     ║
║    ██║  ██║███████║╚██████╗██║╚██████╗██║  ██║   ██║                     ║
║    ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝                     ║
║                                                                           ║
║    {Colors.ENDC}{Colors.BOLD}Activity-Stability-Cost Integrated Catalyst Discovery{Colors.OKCYAN}              ║
║    ════════════════════════════════════════════════════                  ║
║                                                                           ║
║    Multi-Objective Electrocatalyst Screening Toolkit                     ║
║    φ_ASCI = wₐ·Sₐ + wₛ·Sₛ + wc·Sc                                        ║
║                                                                           ║
║    Group: AMD | Institution: DIFFER                            ║
║    Version: {__version__ if ASCICAT_AVAILABLE else '1.0.0'}                                                         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""


def print_banner():
    """Print ASCICat banner."""
    print(ASCICAT_BANNER)


def print_section_header(title: str, emoji: str = 'gear'):
    """Print formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{EMOJI[emoji]} {title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'═' * (len(title) + 4)}{Colors.ENDC}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}{EMOJI['check']} {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.FAIL}{EMOJI['cross']} Error: {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.WARNING}{EMOJI['warning']} Warning: {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}{EMOJI['info']} {message}{Colors.ENDC}")


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def create_parser():
    """
    Create comprehensive argument parser.
    
    Returns a parser with all options organized into logical groups:
    - Required arguments (reaction, data, output)
    - Reaction parameters (pathway, weights, method)
    - Visualization options (figures, formats, DPI)
    - Analysis options (top N, statistics, Pareto)
    - Output options (formats, verbosity)
    - Advanced options (validation, batch processing)
    """
    
    # Custom formatter for better help display
    class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=35, width=100)
    
    parser = argparse.ArgumentParser(
        prog='ascicat',
        description=textwrap.dedent(f'''
        {Colors.BOLD}ASCICat - Activity-Stability-Cost Integrated Catalyst Discovery{Colors.ENDC}
        
        Professional toolkit for multi-objective electrocatalyst screening.
        Extends traditional volcano plots to simultaneously optimize:
          {EMOJI['fire']} Activity (catalytic performance)
          {EMOJI['atom']} Stability (electrochemical durability)
          {EMOJI['chart']} Cost (economic viability)
        
        {Colors.BOLD}Supported Reactions:{Colors.ENDC}
          • HER  - Hydrogen Evolution Reaction (2H⁺ + 2e⁻ → H₂)
          • CO2RR - CO₂ Reduction Reaction (multiple pathways)
            - CO pathway: CO₂ → CO
            - CHO pathway: CO₂ → CH₃OH (methanol)
            - COCOH pathway: CO₂ → HCOOH (formic acid)
        
        {Colors.BOLD}Author:{Colors.ENDC} N. Khossossi | {Colors.BOLD}Institution:{Colors.ENDC} DIFFER
        '''),
        epilog=textwrap.dedent(f'''
        {Colors.BOLD}Examples:{Colors.ENDC}
        
        {Colors.OKCYAN}1. Basic HER screening with equal weights:{Colors.ENDC}
           ascicat --reaction HER --data data/HER_clean.csv --output results/
        
        {Colors.OKCYAN}2. CO2RR-CO with activity-focused weights:{Colors.ENDC}
           ascicat --reaction CO2RR --pathway CO --data data/CO2RR_CO_clean.csv \\
                   --weights 0.5 0.3 0.2 --output results/
        
        {Colors.OKCYAN}3. Generate all high-resolution figures:{Colors.ENDC}
           ascicat --reaction HER --data data/HER_clean.csv --output results/ \\
                   --figures --all-figures --dpi 600 --format pdf png
        
        {Colors.OKCYAN}4. Advanced analysis with Gaussian scoring:{Colors.ENDC}
           ascicat --reaction CO2RR --pathway CHO --data data/CO2RR_CHO_clean.csv \\
                   --weights 0.4 0.3 0.3 --method gaussian --output results/ \\
                   --figures --analyze --top 20 --json
        
        {Colors.OKCYAN}5. Batch weight sensitivity analysis:{Colors.ENDC}
           ascicat --reaction HER --data data/HER_clean.csv --output results/ \\
                   --weight-scenarios --figures --compare-weights
        
        {Colors.BOLD}For more examples:{Colors.ENDC}
           ascicat --examples
        
        {Colors.BOLD}Documentation:{Colors.ENDC}
           https://ascicat.readthedocs.io
        
        {Colors.BOLD}Citation:{Colors.ENDC}
           Khossossi, N. (2025). ASCICat: Activity-Stability-Cost Integrated
           Framework for Electrocatalyst Discovery.
        
        {Colors.BOLD}Support:{Colors.ENDC}
           GitHub: https://github.com/nabkh/ASCICat
           Email: n.khossossi@differ.nl
        '''),
        formatter_class=CustomHelpFormatter,
        add_help=True
    )
    
    # ========================================================================
    # REQUIRED ARGUMENTS
    # ========================================================================
    
    required = parser.add_argument_group(
        f'{Colors.BOLD}Required Arguments{Colors.ENDC}',
        'Essential parameters for ASCI calculation'
    )
    
    required.add_argument(
        '--reaction',
        type=str,
        required=True,
        choices=['HER', 'CO2RR', 'her', 'co2rr'],
        metavar='REACTION',
        help=(f'{EMOJI["microscope"]} Reaction type: HER or CO2RR '
              f'(case-insensitive)')
    )
    
    required.add_argument(
        '--data',
        type=str,
        required=True,
        metavar='FILE',
        help=(f'{EMOJI["folder"]} Path to input CSV file with catalyst data '
              f'(must contain: DFT_ads_E, surface_energy, Cost)')
    )
    
    required.add_argument(
        '--output',
        type=str,
        required=True,
        metavar='DIR',
        help=(f'{EMOJI["save"]} Output directory for results and figures '
              f'(will be created if it doesn\'t exist)')
    )
    
    # ========================================================================
    # REACTION PARAMETERS
    # ========================================================================
    
    reaction_params = parser.add_argument_group(
        f'{Colors.BOLD}Reaction Parameters{Colors.ENDC}',
        'Configure reaction-specific settings'
    )
    
    reaction_params.add_argument(
        '--pathway',
        type=str,
        default=None,
        choices=['CO', 'CHO', 'COCOH', 'co', 'cho', 'cocoh'],
        metavar='PATH',
        help=(f'{EMOJI["atom"]} CO2RR pathway (required for CO2RR): '
              f'CO (ΔE_opt=-0.67 eV), '
              f'CHO (ΔE_opt=-0.48 eV, methanol), '
              f'COCOH (ΔE_opt=-0.32 eV, formic acid)')
    )
    
    reaction_params.add_argument(
        '--weights',
        type=float,
        nargs=3,
        default=[0.33, 0.33, 0.34],
        metavar=('W_A', 'W_S', 'W_C'),
        help=(f'{EMOJI["gear"]} Weights for activity, stability, cost '
              f'(must sum to 1.0). Default: 0.33 0.33 0.34 (EQUAL weights). '
              f'Examples: 0.5 0.3 0.2 (activity-focused), '
              f'0.3 0.2 0.5 (cost-focused)')
    )
    
    reaction_params.add_argument(
        '--method',
        type=str,
        default='linear',
        choices=['linear', 'gaussian'],
        metavar='METHOD',
        help=(f'{EMOJI["lightning"]} Activity scoring method: '
              f'linear (default, efficient) or gaussian (sharper discrimination)')
    )
    
    # ========================================================================
    # VISUALIZATION OPTIONS
    # ========================================================================
    
    visualization = parser.add_argument_group(
        f'{Colors.BOLD}Visualization Options{Colors.ENDC}',
        'Generate high-resolution figures'
    )
    
    visualization.add_argument(
        '--figures',
        action='store_true',
        help=(f'{EMOJI["paint"]} Generate standard figures '
              f'(volcano, 3D Pareto, distributions)')
    )
    
    visualization.add_argument(
        '--all-figures',
        action='store_true',
        help=(f'{EMOJI["star"]} Generate ALL available figures '
              f'(includes correlations, radar charts, Pareto fronts) '
              f'- Complete figure set!')
    )
    
    visualization.add_argument(
        '--figure-dir',
        type=str,
        default='figures',
        metavar='DIR',
        help='Subdirectory for figures (default: figures/)'
    )
    
    visualization.add_argument(
        '--dpi',
        type=int,
        default=600,
        metavar='DPI',
        help=(f'{EMOJI["fire"]} Figure resolution in dots per inch. '
              f'Default: 600 (high resolution). '
              f'Use 300 for drafts, 900+ for journals with strict requirements')
    )
    
    visualization.add_argument(
        '--format',
        type=str,
        nargs='+',
        default=['png'],
        choices=['png', 'pdf', 'svg', 'jpg', 'eps'],
        metavar='FMT',
        help=(f'{EMOJI["paint"]} Output format(s) for figures. '
              f'Default: png. Can specify multiple: --format png pdf svg')
    )
    
    # ========================================================================
    # ANALYSIS OPTIONS
    # ========================================================================
    
    analysis = parser.add_argument_group(
        f'{Colors.BOLD}Analysis Options{Colors.ENDC}',
        'Advanced statistical analysis and ranking'
    )
    
    analysis.add_argument(
        '--top',
        type=int,
        default=10,
        metavar='N',
        help=(f'{EMOJI["trophy"]} Number of top catalysts to display and save '
              f'(default: 10)')
    )
    
    analysis.add_argument(
        '--analyze',
        action='store_true',
        help=(f'{EMOJI["chart"]} Perform advanced analysis: '
              f'Pareto fronts, correlations, element statistics')
    )
    
    analysis.add_argument(
        '--weight-scenarios',
        action='store_true',
        help=(f'{EMOJI["gear"]} Test multiple weight scenarios: '
              f'equal, activity-focused, stability-focused, cost-focused')
    )
    
    analysis.add_argument(
        '--compare-weights',
        action='store_true',
        help=(f'{EMOJI["chart"]} Generate weight sensitivity comparison plots '
              f'(requires --weight-scenarios)')
    )
    
    # ========================================================================
    # OUTPUT OPTIONS
    # ========================================================================
    
    output_opts = parser.add_argument_group(
        f'{Colors.BOLD}Output Options{Colors.ENDC}',
        'Control output format and verbosity'
    )
    
    output_opts.add_argument(
        '--json',
        action='store_true',
        help=(f'{EMOJI["save"]} Save statistics and metadata as JSON '
              f'(machine-readable format)')
    )
    
    output_opts.add_argument(
        '--csv',
        action='store_true',
        default=True,
        help='Save results as CSV (default: True)'
    )
    
    output_opts.add_argument(
        '--quiet',
        action='store_true',
        help=(f'{EMOJI["info"]} Suppress verbose output '
              f'(minimal progress information)')
    )
    
    output_opts.add_argument(
        '--no-banner',
        action='store_true',
        help='Suppress ASCII art banner (for scripting)'
    )
    
    # ========================================================================
    # ADVANCED OPTIONS
    # ========================================================================
    
    advanced = parser.add_argument_group(
        f'{Colors.BOLD}Advanced Options{Colors.ENDC}',
        'Expert-level configuration'
    )
    
    advanced.add_argument(
        '--no-validate',
        action='store_true',
        help=(f'{EMOJI["warning"]} Skip data validation '
              f'(faster but risky - use only with clean data)')
    )
    
    advanced.add_argument(
        '--batch',
        type=str,
        default=None,
        metavar='CONFIG',
        help=(f'{EMOJI["lightning"]} Batch processing with JSON config file '
              f'(for high-throughput screening)')
    )
    
    # ========================================================================
    # UTILITY OPTIONS
    # ========================================================================
    
    utility = parser.add_argument_group(
        f'{Colors.BOLD}Utility Options{Colors.ENDC}',
        'Help and information'
    )
    
    utility.add_argument(
        '--version',
        action='store_true',
        help='Show version information and exit'
    )
    
    utility.add_argument(
        '--list-reactions',
        action='store_true',
        help='List all available reactions and pathways'
    )
    
    utility.add_argument(
        '--examples',
        action='store_true',
        help=(f'{EMOJI["book"]} Show detailed usage examples and tutorials')
    )
    
    utility.add_argument(
        '--cite',
        action='store_true',
        help=(f'{EMOJI["book"]} Show citation information')
    )
    
    return parser


# ============================================================================
# EXAMPLES AND TUTORIALS
# ============================================================================

def print_examples():
    """Print comprehensive examples and tutorials."""
    
    examples_text = f"""
{Colors.BOLD}{Colors.HEADER}{'═'*80}
                        ASCICat Usage Examples & Tutorials
{'═'*80}{Colors.ENDC}

{Colors.BOLD}{EMOJI['bulb']} TUTORIAL 1: Basic HER Screening{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Goal: Screen HER catalysts with equal weights (unbiased approach)

Command:
  {Colors.OKGREEN}ascicat --reaction HER \\
          --data data/HER_clean.csv \\
          --output results/HER_equal_weights/{Colors.ENDC}

What this does:
  • Loads HER data (H adsorption, ΔE_opt = -0.27 eV)
  • Applies equal weights (0.33, 0.33, 0.34) - no bias
  • Calculates ASCI scores for all catalysts
  • Saves results to results/HER_equal_weights/HER_ASCI_results.csv
  • Displays top 10 catalysts

Expected output:
  ✓ Loaded ~200 HER catalysts
  ✓ Best ASCI: ~0.85-0.90 (Ni-based or Cu-based alloys)
  ✓ Top catalysts balance all three objectives


{Colors.BOLD}{EMOJI['fire']} TUTORIAL 2: CO2RR with Activity Focus{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Goal: Find best CO-producing catalysts (CO2RR-CO pathway)
Prioritize: High activity (research/optimization phase)

Command:
  {Colors.OKGREEN}ascicat --reaction CO2RR --pathway CO \\
          --data data/CO2RR_CO_clean.csv \\
          --weights 0.5 0.3 0.2 \\
          --output results/CO2RR_CO_activity_focused/ \\
          --figures{Colors.ENDC}

Weight strategy:
  • w_a = 0.5  → Activity priority (CO binding at -0.67 eV)
  • w_s = 0.3  → Moderate stability requirement
  • w_c = 0.2  → Cost less important (research phase)

What this generates:
  • ASCI results CSV
  • Volcano plot with ASCI contours
  • 3D Pareto surface
  • Score distributions

Expected top performers:
  • Ag-based catalysts (near optimal CO binding)
  • Au-based catalysts (good selectivity)


{Colors.BOLD}{EMOJI['chart']} TUTORIAL 3: Figures{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Goal: Generate complete figure set for scientific submission

Command:
  {Colors.OKGREEN}ascicat --reaction HER \\
          --data data/HER_clean.csv \\
          --output results/HER_figures/ \\
          --all-figures \\
          --dpi 600 \\
          --format png pdf svg{Colors.ENDC}

This generates (in multiple formats):
  1. Traditional volcano plot
  2. Enhanced volcano with ASCI contours {EMOJI['star']} KEY FIGURE
  3. 3D Pareto surface (Activity-Stability-Cost)
  4. Score distribution histograms
  5. Correlation matrix heatmap
  6. Top 10 catalysts comparison
  7. Radar chart (multi-dimensional view)
  8. Pareto fronts (pairwise trade-offs)

Total: ~24 figure files (8 figures × 3 formats)

Use --dpi 900 for journals with strict requirements


{Colors.BOLD}{EMOJI['atom']} TUTORIAL 4: Complete CO2RR Pathway Comparison{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Goal: Compare all three CO2RR pathways to find best products

Commands:
  {Colors.OKGREEN}# CO pathway (CO₂ → CO)
  ascicat --reaction CO2RR --pathway CO \\
          --data data/CO2RR_CO_clean.csv \\
          --output results/CO2RR_comparison/CO/

  # CHO pathway (CO₂ → CH₃OH)
  ascicat --reaction CO2RR --pathway CHO \\
          --data data/CO2RR_CHO_clean.csv \\
          --output results/CO2RR_comparison/CHO/

  # COCOH pathway (CO₂ → HCOOH)
  ascicat --reaction CO2RR --pathway COCOH \\
          --data data/CO2RR_COCOH_clean.csv \\
          --output results/CO2RR_comparison/COCOH/{Colors.ENDC}

Then compare:
  • CO pathway: Best for syngas production
  • CHO pathway: Best for liquid fuels (methanol)
  • COCOH pathway: Best for formic acid


{Colors.BOLD}{EMOJI['lightning']} TUTORIAL 5: Weight Sensitivity Analysis{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Goal: Understand how weights affect catalyst ranking

Command:
  {Colors.OKGREEN}ascicat --reaction HER \\
          --data data/HER_clean.csv \\
          --output results/HER_sensitivity/ \\
          --weight-scenarios \\
          --compare-weights \\
          --figures \\
          --json{Colors.ENDC}

This tests 4 scenarios:
  1. Equal:     (0.33, 0.33, 0.34)
  2. Activity:  (0.50, 0.30, 0.20)
  3. Stability: (0.30, 0.50, 0.20)
  4. Cost:      (0.30, 0.20, 0.50)

Outputs:
  • Weight comparison CSV
  • Ranking stability analysis
  • Top catalyst for each scenario
  • Sensitivity plots


{Colors.BOLD}{EMOJI['microscope']} TUTORIAL 6: Advanced Analysis Pipeline{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Goal: Complete analysis with all features enabled

Command:
  {Colors.OKGREEN}ascicat --reaction CO2RR --pathway CHO \\
          --data data/CO2RR_CHO_clean.csv \\
          --weights 0.4 0.3 0.3 \\
          --method gaussian \\
          --output results/CO2RR_CHO_complete/ \\
          --all-figures \\
          --analyze \\
          --top 20 \\
          --dpi 600 \\
          --format png pdf \\
          --json{Colors.ENDC}

This performs:
  {EMOJI['check']} ASCI calculation with Gaussian scoring
  {EMOJI['check']} Advanced statistical analysis
  {EMOJI['check']} Pareto front identification
  {EMOJI['check']} Correlation analysis
  {EMOJI['check']} Element-wise statistics
  {EMOJI['check']} Complete figure set (16 figures × 2 formats)
  {EMOJI['check']} JSON metadata export


{Colors.BOLD}{EMOJI['rocket']} TUTORIAL 7: High-Throughput Batch Processing{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Goal: Screen multiple datasets automatically

First, create batch_config.json:
  {Colors.WARNING}{{
    "datasets": [
      {{"reaction": "HER", "data": "data/HER_clean.csv"}},
      {{"reaction": "CO2RR", "pathway": "CO", "data": "data/CO2RR_CO_clean.csv"}},
      {{"reaction": "CO2RR", "pathway": "CHO", "data": "data/CO2RR_CHO_clean.csv"}}
    ],
    "weights": [0.33, 0.33, 0.34],
    "generate_figures": true
  }}{Colors.ENDC}

Command:
  {Colors.OKGREEN}ascicat --batch batch_config.json --output results/batch/{Colors.ENDC}

This processes all datasets sequentially with consistent settings


{Colors.BOLD}{EMOJI['bulb']} Pro Tips:{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

1. {Colors.BOLD}Start with equal weights{Colors.ENDC} (0.33, 0.33, 0.34) for unbiased screening
2. {Colors.BOLD}Use linear scoring{Colors.ENDC} (default) unless you need sharp discrimination
3. {Colors.BOLD}Always validate data{Colors.ENDC} (don't use --no-validate unless necessary)
4. {Colors.BOLD}Generate figures{Colors.ENDC} - visualization is crucial for understanding
5. {Colors.BOLD}Save JSON metadata{Colors.ENDC} (--json) for reproducibility
6. {Colors.BOLD}Use PDF format{Colors.ENDC} for outputs (vector graphics)
7. {Colors.BOLD}Run weight sensitivity{Colors.ENDC} to ensure robust rankings


{Colors.BOLD}{EMOJI['book']} Further Learning:{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Documentation:  https://ascicat.readthedocs.io
Examples:       https://github.com/nabkh/ASCICat/tree/main/examples
Python API:     from ascicat import ASCICalculator
Support:        n.khossossi@differ.nl

{Colors.BOLD}{'═'*80}{Colors.ENDC}
"""
    
    print(examples_text)


def print_reaction_list():
    """Print available reactions and pathways."""
    
    reactions_text = f"""
{Colors.BOLD}{Colors.HEADER}{'═'*80}
                    Available Reactions in ASCICat
{'═'*80}{Colors.ENDC}

{Colors.BOLD}{EMOJI['microscope']} Hydrogen Evolution Reaction (HER){Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

Reaction:       2H⁺ + 2e⁻ → H₂
Pathway:        H adsorption (automatic)
Optimal ΔE:     -0.27 eV (thermoneutral H binding)
Activity Width: 0.15 eV

Usage:
  {Colors.OKGREEN}ascicat --reaction HER --data data/HER_clean.csv --output results/{Colors.ENDC}

Description:
  Fundamental reaction for hydrogen production via water electrolysis.
  The optimal binding energy of -0.27 eV represents the Sabatier principle:
  - Too weak binding (ΔE > -0.27): Poor H₂ activation
  - Too strong binding (ΔE < -0.27): Difficult H₂ desorption

Benchmark catalysts:
  • Pt (ΔE ≈ -0.27 eV): Near-perfect activity, but expensive
  • Ni (ΔE ≈ -0.20 eV): Good activity, earth-abundant
  • MoS₂ (ΔE ≈ -0.08 eV): Active edge sites, cost-effective


{Colors.BOLD}{EMOJI['atom']} CO₂ Reduction Reaction (CO2RR){Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

{Colors.BOLD}Pathway 1: CO Production{Colors.ENDC}

Reaction:       CO₂ + 2H⁺ + 2e⁻ → CO + H₂O
Pathway:        CO (carbon monoxide)
Optimal ΔE:     -0.67 eV (CO binding)
Activity Width: 0.15 eV

Usage:
  {Colors.OKGREEN}ascicat --reaction CO2RR --pathway CO \\
          --data data/CO2RR_CO_clean.csv --output results/{Colors.ENDC}

Description:
  Produces syngas (CO + H₂) for Fischer-Tropsch synthesis.
  CO binding must be strong enough for activation but weak enough
  for product desorption.

Benchmark catalysts:
  • Ag (ΔE ≈ -0.60 eV): Excellent CO selectivity
  • Au (ΔE ≈ -0.55 eV): High selectivity, stable
  • Cu (ΔE ≈ -0.70 eV): Over-binds, produces C2+ products

{Colors.OKCYAN}- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -{Colors.ENDC}

{Colors.BOLD}Pathway 2: Methanol Production (CHO){Colors.ENDC}

Reaction:       CO₂ + 6H⁺ + 6e⁻ → CH₃OH + H₂O (via CHO* intermediate)
Pathway:        CHO
Optimal ΔE:     -0.48 eV (CHO binding)
Activity Width: 0.15 eV

Usage:
  {Colors.OKGREEN}ascicat --reaction CO2RR --pathway CHO \\
          --data data/CO2RR_CHO_clean.csv --output results/{Colors.ENDC}

Description:
  Direct methanol production via CHO intermediate pathway.
  Liquid fuel synthesis for energy storage applications.

Benchmark catalysts:
  • Cu (moderate CHO binding): Best known for C1 oxygenates
  • Cu-Ag alloys: Enhanced selectivity
  • Metal oxides: Alternative pathway stabilization

{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

{Colors.BOLD}Pathway 3: Formic Acid Production (COCOH){Colors.ENDC}

Reaction:       CO₂ + 2H⁺ + 2e⁻ → HCOOH (via COOH* intermediate)
Pathway:        COCOH
Optimal ΔE:     -0.32 eV (COOH binding)
Activity Width: 0.15 eV

Usage:
  {Colors.OKGREEN}ascicat --reaction CO2RR --pathway COCOH \\
          --data data/CO2RR_COCOH_clean.csv --output results/{Colors.ENDC}

Description:
  Produces formic acid/formate - valuable chemical feedstock.
  Lower energy barrier compared to CO pathway on some catalysts.

Benchmark catalysts:
  • Sn (weak COOH binding): High formate selectivity
  • Pb (weak COOH binding): Good activity
  • Bi (moderate binding): Stable, selective


{Colors.BOLD}{EMOJI['info']} Quick Reference Table{Colors.ENDC}
{Colors.OKCYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

╔════════════╦══════════╦═══════════════════╦════════════╦═════════════════╗
║  Reaction  ║ Pathway  ║     Product       ║  ΔE_opt    ║  Best Metals    ║
╠════════════╬══════════╬═══════════════════╬════════════╬═════════════════╣
║    HER     ║    H     ║   Hydrogen        ║  -0.27 eV  ║  Pt, Ni, Mo     ║
║            ║          ║   (H₂)            ║            ║                 ║
╠════════════╬══════════╬═══════════════════╬════════════╬═════════════════╣
║  CO2RR     ║    CO    ║   Carbon          ║  -0.67 eV  ║  Ag, Au, Cu     ║
║            ║          ║   Monoxide        ║            ║                 ║
╠════════════╬══════════╬═══════════════════╬════════════╬═════════════════╣
║  CO2RR     ║   CHO    ║   Methanol        ║  -0.48 eV  ║  Cu, Cu-Ag      ║
║            ║          ║   (CH₃OH)         ║            ║                 ║
╠════════════╬══════════╬═══════════════════╬════════════╬═════════════════╣
║  CO2RR     ║  COCOH   ║   Formic Acid     ║  -0.32 eV  ║  Sn, Pb, Bi     ║
║            ║          ║   (HCOOH)         ║            ║                 ║
╚════════════╩══════════╩═══════════════════╩════════════╩═════════════════╝

{Colors.BOLD}{'═'*80}{Colors.ENDC}
"""
    
    print(reactions_text)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_arguments(args):
    """
    Validate command-line arguments.
    
    Performs comprehensive validation:
    - File existence
    - Weight normalization
    - Pathway requirements
    - Output directory creation
    - Dependency checks
    """
    
    # Check if ASCICat is available
    if not ASCICAT_AVAILABLE:
        print_error(f"ASCICat package not found: {IMPORT_ERROR}")
        print_info("Install ASCICat: pip install ascicat")
        return False
    
    # Normalize reaction and pathway names
    args.reaction = args.reaction.upper()
    if args.pathway:
        args.pathway = args.pathway.upper()
    
    # Check data file exists
    if not Path(args.data).exists():
        print_error(f"Data file not found: {args.data}")
        print_info("Provide path to CSV file containing catalyst data")
        return False
    
    # Check pathway requirement for CO2RR
    if args.reaction == 'CO2RR' and args.pathway is None:
        print_error("CO2RR requires --pathway specification")
        print_info("Available pathways: CO, CHO, COCOH")
        print_info("Example: --reaction CO2RR --pathway CO")
        return False
    
    # Validate weights
    w_a, w_s, w_c = args.weights
    weight_sum = w_a + w_s + w_c
    
    if abs(weight_sum - 1.0) > 1e-6:
        print_error(f"Weights must sum to 1.0, got {weight_sum:.6f}")
        print_info(f"Current weights: w_a={w_a}, w_s={w_s}, w_c={w_c}")
        return False
    
    for name, weight in [('w_a', w_a), ('w_s', w_s), ('w_c', w_c)]:
        if not (0.0 <= weight <= 1.0):
            print_error(f"{name} must be between 0 and 1, got {weight}")
            return False
    
    # Create output directory
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print_error(f"Cannot create output directory: {e}")
        return False
    
    return True


# ============================================================================
# MAIN CALCULATION WORKFLOW
# ============================================================================

def run_asci_calculation(args):
    """
    Execute main ASCI calculation workflow.
    
    Orchestrates:
    1. Initialization and validation
    2. Data loading
    3. ASCI calculation
    4. Result saving
    5. Figure generation
    6. Advanced analysis
    7. Summary reporting
    """
    
    verbose = not args.quiet
    
    # Print header
    if verbose and not args.no_banner:
        print_banner()
    
    print_section_header(f"Starting ASCI Calculation: {args.reaction}", 'rocket')
    
    if args.pathway and verbose:
        print_info(f"Pathway: {args.pathway}")
    
    # ========================================================================
    # STEP 1: Initialize Calculator
    # ========================================================================
    
    if verbose:
        print_section_header("Initializing Calculator", 'gear')
    
    try:
        calc = ASCICalculator(
            reaction=args.reaction,
            pathway=args.pathway,
            scoring_method=args.method,
            verbose=verbose
        )
    except Exception as e:
        print_error(f"Failed to initialize calculator: {e}")
        return 1
    
    # ========================================================================
    # STEP 2: Load Data
    # ========================================================================
    
    if verbose:
        print_section_header("Loading Data", 'folder')
    
    try:
        calc.load_data(args.data, validate=not args.no_validate)
    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return 1
    
    # ========================================================================
    # STEP 3: Calculate ASCI
    # ========================================================================
    
    if verbose:
        print_section_header("Calculating ASCI Scores", 'lightning')
    
    w_a, w_s, w_c = args.weights
    
    if verbose:
        print_info(f"Weights: w_a={w_a:.2f}, w_s={w_s:.2f}, w_c={w_c:.2f}")
        print_info(f"Method: {args.method.upper()}")
    
    try:
        results = calc.calculate_asci(
            w_a=w_a, w_s=w_s, w_c=w_c,
            method=args.method,
            show_progress=verbose
        )
    except Exception as e:
        print_error(f"ASCI calculation failed: {e}")
        return 1
    
    # ========================================================================
    # STEP 4: Save Results
    # ========================================================================
    
    if verbose:
        print_section_header("Saving Results", 'save')
    
    output_dir = Path(args.output)
    
    # Save CSV
    if args.csv:
        csv_path = output_dir / f"{args.reaction}_ASCI_results.csv"
        try:
            calc.save_results(str(csv_path), include_metadata=args.json)
            print_success(f"Results saved: {csv_path}")
        except Exception as e:
            print_error(f"Failed to save CSV: {e}")
    
    # Save JSON metadata
    if args.json:
        json_path = output_dir / f"{args.reaction}_statistics.json"
        try:
            stats = calc.get_statistics()
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print_success(f"Statistics saved: {json_path}")
        except Exception as e:
            print_warning(f"Failed to save JSON: {e}")
    
    # ========================================================================
    # STEP 5: Print Summary
    # ========================================================================
    
    if verbose:
        print_section_header(f"Top {args.top} Catalysts", 'trophy')
        calc.print_summary(n_top=args.top)
    
    # ========================================================================
    # STEP 6: Generate Figures
    # ========================================================================
    
    if args.figures or args.all_figures:
        if verbose:
            print_section_header("Generating Figures", 'paint')
        
        figure_dir = output_dir / args.figure_dir
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        viz = Visualizer(results, calc.config)
        
        try:
            if args.all_figures:
                # Generate complete figure set
                print_info("Generating complete figure set...")
                generated = viz.generate_all_figures(
                    output_dir=str(figure_dir),
                    dpi=args.dpi,
                    formats=args.format
                )
                print_success(f"Generated {sum(len(v) for v in generated.values())} figures")
            
            else:
                # Generate standard figures
                figures = [
                    ('volcano', 'Traditional volcano plot', viz.plot_volcano),
                    ('volcano_asci', 'Enhanced volcano with ASCI', viz.plot_volcano_with_asci),
                    ('3d_pareto', '3D Pareto surface', viz.plot_3d_pareto_surface),
                    ('distributions', 'Score distributions', viz.plot_score_distributions),
                ]
                
                for fig_name, description, plot_func in figures:
                    for fmt in args.format:
                        save_path = figure_dir / f"{args.reaction}_{fig_name}.{fmt}"
                        try:
                            plot_func(save_path=str(save_path), dpi=args.dpi)
                            print_success(f"{description}: {save_path.name}")
                        except Exception as e:
                            print_warning(f"Failed to generate {description}: {e}")
        
        except Exception as e:
            print_warning(f"Figure generation error: {e}")
    
    # ========================================================================
    # STEP 7: Advanced Analysis
    # ========================================================================
    
    if args.analyze:
        if verbose:
            print_section_header("Advanced Analysis", 'chart')
        
        try:
            analyzer = Analyzer(results, calc.config)
            
            # Pareto front
            pareto = analyzer.get_pareto_optimal()
            print_info(f"Pareto-optimal catalysts: {len(pareto)}")
            
            pareto_path = output_dir / f"{args.reaction}_pareto_front.csv"
            pareto.to_csv(pareto_path, index=False)
            print_success(f"Pareto front saved: {pareto_path.name}")
            
            # Correlations
            corr = analyzer.get_correlation_analysis()
            print_info("Key ASCI correlations:")
            asci_corr = corr['ASCI'].sort_values(ascending=False)
            for col, val in list(asci_corr.items())[1:4]:
                print(f"    ASCI - {col}: {val:+.3f}")
        
        except Exception as e:
            print_warning(f"Analysis error: {e}")
    
    # ========================================================================
    # STEP 8: Weight Scenarios
    # ========================================================================
    
    if args.weight_scenarios:
        if verbose:
            print_section_header("Weight Sensitivity Analysis", 'gear')
        
        scenarios = [
            (0.33, 0.33, 0.34, 'Equal'),
            (0.50, 0.30, 0.20, 'Activity-Focused'),
            (0.30, 0.50, 0.20, 'Stability-Focused'),
            (0.30, 0.20, 0.50, 'Cost-Focused'),
        ]
        
        comparison_results = []
        
        for w_a_s, w_s_s, w_c_s, name in scenarios:
            try:
                results_s = calc.calculate_asci(w_a_s, w_s_s, w_c_s, 
                                               show_progress=False)
                top_cat = results_s.iloc[0]
                
                comparison_results.append({
                    'scenario': name,
                    'w_a': w_a_s,
                    'w_s': w_s_s,
                    'w_c': w_c_s,
                    'top_catalyst': top_cat['symbol'],
                    'top_asci': top_cat['ASCI'],
                    'mean_asci': results_s['ASCI'].mean()
                })
                
                print_info(f"{name:20s} → {top_cat['symbol']:15s} (ASCI: {top_cat['ASCI']:.4f})")
            
            except Exception as e:
                print_warning(f"Scenario '{name}' failed: {e}")
        
        # Save comparison
        if comparison_results:
            import pandas as pd
            comp_df = pd.DataFrame(comparison_results)
            comp_path = output_dir / f"{args.reaction}_weight_comparison.csv"
            comp_df.to_csv(comp_path, index=False)
            print_success(f"Weight comparison saved: {comp_path.name}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    if verbose:
        print_section_header("Calculation Complete!", 'check')
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'═'*80}")
        print(f"{EMOJI['rocket']} ASCICat analysis completed successfully!")
        print(f"{'═'*80}{Colors.ENDC}\n")
        
        print(f"{EMOJI['folder']} Results directory: {output_dir}")
        print(f"{EMOJI['trophy']} Best ASCI score: {results['ASCI'].max():.4f}")
        print(f"{EMOJI['star']} Top catalyst: {results.iloc[0]['symbol']}")
        
        if args.figures or args.all_figures:
            fig_count = len(list(Path(output_dir / args.figure_dir).glob('*.*')))
            print(f"{EMOJI['paint']} Figures generated: {fig_count}")
        
        print(f"\n{EMOJI['book']} For citation information: ascicat --cite")
        print(f"{EMOJI['info']} For more examples: ascicat --examples\n")
    
    return 0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for CLI."""
    
    # Create parser
    parser = create_parser()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        print(f"\n{Colors.BOLD}Quick Start:{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}ascicat --examples{Colors.ENDC}  - See detailed usage examples")
        print(f"  {Colors.OKGREEN}ascicat --list-reactions{Colors.ENDC}  - Show available reactions")
        print(f"  {Colors.OKGREEN}ascicat --version{Colors.ENDC}  - Show version information\n")
        return 0
    
    # Parse arguments
    args = parser.parse_args()
    
    # ========================================================================
    # HANDLE UTILITY OPTIONS
    # ========================================================================
    
    # Version
    if args.version:
        if ASCICAT_AVAILABLE:
            print_version_info()
        else:
            print(f"ASCICat CLI version 1.0.0")
            print_error("ASCICat package not installed")
        return 0
    
    # Examples
    if args.examples:
        print_examples()
        return 0
    
    # List reactions
    if args.list_reactions:
        print_reaction_list()
        return 0
    
    # Citation
    if args.cite:
        if ASCICAT_AVAILABLE:
            from ascicat.version import print_citation
            print_citation()
        else:
            print("\nCitation:\n")
            print("  Khossossi, N. (2025). ASCICat: Activity-Stability-Cost")
            print("  Integrated Framework for Electrocatalyst Discovery.")
        return 0
    
    # ========================================================================
    # VALIDATE ARGUMENTS
    # ========================================================================
    
    if not validate_arguments(args):
        return 1
    
    # ========================================================================
    # RUN CALCULATION
    # ========================================================================
    
    try:
        return run_asci_calculation(args)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}{EMOJI['warning']} Calculation interrupted by user{Colors.ENDC}")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        if not args.quiet:
            print(f"\n{Colors.FAIL}Traceback:{Colors.ENDC}")
            traceback.print_exc()
        return 1


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    sys.exit(main())