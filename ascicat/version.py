"""
ASCICat Version Information
Activity-Stability-Cost Integrated Catalyst Discovery
Author: N. Khossossi
Institution: DIFFER (Dutch Institute for Fundamental Energy Research)
"""

__version__ = '1.0.0'
__version_info__ = (1, 0, 0)

# Author Information
__author__ = 'N. Khossossi'
__email__ = 'n.khossossi@differ.nl'
__institution__ = 'Dutch Institute for Fundamental Energy Research (DIFFER)'

# Project URLs
__url__ = 'https://github.com/nabkh/ASCICat'
__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2025 N. Khossossi, DIFFER'

# Description
__description__ = (
    'Activity-Stability-Cost Integrated Framework for '
    'Electrocatalyst Discovery - Extending Nørskov\'s '
    'volcano plot approach to multi-objective optimization'
)

# Citation Information
__citation__ = """
If you use ASCICat in your research, please cite:

    Khossossi, N. (2025). ASCICat: Activity-Stability-Cost Integrated
    Framework for Electrocatalyst Discovery. Journal of Computational
    Chemistry, XX, XXX-XXX. DOI: 10.xxxx/xxxxxx

BibTeX:
    @article{khossossi2025ascicat,
      title={{ASCICat: Activity-Stability-Cost Integrated Framework for 
             Electrocatalyst Discovery}},
      author={Khossossi, N.},
      journal={Journal of Computational Chemistry},
      volume={XX},
      pages={XXX--XXX},
      year={2025},
      doi={10.xxxx/xxxxxx},
      publisher={Wiley}
    }

Preprint: arXiv:XXXX.XXXXX
"""

# Release Information
__release_date__ = '2025-01-15'
__status__ = 'Production/Stable'

# Scientific References - Theoretical Foundation
__references__ = [
    'Nørskov, J. K. et al. Nat. Chem. 1, 37 (2009) - Computational catalyst design',
    'Greeley, J. et al. Nat. Mater. 5, 909 (2006) - Volcano plot screening methodology',
    'Sabatier, P. Ber. Dtsch. Chem. Ges. 44, 1984 (1911) - Sabatier principle foundation',
    'Peterson, A. A. & Nørskov, J. K. J. Phys. Chem. Lett. 3, 251 (2012) - CO2RR mechanisms',
]

# Package Keywords
__keywords__ = [
    'catalysis',
    'electrocatalysis',
    'DFT',
    'volcano plot',
    'materials science',
    'computational chemistry',
    'multi-objective optimization',
    'catalyst screening',
    'hydrogen evolution',
    'CO2 reduction',
    'machine learning',
]


def print_version_info():
    """
    Print comprehensive version information to console.
    
    Displays package version, author, institution, and URLs.
    Useful for debugging and citation purposes.
    
    Examples
    --------
    >>> from ascicat.version import print_version_info
    >>> print_version_info()
    
    ASCICat v1.0.0
    Author: N. Khossossi
    Institution: DIFFER
    ...
    """
    print(f"\n{'='*70}")
    print(f"ASCICat v{__version__}")
    print(f"{'='*70}")
    print(f"Author:      {__author__}")
    print(f"Email:       {__email__}")
    print(f"Institution: {__institution__}")
    print(f"License:     {__license__}")
    print(f"URL:         {__url__}")
    print(f"Released:    {__release_date__}")
    print(f"Status:      {__status__}")
    print(f"{'='*70}\n")


def print_citation():
    """
    Print citation information to console.
    
    Displays proper citation format for academic use,
    including BibTeX entry and preprint information.
    
    Examples
    --------
    >>> from ascicat.version import print_citation
    >>> print_citation()
    """
    print("\n" + "="*70)
    print("CITATION INFORMATION")
    print("="*70)
    print(__citation__)
    print("="*70 + "\n")


def get_version_dict():
    """
    Return version information as dictionary.
    
    Provides programmatic access to all version metadata.
    Useful for automated testing and validation.
    
    Returns
    -------
    dict
        Dictionary containing all version metadata:
        - version: str, version number
        - version_info: tuple, version components
        - author: str, author name
        - email: str, contact email
        - institution: str, institutional affiliation
        - url: str, project URL
        - license: str, license type
        - description: str, package description
        - release_date: str, release date
        - status: str, development status
        - keywords: list, package keywords
    
    Examples
    --------
    >>> from ascicat.version import get_version_dict
    >>> info = get_version_dict()
    >>> print(f"Version: {info['version']}")
    Version: 1.0.0
    >>> print(f"Author: {info['author']}")
    Author: N. Khossossi
    """
    return {
        'version': __version__,
        'version_info': __version_info__,
        'author': __author__,
        'email': __email__,
        'institution': __institution__,
        'url': __url__,
        'license': __license__,
        'description': __description__,
        'release_date': __release_date__,
        'status': __status__,
        'keywords': __keywords__,
    }


def check_dependencies():
    """
    Check if required dependencies are installed.
    
    Validates that all necessary packages are available
    and reports their versions. Useful for troubleshooting.
    
    Returns
    -------
    dict
        Dictionary mapping package names to version strings.
        Returns None for missing packages.
    
    Examples
    --------
    >>> from ascicat.version import check_dependencies
    >>> deps = check_dependencies()
    >>> print(deps['numpy'])
    1.24.3
    """
    dependencies = {}
    
    # Core scientific packages
    required_packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'sklearn',
        'tqdm',
    ]
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            dependencies[package] = version
        except ImportError:
            dependencies[package] = None
    
    return dependencies


def print_dependencies():
    """
    Print status of all dependencies.
    
    Displays installed versions of required packages
    and highlights any missing dependencies.
    
    Examples
    --------
    >>> from ascicat.version import print_dependencies
    >>> print_dependencies()
    
    Dependency Status:
    ✓ numpy: 1.24.3
    ✓ pandas: 2.0.1
    ...
    """
    deps = check_dependencies()
    
    print(f"\n{'='*70}")
    print("DEPENDENCY STATUS")
    print(f"{'='*70}")
    
    all_installed = True
    for package, version in deps.items():
        if version is None:
            print(f"✗ {package:15s}: NOT INSTALLED")
            all_installed = False
        else:
            print(f"✓ {package:15s}: {version}")
    
    print(f"{'='*70}")
    
    if all_installed:
        print("✓ All dependencies satisfied")
    else:
        print("⚠ Some dependencies missing - install with: pip install ascicat")
    
    print(f"{'='*70}\n")


# ASCII Art Logo for Installation
ASCICAT_LOGO = r"""
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
    """
    Print ASCICat ASCII art logo.
    
    Displays the package logo with version and author info.
    Used during package import and installation.
    
    Examples
    --------
    >>> from ascicat.version import print_logo
    >>> print_logo()
    """
    print(ASCICAT_LOGO)


def print_welcome():
    """
    Print welcome message with quick start guide.
    
    Displays logo, version info, and basic usage instructions.
    Automatically called on package import if enabled.
    
    Examples
    --------
    >>> from ascicat.version import print_welcome
    >>> print_welcome()
    """
    print_logo()
    print(f"\n✓ ASCICat v{__version__} loaded successfully!")
    print(f"📚 Documentation: {__url__}")
    print(f"📧 Contact: {__email__}")
    print(f"\n💡 Quick Start:")
    print(f"    from ascicat import ASCICalculator")
    print(f"    calc = ASCICalculator(reaction='HER')")
    print(f"    calc.load_data('data/HER_clean.csv')")
    print(f"    results = calc.calculate_asci()\n")


# Semantic versioning helper
def get_major_version():
    """Get major version number."""
    return __version_info__[0]


def get_minor_version():
    """Get minor version number."""
    return __version_info__[1]


def get_patch_version():
    """Get patch version number."""
    return __version_info__[2]


# For programmatic access
def get_version():
    """Get version string."""
    return __version__


def get_author():
    """Get author name."""
    return __author__


def get_email():
    """Get contact email."""
    return __email__


# Module-level test
if __name__ == '__main__':
    # Run all display functions for testing
    print_logo()
    print_version_info()
    print_dependencies()
    print_citation()
    
    # Test dictionary access
    info = get_version_dict()
    print("\nVersion Dictionary:")
    for key, value in info.items():
        print(f"  {key}: {value}")