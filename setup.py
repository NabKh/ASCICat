"""
ASCICat Setup Configuration
Activity-Stability-Cost Integrated Catalyst Discovery
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

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
║    Multi-Objective Electrocatalyst Screening Toolkit               ║
║                                                                       ║
║       φ_ASCI = wₐ·Sₐ + wₛ·Sₛ + w_c·S_c                                 ║
║                                                                       ║
║       ▸ Deterministic multi-objective catalyst ranking                ║
║       ▸ Integrates DFT, ML, and experimental descriptors              ║
║       ▸ Reproducible, weighted, transparent decisions                 ║
║                                                                       ║
║                                                                       ║
║       “Because even catalysts deserve a fair evaluation.”             ║
║                                                                       ║
║                                                                       ║
║    Version: 1.0.0                                                     ║
║    Author: N. Khossossi                                               ║
║    Institution: DIFFER (Dutch Institute for Fundamental Energy)       ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def print_installation_message():
    """Print installation success message with logo"""
    print("\n" + ASCICAT_LOGO)
    print("\n ASCICat successfully installed!")
    print("\n Quick Start:")
    print("   - Documentation: https://ascicat.readthedocs.io")
    print("   - Examples: See examples/ directory")
    print("   - CLI: Run 'ascicat --help'")
    print("   - GUI: Run 'ascicat-gui'")
    print("   Integrated Framework for Electrocatalyst Discovery.")
    print("\n Happy Catalyst Screening! \n")

# Read long description from README
readme_path = Path(__file__).parent / 'README.md'
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "ASCICat: Activity-Stability-Cost Integrated Catalyst Discovery"

# Read version
version = {}
version_path = Path(__file__).parent / 'ascicat' / 'version.py'
if version_path.exists():
    with open(version_path, 'r', encoding='utf-8') as f:
        exec(f.read(), version)
else:
    version['__version__'] = '1.0.0'

# Core dependencies
INSTALL_REQUIRES = [
    'numpy>=1.21.0',
    'pandas>=1.3.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
    'tqdm>=4.62.0',
]

# Optional dependencies for GUI
GUI_REQUIRES = [
    'PyQt5>=5.15.0',
    'pyqtgraph>=0.12.0',
]

# Optional dependencies for interactive visualizations
INTERACTIVE_REQUIRES = [
    'plotly>=5.0.0',
    'ipywidgets>=7.6.0',
]

# Development dependencies
DEV_REQUIRES = [
    'pytest>=6.2.0',
    'pytest-cov>=3.0.0',
    'black>=21.0',
    'flake8>=4.0.0',
    'mypy>=0.910',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=1.0.0',
    'nbsphinx>=0.8.0',
]

# All optional dependencies
EXTRAS_REQUIRE = {
    'gui': GUI_REQUIRES,
    'interactive': INTERACTIVE_REQUIRES,
    'dev': DEV_REQUIRES + GUI_REQUIRES + INTERACTIVE_REQUIRES,
    'all': GUI_REQUIRES + INTERACTIVE_REQUIRES,
}

setup(
    name='ascicat',
    version=version['__version__'],
    
    # Package metadata
    description='Activity-Stability-Cost Integrated Catalyst Discovery Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # Author information
    author='N. Khossossi',
    author_email='n.khossossi@differ.nl',
    
    # Project URLs
    url='https://github.com/nabkh/ASCICat',
    project_urls={
        'Documentation': 'https://ascicat.readthedocs.io',
        'Source': 'https://github.com/nabkh/ASCICat',
        'Tracker': 'https://github.com/nabkh/ASCICat/issues',
        #'Paper': 'https://doi.org/10.xxxx/xxxxxx',  # Update when published
    },
    
    # License
    license='MIT',
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords for searchability
    keywords=[
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
    ],
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Package discovery
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    
    # Include package data
    include_package_data=True,
    package_data={
        'ascicat': [
            'data/*.csv',
            'data/*.json',
        ],
    },
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'ascicat=ascicat.cli:main',
            'ascicat-gui=ascicat.gui:main',
            'ascicat-batch=ascicat.batch:main',
        ],
    },
    
    # Zip safety
    zip_safe=False,
)

# Print installation message
if 'install' in sys.argv or 'develop' in sys.argv or 'egg_info' in sys.argv:
    print_installation_message()