# Installation

This guide covers all installation methods for ASCICat.

## Requirements

- **Python 3.8 or higher** (3.10+ recommended)
- **pip** package manager

### Core Dependencies

These are automatically installed:

| Package | Version | Purpose |
|:--------|:--------|:--------|
| NumPy | >=1.21.0 | Numerical operations |
| Pandas | >=1.3.0 | Data manipulation |
| Matplotlib | >=3.4.0 | Static visualizations |
| Seaborn | >=0.11.0 | Statistical plots |
| SciPy | >=1.7.0 | Scientific computing |
| scikit-learn | >=1.0.0 | Machine learning utilities |
| tqdm | >=4.62.0 | Progress bars |

## Installation Methods

### From PyPI (Recommended)

```bash
pip install ascicat
```

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/NabKh/ASCICat.git
cd ASCICat

# Install in development mode
pip install -e .
```

### With Optional Dependencies

=== "GUI Support"

    For the graphical user interface:

    ```bash
    pip install ascicat[gui]
    ```

    This installs:

    - PyQt5 >=5.15.0
    - pyqtgraph >=0.12.0

=== "Interactive Visualizations"

    For Plotly-based interactive plots:

    ```bash
    pip install ascicat[interactive]
    ```

    This installs:

    - Plotly >=5.0.0
    - ipywidgets >=7.6.0

=== "All Features"

    Install everything:

    ```bash
    pip install ascicat[all]
    ```

=== "Development"

    For contributors:

    ```bash
    pip install ascicat[dev]
    ```

    Includes testing, documentation, and linting tools.

## Virtual Environment (Recommended)

We strongly recommend using a virtual environment:

=== "venv"

    ```bash
    # Create virtual environment
    python -m venv ascicat-env

    # Activate (Linux/macOS)
    source ascicat-env/bin/activate

    # Activate (Windows)
    ascicat-env\Scripts\activate

    # Install ASCICat
    pip install ascicat
    ```

=== "conda"

    ```bash
    # Create conda environment
    conda create -n ascicat python=3.10

    # Activate
    conda activate ascicat

    # Install ASCICat
    pip install ascicat
    ```

## Verify Installation

After installation, verify everything works:

```python
import ascicat
print(f"ASCICat version: {ascicat.__version__}")

# Check dependencies
from ascicat.version import print_dependencies
print_dependencies()
```

Expected output:

```
ASCICat version: 1.0.0

======================================================================
DEPENDENCY STATUS
======================================================================
✓ numpy          : 1.24.3
✓ pandas         : 2.0.1
✓ matplotlib     : 3.7.1
✓ seaborn        : 0.12.2
✓ scipy          : 1.10.1
✓ sklearn        : 1.2.2
✓ tqdm           : 4.65.0
======================================================================
✓ All dependencies satisfied
======================================================================
```

## Command-Line Interface

After installation, the following commands are available:

```bash
# Main CLI
ascicat --help

# Graphical interface (requires [gui] extras)
ascicat-gui

# Batch processing
ascicat-batch --help
```

## Troubleshooting

### Common Issues

??? question "ImportError: No module named 'ascicat'"

    Ensure you've activated your virtual environment and installed the package:

    ```bash
    pip install ascicat
    ```

??? question "PyQt5 installation fails on macOS"

    Try installing via conda:

    ```bash
    conda install pyqt
    pip install ascicat[gui]
    ```

??? question "Matplotlib backend issues"

    Set the backend explicitly:

    ```python
    import matplotlib
    matplotlib.use('Agg')  # For headless servers
    import matplotlib.pyplot as plt
    ```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/NabKh/ASCICat/issues)
2. Search existing discussions
3. Open a new issue with:
   - Python version (`python --version`)
   - ASCICat version (`ascicat --version`)
   - Full error traceback
   - Minimal reproducible example

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade ascicat
```

## Uninstalling

To remove ASCICat:

```bash
pip uninstall ascicat
```
