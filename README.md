<p align="center">
  <img src="https://raw.githubusercontent.com/NabKh/ASCICat/main/logo/logo.png" alt="ASCICat Logo" width="280"/>
</p>

<h3 align="center">Activity-Stability-Cost Integrated Catalyst Discovery</h3>

<p align="center">
  <em>A unified multi-objective framework for translating computational catalyst data into reproducible, experimentally-actionable rankings</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://doi.org/10.xxxx/xxxxxx"><img src="https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue.svg" alt="DOI"></a>
  <a href="https://github.com/NabKh/ASCICat/stargazers"><img src="https://img.shields.io/github/stars/NabKh/ASCICat?style=social" alt="GitHub Stars"></a>
</p>

<p align="center">
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/01_Introduction_Quick_Start.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://mybinder.org/v2/gh/NabKh/ASCICat/master?labpath=tutorials"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
</p>

---

## Overview

The computational catalysis community has generated **massive ML/DFT databases** (OpenCatalyst, Materials Project, Catalysis-Hub, NOMAD, AFLOW) containing thousands of calculated catalyst properties. However, there exists no standardized framework to translate this wealth of data into actionable experimental priorities.

**ASCICat bridges this gap** by providing a unified, transparent, and reproducible framework for multi-objective catalyst prioritization.

<p align="center">
  <img src="https://img.shields.io/badge/Input-DFT%20Data-orange?style=for-the-badge" alt="Input"/>
  &nbsp;→&nbsp;
  <img src="https://img.shields.io/badge/ASCICat-Processing-blue?style=for-the-badge" alt="ASCICat"/>
  &nbsp;→&nbsp;
  <img src="https://img.shields.io/badge/Output-Ranked%20Catalysts-green?style=for-the-badge" alt="Output"/>
</p>

---

## The Problem We Solve

| Current Challenge | ASCICat Solution |
|:-----------------|:-----------------|
| No unified framework for catalyst selection | Standardized ASCI metric applicable to any catalyst dataset |
| Ad-hoc, non-reproducible selection criteria | Transparent weighting with explicit trade-off documentation |
| Results cannot be compared across studies | Common metric enables direct cross-study comparison |
| Hidden assumptions in catalyst ranking | Built-in sensitivity analysis reveals weight dependencies |

---

## Scientific Framework

ASCICat implements a **three-pillar scoring system** grounded in fundamental catalysis principles:

<table>
<tr>
<td align="center" width="33%">
<h3>Activity (S<sub>a</sub>)</h3>
<strong>Sabatier Principle</strong><br/>
Optimal binding energy for reaction kinetics
</td>
<td align="center" width="33%">
<h3>Stability (S<sub>s</sub>)</h3>
<strong>Surface Thermodynamics</strong><br/>
Dissolution resistance and durability
</td>
<td align="center" width="33%">
<h3>Cost (S<sub>c</sub>)</h3>
<strong>Economic Viability</strong><br/>
Material pricing and availability
</td>
</tr>
</table>

**The Unified ASCI Metric:**

```
φ_ASCI = w_a · S_a + w_s · S_s + w_c · S_c

where:  w_a + w_s + w_c = 1
        S_a, S_s, S_c ∈ [0, 1]
```

---

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/NabKh/ASCICat.git
cd ASCICat

# Install in development mode
pip install -e .
```

### Optional Dependencies

```bash
# For GUI interface
pip install -e ".[gui]"

# For interactive Jupyter visualizations
pip install -e ".[interactive]"

# For full development environment
pip install -e ".[dev]"
```

### Requirements

- Python 3.8+
- NumPy, Pandas, SciPy, Matplotlib, Seaborn
- See `setup.py` for complete dependency list

---

## Quick Start

### Python API

```python
from ascicat import ASCICalculator

# Initialize calculator for HER reaction
calc = ASCICalculator(reaction='HER')

# Load your DFT data
calc.load_data('data/HER_clean.csv')

# Calculate ASCI scores with custom weights
results = calc.calculate_asci(
    w_a=0.4,  # 40% weight on Activity
    w_s=0.3,  # 30% weight on Stability
    w_c=0.3   # 30% weight on Cost
)

# Get top-ranked catalysts
top_catalysts = calc.get_top_catalysts(n=10)
print(top_catalysts[['symbol', 'ASCI', 'activity_score', 'stability_score', 'cost_score']])
```

**Output:**
```
      symbol     ASCI  activity_score  stability_score  cost_score
0      Fe2Sb4   0.899           0.923            0.851       0.924
1       Cu3Sb   0.887           0.912            0.867       0.883
2      Cu6Sb2   0.876           0.889            0.856       0.882
...
```

### Command-Line Interface

```bash
# HER screening with default weights
ascicat --reaction HER --data data/HER_clean.csv --output results/

# CO2RR screening with custom weights
ascicat --reaction CO2RR --pathway CO --weights 0.5 0.3 0.2 --output results/

# Launch graphical interface
ascicat-gui
```

---

## Interactive Tutorials

Run the tutorials directly in your browser — no installation required:

| Tutorial | Description | Launch |
|:---------|:------------|:-------|
| **01 - Introduction** | ASCICat framework and basic workflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/01_Introduction_Quick_Start.ipynb) |
| **02 - HER Screening** | Complete HER catalyst analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/02_HER_Catalyst_Screening.ipynb) |
| **03 - CO2RR Analysis** | Multi-pathway CO2RR screening | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/03_CO2RR_Multi_Pathway_Analysis.ipynb) |
| **04 - Visualization** | Publication-quality figure generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/04_Visualization_Guide.ipynb) |
| **05 - Sensitivity** | Weight sensitivity analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/05_Sensitivity_Analysis.ipynb) |

Or launch all tutorials in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NabKh/ASCICat/master?labpath=tutorials)

---

## Features

### Core Capabilities

- **Multi-objective optimization** with customizable weight preferences
- **Multiple reaction pathways**: HER, CO2RR-CO, CO2RR-CHO, CO2RR-COCOH
- **Publication-quality visualizations** at 600 DPI
- **Comprehensive sensitivity analysis** for robust screening
- **Pareto frontier analysis** for trade-off exploration
- **Batch processing** for high-throughput screening
- **Colorblind-friendly** visualization palettes

### Supported Reactions

| Reaction | Pathway | Optimal ΔE | Description |
|:---------|:--------|:-----------|:------------|
| **HER** | H adsorption | -0.27 eV | Hydrogen Evolution Reaction |
| **CO2RR** | CO | -0.67 eV | Carbon monoxide production |
| **CO2RR** | CHO | -0.48 eV | Methanol pathway |
| **CO2RR** | COCOH | -0.32 eV | Formic acid pathway |

---

## Sensitivity Analysis

ASCICat includes built-in tools to address the critical "weight selection problem":

```python
from ascicat import SensitivityAnalyzer, SensitivityVisualizer

# Analyze sensitivity across weight space
analyzer = SensitivityAnalyzer(calc)
results = analyzer.analyze_full_weight_space(resolution=20)

# Generate visualization suite
visualizer = SensitivityVisualizer(results)
visualizer.plot_ternary_heatmap()
visualizer.plot_catalyst_dominance()
visualizer.plot_rank_trajectories()
```

This enables researchers to:
- Understand how rankings depend on weight choices
- Identify **robust candidates** that rank well regardless of weights
- Document the sensitivity of conclusions transparently

---

## Examples

The `examples/` directory contains complete, documented workflows:

| Example | Description |
|:--------|:------------|
| [`example_1_HER_screening.py`](examples/example_1_HER_screening.py) | Complete HER catalyst screening workflow |
| [`example_2_CO2RR_screening.py`](examples/example_2_CO2RR_screening.py) | Multi-pathway CO2RR analysis |
| [`example_3_visualization.py`](examples/example_3_visualization.py) | Publication figure generation |
| [`example_4_sensitivity_analysis.py`](examples/example_4_sensitivity_analysis.py) | Weight sensitivity study |
| [`example_5_ascicat_vs_pareto.py`](examples/example_5_ascicat_vs_pareto.py) | Comparison with Pareto methods |

---

## Data Format

ASCICat accepts CSV files with the following structure:

| Column | Type | Description | Unit |
|:-------|:-----|:------------|:-----|
| `DFT_ads_E` | float | Adsorption energy | eV |
| `surface_energy` | float | Surface energy | J/m² |
| `Cost` | float | Material cost | $/kg |
| `symbol` | str | Catalyst identifier | - |
| `optimal_energy` | float | Sabatier optimum | eV |

See [`data/README_DATA.md`](data/README_DATA.md) for detailed specifications.

---

## Complementarity with Pareto Analysis

ASCICat and Pareto frontier methods are **complementary approaches**:

| Pareto Analysis | ASCICat |
|:----------------|:--------|
| Identifies non-dominated trade-off set | Provides deterministic ranking within the set |
| Multiple equivalent solutions | Single prioritized list |
| No weight specification needed | Explicit, documented weights |
| Difficult cross-study comparison | Reproducible comparison metric |

**Key insight:** Top ASCI-ranked catalysts are predominantly Pareto-optimal, validating both methodologies.

---

## Project Structure

```
ASCICat/
├── ascicat/                 # Core package
│   ├── calculator.py        # Main ASCICalculator class
│   ├── scoring.py           # Score computation
│   ├── visualizer.py        # Visualization tools
│   ├── analyzer.py          # Statistical analysis
│   └── sensitivity.py       # Sensitivity analysis
├── scripts/                 # CLI and GUI tools
├── examples/                # Usage examples
├── tutorials/               # Jupyter notebook tutorials
├── data/                    # Catalyst datasets
├── tests/                   # Test suite
└── logo/                    # Branding assets
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for:

- Development setup instructions
- Coding standards (NumPy docstrings, PEP 8)
- Testing requirements
- Pull request process

---

## References

ASCICat is built on established theoretical foundations:

- Nørskov, J. K. et al. *Towards the computational design of solid catalysts.* Nat. Chem. **1**, 37 (2009)
- Greeley, J. et al. *Computational high-throughput screening of electrocatalytic materials.* Nat. Mater. **5**, 909 (2006)
- Sabatier, P. *Hydrogénations et déshydrogénations par catalyse.* Ber. Dtsch. Chem. Ges. **44**, 1984 (1911)
- Oguz, I. C., Khossossi, N., Brunacci, M., Bucak, H. & Er, S. *ACS Catal.* **15**, 19461–19474 (2025)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

<table>
<tr>
<td><strong>Author</strong></td>
<td>Nabil Khossossi</td>
</tr>
<tr>
<td><strong>Email</strong></td>
<td><a href="mailto:n.khossossi@differ.nl">n.khossossi@differ.nl</a></td>
</tr>
<tr>
<td><strong>Institution</strong></td>
<td>Dutch Institute for Fundamental Energy Research (DIFFER)</td>
</tr>
<tr>
<td><strong>Issues</strong></td>
<td><a href="https://github.com/NabKh/ASCICat/issues">GitHub Issues</a></td>
</tr>
</table>

---

## Acknowledgments

This work was supported by the **Dutch Institute for Fundamental Energy Research (DIFFER)**.

---

<p align="center">
  <strong>ASCICat</strong> — Bridging computational databases and experimental priorities for accelerated catalyst discovery
</p>

<p align="center">
  <a href="#ascicat">Back to Top</a>
</p>
