# ASCICat Tutorials

This folder contains Jupyter notebook tutorials demonstrating complete workflows from data loading through figure generation for the ASCICat framework.

## Tutorial Overview

| Tutorial | Description | Topics Covered |
|----------|-------------|----------------|
| **01_Introduction_Quick_Start.ipynb** | Introduction to ASCICat and basic workflow | ASCI framework, installation, quick-start example |
| **02_HER_Catalyst_Screening.ipynb** | Complete HER analysis workflow | Data exploration, ASCI calculation, Pareto analysis, figure generation |
| **03_CO2RR_Multi_Pathway_Analysis.ipynb** | Multi-pathway CO2RR screening | CO/CHO/COCOH pathways, cross-pathway comparison, versatile catalysts |
| **04_Visualization_Guide.ipynb** | Advanced visualization techniques | 3D plots, volcano plots, radar charts, high-resolution exports |
| **05_Sensitivity_Analysis.ipynb** | Weight sensitivity and robustness analysis | Weight sweep, ternary diagrams, rank correlation, robustness metrics |

## Getting Started

1. Ensure ASCICat is installed or run from the repository root
2. Start with Tutorial 01 for an introduction
3. Progress through tutorials 02-05 for advanced features

## Requirements

- Python 3.8+
- ASCICat package
- Jupyter Notebook or JupyterLab
- Standard scientific Python libraries (numpy, pandas, matplotlib, seaborn)

## Running the Tutorials

```bash
# From the tutorials folder
jupyter notebook
```

Or open individual notebooks in VS Code, JupyterLab, or any Jupyter-compatible environment.

## Data Files

Tutorials use data from the `../data/` directory:
- `HER_clean.csv` - Hydrogen Evolution Reaction catalyst data
- `CO2RR_CO_clean.csv` - CO2RR CO pathway data
- `CO2RR_CHO_clean.csv` - CO2RR CHO pathway data
- `CO2RR_COCOH_clean.csv` - CO2RR COCOH pathway data

## Output

Tutorial outputs are saved to `../results/tutorial_*/` directories.
