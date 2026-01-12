# Changelog

All notable changes to ASCICat will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-03

### Added

#### Core Framework
- **ASCICalculator**: Main calculation engine for Activity-Stability-Cost Index
- **Multi-reaction support**: HER and CO2RR (CO, CHO, COCOH pathways)
- **Flexible scoring**: Linear and Gaussian activity scoring methods
- **Data-driven normalization**: Automatic min/max detection for stability and cost scores
- **Weight validation**: Ensures weights sum to 1.0 with tolerance

#### Sensitivity Analysis (New)
- **SensitivityAnalyzer**: Comprehensive weight sensitivity analysis
  - Weight grid generation (simplex, full simplex, Standard)
  - Bootstrap confidence intervals for rankings
  - Variance-based sensitivity indices (Sobol-like)
  - Spearman rank correlations between weight scenarios
  - Friedman statistical tests for ranking significance
  - Kendall's W concordance coefficient
- **SensitivityVisualizer**: High-resolution sensitivity plots
  - Ternary diagram heatmaps (proper 3-weight representation)
  - Rank confidence interval plots
  - Sensitivity index bar charts
  - Correlation vs distance analysis
  - Robustness quadrant plots
  - Comprehensive 4-panel summary figures

#### Visualization
- **Visualizer class**: High-resolution figures (600 DPI)
  - Enhanced volcano plots with ASCI contours
  - 3D Pareto surface visualization
  - Score distribution histograms
  - Correlation matrices
  - Radar/spider charts for catalyst comparison
  - 2D Pareto front analysis
  - Ternary sensitivity diagrams
- **Colorblind-friendly palettes**
- **Multiple output formats**: PNG, PDF, SVG

#### Analysis Tools
- **Analyzer class**: Advanced analysis capabilities
  - Pareto frontier identification
  - Statistical analysis
  - Ranking robustness metrics

#### User Interfaces
- **Command-line interface (CLI)**: `ascicat` command with full functionality
- **Graphical interface (GUI)**: PyQt5-based interactive application
- **Batch processing**: `ascicat-batch` for high-throughput screening

#### Data Handling
- **DataLoader**: Robust data loading with validation
  - Automatic column detection
  - Missing value handling
  - Outlier detection
  - Physical validity checks

#### Utilities
- **Periodic table data**: Element costs and properties
- **Format functions**: Catalyst naming conventions
- **Export functions**: JSON, CSV output

### Documentation
- Comprehensive README with examples
- NumPy-style docstrings throughout
- Example scripts (6 complete examples)
- CONTRIBUTING guidelines
- MIT License

### Scientific Framework
- **Activity scoring**: Based on Sabatier principle
  - Linear: S_a = max(0, 1 - |ΔE - ΔE_opt| / σ)
  - Gaussian: S_a = exp(-(ΔE - ΔE_opt)² / (2σ²))
- **Stability scoring**: Inverse surface energy normalization
  - S_s = (γ_max - γ) / (γ_max - γ_min)
- **Cost scoring**: Logarithmic normalization
  - S_c = (log C_max - log C) / (log C_max - log C_min)
- **Combined ASCI**: Weighted linear combination
  - φ_ASCI = w_a·S_a + w_s·S_s + w_c·S_c

### Data
- HER catalyst database (~1000 entries)
- CO2RR pathway databases (CO, CHO, COCOH)
- Data format documentation

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2025-01-03 | Initial public release |

---

## Future Roadmap

### Planned for v1.1.0
- [ ] Machine learning integration for descriptor prediction
- [ ] Uncertainty quantification for DFT data
- [ ] Additional reaction pathways (OER, ORR)
- [ ] Interactive Plotly visualizations

### Planned for v1.2.0
- [ ] Automatic descriptor extraction from computational databases
- [ ] Integration with Materials Project and OQMD
- [ ] Jupyter notebook widgets

---

## Citation

If you use ASCICat in your research, please cite:

```bibtex
@article{khossossi2025ascicat,
  title={{ASCICat: Activity-Stability-Cost Integrated Framework for
         Electrocatalyst Discovery}},
  author={Khossossi, N.},
  journal={Journal of Computational Chemistry},
  volume={XX},
  pages={XXX--XXX},
  year={2025},
  doi={10.xxxx/xxxxxx}
}
```
