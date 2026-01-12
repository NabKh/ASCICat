# Visualization

ASCICat provides comprehensive visualization tools for high-quality figures.

## Overview

The `Visualizer` class generates:

- **Static figures** (PNG, PDF) at 600 DPI
- **Interactive 3D plots** (HTML with Plotly)
- **Multi-panel figures** for analysis
- **Colorblind-friendly** palettes

## Quick Start

```python
from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer

# Calculate ASCI
calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')
results = calc.calculate_asci()

# Generate all figures
viz = Visualizer(results, calc.config)
viz.generate_all_outputs(output_dir='figures/')
```

## Four-Panel Figure

The signature visualization is a four-panel figure:

```python
viz.generate_figures(
    output_dir='figures/',
    dpi=600,
    formats=['png', 'pdf']
)
```

### Panel A: 3D ASCI Space

Shows catalysts in 3D score space:

- **X**: Activity score
- **Y**: Stability score
- **Z**: Cost score
- **Color**: ASCI value
- **Stars**: Top 10 catalysts

```python
fig = viz.plot_3d_component_space(
    highlight_top_n=10,
    figsize=(8, 7),
    elev=25,
    azim=45
)
fig.savefig('panel_a.png', dpi=600)
```

### Panel B: Rank vs. Adsorption Energy

Volcano-style plot showing activity relationship:

```python
fig = viz.plot_rank_vs_adsorption(
    show_optimal=True,
    show_window=True,
    figsize=(8, 6)
)
fig.savefig('panel_b.png', dpi=600)
```

### Panel C: Optimization Landscape

Contour plot of ASCI over energy-cost space:

```python
fig = viz.plot_volcano_optimization(
    n_contours=20,
    cmap='plasma',
    figsize=(8, 6)
)
fig.savefig('panel_c.png', dpi=600)
```

### Panel D: Top Performers

Bar chart of score components:

```python
fig = viz.plot_top_performers(
    n_show=15,
    figsize=(10, 6)
)
fig.savefig('panel_d.png', dpi=600)
```

## Interactive 3D Visualization

Create interactive HTML plots:

```python
viz.create_interactive_3d(
    output_path='interactive.html',
    title='HER Catalyst Screening',
    highlight_top_n=20
)
```

Features:

- Rotate, zoom, pan
- Hover for catalyst details
- Click to highlight
- Export to PNG

## Individual Plots

### ASCI Distribution

```python
fig = viz.plot_asci_distribution(
    bins=50,
    show_statistics=True,
    figsize=(8, 5)
)
```

### Score Correlations

```python
fig = viz.plot_score_correlations(
    figsize=(10, 8)
)
```

### Pareto Frontier (2D)

```python
fig = viz.plot_pareto_2d(
    x_score='activity',
    y_score='cost',
    highlight_top_n=20
)
```

### Radar/Spider Plot

```python
fig = viz.plot_radar_chart(
    catalyst_symbols=['Pt111', 'Fe2Sb4', 'Cu3Sb'],
    figsize=(8, 8)
)
```

## Customization

### Figure Settings

```python
viz = Visualizer(
    results,
    config,
    # Figure defaults
    default_dpi=600,
    default_figsize=(10, 8),
    # Style
    color_palette='colorblind',  # or 'deep', 'muted', etc.
    font_scale=1.2
)
```

### Color Maps

```python
# Available colormaps
viz.plot_3d_component_space(cmap='viridis')  # Default
viz.plot_3d_component_space(cmap='plasma')
viz.plot_3d_component_space(cmap='RdYlGn')
viz.plot_3d_component_space(cmap='coolwarm')
```

### Custom Colors for Categories

```python
from ascicat.config import ASCIConstants

# Use built-in colorblind-safe palette
colors = ASCIConstants.CATEGORICAL_COLORS

# Activity = green, Stability = blue, Cost = orange
viz.plot_top_performers(
    activity_color='#2ca02c',
    stability_color='#1f77b4',
    cost_color='#ff7f0e'
)
```

## Large Dataset Handling

For datasets with >5000 catalysts, automatic stratified sampling is applied:

```python
viz = Visualizer(results, config, auto_sample=True)

# Or explicit sampling
viz_sampled = Visualizer(
    results.sample(n=2000, random_state=42),
    config
)
```

!!! note "Sampling Strategy"

    ASCICat uses stratified sampling based on ASCI quartiles to preserve the score distribution in visualizations.

## Export Formats

```python
# Multiple formats
viz.generate_figures(
    output_dir='figures/',
    formats=['png', 'pdf', 'svg']
)

# Individual figure
fig = viz.plot_asci_distribution()
fig.savefig('dist.png', dpi=600, bbox_inches='tight')
fig.savefig('dist.pdf', dpi=600, bbox_inches='tight')
fig.savefig('dist.svg', bbox_inches='tight')
```

## Matplotlib Integration

All plot methods return matplotlib `Figure` objects:

```python
import matplotlib.pyplot as plt

fig = viz.plot_rank_vs_adsorption()

# Customize further
ax = fig.axes[0]
ax.set_title('My Custom Title', fontsize=16)
ax.axhline(y=100, color='red', linestyle='--')

# Add annotation
ax.annotate('Interesting point', xy=(-0.27, 1),
            xytext=(-0.1, 10), fontsize=10,
            arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
fig.savefig('customized.png', dpi=600)
```

## Sensitivity Visualization

For sensitivity analysis results:

```python
from ascicat.sensitivity import SensitivityVisualizer

sens_viz = SensitivityVisualizer(output_dir='sensitivity/')

# Ternary diagram
sens_viz.plot_ternary_heatmap(sensitivity_results)

# Confidence intervals
sens_viz.plot_rank_confidence_intervals(rank_stats)

# Comprehensive summary
sens_viz.plot_comprehensive_summary(
    sensitivity_results,
    rank_stats,
    indices
)
```

## Figure Gallery

### Standard Outputs

| File | Description |
|:-----|:------------|
| `panel_a_3d_pareto.png` | 3D ASCI component space |
| `panel_b_rank_vs_adsorption.png` | Activity relationship |
| `panel_c_volcano_optimization.png` | Optimization landscape |
| `panel_d_top_performers.png` | Top catalyst breakdown |
| `interactive_3d.html` | Interactive explorer |

### Sensitivity Outputs

| File | Description |
|:-----|:------------|
| `ternary_sensitivity.png` | Weight space heatmap |
| `rank_confidence_intervals.png` | Bootstrap CIs |
| `sensitivity_indices.png` | Weight importance |
| `sensitivity_summary.png` | 4-panel summary |

## Best Practices

!!! tip "High Quality Output"

    - Use 600 DPI for print
    - Save as PDF for vector graphics
    - Use colorblind-friendly palettes
    - Include scale bars and legends

!!! tip "Large Datasets"

    - Enable auto-sampling for >5000 points
    - Use interactive plots for exploration
    - Static plots for final figures

!!! tip "Customization"

    - All plots return matplotlib figures
    - Modify axes, labels, annotations as needed
    - Chain with matplotlib functions
