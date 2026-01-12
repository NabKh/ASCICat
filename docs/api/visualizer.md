# Visualizer

High-quality figure generation.

## Class Reference

::: ascicat.visualizer.Visualizer
    options:
      show_root_heading: true
      show_source: false

## Quick Reference

```python
from ascicat.visualizer import Visualizer

viz = Visualizer(results, config, auto_sample=True)

# Generate all figures
viz.generate_all_outputs(output_dir='figures/')

# Individual plots
fig = viz.plot_3d_component_space()
fig = viz.plot_rank_vs_adsorption()
fig = viz.plot_volcano_optimization()
fig = viz.plot_top_performers()
```

## Initialization

```python
viz = Visualizer(
    results,           # DataFrame from calculate_asci()
    config,            # ReactionConfig from calculator
    auto_sample=True,  # Sample large datasets
    default_dpi=600,   # Output resolution
    color_palette='colorblind'  # Color scheme
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `results` | DataFrame | Required | ASCI results |
| `config` | ReactionConfig | Required | Reaction configuration |
| `auto_sample` | bool | True | Sample datasets > 5000 |
| `default_dpi` | int | 600 | Default resolution |
| `color_palette` | str | 'colorblind' | Seaborn palette |

## Methods

### generate_all_outputs

Generate complete figure set.

```python
viz.generate_all_outputs(
    output_dir='figures/',
    dpi=600,
    formats=['png', 'pdf'],
    include_interactive=True
)
```

### generate_figures

Generate four-panel figure.

```python
viz.generate_figures(
    output_dir='figures/',
    dpi=600,
    formats=['png', 'pdf']
)
```

### plot_3d_component_space

3D visualization of score space.

```python
fig = viz.plot_3d_component_space(
    highlight_top_n=10,
    figsize=(8, 7),
    elev=25,
    azim=45
)
```

### plot_rank_vs_adsorption

Activity volcano-style plot.

```python
fig = viz.plot_rank_vs_adsorption(
    show_optimal=True,
    show_window=True,
    figsize=(8, 6)
)
```

### plot_volcano_optimization

ASCI contour landscape.

```python
fig = viz.plot_volcano_optimization(
    n_contours=20,
    cmap='plasma',
    figsize=(8, 6)
)
```

### plot_top_performers

Score breakdown bar chart.

```python
fig = viz.plot_top_performers(
    n_show=15,
    figsize=(10, 6)
)
```

### plot_asci_distribution

ASCI score histogram.

```python
fig = viz.plot_asci_distribution(
    bins=50,
    show_statistics=True
)
```

### plot_score_correlations

Correlation matrix heatmap.

```python
fig = viz.plot_score_correlations(figsize=(8, 7))
```

### plot_pareto_2d

2D Pareto frontier projection.

```python
fig = viz.plot_pareto_2d(
    x_score='activity',
    y_score='cost',
    highlight_top_n=20
)
```

### plot_radar_chart

Spider/radar chart for catalyst comparison.

```python
fig = viz.plot_radar_chart(
    catalyst_symbols=['Pt111', 'Fe2Sb4', 'Cu3Sb']
)
```

### create_interactive_3d

Interactive Plotly 3D plot.

```python
viz.create_interactive_3d(
    output_path='interactive.html',
    title='Catalyst Screening'
)
```

## Output Formats

All static plots support:

- PNG (raster, default 600 DPI)
- PDF (vector)
- SVG (vector)
- EPS (vector)

```python
fig.savefig('plot.png', dpi=600, bbox_inches='tight')
fig.savefig('plot.pdf', bbox_inches='tight')
fig.savefig('plot.svg', bbox_inches='tight')
```
