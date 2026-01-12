# Tutorial: High-Quality Figure Generation

This tutorial demonstrates how to generate high-quality figures for your research.

## Figure Quality Standards

ASCICat generates figures meeting scientific standards:

| Specification | Value |
|:--------------|:------|
| Resolution | 600 DPI |
| Formats | PNG, PDF, SVG |
| Color scheme | Colorblind-friendly |
| Font | Sans-serif (Arial-like) |

## The Four-Panel Figure

The signature ASCICat figure consists of four panels:

```python
from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from pathlib import Path

# Setup
output_dir = Path('figures/output')
output_dir.mkdir(parents=True, exist_ok=True)

# Calculate ASCI
calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')
results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

# Initialize visualizer
viz = Visualizer(results, calc.config, auto_sample=True)

# Generate all panels
viz.generate_figures(
    output_dir=str(output_dir),
    dpi=600,
    formats=['png', 'pdf']
)
```

### Panel A: 3D ASCI Component Space

```python
fig = viz.plot_3d_component_space(
    highlight_top_n=10,
    figsize=(8, 7),
    elev=25,           # Elevation angle
    azim=45,           # Azimuth angle
    marker_size=50,
    alpha=0.6
)
fig.savefig(output_dir / 'panel_a.png', dpi=600, bbox_inches='tight')
fig.savefig(output_dir / 'panel_a.pdf', bbox_inches='tight')
```

### Panel B: Rank vs. Adsorption Energy

```python
fig = viz.plot_rank_vs_adsorption(
    show_optimal=True,      # Mark optimal energy
    show_window=True,       # Show activity window
    n_highlight=10,         # Label top catalysts
    figsize=(8, 6)
)
fig.savefig(output_dir / 'panel_b.png', dpi=600, bbox_inches='tight')
```

### Panel C: Volcano Optimization Landscape

```python
fig = viz.plot_volcano_optimization(
    n_contours=20,
    cmap='plasma',
    figsize=(8, 6),
    show_annotations=True
)
fig.savefig(output_dir / 'panel_c.png', dpi=600, bbox_inches='tight')
```

### Panel D: Top Performers Breakdown

```python
fig = viz.plot_top_performers(
    n_show=15,
    figsize=(10, 6),
    show_values=True
)
fig.savefig(output_dir / 'panel_d.png', dpi=600, bbox_inches='tight')
```

## Customizing Figures

### Color Schemes

```python
# Colorblind-friendly palette (default)
viz = Visualizer(results, calc.config, color_palette='colorblind')

# Alternative palettes
viz = Visualizer(results, calc.config, color_palette='deep')
viz = Visualizer(results, calc.config, color_palette='muted')
```

### Custom Colors

```python
import matplotlib.pyplot as plt

fig = viz.plot_top_performers(n_show=15)
ax = fig.axes[0]

# Customize colors for score components
# Activity = green, Stability = blue, Cost = orange
for patch in ax.patches[:15]:
    patch.set_facecolor('#2ca02c')  # Activity bars

plt.tight_layout()
fig.savefig('custom_colors.png', dpi=600)
```

### Font Sizes

```python
import matplotlib.pyplot as plt

# Set global font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

fig = viz.plot_asci_distribution()
fig.savefig('custom_fonts.png', dpi=600)
```

## Supplementary Figures

### ASCI Score Distribution

```python
fig = viz.plot_asci_distribution(
    bins=50,
    show_statistics=True,
    figsize=(8, 5)
)

# Add annotations
ax = fig.axes[0]
ax.axvline(results['ASCI'].median(), color='red',
           linestyle='--', label='Median')
ax.legend()

fig.savefig(output_dir / 'supp_distribution.png', dpi=600)
```

### Score Correlation Matrix

```python
fig = viz.plot_score_correlations(figsize=(8, 7))
fig.savefig(output_dir / 'supp_correlations.png', dpi=600)
```

### Pareto Frontier (2D)

```python
# Activity vs Cost trade-off
fig = viz.plot_pareto_2d(
    x_score='activity',
    y_score='cost',
    highlight_top_n=20,
    figsize=(8, 6)
)
fig.savefig(output_dir / 'supp_pareto_activity_cost.png', dpi=600)

# Activity vs Stability trade-off
fig = viz.plot_pareto_2d(
    x_score='activity',
    y_score='stability',
    highlight_top_n=20
)
fig.savefig(output_dir / 'supp_pareto_activity_stability.png', dpi=600)
```

### Radar Chart for Top Catalysts

```python
# Compare score profiles
top_symbols = results.head(5)['symbol'].tolist()

fig = viz.plot_radar_chart(
    catalyst_symbols=top_symbols,
    figsize=(8, 8)
)
fig.savefig(output_dir / 'supp_radar.png', dpi=600)
```

## Interactive Figures

For presentations or supplementary materials:

```python
# Interactive 3D plot (HTML)
viz.create_interactive_3d(
    output_path=str(output_dir / 'interactive_3d.html'),
    title='HER Catalyst Screening Results',
    highlight_top_n=20
)
```

## Figure Legends

### Suggested Panel Descriptions

```markdown
**Figure 1.** ASCI-based HER catalyst screening results.
**(A)** Three-dimensional visualization of catalyst scores in the
Activity-Stability-Cost space. Points are colored by ASCI score
(colorbar). Stars indicate the top 10 ranked catalysts.
**(B)** ASCI rank versus hydrogen adsorption energy. The red dashed
line marks the Sabatier optimal energy (−0.27 eV). The green shaded
region indicates the activity scoring window (±σ_a).
**(C)** Volcano optimization landscape showing ASCI contours over the
energy-cost parameter space. The optimal region combines near-thermoneutral
binding with low material cost.
**(D)** Score decomposition for the top 15 catalysts. Bar segments show
contributions from activity (green), stability (blue), and cost (orange)
components. Black markers indicate overall ASCI scores.
```

## Export Settings

### High Resolution Output

```python
# Single column width: 89 mm
# Double column width: 183 mm

fig = viz.plot_top_performers(n_show=10)
fig.set_size_inches(3.5, 3)  # Single column
fig.savefig('single_column.pdf', dpi=600, bbox_inches='tight')

fig = viz.plot_3d_component_space()
fig.set_size_inches(7.2, 5)  # Double column
fig.savefig('double_column.pdf', dpi=600, bbox_inches='tight')
```

### Standard Sizes

```python
# Single column: 3.25 inches
# Double column: 7 inches

fig = viz.plot_asci_distribution()
fig.set_size_inches(3.25, 2.5)
fig.savefig('standard_single.pdf', dpi=600, bbox_inches='tight')
```

## Complete Figure Script

```python
#!/usr/bin/env python3
"""Generate all figures for HER screening."""

from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from pathlib import Path
import matplotlib.pyplot as plt

# Configure matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0
})

def main():
    output = Path('figures/output')
    output.mkdir(parents=True, exist_ok=True)

    # Calculate
    calc = ASCICalculator(reaction='HER')
    calc.load_data('data/HER_clean.csv')
    results = calc.calculate_asci()

    # Visualize
    viz = Visualizer(results, calc.config)

    # Main figure panels
    viz.generate_figures(
        output_dir=str(output),
        dpi=600,
        formats=['png', 'pdf']
    )

    # Supplementary
    viz.plot_asci_distribution().savefig(
        output / 'supp_distribution.pdf', dpi=600)
    viz.plot_score_correlations().savefig(
        output / 'supp_correlations.pdf', dpi=600)

    # Interactive
    viz.create_interactive_3d(str(output / 'interactive.html'))

    print(f"Figures saved to: {output}/")

if __name__ == '__main__':
    main()
```
