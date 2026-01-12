# First Analysis: HER Catalyst Screening

This tutorial walks you through a complete HER (Hydrogen Evolution Reaction) catalyst screening analysis using ASCICat.

## Objective

By the end of this tutorial, you will:

1. Understand how ASCICat processes catalyst data
2. Calculate ASCI scores for thousands of catalysts
3. Generate high-quality figures
4. Interpret the results

## Prerequisites

- ASCICat installed (`pip install ascicat`)
- Sample data file (`data/HER_clean.csv`)

## The Science Behind HER

The Hydrogen Evolution Reaction is a key process in water electrolysis:

$$2H^+ + 2e^- \rightarrow H_2$$

According to the **Sabatier principle**, the optimal catalyst binds hydrogen neither too strongly nor too weakly:

- **Optimal binding energy**: $\Delta E_{opt} = -0.27$ eV
- **Activity width**: $\sigma_a = 0.15$ eV

!!! note "Sabatier Volcano"

    Catalysts at the peak of the volcano plot (near -0.27 eV) show maximum activity. This is where platinum-group metals naturally fall, explaining their excellent HER performance.

## Step-by-Step Analysis

### 1. Setup and Initialization

```python
from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from pathlib import Path

# Create output directory
output_dir = Path('results/HER_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize calculator
calc = ASCICalculator(
    reaction='HER',
    scoring_method='linear',  # or 'gaussian'
    verbose=True
)
```

**What happens here:**

- ASCICat loads the HER configuration with $\Delta E_{opt} = -0.27$ eV
- Linear scoring method is selected (simpler, more interpretable)
- Verbose mode shows detailed progress

### 2. Load and Validate Data

```python
# Load catalyst database
data_path = 'data/HER_clean.csv'
calc.load_data(data_path)

# View data summary
print(f"\nDataset Overview:")
print(f"  Total catalysts: {len(calc.data):,}")
print(f"  Adsorption energy range: [{calc.data['DFT_ads_E'].min():.2f}, "
      f"{calc.data['DFT_ads_E'].max():.2f}] eV")
print(f"  Surface energy range: [{calc.data['surface_energy'].min():.2f}, "
      f"{calc.data['surface_energy'].max():.2f}] J/m²")
print(f"  Cost range: [${calc.data['Cost'].min():.2f}, "
      f"${calc.data['Cost'].max():,.0f}]")
```

**Expected output:**

```
📂 Loading HER data from: data/HER_clean.csv

📊 Data Summary:
   Total catalysts: 48,312

   Activity Descriptor (ΔE):
      Range: [-2.500, +1.200] eV
      Mean:  -0.450 eV
      Optimal for HER: -0.270 eV

   Stability Descriptor (γ):
      Range: [0.520, 4.500] J/m²
      Mean:  1.850 J/m²

   Cost Descriptor:
      Range: [$2.67, $107,544] per kg
      Median: $450 per kg
```

### 3. Calculate ASCI Scores

```python
# Calculate with equal weights (unbiased screening)
results = calc.calculate_asci(
    w_a=0.33,  # Activity weight
    w_s=0.33,  # Stability weight
    w_c=0.34   # Cost weight
)

# View statistics
stats = calc.get_statistics()
print(f"\nASCI Statistics:")
print(f"  Mean ASCI:   {stats['asci']['mean']:.4f}")
print(f"  Std ASCI:    {stats['asci']['std']:.4f}")
print(f"  Max ASCI:    {stats['asci']['max']:.4f}")
print(f"  Min ASCI:    {stats['asci']['min']:.4f}")
```

### 4. Examine Top Catalysts

```python
# Get top 20 catalysts
top20 = calc.get_top_catalysts(n=20)

# Display detailed ranking table
print("\n" + "="*80)
print("TOP 20 HER CATALYSTS (Equal Weights)")
print("="*80)
print(f"\n{'Rank':<6} {'Catalyst':<14} {'ASCI':<8} {'S_a':<8} "
      f"{'S_s':<8} {'S_c':<8} {'ΔE (eV)':<10}")
print("-"*80)

for i, (_, row) in enumerate(top20.iterrows(), 1):
    print(f"{i:<6} {row['symbol']:<14} {row['ASCI']:.4f}   "
          f"{row['activity_score']:.4f}   {row['stability_score']:.4f}   "
          f"{row['cost_score']:.4f}   {row['DFT_ads_E']:+.3f}")
```

**Example output:**

```
================================================================================
TOP 20 HER CATALYSTS (Equal Weights)
================================================================================

Rank   Catalyst       ASCI     S_a      S_s      S_c      ΔE (eV)
--------------------------------------------------------------------------------
1      Fe2Sb4         0.9234   0.9156   0.8912   0.9632   -0.257
2      Cu3Sb          0.9087   0.8967   0.8845   0.9448   -0.285
3      Cu6Sb2         0.8956   0.8823   0.8756   0.9289   -0.298
...
```

### 5. Generate Visualizations

```python
# Initialize visualizer (auto-samples large datasets)
viz = Visualizer(results, calc.config, auto_sample=True)

# Generate all figures
viz.generate_figures(
    output_dir=str(output_dir),
    dpi=600,
    formats=['png', 'pdf']
)

# Generate interactive 3D visualization
viz.create_interactive_3d(
    output_path=str(output_dir / 'interactive_3d.html')
)

print(f"\nFigures saved to: {output_dir}/")
```

**Generated files:**

```
results/HER_analysis/
├── panel_a_3d_pareto.png
├── panel_a_3d_pareto.pdf
├── panel_b_rank_vs_adsorption.png
├── panel_b_rank_vs_adsorption.pdf
├── panel_c_volcano_optimization.png
├── panel_c_volcano_optimization.pdf
├── panel_d_top_performers.png
├── panel_d_top_performers.pdf
└── interactive_3d.html
```

### 6. Export Results

```python
# Save full results
results.to_csv(output_dir / 'HER_full_results.csv', index=False)

# Save top 100
top100 = calc.get_top_catalysts(n=100)
top100.to_csv(output_dir / 'HER_top100.csv', index=False)

# Print summary
calc.print_summary(n_top=10)
```

## Understanding the Figures

### Panel A: 3D ASCI Component Space

Shows all catalysts in the three-dimensional score space:

- **X-axis**: Activity score $S_a$
- **Y-axis**: Stability score $S_s$
- **Z-axis**: Cost score $S_c$
- **Color**: Overall ASCI score
- **Stars**: Top 10 catalysts

!!! tip "Interpretation"

    Catalysts in the upper-right-back corner (high scores on all axes) are the best performers. The ideal catalyst would be at position (1, 1, 1).

### Panel B: Rank vs. Adsorption Energy

Shows the relationship between ASCI rank and binding energy:

- **Red dashed line**: Optimal energy ($\Delta E_{opt} = -0.27$ eV)
- **Green shading**: Activity window
- **Trend line**: Quadratic fit showing volcano behavior

### Panel C: Volcano Optimization Landscape

Contour plot showing ASCI scores over the energy-cost space:

- **X-axis**: Adsorption energy $\Delta E$
- **Y-axis**: Log cost
- **Contours**: ASCI iso-lines

### Panel D: Top Performers Breakdown

Bar chart showing score components for the top catalysts:

- **Green bars**: Activity scores
- **Blue bars**: Stability scores
- **Orange bars**: Cost scores
- **Black markers**: Overall ASCI

## Weight Sensitivity

Try different weight scenarios to see how rankings change:

```python
# Activity-focused screening
results_activity = calc.calculate_asci(w_a=0.5, w_s=0.3, w_c=0.2)

# Cost-focused screening
results_cost = calc.calculate_asci(w_a=0.3, w_s=0.2, w_c=0.5)

# Compare top catalysts
print("\nTop catalyst by scenario:")
print(f"  Equal weights:    {calc.get_top_catalysts(1).iloc[0]['symbol']}")
print(f"  Activity-focused: {results_activity.iloc[0]['symbol']}")
print(f"  Cost-focused:     {results_cost.iloc[0]['symbol']}")
```

## Complete Script

Here's the complete analysis script:

```python
#!/usr/bin/env python3
"""Complete HER Catalyst Screening Analysis with ASCICat"""

from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from pathlib import Path

def main():
    # Setup
    output_dir = Path('results/HER_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    calc = ASCICalculator(reaction='HER', verbose=True)

    # Load data
    calc.load_data('data/HER_clean.csv')

    # Calculate ASCI (equal weights)
    results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

    # Display results
    calc.print_summary(n_top=20)

    # Generate figures
    viz = Visualizer(results, calc.config)
    viz.generate_all_outputs(output_dir=str(output_dir))

    # Export
    results.to_csv(output_dir / 'HER_results.csv', index=False)

    print(f"\n✓ Analysis complete! Results in: {output_dir}/")

if __name__ == '__main__':
    main()
```

## Next Steps

- [CO2RR Analysis Tutorial](../tutorials/co2rr-analysis.md) - Multi-pathway screening
- [Sensitivity Analysis](../tutorials/sensitivity-analysis.md) - Weight dependency study
- [User Guide](../user-guide/index.md) - Detailed documentation
