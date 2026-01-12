# Tutorial: HER Catalyst Screening

This tutorial demonstrates a complete HER (Hydrogen Evolution Reaction) catalyst screening workflow using ASCICat.

## Objectives

By the end of this tutorial, you will be able to:

- Initialize an ASCICat calculator
- Load and validate catalyst data
- Calculate ASCI scores with custom weights
- Analyze and visualize results
- Export findings for further analysis

## Prerequisites

```bash
pip install ascicat
```

## Step 1: Setup

```python
from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from pathlib import Path
import pandas as pd

# Create output directory
output_dir = Path('results/HER_tutorial')
output_dir.mkdir(parents=True, exist_ok=True)

print("Setup complete!")
```

## Step 2: Initialize Calculator

```python
# Initialize for HER with verbose output
calc = ASCICalculator(
    reaction='HER',
    scoring_method='linear',
    verbose=True
)
```

**Expected output:**

```
================================================================================
ASCICat Calculator Initialized
================================================================================

🔬 Reaction: HER
   Pathway: H adsorption
   Optimal Energy: ΔE_opt = -0.270 eV
   Description: Hydrogen Evolution Reaction (2H⁺ + 2e⁻ → H₂)

⚙️  Configuration:
   Activity Width: σ_a = 0.150 eV
   Scoring Method: LINEAR
   Default Weights: w_a=0.33, w_s=0.33, w_c=0.34 (EQUAL)
================================================================================
```

## Step 3: Load Data

```python
# Load HER catalyst database
data_path = 'data/HER_clean.csv'
calc.load_data(data_path)

# Explore the data
print(f"\nDataset size: {len(calc.data):,} catalysts")
print(f"\nColumn names: {list(calc.data.columns)}")

# Check descriptor distributions
print("\nDescriptor Statistics:")
print(calc.data[['DFT_ads_E', 'surface_energy', 'Cost']].describe())
```

## Step 4: Calculate ASCI Scores

### Equal Weights (Unbiased Screening)

```python
# Calculate with equal weights
results = calc.calculate_asci(
    w_a=0.33,  # Activity
    w_s=0.33,  # Stability
    w_c=0.34   # Cost
)

print(f"\nASCI calculated for {len(results):,} catalysts")
print(f"ASCI range: [{results['ASCI'].min():.3f}, {results['ASCI'].max():.3f}]")
```

### Activity-Focused Screening

```python
# For fundamental research - prioritize activity
results_activity = calc.calculate_asci(
    w_a=0.50,
    w_s=0.30,
    w_c=0.20
)
```

### Cost-Focused Screening

```python
# For large-scale deployment - prioritize cost
results_cost = calc.calculate_asci(
    w_a=0.30,
    w_s=0.20,
    w_c=0.50
)
```

## Step 5: Analyze Top Catalysts

```python
# Get top 20 catalysts
top20 = calc.get_top_catalysts(n=20)

# Display detailed ranking
print("\n" + "="*85)
print("TOP 20 HER CATALYSTS (Equal Weights)")
print("="*85)
print(f"\n{'Rank':<6} {'Symbol':<14} {'ASCI':<8} {'S_a':<8} "
      f"{'S_s':<8} {'S_c':<8} {'ΔE (eV)':<10}")
print("-"*85)

for i, (_, row) in enumerate(top20.iterrows(), 1):
    print(f"{i:<6} {row['symbol']:<14} {row['ASCI']:.4f}   "
          f"{row['activity_score']:.4f}   {row['stability_score']:.4f}   "
          f"{row['cost_score']:.4f}   {row['DFT_ads_E']:+.3f}")
```

### Compare Across Weight Scenarios

```python
# Get top catalyst from each scenario
print("\n" + "="*60)
print("TOP CATALYST BY WEIGHT SCENARIO")
print("="*60)

scenarios = {
    'Equal weights (0.33/0.33/0.34)': results,
    'Activity-focused (0.50/0.30/0.20)': results_activity,
    'Cost-focused (0.30/0.20/0.50)': results_cost
}

for name, res in scenarios.items():
    top = res.iloc[0]
    print(f"\n{name}:")
    print(f"  {top['symbol']} (ASCI={top['ASCI']:.3f})")
    print(f"  Activity: {top['activity_score']:.3f}, "
          f"Stability: {top['stability_score']:.3f}, "
          f"Cost: {top['cost_score']:.3f}")
```

## Step 6: Generate Visualizations

```python
# Initialize visualizer
viz = Visualizer(results, calc.config, auto_sample=True)

# Generate figures
print("\nGenerating figures...")
viz.generate_figures(
    output_dir=str(output_dir),
    dpi=600,
    formats=['png', 'pdf']
)

# Generate interactive 3D plot
viz.create_interactive_3d(
    output_path=str(output_dir / 'interactive_3d.html'),
    title='HER Catalyst Screening Results'
)

print(f"\nFigures saved to: {output_dir}/")
```

## Step 7: Export Results

```python
# Save full results
results.to_csv(output_dir / 'HER_full_results.csv', index=False)

# Save top 100 catalysts
top100 = calc.get_top_catalysts(n=100)
top100.to_csv(output_dir / 'HER_top100.csv', index=False)

# Save summary statistics
summary = {
    'n_catalysts': len(results),
    'best_catalyst': results.iloc[0]['symbol'],
    'best_asci': results.iloc[0]['ASCI'],
    'mean_asci': results['ASCI'].mean(),
    'std_asci': results['ASCI'].std(),
    'weights': '0.33/0.33/0.34 (equal)'
}

pd.Series(summary).to_csv(output_dir / 'HER_summary.csv')

print("\nExported files:")
for f in output_dir.glob('*'):
    print(f"  {f.name}")
```

## Step 8: Interpret Results

### Understanding the Top Catalyst

```python
# Detailed analysis of #1 catalyst
top1 = results.iloc[0]

print("\n" + "="*60)
print(f"DETAILED ANALYSIS: {top1['symbol']}")
print("="*60)

print(f"\n📊 ASCI Score: {top1['ASCI']:.4f}")
print(f"\n🎯 Score Breakdown:")
print(f"   Activity (w=0.33):  {top1['activity_score']:.4f} "
      f"→ contributes {0.33*top1['activity_score']:.4f}")
print(f"   Stability (w=0.33): {top1['stability_score']:.4f} "
      f"→ contributes {0.33*top1['stability_score']:.4f}")
print(f"   Cost (w=0.34):      {top1['cost_score']:.4f} "
      f"→ contributes {0.34*top1['cost_score']:.4f}")

print(f"\n📏 Raw Values:")
print(f"   ΔE_H = {top1['DFT_ads_E']:+.3f} eV "
      f"(optimal: -0.27 eV, deviation: {abs(top1['DFT_ads_E']+0.27):.3f} eV)")
print(f"   γ = {top1['surface_energy']:.2f} J/m²")
print(f"   Cost = ${top1['Cost']:,.0f}/kg")

# Identify limiting factor
scores = {
    'Activity': top1['activity_score'],
    'Stability': top1['stability_score'],
    'Cost': top1['cost_score']
}
limiting = min(scores, key=scores.get)
print(f"\n⚠️  Limiting factor: {limiting} ({scores[limiting]:.3f})")
```

## Complete Script

Here's the complete tutorial as a single script:

```python
#!/usr/bin/env python3
"""
HER Catalyst Screening Tutorial
Complete workflow with ASCICat
"""

from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from pathlib import Path

def main():
    # Setup
    output_dir = Path('results/HER_tutorial')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    calc = ASCICalculator(reaction='HER', verbose=True)

    # Load data
    calc.load_data('data/HER_clean.csv')

    # Calculate ASCI (equal weights)
    results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

    # Display top catalysts
    print("\nTOP 10 HER CATALYSTS:")
    print("-" * 50)
    top10 = calc.get_top_catalysts(n=10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:2}. {row['symbol']:<12} ASCI={row['ASCI']:.3f}")

    # Generate figures
    viz = Visualizer(results, calc.config)
    viz.generate_all_outputs(output_dir=str(output_dir))

    # Export
    results.to_csv(output_dir / 'HER_results.csv', index=False)

    print(f"\n✓ Tutorial complete! Results in: {output_dir}/")

if __name__ == '__main__':
    main()
```

## Next Steps

- [CO2RR Multi-Pathway Analysis](co2rr-analysis.md) - Screen multiple reaction pathways
- [Sensitivity Analysis](sensitivity-analysis.md) - Evaluate ranking robustness
- [Figure Generation](figure-generation.md) - Create high-quality graphics
