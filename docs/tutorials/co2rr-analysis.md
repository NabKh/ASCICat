# Tutorial: CO2RR Multi-Pathway Analysis

This tutorial demonstrates screening catalysts for CO₂ reduction across multiple product pathways.

## Objectives

- Screen catalysts for CO, CHO, and COCOH pathways
- Compare top candidates across pathways
- Identify versatile catalysts
- Generate comparative visualizations

## The CO2RR Challenge

CO₂ reduction can produce various products:

| Pathway | Product | Optimal $\Delta E$ |
|:--------|:--------|:-------------------|
| CO | Carbon monoxide | -0.67 eV |
| CHO | Methanol precursor | -0.48 eV |
| COCOH | Formate precursor | -0.32 eV |

Different catalysts favor different pathways based on their binding characteristics.

## Complete Workflow

```python
from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Setup
output_dir = Path('results/CO2RR_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Define pathways
pathways = {
    'CO': {
        'data': 'data/CO2RR_CO_clean.csv',
        'optimal': -0.67,
        'description': 'CO production'
    },
    'CHO': {
        'data': 'data/CO2RR_CHO_clean.csv',
        'optimal': -0.48,
        'description': 'Methanol pathway'
    },
    'COCOH': {
        'data': 'data/CO2RR_COCOH_clean.csv',
        'optimal': -0.32,
        'description': 'Formate pathway'
    }
}

# Store results
all_results = {}
all_top = {}

# Screen each pathway
for pathway, info in pathways.items():
    print(f"\n{'='*60}")
    print(f"Screening {pathway} Pathway: {info['description']}")
    print(f"Optimal ΔE: {info['optimal']} eV")
    print('='*60)

    # Initialize calculator
    calc = ASCICalculator(
        reaction='CO2RR',
        pathway=pathway,
        verbose=False
    )

    # Load pathway-specific data
    calc.load_data(info['data'])
    print(f"Loaded {len(calc.data):,} catalysts")

    # Calculate ASCI with equal weights
    results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

    # Store results
    all_results[pathway] = results
    all_top[pathway] = calc.get_top_catalysts(n=20)

    # Display top 5
    print(f"\nTop 5 for {pathway}:")
    for i, (_, row) in enumerate(all_top[pathway].head(5).iterrows(), 1):
        print(f"  {i}. {row['symbol']:<12} ASCI={row['ASCI']:.3f} "
              f"(ΔE={row['DFT_ads_E']:+.2f} eV)")

    # Generate pathway-specific figures
    viz = Visualizer(results, calc.config)
    viz.generate_figures(
        output_dir=str(output_dir / pathway),
        dpi=600
    )
```

## Cross-Pathway Comparison

```python
print("\n" + "="*70)
print("CROSS-PATHWAY COMPARISON")
print("="*70)

# Find catalysts that appear in top 20 for multiple pathways
from collections import Counter

all_symbols = []
for pathway, top in all_top.items():
    all_symbols.extend(top['symbol'].tolist())

symbol_counts = Counter(all_symbols)
versatile = {s: c for s, c in symbol_counts.items() if c > 1}

print(f"\nVersatile catalysts (appear in top 20 for multiple pathways):")
for symbol, count in sorted(versatile.items(), key=lambda x: -x[1]):
    pathways_in = []
    for pathway, top in all_top.items():
        if symbol in top['symbol'].values:
            rank = top[top['symbol'] == symbol].index[0] + 1
            pathways_in.append(f"{pathway}(#{rank})")
    print(f"  {symbol}: {count} pathways - {', '.join(pathways_in)}")
```

## Comparative Visualization

```python
# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (pathway, results) in zip(axes, all_results.items()):
    top10 = results.head(10)

    # Bar chart of ASCI scores
    colors = plt.cm.viridis(top10['ASCI'].values / top10['ASCI'].max())
    bars = ax.barh(range(10), top10['ASCI'], color=colors)

    ax.set_yticks(range(10))
    ax.set_yticklabels(top10['symbol'], fontsize=9)
    ax.set_xlabel('ASCI Score')
    ax.set_title(f'{pathway} Pathway', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Add optimal energy annotation
    optimal = pathways[pathway]['optimal']
    ax.text(0.95, 0.05, f'ΔE_opt = {optimal} eV',
            transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
fig.savefig(output_dir / 'pathway_comparison.png', dpi=600, bbox_inches='tight')
plt.close()

print(f"\nComparison figure saved: {output_dir}/pathway_comparison.png")
```

## Identify Pathway-Specific Champions

```python
print("\n" + "="*70)
print("PATHWAY-SPECIFIC CHAMPIONS")
print("="*70)

# Best catalyst for each pathway
for pathway, top in all_top.items():
    best = top.iloc[0]
    print(f"\n{pathway} Champion: {best['symbol']}")
    print(f"  ASCI: {best['ASCI']:.4f}")
    print(f"  Activity: {best['activity_score']:.3f}")
    print(f"  Stability: {best['stability_score']:.3f}")
    print(f"  Cost: {best['cost_score']:.3f}")
    print(f"  ΔE: {best['DFT_ads_E']:+.3f} eV")
```

## Export Combined Results

```python
# Create summary DataFrame
summary_data = []

for pathway, top in all_top.items():
    for rank, (_, row) in enumerate(top.head(10).iterrows(), 1):
        summary_data.append({
            'pathway': pathway,
            'rank': rank,
            'symbol': row['symbol'],
            'ASCI': row['ASCI'],
            'activity_score': row['activity_score'],
            'stability_score': row['stability_score'],
            'cost_score': row['cost_score'],
            'DFT_ads_E': row['DFT_ads_E']
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_dir / 'CO2RR_pathway_summary.csv', index=False)

# Pivot table for easy comparison
pivot = summary_df.pivot_table(
    index='symbol',
    columns='pathway',
    values='ASCI',
    aggfunc='first'
)
pivot.to_csv(output_dir / 'CO2RR_pivot_comparison.csv')

print(f"\nResults exported to: {output_dir}/")
```

## Weight Sensitivity Across Pathways

```python
# Compare how weight changes affect rankings across pathways

weight_scenarios = {
    'Equal': (0.33, 0.33, 0.34),
    'Activity-focused': (0.50, 0.30, 0.20),
    'Cost-focused': (0.30, 0.20, 0.50)
}

print("\n" + "="*70)
print("WEIGHT SENSITIVITY COMPARISON")
print("="*70)

for pathway in pathways:
    print(f"\n{pathway} Pathway - Top catalyst by scenario:")

    calc = ASCICalculator(reaction='CO2RR', pathway=pathway, verbose=False)
    calc.load_data(pathways[pathway]['data'])

    for scenario, (w_a, w_s, w_c) in weight_scenarios.items():
        results = calc.calculate_asci(w_a=w_a, w_s=w_s, w_c=w_c)
        top = results.iloc[0]
        print(f"  {scenario:18}: {top['symbol']:<12} (ASCI={top['ASCI']:.3f})")
```

## Key Findings Summary

```python
print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)

print("""
1. PATHWAY-SPECIFIC OPTIMIZATION
   - Different pathways have different optimal catalysts
   - Activity score varies significantly with pathway choice

2. VERSATILE CATALYSTS
   - Some catalysts perform well across multiple pathways
   - These may be good experimental targets

3. WEIGHT SENSITIVITY
   - Rankings can change with weight selection
   - Sensitivity analysis recommended before final selection

4. RECOMMENDATIONS
   - For CO production: Focus on catalysts with ΔE ≈ -0.67 eV
   - For methanol: Look for ΔE ≈ -0.48 eV
   - For formate: Target ΔE ≈ -0.32 eV
""")
```

## Next Steps

- [Sensitivity Analysis](sensitivity-analysis.md) - Test ranking robustness
- [Figure Generation](figure-generation.md) - Create high-quality figures
- [Custom Reactions](../user-guide/reactions/custom.md) - Add new pathways
