# Tutorial: ASCICat vs. Pareto Analysis

This tutorial compares ASCICat ranking with Pareto frontier analysis.

## Overview

Both methods address multi-objective optimization but differently:

| Aspect | Pareto | ASCICat |
|:-------|:-------|:--------|
| Output | Set of non-dominated solutions | Ranked list |
| Preferences | Not required | Explicit weights |
| Comparability | Within dataset | Across studies |
| Interpretation | Trade-off set | Prioritized ranking |

## Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ascicat import ASCICalculator
from pathlib import Path

# Setup
output_dir = Path('results/pareto_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

# Calculate ASCI
calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')
results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

print(f"Dataset: {len(results):,} catalysts")
```

## Pareto Frontier Calculation

```python
def is_pareto_optimal(costs):
    """
    Find Pareto-optimal points.

    Parameters
    ----------
    costs : array-like, shape (n_samples, n_objectives)
        Objective values to minimize (lower is better)

    Returns
    -------
    is_optimal : array, shape (n_samples,)
        Boolean array indicating Pareto-optimal points
    """
    is_optimal = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_optimal[i]:
            # Keep any point not dominated by c
            is_optimal[is_optimal] = np.any(
                costs[is_optimal] < c, axis=1
            ) | np.all(costs[is_optimal] == c, axis=1)
            is_optimal[i] = True
    return is_optimal


# Prepare objectives (minimize = better)
# Activity: maximize → minimize (1 - score)
# Stability: maximize → minimize (1 - score)
# Cost: maximize → minimize (1 - score)
objectives = np.column_stack([
    1 - results['activity_score'].values,
    1 - results['stability_score'].values,
    1 - results['cost_score'].values
])

# Find Pareto-optimal points
pareto_mask = is_pareto_optimal(objectives)
n_pareto = pareto_mask.sum()

print(f"\nPareto-optimal catalysts: {n_pareto}")
print(f"Percentage of dataset: {100*n_pareto/len(results):.1f}%")
```

## Compare with ASCI Ranking

```python
# Get top ASCI-ranked catalysts
top_n = 100
asci_top = set(results.head(top_n)['symbol'].values)

# Get Pareto-optimal catalysts
pareto_catalysts = set(results[pareto_mask]['symbol'].values)

# Overlap analysis
overlap = asci_top & pareto_catalysts
asci_only = asci_top - pareto_catalysts
pareto_only = pareto_catalysts - asci_top

print(f"\n{'='*60}")
print(f"COMPARISON: Top {top_n} ASCI vs Pareto Frontier")
print(f"{'='*60}")
print(f"\nOverlap: {len(overlap)} catalysts")
print(f"  ({100*len(overlap)/top_n:.1f}% of top ASCI are Pareto-optimal)")
print(f"\nASCI top {top_n} only: {len(asci_only)} catalysts")
print(f"Pareto-optimal only: {len(pareto_only)} catalysts")
```

## Visualize Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Colors
colors = np.where(pareto_mask, 'red', 'lightgray')
colors = np.where(results.index < top_n, 'blue', colors)
# Overlap: purple
for i, row in results.iterrows():
    if row['symbol'] in overlap:
        colors[i] = 'purple'

# Plot 2D projections
score_pairs = [
    ('activity_score', 'cost_score', 'Activity vs Cost'),
    ('activity_score', 'stability_score', 'Activity vs Stability'),
    ('stability_score', 'cost_score', 'Stability vs Cost')
]

for ax, (x_col, y_col, title) in zip(axes, score_pairs):
    # All points (gray background)
    ax.scatter(results[x_col], results[y_col],
               c='lightgray', s=10, alpha=0.3, label='_nolegend_')

    # Pareto points (red)
    pareto_data = results[pareto_mask]
    ax.scatter(pareto_data[x_col], pareto_data[y_col],
               c='red', s=30, alpha=0.7, label='Pareto-optimal')

    # Top ASCI (blue)
    asci_data = results.head(top_n)
    ax.scatter(asci_data[x_col], asci_data[y_col],
               c='blue', s=30, alpha=0.7, label=f'Top {top_n} ASCI')

    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(output_dir / 'pareto_vs_asci_2d.png', dpi=600)
plt.close()

print(f"\nFigure saved: {output_dir}/pareto_vs_asci_2d.png")
```

## Rank Distribution of Pareto Points

```python
# Where do Pareto-optimal points rank in ASCI?
pareto_ranks = results[pareto_mask].index + 1  # 1-indexed ranks

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(pareto_ranks, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(top_n, color='red', linestyle='--',
           label=f'Top {top_n} threshold')
ax.set_xlabel('ASCI Rank')
ax.set_ylabel('Count of Pareto-Optimal Catalysts')
ax.set_title('Distribution of Pareto-Optimal Catalysts by ASCI Rank')
ax.legend()

# Add statistics
stats_text = f'n = {n_pareto} Pareto-optimal\n'
stats_text += f'Median rank: {np.median(pareto_ranks):.0f}\n'
stats_text += f'Mean rank: {np.mean(pareto_ranks):.0f}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
        fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white'))

fig.savefig(output_dir / 'pareto_rank_distribution.png', dpi=600)
plt.close()

print(f"Figure saved: {output_dir}/pareto_rank_distribution.png")
```

## Detailed Comparison Table

```python
# Create comparison DataFrame
comparison_data = []

# Top 20 ASCI-ranked
for rank, (_, row) in enumerate(results.head(20).iterrows(), 1):
    is_pareto = row['symbol'] in pareto_catalysts
    comparison_data.append({
        'ASCI_Rank': rank,
        'Symbol': row['symbol'],
        'ASCI': row['ASCI'],
        'Activity': row['activity_score'],
        'Stability': row['stability_score'],
        'Cost': row['cost_score'],
        'Pareto_Optimal': is_pareto
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*80)
print("TOP 20 ASCI-RANKED CATALYSTS: Pareto Status")
print("="*80)
print(f"\n{'Rank':<6} {'Symbol':<14} {'ASCI':<8} {'S_a':<8} "
      f"{'S_s':<8} {'S_c':<8} {'Pareto?':<10}")
print("-"*80)

for _, row in comparison_df.iterrows():
    pareto_str = "Yes" if row['Pareto_Optimal'] else "No"
    print(f"{row['ASCI_Rank']:<6} {row['Symbol']:<14} {row['ASCI']:.4f}   "
          f"{row['Activity']:.4f}   {row['Stability']:.4f}   "
          f"{row['Cost']:.4f}   {pareto_str:<10}")

# Summary
n_pareto_in_top20 = comparison_df['Pareto_Optimal'].sum()
print(f"\n→ {n_pareto_in_top20}/20 top ASCI catalysts are Pareto-optimal "
      f"({100*n_pareto_in_top20/20:.0f}%)")
```

## When to Use Each Method

```python
print("\n" + "="*70)
print("WHEN TO USE EACH METHOD")
print("="*70)

print("""
PARETO FRONTIER
---------------
✓ Initial exploration without preference bias
✓ Identifying the full trade-off space
✓ When stakeholders disagree on priorities
✓ Generating options for discussion

ASCI SCORING
------------
✓ When priorities can be quantified
✓ For reproducible, documented rankings
✓ Cross-study comparisons
✓ Final prioritization for experiments

COMPLEMENTARY USE
-----------------
1. First: Use Pareto to identify non-dominated set
2. Then: Apply ASCI within Pareto set for ranking
3. Verify: Top ASCI candidates should be Pareto-optimal
""")
```

## Combined Workflow

```python
# Filter to Pareto-optimal catalysts only
pareto_results = results[pareto_mask].copy()
pareto_results = pareto_results.reset_index(drop=True)

# Re-rank within Pareto set
pareto_results['Pareto_Rank'] = range(1, len(pareto_results) + 1)

print(f"\nFiltered to {len(pareto_results)} Pareto-optimal catalysts")
print("\nTop 10 within Pareto set:")
print(pareto_results[['symbol', 'ASCI', 'activity_score',
                      'stability_score', 'cost_score']].head(10))
```

## Key Findings

```python
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

findings = f"""
1. OVERLAP VALIDATION
   - {100*len(overlap)/top_n:.0f}% of top {top_n} ASCI catalysts are Pareto-optimal
   - This validates ASCI's focus on high-performers

2. PARETO SET SIZE
   - {n_pareto} Pareto-optimal catalysts ({100*n_pareto/len(results):.1f}% of dataset)
   - ASCI reduces this to a single prioritized list

3. COMPLEMENTARITY
   - Pareto shows the trade-off space
   - ASCI provides actionable priorities

4. RECOMMENDATION
   - Use Pareto for understanding trade-offs
   - Use ASCI for final experimental prioritization
"""

print(findings)

# Save comparison
comparison_df.to_csv(output_dir / 'asci_pareto_comparison.csv', index=False)
print(f"\nResults saved to: {output_dir}/")
```
