# Tutorial: Sensitivity Analysis

This tutorial demonstrates comprehensive sensitivity analysis to evaluate ranking robustness.

## Why Sensitivity Analysis?

Different weight choices lead to different rankings. Sensitivity analysis:

1. Reveals which weights most influence rankings
2. Identifies robust candidates that rank well regardless of weights
3. Quantifies uncertainty in rankings
4. Strengthens scientific conclusions

## Complete Workflow

```python
from ascicat import ASCICalculator
from ascicat.sensitivity import (
    SensitivityAnalyzer,
    SensitivityVisualizer,
    run_enhanced_sensitivity_analysis
)
from pathlib import Path

# Setup
output_dir = Path('results/sensitivity')
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize calculator
calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')

print("Running comprehensive sensitivity analysis...")
print("This may take a few minutes for large datasets.\n")
```

## Quick Method: All-in-One Function

```python
# Run complete analysis with one function
results = run_enhanced_sensitivity_analysis(
    calculator=calc,
    output_dir=str(output_dir),
    n_weight_points=15,    # Grid resolution
    n_bootstrap=100,       # Bootstrap iterations
    verbose=True
)

# Access results
sensitivity_results = results['sensitivity_results']
rank_stats = results['rank_stats']
indices = results['indices']
tests = results['tests']
```

## Step-by-Step Analysis

For more control:

### 1. Initialize Analyzer

```python
analyzer = SensitivityAnalyzer(
    calculator=calc,
    n_bootstrap=100,
    random_state=42
)
```

### 2. Generate Weight Grid

```python
# Standard grid with constraints
weights = analyzer.generate_weight_grid(
    n_points=15,
    min_weight=0.1,    # No weight below 10%
    max_weight=0.8     # No weight above 80%
)
print(f"Generated {len(weights)} weight combinations")

# Or full simplex (covers entire space)
weights_full = analyzer.generate_full_simplex_grid(n_points=20)
print(f"Full simplex: {len(weights_full)} combinations")
```

### 3. Run Sensitivity Calculations

```python
sensitivity_results = analyzer.run_full_sensitivity(
    weights=weights,
    track_top_n=50,
    verbose=True
)

print(f"\nResults structure:")
print(f"  - weight_results: {len(sensitivity_results['weight_results'])} rows")
print(f"  - rank_matrix: {sensitivity_results['rank_matrix'].shape}")
print(f"  - symbols: {len(sensitivity_results['symbols'])} catalysts")
```

### 4. Compute Rank Statistics

```python
rank_stats = analyzer.compute_rank_statistics(sensitivity_results)

print("\nTop 10 Most Robust Catalysts:")
print("-" * 70)
print(f"{'Rank':<6} {'Symbol':<14} {'Mean Rank':<12} {'Std':<10} "
      f"{'Top10 Freq':<12} {'Robustness':<10}")
print("-" * 70)

for i, (_, row) in enumerate(rank_stats.head(10).iterrows(), 1):
    print(f"{i:<6} {row['symbol']:<14} {row['mean_rank']:<12.1f} "
          f"{row['std_rank']:<10.1f} {row['top10_frequency']:<12.2%} "
          f"{row['robustness_score']:<10.3f}")
```

### 5. Compute Sensitivity Indices

```python
indices = analyzer.compute_sensitivity_indices(sensitivity_results)

print("\nSensitivity Indices (contribution to ranking variance):")
print("-" * 40)
print(f"  Activity (w_a):  {indices['S_activity']:.3f}")
print(f"  Stability (w_s): {indices['S_stability']:.3f}")
print(f"  Cost (w_c):      {indices['S_cost']:.3f}")

# Interpret
dominant = max(indices, key=indices.get)
print(f"\n→ {dominant.replace('S_', '').title()} weight has the "
      f"strongest influence on rankings")
```

### 6. Statistical Tests

```python
tests = analyzer.statistical_tests(sensitivity_results, top_n=20)

print("\nStatistical Tests:")
print("-" * 50)

# Kendall's W (concordance)
print(f"Kendall's W: {tests['kendall_w']:.3f}")
print(f"Interpretation: {tests['interpretation']}")

# Friedman test
friedman = tests['friedman_test']
if friedman['significant']:
    print(f"\nFriedman test: Rankings differ significantly (p < 0.05)")
else:
    print(f"\nFriedman test: No significant difference in rankings")
```

## Visualization

### Ternary Heatmap

```python
viz = SensitivityVisualizer(output_dir=str(output_dir))

# Best ASCI across weight space
viz.plot_ternary_heatmap(
    sensitivity_results,
    metric='best_asci',
    title='Best ASCI Score Across Weight Space',
    save_name='ternary_best_asci'
)
```

### Rank Confidence Intervals

```python
viz.plot_rank_confidence_intervals(
    rank_stats,
    n_show=20,
    save_name='rank_confidence_intervals'
)
```

### Sensitivity Indices Bar Chart

```python
viz.plot_sensitivity_indices(
    indices,
    save_name='sensitivity_indices'
)
```

### Robustness Quadrant

```python
viz.plot_robustness_quadrant(
    rank_stats,
    n_label=15,
    save_name='robustness_quadrant'
)
```

### Comprehensive Summary

```python
viz.plot_comprehensive_summary(
    sensitivity_results,
    rank_stats,
    indices,
    save_name='sensitivity_summary'
)
```

## Interpreting Results

### Sensitivity Indices

| Index Range | Interpretation |
|:------------|:---------------|
| ~0.33 each | Equal influence - balanced screening |
| > 0.5 for one | Dominant factor - rankings highly sensitive |
| < 0.2 for one | Minor influence - can be deprioritized |

### Kendall's W (Concordance)

| W Value | Interpretation | Action |
|:--------|:---------------|:-------|
| < 0.3 | Low agreement | Rankings highly weight-dependent |
| 0.3-0.5 | Moderate | Some stable patterns |
| 0.5-0.7 | Good | Reasonably robust rankings |
| > 0.7 | Strong | Very robust - confident in results |

### Robustness Score

| Score | Category | Recommendation |
|:------|:---------|:---------------|
| > 0.7 | Excellent | Strong candidate |
| 0.5-0.7 | Good | Viable candidate |
| 0.3-0.5 | Moderate | Consider with caution |
| < 0.3 | Poor | Highly weight-dependent |

## Export Results

```python
# Save rank statistics
rank_stats.to_csv(output_dir / 'rank_statistics.csv', index=False)

# Save weight sensitivity data
sensitivity_results['weight_results'].to_csv(
    output_dir / 'weight_sensitivity.csv', index=False
)

# Save summary report
report = f"""
SENSITIVITY ANALYSIS REPORT
===========================

Dataset: HER Catalysts
Weight Combinations: {len(weights)}
Bootstrap Iterations: {analyzer.n_bootstrap}

SENSITIVITY INDICES
-------------------
Activity:  {indices['S_activity']:.3f}
Stability: {indices['S_stability']:.3f}
Cost:      {indices['S_cost']:.3f}

RANKING CONCORDANCE
-------------------
Kendall's W: {tests['kendall_w']:.3f}
{tests['interpretation']}

TOP 5 ROBUST CATALYSTS
----------------------
"""

for i, (_, row) in enumerate(rank_stats.head(5).iterrows(), 1):
    report += f"{i}. {row['symbol']}: R={row['robustness_score']:.3f}, "
    report += f"mean_rank={row['mean_rank']:.1f}\n"

with open(output_dir / 'sensitivity_report.txt', 'w') as f:
    f.write(report)

print(f"\nResults exported to: {output_dir}/")
```

## Key Takeaways

```python
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)

# Most influential weight
max_index = max(indices, key=indices.get)
print(f"\n1. Most influential weight: {max_index.replace('S_', '')}")
print(f"   ({indices[max_index]:.0%} of ranking variance)")

# Ranking robustness
if tests['kendall_w'] > 0.5:
    print("\n2. Rankings are reasonably robust across weight choices")
else:
    print("\n2. Rankings are sensitive to weight choices - interpret carefully")

# Top robust candidate
top = rank_stats.iloc[0]
print(f"\n3. Most robust candidate: {top['symbol']}")
print(f"   - Robustness score: {top['robustness_score']:.3f}")
print(f"   - Top 10 frequency: {top['top10_frequency']:.0%}")
print(f"   - Mean rank: {top['mean_rank']:.1f} (95% CI: "
      f"[{top['ci_lower']:.1f}, {top['ci_upper']:.1f}])")
```
