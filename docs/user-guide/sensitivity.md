# Sensitivity Analysis

Sensitivity analysis addresses the critical "weight selection problem" by quantifying how rankings depend on weight choices.

## Why Sensitivity Analysis?

!!! question "The Weight Selection Problem"

    How do you choose weights? Different weights lead to different rankings. Without sensitivity analysis, your conclusions may be highly dependent on arbitrary choices.

Sensitivity analysis answers:

1. **How robust is my top candidate?** Does it rank well across weight ranges?
2. **Which weight matters most?** Is activity, stability, or cost driving rankings?
3. **Are there robust alternatives?** Which catalysts perform consistently?

## Quick Start

```python
from ascicat import ASCICalculator
from ascicat.sensitivity import (
    SensitivityAnalyzer,
    SensitivityVisualizer,
    run_enhanced_sensitivity_analysis
)

# Initialize calculator
calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')

# Run full sensitivity analysis
results = run_enhanced_sensitivity_analysis(
    calculator=calc,
    output_dir='results/sensitivity',
    n_weight_points=15,
    n_bootstrap=100
)
```

## The SensitivityAnalyzer Class

### Initialization

```python
analyzer = SensitivityAnalyzer(
    calculator=calc,
    n_bootstrap=100,    # Bootstrap iterations
    random_state=42     # Reproducibility
)
```

### Weight Grid Generation

Generate systematic weight combinations:

```python
# Standard grid (constrained to min_weight, max_weight)
weights = analyzer.generate_weight_grid(
    n_points=21,
    min_weight=0.1,
    max_weight=0.8
)
print(f"Generated {len(weights)} weight combinations")

# Full simplex grid (covers entire weight space)
weights_full = analyzer.generate_full_simplex_grid(n_points=25)
print(f"Full simplex: {len(weights_full)} combinations")
```

### Full Sensitivity Run

```python
# Calculate ASCI for all weight combinations
sensitivity_results = analyzer.run_full_sensitivity(
    weights=weights,
    track_top_n=50,
    verbose=True
)
```

**Returns:**

| Key | Description |
|:----|:------------|
| `weight_results` | DataFrame with per-weight metrics |
| `rank_matrix` | Catalyst ranks across all weights |
| `asci_matrix` | ASCI scores across all weights |
| `symbols` | Catalyst identifiers |
| `weights` | Weight combinations used |

## Rank Statistics

Compute comprehensive statistics for each catalyst:

```python
rank_stats = analyzer.compute_rank_statistics(sensitivity_results)

# View top robust catalysts
print(rank_stats[['symbol', 'mean_rank', 'std_rank',
                  'robustness_score', 'top10_frequency']].head(10))
```

### Available Statistics

| Metric | Description |
|:-------|:------------|
| `mean_rank` | Average rank across all weights |
| `std_rank` | Standard deviation of rank |
| `median_rank` | Median rank |
| `min_rank`, `max_rank` | Best/worst rank achieved |
| `ci_lower`, `ci_upper` | 95% bootstrap CI for mean rank |
| `top5_frequency` | Fraction of weights where rank ≤ 5 |
| `top10_frequency` | Fraction of weights where rank ≤ 10 |
| `robustness_score` | Composite robustness metric [0,1] |
| `cv` | Coefficient of variation |
| `iqr` | Interquartile range |

### Robustness Score

The composite robustness score combines:

$$R = 0.35 \times f_{top10} + 0.25 \times f_{top20} + 0.25 \times S_{stability} + 0.15 \times S_{rank}$$

Where:

- $f_{topN}$ = Frequency in top N
- $S_{stability}$ = 1 - coefficient of variation
- $S_{rank}$ = Normalized mean rank

## Sensitivity Indices

Quantify how much each weight contributes to ranking variance:

```python
indices = analyzer.compute_sensitivity_indices(sensitivity_results)

print(f"Activity sensitivity:  {indices['S_activity']:.3f}")
print(f"Stability sensitivity: {indices['S_stability']:.3f}")
print(f"Cost sensitivity:      {indices['S_cost']:.3f}")
```

**Interpretation:**

| Index Value | Interpretation |
|:------------|:---------------|
| ~0.33 | Equal influence |
| > 0.4 | Dominant factor |
| < 0.2 | Minor influence |

## Statistical Tests

Perform formal statistical tests:

```python
tests = analyzer.statistical_tests(sensitivity_results, top_n=20)

# Kendall's W (concordance)
print(f"Kendall's W: {tests['kendall_w']:.3f}")
print(f"Interpretation: {tests['interpretation']}")

# Friedman test
if tests['friedman_test']['significant']:
    print("Rankings differ significantly across weights (p < 0.05)")
```

### Kendall's W Interpretation

| W Value | Agreement Level |
|:--------|:----------------|
| < 0.1 | Very low - rankings highly weight-dependent |
| 0.1 - 0.3 | Low - moderate sensitivity |
| 0.3 - 0.5 | Moderate - some stability |
| 0.5 - 0.7 | Good - reasonably robust |
| > 0.7 | Strong - highly robust rankings |

## Visualization

### Ternary Heatmap

Visualize metrics across the weight simplex:

```python
from ascicat.sensitivity import SensitivityVisualizer

viz = SensitivityVisualizer(output_dir='sensitivity/')

viz.plot_ternary_heatmap(
    sensitivity_results,
    metric='best_asci',
    save_name='ternary_asci'
)
```

### Rank Confidence Intervals

```python
viz.plot_rank_confidence_intervals(
    rank_stats,
    n_show=20,
    save_name='rank_ci'
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

Performance vs. robustness trade-off:

```python
viz.plot_robustness_quadrant(
    rank_stats,
    n_label=15,
    save_name='robustness_quadrant'
)
```

### Comprehensive Summary

Four-panel summary figure:

```python
viz.plot_comprehensive_summary(
    sensitivity_results,
    rank_stats,
    indices,
    save_name='sensitivity_summary'
)
```

## Complete Workflow

```python
from ascicat import ASCICalculator
from ascicat.sensitivity import (
    SensitivityAnalyzer,
    SensitivityVisualizer
)
from pathlib import Path

# Setup
calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')
output_dir = Path('results/sensitivity')
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize
analyzer = SensitivityAnalyzer(calc, n_bootstrap=100)
viz = SensitivityVisualizer(output_dir=str(output_dir))

# Generate weights
weights = analyzer.generate_weight_grid(n_points=15)

# Run analysis
sensitivity_results = analyzer.run_full_sensitivity(weights)
rank_stats = analyzer.compute_rank_statistics(sensitivity_results)
indices = analyzer.compute_sensitivity_indices(sensitivity_results)
tests = analyzer.statistical_tests(sensitivity_results)

# Visualize
viz.plot_ternary_heatmap(sensitivity_results)
viz.plot_rank_confidence_intervals(rank_stats)
viz.plot_sensitivity_indices(indices)
viz.plot_robustness_quadrant(rank_stats)
viz.plot_comprehensive_summary(sensitivity_results, rank_stats, indices)

# Report
print("\n" + "="*70)
print("SENSITIVITY ANALYSIS SUMMARY")
print("="*70)
print(f"\nWeight combinations tested: {len(weights)}")
print(f"\nSensitivity Indices:")
print(f"  Activity:  {indices['S_activity']:.3f}")
print(f"  Stability: {indices['S_stability']:.3f}")
print(f"  Cost:      {indices['S_cost']:.3f}")
print(f"\nKendall's W: {tests['kendall_w']:.3f}")
print(f"  {tests['interpretation']}")
print(f"\nMost Robust Catalysts:")
for i, row in rank_stats.head(5).iterrows():
    print(f"  {i+1}. {row['symbol']}: R={row['robustness_score']:.3f}, "
          f"mean_rank={row['mean_rank']:.1f}")
```

## Best Practices

!!! tip "Grid Resolution"

    - Quick analysis: n_points=10 (~50 combinations)
    - Standard: n_points=15 (~100 combinations)
    - Thorough: n_points=25 (~300 combinations)

!!! tip "Bootstrap Iterations"

    - Quick: n_bootstrap=50
    - Standard: n_bootstrap=100
    - Thorough: n_bootstrap=1000

!!! tip "Reporting"

    Always report:

    1. Weight grid parameters
    2. Sensitivity indices
    3. Kendall's W and interpretation
    4. Top robust candidates with CIs
