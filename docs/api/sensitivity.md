# Sensitivity Analysis

Weight sensitivity analysis classes and functions.

## Classes

### SensitivityAnalyzer

Main class for sensitivity analysis.

```python
from ascicat.sensitivity import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(
    calculator,        # ASCICalculator instance
    n_bootstrap=100,   # Bootstrap iterations
    random_state=42    # Reproducibility
)
```

#### Methods

##### generate_weight_grid

Generate systematic weight combinations.

```python
weights = analyzer.generate_weight_grid(
    n_points=15,
    min_weight=0.1,
    max_weight=0.8
)
```

##### generate_full_simplex_grid

Generate complete simplex coverage.

```python
weights = analyzer.generate_full_simplex_grid(n_points=25)
```

##### run_full_sensitivity

Run analysis across all weights.

```python
results = analyzer.run_full_sensitivity(
    weights=weights,
    track_top_n=50,
    verbose=True
)
```

**Returns dict with:**

- `weight_results`: DataFrame with per-weight metrics
- `rank_matrix`: Catalyst ranks across weights
- `asci_matrix`: ASCI scores across weights
- `symbols`: Catalyst identifiers
- `weights`: Weight combinations

##### compute_rank_statistics

Compute comprehensive rank statistics.

```python
rank_stats = analyzer.compute_rank_statistics(results)
```

**Returns DataFrame with:**

- `mean_rank`, `std_rank`, `median_rank`
- `min_rank`, `max_rank`, `rank_range`
- `ci_lower`, `ci_upper`, `ci_width`
- `top5_frequency`, `top10_frequency`, `top20_frequency`
- `robustness_score`

##### compute_sensitivity_indices

Compute variance-based indices.

```python
indices = analyzer.compute_sensitivity_indices(results)
# Returns: {'S_activity': 0.35, 'S_stability': 0.30, 'S_cost': 0.35}
```

##### statistical_tests

Perform significance tests.

```python
tests = analyzer.statistical_tests(results, top_n=20)
# Returns: {'friedman_test': {...}, 'kendall_w': 0.65, 'interpretation': '...'}
```

### SensitivityVisualizer

Visualization for sensitivity results.

```python
from ascicat.sensitivity import SensitivityVisualizer

viz = SensitivityVisualizer(output_dir='sensitivity/')
```

#### Methods

##### plot_ternary_heatmap

Weight space visualization.

```python
viz.plot_ternary_heatmap(
    sensitivity_results,
    metric='best_asci',
    save_name='ternary'
)
```

##### plot_rank_confidence_intervals

Bootstrap CI plot.

```python
viz.plot_rank_confidence_intervals(
    rank_stats,
    n_show=20,
    save_name='rank_ci'
)
```

##### plot_sensitivity_indices

Index bar chart.

```python
viz.plot_sensitivity_indices(indices, save_name='indices')
```

##### plot_robustness_quadrant

Performance vs robustness scatter.

```python
viz.plot_robustness_quadrant(rank_stats, save_name='quadrant')
```

##### plot_comprehensive_summary

Four-panel summary figure.

```python
viz.plot_comprehensive_summary(
    sensitivity_results,
    rank_stats,
    indices,
    save_name='summary'
)
```

## Convenience Function

### run_enhanced_sensitivity_analysis

All-in-one analysis function.

```python
from ascicat.sensitivity import run_enhanced_sensitivity_analysis

results = run_enhanced_sensitivity_analysis(
    calculator=calc,
    output_dir='sensitivity/',
    n_weight_points=15,
    n_bootstrap=100,
    verbose=True
)
```

**Returns dict with:**

- `sensitivity_results`
- `rank_stats`
- `indices`
- `correlations`
- `tests`
