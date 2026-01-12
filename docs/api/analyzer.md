# Analyzer

Statistical analysis utilities for ASCI results.

## Class Reference

::: ascicat.analyzer.Analyzer
    options:
      show_root_heading: true
      show_source: false

## Quick Reference

```python
from ascicat.analyzer import Analyzer

analyzer = Analyzer(results, config)

# Basic statistics
stats = analyzer.get_statistics()

# Correlation analysis
corr = analyzer.compute_correlations()

# Distribution analysis
dist = analyzer.analyze_distribution()
```

## Initialization

```python
analyzer = Analyzer(
    results,    # DataFrame from calculate_asci()
    config      # ReactionConfig (optional)
)
```

## Methods

### get_statistics

Get summary statistics for all scores.

```python
stats = analyzer.get_statistics()
print(stats['asci']['mean'])
print(stats['activity']['std'])
```

**Returns:** Nested dictionary with mean, std, min, max, median for each score.

### compute_correlations

Compute correlations between scores.

```python
corr_matrix = analyzer.compute_correlations()
print(corr_matrix)
```

**Returns:** DataFrame with Pearson correlations.

### analyze_distribution

Analyze ASCI score distribution.

```python
dist = analyzer.analyze_distribution()
print(f"Skewness: {dist['skewness']}")
print(f"Kurtosis: {dist['kurtosis']}")
```

**Returns:** Dictionary with distribution metrics.

### identify_outliers

Find outlier catalysts.

```python
outliers = analyzer.identify_outliers(method='iqr', threshold=1.5)
print(f"Found {len(outliers)} outliers")
```

### compare_groups

Compare catalyst groups.

```python
comparison = analyzer.compare_groups(
    group_col='Ametal',
    metric='ASCI'
)
```

## Example Usage

```python
from ascicat import ASCICalculator
from ascicat.analyzer import Analyzer

# Calculate ASCI
calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')
results = calc.calculate_asci()

# Analyze
analyzer = Analyzer(results, calc.config)

# Statistics
stats = analyzer.get_statistics()
print(f"ASCI: {stats['asci']['mean']:.3f} ± {stats['asci']['std']:.3f}")

# Correlations
corr = analyzer.compute_correlations()
print("\nScore Correlations:")
print(corr)

# Distribution
dist = analyzer.analyze_distribution()
print(f"\nSkewness: {dist['skewness']:.3f}")
```
