# ASCICalculator

The main class for ASCI calculations.

## Class Definition

::: ascicat.calculator.ASCICalculator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - load_data
        - calculate_asci
        - get_top_catalysts
        - get_statistics
        - print_summary

## Quick Reference

### Initialization

```python
from ascicat import ASCICalculator

# For HER
calc = ASCICalculator(reaction='HER', verbose=True)

# For CO2RR with specific pathway
calc = ASCICalculator(reaction='CO2RR', pathway='CO')

# With custom configuration
from ascicat.config import create_custom_config
config = create_custom_config(name='OER', optimal_energy=1.6)
calc = ASCICalculator(config=config)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `reaction` | str | Required | Reaction type ('HER', 'CO2RR') |
| `pathway` | str | None | Pathway for CO2RR ('CO', 'CHO', 'COCOH') |
| `config` | ReactionConfig | None | Custom configuration |
| `scoring_method` | str | 'linear' | 'linear' or 'gaussian' |
| `verbose` | bool | True | Print progress messages |

## Methods

### load_data

Load catalyst data from file or DataFrame.

```python
# From CSV file
calc.load_data('data/HER_clean.csv')

# From DataFrame
import pandas as pd
df = pd.read_csv('data/HER_clean.csv')
calc.load_data(df)
```

**Parameters:**

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `data_source` | str or DataFrame | Path to CSV or DataFrame |

**Returns:** None (data stored in `calc.data`)

### calculate_asci

Calculate ASCI scores for all catalysts.

```python
results = calc.calculate_asci(
    w_a=0.33,      # Activity weight
    w_s=0.33,      # Stability weight
    w_c=0.34,      # Cost weight
    show_progress=True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `w_a` | float | 0.33 | Activity weight |
| `w_s` | float | 0.33 | Stability weight |
| `w_c` | float | 0.34 | Cost weight |
| `show_progress` | bool | True | Show progress bar |

**Returns:** `pd.DataFrame` with columns:

- Original data columns
- `activity_score` - Activity score S_a
- `stability_score` - Stability score S_s
- `cost_score` - Cost score S_c
- `ASCI` - Combined ASCI score
- Sorted by ASCI (descending)

### get_top_catalysts

Get top-ranked catalysts.

```python
top10 = calc.get_top_catalysts(n=10)
top100 = calc.get_top_catalysts(n=100)
```

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `n` | int | 10 | Number of top catalysts |

**Returns:** `pd.DataFrame` with top n rows

### get_statistics

Get summary statistics.

```python
stats = calc.get_statistics()
print(stats['asci']['mean'])
print(stats['activity']['std'])
```

**Returns:** `dict` with nested dictionaries:

```python
{
    'asci': {'mean', 'std', 'min', 'max', 'median'},
    'activity': {'mean', 'std', 'min', 'max', 'median'},
    'stability': {'mean', 'std', 'min', 'max', 'median'},
    'cost': {'mean', 'std', 'min', 'max', 'median'}
}
```

### print_summary

Print formatted summary to console.

```python
calc.print_summary(n_top=10)
```

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `n_top` | int | 10 | Number of top catalysts to show |

## Properties

### data

Access loaded data:

```python
df = calc.data
print(f"Loaded {len(df)} catalysts")
```

### config

Access reaction configuration:

```python
print(calc.config.optimal_energy)
print(calc.config.activity_width)
```

### results

Access latest calculation results:

```python
results = calc.results  # Same as calculate_asci() return
```

## Example Usage

```python
from ascicat import ASCICalculator

# Initialize
calc = ASCICalculator(reaction='HER', verbose=True)

# Load data
calc.load_data('data/HER_clean.csv')

# Calculate with equal weights
results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

# Analyze
stats = calc.get_statistics()
print(f"Mean ASCI: {stats['asci']['mean']:.3f}")

# Get top catalysts
top20 = calc.get_top_catalysts(n=20)
print(top20[['symbol', 'ASCI', 'activity_score']])

# Print summary
calc.print_summary()

# Access data
print(f"Dataset size: {len(calc.data)}")
print(f"Optimal energy: {calc.config.optimal_energy} eV")
```
