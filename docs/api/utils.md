# Utilities

Helper functions and utilities.

## Module Reference

::: ascicat.utils
    options:
      show_root_heading: false
      show_source: false

## Functions

### generate_unique_labels

Generate unique labels for catalysts with duplicate symbols.

```python
from ascicat.utils import generate_unique_labels

df = generate_unique_labels(df, label_col='display_label')
```

### sample_stratified

Stratified sampling by ASCI score.

```python
from ascicat.utils import sample_stratified

sampled = sample_stratified(
    df,
    n_samples=2000,
    strata_col='ASCI',
    n_strata=4
)
```

### validate_data

Validate input data format.

```python
from ascicat.utils import validate_data

is_valid, errors = validate_data(df)
if not is_valid:
    print("Validation errors:", errors)
```

### compute_pareto_mask

Identify Pareto-optimal points.

```python
from ascicat.utils import compute_pareto_mask
import numpy as np

# Objectives to minimize (1 - score)
objectives = np.column_stack([
    1 - df['activity_score'],
    1 - df['stability_score'],
    1 - df['cost_score']
])

pareto_mask = compute_pareto_mask(objectives)
pareto_catalysts = df[pareto_mask]
```

### format_results_table

Format results for display.

```python
from ascicat.utils import format_results_table

table = format_results_table(
    results.head(10),
    columns=['symbol', 'ASCI', 'activity_score']
)
print(table)
```

## Data Utilities

### load_example_data

Load built-in example datasets.

```python
from ascicat.utils import load_example_data

# Load HER data
her_data = load_example_data('HER')

# Load CO2RR data
co2rr_data = load_example_data('CO2RR', pathway='CO')
```

### export_results

Export results in various formats.

```python
from ascicat.utils import export_results

export_results(
    results,
    output_path='results.csv',
    format='csv'  # or 'xlsx', 'json'
)
```

## Visualization Helpers

### setup_figure_style

Configure matplotlib for high-quality output.

```python
from ascicat.utils import setup_figure_style

setup_figure_style(
    font_scale=1.2,
    dpi=600
)
```

### get_colorblind_palette

Get colorblind-safe color palette.

```python
from ascicat.utils import get_colorblind_palette

colors = get_colorblind_palette(n_colors=5)
```
