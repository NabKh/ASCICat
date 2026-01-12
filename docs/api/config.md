# Configuration

Reaction configurations and constants.

## ReactionConfig

Dataclass for reaction parameters.

```python
from ascicat.config import ReactionConfig

config = ReactionConfig(
    name='HER',
    pathway='H_adsorption',
    activity_descriptor='DFT_ads_E',
    optimal_energy=-0.27,
    activity_width=0.15,
    activity_window=(-0.60, 0.10),
    stability_range=(0.1, 5.0),
    cost_range=(1.0, 200000.0),
    default_weights=(0.33, 0.33, 0.34),
    description='Hydrogen Evolution Reaction',
    references=['Greeley et al. Nat. Mater. 2006']
)
```

### Attributes

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `name` | str | Reaction identifier |
| `pathway` | str | Specific pathway |
| `optimal_energy` | float | Sabatier optimum (eV) |
| `activity_width` | float | Scoring width σ_a (eV) |
| `activity_window` | tuple | (min, max) for analysis |
| `stability_range` | tuple | (min, max) surface energy |
| `cost_range` | tuple | (min, max) cost |
| `default_weights` | tuple | (w_a, w_s, w_c) |
| `description` | str | Human-readable description |
| `references` | list | Scientific citations |

### Methods

```python
config.print_summary()
config.get_weight_dict()  # {'activity': 0.33, 'stability': 0.33, 'cost': 0.34}
```

## Predefined Configurations

```python
from ascicat.config import (
    HER_CONFIG,
    CO2RR_CO_CONFIG,
    CO2RR_CHO_CONFIG,
    CO2RR_COCOH_CONFIG
)
```

| Config | Reaction | $\Delta E_{opt}$ |
|:-------|:---------|:-----------------|
| `HER_CONFIG` | HER | -0.27 eV |
| `CO2RR_CO_CONFIG` | CO2RR-CO | -0.67 eV |
| `CO2RR_CHO_CONFIG` | CO2RR-CHO | -0.48 eV |
| `CO2RR_COCOH_CONFIG` | CO2RR-COCOH | -0.32 eV |

## Helper Functions

### get_reaction_config

Retrieve configuration by name.

```python
from ascicat.config import get_reaction_config

config = get_reaction_config('HER')
config = get_reaction_config('CO2RR', pathway='CO')
```

### list_available_reactions

List all reactions.

```python
from ascicat.config import list_available_reactions

reactions = list_available_reactions()
# {'HER': ['H_adsorption'], 'CO2RR': ['CO', 'CHO', 'COCOH']}
```

### print_available_reactions

Print formatted list.

```python
from ascicat.config import print_available_reactions

print_available_reactions()
```

### create_custom_config

Create custom configuration.

```python
from ascicat.config import create_custom_config

config = create_custom_config(
    name='ORR',
    optimal_energy=-0.45,
    activity_width=0.12,
    description='Oxygen Reduction Reaction'
)
```

### validate_weights

Validate weight values.

```python
from ascicat.config import validate_weights

validate_weights(0.33, 0.33, 0.34)  # OK
validate_weights(0.5, 0.3, 0.3)     # ValueError: sum != 1
```

### normalize_weights

Normalize weights to sum to 1.

```python
from ascicat.config import normalize_weights

w_a, w_s, w_c = normalize_weights(1, 1, 1)
# (0.333..., 0.333..., 0.333...)
```

## ASCIConstants

Global constants.

```python
from ascicat.config import ASCIConstants

# Score bounds
ASCIConstants.SCORE_MIN  # 0.0
ASCIConstants.SCORE_MAX  # 1.0

# Weight constraints
ASCIConstants.WEIGHT_MIN  # 0.0
ASCIConstants.WEIGHT_MAX  # 1.0
ASCIConstants.WEIGHT_SUM  # 1.0

# Numerical stability
ASCIConstants.EPSILON     # 1e-10
ASCIConstants.TOLERANCE   # 1e-6

# Visualization
ASCIConstants.DEFAULT_DPI  # 600
ASCIConstants.DEFAULT_FIGURE_SIZE  # (12, 8)
ASCIConstants.CATEGORICAL_COLORS  # List of 10 colors

# Data requirements
ASCIConstants.REQUIRED_COLUMNS  # List of required columns
ASCIConstants.OPTIONAL_COLUMNS  # List of optional columns
```
