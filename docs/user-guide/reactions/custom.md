# Custom Reactions

Define your own reaction configurations for reactions not included by default.

## Creating Custom Configuration

Use the `create_custom_config` helper:

```python
from ascicat.config import create_custom_config

# Define custom ORR configuration
orr_config = create_custom_config(
    name='ORR',
    optimal_energy=-0.45,       # eV (your Sabatier optimum)
    activity_width=0.12,        # eV (tolerance)
    pathway='O2_reduction',
    description='Oxygen Reduction Reaction (4e⁻ pathway)',
    activity_window=(-0.9, 0.0),
    stability_range=(0.2, 4.0),
    cost_range=(1.0, 200000.0)
)

orr_config.print_summary()
```

## Using Custom Configuration

### With ASCICalculator

```python
from ascicat import ASCICalculator

# Initialize with custom config
calc = ASCICalculator(config=orr_config, verbose=True)

# Load your data
calc.load_data('data/ORR_catalysts.csv')

# Calculate ASCI
results = calc.calculate_asci(w_a=0.4, w_s=0.35, w_c=0.25)
```

### Manual Configuration

For full control, create a `ReactionConfig` directly:

```python
from ascicat.config import ReactionConfig

my_config = ReactionConfig(
    # Required
    name='OER',
    pathway='water_oxidation',
    activity_descriptor='DFT_ads_E',
    optimal_energy=1.60,           # eV
    activity_width=0.20,           # eV
    activity_window=(1.0, 2.2),    # eV
    stability_range=(0.5, 5.0),    # J/m²
    cost_range=(1.0, 200000.0),    # $/kg

    # Optional
    default_weights=(0.40, 0.35, 0.25),
    description='Oxygen Evolution Reaction - key for water splitting',
    references=[
        'Man, I. C. et al. ChemCatChem 3, 1159 (2011)',
        'Suntivich, J. et al. Science 334, 1383 (2011)'
    ]
)
```

## Parameters Reference

### Required Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | str | Reaction identifier (e.g., 'OER') |
| `pathway` | str | Specific pathway name |
| `activity_descriptor` | str | Column name (typically 'DFT_ads_E') |
| `optimal_energy` | float | Sabatier optimum (eV) |
| `activity_width` | float | Scoring tolerance (eV) |
| `activity_window` | tuple | (min, max) for analysis (eV) |
| `stability_range` | tuple | (min, max) surface energy (J/m²) |
| `cost_range` | tuple | (min, max) cost ($/kg) |

### Optional Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `default_weights` | tuple | (0.33, 0.33, 0.34) | (w_a, w_s, w_c) |
| `description` | str | "" | Human-readable description |
| `references` | list | [] | Scientific citations |

## Example: Nitrogen Reduction Reaction

```python
from ascicat.config import create_custom_config

nrr_config = create_custom_config(
    name='NRR',
    optimal_energy=-0.40,          # eV (*N2H binding)
    activity_width=0.15,
    pathway='N2_to_NH3',
    description='Nitrogen Reduction Reaction to Ammonia',
    activity_window=(-0.8, 0.0),
    stability_range=(0.3, 4.0),
    cost_range=(1.0, 150000.0),
    references=[
        'Skulason, E. et al. Phys. Chem. Chem. Phys. 14, 1235 (2012)'
    ]
)

# Use with calculator
from ascicat import ASCICalculator

calc = ASCICalculator(config=nrr_config)
calc.load_data('data/NRR_catalysts.csv')
results = calc.calculate_asci()
```

## Example: Methane Activation

```python
ch4_config = create_custom_config(
    name='CH4_activation',
    optimal_energy=-0.85,          # eV (*CH3 binding)
    activity_width=0.18,
    pathway='CH4_to_CH3OH',
    description='Methane partial oxidation to methanol',
    activity_window=(-1.5, -0.2),
    stability_range=(0.5, 3.5),
    cost_range=(1.0, 100000.0)
)
```

## Validation

Configurations are validated on creation:

```python
try:
    invalid_config = create_custom_config(
        name='Test',
        optimal_energy=-0.5,
        activity_width=-0.1,  # Error: must be positive
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

**Validations performed:**

- Weights sum to 1.0
- All weights in [0, 1]
- activity_width > 0
- Window min < max
- Stability/cost ranges valid

## Registering Custom Reactions

For repeated use, register your config:

```python
from ascicat.config import REACTION_REGISTRY

# Add to registry
REACTION_REGISTRY['OER'] = {
    'default': oer_config,
    'water_oxidation': oer_config
}

# Now accessible via standard interface
from ascicat import ASCICalculator
calc = ASCICalculator(reaction='OER')
```

## Data Requirements

Your custom reaction data must include:

| Column | Description |
|:-------|:------------|
| `DFT_ads_E` | Binding energy of key intermediate |
| `surface_energy` | Surface stability descriptor |
| `Cost` | Material cost |
| `symbol` | Catalyst identifier |
| `reaction_type` | Your reaction name |
| `optimal_energy` | Your $\Delta E_{opt}$ |
| `activity_width` | Your $\sigma_a$ |

## Best Practices

!!! tip "Determining Optimal Energy"

    Literature review or:

    1. Compile experimental activity data
    2. Plot activity vs. binding energy
    3. Fit volcano curve
    4. Identify peak position

!!! tip "Setting Activity Width"

    - Default 0.15 eV works for most reactions
    - Increase for reactions with broad optima
    - Decrease for sharp selectivity requirements

!!! tip "Documentation"

    Always include:

    - References for optimal energy choice
    - Description of the reaction
    - Any assumptions made
