# Data Format

This guide explains the expected data format for ASCICat.

## Required Columns

Your CSV file must contain these columns:

| Column | Type | Description | Unit |
|:-------|:-----|:------------|:-----|
| `DFT_ads_E` | float | Adsorption energy | eV |
| `surface_energy` | float | Surface energy | J/m² |
| `Cost` | float | Material cost | $/kg |
| `symbol` | string | Short catalyst identifier | - |
| `AandB` | string | Detailed identifier | - |
| `reaction_type` | string | Reaction name | - |
| `optimal_energy` | float | Sabatier optimum | eV |
| `activity_width` | float | Scoring width | eV |

## Optional Columns

These columns provide additional information:

| Column | Type | Description |
|:-------|:-----|:------------|
| `Ametal` | string | Primary metal element |
| `Bmetal` | string | Secondary metal element |
| `Cmetal` | string | Tertiary element (ternary alloys) |
| `slab_millers` | string | Surface facet (e.g., '111') |
| `composition` | string | Stoichiometry string |
| `bulk_structure` | string | Crystal structure |
| `magnetic_moment` | float | Total magnetic moment |

## Example CSV

```csv
symbol,AandB,DFT_ads_E,surface_energy,Cost,reaction_type,optimal_energy,activity_width,Ametal,Bmetal,slab_millers
Pt111,Pt-Pt,−0.09,0.52,30000,HER,-0.27,0.15,Pt,Pt,111
Fe2Sb4,Fe-Sb,-0.25,1.12,45,HER,-0.27,0.15,Fe,Sb,111
Cu3Sb,Cu-Sb,-0.285,1.05,12,HER,-0.27,0.15,Cu,Sb,111
Au111,Au-Au,0.18,0.32,60000,HER,-0.27,0.15,Au,Au,111
Ni111,Ni-Ni,-0.47,0.62,20,HER,-0.27,0.15,Ni,Ni,111
```

## Column Details

### DFT_ads_E (Adsorption Energy)

The binding energy of the reaction intermediate to the catalyst surface.

**Convention:**

$$\Delta E = E_{slab+adsorbate} - E_{slab} - E_{reference}$$

Where $E_{reference}$ depends on the reaction:

| Reaction | Reference |
|:---------|:----------|
| HER | ½ E(H₂) |
| CO2RR-CO | E(CO) |
| CO2RR-CHO | E(CHO) |

**Typical ranges:**

- HER: -2.5 to +1.5 eV
- CO2RR: -3.0 to +0.5 eV

### surface_energy

Surface energy in J/m²:

$$\gamma = \frac{E_{slab} - n \cdot E_{bulk}}{2A}$$

**Typical ranges:**

- Noble metals: 0.3 - 0.6 J/m²
- Transition metals: 0.6 - 2.5 J/m²
- Alloys: 0.5 - 4.0 J/m²

### Cost

Material cost in $/kg.

**Sources:**

- USGS Mineral Commodity Summaries
- London Metal Exchange
- Commercial suppliers

For alloys, use composition-weighted average:

$$C_{alloy} = \sum_i x_i \cdot C_i$$

### symbol

Short catalyst identifier for display (max ~12 characters recommended).

**Examples:**

- `Pt111` (pure metal with facet)
- `Fe2Sb4` (intermetallic)
- `Cu3Au` (alloy)
- `PtRu-111` (bimetallic)

### AandB

Detailed identifier, typically `Ametal-Bmetal` format.

**Examples:**

- `Pt-Pt` (pure Pt)
- `Fe-Sb` (Fe-Sb alloy)
- `Cu-Au` (Cu-Au alloy)

## Data Validation

ASCICat validates data on load:

```python
calc = ASCICalculator(reaction='HER')
calc.load_data('data/catalysts.csv')

# Validation checks:
# ✓ Required columns present
# ✓ No missing values in critical columns
# ✓ Physical value ranges
# ✓ Positive costs
# ✓ Non-negative surface energies
```

### Handling Missing Values

```python
# Check for missing values
import pandas as pd

data = pd.read_csv('data/catalysts.csv')
print(data.isnull().sum())

# Drop rows with missing critical values
data_clean = data.dropna(subset=['DFT_ads_E', 'surface_energy', 'Cost'])

# Or impute missing optional values
data['Bmetal'] = data['Bmetal'].fillna('N/A')
```

### Physical Bounds

Values outside these ranges trigger warnings:

| Column | Valid Range | Warning Condition |
|:-------|:------------|:------------------|
| `DFT_ads_E` | [-10, +10] eV | |ΔE| > 10 |
| `surface_energy` | [0, 15] J/m² | γ < 0 or γ > 15 |
| `Cost` | [0.01, 10⁷] $/kg | C ≤ 0 or C > 10⁷ |

## Data Preparation Script

```python
import pandas as pd
import numpy as np

def prepare_ascicat_data(input_file, reaction='HER'):
    """Prepare data file for ASCICat."""

    df = pd.read_csv(input_file)

    # Rename columns if needed
    column_mapping = {
        'adsorption_energy': 'DFT_ads_E',
        'surf_energy': 'surface_energy',
        'price': 'Cost',
        'catalyst': 'symbol'
    }
    df = df.rename(columns=column_mapping)

    # Add reaction metadata
    reaction_params = {
        'HER': {'optimal_energy': -0.27, 'activity_width': 0.15},
        'CO2RR-CO': {'optimal_energy': -0.67, 'activity_width': 0.15},
    }

    params = reaction_params.get(reaction)
    df['reaction_type'] = reaction
    df['optimal_energy'] = params['optimal_energy']
    df['activity_width'] = params['activity_width']

    # Validate
    assert not df['DFT_ads_E'].isnull().any(), "Missing adsorption energies"
    assert not df['surface_energy'].isnull().any(), "Missing surface energies"
    assert not df['Cost'].isnull().any(), "Missing costs"
    assert (df['Cost'] > 0).all(), "Costs must be positive"
    assert (df['surface_energy'] >= 0).all(), "Surface energies must be non-negative"

    # Create AandB if not present
    if 'AandB' not in df.columns:
        if 'Ametal' in df.columns and 'Bmetal' in df.columns:
            df['AandB'] = df['Ametal'] + '-' + df['Bmetal']
        else:
            df['AandB'] = df['symbol']

    return df

# Usage
data = prepare_ascicat_data('raw_data.csv', reaction='HER')
data.to_csv('data/HER_clean.csv', index=False)
```

## Loading Custom Data

```python
from ascicat import ASCICalculator

calc = ASCICalculator(reaction='HER')

# From CSV
calc.load_data('data/my_catalysts.csv')

# From DataFrame
import pandas as pd
df = pd.read_csv('data/my_catalysts.csv')
calc.load_data(df)
```

## Sample Datasets

ASCICat includes sample datasets:

| File | Catalysts | Reaction |
|:-----|:----------|:---------|
| `data/HER_clean.csv` | 48,312 | HER |
| `data/CO2RR_CO_clean.csv` | 25,000+ | CO2RR-CO |
| `data/CO2RR_CHO_clean.csv` | 25,000+ | CO2RR-CHO |
| `data/CO2RR_COCOH_clean.csv` | 25,000+ | CO2RR-COCOH |

## Troubleshooting

!!! warning "Common Issues"

    **"Missing required column: DFT_ads_E"**

    - Check column names match exactly (case-sensitive)
    - Rename columns using pandas

    **"Found X negative cost values"**

    - Verify cost data is positive
    - Check for sign conventions

    **"Surface energy outliers detected"**

    - Values > 10 J/m² are unusual
    - Verify calculation methodology

!!! tip "Data Quality"

    - Ensure consistent DFT settings across all calculations
    - Document exchange-correlation functional used
    - Include metadata about surface terminations
