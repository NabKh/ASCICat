# CO₂ Reduction Reaction (CO2RR)

The CO₂ Reduction Reaction converts carbon dioxide into valuable chemicals and fuels.

## Overview

CO2RR can produce various products depending on the catalyst and conditions:

| Product | Electrons | Pathway |
|:--------|:----------|:--------|
| CO | 2e⁻ | CO pathway |
| HCOOH | 2e⁻ | Formate pathway |
| CH₃OH | 6e⁻ | Methanol pathway |
| CH₄ | 8e⁻ | Methane pathway |
| C₂H₄ | 12e⁻ | Ethylene pathway |

## Supported Pathways

ASCICat supports three major CO2RR pathways:

### CO Pathway

$$CO_2 + 2H^+ + 2e^- \rightarrow CO + H_2O$$

**Key intermediate:** *CO bound to surface

```python
from ascicat import ASCICalculator

calc = ASCICalculator(reaction='CO2RR', pathway='CO')
```

| Parameter | Value |
|:----------|:------|
| $\Delta E_{opt}$ | -0.67 eV |
| $\sigma_a$ | 0.15 eV |
| Window | [-1.2, -0.2] eV |

### CHO Pathway (Methanol)

$$CO_2 + 6H^+ + 6e^- \rightarrow CH_3OH + H_2O$$

**Key intermediate:** *CHO bound to surface

```python
calc = ASCICalculator(reaction='CO2RR', pathway='CHO')
```

| Parameter | Value |
|:----------|:------|
| $\Delta E_{opt}$ | -0.48 eV |
| $\sigma_a$ | 0.15 eV |
| Window | [-0.9, -0.1] eV |

### COCOH Pathway (Formate)

$$CO_2 + 2H^+ + 2e^- \rightarrow HCOOH$$

**Key intermediate:** *COOH bound to surface

```python
calc = ASCICalculator(reaction='CO2RR', pathway='COCOH')
```

| Parameter | Value |
|:----------|:------|
| $\Delta E_{opt}$ | -0.32 eV |
| $\sigma_a$ | 0.15 eV |
| Window | [-0.7, 0.05] eV |

## Multi-Pathway Analysis

Screen across all pathways:

```python
from ascicat import ASCICalculator
import pandas as pd

pathways = ['CO', 'CHO', 'COCOH']
all_results = {}

for pathway in pathways:
    calc = ASCICalculator(reaction='CO2RR', pathway=pathway)
    calc.load_data(f'data/CO2RR_{pathway}_clean.csv')
    results = calc.calculate_asci()
    all_results[pathway] = calc.get_top_catalysts(n=10)

# Compare top catalysts across pathways
for pathway, top in all_results.items():
    print(f"\n{pathway} Pathway - Top 3:")
    for i, (_, row) in enumerate(top.head(3).iterrows(), 1):
        print(f"  {i}. {row['symbol']} (ASCI={row['ASCI']:.3f})")
```

## Benchmark Materials

### CO Pathway

| Material | $\Delta E_{CO}$ (eV) | Selectivity |
|:---------|:---------------------|:------------|
| Au(111) | -0.55 | ~100% CO |
| Ag(111) | -0.48 | ~90% CO |
| Zn | -0.60 | ~80% CO |

### Formate Pathway

| Material | $\Delta E_{COOH}$ (eV) | Selectivity |
|:---------|:-----------------------|:------------|
| Sn | -0.45 | ~90% HCOOH |
| Pb | -0.50 | ~80% HCOOH |
| Bi | -0.38 | ~95% HCOOH |

### Multi-Carbon Products

| Material | Products | Notes |
|:---------|:---------|:------|
| Cu(111) | CH₄, C₂H₄, C₂H₅OH | Unique multi-carbon activity |
| Cu-Ag | C₂+ products | Synergistic effect |

## Configuration Details

```python
from ascicat.config import get_reaction_config

# Get all CO2RR configurations
for pathway in ['CO', 'CHO', 'COCOH']:
    config = get_reaction_config('CO2RR', pathway=pathway)
    print(f"\n{pathway} Pathway:")
    print(f"  Optimal: {config.optimal_energy} eV")
    print(f"  Width: {config.activity_width} eV")
    print(f"  Window: {config.activity_window}")
```

## Selectivity Considerations

!!! warning "Competing HER"

    All CO2RR catalysts also catalyze HER. The competitive adsorption of H vs. CO₂ determines selectivity. ASCICat scores intrinsic activity but doesn't directly model selectivity.

To account for selectivity:

1. Pre-filter catalysts known for high CO2RR selectivity
2. Consider including a selectivity descriptor in custom analysis
3. Use experimental validation for top candidates

## Scientific References

- Peterson, A. A. & Nørskov, J. K. *J. Phys. Chem. Lett.* **3**, 251 (2012)
- Nitopi, S. et al. *Chem. Rev.* **119**, 7610 (2019)
- Kuhl, K. P. et al. *Energy Environ. Sci.* **5**, 7050 (2012)
- Hansen, H. A. et al. *J. Phys. Chem. Lett.* **4**, 388 (2013)
