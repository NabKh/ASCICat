# Reactions

ASCICat supports multiple electrocatalytic reactions with predefined configurations.

## Supported Reactions

| Reaction | Pathways | Description |
|:---------|:---------|:------------|
| **HER** | H adsorption | Hydrogen Evolution Reaction |
| **CO2RR** | CO, CHO, COCOH | CO₂ Reduction Reaction |

## Reaction Selection

```python
from ascicat import ASCICalculator

# HER
calc_her = ASCICalculator(reaction='HER')

# CO2RR with specific pathway
calc_co = ASCICalculator(reaction='CO2RR', pathway='CO')
calc_cho = ASCICalculator(reaction='CO2RR', pathway='CHO')
calc_cocoh = ASCICalculator(reaction='CO2RR', pathway='COCOH')
```

## Reaction Parameters

Each reaction has specific parameters:

```python
from ascicat.config import get_reaction_config

config = get_reaction_config('HER')
config.print_summary()
```

**Output:**

```
======================================================================
Reaction Configuration: HER
Pathway: H_adsorption
======================================================================

Activity Parameters:
  Optimal Energy:    -0.270 eV
  Activity Width:    0.150 eV
  Plotting Window:   [-0.60, +0.10] eV

Stability Parameters:
  Surface Energy:    [0.10, 5.00] J/m²

Cost Parameters:
  Cost Range:        [$1.00, $200,000] per kg

Default Weights (EQUAL):
  Activity:  0.33
  Stability: 0.33
  Cost:      0.34
======================================================================
```

## Quick Reference

| Parameter | HER | CO2RR-CO | CO2RR-CHO | CO2RR-COCOH |
|:----------|:----|:---------|:----------|:------------|
| $\Delta E_{opt}$ | -0.27 eV | -0.67 eV | -0.48 eV | -0.32 eV |
| $\sigma_a$ | 0.15 eV | 0.15 eV | 0.15 eV | 0.15 eV |
| Activity window | [-0.6, 0.1] | [-1.2, -0.2] | [-0.9, -0.1] | [-0.7, 0.05] |

## Detailed Documentation

<div class="grid cards" markdown>

-   :material-hydrogen-station:{ .lg .middle } **HER**

    ---

    Hydrogen Evolution Reaction parameters and applications

    [:octicons-arrow-right-24: HER Documentation](her.md)

-   :material-molecule-co2:{ .lg .middle } **CO2RR**

    ---

    CO₂ Reduction with multiple product pathways

    [:octicons-arrow-right-24: CO2RR Documentation](co2rr.md)

-   :material-tune:{ .lg .middle } **Custom Reactions**

    ---

    Define your own reaction configurations

    [:octicons-arrow-right-24: Custom Reactions](custom.md)

</div>

## Listing Available Reactions

```python
from ascicat.config import print_available_reactions

print_available_reactions()
```

**Output:**

```
======================================================================
Available Reactions in ASCICat
======================================================================

HER (Hydrogen Evolution Reaction):
  • H_adsorption

CO2RR (CO₂ Reduction Reaction):
  • CO
  • CHO
  • COCOH
======================================================================
```
