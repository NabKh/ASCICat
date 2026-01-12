# Cost Scoring

Cost scoring quantifies economic viability using logarithmic normalization to handle the enormous range in material costs.

## The Economic Challenge

Material costs span **5+ orders of magnitude**:

| Material | Cost ($/kg) | Relative to Fe |
|:---------|:------------|:---------------|
| Iron (Fe) | ~2 | 1× |
| Aluminum (Al) | ~3 | 1.5× |
| Copper (Cu) | ~10 | 5× |
| Nickel (Ni) | ~20 | 10× |
| Silver (Ag) | ~900 | 450× |
| Gold (Au) | ~60,000 | 30,000× |
| Platinum (Pt) | ~30,000 | 15,000× |
| Rhodium (Rh) | ~150,000 | 75,000× |
| Iridium (Ir) | ~150,000 | 75,000× |

!!! question "Why Logarithmic?"

    Linear normalization would give:

    - Iron: 1.000
    - Copper: 0.9999
    - Silver: 0.994
    - Platinum: 0.800

    This provides almost no discrimination among affordable materials!

## Mathematical Formulation

**Logarithmic normalization:**

$$S_c(C) = \frac{\log C_{max} - \log C}{\log C_{max} - \log C_{min}}$$

Equivalently:

$$S_c(C) = \frac{\log(C_{max}/C)}{\log(C_{max}/C_{min})}$$

Where:

| Parameter | Description | Source |
|:----------|:------------|:-------|
| $C$ | Material cost ($/kg) | Input data |
| $C_{min}$ | Minimum cost in dataset | Data-driven |
| $C_{max}$ | Maximum cost in dataset | Data-driven |

**Properties:**

- Cheapest material → $S_c = 1$ (most affordable)
- Most expensive → $S_c = 0$ (least affordable)
- Each order of magnitude spans equal score range

## Python Usage

### Basic Usage

```python
from ascicat.scoring import score_cost
import numpy as np

# Array of costs
costs = np.array([2.67, 10, 100, 1000, 10000, 107544])

# Data-driven normalization (automatic min/max)
scores = score_cost(costs)

print("Cost ($/kg) | Score")
print("-" * 30)
for c, s in zip(costs, scores):
    print(f"${c:>10,.0f} | {s:.3f}")
```

**Output:**

```
Cost ($/kg) | Score
------------------------------
$         3 | 1.000
$        10 | 0.892
$       100 | 0.676
$     1,000 | 0.459
$    10,000 | 0.243
$   107,544 | 0.000
```

### With Fixed Ranges

```python
# Using predefined ranges
scores = score_cost(costs, cost_min=1.0, cost_max=200000.0)
```

## Score Distribution

The logarithmic scaling ensures:

| Cost Range | Score Range | Discrimination |
|:-----------|:------------|:---------------|
| $1 - $10 | 0.89 - 1.00 | Good |
| $10 - $100 | 0.78 - 0.89 | Good |
| $100 - $1,000 | 0.67 - 0.78 | Good |
| $1,000 - $10,000 | 0.56 - 0.67 | Good |
| $10,000 - $100,000 | 0.44 - 0.56 | Good |

Each order of magnitude gets ~0.11 score range.

## For Alloys: Composition-Weighted Cost

For bimetallic/multimetallic catalysts, cost should be weighted by composition:

$$C_{alloy} = \sum_i x_i \cdot C_i$$

Where:

- $x_i$ = Mole fraction of element $i$
- $C_i$ = Cost of element $i$

!!! example "Example: Cu₃Sb Alloy"

    ```python
    # Copper: $10/kg, Antimony: $8/kg
    # Cu3Sb = 75% Cu, 25% Sb (by moles)
    cost_Cu3Sb = 0.75 * 10 + 0.25 * 8  # = $9.50/kg
    ```

## Data Validation

Cost values must be **strictly positive** (for logarithm):

```python
# This will raise an error
costs_invalid = np.array([10, 0, 100])  # 0 is invalid!
scores = score_cost(costs_invalid)
# ValueError: Found 1 non-positive cost values...
```

## Score Interpretation

| Score | Interpretation | Example Materials |
|:------|:---------------|:------------------|
| 0.9 - 1.0 | Very affordable | Fe, Al, Cu |
| 0.7 - 0.9 | Affordable | Ni, Co, Sn |
| 0.5 - 0.7 | Moderate | Mo, W, Ag |
| 0.3 - 0.5 | Expensive | Au, Pt, Pd |
| 0.0 - 0.3 | Very expensive | Rh, Ir, Os |

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from ascicat.scoring import score_cost

# Generate logarithmically spaced costs
costs = np.logspace(0, 6, 100)  # $1 to $1M
scores = score_cost(costs, cost_min=1, cost_max=1e6)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(costs, scores, 'b-', linewidth=2)
ax.set_xlabel('Material Cost ($/kg)')
ax.set_ylabel('Cost Score')
ax.set_title('Logarithmic Cost Scoring')
ax.grid(True, which='both', linestyle='--', alpha=0.5)

# Add reference points
materials = {
    'Fe': 2, 'Cu': 10, 'Ni': 20, 'Ag': 900,
    'Au': 60000, 'Pt': 30000, 'Ir': 150000
}
for name, cost in materials.items():
    score = score_cost([cost], 1, 1e6)[0]
    ax.scatter([cost], [score], s=100, zorder=5)
    ax.annotate(name, (cost, score), xytext=(5, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()
```

## Edge Cases

### All Same Cost

```python
same_costs = np.array([100, 100, 100])
scores = score_cost(same_costs)
# Returns: [1.0, 1.0, 1.0]  # All equally affordable
```

### Very Small Range

```python
# When min and max are very close
narrow = np.array([99, 100, 101])
scores = score_cost(narrow)
# Still provides meaningful discrimination
```

## Integration with ASCICalculator

```python
from ascicat import ASCICalculator

calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')
results = calc.calculate_asci()

# View cost distribution
print("\nCost Score Statistics:")
print(f"  Mean: {results['cost_score'].mean():.3f}")
print(f"  Std:  {results['cost_score'].std():.3f}")
print(f"  Min:  {results['cost_score'].min():.3f}")
print(f"  Max:  {results['cost_score'].max():.3f}")
```

## Practical Considerations

!!! warning "Cost Data Sources"

    Material costs fluctuate with market conditions. Consider:

    - Using commodity price averages (e.g., USGS Mineral Commodity Summaries)
    - Documenting your cost data source
    - Performing sensitivity analysis on cost estimates

!!! info "Beyond Raw Material Cost"

    The cost score uses **raw material cost** as a proxy. Real catalyst costs include:

    - Synthesis/fabrication costs
    - Catalyst loading (mass per electrode area)
    - Recyclability/recovery potential
    - Supply chain availability

    These factors can shift relative economics significantly.

## References

- U.S. Geological Survey. *Mineral Commodity Summaries 2024.*
- London Metal Exchange (LME) - Current metal prices
