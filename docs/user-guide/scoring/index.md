# Scoring System

ASCICat uses a three-pillar scoring system to evaluate catalysts comprehensively.

## Overview

Each catalyst receives three normalized scores:

| Score | Symbol | Based On | Range |
|:------|:-------|:---------|:------|
| Activity | $S_a$ | Adsorption energy $\Delta E$ | [0, 1] |
| Stability | $S_s$ | Surface energy $\gamma$ | [0, 1] |
| Cost | $S_c$ | Material cost $C$ | [0, 1] |

These scores are combined into the unified ASCI metric:

$$\phi_{ASCI} = w_a \cdot S_a + w_s \cdot S_s + w_c \cdot S_c$$

## Scoring Philosophy

### Why Normalize to [0, 1]?

1. **Comparability**: All scores use the same scale
2. **Interpretability**: 0 = worst, 1 = best (intuitive)
3. **Weighted combination**: Weights directly reflect importance
4. **Bounded output**: ASCI is also in [0, 1]

### Data-Driven vs. Fixed Ranges

| Approach | Stability/Cost | Pros | Cons |
|:---------|:---------------|:-----|:-----|
| Data-driven | min/max from dataset | Full [0,1] range used | Scores not comparable across datasets |
| Fixed ranges | Predetermined values | Cross-study comparison | Some scores may cluster |

ASCICat uses **data-driven normalization** by default for stability and cost scores.

## Detailed Documentation

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Activity Scoring**

    ---

    Sabatier principle implementation with linear and Gaussian methods

    [:octicons-arrow-right-24: Activity Score](activity.md)

-   :material-shield-check:{ .lg .middle } **Stability Scoring**

    ---

    Surface energy-based stability assessment

    [:octicons-arrow-right-24: Stability Score](stability.md)

-   :material-currency-usd:{ .lg .middle } **Cost Scoring**

    ---

    Logarithmic economic viability scoring

    [:octicons-arrow-right-24: Cost Score](cost.md)

-   :material-sigma:{ .lg .middle } **ASCI Integration**

    ---

    Combining scores into the unified metric

    [:octicons-arrow-right-24: ASCI Integration](asci.md)

</div>

## Quick Reference

### Activity Score

```python
from ascicat.scoring import score_activity

# Linear method (default)
S_a = score_activity(
    delta_E=-0.25,       # Your adsorption energy
    optimal_E=-0.27,     # Sabatier optimum
    width=0.15,          # Tolerance
    method='linear'
)
# Returns: 0.867
```

### Stability Score

```python
from ascicat.scoring import score_stability

# Data-driven normalization
gammas = [0.52, 1.0, 2.0, 3.0, 4.5]
S_s = score_stability(gammas)
# Returns: [1.0, 0.88, 0.63, 0.38, 0.0]
```

### Cost Score

```python
from ascicat.scoring import score_cost

# Logarithmic normalization
costs = [10, 100, 1000, 10000]
S_c = score_cost(costs)
# Returns: [1.0, 0.67, 0.33, 0.0]
```

### Combined ASCI

```python
from ascicat.scoring import calculate_asci

phi = calculate_asci(
    activity_score=0.85,
    stability_score=0.70,
    cost_score=0.90,
    w_a=0.33, w_s=0.33, w_c=0.34
)
# Returns: 0.818
```

## Validation

All scores are validated:

- **Range checking**: Inputs must be physical (e.g., cost > 0)
- **Weight validation**: Must sum to 1.0
- **Output clipping**: Scores guaranteed in [0, 1]

```python
from ascicat.config import validate_weights

# This passes
validate_weights(0.33, 0.33, 0.34)

# This raises ValueError
validate_weights(0.5, 0.3, 0.3)  # Sum = 1.1
```
