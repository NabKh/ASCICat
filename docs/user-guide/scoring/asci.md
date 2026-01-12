# ASCI Integration

The Activity-Stability-Cost Index (ASCI) combines three individual scores into a unified catalyst ranking metric.

## The ASCI Formula

$$\phi_{ASCI} = w_a \cdot S_a + w_s \cdot S_s + w_c \cdot S_c$$

Where:

| Symbol | Component | Range | Weight |
|:-------|:----------|:------|:-------|
| $S_a$ | Activity score | [0, 1] | $w_a$ |
| $S_s$ | Stability score | [0, 1] | $w_s$ |
| $S_c$ | Cost score | [0, 1] | $w_c$ |

**Constraint:** $w_a + w_s + w_c = 1$

## Properties

### Bounded Output

Since all $S_i \in [0, 1]$ and weights sum to 1:

$$\phi_{ASCI} \in [0, 1]$$

- **$\phi_{ASCI} = 1$**: Perfect catalyst (S_a = S_s = S_c = 1)
- **$\phi_{ASCI} = 0$**: Worst catalyst (all scores = 0)

### Interpretability

The ASCI score directly answers: *"On a 0-1 scale, how good is this catalyst?"*

### Transparency

Each component's contribution is explicit:

```
ASCI = 0.75
     = 0.33 × 0.90 (activity)
     + 0.33 × 0.70 (stability)
     + 0.34 × 0.65 (cost)
```

## Python Usage

### Direct Calculation

```python
from ascicat.scoring import calculate_asci

phi = calculate_asci(
    activity_score=0.90,
    stability_score=0.70,
    cost_score=0.65,
    w_a=0.33,
    w_s=0.33,
    w_c=0.34
)
print(f"ASCI: {phi:.3f}")  # 0.749
```

### Array Operations

```python
import numpy as np

# Multiple catalysts
S_a = np.array([0.9, 0.7, 0.5, 0.3])
S_s = np.array([0.8, 0.9, 0.7, 0.6])
S_c = np.array([0.6, 0.5, 0.95, 0.98])

# Calculate ASCI for all
asci = calculate_asci(S_a, S_s, S_c, 0.33, 0.33, 0.34)
print("ASCI scores:", asci.round(3))
# [0.764, 0.694, 0.718, 0.630]
```

### Using ScoringFunctions Class

```python
from ascicat.scoring import ScoringFunctions

scorer = ScoringFunctions()
asci = scorer.combined_asci_score(
    activity_score=0.85,
    stability_score=0.75,
    cost_score=0.80,
    w_a=0.4, w_s=0.35, w_c=0.25
)
```

## Weight Selection

### Default: Equal Weights

```python
# Unbiased exploratory screening
w_a, w_s, w_c = 0.33, 0.33, 0.34
```

!!! success "When to Use"

    - Initial screening with no prior preference
    - Comparing catalysts without bias
    - Exploratory research

### Activity-Focused

```python
# Performance-critical applications
w_a, w_s, w_c = 0.50, 0.30, 0.20
```

!!! success "When to Use"

    - Fundamental research
    - When stability is not limiting
    - Performance benchmarking

### Stability-Focused

```python
# Durability-critical applications
w_a, w_s, w_c = 0.30, 0.50, 0.20
```

!!! success "When to Use"

    - Industrial applications
    - Harsh operating conditions
    - Long-term operation required

### Cost-Focused

```python
# Economic viability critical
w_a, w_s, w_c = 0.30, 0.20, 0.50
```

!!! success "When to Use"

    - Large-scale deployment
    - Commercial applications
    - Earth-abundant materials priority

## Weight Validation

Weights must satisfy constraints:

```python
from ascicat.config import validate_weights

# Valid weights
validate_weights(0.33, 0.33, 0.34)  # OK
validate_weights(0.5, 0.3, 0.2)     # OK
validate_weights(0.0, 0.5, 0.5)     # OK (zero weight allowed)

# Invalid weights
validate_weights(0.5, 0.3, 0.3)     # Error: sum = 1.1
validate_weights(-0.1, 0.6, 0.5)    # Error: negative weight
validate_weights(0.5, 0.3, 0.1)     # Error: sum = 0.9
```

### Weight Normalization

For approximate weights, use normalization:

```python
from ascicat.config import normalize_weights

# Input weights that don't sum to 1
w_a, w_s, w_c = normalize_weights(1, 1, 1)
print(w_a, w_s, w_c)  # 0.333, 0.333, 0.333
```

## Score Validation

Input scores are validated:

```python
import numpy as np

# Scores outside [0, 1] trigger warning
activity = np.array([0.9, 1.2, 0.8])  # 1.2 is out of range

asci = calculate_asci(activity, [0.7]*3, [0.8]*3, 0.33, 0.33, 0.34)
# Warning: Activity scores outside [0, 1] range...
# Scores clipped to [0, 1]
```

## Component Analysis

### Decomposing ASCI

```python
# For a catalyst with ASCI = 0.75
S_a, S_s, S_c = 0.90, 0.70, 0.65
w_a, w_s, w_c = 0.33, 0.33, 0.34

# Contributions
contrib_a = w_a * S_a  # 0.297
contrib_s = w_s * S_s  # 0.231
contrib_c = w_c * S_c  # 0.221

print(f"Activity contribution:  {contrib_a:.3f} ({contrib_a/0.749*100:.1f}%)")
print(f"Stability contribution: {contrib_s:.3f} ({contrib_s/0.749*100:.1f}%)")
print(f"Cost contribution:      {contrib_c:.3f} ({contrib_c/0.749*100:.1f}%)")
```

### Identifying Limiting Factor

```python
import pandas as pd

# For top catalyst
top = results.iloc[0]

# Which score is limiting?
scores = {
    'Activity': top['activity_score'],
    'Stability': top['stability_score'],
    'Cost': top['cost_score']
}

limiting = min(scores, key=scores.get)
print(f"Limiting factor: {limiting} ({scores[limiting]:.3f})")
```

## ASCI vs. Individual Scores

| Scenario | $S_a$ | $S_s$ | $S_c$ | ASCI | Interpretation |
|:---------|:------|:------|:------|:-----|:---------------|
| Balanced | 0.8 | 0.8 | 0.8 | 0.80 | Good all-rounder |
| Activity star | 1.0 | 0.5 | 0.5 | 0.67 | Limited by stability/cost |
| Stability star | 0.5 | 1.0 | 0.5 | 0.67 | Limited by activity/cost |
| Cost star | 0.5 | 0.5 | 1.0 | 0.67 | Limited by performance |
| Extreme | 0.9 | 0.9 | 0.3 | 0.70 | Cost is limiting |

## Ranking Algorithm

ASCICat ranks catalysts by ASCI in descending order:

```python
# Results are automatically sorted
results = calc.calculate_asci()

# Rank 1 = highest ASCI
top_catalyst = results.iloc[0]
print(f"#1: {top_catalyst['symbol']} (ASCI = {top_catalyst['ASCI']:.3f})")
```

### Tie-Breaking

For identical ASCI scores (rare), ranking is arbitrary. For reproducibility, the original data order is preserved.

## Best Practices

!!! tip "Document Your Weights"

    Always document:

    1. Weight values used
    2. Rationale for weight selection
    3. Any sensitivity analysis performed

!!! tip "Compare Apples to Apples"

    Don't compare ASCI scores from different weight scenarios directly. A catalyst with ASCI = 0.7 under equal weights is not comparable to ASCI = 0.8 under activity-focused weights.

!!! tip "Use Sensitivity Analysis"

    Before finalizing weights, run sensitivity analysis to understand how rankings change across the weight space.
