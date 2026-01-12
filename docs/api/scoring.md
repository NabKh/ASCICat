# Scoring Functions

Functions for calculating activity, stability, and cost scores.

## Module Reference

::: ascicat.scoring
    options:
      show_root_heading: false
      show_source: false

## Quick Reference

### Activity Scoring

```python
from ascicat.scoring import score_activity

# Linear method (default)
S_a = score_activity(
    delta_E=-0.25,
    optimal_E=-0.27,
    width=0.15,
    method='linear'
)

# Gaussian method
S_a = score_activity(
    delta_E=-0.25,
    optimal_E=-0.27,
    width=0.15,
    method='gaussian'
)
```

### Stability Scoring

```python
from ascicat.scoring import score_stability

# Data-driven normalization
gammas = [0.5, 1.0, 2.0, 3.0]
scores = score_stability(gammas)

# With explicit range
scores = score_stability(gammas, gamma_min=0.1, gamma_max=5.0)
```

### Cost Scoring

```python
from ascicat.scoring import score_cost

# Data-driven normalization
costs = [10, 100, 1000, 10000]
scores = score_cost(costs)

# With explicit range
scores = score_cost(costs, cost_min=1, cost_max=200000)
```

### Combined ASCI

```python
from ascicat.scoring import calculate_asci

phi = calculate_asci(
    activity_score=0.85,
    stability_score=0.70,
    cost_score=0.90,
    w_a=0.33,
    w_s=0.33,
    w_c=0.34
)
```

## Function Details

### score_activity

Calculate activity score based on Sabatier principle.

```python
def score_activity(
    delta_E: float | np.ndarray,
    optimal_E: float,
    width: float,
    method: str = 'linear'
) -> float | np.ndarray:
```

**Parameters:**

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `delta_E` | float or array | Adsorption energy (eV) |
| `optimal_E` | float | Sabatier optimum (eV) |
| `width` | float | Activity width σ_a (eV) |
| `method` | str | 'linear' or 'gaussian' |

**Returns:** Score(s) in [0, 1]

**Formulas:**

Linear:
$$S_a = \max(0, 1 - |\Delta E - \Delta E_{opt}| / \sigma_a)$$

Gaussian:
$$S_a = \exp(-(\Delta E - \Delta E_{opt})^2 / (2\sigma_a^2))$$

### score_stability

Calculate stability score from surface energy.

```python
def score_stability(
    gamma: float | np.ndarray,
    gamma_min: float = None,
    gamma_max: float = None
) -> float | np.ndarray:
```

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `gamma` | float or array | Required | Surface energy (J/m²) |
| `gamma_min` | float | None | Min for normalization (auto if None) |
| `gamma_max` | float | None | Max for normalization (auto if None) |

**Returns:** Score(s) in [0, 1]

**Formula:**
$$S_s = (\gamma_{max} - \gamma) / (\gamma_{max} - \gamma_{min})$$

### score_cost

Calculate cost score with logarithmic normalization.

```python
def score_cost(
    cost: float | np.ndarray,
    cost_min: float = None,
    cost_max: float = None
) -> float | np.ndarray:
```

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `cost` | float or array | Required | Material cost ($/kg) |
| `cost_min` | float | None | Min for normalization (auto if None) |
| `cost_max` | float | None | Max for normalization (auto if None) |

**Returns:** Score(s) in [0, 1]

**Formula:**
$$S_c = (\log C_{max} - \log C) / (\log C_{max} - \log C_{min})$$

### calculate_asci

Calculate combined ASCI score.

```python
def calculate_asci(
    activity_score: float | np.ndarray,
    stability_score: float | np.ndarray,
    cost_score: float | np.ndarray,
    w_a: float,
    w_s: float,
    w_c: float
) -> float | np.ndarray:
```

**Parameters:**

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `activity_score` | float or array | Activity score S_a |
| `stability_score` | float or array | Stability score S_s |
| `cost_score` | float or array | Cost score S_c |
| `w_a` | float | Activity weight |
| `w_s` | float | Stability weight |
| `w_c` | float | Cost weight |

**Returns:** ASCI score(s) in [0, 1]

**Formula:**
$$\phi_{ASCI} = w_a \cdot S_a + w_s \cdot S_s + w_c \cdot S_c$$

## ScoringFunctions Class

```python
from ascicat.scoring import ScoringFunctions

scorer = ScoringFunctions()

# Activity scores
S_a = scorer.activity_linear(delta_E, optimal_E, width)
S_a = scorer.activity_gaussian(delta_E, optimal_E, width)

# Stability score
S_s = scorer.stability_inverse_linear(gamma, gamma_min, gamma_max)

# Cost score
S_c = scorer.cost_logarithmic(cost, cost_min, cost_max)

# Combined
phi = scorer.combined_asci_score(S_a, S_s, S_c, w_a, w_s, w_c)
```

## ActivityScorer Class

Specialized class for activity scoring:

```python
from ascicat.scoring import ActivityScorer

scorer = ActivityScorer()

# Linear method
S_a = scorer.linear(delta_E=-0.25, optimal_E=-0.27, width=0.15)

# Gaussian method
S_a = scorer.gaussian(delta_E=-0.25, optimal_E=-0.27, width=0.15)

# With array input
import numpy as np
energies = np.array([-0.42, -0.27, -0.12])
scores = scorer.linear(energies, optimal_E=-0.27, width=0.15)
```
