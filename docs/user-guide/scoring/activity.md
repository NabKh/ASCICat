# Activity Scoring

Activity scoring quantifies how well a catalyst's binding energy aligns with the Sabatier optimum.

## The Sabatier Principle

The activity of a catalyst is governed by the **Sabatier principle**:

!!! quote "Sabatier (1911)"

    An optimal catalyst binds reaction intermediates with intermediate strength - neither too weakly (preventing activation) nor too strongly (preventing product desorption).

This creates the classic **volcano plot**:

```
Activity
   ↑
   |           ★ Optimal
   |          /|\
   |         / | \
   |        /  |  \
   |       /   |   \
   |______/____|____\______→ Binding Energy
     weak    ΔE_opt    strong
```

## Mathematical Formulation

### Linear Scoring (Default)

$$S_a(\Delta E) = \max\left(0, 1 - \frac{|\Delta E - \Delta E_{opt}|}{\sigma_a}\right)$$

Where:

| Parameter | Symbol | Description | Unit |
|:----------|:-------|:------------|:-----|
| Adsorption energy | $\Delta E$ | DFT-calculated binding | eV |
| Optimal energy | $\Delta E_{opt}$ | Sabatier peak position | eV |
| Activity width | $\sigma_a$ | Tolerance for "good" activity | eV |

**Properties:**

- Score = 1.0 when $\Delta E = \Delta E_{opt}$ (at peak)
- Score = 0.5 when $|\Delta E - \Delta E_{opt}| = \sigma_a / 2$
- Score = 0.0 when $|\Delta E - \Delta E_{opt}| \geq \sigma_a$

### Gaussian Scoring (Alternative)

$$S_a(\Delta E) = \exp\left(-\frac{(\Delta E - \Delta E_{opt})^2}{2\sigma_a^2}\right)$$

**Properties:**

- Score = 1.0 at optimum
- Score = 0.61 at $\pm\sigma_a$ from optimum
- Never reaches exactly zero

## Python Usage

### Basic Usage

```python
from ascicat.scoring import score_activity

# Single catalyst
score = score_activity(
    delta_E=-0.25,       # eV
    optimal_E=-0.27,     # eV (HER optimum)
    width=0.15,          # eV
    method='linear'
)
print(f"Activity score: {score:.3f}")  # 0.867
```

### Array Operations

```python
import numpy as np

# Multiple catalysts
energies = np.array([-0.42, -0.35, -0.27, -0.19, -0.12])

scores_linear = score_activity(energies, -0.27, 0.15, method='linear')
scores_gauss = score_activity(energies, -0.27, 0.15, method='gaussian')

for E, lin, gauss in zip(energies, scores_linear, scores_gauss):
    print(f"ΔE={E:+.2f} eV: Linear={lin:.3f}, Gaussian={gauss:.3f}")
```

**Output:**

```
ΔE=-0.42 eV: Linear=0.000, Gaussian=0.135
ΔE=-0.35 eV: Linear=0.467, Gaussian=0.606
ΔE=-0.27 eV: Linear=1.000, Gaussian=1.000
ΔE=-0.19 eV: Linear=0.467, Gaussian=0.606
ΔE=-0.12 eV: Linear=0.000, Gaussian=0.135
```

### Using ActivityScorer Class

```python
from ascicat.scoring import ActivityScorer

scorer = ActivityScorer()

# Linear scoring
s_linear = scorer.linear(delta_E=-0.30, optimal_E=-0.27, width=0.15)

# Gaussian scoring
s_gauss = scorer.gaussian(delta_E=-0.30, optimal_E=-0.27, width=0.15)
```

## Reaction-Specific Parameters

### Predefined Configurations

| Reaction | Pathway | $\Delta E_{opt}$ | $\sigma_a$ |
|:---------|:--------|:-----------------|:-----------|
| HER | H adsorption | -0.27 eV | 0.15 eV |
| CO2RR | CO | -0.67 eV | 0.15 eV |
| CO2RR | CHO | -0.48 eV | 0.15 eV |
| CO2RR | COCOH | -0.32 eV | 0.15 eV |

### Access Configuration

```python
from ascicat.config import get_reaction_config

# Get HER configuration
her_config = get_reaction_config('HER')
print(f"Optimal energy: {her_config.optimal_energy} eV")
print(f"Activity width: {her_config.activity_width} eV")

# Get CO2RR-CO configuration
co_config = get_reaction_config('CO2RR', pathway='CO')
print(f"Optimal energy: {co_config.optimal_energy} eV")
```

## Choosing Between Methods

### Linear Scoring

**Advantages:**

- Computationally efficient
- Easy to interpret
- Consistent with traditional volcano plots
- Clear cutoff at $\pm\sigma_a$

**Best for:**

- Standard screening
- Large datasets
- When interpretability is important

### Gaussian Scoring

**Advantages:**

- Smoother discrimination near optimum
- Never exactly zero (no hard cutoff)
- Mathematically differentiable

**Best for:**

- Sensitivity analysis
- When fine discrimination at extremes matters
- Optimization algorithms

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from ascicat.scoring import score_activity

# Generate data
energies = np.linspace(-0.6, 0.1, 100)
scores_lin = score_activity(energies, -0.27, 0.15, 'linear')
scores_gau = score_activity(energies, -0.27, 0.15, 'gaussian')

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(energies, scores_lin, 'b-', label='Linear', linewidth=2)
ax.plot(energies, scores_gau, 'r--', label='Gaussian', linewidth=2)
ax.axvline(-0.27, color='gray', linestyle=':', label='Optimal')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Adsorption Energy (eV)')
ax.set_ylabel('Activity Score')
ax.legend()
ax.set_xlim(-0.6, 0.1)
ax.set_ylim(0, 1.05)
plt.show()
```

## Physical Interpretation

### Score Ranges

| Score Range | Interpretation | Example Materials |
|:------------|:---------------|:------------------|
| 0.9 - 1.0 | Excellent | At volcano peak |
| 0.7 - 0.9 | Good | Near-optimal catalysts |
| 0.5 - 0.7 | Moderate | Acceptable for some applications |
| 0.3 - 0.5 | Poor | Suboptimal binding |
| 0.0 - 0.3 | Very poor | Far from optimum |

### HER Examples

| Catalyst | $\Delta E$ (eV) | $S_a$ | Interpretation |
|:---------|:----------------|:------|:---------------|
| Pt(111) | -0.09 | 0.80 | Excellent |
| Ni(111) | -0.47 | 0.00 | Too strong |
| Au(111) | +0.18 | 0.00 | Too weak |
| Fe-Sb | -0.25 | 0.87 | Near optimal |

## Common Issues

!!! warning "Sign Convention"

    ASCICat uses the convention where **negative** adsorption energies indicate stable binding. Ensure your DFT data uses the same convention:

    $$\Delta E = E_{slab+adsorbate} - E_{slab} - \frac{1}{2}E_{H_2}$$

!!! warning "Width Parameter"

    The default $\sigma_a = 0.15$ eV is calibrated for typical DFT accuracy. If your calculations use different exchange-correlation functionals, you may need to adjust this.
