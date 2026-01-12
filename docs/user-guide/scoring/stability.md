# Stability Scoring

Stability scoring quantifies the thermodynamic resistance of a catalyst surface to dissolution and reconstruction.

## Physical Basis

**Surface energy** ($\gamma$) is the energy required to create a unit area of surface:

$$\gamma = \frac{E_{slab} - n \cdot E_{bulk}}{2A}$$

Where:

- $E_{slab}$ = Total energy of the slab
- $E_{bulk}$ = Energy per atom in bulk
- $n$ = Number of atoms
- $A$ = Surface area
- Factor of 2 accounts for two surfaces

### Stability Correlation

Lower surface energy indicates:

- **Stronger metal-metal bonds** at the surface
- **Reduced tendency to reconstruct**
- **Better resistance to dissolution**
- **Longer operational lifetime**

## Mathematical Formulation

ASCICat uses **inverse linear normalization**:

$$S_s(\gamma) = \frac{\gamma_{max} - \gamma}{\gamma_{max} - \gamma_{min}}$$

Where:

| Parameter | Description | Source |
|:----------|:------------|:-------|
| $\gamma$ | Surface energy (J/m²) | DFT calculation |
| $\gamma_{min}$ | Minimum in dataset | Data-driven |
| $\gamma_{max}$ | Maximum in dataset | Data-driven |

**Properties:**

- Lowest $\gamma$ → $S_s = 1$ (most stable)
- Highest $\gamma$ → $S_s = 0$ (least stable)
- Linear interpolation between extremes

## Python Usage

### Basic Usage

```python
from ascicat.scoring import score_stability
import numpy as np

# Single value
gamma = 1.5  # J/m²
# Need reference range for normalization
score = score_stability(gamma, gamma_min=0.5, gamma_max=4.0)
print(f"Stability score: {score:.3f}")  # 0.714
```

### Data-Driven Normalization (Recommended)

```python
# Array of surface energies
gammas = np.array([0.52, 1.0, 2.0, 3.0, 4.5])

# Automatic min/max from data
scores = score_stability(gammas)
print("Surface Energy | Stability Score")
print("-" * 35)
for g, s in zip(gammas, scores):
    print(f"    {g:.2f} J/m²   |      {s:.3f}")
```

**Output:**

```
Surface Energy | Stability Score
-----------------------------------
    0.52 J/m²   |      1.000
    1.00 J/m²   |      0.880
    2.00 J/m²   |      0.628
    3.00 J/m²   |      0.377
    4.50 J/m²   |      0.000
```

### With Fixed Ranges

```python
# Using configuration-defined ranges
from ascicat.config import HER_CONFIG

gamma_min, gamma_max = HER_CONFIG.stability_range
scores = score_stability(gammas, gamma_min, gamma_max)
```

## Typical Surface Energy Values

### Pure Metals (Low-Index Surfaces)

| Metal | (111) γ (J/m²) | (100) γ (J/m²) | (110) γ (J/m²) |
|:------|:---------------|:---------------|:---------------|
| Pt | 0.52 | 0.58 | 0.67 |
| Pd | 0.49 | 0.54 | 0.62 |
| Au | 0.32 | 0.37 | 0.42 |
| Ag | 0.35 | 0.40 | 0.46 |
| Cu | 0.58 | 0.65 | 0.75 |
| Ni | 0.62 | 0.70 | 0.81 |
| Co | 0.76 | 0.85 | 0.98 |
| Fe | 0.78 | 0.88 | 1.02 |

!!! note "Surface Orientation"

    Close-packed surfaces (111 for FCC, 110 for BCC) have lower surface energies and higher stability.

### General Ranges

| Category | γ Range (J/m²) | Examples |
|:---------|:---------------|:---------|
| Very stable | 0.3 - 0.6 | Au, Ag, Pt |
| Stable | 0.6 - 1.2 | Cu, Ni, Pd |
| Moderate | 1.2 - 2.0 | Co, Fe, Mo |
| Less stable | 2.0 - 3.5 | W, Ta, open surfaces |
| Unstable | > 3.5 | Highly stepped surfaces |

## Data Validation

ASCICat validates surface energy inputs:

```python
# Negative values trigger warning
gammas_invalid = np.array([0.5, -0.2, 1.5])  # -0.2 is invalid
scores = score_stability(gammas_invalid)
# Warning: Found 1 negative surface energy values...
# Negative values clipped to zero
```

!!! warning "Physical Validity"

    Surface energies must be **non-negative** (thermodynamic requirement). Negative values in your data may indicate calculation errors.

## Score Interpretation

| Score Range | Stability | Practical Implications |
|:------------|:----------|:-----------------------|
| 0.9 - 1.0 | Excellent | Long operational lifetime |
| 0.7 - 0.9 | Good | Suitable for most applications |
| 0.5 - 0.7 | Moderate | May require protective measures |
| 0.3 - 0.5 | Poor | Limited durability |
| 0.0 - 0.3 | Very poor | Rapid degradation expected |

## Edge Cases

### All Identical Values

```python
# When all surface energies are the same
identical = np.array([1.5, 1.5, 1.5])
scores = score_stability(identical)
# Returns: [1.0, 1.0, 1.0]  # All equally "most stable"
```

### Single Value with Data-Driven

```python
# Single value without reference range
single = np.array([1.5])
scores = score_stability(single)
# Returns: [1.0]  # Single value is its own best
```

## Integration with ASCICalculator

When using `ASCICalculator`, stability scoring is handled automatically:

```python
from ascicat import ASCICalculator

calc = ASCICalculator(reaction='HER')
calc.load_data('data/HER_clean.csv')

# Stability scores computed automatically
results = calc.calculate_asci()

# Access individual scores
print(results[['symbol', 'surface_energy', 'stability_score']].head())
```

## Limitations

!!! info "What Stability Scoring Doesn't Capture"

    - **Electrochemical dissolution** - pH and potential effects
    - **Adsorbate-induced reconstruction** - Surface rearrangement under operation
    - **Oxidation/corrosion** - Chemical degradation
    - **Poisoning** - Irreversible adsorption of contaminants
    - **Agglomeration** - Nanoparticle sintering

Surface energy is a **thermodynamic descriptor** that correlates with stability but doesn't capture all degradation mechanisms.

## References

- Hansen, H. A. et al. *Surface energies of late transition metals.* Phys. Chem. Chem. Phys. 10, 3722 (2008)
- Vitos, L. et al. *The surface energy of metals.* Surf. Sci. 411, 186 (1998)
