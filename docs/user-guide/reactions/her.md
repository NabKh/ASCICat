# Hydrogen Evolution Reaction (HER)

The Hydrogen Evolution Reaction is the cathodic reaction in water electrolysis for green hydrogen production.

## Reaction

$$2H^+ + 2e^- \rightarrow H_2$$

In alkaline conditions:

$$2H_2O + 2e^- \rightarrow H_2 + 2OH^-$$

## Key Descriptor

The activity descriptor is the **hydrogen adsorption energy** ($\Delta E_H$):

$$\Delta E_H = E_{surface+H} - E_{surface} - \frac{1}{2}E_{H_2}$$

## Sabatier Optimum

Based on the pioneering work of Nørskov and coworkers:

!!! quote "Optimal Binding"

    $\Delta E_{opt} = -0.27$ eV

    Catalysts at this energy show thermoneutral hydrogen binding - the ideal balance between H⁺ adsorption and H₂ desorption.

## Configuration

```python
from ascicat.config import HER_CONFIG

print(f"Optimal energy: {HER_CONFIG.optimal_energy} eV")
print(f"Activity width: {HER_CONFIG.activity_width} eV")
print(f"Activity window: {HER_CONFIG.activity_window} eV")
```

### Full Configuration

| Parameter | Value | Description |
|:----------|:------|:------------|
| `optimal_energy` | -0.27 eV | Sabatier peak |
| `activity_width` | 0.15 eV | Scoring tolerance |
| `activity_window` | [-0.60, +0.10] eV | Analysis range |
| `stability_range` | [0.1, 5.0] J/m² | Surface energy range |
| `cost_range` | [$1, $200,000] | Material cost range |

## Usage Example

```python
from ascicat import ASCICalculator

# Initialize for HER
calc = ASCICalculator(reaction='HER', verbose=True)

# Load HER data
calc.load_data('data/HER_clean.csv')

# Calculate with equal weights
results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

# Top catalysts
top = calc.get_top_catalysts(n=10)
print(top[['symbol', 'ASCI', 'DFT_ads_E']])
```

## Benchmark Materials

Known HER catalysts and their binding energies:

| Material | $\Delta E_H$ (eV) | Activity | Cost |
|:---------|:------------------|:---------|:-----|
| Pt(111) | -0.09 | Excellent | Very High |
| Ni(111) | -0.47 | Moderate | Low |
| Co(111) | -0.35 | Good | Low |
| Au(111) | +0.18 | Poor | High |
| Fe(110) | -0.40 | Moderate | Very Low |
| MoS₂ | -0.08 | Good | Low |

## The HER Volcano

```
Exchange
Current
   ↑
   |        Pt★
   |       /  \
   |    Pd/    \Ir
   |   Rh/      \
   |   /         \Au  Ag
   | Ni  Mo  W    \
   |/              \
   |________________\_____→ ΔE_H (eV)
  strong          weak
  binding        binding
```

## Screening Strategy

For HER catalyst discovery:

1. **Equal weights** for initial exploration
2. Focus on catalysts with $|\Delta E_H - (-0.27)| < 0.15$ eV
3. Prioritize stability for industrial applications
4. Consider cost for large-scale deployment

### Weight Recommendations

| Application | $(w_a, w_s, w_c)$ | Rationale |
|:------------|:------------------|:----------|
| Fundamental research | (0.5, 0.3, 0.2) | Focus on mechanism |
| Lab-scale electrolyzer | (0.4, 0.4, 0.2) | Balance performance/durability |
| Commercial electrolyzer | (0.35, 0.40, 0.25) | Emphasize durability |
| Large-scale deployment | (0.35, 0.30, 0.35) | Consider economics |

## Scientific References

- Nørskov, J. K. et al. *J. Electrochem. Soc.* **152**, J23 (2005)
- Greeley, J. et al. *Nat. Mater.* **5**, 909 (2006)
- Seh, Z. W. et al. *Science* **355**, eaad4998 (2017)
