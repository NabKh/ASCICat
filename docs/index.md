---
hide:
  - navigation
  - toc
---

<style>
.md-typeset h1 {
  display: none;
}
</style>

<div align="center" markdown>

# **ASCICat**

## Activity-Stability-Cost Integrated Catalyst Discovery

*A unified multi-objective framework for translating computational catalyst data into reproducible, experimentally-actionable rankings*

[Get Started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/NabKh/ASCICat){ .md-button }

</div>

---

## The Challenge

The computational catalysis community has generated **massive ML/DFT databases** containing thousands of calculated catalyst properties. However, there exists no standardized framework to translate this wealth of data into actionable experimental priorities.

| Current Challenge | ASCICat Solution |
|:------------------|:-----------------|
| No unified framework for catalyst selection | Standardized ASCI metric applicable to any catalyst dataset |
| Ad-hoc, non-reproducible selection criteria | Transparent weighting with explicit trade-off documentation |
| Results cannot be compared across studies | Common metric enables direct cross-study comparison |
| Hidden assumptions in catalyst ranking | Built-in sensitivity analysis reveals weight dependencies |

---

## The ASCI Framework

ASCICat implements a **three-pillar scoring system** grounded in fundamental catalysis principles:

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Activity Score (S~a~)**

    ---

    Based on the **Sabatier Principle**: optimal binding energy for reaction kinetics. Too weak binding prevents activation; too strong binding prevents desorption.

    $$S_a = \max\left(0, 1 - \frac{|\Delta E - \Delta E_{opt}|}{\sigma_a}\right)$$

-   :material-shield-check:{ .lg .middle } **Stability Score (S~s~)**

    ---

    Based on **Surface Thermodynamics**: lower surface energy indicates stronger metal-metal bonding and enhanced resistance to dissolution.

    $$S_s = \frac{\gamma_{max} - \gamma}{\gamma_{max} - \gamma_{min}}$$

-   :material-currency-usd:{ .lg .middle } **Cost Score (S~c~)**

    ---

    Based on **Economic Viability**: logarithmic normalization handles the enormous range in material costs while maintaining discrimination.

    $$S_c = \frac{\log C_{max} - \log C}{\log C_{max} - \log C_{min}}$$

</div>

### The Unified ASCI Metric

$$\phi_{ASCI} = w_a \cdot S_a + w_s \cdot S_s + w_c \cdot S_c$$

where $w_a + w_s + w_c = 1$ and all scores $S_i \in [0, 1]$

---

## Quick Example

```python
from ascicat import ASCICalculator

# Initialize for HER reaction
calc = ASCICalculator(reaction='HER')

# Load your DFT data
calc.load_data('data/HER_clean.csv')

# Calculate ASCI scores with custom weights
results = calc.calculate_asci(
    w_a=0.4,  # 40% Activity
    w_s=0.3,  # 30% Stability
    w_c=0.3   # 30% Cost
)

# Get top-ranked catalysts
top_catalysts = calc.get_top_catalysts(n=10)
print(top_catalysts[['symbol', 'ASCI', 'activity_score']])
```

**Output:**
```
      symbol     ASCI  activity_score
0      Fe2Sb4   0.899           0.923
1       Cu3Sb   0.887           0.912
2      Cu6Sb2   0.876           0.889
...
```

---

## Supported Reactions

| Reaction | Pathway | Optimal $\Delta E$ | Description |
|:---------|:--------|:-------------------|:------------|
| **HER** | H adsorption | -0.27 eV | Hydrogen Evolution Reaction |
| **CO2RR** | CO | -0.67 eV | Carbon monoxide production |
| **CO2RR** | CHO | -0.48 eV | Methanol pathway |
| **CO2RR** | COCOH | -0.32 eV | Formic acid pathway |

---

## Key Features

<div class="grid cards" markdown>

-   :material-tune:{ .lg .middle } **Customizable Weights**

    ---

    Balance activity, stability, and cost according to your application requirements

-   :material-chart-scatter-plot:{ .lg .middle } **High-Quality Figures**

    ---

    Generate 600 DPI figures including 3D Pareto spaces, volcano plots, and sensitivity diagrams

-   :material-chart-bell-curve:{ .lg .middle } **Sensitivity Analysis**

    ---

    Ternary diagrams, bootstrap confidence intervals, and variance-based sensitivity indices

-   :material-rocket-launch:{ .lg .middle } **High-Throughput Ready**

    ---

    Process datasets with 50,000+ catalysts with automatic stratified sampling

-   :material-application:{ .lg .middle } **Multiple Interfaces**

    ---

    Python API, command-line interface, and graphical user interface

-   :material-scale-balance:{ .lg .middle } **Pareto Complementarity**

    ---

    Works alongside Pareto frontier methods for comprehensive analysis

</div>

---

## Scientific Foundation

ASCICat is built on established theoretical foundations:

!!! quote "References"

    - Nørskov, J. K. et al. *Towards the computational design of solid catalysts.* **Nat. Chem.** 1, 37 (2009)
    - Greeley, J. et al. *Computational high-throughput screening of electrocatalytic materials.* **Nat. Mater.** 5, 909 (2006)
    - Sabatier, P. *Hydrogénations et déshydrogénations par catalyse.* **Ber. Dtsch. Chem. Ges.** 44, 1984 (1911)

---

## Installation

=== "pip"

    ```bash
    pip install ascicat
    ```

=== "From Source"

    ```bash
    git clone https://github.com/NabKh/ASCICat.git
    cd ASCICat
    pip install -e .
    ```

=== "With GUI"

    ```bash
    pip install ascicat[gui]
    ```

---

<div align="center" markdown>

**Ready to start screening catalysts?**

[Get Started :material-arrow-right:](getting-started/index.md){ .md-button .md-button--primary }

---

*Developed at the Dutch Institute for Fundamental Energy Research (DIFFER)*

**Author:** N. Khossossi | **Contact:** [n.khossossi@differ.nl](mailto:n.khossossi@differ.nl)

</div>
