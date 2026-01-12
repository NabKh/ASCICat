# Theoretical Background

The scientific foundation of the ASCI framework.

## The Catalyst Selection Problem

Computational catalysis databases contain thousands of DFT-calculated catalyst properties. The challenge is translating this data into **actionable experimental priorities**.

Traditional approaches:

1. **Single-objective ranking** - Sort by one metric (e.g., activity)
2. **Manual multi-criteria** - Ad-hoc weighting, poorly documented
3. **Pareto analysis** - Non-dominated set, no ranking

**ASCICat provides:**

- Unified, interpretable metric
- Transparent, reproducible rankings
- Built-in sensitivity analysis

## Three-Pillar Framework

### Activity (Thermodynamic)

Based on the **Sabatier principle** and volcano relationships:

$$S_a(\Delta E) = \max\left(0, 1 - \frac{|\Delta E - \Delta E_{opt}|}{\sigma_a}\right)$$

**Physical basis:**

- DFT binding energies correlate with activation barriers
- Optimal binding balances adsorption and desorption
- Volcano plots validated experimentally

### Stability (Thermodynamic)

Based on **surface thermodynamics**:

$$S_s(\gamma) = \frac{\gamma_{max} - \gamma}{\gamma_{max} - \gamma_{min}}$$

**Physical basis:**

- Lower surface energy → stronger surface bonding
- Correlates with dissolution resistance
- Validated for metal dissolution in electrochemistry

### Cost (Economic)

Based on **material economics**:

$$S_c(C) = \frac{\log C_{max} - \log C}{\log C_{max} - \log C_{min}}$$

**Physical basis:**

- Raw material cost as proxy for catalyst cost
- Logarithmic scaling handles 5+ orders of magnitude
- Economic viability critical for deployment

## Integration: The ASCI Score

Weighted linear combination:

$$\phi_{ASCI} = w_a \cdot S_a + w_s \cdot S_s + w_c \cdot S_c$$

**Properties:**

- Bounded: $\phi \in [0, 1]$
- Interpretable: Higher = better
- Customizable: Weights reflect priorities
- Constraint: $w_a + w_s + w_c = 1$

## Assumptions

1. **DFT accuracy** - Binding energies are meaningful
2. **Descriptor validity** - Chosen descriptors capture key properties
3. **Linear aggregation** - Trade-offs are linear
4. **Independence** - Scores are approximately independent

## Limitations

- Kinetic barriers not directly captured
- Support effects not included
- Electrolyte effects not modeled
- Deactivation mechanisms not predicted

## Validation

ASCICat rankings are validated by:

1. **Pareto consistency** - Top ASCI catalysts are mostly Pareto-optimal
2. **Known benchmarks** - Pt-group metals rank highly for HER
3. **Sensitivity robustness** - Rankings stable across weight ranges
