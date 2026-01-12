# Core Concepts

This page explains the theoretical foundation and key principles behind ASCICat.

## The Multi-Objective Challenge

Traditional catalyst screening focuses on a single objective: **activity**. The classic volcano plot identifies catalysts with optimal binding energies. However, real-world catalyst selection must balance:

1. **Activity** - Can it catalyze the reaction efficiently?
2. **Stability** - Will it survive under operating conditions?
3. **Cost** - Is it economically viable at scale?

!!! quote "The Fundamental Trade-off"

    A catalyst might be highly active (like platinum) but prohibitively expensive. Another might be cheap (like iron) but unstable. ASCICat provides a systematic framework to navigate these trade-offs.

## The Sabatier Principle

The foundation of activity scoring is the **Sabatier principle** (1911):

> *An optimal catalyst binds reaction intermediates neither too strongly nor too weakly.*

```
Activity
   ↑
   |        /\
   |       /  \
   |      /    \
   |     /      \
   |____/________\____→ Binding Energy
        weak   optimal   strong
```

**Mathematical formulation:**

$$S_a(\Delta E) = \max\left(0, 1 - \frac{|\Delta E - \Delta E_{opt}|}{\sigma_a}\right)$$

Where:

- $\Delta E$ = Adsorption energy (from DFT calculations)
- $\Delta E_{opt}$ = Sabatier-optimal energy (reaction-specific)
- $\sigma_a$ = Activity tolerance (typically 0.15 eV)

## Surface Energy and Stability

Catalyst stability correlates with **surface energy** ($\gamma$):

- **Low $\gamma$** → Strong metal-metal bonds → Resistant to dissolution
- **High $\gamma$** → Weaker bonding → Prone to reconstruction

**Inverse linear normalization:**

$$S_s(\gamma) = \frac{\gamma_{max} - \gamma}{\gamma_{max} - \gamma_{min}}$$

This ensures:

- Lowest surface energy → $S_s = 1$ (most stable)
- Highest surface energy → $S_s = 0$ (least stable)

## Economic Considerations

Material costs span 5+ orders of magnitude:

| Material | Cost ($/kg) | Log₁₀(Cost) |
|:---------|:------------|:------------|
| Iron | ~2 | 0.3 |
| Copper | ~10 | 1.0 |
| Silver | ~900 | 2.95 |
| Gold | ~60,000 | 4.78 |
| Platinum | ~30,000 | 4.48 |
| Iridium | ~150,000 | 5.18 |

**Logarithmic normalization** handles this range:

$$S_c(C) = \frac{\log C_{max} - \log C}{\log C_{max} - \log C_{min}}$$

!!! info "Why Logarithmic?"

    Linear scaling would make all precious metals indistinguishable (all near zero). Logarithmic scaling preserves discrimination across the full cost spectrum.

## The ASCI Integration

The **Activity-Stability-Cost Index** combines all three scores:

$$\phi_{ASCI} = w_a \cdot S_a + w_s \cdot S_s + w_c \cdot S_c$$

### Properties

1. **Bounded**: $\phi_{ASCI} \in [0, 1]$
2. **Interpretable**: Higher = better
3. **Customizable**: Weights reflect priorities
4. **Transparent**: Each component is traceable

### Weight Constraint

$$w_a + w_s + w_c = 1$$

This ensures:

- Scores are directly comparable
- Maximum possible ASCI is 1.0
- Weights represent true relative importance

## Why Not Just Use Pareto?

Pareto frontier analysis identifies **non-dominated solutions** - catalysts where no other catalyst is better on all objectives. However:

| Pareto Analysis | ASCI |
|:----------------|:-----|
| Produces a *set* of solutions | Produces a *ranked list* |
| No preference required | Explicit preferences (weights) |
| Hard to compare across studies | Reproducible comparison |
| Requires subjective final selection | Deterministic ranking |

!!! success "Complementary Approaches"

    ASCICat works **alongside** Pareto analysis. Top ASCI-ranked catalysts are predominantly Pareto-optimal, validating both methodologies.

## Scoring Methods

### Linear Scoring (Default)

```
Score = max(0, 1 - |deviation| / tolerance)
```

**Advantages:**

- Computationally efficient
- Easy to interpret
- Consistent with volcano plots

### Gaussian Scoring (Alternative)

```
Score = exp(-(deviation)² / (2 × tolerance²))
```

**Advantages:**

- Smoother discrimination
- Never reaches exactly zero
- Sharper peak at optimum

## Data-Driven Normalization

For stability and cost scores, ASCICat uses **data-driven normalization**:

$$S = \frac{x_{max} - x}{x_{max} - x_{min}}$$

Where $x_{max}$ and $x_{min}$ are computed from **your dataset**.

!!! warning "Important"

    This means scores are relative to your specific dataset, not absolute. The same catalyst might have different scores in different datasets.

## Key Assumptions

ASCICat makes these assumptions:

1. **DFT accuracy** - Calculated binding energies are reliable proxies for experimental values
2. **Surface energy correlation** - Lower surface energy indicates better stability
3. **Material cost proxy** - Bulk material costs reflect catalyst fabrication costs
4. **Linear combination** - Trade-offs can be captured by weighted sums
5. **Score independence** - Activity, stability, and cost can be scored separately

## Limitations

Be aware of these limitations:

- **Kinetic barriers** - ASCI focuses on thermodynamics, not kinetics
- **Support effects** - Metal-support interactions not captured
- **Electrolyte effects** - pH, ion concentration not considered
- **Mass transport** - Only intrinsic properties considered
- **Deactivation mechanisms** - Specific degradation pathways not modeled

## Further Reading

- [Activity Scoring](scoring/activity.md) - Detailed activity score documentation
- [Stability Scoring](scoring/stability.md) - Surface energy scoring
- [Cost Scoring](scoring/cost.md) - Economic viability scoring
- [Scientific Background](../science/theory.md) - Theoretical foundations
