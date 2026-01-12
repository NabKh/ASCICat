# Multi-Objective Optimization

How ASCICat handles multiple objectives.

## The Challenge

Catalyst selection involves competing objectives:

- **Maximize** activity
- **Maximize** stability
- **Minimize** cost

No single catalyst optimizes all simultaneously.

## Approaches

### 1. Weighted Sum (ASCICat)

$$\phi = \sum_i w_i \cdot S_i$$

**Advantages:**

- Single interpretable metric
- Reproducible rankings
- Explicit trade-offs

**Limitations:**

- Non-convex Pareto regions missed
- Weight selection required

### 2. Pareto Optimization

Find non-dominated solutions where no other solution is better on all objectives.

**Advantages:**

- No weight specification needed
- Complete trade-off information

**Limitations:**

- Returns set, not ranking
- Requires subjective final selection

### 3. Goal Programming

Minimize deviation from targets:

$$\min \sum_i |S_i - S_i^{target}|$$

**Advantages:**

- Aspiration-based
- Handles constraints

**Limitations:**

- Targets often arbitrary

## Why Weighted Sum?

ASCICat uses weighted sum because:

1. **Interpretability** - Weights directly reflect priorities
2. **Reproducibility** - Same weights = same ranking
3. **Comparability** - Enables cross-study comparison
4. **Simplicity** - Easy to understand and implement

## Weight Selection

### Default: Equal Weights

$$w_a = w_s = w_c \approx 0.33$$

Unbiased starting point for exploration.

### Application-Specific

| Application | Suggested Weights |
|:------------|:------------------|
| Fundamental research | (0.5, 0.3, 0.2) |
| Industrial catalyst | (0.35, 0.40, 0.25) |
| Large-scale deployment | (0.30, 0.25, 0.45) |

### Sensitivity Analysis

Don't trust single weight choice - explore the weight space!

## Mathematical Properties

### Convex Combination

For $w_i \geq 0$ and $\sum w_i = 1$:

- $\phi \in [\min S_i, \max S_i]$
- Linear in scores
- Continuous in weights

### Pareto Optimality

Weighted sum solutions are Pareto-optimal when:

- All weights positive
- Pareto frontier is convex

## Trade-off Analysis

For two catalysts A and B:

$$\Delta \phi = w_a(S_a^A - S_a^B) + w_s(S_s^A - S_s^B) + w_c(S_c^A - S_c^B)$$

A is preferred when $\Delta \phi > 0$.

## References

- Marler, R. T. & Arora, J. S. *Struct. Multidiscip. Optim.* **26**, 369 (2004)
- Hwang, C.-L. & Masud, A. S. M. *Multiple Objective Decision Making* (Springer, 1979)
