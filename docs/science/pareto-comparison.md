# Comparison with Pareto Analysis

How ASCICat relates to Pareto frontier methods.

## Overview

| Aspect | Pareto | ASCICat |
|:-------|:-------|:--------|
| Output | Non-dominated set | Ranked list |
| Preferences | Not required | Explicit weights |
| Comparability | Within dataset | Across studies |
| Final selection | Subjective | Deterministic |

## Pareto Frontier

A solution is **Pareto-optimal** if no other solution is better on all objectives.

```
Stability
    ↑
    |   ★ Pareto frontier
    |  ★ ★
    | ★   ★
    |★     ★
    |       ★
    |_________★___→ Activity
```

## Complementary Methods

### Pareto First, ASCI Second

1. Identify Pareto-optimal catalysts
2. Apply ASCI ranking within Pareto set
3. Get prioritized list of non-dominated solutions

```python
# Find Pareto-optimal
pareto_mask = is_pareto_optimal(objectives)
pareto_results = results[pareto_mask]

# Rank within Pareto set
pareto_ranked = pareto_results.sort_values('ASCI', ascending=False)
```

### Validation

Top ASCI catalysts should be predominantly Pareto-optimal:

- If not → scoring functions may be miscalibrated
- Typical overlap: 80-95% of top 100

## When to Use Each

### Use Pareto When:

- Exploring trade-off space
- Stakeholders disagree on priorities
- No clear weight rationale
- Generating options for discussion

### Use ASCI When:

- Priorities can be quantified
- Need single prioritized list
- Cross-study comparison needed
- Reproducibility required

## Empirical Observations

From HER catalyst screening:

| Metric | Value |
|:-------|:------|
| Total catalysts | 48,312 |
| Pareto-optimal | ~2,000 (4%) |
| Top 100 ASCI in Pareto | ~90% |
| Top 10 ASCI in Pareto | ~100% |

## Key Insight

!!! success "Complementarity"

    ASCICat and Pareto methods are **complementary, not competing**.

    - Pareto shows the possible trade-offs
    - ASCI provides actionable priorities

## References

- Miettinen, K. *Nonlinear Multiobjective Optimization* (Springer, 1998)
- Ehrgott, M. *Multicriteria Optimization* (Springer, 2005)
