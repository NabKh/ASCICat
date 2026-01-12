# Getting Started

Welcome to ASCICat! This guide will help you get up and running with multi-objective catalyst screening.

## What is ASCICat?

ASCICat (Activity-Stability-Cost Integrated Catalyst Discovery) is a Python framework that enables researchers to:

- **Rank catalysts** based on multiple performance criteria simultaneously
- **Generate high-quality figures** for scientific analysis
- **Perform sensitivity analysis** to understand how weight choices affect rankings
- **Compare results reproducibly** across different studies

## The ASCI Score

The core of ASCICat is the unified ASCI metric:

$$\phi_{ASCI} = w_a \cdot S_a + w_s \cdot S_s + w_c \cdot S_c$$

| Component | Description | Range |
|:----------|:------------|:------|
| $S_a$ | Activity score (Sabatier principle) | [0, 1] |
| $S_s$ | Stability score (surface energy) | [0, 1] |
| $S_c$ | Cost score (economic viability) | [0, 1] |
| $w_a, w_s, w_c$ | Customizable weights | Sum to 1 |

!!! tip "Equal Weights by Default"

    ASCICat uses equal weights (0.33, 0.33, 0.34) by default for unbiased exploratory screening. You can customize these based on your application requirements.

## Quick Links

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install ASCICat via pip or from source

    [:octicons-arrow-right-24: Installation Guide](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Run your first ASCI calculation in 5 minutes

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-test-tube:{ .lg .middle } **First Analysis**

    ---

    Complete walkthrough of an HER screening analysis

    [:octicons-arrow-right-24: First Analysis](first-analysis.md)

</div>

## System Requirements

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| Storage | 100 MB | 500 MB |

## Supported Platforms

- :fontawesome-brands-linux: Linux (Ubuntu, CentOS, Fedora)
- :fontawesome-brands-apple: macOS (10.14+)
- :fontawesome-brands-windows: Windows (10/11)

## Next Steps

1. **[Install ASCICat](installation.md)** - Set up the package
2. **[Quick Start](quickstart.md)** - Run your first analysis
3. **[Explore Tutorials](../tutorials/index.md)** - Learn advanced features
