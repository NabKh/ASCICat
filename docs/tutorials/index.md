# Tutorials

Step-by-step tutorials demonstrating ASCICat workflows for common use cases.

## Available Tutorials

<div class="grid cards" markdown>

-   :material-flask:{ .lg .middle } **HER Catalyst Screening**

    ---

    Complete workflow for hydrogen evolution reaction screening

    [:octicons-arrow-right-24: HER Tutorial](her-screening.md)

-   :material-molecule-co2:{ .lg .middle } **CO2RR Multi-Pathway Analysis**

    ---

    Screen catalysts across multiple CO₂ reduction pathways

    [:octicons-arrow-right-24: CO2RR Tutorial](co2rr-analysis.md)

-   :material-chart-box:{ .lg .middle } **Figure Generation**

    ---

    Generate high-quality figures for your research

    [:octicons-arrow-right-24: Figures Tutorial](figure-generation.md)

-   :material-chart-bell-curve:{ .lg .middle } **Sensitivity Analysis**

    ---

    Analyze weight dependencies and ranking robustness

    [:octicons-arrow-right-24: Sensitivity Tutorial](sensitivity-analysis.md)

-   :material-scale-balance:{ .lg .middle } **ASCICat vs Pareto**

    ---

    Compare ASCI ranking with Pareto frontier analysis

    [:octicons-arrow-right-24: Comparison Tutorial](ascicat-vs-pareto.md)

</div>

## Interactive Notebooks

Run tutorials directly in your browser:

| Tutorial | Colab | Binder |
|:---------|:------|:-------|
| Introduction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/01_Introduction_Quick_Start.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NabKh/ASCICat/master?labpath=tutorials/01_Introduction_Quick_Start.ipynb) |
| HER Screening | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/02_HER_Catalyst_Screening.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NabKh/ASCICat/master?labpath=tutorials/02_HER_Catalyst_Screening.ipynb) |
| CO2RR Analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/03_CO2RR_Multi_Pathway_Analysis.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NabKh/ASCICat/master?labpath=tutorials/03_CO2RR_Multi_Pathway_Analysis.ipynb) |
| Visualization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/04_Visualization_Guide.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NabKh/ASCICat/master?labpath=tutorials/04_Visualization_Guide.ipynb) |
| Sensitivity | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NabKh/ASCICat/blob/master/tutorials/05_Sensitivity_Analysis.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NabKh/ASCICat/master?labpath=tutorials/05_Sensitivity_Analysis.ipynb) |

## Prerequisites

Before starting the tutorials:

1. **Install ASCICat**
   ```bash
   pip install ascicat
   ```

2. **Download sample data** (included with package)

3. **Basic Python knowledge** - familiarity with pandas, numpy, matplotlib

## Learning Path

### Beginner

1. Start with **HER Catalyst Screening** - basic workflow
2. Learn **Figure Generation** - visualization basics

### Intermediate

3. Explore **CO2RR Multi-Pathway** - advanced screening
4. Master **Sensitivity Analysis** - robustness evaluation

### Advanced

5. Compare **ASCICat vs Pareto** - methodological insights
6. Create **Custom Reactions** - extend to new chemistries

## Example Files

All tutorials have corresponding Python scripts in `examples/`:

```
examples/
├── example_1_HER_screening.py
├── example_2_CO2RR_screening.py
├── example_3_visualization.py
├── example_4_sensitivity_analysis.py
└── example_5_ascicat_vs_pareto.py
```

Run any example:

```bash
cd ASCICat
python examples/example_1_HER_screening.py
```
