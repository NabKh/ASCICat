# API Reference

Complete API documentation for ASCICat.

## Main Classes

<div class="grid cards" markdown>

-   :material-calculator:{ .lg .middle } **ASCICalculator**

    ---

    Main calculator class for ASCI scoring

    [:octicons-arrow-right-24: ASCICalculator](calculator.md)

-   :material-function:{ .lg .middle } **Scoring Functions**

    ---

    Activity, stability, and cost scoring

    [:octicons-arrow-right-24: Scoring](scoring.md)

-   :material-chart-bar:{ .lg .middle } **Visualizer**

    ---

    High-quality figure generation

    [:octicons-arrow-right-24: Visualizer](visualizer.md)

-   :material-magnify:{ .lg .middle } **Analyzer**

    ---

    Statistical analysis utilities

    [:octicons-arrow-right-24: Analyzer](analyzer.md)

-   :material-chart-bell-curve:{ .lg .middle } **Sensitivity**

    ---

    Weight sensitivity analysis

    [:octicons-arrow-right-24: Sensitivity](sensitivity.md)

-   :material-cog:{ .lg .middle } **Configuration**

    ---

    Reaction configurations and constants

    [:octicons-arrow-right-24: Configuration](config.md)

</div>

## Quick Import Reference

```python
# Main calculator
from ascicat import ASCICalculator

# Visualization
from ascicat.visualizer import Visualizer

# Sensitivity analysis
from ascicat.sensitivity import (
    SensitivityAnalyzer,
    SensitivityVisualizer,
    run_enhanced_sensitivity_analysis
)

# Configuration
from ascicat.config import (
    get_reaction_config,
    ReactionConfig,
    ASCIConstants,
    validate_weights
)

# Scoring functions
from ascicat.scoring import (
    ScoringFunctions,
    ActivityScorer,
    score_activity,
    score_stability,
    score_cost,
    calculate_asci
)

# Analysis
from ascicat.analyzer import Analyzer

# Version info
from ascicat import __version__
from ascicat.version import print_version_info, print_citation
```

## Class Hierarchy

```
ascicat
├── ASCICalculator          # Main entry point
├── config
│   ├── ReactionConfig      # Reaction parameters
│   ├── ASCIConstants       # Global constants
│   └── get_reaction_config # Config retrieval
├── scoring
│   ├── ScoringFunctions    # All scoring methods
│   └── ActivityScorer      # Activity-specific
├── visualizer
│   └── Visualizer          # Figure generation
├── analyzer
│   └── Analyzer            # Statistical analysis
└── sensitivity
    ├── SensitivityAnalyzer # Weight sensitivity
    └── SensitivityVisualizer # Sensitivity plots
```

## Common Patterns

### Basic Workflow

```python
from ascicat import ASCICalculator

calc = ASCICalculator(reaction='HER')
calc.load_data('data.csv')
results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)
top = calc.get_top_catalysts(n=10)
```

### With Visualization

```python
from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer

calc = ASCICalculator(reaction='HER')
calc.load_data('data.csv')
results = calc.calculate_asci()

viz = Visualizer(results, calc.config)
viz.generate_all_outputs(output_dir='figures/')
```

### With Sensitivity Analysis

```python
from ascicat import ASCICalculator
from ascicat.sensitivity import run_enhanced_sensitivity_analysis

calc = ASCICalculator(reaction='HER')
calc.load_data('data.csv')

results = run_enhanced_sensitivity_analysis(
    calculator=calc,
    output_dir='sensitivity/'
)
```

## Type Annotations

ASCICat uses type hints throughout:

```python
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

def calculate_asci(
    w_a: float,
    w_s: float,
    w_c: float,
    show_progress: bool = True
) -> pd.DataFrame:
    ...
```
