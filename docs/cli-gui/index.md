# CLI & GUI

ASCICat provides multiple interfaces for different use cases.

## Available Interfaces

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } **Command Line Interface**

    ---

    For scripting, automation, and headless servers

    [:octicons-arrow-right-24: CLI Guide](cli.md)

-   :material-application:{ .lg .middle } **Graphical Interface**

    ---

    Interactive analysis with point-and-click controls

    [:octicons-arrow-right-24: GUI Guide](gui.md)

</div>

## Quick Start

### CLI

```bash
# Basic HER screening
ascicat --reaction HER --data data/HER_clean.csv --output results/

# CO2RR with custom weights
ascicat --reaction CO2RR --pathway CO --weights 0.5 0.3 0.2

# Help
ascicat --help
```

### GUI

```bash
# Launch graphical interface
ascicat-gui
```

## Comparison

| Feature | CLI | GUI |
|:--------|:----|:----|
| Interactive exploration | Limited | Full |
| Scripting/automation | Excellent | Limited |
| Batch processing | Yes | No |
| Remote/headless use | Yes | No |
| Learning curve | Moderate | Low |
