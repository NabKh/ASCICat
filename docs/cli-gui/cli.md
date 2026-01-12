# Command Line Interface

The ASCICat CLI provides command-line access to all functionality.

## Installation

The CLI is installed automatically with ASCICat:

```bash
pip install ascicat
```

## Basic Usage

```bash
ascicat [OPTIONS] COMMAND [ARGS]
```

## Commands

### Main Screening

```bash
# HER screening with default weights
ascicat --reaction HER --data data/HER_clean.csv --output results/

# CO2RR screening with custom weights
ascicat --reaction CO2RR --pathway CO \
        --data data/CO2RR_CO_clean.csv \
        --weights 0.4 0.35 0.25 \
        --output results/

# Show top N catalysts
ascicat --reaction HER --data data.csv --top 20
```

### Options

| Option | Short | Description | Default |
|:-------|:------|:------------|:--------|
| `--reaction` | `-r` | Reaction type (HER, CO2RR) | Required |
| `--pathway` | `-p` | CO2RR pathway (CO, CHO, COCOH) | CO |
| `--data` | `-d` | Input CSV file | Required |
| `--output` | `-o` | Output directory | results/ |
| `--weights` | `-w` | Weights (w_a w_s w_c) | 0.33 0.33 0.34 |
| `--top` | `-n` | Show top N catalysts | 10 |
| `--figures` | `-f` | Generate figures | False |
| `--interactive` | `-i` | Generate interactive plots | False |
| `--verbose` | `-v` | Verbose output | False |
| `--help` | `-h` | Show help | - |

## Examples

### Basic Screening

```bash
# HER with equal weights
ascicat -r HER -d data/HER_clean.csv -o results/HER/

# Show top 50
ascicat -r HER -d data/HER_clean.csv --top 50
```

### Custom Weights

```bash
# Activity-focused (50/30/20)
ascicat -r HER -d data/HER_clean.csv -w 0.5 0.3 0.2 -o results/

# Cost-focused (30/20/50)
ascicat -r HER -d data/HER_clean.csv -w 0.3 0.2 0.5 -o results/
```

### With Figures

```bash
# Generate all figures
ascicat -r HER -d data/HER_clean.csv -o results/ --figures

# Include interactive 3D
ascicat -r HER -d data/HER_clean.csv -o results/ --figures --interactive
```

### CO2RR Pathways

```bash
# CO pathway
ascicat -r CO2RR -p CO -d data/CO2RR_CO.csv -o results/CO/

# CHO pathway
ascicat -r CO2RR -p CHO -d data/CO2RR_CHO.csv -o results/CHO/

# COCOH pathway
ascicat -r CO2RR -p COCOH -d data/CO2RR_COCOH.csv -o results/COCOH/
```

## Batch Processing

### Multiple Files

```bash
# Process all CO2RR pathways
for pathway in CO CHO COCOH; do
    ascicat -r CO2RR -p $pathway \
            -d data/CO2RR_${pathway}_clean.csv \
            -o results/CO2RR_${pathway}/ \
            --figures
done
```

### Weight Sensitivity

```bash
# Run with multiple weight scenarios
for wa in 0.2 0.33 0.5; do
    ws=$(echo "scale=2; (1 - $wa) / 2" | bc)
    wc=$(echo "scale=2; 1 - $wa - $ws" | bc)
    ascicat -r HER -d data/HER_clean.csv \
            -w $wa $ws $wc \
            -o results/weights_${wa}/
done
```

## Output

### Files Generated

```
results/
├── ASCI_results.csv        # Full results
├── top_catalysts.csv       # Top N catalysts
├── summary.txt             # Summary statistics
├── panel_a_3d_pareto.png   # 3D score space
├── panel_b_rank_vs_ads.png # Activity relationship
├── panel_c_volcano.png     # Optimization landscape
├── panel_d_breakdown.png   # Score breakdown
└── interactive_3d.html     # Interactive plot
```

### CSV Output Format

```csv
symbol,ASCI,activity_score,stability_score,cost_score,DFT_ads_E,...
Fe2Sb4,0.923,0.915,0.891,0.962,-0.257,...
Cu3Sb,0.908,0.896,0.884,0.944,-0.285,...
...
```

## Exit Codes

| Code | Meaning |
|:-----|:--------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Data validation error |
| 4 | File not found |

## Environment Variables

| Variable | Description |
|:---------|:------------|
| `ASCICAT_DATA_DIR` | Default data directory |
| `ASCICAT_OUTPUT_DIR` | Default output directory |
| `ASCICAT_DPI` | Default figure DPI |

```bash
export ASCICAT_DATA_DIR=/path/to/data
ascicat -r HER -d HER_clean.csv  # Uses ASCICAT_DATA_DIR
```
