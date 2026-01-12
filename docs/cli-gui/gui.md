# Graphical User Interface

The ASCICat GUI provides interactive point-and-click analysis.

## Installation

The GUI requires additional dependencies:

```bash
pip install ascicat[gui]
```

## Launch

```bash
ascicat-gui
```

Or from Python:

```python
from ascicat.gui import launch_gui
launch_gui()
```

## Interface Overview

The GUI consists of four main panels:

### 1. Data Panel

- **Load Data**: Select CSV file
- **Preview**: View first 100 rows
- **Validate**: Check data format
- **Statistics**: View descriptor ranges

### 2. Configuration Panel

- **Reaction**: Select HER or CO2RR
- **Pathway**: For CO2RR, select pathway
- **Weights**: Slider controls for w_a, w_s, w_c
- **Method**: Linear or Gaussian scoring

### 3. Results Panel

- **Calculate**: Run ASCI calculation
- **Table**: View ranked catalysts
- **Filter**: Search/filter results
- **Export**: Save to CSV/Excel

### 4. Visualization Panel

- **3D Plot**: Interactive score space
- **Volcano**: Activity relationship
- **Breakdown**: Score components
- **Export**: Save figures

## Features

### Interactive Weight Adjustment

- Drag sliders to change weights
- Real-time ranking updates
- Weights automatically normalize to 1.0

### Data Exploration

- Sort by any column
- Filter by catalyst name
- Color-code by score
- Select catalysts for comparison

### Figure Generation

- Preview before export
- Adjust DPI (150-1200)
- Multiple formats (PNG, PDF, SVG)
- Batch export all figures

### Sensitivity Analysis

- Launch sensitivity wizard
- Configure weight grid
- View results in real-time
- Export ternary diagrams

## Keyboard Shortcuts

| Shortcut | Action |
|:---------|:-------|
| `Ctrl+O` | Open data file |
| `Ctrl+S` | Save results |
| `Ctrl+E` | Export figures |
| `Ctrl+R` | Run calculation |
| `Ctrl+Q` | Quit |
| `F5` | Refresh view |
| `F11` | Toggle fullscreen |

## Requirements

- PyQt5 >= 5.15.0
- pyqtgraph >= 0.12.0
- OpenGL (for 3D visualization)

## Troubleshooting

### GUI Won't Launch

```bash
# Check PyQt5 installation
python -c "from PyQt5.QtWidgets import QApplication"

# Install if missing
pip install PyQt5
```

### 3D Plot Issues

```bash
# Install OpenGL dependencies (Linux)
sudo apt-get install libgl1-mesa-glx

# macOS
brew install mesa
```

### Display Issues

```bash
# Force software rendering
export QT_QUICK_BACKEND=software
ascicat-gui
```
