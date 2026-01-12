#!/usr/bin/env python3
"""
ascicat_gui.py
Professional Graphical User Interface for ASCICat

Beautiful PyQt5-based GUI for Activity-Stability-Cost Integrated catalyst screening.
Designed for researchers, students, and anyone who prefers visual interaction.

Features:
- Modern, intuitive interface with tabbed organization
- Drag-and-drop file loading
- Real-time parameter adjustment with instant feedback
- Interactive matplotlib plots embedded in GUI
- Live results preview and filtering
- One-click figure export (high-resolution)
- Progress tracking and status updates
- Comprehensive tooltips and help
- Dark/Light theme support
- Session saving and loading

Author: N. Khossossi
Institution: DIFFER (Dutch Institute for Fundamental Energy Research)

Requirements:
    pip install PyQt5 matplotlib numpy pandas

Usage:
    python ascicat_gui.py
    
    Or from command line:
    ascicat-gui
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime

# ============================================================================
# FIX: Add parent directory to Python path to find ascicat package
# ============================================================================
script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Check PyQt5 availability
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
        QCheckBox, QFileDialog, QMessageBox, QProgressBar, QTabWidget,
        QTableWidget, QTableWidgetItem, QTextEdit, QGroupBox, QSlider,
        QSplitter, QStatusBar, QMenuBar, QAction, QDialog, QDialogButtonBox,
        QGridLayout, QFrame, QScrollArea
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
    from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Check other dependencies
try:
    import numpy as np
    import pandas as pd
    NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False

# Check ASCICat availability - TRY MULTIPLE IMPORT STRATEGIES
ASCICAT_AVAILABLE = False
try:
    # Strategy 1: Try normal package import
    from ascicat import (
        ASCICalculator,
        Visualizer,
        Analyzer,
        get_reaction_config,
        list_available_reactions,
        __version__
    )
    ASCICAT_AVAILABLE = True
    print("✓ ASCICat imported successfully (package)")
except ImportError:
    try:
        # Strategy 2: Try importing from parent directory
        sys.path.insert(0, str(repo_root))
        from ascicat import (
            ASCICalculator,
            Visualizer,
            Analyzer,
            get_reaction_config,
            list_available_reactions,
            __version__
        )
        ASCICAT_AVAILABLE = True
        print("✓ ASCICat imported successfully (local)")
    except ImportError as e:
        print(f"❌ Failed to import ASCICat: {e}")
        ASCICAT_AVAILABLE = False


# ============================================================================
# CHECK ALL DEPENDENCIES
# ============================================================================

ALL_DEPENDENCIES_AVAILABLE = (
    PYQT5_AVAILABLE and 
    MATPLOTLIB_AVAILABLE and 
    NUMPY_PANDAS_AVAILABLE and 
    ASCICAT_AVAILABLE
)

if not ALL_DEPENDENCIES_AVAILABLE:
    missing = []
    if not PYQT5_AVAILABLE:
        missing.append("PyQt5 (GUI framework)")
    if not MATPLOTLIB_AVAILABLE:
        missing.append("matplotlib (plotting)")
    if not NUMPY_PANDAS_AVAILABLE:
        missing.append("numpy/pandas (data processing)")
    if not ASCICAT_AVAILABLE:
        missing.append("ascicat (core package)")
    
    print("\n" + "="*80)
    print("❌ ASCICat GUI - Missing Dependencies")
    print("="*80)
    print("\nThe following dependencies are missing:")
    for dep in missing:
        print(f"  • {dep}")
    print("\nInstall with:")
    print("  pip install PyQt5 matplotlib numpy pandas")
    print("\nASCICat should already be installed if you ran 'pip install -e .'")
    print("Current Python path:")
    for p in sys.path[:5]:
        print(f"  - {p}")
    print("\n" + "="*80)
    
    if not PYQT5_AVAILABLE:
        sys.exit(1)  # Can't continue without PyQt5


# ============================================================================
# CALCULATION WORKER THREAD
# ============================================================================

class CalculationWorker(QThread):
    """
    Worker thread for ASCI calculations.
    
    Runs calculations in background to keep GUI responsive.
    Emits signals for progress updates and completion.
    """
    
    # Signals
    progress = pyqtSignal(int, str)  # (percentage, status_message)
    finished = pyqtSignal(object)    # (results_dataframe)
    error = pyqtSignal(str)          # (error_message)
    
    def __init__(self, 
                 reaction: str,
                 data_file: str,
                 pathway: Optional[str],
                 weights: tuple,
                 method: str):
        """
        Initialize calculation worker.
        
        Parameters
        ----------
        reaction : str
            'HER' or 'CO2RR'
        data_file : str
            Path to CSV data file
        pathway : str or None
            CO2RR pathway
        weights : tuple
            (w_a, w_s, w_c)
        method : str
            'linear' or 'gaussian'
        """
        super().__init__()
        self.reaction = reaction
        self.data_file = data_file
        self.pathway = pathway
        self.weights = weights
        self.method = method
        self.calculator = None
    
    def run(self):
        """Execute calculation in background thread."""
        try:
            # Step 1: Initialize calculator
            self.progress.emit(10, "Initializing calculator...")
            self.calculator = ASCICalculator(
                reaction=self.reaction,
                pathway=self.pathway,
                scoring_method=self.method,
                verbose=False
            )
            
            # Step 2: Load data
            self.progress.emit(30, "Loading catalyst data...")
            self.calculator.load_data(self.data_file, validate=True)
            
            # Step 3: Calculate ASCI
            self.progress.emit(50, "Calculating ASCI scores...")
            w_a, w_s, w_c = self.weights
            results = self.calculator.calculate_asci(
                w_a=w_a, w_s=w_s, w_c=w_c,
                method=self.method,
                show_progress=False
            )
            
            # Step 4: Complete
            self.progress.emit(100, "Calculation complete!")
            self.finished.emit(results)
        
        except Exception as e:
            self.error.emit(str(e))


# ============================================================================
# CUSTOM MATPLOTLIB CANVAS
# ============================================================================

class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas for embedding plots in PyQt5.
    
    Provides interactive plotting with zoom, pan, and save capabilities.
    """
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Initialize canvas.
        
        Parameters
        ----------
        parent : QWidget
            Parent widget
        width : float
            Figure width in inches
        height : float
            Figure height in inches
        dpi : int
            Dots per inch
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

# ============================================================================
# MAIN GUI WINDOW
# ============================================================================

class ASCICatGUI(QMainWindow):
    """
    Main ASCICat GUI Application.
    
    Professional interface for catalyst screening with:
    - File loading and data preview
    - Parameter configuration
    - Real-time calculation
    - Interactive visualization
    - Results export
    """
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        
        # Application state
        self.calculator = None
        self.results = None
        self.current_file = None
        self.worker = None
        
        # Initialize UI
        self.init_ui()
        
        # Apply stylesheet
        self.apply_stylesheet()
        
        # Show welcome message
        self.statusBar().showMessage("Ready to screen catalysts! Load data to begin.", 5000)
    
    def init_ui(self):
        """Initialize user interface."""
        
        # Window properties
        self.setWindowTitle(f"ASCICat v{__version__} - Catalyst Screening Toolkit")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Create main content (tabbed interface)
        tabs = self.create_tabs()
        main_layout.addWidget(tabs)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_menu_bar(self):
        """Create menu bar with File, Tools, and Help menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QAction('📂 Open Data File...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_data_file)
        file_menu.addAction(open_action)
        
        save_action = QAction('💾 Save Results...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('❌ Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        calc_action = QAction('⚡ Run Calculation', self)
        calc_action.setShortcut('Ctrl+R')
        calc_action.triggered.connect(self.run_calculation)
        tools_menu.addAction(calc_action)
        
        export_action = QAction('🎨 Export Figures...', self)
        export_action.triggered.connect(self.export_figures)
        tools_menu.addAction(export_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('ℹ️  About ASCICat', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        docs_action = QAction('📚 Documentation', self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        cite_action = QAction('📖 Citation', self)
        cite_action.triggered.connect(self.show_citation)
        help_menu.addAction(cite_action)
    
    def create_header(self):
        """Create header with logo and title."""
        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        header.setStyleSheet("background-color: #2C3E50; border-radius: 10px;")
        
        layout = QVBoxLayout(header)
        
        # Title
        title = QLabel("🔬 ASCICat - Activity-Stability-Cost Integrated Catalyst Discovery")
        title.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Multi-Objective Electrocatalyst Screening Toolkit")
        subtitle.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # Author info
        author = QLabel(f"Version {__version__} | N. Khossossi | DIFFER")
        author.setStyleSheet("color: #BDC3C7; font-size: 11px;")
        author.setAlignment(Qt.AlignCenter)
        layout.addWidget(author)
        
        return header
    
    def create_tabs(self):
        """Create tabbed interface for different sections."""
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #BDC3C7;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #ECF0F1;
                padding: 10px 20px;
                margin-right: 5px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #3498DB;
                color: white;
                font-weight: bold;
            }
        """)
        
        # Tab 1: Setup & Configuration
        setup_tab = self.create_setup_tab()
        tabs.addTab(setup_tab, "📁 1. Data & Setup")
        
        # Tab 2: Calculation
        calc_tab = self.create_calculation_tab()
        tabs.addTab(calc_tab, "⚡ 2. Calculate ASCI")
        
        # Tab 3: Results
        results_tab = self.create_results_tab()
        tabs.addTab(results_tab, "🏆 3. Results")
        
        # Tab 4: Visualization
        viz_tab = self.create_visualization_tab()
        tabs.addTab(viz_tab, "📊 4. Visualization")
        
        return tabs
    
    def create_setup_tab(self):
        """Create setup tab for data loading and configuration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # ====================================================================
        # DATA LOADING SECTION
        # ====================================================================
        
        data_group = QGroupBox("📂 Data Loading")
        data_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        data_layout = QVBoxLayout(data_group)
        
        # File selection
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        file_layout.addWidget(self.file_label)
        
        browse_btn = QPushButton("📂 Browse...")
        browse_btn.clicked.connect(self.load_data_file)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        file_layout.addWidget(browse_btn)
        
        data_layout.addLayout(file_layout)
        
        # Data preview
        preview_label = QLabel("Data Preview:")
        preview_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        data_layout.addWidget(preview_label)
        
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setMaximumHeight(150)
        self.data_preview.setPlaceholderText("Load a CSV file to see data preview...")
        data_layout.addWidget(self.data_preview)
        
        layout.addWidget(data_group)
        
        # ====================================================================
        # REACTION CONFIGURATION SECTION
        # ====================================================================
        
        config_group = QGroupBox("⚙️ Reaction Configuration")
        config_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        config_layout = QGridLayout(config_group)
        
        # Reaction type
        config_layout.addWidget(QLabel("Reaction:"), 0, 0)
        self.reaction_combo = QComboBox()
        self.reaction_combo.addItems(["HER", "CO2RR"])
        self.reaction_combo.currentTextChanged.connect(self.on_reaction_changed)
        self.reaction_combo.setToolTip("Select reaction type:\n"
                                      "• HER: Hydrogen Evolution (2H⁺ + 2e⁻ → H₂)\n"
                                      "• CO2RR: CO₂ Reduction (multiple pathways)")
        config_layout.addWidget(self.reaction_combo, 0, 1)
        
        # Pathway (for CO2RR)
        config_layout.addWidget(QLabel("Pathway:"), 0, 2)
        self.pathway_combo = QComboBox()
        self.pathway_combo.addItems(["CO", "CHO", "COCOH"])
        self.pathway_combo.setEnabled(False)
        self.pathway_combo.setToolTip("CO2RR pathway:\n"
                                     "• CO: CO₂ → CO (ΔE_opt = -0.67 eV)\n"
                                     "• CHO: CO₂ → CH₃OH (ΔE_opt = -0.48 eV)\n"
                                     "• COCOH: CO₂ → HCOOH (ΔE_opt = -0.32 eV)")
        config_layout.addWidget(self.pathway_combo, 0, 3)
        
        # Scoring method
        config_layout.addWidget(QLabel("Method:"), 1, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Linear", "Gaussian"])
        self.method_combo.setToolTip("Activity scoring method:\n"
                                    "• Linear: Default, computationally efficient\n"
                                    "• Gaussian: Sharper discrimination near optimum")
        config_layout.addWidget(self.method_combo, 1, 1)
        
        layout.addWidget(config_group)
        
        # ====================================================================
        # WEIGHT CONFIGURATION SECTION
        # ====================================================================
        
        weight_group = QGroupBox("⚖️ Weight Configuration")
        weight_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        weight_layout = QVBoxLayout(weight_group)
        
        # Weight presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        equal_btn = QPushButton("Equal (0.33, 0.33, 0.34)")
        equal_btn.clicked.connect(lambda: self.set_weights(0.33, 0.33, 0.34))
        preset_layout.addWidget(equal_btn)
        
        activity_btn = QPushButton("Activity (0.5, 0.3, 0.2)")
        activity_btn.clicked.connect(lambda: self.set_weights(0.5, 0.3, 0.2))
        preset_layout.addWidget(activity_btn)
        
        stability_btn = QPushButton("Stability (0.3, 0.5, 0.2)")
        stability_btn.clicked.connect(lambda: self.set_weights(0.3, 0.5, 0.2))
        preset_layout.addWidget(stability_btn)
        
        cost_btn = QPushButton("Cost (0.3, 0.2, 0.5)")
        cost_btn.clicked.connect(lambda: self.set_weights(0.3, 0.2, 0.5))
        preset_layout.addWidget(cost_btn)
        
        weight_layout.addLayout(preset_layout)
        
        # Individual weight sliders
        slider_layout = QGridLayout()
        
        # Activity weight
        slider_layout.addWidget(QLabel("🔥 Activity:"), 0, 0)
        self.w_a_slider = QSlider(Qt.Horizontal)
        self.w_a_slider.setMinimum(0)
        self.w_a_slider.setMaximum(100)
        self.w_a_slider.setValue(33)
        self.w_a_slider.valueChanged.connect(self.update_weights)
        slider_layout.addWidget(self.w_a_slider, 0, 1)
        self.w_a_label = QLabel("0.33")
        self.w_a_label.setMinimumWidth(40)
        slider_layout.addWidget(self.w_a_label, 0, 2)
        
        # Stability weight
        slider_layout.addWidget(QLabel("⚛️ Stability:"), 1, 0)
        self.w_s_slider = QSlider(Qt.Horizontal)
        self.w_s_slider.setMinimum(0)
        self.w_s_slider.setMaximum(100)
        self.w_s_slider.setValue(33)
        self.w_s_slider.valueChanged.connect(self.update_weights)
        slider_layout.addWidget(self.w_s_slider, 1, 1)
        self.w_s_label = QLabel("0.33")
        self.w_s_label.setMinimumWidth(40)
        slider_layout.addWidget(self.w_s_label, 1, 2)
        
        # Cost weight
        slider_layout.addWidget(QLabel("💰 Cost:"), 2, 0)
        self.w_c_slider = QSlider(Qt.Horizontal)
        self.w_c_slider.setMinimum(0)
        self.w_c_slider.setMaximum(100)
        self.w_c_slider.setValue(34)
        self.w_c_slider.valueChanged.connect(self.update_weights)
        slider_layout.addWidget(self.w_c_slider, 2, 1)
        self.w_c_label = QLabel("0.34")
        self.w_c_label.setMinimumWidth(40)
        slider_layout.addWidget(self.w_c_label, 2, 2)
        
        # Sum indicator
        self.sum_label = QLabel("Sum: 1.00 ✓")
        self.sum_label.setStyleSheet("color: green; font-weight: bold;")
        slider_layout.addWidget(self.sum_label, 3, 1)
        
        weight_layout.addLayout(slider_layout)
        
        layout.addWidget(weight_group)
        
        # Spacer
        layout.addStretch()
        
        return tab
    
    def create_calculation_tab(self):
        """Create calculation tab with run button and progress."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Info section
        info_group = QGroupBox("ℹ️ Calculation Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "<b>The ASCI calculation will:</b><br>"
            "1. Load and validate your catalyst data<br>"
            "2. Score activity based on Sabatier principle<br>"
            "3. Score stability using surface energy<br>"
            "4. Score cost using logarithmic normalization<br>"
            "5. Combine into unified ASCI metric<br>"
            "6. Rank catalysts by overall performance<br><br>"
            "<b>Formula:</b> φ_ASCI = w_a·S_a + w_s·S_s + w_c·S_c<br><br>"
            "Results will include activity, stability, cost, and ASCI scores "
            "for each catalyst in your dataset."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # Configuration summary
        summary_group = QGroupBox("📋 Configuration Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.config_summary = QTextEdit()
        self.config_summary.setReadOnly(True)
        self.config_summary.setMaximumHeight(150)
        self.update_config_summary()
        summary_layout.addWidget(self.config_summary)
        
        layout.addWidget(summary_group)
        
        # Run button
        self.run_button = QPushButton("⚡ RUN CALCULATION")
        self.run_button.setMinimumHeight(60)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95A5A6;
            }
        """)
        self.run_button.clicked.connect(self.run_calculation)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)
        
        # Progress section
        progress_group = QGroupBox("⏳ Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.calc_progress = QProgressBar()
        self.calc_progress.setValue(0)
        progress_layout.addWidget(self.calc_progress)
        
        self.status_label = QLabel("Ready to calculate")
        self.status_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Spacer
        layout.addStretch()
        
        return tab
    
    def create_results_tab(self):
        """Create results tab with table and statistics."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Statistics section
        stats_group = QGroupBox("📊 Statistics")
        stats_layout = QHBoxLayout(stats_group)

        self.stats_label = QLabel("No results yet. Run calculation first.")
        self.stats_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        # Results table
        table_group = QGroupBox("🏆 Top Catalysts")
        table_layout = QVBoxLayout(table_group)

        # Number of results to show
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Show top:"))

        self.top_spin = QSpinBox()
        self.top_spin.setMinimum(5)
        self.top_spin.setMaximum(100)
        self.top_spin.setValue(10)
        self.top_spin.setSuffix(" catalysts")
        self.top_spin.valueChanged.connect(self.update_results_table)
        top_layout.addWidget(self.top_spin)

        top_layout.addStretch()

        export_table_btn = QPushButton("💾 Export Table")
        export_table_btn.clicked.connect(self.export_results_table)
        top_layout.addWidget(export_table_btn)

        table_layout.addLayout(top_layout)

        # Table with slab_millers column
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "Rank", "Catalyst", "Miller Index", "ASCI", "Activity",
            "Stability", "Cost", "ΔE (eV)", "γ (J/m²)"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.results_table)

        layout.addWidget(table_group)

        return tab
    
    def create_visualization_tab(self):
        """Create visualization tab with matplotlib plots."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Plot type selector
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Plot Type:"))

        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            "Panel A: 3D Pareto Space",
            "Panel B: Rank vs Adsorption Energy",
            "Panel C: Volcano Optimization Contour",
            "Panel D: Top Performers Breakdown",
            "Score Distributions",
            "Activity-Stability Trade-off",
            "Activity-Cost Trade-off",
            "Stability-Cost Trade-off"
        ])
        self.plot_combo.currentTextChanged.connect(self.update_plot)
        control_layout.addWidget(self.plot_combo)

        control_layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.update_plot)
        control_layout.addWidget(refresh_btn)

        save_plot_btn = QPushButton("💾 Save Plot")
        save_plot_btn.clicked.connect(self.save_current_plot)
        control_layout.addWidget(save_plot_btn)

        layout.addLayout(control_layout)

        # Matplotlib canvas
        self.canvas = MplCanvas(self, width=10, height=7, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        return tab
    
    def apply_stylesheet(self):
        """Apply custom stylesheet for modern look."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ECF0F1;
            }
            QGroupBox {
                background-color: white;
                border: 2px solid #BDC3C7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #21618C;
            }
            QComboBox {
                padding: 5px;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                background-color: white;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #BDC3C7;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #27AE60;
                border-radius: 3px;
            }
        """)
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    
    def on_reaction_changed(self, reaction: str):
        """Handle reaction type change."""
        if reaction == "CO2RR":
            self.pathway_combo.setEnabled(True)
        else:
            self.pathway_combo.setEnabled(False)
        
        self.update_config_summary()
    
    def load_data_file(self):
        """Open file dialog and load data."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Catalyst Data File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load and preview data
                df = pd.read_csv(file_path)
                
                self.current_file = file_path
                self.file_label.setText(Path(file_path).name)
                self.file_label.setStyleSheet("color: #27AE60; font-weight: bold;")
                
                # Show preview
                preview_text = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
                preview_text += "Columns:\n" + ", ".join(df.columns.tolist()[:10])
                if len(df.columns) > 10:
                    preview_text += f", ... ({len(df.columns) - 10} more)"
                
                preview_text += f"\n\nFirst 5 rows:\n{df.head().to_string()}"
                
                self.data_preview.setText(preview_text)
                
                # Enable run button
                self.run_button.setEnabled(True)
                
                self.statusBar().showMessage(f"Loaded: {Path(file_path).name}", 3000)
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
    
    def set_weights(self, w_a: float, w_s: float, w_c: float):
        """Set weight sliders to specific values."""
        self.w_a_slider.setValue(int(w_a * 100))
        self.w_s_slider.setValue(int(w_s * 100))
        self.w_c_slider.setValue(int(w_c * 100))
    
    def update_weights(self):
        """Update weight labels and sum indicator."""
        w_a = self.w_a_slider.value() / 100.0
        w_s = self.w_s_slider.value() / 100.0
        w_c = self.w_c_slider.value() / 100.0
        
        self.w_a_label.setText(f"{w_a:.2f}")
        self.w_s_label.setText(f"{w_s:.2f}")
        self.w_c_label.setText(f"{w_c:.2f}")
        
        total = w_a + w_s + w_c
        
        if abs(total - 1.0) < 0.01:
            self.sum_label.setText(f"Sum: {total:.2f} ✓")
            self.sum_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.sum_label.setText(f"Sum: {total:.2f} ⚠️")
            self.sum_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.update_config_summary()
    
    def update_config_summary(self):
        """Update configuration summary text."""
        reaction = self.reaction_combo.currentText()
        pathway = self.pathway_combo.currentText() if reaction == "CO2RR" else "N/A"
        method = self.method_combo.currentText()
        
        w_a = self.w_a_slider.value() / 100.0
        w_s = self.w_s_slider.value() / 100.0
        w_c = self.w_c_slider.value() / 100.0
        
        summary = f"""
<b>Reaction Configuration:</b>
- Reaction: {reaction}
- Pathway: {pathway}
- Scoring Method: {method}

<b>Weight Configuration:</b>
- Activity Weight (w_a): {w_a:.2f}
- Stability Weight (w_s): {w_s:.2f}
- Cost Weight (w_c): {w_c:.2f}
- Sum: {w_a + w_s + w_c:.2f}

<b>Data File:</b>
{self.file_label.text()}
"""
        self.config_summary.setHtml(summary)
    
    def run_calculation(self):
        """Start ASCI calculation in background thread."""
        if self.current_file is None:
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return
        
        # Get parameters
        reaction = self.reaction_combo.currentText()
        pathway = self.pathway_combo.currentText() if reaction == "CO2RR" else None
        method = self.method_combo.currentText().lower()
        
        w_a = self.w_a_slider.value() / 100.0
        w_s = self.w_s_slider.value() / 100.0
        w_c = self.w_c_slider.value() / 100.0
        weights = (w_a, w_s, w_c)
        
        # Validate weights
        if abs(sum(weights) - 1.0) > 0.01:
            QMessageBox.warning(
                self, 
                "Invalid Weights", 
                f"Weights must sum to 1.0 (currently {sum(weights):.2f})"
            )
            return
        
        # Disable run button
        self.run_button.setEnabled(False)
        self.calc_progress.setValue(0)
        self.status_label.setText("Starting calculation...")
        self.progress_bar.setVisible(True)
        
        # Create and start worker thread
        self.worker = CalculationWorker(
            reaction=reaction,
            data_file=self.current_file,
            pathway=pathway,
            weights=weights,
            method=method
        )
        
        self.worker.progress.connect(self.on_calculation_progress)
        self.worker.finished.connect(self.on_calculation_finished)
        self.worker.error.connect(self.on_calculation_error)
        
        self.worker.start()
    
    def on_calculation_progress(self, value: int, message: str):
        """Handle calculation progress updates."""
        self.calc_progress.setValue(value)
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
    
    def on_calculation_finished(self, results: pd.DataFrame):
        """Handle calculation completion."""
        self.results = results
        self.calculator = self.worker.calculator
        
        # Update UI
        self.calc_progress.setValue(100)
        self.progress_bar.setVisible(False)
        self.status_label.setText("✓ Calculation complete!")
        self.status_label.setStyleSheet("color: #27AE60; font-weight: bold;")
        self.run_button.setEnabled(True)
        
        # Update statistics
        self.update_statistics()
        
        # Update results table
        self.update_results_table()
        
        # Update plot
        self.update_plot()
        
        # Show success message
        QMessageBox.information(
            self,
            "Success",
            f"ASCI calculation completed successfully!\n\n"
            f"Processed {len(results)} catalysts.\n"
            f"Best ASCI score: {results['ASCI'].max():.4f}\n"
            f"Top catalyst: {results.iloc[0]['symbol']}"
        )
        
        self.statusBar().showMessage("✓ Calculation complete! View results in tabs.", 5000)
    
    def on_calculation_error(self, error_message: str):
        """Handle calculation error."""
        self.calc_progress.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Calculation failed")
        self.status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
        self.run_button.setEnabled(True)
        
        QMessageBox.critical(self, "Calculation Error", f"Error:\n{error_message}")
    
    def update_statistics(self):
        """Update statistics display."""
        if self.results is None:
            return
        
        stats_text = f"""
<b>Overall Statistics:</b><br>
- Total catalysts: {len(self.results):,}<br>
- Best ASCI: {self.results['ASCI'].max():.4f}<br>
- Mean ASCI: {self.results['ASCI'].mean():.4f} ± {self.results['ASCI'].std():.4f}<br>
- Top catalyst: {self.results.iloc[0]['symbol']}<br>
<br>
<b>Score Ranges:</b><br>
- Activity: [{self.results['activity_score'].min():.3f}, {self.results['activity_score'].max():.3f}]<br>
- Stability: [{self.results['stability_score'].min():.3f}, {self.results['stability_score'].max():.3f}]<br>
- Cost: [{self.results['cost_score'].min():.3f}, {self.results['cost_score'].max():.3f}]<br>
"""
        self.stats_label.setText(stats_text)
        self.stats_label.setStyleSheet("font-size: 12px;")
    
    def update_results_table(self):
        """Update results table with top catalysts."""
        if self.results is None:
            return

        n_top = self.top_spin.value()
        top_results = self.results.head(n_top)

        self.results_table.setRowCount(len(top_results))

        for i, (idx, row) in enumerate(top_results.iterrows()):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(row['rank'])))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(row['symbol'])))
            # Add slab_millers column if exists
            miller = str(row.get('slab_millers', 'N/A'))
            self.results_table.setItem(i, 2, QTableWidgetItem(miller))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{row['ASCI']:.4f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{row['activity_score']:.4f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{row['stability_score']:.4f}"))
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{row['cost_score']:.4f}"))
            self.results_table.setItem(i, 7, QTableWidgetItem(f"{row['DFT_ads_E']:+.3f}"))
            # Add surface energy column
            gamma = row.get('surface_energy', 0)
            self.results_table.setItem(i, 8, QTableWidgetItem(f"{gamma:.3f}"))
    
    def update_plot(self):
        """Update matplotlib plot based on selected type."""
        if self.results is None or self.calculator is None:
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, 'No results to plot\nRun calculation first',
                                 ha='center', va='center', fontsize=14, color='gray')
            self.canvas.draw()
            return

        plot_type = self.plot_combo.currentText()

        # Clear figure completely for fresh plot
        self.canvas.fig.clear()
        self.canvas.axes = self.canvas.fig.add_subplot(111)

        try:
            if plot_type == "Panel A: 3D Pareto Space":
                self._plot_3d_pareto()

            elif plot_type == "Panel B: Rank vs Adsorption Energy":
                self._plot_rank_vs_adsorption()

            elif plot_type == "Panel C: Volcano Optimization Contour":
                self._plot_volcano_contour()

            elif plot_type == "Panel D: Top Performers Breakdown":
                self._plot_top_performers()

            elif plot_type == "Score Distributions":
                self._plot_score_distributions()

            elif plot_type == "Activity-Stability Trade-off":
                self._plot_tradeoff('activity_score', 'stability_score',
                                   'Activity Score', 'Stability Score',
                                   'Activity vs Stability Trade-off')

            elif plot_type == "Activity-Cost Trade-off":
                self._plot_tradeoff('activity_score', 'cost_score',
                                   'Activity Score', 'Cost Score',
                                   'Activity vs Cost Trade-off')

            elif plot_type == "Stability-Cost Trade-off":
                self._plot_tradeoff('stability_score', 'cost_score',
                                   'Stability Score', 'Cost Score',
                                   'Stability vs Cost Trade-off')

            else:
                self.canvas.axes.text(0.5, 0.5, f'{plot_type}\nNot implemented',
                                     ha='center', va='center', fontsize=14, color='gray')

            self.canvas.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, f'Error generating plot:\n{str(e)}',
                                 ha='center', va='center', fontsize=12, color='red')
            self.canvas.draw()

    def _plot_3d_pareto(self):
        """Panel A: 3D Pareto Space visualization."""
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111, projection='3d')

        # Main scatter
        scatter = ax.scatter(
            self.results['activity_score'],
            self.results['stability_score'],
            self.results['cost_score'],
            c=self.results['ASCI'],
            cmap='viridis', s=30, alpha=0.7,
            edgecolors='white', linewidths=0.2,
            vmin=0, vmax=1
        )

        # Highlight top 10
        top10 = self.results.head(10)
        ax.scatter(
            top10['activity_score'],
            top10['stability_score'],
            top10['cost_score'],
            s=100, marker='*', c='red',
            edgecolors='darkred', linewidths=1,
            label='Top 10 ASCI'
        )

        ax.set_xlabel('Activity Score', fontsize=10, fontweight='bold')
        ax.set_ylabel('Stability Score', fontsize=10, fontweight='bold')
        ax.set_zlabel('Cost Score', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.view_init(elev=20, azim=30)
        ax.legend(loc='upper left', fontsize=8)

        cbar = self.canvas.fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('ASCI Score', fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

        self.canvas.axes = ax

    def _plot_rank_vs_adsorption(self):
        """Panel B: ASCI Rank vs Adsorption Energy (volcano-style with quadratic fit)."""
        ax = self.canvas.axes
        df = self.results.copy()
        df['rank'] = range(1, len(df) + 1)

        # Main scatter
        scatter = ax.scatter(
            df['DFT_ads_E'], df['rank'],
            c=df['ASCI'], cmap='viridis', s=40, alpha=0.8,
            edgecolors='white', linewidths=0.3,
            vmin=0, vmax=1
        )

        # Optimal energy line
        opt_E = self.calculator.config.optimal_energy
        ax.axvline(opt_E, color='red', linestyle='--', linewidth=2,
                  label=f'Optimal ΔE = {opt_E:.2f} eV')

        # Highlight top 10
        top10 = df.head(10)
        ax.scatter(top10['DFT_ads_E'], top10['rank'],
                  s=120, marker='*', facecolors='none',
                  edgecolors='red', linewidths=1.5, label='Top 10 ASCI')

        # Quadratic trend fit on high performers
        high_score = df[df['ASCI'] > df['ASCI'].quantile(0.75)]
        if len(high_score) > 5:
            e_min, e_max = df['DFT_ads_E'].min(), df['DFT_ads_E'].max()
            z = np.polyfit(high_score['DFT_ads_E'], high_score['rank'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(e_min, e_max, 100)
            ax.plot(x_range, p(x_range), '--', color='green',
                   linewidth=1.5, alpha=0.7, label='Performance Trend')

        ax.set_xlabel('Adsorption Energy ΔE (eV)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'ASCI Rank (1-{len(df)})', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)

        cbar = self.canvas.fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('ASCI Score', fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    def _plot_volcano_contour(self):
        """Panel C: Volcano Optimization Contour (ASCI landscape)."""
        from scipy.interpolate import griddata
        ax = self.canvas.axes

        x = self.results['DFT_ads_E'].values
        y = np.log10(self.results['Cost'].values)

        # Grid for interpolation
        x_range = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
        y_range = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x_range, y_range)

        # Calculate ASCI on grid using formula
        max_dev = self.calculator.config.activity_width
        opt_E = self.calculator.config.optimal_energy
        activity_grid = np.clip(1 - np.abs(X - opt_E) / max_dev, 0, 1)

        try:
            points = np.vstack((x, y)).T
            stab = griddata(points, self.results['stability_score'].values, (X, Y), method='linear', fill_value=0)
            cost = griddata(points, self.results['cost_score'].values, (X, Y), method='linear', fill_value=0)
            Z = np.clip(0.33 * activity_grid + 0.33 * stab + 0.34 * cost, 0, 1)
        except:
            Z = activity_grid

        # Contour plot
        levels = np.linspace(0, 1, 21)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8, vmin=0, vmax=1)

        # Contour lines
        contour_lines = ax.contour(X, Y, Z, levels=[0.3, 0.45, 0.6, 0.75],
                                   colors='white', linewidths=0.7, alpha=0.8)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

        # Data points
        ax.scatter(x, y, c='white', s=15, alpha=0.5, edgecolors='black', linewidths=0.2)

        # Top 10
        top10 = self.results.head(10)
        ax.scatter(top10['DFT_ads_E'], np.log10(top10['Cost']),
                  s=100, marker='*', facecolors='none',
                  edgecolors='red', linewidths=1.5, label='Top 10 ASCI')

        # Optimal line
        ax.axvline(opt_E, color='red', linestyle='--', linewidth=1.5,
                  label=f'Optimal ΔE = {opt_E:.2f} eV')

        ax.set_xlabel('Adsorption Energy ΔE (eV)', fontsize=11, fontweight='bold')
        ax.set_ylabel('log₁₀ Cost (USD/kg)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)

        cbar = self.canvas.fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('ASCI Score', fontsize=9)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def _plot_top_performers(self):
        """Panel D: Top Performers Breakdown (grouped bar chart)."""
        ax = self.canvas.axes
        top10 = self.results.head(10)
        x = np.arange(len(top10))
        width = 0.2

        ax.bar(x - 1.5*width, top10['activity_score'], width,
               label='Activity', color='#2ecc71', edgecolor='black', linewidth=0.5)
        ax.bar(x - 0.5*width, top10['stability_score'], width,
               label='Stability', color='#3498db', edgecolor='black', linewidth=0.5)
        ax.bar(x + 0.5*width, top10['cost_score'], width,
               label='Cost', color='#e67e22', edgecolor='black', linewidth=0.5)
        ax.bar(x + 1.5*width, top10['ASCI'], width,
               label='ASCI', color='#9b59b6', edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(top10['symbol'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_xlabel('Catalyst', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Catalysts: Score Breakdown', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', ncol=4, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)

    def _plot_score_distributions(self):
        """Generate score distribution histograms."""
        ax = self.canvas.axes
        scores = ['activity_score', 'stability_score', 'cost_score', 'ASCI']
        colors = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6']
        labels = ['Activity (S_a)', 'Stability (S_s)', 'Cost (S_c)', 'ASCI']

        for score, color, label in zip(scores, colors, labels):
            ax.hist(self.results[score], bins=25, alpha=0.5,
                   color=color, label=label, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Score [0, 1]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Score Distributions', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, 1)

    def _plot_tradeoff(self, x_col, y_col, x_label, y_label, title):
        """Generate trade-off scatter plot."""
        ax = self.canvas.axes

        scatter = ax.scatter(self.results[x_col], self.results[y_col],
                            c=self.results['ASCI'], cmap='plasma',
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.3)

        # Highlight top 10
        top10 = self.results.head(10)
        ax.scatter(top10[x_col], top10[y_col],
                  s=120, facecolors='none', edgecolors='lime', linewidth=2)

        for _, row in top10.head(5).iterrows():
            ax.annotate(row['symbol'], (row[x_col], row[y_col]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.canvas.fig.colorbar(scatter, ax=ax, label='ASCI Score', shrink=0.8)

    def save_results(self):
        """Save results to CSV file."""
        if self.results is None:
            QMessageBox.warning(self, "No Results", "No results to save. Run calculation first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            f"{self.calculator.reaction}_results.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.results.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Results saved to:\n{file_path}")
                self.statusBar().showMessage(f"Results saved: {Path(file_path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def export_results_table(self):
        """Export current results table."""
        if self.results is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Table",
            f"{self.calculator.reaction}_top{self.top_spin.value()}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                top = self.results.head(self.top_spin.value())
                top.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Table exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
    def save_current_plot(self):
        """Save current matplotlib plot."""
        if self.results is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            f"{self.calculator.reaction}_{self.plot_combo.currentText().replace(' ', '_')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                self.canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
                self.statusBar().showMessage(f"Plot saved: {Path(file_path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
    
    def export_figures(self):
        """Export all figures using Visualizer."""
        if self.results is None or self.calculator is None:
            QMessageBox.warning(self, "No Results", "No results available. Run calculation first.")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        
        if output_dir:
            try:
                viz = Visualizer(self.results, self.calculator.config)
                generated = viz.generate_all_figures(output_dir=output_dir, dpi=600)
                
                total_files = sum(len(files) for files in generated.values())
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Generated {total_files} figures in:\n{output_dir}"
                )
                self.statusBar().showMessage(f"Exported {total_files} figures", 3000)
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export figures:\n{str(e)}")
    
    def show_about_dialog(self):
        """Show about dialog."""
        about_text = f"""
<h2>ASCICat v{__version__}</h2>
<p><b>Activity-Stability-Cost Integrated Catalyst Discovery</b></p>

<p>Multi-objective electrocatalyst screening toolkit that extends
traditional volcano plots to simultaneously optimize catalytic activity,
electrochemical stability, and economic viability.</p>

<p><b>Author:</b> N. Khossossi<br>
<b>Institution:</b> DIFFER (Dutch Institute for Fundamental Energy Research)<br>
<b>Email:</b> n.khossossi@differ.nl</p>

<p><b>Framework:</b> φ_ASCI = w_a·S_a + w_s·S_s + w_c·S_c</p>

<p><b>License:</b> MIT<br>
<b>GitHub:</b> https://github.com/nabkh/ASCICat</p>
"""
        
        QMessageBox.about(self, "About ASCICat", about_text)
    
    def show_documentation(self):
        """Show documentation information."""
        docs_text = """
<h3>📚 ASCICat Documentation</h3>

<p><b>Online Documentation:</b><br>
https://ascicat.readthedocs.io</p>

<p><b>Quick Start Guide:</b><br>
1. Load your catalyst data (CSV format)<br>
2. Select reaction type and pathway<br>
3. Adjust weights using sliders or presets<br>
4. Click "RUN CALCULATION"<br>
5. View results and generate figures</p>

<p><b>Required Data Columns:</b><br>
- DFT_ads_E - Adsorption energy (eV)<br>
- surface_energy - Surface energy (J/m²)<br>
- Cost - Material cost ($/kg)<br>
- symbol - Catalyst identifier</p>

<p><b>Support:</b><br>
- Email: n.khossossi@differ.nl<br>
- Issues: GitHub Issues page</p>
"""
        
        QMessageBox.information(self, "Documentation", docs_text)
    
    def show_citation(self):
        """Show citation information."""
        citation_text = """
<h3>📖 Citation</h3>

<p>If you use ASCICat in your research, please cite:</p>

<p><b>Khossossi, N.</b> (2025). ASCICat: Activity-Stability-Cost Integrated
Framework for Electrocatalyst Discovery. <i>Journal</i>, <b>XX</b>, XXX-XXX.<br>
DOI: 10.xxxx/xxxxxx</p>

<p><b>BibTeX:</b></p>
<pre>
@article{khossossi2025ascicat,
  title={{ASCICat: Activity-Stability-Cost Integrated 
         Framework for Electrocatalyst Discovery}},
  author={Khossossi, N.},
  journal={Journal Name},
  volume={XX},
  pages={XXX--XXX},
  year={2025},
  doi={10.xxxx/xxxxxx}
}
</pre>

<p><b>Preprint:</b> arXiv:XXXX.XXXXX</p>
"""
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Citation")
        msg.setText(citation_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for GUI application."""
    
    # Check dependencies
    if not ALL_DEPENDENCIES_AVAILABLE:
        print("\n" + "="*80)
        print("❌ ASCICat GUI - Missing Dependencies")
        print("="*80)
        print("\nThe following dependencies are required:")
        print("  • PyQt5 (GUI framework)")
        print("  • matplotlib (plotting)")
        print("  • numpy (numerical computing)")
        print("  • pandas (data manipulation)")
        print("  • ascicat (core ASCICat package)")
        print("\nInstall all at once:")
        print("  pip install PyQt5 matplotlib numpy pandas ascicat")
        print("\nOr install ASCICat with GUI support:")
        print("  pip install ascicat[gui]")
        print("\n" + "="*80)
        return 1
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("ASCICat")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("DIFFER")
    
    # Create and show main window
    window = ASCICatGUI()
    window.show()
    
    # Run application
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())