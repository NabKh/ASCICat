"""
ASCICat Graphical User Interface
Entry point wrapper for the GUI tool.
"""

def main():
    """Main entry point for ascicat-gui."""
    from scripts.ascicat_gui import main as gui_main
    gui_main()

if __name__ == '__main__':
    main()
