"""
ASCICat Command-Line Interface
Entry point wrapper for the CLI tool.
"""

def main():
    """Main entry point for ascicat CLI."""
    from scripts.ascicat_cli import main as cli_main
    cli_main()

if __name__ == '__main__':
    main()
