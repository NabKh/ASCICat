"""
ASCICat Batch Processor
Entry point wrapper for batch processing tool.
"""

def main():
    """Main entry point for ascicat-batch."""
    from scripts.batch_processor import main as batch_main
    batch_main()

if __name__ == '__main__':
    main()
