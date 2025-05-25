#!/usr/bin/env python3
"""
Convenient script to download ESC-50 dataset.
This script provides easy access to the ESC-50 download functionality.
"""

import sys
from pathlib import Path

# Add current directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from esc50.download import download_esc50, main as esc50_main

def main():
    """Main function that delegates to the ESC-50 download script."""
    return esc50_main()

if __name__ == "__main__":
    exit(main()) 