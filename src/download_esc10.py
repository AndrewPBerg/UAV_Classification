#!/usr/bin/env python3
"""
Convenient script to create ESC-10 dataset from existing ESC-50 dataset.
This script provides easy access to the ESC-10 extraction functionality.
"""

import sys
from pathlib import Path

# Add current directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from esc10.download import download_esc10, main as esc10_main

def main():
    """Main function that delegates to the ESC-10 download script."""
    return esc10_main()

if __name__ == "__main__":
    exit(main()) 