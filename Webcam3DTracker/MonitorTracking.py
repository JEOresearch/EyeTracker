"""Deprecated entry point. Use the modular implementation via python -m src.monitor_tracking."""

import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.monitor_tracking import main

if __name__ == "__main__":
    main()
