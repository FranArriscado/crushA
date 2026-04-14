"""
config.py — Central path configuration for CrushAnalytica.

All scripts import paths from here instead of hardcoding them.
Paths are relative to this file, so they work on any machine.

Usage in any script:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # if in a subfolder
    from config import DATA_DIR, RESULTS_NORIB, RESULTS_RIB, PYTHON_EXE
"""

import sys
from pathlib import Path

# Project root = wherever this file lives
ROOT = Path(__file__).resolve().parent

# Input data (partition_data_*.mat files)
DATA_DIR = ROOT / "NoRib" / "data"

# Output directories
RESULTS_NORIB = ROOT / "results" / "NoRib"
RESULTS_RIB   = ROOT / "results" / "Rib"

# Python executable — always points to the active interpreter (local venv)
PYTHON_EXE = sys.executable

# Auto-create output folders on import
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_NORIB.mkdir(parents=True, exist_ok=True)
RESULTS_RIB.mkdir(parents=True, exist_ok=True)
