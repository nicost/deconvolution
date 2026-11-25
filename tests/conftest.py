"""Shared test setup.

pytest is configured (in pyproject.toml) with pythonpath = ["src"], so the
scripts in src/ import normally.  We also force a non-interactive matplotlib
backend so importing the scripts never tries to open a window, and add src/ to
the path explicitly as a fallback for running pytest from odd working dirs.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
