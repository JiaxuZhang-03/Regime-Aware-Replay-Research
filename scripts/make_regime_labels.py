#!/usr/bin/env python
"""Wrapper for local script execution without installing the package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regime_labeling.cli import main


if __name__ == "__main__":
    main()
