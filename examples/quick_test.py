#!/usr/bin/env python3
"""Run the public CPU-only GeoMoE-ZW example from the examples directory."""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(repository_root / "quick_run.py"), run_name="__main__")
