#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that mapshaper is installed and callable."""
from __future__ import annotations
import argparse
import subprocess
from rb_afl_system.data.attacks.mapshaper_attacks import require_mapshaper


def main() -> None:
    ap = argparse.ArgumentParser(description="Check mapshaper CLI installation")
    ap.add_argument("--mapshaper_bin", default="mapshaper", help="Executable name or full path")
    ns = ap.parse_args()
    exe = require_mapshaper(ns.mapshaper_bin)
    print(f"[OK] mapshaper executable: {exe}")
    result = subprocess.run([exe, "-v"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    out = (result.stdout or result.stderr).strip()
    if out:
        print(out)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
