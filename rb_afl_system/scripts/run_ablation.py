#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from rb_afl_system.config import load_config
from rb_afl_system.engine.ablation_runner import run_ablation


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RB-AFL ablation experiments")
    ap.add_argument("--config", required=True)
    ns = ap.parse_args()
    print(run_ablation(load_config(ns.config)))

if __name__ == "__main__":
    main()
