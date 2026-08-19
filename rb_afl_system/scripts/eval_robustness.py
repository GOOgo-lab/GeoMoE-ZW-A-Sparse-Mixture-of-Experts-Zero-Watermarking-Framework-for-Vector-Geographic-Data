#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from rb_afl_system.config import load_config
from rb_afl_system.engine.robustness_evaluator import evaluate_robustness


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RB-AFL robustness and zero-watermark recovery")
    ap.add_argument("--config", required=True)
    ns = ap.parse_args()
    print(evaluate_robustness(load_config(ns.config)))

if __name__ == "__main__":
    main()
