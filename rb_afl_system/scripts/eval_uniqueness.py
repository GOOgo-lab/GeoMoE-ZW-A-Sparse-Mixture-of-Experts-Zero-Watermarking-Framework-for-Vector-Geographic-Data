#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from rb_afl_system.config import load_config
from rb_afl_system.engine.uniqueness_evaluator import evaluate_uniqueness


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RB-AFL uniqueness")
    ap.add_argument("--config", required=True)
    ns = ap.parse_args()
    print(evaluate_uniqueness(load_config(ns.config)))

if __name__ == "__main__":
    main()
