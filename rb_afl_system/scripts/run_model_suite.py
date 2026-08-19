#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json

from rb_afl_system.config import load_config
from rb_afl_system.engine.model_suite_runner import run_model_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RB-AFL model train/eval/compare suite")
    ap.add_argument("--config", required=True, help="JSON model suite config")
    ns = ap.parse_args()
    summary = run_model_suite(load_config(ns.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
