#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from rb_afl_system.config import load_config
from rb_afl_system.engine.adversarial_trainer import train_adversarial


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RB-AFL adversarial feature model")
    ap.add_argument("--config", required=True, help="JSON/YAML config path")
    ns = ap.parse_args()
    summary = train_adversarial(load_config(ns.config))
    print(summary)

if __name__ == "__main__":
    main()
