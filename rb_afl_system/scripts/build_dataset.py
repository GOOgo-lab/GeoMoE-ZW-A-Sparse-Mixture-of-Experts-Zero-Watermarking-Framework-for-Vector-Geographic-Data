#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from rb_afl_system.config import load_config
from rb_afl_system.data.dataset.build_dataset import build_identity_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RB-AFL identity dataset from shapefiles")
    ap.add_argument("--config", required=True, help="JSON/YAML config path")
    ns = ap.parse_args()
    info = build_identity_dataset(load_config(ns.config))
    print(info)

if __name__ == "__main__":
    main()
