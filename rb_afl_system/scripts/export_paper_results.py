#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from rb_afl_system.engine.paper_exporter import export_ablation_tables, export_model_compare_tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Export RB-AFL paper-ready tables")
    ap.add_argument("--ablation_csv", default="")
    ap.add_argument("--model_compare_csv", default="")
    ap.add_argument("--output_xlsx", required=True)
    ns = ap.parse_args()
    if ns.ablation_csv:
        export_ablation_tables(ns.ablation_csv, ns.output_xlsx)
    elif ns.model_compare_csv:
        export_model_compare_tables(ns.model_compare_csv, ns.output_xlsx)
    else:
        raise ValueError("Provide either --ablation_csv or --model_compare_csv")
    print({"saved": ns.output_xlsx})

if __name__ == "__main__":
    main()
