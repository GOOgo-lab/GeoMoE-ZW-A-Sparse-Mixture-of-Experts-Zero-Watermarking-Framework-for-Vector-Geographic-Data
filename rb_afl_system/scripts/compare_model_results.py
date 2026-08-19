#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rb_afl_system.config import load_config
from rb_afl_system.engine.eval_reporter import compare_model_results, export_paper_ready_tables


def _parse_item(text: str) -> dict:
    parts = text.split("|")
    if len(parts) != 3:
        raise ValueError("--item must be 'name|uniqueness_dir|robustness_dir'")
    return {"name": parts[0], "uniqueness_dir": parts[1], "robustness_dir": parts[2]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare RB-AFL model evaluation results")
    ap.add_argument("--config", default="", help="JSON config with {items:[...], output_dir:...}")
    ap.add_argument("--item", action="append", default=[], help="Repeated: name|uniqueness_dir|robustness_dir")
    ap.add_argument("--output_dir", default="", help="Output directory for model_compare.csv/json")
    ap.add_argument("--output_xlsx", default="", help="Optional paper-ready xlsx path")
    ns = ap.parse_args()
    if ns.config:
        cfg = load_config(ns.config)
        items = cfg.get("items", [])
        output_dir = ns.output_dir or cfg.get("output_dir", "model_compare")
        output_xlsx = ns.output_xlsx or cfg.get("output_xlsx", "")
    else:
        items = [_parse_item(x) for x in ns.item]
        output_dir = ns.output_dir
        output_xlsx = ns.output_xlsx
    if not items:
        raise ValueError("No model result items provided")
    if not output_dir:
        raise ValueError("output_dir is required")
    summary = compare_model_results(items, output_dir)
    if output_xlsx:
        export_paper_ready_tables(Path(summary["model_compare_csv"]), output_xlsx)
        summary["output_xlsx"] = str(output_xlsx)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
