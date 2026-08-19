#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export visual reports for a completed RB-AFL model suite."""
from __future__ import annotations

import argparse
import json

from rb_afl_system.engine.visual_reporter import export_suite_visual_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", required=True, help="run_model_suite output root containing compare/model_compare.csv")
    ap.add_argument("--output_dir", default="", help="visual output directory; default: suite_root/visuals")
    ap.add_argument("--timestamp", default="", help="optional suffix used in generated report names")
    ns = ap.parse_args()
    summary = export_suite_visual_report(
        suite_root=ns.suite_root,
        output_dir=ns.output_dir or None,
        timestamp=ns.timestamp or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
