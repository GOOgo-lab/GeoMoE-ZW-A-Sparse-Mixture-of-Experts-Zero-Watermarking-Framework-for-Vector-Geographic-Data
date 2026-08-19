#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export paper-ready tables from CSV/JSON results."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from rb_afl_system.engine.eval_reporter import export_paper_ready_tables


def export_ablation_tables(ablation_csv: str | Path, output_xlsx: str | Path) -> None:
    src = Path(ablation_csv)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    df = pd.read_csv(src)
    out = Path(output_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        df.to_excel(writer, index=False, sheet_name="ablation_all")
        cols = [c for c in ["exp_id", "exp_name", "channels", "generator", "discriminator", "mean_unique_nc", "max_unique_nc", "mean_robust_nc", "min_robust_nc", "mean_robust_ber"] if c in df.columns]
        if cols:
            df[cols].to_excel(writer, index=False, sheet_name="paper_core")


def export_model_compare_tables(compare_csv: str | Path, output_xlsx: str | Path) -> None:
    export_paper_ready_tables(compare_csv, output_xlsx)
