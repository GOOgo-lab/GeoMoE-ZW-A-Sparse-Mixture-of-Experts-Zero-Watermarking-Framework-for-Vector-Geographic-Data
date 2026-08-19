#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation result collection and comparison utilities."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable
import json
import pandas as pd
from rb_afl_system.utils import ensure_dir, write_json


def _read_json_if_exists(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _model_quality_scores(row: dict) -> dict:
    mean_unique_ber = float(row.get("uniq_mean_unique_ber", 0.0) or 0.0)
    mean_unique_nc = float(row.get("uniq_mean_unique_nc", 1.0) or 1.0)
    max_unique_nc = float(row.get("uniq_max_unique_nc", 1.0) or 1.0)
    mean_robust_nc = float(row.get("rob_mean_robust_nc", 0.0) or 0.0)
    mean_robust_ber = float(row.get("rob_mean_robust_ber", 1.0) or 1.0)
    min_robust_nc = float(row.get("rob_min_robust_nc", 0.0) or 0.0)
    # BER-based legacy scores are kept for compatibility.
    joint_score = mean_unique_ber * mean_robust_nc
    conservative_joint_score = mean_unique_ber * min_robust_nc * max(0.0, 1.0 - mean_robust_ber)
    # NC-priority scores are preferred for uniqueness-first evaluation.
    nc_joint_score = (1.0 - mean_unique_nc) * mean_robust_nc
    nc_conservative_score = (1.0 - max_unique_nc) * min_robust_nc
    ber_margin = mean_unique_ber - mean_robust_ber
    nc_margin = mean_robust_nc - mean_unique_nc
    return {
        "joint_score_uniqueBER_x_robustNC": joint_score,
        "conservative_joint_score": conservative_joint_score,
        "nc_joint_score": nc_joint_score,
        "nc_conservative_score": nc_conservative_score,
        "ber_margin_unique_minus_robust": ber_margin,
        "nc_margin_robust_minus_unique": nc_margin,
    }


def compare_model_results(items: Iterable[dict], output_dir: str | Path) -> dict:
    """Compare models from uniqueness/robustness result directories.

    Each item must contain:
      - name
      - uniqueness_dir
      - robustness_dir
    """
    out_dir = ensure_dir(output_dir)
    rows: list[dict] = []
    for item in items:
        name = str(item["name"])
        uniq_dir = Path(item["uniqueness_dir"])
        rob_dir = Path(item["robustness_dir"])
        row: dict[str, Any] = {
            "model": name,
            "uniqueness_dir": str(uniq_dir),
            "robustness_dir": str(rob_dir),
        }
        uniq = _read_json_if_exists(uniq_dir / "uniqueness_summary.json")
        rob = _read_json_if_exists(rob_dir / "robustness_summary.json")
        row.update({f"uniq_{k}": v for k, v in uniq.items()})
        row.update({f"rob_{k}": v for k, v in rob.items()})
        row.update(_model_quality_scores(row))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "nc_joint_score" in df.columns:
        df = df.sort_values("nc_joint_score", ascending=False)
    csv_path = out_dir / "model_compare.csv"
    json_path = out_dir / "model_compare.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_compare_csv": str(csv_path), "model_compare_json": str(json_path), "num_models": int(len(rows))}


def export_paper_ready_tables(compare_csv: str | Path, output_xlsx: str | Path) -> None:
    src = Path(compare_csv)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    df = pd.read_csv(src)
    out = Path(output_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    core_cols = [
        "model",
        "uniq_mean_unique_nc",
        "uniq_max_unique_nc",
        "uniq_mean_unique_ber",
        "uniq_min_unique_ber",
        "rob_mean_robust_nc",
        "rob_min_robust_nc",
        "rob_mean_robust_ber",
        "rob_max_robust_ber",
        "rob_mean_watermark_nc",
        "rob_mean_watermark_ber",
        "joint_score_uniqueBER_x_robustNC",
        "conservative_joint_score",
        "nc_joint_score",
        "nc_conservative_score",
        "ber_margin_unique_minus_robust",
        "nc_margin_robust_minus_unique",
    ]
    with pd.ExcelWriter(out) as writer:
        df.to_excel(writer, index=False, sheet_name="all_fields")
        cols = [c for c in core_cols if c in df.columns]
        if cols:
            df[cols].to_excel(writer, index=False, sheet_name="paper_core")
