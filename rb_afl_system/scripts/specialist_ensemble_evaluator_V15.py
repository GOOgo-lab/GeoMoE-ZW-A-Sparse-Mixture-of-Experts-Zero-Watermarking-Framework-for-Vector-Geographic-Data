#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V15 specialist ensemble evaluator with de-duplicated attack-sample summary.

V13 reported routed rows, which can double count attacks whose names match both
local and boundary roles (for example ``boundary_simplify``).  V15 preserves the
route-level report and additionally writes a canonical de-duplicated report.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from rb_afl_system.scripts.specialist_ensemble_evaluator_V13 import (
    _load_profile,
    _load_selected,
    _load_unique_pairs,
    _route_attack_rows,
    _standardize_rob_rows,
    _summary_from_rob_rows,
    _summary_from_unique_pairs,
    _safe_float,
)
from rb_afl_system.utils import ensure_dir, write_json

ROLE_PRIORITY: dict[str, int] = {
    "topology": 0,
    "boundary": 1,
    "rotate": 2,
    "scale": 3,
    "jitter": 4,
    "quantize": 5,
    "simplify": 6,
    "fallback_other": 7,
}


def _dedup_key_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        ["identity", "sample", "sample_dir"],
        ["identity", "attack_type", "attack_value", "sample_dir"],
        ["identity", "attack_type", "attack_engine", "attack_value"],
        ["identity", "attack_type", "attack_value"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    fallback = [c for c in ["identity", "attack_type", "attack_value"] if c in df.columns]
    return fallback or list(df.columns[: min(3, len(df.columns))])


def _deduplicate_selected_rows(selected_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if selected_rows.empty:
        return selected_rows.copy(), {"dedup_key_columns": [], "duplicate_rows_removed": 0}
    out = selected_rows.copy()
    out["__role_priority"] = out.get("specialist_role", "fallback_other").astype(str).map(lambda r: ROLE_PRIORITY.get(r, 99))
    key_cols = _dedup_key_columns(out)
    before = int(len(out))
    out = out.sort_values(["__role_priority", "feature_ber", "feature_nc"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=key_cols, keep="first").drop(columns=["__role_priority"])
    out = out.reset_index(drop=True)
    return out, {"dedup_key_columns": key_cols, "duplicate_rows_removed": int(before - len(out))}


def evaluate_specialist_ensemble_v15(
    profile_csv: str | Path,
    selected_specialists_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    profile_df = _load_profile(profile_csv)
    model_by_role = _load_selected(selected_specialists_json)
    out_dir = ensure_dir(output_dir)

    selected_rows, by_role = _route_attack_rows(profile_df, model_by_role)
    selected_rows = _standardize_rob_rows(selected_rows)
    selected_rows.to_csv(out_dir / "specialist_ensemble_rows.csv", index=False, encoding="utf-8-sig")
    by_role.to_csv(out_dir / "specialist_ensemble_by_role.csv", index=False, encoding="utf-8-sig")

    dedup_rows, dedup_meta = _deduplicate_selected_rows(selected_rows)
    dedup_rows.to_csv(out_dir / "specialist_ensemble_rows_dedup.csv", index=False, encoding="utf-8-sig")

    unique_model = model_by_role.get("unique_gate", "")
    unique_pairs = _load_unique_pairs(profile_df, unique_model)
    if not unique_pairs.empty:
        unique_pairs.to_csv(out_dir / "specialist_unique_gate_pairs.csv", index=False, encoding="utf-8-sig")

    routed_summary = _summary_from_rob_rows(selected_rows)
    dedup_summary = _summary_from_rob_rows(dedup_rows)
    uniq_summary = _summary_from_unique_pairs(unique_pairs)

    mean_unique_nc = _safe_float(uniq_summary.get("unique_gate_mean_unique_nc", 1.0), 1.0)
    max_unique_nc = _safe_float(uniq_summary.get("unique_gate_max_unique_nc", 1.0), 1.0)
    mean_robust_nc = _safe_float(dedup_summary.get("mean_robust_nc", 0.0))
    min_robust_nc = _safe_float(dedup_summary.get("min_robust_nc", 0.0))

    summary = {
        "profile_csv": str(profile_csv),
        "selected_specialists_json": str(selected_specialists_json),
        "unique_gate_model": unique_model,
        "model_by_role": model_by_role,
        **{f"routed_{k}": v for k, v in routed_summary.items()},
        **{f"dedup_{k}": v for k, v in dedup_summary.items()},
        **uniq_summary,
        **dedup_meta,
        "specialist_nc_joint_score_dedup": float((1.0 - mean_unique_nc) * mean_robust_nc),
        "specialist_nc_conservative_score_dedup": float((1.0 - max_unique_nc) * min_robust_nc),
        "specialist_nc_margin_robust_minus_unique_dedup": float(mean_robust_nc - mean_unique_nc),
        "note": "V15 reports both routed and de-duplicated attack-sample summaries; use dedup_* for formal paper tables.",
    }
    write_json(out_dir / "specialist_ensemble_summary_v15.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "specialist_ensemble_summary_v15.csv", index=False, encoding="utf-8-sig")

    # Backward-compatible file names also point to the V15 summary.
    write_json(out_dir / "specialist_ensemble_summary.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "specialist_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "specialist_ensemble_summary_json": str(out_dir / "specialist_ensemble_summary_v15.json"),
        "specialist_ensemble_summary_csv": str(out_dir / "specialist_ensemble_summary_v15.csv"),
        "specialist_ensemble_rows_csv": str(out_dir / "specialist_ensemble_rows.csv"),
        "specialist_ensemble_rows_dedup_csv": str(out_dir / "specialist_ensemble_rows_dedup.csv"),
        "dedup_num_rows": int(dedup_summary.get("num_rows", 0)),
        "routed_num_rows": int(routed_summary.get("num_rows", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate V15 specialist ensemble with de-duplicated summary")
    ap.add_argument("--profile_csv", required=True)
    ap.add_argument("--selected_json", required=True)
    ap.add_argument("--output_dir", default="")
    ns = ap.parse_args()
    try:
        selected_path = Path(ns.selected_json)
        out_dir = Path(ns.output_dir) if ns.output_dir else selected_path.parent / "ensemble_eval"
        result = evaluate_specialist_ensemble_v15(ns.profile_csv, selected_path, out_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] specialist_ensemble_evaluator_V15: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
