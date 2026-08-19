#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V15 single-model all-ability evaluator for ablation tables.

This script converts an existing suite into paper-ready single-model reports:
- one long attack matrix per model and attack type;
- one wide summary per model with unique and all-direction robustness metrics;
- rank tables for each ability direction;
- optional comparison against a specialist ensemble report.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rb_afl_system.scripts.capability_profiler_V13 import build_capability_profile
from rb_afl_system.utils import ensure_dir, write_json

DIRECTION_SCORE_COLS = [
    "unique_score",
    "local_score",
    "rotate_score",
    "scale_score",
    "jitter_score",
    "quantize_score",
    "simplify_score",
    "topology_score",
    "boundary_score",
    "overall_score",
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p)


def _resolve_profile(suite_root: Path, profile_csv: str, output_dir: Path) -> Path:
    if profile_csv:
        p = Path(profile_csv)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    compare = suite_root / "compare" / "model_compare.csv"
    if not compare.is_file():
        raise FileNotFoundError(str(compare))
    summary = build_capability_profile(compare_csv=compare, suite_root=suite_root, output_dir=output_dir / "_profile_cache")
    return Path(summary["model_capability_profile_csv"])


def _standardize_rob_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "feature_nc" not in out.columns and "mean_feature_nc" in out.columns:
        out["feature_nc"] = out["mean_feature_nc"]
    if "feature_ber" not in out.columns and "mean_feature_ber" in out.columns:
        out["feature_ber"] = out["mean_feature_ber"]
    if "watermark_nc" not in out.columns and "mean_watermark_nc" in out.columns:
        out["watermark_nc"] = out["mean_watermark_nc"]
    if "watermark_ber" not in out.columns and "mean_watermark_ber" in out.columns:
        out["watermark_ber"] = out["mean_watermark_ber"]
    for col in ["feature_nc", "feature_ber", "watermark_nc", "watermark_ber"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _rob_dir_from_profile(row: pd.Series) -> Path:
    return Path(str(row.get("robustness_dir", "")))


def _uniq_dir_from_profile(row: pd.Series) -> Path:
    return Path(str(row.get("uniqueness_dir", "")))


def _load_robust_rows(row: pd.Series) -> pd.DataFrame:
    rob_dir = _rob_dir_from_profile(row)
    for name in ["robustness_rows.csv", "robustness_by_attack_engine_value.csv", "robustness_by_attack.csv"]:
        p = rob_dir / name
        if p.is_file():
            return _standardize_rob_rows(pd.read_csv(p))
    return pd.DataFrame()


def _load_unique_pairs(row: pd.Series) -> pd.DataFrame:
    p = _uniq_dir_from_profile(row) / "uniqueness_pairs.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


def _metric_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"num_rows": 0, "mean_nc": 0.0, "min_nc": 0.0, "mean_ber": 1.0, "max_ber": 1.0, "mean_wm_nc": 0.0, "min_wm_nc": 0.0}
    return {
        "num_rows": int(len(df)),
        "mean_nc": float(df["feature_nc"].mean(skipna=True)),
        "min_nc": float(df["feature_nc"].min(skipna=True)),
        "mean_ber": float(df["feature_ber"].mean(skipna=True)),
        "max_ber": float(df["feature_ber"].max(skipna=True)),
        "mean_wm_nc": float(df["watermark_nc"].mean(skipna=True)) if "watermark_nc" in df.columns else 0.0,
        "min_wm_nc": float(df["watermark_nc"].min(skipna=True)) if "watermark_nc" in df.columns else 0.0,
    }


def build_single_model_reports(profile_csv: str | Path, output_dir: str | Path, ensemble_summary_csv: str = "") -> dict[str, Any]:
    profile = pd.read_csv(profile_csv)
    out_dir = ensure_dir(output_dir)
    attack_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for _, row in profile.iterrows():
        model = str(row["model"])
        rob_rows = _load_robust_rows(row)
        uniq_pairs = _load_unique_pairs(row)
        overall = _metric_summary(rob_rows)

        if not rob_rows.empty and "attack_type" in rob_rows.columns:
            group_cols = ["attack_type"]
            if "attack_value" in rob_rows.columns:
                group_cols.append("attack_value")
            for keys, part in rob_rows.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                rec = {"model": model, "attack_type": str(keys[0])}
                if len(keys) > 1:
                    rec["attack_value"] = keys[1]
                rec.update(_metric_summary(part))
                attack_rows.append(rec)

        unique_nc = pd.to_numeric(uniq_pairs.get("unique_nc", pd.Series(dtype=float)), errors="coerce").dropna()
        summary = {
            "model": model,
            "family": row.get("family", ""),
            "unique_mean_nc": float(unique_nc.mean()) if not unique_nc.empty else float(row.get("uniq_mean_unique_nc", 1.0)),
            "unique_max_nc": float(unique_nc.max()) if not unique_nc.empty else float(row.get("uniq_max_unique_nc", 1.0)),
            "unique_nc_gt_0_7": int((unique_nc > 0.7).sum()) if not unique_nc.empty else int(row.get("uniq_pair_nc_gt_0_7", 0)),
            "unique_nc_gt_0_8": int((unique_nc > 0.8).sum()) if not unique_nc.empty else int(row.get("uniq_pair_nc_gt_0_8", 0)),
            "unique_nc_gt_0_9": int((unique_nc > 0.9).sum()) if not unique_nc.empty else int(row.get("uniq_pair_nc_gt_0_9", 0)),
            "robust_rows": overall["num_rows"],
            "robust_mean_nc": overall["mean_nc"],
            "robust_min_nc": overall["min_nc"],
            "robust_mean_ber": overall["mean_ber"],
            "robust_max_ber": overall["max_ber"],
            "single_model_joint_score": float((1.0 - (float(unique_nc.mean()) if not unique_nc.empty else float(row.get("uniq_mean_unique_nc", 1.0)))) * overall["mean_nc"]),
            "single_model_conservative_score": float((1.0 - (float(unique_nc.max()) if not unique_nc.empty else float(row.get("uniq_max_unique_nc", 1.0)))) * overall["min_nc"]),
        }
        for col in DIRECTION_SCORE_COLS:
            if col in row.index:
                summary[col] = row[col]
        summary_rows.append(summary)

    attack_df = pd.DataFrame(attack_rows)
    summary_df = pd.DataFrame(summary_rows)
    attack_csv = out_dir / "single_model_attack_matrix.csv"
    summary_csv = out_dir / "single_model_all_ability_summary.csv"
    attack_df.to_csv(attack_csv, index=False, encoding="utf-8-sig")
    summary_df.sort_values(["overall_score", "single_model_conservative_score"], ascending=False, inplace=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    rank_parts = []
    for col in DIRECTION_SCORE_COLS:
        if col not in summary_df.columns:
            continue
        tmp = summary_df[["model", "family", col]].copy()
        tmp = tmp.sort_values(col, ascending=False).reset_index(drop=True)
        tmp.insert(0, "direction", col.replace("_score", ""))
        tmp["rank"] = tmp.index + 1
        tmp.rename(columns={col: "score"}, inplace=True)
        rank_parts.append(tmp)
    rank_df = pd.concat(rank_parts, ignore_index=True) if rank_parts else pd.DataFrame()
    rank_csv = out_dir / "single_model_rank_by_direction.csv"
    rank_df.to_csv(rank_csv, index=False, encoding="utf-8-sig")

    comparison_csv = ""
    if ensemble_summary_csv and Path(ensemble_summary_csv).is_file():
        ens = pd.read_csv(ensemble_summary_csv)
        best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        comp = {
            "best_single_model": best.get("model", ""),
            "best_single_robust_mean_nc": best.get("robust_mean_nc", 0.0),
            "best_single_robust_min_nc": best.get("robust_min_nc", 0.0),
            "best_single_unique_max_nc": best.get("unique_max_nc", 1.0),
        }
        if not ens.empty:
            er = ens.iloc[0].to_dict()
            comp.update({
                "ensemble_dedup_mean_nc": er.get("dedup_mean_robust_nc", er.get("mean_robust_nc", 0.0)),
                "ensemble_dedup_min_nc": er.get("dedup_min_robust_nc", er.get("min_robust_nc", 0.0)),
                "ensemble_unique_max_nc": er.get("unique_gate_max_unique_nc", 1.0),
            })
        comparison_csv = str(out_dir / "single_vs_specialist_comparison.csv")
        pd.DataFrame([comp]).to_csv(comparison_csv, index=False, encoding="utf-8-sig")

    result = {
        "single_model_attack_matrix_csv": str(attack_csv),
        "single_model_all_ability_summary_csv": str(summary_csv),
        "single_model_rank_by_direction_csv": str(rank_csv),
        "single_vs_specialist_comparison_csv": comparison_csv,
        "num_models": int(len(summary_df)),
    }
    write_json(out_dir / "single_model_all_ability_summary.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Build V15 single-model all-ability reports for ablation/comparison")
    ap.add_argument("--suite_root", default="", help="Suite root containing compare/model_compare.csv")
    ap.add_argument("--profile_csv", default="", help="Existing model_capability_profile.csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ensemble_summary_csv", default="")
    ns = ap.parse_args()
    try:
        out_dir = Path(ns.output_dir)
        if not ns.profile_csv and not ns.suite_root:
            raise ValueError("Either --suite_root or --profile_csv is required")
        profile_csv = _resolve_profile(Path(ns.suite_root), ns.profile_csv, out_dir) if ns.suite_root else Path(ns.profile_csv)
        result = build_single_model_reports(profile_csv, out_dir, ensemble_summary_csv=ns.ensemble_summary_csv)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] single_model_all_ability_evaluator_V15: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
