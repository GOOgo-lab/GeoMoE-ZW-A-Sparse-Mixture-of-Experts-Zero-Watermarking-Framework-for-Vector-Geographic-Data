#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paper-ready tables and curves for attack strength-sweep robustness.

Input is an evaluation output directory containing robustness_rows.csv produced
by rb_afl_system.engine.robustness_evaluator.evaluate_robustness().
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from rb_afl_system.utils import ensure_dir, write_json


SPAN_BASED_ATTACKS = {
    "translate",
    "simplify",
    "quantize",
    "jitter",
    "topology_clean",
    "boundary_jitter",
    "boundary_simplify",
    "boundary_smooth",
}


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_strength_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "attack_value_original" not in out.columns:
        out["attack_value_original"] = out.get("attack_value", "")
    if "attack_value_mode" not in out.columns:
        out["attack_value_mode"] = ""
    out["strength_raw"] = _to_numeric(out["attack_value_original"])
    missing = out["strength_raw"].isna()
    if missing.any():
        out.loc[missing, "strength_raw"] = _to_numeric(out.loc[missing, "attack_value"])
    out["strength_resolved"] = _to_numeric(out.get("attack_value", pd.Series(index=out.index, dtype=object)))
    out["strength_mode"] = out["attack_value_mode"].fillna("").astype(str)
    out.loc[out["strength_mode"].eq(""), "strength_mode"] = out.loc[out["strength_mode"].eq(""), "attack_type"].map(
        lambda t: "span_ratio" if str(t) in SPAN_BASED_ATTACKS else "absolute"
    )

    # Rank unique raw strengths inside each attack type as L1..Ln.
    out["strength_level"] = 0
    for attack_type, group in out.groupby("attack_type", dropna=False):
        unique_values = sorted(v for v in group["strength_raw"].dropna().unique())
        mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
        mask = out["attack_type"].eq(attack_type)
        out.loc[mask, "strength_level"] = out.loc[mask, "strength_raw"].map(mapping).fillna(0).astype(int)
    return out


def _aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["attack_type", "strength_level", "strength_raw", "strength_mode"]
    summary = rows.groupby(group_cols, dropna=False).agg(
        mean_feature_nc=("feature_nc", "mean"),
        min_feature_nc=("feature_nc", "min"),
        std_feature_nc=("feature_nc", "std"),
        mean_feature_ber=("feature_ber", "mean"),
        max_feature_ber=("feature_ber", "max"),
        mean_watermark_nc=("watermark_nc", "mean"),
        min_watermark_nc=("watermark_nc", "min"),
        mean_watermark_ber=("watermark_ber", "mean"),
        max_watermark_ber=("watermark_ber", "max"),
        count=("identity", "count"),
        nc_lt_09=("feature_nc", lambda s: int((s < 0.9).sum())),
        nc_lt_08=("feature_nc", lambda s: int((s < 0.8).sum())),
    ).reset_index()
    summary["nc_lt_09_rate"] = summary["nc_lt_09"] / summary["count"].clip(lower=1)
    summary["nc_lt_08_rate"] = summary["nc_lt_08"] / summary["count"].clip(lower=1)
    return summary.sort_values(["attack_type", "strength_level"]).reset_index(drop=True)


def _plot_curves(summary: pd.DataFrame, out_dir: Path) -> list[str]:
    plot_dir = ensure_dir(out_dir / "curves")
    saved: list[str] = []

    for metric in ["mean_feature_nc", "min_feature_nc", "mean_feature_ber", "max_feature_ber"]:
        plt.figure(figsize=(10, 6))
        for attack_type, group in summary.groupby("attack_type"):
            group = group.sort_values("strength_level")
            plt.plot(group["strength_level"], group[metric], marker="o", label=str(attack_type))
        plt.xlabel("Attack strength level")
        plt.ylabel(metric)
        plt.xticks(sorted(summary["strength_level"].dropna().astype(int).unique()))
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        path = plot_dir / f"{metric}_all_attacks.png"
        plt.savefig(path, dpi=220)
        plt.close()
        saved.append(str(path))

    # One clear per-attack min/mean NC chart for paper appendix.
    for attack_type, group in summary.groupby("attack_type"):
        group = group.sort_values("strength_level")
        plt.figure(figsize=(6, 4))
        plt.plot(group["strength_level"], group["mean_feature_nc"], marker="o", label="mean NC")
        plt.plot(group["strength_level"], group["min_feature_nc"], marker="s", label="min NC")
        plt.xlabel("Attack strength level")
        plt.ylabel("Feature NC")
        plt.title(str(attack_type))
        plt.xticks(sorted(summary["strength_level"].dropna().astype(int).unique()))
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        safe_name = str(attack_type).replace("/", "_").replace("\\", "_")
        path = plot_dir / f"{safe_name}_nc_curve.png"
        plt.savefig(path, dpi=220)
        plt.close()
        saved.append(str(path))

    return saved


def build_attack_strength_sweep_report(
    robustness_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    rob_dir = Path(robustness_dir)
    rows_csv = rob_dir / "robustness_rows.csv"
    if not rows_csv.is_file():
        raise FileNotFoundError(str(rows_csv))
    out_dir = ensure_dir(output_dir or (rob_dir / "attack_strength_sweep_report"))

    rows = pd.read_csv(rows_csv)
    if rows.empty:
        raise ValueError(f"Empty robustness rows: {rows_csv}")
    for col in ["feature_nc", "feature_ber", "watermark_nc", "watermark_ber"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")

    rows = _parse_strength_columns(rows)
    rows.to_csv(out_dir / "attack_strength_sweep_rows.csv", index=False, encoding="utf-8-sig")

    summary = _aggregate(rows)
    summary.to_csv(out_dir / "attack_strength_sweep_summary.csv", index=False, encoding="utf-8-sig")

    for metric in ["mean_feature_nc", "min_feature_nc", "mean_feature_ber", "max_feature_ber"]:
        pivot = summary.pivot(index="attack_type", columns="strength_level", values=metric)
        pivot.to_csv(out_dir / f"pivot_{metric}.csv", encoding="utf-8-sig")

    saved_plots = _plot_curves(summary, out_dir)

    meta = {
        "robustness_dir": str(rob_dir),
        "output_dir": str(out_dir),
        "rows_csv": str(out_dir / "attack_strength_sweep_rows.csv"),
        "summary_csv": str(out_dir / "attack_strength_sweep_summary.csv"),
        "num_rows": int(len(rows)),
        "num_attack_types": int(summary["attack_type"].nunique()),
        "plots": saved_plots,
    }
    write_json(out_dir / "attack_strength_sweep_report.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Create attack strength-sweep tables and curves")
    ap.add_argument("--robustness_dir", required=True)
    ap.add_argument("--output_dir", default="")
    ns = ap.parse_args()

    try:
        summary = build_attack_strength_sweep_report(
            robustness_dir=ns.robustness_dir,
            output_dir=ns.output_dir or None,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] attack_strength_sweep_report_V19: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
