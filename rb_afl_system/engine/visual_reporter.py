#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual reporting utilities for RB-AFL experiment suites.

This module intentionally works from exported files instead of model objects:
- compare/model_compare.csv
- runs/*/history.json

It can be used after any completed suite without re-training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import re

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rb_afl_system.utils import ensure_dir


NUMERIC_COLUMNS = [
    "uniq_mean_unique_nc",
    "uniq_max_unique_nc",
    "uniq_mean_unique_ber",
    "uniq_min_unique_ber",
    "rob_mean_robust_nc",
    "rob_min_robust_nc",
    "rob_mean_robust_ber",
    "rob_max_robust_ber",
    "rob_mean_watermark_nc",
    "rob_min_watermark_nc",
    "rob_mean_watermark_ber",
    "rob_max_watermark_ber",
    "joint_score_uniqueBER_x_robustNC",
    "conservative_joint_score",
    "ber_margin_unique_minus_robust",
    "nc_margin_robust_minus_unique",
]

CORE_COLUMNS = [
    "model",
    "family",
    "uniq_mean_unique_nc",
    "uniq_max_unique_nc",
    "rob_mean_robust_nc",
    "rob_min_robust_nc",
    "nc_joint_score",
    "nc_conservative_score",
    "uniq_mean_unique_ber",
    "rob_mean_robust_ber",
    "rob_worst_identity",
    "rob_worst_attack_type",
    "rob_worst_attack_engine",
    "rob_worst_attack_value",
    "rob_worst_feature_nc",
]


def _safe_timestamp_suffix(timestamp: str | None) -> str:
    if not timestamp:
        return ""
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", str(timestamp)).strip("_")
    return f"_{cleaned}" if cleaned else ""


def _to_numeric_columns(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _family_name(model_name: str) -> str:
    # Common names: xxx_seed20260318_90ep or xxx_seed20260318
    name = re.sub(r"_seed\d+(_\d+ep)?$", "", str(model_name))
    name = re.sub(r"_\d+ep$", "", name)
    return name


def build_nc_priority_tables(compare_csv: str | Path, output_dir: str | Path, timestamp: str | None = None, output_xlsx: str | Path | None = None) -> dict[str, Any]:
    """Create NC-priority CSV/XLSX tables from a model_compare.csv.

    Unique NC is treated as the primary uniqueness metric:
      nc_joint_score = (1 - mean_unique_nc) * mean_robust_nc
      nc_conservative_score = (1 - max_unique_nc) * min_robust_nc
    """
    src = Path(compare_csv)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    out_dir = ensure_dir(output_dir)
    suffix = _safe_timestamp_suffix(timestamp)

    df = pd.read_csv(src)
    df = _to_numeric_columns(df)
    if "model" not in df.columns:
        raise ValueError(f"compare csv has no model column: {src}")

    required = ["uniq_mean_unique_nc", "uniq_max_unique_nc", "rob_mean_robust_nc", "rob_min_robust_nc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns for NC-priority report: {missing}")

    df["family"] = df["model"].map(_family_name)
    df["nc_joint_score"] = (1.0 - df["uniq_mean_unique_nc"]) * df["rob_mean_robust_nc"]
    df["nc_conservative_score"] = (1.0 - df["uniq_max_unique_nc"]) * df["rob_min_robust_nc"]
    df["unique_nc_margin"] = 1.0 - df["uniq_mean_unique_nc"]
    df["worst_unique_nc_margin"] = 1.0 - df["uniq_max_unique_nc"]

    sorted_df = df.sort_values(
        ["nc_joint_score", "nc_conservative_score", "rob_min_robust_nc"],
        ascending=[False, False, False],
    )

    family = df.groupby("family").agg(
        runs=("model", "count"),
        uniq_nc_mean=("uniq_mean_unique_nc", "mean"),
        uniq_nc_std=("uniq_mean_unique_nc", "std"),
        uniq_max_nc_mean=("uniq_max_unique_nc", "mean"),
        uniq_max_nc_max=("uniq_max_unique_nc", "max"),
        rob_nc_mean=("rob_mean_robust_nc", "mean"),
        rob_nc_std=("rob_mean_robust_nc", "std"),
        rob_min_nc_mean=("rob_min_robust_nc", "mean"),
        rob_min_nc_min=("rob_min_robust_nc", "min"),
        nc_joint_mean=("nc_joint_score", "mean"),
        nc_joint_std=("nc_joint_score", "std"),
        nc_conservative_mean=("nc_conservative_score", "mean"),
        uniq_ber_mean=("uniq_mean_unique_ber", "mean") if "uniq_mean_unique_ber" in df.columns else ("model", "count"),
        rob_ber_mean=("rob_mean_robust_ber", "mean") if "rob_mean_robust_ber" in df.columns else ("model", "count"),
    ).reset_index()
    family = family.sort_values(
        ["nc_joint_mean", "nc_conservative_mean", "rob_min_nc_mean"],
        ascending=[False, False, False],
    )

    core_cols = [c for c in CORE_COLUMNS if c in sorted_df.columns]
    all_runs_csv = out_dir / f"model_compare_nc_priority{suffix}.csv"
    family_csv = out_dir / f"family_summary_nc_priority{suffix}.csv"
    sorted_df[core_cols].to_csv(all_runs_csv, index=False, encoding="utf-8-sig")
    family.to_csv(family_csv, index=False, encoding="utf-8-sig")

    xlsx_path = Path(output_xlsx) if output_xlsx else out_dir / f"nc_priority{suffix}.xlsx"
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        sorted_df[core_cols].to_excel(writer, sheet_name="all_runs_nc_priority", index=False)
        family.to_excel(writer, sheet_name="family_summary", index=False)
        sorted_df.to_excel(writer, sheet_name="all_fields", index=False)

    return {
        "model_compare_nc_priority_csv": str(all_runs_csv),
        "family_summary_nc_priority_csv": str(family_csv),
        "nc_priority_xlsx": str(xlsx_path),
        "num_runs": int(len(sorted_df)),
        "num_families": int(len(family)),
    }


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, rotate: int = 60) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    plt.figure(figsize=(max(10, min(24, len(df) * 0.8)), 6))
    plt.bar(df[x_col].astype(str), pd.to_numeric(df[y_col], errors="coerce"))
    plt.xticks(rotation=rotate, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_history(history_path: Path, out_dir: Path) -> str | None:
    try:
        hist = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(hist, list) or not hist:
            return None
        rows: list[dict[str, Any]] = []
        for item in hist:
            if not isinstance(item, dict):
                continue
            row: dict[str, Any] = {"epoch": item.get("epoch", len(rows) + 1)}
            train = item.get("train", {}) if isinstance(item.get("train", {}), dict) else {}
            val = item.get("val", {}) if isinstance(item.get("val", {}), dict) else {}
            for key, value in train.items():
                row[f"train_{key}"] = value
            for key, value in val.items():
                row[f"val_{key}"] = value
            for key, value in item.items():
                if key not in {"train", "val"}:
                    row[key] = value
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        if "epoch" not in df.columns:
            df["epoch"] = range(1, len(df) + 1)
        run_name = history_path.parent.name
        plt.figure(figsize=(10, 6))
        plotted = False
        for col in [
            "train_loss_g",
            "val_loss_g",
            "val_loss_d",
            "val_loss_cons",
            "val_loss_triplet",
            "val_loss_supcon",
            "val_loss_hard_neg",
            "val_loss_hard_pos",
            "val_d_acc",
            "train_loss_hard_neg",
            "train_loss_hard_pos",
            "train_g",
            "val_g",
            "val_d",
            "val_cons",
            "val_trip",
            "val_supcon",
        ]:
            if col in df.columns:
                plt.plot(df["epoch"], pd.to_numeric(df[col], errors="coerce"), label=col)
                plotted = True
        if not plotted:
            plt.close()
            return None
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"Training Curves - {run_name}")
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"loss_curve_{run_name}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        return str(out_path)
    except Exception:
        return None


def export_suite_visual_report(suite_root: str | Path, output_dir: str | Path | None = None, timestamp: str | None = None) -> dict[str, Any]:
    """Export NC-priority tables, loss curves, bar charts, and summary.md.

    Parameters
    ----------
    suite_root:
        A run_model_suite output root containing compare/model_compare.csv and runs/*/history.json.
    output_dir:
        Destination for visuals. Defaults to suite_root/visuals.
    timestamp:
        Optional suffix for generated report files.
    """
    root = Path(suite_root)
    compare_csv = root / "compare" / "model_compare.csv"
    if not compare_csv.is_file():
        raise FileNotFoundError(str(compare_csv))
    vis_dir = ensure_dir(output_dir or root / "visuals")
    compare_dir = ensure_dir(root / "compare")
    suffix = _safe_timestamp_suffix(timestamp)

    report = build_nc_priority_tables(compare_csv, compare_dir, timestamp=timestamp, output_xlsx=vis_dir / f"nc_priority{suffix}.xlsx")
    nc_csv = Path(report["model_compare_nc_priority_csv"])
    family_csv = Path(report["family_summary_nc_priority_csv"])

    df = pd.read_csv(nc_csv)
    df = _to_numeric_columns(df, NUMERIC_COLUMNS + ["nc_joint_score", "nc_conservative_score"])
    if "nc_joint_score" in df.columns:
        df = df.sort_values("nc_joint_score", ascending=False)

    family = pd.read_csv(family_csv) if family_csv.is_file() else pd.DataFrame()
    family = _to_numeric_columns(family, ["uniq_nc_mean", "rob_nc_mean", "rob_min_nc_mean", "nc_joint_mean", "nc_conservative_mean"])

    history_plots: list[str] = []
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        loss_dir = ensure_dir(vis_dir / "loss_curves")
        for history_path in sorted(runs_dir.rglob("history.json")):
            plot_path = _plot_history(history_path, loss_dir)
            if plot_path:
                history_plots.append(plot_path)

    bar_dir = ensure_dir(vis_dir / "bars")
    for col in [
        "uniq_mean_unique_nc",
        "uniq_max_unique_nc",
        "rob_mean_robust_nc",
        "rob_min_robust_nc",
        "nc_joint_score",
        "nc_conservative_score",
    ]:
        _plot_bar(df, "model", col, col, bar_dir / f"bar_{col}.png")

    if not family.empty:
        for col in ["uniq_nc_mean", "uniq_max_nc_mean", "rob_nc_mean", "rob_min_nc_mean", "nc_joint_mean", "nc_conservative_mean"]:
            _plot_bar(family, "family", col, f"Family - {col}", bar_dir / f"bar_family_{col}.png", rotate=45)

    lines: list[str] = [f"# RB-AFL Visual Report {timestamp or ''}".strip(), ""]
    if not df.empty:
        top = df.iloc[0]
        lines.extend([
            "## Top run by NC-priority joint score",
            f"- model: {top.get('model', '')}",
            f"- uniq_mean_unique_nc: {top.get('uniq_mean_unique_nc', '')}",
            f"- uniq_max_unique_nc: {top.get('uniq_max_unique_nc', '')}",
            f"- rob_mean_robust_nc: {top.get('rob_mean_robust_nc', '')}",
            f"- rob_min_robust_nc: {top.get('rob_min_robust_nc', '')}",
            f"- nc_joint_score: {top.get('nc_joint_score', '')}",
            "",
        ])
    if not family.empty:
        fam_top = family.sort_values("nc_joint_mean", ascending=False).iloc[0]
        lines.extend([
            "## Top family by mean NC-priority score",
            f"- family: {fam_top.get('family', '')}",
            f"- uniq_nc_mean: {fam_top.get('uniq_nc_mean', '')}",
            f"- rob_nc_mean: {fam_top.get('rob_nc_mean', '')}",
            f"- rob_min_nc_mean: {fam_top.get('rob_min_nc_mean', '')}",
            f"- nc_joint_mean: {fam_top.get('nc_joint_mean', '')}",
            "",
        ])
    lines.extend([
        "## Generated files",
        f"- NC-priority CSV: {report['model_compare_nc_priority_csv']}",
        f"- Family CSV: {report['family_summary_nc_priority_csv']}",
        f"- NC-priority XLSX: {report['nc_priority_xlsx']}",
        f"- Loss curve count: {len(history_plots)}",
    ])
    summary_path = vis_dir / f"summary{suffix}.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    report.update({
        "visual_dir": str(vis_dir),
        "summary_md": str(summary_path),
        "loss_curve_count": len(history_plots),
    })
    return report
