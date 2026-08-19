#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V13 direction-specialist capability profiler for RB-AFL model suites.

This script reads existing V10/V11/V11-strong style evaluation outputs and builds
one capability table per model.  It does not retrain models and it does not
change the original suite outputs.

Typical usage:
  python -m rb_afl_system.scripts.capability_profiler_V13 \
    --suite_root runs/model_suite_xxx \
    --output_dir runs/model_suite_xxx/specialists_v12 \
    --auto_select true
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from rb_afl_system.utils import ensure_dir, write_json


ATTACK_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rotate_score": ("rotate", "rotation", "rot"),
    "scale_score": ("scale", "uniform_scale", "nonuniform_scale", "non_uniform_scale", "resize"),
    "jitter_score": ("jitter", "noise", "perturb"),
    "quantize_score": ("quantize", "quantization", "round", "precision"),
    "simplify_score": ("simplify", "mapshaper", "douglas", "dp_simplify"),
    "topology_score": (
        "topology",
        "topology_delete_features",
        "topology_component_drop",
        "topology_clean",
        "repair",
        "clean",
        "delete",
        "component",
        "multipart",
        "hole",
    ),
    "boundary_score": (
        "boundary",
        "boundary_jitter",
        "boundary_simplify",
        "boundary_smooth",
        "smooth",
        "vertex",
        "curve",
        "curvature",
        "edge",
    ),
}

ROLE_TO_PROFILE_METRIC: dict[str, str] = {
    "unique_gate": "unique_score",
    "local_attack": "local_score",
    "rotate": "rotate_score",
    "scale": "scale_score",
    "jitter": "jitter_score",
    "quantize": "quantize_score",
    "simplify": "simplify_score",
    "topology": "topology_score",
    "boundary": "boundary_score",
    "grid_aux": "grid_aux_score",
    "robust_aux": "robust_aux_score",
    "overall": "overall_score",
}

FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GeoVecFormer", ("geovecformer", "fusion")),
    ("Token", ("token", "geotoken")),
    ("Graph", ("graph", "gcn", "graphormer")),
    ("ResNet-SE", ("resnet", "se_", "spectral")),
    ("CNN", ("cnn", "grid")),
    ("Boundary", ("boundary", "dgcnn", "point")),
    ("Descriptor", ("fourier", "zernike", "hu", "moment")),
)


class CapabilityProfilerError(RuntimeError):
    """Raised when profiler inputs are missing or incompatible."""


def _bool_arg(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return default
    out = float(converted)
    if math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    out = _safe_float(value, float(default))
    return int(out)


def _norm_higher(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = vals.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    mn = float(valid.min())
    mx = float(valid.max())
    if mx - mn < 1e-12:
        return pd.Series(np.ones(len(values), dtype=np.float32), index=values.index)
    return ((vals.fillna(mn) - mn) / (mx - mn)).clip(0.0, 1.0)


def _norm_lower(values: pd.Series) -> pd.Series:
    return 1.0 - _norm_higher(values)


def _infer_family(model_name: str) -> str:
    low = model_name.lower()
    for family, keys in FAMILY_KEYWORDS:
        if any(k in low for k in keys):
            return family
    return "Other"


def _find_compare_csv(suite_root: Optional[Path], compare_csv: str) -> Path:
    if compare_csv:
        p = Path(compare_csv)
    elif suite_root is not None:
        p = suite_root / "compare" / "model_compare.csv"
    else:
        raise CapabilityProfilerError("Either --suite_root or --compare_csv is required")
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def _row_path(value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("__missing__")
    return Path(text)


def _resolve_eval_dirs_from_compare(compare_df: pd.DataFrame, suite_root: Optional[Path]) -> pd.DataFrame:
    df = compare_df.copy()
    if "model" not in df.columns:
        raise CapabilityProfilerError("model_compare.csv must contain a 'model' column")
    for col, prefix in (("uniqueness_dir", "uniqueness_"), ("robustness_dir", "robustness_")):
        if col not in df.columns:
            if suite_root is None:
                raise CapabilityProfilerError(f"Missing column {col!r}; provide --suite_root so it can be inferred")
            df[col] = df["model"].map(lambda m: str(suite_root / "evals" / f"{prefix}{m}"))
    return df


def _count_unique_thresholds(pair_df: pd.DataFrame) -> dict[str, int]:
    if pair_df.empty or "unique_nc" not in pair_df.columns:
        return {"uniq_pair_nc_gt_0_7": 0, "uniq_pair_nc_gt_0_8": 0, "uniq_pair_nc_gt_0_9": 0}
    nc = pd.to_numeric(pair_df["unique_nc"], errors="coerce").fillna(0.0)
    return {
        "uniq_pair_nc_gt_0_7": int((nc > 0.7).sum()),
        "uniq_pair_nc_gt_0_8": int((nc > 0.8).sum()),
        "uniq_pair_nc_gt_0_9": int((nc > 0.9).sum()),
    }


def _attack_mask(df: pd.DataFrame, keywords: Iterable[str]) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    text_cols = [c for c in ["attack_type", "attack_engine", "attack_label", "attack_name", "attack_value"] if c in df.columns]
    if not text_cols:
        return pd.Series([False] * len(df), index=df.index)
    joined = pd.Series("", index=df.index, dtype="object")
    for col in text_cols:
        joined = joined + " " + df[col].astype(str).str.lower()
    return joined.map(lambda s: any(k.lower() in s for k in keywords))


def _metric_from_attack_rows(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"mean_nc": 0.0, "min_nc": 0.0, "mean_ber": 1.0, "max_ber": 1.0, "count": 0.0}
    nc_col = "feature_nc" if "feature_nc" in df.columns else "mean_feature_nc"
    ber_col = "feature_ber" if "feature_ber" in df.columns else "mean_feature_ber"
    nc = pd.to_numeric(df.get(nc_col, pd.Series(dtype=float)), errors="coerce").dropna()
    ber = pd.to_numeric(df.get(ber_col, pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "mean_nc": float(nc.mean()) if not nc.empty else 0.0,
        "min_nc": float(nc.min()) if not nc.empty else 0.0,
        "mean_ber": float(ber.mean()) if not ber.empty else 1.0,
        "max_ber": float(ber.max()) if not ber.empty else 1.0,
        "count": float(len(df)),
    }


def _score_attack_subset(df: pd.DataFrame, keywords: Iterable[str]) -> tuple[float, dict[str, float]]:
    if df.empty:
        metrics = _metric_from_attack_rows(df)
        return 0.0, metrics
    subset = df[_attack_mask(df, keywords)]
    metrics = _metric_from_attack_rows(subset)
    # Direction robustness is worst-case oriented: min NC dominates, mean NC stabilizes.
    score = 0.70 * metrics["min_nc"] + 0.30 * metrics["mean_nc"]
    return float(score), metrics


def _load_eval_details(row: pd.Series) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uniq_dir = _row_path(row.get("uniqueness_dir", ""))
    rob_dir = _row_path(row.get("robustness_dir", ""))
    uniq_summary = _read_json_if_exists(uniq_dir / "uniqueness_summary.json")
    rob_summary = _read_json_if_exists(rob_dir / "robustness_summary.json")
    uniq_pairs = _read_csv_if_exists(uniq_dir / "uniqueness_pairs.csv")
    rob_rows = _read_csv_if_exists(rob_dir / "robustness_rows.csv")
    rob_by_attack = _read_csv_if_exists(rob_dir / "robustness_by_attack.csv")
    return uniq_summary, rob_summary, uniq_pairs, rob_rows, rob_by_attack


def build_capability_profile(compare_csv: str | Path, suite_root: str | Path | None, output_dir: str | Path) -> dict[str, Any]:
    """Build and save model_capability_profile.csv/json from a model_compare.csv."""
    suite_path = Path(suite_root) if suite_root else None
    compare_df = pd.read_csv(compare_csv)
    compare_df = _resolve_eval_dirs_from_compare(compare_df, suite_path)
    rows: list[dict[str, Any]] = []

    for _, src in compare_df.iterrows():
        model = str(src["model"])
        uniq_summary, rob_summary, uniq_pairs, rob_rows, rob_by_attack = _load_eval_details(src)
        threshold_counts = _count_unique_thresholds(uniq_pairs)
        detail_df = rob_rows if not rob_rows.empty else rob_by_attack

        attack_scores: dict[str, float] = {}
        attack_metrics: dict[str, float] = {}
        for score_col, keys in ATTACK_ROLE_KEYWORDS.items():
            score, metrics = _score_attack_subset(detail_df, keys)
            attack_scores[score_col] = score
            prefix = score_col.replace("_score", "")
            for metric_name, metric_value in metrics.items():
                attack_metrics[f"{prefix}_{metric_name}"] = metric_value

        local_components = [attack_scores.get(x, 0.0) for x in ("jitter_score", "quantize_score", "simplify_score")]
        local_valid = [v for v in local_components if v > 0]
        local_score_raw = float(np.mean(local_valid)) if local_valid else 0.0
        robust_aux_raw = 0.65 * _safe_float(rob_summary.get("min_robust_nc", src.get("rob_min_robust_nc", 0.0))) + 0.35 * _safe_float(rob_summary.get("mean_robust_nc", src.get("rob_mean_robust_nc", 0.0)))
        row = {
            "model": model,
            "family": _infer_family(model),
            "uniqueness_dir": str(src.get("uniqueness_dir", "")),
            "robustness_dir": str(src.get("robustness_dir", "")),
            "uniq_mean_unique_nc": _safe_float(uniq_summary.get("mean_unique_nc", src.get("uniq_mean_unique_nc", 1.0)), 1.0),
            "uniq_max_unique_nc": _safe_float(uniq_summary.get("max_unique_nc", src.get("uniq_max_unique_nc", 1.0)), 1.0),
            "uniq_mean_unique_ber": _safe_float(uniq_summary.get("mean_unique_ber", src.get("uniq_mean_unique_ber", 0.0))),
            "uniq_min_unique_ber": _safe_float(uniq_summary.get("min_unique_ber", src.get("uniq_min_unique_ber", 0.0))),
            "rob_mean_robust_nc": _safe_float(rob_summary.get("mean_robust_nc", src.get("rob_mean_robust_nc", 0.0))),
            "rob_min_robust_nc": _safe_float(rob_summary.get("min_robust_nc", src.get("rob_min_robust_nc", 0.0))),
            "rob_mean_robust_ber": _safe_float(rob_summary.get("mean_robust_ber", src.get("rob_mean_robust_ber", 1.0)), 1.0),
            "rob_max_robust_ber": _safe_float(rob_summary.get("max_robust_ber", src.get("rob_max_robust_ber", 1.0)), 1.0),
            "num_unique_pairs": int(len(uniq_pairs)) if not uniq_pairs.empty else 0,
            "num_robust_rows": _safe_int(rob_summary.get("num_rows", len(detail_df))),
            "local_score_raw": local_score_raw,
            "grid_aux_score_raw": robust_aux_raw,
            "robust_aux_score_raw": robust_aux_raw,
            **threshold_counts,
            **attack_scores,
            **attack_metrics,
        }
        rows.append(row)

    if not rows:
        raise CapabilityProfilerError("No model rows were found in compare CSV")

    df = pd.DataFrame(rows)
    # Unique score: penalize worst-case unique NC and high-NC hard-negative pair counts.
    df["unique_score"] = (
        0.50 * _norm_lower(df["uniq_max_unique_nc"])
        + 0.20 * _norm_lower(df["uniq_mean_unique_nc"])
        + 0.15 * _norm_lower(df["uniq_pair_nc_gt_0_7"])
        + 0.10 * _norm_lower(df["uniq_pair_nc_gt_0_8"])
        + 0.05 * _norm_lower(df["uniq_pair_nc_gt_0_9"])
    ).clip(0.0, 1.0)
    for col in ["local_score_raw", "grid_aux_score_raw", "robust_aux_score_raw"]:
        out_col = col.replace("_raw", "")
        if float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()) > 0.0:
            df[out_col] = _norm_higher(df[col])
        else:
            df[out_col] = 0.0
    for col in ATTACK_ROLE_KEYWORDS:
        # If no rows matched a direction, the score remains 0.  This makes absence explicit.
        if float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()) > 0.0:
            df[col] = _norm_higher(df[col])
        else:
            df[col] = 0.0

    direction_cols = [
        "unique_score",
        "local_score",
        "rotate_score",
        "scale_score",
        "jitter_score",
        "quantize_score",
        "simplify_score",
        "topology_score",
        "boundary_score",
        "robust_aux_score",
    ]
    df["overall_score"] = df[direction_cols].mean(axis=1).clip(0.0, 1.0)
    df["recommended_role"] = df[direction_cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    df = df.sort_values(["overall_score", "unique_score", "robust_aux_score"], ascending=False).reset_index(drop=True)

    out_dir = ensure_dir(output_dir)
    csv_path = out_dir / "model_capability_profile.csv"
    json_path = out_dir / "model_capability_profile.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(df.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "model_capability_profile_csv": str(csv_path),
        "model_capability_profile_json": str(json_path),
        "num_models": int(len(df)),
        "score_columns": direction_cols,
    }
    write_json(out_dir / "capability_profiler_summary.json", summary)
    return summary


def _select_specialists_from_profile(profile_csv: Path, output_dir: Path, selection_policy: str) -> dict[str, Any]:
    # Local import keeps the profiler usable even if selector is copied alone.
    from rb_afl_system.scripts.specialist_selector_V13 import select_specialists

    return select_specialists(profile_csv=profile_csv, output_dir=output_dir, selection_policy=selection_policy)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build V13 direction-specialist capability profile from RB-AFL suite outputs")
    ap.add_argument("--suite_root", default="", help="Suite root containing compare/model_compare.csv and evals/*")
    ap.add_argument("--compare_csv", default="", help="Optional explicit model_compare.csv path")
    ap.add_argument("--output_dir", default="", help="Output directory; default: suite_root/specialists_v12")
    ap.add_argument("--auto_select", type=_bool_arg, default=True, help="Also write selected_specialists.json")
    ap.add_argument("--selection_policy", default="gated", choices=["gated", "strict", "role_only"], help="Selection policy used when --auto_select=true")
    ns = ap.parse_args()

    try:
        suite_root = Path(ns.suite_root) if ns.suite_root else None
        compare_csv = _find_compare_csv(suite_root, ns.compare_csv)
        if ns.output_dir:
            output_dir = Path(ns.output_dir)
        elif suite_root is not None:
            output_dir = suite_root / "specialists_v12"
        else:
            output_dir = Path(compare_csv).parent / "specialists_v12"
        summary = build_capability_profile(compare_csv=compare_csv, suite_root=suite_root, output_dir=output_dir)
        if ns.auto_select:
            selected = _select_specialists_from_profile(Path(summary["model_capability_profile_csv"]), output_dir, selection_policy=str(ns.selection_policy))
            summary["selected_specialists_json"] = selected.get("selected_specialists_json", "")
            summary["selected_specialists_csv"] = selected.get("selected_specialists_csv", "")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] capability_profiler_V13: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
