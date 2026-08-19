#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V13 direction-specialist ensemble evaluator for RB-AFL suite outputs.

This evaluator composes an ensemble report from already-computed per-model
uniqueness and robustness CSVs.  It does not rerun neural inference; therefore it
is fast and suitable for comparing specialist routing policies after a suite
finishes.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from rb_afl_system.utils import ensure_dir, write_json


ROLE_ATTACK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rotate": ("rotate", "rotation", "rot"),
    "scale": ("scale", "uniform_scale", "nonuniform_scale", "non_uniform_scale", "resize"),
    "jitter": ("jitter", "noise", "perturb"),
    "quantize": ("quantize", "quantization", "round", "precision"),
    "simplify": ("simplify", "mapshaper", "douglas", "dp_simplify"),
    "topology": (
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
    "boundary": (
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

LOCAL_ROLES: tuple[str, ...] = ("jitter", "quantize", "simplify")


class SpecialistEnsembleError(RuntimeError):
    """Raised when ensemble inputs are missing or incompatible."""


def _bool_arg(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return default
    return float(converted)


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _attack_text(df: pd.DataFrame) -> pd.Series:
    text_cols = [c for c in ["attack_type", "attack_engine", "attack_label", "attack_name", "attack_value"] if c in df.columns]
    joined = pd.Series("", index=df.index, dtype="object")
    for col in text_cols:
        joined = joined + " " + df[col].astype(str).str.lower()
    return joined


def _mask_keywords(df: pd.DataFrame, keywords: Iterable[str]) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    text = _attack_text(df)
    keys = tuple(k.lower() for k in keywords)
    return text.map(lambda s: any(k in s for k in keys))


def _load_selected(path: str | Path) -> dict[str, str]:
    payload = _read_json(path)
    if "model_by_role" in payload:
        return {str(k): ("" if v is None else str(v)) for k, v in dict(payload["model_by_role"]).items()}
    if "roles" in payload:
        return {str(k): ("" if v.get("model", "") is None else str(v.get("model", ""))) for k, v in dict(payload["roles"]).items()}
    return {str(k): ("" if v is None else str(v)) for k, v in payload.items()}


def _load_profile(profile_csv: str | Path) -> pd.DataFrame:
    df = _read_csv(profile_csv)
    if "model" not in df.columns:
        raise SpecialistEnsembleError("profile CSV must contain a 'model' column")
    return df


def _dir_for_model(profile_df: pd.DataFrame, model: str, kind: str) -> Path:
    if not model:
        return Path("__missing__")
    sub = profile_df[profile_df["model"].astype(str) == str(model)]
    if sub.empty:
        return Path("__missing__")
    col = "robustness_dir" if kind == "robustness" else "uniqueness_dir"
    if col not in sub.columns:
        return Path("__missing__")
    return Path(str(sub.iloc[0].get(col, "")))


def _load_model_robust_rows(profile_df: pd.DataFrame, model: str) -> pd.DataFrame:
    rob_dir = _dir_for_model(profile_df, model, "robustness")
    for name in ["robustness_rows.csv", "robustness_by_attack_engine_value.csv", "robustness_by_attack.csv"]:
        path = rob_dir / name
        if path.is_file():
            df = pd.read_csv(path)
            df["__source_model"] = model
            df["__source_file"] = str(path)
            return df
    return pd.DataFrame()


def _load_unique_pairs(profile_df: pd.DataFrame, model: str) -> pd.DataFrame:
    uniq_dir = _dir_for_model(profile_df, model, "uniqueness")
    path = uniq_dir / "uniqueness_pairs.csv"
    if path.is_file():
        df = pd.read_csv(path)
        df["__source_model"] = model
        df["__source_file"] = str(path)
        return df
    return pd.DataFrame()


def _standardize_rob_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "feature_nc" not in out.columns and "mean_feature_nc" in out.columns:
        out["feature_nc"] = pd.to_numeric(out["mean_feature_nc"], errors="coerce")
    if "feature_ber" not in out.columns and "mean_feature_ber" in out.columns:
        out["feature_ber"] = pd.to_numeric(out["mean_feature_ber"], errors="coerce")
    if "watermark_nc" not in out.columns and "mean_watermark_nc" in out.columns:
        out["watermark_nc"] = pd.to_numeric(out["mean_watermark_nc"], errors="coerce")
    if "watermark_ber" not in out.columns and "mean_watermark_ber" in out.columns:
        out["watermark_ber"] = pd.to_numeric(out["mean_watermark_ber"], errors="coerce")
    for col in ["feature_nc", "feature_ber", "watermark_nc", "watermark_ber"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _summary_from_rob_rows(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "num_rows": 0,
            "mean_robust_nc": 0.0,
            "min_robust_nc": 0.0,
            "mean_robust_ber": 1.0,
            "max_robust_ber": 1.0,
            "mean_watermark_nc": 0.0,
            "min_watermark_nc": 0.0,
            "mean_watermark_ber": 1.0,
            "max_watermark_ber": 1.0,
        }
    return {
        "num_rows": int(len(df)),
        "mean_robust_nc": float(df["feature_nc"].mean(skipna=True)),
        "min_robust_nc": float(df["feature_nc"].min(skipna=True)),
        "mean_robust_ber": float(df["feature_ber"].mean(skipna=True)),
        "max_robust_ber": float(df["feature_ber"].max(skipna=True)),
        "mean_watermark_nc": float(df["watermark_nc"].mean(skipna=True)) if "watermark_nc" in df.columns else 0.0,
        "min_watermark_nc": float(df["watermark_nc"].min(skipna=True)) if "watermark_nc" in df.columns else 0.0,
        "mean_watermark_ber": float(df["watermark_ber"].mean(skipna=True)) if "watermark_ber" in df.columns else 1.0,
        "max_watermark_ber": float(df["watermark_ber"].max(skipna=True)) if "watermark_ber" in df.columns else 1.0,
    }


def _summary_from_unique_pairs(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "unique_nc" not in df.columns:
        return {
            "unique_gate_num_pairs": 0,
            "unique_gate_mean_unique_nc": 1.0,
            "unique_gate_max_unique_nc": 1.0,
            "unique_gate_nc_gt_0_7": 0,
            "unique_gate_nc_gt_0_8": 0,
            "unique_gate_nc_gt_0_9": 0,
        }
    nc = pd.to_numeric(df["unique_nc"], errors="coerce").fillna(0.0)
    return {
        "unique_gate_num_pairs": int(len(df)),
        "unique_gate_mean_unique_nc": float(nc.mean()),
        "unique_gate_max_unique_nc": float(nc.max()),
        "unique_gate_nc_gt_0_7": int((nc > 0.7).sum()),
        "unique_gate_nc_gt_0_8": int((nc > 0.8).sum()),
        "unique_gate_nc_gt_0_9": int((nc > 0.9).sum()),
    }


def _get_cached_rob_rows(profile_df: pd.DataFrame, cache: dict[str, pd.DataFrame], model: str) -> pd.DataFrame:
    if model not in cache:
        cache[model] = _standardize_rob_rows(_load_model_robust_rows(profile_df, model))
    return cache[model]


def _route_attack_rows(profile_df: pd.DataFrame, model_by_role: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    selected_parts: list[pd.DataFrame] = []
    route_rows: list[dict[str, Any]] = []

    used_index_by_model: dict[str, set[int]] = {}
    for role, keywords in ROLE_ATTACK_KEYWORDS.items():
        model = model_by_role.get(role, "") or (model_by_role.get("local_attack", "") if role in LOCAL_ROLES else "")
        if not model:
            continue
        rows = _get_cached_rob_rows(profile_df, cache, model)
        if rows.empty:
            route_rows.append({"role": role, "model": model, "status": "missing_robustness_rows", "num_rows": 0})
            continue
        mask = _mask_keywords(rows, keywords)
        part = rows[mask].copy()
        if part.empty:
            route_rows.append({"role": role, "model": model, "status": "no_matching_attack", "num_rows": 0})
            continue
        part["specialist_role"] = role
        part["specialist_model"] = model
        selected_parts.append(part)
        used_index_by_model.setdefault(model, set()).update(int(i) for i in part.index)
        stats = _summary_from_rob_rows(part)
        route_rows.append({"role": role, "model": model, "status": "selected", **stats})

    # Fallback for attack rows that did not match any direction keyword.
    fallback_model = model_by_role.get("robust_aux", "") or model_by_role.get("overall", "")
    if fallback_model:
        rows = _get_cached_rob_rows(profile_df, cache, fallback_model)
        if not rows.empty:
            unmatched_mask = pd.Series([True] * len(rows), index=rows.index)
            for keywords in ROLE_ATTACK_KEYWORDS.values():
                unmatched_mask &= ~_mask_keywords(rows, keywords)
            fallback = rows[unmatched_mask].copy()
            if not fallback.empty:
                fallback["specialist_role"] = "fallback_other"
                fallback["specialist_model"] = fallback_model
                selected_parts.append(fallback)
                stats = _summary_from_rob_rows(fallback)
                route_rows.append({"role": "fallback_other", "model": fallback_model, "status": "selected", **stats})

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    by_role = pd.DataFrame(route_rows)
    return selected, by_role


def evaluate_specialist_ensemble(
    profile_csv: str | Path,
    selected_specialists_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    profile_df = _load_profile(profile_csv)
    model_by_role = _load_selected(selected_specialists_json)
    out_dir = ensure_dir(output_dir)

    selected_rows, by_role = _route_attack_rows(profile_df, model_by_role)
    selected_rows = _standardize_rob_rows(selected_rows)
    if not selected_rows.empty:
        selected_rows.to_csv(out_dir / "specialist_ensemble_rows.csv", index=False, encoding="utf-8-sig")
    by_role.to_csv(out_dir / "specialist_ensemble_by_role.csv", index=False, encoding="utf-8-sig")

    unique_model = model_by_role.get("unique_gate", "")
    unique_pairs = _load_unique_pairs(profile_df, unique_model)
    if not unique_pairs.empty:
        unique_pairs.to_csv(out_dir / "specialist_unique_gate_pairs.csv", index=False, encoding="utf-8-sig")

    rob_summary = _summary_from_rob_rows(selected_rows)
    uniq_summary = _summary_from_unique_pairs(unique_pairs)
    mean_unique_nc = _safe_float(uniq_summary.get("unique_gate_mean_unique_nc", 1.0), 1.0)
    max_unique_nc = _safe_float(uniq_summary.get("unique_gate_max_unique_nc", 1.0), 1.0)
    mean_robust_nc = _safe_float(rob_summary.get("mean_robust_nc", 0.0))
    min_robust_nc = _safe_float(rob_summary.get("min_robust_nc", 0.0))
    summary = {
        "profile_csv": str(profile_csv),
        "selected_specialists_json": str(selected_specialists_json),
        "unique_gate_model": unique_model,
        "model_by_role": model_by_role,
        **rob_summary,
        **uniq_summary,
        "specialist_nc_joint_score": float((1.0 - mean_unique_nc) * mean_robust_nc),
        "specialist_nc_conservative_score": float((1.0 - max_unique_nc) * min_robust_nc),
        "specialist_nc_margin_robust_minus_unique": float(mean_robust_nc - mean_unique_nc),
        "note": "This report composes existing per-model eval CSVs; it is a routing/evaluation layer, not neural re-inference.",
    }
    write_json(out_dir / "specialist_ensemble_summary.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "specialist_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "specialist_ensemble_summary_json": str(out_dir / "specialist_ensemble_summary.json"),
        "specialist_ensemble_summary_csv": str(out_dir / "specialist_ensemble_summary.csv"),
        "specialist_ensemble_by_role_csv": str(out_dir / "specialist_ensemble_by_role.csv"),
        "specialist_ensemble_rows_csv": str(out_dir / "specialist_ensemble_rows.csv"),
        "num_rows": int(rob_summary.get("num_rows", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate V13 direction-specialist ensemble from existing RB-AFL eval outputs")
    ap.add_argument("--profile_csv", required=True, help="model_capability_profile.csv")
    ap.add_argument("--selected_json", required=True, help="selected_specialists.json")
    ap.add_argument("--output_dir", default="", help="Output directory; default: selected_json parent/ensemble_eval")
    ns = ap.parse_args()

    try:
        selected_path = Path(ns.selected_json)
        out_dir = Path(ns.output_dir) if ns.output_dir else selected_path.parent / "ensemble_eval"
        summary = evaluate_specialist_ensemble(
            profile_csv=ns.profile_csv,
            selected_specialists_json=selected_path,
            output_dir=out_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] specialist_ensemble_evaluator_V13: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
