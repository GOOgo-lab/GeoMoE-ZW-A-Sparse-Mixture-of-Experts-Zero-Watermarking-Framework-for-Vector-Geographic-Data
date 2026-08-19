#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V13 specialist selector for RB-AFL capability profiles.

V13 fixes two issues from the first V12 specialist selector:

1. Direction specialists are selected by their direction robustness first.  In a
   gated multi-instance watermark design, W_unique handles cross-identity
   separation, so a robust specialist is not forced to be the best uniqueness
   model.  The selector still reports uniqueness risk, but it does not silently
   replace a stronger robust model unless the user explicitly selects
   --selection_policy strict.
2. Roles with no attack rows are marked as no_attack_data.  They are no longer
   filled by an arbitrary zero-score fallback model.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from rb_afl_system.utils import ensure_dir, write_json


DEFAULT_ROLE_METRICS: dict[str, str] = {
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

DEFAULT_MIN_SCORE: dict[str, float] = {
    "unique_gate": 0.0,
    "local_attack": 0.0,
    "rotate": 0.0,
    "scale": 0.0,
    "jitter": 0.0,
    "quantize": 0.0,
    "simplify": 0.0,
    "topology": 0.0,
    "boundary": 0.0,
    "grid_aux": 0.0,
    "robust_aux": 0.0,
    "overall": 0.0,
}

ROBUST_ROLES: set[str] = {
    "local_attack",
    "rotate",
    "scale",
    "jitter",
    "quantize",
    "simplify",
    "topology",
    "boundary",
    "grid_aux",
    "robust_aux",
}

ROLE_TO_ATTACK_COUNT_COL: dict[str, str] = {
    "local_attack": "__local_count__",
    "rotate": "rotate_count",
    "scale": "scale_count",
    "jitter": "jitter_count",
    "quantize": "quantize_count",
    "simplify": "simplify_count",
    "topology": "topology_count",
    "boundary": "boundary_count",
}


class SpecialistSelectorError(RuntimeError):
    """Raised when selector input is invalid."""


def _bool_arg(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def _load_role_config(path: str) -> tuple[dict[str, str], dict[str, float]]:
    role_metrics = dict(DEFAULT_ROLE_METRICS)
    min_scores = dict(DEFAULT_MIN_SCORE)
    if not path:
        return role_metrics, min_scores
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    cfg = json.loads(p.read_text(encoding="utf-8"))
    role_metrics.update({str(k): str(v) for k, v in dict(cfg.get("role_metrics", {})).items()})
    min_scores.update({str(k): float(v) for k, v in dict(cfg.get("min_scores", {})).items()})
    return role_metrics, min_scores


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return default
    return float(converted)


def _uniqueness_risk(row: dict[str, Any]) -> str:
    uniq_max = _safe_float(row.get("uniq_max_unique_nc", 1.0), 1.0)
    gt07 = _safe_float(row.get("uniq_pair_nc_gt_0_7", 0.0), 0.0)
    gt08 = _safe_float(row.get("uniq_pair_nc_gt_0_8", 0.0), 0.0)
    gt09 = _safe_float(row.get("uniq_pair_nc_gt_0_9", 0.0), 0.0)
    if uniq_max >= 0.90 or gt09 > 0:
        return "high"
    if uniq_max >= 0.82 or gt08 > 0:
        return "medium"
    if uniq_max >= 0.75 or gt07 > 0:
        return "low"
    return "safe"


def _has_attack_data(df: pd.DataFrame, role: str, metric: str) -> bool:
    count_col = ROLE_TO_ATTACK_COUNT_COL.get(role, "")
    if count_col == "__local_count__":
        local_cols = [c for c in ["jitter_count", "quantize_count", "simplify_count"] if c in df.columns]
        if local_cols:
            counts = sum(pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in local_cols)
            return bool((counts > 0).any())
    if count_col and count_col in df.columns:
        counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0)
        return bool((counts > 0).any())
    if metric in df.columns:
        values = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        return bool((values > 0).any())
    return False


def _strict_uniqueness_filter(work: pd.DataFrame, role: str) -> pd.DataFrame:
    """Optional strict mode for papers that require each subwatermark to stand alone."""
    if role == "rotate":
        threshold = 0.95
    elif role in {"scale", "topology", "boundary"}:
        threshold = 0.88
    else:
        threshold = 0.82
    if "uniq_max_unique_nc" not in work.columns:
        return work
    safe = work[pd.to_numeric(work["uniq_max_unique_nc"], errors="coerce").fillna(1.0) <= threshold].copy()
    return safe if not safe.empty else work


def _select_one(df: pd.DataFrame, role: str, metric: str, min_score: float, selection_policy: str) -> dict[str, Any]:
    if metric not in df.columns:
        return {"model": "", "metric": metric, "score": 0.0, "status": "missing_metric"}
    if role in ROLE_TO_ATTACK_COUNT_COL and not _has_attack_data(df, role, metric):
        return {"model": "", "metric": metric, "score": 0.0, "status": "no_attack_data"}

    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce").fillna(0.0)
    for col in ["overall_score", "unique_score", "robust_aux_score", "uniq_max_unique_nc"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    if selection_policy == "strict" and role in ROBUST_ROLES:
        work = _strict_uniqueness_filter(work, role)

    # gated / role_only: direction robustness is the first key. unique_score is a
    # tie-breaker only. This matches the separated W_unique + W_direction design.
    if selection_policy == "role_only" and role in ROBUST_ROLES:
        sort_cols = [metric, "robust_aux_score", "overall_score"]
    else:
        sort_cols = [metric, "robust_aux_score", "overall_score", "unique_score"]
    work = work.sort_values(sort_cols, ascending=False)

    if work.empty:
        return {"model": "", "metric": metric, "score": 0.0, "status": "empty_profile"}
    best = work.iloc[0].to_dict()
    score = _safe_float(best.get(metric, 0.0))
    if score < min_score:
        status = "below_min_score"
    elif score <= 0.0:
        status = "zero_score_no_positive_metric"
    else:
        status = "selected"
    risk = _uniqueness_risk(best)
    if role in ROBUST_ROLES and selection_policy in {"gated", "role_only"} and risk in {"medium", "high"}:
        status = f"{status}_unique_risk_{risk}"
    return {
        "model": str(best.get("model", "")),
        "family": str(best.get("family", "")),
        "metric": metric,
        "score": score,
        "status": status,
        "selection_policy": selection_policy,
        "uniqueness_risk": risk,
        "uniqueness_dir": str(best.get("uniqueness_dir", "")),
        "robustness_dir": str(best.get("robustness_dir", "")),
        "overall_score": _safe_float(best.get("overall_score", 0.0)),
        "unique_score": _safe_float(best.get("unique_score", 0.0)),
        "robust_aux_score": _safe_float(best.get("robust_aux_score", 0.0)),
        "uniq_max_unique_nc": _safe_float(best.get("uniq_max_unique_nc", 1.0), 1.0),
        "uniq_pair_nc_gt_0_7": int(_safe_float(best.get("uniq_pair_nc_gt_0_7", 0.0))),
        "uniq_pair_nc_gt_0_8": int(_safe_float(best.get("uniq_pair_nc_gt_0_8", 0.0))),
        "uniq_pair_nc_gt_0_9": int(_safe_float(best.get("uniq_pair_nc_gt_0_9", 0.0))),
    }


def select_specialists(
    profile_csv: str | Path,
    output_dir: str | Path,
    role_metrics: dict[str, str] | None = None,
    min_scores: dict[str, float] | None = None,
    allow_same_model_multiple_roles: bool = True,
    selection_policy: str = "gated",
) -> dict[str, Any]:
    """Select specialists and save selected_specialists.json/csv.

    selection_policy:
      gated:     default. W_unique handles identity separation; robust roles are
                 selected mainly by direction robustness and uniqueness is only
                 reported as risk.
      strict:    filters robust roles by a loose uniqueness ceiling before
                 choosing, useful for stand-alone subwatermark claims.
      role_only: ignores uniqueness even as a tie-breaker for robust roles.
    """
    if selection_policy not in {"gated", "strict", "role_only"}:
        raise SpecialistSelectorError("selection_policy must be one of: gated, strict, role_only")
    profile_path = Path(profile_csv)
    if not profile_path.is_file():
        raise FileNotFoundError(str(profile_path))
    df = pd.read_csv(profile_path)
    if "model" not in df.columns:
        raise SpecialistSelectorError("profile CSV must contain a 'model' column")
    if "overall_score" not in df.columns:
        df["overall_score"] = 0.0
    if "unique_score" not in df.columns:
        df["unique_score"] = 0.0
    if "robust_aux_score" not in df.columns:
        df["robust_aux_score"] = 0.0
    metrics = dict(DEFAULT_ROLE_METRICS)
    if role_metrics:
        metrics.update(role_metrics)
    mins = dict(DEFAULT_MIN_SCORE)
    if min_scores:
        mins.update(min_scores)

    selections: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    available_df = df.copy()
    for role, metric in metrics.items():
        chosen = _select_one(
            available_df,
            role=role,
            metric=metric,
            min_score=float(mins.get(role, 0.0)),
            selection_policy=selection_policy,
        )
        selections[role] = chosen
        selected_rows.append({"role": role, **chosen})
        if chosen.get("model") and not allow_same_model_multiple_roles:
            available_df = available_df[available_df["model"].astype(str) != str(chosen["model"])]
            if available_df.empty:
                available_df = df.copy()

    out_dir = ensure_dir(output_dir)
    json_path = out_dir / "selected_specialists.json"
    csv_path = out_dir / "selected_specialists.csv"
    payload = {
        "profile_csv": str(profile_path),
        "allow_same_model_multiple_roles": bool(allow_same_model_multiple_roles),
        "selection_policy": selection_policy,
        "roles": selections,
        "model_by_role": {role: item.get("model", "") for role, item in selections.items()},
    }
    write_json(json_path, payload)
    pd.DataFrame(selected_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {
        "selected_specialists_json": str(json_path),
        "selected_specialists_csv": str(csv_path),
        "num_roles": int(len(selected_rows)),
        "selection_policy": selection_policy,
        "model_by_role": payload["model_by_role"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Select V13 direction specialists from model_capability_profile.csv")
    ap.add_argument("--profile_csv", required=True, help="model_capability_profile.csv from capability_profiler_V13")
    ap.add_argument("--output_dir", default="", help="Output directory; default: profile_csv parent")
    ap.add_argument("--role_config", default="", help="Optional JSON with role_metrics/min_scores")
    ap.add_argument("--allow_same_model_multiple_roles", type=_bool_arg, default=True)
    ap.add_argument("--selection_policy", default="gated", choices=["gated", "strict", "role_only"])
    ns = ap.parse_args()

    try:
        role_metrics, min_scores = _load_role_config(ns.role_config)
        out_dir = Path(ns.output_dir) if ns.output_dir else Path(ns.profile_csv).parent
        summary = select_specialists(
            profile_csv=ns.profile_csv,
            output_dir=out_dir,
            role_metrics=role_metrics,
            min_scores=min_scores,
            allow_same_model_multiple_roles=bool(ns.allow_same_model_multiple_roles),
            selection_policy=str(ns.selection_policy),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] specialist_selector_V13: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
