#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V16 sparse mixture-of-specialists router for zero-watermarking.

This is the first practical V16 layer: experts are frozen, and a lightweight
router learns to activate only a top-k subset of robust specialists.  It supports
three analyses that are important for the paper:

1. static gated routing from V15 (attack-label based upper baseline);
2. oracle routing upper bound from per-sample expert NC values;
3. descriptor-based sparse router with identity-group cross validation.

The script does not retrain G/D experts.  It consumes existing V15 suite outputs
(`model_capability_profile.csv` + `selected_specialists.json`) and produces
router-ready tables and paper metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rb_afl_system.router.geometry_descriptor import geometry_descriptor
from rb_afl_system.scripts.specialist_ensemble_evaluator_V13 import (
    _load_model_robust_rows,
    _load_profile,
    _load_selected,
    _standardize_rob_rows,
)
from rb_afl_system.scripts.specialist_ensemble_evaluator_V15 import _deduplicate_selected_rows
from rb_afl_system.utils import ensure_dir, write_json

BASE_EXPERT_ROLES = ["local_attack", "rotate", "scale", "topology", "boundary", "fallback_other"]
CANONICAL_ROLE_PRIORITY = [
    "topology",
    "boundary",
    "rotate",
    "scale",
    "jitter",
    "quantize",
    "simplify",
    "fallback_other",
]
LOCAL_CANONICAL_ROLES = {"jitter", "quantize", "simplify"}


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sample_key_cols(df: pd.DataFrame) -> list[str]:
    for cols in [
        ["identity", "sample", "sample_dir"],
        ["identity", "attack_type", "attack_engine", "attack_value", "sample_dir"],
        ["identity", "attack_type", "attack_engine", "attack_value"],
        ["identity", "attack_type", "attack_value"],
    ]:
        if all(c in df.columns for c in cols):
            return cols
    return [c for c in ["identity", "attack_type", "attack_value"] if c in df.columns]


def _row_key(df: pd.DataFrame) -> pd.Series:
    cols = _sample_key_cols(df)
    if not cols:
        return pd.Series([str(i) for i in range(len(df))], index=df.index)
    key = pd.Series("", index=df.index, dtype="object")
    for col in cols:
        key = key + "|" + df[col].astype(str)
    return key.str.strip("|")


def _attack_text(row: pd.Series) -> str:
    parts = []
    for col in ["attack_type", "attack_engine", "attack_label", "attack_name", "attack_value"]:
        if col in row.index:
            parts.append(str(row.get(col, "")).lower())
    return " ".join(parts)


def canonical_attack_role(row: pd.Series) -> str:
    text = _attack_text(row)
    if any(k in text for k in ["topology", "component", "delete", "clean", "repair", "multipart"]):
        return "topology"
    if any(k in text for k in ["boundary", "smooth", "vertex", "curve", "curvature", "edge"]):
        return "boundary"
    if any(k in text for k in ["rotate", "rotation", "rot"]):
        return "rotate"
    if any(k in text for k in ["scale", "resize", "uniform_scale", "nonuniform"]):
        return "scale"
    if any(k in text for k in ["jitter", "noise", "perturb"]):
        return "jitter"
    if any(k in text for k in ["quantize", "quantization", "round", "precision"]):
        return "quantize"
    if any(k in text for k in ["simplify", "mapshaper", "douglas"]):
        return "simplify"
    return "fallback_other"


def expert_role_for_canonical(canonical_role: str) -> str:
    if canonical_role in LOCAL_CANONICAL_ROLES:
        return "local_attack"
    return canonical_role


def _selected_expert_map(model_by_role: dict[str, str], expert_roles: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for role in expert_roles:
        if role == "fallback_other":
            model = model_by_role.get("robust_aux", "") or model_by_role.get("overall", "")
        elif role == "local_attack":
            model = model_by_role.get("local_attack", "") or model_by_role.get("jitter", "") or model_by_role.get("simplify", "")
        else:
            model = model_by_role.get(role, "")
        if model:
            out[role] = model
    return out


def _metadata_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in [
        "identity",
        "attack_type",
        "attack_engine",
        "attack_value",
        "attack_name",
        "attack_value_original",
        "attack_value_mode",
        "sample_dir",
        "attack_label",
    ]:
        if col in df.columns:
            cols.append(col)
    return cols


def build_sparse_moe_table(
    profile_csv: str | Path,
    selected_json: str | Path,
    output_dir: str | Path,
    expert_roles: list[str] | None = None,
    use_descriptors: bool = True,
) -> dict[str, Any]:
    profile = _load_profile(profile_csv)
    model_by_role = _load_selected(selected_json)
    roles = expert_roles or BASE_EXPERT_ROLES
    expert_map = _selected_expert_map(model_by_role, roles)
    if not expert_map:
        raise RuntimeError("No robust expert roles resolved from selected specialists")

    out_dir = ensure_dir(output_dir)
    merged: pd.DataFrame | None = None
    model_columns: list[dict[str, str]] = []

    for role, model in expert_map.items():
        rows = _standardize_rob_rows(_load_model_robust_rows(profile, model))
        if rows.empty:
            print(f"[WARN] empty robustness rows for role={role}, model={model}", flush=True)
            continue
        rows = rows.copy()
        rows["sample_key"] = _row_key(rows)
        meta_cols = _metadata_columns(rows)
        metric_cols = ["sample_key", "feature_nc", "feature_ber", "watermark_nc", "watermark_ber"]
        keep_cols = [c for c in meta_cols if c in rows.columns] + metric_cols
        slim = rows[keep_cols].drop_duplicates(subset=["sample_key"]).copy()
        slim = slim.rename(
            columns={
                "feature_nc": f"nc__{role}",
                "feature_ber": f"ber__{role}",
                "watermark_nc": f"wm_nc__{role}",
                "watermark_ber": f"wm_ber__{role}",
            }
        )
        if merged is None:
            merged = slim
        else:
            value_cols = ["sample_key", f"nc__{role}", f"ber__{role}", f"wm_nc__{role}", f"wm_ber__{role}"]
            merged = merged.merge(slim[value_cols], on="sample_key", how="outer")
        model_columns.append({"role": role, "model": model})

    if merged is None or merged.empty:
        raise RuntimeError("Failed to build sparse MoE table; no expert rows found")

    # Fill missing metadata from sample key only if necessary.
    for col in ["identity", "attack_type", "attack_engine", "attack_value", "sample_dir"]:
        if col not in merged.columns:
            merged[col] = ""
    merged["canonical_attack_role"] = merged.apply(canonical_attack_role, axis=1)
    merged["static_expert_role"] = merged["canonical_attack_role"].map(expert_role_for_canonical)
    merged.loc[~merged["static_expert_role"].isin(expert_map.keys()), "static_expert_role"] = "fallback_other"
    if "fallback_other" not in expert_map:
        merged.loc[merged["static_expert_role"] == "fallback_other", "static_expert_role"] = "local_attack"

    nc_cols = [f"nc__{role}" for role in expert_map.keys() if f"nc__{role}" in merged.columns]
    role_order = [col.replace("nc__", "") for col in nc_cols]
    nc_matrix = merged[nc_cols].apply(pd.to_numeric, errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
    best_idx = nc_matrix.argmax(axis=1)
    best_nc = nc_matrix[np.arange(len(merged)), best_idx]
    merged["oracle_top1_role"] = [role_order[int(i)] for i in best_idx]
    merged["oracle_top1_nc"] = best_nc
    merged["oracle_margin_multi_roles"] = ""
    for i in range(len(merged)):
        roles_i = [role_order[j] for j, v in enumerate(nc_matrix[i]) if v >= best_nc[i] - 0.02]
        merged.at[i, "oracle_margin_multi_roles"] = ",".join(roles_i)

    static_nc = []
    for _, row in merged.iterrows():
        role = str(row.get("static_expert_role", ""))
        static_nc.append(_safe_float(row.get(f"nc__{role}", np.nan), 0.0))
    merged["static_gated_nc"] = static_nc

    if use_descriptors:
        desc_rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            sample_dir = str(row.get("sample_dir", ""))
            desc = geometry_descriptor(sample_dir) if sample_dir else {}
            desc_rows.append({f"desc__{k}": v for k, v in desc.items()})
        desc_df = pd.DataFrame(desc_rows).fillna(0.0)
        merged = pd.concat([merged.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

    merged = merged.sort_values(["identity", "attack_type", "attack_value", "sample_key"]).reset_index(drop=True)
    table_csv = out_dir / "sparse_moe_expert_table_v16.csv"
    merged.to_csv(table_csv, index=False, encoding="utf-8-sig")
    write_json(
        out_dir / "sparse_moe_expert_map_v16.json",
        {
            "profile_csv": str(profile_csv),
            "selected_json": str(selected_json),
            "expert_map": expert_map,
            "role_order": role_order,
            "num_rows": int(len(merged)),
            "use_descriptors": bool(use_descriptors),
        },
    )
    return {
        "expert_table_csv": str(table_csv),
        "expert_map_json": str(out_dir / "sparse_moe_expert_map_v16.json"),
        "num_rows": int(len(merged)),
        "expert_roles": role_order,
    }


def _summary_from_scores(scores: np.ndarray, prefix: str) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0:
        return {
            f"{prefix}_num_rows": 0,
            f"{prefix}_mean_nc": 0.0,
            f"{prefix}_min_nc": 0.0,
            f"{prefix}_nc_lt_0_9": 0,
            f"{prefix}_nc_lt_0_8": 0,
        }
    return {
        f"{prefix}_num_rows": int(s.size),
        f"{prefix}_mean_nc": float(s.mean()),
        f"{prefix}_min_nc": float(s.min()),
        f"{prefix}_nc_lt_0_9": int((s < 0.9).sum()),
        f"{prefix}_nc_lt_0_8": int((s < 0.8).sum()),
    }


def _attack_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    return (
        work.groupby("attack_type", as_index=False)
        .agg(
            num_rows=(score_col, "size"),
            mean_nc=(score_col, "mean"),
            min_nc=(score_col, "min"),
            nc_lt_0_9=(score_col, lambda s: int((pd.to_numeric(s, errors="coerce") < 0.9).sum())),
            nc_lt_0_8=(score_col, lambda s: int((pd.to_numeric(s, errors="coerce") < 0.8).sum())),
        )
        .sort_values("mean_nc", ascending=False)
    )


def evaluate_oracle_and_static(table_csv: str | Path, output_dir: str | Path, top_k: int = 2) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    df = pd.read_csv(table_csv)
    nc_cols = [c for c in df.columns if c.startswith("nc__")]
    if not nc_cols:
        raise RuntimeError("expert table has no nc__ columns")
    roles = [c.replace("nc__", "") for c in nc_cols]
    nc = df[nc_cols].apply(pd.to_numeric, errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
    oracle_top1 = nc.max(axis=1)
    idx_sorted = np.argsort(-nc, axis=1)
    k = max(1, min(int(top_k), len(roles)))
    topk_idx = idx_sorted[:, :k]
    oracle_topk_mean = np.take_along_axis(nc, topk_idx, axis=1).mean(axis=1)
    dense_mean = nc.mean(axis=1)
    static = pd.to_numeric(df.get("static_gated_nc", pd.Series([0.0] * len(df))), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    out = df.copy()
    out["oracle_top1_nc"] = oracle_top1
    out[f"oracle_top{k}_mean_nc"] = oracle_topk_mean
    out["dense_mean_nc"] = dense_mean
    out["static_gated_nc"] = static
    out.to_csv(out_dir / "sparse_moe_oracle_rows_v16.csv", index=False, encoding="utf-8-sig")

    summary = {
        "table_csv": str(table_csv),
        "roles": roles,
        "top_k": int(k),
        **_summary_from_scores(static, "static_gated"),
        **_summary_from_scores(oracle_top1, "oracle_top1"),
        **_summary_from_scores(oracle_topk_mean, f"oracle_top{k}_mean"),
        **_summary_from_scores(dense_mean, "dense_mean"),
        "oracle_gain_min_nc_vs_static": float(oracle_top1.min() - static.min()),
        "oracle_gain_mean_nc_vs_static": float(oracle_top1.mean() - static.mean()),
    }
    write_json(out_dir / "sparse_moe_oracle_summary_v16.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "sparse_moe_oracle_summary_v16.csv", index=False, encoding="utf-8-sig")

    for score_col in ["static_gated_nc", "oracle_top1_nc", f"oracle_top{k}_mean_nc", "dense_mean_nc"]:
        _attack_summary(out, score_col).to_csv(out_dir / f"attack_summary_{score_col}_v16.csv", index=False, encoding="utf-8-sig")

    return {
        "oracle_rows_csv": str(out_dir / "sparse_moe_oracle_rows_v16.csv"),
        "oracle_summary_csv": str(out_dir / "sparse_moe_oracle_summary_v16.csv"),
        "oracle_gain_min_nc_vs_static": summary["oracle_gain_min_nc_vs_static"],
    }


def _make_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    desc_cols = [c for c in df.columns if c.startswith("desc__")]
    if not desc_cols:
        raise RuntimeError("No descriptor columns found. Rebuild table with --use_descriptors true.")
    x = df[desc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return x, desc_cols


def _make_oracle_labels(df: pd.DataFrame, role_order: list[str], margin: float) -> np.ndarray:
    nc_cols = [f"nc__{role}" for role in role_order]
    nc = df[nc_cols].apply(pd.to_numeric, errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
    best = nc.max(axis=1, keepdims=True)
    y = (nc >= best - float(margin)).astype(np.float32)
    # If a row is all invalid, assign local/fallback first column to avoid empty target.
    empty = y.sum(axis=1) <= 0
    if empty.any():
        y[empty, 0] = 1.0
    return y


def _standardize_train_eval(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_eval - mean) / std, {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}


def _train_multilabel_logreg(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    k = y.shape[1]
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float32)], axis=1)
    w = rng.normal(0.0, 0.01, size=(d + 1, k)).astype(np.float32)
    losses: list[float] = []
    pos = np.maximum(y.sum(axis=0), 1.0)
    neg = np.maximum(n - pos, 1.0)
    pos_weight = np.clip(neg / pos, 1.0, 20.0).reshape(1, -1)
    for _ in range(int(epochs)):
        logits = xb @ w
        p = _sigmoid(logits)
        weights = np.where(y > 0.5, pos_weight, 1.0)
        grad_logits = (p - y) * weights / float(n)
        grad = xb.T @ grad_logits + float(l2) * w
        w -= float(lr) * grad.astype(np.float32)
        if len(losses) < 5 or (_ + 1) % max(1, int(epochs) // 20) == 0:
            eps = 1e-7
            bce = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)) * weights
            losses.append(float(bce.mean() + 0.5 * float(l2) * (w * w).mean()))
    return w, losses


def _predict_probs(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    return _sigmoid(xb @ w).astype(np.float32)


def _sparse_scores_from_probs(
    probs: np.ndarray,
    nc: np.ndarray,
    role_order: list[str],
    top_k: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    k = max(1, min(int(top_k), probs.shape[1]))
    idx = np.argsort(-probs, axis=1)[:, :k]
    selected_prob = np.take_along_axis(probs, idx, axis=1)
    denom = selected_prob.sum(axis=1, keepdims=True)
    weights = np.where(denom > 1e-8, selected_prob / denom, np.full_like(selected_prob, 1.0 / k))
    selected_nc = np.take_along_axis(nc, idx, axis=1)
    scores = (weights * selected_nc).sum(axis=1)
    role_names = [",".join(role_order[int(j)] for j in row) for row in idx]
    return scores.astype(np.float32), role_names, weights.astype(np.float32)


def train_router_cv(
    table_csv: str | Path,
    output_dir: str | Path,
    top_k: int = 2,
    oracle_margin: float = 0.02,
    cv_folds: int = 5,
    epochs: int = 1200,
    lr: float = 0.05,
    l2: float = 1e-4,
    seed: int = 20260318,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    df = pd.read_csv(table_csv)
    nc_cols = [c for c in df.columns if c.startswith("nc__")]
    role_order = [c.replace("nc__", "") for c in nc_cols]
    if len(role_order) < 2:
        raise RuntimeError("Router needs at least two expert nc__ columns")
    x, feature_cols = _make_feature_matrix(df)
    y = _make_oracle_labels(df, role_order, oracle_margin)
    nc = df[nc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    identities = df["identity"].astype(str).fillna("__unknown__").to_numpy()
    uniq_ids = sorted(set(identities.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_ids)
    folds = np.array_split(np.array(uniq_ids, dtype=object), max(2, min(int(cv_folds), len(uniq_ids))))

    pred_rows: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_idx, eval_ids_arr in enumerate(folds):
        eval_ids = set(str(v) for v in eval_ids_arr.tolist())
        eval_mask = np.array([v in eval_ids for v in identities], dtype=bool)
        train_mask = ~eval_mask
        if train_mask.sum() <= 0 or eval_mask.sum() <= 0:
            continue
        x_train, x_eval, scaler = _standardize_train_eval(x[train_mask], x[eval_mask])
        w, losses = _train_multilabel_logreg(x_train, y[train_mask], epochs=epochs, lr=lr, l2=l2, seed=seed + fold_idx)
        probs = _predict_probs(x_eval, w)
        sparse_score, selected_roles, weights = _sparse_scores_from_probs(probs, nc[eval_mask], role_order, top_k=top_k)
        eval_df = df.loc[eval_mask].copy().reset_index(drop=True)
        eval_df["router_fold"] = fold_idx
        eval_df[f"router_top{top_k}_score_nc"] = sparse_score
        eval_df[f"router_top{top_k}_roles"] = selected_roles
        eval_df["router_top1_role"] = [role_order[int(i)] for i in probs.argmax(axis=1)]
        eval_df["router_top1_prob"] = probs.max(axis=1)
        eval_df["oracle_top1_role_cv"] = [role_order[int(i)] for i in nc[eval_mask].argmax(axis=1)]
        eval_df["oracle_top1_nc_cv"] = nc[eval_mask].max(axis=1)
        for j, role in enumerate(role_order):
            eval_df[f"router_prob__{role}"] = probs[:, j]
        pred_rows.append(eval_df)
        fold_summary = {
            "fold": fold_idx,
            "train_rows": int(train_mask.sum()),
            "eval_rows": int(eval_mask.sum()),
            "eval_identities": sorted(eval_ids),
            "final_loss": float(losses[-1] if losses else 0.0),
            **_summary_from_scores(sparse_score, f"router_top{top_k}"),
        }
        fold_summaries.append(fold_summary)
        write_json(out_dir / f"router_fold_{fold_idx}_model_v16.json", {"weights": w.tolist(), "scaler": scaler, "feature_cols": feature_cols, "roles": role_order, "losses": losses})

    if not pred_rows:
        raise RuntimeError("No CV predictions generated")
    pred = pd.concat(pred_rows, ignore_index=True)
    pred.to_csv(out_dir / "sparse_moe_router_cv_predictions_v16.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_summaries).to_csv(out_dir / "sparse_moe_router_cv_fold_summary_v16.csv", index=False, encoding="utf-8-sig")

    router_scores = pd.to_numeric(pred[f"router_top{top_k}_score_nc"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    static_scores = pd.to_numeric(pred.get("static_gated_nc", pd.Series([0.0] * len(pred))), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    oracle_scores = pd.to_numeric(pred.get("oracle_top1_nc_cv", pd.Series([0.0] * len(pred))), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    hit_top1 = (pred["router_top1_role"].astype(str) == pred["oracle_top1_role_cv"].astype(str)).mean()
    coverage = pred.apply(lambda r: str(r["oracle_top1_role_cv"]) in str(r[f"router_top{top_k}_roles"]).split(","), axis=1).mean()

    summary = {
        "table_csv": str(table_csv),
        "top_k": int(top_k),
        "oracle_margin": float(oracle_margin),
        "cv_folds": int(len(fold_summaries)),
        "roles": role_order,
        "num_rows": int(len(pred)),
        "router_top1_hit_rate_vs_oracle": float(hit_top1),
        f"router_top{top_k}_coverage_vs_oracle": float(coverage),
        **_summary_from_scores(static_scores, "static_gated_on_cv_rows"),
        **_summary_from_scores(router_scores, f"router_top{top_k}"),
        **_summary_from_scores(oracle_scores, "oracle_top1_on_cv_rows"),
        "router_gain_min_nc_vs_static": float(router_scores.min() - static_scores.min()),
        "router_gain_mean_nc_vs_static": float(router_scores.mean() - static_scores.mean()),
        "oracle_gain_min_nc_vs_static": float(oracle_scores.min() - static_scores.min()),
        "note": "Router is descriptor-based, identity-group CV. It is sparse top-k and does not use attack labels as input.",
    }
    write_json(out_dir / "sparse_moe_router_cv_summary_v16.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "sparse_moe_router_cv_summary_v16.csv", index=False, encoding="utf-8-sig")
    _attack_summary(pred, f"router_top{top_k}_score_nc").to_csv(out_dir / f"attack_summary_router_top{top_k}_v16.csv", index=False, encoding="utf-8-sig")
    return {
        "router_cv_predictions_csv": str(out_dir / "sparse_moe_router_cv_predictions_v16.csv"),
        "router_cv_summary_csv": str(out_dir / "sparse_moe_router_cv_summary_v16.csv"),
        "router_top1_hit_rate_vs_oracle": summary["router_top1_hit_rate_vs_oracle"],
        f"router_top{top_k}_coverage_vs_oracle": summary[f"router_top{top_k}_coverage_vs_oracle"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/evaluate V16 sparse MoE specialist router from V15 suite outputs")
    ap.add_argument("--profile_csv", required=True, help="V15 model_capability_profile.csv")
    ap.add_argument("--selected_json", required=True, help="V15 selected_specialists.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--expert_roles", default=",".join(BASE_EXPERT_ROLES), help="Comma-separated expert roles to include")
    ap.add_argument("--use_descriptors", type=_bool_arg, default=True, help="Read grid/tokens/graph descriptors from sample_dir")
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--oracle_margin", type=float, default=0.02)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--router_epochs", type=int, default=1200)
    ap.add_argument("--router_lr", type=float, default=0.05)
    ap.add_argument("--router_l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260318)
    ap.add_argument("--build_only", type=_bool_arg, default=False)
    ns = ap.parse_args()

    try:
        roles = [x.strip() for x in str(ns.expert_roles).split(",") if x.strip()]
        out_dir = ensure_dir(ns.output_dir)
        build = build_sparse_moe_table(
            profile_csv=ns.profile_csv,
            selected_json=ns.selected_json,
            output_dir=out_dir,
            expert_roles=roles,
            use_descriptors=bool(ns.use_descriptors),
        )
        oracle = evaluate_oracle_and_static(build["expert_table_csv"], out_dir / "oracle_eval", top_k=int(ns.top_k))
        result: dict[str, Any] = {"build": build, "oracle": oracle}
        if not ns.build_only:
            router = train_router_cv(
                table_csv=build["expert_table_csv"],
                output_dir=out_dir / "router_cv",
                top_k=int(ns.top_k),
                oracle_margin=float(ns.oracle_margin),
                cv_folds=int(ns.cv_folds),
                epochs=int(ns.router_epochs),
                lr=float(ns.router_lr),
                l2=float(ns.router_l2),
                seed=int(ns.seed),
            )
            result["router_cv"] = router
        write_json(out_dir / "sparse_moe_router_V16_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] sparse_moe_router_V16: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
