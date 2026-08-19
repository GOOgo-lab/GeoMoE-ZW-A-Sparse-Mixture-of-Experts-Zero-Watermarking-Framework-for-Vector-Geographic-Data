#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V16.1 paired-descriptor sparse MoE router experiments.

This script extends V16 frozen-expert sparse routing with several ablations:

1. query-only descriptors (V16 baseline);
2. paired descriptors: desc(query), desc(reference), abs-diff and safe ratio;
3. oracle-label routing, attack-role auxiliary routing, and topology-boost routing;
4. fixed top-k and adaptive top-k sparse activation;
5. side-by-side comparison against static gated, oracle upper bound and dense mean.

It still does NOT retrain watermark experts.  It only trains lightweight routers
on top of the existing V15 expert pool and is intended to generate MoE routing
ablation tables for the paper.
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

from rb_afl_system.router.geometry_descriptor import paired_geometry_descriptor
from rb_afl_system.scripts.sparse_moe_router_V16 import (
    BASE_EXPERT_ROLES,
    _attack_summary,
    _bool_arg,
    _make_oracle_labels,
    _predict_probs,
    _safe_float,
    _sigmoid,
    _standardize_train_eval,
    _summary_from_scores,
    _train_multilabel_logreg,
    build_sparse_moe_table,
    canonical_attack_role,
    evaluate_oracle_and_static,
    expert_role_for_canonical,
)
from rb_afl_system.utils import ensure_dir, write_json


def _role_order_from_table(df: pd.DataFrame) -> list[str]:
    return [c.replace("nc__", "") for c in df.columns if c.startswith("nc__")]


def enrich_paired_descriptors(table_csv: str | Path, output_csv: str | Path, max_token_dims: int = 8) -> dict[str, Any]:
    """Add paired query/reference descriptors to a V16 expert table."""
    table_path = Path(table_csv)
    df = pd.read_csv(table_path)
    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        sample_dir = str(row.get("sample_dir", ""))
        desc = paired_geometry_descriptor(sample_dir, max_token_dims=max_token_dims) if sample_dir else {"paired_ref_found": 0.0}
        rows.append({f"pdesc__{k}": v for k, v in desc.items()})
    desc_df = pd.DataFrame(rows).fillna(0.0)
    out = pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "input_csv": str(table_csv),
        "output_csv": str(output_csv),
        "num_rows": int(len(out)),
        "num_paired_descriptor_cols": int(len(desc_df.columns)),
        "paired_ref_found_rate": float(out.get("pdesc__paired_ref_found", pd.Series([0.0] * len(out))).mean()),
    }


def _feature_matrix(df: pd.DataFrame, descriptor_mode: str) -> tuple[np.ndarray, list[str]]:
    mode = str(descriptor_mode).strip().lower()
    query_cols = [c for c in df.columns if c.startswith("desc__")]
    pair_cols = [c for c in df.columns if c.startswith("pdesc__")]
    if mode == "query":
        cols = query_cols
    elif mode == "paired":
        cols = pair_cols
    elif mode == "query_paired":
        cols = query_cols + pair_cols
    else:
        raise ValueError(f"Unknown descriptor_mode={descriptor_mode!r}; use query/paired/query_paired")
    if not cols:
        raise RuntimeError(f"No descriptor columns found for mode={descriptor_mode!r}")
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return x, cols


def _attack_role_labels(df: pd.DataFrame, role_order: list[str]) -> np.ndarray:
    role_to_idx = {r: i for i, r in enumerate(role_order)}
    y = np.zeros((len(df), len(role_order)), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        canonical = canonical_attack_role(row)
        role = expert_role_for_canonical(canonical)
        if role not in role_to_idx:
            role = "fallback_other" if "fallback_other" in role_to_idx else role_order[0]
        y[i, role_to_idx[role]] = 1.0
    return y


def _compose_router_targets(
    df: pd.DataFrame,
    role_order: list[str],
    oracle_margin: float,
    target_mode: str,
    attack_aux_weight: float,
    topology_boost: bool,
) -> np.ndarray:
    mode = str(target_mode).strip().lower()
    y_oracle = _make_oracle_labels(df, role_order, oracle_margin)
    y_attack = _attack_role_labels(df, role_order)
    if mode == "oracle":
        y = y_oracle
    elif mode == "attack":
        y = y_attack
    elif mode in {"oracle_attack", "oracle_plus_attack"}:
        # Continuous soft-label blend.  It keeps oracle information while adding
        # attack-role semantics as an auxiliary supervisory signal.
        w = float(np.clip(attack_aux_weight, 0.0, 1.0))
        y = np.clip((1.0 - w) * y_oracle + w * y_attack, 0.0, 1.0)
        # Preserve oracle positive labels as full positives; otherwise a strong
        # auxiliary weight could accidentally weaken the true oracle expert.
        y = np.maximum(y, y_oracle)
    elif mode == "oracle_attack_union":
        y = np.maximum(y_oracle, y_attack)
    else:
        raise ValueError(f"Unknown target_mode={target_mode!r}")

    if topology_boost and "topology" in role_order:
        topo_idx = role_order.index("topology")
        for i, (_, row) in enumerate(df.iterrows()):
            if canonical_attack_role(row) == "topology":
                y[i, topo_idx] = max(float(y[i, topo_idx]), 1.0)
    empty = y.sum(axis=1) <= 0
    if empty.any():
        y[empty, 0] = 1.0
    return y.astype(np.float32)


def _entropy_normalized(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float32)
    denom = p.sum(axis=1, keepdims=True)
    q = np.where(denom > 1e-8, p / denom, np.full_like(p, 1.0 / max(1, p.shape[1])))
    ent = -(q * np.log(q + 1e-8)).sum(axis=1)
    return ent / max(math.log(max(2, p.shape[1])), 1e-8)


def sparse_scores_from_probs(
    probs: np.ndarray,
    nc: np.ndarray,
    role_order: list[str],
    top_k: int,
    adaptive: bool = False,
    adaptive_max_k: int = 3,
    confidence_gap: float = 0.15,
    entropy_threshold: float = 0.78,
) -> tuple[np.ndarray, list[str], list[int], np.ndarray]:
    """Compute sparse weighted NC from router probabilities.

    If adaptive=True, uncertain samples use ``adaptive_max_k`` while confident
    samples use ``top_k``.  Uncertainty is detected by small top1-top2 gap or
    high normalized entropy.
    """
    p = np.asarray(probs, dtype=np.float32)
    n_roles = p.shape[1]
    base_k = max(1, min(int(top_k), n_roles))
    max_k = max(base_k, min(int(adaptive_max_k), n_roles))
    order = np.argsort(-p, axis=1)
    ent = _entropy_normalized(p)
    sorted_probs = np.take_along_axis(p, order, axis=1)
    gap = sorted_probs[:, 0] - sorted_probs[:, 1] if n_roles >= 2 else np.ones(p.shape[0], dtype=np.float32)

    scores: list[float] = []
    role_strings: list[str] = []
    used_ks: list[int] = []
    weights_full = np.zeros_like(p, dtype=np.float32)
    for i in range(p.shape[0]):
        k = max_k if adaptive and (gap[i] < confidence_gap or ent[i] > entropy_threshold) else base_k
        idx = order[i, :k]
        selected_prob = p[i, idx]
        denom = float(selected_prob.sum())
        if denom <= 1e-8:
            weights = np.full(k, 1.0 / float(k), dtype=np.float32)
        else:
            weights = selected_prob / denom
        selected_nc = nc[i, idx]
        scores.append(float((weights * selected_nc).sum()))
        role_strings.append(",".join(role_order[int(j)] for j in idx))
        used_ks.append(int(k))
        weights_full[i, idx] = weights.astype(np.float32)
    return np.asarray(scores, dtype=np.float32), role_strings, used_ks, weights_full


def _coverage_at_k(pred_roles: pd.Series, oracle_roles: pd.Series) -> float:
    return float(
        pd.concat([pred_roles.astype(str), oracle_roles.astype(str)], axis=1)
        .apply(lambda r: r.iloc[1] in [x for x in r.iloc[0].split(",") if x], axis=1)
        .mean()
    )


def train_router_variant_cv(
    df: pd.DataFrame,
    variant: dict[str, Any],
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    nc_cols = [c for c in df.columns if c.startswith("nc__")]
    role_order = [c.replace("nc__", "") for c in nc_cols]
    x, feature_cols = _feature_matrix(df, str(variant.get("descriptor_mode", "query")))
    y = _compose_router_targets(
        df=df,
        role_order=role_order,
        oracle_margin=float(variant.get("oracle_margin", 0.02)),
        target_mode=str(variant.get("target_mode", "oracle")),
        attack_aux_weight=float(variant.get("attack_aux_weight", 0.25)),
        topology_boost=bool(variant.get("topology_boost", False)),
    )
    nc = df[nc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    identities = df["identity"].astype(str).fillna("__unknown__").to_numpy()
    uniq_ids = sorted(set(identities.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_ids)
    cv_folds = max(2, min(int(variant.get("cv_folds", 5)), len(uniq_ids)))
    folds = np.array_split(np.array(uniq_ids, dtype=object), cv_folds)

    pred_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    top_k = int(variant.get("top_k", 2))
    adaptive = bool(variant.get("adaptive", False))
    adaptive_max_k = int(variant.get("adaptive_max_k", max(3, top_k)))
    for fold_idx, eval_ids_arr in enumerate(folds):
        eval_ids = set(str(v) for v in eval_ids_arr.tolist())
        eval_mask = np.array([v in eval_ids for v in identities], dtype=bool)
        train_mask = ~eval_mask
        if train_mask.sum() <= 0 or eval_mask.sum() <= 0:
            continue
        x_train, x_eval, scaler = _standardize_train_eval(x[train_mask], x[eval_mask])
        w, losses = _train_multilabel_logreg(
            x_train,
            y[train_mask],
            epochs=int(variant.get("epochs", 1600)),
            lr=float(variant.get("lr", 0.04)),
            l2=float(variant.get("l2", 1e-4)),
            seed=seed + 1009 * fold_idx,
        )
        probs = _predict_probs(x_eval, w)
        scores, roles, used_ks, weights_full = sparse_scores_from_probs(
            probs,
            nc[eval_mask],
            role_order,
            top_k=top_k,
            adaptive=adaptive,
            adaptive_max_k=adaptive_max_k,
            confidence_gap=float(variant.get("confidence_gap", 0.15)),
            entropy_threshold=float(variant.get("entropy_threshold", 0.78)),
        )
        eval_df = df.loc[eval_mask].copy().reset_index(drop=True)
        eval_df["router_variant"] = str(variant["name"])
        eval_df["router_fold"] = int(fold_idx)
        score_col = f"router_{variant['name']}_score_nc"
        eval_df[score_col] = scores
        eval_df["router_score_nc"] = scores
        eval_df["router_roles"] = roles
        eval_df["router_used_k"] = used_ks
        eval_df["router_top1_role"] = [role_order[int(i)] for i in probs.argmax(axis=1)]
        eval_df["router_top1_prob"] = probs.max(axis=1)
        eval_df["router_entropy_norm"] = _entropy_normalized(probs)
        eval_df["oracle_top1_role_cv"] = [role_order[int(i)] for i in nc[eval_mask].argmax(axis=1)]
        eval_df["oracle_top1_nc_cv"] = nc[eval_mask].max(axis=1)
        for j, role in enumerate(role_order):
            eval_df[f"router_prob__{role}"] = probs[:, j]
            eval_df[f"router_weight__{role}"] = weights_full[:, j]
        pred_rows.append(eval_df)
        fold_rows.append(
            {
                "variant": str(variant["name"]),
                "fold": int(fold_idx),
                "train_rows": int(train_mask.sum()),
                "eval_rows": int(eval_mask.sum()),
                "eval_identities": sorted(eval_ids),
                "final_loss": float(losses[-1] if losses else 0.0),
                **_summary_from_scores(scores, "router"),
            }
        )
        write_json(
            out_dir / f"router_{variant['name']}_fold_{fold_idx}_model_v16_1.json",
            {
                "variant": variant,
                "weights": w.tolist(),
                "scaler": scaler,
                "feature_cols": feature_cols,
                "roles": role_order,
                "losses": losses,
            },
        )

    if not pred_rows:
        raise RuntimeError(f"No predictions for router variant {variant['name']}")
    pred = pd.concat(pred_rows, ignore_index=True)
    static_scores = pd.to_numeric(pred.get("static_gated_nc", pd.Series([0.0] * len(pred))), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    router_scores = pd.to_numeric(pred["router_score_nc"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    oracle_scores = pd.to_numeric(pred["oracle_top1_nc_cv"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    hit_top1 = float((pred["router_top1_role"].astype(str) == pred["oracle_top1_role_cv"].astype(str)).mean())
    coverage = _coverage_at_k(pred["router_roles"], pred["oracle_top1_role_cv"])
    pred_csv = out_dir / f"predictions_{variant['name']}_v16_1.csv"
    pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out_dir / f"fold_summary_{variant['name']}_v16_1.csv", index=False, encoding="utf-8-sig")
    _attack_summary(pred, "router_score_nc").to_csv(out_dir / f"attack_summary_{variant['name']}_v16_1.csv", index=False, encoding="utf-8-sig")
    used_k_counts = pred["router_used_k"].value_counts().sort_index().to_dict()
    summary = {
        "variant": str(variant["name"]),
        "descriptor_mode": str(variant.get("descriptor_mode", "query")),
        "target_mode": str(variant.get("target_mode", "oracle")),
        "top_k": int(top_k),
        "adaptive": bool(adaptive),
        "adaptive_max_k": int(adaptive_max_k),
        "oracle_margin": float(variant.get("oracle_margin", 0.02)),
        "attack_aux_weight": float(variant.get("attack_aux_weight", 0.0)),
        "topology_boost": bool(variant.get("topology_boost", False)),
        "cv_folds": int(len(fold_rows)),
        "num_rows": int(len(pred)),
        "router_top1_hit_rate_vs_oracle": hit_top1,
        "router_coverage_vs_oracle": coverage,
        "router_used_k_counts": json.dumps({str(k): int(v) for k, v in used_k_counts.items()}, ensure_ascii=False),
        **_summary_from_scores(static_scores, "static_gated_on_cv_rows"),
        **_summary_from_scores(router_scores, "router"),
        **_summary_from_scores(oracle_scores, "oracle_top1_on_cv_rows"),
        "router_gain_min_nc_vs_static": float(router_scores.min() - static_scores.min()),
        "router_gain_mean_nc_vs_static": float(router_scores.mean() - static_scores.mean()),
        "oracle_gain_min_nc_vs_static": float(oracle_scores.min() - static_scores.min()),
        "predictions_csv": str(pred_csv),
    }
    return summary


def _default_variants(cv_folds: int, epochs: int, lr: float, l2: float, oracle_margin: float) -> list[dict[str, Any]]:
    base = {
        "cv_folds": int(cv_folds),
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "oracle_margin": float(oracle_margin),
    }
    variants = [
        {**base, "name": "query_oracle_top2", "descriptor_mode": "query", "target_mode": "oracle", "top_k": 2},
        {**base, "name": "paired_oracle_top2", "descriptor_mode": "paired", "target_mode": "oracle", "top_k": 2},
        {**base, "name": "paired_oracle_attack_top2", "descriptor_mode": "paired", "target_mode": "oracle_attack", "attack_aux_weight": 0.30, "top_k": 2},
        {**base, "name": "paired_oracle_attack_top3", "descriptor_mode": "paired", "target_mode": "oracle_attack", "attack_aux_weight": 0.30, "top_k": 3},
        {
            **base,
            "name": "paired_oracle_attack_adaptive_top2_3",
            "descriptor_mode": "paired",
            "target_mode": "oracle_attack",
            "attack_aux_weight": 0.30,
            "top_k": 2,
            "adaptive": True,
            "adaptive_max_k": 3,
            "confidence_gap": 0.16,
            "entropy_threshold": 0.76,
        },
        {
            **base,
            "name": "paired_attack_union_top2",
            "descriptor_mode": "paired",
            "target_mode": "oracle_attack_union",
            "top_k": 2,
        },
        {
            **base,
            "name": "paired_topology_boost_adaptive_top2_3",
            "descriptor_mode": "paired",
            "target_mode": "oracle_attack",
            "attack_aux_weight": 0.25,
            "topology_boost": True,
            "top_k": 2,
            "adaptive": True,
            "adaptive_max_k": 3,
            "confidence_gap": 0.18,
            "entropy_threshold": 0.78,
        },
        {**base, "name": "query_attack_role_top2", "descriptor_mode": "query", "target_mode": "attack", "top_k": 2},
        {**base, "name": "paired_attack_role_top2", "descriptor_mode": "paired", "target_mode": "attack", "top_k": 2},
    ]
    return variants


def run_v16_1(
    profile_csv: str | Path,
    selected_json: str | Path,
    output_dir: str | Path,
    expert_roles: list[str],
    use_descriptors: bool,
    top_k: int,
    oracle_margin: float,
    cv_folds: int,
    router_epochs: int,
    router_lr: float,
    router_l2: float,
    seed: int,
    variants_json: str | Path | None = None,
    build_only: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    build = build_sparse_moe_table(
        profile_csv=profile_csv,
        selected_json=selected_json,
        output_dir=out_dir,
        expert_roles=expert_roles,
        use_descriptors=use_descriptors,
    )
    paired = enrich_paired_descriptors(
        table_csv=build["expert_table_csv"],
        output_csv=out_dir / "sparse_moe_expert_table_paired_v16_1.csv",
    )
    oracle = evaluate_oracle_and_static(paired["output_csv"], out_dir / "oracle_eval", top_k=int(top_k))
    result: dict[str, Any] = {"build": build, "paired_descriptors": paired, "oracle": oracle}
    if build_only:
        return result

    df = pd.read_csv(paired["output_csv"])
    if variants_json is not None:
        variants = json.loads(Path(variants_json).read_text(encoding="utf-8"))
    else:
        variants = _default_variants(
            cv_folds=cv_folds,
            epochs=router_epochs,
            lr=router_lr,
            l2=router_l2,
            oracle_margin=oracle_margin,
        )
    router_dir = ensure_dir(out_dir / "router_variants")
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[ROUTER-VARIANT] {variant.get('name')}", flush=True)
        summary = train_router_variant_cv(df=df, variant=variant, output_dir=router_dir / str(variant["name"]), seed=seed)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    cols_first = [
        "variant",
        "descriptor_mode",
        "target_mode",
        "top_k",
        "adaptive",
        "router_top1_hit_rate_vs_oracle",
        "router_coverage_vs_oracle",
        "static_gated_on_cv_rows_mean_nc",
        "static_gated_on_cv_rows_min_nc",
        "router_mean_nc",
        "router_min_nc",
        "router_nc_lt_0_9",
        "router_nc_lt_0_8",
        "router_gain_mean_nc_vs_static",
        "router_gain_min_nc_vs_static",
        "oracle_top1_on_cv_rows_mean_nc",
        "oracle_top1_on_cv_rows_min_nc",
        "oracle_gain_min_nc_vs_static",
        "router_used_k_counts",
    ]
    keep_cols = [c for c in cols_first if c in summary_df.columns] + [c for c in summary_df.columns if c not in cols_first]
    summary_df = summary_df[keep_cols]
    summary_csv = out_dir / "router_variant_summary_v16_1.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_json(out_dir / "router_variant_summary_v16_1.json", {"variants": summaries})
    result["router_variants"] = {"summary_csv": str(summary_csv), "num_variants": int(len(summaries))}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="V16.1 paired-descriptor sparse MoE router ablation runner")
    ap.add_argument("--profile_csv", required=True)
    ap.add_argument("--selected_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--expert_roles", default=",".join(BASE_EXPERT_ROLES))
    ap.add_argument("--use_descriptors", type=_bool_arg, default=True)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--oracle_margin", type=float, default=0.02)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--router_epochs", type=int, default=1800)
    ap.add_argument("--router_lr", type=float, default=0.04)
    ap.add_argument("--router_l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260318)
    ap.add_argument("--variants_json", default=None)
    ap.add_argument("--build_only", type=_bool_arg, default=False)
    ns = ap.parse_args()
    try:
        roles = [x.strip() for x in str(ns.expert_roles).split(",") if x.strip()]
        result = run_v16_1(
            profile_csv=ns.profile_csv,
            selected_json=ns.selected_json,
            output_dir=ns.output_dir,
            expert_roles=roles,
            use_descriptors=bool(ns.use_descriptors),
            top_k=int(ns.top_k),
            oracle_margin=float(ns.oracle_margin),
            cv_folds=int(ns.cv_folds),
            router_epochs=int(ns.router_epochs),
            router_lr=float(ns.router_lr),
            router_l2=float(ns.router_l2),
            seed=int(ns.seed),
            variants_json=ns.variants_json,
            build_only=bool(ns.build_only),
        )
        write_json(Path(ns.output_dir) / "sparse_moe_router_V16_1_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] sparse_moe_router_V16_1: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
