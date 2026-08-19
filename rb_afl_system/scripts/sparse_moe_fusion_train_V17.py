#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V17-A frozen-expert sparse MoE router + trainable score fusion.

This script is the first MoE-training stage for the vector zero-watermarking
pipeline.  It keeps all V15/V16 experts frozen and trains only lightweight
routing/fusion logic on top of the existing expert NC table.

Compared with V16.1, this script adds:

1. train-fold selected score-fusion mode instead of fixed mean fusion;
2. max / confidence / softmax-score sparse fusion variants;
3. protected sparse routing variants that keep rotate/local experts when the
   router is uncertain;
4. side-by-side ablations against static gated, dense mean and oracle routing.

It deliberately does not fine-tune expert backbones or W_unique.  The output is
intended for MoE ablation tables before moving to heavier projection-head or
expert fine-tuning.
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

from rb_afl_system.scripts.sparse_moe_router_V16 import (
    BASE_EXPERT_ROLES,
    _attack_summary,
    _bool_arg,
    _make_oracle_labels,
    _predict_probs,
    _safe_float,
    _standardize_train_eval,
    _summary_from_scores,
    _train_multilabel_logreg,
    build_sparse_moe_table,
    evaluate_oracle_and_static,
)
from rb_afl_system.scripts.sparse_moe_router_V16_1 import (
    _compose_router_targets,
    _coverage_at_k,
    _feature_matrix,
    enrich_paired_descriptors,
)
from rb_afl_system.utils import ensure_dir, write_json


def _role_order_from_table(df: pd.DataFrame) -> list[str]:
    return [c.replace("nc__", "") for c in df.columns if c.startswith("nc__")]


def _entropy_normalized(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float32)
    denom = p.sum(axis=1, keepdims=True)
    q = np.where(denom > 1e-8, p / denom, np.full_like(p, 1.0 / max(1, p.shape[1])))
    ent = -(q * np.log(q + 1e-8)).sum(axis=1)
    return ent / max(math.log(max(2, p.shape[1])), 1e-8)


def _top_indices(
    probs: np.ndarray,
    top_k: int,
    adaptive: bool = False,
    adaptive_max_k: int = 3,
    confidence_gap: float = 0.16,
    entropy_threshold: float = 0.76,
) -> tuple[list[np.ndarray], list[int], np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float32)
    n_roles = p.shape[1]
    base_k = max(1, min(int(top_k), n_roles))
    max_k = max(base_k, min(int(adaptive_max_k), n_roles))
    order = np.argsort(-p, axis=1)
    sorted_probs = np.take_along_axis(p, order, axis=1)
    gap = sorted_probs[:, 0] - sorted_probs[:, 1] if n_roles >= 2 else np.ones(p.shape[0], dtype=np.float32)
    ent = _entropy_normalized(p)

    indices: list[np.ndarray] = []
    used_ks: list[int] = []
    for i in range(p.shape[0]):
        k = max_k if adaptive and (gap[i] < confidence_gap or ent[i] > entropy_threshold) else base_k
        indices.append(order[i, :k].astype(np.int64))
        used_ks.append(int(k))
    return indices, used_ks, gap, ent


def _apply_protected_roles(
    selected: list[np.ndarray],
    probs: np.ndarray,
    role_order: list[str],
    protected_roles: list[str],
    protect_when_uncertain: bool,
    gap: np.ndarray,
    ent: np.ndarray,
    confidence_gap: float,
    entropy_threshold: float,
    max_k: int,
) -> list[np.ndarray]:
    role_to_idx = {r: i for i, r in enumerate(role_order)}
    protected_idx = [role_to_idx[r] for r in protected_roles if r in role_to_idx]
    if not protected_idx:
        return selected
    out: list[np.ndarray] = []
    order = np.argsort(-probs, axis=1)
    for i, idx in enumerate(selected):
        idx_list = [int(v) for v in idx.tolist()]
        should_protect = True
        if protect_when_uncertain:
            should_protect = bool(gap[i] < confidence_gap or ent[i] > entropy_threshold)
        if should_protect:
            for pidx in protected_idx:
                if pidx not in idx_list:
                    idx_list.append(pidx)
        # Re-rank union by router probability and cut to max_k.
        idx_list = sorted(set(idx_list), key=lambda j: float(probs[i, j]), reverse=True)
        if len(idx_list) < min(max_k, probs.shape[1]):
            for cand in order[i].tolist():
                cand_i = int(cand)
                if cand_i not in idx_list:
                    idx_list.append(cand_i)
                if len(idx_list) >= min(max_k, probs.shape[1]):
                    break
        out.append(np.asarray(idx_list[: min(max_k, probs.shape[1])], dtype=np.int64))
    return out


def _fuse_scores_for_indices(
    probs: np.ndarray,
    nc: np.ndarray,
    selected: list[np.ndarray],
    role_order: list[str],
    fusion_mode: str,
    alpha: float = 8.0,
    router_power: float = 1.0,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    mode = str(fusion_mode).strip().lower()
    scores: list[float] = []
    role_strings: list[str] = []
    used_ks: list[int] = []
    weight_strings: list[str] = []

    for i, idx in enumerate(selected):
        idx = np.asarray(idx, dtype=np.int64)
        selected_prob = np.clip(probs[i, idx].astype(np.float32), 0.0, None)
        selected_nc = nc[i, idx].astype(np.float32)
        if len(idx) == 0:
            scores.append(0.0)
            role_strings.append("")
            used_ks.append(0)
            weight_strings.append("")
            continue

        if mode in {"mean", "router_mean", "weighted_mean"}:
            weights = selected_prob ** float(router_power)
            denom = float(weights.sum())
            if denom <= 1e-8:
                weights = np.full(len(idx), 1.0 / float(len(idx)), dtype=np.float32)
            else:
                weights = weights / denom
            score = float((weights * selected_nc).sum())
        elif mode in {"max", "topk_max", "score_max"}:
            best = int(np.argmax(selected_nc))
            weights = np.zeros(len(idx), dtype=np.float32)
            weights[best] = 1.0
            score = float(selected_nc[best])
        elif mode in {"softmax_score", "confidence", "confidence_weighted"}:
            centered = selected_nc - float(selected_nc.max())
            router_term = np.maximum(selected_prob, 1e-8) ** float(router_power)
            score_term = np.exp(np.clip(float(alpha) * centered, -50.0, 50.0))
            weights = router_term * score_term
            denom = float(weights.sum())
            if denom <= 1e-8:
                weights = np.full(len(idx), 1.0 / float(len(idx)), dtype=np.float32)
            else:
                weights = weights / denom
            score = float((weights * selected_nc).sum())
        else:
            raise ValueError(f"Unknown fusion_mode={fusion_mode!r}")

        scores.append(score)
        role_strings.append(",".join(role_order[int(j)] for j in idx))
        used_ks.append(int(len(idx)))
        weight_strings.append(",".join(f"{role_order[int(j)]}:{float(w):.4f}" for j, w in zip(idx, weights)))
    return np.asarray(scores, dtype=np.float32), role_strings, used_ks, weight_strings


def _objective(scores: np.ndarray, metric: str) -> float:
    s = np.asarray(scores, dtype=np.float32)
    mode = str(metric).strip().lower()
    if mode == "mean":
        return float(s.mean())
    if mode == "min":
        return float(s.min())
    if mode == "mean_plus_min":
        return float(s.mean() + 0.50 * s.min())
    if mode == "paper":
        return float(s.mean() + 0.70 * s.min() - 0.02 * float((s < 0.9).sum()) - 0.05 * float((s < 0.8).sum()))
    raise ValueError(f"Unknown fusion_select_metric={metric!r}")


def _choose_fusion_alpha(
    probs_train: np.ndarray,
    nc_train: np.ndarray,
    selected_train: list[np.ndarray],
    role_order: list[str],
    fusion_mode: str,
    alpha_grid: list[float],
    router_power_grid: list[float],
    metric: str,
) -> tuple[float, float, float]:
    mode = str(fusion_mode).strip().lower()
    if mode not in {"softmax_score", "confidence", "confidence_weighted", "weighted_mean", "mean", "router_mean"}:
        return 0.0, 1.0, 0.0
    best_obj = -1e18
    best_alpha = float(alpha_grid[0]) if alpha_grid else 8.0
    best_power = float(router_power_grid[0]) if router_power_grid else 1.0
    alpha_values = alpha_grid if mode in {"softmax_score", "confidence", "confidence_weighted"} else [0.0]
    for alpha in alpha_values:
        for power in router_power_grid:
            scores, _, _, _ = _fuse_scores_for_indices(
                probs=probs_train,
                nc=nc_train,
                selected=selected_train,
                role_order=role_order,
                fusion_mode=fusion_mode,
                alpha=float(alpha),
                router_power=float(power),
            )
            obj = _objective(scores, metric)
            if obj > best_obj:
                best_obj = obj
                best_alpha = float(alpha)
                best_power = float(power)
    return best_alpha, best_power, float(best_obj)


def _attack_role_from_row(row: pd.Series) -> str:
    text = " ".join(str(row.get(c, "")).lower() for c in ["attack_type", "attack_engine", "attack_value"])
    if "topology" in text or "component" in text or "delete" in text or "clean" in text:
        return "topology"
    if "boundary" in text or "smooth" in text:
        return "boundary"
    if "rotate" in text:
        return "rotate"
    if "scale" in text:
        return "scale"
    if "jitter" in text:
        return "local_attack"
    if "quantize" in text:
        return "local_attack"
    if "simplify" in text or "mapshaper" in text:
        return "local_attack"
    return "fallback_other"


def _candidate_variant_defaults(cv_folds: int, epochs: int, lr: float, l2: float, oracle_margin: float) -> list[dict[str, Any]]:
    base = {
        "descriptor_mode": "paired",
        "target_mode": "oracle",
        "oracle_margin": float(oracle_margin),
        "cv_folds": int(cv_folds),
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "alpha_grid": [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0],
        "router_power_grid": [0.5, 1.0, 1.5, 2.0],
        "fusion_select_metric": "paper",
    }
    return [
        {**base, "name": "v17_paired_top2_mean", "top_k": 2, "fusion_mode": "weighted_mean"},
        {**base, "name": "v17_paired_top2_max", "top_k": 2, "fusion_mode": "topk_max"},
        {**base, "name": "v17_paired_top2_confidence", "top_k": 2, "fusion_mode": "softmax_score"},
        {**base, "name": "v17_paired_top3_max", "top_k": 3, "fusion_mode": "topk_max"},
        {**base, "name": "v17_paired_top3_confidence", "top_k": 3, "fusion_mode": "softmax_score"},
        {
            **base,
            "name": "v17_paired_adaptive_top2_3_confidence",
            "top_k": 2,
            "adaptive": True,
            "adaptive_max_k": 3,
            "fusion_mode": "softmax_score",
            "confidence_gap": 0.16,
            "entropy_threshold": 0.76,
        },
        {
            **base,
            "name": "v17_paired_protect_rotate_local_top2_confidence",
            "top_k": 2,
            "adaptive": False,
            "max_k_after_protect": 3,
            "protected_roles": ["rotate", "local_attack"],
            "protect_when_uncertain": True,
            "fusion_mode": "softmax_score",
            "confidence_gap": 0.18,
            "entropy_threshold": 0.75,
        },
        {
            **base,
            "name": "v17_paired_oracle_attack_top2_confidence",
            "top_k": 2,
            "target_mode": "oracle_attack",
            "attack_aux_weight": 0.15,
            "fusion_mode": "softmax_score",
        },
        {
            **base,
            "name": "v17_query_top2_confidence_baseline",
            "descriptor_mode": "query",
            "top_k": 2,
            "fusion_mode": "softmax_score",
        },
    ]


def train_moe_fusion_variant_cv(df: pd.DataFrame, variant: dict[str, Any], output_dir: str | Path, seed: int) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    nc_cols = [c for c in df.columns if c.startswith("nc__")]
    role_order = [c.replace("nc__", "") for c in nc_cols]
    x, feature_cols = _feature_matrix(df, str(variant.get("descriptor_mode", "paired")))
    y = _compose_router_targets(
        df=df,
        role_order=role_order,
        oracle_margin=float(variant.get("oracle_margin", 0.02)),
        target_mode=str(variant.get("target_mode", "oracle")),
        attack_aux_weight=float(variant.get("attack_aux_weight", 0.0)),
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
    confidence_gap = float(variant.get("confidence_gap", 0.16))
    entropy_threshold = float(variant.get("entropy_threshold", 0.76))
    fusion_mode = str(variant.get("fusion_mode", "softmax_score"))
    alpha_grid = [float(v) for v in variant.get("alpha_grid", [0.0, 4.0, 8.0, 16.0])]
    router_power_grid = [float(v) for v in variant.get("router_power_grid", [1.0])]
    fusion_select_metric = str(variant.get("fusion_select_metric", "paper"))
    protected_roles = [str(x) for x in variant.get("protected_roles", [])]
    max_k_after_protect = int(variant.get("max_k_after_protect", max(adaptive_max_k, top_k)))
    protect_when_uncertain = bool(variant.get("protect_when_uncertain", True))

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
            epochs=int(variant.get("epochs", 1800)),
            lr=float(variant.get("lr", 0.04)),
            l2=float(variant.get("l2", 1e-4)),
            seed=seed + fold_idx,
        )
        probs_train = _predict_probs(x_train, w)
        probs_eval = _predict_probs(x_eval, w)
        selected_train, _, gap_train, ent_train = _top_indices(
            probs_train,
            top_k=top_k,
            adaptive=adaptive,
            adaptive_max_k=adaptive_max_k,
            confidence_gap=confidence_gap,
            entropy_threshold=entropy_threshold,
        )
        selected_eval, used_ks, gap_eval, ent_eval = _top_indices(
            probs_eval,
            top_k=top_k,
            adaptive=adaptive,
            adaptive_max_k=adaptive_max_k,
            confidence_gap=confidence_gap,
            entropy_threshold=entropy_threshold,
        )
        if protected_roles:
            selected_train = _apply_protected_roles(
                selected_train,
                probs_train,
                role_order,
                protected_roles,
                protect_when_uncertain,
                gap_train,
                ent_train,
                confidence_gap,
                entropy_threshold,
                max_k_after_protect,
            )
            selected_eval = _apply_protected_roles(
                selected_eval,
                probs_eval,
                role_order,
                protected_roles,
                protect_when_uncertain,
                gap_eval,
                ent_eval,
                confidence_gap,
                entropy_threshold,
                max_k_after_protect,
            )
            used_ks = [int(len(x)) for x in selected_eval]

        best_alpha, best_power, best_train_obj = _choose_fusion_alpha(
            probs_train=probs_train,
            nc_train=nc[train_mask],
            selected_train=selected_train,
            role_order=role_order,
            fusion_mode=fusion_mode,
            alpha_grid=alpha_grid,
            router_power_grid=router_power_grid,
            metric=fusion_select_metric,
        )
        scores, roles, used_ks, weights = _fuse_scores_for_indices(
            probs=probs_eval,
            nc=nc[eval_mask],
            selected=selected_eval,
            role_order=role_order,
            fusion_mode=fusion_mode,
            alpha=best_alpha,
            router_power=best_power,
        )
        eval_df = df.loc[eval_mask].copy().reset_index(drop=True)
        eval_df["fold"] = int(fold_idx)
        eval_df["router_score_nc"] = scores
        eval_df["router_roles"] = roles
        eval_df["router_weights"] = weights
        eval_df["router_used_k"] = used_ks
        eval_df["router_top1_role"] = [r.split(",")[0] if r else "" for r in roles]
        eval_df["router_gap"] = gap_eval
        eval_df["router_entropy_norm"] = ent_eval
        eval_df["oracle_top1_role_cv"] = eval_df["oracle_top1_role"].astype(str)
        eval_df["oracle_top1_nc_cv"] = pd.to_numeric(eval_df["oracle_top1_nc"], errors="coerce").fillna(0.0)
        eval_df["attack_role_label"] = eval_df.apply(_attack_role_from_row, axis=1)
        eval_df["fold_best_alpha"] = float(best_alpha)
        eval_df["fold_best_router_power"] = float(best_power)
        pred_rows.append(eval_df)
        fold_rows.append(
            {
                "variant": str(variant["name"]),
                "fold": int(fold_idx),
                "train_rows": int(train_mask.sum()),
                "eval_rows": int(eval_mask.sum()),
                "best_alpha": float(best_alpha),
                "best_router_power": float(best_power),
                "best_train_objective": float(best_train_obj),
                "eval_mean_nc": float(scores.mean()),
                "eval_min_nc": float(scores.min()),
                "loss_first": float(losses[0]) if losses else np.nan,
                "loss_last": float(losses[-1]) if losses else np.nan,
            }
        )
        write_json(
            out_dir / f"fold_{fold_idx}_model_v17.json",
            {
                "variant": variant,
                "weights": w.tolist(),
                "scaler": scaler,
                "feature_cols": feature_cols,
                "roles": role_order,
                "losses": losses,
                "best_alpha": best_alpha,
                "best_router_power": best_power,
                "best_train_objective": best_train_obj,
            },
        )

    if not pred_rows:
        raise RuntimeError(f"No predictions for variant {variant['name']}")
    pred = pd.concat(pred_rows, ignore_index=True)
    static_scores = pd.to_numeric(pred.get("static_gated_nc", pd.Series([0.0] * len(pred))), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    router_scores = pd.to_numeric(pred["router_score_nc"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    oracle_scores = pd.to_numeric(pred["oracle_top1_nc_cv"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    hit_top1 = float((pred["router_top1_role"].astype(str) == pred["oracle_top1_role_cv"].astype(str)).mean())
    coverage = _coverage_at_k(pred["router_roles"], pred["oracle_top1_role_cv"])

    pred_csv = out_dir / f"predictions_{variant['name']}_v17.csv"
    pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out_dir / f"fold_summary_{variant['name']}_v17.csv", index=False, encoding="utf-8-sig")
    _attack_summary(pred, "router_score_nc").to_csv(out_dir / f"attack_summary_{variant['name']}_v17.csv", index=False, encoding="utf-8-sig")
    used_k_counts = pred["router_used_k"].value_counts().sort_index().to_dict()
    summary = {
        "variant": str(variant["name"]),
        "descriptor_mode": str(variant.get("descriptor_mode", "paired")),
        "target_mode": str(variant.get("target_mode", "oracle")),
        "fusion_mode": str(fusion_mode),
        "top_k": int(top_k),
        "adaptive": bool(adaptive),
        "protected_roles": ",".join(protected_roles),
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


def run_v17(
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
        output_csv=out_dir / "sparse_moe_expert_table_paired_v17.csv",
    )
    oracle = evaluate_oracle_and_static(paired["output_csv"], out_dir / "oracle_eval", top_k=int(top_k))
    result: dict[str, Any] = {"build": build, "paired_descriptors": paired, "oracle": oracle}
    if build_only:
        return result

    if variants_json is not None:
        variants = json.loads(Path(variants_json).read_text(encoding="utf-8"))
    else:
        variants = _candidate_variant_defaults(
            cv_folds=cv_folds,
            epochs=router_epochs,
            lr=router_lr,
            l2=router_l2,
            oracle_margin=oracle_margin,
        )
    df = pd.read_csv(paired["output_csv"])
    variant_dir = ensure_dir(out_dir / "moe_fusion_variants")
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[V17-VARIANT] {variant.get('name')}", flush=True)
        summary = train_moe_fusion_variant_cv(df, variant, variant_dir / str(variant["name"]), seed=seed)
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    cols_first = [
        "variant",
        "descriptor_mode",
        "target_mode",
        "fusion_mode",
        "top_k",
        "adaptive",
        "protected_roles",
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
    summary_csv = out_dir / "moe_fusion_variant_summary_v17.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_json(out_dir / "moe_fusion_variant_summary_v17.json", {"variants": summaries})
    result["moe_fusion_variants"] = {"summary_csv": str(summary_csv), "num_variants": int(len(summaries))}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="V17-A frozen-expert sparse MoE router and trained score-fusion ablation runner")
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
        result = run_v17(
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
        write_json(Path(ns.output_dir) / "sparse_moe_fusion_train_V17_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] sparse_moe_fusion_train_V17: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
