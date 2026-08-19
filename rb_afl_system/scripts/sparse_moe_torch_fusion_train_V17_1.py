#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V17.1 PyTorch sparse MoE router/fusion training.

This script is the next stage after V17-A.  It still keeps the V15 expert
backbones and W_unique frozen, but replaces the previous linear/logistic router
with a small PyTorch RouterNet and optimizes a differentiable score-level
fusion objective.

What is trained:
    * RouterNet over paired/query descriptors.
    * Score-fusion calibration through a differentiable robust-score loss.

What is not trained:
    * CNN / GeoVecFormer / graph expert backbones.
    * W_unique identity gate.
    * G/D adversarial expert checkpoints.

This is intentionally a middle stage between post-hoc routing and full
end-to-end expert fine-tuning.  It produces epoch logs, fold model checkpoints
(.pt), CV predictions, attack summaries, and variant summaries suitable for
MoE ablation tables.
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - runtime dependency check
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from rb_afl_system.scripts.sparse_moe_router_V16 import (
    BASE_EXPERT_ROLES,
    _attack_summary,
    _bool_arg,
    _make_oracle_labels,
    _safe_float,
    _standardize_train_eval,
    _summary_from_scores,
    build_sparse_moe_table,
    evaluate_oracle_and_static,
)
from rb_afl_system.scripts.sparse_moe_router_V16_1 import (
    _compose_router_targets,
    _coverage_at_k,
    _feature_matrix,
    enrich_paired_descriptors,
)
from rb_afl_system.scripts.sparse_moe_fusion_train_V17 import (
    _apply_protected_roles,
    _choose_fusion_alpha,
    _entropy_normalized,
    _fuse_scores_for_indices,
    _top_indices,
)
from rb_afl_system.utils import ensure_dir, write_json


@dataclass
class TorchVariant:
    name: str
    descriptor_mode: str = "paired"
    target_mode: str = "oracle"
    top_k: int = 2
    fusion_mode: str = "max"
    adaptive: bool = False
    adaptive_max_k: int = 3
    protected_roles: tuple[str, ...] = ()
    protect_when_uncertain: bool = True
    confidence_gap: float = 0.16
    entropy_threshold: float = 0.76
    attack_aux_weight: float = 0.20
    topology_boost: bool = False
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.10
    temperature: float = 0.80
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 800
    patience: int = 160
    lambda_route: float = 1.0
    lambda_pos: float = 1.5
    lambda_entropy: float = 0.04
    lambda_balance: float = 0.08
    pos_target: float = 0.92
    batch_size: int = 0


if nn is not None:
    class RouterNet(nn.Module):
        """Small MLP router over cheap descriptors."""

        def __init__(self, input_dim: int, num_roles: int, hidden_dims: list[int], dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            last_dim = int(input_dim)
            for hidden in hidden_dims:
                hidden_i = int(hidden)
                layers.append(nn.Linear(last_dim, hidden_i))
                layers.append(nn.LayerNorm(hidden_i))
                layers.append(nn.GELU())
                if float(dropout) > 0:
                    layers.append(nn.Dropout(float(dropout)))
                last_dim = hidden_i
            layers.append(nn.Linear(last_dim, int(num_roles)))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError(f"PyTorch is required for V17.1 torch MoE training: {_TORCH_IMPORT_ERROR!r}")


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _role_order_from_table(df: pd.DataFrame) -> list[str]:
    return [c.replace("nc__", "") for c in df.columns if c.startswith("nc__")]


def _torch_entropy_normalized(probs: torch.Tensor) -> torch.Tensor:
    q = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ent = -(q * torch.log(q.clamp_min(1e-8))).sum(dim=1)
    return ent / max(math.log(max(2, probs.shape[1])), 1e-8)


def _soft_balance_target(y: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float32)
    freq = yy.sum(axis=0)
    if float(freq.sum()) <= 1e-8:
        return np.full(yy.shape[1], 1.0 / max(1, yy.shape[1]), dtype=np.float32)
    freq = freq / float(freq.sum())
    # Smooth to avoid forcing unused rare experts to zero exactly.
    smooth = 0.03 / float(max(1, yy.shape[1]))
    freq = 0.97 * freq + smooth
    freq = freq / float(freq.sum())
    return freq.astype(np.float32)


def _make_batches(num_rows: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    if batch_size <= 0 or batch_size >= num_rows:
        return [np.arange(num_rows, dtype=np.int64)]
    idx = np.arange(num_rows, dtype=np.int64)
    rng.shuffle(idx)
    return [idx[i:i + batch_size] for i in range(0, num_rows, batch_size)]


def _train_router_torch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    nc_train: np.ndarray,
    variant: TorchVariant,
    seed: int,
    device: str,
    log_prefix: str,
) -> tuple[Any, dict[str, Any]]:
    _require_torch()
    _set_seed(seed)
    dev = torch.device(device)
    model = RouterNet(
        input_dim=int(x_train.shape[1]),
        num_roles=int(y_train.shape[1]),
        hidden_dims=list(variant.hidden_dims),
        dropout=float(variant.dropout),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(variant.weight_decay))

    x_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    nc_t = torch.tensor(nc_train, dtype=torch.float32, device=dev)
    balance_target = torch.tensor(_soft_balance_target(y_train), dtype=torch.float32, device=dev)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed + 7001)

    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_obj = 1e18
    stale = 0
    epochs = int(max(1, variant.epochs))
    patience = int(max(1, variant.patience))
    batch_size = int(variant.batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_route: list[float] = []
        epoch_pos: list[float] = []
        epoch_ent: list[float] = []
        epoch_bal: list[float] = []
        for batch_idx in _make_batches(int(x_train.shape[0]), batch_size, rng):
            xb = x_t[batch_idx]
            yb = y_t[batch_idx]
            ncb = nc_t[batch_idx]
            logits = model(xb)
            route_loss = bce(logits, yb)
            router_probs = torch.softmax(logits / max(float(variant.temperature), 1e-4), dim=1)
            fused_soft = (router_probs * ncb).sum(dim=1)
            pos_loss = F.relu(float(variant.pos_target) - fused_soft).mean()
            ent = _torch_entropy_normalized(router_probs).mean()
            bal = F.mse_loss(router_probs.mean(dim=0), balance_target)
            loss = (
                float(variant.lambda_route) * route_loss
                + float(variant.lambda_pos) * pos_loss
                + float(variant.lambda_entropy) * ent
                + float(variant.lambda_balance) * bal
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_route.append(float(route_loss.detach().cpu()))
            epoch_pos.append(float(pos_loss.detach().cpu()))
            epoch_ent.append(float(ent.detach().cpu()))
            epoch_bal.append(float(bal.detach().cpu()))

        model.eval()
        with torch.no_grad():
            logits_all = model(x_t)
            route_all = bce(logits_all, y_t)
            probs_all = torch.softmax(logits_all / max(float(variant.temperature), 1e-4), dim=1)
            fused_all = (probs_all * nc_t).sum(dim=1)
            pos_all = F.relu(float(variant.pos_target) - fused_all).mean()
            ent_all = _torch_entropy_normalized(probs_all).mean()
            bal_all = F.mse_loss(probs_all.mean(dim=0), balance_target)
            obj = (
                float(variant.lambda_route) * route_all
                + float(variant.lambda_pos) * pos_all
                + float(variant.lambda_entropy) * ent_all
                + float(variant.lambda_balance) * bal_all
            )
            train_mean_score = float(fused_all.mean().detach().cpu())
            train_min_score = float(fused_all.min().detach().cpu())
            obj_float = float(obj.detach().cpu())

        row = {
            "epoch": float(epoch),
            "loss": float(np.mean(epoch_losses)),
            "route_loss": float(route_all.detach().cpu()),
            "pos_loss": float(pos_all.detach().cpu()),
            "entropy_loss": float(ent_all.detach().cpu()),
            "balance_loss": float(bal_all.detach().cpu()),
            "train_soft_mean_nc": train_mean_score,
            "train_soft_min_nc": train_min_score,
        }
        history.append(row)

        if obj_float < best_obj - 1e-7:
            best_obj = obj_float
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(
                f"[{log_prefix}] epoch={epoch:04d} loss={row['loss']:.6f} "
                f"route={row['route_loss']:.6f} pos={row['pos_loss']:.6f} "
                f"soft_mean={train_mean_score:.6f} soft_min={train_min_score:.6f}",
                flush=True,
            )
        if stale >= patience:
            print(f"[{log_prefix}] early_stop epoch={epoch} best_obj={best_obj:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "history": history,
        "best_objective": float(best_obj),
        "epochs_ran": int(len(history)),
        "early_stopped": bool(len(history) < epochs),
    }
    return model, info


def _predict_torch(model: Any, x_eval: np.ndarray, device: str, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    _require_torch()
    model.eval()
    dev = torch.device(device)
    with torch.no_grad():
        x_t = torch.tensor(x_eval, dtype=torch.float32, device=dev)
        logits = model(x_t)
        sigmoid_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        softmax_probs = torch.softmax(logits / max(float(temperature), 1e-4), dim=1).detach().cpu().numpy().astype(np.float32)
    return sigmoid_probs, softmax_probs


def _variant_to_dataclass(variant: dict[str, Any], default_epochs: int, default_lr: float, default_weight_decay: float) -> TorchVariant:
    return TorchVariant(
        name=str(variant.get("name", "torch_variant")),
        descriptor_mode=str(variant.get("descriptor_mode", "paired")),
        target_mode=str(variant.get("target_mode", "oracle")),
        top_k=int(variant.get("top_k", 2)),
        fusion_mode=str(variant.get("fusion_mode", "max")),
        adaptive=bool(variant.get("adaptive", False)),
        adaptive_max_k=int(variant.get("adaptive_max_k", 3)),
        protected_roles=tuple(str(x).strip() for x in variant.get("protected_roles", []) if str(x).strip()),
        protect_when_uncertain=bool(variant.get("protect_when_uncertain", True)),
        confidence_gap=float(variant.get("confidence_gap", 0.16)),
        entropy_threshold=float(variant.get("entropy_threshold", 0.76)),
        attack_aux_weight=float(variant.get("attack_aux_weight", 0.20)),
        topology_boost=bool(variant.get("topology_boost", False)),
        hidden_dims=tuple(int(x) for x in variant.get("hidden_dims", [128, 64])),
        dropout=float(variant.get("dropout", 0.10)),
        temperature=float(variant.get("temperature", 0.80)),
        lr=float(variant.get("lr", default_lr)),
        weight_decay=float(variant.get("weight_decay", default_weight_decay)),
        epochs=int(variant.get("epochs", default_epochs)),
        patience=int(variant.get("patience", max(120, default_epochs // 5))),
        lambda_route=float(variant.get("lambda_route", 1.0)),
        lambda_pos=float(variant.get("lambda_pos", 1.5)),
        lambda_entropy=float(variant.get("lambda_entropy", 0.04)),
        lambda_balance=float(variant.get("lambda_balance", 0.08)),
        pos_target=float(variant.get("pos_target", 0.92)),
        batch_size=int(variant.get("batch_size", 0)),
    )


def _default_torch_variants(default_epochs: int, default_lr: float, default_weight_decay: float) -> list[dict[str, Any]]:
    base = {
        "epochs": int(default_epochs),
        "lr": float(default_lr),
        "weight_decay": float(default_weight_decay),
        "hidden_dims": [128, 64],
        "dropout": 0.10,
        "temperature": 0.80,
        "target_mode": "oracle",
        "descriptor_mode": "paired",
        "top_k": 2,
    }
    return [
        {**base, "name": "torch_paired_top2_mean", "fusion_mode": "weighted_mean"},
        {**base, "name": "torch_paired_top2_max", "fusion_mode": "max"},
        {**base, "name": "torch_paired_top2_confidence", "fusion_mode": "confidence", "lambda_pos": 1.8},
        {**base, "name": "torch_paired_top3_max", "top_k": 3, "fusion_mode": "max"},
        {**base, "name": "torch_paired_top3_confidence", "top_k": 3, "fusion_mode": "confidence", "lambda_pos": 1.8},
        {
            **base,
            "name": "torch_paired_protect_rotate_local_top2_max",
            "fusion_mode": "max",
            "protected_roles": ["rotate", "local_attack"],
            "confidence_gap": 0.18,
            "entropy_threshold": 0.76,
        },
        {
            **base,
            "name": "torch_paired_oracle_attack_top2_max",
            "fusion_mode": "max",
            "target_mode": "oracle_attack",
            "attack_aux_weight": 0.12,
        },
        {**base, "name": "torch_query_top2_max_baseline", "descriptor_mode": "query", "fusion_mode": "max"},
    ]


def train_torch_variant_cv(
    df: pd.DataFrame,
    variant_dict: dict[str, Any],
    output_dir: str | Path,
    seed: int,
    cv_folds: int,
    oracle_margin: float,
    device: str,
    default_epochs: int,
    default_lr: float,
    default_weight_decay: float,
) -> dict[str, Any]:
    variant = _variant_to_dataclass(variant_dict, default_epochs, default_lr, default_weight_decay)
    out_dir = ensure_dir(output_dir)
    nc_cols = [c for c in df.columns if c.startswith("nc__")]
    role_order = [c.replace("nc__", "") for c in nc_cols]
    x, feature_cols = _feature_matrix(df, variant.descriptor_mode)
    y = _compose_router_targets(
        df=df,
        role_order=role_order,
        oracle_margin=float(oracle_margin),
        target_mode=variant.target_mode,
        attack_aux_weight=float(variant.attack_aux_weight),
        topology_boost=bool(variant.topology_boost),
    )
    nc = df[nc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    identities = df["identity"].astype(str).fillna("__unknown__").to_numpy()
    uniq_ids = sorted(set(identities.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_ids)
    folds = np.array_split(np.array(uniq_ids, dtype=object), max(2, min(int(cv_folds), len(uniq_ids))))

    pred_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    alpha_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    power_grid = [0.5, 1.0, 1.5, 2.0]

    for fold_idx, eval_ids_arr in enumerate(folds):
        eval_ids = set(str(v) for v in eval_ids_arr.tolist())
        eval_mask = np.array([v in eval_ids for v in identities], dtype=bool)
        train_mask = ~eval_mask
        if train_mask.sum() <= 0 or eval_mask.sum() <= 0:
            continue
        x_train, x_eval, scaler = _standardize_train_eval(x[train_mask], x[eval_mask])
        log_prefix = f"V17.1 {variant.name} fold={fold_idx}"
        model, train_info = _train_router_torch(
            x_train=x_train,
            y_train=y[train_mask],
            nc_train=nc[train_mask],
            variant=variant,
            seed=seed + 1777 * fold_idx,
            device=device,
            log_prefix=log_prefix,
        )
        sigmoid_probs_eval, softmax_probs_eval = _predict_torch(model, x_eval, device=device, temperature=variant.temperature)
        sigmoid_probs_train, _ = _predict_torch(model, x_train, device=device, temperature=variant.temperature)
        selected_train, _, _, _ = _top_indices(
            sigmoid_probs_train,
            top_k=variant.top_k,
            adaptive=variant.adaptive,
            adaptive_max_k=variant.adaptive_max_k,
            confidence_gap=variant.confidence_gap,
            entropy_threshold=variant.entropy_threshold,
        )
        alpha, router_power, train_obj = _choose_fusion_alpha(
            sigmoid_probs_train,
            nc[train_mask],
            selected_train,
            role_order,
            fusion_mode=variant.fusion_mode,
            alpha_grid=alpha_grid,
            router_power_grid=power_grid,
            metric="paper",
        )
        selected_eval, used_ks, gap, ent = _top_indices(
            sigmoid_probs_eval,
            top_k=variant.top_k,
            adaptive=variant.adaptive,
            adaptive_max_k=variant.adaptive_max_k,
            confidence_gap=variant.confidence_gap,
            entropy_threshold=variant.entropy_threshold,
        )
        selected_eval = _apply_protected_roles(
            selected=selected_eval,
            probs=sigmoid_probs_eval,
            role_order=role_order,
            protected_roles=list(variant.protected_roles),
            protect_when_uncertain=bool(variant.protect_when_uncertain),
            gap=gap,
            ent=ent,
            confidence_gap=variant.confidence_gap,
            entropy_threshold=variant.entropy_threshold,
            max_k=max(int(variant.top_k), int(variant.adaptive_max_k), int(variant.top_k) + len(variant.protected_roles)),
        )
        scores, roles, used_ks, weight_strings = _fuse_scores_for_indices(
            probs=sigmoid_probs_eval,
            nc=nc[eval_mask],
            selected=selected_eval,
            role_order=role_order,
            fusion_mode=variant.fusion_mode,
            alpha=float(alpha),
            router_power=float(router_power),
        )

        eval_df = df.loc[eval_mask].copy().reset_index(drop=True)
        eval_df["router_variant"] = variant.name
        eval_df["router_fold"] = int(fold_idx)
        eval_df["router_score_nc"] = scores
        eval_df["router_roles"] = roles
        eval_df["router_weights"] = weight_strings
        eval_df["router_used_k"] = used_ks
        eval_df["router_top1_role"] = [role_order[int(i)] for i in sigmoid_probs_eval.argmax(axis=1)]
        eval_df["router_top1_prob"] = sigmoid_probs_eval.max(axis=1)
        eval_df["router_entropy_norm"] = _entropy_normalized(sigmoid_probs_eval)
        eval_df["oracle_top1_role_cv"] = [role_order[int(i)] for i in nc[eval_mask].argmax(axis=1)]
        eval_df["oracle_top1_nc_cv"] = nc[eval_mask].max(axis=1)
        for j, role in enumerate(role_order):
            eval_df[f"router_prob__{role}"] = sigmoid_probs_eval[:, j]
        pred_rows.append(eval_df)

        model_path = out_dir / f"fold_{fold_idx}_router_net_v17_1.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "variant": variant.__dict__,
                "feature_cols": feature_cols,
                "roles": role_order,
                "scaler": scaler,
                "alpha": float(alpha),
                "router_power": float(router_power),
                "train_info": train_info,
            },
            model_path,
        )
        hist_csv = out_dir / f"fold_{fold_idx}_training_history_v17_1.csv"
        pd.DataFrame(train_info["history"]).to_csv(hist_csv, index=False, encoding="utf-8-sig")
        write_json(
            out_dir / f"fold_{fold_idx}_model_meta_v17_1.json",
            {
                "variant": variant.__dict__,
                "model_path": str(model_path),
                "history_csv": str(hist_csv),
                "feature_cols": feature_cols,
                "roles": role_order,
                "scaler": scaler,
                "alpha": float(alpha),
                "router_power": float(router_power),
                "train_objective": float(train_obj),
                "train_info": train_info,
            },
        )
        fold_rows.append(
            {
                "variant": variant.name,
                "fold": int(fold_idx),
                "train_rows": int(train_mask.sum()),
                "eval_rows": int(eval_mask.sum()),
                "eval_identities": sorted(eval_ids),
                "alpha": float(alpha),
                "router_power": float(router_power),
                "epochs_ran": int(train_info.get("epochs_ran", 0)),
                "best_objective": float(train_info.get("best_objective", 0.0)),
                **_summary_from_scores(scores, "router"),
            }
        )

    if not pred_rows:
        raise RuntimeError(f"No predictions for variant {variant.name}")
    pred = pd.concat(pred_rows, ignore_index=True)
    pred_csv = out_dir / f"predictions_{variant.name}_v17_1.csv"
    pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / f"fold_summary_{variant.name}_v17_1.csv", index=False, encoding="utf-8-sig")
    _attack_summary(pred, "router_score_nc").to_csv(out_dir / f"attack_summary_{variant.name}_v17_1.csv", index=False, encoding="utf-8-sig")

    static_scores = pd.to_numeric(pred.get("static_gated_nc", pd.Series([0.0] * len(pred))), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    router_scores = pd.to_numeric(pred["router_score_nc"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    oracle_scores = pd.to_numeric(pred["oracle_top1_nc_cv"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    hit_top1 = float((pred["router_top1_role"].astype(str) == pred["oracle_top1_role_cv"].astype(str)).mean())
    coverage = _coverage_at_k(pred["router_roles"], pred["oracle_top1_role_cv"])
    used_k_counts = pred["router_used_k"].value_counts().sort_index().to_dict()
    summary = {
        "variant": variant.name,
        "descriptor_mode": variant.descriptor_mode,
        "target_mode": variant.target_mode,
        "fusion_mode": variant.fusion_mode,
        "top_k": int(variant.top_k),
        "adaptive": bool(variant.adaptive),
        "protected_roles": ",".join(variant.protected_roles),
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


def run_v17_1(
    profile_csv: str | Path,
    selected_json: str | Path,
    output_dir: str | Path,
    expert_roles: list[str],
    use_descriptors: bool,
    top_k: int,
    oracle_margin: float,
    cv_folds: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    variants_json: str | Path | None,
    build_only: bool,
) -> dict[str, Any]:
    _require_torch()
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
        output_csv=out_dir / "sparse_moe_expert_table_paired_v17_1.csv",
    )
    oracle = evaluate_oracle_and_static(paired["output_csv"], out_dir / "oracle_eval", top_k=int(top_k))
    result: dict[str, Any] = {"build": build, "paired_descriptors": paired, "oracle": oracle}
    if build_only:
        return result

    df = pd.read_csv(paired["output_csv"])
    if variants_json:
        variants = json.loads(Path(variants_json).read_text(encoding="utf-8"))
    else:
        variants = _default_torch_variants(default_epochs=int(epochs), default_lr=float(lr), default_weight_decay=float(weight_decay))
    variant_dir = ensure_dir(out_dir / "torch_moe_variants")
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[TORCH-MOE-VARIANT] {variant.get('name')}", flush=True)
        summary = train_torch_variant_cv(
            df=df,
            variant_dict=variant,
            output_dir=variant_dir / str(variant["name"]),
            seed=int(seed),
            cv_folds=int(cv_folds),
            oracle_margin=float(oracle_margin),
            device=str(device),
            default_epochs=int(epochs),
            default_lr=float(lr),
            default_weight_decay=float(weight_decay),
        )
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    first_cols = [
        "variant",
        "descriptor_mode",
        "target_mode",
        "fusion_mode",
        "top_k",
        "adaptive",
        "protected_roles",
        "router_mean_nc",
        "router_min_nc",
        "router_nc_lt_0_9",
        "router_nc_lt_0_8",
        "router_gain_mean_nc_vs_static",
        "router_gain_min_nc_vs_static",
        "router_top1_hit_rate_vs_oracle",
        "router_coverage_vs_oracle",
        "router_used_k_counts",
    ]
    keep_cols = [c for c in first_cols if c in summary_df.columns] + [c for c in summary_df.columns if c not in first_cols]
    summary_df = summary_df[keep_cols]
    summary_csv = out_dir / "torch_moe_fusion_variant_summary_v17_1.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_json(out_dir / "torch_moe_fusion_variant_summary_v17_1.json", {"variants": summaries})
    result["torch_moe_variants"] = {"summary_csv": str(summary_csv), "num_variants": int(len(summaries))}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="V17.1 PyTorch RouterNet + score-level sparse MoE fusion trainer")
    ap.add_argument("--profile_csv", required=True)
    ap.add_argument("--selected_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--expert_roles", default=",".join(BASE_EXPERT_ROLES))
    ap.add_argument("--use_descriptors", type=_bool_arg, default=True)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--oracle_margin", type=float, default=0.02)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--seed", type=int, default=20260318)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants_json", default=None)
    ap.add_argument("--build_only", type=_bool_arg, default=False)
    ns = ap.parse_args()
    try:
        roles = [x.strip() for x in str(ns.expert_roles).split(",") if x.strip()]
        result = run_v17_1(
            profile_csv=ns.profile_csv,
            selected_json=ns.selected_json,
            output_dir=ns.output_dir,
            expert_roles=roles,
            use_descriptors=bool(ns.use_descriptors),
            top_k=int(ns.top_k),
            oracle_margin=float(ns.oracle_margin),
            cv_folds=int(ns.cv_folds),
            epochs=int(ns.epochs),
            lr=float(ns.lr),
            weight_decay=float(ns.weight_decay),
            seed=int(ns.seed),
            device=str(ns.device),
            variants_json=ns.variants_json,
            build_only=bool(ns.build_only),
        )
        write_json(Path(ns.output_dir) / "sparse_moe_torch_fusion_train_V17_1_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] sparse_moe_torch_fusion_train_V17_1: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
