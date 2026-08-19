#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adversarial / metric trainer with V11 hard mining.

V11 changes the training objective from only average triplet/consistency losses
to explicit worst-case learning:
- online hard-negative mining from current base embeddings,
- online hard-attack positive mining from current base-vs-attack embeddings,
- pair discriminator support with useful D diagnostics,
- early stopping and ReduceLROnPlateau.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from rb_afl_system.losses.proxy_metric_losses import (
    UniqueProxyHead,
    arcface_proxy_loss,
    proxy_anchor_loss,
    proxy_pair_hard_negative_loss,
)

from rb_afl_system.data.dataset.triplet_dataset import IdentityTripletDataset, _load_sample, collate_triplet
from rb_afl_system.engine.checkpoint_io import save_checkpoint
from rb_afl_system.engine.device import resolve_device
from rb_afl_system.losses.adversarial_loss import discriminator_loss
from rb_afl_system.losses.total_loss import generator_total_loss
from rb_afl_system.models.model_registry import build_discriminator, build_generator
from rb_afl_system.utils import ensure_dir, log, set_seed, write_json


def _sample_to_device(sample: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}


def _forward_generator(generator, sample: dict) -> torch.Tensor:
    return generator(
        grid=sample["grid"],
        tokens=sample["tokens"],
        token_mask=sample["token_mask"],
        graph_nodes=sample["graph_nodes"],
        graph_adj=sample["graph_adj"],
        graph_mask=sample["graph_mask"],
    )


def _labels_from_batch(batch: dict, device: torch.device) -> torch.Tensor:
    ids = list(batch.get("identity", []))
    neg_ids = list(batch.get("negative_identity", []))
    all_ids = ids + ids + neg_ids
    mapping: dict[str, int] = {}
    vals = []
    for item in all_ids:
        key = str(item)
        if key not in mapping:
            mapping[key] = len(mapping)
        vals.append(mapping[key])
    return torch.tensor(vals, dtype=torch.long, device=device)




def _global_labels_from_batch(batch: dict, identity_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    ids = list(batch.get("identity", []))
    neg_ids = list(batch.get("negative_identity", []))
    all_ids = ids + ids + neg_ids
    vals = []
    for item in all_ids:
        key = str(item)
        if key not in identity_to_idx:
            raise KeyError(f"Identity {key!r} not found in identity_to_idx")
        vals.append(int(identity_to_idx[key]))
    return torch.tensor(vals, dtype=torch.long, device=device)

def _row_to_device_sample(row: dict, channels: list[str], device: torch.device) -> dict:
    sample = _load_sample(row["sample_dir"], channels)
    return {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}


def _cos_np(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def update_hard_mining_maps(
    generator,
    dataset: IdentityTripletDataset,
    device: torch.device,
    cfg: dict,
    epoch: int,
    out_dir,
) -> dict:
    """Compute current embedding-space hard maps and update dataset."""
    generator.eval()
    channels = list(cfg.get("channels", ["occ", "dist", "orient", "density"]))
    top_k = int(cfg.get("hard_mining_top_k", 16))
    unique_threshold = float(cfg.get("hard_negative_update_threshold", 0.0))
    positive_bottom_k = int(cfg.get("hard_positive_top_k", 6))
    use_threshold = bool(cfg.get("hard_negative_use_threshold", False))

    base_rows = [r for r in dataset.rows if str(r.get("attack_type")) == "base"]
    if len(base_rows) < 2:
        return {"hard_negative_pairs": 0, "hard_positive_samples": 0}

    base_emb: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for row in base_rows:
            sample = _row_to_device_sample(row, channels, device)
            z = _forward_generator(generator, sample).detach().cpu().numpy()[0]
            base_emb[str(row["identity"])] = z

    all_pair_scores: list[tuple[str, str, float]] = []
    ids = sorted(base_emb)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sim = _cos_np(base_emb[a], base_emb[b])
            all_pair_scores.append((a, b, sim))
    all_pair_scores.sort(key=lambda x: x[2], reverse=True)

    if use_threshold:
        pair_scores = [x for x in all_pair_scores if x[2] >= unique_threshold]
    else:
        pair_scores = list(all_pair_scores)

    hard_neg: dict[str, list[str]] = defaultdict(list)
    for a, b, sim in pair_scores:
        if b not in hard_neg[a] and len(hard_neg[a]) < top_k:
            hard_neg[a].append(b)
        if a not in hard_neg[b] and len(hard_neg[b]) < top_k:
            hard_neg[b].append(a)
    hard_neg = {k: v[:top_k] for k, v in hard_neg.items()}

    hard_pos: dict[str, list[str]] = defaultdict(list)
    rows_by_identity: dict[str, list[dict]] = defaultdict(list)
    for row in dataset.rows:
        rows_by_identity[str(row["identity"])].append(row)

    with torch.no_grad():
        for identity, rows in rows_by_identity.items():
            if identity not in base_emb:
                continue
            candidates: list[tuple[str, float]] = []
            for row in rows:
                if str(row.get("attack_type")) == "base":
                    continue
                sample = _row_to_device_sample(row, channels, device)
                z = _forward_generator(generator, sample).detach().cpu().numpy()[0]
                sim = _cos_np(base_emb[identity], z)
                candidates.append((str(row["sample_dir"]), sim))
            candidates.sort(key=lambda x: x[1])
            hard_pos[identity] = [p for p, _ in candidates[:positive_bottom_k]]

    dataset.set_hard_maps(
        hard_negative_map=hard_neg,
        hard_positive_map=hard_pos,
        hard_negative_prob=float(cfg.get("hard_negative_prob", 0.75)),
        hard_positive_prob=float(cfg.get("hard_positive_prob", 0.75)),
    )

    report = {
        "epoch": int(epoch),
        "hard_negative_pairs": int(len(pair_scores)),
        "hard_negative_pairs_above_threshold": int(sum(1 for _, _, sim in all_pair_scores if sim >= unique_threshold)),
        "hard_negative_identities": int(len(hard_neg)),
        "hard_positive_identities": int(len(hard_pos)),
        "hard_positive_samples": int(sum(len(v) for v in hard_pos.values())),
        "top_pairs": [
            {"identity_a": a, "identity_b": b, "cosine": float(sim)}
            for a, b, sim in all_pair_scores[: min(50, len(all_pair_scores))]
        ],
    }
    mining_dir = ensure_dir(out_dir / "hard_mining")
    (mining_dir / f"hard_mining_epoch_{epoch:03d}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"[HARD-MINING] epoch={epoch} hard_pairs={report['hard_negative_pairs']} "
        f"hard_neg_ids={report['hard_negative_identities']} hard_pos_samples={report['hard_positive_samples']}"
    )
    return report


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, device: torch.device, config: dict, train: bool = True, proxy_head=None, identity_to_idx: dict[str, int] | None = None) -> dict:
    generator.train(train)
    if discriminator is not None:
        discriminator.train(train)
    sums: dict[str, float] = {
        "loss_g": 0.0,
        "loss_d": 0.0,
        "loss_adv": 0.0,
        "loss_cons": 0.0,
        "loss_triplet": 0.0,
        "loss_bit": 0.0,
        "loss_supcon": 0.0,
        "loss_hard_neg": 0.0,
        "loss_hard_pos": 0.0,
        "loss_batch_unique": 0.0,
        "loss_arcface": 0.0,
        "loss_proxy_anchor": 0.0,
        "loss_proxy_pair_hard_neg": 0.0,
        "d_acc": 0.0,
        "d_pos_logit": 0.0,
        "d_neg_logit": 0.0,
    }
    count = 0
    d_steps = max(1, int(config.get("d_steps", 1)))
    for batch in loader:
        anchor = _sample_to_device(batch["anchor"], device)
        positive = _sample_to_device(batch["positive"], device)
        negative = _sample_to_device(batch["negative"], device)
        bsz = int(anchor["grid"].shape[0])

        if train and discriminator is not None and opt_d is not None:
            loss_d = torch.zeros((), device=device)
            d_parts = {"d_acc": 0.0, "d_pos_logit": 0.0, "d_neg_logit": 0.0}
            for _ in range(d_steps):
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z_a_det = _forward_generator(generator, anchor)
                    z_p_det = _forward_generator(generator, positive)
                    z_n_det = _forward_generator(generator, negative)
                loss_d, d_parts = discriminator_loss(
                    discriminator,
                    z_a_det,
                    z_p_det,
                    z_n_det,
                    label_smoothing=float(config.get("d_label_smoothing", 0.05)),
                )
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(config.get("max_grad_norm_d", 5.0)))
                opt_d.step()
        else:
            with torch.no_grad() if not train else torch.enable_grad():
                z_a_det = _forward_generator(generator, anchor)
                z_p_det = _forward_generator(generator, positive)
                z_n_det = _forward_generator(generator, negative)
            loss_d, d_parts = discriminator_loss(discriminator, z_a_det, z_p_det, z_n_det)

        if train:
            opt_g.zero_grad(set_to_none=True)
        z_a = _forward_generator(generator, anchor)
        z_p = _forward_generator(generator, positive)
        z_n = _forward_generator(generator, negative)
        contrast_features = torch.cat([z_a, z_p, z_n], dim=0)
        contrast_labels = _labels_from_batch(batch, device)
        loss_g, parts = generator_total_loss(
            discriminator,
            z_a,
            z_p,
            z_n,
            config,
            contrast_features=contrast_features,
            contrast_labels=contrast_labels,
        )
        # V15 formal split note:
        # The proxy head is an identity-class training objective whose proxies are
        # built only from train identities.  In a strict identity-level formal
        # split, validation identities are intentionally unseen and therefore do
        # not exist in identity_to_idx.  Applying ArcFace/ProxyAnchor on val would
        # either crash or leak validation identities into the train-time proxy
        # table.  Keep proxy losses training-only; val_g remains based on the
        # class-agnostic metric/adversarial losses.
        if train and proxy_head is not None and identity_to_idx is not None:
            proxy_labels = _global_labels_from_batch(batch, identity_to_idx, device)
            if float(config.get("lambda_arcface", 0.0)) > 0.0:
                loss_arc = arcface_proxy_loss(
                    proxy_head,
                    contrast_features,
                    proxy_labels,
                    margin=float(config.get("arcface_margin", 0.35)),
                    scale=float(config.get("arcface_scale", 24.0)),
                )
            else:
                loss_arc = torch.zeros((), device=device)
            if float(config.get("lambda_proxy_anchor", 0.0)) > 0.0:
                loss_proxy = proxy_anchor_loss(
                    proxy_head,
                    contrast_features,
                    proxy_labels,
                    margin=float(config.get("proxy_anchor_margin", 0.10)),
                    alpha=float(config.get("proxy_anchor_alpha", 32.0)),
                )
            else:
                loss_proxy = torch.zeros((), device=device)
            if float(config.get("lambda_proxy_pair_hard_neg", 0.0)) > 0.0:
                loss_pair_hn = proxy_pair_hard_negative_loss(
                    contrast_features,
                    proxy_labels,
                    margin=float(config.get("proxy_pair_hard_neg_margin", 0.35)),
                )
            else:
                loss_pair_hn = torch.zeros((), device=device)
            loss_g = (
                loss_g
                + float(config.get("lambda_arcface", 0.0)) * loss_arc
                + float(config.get("lambda_proxy_anchor", 0.0)) * loss_proxy
                + float(config.get("lambda_proxy_pair_hard_neg", 0.0)) * loss_pair_hn
            )
            parts["loss_arcface"] = float(loss_arc.detach().cpu())
            parts["loss_proxy_anchor"] = float(loss_proxy.detach().cpu())
            parts["loss_proxy_pair_hard_neg"] = float(loss_pair_hn.detach().cpu())
        else:
            parts["loss_arcface"] = 0.0
            parts["loss_proxy_anchor"] = 0.0
            parts["loss_proxy_pair_hard_neg"] = 0.0
        if train:
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), float(config.get("max_grad_norm", 5.0)))
            opt_g.step()
        sums["loss_g"] += float(loss_g.detach().cpu()) * bsz
        sums["loss_d"] += float(loss_d.detach().cpu()) * bsz
        for k in ["loss_adv", "loss_cons", "loss_triplet", "loss_bit", "loss_supcon", "loss_hard_neg", "loss_hard_pos", "loss_batch_unique", "loss_arcface", "loss_proxy_anchor", "loss_proxy_pair_hard_neg"]:
            sums[k] += float(parts.get(k, 0.0)) * bsz
        for k in ["d_acc", "d_pos_logit", "d_neg_logit"]:
            sums[k] += float(d_parts.get(k, 0.0)) * bsz
        count += bsz
    return {k: v / max(1, count) for k, v in sums.items()}


def train_adversarial(config: Dict[str, Any]) -> dict:
    set_seed(int(config.get("seed", 20260318)))
    out_dir = ensure_dir(config["output_dir"])
    device = resolve_device(config.get("device", "auto"))
    log(f"[DEVICE] {device}")
    channels = list(config.get("channels", ["occ", "dist", "orient", "density"]))
    cfg = dict(config)
    cfg["channels"] = channels
    cfg["in_channels"] = len(channels)
    train_dataset_root = str(cfg.get("train_dataset_root", cfg["dataset_root"]))
    val_dataset_root = str(cfg.get("val_dataset_root", "") or "")
    dataset = IdentityTripletDataset(
        train_dataset_root,
        channels=channels,
        positive_mode=str(cfg.get("positive_mode", "shp_aug")),
        seed=int(cfg.get("seed", 20260318)),
        hard_negative_prob=float(cfg.get("hard_negative_prob", 0.0)),
        hard_positive_prob=float(cfg.get("hard_positive_prob", 0.0)),
        positive_attack_keywords=list(cfg.get("positive_attack_keywords", [])),
    )
    dims = dataset.infer_dims()
    cfg.update(dims)
    cfg["train_dataset_root"] = train_dataset_root
    log(f"[DATA-DIMS] {dims}")

    if val_dataset_root:
        val_dataset = IdentityTripletDataset(
            val_dataset_root,
            channels=channels,
            positive_mode=str(cfg.get("positive_mode", "shp_aug")),
            seed=int(cfg.get("seed", 20260318)) + 17,
            hard_negative_prob=0.0,
            hard_positive_prob=0.0,
            positive_attack_keywords=list(cfg.get("val_positive_attack_keywords", cfg.get("positive_attack_keywords", []))),
        )
        train_ds = dataset
        val_ds = val_dataset
        cfg["val_dataset_root"] = val_dataset_root
        cfg["formal_split_mode"] = True
        log(f"[FORMAL-SPLIT] train_dataset_root={train_dataset_root}")
        log(f"[FORMAL-SPLIT] val_dataset_root={val_dataset_root}")
    else:
        val_ratio = float(cfg.get("val_ratio", 0.2))
        val_size = max(1, int(len(dataset) * val_ratio))
        train_size = max(1, len(dataset) - val_size)
        if train_size + val_size > len(dataset):
            val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 20260318))),
        )
        cfg["formal_split_mode"] = False
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_triplet,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_triplet,
    )
    generator = build_generator(cfg).to(device)
    discriminator = build_discriminator(cfg)
    if discriminator is not None:
        discriminator = discriminator.to(device)
    identity_to_idx = {str(identity): i for i, identity in enumerate(dataset.identities)}
    cfg["num_train_identities"] = int(len(identity_to_idx))
    cfg["identity_to_idx"] = identity_to_idx
    use_proxy_head = any(float(cfg.get(k, 0.0)) > 0.0 for k in ["lambda_arcface", "lambda_proxy_anchor", "lambda_proxy_pair_hard_neg"])
    proxy_head = UniqueProxyHead(feat_dim=int(cfg.get("feat_dim", 256)), num_classes=len(identity_to_idx)).to(device) if use_proxy_head else None
    g_params = list(generator.parameters())
    if proxy_head is not None:
        g_params += list(proxy_head.parameters())
        log(f"[UNIQUE-PROXY] enabled num_classes={len(identity_to_idx)} arcface={cfg.get('lambda_arcface', 0.0)} proxy_anchor={cfg.get('lambda_proxy_anchor', 0.0)}")
    opt_g = torch.optim.AdamW(
        g_params,
        lr=float(cfg.get("lr_g", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    opt_d = None if discriminator is None else torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(cfg.get("lr_d", 1e-3)),
        weight_decay=float(cfg.get("weight_decay_d", cfg.get("weight_decay", 1e-4))),
    )

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_g,
        mode="min",
        factor=float(cfg.get("lr_plateau_factor", 0.5)),
        patience=int(cfg.get("lr_plateau_patience", 8)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
    ) if bool(cfg.get("use_lr_scheduler", True)) else None
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_d,
        mode="min",
        factor=float(cfg.get("lr_plateau_factor", 0.5)),
        patience=int(cfg.get("lr_plateau_patience", 8)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
    ) if opt_d is not None and bool(cfg.get("use_lr_scheduler", True)) else None

    best_val = float("inf")
    best_epoch = 0
    history = []
    epochs = int(cfg.get("epochs", 30))
    patience = int(cfg.get("early_stop_patience", 0))
    min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
    mining_interval = int(cfg.get("hard_mining_interval", 1))
    mining_start_epoch = int(cfg.get("hard_mining_start_epoch", 1))
    mining_report: dict[str, Any] = {}

    write_json(out_dir / "train_config.json", cfg)
    if bool(cfg.get("enable_hard_mining", False)) and mining_start_epoch <= 1:
        mining_report = update_hard_mining_maps(generator, dataset, device, cfg, 0, out_dir)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if bool(cfg.get("enable_hard_mining", False)) and epoch >= mining_start_epoch and (epoch - mining_start_epoch) % max(1, mining_interval) == 0:
            mining_report = update_hard_mining_maps(generator, dataset, device, cfg, epoch, out_dir)
        train_stats = train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d, device, cfg, train=True, proxy_head=proxy_head, identity_to_idx=identity_to_idx)
        with torch.no_grad():
            val_stats = train_one_epoch(generator, discriminator, val_loader, opt_g, opt_d, device, cfg, train=False, proxy_head=proxy_head, identity_to_idx=identity_to_idx)
        if scheduler_g is not None:
            scheduler_g.step(val_stats["loss_g"])
        if scheduler_d is not None:
            scheduler_d.step(val_stats["loss_d"])
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "elapsed_s": elapsed,
            "lr_g": float(opt_g.param_groups[0]["lr"]),
            "lr_d": float(opt_d.param_groups[0]["lr"]) if opt_d is not None else 0.0,
            "hard_mining": mining_report,
        }
        history.append(rec)
        log(
            f"[EPOCH {epoch:03d}] train_g={train_stats['loss_g']:.6f} "
            f"val_g={val_stats['loss_g']:.6f} val_d={val_stats['loss_d']:.6f} "
            f"val_d_acc={val_stats['d_acc']:.3f} val_hneg={val_stats['loss_hard_neg']:.6f} "
            f"val_hpos={val_stats['loss_hard_pos']:.6f} val_buniq={val_stats['loss_batch_unique']:.6f} "
            f"val_supcon={val_stats['loss_supcon']:.6f} "
            f"val_arc={val_stats['loss_arcface']:.6f} val_proxy={val_stats['loss_proxy_anchor']:.6f} "
            f"lr_g={opt_g.param_groups[0]['lr']:.2e} time={elapsed:.2f}s"
        )
        save_checkpoint(out_dir / "last.pt", generator, discriminator, opt_g, opt_d, epoch, best_val, cfg)
        improved = val_stats["loss_g"] < (best_val - min_delta)
        if improved:
            best_val = val_stats["loss_g"]
            best_epoch = epoch
            save_checkpoint(out_dir / "best.pt", generator, discriminator, opt_g, opt_d, epoch, best_val, cfg)
            log(f"[SAVE-BEST] {out_dir / 'best.pt'} val_g={best_val:.6f}")
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        min_epochs = int(cfg.get("early_stop_min_epochs", 0))
        if patience > 0 and epoch >= min_epochs and epoch - best_epoch >= patience:
            log(f"[EARLY-STOP] epoch={epoch} best_epoch={best_epoch} best_val={best_val:.6f}")
            break
    return {"out_dir": str(out_dir), "best_val": best_val, "best_epoch": best_epoch, "epochs": len(history)}
