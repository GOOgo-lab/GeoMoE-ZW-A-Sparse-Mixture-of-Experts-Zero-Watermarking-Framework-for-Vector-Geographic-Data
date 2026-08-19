#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Proxy/Angular-margin losses for uniqueness-focused RB-AFL training.

These losses are intentionally used only during training.  The learned proxy
classifier is not needed at evaluation time; it acts as an identity-separation
regularizer for the generator embedding space.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class UniqueProxyHead(torch.nn.Module):
    """Linear identity proxy head with normalized weights."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("UniqueProxyHead requires at least two classes")
        self.weight = torch.nn.Parameter(torch.empty(num_classes, feat_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = F.normalize(features, dim=1)
        w = F.normalize(self.weight, dim=1)
        return z @ w.t()


def arcface_proxy_loss(
    proxy_head: UniqueProxyHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.35,
    scale: float = 24.0,
) -> torch.Tensor:
    """ArcFace-style angular margin classification loss.

    Args:
        proxy_head: Learnable normalized class proxy head.
        features: Embeddings of shape ``[B, D]``.
        labels: Global identity labels of shape ``[B]``.
        margin: Additive angular margin in radians.
        scale: Logit scale.
    """
    if features.numel() == 0:
        return torch.zeros((), device=features.device, dtype=features.dtype)
    labels = labels.long().to(features.device)
    cosine = proxy_head(features).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cosine)
    target_logits = torch.cos(theta + float(margin))
    one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(dtype=cosine.dtype, device=features.device)
    logits = cosine * (1.0 - one_hot) + target_logits * one_hot
    return F.cross_entropy(logits * float(scale), labels)


def proxy_anchor_loss(
    proxy_head: UniqueProxyHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.10,
    alpha: float = 32.0,
) -> torch.Tensor:
    """ProxyAnchor-style metric loss with trainable identity proxies.

    This implementation follows the key idea of ProxyAnchor: pull samples toward
    their own proxy and push them away from all other proxies.  It is compact and
    dependency-free, which keeps it compatible with the existing RB-AFL trainer.
    """
    if features.numel() == 0:
        return torch.zeros((), device=features.device, dtype=features.dtype)
    labels = labels.long().to(features.device)
    sim = proxy_head(features)
    num_classes = int(sim.shape[1])
    one_hot = F.one_hot(labels, num_classes=num_classes).to(dtype=sim.dtype, device=features.device)

    pos_exp = torch.exp(-float(alpha) * (sim - float(margin))) * one_hot
    neg_exp = torch.exp(float(alpha) * (sim + float(margin))) * (1.0 - one_hot)

    pos_den = one_hot.sum(dim=0).clamp_min(1.0)
    pos_term = torch.log1p(pos_exp.sum(dim=0) / pos_den)
    neg_term = torch.log1p(neg_exp.sum(dim=0) / max(1, int(features.shape[0])))

    valid_pos = one_hot.sum(dim=0) > 0
    if bool(valid_pos.any()):
        pos_loss = pos_term[valid_pos].mean()
    else:
        pos_loss = torch.zeros((), device=features.device, dtype=features.dtype)
    neg_loss = neg_term.mean()
    return pos_loss + neg_loss


def proxy_pair_hard_negative_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.35,
) -> torch.Tensor:
    """Batch hard-negative cosine penalty with explicit global labels."""
    if features.ndim != 2 or features.shape[0] < 2:
        return torch.zeros((), device=features.device, dtype=features.dtype)
    labels = labels.long().to(features.device)
    z = F.normalize(features, dim=1)
    sim = z @ z.t()
    diff_label = labels[:, None] != labels[None, :]
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=features.device)
    mask = diff_label & (~eye)
    if not bool(mask.any()):
        return torch.zeros((), device=features.device, dtype=features.dtype)
    return F.relu(sim[mask] - float(margin)).pow(2).mean()
