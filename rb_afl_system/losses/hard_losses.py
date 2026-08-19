#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard-pair losses for NC-priority training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_negative_cosine_loss(z_anchor: torch.Tensor, z_negative: torch.Tensor, margin: float = 0.6) -> torch.Tensor:
    """Penalize different identities whose cosine similarity exceeds margin."""
    sim = F.cosine_similarity(z_anchor, z_negative, dim=1)
    return F.relu(sim - float(margin)).pow(2).mean()


def hard_positive_cosine_loss(z_anchor: torch.Tensor, z_positive: torch.Tensor, target: float = 0.9) -> torch.Tensor:
    """Penalize attack positives whose cosine similarity is below target."""
    sim = F.cosine_similarity(z_anchor, z_positive, dim=1)
    return F.relu(float(target) - sim).pow(2).mean()


def batch_unique_cosine_loss(z_anchor: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Penalize high similarity among different anchors inside the same batch.

    This is a V11.1 addition.  It is not a replacement for memory-bank hard
    mining; it adds an always-on within-batch separation term whenever batch
    size is larger than 1.  The loss directly targets the observed failure mode:
    maximum unique NC remains high even when average losses look acceptable.
    """
    if z_anchor.ndim != 2 or z_anchor.shape[0] < 2:
        return torch.zeros((), device=z_anchor.device, dtype=z_anchor.dtype)
    z = F.normalize(z_anchor, dim=1)
    sim = z @ z.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    if not mask.any():
        return torch.zeros((), device=z_anchor.device, dtype=z_anchor.dtype)
    hard_sim = sim[mask]
    return F.relu(hard_sim - float(margin)).pow(2).mean()
