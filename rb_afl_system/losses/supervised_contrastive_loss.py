#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervised contrastive loss for identity-aware embedding learning."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected features [N,D], got {features.shape}")
    z = F.normalize(features, p=2, dim=1)
    sim = torch.matmul(z, z.t()) / float(temperature)
    logits_mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()) & logits_mask
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * logits_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = positive_mask.float().sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return torch.zeros((), device=features.device, dtype=features.dtype)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()
