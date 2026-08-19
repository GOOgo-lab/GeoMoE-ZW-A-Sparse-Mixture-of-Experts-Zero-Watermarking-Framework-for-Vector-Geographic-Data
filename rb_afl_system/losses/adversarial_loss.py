#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adversarial losses for feature-space discriminators."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _is_pair_discriminator(discriminator) -> bool:
    return bool(getattr(discriminator, "is_pair_discriminator", False))


def discriminator_loss(
    discriminator,
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negative: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Train discriminator.

    Pair-D mode:
      same pair (anchor, positive) -> 1
      diff pair (anchor, negative) -> 0

    Legacy mode:
      anchor -> 1, positive -> 0.
    """
    if discriminator is None:
        zero = torch.zeros((), device=z_anchor.device, dtype=z_anchor.dtype)
        return zero, {"d_acc": 0.0, "d_pos_logit": 0.0, "d_neg_logit": 0.0}

    smooth = float(label_smoothing)
    pos_label = 1.0 - smooth
    neg_label = smooth

    if _is_pair_discriminator(discriminator) and z_negative is not None:
        same_logits = discriminator.forward_pair(z_anchor.detach(), z_positive.detach())
        diff_logits = discriminator.forward_pair(z_anchor.detach(), z_negative.detach())
        same_labels = torch.full_like(same_logits, pos_label)
        diff_labels = torch.full_like(diff_logits, neg_label)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(same_logits, same_labels)
            + F.binary_cross_entropy_with_logits(diff_logits, diff_labels)
        )
        with torch.no_grad():
            same_ok = (torch.sigmoid(same_logits) >= 0.5).float().mean()
            diff_ok = (torch.sigmoid(diff_logits) < 0.5).float().mean()
            acc = 0.5 * (same_ok + diff_ok)
        return loss, {
            "d_acc": float(acc.detach().cpu()),
            "d_pos_logit": float(same_logits.detach().mean().cpu()),
            "d_neg_logit": float(diff_logits.detach().mean().cpu()),
        }

    real_logits = discriminator(z_anchor.detach())
    attack_logits = discriminator(z_positive.detach())
    real_labels = torch.full_like(real_logits, pos_label)
    attack_labels = torch.full_like(attack_logits, neg_label)
    loss = 0.5 * (
        F.binary_cross_entropy_with_logits(real_logits, real_labels)
        + F.binary_cross_entropy_with_logits(attack_logits, attack_labels)
    )
    with torch.no_grad():
        real_ok = (torch.sigmoid(real_logits) >= 0.5).float().mean()
        attack_ok = (torch.sigmoid(attack_logits) < 0.5).float().mean()
        acc = 0.5 * (real_ok + attack_ok)
    return loss, {
        "d_acc": float(acc.detach().cpu()),
        "d_pos_logit": float(real_logits.detach().mean().cpu()),
        "d_neg_logit": float(attack_logits.detach().mean().cpu()),
    }


def generator_adv_loss(
    discriminator,
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negative: torch.Tensor | None = None,
) -> torch.Tensor:
    if discriminator is None:
        return torch.zeros((), device=z_positive.device, dtype=z_positive.dtype)

    if _is_pair_discriminator(discriminator) and z_negative is not None:
        same_logits = discriminator.forward_pair(z_anchor, z_positive)
        diff_logits = discriminator.forward_pair(z_anchor, z_negative)
        same_loss = F.binary_cross_entropy_with_logits(same_logits, torch.ones_like(same_logits))
        diff_loss = F.binary_cross_entropy_with_logits(diff_logits, torch.zeros_like(diff_logits))
        return 0.5 * (same_loss + diff_loss)

    logits = discriminator(z_positive)
    labels = torch.ones_like(logits)
    return F.binary_cross_entropy_with_logits(logits, labels)
