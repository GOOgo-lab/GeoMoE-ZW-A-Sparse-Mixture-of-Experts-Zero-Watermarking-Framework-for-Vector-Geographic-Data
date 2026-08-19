#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Total generator loss with optional supervised contrastive and hard-pair learning."""
from __future__ import annotations

import torch

from rb_afl_system.losses.adversarial_loss import generator_adv_loss
from rb_afl_system.losses.bit_balance_loss import bit_balance_loss
from rb_afl_system.losses.consistency_loss import consistency_loss
from rb_afl_system.losses.hard_losses import batch_unique_cosine_loss, hard_negative_cosine_loss, hard_positive_cosine_loss
from rb_afl_system.losses.supervised_contrastive_loss import supervised_contrastive_loss
from rb_afl_system.losses.triplet_loss import triplet_loss


def generator_total_loss(
    discriminator,
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negative: torch.Tensor,
    config: dict,
    contrast_features: torch.Tensor | None = None,
    contrast_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute all active generator-side losses."""
    loss_adv = generator_adv_loss(discriminator, z_anchor, z_positive, z_negative)
    loss_cons = consistency_loss(z_anchor, z_positive, mode=str(config.get("consistency_mode", "mse")))
    loss_trip = triplet_loss(z_anchor, z_positive, z_negative, margin=float(config.get("triplet_margin", 0.8)))
    loss_bit = bit_balance_loss(torch.cat([z_anchor, z_positive, z_negative], dim=0))
    if contrast_features is not None and contrast_labels is not None and float(config.get("lambda_supcon", 0.0)) > 0.0:
        loss_supcon = supervised_contrastive_loss(
            contrast_features,
            contrast_labels,
            temperature=float(config.get("supcon_temperature", 0.2)),
        )
    else:
        loss_supcon = torch.zeros((), device=z_anchor.device, dtype=z_anchor.dtype)
    loss_hard_neg = hard_negative_cosine_loss(
        z_anchor,
        z_negative,
        margin=float(config.get("hard_negative_margin", config.get("unique_nc_margin", 0.6))),
    )
    loss_hard_pos = hard_positive_cosine_loss(
        z_anchor,
        z_positive,
        target=float(config.get("hard_positive_target", config.get("robust_nc_target", 0.9))),
    )
    loss_batch_unique = batch_unique_cosine_loss(
        z_anchor,
        margin=float(config.get("batch_unique_margin", config.get("hard_negative_margin", 0.5))),
    )
    total = (
        float(config.get("w_adv", 1.0)) * loss_adv
        + float(config.get("lambda_cons", 5.0)) * loss_cons
        + float(config.get("gamma_triplet", 6.0)) * loss_trip
        + float(config.get("lambda_bit", 0.0)) * loss_bit
        + float(config.get("lambda_supcon", 0.0)) * loss_supcon
        + float(config.get("lambda_hard_neg", 0.0)) * loss_hard_neg
        + float(config.get("lambda_hard_pos", 0.0)) * loss_hard_pos
        + float(config.get("lambda_batch_unique", 0.0)) * loss_batch_unique
    )
    return total, {
        "loss_adv": float(loss_adv.detach().cpu()),
        "loss_cons": float(loss_cons.detach().cpu()),
        "loss_triplet": float(loss_trip.detach().cpu()),
        "loss_bit": float(loss_bit.detach().cpu()),
        "loss_supcon": float(loss_supcon.detach().cpu()),
        "loss_hard_neg": float(loss_hard_neg.detach().cpu()),
        "loss_hard_pos": float(loss_hard_pos.detach().cpu()),
        "loss_batch_unique": float(loss_batch_unique.detach().cpu()),
    }
