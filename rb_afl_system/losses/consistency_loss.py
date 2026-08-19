#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn.functional as F


def consistency_loss(z_anchor: torch.Tensor, z_positive: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    if mode == "mse":
        return F.mse_loss(z_anchor, z_positive)
    if mode == "cosine":
        return (1.0 - F.cosine_similarity(z_anchor, z_positive, dim=1)).mean()
    raise ValueError(f"Unsupported consistency mode: {mode}")
