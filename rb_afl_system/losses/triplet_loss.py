#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn.functional as F


def triplet_loss(z_anchor: torch.Tensor, z_positive: torch.Tensor, z_negative: torch.Tensor, margin: float = 0.8) -> torch.Tensor:
    return F.triplet_margin_loss(z_anchor, z_positive, z_negative, margin=float(margin), p=2, reduction="mean")
