#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch


def bit_balance_loss(z: torch.Tensor) -> torch.Tensor:
    # Encourage each feature dimension to have batch mean near zero before thresholding.
    return torch.mean(torch.square(z.mean(dim=0)))
