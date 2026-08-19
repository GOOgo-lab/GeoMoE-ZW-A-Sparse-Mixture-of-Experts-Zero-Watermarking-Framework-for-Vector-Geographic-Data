#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fusion modules."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, dims: list[int], out_dim: int = 256):
        super().__init__()
        total = sum(dims)
        self.value = nn.Sequential(nn.Linear(total, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.gate = nn.Sequential(nn.Linear(total, len(dims)), nn.Softmax(dim=-1))
        self.proj_each = nn.ModuleList([nn.Linear(d, out_dim) for d in dims])

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(xs, dim=-1)
        weights = self.gate(cat)
        parts = [proj(x) for proj, x in zip(self.proj_each, xs)]
        stacked = torch.stack(parts, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1) + self.value(cat)
        return F.normalize(fused, p=2, dim=1)
