#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid encoder modules."""
from __future__ import annotations
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class GridEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, out_dim: int = 256, width: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 2, 1), nn.BatchNorm2d(width), nn.GELU(), SEBlock(width),
            nn.Conv2d(width, width * 2, 3, 2, 1), nn.BatchNorm2d(width * 2), nn.GELU(), SEBlock(width * 2),
            nn.Conv2d(width * 2, width * 4, 3, 2, 1), nn.BatchNorm2d(width * 4), nn.GELU(), SEBlock(width * 4),
            nn.Conv2d(width * 4, width * 4, 3, 2, 1), nn.BatchNorm2d(width * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, out_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)
