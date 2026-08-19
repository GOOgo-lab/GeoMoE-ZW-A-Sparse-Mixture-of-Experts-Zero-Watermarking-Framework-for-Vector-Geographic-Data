#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Residual SE grid generator variants for RB-AFL V09."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, int(channels) // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x)).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEBlock(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class ResNetSEGridGenerator(nn.Module):
    """Grid-only residual SE generator selectable from config."""

    def __init__(self, in_channels: int = 4, feat_dim: int = 256, base_channels: int = 32):
        super().__init__()
        c = int(base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            ResidualSEBlock(c, c, stride=1),
            ResidualSEBlock(c, c * 2, stride=2),
            ResidualSEBlock(c * 2, c * 4, stride=2),
            ResidualSEBlock(c * 4, c * 4, stride=2),
            ResidualSEBlock(c * 4, c * 8, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c * 8, feat_dim))

    def forward(
        self,
        grid: torch.Tensor,
        tokens: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        graph_nodes: torch.Tensor | None = None,
        graph_adj: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.body(self.stem(grid))
        z = self.fc(h)
        return F.normalize(z, p=2, dim=1)
