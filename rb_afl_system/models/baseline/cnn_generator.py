#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNN baseline generator for robust feature extraction."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGenerator(nn.Module):
    def __init__(self, in_channels: int = 4, feat_dim: int = 256, base_channels: int = 32):
        super().__init__()
        c = int(base_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, grid: torch.Tensor, tokens: torch.Tensor | None = None, token_mask: torch.Tensor | None = None, graph_nodes: torch.Tensor | None = None, graph_adj: torch.Tensor | None = None, graph_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(grid).flatten(1)
        z = self.fc(h)
        return F.normalize(z, p=2, dim=1)
