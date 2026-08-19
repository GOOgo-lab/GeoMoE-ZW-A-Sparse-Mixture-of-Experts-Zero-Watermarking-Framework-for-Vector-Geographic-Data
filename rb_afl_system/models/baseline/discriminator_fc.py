#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature-space discriminator variants.

V11 adds pair discriminators.  The old single-feature discriminator was trained
as anchor=real vs positive=attack, while the generator simultaneously forced
anchor/positive to be identical.  That makes D collapse to random guessing
(val_d ~= 0.693).  Pair discriminators instead classify same-identity pairs
(anchor, positive) against different-identity pairs (anchor, negative), giving D
an objective aligned with uniqueness and robustness.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DiscriminatorFC(nn.Module):
    is_pair_discriminator = False

    def __init__(self, feat_dim: int = 256, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        in_dim = int(feat_dim)
        for _ in range(max(1, int(depth) - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1)


class DeepDiscriminatorFC(nn.Module):
    """Deeper MLP discriminator for stronger feature-space adversarial training."""
    is_pair_discriminator = False

    def __init__(self, feat_dim: int = 256, hidden_dim: int = 256, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(feat_dim)
        for _ in range(max(2, int(depth))):
            layers.extend([
                nn.Linear(in_dim, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1)


class SpectralDiscriminatorFC(nn.Module):
    """Spectral-normalized discriminator for more stable adversarial updates."""
    is_pair_discriminator = False

    def __init__(self, feat_dim: int = 256, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.05):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(feat_dim)
        for _ in range(max(1, int(depth) - 1)):
            layers.append(spectral_norm(nn.Linear(in_dim, int(hidden_dim))))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        layers.append(spectral_norm(nn.Linear(in_dim, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1)


class PairDiscriminatorFC(nn.Module):
    """Discriminator over feature pairs.

    Input pair representation:
    [z1, z2, |z1-z2|, z1*z2, cosine(z1,z2)].
    """
    is_pair_discriminator = True

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        spectral: bool = False,
    ):
        super().__init__()
        in_dim = int(feat_dim) * 4 + 1
        layers: list[nn.Module] = []
        linear = spectral_norm if spectral else (lambda x: x)
        for _ in range(max(1, int(depth))):
            layers.append(linear(nn.Linear(in_dim, int(hidden_dim))))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        layers.append(linear(nn.Linear(in_dim, 1)))
        self.net = nn.Sequential(*layers)

    def pair_features(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        cos = torch.sum(z1 * z2, dim=1, keepdim=True)
        return torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2, cos], dim=1)

    def forward_pair(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.net(self.pair_features(z1, z2)).view(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1)
