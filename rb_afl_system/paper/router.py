"""RouterNet and paper-aligned multi-label routing objective."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterNet(nn.Module):
    """Three-layer MLP: d_in -> 128 -> 64 -> number of registered experts."""

    def __init__(self, input_dim: int = 128, num_experts: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_experts),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.net(descriptor)


def make_oracle_multilabel_targets(expert_nc: torch.Tensor, delta: float = 0.02) -> torch.Tensor:
    """Mark every expert within ``delta`` of the best NC as a positive route."""
    if expert_nc.ndim != 2 or expert_nc.shape[1] == 0:
        raise ValueError("expert_nc must have shape [batch, registered_experts]")
    best = expert_nc.max(dim=1, keepdim=True).values
    return (best - expert_nc <= float(delta)).to(expert_nc.dtype)


def router_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    expert_nc: torch.Tensor,
    quality_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-label BCE plus NC-weighted routing-quality loss (paper Eq. 14)."""
    if logits.shape != targets.shape or logits.shape != expert_nc.shape:
        raise ValueError("logits, targets and expert_nc must have identical shapes")
    classification = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    weights = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    quality = 1.0 - (weights * expert_nc.clamp(0.0, 1.0)).sum(dim=1).mean()
    total = classification + float(quality_weight) * quality
    return total, {"classification": classification.detach(), "quality": quality.detach()}


def top_k_experts(logits: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, registered_experts]")
    k = min(max(1, int(k)), int(logits.shape[1]))
    probabilities = torch.sigmoid(logits)
    return torch.topk(probabilities, k=k, dim=1)
