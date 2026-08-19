#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topology graph Transformer-style branch without torch_geometric dependency."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class GraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.num_heads
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        graph_allowed = (adj > 0) | torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0)
        scores = scores.masked_fill(~graph_allowed[:, None, :, :], -1e4)
        scores = scores.masked_fill((~mask.bool())[:, None, None, :], -1e4)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(b, n, d)
        x = x + self.dropout(self.proj(out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        x = x * mask.float().unsqueeze(-1)
        return x


class GraphTransformerEncoder(nn.Module):
    def __init__(self, node_dim: int = 12, model_dim: int = 128, out_dim: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, model_dim)
        self.degree_proj = nn.Linear(1, model_dim)
        self.layers = nn.ModuleList([GraphAttentionBlock(model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(model_dim, out_dim)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1, keepdim=True)
        x = (self.node_proj(nodes) + self.degree_proj(degree)) * mask.float().unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, adj, mask)
        m = mask.float().unsqueeze(-1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.fc(pooled)
