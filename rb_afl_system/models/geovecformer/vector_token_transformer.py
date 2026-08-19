#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vector token Transformer-style branch implemented with explicit attention.

We avoid torch.nn.TransformerEncoder here to keep CPU execution predictable across environments.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.num_heads
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = ~mask.bool()
        scores = scores.masked_fill(key_mask[:, None, None, :], -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        x = x + self.dropout(self.proj(out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        x = x * mask.float().unsqueeze(-1)
        return x


class VectorTokenTransformer(nn.Module):
    def __init__(self, in_dim: int = 12, model_dim: int = 128, out_dim: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)
        self.layers = nn.ModuleList([SelfAttentionBlock(model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(model_dim, out_dim)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(tokens) * token_mask.float().unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, token_mask)
        mask = token_mask.float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.fc(pooled)
