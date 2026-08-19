#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Branch-specific and partial-fusion GeoVecFormer generator variants."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rb_afl_system.models.geovecformer.grid_encoder import GridEncoder
from rb_afl_system.models.geovecformer.vector_token_transformer import VectorTokenTransformer
from rb_afl_system.models.geovecformer.graph_transformer_encoder import GraphTransformerEncoder
from rb_afl_system.models.geovecformer.fusion import GatedFusion


class _NormalizeHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), p=2, dim=1)


class GeoGridOnlyGenerator(nn.Module):
    def __init__(self, in_channels: int = 4, feat_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        self.grid_encoder = GridEncoder(in_channels=in_channels, out_dim=branch_dim)
        self.head = _NormalizeHead(branch_dim, feat_dim)

    def forward(self, grid: torch.Tensor, tokens=None, token_mask=None, graph_nodes=None, graph_adj=None, graph_mask=None) -> torch.Tensor:
        return self.head(self.grid_encoder(grid))


class GeoTokenOnlyGenerator(nn.Module):
    def __init__(self, token_dim: int = 12, feat_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        self.token_encoder = VectorTokenTransformer(in_dim=token_dim, out_dim=branch_dim)
        self.head = _NormalizeHead(branch_dim, feat_dim)

    def forward(self, grid=None, tokens: torch.Tensor | None = None, token_mask: torch.Tensor | None = None, graph_nodes=None, graph_adj=None, graph_mask=None) -> torch.Tensor:
        if tokens is None or token_mask is None:
            raise ValueError("GeoTokenOnlyGenerator requires tokens and token_mask")
        return self.head(self.token_encoder(tokens, token_mask))


class GeoGraphOnlyGenerator(nn.Module):
    def __init__(self, node_dim: int = 12, feat_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        self.graph_encoder = GraphTransformerEncoder(node_dim=node_dim, out_dim=branch_dim)
        self.head = _NormalizeHead(branch_dim, feat_dim)

    def forward(self, grid=None, tokens=None, token_mask=None, graph_nodes: torch.Tensor | None = None, graph_adj: torch.Tensor | None = None, graph_mask: torch.Tensor | None = None) -> torch.Tensor:
        if graph_nodes is None or graph_adj is None or graph_mask is None:
            raise ValueError("GeoGraphOnlyGenerator requires graph_nodes, graph_adj and graph_mask")
        return self.head(self.graph_encoder(graph_nodes, graph_adj, graph_mask))


class GeoGridTokenGenerator(nn.Module):
    def __init__(self, in_channels: int = 4, token_dim: int = 12, feat_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        self.grid_encoder = GridEncoder(in_channels=in_channels, out_dim=branch_dim)
        self.token_encoder = VectorTokenTransformer(in_dim=token_dim, out_dim=branch_dim)
        self.fusion = GatedFusion([branch_dim, branch_dim], out_dim=feat_dim)

    def forward(self, grid: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor, graph_nodes=None, graph_adj=None, graph_mask=None) -> torch.Tensor:
        return self.fusion([self.grid_encoder(grid), self.token_encoder(tokens, token_mask)])


class GeoGridGraphGenerator(nn.Module):
    def __init__(self, in_channels: int = 4, node_dim: int = 12, feat_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        self.grid_encoder = GridEncoder(in_channels=in_channels, out_dim=branch_dim)
        self.graph_encoder = GraphTransformerEncoder(node_dim=node_dim, out_dim=branch_dim)
        self.fusion = GatedFusion([branch_dim, branch_dim], out_dim=feat_dim)

    def forward(self, grid: torch.Tensor, tokens=None, token_mask=None, graph_nodes: torch.Tensor | None = None, graph_adj: torch.Tensor | None = None, graph_mask: torch.Tensor | None = None) -> torch.Tensor:
        if graph_nodes is None or graph_adj is None or graph_mask is None:
            raise ValueError("GeoGridGraphGenerator requires graph_nodes, graph_adj and graph_mask")
        return self.fusion([self.grid_encoder(grid), self.graph_encoder(graph_nodes, graph_adj, graph_mask)])
