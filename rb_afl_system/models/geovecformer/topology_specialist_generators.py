#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topology-specialist generators for V15.

The original graph branch is intentionally lightweight.  V15 adds two stronger
specialists aimed at attacks such as feature deletion, component dropping, and
geometry cleaning:

- RelationGraphTransformerGenerator: graph-only, relation-aware pooling.
- ComponentRelationTransformerGenerator: graph + vector token relation fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rb_afl_system.models.geovecformer.fusion import GatedFusion
from rb_afl_system.models.geovecformer.graph_transformer_encoder import GraphTransformerEncoder
from rb_afl_system.models.geovecformer.vector_token_transformer import VectorTokenTransformer


class _NormalizeHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), p=2, dim=1)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _masked_mean(x, mask).unsqueeze(1)
    m = mask.float().unsqueeze(-1)
    var = ((x - mean).pow(2) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return torch.sqrt(var.clamp_min(1e-8))


def _graph_relation_stats(nodes: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build compact topology statistics from padded graph tensors."""
    m = mask.float()
    node_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    degree = (adj * mask[:, None, :].float()).sum(dim=-1) * m
    degree_mean = degree.sum(dim=1, keepdim=True) / node_count
    degree_max = degree.masked_fill(~mask.bool(), 0.0).max(dim=1, keepdim=True).values
    edge_count = (adj * mask[:, :, None].float() * mask[:, None, :].float()).sum(dim=(1, 2), keepdim=True).view(nodes.shape[0], 1)
    density = edge_count / node_count.pow(2).clamp_min(1.0)

    # The vector-token graph builder stores centroid-like coordinates at indices 3:5.
    if nodes.shape[-1] >= 5:
        xy = nodes[..., 3:5]
        mean_xy = _masked_mean(xy, mask)
        xy_std = _masked_std(xy, mask)
    else:
        mean_xy = torch.zeros((nodes.shape[0], 2), device=nodes.device, dtype=nodes.dtype)
        xy_std = torch.zeros((nodes.shape[0], 2), device=nodes.device, dtype=nodes.dtype)
    return torch.cat([node_count, degree_mean, degree_max, density, mean_xy, xy_std], dim=1)


class RelationGraphTransformerGenerator(nn.Module):
    """Graph-only topology specialist with relation statistics."""

    def __init__(self, node_dim: int = 24, feat_dim: int = 256, branch_dim: int = 160, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.graph_encoder = GraphTransformerEncoder(
            node_dim=node_dim,
            model_dim=branch_dim,
            out_dim=branch_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(8, branch_dim // 2),
            nn.GELU(),
            nn.Linear(branch_dim // 2, branch_dim // 2),
        )
        self.head = _NormalizeHead(branch_dim + branch_dim // 2, feat_dim)

    def forward(self, grid=None, tokens=None, token_mask=None, graph_nodes=None, graph_adj=None, graph_mask=None) -> torch.Tensor:
        if graph_nodes is None or graph_adj is None or graph_mask is None:
            raise ValueError("RelationGraphTransformerGenerator requires graph_nodes, graph_adj and graph_mask")
        graph_feat = self.graph_encoder(graph_nodes, graph_adj, graph_mask)
        rel_feat = self.relation_mlp(_graph_relation_stats(graph_nodes, graph_adj, graph_mask))
        return self.head(torch.cat([graph_feat, rel_feat], dim=1))


class ComponentRelationTransformerGenerator(nn.Module):
    """Topology specialist using both graph relations and component tokens."""

    def __init__(self, token_dim: int = 24, node_dim: int = 24, feat_dim: int = 256, branch_dim: int = 160, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.graph_encoder = GraphTransformerEncoder(
            node_dim=node_dim,
            model_dim=branch_dim,
            out_dim=branch_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
        )
        self.token_encoder = VectorTokenTransformer(in_dim=token_dim, out_dim=branch_dim, model_dim=branch_dim, num_layers=2, num_heads=num_heads)
        self.relation_mlp = nn.Sequential(nn.Linear(8, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim))
        self.fusion = GatedFusion([branch_dim, branch_dim, branch_dim], out_dim=feat_dim)

    def forward(self, grid=None, tokens=None, token_mask=None, graph_nodes=None, graph_adj=None, graph_mask=None) -> torch.Tensor:
        if tokens is None or token_mask is None:
            raise ValueError("ComponentRelationTransformerGenerator requires tokens and token_mask")
        if graph_nodes is None or graph_adj is None or graph_mask is None:
            raise ValueError("ComponentRelationTransformerGenerator requires graph_nodes, graph_adj and graph_mask")
        graph_feat = self.graph_encoder(graph_nodes, graph_adj, graph_mask)
        token_feat = self.token_encoder(tokens, token_mask)
        rel_feat = self.relation_mlp(_graph_relation_stats(graph_nodes, graph_adj, graph_mask))
        return self.fusion([graph_feat, token_feat, rel_feat])
