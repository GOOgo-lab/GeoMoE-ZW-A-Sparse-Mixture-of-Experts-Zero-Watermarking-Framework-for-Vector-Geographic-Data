#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeoVecFormer-ZW: grid + vector-token + topology-graph fusion generator."""
from __future__ import annotations
import torch
import torch.nn as nn
from rb_afl_system.models.geovecformer.grid_encoder import GridEncoder
from rb_afl_system.models.geovecformer.vector_token_transformer import VectorTokenTransformer
from rb_afl_system.models.geovecformer.graph_transformer_encoder import GraphTransformerEncoder
from rb_afl_system.models.geovecformer.fusion import GatedFusion


class GeoVecFormerZW(nn.Module):
    def __init__(self, in_channels: int = 4, token_dim: int = 12, node_dim: int = 12, feat_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        self.grid_encoder = GridEncoder(in_channels=in_channels, out_dim=branch_dim)
        self.token_encoder = VectorTokenTransformer(in_dim=token_dim, out_dim=branch_dim)
        self.graph_encoder = GraphTransformerEncoder(node_dim=node_dim, out_dim=branch_dim)
        self.fusion = GatedFusion([branch_dim, branch_dim, branch_dim], out_dim=feat_dim)

    def forward(self, grid: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor, graph_nodes: torch.Tensor, graph_adj: torch.Tensor, graph_mask: torch.Tensor) -> torch.Tensor:
        z_grid = self.grid_encoder(grid)
        z_tok = self.token_encoder(tokens, token_mask)
        z_graph = self.graph_encoder(graph_nodes, graph_adj, graph_mask)
        return self.fusion([z_grid, z_tok, z_graph])
