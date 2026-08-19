#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model registries for generators and discriminators."""
from __future__ import annotations

from typing import Any, Dict

from rb_afl_system.models.baseline.cnn_generator import CNNGenerator
from rb_afl_system.models.baseline.discriminator_fc import (
    DeepDiscriminatorFC,
    DiscriminatorFC,
    PairDiscriminatorFC,
    SpectralDiscriminatorFC,
)
from rb_afl_system.models.baseline.resnet_se_generator import ResNetSEGridGenerator
from rb_afl_system.models.geovecformer.branch_generators import (
    GeoGraphOnlyGenerator,
    GeoGridGraphGenerator,
    GeoGridOnlyGenerator,
    GeoGridTokenGenerator,
    GeoTokenOnlyGenerator,
)
from rb_afl_system.models.geovecformer.geovecformer_zw import GeoVecFormerZW
from rb_afl_system.models.geovecformer.topology_specialist_generators import (
    ComponentRelationTransformerGenerator,
    RelationGraphTransformerGenerator,
)


def build_generator(config: Dict[str, Any]):
    name = str(config.get("generator", "cnn_baseline"))
    feat_dim = int(config.get("feat_dim", 256))
    in_channels = int(config.get("in_channels", 4))
    base_channels = int(config.get("base_channels", 32))
    branch_dim = int(config.get("branch_dim", 128))
    token_dim = int(config.get("token_dim", 24))
    node_dim = int(config.get("node_dim", 12))

    if name == "cnn_baseline":
        return CNNGenerator(in_channels=in_channels, feat_dim=feat_dim, base_channels=base_channels)
    if name == "resnet_se_grid":
        return ResNetSEGridGenerator(in_channels=in_channels, feat_dim=feat_dim, base_channels=base_channels)
    if name == "geovecformer_zw":
        return GeoVecFormerZW(in_channels=in_channels, feat_dim=feat_dim, branch_dim=branch_dim, token_dim=token_dim, node_dim=node_dim)
    if name == "geogrid_only":
        return GeoGridOnlyGenerator(in_channels=in_channels, feat_dim=feat_dim, branch_dim=branch_dim)
    if name == "geotoken_only":
        return GeoTokenOnlyGenerator(token_dim=token_dim, feat_dim=feat_dim, branch_dim=branch_dim)
    if name == "geograph_only":
        return GeoGraphOnlyGenerator(node_dim=node_dim, feat_dim=feat_dim, branch_dim=branch_dim)
    if name == "geogrid_token":
        return GeoGridTokenGenerator(in_channels=in_channels, token_dim=token_dim, feat_dim=feat_dim, branch_dim=branch_dim)
    if name == "geogrid_graph":
        return GeoGridGraphGenerator(in_channels=in_channels, node_dim=node_dim, feat_dim=feat_dim, branch_dim=branch_dim)
    if name == "relation_graph_transformer":
        return RelationGraphTransformerGenerator(node_dim=node_dim, feat_dim=feat_dim, branch_dim=max(branch_dim, 160))
    if name == "component_relation_transformer":
        return ComponentRelationTransformerGenerator(token_dim=token_dim, node_dim=node_dim, feat_dim=feat_dim, branch_dim=max(branch_dim, 160))
    raise ValueError(f"Unsupported generator: {name}")


def build_discriminator(config: Dict[str, Any]):
    name = str(config.get("discriminator", "fc"))
    feat_dim = int(config.get("feat_dim", 256))
    hidden_dim = int(config.get("disc_hidden_dim", 128))
    depth = int(config.get("disc_depth", 2))
    dropout = float(config.get("disc_dropout", 0.1))
    if name == "none":
        return None
    if name == "fc":
        return DiscriminatorFC(feat_dim=feat_dim, hidden_dim=hidden_dim, depth=depth)
    if name == "deep_fc":
        return DeepDiscriminatorFC(feat_dim=feat_dim, hidden_dim=max(hidden_dim, 192), depth=max(depth, 4), dropout=dropout)
    if name == "spectral_fc":
        return SpectralDiscriminatorFC(feat_dim=feat_dim, hidden_dim=hidden_dim, depth=max(depth, 3), dropout=dropout)
    if name == "pair_fc":
        return PairDiscriminatorFC(feat_dim=feat_dim, hidden_dim=max(hidden_dim, 192), depth=max(depth, 3), dropout=dropout, spectral=False)
    if name == "pair_spectral_fc":
        return PairDiscriminatorFC(feat_dim=feat_dim, hidden_dim=max(hidden_dim, 192), depth=max(depth, 3), dropout=dropout, spectral=True)
    if name == "pair_deep_fc":
        return PairDiscriminatorFC(feat_dim=feat_dim, hidden_dim=max(hidden_dim, 256), depth=max(depth, 5), dropout=dropout, spectral=False)
    raise ValueError(f"Unsupported discriminator: {name}")
