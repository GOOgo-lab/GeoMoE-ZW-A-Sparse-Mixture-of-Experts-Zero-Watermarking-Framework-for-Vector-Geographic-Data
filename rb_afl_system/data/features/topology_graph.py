#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight topology graph features saved as NPZ."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
try:
    import geopandas as gpd
except Exception as exc:
    raise RuntimeError("Topology graph builder requires geopandas") from exc
from rb_afl_system.data.features.vector_tokens import build_vector_tokens, TokenConfig

@dataclass
class GraphConfig:
    max_nodes: int = 256
    distance_quantile: float = 0.15
    def to_dict(self) -> dict:
        return asdict(self)

def build_topology_graph(gdf: gpd.GeoDataFrame, config: GraphConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cfg = config or GraphConfig()
    tokens, _token_mask, token_meta = build_vector_tokens(gdf, TokenConfig(max_tokens=cfg.max_nodes))
    nodes = tokens[:cfg.max_nodes].astype(np.float32)
    n = int(nodes.shape[0])
    if n == 0:
        nodes = np.zeros((1, tokens.shape[1]), dtype=np.float32)
        n = 1
    xy = nodes[:, 3:5]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
    nonzero = dist[dist > 0]
    if nonzero.size == 0:
        thresh = 0.0
    else:
        thresh = float(np.quantile(nonzero, np.clip(cfg.distance_quantile, 0.01, 1.0)))
    adj = ((dist <= thresh) & (dist > 0)).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    mask = np.ones((n,), dtype=np.bool_)
    meta = {"config": cfg.to_dict(), "num_nodes": n, "node_feature_names": token_meta["feature_names"]}
    return nodes, adj, mask, meta

def save_graph(path: str, nodes: np.ndarray, adj: np.ndarray, mask: np.ndarray, meta: dict) -> None:
    np.savez_compressed(path, nodes=nodes.astype(np.float32), adj=adj.astype(np.float32), mask=mask.astype(np.bool_), meta=np.array([meta], dtype=object))

def load_graph(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["nodes"].astype(np.float32), data["adj"].astype(np.float32), data["mask"].astype(np.bool_)
