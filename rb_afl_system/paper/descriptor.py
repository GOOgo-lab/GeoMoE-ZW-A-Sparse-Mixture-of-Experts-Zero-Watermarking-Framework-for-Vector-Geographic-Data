"""Deterministic 128-dimensional paired descriptor from paper Section 2.4.1."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rb_afl_system.router.geometry_descriptor import geometry_descriptor

BASE_KEYS = tuple(
    [f"grid_c{channel}_{stat}" for channel in range(4) for stat in ("mean", "std", "min", "max", "nz_ratio")]
    + ["token_count", "token_dim", "token_mean_0", "token_std_0", "token_mean_1", "token_std_1"]
    + [
        "graph_node_count",
        "graph_node_dim",
        "graph_edge_density",
        "graph_degree_mean",
        "graph_degree_std",
        "graph_degree_max",
    ]
)
BASE_DIM = 32
DESCRIPTOR_DIM = 128


def _vector(sample_dir: str | Path) -> np.ndarray:
    values = geometry_descriptor(sample_dir, max_token_dims=8)
    vector = np.asarray([values.get(key, 0.0) for key in BASE_KEYS], dtype=np.float32)
    if vector.shape != (BASE_DIM,):
        raise RuntimeError(f"Expected {BASE_DIM} base descriptor values, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def paired_descriptor_128(
    registered_sample_dir: str | Path,
    query_sample_dir: str | Path,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Concatenate query, registered, absolute difference and safe ratio.

    The order exactly follows the manuscript: ``[q, r, |q-r|, q/(|r|+eps)]``.
    """
    q = _vector(query_sample_dir)
    r = _vector(registered_sample_dir)
    ratio = np.clip(q / np.maximum(np.abs(r), float(eps)), -1.0e3, 1.0e3)
    paired = np.concatenate([q, r, np.abs(q - r), ratio]).astype(np.float32, copy=False)
    if paired.shape != (DESCRIPTOR_DIM,):
        raise RuntimeError(f"Expected {DESCRIPTOR_DIM} paired values, got {paired.shape}")
    return paired
