#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cheap geometry/topology descriptors for sparse specialist routing.

The descriptor intentionally uses only files available before running neural
experts (grid/tokens/graph/metadata).  It does not use attack labels by default,
so a router trained on these descriptors can be interpreted as a true sparse
conditional-computation module rather than an attack-label lookup table.
"""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _nan_stats(prefix: str, arr: np.ndarray, out: dict[str, float]) -> None:
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        out[f"{prefix}_mean"] = 0.0
        out[f"{prefix}_std"] = 0.0
        out[f"{prefix}_min"] = 0.0
        out[f"{prefix}_max"] = 0.0
        out[f"{prefix}_nz_ratio"] = 0.0
        return
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    out[f"{prefix}_mean"] = float(a.mean())
    out[f"{prefix}_std"] = float(a.std())
    out[f"{prefix}_min"] = float(a.min())
    out[f"{prefix}_max"] = float(a.max())
    out[f"{prefix}_nz_ratio"] = float((np.abs(a) > 1e-8).mean())


def _load_npz_key(path: Path, preferred: tuple[str, ...]) -> np.ndarray | None:
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    for key in preferred:
        if key in data.files:
            return np.asarray(data[key])
    if data.files:
        return np.asarray(data[data.files[0]])
    return None


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] failed to parse metadata JSON: {path}: {exc}", flush=True)
        traceback.print_exc()
        return {}


def geometry_descriptor(sample_dir: str | Path, max_token_dims: int = 8) -> dict[str, float]:
    """Return cheap descriptor features for a dataset sample directory.

    Parameters
    ----------
    sample_dir:
        Directory containing ``grid.npy``, optional ``tokens.npz``, optional
        ``graph.npz`` and optional ``metadata.json``.
    max_token_dims:
        Number of token dimensions for which mean/std summaries are emitted.
    """
    root = Path(sample_dir)
    out: dict[str, float] = {}

    # Grid descriptors: channel-level statistics of the 4-channel raster.
    grid_path = root / "grid.npy"
    if grid_path.is_file():
        grid = np.load(grid_path, allow_pickle=False)
        grid = np.asarray(grid, dtype=np.float32)
        out["grid_ndim"] = float(grid.ndim)
        if grid.ndim == 3:
            out["grid_channels"] = float(grid.shape[0])
            out["grid_height"] = float(grid.shape[1])
            out["grid_width"] = float(grid.shape[2])
            for ci in range(min(int(grid.shape[0]), 8)):
                _nan_stats(f"grid_c{ci}", grid[ci], out)
        else:
            out["grid_channels"] = 0.0
            _nan_stats("grid_all", grid, out)
    else:
        out["grid_ndim"] = 0.0
        out["grid_channels"] = 0.0

    # Token descriptors: object/component statistics.
    tokens = _load_npz_key(root / "tokens.npz", ("tokens", "x", "arr_0"))
    token_mask = _load_npz_key(root / "tokens.npz", ("mask", "token_mask"))
    if tokens is not None:
        t = np.asarray(tokens, dtype=np.float32)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        if token_mask is not None and len(np.asarray(token_mask).reshape(-1)) == t.shape[0]:
            mask = np.asarray(token_mask).reshape(-1).astype(bool)
            valid = t[mask] if mask.any() else t[:0]
        else:
            valid = t
        valid = np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        out["token_count"] = float(valid.shape[0])
        out["token_dim"] = float(valid.shape[1] if valid.ndim == 2 else 0)
        if valid.size:
            means = valid.mean(axis=0)
            stds = valid.std(axis=0)
            for i in range(min(max_token_dims, valid.shape[1])):
                out[f"token_mean_{i}"] = float(means[i])
                out[f"token_std_{i}"] = float(stds[i])
        else:
            for i in range(max_token_dims):
                out[f"token_mean_{i}"] = 0.0
                out[f"token_std_{i}"] = 0.0
    else:
        out["token_count"] = 0.0
        out["token_dim"] = 0.0
        for i in range(max_token_dims):
            out[f"token_mean_{i}"] = 0.0
            out[f"token_std_{i}"] = 0.0

    # Graph descriptors: degree/density summaries.
    nodes = _load_npz_key(root / "graph.npz", ("nodes", "x", "arr_0"))
    adj = _load_npz_key(root / "graph.npz", ("adj", "edge", "edges", "arr_1"))
    mask_arr = _load_npz_key(root / "graph.npz", ("mask", "node_mask", "arr_2"))
    if nodes is not None:
        n = np.asarray(nodes, dtype=np.float32)
        if n.ndim == 1:
            n = n.reshape(1, -1)
        if mask_arr is not None and len(np.asarray(mask_arr).reshape(-1)) == n.shape[0]:
            m = np.asarray(mask_arr).reshape(-1).astype(bool)
            node_count = int(m.sum()) if m.any() else int(n.shape[0])
        else:
            node_count = int(n.shape[0])
        out["graph_node_count"] = float(node_count)
        out["graph_node_dim"] = float(n.shape[1] if n.ndim == 2 else 0)
    else:
        node_count = 0
        out["graph_node_count"] = 0.0
        out["graph_node_dim"] = 0.0

    if adj is not None:
        a = np.asarray(adj, dtype=np.float32)
        if a.ndim == 2 and a.size:
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            deg = (a > 0).sum(axis=1).astype(np.float32)
            out["graph_edge_density"] = float((a > 0).mean())
            out["graph_degree_mean"] = float(deg.mean())
            out["graph_degree_std"] = float(deg.std())
            out["graph_degree_max"] = float(deg.max())
        else:
            out["graph_edge_density"] = 0.0
            out["graph_degree_mean"] = 0.0
            out["graph_degree_std"] = 0.0
            out["graph_degree_max"] = 0.0
    else:
        out["graph_edge_density"] = 0.0
        out["graph_degree_mean"] = 0.0
        out["graph_degree_std"] = 0.0
        out["graph_degree_max"] = 0.0

    meta = _read_metadata(root / "metadata.json")
    for key in [
        "num_features",
        "num_geometries",
        "num_vertices",
        "area",
        "length",
        "bbox_width",
        "bbox_height",
    ]:
        if key in meta:
            out[f"meta_{key}"] = _safe_float(meta.get(key))

    # Derived ratios with safe division.
    bbox_w = out.get("meta_bbox_width", 0.0)
    bbox_h = out.get("meta_bbox_height", 0.0)
    out["meta_bbox_aspect"] = float(bbox_w / max(bbox_h, 1e-8)) if bbox_w or bbox_h else 0.0
    out["token_count_log1p"] = float(np.log1p(max(out.get("token_count", 0.0), 0.0)))
    out["graph_node_count_log1p"] = float(np.log1p(max(out.get("graph_node_count", 0.0), 0.0)))
    return out


def find_reference_sample_dir(sample_dir: str | Path) -> Path | None:
    """Best-effort lookup of the registered/base sample directory for a query sample.

    The V14/V15 dataset layout is usually::

        dataset_root/<identity>/base
        dataset_root/<identity>/attacks/<attack_sample>

    Some older builds used slightly different base names.  This helper is
    deliberately conservative and never raises; a router can still fall back to
    query-only descriptors if no reference sample is found.
    """
    sample = Path(sample_dir)
    candidates: list[Path] = []

    # Typical attacked sample path: <identity>/attacks/<sample>
    if sample.parent.name == "attacks":
        identity_root = sample.parent.parent
        candidates.extend(
            [
                identity_root / "base",
                identity_root / "original",
                identity_root / "source",
                identity_root / "clean",
                identity_root / "base_000",
                identity_root / "base_0000",
            ]
        )
        # Also allow a direct identity directory if it itself contains grid.npy.
        candidates.append(identity_root)
    else:
        # Already base-like or an unknown layout.
        candidates.extend([sample, sample.parent / "base", sample.parent / "original"])

    for cand in candidates:
        if cand.is_dir() and (cand / "grid.npy").is_file():
            return cand

    # Fallback: search one level below identity root for any non-attacked sample.
    try:
        identity_root = sample.parent.parent if sample.parent.name == "attacks" else sample.parent
        if identity_root.is_dir():
            for child in sorted(identity_root.iterdir()):
                if child.name == "attacks" or not child.is_dir():
                    continue
                if (child / "grid.npy").is_file():
                    return child
    except OSError:
        return None
    return None


def paired_geometry_descriptor(
    sample_dir: str | Path,
    reference_dir: str | Path | None = None,
    max_token_dims: int = 8,
) -> dict[str, float]:
    """Return query/reference/difference descriptors for sparse MoE routing.

    This is designed for verification-time routing where the candidate identity
    is known after ``W_unique`` gate has produced a short list.  The router then
    sees both the registered/base sample descriptor and the attacked/query
    descriptor, which is much more informative than looking at the query alone.

    Output prefixes:
      - ``q__``: query/attacked descriptor
      - ``r__``: registered/base descriptor
      - ``d__``: absolute difference ``abs(q-r)``
      - ``ratio__``: safe ratio ``q/(abs(r)+eps)`` clipped to [-1e3, 1e3]
      - ``paired_ref_found``: 1 if a reference directory was found
    """
    q = geometry_descriptor(sample_dir, max_token_dims=max_token_dims)
    ref_path = Path(reference_dir) if reference_dir is not None else find_reference_sample_dir(sample_dir)
    ref_found = bool(ref_path is not None and Path(ref_path).is_dir())
    r = geometry_descriptor(ref_path, max_token_dims=max_token_dims) if ref_found and ref_path is not None else {}

    keys = sorted(set(q.keys()) | set(r.keys()))
    out: dict[str, float] = {"paired_ref_found": 1.0 if ref_found else 0.0}
    for key in keys:
        qv = _safe_float(q.get(key, 0.0))
        rv = _safe_float(r.get(key, 0.0))
        out[f"q__{key}"] = qv
        out[f"r__{key}"] = rv
        out[f"d__{key}"] = abs(qv - rv)
        denom = max(abs(rv), 1e-6)
        ratio = qv / denom
        out[f"ratio__{key}"] = float(np.clip(ratio, -1.0e3, 1.0e3))
    return out
