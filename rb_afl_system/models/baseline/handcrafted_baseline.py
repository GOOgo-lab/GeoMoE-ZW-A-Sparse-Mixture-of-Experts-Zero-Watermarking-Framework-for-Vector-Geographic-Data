#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handcrafted multi-scale feature baseline."""
from __future__ import annotations
import numpy as np


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _pool(arr: np.ndarray, out_hw: int, mode: str) -> np.ndarray:
    h, w = arr.shape
    gh = max(1, h // out_hw)
    gw = max(1, w // out_hw)
    cropped = arr[: gh * out_hw, : gw * out_hw]
    blocks = cropped.reshape(out_hw, gh, out_hw, gw).transpose(0, 2, 1, 3)
    if mode == "mean":
        return blocks.mean(axis=(2, 3)).astype(np.float32)
    if mode == "max":
        return blocks.max(axis=(2, 3)).astype(np.float32)
    raise ValueError(f"Unsupported pool mode: {mode}")


def handcrafted_feature(tensor: np.ndarray, feat_dim: int = 256, seed: int = 20260318) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor (C,H,W), got {tensor.shape}")
    parts = []
    for c in range(tensor.shape[0]):
        arr = _norm01(tensor[c])
        for s in [4, 8, 16]:
            parts.append(_pool(arr, s, "mean").reshape(-1))
            parts.append(_pool(arr, s, "max").reshape(-1))
        hist, _ = np.histogram(arr.reshape(-1), bins=32, range=(0.0, 1.0))
        parts.append(_norm01(hist.astype(np.float32)))
    raw = np.concatenate([p.astype(np.float32).reshape(-1) for p in parts])
    raw = raw - raw.mean()
    std = float(raw.std())
    if std > 1e-12:
        raw = raw / std
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((raw.size, int(feat_dim))).astype(np.float32)
    z = raw @ proj
    norm = float(np.linalg.norm(z))
    if norm > 1e-12:
        z = z / norm
    return z.astype(np.float32)
