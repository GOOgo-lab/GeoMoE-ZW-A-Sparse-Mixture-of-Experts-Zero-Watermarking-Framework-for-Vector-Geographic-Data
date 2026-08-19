#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np


def feature_to_bits(feature: np.ndarray, bit_length: int = 256, threshold_mode: str = "mean") -> np.ndarray:
    z = np.asarray(feature, dtype=np.float32).reshape(-1)
    if z.size == 0:
        return np.zeros((bit_length,), dtype=np.uint8)
    if z.size != bit_length:
        xp = np.linspace(0.0, 1.0, z.size)
        xq = np.linspace(0.0, 1.0, int(bit_length))
        z = np.interp(xq, xp, z).astype(np.float32)
    if threshold_mode == "mean":
        th = float(z.mean())
    elif threshold_mode == "median":
        th = float(np.median(z))
    elif threshold_mode == "zero":
        th = 0.0
    else:
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
    return (z >= th).astype(np.uint8)
