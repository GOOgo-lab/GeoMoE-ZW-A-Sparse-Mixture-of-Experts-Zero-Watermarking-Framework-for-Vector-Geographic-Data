#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete zero-watermark registration and recovery."""
from __future__ import annotations
import numpy as np
from rb_afl_system.watermark.metrics import nc_score, ber_score


def xor_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.uint8).reshape(-1)
    y = np.asarray(b, dtype=np.uint8).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("Cannot XOR empty bit arrays")
    if not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("XOR inputs must contain only binary values 0 and 1")
    # Preserve compatibility with the original implementation: when legacy
    # inputs differ in length, compare only their shared prefix.
    n = min(x.size, y.size)
    return np.bitwise_xor(x[:n], y[:n]).astype(np.uint8)


def make_random_copyright_bits(bit_length: int, seed: int = 20260318) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(int(bit_length),), dtype=np.uint8)


def register_zero_watermark(copyright_bits: np.ndarray, feature_bits: np.ndarray) -> np.ndarray:
    return xor_bits(copyright_bits, feature_bits)


def recover_watermark(registered_zero_watermark: np.ndarray, attacked_feature_bits: np.ndarray) -> np.ndarray:
    return xor_bits(registered_zero_watermark, attacked_feature_bits)


def evaluate_recovery(copyright_bits: np.ndarray, recovered_bits: np.ndarray) -> dict:
    return {"nc": nc_score(copyright_bits, recovered_bits), "ber": ber_score(copyright_bits, recovered_bits)}
