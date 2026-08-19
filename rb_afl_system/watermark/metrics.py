#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized correlation (NC) and bit error rate (BER) metrics."""
from __future__ import annotations
import numpy as np


def nc_score(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a).reshape(-1)
    y = np.asarray(b).reshape(-1)
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    if not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("NC inputs must contain only binary values 0 and 1")
    # XNOR-based bitwise normalized correlation used by the paper protocol.
    # A matching bit contributes 1 and a mismatching bit contributes 0, so
    # NC = 1 - BER for every non-empty binary pair.
    return float(np.mean(x == y))


def ber_score(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.uint8).reshape(-1)
    y = np.asarray(b, dtype=np.uint8).reshape(-1)
    n = min(x.size, y.size)
    if n == 0:
        return 1.0
    return float(np.mean(x[:n] != y[:n]))
