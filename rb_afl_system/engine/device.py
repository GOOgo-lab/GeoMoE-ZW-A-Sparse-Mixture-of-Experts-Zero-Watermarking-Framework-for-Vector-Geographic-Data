#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Device resolution helpers.

Supports user-facing config values such as ``auto``, ``cuda``, ``cuda:0`` and
``cpu``. ``auto`` selects CUDA when available and falls back to CPU.
"""
from __future__ import annotations

from typing import Any

import torch


def resolve_device(value: Any = None) -> torch.device:
    """Resolve a config value into a valid PyTorch device."""
    raw = "auto" if value is None else str(value).strip().lower()
    if raw in {"", "auto", "gpu", "cuda_if_available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested device='cuda', but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Config requested device={value!r}, but CUDA is not available")
        return torch.device(raw)
    if raw == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(raw)
    except Exception as exc:
        raise ValueError(f"Invalid device config {value!r}. Use 'auto', 'cuda', 'cuda:0', or 'cpu'.") from exc


def resolve_map_location(value: Any = None) -> str:
    """Resolve checkpoint map_location from the same device config."""
    return str(resolve_device(value))
