#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utility functions."""

from __future__ import annotations

import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(msg, flush=True)


def log_exc(tag: str, exc: Exception) -> None:
    print(f"[EXC] {tag}: {exc}", flush=True)
    traceback.print_exc()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception as exc:
        raise RuntimeError("Failed to set torch seed. Is torch installed correctly?") from exc


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    bad = '<>:"/\\|?*\n\r\t'
    for ch in bad:
        stem = stem.replace(ch, "_")
    stem = stem.strip().strip(".")
    if not stem:
        raise ValueError(f"Cannot make safe stem from {name!r}")
    return stem


def list_files(root: str | Path, suffix: str) -> list[Path]:
    r = Path(root)
    if not r.is_dir():
        raise NotADirectoryError(str(r))
    return sorted(p for p in r.rglob(f"*{suffix}") if p.is_file())


def normalize01_np(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return a
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if mx - mn < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - mn) / (mx - mn)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def require_env_path(name: str) -> Path:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Environment variable {name} is not set")
    p = Path(val)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p
