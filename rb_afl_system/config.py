#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration helpers for RB-AFL system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml") from exc
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping, got {type(data)!r}: {p}")
        return data
    raise ValueError(f"Unsupported config suffix: {suffix}. Use .json/.yaml/.yml")


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out
