#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a multi-level single-attack strength-sweep config.

This script writes a dataset-extension config whose attacks cover 12 single
attack types with paper-grade strength levels.  Span-ratio attacks keep their raw
ratio in the config; the existing dataset builder resolves them per sample.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from rb_afl_system.data.attacks.internal_attacks import make_strength_sweep_attack_specs


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def generate_attack_strength_sweep_config(
    output_config: str | Path,
    base_config: str | Path | None = None,
    grid_size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate and save a multi-level single-attack strength-sweep config."""
    cfg = _load_json_if_exists(base_config)
    if grid_size is not None:
        cfg["grid_size"] = int(grid_size)
    if seed is not None:
        cfg["seed"] = int(seed)
    cfg["attacks"] = [spec.to_dict() for spec in make_strength_sweep_attack_specs()]
    from rb_afl_system.data.attacks.internal_attacks import default_attack_strength_levels

    level_map = {k: len(v) for k, v in default_attack_strength_levels().items()}
    cfg["attack_strength_sweep"] = {
        "num_attack_types": len(level_map),
        "levels_per_attack": level_map,
        "num_attack_specs": len(cfg["attacks"]),
        "notes": (
            "Rotation uses 9 levels and uniform scaling uses 7 levels; "
            "other attacks use 5 levels. For span_ratio attacks, value is "
            "multiplied by each vector sample's bounds span during dataset construction."
        ),
    }

    out = Path(output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_config": str(out),
        "num_attack_specs": len(cfg["attacks"]),
        "attack_strength_sweep": cfg["attack_strength_sweep"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate multi-level attack strength-sweep config")
    ap.add_argument("--output_config", required=True)
    ap.add_argument("--base_config", default="")
    ap.add_argument("--grid_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ns = ap.parse_args()

    try:
        summary = generate_attack_strength_sweep_config(
            output_config=ns.output_config,
            base_config=ns.base_config or None,
            grid_size=ns.grid_size,
            seed=ns.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] make_attack_strength_sweep_config_V19: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
