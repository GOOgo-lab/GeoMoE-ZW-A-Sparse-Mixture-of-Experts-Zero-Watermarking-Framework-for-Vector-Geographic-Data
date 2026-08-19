#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attack registry for internal and mapshaper attacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Attack registry requires geopandas") from exc

from rb_afl_system.data.attacks.internal_attacks import AttackSpec, apply_internal_attack
from rb_afl_system.data.attacks.mapshaper_attacks import mapshaper_simplify_gdf
from rb_afl_system.data.geometry.geometry_utils import quality_report, repair_gdf


def spec_from_dict(d: Dict[str, Any]) -> AttackSpec:
    return AttackSpec(
        attack_type=str(d["attack_type"]),
        value=float(d.get("value", 0.0)),
        params=dict(d.get("params", {})),
        engine=str(d.get("engine", "internal")),
    )


def apply_attack(gdf: gpd.GeoDataFrame, spec: AttackSpec, mapshaper_bin: str = "mapshaper", seed: int = 20260318) -> tuple[gpd.GeoDataFrame, dict]:
    before = gdf.copy()
    if spec.engine == "internal":
        attacked = apply_internal_attack(gdf, spec, seed=seed)
    elif spec.engine == "mapshaper":
        if spec.attack_type.lower() != "simplify":
            raise ValueError("Current mapshaper engine supports attack_type='simplify' only")
        method = str(spec.params.get("method", "weighted"))
        clean = bool(spec.params.get("clean", True))
        keep_shapes = bool(spec.params.get("keep_shapes", True))
        attacked = mapshaper_simplify_gdf(
            gdf,
            keep_percent=float(spec.value),
            method=method,
            clean=clean,
            keep_shapes=keep_shapes,
            binary=mapshaper_bin,
        )
        attacked = repair_gdf(attacked)
    else:
        raise ValueError(f"Unsupported attack engine: {spec.engine}")
    report = quality_report(before, attacked)
    report["attack"] = spec.to_dict()
    return attacked, report
