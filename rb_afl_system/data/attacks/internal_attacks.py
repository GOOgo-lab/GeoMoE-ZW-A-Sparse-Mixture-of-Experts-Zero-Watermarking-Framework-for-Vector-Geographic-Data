#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal geometry-level attacks implemented with shapely/geopandas.

V13 notes
---------
- Keeps the V10/V11 attack behavior for rotate / scale / translate / simplify /
  quantize / jitter / delete_features.
- Adds explicit topology and boundary attack types so capability profiling can
  distinguish W_topology and W_boundary instead of treating them as no-data
  placeholders.
- New attack aliases are conservative and dataset-builder friendly: all of them
  repair and filter empty geometries before saving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import geopandas as gpd
    from shapely import affinity
    from shapely.geometry import (
        GeometryCollection,
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )
    from shapely.geometry.base import BaseGeometry
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Internal attacks require geopandas and shapely") from exc

from rb_afl_system.data.geometry.geometry_utils import repair_gdf


@dataclass
class AttackSpec:
    attack_type: str
    value: float
    params: Dict[str, Any]
    engine: str = "internal"

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_geom_safe(gdf: gpd.GeoDataFrame, func, repair: bool = True) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.geometry = out.geometry.apply(lambda geom: func(geom) if geom is not None and not geom.is_empty else geom)
    out = out[out.geometry.notnull() & ~out.geometry.is_empty].copy()
    out.reset_index(drop=True, inplace=True)
    if repair:
        out = repair_gdf(out)
    return out


def rotate_attack(gdf: gpd.GeoDataFrame, degree_clockwise: float, repair: bool = True) -> gpd.GeoDataFrame:
    return _apply_geom_safe(
        gdf,
        lambda geom: affinity.rotate(geom, -float(degree_clockwise), origin="centroid", use_radians=False),
        repair=repair,
    )


def scale_attack(
    gdf: gpd.GeoDataFrame,
    x_factor: float,
    y_factor: Optional[float] = None,
    repair: bool = True,
) -> gpd.GeoDataFrame:
    yf = float(x_factor if y_factor is None else y_factor)
    return _apply_geom_safe(
        gdf,
        lambda geom: affinity.scale(geom, xfact=float(x_factor), yfact=yf, origin="centroid"),
        repair=repair,
    )


def translate_attack(gdf: gpd.GeoDataFrame, xoff: float, yoff: Optional[float] = None, repair: bool = True) -> gpd.GeoDataFrame:
    yf = float(xoff if yoff is None else yoff)
    return _apply_geom_safe(
        gdf,
        lambda geom: affinity.translate(geom, xoff=float(xoff), yoff=yf),
        repair=repair,
    )


def simplify_attack(gdf: gpd.GeoDataFrame, tolerance: float, preserve_topology: bool = True, repair: bool = True) -> gpd.GeoDataFrame:
    return _apply_geom_safe(
        gdf,
        lambda geom: geom.simplify(float(tolerance), preserve_topology=preserve_topology),
        repair=repair,
    )


def _apply_mapper_to_coords(coords, mapper):
    return list(mapper(coords))


def _quantize_coords(coords, step: float):
    for x, y in coords:
        qx = round(float(x) / step) * step
        qy = round(float(y) / step) * step
        yield (qx, qy)


def _jitter_coords(coords, amplitude: float, rng: np.random.Generator):
    for x, y in coords:
        yield (float(x) + float(rng.uniform(-amplitude, amplitude)), float(y) + float(rng.uniform(-amplitude, amplitude)))


def _nonempty_geoms(geoms: Iterable[BaseGeometry]) -> list[BaseGeometry]:
    return [g for g in geoms if g is not None and not g.is_empty]


def _map_coords_geom(geom, mapper):
    """Map coordinates for all atomic geometries, including GeometryCollection.

    This function intentionally preserves the high-level geometry type where
    possible. For GeometryCollection, child geometries are mapped recursively and
    rebuilt into a new GeometryCollection. This keeps V10 datasets with mixed
    polygon collections safe during coordinate attacks.
    """
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, Point):
        pts = _apply_mapper_to_coords([(geom.x, geom.y)], mapper)
        if not pts:
            return geom
        x, y = pts[0]
        return Point(x, y)
    if isinstance(geom, LineString):
        pts = _apply_mapper_to_coords(geom.coords, mapper)
        if len(pts) < 2:
            return LineString()
        return LineString(pts)
    if isinstance(geom, Polygon):
        exterior = _apply_mapper_to_coords(geom.exterior.coords, mapper)
        if len(exterior) < 4:
            return Polygon()
        interiors = []
        for ring in geom.interiors:
            mapped_ring = _apply_mapper_to_coords(ring.coords, mapper)
            if len(mapped_ring) >= 4:
                interiors.append(mapped_ring)
        return Polygon(exterior, interiors)
    if isinstance(geom, MultiPoint):
        children = _nonempty_geoms(_map_coords_geom(g, mapper) for g in geom.geoms)
        return MultiPoint(children) if children else MultiPoint([])
    if isinstance(geom, MultiLineString):
        children = _nonempty_geoms(_map_coords_geom(g, mapper) for g in geom.geoms)
        return MultiLineString(children) if children else MultiLineString([])
    if isinstance(geom, MultiPolygon):
        children = _nonempty_geoms(_map_coords_geom(g, mapper) for g in geom.geoms)
        return MultiPolygon(children) if children else MultiPolygon([])
    if isinstance(geom, GeometryCollection):
        children = _nonempty_geoms(_map_coords_geom(g, mapper) for g in geom.geoms)
        return GeometryCollection(children) if children else GeometryCollection()
    if hasattr(geom, "geoms"):
        children = _nonempty_geoms(_map_coords_geom(g, mapper) for g in geom.geoms)
        return GeometryCollection(children) if children else GeometryCollection()
    raise TypeError(f"Unsupported geometry type for coordinate mapping: {type(geom)!r}")


def quantize_attack(gdf: gpd.GeoDataFrame, step: float, repair: bool = True) -> gpd.GeoDataFrame:
    if step <= 0:
        raise ValueError(f"quantize step must be positive, got {step}")
    return _apply_geom_safe(gdf, lambda geom: _map_coords_geom(geom, lambda coords: _quantize_coords(coords, float(step))), repair=repair)


def jitter_attack(gdf: gpd.GeoDataFrame, amplitude: float, seed: int = 20260318, repair: bool = True) -> gpd.GeoDataFrame:
    if amplitude < 0:
        raise ValueError(f"jitter amplitude must be non-negative, got {amplitude}")
    rng = np.random.default_rng(int(seed))
    return _apply_geom_safe(gdf, lambda geom: _map_coords_geom(geom, lambda coords: _jitter_coords(coords, float(amplitude), rng)), repair=repair)


def delete_features_attack(gdf: gpd.GeoDataFrame, fraction: float, seed: int = 20260318) -> gpd.GeoDataFrame:
    if not (0.0 <= fraction < 1.0):
        raise ValueError(f"delete fraction must be in [0,1), got {fraction}")
    rng = np.random.default_rng(int(seed))
    n = len(gdf)
    if n == 0:
        raise ValueError("Cannot delete features from empty GeoDataFrame")
    keep = rng.random(n) >= float(fraction)
    if not np.any(keep):
        keep[rng.integers(0, n)] = True
    out = gdf.loc[keep].copy()
    out.reset_index(drop=True, inplace=True)
    return repair_gdf(out)


def _drop_components_geom(geom: BaseGeometry, fraction: float, rng: np.random.Generator) -> BaseGeometry:
    """Randomly remove children from multipart geometries while keeping one child.

    Single-part geometries are kept unchanged. Feature-level deletion is handled
    by delete_features_attack; this function focuses on multipart topology.
    """
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, (Point, LineString, Polygon)):
        return geom
    if isinstance(geom, MultiPoint):
        children = list(geom.geoms)
        kept = _choose_kept_children(children, fraction, rng)
        return MultiPoint(kept) if kept else MultiPoint([])
    if isinstance(geom, MultiLineString):
        children = list(geom.geoms)
        kept = _choose_kept_children(children, fraction, rng)
        return MultiLineString(kept) if kept else MultiLineString([])
    if isinstance(geom, MultiPolygon):
        children = list(geom.geoms)
        kept = _choose_kept_children(children, fraction, rng)
        return MultiPolygon(kept) if kept else MultiPolygon([])
    if isinstance(geom, GeometryCollection) or hasattr(geom, "geoms"):
        children = [_drop_components_geom(g, fraction, rng) for g in geom.geoms]
        kept = _choose_kept_children(_nonempty_geoms(children), fraction, rng)
        return GeometryCollection(kept) if kept else GeometryCollection()
    return geom


def _choose_kept_children(children: list[BaseGeometry], fraction: float, rng: np.random.Generator) -> list[BaseGeometry]:
    if not children:
        return []
    keep = rng.random(len(children)) >= float(fraction)
    if not np.any(keep):
        keep[int(rng.integers(0, len(children)))] = True
    return [g for g, k in zip(children, keep) if bool(k) and g is not None and not g.is_empty]


def topology_component_drop_attack(gdf: gpd.GeoDataFrame, fraction: float, seed: int = 20260318, repair: bool = True) -> gpd.GeoDataFrame:
    if not (0.0 <= fraction < 1.0):
        raise ValueError(f"topology component drop fraction must be in [0,1), got {fraction}")
    rng = np.random.default_rng(int(seed))
    return _apply_geom_safe(gdf, lambda geom: _drop_components_geom(geom, float(fraction), rng), repair=repair)


def _topology_clean_geom(geom: BaseGeometry, distance: float) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom.buffer(distance).buffer(-distance)
    if isinstance(geom, GeometryCollection) or hasattr(geom, "geoms"):
        children = _nonempty_geoms(_topology_clean_geom(g, distance) for g in geom.geoms)
        return GeometryCollection(children) if children else GeometryCollection()
    return geom


def topology_clean_attack(gdf: gpd.GeoDataFrame, distance: float = 0.0, repair: bool = True) -> gpd.GeoDataFrame:
    """Apply a light topology clean/repair pass.

    distance=0 only repairs invalid geometry. A positive distance applies a very
    small buffer-out / buffer-in clean step on polygon-like geometries.
    """
    distance = float(distance)
    if distance < 0:
        raise ValueError(f"topology clean distance must be non-negative, got {distance}")
    if distance <= 0:
        return repair_gdf(gdf)
    return _apply_geom_safe(gdf, lambda geom: _topology_clean_geom(geom, distance), repair=repair)


def boundary_jitter_attack(gdf: gpd.GeoDataFrame, amplitude: float, seed: int = 20260318, repair: bool = True) -> gpd.GeoDataFrame:
    """Explicit boundary perturbation alias for capability profiling.

    It uses the same safe coordinate mapper as jitter_attack, but keeps a distinct
    attack_type so W_boundary can be evaluated separately from generic jitter.
    """
    return jitter_attack(gdf, amplitude=amplitude, seed=seed, repair=repair)


def boundary_simplify_attack(gdf: gpd.GeoDataFrame, tolerance: float, preserve_topology: bool = True, repair: bool = True) -> gpd.GeoDataFrame:
    return simplify_attack(gdf, tolerance=tolerance, preserve_topology=preserve_topology, repair=repair)


def _boundary_smooth_geom(geom: BaseGeometry, distance: float) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom.buffer(distance).buffer(-distance)
    if isinstance(geom, (LineString, MultiLineString)):
        return geom.simplify(distance * 0.5, preserve_topology=True)
    if isinstance(geom, GeometryCollection) or hasattr(geom, "geoms"):
        children = _nonempty_geoms(_boundary_smooth_geom(g, distance) for g in geom.geoms)
        return GeometryCollection(children) if children else GeometryCollection()
    return geom


def boundary_smooth_attack(gdf: gpd.GeoDataFrame, distance: float, repair: bool = True) -> gpd.GeoDataFrame:
    """Smooth polygon boundaries through a buffer-in/out operation.

    For line and point layers the operation falls back to a mild simplify with
    half the given distance, so it remains meaningful on mixed vector datasets.
    """
    distance = float(distance)
    if distance < 0:
        raise ValueError(f"boundary smooth distance must be non-negative, got {distance}")
    if distance <= 0:
        return repair_gdf(gdf)
    return _apply_geom_safe(gdf, lambda geom: _boundary_smooth_geom(geom, distance), repair=repair)


def apply_internal_attack(gdf: gpd.GeoDataFrame, spec: AttackSpec, seed: int = 20260318) -> gpd.GeoDataFrame:
    t = spec.attack_type.lower()
    value = float(spec.value)
    params = dict(spec.params or {})
    if t == "rotate":
        return rotate_attack(gdf, degree_clockwise=value)
    if t == "uniform_scale":
        return scale_attack(gdf, x_factor=value)
    if t == "nonuniform_scale_x":
        return scale_attack(gdf, x_factor=value, y_factor=float(params.get("y_factor", 1.0)))
    if t == "translate":
        return translate_attack(gdf, xoff=value, yoff=float(params.get("yoff", value)))
    if t == "simplify":
        return simplify_attack(gdf, tolerance=value, preserve_topology=bool(params.get("preserve_topology", True)))
    if t == "quantize":
        return quantize_attack(gdf, step=value)
    if t == "jitter":
        return jitter_attack(gdf, amplitude=value, seed=int(params.get("seed", seed)))
    if t == "delete_features":
        return delete_features_attack(gdf, fraction=value, seed=int(params.get("seed", seed)))
    if t == "topology_delete_features":
        return delete_features_attack(gdf, fraction=value, seed=int(params.get("seed", seed)))
    if t == "topology_component_drop":
        return topology_component_drop_attack(gdf, fraction=value, seed=int(params.get("seed", seed)))
    if t == "topology_clean":
        return topology_clean_attack(gdf, distance=value)
    if t == "boundary_jitter":
        return boundary_jitter_attack(gdf, amplitude=value, seed=int(params.get("seed", seed)))
    if t == "boundary_simplify":
        return boundary_simplify_attack(gdf, tolerance=value, preserve_topology=bool(params.get("preserve_topology", True)))
    if t == "boundary_smooth":
        return boundary_smooth_attack(gdf, distance=value)
    raise ValueError(f"Unsupported internal attack type: {spec.attack_type}")


def default_attack_strength_levels() -> dict[str, list[float]]:
    """Return paper-grade single-attack strength settings.

    Conventions
    -----------
    - RST attacks use direct physical factors: degrees for rotation and scale
      multipliers for uniform scaling.
    - Coordinate / simplify / boundary distances use a ratio of the source
      vector bounds span when converted by dataset builders.
    - Topology deletion attacks use feature/component deletion fractions.

    Most attacks use five levels. Rotation uses nine levels to cover typical periodic angles; uniform scaling uses seven levels to cover mild, moderate, and strong scale changes.
    """
    return {
        "rotate": [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 180.0, 270.0],
        "uniform_scale": [0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 1.50],
        "translate": [0.0025, 0.005, 0.01, 0.02, 0.05],
        "simplify": [0.0005, 0.001, 0.002, 0.004, 0.008],
        "quantize": [0.0005, 0.001, 0.002, 0.004, 0.008],
        "jitter": [0.0005, 0.001, 0.002, 0.004, 0.008],
        "topology_delete_features": [0.05, 0.10, 0.15, 0.20, 0.30],
        "topology_component_drop": [0.05, 0.10, 0.20, 0.30, 0.40],
        "topology_clean": [0.00025, 0.0005, 0.001, 0.002, 0.004],
        "boundary_jitter": [0.00025, 0.0005, 0.001, 0.002, 0.004],
        "boundary_simplify": [0.0005, 0.001, 0.002, 0.004, 0.008],
        "boundary_smooth": [0.00025, 0.0005, 0.001, 0.002, 0.004],
    }


def make_strength_sweep_attack_specs(
    value_mode: str = "span_ratio",
    include_level_meta: bool = True,
) -> list[AttackSpec]:
    """Build multi-level attack specs for dataset config generation.

    The returned values are unresolved ratios for span-based attacks. Dataset
    builders resolve them to absolute coordinate units per vector sample and
    preserve the original value in metadata.
    """
    levels = default_attack_strength_levels()
    span_based = {
        "translate",
        "simplify",
        "quantize",
        "jitter",
        "topology_clean",
        "boundary_jitter",
        "boundary_simplify",
        "boundary_smooth",
    }
    preserve_topology_attacks = {"simplify", "boundary_simplify"}
    specs: list[AttackSpec] = []
    for attack_type, values in levels.items():
        for level_idx, value in enumerate(values, start=1):
            params: dict[str, Any] = {}
            if attack_type in span_based:
                params["value_mode"] = value_mode
            if attack_type in preserve_topology_attacks:
                params["preserve_topology"] = True
            if include_level_meta:
                params["strength_level"] = level_idx
                params["strength_value_raw"] = value
            specs.append(AttackSpec(attack_type, float(value), params, engine="internal"))
    return specs


def default_attack_specs(bounds_span: float) -> list[AttackSpec]:
    """Return resolved five-level default attacks for direct API users.

    Dataset builders usually receive unresolved configs generated by
    make_strength_sweep_attack_specs() and resolve span ratios themselves.
    This function remains useful for direct calls that only know bounds_span.
    """
    span = max(float(bounds_span), 1e-6)
    specs: list[AttackSpec] = []
    span_based = {
        "translate",
        "simplify",
        "quantize",
        "jitter",
        "topology_clean",
        "boundary_jitter",
        "boundary_simplify",
        "boundary_smooth",
    }
    for spec in make_strength_sweep_attack_specs(value_mode="absolute"):
        params = dict(spec.params or {})
        raw_value = float(spec.value)
        if spec.attack_type in span_based:
            value = raw_value * span
            params["resolved_from"] = {
                "value": raw_value,
                "value_mode": "span_ratio",
            }
        else:
            value = raw_value
            params["resolved_from"] = {
                "value": raw_value,
                "value_mode": "absolute",
            }
        specs.append(AttackSpec(spec.attack_type, value, params, engine=spec.engine))
    return specs
