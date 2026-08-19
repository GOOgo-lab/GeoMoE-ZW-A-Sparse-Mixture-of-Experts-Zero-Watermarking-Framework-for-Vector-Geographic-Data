#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vector object token feature extraction.

V11 expands the token descriptor from coarse object statistics to a richer
shape-aware vector.  This is still lightweight and deterministic, but it adds
compactness, extent, convexity, centroid/bbox relation and radial-moment style
features so token-only / GeoVecFormer models can distinguish hard-negative
administrative polygons better than with the original 12-D token.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon
except Exception as exc:
    raise RuntimeError("Vector token builder requires geopandas and shapely") from exc

from rb_afl_system.data.geometry.geometry_utils import count_vertices_geom, iter_atomic_geoms

TOKEN_FEATURE_NAMES = [
    "type_point",
    "type_line",
    "type_polygon",
    "centroid_x",
    "centroid_y",
    "bbox_w",
    "bbox_h",
    "area",
    "length",
    "vertex_count_log",
    "orientation_cos",
    "orientation_sin",
    "bbox_aspect_log",
    "extent_area_over_bbox",
    "compactness",
    "convexity",
    "centroid_radius",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_area",
    "length_sqrt_area_ratio",
    "radial_mean",
    "radial_std",
    "radial_max",
]


@dataclass
class TokenConfig:
    max_tokens: int = 512
    normalize_bounds: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _type_onehot(geom) -> tuple[float, float, float]:
    if isinstance(geom, Point):
        return 1.0, 0.0, 0.0
    if isinstance(geom, LineString):
        return 0.0, 1.0, 0.0
    if isinstance(geom, Polygon):
        return 0.0, 0.0, 1.0
    return 0.0, 0.0, 0.0


def _orientation(geom) -> tuple[float, float]:
    coords: list[tuple[float, float]] = []
    if isinstance(geom, LineString):
        coords = [(float(x), float(y)) for x, y in geom.coords]
    elif isinstance(geom, Polygon):
        coords = [(float(x), float(y)) for x, y in geom.exterior.coords]
    if len(coords) < 2:
        return 0.0, 0.0
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    norm = float(np.hypot(dx, dy))
    if norm < 1e-12:
        return 0.0, 0.0
    return float(dx / norm), float(dy / norm)


def _exterior_coords(geom) -> np.ndarray:
    if isinstance(geom, LineString):
        pts = list(geom.coords)
    elif isinstance(geom, Polygon):
        pts = list(geom.exterior.coords)
    elif isinstance(geom, Point):
        pts = [(geom.x, geom.y)]
    else:
        pts = []
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([(float(x), float(y)) for x, y in pts], dtype=np.float32)


def _safe_float(x: float, default: float = 0.0) -> float:
    try:
        val = float(x)
        if not np.isfinite(val):
            return float(default)
        return val
    except Exception:
        return float(default)


def build_vector_tokens(gdf: gpd.GeoDataFrame, config: TokenConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = config or TokenConfig()
    rows: List[List[float]] = []
    minx, miny, maxx, maxy = gdf.total_bounds if len(gdf) > 0 else (0.0, 0.0, 1.0, 1.0)
    span_x = max(float(maxx - minx), 1e-12)
    span_y = max(float(maxy - miny), 1e-12)
    span_diag = max(float(np.hypot(span_x, span_y)), 1e-12)

    for geom in gdf.geometry:
        for atom in iter_atomic_geoms(geom):
            if atom is None or atom.is_empty:
                continue
            t0, t1, t2 = _type_onehot(atom)
            c = atom.centroid
            bx0, by0, bx1, by1 = atom.bounds
            raw_bw = max(float(bx1 - bx0), 1e-12)
            raw_bh = max(float(by1 - by0), 1e-12)
            raw_area = max(float(atom.area), 0.0)
            raw_len = max(float(atom.length), 0.0)
            bbox_area_raw = max(raw_bw * raw_bh, 1e-12)

            cx = (float(c.x) - minx) / span_x if cfg.normalize_bounds else float(c.x)
            cy = (float(c.y) - miny) / span_y if cfg.normalize_bounds else float(c.y)
            bw = raw_bw / span_x if cfg.normalize_bounds else raw_bw
            bh = raw_bh / span_y if cfg.normalize_bounds else raw_bh
            area = raw_area / max(span_x * span_y, 1e-12) if cfg.normalize_bounds else raw_area
            length = raw_len / max(span_x + span_y, 1e-12) if cfg.normalize_bounds else raw_len
            bbox_center_x = ((float(bx0 + bx1) * 0.5) - minx) / span_x if cfg.normalize_bounds else float(bx0 + bx1) * 0.5
            bbox_center_y = ((float(by0 + by1) * 0.5) - miny) / span_y if cfg.normalize_bounds else float(by0 + by1) * 0.5
            bbox_area = bbox_area_raw / max(span_x * span_y, 1e-12) if cfg.normalize_bounds else bbox_area_raw
            vcount = float(count_vertices_geom(atom))
            oc, os = _orientation(atom)

            bbox_aspect_log = float(np.log((raw_bw + 1e-12) / (raw_bh + 1e-12)))
            extent = raw_area / bbox_area_raw if bbox_area_raw > 0 else 0.0
            compactness = 4.0 * np.pi * raw_area / max(raw_len * raw_len, 1e-12) if raw_len > 0 else 0.0
            try:
                hull_area = max(float(atom.convex_hull.area), 1e-12)
                convexity = raw_area / hull_area
            except Exception:
                convexity = 0.0
            centroid_radius = np.hypot((float(c.x) - minx) / span_x - 0.5, (float(c.y) - miny) / span_y - 0.5)
            length_sqrt_area_ratio = raw_len / max(np.sqrt(raw_area), 1e-12) if raw_area > 0 else 0.0

            coords = _exterior_coords(atom)
            if coords.shape[0] > 1:
                d = np.hypot(coords[:, 0] - float(c.x), coords[:, 1] - float(c.y)) / span_diag
                radial_mean = float(np.mean(d))
                radial_std = float(np.std(d))
                radial_max = float(np.max(d))
            else:
                radial_mean = 0.0
                radial_std = 0.0
                radial_max = 0.0

            rows.append([
                t0,
                t1,
                t2,
                _safe_float(cx),
                _safe_float(cy),
                _safe_float(bw),
                _safe_float(bh),
                _safe_float(area),
                _safe_float(length),
                _safe_float(np.log1p(vcount)),
                _safe_float(oc),
                _safe_float(os),
                _safe_float(bbox_aspect_log),
                _safe_float(extent),
                _safe_float(compactness),
                _safe_float(convexity),
                _safe_float(centroid_radius),
                _safe_float(bbox_center_x),
                _safe_float(bbox_center_y),
                _safe_float(bbox_area),
                _safe_float(length_sqrt_area_ratio),
                _safe_float(radial_mean),
                _safe_float(radial_std),
                _safe_float(radial_max),
            ])

    if not rows:
        arr = np.zeros((1, len(TOKEN_FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros((1,), dtype=np.bool_)
    else:
        arr = np.asarray(rows, dtype=np.float32)
        if arr.shape[0] > cfg.max_tokens:
            score = arr[:, 7] + arr[:, 8] + arr[:, 19]
            idx = np.argsort(-score)[:cfg.max_tokens]
            arr = arr[np.sort(idx)]
        mask = np.ones((arr.shape[0],), dtype=np.bool_)
    meta = {"feature_names": TOKEN_FEATURE_NAMES, "config": cfg.to_dict(), "num_tokens": int(arr.shape[0])}
    return arr.astype(np.float32), mask, meta


def save_tokens(path: str, tokens: np.ndarray, mask: np.ndarray, meta: dict) -> None:
    np.savez_compressed(path, tokens=tokens.astype(np.float32), mask=mask.astype(np.bool_), meta=np.array([meta], dtype=object))


def load_tokens(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["tokens"].astype(np.float32), data["mask"].astype(np.bool_)
