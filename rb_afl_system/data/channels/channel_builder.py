#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build four-channel geometric tensors from vector geodata."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

try:
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from shapely.geometry import LineString, Point, Polygon
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Channel builder requires geopandas, shapely and scipy. Install with: pip install geopandas shapely scipy"
    ) from exc

from rb_afl_system.data.geometry.geometry_utils import (
    collect_coords_from_gdf,
    iter_atomic_geoms,
    make_bounds_from_gdf,
    to_grid_xy,
)
from rb_afl_system.utils import normalize01_np


CHANNEL_NAMES = ["occ", "dist", "orient", "density"]


@dataclass
class ChannelBuildConfig:
    grid_size: int = 256
    pad_ratio: float = 0.03
    density_sigma: float = 2.0
    dist_sigma: float = 1.0
    orient_mode: str = "angle01"

    def to_dict(self) -> dict:
        return asdict(self)


def rasterize_line(canvas: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], value: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0 = p0
    x1, y1 = p1
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xi = np.clip(np.round(xs).astype(np.int32), 0, canvas.shape[1] - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, canvas.shape[0] - 1)
    canvas[yi, xi] = np.maximum(canvas[yi, xi], value)
    return xi, yi


def _line_coords_from_polygon(poly: Polygon) -> List[List[Tuple[float, float]]]:
    rings: List[List[Tuple[float, float]]] = []
    rings.append([(float(x), float(y)) for x, y in poly.exterior.coords])
    for ring in poly.interiors:
        rings.append([(float(x), float(y)) for x, y in ring.coords])
    return rings


def build_four_channels(gdf: gpd.GeoDataFrame, config: ChannelBuildConfig | None = None) -> Tuple[np.ndarray, dict]:
    cfg = config or ChannelBuildConfig()
    if cfg.grid_size <= 8:
        raise ValueError(f"grid_size must be > 8, got {cfg.grid_size}")
    coords = collect_coords_from_gdf(gdf)
    if not coords:
        tensor = np.zeros((4, cfg.grid_size, cfg.grid_size), dtype=np.float32)
        return tensor, {"channel_names": CHANNEL_NAMES, "bounds": [0.0, 0.0, 1.0, 1.0], "config": cfg.to_dict()}

    bounds = make_bounds_from_gdf(gdf, pad_ratio=cfg.pad_ratio)
    h = w = cfg.grid_size
    occ = np.zeros((h, w), dtype=np.float32)
    density = np.zeros((h, w), dtype=np.float32)
    orient_x = np.zeros((h, w), dtype=np.float32)
    orient_y = np.zeros((h, w), dtype=np.float32)
    orient_count = np.zeros((h, w), dtype=np.float32)

    for geom in gdf.geometry:
        for atom in iter_atomic_geoms(geom):
            if isinstance(atom, Point):
                px, py = to_grid_xy(atom.x, atom.y, bounds, cfg.grid_size)
                occ[py, px] = 1.0
                density[py, px] += 1.0
                continue
            if isinstance(atom, LineString):
                seqs = [[(float(x), float(y)) for x, y in atom.coords]]
            elif isinstance(atom, Polygon):
                seqs = _line_coords_from_polygon(atom)
            else:
                continue
            for coords_seq in seqs:
                if len(coords_seq) < 2:
                    continue
                for i in range(len(coords_seq) - 1):
                    x0, y0 = coords_seq[i]
                    x1, y1 = coords_seq[i + 1]
                    p0 = to_grid_xy(x0, y0, bounds, cfg.grid_size)
                    p1 = to_grid_xy(x1, y1, bounds, cfg.grid_size)
                    xi, yi = rasterize_line(occ, p0, p1, 1.0)
                    rasterize_line(density, p0, p1, 1.0)
                    dx = float(x1 - x0)
                    dy = float(y1 - y0)
                    norm = float(np.hypot(dx, dy))
                    if norm > 1e-12:
                        vx = dx / norm
                        vy = dy / norm
                        orient_x[yi, xi] += vx
                        orient_y[yi, xi] += vy
                        orient_count[yi, xi] += 1.0

    occ = normalize01_np(occ)
    dist_raw = distance_transform_edt(1.0 - occ).astype(np.float32)
    dist = 1.0 - normalize01_np(dist_raw)
    if cfg.dist_sigma > 0:
        dist = gaussian_filter(dist, sigma=cfg.dist_sigma).astype(np.float32)
    dist = normalize01_np(dist)

    m = orient_count > 0
    ox = np.zeros_like(orient_x)
    oy = np.zeros_like(orient_y)
    ox[m] = orient_x[m] / np.maximum(orient_count[m], 1e-12)
    oy[m] = orient_y[m] / np.maximum(orient_count[m], 1e-12)
    if cfg.orient_mode == "angle01":
        angle = np.arctan2(oy, ox)
        orient = ((angle + np.pi) / (2.0 * np.pi)).astype(np.float32)
        orient[~m] = 0.0
    elif cfg.orient_mode == "magnitude":
        orient = np.sqrt(ox * ox + oy * oy).astype(np.float32)
    else:
        raise ValueError(f"Unsupported orient_mode: {cfg.orient_mode}")
    orient = normalize01_np(orient)

    if cfg.density_sigma > 0:
        density = gaussian_filter(density, sigma=cfg.density_sigma).astype(np.float32)
    density = normalize01_np(density)

    tensor = np.stack([occ, dist, orient, density], axis=0).astype(np.float32)
    meta = {"channel_names": CHANNEL_NAMES, "bounds": list(map(float, bounds)), "config": cfg.to_dict()}
    return tensor, meta


def select_channels(tensor4: np.ndarray, channels: List[str]) -> np.ndarray:
    if tensor4.ndim != 3 or tensor4.shape[0] != 4:
        raise ValueError(f"Expected tensor shape (4,H,W), got {tensor4.shape}")
    idx = []
    for name in channels:
        if name not in CHANNEL_NAMES:
            raise ValueError(f"Unsupported channel {name!r}; expected one of {CHANNEL_NAMES}")
        idx.append(CHANNEL_NAMES.index(name))
    return tensor4[idx].astype(np.float32)
