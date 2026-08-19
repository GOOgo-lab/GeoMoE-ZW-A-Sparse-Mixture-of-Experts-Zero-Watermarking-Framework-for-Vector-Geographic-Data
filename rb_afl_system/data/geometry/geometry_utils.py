#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry utilities shared by dataset building and attack evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import (
        GeometryCollection,
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
        box,
    )
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import unary_union
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Geometry utilities require geopandas and shapely. Install with: pip install geopandas shapely"
    ) from exc


@dataclass
class GeometryStats:
    feature_count: int
    empty_count: int
    invalid_count: int
    point_like_count: int
    line_like_count: int
    polygon_like_count: int
    vertex_count: int
    total_area: float
    total_length: float

    def to_dict(self) -> dict:
        return asdict(self)


def iter_atomic_geoms(geom: BaseGeometry) -> Iterable[BaseGeometry]:
    """Yield atomic geometries from any shapely geometry without changing coordinates."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, (Point, LineString, Polygon)):
        yield geom
        return
    if isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        for sub in geom.geoms:
            yield from iter_atomic_geoms(sub)
        return
    try:
        # Some shapely geometry types expose .geoms but are not in the classes above.
        for sub in geom.geoms:  # type: ignore[attr-defined]
            yield from iter_atomic_geoms(sub)
    except AttributeError:
        yield geom


def collect_coords_from_geom(geom: BaseGeometry) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for atom in iter_atomic_geoms(geom):
        if isinstance(atom, Point):
            coords.append((float(atom.x), float(atom.y)))
        elif isinstance(atom, LineString):
            coords.extend((float(x), float(y)) for x, y in atom.coords)
        elif isinstance(atom, Polygon):
            coords.extend((float(x), float(y)) for x, y in atom.exterior.coords)
            for ring in atom.interiors:
                coords.extend((float(x), float(y)) for x, y in ring.coords)
        else:
            try:
                coords.extend((float(x), float(y)) for x, y in atom.coords)  # type: ignore[attr-defined]
            except Exception as exc:
                raise TypeError(f"Unsupported geometry type while collecting coords: {type(atom)!r}") from exc
    return coords


def collect_coords_from_gdf(gdf: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for geom in gdf.geometry:
        if geom is not None and not geom.is_empty:
            coords.extend(collect_coords_from_geom(geom))
    return coords


def make_bounds_from_gdf(gdf: gpd.GeoDataFrame, pad_ratio: float = 0.03) -> Tuple[float, float, float, float]:
    coords = collect_coords_from_gdf(gdf)
    if not coords:
        return 0.0, 0.0, 1.0, 1.0
    arr = np.asarray(coords, dtype=np.float64)
    minx = float(arr[:, 0].min())
    maxx = float(arr[:, 0].max())
    miny = float(arr[:, 1].min())
    maxy = float(arr[:, 1].max())
    if maxx - minx < 1e-9:
        maxx = minx + 1.0
    if maxy - miny < 1e-9:
        maxy = miny + 1.0
    padx = (maxx - minx) * float(pad_ratio)
    pady = (maxy - miny) * float(pad_ratio)
    return minx - padx, miny - pady, maxx + padx, maxy + pady


def to_grid_xy(x: float, y: float, bounds: Tuple[float, float, float, float], grid_size: int) -> Tuple[int, int]:
    minx, miny, maxx, maxy = bounds
    gx = (float(x) - minx) / max(1e-12, maxx - minx)
    gy = (float(y) - miny) / max(1e-12, maxy - miny)
    px = int(np.clip(round(gx * (grid_size - 1)), 0, grid_size - 1))
    py = int(np.clip(round((1.0 - gy) * (grid_size - 1)), 0, grid_size - 1))
    return px, py


def count_vertices_geom(geom: BaseGeometry) -> int:
    return len(collect_coords_from_geom(geom)) if geom is not None and not geom.is_empty else 0


def geometry_stats(gdf: gpd.GeoDataFrame) -> GeometryStats:
    feature_count = int(len(gdf))
    empty_count = 0
    invalid_count = 0
    point_count = 0
    line_count = 0
    polygon_count = 0
    vertex_count = 0
    total_area = 0.0
    total_length = 0.0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            empty_count += 1
            continue
        if not geom.is_valid:
            invalid_count += 1
        total_area += float(getattr(geom, "area", 0.0))
        total_length += float(getattr(geom, "length", 0.0))
        vertex_count += count_vertices_geom(geom)
        for atom in iter_atomic_geoms(geom):
            if isinstance(atom, Point):
                point_count += 1
            elif isinstance(atom, LineString):
                line_count += 1
            elif isinstance(atom, Polygon):
                polygon_count += 1
    return GeometryStats(
        feature_count=feature_count,
        empty_count=empty_count,
        invalid_count=invalid_count,
        point_like_count=point_count,
        line_like_count=line_count,
        polygon_like_count=polygon_count,
        vertex_count=vertex_count,
        total_area=total_area,
        total_length=total_length,
    )


def repair_geometry(geom: BaseGeometry, method: str = "make_valid") -> BaseGeometry:
    if geom is None:
        return geom
    if geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    if method == "make_valid":
        try:
            from shapely.validation import make_valid
            fixed = make_valid(geom)
        except Exception:
            fixed = geom.buffer(0)
    elif method == "buffer0":
        fixed = geom.buffer(0)
    else:
        raise ValueError(f"Unsupported repair method: {method}")
    if fixed is None:
        raise RuntimeError("Geometry repair returned None")
    return fixed


def repair_gdf(gdf: gpd.GeoDataFrame, method: str = "make_valid") -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.geometry = out.geometry.apply(lambda geom: repair_geometry(geom, method=method))
    out = out[~out.geometry.is_empty & out.geometry.notnull()].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def quality_report(before: gpd.GeoDataFrame, after: gpd.GeoDataFrame) -> dict:
    b = geometry_stats(before)
    a = geometry_stats(after)
    area_ratio = a.total_area / max(1e-12, b.total_area)
    length_ratio = a.total_length / max(1e-12, b.total_length)
    vertex_ratio = a.vertex_count / max(1, b.vertex_count)
    accepted = a.feature_count > 0 and a.empty_count == 0 and a.invalid_count == 0
    reason = "ok" if accepted else "invalid_or_empty_geometry"
    return {
        "before": b.to_dict(),
        "after": a.to_dict(),
        "area_ratio": float(area_ratio),
        "length_ratio": float(length_ratio),
        "vertex_ratio": float(vertex_ratio),
        "is_accepted": bool(accepted),
        "reject_reason": reason,
    }


def clip_to_bounds(gdf: gpd.GeoDataFrame, bounds: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    mask = box(minx, miny, maxx, maxy)
    out = gdf.copy()
    out.geometry = out.geometry.intersection(mask)
    out = out[~out.geometry.is_empty & out.geometry.notnull()].copy()
    out.reset_index(drop=True, inplace=True)
    return out
