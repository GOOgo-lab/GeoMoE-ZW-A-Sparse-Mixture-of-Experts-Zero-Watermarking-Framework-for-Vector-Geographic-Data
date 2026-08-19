#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shapefile scanner and validation utilities.

The dataset builder uses this module before constructing samples. The goal is
not to silently trust every .shp file: we record missing sidecars, unreadable
files, empty layers, invalid geometries, CRS, bounds and geometry statistics so
that the dataset is auditable.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Shapefile scanning requires geopandas") from exc

from rb_afl_system.data.geometry.geometry_utils import collect_coords_from_gdf, geometry_stats, repair_gdf
from rb_afl_system.utils import ensure_dir, log, safe_stem


REQUIRED_SIDECARS = [".shp", ".shx", ".dbf"]
OPTIONAL_SIDECARS = [".prj", ".cpg"]


@dataclass
class ShpScanConfig:
    require_sidecars: bool = True
    repair_before_stats: bool = True
    min_features: int = 1
    allow_missing_crs: bool = True


@dataclass
class ShpScanRow:
    shp_path: str
    rel_path: str
    name: str
    parent: str
    ok: bool
    skip_reason: str
    missing_sidecars: str
    optional_missing_sidecars: str
    feature_count: int
    crs: str
    geom_types: str
    bounds: str
    empty_count: int
    invalid_count: int
    vertex_count: int
    total_area: float
    total_length: float
    error: str
    traceback_text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _sidecar_status(shp_path: Path) -> Tuple[List[str], List[str]]:
    missing_required: List[str] = []
    missing_optional: List[str] = []
    for suffix in REQUIRED_SIDECARS:
        if not shp_path.with_suffix(suffix).is_file():
            missing_required.append(shp_path.with_suffix(suffix).name)
    for suffix in OPTIONAL_SIDECARS:
        if not shp_path.with_suffix(suffix).is_file():
            missing_optional.append(shp_path.with_suffix(suffix).name)
    return missing_required, missing_optional


def _bounds_text(gdf: gpd.GeoDataFrame) -> str:
    try:
        vals = list(map(float, gdf.total_bounds))
        return ",".join(f"{v:.8f}" for v in vals)
    except Exception:
        return ""


def scan_one_shp(shp_path: Path, source_root: Path, cfg: ShpScanConfig) -> ShpScanRow:
    try:
        rel_obj = shp_path.relative_to(source_root)
    except ValueError:
        rel_obj = Path(shp_path.name)
    rel = str(rel_obj)
    missing_required, missing_optional = _sidecar_status(shp_path)
    base_kwargs = {
        "shp_path": str(shp_path),
        "rel_path": rel,
        "name": shp_path.stem,
        "parent": shp_path.parent.name,
        "ok": False,
        "skip_reason": "",
        "missing_sidecars": ";".join(missing_required),
        "optional_missing_sidecars": ";".join(missing_optional),
        "feature_count": 0,
        "crs": "",
        "geom_types": "",
        "bounds": "",
        "empty_count": 0,
        "invalid_count": 0,
        "vertex_count": 0,
        "total_area": 0.0,
        "total_length": 0.0,
        "error": "",
        "traceback_text": "",
    }
    if cfg.require_sidecars and missing_required:
        base_kwargs["skip_reason"] = "missing_required_sidecars"
        base_kwargs["error"] = f"Missing required sidecars: {', '.join(missing_required)}"
        return ShpScanRow(**base_kwargs)

    try:
        gdf = gpd.read_file(shp_path)
        if cfg.repair_before_stats:
            gdf = repair_gdf(gdf)
        feature_count = int(len(gdf))
        geom_types = sorted(map(str, gdf.geometry.geom_type.dropna().unique())) if feature_count > 0 else []
        stats = geometry_stats(gdf)
        coords = collect_coords_from_gdf(gdf)
        crs_text = str(gdf.crs) if gdf.crs is not None else ""

        skip_reason = ""
        ok = True
        if feature_count < int(cfg.min_features):
            ok = False
            skip_reason = "empty_or_too_few_features"
        elif not coords:
            ok = False
            skip_reason = "no_valid_coordinates"
        elif stats.invalid_count > 0:
            ok = False
            skip_reason = "invalid_geometry_after_repair"
        elif not crs_text and not cfg.allow_missing_crs:
            ok = False
            skip_reason = "missing_crs"

        return ShpScanRow(
            **{
                **base_kwargs,
                "ok": bool(ok),
                "skip_reason": skip_reason,
                "feature_count": feature_count,
                "crs": crs_text,
                "geom_types": ";".join(geom_types),
                "bounds": _bounds_text(gdf),
                "empty_count": int(stats.empty_count),
                "invalid_count": int(stats.invalid_count),
                "vertex_count": int(stats.vertex_count),
                "total_area": float(stats.total_area),
                "total_length": float(stats.total_length),
            }
        )
    except Exception as exc:
        return ShpScanRow(
            **{
                **base_kwargs,
                "skip_reason": "read_failed",
                "error": repr(exc),
                "traceback_text": traceback.format_exc(),
            }
        )


def find_shapefiles(source_root: str | Path) -> List[Path]:
    root = Path(source_root)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    return sorted(p for p in root.rglob("*.shp") if p.is_file())


def write_scan_reports(rows: Iterable[ShpScanRow], output_root: str | Path) -> Dict[str, str]:
    out = ensure_dir(output_root)
    row_dicts = [r.to_dict() for r in rows]
    csv_path = out / "scan_report.csv"
    json_path = out / "scan_report.json"
    pd.DataFrame(row_dicts).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(row_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"scan_report_csv": str(csv_path), "scan_report_json": str(json_path)}


def scan_shapefiles(source_root: str | Path, output_root: str | Path, cfg: ShpScanConfig) -> List[ShpScanRow]:
    root = Path(source_root)
    files = find_shapefiles(root)
    if not files:
        raise FileNotFoundError(f"No .shp files found under {root}")
    rows: List[ShpScanRow] = []
    for shp in files:
        row = scan_one_shp(shp, root, cfg)
        rows.append(row)
        if row.ok:
            log(f"[SCAN OK] {row.rel_path} features={row.feature_count} types={row.geom_types or 'unknown'}")
        else:
            log(f"[SCAN SKIP] {row.rel_path} reason={row.skip_reason} error={row.error}")
    write_scan_reports(rows, output_root)
    return rows


def make_unique_identity(rel_path: str, used: set[str]) -> str:
    p = Path(rel_path)
    stem = safe_stem(p.stem)
    parent_text = p.parent.as_posix().replace("/", "_")
    parent = safe_stem(parent_text) if parent_text not in (".", "") else ""
    candidates = []
    if parent:
        candidates.append(f"{parent}__{stem}")
    candidates.append(stem)
    for cand in candidates:
        if cand not in used:
            used.add(cand)
            return cand
    idx = 2
    while True:
        cand = f"{stem}_{idx:03d}"
        if cand not in used:
            used.add(cand)
            return cand
        idx += 1
