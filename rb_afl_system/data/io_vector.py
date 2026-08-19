#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vector dataset writing helpers.

The project stores every base/attack sample with both model-ready arrays and a
real vector copy. Keeping a vector copy is important for academic auditability:
we can inspect what the attack actually did to the shapefile geometry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Vector IO helpers require geopandas") from exc


def save_vector_copy(gdf: "gpd.GeoDataFrame", out_dir: str | Path, stem: str = "vector", preferred: str = "gpkg") -> Optional[str]:
    """Save a real vector copy of *gdf* next to derived model tensors.

    Parameters
    ----------
    gdf:
        GeoDataFrame to save.
    out_dir:
        Sample directory.
    stem:
        File stem. The default writes ``vector.gpkg`` or ``vector.geojson``.
    preferred:
        ``gpkg`` or ``geojson``. If GPKG export fails, GeoJSON is attempted.

    Returns
    -------
    str | None
        Path to the saved vector file, or ``None`` if both exports fail. The
        function never silently swallows the error if both exports fail; the
        caller receives a clear RuntimeError.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preferred = preferred.lower().strip()
    attempts = []
    if preferred == "gpkg":
        attempts.append((out / f"{stem}.gpkg", "GPKG", {"layer": "features"}))
        attempts.append((out / f"{stem}.geojson", "GeoJSON", {}))
    elif preferred == "geojson":
        attempts.append((out / f"{stem}.geojson", "GeoJSON", {}))
        attempts.append((out / f"{stem}.gpkg", "GPKG", {"layer": "features"}))
    else:
        raise ValueError(f"Unsupported preferred vector format: {preferred!r}")

    errors = []
    for path, driver, kwargs in attempts:
        try:
            gdf.to_file(path, driver=driver, **kwargs)
            return str(path)
        except Exception as exc:  # pragma: no cover - depends on GDAL drivers
            errors.append(f"{driver} -> {path}: {exc}")
    raise RuntimeError("Failed to save vector copy. Tried:\n" + "\n".join(errors))
