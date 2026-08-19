#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build identity dataset from shapefiles with base and attack samples.

V03 changes:
- Automatically scans source_root before building samples.
- Writes scan_report.csv/json without requiring a separate manual scan step.
- Skips unreadable, incomplete, empty or invalid shapefiles by default, while
  recording exact reasons and tracebacks.
- Supports span-ratio attack values so simplify/jitter/quantize/translation do
  not accidentally destroy degree-based or meter-based datasets.
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Dataset builder requires geopandas") from exc

from rb_afl_system.data.attacks.attack_registry import apply_attack, spec_from_dict
from rb_afl_system.data.attacks.internal_attacks import AttackSpec
from rb_afl_system.data.channels.channel_builder import CHANNEL_NAMES, ChannelBuildConfig, build_four_channels
from rb_afl_system.data.dataset.shp_scanner import ShpScanConfig, make_unique_identity, scan_shapefiles
from rb_afl_system.data.features.topology_graph import GraphConfig, build_topology_graph, save_graph
from rb_afl_system.data.features.vector_tokens import TokenConfig, build_vector_tokens, save_tokens
from rb_afl_system.data.geometry.geometry_utils import geometry_stats, make_bounds_from_gdf, repair_gdf
from rb_afl_system.data.io_vector import save_vector_copy
from rb_afl_system.utils import ensure_dir, log, write_json


def _copy_shapefile_family(shp_path: Path, dst_dir: Path) -> str:
    """Copy a shapefile family, not just .shp, when requested."""
    ensure_dir(dst_dir)
    copied_shp = dst_dir / shp_path.name
    for p in shp_path.parent.glob(f"{shp_path.stem}.*"):
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)
    return str(copied_shp)


def _save_sample(
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    sample_name: str,
    channel_cfg: ChannelBuildConfig,
    token_cfg: TokenConfig,
    graph_cfg: GraphConfig,
    source_path: str | None,
    attack_report: dict | None,
    save_vector: bool = True,
    vector_format: str = "gpkg",
) -> dict:
    ensure_dir(out_dir)
    tensor, channel_meta = build_four_channels(gdf, channel_cfg)
    tokens, token_mask, token_meta = build_vector_tokens(gdf, token_cfg)
    nodes, adj, graph_mask, graph_meta = build_topology_graph(gdf, graph_cfg)

    grid_path = out_dir / "grid.npy"
    token_path = out_dir / "tokens.npz"
    graph_path = out_dir / "graph.npz"
    meta_path = out_dir / "metadata.json"
    vector_path = save_vector_copy(gdf, out_dir, stem="vector", preferred=vector_format) if save_vector else None

    import numpy as np

    np.save(grid_path, tensor.astype("float32"))
    save_tokens(str(token_path), tokens, token_mask, token_meta)
    save_graph(str(graph_path), nodes, adj, graph_mask, graph_meta)
    meta = {
        "sample_name": sample_name,
        "source_path": source_path,
        "grid_path": str(grid_path),
        "tokens_path": str(token_path),
        "graph_path": str(graph_path),
        "vector_path": vector_path,
        "channel_meta": channel_meta,
        "token_meta": token_meta,
        "graph_meta": graph_meta,
        "geometry_stats": geometry_stats(gdf).to_dict(),
        "attack_report": attack_report,
    }
    write_json(meta_path, meta)
    return meta


def _bounds_span(gdf: gpd.GeoDataFrame) -> float:
    minx, miny, maxx, maxy = make_bounds_from_gdf(gdf, pad_ratio=0.0)
    return max(float(maxx - minx), float(maxy - miny), 1e-12)


def _resolve_attack_spec(spec: AttackSpec, gdf: gpd.GeoDataFrame) -> AttackSpec:
    """Resolve per-dataset relative attack values into concrete units.

    Config convention:
      {"value": 0.002, "params": {"value_mode": "span_ratio"}}
    means actual value = 0.002 * max(width, height) of this shapefile.
    """
    params = dict(spec.params or {})
    value_mode = str(params.pop("value_mode", "absolute")).lower()
    value = float(spec.value)
    if value_mode in ("span_ratio", "bounds_ratio", "relative_span"):
        value = value * _bounds_span(gdf)
    elif value_mode in ("absolute", "raw", "keep_percent"):
        value = value
    else:
        raise ValueError(f"Unsupported attack value_mode={value_mode!r} in {spec.to_dict()}")
    out = AttackSpec(attack_type=spec.attack_type, value=value, params=params, engine=spec.engine)
    out.params["resolved_from"] = {"value": spec.value, "value_mode": value_mode}
    return out


def _write_build_errors(errors: List[dict], output_root: Path) -> Dict[str, str]:
    import pandas as pd

    json_path = output_root / "build_errors.json"
    csv_path = output_root / "build_errors.csv"
    json_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(errors).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {"build_errors_json": str(json_path), "build_errors_csv": str(csv_path)}


def build_identity_dataset(config: Dict[str, Any]) -> dict:
    source_root = Path(config["source_root"])
    output_root = ensure_dir(config["output_root"])
    grid_size = int(config.get("grid_size", 256))
    channel_cfg = ChannelBuildConfig(grid_size=grid_size, pad_ratio=float(config.get("pad_ratio", 0.03)))
    token_cfg = TokenConfig(max_tokens=int(config.get("max_tokens", 512)))
    graph_cfg = GraphConfig(max_nodes=int(config.get("max_graph_nodes", 256)))
    repair = bool(config.get("repair_geometry", True))
    copy_source = bool(config.get("copy_source_shp", False))
    save_vector = bool(config.get("save_vector_copy", True))
    vector_format = str(config.get("vector_format", "gpkg"))
    attack_specs = [spec_from_dict(d) for d in config.get("attacks", [])]
    mapshaper_bin = str(config.get("mapshaper_bin", "mapshaper"))
    seed = int(config.get("seed", 20260318))
    skip_invalid_files = bool(config.get("skip_invalid_files", True))
    skip_failed_samples = bool(config.get("skip_failed_samples", True))
    skip_rejected_attacks = bool(config.get("skip_rejected_attacks", True))
    fail_if_no_valid = bool(config.get("fail_if_no_valid", True))

    scan_cfg = ShpScanConfig(
        require_sidecars=bool(config.get("require_sidecars", True)),
        repair_before_stats=bool(config.get("repair_geometry", True)),
        min_features=int(config.get("min_features", 1)),
        allow_missing_crs=bool(config.get("allow_missing_crs", True)),
    )
    scan_rows = scan_shapefiles(source_root, output_root, scan_cfg)
    valid_rows = [r for r in scan_rows if r.ok]
    if not valid_rows:
        msg = f"No valid shapefiles after scan. See {output_root / 'scan_report.csv'}"
        if fail_if_no_valid:
            raise RuntimeError(msg)
        log(f"[WARN] {msg}")
    if len(valid_rows) < len(scan_rows):
        log(f"[WARN] valid shapefiles: {len(valid_rows)}/{len(scan_rows)}. Details: {output_root / 'scan_report.csv'}")
        if not skip_invalid_files:
            raise RuntimeError("Invalid shapefiles detected and skip_invalid_files=false. See scan_report.csv")

    rows: List[dict] = []
    quality_rows: List[dict] = []
    errors: List[dict] = []
    used_identities: set[str] = set()

    for scan_row in valid_rows:
        shp_path = Path(scan_row.shp_path)
        identity = make_unique_identity(scan_row.rel_path, used_identities)
        ident_dir = output_root / identity
        base_dir = ident_dir / "base"
        log(f"[BUILD] identity={identity} shp={shp_path}")
        try:
            gdf = gpd.read_file(shp_path)
            if repair:
                gdf = repair_gdf(gdf)
            source_copy_path = None
            if copy_source:
                source_copy_path = _copy_shapefile_family(shp_path, base_dir)
            base_meta = _save_sample(
                gdf,
                base_dir,
                sample_name="base",
                channel_cfg=channel_cfg,
                token_cfg=token_cfg,
                graph_cfg=graph_cfg,
                source_path=str(shp_path if source_copy_path is None else source_copy_path),
                attack_report=None,
                save_vector=save_vector,
                vector_format=vector_format,
            )
            rows.append({
                "identity": identity,
                "sample": "base",
                "sample_dir": str(base_dir),
                "source_shp": str(shp_path),
                "source_rel_path": scan_row.rel_path,
                "attack_type": "base",
                "attack_engine": "none",
                "attack_value": "",
                "attack_name": "base",
                "attack_value_original": "",
                "attack_value_mode": "",
                "vector_path": str(base_meta.get("vector_path") or ""),
            })
        except Exception as exc:
            err = {
                "stage": "base_sample",
                "identity": identity,
                "source_shp": str(shp_path),
                "error": repr(exc),
                "traceback_text": traceback.format_exc(),
            }
            errors.append(err)
            log(f"[BUILD ERROR] base_sample identity={identity} error={exc}")
            if not skip_failed_samples:
                raise
            continue

        for i, raw_spec in enumerate(attack_specs, start=1):
            try:
                spec = _resolve_attack_spec(raw_spec, gdf)
                attack_name = f"{spec.engine}_{spec.attack_type}_{str(round(float(spec.value), 8)).replace('.', 'p')}_{i:03d}"
                attack_dir = ident_dir / "attacks" / attack_name
                attacked_gdf, report = apply_attack(gdf, spec, mapshaper_bin=mapshaper_bin, seed=seed + i)
                report["identity"] = identity
                report["attack_name"] = attack_name
                report["source_shp"] = str(shp_path)
                report["source_rel_path"] = scan_row.rel_path
                report["resolved_attack"] = spec.to_dict()
                quality_rows.append(report)
                if skip_rejected_attacks and not bool(report.get("is_accepted", False)):
                    log(f"[ATTACK SKIP] identity={identity} attack={attack_name} reason={report.get('reject_reason')}")
                    continue
                meta = _save_sample(
                    attacked_gdf,
                    attack_dir,
                    sample_name=attack_name,
                    channel_cfg=channel_cfg,
                    token_cfg=token_cfg,
                    graph_cfg=graph_cfg,
                    source_path=str(shp_path),
                    attack_report=report,
                    save_vector=save_vector,
                    vector_format=vector_format,
                )
                rows.append({
                    "identity": identity,
                    "sample": attack_name,
                    "sample_dir": str(attack_dir),
                    "source_shp": str(shp_path),
                    "source_rel_path": scan_row.rel_path,
                    "attack_type": spec.attack_type,
                    "attack_engine": spec.engine,
                    "attack_value": spec.value,
                    "attack_name": attack_name,
                    "attack_value_original": (spec.params.get("resolved_from", {}) or {}).get("value", spec.value),
                    "attack_value_mode": (spec.params.get("resolved_from", {}) or {}).get("value_mode", "absolute"),
                    "vector_path": str(meta.get("vector_path") or ""),
                })
            except Exception as exc:
                err = {
                    "stage": "attack_sample",
                    "identity": identity,
                    "source_shp": str(shp_path),
                    "attack": raw_spec.to_dict(),
                    "error": repr(exc),
                    "traceback_text": traceback.format_exc(),
                }
                errors.append(err)
                log(f"[BUILD ERROR] attack_sample identity={identity} attack={raw_spec.to_dict()} error={exc}")
                if not skip_failed_samples:
                    raise
                continue

    import pandas as pd

    metadata_csv = output_root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False, encoding="utf-8-sig")
    quality_json = output_root / "attack_quality_report.json"
    quality_csv = output_root / "attack_quality_report.csv"
    quality_json.write_text(json.dumps(quality_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(quality_rows).to_csv(quality_csv, index=False, encoding="utf-8-sig")
    error_paths = _write_build_errors(errors, output_root)
    info = {
        "format_version": "RB_AFL_DATASET_V08",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "num_shapefiles_scanned": len(scan_rows),
        "num_valid_shapefiles": len(valid_rows),
        "num_identities": len({r["identity"] for r in rows}) if rows else 0,
        "num_samples": len(rows),
        "num_attack_reports": len(quality_rows),
        "num_build_errors": len(errors),
        "grid_size": grid_size,
        "channel_names": CHANNEL_NAMES,
        "tensor_shape": [4, grid_size, grid_size],
        "scan_report_csv": str(output_root / "scan_report.csv"),
        "metadata_csv": str(metadata_csv),
        "attack_quality_report": str(quality_json),
        "attack_quality_report_csv": str(quality_csv),
        "build_errors_json": error_paths["build_errors_json"],
        "build_errors_csv": error_paths["build_errors_csv"],
        "save_vector_copy": save_vector,
        "vector_format": vector_format,
        "config": config,
    }
    write_json(output_root / "dataset_info.json", info)
    log(f"[DONE] dataset_info={output_root / 'dataset_info.json'}")
    log(f"[DONE] metadata={metadata_csv}")
    log(f"[DONE] scan_report={output_root / 'scan_report.csv'}")
    log(f"[DONE] attack_quality={quality_csv}")
    if errors:
        log(f"[WARN] build errors were recorded: {error_paths['build_errors_csv']}")
    return info
