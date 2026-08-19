#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extend an existing RB-AFL identity dataset with additional attack samples.

This is intended for the V10-base specialist line.  It reads base vectors from an
already-built dataset metadata.csv, rebuilds base/grid/token/graph samples, then
applies the attacks listed in a config file.  This avoids needing the original
source_shp directory when the V10 dataset already saved vector copies.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("extend_dataset_attacks_V13 requires geopandas") from exc

from rb_afl_system.config import load_config
from rb_afl_system.data.attacks.attack_registry import apply_attack, spec_from_dict
from rb_afl_system.data.attacks.internal_attacks import AttackSpec
from rb_afl_system.data.channels.channel_builder import CHANNEL_NAMES, ChannelBuildConfig
from rb_afl_system.data.dataset.build_dataset import _resolve_attack_spec, _save_sample
from rb_afl_system.data.features.topology_graph import GraphConfig
from rb_afl_system.data.features.vector_tokens import TokenConfig
from rb_afl_system.data.geometry.geometry_utils import repair_gdf
from rb_afl_system.utils import ensure_dir, log, write_json


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_metadata_csv(input_dataset: Path) -> Path:
    path = input_dataset / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {path}")
    return path


def _base_vector_path(row: pd.Series) -> Path:
    candidates = []
    value = str(row.get("vector_path", "") or "").strip()
    if value:
        candidates.append(Path(value))
    sample_dir = Path(str(row.get("sample_dir", "") or ""))
    if sample_dir:
        candidates.extend([sample_dir / "vector.gpkg", sample_dir / "vector.shp", sample_dir / "vector.geojson"])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No readable base vector found for identity={row.get('identity', '')}")


def _safe_sample_name(text: str) -> str:
    return text.replace(".", "p").replace("-", "m").replace("+", "p").replace("/", "_").replace("\\", "_")


def _load_attack_specs(config: dict[str, Any]) -> list[AttackSpec]:
    specs = [spec_from_dict(d) for d in config.get("attacks", [])]
    if specs:
        return specs
    raise ValueError("attacks_config must contain a non-empty 'attacks' list")


def _config_from_input_dataset(input_dataset: Path, attack_config: dict[str, Any]) -> dict[str, Any]:
    info = _read_json_if_exists(input_dataset / "dataset_info.json")
    old_cfg = dict(info.get("config", {}) or {})
    merged = dict(old_cfg)
    merged.update(dict(attack_config or {}))
    return merged


def extend_dataset_attacks(
    input_dataset: str | Path,
    output_dataset: str | Path,
    attacks_config: str | Path,
    include_original_nonbase: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_dataset)
    output_root = ensure_dir(output_dataset)
    config_path = Path(attacks_config)
    if not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    attack_config = load_config(config_path)
    config = _config_from_input_dataset(input_root, attack_config)

    metadata_csv = _find_metadata_csv(input_root)
    old_meta = pd.read_csv(metadata_csv)
    if "identity" not in old_meta.columns or "attack_type" not in old_meta.columns:
        raise ValueError("input metadata.csv must contain identity and attack_type columns")
    base_rows = old_meta[old_meta["attack_type"].astype(str) == "base"].copy()
    if base_rows.empty:
        raise ValueError("input metadata.csv contains no base rows")

    grid_size = int(config.get("grid_size", 256))
    channel_cfg = ChannelBuildConfig(grid_size=grid_size, pad_ratio=float(config.get("pad_ratio", 0.03)))
    token_cfg = TokenConfig(max_tokens=int(config.get("max_tokens", 512)))
    graph_cfg = GraphConfig(max_nodes=int(config.get("max_graph_nodes", 256)))
    vector_format = str(config.get("vector_format", "gpkg"))
    save_vector = bool(config.get("save_vector_copy", True))
    repair = bool(config.get("repair_geometry", True))
    skip_rejected_attacks = bool(config.get("skip_rejected_attacks", True))
    mapshaper_bin = str(config.get("mapshaper_bin", "mapshaper"))
    seed = int(config.get("seed", 20260318))
    attack_specs = _load_attack_specs(config)

    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, base_row in base_rows.reset_index(drop=True).iterrows():
        identity = str(base_row["identity"])
        ident_dir = output_root / identity
        base_dir = ident_dir / "base"
        try:
            vector_path = _base_vector_path(base_row)
            log(f"[EXTEND] identity={identity} vector={vector_path}")
            gdf = gpd.read_file(vector_path)
            if repair:
                gdf = repair_gdf(gdf)
            base_meta = _save_sample(
                gdf,
                base_dir,
                sample_name="base",
                channel_cfg=channel_cfg,
                token_cfg=token_cfg,
                graph_cfg=graph_cfg,
                source_path=str(vector_path),
                attack_report=None,
                save_vector=save_vector,
                vector_format=vector_format,
            )
            rows.append({
                "identity": identity,
                "sample": "base",
                "sample_dir": str(base_dir),
                "source_shp": str(base_row.get("source_shp", "")),
                "source_rel_path": str(base_row.get("source_rel_path", "")),
                "attack_type": "base",
                "attack_engine": "none",
                "attack_value": "",
                "attack_name": "base",
                "attack_value_original": "",
                "attack_value_mode": "",
                "vector_path": str(base_meta.get("vector_path") or ""),
            })
            if include_original_nonbase:
                old_parts = old_meta[(old_meta["identity"].astype(str) == identity) & (old_meta["attack_type"].astype(str) != "base")]
                for _, old_attack in old_parts.iterrows():
                    old_vector = _base_vector_path(old_attack)
                    old_gdf = gpd.read_file(old_vector)
                    if repair:
                        old_gdf = repair_gdf(old_gdf)
                    old_name = str(old_attack.get("attack_name", old_attack.get("sample", "old_attack")))
                    old_dir = ident_dir / "attacks" / _safe_sample_name(old_name)
                    old_meta_saved = _save_sample(
                        old_gdf,
                        old_dir,
                        sample_name=old_name,
                        channel_cfg=channel_cfg,
                        token_cfg=token_cfg,
                        graph_cfg=graph_cfg,
                        source_path=str(old_vector),
                        attack_report=None,
                        save_vector=save_vector,
                        vector_format=vector_format,
                    )
                    rows.append({
                        "identity": identity,
                        "sample": old_name,
                        "sample_dir": str(old_dir),
                        "source_shp": str(old_attack.get("source_shp", "")),
                        "source_rel_path": str(old_attack.get("source_rel_path", "")),
                        "attack_type": str(old_attack.get("attack_type", "unknown")),
                        "attack_engine": str(old_attack.get("attack_engine", "unknown")),
                        "attack_value": str(old_attack.get("attack_value", "")),
                        "attack_name": old_name,
                        "attack_value_original": str(old_attack.get("attack_value_original", "")),
                        "attack_value_mode": str(old_attack.get("attack_value_mode", "")),
                        "vector_path": str(old_meta_saved.get("vector_path") or ""),
                    })
        except Exception as exc:
            err = {
                "stage": "base_or_original_copy",
                "identity": identity,
                "error": repr(exc),
                "traceback_text": traceback.format_exc(),
            }
            errors.append(err)
            log(f"[EXTEND ERROR] identity={identity} base stage error={exc}")
            continue

        for attack_idx, raw_spec in enumerate(attack_specs, start=1):
            try:
                spec = _resolve_attack_spec(raw_spec, gdf)
                attack_name = f"{spec.engine}_{spec.attack_type}_{_safe_sample_name(str(round(float(spec.value), 8)))}_{attack_idx:03d}"
                attack_dir = ident_dir / "attacks" / attack_name
                attacked_gdf, report = apply_attack(gdf, spec, mapshaper_bin=mapshaper_bin, seed=seed + attack_idx + idx * 1000)
                report["identity"] = identity
                report["attack_name"] = attack_name
                report["source_vector"] = str(vector_path)
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
                    source_path=str(vector_path),
                    attack_report=report,
                    save_vector=save_vector,
                    vector_format=vector_format,
                )
                rows.append({
                    "identity": identity,
                    "sample": attack_name,
                    "sample_dir": str(attack_dir),
                    "source_shp": str(base_row.get("source_shp", "")),
                    "source_rel_path": str(base_row.get("source_rel_path", "")),
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
                    "stage": "new_attack_sample",
                    "identity": identity,
                    "attack": raw_spec.to_dict(),
                    "error": repr(exc),
                    "traceback_text": traceback.format_exc(),
                }
                errors.append(err)
                log(f"[EXTEND ERROR] identity={identity} attack={raw_spec.to_dict()} error={exc}")
                continue

    out_meta = output_root / "metadata.csv"
    pd.DataFrame(rows).to_csv(out_meta, index=False, encoding="utf-8-sig")
    quality_csv = output_root / "attack_quality_report.csv"
    quality_json = output_root / "attack_quality_report.json"
    pd.DataFrame(quality_rows).to_csv(quality_csv, index=False, encoding="utf-8-sig")
    quality_json.write_text(json.dumps(quality_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_csv = output_root / "build_errors.csv"
    errors_json = output_root / "build_errors.json"
    pd.DataFrame(errors).to_csv(errors_csv, index=False, encoding="utf-8-sig")
    errors_json.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "format_version": "RB_AFL_DATASET_V13_EXTENDED_FROM_EXISTING",
        "input_dataset": str(input_root),
        "output_root": str(output_root),
        "num_input_base_identities": int(len(base_rows)),
        "num_samples": int(len(rows)),
        "num_attack_reports": int(len(quality_rows)),
        "num_errors": int(len(errors)),
        "grid_size": grid_size,
        "channel_names": CHANNEL_NAMES,
        "tensor_shape": [4, grid_size, grid_size],
        "metadata_csv": str(out_meta),
        "attack_quality_report_csv": str(quality_csv),
        "attack_quality_report": str(quality_json),
        "build_errors_csv": str(errors_csv),
        "build_errors_json": str(errors_json),
        "attacks_config": str(config_path),
        "config": config,
    }
    write_json(output_root / "dataset_info.json", info)
    log(f"[DONE] extended dataset_info={output_root / 'dataset_info.json'}")
    return info


def _bool_arg(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend an existing RB-AFL dataset with V13 topology/boundary attacks")
    ap.add_argument("--input_dataset", required=True, help="Existing V10 dataset root containing metadata.csv")
    ap.add_argument("--output_dataset", required=True, help="New dataset root to write")
    ap.add_argument("--attacks_config", required=True, help="JSON config containing attacks/grid/token settings")
    ap.add_argument("--include_original_nonbase", type=_bool_arg, default=False, help="Also copy old non-base attack samples when vector_path exists")
    ns = ap.parse_args()

    try:
        summary = extend_dataset_attacks(
            input_dataset=ns.input_dataset,
            output_dataset=ns.output_dataset,
            attacks_config=ns.attacks_config,
            include_original_nonbase=bool(ns.include_original_nonbase),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] extend_dataset_attacks_V13: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
