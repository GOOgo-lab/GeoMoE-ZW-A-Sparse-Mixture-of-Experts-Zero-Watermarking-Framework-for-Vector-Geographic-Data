#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:
    raise RuntimeError("geopandas is required") from exc

from rb_afl_system.data.attacks.attack_registry import apply_attack, spec_from_dict
from rb_afl_system.data.channels.channel_builder import ChannelBuildConfig, CHANNEL_NAMES
from rb_afl_system.data.dataset.build_dataset import _save_sample, _resolve_attack_spec
from rb_afl_system.data.features.topology_graph import GraphConfig
from rb_afl_system.data.features.vector_tokens import TokenConfig
from rb_afl_system.data.geometry.geometry_utils import repair_gdf
from rb_afl_system.utils import ensure_dir, write_json


def _bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower().strip()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(v)


def _copy_file_hardlink_fallback(a: str, b: str) -> None:
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def _copy_or_link(src: str, dst: str) -> None:
    src_p = Path(src)
    dst_p = Path(dst)
    if dst_p.exists():
        raise FileExistsError(f"output already exists: {dst_p}")
    shutil.copytree(src_p, dst_p, copy_function=_copy_file_hardlink_fallback)


def _rewrite_prefix(x: Any, old_root: Path, new_root: Path) -> Any:
    if not isinstance(x, str):
        return x
    old = str(old_root)
    new = str(new_root)
    return new + x[len(old):] if x.startswith(old) else x


def _safe_name(x: Any) -> str:
    s = str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = s.replace(".", "p").replace(":", "").replace("|", "_")
    return s


def _find_source_shp(row: dict, raw_root: Path) -> str:
    src = str(row.get("source_shp", "") or "")
    if src and Path(src).is_file():
        return src
    ident = str(row.get("identity", ""))
    exact = list(raw_root.rglob(f"{ident}.shp"))
    if exact:
        return str(exact[0])
    stem_hits = [p for p in raw_root.rglob("*.shp") if p.stem == ident]
    if stem_hits:
        return str(stem_hits[0])
    fuzzy = [p for p in raw_root.rglob("*.shp") if ident in p.stem or p.stem in ident]
    if fuzzy:
        return str(fuzzy[0])
    raise FileNotFoundError(f"cannot resolve source_shp for identity={ident!r}; old source_shp={src!r}")


def _make_attack_value(chain: list[dict]) -> str:
    return json.dumps(chain, ensure_ascii=False, separators=(",", ":"))


def extend_mixed(
    base_dataset_root: Path,
    base_split_root: Path,
    source_root: Path,
    output_dataset_root: Path,
    output_split_root: Path,
    mixed_config_path: Path,
    force: bool,
    seed: int,
    mapshaper_bin: str,
) -> dict:
    if force:
        if output_dataset_root.exists():
            shutil.rmtree(output_dataset_root)
        if output_split_root.exists():
            shutil.rmtree(output_split_root)

    print(f"[COPY] hardlink/copy base dataset: {base_dataset_root} -> {output_dataset_root}", flush=True)
    _copy_or_link(str(base_dataset_root), str(output_dataset_root))

    cfg = json.loads(mixed_config_path.read_text(encoding="utf-8"))
    grid_size = int(cfg.get("grid_size", 256))
    mixed_attacks = list(cfg.get("mixed_attacks", []))
    if not mixed_attacks:
        raise ValueError("mixed_attacks is empty")

    channel_cfg = ChannelBuildConfig(grid_size=grid_size, pad_ratio=float(cfg.get("pad_ratio", 0.03)))
    token_cfg = TokenConfig(max_tokens=int(cfg.get("max_tokens", 512)))
    graph_cfg = GraphConfig(max_nodes=int(cfg.get("max_graph_nodes", 256)))
    vector_format = str(cfg.get("vector_format", "gpkg"))

    old_meta = pd.read_csv(base_dataset_root / "metadata.csv")
    meta = old_meta.copy()
    for col in ["sample_dir", "vector_path"]:
        if col in meta.columns:
            meta[col] = meta[col].map(lambda x: _rewrite_prefix(x, base_dataset_root, output_dataset_root))
    base_rows = old_meta[old_meta["attack_type"].astype(str) == "base"].to_dict("records")

    new_rows = []
    quality_rows = []
    error_rows = []

    for ident_idx, base_row in enumerate(base_rows):
        identity = str(base_row["identity"])
        try:
            src_shp = _find_source_shp(base_row, source_root)
            print(f"[MIXED] identity={identity} source={src_shp}", flush=True)
            gdf0 = gpd.read_file(src_shp)
            gdf0 = repair_gdf(gdf0)
        except Exception as exc:
            error_rows.append({
                "stage": "read_base",
                "identity": identity,
                "error": repr(exc),
                "traceback_text": traceback.format_exc(),
            })
            print(f"[WARN] skip identity={identity} read error={exc}", flush=True)
            continue

        for mix_idx, mix in enumerate(mixed_attacks, start=1):
            name = str(mix["name"])
            chain = list(mix["chain"])
            attack_name = f"{name}_{mix_idx:03d}"
            attack_dir = output_dataset_root / identity / "attacks" / attack_name
            try:
                cur = gdf0.copy()
                chain_reports = []
                resolved_chain = []
                for step_idx, raw_spec_dict in enumerate(chain, start=1):
                    raw_spec = spec_from_dict(raw_spec_dict)
                    spec = _resolve_attack_spec(raw_spec, cur)
                    cur, report = apply_attack(cur, spec, mapshaper_bin=mapshaper_bin, seed=seed + ident_idx * 1000 + mix_idx * 100 + step_idx)
                    report["step_idx"] = step_idx
                    report["step_attack_type"] = spec.attack_type
                    report["step_attack_engine"] = spec.engine
                    report["step_attack_value"] = spec.value
                    chain_reports.append(report)
                    resolved_chain.append(spec.to_dict())
                    if cur.empty:
                        raise RuntimeError(f"empty geometry after step {step_idx}: {spec.to_dict()}")

                final_report = {
                    "identity": identity,
                    "attack_name": attack_name,
                    "attack_type": name,
                    "attack_family": "mixed",
                    "attack_chain": "|".join([str(s.get("attack_type")) for s in resolved_chain]),
                    "resolved_chain": resolved_chain,
                    "chain_reports": chain_reports,
                    "source_shp": src_shp,
                    "is_accepted": True,
                }
                meta_obj = _save_sample(
                    cur,
                    attack_dir,
                    sample_name=attack_name,
                    channel_cfg=channel_cfg,
                    token_cfg=token_cfg,
                    graph_cfg=graph_cfg,
                    source_path=src_shp,
                    attack_report=final_report,
                    save_vector=True,
                    vector_format=vector_format,
                )

                row = {
                    "identity": identity,
                    "sample": attack_name,
                    "sample_dir": str(attack_dir),
                    "source_shp": src_shp,
                    "source_rel_path": str(base_row.get("source_rel_path", "")),
                    "attack_type": name,
                    "attack_engine": "mixed",
                    "attack_value": _make_attack_value(resolved_chain),
                    "attack_name": attack_name,
                    "attack_value_original": _make_attack_value(chain),
                    "attack_value_mode": "mixed_chain",
                    "attack_family": "mixed",
                    "attack_chain": final_report["attack_chain"],
                    "source_base_sample": str(base_row.get("sample", "base")),
                    "vector_path": str(meta_obj.get("vector_path") or ""),
                }
                new_rows.append(row)
                quality_rows.append(final_report)
            except Exception as exc:
                error_rows.append({
                    "stage": "mixed_attack",
                    "identity": identity,
                    "attack_name": attack_name,
                    "chain": chain,
                    "error": repr(exc),
                    "traceback_text": traceback.format_exc(),
                })
                print(f"[WARN] mixed attack failed identity={identity} attack={attack_name}: {exc}", flush=True)

    full_meta = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)
    full_meta.to_csv(output_dataset_root / "metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality_rows).to_csv(output_dataset_root / "mixed_attack_quality_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(output_dataset_root / "mixed_attack_errors.csv", index=False, encoding="utf-8-sig")

    old_info = {}
    info_path = base_dataset_root / "dataset_info.json"
    if info_path.exists():
        old_info = json.loads(info_path.read_text(encoding="utf-8"))
    info = dict(old_info)
    info.update({
        "format_version": "RB_AFL_DATASET_V17_2_MIXED_EXT",
        "base_dataset_root": str(base_dataset_root),
        "base_split_root": str(base_split_root),
        "source_root_for_mixed": str(source_root),
        "output_root": str(output_dataset_root),
        "num_samples": int(len(full_meta)),
        "num_original_samples_reused": int(len(meta)),
        "num_mixed_samples_added": int(len(new_rows)),
        "num_mixed_errors": int(len(error_rows)),
        "grid_size": grid_size,
        "channel_names": CHANNEL_NAMES,
        "tensor_shape": [4, grid_size, grid_size],
        "mixed_config": cfg,
    })
    write_json(output_dataset_root / "dataset_info.json", info)

    ensure_dir(output_split_root)
    split_summary = []
    for split in ["train", "val", "test"]:
        src_split_csv = base_split_root / split / "metadata.csv"
        if not src_split_csv.exists():
            raise FileNotFoundError(src_split_csv)
        old_split = pd.read_csv(src_split_csv)
        split_ids = set(old_split["identity"].astype(str))
        out_split_dir = ensure_dir(output_split_root / split)
        out_split = full_meta[full_meta["identity"].astype(str).isin(split_ids)].copy()
        out_split.to_csv(out_split_dir / "metadata.csv", index=False, encoding="utf-8-sig")
        split_summary.append({
            "split": split,
            "rows": int(len(out_split)),
            "identities": int(out_split["identity"].astype(str).nunique()),
            "mixed_rows": int((out_split["attack_engine"].astype(str) == "mixed").sum()),
            "base_rows": int((out_split["attack_type"].astype(str) == "base").sum()),
        })

        # mixed-only eval split: base rows + mixed rows only.
        out_mixed_dir = ensure_dir(output_split_root / f"{split}_mixed_only")
        out_mixed = out_split[
            (out_split["attack_type"].astype(str) == "base") |
            (out_split["attack_engine"].astype(str) == "mixed")
        ].copy()
        out_mixed.to_csv(out_mixed_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(split_summary).to_csv(output_split_root / "split_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "output_dataset_root": str(output_dataset_root),
        "output_split_root": str(output_split_root),
        "num_rows_total": int(len(full_meta)),
        "num_mixed_rows_added": int(len(new_rows)),
        "num_errors": int(len(error_rows)),
        "split_summary": split_summary,
    }
    write_json(output_split_root / "mixed_extension_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dataset_root", required=True)
    ap.add_argument("--base_split_root", required=True)
    ap.add_argument("--source_root", required=True)
    ap.add_argument("--output_dataset_root", required=True)
    ap.add_argument("--output_split_root", required=True)
    ap.add_argument("--mixed_config", required=True)
    ap.add_argument("--force", type=_bool, default=False)
    ap.add_argument("--seed", type=int, default=20260318)
    ap.add_argument("--mapshaper_bin", default="mapshaper")
    ns = ap.parse_args()
    extend_mixed(
        base_dataset_root=Path(ns.base_dataset_root),
        base_split_root=Path(ns.base_split_root),
        source_root=Path(ns.source_root),
        output_dataset_root=Path(ns.output_dataset_root),
        output_split_root=Path(ns.output_split_root),
        mixed_config_path=Path(ns.mixed_config),
        force=bool(ns.force),
        seed=int(ns.seed),
        mapshaper_bin=str(ns.mapshaper_bin),
    )


if __name__ == "__main__":
    main()
