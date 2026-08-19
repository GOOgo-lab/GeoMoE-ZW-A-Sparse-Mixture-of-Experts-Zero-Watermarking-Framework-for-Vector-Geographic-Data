#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uniqueness evaluator using base samples only."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
from rb_afl_system.data.dataset.triplet_dataset import _load_sample
from rb_afl_system.engine.checkpoint_io import load_checkpoint
from rb_afl_system.engine.device import resolve_device, resolve_map_location
from rb_afl_system.models.model_registry import build_generator
from rb_afl_system.watermark.feature_to_bits import feature_to_bits
from rb_afl_system.watermark.metrics import nc_score, ber_score
from rb_afl_system.utils import ensure_dir, write_json


def _sample_to_device(sample: dict, device: torch.device) -> dict:
    return {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}


def _forward(generator, sample: dict) -> torch.Tensor:
    return generator(grid=sample["grid"], tokens=sample["tokens"], token_mask=sample["token_mask"], graph_nodes=sample["graph_nodes"], graph_adj=sample["graph_adj"], graph_mask=sample["graph_mask"])


def evaluate_uniqueness(config: Dict[str, Any]) -> dict:
    from pathlib import Path
    dataset_root = Path(config["dataset_root"])
    output_dir = ensure_dir(config["output_dir"])
    device = resolve_device(config.get("device", "auto"))
    ckpt = load_checkpoint(config["ckpt_path"], map_location=resolve_map_location(config.get("device", "auto")))
    model_cfg = dict(ckpt.get("config", {}))
    model_cfg.update({"generator": ckpt.get("generator_name", model_cfg.get("generator", "cnn_baseline")), "in_channels": len(ckpt.get("channel_names", model_cfg.get("channels", ["occ", "dist", "orient", "density"])))})
    channels = list(ckpt.get("channel_names", model_cfg.get("channels", ["occ", "dist", "orient", "density"])))

    generator = build_generator(model_cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    meta = pd.read_csv(dataset_root / "metadata.csv")
    base_rows = meta[meta["attack_type"].astype(str) == "base"].to_dict("records")
    if len(base_rows) < 2:
        raise ValueError("Uniqueness requires at least 2 base samples")
    names: List[str] = []
    bits_list: List[np.ndarray] = []
    with torch.no_grad():
        for row in base_rows:
            sample = _load_sample(row["sample_dir"], channels)
            sample = _sample_to_device(sample, device)
            z = _forward(generator, sample).detach().cpu().numpy()[0]
            bits = feature_to_bits(z, bit_length=int(config.get("bit_length", z.size)), threshold_mode=str(config.get("threshold_mode", "mean")))
            names.append(str(row["identity"]))
            bits_list.append(bits)
    n = len(bits_list)
    nc = np.zeros((n, n), dtype=np.float32)
    ber = np.zeros((n, n), dtype=np.float32)
    pair_rows: list[dict] = []
    for i in range(n):
        for j in range(n):
            nc[i, j] = nc_score(bits_list[i], bits_list[j])
            ber[i, j] = ber_score(bits_list[i], bits_list[j])
            if i < j:
                pair_rows.append({
                    "identity_a": names[i],
                    "identity_b": names[j],
                    "unique_nc": float(nc[i, j]),
                    "unique_ber": float(ber[i, j]),
                })
    pd.DataFrame(nc, index=names, columns=names).to_csv(output_dir / "uniqueness_nc_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(ber, index=names, columns=names).to_csv(output_dir / "uniqueness_ber_matrix.csv", encoding="utf-8-sig")
    pair_df = pd.DataFrame(pair_rows).sort_values(["unique_nc", "unique_ber"], ascending=[False, True])
    pair_df.to_csv(output_dir / "uniqueness_pairs.csv", index=False, encoding="utf-8-sig")
    off = ~np.eye(n, dtype=bool)
    worst_pair = pair_df.iloc[0].to_dict()
    summary = {
        "num_samples": n,
        "mean_unique_nc": float(nc[off].mean()),
        "max_unique_nc": float(nc[off].max()),
        "mean_unique_ber": float(ber[off].mean()),
        "min_unique_ber": float(ber[off].min()),
        "worst_pair_identity_a": str(worst_pair.get("identity_a", "")),
        "worst_pair_identity_b": str(worst_pair.get("identity_b", "")),
        "worst_pair_nc": float(worst_pair.get("unique_nc", 0.0)),
        "worst_pair_ber": float(worst_pair.get("unique_ber", 0.0)),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "uniqueness_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "uniqueness_summary.json", summary)
    return summary
