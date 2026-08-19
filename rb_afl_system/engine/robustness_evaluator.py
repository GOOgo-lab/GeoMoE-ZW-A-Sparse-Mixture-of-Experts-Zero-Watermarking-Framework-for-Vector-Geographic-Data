#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness and zero-watermark evaluator."""
from __future__ import annotations
from pathlib import Path
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
from rb_afl_system.watermark.zero_watermark import make_random_copyright_bits, register_zero_watermark, recover_watermark, evaluate_recovery
from rb_afl_system.watermark.copyright_image import load_copyright_bits
from rb_afl_system.utils import ensure_dir, write_json


def _sample_to_device(sample: dict, device: torch.device) -> dict:
    return {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}


def _forward(generator, sample: dict) -> np.ndarray:
    z = generator(grid=sample["grid"], tokens=sample["tokens"], token_mask=sample["token_mask"], graph_nodes=sample["graph_nodes"], graph_adj=sample["graph_adj"], graph_mask=sample["graph_mask"])
    return z.detach().cpu().numpy()[0]


def _safe_str_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _add_attack_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "attack_value" not in out.columns:
        out["attack_value"] = ""
    if "attack_name" not in out.columns:
        out["attack_name"] = ""
    out["attack_value"] = out["attack_value"].map(_safe_str_value)
    out["attack_name"] = out["attack_name"].map(_safe_str_value)
    out["attack_label"] = out.apply(
        lambda r: "|".join([
            _safe_str_value(r.get("attack_engine", "unknown")),
            _safe_str_value(r.get("attack_type", "unknown")),
            _safe_str_value(r.get("attack_value", "")),
        ]),
        axis=1,
    )
    return out


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, dropna=False).agg(
        mean_feature_nc=("feature_nc", "mean"),
        min_feature_nc=("feature_nc", "min"),
        max_feature_ber=("feature_ber", "max"),
        mean_feature_ber=("feature_ber", "mean"),
        mean_watermark_nc=("watermark_nc", "mean"),
        min_watermark_nc=("watermark_nc", "min"),
        max_watermark_ber=("watermark_ber", "max"),
        mean_watermark_ber=("watermark_ber", "mean"),
        count=("identity", "count"),
    ).reset_index()


def evaluate_robustness(config: Dict[str, Any]) -> dict:
    dataset_root = Path(config["dataset_root"])
    output_dir = ensure_dir(config["output_dir"])
    device = resolve_device(config.get("device", "auto"))
    ckpt = load_checkpoint(config["ckpt_path"], map_location=resolve_map_location(config.get("device", "auto")))
    model_cfg = dict(ckpt.get("config", {}))
    channels = list(ckpt.get("channel_names", model_cfg.get("channels", ["occ", "dist", "orient", "density"])))
    model_cfg.update({"generator": ckpt.get("generator_name", model_cfg.get("generator", "cnn_baseline")), "in_channels": len(channels)})

    generator = build_generator(model_cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    meta = pd.read_csv(dataset_root / "metadata.csv")
    bit_length = int(config.get("bit_length", ckpt.get("feat_dim", 256)))
    copyright_image_path = str(config.get("copyright_image_path", "")).strip()
    if copyright_image_path:
        copyright_bits = load_copyright_bits(
            copyright_image_path,
            bit_length=bit_length,
            threshold=int(config.get("copyright_image_threshold", 128)),
            arnold_iterations=int(config.get("arnold_iterations", 0)),
        )
    else:
        copyright_bits = make_random_copyright_bits(bit_length, seed=int(config.get("copyright_seed", 20260318)))
    nc_threshold = float(config.get("verification_nc_threshold", 0.80))
    ber_threshold = float(config.get("verification_ber_threshold", 1.0))
    threshold_mode = str(config.get("threshold_mode", "mean"))
    rows: List[dict] = []
    with torch.no_grad():
        for identity, group in meta.groupby("identity"):
            base_group = group[group["attack_type"].astype(str) == "base"]
            if base_group.empty:
                continue
            base_row = base_group.iloc[0].to_dict()
            base_sample = _sample_to_device(_load_sample(base_row["sample_dir"], channels), device)
            base_bits = feature_to_bits(_forward(generator, base_sample), bit_length=bit_length, threshold_mode=threshold_mode)
            registered = register_zero_watermark(copyright_bits, base_bits)
            attacks = group[group["attack_type"].astype(str) != "base"]
            for _, row in attacks.iterrows():
                sample = _sample_to_device(_load_sample(row["sample_dir"], channels), device)
                attack_bits = feature_to_bits(_forward(generator, sample), bit_length=bit_length, threshold_mode=threshold_mode)
                recovered = recover_watermark(registered, attack_bits)
                rec_eval = evaluate_recovery(copyright_bits, recovered)
                rows.append({
                    "identity": str(identity),
                    "attack_type": str(row.get("attack_type", "unknown")),
                    "attack_engine": str(row.get("attack_engine", "unknown")),
                    "attack_value": _safe_str_value(row.get("attack_value", "")),
                    "attack_name": _safe_str_value(row.get("attack_name", row.get("sample", ""))),
                    "attack_value_original": _safe_str_value(row.get("attack_value_original", "")),
                    "attack_value_mode": _safe_str_value(row.get("attack_value_mode", "")),
                    "feature_nc": nc_score(base_bits, attack_bits),
                    "feature_ber": ber_score(base_bits, attack_bits),
                    "watermark_nc": rec_eval["nc"],
                    "watermark_ber": rec_eval["ber"],
                    "accepted": bool(rec_eval["nc"] >= nc_threshold and rec_eval["ber"] <= ber_threshold),
                    "sample_dir": str(row["sample_dir"]),
                })
    if not rows:
        raise ValueError("No attack samples found for robustness evaluation")
    df = _add_attack_labels(pd.DataFrame(rows))
    df.to_csv(output_dir / "robustness_rows.csv", index=False, encoding="utf-8-sig")

    by_attack = _aggregate(df, ["attack_type"])
    by_attack.to_csv(output_dir / "robustness_by_attack.csv", index=False, encoding="utf-8-sig")

    by_engine = _aggregate(df, ["attack_type", "attack_engine"])
    by_engine.to_csv(output_dir / "robustness_by_attack_engine.csv", index=False, encoding="utf-8-sig")

    detail_cols = ["attack_type", "attack_engine", "attack_value", "attack_value_original", "attack_value_mode"]
    by_detail = _aggregate(df, detail_cols)
    by_detail.to_csv(output_dir / "robustness_by_attack_engine_value.csv", index=False, encoding="utf-8-sig")

    worst_k = int(config.get("worst_k", 20))
    worst = df.sort_values(["feature_nc", "watermark_nc", "feature_ber"], ascending=[True, True, False]).head(worst_k)
    worst.to_csv(output_dir / "worst_robustness_cases.csv", index=False, encoding="utf-8-sig")

    worst_row = df.sort_values("feature_nc", ascending=True).iloc[0].to_dict()
    summary = {
        "num_rows": int(len(df)),
        "mean_robust_nc": float(df["feature_nc"].mean()),
        "min_robust_nc": float(df["feature_nc"].min()),
        "mean_robust_ber": float(df["feature_ber"].mean()),
        "max_robust_ber": float(df["feature_ber"].max()),
        "mean_watermark_nc": float(df["watermark_nc"].mean()),
        "min_watermark_nc": float(df["watermark_nc"].min()),
        "mean_watermark_ber": float(df["watermark_ber"].mean()),
        "max_watermark_ber": float(df["watermark_ber"].max()),
        "worst_identity": str(worst_row.get("identity", "")),
        "worst_attack_type": str(worst_row.get("attack_type", "")),
        "worst_attack_engine": str(worst_row.get("attack_engine", "")),
        "worst_attack_value": str(worst_row.get("attack_value", "")),
        "worst_feature_nc": float(worst_row.get("feature_nc", 0.0)),
        "worst_feature_ber": float(worst_row.get("feature_ber", 0.0)),
        "worst_watermark_nc": float(worst_row.get("watermark_nc", 0.0)),
        "nc_threshold": nc_threshold,
        "ber_threshold": ber_threshold,
        "nc_below_threshold": int((df["watermark_nc"] < nc_threshold).sum()),
        "rpr": float(df["accepted"].mean()),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "robustness_summary.json", summary)
    return summary
