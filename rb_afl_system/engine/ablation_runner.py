#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation runner for channel/model/loss/discriminator variants."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from rb_afl_system.config import load_config
from rb_afl_system.engine.adversarial_trainer import train_adversarial
from rb_afl_system.engine.uniqueness_evaluator import evaluate_uniqueness
from rb_afl_system.engine.robustness_evaluator import evaluate_robustness
from rb_afl_system.models.baseline.handcrafted_baseline import handcrafted_feature
from rb_afl_system.utils import ensure_dir, write_json, log


def run_ablation(config: Dict[str, Any]) -> dict:
    dataset_root = config["dataset_root"]
    output_root = ensure_dir(config["output_root"])
    base_train = dict(config.get("train", {}))
    base_eval = dict(config.get("eval", {}))
    experiments = list(config.get("experiments", []))
    if not experiments:
        raise ValueError("Ablation config has no experiments")
    results: List[dict] = []
    for exp in experiments:
        exp_id = str(exp["id"])
        exp_name = str(exp.get("name", exp_id))
        log(f"========== ABLATION {exp_id} {exp_name} ==========")
        exp_dir = ensure_dir(output_root / f"{exp_id}_{exp_name}")
        mode = str(exp.get("mode", "train"))
        if mode == "handcrafted":
            # Handcrafted evaluation can be added to a paper-specific baseline pipeline.
            row = {"exp_id": exp_id, "exp_name": exp_name, "mode": mode, "note": "handcrafted baseline is available via models.baseline.handcrafted_baseline"}
            results.append(row)
            write_json(exp_dir / "ablation_result.json", row)
            continue
        train_cfg = dict(base_train)
        train_cfg.update(exp)
        train_cfg["dataset_root"] = dataset_root
        train_cfg["output_dir"] = str(exp_dir / "train")
        train_summary = train_adversarial(train_cfg)
        ckpt_path = str(Path(train_summary["out_dir"]) / "best.pt")
        uniq_dir = exp_dir / "uniqueness"
        robust_dir = exp_dir / "robustness"
        uniq = evaluate_uniqueness({"dataset_root": dataset_root, "ckpt_path": ckpt_path, "output_dir": str(uniq_dir), **base_eval})
        robust = evaluate_robustness({"dataset_root": dataset_root, "ckpt_path": ckpt_path, "output_dir": str(robust_dir), **base_eval})
        row = {"exp_id": exp_id, "exp_name": exp_name, "mode": mode, **{k: exp.get(k) for k in exp.keys()}, **uniq, **robust, "ckpt_path": ckpt_path}
        results.append(row)
        write_json(exp_dir / "ablation_result.json", row)
    df = pd.DataFrame(results)
    df.to_csv(output_root / "ablation_results.csv", index=False, encoding="utf-8-sig")
    write_json(output_root / "ablation_results.json", {"results": results})
    return {"output_root": str(output_root), "num_experiments": len(results)}
