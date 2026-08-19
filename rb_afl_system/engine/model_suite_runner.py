#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run train/eval/compare suites for multiple RB-AFL models."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from rb_afl_system.engine.adversarial_trainer import train_adversarial
from rb_afl_system.engine.eval_reporter import compare_model_results, export_paper_ready_tables
from rb_afl_system.engine.robustness_evaluator import evaluate_robustness
from rb_afl_system.engine.uniqueness_evaluator import evaluate_uniqueness
from rb_afl_system.utils import ensure_dir, log, write_json


def _merge_dicts(*items: dict) -> dict:
    out: dict[str, Any] = {}
    for item in items:
        out.update(dict(item or {}))
    return out


def _run_visual_report_subprocess(output_root: Path, visual_dir: str, timestamp: str | None) -> dict[str, Any]:
    """Run visual report in a separate Python process.

    Windows can hit OpenMP conflicts when matplotlib/numpy/torch are loaded in
    the same long-lived training process.  V11.1 isolates the visualization step
    and sets conservative OpenMP-related environment variables so training
    results are not lost if plotting fails.
    """
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    cmd = [
        sys.executable,
        "-m",
        "rb_afl_system.scripts.export_visual_report",
        "--suite_root",
        str(output_root),
        "--output_dir",
        str(visual_dir),
    ]
    if timestamp:
        cmd.extend(["--timestamp", str(timestamp)])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    report = {
        "subprocess": True,
        "returncode": int(proc.returncode),
        "cmd": cmd,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-80:]),
    }
    if proc.returncode != 0:
        log(f"[WARN] visual report subprocess failed returncode={proc.returncode}")
        log(report["stdout_tail"])
    else:
        log(f"[OK] visual report generated in subprocess: {visual_dir}")
    return report


def run_model_suite(config: Dict[str, Any]) -> dict:
    """Run a multi-model train/eval suite.

    Config structure:
      dataset_root: path
      output_root: path
      common_train: dict
      common_eval: dict
      skip_train_if_best_exists: bool
      models: [{name:..., train:{...}}]
    """
    dataset_root = str(config["dataset_root"])
    train_dataset_root = str(config.get("train_dataset_root", dataset_root))
    val_dataset_root = str(config.get("val_dataset_root", "") or "")
    eval_dataset_root = str(config.get("eval_dataset_root", dataset_root))
    output_root = ensure_dir(config["output_root"])
    runs_root = ensure_dir(output_root / "runs")
    eval_root = ensure_dir(output_root / "evals")
    compare_root = ensure_dir(output_root / "compare")
    common_train = dict(config.get("common_train", {}))
    common_eval = dict(config.get("common_eval", {}))
    models = list(config.get("models", []))
    if not models:
        raise ValueError("model suite config has no models")
    skip_existing = bool(config.get("skip_train_if_best_exists", False))
    compare_items: list[dict] = []
    run_rows: list[dict] = []

    write_json(output_root / "model_suite_config.json", config)
    for item in models:
        name = str(item["name"])
        log(f"========== MODEL SUITE {name} ==========")
        model_run_dir = ensure_dir(runs_root / name)
        uniq_dir = ensure_dir(eval_root / f"uniqueness_{name}")
        rob_dir = ensure_dir(eval_root / f"robustness_{name}")
        train_cfg = _merge_dicts(common_train, item.get("train", {}))
        train_cfg["dataset_root"] = train_dataset_root
        train_cfg["train_dataset_root"] = train_dataset_root
        if val_dataset_root:
            train_cfg["val_dataset_root"] = val_dataset_root
        train_cfg["output_dir"] = str(model_run_dir)
        ckpt_path = model_run_dir / "best.pt"
        train_summary: dict[str, Any]
        if skip_existing and ckpt_path.is_file():
            log(f"[SKIP-TRAIN] existing best.pt: {ckpt_path}")
            train_summary = {"out_dir": str(model_run_dir), "best_val": None, "epochs": int(train_cfg.get("epochs", 0)), "skipped": True}
        else:
            train_summary = train_adversarial(train_cfg)
            ckpt_path = Path(train_summary["out_dir"]) / "best.pt"
        eval_base = dict(common_eval)
        eval_base.update({"dataset_root": eval_dataset_root, "ckpt_path": str(ckpt_path)})
        uniq = evaluate_uniqueness({**eval_base, "output_dir": str(uniq_dir)})
        rob = evaluate_robustness({**eval_base, "output_dir": str(rob_dir)})
        compare_items.append({"name": name, "uniqueness_dir": str(uniq_dir), "robustness_dir": str(rob_dir)})
        row = {"name": name, "ckpt_path": str(ckpt_path), **{f"train_{k}": v for k, v in train_summary.items()}, **{f"uniq_{k}": v for k, v in uniq.items()}, **{f"rob_{k}": v for k, v in rob.items()}}
        run_rows.append(row)
        write_json(output_root / f"{name}_suite_summary.json", row)

    compare = compare_model_results(compare_items, compare_root)
    xlsx_path = config.get("output_xlsx", "")
    if xlsx_path:
        export_paper_ready_tables(compare["model_compare_csv"], xlsx_path)
        compare["output_xlsx"] = str(xlsx_path)
    if bool(config.get("auto_visual_report", False)):
        visual_dir = config.get("visual_output_dir", "") or str(output_root / "visuals")
        visual_report = _run_visual_report_subprocess(
            output_root=output_root,
            visual_dir=visual_dir,
            timestamp=str(config.get("timestamp", "") or "") or None,
        )
        compare["visual_report"] = visual_report
    summary = {
        "output_root": str(output_root),
        "num_models": len(models),
        "dataset_root": dataset_root,
        "train_dataset_root": train_dataset_root,
        "val_dataset_root": val_dataset_root,
        "eval_dataset_root": eval_dataset_root,
        "compare": compare,
        "runs": run_rows,
    }
    write_json(output_root / "model_suite_summary.json", summary)
    return summary
