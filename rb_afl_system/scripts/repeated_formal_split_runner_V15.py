#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V15 repeated formal identity split runner.

It creates repeated identity-level train/val/test splits and, optionally, runs a
full model suite + profiler + specialist/single-model reports for each split.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from rb_afl_system.utils import ensure_dir, write_json


def _bool_arg(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed returncode={proc.returncode}: {' '.join(cmd)}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_suite_config(template: Path, split_root: Path, output_root: Path, dst: Path, tag: str) -> None:
    cfg = _load_json(template)
    cfg["dataset_root"] = str(split_root / "train")
    cfg["train_dataset_root"] = str(split_root / "train")
    cfg["val_dataset_root"] = str(split_root / "val")
    cfg["eval_dataset_root"] = str(split_root / "test")
    cfg["output_root"] = str(output_root)
    cfg["output_xlsx"] = str(output_root / f"{tag}.xlsx")
    cfg["timestamp"] = tag
    cfg.setdefault("skip_train_if_best_exists", False)
    dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _aggregate_repeat_summaries(repeat_dirs: list[Path], output_root: Path) -> None:
    rows = []
    single_rows = []
    for rep_dir in repeat_dirs:
        rep = rep_dir.name
        for policy in ["gated", "strict"]:
            p = rep_dir / f"specialists_{policy}" / "ensemble_eval" / "specialist_ensemble_summary_v15.csv"
            if p.is_file():
                df = pd.read_csv(p)
                if not df.empty:
                    rec = df.iloc[0].to_dict()
                    rec["repeat"] = rep
                    rec["policy"] = policy
                    rows.append(rec)
        p2 = rep_dir / "single_model_all_ability" / "single_model_all_ability_summary.csv"
        if p2.is_file():
            df2 = pd.read_csv(p2)
            df2.insert(0, "repeat", rep)
            single_rows.append(df2)
    if rows:
        ens = pd.DataFrame(rows)
        ens.to_csv(output_root / "repeated_specialist_ensemble_summary.csv", index=False, encoding="utf-8-sig")
        numeric = ens.select_dtypes(include="number").columns.tolist()
        group = ens.groupby("policy")[numeric].agg(["mean", "std", "min", "max"])
        group.to_csv(output_root / "repeated_specialist_ensemble_summary_stats.csv", encoding="utf-8-sig")
    if single_rows:
        singles = pd.concat(single_rows, ignore_index=True)
        singles.to_csv(output_root / "repeated_single_model_all_ability_summary.csv", index=False, encoding="utf-8-sig")
        if "model" in singles.columns:
            numeric = singles.select_dtypes(include="number").columns.tolist()
            stats = singles.groupby("model")[numeric].agg(["mean", "std", "min", "max"])
            stats.to_csv(output_root / "repeated_single_model_all_ability_stats.csv", encoding="utf-8-sig")


def run_repeated_formal_splits(
    dataset_root: str | Path,
    suite_template: str | Path,
    output_root: str | Path,
    repeats: int = 5,
    base_seed: int = 20260318,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    run_suite: bool = True,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    suite_template = Path(suite_template)
    output_root = ensure_dir(output_root)
    repeat_dirs: list[Path] = []

    for i in range(int(repeats)):
        seed = int(base_seed) + i * 9973
        rep_dir = ensure_dir(output_root / f"repeat_{i:02d}_seed_{seed}")
        split_root = rep_dir / "splits"
        suite_out = rep_dir / "suite_output"
        suite_cfg = rep_dir / "suite_config.local.json"
        tag = f"v15_repeat_{i:02d}_seed_{seed}"
        repeat_dirs.append(rep_dir)

        _run([
            sys.executable,
            "-m",
            "rb_afl_system.scripts.split_dataset_formal_V14",
            "--dataset_root", str(dataset_root),
            "--output_root", str(split_root),
            "--train_ratio", str(train_ratio),
            "--val_ratio", str(val_ratio),
            "--test_ratio", str(test_ratio),
            "--seed", str(seed),
            "--min_identities_per_split", "2",
            "--check_paths", "true",
        ])
        _write_suite_config(suite_template, split_root, suite_out, suite_cfg, tag)
        if not run_suite:
            continue
        _run([sys.executable, "-m", "rb_afl_system.scripts.run_model_suite", "--config", str(suite_cfg)])
        for policy in ["gated", "strict"]:
            spec_dir = rep_dir / f"specialists_{policy}"
            _run([
                sys.executable,
                "-m",
                "rb_afl_system.scripts.capability_profiler_V13",
                "--suite_root", str(suite_out),
                "--output_dir", str(spec_dir),
                "--auto_select", "true",
                "--selection_policy", policy,
            ])
            _run([
                sys.executable,
                "-m",
                "rb_afl_system.scripts.specialist_ensemble_evaluator_V15",
                "--profile_csv", str(spec_dir / "model_capability_profile.csv"),
                "--selected_json", str(spec_dir / "selected_specialists.json"),
                "--output_dir", str(spec_dir / "ensemble_eval"),
            ])
        _run([
            sys.executable,
            "-m",
            "rb_afl_system.scripts.single_model_all_ability_evaluator_V15",
            "--suite_root", str(suite_out),
            "--output_dir", str(rep_dir / "single_model_all_ability"),
            "--ensemble_summary_csv", str(rep_dir / "specialists_gated" / "ensemble_eval" / "specialist_ensemble_summary_v15.csv"),
        ])

    if run_suite:
        _aggregate_repeat_summaries(repeat_dirs, output_root)
    result = {
        "dataset_root": str(dataset_root),
        "suite_template": str(suite_template),
        "output_root": str(output_root),
        "repeats": int(repeats),
        "run_suite": bool(run_suite),
        "repeat_dirs": [str(p) for p in repeat_dirs],
    }
    write_json(output_root / "repeated_formal_split_runner_summary.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run repeated V15 formal identity splits with optional suite/profiler/evaluator")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--suite_template", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=20260318)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--run_suite", type=_bool_arg, default=True)
    ns = ap.parse_args()
    try:
        result = run_repeated_formal_splits(**vars(ns))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] repeated_formal_split_runner_V15: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
