#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create formal train/val/test dataset splits by identity.

The existing RB-AFL dataset layout stores each sample directory on disk and uses
metadata.csv to point to those sample directories.  This splitter creates three
lightweight dataset roots containing only metadata.csv files whose sample_dir and
vector_path entries are absolute paths back to the original built samples.

Splitting by identity is mandatory for zero-watermark evaluation: the same
identity must never appear in both train and test, otherwise uniqueness and
robustness numbers can be contaminated by identity leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


def _bool_arg(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _abs_path(value: Any, dataset_root: Path, must_be_dir: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    p = Path(text)
    if not p.is_absolute():
        p = dataset_root / p
    p = p.resolve()
    if must_be_dir and not p.is_dir():
        raise FileNotFoundError(f"sample_dir not found: {p}")
    return str(p)


def _compute_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float, min_ids: int) -> tuple[int, int, int]:
    if n < 3 * min_ids:
        raise ValueError(
            f"Need at least {3 * min_ids} identities for three formal splits with min_ids={min_ids}; got {n}"
        )
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must be positive")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    n_train = max(min_ids, int(round(n * train_ratio)))
    n_val = max(min_ids, int(round(n * val_ratio)))
    n_test = n - n_train - n_val
    if n_test < min_ids:
        deficit = min_ids - n_test
        take_train = min(deficit, max(0, n_train - min_ids))
        n_train -= take_train
        deficit -= take_train
        if deficit > 0:
            take_val = min(deficit, max(0, n_val - min_ids))
            n_val -= take_val
            deficit -= take_val
        n_test = n - n_train - n_val
    if n_test < min_ids:
        raise ValueError(f"Unable to allocate test split with min_ids={min_ids}: n={n}")
    return n_train, n_val, n_test


def _summarize(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    attack_counts = df["attack_type"].astype(str).value_counts().sort_index().to_dict() if not df.empty else {}
    return {
        "split": split_name,
        "num_rows": int(len(df)),
        "num_identities": int(df["identity"].astype(str).nunique()) if not df.empty else 0,
        "num_base_rows": int((df["attack_type"].astype(str) == "base").sum()) if not df.empty else 0,
        "attack_counts": {str(k): int(v) for k, v in attack_counts.items()},
    }


def split_dataset_formal(
    dataset_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 20260318,
    min_identities_per_split: int = 2,
    check_paths: bool = True,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    meta_path = dataset_root / "metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")
    df = pd.read_csv(meta_path)
    required = {"identity", "sample", "sample_dir", "attack_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")

    df = df.copy()
    df["identity"] = df["identity"].astype(str)
    df["sample_dir"] = df["sample_dir"].apply(lambda x: _abs_path(x, dataset_root, must_be_dir=check_paths))
    if "vector_path" in df.columns:
        df["vector_path"] = df["vector_path"].apply(lambda x: _abs_path(x, dataset_root, must_be_dir=False))

    identities = sorted(df["identity"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(identities)
    n_train, n_val, n_test = _compute_counts(
        len(identities), train_ratio, val_ratio, test_ratio, min_identities_per_split
    )
    train_ids = sorted(identities[:n_train])
    val_ids = sorted(identities[n_train:n_train + n_val])
    test_ids = sorted(identities[n_train + n_val:n_train + n_val + n_test])
    assigned = set(train_ids) | set(val_ids) | set(test_ids)
    if len(assigned) != len(identities):
        raise RuntimeError("identity assignment is not exhaustive or has duplicates")

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_info = _read_json(dataset_root / "dataset_info.json")
    summaries: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []

    for split_name, ids in split_map.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        part = df[df["identity"].isin(ids)].copy().reset_index(drop=True)
        if part.empty:
            raise ValueError(f"split {split_name} is empty")
        if (part["attack_type"].astype(str) == "base").sum() < len(ids):
            raise ValueError(f"split {split_name} is missing base rows for some identities")
        part.to_csv(split_dir / "metadata.csv", index=False, encoding="utf-8-sig")
        summary = _summarize(part, split_name)
        summaries.append(summary)
        _write_json(split_dir / "dataset_info.json", {
            "format_version": "RB_AFL_FORMAL_SPLIT_V14",
            "split": split_name,
            "source_dataset_root": str(dataset_root),
            "split_root": str(split_dir),
            "metadata_csv": str(split_dir / "metadata.csv"),
            "identity_split_seed": int(seed),
            "identity_split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "identity_list": ids,
            "summary": summary,
            "source_dataset_info": dataset_info,
        })
        for ident in ids:
            assignments.append({"identity": ident, "split": split_name})

    pd.DataFrame(assignments).sort_values(["split", "identity"]).to_csv(
        output_root / "split_assignments.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    for summary in summaries:
        row = {k: v for k, v in summary.items() if k != "attack_counts"}
        for attack, count in summary["attack_counts"].items():
            row[f"attack_{attack}"] = count
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "split_summary.csv", index=False, encoding="utf-8-sig")
    out_info = {
        "format_version": "RB_AFL_FORMAL_SPLIT_V14",
        "source_dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "seed": int(seed),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "paths": {
            "train_dataset_root": str(output_root / "train"),
            "val_dataset_root": str(output_root / "val"),
            "test_dataset_root": str(output_root / "test"),
            "split_assignments_csv": str(output_root / "split_assignments.csv"),
            "split_summary_csv": str(output_root / "split_summary.csv"),
        },
        "summaries": summaries,
    }
    _write_json(output_root / "split_info.json", out_info)
    return out_info


def main() -> None:
    ap = argparse.ArgumentParser(description="Create formal train/val/test splits by identity for RB-AFL datasets")
    ap.add_argument("--dataset_root", required=True, help="Built dataset root containing metadata.csv")
    ap.add_argument("--output_root", required=True, help="Output root containing train/val/test metadata roots")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260318)
    ap.add_argument("--min_identities_per_split", type=int, default=2)
    ap.add_argument("--check_paths", type=_bool_arg, default=True)
    ns = ap.parse_args()
    try:
        info = split_dataset_formal(
            dataset_root=ns.dataset_root,
            output_root=ns.output_root,
            train_ratio=float(ns.train_ratio),
            val_ratio=float(ns.val_ratio),
            test_ratio=float(ns.test_ratio),
            seed=int(ns.seed),
            min_identities_per_split=int(ns.min_identities_per_split),
            check_paths=bool(ns.check_paths),
        )
        print(json.dumps(info, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[EXC] split_dataset_formal_V14: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
