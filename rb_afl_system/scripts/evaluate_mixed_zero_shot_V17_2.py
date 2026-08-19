#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rb_afl_system.engine.robustness_evaluator import evaluate_robustness
from rb_afl_system.scripts.specialist_ensemble_evaluator_V15 import evaluate_specialist_ensemble_v15
from rb_afl_system.scripts.sparse_moe_router_V16 import build_sparse_moe_table, evaluate_oracle_and_static, _summary_from_scores, _attack_summary
from rb_afl_system.scripts.sparse_moe_router_V16_1 import enrich_paired_descriptors
from rb_afl_system.scripts.sparse_moe_torch_fusion_train_V17_1 import RouterNet, _predict_torch
from rb_afl_system.scripts.sparse_moe_fusion_train_V17 import _top_indices, _apply_protected_roles, _fuse_scores_for_indices
from rb_afl_system.utils import ensure_dir, write_json


def _load_selected_models(selected_json: Path) -> list[str]:
    obj = json.loads(selected_json.read_text(encoding="utf-8"))
    models = []
    if "roles" in obj and isinstance(obj["roles"], dict):
        for _, v in obj["roles"].items():
            if isinstance(v, dict):
                m = v.get("model", "")
            else:
                m = v
            if m:
                models.append(str(m))
    else:
        for _, v in obj.items():
            if isinstance(v, dict):
                m = v.get("model", "")
            else:
                m = v
            if m and isinstance(m, str):
                models.append(str(m))
    return sorted(set(models))


def _ckpt_for_model(suite_root: Path, model: str) -> Path:
    # V17.2 mixed zero-shot note:
    # selected_specialists may come from a later optimized profile directory
    # such as specialists_v15_2_param_opt_gated, while suite_root may point to
    # the original formal suite.  Therefore checkpoint lookup must search the
    # whole BASE_RUN / CKPT_SEARCH_ROOT, not only suite_root/runs.
    import os

    candidates = [
        suite_root / "runs" / model / "best.pt",
        suite_root / model / "best.pt",
    ]

    search_roots = [suite_root]
    env_root = os.environ.get("CKPT_SEARCH_ROOT", "").strip()
    if env_root:
        search_roots.append(Path(env_root))

    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(sorted(root.glob(f"**/runs/{model}/best.pt")))
        candidates.extend(sorted(root.glob(f"**/{model}/best.pt")))

    seen = set()
    uniq = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    for c in uniq:
        if c.exists():
            print(f"[CKPT] model={model} -> {c}", flush=True)
            return c

    checked = "\n".join(str(c) for c in uniq[:80])
    raise FileNotFoundError(
        f"best.pt not found for model={model}. Checked candidates:\n{checked}"
    )


def _make_mixed_profile(
    base_profile_csv: Path,
    selected_models: list[str],
    eval_root: Path,
    out_csv: Path,
) -> pd.DataFrame:
    prof = pd.read_csv(base_profile_csv)
    prof = prof[prof["model"].astype(str).isin(selected_models)].copy()
    if prof.empty:
        raise RuntimeError("No selected model rows found in base profile")

    for idx, row in prof.iterrows():
        model = str(row["model"])
        rob_dir = eval_root / "evals" / f"robustness_{model}"
        prof.at[idx, "robustness_dir"] = str(rob_dir)
        summary_csv = rob_dir / "robustness_summary.csv"
        if summary_csv.exists():
            s = pd.read_csv(summary_csv).iloc[0].to_dict()
            prof.at[idx, "rob_mean_robust_nc"] = float(s.get("mean_robust_nc", 0.0))
            prof.at[idx, "rob_min_robust_nc"] = float(s.get("min_robust_nc", 0.0))
            prof.at[idx, "rob_mean_robust_ber"] = float(s.get("mean_robust_ber", 0.0))
            prof.at[idx, "rob_max_robust_ber"] = float(s.get("max_robust_ber", 0.0))
    prof.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return prof


def _standardize_x(df: pd.DataFrame, feature_cols: list[str], scaler: dict[str, Any]) -> np.ndarray:
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, -1)
    std = np.where(std < 1e-6, 1.0, std)
    if mean.shape[1] != x.shape[1]:
        raise RuntimeError(f"scaler dim mismatch: mean={mean.shape}, x={x.shape}")
    return (x - mean) / std


def _zero_shot_router_eval(
    paired_table_csv: Path,
    base_moe_variant_dir: Path,
    out_dir: Path,
    device: str,
) -> dict[str, Any]:
    import torch

    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(paired_table_csv)
    nc_cols = [c for c in df.columns if c.startswith("nc__")]
    if not nc_cols:
        raise RuntimeError("paired table has no nc__ columns")
    role_order = [c.replace("nc__", "") for c in nc_cols]
    nc = df[nc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    model_files = sorted(base_moe_variant_dir.glob("fold_*_router_net_v17_1.pt"))
    if not model_files:
        msg = f"RouterNet fold checkpoints not found: {base_moe_variant_dir}"
        print("[WARN]", msg, flush=True)
        write_json(out_dir / "router_zero_shot_missing.json", {"warning": msg})
        return {"router_missing": True, "warning": msg}

    probs_acc = None
    used_meta = []
    first_variant = None
    first_feature_cols = None

    for model_file in model_files:
        ckpt = torch.load(model_file, map_location="cpu", weights_only=False)
        variant = dict(ckpt["variant"])
        feature_cols = list(ckpt["feature_cols"])
        roles_ckpt = list(ckpt["roles"])
        if roles_ckpt != role_order:
            raise RuntimeError(f"role_order mismatch in {model_file}: ckpt={roles_ckpt}, table={role_order}")
        x = _standardize_x(df, feature_cols, dict(ckpt["scaler"]))
        model = RouterNet(
            input_dim=len(feature_cols),
            num_roles=len(role_order),
            hidden_dims=list(variant.get("hidden_dims", [128, 64])),
            dropout=float(variant.get("dropout", 0.10)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        sigmoid_probs, _ = _predict_torch(model, x, device=device, temperature=float(variant.get("temperature", 0.80)))
        probs_acc = sigmoid_probs if probs_acc is None else probs_acc + sigmoid_probs
        used_meta.append({"model_file": str(model_file), "variant": variant})
        first_variant = variant
        first_feature_cols = feature_cols

    probs = probs_acc / float(len(model_files))
    variant = first_variant or {}
    selected, used_ks, gap, ent = _top_indices(
        probs,
        top_k=int(variant.get("top_k", 2)),
        adaptive=bool(variant.get("adaptive", False)),
        adaptive_max_k=int(variant.get("adaptive_max_k", 3)),
        confidence_gap=float(variant.get("confidence_gap", 0.16)),
        entropy_threshold=float(variant.get("entropy_threshold", 0.76)),
    )
    selected = _apply_protected_roles(
        selected=selected,
        probs=probs,
        role_order=role_order,
        protected_roles=list(variant.get("protected_roles", [])),
        protect_when_uncertain=bool(variant.get("protect_when_uncertain", True)),
        gap=gap,
        ent=ent,
        confidence_gap=float(variant.get("confidence_gap", 0.16)),
        entropy_threshold=float(variant.get("entropy_threshold", 0.76)),
        max_k=max(
            int(variant.get("top_k", 2)),
            int(variant.get("adaptive_max_k", 3)),
            int(variant.get("top_k", 2)) + len(list(variant.get("protected_roles", []))),
        ),
    )
    scores, roles, used_ks, weight_strings = _fuse_scores_for_indices(
        probs=probs,
        nc=nc,
        selected=selected,
        role_order=role_order,
        fusion_mode=str(variant.get("fusion_mode", "max")),
        alpha=8.0,
        router_power=1.0,
    )

    out = df.copy()
    out["router_zero_shot_score_nc"] = scores
    out["router_roles"] = roles
    out["router_weights"] = weight_strings
    out["router_used_k"] = used_ks
    out["router_top1_role"] = [role_order[int(i)] for i in probs.argmax(axis=1)]
    out["router_top1_prob"] = probs.max(axis=1)
    out["oracle_top1_role"] = [role_order[int(i)] for i in nc.argmax(axis=1)]
    out["oracle_top1_nc"] = nc.max(axis=1)
    for j, role in enumerate(role_order):
        out[f"router_prob__{role}"] = probs[:, j]

    rows_csv = out_dir / "mixed_router_zero_shot_rows_v17_2.csv"
    out.to_csv(rows_csv, index=False, encoding="utf-8-sig")

    attack_df = _attack_summary(out, "router_zero_shot_score_nc")
    attack_csv = out_dir / "mixed_router_zero_shot_by_attack_v17_2.csv"
    attack_df.to_csv(attack_csv, index=False, encoding="utf-8-sig")

    summary = {
        **_summary_from_scores(np.asarray(scores, dtype=np.float32), "router_zero_shot"),
        "variant": variant,
        "num_fold_models": len(model_files),
        "role_order": role_order,
        "feature_cols": first_feature_cols,
        "rows_csv": str(rows_csv),
        "attack_csv": str(attack_csv),
        "model_files": [str(p) for p in model_files],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "mixed_router_zero_shot_summary_v17_2.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "mixed_router_zero_shot_summary_v17_2.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", required=True)
    ap.add_argument("--profile_csv", required=True)
    ap.add_argument("--selected_json", required=True)
    ap.add_argument("--mixed_test_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_moe_variant_dir", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bit_length", type=int, default=256)
    ap.add_argument("--threshold_mode", default="mean")
    ns = ap.parse_args()

    suite_root = Path(ns.suite_root)
    profile_csv = Path(ns.profile_csv)
    selected_json = Path(ns.selected_json)
    mixed_test_root = Path(ns.mixed_test_root)
    out_dir = ensure_dir(ns.output_dir)
    evals_dir = ensure_dir(out_dir / "evals")

    selected_models = _load_selected_models(selected_json)
    print("[SELECTED MODELS]", selected_models, flush=True)

    for model in selected_models:
        ckpt = _ckpt_for_model(suite_root, model)
        rob_out = ensure_dir(evals_dir / f"robustness_{model}")
        print(f"[EVAL ROBUST] model={model} ckpt={ckpt}", flush=True)
        evaluate_robustness({
            "dataset_root": str(mixed_test_root),
            "ckpt_path": str(ckpt),
            "output_dir": str(rob_out),
            "device": ns.device,
            "bit_length": ns.bit_length,
            "threshold_mode": ns.threshold_mode,
        })

    mixed_profile_csv = out_dir / "mixed_model_capability_profile_reused_unique_v17_2.csv"
    _make_mixed_profile(profile_csv, selected_models, out_dir, mixed_profile_csv)

    static_dir = ensure_dir(out_dir / "mixed_static_gated_eval")
    static_summary = evaluate_specialist_ensemble_v15(
        profile_csv=mixed_profile_csv,
        selected_specialists_json=selected_json,
        output_dir=static_dir,
    )

    moe_table_dir = ensure_dir(out_dir / "mixed_sparse_moe_table")
    table_info = build_sparse_moe_table(
        profile_csv=mixed_profile_csv,
        selected_json=selected_json,
        output_dir=moe_table_dir,
        use_descriptors=True,
    )
    paired_csv = moe_table_dir / "sparse_moe_expert_table_paired_v17_2.csv"
    enrich_paired_descriptors(table_info["expert_table_csv"], paired_csv)

    oracle_dir = ensure_dir(out_dir / "mixed_oracle_static_eval")
    oracle_summary = evaluate_oracle_and_static(paired_csv, oracle_dir, top_k=2)

    router_summary = {}
    if ns.base_moe_variant_dir:
        try:
            router_summary = _zero_shot_router_eval(
                paired_table_csv=paired_csv,
                base_moe_variant_dir=Path(ns.base_moe_variant_dir),
                out_dir=ensure_dir(out_dir / "mixed_router_zero_shot_eval"),
                device=ns.device if ns.device != "auto" else "cpu",
            )
        except Exception:
            traceback.print_exc()
            router_summary = {"router_error": traceback.format_exc()}
            write_json(out_dir / "mixed_router_zero_shot_error.json", router_summary)

    final = {
        "selected_models": selected_models,
        "mixed_profile_csv": str(mixed_profile_csv),
        "static_summary": static_summary,
        "table_info": table_info,
        "oracle_summary": oracle_summary,
        "router_summary": router_summary,
    }
    write_json(out_dir / "mixed_zero_shot_final_summary_v17_2.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2)[:6000], flush=True)


if __name__ == "__main__":
    main()
