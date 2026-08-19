"""Paper-aligned GeoMoE-ZW router training, registration and verification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rb_afl_system.data.dataset.triplet_dataset import _load_sample
from rb_afl_system.engine.checkpoint_io import load_checkpoint
from rb_afl_system.engine.device import resolve_device, resolve_map_location
from rb_afl_system.models.model_registry import build_generator
from rb_afl_system.paper.descriptor import paired_descriptor_128
from rb_afl_system.paper.registry import ExpertProfile, select_registration_experts
from rb_afl_system.paper.router import RouterNet, make_oracle_multilabel_targets, router_loss, top_k_experts
from rb_afl_system.utils import ensure_dir, write_json
from rb_afl_system.watermark.copyright_image import load_copyright_bits
from rb_afl_system.watermark.feature_to_bits import feature_to_bits
from rb_afl_system.watermark.metrics import ber_score, nc_score
from rb_afl_system.watermark.zero_watermark import recover_watermark, register_zero_watermark


class _Expert:
    def __init__(self, name: str, checkpoint: Path, device: torch.device):
        ckpt = load_checkpoint(checkpoint, map_location=resolve_map_location(str(device)))
        model_cfg = dict(ckpt.get("config", {}))
        self.channels = list(ckpt.get("channel_names", model_cfg.get("channels", ["occ", "dist", "orient", "density"])))
        model_cfg.update({"generator": ckpt.get("generator_name", model_cfg.get("generator", "cnn_baseline")), "in_channels": len(self.channels)})
        self.name = name
        self.model = build_generator(model_cfg).to(device)
        self.model.load_state_dict(ckpt["generator"])
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def bits(self, sample_dir: str | Path, bit_length: int, threshold_mode: str) -> np.ndarray:
        sample = _load_sample(sample_dir, self.channels)
        sample = {key: (value.unsqueeze(0).to(self.device) if torch.is_tensor(value) else value) for key, value in sample.items()}
        embedding = self.model(
            grid=sample["grid"], tokens=sample["tokens"], token_mask=sample["token_mask"],
            graph_nodes=sample["graph_nodes"], graph_adj=sample["graph_adj"], graph_mask=sample["graph_mask"],
        ).detach().cpu().numpy()[0]
        return feature_to_bits(embedding, bit_length=bit_length, threshold_mode=threshold_mode)


def _load_experts(model_names: list[str], model_root: Path, device: torch.device) -> dict[str, _Expert]:
    experts: dict[str, _Expert] = {}
    for name in model_names:
        checkpoint = model_root / name / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing expert checkpoint: {checkpoint}")
        experts[name] = _Expert(name, checkpoint, device)
    return experts


def _base_by_identity(metadata: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for identity, group in metadata.groupby("identity"):
        bases = group[group["attack_type"].astype(str) == "base"]
        if not bases.empty:
            result[str(identity)] = bases.iloc[0].to_dict()
    return result


def _build_training_table(
    metadata: pd.DataFrame,
    experts: dict[str, _Expert],
    copyright_bits: np.ndarray,
    bit_length: int,
    threshold_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    names = list(experts)
    bases = _base_by_identity(metadata)
    base_bits = {
        identity: {name: expert.bits(row["sample_dir"], bit_length, threshold_mode) for name, expert in experts.items()}
        for identity, row in bases.items()
    }
    descriptors: list[np.ndarray] = []
    scores: list[list[float]] = []
    attack_types: list[str] = []
    for _, row in metadata[metadata["attack_type"].astype(str) != "base"].iterrows():
        identity = str(row["identity"])
        if identity not in bases:
            continue
        descriptors.append(paired_descriptor_128(bases[identity]["sample_dir"], row["sample_dir"]))
        row_scores: list[float] = []
        for name, expert in experts.items():
            registered = register_zero_watermark(copyright_bits, base_bits[identity][name])
            query_bits = expert.bits(row["sample_dir"], bit_length, threshold_mode)
            row_scores.append(nc_score(copyright_bits, recover_watermark(registered, query_bits)))
        scores.append(row_scores)
        attack_types.append(str(row["attack_type"]))
    if not descriptors:
        raise ValueError("No attacked training rows were available for RouterNet")
    return np.stack(descriptors), np.asarray(scores, dtype=np.float32), attack_types, names


def _select_profiles(scores: np.ndarray, attacks: list[str], names: list[str], threshold: float) -> list[str]:
    profiles: list[ExpertProfile] = []
    attack_array = np.asarray(attacks)
    for index, name in enumerate(names):
        direction_scores = {
            attack: float(scores[attack_array == attack, index].mean()) for attack in sorted(set(attacks))
        }
        profiles.append(ExpertProfile(name=name, attack_nc=direction_scores))
    return [profile.name for profile in select_registration_experts(profiles, threshold=threshold, minimum=3)]


def _train_router(
    descriptors: np.ndarray,
    scores: np.ndarray,
    selected_indices: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    delta: float,
    device: torch.device,
    seed: int,
) -> tuple[RouterNet, np.ndarray, np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = descriptors.mean(axis=0).astype(np.float32)
    std = descriptors.std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    x = torch.as_tensor((descriptors - mean) / std, dtype=torch.float32, device=device)
    nc = torch.as_tensor(scores[:, selected_indices], dtype=torch.float32, device=device)
    targets = make_oracle_multilabel_targets(nc, delta=delta)
    model = RouterNet(input_dim=128, num_experts=len(selected_indices), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        logits = model(x)
        loss, parts = router_loss(logits, targets, nc, quality_weight=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % 50 == 0:
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), **{key: float(value.cpu()) for key, value in parts.items()}})
    model.eval()
    return model, mean, std, history


def train_paper_router(config: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])
    model_names = list(config["model_names"])
    experts = _load_experts(model_names, Path(config["model_root"]), device)
    metadata = pd.read_csv(Path(config["train_dataset_root"]) / "metadata.csv")
    copyright_bits = load_copyright_bits(
        config["copyright_image_path"], int(config.get("bit_length", 256)),
        int(config.get("copyright_image_threshold", 128)), int(config.get("arnold_iterations", 0)),
    )
    descriptors, scores, attacks, names = _build_training_table(
        metadata, experts, copyright_bits, int(config.get("bit_length", 256)), str(config.get("threshold_mode", "mean"))
    )
    selected_names = _select_profiles(scores, attacks, names, float(config.get("nc_threshold", 0.80)))
    selected_indices = [names.index(name) for name in selected_names]
    model, mean, std, history = _train_router(
        descriptors, scores, selected_indices, int(config.get("router_epochs", 800)),
        float(config.get("router_lr", 0.002)), float(config.get("router_weight_decay", 0.0001)),
        float(config.get("oracle_margin", 0.02)), device, int(config.get("seed", 20260318)),
    )
    checkpoint = output_dir / "paper_router.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "selected_experts": selected_names,
        "descriptor_mean": mean, "descriptor_std": std, "history": history,
        "config": config,
    }, checkpoint)
    write_json(output_dir / "paper_router_summary.json", {
        "checkpoint": str(checkpoint), "selected_experts": selected_names,
        "training_rows": int(len(descriptors)), "history": history,
    })
    return {"checkpoint": str(checkpoint), "selected_experts": selected_names}


def evaluate_paper_moe(config: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])
    router_ckpt = torch.load(config["router_checkpoint"], map_location="cpu", weights_only=False)
    selected_names = list(router_ckpt["selected_experts"])
    experts = _load_experts(selected_names, Path(config["model_root"]), device)
    router = RouterNet(128, len(selected_names), 0.1).to(device)
    router.load_state_dict(router_ckpt["model_state_dict"])
    router.eval()
    mean = np.asarray(router_ckpt["descriptor_mean"], dtype=np.float32)
    std = np.asarray(router_ckpt["descriptor_std"], dtype=np.float32)
    metadata = pd.read_csv(Path(config["dataset_root"]) / "metadata.csv")
    bases = _base_by_identity(metadata)
    bit_length = int(config.get("bit_length", 256))
    threshold_mode = str(config.get("threshold_mode", "mean"))
    nc_threshold = float(config.get("nc_threshold", 0.80))
    top_k = int(config.get("top_k", 2))
    copyright_bits = load_copyright_bits(
        config["copyright_image_path"], bit_length, int(config.get("copyright_image_threshold", 128)),
        int(config.get("arnold_iterations", 0)),
    )
    base_bits = {
        identity: {name: expert.bits(row["sample_dir"], bit_length, threshold_mode) for name, expert in experts.items()}
        for identity, row in bases.items()
    }
    registered = {
        identity: {name: register_zero_watermark(copyright_bits, bits) for name, bits in expert_bits.items()}
        for identity, expert_bits in base_bits.items()
    }

    def route(reference_dir: str, query_dir: str) -> list[str]:
        descriptor = (paired_descriptor_128(reference_dir, query_dir) - mean) / std
        with torch.inference_mode():
            logits = router(torch.as_tensor(descriptor, dtype=torch.float32, device=device).reshape(1, -1))
            _, indices = top_k_experts(logits, top_k)
        return [selected_names[int(index)] for index in indices[0].cpu().tolist()]

    rows: list[dict[str, Any]] = []
    for _, row in metadata[metadata["attack_type"].astype(str) != "base"].iterrows():
        identity = str(row["identity"])
        if identity not in bases:
            continue
        chosen = route(bases[identity]["sample_dir"], row["sample_dir"])
        candidates = []
        for name in chosen:
            query_bits = experts[name].bits(row["sample_dir"], bit_length, threshold_mode)
            recovered = recover_watermark(registered[identity][name], query_bits)
            candidates.append((nc_score(copyright_bits, recovered), ber_score(copyright_bits, recovered), name))
        best_nc, best_ber, best_name = max(candidates, key=lambda item: item[0])
        rows.append({
            "identity": identity, "attack_type": str(row["attack_type"]), "attack_name": str(row.get("attack_name", "")),
            "selected_experts": ",".join(chosen), "best_expert": best_name,
            "watermark_nc": best_nc, "watermark_ber": best_ber, "accepted": best_nc >= nc_threshold,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No attacked rows were evaluated")
    detail.to_csv(output_dir / "geomoe_robustness_rows.csv", index=False, encoding="utf-8-sig")
    attack_summary = detail.groupby("attack_type").agg(
        mean_nc=("watermark_nc", "mean"), min_nc=("watermark_nc", "min"),
        mean_ber=("watermark_ber", "mean"), rpr=("accepted", "mean"), count=("identity", "count"),
    ).reset_index()
    attack_summary.to_csv(output_dir / "geomoe_by_attack.csv", index=False, encoding="utf-8-sig")

    false_rows: list[dict[str, Any]] = []
    identities = sorted(bases)
    for registered_identity in identities:
        for query_identity in identities:
            if registered_identity == query_identity:
                continue
            chosen = route(bases[registered_identity]["sample_dir"], bases[query_identity]["sample_dir"])
            scores = []
            for name in chosen:
                query_bits = base_bits[query_identity][name]
                recovered = recover_watermark(registered[registered_identity][name], query_bits)
                scores.append((nc_score(copyright_bits, recovered), name))
            score, name = max(scores, key=lambda item: item[0])
            false_rows.append({
                "registered_identity": registered_identity, "query_identity": query_identity,
                "selected_experts": ",".join(chosen), "best_expert": name,
                "cross_nc": score, "false_match": score >= nc_threshold,
            })
    false_df = pd.DataFrame(false_rows)
    false_df.to_csv(output_dir / "geomoe_cross_identity_rows.csv", index=False, encoding="utf-8-sig")
    summary = {
        "selected_experts": selected_names, "top_k": min(top_k, len(selected_names)),
        "num_robust_rows": int(len(detail)), "mean_nc": float(detail["watermark_nc"].mean()),
        "min_nc": float(detail["watermark_nc"].min()),
        "nc_below_0_80": int((detail["watermark_nc"] < nc_threshold).sum()),
        "rpr_at_0_80": float(detail["accepted"].mean()),
        "num_cross_identity_pairs": int(len(false_df)),
        "max_cross_identity_nc": float(false_df["cross_nc"].max()) if not false_df.empty else 0.0,
        "fmr_at_0_80": float(false_df["false_match"].mean()) if not false_df.empty else 0.0,
        "nc_threshold": nc_threshold,
    }
    write_json(output_dir / "geomoe_summary.json", summary)
    with pd.ExcelWriter(output_dir / "GeoMoE_ZW_paper_results.xlsx") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        attack_summary.to_excel(writer, sheet_name="by_attack", index=False)
        detail.to_excel(writer, sheet_name="robustness_rows", index=False)
        false_df.to_excel(writer, sheet_name="cross_identity", index=False)
    return summary
