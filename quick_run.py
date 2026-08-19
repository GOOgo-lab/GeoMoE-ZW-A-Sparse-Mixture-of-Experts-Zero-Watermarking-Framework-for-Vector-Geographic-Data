#!/usr/bin/env python3
"""CPU-only GeoMoE-ZW quick run using deterministic synthetic features.

This smoke demo validates the paper pipeline without vector files, PyTorch,
CUDA or trained checkpoints. It is not a substitute for the formal experiment.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np


BIT_LENGTH = 256
NC_THRESHOLD = 0.80
NUM_CANDIDATES = 10
ATTACK_DIRECTIONS = 12


@dataclass(frozen=True)
class Expert:
    name: str
    projection: np.ndarray
    worst_nc: float

    def bits(self, features: np.ndarray) -> np.ndarray:
        scores = self.projection @ features
        return (scores >= np.median(scores)).astype(np.uint8)


def nc_score(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1.0e-12 else 0.0


def paired_descriptor(base: np.ndarray, query: np.ndarray) -> np.ndarray:
    ratio = np.clip(query / np.maximum(np.abs(base), 1.0e-6), -1.0e3, 1.0e3)
    result = np.concatenate([query, base, np.abs(query - base), ratio]).astype(np.float32)
    if result.shape != (128,):
        raise RuntimeError(f"Expected a 128-D descriptor, got {result.shape}")
    return result


def select_experts(experts: list[Expert], threshold: float = 0.80, minimum: int = 3) -> list[Expert]:
    ranked = sorted(experts, key=lambda item: (-item.worst_nc, item.name))
    qualified = [item for item in ranked if item.worst_nc >= threshold]
    return ranked[: min(max(minimum, len(qualified)), len(ranked))]


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoMoE-ZW CPU quick smoke run")
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--attack-strength", type=float, default=0.035)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    base = rng.normal(size=32).astype(np.float32)
    query = base + rng.normal(scale=args.attack_strength, size=32).astype(np.float32)
    descriptor = paired_descriptor(base, query)
    copyright_bits = rng.integers(0, 2, size=BIT_LENGTH, dtype=np.uint8)

    experts: list[Expert] = []
    for index in range(NUM_CANDIDATES):
        projection = rng.normal(size=(BIT_LENGTH, 32)).astype(np.float32)
        simulated_profile = np.clip(
            rng.normal(loc=0.84 - 0.012 * index, scale=0.025, size=ATTACK_DIRECTIONS), 0.0, 1.0
        )
        experts.append(Expert(f"expert_{index + 1:02d}", projection, float(simulated_profile.min())))

    registered = select_experts(experts)
    zero_watermarks = {
        expert.name: np.bitwise_xor(copyright_bits, expert.bits(base)) for expert in registered
    }

    # Lightweight deterministic routing surrogate for a fast dependency-free check.
    router_weights = rng.normal(scale=0.05, size=(len(registered), 128)).astype(np.float32)
    capability_bias = np.asarray([expert.worst_nc for expert in registered], dtype=np.float32)
    route_scores = router_weights @ descriptor + capability_bias
    top2_indices = np.argsort(route_scores)[-min(2, len(registered)):][::-1]
    selected = [registered[int(index)] for index in top2_indices]

    candidates: list[tuple[float, str, np.ndarray]] = []
    for expert in selected:
        recovered = np.bitwise_xor(zero_watermarks[expert.name], expert.bits(query))
        candidates.append((nc_score(copyright_bits, recovered), expert.name, recovered))
    best_nc, best_expert, best_bits = max(candidates, key=lambda item: item[0])
    result = {
        "mode": "synthetic_cpu_smoke_test",
        "candidate_experts": NUM_CANDIDATES,
        "registered_experts": [expert.name for expert in registered],
        "registered_worst_nc": {expert.name: round(expert.worst_nc, 4) for expert in registered},
        "descriptor_dim": int(descriptor.size),
        "top_k": len(selected),
        "selected_experts": [expert.name for expert in selected],
        "per_expert_nc": {name: round(score, 6) for score, name, _ in candidates},
        "best_expert": best_expert,
        "best_nc": round(best_nc, 6),
        "ber": round(float(np.mean(best_bits != copyright_bits)), 6),
        "threshold": NC_THRESHOLD,
        "accepted": bool(best_nc >= NC_THRESHOLD),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if descriptor.size != 128 or len(selected) > 2 or best_bits.size != BIT_LENGTH:
        raise SystemExit("Quick-run structural validation failed")


if __name__ == "__main__":
    main()
