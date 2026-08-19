"""End-to-end registration and Top-2 verification protocol for GeoMoE-ZW."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from rb_afl_system.watermark.feature_to_bits import feature_to_bits
from rb_afl_system.watermark.metrics import nc_score
from rb_afl_system.watermark.zero_watermark import recover_watermark, register_zero_watermark

from .registry import ExpertProfile, GeoMoERegistry, select_registration_experts
from .router import RouterNet, top_k_experts

FeatureExtractor = Callable[[object], np.ndarray]


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    nc: float
    selected_experts: tuple[str, ...]
    recovered_bits: np.ndarray
    per_expert_nc: dict[str, float]


class GeoMoEZWProtocol:
    def __init__(self, router: RouterNet, bit_length: int = 256, top_k: int = 2, nc_threshold: float = 0.80):
        self.router = router
        self.bit_length = int(bit_length)
        self.top_k = int(top_k)
        self.nc_threshold = float(nc_threshold)

    def register(
        self,
        data_id: str,
        owner: str,
        sample: object,
        base_descriptor_32: np.ndarray,
        copyright_bits: np.ndarray,
        profiles: list[ExpertProfile],
        extractors: dict[str, FeatureExtractor],
    ) -> GeoMoERegistry:
        selected = select_registration_experts(profiles, threshold=self.nc_threshold, minimum=3)
        copyright_arr = np.asarray(copyright_bits, dtype=np.uint8).reshape(-1)
        if copyright_arr.size != self.bit_length:
            raise ValueError(f"Expected {self.bit_length} copyright bits")
        zero_watermarks: dict[str, list[int]] = {}
        for profile in selected:
            if profile.name not in extractors:
                raise KeyError(f"Missing feature extractor for {profile.name}")
            bits = feature_to_bits(extractors[profile.name](sample), bit_length=self.bit_length)
            zero_watermarks[profile.name] = register_zero_watermark(copyright_arr, bits).tolist()
        registry = GeoMoERegistry(
            data_id=data_id,
            owner=owner,
            registered_descriptor=np.asarray(base_descriptor_32, dtype=np.float32).reshape(-1).tolist(),
            copyright_bits=copyright_arr.tolist(),
            expert_zero_watermarks=zero_watermarks,
            expert_profiles={profile.name: profile.attack_nc for profile in selected},
            bit_length=self.bit_length,
            nc_threshold=self.nc_threshold,
        )
        registry.validate()
        return registry

    @torch.inference_mode()
    def verify(
        self,
        query_sample: object,
        paired_descriptor: np.ndarray,
        registry: GeoMoERegistry,
        extractors: dict[str, FeatureExtractor],
        device: str = "cpu",
    ) -> VerificationResult:
        registry.validate()
        names = list(registry.expert_zero_watermarks)
        x = torch.as_tensor(paired_descriptor, dtype=torch.float32, device=device).reshape(1, -1)
        self.router.to(device).eval()
        logits = self.router(x)
        if logits.shape[1] != len(names):
            raise ValueError("Router output dimension does not match registered expert count")
        _, indices = top_k_experts(logits, self.top_k)
        chosen = tuple(names[int(index)] for index in indices[0].cpu().tolist())
        target = registry.copyright_array()
        candidates: list[tuple[float, str, np.ndarray]] = []
        for name in chosen:
            if name not in extractors:
                raise KeyError(f"Missing feature extractor for {name}")
            feature_bits = feature_to_bits(extractors[name](query_sample), bit_length=self.bit_length)
            recovered = recover_watermark(np.asarray(registry.expert_zero_watermarks[name], dtype=np.uint8), feature_bits)
            candidates.append((nc_score(target, recovered), name, recovered))
        best_nc, _, best_bits = max(candidates, key=lambda item: item[0])
        return VerificationResult(
            accepted=bool(best_nc >= registry.nc_threshold),
            nc=float(best_nc),
            selected_experts=chosen,
            recovered_bits=best_bits,
            per_expert_nc={name: float(score) for score, name, _ in candidates},
        )
