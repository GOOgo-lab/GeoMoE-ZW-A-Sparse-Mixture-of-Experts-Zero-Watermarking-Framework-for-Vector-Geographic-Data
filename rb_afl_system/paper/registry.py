"""Capability profiling and serializable zero-watermark registration records."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExpertProfile:
    name: str
    attack_nc: dict[str, float]

    @property
    def worst_nc(self) -> float:
        return min(self.attack_nc.values()) if self.attack_nc else 0.0


def select_registration_experts(
    profiles: list[ExpertProfile], threshold: float = 0.80, minimum: int = 3
) -> list[ExpertProfile]:
    """Keep all experts above the worst-direction threshold, but at least three."""
    if not profiles:
        raise ValueError("At least one candidate expert is required")
    ranked = sorted(profiles, key=lambda item: (-item.worst_nc, item.name))
    selected = [item for item in ranked if item.worst_nc >= float(threshold)]
    return ranked[: min(max(int(minimum), len(selected)), len(ranked))]


@dataclass
class GeoMoERegistry:
    data_id: str
    owner: str
    registered_descriptor: list[float]
    copyright_bits: list[int]
    expert_zero_watermarks: dict[str, list[int]]
    expert_profiles: dict[str, dict[str, float]]
    bit_length: int = 256
    nc_threshold: float = 0.80

    def validate(self) -> None:
        if len(self.copyright_bits) != self.bit_length:
            raise ValueError("copyright_bits length does not match bit_length")
        if len(self.registered_descriptor) != 32:
            raise ValueError("registered_descriptor must contain 32 base features")
        if not self.expert_zero_watermarks:
            raise ValueError("registry contains no experts")
        for name, bits in self.expert_zero_watermarks.items():
            if len(bits) != self.bit_length:
                raise ValueError(f"zero-watermark length mismatch for expert {name}")

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GeoMoERegistry":
        value = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        value.validate()
        return value

    def copyright_array(self) -> np.ndarray:
        return np.asarray(self.copyright_bits, dtype=np.uint8)
