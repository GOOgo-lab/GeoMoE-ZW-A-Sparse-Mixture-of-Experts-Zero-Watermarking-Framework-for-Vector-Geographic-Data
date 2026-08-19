"""Paper-aligned GeoMoE-ZW registration and verification components."""

from .descriptor import DESCRIPTOR_DIM, paired_descriptor_128
from .protocol import GeoMoEZWProtocol, VerificationResult
from .registry import ExpertProfile, GeoMoERegistry, select_registration_experts
from .router import RouterNet, make_oracle_multilabel_targets, router_loss

__all__ = [
    "DESCRIPTOR_DIM",
    "ExpertProfile",
    "GeoMoERegistry",
    "GeoMoEZWProtocol",
    "RouterNet",
    "VerificationResult",
    "make_oracle_multilabel_targets",
    "paired_descriptor_128",
    "router_loss",
    "select_registration_experts",
]
