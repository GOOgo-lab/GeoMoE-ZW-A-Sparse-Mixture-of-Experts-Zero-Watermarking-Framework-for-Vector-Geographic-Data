from __future__ import annotations

import numpy as np
import torch

from rb_afl_system.paper.registry import ExpertProfile, select_registration_experts
from rb_afl_system.paper.router import RouterNet, make_oracle_multilabel_targets, router_loss, top_k_experts
from rb_afl_system.watermark.metrics import nc_score
from rb_afl_system.watermark.zero_watermark import xor_bits


def test_legacy_prefix_xor_and_xnor_nc() -> None:
    assert xor_bits(np.array([0, 1, 1]), np.array([1, 1])).tolist() == [1, 0]
    assert np.isclose(nc_score(np.array([0, 1]), np.array([0, 1, 1])), 1.0)
    assert np.isclose(nc_score(np.array([0, 1]), np.array([1, 0])), 0.0)
    assert np.isclose(nc_score(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])), 0.5)


def test_router_shape_targets_and_loss() -> None:
    model = RouterNet(input_dim=128, num_experts=4, dropout=0.0)
    x = torch.randn(3, 128)
    nc = torch.tensor([[0.90, 0.89, 0.70, 0.60], [0.81, 0.84, 0.83, 0.50], [0.75, 0.75, 0.75, 0.75]])
    targets = make_oracle_multilabel_targets(nc, delta=0.02)
    logits = model(x)
    loss, parts = router_loss(logits, targets, nc)
    assert logits.shape == (3, 4)
    assert targets[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert torch.isfinite(loss)
    assert set(parts) == {"classification", "quality"}
    _, indices = top_k_experts(logits, 2)
    assert indices.shape == (3, 2)


def test_registration_selection_rule() -> None:
    profiles = [
        ExpertProfile("a", {"rotate": 0.91, "jitter": 0.82}),
        ExpertProfile("b", {"rotate": 0.88, "jitter": 0.81}),
        ExpertProfile("c", {"rotate": 0.79, "jitter": 0.78}),
        ExpertProfile("d", {"rotate": 0.70, "jitter": 0.69}),
    ]
    selected = select_registration_experts(profiles, threshold=0.80, minimum=3)
    assert [item.name for item in selected] == ["a", "b", "c"]
    assert np.isclose(selected[0].worst_nc, 0.82)
