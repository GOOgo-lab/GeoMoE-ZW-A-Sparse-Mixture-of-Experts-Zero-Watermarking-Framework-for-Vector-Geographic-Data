#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast smoke test for model registry and zero-watermark utilities."""
from __future__ import annotations
import torch
import numpy as np
from rb_afl_system.models.model_registry import build_generator, build_discriminator
from rb_afl_system.watermark.feature_to_bits import feature_to_bits
from rb_afl_system.watermark.zero_watermark import make_random_copyright_bits, register_zero_watermark, recover_watermark, evaluate_recovery


def main() -> None:
    cfg = {"generator": "geovecformer_zw", "discriminator": "fc", "in_channels": 4, "token_dim": 12, "node_dim": 12, "feat_dim": 32, "branch_dim": 16, "disc_hidden_dim": 8}
    g = build_generator(cfg)
    d = build_discriminator(cfg)
    batch = 2
    z = g(
        torch.randn(batch, 4, 32, 32),
        torch.randn(batch, 5, 12),
        torch.ones(batch, 5, dtype=torch.bool),
        torch.randn(batch, 6, 12),
        torch.eye(6).repeat(batch, 1, 1),
        torch.ones(batch, 6, dtype=torch.bool),
    )
    logits = d(z)
    bits = feature_to_bits(z.detach().numpy()[0], bit_length=32)
    w = make_random_copyright_bits(32, seed=1)
    zw = register_zero_watermark(w, bits)
    rec = recover_watermark(zw, bits)
    metrics = evaluate_recovery(w, rec)
    if z.shape != (batch, 32):
        raise RuntimeError(f"Unexpected feature shape: {z.shape}")
    if logits.shape != (batch,):
        raise RuntimeError(f"Unexpected discriminator shape: {logits.shape}")
    if metrics["ber"] != 0.0:
        raise RuntimeError(f"Watermark recovery failed: {metrics}")
    print({"feature_shape": list(z.shape), "logit_shape": list(logits.shape), "watermark_metrics": metrics})


if __name__ == "__main__":
    main()
