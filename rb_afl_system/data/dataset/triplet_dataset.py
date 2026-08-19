#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset and collate functions for RB-AFL training.

V11 adds hard-negative and hard-positive sampling hooks.  The trainer updates
these maps online from the current embedding space; the dataset remains a normal
PyTorch Dataset and keeps backward compatibility with V10 configs.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from rb_afl_system.data.channels.channel_builder import CHANNEL_NAMES, select_channels
from rb_afl_system.data.features.topology_graph import load_graph
from rb_afl_system.data.features.vector_tokens import load_tokens


def _sample_paths(sample_dir: str | Path) -> dict:
    d = Path(sample_dir)
    return {"grid": d / "grid.npy", "tokens": d / "tokens.npz", "graph": d / "graph.npz", "metadata": d / "metadata.json"}


def _load_sample(sample_dir: str | Path, channels: List[str]) -> dict:
    paths = _sample_paths(sample_dir)
    grid4 = np.load(paths["grid"]).astype(np.float32)
    grid = select_channels(grid4, channels)
    tokens, token_mask = load_tokens(str(paths["tokens"]))
    nodes, adj, graph_mask = load_graph(str(paths["graph"]))
    return {
        "grid": torch.from_numpy(grid),
        "tokens": torch.from_numpy(tokens),
        "token_mask": torch.from_numpy(token_mask),
        "graph_nodes": torch.from_numpy(nodes),
        "graph_adj": torch.from_numpy(adj),
        "graph_mask": torch.from_numpy(graph_mask),
    }


class IdentityTripletDataset(Dataset):
    """Triplet dataset using shp-level attack samples as positives.

    metadata.csv must contain columns: identity, sample, sample_dir, attack_type.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        channels: List[str] | None = None,
        positive_mode: str = "shp_aug",
        seed: int = 20260318,
        hard_negative_prob: float = 0.0,
        hard_positive_prob: float = 0.0,
        positive_attack_keywords: List[str] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.channels = channels or list(CHANNEL_NAMES)
        self.positive_mode = positive_mode
        self.rng = random.Random(seed)
        self.hard_negative_prob = float(hard_negative_prob)
        self.hard_positive_prob = float(hard_positive_prob)
        self.positive_attack_keywords = [str(x).lower() for x in (positive_attack_keywords or []) if str(x).strip()]
        self.hard_negative_map: dict[str, list[str]] = {}
        self.hard_positive_map: dict[str, list[str]] = {}

        meta = self.dataset_root / "metadata.csv"
        if not meta.is_file():
            raise FileNotFoundError(str(meta))
        import pandas as pd

        df = pd.read_csv(meta)
        required = {"identity", "sample", "sample_dir", "attack_type"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")
        self.rows = df.to_dict("records")
        self.by_identity: dict[str, list[dict]] = defaultdict(list)
        self.by_sample_dir: dict[str, dict] = {}
        for row in self.rows:
            row["identity"] = str(row["identity"])
            row["sample_dir"] = str(row["sample_dir"])
            self.by_identity[row["identity"]].append(row)
            self.by_sample_dir[row["sample_dir"]] = row
        self.identities = sorted(self.by_identity)
        if len(self.identities) < 2:
            raise ValueError("Triplet training requires at least 2 identities")
        self.anchor_rows = [row for row in self.rows if str(row.get("attack_type")) == "base"]
        if not self.anchor_rows:
            self.anchor_rows = list(self.rows)

    def __len__(self) -> int:
        return len(self.anchor_rows)

    def infer_dims(self) -> dict[str, int]:
        sample = _load_sample(self.rows[0]["sample_dir"], self.channels)
        return {
            "in_channels": int(sample["grid"].shape[0]),
            "token_dim": int(sample["tokens"].shape[1]),
            "node_dim": int(sample["graph_nodes"].shape[1]),
            "grid_size": int(sample["grid"].shape[-1]),
        }

    def set_hard_maps(
        self,
        hard_negative_map: dict[str, list[str]] | None = None,
        hard_positive_map: dict[str, list[str]] | None = None,
        hard_negative_prob: float | None = None,
        hard_positive_prob: float | None = None,
    ) -> None:
        if hard_negative_map is not None:
            self.hard_negative_map = {str(k): [str(x) for x in v] for k, v in hard_negative_map.items()}
        if hard_positive_map is not None:
            self.hard_positive_map = {str(k): [str(x) for x in v] for k, v in hard_positive_map.items()}
        if hard_negative_prob is not None:
            self.hard_negative_prob = float(hard_negative_prob)
        if hard_positive_prob is not None:
            self.hard_positive_prob = float(hard_positive_prob)

    def _choose_positive(self, identity: str, anchor_row: dict) -> dict:
        if self.hard_positive_map and self.rng.random() < self.hard_positive_prob:
            sample_dirs = self.hard_positive_map.get(identity, [])
            valid = [self.by_sample_dir[p] for p in sample_dirs if p in self.by_sample_dir and p != anchor_row["sample_dir"]]
            if valid:
                return self.rng.choice(valid)

        candidates = [r for r in self.by_identity[identity] if r["sample_dir"] != anchor_row["sample_dir"]]
        if self.positive_attack_keywords:
            preferred = []
            for r in candidates:
                attack_text = str(r.get("attack_type", "")).lower()
                if any(k in attack_text for k in self.positive_attack_keywords):
                    preferred.append(r)
            if preferred:
                return self.rng.choice(preferred)
        if self.positive_mode == "shp_aug":
            aug = [r for r in candidates if str(r.get("attack_type")) != "base"]
            if aug:
                return self.rng.choice(aug)
            if candidates:
                return self.rng.choice(candidates)
            return anchor_row
        if self.positive_mode == "base_pair":
            return self.rng.choice(candidates) if candidates else anchor_row
        raise ValueError(f"Unsupported positive_mode: {self.positive_mode}")

    def _choose_negative(self, identity: str) -> dict:
        if self.hard_negative_map and self.rng.random() < self.hard_negative_prob:
            hard_ids = [x for x in self.hard_negative_map.get(identity, []) if x != identity and x in self.by_identity]
            if hard_ids:
                neg_id = self.rng.choice(hard_ids)
                base_rows = [r for r in self.by_identity[neg_id] if str(r.get("attack_type")) == "base"]
                return self.rng.choice(base_rows or self.by_identity[neg_id])

        other_ids = [x for x in self.identities if x != identity]
        neg_id = self.rng.choice(other_ids)
        base_rows = [r for r in self.by_identity[neg_id] if str(r.get("attack_type")) == "base"]
        return self.rng.choice(base_rows or self.by_identity[neg_id])

    def __getitem__(self, idx: int) -> dict:
        anchor_row = self.anchor_rows[idx]
        identity = str(anchor_row["identity"])
        positive_row = self._choose_positive(identity, anchor_row)
        negative_row = self._choose_negative(identity)
        return {
            "anchor": _load_sample(anchor_row["sample_dir"], self.channels),
            "positive": _load_sample(positive_row["sample_dir"], self.channels),
            "negative": _load_sample(negative_row["sample_dir"], self.channels),
            "identity": identity,
            "positive_attack": str(positive_row.get("attack_type", "unknown")),
            "positive_sample_dir": str(positive_row.get("sample_dir", "")),
            "negative_identity": str(negative_row["identity"]),
        }


def _pad_2d(seq: list[torch.Tensor], pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    max_n = max(int(x.shape[0]) for x in seq)
    feat_dim = int(seq[0].shape[1])
    out = torch.full((len(seq), max_n, feat_dim), float(pad_value), dtype=seq[0].dtype)
    mask = torch.zeros((len(seq), max_n), dtype=torch.bool)
    for i, x in enumerate(seq):
        n = int(x.shape[0])
        out[i, :n] = x
        mask[i, :n] = True
    return out, mask


def _pad_adj(seq: list[torch.Tensor]) -> torch.Tensor:
    max_n = max(int(x.shape[0]) for x in seq)
    out = torch.zeros((len(seq), max_n, max_n), dtype=seq[0].dtype)
    for i, x in enumerate(seq):
        n = int(x.shape[0])
        out[i, :n, :n] = x
    return out


def collate_triplet(batch: list[dict]) -> dict:
    out: dict[str, Any] = {
        "identity": [b["identity"] for b in batch],
        "positive_attack": [b["positive_attack"] for b in batch],
        "positive_sample_dir": [b.get("positive_sample_dir", "") for b in batch],
        "negative_identity": [b.get("negative_identity", "") for b in batch],
    }
    for key in ["anchor", "positive", "negative"]:
        grids = torch.stack([b[key]["grid"] for b in batch], dim=0)
        tokens, token_mask = _pad_2d([b[key]["tokens"] for b in batch])
        nodes, graph_mask = _pad_2d([b[key]["graph_nodes"] for b in batch])
        adj = _pad_adj([b[key]["graph_adj"] for b in batch])
        out[key] = {
            "grid": grids,
            "tokens": tokens,
            "token_mask": token_mask,
            "graph_nodes": nodes,
            "graph_adj": adj,
            "graph_mask": graph_mask,
        }
    return out
