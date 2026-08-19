#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 command alias for topology/boundary dataset extension.

The implementation is inherited from the V13 V10-base extension script; V14 adds
formal splitting and updated attack config, so this wrapper keeps command names
consistent for the formal-split pipeline.
"""
from __future__ import annotations

from rb_afl_system.scripts.extend_dataset_attacks_V13 import extend_dataset_attacks, main

__all__ = ["extend_dataset_attacks", "main"]


if __name__ == "__main__":
    main()
