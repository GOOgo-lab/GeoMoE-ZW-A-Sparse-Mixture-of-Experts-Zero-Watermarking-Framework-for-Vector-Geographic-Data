#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper: V12 command now uses the V13 evaluator implementation."""
from rb_afl_system.scripts.specialist_ensemble_evaluator_V13 import *  # noqa: F401,F403
from rb_afl_system.scripts.specialist_ensemble_evaluator_V13 import main


if __name__ == "__main__":
    main()
