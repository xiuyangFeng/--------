#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated shim — use ``python -m training_wss_min.experiments.a0d_simple_overfit`` instead."""
from __future__ import annotations

import runpy
import sys
import warnings

warnings.warn(
    "training_wss_min.a0d_simple_overfit is deprecated; use training_wss_min.experiments.a0d_simple_overfit",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.argv[0] = "training_wss_min.experiments.a0d_simple_overfit"
    runpy.run_module("training_wss_min.experiments.a0d_simple_overfit", run_name="__main__", alter_sys=True)
