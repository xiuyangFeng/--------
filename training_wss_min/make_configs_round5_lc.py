#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated shim — use ``python -m training_wss_min.tools.make_configs_learning_curve`` instead."""
from __future__ import annotations

import runpy
import sys
import warnings

warnings.warn(
    "training_wss_min.make_configs_round5_lc is deprecated; use training_wss_min.tools.make_configs_learning_curve",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.argv[0] = "training_wss_min.tools.make_configs_learning_curve"
    runpy.run_module("training_wss_min.tools.make_configs_learning_curve", run_name="__main__", alter_sys=True)
