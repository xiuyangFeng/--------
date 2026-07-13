#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated shim — use ``python -m training_wss_min.tools.make_configs_xyz_scale`` instead."""
from __future__ import annotations

import runpy
import sys
import warnings

warnings.warn(
    "training_wss_min.make_configs_round6_scale is deprecated; use training_wss_min.tools.make_configs_xyz_scale",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.argv[0] = "training_wss_min.tools.make_configs_xyz_scale"
    runpy.run_module("training_wss_min.tools.make_configs_xyz_scale", run_name="__main__", alter_sys=True)
