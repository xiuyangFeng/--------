#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated shim — use ``python -m training_wss_min.tools.visualize_sampling`` instead."""
from __future__ import annotations

import runpy
import sys
import warnings

warnings.warn(
    "training_wss_min.visualize_sampling is deprecated; use training_wss_min.tools.visualize_sampling",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.argv[0] = "training_wss_min.tools.visualize_sampling"
    runpy.run_module("training_wss_min.tools.visualize_sampling", run_name="__main__", alter_sys=True)
