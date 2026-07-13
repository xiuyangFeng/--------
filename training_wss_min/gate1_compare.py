#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated shim — use ``training_wss_min.tools.gate1_compare``."""
from __future__ import annotations

import warnings

warnings.warn(
    "training_wss_min.gate1_compare is deprecated; use training_wss_min.tools.gate1_compare",
    DeprecationWarning,
    stacklevel=2,
)

from training_wss_min.tools.gate1_compare import *  # noqa: F401,F403
from training_wss_min.tools.gate1_compare import gate1, load_val  # noqa: F401

if __name__ == "__main__":
    from training_wss_min.tools.gate1_compare import main
    main()
