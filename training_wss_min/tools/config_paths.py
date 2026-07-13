#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map historical ExpConfig.name -> configs/ theme-relative path.

Filenames are semantic; JSON ``name`` (and thus runs/<name>/) stay historical.
"""

from __future__ import annotations

from pathlib import Path

from training_wss_min import config as C

CONFIG_ROOT = C.PROJECT_ROOT / "training_wss_min" / "configs"
SWEEPS_DIR = CONFIG_ROOT / "sweeps"


def config_relpath(name: str) -> str:
    """Path relative to ``configs/`` for a given run ``name``."""
    if name.startswith("r2_"):
        return f"loss_aug_ablation/{name[3:]}.json"
    if name.startswith("r3_"):
        return f"clean_data/{name[3:]}.json"
    if name.startswith("r4_dev1_pc_"):
        return f"pointcount_curve/{name[len('r4_dev1_'):]}.json"
    if name.startswith("r4_dev1_"):
        return f"protocol_gates/{name[len('r4_dev1_'):]}.json"
    if name.startswith("r5_lc_"):
        return f"fit_lc_diagnosis/lc/{name[3:]}.json"
    if name.startswith("r5_"):
        return f"fit_lc_diagnosis/{name[3:]}.json"
    if name.startswith("r6_scale_"):
        return f"xyz_scale_diag/{name[3:]}.json"  # scale_A_xyz_s1234.json
    if name.startswith("r6_press_"):
        return f"multitarget/{name[3:]}.json"  # press_wall_xyz_s1234.json
    if name.startswith("r6_l1_"):
        return f"loss_aug_ablation/{name[3:]}.json"  # l1_rawhuber_lam0p3_s1234.json
    if name.startswith("r6_w3_"):
        return f"round6_w3/{name[3:]}.json"  # w3_e_rawhuber_lam1_s1234.json
    if name.startswith("r6_"):
        return f"xyz_scale_diag/{name[3:]}.json"
    if (
        name.startswith("pc_")
        or name.startswith("feat_")
        or name.startswith("samp_")
    ):
        return f"baseline_sweep/{name}.json"
    # protocols / diagnostics already under fit_lc_diagnosis by theme
    if name.startswith(("a0", "a1", "brep_")):
        return f"fit_lc_diagnosis/{name}.json"
    return f"{name}.json"


def config_path(name: str) -> Path:
    return CONFIG_ROOT / config_relpath(name)


def repo_config_path(name: str) -> str:
    """Repo-relative path string for manifests."""
    return f"training_wss_min/configs/{config_relpath(name)}"
