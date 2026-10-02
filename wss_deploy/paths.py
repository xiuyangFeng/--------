from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = Path(os.environ.get("WSS_DEPLOY_RELEASE", PROJECT_ROOT / "outputs/wss_deploy_release/X5Dcap_asym2_v52d_3seed_20261002"))
# Retired releases (2026-10-02: the v5.1 X5D_v51 / M1_3head packages) live here, outside the release root: they are
# never listed, preloaded or offered for new work.  Finished jobs bound to them keep their files; the golden
# regression and report rebuilds may still read them (read-only).
RETIRED_RELEASE_ROOT = Path(os.environ.get("WSS_DEPLOY_RETIRED_RELEASE_ROOT", PROJECT_ROOT / "outputs/wss_deploy_release_retired"))
VESSEL_GEOM_DIR = Path(os.environ.get("WSS_DEPLOY_VESSEL_GEOM", PROJECT_ROOT / "outputs/vessel_geom_toolkit_2026-09-17"))
VMTK_PYTHON = Path(os.environ.get("WSS_DEPLOY_VMTK_PYTHON", Path.home() / ".conda/envs/GNN_vmtk/bin/python"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
SEEDS = (1234, 7, 2025, 11, 2026)
OUTLET_NAMES = ("out-le", "out-li", "out-re", "out-ri")
OUTLET_CN = {"inlet": "入口（主动脉）", "out-le": "左髂外", "out-li": "左髂内", "out-re": "右髂外", "out-ri": "右髂内"}
BRANCH_CN = {"root": "主动脉", "left_cia": "左髂总", "right_cia": "右髂总", "out-le": "左髂外", "out-li": "左髂内", "out-re": "右髂外", "out-ri": "右髂内"}
