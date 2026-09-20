"""Frozen constants of the V5 data contract (schema v5.0).

Everything a downstream loader may rely on is declared here: units, time axis,
module/dataset names, field dependency tags and the pilot gate thresholds.
"""
from __future__ import annotations

from pathlib import Path

from wss_pinn.utils import ROOT

SCHEMA_VERSION = "wss_v5.1"  # 2026-09: 26 例 RCR 出口面积修正重算 + 剔除 2 例重复（train136/test34）
GEOMETRY_PROGRAM_VERSION = "pc-v3"  # pc-v2 + outlet labels re-derived from geometry (nearest anatomy interface)
SNAPSHOT_NAME = "anatomy_pointcloud_v5_1_20260916"  # v5.0 快照 anatomy_pointcloud_v5_20260906 原样保留
SNAPSHOT_ROOT = ROOT / "data_wss_v5" / SNAPSHOT_NAME

# ---------------------------------------------------------------- sources
PREP_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
TOPOLOGY_AUDIT_DIR = PREP_ROOT / "audits/topology/cases"
WALL_AUDIT_SUMMARY = PREP_ROOT / "audits/wall_wss/summary.json"
SOLVER_AUDIT_SUMMARY = PREP_ROOT / "audits/solver_convergence/summary.json"
ATLAS_DIR = PREP_ROOT / "atlas/cases"
SPLIT_PATH = ROOT / "wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train136_test34_s1234.json"
RAW_ROOT = ROOT / "data_new"

# ---------------------------------------------------------------- physics / time
EXPECTED_STEPS = list(range(1120, 1281, 2))  # 81 frames, last (8th) cycle
STEP_DT_S = 0.005
PERIOD_S = 0.8
CYCLE_START_STEP = 1120  # t = 5.6 s = 7 T
RHO_KG_M3 = 1060.0
LENGTH_M_TO_MM = 1000.0  # Fluent SI metres -> true millimetres (frozen 09-03 unit contract)

OUTLET_ORDER = ("out-le", "out-li", "out-ri", "out-re")
INTERFACE_ORDER = ("inlet",) + OUTLET_ORDER
# atlas segment semantics (7 unique segments: trunk, two common iliacs, four terminal branches)
SEMANTIC_LABELS = {
    0: "trunk",          # aorta incl. aneurysm sac, from inlet to aortic bifurcation
    1: "left_cia",
    2: "right_cia",
    3: "left_external",  # out-le
    4: "left_internal",  # out-li
    5: "right_external", # out-re
    6: "right_internal", # out-ri
}
OUTLET_TO_SEMANTIC = {"out-le": 3, "out-li": 4, "out-re": 5, "out-ri": 6}

# ---------------------------------------------------------------- identity tolerances
CELL_TOLERANCE_FRACTION = 0.2   # of local cbrt(volume), as in the 09-03 topology audit
NODE_TOLERANCE_M = 1.0e-7
FRAME_COORD_TOL_M = 1.0e-9      # coordinates must repeat exactly across the 81 frames

# ---------------------------------------------------------------- point-cloud geometry program
PCA_NORMAL_K = 16
CAP_POINT_SPACING_FACTOR = 1.0  # cap disk sampled at ~ the median wall point spacing
INSIDE_VOTE_K = 8
TUBE_MARGIN = 0.15              # candidate radius = R * (1 + margin)
WINDING_SUBSAMPLE = 10_000      # queries per class for the winding-number check

# ---------------------------------------------------------------- pilot gate thresholds (proposals; reported, not hidden)
GATES = {
    "atlas_alignment_median_rel_gap_max": 0.25,   # median |dist - R| / R over wall nodes
    "pca_normal_angle_median_deg_max": 10.0,
    "pca_normal_angle_p95_deg_max": 30.0,
    "pca_normal_flipped_fraction_max": 0.005,     # PCA normal > 90 deg from the mesh normal (judging only)
    "inside_recall_min": 0.99,                    # true anatomy cell centres classified inside
    "extension_false_inside_max": 0.02,           # extension cells (outside anatomy) classified inside
    "shell_false_inside_max": 0.02,               # points 1-3 mm outside the wall classified inside
    "wall_missing_nodes_max": 4,
    "wall_pressure_delta_median_pa_max": 1.0,     # wall vs adjacent-cell pressure (same solution)
    "volume_sum_rel_tol": 1.0e-6,
    "area_sum_rel_tol": 1.0e-9,
}

# ---------------------------------------------------------------- dependency tags (design v0.3 §8.2)
TAG_MODEL_FEATURE = "model_feature"          # allowed as default patient input (xyz + local geometry / centerline)
TAG_QUADRATURE = "quadrature_only"           # weights for integrals / losses / metrics, never an input
TAG_AUDIT = "audit_only"                     # provenance, identity, protocol metadata
TAG_OFFLINE_REF = "offline_reference_only"   # area-based proxies, protocol waveform samples, flux estimates
TAG_LABEL = "label"                          # CFD supervision targets

HDF5_CHUNK_POINTS = 65_536
HDF5_COMPRESSION = "lzf"


def case_dir(canonical_id: str, root: Path = SNAPSHOT_ROOT) -> Path:
    return root / "cases" / canonical_id.replace("/", "__")
