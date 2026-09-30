#!/usr/bin/env python3
"""Write the original postview point cloud as an EnSight point part.

No tetrahedra, hexahedra, or triangles. Coordinates and values are copied
from the VTP point clouds. Formal R²/MAE stay on the same-point CSV.

  /public/newhome/cy/.conda/envs/GNN/bin/python3.10 training_wss_min/tools/build_ensight_pointcloud.py \
      --case-dir <postview>/AAA__unruputer__SHEN_FANG_JIN
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sliceable_tets as bt


def log(msg: str) -> None:
    print(f"[ensight_points] {msg}", flush=True)


def as_vertices(xyz: np.ndarray, arrays: dict[str, np.ndarray]):
    import pyvista as pv

    n = len(xyz)
    cells = np.column_stack([np.ones(n, np.int64), np.arange(n, dtype=np.int64)]).ravel()
    ug = pv.UnstructuredGrid(cells, np.full(n, pv.CellType.VERTEX, np.uint8), np.asarray(xyz, np.float64))
    for name, arr in arrays.items():
        ug.point_data[name] = np.asarray(arr)
    n_poly = int(np.sum(np.asarray(ug.celltypes) != pv.CellType.VERTEX))
    if n_poly:
        raise RuntimeError(f"{n_poly} cells are not vertices")
    return ug


def velocity_cloud(case_dir: Path, arms: list[str], phases: list[str]):
    """Interior cell centres only. Wall nodes are not added."""
    import pyvista as pv

    ref_path = case_dir / "velocity" / arms[0] / phases[0] / "interior_pointcloud.vtp"
    P = np.asarray(pv.read(ref_path).points, np.float64)
    arrays: dict[str, np.ndarray] = {}
    for arm in arms:
        for ph in phases:
            f = case_dir / "velocity" / arm / ph / "interior_pointcloud.vtp"
            if not f.is_file():
                continue
            got = bt.load_matching(f, P, bt.VEL_CFD_ARRAYS + bt.VEL_PRED_ARRAYS)
            for nm in bt.VEL_CFD_ARRAYS:
                if nm not in got:
                    continue
                key = f"{ph}_{nm}"
                if key in arrays and not np.allclose(arrays[key], got[nm], atol=1e-6, equal_nan=True):
                    arrays[f"{arm}_{ph}_{nm}"] = got[nm]
                else:
                    arrays[key] = got[nm]
            for nm in bt.VEL_PRED_ARRAYS:
                if nm in got:
                    arrays[f"{arm}_{ph}_{nm}"] = got[nm]
    return P, arrays, str(ref_path)


def pressure_cloud(case_dir: Path, arms: list[str], phases: list[str]):
    """Wall ∪ interior, file order wall-first, values copied from full_pointcloud.vtp."""
    import pyvista as pv

    ref_path = case_dir / "pressure" / arms[0] / phases[0] / "full_pointcloud.vtp"
    pd = pv.read(ref_path)
    kind = np.asarray(pd.point_data[bt.KIND_ARRAY]).astype(np.int64)
    order = np.argsort(kind, kind="stable")
    P = np.asarray(pd.points, np.float64)[order]
    arrays: dict[str, np.ndarray] = {"is_wall": (kind[order] == 0).astype(np.uint8)}
    for arm in arms:
        for ph in phases:
            f = case_dir / "pressure" / arm / ph / "full_pointcloud.vtp"
            if not f.is_file():
                continue
            got = bt.load_matching(f, P, bt.PRS_CFD_ARRAYS + bt.PRS_PRED_ARRAYS)
            for nm in bt.PRS_CFD_ARRAYS:
                if nm not in got:
                    continue
                key = f"{ph}_{nm}"
                if key in arrays and not np.allclose(arrays[key], got[nm], atol=1e-6, equal_nan=True):
                    arrays[f"{arm}_{ph}_{nm}"] = got[nm]
                else:
                    arrays[key] = got[nm]
            for nm in bt.PRS_PRED_ARRAYS:
                if nm in got:
                    arrays[f"{arm}_{ph}_{nm}"] = got[nm]
    return P, arrays, str(ref_path)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    case_dir = args.case_dir.resolve()
    out = (args.out or case_dir / "sliceable_tets" / "ensight_points").resolve()
    out.mkdir(parents=True, exist_ok=True)
    vel_arms, prs_arms, phases = bt.discover(case_dir)
    report: dict = {"tool": "build_ensight_pointcloud.py", "element": "point", "no_surface": True}

    if vel_arms:
        P, arrays, src = velocity_cloud(case_dir, vel_arms, phases)
        ug = as_vertices(P, arrays)
        bt.write_ensight(ug, out, "velocity_points", report)
        report["velocity"] = dict(source=src, n_points=int(len(P)), n_arrays=len(arrays), n_wall=0)
        log(f"velocity points {len(P)} arrays {len(arrays)}")
    if prs_arms:
        P, arrays, src = pressure_cloud(case_dir, prs_arms, phases)
        ug = as_vertices(P, arrays)
        bt.write_ensight(ug, out, "pressure_points", report)
        report["pressure"] = dict(source=src, n_points=int(len(P)), n_arrays=len(arrays),
                                   n_wall=int(np.asarray(arrays["is_wall"]).sum()))
        log(f"pressure points {len(P)} arrays {len(arrays)}")

    report["elapsed_s"] = round(time.time() - t0, 1)
    (out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"done in {report['elapsed_s']} s → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
