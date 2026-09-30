"""V5 volume view for training_wss_min: interior (anatomy cell) points aligned to the wall view frame.

Per case writes ``volume.npz`` next to the wall ``bundle.npz`` of ``wss_min_view`` (same atlas anatomical
frame, same ``coord_scale``), holding
* static geometric features for every anatomy cell with the *same names* as the wall features
  (abscissa_norm, local_radius, curvature, radius_gradient=dR/ds, rho, theta_sin/cos, dr_ds,
  dist_to_junction_mm, dist_to_endpoint_mm, end_zone) plus ``dist_to_wall_mm`` and the outward radial unit
  vector (``vol_radial_aligned``: the interior analogue of the aligned wall normal);
* labels at the peak step (1162, same as the WSS label): velocity components rotated into the aligned frame
  (m/s), speed, and relative pressure ``p - p_ref`` with ``p_ref`` = volume-mean pressure of the case at the
  peak step. For the bundle's valid wall nodes a pressure label with the *same* reference is exported so wall and
  interior points can be mixed in one pressure target; it is taken from the nearest anatomy cell (the
  ``label``-tagged volume field, first cell 0.5-1 mm away, |Δp| p90 < 35 Pa on healthy exports) because the
  wall export pressure column is ``audit_only`` in the V5 contract and is broken for HAN_JIAN_FU (≈0 Pa).
Afterwards ``volume_stats_train138.json`` gives train-only linear statistics for the new targets.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

from wss_v5 import contract as C
from wss_v5.views import wss_min_view as W

VOLUME_VIEW_VERSION = "v5_volume_view_v1.1"  # v1.1: wall pressure label from nearest volume cell
FEATURES = ("s_from_root_mm", "radius_mm", "curvature_per_mm", "dr_ds", "rho", "theta_rad", "dist_to_junction_mm",
            "dist_to_endpoint_mm", "end_zone", "dist_to_wall_mm", "semantic_id", "segment_id", "volume_m3", "atlas_row")


def build_case_volume(canonical_id: str, snapshot_root: Path, out_root: Path) -> dict[str, Any]:
    started = time.time()
    bundle_path = out_root / canonical_id / "bundle.npz"
    with np.load(bundle_path, allow_pickle=True) as d:
        origin = d["transform_centroid"].astype(np.float64)
        rotation = d["transform_rotation"].astype(np.float64)
        scale = float(d["coord_scale"])
        frame_version = str(np.asarray(d["transform_frame_version"]).item())
        steps = d["steps"].tolist()
        peak = int(d["peak_step"])
        si = steps.index(peak)
        s_max = float(d["wall_abscissa_mm"].max())
        wall_export_pressure_peak = d["wall_pressure"][si].astype(np.float64)  # audit only
        wall_xyz_raw = d["wall_coords_raw"].astype(np.float64)
    h5_path = C.case_dir(canonical_id, snapshot_root) / "case.h5"
    with h5py.File(h5_path, "r") as h5:
        g = h5["geometry"]
        table = g["atlas_table"][()]
        columns = json.loads(g.attrs["atlas_columns"])
        vs = h5["volume_static"]
        xyz = vs["xyz_mm"][()].astype(np.float64)
        f = {name: vs[name][()] for name in FEATURES}
        h5_steps = h5["wall_temporal/step"][()].tolist()
        k = h5_steps.index(peak)
        vel = h5["volume_temporal/velocity_m_s"][k].astype(np.float64)
        pres = h5["volume_temporal/pressure_pa"][k].astype(np.float64)
        p_ref = float(h5["pressure_reference/p_volume_mean_pa"][k])
    if abs(pres.mean() - p_ref) > 1.0e-3 * max(1.0, abs(p_ref)):
        # p_volume_mean is volume weighted in the bundle; plain mean differs. Keep the bundle reference.
        pass
    col = lambda name: table[:, columns.index(name)]
    atlas_xyz = np.stack([col("x_mm"), col("y_mm"), col("z_mm")], axis=1)
    atlas_tan = np.stack([col("tangent_x"), col("tangent_y"), col("tangent_z")], axis=1)
    row = f["atlas_row"].astype(np.int64)
    if row.min() < 0 or row.max() >= len(table):
        raise ValueError(f"{canonical_id}: atlas_row out of range")
    dvec = xyz - atlas_xyz[row]
    tan = atlas_tan[row]
    tan = tan / np.clip(np.linalg.norm(tan, axis=1, keepdims=True), 1.0e-12, None)
    dvec = dvec - np.sum(dvec * tan, axis=1, keepdims=True) * tan
    dn = np.linalg.norm(dvec, axis=1, keepdims=True)
    radial = np.where(dn > 1.0e-6, dvec / np.clip(dn, 1.0e-12, None), 0.0)
    aligned = (xyz - origin) @ rotation.T
    nn_dist, nn_idx = cKDTree(xyz).query(wall_xyz_raw)
    wall_pressure_peak = pres[nn_idx]  # label-tagged volume pressure at the nearest cell
    wall_export_diff = wall_export_pressure_peak - wall_pressure_peak
    payload: dict[str, Any] = {
        "canonical_id": np.asarray(canonical_id),
        "volume_view_version": np.asarray(VOLUME_VIEW_VERSION),
        "transform_frame_version": np.asarray(frame_version),
        "peak_step": np.int32(peak),
        "p_ref_pa": np.float64(p_ref),
        "p_ref_kind": np.asarray("volume_mean_pressure_at_peak_step"),
        "n_cells": np.int64(len(xyz)),
        "s_max_mm": np.float64(s_max),
        "coord_scale": np.float32(scale),
        "vol_coords_norm": (aligned / scale).astype(np.float32),
        "vol_coords_aligned_mm": aligned.astype(np.float32),
        "vol_coords_raw": xyz.astype(np.float32),
        "vol_abscissa_norm": (f["s_from_root_mm"] / s_max).astype(np.float32),
        "vol_local_radius": f["radius_mm"].astype(np.float32),
        "vol_curvature": f["curvature_per_mm"].astype(np.float32),
        "vol_radius_gradient": f["dr_ds"].astype(np.float32),
        "vol_rho": f["rho"].astype(np.float32),
        "vol_theta_sin": np.sin(f["theta_rad"]).astype(np.float32),
        "vol_theta_cos": np.cos(f["theta_rad"]).astype(np.float32),
        "vol_dr_ds": f["dr_ds"].astype(np.float32),
        "vol_dist_to_junction_mm": f["dist_to_junction_mm"].astype(np.float32),
        "vol_dist_to_endpoint_mm": f["dist_to_endpoint_mm"].astype(np.float32),
        "vol_end_zone": f["end_zone"].astype(np.int8),
        "vol_dist_to_wall_mm": f["dist_to_wall_mm"].astype(np.float32),
        "vol_semantic_id": f["semantic_id"].astype(np.int16),
        "vol_segment_id": f["segment_id"].astype(np.int16),
        "vol_volume_m3": f["volume_m3"].astype(np.float64),
        "vol_radial_aligned": (radial @ rotation.T).astype(np.float32),
        "vol_velocity_aligned_peak": (vel @ rotation.T).astype(np.float32),
        "vol_speed_peak": np.linalg.norm(vel, axis=1).astype(np.float32),
        "vol_pressure_rel_peak": (pres - p_ref).astype(np.float32),
        "wall_pressure_rel_peak": (wall_pressure_peak - p_ref).astype(np.float32),
        "wall_pressure_source": np.asarray("nearest_volume_cell_pressure_pa"),
        "wall_nearest_cell_dist_mm": nn_dist.astype(np.float32),
    }
    for key in ("vol_velocity_aligned_peak", "vol_pressure_rel_peak", "vol_coords_norm", "vol_dist_to_wall_mm"):
        if not np.isfinite(payload[key]).all():
            raise ValueError(f"{canonical_id}: non-finite {key}")
    out_path = out_root / canonical_id / "volume.npz"
    tmp = out_root / canonical_id / "volume.tmp.npz"
    np.savez(tmp, **payload)
    tmp.replace(out_path)
    return {
        "canonical_id": canonical_id, "n_cells": int(len(xyz)), "p_ref_pa": p_ref,
        "radial_zero_frac": float(np.mean(dn[:, 0] <= 1.0e-6)),
        "speed_p50_p99_max": [float(np.percentile(payload["vol_speed_peak"], 50)), float(np.percentile(payload["vol_speed_peak"], 99)), float(payload["vol_speed_peak"].max())],
        "pressure_rel_std_min_max": [float(payload["vol_pressure_rel_peak"].std()), float(payload["vol_pressure_rel_peak"].min()), float(payload["vol_pressure_rel_peak"].max())],
        "wall_pressure_rel_mean": float(payload["wall_pressure_rel_peak"].mean()),
        "wall_nearest_cell_dist_p50_p90_mm": [float(np.percentile(nn_dist, 50)), float(np.percentile(nn_dist, 90))],
        "wall_export_minus_cell_pressure_median_p90abs": [float(np.median(wall_export_diff)), float(np.percentile(np.abs(wall_export_diff), 90))],
        "coords_norm_absmax": float(np.abs(payload["vol_coords_norm"]).max()),
        "seconds": time.time() - started,
    }


def _worker(args):
    cid, snap, out = args
    try:
        return build_case_volume(cid, Path(snap), Path(out))
    except Exception as exc:  # noqa: BLE001 - reported per case
        return {"canonical_id": cid, "error": f"{type(exc).__name__}: {exc}"}


def write_volume_stats(train_cases: list[str], out_root: Path, split_path: Path) -> Path:
    acc = {"vel": np.zeros(3), "vel2": np.zeros(3), "sp": 0.0, "sp2": 0.0, "n_int": 0,
           "pw": 0.0, "pw2": 0.0, "n_wall": 0, "pi": 0.0, "pi2": 0.0}
    for cid in train_cases:
        with np.load(out_root / cid / "volume.npz", allow_pickle=True) as d:
            v = d["vol_velocity_aligned_peak"].astype(np.float64)
            sp = d["vol_speed_peak"].astype(np.float64)
            pi = d["vol_pressure_rel_peak"].astype(np.float64)
            pw = d["wall_pressure_rel_peak"].astype(np.float64)
        acc["vel"] += v.sum(0); acc["vel2"] += (v ** 2).sum(0); acc["sp"] += sp.sum(); acc["sp2"] += (sp ** 2).sum(); acc["n_int"] += len(v)
        acc["pi"] += pi.sum(); acc["pi2"] += (pi ** 2).sum(); acc["pw"] += pw.sum(); acc["pw2"] += (pw ** 2).sum(); acc["n_wall"] += len(pw)
    def ms(s1, s2, n):
        m = s1 / n
        return float(m) if np.ndim(m) == 0 else m.tolist(), (np.sqrt(np.maximum(s2 / n - m ** 2, 0.0))).tolist() if np.ndim(m) else float(np.sqrt(max(s2 / n - m ** 2, 0.0)))
    vm, vsd = ms(acc["vel"], acc["vel2"], acc["n_int"])
    spm, spsd = ms(acc["sp"], acc["sp2"], acc["n_int"])
    pim, pisd = ms(acc["pi"], acc["pi2"], acc["n_int"])
    pwm, pwsd = ms(acc["pw"], acc["pw2"], acc["n_wall"])
    n_all = acc["n_int"] + acc["n_wall"]
    pam, pasd = ms(acc["pi"] + acc["pw"], acc["pi2"] + acc["pw2"], n_all)
    payload = {
        "schema_version": 1, "generated_at": W.utc_now(), "method": "linear", "eps": 0.0,
        "required_frame_version": W.FRAME_VERSION, "volume_view_version": VOLUME_VIEW_VERSION, "timesteps_scope": "peak",
        "peak_step": W.PEAK_STEP, "statistics_scope": f"train partition only (V5 train{len(train_cases)}): all anatomy cells + valid wall nodes",
        "split_path": str(split_path), "split_sha256": W.sha256_file(split_path), "n_cases": len(train_cases),
        "n_interior_points": int(acc["n_int"]), "n_wall_points": int(acc["n_wall"]),
        "linear": {"mean": pam, "std": pasd, "target": "pressure_rel_peak (wall ∪ interior), Pa"},
        "pressure_rel_interior": {"mean": pim, "std": pisd}, "pressure_rel_wall": {"mean": pwm, "std": pwsd},
        "velocity": {"mean": vm, "std": vsd, "frame": "atlas anatomical frame (aligned), m/s"},
        "speed": {"mean": spm, "std": spsd, "units": "m/s"},
        "train_units": train_cases,
    }
    path = out_root / f"volume_stats_train{len(train_cases)}.json"  # 2026-09-15: 按病例数派生（train138 时与原名逐字相同）
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-root", default=str(C.SNAPSHOT_ROOT))
    ap.add_argument("--out-root", default=str(W.VIEW_ROOT))
    ap.add_argument("--cases", nargs="*", default=None, help="canonical ids; default = all cases of the view split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--split", default=None, help="view split json; default = the unique split_V5_train*_test*.json in out-root (2026-09-16: v5.1 = train136)")
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    if args.split:
        split_path = Path(args.split)
    else:
        found = sorted(out_root.glob("split_V5_train*_test*.json"))
        if len(found) != 1:
            raise FileNotFoundError(f"expected exactly one split_V5_train*_test*.json in {out_root}, found {[f.name for f in found]}")
        split_path = found[0]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_cases = list(split["train_cases"])
    all_cases = train_cases + list(split["test_cases"])
    cases = args.cases or all_cases
    reports = []
    if not args.stats_only:
        jobs = [(cid, args.snapshot_root, str(out_root)) for cid in cases]
        with mp.Pool(max(1, args.workers)) as pool:
            for i, rep in enumerate(pool.imap_unordered(_worker, jobs)):
                reports.append(rep)
                msg = rep.get("error") or f"n={rep['n_cells']} p_ref={rep['p_ref_pa']:.0f} speed_max={rep['speed_p50_p99_max'][2]:.2f} {rep['seconds']:.1f}s"
                print(f"[volume_view] {i + 1}/{len(jobs)} {rep['canonical_id']} {msg}", flush=True)
        errors = [r for r in reports if "error" in r]
        (out_root / "volume_view_report.json").write_text(json.dumps({
            "volume_view_version": VOLUME_VIEW_VERSION, "generated_at": W.utc_now(), "n_cases": len(cases), "n_errors": len(errors),
            "cases": sorted(reports, key=lambda r: r["canonical_id"])}, indent=1, ensure_ascii=False), encoding="utf-8")
        if errors:
            print(f"[volume_view] {len(errors)} case(s) failed; stats not written", flush=True)
            return 1
    if args.cases is None or args.stats_only:
        path = write_volume_stats(train_cases, out_root, split_path)
        print(f"[volume_view] stats -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
