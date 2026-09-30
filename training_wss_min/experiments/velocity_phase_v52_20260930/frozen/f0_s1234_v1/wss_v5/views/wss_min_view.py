"""Derive the legacy ``training_wss_min`` bundle view from V5 case bundles.

One ``bundle.npz`` per case in the legacy schema so the tuned PointNeXt-R /
LocalGeoPE / L-SA2 code path runs unchanged:

* coordinates: atlas anatomical frame (flow-divider origin, trunk -> +Z,
  left common iliac -> +X) then per-case max-abs scaling to [-1, 1]
  (``wall_coords_norm``); true-mm Fluent-frame coordinates kept in
  ``wall_coords_raw``;
* features: ``wall_abscissa_norm`` = s_from_root / max, ``wall_local_radius`` =
  atlas radius (true mm), ``wall_curvature`` = atlas curvature (1/mm),
  ``wall_radius_gradient`` = dr/ds, plus the extra V5 point features for later
  arms;
* labels: 81-frame ``wall_wss`` / ``wall_pressure`` / ``wall_wss_vec`` at the
  valid anatomy wall nodes only (non-exported nodes dropped).

Also writes the legacy-schema split and the train-only log_z WSS statistics.
Read-only with respect to the master bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from wss_pinn.utils import ROOT, sha256_file, utc_now

from .. import contract as C
from ..sources import Registry

VIEW_NAME = "wss_min_view_v1"
FRAME_VERSION = "v5_atlas_frame_v1"  # v1.1: + wall_normal_pca_aligned (same frame/features otherwise)
PEAK_STEP = 1162
VIEW_ROOT = ROOT / "data_wss_v5" / "views" / VIEW_NAME
EXTRA_FEATURES = ("rho", "theta_rad", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm", "end_zone",
                  "segment_id", "semantic_id", "s_local_mm", "junction_ambiguous")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-9:
        raise ValueError("degenerate axis")
    return v / n


def anatomical_frame(atlas_table: np.ndarray, columns: list[str], semantic_of_segment: dict[str, int]) -> dict[str, Any]:
    """Origin at the aortic flow divider, +Z toward the inlet along the trunk chord, +X toward the left common iliac."""
    col = lambda n: atlas_table[:, columns.index(n)]
    seg = col("segment_id").astype(int)
    idx = col("sample_index").astype(int)
    xyz = np.stack([col("x_mm"), col("y_mm"), col("z_mm")], axis=1)
    trunk_rows = np.flatnonzero(seg == 0)
    trunk_rows = trunk_rows[np.argsort(idx[trunk_rows])]
    origin = xyz[trunk_rows[-1]]           # trunk end = flow divider
    inlet_pt = xyz[trunk_rows[0]]
    z_axis = _unit(inlet_pt - origin)      # trunk points to +Z, iliac branches to -Z
    sem = {int(k): int(v) for k, v in semantic_of_segment.items()}
    left = [s for s, v in sem.items() if v == 1]
    right = [s for s, v in sem.items() if v == 2]
    source = "left_minus_right_cia"
    if left and right:
        lateral = xyz[np.isin(seg, left)].mean(axis=0) - xyz[np.isin(seg, right)].mean(axis=0)
    else:
        lateral = np.array([1.0, 0.0, 0.0])
        source = "world_x_fallback"
    x_axis = lateral - (lateral @ z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1.0e-6:
        x_axis = np.array([1.0, 0.0, 0.0]) - z_axis[0] * z_axis
        source = "world_x_fallback"
    x_axis = _unit(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)  # rows: new axes in old coords -> p_new = R @ (p - origin)
    world_x_agreement = float(np.sign(x_axis[0])) if abs(x_axis[0]) > 1.0e-6 else 0.0
    return {"origin_mm": origin, "rotation": rotation, "x_source": source, "left_axis_world_x_sign": world_x_agreement,
            "trunk_length_mm": float(np.linalg.norm(inlet_pt - origin))}


def build_case_view(canonical_id: str, snapshot_root: Path, out_root: Path) -> dict[str, Any]:
    started = time.time()
    case_dir = C.case_dir(canonical_id, snapshot_root)
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    h5_path = case_dir / "case.h5"
    with h5py.File(h5_path, "r") as h5:
        g = h5["geometry"]
        table = g["atlas_table"][()]
        columns = json.loads(g.attrs["atlas_columns"])
        semantic_of_segment = json.loads(g.attrs["semantic_of_segment"])
        ws = h5["wall_static"]
        xyz_mm = ws["xyz_mm"][()]
        valid = ws["valid"][()].astype(bool)
        source_row = ws["source_row"][()]
        node_id = ws["node_id_cas"][()]
        s_root = ws["s_from_root_mm"][()]
        radius = ws["radius_mm"][()]
        curvature = ws["curvature_per_mm"][()]
        extras = {name: ws[name][()] for name in EXTRA_FEATURES}
        normal_pca = ws["normal_out_pca"][()]
        wt = h5["wall_temporal"]
        steps = wt["step"][()]
        wss = wt["wss_scalar_pa"][()]
        pressure = wt["pressure_pa"][()]
        wss_vec = wt["wss_vector_pa"][()]
    frame = anatomical_frame(table, columns, semantic_of_segment)
    keep = valid
    xyz_keep = xyz_mm[keep]
    aligned = (xyz_keep - frame["origin_mm"]) @ frame["rotation"].T
    scale = float(np.abs(aligned).max())
    scale = scale if scale > 1.0e-9 else 1.0
    s_max = float(np.max(s_root[keep]))
    payload: dict[str, Any] = {
        "case": np.asarray(canonical_id.split("/")[-1]),
        "cohort": np.asarray(canonical_id.split("/")[0]),
        "canonical_id": np.asarray(canonical_id),
        "steps": steps.astype(np.int32),
        "peak_step": np.int32(PEAK_STEP),
        "coord_scale": np.float32(scale),
        "coord_scale_on": np.asarray("wall"),
        "unit_factor": np.float64(C.LENGTH_M_TO_MM),
        "unit_extent_mismatch": np.bool_(False),
        "wall_crop_applied": np.bool_(False),
        "wall_crop_frac": np.float64(0.0),
        "transform_centroid": frame["origin_mm"].astype(np.float64),
        "transform_rotation": frame["rotation"].astype(np.float64),
        "transform_frame_version": np.asarray(FRAME_VERSION),
        "transform_origin_kind": np.asarray("atlas_trunk_end_flow_divider"),
        "transform_main_axis_mode": np.asarray("trunk_end_to_inlet_chord_plus_z"),
        "transform_roll_source": np.asarray(frame["x_source"]),
        "transform_left_axis_world_x_sign": np.float64(frame["left_axis_world_x_sign"]),
        "wall_coords_norm": (aligned / scale).astype(np.float32),
        "wall_coords_aligned_mm": aligned.astype(np.float32),
        "wall_coords_raw": xyz_keep.astype(np.float32),
        "wall_nodenumber": (source_row[keep] + 1).astype(np.int64),
        "wall_node_id_cas": node_id[keep].astype(np.int64),
        "wall_abscissa_norm": (s_root[keep] / s_max).astype(np.float32),
        "wall_abscissa_mm": s_root[keep].astype(np.float32),
        "wall_local_radius": radius[keep].astype(np.float32),
        "wall_curvature": curvature[keep].astype(np.float32),
        "wall_radius_gradient": extras["dr_ds"][keep].astype(np.float32),
        "wall_normal_pca": normal_pca[keep].astype(np.float32),
        "wall_normal_pca_aligned": (normal_pca[keep] @ frame["rotation"].T).astype(np.float32),
        "wall_wss": wss[:, keep].astype(np.float32),
        "wall_pressure": pressure[:, keep].astype(np.float32),
        "wall_wss_vec": wss_vec[:, keep].astype(np.float32),
    }
    for name in EXTRA_FEATURES:
        payload[f"wall_{name}"] = extras[name][keep]
    payload["wall_theta_sin"] = np.sin(extras["theta_rad"][keep]).astype(np.float32)
    payload["wall_theta_cos"] = np.cos(extras["theta_rad"][keep]).astype(np.float32)
    if not np.isfinite(payload["wall_wss"]).all():
        raise ValueError(f"{canonical_id}: NaN WSS among valid nodes")
    out_dir = out_root / canonical_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bundle.npz"
    tmp = out_dir / "bundle.tmp.npz"  # np.savez appends .npz unless the name already ends with it
    np.savez(tmp, **payload)
    tmp.replace(out_path)
    report = {
        "canonical_id": canonical_id, "view": VIEW_NAME, "frame_version": FRAME_VERSION, "created_at": utc_now(),
        "source": {"case_h5": str(h5_path), "dataset_digest_sha256": manifest["bundle"]["dataset_digest_sha256"],
                   "geometry_program_version": manifest.get("geometry_program_version")},
        "n_wall_valid": int(keep.sum()), "n_wall_dropped": int((~keep).sum()), "coord_scale_mm": scale,
        "frame": {"origin_mm": frame["origin_mm"].tolist(), "rotation": frame["rotation"].tolist(), "x_source": frame["x_source"],
                  "left_axis_world_x_sign": frame["left_axis_world_x_sign"], "trunk_length_mm": frame["trunk_length_mm"]},
        "abscissa_max_mm": s_max, "peak_step": PEAK_STEP, "bundle_sha256": sha256_file(out_path), "seconds": time.time() - started,
    }
    (out_dir / "view_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def write_split(registry: Registry, out_root: Path) -> Path:
    train = sorted(c for c in registry.all_cases() if registry.role(c) == "train")
    test = sorted(c for c in registry.all_cases() if registry.role(c) == "test")
    payload = {
        "schema_version": 1, "created_at": utc_now(), "pipeline": "wss_v5.views.wss_min_view", "data_root": str(out_root),
        "required_frame_version": FRAME_VERSION,
        "id_format": "canonical cohort/subset/case; ILO subset is patient-0|1 and case is before|after",
        "split_version": f"split_V5_anatomy_only_train{len(train)}_test{len(test)}_wss_min_view_v1",
        "source_split": str(C.SPLIT_PATH), "source_split_sha256": sha256_file(C.SPLIT_PATH),
        # 2026-09-15: 源 split（schema 3 起）自带重复几何组；旧源 split 没有时沿用原常量，输出逐字不变
        "duplicate_geometry_groups": registry.split.get("duplicate_geometry_groups") or [["AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"]],
        "explicit_related_groups": registry.split.get("explicit_related_groups") or [["AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"]],
        "train_cases": train, "val_cases": [], "test_cases": test, "unused_cases": [],
        "counts": {"train": len(train), "val": 0, "test": len(test)}, "expected_counts": {"train": len(train), "val": 0, "test": len(test)},
    }
    path = out_root / f"split_V5_train{len(train)}_test{len(test)}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def write_wss_stats(train_cases: list[str], out_root: Path, split_path: Path,
                    filename: str = "wss_global_stats_train138.json",
                    scope: str = "train partition only (V5 train138, valid anatomy wall nodes)") -> Path:
    eps = 1.0e-6
    n = 0
    s1 = s2 = 0.0
    l1 = l2 = 0.0
    zero = 0
    samples = []
    for cid in train_cases:
        with np.load(out_root / cid / "bundle.npz", allow_pickle=True) as d:
            steps = d["steps"].tolist()
            y = d["wall_wss"][steps.index(PEAK_STEP)].astype(np.float64)
        n += len(y)
        s1 += y.sum(); s2 += (y ** 2).sum()
        lg = np.log(np.clip(y, 0, None) + eps)
        l1 += lg.sum(); l2 += (lg ** 2).sum()
        zero += int(np.sum(y <= 0))
        samples.append(y)
    allv = np.concatenate(samples)
    lin_mean = s1 / n; lin_std = float(np.sqrt(max(s2 / n - lin_mean ** 2, 0.0)))
    log_mean = l1 / n; log_std = float(np.sqrt(max(l2 / n - log_mean ** 2, 0.0)))
    payload = {
        "schema_version": 1, "generated_at": utc_now(), "method": "log_z", "eps": eps, "required_frame_version": FRAME_VERSION,
        "timesteps_scope": "peak", "statistics_scope": scope,
        "split_path": str(split_path), "split_sha256": sha256_file(split_path), "n_cases": len(train_cases), "n_points": int(n),
        "zero_frac": zero / n, "linear": {"mean": lin_mean, "std": lin_std}, "log": {"mean": log_mean, "std": log_std},
        "raw_min": float(allv.min()), "raw_max": float(allv.max()),
        "raw_percentiles": {f"p{q}": float(np.percentile(allv, q)) for q in (50, 90, 95, 99)},
        "train_units": train_cases,
    }
    path = out_root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-root", type=Path, default=C.SNAPSHOT_ROOT)
    parser.add_argument("--out-root", type=Path, default=VIEW_ROOT)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--no-cohort-files", action="store_true",
                        help="2026-09-22: only build the per-case bundles (no split / train statistics rewrite) — for new units evaluated with an existing run's statistics")
    args = parser.parse_args(argv)
    registry = Registry.load()
    cases = args.cases or registry.all_cases()
    args.out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, cid in enumerate(cases):
        rep = build_case_view(cid, args.snapshot_root, args.out_root)
        reports.append(rep)
        if i % 20 == 0 or i == len(cases) - 1:
            print(f"[wss_min_view] {i + 1}/{len(cases)} {cid} n={rep['n_wall_valid']} scale={rep['coord_scale_mm']:.1f}mm x_src={rep['frame']['x_source']}", flush=True)
    if args.no_cohort_files:
        split_path = stats_path = None
        # 2026-09-23: per-case runs share one view root; merge this run's reports into the existing manifest instead of
        # replacing it (two concurrent runs / a later eval reading the manifest would otherwise lose the other cases)
        manifest_path = args.out_root / "view_manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("reports", [])
            merged = {r["canonical_id"]: r for r in previous}
            merged.update({r["canonical_id"]: r for r in reports})
            reports = [merged[k] for k in sorted(merged)]
    else:
        split_path = write_split(registry, args.out_root)
        train = sorted(c for c in registry.all_cases() if registry.role(c) == "train")
        stats_path = write_wss_stats(train, args.out_root, split_path, filename=f"wss_global_stats_train{len(train)}.json",
                                     scope=f"train partition only (V5 train{len(train)}, valid anatomy wall nodes)")
    signs = [r["frame"]["left_axis_world_x_sign"] for r in reports]
    summary = {
        "view": VIEW_NAME, "frame_version": FRAME_VERSION, "created_at": utc_now(), "snapshot_root": str(args.snapshot_root),
        "cases": len(reports), "split_path": str(split_path) if split_path else None, "wss_stats_path": str(stats_path) if stats_path else None,
        "left_axis_world_x_positive_fraction": float(np.mean(np.asarray(signs) > 0)) if signs else None,
        "x_source_counts": {s: sum(1 for r in reports if r["frame"]["x_source"] == s) for s in {r["frame"]["x_source"] for r in reports}},
        "n_wall_dropped_total": int(sum(r["n_wall_dropped"] for r in reports)),
        "coord_scale_mm": {"min": min(r["coord_scale_mm"] for r in reports), "median": float(np.median([r["coord_scale_mm"] for r in reports])),
                           "max": max(r["coord_scale_mm"] for r in reports)},
        "reports": reports,
    }
    (args.out_root / "view_manifest.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
