"""Refresh the geometry-program datasets of built bundles without re-parsing raw frames.

Recomputes PCA normals, virtual caps, the point-cloud domain gate and the
generated internal queries from the stored wall/volume coordinates + atlas +
Fluent case (mesh only, for judging), rewrites the affected datasets and
updates report/manifest/gates.  ``--dry-run`` only prints the new metrics.

    python -m wss_v5.refresh_geometry --cases AAA/ruputer/WANG_KUI_WU --dry-run
    python -m wss_v5.refresh_geometry --all-built --workers 20 --run-name full_20260906_pcv2
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from wss_pinn.utils import git_state, utc_now

from . import contract as C
from .build import refresh_snapshot_manifest, summarize
from .build_case import MM, _digests_from, _gates, _json_ready, _pointcloud_gate
from .centerline_features import load_atlas, map_points, relabel_outlets
from .mesh_topology import load_mesh
from .sources import Registry, locate
from .store import aggregate_digest, array_digest

REPLACED = ("wall_static/normal_out_pca", "geometry/cap_center_mm", "geometry/cap_radius_mm", "query_geometry/generated_internal_sample_mm",
            "wall_static/semantic_id", "volume_static/semantic_id")


def _write(dst: h5py.File, name: str, array: np.ndarray, *, units: str, tag: str, description: str = "") -> None:
    array = np.asarray(array)
    kwargs = {"chunks": True} if array.ndim >= 1 and array.size > 4096 else {}
    ds = dst.create_dataset(name, data=array, **kwargs)
    ds.attrs["units"] = units
    ds.attrs["dependency"] = tag
    if description:
        ds.attrs["description"] = description


def _selective_copy(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value

    def copy_group(sg: h5py.Group, dg: h5py.Group) -> None:
        for key, value in sg.attrs.items():
            dg.attrs[key] = value
        for name, obj in sg.items():
            full = obj.name.lstrip("/")
            if full in REPLACED:
                continue
            if isinstance(obj, h5py.Group):
                copy_group(obj, dg.require_group(name))
            else:
                sg.copy(obj, dg, name=name)

    copy_group(src, dst)


def refresh_case(canonical_id: str, registry: Registry, root: Path, *, write: bool, seed: int, domain_subsample: int, n_generate: int) -> dict[str, Any]:
    started = time.time()
    case_out = C.case_dir(canonical_id, root)
    h5_path = case_out / "case.h5"
    report = json.loads((case_out / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "built" or not h5_path.is_file():
        report["refresh_error"] = "case not built"
        return report
    try:
        src = locate(canonical_id, registry)
        md = load_mesh(src.cas_path)
        with h5py.File(h5_path, "r") as h5:
            wall_xyz_mm = h5["wall_static/xyz_mm"][()]
            vol_xyz_mm = h5["volume_static/xyz_mm"][()]
            cell_id = h5["volume_static/cell_id_cas"][()]
            interfaces = {label: {"center_mm": np.asarray(json.loads(g.attrs["center_mm"])), "normal_out": np.asarray(json.loads(g.attrs["normal_out"])),
                                  "area_m2": float(g.attrs["area_m2"])}
                          for label, g in h5["interfaces"].items()}
        if np.max(np.abs(wall_xyz_mm - md.wall_node_coords_m * MM)) > 1.0e-6:
            raise ValueError("stored wall coordinates do not match the Fluent case")
        atlas, relabel = relabel_outlets(load_atlas(src.atlas_npz, src.atlas_summary), {k: v["center_mm"] for k, v in interfaces.items()})
        vol = SimpleNamespace(xyz_m=vol_xyz_mm / MM, cell_id_cas=cell_id)
        pc = _pointcloud_gate(md, vol, atlas, wall_xyz_mm, interfaces, seed, domain_subsample, n_generate)
        wall_sem = map_points(wall_xyz_mm, atlas)["semantic_id"]
        vol_sem = map_points(vol_xyz_mm, atlas)["semantic_id"]
        report["metrics"]["pointcloud"] = {k: v for k, v in pc.items() if k not in ("generated_sample_mm", "pca_normals_out")}
        report["metrics"]["atlas_relabel"] = relabel
        report["geometry_program_version"] = C.GEOMETRY_PROGRAM_VERSION
        report["gates"] = _gates(report, src)
        report["refreshed_at"] = utc_now()
        if write:
            tmp = h5_path.with_suffix(".h5.refresh")
            if tmp.exists():
                tmp.unlink()
            with h5py.File(h5_path, "r") as src_h5, h5py.File(tmp, "w") as dst:
                _selective_copy(src_h5, dst)
                _write(dst, "wall_static/normal_out_pca", pc["pca_normals_out"], units="unit", tag=C.TAG_MODEL_FEATURE,
                       description="PCA normal from the bare point cloud, MST-consistent, oriented outward")
                caps = pc["cloud"]["caps"]
                _write(dst, "geometry/cap_center_mm", np.array([c["center_mm"] for c in caps]), units="mm", tag=C.TAG_MODEL_FEATURE)
                _write(dst, "geometry/cap_radius_mm", np.array([c["radius_mm"] for c in caps]), units="mm", tag=C.TAG_MODEL_FEATURE)
                g = dst["geometry"]
                g.attrs["cap_labels"] = json.dumps([c["label"] for c in caps])
                g.attrs["cap_normals_out"] = json.dumps([c["outward"] for c in caps])
                g.attrs["caps_detail"] = json.dumps(caps, ensure_ascii=False)
                g.attrs["wall_point_spacing_mm"] = pc["cloud"]["spacing_mm"]
                g.attrs["atlas_segments"] = json.dumps(atlas.segments, ensure_ascii=False)
                g.attrs["semantic_of_segment"] = json.dumps(atlas.semantic_of_segment)
                g.attrs["outlet_relabel"] = json.dumps(_json_ready(relabel), ensure_ascii=False)
                _write(dst, "wall_static/semantic_id", wall_sem, units="id", tag=C.TAG_MODEL_FEATURE, description="branch semantic id (outlets relabelled by geometry)")
                _write(dst, "volume_static/semantic_id", vol_sem, units="id", tag=C.TAG_MODEL_FEATURE, description="branch semantic id (outlets relabelled by geometry)")
                _write(dst, "query_geometry/generated_internal_sample_mm", pc["generated_sample_mm"], units="mm", tag=C.TAG_MODEL_FEATURE,
                       description="internal query points generated from the bare wall point cloud + atlas (hybrid inside test)")
                q = dst.require_group("query_geometry")
                q.attrs["recipe"] = json.dumps({"pca_k": C.PCA_NORMAL_K, "vote_k": C.INSIDE_VOTE_K, "tube_margin": C.TUBE_MARGIN, "seed": seed,
                                                "geometry_program_version": C.GEOMETRY_PROGRAM_VERSION})
                q.attrs["generated_queries"] = json.dumps(_json_ready(pc["generated_queries"]))
                q.attrs["inside_test"] = json.dumps(_json_ready(pc["inside_test"]))
                dst.attrs["geometry_program_version"] = C.GEOMETRY_PROGRAM_VERSION
                dst.attrs["refreshed_at"] = report["refreshed_at"]
            os.replace(tmp, h5_path)
            digests = _digests_from(h5_path)
            datasets = {}
            with h5py.File(h5_path, "r") as h5:
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype), "units": obj.attrs.get("units", ""), "dependency": obj.attrs.get("dependency", "")}
                h5.visititems(visit)
            report["datasets"] = datasets
            report["bundle"] = {"path": str(h5_path), "bytes": h5_path.stat().st_size, "datasets": len(datasets),
                                "dataset_digest_sha256": aggregate_digest({k: {"sha256": v} for k, v in digests.items()})}
            report["code"] = git_state()
            payload = _json_ready(report)
            (case_out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            manifest = {k: payload[k] for k in ("schema_version", "geometry_program_version", "canonical_id", "cohort", "role", "validation_fold", "patient_group",
                                                 "created_at", "sources", "bundle", "datasets", "gates", "code") if k in payload}
            manifest["refreshed_at"] = payload["refreshed_at"]
            manifest["counts"] = payload["metrics"]["counts"]
            manifest["conditions_summary"] = payload["metrics"]["conditions"]
            (case_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        report["refresh_error"] = f"{type(exc).__name__}: {exc}"
        report["refresh_traceback"] = traceback.format_exc()
        report.setdefault("gates", {})["pass"] = False
    report.setdefault("timings_s", {})["refresh"] = time.time() - started
    return _json_ready(report)


def _worker(canonical_id: str, root: str, write: bool, seed: int, domain_subsample: int, n_generate: int) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    return refresh_case(canonical_id, Registry.load(), Path(root), write=write, seed=seed, domain_subsample=domain_subsample, n_generate=n_generate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cases", nargs="+")
    group.add_argument("--all-built", action="store_true")
    parser.add_argument("--root", type=Path, default=C.SNAPSHOT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain-subsample", type=int, default=6000)
    parser.add_argument("--n-generate", type=int, default=60_000)
    args = parser.parse_args(argv)
    registry = Registry.load()
    if args.all_built:
        cases = [json.loads(p.read_text(encoding="utf-8"))["canonical_id"] for p in sorted((args.root / "cases").glob("*/report.json"))]
    else:
        cases = list(args.cases)
    print(f"[v5-refresh] cases={len(cases)} write={not args.dry_run} workers={args.workers}", flush=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, c, str(args.root), not args.dry_run, args.seed, args.domain_subsample, args.n_generate): c for c in cases}
        for future in as_completed(futures):
            r = future.result()
            reports.append(r)
            pc = r.get("metrics", {}).get("pointcloud", {})
            it = pc.get("inside_test", {})
            print(f"[v5-refresh] {r['canonical_id']}: gate={r.get('gates', {}).get('pass')} "
                  f"recall={it.get('anatomy_cells', {}).get('inside_fraction_hybrid')} ext={it.get('extension_cells_near_interface', {}).get('inside_fraction_hybrid')} "
                  f"shell={it.get('shell_1_3mm_outside', {}).get('inside_fraction_hybrid')} flipped={pc.get('normals_pca_vs_mesh', {}).get('flipped_fraction')} "
                  f"failed={[k for k, v in r.get('gates', {}).get('checks', {}).items() if not v]} relabel={r.get('metrics', {}).get('atlas_relabel', {}).get('changed')} {r.get('refresh_error', '')}", flush=True)
    if args.run_name and not args.dry_run:
        order = {c: i for i, c in enumerate(cases)}
        reports.sort(key=lambda r: order.get(r["canonical_id"], 1 << 30))
        summary = summarize(reports, args.root, args.run_name)
        refresh_snapshot_manifest(args.root, registry)
        print(f"[v5-refresh] {summary['gate_pass']}/{summary['cases']} pass -> {summary['gate_result']}", flush=True)
        return 0 if summary["gate_result"] == "pass" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
