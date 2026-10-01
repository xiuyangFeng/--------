"""Canonical STL manifest of the CFD library (2026-10-01).

For every real unit of the anatomy library: each STL in its ``data_new`` folder (hash, connected pieces, openings
of the main piece, flat extra pieces, distances to the CFD wall from the 10-01 library audit), a category, and the
*canonical* surface to use as a deployment-style test input:

* ``exact``          one piece, five openings, the CFD wall lies on it (median <= 1 mm) → the ``data_new`` STL;
* ``assembly``       vessel + straight extension tubes + flat cutting planes (pre-processing file);
* ``capped``         closed main surface (no openings; LIN_SHU_TIAN also carries 22 tiny debris pieces);
* ``port_closed``    one piece, fewer than five openings;
* ``openings_extra`` one piece, more than five openings;
* ``unit_scaled``    the STL is ~1000× (or 1/1000×) the CFD geometry;
* ``other_frame``    same geometry as the CFD wall only after a rigid registration (other coordinate frame);
* ``missing``        no STL in the folder.

For every category but ``exact`` the canonical surface is the CFD wall itself, exported from ``case.h5``
(``wall_static/xyz_mm`` + ``topology/wall_triangles``): same domain, frame and units as the labels, open at the
five ports.  The library build never reads these STLs (geometry and centrelines come from the Fluent mesh), so the
manifest only serves tools that start from a surface (deployment tests, naming calibration).  Patient-derived
outputs go under ``data_wss_v5/`` (not in git).

    python tools/stl_manifest.py build  --out data_wss_v5/stl_canonical_v5_2d_20261001 --processes 12
    python tools/stl_manifest.py verify --out data_wss_v5/stl_canonical_v5_2d_20261001 --processes 12
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import shutil
import tempfile
import time
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # wss_deploy / training_wss_min for the workers
LIBRARY = ROOT / "data_wss_v5/anatomy_pointcloud_v5_2d_20261001"
DATA_NEW = ROOT / "data_new"
AUDIT = ROOT / "outputs/cfd_auto_trial_20260927/_protocol_study/library_audit_20260930"
NEAR_MM = 1.0          # CFD wall on the STL: median distance (the audit's mesh_not_own_stl threshold)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def real_units(library: Path = LIBRARY) -> list[str]:
    return sorted(p.name.replace("__", "/") for p in (library / "cases").iterdir() if "~" not in p.name)


def case_folder(uid: str) -> Path:
    cohort, subset, case = uid.split("/")
    return DATA_NEW / cohort / subset / case


def audit_tables() -> tuple[dict, dict]:
    extra = {r["unit"]: r for r in json.loads((AUDIT / "extra_checks_v52d.json").read_text())}
    registered = json.loads((AUDIT / "stl_registered.json").read_text()) if (AUDIT / "stl_registered.json").exists() else {}
    return extra, registered


def describe_stl(path: Path, cfd_bbox_diag: float) -> dict:
    """Pieces, openings of the main piece, flatness of the other pieces, size relative to the CFD wall."""
    import pyvista as pv
    mesh = pv.read(str(path)).clean()
    conn = mesh.connectivity(extraction_mode="all")
    rid = np.asarray(conn.cell_data["RegionId"])
    main_id = int(np.bincount(rid).argmax())
    main = conn.extract_cells(np.flatnonzero(rid == main_id)).extract_surface()
    edges = main.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
    openings = int(np.unique(np.asarray(edges.connectivity().point_data["RegionId"])).size) if edges.n_points else 0
    flat = []
    for r in np.unique(rid):
        if r == main_id:
            continue
        p = np.asarray(conn.extract_cells(np.flatnonzero(rid == r)).points)
        sv = np.linalg.svd(p - p.mean(0), compute_uv=False)
        flat.append(bool(sv[2] / max(sv[0], 1e-12) < 0.01))
    diag = float(np.linalg.norm(np.ptp(np.asarray(mesh.points), 0)))
    return {"pieces": int(np.unique(rid).size), "main_openings": openings, "extra_pieces_flat": flat,
            "main_area_frac": round(float(main.area / mesh.area), 4) if mesh.area else None,
            "bbox_diag_mm": round(diag, 1), "size_ratio_to_cfd": round(diag / cfd_bbox_diag, 3)}


def classify(info: dict, audit_row: dict | None, registered: float | None) -> str:
    if info["size_ratio_to_cfd"] > 10 or info["size_ratio_to_cfd"] < 0.1:
        return "unit_scaled"
    if info["main_openings"] == 0:
        return "capped"                       # closed main surface (other pieces, if any, are debris)
    if info["pieces"] > 1 and info["extra_pieces_flat"] and all(info["extra_pieces_flat"]):
        return "assembly"
    if info["pieces"] == 1 and info["main_openings"] < 5:
        return "port_closed"
    if info["pieces"] == 1 and info["main_openings"] > 5:
        return "openings_extra"
    near = audit_row.get("mesh_to_stl_median_mm") if audit_row else None
    if info["pieces"] == 1 and near is not None and near <= NEAR_MM:
        return "exact"
    if registered is not None and registered <= NEAR_MM:
        return "other_frame"
    return "unresolved"


def export_cfd_wall(uid: str, out: Path, library: Path = LIBRARY) -> Path:
    """The CFD wall of ``uid`` as a binary STL in mm (CFD frame, open at the ports)."""
    import h5py
    import pyvista as pv
    with h5py.File(library / "cases" / uid.replace("/", "__") / "case.h5", "r") as h:
        xyz = np.asarray(h["wall_static/xyz_mm"][()], dtype=np.float64)
        tri = np.asarray(h["topology/wall_triangles"][()], dtype=np.int64)
    # The wall triangulation of polyhedral faces can repeat a triangle (5 of 24 exports had duplicates and the
    # non-manifold edges they cause): keep one copy of each vertex triple and drop degenerate triangles.
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    _, first = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)
    tri = tri[np.sort(first)]
    # An edge shared by three triangles (1–4 per export in 5 of 24 units, away from the ports) carries one sliver
    # "fin" a tenth the size of its neighbours whose two other edges hang free: drop the smallest triangle at each
    # such edge until the surface is manifold.
    removed = 0
    while True:
        edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
        owner = np.tile(np.arange(len(tri)), 3)
        uniq, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
        bad = np.flatnonzero(counts[inv.ravel()] > 2)
        if not len(bad):
            break
        p = xyz[tri]
        area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        drop = set()
        for e in np.unique(inv.ravel()[bad]):
            ts = owner[np.flatnonzero(inv.ravel() == e)]
            drop.add(int(ts[np.argmin(area[ts])]))
        tri = np.delete(tri, sorted(drop), axis=0)
        removed += len(drop)
    faces = np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri]).ravel()
    out.parent.mkdir(parents=True, exist_ok=True)
    pv.PolyData(xyz, faces).clean().save(str(out), binary=True)
    export_cfd_wall.removed_fins = removed
    return out


def build_unit(args) -> dict:
    uid, out_dir, extra, registered = args
    import h5py
    with h5py.File(LIBRARY / "cases" / uid.replace("/", "__") / "case.h5", "r") as h:
        wall = np.asarray(h["wall_static/xyz_mm"][()])
    cfd_diag = float(np.linalg.norm(np.ptp(wall, 0)))
    folder = case_folder(uid)
    audit_by_file = {r["file"]: r for r in (extra.get(uid) or {}).get("stl") or []}
    files = []
    for path in sorted(Path(p) for p in glob.glob(str(folder / "*.stl"))):
        info = describe_stl(path, cfd_diag)
        a = audit_by_file.get(path.name)
        info.update(name=path.name, sha256=sha256(path), bytes=path.stat().st_size,
                    mtime=time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime)),
                    cfd_wall_to_stl_median_mm=(round(a["mesh_to_stl_median_mm"], 4) if a else None),
                    stl_to_cfd_wall_median_mm=(round(a["stl_to_mesh_median_mm"], 4) if a and "stl_to_mesh_median_mm" in a else None))
        info["category"] = classify(info, a, registered.get(uid))
        files.append(info)
    row = {"unit": uid, "case_folder": str(folder.relative_to(ROOT)), "stl_files": files}
    exact = [f for f in files if f["category"] == "exact"]
    if exact:
        best = min(exact, key=lambda f: f["cfd_wall_to_stl_median_mm"])
        row.update(category="exact", canonical={"kind": "data_new_stl", "path": str((folder / best["name"]).relative_to(ROOT)),
                                                  "sha256": best["sha256"], "frame": "cfd"})
    else:
        row["category"] = files[0]["category"] if len(files) == 1 else ("missing" if not files else
                          "/".join(sorted({f["category"] for f in files})))
        target = Path(out_dir) / "cfd_wall" / (uid.replace("/", "__") + ".stl")
        export_cfd_wall(uid, target)
        row["canonical"] = {"kind": "cfd_wall", "path": str(target.relative_to(ROOT)), "sha256": sha256(target), "frame": "cfd",
                            "source": "case.h5 wall_static/xyz_mm + topology/wall_triangles",
                            "repair": {"duplicate_triangles_removed": True, "sliver_fins_removed": getattr(export_cfd_wall, "removed_fins", 0)}}
    if registered.get(uid) is not None:
        row["audit_registered_median_mm"] = registered[uid]
    return row


def verify_unit(args) -> dict:
    """The canonical surface through the deployment input check (and, for CFD-wall exports, the whole stage A)."""
    row, out_dir = args
    from wss_deploy.ingest import ingest
    path = ROOT / row["canonical"]["path"]
    tmp = Path(tempfile.mkdtemp(prefix="stlman_"))
    try:
        ic = ingest(path, tmp, units="auto")
        res = {"ingest_status": ic.get("status"), "pieces": ic.get("components"), "openings": ic.get("openings"),
               "errors": ic.get("errors") or []}
        if row["canonical"]["kind"] == "cfd_wall" and ic.get("status") == "pass":
            from wss_deploy.pipeline import stage_a
            from wss_deploy.naming_calibration import compare
            a = stage_a(path, tmp / "a", units="auto")
            if a.get("stage") == "A":
                c = compare({}, a, row["unit"])
                res.update(stage_a="pass", naming_auto_ok=c.get("auto_ok"), naming_joint_correct=c.get("joint_correct"),
                           naming_confidence=c.get("confidence"))
            else:
                res["stage_a"] = "input check did not pass"
        return {"unit": row["unit"], **res}
    except Exception as exc:  # noqa: BLE001
        return {"unit": row["unit"], "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


MANIFEST = ROOT / "data_wss_v5/stl_canonical_v5_2d_20261001/manifest.json"
_CANONICAL: dict | None = None


def canonical_stl(uid: str, manifest: Path = MANIFEST) -> Path | None:
    """The canonical test surface of a library unit (``data_new`` STL or CFD-wall export), or None if unknown.

    Meant to replace "the first ``*.stl`` of the folder" in the tools that start from a surface."""
    global _CANONICAL
    if _CANONICAL is None:
        _CANONICAL = {r["unit"]: r for r in json.loads(Path(manifest).read_text())["units"]}
    row = _CANONICAL.get(uid)
    return ROOT / row["canonical"]["path"] if row else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("build", "verify"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if args.cmd == "build":
        extra, registered = audit_tables()
        units = real_units()
        with mp.get_context("spawn").Pool(args.processes) as pool:
            rows = pool.map(build_unit, [(u, str(out), extra, registered) for u in units])
        doc = {"schema": "stl-canonical-manifest/v1", "library": str(LIBRARY.relative_to(ROOT)), "created_at": time.strftime("%Y-%m-%d %H:%M"),
               "near_mm": NEAR_MM, "audit": str(AUDIT.relative_to(ROOT)), "units": rows}
        manifest_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        from collections import Counter
        print(json.dumps({"units": len(rows), "categories": Counter(r["category"] for r in rows),
                          "canonical": Counter(r["canonical"]["kind"] for r in rows)}, ensure_ascii=False))
    else:
        doc = json.loads(manifest_path.read_text())
        with mp.get_context("spawn").Pool(args.processes) as pool:
            checks = pool.map(verify_unit, [(r, str(out)) for r in doc["units"]])
        by = {c["unit"]: c for c in checks}
        for r in doc["units"]:
            r["verify"] = {k: v for k, v in by[r["unit"]].items() if k != "unit"}
        doc["verified_at"] = time.strftime("%Y-%m-%d %H:%M")
        manifest_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        from collections import Counter
        print(json.dumps({"ingest": Counter(r["canonical"]["kind"] + ":" + str(r["verify"].get("ingest_status") or r["verify"].get("error", "")[:40]) for r in doc["units"]),
                          "cfd_wall_stage_a": Counter(str(r["verify"].get("naming_joint_correct")) for r in doc["units"] if r["canonical"]["kind"] == "cfd_wall")},
                         ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
