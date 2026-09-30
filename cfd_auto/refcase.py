"""Reference (library) case: read-only profile + a path-rewritten copy that Fluent may read inside the work directory.

The profile records what the new case must reproduce: opening names/types/centres, extension lengths, zone naming
(blood/bloodN, wall/wallN, in+/out-*+), prism-layer statistics and the UDF constants. Python only reads the library
files; Fluent never sees a library path (it would auto-compile libudf inside the library directory, observed 2026-09-27).
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
from pathlib import Path

import numpy as np

from cfd_auto import guard
from cfd_auto import udf as udf_io

LIB_PATH_RE = re.compile(rb"/public/newhome/cy/Digital_twin/GNN/data(?:_new)?/[^\"\s)]*")
UDF_NUM = r"([-+0-9.eE]+)"


def find_case_file(case_dir: Path) -> Path:
    cas = sorted(case_dir.glob("*.cas.gz")) or sorted(case_dir.glob("*.cas"))
    if len(cas) != 1:
        raise FileNotFoundError(f"{case_dir}: expected exactly one case file, found {[c.name for c in cas]}")
    return cas[0]


def find_udf(case_dir: Path) -> Path:
    for pat in ("udf-inlet4.c", "udf-inlet.c", "libudf/src/udf-inlet*.c"):
        hit = sorted(case_dir.glob(pat))
        if hit:
            return hit[0]
    raise FileNotFoundError(f"{case_dir}: no udf-inlet*.c")


def parse_udf(text: str) -> dict:
    """Mesh-dependent and protocol constants of the library UDF (thread ids, inlet area divisor, RCR per outlet)."""
    threads = {m.group(2): int(m.group(1)) for m in re.finditer(r"Lookup_Thread\(d,\s*(\d+)\);\s*/\*\s*(out\w\w)", text)}
    inlet = re.search(r"DEFINE_PROFILE\(my_inlet.*?end_f_loop", text, re.S)
    area = re.search(r"\)\s*/\s*" + UDF_NUM + r"\s*;", inlet.group(0)) if inlet else None
    rcr = {}
    for m in re.finditer(r"DEFINE_PROFILE\(pressure_(out\w\w),.*?R1\s*=\s*" + UDF_NUM + r";.*?R2\s*=\s*" + UDF_NUM + r";.*?C\s*=\s*" + UDF_NUM + r";", text, re.S):
        rcr[m.group(1)] = {"R1": float(m.group(2)), "R2": float(m.group(3)), "C": float(m.group(4))}
    return {"threads": threads, "inlet_area_m2": float(area.group(1)) if area else None, "rcr": rcr}


def profile(case_dir: str | Path) -> dict:
    """Read-only profile of a library case (Fluent case + UDF)."""
    from wss_pinn.v4.fluent_topology import anatomy_topology, anatomy_wall_faces, read_fluent_mesh
    case_dir = Path(case_dir)
    cas = find_case_file(case_dir)
    mesh = read_fluent_mesh(cas, keep_interior=True)
    topo = anatomy_topology(mesh)
    cz = topo["cell_zone"]
    openings, extensions = [], []
    for itf in topo["interfaces"]:
        n = itf.unit_normal
        c = itf.centres_m.mean(0)
        ext = itf.extension_zone_id
        wall_nodes = []
        wall_zone = None
        for sec in mesh.face_sections_of_type(3):
            m = cz[sec.c0] == ext
            if m.any():
                _, nodes = sec.subset_connectivity(np.flatnonzero(m)); wall_nodes.append(mesh.nodes_m[np.unique(nodes)]); wall_zone = mesh.zone_name(sec.zone_id)
        P = np.concatenate(wall_nodes)
        ax = (P - c) @ n
        if np.median(ax) < 0:
            n, ax = -n, -ax
        interface_names = sorted({mesh.zone_name(z) for z in itf.face_section_zone_ids})
        openings.append({"name": itf.distal_bc_name, "bc_type": {10: "velocity-inlet", 5: "pressure-outlet"}.get(itf.distal_bc_type, str(itf.distal_bc_type)),
                         "centre_mm": (c * 1e3).round(4).tolist(), "outward_normal": n.round(6).tolist(), "area_mm2": round(itf.area_m2 * 1e6, 4)})
        extensions.append({"opening": itf.distal_bc_name, "fluid_zone": itf.extension_zone_name, "wall_zone": wall_zone, "interface_zone": interface_names,
                           "length_mm": round(float(ax.max()) * 1e3, 3), "interface_faces": int(len(itf.centres_m))})
    walls = anatomy_wall_faces(mesh, topo)
    udf = find_udf(case_dir)
    s = mesh.summary()
    consts = parse_udf(udf_io.read_udf(udf))
    # which zone each UDF outlet thread id points at (the UDF keys outle/outli/outri/outre are zone-name agnostic)
    thread_zones = {k: mesh.zone_name(v) for k, v in consts["threads"].items() if v in mesh.zone_names}
    poly = s["cell_types"].get("polyhedron", 0) > 0.5 * s["cells"]
    out = {"case_dir": str(case_dir), "cas": str(cas), "udf": str(udf), "udf_constants": consts, "udf_thread_zones": thread_zones,
           "anatomy_zone": mesh.zone_name(topo["anatomy_zone_id"]), "anatomy_wall_zones": sorted({mesh.zone_name(z) for z in np.unique(walls["face_zone_ids"])}),
           "anatomy_wall_faces": int(len(walls["centres_m"])), "anatomy_wall_nodes": int(len(walls["node_ids"])),
           "cells": s["cells"], "cell_types": s["cell_types"], "cell_zones": {k: v["count"] for k, v in s["cell_zones"].items()},
           "family": "poly" if poly else "tet-prism", "openings": openings, "extensions": extensions}
    if poly:
        out["poly_mesh_estimate"] = _poly_estimate(mesh, cz == topo["anatomy_zone_id"], walls)
    return out


def _poly_estimate(mesh, anatomy: np.ndarray, walls: dict) -> dict:
    """Library poly-hexcore meshes: octree hex sizes give the max size (hex edge = max / 2^k); the wall polygons are the
    duals of a triangle surface mesh (edge ~ sqrt(polygon area) / 0.93) whose 1st percentile gives the min size."""
    _, vol = mesh.cell_centroids_and_volumes()
    hexa = anatomy & (mesh.cell_types == 4)
    hex_sizes = np.cbrt(vol[hexa]) * 1e3
    area = np.linalg.norm(walls["area_vectors_m2"], axis=1)
    edge = np.sqrt(area) / 0.93 * 1e3
    levels = np.unique(np.round(hex_sizes, 2)) if len(hex_sizes) else np.array([])
    return {"hex_cells": int(hexa.sum()), "hex_size_levels_mm": levels.tolist()[:8], "max_size_mm": float(np.round(np.percentile(hex_sizes, 99.5), 2)) if len(hex_sizes) else None,
            "wall_edge_mm_p1_50_99": np.percentile(edge, [1, 50, 99]).round(3).tolist(), "min_size_mm": float(np.round(np.percentile(edge, 1) / 0.05) * 0.05)}


def copy_for_fluent(case_dir: str | Path, workdir: str | Path, dest_name: str = "reference_settings.cas.gz", rename_to: str | None = None) -> dict:
    """Copy the library case into ``workdir`` with every library path string rewritten into ``workdir`` (export prefixes,
    UDF sources, autosave roots). Fluent later reads only this copy. ``rename_to``: when the settings come from another
    (template) case, its name in the export prefixes is replaced so the exports carry the rebuilt case's name."""
    case_dir, workdir = Path(case_dir), Path(workdir)
    guard.assert_inside(workdir, workdir)
    cas = find_case_file(case_dir)
    data = gzip.open(cas, "rb").read() if cas.suffix == ".gz" else cas.read_bytes()
    found = sorted(set(LIB_PATH_RE.findall(data)))
    new_dir = str(workdir).encode()
    template_name = cas.name.split(".cas")[0].encode()
    replaced = []
    for old in found:
        tail = old.split(b"/")[-1]
        if rename_to:
            tail = tail.replace(template_name, rename_to.encode())
        rep = new_dir + b"/" + (b"ascii/" + tail if b"/ascii/" in old else tail)
        data = data.replace(old, rep); replaced.append((old.decode(), rep.decode()))
    left = [x for x in LIB_PATH_RE.findall(data)]
    if left:
        raise AssertionError(f"library paths remain in the copied case: {left[:3]}")
    dest = workdir / dest_name
    with gzip.open(dest, "wb", compresslevel=6) as fh:
        fh.write(data)
    return {"source": str(cas), "copy": str(dest), "paths_rewritten": replaced}


def fix_export_layout(case_copy: str | Path, workdir: str | Path) -> dict:
    """The library layout needs the wall export in ``<work>/ascii/<name>`` and the volume export in ``<work>/<name>``.
    Some library cases write both to the same file name (ILO/ZHANG_YAN_SHAN-0/before since its 09-04 re-save): Fluent then
    skips the second export of every step (it never overwrites), so a rerun produces no wall frames. Rewrites, in the
    work-directory copy only, the file-name of an export with surfaces that shares its name with another export or does
    not write into ascii/. Returns what was changed (empty when the layout is already right)."""
    from cfd_auto import settings_diff
    case_copy, workdir = Path(case_copy), Path(workdir)
    guard.assert_inside(case_copy, workdir)
    data = gzip.open(case_copy, "rb").read() if case_copy.suffix == ".gz" else case_copy.read_bytes()
    exports = settings_diff._exports(settings_diff.sections(settings_diff.case_text(case_copy))["rp"])
    names = {k: re.findall(r'"([^"]*)"', v.get("file-name", "")) for k, v in exports.items()}
    names = {k: v[0] for k, v in names.items() if v}
    changed = []
    for k, e in exports.items():
        if e.get("surfaces", "()") in ("()", "") or k not in names:
            continue
        fn = names[k]
        clash = any(o != k and names.get(o) == fn for o in names)
        if "/ascii/" in fn and not clash:
            continue
        new = str(workdir / "ascii" / Path(fn).name)
        start = data.find(b"(" + k.encode() + b" (")
        if start < 0:
            raise RuntimeError(f"export {k}: definition not found in the case text")
        old_b = b'(file-name . "' + fn.encode() + b'")'
        j = data.find(old_b, start)
        if j < 0 or j - start > 20000:
            raise RuntimeError(f"export {k}: file-name not found near its definition")
        data = data[:j] + b'(file-name . "' + new.encode() + b'")' + data[j + len(old_b):]
        changed.append({"export": k, "from": fn, "to": new, "reason": "shared file name" if clash else "wall export outside ascii/"})
    if changed:
        with gzip.open(case_copy, "wb", compresslevel=6) as fh:
            fh.write(data)
        after = settings_diff._exports(settings_diff.sections(settings_diff.case_text(case_copy))["rp"])
        for c in changed:
            got = re.findall(r'"([^"]*)"', after[c["export"]]["file-name"])[0]
            if got != c["to"]:
                raise RuntimeError(f"export {c['export']}: rewrite did not take ({got})")
    return {"changed": changed}
