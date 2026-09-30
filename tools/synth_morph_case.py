"""Synthetic CFD cases by morphing an existing Fluent case (2026-09-26, direction 3 of the v5.2 improvement plan).

The parent .cas.gz keeps its zones, boundary conditions, UDF thread ids and export definitions; only node coordinates
change (in place inside the binary node section), so the whole ingestion chain runs unchanged on the child.

Deformation (deployment-irrelevant, label-generating only): every node of the anatomy zone is assigned to its nearest
centerline station (atlas of the parent's V5 bundle, raw CFD frame); the node is moved radially about the station centre
by a smooth factor f(s) = 1 + sum_k a_k exp(-(s - s_k)^2 / 2 sigma_k^2) (stenoses a_k < 0, dilations a_k > 0), tapered to
f = 1 within TAPER_END mm of every opening (so inlet/outlet areas and hence the RCR/inlet protocol stay bit-identical)
and within TAPER_JUNCTION x radius of the junctions (so the three branches stay consistent). Extension-zone nodes are
never moved. Validity: every cell volume must stay positive and no cell may shrink below MIN_VOLUME_RATIO of its
original volume; Fluent's own mesh/check runs in the smoke job before any solve.

    python tools/synth_morph_case.py --parent AG/slow/QIN_SI_FU --child-tag m01 --seed 20260926 [--dry-run]
    python tools/synth_morph_case.py --plan synth_plan.json          # many children
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from wss_pinn.v4 import new_case_sources as ncs  # noqa: E402
from wss_pinn.v4.fluent_topology import read_fluent_mesh  # noqa: E402
from wss_v5 import contract as C  # noqa: E402

TAPER_END_MM = 12.0
TAPER_JUNCTION_R = 1.5
MIN_VOLUME_RATIO = 0.25
MAX_VOLUME_RATIO = 4.0
F_MIN, F_MAX = 0.55, 1.45
NODE_RE = re.compile(rb"\(3010 \(([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+)(?: ([0-9a-fA-F]+))?\)\n\(")


def child_id(parent: str, tag: str) -> str:
    """AG/slow/NAME -> AG/slow/NAME~m01 ; ILO/NAME-0/before -> ILO/NAME-0~m01/before (patient group stays NAME)."""
    parts = parent.split("/")
    if parts[0] == "ILO":
        return f"{parts[0]}/{parts[1]}~{tag}/{parts[2]}"
    return f"{parts[0]}/{parts[1]}/{parts[2]}~{tag}"


def load_atlas(parent: str):
    for root in ("data_wss_v5/anatomy_pointcloud_v5_2_20260922", "data_wss_v5/anatomy_pointcloud_v5_1_20260916"):
        p = ROOT / root / "cases" / parent.replace("/", "__") / "case.h5"
        if p.is_file():
            with h5py.File(p) as f:
                g = f["geometry"]; cols = json.loads(g.attrs["atlas_columns"]); T = g["atlas_table"][...]
                segs = json.loads(g.attrs["atlas_segments"])
            return T, cols, segs, str(p)
    raise FileNotFoundError(f"{parent}: no V5 bundle")


def modulation(T, cols, segs, rng, n_bumps: int, amp_range: tuple[float, float], sigma_range: tuple[float, float]):
    """f per atlas sample: smooth bumps placed away from openings/junctions on randomly chosen segments."""
    ci = {c: cols.index(c) for c in ("segment_id", "s_local_mm", "radius_mm", "dist_to_endpoint_mm", "dist_to_junction_mm")}
    seg = T[:, ci["segment_id"]].astype(int); s = T[:, ci["s_local_mm"]]; r = T[:, ci["radius_mm"]]
    d_end = T[:, ci["dist_to_endpoint_mm"]]; d_j = T[:, ci["dist_to_junction_mm"]]
    f = np.ones(len(T)); spec = []
    for _ in range(n_bumps):
        # the bump must have faded (2.5 sigma) before the junction / opening taper zones, otherwise nodes of the two
        # sibling branches next to the bifurcation get inconsistent factors and thin cells invert (QIN_SI_FU dry run)
        sigma = float(rng.uniform(*sigma_range))
        eligible = np.flatnonzero((d_end > TAPER_END_MM + 2.5 * sigma) & (d_j > TAPER_JUNCTION_R * r + 2.5 * sigma))
        if not len(eligible):
            sigma = sigma_range[0]
            eligible = np.flatnonzero((d_end > TAPER_END_MM + 2.5 * sigma) & (d_j > TAPER_JUNCTION_R * r + 2.5 * sigma))
        if not len(eligible):
            raise RuntimeError("no eligible stations for a bump")
        row = int(rng.choice(eligible)); sid = seg[row]; s0 = s[row]
        amp = float(rng.uniform(*amp_range)) * float(rng.choice([-1.0, 1.0]))
        m = seg == sid
        f[m] += amp * np.exp(-0.5 * ((s[m] - s0) / sigma) ** 2)
        spec.append({"segment_id": int(sid), "s_local_mm": float(s0), "amplitude": amp, "sigma_mm": sigma, "radius_at_center_mm": float(r[row])})
    # taper to 1 near openings and junctions (smoothstep)
    def smooth(x):
        x = np.clip(x, 0.0, 1.0); return x * x * (3.0 - 2.0 * x)
    w = smooth((d_end - TAPER_END_MM) / TAPER_END_MM) * smooth((d_j - TAPER_JUNCTION_R * r) / np.maximum(TAPER_JUNCTION_R * r, 1e-6))
    f = 1.0 + (f - 1.0) * w
    f = np.clip(f, F_MIN, F_MAX)
    return f, spec


def morph(parent: str, tag: str, seed: int, n_bumps: int, amp_range, sigma_range, dry_run: bool, out_root: Path) -> dict:
    t0 = time.time(); rng = np.random.default_rng(seed)
    raw = ROOT / "data_new" / parent; cas_path = ncs.fluent_case(parent)
    mesh = read_fluent_mesh(cas_path, keep_interior=True)
    T, cols, segs, h5path = load_atlas(parent)
    ci = {c: cols.index(c) for c in ("x_raw_mm", "y_raw_mm", "z_raw_mm")}
    atlas_xyz_m = T[:, [ci["x_raw_mm"], ci["y_raw_mm"], ci["z_raw_mm"]]] / 1000.0
    f_station, spec = modulation(T, cols, segs, rng, n_bumps, amp_range, sigma_range)
    # ---- radial scaling about the centerline, kernel-averaged over the K nearest stations with a soft branch weight
    # (d / R_k): a node inside limb A is < 1 R_A from its own centerline but > 1.3 R_B from a side-by-side limb B, so
    # foreign stations are suppressed ~50x while the field stays continuous. Harmonic / diffusion extensions (tried
    # 2026-09-26) distribute the compression non-uniformly and collapse coarse interior cells; pure radial scaling keeps
    # every cell's in-plane ratio at f^2 and the boundary-layer prisms move with the wall by construction.
    cz = mesh.cell_zone_array()
    from wss_pinn.v4.fluent_topology import resolve_anatomy_zone
    zid, _ = resolve_anatomy_zone(mesh, "blood", cz); anat = cz == zid
    node_anat = np.zeros(len(mesh.nodes_m), dtype=bool)
    for sec in mesh.face_sections:
        if not sec.attached:
            continue
        a = anat[sec.c0] | anat[np.where(sec.c1 > 0, sec.c1, 0)]
        if a.any():
            node_anat[sec.nodes[np.repeat(a, np.diff(sec.offsets))]] = True
    idx = np.flatnonzero(node_anat); X = mesh.nodes_m[idx]
    ci2 = {c: cols.index(c) for c in ("segment_id", "s_local_mm", "tangent_x", "tangent_y", "tangent_z", "radius_mm")}
    seg_st = T[:, ci2["segment_id"]].astype(int); s_st = T[:, ci2["s_local_mm"]]; r_st = T[:, ci2["radius_mm"]] / 1000.0
    tan = T[:, [ci2["tangent_x"], ci2["tangent_y"], ci2["tangent_z"]]]; tan = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    f_interp = {}
    for sid in np.unique(seg_st):
        rows = np.flatnonzero(seg_st == sid); order = np.argsort(s_st[rows], kind="stable"); xs = s_st[rows][order]; ys = f_station[rows][order]
        keep = np.r_[True, np.diff(xs) > 1e-9]; f_interp[int(sid)] = (xs[keep], ys[keep])
    K_NEAR, H_KERNEL_M, H_BRANCH = 12, 1.5e-3, 0.35
    d, st = cKDTree(atlas_xyz_m).query(X, k=K_NEAR)
    disp = np.zeros_like(X); wsum = np.zeros(len(X))
    for j in range(K_NEAR):
        k = st[:, j]; t_k = tan[k]; c_k = atlas_xyz_m[k]
        along = np.einsum("ij,ij->i", X - c_k, t_k); c_star = c_k + along[:, None] * t_k
        s_star = s_st[k] + along * 1000.0
        f_j = np.empty(len(X))
        for sid in np.unique(seg_st[k]):
            m = seg_st[k] == sid; xs, ys = f_interp[int(sid)]; f_j[m] = np.interp(s_star[m], xs, ys)
        rho_norm = np.linalg.norm(X - c_star, axis=1) / np.maximum(r_st[k], 1e-6)
        w = np.exp(-0.5 * (d[:, j] / H_KERNEL_M) ** 2) * np.exp(-0.5 * (rho_norm / H_BRANCH) ** 2) + 1e-30
        # only the tubular lumen is scaled: sac regions (rho > 1.2-1.8 R, e.g. a common sac straddling both CIA
        # centerlines) keep their shape, with a smooth hand-over
        g = np.clip((1.8 - rho_norm) / 0.6, 0.0, 1.0); g = g * g * (3.0 - 2.0 * g)
        disp += w[:, None] * (((f_j - 1.0) * g)[:, None] * (X - c_star)); wsum += w
    disp /= wsum[:, None]
    new_nodes = mesh.nodes_m.copy(); new_nodes[idx] = X + disp
    Xw = X; c_star = X - disp  # for the f summary below
    f = 1.0 + np.linalg.norm(disp, axis=1) / np.maximum(np.linalg.norm(X - atlas_xyz_m[st[:, 0]], axis=1), 1e-9) * np.sign(np.einsum("ij,ij->i", disp, X - atlas_xyz_m[st[:, 0]]) + 1e-30)
    # validity: signed cell volumes (strict=False so a bad morph is reported, not raised)
    _, vol0 = mesh.cell_centroids_and_volumes()
    moved = dataclasses.replace(mesh, nodes_m=new_nodes)
    moved._c0_sign_cache = mesh.c0_sign  # the file's c0 orientation convention does not change when nodes move; do not re-infer it on the morphed mesh
    _, vol1 = moved.cell_centroids_and_volumes(strict=False)
    ratio = vol1[1:] / np.maximum(vol0[1:], 1e-30)
    report = {"parent": parent, "child": child_id(parent, tag), "seed": seed, "bumps": spec, "atlas": h5path, "cas": str(cas_path),
              "nodes_moved": int(np.sum(np.linalg.norm(new_nodes[1:] - mesh.nodes_m[1:], axis=1) > 1e-9)), "f_min": float(f.min()), "f_max": float(f.max()),
              "cell_volume_ratio_min": float(ratio.min()), "cell_volume_ratio_max": float(ratio.max()), "negative_volumes": int(np.sum(vol1[1:] <= 0)),
              "anatomy_volume_ratio": float(vol1[1:][anat[1:]].sum() / vol0[1:][anat[1:]].sum()), "max_displacement_mm": float(np.linalg.norm(new_nodes[1:] - mesh.nodes_m[1:], axis=1).max() * 1000)}
    ok = report["negative_volumes"] == 0 and report["cell_volume_ratio_min"] >= MIN_VOLUME_RATIO and report["cell_volume_ratio_max"] <= MAX_VOLUME_RATIO
    report["valid"] = bool(ok)
    if dry_run or not ok:
        report["seconds"] = round(time.time() - t0, 1); return report
    # write child case: in-place node payload + path strings
    data = gzip.open(cas_path, "rb").read()
    m = NODE_RE.search(data); assert m, "node section header not found"
    first, last = int(m.group(2), 16), int(m.group(3), 16); dim = int(m.group(5), 16) if m.group(5) else 3
    count = last - first + 1; assert count == mesh.node_count and dim == 3, (count, mesh.node_count, dim)
    start = m.end(); payload = new_nodes[first:last + 1].astype("<f8").tobytes(); assert len(payload) == count * 3 * 8
    data = data[:start] + payload + data[start + len(payload):]
    child = child_id(parent, tag); cdir = out_root / child; cdir.mkdir(parents=True, exist_ok=False)
    # every absolute path string in the case (export prefixes, UDF source) must point into the child directory; library
    # parents still carry their original `.../GNN/data/<cohort>/...` prefixes, so rewrite by pattern, not by exact match
    PATH_RE = re.compile(rb"/public/newhome/cy/Digital_twin/GNN/data(?:_new)?/[^\"\s)]*")
    new_dir = str(cdir).encode(); found = sorted(set(PATH_RE.findall(data)))
    n_paths = 0
    for old in found:
        tail = old.split(b"/")[-1]                      # keep the file / prefix basename (e.g. ascii/NAME -> NAME with the ascii dir)
        sub = b"ascii/" + tail if b"/ascii/" in old else tail
        rep = new_dir + b"/" + sub
        n_paths += data.count(old); data = data.replace(old, rep)
    left = set(PATH_RE.findall(data)) - {x for x in PATH_RE.findall(data) if x.startswith(new_dir)}
    assert not left, f"foreign paths remain in the child case: {sorted(left)[:3]}"
    cas_child = cdir / cas_path.name
    with gzip.open(cas_child, "wb", compresslevel=6) as fh:
        fh.write(data)
    check = read_fluent_mesh(cas_child, keep_interior=False)
    assert np.allclose(check.nodes_m[1:], new_nodes[1:]), "re-read nodes differ"
    udf_src = ncs.udf_path(raw)                       # udf-inlet4.c / udf-inlet.c, else the compiled copy libudf/src/udf-inlet*.c
    shutil.copy2(udf_src, cdir / udf_src.name)
    jou = re.sub(r"(/file/read-case\s+)(\S+)", lambda m: m.group(1) + str(cas_child), (raw / "2.jou").read_text(), count=1)
    assert str(cas_child) in jou, "journal read-case rewrite failed"
    (cdir / "2.jou").write_text(jou); shutil.copy2(raw / "fluent.slurm", cdir / "fluent.slurm"); (cdir / "ascii").mkdir()
    report.update({"child_dir": str(cdir), "cas_child": str(cas_child), "path_strings_replaced": n_paths, "seconds": round(time.time() - t0, 1)})
    (cdir / "synth_morph.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent"); ap.add_argument("--child-tag", default="m01"); ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--plan", type=Path, help="json list of {parent, tag, seed, n_bumps, amp, sigma}")
    ap.add_argument("--n-bumps", type=int, default=2); ap.add_argument("--amp", type=float, nargs=2, default=[0.15, 0.40]); ap.add_argument("--sigma", type=float, nargs=2, default=[6.0, 14.0])
    ap.add_argument("--out-root", type=Path, default=ROOT / "data_new"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    jobs = json.loads(a.plan.read_text()) if a.plan else [dict(parent=a.parent, tag=a.child_tag, seed=a.seed, n_bumps=a.n_bumps, amp=a.amp, sigma=a.sigma)]
    reports = []
    for j in jobs:
        r = morph(j["parent"], j.get("tag", "m01"), int(j.get("seed", 20260926)), int(j.get("n_bumps", 2)), tuple(j.get("amp", a.amp)), tuple(j.get("sigma", a.sigma)), a.dry_run, a.out_root)
        reports.append(r); print(json.dumps({k: v for k, v in r.items() if k != "bumps"}), flush=True)
    if a.report:
        a.report.write_text(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()
