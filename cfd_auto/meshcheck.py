"""Mesh statistics on the anatomy region, computed identically for a library case and a cfd_auto mesh.

Boundary layer: wedge cells are assigned to their nearest wall face; faces carrying exactly ``n_layers`` wedges give the
layer-centre distances, hence per-layer heights (h1 = 2 d1, h_k = 2 (d_k - d_{k-1}) - h_{k-1}). Core: tetrahedra as an
equivalent edge length (6 sqrt(2) V)^(1/3), binned by distance to the wall.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

WALL_DIST_BINS_MM = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 1000))


def zname(mesh, zone_id: int) -> str:
    """Zone name without the ``)(`` tail that Fluent Meshing's ``(45 (id type name)())`` declarations leave in the
    library reader's regex."""
    return mesh.zone_name(zone_id).split(")")[0]


def wall_faces(mesh, wall_zone_names: tuple[str, ...] = ("wall",), cell_mask: np.ndarray | None = None):
    """Centres (m), unit normals (outward from the fluid) and areas of the wall faces (optionally only faces whose
    fluid cell is in ``cell_mask``)."""
    cen, nrm, area = [], [], []
    for sec in mesh.face_sections_of_type(3):
        if zname(mesh, sec.zone_id) not in wall_zone_names or not sec.attached:
            continue
        c, av = mesh.face_geometry(sec)
        keep = np.ones(len(c), bool) if cell_mask is None else cell_mask[sec.c0]
        a = np.linalg.norm(av, axis=1)
        cen.append(c[keep]); nrm.append((av / a[:, None])[keep]); area.append(a[keep])
    return np.concatenate(cen), np.concatenate(nrm), np.concatenate(area)


def stats(mesh, cell_mask: np.ndarray, wall_zone_names: tuple[str, ...] = ("wall",), n_layers: int = 10) -> dict:
    cen, vol = mesh.cell_centroids_and_volumes()
    fc, fn, fa = wall_faces(mesh, wall_zone_names, cell_mask)
    edge = np.sqrt(4 * fa / np.sqrt(3))
    wedge = cell_mask & (mesh.cell_types == 6)
    tet = cell_mask & (mesh.cell_types == 2)
    tree = cKDTree(fc)
    _, k = tree.query(cen[wedge])
    dn = np.abs(np.einsum("ij,ij->i", cen[wedge] - fc[k], fn[k]))
    cnt = np.bincount(k, minlength=len(fc))
    sel = np.flatnonzero(cnt[k] == n_layers)
    order = np.lexsort((dn[sel], k[sel]))
    D = dn[sel][order].reshape(-1, n_layers)
    first = k[sel][order].reshape(-1, n_layers)[:, 0]
    H = np.zeros_like(D); H[:, 0] = 2 * D[:, 0]
    for j in range(1, n_layers):
        H[:, j] = 2 * (D[:, j] - D[:, j - 1]) - H[:, j - 1]
    total = D[:, -1] + H[:, -1] / 2
    L = (6 * np.sqrt(2) * vol[tet]) ** (1 / 3)
    dt, _ = tree.query(cen[tet])
    bins = {}
    for lo, hi in WALL_DIST_BINS_MM:
        m = (dt >= lo * 1e-3) & (dt < hi * 1e-3)
        bins[f"{lo}-{hi}mm"] = {"n": int(m.sum()), "L_median_mm": round(float(np.median(L[m])) * 1e3, 3) if m.any() else None}
    types, counts = np.unique(mesh.cell_types[cell_mask], return_counts=True)
    return {
        "cells": int(cell_mask.sum()), "cell_types": {int(t): int(c) for t, c in zip(types, counts)},
        "volume_mm3": round(float(vol[cell_mask].sum()) * 1e9, 2),
        "wall_faces": int(len(fc)), "wall_area_mm2": round(float(fa.sum()) * 1e6, 2),
        "faces_with_n_layers": round(float(np.mean(cnt == n_layers)), 4),
        "layer_height_mm_median": (np.median(H, 0) * 1e3).round(4).tolist(),
        "growth_ratio_median": np.median(H[:, 1:] / H[:, :-1], 0).round(3).tolist(),
        "first_height_over_edge_median": round(float(np.median(H[:, 0] / edge[first])), 4),
        "bl_total_mm_p5_50_95": (np.percentile(total, [5, 50, 95]) * 1e3).round(3).tolist(),
        "tets": int(tet.sum()), "tet_L_mm_p5_25_50_75_95_max": (np.percentile(L, [5, 25, 50, 75, 95, 100]) * 1e3).round(3).tolist(),
        "tet_L_by_wall_distance": bins,
    }


def anatomy_mask(mesh) -> np.ndarray:
    """Library case: cells of the anatomy zone. cfd_auto anatomy-only mesh: every cell."""
    from wss_pinn.v4.fluent_topology import resolve_anatomy_zone
    cz = mesh.cell_zone_array()
    if len(mesh.cell_zone_ranges) > 1 and any(zname(mesh, z).startswith("blood") for z in mesh.cell_zone_ranges):
        zid, _ = resolve_anatomy_zone(mesh, "blood", cz)
        return cz == zid
    return cz > 0


def compare(ref: dict, new: dict) -> dict:
    """Relative differences new vs reference for the headline numbers."""
    rel = lambda a, b: round((b - a) / a, 4) if a else None
    return {
        "anatomy_cells": rel(ref["cells"], new["cells"]), "tets": rel(ref["tets"], new["tets"]), "volume": rel(ref["volume_mm3"], new["volume_mm3"]),
        "first_height_over_edge": rel(ref["first_height_over_edge_median"], new["first_height_over_edge_median"]),
        "bl_total_median": rel(ref["bl_total_mm_p5_50_95"][1], new["bl_total_mm_p5_50_95"][1]),
        "tet_L_median": rel(ref["tet_L_mm_p5_25_50_75_95_max"][2], new["tet_L_mm_p5_25_50_75_95_max"][2]),
        "tet_L_by_wall_distance": {k: rel(ref["tet_L_by_wall_distance"][k]["L_median_mm"] or 0, new["tet_L_by_wall_distance"][k]["L_median_mm"] or 0)
                                   for k in ref["tet_L_by_wall_distance"]},
    }


def prism_stacks(mesh, cell_mask: np.ndarray, wall_zone_names: tuple[str, ...] = ("wall",), max_layers: int = 40, sample: int | None = 4000, seed: int = 0) -> dict:
    """Boundary-layer stacks walked face by face (works for wedge and polyhedral prism layers): from a wall face into
    its cell, take the opposite face (same node count, |cos| > 0.95 with the wall normal, area ratio 0.5-2, moved inward);
    continue across it until no such face exists. Returns layer-count histogram and per-layer heights (median)."""
    secs = [s for s in mesh.face_sections if s.attached]
    cen, nrm, area, nn, c0, c1 = [], [], [], [], [], []
    for s in secs:
        c, av = mesh.face_geometry(s)
        a = np.linalg.norm(av, axis=1)
        cen.append(c); nrm.append(av / np.maximum(a, 1e-30)[:, None]); area.append(a); nn.append(np.diff(s.offsets)); c0.append(s.c0); c1.append(s.c1)
    cen, nrm, area, nn, c0, c1 = map(np.concatenate, (cen, nrm, area, nn, c0, c1))
    nf = len(cen)
    # cell -> faces (CSR)
    owner = np.concatenate([c0, c1]); fid = np.concatenate([np.arange(nf), np.arange(nf)])
    keep = owner > 0; owner, fid = owner[keep], fid[keep]
    order = np.argsort(owner, kind="stable"); owner, fid = owner[order], fid[order]
    ptr = np.searchsorted(owner, np.arange(mesh.cell_count + 2))
    wall_ids, off = [], 0
    for s in secs:
        if s.bc_type == 3 and zname(mesh, s.zone_id) in wall_zone_names:
            idx = np.arange(off, off + s.count)
            wall_ids.append(idx[cell_mask[s.c0]])
        off += s.count
    wall_ids = np.concatenate(wall_ids)
    if sample and len(wall_ids) > sample:
        wall_ids = np.random.default_rng(seed).choice(wall_ids, sample, replace=False)
    counts, heights, first_rel = [], [], []
    for f in wall_ids:
        n_in = -nrm[f] if np.dot(nrm[f], cen[f] - mesh.nodes_m[1:].mean(0)) < 0 else -nrm[f]
        cell = c0[f]; cur = f; h = []
        # inward normal: from the wall face towards its cell centre
        cf = fid[ptr[cell]:ptr[cell + 1]]
        n_in = -nrm[f] if np.dot(cen[cf].mean(0) - cen[f], -nrm[f]) > 0 else nrm[f]
        for _ in range(max_layers):
            cf = fid[ptr[cell]:ptr[cell + 1]]; cf = cf[cf != cur]
            if not len(cf):
                break
            disp = (cen[cf] - cen[cur]) @ n_in
            ok = (nn[cf] == nn[cur]) & (np.abs(nrm[cf] @ n_in) > 0.95) & (area[cf] / area[cur] > 0.5) & (area[cf] / area[cur] < 2) & (disp > 0)
            if not ok.any():
                break
            g = cf[ok][np.argmax(disp[ok])]
            h.append(float(disp[ok].max()))
            nxt = c1[g] if c0[g] == cell else c0[g]
            cur = g
            if nxt <= 0 or not cell_mask[nxt]:
                break
            cell = nxt
        counts.append(len(h)); heights.append(h); first_rel.append(h[0] / np.sqrt(area[f]) if h else np.nan)
    counts = np.array(counts)
    mode = int(np.bincount(counts).argmax())
    H = np.array([h[:mode] for h in heights if len(h) >= mode]) if mode else np.zeros((0, 0))
    return {"wall_faces_sampled": int(len(wall_ids)), "layer_count_hist": {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
            "layers_mode": mode, "height_mm_median": (np.median(H, 0) * 1e3).round(4).tolist() if len(H) else [],
            "growth_median": np.median(H[:, 1:] / H[:, :-1], 0).round(3).tolist() if len(H) and mode > 1 else [],
            "first_height_over_sqrt_area_median": float(np.nanmedian(first_rel)), "total_mm_median": float(np.median(H.sum(1)) * 1e3) if len(H) else 0.0}


GATE_TOL = {"dual_cells": 0.15, "dual_wall_faces": 0.10, "wall_edge_median": 0.10, "bl_first_over_edge": 0.10, "bl_total_mm": 0.15, "core_size": 0.15}


def gate_summary(mesh, cell_mask: np.ndarray, wall_zone_names: tuple[str, ...] = ("wall",)) -> dict:
    """Family-agnostic numbers for the mesh gate (same code on the library mesh and the rebuilt one)."""
    cen, vol = mesh.cell_centroids_and_volumes()
    fc, _, fa = wall_faces(mesh, wall_zone_names, cell_mask)
    ps = prism_stacks(mesh, cell_mask, wall_zone_names, sample=2000)
    d = cKDTree(fc).query(cen[cell_mask])[0] * 1e3
    L = np.cbrt(vol[cell_mask]) * 1e3
    core = {}
    for lo, hi in WALL_DIST_BINS_MM:
        m = (d >= lo) & (d < hi)
        core[f"{lo}-{hi}mm"] = {"n": int(m.sum()), "size_median_mm": round(float(np.median(L[m])), 4) if m.sum() else None}
    types, counts = np.unique(mesh.cell_types[cell_mask], return_counts=True)
    poly = int(np.sum(mesh.cell_types[cell_mask] == 7)) > 0.5 * int(cell_mask.sum())
    # dual-equivalent resolution: a polyhedral (dual) mesh has one cell per node and one wall polygon per wall node of the
    # triangle/tet mesh it came from, so poly cells ~ tet-mesh nodes and poly wall faces ~ triangle-wall nodes
    wall_nodes = 0
    for sec in mesh.face_sections_of_type(3):
        if zname(mesh, sec.zone_id) in wall_zone_names and sec.attached:
            keep = cell_mask[sec.c0]
            _, nodes = sec.subset_connectivity(np.flatnonzero(keep)); wall_nodes += len(np.unique(nodes))
    vol_nodes = 0
    if not poly:
        used = np.zeros(len(mesh.nodes_m), bool)
        for sec in mesh.face_sections:
            if sec.attached:
                m = cell_mask[sec.c0] | cell_mask[np.where(sec.c1 > 0, sec.c1, 0)]
                if m.any():
                    used[sec.nodes[np.repeat(m, np.diff(sec.offsets))]] = True
        vol_nodes = int(used.sum())
    tri = np.sqrt(4 * fa / np.sqrt(3)); dual = np.sqrt(fa) / 0.93
    edge = dual if poly else tri
    return {"cells": int(cell_mask.sum()), "cell_types": {int(t): int(c) for t, c in zip(types, counts)}, "poly": poly,
            "dual_cells": int(cell_mask.sum()) if poly else vol_nodes, "dual_wall_faces": int(len(fc)) if poly else wall_nodes,
            "wall_edge_equiv_mm_p5_50_95": (np.percentile(edge, [5, 50, 95]) * 1e3).round(4).tolist(),
            "bl_first_over_edge": round(ps["first_height_over_sqrt_area_median"] * (0.93 if poly else 1 / 1.5197), 4), "wall_faces": int(len(fc)),
            "wall_area_mm2": round(float(fa.sum()) * 1e6, 1), "wall_face_size_mm_p5_50_95": (np.percentile(np.sqrt(fa), [5, 50, 95]) * 1e3).round(4).tolist(),
            "bl_layers_mode": ps["layers_mode"], "bl_layer_hist": ps["layer_count_hist"], "bl_first_over_sqrt_area": round(ps["first_height_over_sqrt_area_median"], 4),
            "bl_total_mm": round(ps["total_mm_median"], 4), "bl_growth": ps["growth_median"], "core_size_by_wall_distance": core}


def gate_compare(ref: dict, new: dict) -> tuple[dict, dict]:
    rel = lambda a, b: round((b - a) / a, 4) if a else None
    out = {"dual_cells": rel(ref["dual_cells"], new["dual_cells"]), "dual_wall_faces": rel(ref["dual_wall_faces"], new["dual_wall_faces"]),
           "wall_edge_median": rel(ref["wall_edge_equiv_mm_p5_50_95"][1], new["wall_edge_equiv_mm_p5_50_95"][1]),
           "bl_first_over_edge": rel(ref["bl_first_over_edge"], new["bl_first_over_edge"]), "bl_total_mm": rel(ref["bl_total_mm"], new["bl_total_mm"]),
           "bl_layers": [ref["bl_layers_mode"], new["bl_layers_mode"]]}
    core = {}
    if ref["poly"] == new["poly"]:                       # cell sizes are only comparable within one cell family
        for k, v in ref["core_size_by_wall_distance"].items():
            w = new["core_size_by_wall_distance"].get(k, {})
            if v["n"] > 100 and w.get("n", 0) > 100:
                core[k] = rel(v["size_median_mm"], w["size_median_mm"])
    out["core_size"] = core
    fails = {k: out[k] for k in ("dual_cells", "dual_wall_faces", "wall_edge_median", "bl_first_over_edge", "bl_total_mm") if out[k] is None or abs(out[k]) > GATE_TOL[k]}
    if out["bl_layers"][0] != out["bl_layers"][1]:
        fails["bl_layers"] = out["bl_layers"]
    bad_core = {k: v for k, v in core.items() if abs(v) > GATE_TOL["core_size"]}
    if bad_core:
        fails["core_size"] = bad_core
    return out, fails


def read_boundary_msh(path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Surface-only Fluent mesh (Fluent Meshing boundary mesh, no cells): node coordinates (m, row 0 unused) and
    triangle connectivity per zone name. Uses the library reader's section parsers (which require cell zones)."""
    from pathlib import Path
    from wss_pinn.v4 import fluent_topology as ft
    data = ft._read_bytes(Path(path))
    nodes, faces, names, pos = {}, {}, {}, 0
    while True:
        start = data.find(b"(", pos)
        if start < 0:
            break
        m = ft.HEADER_RE.match(data, start)
        if m is None or (start > 0 and data[start - 1:start] not in (b"\n", b"\r")):
            pos = start + 1; continue
        idx = int(m.group(1)); fields = m.group(2).split()
        if idx in (39, 45):
            d = ft.ZONE_DECL_RE.match(data, start)
            if d:
                names[int(d.group(2))] = d.group(4).decode().split(")")[0]
            pos = m.end(); continue
        if not fields or not all(ft.HEX_RE.match(v) for v in fields):
            pos = start + 1; continue
        pos = m.end(); h = [int(v, 16) for v in fields]
        if idx in (3010, 2010) and h[0]:
            n = h[2] - h[1] + 1; dt = np.dtype("<f8" if idx == 3010 else "<f4"); ps = data.index(b"(", pos) + 1
            nodes[(h[1], h[2])] = np.frombuffer(data[ps:ps + n * 3 * dt.itemsize], dt).reshape(n, 3).astype(float)
            pos = ft._binary_end(data, ps + n * 3 * dt.itemsize, idx)
        elif idx in (3013, 2013) and h[0]:
            n = h[2] - h[1] + 1; ps = data.index(b"(", pos) + 1; mk = data.find(ft.END_MARKER, ps)
            buf = np.frombuffer(data[ps:mk][: (mk - ps) // 4 * 4], "<i4"); pos = mk + len(ft.END_MARKER)
            off, fn, _, _ = ft._decode_faces(buf, n, h[4] if len(h) > 4 else 0)
            faces[h[0]] = (off, fn)
    total = max(l for _, l in nodes); P = np.full((total + 1, 3), np.nan)
    for (a, b), blk in nodes.items():
        P[a:b + 1] = blk
    out = {}
    for zid, (off, fn) in faces.items():
        cnt = np.diff(off)
        if np.all(cnt == 3):
            out[names.get(zid, str(zid))] = fn.reshape(-1, 3)
    return P, out
