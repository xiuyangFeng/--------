#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体场点云 → 病例自己的 Fluent 体网格节点 → EnSight Gold 瞬态 case（时间轴 = 心动相位）。

为什么要插值：``interior_pointcloud.vtp`` 只有点、没有体单元，EnSight / ParaView 的 Clip
只能在单元内部插值，对纯点云切出来就是散点。这里把点云值放回它原本所在的 CFD 网格：

* 网格 = 该例 Fluent ``.cas`` 里解剖区 ``blood`` 的全部单元（不含 blood1–5 延长段），
  原样按 Fluent 面表组装成 EnSight ``nfaced`` 多面体（hex 也按 6 面体写），面法向统一朝外；
  坐标做 V5 刚体配准（毫米），与点云、``aligned_geometry.stl`` 同一坐标系。
* 体内点云 677082 个点 = 这些单元的体积中心（按 ``cell_id_cas`` 身份对应，逐点核对坐标）。
* 节点值 = 相邻单元中心值按 1/距离 加权平均（Fluent / CFD-Post 的 node value 做法）；
  壁面节点：速度 = 0（无滑移边界条件），压力 = 壁面点云自身的值（点云里本来就有壁面节点）。
* 每个相位写一个时间步（time value = 周期内时刻 s），变量名不带相位，EnSight 拖时间条即换相位。

只用于显示；正式 R² / MAE 仍在同点 CSV / 点云上算。

用法（GNN 环境）::

    python training_wss_min/tools/build_ensight_cfd_mesh_case.py \
        --case-dir training_wss_min/experiments/volume_time_20260919/postview_median_phases_20260921/AAA__unruputer__SHEN_FANG_JIN
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from wss_v5 import contract as VC  # noqa: E402
from wss_v5.mesh_topology import load_mesh  # noqa: E402

ENS_CHECKER = Path("/1/ansys_inc/2023r1/v231/CEI/bin/ens_checker231")
VELOCITY_REF_ARM = "VT0"
PRESSURE_REF_ARM = "PT0"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- inputs
def read_vtp(path: Path, names: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import pyvista as pv

    poly = pv.read(path)
    missing = [n for n in names if n not in poly.point_data]
    if missing:
        raise RuntimeError(f"{path}: missing arrays {missing}")
    return np.asarray(poly.points, dtype=np.float64), {n: np.asarray(poly.point_data[n]) for n in names}


def discover(case_dir: Path) -> dict:
    manifest = json.loads((case_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    selection = manifest["selection"]
    canonical = selection["case"] if isinstance(selection, dict) and "case" in selection else None
    if canonical is None:
        canonical = case_dir.name.replace("__", "/")
    phases = [p["name"] for p in manifest["phases"]]
    frames = {p["name"]: int(p["frame_index"]) for p in manifest["phases"]}
    steps = {p["name"]: int(p["step"]) for p in manifest["phases"]}
    q_norm = {p["name"]: float(p["q_norm"]) for p in manifest["phases"]}

    def arms(field: str, probe: str) -> list[str]:
        root = case_dir / field
        return sorted(d.name for d in root.iterdir() if (d / phases[0] / probe).is_file())

    vel_arms = arms("velocity", "interior_pointcloud.vtp")
    pres_arms = arms("pressure", "full_pointcloud.vtp")
    if VELOCITY_REF_ARM not in vel_arms or PRESSURE_REF_ARM not in pres_arms:
        raise RuntimeError(f"reference arms missing: velocity {vel_arms}, pressure {pres_arms}")
    return {
        "canonical": canonical,
        "case_tag": case_dir.name,
        "phases": phases,
        "frames": frames,
        "steps": steps,
        "q_norm": q_norm,
        "velocity_arms": vel_arms,
        "pressure_arms": pres_arms,
    }


# --------------------------------------------------------------------------- mesh
def lumen_polyhedra(mesh, zone_id: int) -> dict:
    """Blood-zone cells as outward-oriented polyhedra straight from the Fluent face table."""
    zone = mesh.cell_zone_array()
    in_zone = zone == int(zone_id)
    sign0 = mesh.c0_sign  # +1: right-hand normal points out of c0
    owner, counts, nodes_parts, flip = [], [], [], []
    for section in mesh.face_sections:
        if section.bc_type == 31 or not section.attached:  # parent faces duplicate children
            continue
        n_per = section.node_counts
        for cells, outward_as_stored in ((section.c0, sign0 > 0), (section.c1, sign0 < 0)):
            idx = np.flatnonzero((cells > 0) & in_zone[np.maximum(cells, 0)])
            if not len(idx):
                continue
            _, sub_nodes = section.subset_connectivity(idx)
            owner.append(cells[idx].astype(np.int64))
            counts.append(n_per[idx].astype(np.int64))
            nodes_parts.append(sub_nodes.astype(np.int64))
            flip.append(np.full(len(idx), not outward_as_stored))
    owner = np.concatenate(owner)
    counts = np.concatenate(counts)
    face_nodes = np.concatenate(nodes_parts)
    flip = np.concatenate(flip)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    order = np.argsort(owner, kind="stable")
    owner, counts, starts, flip = owner[order], counts[order], starts[order], flip[order]
    local = np.arange(int(counts.sum())) - np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    rep_counts = np.repeat(counts, counts)
    local = np.where(np.repeat(flip, counts), rep_counts - 1 - local, local)
    conn = face_nodes[np.repeat(starts, counts) + local]

    cell_ids, faces_per_cell = np.unique(owner, return_counts=True)
    expected = np.flatnonzero(in_zone)
    if not np.array_equal(cell_ids, expected):
        raise RuntimeError(f"zone cells with faces {len(cell_ids)} != zone cells {len(expected)}")
    return {
        "cell_ids": cell_ids,             # Fluent 1-based cell ids, ascending
        "faces_per_cell": faces_per_cell,
        "nodes_per_face": counts,
        "conn_fluent": conn,              # Fluent 1-based node ids
        "face_owner_row": np.repeat(np.arange(len(cell_ids)), faces_per_cell),
    }


def polyhedra_volumes(xyz: np.ndarray, poly: dict) -> np.ndarray:
    """Divergence-theorem volume per cell (positive ⇔ faces point outward)."""
    counts = poly["nodes_per_face"]
    conn = poly["conn_local0"]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    p = xyz[conn]
    nxt = np.arange(len(conn)) + 1
    last = starts + counts - 1
    nxt[last] = starts
    cross = np.cross(p, xyz[conn[nxt]])
    area2 = np.add.reduceat(cross, starts, axis=0)          # 2 × area vector (Newell)
    centre = np.add.reduceat(p, starts, axis=0) / counts[:, None]
    face_term = np.einsum("ij,ij->i", centre, area2) / 6.0   # (c·A)/3 with A = area2/2
    return np.bincount(poly["face_owner_row"], weights=face_term, minlength=len(poly["cell_ids"]))


def open_or_misoriented_cells(poly: dict, n_nodes: int) -> int:
    """Cells whose edges are not each used exactly twice in opposite directions (closed, consistent)."""
    counts = poly["nodes_per_face"]
    conn = poly["conn_local0"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    nxt = np.arange(len(conn)) + 1
    nxt[starts + counts - 1] = starts
    a, b = conn, conn[nxt]
    cell = np.repeat(poly["face_owner_row"], counts).astype(np.int64)
    k1 = cell * n_nodes + np.minimum(a, b)  # (cell, low node); high node sorted separately → no int64 overflow
    k2 = np.maximum(a, b)
    order = np.lexsort((k2, k1))
    k1, k2 = k1[order], k2[order]
    first = np.flatnonzero(np.r_[True, (k1[1:] != k1[:-1]) | (k2[1:] != k2[:-1])])
    uses = np.diff(np.r_[first, len(k1)])
    net = np.add.reduceat(np.where(a < b, 1, -1)[order], first)
    bad = (uses != 2) | (net != 0)
    return int(len(np.union1d(k1[first[bad]] // n_nodes, cell[a == b])))


# --------------------------------------------------------------------------- EnSight Gold (C binary)
def _s80(text: str) -> bytes:
    raw = text.encode("ascii")
    if len(raw) > 80:
        raise ValueError(f"EnSight line longer than 80 chars: {text!r}")
    return raw.ljust(80, b"\0")


def _i32(values) -> bytes:
    return np.ascontiguousarray(values, dtype="<i4").tobytes()


def _f32(values) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def write_geo(path: Path, xyz: np.ndarray, poly: dict, part_desc: str, desc: tuple[str, str]) -> None:
    with open(path, "wb") as fh:
        fh.write(_s80("C Binary"))
        fh.write(_s80(desc[0]))
        fh.write(_s80(desc[1]))
        fh.write(_s80("node id off"))
        fh.write(_s80("element id off"))
        fh.write(_s80("part"))
        fh.write(_i32([1]))
        fh.write(_s80(part_desc))
        fh.write(_s80("coordinates"))
        fh.write(_i32([len(xyz)]))
        for axis in range(3):
            fh.write(_f32(xyz[:, axis]))
        fh.write(_s80("nfaced"))
        fh.write(_i32([len(poly["cell_ids"])]))
        fh.write(_i32(poly["faces_per_cell"]))
        fh.write(_i32(poly["nodes_per_face"]))
        fh.write(_i32(poly["conn_local0"] + 1))


def write_node_var(path: Path, values: np.ndarray, desc: str) -> None:
    values = np.asarray(values, dtype=np.float32)
    with open(path, "wb") as fh:
        fh.write(_s80(desc))
        fh.write(_s80("part"))
        fh.write(_i32([1]))
        fh.write(_s80("coordinates"))
        if values.ndim == 1:
            fh.write(_f32(values))
        else:
            for axis in range(3):
                fh.write(_f32(values[:, axis]))


# --------------------------------------------------------------------------- previews
def slice_previews(case_path: Path, time_values: list[float], phases: list[str], planes: list[dict],
                   groups: list[dict], out_dir: Path) -> dict:
    """Read the written case back with VTK and draw plane cuts (Gouraud = EnSight smooth shading)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import pyvista as pv

    reader = pv.get_reader(str(case_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"pngs": [], "readback": {}}
    cuts: dict[tuple[str, str], pv.PolyData] = {}
    for phase, tval in zip(phases, time_values):
        reader.set_active_time_value(tval)
        data = reader.read()
        grid = data[0] if isinstance(data, pv.MultiBlock) else data
        report["readback"][phase] = {"cells": int(grid.n_cells), "points": int(grid.n_points),
                                     "arrays": sorted(grid.point_data.keys())}
        for plane in planes:
            cut = grid.slice(normal=plane["normal"], origin=plane["origin"]).triangulate()
            cuts[(phase, plane["key"])] = cut
    for plane in planes:
        e1 = np.asarray(plane["e1"], dtype=np.float64)
        e2 = np.asarray(plane["e2"], dtype=np.float64)
        span = np.ptp(np.column_stack([(np.asarray(cuts[(phases[0], plane["key"])].points)
                                        - np.asarray(plane["origin"])) @ e for e in (e1, e2)]), axis=0)
        panel_w = 3.0 * float(np.clip(span[0] / max(span[1], 1e-6), 0.45, 1.4))
        for group in groups:
            cols = group["columns"]
            fig, axes = plt.subplots(len(phases), len(cols),
                                     figsize=(max(panel_w * len(cols) + 1.2, 6.0), 3.0 * len(phases) + 0.4),
                                     dpi=130, squeeze=False, constrained_layout=True)
            for r, phase in enumerate(phases):
                cut = cuts[(phase, plane["key"])]
                xyz = np.asarray(cut.points) - np.asarray(plane["origin"])
                uv = np.column_stack([xyz @ e1, xyz @ e2])
                tri = mtri.Triangulation(uv[:, 0], uv[:, 1], cut.faces.reshape(-1, 4)[:, 1:])
                vmin, vmax = group["range"](cut, phase)
                for c, name in enumerate(cols):
                    ax = axes[r, c]
                    tpc = ax.tripcolor(tri, np.asarray(cut.point_data[name]), shading="gouraud",
                                       cmap=group["cmap"], vmin=vmin, vmax=vmax)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r == 0:
                        ax.set_title(name, fontsize=10)
                    if c == 0:
                        ax.set_ylabel(phase, fontsize=10)
                fig.colorbar(tpc, ax=axes[r, :].tolist(), fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
            where, _, why = plane["label"].partition(" (")
            fig.suptitle(f"{group['title']}\nplane {where}  ({why}\nread back from the EnSight case", fontsize=9)
            png = out_dir / f"preview_{group['key']}_{plane['key']}.png"
            fig.savefig(png)
            plt.close(fig)
            report["pngs"].append(png.name)
    return report


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", type=Path, required=True, help="postview 病例目录（含 velocity/ pressure/）")
    ap.add_argument("--out-name", default="ensight", help="输出子目录名（ASCII，EnSight 在 Windows 上不认中文路径）")
    ap.add_argument("--skip-preview", action="store_true")
    args = ap.parse_args()
    t_all = time.time()
    case_dir = args.case_dir.resolve()
    info = discover(case_dir)
    phases = info["phases"]
    short = info["canonical"].split("/")[-1]
    out = case_dir / args.out_name
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first (no silent overwrite)")
    tmp = case_dir / f".{args.out_name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "vars").mkdir(parents=True)
    log(f"case {info['canonical']}  phases {phases}  velocity arms {info['velocity_arms']}  "
        f"pressure arms {info['pressure_arms']}")

    # ---- Fluent mesh
    audit = json.loads((VC.TOPOLOGY_AUDIT_DIR / f"{info['case_tag']}.json").read_text())
    cas_path = Path(audit["fluent_case"]["path"])
    t0 = time.time()
    md = load_mesh(cas_path)
    mesh = md.mesh
    if mesh.zone_name(md.anatomy_zone_id) != "blood":
        raise RuntimeError(f"anatomy zone is {mesh.zone_name(md.anatomy_zone_id)!r}, expected 'blood'")
    log(f"read {cas_path.name}: {mesh.cell_count:,} cells / {mesh.node_count:,} nodes ({time.time() - t0:.0f}s)")
    poly = lumen_polyhedra(mesh, md.anatomy_zone_id)
    node_ids, conn_local0 = np.unique(poly["conn_fluent"], return_inverse=True)
    poly["conn_local0"] = conn_local0.astype(np.int64)
    n_cells, n_nodes = len(poly["cell_ids"]), len(node_ids)
    log(f"blood zone: {n_cells:,} polyhedra, {len(poly['nodes_per_face']):,} cell-faces, {n_nodes:,} nodes")

    # ---- V5 aligned frame (mm)
    bundle = VC.SNAPSHOT_ROOT.parent / "views_v5_1/wss_min_view_v1" / info["canonical"] / "bundle.npz"
    with np.load(bundle, allow_pickle=True) as b:
        centroid = b["transform_centroid"].astype(np.float64)
        rotation = b["transform_rotation"].astype(np.float64)
        wall_node_id_cas = b["wall_node_id_cas"].astype(np.int64)
    wall_xyz_pc, _ = read_vtp(case_dir / "pressure" / PRESSURE_REF_ARM / phases[0] / "wall_pointcloud.vtp", ())
    raw_wall_mm = mesh.nodes_m[wall_node_id_cas] * VC.LENGTH_M_TO_MM
    err = {k: float(np.abs((raw_wall_mm - centroid) @ R - wall_xyz_pc).max())
           for k, R in (("@R", rotation), ("@R.T", rotation.T))}
    conv = min(err, key=err.get)
    R_apply = rotation if conv == "@R" else rotation.T
    if err[conv] > 1e-2:
        raise RuntimeError(f"wall point cloud not reproduced from Fluent wall nodes: {err}")
    log(f"frame {conv}: Fluent wall nodes → wall point cloud max |Δ| {err[conv]:.2e} mm")
    xyz = (mesh.nodes_m[node_ids] * VC.LENGTH_M_TO_MM - centroid) @ R_apply

    vol = polyhedra_volumes(xyz, poly)
    vol_fluent = md.volumes_m3[poly["cell_ids"]] * 1e9
    rel = np.abs(vol - vol_fluent) / vol_fluent
    log(f"orientation: {int((vol <= 0).sum())} non-positive cells; volume {vol.sum():.0f} mm³ "
        f"(Fluent {vol_fluent.sum():.0f}); per-cell rel diff p99 {np.percentile(rel, 99):.2e}")
    if np.any(vol <= 0):
        raise RuntimeError("some polyhedra are inside-out after orientation")
    n_open = open_or_misoriented_cells(poly, n_nodes)
    log(f"closed & consistently oriented: {n_cells - n_open:,}/{n_cells:,} cells")
    if n_open:
        raise RuntimeError(f"{n_open} polyhedra are open or have inconsistent face orientation")

    # ---- point cloud rows ↔ cells
    import h5py

    with h5py.File(VC.case_dir(info["canonical"]) / "case.h5", "r") as h5:
        cell_id_cas = h5["volume_static/cell_id_cas"][()].astype(np.int64)
    lookup = np.full(int(max(cell_id_cas.max(), poly["cell_ids"].max())) + 1, -1, dtype=np.int64)
    lookup[cell_id_cas] = np.arange(len(cell_id_cas))
    row_of_cell = lookup[poly["cell_ids"]]
    if np.any(row_of_cell < 0) or len(np.unique(row_of_cell)) != n_cells or len(cell_id_cas) != n_cells:
        raise RuntimeError("point cloud rows do not cover the blood cells one-to-one")
    interior_xyz, _ = read_vtp(case_dir / "velocity" / VELOCITY_REF_ARM / phases[0] / "interior_pointcloud.vtp", ())
    centre = (md.centroids_m[poly["cell_ids"]] * VC.LENGTH_M_TO_MM - centroid) @ R_apply
    d_id = np.linalg.norm(centre - interior_xyz[row_of_cell], axis=1)
    log(f"identity: Fluent volume centroid ↔ point cloud max |Δ| {d_id.max():.2e} mm")
    if d_id.max() > 1e-2:
        raise RuntimeError("point cloud is not the Fluent cell-centroid set")

    # ---- cell→node inverse-distance weights (unique cell–node pairs)
    cell_of_entry = np.repeat(poly["face_owner_row"], poly["nodes_per_face"])
    key = np.unique(cell_of_entry * np.int64(n_nodes) + poly["conn_local0"])
    pair_cell, pair_node = key // n_nodes, key % n_nodes
    dist = np.linalg.norm(xyz[pair_node] - centre[pair_cell], axis=1)
    w = 1.0 / np.maximum(dist, 1e-6)
    wsum = np.bincount(pair_node, weights=w, minlength=n_nodes)
    pair_row = row_of_cell[pair_cell]

    def idw(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            return np.bincount(pair_node, weights=w * values[pair_row], minlength=n_nodes) / wsum
        return np.column_stack([np.bincount(pair_node, weights=w * values[pair_row, k], minlength=n_nodes)
                                for k in range(values.shape[1])]) / wsum[:, None]

    node_lookup = np.full(mesh.node_count + 1, -1, dtype=np.int64)
    node_lookup[node_ids] = np.arange(n_nodes)
    wall_local = node_lookup[wall_node_id_cas]
    if np.any(wall_local < 0):
        raise RuntimeError("some wall point-cloud nodes are not nodes of the blood zone")
    is_wall = np.zeros(n_nodes, dtype=bool)
    is_wall[wall_local] = True
    log(f"nodes: {n_nodes:,} total, {int(is_wall.sum()):,} wall (= wall point cloud), "
        f"pairs {len(key):,}")

    # ---- variables per phase
    vel_labels = ["CFD"] + info["velocity_arms"]
    pres_labels = ["CFD"] + info["pressure_arms"]
    var_specs: list[tuple[str, str, str]] = []  # (kind, name, description)
    for lab in vel_labels:
        var_specs.append(("scalar", f"speed_{lab}", f"|u| {lab} m/s node-interpolated"))
    for arm in info["velocity_arms"]:
        var_specs.append(("scalar", f"speed_err_{arm}", f"|u| {arm} minus CFD m/s"))
    for lab in vel_labels:
        var_specs.append(("vector", f"velocity_{lab}", f"velocity {lab} m/s (aligned mm frame)"))
    for lab in pres_labels:
        var_specs.append(("scalar", f"pressure_{lab}", f"p - p_volmean(t) {lab} Pa"))
    for arm in info["pressure_arms"]:
        var_specs.append(("scalar", f"pres_err_{arm}", f"pressure {arm} minus CFD Pa"))

    stats: dict[str, dict] = {}
    fidelity: dict[str, dict] = {}
    for ti, phase in enumerate(phases):
        fields: dict[str, np.ndarray] = {}
        cloud_speed: dict[str, np.ndarray] = {}
        cfd_ref = None
        for arm in info["velocity_arms"]:
            _, a = read_vtp(case_dir / "velocity" / arm / phase / "interior_pointcloud.vtp",
                            ("velocity_cfd_aligned_m_s", "velocity_pred_aligned_m_s"))
            if cfd_ref is None:
                cfd_ref = a["velocity_cfd_aligned_m_s"].astype(np.float64)
            elif np.abs(a["velocity_cfd_aligned_m_s"] - cfd_ref).max() > 1e-6:
                raise RuntimeError(f"{arm}/{phase}: CFD velocity differs from {VELOCITY_REF_ARM}")
            fields[f"velocity_{arm}"] = a["velocity_pred_aligned_m_s"].astype(np.float64)
        fields["velocity_CFD"] = cfd_ref
        for lab in vel_labels:
            cloud_speed[lab] = np.linalg.norm(fields[f"velocity_{lab}"], axis=1)
            node_vec = idw(fields[f"velocity_{lab}"])
            node_vec[is_wall] = 0.0  # no-slip wall
            fields[f"velocity_{lab}"] = node_vec
            fields[f"speed_{lab}"] = np.linalg.norm(node_vec, axis=1)
        for arm in info["velocity_arms"]:
            fields[f"speed_err_{arm}"] = fields[f"speed_{arm}"] - fields["speed_CFD"]

        p_cfd = None
        for arm in info["pressure_arms"]:
            _, a = read_vtp(case_dir / "pressure" / arm / phase / "full_pointcloud.vtp",
                            ("pressure_cfd", "pressure_pred"))
            n_wall = len(wall_node_id_cas)
            if p_cfd is None:
                p_cfd = a["pressure_cfd"].astype(np.float64)
            elif np.abs(a["pressure_cfd"] - p_cfd).max() > 1e-3:
                raise RuntimeError(f"{arm}/{phase}: CFD pressure differs from {PRESSURE_REF_ARM}")
            fields[f"pressure_{arm}"] = a["pressure_pred"].astype(np.float64)
        fields["pressure_CFD"] = p_cfd
        for lab in pres_labels:
            full = fields[f"pressure_{lab}"]
            node_p = idw(full[n_wall:])
            node_p[wall_local] = full[:n_wall]  # wall nodes carry their own point-cloud value
            fields[f"pressure_{lab}"] = node_p
        for arm in info["pressure_arms"]:
            fields[f"pres_err_{arm}"] = fields[f"pressure_{arm}"] - fields["pressure_CFD"]

        # fidelity: node field averaged back to cells vs the original point value (display check only)
        fid = {}
        for lab in vel_labels:
            back = np.bincount(pair_cell, weights=w * fields[f"speed_{lab}"][pair_node], minlength=n_cells) \
                / np.bincount(pair_cell, weights=w, minlength=n_cells)
            orig = cloud_speed[lab][row_of_cell]
            fid[f"speed_{lab}"] = float(1 - np.sum((back - orig) ** 2) / np.sum((orig - orig.mean()) ** 2))
        fidelity[phase] = fid

        for kind, name, desc in var_specs:
            write_node_var(tmp / "vars" / f"{name}.{ti:04d}", fields[name], f"{name} {phase}: {desc}"[:80])
            v = fields[name] if kind == "scalar" else np.linalg.norm(fields[name], axis=1)
            st = stats.setdefault(name, {})
            st[phase] = {"min": float(v.min()), "p01": float(np.percentile(v, 1)),
                         "p99": float(np.percentile(v, 99)), "max": float(v.max())}
        log(f"  {phase}: wrote {len(var_specs)} node variables; back-to-cell R² "
            + ", ".join(f"{k} {v:.3f}" for k, v in fid.items()))

    # ---- geometry + case file
    geo_name = f"{short}_lumen.geo"
    write_geo(tmp / geo_name, xyz, poly, "lumen (Fluent blood zone)",
              (f"{info['canonical']} Fluent anatomy blood zone, V5 aligned frame, mm",
               "point cloud values interpolated to mesh nodes (display only)"))
    frame_dt = VC.STEP_DT_S * 2
    time_values = [round(info["frames"][p] * frame_dt, 6) for p in phases]
    lines = ["FORMAT", "type: ensight gold", "", "GEOMETRY", f"model: {geo_name}", "", "VARIABLE"]
    # No "constant per case" lines: ens_checker231 wants transient values on the next line, VTK/ParaView
    # rejects exactly that. Phase ↔ time value is documented in the README / build_report instead.
    for kind, name, _ in var_specs:
        lines.append(f"{kind} per node: 1 {name} vars/{name}.****")
    lines += ["", "TIME", "time set: 1 cardiac_phase",
              f"number of steps: {len(phases)}", "filename start number: 0", "filename increment: 1",
              "time values: " + " ".join(f"{t:g}" for t in time_values), ""]
    case_name = f"{short}_flow.case"
    (tmp / case_name).write_text("\n".join(lines), encoding="ascii")

    # ---- checks on the written files
    checker = {"ran": False}
    if ENS_CHECKER.is_file():
        res = subprocess.run([str(ENS_CHECKER), case_name], cwd=tmp, capture_output=True, text=True,
                             timeout=1800, input="\n")
        text = res.stdout + res.stderr
        (tmp / "ens_checker.log").write_text(text, encoding="utf-8")
        # ens_checker exits 1 even on success; the verdict is the banner text.
        checker = {"ran": True, "passed": "Data format verification SUCCESSFUL" in text,
                   "no_warnings": "with No Warnings" in text}
        log(f"ens_checker231: {checker}")
        if not checker["passed"]:
            raise RuntimeError(f"ens_checker failed; see {tmp / 'ens_checker.log'}")

    # recommended planes (aligned frame: z = main vessel axis)
    wall_xyz = xyz[is_wall]
    z_levels = np.linspace(np.percentile(wall_xyz[:, 2], 5), np.percentile(wall_xyz[:, 2], 95), 91)
    band = 1.0
    width = []
    for z in z_levels:
        sel = wall_xyz[np.abs(wall_xyz[:, 2] - z) < band]
        width.append(np.ptp(sel[:, 0]) * np.ptp(sel[:, 1]) if len(sel) > 10 else 0.0)
    z_max = float(z_levels[int(np.argmax(width))])
    sel = wall_xyz[np.abs(wall_xyz[:, 2] - z_max) < band]
    cx = float(sel[:, 0].min() + sel[:, 0].max()) / 2
    cy = float(sel[:, 1].min() + sel[:, 1].max()) / 2
    # preview axes: horizontal e1, vertical e2 (+Z up = towards the inlet in the V5 frame)
    planes = [
        {"key": "Z", "label": f"Z = {z_max:.1f} mm (transverse, widest aneurysm level; view X right, Y up)",
         "origin": [cx, cy, z_max], "normal": [0.0, 0.0, 1.0], "e1": [1.0, 0.0, 0.0], "e2": [0.0, 1.0, 0.0]},
        {"key": "Y", "label": f"Y = {cy:.1f} mm (longitudinal through aneurysm centre; view X right, Z up)",
         "origin": [cx, cy, z_max], "normal": [0.0, 1.0, 0.0], "e1": [1.0, 0.0, 0.0], "e2": [0.0, 0.0, 1.0]},
    ]

    preview = {}
    if not args.skip_preview:
        spd_max = max(stats[f"speed_{lab}"][p]["p99"] for lab in vel_labels for p in phases)

        def speed_range(cut, phase):
            return 0.0, float(max(np.percentile(cut.point_data[f"speed_{lab}"], 99.5) for lab in vel_labels))

        def pres_range(cut, phase):
            vals = np.concatenate([np.asarray(cut.point_data[f"pressure_{lab}"]) for lab in pres_labels])
            return float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))

        groups = [
            {"key": "speed", "title": f"{short} |u| (m/s), same colour range per row", "cmap": "turbo",
             "columns": [f"speed_{lab}" for lab in vel_labels], "range": speed_range},
            {"key": "pressure", "title": f"{short} p - p_volmean(t) (Pa), same colour range per row",
             "cmap": "coolwarm", "columns": [f"pressure_{lab}" for lab in pres_labels], "range": pres_range},
        ]
        preview = slice_previews(tmp / case_name, time_values, phases, planes, groups, tmp / "preview")
        preview["speed_p99_all"] = spd_max
        for phase, rb in preview["readback"].items():
            if rb["cells"] != n_cells:
                raise RuntimeError(f"VTK read-back {phase}: {rb['cells']} cells != {n_cells}")
        log(f"previews: {preview['pngs']}")

    report = {
        "tool": "training_wss_min/tools/build_ensight_cfd_mesh_case.py",
        "case": info["canonical"],
        "cas_path": str(cas_path),
        "mesh": {"zone": "blood", "excluded": "blood1-5 extension zones", "cells": int(n_cells),
                 "nodes": int(n_nodes), "element_type": "nfaced (Fluent polyhedra and hex, faces outward)",
                 "volume_mm3": float(vol.sum()), "fluent_volume_mm3": float(vol_fluent.sum()),
                 "closed_consistently_oriented_cells": int(n_cells - n_open),
                 "frame": f"V5 aligned mm ({conv}), same as point clouds and aligned_geometry.stl"},
        "identity": {"cell_centroid_vs_pointcloud_max_mm": float(d_id.max()),
                     "wall_nodes_vs_wall_pointcloud_max_mm": err[conv], "wall_nodes": int(is_wall.sum())},
        "interpolation": {
            "interior_nodes": "inverse-distance (1/d) average of adjacent blood cells' point values",
            "wall_nodes_velocity": "0 (no-slip boundary condition, not a prediction)",
            "wall_nodes_pressure": "wall point-cloud value at the same Fluent node",
            "speed": "|node velocity vector| so colour-by-speed equals EnSight vector magnitude",
            "back_to_cell_r2_speed": fidelity,
        },
        "time": {"time_set": "cardiac_phase", "values_s": dict(zip(phases, time_values)),
                 "frames": info["frames"], "steps": info["steps"], "q_norm": info["q_norm"],
                 "period_s": VC.PERIOD_S},
        "variables": [{"kind": k, "name": n, "description": d} for k, n, d in var_specs],
        "value_ranges": stats,
        "recommended_planes": [{k: p[k] for k in ("label", "origin", "normal")} for p in planes],
        "ens_checker": checker,
        "preview": preview,
        "note": "display only; formal R2/MAE stay on the same-point CSV / point clouds",
        "elapsed_s": round(time.time() - t_all, 1),
    }
    (tmp / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(out)
    log(f"done in {report['elapsed_s']} s → {out}")


if __name__ == "__main__":
    main()
