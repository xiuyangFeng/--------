#!/usr/bin/env python
"""合成子例 vs 母例：逐面复算 Fluent mesh/check 的"左手面"（面法向与 c0→c1 方向不一致），定位是否形变引入（只读）。

冒烟 15730 对 GONG_HAI_ZENG-1~m34 / LI_YOU_ZHI-0~m48 / WENG_ZHAO_GUANG-0~m03 报 "Mesh check failed: left-handed faces"。
判据：内部面 s = c0_sign · A·(x_c1 − x_c0)，边界面 s = c0_sign · A·(x_f − x_c0)；s ≤ 0 记为左手面（质心取 Fluent 同口径体积加权质心）。

    /public/newhome/cy/.conda/envs/GNN/bin/python check_face_handedness.py   # → face_handedness_60.csv
"""
from __future__ import annotations

import csv
import re
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from wss_pinn.v4 import new_case_sources as ncs  # noqa: E402
from wss_pinn.v4.fluent_topology import read_fluent_mesh, resolve_anatomy_zone  # noqa: E402

SYN = ROOT / "docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926"
OUT = Path(__file__).resolve().parent


def left_handed(mesh, sign0=None):
    """返回 [(zone_id, face_index_in_section, face_centre_m, s)] 以及每个面区的面数；sign0 给定时沿用该取向约定。"""
    cent, vol = mesh.cell_centroids_and_volumes(strict=False)
    sign0 = mesh.c0_sign if sign0 is None else sign0
    bad, n_faces = [], {}
    for sec in mesh.face_sections:
        if not sec.attached:
            continue
        fc, A = mesh.face_geometry(sec)
        c0 = sec.c0; c1 = sec.c1
        other = np.where(c1[:, None] > 0, cent[np.where(c1 > 0, c1, 0)], fc)
        s = sign0 * np.einsum("ij,ij->i", A, other - cent[np.where(c0 > 0, c0, 0)])
        s = np.where(c0 > 0, s, np.nan)                  # c0 = 0 的面（只挂 c1）按 −c1 方向判
        m0 = c0 == 0
        if m0.any():
            s[m0] = -sign0 * np.einsum("ij,ij->i", A[m0], fc[m0] - cent[c1[m0]])
        n_faces[int(sec.zone_id)] = n_faces.get(int(sec.zone_id), 0) + len(c0)
        for k in np.flatnonzero(s <= 0):
            bad.append((int(sec.zone_id), int(k), fc[k], float(s[k])))
    return bad, n_faces, vol, sign0


def one(child: str) -> dict:
    parent = re.sub(r"~m\d+", "", child)
    morph = json.loads((ROOT / "data_new" / child / "synth_morph.json").read_text())
    mp = read_fluent_mesh(ncs.fluent_case(parent), keep_interior=True)
    mc = read_fluent_mesh(Path(morph["cas_child"]), keep_interior=True)
    bad_p, nf, vol_p, sign0 = left_handed(mp)
    bad_c, _, vol_c, _ = left_handed(mc, sign0)       # 子例沿用母例取向约定（节点移动不改 c0 约定）
    cz = mc.cell_zone_array(); zid, _ = resolve_anatomy_zone(mc, "blood", cz)
    # 左手面到最近壁面节点的距离（mm）+ 所在面区名 + 母例同一面是否已是左手面
    wall_nodes = np.unique(np.concatenate([s.nodes for s in mc.face_sections if s.bc_type == 3 and s.attached]))
    tree = cKDTree(mc.nodes_m[wall_nodes])
    key_p = {(z, k) for z, k, _, _ in bad_p}
    items = []
    for z, k, fc, s in bad_c:
        d, _ = tree.query(fc)
        items.append({"zone": z, "zone_name": mc.zone_name(z), "zone_type": mc.zone_type_name(z), "dist_to_wall_mm": round(float(d) * 1000, 3),
                      "also_in_parent": (z, k) in key_p})
    return {"child": child, "parent": parent, "left_handed_parent": len(bad_p), "left_handed_child": len(bad_c),
            "new_in_child": sum(not it["also_in_parent"] for it in items), "items": items,
            "min_cell_volume_child_m3": float(np.nanmin(vol_c[1:])), "nonpositive_cells_child": int(np.sum(vol_c[1:] <= 0)),
            "nonpositive_cells_parent": int(np.sum(vol_p[1:] <= 0)), "anatomy_zone": mc.zone_name(zid)}


def main():
    kids = [x.strip() for x in (SYN / "synth_children_all60_clean.txt").read_text().splitlines() if x.strip()]
    with Pool(10) as pool:
        res = pool.map(one, kids, chunksize=1)
    with open(OUT / "face_handedness_60.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = ["child", "parent", "left_handed_parent", "left_handed_child", "new_in_child", "nonpositive_cells_parent", "nonpositive_cells_child", "anatomy_zone"]
        w.writerow(keys + ["items"])
        for r in res:
            w.writerow([r[k] for k in keys] + [json.dumps(r["items"], ensure_ascii=False)])
    for r in res:
        if r["left_handed_child"] or r["left_handed_parent"]:
            print(r["child"], "parent", r["left_handed_parent"], "child", r["left_handed_child"], "new", r["new_in_child"], r["items"][:12])
    print("children with any left-handed face:", sum(r["left_handed_child"] > 0 for r in res), "/", len(res),
          "| with new (morph-induced) ones:", sum(r["new_in_child"] > 0 for r in res))


if __name__ == "__main__":
    main()
