#!/usr/bin/env python
"""§36 方向三合成 CFD 子例：60 例核验总表 + 随机抽 15 例的形状 / 沿程半径 / WSS 图（只读，不改任何数据）。

    /public/newhome/cy/.conda/envs/GNN/bin/python make_synth_viz.py

输出（本目录）：verification_60.csv、summary.json、fig1_shape_gallery.png、fig2_radius_profiles.png、
fig3_wss_peak_{1,2,3}.png、fig4_gate_matrix.png、fig5_numeric_checks.png。
抽样：按队列分层、比例分配（AG 5 / AAA 4 / ILO 6），np.random.default_rng(20260928)，只从 59 个入库子例里抽。
"""
from __future__ import annotations

import csv
import json
import re
from multiprocessing import Pool
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
SYN = ROOT / "docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926"
V5 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_2_20260922/cases"
V51 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"
AUD = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits"
SPL = ROOT / "data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1/syn_v52"
LOG_INGEST = SYN.parent / "ingest_wave2_15783.out"
OUT = Path(__file__).resolve().parent
SEED = 20260928
PEAK_STEP = 1162
N_PER_COHORT = {"AG": 5, "AAA": 4, "ILO": 6}
# synth_morph_case.py 的常量（设计 f(s) 复算用）
TAPER_END_MM, TAPER_JUNCTION_R, F_MIN, F_MAX = 12.0, 1.5, 0.55, 1.45

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
# .ttc 注册成集合里第一个字体的族名（JP）；汉字字形全覆盖
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b0af", "#e34948", "#8e1f1f"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#061f40"])
SEM_ZH = {"trunk": "主干", "left_cia": "左髂总", "right_cia": "右髂总", "left_external": "左髂外", "left_internal": "左髂内",
          "right_external": "右髂外", "right_internal": "右髂内"}


def parent_of(child: str) -> str:
    return re.sub(r"~m\d+", "", child)


def h5path(cid: str) -> Path:
    """与 synth_morph_case.load_atlas 同序：先 v5.2 根（新回收单元 + 子例），再 v5.1 根（170 例库内母例）。"""
    for root in (V5, V51):
        p = root / cid.replace("/", "__") / "case.h5"
        if p.is_file():
            return p
    raise FileNotFoundError(cid)


def short(cid: str) -> str:
    p = cid.split("/")
    return f"{p[1]}/{p[2]}" if p[0] == "ILO" else p[2]


def panel_title(cid: str) -> str:
    """两行：队列（ILO 带术前/术后） / 病例名。"""
    p = cid.split("/")
    if p[0] == "ILO":
        return f"ILO {'术后' if p[2] == 'after' else '术前'}\n{p[1]}"
    return f"{p[0]} {p[1]}\n{p[2]}"


def solver_tier(cid: str) -> tuple[str, float | None]:
    p = AUD / "solver_convergence/cases" / (cid.replace("/", "__") + ".json")
    if not p.is_file():
        return "?", None
    d = json.loads(p.read_text())
    sq = d.get("solver_quality", {})
    return sq.get("tier", d.get("tier", "?")), sq.get("continuity_last_max", d.get("continuity_last_max"))


def outlet_labels(cid: str) -> dict[str, str]:
    d = json.loads((AUD / "topology/cases" / (cid.replace("/", "__") + ".json")).read_text())
    return {i["extension_zone"]: i["semantic_label"] for i in d["interfaces"]}


def read_list(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def load_atlas(f):
    g = f["geometry"]
    cols = json.loads(g.attrs["atlas_columns"]); T = g["atlas_table"][...]
    sem_of_seg = {int(k): int(v) for k, v in json.loads(g.attrs["semantic_of_segment"]).items()}
    sem_lab = {int(k): v for k, v in json.loads(g.attrs["semantic_labels"]).items()}
    return T, {c: i for i, c in enumerate(cols)}, sem_of_seg, sem_lab


def design_f(T, ci, bumps):
    """synth_morph_case.modulation 去掉随机数的复算：凸包 + 开口/分叉渐变 + 截断。"""
    seg = T[:, ci["segment_id"]].astype(int); s = T[:, ci["s_local_mm"]]; r = T[:, ci["radius_mm"]]
    d_end = T[:, ci["dist_to_endpoint_mm"]]; d_j = T[:, ci["dist_to_junction_mm"]]
    f = np.ones(len(T))
    for b in bumps:
        m = seg == b["segment_id"]
        f[m] += b["amplitude"] * np.exp(-0.5 * ((s[m] - b["s_local_mm"]) / b["sigma_mm"]) ** 2)
    w = smooth((d_end - TAPER_END_MM) / TAPER_END_MM) * smooth((d_j - TAPER_JUNCTION_R * r) / np.maximum(TAPER_JUNCTION_R * r, 1e-6))
    return np.clip(1.0 + (f - 1.0) * w, F_MIN, F_MAX)


def child_radius_on_parent_stations(Tp, cip, Tc, cic, rows_p, seg_rows_p=None):
    """父站点 → 子 atlas 半径。子段按几何多数票确定（父段全部站点的最近子站点所在段的众数），不依赖语义标签；
    子中心线由入库链从形变后的面重新提取。"""
    xyz = [cip["x_raw_mm"], cip["y_raw_mm"], cip["z_raw_mm"]]; xyzc = [cic["x_raw_mm"], cic["y_raw_mm"], cic["z_raw_mm"]]
    seg_rows = np.flatnonzero(Tp[:, cip["segment_id"]].astype(int) == int(Tp[rows_p[0], cip["segment_id"]])) if seg_rows_p is None else seg_rows_p
    segc = Tc[:, cic["segment_id"]].astype(int)
    _, j0 = cKDTree(Tc[:, xyzc]).query(Tp[seg_rows][:, xyz])
    best = np.bincount(segc[j0]).argmax(); mc = segc == best
    _, j = cKDTree(Tc[mc][:, xyzc]).query(Tp[rows_p][:, xyz])
    return Tc[mc][j, cic["radius_mm"]]


# ---------------------------------------------------------------------------------------------- per-case metrics
def case_metrics(child: str) -> dict:
    parent = parent_of(child)
    morph = json.loads((ROOT / "data_new" / child / "synth_morph.json").read_text())
    r = {"child": child, "parent": parent, "cohort": child.split("/")[0], "seed": morph["seed"],
         "n_bumps": len(morph["bumps"]), "negative_volumes": morph["negative_volumes"],
         "cell_volume_ratio_min": morph["cell_volume_ratio_min"], "cell_volume_ratio_max": morph["cell_volume_ratio_max"],
         "anatomy_volume_ratio": morph["anatomy_volume_ratio"], "f_min": morph["f_min"], "f_max": morph["f_max"],
         "max_displacement_mm": morph["max_displacement_mm"], "nodes_moved": morph["nodes_moved"], "morph_valid": morph["valid"]}
    with h5py.File(h5path(child), "r") as c, h5py.File(h5path(parent), "r") as p:
        idc = c["wall_static/node_id_cas"][...]; idp = p["wall_static/node_id_cas"][...]
        r["wall_nodes_identical_order"] = bool(np.array_equal(idc, idp))
        kc = int(np.flatnonzero(c["wall_temporal/step"][...] == PEAK_STEP)[0]); kp = int(np.flatnonzero(p["wall_temporal/step"][...] == PEAK_STEP)[0])
        wc_all = c["wall_temporal/wss_scalar_pa"][...]; wp_all = p["wall_temporal/wss_scalar_pa"][...]
        pc = c["wall_temporal/pressure_pa"][kc]; pp = p["wall_temporal/pressure_pa"][kp]
        xc = c["wall_static/xyz_mm"][...]; xp = p["wall_static/xyz_mm"][...]
        nrm = p["wall_static/normal_out_mesh"][...].astype(float)
        seg = p["wall_static/segment_id"][...]; sl = p["wall_static/s_local_mm"][...]
        Tp, cip, sem_p, sem_lab = load_atlas(p); Tc, cic, sem_c, _ = load_atlas(c)
        r["child_labels_finite"] = bool(np.isfinite(wc_all).all() and np.isfinite(c["wall_temporal/pressure_pa"][...]).all())
    wc, wp = wc_all[kc], wp_all[kp]
    tc, tp = wc_all.mean(0), wp_all.mean(0)
    disp = xc - xp; dn = np.einsum("ij,ij->i", disp, nrm)
    r.update({"wall_disp_normal_min_mm": float(dn.min()), "wall_disp_normal_max_mm": float(dn.max()),
              "wall_nodes_moved_frac": float(np.mean(np.linalg.norm(disp, axis=1) > 1e-6)),
              "wss_p50_child": float(np.percentile(wc, 50)), "wss_p99_child": float(np.percentile(wc, 99)), "wss_max_child": float(wc.max()),
              "wss_p50_parent": float(np.percentile(wp, 50)), "wss_p99_parent": float(np.percentile(wp, 99)), "wss_max_parent": float(wp.max()),
              "wss_min_child": float(wc_all.min()),
              "pwall_p99_child": float(np.percentile(pc, 99)), "pwall_p99_parent": float(np.percentile(pp, 99))})
    # 每个凸包：设计 f（中心处，含渐变与叠加）、子 atlas 实测半径比、凸包区 TAWSS 子/母
    fd = design_f(Tp, cip, morph["bumps"]); segT = Tp[:, cip["segment_id"]].astype(int); sT = Tp[:, cip["s_local_mm"]]
    bumps = []
    for b in morph["bumps"]:
        rows = np.flatnonzero(segT == b["segment_id"])
        j = rows[np.argmin(np.abs(sT[rows] - b["s_local_mm"]))]
        rc = child_radius_on_parent_stations(Tp, cip, Tc, cic, np.array([j]))[0]
        reg = (seg == b["segment_id"]) & (np.abs(sl - b["s_local_mm"]) < b["sigma_mm"])
        bumps.append({"segment": SEM_ZH.get(sem_lab[sem_p[b["segment_id"]]], str(b["segment_id"])), "amplitude": b["amplitude"],
                      "sigma_mm": b["sigma_mm"], "radius_parent_mm": float(Tp[j, cip["radius_mm"]]),
                      "f_design_center": float(fd[j]), "radius_ratio_child_atlas": float(rc / Tp[j, cip["radius_mm"]]),
                      "tawss_ratio_region": float(np.median(tc[reg]) / np.median(tp[reg])) if reg.sum() > 20 else float("nan"),
                      "region_nodes": int(reg.sum())})
    r["bumps"] = bumps
    r["parent_solver_tier"], r["parent_continuity_last_max"] = solver_tier(parent)
    lp, lc = outlet_labels(parent), outlet_labels(child)
    r["outlet_labels_match_parent"] = lp == lc
    r["outlet_label_diff"] = ";".join(f"{z}:{lp[z]}->{lc.get(z)}" for z in sorted(lp) if lp[z] != lc.get(z))
    return r


# ---------------------------------------------------------------------------------------------- rendering helpers
def view_basis(xyz, inlet, left_outlet):
    c = xyz.mean(0); X = xyz[:: max(1, len(xyz) // 30000)] - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0] if np.dot(inlet - c, vt[0]) > 0 else -vt[0]          # 纵轴 = 最长主轴，入口朝上
    u = vt[1] - np.dot(vt[1], v) * v; u /= np.linalg.norm(u)
    if left_outlet is not None and np.dot(left_outlet - c, u) < 0:  # 放射学习惯：患者左侧在画面右侧
        u = -u
    return c, u, v, np.cross(u, v)


def project(xyz, basis):
    c, u, v, w = basis
    X = xyz - c
    return np.stack([X @ u, X @ v, X @ w], 1)


def draw_surface(ax, P, tri, rgb, shade=(0.55, 0.45)):
    T = P[tri]; n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    order = np.argsort(T[:, :, 2].mean(1))
    lam = shade[0] + shade[1] * np.abs(n[:, 2])
    fc = np.clip(rgb * lam[:, None], 0, 1)[order]
    pc = PolyCollection(T[order][:, :, :2], facecolors=fc, edgecolors=fc, linewidths=0.15, rasterized=True)
    ax.add_collection(pc)


def outline_mask(P, tri, lim, res=0.25):
    (x0, x1), (y0, y1) = lim
    W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
    Q = np.stack([(P[:, 0] - x0) / res, (y1 - P[:, 1]) / res], 1)
    for t in tri:
        dr.polygon([tuple(Q[t[0]]), tuple(Q[t[1]]), tuple(Q[t[2]])], fill=255)
    m = np.asarray(img)[::-1] > 0
    return np.linspace(x0, x0 + (W - 1) * res, W), np.linspace(y1 - (H - 1) * res, y1, H), m


def load_geom(cid):
    with h5py.File(h5path(cid), "r") as f:
        xyz = f["wall_static/xyz_mm"][...]; tri = f["topology/wall_triangles"][...]
        k = int(np.flatnonzero(f["wall_temporal/step"][...] == PEAK_STEP)[0])
        wss = f["wall_temporal/wss_scalar_pa"][k].astype(float)
        nrm = f["wall_static/normal_out_mesh"][...].astype(float)
        inlet = np.asarray(json.loads(f["interfaces/inlet"].attrs["center_mm"]), float)
        le = np.asarray(json.loads(f["interfaces/out-le"].attrs["center_mm"]), float) if "out-le" in f["interfaces"] else None
    return {"xyz": xyz, "tri": tri, "wss": wss, "nrm": nrm, "inlet": inlet, "le": le}


def face_val(v, tri):
    return v[tri].mean(1)


def bump_text(bumps):
    out = []
    for b in bumps:
        kind = "狭窄" if b["amplitude"] < 0 else "扩张"
        out.append(f"{b['segment']}{kind}{abs(b['amplitude']) * 100:.0f}%")
    return " · ".join(out)


# ---------------------------------------------------------------------------------------------- main
def main():
    all60 = read_list(SYN / "synth_children_all60_clean.txt"); ok59 = set(read_list(SYN / "synth_children_ok59.txt"))
    rcr = {row["case"]: row for row in csv.DictReader(open(SYN / "rcr_runtime_audit_syn_all60.csv"))}
    scan = {row["case"]: row for row in csv.DictReader(open(SYN / "wall_scan_1162_syn_all60.csv"))}
    # Fluent 冒烟：同一子例多次冒烟以日志文件时间最新者为准
    smoke = {}
    for lp in sorted(SYN.glob("smoke_synth_*.out"), key=lambda p: p.stat().st_mtime):
        for m in re.finditer(r"===== SMOKE (\S+) =====(.*?)===== fluent exit code (\d+) =====", lp.read_text(errors="replace"), re.S):
            cid = m.group(1).replace(str(ROOT / "data_new") + "/", "")
            body = m.group(2)
            smoke[cid] = {"exit": int(m.group(3)), "mesh_check_done": "Checking mesh" in body,
                          "mesh_check_failed": "Mesh check failed" in body,
                          "left_handed": sum(int(x) for x in re.findall(r"right-handed,\s+(\d+) left-handed", body)),
                          "poor_quality_info": "invalid or of poor quality" in body,
                          "errors": len(re.findall(r"(?im)^\s*error|negative (cell )?volume", body)), "log": lp.name}
    # check_face_handedness.py 的逐面复算（母例 vs 子例，同一取向约定）
    hand = {row["child"]: row for row in csv.DictReader(open(OUT / "face_handedness_60.csv"))} if (OUT / "face_handedness_60.csv").is_file() else {}
    # 母例运行态 RCR（只取 verdict ok 的行，后读的文件覆盖先读的）
    rcr_parent = {}
    for name in ("rcr_runtime_audit.csv", "rcr_runtime_audit_alayer.csv", "rcr_runtime_audit_formal170.csv", "rcr_runtime_audit_reruns.csv"):
        for row in csv.DictReader(open(SYN.parent / name)):
            if row["verdict"] == "ok" and row["L"]:
                rcr_parent[row["case"]] = dict(row, _source=name)
    cl_pass = set(read_list(SYN.parent / "ingest_syn60_cl_pass.txt"))
    m = re.search(r"review-required: \[(.*?)\]", LOG_INGEST.read_text())
    cl_review = set(re.findall(r"'([^']+)'", m.group(1))) if m else set()
    topo = json.loads((AUD / "topology/summary.json").read_text())["cases_report"]
    wall = json.loads((AUD / "wall_wss/summary.json").read_text())["cases_report"]
    # 划分合规：子例从不进测试折；子例只出现在母例不在测试折的训练集里
    split_ok, split_notes = {c: True for c in all60}, []
    for sp in sorted(SPL.glob("*_syn.json")):
        d = json.loads(sp.read_text()); test = set(d["test_cases"]) | set(d["val_cases"]); train = set(d["train_cases"])
        for c in all60:
            if c in test or (c in train and parent_of(c) in test):
                split_ok[c] = False; split_notes.append(f"{sp.name}: {c}")
        if set(d["synthetic"]["children"]) - ok59:
            split_notes.append(f"{sp.name}: 含未入库子例 {sorted(set(d['synthetic']['children']) - ok59)}")

    with Pool(12) as pool:
        mets = pool.map(case_metrics, all60)
    rows = []
    for r in mets:
        c = r["child"]; rep = json.loads((h5path(c).parent / "report.json").read_text())
        rr, ss, sm = rcr.get(c, {}), scan.get(c, {}), smoke.get(c, {})
        hp, rp = hand.get(c, {}), rcr_parent.get(parent_of(c), {})
        r.update({
            "in_ok59": c in ok59,
            "smoke_exit0": sm.get("exit") == 0 and sm.get("mesh_check_done", False) and sm.get("errors", 1) == 0,
            "smoke_mesh_check_failed": sm.get("mesh_check_failed", False), "smoke_left_handed_faces": sm.get("left_handed", 0),
            "smoke_poor_quality_info": sm.get("poor_quality_info", False), "smoke_log": sm.get("log", ""),
            "left_handed_faces_recomputed_child": int(hp["left_handed_child"]) if hp else -1,
            "left_handed_faces_recomputed_parent": int(hp["left_handed_parent"]) if hp else -1,
            "left_handed_faces_new": int(hp["new_in_child"]) if hp else -1,
            "left_handed_dist_to_wall_mm_max": max([it["dist_to_wall_mm"] for it in json.loads(hp["items"])], default=0.0) if hp else float("nan"),
            "rcr_L_parent": float(rp["L"]) if rp else float("nan"), "rcr_parent_source": rp.get("_source", ""),
            "rcr_p_out_kpa_mean": float(np.mean([float(rr[k]) for k in ("p_le", "p_li", "p_re", "p_ri")])) / 1000 if rr else float("nan"),
            "rcr_p_out_kpa_mean_parent": float(np.mean([float(rp[k]) for k in ("p_le", "p_li", "p_re", "p_ri")])) / 1000 if rp else float("nan"),
            "rcr_verdict": rr.get("verdict", ""), "rcr_L": float(rr.get("L", "nan")), "rcr_mass_balance": float(rr.get("mass_balance", "nan")),
            "rcr_p_out_kpa_min": min(float(rr[k]) for k in ("p_le", "p_li", "p_re", "p_ri")) / 1000 if rr else float("nan"),
            "rcr_p_out_kpa_max": max(float(rr[k]) for k in ("p_le", "p_li", "p_re", "p_ri")) / 1000 if rr else float("nan"),
            "rcr_steps": int(rr.get("steps", 0) or 0), "scan_wall_rows": int(ss.get("n_wall_rows", 0) or 0),
            "frames_81": bool(rep["gates"]["checks"].get("volume_frames_81") and rep["gates"]["checks"].get("wall_frames_81")),
            "centerline": "review" if c in cl_review else ("pass" if c in cl_pass else "fail"),
            "topology_audit": bool(topo.get(c, {}).get("gate_pass", False)), "wall_audit": bool(wall.get(c, {}).get("gate_pass", False)),
            "solver_tier": rep["sources"]["solver"]["tier"], "continuity_last_max": rep["sources"]["solver"]["continuity_last_max"],
            "v5_gates_pass": bool(rep["gates"]["pass"]), "v5_gates_n": len(rep["gates"]["checks"]),
            "v5_failed": ";".join(k for k, v in rep["gates"]["checks"].items() if not v), "v5_waivers": ";".join(rep["gates"]["waivers"]),
            "split_ok": split_ok[c]})
        rows.append(r)
    order = {"AG": 0, "AAA": 1, "ILO": 2}
    rows.sort(key=lambda r: (order[r["cohort"]], r["child"]))

    # 抽样（分层、比例分配，只从 59 入库子例）
    rng = np.random.default_rng(SEED); sample = []
    for coh, n in N_PER_COHORT.items():
        pool_ = sorted(r["child"] for r in rows if r["cohort"] == coh and r["in_ok59"])
        sample += list(rng.choice(pool_, size=n, replace=False))
    sample = sorted(sample, key=lambda c: (order[c.split("/")[0]], c))
    for r in rows:
        r["sampled"] = r["child"] in sample

    # ------------------------------------------------------------------ 表
    flat_keys = [k for k in rows[0] if k != "bumps"]
    with open(OUT / "verification_60.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(flat_keys + ["bumps"])
        for r in rows:
            w.writerow([r[k] for k in flat_keys] + [json.dumps(r["bumps"], ensure_ascii=False)])
    byid = {r["child"]: r for r in rows}

    # ------------------------------------------------------------------ 图 1：形状总览（子例按法向位移着色 + 母例外轮廓）
    geoms = {}
    for c in sample:
        geoms[c] = (load_geom(parent_of(c)), load_geom(c))
    fig, axes = plt.subplots(3, 5, figsize=(17, 17.5))
    DMAX = 5.0; norm_d = Normalize(-DMAX, DMAX)
    for ax, c in zip(axes.flat, sample):
        gp, gc = geoms[c]; basis = view_basis(gp["xyz"], gp["inlet"], gp["le"])
        Pp, Pc = project(gp["xyz"], basis), project(gc["xyz"], basis)
        dn = np.einsum("ij,ij->i", gc["xyz"] - gp["xyz"], gp["nrm"])
        rgb = DIV(norm_d(face_val(dn, gc["tri"])))[:, :3]
        draw_surface(ax, Pc, gc["tri"], rgb)
        pad = 4.0; lim = ((min(Pp[:, 0].min(), Pc[:, 0].min()) - pad, max(Pp[:, 0].max(), Pc[:, 0].max()) + pad),
                          (min(Pp[:, 1].min(), Pc[:, 1].min()) - pad, max(Pp[:, 1].max(), Pc[:, 1].max()) + pad))
        X, Y, M = outline_mask(Pp, gp["tri"], lim)
        ax.contour(X, Y, M.astype(float), levels=[0.5], colors=INK, linewidths=0.7, linestyles="--")
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.axis("off")
        r = byid[c]
        ax.set_title(panel_title(c), fontsize=10, fontweight="bold", loc="left", color=INK)
        ax.text(0.0, -0.01, f"{bump_text(r['bumps'])}\n最大位移 {r['max_displacement_mm']:.1f} mm · 解剖区体积比 {r['anatomy_volume_ratio']:.3f}",
                transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
    cax = fig.add_axes([0.30, 0.03, 0.40, 0.010])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_d, cmap=DIV), cax=cax, orientation="horizontal", extend="both")
    cb.set_label("子例壁面相对母例的法向位移 (mm)：蓝 = 向内（狭窄），红 = 向外（扩张）；黑虚线 = 母例外轮廓", color=INK2)
    fig.suptitle("合成子例形状：随机抽取 15 例（AG 5 / AAA 4 / ILO 6），壁面按形变着色、母例轮廓叠加", fontsize=14, x=0.02, ha="left", y=0.995, color=INK)
    fig.text(0.02, 0.975, "视角：母例壁面点主成分平面（最长轴竖直、入口朝上、患者左侧在画面右侧）；只画解剖区壁面（延伸段节点不动）。每例 2 个高斯凸包，幅值 15–40%。",
             fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=0.07, wspace=0.08, hspace=0.24)
    fig.savefig(OUT / "fig1_shape_gallery.png", dpi=140); plt.close(fig)

    # ------------------------------------------------------------------ 图 2：沿程半径（母例 / 设计 / 子例重提中心线）
    fig, axes = plt.subplots(3, 5, figsize=(17, 11.5))
    for ax, c in zip(axes.flat, sample):
        morph = json.loads((ROOT / "data_new" / c / "synth_morph.json").read_text())
        with h5py.File(h5path(parent_of(c)), "r") as p, h5py.File(h5path(c), "r") as ch:
            Tp, cip, sem_p, sem_lab = load_atlas(p); Tc, cic, sem_c, _ = load_atlas(ch)
        fd = design_f(Tp, cip, morph["bumps"]); segT = Tp[:, cip["segment_id"]].astype(int)
        for i, sid in enumerate(dict.fromkeys(b["segment_id"] for b in morph["bumps"])):
            rows_p = np.flatnonzero(segT == sid); rows_p = rows_p[np.argsort(Tp[rows_p, cip["s_local_mm"]])]
            s = Tp[rows_p, cip["s_local_mm"]]; rp = Tp[rows_p, cip["radius_mm"]]
            rc = child_radius_on_parent_stations(Tp, cip, Tc, cic, rows_p)
            col = (BLUE, ORANGE)[i % 2]; name = SEM_ZH.get(sem_lab[sem_p[sid]], str(sid))
            ax.plot(s, rp, color=MUTED, lw=1.4, ls="--")
            ax.plot(s, rp * fd[rows_p], color=col, lw=1.0, ls=":")
            ax.plot(s, rc, color=col, lw=2.0, label=name)
        for b in morph["bumps"]:
            ax.axvline(b["s_local_mm"], color=GRID, lw=0.8, zorder=0)
        ax.set_title(panel_title(c), fontsize=9.5, fontweight="bold", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_xlabel("段内弧长 s (mm)"); ax.set_ylabel("半径 (mm)")
        ax.legend(fontsize=7.5, frameon=False, loc="best", title="形变段", title_fontsize=7.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=1.4, ls="--", label="母例中心线半径"),
                        Line2D([], [], color=INK2, lw=1.0, ls=":", label="设计值 R_母·f(s)"),
                        Line2D([], [], color=INK2, lw=2.0, label="子例：入库链从形变后的面重新提取的中心线半径")],
               loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("沿程半径：形变段上的母例、设计值与子例实测", fontsize=13, x=0.01, ha="left", y=0.985, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.savefig(OUT / "fig2_radius_profiles.png", dpi=140); plt.close(fig)

    # ------------------------------------------------------------------ 图 3：峰值帧 WSS（母 / 子 / 差），每张 5 例
    for part in range(3):
        cs = sample[part * 5:(part + 1) * 5]
        fig, axes = plt.subplots(3, 5, figsize=(17, 18))
        for j, c in enumerate(cs):
            gp, gc = geoms[c]; basis = view_basis(gp["xyz"], gp["inlet"], gp["le"])
            Pp, Pc = project(gp["xyz"], basis), project(gc["xyz"], basis)
            vmax = float(np.percentile(np.r_[gp["wss"], gc["wss"]], 99)); nw = Normalize(0, vmax)
            dw = gc["wss"] - gp["wss"]; dmax = max(2.0, float(np.percentile(np.abs(dw), 99))); nd = Normalize(-dmax, dmax)
            for i, (P, g, vals, cmap, nrm_) in enumerate([(Pp, gp, gp["wss"], SEQ, nw), (Pc, gc, gc["wss"], SEQ, nw), (Pc, gc, dw, DIV, nd)]):
                ax = axes[i, j]
                draw_surface(ax, P, g["tri"], cmap(nrm_(face_val(vals, g["tri"])))[:, :3], shade=(0.8, 0.2))
                allP = np.r_[Pp, Pc]; ax.set_xlim(allP[:, 0].min() - 3, allP[:, 0].max() + 3); ax.set_ylim(allP[:, 1].min() - 3, allP[:, 1].max() + 3)
                ax.set_aspect("equal"); ax.axis("off")
                if i == 0:
                    ax.set_title(f"{panel_title(c)}\n母例 · p99 {np.percentile(vals, 99):.1f} Pa · max {vals.max():.1f} Pa", fontsize=9, loc="left")
                elif i == 1:
                    ax.set_title(f"子例 · p99 {np.percentile(vals, 99):.1f} Pa · max {vals.max():.1f} Pa", fontsize=9, loc="left")
                else:
                    ax.set_title(f"差值 子 − 母（色标 ±{dmax:.1f} Pa）", fontsize=9, loc="left")
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm_, cmap=cmap), ax=ax, orientation="horizontal", fraction=0.03, pad=0.01, shrink=0.8)
                cb.ax.tick_params(labelsize=7); cb.set_label("WSS (Pa)" if i < 2 else "ΔWSS (Pa)", fontsize=7, color=INK2, labelpad=1)
        for j in range(len(cs), 5):
            for i in range(3):
                axes[i, j].axis("off")
        fig.suptitle(f"峰值帧（step {PEAK_STEP}）壁面 WSS：第 1 行母例、第 2 行子例（同一色标），第 3 行差值（蓝 = 子例更低，红 = 子例更高）· 第 {part + 1}/3 张",
                     fontsize=13, x=0.01, ha="left", y=0.995, color=INK)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.05, hspace=0.22)
        fig.savefig(OUT / f"fig3_wss_peak_{part + 1}.png", dpi=120); plt.close(fig)

    # ------------------------------------------------------------------ 图 4：60 例核验状态矩阵
    def st_mesh(r):
        if r["smoke_mesh_check_failed"]:
            return "crit", f"左手面 {r['smoke_left_handed_faces']}"
        if r["smoke_poor_quality_info"]:
            return "warn", "质量提示"
        return ("good", "✓") if r["smoke_exit0"] else ("crit", "✗")

    def st_tier(r):
        t, tp = r["solver_tier"], r["parent_solver_tier"]
        txt = t if t == tp else f"{t}（母{tp}）"
        return ({"A": "good", "B": "good", "C": "warn", "D": "serious"}.get(t, "warn"), txt)

    checks = [("形变体积\n检查", lambda r: ("good", "✓") if r["morph_valid"] and r["negative_volumes"] == 0 else ("crit", "✗")),
              ("Fluent\n网格检查", st_mesh),
              ("81 帧\n导出", lambda r: ("good", "✓") if r["frames_81"] else ("crit", "✗")),
              ("RCR\n运行态", lambda r: ("good", "✓") if r["rcr_verdict"] == "ok" else ("crit", r["rcr_verdict"] or "—")),
              ("中心线", lambda r: {"pass": ("good", "✓"), "review": ("warn", "复核"), "fail": ("crit", "✗")}[r["centerline"]]),
              ("拓扑\n审计", lambda r: ("good", "✓") if r["topology_audit"] else ("crit", "✗")),
              ("出口标签\n同母例", lambda r: ("good", "✓") if r["outlet_labels_match_parent"] else ("warn", "左内/外互换")),
              ("壁面\n审计", lambda r: ("good", "✓") if r["wall_audit"] else ("crit", "✗")),
              ("求解层级", st_tier),
              ("V5 门\n(19 项)", lambda r: ("good", "✓") if r["v5_gates_pass"] else ("crit", "✗ pca 法向")),
              ("划分\n合规", lambda r: ("good", "✓") if r["split_ok"] else ("crit", "✗")),
              ("入库", lambda r: ("good", "✓") if r["in_ok59"] else ("crit", "剔除"))]
    COL = {"good": (GOOD, 0.20), "warn": (WARN, 0.55), "serious": (SERIOUS, 0.65), "crit": (CRIT, 0.55)}
    n = len(rows); W = [1.0, 1.25, 0.9, 0.9, 0.9, 0.8, 1.35, 0.8, 1.2, 1.15, 0.8, 0.8]
    x0s = np.r_[0, np.cumsum(W)[:-1]]
    fig, ax = plt.subplots(figsize=(13.5, 0.22 * n + 2.0))
    for i, r in enumerate(rows):
        y = n - 1 - i
        for j, (name, fn) in enumerate(checks):
            st, txt = fn(r); col, al = COL[st]
            ax.add_patch(plt.Rectangle((x0s[j] + 0.03, y + 0.07), W[j] - 0.06, 0.86, color=col, alpha=al, lw=0))
            ax.text(x0s[j] + W[j] / 2, y + 0.5, txt, ha="center", va="center", fontsize=7.3, color=INK)
        lab = f"{'● ' if r['sampled'] else ''}{r['cohort']} · {short(r['child'])}"
        ax.text(-0.1, y + 0.5, lab, ha="right", va="center", fontsize=7.5, color=INK if r["sampled"] else INK2,
                fontweight="bold" if r["sampled"] else "normal")
    for j, (name, _) in enumerate(checks):
        ax.text(x0s[j] + W[j] / 2, n + 0.2, name, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(-6.4, x0s[-1] + W[-1]); ax.set_ylim(-0.2, n + 2.0); ax.axis("off")
    ax.set_title("60 个合成子例逐项核验（● = 本次抽样的 15 例；绿 = 通过，黄 = 通过但有提示，橙 = 求解层级 D，红 = 该项不过）", loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.004,
             "Fluent 网格检查 = 冒烟作业 read-case + mesh/check 的原文；“左手面 N” = Fluent 报 Mesh check failed（形变引入，母例为 0，位于贴壁 0.1–0.5 mm 边界层内部面），当时未作为门、仍进入求解。\n"
             "求解层级 = 81 个导出步最后一次连续性残差最大值 A ≤ 1e-3 / B < 2e-3 / C < 5e-3 / D ≥ 5e-3（母库同一口径，不是门）；括号内为母例层级。出口标签 = 拓扑审计给 5 个延伸段贴的 inlet/out-** 语义。",
             fontsize=7.3, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "fig4_gate_matrix.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ 图 5：数值核验分布（60 例，抽样 15 例高亮）
    xpos = {"AG": 0, "AAA": 1, "ILO": 2}
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, n)
    samp = np.array([r["sampled"] for r in rows]); xs = np.array([xpos[r["cohort"]] for r in rows]) + jit

    def dots(ax, x, y, s):
        ax.scatter(x[~s], y[~s], s=22, facecolors="none", edgecolors=MUTED, lw=0.9, zorder=3)
        ax.scatter(x[s], y[s], s=30, color=BLUE, edgecolors="#fcfcfb", lw=0.8, zorder=4)

    def strip(ax, title, v, lines=(), note=None):
        dots(ax, xs, v, samp)
        for y, lab in lines:
            ax.axhline(y, color=INK2, lw=0.8, ls="--")
            if lab:
                ax.text(-0.46, y, lab, fontsize=7.3, color=INK2, va="bottom", ha="left", zorder=5,
                        bbox=dict(fc="#fcfcfb", ec="none", pad=0.6, alpha=0.85))
        ax.set_xticks([0, 1, 2], ["AG", "AAA", "ILO"]); ax.set_xlim(-0.5, 2.5)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_title(title, loc="left", fontsize=10)
        if note:
            ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=7.3, color=INK2, va="top")

    def name_pts(ax, x, y, names):
        for xi, yi, nm in zip(x, y, names):
            ax.annotate(nm, (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=6.8, color=INK2)

    def diag(ax, title, a, b, s, xl, yl, lim=None, note=None):
        lim = lim or (min(np.nanmin(a), np.nanmin(b)), max(np.nanmax(a), np.nanmax(b)))
        pad = 0.05 * (lim[1] - lim[0]); lim = (lim[0] - pad, lim[1] + pad)
        ax.plot(lim, lim, color=GRID, lw=1, zorder=1); dots(ax, a, b, s)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID, lw=0.5)
        ax.set_title(title, loc="left", fontsize=10)
        if note:
            ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=7.3, color=INK2, va="top")

    fig, axes = plt.subplots(2, 5, figsize=(19, 8.6)); A = axes.flat
    strip(A[0], "单元体积比 最小值（子/母）", np.array([r["cell_volume_ratio_min"] for r in rows]), [(0.25, "筛选门 0.25"), (0.5, "09-27 教训建议 0.5")])
    strip(A[1], "解剖区总体积 子/母", np.array([r["anatomy_volume_ratio"] for r in rows]), [(1.0, "")])
    strip(A[2], "最大节点位移 (mm)", np.array([r["max_displacement_mm"] for r in rows]))
    La, Lb = np.array([r["rcr_L_parent"] for r in rows]), np.array([r["rcr_L"] for r in rows]); m = np.isfinite(La)
    diag(A[3], "左侧流量份额 L：子 vs 母（运行态）", La[m], Lb[m], samp[m], "母例 L", "子例 L", (0.46, 0.56),
         f"{int(m.sum())} 对（4 例母例无运行态记录）\n中位 |ΔL| {np.median(np.abs(Lb[m] - La[m])):.4f}\n* = 母例记录来自 A 层旧审计（母例其后已重算）")
    big = m & (np.abs(Lb - La) > 0.01)
    name_pts(A[3], La[big], Lb[big], [short(r["child"]).split("/")[0] + ("*" if r["rcr_parent_source"] == "rcr_runtime_audit_alayer.csv" else "")
                                      for r, b_ in zip(rows, big) if b_])
    strip(A[4], "质量守恒 出口/入口", np.array([r["rcr_mass_balance"] for r in rows]), [(1.0, "")])
    Pa_, Pb_ = np.array([r["rcr_p_out_kpa_mean_parent"] for r in rows]), np.array([r["rcr_p_out_kpa_mean"] for r in rows]); mp = np.isfinite(Pa_)
    diag(A[5], "四出口平均压力 (kPa)：子 vs 母（运行态）", Pa_[mp], Pb_[mp], samp[mp], "母例 (kPa)", "子例 (kPa)", None,
         f"{int(mp.sum())} 对 · 中位 |Δ| {np.median(np.abs(Pb_[mp] - Pa_[mp])) * 1000:.0f} Pa\nAG 协议约 15 kPa，AAA/ILO 约 13 kPa")
    bigp = mp & (np.abs(Pb_ - Pa_) > 0.5)
    name_pts(A[5], Pa_[bigp], Pb_[bigp], [short(r["child"]).split("/")[0] + ("*" if r["rcr_parent_source"] == "rcr_runtime_audit_alayer.csv" else "") for r, b_ in zip(rows, bigp) if b_])
    ok_p = np.array([not r["child"].startswith("AAA/unruputer/HAN_JIAN_FU") for r in rows])
    pw = np.array([(r["pwall_p99_child"] / r["pwall_p99_parent"] - 1) * 100 for r in rows]); pw[~ok_p] = np.nan
    strip(A[6], "壁面压力 p99：子/母 − 1 (%)", pw, [(5, "±5%"), (-5, "")], "HAN_JIAN_FU~m36 未画：母例壁面压力导出本身坏（168 Pa，已知）")
    diag(A[7], "峰值 WSS p99 (Pa)：子 vs 母", np.array([r["wss_p99_parent"] for r in rows]), np.array([r["wss_p99_child"] for r in rows]), samp,
         "母例 p99 (Pa)", "子例 p99 (Pa)", None, "离对角线远的点 = 狭窄段新形成的射流热点")
    wa, wb = np.array([r["wss_p99_parent"] for r in rows]), np.array([r["wss_p99_child"] for r in rows])
    top = np.argsort(-np.abs(np.log(wb / wa)))[:4]
    name_pts(A[7], wa[top], wb[top], [short(rows[i]["child"]).split("/")[0] for i in top])
    B = [(b, r["sampled"]) for r in rows if r["in_ok59"] for b in r["bumps"]]
    fdz = np.array([b["f_design_center"] for b, _ in B]); rr_ = np.array([b["radius_ratio_child_atlas"] for b, _ in B])
    tw = np.array([b["tawss_ratio_region"] for b, _ in B]); sb = np.array([s for _, s in B])
    err = np.abs(rr_ - fdz)
    diag(A[8], "形变是否按设计落地（凸包中心）", fdz, rr_, sb, "设计 f", "子/母 中心线半径比（入库链实测）", (0.5, 1.5),
         f"{len(B)} 个凸包 · 中位 |误差| {np.median(err):.3f} · p90 {np.percentile(err, 90):.3f}")
    ax = A[9]; fx = np.linspace(0.55, 1.45, 100); ax.plot(fx, fx ** -3, color=INK2, lw=0.9, ls="--", zorder=2)
    ax.text(0.66, 0.66 ** -3 * 1.35, "f⁻³（泊肃叶）", fontsize=7.5, color=INK2, ha="left", va="bottom")
    dots(ax, rr_, tw, sb); ax.axhline(1, color=GRID, lw=1, zorder=1); ax.axvline(1, color=GRID, lw=1, zorder=1)
    ax.set_yscale("log"); ax.set_xlabel("子/母 中心线半径比"); ax.set_ylabel("凸包区 TAWSS 子/母（中位数）")
    ax.set_title("WSS 对形变的物理响应（凸包 ±σ 区）", loc="left", fontsize=10); ax.grid(color=GRID, lw=0.5, which="both")
    st_ = rr_ < 0.97; di_ = rr_ > 1.03
    ax.text(0.98, 0.98, f"狭窄 {int(np.sum(st_ & (tw > 1)))}/{int(st_.sum())} 个 WSS 升高\n扩张 {int(np.sum(di_ & (tw < 1)))}/{int(di_.sum())} 个 WSS 降低",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top", ha="right")
    fig.suptitle("数值核验：60 个合成子例（蓝实心 = 本次抽样 15 例，灰空心 = 其余）；最后两格为 59 个入库子例的全部形变凸包", fontsize=12.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "fig5_numeric_checks.png", dpi=140); plt.close(fig)

    # ------------------------------------------------------------------ 汇总
    ok = [r for r in rows if r["in_ok59"]]
    summ = {"sample_seed": SEED, "sample": sample, "split_notes": split_notes, "n_all": len(rows), "n_ok": len(ok),
            "excluded": [(r["child"], r["v5_failed"]) for r in rows if not r["in_ok59"]],
            "counts": {k.replace(chr(10), ""): {st: int(sum(fn(r)[0] == st for r in rows)) for st in ("good", "warn", "serious", "crit")} for k, fn in checks},
            "centerline_review": sorted(cl_review), "solver_tiers": {t: sum(r["solver_tier"] == t for r in rows) for t in "ABCD"},
            "wall_nodes_identical_order": sum(r["wall_nodes_identical_order"] for r in rows),
            "labels_finite": sum(r["child_labels_finite"] for r in rows), "wss_min_child_min": min(r["wss_min_child"] for r in rows),
            "ranges": {k: [float(np.min([r[k] for r in rows])), float(np.median([r[k] for r in rows])), float(np.max([r[k] for r in rows]))]
                       for k in ("cell_volume_ratio_min", "cell_volume_ratio_max", "anatomy_volume_ratio", "f_min", "f_max", "max_displacement_mm", "rcr_L",
                                 "rcr_mass_balance", "rcr_p_out_kpa_min", "rcr_p_out_kpa_max", "wss_p99_child", "wss_max_child", "continuity_last_max")},
            "pwall_p99_rel_pct": sorted([(round((r["pwall_p99_child"] / r["pwall_p99_parent"] - 1) * 100, 2), r["child"]) for r in rows]),
            "bump_radius_ratio_vs_design": {"median_abs_err": float(np.nanmedian(np.abs(rr_ - fdz))), "p90_abs_err": float(np.nanpercentile(np.abs(rr_ - fdz), 90)),
                                            "corr": float(np.corrcoef(fdz, rr_)[0, 1])},
            "bump_tawss": {"n": int(np.isfinite(tw).sum()), "stenosis_up": int(np.sum((rr_ < 0.97) & (tw > 1))), "stenosis_n": int(np.sum(rr_ < 0.97)),
                           "dilation_down": int(np.sum((rr_ > 1.03) & (tw < 1))), "dilation_n": int(np.sum(rr_ > 1.03)),
                           "spearman_log": float(np.corrcoef(np.argsort(np.argsort(rr_[np.isfinite(tw)])), np.argsort(np.argsort(tw[np.isfinite(tw)])))[0, 1])},
            "mesh_check_failed": [(r["child"], r["smoke_left_handed_faces"], r["left_handed_faces_recomputed_child"], r["left_handed_faces_recomputed_parent"]) for r in rows if r["smoke_mesh_check_failed"] or r["left_handed_faces_recomputed_child"] > 0],
            "outlet_label_mismatch": [(r["child"], r["outlet_label_diff"]) for r in rows if not r["outlet_labels_match_parent"]],
            "tier_pairs": sorted([(r["child"], r["parent_solver_tier"], r["solver_tier"]) for r in rows if r["solver_tier"] != r["parent_solver_tier"]]),
            "sample_rows": {c: {k: byid[c][k] for k in ("max_displacement_mm", "anatomy_volume_ratio", "cell_volume_ratio_min", "f_min", "f_max", "solver_tier",
                                                         "centerline", "wss_p99_parent", "wss_p99_child", "pwall_p99_child", "pwall_p99_parent", "bumps")} for c in sample}}
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("sample_rows", "pwall_p99_rel_pct")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
