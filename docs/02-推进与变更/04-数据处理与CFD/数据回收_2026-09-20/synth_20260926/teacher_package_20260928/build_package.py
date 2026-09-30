#!/usr/bin/env python
"""59 个入库合成 CFD 子例 → 交给老师的数据包（只读源数据，输出到 outputs/synth59_teacher_package_2026-09-28/）。

每例（目录 = 队列/亚组/病例，与库内一致）：
  <name>.h5                  壁面点云（坐标/外法向/面积 + 81 帧压力、WSS 三分量、WSS 标量）+ 体点（单元中心坐标/体积 + 81 帧压力、速度）
                             + 5 个开口的几何与逐帧流量/平均压力 + 时间轴；属性里带血流动力学设置、形变参数、质量标记
  <name>.stl                 解剖区壁面（形变后 Fluent 网格壁面导出，mm，开口面，与 wall/xyz_mm 逐点同源）
  <name>_hemodynamics.json   入口波形、四出口 RCR、血液流变、密度、求解设置 + 与母例一致性核验结论
  cfd_setup/                 原始 Fluent 输入：<cas>.cas.gz（形变后网格 + 全部设置）、udf-inlet*.c、2.jou
另有 manifest.csv、verification/settings_identity_59.csv、verification/package_selfcheck_59.csv、README.md、read_example.py。
建完后再跑 annotate_package.py 给 json / h5 加中文注释。

与母例"设置一模一样"的判据（逐字节）：UDF 相同；2.jou 除 read-case 路径外相同；fluent.slurm 相同；
.cas.gz 解压后去掉节点坐标段（3010 段载荷）并把绝对路径归一到文件名后逐字节相同。

    /public/newhome/cy/.conda/envs/GNN/bin/python build_package.py [--cases ID ...] [--workers 6]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from wss_pinn.v4 import new_case_sources as ncs  # noqa: E402

SYN = ROOT / "docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926"
REC = SYN.parent
V52 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_2_20260922/cases"
V51 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"
CL = ROOT / "outputs/centerline_v2_meshwall_20260927_syn60/cases"
AUD = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits"
VIZ = SYN / "viz_sample15_20260928"
OUT = ROOT / "outputs/synth59_teacher_package_2026-09-28"
PEAK_STEP = 1162
# 与 tools/synth_morph_case.py 相同的两个正则（节点段头、绝对路径串）
NODE_RE = re.compile(rb"\(3010 \(([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+)(?: ([0-9a-fA-F]+))?\)\n\(")
PATH_RE = re.compile(rb"/public/newhome/cy/Digital_twin/GNN/data(?:_new)?/[^\"\s)]*")
GZ = dict(compression="gzip", compression_opts=4, shuffle=True)
ANAT = {"inlet": ("aorta_inlet", "主动脉入口"), "out-le": ("left_external_iliac", "左髂外"), "out-li": ("left_internal_iliac", "左髂内"),
        "out-re": ("right_external_iliac", "右髂外"), "out-ri": ("right_internal_iliac", "右髂内")}
SEM_ZH = {"trunk": "主干", "left_cia": "左髂总", "right_cia": "右髂总", "left_external": "左髂外", "left_internal": "左髂内",
          "right_external": "右髂外", "right_internal": "右髂内"}
UDF_PROFILE = {"out-le": "pressure_outle", "out-li": "pressure_outli", "out-re": "pressure_outre", "out-ri": "pressure_outri"}


def parent_of(cid: str) -> str:
    return re.sub(r"~m\d+", "", cid)


def file_stem(cid: str) -> str:
    p = cid.split("/")
    return f"{p[1]}_{p[2]}" if p[0] == "ILO" else p[2]


def bundle(cid: str) -> Path:
    for root in (V52, V51):
        p = root / cid.replace("/", "__") / "case.h5"
        if p.is_file():
            return p
    raise FileNotFoundError(cid)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def jattr(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    if isinstance(v, np.generic):
        return v.item()
    return v


# ------------------------------------------------------------------------------------------------ 与母例一致性
def cas_identity(parent_cas: Path, child_cas: Path) -> dict:
    a = gzip.open(parent_cas, "rb").read(); b = gzip.open(child_cas, "rb").read()
    ma, mb = NODE_RE.search(a), NODE_RE.search(b)
    out = {"node_header_equal": bool(ma and mb and ma.group(0) == mb.group(0))}
    if not out["node_header_equal"]:
        out["identical_except_nodes_and_paths"] = False
        return out
    first, last = int(ma.group(2), 16), int(ma.group(3), 16); dim = int(ma.group(5), 16) if ma.group(5) else 3
    n = (last - first + 1) * dim * 8
    pa = np.frombuffer(a[ma.end():ma.end() + n], "<f8").reshape(-1, 3); pb = np.frombuffer(b[mb.end():mb.end() + n], "<f8").reshape(-1, 3)
    out["n_nodes"] = int(len(pa)); out["n_nodes_moved"] = int(np.sum(np.linalg.norm(pa - pb, axis=1) > 1e-12))
    out["max_node_displacement_mm"] = float(np.linalg.norm(pa - pb, axis=1).max() * 1000)
    norm = lambda d: PATH_RE.sub(lambda m: b"<PATH>/" + m.group(0).rsplit(b"/", 1)[-1], d)  # noqa: E731
    ra, rb = norm(a[:ma.end()] + a[ma.end() + n:]), norm(b[:mb.end()] + b[mb.end() + n:])
    out["bytes_compared"] = len(ra); out["identical_except_nodes_and_paths"] = ra == rb
    out["path_strings_parent"] = len(PATH_RE.findall(a)); out["path_strings_child"] = len(PATH_RE.findall(b))
    if ra != rb:
        k = next((i for i in range(min(len(ra), len(rb))) if ra[i] != rb[i]), min(len(ra), len(rb)))
        out["first_diff"] = {"offset": k, "parent": ra[max(0, k - 60):k + 60].decode("latin-1"), "child": rb[max(0, k - 60):k + 60].decode("latin-1")}
    return out


def journal_norm(text: str) -> str:
    return re.sub(r"(/file/read-case\s+)\S+", r"\1<CASE>", text)


def slurm_norm(text: str) -> str:
    """作业调度差异（#SBATCH -w 节点指定、--ntasks-per-node 与 fluent -tN 并行核数）不属于血流动力学设置。"""
    text = "\n".join(l for l in text.splitlines() if not re.match(r"\s*#SBATCH\s+(-w|--nodelist|--ntasks-per-node|--ntasks|-n)\b", l))
    return re.sub(r"(\bfluent\b.*?)-t\d+", r"\1-tN", text)


def parse_udf(text: str) -> dict:
    d = {k: float(v) for k, v in re.findall(r"#define\s+(\w+)\s+([-+0-9.eE]+)", text)}
    rcr = {}
    for lab, fn in UDF_PROFILE.items():
        m = re.search(rf"DEFINE_PROFILE\({fn}\b(.*?)\n\}}", text, re.S)
        if m:
            body = m.group(1)
            g = {k: re.search(rf"\b{k}\s*=\s*([-+0-9.eE]+)\s*;", body) for k in ("R1", "R2", "C")}
            rcr[lab] = {k: float(v.group(1)) for k, v in g.items() if v}
    area = re.search(r"\)\)\s*/\s*([0-9.eE+-]+)\s*;", text)
    threads = dict(re.findall(r"\b(t[1-4])\s*=\s*Lookup_Thread\s*\(\s*d\s*,\s*([0-9]+)\s*\)", text))
    return {"defines": d, "rcr": rcr, "inlet_area_constant_m2": float(area.group(1)) if area else None, "lookup_threads": threads}


# ------------------------------------------------------------------------------------------------ 壁面三角网 / STL
def _ear_clip(pts2: np.ndarray) -> list[tuple[int, int, int]]:
    """逆时针简单多边形的耳切（n ≤ 9，只在扇形剖分全部失败时用）。"""
    idx = list(range(len(pts2))); out = []
    cross = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])  # noqa: E731
    guard = 0
    while len(idx) > 3 and guard < 100:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts2[i0], pts2[i1], pts2[i2]
            if cross(a, b, c) <= 0:
                continue
            if any(cross(a, b, pts2[j]) >= 0 and cross(b, c, pts2[j]) >= 0 and cross(c, a, pts2[j]) >= 0 for j in idx if j not in (i0, i1, i2)):
                continue
            out.append((i0, i1, i2)); idx.pop(k); break
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


def outward_triangles(xyz: np.ndarray, fnodes: np.ndarray, off: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, dict]:
    """把 Fluent 壁面多边形剖成三角形，不加新顶点：每个三角形法向与该多边形的外向面积向量同向。
    bundle 的 topology/wall_triangles 是从第一个顶点扇形剖分、整体朝内，非凸多边形会折出少量反向三角形；
    这里先按多边形逐个换扇形起点，全部失败再在多边形平面内耳切。"""
    k = np.diff(off); tris = np.empty((int((k - 2).sum()), 3), np.int64); pos = np.r_[0, np.cumsum(k - 2)]
    stats = {"polygons": int(len(k)), "refanned": 0, "ear_clipped": 0}
    for size in np.unique(k):
        sel = np.flatnonzero(k == size); P = fnodes[off[sel][:, None] + np.arange(size)]; done = np.zeros(len(sel), bool)
        for r in range(size):
            Pr = np.roll(P, -r, axis=1)
            T = np.stack([np.repeat(Pr[:, :1], size - 2, 1), Pr[:, 2:], Pr[:, 1:-1]], axis=2)  # (m, size-2, 3)，反转原绕向 → 朝外
            v = xyz[T]; nrm = np.cross(v[..., 1, :] - v[..., 0, :], v[..., 2, :] - v[..., 0, :])
            good = (np.einsum("mtj,mj->mt", nrm, A[sel]) > 0).all(1) & ~done
            for m in np.flatnonzero(good):
                tris[pos[sel[m]]:pos[sel[m] + 1]] = T[m]
            if r > 0:
                stats["refanned"] += int(good.sum())
            done |= good
        for m in np.flatnonzero(~done):  # 耳切兜底
            poly = P[m]; nA = A[sel[m]] / np.linalg.norm(A[sel[m]])
            e1 = np.cross(nA, [1.0, 0, 0]) if abs(nA[0]) < 0.9 else np.cross(nA, [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nA, e1)
            q = xyz[poly] - xyz[poly].mean(0); pts2 = np.stack([q @ e1, q @ e2], 1)
            ear = _ear_clip(pts2) if len(pts2) > 3 else [(0, 1, 2)]
            if len(ear) != size - 2:
                raise RuntimeError(f"ear clipping failed on polygon {sel[m]}")
            tris[pos[sel[m]]:pos[sel[m] + 1]] = poly[np.array(ear)]
            stats["ear_clipped"] += 1
    v = xyz[tris]; nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    face = np.repeat(np.arange(len(k)), k - 2)
    stats["outward_fraction"] = float(np.mean(np.einsum("ij,ij->i", nrm, A[face]) > 0))
    # 不变量：同一多边形任意不加点的剖分，三角形面积向量之和 = 多边形面积向量（非平面多边形也成立；xyz 为 mm、A 为 m²）
    vec = np.zeros_like(A)
    for ax in range(3):
        vec[:, ax] = np.bincount(face, weights=0.5 * nrm[:, ax], minlength=len(k)) * 1e-6
    stats["area_rel_err"] = float((np.linalg.norm(vec - A, axis=1) / np.linalg.norm(A, axis=1)).max())
    return tris.astype(np.int32), stats


def write_stl(path: Path, xyz: np.ndarray, tris: np.ndarray, header: str) -> None:
    v = xyz[tris]; n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = n; rec["v"] = v
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii")[:80].ljust(80, b" ")); fh.write(np.uint32(len(tris)).tobytes()); fh.write(rec.tobytes())


# ------------------------------------------------------------------------------------------------ 每例
def build_case(args) -> dict:
    cid, flags = args
    t0 = time.time()
    parent = parent_of(cid); stem = file_stem(cid)
    cdir = OUT / cid; cdir.mkdir(parents=True, exist_ok=True); (cdir / "cfd_setup").mkdir(exist_ok=True)
    raw_c, raw_p = ROOT / "data_new" / cid, ROOT / "data_new" / parent
    morph = json.loads((raw_c / "synth_morph.json").read_text())
    with h5py.File(bundle(parent), "r") as pf:  # 凸包所在血管段名（母例 atlas 的段编号 → 语义）
        sem_of = {int(k): int(v) for k, v in json.loads(pf["geometry"].attrs["semantic_of_segment"]).items()}
        sem_lab = {int(k): v for k, v in json.loads(pf["geometry"].attrs["semantic_labels"]).items()}
    for bmp in morph["bumps"]:
        bmp["segment_name"] = sem_lab[sem_of[int(bmp["segment_id"])]]; bmp["segment_zh"] = SEM_ZH[bmp["segment_name"]]
    # 母例库内出口标签与本包（出口边界名 + UDF）相反的病例（CHEN_SHU_LIN~m25），形变段名随本包标签改正；annotate_package.py 同一规则
    lab_sem = {"out-le": "left_external", "out-li": "left_internal", "out-re": "right_external", "out-ri": "right_internal"}
    if flags.get("outlet_label_diff"):
        swap = {lab_sem[a]: lab_sem[b] for a, b in (it.split(":", 1)[1].split("->") for it in flags["outlet_label_diff"].split(";"))}
        for bmp in morph["bumps"]:
            bmp["segment_name"] = swap.get(bmp["segment_name"], bmp["segment_name"]); bmp["segment_zh"] = SEM_ZH[bmp["segment_name"]]
        morph["segment_labels_follow_package"] = True
    cas_c = Path(morph["cas_child"]); cas_p = ncs.fluent_case(parent)
    udf_c, udf_p = ncs.udf_path(raw_c), ncs.udf_path(raw_p)
    udf_text = udf_c.read_text(errors="ignore"); udf = parse_udf(udf_text)
    jou_c, jou_p = (raw_c / "2.jou").read_text(), (raw_p / "2.jou").read_text()
    topo = json.loads((AUD / "topology/cases" / (cid.replace("/", "__") + ".json")).read_text())
    iface_audit = {i["semantic_label"]: i for i in topo["interfaces"]}

    # ---- 一致性核验
    ident = {"parent": parent, "udf_parent_file": udf_p.name, "udf_child_file": udf_c.name,
             "udf_identical": sha256(udf_c) == sha256(udf_p), "udf_sha256": sha256(udf_c),
             "journal_identical_except_case_path": journal_norm(jou_c) == journal_norm(jou_p),
             "slurm_identical": (raw_c / "fluent.slurm").read_bytes() == (raw_p / "fluent.slurm").read_bytes(),
             "slurm_identical_except_scheduling": slurm_norm((raw_c / "fluent.slurm").read_text()) == slurm_norm((raw_p / "fluent.slurm").read_text()),
             "cas": cas_identity(cas_p, cas_c)}
    ident["all_settings_identical"] = bool(ident["udf_identical"] and ident["journal_identical_except_case_path"]
                                           and ident["cas"].get("identical_except_nodes_and_paths", False))

    src = h5py.File(bundle(cid), "r")
    cond = {k: jattr(v) for k, v in src["conditions"].attrs.items()}
    # RCR 协议 = Murray r³ 按出口面积：同侧两支导纳比应 ≈ 开口面积比^1.5
    po = cond["protocol"]["per_outlet"]; area = {k: float(src[f"interfaces/{k}"].attrs["area_m2"]) for k in po}
    murray = {}
    for side, (e, i) in {"left": ("out-le", "out-li"), "right": ("out-re", "out-ri")}.items():
        ra = (area[e] / area[i]) ** 1.5; rg = po[i]["R_sum"] / po[e]["R_sum"]
        murray[side] = {"area_ratio_pow1.5_ext_over_int": ra, "conductance_ratio_ext_over_int": rg, "log_deviation": float(np.log(rg / ra))}
    flags = dict(flags, notes=list(flags["notes"]), rcr_murray_check=murray)
    for side, mv in murray.items():
        if abs(mv["log_deviation"]) > 0.35:
            zh = "左" if side == "left" else "右"
            flags["notes"].append(f"{zh}侧髂内/髂外两出口的 RCR 与开口面积按 Murray 规则互换（母例 CFD 设置即如此，子例原样继承）："
                                  f"导纳比 外/内 = {mv['conductance_ratio_ext_over_int']:.2f}，按面积应为 {mv['area_ratio_pow1.5_ext_over_int']:.2f}；"
                                  f"结果是{zh}髂外血流偏少、{zh}髂内偏多")
    steps = src["wall_temporal/step"][...]; kpk = int(np.flatnonzero(steps == PEAK_STEP)[0])
    js = re.search(r"/solve/set/time-step\s+([0-9.eE+-]+)", jou_c); jd = re.search(r"/solve/dual-time-iterate\s+(\d+)\s+(\d+)", jou_c)
    rho = float(cond["rho_kg_m3"])
    rcr_bundle = {r["outlet"]: r for r in cond["rcr"]}
    outlets = []
    for lab in ("out-le", "out-li", "out-re", "out-ri"):
        r = rcr_bundle[lab]; ia = iface_audit.get(lab, {})
        outlets.append({"label": lab, "vessel": ANAT[lab][0], "vessel_zh": ANAT[lab][1],
                        "fluent_bc_name": ia.get("distal_bc_name"), "fluent_bc_zone_id": ia.get("distal_bc_zone_id"),
                        "fluent_interface_face_zone": ia.get("face_zone_names"), "udf_profile": UDF_PROFILE[lab],
                        "R1_Pa_s_per_kg": r["R1"], "R2_Pa_s_per_kg": r["R2"], "C_kg_per_Pa": r["C"],
                        "R1_Pa_s_per_m3": r["R1"] * rho, "R2_Pa_s_per_m3": r["R2"] * rho, "C_m3_per_Pa": r["C"] / rho,
                        "udf_matches_bundle": udf["rcr"].get(lab) == {"R1": r["R1"], "R2": r["R2"], "C": r["C"]},
                        "opening_area_m2": float(src[f"interfaces/{lab}"].attrs["area_m2"])})
    fr = cond["fourier"]; rh = cond["rheology"]
    hemo = {
        "case_id": cid, "parent_id": parent,
        "note": "合成子例只移动了母例 Fluent 网格解剖区节点坐标；边界条件、UDF、流变、求解设置与母例逐字节相同（见 identical_to_parent）。",
        "fluid": {"density_kg_m3": rho, "viscosity_model": "Carreau-Yasuda（UDF cell_viscosity）",
                  "formula": "mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * shear_rate)^a)^((n - 1) / a)",
                  "mu_inf_Pa_s": rh["mu_inf_pa_s"], "mu_0_Pa_s": rh["mu_zero_pa_s"], "lambda_s": rh["lambda_s"], "a": rh["a"], "n": rh["n"],
                  "udf_defines_match": [udf["defines"].get(k) for k in ("A1", "B", "D", "E", "n")] == [rh["mu_inf_pa_s"], rh["mu_zero_pa_s"], rh["lambda_s"], rh["a"], rh["n"]]},
        "inlet": {"bc_type": "velocity-inlet，截面均匀速度（UDF my_inlet）",
                  "flow_formula": "Q(t) = scale * (a0 + sum_k a_k cos(k w tt) + b_k sin(k w tt))，tt = t mod period；速度 = Q(t) / A_udf",
                  "fourier": fr, "period_s": cond["period_s"], "A_udf_m2": cond["A_inlet_udf_m2"], "A_inlet_face_m2": cond["A_inlet_bc_face_m2"],
                  "inlet_area_ratio_face_over_udf": cond["inlet_area_ratio"],
                  "actual_inflow_note": "实际入流量 = Q(t) × A_inlet_face / A_udf；逐帧实测入流见 h5 boundary/inlet/flux_outward_m3s（取负）",
                  "q_nominal_m3s_at_frames": [float(x) for x in src["conditions/q_nom_m3s"][...]]},
        "outlets": {"bc_type": "pressure-outlet，三元 Windkessel（RCR，UDF）",
                    "discretization": "P_n = ((R1 + R2 + R1*beta) * Q_n + beta * P_{n-1} - R1 * beta * Q_{n-1}) / (1 + beta)，beta = R2*C/dt，Q = 出口质量流量 (kg/s)",
                    "units_note": "UDF 按质量流量积分：R 单位 Pa·s/kg、C 单位 kg/Pa；*_Pa_s_per_m3 / C_m3_per_Pa 是按密度换算的体积流量口径",
                    "per_outlet": outlets, "protocol": cond.get("protocol", {}).get("protocol_id"),
                    "rcr_rule": "Murray r³ 按出口面积分配（协议见组内 RCR 审计）；开口 12 mm 内节点不动 → 出口面积与母例相同 → RCR 无需重算"},
        "walls": "刚性壁、无滑移",
        "solver": {"software": "ANSYS Fluent 2023 R1（fluent/231），3ddp，64 进程", "flow": "层流（残差只有连续性 + 三个速度分量）",
                   "time_scheme": "瞬态 dual-time", "time_step_s": float(js.group(1)) if js else None,
                   "n_time_steps": int(jd.group(1)) if jd else None, "max_iterations_per_step": int(jd.group(2)) if jd else None,
                   "operating_pressure_Pa": 101325, "pressure_output": "表压（相对操作压力）",
                   "exported_frames": "最后一个周期：step 1120–1280 每 2 步一帧，共 81 帧（time 5.60–6.40 s，phase 0–0.80 s）",
                   "peak_frame": {"step": PEAK_STEP, "index": kpk, "phase_s": float(src["wall_temporal/phase_s"][kpk])},
                   "continuity_residual_tier": cond["solver"]["tier"], "continuity_last_max": cond["solver"]["continuity_last_max"]},
        "identical_to_parent": ident,
    }
    (cdir / f"{stem}_hemodynamics.json").write_text(json.dumps(hemo, ensure_ascii=False, indent=1))

    # ---- HDF5
    tmp = cdir / f".{stem}.h5.tmp"; dst_path = cdir / f"{stem}.h5"
    nw = src["wall_static/xyz_mm"].shape[0]; nv = src["volume_static/xyz_mm"].shape[0]; nf = len(steps)
    with h5py.File(tmp, "w") as h:
        h.attrs.update({"case_id": cid, "parent_id": parent, "cohort": cid.split("/")[0], "file_format_version": "synth59-teacher-1",
                        "created": time.strftime("%Y-%m-%d"), "coordinate_frame": "Fluent 算例坐标系，mm（Fluent 米 × 1000），与 STL 相同",
                        "units": json.dumps({"xyz": "mm", "pressure": "Pa（表压）", "wss": "Pa", "velocity": "m/s", "area": "m^2", "volume": "m^3", "time": "s"}, ensure_ascii=False),
                        "hemodynamics": json.dumps(hemo, ensure_ascii=False), "morph": json.dumps(morph, ensure_ascii=False),
                        "quality_flags": json.dumps(flags, ensure_ascii=False)})
        g = h.create_group("time")
        g["step"] = steps.astype(np.int32); g["time_s"] = src["wall_temporal/time_s"][...]; g["phase_s"] = src["wall_temporal/phase_s"][...]
        g.attrs.update({"peak_index": kpk, "peak_step": PEAK_STEP, "period_s": cond["period_s"]})
        w = h.create_group("wall")
        w.attrs["description"] = "解剖区壁面节点（与 STL 顶点逐点对应）；壁面速度按无滑移恒为 0，故速度只在 volume 组"
        w.create_dataset("xyz_mm", data=src["wall_static/xyz_mm"][...], **GZ)
        w.create_dataset("normal_out", data=src["wall_static/normal_out_mesh"][...].astype(np.float32), **GZ)
        w.create_dataset("area_m2", data=src["wall_static/area_m2"][...], **GZ)
        w.create_dataset("fluent_node_id", data=src["wall_static/node_id_cas"][...], **GZ)
        tris, tri_stats = outward_triangles(src["wall_static/xyz_mm"][...], src["topology/wall_face_nodes"][...],
                                            src["topology/wall_face_offsets"][...], src["topology/wall_face_area_vec_m2"][...])
        w.create_dataset("triangles", data=tris, **GZ)
        w["triangles"].attrs["note"] = "壁面三角网（索引 wall 行，从 0 开始），由 Fluent 壁面多边形剖分、不加新顶点，法向朝外；与 STL 三角形逐个相同"
        for name, sname, shape in (("pressure_pa", "wall_temporal/pressure_pa", (nf, nw)), ("wss_scalar_pa", "wall_temporal/wss_scalar_pa", (nf, nw)),
                                   ("wss_vector_pa", "wall_temporal/wss_vector_pa", (nf, nw, 3))):
            d = w.create_dataset(name, shape=shape, dtype=np.float32, chunks=(1,) + shape[1:], **GZ)
            for t in range(nf):
                d[t] = src[sname][t]
        w["wss_scalar_pa"].attrs["note"] = "Fluent wall-shear 标量导出列；与 wss_vector_pa 模长的相对差各例中位 0.1–0.6%（插值差），做标量任务用本列"
        v = h.create_group("volume")
        v.attrs["description"] = "解剖区（与壁面同一区域，不含入口/出口延伸段）网格单元中心；cell-centred 值"
        v.create_dataset("xyz_mm", data=src["volume_static/xyz_mm"][...], **GZ)
        v.create_dataset("volume_m3", data=src["volume_static/volume_m3"][...], **GZ)
        v.create_dataset("fluent_cell_id", data=src["volume_static/cell_id_cas"][...], **GZ)
        vch = min(nv, 1 << 18)
        for name, sname, shape, ch in (("pressure_pa", "volume_temporal/pressure_pa", (nf, nv), (1, vch)),
                                       ("velocity_m_s", "volume_temporal/velocity_m_s", (nf, nv, 3), (1, vch, 3))):
            d = v.create_dataset(name, shape=shape, dtype=np.float32, chunks=ch, **GZ)
            for t in range(nf):
                d[t] = src[sname][t]
        b = h.create_group("boundary")
        b.attrs["description"] = "解剖区与延伸段交界的 5 个开口；flux_outward_m3s 正 = 流出，入口为负；p_mean_pa = 面积平均压力"
        for lab in ("inlet", "out-le", "out-li", "out-re", "out-ri"):
            s = src[f"interfaces/{lab}"]; gb = b.create_group(lab); ia = iface_audit.get(lab, {})
            gb.attrs.update({"vessel": ANAT[lab][0], "vessel_zh": ANAT[lab][1], "area_m2": float(s.attrs["area_m2"]),
                             "center_mm": np.asarray(jattr(s.attrs["center_mm"]), float), "normal_out": np.asarray(jattr(s.attrs["normal_out"]), float),
                             "fluent_bc_name": str(ia.get("distal_bc_name", "")), "fluent_bc_zone_id": int(ia.get("distal_bc_zone_id", -1))})
            for k in ("flux_outward_m3s", "p_mean_pa", "face_center_mm", "face_area_vec_m2"):
                gb.create_dataset(k, data=s[k][...])
    os.replace(tmp, dst_path)

    # ---- STL + CFD 设置原件
    stl_dst = cdir / f"{stem}.stl"
    with h5py.File(dst_path, "r") as h:
        write_stl(stl_dst, h["wall/xyz_mm"][...], h["wall/triangles"][...], f"{cid} anatomy wall, mm, Fluent frame, outward normals")
    for f in (cas_c, udf_c, raw_c / "2.jou"):
        shutil.copy2(f, cdir / "cfd_setup" / f.name)

    # ---- 写后自检（逐位对照源 bundle + STL 顶点）
    chk = {"case_id": cid}
    with h5py.File(dst_path, "r") as h:
        for grp, key, sname in (("wall", "pressure_pa", "wall_temporal/pressure_pa"), ("wall", "wss_vector_pa", "wall_temporal/wss_vector_pa"),
                                ("wall", "wss_scalar_pa", "wall_temporal/wss_scalar_pa"), ("volume", "pressure_pa", "volume_temporal/pressure_pa"),
                                ("volume", "velocity_m_s", "volume_temporal/velocity_m_s")):
            ok = all(np.array_equal(h[grp][key][t], src[sname][t]) for t in (0, kpk, nf - 1))
            chk[f"{grp}_{key}_bitwise"] = ok
            chk[f"{grp}_{key}_finite"] = bool(np.isfinite(h[grp][key][kpk]).all())
        chk["wss_scalar_nonneg"] = bool((h["wall/wss_scalar_pa"][kpk] >= 0).all())
        chk["n_wall"] = nw; chk["n_volume"] = nv; chk["n_frames"] = nf; chk["n_triangles"] = int(h["wall/triangles"].shape[0])
        xyz = h["wall/xyz_mm"][...]
    raw = stl_dst.read_bytes(); nt = int(np.frombuffer(raw[80:84], "<u4")[0])
    rec = np.frombuffer(raw[84:84 + nt * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    dd, ii = cKDTree(xyz).query(rec["v"].reshape(-1, 3).astype(float))
    chk.update({"stl_triangles": nt, "stl_triangles_equal": nt == chk["n_triangles"], "stl_vertex_max_dist_mm": float(dd.max()),
                "stl_vertex_within_1e-4mm": bool(dd.max() < 1e-4), "stl_covers_all_wall_nodes": int(len(np.unique(ii))) == nw,
                "tri_outward_fraction": tri_stats["outward_fraction"], "tri_all_outward": tri_stats["outward_fraction"] == 1.0,
                "tri_area_rel_err": tri_stats["area_rel_err"], "tri_area_ok": tri_stats["area_rel_err"] < 1e-6,
                "tri_refanned": tri_stats["refanned"], "tri_ear_clipped": tri_stats["ear_clipped"]})
    src.close()
    sizes = {p.name: p.stat().st_size for p in cdir.rglob("*") if p.is_file()}
    row = {"case_id": cid, "parent_id": parent, "cohort": cid.split("/")[0], "stem": stem, "n_wall": nw, "n_volume": nv, "n_frames": nf,
           "all_settings_identical": ident["all_settings_identical"], "h5_bytes": dst_path.stat().st_size, "case_dir_bytes": sum(sizes.values()),
           "h5_sha256": sha256(dst_path), "stl_sha256": sha256(stl_dst), "seconds": round(time.time() - t0, 1)}
    return {"row": row, "ident": ident, "check": chk, "hemo": hemo, "morph": morph, "flags": flags}


def load_flags() -> dict:
    ver = {r["child"]: r for r in csv.DictReader(open(VIZ / "verification_60.csv"))}
    hand = {r["child"]: r for r in csv.DictReader(open(VIZ / "face_handedness_60.csv"))}
    flags = {}
    for c, r in ver.items():
        h = hand.get(c, {}); notes = []
        lh = int(h.get("left_handed_child", 0) or 0)
        if r["smoke_mesh_check_failed"] == "True" or lh > 0:
            notes.append(f"形变引入网格左手面（Fluent 冒烟报 {r['smoke_left_handed_faces']} 个，逐面复算 {lh} 个，均在离壁 ≤0.5 mm 边界层内部面，母例 0；已正常求解并过全部门）")
        if r["outlet_labels_match_parent"] != "True":
            notes.append("左髂内/左髂外出口标签与我们库内母例标签相反：母网格左侧内部接口面名（leftwai/leftnei）与出口边界名（leftwai+/leftnei+）交叉；本包按出口边界名与 UDF 标注（按开口位置与右侧对称判断为解剖上正确），物理设置与母例相同")
        if r["solver_tier"] in ("C", "D"):
            notes.append(f"求解层级 {r['solver_tier']}（81 个导出步最后一次连续性残差最大值 {float(r['continuity_last_max']):.1e}；母例 {r['parent_solver_tier']}）")
        if r["centerline"] == "review":
            notes.append("中心线硬门通过、标记人工复核（不影响本包数据）")
        flags[c] = {"left_handed_faces_fluent": int(r["smoke_left_handed_faces"]), "left_handed_faces_recomputed": lh,
                    "outlet_labels_match_parent_library": r["outlet_labels_match_parent"] == "True", "outlet_label_diff": r["outlet_label_diff"],
                    "solver_tier": r["solver_tier"], "parent_solver_tier": r["parent_solver_tier"],
                    "continuity_last_max": float(r["continuity_last_max"]), "min_cell_volume_ratio": float(r["cell_volume_ratio_min"]),
                    "rcr_left_share_child": float(r["rcr_L"]), "rcr_left_share_parent": float(r["rcr_L_parent"]),
                    "notes": notes}
    return flags


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", nargs="*"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    ids = a.cases or [x.strip() for x in (SYN / "synth_children_ok59.txt").read_text().splitlines() if x.strip()]
    flags = load_flags()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "verification").mkdir(exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.map(build_case, [(c, flags[c]) for c in ids], chunksize=1)
    order = {"AG": 0, "AAA": 1, "ILO": 2}
    res.sort(key=lambda r: (order[r["row"]["cohort"]], r["row"]["case_id"]))
    tag = "" if a.cases is None else "_partial"
    # 一致性表
    with open(OUT / "verification" / f"settings_identity_59{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "parent_id", "all_settings_identical", "udf_identical", "journal_identical_except_case_path", "slurm_identical",
                    "slurm_identical_except_scheduling", "cas_identical_except_nodes_and_paths", "cas_bytes_compared", "cas_nodes", "cas_nodes_moved", "max_node_displacement_mm",
                    "rcr_udf_matches_bundle", "rheology_udf_matches_bundle", "left_share_child", "left_share_parent"])
        for r in res:
            i, hm = r["ident"], r["hemo"]; c = i["cas"]
            w.writerow([r["row"]["case_id"], i["parent"], i["all_settings_identical"], i["udf_identical"], i["journal_identical_except_case_path"],
                        i["slurm_identical"], i["slurm_identical_except_scheduling"], c.get("identical_except_nodes_and_paths"), c.get("bytes_compared"), c.get("n_nodes"), c.get("n_nodes_moved"),
                        round(c.get("max_node_displacement_mm", float("nan")), 3), all(o["udf_matches_bundle"] for o in hm["outlets"]["per_outlet"]),
                        hm["fluid"]["udf_defines_match"], flags[r["row"]["case_id"]]["rcr_left_share_child"], flags[r["row"]["case_id"]]["rcr_left_share_parent"]])
    with open(OUT / "verification" / f"package_selfcheck_59{tag}.csv", "w", newline="") as fh:
        keys = list(res[0]["check"].keys()); w = csv.writer(fh); w.writerow(keys)
        for r in res:
            w.writerow([r["check"][k] for k in keys])
    # 总表
    with open(OUT / f"manifest{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "cohort", "subgroup", "stage", "parent_id", "dir", "h5", "stl", "n_wall_points", "n_volume_points", "n_frames",
                    "bump1", "bump2", "morph_seed", "max_node_displacement_mm", "anatomy_volume_ratio", "min_cell_volume_ratio",
                    "settings_identical_to_parent", "solver_tier", "quality_notes", "h5_sha256", "stl_sha256", "case_dir_MB"])
        for r in res:
            row, m, fl = r["row"], r["morph"], r["flags"]; p = row["case_id"].split("/")
            bt = [f"{b['segment_zh']}{'狭窄' if b['amplitude'] < 0 else '扩张'} a={b['amplitude']:+.3f} σ={b['sigma_mm']:.1f}mm s={b['s_local_mm']:.1f}mm (seg {b['segment_id']})" for b in m["bumps"]]
            w.writerow([row["case_id"], p[0], p[1] if p[0] != "ILO" else "", p[2] if p[0] == "ILO" else "", row["parent_id"], row["case_id"],
                        f"{row['stem']}.h5", f"{row['stem']}.stl", row["n_wall"], row["n_volume"], row["n_frames"], bt[0], bt[1] if len(bt) > 1 else "",
                        m["seed"], round(m["max_displacement_mm"], 3), round(m["anatomy_volume_ratio"], 4), round(m["cell_volume_ratio_min"], 3),
                        row["all_settings_identical"], fl["solver_tier"], "；".join(fl["notes"]), row["h5_sha256"], row["stl_sha256"],
                        round(row["case_dir_bytes"] / 1e6, 1)])
    bad = [r["row"]["case_id"] for r in res if not r["ident"]["all_settings_identical"]]
    chk_bad = [r["row"]["case_id"] for r in res if not all(v for k, v in r["check"].items() if isinstance(v, bool))]
    print(json.dumps({"cases": len(res), "settings_not_identical": bad, "selfcheck_failed": chk_bad,
                      "total_GB": round(sum(r["row"]["case_dir_bytes"] for r in res) / 1e9, 2),
                      "slowest_s": max(r["row"]["seconds"] for r in res)}, ensure_ascii=False, indent=1))
    for r in res:
        if not r["ident"]["all_settings_identical"]:
            print(r["row"]["case_id"], json.dumps(r["ident"], ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
