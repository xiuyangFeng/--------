#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单病例预处理 -> bundle。

顺序（严格按需求：先配准/正交化/标准化，最后才稀疏化——稀疏化不在这里，在 build_samples）：
1. 读几何（坐标静态，只读一次）+ 中心线 + 峰值收缩期
2. 中心线主导刚性配准（正交化：正交基 + det=+1）
3. 坐标逐病例各向同性标准化到 [-1,1]（保形）
4. 点分类掩码：wall / near_wall / interior（KNN 到壁面的距离）
5. 保留几何特征（dist_to_wall, abscissa_norm, local_radius, curvature）供后续 mask
6. 堆叠全部时间步的壁面 WSS（标量必存；矢量/内部按配置）
7. 存 .npz bundle（原始 WSS 不做全局标准化——全局标准化在 build_samples 用全局统计做）
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import config as C
from . import raw_io
from . import reporting
from . import surface_io
from .registration import (
    compute_transform, RigidTransform, untraced_inlet_crop_masks, AXIS_INDEX,
)


# 点类型编码
WALL, NEAR_WALL, INTERIOR = 2, 1, 0

# 壁面必须列（否则无法提供 WSS 目标；ID 列用 nodenumber/cellnumber 单独校验）
_REQUIRED_WALL_COLS = ["x-coordinate", "y-coordinate", "z-coordinate", "wall-shear"]
_REQUIRED_INT_COLS = ["x-coordinate", "y-coordinate", "z-coordinate"]


class SkipCase(Exception):
    """原始数据不完整，按“跳过”处理（非报错）。message 即原因。"""


def _validate_case(case_dir: Path, case_name: str, steps, cfg) -> Dict:
    """跑重活前先校验原始文件完整性；返回探测信息（分隔符/列）。"""
    if not (case_dir / C.RAW_LAYOUT["wall_ascii_dir"]).is_dir():
        raise SkipCase("缺少壁面 ascii 目录")
    if not steps:
        raise SkipCase("无导出时间步")
    if not (case_dir / C.RAW_LAYOUT["centerline_csv"]).is_file():
        raise SkipCase("缺少 centerline_points.csv")

    ref = steps[0]
    wall_f = raw_io._step_file(case_dir, C.RAW_LAYOUT["wall_ascii_dir"], case_name, ref)
    int_f = raw_io._step_file(case_dir, C.RAW_LAYOUT["interior_ascii_dir"], case_name, ref)
    if not wall_f.is_file():
        raise SkipCase(f"壁面首步文件缺失: {wall_f.name}")
    if not int_f.is_file():
        raise SkipCase(f"内部首步文件缺失: {int_f.name}")

    wall_cols = raw_io.header_columns(wall_f)
    int_cols = raw_io.header_columns(int_f)
    wall_id_col = raw_io.wall_id_column(wall_cols)
    if wall_id_col is None:
        raise SkipCase("壁面导出缺少 nodenumber/cellnumber ID 列")
    miss_w = [c for c in _REQUIRED_WALL_COLS if c not in wall_cols]
    if miss_w:
        raise SkipCase(f"壁面导出缺列: {miss_w}")
    miss_i = [c for c in _REQUIRED_INT_COLS if c not in int_cols]
    if miss_i:
        raise SkipCase(f"内部导出缺列: {miss_i}")

    return {
        "wall_delimiter": raw_io.delimiter_of(wall_f),
        "wall_id_column": wall_id_col,
        "interior_delimiter": raw_io.delimiter_of(int_f),
    }


def _bbox_diag(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _resolve_unit_factor(wall_native: np.ndarray, cl: Dict[str, np.ndarray], ucfg):
    """反推 mesh 原生坐标 -> 毫米 的缩放因子（鲁棒版）。

    正常情况下网格与中心线取自同一段几何、范围一致，壁面包围盒对角线 × factor ≈
    中心线对角线，用比值 factor_ratio = cl_diag / mesh_diag 即可（壁面比中心线略大
    ~1 个半径，几个百分点误差不影响判定）。

    但个别病例中心线只描了远端一段、壁面网格却含整条近端主动脉（覆盖范围不一致），
    比值法会把 factor 压到真实值的一半，导致坐标缩放/配准全线偏移。此时改用
    “米制整十次幂”兜底：Fluent ascii 基本为 SI 米，选使壁面对角线落到生理尺度
    (~phys_diag_mm) 的最近 10^k 作为 factor，并标记 extent_mismatch 待人工复核。
    返回 (factor, is_anomaly, extent_mismatch)。
    """
    if ucfg.mode == "fixed":
        f = ucfg.fixed_factor
        anomaly = abs(np.log10(f) - 3.0) > ucfg.anomaly_log10_tol if f > 0 else True
        return f, anomaly, False

    mesh_diag = _bbox_diag(wall_native)
    cl_diag = _bbox_diag(cl["coords"])
    if mesh_diag <= 1e-12:
        return ucfg.fixed_factor, True, False

    f_ratio = cl_diag / mesh_diag
    # 整十次幂兜底：让壁面对角线最接近生理尺度（对米制/异常单位都自适应）
    k = int(round(np.log10(ucfg.phys_diag_mm / mesh_diag)))
    f_snap = 10.0 ** k

    # 覆盖范围一致时（正常病例）比值≈整十次幂，采用精细比值；两者偏离超过
    # 偏离超过可信倍数时判定壁面与中心线覆盖不一致，改用整十次幂因子。
    ratio_to_snap = (f_ratio / f_snap) if f_snap > 0 else np.inf
    extent_mismatch = not (1.0 / ucfg.ratio_trust <= ratio_to_snap <= ucfg.ratio_trust)
    f = f_snap if extent_mismatch else f_ratio

    anomaly = bool(extent_mismatch or (
        abs(np.log10(f) - 3.0) > ucfg.anomaly_log10_tol if f > 0 else True))
    return f, anomaly, extent_mismatch


def _nearest_centerline_feats(pts: np.ndarray, cl: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """每个点最近的中心线采样点 -> abscissa(归一到[0,1]), 局部内切半径, 曲率。"""
    nn = NearestNeighbors(n_neighbors=1).fit(cl["coords"])
    _, idx = nn.kneighbors(pts)
    idx = idx[:, 0]
    absc = cl["abscissa"]
    absc_span = max(absc.max() - absc.min(), 1e-9)
    return {
        "abscissa_norm": (absc[idx] - absc.min()) / absc_span,
        "local_radius": cl["radius"][idx],
        "curvature": cl["curvature"][idx],
    }


def _same_id_set(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return len(a) == len(b) and np.array_equal(np.sort(a), np.sort(b))


def _select_stable_wall_reference(case_dir: Path, case_name: str, steps, peak: int):
    """由首步/峰值步/末步的共识节点集合选择静态壁面参考。"""
    # 同等共识下优先峰值步，确保正式单步目标与几何严格同点；首/末步只作稳定性佐证。
    candidate_steps = list(dict.fromkeys((int(peak), int(steps[0]), int(steps[-1]))))
    candidates = [(step, raw_io.read_wall_fields(case_dir, case_name, step))
                  for step in candidate_steps]
    agreement = []
    for _, fields in candidates:
        ids = fields["nodenumber"]
        agreement.append(sum(_same_id_set(ids, other["nodenumber"])
                             for _, other in candidates))
    best = int(np.argmax(agreement))
    if len(candidates) >= 3 and agreement[best] < 2:
        detail = [(step, len(fields["nodenumber"])) for step, fields in candidates]
        raise RuntimeError(f"壁面参考节点集合无跨时间步共识: {detail}")
    return candidates[best]


def _spatially_align_wall_fields_to_reference(
    fields: Dict[str, np.ndarray],
    ref_coords: np.ndarray,
    ref_ids: np.ndarray,
    step: int,
    tol: float = 1e-9,
) -> tuple[Dict[str, np.ndarray], bool, float]:
    """修复极少量 nodenumber↔坐标循环错配，但不接受真实移动网格。

    只有当前步与参考步的坐标集合一一完全匹配时才按坐标重排字段；若点集本身变化，
    保持原样交给 QA 拒绝。这样可处理 Fluent 少量节点 ID 循环置换而不掩盖移动壁面。
    """
    coords = np.asarray(fields["coords"], dtype=np.float64)
    ref_coords = np.asarray(ref_coords, dtype=np.float64)
    if len(coords) != len(ref_coords):
        return fields, False, float("inf")
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords)
    distances, indices = nn.kneighbors(ref_coords)
    distances = distances[:, 0]
    indices = indices[:, 0]
    max_distance = float(distances.max()) if len(distances) else 0.0
    if max_distance > tol or len(np.unique(indices)) != len(indices):
        return fields, False, max_distance
    if np.array_equal(indices, np.arange(len(indices))):
        return fields, False, max_distance
    out: Dict[str, np.ndarray] = {}
    for key, val in fields.items():
        arr = np.asarray(val)
        out[key] = arr[indices] if len(arr) == len(coords) else val
    out["nodenumber"] = np.asarray(ref_ids, dtype=np.int64).copy()
    out["coords"] = ref_coords.copy()
    return out, True, max_distance


def _align_wall_fields_to_reference(
    fields: Dict[str, np.ndarray],
    ref_ids: np.ndarray,
    step: int,
) -> tuple[Dict[str, np.ndarray], bool, int]:
    """按稳定参考 nodenumber 对齐，允许丢弃极少量非稳定额外节点。

    稳定节点缺失、重复或额外比例过大仍直接报错，防止标签与几何静默错位。
    """
    ids = np.asarray(fields["nodenumber"], dtype=np.int64)
    ref_ids = np.asarray(ref_ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"step {step} nodenumber 存在重复值")
    if np.array_equal(ids, ref_ids):
        return fields, False, 0

    order = np.argsort(ids)
    ids_sorted = ids[order]
    missing_all = np.setdiff1d(ref_ids, ids, assume_unique=False)
    extra_all = np.setdiff1d(ids, ref_ids, assume_unique=False)
    if len(missing_all):
        raise RuntimeError(
            f"step {step} 缺稳定壁面节点: missing={missing_all[:5].tolist()} "
            f"extra={extra_all[:5].tolist()}"
        )
    max_extra = max(1, min(8, int(np.ceil(len(ref_ids) * 1e-4))))
    if len(extra_all) > max_extra:
        raise RuntimeError(
            f"step {step} 额外壁面节点过多: n_extra={len(extra_all)} "
            f"max={max_extra} examples={extra_all[:5].tolist()}"
        )

    pos = np.searchsorted(ids_sorted, ref_ids)
    aligned_idx = order[pos]
    out: Dict[str, np.ndarray] = {}
    for key, val in fields.items():
        arr = np.asarray(val)
        out[key] = arr[aligned_idx] if len(arr) == len(ids) else val
    return out, True, int(len(extra_all))


def preprocess_case(
    cohort_rel: str,
    case_name: str,
    cfg: C.PipelineConfig | None = None,
    verbose: bool = True,
    out_root: str | Path | None = None,
) -> Dict:
    """处理单病例。成功返回 report(dict, status=ok)；原始数据不完整抛 SkipCase。"""
    cfg = cfg or C.DEFAULT
    log = reporting.get_logger()
    case_dir = C.raw_case_dir(cohort_rel, case_name)
    t0 = time.time()
    log.info("[preprocess] %s/%s ...", cohort_rel, case_name)

    # ---- 1. 时间步 + 峰值收缩期 ----
    steps = raw_io.list_timesteps(case_dir)
    if cfg.timestep.step_min is not None:
        steps = [s for s in steps if s >= cfg.timestep.step_min]
    if cfg.timestep.step_max is not None:
        steps = [s for s in steps if s <= cfg.timestep.step_max]

    # ---- 0. 原始文件完整性校验（不完整 -> SkipCase） ----
    probe = _validate_case(case_dir, case_name, steps, cfg)

    peak = raw_io.peak_systole_step(case_dir, steps) if cfg.timestep.peak_from_waveform else steps[len(steps)//2]

    # ---- 几何（静态；首步/峰值步/末步按节点集合共识选参考） ----
    wall_ref_step, ref_wall_fields = _select_stable_wall_reference(
        case_dir, case_name, steps, peak)
    wall_native = np.asarray(ref_wall_fields["coords"], dtype=np.float64)
    ref_node_ids = ref_wall_fields["nodenumber"].astype(np.int64)
    if len(np.unique(ref_node_ids)) != len(ref_node_ids):
        raise RuntimeError(f"参考步 nodenumber 存在重复值: {cohort_rel}/{case_name}")
    int_native = raw_io.read_interior_geometry(case_dir, case_name, wall_ref_step)
    cl = raw_io.read_centerline(case_dir)

    # ---- 单位统一 -> 毫米（逐病例，用中心线包围盒反推，鲁棒处理异常单位病例） ----
    unit_factor, unit_anomaly, unit_extent_mismatch = _resolve_unit_factor(wall_native, cl, cfg.unit)
    if unit_extent_mismatch:
        log.warning("  壁面/中心线覆盖范围不一致 %s/%s: 比值法不可信，已用整十次幂兜底 "
                    "factor=%.4g（中心线可能只描了远端一段），朝向/配准请人工复核",
                    cohort_rel, case_name, unit_factor)
    elif unit_anomaly:
        log.warning("  单位异常 %s/%s: mesh->mm factor=%.4g（正常≈1000），已按中心线校正，"
                    "WSS 物理量级请人工复核", cohort_rel, case_name, unit_factor)
    wall_pts = wall_native * unit_factor
    int_pts = int_native * unit_factor

    # 原始 STL 是 v4 坐标架的权威几何与左右世界方向来源。多 STL 病例按与 CFD 壁面
    # 的原始包围盒匹配度自动选版本；缺失/不可读时保守退回壁面并写入审计。
    anatomy_pts = None
    anatomy_source = "wall_fallback"
    stl_path = ""
    stl_scale_to_mm = np.nan
    stl_match_score = np.nan
    stl_n_candidates = 0
    try:
        selected = surface_io.select_case_surface(case_dir, wall_pts, unit_factor)
        anatomy_pts = selected.points_mm
        anatomy_source = "original_stl"
        stl_path = str(selected.path)
        stl_scale_to_mm = float(selected.scale_to_mm)
        stl_match_score = float(selected.match_score)
        stl_n_candidates = int(selected.n_candidates)
        log.info("  原始 STL 选择 %s/%s: %s scale=%.4g score=%.4f candidates=%d",
                 cohort_rel, case_name, selected.path.name, selected.scale_to_mm,
                 selected.match_score, selected.n_candidates)
    except Exception as exc:  # noqa: BLE001 - AG 历史病例允许壁面回退，新队列由 QA 拒绝
        log.warning("  原始 STL 不可用 %s/%s: %s；本例配准退回 CFD 壁面几何",
                    cohort_rel, case_name, exc)

    # ---- 2. 刚性配准 ----
    reg_cfg = C.registration_for_case(cohort_rel, case_name, cfg.registration)
    T: RigidTransform = compute_transform(
        wall_pts, int_pts, cl, reg_cfg,
        anatomy_pts=anatomy_pts, anatomy_source=anatomy_source,
    )
    cl_effective = dict(cl)
    if T.centerline_translation_applied:
        cl_effective["coords"] = cl["coords"] + T.centerline_translation_mm
        log.warning("  centerline 纯平移修复 %s/%s: offset_frac=%.3f shift_mm=%s "
                    "NN(p50/p90)=%.2f/%.2fmm",
                    cohort_rel, case_name, T.centerline_offset_diag_frac,
                    np.round(T.centerline_translation_mm, 3).tolist(),
                    T.centerline_repair_p50_mm, T.centerline_repair_p90_mm)
    if not T.roll_sign_reliable:
        log.warning("  左右轴符号不可靠 %s/%s: roll_source=%s sign_source=%s |cos|=%.3f，"
                    "已回退世界轴锚，朝向可能左右翻转，建议人工复核",
                    cohort_rel, case_name, T.roll_source, T.roll_sign_source, T.roll_sign_cos)
    if T.trunk_centering_applied:
        log.info("  主干横向二次居中 %s/%s: offset_frac=%.3f offset_mm=%s",
                 cohort_rel, case_name, T.trunk_centering_offset_frac,
                 np.round(T.trunk_centering_offset_mm, 3).tolist())
    wall_aln = T.apply_points(wall_pts)
    int_aln = T.apply_points(int_pts)

    # ---- 2b. 未描主动脉尾巴裁剪（配准后、归一化前；覆盖一致时不裁剪） ----
    cl_aln = T.apply_points(cl_effective["coords"])
    tgt = AXIS_INDEX[T.principal_axis_target]
    wall_keep, int_keep, crop_frac, crop_applied = untraced_inlet_crop_masks(
        wall_aln, int_aln, cl_aln, cl_effective["abscissa"], tgt, reg_cfg)
    if crop_applied:
        log.info("  未描主动脉尾巴裁剪 %s/%s: 裁掉壁面 %.1f%%（%d->%d 点）",
                 cohort_rel, case_name, 100.0 * crop_frac, len(wall_pts), int(wall_keep.sum()))
        wall_pts = wall_pts[wall_keep]
        int_pts = int_pts[int_keep]
        wall_aln = wall_aln[wall_keep]
        int_aln = int_aln[int_keep]

    # ---- 3. 坐标逐病例标准化 ----
    if cfg.normalization.coord_scope == "per_case":
        # 缩放参照：默认只按壁面范围（视角一致，不受内部 CFD 流动延伸段长度影响）
        scale_ref = wall_aln if cfg.normalization.coord_scale_on == "wall" \
            else np.vstack([wall_aln, int_aln])
        if cfg.normalization.coord_method == "std":
            scale = float(scale_ref.std())
        else:  # max_abs -> [-1,1]
            scale = float(np.abs(scale_ref).max())
        scale = scale if scale > 1e-9 else 1.0
    else:
        scale = 1.0  # 全局缩放在 build_samples 阶段处理
    wall_norm = (wall_aln / scale).astype(np.float32)
    int_norm = (int_aln / scale).astype(np.float32)

    # ---- 4. 点分类掩码：内部点到壁面距离 ----
    nn_wall = NearestNeighbors(n_neighbors=1).fit(wall_pts)  # 用原始 mm 距离判定近壁
    d_int, _ = nn_wall.kneighbors(int_pts)
    d_int = d_int[:, 0]
    thr = cfg.tagging.near_wall_threshold_mm
    is_near = d_int <= thr
    # 近壁点上限：按距离最近的优先
    if cfg.tagging.near_wall_max_points > 0 and is_near.sum() > cfg.tagging.near_wall_max_points:
        keep = np.argsort(d_int)[: cfg.tagging.near_wall_max_points]
        mask = np.zeros_like(is_near)
        mask[keep] = True
        is_near = is_near & mask
    int_type = np.where(is_near, NEAR_WALL, INTERIOR).astype(np.int8)

    # ---- 5. 几何特征（完整保留，后续由特征掩码选择） ----
    wall_geom = _nearest_centerline_feats(wall_pts, cl_effective)
    wall_dist = np.zeros(len(wall_pts), dtype=np.float32)  # 壁面点 dist_to_wall=0
    int_geom = _nearest_centerline_feats(int_pts, cl_effective)

    # ---- 6. 堆叠时间步场 ----
    n_wall = len(wall_pts)
    wss_ts = np.empty((len(steps), n_wall), dtype=np.float32)
    wp_ts = np.empty((len(steps), n_wall), dtype=np.float32)
    wvec_ts = np.empty((len(steps), n_wall, 3), dtype=np.float32) if cfg.tagging.store_wall_wss_vector else None
    nodenumber_reordered_steps = []
    nodenumber_extra_dropped_steps = []
    nodenumber_extra_dropped_n = 0
    wall_coord_spatial_remap_steps = []
    wall_coord_mismatch_steps = []
    wall_coord_max_abs_delta = 0.0
    for i, s in enumerate(steps):
        wf = raw_io.read_wall_fields(case_dir, case_name, s)
        wf, reordered, n_extra_dropped = _align_wall_fields_to_reference(
            wf, ref_node_ids, s)
        if reordered:
            nodenumber_reordered_steps.append(int(s))
        if n_extra_dropped:
            nodenumber_extra_dropped_steps.append(int(s))
            nodenumber_extra_dropped_n += int(n_extra_dropped)
        coord_delta = float(np.max(np.abs(wf["coords"] - wall_native))) if len(wall_native) else 0.0
        if coord_delta > 1e-8:
            wf, spatial_remapped, spatial_max_distance = _spatially_align_wall_fields_to_reference(
                wf, wall_native, ref_node_ids, s,
            )
            if spatial_remapped:
                wall_coord_spatial_remap_steps.append(int(s))
                coord_delta = float(np.max(np.abs(wf["coords"] - wall_native)))
                log.warning(
                    "  step %d 少量节点 ID/坐标错配，已按完全一致坐标集合重排 "
                    "max_nn_distance=%.3e", s, spatial_max_distance,
                )
        wall_coord_max_abs_delta = max(wall_coord_max_abs_delta, coord_delta)
        if coord_delta > 1e-8:
            wall_coord_mismatch_steps.append(int(s))
        # 壁面场按裁剪 mask 过滤，和 wall_pts 保持同序同长
        wss_ts[i] = wf["wss"][wall_keep] if crop_applied else wf["wss"]
        wp_ts[i] = wf["pressure"][wall_keep] if crop_applied else wf["pressure"]
        if wvec_ts is not None:
            vec = wf["wss_vec"][wall_keep] if crop_applied else wf["wss_vec"]
            wvec_ts[i] = T.apply_vectors(vec)  # 矢量同步旋转

    # 近壁速度时间序列（第二条路径，可选）
    nw_idx = np.where(int_type == NEAR_WALL)[0]
    nw_vel_ts = nw_vmag_ts = None
    if cfg.tagging.store_near_wall_timeseries and len(nw_idx) > 0:
        nw_vel_ts = np.empty((len(steps), len(nw_idx), 3), dtype=np.float32)
        nw_vmag_ts = np.empty((len(steps), len(nw_idx)), dtype=np.float32)
        for i, s in enumerate(steps):
            f = raw_io.read_interior_fields(case_dir, case_name, s)
            nw_vel_ts[i] = T.apply_vectors(f["vel"][nw_idx])
            nw_vmag_ts[i] = f["vel_mag"][nw_idx]

    # ---- 7. 保存 bundle ----
    out_dir = C.out_case_dir(cohort_rel, case_name, out_root=out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bundle.npz"

    def _vec3_or_nan(value):
        return np.full(3, np.nan, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)

    landmark_trunk = _vec3_or_nan(T.landmark_trunk_point)
    landmark_left = _vec3_or_nan(T.landmark_left_point)
    landmark_right = _vec3_or_nan(T.landmark_right_point)
    centerline_shift = _vec3_or_nan(T.centerline_translation_mm)

    payload = dict(
        # 元信息
        case=case_name, cohort=cohort_rel,
        steps=np.asarray(steps, dtype=np.int32),
        peak_step=np.int32(peak),
        wall_reference_step=np.int32(wall_ref_step),
        coord_scale=np.float32(scale),
        coord_scale_on=np.asarray(cfg.normalization.coord_scale_on),
        unit_factor=np.float64(unit_factor),
        unit_extent_mismatch=np.bool_(unit_extent_mismatch),
        wall_crop_applied=np.bool_(crop_applied),
        wall_crop_frac=np.float64(crop_frac),
        transform_centroid=T.centroid.astype(np.float64),
        transform_rotation=T.rotation.astype(np.float64),
        principal_axis_target=T.principal_axis_target,
        transform_origin_kind=T.origin_kind,
        transform_main_axis_mode=T.main_axis_mode,
        transform_main_axis_source=T.main_axis_source,
        transform_main_axis_wall_sep_delta=np.float64(T.main_axis_wall_sep_delta),
        transform_roll_source=T.roll_source,
        transform_roll_sign_source=T.roll_sign_source,
        transform_roll_sign_cos=np.float64(T.roll_sign_cos),
        transform_roll_sign_reliable=np.bool_(T.roll_sign_reliable),
        transform_trunk_centering_applied=np.bool_(T.trunk_centering_applied),
        transform_trunk_centering_offset_mm=T.trunk_centering_offset_mm.astype(np.float64),
        transform_trunk_centering_offset_frac=np.float64(T.trunk_centering_offset_frac),
        transform_frame_version=np.asarray(T.frame_version),
        transform_landmark_source=np.asarray(T.landmark_source),
        transform_superior_direction=np.asarray(T.superior_direction),
        transform_landmark_trunk_point=landmark_trunk,
        transform_landmark_left_point=landmark_left,
        transform_landmark_right_point=landmark_right,
        transform_fork_spread_ratio=np.float64(T.fork_spread_ratio),
        transform_lr_separation=np.float64(T.lr_separation),
        transform_centerline_translation_applied=np.bool_(T.centerline_translation_applied),
        transform_centerline_translation_candidate=np.bool_(T.centerline_translation_candidate),
        transform_centerline_translation_mm=centerline_shift,
        transform_centerline_offset_diag_frac=np.float64(T.centerline_offset_diag_frac),
        transform_centerline_repair_p50_mm=np.float64(T.centerline_repair_p50_mm),
        transform_centerline_repair_p90_mm=np.float64(T.centerline_repair_p90_mm),
        original_stl_path=np.asarray(stl_path),
        original_stl_scale_to_mm=np.float64(stl_scale_to_mm),
        original_stl_match_score=np.float64(stl_match_score),
        original_stl_n_candidates=np.int32(stl_n_candidates),
        # 壁面（静态几何）
        wall_coords_norm=wall_norm,
        wall_coords_raw=wall_pts.astype(np.float32),
        wall_nodenumber=ref_node_ids[wall_keep].astype(np.int64) if crop_applied else ref_node_ids.astype(np.int64),
        wall_dist_to_wall=wall_dist,
        wall_abscissa_norm=wall_geom["abscissa_norm"].astype(np.float32),
        wall_local_radius=wall_geom["local_radius"].astype(np.float32),
        wall_curvature=wall_geom["curvature"].astype(np.float32),
        # 壁面时间步场
        wall_wss=wss_ts,
        wall_pressure=wp_ts,
    )
    if wvec_ts is not None:
        payload["wall_wss_vec"] = wvec_ts

    if cfg.tagging.store_interior_coords:
        payload.update(
            int_coords_norm=int_norm,
            int_type=int_type,
            int_dist_to_wall=d_int.astype(np.float32),
            int_abscissa_norm=int_geom["abscissa_norm"].astype(np.float32),
            int_local_radius=int_geom["local_radius"].astype(np.float32),
            int_curvature=int_geom["curvature"].astype(np.float32),
        )
    if nw_vel_ts is not None:
        payload.update(near_wall_idx=nw_idx.astype(np.int32),
                       near_wall_vel=nw_vel_ts, near_wall_vel_mag=nw_vmag_ts)

    # 同目录临时文件完成后再原子替换，避免中断留下半个 bundle。
    tmp_path = out_dir / f".bundle.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    bundle_mb = out_path.stat().st_size / 1e6
    rot_det = float(np.linalg.det(T.rotation))
    report = {
        "cohort": cohort_rel, "case": case_name, "status": "ok", "reason": "",
        "n_wall": int(n_wall), "n_interior": int(len(int_pts)),
        "n_near_wall": int(is_near.sum()),
        "near_wall_capped": bool(cfg.tagging.near_wall_max_points > 0
                                 and (d_int <= cfg.tagging.near_wall_threshold_mm).sum()
                                 > cfg.tagging.near_wall_max_points),
        "near_wall_threshold_mm": cfg.tagging.near_wall_threshold_mm,
        "n_steps": len(steps), "step_min": int(steps[0]), "step_max": int(steps[-1]),
        "peak_step": int(peak), "peak_from_waveform": bool(cfg.timestep.peak_from_waveform),
        "unit_factor": round(float(unit_factor), 4), "unit_anomaly": bool(unit_anomaly),
        "unit_extent_mismatch": bool(unit_extent_mismatch),
        "wall_crop_applied": bool(crop_applied),
        "wall_crop_frac": round(float(crop_frac), 4),
        "coord_scale_mm": round(scale, 4), "coord_scope": cfg.normalization.coord_scope,
        "coord_method": cfg.normalization.coord_method,
        "coord_scale_on": cfg.normalization.coord_scale_on,
        "wss_scope": cfg.normalization.wss_scope, "wss_method": cfg.normalization.wss_method,
        "rotation_det": round(rot_det, 6),
        "frame_version": T.frame_version,
        "landmark_source": T.landmark_source,
        "superior_direction": T.superior_direction,
        "fork_spread_ratio": round(float(T.fork_spread_ratio), 4),
        "lr_separation": round(float(T.lr_separation), 4),
        "landmark_trunk_point_mm": [round(float(x), 4) for x in landmark_trunk],
        "landmark_left_point_mm": [round(float(x), 4) for x in landmark_left],
        "landmark_right_point_mm": [round(float(x), 4) for x in landmark_right],
        "centerline_translation_candidate": bool(T.centerline_translation_candidate),
        "centerline_translation_applied": bool(T.centerline_translation_applied),
        "centerline_translation_mm": [round(float(x), 4) for x in centerline_shift],
        "centerline_offset_diag_frac": round(float(T.centerline_offset_diag_frac), 4),
        "centerline_repair_p50_mm": round(float(T.centerline_repair_p50_mm), 4),
        "centerline_repair_p90_mm": round(float(T.centerline_repair_p90_mm), 4),
        "original_stl_path": stl_path,
        "original_stl_scale_to_mm": None if not np.isfinite(stl_scale_to_mm) else float(stl_scale_to_mm),
        "original_stl_match_score": None if not np.isfinite(stl_match_score) else float(stl_match_score),
        "original_stl_n_candidates": stl_n_candidates,
        "principal_axis_target": T.principal_axis_target,
        "origin_kind": T.origin_kind,
        "main_axis_mode": T.main_axis_mode,
        "main_axis_source": T.main_axis_source,
        "main_axis_wall_sep_delta": round(float(T.main_axis_wall_sep_delta), 4),
        "roll_source": T.roll_source,
        "roll_sign_source": T.roll_sign_source,
        "roll_sign_cos": round(float(T.roll_sign_cos), 4),
        "roll_sign_reliable": bool(T.roll_sign_reliable),
        "trunk_centering_applied": bool(T.trunk_centering_applied),
        "trunk_centering_offset_frac": round(float(T.trunk_centering_offset_frac), 4),
        "trunk_centering_offset_mm": [
            round(float(x), 4) for x in T.trunk_centering_offset_mm
        ],
        "centroid_mm": [round(float(x), 4) for x in T.centroid],
        "wss_raw_min": float(wss_ts.min()), "wss_raw_max": float(wss_ts.max()),
        "nodenumber_alignment_ok": True,
        "wall_reference_step": int(wall_ref_step),
        "nodenumber_reordered_n_steps": len(nodenumber_reordered_steps),
        "nodenumber_reordered_steps": nodenumber_reordered_steps[:10],
        "nodenumber_extra_dropped_n": nodenumber_extra_dropped_n,
        "nodenumber_extra_dropped_steps": nodenumber_extra_dropped_steps[:10],
        "wall_coord_spatial_remap_n_steps": len(wall_coord_spatial_remap_steps),
        "wall_coord_spatial_remap_steps": wall_coord_spatial_remap_steps[:10],
        "wall_coord_mismatch_n_steps": len(wall_coord_mismatch_steps),
        "wall_coord_mismatch_steps": wall_coord_mismatch_steps[:10],
        "wall_coord_max_abs_delta": float(wall_coord_max_abs_delta),
        "wall_delimiter": probe["wall_delimiter"],
        "wall_id_column": probe["wall_id_column"],
        "interior_delimiter": probe["interior_delimiter"],
        "bundle_path": str(out_path), "bundle_mb": round(bundle_mb, 2),
        "elapsed_s": round(time.time() - t0, 2),
    }
    reporting.write_case_report(out_dir, report)
    if verbose:
        log.info("  OK wall=%d interior=%d near_wall=%d steps=%d peak=%d scale=%.2fmm "
                 "det=%.4f %.1fMB %.1fs",
                 n_wall, len(int_pts), int(is_near.sum()), len(steps), peak, scale,
                 rot_det, bundle_mb, report["elapsed_s"])
    return report
