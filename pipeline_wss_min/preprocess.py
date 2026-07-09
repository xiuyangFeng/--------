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

import time
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import config as C
from . import raw_io
from . import reporting
from .registration import (
    compute_transform, RigidTransform, untraced_inlet_crop_masks, AXIS_INDEX,
)


# 点类型编码
WALL, NEAR_WALL, INTERIOR = 2, 1, 0

# 壁面必须列（否则无法提供 WSS 目标）
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
    miss_w = [c for c in _REQUIRED_WALL_COLS if c not in wall_cols]
    if miss_w:
        raise SkipCase(f"壁面导出缺列: {miss_w}")
    miss_i = [c for c in _REQUIRED_INT_COLS if c not in int_cols]
    if miss_i:
        raise SkipCase(f"内部导出缺列: {miss_i}")

    return {
        "wall_delimiter": raw_io.delimiter_of(wall_f),
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
    # ratio_trust 倍则判定壁面/中心线覆盖不一致，比值法不可信 -> 采用整十次幂。
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


def preprocess_case(cohort_rel: str, case_name: str, cfg: C.PipelineConfig | None = None,
                    verbose: bool = True) -> Dict:
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
    ref = steps[0]

    # ---- 几何（静态，只读一次；原生单位） ----
    wall_native = raw_io.read_wall_geometry(case_dir, case_name, ref)
    int_native = raw_io.read_interior_geometry(case_dir, case_name, ref)
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

    # ---- 2. 刚性配准 ----
    reg_cfg = C.registration_for_case(cohort_rel, case_name, cfg.registration)
    T: RigidTransform = compute_transform(wall_pts, int_pts, cl, reg_cfg)
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

    # ---- 2b. 未描主动脉尾巴裁剪（配准后、归一化前；对覆盖一致病例为 no-op） ----
    cl_aln = T.apply_points(cl["coords"])
    tgt = AXIS_INDEX[T.principal_axis_target]
    wall_keep, int_keep, crop_frac, crop_applied = untraced_inlet_crop_masks(
        wall_aln, int_aln, cl_aln, cl["abscissa"], tgt, reg_cfg)
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
        ref = wall_aln if cfg.normalization.coord_scale_on == "wall" \
            else np.vstack([wall_aln, int_aln])
        if cfg.normalization.coord_method == "std":
            scale = float(ref.std())
        else:  # max_abs -> [-1,1]
            scale = float(np.abs(ref).max())
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

    # ---- 5. 几何特征（保留供后续 mask） ----
    wall_geom = _nearest_centerline_feats(wall_pts, cl)
    wall_dist = np.zeros(len(wall_pts), dtype=np.float32)  # 壁面点 dist_to_wall=0
    int_geom = _nearest_centerline_feats(int_pts, cl)

    # ---- 6. 堆叠时间步场 ----
    n_wall = len(wall_pts)
    wss_ts = np.empty((len(steps), n_wall), dtype=np.float32)
    wp_ts = np.empty((len(steps), n_wall), dtype=np.float32)
    wvec_ts = np.empty((len(steps), n_wall, 3), dtype=np.float32) if cfg.tagging.store_wall_wss_vector else None
    for i, s in enumerate(steps):
        wf = raw_io.read_wall_fields(case_dir, case_name, s)
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
    out_dir = C.out_case_dir(cohort_rel, case_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bundle.npz"

    payload = dict(
        # 元信息
        case=case_name, cohort=cohort_rel,
        steps=np.asarray(steps, dtype=np.int32),
        peak_step=np.int32(peak),
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
        # 壁面（静态几何）
        wall_coords_norm=wall_norm,
        wall_coords_raw=wall_pts.astype(np.float32),
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

    np.savez_compressed(out_path, **payload)

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
        "wall_delimiter": probe["wall_delimiter"],
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
