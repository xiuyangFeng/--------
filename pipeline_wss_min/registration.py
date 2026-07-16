#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解剖学刚性配准（中心线 + 壁面分支共同约束）。

目标：让每根血管在统一坐标系下、从同一视角看是一致朝向，且完全可复现。

三步：
1. 平移：优先把 VTP `DistToBifurcation≈0` 的分叉区域移到原点。
2. 主轴对齐：默认用中心线入口端 -> 分叉点的 trunk 弦向对齐到 +Z；
   若该弦向不能清楚区分单主干侧/双髂支侧，则用壁面点云 PCA 长轴兜底。
3. 滚转固定：优先从分叉下游壁面点双峰求 L-R 轴，符号由主干弯曲手性与世界轴交叉校验；
   失败时再按 `wall_branches -> branches -> curvature` 逐级回退。

得到的 R 是正交（正交化）且 det=+1 的纯旋转（刚性），
坐标与所有矢量场（速度、WSS 矢量、切线）用同一个 R 同步旋转。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import RegistrationConfig


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class RigidTransform:
    centroid: np.ndarray        # (3,) 世界系重心
    rotation: np.ndarray        # (3,3) 列为新基向量；coords_new = (coords-centroid) @ R
    principal_axis_target: str
    origin_kind: str = "wall"
    main_axis_mode: str = "inlet_to_outlet"
    main_axis_source: str = "centerline_chord"
    main_axis_wall_sep_delta: float = 0.0
    roll_source: str = "curvature"
    roll_sign_source: str = "none"   # 'trunk_bending' | 'world_axis' | 'branches_internal' | 'curvature_internal' | 'none'
    roll_sign_cos: float = 0.0       # 符号置信 |cos|（trunk_bending 模式下有意义）
    roll_sign_reliable: bool = True  # 低于阈值/无参照 -> False，待人工复核
    trunk_centering_applied: bool = False
    trunk_centering_offset_mm: np.ndarray | None = None   # 新坐标系下被减掉的横向平移量
    trunk_centering_offset_frac: float = 0.0
    frame_version: str = "legacy_centerline_v3"
    landmark_source: str = "wall"
    superior_direction: str = "unknown"
    landmark_trunk_point: np.ndarray | None = None
    landmark_left_point: np.ndarray | None = None
    landmark_right_point: np.ndarray | None = None
    fork_spread_ratio: float = 0.0
    lr_separation: float = 0.0
    centerline_translation_applied: bool = False
    centerline_translation_candidate: bool = False
    centerline_translation_mm: np.ndarray | None = None
    centerline_offset_diag_frac: float = 0.0
    centerline_repair_p50_mm: float = 0.0
    centerline_repair_p90_mm: float = 0.0

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.centroid) @ self.rotation

    def apply_vectors(self, vecs: np.ndarray) -> np.ndarray:
        # 矢量只旋转不平移
        return vecs @ self.rotation

    def inverse_points(self, pts_new: np.ndarray) -> np.ndarray:
        return pts_new @ self.rotation.T + self.centroid

    def to_dict(self) -> Dict:
        return {
            "centroid": self.centroid.tolist(),
            "rotation": self.rotation.tolist(),
            "principal_axis_target": self.principal_axis_target,
            "origin_kind": self.origin_kind,
            "main_axis_mode": self.main_axis_mode,
            "main_axis_source": self.main_axis_source,
            "main_axis_wall_sep_delta": self.main_axis_wall_sep_delta,
            "roll_source": self.roll_source,
            "roll_sign_source": self.roll_sign_source,
            "roll_sign_cos": self.roll_sign_cos,
            "roll_sign_reliable": self.roll_sign_reliable,
            "trunk_centering_applied": self.trunk_centering_applied,
            "trunk_centering_offset_mm": (
                None if self.trunk_centering_offset_mm is None
                else self.trunk_centering_offset_mm.tolist()
            ),
            "trunk_centering_offset_frac": self.trunk_centering_offset_frac,
            "frame_version": self.frame_version,
            "landmark_source": self.landmark_source,
            "superior_direction": self.superior_direction,
            "landmark_trunk_point": (
                None if self.landmark_trunk_point is None else self.landmark_trunk_point.tolist()
            ),
            "landmark_left_point": (
                None if self.landmark_left_point is None else self.landmark_left_point.tolist()
            ),
            "landmark_right_point": (
                None if self.landmark_right_point is None else self.landmark_right_point.tolist()
            ),
            "fork_spread_ratio": self.fork_spread_ratio,
            "lr_separation": self.lr_separation,
            "centerline_translation_applied": self.centerline_translation_applied,
            "centerline_translation_candidate": self.centerline_translation_candidate,
            "centerline_translation_mm": (
                None if self.centerline_translation_mm is None
                else self.centerline_translation_mm.tolist()
            ),
            "centerline_offset_diag_frac": self.centerline_offset_diag_frac,
            "centerline_repair_p50_mm": self.centerline_repair_p50_mm,
            "centerline_repair_p90_mm": self.centerline_repair_p90_mm,
        }


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _principal_direction(points_centered: np.ndarray) -> np.ndarray:
    """点集的最大方差方向（PCA 第一主成分）。"""
    # SVD 稳定，避免协方差数值问题
    _, _, vt = np.linalg.svd(points_centered - points_centered.mean(axis=0), full_matrices=False)
    return _unit(vt[0])


def _roll_direction(cl_centered: np.ndarray, main_axis: np.ndarray) -> np.ndarray:
    """中心线在垂直主轴平面内的最大展布方向（弯曲平面方向）。"""
    proj = cl_centered - np.outer(cl_centered @ main_axis, main_axis)  # 去掉主轴分量
    if np.linalg.norm(proj) < 1e-9:
        # 近乎直管：任取一个与主轴正交的稳定向量
        seed = np.array([1.0, 0.0, 0.0]) if abs(main_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return _unit(seed - (seed @ main_axis) * main_axis)
    return _principal_direction(proj)


def _axis_vector(axis_name: str) -> np.ndarray:
    idx = AXIS_INDEX.get(axis_name, 0)
    v = np.zeros(3, dtype=float)
    v[idx] = 1.0
    return v


def _bifurcation_origin(centerline: Dict[str, np.ndarray], cfg: RegistrationConfig):
    d = centerline.get("dist_to_bifurcation")
    coords = centerline["coords"]
    if d is None or len(d) != len(coords):
        return None
    d = np.asarray(d, dtype=float)
    finite = np.isfinite(d)
    if not finite.any():
        return None
    threshold = max(float(cfg.bifurcation_distance_mm), float(np.nanquantile(d[finite], 0.02)))
    mask = finite & (d <= threshold)
    if mask.sum() < 3:
        mask = finite & (d <= np.nanquantile(d[finite], 0.05))
    if mask.sum() == 0:
        return None
    return coords[mask].mean(axis=0)


def _kmeans_points(points: np.ndarray, n_clusters: int, n_iter: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """对分叉附近中心线点执行小规模、确定性的 k-means 聚类。"""
    n = len(points)
    k = max(1, min(int(n_clusters), n))
    centers = [points.mean(axis=0)]
    while len(centers) < k:
        c = np.asarray(centers)
        dist2 = ((points[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        centers.append(points[int(np.argmax(dist2))])
    centers = np.asarray(centers, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist2, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = new_labels == i
            if mask.any():
                new_centers[i] = points[mask].mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return centers, labels


def _flow_divider_origin(centerline: Dict[str, np.ndarray], cfg: RegistrationConfig):
    """三臂分叉原点：主干与左右髂支起点的等权中心。

    VMTK 的 ``DistToBifurcation≈0`` 通常覆盖分支连接处的多个点。直接求均值会
    被某一支的不均匀采样拉偏，因此先把近零点聚成三条解剖臂，再对三臂中心等权
    求均值，使原点对应“左右髂支连接主动脉中部”的直观解剖标志。
    """
    d = centerline.get("dist_to_bifurcation")
    coords = centerline["coords"]
    if d is None or len(d) != len(coords):
        return None
    d = np.asarray(d, dtype=float)
    finite = np.isfinite(d)
    if not finite.any():
        return None

    threshold = max(float(cfg.flow_divider_distance_mm), float(np.nanquantile(d[finite], 0.01)))
    mask = finite & (d <= threshold)
    if mask.sum() < int(cfg.flow_divider_n_clusters):
        mask = finite & (d <= max(float(cfg.bifurcation_distance_mm), float(np.nanquantile(d[finite], 0.03))))
    pts = coords[mask]
    if len(pts) < int(cfg.flow_divider_n_clusters):
        return None

    centers, labels = _kmeans_points(pts, int(cfg.flow_divider_n_clusters))
    counts = np.bincount(labels, minlength=len(centers))
    valid = counts > 0
    centers = centers[valid]
    if len(centers) < int(cfg.flow_divider_n_clusters):
        return None

    sep = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    min_sep = float(sep[np.triu_indices(len(centers), 1)].min())
    if min_sep < float(cfg.flow_divider_min_cluster_sep_mm):
        return None
    return centers.mean(axis=0)


def _endpoint_mean_by_abscissa(cl_centered: np.ndarray, absc: np.ndarray, low: bool) -> np.ndarray:
    order = np.argsort(absc)
    n_end = max(1, int(0.05 * len(cl_centered)))
    idx = order[:n_end] if low else order[-n_end:]
    return cl_centered[idx].mean(axis=0)


def _branch_roll_direction(
    cl_centered: np.ndarray,
    main_axis: np.ndarray,
    centerline: Dict[str, np.ndarray],
    cfg: RegistrationConfig,
):
    d = centerline.get("dist_to_bifurcation")
    if d is None or len(d) != len(cl_centered):
        return None
    d = np.asarray(d, dtype=float)
    z = cl_centered @ main_axis
    finite = np.isfinite(d) & np.isfinite(z)
    downstream = finite & (z > max(1.0, 0.02 * (np.nanmax(z[finite]) - np.nanmin(z[finite]))))
    if downstream.sum() < 8:
        downstream = finite & (z > np.nanmedian(z[finite]))
    if downstream.sum() < 8:
        return None

    q = np.nanquantile(d[downstream], float(cfg.branch_endpoint_quantile))
    mask = downstream & (d >= q)
    if mask.sum() < 6:
        mask = downstream

    pts = cl_centered[mask]
    proj = pts - np.outer(pts @ main_axis, main_axis)
    if np.linalg.norm(proj) < 1e-9:
        return None
    roll = _principal_direction(proj)

    # 符号锚：把稳定世界轴投影到横断面，保证重复运行确定并避免 PCA 任意翻号。
    anchor = _axis_vector(cfg.roll_sign_world_axis)
    anchor = anchor - (anchor @ main_axis) * main_axis
    if np.linalg.norm(anchor) > 1e-9 and np.dot(roll, anchor) < 0:
        roll = -roll
    return roll


def _lr_axis_from_transverse(proj: np.ndarray) -> Tuple[np.ndarray | None, float]:
    """给横断面投影点求左右轴（未定号）与"双峰分离度"。

    分离度 = 两团质心间距 / 平均展布；trunk 单管≈0，两条髂支 splay≈1。返回 (axis, sep)。
    """
    spread = float(np.linalg.norm(proj, axis=1).mean())
    if spread < 1e-6 or len(proj) < 10:
        return None, 0.0
    axis = _principal_direction(proj)                    # 横断面最大展布方向
    t = proj @ axis
    tc = t - t.mean()
    left = proj[tc < 0]
    right = proj[tc >= 0]
    if len(left) < 5 or len(right) < 5:
        return None, 0.0
    conn = right.mean(axis=0) - left.mean(axis=0)        # 两团质心连线
    return _unit(conn), float(np.linalg.norm(conn) / spread)


def _wall_branch_side_candidates(
    wall_centered: np.ndarray,
    main_axis: np.ndarray,
    cfg: RegistrationConfig,
):
    """按某个轴把分叉两侧分别评分，返回每侧的双髂支分离度和 mask。

    正式配准和 QA 图共用这段逻辑，避免出现"配准用两侧自适应、复核图只取单侧"的口径漂移。
    """
    if wall_centered is None or len(wall_centered) < 20:
        return []
    z = wall_centered @ main_axis
    span = float(np.nanmax(z) - np.nanmin(z))
    if not np.isfinite(span) or span <= 1e-6:
        return []
    margin = float(cfg.downstream_wall_axis_frac) * span

    out = []
    for side in (+1.0, -1.0):
        mask = (z * side) > margin
        if mask.sum() < 20:
            out.append({"side": side, "mask": mask, "axis": None, "sep": 0.0})
            continue
        pts = wall_centered[mask]
        proj = pts - np.outer(pts @ main_axis, main_axis)
        axis, sep = _lr_axis_from_transverse(proj)
        out.append({"side": side, "mask": mask, "axis": axis, "sep": float(sep)})
    return out


def _select_wall_branch_side(
    wall_centered: np.ndarray,
    main_axis: np.ndarray,
    cfg: RegistrationConfig,
    min_sep: float = 0.3,
):
    candidates = _wall_branch_side_candidates(wall_centered, main_axis, cfg)
    valid = [c for c in candidates if c["axis"] is not None]
    if not valid:
        return None
    best = max(valid, key=lambda c: c["sep"])
    if best["sep"] < min_sep:
        return None
    return best


def _branch_sep_delta(
    wall_centered: np.ndarray,
    main_axis: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[float, float]:
    """返回 (最大双峰分离度, 两侧分离度差)，用于判断主轴是否能区分主干/髂支侧。"""
    cands = _wall_branch_side_candidates(wall_centered, main_axis, cfg)
    seps = [c["sep"] for c in cands if c["axis"] is not None]
    if not seps:
        return 0.0, 0.0
    if len(seps) == 1:
        return float(seps[0]), float(seps[0])
    seps = sorted(seps, reverse=True)
    return float(seps[0]), float(seps[0] - seps[1])


def _wall_branch_lr_axis(
    wall_centered: np.ndarray,
    main_axis: np.ndarray,
    cfg: RegistrationConfig,
) -> np.ndarray | None:
    """从分叉下游"壁面点"求左右轴（未定号）。

    中心线常只描一条髂支，但壁面/STL 两条髂支都在，且在垂直主轴的横断面里呈双峰。
    分叉在原点、沿主轴分居两侧：一侧是单管主干（trunk），一侧是两条髂支（splay 双峰）。
    中心线 abscissa 方向可能翻转（0 端落在髂支而非入口），故不假设髂支在 +主轴侧，
    而是对 ±两侧各求横断面双峰分离度，取分离度更大且达阈值的那侧为髂支侧。
    """
    best = _select_wall_branch_side(wall_centered, main_axis, cfg, min_sep=0.3)
    if best is None:
        return None
    best_axis = best["axis"]
    return _unit(best_axis - (best_axis @ main_axis) * main_axis)


def _wall_pca_main_axis_fallback(
    wall_centered: np.ndarray,
    chord_axis: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[np.ndarray, str, float]:
    """中心线端点主轴不可信时，用壁面点云长轴兜底。

    只在 chord 不能明显区分"单主干侧/双髂支侧"，而壁面长轴能做到时切换。
    返回 (main_axis, source, chord_sep_delta)。source='wall_pca_fallback' 表示已切换。
    """
    _, chord_delta = _branch_sep_delta(wall_centered, chord_axis, cfg)
    if not cfg.main_axis_wall_fallback:
        return chord_axis, "centerline_chord", chord_delta

    if chord_delta >= float(cfg.main_axis_wall_fallback_sep_delta):
        return chord_axis, "centerline_chord", chord_delta

    wall_axis = _principal_direction(wall_centered)
    best = _select_wall_branch_side(
        wall_centered, wall_axis, cfg,
        min_sep=float(cfg.main_axis_wall_fallback_min_sep),
    )
    if best is None:
        return chord_axis, "centerline_chord_ambiguous", chord_delta

    _, wall_delta = _branch_sep_delta(wall_centered, wall_axis, cfg)
    if wall_delta <= chord_delta:
        return chord_axis, "centerline_chord_ambiguous", chord_delta

    # 壁面 PCA 只提供轴方向，其正负号本身不确定。弯曲主干的横向展开度有时会
    # 超过髂支侧（如 fast/ZHANG_HAO），使侧别评分误判；因此保留中心线“主干→分叉”
    # 的符号，只替换轴的方向估计。
    sign = 1.0 if float(np.dot(wall_axis, chord_axis)) >= 0 else -1.0
    axis = _unit(wall_axis * sign)
    return axis, "wall_pca_fallback", chord_delta


def _trunk_bending_dir(
    cl_centered: np.ndarray,
    main_axis: np.ndarray,
) -> np.ndarray | None:
    """主干（入口->分叉）偏离直弦的弯曲方向，作 A-P 内在手性参照（未定绝对解剖号，但跨病例自洽）。

    分叉原点下，主干点沿主轴投影 z<0（介于入口与分叉之间）。这些点相对主轴直线的横向偏移，
    指示主干朝哪一侧鼓出；用平均偏移锚定符号，得到只依赖主干、恒存在且可复现的参照方向。
    """
    z = cl_centered @ main_axis
    trunk = z < 0
    if trunk.sum() < 5:
        return None
    pts = cl_centered[trunk]
    perp = pts - np.outer(pts @ main_axis, main_axis)
    mean_off = perp.mean(axis=0)
    if np.linalg.norm(mean_off) < 1e-9:
        return None
    d = _principal_direction(perp)
    if np.dot(mean_off, d) < 0:
        d = -d
    return _unit(d)


def _world_axis_sign(
    roll: np.ndarray, main_axis: np.ndarray, cfg: RegistrationConfig
) -> float | None:
    """固定世界轴投影给出的符号（+1/-1），世界系不共享时不可靠；无法判定返回 None。"""
    anchor = _axis_vector(cfg.roll_sign_world_axis)
    anchor = anchor - (anchor @ main_axis) * main_axis
    if np.linalg.norm(anchor) < 1e-9:
        return None
    return 1.0 if float(np.dot(roll, anchor)) >= 0 else -1.0


def _trunk_bending_sign(
    roll: np.ndarray, main_axis: np.ndarray, cl_centered: np.ndarray
) -> Tuple[float | None, float]:
    """主干弯曲(A-P)× 主轴 给出的内在符号(+1/-1)与置信 |cos|；无参照返回 (None, 0)。"""
    ap = _trunk_bending_dir(cl_centered, main_axis)
    if ap is None:
        return None, 0.0
    ref = np.cross(ap, main_axis)
    nref = np.linalg.norm(ref)
    if nref < 1e-9:
        return None, 0.0
    c = float(np.dot(roll, ref / nref))
    return (1.0 if c >= 0 else -1.0), abs(c)


def _sign_lr_axis(
    roll: np.ndarray,
    main_axis: np.ndarray,
    cl_centered: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[np.ndarray, str, float, bool]:
    """给未定号的左右轴定号（两锚交叉校验）。返回 (roll, sign_source, |cos|, reliable)。

    两个独立手性锚：
      - trunk_bending：(主干弯曲 A-P × 主轴)，内在、不依赖世界系；
      - world_axis：固定世界轴投影，网格世界系一致时有效（旧行为）。
    仅当"主锚较弱且两锚给出相反符号"才判不可靠、记审计待复核，避免误报。
    """
    s_world = _world_axis_sign(roll, main_axis, cfg)
    s_bend, cos_bend = _trunk_bending_sign(roll, main_axis, cl_centered)
    strong = cos_bend >= float(cfg.roll_sign_min_cos)
    agree = (s_world is not None) and (s_bend is not None) and (s_world == s_bend)

    if cfg.roll_sign_mode == "trunk_bending" and s_bend is not None:
        if strong or agree or s_world is None:
            # 内在锚强、或两锚一致、或无世界锚可校验 -> 采内在锚，可靠
            sign, source, reliable = s_bend, "trunk_bending", (strong or agree)
        else:
            # 内在锚弱且与世界锚冲突 -> 采世界锚（与多数一致更稳），标记待复核
            sign, source, reliable = s_world, "world_axis_weakbend", False
    elif s_world is not None:
        sign, source = s_world, "world_axis"
        reliable = (s_bend is None) or (s_bend == s_world) or (not strong)
    elif s_bend is not None:
        sign, source, reliable = s_bend, "trunk_bending", strong
    else:
        sign, source, reliable = 1.0, "none", False

    if sign < 0:
        roll = -roll
    return roll, source, cos_bend, bool(reliable)


def _trunk_centering_shift(
    wall_pts: np.ndarray,
    interior_pts: np.ndarray,
    centroid: np.ndarray,
    rotation: np.ndarray,
    target_axis: int,
    cfg: RegistrationConfig,
) -> Tuple[bool, np.ndarray, float]:
    """配准后主干横向二次居中，返回 (是否应用, 新坐标系平移量, 归一化偏移量)。

    主轴已经被放到 target_axis；主干在低 axial 分位。若低位主干段的横向中心明显偏离
    新坐标系中心轴，则将这个横向偏移折算进 centroid，相当于刚性平移，不改变旋转。
    """
    shift_new = np.zeros(3, dtype=float)
    if not getattr(cfg, "trunk_centering", False) or len(wall_pts) < 20:
        return False, shift_new, 0.0

    wall_aligned = (wall_pts - centroid) @ rotation
    int_aligned = (interior_pts - centroid) @ rotation if len(interior_pts) else np.empty((0, 3))
    all_aligned = wall_aligned if len(int_aligned) == 0 else np.vstack([wall_aligned, int_aligned])
    scale = float(np.abs(all_aligned).max()) or 1.0

    axial = wall_aligned[:, target_axis]
    q = np.clip(float(cfg.trunk_centering_quantile), 0.02, 0.45)
    cutoff = float(np.nanquantile(axial, q))
    trunk = wall_aligned[axial <= cutoff]
    if len(trunk) < 10:
        return False, shift_new, 0.0

    transverse = [i for i in range(3) if i != target_axis]
    if getattr(cfg, "trunk_centering_stat", "median") == "mean":
        offset = trunk[:, transverse].mean(axis=0)
    else:
        offset = np.median(trunk[:, transverse], axis=0)

    offset_frac = float(np.linalg.norm(offset) / scale)
    if offset_frac < float(cfg.trunk_centering_min_offset_frac):
        return False, shift_new, offset_frac

    shift_new[transverse] = offset
    return True, shift_new, offset_frac


def _inlet_tangent_crop_reference(
    centerline_aln: np.ndarray,
    abscissa: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float] | None:
    """入口端局部中心线参考：返回 (入口点, 朝向血管内部的切向, 曲线跨度)。"""
    if len(centerline_aln) < 2 or len(abscissa) != len(centerline_aln):
        return None

    absc = np.asarray(abscissa, dtype=float)
    finite = np.isfinite(absc)
    if finite.sum() < 2:
        return None

    order = np.argsort(absc[finite])
    pts = centerline_aln[finite][order]
    s = absc[finite][order]
    span = float(s[-1] - s[0])
    if span <= 1e-9:
        return None

    n = len(pts)
    n0 = max(1, min(n - 1, int(np.ceil(0.05 * n))))
    n1 = max(n0 + 1, min(n, int(np.ceil(0.20 * n))))
    inlet = pts[0]
    near = pts[:n0].mean(axis=0)
    downstream = pts[n0:n1].mean(axis=0) if n1 > n0 else pts[-1]
    tangent = _unit(downstream - near)
    if np.linalg.norm(tangent) <= 1e-9:
        return None
    return inlet, tangent, span


def _superior_inlet_crop_reference(
    centerline_aln: np.ndarray,
    target_axis: int,
) -> Tuple[np.ndarray, np.ndarray, float] | None:
    """v4 不信任 Abscissas 端点号，直接取 +Z 近端主干并估计向管内的局部切向。"""
    if len(centerline_aln) < 5:
        return None
    axial = centerline_aln[:, int(target_axis)]
    lo, hi = float(np.min(axial)), float(np.max(axial))
    span = hi - lo
    if span <= 1e-9:
        return None
    q95 = float(np.quantile(axial, 0.95))
    q70 = float(np.quantile(axial, 0.70))
    q90 = float(np.quantile(axial, 0.90))
    tip_pts = centerline_aln[axial >= q95]
    inner_pts = centerline_aln[(axial >= q70) & (axial <= q90)]
    if len(tip_pts) == 0 or len(inner_pts) == 0:
        return None
    inlet = np.mean(tip_pts, axis=0)
    inner = np.mean(inner_pts, axis=0)
    tangent = _unit(inner - inlet)
    if np.linalg.norm(tangent) <= 1e-9:
        return None
    return inlet, tangent, span


def untraced_inlet_crop_masks(
    wall_aln: np.ndarray,
    interior_aln: np.ndarray,
    centerline_aln: np.ndarray,
    abscissa: np.ndarray,
    target_axis: int,
    cfg: RegistrationConfig,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """配准后（对齐坐标系）裁掉中心线入口端外侧“未描主动脉尾巴”。

    只处理入口(主动脉)侧：优先用中心线入口端局部切平面判断壁面是否延伸到
    中心线覆盖外；若局部切向不可用，再退回主轴方向。对覆盖一致或只擦边超出的
    病例为 no-op。返回 (wall_keep, int_keep, crop_frac, applied)。
    """
    n_wall = len(wall_aln)
    wall_keep = np.ones(n_wall, dtype=bool)
    int_keep = np.ones(len(interior_aln), dtype=bool)
    if not getattr(cfg, "crop_untraced_inlet", False) or n_wall == 0 or len(centerline_aln) < 2:
        return wall_keep, int_keep, 0.0, False

    margin = trigger = None
    if getattr(cfg, "crop_use_inlet_tangent", True):
        if getattr(cfg, "frame_mode", "legacy_centerline") == "stl_landmarks":
            ref = _superior_inlet_crop_reference(centerline_aln, target_axis)
        else:
            ref = _inlet_tangent_crop_reference(centerline_aln, abscissa)
        if ref is not None:
            inlet, tangent, span = ref
            margin = float(cfg.crop_axial_margin_frac) * span
            trigger = float(cfg.crop_trigger_overshoot_frac) * span
            w_signed = (wall_aln - inlet) @ tangent
            overshoot = -float(w_signed.min())
            if overshoot <= trigger:
                return wall_keep, int_keep, 0.0, False
            cut = -margin
            wall_keep = w_signed >= cut
            int_keep = ((interior_aln - inlet) @ tangent) >= cut
            crop_frac = 1.0 - float(wall_keep.mean())
            min_frac = float(getattr(cfg, "crop_min_wall_frac", 0.0))
            if crop_frac < min_frac:
                return np.ones(n_wall, dtype=bool), np.ones(len(interior_aln), dtype=bool), 0.0, False
            if wall_keep.sum() < max(50, int(0.2 * n_wall)):
                return np.ones(n_wall, dtype=bool), np.ones(len(interior_aln), dtype=bool), 0.0, False
            return wall_keep, int_keep, crop_frac, bool(crop_frac > 0.0)

    ax = int(target_axis)
    cl_ax = centerline_aln[:, ax]
    lo, hi = float(cl_ax.min()), float(cl_ax.max())
    span = hi - lo
    if span <= 1e-9:
        return wall_keep, int_keep, 0.0, False

    # 入口端在中心线轴向的哪一侧：由 abscissa 最小点（入口）判定
    inlet_axial = float(centerline_aln[int(np.argmin(abscissa)), ax])
    inlet_low = abs(inlet_axial - lo) <= abs(inlet_axial - hi)

    w_ax = wall_aln[:, ax]
    margin = float(cfg.crop_axial_margin_frac) * span
    trigger = float(cfg.crop_trigger_overshoot_frac) * span

    if inlet_low:
        overshoot = lo - float(w_ax.min())
        if overshoot <= trigger:
            return wall_keep, int_keep, 0.0, False
        cut = lo - margin
        wall_keep = w_ax >= cut
        int_keep = interior_aln[:, ax] >= cut
    else:
        overshoot = float(w_ax.max()) - hi
        if overshoot <= trigger:
            return wall_keep, int_keep, 0.0, False
        cut = hi + margin
        wall_keep = w_ax <= cut
        int_keep = interior_aln[:, ax] <= cut

    crop_frac = 1.0 - float(wall_keep.mean())
    min_frac = float(getattr(cfg, "crop_min_wall_frac", 0.0))
    if crop_frac < min_frac:
        return np.ones(n_wall, dtype=bool), np.ones(len(interior_aln), dtype=bool), 0.0, False
    # 兜底：极端情况下别把壁面裁没了
    if wall_keep.sum() < max(50, int(0.2 * n_wall)):
        return np.ones(n_wall, dtype=bool), np.ones(len(interior_aln), dtype=bool), 0.0, False
    return wall_keep, int_keep, crop_frac, bool(crop_frac > 0.0)


def _bbox_center_diag(points: np.ndarray) -> Tuple[np.ndarray, float]:
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    return (lo + hi) * 0.5, float(np.linalg.norm(hi - lo))


def _nearest_surface_quantiles(
    centerline_pts: np.ndarray,
    surface_pts: np.ndarray,
) -> Tuple[float, float]:
    """中心线到表面的最近距离；表面确定性限流以控制 171 例批处理开销。"""
    from scipy.spatial import cKDTree

    surf = surface_pts
    if len(surf) > 30000:
        idx = np.linspace(0, len(surf) - 1, 30000, dtype=np.int64)
        surf = surf[idx]
    dist, _ = cKDTree(surf).query(centerline_pts, k=1, workers=1)
    return float(np.quantile(dist, 0.50)), float(np.quantile(dist, 0.90))


def _repair_centerline_translation(
    centerline: Dict[str, np.ndarray],
    surface_pts: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """只在“大偏移 + 平移后形状吻合”同时成立时修复 centerline 纯平移错位。"""
    out = dict(centerline)
    cl = np.asarray(centerline["coords"], dtype=float)
    surf_center, surf_diag = _bbox_center_diag(surface_pts)
    cl_center, _ = _bbox_center_diag(cl)
    shift = surf_center - cl_center
    frac = float(np.linalg.norm(shift) / max(surf_diag, 1e-9))
    candidate = bool(
        getattr(cfg, "centerline_translation_repair", False)
        and frac >= float(cfg.centerline_translation_trigger_diag_frac)
    )
    applied = False
    p50 = p90 = 0.0
    if candidate:
        shifted = cl + shift
        p50, p90 = _nearest_surface_quantiles(shifted, surface_pts)
        applied = bool(
            p50 / max(surf_diag, 1e-9) <= float(cfg.centerline_translation_max_p50_diag_frac)
            and p90 / max(surf_diag, 1e-9) <= float(cfg.centerline_translation_max_p90_diag_frac)
        )
        if applied:
            out["coords"] = shifted
    return out, {
        "candidate": candidate,
        "applied": applied,
        "shift": shift if applied else np.zeros(3, dtype=float),
        "offset_frac": frac,
        "p50": p50,
        "p90": p90,
    }


def _transverse_spread(points_centered: np.ndarray, axis: np.ndarray) -> float:
    proj = points_centered - np.outer(points_centered @ axis, axis)
    if len(proj) < 5:
        return 0.0
    direction = _principal_direction(proj)
    t = proj @ direction
    return float(np.quantile(t, 0.95) - np.quantile(t, 0.05))


def _terminal_slices(
    points_centered: np.ndarray,
    axis: np.ndarray,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t = points_centered @ axis
    lo = points_centered[t <= np.quantile(t, q)]
    hi = points_centered[t >= np.quantile(t, 1.0 - q)]
    return lo, hi


def _bbox_center(points: np.ndarray) -> np.ndarray:
    return (np.min(points, axis=0) + np.max(points, axis=0)) * 0.5


def _two_branch_landmarks(
    fork_points_centered: np.ndarray,
    main_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """末端髂支切片聚成两支，返回未定号轴、两支中心与分离置信度。"""
    proj = fork_points_centered - np.outer(fork_points_centered @ main_axis, main_axis)
    centers, labels = _kmeans_points(proj, 2, n_iter=40)
    if len(centers) != 2 or np.any(np.bincount(labels, minlength=2) < 5):
        axis = _principal_direction(proj)
        center = np.mean(fork_points_centered, axis=0)
        return axis, center - axis, center + axis, 0.0

    # 中心取各团原始三维点的包围盒中心，降低 STL 三角片密度不均的影响。
    c0 = _bbox_center(fork_points_centered[labels == 0])
    c1 = _bbox_center(fork_points_centered[labels == 1])
    conn = c1 - c0
    conn = conn - (conn @ main_axis) * main_axis
    axis = _unit(conn)
    within = []
    for i, center in enumerate(centers):
        d = proj[labels == i] - center
        within.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
    separation = float(np.linalg.norm(conn) / max(np.mean(within), 1e-9))
    return axis, c0, c1, separation


def _geometry_bifurcation_origin(
    surface_pts: np.ndarray,
    superior_axis: np.ndarray,
    terminal_q: float,
) -> Tuple[np.ndarray, float]:
    """从 STL 由髂支向主干扫描，找“双管变单管”的首个稳定横断面。

    返回几何分叉中心和扫描置信度（末端双支分离度 / 合流后分离度）。该结果主要
    是 centerline flow-divider 的守卫：当中心线覆盖不完整把原点落到髂支外侧时，
    自动改用 STL 分叉点，保证双髂支严格位于 -Z。
    """
    axis = _unit(superior_axis)
    axial = surface_pts @ axis
    z_lo, z_hi = np.quantile(axial, [0.01, 0.99])
    span = float(z_hi - z_lo)
    if span <= 1e-6:
        return np.mean(surface_pts, axis=0), 0.0

    fork_terminal = surface_pts[axial <= np.quantile(axial, terminal_q)]
    _, _, _, terminal_sep = _two_branch_landmarks(fork_terminal, axis)
    centers = np.linspace(z_lo + 0.08 * span, z_lo + 0.62 * span, 28)
    half_width = 0.032 * span
    rows = []
    for zc in centers:
        slab = surface_pts[np.abs(axial - zc) <= half_width]
        if len(slab) < 30:
            continue
        _, c0, c1, sep = _two_branch_landmarks(slab, axis)
        mid = 0.5 * (c0 + c1)
        mid = mid + (zc - float(mid @ axis)) * axis
        rows.append((float(zc), float(sep), mid))

    if not rows:
        fallback_z = z_lo + 0.28 * span
        mid = np.mean(surface_pts[np.abs(axial - fallback_z) <= 0.06 * span], axis=0)
        return mid, 0.0

    seps = np.asarray([r[1] for r in rows], dtype=float)
    if len(seps) >= 3:
        smooth = np.asarray([
            np.median(seps[max(0, i - 1):min(len(seps), i + 2)])
            for i in range(len(seps))
        ])
    else:
        smooth = seps
    threshold = max(2.2, 0.65 * max(terminal_sep, 1e-9))
    chosen = None
    for i in range(len(rows)):
        stable = smooth[i] <= threshold
        if i + 1 < len(rows):
            stable = stable and smooth[i + 1] <= threshold
        if stable:
            chosen = i
            break
    if chosen is None:
        chosen = int(np.argmin(smooth))
    origin = np.asarray(rows[chosen][2], dtype=float)
    confidence = float(terminal_sep / max(smooth[chosen], 1e-9))
    return origin, confidence


def _compute_stl_landmark_transform(
    wall_pts: np.ndarray,
    interior_pts: np.ndarray,
    centerline: Dict[str, np.ndarray],
    cfg: RegistrationConfig,
    anatomy_pts: np.ndarray | None,
    anatomy_source: str,
) -> RigidTransform:
    """v4 有符号解剖坐标架：分叉原点、主干 +Z、髂支 -Z、STL 世界 +X 定左右。"""
    has_stl = anatomy_pts is not None and len(anatomy_pts) >= 30
    surface = wall_pts if not has_stl else np.asarray(anatomy_pts, dtype=float)
    source = "wall_fallback" if not has_stl else anatomy_source
    # 错位病例中 centerline 与原始 STL 同处 VMTK 局部系，真正保留扫描床平移的是
    # Fluent 壁面。故偏移必须相对 wall 检测；一旦确认纯平移，STL 与 centerline
    # 作为同一解剖刚体整体平移，绝不改变 STL 的原始旋转/左右世界方向。
    cl_fixed, repair = _repair_centerline_translation(centerline, wall_pts, cfg)
    if bool(repair["applied"]) and has_stl:
        # STL 与 centerline 同属局部系但覆盖范围/表面半径不同，二者 bbox 中心并不
        # 必然完全相同；分别做 translation-only bbox 对齐，旋转与尺度仍保持原样。
        wall_box_center, _ = _bbox_center_diag(wall_pts)
        stl_box_center, _ = _bbox_center_diag(surface)
        surface = surface + (wall_box_center - stl_box_center)

    cl_origin = _flow_divider_origin(cl_fixed, cfg)
    cl_origin_kind = "flow_divider"
    if cl_origin is None:
        cl_origin = _bifurcation_origin(cl_fixed, cfg)
        cl_origin_kind = "bifurcation_fallback"

    # 先完全不依赖中心线符号，仅由 STL 两端横向展开度识别髂支端与近端主干端。
    surface_center = _bbox_center(surface)
    geom0 = np.asarray(surface, dtype=float) - surface_center
    axis0 = _principal_direction(geom0)
    q = float(np.clip(cfg.landmark_terminal_quantile, 0.04, 0.25))
    lo, hi = _terminal_slices(geom0, axis0, q)
    if min(len(lo), len(hi)) < int(cfg.landmark_min_side_points):
        raise ValueError("原始 STL 末端点不足，无法建立解剖坐标架")
    lo_spread = _transverse_spread(lo, axis0)
    hi_spread = _transverse_spread(hi, axis0)
    _, _, _, lo_branch_sep = _two_branch_landmarks(lo, axis0)
    _, _, _, hi_branch_sep = _two_branch_landmarks(hi, axis0)
    spread_ratio0 = max(lo_spread, hi_spread) / max(min(lo_spread, hi_spread), 1e-9)
    # 宽度差明显时用宽端；AAA 近端本身也可能很宽，宽度接近时改用“双管/单管”
    # 聚类分离度判端，避免把宽主动脉入口误认成双髂支。
    if spread_ratio0 >= float(cfg.landmark_min_fork_ratio):
        fork_is_hi = hi_spread >= lo_spread
    else:
        fork_is_hi = hi_branch_sep >= lo_branch_sep
    _, trunk0 = (hi, lo) if fork_is_hi else (lo, hi)
    fork_spread = max(lo_spread, hi_spread)
    trunk_spread = min(lo_spread, hi_spread)
    fork_ratio = float(fork_spread / max(trunk_spread, 1e-9))

    main_axis = -axis0 if fork_is_hi else axis0
    geom_origin, geom_origin_conf = _geometry_bifurcation_origin(surface, main_axis, q)
    _, trunk_abs = _terminal_slices(surface, main_axis, q)
    trunk_point = _bbox_center(trunk_abs)
    refined = _unit(trunk_point - geom_origin)
    if np.linalg.norm(refined) > 1e-9:
        main_axis = refined
        geom_origin, geom_origin_conf = _geometry_bifurcation_origin(surface, main_axis, q)

    # 优先保留精确的 VMTK flow-divider；若其不能把“主干/髂支”分到原点两侧，
    # 或与 STL 合流点相距过大，则启用 STL 分叉守卫。
    centroid = geom_origin if cl_origin is None else np.asarray(cl_origin, dtype=float)
    origin_kind = "stl_bifurcation" if cl_origin is None else cl_origin_kind
    _, surface_diag = _bbox_center_diag(surface)
    fork_abs, trunk_abs = _terminal_slices(surface, main_axis, q)
    branch_mid = _bbox_center(fork_abs)
    trunk_point = _bbox_center(trunk_abs)
    cl_valid = bool(
        cl_origin is not None
        and float((trunk_point - centroid) @ main_axis) > 0.0
        and float((branch_mid - centroid) @ main_axis) < 0.0
        and np.linalg.norm(centroid - geom_origin) / max(surface_diag, 1e-9) <= 0.35
    )
    if not cl_valid:
        centroid = geom_origin
        origin_kind = "stl_bifurcation_guard"

    main_axis = _unit(trunk_point - centroid)
    geom = np.asarray(surface, dtype=float) - centroid
    # 用最终原点和已定号主轴重新取末端切片；负端必须为双髂支，正端为单主干。
    fork_pts, trunk_pts = _terminal_slices(geom, main_axis, q)
    neg_spread = _transverse_spread(fork_pts, main_axis)
    pos_spread = _transverse_spread(trunk_pts, main_axis)
    if neg_spread < pos_spread:
        # 初次 PCA 在极端弯曲病例上可能把两端身份判反；以末端展开度作最终守卫。
        main_axis = -main_axis
        fork_pts, trunk_pts = _terminal_slices(geom, main_axis, q)
        neg_spread, pos_spread = pos_spread, neg_spread
        trunk_point = _bbox_center(trunk_pts) + centroid
    else:
        trunk_point = _bbox_center(trunk_pts) + centroid
    fork_ratio = float(neg_spread / max(pos_spread, 1e-9))

    roll, c0, c1, lr_sep = _two_branch_landmarks(fork_pts, main_axis)
    _, _, _, trunk_lr_sep = _two_branch_landmarks(trunk_pts, main_axis)
    fork_ratio = max(
        float(neg_spread / max(pos_spread, 1e-9)),
        float(lr_sep / max(trunk_lr_sep, 1e-9)),
    )
    world_anchor = _axis_vector(getattr(cfg, "lr_world_axis", "x"))
    world_anchor = world_anchor - (world_anchor @ main_axis) * main_axis
    if np.linalg.norm(world_anchor) <= 1e-9:
        world_anchor = _axis_vector("y")
        world_anchor = world_anchor - (world_anchor @ main_axis) * main_axis
    world_anchor = _unit(world_anchor)

    # +X 沿原始 STL 世界 +X；在 DICOM/LPS 来源中即患者左侧。只定旋转，不修改 STL。
    if np.dot(roll, world_anchor) < 0:
        roll = -roll
        c0, c1 = c1, c0
    roll = _unit(roll - (roll @ main_axis) * main_axis)
    left_point = c1 + centroid
    right_point = c0 + centroid

    # 最终硬守卫：两个自动选出的髂支端点必须都在原点下方。仅沿主轴微调原点，
    # 不改变任何旋转或左右关系；同时确保近端主干仍留在 +Z。
    branch_z_max = max(
        float((left_point - centroid) @ main_axis),
        float((right_point - centroid) @ main_axis),
    )
    axial_margin = 0.01 * max(surface_diag, 1e-9)
    shift_axial = branch_z_max + axial_margin
    trunk_z = float((trunk_point - centroid) @ main_axis)
    if shift_axial > 0.0 and trunk_z - shift_axial > axial_margin:
        centroid = centroid + shift_axial * main_axis
        origin_kind = f"{origin_kind}_axial_guard"

    e_x = roll
    e_z = main_axis
    e_y = _unit(np.cross(e_z, e_x))
    e_x = _unit(np.cross(e_y, e_z))
    R = np.column_stack([e_x, e_y, e_z])
    if np.linalg.det(R) < 0:
        e_y = -e_y
        R = np.column_stack([e_x, e_y, e_z])

    fork_ok = fork_ratio >= float(cfg.landmark_min_fork_ratio)
    lr_ok = lr_sep >= float(cfg.landmark_min_lr_sep)
    return RigidTransform(
        centroid=np.asarray(centroid, dtype=float),
        rotation=R,
        principal_axis_target="z",
        origin_kind=origin_kind,
        main_axis_mode="bifurcation_to_proximal_trunk",
        main_axis_source="stl_landmarks" if fork_ok else "stl_landmarks_fork_ambiguous",
        main_axis_wall_sep_delta=float(fork_ratio - 1.0),
        roll_source="stl_branch_endpoints",
        roll_sign_source="stl_world_x",
        roll_sign_cos=abs(float(np.dot(roll, world_anchor))),
        roll_sign_reliable=bool(fork_ok and lr_ok),
        trunk_centering_applied=False,
        trunk_centering_offset_mm=np.zeros(3, dtype=float),
        trunk_centering_offset_frac=0.0,
        frame_version="stl_landmarks_v4",
        landmark_source=source,
        superior_direction="proximal_trunk_+z__iliac_-z",
        landmark_trunk_point=np.asarray(trunk_point, dtype=float),
        landmark_left_point=np.asarray(left_point, dtype=float),
        landmark_right_point=np.asarray(right_point, dtype=float),
        fork_spread_ratio=float(fork_ratio),
        lr_separation=float(lr_sep),
        centerline_translation_applied=bool(repair["applied"]),
        centerline_translation_candidate=bool(repair["candidate"]),
        centerline_translation_mm=np.asarray(repair["shift"], dtype=float),
        centerline_offset_diag_frac=float(repair["offset_frac"]),
        centerline_repair_p50_mm=float(repair["p50"]),
        centerline_repair_p90_mm=float(repair["p90"]),
    )


def compute_transform(
    wall_pts: np.ndarray,
    interior_pts: np.ndarray,
    centerline: Dict[str, np.ndarray],
    cfg: RegistrationConfig,
    anatomy_pts: np.ndarray | None = None,
    anatomy_source: str = "wall",
) -> RigidTransform:
    if getattr(cfg, "frame_mode", "legacy_centerline") == "stl_landmarks":
        return _compute_stl_landmark_transform(
            wall_pts, interior_pts, centerline, cfg, anatomy_pts, anatomy_source)
    # ---- 1. 重心 ----
    origin_kind = cfg.center_on
    if cfg.center_on == "flow_divider":
        centroid = _flow_divider_origin(centerline, cfg)
        if centroid is None:
            centroid = _bifurcation_origin(centerline, cfg)
            origin_kind = "bifurcation_fallback"
        else:
            origin_kind = "flow_divider"
        if centroid is None:
            centroid = wall_pts.mean(axis=0)
            origin_kind = "wall_fallback"
    elif cfg.center_on == "bifurcation":
        centroid = _bifurcation_origin(centerline, cfg)
        if centroid is None:
            centroid = wall_pts.mean(axis=0)
            origin_kind = "wall_fallback"
    elif cfg.center_on == "wall":
        centroid = wall_pts.mean(axis=0)
    elif cfg.center_on == "centerline":
        centroid = centerline["coords"].mean(axis=0)
    else:  # 'all'
        centroid = np.vstack([wall_pts, interior_pts]).mean(axis=0)

    cl = centerline["coords"] - centroid
    absc = centerline["abscissa"]
    wall_centered = wall_pts - centroid

    # ---- 2. 主轴 ----
    # 默认在分叉原点下使用入口端 -> 分叉点的 trunk 方向，避免某一侧髂支
    # 端点把主轴拉偏；缺分叉原点时退回完整中心线首尾弦向。
    inlet = _endpoint_mean_by_abscissa(cl, absc, low=True)
    outlet = _endpoint_mean_by_abscissa(cl, absc, low=False)
    if cfg.main_axis_mode == "inlet_to_bifurcation" and origin_kind in ("bifurcation", "flow_divider", "bifurcation_fallback"):
        chord = -inlet
        main_axis_mode = "inlet_to_bifurcation"
    else:
        chord = outlet - inlet
        main_axis_mode = "inlet_to_outlet"
    if np.linalg.norm(chord) > 1e-6:
        main_axis = _unit(chord)                 # 已按 inlet->outlet 定向
    else:
        main_axis = _principal_direction(cl)     # 退化中心线（inlet≈outlet）时退回 PCA
        if cfg.orient_by_abscissa and np.dot(main_axis, chord) < 0:
            main_axis = -main_axis
    main_axis, main_axis_source, main_axis_wall_sep_delta = _wall_pca_main_axis_fallback(
        wall_centered, main_axis, cfg)

    # ---- 3. 滚转方向（左右轴） ----
    # 新默认：从分叉下游"壁面点"求真正左右轴 + 主干弯曲方向定号；失败按
    # wall_branches -> branches -> curvature 逐级回退（旧行为可通过 roll_source 直接回退）。
    roll = None
    roll_source = "curvature"
    roll_sign_source = "none"
    roll_sign_cos = 0.0
    roll_sign_reliable = True

    if cfg.fix_roll_with_centerline and cfg.roll_source == "wall_branches":
        roll = _wall_branch_lr_axis(wall_centered, main_axis, cfg)
        if roll is not None:
            roll, roll_sign_source, roll_sign_cos, roll_sign_reliable = _sign_lr_axis(
                roll, main_axis, cl, cfg)
            roll_source = "wall_branches"

    if roll is None and cfg.fix_roll_with_centerline and cfg.roll_source in ("wall_branches", "branches"):
        roll = _branch_roll_direction(cl, main_axis, centerline, cfg)  # 旧法：内部已用世界轴定号
        if roll is not None:
            roll_source = "branches"
            roll_sign_source = "branches_internal"

    if roll is None:
        roll = _roll_direction(cl, main_axis)
        roll_source = "curvature"
        # 曲率兜底的符号锚定：让 roll 指向曲率最大处相对入口的偏移方向，确定唯一。
        if cfg.fix_roll_with_centerline:
            curv = centerline.get("curvature")
            if curv is not None and len(curv) == len(cl):
                anchor = cl[int(np.argmax(curv))]
                anchor_perp = anchor - (anchor @ main_axis) * main_axis
                if np.dot(roll, anchor_perp) < 0:
                    roll = -roll
            roll_sign_source = "curvature_internal"

    # 目标基：主轴->target 轴，roll->下一个轴，叉乘补第三个
    tgt = AXIS_INDEX[cfg.principal_axis_target]
    e_axis = main_axis
    e_roll = _unit(roll - (roll @ e_axis) * e_axis)   # 施密特正交化
    e_third = np.cross(e_axis, e_roll)

    # 组装世界系基向量 (e_x_world, e_y_world, e_z_world)，使目标轴 = main_axis
    basis_world = [None, None, None]
    order = [(tgt + 1) % 3, (tgt + 2) % 3]  # 另两个轴按循环顺序
    basis_world[tgt] = e_axis
    basis_world[order[0]] = e_roll
    basis_world[order[1]] = e_third

    R = np.column_stack(basis_world)   # 列 = 新基在世界系的表示
    # 保证右手系 det=+1（刚性旋转，非镜像）
    if np.linalg.det(R) < 0:
        basis_world[order[1]] = -e_third
        R = np.column_stack(basis_world)

    trunk_centering_applied, trunk_centering_offset, trunk_centering_offset_frac = _trunk_centering_shift(
        wall_pts, interior_pts, centroid, R, tgt, cfg)
    if trunk_centering_applied:
        centroid = centroid + trunk_centering_offset @ R.T

    return RigidTransform(
        centroid=centroid,
        rotation=R,
        principal_axis_target=cfg.principal_axis_target,
        origin_kind=origin_kind,
        main_axis_mode=main_axis_mode,
        main_axis_source=main_axis_source,
        main_axis_wall_sep_delta=float(main_axis_wall_sep_delta),
        roll_source=roll_source,
        roll_sign_source=roll_sign_source,
        roll_sign_cos=float(roll_sign_cos),
        roll_sign_reliable=bool(roll_sign_reliable),
        trunk_centering_applied=bool(trunk_centering_applied),
        trunk_centering_offset_mm=trunk_centering_offset.astype(float),
        trunk_centering_offset_frac=float(trunk_centering_offset_frac),
    )
