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
    """Small deterministic k-means for bifurcation near-zero centerline points."""
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
    """Three-arm bifurcation origin: equal-weight center of trunk + two iliac branch starts.

    VMTK's DistToBifurcation≈0 usually marks several points around the branch
    junction. A plain mean can be biased by uneven sampling along one arm. Here
    we cluster the near-zero points into anatomical arms and average the arm
    centers equally, which matches the visual "left/right iliac branches connect
    to the middle aorta" landmark.
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

    # Sign anchor: use a stable world axis projected into the transverse plane.
    # This keeps repeated runs deterministic and avoids arbitrary PCA sign flips.
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

    # Wall PCA is used for the axis direction, but its sign is arbitrary.  The
    # side-score can be fooled when a curved trunk has stronger transverse
    # spread than the iliac side (for example fast/ZHANG_HAO), so preserve the
    # centerline trunk->bifurcation sign and only replace the axis direction.
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


def compute_transform(
    wall_pts: np.ndarray,
    interior_pts: np.ndarray,
    centerline: Dict[str, np.ndarray],
    cfg: RegistrationConfig,
) -> RigidTransform:
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
