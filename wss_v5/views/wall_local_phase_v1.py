"""解剖锚定的截面内周向相位（部署可得：只用中心线 atlas 与壁面点）。

底座里的 ``theta_rad`` 是在 atlas 的 RMF (frame_n, frame_b) 里测的，**零相位逐例任意**，
跨病人不指向同一解剖方向。本模块把它相对两个解剖方向重新锚定：

``bend_*``  相对**局部曲率平面**：Δθ = θ_point − θ_bend，θ_bend 指向曲率中心（= 弯管内壁）。
            ``bend_cos`` = +1 内壁 / −1 外壁；``bend_kr`` = κ·R 是"这段弯不弯"的信任门，
            直段 κR→0 时相位本身无意义，由 kr 告诉网络不要用。
``apex_*``  相对**分叉平面**：子段近分叉端，指向兄弟子段中心线的方向即流量分隔嵴一侧。
            只在子段、且距分叉 ≤ min(15 mm, 3R) 内有效，其余 ``apex_valid`` = 0 且 sin/cos 置 0。

壁面点到 atlas 行的对应**不重新计算**，直接用 bundle 里 build 时写下的 (segment_id, s_local_mm)
精确回查，避免任何坐标系/单位风险；回查后再用 local_radius 与 radius_mm 交叉校验。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

SCHEMA = "wall_local_phase_v1"
FEATURE_NAMES = ("bend_cos", "bend_sin", "bend_kr",
                 "apex_cos", "apex_sin", "apex_valid", "apex_dist_norm")
APEX_MAX_MM = 15.0
APEX_MAX_RADII = 3.0


@dataclass(frozen=True)
class AtlasView:
    table: np.ndarray
    columns: list[str]
    frame_n: np.ndarray
    frame_b: np.ndarray
    segments: list[dict]

    def col(self, name: str) -> np.ndarray:
        return self.table[:, self.columns.index(name)]

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.col("x_mm"), self.col("y_mm"), self.col("z_mm")], axis=1)

    @property
    def tangent(self) -> np.ndarray:
        t = np.stack([self.col("tangent_x"), self.col("tangent_y"), self.col("tangent_z")], axis=1)
        return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def read_atlas(h5) -> AtlasView:
    g = h5["geometry"]
    return AtlasView(table=np.asarray(g["atlas_table"], dtype=np.float64),
                     columns=list(json.loads(g.attrs["atlas_columns"])),
                     frame_n=np.asarray(g["atlas_frame_n"], dtype=np.float64),
                     frame_b=np.asarray(g["atlas_frame_b"], dtype=np.float64),
                     segments=list(json.loads(g.attrs["atlas_segments"])))


def _smooth_vectors(vec: np.ndarray, half: int) -> np.ndarray:
    """Moving average along the sample axis; the direction, not the magnitude, is what we keep."""
    if half <= 0 or len(vec) < 3:
        return vec
    pad = np.pad(vec, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(vec.shape[1])], axis=1)


def bend_phase(atlas: AtlasView) -> np.ndarray:
    """θ_bend per atlas row, measured in the same (frame_n, frame_b) as the stored theta_rad."""
    seg = atlas.col("segment_id").astype(int)
    s = atlas.col("s_local_mm")
    radius = atlas.col("radius_mm")
    tan = atlas.tangent
    theta_bend = np.zeros(len(seg))
    for sid in np.unique(seg):
        rows = np.flatnonzero(seg == sid)
        rows = rows[np.argsort(s[rows])]
        if len(rows) < 3:
            continue
        t = tan[rows]
        ds = np.gradient(s[rows])
        dt = np.gradient(t, axis=0) / np.maximum(ds, 1e-9)[:, None]
        # 沿 ~1 x 局部半径 的窗平滑方向（与 atlas 自身的半径自适应 SG 策略同量级）
        step = float(np.median(np.diff(s[rows]))) if len(rows) > 1 else 0.5
        half = int(np.clip(round(float(np.median(radius[rows])) / max(step, 1e-6) / 2.0), 2, 30))
        dt = _smooth_vectors(dt, half)
        dt = dt - (np.einsum("ij,ij->i", dt, t))[:, None] * t      # 去掉切向分量
        n, b = atlas.frame_n[rows], atlas.frame_b[rows]
        theta_bend[rows] = np.arctan2(np.einsum("ij,ij->i", dt, b), np.einsum("ij,ij->i", dt, n))
    return theta_bend


def apex_phase(atlas: AtlasView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """θ_apex, validity and normalised junction distance per atlas row (children only)."""
    seg = atlas.col("segment_id").astype(int)
    s = atlas.col("s_local_mm")
    radius = atlas.col("radius_mm")
    xyz, tan = atlas.xyz, atlas.tangent
    parent_of = {int(x["segment_id"]): int(x["parent_id"]) for x in atlas.segments}
    children: dict[int, list[int]] = {}
    for sid, pid in parent_of.items():
        if pid >= 0:
            children.setdefault(pid, []).append(sid)
    theta = np.zeros(len(seg))
    valid = np.zeros(len(seg), dtype=bool)
    dist_norm = np.zeros(len(seg))
    for pid, kids in children.items():
        if len(kids) != 2:      # 只处理规范的二分叉；其余保持 invalid
            continue
        a, c = kids
        for me, sib in ((a, c), (c, a)):
            rows_me = np.flatnonzero(seg == me); rows_me = rows_me[np.argsort(s[rows_me])]
            rows_sib = np.flatnonzero(seg == sib); rows_sib = rows_sib[np.argsort(s[rows_sib])]
            if len(rows_me) < 3 or len(rows_sib) < 3:
                continue
            s_me = s[rows_me] - s[rows_me][0]
            s_sib = s[rows_sib] - s[rows_sib][0]
            # 同弧长处的兄弟中心线点（超出兄弟长度则钳到末端）
            idx = np.clip(np.searchsorted(s_sib, s_me), 0, len(rows_sib) - 1)
            d = xyz[rows_sib][idx] - xyz[rows_me]
            t = tan[rows_me]
            d = d - (np.einsum("ij,ij->i", d, t))[:, None] * t
            norm = np.linalg.norm(d, axis=1)
            n, b = atlas.frame_n[rows_me], atlas.frame_b[rows_me]
            th = np.arctan2(np.einsum("ij,ij->i", d, b), np.einsum("ij,ij->i", d, n))
            limit = np.minimum(APEX_MAX_MM, APEX_MAX_RADII * radius[rows_me])
            ok = (s_me <= limit) & (norm > 1e-6)
            theta[rows_me[ok]] = th[ok]
            valid[rows_me[ok]] = True
            dist_norm[rows_me[ok]] = np.minimum(s_me[ok] / np.maximum(radius[rows_me][ok], 1e-6),
                                                APEX_MAX_RADII)
    return theta, valid, dist_norm


def atlas_rows_for_wall(atlas: AtlasView, wall_segment_id, wall_s_local_mm) -> np.ndarray:
    """Exact (segment_id, s_local_mm) lookup — the view copied both verbatim from map_points."""
    seg = atlas.col("segment_id").astype(np.int64)
    s32 = atlas.col("s_local_mm").astype(np.float32)
    table = {}
    for row in range(len(seg)):
        table.setdefault((int(seg[row]), float(s32[row])), row)
    out = np.empty(len(wall_segment_id), dtype=np.int64)
    for i, (sid, sl) in enumerate(zip(np.asarray(wall_segment_id, dtype=np.int64),
                                      np.asarray(wall_s_local_mm, dtype=np.float32))):
        key = (int(sid), float(sl))
        if key not in table:
            raise KeyError(f"wall point {i} has no exact atlas row for segment {sid} s_local {sl}")
        out[i] = table[key]
    return out


def build(atlas: AtlasView, wall_theta_rad, wall_segment_id, wall_s_local_mm,
          wall_local_radius) -> dict[str, np.ndarray]:
    rows = atlas_rows_for_wall(atlas, wall_segment_id, wall_s_local_mm)
    radius = atlas.col("radius_mm")[rows]
    if not np.allclose(radius, np.asarray(wall_local_radius, dtype=np.float64), rtol=1e-4, atol=1e-4):
        raise ValueError("atlas row lookup disagrees with the bundle's local radius")
    th = np.asarray(wall_theta_rad, dtype=np.float64)
    tb = bend_phase(atlas)[rows]
    kr = atlas.col("curvature_times_radius")[rows]
    ta, va, dn = apex_phase(atlas)
    ta, va, dn = ta[rows], va[rows], dn[rows]
    d_bend = th - tb
    d_apex = th - ta
    out = {
        "bend_cos": np.cos(d_bend), "bend_sin": np.sin(d_bend), "bend_kr": kr,
        "apex_cos": np.where(va, np.cos(d_apex), 0.0), "apex_sin": np.where(va, np.sin(d_apex), 0.0),
        "apex_valid": va.astype(np.float64), "apex_dist_norm": dn,
    }
    for name, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name}: non-finite local phase feature")
    return {k: v.astype(np.float32) for k, v in out.items()}
