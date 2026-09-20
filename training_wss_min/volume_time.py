#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体场全周期（压力 / 速度 × 81 帧）的共享口径：评估点集与周期指标。

离线阶段 0（`experiments/volume_time_20260919/offline`）与部署侧 `evaluate.py` 都从这里取
子采样规则和指标定义，避免"上限/T-null 用一套点、模型评估用另一套点"这类口径漂移。

**为什么周期指标只在子采样点上算**：体场每例有 3–7×10⁵ 个内部单元，全点云 × 81 帧的推理
对 T0（逐帧输入随时间变化，必须逐帧前向）不可行。因此约定：

- 峰值帧：全点云评估，与历史单帧体场 run（PF6 / VF6）逐字段可比；
- 整周期：每例固定 2 万内部单元 + 4 千壁面节点（按病例名种子固定，T0/TB 各臂共用同一批点），
  在这批点上算逐帧 R²_cb、谷底四分位、时均场、峰时误差等。

目标空间是**线性**的（压力 Pa、速度逐分量 m/s），归一化与去归一化都是仿射变换，
不像 WSS 线要 exp/ln。
"""
from __future__ import annotations

import zlib
from typing import Dict

import numpy as np

N_FRAMES, PEAK_INDEX, PEAK_STEP = 81, 21, 1162
# 周期指标的固定子采样规模与种子（改动会让新旧周期指标不可比，视为口径变更）
SUBSAMPLE_CELLS, SUBSAMPLE_WALL, SUBSAMPLE_SEED = 20000, 4000, 20260919


def subsample_rows(n: int, k: int, tag: str) -> np.ndarray:
    """逐例固定子采样行号（与种子和病例名绑定，任何脚本、任何进程重算都一致）。

    用 crc32 而不是内置 hash：后者带 PYTHONHASHSEED 随机化，跨进程不可复现。
    """
    if n <= k:
        return np.arange(n, dtype=np.int64)
    seed = (SUBSAMPLE_SEED + zlib.crc32(tag.encode("utf-8"))) % (2 ** 32)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def cycle_eval_rows(case: Dict, target: str) -> np.ndarray:
    """周期指标的固定评估行（整例 wall∪interior 行空间内的下标，升序）。

    压力目标同时取壁面与内部行；速度目标只取内部行（壁面是无滑移占位，从来不是监督目标）。
    与离线 A0 缓存的子采样点集**逐点相同**，因此上限 / T-null / 模型可以直接对读。
    """
    unit_id = str(case["unit_id"])
    n_wall = int(case["n_wall"])
    n_vol = len(case["pos"]) - n_wall
    interior = n_wall + subsample_rows(n_vol, SUBSAMPLE_CELLS, unit_id)
    if target == "velocity":
        return interior
    wall = subsample_rows(n_wall, SUBSAMPLE_WALL, unit_id + "::wall")
    return np.concatenate([wall, interior])


def trough_frames(q_norm, k: int = 20) -> np.ndarray:
    """谷底四分位：名义入口流量最低的 k 帧。"""
    return np.sort(np.argsort(np.asarray(q_norm, np.float64))[:k])


class CaseBalanced:
    """病例等权 R²_cb，与 `metrics.casebalanced_field_metrics` 同口径；支持逐帧向量一起累计。"""

    def __init__(self):
        self.mean, self.within, self.mse, self.mae = [], [], [], []

    def add_moments(self, mean, within, mse, mae) -> None:
        self.mean.append(np.asarray(mean, np.float64))
        self.within.append(np.asarray(within, np.float64))
        self.mse.append(np.asarray(mse, np.float64))
        self.mae.append(np.asarray(mae, np.float64))

    def add(self, yt, yp) -> None:
        yt = np.asarray(yt, np.float64)
        yp = np.asarray(yp, np.float64)
        m = yt.mean(axis=-1)
        self.add_moments(m, ((yt - m[..., None]) ** 2).mean(axis=-1),
                         ((yt - yp) ** 2).mean(axis=-1), np.abs(yt - yp).mean(axis=-1))

    def r2(self):
        mean = np.asarray(self.mean)
        within = np.asarray(self.within)
        mse = np.asarray(self.mse)
        g = mean.mean(axis=0)
        var = within + (mean - g) ** 2
        return 1.0 - mse.mean(axis=0) / var.mean(axis=0)

    def mae_mean(self):
        return np.asarray(self.mae).mean(axis=0)


class VolumeTimeMetrics:
    """体场周期指标（线性物理空间）。口径对齐 WSS 线 TimeMetrics 与部署侧体场评估的标量锚：

    - 压力：主标量 = p_rel(Pa)
    - 速度：主标量 = 速率 |u|(m/s)（与 `evaluate._evaluate_velocity` 一致），
      另报三分量合并 R²_cb 与逐点余弦

    逐帧 R²_cb 病例等权；trough = 名义流量最低 20 帧；tmean = 时间平均场
    （压力的时均相对压 / 速度的时均速率，WSS 线 TAWSS 的同位物）。
    """

    def __init__(self, q_norm, kind: str):
        if kind not in ("pressure", "velocity"):
            raise ValueError(kind)
        self.kind = kind
        self.frame, self.tmean = CaseBalanced(), CaseBalanced()
        self.comp = CaseBalanced() if kind == "velocity" else None
        self.peak_med, self.peak_p90, self.ts_corr, self.amp_err, self.cos = [], [], [], [], []
        self.n_cases = 0
        self.trough = trough_frames(q_norm)

    def add(self, y_true, y_pred) -> None:
        yt = np.asarray(y_true, np.float64)
        yp = np.asarray(y_pred, np.float64)
        if self.kind == "velocity":
            self.comp.add(yt.reshape(len(yt), -1), yp.reshape(len(yp), -1))
            st, sp = np.linalg.norm(yt, axis=2), np.linalg.norm(yp, axis=2)
            den = st * sp
            ok = den > 1e-12
            self.cos.append(float(np.mean((yt * yp).sum(axis=2)[ok] / den[ok])) if ok.any() else float("nan"))
        else:
            st, sp = yt, yp
        self.frame.add(st, sp)
        self.tmean.add(st.mean(axis=0), sp.mean(axis=0))
        d = np.abs(st.argmax(axis=0) - sp.argmax(axis=0))
        d = np.minimum(d, N_FRAMES - 1 - d)
        self.peak_med.append(float(np.median(d)))
        self.peak_p90.append(float(np.quantile(d, 0.9)))
        a, b = st - st.mean(axis=0), sp - sp.mean(axis=0)
        den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
        ok = den > 1e-12
        self.ts_corr.append(float(np.mean((a * b).sum(axis=0)[ok] / den[ok])) if ok.any() else float("nan"))
        self.amp_err.append(float(np.median(np.abs((sp.max(0) - sp.min(0)) - (st.max(0) - st.min(0))))))
        self.n_cases += 1

    def summary(self) -> dict:
        r2 = self.frame.r2()
        out = dict(kind=self.kind, n_cases=self.n_cases,
                   cycle_r2cb=float(r2.mean()), peak_r2cb=float(r2[PEAK_INDEX]),
                   trough_r2cb=float(r2[self.trough].mean()),
                   min_frame_r2cb=float(r2.min()), argmin_frame=int(r2.argmin()),
                   tmean_r2cb=float(self.tmean.r2()), tmean_mae=float(self.tmean.mae_mean()),
                   peak_time_err_med_frames=float(np.mean(self.peak_med)),
                   peak_time_err_p90_frames=float(np.mean(self.peak_p90)),
                   ts_corr=float(np.nanmean(self.ts_corr)),
                   amp_err_med=float(np.mean(self.amp_err)),
                   frame_r2cb=[float(v) for v in r2])
        if self.comp is not None:
            rc = self.comp.r2()
            out.update(comp_cycle_r2cb=float(rc.mean()), comp_peak_r2cb=float(rc[PEAK_INDEX]),
                       comp_trough_r2cb=float(rc[self.trough].mean()),
                       cos_mean=float(np.nanmean(self.cos)))
        return out


def fmt_summary(s: dict) -> str:
    u = "Pa" if s["kind"] == "pressure" else "m/s"
    extra = "" if s["kind"] == "pressure" else f" | comp {s['comp_cycle_r2cb']:.4f} | cos {s['cos_mean']:.3f}"
    return (f"cycle R²cb {s['cycle_r2cb']:.4f} | peak {s['peak_r2cb']:.4f} | trough {s['trough_r2cb']:.4f} | "
            f"min {s['min_frame_r2cb']:.4f}@{s['argmin_frame']} | tmean {s['tmean_r2cb']:.4f} "
            f"(MAE {s['tmean_mae']:.3f} {u}) | peak-time med/p90 {s['peak_time_err_med_frames']:.1f}/"
            f"{s['peak_time_err_p90_frames']:.1f} f | ts-corr {s['ts_corr']:.3f} | amp-err {s['amp_err_med']:.3f}{extra}")
