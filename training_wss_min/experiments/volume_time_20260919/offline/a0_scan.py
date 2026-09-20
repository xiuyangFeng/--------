"""阶段 0 第一遍扫描（只读 + 写新 sidecar）：体场 81 帧数据合同审计 + 逐例时间矩 + 子采样缓存。

一次遍历同时产出后续 A1（逐帧统计量）、A2（时间基与投影上限）、A3（T-null）需要的全部素材：

1. **合同审计**：81 帧 steps/time 与 bundle 一致；`volume_temporal` 逐帧有限；峰值帧的
   p_rel / 对齐速度与 `volume.npz` 的峰值标签逐点一致（容差 1e-4）；逐帧体积平均压的摆幅。
2. **逐例时间矩** `(M, s, n)`（M = Σ_x y y^T (81,81)）：压力内部 / 压力壁面 / 速度 u,v,w / 速率，
   任意分区的逐帧均值-标准差、时间基 PCA 与投影残差都能由它们精确推出（见 common.gram_of_normalized）。
3. **子采样轨迹缓存**：逐例固定 20000 个内部单元 + 4000 个壁面节点的完整 81 帧轨迹，
   供峰时误差、峰谷幅值、T-null 这类不能由二阶矩推出的诊断使用。
4. **训练用 sidecar** `views_v5_1/wss_min_volume_time_v1/<cid>/sidecar.npz`：壁面行的最近内部单元索引
   （压力混合目标的壁面标签来源，与 volume_view v1.1 同口径）、逐帧体积平均压、steps。

    python -u -m training_wss_min.experiments.volume_time_20260919.offline.a0_scan --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from training_wss_min.experiments.volume_time_20260919.offline import common as K  # noqa: E402

SIDECAR_ROOT = K.ROOT / 'data_wss_v5/views_v5_1/wss_min_volume_time_v1'
SIDECAR_VERSION = 'v5_volume_time_sidecar_v1'
# 峰值帧核对容差：视图把 p − p_ref 存成 float32，绝对压 ~16 kPa 的 float32 舍入约 1e-3 Pa，
# 所以压力按 1e-2 Pa 判（仍比 100–1000 Pa 的空间信号低四个量级）；速度是 O(1) m/s，按 1e-5 判。
TOL_PRESSURE_PA, TOL_VELOCITY_MS = 1.0e-2, 1.0e-5


def moments(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """y (81, N) → (M, s, n)，float64 累加。"""
    y64 = y.astype(np.float64, copy=False)
    return y64 @ y64.T, y64.sum(axis=1), int(y.shape[1])


def scan_case(cid: str) -> dict:
    started = time.time()
    rep: dict = {'canonical_id': cid}
    with np.load(K.volume_npz(cid), allow_pickle=True) as v:
        peak_step = int(v['peak_step'])
        p_ref_peak = float(v['p_ref_pa'])
        vol_p_peak = v['vol_pressure_rel_peak'].astype(np.float64)
        vol_vel_peak = v['vol_velocity_aligned_peak'].astype(np.float64)
        wall_p_peak = v['wall_pressure_rel_peak'].astype(np.float64)
        n_cells = int(v['n_cells'])
    with np.load(K.VIEW / cid / 'bundle.npz', allow_pickle=True) as b:
        steps = b['steps'].astype(np.int64)
        rotation = b['transform_rotation'].astype(np.float64)
        wall_xyz_raw = b['wall_coords_raw'].astype(np.float64)

    with h5py.File(K.h5_path(cid), 'r') as h5:
        h5_steps = h5['wall_temporal/step'][()].astype(np.int64)
        time_s = h5['wall_temporal/time_s'][()].astype(np.float64)
        p_volmean = h5['pressure_reference/p_volume_mean_pa'][()].astype(np.float64)
        xyz = h5['volume_static/xyz_mm'][()].astype(np.float64)
        pres = h5['volume_temporal/pressure_pa'][()]
        vel = h5['volume_temporal/velocity_m_s'][()]

    rep['steps_match_bundle'] = bool(np.array_equal(h5_steps, steps))
    rep['n_frames'] = int(len(h5_steps))
    rep['peak_index'] = int(np.flatnonzero(h5_steps == peak_step)[0])
    rep['n_cells'] = n_cells
    rep['n_cells_match'] = bool(len(xyz) == n_cells)
    rep['dt_uniform_ms'] = float(np.ptp(np.diff(time_s)) * 1e3)
    if not (rep['steps_match_bundle'] and rep['n_cells_match'] and rep['n_frames'] == K.N_FRAMES):
        raise ValueError(f'{cid}: frame/cell layout mismatch {rep}')

    # 逐帧压力参考：p_rel(x,t) = p(x,t) − p_volume_mean(t)
    p_rel = (pres.astype(np.float64) - p_volmean[:, None])
    vel_aligned = vel.astype(np.float64) @ rotation.T
    speed = np.linalg.norm(vel_aligned, axis=2)

    si = rep['peak_index']
    rep['peak_pressure_max_abs_diff'] = float(np.max(np.abs(p_rel[si] - vol_p_peak)))
    rep['peak_velocity_max_abs_diff'] = float(np.max(np.abs(vel_aligned[si] - vol_vel_peak)))
    rep['peak_p_ref_diff'] = float(abs(p_volmean[si] - p_ref_peak))
    rep['finite'] = bool(np.isfinite(p_rel).all() and np.isfinite(vel_aligned).all())
    if not rep['finite']:
        raise ValueError(f'{cid}: non-finite volume_temporal')
    if (rep['peak_pressure_max_abs_diff'] > TOL_PRESSURE_PA
            or rep['peak_velocity_max_abs_diff'] > TOL_VELOCITY_MS):
        raise ValueError(f'{cid}: peak-frame labels disagree with volume.npz {rep}')

    # 壁面行 → 最近内部单元（与 volume_view v1.1 的壁面压力标签同规则）
    nn_dist, nn_idx = cKDTree(xyz).query(wall_xyz_raw)
    wall_p = p_rel[:, nn_idx]
    rep['wall_peak_max_abs_diff'] = float(np.max(np.abs(wall_p[si] - wall_p_peak)))
    if rep['wall_peak_max_abs_diff'] > TOL_PRESSURE_PA:
        raise ValueError(f'{cid}: wall peak pressure disagrees with volume.npz {rep}')
    rep['wall_nearest_cell_dist_p50_p90_mm'] = [float(np.percentile(nn_dist, 50)),
                                                float(np.percentile(nn_dist, 90))]

    # DC 摆幅证据：逐帧体积平均压的周期摆动 vs 逐帧空间相对压的散布
    rep['p_volmean_pa_min_max_ptp'] = [float(p_volmean.min()), float(p_volmean.max()), float(np.ptp(p_volmean))]
    rep['p_rel_spatial_std_peak_median_pa'] = [float(p_rel[si].std()), float(np.median(p_rel.std(axis=1)))]
    rep['speed_peak_p99_cycle_max'] = [float(np.percentile(speed[si], 99)), float(speed.max())]

    chans = {
        'pressure_interior': p_rel,
        'pressure_wall': wall_p,
        'velocity_u': vel_aligned[:, :, 0],
        'velocity_v': vel_aligned[:, :, 1],
        'velocity_w': vel_aligned[:, :, 2],
        'speed': speed,
    }
    payload = {}
    for name, arr in chans.items():
        M, s, n = moments(arr)
        payload[f'{name}__M'] = M
        payload[f'{name}__s'] = s
        payload[f'{name}__n'] = np.int64(n)
    out_dir = K.SCAN / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'moments.tmp.npz', peak_index=np.int64(si), steps=steps, **payload)
    (out_dir / 'moments.tmp.npz').replace(out_dir / 'moments.npz')

    rows = K.subsample_rows(n_cells, K.SUBSAMPLE_CELLS, cid)
    wrows = K.subsample_rows(len(nn_idx), K.SUBSAMPLE_WALL, cid + '::wall')
    np.savez(out_dir / 'sub.tmp.npz', rows=rows.astype(np.int64), wall_rows=wrows.astype(np.int64),
             pressure=p_rel[:, rows].astype(np.float32),
             velocity=vel_aligned[:, rows, :].astype(np.float32),
             wall_pressure=wall_p[:, wrows].astype(np.float32),
             peak_index=np.int64(si))
    (out_dir / 'sub.tmp.npz').replace(out_dir / 'sub.npz')

    side = SIDECAR_ROOT / cid
    side.mkdir(parents=True, exist_ok=True)
    np.savez(side / 'sidecar.tmp.npz',
             sidecar_version=np.asarray(SIDECAR_VERSION),
             canonical_id=np.asarray(cid),
             steps=steps, peak_step=np.int32(peak_step), peak_index=np.int32(si),
             n_cells=np.int64(n_cells),
             wall_nearest_cell_index=nn_idx.astype(np.int32),
             wall_nearest_cell_dist_mm=nn_dist.astype(np.float32),
             p_volume_mean_pa=p_volmean,
             transform_rotation=rotation)
    (side / 'sidecar.tmp.npz').replace(side / 'sidecar.npz')

    rep['seconds'] = time.time() - started
    return rep


def _worker(cid: str) -> dict:
    try:
        return scan_case(cid)
    except Exception as exc:  # noqa: BLE001 - 逐例上报
        return {'canonical_id': cid, 'error': f'{type(exc).__name__}: {exc}'}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--cases', nargs='*', default=None)
    args = ap.parse_args()
    cases = args.cases or K.ALL_CASES
    K.SCAN.mkdir(parents=True, exist_ok=True)
    SIDECAR_ROOT.mkdir(parents=True, exist_ok=True)
    print(f'[a0] {len(cases)} cases, workers={args.workers}', flush=True)
    import multiprocessing as mp
    reports = []
    t0 = time.time()
    with mp.Pool(max(1, args.workers)) as pool:
        for i, rep in enumerate(pool.imap_unordered(_worker, cases)):
            reports.append(rep)
            if 'error' in rep:
                print(f"[a0] {i + 1}/{len(cases)} {rep['canonical_id']} ERROR {rep['error']}", flush=True)
            else:
                print(f"[a0] {i + 1}/{len(cases)} {rep['canonical_id']} n={rep['n_cells']} "
                      f"p_volmean_ptp={rep['p_volmean_pa_min_max_ptp'][2]:.0f}Pa "
                      f"p_rel_sd_peak={rep['p_rel_spatial_std_peak_median_pa'][0]:.0f}Pa "
                      f"{rep['seconds']:.1f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    errors = [r for r in reports if 'error' in r]
    ok = [r for r in reports if 'error' not in r]
    summary = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'sidecar_version': SIDECAR_VERSION,
        'n_cases': len(cases), 'n_errors': len(errors),
        'all_steps_match': bool(all(r['steps_match_bundle'] for r in ok)),
        'all_peak_index_21': bool(all(r['peak_index'] == K.PEAK_INDEX for r in ok)),
        'max_peak_pressure_diff': max((r['peak_pressure_max_abs_diff'] for r in ok), default=None),
        'max_peak_velocity_diff': max((r['peak_velocity_max_abs_diff'] for r in ok), default=None),
        'max_wall_peak_diff': max((r['wall_peak_max_abs_diff'] for r in ok), default=None),
        'p_volmean_ptp_pa': {
            'median': float(np.median([r['p_volmean_pa_min_max_ptp'][2] for r in ok])) if ok else None,
            'min': float(np.min([r['p_volmean_pa_min_max_ptp'][2] for r in ok])) if ok else None,
            'max': float(np.max([r['p_volmean_pa_min_max_ptp'][2] for r in ok])) if ok else None,
        },
        'p_rel_spatial_std_peak_pa_median': float(np.median([r['p_rel_spatial_std_peak_median_pa'][0] for r in ok])) if ok else None,
        'cases': sorted(reports, key=lambda r: r['canonical_id']),
    }
    K.write_json(K.OUT / 'a0_scan.json', summary)
    print(f"[a0] done in {time.time() - t0:.0f}s, errors={len(errors)}", flush=True)
    if errors:
        for r in errors:
            print(f"  {r['canonical_id']}: {r['error']}", flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
