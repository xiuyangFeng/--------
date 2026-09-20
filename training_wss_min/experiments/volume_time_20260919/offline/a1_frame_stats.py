"""阶段 0-A1：体场逐帧线性统计量（train136 与 cv3 三折训练侧），产出训练可直接引用的 stats 文件。

输入只有 A0 的逐例时间矩 `scan/<cid>/moments.npz`，不再读原始 81 帧。输出
`volume_stats_<scope>.json`（scope = train136 | fold0 | fold1 | fold2），schema 是现有
`views_v5_1/wss_min_view_v1/volume_stats_train136.json` 的**超集**：

- 峰值帧块 `linear` / `pressure_rel_interior` / `pressure_rel_wall` / `velocity` / `speed`
  与旧文件同定义，峰值帧单帧 run 引用新文件时口径不变；
- 新增 `frame` 块（81 帧）：`linear_mean/std`（压力 wall∪interior）、`velocity_mean/std`（内部逐分量）、
  `speed_mean/std`、以及壁面/内部分开的压力逐帧统计，供 `target_normalization='frame_stats'` 使用。

`--check` 会把 train136 的峰值帧块与视图里的旧文件逐字段比对（float32 视图 vs float64 原始帧的
舍入差应在 1e-3 Pa 量级），差异写进 `a1_frame_stats.json`。

    python -u -m training_wss_min.experiments.volume_time_20260919.offline.a1_frame_stats
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from training_wss_min.experiments.volume_time_20260919.offline import common as K  # noqa: E402

CHANNELS = ('pressure_interior', 'pressure_wall', 'velocity_u', 'velocity_v', 'velocity_w', 'speed')


def load_moments(cid: str) -> dict:
    with np.load(K.SCAN / cid / 'moments.npz', allow_pickle=False) as z:
        out = {name: (z[f'{name}__M'], z[f'{name}__s'], int(z[f'{name}__n'])) for name in CHANNELS}
        out['peak_index'] = int(z['peak_index'])
    return out


def accumulate(cases: list[str], cache: dict) -> dict[str, K.ChannelMoments]:
    acc = {name: K.ChannelMoments() for name in CHANNELS}
    acc['pressure_all'] = K.ChannelMoments()          # 压力 wall∪interior：两块时间矩直接相加
    for cid in cases:
        m = cache[cid]
        for name in CHANNELS:
            acc[name].add(*m[name])
        acc['pressure_all'].add(*m['pressure_interior']).add(*m['pressure_wall'])
    return acc


def stats_payload(scope: str, cases: list[str], acc: dict) -> dict:
    si = K.PEAK_INDEX
    frame = {}
    for tag, name in (('linear', 'pressure_all'), ('pressure_interior', 'pressure_interior'),
                      ('pressure_wall', 'pressure_wall'), ('speed', 'speed')):
        mean, std = acc[name].stats()
        frame[f'{tag}_mean'], frame[f'{tag}_std'] = mean, std
    vm = np.stack([acc[f'velocity_{c}'].stats()[0] for c in K.VELOCITY_COMPONENTS], axis=1)   # (81, 3)
    vs = np.stack([acc[f'velocity_{c}'].stats()[1] for c in K.VELOCITY_COMPONENTS], axis=1)
    frame['velocity_mean'], frame['velocity_std'] = vm, vs
    return {
        'schema_version': 2,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'method': 'linear', 'eps': 0.0,
        'scope': scope, 'n_cases': len(cases),
        'required_frame_version': 'v5_atlas_frame_v1',
        'volume_view_version': 'v5_volume_view_v1.1',
        'timesteps_scope': 'peak (top-level blocks) + all 81 frames (frame block)',
        'peak_step': K.PEAK_STEP, 'peak_index': K.PEAK_INDEX, 'n_frames': K.N_FRAMES,
        'statistics_scope': f'train partition only ({scope}); pressure over wall ∪ interior, velocity/speed over interior',
        'frame_reference': 'p_rel(x, t) = p(x, t) − p_volume_mean(t) per frame',
        'n_interior_points': int(acc['pressure_interior'].n), 'n_wall_points': int(acc['pressure_wall'].n),
        'linear': {'mean': float(frame['linear_mean'][si]), 'std': float(frame['linear_std'][si]),
                   'target': 'pressure_rel_peak (wall ∪ interior), Pa'},
        'pressure_rel_interior': {'mean': float(frame['pressure_interior_mean'][si]),
                                  'std': float(frame['pressure_interior_std'][si])},
        'pressure_rel_wall': {'mean': float(frame['pressure_wall_mean'][si]),
                              'std': float(frame['pressure_wall_std'][si])},
        'velocity': {'mean': vm[si].tolist(), 'std': vs[si].tolist(),
                     'frame': 'atlas anatomical frame (aligned), m/s'},
        'speed': {'mean': float(frame['speed_mean'][si]), 'std': float(frame['speed_std'][si]), 'units': 'm/s'},
        'frame': {k: (v.tolist() if v.ndim == 1 else [row.tolist() for row in v]) for k, v in frame.items()},
        'train_units': cases,
    }


def main() -> int:
    cache = {}
    t0 = time.time()
    for i, cid in enumerate(K.TRAIN):
        cache[cid] = load_moments(cid)
        if (i + 1) % 40 == 0:
            print(f'[a1] loaded {i + 1}/{len(K.TRAIN)} ({time.time() - t0:.0f}s)', flush=True)
    bad = [cid for cid, m in cache.items() if m['peak_index'] != K.PEAK_INDEX]
    if bad:
        raise ValueError(f'peak_index != {K.PEAK_INDEX}: {bad}')

    report = {'scopes': {}}
    for scope, cases in K.SCOPES.items():
        acc = accumulate(cases, cache)
        payload = stats_payload(scope, cases, acc)
        K.write_json(K.OUT / f'volume_stats_{scope}.json', payload)
        f = payload['frame']
        report['scopes'][scope] = {
            'n_cases': len(cases), 'n_interior': payload['n_interior_points'], 'n_wall': payload['n_wall_points'],
            'pressure_peak_mean_std': [payload['linear']['mean'], payload['linear']['std']],
            'pressure_frame_std_min_max': [min(f['linear_std']), max(f['linear_std'])],
            'pressure_frame_std_peak_over_min': max(f['linear_std']) / min(f['linear_std']),
            'speed_frame_std_min_max': [min(f['speed_std']), max(f['speed_std'])],
            'velocity_peak_std': payload['velocity']['std'],
        }
        print(f"[a1] {scope}: n={len(cases)} 压力峰值 mean/std = "
              f"{payload['linear']['mean']:.2f}/{payload['linear']['std']:.2f} Pa; "
              f"逐帧 std {min(f['linear_std']):.1f}–{max(f['linear_std']):.1f} Pa; "
              f"速率峰值 std {payload['speed']['std']:.3f} m/s; "
              f"逐帧速率 std {min(f['speed_std']):.3f}–{max(f['speed_std']):.3f} m/s", flush=True)

    # 与视图里旧的 train136 峰值帧统计对照（float32 视图 vs float64 原始帧）
    old_path = K.VIEW / 'volume_stats_train136.json'
    if old_path.is_file():
        import json
        old = json.loads(old_path.read_text())
        new = K.load_stats('train136')
        diff = {}
        for key in ('linear', 'pressure_rel_interior', 'pressure_rel_wall', 'speed'):
            diff[key] = {k: float(new[key][k]) - float(old[key][k]) for k in ('mean', 'std')}
        diff['velocity'] = {k: (np.asarray(new['velocity'][k]) - np.asarray(old['velocity'][k])).tolist()
                            for k in ('mean', 'std')}
        diff['n_interior_points'] = new['n_interior_points'] - old['n_interior_points']
        diff['n_wall_points'] = new['n_wall_points'] - old['n_wall_points']
        report['train136_vs_view_peak_diff'] = diff
        print('[a1] train136 峰值帧 vs 视图旧文件差：'
              f"linear mean {diff['linear']['mean']:+.2e} std {diff['linear']['std']:+.2e} Pa; "
              f"n_interior {diff['n_interior_points']:+d} n_wall {diff['n_wall_points']:+d}", flush=True)

    K.write_json(K.OUT / 'a1_frame_stats.json', report)
    print(f'[a1] done in {time.time() - t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
