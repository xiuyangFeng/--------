"""D0：数据合同审计（170 例，只读）+ 重建波形 JSON + 全帧逐帧统计（train136 与 cv3 三折）。"""
import sys, json, hashlib, datetime
import numpy as np, h5py
from common import *

T_PERIOD = 0.8
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
cases = (TRAIN + TEST)[:limit]
scopes = {'train136': set(TRAIN), **{f'fold{f}': set(FOLD_TRAIN[f]) for f in range(3)}}
acc = {k: dict(s1=np.zeros(N_FRAMES), s2=np.zeros(N_FRAMES), n=0, lin1=0.0, lin2=0.0, below=0, n_cases=0, ids=[]) for k in scopes}
rows, ref_q, ref_dq, ref_t, ref_phase = [], None, None, None, None
for i, cid in enumerate(cases):
    d = load_bundle(cid, vec=True)
    with h5py.File(H5 / cid.replace('/', '__') / 'case.h5', 'r') as f:
        step = f['wall_temporal/step'][:]; t_s = f['wall_temporal/time_s'][:]; ph = f['wall_temporal/phase_s'][:]
        q = f['conditions/q_nom_m3s'][:]; dq = f['conditions/dq_nom_dt_m3s2'][:]
    if ref_q is None:
        ref_q, ref_dq, ref_t, ref_phase = q, dq, t_s, ph
    tau = d['tau'].astype(np.float64); vmag = np.linalg.norm(d['vec'].astype(np.float64), axis=2)
    big = tau > FLOOR
    rel = np.abs(vmag[big] - tau[big]) / tau[big]
    row = dict(cid=cid, n=int(tau.shape[1]),
               steps_ok=bool(np.array_equal(d['steps'], step) and np.array_equal(d['steps'], np.arange(1120, 1281, 2))),
               time_ok=bool(np.allclose(t_s - t_s[0], ph - ph[0]) and abs((t_s[-1] - t_s[0]) - T_PERIOD) < 1e-6),
               q_maxdiff=float(np.max(np.abs(q - ref_q))), dq_maxdiff=float(np.max(np.abs(dq - ref_dq))),
               vec_scalar_relerr_max=float(rel.max()), vec_scalar_relerr_p99=float(np.quantile(rel, .99)), vec_scalar_ratio_med=float(np.median(vmag[big] / tau[big])),
               below_floor_frac_allframes=float((tau < FLOOR).mean()), below_floor_frac_peak=float((tau[PEAK_INDEX] < FLOOR).mean()),
               zero_frac=float((tau <= 0).mean()), nonfinite=int((~np.isfinite(tau)).sum()))
    rows.append(row)
    lg = ln_pa(tau)
    for k, members in scopes.items():
        if cid in members:
            a = acc[k]; a['s1'] += lg.sum(axis=1); a['s2'] += (lg ** 2).sum(axis=1); a['n'] += tau.shape[1]
            a['lin1'] += tau.sum(); a['lin2'] += (tau ** 2).sum(); a['below'] += int((tau < FLOOR).sum()); a['n_cases'] += 1; a['ids'].append(cid)
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(cases)}', flush=True)

# 汇总
bad = [r['cid'] for r in rows if not (r['steps_ok'] and r['time_ok']) or r['q_maxdiff'] > 0 or r['nonfinite'] > 0]
vs = np.array([r['vec_scalar_relerr_max'] for r in rows])
print(f'\nD0 审计 {len(rows)} 例：steps/time 不一致 {sum(not (r["steps_ok"] and r["time_ok"]) for r in rows)}；'
      f'波形与首例最大差 q {max(r["q_maxdiff"] for r in rows):.3e} dq {max(r["dq_maxdiff"] for r in rows):.3e}；'
      f'|vec|/标量 比值中位 {np.median([r["vec_scalar_ratio_med"] for r in rows]):.4f}（相对误差 p99 中位 {np.median([r["vec_scalar_relerr_p99"] for r in rows]):.3f}，max 最大 {vs.max():.2e}）；'
      f'floor 以下点比例（全帧）中位 {np.median([r["below_floor_frac_allframes"] for r in rows]):.4f}，峰值帧 {np.median([r["below_floor_frac_peak"] for r in rows]):.4f}')
print('异常病例：', bad if bad else '无')

# 波形 JSON（172 例逐位相同 → 用首例）
if limit is None:
    qn = ref_q / ref_q.max(); tn = (ref_phase - ref_phase[0]) / T_PERIOD
    # 周期中心差分：帧 0 ≡ 帧 80（同相位），避免 UDF 在 mod-T 重置点的导数跳变（帧 80 原值归一后 = 1.0）
    qp = qn[:-1]; dq_c = (np.roll(qp, -1) - np.roll(qp, 1)) / (2 * (tn[1] - tn[0])); dq_c = np.append(dq_c, dq_c[0])
    dqn = dq_c / np.max(np.abs(dq_c)); dqn_udf = ref_dq / np.max(np.abs(ref_dq))
    wf = dict(schema='protocol_inlet_waveform_v51', source='case.h5 conditions/q_nom_m3s (172 cases bitwise identical, audited by D0)',
              period_s=T_PERIOD, steps=[int(s) for s in np.arange(1120, 1281, 2)], time_s=[float(v) for v in ref_t], phase_s=[float(v) for v in ref_phase],
              q_nom_m3s=[float(v) for v in ref_q], dq_nom_dt_m3s2=[float(v) for v in ref_dq],
              q_norm=[float(v) for v in qn], dq_norm=[float(v) for v in dqn], dq_norm_udf=[float(v) for v in dqn_udf],
              dq_norm_definition='periodic central difference of q_norm w.r.t. t_norm, normalised by max|dq|; dq_norm_udf = raw UDF derivative (jumps at the mod-T reset)',
              t_norm=[float(v) for v in tn],
              t_sin=[float(v) for v in np.sin(2 * np.pi * tn)], t_cos=[float(v) for v in np.cos(2 * np.pi * tn)],
              peak_index=int(np.argmax(ref_q)), trough_index=int(np.argmin(ref_q)),
              q_min_over_peak=float(ref_q.min() / ref_q.max()), q_mean_over_peak=float(ref_q.mean() / ref_q.max()))
    write_json(OUT / 'protocol_inlet_waveform_v51.json', wf)
    print(f"波形：峰值帧 {wf['peak_index']}（应为 21）谷底帧 {wf['trough_index']}，Q_min/Q_peak {wf['q_min_over_peak']:.4f}，Q_mean/Q_peak {wf['q_mean_over_peak']:.4f}")
    for k, a in acc.items():
        m = a['s1'] / a['n']; sd = np.sqrt(np.maximum(a['s2'] / a['n'] - m ** 2, 0))
        gm = a['s1'].sum() / (a['n'] * N_FRAMES); gsd = np.sqrt(max(a['s2'].sum() / (a['n'] * N_FRAMES) - gm ** 2, 0))
        lm = a['lin1'] / (a['n'] * N_FRAMES); lsd = np.sqrt(max(a['lin2'] / (a['n'] * N_FRAMES) - lm ** 2, 0))
        stats = dict(schema_version=1, generated_at=datetime.datetime.utcnow().isoformat() + '+00:00', method='log_z', eps=EPS, floor=FLOOR,
                     required_frame_version='v5_atlas_frame_v1', timesteps_scope='all_frames',
                     statistics_scope=f'{k} train partition, valid anatomy wall nodes, all 81 frames',
                     n_cases=a['n_cases'], n_points_per_frame=int(a['n']), below_floor_frac=float(a['below'] / (a['n'] * N_FRAMES)),
                     linear=dict(mean=float(lm), std=float(lsd)), log=dict(mean=float(gm), std=float(gsd)),
                     frame=dict(steps=[int(s) for s in np.arange(1120, 1281, 2)], log_mean=[float(v) for v in m], log_std=[float(v) for v in sd]),
                     train_units=sorted(a['ids']))
        write_json(OUT / f'wss_frame_stats_{k}.json', stats)
        print(f'  stats {k}: {a["n_cases"]} 例, 全帧 log mean {gm:.4f} std {gsd:.4f}; 峰值帧 mean {m[PEAK_INDEX]:.4f} std {sd[PEAK_INDEX]:.4f}; '
              f'谷底帧 mean {m.min():.4f} (帧 {m.argmin()}) ; floor 以下 {stats["below_floor_frac"]:.4f}')
write_json(OUT / 'd0_contract_audit.json', dict(rows=rows, bad=bad))
print('wrote d0_contract_audit.json')
