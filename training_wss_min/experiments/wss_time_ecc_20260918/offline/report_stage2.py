"""阶段 2 汇总：T0 / TB8 / TB16 × 三折（留出折，81 帧）→ 门 G2.1–G2.6。
读取 runs/wss_time_ecc_20260918/<arm>/eval/ckpt_{best,last}/metrics.json、同折底座 X5D_v51_f{k} 峰值帧、D1 上限、D2 逐折 T-null。
用法：report_stage2.py [--ckpt best|last]  → 打印 markdown 表并写 stage2_report_<ckpt>.{md,json}"""
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs'
EXP = ROOT / 'training_wss_min/experiments/wss_time_ecc_20260918'
OFF = EXP / 'offline'
ARMS = ['T0', 'TB8', 'TB16']; FOLDS = [0, 1, 2]
ckpt = sys.argv[sys.argv.index('--ckpt') + 1] if '--ckpt' in sys.argv else 'best'


def load(path):
    return json.loads(Path(path).read_text())['test'] if Path(path).is_file() else None


def arm_metrics(t):
    cyc = t.get('cycle', {})
    return dict(peak_pa=t['field_casebalanced']['r2'], peak_ln=t['normalized']['field_casebalanced']['r2'],
                peak_mae=t['field']['mae'], top10_iou=t['hotspot'].get('top10_iou', np.nan), high_wss_r2=t['regional_field'].get('high_wss', {}).get('r2', np.nan) if isinstance(t.get('regional_field'), dict) else np.nan,
                cycle_pa=cyc.get('cycle_r2cb_pa', np.nan), cycle_ln=cyc.get('cycle_r2cb_ln', np.nan), trough_pa=cyc.get('trough_r2cb_pa', np.nan),
                min_frame=cyc.get('min_frame_r2cb_pa', np.nan), tawss=t['tawss']['field_casebalanced']['r2'], tawss_mae=t['tawss']['field']['mae'],
                pooled_pa=t['pooled_frames']['physical']['field_casebalanced']['r2'],
                pk_med=cyc.get('peak_time_err_med_frames', np.nan), pk_p90=cyc.get('peak_time_err_p90_frames', np.nan),
                ts_corr=cyc.get('ts_corr_ln', np.nan), amp_err=cyc.get('amp_err_ln_med', np.nan))


base = {k: load(RUNS / f'wss_v51_wave2a_20260916/X5D_v51_f{k}_s1234/eval/ckpt_best/metrics.json') for k in FOLDS}
d1 = json.loads((OFF / 'd1_ceiling.json').read_text()); d2 = json.loads((OFF / 'd2_tnull.json').read_text())
rows = {}; missing = []
for arm in ARMS:
    for k in FOLDS:
        t = load(RUNS / f'wss_time_ecc_20260918/{arm}_f{k}_s1234/eval/ckpt_{ckpt}/metrics.json')
        if t is None:
            missing.append(f'{arm}_f{k}'); continue
        m = arm_metrics(t); m['base_peak_pa'] = base[k]['field_casebalanced']['r2'] if base[k] else np.nan
        tn = d2.get(f'fold{k}', {}).get('B_scale')
        m['tnull_cycle'] = tn['cycle_r2cb_pa'] if tn else np.nan; m['tnull_tawss'] = tn['tawss_r2cb'] if tn else np.nan; m['tnull_trough'] = tn['trough_r2cb_pa'] if tn else np.nan
        rows[(arm, k)] = m
print(f'ckpt={ckpt}; 缺失：{missing if missing else "无"}')
keys = ['peak_pa', 'base_peak_pa', 'cycle_pa', 'tnull_cycle', 'trough_pa', 'tnull_trough', 'tawss', 'tnull_tawss', 'pooled_pa', 'cycle_ln', 'peak_ln', 'top10_iou', 'pk_med', 'pk_p90', 'ts_corr', 'amp_err', 'tawss_mae', 'peak_mae']
hdr = '| 臂 | 折 | ' + ' | '.join(keys) + ' |'; lines = [hdr, '|' + '---|' * (len(keys) + 2)]
for (arm, k), m in rows.items():
    lines.append(f'| {arm} | {k} | ' + ' | '.join(f'{m[key]:.4f}' if np.isfinite(m[key]) else 'nan' for key in keys) + ' |')
mean = {}
for arm in ARMS:
    got = [rows[(arm, k)] for k in FOLDS if (arm, k) in rows]
    if len(got) == 3:
        mean[arm] = {key: float(np.mean([g[key] for g in got])) for key in keys}
        lines.append(f'| **{arm}** | 均值 | ' + ' | '.join(f'{mean[arm][key]:.4f}' if np.isfinite(mean[arm][key]) else 'nan' for key in keys) + ' |')
print('\n'.join(lines))
gates = {}
for arm in [a for a in ARMS if a in mean]:
    m = mean[arm]; d = [rows[(arm, k)] for k in FOLDS]
    g = {}
    g['G2.1 vs T-null +0.05'] = (m['cycle_pa'] - m['tnull_cycle'] >= 0.05, all(x['cycle_pa'] > x['tnull_cycle'] for x in d), m['cycle_pa'] - m['tnull_cycle'])
    if 'T0' in mean and arm != 'T0':
        g['G2.2 vs T0 +0.03 (cycle & TAWSS)'] = (m['cycle_pa'] - mean['T0']['cycle_pa'] >= 0.03 and m['tawss'] - mean['T0']['tawss'] >= 0.03,
                                             all(rows[(arm, k)]['cycle_pa'] > rows[('T0', k)]['cycle_pa'] for k in FOLDS), m['cycle_pa'] - mean['T0']['cycle_pa'])
    g['G2.3 peak drop <= 0.02'] = (m['base_peak_pa'] - m['peak_pa'] <= 0.02, all(x['base_peak_pa'] - x['peak_pa'] <= 0.02 for x in d), m['peak_pa'] - m['base_peak_pa'])
    g['G2.4 trough > 0 & >= T-null'] = (m['trough_pa'] > 0 and m['trough_pa'] >= m['tnull_trough'], all(x['trough_pa'] > 0 for x in d), m['trough_pa'] - m['tnull_trough'])
    if arm.startswith('TB'):
        c = d1['fold_mean'][arm[2:]]
        g['G2.5 time-shape <= 1.5x ceiling'] = (m['pk_med'] <= 1.5 * max(c['peak_time_err_med_frames'], 1.0) and m['amp_err'] <= 1.5 * c['amp_err_ln_med'] + 0.1, True, m['pk_med'])
    gates[arm] = {k: dict(passed=bool(v[0]), all_folds_same_sign=bool(v[1]), delta=float(v[2])) for k, v in g.items()}
    print(f'\n{arm}: ' + '; '.join(f"{k}: {'PASS' if v[0] else 'FAIL'} (Δ={v[2]:+.4f}, 三折同向={v[1]})" for k, v in g.items()))
out = EXP / f'stage2_report_{ckpt}.md'; out.write_text('\n'.join(lines) + '\n\n' + json.dumps(gates, indent=1, ensure_ascii=False))
(EXP / f'stage2_report_{ckpt}.json').write_text(json.dumps(dict(rows={f'{a}_f{k}': v for (a, k), v in rows.items()}, mean=mean, gates=gates, missing=missing), indent=1))
print('wrote', out)
