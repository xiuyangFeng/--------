"""任务 1 读数：M1cap × v5.2 × CV5（只读已保存预测；口径见同目录 PREREG.md）。可在部分折 / seed 完成时运行（只汇总已完成者）。

- 每 seed：五折留出合并（261 例各一次）的 OSI 全套（p0_common.osi_summary，滞留区用本臂 TAWSS 通道）；校准后 IoU@0.3 =
  本 seed 其余四折留出预测上取 thr_iou 预测阈值后读本折；三 seed 均值 ± sd；三 seed Pa 均值集成（只在三 seed 同折都完成时）。
- 同病例：限 v5.1 train136 的 136 例，对 v5.1 cv3 M1 折外（p0_common.cv_bundle 的 M1）。
- 护栏：逐折 metrics.json 的峰值通道 Pa / 归一化 R²_cb 对同折同 seed X5Dcap；TAWSS 通道一并报。
    python report_m1_v52.py [best|last]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'wss_cycle_osi_probe_20260924/offline'))
from p0_common import RUNS, TRAIN, M, osi_summary, flat, write_json, cv_bundle  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
SEEDS = (1234, 7, 2025)
FOLDS = range(5)
GRID = np.round(np.arange(0.02, 0.4501, 0.005), 4)
KEYS = ('r2_cb', 'ccc_casemed', 'r2_casep10', 'iou_0.1', 'iou_0.2', 'iou_0.3', 'cal_iou_0.3', 'stag_iou', 'fracerr_0.3', 'rel_l2_pooled')


def run(seed, k):
    return RUNS / f'wss_cycle_m1_v52_20260925/M1cap_v52cv_f{k}_s{seed}'


def done(seed, k):
    return (run(seed, k) / f'eval/ckpt_{CKPT}/predictions/test/osi/manifest.json').is_file()


def load_fold(seed, k):
    base = run(seed, k) / f'eval/ckpt_{CKPT}/predictions/test'
    man = json.loads((base / 'osi/manifest.json').read_text())
    out = {}
    for e in man['cases']:
        cid = e['unit_id'] if 'unit_id' in e else str(Path(e['file']).parent)
        d = {}
        for ch in ('osi', 'tawss'):
            with np.load(base / ch / cid / 'predictions.npz') as z:
                d[ch] = z['true_pa'].astype(np.float64); d[f'p_{ch}'] = z['pred_pa'].astype(np.float64)
        out[cid] = d
    return out


def thr_iou(data, T=0.3):
    best, best_t = -1.0, T
    for t in GRID:
        v = [np.count_nonzero((d['osi'] > T) & (d['p_osi'] > t)) / u for d in data.values()
             if (u := np.count_nonzero((d['osi'] > T) | (d['p_osi'] > t)))]
        if np.mean(v) > best:
            best, best_t = float(np.mean(v)), float(t)
    return best_t


def summarize(data, cal_thr=None):
    cases = sorted(data)
    s = osi_summary(cases, [data[c]['osi'] for c in cases], [data[c]['p_osi'] for c in cases],
                    [data[c]['tawss'] for c in cases], [data[c]['p_tawss'] for c in cases])
    row = flat(s)
    if cal_thr is not None:
        vals = []
        for c in cases:
            t = cal_thr[c]
            tm, pm = data[c]['osi'] > 0.3, data[c]['p_osi'] > t
            u = np.count_nonzero(tm | pm)
            if u:
                vals.append(np.count_nonzero(tm & pm) / u)
        row['cal_iou_0.3'] = float(np.mean(vals))
    row['n_cases'] = len(cases)
    row['r2_cb_by_cohort'] = s['r2_cb_by_cohort']
    return row


def guard(seed, k):
    a = json.loads((run(seed, k) / f'eval/ckpt_{CKPT}/metrics.json').read_text())['test']
    b = json.loads((RUNS / f'wss_v52_20260923/X5Dcap_v52cv_f{k}_s{seed}/eval/ckpt_{CKPT}/metrics.json').read_text())['test']
    return {'peak_pa': a['field_casebalanced']['r2'], 'peak_pa_ref': b['field_casebalanced']['r2'],
            'peak_norm': a['normalized']['field_casebalanced']['r2'], 'peak_norm_ref': b['normalized']['field_casebalanced']['r2'],
            'tawss_norm': a['heads']['tawss']['normalized']['field_casebalanced']['r2'], 'tawss_pa': a['heads']['tawss']['field_casebalanced']['r2'],
            'osi_fold_r2': a['heads']['osi']['field_casebalanced']['r2']}


def main():
    folds = {s: {k: load_fold(s, k) for k in FOLDS if done(s, k)} for s in SEEDS}
    lines = [f'=== 任务 1 读数：M1cap × v5.2 × CV5（ckpt_{CKPT}）===', '']
    res = {'per_seed': {}, 'guard': {}, 'library136': {}}
    lines.append(f"{'seed':10s}{'折':>6s}" + ''.join(f'{k[:12]:>13s}' for k in KEYS))
    per_seed_rows = []
    for s, fd in folds.items():
        if not fd:
            continue
        pooled = {c: d for f in fd.values() for c, d in f.items()}
        cal = {}
        for k, f in fd.items():
            others = {c: d for kk, ff in fd.items() if kk != k for c, d in ff.items()}
            t = thr_iou(others) if len(fd) >= 2 else None
            for c in f:
                cal[c] = t if t is not None else 0.3
        row = summarize(pooled, cal if len(fd) >= 2 else None)
        res['per_seed'][s] = {'folds_done': sorted(fd), **row}
        lines.append(f'{s:<10d}{len(fd):>4d}/5' + ''.join(f"{row.get(k, float('nan')):13.4f}" for k in KEYS))
        if len(fd) == 5:
            per_seed_rows.append(row)
        res['guard'][s] = {k: guard(s, k) for k in fd}
        lib = {c: d for c, d in pooled.items() if c in set(TRAIN)}
        res['library136'][s] = summarize(lib, {c: cal[c] for c in lib} if len(fd) >= 2 else None) | {'n': len(lib)}
    if per_seed_rows:
        mean = {k: float(np.mean([r[k] for r in per_seed_rows])) for k in KEYS}
        sd = {k: float(np.std([r[k] for r in per_seed_rows], ddof=1)) if len(per_seed_rows) > 1 else float('nan') for k in KEYS}
        res['seed_mean'], res['seed_sd'], res['n_complete_seeds'] = mean, sd, len(per_seed_rows)
        lines.append(f"{'均值':8s}{len(per_seed_rows):>4d} seed" + ''.join(f'{mean[k]:13.4f}' for k in KEYS))
        lines.append(f"{'sd':10s}{'':>6s}" + ''.join(f'{sd[k]:13.4f}' for k in KEYS))
    # 三 seed 集成（只用三 seed 同折都完成的折）
    common = [k for k in FOLDS if all(k in folds[s] for s in SEEDS)]
    if common:
        ens = {}
        for k in common:
            for c in folds[SEEDS[0]][k]:
                d0 = folds[SEEDS[0]][k][c]
                ens[c] = {'osi': d0['osi'], 'tawss': d0['tawss'],
                          'p_osi': np.clip(np.mean([folds[s][k][c]['p_osi'] for s in SEEDS], axis=0), 0, 0.5),
                          'p_tawss': np.mean([folds[s][k][c]['p_tawss'] for s in SEEDS], axis=0)}
        res['ensemble3'] = summarize(ens) | {'folds': common}
        lines.append(f"{'ens3':10s}{len(common):>4d}/5" + ''.join(f"{res['ensemble3'].get(k, float('nan')):13.4f}" for k in KEYS))
    # 同病例对比：v5.1 cv3 M1 折外（136 例）
    cv = cv_bundle()
    ref = {c: {'osi': d['osi'], 'tawss': d['tawss'], 'p_osi': d['M1'], 'p_tawss': d['M1_tawss']} for c, d in cv.items()}
    ref_cal = {}
    from p0_common import FOLD_HELD
    for k in range(3):
        others = {c: ref[c] for kk in range(3) if kk != k for c in FOLD_HELD[kk]}
        t = thr_iou(others)
        for c in FOLD_HELD[k]:
            ref_cal[c] = t
    res['library136_ref_v51_cv3_M1'] = summarize(ref, ref_cal)
    lines += ['', '[同病例 136 例（v5.1 train136）：v5.2 CV5 折外（训练 206 例）对 v5.1 cv3 折外 M1（训练 90 例）]',
              f"{'':16s}" + ''.join(f'{k[:12]:>13s}' for k in KEYS),
              f"{'v5.1 cv3 M1':16s}" + ''.join(f"{res['library136_ref_v51_cv3_M1'].get(k, float('nan')):13.4f}" for k in KEYS)]
    for s, r in res['library136'].items():
        lines.append(f"{'v5.2 s' + str(s) + ' (' + str(len(folds[s])) + '/5)':16s}" + ''.join(f"{r.get(k, float('nan')):13.4f}" for k in KEYS))
    lines += ['', '[护栏：逐折峰值通道对同折同 seed X5Dcap（Pa / 归一化 R²_cb），TAWSS 通道]']
    for s, g in res['guard'].items():
        for k, v in sorted(g.items()):
            lines.append(f"  s{s} f{k}: 峰值 Pa {v['peak_pa']:.4f}（X5Dcap {v['peak_pa_ref']:.4f}，Δ{v['peak_pa'] - v['peak_pa_ref']:+.4f}）"
                         f" 归一化 {v['peak_norm']:.4f}（{v['peak_norm_ref']:.4f}，Δ{v['peak_norm'] - v['peak_norm_ref']:+.4f}）"
                         f" | TAWSS 归一化 {v['tawss_norm']:.4f} Pa {v['tawss_pa']:.4f} | OSI 折 R² {v['osi_fold_r2']:.4f}")
    allg = [v for g in res['guard'].values() for v in g.values()]
    if allg:
        dn = np.mean([v['peak_norm'] - v['peak_norm_ref'] for v in allg]); dp = np.mean([v['peak_pa'] - v['peak_pa_ref'] for v in allg])
        res['guard_summary'] = {'n_fold_runs': len(allg), 'peak_norm_delta_mean': float(dn), 'peak_pa_delta_mean': float(dp),
                                'peak_norm_drop_le_0.02_all': bool(all(v['peak_norm'] - v['peak_norm_ref'] >= -0.02 for v in allg))}
        lines.append(f"  汇总 {len(allg)} 个折-seed：峰值归一化 Δ均值 {dn:+.4f}、Pa Δ均值 {dp:+.4f}；逐折掉 ≤ 0.02："
                     f"{'全部满足' if res['guard_summary']['peak_norm_drop_le_0.02_all'] else '有不满足'}")
    print('\n'.join(lines))
    (HERE / f'report_{CKPT}.txt').write_text('\n'.join(lines) + '\n')
    write_json(HERE / f'report_{CKPT}.json', res)


if __name__ == '__main__':
    main()
