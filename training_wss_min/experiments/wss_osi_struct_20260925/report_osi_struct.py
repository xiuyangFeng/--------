"""任务 2 读数：OSI 目标结构矩阵（V1 平均切向向量 / V2 rev_frac；判据见同目录 PREREG.md）。

每臂 × 折 × OSI 头：留出折 best 同点预测 → OSI 全套（p0_common.osi_summary，滞留区用本臂 TAWSS 通道）、校准后 IoU@0.3
（本臂同一头其余两折留出预测上取 thr_iou）、峰值 / TAWSS / OSI 通道归一化 R²_cb。V1 读两个头：osi_derived（主）与 osi（直接头）。
对照 = 同折 M1r（wss_osi_tail_20260924，同为 node04 A100），另报已存 M1（master 4090）。
    python report_osi_struct.py [best|last]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'wss_cycle_osi_probe_20260924/offline'))
from p0_common import RUNS, FOLD_HELD, M, load_cycle, osi_summary, flat, write_json  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
GRID = np.round(np.arange(0.02, 0.4501, 0.005), 4)
READS = [('M1', 'osi'), ('M1r', 'osi'), ('V1', 'osi_derived'), ('V1', 'osi'), ('V2', 'osi')]
PRIMARY = {('V1', 'osi_derived'): '主读数', ('V1', 'osi'): '次读数（辅助任务效应）', ('V2', 'osi'): '主读数'}
KEYS = ('r2_cb', 'iou_0.3', 'cal_iou_0.3', 'iou_0.2', 'stag_iou', 'ccc_casemed', 'fracerr_0.3', 'pred_mean_true_ge_0.3',
        'norm_r2_wss', 'norm_r2_tawss')


def run_dir(arm, k):
    return {'M1': RUNS / f'wss_cycle_m1_20260921/M1_f{k}_s1234', 'M1r': RUNS / f'wss_osi_tail_20260924/M1r_f{k}_s1234'}.get(
        arm, RUNS / f'wss_osi_struct_20260925/{arm}_f{k}_s1234')


def load(arm, head, k):
    base = run_dir(arm, k) / f'eval/ckpt_{CKPT}/predictions/test'
    out = {}
    for cid in FOLD_HELD[k]:
        c = load_cycle(cid)
        d = {'osi': c['osi'], 'tawss': c['tawss']}
        for key, ch in (('p_osi', head), ('p_tawss', 'tawss'), ('n_wss', 'wss'), ('n_tawss', 'tawss')):
            with np.load(base / ch / cid / 'predictions.npz') as z:
                if not np.array_equal(z['row_index'], np.arange(len(c['osi']))):
                    raise ValueError(f'{arm} f{k} {cid} {ch}: row order')
                if ch in ('osi', 'osi_derived') and not np.array_equal(z['true_pa'].astype(np.float32), c['osi'].astype(np.float32)):
                    raise ValueError(f'{arm} f{k} {cid} {ch}: truth differs from cycle view')
                d[key] = z['pred_pa'].astype(np.float64) if key.startswith('p_') else (z['true_norm'].astype(np.float64), z['pred_norm'].astype(np.float64))
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


def fold_row(data_by_fold, k):
    data, cases = data_by_fold[k], FOLD_HELD[k]
    s = osi_summary(cases, [data[c]['osi'] for c in cases], [data[c]['p_osi'] for c in cases],
                    [data[c]['tawss'] for c in cases], [data[c]['p_tawss'] for c in cases])
    row = flat(s)
    for ch in ('wss', 'tawss'):
        row[f'norm_r2_{ch}'] = M.casebalanced_field_metrics([data[c][f'n_{ch}'][0] for c in cases], [data[c][f'n_{ch}'][1] for c in cases])['r2']
    t = np.concatenate([data[c]['osi'] for c in cases]); p = np.concatenate([data[c]['p_osi'] for c in cases])
    row['pred_mean_true_ge_0.3'] = float(p[t >= 0.3].mean())
    pool = {c: d for j in range(3) if j != k for c, d in data_by_fold[j].items()}
    thr = thr_iou(pool)
    row['cal_iou_0.3'] = osi_summary(cases, [data[c]['osi'] for c in cases], [data[c]['p_osi'] for c in cases], thr_override={0.3: thr})['mask_0.3']['iou']
    row['cal_thr_0.3'] = thr
    return row


def main():
    res = {}
    for arm, head in READS:
        dbf = {k: load(arm, head, k) for k in range(3)}
        res[f'{arm}:{head}'] = {k: fold_row(dbf, k) for k in range(3)}
        print(arm, head, 'loaded', flush=True)
    mean = {name: {key: float(np.mean([r[k][key] for k in range(3)])) for key in KEYS} for name, r in res.items()}
    lines = [f'=== 任务 2 读数：OSI 目标结构矩阵（ckpt_{CKPT}；v5.1 cv3 三折留出均值）===', '',
             f"{'臂:头':18s}" + ''.join(f'{k[:13]:>14s}' for k in KEYS)]
    for name in res:
        lines.append(f'{name:18s}' + ''.join(f'{mean[name][key]:14.4f}' for key in KEYS))
    lines.append('')
    verdicts = {}
    for (arm, head), role in PRIMARY.items():
        name = f'{arm}:{head}'
        for ref in ('M1r', 'M1'):
            rname = f'{ref}:osi'
            d = {key: mean[name][key] - mean[rname][key] for key in KEYS}
            r2_signs = [int(np.sign(res[name][k]['r2_cb'] - res[rname][k]['r2_cb'])) for k in range(3)]
            guards = d['norm_r2_wss'] >= -0.01 and d['norm_r2_tawss'] >= -0.01
            a = d['cal_iou_0.3'] >= 0.02 and d['r2_cb'] >= -0.01
            b = d['r2_cb'] >= 0.02 and all(s > 0 for s in r2_signs) and d['cal_iou_0.3'] >= -0.01
            verdict = 'Go' if guards and (a or b) else ('No-Go（护栏不过）' if (a or b) else 'No-Go')
            verdicts.setdefault(name, {})[ref] = {'role': role, 'reference_role': '主对照' if ref == 'M1r' else '参考', 'delta': d,
                                                  'r2_fold_signs': r2_signs, 'criterion_a': a, 'criterion_b': b, 'guards': guards, 'verdict': verdict}
            lines.append(f"{name}（{role}）Δ vs {ref}（{'主对照' if ref == 'M1r' else '参考'}）：R² {d['r2_cb']:+.4f}（折符号 {r2_signs}）"
                         f" 原始 IoU@0.3 {d['iou_0.3']:+.4f} 校准后 {d['cal_iou_0.3']:+.4f} 滞留区 {d['stag_iou']:+.4f} CCC {d['ccc_casemed']:+.4f}"
                         f" 峰值 {d['norm_r2_wss']:+.4f} TAWSS {d['norm_r2_tawss']:+.4f} → {verdict}")
    print('\n'.join(lines))
    (HERE / f'report_{CKPT}.txt').write_text('\n'.join(lines) + '\n')
    write_json(HERE / f'report_{CKPT}.json', {'per_fold': {n: {str(k): v for k, v in r.items()} for n, r in res.items()},
                                              'fold_mean': mean, 'verdicts': verdicts})


if __name__ == '__main__':
    main()
