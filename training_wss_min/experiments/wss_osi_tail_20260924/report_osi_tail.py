"""OSI 尾部矩阵读数（只读已保存预测；判据见同目录 PREREG.md）。

每臂 × 折：留出折 best 同点预测（osi / tawss / wss 三通道）→ OSI 全套（p0_common.osi_summary，滞留区用本臂 TAWSS 通道）、
峰值与 TAWSS 通道归一化 R²_cb（护栏）、真值 ≥ 0.3 点的预测均值；校准后 IoU@0.3 = 用本臂其余两折的留出预测在训练折上取
thr_iou 预测阈值后读本折。对照 = 同折 M1r（同硬件）；另报 M1r 对已存 M1（master 4090）的抖动。
    python report_osi_tail.py [best|last]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'wss_cycle_osi_probe_20260924/offline'))
from p0_common import (RUNS, FOLD_HELD, M, load_cycle, osi_summary, flat, write_json)  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
ARMS = ('M1r', 'K1', 'K2', 'K5')
GRID = np.round(np.arange(0.02, 0.4501, 0.005), 4)


def run_dir(arm, k):
    if arm == 'M1':
        return RUNS / f'wss_cycle_m1_20260921/M1_f{k}_s1234'
    return RUNS / f'wss_osi_tail_20260924/{arm}_f{k}_s1234'


def load_arm(arm, k):
    base = run_dir(arm, k) / f'eval/ckpt_{CKPT}/predictions/test'
    out = {}
    for cid in FOLD_HELD[k]:
        c = load_cycle(cid)
        d = {'osi': c['osi'], 'tawss': c['tawss']}
        for ch in ('osi', 'tawss', 'wss'):
            with np.load(base / ch / cid / 'predictions.npz') as z:
                if not np.array_equal(z['row_index'], np.arange(len(c['osi']))):
                    raise ValueError(f'{arm} f{k} {cid} {ch}: row order')
                if ch in ('osi', 'tawss') and not np.array_equal(z['true_pa'].astype(np.float32), c[ch].astype(np.float32)):
                    raise ValueError(f'{arm} f{k} {cid} {ch}: truth differs from cycle view')
                d[f'p_{ch}'] = z['pred_pa'].astype(np.float64)
                d[f'n_{ch}'] = (z['true_norm'].astype(np.float64), z['pred_norm'].astype(np.float64))
        out[cid] = d
    return out


def thr_iou(cases, data, T=0.3):
    best, best_t = -1.0, T
    for t in GRID:
        v = []
        for c in cases:
            tm, pm = data[c]['osi'] > T, data[c]['p_osi'] > t
            u = np.count_nonzero(tm | pm)
            if u:
                v.append(np.count_nonzero(tm & pm) / u)
        m = float(np.mean(v))
        if m > best:
            best, best_t = m, float(t)
    return best_t


def fold_metrics(arm, data_by_fold, k):
    data = data_by_fold[k]
    cases = FOLD_HELD[k]
    s = osi_summary(cases, [data[c]['osi'] for c in cases], [data[c]['p_osi'] for c in cases],
                    [data[c]['tawss'] for c in cases], [data[c]['p_tawss'] for c in cases])
    row = flat(s)
    for ch in ('wss', 'tawss', 'osi'):
        row[f'norm_r2_{ch}'] = M.casebalanced_field_metrics([data[c][f'n_{ch}'][0] for c in cases], [data[c][f'n_{ch}'][1] for c in cases])['r2']
    hi_t = np.concatenate([data[c]['osi'] for c in cases]); hi_p = np.concatenate([data[c]['p_osi'] for c in cases])
    row['pred_mean_true_ge_0.3'] = float(hi_p[hi_t >= 0.3].mean())
    row['pred_gt_0.3_given_true_ge_0.3'] = float((hi_p[hi_t >= 0.3] > 0.3).mean())
    others = [j for j in range(3) if j != k]
    if all(j in data_by_fold for j in others):
        pool = {c: d for j in others for c, d in data_by_fold[j].items()}
        t = thr_iou(list(pool), pool)
        cal = osi_summary(cases, [data[c]['osi'] for c in cases], [data[c]['p_osi'] for c in cases], thr_override={0.3: t})
        row['cal_iou_0.3'] = cal['mask_0.3']['iou']
        row['cal_thr_0.3'] = t
    return row, s


def main():
    arms = [a for a in ('M1',) + ARMS if any((run_dir(a, k) / f'eval/ckpt_{CKPT}/predictions/test/osi').is_dir() for k in range(3))]
    res = {}
    for arm in arms:
        data = {k: load_arm(arm, k) for k in range(3) if (run_dir(arm, k) / f'eval/ckpt_{CKPT}/predictions/test/osi').is_dir()}
        res[arm] = {k: fold_metrics(arm, data, k)[0] for k in data}
        print(arm, 'folds', sorted(data), flush=True)
    keys = ('r2_cb', 'iou_0.3', 'cal_iou_0.3', 'iou_0.2', 'iou_0.1', 'stag_iou', 'ccc_casemed', 'fracerr_0.3', 'pred_mean_true_ge_0.3',
            'pred_gt_0.3_given_true_ge_0.3', 'norm_r2_wss', 'norm_r2_tawss', 'norm_r2_osi')
    lines = [f'=== OSI 尾部矩阵读数（ckpt_{CKPT}；三折留出；对照同折 M1r）===', '',
             f"{'臂':6s}{'折':>4s}" + ''.join(f'{k[:13]:>14s}' for k in keys)]
    summary = {}
    for arm in res:
        folds = sorted(res[arm])
        for k in folds:
            lines.append(f'{arm:6s}{k:>4d}' + ''.join(f"{res[arm][k].get(key, float('nan')):14.4f}" for key in keys))
        if len(folds) == 3:
            mean = {key: float(np.mean([res[arm][k].get(key, np.nan) for k in folds])) for key in keys}
            summary[arm] = mean
            lines.append(f'{arm:6s}{"均值":>3s}' + ''.join(f'{mean[key]:14.4f}' for key in keys))
    lines.append('')
    verdicts = {}
    if 'M1r' in summary and 'M1' in summary:
        jit = {key: summary['M1r'][key] - summary['M1'][key] for key in keys}
        lines.append('抖动 M1r（node04）− 已存 M1（master）：' + ' '.join(f'{k} {v:+.4f}' for k, v in jit.items()
                                                                    if k in ('r2_cb', 'iou_0.3', 'cal_iou_0.3', 'norm_r2_wss', 'norm_r2_tawss')))
        summary['jitter_M1r_minus_M1'] = jit
    # 对照（PREREG 执行变更）：K1 对 M1r（node04 A100）；K2 / K5 对已存 M1（master 4090），另报对 M1r
    PRIMARY_REF = {'K1': 'M1r', 'K2': 'M1', 'K5': 'M1'}
    for arm in ('K1', 'K2', 'K5'):
        for ref_name in dict.fromkeys((PRIMARY_REF[arm], 'M1r', 'M1')):
            if arm not in summary or ref_name not in summary:
                continue
            ref = summary[ref_name]
            d = {key: summary[arm][key] - ref[key] for key in keys}
            signs = [np.sign(res[arm][k]['iou_0.3'] - res[ref_name][k]['iou_0.3']) for k in range(3)]
            go_raw = d['iou_0.3'] >= 0.03 and all(s > 0 for s in signs)
            go_cal = d['cal_iou_0.3'] >= 0.02
            guards = d['r2_cb'] >= -0.01 and d['fracerr_0.3'] <= 0.02 and d['norm_r2_wss'] >= -0.01 and d['norm_r2_tawss'] >= -0.01
            verdict = ('Go（三 seed + test34 确认）' if go_raw and go_cal and guards else
                       '只改标定' if go_raw and guards and not go_cal else
                       'No-Go（护栏不过）' if go_raw and not guards else 'No-Go')
            tag = '主对照' if ref_name == PRIMARY_REF[arm] else '参考'
            verdicts.setdefault(arm, {})[ref_name] = {'role': tag, 'delta': d, 'iou_0.3_fold_signs': [int(s) for s in signs], 'go_raw': go_raw,
                                                      'go_cal': go_cal, 'guards': guards, 'verdict': verdict}
            lines.append(f"{arm} Δ vs {ref_name}（{tag}）：IoU@0.3 {d['iou_0.3']:+.4f}（折符号 {[int(s) for s in signs]}）校准后 {d['cal_iou_0.3']:+.4f} "
                         f"R² {d['r2_cb']:+.4f} 面积误差 {d['fracerr_0.3']:+.4f} 峰值 {d['norm_r2_wss']:+.4f} TAWSS {d['norm_r2_tawss']:+.4f} "
                         f"滞留区 {d['stag_iou']:+.4f} → {verdict}")
    print('\n'.join(lines))
    (HERE / f'report_{CKPT}.txt').write_text('\n'.join(lines) + '\n')
    write_json(HERE / f'report_{CKPT}.json', {'per_fold': {a: {str(k): v for k, v in r.items()} for a, r in res.items()},
                                              'fold_mean': summary, 'verdicts': verdicts})


if __name__ == '__main__':
    main()
