"""阶段 1 门控读数（矩阵 §4）：A1 / O1 / O2 × cv3，best 与 last；对同折 X5D_v51 峰值底座（G1.1）与 C1 自由基线（G1.2 / G2.2 / G2.3 / G3）。

滞留区（G3）需要 A1 与 O* 的同点预测（队列评估带 --save-predictions），在这里离线组合。
    python report_gates.py [best|last]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs/wss_cycle_20260920'
BASE = ROOT / 'training_wss_min/runs/wss_v51_wave2a_20260916'
OUT = Path(__file__).resolve().parent
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
c1 = json.loads((OUT / 'c1_baselines.json').read_text())


def metrics(run, ckpt=CKPT):
    p = run / f'eval/ckpt_{ckpt}/metrics.json'
    return json.loads(p.read_text())['test'] if p.is_file() else None


def stagnation(fold, ckpt=CKPT):
    """A1 × O 的滞留区 IoU（TAWSS < 0.4 ∧ OSI > 0.1），同折同点预测。"""
    out = {}
    a_dir = RUNS / f'A1_f{fold}_s1234/eval/ckpt_{ckpt}/predictions/test'
    if not a_dir.is_dir():
        return out
    for o in ('O1', 'O2'):
        o_dir = RUNS / f'{o}_f{fold}_s1234/eval/ckpt_{ckpt}/predictions/test'
        if not o_dir.is_dir():
            continue
        ious, errs = [], []
        for pa in sorted(a_dir.glob('*/*/*/predictions.npz')):
            rel = pa.relative_to(a_dir); po = o_dir / rel
            if not po.is_file():
                continue
            with np.load(pa) as za, np.load(po) as zo:
                tt, tp = za['true_pa'], za['pred_pa']; ot, op = zo['true_pa'], zo['pred_pa']
            if len(tt) != len(ot):
                raise ValueError(f'row mismatch {rel}')
            tm = (tt < 0.4) & (ot > 0.1); pm = (tp < 0.4) & (op > 0.1)
            u = np.count_nonzero(tm | pm); ious.append(np.count_nonzero(tm & pm) / u if u else np.nan); errs.append(abs(pm.mean() - tm.mean()))
        out[o] = {'iou_casemean': float(np.nanmean(ious)), 'iou_casemed': float(np.nanmedian(ious)), 'frac_abs_err_casemean': float(np.mean(errs)), 'n_cases': len(ious)}
    return out


rows = []
for kind in ('A1', 'O1', 'O2'):
    per_fold = []
    for k in range(3):
        m = metrics(RUNS / f'{kind}_f{k}_s1234')
        if m is None:
            per_fold.append(None); continue
        base = metrics(BASE / f'X5D_v51_f{k}_s1234', 'best')
        r = {'fold': k, 'pa_r2cb': m['field_casebalanced']['r2'], 'norm_r2cb': m['normalized']['field_casebalanced']['r2'],
             'r2_fit_cb': m['field_casebalanced'].get('r2_linear_fit'), 'slope_cb': m['field_casebalanced'].get('linear_fit_slope'),
             'r2_casemean': m['aggregate']['r2_casemean'], 'r2_casemed': m['aggregate']['r2_casemed'], 'neg_cases': m['aggregate']['r2_negative_cases'],
             'spearman': m['hotspot'].get('spearman_all_casemean'), 'mae': m['field']['mae'],
             'base_norm_r2cb': base['normalized']['field_casebalanced']['r2'], 'base_pa_r2cb': base['field_casebalanced']['r2']}
        tm = m.get('threshold_masks', {})
        for key, blk in tm.items():
            r[f'{key}_iou'] = blk['iou_casemean']; r[f'{key}_frac_err_med'] = blk['frac_abs_err_casemed']; r[f'{key}_true_frac'] = blk['true_frac_casemean']; r[f'{key}_pred_frac'] = blk['pred_frac_casemean']
        per_fold.append(r)
    rows.append((kind, per_fold))

report = {'checkpoint': CKPT, 'arms': {}, 'gates': {}}
lines = [f'=== 阶段 1 门控（{CKPT}；三折均值 + 折符号）===']
fm = c1['fold_mean']
for kind, per_fold in rows:
    done = [r for r in per_fold if r]
    if len(done) < 3:
        lines.append(f'[{kind}] 完成 {len(done)}/3 折'); report['arms'][kind] = per_fold; continue
    def mean(k): return float(np.mean([r[k] for r in done]))
    def signs(fn): return sum(1 for r in done if fn(r))
    report['arms'][kind] = per_fold
    if kind == 'A1':
        g11 = [r['norm_r2cb'] - r['base_norm_r2cb'] for r in done]
        base_pa = [max(c1[f'fold{r["fold"]}']['TAWSS_null']['r2_cb'], c1[f'fold{r["fold"]}']['TAWSS_ratio']['r2_cb']) for r in done]
        g12 = [r['pa_r2cb'] - b for r, b in zip(done, base_pa)]
        low_base = [max(c1[f'fold{r["fold"]}']['TAWSS_null']['threshold_masks']['below_0.4']['iou'], c1[f'fold{r["fold"]}']['TAWSS_ratio']['threshold_masks']['below_0.4']['iou']) for r in done]
        low_d = [r['below_0.4_iou'] - b for r, b in zip(done, low_base)]
        report['gates']['G1.1'] = {'delta_norm_vs_base': g11, 'mean': float(np.mean(g11)), 'pass': all(d >= -0.02 for d in g11)}
        report['gates']['G1.2'] = {'delta_pa_vs_best_baseline': g12, 'mean': float(np.mean(g12)), 'pass': float(np.mean(g12)) >= 0.05 and all(d > 0 for d in g12),
                                   'low_tawss_iou_delta': low_d, 'low_iou_not_worse': all(d >= -0.0 for d in low_d)}
        lines.append(f"[A1] norm R2cb {mean('norm_r2cb'):.4f} (base {mean('base_norm_r2cb'):.4f}, Δ {np.mean(g11):+.4f}, 折 {signs(lambda r: r['norm_r2cb']-r['base_norm_r2cb']>=-0.02)}/3 ≥−0.02) | "
                     f"Pa R2cb {mean('pa_r2cb'):.4f} (基线 {np.mean(base_pa):.4f}, Δ {np.mean(g12):+.4f}, {sum(d>0 for d in g12)}/3 同向) | r²fit {mean('r2_fit_cb'):.4f} slope {mean('slope_cb'):.3f} sp {mean('spearman'):.3f} | "
                     f"low<0.4 IoU {mean('below_0.4_iou'):.3f} (基线 {np.mean(low_base):.3f}) | G1.1 {'PASS' if report['gates']['G1.1']['pass'] else 'FAIL'} G1.2 {'PASS' if report['gates']['G1.2']['pass'] else 'FAIL'}")
    else:
        iou_base = [c1[f'fold{r["fold"]}']['OSI_null']['threshold_masks']['above_0.1']['iou'] for r in done]
        d_iou = [r['above_0.1_iou'] - b for r, b in zip(done, iou_base)]
        g21 = all(r['pa_r2cb'] > 0 and r['r2_casemed'] > 0 for r in done)
        g22 = float(np.mean(d_iou)) >= 0.05 and all(d > 0 for d in d_iou)
        g23 = float(np.mean([r['above_0.1_frac_err_med'] for r in done])) <= 0.05
        report['gates'][f'{kind}.G2.1'] = g21; report['gates'][f'{kind}.G2.2'] = {'delta_iou_0p1': d_iou, 'pass': g22}; report['gates'][f'{kind}.G2.3'] = {'frac_err_med': [r['above_0.1_frac_err_med'] for r in done], 'pass': g23}
        lines.append(f"[{kind}] OSI R2cb {mean('pa_r2cb'):.4f} (OSI-null {fm['OSI_null_r2_cb']['mean']:.4f}; casemed {mean('r2_casemed'):.4f}, neg {sum(r['neg_cases'] for r in done)}) | "
                     f"norm R2cb {mean('norm_r2cb'):.4f} r²fit {mean('r2_fit_cb'):.4f} slope {mean('slope_cb'):.3f} sp {mean('spearman'):.3f} | "
                     f">0.1 IoU {mean('above_0.1_iou'):.3f} (null {np.mean(iou_base):.3f}, Δ {np.mean(d_iou):+.3f}, {sum(d>0 for d in d_iou)}/3) >0.3 IoU {mean('above_0.3_iou'):.3f} | "
                     f"面积份额误差 med {mean('above_0.1_frac_err_med'):.3f} (true {mean('above_0.1_true_frac'):.3f} pred {mean('above_0.1_pred_frac'):.3f}) | "
                     f"G2.1 {'PASS' if g21 else 'FAIL'} G2.2 {'PASS' if g22 else 'FAIL'} G2.3 {'PASS' if g23 else 'FAIL'}")
stag = {k: stagnation(k) for k in range(3)}
for o in ('O1', 'O2'):
    vals = [stag[k][o]['iou_casemean'] for k in range(3) if o in stag[k]]
    if len(vals) == 3:
        base = max(fm['stagnation_iou_null_null']['mean'], fm['stagnation_iou_ratio_null']['mean'])
        bf = [max(c1[f'fold{k}']['stagnation_null_null']['iou'], c1[f'fold{k}']['stagnation_ratio_null']['iou']) for k in range(3)]
        d = [v - b for v, b in zip(vals, bf)]
        report['gates'][f'G3.A1x{o}'] = {'iou': vals, 'baseline': bf, 'pass': float(np.mean(d)) >= 0.05 and sum(x > 0 for x in d) >= 2}
        lines.append(f"[G3 A1×{o}] 滞留区 IoU {np.mean(vals):.3f} (基线 {base:.3f}, Δ {np.mean(d):+.3f}, {sum(x>0 for x in d)}/3) → {'PASS' if report['gates'][f'G3.A1x{o}']['pass'] else 'FAIL'}")
report['stagnation'] = stag
print('\n'.join(lines))
(OUT / f'gate_report_{CKPT}.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
(OUT / f'gate_report_{CKPT}.txt').write_text('\n'.join(lines) + '\n')
