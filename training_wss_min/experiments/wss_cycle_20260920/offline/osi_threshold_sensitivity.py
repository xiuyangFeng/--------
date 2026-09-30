"""OSI 掩膜阈值敏感性（只读；验证「0.1 恰在分布中位、掩膜 IoU 对该档不灵敏」的判断，矩阵 §12.2 第 3 点）。

对 O1 / O2 的留出折 best 预测与同折 OSI-null（训练折等渗 OSI ~ ln τ̂_peak），在阈值 0.1 / 0.15 / 0.2 / 0.25 / 0.3 与分位阈值
（逐例真值 OSI 的 top 50% / 30% / 15%，预测用同分位）上算掩膜 IoU、Dice、面积份额误差；报三折均值与 Δ(模型 − OSI-null)。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
import c1_baselines as B  # noqa: E402
from common import FOLD_HELD, FOLD_TRAIN, load_oof_peak_ln  # noqa: E402
from training_wss_min import metrics as M  # noqa: E402

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs/wss_cycle_20260920'
OUT = Path(__file__).resolve().parent
ABS = (0.1, 0.15, 0.2, 0.25, 0.3)
QUANT = (0.5, 0.7, 0.85)


def masks_for(t, p, thr, quantile=False):
    if quantile:
        tm, pm = t >= np.quantile(t, thr), p >= np.quantile(p, thr)
    else:
        tm, pm = t > thr, p > thr
    u = np.count_nonzero(tm | pm)
    return (np.count_nonzero(tm & pm) / u if u else np.nan, M.dice_masks(tm, pm), abs(pm.mean() - tm.mean()))


res = {}
for k in range(3):
    held = FOLD_HELD[k]
    iso, _ = B.fit_osi_null(FOLD_TRAIN[k], 'train')
    preds = {'OSI-null': [], 'O1': [], 'O2': []}; truth = []
    for cid in held:
        c = B.load_cycle(cid); lp = load_oof_peak_ln(cid, 'train', c['ids'])
        truth.append(c['osi']); preds['OSI-null'].append(np.clip(iso.predict(lp), 0, 0.5))
        for o in ('O1', 'O2'):
            with np.load(RUNS / f'{o}_f{k}_s1234/eval/ckpt_best/predictions/test' / cid / 'predictions.npz') as z:
                if not np.array_equal(z['true_pa'].astype(np.float32), c['osi'].astype(np.float32)):
                    raise ValueError(f'{cid}: prediction rows differ from cycle view')
                preds[o].append(z['pred_pa'].astype(np.float64))
    fr = {}
    for name, pp in preds.items():
        fr[name] = {}
        for thr in ABS:
            vals = np.array([masks_for(t, p, thr) for t, p in zip(truth, pp)])
            fr[name][f'abs_{thr:g}'] = {'iou': float(np.nanmean(vals[:, 0])), 'dice': float(np.nanmedian(vals[:, 1])), 'frac_err_med': float(np.median(vals[:, 2])),
                                        'true_frac_med': float(np.median([(t > thr).mean() for t in truth]))}
        for q in QUANT:
            vals = np.array([masks_for(t, p, q, quantile=True) for t, p in zip(truth, pp)])
            fr[name][f'top{int(round((1-q)*100))}pct'] = {'iou': float(np.nanmean(vals[:, 0])), 'dice': float(np.nanmedian(vals[:, 1]))}
    res[f'fold{k}'] = fr
    print(f'fold{k} done', flush=True)

keys = list(res['fold0']['O1'].keys())
lines = ['=== OSI 掩膜阈值敏感性（三折均值；IoU 逐例均值 / Dice 逐例中位 / 面积份额误差中位）===',
         f"{'阈值':12s} {'真值面积份额':>10s} | {'OSI-null IoU':>12s} {'O1 IoU':>8s} {'ΔO1':>7s} {'O2 IoU':>8s} {'ΔO2':>7s} | {'null Dice':>9s} {'O1 Dice':>8s} {'O2 Dice':>8s} | {'O1 frac_err':>11s} {'O2 frac_err':>11s}"]
summary = {}
for key in keys:
    m = lambda name, f: float(np.mean([res[f'fold{k}'][name][key][f] for k in range(3)]))
    tf = float(np.mean([res[f'fold{k}']['O1'][key].get('true_frac_med', np.nan) for k in range(3)]))
    summary[key] = {n: {f: m(n, f) for f in res['fold0'][n][key]} for n in ('OSI-null', 'O1', 'O2')}
    fe = lambda n: f"{m(n, 'frac_err_med'):11.3f}" if 'frac_err_med' in res['fold0'][n][key] else f"{'—':>11s}"
    lines.append(f"{key:12s} {tf:10.3f} | {m('OSI-null','iou'):12.3f} {m('O1','iou'):8.3f} {m('O1','iou')-m('OSI-null','iou'):+7.3f} {m('O2','iou'):8.3f} {m('O2','iou')-m('OSI-null','iou'):+7.3f} | "
                 f"{m('OSI-null','dice'):9.3f} {m('O1','dice'):8.3f} {m('O2','dice'):8.3f} | {fe('O1')} {fe('O2')}")
print('\n'.join(lines))
(OUT / 'osi_threshold_sensitivity.json').write_text(json.dumps({'per_fold': res, 'fold_mean': summary}, indent=1, default=float))
(OUT / 'osi_threshold_sensitivity.txt').write_text('\n'.join(lines) + '\n')
