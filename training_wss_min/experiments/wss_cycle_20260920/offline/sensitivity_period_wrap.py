"""敏感性：剔除周期未收敛例（C0 审计 period_wrap_rel_diff > 5%，41/170）后重算阶段 1 主读数（矩阵 §4「只报不判」）。

臂：A1/O1/O2 × cv3（best 同点预测 predictions.npz）；基线：c1_baselines.evaluate_group 在同一子集上重算。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_baselines as B                      # noqa: E402  (imports common.*, defines evaluate_group)
from common import FOLD_HELD, FOLD_TRAIN, load_frame_stats  # noqa: E402
from training_wss_min import metrics as M     # noqa: E402

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs/wss_cycle_20260920'
CYC = ROOT / 'data_wss_v5/views_v5_1/wss_min_cycle_v1'
OUT = Path(__file__).resolve().parent
man = json.loads((CYC / 'cycle_manifest.json').read_text())
FLAGGED = set(man['period_wrap_flagged'])


def arm_metrics(kind, fold, cases):
    d = RUNS / f'{kind}_f{fold}_s1234/eval/ckpt_best/predictions/test'
    tt, pp = [], []
    for cid in cases:
        with np.load(d / cid / 'predictions.npz') as z:
            tt.append(z['true_pa'].astype(np.float64)); pp.append(z['pred_pa'].astype(np.float64))
    if kind == 'A1':
        return B.scalar_summary(tt, pp, log_space=True, masks_below=(0.4,))
    return B.scalar_summary(tt, pp, masks_above=(0.1, 0.3))


def stag(fold, o, cases):
    a = RUNS / f'A1_f{fold}_s1234/eval/ckpt_best/predictions/test'; b = RUNS / f'{o}_f{fold}_s1234/eval/ckpt_best/predictions/test'
    ious = []
    for cid in cases:
        with np.load(a / cid / 'predictions.npz') as za, np.load(b / cid / 'predictions.npz') as zo:
            tm = (za['true_pa'] < 0.4) & (zo['true_pa'] > 0.1); pm = (za['pred_pa'] < 0.4) & (zo['pred_pa'] > 0.1)
        u = np.count_nonzero(tm | pm); ious.append(np.count_nonzero(tm & pm) / u if u else np.nan)
    return float(np.nanmean(ious))


res = {'flag_threshold': man['period_wrap_flag_threshold'], 'n_flagged_total': len(FLAGGED), 'folds': {}}
for k in range(3):
    held = FOLD_HELD[k]; keep = [c for c in held if c not in FLAGGED]; drop = [c for c in held if c in FLAGGED]
    mu, sd, _ = load_frame_stats(f'fold{k}')
    base_keep = B.evaluate_group(keep, 'train', FOLD_TRAIN[k], mu, sd, f'fold{k} unflagged')
    base_drop = B.evaluate_group(drop, 'train', FOLD_TRAIN[k], mu, sd, f'fold{k} flagged') if len(drop) >= 3 else None
    fr = {'n_keep': len(keep), 'n_drop': len(drop)}
    for kind in ('A1', 'O1', 'O2'):
        fr[kind] = {'keep': arm_metrics(kind, k, keep), 'drop': arm_metrics(kind, k, drop) if len(drop) >= 3 else None}
    fr['baseline_keep'] = {kk: base_keep[kk] for kk in ('TAWSS_null', 'TAWSS_ratio', 'OSI_null', 'stagnation_ratio_null', 'stagnation_null_null')}
    fr['baseline_drop'] = ({kk: base_drop[kk] for kk in ('TAWSS_null', 'TAWSS_ratio', 'OSI_null', 'stagnation_ratio_null', 'stagnation_null_null')} if base_drop else None)
    fr['stagnation_keep'] = {o: stag(k, o, keep) for o in ('O1', 'O2')}
    fr['stagnation_drop'] = {o: stag(k, o, drop) for o in ('O1', 'O2')} if len(drop) >= 3 else None
    res['folds'][str(k)] = fr
    print(f'fold{k}: keep {len(keep)} drop {len(drop)}', flush=True)


def fm(path, sub='keep'):
    vals = []
    for k in range(3):
        v = res['folds'][str(k)]
        for p in path:
            v = v[p] if v is not None else None
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float('nan')


lines = [f"=== 周期未收敛例敏感性（剔除 {len(FLAGGED)} 例后三折均值；括号内为被剔除子集）===",
         f"A1 TAWSS Pa R2cb {fm(('A1','keep','r2_cb')):.4f} ({fm(('A1','drop','r2_cb')):.4f}) vs 全量 0.7162 | 基线 max(null,ratio) {max(fm(('baseline_keep','TAWSS_null','r2_cb')), fm(('baseline_keep','TAWSS_ratio','r2_cb'))):.4f} ({max(fm(('baseline_drop','TAWSS_null','r2_cb')), fm(('baseline_drop','TAWSS_ratio','r2_cb'))):.4f}) | log R2cb {fm(('A1','keep','log_r2_cb')):.4f} ({fm(('A1','drop','log_r2_cb')):.4f}) vs null log {fm(('baseline_keep','TAWSS_null','log_r2_cb')):.4f} | low<0.4 IoU {fm(('A1','keep','threshold_masks','below_0.4','iou')):.3f} vs null {fm(('baseline_keep','TAWSS_null','threshold_masks','below_0.4','iou')):.3f}",
         f"O1 OSI R2cb {fm(('O1','keep','r2_cb')):.4f} ({fm(('O1','drop','r2_cb')):.4f}) vs null {fm(('baseline_keep','OSI_null','r2_cb')):.4f} | >0.1 IoU {fm(('O1','keep','threshold_masks','above_0.1','iou')):.3f} ({fm(('O1','drop','threshold_masks','above_0.1','iou')):.3f}) vs null {fm(('baseline_keep','OSI_null','threshold_masks','above_0.1','iou')):.3f} | frac err med {fm(('O1','keep','threshold_masks','above_0.1','frac_abs_err_casemed')):.3f}",
         f"O2 OSI R2cb {fm(('O2','keep','r2_cb')):.4f} ({fm(('O2','drop','r2_cb')):.4f}) | >0.1 IoU {fm(('O2','keep','threshold_masks','above_0.1','iou')):.3f} ({fm(('O2','drop','threshold_masks','above_0.1','iou')):.3f}) | frac err med {fm(('O2','keep','threshold_masks','above_0.1','frac_abs_err_casemed')):.3f}",
         f"滞留区 IoU A1×O1 {fm(('stagnation_keep','O1')):.3f} ({fm(('stagnation_drop','O1')):.3f}) A1×O2 {fm(('stagnation_keep','O2')):.3f} | 基线 {max(fm(('baseline_keep','stagnation_null_null','iou')), fm(('baseline_keep','stagnation_ratio_null','iou'))):.3f}"]
print('\n'.join(lines))
(OUT / 'sensitivity_period_wrap.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
(OUT / 'sensitivity_period_wrap.txt').write_text('\n'.join(lines) + '\n')
