"""C1：自由基线（不训练）。矩阵 §3 C1；判据 G1.2（TAWSS 对 max(TAWSS-null, TAWSS-ratio) + 0.05）、G2.2（OSI 对 OSI-null + 0.05）、G3（滞留区）。

TAWSS-null : T-null B_scale 按 80 帧定义重算：ln τ̂(x,t) = μ(t) + σ(t)·(lp(x) − μ(21))/σ(21)，TAWSS_null = (1/80) Σ_{t<80} (exp(ln τ̂) − eps)
TAWSS-ratio: TAWSS_ratio(x) = c · exp(lp(x))，c = 训练折逐点 TAWSS_true / τ_peak_true 的中位（部署可得常数）
OSI-null   : 训练折上等渗回归 OSI_true ~ f(lp)，lp = 折外峰值 ln 预测；套到留出折的 lp 上
其中 lp：train136 = wss_min_cascade_v1 折外（cv3_v51 seed 1234）；test34 = X5D_v51 五 seed Pa 均值集成。
留出折 k 的"训练折" = 其余两折（它们的 lp 同样是折外值，不泄漏）；test34 的训练 = train136。
指标复用 training_wss_min.metrics（病例等权 R²_cb、线性拟合、阈值掩膜），与训练臂评估同定义。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
from common import (ROOT, TRAIN, TEST, FOLD_OF, FOLD_HELD, FOLD_TRAIN, PEAK_INDEX, EPS, FLOOR,  # noqa: E402
                    load_oof_peak_ln, load_frame_stats)
from training_wss_min import metrics as M  # noqa: E402

CYC = ROOT / 'data_wss_v5/views_v5_1/wss_min_cycle_v1'
OUT = Path(__file__).resolve().parent
N_USED = 80
ISO_MAX_POINTS = 600_000
rng = np.random.default_rng(20260920)


def load_cycle(cid):
    with np.load(CYC / cid / 'cycle.npz', allow_pickle=False) as z:
        return dict(ids=z['wall_node_id_cas'], tawss=z['wall_tawss'].astype(np.float64), osi=z['wall_osi'].astype(np.float64))


def load_peak_true(cid, ids):
    with np.load(ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1' / cid / 'bundle.npz', allow_pickle=False) as z:
        if not np.array_equal(z['wall_node_id_cas'], ids):
            raise ValueError(f'{cid}: bundle rows differ from cycle rows')
        return z['wall_wss'][PEAK_INDEX].astype(np.float64)


def ln_floor(x):
    return np.log(np.clip(np.asarray(x, dtype=np.float64), FLOOR, None) + EPS)


def scalar_summary(true_by_case, pred_by_case, *, log_space=False, masks_above=(), masks_below=()):
    cb = M.casebalanced_field_metrics(true_by_case, pred_by_case)
    cb.update(M.casebalanced_linear_fit_metrics(true_by_case, pred_by_case))
    per_r2 = np.array([M.r2_score(t, p) for t, p in zip(true_by_case, pred_by_case)])
    sp = np.array([spearmanr(t, p).correlation for t, p in zip(true_by_case, pred_by_case)])
    out = {'r2_cb': cb['r2'], 'r2_fit_cb': cb['r2_linear_fit'], 'slope_cb': cb['linear_fit_slope'], 'mae_cb': cb['mae'],
           'r2_casemean': float(per_r2.mean()), 'r2_casemed': float(np.median(per_r2)), 'r2_casep10': float(np.quantile(per_r2, .1)),
           'r2_negative_cases': int((per_r2 < 0).sum()), 'spearman_casemean': float(np.nanmean(sp)), 'n_cases': len(true_by_case)}
    if log_space:
        lt = [ln_floor(t) for t in true_by_case]; lpred = [ln_floor(p) for p in pred_by_case]
        lcb = M.casebalanced_field_metrics(lt, lpred); lcb.update(M.casebalanced_linear_fit_metrics(lt, lpred))
        out.update({'log_r2_cb': lcb['r2'], 'log_r2_fit_cb': lcb['r2_linear_fit'], 'log_slope_cb': lcb['linear_fit_slope'],
                    'log_r2_casemean': float(np.mean([M.r2_score(t, p) for t, p in zip(lt, lpred)]))})
    masks = {}
    for thr in masks_above:
        rows = [M.threshold_mask_metrics(t, p, thr, above=True) for t, p in zip(true_by_case, pred_by_case)]
        masks[f'above_{thr:g}'] = {k: float(np.nanmean([r[k] for r in rows])) for k in ('iou', 'precision', 'recall', 'true_frac', 'pred_frac', 'frac_abs_err')} | \
                                  {'frac_abs_err_casemed': float(np.nanmedian([r['frac_abs_err'] for r in rows]))}
    for thr in masks_below:
        rows = [M.threshold_mask_metrics(t, p, thr, above=False) for t, p in zip(true_by_case, pred_by_case)]
        masks[f'below_{thr:g}'] = {k: float(np.nanmean([r[k] for r in rows])) for k in ('iou', 'precision', 'recall', 'true_frac', 'pred_frac', 'frac_abs_err')} | \
                                  {'frac_abs_err_casemed': float(np.nanmedian([r['frac_abs_err'] for r in rows]))}
    if masks:
        out['threshold_masks'] = masks
    return out


def stagnation_summary(ta_true, osi_true, ta_pred, osi_pred):
    rows = []
    for tt, ot, tp, op in zip(ta_true, osi_true, ta_pred, osi_pred):
        tm = (tt < 0.4) & (ot > 0.1); pm = (tp < 0.4) & (op > 0.1)
        inter = np.count_nonzero(tm & pm); union = np.count_nonzero(tm | pm)
        rows.append(dict(iou=inter / union if union else np.nan, true_frac=tm.mean(), pred_frac=pm.mean(), frac_abs_err=abs(pm.mean() - tm.mean())))
    return {k: float(np.nanmean([r[k] for r in rows])) for k in ('iou', 'true_frac', 'pred_frac', 'frac_abs_err')} | \
           {'iou_casemed': float(np.nanmedian([r['iou'] for r in rows])), 'n_cases': len(rows)}


def fit_osi_null(train_cases, partition):
    xs, ys = [], []
    for cid in train_cases:
        c = load_cycle(cid); lp = load_oof_peak_ln(cid, partition, c['ids'])
        xs.append(lp); ys.append(c['osi'])
    x = np.concatenate(xs); y = np.concatenate(ys)
    if x.size > ISO_MAX_POINTS:
        pick = rng.choice(x.size, ISO_MAX_POINTS, replace=False); x, y = x[pick], y[pick]
    iso = IsotonicRegression(increasing='auto', out_of_bounds='clip', y_min=0.0, y_max=0.5).fit(x, y)
    return iso, {'n_fit_points': int(x.size), 'increasing': bool(iso.increasing_), 'n_train_cases': len(train_cases)}


def fit_ratio(train_cases):
    ratios = []
    for cid in train_cases:
        c = load_cycle(cid); pk = load_peak_true(cid, c['ids'])
        ratios.append(np.median(c['tawss'] / np.clip(pk, FLOOR, None)))
    return float(np.median(ratios)), {'per_case_median_ratio_med': float(np.median(ratios)), 'p10': float(np.quantile(ratios, .1)), 'p90': float(np.quantile(ratios, .9))}


def evaluate_group(held, partition, train_cases, mu, sd, tag, limit=None):
    c_ratio, ratio_meta = fit_ratio(train_cases)
    iso, iso_meta = fit_osi_null(train_cases, 'train')
    ta_t, osi_t, ta_null, ta_ratio, osi_null = [], [], [], [], []
    for cid in held[:limit]:
        c = load_cycle(cid); lp = load_oof_peak_ln(cid, partition, c['ids'])
        z = (lp - mu[PEAK_INDEX]) / sd[PEAK_INDEX]
        ln_frames = mu[:N_USED, None] + sd[:N_USED, None] * z[None, :]
        ta_null.append(np.clip(np.exp(ln_frames) - EPS, 0, None).mean(axis=0))
        ta_ratio.append(c_ratio * np.clip(np.exp(lp) - EPS, 0, None))
        osi_null.append(np.clip(iso.predict(lp), 0.0, 0.5))
        ta_t.append(c['tawss']); osi_t.append(c['osi'])
    res = {
        'TAWSS_null': scalar_summary(ta_t, ta_null, log_space=True, masks_below=(0.4,)),
        'TAWSS_ratio': scalar_summary(ta_t, ta_ratio, log_space=True, masks_below=(0.4,)) | {'ratio_c': c_ratio, 'ratio_meta': ratio_meta},
        'OSI_null': scalar_summary(osi_t, osi_null, masks_above=(0.1, 0.3)) | {'isotonic': iso_meta},
        'stagnation_null_null': stagnation_summary(ta_t, osi_t, ta_null, osi_null),
        'stagnation_ratio_null': stagnation_summary(ta_t, osi_t, ta_ratio, osi_null),
        'n_cases': len(ta_t), 'tag': tag,
    }
    return res


def fmt(res):
    t, r, o = res['TAWSS_null'], res['TAWSS_ratio'], res['OSI_null']
    return (f"TAWSS-null Pa R2cb {t['r2_cb']:.4f} (log {t['log_r2_cb']:.4f}, casemean {t['r2_casemean']:.4f}, sp {t['spearman_casemean']:.3f}, low<0.4 IoU {t['threshold_masks']['below_0.4']['iou']:.3f}) | "
            f"TAWSS-ratio Pa {r['r2_cb']:.4f} (log {r['log_r2_cb']:.4f}, c={r['ratio_c']:.3f}, low IoU {r['threshold_masks']['below_0.4']['iou']:.3f}) | "
            f"OSI-null R2cb {o['r2_cb']:.4f} (casemean {o['r2_casemean']:.4f}, sp {o['spearman_casemean']:.3f}, >0.1 IoU {o['threshold_masks']['above_0.1']['iou']:.3f}, "
            f">0.3 IoU {o['threshold_masks']['above_0.3']['iou']:.3f}, frac err med {o['threshold_masks']['above_0.1']['frac_abs_err_casemed']:.3f}) | "
            f"stagnation IoU null/null {res['stagnation_null_null']['iou']:.3f} ratio/null {res['stagnation_ratio_null']['iou']:.3f}")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    out = {'definition': __doc__.strip().splitlines()[0], 'n_frames_used': N_USED}
    lines = ['=== C1 自由基线（80 帧定义；指标与训练臂同定义）===']
    for k in range(3):
        mu, sd, _ = load_frame_stats(f'fold{k}')
        out[f'fold{k}'] = evaluate_group(FOLD_HELD[k], 'train', FOLD_TRAIN[k], mu, sd, f'fold{k} held-out (cascade OOF)', limit)
        lines.append(f'[fold{k} n={out[f"fold{k}"]["n_cases"]}] ' + fmt(out[f'fold{k}']))
        print(lines[-1], flush=True)
    mu, sd, _ = load_frame_stats('train136')
    out['test34'] = evaluate_group(TEST, 'test', TRAIN, mu, sd, 'test34 (five-seed ensemble)', limit)
    lines.append(f'[test34 n={out["test34"]["n_cases"]}] ' + fmt(out['test34'])); print(lines[-1], flush=True)

    def mean_of(path):
        vals = []
        for k in range(3):
            v = out[f'fold{k}']
            for p in path:
                v = v[p]
            vals.append(float(v))
        return {'mean': float(np.mean(vals)), 'folds': vals}
    out['fold_mean'] = {
        'TAWSS_null_r2_cb': mean_of(('TAWSS_null', 'r2_cb')), 'TAWSS_null_log_r2_cb': mean_of(('TAWSS_null', 'log_r2_cb')),
        'TAWSS_ratio_r2_cb': mean_of(('TAWSS_ratio', 'r2_cb')), 'TAWSS_ratio_log_r2_cb': mean_of(('TAWSS_ratio', 'log_r2_cb')),
        'TAWSS_low_iou_null': mean_of(('TAWSS_null', 'threshold_masks', 'below_0.4', 'iou')), 'TAWSS_low_iou_ratio': mean_of(('TAWSS_ratio', 'threshold_masks', 'below_0.4', 'iou')),
        'OSI_null_r2_cb': mean_of(('OSI_null', 'r2_cb')), 'OSI_null_iou_0p1': mean_of(('OSI_null', 'threshold_masks', 'above_0.1', 'iou')),
        'OSI_null_iou_0p3': mean_of(('OSI_null', 'threshold_masks', 'above_0.3', 'iou')),
        'stagnation_iou_null_null': mean_of(('stagnation_null_null', 'iou')), 'stagnation_iou_ratio_null': mean_of(('stagnation_ratio_null', 'iou')),
    }
    g = out['fold_mean']
    best_tawss = max(g['TAWSS_null_r2_cb']['mean'], g['TAWSS_ratio_r2_cb']['mean'])
    out['gates'] = {'G1.2_tawss_pa_r2cb_threshold': best_tawss + 0.05, 'G1.2_reference': 'max(TAWSS_null, TAWSS_ratio) fold mean',
                    'G2.2_osi_iou_0p1_threshold': g['OSI_null_iou_0p1']['mean'] + 0.05,
                    'G3_stagnation_iou_threshold': max(g['stagnation_iou_null_null']['mean'], g['stagnation_iou_ratio_null']['mean']) + 0.05}
    lines.append(f"[fold mean] TAWSS-null {g['TAWSS_null_r2_cb']['mean']:.4f} / ratio {g['TAWSS_ratio_r2_cb']['mean']:.4f} (log {g['TAWSS_null_log_r2_cb']['mean']:.4f} / {g['TAWSS_ratio_log_r2_cb']['mean']:.4f}); "
                 f"OSI-null R2cb {g['OSI_null_r2_cb']['mean']:.4f}, >0.1 IoU {g['OSI_null_iou_0p1']['mean']:.3f}, >0.3 IoU {g['OSI_null_iou_0p3']['mean']:.3f}; "
                 f"stagnation IoU {g['stagnation_iou_null_null']['mean']:.3f}/{g['stagnation_iou_ratio_null']['mean']:.3f}")
    lines.append(f"[gates] G1.2 TAWSS Pa R2cb ≥ {out['gates']['G1.2_tawss_pa_r2cb_threshold']:.4f}; G2.2 OSI>0.1 IoU ≥ {out['gates']['G2.2_osi_iou_0p1_threshold']:.3f}; "
                 f"G3 stagnation IoU ≥ {out['gates']['G3_stagnation_iou_threshold']:.3f}")
    print('\n'.join(lines[-2:]))
    suffix = f'_limit{limit}' if limit else ''
    (OUT / f'c1_baselines{suffix}.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (OUT / f'c1_baselines{suffix}.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
