"""阶段 3 读数（矩阵 §13）：A1 / O1 / O2 × seed 1234/7/2025，train136 → test34 读一次。

逐 seed：metrics.json 的归一化 / 物理 R²_cb、逐例分布、Spearman、阈值掩膜 IoU、cycle_agreement 摘要；对照同 seed 已部署峰值 X5D_v51_s{seed}
（归一化 R²_cb）与 C1 的 test34 自由基线。三 seed 集成：逐例 Pa 预测取均值，重算全套（归一化空间用 train136 统计量把集成 Pa 映回 z）。
滞留区：A1 集成 × O 集成。
    python report_stage3.py [best|last]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import metrics as M, dataset as D  # noqa: E402

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs/wss_cycle_stage3_20260921'
PEAK = ROOT / 'training_wss_min/runs/wss_v51_wave1_20260916'
CYC = ROOT / 'data_wss_v5/views_v5_1/wss_min_cycle_v1'
OUT = Path(__file__).resolve().parent
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
SEEDS = (1234, 7, 2025)
c1 = json.loads((ROOT / 'training_wss_min/experiments/wss_cycle_20260920/offline/c1_baselines.json').read_text())['test34']
STATS = {'A1': CYC / 'stats/cycle_stats_tawss_train136.json', 'O1': CYC / 'stats/cycle_stats_osi_linear_train136.json', 'O2': CYC / 'stats/cycle_stats_osi_logit_train136.json'}


def metrics(kind, seed):
    p = RUNS / f'{kind}_s{seed}/eval/ckpt_{CKPT}/metrics.json'
    return json.loads(p.read_text())['test'] if p.is_file() else None


def preds(kind, seed):
    d = RUNS / f'{kind}_s{seed}/eval/ckpt_{CKPT}/predictions/test'
    out = {}
    for pa in sorted(d.glob('*/*/*/predictions.npz')):
        cid = '/'.join(pa.relative_to(d).parts[:-1])
        with np.load(pa) as z:
            out[cid] = (z['true_pa'].astype(np.float64), z['pred_pa'].astype(np.float64))
    return out


def semantic(cid):
    with np.load(ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1' / cid / 'bundle.npz', allow_pickle=True) as b:
        return np.asarray(b['wall_semantic_id']).astype(np.int64)


def summarise(kind, true_by_case, pred_by_case, cids):
    stats = json.loads(STATS[kind].read_text()); is_osi = kind != 'A1'
    tn = [D.normalize_wss(t, stats) for t in true_by_case]; pn = [D.normalize_wss(p, stats) for p in pred_by_case]
    cb = M.casebalanced_field_metrics(true_by_case, pred_by_case); cb.update(M.casebalanced_linear_fit_metrics(true_by_case, pred_by_case))
    ncb = M.casebalanced_field_metrics(tn, pn)
    per_r2 = np.array([M.r2_score(t, p) for t, p in zip(true_by_case, pred_by_case)])
    sp = np.array([spearmanr(t, p).correlation for t, p in zip(true_by_case, pred_by_case)])
    above, below = ((0.1, 0.3), ()) if is_osi else ((), (0.4,))
    per_case = {cid: M.cycle_agreement_case_metrics(t, p, log_space=not is_osi, floor=(0.05 if not is_osi else 0.01), segment_ids=semantic(cid),
                                                    masks_above=above, masks_below=below) for cid, t, p in zip(cids, true_by_case, pred_by_case)}
    agg = M.cycle_agreement_aggregate(per_case)
    masks = {}
    for thr, up in [(t, True) for t in above] + [(t, False) for t in below]:
        rows = [M.threshold_mask_metrics(t, p, thr, above=up) for t, p in zip(true_by_case, pred_by_case)]
        masks[f"{'above' if up else 'below'}_{thr:g}"] = {'iou': float(np.nanmean([r['iou'] for r in rows])), 'frac_abs_err_casemed': float(np.nanmedian([r['frac_abs_err'] for r in rows]))}
    return {'pa_r2cb': cb['r2'], 'norm_r2cb': ncb['r2'], 'r2_fit_cb': cb['r2_linear_fit'], 'slope_cb': cb['linear_fit_slope'], 'mae_cb': cb['mae'],
            'r2_casemean': float(per_r2.mean()), 'r2_casemed': float(np.median(per_r2)), 'r2_casep10': float(np.quantile(per_r2, .1)), 'neg': int((per_r2 < 0).sum()),
            'spearman': float(np.nanmean(sp)), 'threshold_masks': masks,
            'agreement': {k: agg[k] for k in ('ccc', 'rel_err_med', 'within_tol', 'top_dice', 'seg_rel_err_med', 'seg_rel_err_max', 'case_mean', 'masks')},
            'n_cases': len(cids)}


def from_metrics(m):
    ca = m.get('cycle_agreement', {})
    return {'pa_r2cb': m['field_casebalanced']['r2'], 'norm_r2cb': m['normalized']['field_casebalanced']['r2'], 'r2_fit_cb': m['field_casebalanced'].get('r2_linear_fit'),
            'slope_cb': m['field_casebalanced'].get('linear_fit_slope'), 'mae_cb': m['field_casebalanced']['mae'], 'r2_casemean': m['aggregate']['r2_casemean'],
            'r2_casemed': m['aggregate']['r2_casemed'], 'r2_casep10': m['aggregate']['r2_casep10'], 'neg': int(m['aggregate']['r2_negative_cases']),
            'spearman': m['hotspot'].get('spearman_all_casemean'),
            'threshold_masks': {k: {'iou': v['iou_casemean'], 'frac_abs_err_casemed': v['frac_abs_err_casemed']} for k, v in (m.get('threshold_masks') or {}).items()},
            'agreement': {k: ca.get(k) for k in ('ccc', 'rel_err_med', 'within_tol', 'top_dice', 'seg_rel_err_med', 'seg_rel_err_max', 'case_mean', 'masks')},
            'n_cases': m['aggregate']['n_cases']}


def line(tag, s, base_norm=None, base_txt=''):
    a = s['agreement']; masks = s['threshold_masks']
    mk = ' '.join(f"{k}:IoU {v['iou']:.3f}/err {v['frac_abs_err_casemed']:.3f}" for k, v in masks.items())
    bn = f" (峰值同seed {base_norm:.4f}, Δ {s['norm_r2cb']-base_norm:+.4f})" if base_norm is not None else ''
    ag = (f"CCC {a['ccc']['med']:.3f}/p10 {a['ccc']['p10']:.3f}/min {a['ccc']['min']:.3f} seg {a['seg_rel_err_med']['med']:.3f}/worst {a['seg_rel_err_max']['med']:.3f} "
          f"BA {a['case_mean']['bias']:+.3f}[{a['case_mean']['loa_low']:+.3f},{a['case_mean']['loa_high']:+.3f}]") if a.get('ccc') else 'agreement —'
    return (f"{tag:14s} norm {s['norm_r2cb']:.4f}{bn} | Pa {s['pa_r2cb']:.4f}{base_txt} casemean {s['r2_casemean']:.4f} p10 {s['r2_casep10']:.4f} neg {s['neg']} "
            f"sp {s['spearman']:.3f} r²fit {s['r2_fit_cb']:.4f} slope {s['slope_cb']:.3f} | {mk} | {ag}")


report = {'checkpoint': CKPT, 'seeds': SEEDS, 'test34_baselines': c1, 'arms': {}, 'ensembles': {}}
lines = [f'=== 阶段 3 test34 读数（{CKPT}）；自由基线 TAWSS-null Pa {c1["TAWSS_null"]["r2_cb"]:.4f} (log {c1["TAWSS_null"]["log_r2_cb"]:.4f}) / ratio {c1["TAWSS_ratio"]["r2_cb"]:.4f} / OSI-null {c1["OSI_null"]["r2_cb"]:.4f} (IoU>0.1 {c1["OSI_null"]["threshold_masks"]["above_0.1"]["iou"]:.3f}) ===']
ens_pred = {}
for kind in ('A1', 'O1', 'O2'):
    per_seed = {}
    allp = {}
    for seed in SEEDS:
        m = metrics(kind, seed)
        if m is None:
            lines.append(f'{kind}_s{seed}: 未完成'); continue
        pk = json.loads((PEAK / f'X5D_v51_s{seed}/eval/ckpt_best/metrics.json').read_text())['test']['normalized']['field_casebalanced']['r2']
        s = from_metrics(m); s['peak_same_seed_norm_r2cb'] = pk; per_seed[str(seed)] = s
        if kind == 'A1':
            bt = f" (基线 max {max(c1['TAWSS_null']['r2_cb'], c1['TAWSS_ratio']['r2_cb']):.4f}, Δ {s['pa_r2cb']-max(c1['TAWSS_null']['r2_cb'], c1['TAWSS_ratio']['r2_cb']):+.4f})"
        else:
            bt = f" (OSI-null {c1['OSI_null']['r2_cb']:.4f}, Δ {s['pa_r2cb']-c1['OSI_null']['r2_cb']:+.4f})"
        lines.append(line(f'{kind}_s{seed}', s, pk if kind == 'A1' else None, bt))
        allp[seed] = preds(kind, seed)
    report['arms'][kind] = per_seed
    if len(allp) == len(SEEDS):
        cids = sorted(allp[SEEDS[0]])
        tt = [allp[SEEDS[0]][c][0] for c in cids]; pp = [np.mean([allp[s][c][1] for s in SEEDS], axis=0) for c in cids]
        if kind != 'A1':
            pp = [np.clip(p, 0.0, 0.5) for p in pp]
        ens = summarise(kind, tt, pp, cids); ens['seed_mean'] = {k: float(np.mean([per_seed[str(s)][k] for s in SEEDS])) for k in ('norm_r2cb', 'pa_r2cb', 'r2_casemean')}
        ens['seed_sd'] = {k: float(np.std([per_seed[str(s)][k] for s in SEEDS], ddof=1)) for k in ('norm_r2cb', 'pa_r2cb')}
        report['ensembles'][kind] = ens; ens_pred[kind] = dict(zip(cids, zip(tt, pp)))
        lines.append(line(f'{kind}_ens3', ens, None, f" (三seed均值 {ens['seed_mean']['pa_r2cb']:.4f} ± {ens['seed_sd']['pa_r2cb']:.4f})"))
if 'A1' in ens_pred:
    for o in ('O1', 'O2'):
        if o not in ens_pred:
            continue
        ious, dices, tf, pf = [], [], [], []
        for cid, (ta, pa) in ens_pred['A1'].items():
            to, po = ens_pred[o][cid]
            tm = (ta < 0.4) & (to > 0.1); pm = (pa < 0.4) & (po > 0.1); u = np.count_nonzero(tm | pm)
            ious.append(np.count_nonzero(tm & pm) / u if u else np.nan); dices.append(M.dice_masks(tm, pm)); tf.append(tm.mean()); pf.append(pm.mean())
        ba = M._bland_altman(tf, pf, relative=False)
        base = max(c1['stagnation_null_null']['iou'], c1['stagnation_ratio_null']['iou'])
        report['ensembles'][f'stagnation_A1x{o}'] = {'iou_casemean': float(np.nanmean(ious)), 'dice_med': float(np.nanmedian(dices)), 'area_share_ba': ba, 'baseline_iou': base}
        lines.append(f"滞留区 A1×{o} 集成: IoU {np.nanmean(ious):.3f} (基线 {base:.3f}, Δ {np.nanmean(ious)-base:+.3f}) Dice med {np.nanmedian(dices):.3f} 面积份额 BA {ba['bias']:+.3f}[{ba['loa_low']:+.3f},{ba['loa_high']:+.3f}]")
print('\n'.join(lines))
(OUT / f'stage3_report_{CKPT}.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
(OUT / f'stage3_report_{CKPT}.txt').write_text('\n'.join(lines) + '\n')
