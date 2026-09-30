"""M1 三头三 seed 确认读数（矩阵 §14.2）：M1_s{1234,7,2025} train136 → test34；逐 seed + 逐通道三 seed Pa 均值集成；
参照 = 已部署峰值三 seed 集成（通道 0）、阶段 3 单头集成 A1_ens3 / O2_ens3（通道 1 / 2）、C1 test34 自由基线。
    python report_m1_stage3.py [best|last]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import metrics as M, dataset as D, config as C  # noqa: E402

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs/wss_cycle_m1_stage3_20260922'
S3 = ROOT / 'training_wss_min/experiments/wss_cycle_stage3_20260921/offline'
OUT = Path(__file__).resolve().parent
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
SEEDS = (1234, 7, 2025)
CH = list(C.MULTI_CHANNELS)
stats_all = json.loads((ROOT / 'training_wss_min/experiments/wss_cycle_m1_stage3_20260922/stats/multi_stats_train136.json').read_text())
c1 = json.loads((S3 / 'c1_baselines.json').read_text())['test34'] if (S3 / 'c1_baselines.json').is_file() else json.loads((ROOT / 'training_wss_min/experiments/wss_cycle_20260920/offline/c1_baselines.json').read_text())['test34']
peak_ref = json.loads((S3 / 'peak_ensemble_reference.json').read_text())
s3rep = json.loads((S3 / 'stage3_report_best.json').read_text())


def semantic(cid):
    with np.load(ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1' / cid / 'bundle.npz', allow_pickle=True) as b:
        return np.asarray(b['wall_semantic_id']).astype(np.int64)


def channel_result(m, ch):
    return m if ch == 'wss' else m['heads'][ch]


def from_metrics(r):
    ca = r.get('cycle_agreement') or {}
    return {'norm_r2cb': r['normalized']['field_casebalanced']['r2'], 'pa_r2cb': r['field_casebalanced']['r2'], 'r2_casemean': r['aggregate']['r2_casemean'],
            'r2_casep10': r['aggregate']['r2_casep10'], 'neg': int(r['aggregate']['r2_negative_cases']), 'spearman': r['hotspot'].get('spearman_all_casemean'),
            'top10_iou': r['hotspot'].get('top10_iou_casemean'), 'slope_cb': r['field_casebalanced'].get('linear_fit_slope'),
            'masks': {k: {'iou': v['iou_casemean'], 'frac_abs_err_casemed': v['frac_abs_err_casemed']} for k, v in (r.get('threshold_masks') or {}).items()},
            'ccc': (ca.get('ccc') or {}).get('med'), 'seg': (ca.get('seg_rel_err_med') or {}).get('med')}


def summarise(ch, tt, pp, cids):
    st = stats_all[ch]; is_osi = ch == 'osi'
    cb = M.casebalanced_field_metrics(tt, pp); cb.update(M.casebalanced_linear_fit_metrics(tt, pp))
    ncb = M.casebalanced_field_metrics([D.normalize_wss(t, st) for t in tt], [D.normalize_wss(p, st) for p in pp])
    per = np.array([M.r2_score(t, p) for t, p in zip(tt, pp)]); sp = np.array([spearmanr(t, p).correlation for t, p in zip(tt, pp)])
    hot = [M.hotspot_localization_metrics(t, p, np.zeros((len(t), 3))) for t, p in zip(tt, pp)] if ch == 'wss' else None
    above, below = ((0.1, 0.3), ()) if is_osi else (((), (0.4,)) if ch == 'tawss' else ((), ()))
    agr = None
    if ch != 'wss':
        per_case = {c: M.cycle_agreement_case_metrics(t, p, log_space=(ch == 'tawss'), floor=(0.05 if ch == 'tawss' else 0.01), segment_ids=semantic(c),
                                                      masks_above=above, masks_below=below) for c, t, p in zip(cids, tt, pp)}
        agr = M.cycle_agreement_aggregate(per_case)
    masks = {}
    for thr, up in [(t, True) for t in above] + [(t, False) for t in below]:
        rows = [M.threshold_mask_metrics(t, p, thr, above=up) for t, p in zip(tt, pp)]
        masks[f"{'above' if up else 'below'}_{thr:g}"] = {'iou': float(np.nanmean([r['iou'] for r in rows])), 'frac_abs_err_casemed': float(np.nanmedian([r['frac_abs_err'] for r in rows]))}
    return {'norm_r2cb': ncb['r2'], 'pa_r2cb': cb['r2'], 'r2_casemean': float(per.mean()), 'r2_casep10': float(np.quantile(per, .1)), 'neg': int((per < 0).sum()),
            'spearman': float(np.nanmean(sp)), 'slope_cb': cb['linear_fit_slope'], 'top10_iou': float(np.mean([h['top10_iou'] for h in hot])) if hot else None,
            'masks': masks, 'ccc': agr['ccc']['med'] if agr else None, 'seg': agr['seg_rel_err_med']['med'] if agr else None,
            'agreement': {k: agr[k] for k in ('ccc', 'rel_err_med', 'within_tol', 'top_dice', 'seg_rel_err_med', 'seg_rel_err_max', 'case_mean', 'masks')} if agr else None, 'n_cases': len(cids)}


def preds(seed, ch):
    d = RUNS / f'M1_s{seed}/eval/ckpt_{CKPT}/predictions/test/{ch}'
    out = {}
    for pa in sorted(d.glob('*/*/*/predictions.npz')):
        cid = '/'.join(pa.relative_to(d).parts[:-1])
        with np.load(pa) as z:
            out[cid] = (z['true_pa'].astype(np.float64), z['pred_pa'].astype(np.float64))
    return out


def fmt(tag, s, ref=None, ref_name=''):
    mk = ' '.join(f"{k}:IoU {v['iou']:.3f}/err {v['frac_abs_err_casemed']:.3f}" for k, v in s['masks'].items())
    r = f" | {ref_name} norm {ref['norm_r2cb']:.4f} Δ {s['norm_r2cb']-ref['norm_r2cb']:+.4f}, Pa {ref['pa_r2cb']:.4f} Δ {s['pa_r2cb']-ref['pa_r2cb']:+.4f}" if ref else ''
    extra = f" CCC {s['ccc']:.3f} seg {s['seg']:.3f}" if s.get('ccc') is not None else (f" top10 IoU {s['top10_iou']:.3f}" if s.get('top10_iou') is not None else '')
    return f"{tag:16s} norm {s['norm_r2cb']:.4f} Pa {s['pa_r2cb']:.4f} casemean {s['r2_casemean']:.4f} p10 {s['r2_casep10']:.4f} neg {s['neg']} sp {s['spearman']:.3f} slope {s['slope_cb']:.3f}{extra} | {mk}{r}"


report = {'checkpoint': CKPT, 'seeds': SEEDS, 'per_seed': {}, 'ensembles': {}, 'references': {'peak_ens3': peak_ref, 'A1_ens3': s3rep['ensembles'].get('A1'), 'O2_ens3': s3rep['ensembles'].get('O2'), 'test34_baselines': c1}}
lines = [f'=== M1 三 seed 确认 test34（{CKPT}）===']
refs = {'wss': (peak_ref, '峰值三seed集成'), 'tawss': (s3rep['ensembles'].get('A1'), 'A1_ens3'), 'osi': (s3rep['ensembles'].get('O2'), 'O2_ens3')}
for ch in CH:
    lines.append(f'--- 通道 {ch} ---')
    per_seed = {}; allp = {}
    for seed in SEEDS:
        p = RUNS / f'M1_s{seed}/eval/ckpt_{CKPT}/metrics.json'
        if not p.is_file():
            lines.append(f'M1_s{seed}: 未完成'); continue
        s = from_metrics(channel_result(json.loads(p.read_text())['test'], ch)); per_seed[str(seed)] = s
        same = json.loads((ROOT / f'training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s{seed}/eval/ckpt_best/metrics.json').read_text())['test'] if ch == 'wss' else \
               json.loads((ROOT / f"training_wss_min/runs/wss_cycle_stage3_20260921/{'A1' if ch == 'tawss' else 'O2'}_s{seed}/eval/ckpt_best/metrics.json").read_text())['test']
        sref = {'norm_r2cb': same['normalized']['field_casebalanced']['r2'], 'pa_r2cb': same['field_casebalanced']['r2']}
        s['same_seed_reference'] = sref
        lines.append(fmt(f'M1_s{seed}[{ch}]', s, sref, '同seed单头' if ch != 'wss' else '同seed峰值'))
        allp[seed] = preds(seed, ch)
    report['per_seed'][ch] = per_seed
    if len(allp) == len(SEEDS):
        cids = sorted(allp[SEEDS[0]]); tt = [allp[SEEDS[0]][c][0] for c in cids]
        pp = [np.mean([allp[s][c][1] for s in SEEDS], axis=0) for c in cids]
        if ch == 'osi':
            pp = [np.clip(p, 0.0, 0.5) for p in pp]
        ens = summarise(ch, tt, pp, cids)
        ens['seed_mean'] = {k: float(np.mean([per_seed[str(s)][k] for s in SEEDS])) for k in ('norm_r2cb', 'pa_r2cb')}
        ens['seed_sd'] = {k: float(np.std([per_seed[str(s)][k] for s in SEEDS], ddof=1)) for k in ('norm_r2cb', 'pa_r2cb')}
        report['ensembles'][ch] = ens
        ref, rname = refs[ch]
        lines.append(fmt(f'M1_ens3[{ch}]', ens, ref, rname) + f" (三seed均值 Pa {ens['seed_mean']['pa_r2cb']:.4f} ± {ens['seed_sd']['pa_r2cb']:.4f})")
        ens['_pred'] = dict(zip(cids, zip(tt, pp)))
if all(ch in report['ensembles'] for ch in ('tawss', 'osi')):
    a, o = report['ensembles']['tawss']['_pred'], report['ensembles']['osi']['_pred']
    ious, dices, tf, pf = [], [], [], []
    for cid, (ta, pa) in a.items():
        to, po = o[cid]; tm = (ta < 0.4) & (to > 0.1); pm = (pa < 0.4) & (po > 0.1); u = np.count_nonzero(tm | pm)
        ious.append(np.count_nonzero(tm & pm) / u if u else np.nan); dices.append(M.dice_masks(tm, pm)); tf.append(tm.mean()); pf.append(pm.mean())
    ba = M._bland_altman(tf, pf, relative=False); base = max(c1['stagnation_null_null']['iou'], c1['stagnation_ratio_null']['iou'])
    report['ensembles']['stagnation'] = {'iou_casemean': float(np.nanmean(ious)), 'dice_med': float(np.nanmedian(dices)), 'area_share_ba': ba, 'baseline_iou': base,
                                         'single_head_A1xO2': s3rep['ensembles'].get('stagnation_A1xO2', {}).get('iou_casemean')}
    lines.append(f"滞留区 M1 集成 TAWSS×OSI: IoU {np.nanmean(ious):.3f} (基线 {base:.3f}; 单头 A1×O2 集成 {report['ensembles']['stagnation']['single_head_A1xO2']}) Dice med {np.nanmedian(dices):.3f} 面积份额 BA {ba['bias']:+.3f}[{ba['loa_low']:+.3f},{ba['loa_high']:+.3f}]")
for ch in list(report['ensembles']):
    if isinstance(report['ensembles'][ch], dict):
        report['ensembles'][ch].pop('_pred', None)
print('\n'.join(lines))
(OUT / f'm1_stage3_report_{CKPT}.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
(OUT / f'm1_stage3_report_{CKPT}.txt').write_text('\n'.join(lines) + '\n')
