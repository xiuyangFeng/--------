"""OSI 零训练探针（P0）公共读取与指标（只读）。跟踪 §34.9 / §34.10；预注册见同目录 PREREG.md。

预测来源（全部已保存，不做推理）：
- cv3 折外（train136，cv3_v51 三折，seed 1234）：O1 / O2 单头 = runs/wss_cycle_20260920/{O1,O2}_f{k}_s1234；
  M1 三头 = runs/wss_cycle_m1_20260921/M1_f{k}_s1234（osi / tawss 通道子目录）；A1 = runs/wss_cycle_20260920/A1_f{k}_s1234
- test34（train136 训练）：{A1,O1,O2}_s{1234,7,2025} = runs/wss_cycle_stage3_20260921；M1_s{…} = runs/wss_cycle_m1_stage3_20260922
标签：v5.1 周期视图 wss_min_cycle_v1（帧 0–79）。指标复用 training_wss_min.metrics（病例等权 R²_cb 等），与训练臂评估同定义。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training_wss_min/experiments/wss_time_ecc_20260918/offline'))
from common import TRAIN, TEST, FOLD_HELD, FOLD_TRAIN, FOLD_OF  # noqa: E402,F401
from training_wss_min import metrics as M  # noqa: E402

RUNS = ROOT / 'training_wss_min/runs'
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
CYC = ROOT / 'data_wss_v5/views_v5_1/wss_min_cycle_v1'
OUT = Path(__file__).resolve().parent
P = 'eval/ckpt_best/predictions/test'
SEEDS3 = (1234, 7, 2025)
OSI_THR = (0.1, 0.2, 0.3)
STAG_TAWSS, STAG_OSI = 0.4, 0.1

MANIFEST = json.loads((CYC / 'cycle_manifest.json').read_text())
WRAP = {c['case']: float(c['period_wrap_rel_diff']) for c in MANIFEST['cases']}
FLAGGED = set(MANIFEST['period_wrap_flagged'])


def cv_dir(arm, k, channel=None):
    if arm == 'M1':
        return RUNS / f'wss_cycle_m1_20260921/M1_f{k}_s1234' / P / channel
    return RUNS / f'wss_cycle_20260920/{arm}_f{k}_s1234' / P


def t34_dir(arm, seed, channel=None):
    if arm == 'M1':
        return RUNS / f'wss_cycle_m1_stage3_20260922/M1_s{seed}' / P / channel
    return RUNS / f'wss_cycle_stage3_20260921/{arm}_s{seed}' / P


def load_cycle(cid):
    with np.load(CYC / cid / 'cycle.npz', allow_pickle=False) as z:
        return dict(ids=z['wall_node_id_cas'], tawss=z['wall_tawss'].astype(np.float64), osi=z['wall_osi'].astype(np.float64),
                    rev_frac=z['wall_rev_frac'].astype(np.float64))


def load_pred(d, cid, truth):
    """同点预测（物理量，已裁剪）；行序与周期视图逐位核对。"""
    with np.load(d / cid / 'predictions.npz') as z:
        if not np.array_equal(z['row_index'], np.arange(len(truth))):
            raise ValueError(f'{d}/{cid}: unexpected row order')
        if not np.array_equal(z['true_pa'].astype(np.float32), truth.astype(np.float32)):
            raise ValueError(f'{d}/{cid}: prediction truth differs from cycle view')
        return z['pred_pa'].astype(np.float64)


def cohort(cid):
    return cid.split('/')[0]


def cv_bundle():
    """cv3 折外：每例 {fold, osi, tawss, O1, O2, M1(osi 通道), M1_tawss}。"""
    data = {}
    for k in range(3):
        for cid in FOLD_HELD[k]:
            c = load_cycle(cid)
            d = dict(fold=k, osi=c['osi'], tawss=c['tawss'], rev_frac=c['rev_frac'])
            for arm in ('O1', 'O2'):
                d[arm] = load_pred(cv_dir(arm, k), cid, c['osi'])
            d['M1'] = load_pred(cv_dir('M1', k, 'osi'), cid, c['osi'])
            d['M1_tawss'] = load_pred(cv_dir('M1', k, 'tawss'), cid, c['tawss'])
            data[cid] = d
    return data


def t34_bundle():
    """test34：每例 {osi, tawss, O1_s*, O2_s*, M1_s*(osi), M1_tawss(三 seed 均值 = 已部署口径)}。"""
    data = {}
    for cid in TEST:
        c = load_cycle(cid)
        d = dict(osi=c['osi'], tawss=c['tawss'], rev_frac=c['rev_frac'])
        for arm in ('O1', 'O2'):
            for s in SEEDS3:
                d[f'{arm}_s{s}'] = load_pred(t34_dir(arm, s), cid, c['osi'])
        for s in SEEDS3:
            d[f'M1_s{s}'] = load_pred(t34_dir('M1', s, 'osi'), cid, c['osi'])
        d['M1_tawss'] = np.mean([load_pred(t34_dir('M1', s, 'tawss'), cid, c['tawss']) for s in SEEDS3], axis=0)
        data[cid] = d
    return data


def cv_ensemble(d, members):
    return np.mean([d[m] for m in members], axis=0)


CV_ENSEMBLES = {'M1': ('M1',), 'O1': ('O1',), 'O2': ('O2',), 'O1+O2': ('O1', 'O2'), 'O1+M1': ('O1', 'M1'),
                'O2+M1': ('O2', 'M1'), 'ALL3': ('O1', 'O2', 'M1')}
T34_ENSEMBLES = {'M1_ens3': tuple(f'M1_s{s}' for s in SEEDS3), 'O1_ens3': tuple(f'O1_s{s}' for s in SEEDS3),
                 'O2_ens3': tuple(f'O2_s{s}' for s in SEEDS3),
                 'O1O2_ens6': tuple(f'{a}_s{s}' for a in ('O1', 'O2') for s in SEEDS3),
                 'ALL9': tuple(f'{a}_s{s}' for a in ('O1', 'O2', 'M1') for s in SEEDS3)}


def osi_summary(cases, osi_t, osi_p, ta_t=None, ta_p=None, thr_override=None, stag_osi_thr=None):
    """OSI 全套读数。thr_override = {T: t'}：阈值掩膜用预测阈值 t' 对真值阈值 T（阈值校准）；None = 同阈值。"""
    cb = M.casebalanced_field_metrics(osi_t, osi_p)
    per_r2 = np.array([M.r2_score(t, p) for t, p in zip(osi_t, osi_p)])
    sp = np.array([spearmanr(t, p).correlation for t, p in zip(osi_t, osi_p)])
    ccc = np.array([M.lin_ccc(t, p) for t, p in zip(osi_t, osi_p)])
    tt, pp = np.concatenate(osi_t), np.concatenate(osi_p)
    out = {'n_cases': len(cases), 'r2_cb': cb['r2'], 'mae_cb': cb['mae'],
           'r2_casemean': float(per_r2.mean()), 'r2_casemed': float(np.median(per_r2)), 'r2_casep10': float(np.quantile(per_r2, .1)),
           'r2_negative_cases': int((per_r2 < 0).sum()), 'spearman_casemean': float(np.nanmean(sp)),
           'ccc_casemed': float(np.nanmedian(ccc)), 'ccc_casep10': float(np.nanquantile(ccc, .1)),
           'rel_l2_pooled': float(np.sqrt(np.sum((pp - tt) ** 2) / np.sum(tt ** 2))),
           'rel_l2_casemean': float(np.mean([np.sqrt(np.sum((p - t) ** 2) / np.sum(t ** 2)) for t, p in zip(osi_t, osi_p)])),
           'pred_mean': float(np.mean([p.mean() for p in osi_p])), 'true_mean': float(np.mean([t.mean() for t in osi_t]))}
    for T in OSI_THR:
        tp = (thr_override or {}).get(T, T)
        rows = []
        for t, p in zip(osi_t, osi_p):
            tm, pm = t > T, p > tp
            u = np.count_nonzero(tm | pm)
            rows.append((np.count_nonzero(tm & pm) / u if u else np.nan, M.dice_masks(tm, pm), tm.mean(), pm.mean()))
        r = np.array(rows, dtype=np.float64)
        out[f'mask_{T:g}'] = {'iou': float(np.nanmean(r[:, 0])), 'dice_casemed': float(np.nanmedian(r[:, 1])),
                              'true_frac': float(r[:, 2].mean()), 'pred_frac': float(r[:, 3].mean()),
                              'frac_abs_err_casemed': float(np.median(np.abs(r[:, 3] - r[:, 2]))),
                              'frac_bias': float(np.mean(r[:, 3] - r[:, 2])), 'pred_threshold': float(tp)}
    if ta_t is not None:
        so = STAG_OSI if stag_osi_thr is None else stag_osi_thr
        rows = []
        for a, o, ap, op in zip(ta_t, osi_t, ta_p, osi_p):
            tm = (a < STAG_TAWSS) & (o > STAG_OSI); pm = (ap < STAG_TAWSS) & (op > so)
            u = np.count_nonzero(tm | pm)
            rows.append((np.count_nonzero(tm & pm) / u if u else np.nan, tm.mean(), pm.mean()))
        r = np.array(rows, dtype=np.float64)
        out['stagnation'] = {'iou': float(np.nanmean(r[:, 0])), 'iou_casemed': float(np.nanmedian(r[:, 0])),
                             'true_frac': float(r[:, 1].mean()), 'pred_frac': float(r[:, 2].mean()),
                             'frac_abs_err_casemed': float(np.median(np.abs(r[:, 2] - r[:, 1])))}
    by = {}
    for g in ('AG', 'AAA', 'ILO'):
        idx = [i for i, c in enumerate(cases) if cohort(c) == g]
        if idx:
            by[g] = M.casebalanced_field_metrics([osi_t[i] for i in idx], [osi_p[i] for i in idx])['r2']
    out['r2_cb_by_cohort'] = by
    out['per_case_r2'] = {c: float(v) for c, v in zip(cases, per_r2)}
    return out


def fold_mean(per_fold, keys=('r2_cb', 'ccc_casemed', 'r2_casemean', 'r2_casep10', 'spearman_casemean', 'rel_l2_pooled', 'mae_cb')):
    """三折留出均值（与矩阵 §12 同口径：先每折算 R²_cb 再平均）。"""
    out = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
    for T in OSI_THR:
        out[f'iou_{T:g}'] = float(np.mean([f[f'mask_{T:g}']['iou'] for f in per_fold]))
        out[f'fracerr_{T:g}'] = float(np.mean([f[f'mask_{T:g}']['frac_abs_err_casemed'] for f in per_fold]))
        out[f'fracbias_{T:g}'] = float(np.mean([f[f'mask_{T:g}']['frac_bias'] for f in per_fold]))
    if 'stagnation' in per_fold[0]:
        out['stag_iou'] = float(np.mean([f['stagnation']['iou'] for f in per_fold]))
    out['r2_cb_folds'] = [float(f['r2_cb']) for f in per_fold]
    return out


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


ROW_KEYS = ('r2_cb', 'ccc_casemed', 'r2_casep10', 'iou_0.1', 'iou_0.2', 'iou_0.3', 'fracerr_0.1', 'fracerr_0.3', 'stag_iou', 'rel_l2_pooled')


def flat(s):
    """单次汇总 → 与 fold_mean 同键的扁平行。"""
    if 'iou_0.1' in s:
        return s
    out = {k: s[k] for k in ('r2_cb', 'ccc_casemed', 'r2_casemean', 'r2_casep10', 'spearman_casemean', 'rel_l2_pooled', 'mae_cb')}
    for T in OSI_THR:
        out[f'iou_{T:g}'] = s[f'mask_{T:g}']['iou']
        out[f'fracerr_{T:g}'] = s[f'mask_{T:g}']['frac_abs_err_casemed']
        out[f'fracbias_{T:g}'] = s[f'mask_{T:g}']['frac_bias']
    if 'stagnation' in s:
        out['stag_iou'] = s['stagnation']['iou']
    return out


def table(rows, ref=None, keys=ROW_KEYS):
    """rows: [(name, flat dict)] → 定宽文本表；ref = 参照行名，另起 Δ 行。"""
    head = f"{'':28s}" + ''.join(f'{k:>13s}' for k in keys)
    lines = [head]
    refd = dict(rows).get(ref) if ref else None
    for name, d in rows:
        lines.append(f'{name:28s}' + ''.join(f"{d.get(k, float('nan')):13.4f}" for k in keys))
        if refd is not None and name != ref:
            lines.append(f"{'  Δ vs ' + ref:28s}" + ''.join(f"{d.get(k, np.nan) - refd.get(k, np.nan):+13.4f}" for k in keys))
    return '\n'.join(lines)
