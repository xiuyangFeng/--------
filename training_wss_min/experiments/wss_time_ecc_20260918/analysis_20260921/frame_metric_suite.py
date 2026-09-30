"""逐帧「型态 vs 幅值」指标套件（2026-09-21，只读已有产物，不训练、不改代码）。

臂：
  X5D_v51_frozen  = 部署底座 X5D_v51 三折折外峰值预测，冻结后对 81 帧真值逐帧评估（无时间输入的模型在非峰值帧就是这个）
  Tnull_Bscale    = D2 自由基线 B_scale：z_peak = (ln τ̂_peak − μ_f(peak)) / σ_f(peak)；ln τ̂(t) = μ_f(t) + σ_f(t)·z_peak（训练折统计）
  T0 / TB8 / TB16 = 阶段 2 时间臂（ckpt best），直接读 metrics.json 的逐帧字段
指标（Pa 空间；R²_cb / r / α / β / CCC 用 field_casebalanced，Spearman / top10 IoU / NMAE_case 为逐例均值，
approx_disp / NMAE_frame / NMAE_cyclemax 为 pooled）：
  r2cb        病例等权 R²（现行门控口径）
  r, r2_fit   病例等权线性拟合 pred = a·true + b 的 Pearson r 与 r²（r2_linear_fit）
  alpha       σ_pred/σ_true = a / r；beta_n2 = 2αr − α² − R²（Gupta 2009 分解反解）
  ccc         Lin 一致性相关 = 2rα / (1 + α² + β_n²)
  spearman    逐例 Spearman 均值（hotspot.spearman_all_casemean）
  top10_iou   逐例 top-10% 掩膜 IoU 均值
  approx_disp √(Σ(τ−τ̂)² / Στ²)（Suk / Rygiel 的 approximation disparity，pooled）。存量臂由
              RMSE、pooled R²、NRMSE_mean 精确还原：σ² = RMSE²/(1−R²)，μ = RMSE/NRMSE_mean，disp = RMSE/√(σ²+μ²)
  nmae_frame  pooled MAE / 该帧 pooled 真值 range（现行 nmae_range）
  nmae_case   逐例 MAE / 逐例 range 的均值（aggregate.nmae_casemean）
  nmae_cyclemax  pooled MAE / 全周期各帧 pooled 真值 range 的最大值（≈ Rygiel 的 ÷ max over space & time）
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

GNN = Path('/public/newhome/cy/Digital_twin/GNN')
sys.path.insert(0, str(GNN))
from training_wss_min import metrics as M  # noqa: E402

TW = GNN / 'training_wss_min'
EXP = TW / 'experiments/wss_time_ecc_20260918'
OUT = EXP / 'analysis_20260921'
VIEW = GNN / 'data_wss_v5/views_v5_1/wss_min_view_v1'
X5D = TW / 'runs/wss_v51_wave2a_20260916'
TIME_RUNS = TW / 'runs/wss_time_ecc_20260918'
FOLDS = (0, 1, 2)
FLOOR, EPS, PEAK_INDEX = 0.05, 1e-6, 21
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
STORED_ARMS = ('T0', 'TB8', 'TB16')
KEYS = ['r2cb', 'r', 'r2_fit', 'alpha', 'beta_n2', 'ccc', 'spearman', 'top10_iou', 'approx_disp',
        'nmae_frame', 'nmae_case', 'nmae_cyclemax', 'mae', 'rmse', 'r2_raw', 'nrmse_mean', 'range_pooled']

# ---------- 相位（与 analysis_20260919 同一套） ----------
phase_of, q_of, steps = {}, {}, []
with open(EXP / 'analysis_20260919/frame_r2_source.csv') as f:
    for row in csv.DictReader(f):
        s = int(row['step']); steps.append(s); q_of[s] = float(row['q_norm'])
        phase_of[s] = ('trough' if row['in_trough'] == '1' else 'peak' if row['in_peak'] == '1'
                       else 'accel' if row['in_systole'] == '1' else 'decel')
steps = sorted(steps); assert len(steps) == 81


def decompose(cb: dict, fld: dict, agg: dict, hot: dict) -> dict:
    r2 = cb['r2']; r = math.sqrt(max(cb['r2_linear_fit'], 0.0)); a = cb['linear_fit_slope']
    alpha = a / r if r > 1e-9 else float('nan')
    beta2 = 2 * alpha * r - alpha ** 2 - r2 if np.isfinite(alpha) else float('nan')
    ccc = 2 * r * alpha / (1 + alpha ** 2 + max(beta2, 0.0)) if np.isfinite(alpha) else float('nan')
    rmse, r2raw, nrm = fld['rmse'], fld['r2'], fld['nrmse_mean']
    sig2 = rmse ** 2 / (1 - r2raw) if r2raw < 1 else float('nan')
    mu = rmse / nrm
    disp = rmse / math.sqrt(sig2 + mu ** 2)
    return dict(r2cb=r2, r=r, r2_fit=r ** 2, alpha=alpha, beta_n2=beta2, ccc=ccc,
                spearman=hot['spearman_all_casemean'], top10_iou=hot['top10_iou_casemean'],
                approx_disp=disp, nmae_frame=fld['nmae_range'], nmae_case=agg['nmae_casemean'],
                mae=fld['mae'], rmse=rmse, r2_raw=r2raw, nrmse_mean=nrm,
                range_pooled=fld['mae'] / fld['nmae_range'])


rows = []
# ---------- 存量臂 ----------
for arm in STORED_ARMS:
    for fo in FOLDS:
        m = json.load(open(TIME_RUNS / f'{arm}_f{fo}_s1234/eval/ckpt_{CKPT}/metrics.json'))['test']['frames']
        for s_str, e in m.items():
            ph = e['physical']
            d = decompose(ph['field_casebalanced'], ph['field'], ph['aggregate'], ph['hotspot'])
            rows.append(dict(arm=arm, fold=fo, step=int(s_str), **d))

# ---------- 计算臂：X5D_v51 冻结 + T-null B_scale ----------
def ln_pa(t):
    return np.log(np.clip(np.asarray(t, np.float64), FLOOR, None) + EPS)


def frame_suite(true_by_case, pred_by_case, pos_by_case):
    cb = M.casebalanced_field_metrics(true_by_case, pred_by_case)
    cb.update(M.casebalanced_linear_fit_metrics(true_by_case, pred_by_case))
    pt = np.concatenate(true_by_case); pp = np.concatenate(pred_by_case)
    fld = M.basic_metrics(pt, pp)
    sp, iou, nm = [], [], []
    for yt, yp in zip(true_by_case, pred_by_case):
        sp.append(M._spearman(yt, yp))
        tm = yt >= np.percentile(yt, 90); pm = yp >= np.percentile(yp, 90)
        u = np.logical_or(tm, pm).sum(); iou.append(np.logical_and(tm, pm).sum() / u if u else np.nan)
        nm.append(M.nmae(yt, yp, 'range'))
    hot = dict(spearman_all_casemean=float(np.nanmean(sp)), top10_iou_casemean=float(np.nanmean(iou)))
    agg = dict(nmae_casemean=float(np.nanmean(nm)))
    d = decompose(cb, fld, agg, hot)
    d['approx_disp_direct'] = float(np.sqrt(np.sum((pt - pp) ** 2) / np.sum(pt ** 2)))
    d['ccc_pooled_direct'] = M.lin_ccc(pt, pp)
    return d


checks = {}
for fo in FOLDS:
    split = json.load(open(VIEW / f'cv3_v51/fold{fo}.json'))
    cids = list(split['test_cases'])
    st = json.load(open(EXP / f'offline/wss_frame_stats_fold{fo}.json'))['frame']
    mu, sd = np.asarray(st['log_mean'], np.float64), np.asarray(st['log_std'], np.float64)
    tau, pred, pos = [], [], []
    for cid in cids:
        with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
            bsteps = z['steps'].astype(int); assert list(bsteps) == steps, cid
            assert int(np.flatnonzero(bsteps == int(z['peak_step']))[0]) == PEAK_INDEX
            tau.append(z['wall_wss'].astype(np.float32)); pos.append(z['wall_coords_aligned_mm'].astype(np.float64))
        with np.load(X5D / f'X5D_v51_f{fo}_s1234/eval/ckpt_best/predictions/test/{cid}/predictions.npz') as z:
            p = z['pred_pa'].astype(np.float64); tp = z['true_pa'].astype(np.float32)
            assert np.array_equal(z['row_index'], np.arange(len(p)))
            assert tp.shape == tau[-1][PEAK_INDEX].shape and np.allclose(tp, tau[-1][PEAK_INDEX], rtol=1e-5, atol=1e-6), f'{cid}: true_pa ≠ bundle peak frame'
            pred.append(np.clip(p, 0, None))
    zpk = [(ln_pa(p) - mu[PEAK_INDEX]) / sd[PEAK_INDEX] for p in pred]
    for k, s in enumerate(steps):
        tb = [t[k].astype(np.float64) for t in tau]
        d = frame_suite(tb, pred, pos); rows.append(dict(arm='X5D_v51_frozen', fold=fo, step=s, **d))
        tn = [np.clip(np.exp(mu[k] + sd[k] * z) - EPS, 0, None) for z in zpk]
        d = frame_suite(tb, tn, pos); rows.append(dict(arm='Tnull_Bscale', fold=fo, step=s, **d))
    pk = [x for x in rows if x['arm'] == 'X5D_v51_frozen' and x['fold'] == fo and x['step'] == steps[PEAK_INDEX]][0]
    stored = json.load(open(X5D / f'X5D_v51_f{fo}_s1234/eval/ckpt_best/metrics.json'))['test']
    checks[f'fold{fo}'] = dict(peak_r2cb_recomputed=pk['r2cb'], peak_r2cb_stored=stored['field_casebalanced']['r2'],
                              peak_spearman_recomputed=pk['spearman'], peak_spearman_stored=stored['hotspot']['spearman_all_casemean'],
                              peak_nmae_case_recomputed=pk['nmae_case'], peak_nmae_case_stored=stored['aggregate']['nmae_casemean'],
                              approx_disp_formula=pk['approx_disp'], approx_disp_direct=pk['approx_disp_direct'],
                              n_cases=len(cids))
    print(f'fold{fo}: ' + ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in checks[f'fold{fo}'].items()), flush=True)

# nmae_cyclemax：每臂每折的全周期 pooled range 最大值
for arm in {x['arm'] for x in rows}:
    for fo in FOLDS:
        sel = [x for x in rows if x['arm'] == arm and x['fold'] == fo]
        gmax = max(x['range_pooled'] for x in sel)
        for x in sel:
            x['nmae_cyclemax'] = x['mae'] / gmax
for x in rows:
    x['phase'] = phase_of[x['step']]; x['q_norm'] = q_of[x['step']]

with open(OUT / f'frame_metric_suite_{CKPT}.csv', 'w', newline='') as f:
    fields = ['arm', 'fold', 'step', 'phase', 'q_norm'] + KEYS + ['approx_disp_direct', 'ccc_pooled_direct']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

# 相位汇总：三折均值（每帧 3 折等权）
ARMS = ['X5D_v51_frozen', 'Tnull_Bscale', 'T0', 'TB8', 'TB16']
PHASES = ['accel', 'peak', 'decel', 'trough', 'cycle']
summary = {}
for arm in ARMS:
    for ph in PHASES:
        sel = [x for x in rows if x['arm'] == arm and (ph == 'cycle' or x['phase'] == ph)]
        summary[(arm, ph)] = {k: float(np.nanmean([x[k] for x in sel])) for k in KEYS}
        summary[(arm, ph)]['n_frames'] = len({x['step'] for x in sel})
        # 最差帧（三折均值后）
        per_step = {}
        for x in sel:
            per_step.setdefault(x['step'], []).append(x['r2cb'])
        worst = min(per_step.items(), key=lambda kv: np.mean(kv[1]))
        summary[(arm, ph)]['worst_r2cb'] = float(np.mean(worst[1])); summary[(arm, ph)]['worst_step'] = int(worst[0])
json.dump({'ckpt': CKPT, 'checks': checks, 'phase_frames': {p: sum(1 for s in steps if phase_of[s] == p) for p in PHASES[:-1]},
           'summary': {f'{a}|{p}': v for (a, p), v in summary.items()}}, open(OUT / f'frame_metric_suite_{CKPT}.json', 'w'), indent=1)

show = ['r2cb', 'r2_fit', 'spearman', 'ccc', 'approx_disp', 'nmae_case', 'nmae_cyclemax', 'top10_iou', 'alpha']
print(f'\n== ckpt={CKPT} 三折均值 × 相位 ==')
print(f"{'arm':15s} {'phase':7s} {'n':>3s} " + ' '.join(f'{k:>11s}' for k in show) + ' worst_r2cb@step')
for arm in ARMS:
    for ph in PHASES:
        v = summary[(arm, ph)]
        print(f"{arm:15s} {ph:7s} {v['n_frames']:3d} " + ' '.join(f'{v[k]:11.3f}' for k in show) + f"  {v['worst_r2cb']:.3f}@{v['worst_step']}")
print('wrote', OUT / f'frame_metric_suite_{CKPT}.csv')
