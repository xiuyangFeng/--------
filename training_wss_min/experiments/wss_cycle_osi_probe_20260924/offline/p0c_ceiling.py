"""P0c OSI 天花板与误差结构探针（零训练，只读；cv3 折外 136 例）。预注册见 PREREG.md §P0c。

(a) 空间尺度分解：壁面 kNN 图（k=10）上的测地距离 Gaussian 平滑 S_s，s ∈ {1, 2, 4} mm（截断 3s），每例 2000 个评估点。
    R²(S_s y, y) = 尺度 > s 的真值方差份额（「分辨率天花板」）；R²(S_s ŷ, S_s y) = 粗尺度上模型精度；
    R²(ŷ, S_s y)；细尺度 R²(ŷ − S_s ŷ, y − S_s y)；误差能量中粗尺度份额 Σ(S_s e)² / Σe²。
(b) 标签周期性：点级 w = ‖τ⃗_80 − τ⃗_0‖ / max(TAWSS, 0.05)；平移一帧标签 OSI'（帧 1–80）对 OSI（帧 0–79）的 R² = 标签噪声下界；
    逐例模型 R² 与病例级 period_wrap 的 Spearman（OSI 对 TAWSS 做差中之差）；按 w 分箱的点份额 / SSE 份额。
(c) 标定曲线：按 ŷ 分箱的 E[y]，按 y 分箱的 E[ŷ]（量化「高 OSI 幅值偏低」）。
(d) SSE 分布：按队列、语义血管段、真值 OSI 档、真值 TAWSS 档。
模型：ALL3（O1+O2+M1 等权，P0a 主臂）与 M1（已部署模型族）；TAWSS 模型对照用 M1 的 TAWSS 通道。
"""
from __future__ import annotations

import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from p0_common import (VIEW, OUT, WRAP, FLAGGED, cv_bundle, cv_ensemble, cohort, write_json, M)

SCALES = (1.0, 2.0, 4.0)
N_EVAL = 2000
KNN = 10
CHUNK = 200
FLOOR = 0.05
SEM = {0: 'trunk', 1: 'L_CIA', 2: 'R_CIA', 3: 'L_ext', 4: 'L_int', 5: 'R_ext', 6: 'R_int'}
W_BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf)
OSI_BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5001)
TAWSS_BINS = (0.0, 0.1, 0.2, 0.4, 1.0, np.inf)
PRED_BINS = np.round(np.arange(0.0, 0.5001, 0.025), 3)


def osi_from(vec):
    s = np.linalg.norm(vec, axis=2).sum(axis=0)
    m = np.linalg.norm(vec.sum(axis=0), axis=1)
    return 0.5 * (1.0 - np.divide(m, s, out=np.ones_like(s), where=s > 0))


def worker(args):
    cid, fields = args
    rng = np.random.default_rng(zlib.crc32(cid.encode()))
    with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
        xyz = z['wall_coords_aligned_mm'].astype(np.float64)
        vec = z['wall_wss_vec'].astype(np.float64)
        sem = z['wall_semantic_id'].astype(np.int64)
    osi0 = osi_from(vec[:80])
    if np.max(np.abs(osi0 - fields['osi'])) > 1e-4:
        raise ValueError(f'{cid}: recomputed OSI (frames 0-79) differs from cycle view')
    osi_shift = osi_from(vec[1:81])
    w = np.linalg.norm(vec[80] - vec[0], axis=1) / np.maximum(fields['tawss'], FLOOR)
    del vec
    n = len(xyz)
    dist, nbr = cKDTree(xyz).query(xyz, k=KNN + 1)
    rows = np.repeat(np.arange(n), KNN); cols = nbr[:, 1:].ravel(); d = dist[:, 1:].ravel()
    g = coo_matrix((d, (rows, cols)), shape=(n, n)).tocsr()
    g = g.maximum(g.T)
    ev = np.sort(rng.choice(n, min(N_EVAL, n), replace=False))
    keys = ('osi', 'ALL3', 'M1')
    sm = {s: {k: np.empty(len(ev)) for k in keys} for s in SCALES}
    limit = 3.0 * max(SCALES)
    for a in range(0, len(ev), CHUNK):
        idx = ev[a:a + CHUNK]
        D = dijkstra(g, directed=False, indices=idx, limit=limit)
        for s in SCALES:
            W = np.where(D <= 3.0 * s, np.exp(-0.5 * (np.nan_to_num(D, posinf=1e9) / s) ** 2), 0.0)
            den = W.sum(axis=1)
            for k in keys:
                sm[s][k][a:a + CHUNK] = W @ fields[k] / den
        del D
    return cid, dict(ev=ev, sm=sm, osi_shift=osi_shift, w=w, sem=sem,
                     nn_spacing_mm=float(np.median(dist[:, 1])))


def r2cb(t, p):
    return M.casebalanced_field_metrics(t, p)['r2']


def main():
    t0 = time.time()
    cv = cv_bundle()
    for c in cv:
        cv[c]['ALL3'] = cv_ensemble(cv[c], ('O1', 'O2', 'M1'))
    cases = list(cv)
    jobs = [(c, {k: cv[c][k] for k in ('osi', 'tawss', 'ALL3', 'M1')}) for c in cases]
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    out = {}
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for i, (cid, r) in enumerate(ex.map(worker, jobs, chunksize=1)):
            out[cid] = r
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(cases)} {time.time() - t0:.0f}s', flush=True)
    res = {'n_cases': len(cases), 'median_nn_spacing_mm': float(np.median([out[c]['nn_spacing_mm'] for c in cases]))}
    text = ['=== P0c OSI 天花板与误差结构（cv3 折外 136 例，零训练）===',
            f"壁面点中位最近邻间距 {res['median_nn_spacing_mm']:.3f} mm；每例 {N_EVAL} 个评估点；测地 Gaussian 平滑"]

    # (a) 尺度分解
    res['scale'] = {}
    text += ['', '[(a) 空间尺度分解，病例等权 R²；s = 平滑尺度 (mm)]',
             f"{'':8s}{'R²(Sy,y)':>10s}" + ''.join(f'{m + " " + k:>18s}' for m in ('ALL3', 'M1') for k in ('R²(ŷ,y)@ev', 'R²(Sŷ,Sy)', 'R²(ŷ,Sy)', 'R²细(ŷ,y)', '粗误差份额'))]
    for s in SCALES:
        y = [cv[c]['osi'][out[c]['ev']] for c in cases]
        sy = [out[c]['sm'][s]['osi'] for c in cases]
        row = {'r2_Sy_y': r2cb(y, sy)}
        for m in ('ALL3', 'M1'):
            p = [cv[c][m][out[c]['ev']] for c in cases]
            sp = [out[c]['sm'][s][m] for c in cases]
            e = [pp - yy for pp, yy in zip(p, y)]; se = [a - b for a, b in zip(sp, sy)]
            row[m] = {'r2_p_y_ev': r2cb(y, p), 'r2_Sp_Sy': r2cb(sy, sp), 'r2_p_Sy': r2cb(sy, p),
                      'r2_fine': r2cb([a - b for a, b in zip(y, sy)], [a - b for a, b in zip(p, sp)]),
                      'coarse_err_share': float(np.mean([np.sum(a ** 2) / np.sum(b ** 2) for a, b in zip(se, e)]))}
        res['scale'][f'{s:g}mm'] = row
        text.append(f'{s:>6g}mm{row["r2_Sy_y"]:10.3f}' + ''.join(f'{row[m][k]:18.3f}' for m in ('ALL3', 'M1')
                                                            for k in ('r2_p_y_ev', 'r2_Sp_Sy', 'r2_p_Sy', 'r2_fine', 'coarse_err_share')))

    # (b) 标签周期性
    y = [cv[c]['osi'] for c in cases]
    ysh = [out[c]['osi_shift'] for c in cases]
    fl = [i for i, c in enumerate(cases) if c in FLAGGED]; ok = [i for i, c in enumerate(cases) if c not in FLAGGED]
    lab = {'r2_shift_all': r2cb(y, ysh), 'r2_shift_flagged': r2cb([y[i] for i in fl], [ysh[i] for i in fl]),
           'r2_shift_converged': r2cb([y[i] for i in ok], [ysh[i] for i in ok]), 'n_flagged': len(fl)}
    per = {}
    for m in ('ALL3', 'M1'):
        per[m] = np.array([M.r2_score(cv[c]['osi'], cv[c][m]) for c in cases])
    lt = [np.log(np.maximum(cv[c]['tawss'], FLOOR)) for c in cases]; lp = [np.log(np.maximum(cv[c]['M1_tawss'], FLOOR)) for c in cases]
    per['M1_tawss_ln'] = np.array([M.r2_score(a, b) for a, b in zip(lt, lp)])
    wrap = np.array([WRAP[c] for c in cases])
    lab['per_case'] = {}
    for m, v in per.items():
        lab['per_case'][m] = {'spearman_vs_wrap': float(spearmanr(v, wrap).correlation),
                              'mean_converged': float(v[ok].mean()), 'mean_flagged': float(v[fl].mean()),
                              'gap': float(v[ok].mean() - v[fl].mean())}
    wall = np.concatenate([out[c]['w'] for c in cases])
    rows = []
    for m in ('ALL3', 'M1'):
        # 病例等权：每例点等权后病例平均
        pt_share = np.zeros(len(W_BINS) - 1); sse_share = np.zeros_like(pt_share)
        for c in cases:
            e2 = (cv[c][m] - cv[c]['osi']) ** 2; b = np.digitize(out[c]['w'], W_BINS) - 1
            pt_share += np.bincount(b, minlength=len(pt_share))[:len(pt_share)] / len(b)
            sse_share += np.bincount(b, weights=e2, minlength=len(pt_share))[:len(pt_share)] / e2.sum()
        rows.append((m, pt_share / len(cases), sse_share / len(cases)))
    lab_shift_by_w = []
    for j in range(len(W_BINS) - 1):
        vals = [np.abs(out[c]['osi_shift'] - cv[c]['osi'])[(out[c]['w'] >= W_BINS[j]) & (out[c]['w'] < W_BINS[j + 1])] for c in cases]
        vals = np.concatenate(vals)
        lab_shift_by_w.append(float(vals.mean()) if vals.size else float('nan'))
    lab['w_bins'] = {'edges': [float(x) for x in W_BINS], 'pooled_point_share': np.histogram(wall, W_BINS)[0].tolist(),
                     'case_point_share': rows[0][1].tolist(), 'sse_share_ALL3': rows[0][2].tolist(), 'sse_share_M1': rows[1][2].tolist(),
                     'label_shift_abs_mean': lab_shift_by_w}
    res['label'] = lab
    text += ['', '[(b) 标签周期性]',
             f"平移一帧标签 OSI'(帧1–80) 对 OSI(帧0–79) R²_cb：全体 {lab['r2_shift_all']:.3f}｜已收敛 {lab['r2_shift_converged']:.3f}｜未收敛({len(fl)} 例) {lab['r2_shift_flagged']:.3f}（标签噪声下界）",
             '逐例 R² 与病例级 period_wrap：']
    for m, d in lab['per_case'].items():
        text.append(f"  {m:12s} Spearman {d['spearman_vs_wrap']:+.3f}｜已收敛均值 {d['mean_converged']:.3f}｜未收敛均值 {d['mean_flagged']:.3f}｜差 {d['gap']:+.3f}")
    text.append('点级非周期性 w 分箱（病例等权）：' + ' '.join(f'[{W_BINS[j]:g},{W_BINS[j + 1]:g})' for j in range(len(W_BINS) - 1)))
    text.append('  点份额     ' + ' '.join(f'{v:8.3f}' for v in rows[0][1]))
    text.append('  SSE份额ALL3' + ' '.join(f'{v:8.3f}' for v in rows[0][2]))
    text.append('  SSE份额M1  ' + ' '.join(f'{v:8.3f}' for v in rows[1][2]))
    text.append("  |OSI'−OSI| " + ' '.join(f'{v:8.4f}' for v in lab_shift_by_w))

    # (c) 标定曲线
    cal = {}
    text += ['', '[(c) 标定：按真值 OSI 档的预测均值 / 按预测档的真值均值（点池化）]']
    yy = np.concatenate([cv[c]['osi'] for c in cases])
    for m in ('ALL3', 'M1'):
        pp = np.concatenate([cv[c][m] for c in cases])
        bt = np.digitize(yy, OSI_BINS) - 1
        by_true = [{'bin': [OSI_BINS[j], OSI_BINS[j + 1]], 'share': float((bt == j).mean()), 'true_mean': float(yy[bt == j].mean()),
                    'pred_mean': float(pp[bt == j].mean()), 'pred_gt_0.3': float((pp[bt == j] > 0.3).mean())} for j in range(len(OSI_BINS) - 1)]
        bp = np.digitize(pp, PRED_BINS) - 1
        by_pred = [{'bin': [float(PRED_BINS[j]), float(PRED_BINS[j + 1])], 'share': float((bp == j).mean()), 'true_mean': float(yy[bp == j].mean())}
                   for j in range(len(PRED_BINS) - 1) if (bp == j).any()]
        cal[m] = {'by_true': by_true, 'by_pred': by_pred, 'pred_max_p99': float(np.quantile(pp, .99)), 'true_p99': float(np.quantile(yy, .99))}
        text.append(f'  {m}: 真值 p99 {cal[m]["true_p99"]:.3f} / 预测 p99 {cal[m]["pred_max_p99"]:.3f}')
        for b in by_true:
            text.append(f"    真值 [{b['bin'][0]:.2f},{b['bin'][1]:.2f}) 份额 {b['share']:.3f}  真值均值 {b['true_mean']:.3f}  预测均值 {b['pred_mean']:.3f}  预测>0.3 比例 {b['pred_gt_0.3']:.3f}")
    res['calibration'] = cal

    # (d) SSE 分布（病例等权份额）
    sse = {}
    text += ['', '[(d) SSE 份额（病例内归一后病例平均；点份额对照）]']
    for m in ('ALL3', 'M1'):
        d = {}
        mse_case = np.array([np.mean((cv[c][m] - cv[c]['osi']) ** 2) for c in cases])
        d['cohort_mse_share'] = {g: float(mse_case[[cohort(c) == g for c in cases]].sum() / mse_case.sum()) for g in ('AG', 'AAA', 'ILO')}
        d['cohort_case_share'] = {g: float(np.mean([cohort(c) == g for c in cases])) for g in ('AG', 'AAA', 'ILO')}
        for key, getter, edges_or_names in (('segment', lambda c: out[c]['sem'], SEM),
                                            ('true_osi', lambda c: np.digitize(cv[c]['osi'], OSI_BINS) - 1, OSI_BINS),
                                            ('true_tawss', lambda c: np.digitize(cv[c]['tawss'], TAWSS_BINS) - 1, TAWSS_BINS)):
            nb = len(SEM) if key == 'segment' else len(edges_or_names) - 1
            ps = np.zeros(nb); ss = np.zeros(nb)
            for c in cases:
                b = getter(c); ok_ = (b >= 0) & (b < nb); e2 = (cv[c][m] - cv[c]['osi']) ** 2
                ps += np.bincount(b[ok_], minlength=nb) / len(b)
                ss += np.bincount(b[ok_], weights=e2[ok_], minlength=nb) / e2.sum()
            d[key] = {'point_share': (ps / len(cases)).tolist(), 'sse_share': (ss / len(cases)).tolist()}
        sse[m] = d
        text.append(f'  {m}: 队列 MSE 份额 ' + ' '.join(f"{g} {d['cohort_mse_share'][g]:.3f}(病例 {d['cohort_case_share'][g]:.2f})" for g in ('AG', 'AAA', 'ILO')))
        text.append('    血管段 ' + ' '.join(f"{SEM[j]} {d['segment']['sse_share'][j]:.3f}/{d['segment']['point_share'][j]:.3f}" for j in range(len(SEM))) + '  (SSE/点)')
        text.append('    真值OSI档 ' + ' '.join(f"[{OSI_BINS[j]:g},{OSI_BINS[j + 1]:g}) {d['true_osi']['sse_share'][j]:.3f}/{d['true_osi']['point_share'][j]:.3f}" for j in range(len(OSI_BINS) - 1)))
        text.append('    真值TAWSS档 ' + ' '.join(f"[{TAWSS_BINS[j]:g},{TAWSS_BINS[j + 1]:g}) {d['true_tawss']['sse_share'][j]:.3f}/{d['true_tawss']['point_share'][j]:.3f}" for j in range(len(TAWSS_BINS) - 1)))
    res['sse'] = sse
    text.append(f'\n耗时 {time.time() - t0:.0f}s，{nproc} 进程')
    print('\n'.join(text))
    (OUT / 'p0c_ceiling.txt').write_text('\n'.join(text) + '\n')
    write_json(OUT / 'p0c_ceiling.json', res)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
