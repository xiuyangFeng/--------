"""P0d 体场信息探针（零训练网络；轻量堆叠回归；cv3 折外 136 例）。跟踪 §34.10 第 3 条；预注册见 PREREG.md §P0d。

问题：部署可得的体场预测（PF6_v51 峰值压力、VF6_v51 峰值速度向量，cv3 折外）对 OSI 有没有「现有 OSI/TAWSS/峰值预测之外」的信息？
做法：同一个 HistGradientBoosting 回归器，两套输入：
  base   = M1 的 OSI / ln TAWSS / ln 峰值三通道 + O1 / O2 的 OSI（全部 cv3 折外）
  base+V = base + 体场特征：近壁（最近 8 个内部单元）预测速度 |u_nw|、|u_nw| / 截面平均速率、cos(u_nw, 截面平均速度)（负 = 峰值时近壁反向）、
           预测壁压相对截面中位的偏差、沿程预测壁压梯度 dp/ds（逆压梯度指示）
截面 = 同一 segment 内按 abscissa_norm 分 2 mm 当量的箱（体、壁共用箱）。留出折 k 的回归器只用其余两折拟合（每例抽 3000 点），在留出折全点上评。
判据（预注册）：base+V 对 base 的 OSI R²_cb 增量 < +0.02 或 IoU@0.3 增量 < +0.03 → 不开「体场辅助头 + 周期量」联合训练。
"""
from __future__ import annotations

import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

from p0_common import (RUNS, VIEW, OUT, FOLD_HELD, P, cv_bundle, cv_dir, osi_summary, fold_mean, table, write_json)

VOL_RUNS = RUNS / 'volume_time_20260919'
K_NW = 8
BIN_MM = 2.0
N_FIT = 3000
FLOOR = 0.05
VOL_FEATS = ('u_nw', 'u_nw_rel', 'cos_nw_sec', 'dp_sec', 'dpds')


def section_bins(seg, s_norm, s_max_mm):
    """segment × abscissa 箱号；箱宽 ≈ BIN_MM（abscissa_norm 以 s_max 归一）。"""
    width = BIN_MM / max(float(s_max_mm), 1.0)
    return seg.astype(np.int64) * 100000 + np.floor(s_norm / width).astype(np.int64)


def worker(args):
    cid, k = args
    with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
        wxyz = z['wall_coords_aligned_mm'].astype(np.float64)
        wseg, ws = z['wall_segment_id'], z['wall_abscissa_norm'].astype(np.float64)
    with np.load(VIEW / cid / 'volume.npz', allow_pickle=False) as z:
        vxyz = z['vol_coords_aligned_mm'].astype(np.float64)
        vseg, vs = z['vol_segment_id'], z['vol_abscissa_norm'].astype(np.float64)
        s_max = float(z['s_max_mm'])
    with np.load(VOL_RUNS / f'PF6_v51_f{k}_s1234' / P / cid / 'predictions.npz') as z:
        nw = int(z['n_wall'])
        if nw != len(wxyz) or not np.array_equal(z['query_idx'][:nw], np.arange(nw)):
            raise ValueError(f'{cid}: PF6 wall rows differ')
        p_wall = z['pred_raw'][:nw].astype(np.float64)
    with np.load(VOL_RUNS / f'VF6_v51_f{k}_s1234' / P / cid / 'predictions.npz') as z:
        if not np.array_equal(z['query_idx'], np.arange(nw, nw + len(vxyz))):
            raise ValueError(f'{cid}: VF6 rows differ')
        u = z['pred_raw'].astype(np.float64)
    _, nb = cKDTree(vxyz).query(wxyz, k=K_NW)
    u_nw = u[nb].mean(axis=1)
    speed_nw = np.linalg.norm(u[nb], axis=2).mean(axis=1)
    wb = section_bins(wseg, ws, s_max); vb = section_bins(vseg, vs, s_max)
    keys, inv = np.unique(vb, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    usec = np.stack([np.bincount(inv, weights=u[:, j]) for j in range(3)], axis=1) / cnt[:, None]
    ssec = np.bincount(inv, weights=np.linalg.norm(u, axis=1)) / cnt
    pos = np.searchsorted(keys, wb); pos = np.clip(pos, 0, len(keys) - 1)
    hit = keys[pos] == wb
    # 壁面箱没有内部单元（极少）：退回最近内部单元所在箱
    if not hit.all():
        near = inv[nb[:, 0]]
        pos = np.where(hit, pos, near)
    us, ss = usec[pos], ssec[pos]
    cos = np.einsum('ij,ij->i', u_nw, us) / np.maximum(np.linalg.norm(u_nw, axis=1) * np.linalg.norm(us, axis=1), 1e-9)
    # 壁压：截面中位偏差 + 沿程梯度（同 segment 内箱中位对箱中心 abscissa 求导）
    wkeys, winv = np.unique(wb, return_inverse=True)
    med = np.array([np.median(p_wall[winv == i]) for i in range(len(wkeys))])
    dp_sec = p_wall - med[winv]
    dpds_bin = np.zeros(len(wkeys))
    segs = wkeys // 100000
    for sid in np.unique(segs):
        m = np.flatnonzero(segs == sid)
        if len(m) >= 3:
            dpds_bin[m] = np.gradient(med[m], (wkeys[m] % 100000) * BIN_MM)
    feats = np.column_stack([speed_nw, speed_nw / np.maximum(ss, 1e-6), cos, dp_sec, dpds_bin[winv]])
    return cid, feats.astype(np.float32)


def main():
    t0 = time.time()
    cv = cv_bundle()
    # M1 峰值通道：只核行序（其真值是峰值帧 WSS，不是周期视图的量）
    for c in cv:
        k = cv[c]['fold']
        with np.load(cv_dir('M1', k, 'wss') / c / 'predictions.npz') as z:
            if not np.array_equal(z['row_index'], np.arange(len(cv[c]['osi']))):
                raise ValueError(f'{c}: M1 wss rows differ')
            cv[c]['M1_wss'] = z['pred_pa'].astype(np.float64)
    jobs = [(c, cv[c]['fold']) for c in cv]
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        vol = dict(ex.map(worker, jobs, chunksize=1))
    print(f'volume features {len(vol)} cases {time.time() - t0:.0f}s', flush=True)

    def base_X(c):
        d = cv[c]
        return np.column_stack([d['M1'], np.log(np.maximum(d['M1_tawss'], FLOOR)), np.log(np.maximum(d['M1_wss'], FLOOR)), d['O1'], d['O2']])

    sets = {'base': lambda c: base_X(c), 'base+V': lambda c: np.column_stack([base_X(c), vol[c]])}
    res, importances = {}, {}
    for name, fx in sets.items():
        per_fold = []
        for k in range(3):
            train = [c for kk in range(3) if kk != k for c in FOLD_HELD[kk]]
            Xs, ys = [], []
            for c in train:
                rng = np.random.default_rng(zlib.crc32(c.encode()))
                X = fx(c); pick = rng.choice(len(X), min(N_FIT, len(X)), replace=False)
                Xs.append(X[pick]); ys.append(cv[c]['osi'][pick])
            reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                                l2_regularization=1.0, early_stopping=False, random_state=20260924)
            reg.fit(np.concatenate(Xs), np.concatenate(ys))
            held = FOLD_HELD[k]
            pred = [np.clip(reg.predict(fx(c)), 0.0, 0.5) for c in held]
            per_fold.append(osi_summary(held, [cv[c]['osi'] for c in held], pred, [cv[c]['tawss'] for c in held], [cv[c]['M1_tawss'] for c in held]))
        res[name] = {'per_fold': per_fold, 'fold_mean': fold_mean(per_fold)}
        print(f'{name} done {time.time() - t0:.0f}s', flush=True)

    # 体场特征与 OSI 真值 / 与 base 残差的相关（病例内 Spearman 中位）
    from scipy.stats import spearmanr
    corr = {}
    for j, f in enumerate(VOL_FEATS):
        r_y, r_res = [], []
        for c in cv:
            x = vol[c][:, j].astype(np.float64); y = cv[c]['osi']; resid = y - cv[c]['M1']
            if np.std(x) > 0:
                r_y.append(spearmanr(x, y).correlation); r_res.append(spearmanr(x, resid).correlation)
        corr[f] = {'spearman_vs_osi_casemed': float(np.median(r_y)), 'spearman_vs_M1_residual_casemed': float(np.median(r_res))}

    text = ['=== P0d 体场信息探针（OSI；堆叠回归，cv3 折外三折留出均值）===', '',
            table([(n, res[n]['fold_mean']) for n in sets], ref='base'), '',
            '体场特征：病例内 Spearman 中位（对 OSI 真值 / 对 M1 残差 y − ŷ）：']
    for f, d in corr.items():
        text.append(f"  {f:12s} {d['spearman_vs_osi_casemed']:+.3f} / {d['spearman_vs_M1_residual_casemed']:+.3f}")
    dr2 = res['base+V']['fold_mean']['r2_cb'] - res['base']['fold_mean']['r2_cb']
    diou = res['base+V']['fold_mean']['iou_0.3'] - res['base']['fold_mean']['iou_0.3']
    verdict = '不开联合训练（增量低于预注册门槛）' if (dr2 < 0.02 or diou < 0.03) else '过门：值得评估体场辅助头'
    text += ['', f'ΔR²_cb {dr2:+.4f}，ΔIoU@0.3 {diou:+.4f} → {verdict}', f'耗时 {time.time() - t0:.0f}s，{nproc} 进程']
    print('\n'.join(text))
    (OUT / 'p0d_volume_probe.txt').write_text('\n'.join(text) + '\n')
    write_json(OUT / 'p0d_volume_probe.json', {'results': res, 'feature_corr': corr, 'delta_r2_cb': dr2, 'delta_iou_0.3': diou, 'verdict': verdict})


if __name__ == '__main__':
    main()
