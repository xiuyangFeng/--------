"""Phase 0d：区间内残差的周向结构是不是"沿流向被输运"的（只读）。

把壁面展开成柱面坐标 u = (s_local, R̄cosθ, R̄sinθ)，对每个点取其**上游 L mm 处同周向相位**
的最近邻，算残差相关；对照组取**上游 L mm 处反相位（+180°）**的最近邻。

同相位相关随 L 衰减得慢、且显著高于反相位 → 结构沿流向输运，杠杆是"保留周向相位的上游条件"。
两者一起快速衰减 → 只是局部平滑，没有可用的上游结构。
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
CAS = ROOT / 'data_wss_v5/views_v5_1/wss_min_cascade_v1'
BASE = ROOT / 'training_wss_min/runs/wss_v51_wave1_20260916'
OUT = Path(__file__).resolve().parent
BIN_MM, MIN_PTS = 4.0, 40
LAGS = (2.0, 5.0, 10.0, 20.0)
SEEDS5 = (1234, 7, 2025, 11, 2026)
P = 'eval/ckpt_best/predictions/test'
split = json.loads((VIEW / 'split_V5_train136_test34.json').read_text())


def load(cid, partition):
    with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
        hit = np.flatnonzero(z['steps'] == int(z['peak_step']))
        true = z['wall_wss'][int(hit[0])].astype(np.float64)
        th = z['wall_theta_rad'].astype(np.float64); seg = z['wall_segment_id'].astype(int)
        sl = z['wall_s_local_mm'].astype(np.float64)
        rad = z['wall_local_radius'].astype(np.float64)
        ids = z['wall_node_id_cas']
    if partition == 'train':
        with np.load(CAS / cid / 'features.npz', allow_pickle=False) as z:
            cid_ids, base = z['wall_node_id_cas'], z['wall_log_wss_base'].astype(np.float64)
        if not np.array_equal(ids, cid_ids):
            o = {v: i for i, v in enumerate(cid_ids)}
            base = base[np.array([o[v] for v in ids])]
        lp = base
    else:
        preds = [np.load(BASE / f'X5D_v51_s{s}' / P / cid / 'predictions.npz')['pred_pa'] for s in SEEDS5]
        lp = np.log(np.maximum(np.mean(preds, axis=0), 1e-6))
    return np.log(np.maximum(true, 1e-6)) - lp, th, seg, sl, rad


def analyse(cid, partition):
    r, th, seg, sl, rad = load(cid, partition)
    # 只看区间内部分量（去掉 4mm 区间均值），与 D1 口径一致
    key = seg.astype(np.int64) * 100000 + np.floor(sl / BIN_MM).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    rl = r.copy()
    ok = np.zeros(len(r), dtype=bool)
    for b in range(len(uk)):
        m_ = inv == b
        if m_.sum() < MIN_PTS:
            continue
        rl[m_] = r[m_] - r[m_].mean(); ok[m_] = True
    row = dict(canonical_id=cid, partition=partition, n=int(ok.sum()))
    for sid in (None,):
        pass
    out_same = {L: [] for L in LAGS}; out_opp = {L: [] for L in LAGS}
    for sid in np.unique(seg):
        m_ = ok & (seg == sid)
        if m_.sum() < 200:
            continue
        R = float(np.median(rad[m_]))
        u = np.column_stack([sl[m_], R * np.cos(th[m_]), R * np.sin(th[m_])])
        tree = cKDTree(u)
        v = rl[m_]
        for L in LAGS:
            for tag, sign in (('same', 1.0), ('opp', -1.0)):
                q = np.column_stack([sl[m_] - L, sign * R * np.cos(th[m_]), sign * R * np.sin(th[m_])])
                d, j = tree.query(q, k=1)
                good = d <= max(1.5, 0.15 * R)          # 找不到足够近的配对就丢弃
                if good.sum() < 100:
                    continue
                a, b = v[good], v[j[good]]
                if a.std() > 0 and b.std() > 0:
                    (out_same if tag == 'same' else out_opp)[L].append(
                        (float(np.corrcoef(a, b)[0, 1]), int(good.sum())))
    for L in LAGS:
        for tag, acc in (('same', out_same), ('opp', out_opp)):
            vals = acc[L]
            row[f'{tag}_{L:g}'] = (float(np.average([x[0] for x in vals], weights=[x[1] for x in vals]))
                                   if vals else float('nan'))
    return row


def report(rows, tag):
    print(f'\n=== {tag}（{len(rows)} 例，中位数）')
    print(f'  {"上游滞后":10s} {"同相位 corr":>12s} {"反相位 corr":>12s} {"差":>8s}')
    for L in LAGS:
        a = np.array([x[f'same_{L:g}'] for x in rows], dtype=float); a = a[np.isfinite(a)]
        b = np.array([x[f'opp_{L:g}'] for x in rows], dtype=float); b = b[np.isfinite(b)]
        print(f'  {L:6.0f} mm   {np.median(a):12.4f} {np.median(b):12.4f} {np.median(a)-np.median(b):+8.4f}')


def main():
    out = []
    for partition, cases in (('train', split['train_cases']), ('test', split['test_cases'])):
        for i, cid in enumerate(cases):
            out.append(analyse(cid, partition))
            if (i + 1) % 30 == 0:
                print(f'  {partition} {i+1}/{len(cases)}', flush=True)
    report([x for x in out if x['partition'] == 'train'], 'train136 折外')
    report([x for x in out if x['partition'] == 'test'], 'test34 五 seed 集成')
    (OUT / 'phase0d_advection.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print('\nwrote', OUT / 'phase0d_advection.json')


if __name__ == '__main__':
    main()
