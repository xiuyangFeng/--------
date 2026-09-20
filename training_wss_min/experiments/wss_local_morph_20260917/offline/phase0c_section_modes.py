"""Phase 0c：截面**几何**的低阶周向形变，能不能解释残差的周向模态（只读）。

D1 已证明区间内残差有 m=1..3 的真实周向结构（净 0.589），但 D2/D3 证明它不对齐任何固定
解剖方向。本检验换一个假设：**结构由截面自身的形状决定**。

对每个 (分支 × 4 mm 区间)，用壁面点自己的 (θ, r) 拟合截面半径轮廓
    r(θ) ≈ c0 · [1 + Σ_{m=1..3} A_m cos(m θ − φ_m)]
A_m 是无量纲形变幅值，φ_m 是该模态的鼓出方位。逐点特征 = A_m 与 cos(mθ − φ_m)。
然后看这些**纯几何**特征能解释多少区间内残差方差——与 D2 的 0.007 直接可比。
r 由 bundle 的 rho × local_radius 还原，θ 用 bundle 的 theta_rad，全部部署可得。
"""
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
CAS = ROOT / 'data_wss_v5/views_v5_1/wss_min_cascade_v1'
BASE = ROOT / 'training_wss_min/runs/wss_v51_wave1_20260916'
OUT = Path(__file__).resolve().parent
BIN_MM, MIN_PTS, MODES = 4.0, 40, (1, 2, 3)
SEEDS5 = (1234, 7, 2025, 11, 2026)
P = 'eval/ckpt_best/predictions/test'
split = json.loads((VIEW / 'split_V5_train136_test34.json').read_text())


def load(cid, partition):
    with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
        hit = np.flatnonzero(z['steps'] == int(z['peak_step']))
        true = z['wall_wss'][int(hit[0])].astype(np.float64)
        th = z['wall_theta_rad'].astype(np.float64)
        seg = z['wall_segment_id'].astype(int)
        sl = z['wall_s_local_mm'].astype(np.float64)
        r_mm = z['wall_rho'].astype(np.float64) * z['wall_local_radius'].astype(np.float64)
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
    return np.log(np.maximum(true, 1e-6)) - lp, th, seg, sl, r_mm


def section_modes(th, r):
    """Fit r(θ) = c0 (1 + Σ A_m cos(mθ − φ_m)); return c0, A, φ and the unexplained radial part."""
    cols = [np.ones(len(th))]
    for m in MODES:
        cols += [np.cos(m * th), np.sin(m * th)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    c0 = coef[0]
    if not np.isfinite(c0) or c0 <= 1e-6:
        return None
    amp, pha = {}, {}
    for i, m in enumerate(MODES):
        a, b = coef[1 + 2 * i], coef[2 + 2 * i]
        amp[m] = float(np.hypot(a, b) / c0)
        pha[m] = float(np.arctan2(b, a))
    return c0, amp, pha, (r - A @ coef) / c0


def analyse(cid, partition):
    r, th, seg, sl, r_mm = load(cid, partition)
    key = seg.astype(np.int64) * 100000 + np.floor(sl / BIN_MM).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    pool_r, pool_x, amps = [], [], {m: [] for m in MODES}
    for b in range(len(uk)):
        m_ = inv == b
        if m_.sum() < MIN_PTS:
            continue
        got = section_modes(th[m_], r_mm[m_])
        if got is None:
            continue
        c0, amp, pha, dev = got
        rl = r[m_] - r[m_].mean()
        feats = []
        for m in MODES:
            c = np.cos(m * th[m_] - pha[m]); s = np.sin(m * th[m_] - pha[m])
            feats += [c, s, amp[m] * c, amp[m] * s]
            amps[m].append(amp[m])
        feats.append(dev)                      # 低阶模态之外的局部凹凸
        pool_r.append(rl); pool_x.append(np.column_stack(feats))
    if not pool_r:
        return None
    y = np.concatenate(pool_r); X = np.vstack(pool_x)
    if len(y) < 200 or y.var() <= 0:
        return None
    def r2(cols):
        A = np.column_stack([X[:, cols], np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        e = y - A @ coef
        return float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
    n_per = 4
    idx_all = list(range(len(MODES) * n_per + 1))
    idx_m1 = list(range(0, n_per))
    idx_dev = [len(MODES) * n_per]
    # 随机相位对照：把每个 bin 的模态相位打乱再拟合，给同参数量的偶然解释率
    rng = np.random.default_rng(0)
    Xn = X.copy()
    rng.shuffle(Xn)
    An = np.column_stack([Xn[:, idx_all], np.ones(len(y))])
    cn, *_ = np.linalg.lstsq(An, y, rcond=None)
    null = float(1 - ((y - An @ cn) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return dict(canonical_id=cid, partition=partition, n_points=int(len(y)),
                r2_all=r2(idx_all), r2_m1=r2(idx_m1), r2_dev=r2(idx_dev), r2_null=null,
                **{f'amp_m{m}': float(np.median(amps[m])) for m in MODES})


def report(rows, tag):
    print(f'\n=== {tag}（{len(rows)} 例，中位数）')
    for k, label in (('r2_all', '截面模态全套（13 列）解释区间内残差 R²'),
                     ('r2_m1', '仅 m=1 的 4 列'),
                     ('r2_dev', '仅"低阶之外的局部凹凸" 1 列'),
                     ('r2_null', '同参数量随机对照')):
        v = np.array([x[k] for x in rows])
        print(f'  {label:36s} {np.median(v):7.4f}   [p10 {np.quantile(v,.1):7.4f}, p90 {np.quantile(v,.9):7.4f}]')
    for m in MODES:
        v = np.array([x[f'amp_m{m}'] for x in rows])
        print(f'  截面 m={m} 形变幅值 A_m（中位）        {np.median(v):7.4f}')
    net = np.array([x['r2_all'] - x['r2_null'] for x in rows])
    print(f'  **净解释力（减随机对照）             {np.median(net):+7.4f}**   对照 D2 曲率锚定 0.0069')


def main():
    out = []
    for partition, cases in (('train', split['train_cases']), ('test', split['test_cases'])):
        for i, cid in enumerate(cases):
            row = analyse(cid, partition)
            if row:
                out.append(row)
            if (i + 1) % 20 == 0:
                print(f'  {partition} {i+1}/{len(cases)}', flush=True)
    report([x for x in out if x['partition'] == 'train'], 'train136 折外')
    report([x for x in out if x['partition'] == 'test'], 'test34 五 seed 集成')
    (OUT / 'phase0c_section_modes.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print('\nwrote', OUT / 'phase0c_section_modes.json')


if __name__ == '__main__':
    main()
