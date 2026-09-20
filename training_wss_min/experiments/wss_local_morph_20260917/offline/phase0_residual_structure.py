"""Phase 0：局部形态残差的空间结构诊断（只读，不训练不推理）。

口径：残差 r = ln(true_Pa) − ln(pred_Pa)。
  train136 用 wss_min_cascade_v1 的逐点**折外**预测（cv3_v51 三折，seed 1234）。
  test34 用 X5D_v51 五 seed 的 Pa 均值集成。
分箱 = (segment_id, floor(s_local_mm / 4mm))，与 §28.9 的"分支 × 4 mm 区间"一致。

D1 区间内残差按 θ 的低阶傅里叶模态份额（m=1/2/3）与相位在分支内的一致性
D4 区间内残差对"去均值预测值"的回归斜率 β：β<0 表示型态对但被压缩（幅值问题），
   β≈0 表示型态本身错（结构问题）
D5 真值/预测 top10% 的质心位移，分解到轴向(s) 与周向(θ·R)
D6 K16 query patch 的实际毫米跨度与周向跨度 ÷ 局部半径
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
CAS = ROOT / 'data_wss_v5/views_v5_1/wss_min_cascade_v1'
RUNS = ROOT / 'training_wss_min/runs'
BASE = RUNS / 'wss_v51_wave1_20260916'
OUT = Path(__file__).resolve().parent
BIN_MM = 4.0
MIN_PTS = 40
SEEDS5 = (1234, 7, 2025, 11, 2026)
P = 'eval/ckpt_best/predictions/test'
split = json.loads((VIEW / 'split_V5_train136_test34.json').read_text())


def circ_mean(theta, w=None):
    c = np.average(np.cos(theta), weights=w); s = np.average(np.sin(theta), weights=w)
    return np.arctan2(s, c), np.hypot(c, s)


def case_arrays(cid, partition):
    with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
        steps = z['steps']
        hit = np.flatnonzero(steps == int(z['peak_step']))
        if len(hit) != 1:
            raise ValueError(f'{cid}: peak_step not found in steps')
        true = z['wall_wss'][int(hit[0])].astype(np.float64)
        theta = z['wall_theta_rad'].astype(np.float64)
        seg = z['wall_segment_id'].astype(int)
        s_loc = z['wall_s_local_mm'].astype(np.float64)
        rad = z['wall_local_radius'].astype(np.float64)
        pos = z['wall_coords_aligned_mm'].astype(np.float64)
        ids = z['wall_node_id_cas']
    if partition == 'train':
        with np.load(CAS / cid / 'features.npz', allow_pickle=False) as z:
            cid_ids, base = z['wall_node_id_cas'], z['wall_log_wss_base'].astype(np.float64)
        if not np.array_equal(ids, cid_ids):
            order = {v: i for i, v in enumerate(cid_ids)}
            base = base[np.array([order[v] for v in ids])]
        lp = base
    else:
        preds = []
        for s in SEEDS5:
            with np.load(BASE / f'X5D_v51_s{s}' / P / cid / 'predictions.npz') as z:
                preds.append(z['pred_pa']); rows = z['row_index']; tp = z['true_pa'].astype(np.float64)
        if not np.array_equal(rows, np.arange(len(true))):
            raise ValueError(f'{cid}: unexpected prediction row order')
        if not np.allclose(tp, true, rtol=1e-5, atol=1e-6):
            raise ValueError(f'{cid}: bundle peak WSS differs from saved true_pa')
        lp = np.log(np.maximum(np.mean(preds, axis=0), 1e-6))
    lt = np.log(np.maximum(true, 1e-6))
    return dict(r=lt - lp, lt=lt, lp=lp, true=true, theta=theta, seg=seg, s=s_loc, rad=rad, pos=pos)


def analyse(cid, partition, rng):
    a = case_arrays(cid, partition)
    r, theta, seg, s = a['r'], a['theta'], a['seg'], a['s']
    key = seg.astype(np.int64) * 100000 + np.floor(s / BIN_MM).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    tot_var = r.var()
    bin_mean = np.zeros(len(uk)); counts = np.bincount(inv, minlength=len(uk))
    for b in range(len(uk)):
        bin_mean[b] = r[inv == b].mean()
    between = float(np.average((bin_mean - r.mean()) ** 2, weights=counts))
    within = float(tot_var - between)
    modes = {m: [] for m in (1, 2, 3)}
    betas, phases, wts = [], [], []
    for b in range(len(uk)):
        m_ = inv == b
        if counts[b] < MIN_PTS:
            continue
        rl = r[m_] - r[m_].mean(); th = theta[m_]
        v = rl.var()
        if v <= 0:
            continue
        for m in (1, 2, 3):
            A = np.column_stack([np.cos(m * th), np.sin(m * th)])
            coef, *_ = np.linalg.lstsq(A, rl, rcond=None)
            modes[m].append(float(((A @ coef) ** 2).mean() / v))
            if m == 1:
                phases.append(float(np.arctan2(coef[1], coef[0])))
        pl = a['lp'][m_] - a['lp'][m_].mean()
        if pl.var() > 0:
            betas.append(float((pl @ rl) / (pl @ pl)))
        wts.append(int(counts[b]))
    # 相位在同一分支内的一致性（θ 零相位逐例任意，但同一病例内可比）
    phase_R = float(circ_mean(np.array(phases))[1]) if len(phases) > 3 else float('nan')
    # D5 热点位移
    q = np.quantile(a['true'], .9); qp = np.quantile(np.exp(a['lp']), .9)
    mt, mp = a['true'] >= q, np.exp(a['lp']) >= qp
    d_s = float(s[mp].mean() - s[mt].mean())
    th_t, _ = circ_mean(theta[mt]); th_p, _ = circ_mean(theta[mp])
    dth = float(np.arctan2(np.sin(th_p - th_t), np.cos(th_p - th_t)))
    d_circ = float(dth * np.median(a['rad'][mt]))
    # D6 K16 patch 跨度
    idx = rng.permutation(len(s))[:400]
    tree = cKDTree(a['pos'])
    dist, nb = tree.query(a['pos'][idx], k=16)
    span_mm = float(np.median(dist[:, -1]))
    th_span = []
    for i, row in enumerate(nb):
        d = np.arctan2(np.sin(theta[row] - theta[idx[i]]), np.cos(theta[row] - theta[idx[i]]))
        th_span.append(d.max() - d.min())
    circ_frac = float(np.median(np.array(th_span) / (2 * np.pi)))
    return dict(canonical_id=cid, partition=partition, n=len(s), n_bins=int((counts >= MIN_PTS).sum()),
                within_share=float(within / tot_var),
                m1=float(np.average(modes[1], weights=wts)) if wts else float('nan'),
                m2=float(np.average(modes[2], weights=wts)) if wts else float('nan'),
                m3=float(np.average(modes[3], weights=wts)) if wts else float('nan'),
                beta=float(np.average(betas, weights=wts)) if betas else float('nan'),
                phase_concentration=phase_R, hotspot_ds_mm=d_s, hotspot_dcirc_mm=d_circ,
                k16_span_mm=span_mm, k16_circ_fraction=circ_frac,
                median_radius_mm=float(np.median(a['rad'])))


def main():
    rng = np.random.default_rng(1234)
    rows = []
    for partition, cases in (('train', split['train_cases']), ('test', split['test_cases'])):
        for i, cid in enumerate(cases):
            rows.append(analyse(cid, partition, rng))
            if (i + 1) % 20 == 0:
                print(f'  {partition} {i+1}/{len(cases)}', flush=True)
    for partition in ('train', 'test'):
        sub = [x for x in rows if x['partition'] == partition]
        tag = 'train136 折外' if partition == 'train' else 'test34 五 seed 集成'
        print(f'\n=== {tag}（{len(sub)} 例，中位数）')
        for k, label in (('within_share', 'D0 4mm 区间内残差方差份额'),
                         ('m1', 'D1 区间内 m=1 周向模态份额'),
                         ('m2', 'D1 m=2 份额'), ('m3', 'D1 m=3 份额'),
                         ('phase_concentration', 'D1 m=1 相位在病例内的集中度 R'),
                         ('beta', 'D4 区间内残差对去均值预测的斜率 β'),
                         ('hotspot_ds_mm', 'D5 热点轴向位移 (mm)'),
                         ('hotspot_dcirc_mm', 'D5 热点周向位移 (mm)'),
                         ('k16_span_mm', 'D6 K16 patch 半径 (mm)'),
                         ('k16_circ_fraction', 'D6 K16 周向跨度 ÷ 整圈'),
                         ('median_radius_mm', '参考：病例中位局部半径 (mm)')):
            v = np.array([x[k] for x in sub], dtype=float); v = v[np.isfinite(v)]
            print(f'  {label:36s} {np.median(v):8.4f}   [p10 {np.quantile(v,.1):7.4f}, p90 {np.quantile(v,.9):7.4f}]')
        m123 = np.array([x['m1'] + x['m2'] + x['m3'] for x in sub])
        print(f'  {"D1 m=1..3 合计份额":36s} {np.median(m123):8.4f}   [p10 {np.quantile(m123,.1):7.4f}, p90 {np.quantile(m123,.9):7.4f}]')
    (OUT / 'phase0_residual_structure.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print('\nwrote', OUT / 'phase0_residual_structure.json')


if __name__ == '__main__':
    main()
