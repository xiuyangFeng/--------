"""Phase 0b：D1 的零假设校验 + D2/D3 解剖锚定检验（只读）。

D1-null  每个 bin 内把 θ 随机置换后重拟同样的 m=1..3，给偶然解释率；真实份额必须显著高于它。
D2       把 θ 换成"相对局部曲率平面"的 Δθ_bend（内弯 = 0），检验区间内残差的 m=1 方向是否因此变得一致：
         (a) m=1 相位在 RMF 坐标 vs 锚定坐标下的集中度 R；
         (b) 逐例 pooled 回归 r_local ~ [cosΔθ_bend, sinΔθ_bend] 的 R² 与 cos 系数符号一致性
             （cos 系数 > 0 = 内弯侧被系统性低估）。按 κR 分层。
D3       同上，用分叉锚定 Δθ_apex，只在 apex_valid 点上。
"""
import json, sys
import numpy as np
import h5py
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from wss_v5.views import wall_local_phase_v1 as LP

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
CAS = ROOT / 'data_wss_v5/views_v5_1/wss_min_cascade_v1'
H5 = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases'
BASE = ROOT / 'training_wss_min/runs/wss_v51_wave1_20260916'
OUT = Path(__file__).resolve().parent
BIN_MM, MIN_PTS, N_PERM = 4.0, 40, 5
KR_HI = 0.15
SEEDS5 = (1234, 7, 2025, 11, 2026)
P = 'eval/ckpt_best/predictions/test'
split = json.loads((VIEW / 'split_V5_train136_test34.json').read_text())


def circ_R(angles, w=None):
    c = np.average(np.cos(angles), weights=w); s = np.average(np.sin(angles), weights=w)
    return float(np.hypot(c, s)), float(np.arctan2(s, c))


def load(cid, partition):
    with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
        hit = np.flatnonzero(z['steps'] == int(z['peak_step']))
        true = z['wall_wss'][int(hit[0])].astype(np.float64)
        th = z['wall_theta_rad'].astype(np.float64); seg = z['wall_segment_id'].astype(int)
        sl = z['wall_s_local_mm'].astype(np.float64); lr = z['wall_local_radius'].astype(np.float64)
        ids = z['wall_node_id_cas']
    if partition == 'train':
        with np.load(CAS / cid / 'features.npz', allow_pickle=False) as z:
            cid_ids, base = z['wall_node_id_cas'], z['wall_log_wss_base'].astype(np.float64)
        if not np.array_equal(ids, cid_ids):
            o = {v: i for i, v in enumerate(cid_ids)}
            base = base[np.array([o[v] for v in ids])]
        lp = base
    else:
        preds = []
        for s in SEEDS5:
            with np.load(BASE / f'X5D_v51_s{s}' / P / cid / 'predictions.npz') as z:
                preds.append(z['pred_pa'])
        lp = np.log(np.maximum(np.mean(preds, axis=0), 1e-6))
    with h5py.File(H5 / cid.replace('/', '__') / 'case.h5', 'r') as f:
        feats = LP.build(LP.read_atlas(f), th, seg, sl, lr)
    return dict(r=np.log(np.maximum(true, 1e-6)) - lp, theta=th, seg=seg, s=sl, ph=feats)


def modes(rl, th, rng, perms=0):
    v = rl.var()
    if v <= 0:
        return None
    share, null = {}, {}
    for m in (1, 2, 3):
        A = np.column_stack([np.cos(m * th), np.sin(m * th)])
        coef, *_ = np.linalg.lstsq(A, rl, rcond=None)
        share[m] = float(((A @ coef) ** 2).mean() / v)
        if m == 1:
            share['phase'] = float(np.arctan2(coef[1], coef[0]))
        if perms:
            acc = []
            for _ in range(perms):
                tp = rng.permutation(th)
                B = np.column_stack([np.cos(m * tp), np.sin(m * tp)])
                cf, *_ = np.linalg.lstsq(B, rl, rcond=None)
                acc.append(((B @ cf) ** 2).mean() / v)
            null[m] = float(np.mean(acc))
    return share, null


def pooled_dir(rl_all, cos_all, sin_all):
    """One global [cos, sin] regression of the within-bin residual on an anchored direction."""
    if len(rl_all) < 50 or np.var(rl_all) <= 0:
        return None
    A = np.column_stack([cos_all, sin_all, np.ones(len(rl_all))])
    coef, *_ = np.linalg.lstsq(A, rl_all, rcond=None)
    e = rl_all - A @ coef
    return dict(r2=float(1 - (e ** 2).sum() / ((rl_all - rl_all.mean()) ** 2).sum()),
                a_cos=float(coef[0]), a_sin=float(coef[1]), n=int(len(rl_all)))


def analyse(cid, partition, rng):
    a = load(cid, partition)
    r, th, seg, s, ph = a['r'], a['theta'], a['seg'], a['s'], a['ph']
    d_bend = th - np.arctan2(ph['bend_sin'].astype(np.float64), ph['bend_cos'].astype(np.float64))
    # ph['bend_cos/sin'] 已经是 cos/sin(θ − θ_bend)，直接用即可
    cb, sb = ph['bend_cos'].astype(np.float64), ph['bend_sin'].astype(np.float64)
    ca, sa = ph['apex_cos'].astype(np.float64), ph['apex_sin'].astype(np.float64)
    kr, av = ph['bend_kr'].astype(np.float64), ph['apex_valid'] > 0.5
    key = seg.astype(np.int64) * 100000 + np.floor(s / BIN_MM).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    sh = {1: [], 2: [], 3: []}; nl = {1: [], 2: [], 3: []}
    ph_rmf, ph_bend, wts = [], [], []
    pool = {k: [] for k in ('r', 'cb', 'sb', 'ca', 'sa', 'kr', 'av')}
    for b in range(len(uk)):
        m_ = inv == b
        if m_.sum() < MIN_PTS:
            continue
        rl = r[m_] - r[m_].mean()
        got = modes(rl, th[m_], rng, perms=N_PERM)
        if got is None:
            continue
        share, null = got
        for m in (1, 2, 3):
            sh[m].append(share[m]); nl[m].append(null[m])
        ph_rmf.append(share['phase'])
        # 锚定坐标下重拟 m=1 的相位
        dth = np.arctan2(sb[m_], cb[m_])
        got2 = modes(rl, dth, rng)
        ph_bend.append(got2[0]['phase'])
        wts.append(int(m_.sum()))
        for k, v in (('r', rl), ('cb', cb[m_]), ('sb', sb[m_]),
                     ('ca', ca[m_]), ('sa', sa[m_]), ('kr', kr[m_]), ('av', av[m_])):
            pool[k].append(v)
    if not wts:
        return None
    pool = {k: np.concatenate(v) for k, v in pool.items()}
    hi = pool['kr'] >= KR_HI
    row = dict(canonical_id=cid, partition=partition, n_bins=len(wts),
               phase_R_rmf=circ_R(np.array(ph_rmf), wts)[0],
               phase_R_bend=circ_R(np.array(ph_bend), wts)[0],
               bend_kr_hi_fraction=float(hi.mean()), apex_fraction=float(pool['av'].mean()))
    for m in (1, 2, 3):
        row[f'm{m}'] = float(np.average(sh[m], weights=wts))
        row[f'm{m}_null'] = float(np.average(nl[m], weights=wts))
    row['bend_all'] = pooled_dir(pool['r'], pool['cb'], pool['sb'])
    row['bend_hi'] = pooled_dir(pool['r'][hi], pool['cb'][hi], pool['sb'][hi]) if hi.sum() > 50 else None
    m_ap = pool['av']
    row['apex'] = pooled_dir(pool['r'][m_ap], pool['ca'][m_ap], pool['sa'][m_ap]) if m_ap.sum() > 50 else None
    return row


def report(rows, tag):
    print(f'\n=== {tag}（{len(rows)} 例，中位数)')
    for m in (1, 2, 3):
        v = np.array([x[f'm{m}'] for x in rows]); n = np.array([x[f'm{m}_null'] for x in rows])
        print(f'  m={m} 实际 {np.median(v):.4f}   零假设 {np.median(n):.4f}   净 {np.median(v - n):+.4f}')
    tot = np.array([sum(x[f'm{m}'] for m in (1, 2, 3)) for x in rows])
    totn = np.array([sum(x[f'm{m}_null'] for m in (1, 2, 3)) for x in rows])
    print(f'  m=1..3 合计 实际 {np.median(tot):.4f}   零假设 {np.median(totn):.4f}   **净 {np.median(tot - totn):+.4f}**')
    print(f'  m=1 相位集中度 R：RMF {np.median([x["phase_R_rmf"] for x in rows]):.4f}'
          f'   曲率锚定 {np.median([x["phase_R_bend"] for x in rows]):.4f}')
    for key, label in (('bend_all', 'D2 曲率锚定方向 pooled 回归（全部点）'),
                       ('bend_hi', f'D2 同上，仅 κR ≥ {KR_HI}'),
                       ('apex', 'D3 分叉锚定方向 pooled 回归（apex 有效点）')):
        got = [x[key] for x in rows if x.get(key)]
        if not got:
            print(f'  {label}: 无足够样本'); continue
        r2 = np.array([g['r2'] for g in got]); ac = np.array([g['a_cos'] for g in got])
        asn = np.array([g['a_sin'] for g in got])
        print(f'  {label}: R² 中位 {np.median(r2):.4f}  a_cos 中位 {np.median(ac):+.4f}'
              f'（{int((ac > 0).sum())}/{len(ac)} 为正）  a_sin 中位 {np.median(asn):+.4f}'
              f'（{int((asn > 0).sum())}/{len(asn)} 为正）')


def main():
    rng = np.random.default_rng(1234)
    out = []
    for partition, cases in (('train', split['train_cases']), ('test', split['test_cases'])):
        for i, cid in enumerate(cases):
            row = analyse(cid, partition, rng)
            if row:
                out.append(row)
            if (i + 1) % 20 == 0:
                print(f'  {partition} {i+1}/{len(cases)}', flush=True)
    report([x for x in out if x['partition'] == 'train'], 'train136 折外')
    report([x for x in out if x['partition'] == 'test'], 'test34 五 seed 集成')
    (OUT / 'phase0b_anchored_modes.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print('\nwrote', OUT / 'phase0b_anchored_modes.json')


if __name__ == '__main__':
    main()
