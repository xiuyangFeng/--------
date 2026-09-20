"""按边际增量给 16 个沿程 value 通道排序（选择只用 train136 折外残差，不碰 test34）。

残差 r = ln(true_Pa) − log_wss_base，其中 log_wss_base 来自 wss_min_cascade_v1
（cv3_v51 三折的逐点折外预测，seed 1234）。先把原 27 维（逐例去均值）回归掉，
再对 (value, mask) 成对做前向逐步选择，报告累计 R² 与每步边际增量；
另做按病例分半的稳定性检查。只读，不训练。
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import longitudinal_geometry as L

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
CFG = json.loads((ROOT / 'training_wss_min/configs/wss_x5d_longitudinal_20260917/X5D_long_s1234.json').read_text())
CAS = ROOT / 'data_wss_v5/views_v5_1/wss_min_cascade_v1'
SIDE = Path(CFG['data']['point_features_root'][-1])
OUT = Path(__file__).resolve().parent
names = tuple(CFG['data']['input_features'])
stats = json.loads(Path(CFG['data']['feature_stats_path']).read_text())
train = json.loads(Path(CFG['data']['split_path']).read_text())['train_cases']
VALUES = list(L.VALUE_KEYS)
N_PTS = 4000
rng = np.random.default_rng(1234)
kw = dict(target=CFG['data']['target'], target_normalization=CFG['data']['target_normalization'],
          data_root=CFG['data']['data_root'], required_frame_version=CFG['data']['required_frame_version'],
          timesteps=CFG['data']['timesteps'],
          extra_point_features=tuple(k for k in names if k in C.SIDECAR_FEATURE_KEYS),
          point_features_root=CFG['data']['point_features_root'])
wss = D.load_wss_stats(CFG['data']['wss_stats_path'])

Rs, Xs, Fs, Cs = [], [], [], []
for k, cid in enumerate(train):
    cohort, nm = cid.rsplit('/', 1)
    case = D.load_case(cohort, nm, wss, **kw)
    with np.load(SIDE / cid / 'features.npz', allow_pickle=False) as z:
        ids = z['wall_node_id_cas']
    with np.load(CAS / cid / 'features.npz', allow_pickle=False) as z:
        cid_ids, base = z['wall_node_id_cas'], z['wall_log_wss_base'].astype(np.float64)
    if not np.array_equal(ids, cid_ids):
        order = {v: i for i, v in enumerate(cid_ids)}
        base = base[np.array([order[v] for v in ids])]
    r = np.log(np.maximum(case['y_raw'].astype(np.float64), 1e-6)) - base
    idx = rng.permutation(len(case['pos']))[:min(N_PTS, len(case['pos']))]
    f = D.build_features(case, idx, names, stats)
    Rs.append(r[idx]); Xs.append(f[:, :27]); Fs.append(f[:, 27:]); Cs.append(np.full(len(idx), k))
r = np.concatenate(Rs); X = np.concatenate(Xs); F = np.concatenate(Fs); Cc = np.concatenate(Cs)
print(f'train136 折外残差：{len(r)} 点，sd {r.std():.4f}（stage1 ln R² 约 0.815）', flush=True)


def demean(a, c):
    o = a.copy()
    for k in np.unique(c):
        s = c == k
        o[s] = a[s] - a[s].mean(axis=0)
    return o


def resid_r2(A, y):
    if A.shape[1] == 0:
        return 0.0, y
    A = np.column_stack([A, np.ones(len(A))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - (e ** 2).sum() / ss), e


rw, Xw, Fw = demean(r[:, None], Cc)[:, 0], demean(X, Cc), demean(F, Cc)
r27, rres = resid_r2(Xw, rw)
print(f'原 27 维（线性、逐例去均值）能解释折外残差 R² {r27:.4f} → 之后只在剩余部分上选通道\n')


def forward(y, cases, pool):
    chosen, cum = [], []
    remaining = list(pool)
    cur = 0.0
    while remaining:
        best = None
        for j in remaining:
            cols = [c for jj in chosen + [j] for c in (2 * jj, 2 * jj + 1)]
            v, _ = resid_r2(Fw[:, cols][cases], y[cases])
            if best is None or v > best[0]:
                best = (v, j)
        v, j = best
        chosen.append(j); cum.append(v)
        remaining.remove(j)
        cur = v
    return chosen, cum


allc = np.ones(len(rw), bool)
order, cum = forward(rres, allc, range(16))
print(f'{"步":>3s} {"通道":34s} {"累计 R²":>9s} {"边际 ΔR²":>10s}')
prev = 0.0
rows = []
for i, (j, v) in enumerate(zip(order, cum), 1):
    print(f'{i:3d} {VALUES[j]:34s} {v:9.5f} {v-prev:10.5f}')
    rows.append({'rank': i, 'channel': VALUES[j], 'cum_r2': v, 'marginal_r2': v - prev})
    prev = v

half = rng.permutation(len(train))
h1 = np.isin(Cc, half[:68]); h2 = np.isin(Cc, half[68:])
o1, _ = forward(rres, h1, range(16))
o2, _ = forward(rres, h2, range(16))
print('\n按病例分半的前 6 名（稳定性检查）')
print('  A 半：' + ', '.join(VALUES[j].replace('geom_', '') for j in o1[:6]))
print('  B 半：' + ', '.join(VALUES[j].replace('geom_', '') for j in o2[:6]))
print('  全体：' + ', '.join(VALUES[j].replace('geom_', '') for j in order[:6]))

REFJ = [VALUES.index(v) for v in VALUES if v.startswith('geom_ref_')]
o_ref, c_ref = forward(rres, allc, REFJ)
print('\n只在 ref 8 内部前向选择（剪枝臂的通道顺序）')
prev = 0.0
ref_rows = []
for i, (j, v) in enumerate(zip(o_ref, c_ref), 1):
    print(f'  {i:2d} {VALUES[j]:34s} 累计 R² {v:.5f}  边际 {v-prev:.5f}')
    ref_rows.append({'rank': i, 'channel': VALUES[j], 'cum_r2': v, 'marginal_r2': v - prev})
    prev = v

print('\n候选子集的解释力（同一残差口径）')
SUB = {'全部 16 value': list(range(16)),
       'ref 8': [VALUES.index(v) for v in VALUES if v.startswith('geom_ref_')],
       'ref 8 + pc_roundness': [VALUES.index(v) for v in VALUES if v.startswith('geom_ref_')] + [VALUES.index('geom_pc_roundness')],
       '前 4（前向选出）': order[:4], '前 6（前向选出）': order[:6],
       '"新"4 ref（圆度/偏心/上下游距离）': [VALUES.index(v) for v in
            ('geom_ref_roundness', 'geom_ref_eccentricity', 'geom_ref_upstream_min_distance', 'geom_ref_downstream_min_distance')]}
subs = {}
for k, js in SUB.items():
    cols = [c for j in js for c in (2 * j, 2 * j + 1)]
    v, _ = resid_r2(Fw[:, cols], rres)
    subs[k] = v
    print(f'  {k:34s} {len(js):2d} 个 value / {2*len(js):2d} 列   R² {v:.5f}')
(OUT / 'select_channels_oof.json').write_text(json.dumps(
    {'n_points': int(len(r)), 'resid_sd': float(r.std()), 'base27_linear_r2': r27,
     'forward': rows, 'half_a': [VALUES[j] for j in o1], 'half_b': [VALUES[j] for j in o2],
     'subsets': subs, 'ref_only_forward': ref_rows}, indent=1, ensure_ascii=False))
print('\nwrote', OUT / 'select_channels_oof.json')
