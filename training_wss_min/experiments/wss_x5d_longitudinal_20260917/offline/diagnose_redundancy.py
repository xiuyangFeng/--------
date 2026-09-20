"""诊断一：32 个沿程通道里有多少信息是原 27 维已经携带的。

对 train136 的一个子集，在 valid 点上用原 27 维（模型实际看到的标准化值）线性回归每个沿程
value 通道，报告 pooled R²（整体可预测度）与去掉逐例均值后的 within-case R²（空间型态可预测度）。
另外报告 ref/pc 同名通道的相关性（是否互为重复）与 mask 的有效率分布。只读，不训练。
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import longitudinal_geometry as L

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
EXP = ROOT / 'training_wss_min/experiments/wss_x5d_longitudinal_20260917'
CFG = json.loads((ROOT / 'training_wss_min/configs/wss_x5d_longitudinal_20260917/X5D_long_s1234.json').read_text())
OUT = Path(__file__).resolve().parent
N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_PTS = 4000
RNG = np.random.default_rng(1234)

names = tuple(CFG['data']['input_features'])
base_names = names[:27]
stats = json.loads(Path(CFG['data']['feature_stats_path']).read_text())
split = json.loads(Path(CFG['data']['split_path']).read_text())
train = split['train_cases']
pick = [train[i] for i in RNG.permutation(len(train))[:N_CASES]]

kw = dict(target=CFG['data']['target'], target_normalization=CFG['data']['target_normalization'],
          data_root=CFG['data']['data_root'], required_frame_version=CFG['data']['required_frame_version'],
          timesteps=CFG['data']['timesteps'],
          extra_point_features=tuple(k for k in names if k in C.SIDECAR_FEATURE_KEYS),
          point_features_root=CFG['data']['point_features_root'])
wss_stats = D.load_wss_stats(CFG['data']['wss_stats_path'])

X, V, M, CASE, Y = [], [], [], [], []
for k, cid in enumerate(pick):
    cohort, case_name = cid.rsplit('/', 1)
    case = D.load_case(cohort, case_name, wss_stats, **kw)
    n = len(case['pos'])
    idx = RNG.permutation(n)[:min(N_PTS, n)]
    f = D.build_features(case, idx, names, stats)
    X.append(f[:, :27]); V.append(f[:, 27::2]); M.append(f[:, 28::2])
    CASE.append(np.full(len(idx), k)); Y.append(case['y_norm'][idx])
    print(f'  loaded {cid} ({n} pts)', flush=True)
X = np.concatenate(X); V = np.concatenate(V); M = np.concatenate(M)
CASE = np.concatenate(CASE); Y = np.concatenate(Y)
VALUE_NAMES = list(L.VALUE_KEYS)
print(f'\npooled points {len(X)} from {len(pick)} train cases\n')


def ols_r2(A, y):
    A = np.column_stack([A, np.ones(len(A))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / ss) if ss > 0 else float('nan')


def demean_by_case(a, case_ids):
    out = a.copy()
    for c in np.unique(case_ids):
        m = case_ids == c
        out[m] = a[m] - a[m].mean(axis=0)
    return out


Xw = demean_by_case(X, CASE)
res = {'n_points': int(len(X)), 'n_cases': len(pick), 'channels': {}}
print(f'{"通道":34s} {"有效率":>7s} {"pooled R²":>10s} {"within-case R²":>15s}')
for j, nm in enumerate(VALUE_NAMES):
    valid = M[:, j] > 0.5
    v = V[valid, j]
    r_pool = ols_r2(X[valid], v)
    r_within = ols_r2(Xw[valid], demean_by_case(v[:, None], CASE[valid])[:, 0])
    res['channels'][nm] = {'valid_frac': float(valid.mean()), 'pooled_r2': r_pool, 'within_case_r2': r_within}
    print(f'{nm:34s} {valid.mean():7.3f} {r_pool:10.3f} {r_within:15.3f}')

print('\nref / pc 同名通道相关（两者都 valid 的点上）')
res['ref_pc_corr'] = {}
for f in L.FIELD_SPECS:
    a, b = VALUE_NAMES.index(f'geom_ref_{f}'), VALUE_NAMES.index(f'geom_pc_{f}')
    m = (M[:, a] > 0.5) & (M[:, b] > 0.5)
    c = float(np.corrcoef(V[m, a], V[m, b])[0, 1])
    res['ref_pc_corr'][f] = c
    print(f'  {f:28s} corr {c:+.4f}  (共同有效 {m.mean():.3f})')

print('\n与目标 log_z(WSS) 的单变量相关（valid 点上）')
res['target_corr'] = {}
for j, nm in enumerate(VALUE_NAMES):
    valid = M[:, j] > 0.5
    c = float(np.corrcoef(V[valid, j], Y[valid])[0, 1])
    res['target_corr'][nm] = c
    print(f'  {nm:34s} {c:+.4f}')
print('\n对照：原 27 维里最相关的几个')
for j in np.argsort(-np.abs([np.corrcoef(X[:, j], Y)[0, 1] for j in range(27)]))[:6]:
    print(f'  {base_names[j]:34s} {np.corrcoef(X[:, j], Y)[0, 1]:+.4f}')

(OUT / 'diagnose_redundancy.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
print('\nwrote', OUT / 'diagnose_redundancy.json')
