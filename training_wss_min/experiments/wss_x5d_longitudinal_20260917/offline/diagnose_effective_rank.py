"""诊断四：把原 27 维回归掉之后，32 个沿程通道还剩几个有效自由度。"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import longitudinal_geometry as L

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
CFG = json.loads((ROOT / 'training_wss_min/configs/wss_x5d_longitudinal_20260917/X5D_long_s1234.json').read_text())
OUT = Path(__file__).resolve().parent
names = tuple(CFG['data']['input_features'])
stats = json.loads(Path(CFG['data']['feature_stats_path']).read_text())
train = json.loads(Path(CFG['data']['split_path']).read_text())['train_cases']
rng = np.random.default_rng(1234)
pick = [train[i] for i in rng.permutation(len(train))[:40]]
kw = dict(target=CFG['data']['target'], target_normalization=CFG['data']['target_normalization'],
          data_root=CFG['data']['data_root'], required_frame_version=CFG['data']['required_frame_version'],
          timesteps=CFG['data']['timesteps'],
          extra_point_features=tuple(k for k in names if k in C.SIDECAR_FEATURE_KEYS),
          point_features_root=CFG['data']['point_features_root'])
wss = D.load_wss_stats(CFG['data']['wss_stats_path'])
X, V, M, Cc = [], [], [], []
for k, cid in enumerate(pick):
    cohort, nm = cid.rsplit('/', 1)
    case = D.load_case(cohort, nm, wss, **kw)
    idx = rng.permutation(len(case['pos']))[:4000]
    f = D.build_features(case, idx, names, stats)
    X.append(f[:, :27]); V.append(f[:, 27::2]); M.append(f[:, 28::2]); Cc.append(np.full(len(idx), k))
X = np.concatenate(X); V = np.concatenate(V); M = np.concatenate(M); Cc = np.concatenate(Cc)
keep = M.all(axis=1)
X, V, Cc = X[keep], V[keep], Cc[keep]


def demean(a, c):
    o = a.copy()
    for k in np.unique(c):
        s = c == k
        o[s] = a[s] - a[s].mean(axis=0)
    return o


Xw, Vw = demean(X, Cc), demean(V, Cc)
A = np.column_stack([Xw, np.ones(len(Xw))])
coef, *_ = np.linalg.lstsq(A, Vw, rcond=None)
R = Vw - A @ coef                      # 27 维解释不掉的部分
tot = (Vw ** 2).sum(axis=0)
new = (R ** 2).sum(axis=0)
print(f'同时有效点 {len(X)}（{keep.mean():.3f}），逐例去均值口径')
print(f'16 个 value 通道：总方差被原 27 维解释掉 {1 - new.sum()/tot.sum():.4f}，剩余 {new.sum()/tot.sum():.4f}')
Rn = R / (R.std(axis=0) + 1e-12)
ev = np.linalg.eigvalsh(np.corrcoef(Rn, rowvar=False))[::-1]
frac = np.cumsum(ev) / ev.sum()
k95 = int(np.searchsorted(frac, 0.95) + 1)
part = ev / ev.sum()
eff = float(np.exp(-(part * np.log(part + 1e-12)).sum()))
print(f'残差的相关矩阵特征值（前 8）：' + ' '.join(f'{v:.2f}' for v in ev[:8]))
print(f'达到 95% 残差方差需要 {k95} 个主成分；参与度熵给出的有效维数 {eff:.2f}（满分 16）')
print(f'=> 32 个通道（16 value + 16 mask）在原 27 维之外的有效新自由度约 {eff:.1f} 个')
(OUT / 'diagnose_effective_rank.json').write_text(json.dumps(
    {'n_points': int(len(X)), 'all_valid_frac': float(keep.mean()),
     'variance_explained_by_base27': float(1 - new.sum()/tot.sum()),
     'eigenvalues': ev.tolist(), 'k95': k95, 'effective_dim': eff}, indent=1))
