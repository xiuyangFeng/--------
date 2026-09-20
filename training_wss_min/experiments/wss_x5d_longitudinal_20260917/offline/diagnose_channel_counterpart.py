"""诊断五：每个沿程 value 通道最像原 27 维里的哪一个特征。

`diagnose_redundancy.py` 回答的是"这个通道有多少能被原 27 维整体复现"；本脚本回答下一个问题：
**是被哪一个复现的**。对同一批采样点（同 seed、同例数、同点数，可与 diagnose_redundancy 对齐），
在该通道的 valid 点上对每个原始特征做单变量线性拟合，报告 |r| 最大的前三名与其 R²，
pooled 与逐例去均值两种口径都给。只读，不训练、不推理。

用法：python diagnose_channel_counterpart.py [n_cases]
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import longitudinal_geometry as L

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
CFG = json.loads((ROOT / 'training_wss_min/configs/wss_x5d_longitudinal_20260917'
                  / 'X5D_long_s1234.json').read_text())
OUT = Path(__file__).resolve().parent
N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_PTS = 4000
RNG = np.random.default_rng(1234)

names = tuple(CFG['data']['input_features'])
base_names = list(names[:27])
stats = json.loads(Path(CFG['data']['feature_stats_path']).read_text())
split = json.loads(Path(CFG['data']['split_path']).read_text())
train = split['train_cases']
pick = [train[i] for i in RNG.permutation(len(train))[:N_CASES]]

kw = dict(target=CFG['data']['target'], target_normalization=CFG['data']['target_normalization'],
          data_root=CFG['data']['data_root'],
          required_frame_version=CFG['data']['required_frame_version'],
          timesteps=CFG['data']['timesteps'],
          extra_point_features=tuple(k for k in names if k in C.SIDECAR_FEATURE_KEYS),
          point_features_root=CFG['data']['point_features_root'])
wss_stats = D.load_wss_stats(CFG['data']['wss_stats_path'])

X, V, M, CASE = [], [], [], []
for k, cid in enumerate(pick):
    cohort, case_name = cid.rsplit('/', 1)
    case = D.load_case(cohort, case_name, wss_stats, **kw)
    n = len(case['pos'])
    idx = RNG.permutation(n)[:min(N_PTS, n)]
    f = D.build_features(case, idx, names, stats)
    X.append(f[:, :27]); V.append(f[:, 27::2]); M.append(f[:, 28::2])
    CASE.append(np.full(len(idx), k))
    print(f'  loaded {cid} ({n} pts)', flush=True)
X = np.concatenate(X); V = np.concatenate(V); M = np.concatenate(M)
CASE = np.concatenate(CASE)
VALUE_NAMES = list(L.VALUE_KEYS)
print(f'\npooled points {len(X)} from {len(pick)} train cases\n')


def demean_by_case(a, case_ids):
    out = a.astype(float).copy()
    for c in np.unique(case_ids):
        m = case_ids == c
        out[m] = a[m] - a[m].mean(axis=0)
    return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


res = {'n_points': int(len(X)), 'n_cases': len(pick),
       'note': 'univariate fit of each base feature to each longitudinal value channel, '
               'on that channel\'s valid points only',
       'channels': {}}
lines = []
for j, vname in enumerate(VALUE_NAMES):
    valid = M[:, j] > .5
    if valid.sum() < 1000:
        continue
    y = V[valid, j]
    yw = demean_by_case(y[:, None], CASE[valid])[:, 0]
    Xv = X[valid]
    Xw = demean_by_case(Xv, CASE[valid])
    rows = []
    for i, bname in enumerate(base_names):
        r = corr(Xv[:, i], y)
        rw = corr(Xw[:, i], yw)
        rows.append({'base': bname, 'r_pooled': r, 'r2_pooled': r * r,
                     'r_within_case': rw, 'r2_within_case': rw * rw})
    rows.sort(key=lambda d: -abs(d['r_within_case']))
    res['channels'][vname] = {'valid_frac': float(valid.mean()), 'top': rows[:3]}
    top = rows[0]
    lines.append(f"{vname:<36} 最像 {top['base']:<22} "
                 f"|r|={abs(top['r_within_case']):.3f} R²={top['r2_within_case']:.3f}"
                 f"  次：{rows[1]['base']}({abs(rows[1]['r_within_case']):.2f})"
                 f"、{rows[2]['base']}({abs(rows[2]['r_within_case']):.2f})")

text = ('每个沿程 value 通道最像的原始特征（逐例去均值口径，valid 点）\n'
        f'样本 {len(X)} 点 / {len(pick)} 例 train\n\n' + '\n'.join(lines) + '\n')
(OUT / 'diagnose_channel_counterpart.txt').write_text(text)
(OUT / 'diagnose_channel_counterpart.json').write_text(
    json.dumps(res, ensure_ascii=False, indent=2) + '\n')
print(text)
