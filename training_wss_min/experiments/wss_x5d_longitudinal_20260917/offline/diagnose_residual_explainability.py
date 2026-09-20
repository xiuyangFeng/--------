"""诊断三：底座模型在 test34 上犯的错，32 个沿程通道能线性解释多少。

残差 r = ln(true_Pa) − ln(pred_Pa)，pred 用 X5D_v51 三 seed Pa 均值集成。
沿程通道按冻结的 train136 统计量标准化后做 OLS：pooled 与逐例去均值两种口径。
逐点对齐用 predictions.npz 的 row_index。只读，不推理。
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import longitudinal_geometry as L

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs'
BASE = RUNS / 'wss_v51_wave1_20260916'
LONG = RUNS / 'wss_x5d_longitudinal_20260917'
EXP = ROOT / 'training_wss_min/experiments/wss_x5d_longitudinal_20260917'
SIDE = EXP / 'frozen_longitudinal_features'
OUT = Path(__file__).resolve().parent
STATS = json.loads((EXP / 'feature_stats/all_train136.json').read_text())
SEEDS = (1234, 7, 2025)
P = 'eval/ckpt_best/predictions/test'
units = sorted(str(p.parent.relative_to(BASE / 'X5D_v51_s1234' / P))
               for p in (BASE / 'X5D_v51_s1234' / P).glob('*/*/*/predictions.npz'))
VALUES = list(L.VALUE_KEYS)
REF = [v for v in VALUES if v.startswith('geom_ref_')]


def ols_r2(A, y):
    A = np.column_stack([A, np.ones(len(A))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - A @ coef) ** 2).sum() / ss) if ss > 0 else float('nan')


def collect(run_family):
    Rs, Fs, Ms, Cs = [], [], [], []
    for k, u in enumerate(units):
        preds = []
        for s in SEEDS:
            with np.load(run_family / f'{"X5D_v51" if run_family is BASE else "X5D_long"}_s{s}' / P / u / 'predictions.npz') as z:
                preds.append(z['pred_pa']); rows = z['row_index']; true = z['true_pa']
        pred = np.mean(preds, axis=0)
        r = np.log(np.maximum(true.astype(np.float64), 1e-6)) - np.log(np.maximum(pred, 1e-6))
        with np.load(SIDE / u / 'features.npz', allow_pickle=False) as z:
            f = np.stack([z[f'wall_{n}'][rows].astype(np.float64) for n in VALUES], axis=1)
            m = np.stack([z[f'wall_{n}_valid'][rows].astype(bool) for n in VALUES], axis=1)
        f = np.where(m, f, 0.0)
        for j, n in enumerate(VALUES):
            f[m[:, j], j] = (f[m[:, j], j] - STATS[n]['mean']) / STATS[n]['std']
        Rs.append(r); Fs.append(f); Ms.append(m); Cs.append(np.full(len(r), k))
    return (np.concatenate(Rs), np.concatenate(Fs), np.concatenate(Ms), np.concatenate(Cs))


def demean(a, c):
    out = a.copy()
    for k in np.unique(c):
        s = c == k
        out[s] = a[s] - a[s].mean(axis=0)
    return out


res = {}
for tag, fam in (('X5D_v51 底座', BASE), ('X5D_long 沿程臂', LONG)):
    r, F, M, C = collect(fam)
    keep = M.all(axis=1)
    print(f'\n=== {tag}：残差 ln(true)−ln(pred)（三 seed Pa 均值集成）')
    print(f'  test34 共 {len(r)} 点，32 通道同时有效的点 {keep.sum()} ({keep.mean():.3f})')
    print(f'  残差 sd 全体 {r.std():.4f}，同时有效子集 {r[keep].std():.4f}')
    rk, Fk, Ck = r[keep], F[keep], C[keep]
    rw, Fw = demean(rk[:, None], Ck)[:, 0], demean(Fk, Ck)
    row = {
        'n_points': int(len(r)), 'n_all_valid': int(keep.sum()),
        'resid_sd': float(r.std()),
        'ols_r2_16values_pooled': ols_r2(Fk, rk),
        'ols_r2_16values_within_case': ols_r2(Fw, rw),
        'ols_r2_8ref_pooled': ols_r2(Fk[:, [VALUES.index(n) for n in REF]], rk),
    }
    print(f'  16 个 value 通道 OLS 解释残差：pooled R² {row["ols_r2_16values_pooled"]:.4f}，'
          f'逐例去均值 R² {row["ols_r2_16values_within_case"]:.4f}')
    print(f'  只用 ref 的 8 个：pooled R² {row["ols_r2_8ref_pooled"]:.4f}')
    print('  单通道 |corr| 与残差（逐例去均值）：')
    cors = {}
    for j, n in enumerate(VALUES):
        c = float(np.corrcoef(Fw[:, j], rw)[0, 1]) if Fw[:, j].std() > 0 else 0.0
        cors[n] = c
    for n, c in sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f'    {n:34s} {c:+.4f}')
    row['resid_corr_within_case'] = cors
    res[tag] = row

(OUT / 'diagnose_residual_explainability.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
print('\nwrote', OUT / 'diagnose_residual_explainability.json')
