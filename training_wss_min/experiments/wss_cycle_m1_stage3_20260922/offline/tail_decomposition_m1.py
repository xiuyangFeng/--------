"""M1 三 seed 集成的 Pa 差距尾部分解（矩阵 §14.3；只读）：峰值通道 vs 已部署峰值三 seed 集成、TAWSS 通道 vs A1_ens3，
逐例剔除真值最高 1% / 5% / 10% 后重算 test34 Pa R²_cb；另给 ln 空间 R²_cb、逐例胜负、top10 幅值比。"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import metrics as M  # noqa: E402
R = Path('/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs'); SEEDS = (1234, 7, 2025); OUT = Path(__file__).resolve().parent


def ens(dirs):
    out = {}
    cids = sorted('/'.join(Path(p).relative_to(dirs[0]).parts[:-1]) for p in glob.glob(str(dirs[0] / '*/*/*/predictions.npz')))
    for c in cids:
        ps = []
        for d in dirs:
            with np.load(d / c / 'predictions.npz') as z:
                ps.append(z['pred_pa'].astype(np.float64)); t = z['true_pa'].astype(np.float64)
        out[c] = (t, np.mean(ps, 0))
    return out


def r2q(P, q):
    tt, pp = [], []
    for t, p in P.values():
        m = t <= np.quantile(t, q) if q < 1 else np.ones_like(t, bool); tt.append(t[m]); pp.append(p[m])
    return M.casebalanced_field_metrics(tt, pp)['r2']


lg = lambda a: np.log(np.clip(a, 0.05, None) + 1e-6)
pairs = {
    'peak': (ens([R / f'wss_cycle_m1_stage3_20260922/M1_s{s}/eval/ckpt_best/predictions/test/wss' for s in SEEDS]),
             ens([R / f'wss_v51_wave1_20260916/X5D_v51_s{s}/eval/ckpt_best/predictions/test' for s in SEEDS]), 'X5D_v51 peak ens3'),
    'tawss': (ens([R / f'wss_cycle_m1_stage3_20260922/M1_s{s}/eval/ckpt_best/predictions/test/tawss' for s in SEEDS]),
              ens([R / f'wss_cycle_stage3_20260921/A1_s{s}/eval/ckpt_best/predictions/test' for s in SEEDS]), 'A1_ens3'),
}
res, lines = {}, ['=== M1_ens3 Pa 差距尾部分解（test34，逐例剔除真值最高分位）===']
for name, (P, Q, rname) in pairs.items():
    row = {f'{q:.2f}': {'M1': r2q(P, q), 'ref': r2q(Q, q)} for q in (1.0, 0.99, 0.95, 0.90)}
    lt = [lg(t) for t, _ in P.values()]
    top = lambda S: float(np.mean([S[c][1][t >= np.quantile(t, .9)].mean() / t[t >= np.quantile(t, .9)].mean() for c, (t, _) in P.items()]))
    res[name] = {'reference': rname, 'by_kept_quantile': row,
                 'ln_r2cb': {'M1': M.casebalanced_field_metrics(lt, [lg(p) for _, p in P.values()])['r2'], 'ref': M.casebalanced_field_metrics(lt, [lg(p) for _, p in Q.values()])['r2']},
                 'per_case_wins': int(sum(M.r2_score(t, p) > M.r2_score(t, Q[c][1]) for c, (t, p) in P.items())), 'n_cases': len(P),
                 'top10_amplitude_ratio': {'M1': top(P), 'ref': top(Q)}}
    lines.append(f"[{name} vs {rname}] " + ' | '.join(f"q{k}: {v['M1']:.4f}/{v['ref']:.4f} (Δ{v['M1']-v['ref']:+.4f})" for k, v in row.items())
                 + f" | ln {res[name]['ln_r2cb']['M1']:.4f}/{res[name]['ln_r2cb']['ref']:.4f} | 逐例胜 {res[name]['per_case_wins']}/{len(P)} | top10 幅值比 {res[name]['top10_amplitude_ratio']['M1']:.3f}/{res[name]['top10_amplitude_ratio']['ref']:.3f}")
print('\n'.join(lines)); (OUT / 'tail_decomposition_m1.json').write_text(json.dumps(res, indent=1, default=float)); (OUT / 'tail_decomposition_m1.txt').write_text('\n'.join(lines) + '\n')
