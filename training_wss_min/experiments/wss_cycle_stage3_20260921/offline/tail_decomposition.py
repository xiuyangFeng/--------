"""test34 上 TAWSS Pa R²_cb 的尾部分解（矩阵 §13.1 用；只读）：
A1 三 seed 集成 vs 自由基线（TAWSS-null / TAWSS-ratio），逐例剔除真值 TAWSS 最高 1% / 5% / 10% 的点后重算 Pa R²_cb；
另给逐例 R² 胜负计数与 ln 空间 R²_cb。回答「Pa 口径持平是不是只由高值尾部造成」。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_cycle_20260920/offline')
import c1_baselines as B  # noqa: E402
from common import TRAIN, TEST, load_frame_stats, load_oof_peak_ln, PEAK_INDEX, EPS  # noqa: E402
from training_wss_min import metrics as M  # noqa: E402

ROOT = Path('/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs')
OUT = Path(__file__).resolve().parent
SEEDS = (1234, 7, 2025)
mu, sd, _ = load_frame_stats('train136'); c_ratio, _ = B.fit_ratio(TRAIN)
truth, null, ratio, ens = {}, {}, {}, {}
for cid in TEST:
    c = B.load_cycle(cid); lp = load_oof_peak_ln(cid, 'test', c['ids'])
    z = (lp - mu[PEAK_INDEX]) / sd[PEAK_INDEX]; ln = mu[:80, None] + sd[:80, None] * z[None, :]
    truth[cid] = c['tawss']; null[cid] = np.clip(np.exp(ln) - EPS, 0, None).mean(0); ratio[cid] = c_ratio * np.clip(np.exp(lp) - EPS, 0, None)
    ps = []
    for s in SEEDS:
        with np.load(ROOT / f'wss_cycle_stage3_20260921/A1_s{s}/eval/ckpt_best/predictions/test/{cid}/predictions.npz') as zz:
            ps.append(zz['pred_pa'].astype(np.float64)); t = zz['true_pa']
    if not np.allclose(t, truth[cid], atol=1e-4):
        raise ValueError(f'{cid}: prediction rows differ from cycle view')
    ens[cid] = np.mean(ps, 0)


def r2_excl(pred, q):
    tt, pp = [], []
    for cid in TEST:
        t = truth[cid]; m = t <= np.quantile(t, q) if q < 1 else np.ones_like(t, bool); tt.append(t[m]); pp.append(pred[cid][m])
    return M.casebalanced_field_metrics(tt, pp)['r2']


lg = lambda a: np.log(np.clip(a, 0.05, None) + EPS)
res = {'definition': __doc__.strip().splitlines()[0], 'n_cases': len(TEST), 'seeds': SEEDS, 'ratio_c': c_ratio, 'by_kept_quantile': {}}
lines = ['=== test34 TAWSS Pa R²_cb 尾部分解（逐例剔除真值最高分位后重算）===', f"{'保留 ≤ 分位':12s} {'A1_ens3':>8s} {'TAWSS-null':>11s} {'ratio':>8s}"]
for q in (1.0, 0.99, 0.95, 0.90):
    row = {'A1_ens3': r2_excl(ens, q), 'TAWSS_null': r2_excl(null, q), 'TAWSS_ratio': r2_excl(ratio, q)}
    res['by_kept_quantile'][f'{q:.2f}'] = row
    lines.append(f"{q:12.2f} {row['A1_ens3']:8.4f} {row['TAWSS_null']:11.4f} {row['TAWSS_ratio']:8.4f}")
wins = sum(M.r2_score(truth[c], ens[c]) > max(M.r2_score(truth[c], null[c]), M.r2_score(truth[c], ratio[c])) for c in TEST)
res['per_case_wins_vs_best_baseline'] = int(wins)
res['log_r2cb'] = {'A1_ens3': M.casebalanced_field_metrics([lg(truth[c]) for c in TEST], [lg(ens[c]) for c in TEST])['r2'],
                   'TAWSS_null': M.casebalanced_field_metrics([lg(truth[c]) for c in TEST], [lg(null[c]) for c in TEST])['r2'],
                   'TAWSS_ratio': M.casebalanced_field_metrics([lg(truth[c]) for c in TEST], [lg(ratio[c]) for c in TEST])['r2']}
lines.append(f"逐例 R²：A1_ens3 胜 max(null, ratio) {wins}/{len(TEST)} 例")
lines.append(f"ln 空间 R²_cb：A1_ens3 {res['log_r2cb']['A1_ens3']:.4f} | null {res['log_r2cb']['TAWSS_null']:.4f} | ratio {res['log_r2cb']['TAWSS_ratio']:.4f}")
print('\n'.join(lines))
(OUT / 'tail_decomposition.json').write_text(json.dumps(res, indent=1, default=float))
(OUT / 'tail_decomposition.txt').write_text('\n'.join(lines) + '\n')
