"""C2：标签内分析（只用真值，不作输入、不设门）。矩阵 §3 C2。

170 例 wss_min_cycle_v1：逐例 Spearman(OSI, ln TAWSS)、Spearman(OSI, rev_frac)、高 OSI 点落在该例 TAWSS 最低 30% 区的份额、
OSI > 0.1 / > 0.3 面积、滞留区（TAWSS < 0.4 Pa ∧ OSI > 0.1）面积；按队列汇总；附 C0 审计（周期未收敛例）。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
CYC = ROOT / 'data_wss_v5/views_v5_1/wss_min_cycle_v1'
OUT = Path(__file__).resolve().parent
FLOOR = 0.05

manifest = json.loads((CYC / 'cycle_manifest.json').read_text())
rows = []
for rec in manifest['cases']:
    cid = rec['case']
    with np.load(CYC / cid / 'cycle.npz', allow_pickle=False) as z:
        ta = z['wall_tawss'].astype(np.float64); osi = z['wall_osi'].astype(np.float64); rev = z['wall_rev_frac'].astype(np.float64)
    ln_ta = np.log(np.clip(ta, FLOOR, None))
    hi = osi > 0.1; q30 = np.quantile(ta, 0.3)
    rows.append(dict(case=cid, cohort=cid.split('/')[0], n=int(ta.size),
                     sp_osi_lntawss=float(spearmanr(osi, ln_ta).correlation),
                     sp_osi_revfrac=float(spearmanr(osi, rev).correlation),
                     hi_osi_in_low30_tawss=float((hi & (ta < q30)).sum() / max(hi.sum(), 1)),
                     osi_gt_0p1=float(hi.mean()), osi_gt_0p3=float((osi > 0.3).mean()),
                     tawss_lt_0p4=float((ta < 0.4).mean()), stagnation=float(((ta < 0.4) & hi).mean()),
                     tawss_median=float(np.median(ta)), osi_median=float(np.median(osi)),
                     period_wrap_rel_diff=float(rec['period_wrap_rel_diff'])))
keys = ['sp_osi_lntawss', 'sp_osi_revfrac', 'hi_osi_in_low30_tawss', 'osi_gt_0p1', 'osi_gt_0p3', 'tawss_lt_0p4', 'stagnation', 'tawss_median', 'osi_median']
def summ(sel):
    return {k: {'median': float(np.median([r[k] for r in sel])), 'p10': float(np.quantile([r[k] for r in sel], .1)),
                'p90': float(np.quantile([r[k] for r in sel], .9))} for k in keys} | {'n_cases': len(sel)}
res = {'all': summ(rows)}
for co in ('AG', 'AAA', 'ILO'):
    res[co] = summ([r for r in rows if r['cohort'] == co])
flag = [r for r in rows if r['period_wrap_rel_diff'] > manifest['period_wrap_flag_threshold']]
res['period_wrap'] = {'threshold': manifest['period_wrap_flag_threshold'], 'n_flagged': len(flag),
                      'flagged': [(r['case'], round(r['period_wrap_rel_diff'], 3)) for r in sorted(flag, key=lambda r: -r['period_wrap_rel_diff'])],
                      'flagged_summary': summ(flag) if flag else None}
res['per_case'] = rows
(OUT / 'c2_label_analysis.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
lines = ['=== C2 标签内分析（170 例，只读真值）===']
for grp in ('all', 'AG', 'AAA', 'ILO'):
    s = res[grp]; lines.append(f"[{grp}] n={s['n_cases']} " + ' '.join(f"{k}={s[k]['median']:.3f}[{s[k]['p10']:.2f},{s[k]['p90']:.2f}]" for k in keys))
lines.append(f"period-wrap flagged (> {res['period_wrap']['threshold']:.0%}): {res['period_wrap']['n_flagged']} → {res['period_wrap']['flagged'][:12]}")
print('\n'.join(lines)); (OUT / 'c2_label_analysis.txt').write_text('\n'.join(lines) + '\n')
