"""X5D + 32 沿程几何通道 的三 seed 离线对比分析。

读 best/last checkpoint 保存的逐点预测，与同 seed 的 X5D_v51 父臂配对比较：
物理/归一化 R²_cb、逐例胜负、AG/AAA/ILO 分队列、高 WSS 尾部，以及三 seed 集成。
额外做一项沿程特有诊断：逐例增益 vs 该例沿程几何有效覆盖率。
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs'
NEW = RUNS / 'wss_x5d_longitudinal_20260917'
BASE = RUNS / 'wss_v51_wave1_20260916'
OUT = Path(__file__).resolve().parent
SEEDS = (1234, 7, 2025)
BASE_SEEDS5 = (1234, 7, 2025, 11, 2026)
COHORTS = ('AG', 'AAA', 'ILO')


def pred_dir(run, ckpt):
    return run / f'eval/ckpt_{ckpt}/predictions/test'


units = sorted(str(p.parent.relative_to(pred_dir(BASE / 'X5D_v51_s1234', 'best')))
               for p in pred_dir(BASE / 'X5D_v51_s1234', 'best').glob('*/*/*/predictions.npz'))
assert len(units) == 34, units
coh = [u.split('/')[0] for u in units]


def load(run, ckpt='best'):
    out = {}
    for u in units:
        with np.load(pred_dir(run, ckpt) / u / 'predictions.npz') as z:
            out[u] = {k: z[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
    return out


def cb(t, p):
    return M.casebalanced_field_metrics(t, p)['r2']


def r2(t, p):
    return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def tail(t_list, p_list):
    t = np.concatenate(t_list); p = np.concatenate(p_list)
    q = np.quantile(t, .9); m = t >= q
    return dict(high_r2=float(1 - ((t[m] - p[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()),
                top10_ratio=float(p[m].mean() / t[m].mean()))


def per_cohort(truth, preds):
    return {c: cb([truth[i] for i in range(34) if coh[i] == c], [preds[i] for i in range(34) if coh[i] == c])
            for c in COHORTS}


res = {}


def summarize(name, truth, truth_n, preds, preds_n=None):
    pc = np.array([r2(t, p) for t, p in zip(truth, preds)])
    d = dict(pa_r2_cb=cb(truth, preds), case_r2_mean=float(pc.mean()), case_r2_p10=float(np.quantile(pc, .1)),
             **tail(truth, preds), per_cohort=per_cohort(truth, preds))
    if preds_n is not None:
        d['norm_r2_cb'] = cb(truth_n, preds_n)
    res[name] = d
    print(f'{name:32s} Pa R2_cb {d["pa_r2_cb"]:.4f}' + (f' | 归一化 {d["norm_r2_cb"]:.4f}' if preds_n is not None else '')
          + f' | 逐例 mean/p10 {d["case_r2_mean"]:.3f}/{d["case_r2_p10"]:.3f}'
          + f' | high-WSS {d["high_r2"]:.3f} top10 {d["top10_ratio"]:.3f}'
          + ' | AG/AAA/ILO ' + '/'.join(f'{v:.3f}' for v in d['per_cohort'].values()))
    return d


def main(ckpt='best'):
    arms = {f'X5D_long_s{s}': NEW / f'X5D_long_s{s}' for s in SEEDS}
    arms.update({f'X5D_v51_s{s}': BASE / f'X5D_v51_s{s}' for s in BASE_SEEDS5})
    data = {}
    for name, run in arms.items():
        take = ckpt if name.startswith('X5D_long') else 'best'
        data[name] = load(run, take)
    truth = [data['X5D_v51_s1234'][u]['true_pa'] for u in units]
    truth_n = [data['X5D_v51_s1234'][u]['true_norm'] for u in units]
    for name in arms:
        take = ckpt if name.startswith('X5D_long') else 'best'
        assert all(np.array_equal(data[name][u]['true_pa'], truth[i]) for i, u in enumerate(units)), name

    print(f'=== 单臂读数（沿程臂用 ckpt_{ckpt}，底座用 ckpt_best）')
    for name in [f'X5D_v51_s{s}' for s in SEEDS] + [f'X5D_long_s{s}' for s in SEEDS]:
        summarize(name, truth, truth_n, [data[name][u]['pred_pa'] for u in units],
                  [data[name][u]['pred_norm'] for u in units])

    print('\n=== 同 seed 配对 Δ（沿程 − X5D_v51）')
    rows = []
    for s in SEEDS:
        a, b = f'X5D_long_s{s}', f'X5D_v51_s{s}'
        pa = [data[a][u]['pred_pa'] for u in units]; pb = [data[b][u]['pred_pa'] for u in units]
        na = [data[a][u]['pred_norm'] for u in units]; nb = [data[b][u]['pred_norm'] for u in units]
        d_pa = cb(truth, pa) - cb(truth, pb); d_n = cb(truth_n, na) - cb(truth_n, nb)
        case_d = {u: r2(truth[i], pa[i]) - r2(truth[i], pb[i]) for i, u in enumerate(units)}
        wins = int(sum(v > 0 for v in case_d.values()))
        ca, cbh = per_cohort(truth, pa), per_cohort(truth, pb)
        ta, tb = tail(truth, pa), tail(truth, pb)
        rows.append(dict(seed=s, d_pa=d_pa, d_norm=d_n, wins=wins,
                         d_coh={c: ca[c] - cbh[c] for c in COHORTS},
                         d_high=ta['high_r2'] - tb['high_r2'], d_top10=ta['top10_ratio'] - tb['top10_ratio'],
                         case_delta=case_d))
        print(f'  {a:16s} Δ物理 {d_pa:+.4f} Δ归一化 {d_n:+.4f} 逐例胜 {wins}/34 | AG/AAA/ILO '
              + '/'.join(f'{ca[c]-cbh[c]:+.3f}' for c in COHORTS)
              + f' | Δhigh-WSS {ta["high_r2"]-tb["high_r2"]:+.3f} Δtop10 {ta["top10_ratio"]-tb["top10_ratio"]:+.3f}')
    m_pa = float(np.mean([r['d_pa'] for r in rows])); s_pa = float(np.std([r['d_pa'] for r in rows]))
    m_n = float(np.mean([r['d_norm'] for r in rows])); s_n = float(np.std([r['d_norm'] for r in rows]))
    pos_pa = sum(r['d_pa'] > 0 for r in rows); pos_n = sum(r['d_norm'] > 0 for r in rows)
    print(f'  三 seed 均值: Δ物理 {m_pa:+.4f} (sd {s_pa:.4f}, {pos_pa}/3 正) '
          f'Δ归一化 {m_n:+.4f} (sd {s_n:.4f}, {pos_n}/3 正)')
    res[f'paired_{ckpt}'] = dict(rows=[{k: v for k, v in r.items() if k != 'case_delta'} for r in rows],
                                 mean_d_pa=m_pa, sd_d_pa=s_pa, mean_d_norm=m_n, sd_d_norm=s_n,
                                 positive_pa=pos_pa, positive_norm=pos_n)

    print('\n=== 三 seed 集成（Pa 空间均值）与部署底座')
    def ens(names):
        return [np.mean([data[n][u]['pred_pa'] for n in names], axis=0) for u in units]
    def ens_n(names):
        return [np.mean([data[n][u]['pred_norm'] for n in names], axis=0) for u in units]
    e_long = summarize('X5D_long 3-seed', truth, truth_n, ens([f'X5D_long_s{s}' for s in SEEDS]),
                       ens_n([f'X5D_long_s{s}' for s in SEEDS]))
    e_base3 = summarize('X5D_v51 3-seed', truth, truth_n, ens([f'X5D_v51_s{s}' for s in SEEDS]),
                        ens_n([f'X5D_v51_s{s}' for s in SEEDS]))
    e_base5 = summarize('X5D_v51 5-seed(部署底座)', truth, truth_n, ens([f'X5D_v51_s{s}' for s in BASE_SEEDS5]),
                        ens_n([f'X5D_v51_s{s}' for s in BASE_SEEDS5]))
    print(f'  集成 Δ（3-seed 对 3-seed）物理 {e_long["pa_r2_cb"]-e_base3["pa_r2_cb"]:+.4f} '
          f'归一化 {e_long["norm_r2_cb"]-e_base3["norm_r2_cb"]:+.4f}；'
          f'对现底座（5-seed 0.7749）{e_long["pa_r2_cb"]-e_base5["pa_r2_cb"]:+.4f}')
    res[f'ensemble_{ckpt}'] = dict(long3=e_long, base3=e_base3, base5=e_base5,
                                   d_pa_vs_base3=e_long['pa_r2_cb'] - e_base3['pa_r2_cb'],
                                   d_norm_vs_base3=e_long['norm_r2_cb'] - e_base3['norm_r2_cb'],
                                   d_pa_vs_base5=e_long['pa_r2_cb'] - e_base5['pa_r2_cb'])

    # 沿程特有诊断：逐例增益 vs 该例沿程几何有效覆盖率
    audit = json.loads((OUT.parent / 'prepared_audit.json').read_text())
    cov = {}
    for row in audit['cases']:
        if row['partition'] != 'test':
            continue
        n = row['wall_points']
        cov[row['canonical_id']] = dict(
            ref=row['valid_counts']['geom_ref_log_area'] / n,
            pc=row['valid_counts']['geom_pc_log_area'] / n)
    mean_case_d = {u: float(np.mean([r['case_delta'][u] for r in rows])) for u in units}
    xs_ref = np.array([cov[u]['ref'] for u in units]); xs_pc = np.array([cov[u]['pc'] for u in units])
    ys = np.array([mean_case_d[u] for u in units])
    c_ref = float(np.corrcoef(xs_ref, ys)[0, 1]); c_pc = float(np.corrcoef(xs_pc, ys)[0, 1])
    lo = ys[xs_ref < 0.6]; hi = ys[xs_ref >= 0.6]
    print('\n=== 覆盖率诊断（三 seed 平均逐例 ΔR²）')
    print(f'  相关系数：参考面覆盖 {c_ref:+.3f}，点云覆盖 {c_pc:+.3f}')
    print(f'  参考面覆盖 <60% 的 {len(lo)} 例平均 Δ {lo.mean() if len(lo) else float("nan"):+.4f}；'
          f'≥60% 的 {len(hi)} 例平均 Δ {hi.mean():+.4f}')
    worst = sorted(mean_case_d.items(), key=lambda kv: kv[1])[:5]
    best = sorted(mean_case_d.items(), key=lambda kv: -kv[1])[:5]
    print('  最差 5 例：' + ', '.join(f'{u} {d:+.3f}(ref {cov[u]["ref"]:.2f})' for u, d in worst))
    print('  最好 5 例：' + ', '.join(f'{u} {d:+.3f}(ref {cov[u]["ref"]:.2f})' for u, d in best))
    res[f'coverage_{ckpt}'] = dict(corr_ref=c_ref, corr_pc=c_pc,
                                   low_coverage_cases=int(len(lo)),
                                   low_coverage_mean_delta=float(lo.mean()) if len(lo) else None,
                                   high_coverage_mean_delta=float(hi.mean()),
                                   case_delta_mean=mean_case_d, coverage=cov)


if __name__ == '__main__':
    for ckpt in ('best', 'last'):
        print(f'\n########## checkpoint = {ckpt} ##########')
        main(ckpt)
    (OUT / 'analyze_longitudinal.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print('\nwrote', OUT / 'analyze_longitudinal.json')
