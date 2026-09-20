"""沿程通道剪枝阶梯（单 seed 1234）的读数：从 32 列一路砍到 2 列，看哪一档开始掉。

同 seed 参照：X5D_v51_s1234（无沿程通道）与 X5D_long_s1234（全 32 列）。
单 seed 口径，配对 Δ 噪声带约 ±0.014（物理）/ ±0.004（归一化）；
判定规则 = 取"不比全 32 列明显差"的最小通道集合。
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
from training_wss_min import longitudinal_geometry as L
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
PRUNE = RUNS / 'wss_x5d_long_prune_20260917'
OUT = Path(__file__).resolve().parent
P = 'eval/ckpt_best/predictions/test'
SEED = 1234
MATRIX = json.loads((ROOT / 'training_wss_min/configs/wss_x5d_long_prune_20260917/matrix.json').read_text())
CHAN = {a['id']: a['channels'] for a in MATRIX['arms']}
DIM = {a['id']: a['input_dim'] for a in MATRIX['arms']}
ARMS = [('X5D_v51 底座', RUNS / f'wss_v51_wave1_20260916/X5D_v51_s{SEED}', [], 27),
        ('X5D_long 全32列', RUNS / f'wss_x5d_longitudinal_20260917/X5D_long_s{SEED}', list(L.VALUE_KEYS), 59)]
for aid in ('L8_s1234', 'L5_s1234', 'L2_s1234', 'L1_s1234'):
    ARMS.append((aid, PRUNE / aid, CHAN[aid], DIM[aid]))

units = sorted(str(p.parent.relative_to(ARMS[0][1] / P)) for p in (ARMS[0][1] / P).glob('*/*/*/predictions.npz'))
assert len(units) == 34
coh = [u.split('/')[0] for u in units]
SIDE = ROOT / 'training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features'


def load(run):
    out = {}
    for u in units:
        with np.load(run / P / u / 'predictions.npz') as z:
            out[u] = {k: z[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
            out[u]['rows'] = z['row_index']
    return out


def r2(t, p):
    return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def cb(t, p):
    return M.casebalanced_field_metrics(t, p)['r2']


def tail(t_list, p_list):
    t = np.concatenate(t_list); p = np.concatenate(p_list); q = np.quantile(t, .9); m = t >= q
    return (float(1 - ((t[m] - p[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()), float(p[m].mean() / t[m].mean()))


def iou(truth, preds):
    v = []
    for t, p in zip(truth, preds):
        mt = t >= np.quantile(t, .9); mp = p >= np.quantile(p, .9)
        v.append((mt & mp).sum() / (mt | mp).sum())
    return float(np.mean(v))


data = {}
for name, run, _, _ in ARMS:
    if not (run / P).is_dir():
        print(f'!! missing predictions: {run}'); sys.exit(1)
    data[name] = load(run)
truth = [data[ARMS[0][0]][u]['true_pa'] for u in units]
truth_n = [data[ARMS[0][0]][u]['true_norm'] for u in units]

valid = {}
for name, _, ch, _ in ARMS:
    if not ch:
        valid[name] = None
        continue
    m = []
    for u in units:
        rows = data[name][u]['rows']
        with np.load(SIDE / u / 'features.npz', allow_pickle=False) as z:
            ok = np.ones(len(rows), bool)
            for c in ch:
                ok &= z[f'wall_{c}_valid'][rows].astype(bool)
        m.append(ok)
    valid[name] = m

base_name = ARMS[0][0]
pb = [data[base_name][u]['pred_pa'] for u in units]
pbn = [data[base_name][u]['pred_norm'] for u in units]
res = {}
print(f'{"臂":20s} {"新列":>4s} {"Pa R2_cb":>9s} {"Δ底座":>8s} {"归一化":>8s} {"Δ":>8s} {"胜":>7s} {"high":>6s} {"IoU":>6s} {"覆盖":>6s}  AG/AAA/ILO')
for name, run, ch, dim in ARMS:
    p = [data[name][u]['pred_pa'] for u in units]
    pn = [data[name][u]['pred_norm'] for u in units]
    v_pa, v_n = cb(truth, p), cb(truth_n, pn)
    d_pa, d_n = v_pa - cb(truth, pb), v_n - cb(truth_n, pbn)
    wins = sum(r2(truth[i], p[i]) > r2(truth[i], pb[i]) for i in range(34))
    hi, top = tail(truth, p)
    per = {c: cb([truth[i] for i in range(34) if coh[i] == c], [p[i] for i in range(34) if coh[i] == c])
           for c in ('AG', 'AAA', 'ILO')}
    cov = float(np.concatenate(valid[name]).mean()) if valid[name] is not None else float('nan')
    res[name] = dict(input_dim=dim, n_channels=len(ch), channels=ch, pa_r2_cb=v_pa, norm_r2_cb=v_n,
                     d_pa=d_pa, d_norm=d_n, wins=int(wins), high_wss_r2=hi, top10_ratio=top,
                     top10_iou=iou(truth, p), coverage=cov, per_cohort=per)
    print(f'{name:20s} {2*len(ch):4d} {v_pa:9.4f} {d_pa:+8.4f} {v_n:8.4f} {d_n:+8.4f} {wins:5d}/34 '
          f'{hi:6.3f} {res[name]["top10_iou"]:6.3f} {cov:6.3f}  ' + '/'.join(f'{x:.3f}' for x in per.values()))

print('\n逐点 ln 空间 MSE 相对底座的改善（按该臂自己的通道有效性分桶）')
print(f'{"臂":20s} {"全部":>9s} {"有效区":>9s} {"缺失区":>9s} {"有效占比":>9s}')
eb = np.concatenate([(np.log(np.maximum(truth[i], 1e-6)) - np.log(np.maximum(pb[i], 1e-6))) ** 2 for i in range(34)])
for name, run, ch, dim in ARMS[1:]:
    el = np.concatenate([(np.log(np.maximum(truth[i], 1e-6))
                          - np.log(np.maximum(data[name][units[i]]['pred_pa'], 1e-6))) ** 2 for i in range(34)])
    mk = np.concatenate(valid[name])
    row = {}
    for tag, sel in (('all', np.ones(len(eb), bool)), ('valid', mk), ('missing', ~mk)):
        a, b = eb[sel].mean(), el[sel].mean()
        row[tag] = float((a - b) / a * 100)
    res[name]['mse_buckets'] = row
    print(f'{name:20s} {row["all"]:+8.2f}% {row["valid"]:+8.2f}% {row["missing"]:+8.2f}% {mk.mean():9.3f}')

print('\n与底座的两模型等权平均（Pa 空间）与按有效性门控')
for name, run, ch, dim in ARMS[1:]:
    p = [data[name][u]['pred_pa'] for u in units]
    mix = [(a + b) / 2 for a, b in zip(pb, p)]
    gat = [np.where(g, b, a) for a, b, g in zip(pb, p, valid[name])]
    res[name]['mix_pa_r2_cb'] = cb(truth, mix)
    res[name]['gate_pa_r2_cb'] = cb(truth, gat)
    print(f'  {name:20s} 等权 {cb(truth, mix):.4f}   门控 {cb(truth, gat):.4f}')

(OUT / 'analyze_ladder.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print('\nwrote', OUT / 'analyze_ladder.json')
