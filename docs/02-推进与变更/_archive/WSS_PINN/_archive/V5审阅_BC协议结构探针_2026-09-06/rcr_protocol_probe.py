"""只读探针：171/172 formal 病例的出口 RCR 是否遵循协议规则，以及分配能被几何（出口面积）解释多少。

用途：V5 交叉审阅 §2 的数字来源。RCR 不作为模型输入（导师意见）；本表只用于天花板估计与分层评价。
运行：在仓库根目录 `python3 docs/02-推进与变更/_archive/WSS_PINN/_archive/V5审阅_BC协议结构探针_2026-09-06/rcr_protocol_probe.py`
输入：拓扑审计 `interfaces[].area_m2/semantic_label`（解剖切面）、各病例根目录 `udf-inlet*.c` 的四个 pressure_out{le,li,re,ri} 块、
      第五轮 `training_wss_min/runs/_round5/bc_audit/bc_per_case.csv`（AG 实测出口平均流量）。
输出：同目录 `rcr_protocol_per_case.csv`、`rcr_protocol_summary.txt`。
"""
from __future__ import annotations
import csv, glob, json, os, re, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TOPO = ROOT / 'outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits/topology/cases'
AG_CSV = ROOT / 'training_wss_min/runs/_round5/bc_audit/bc_per_case.csv'
OUTS = ['le', 'li', 're', 'ri']
NUM = r'([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)'


def parse_udf(path: Path):
    txt = path.read_text(errors='ignore')
    res = {}
    for o in OUTS:
        m = re.search(r'DEFINE_PROFILE\(\s*pressure_out' + o + r'\b(.*?)(?=DEFINE_PROFILE|\Z)', txt, re.S)
        if not m:
            return None
        blk = m.group(1)
        try:
            res[o] = tuple(float(re.search(r'\b' + k + r'\s*=\s*' + NUM, blk).group(1)) for k in ('R1', 'R2', 'C'))
        except AttributeError:
            return None
    return res


def r2(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy * sxy / (sxx * syy)


def resid_sd(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    a0 = my - b * mx
    return st.pstdev([c - (a0 + b * a) for a, c in zip(x, y)])


rows = []
missing = []
for f in sorted(TOPO.glob('*.json')):
    d = json.load(open(f))
    cid = d['canonical_id']
    areas = {it['semantic_label']: it['area_m2'] for it in d['interfaces']}
    casedir = Path(d['fluent_case']['path']).parent
    udfs = sorted(casedir.glob('udf-inlet*.c'), key=lambda p: ('4' not in p.name, p.name))
    if not udfs:  # WANG_YONG_FAN keeps its UDF only under libudf/src (byte-identical to the compiled copy)
        udfs = sorted(casedir.glob('libudf/src/udf-inlet*.c'))
    rcr = parse_udf(udfs[0]) if udfs else None
    if not rcr:
        missing.append(cid)
        continue
    g = {o: 1.0 / (rcr[o][0] + rcr[o][1]) for o in OUTS}
    G = sum(g.values())
    rec = {
        'canonical_id': cid, 'cohort': cid.split('/')[0], 'role': d['role'],
        'udf_file': str(udfs[0].relative_to(ROOT)),
        'r_total_pa_s_per_kg': 1.0 / G,  # UDF Windkessel uses F_FLUX mass flow (kg/s), so R is Pa·s/kg
        'left_conductance_share': (g['le'] + g['li']) / G,
        'ext_share_left': g['le'] / (g['le'] + g['li']),
        'ext_share_right': g['re'] / (g['re'] + g['ri']),
        'area_share_left': areas['out-le'] / (areas['out-le'] + areas['out-li']),
        'area_share_right': areas['out-re'] / (areas['out-re'] + areas['out-ri']),
        'tau_rc_median_s': st.median(rcr[o][1] * rcr[o][2] for o in OUTS),
        'area_inlet_m2': areas['inlet'],
    }
    for o in OUTS:
        rec[f'area_out{o}_m2'] = areas['out-' + o]
        rec[f'R1_out{o}'], rec[f'R2_out{o}'], rec[f'C_out{o}'] = rcr[o]
    rows.append(rec)

fields = list(rows[0].keys())
with open(HERE / 'rcr_protocol_per_case.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

lines = [f'cases parsed: {len(rows)} / {len(rows) + len(missing)}; missing UDF at case root: {missing}']
by = defaultdict(list)
for r in rows:
    by[r['cohort']].append(r)
    by['ALL'].append(r)
for coh in ('AG', 'AAA', 'ILO', 'ALL'):
    rs = by[coh]
    Rt = [r['r_total_pa_s_per_kg'] for r in rs]
    lshare = [r['left_conductance_share'] for r in rs]
    ext = [r['ext_share_left'] for r in rs] + [r['ext_share_right'] for r in rs]
    ash = [r['area_share_left'] for r in rs] + [r['area_share_right'] for r in rs]
    tau = [r['tau_rc_median_s'] for r in rs]
    rsd = resid_sd(ash, ext)
    me = st.mean(ext)
    lines += [
        f'=== {coh} n={len(rs)} ===',
        f'  R_total (Pa·s/kg): min {min(Rt):.4g} median {st.median(Rt):.4g} max {max(Rt):.4g} CoV {st.pstdev(Rt) / st.mean(Rt):.3f}; '
        f'MAP≈rho*Qbar*R_total = {1060 * 3.351e-5 * st.median(Rt) / 133.32:.0f} mmHg',
        f'  left conductance share: mean {st.mean(lshare):.3f} sd {st.pstdev(lshare):.4f}',
        f'  external share within side: mean {me:.3f} sd {st.pstdev(ext):.3f}',
        f'  R2(ext share ~ area share) {r2(ash, ext):.3f}; residual sd {rsd:.3f}; residual CoV ext {rsd / me:.2f} / int {rsd / (1 - me):.2f}',
        f'  R2*C median {st.median(tau):.3f} s (p5 {sorted(tau)[int(.05 * len(tau))]:.3f}, p95 {sorted(tau)[int(.95 * len(tau)) - 1]:.3f})',
    ]
if AG_CSV.exists():
    csvr = {'AG/' + r['case']: r for r in csv.DictReader(open(AG_CSV))}
    xa, xg, yq = [], [], []
    for r in by['AG']:
        c = csvr.get(r['canonical_id'])
        if not c:
            continue
        try:
            q = {o: float(c[f'vf_out{o}_absmean_m3s']) for o in OUTS}
        except (KeyError, ValueError):
            continue
        if min(q.values()) <= 0:
            continue
        for side, e, i in (('left', 'le', 'li'), ('right', 're', 'ri')):
            yq.append(q[e] / (q[e] + q[i]))
            xa.append(r[f'area_share_{side}'])
            xg.append(r[f'ext_share_{side}'])
    lines += [
        f'=== AG measured within-side external flow share (Fluent monitors), n_cases={len(yq) // 2} ===',
        f'  R2(measured ~ area share) {r2(xa, yq):.3f}; R2(measured ~ RCR conductance share) {r2(xg, yq):.3f}; '
        f'measured sd {st.pstdev(yq):.3f}; residual sd after area {resid_sd(xa, yq):.3f}',
    ]
(HERE / 'rcr_protocol_summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
