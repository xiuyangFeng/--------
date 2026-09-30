"""RCR 出口面积录入核查（2026-09-15，只读）。
协议：R_total 常量、左右电导 50/50、每侧髂外/髂内按 Murray r^3 分配（r 由出口面积 s 反算）、R1 只依赖出口半径、C = τ/(R1+R2)。
从每例 UDF 的 R1/R2/C 反推操作者表格里填的出口半径/面积，与网格出口面积对比，找出录入错误。
输入：_archive/V5审阅_BC协议结构探针_2026-09-06/rcr_protocol_per_case.csv（UDF 参数 + 网格出口面积）+ case.h5 峰值出口通量。
输出：rcr_area_audit.csv（172 例 × 4 出口）与标准输出的清单。"""
import csv, json, sys
import numpy as np, h5py
from pathlib import Path
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); HERE = Path(__file__).resolve().parent
H5 = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_20260906/cases'
rows = list(csv.DictReader(open(ROOT / 'docs/02-推进与变更/_archive/WSS_PINN/_archive/V5审阅_BC协议结构探针_2026-09-06/rcr_protocol_per_case.csv')))
OUT = ('le', 'li', 're', 'ri'); SIDES = (('le', 'li'), ('re', 'ri'))
def r_mm(a): return np.sqrt(float(a) / np.pi) * 1e3
recs = []
for x in rows:
    with h5py.File(H5 / (x['canonical_id'].replace('/', '__')) / 'case.h5', 'r') as h:
        step = h['wall_temporal/step'][:]; k = int(np.where(step == 1162)[0][0]) if 1162 in step else int(np.argmax(np.abs(h['interfaces/inlet/flux_outward_m3s'][:])))
        q = {o: float(h[f'interfaces/out-{o}/flux_outward_m3s'][k]) for o in OUT}
    tot = sum(q.values())
    for o in OUT:
        R1, R2, C, A = (float(x[f'R1_out{o}']), float(x[f'R2_out{o}']), float(x[f'C_out{o}']), float(x[f'area_out{o}_m2']))
        recs.append(dict(case=x['canonical_id'], cohort=x['cohort'], role=x['role'], outlet=o, R1=R1, R2=R2, C=C, tau=(R1 + R2) * C,
                         area_mesh=A, r_mesh=r_mm(A), G=1.0 / (R1 + R2), q_share_peak=q[o] / tot))
# ---- R1 ~ r 幂律（逐队列稳健拟合，剔除 |残差|>0.2 后重拟合）
for coh in ('AG', 'AAA', 'ILO'):
    idx = [i for i, r in enumerate(recs) if r['cohort'] == coh]
    lr = np.log([recs[i]['r_mesh'] for i in idx]); lR = np.log([recs[i]['R1'] for i in idx]); m = np.ones(len(idx), bool)
    for _ in range(5):
        p, a = np.polyfit(lr[m], lR[m], 1); res = lR - (a + p * lr); m = np.abs(res) < 0.2
    print(f'R1 幂律 {coh}: log R1 = {a:.3f} {p:+.3f} log r_mm；内点 {m.sum()}/{len(idx)}，内点残差 sd {res[m].std():.3f}')
    for i, j in enumerate(idx):
        recs[j]['r_implied_R1'] = float(np.exp((np.log(recs[j]['R1']) - a) / p)); recs[j]['area_ratio_R1'] = (recs[j]['r_implied_R1'] / recs[j]['r_mesh']) ** 2
# ---- 每侧：UDF 电导份额 vs 网格 r^3 份额
by_case = {}
for r in recs: by_case.setdefault(r['case'], {})[r['outlet']] = r
for cid, d in by_case.items():
    for ext, inte in SIDES:
        e, n = d[ext], d[inte]
        d_udf = e['G'] / (e['G'] + n['G']); d_mesh = e['r_mesh'] ** 3 / (e['r_mesh'] ** 3 + n['r_mesh'] ** 3)
        for r in (e, n): r['ext_share_udf'] = d_udf; r['ext_share_mesh_r3'] = d_mesh; r['side_dev'] = abs(d_udf - d_mesh)
# ---- 判定
def classify(r, sib):
    ar = r['area_ratio_R1']; lg = np.log10(ar)
    if abs(lg) < np.log10(1.5): return 'ok'
    if abs(abs(lg) - 1) < 0.12: return f'小数点错一位（表内面积≈网格×{ar:.2f}）'
    if abs(np.log(r['r_implied_R1'] ** 2 * np.pi / sib['area_mesh'] * 1e-6)) < np.log(1.15): return f'与同侧另一出口面积互换（表内≈{sib["outlet"]} 的面积）'
    return f'面积不符（表内≈网格×{ar:.2f}）'
for cid, d in by_case.items():
    for ext, inte in SIDES:
        d[ext]['verdict'] = classify(d[ext], d[inte]); d[inte]['verdict'] = classify(d[inte], d[ext])
cols = ['case', 'cohort', 'role', 'outlet', 'area_mesh', 'r_mesh', 'R1', 'R2', 'C', 'tau', 'r_implied_R1', 'area_ratio_R1', 'ext_share_udf', 'ext_share_mesh_r3', 'side_dev', 'q_share_peak', 'verdict']
with open(HERE / 'rcr_area_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in recs: w.writerow({c: (f'{r[c]:.6g}' if isinstance(r[c], float) else r[c]) for c in cols})
tau = np.array([r['tau'] for r in recs]); print(f'\n协议核对：τ=(R1+R2)·C 中位 {np.median(tau):.3f} s，p1/p99 {np.quantile(tau,.01):.3f}/{np.quantile(tau,.99):.3f}')
bad_cases = sorted({r['case'] for r in recs if r['verdict'] != 'ok'}, key=lambda c: -max(by_case[c][o]['side_dev'] for o in OUT))
print(f'\n出口面积录入与网格不符的病例：{len(bad_cases)} 例（出口 {sum(r["verdict"]!="ok" for r in recs)} 个）')
print('case | role | 出口 | 网格面积 m² | 表内隐含面积 m² | 比值 | 该侧髂外份额 UDF/网格r³ | 该出口峰值流量份额 | 判定')
for c in bad_cases:
    for o in OUT:
        r = by_case[c][o]
        if r['verdict'] != 'ok':
            print(f"{c:36s} {r['role']:5s} out-{o}  {r['area_mesh']:.3e}  {r['r_implied_R1']**2*np.pi*1e-6:.3e}  x{r['area_ratio_R1']:.3f}  {r['ext_share_udf']:.3f}/{r['ext_share_mesh_r3']:.3f}  {r['q_share_peak']:+.3f}  {r['verdict']}")
# 之前按 |UDF 髂外电导份额 − 面积份额| > 0.4 圈出的 12 例是否都被解释
dev12 = sorted({r['case'] for r in recs if abs(float([x for x in rows if x['canonical_id']==r['case']][0]['ext_share_left']) - float([x for x in rows if x['canonical_id']==r['case']][0]['area_share_left'])) > 0.4 or abs(float([x for x in rows if x['canonical_id']==r['case']][0]['ext_share_right']) - float([x for x in rows if x['canonical_id']==r['case']][0]['area_share_right'])) > 0.4})
print('\n此前 12 例：', [(c, '已解释' if c in bad_cases else '未解释') for c in dev12])
