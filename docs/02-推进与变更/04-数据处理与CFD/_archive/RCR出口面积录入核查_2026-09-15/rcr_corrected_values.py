"""按操作者表格 `计算R和C-质量流量-new计算结果.xlsx` 的公式，用网格出口面积重算需要修正的病例的 R1/R2/C（可直接替换 UDF 的块）。
公式：r=sqrt(s/π)·1000 mm；d=(2r)³/((2r_nei)³+(2r_wai)³)；flow=A1·d·0.5；Rt=93.33·133/flow；C=1.79/Rt；R1=13.3/(2r)^0.3/s；R2=Rt−R1。
A1（总质量流量）逐例从 UDF 精确反推：R1 反解表内隐含半径 → 表内 d → A1 = 2P/(Rt·d)（四个出口应一致）。
修正范围（2026-09-15 用户拍板"应该改的和建议改的都改"）：
  病例入选 = 任一出口面积录错（verdict != ok）或任一侧 |UDF 髂外份额 − 网格 r³ 份额| > 0.05；
  入选病例里，某侧 side_dev > 0.03 的两个出口都重写（另一侧原样保留）。"""
import csv, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
SIDE_NEW, SIDE_RERUN = 0.05, 0.03
recs = list(csv.DictReader(open(HERE / 'rcr_area_audit.csv')))
by = {}
for r in recs: by.setdefault(r['case'], {})[r['outlet']] = r
P = 93.33 * 133
def rcr(s_nei, s_wai, A1):
    r_n, r_w = np.sqrt(s_nei / np.pi) * 1e3, np.sqrt(s_wai / np.pi) * 1e3; out = {}
    for name, r, s in (('nei', r_n, s_nei), ('wai', r_w, s_wai)):
        d = (2 * r) ** 3 / ((2 * r_n) ** 3 + (2 * r_w) ** 3); flow = A1 * d * 0.5; Rt = P / flow; R1 = 13.3 / (2 * r) ** 0.3 / s
        out[name] = dict(d=d, Rt=Rt, C=1.79 / Rt, R1=R1, R2=Rt - R1)
    return out
def r_from_R1(R1):  # 精确反解 R1 = 13.3/((2r)^0.3 · π (r·1e-3)²)，r 单位 mm
    f = lambda r: 13.3 / ((2 * r) ** 0.3 * np.pi * (r * 1e-3) ** 2) - R1; lo, hi = 0.1, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return mid
A1_case, A1_cv = {}, {}
for c, d in by.items():
    ri = {o: r_from_R1(float(d[o]['R1'])) for o in d}; vals = []
    for ext, inte in (('le', 'li'), ('re', 'ri')):
        dd = (2 * ri[ext]) ** 3 / ((2 * ri[ext]) ** 3 + (2 * ri[inte]) ** 3)
        for o, dv in ((ext, dd), (inte, 1 - dd)): vals.append(2 * P / ((float(d[o]['R1']) + float(d[o]['R2'])) * dv))
    A1_case[c] = float(np.median(vals)); A1_cv[c] = float(np.std(vals) / np.mean(vals))
for coh in ('AG', 'AAA', 'ILO'):
    v = [A1_case[c] for c in by if by[c]['le']['cohort'] == coh]; cv = [A1_cv[c] for c in by if by[c]['le']['cohort'] == coh]
    print(f'A1 {coh}: 中位 {np.median(v):.5f} kg/s，min/max {min(v):.5f}/{max(v):.5f}，四出口一致性 cv 最大 {max(cv):.1e}，n={len(v)}')
def side_dev(c, side): return float(by[c][side + 'e']['side_dev'])
bad = sorted(c for c, d in by.items() if any(r['verdict'] != 'ok' for r in d.values()) or max(side_dev(c, 'l'), side_dev(c, 'r')) > SIDE_NEW)
rows, md = [], ['| 病例 | A1 kg/s | 出口 | 应填面积 s (m²) | R1 | R2 | C | 髂外份额 d（原 UDF → 新） |', '|---|---|---|---|---|---|---|---|']
for c in bad:
    d = by[c]; A1 = A1_case[c]
    for ext, inte in (('le', 'li'), ('re', 'ri')):
        out = rcr(float(d[inte]['area_mesh']), float(d[ext]['area_mesh']), A1); rewrite = side_dev(c, ext[0]) > SIDE_RERUN or any(d[o]['verdict'] != 'ok' for o in (ext, inte))
        for o, nm in ((ext, 'wai'), (inte, 'nei')):
            v = out[nm]
            rows.append(dict(case=c, A1=f'{A1:.5f}', A1_source='udf-implied', outlet=f'out-{o}', udf_block=f'pressure_out{o}', area_mesh_m2=f"{float(d[o]['area_mesh']):.6e}", R1=f"{v['R1']:.4E}", R2=f"{v['R2']:.4E}", C=f"{v['C']:.4E}", d_new=f"{v['d']:.3f}", d_udf=f"{float(d[o]['ext_share_udf']):.3f}", R1_udf=f"{float(d[o]['R1']):.4E}", R2_udf=f"{float(d[o]['R2']):.4E}", C_udf=f"{float(d[o]['C']):.4E}", changed=rewrite, mis_entered=d[o]['verdict'] != 'ok', side_dev=f"{side_dev(c, ext[0]):.3f}"))
            if v['R2'] <= 0: print('!! R2 <= 0:', c, o, v)
        tag = lambda o: (' ←改' if rewrite else '') + ('(录错)' if d[o]['verdict'] != 'ok' else '')
        md.append(f"| `{c}` | {A1:.4f} | out-{ext}{tag(ext)} / out-{inte}{tag(inte)} | {float(d[ext]['area_mesh']):.4e} / {float(d[inte]['area_mesh']):.4e} | {out['wai']['R1']:.4E} / {out['nei']['R1']:.4E} | {out['wai']['R2']:.4E} / {out['nei']['R2']:.4E} | {out['wai']['C']:.4E} / {out['nei']['C']:.4E} | {float(d[ext]['ext_share_udf']):.3f} → {out['wai']['d']:.3f}{'' if rewrite else '（不动）'} |")
with open(HERE / 'rcr_corrected_udf_values.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
open(HERE / '_corrected_table.md', 'w').write('\n'.join(md) + '\n')
print(f'入选 {len(bad)} 例；重写的侧 {sum(1 for r in rows if r["changed"]) // 2} 个；病例：', ', '.join(bad))
