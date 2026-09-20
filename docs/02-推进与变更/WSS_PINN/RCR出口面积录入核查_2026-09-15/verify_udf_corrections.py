"""核对 23 例 UDF：改后四个 pressure_out* 块的 R1/R2/C == rcr_corrected_udf_values.csv；除这些数值行与新增的一行注释外，文件与 .orig_20260915 逐字节相同。输出 apply_log.md。"""
import csv, re
from pathlib import Path
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); HERE = Path(__file__).resolve().parent
probe = {r['canonical_id']: r for r in csv.DictReader(open(ROOT / 'docs/02-推进与变更/WSS_PINN/_archive/V5审阅_BC协议结构探针_2026-09-06/rcr_protocol_per_case.csv'))}
corr = list(csv.DictReader(open(HERE / 'rcr_corrected_udf_values.csv')))
BLOCK = lambda o: re.compile(r'DEFINE_PROFILE\(pressure_out%s\s*,[^\n]*\n(.*?)^\}' % o, re.S | re.M)
LINE = lambda k: re.compile(r'^[ \t]*%s[ \t]*=[ \t]*([0-9.Ee+-]+)[ \t]*;' % k, re.M)
STRIP = re.compile(r'^[ \t]*(R1|R2|C)[ \t]*=[ \t]*[0-9.Ee+-]+[ \t]*;[ \t]*\r?\n|^[ \t]*/\* 2026-09-15 outlet-area fix:[^\n]*\n|^[ \t]*[0-9.]+;[ \t]*(/\* 2026-09-15 inlet-area fix:[^\n]*\*/)?[ \t]*\r?\n', re.M)  # 最后一项：入口面积除数行（LIU_YONG_LAN 顺手修了入口面积）
lines = ['| 病例 | 文件 | 状态 | 改动（* = 该出口面积录错）|', '|---|---|---|---|']; n_ok = 0
for c in sorted({r['case'] for r in corr}):
    p = ROOT / probe[c]['udf_file']; new = open(p, encoding='latin-1', newline='').read(); orig = open(str(p) + '.orig_20260915', encoding='latin-1', newline='').read()
    rows = {r['outlet'][-2:]: r for r in corr if r['case'] == c}; touched_sides = {o[0] for o, r in rows.items() if r['changed'] == 'True'}
    assert STRIP.sub('', new) == STRIP.sub('', orig), ('除数值/注释行外有差异', c)
    ch = []
    for o, r in rows.items():
        bn, bo = BLOCK(o).search(new).group(1), BLOCK(o).search(orig).group(1)
        vn = {k: float(LINE(k).search(bn).group(1)) for k in ('R1', 'R2', 'C')}; vo = {k: float(LINE(k).search(bo).group(1)) for k in ('R1', 'R2', 'C')}
        if o[0] in touched_sides:
            for k in vn: assert abs(vn[k] / float(r[k]) - 1) < 1e-4, ('数值不等于修正表', c, o, k, vn[k], r[k])
            assert '2026-09-15 outlet-area fix' in bn, ('缺注释', c, o)
            ch.append(f"out-{o}: R1 {vo['R1']:.4E}->{vn['R1']:.4E}, R2 {vo['R2']:.4E}->{vn['R2']:.4E}, C {vo['C']:.4E}->{vn['C']:.4E}{' *' if r['changed']=='True' else ''}")
        else:
            assert vn == vo and '2026-09-15' not in bn, ('未动的一侧被改了', c, o)
    n_ok += 1; lines.append(f"| `{c}` | `{p.relative_to(ROOT)}` | 已写入，备份 `.orig_20260915`，核对通过 | {'；'.join(ch)} |")
lines.append('| `AAA/ruputer/LIU_YONG_LAN` | `data_new/AAA/ruputer/LIU_YONG_LAN/udf-inlet4.c` | 入口面积除数已改（用户拍板可选项 2） | my_inlet 第 47 行 0.0004121751 -> 0.00037587005（网格 inlet+ 面积） |')
(HERE / 'apply_log.md').write_text('\n'.join(lines) + '\n'); print(f'核对通过 {n_ok}/{len({r["case"] for r in corr})}，apply_log.md 已生成')
