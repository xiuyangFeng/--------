"""把 rcr_corrected_udf_values.csv 里 changed=True 的出口块（R1/R2/C）写回各病例 UDF。幂等：某侧已是修正值则跳过；是原值则改；否则报错。
用法：python apply_udf_corrections.py [--dry-run] [--only CASE ...]
安全：原文件首次修改前备份为 <name>.orig_20260915；换行风格保留；改后重新解析核对；libudf/src 的编译副本不动。"""
import argparse, csv, re, shutil
from pathlib import Path
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--only', nargs='*', default=None); args = ap.parse_args()
probe = {r['canonical_id']: r for r in csv.DictReader(open(ROOT / 'docs/02-推进与变更/_archive/WSS_PINN/_archive/V5审阅_BC协议结构探针_2026-09-06/rcr_protocol_per_case.csv'))}
corr = list(csv.DictReader(open(HERE / 'rcr_corrected_udf_values.csv')))
BLOCK = lambda o: re.compile(r'(DEFINE_PROFILE\(pressure_out%s\s*,[^\n]*\n)(.*?)(^\})' % o, re.S | re.M)
LINE = lambda k: re.compile(r'^([ \t]*)%s[ \t]*=[ \t]*([0-9.Ee+-]+)[ \t]*;' % k, re.M)
close = lambda a, b: abs(float(a) / float(b) - 1) < 1e-4
for c in sorted({r['case'] for r in corr}):
    if args.only is not None and c not in args.only: continue
    rows = {r['outlet'][-2:]: r for r in corr if r['case'] == c and r['changed'] == 'True'}
    if not rows: continue
    p = ROOT / probe[c]['udf_file']; s = open(p, encoding='latin-1', newline='').read(); new = s; NL = '\r\n' if '\r\n' in s else '\n'; log = []
    for o, r in rows.items():
        m = BLOCK(o).search(s); assert m, (c, o); head, body, tail = m.groups()
        cur = {k: LINE(k).search(body).group(2) for k in ('R1', 'R2', 'C')}
        if all(close(cur[k], r[k]) for k in cur): log.append(f'out-{o}: 已是修正值'); continue
        is_orig = all(close(cur[k], r[k + '_udf']) for k in cur); is_prev = close(cur['R1'], r['R1']) and '2026-09-15 outlet-area fix' in body  # 上一轮修正值：R1 只依赖网格面积所以已相等，R2/C 因 A1 推断略有出入
        assert is_orig or is_prev, ('文件现值既不是原值也不是修正值', c, o, cur)
        body2 = body
        for k in ('R1', 'R2', 'C'): body2 = LINE(k).sub(lambda mm, k=k: f'{mm.group(1)}{k} = {float(r[k]):.4E};', body2, count=1)
        why = ' (this outlet area was mis-entered)' if r['mis_entered'] == 'True' else ' (side split off mesh Murray -> Rt/C changed)'
        comment = f'  /* 2026-09-15 outlet-area fix: s_mesh={float(r["area_mesh_m2"]):.4e} m2, A1={r["A1"]} kg/s; old R1={float(r["R1_udf"]):.4E} R2={float(r["R2_udf"]):.4E} C={float(r["C_udf"]):.4E}{why} */' + NL
        if '2026-09-15 outlet-area fix' in body: comment = ''  # 已有注释（记录的是真正的原值），不再加
        new = new.replace(head + body + tail, head + comment + body2 + tail, 1)
        log.append(f"out-{o}: {'原值' if is_orig else '旧修正值'} R1 {cur['R1']}->{float(r['R1']):.4E}, R2 {cur['R2']}->{float(r['R2']):.4E}, C {cur['C']}->{float(r['C']):.4E}")
    if new != s:
        for o, r in rows.items():
            body = BLOCK(o).search(new).group(2); assert all(close(LINE(k).search(body).group(2), r[k]) for k in ('R1', 'R2', 'C')), ('改后核对失败', c, o)
        if not args.dry_run:
            bak = p.with_name(p.name + '.orig_20260915'); shutil.copy2(p, bak) if not bak.exists() else None
            open(p, 'w', encoding='latin-1', newline='').write(new)
    print(f"{c:36s} {'DRY-RUN' if args.dry_run else ('已写入' if new != s else '无需改动')}  {'；'.join(log)}")
