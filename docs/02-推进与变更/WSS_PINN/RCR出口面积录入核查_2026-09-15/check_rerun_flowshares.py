"""重跑核对：从 Fluent transcript 里 UDF 每步打印的 Q_ave_out{le,li,ri,re}（kg/s）算各出口流量份额，
与旧运行（Global_conditions_old_20260915/Fluent_*.out）及 UDF 设定（旧/新 d）对比。
用法：python check_rerun_flowshares.py [病例目录 ...]   默认 = rerun_cases.txt 里已有 Fluent_*.out 的病例。"""
import csv, re, sys
import numpy as np
from pathlib import Path
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); HERE = Path(__file__).resolve().parent
corr = {(r['case'], r['outlet'][-2:]): r for r in csv.DictReader(open(HERE / 'rcr_corrected_udf_values.csv'))}
audit = {(r['case'], r['outlet']): r for r in csv.DictReader(open(HERE / 'rcr_area_audit.csv'))}
PAT = re.compile(r'Q_ave_out(le|li|ri|re)=(-?[0-9.]+(?:e[-+]?\d+)?)')
def parse(f):
    q = {o: [] for o in ('le', 'li', 'ri', 're')}; last = {}
    for line in open(f, errors='replace'):
        m = PAT.search(line)
        if not m: continue
        o = m.group(1); s = line.strip()
        if last.get(o) == s: continue   # 每步每个 MPI 进程各打一行，值相同
        last[o] = s; q[o].append(float(m.group(2)))
    n = min(len(v) for v in q.values()); return {o: np.array(v[:n]) for o, v in q.items()}, n
def case_id(cdir):
    rel = str(cdir.relative_to(ROOT / 'data_new')); return rel
dirs = [Path(a).resolve() for a in sys.argv[1:]] or [ROOT / l.strip() for l in open(HERE / 'rerun_cases.txt') if l.strip()]
for cd in dirs:
    outs = sorted(list(cd.glob('Fluent_*.out')) + list((cd / 'Global_conditions').glob('Fluent_*.out')), key=lambda p: p.stat().st_mtime)
    olds = sorted((cd / 'Global_conditions_old_20260915').glob('Fluent_*.out'))
    if not outs: continue
    if not (cd / 'Global_conditions_old_20260915').exists() and not list(cd.glob('Fluent_*.out')):
        print(f'== {case_id(cd)}: 尚未重跑（只有旧运行），跳过'); continue
    case = case_id(cd); qn, nn = parse(outs[-1]); qo, no = parse(olds[-1]) if olds else ({}, 0); n = min(nn, no) if no else nn
    if nn < 60: print(f'== {case}: 只有 {nn} 步，先不评'); continue
    lo = min(40, n - 1); tot_n = sum(qn[o] for o in qn); tot_o = sum(qo[o] for o in qo) if no else None
    sn = {o: float(np.sum(qn[o][lo:n]) / np.sum(tot_n[lo:n])) for o in qn}; so = {o: float(np.sum(qo[o][lo:n]) / np.sum(tot_o[lo:n])) for o in qo} if no else {}
    status = '完成 1280 步' if nn >= 1280 else f'进行中 {nn}/1280 步'
    print(f'== {case}（{status}；对比前 {n} 步）')
    for side, (e, i) in (('左', ('le', 'li')), ('右', ('re', 'ri'))):
        d_old = float(audit[(case, e)]['ext_share_udf']); d_new = float(corr[(case, e)]['d_new']) if (case, e) in corr else d_old
        run_old = so[e] / (so[e] + so[i]) if so else float('nan'); run_new = sn[e] / (sn[e] + sn[i])
        flag = 'OK' if abs(run_new - d_new) < 0.05 else '偏离>0.05'
        print(f'   {side}侧髂外占该侧: UDF 旧 {d_old:.3f} → 新 {d_new:.3f} | 运行 旧 {run_old:.3f} → 新 {run_new:.3f}  {flag}')
    print(f'   左侧占总流量 {sn["le"] + sn["li"]:.3f}（协议 0.5）；各出口份额 新 ' + ' '.join(f'{o}={sn[o]:.3f}' for o in ('le', 'li', 're', 'ri')) + (' | 旧 ' + ' '.join(f'{o}={so[o]:.3f}' for o in ('le', 'li', 're', 'ri')) if so else ''))
