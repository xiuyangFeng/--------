import json, re, collections
P='/tmp/wss_audit.json'
x=json.load(open(P))
def annotate(c):
    n=c['source_row'];h=c['headers'];v=c['values'];ref='unknown';ev='源表未给出可明确确定的参照';pc=None
    refs={}
    if 71<=n<=143 and h.get('B')=='对照':
        ref=str(v['B']);ev=f'汇总对比!B{n}明确列为“对照”；各区块说明Δ相对reference_id';pc='D'
    elif 147<=n<=153:
        ref=str(v['C']);ev=f'汇总对比!C{n}明确列为“对照臂”';pc='E'
    elif 157<=n<=168 or 175<=n<=186:
        ref='S3-GEOPE 父模板｜seed '+str(v['B']);ev='汇总对比!A155 / A173：与S3父模板同seed配对';pc='D'
    elif n in [170,171,172]:
        ref='S3-GEOPE 父模板｜各seed分别配对后取均值';ev='汇总对比!A169：三种子 vs S3 父（同种子）ΔR²_cb均值';pc='C'
    elif 189<=n<=192:
        ref='S3-GEOPE 父模板｜各seed分别配对后取均值';ev='汇总对比!A173：正则化三种子矩阵（vs 同 seed S3 父模板）';pc='B'
    elif n==200:
        ref='S3-GEOPE｜seed1234';ev='汇总对比!A198：S3-GEOPE + SA3 coarse attention，seed1234';pc='D'
    elif 203<=n<=222:
        ref='S3-GEOPE｜seed1234';ev='汇总对比!C202：ΔR²_cb vs S3；A201：seed1234';pc='C'
    elif n in [225,226]:
        ref='S2-PNXR '+('xyz-only' if n==225 else 'xyz+geom')+' 父模型｜seed1234';ev=f'汇总对比!A{n}明确比较名 '+str(v['A']);pc='C'
    elif n==229:
        ref='D2 c125×k64｜106/0/27｜seed1234';ev='汇总对比!A227与B228（D2 R²_cb）';pc='D'
    elif 232<=n<=239 or 246<=n<=248:
        ref='REG-P10｜seed1234';ev='汇总对比!A230 / A244：REG-P10模块矩阵';pc='D'
        if 'O' in v:refs['O']=str(v['A'])+' 的best（同臂last−best）'
    elif 242<=n<=243:
        ref='D2 PNXR+GeoPE｜106/0/27｜seed1234';ev='汇总对比!A240：固定 D2 PNXR+GeoPE：local-SA1 / local-SA2';pc='C'
        if 'N' in v:refs['N']=ref+' 的last（处理last−parent last）'
    elif 251<=n<=253 or n in [256,257]:
        ref='RCR-O0 geometry｜seed1234';ev='汇总对比!A251：geometry基准；A254：O0 hotspot BCE / q90 pinball单变量筛选';pc='D'
    elif 260<=n<=265:
        ref='LSA2 '+('IND' if n in [264,265] else 'SAME')+' MSE｜seed1234';ev='汇总对比!A258–O265：SAME/IND × MSE/H1/H2并发矩阵；D列差值与各自MSE基线一致，IND MSE自身与SAME MSE比较';pc='D'
    elif n in [268,269]:
        ref='LSA2 H2 concurrent repro｜seed1234';ev='汇总对比!A266及N268：正式并发对照';pc='D'
    elif 325<=n<=350:
        if v.get('N') not in [None,'—','']:
            ref=str(v['N']);ev=f'汇总对比!N{n}明确列为“参照”';pc='D'
    elif 359<=n<=370 or 374<=n<=385:
        ref='V5 R4 基准｜5seed指标均值';ev='汇总对比!A356、A357、A371、A372：参照R4全部5seed均值';pc='E' if n<=370 else 'D'
    elif 388<=n<=403:
        ref='V6 A5｜seed1234｜best';ev='汇总对比!E387：Δ vs A5；B388：B0=A5；F387：Δ vs父臂';pc='E'
        if 'F' in v:refs['F']=str(v.get('B','unknown'))+'｜seed1234｜best（父臂）'
    elif 407<=n<=413:
        ref='MS0=M2｜seed1234｜best';ev='汇总对比!E406：Δ vs MS0；A407：MS0=M2；F406：Δ vs父臂';pc='E'
        if 'F' in v:refs['F']=str(v.get('B','unknown'))+'｜seed1234｜best（父臂）'
    elif 421<=n<=441:
        ref='historical_m2｜seed1234｜best';ev='汇总对比!E420/F420/H420/I420分别指定checkpoint与参照';pc='E'
        refs.update({'E':ref,'F':'MO0｜seed1234｜best','H':'historical_m2｜seed1234｜last','I':'MO0｜seed1234｜last'})
    for k,label in h.items():
        if k in v and isinstance(v[k],(int,float)) and ('Δ' in label or 'last−' in label):refs.setdefault(k,ref)
    if 203<=n<=222 and 'G' in v:refs['G']='unknown（pair interaction是交互效应，不能视作相对单个父臂）'
    c['reference']=ref
    c['reference_evidence']=ev
    c['reference_by_column']=refs
    if pc in v and isinstance(v[pc],(int,float)):
        c['primary_delta_column']=pc;c['primary_delta']=v[pc];c['primary_delta_label']=h.get(pc,pc)
        c['primary_delta_reference']=refs.get(pc,ref)
    else:
        c['primary_delta_column']=None;c['primary_delta']=None;c['primary_delta_label']=None;c['primary_delta_reference']='unknown'
for c in x['supplemental_rows']:annotate(c)
for r in x['records']:
    for c in r.get('comparisons',[]):annotate(c)
x['comparison_reference_schema']={'reference':'主要物理R²差值的参照（unknown=无法明确）','reference_evidence':'来源位置与明确证据','reference_by_column':'同一比较行各差值列的参照；多参照时必须按列取，不可统一覆盖','primary_delta_column':'建议前置展示的物理R²差值列；无数值为null','primary_delta':'对应原单元格数值','primary_delta_reference':'仅与primary_delta一一配对的参照','primary_delta_label':'原表头，保留best/last等信息'}
json.dump(x,open(P,'w'),ensure_ascii=False,indent=2)
print(json.dumps({'comparisons':len(x['supplemental_rows']),'known_reference':sum(c['reference']!='unknown' for c in x['supplemental_rows']),'paired_primary_delta':sum(c['primary_delta'] is not None for c in x['supplemental_rows']),'records':len(x['records'])},ensure_ascii=False))
