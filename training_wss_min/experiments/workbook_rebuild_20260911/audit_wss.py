import json, math, re, copy, statistics
from pathlib import Path
import openpyxl

ROOT=Path('/public/newhome/cy/Digital_twin/GNN')
X=json.load(open('/tmp/wss_workbook_inventory.json'))
S={s:{r['row']:r['values'] for r in v['rows']} for s,v in X.items()}
W=openpyxl.load_workbook(ROOT/'docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx')
headers=S['实验矩阵总览'][2]
def get(d,p):
    for k in p.split('.'):
        if not isinstance(d,dict):return None
        d=d.get(k)
    return d
MAP={'G':'field_casebalanced.r2','H':'field.r2','I':'aggregate.r2_casemean','J':'aggregate.r2_casemed','K':'aggregate.r2_casep10','M':'field.rmse','N':'field.mae','O':'field.nrmse_range','P':'aggregate.nrmse_casemean','Q':'field.nmae_range','R':'aggregate.nmae_casemean','S':'regional_field.high_wss.r2','T':'calibration.top10_pred_true_ratio','U':'hotspot.top10_iou_casemean','V':'normalized.field_casebalanced.r2','W':'normalized.field.r2','X':'normalized.aggregate.r2_casemean','Y':'normalized.aggregate.r2_casemed','Z':'normalized.aggregate.r2_casep10','AA':'normalized.field.mae','AB':'normalized.field.rmse','AC':'normalized.field.nrmse_range','AD':'normalized.aggregate.nrmse_casemean','AE':'normalized.field.nmae_range','AF':'normalized.aggregate.nmae_casemean','AG':'normalized.hotspot.spearman_all_casemean','AH':'normalized.hotspot.spearman_high_wss_casemean','AI':'normalized.calibration.top10_pred_true_ratio','AJ':'normalized.calibration.p99_pred_true_ratio','AL':'field_casebalanced.r2_linear_fit','AM':'field_casebalanced.linear_fit_slope','AN':'field_casebalanced.linear_fit_intercept','AO':'field.r2_linear_fit','AP':'field.linear_fit_slope','AQ':'field.linear_fit_intercept','AR':'normalized.field_casebalanced.r2_linear_fit','AS':'normalized.field_casebalanced.linear_fit_slope','AT':'normalized.field_casebalanced.linear_fit_intercept','AU':'normalized.field.r2_linear_fit','AV':'normalized.field.linear_fit_slope','AW':'normalized.field.linear_fit_intercept'}
EXTRA={'Pa p99比':'calibration.p99_pred_true_ratio','Pa high-WSS MAE':'regional_field.high_wss.mae','Pa high-WSS RMSE':'regional_field.high_wss.rmse','Pa high-WSS range-NRMSE':'regional_field.high_wss.nrmse_range','Pa MAE（病例等权）':'field_casebalanced.mae','Pa RMSE（病例等权）':'field_casebalanced.rmse','归一化 MAE（病例等权）':'normalized.field_casebalanced.mae','归一化 RMSE（病例等权）':'normalized.field_casebalanced.rmse','物理 Spearman':'hotspot.spearman_all_casemean','物理 high-WSS Spearman':'hotspot.spearman_high_wss_casemean','归一化 high-WSS R²':'normalized.regional_field.high_wss.r2','AG 物理R²_cb':'group_casebalanced.AG.r2','AAA 物理R²_cb':'group_casebalanced.AAA.r2','ILO 物理R²_cb':'group_casebalanced.ILO.r2','参数量':'efficiency.parameters'}
def extract(d):
    z={k:get(d,p) for k,p in MAP.items() if get(d,p) is not None}
    nc=get(d,'aggregate.n_cases');neg=get(d,'aggregate.r2_negative_cases')
    if nc is not None and neg is not None:z['L']=f'{int(neg)}/{int(nc)}'
    return z,{k:get(d,p) for k,p in EXTRA.items() if get(d,p) is not None}
def norm(x):return re.sub(r'[★\s]+','',str(x))

# Build a light index, retaining only metric dictionaries that can match table records.
idx=[]
paths=list((ROOT/'training_wss_min/runs').rglob('metrics.json'))+list((ROOT/'training_wss_min/experiments/v5_rerun_20260906/legacy_overlap34').rglob('metrics.json'))+[ROOT/'training_wss_min/experiments/v5_velocity_to_wss_20260909/metrics.json']
for p in paths:
    try: data=json.load(open(p))
    except Exception:continue
    for partition,d in data.items():
        if not isinstance(d,dict):continue
        g=get(d,'field_casebalanced.r2');v=get(d,'normalized.field_casebalanced.r2')
        if isinstance(g,(float,int)) or isinstance(v,(float,int)):
            vals,ex=extract(d)
            idx.append({'path':str(p),'partition':partition,'g':g,'v':v,'values':vals,'extra':ex,'space':d.get('metric_space'),'back_transform':d.get('back_transform'),'normalization':get(d,'normalized.normalization')})

# Map original referenced rows to actual comparison matrix headings.
group_by_row={}; group='历史基线'
for n,v in S['汇总对比'].items():
    if len(v)==1 and 'A' in v and not str(v['A']).startswith('='):group=v['A']
    for value in v.values():
        if isinstance(value,str) and value.startswith('='):
            for r in re.findall(r'实验矩阵总览!A(\d+)',value):group_by_row[int(r)]=group
manual=[(3,9,'早期 PointNet｜test16'),(10,13,'v4架构与AAA数据｜test15'),(14,21,'分层划分与Phase-V｜test27'),(22,27,'Q2V数据扩容与迁移'),(28,33,'Q2V架构搜索｜val21'),(34,34,'Q2V零样本迁移｜test36'),(37,40,'Q2V半径与采样探索｜test27'),(42,44,'Q1V半径探索｜test27'),(46,53,'QAD-Lite三seed配对｜val21'),(55,61,'QAD精确Q2V三seed｜test27'),(63,63,'Q1V半径0.6×补充｜test27'),(65,84,'SA1覆盖与重叠矩阵｜test27'),(85,90,'归一化口径×尾部损失｜pool2025 test36'),(92,109,'SA1尺度与容量矩阵｜test27'),(110,118,'ILO数据扩容与S0–S5骨干矩阵'),(119,130,'S3口径与域条件根因矩阵'),(131,142,'S3正则化三seed矩阵'),(143,145,'三锚点纯XYZ输入对照'),(146,146,'S3 SA3全局注意力'),(147,166,'正则化强度与两两交叉'),(167,167,'S2 XYZ + LocalGeoPE'),(168,168,'D2 + PointNeXt-R + LocalGeoPE'),(169,176,'REG-P10局部SA注意力矩阵'),(177,178,'D2局部SA注意力'),(179,181,'REG-P10 EdgeConv矩阵'),(182,184,'RCR输入对照（WSS任务）'),(185,186,'Hotspot BCE / Pinball筛选'),(187,192,'LSA2 SAME/IND × MSE/H1/H2'),(193,194,'LSA2并发复现与log半径'),(195,199,'旧checkpoint在test34交集重评'),(200,218,'V5新数据与几何/损失消融'),(219,220,'时间条件模型｜峰值1162'),(227,253,'Bottleneck Transformer矩阵'),(255,265,'V6单帧WSS矩阵'),(267,281,'V6-A5后续｜D/L/M矩阵'),(283,288,'V6多半径BT矩阵'),(290,290,'预测速度派生WSS'),(293,313,'M2优化｜基础矩阵')]
for lo,hi,g in manual:
    for r in range(lo,hi+1):group_by_row[r]=g

records=[];corrections=[];missing=[]
for n,v in S['实验矩阵总览'].items():
    if not(3<=n<=313) or 222<=n<=225 or n==292 or not isinstance(v.get('C'),str) or not isinstance(v.get('F'),(int,float,str)):continue
    if len(v)<20:continue
    rec={'source_sheet':'实验矩阵总览','source_row':n,'group':group_by_row.get(n,'历史WSS'),'record_type':'seed_mean' if '3seed均值' in str(v.get('A')) else 'run','checkpoint':'历史未标注','values':copy.deepcopy(v),'extra_metrics':{},'notes':[],'source_paths':[],'comparisons':[]}
    for c in W['实验矩阵总览'][n]:
        if c.comment and '|slot:' not in c.comment.text:rec['notes'].append(f'旧表{c.coordinate}批注：{c.comment.text}')
    matches=[]
    for it in idx:
        g,vv=v.get('G'),v.get('V');a,b=it['g'],it['v']
        if isinstance(g,(int,float)) and isinstance(a,(int,float)) and abs(g-a)<1e-8 and (not isinstance(vv,(int,float)) or not isinstance(b,(int,float)) or abs(vv-b)<1e-8):matches.append(it)
    early_paths={3:'pointnet_trainloss_e400/outputs/pointnet_xyzgeom',4:'pointnet_distribution_matrix/outputs/e2_global',5:'pointnet_distribution_matrix/outputs/e3_global',6:'pointnet_distribution_matrix/outputs/e23_global',9:'pointnet_deeper/outputs/e4_deep_global'}
    if n in early_paths:
        exact=[it for it in idx if early_paths[n]+'/eval/ckpt_best/' in it['path'] and it['partition']=='test']
        if not exact:exact=[it for it in idx if early_paths[n]+'/eval/' in it['path'] and it['partition']=='test']
        if exact:matches=exact
    if matches:
        ak=str(v.get('AK',''))
        matches.sort(key=lambda z:((ak not in z['path'] and ak.split('/')[-1] not in z['path']), 'ckpt_best' not in z['path'],len(z['path'])))
        m=matches[0];rec['source_paths'].append(m['path']);rec['checkpoint']='best' if '/ckpt_best/' in m['path'] else ('last' if '/ckpt_last/' in m['path'] else '历史默认评估')
        rec['extra_metrics'].update(m['extra'])
        rec['metric_partition']=m['partition'];rec['metric_normalization']=m['normalization']
        for k,z in m['values'].items():
            if k not in rec['values']:
                rec['values'][k]=z;corrections.append({'row':n,'column':k,'action':'历史产物补齐','value':z,'source':m['path']})
            elif k in ['AA','AB','AJ','AG','AH'] and isinstance(z,(int,float)) and isinstance(rec['values'][k],(int,float)) and abs(rec['values'][k]-z)>1e-6:
                old=rec['values'][k]; rec['values'][k]=z
                reason={'AA':'病例等权MAE错填pooled列','AB':'病例等权RMSE错填pooled列','AJ':'物理p99错填归一化列','AG':'物理Spearman错填归一化列','AH':'物理high-WSS Spearman错填归一化列'}[k]
                rec['notes'].append(f'口径更正：旧{k}={old:.9g}（{reason}），本列现按真实评估产物的{MAP[k]}填写；原口径数值保留补充指标。')
                corrections.append({'row':n,'column':k,'action':reason,'old':old,'value':z,'source':m['path']})
        if m['back_transform']:rec['notes'].append('物理回变换：'+str(m['back_transform']))
        rec['matched_metric_path']=m['path']
    if n>=255:rec['checkpoint']='best'
    if n in [7,8]:
        p=ROOT/'training_wss_min/runs/pointnet_distribution_matrix/outputs'/('e2_case' if n==7 else 'e3_case')/'eval/ckpt_best/metrics.json'
        d=json.load(open(p))['test']; wrapped={'normalized':d}; vals,ex=extract(wrapped)
        for k,z in vals.items():
            if k in MAP and MAP[k].startswith('normalized.') and k not in rec['values']:rec['values'][k]=z
        rec['extra_metrics'].update(ex);rec['source_paths'].append(str(p));rec['checkpoint']='best';rec['metric_normalization']='wss/wssmax'
    if 227<=n<=253:
        stem=v['AK'].split('/')[-1];cp=ROOT/'training_wss_min/configs/v5_rerun_20260906'/f'{stem}.json'
        if cp.exists():
            conf=json.load(open(cp));mo=conf['model'];token=mo['sa_center_counts'][-1]
            att=[]
            if mo.get('bottleneck_transformer'):att.append(f"BT {mo.get('bottleneck_layers',1)}层/{mo.get('bottleneck_heads',8)}头")
            if mo.get('coarse_attention'):att.append('coarse global attention')
            stages=mo.get('local_transformer_stages',[])
            if stages:att.append('local SA'+','.join(map(str,stages)))
            desc=f"中心数 {mo['sa_center_counts']}；粗token {token}；"+' + '.join(att)
            if mo.get('bottleneck_transformer'):desc+=f"；条件 {mo.get('bottleneck_pos_enc')}；几何偏置 {mo.get('bottleneck_geo_bias')}；dropout {mo.get('bottleneck_dropout')}"
            rec['values']['D']=desc;rec['values']['C']='PointNeXt-R + LocalGeoPE + '+' + '.join(att);rec['source_paths'].append(str(cp));rec['notes'].append('已按config更正旧表的“token None / 注意力 无”结构误写。')
            corrections.append({'row':n,'column':'C/D','action':'修正BT结构','source':str(cp)})
    if n in [7,8]:rec['notes'].append('训练目标为逐病例wss/wssmax；旧产物未保存可追溯物理回变换，物理Pa指标缺失，不能用归一化值代填。')
    if n==3:rec['notes'].append('E0旧版评估未保存归一化RMSE/逐点残差，归一化range-NRMSE无法可靠重建。')
    if n in [219,220]:rec['notes'].append('本行仅峰值1162；全81帧/TAWSS补充指标见comparisons及time_runs_tables.md。')
    if rec['record_type']=='seed_mean':rec['checkpoint']='各seed既定checkpoint均值'
    records.append(rec)

# Correct the four original QAD aggregate rows after correcting their member runs.
for dest,srcs in {52:[46,48,50],53:[47,49,51],60:[20,56,58],61:[55,57,59]}.items():
    rec=next(r for r in records if r['source_row']==dest);members=[next(r for r in records if r['source_row']==n) for n in srcs]
    for k in ['AJ','AA','AB']:
        ar=[r['values'].get(k) for r in members]
        if all(isinstance(z,(int,float)) for z in ar):
            old=rec['values'].get(k);val=statistics.mean(ar);rec['values'][k]=val
            if isinstance(old,(int,float)) and abs(val-old)>1e-6:corrections.append({'row':dest,'column':k,'action':'按已校正的三seed原始指标重算均值','old':old,'value':val,'member_rows':srcs})
    for k in EXTRA:
        ar=[r['extra_metrics'].get(k) for r in members]
        if all(isinstance(z,(int,float)) for z in ar):rec['extra_metrics'][k]=statistics.mean(ar)
    rec['member_source_rows']=srcs;rec['source_paths']=sum([r['source_paths'] for r in members],[]);rec['notes'].append('三seed均值已按校正后的各seed指标重算；旧归一化p99误用物理p99，物理值保留补充指标。')

# Keep every static comparison cell with its local header, including confidence intervals,
# cohort deltas, interaction effects, parent reference and decision wording.
blocks=[(70,71,74),(76,77,83),(85,86,91),(93,94,99),(101,102,107),(110,111,118),(120,121,122),(125,126,130),(133,134,139),(142,143,143),(146,147,153),(156,157,168),(174,175,186),(188,189,192),(194,195,197),(199,200,200),(202,203,222),(224,225,226),(228,229,229),(231,232,239),(241,242,243),(245,246,248),(250,251,253),(255,256,257),(259,260,265),(267,268,269),(324,325,350),(358,359,370),(373,374,385),(387,388,403),(406,407,413),(416,417,417),(420,421,441)]
supp=[]
def find_record(n,v):
    a=norm(v.get('A')); candidates=[r for r in records if norm(r['values']['A'])==a]
    if candidates:return candidates[0]
    if 389<=n<=403:return next(r for r in records if r['source_row']==267+n-389)
    if 408<=n<=413:return next(r for r in records if r['source_row']==283+n-408)
    if 421<=n<=441:return next(r for r in records if r['source_row']==293+n-421)
    if n==417:return next(r for r in records if r['source_row']==290)
    if n==200:return next(r for r in records if r['source_row']==146)
    if n in [195,196,197]:return next(r for r in records if r['source_row']==143+n-195)
    if n in [225,226]:return next(r for r in records if r['source_row']==167)
    if n==229:return next(r for r in records if r['source_row']==168)
    special={111:65,126:72,134:67,135:72,136:73,147:110,148:112,149:113,150:114,151:115,152:116,153:118,363:236,366:245,367:246,368:247,388:265,407:278}
    if n in special:return next(r for r in records if r['source_row']==special[n])
    return None
for hn,lo,hi in blocks:
    h=S['汇总对比'][hn]
    for n in range(lo,hi+1):
        v=S['汇总对比'].get(n,{})
        obj={'source_sheet':'汇总对比','source_row':n,'headers':h,'values':v,'named_values':{f'{h.get(k,k)} [{k}]':z for k,z in v.items()}}
        supp.append(obj);rec=find_record(n,v)
        if rec:rec['comparisons'].append(obj)

# Add independent summaries only when the source aggregates multiple seed runs.
for n in range(202,214):
    tv=S['教师汇报视图'][n];name=str(tv['B']);match=re.search(r'（(\d+) seed 均值）',name)
    if not match:continue
    stem=name[:match.start()];members=[r for r in records if r['source_row'] and 227<=r['source_row']<=253 and r['values']['A'].split('·s')[0]==stem]
    if stem.startswith('R4 基准'):members=[r for r in records if r['source_row'] in [207,208,209,217,218]]
    if not members:continue
    vals={'A':name,'B':'V5 train138/test34；跨seed算术均值','C':members[0]['values']['C'],'D':members[0]['values']['D'],'E':members[0]['values']['E'],'F':members[0]['values']['F'],'AK':'均值：'+','.join(str(r['values'].get('AK','')) for r in members)}
    for k in MAP:
        ar=[r['values'].get(k) for r in members]
        if all(isinstance(z,(int,float)) for z in ar):vals[k]=statistics.mean(ar)
    ex={}
    for k in EXTRA:
        ar=[r['extra_metrics'].get(k) for r in members]
        if all(isinstance(z,(int,float)) for z in ar):ex[k]=statistics.mean(ar)
    ex['seed数']=len(members)
    rec={'source_sheet':'教师汇报视图','source_row':n,'group':'Bottleneck Transformer矩阵｜多seed汇总','record_type':'seed_mean','checkpoint':'各seed best均值','values':vals,'extra_metrics':ex,'notes':['跨seed指标算术均值；不等于拼接全部seed预测后重算指标。负病例数不取非整数均值，详见各seed行。'],'source_paths':sum([r['source_paths'] for r in members],[]),'comparisons':[],'member_source_rows':[r['source_row'] for r in members]}
    for ob in supp:
        if 359<=ob['source_row']<=370 and norm(ob['values'].get('A'))==norm(stem):rec['comparisons'].append(ob)
    records.append(rec)

# Explicit three-seed root-cause / regularization summaries were unique to the comparison sheet.
for dest,srcs in {170:[120,121,122],171:[123,124,125],172:[126,127,128],189:[131,132,133],190:[134,135,136],191:[137,138,139],192:[140,141,142]}.items():
    tv=S['汇总对比'][dest];members=[next(r for r in records if r['source_sheet']=='实验矩阵总览' and r['source_row']==n) for n in srcs]
    vals={k:members[0]['values'][k] for k in ['B','C','D','E','F']};vals['A']=tv['A']+'｜3seed均值';vals['B']='138/0/36；s1234/s7/s2025 指标算术均值';vals['AK']='均值：'+','.join(str(r['values'].get('AK','')) for r in members)
    for k in MAP:
        ar=[r['values'].get(k) for r in members]
        if all(isinstance(z,(int,float)) for z in ar):vals[k]=statistics.mean(ar)
    ex={'seed数':3}
    for k in EXTRA:
        ar=[r['extra_metrics'].get(k) for r in members]
        if all(isinstance(z,(int,float)) for z in ar):ex[k]=statistics.mean(ar)
    h=({'A':'处理臂','C':'同seed相对S3父模板 ΔR²_cb均值','D':'seed胜/负','E':'各seed ΔR²_cb'} if dest<180 else S['汇总对比'][188])
    ob={'source_sheet':'汇总对比','source_row':dest,'headers':h,'values':tv,'named_values':{f'{h.get(k,k)} [{k}]':z for k,z in tv.items()}}
    if dest<180:supp.append(ob)
    records.append({'source_sheet':'汇总对比','source_row':dest,'group':members[0]['group']+'｜三seed汇总','record_type':'seed_mean','checkpoint':'各seed best均值','values':vals,'extra_metrics':ex,'notes':['各seed指标算术均值；Δ采用每个seed与其配对S3父模型比较后取均值。病例R²胜负与seed胜负口径不同。'],'source_paths':sum([r['source_paths'] for r in members],[]),'comparisons':[ob],'member_source_rows':srcs})

# Attach reference-only summaries, including the R4 row used by V6.
ref=next(r for r in records if r['source_sheet']=='教师汇报视图' and r['source_row']==202)
ref['comparisons'].extend([ob for ob in supp if ob['source_row']==374])

# Block-wide source remarks are data provenance, not execution instructions.
for lo,hi,rows in [(255,265,[371,372]),(267,281,[404]),(283,288,[414]),(290,290,[418]),(293,313,[446])]:
    text='\n'.join(str(z) for nr in rows for z in S['汇总对比'][nr].values())
    for r in records:
        if r['source_sheet']=='实验矩阵总览' and lo<=r['source_row']<=hi:r['notes'].append(text)

# Preserve time-model complements in their own named fields, without mixing them with peak-frame columns.
for row,extras in [(219,{'81帧 pooled 物理R²_cb':0.5021,'81帧 pooled R²':0.4467,'81帧 pooled MAE Pa':0.789,'81帧 pooled 归一化R²_cb':0.7365,'TAWSS R²_cb':0.4427,'TAWSS pooled R²':0.4080,'TAWSS 病例均值R²':0.4367,'TAWSS 病例P10':0.2383,'TAWSS MAE Pa':0.670,'TAWSS high-WSS R²':0.020,'TAWSS top10幅值比':0.460,'TAWSS top10 IoU':0.299}),(220,{'81帧 pooled 物理R²_cb':0.4995,'81帧 pooled R²':0.4246,'81帧 pooled MAE Pa':0.776,'81帧 pooled 归一化R²_cb':0.7503,'TAWSS R²_cb':0.4099,'TAWSS pooled R²':0.3437,'TAWSS 病例均值R²':0.4941,'TAWSS 病例P10':0.2756,'TAWSS MAE Pa':0.657,'TAWSS high-WSS R²':-0.109,'TAWSS top10幅值比':0.439,'TAWSS top10 IoU':0.343})]:
    rec=next(r for r in records if r['source_sheet']=='实验矩阵总览' and r['source_row']==row);rec['extra_metrics'].update(extras);rec['source_paths'].append(str(ROOT/'training_wss_min/experiments/v5_rerun_20260906/time_runs_tables.md'));rec['notes'].append('时间补充值来自time_runs_tables.md，保留原文显示精度；81帧pool受相位变化放大，不与峰值单帧直接比较。TAWSS为81帧|WSS|算术均值。')

# M2 summary contains last-only information; recover the complete matching metric vectors.
for source in list(records):
    if source['source_sheet']!='实验矩阵总览' or not 293<=source['source_row']<=313:continue
    ak=source['values']['AK'];p=ROOT/'training_wss_min/runs'/ak/'eval/ckpt_last/metrics.json'
    if not p.exists():missing.append({'experiment':source['values']['A'],'missing':'last metrics.json','path':str(p)});continue
    data=json.load(open(p));d=data['test'];vals,ex=extract(d)
    rec=copy.deepcopy(source);rec['source_sheet']='历史评估产物';rec['source_row']=None;rec['checkpoint']='last';rec['record_type']='run';rec['source_paths']=[str(p)];rec['values']={k:z for k,z in source['values'].items() if k in ['A','B','C','D','E','F','AK']};rec['values'].update(vals);rec['values']['A']+='｜last';rec['extra_metrics']=ex;rec['comparisons']=[];rec['notes']=['固定epoch399（第400轮），用于与best交叉检查；不得best/last择优。'];rec['matched_metric_path']=str(p);records.append(rec)

# Add useful metric definitions and preserve block-wide remarks without interpreting them as instructions.
remarks=[]
for s,rr in [('汇总对比',[2,29,68,108,109,119,123,131,132,140,141,144,145,154,155,173,193,198,201,223,227,230,240,244,249,254,258,266,356,357,371,372,404,414,418,446]),('教师汇报视图',[29,278]),('实验矩阵总览',[318])]:
    for n in rr:
        if n in S[s]:remarks.append({'source_sheet':s,'source_row':n,'values':S[s][n]})

out={'schema_version':1,'metric_headers':headers,'metric_paths':MAP,'extra_metric_paths':EXTRA,'records':records,'supplemental_rows':supp,'remarks':remarks,'corrections':corrections,'unresolved_missing':missing,'audit_notes':['总览49列全部保留；222:225体场行交由体场审计迁移。','RCR仅删页面；WSS任务的RCR-O0/O1a/O2实验仍保留。','百分比指标O/P/Q/R/AC/AD/AE/AF原始值为比例，应用0.00%显示，不要额外乘100后再用百分比格式。','总览AJ为归一化p99比；物理Pa p99比另列，不互相覆盖。','物理RMSE/MAE与归一化RMSE/MAE均按field（顶点pooled）保持原口径；病例等权误差属于附加指标。','sheet filename last不是checkpoint标签；best必须明确注明按各自训练总损失选择。']}
json.dump(out,open('/tmp/wss_audit.json','w'),ensure_ascii=False,indent=2)
print(json.dumps({'records':len(records),'main_source':sum(r['source_sheet']=='实验矩阵总览' for r in records),'multi_seed_added':sum(r['source_sheet']=='教师汇报视图' for r in records),'last_added':sum(r['checkpoint']=='last' and r['source_sheet']=='历史评估产物' for r in records),'metric_matches':sum(bool(r.get('matched_metric_path')) for r in records),'supplemental_rows':len(supp),'corrections':len(corrections),'unresolved':missing},ensure_ascii=False))
