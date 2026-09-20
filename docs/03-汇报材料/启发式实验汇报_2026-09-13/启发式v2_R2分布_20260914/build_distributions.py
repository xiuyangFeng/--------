#!/usr/bin/env python3
"""Read existing metrics only; generate auditable per-case R2 plots and selections."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
OUT = Path(__file__).resolve().parent
RUNS = ROOT/'training_wss_min/runs'
EXP = ROOT/'training_wss_min/experiments'
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 11})
COLORS = {'worst': '#C84443', 'most_frequent': '#DA920B', 'best': '#268E5D'}
LABELS = {'worst': '最差', 'most_frequent': '最常见区间代表', 'best': '最好'}
DIST = []
SOURCE_AUDIT = []

def read(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCE_AUDIT.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)

def metric_path(wave, run):
    return RUNS/(wave+'_20260912' if wave in ['wss_local_wave1','wss_local_wave1b','wss_local_wave2'] else wave+'_20260913')/run/'eval/ckpt_best/metrics.json'

def modern(key, title, paths, target, group, note, partition='test', expected=34):
    payloads = [read(p)[partition] for p in paths]
    ids = sorted(payloads[0]['per_case'])
    assert len(ids) == expected
    for d in payloads:
        assert sorted(d['per_case']) == ids
        vals = [d['per_case'][k]['overall']['r2'] for k in ids]
        assert np.all(np.isfinite(vals))
        assert abs(np.mean(vals)-d['aggregate']['r2_casemean']) < 1e-8
    cases = []
    for c in ids:
        rows = [d['per_case'][c]['overall'] for d in payloads]
        vals = [float(row['r2']) for row in rows]
        cases.append({'case_id': c, 'r2': float(np.mean(vals)),
                      'r2_sd': float(np.std(vals, ddof=1)) if len(vals)>1 else 0.,
                      'seed_values': vals,
                      'mae': float(np.mean([row['mae'] for row in rows])),
                      'nmae': float(np.mean([row['nmae_range'] for row in rows])),
                      'nmae_definition': 'MAE / within-case truth range',
                      'point_count': int(rows[0]['n'])})
    cb = [d['field_casebalanced']['r2'] for d in payloads]
    mode = '每例三个 seed 的预测R²均值；横须=seed间样本SD（非置信区间）' if len(paths)==3 else '每个点=一例的预测R²；单seed/固定上游checkpoint'
    rec = dict(key=key,title=title,subtitle=f'{target} · test{expected} · 峰值1162 · best · '+mode,
               group=group,note=note,source_paths=[str(p) for p in paths],source_field=f'{partition}.per_case[case_id].overall.r2',
               official_field_cb=float(np.mean(cb)),official_field_cb_sd=float(np.std(cb,ddof=1)) if len(cb)>1 else 0.,
               cases=cases,replicate_count=len(paths),target=target)
    DIST.append(rec)

def historical():
    path=ROOT/'outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak/V4-SP-PN-BC-PDE-EMA-s1234/evaluation_official_last_converged_full.json'
    data=read(path)
    for target,label in [('speed','速度幅值 |u|'),('pressure','相对压力')]:
        cases=[]
        for c in data['cases']:
            row=c['metrics'][target]
            cases.append(dict(case_id=c['case_id'],r2=float(row['r2']),r2_sd=0.,seed_values=[float(row['r2'])],
                              mae=float(row['mae']),nmae=float(row['nmae']),nmae_definition=row.get('nmae_denominator','see original'),point_count=row['count']))
        assert len(cases)==35
        DIST.append(dict(key='C_V4_'+target,title='方向3｜历史 V4 PINN · '+label,
           subtitle='V4-SP-PN-BC-PDE-EMA · test35 · steady peak · s1234 · epoch9999 · converged:false',group='C 历史物理约束',
           note='旧数据/旧输入/旧协议；此页用于诊断分布，不与V5作同条件排名。最好仅指本组相对最高。',
           source_paths=[str(path)],source_field=f'cases[].metrics.{target}.r2',cases=cases,replicate_count=1,target=label))
    path=ROOT/'outputs/wss_pinn/audits/v4_fullwall_20260901/arms/V4-SP-PN-BC-PDE-EMA-s1234/wss_metrics.json'
    data=read(path)
    cases=[]
    for c in data['cases']:
        cases.append(dict(case_id=c['case_id'],r2=float(c['wss_r2']),r2_sd=0.,seed_values=[float(c['wss_r2'])],
                          mae=float(c['wss_mae_pa']),nmae=float(c['wss_nmae_range']),nmae_definition='MAE / within-case truth range',point_count=c['points']))
    assert len(cases)==35
    assert sum(c['point_count'] for c in cases)==1328017
    DIST.append(dict(key='C_V4_wss',title='方向3｜历史 V4 PINN · 速度派生 WSS',
      subtitle='V4-SP-PN-BC-PDE-EMA · test35 · peak full-wall · s1234 · epoch9999 · converged:false',group='C 历史物理约束',
      note='冻结Profile-Secant V3；已核验35例共1,328,017壁面点。旧JSON“1200”为陈旧文字；非V5 PINN结果。',
      source_paths=[str(path)],source_field='cases[].wss_r2',cases=cases,replicate_count=1,target='WSS · Pa'))

def representatives(cases,width=0.1):
    ordered=sorted(cases,key=lambda c:(c['r2'],c['case_id']))
    x=np.array([c['r2'] for c in ordered])
    # Zero-anchored bins, common width across all figures, no removal of negatives.
    low=math.floor(float(x.min())/width)
    high=math.floor(float(x.max())/width)+1
    if high<=low: high=low+1
    edges=np.round(np.arange(low,high+1)*width,12)
    counts,_=np.histogram(x,bins=edges)
    tied=np.flatnonzero(counts==counts.max())
    median=float(np.median(x))
    selected=min(tied,key=lambda i:(abs((edges[i]+edges[i+1])/2-median),int(i)))
    membership=np.clip(np.searchsorted(edges,x,side='right')-1,0,len(counts)-1)
    eligible=[c for i,c in enumerate(ordered) if membership[i]==selected]
    centre=float((edges[selected]+edges[selected+1])/2)
    modal=min(eligible,key=lambda c:(abs(c['r2']-centre),c['case_id']))
    reps={'worst':ordered[0],'most_frequent':modal,'best':ordered[-1]}
    bin_info=dict(width=width,edges=edges.tolist(),counts=counts.tolist(),modal_indices=tied.tolist(),
                  selected_index=int(selected),selected_range=edges[selected:selected+2].tolist(),
                  selected_count=int(counts[selected]),modal_case_id=modal['case_id'])
    assert sum(counts)==len(cases)
    assert modal in eligible
    return ordered,reps,bin_info

def plot(rec):
    ordered,reps,bins=representatives(rec['cases'])
    x=np.array([c['r2'] for c in ordered]); sd=np.array([c['r2_sd'] for c in ordered]); n=len(x)
    stats=dict(n_cases=n,case_mean=float(x.mean()),case_median=float(np.median(x)),case_p10=float(np.percentile(x,10)),
               negative_count=int((x<0).sum()),minimum=float(x.min()),maximum=float(x.max()))
    rec.update(stats=stats,representatives={k:{'case_id':v['case_id'],'r2':v['r2'],'r2_sd':v['r2_sd']} for k,v in reps.items()},histogram=bins)
    rec['modal_bin_sensitivity']={str(w):representatives(rec['cases'],w)[2] for w in [.05,.1,.2]}
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.055,.945,rec['title'],fontsize=22,fontweight='bold',color='#16334A')
    fig.text(.055,.899,rec['subtitle'],fontsize=10.8,color='#52677A')
    fig.text(.055,.855,f"病例R²均值 {stats['case_mean']:.3f}   中位 {stats['case_median']:.3f}   P10 {stats['case_p10']:.3f}   负R² {stats['negative_count']}/{n}",fontsize=12.3,color='#16334A')
    ax=fig.add_axes([.075,.325,.50,.475]); hist=fig.add_axes([.67,.325,.285,.475])
    ranks=np.arange(1,n+1)
    ax.plot(x,ranks,'-',color='#79A7C6',lw=1.5,zorder=1)
    if rec['replicate_count']>1:
        ax.errorbar(x,ranks,xerr=sd,fmt='none',ecolor='#B8C4CD',elinewidth=1.,capsize=2,zorder=1)
    ax.scatter(x,ranks,s=26,color='#377FAA',zorder=3)
    for key,c in reps.items():
        rank=next(i+1 for i,v in enumerate(ordered) if v['case_id']==c['case_id'])
        ax.axvline(c['r2'],color=COLORS[key],alpha=.8,lw=1.3,zorder=0)
        ax.scatter([c['r2']],[rank],s=75,color=COLORS[key],edgecolors='white',zorder=5)
    edges=np.array(bins['edges']); counts=np.array(bins['counts']); centres=(edges[:-1]+edges[1:])/2
    barcols=['#F7DFAB' if i in bins['modal_indices'] else '#B6D7DF' for i in range(len(counts))]
    barcols[bins['selected_index']]=COLORS['most_frequent']
    hist.bar(centres,counts,width=.096,color=barcols,edgecolor='white',linewidth=1.)
    for cx,cnt in zip(centres,counts):
        if cnt: hist.text(cx,cnt+.2,str(cnt),ha='center',fontsize=10)
    for key,c in reps.items(): hist.axvline(c['r2'],color=COLORS[key],lw=1.1,alpha=.8)
    # Both panels share limits, including the seed-SD whiskers and every negative case.
    lo=min(float((x-sd).min()),float(edges[0])); hi=max(float((x+sd).max()),float(edges[-1])); span=max(hi-lo,.2)
    lim=(lo-.04*span,hi+.04*span)
    for a in [ax,hist]:
        a.set_xlim(lim);a.set_xlabel('逐病例预测 R²（越右越好）',fontsize=11)
        if lim[0]<0<lim[1]: a.axvline(0,ls=':',color='#555555',lw=1.)
        a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
        a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
        a.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylim(.3,n+.7);ax.set_ylabel('病例排序：最差 → 最好');ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
    hist.set_ylim(0,max(counts)*1.19+1);hist.set_ylabel('病例数');hist.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
    ax.set_title('横向 R² 散点：每个点是一例',fontsize=12,pad=10)
    hist.set_title('直方图：同宽分箱 ΔR²=0.10',fontsize=12,pad=10)
    for i,key in enumerate(['worst','most_frequent','best']):
        c=reps[key]
        fig.text(.075,.231-i*.041,f"● {LABELS[key]}：{c['case_id']}   R²={c['r2']:.4f}",fontsize=11.7,color=COLORS[key])
    lo,hi=bins['selected_range']
    extra=f"最高频区间 [{lo:.2f}, {hi:.2f})：{bins['selected_count']}/{n}例。"
    if len(bins['modal_indices'])>1: extra+=f" {len(bins['modal_indices'])}箱并列；按距中位数最近的箱择一。"
    fig.text(.075,.079,extra+'橙点取箱内距中心最近病例；Most ≠ Median。',fontsize=10,color='#52677A')
    fig.text(.075,.038,rec['note'],fontsize=9.6,color='#52677A')
    rec['figure_png']=str(OUT/(rec['key']+'_distribution.png'))
    fig.savefig(OUT/(rec['key']+'_distribution.png'),dpi=180)
    plt.close(fig)
    # Keep full-precision fields and identifiers for every shown point.
    with (OUT/(rec['key']+'_cases.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        fields=['rank','case_id','r2','r2_sd','seed_values','mae','nmae','nmae_definition','point_count','representative']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,c in enumerate(ordered,1):
            row=dict(c,rank=i,representative=';'.join(k for k,v in reps.items() if v['case_id']==c['case_id']))
            row['seed_values']=json.dumps(row['seed_values']); w.writerow(row)

def main():
    for model,group,title,target in [('X5','A 当前直接WSS','方向1｜X5 · 直接 WSS','WSS · Pa'),
                                     ('X5X11','A 当前直接WSS','方向1｜X5X11 · 截面上下文候选','WSS · Pa'),
                                     ('VF6','B 当前数据驱动体场','方向2｜VF6 · 速度幅值','速度幅值 |u| · m/s'),
                                     ('PF6','B 当前数据驱动体场','方向2｜PF6 · 相对压力','相对压力 · Pa')]:
        waves=['wss_local_wave1','wss_local_wave1b','wss_local_wave1b'] if model=='X5' else ['wss_local_wave2']*3 if model=='X5X11' else ['wss_local_wave2','wss_local_wave3','wss_local_wave3']
        paths=[metric_path(w,f'{model}_s{seed}') for w,seed in zip(waves,[1234,7,2025])]
        modern('A_'+model if model.startswith('X') else 'B_'+model,title,paths,target,group,
               '固定v2的三个seed：1234/7/2025；图中是病例分数均值，既非集成预测R²，也非总体R²_cb。')
    r5root=RUNS/'v5_rerun_20260906/outputs'
    modern('B_R5V_speed','方向2诊断配套｜旧 R5V 上游速度',[r5root/'r5v_velocity_qad_s1234/eval/ckpt_best/metrics.json'],
           '速度幅值 |u| · m/s','B 历史R5链路','与R5V→WSS使用同一上游模型；不是最新VF6。')
    modern('B_R5P_pressure','方向2历史参考｜旧 R5P 相对压力',[r5root/'r5p_pressure_mixed_qad_s1234/eval/ckpt_best/metrics.json'],
           '相对压力 · Pa','B 历史R5链路','压力是独立目标；本页不等同VF6速度派生WSS的输入。')
    for part,key,title in [('test','B_R5V_wss','方向2｜旧 R5V 预测速度 → WSS'),
                           ('oracle','B_CFD_oracle','方向2诊断｜CFD 真值速度 → 同一 WSS 算子'),
                           ('physics_only','B_R5V_physics','方向2诊断｜旧 R5V → 未校准物理核')]:
        modern(key,title,[EXP/'v5_velocity_to_wss_20260909/metrics.json'],'WSS · Pa','B 历史R5链路',
               '冻结Profile-Secant V3及其旧半径映射；oracle是诊断参考，不是部署模型成绩或严格上界。',partition=part)
    historical()
    for rec in DIST: plot(rec)
    manifest=dict(date='2026-09-14',metric='physical per-case prediction R2 (not fitted-line R2)',
                  replicate_policy='one patient per dot; V5 per-patient arithmetic mean of seed R2, sample SD whiskers; not ensemble predictions',
                  modal_rule='zero-anchored fixed-width 0.10 bins; max-count ties: nearest to sample median, then lower index; pick in-bin case nearest centre, then canonical ID',
                  source_audit=SOURCE_AUDIT,distributions=DIST)
    (OUT/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    for d in DIST:
        for role,c in d['representatives'].items(): rows.append(dict(series=d['key'],target=d['target'],role=role,**c))
    with (OUT/'representative_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 病例R²分布与选例清单（2026-09-14）','',
           '每组提供横向排名散点 + 同宽直方图；最好/最常见/最差从该组原始预测R²选择。负R²全部保留。',
           'V5固定v2所用seed1234/7/2025：一例一个点，横须为三seed病例R²的样本标准差；不是102例或集成预测。',
           '最常见：0为锚点、箱宽0.10；并列最高频箱取距中位数最近者，再取较低箱。代表取该箱内距中心最近病例；ID打破并列。',
           '此定义不等于中位病例；分箱敏感性（0.05/0.10/0.20）保存在summary.json，不能称唯一的典型病例。','',
           '| 组 | 例数 | 病例R²均值 / 中位 / P10 | 负R² | 最差 | 最常见区间代表 | 最好 |',
           '|---|---:|---|---:|---|---|---|']
    for d in DIST:
        st=d['stats'];r=d['representatives']
        lines.append(f"| {d['key']} | {st['n_cases']} | {st['case_mean']:.4f} / {st['case_median']:.4f} / {st['case_p10']:.4f} | {st['negative_count']} | "+' | '.join(f"{r[k]['case_id']} ({r[k]['r2']:.4f})" for k in ['worst','most_frequent','best'])+' |')
    lines+=['','## 使用方式','',
            '当前X5/X5X11/VF6/PF6、历史R5诊断链、历史V4 PINN均分组标识。test34与test35、三seed与单seed不作同条件冠军排名。',
            'wave4已完成，X5保留为底座、五seed部署集成R²_cb=0.7354；本批仍固定v2三个seed，集成总体成绩另列。',
            '每页CSV包括原始病例ID、各seed R²、病例均值/SD、可用MAE/NMAE和定义；跨版本NMAE分母不同，不直接合并。',
            '代表病例云图应与所展示run/checkpoint对应。三seed均值选例时，若只画s1234云图，必须标s1234病例R²；均值不冒充单seed或集成场的分数。',
            '已有跨目标共同病例保留用于横向比较；本批每目标自己的最好/最常见/最差用于诊断，二者是不同选例任务。',
            'V4源文件名为last_converged，实际记录epoch9999、converged:false；不能称“已收敛模型”。V4 test35比V5 test34多AAA/ruputer/YANG_BAO_KUI。',
            '原V4横向示例按每组min/max划5箱，本批按0锚定、宽0.10分箱，因此橙色Most代表可能变化，Best/Worst仍取真实预测R²极值。',
            '历史目录另有拟合直线R²图；本批只读原始预测R²，未用相关系数平方或拟合线R²替代。',
            '病例预测R²=1−Σ(Pred−CFD)²/Σ(CFD−病例CFD均值)²；横轴越右越好。负值表示在该指标上比预测该病例真值均值更差。',
            '每组各有*_distribution.png及*_cases.csv；representative_cases.csv是36条模型/目标/角色选例记录，病例可重复。summary.json保存统计、分箱、源路径与SHA256。',
            '重绘：python3 build_distributions.py。仅读取现有JSON，不调用训练或推理；图件PNG/PDF/SVG可独立使用。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{'key':d['key'],'n':d['stats']['n_cases'],'reps':d['representatives']} for d in DIST],ensure_ascii=False,indent=2))

if __name__=='__main__': main()
