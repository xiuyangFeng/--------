# Static dependency-expanded audit input: actual plot_style.py followed by build_quantitative.py.
# This copy is for static audit, not an independent renderer; use build_quantitative.py.
"""Shared Matplotlib drawing/export contract for the scientific report figures."""
from pathlib import Path
import hashlib, json, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p/'training_wss_min').is_dir())
SKILL = Path('/public/newhome/cy/.codex/skills/nature-figure')
sys.path.insert(0, str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

INK='#203140'; MUTED='#62717B'; BLUE='#477DA5'; TEAL='#338F88'
PURPLE='#8875AB'; ORANGE='#BA8650'; RED='#B45F5B'; GREY='#A6B1B7'
LIGHT='#EDF1F3'; LINE='#CBD4D9'
for f in ('NotoSansCJK-Regular.ttc','NotoSansCJK-Bold.ttc'):
    p=Path('/usr/share/fonts/opentype/noto')/f
    if p.exists():font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
    'font.size':14,'axes.titlesize':17,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,
    'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
    'figure.facecolor':'white','savefig.facecolor':'white'})

def page(title, subtitle='', kicker='', footer=''):
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.045,.95,kicker,fontsize=11,color=TEAL,weight='bold',va='top')
    fig.text(.045,.905,title,fontsize=27,weight='bold',va='top')
    if subtitle:fig.text(.045,.837,subtitle,fontsize=14,color=MUTED,va='top')
    if footer:fig.text(.045,.045,footer,fontsize=10,color=MUTED,va='bottom')
    return fig

def canvas(fig,rect=(.04,.12,.92,.66)):
    ax=fig.add_axes(rect);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off');return ax

def text(ax,x,y,s,size=15,color=INK,bold=False,ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight='bold' if bold else 'normal',ha=ha,va=va,linespacing=1.45,**kw)

def node(ax,x,y,w,h,title,body='',color=BLUE,face=None,fontsize=15,body_size=12):
    if face is None:face=matplotlib.colors.to_rgba(color,.08)
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.8',lw=.9,ec=color,fc=face))
    if body:
        text(ax,x+w/2,y+h*.70,title,fontsize,color,True,'center')
        text(ax,x+w/2,y+h*.30,body,body_size,INK,False,'center')
    else:text(ax,x+w/2,y+h/2,title,fontsize,color,True,'center')

def arrow(ax,start,end,color=GREY,style='-',rad=0):
    a=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=color,
        linestyle=style,connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a);return a

def route(ax,points,color=GREY,dashed=False):
    for i,(a,b) in enumerate(zip(points,points[1:])):
        if i==len(points)-2:arrow(ax,a,b,color,'--' if dashed else '-')
        else:ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.2,ls='--' if dashed else '-',solid_capstyle='butt')

def panel_label(ax,label):
    ax.annotate(label,(0,1),xycoords='axes fraction',xytext=(-15,10),textcoords='offset points',fontsize=16,weight='bold',va='bottom')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(fig,stem,*,claim='',sources=(),axes=None,row_groups=None,column_groups=None,exemptions=(),note=''):
    """Save one fixed-size figure and fresh skill audits; FAILs remain explicit."""
    dest=OUT/'figures'/stem;qa=OUT/'qa'/stem
    dest.parent.mkdir(exist_ok=True);qa.parent.mkdir(exist_ok=True)
    opts={'axes':axes,'exemptions':exemptions}
    if row_groups is not None:opts['row_groups']=row_groups
    if column_groups is not None:opts['column_groups']=column_groups
    alignment=require_matplotlib_panel_alignment(fig,json_out=str(qa)+'.alignment.json',
        overlay_svg=str(qa)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,**opts)
    for ext in ['pdf','svg','png']:fig.savefig(str(dest)+'.'+ext,dpi=300)
    plt.close(fig)
    pdf=Path(str(dest)+'.pdf')
    audits={}
    commands={
      'text':[sys.executable,str(SKILL/'scripts/audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],
      'collision':[sys.executable,str(SKILL/'scripts/audit_figure_collisions.py'),str(pdf),
        '--json-out',str(qa)+'.collision.json','--overlay-pdf',str(qa)+'.collision.pdf']}
    for key,cmd in commands.items():
        p=subprocess.run(cmd,text=True,capture_output=True)
        (Path(str(qa)+'.'+key+'.log')).write_text(p.stdout+p.stderr)
        audits[key]={'exit_code':p.returncode,'report':str(qa.relative_to(OUT))+'.'+key+('.json' if key=='collision' else '.log')}
    meta={'stem':stem,'claim':claim,'sources':[{'path':str(Path(p).resolve()),'sha256':sha(p)} for p in sources],
        'exports':{ext:str(dest.relative_to(OUT))+'.'+ext for ext in ['pdf','svg','png']},
        'alignment':alignment.get('verdict',alignment.get('status')),'audits':audits,'note':note}
    Path(str(qa)+'.metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(stem,{k:v['exit_code'] for k,v in audits.items()},flush=True)
    return meta

"""Evidence-first quantitative report figures; cached observations only.

Contract: source_data/quantitative_contract.json. Python is the selected backend.
Replicates are matched training seeds; SD uses ddof=1, never a confidence interval.
Every cached case is retained. No model training, inference, fitting or simulated data.
"""
from pathlib import Path
import csv, json, math
import numpy as np
from plot_style import *

SOURCE = OUT/'source_data'
SUMMARY = OUT.parent/'启发式v2_R2分布_20260914/summary.json'
summary = json.loads(SUMMARY.read_text())
dists = {x['key']:x for x in summary['distributions']}
wave_paths = {w:next((ROOT/'training_wss_min/experiments').glob(f'wss_local_wave{w}_*/results.json')) for w in ['1','1b','2','3']}
waves={w:json.loads(p.read_text())['arms'] for w,p in wave_paths.items()}
SEEDS=[1234,7,2025]

def export_csv(stem,rows):
    p=SOURCE/(stem+'.csv'); p.parent.mkdir(exist_ok=True)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return p

def arm(w,key):
    entry=waves[w][key]
    assert entry['status']=='complete'
    return entry['checkpoints']['best']

def x5(seed):return arm('1','X5') if seed==1234 else arm('1b',f'X5_s{seed}')
def x0(seed):return arm('1','X0') if seed==1234 else arm('1b',f'X0_s{seed}')
def fmt(v):return f'{v:+.4f}'

def delta_panel(ax,values,title,color=TEAL,unit='ΔR²_cb',limits=None):
    values=np.asarray(values,dtype=float);assert values.shape==(3,) and np.isfinite(values).all()
    mean=values.mean();sd=values.std(ddof=1)
    ax.axhline(0,color=GREY,lw=1,ls='--',zorder=0)
    ax.scatter(range(3),values,s=64,c=color,zorder=4)
    ax.errorbar([3.3],[mean],yerr=[sd],fmt='D',ms=7,color=INK,elinewidth=1.8,capsize=6,zorder=5)
    ax.set_xticks([0,1,2,3.3],['1234','7','2025','均值 ± SD'])
    ax.set_xlim(-.55,4.05);ax.set_ylabel(unit)
    ax.set_title(title,loc='left',fontsize=18,pad=18)
    ax.tick_params(axis='both',pad=8);ax.set_axisbelow(True)
    if limits:ax.set_ylim(*limits)
    else:
        lo=min(0,*(values),mean-sd);hi=max(0,*(values),mean+sd)
        span=hi-lo;ax.set_ylim(lo-span*.20,hi+span*.38)
    # Statistical summary is outside the data area, so error bars cannot cross text.
    ax.text(.5,.99,f'配对均值 {fmt(mean)}',transform=ax.transAxes,ha='center',va='top',fontsize=15,color=color)
    return mean,sd

def fig07():
    stem='07_murray_paired_effect';rows=[];series=[]
    specs=[('直接 WSS','X0 → X5',TEAL),('相对压力','P02 → PF6',PURPLE),('速度大小','V07 → VF6',BLUE)]
    for i,(target,comparison,color) in enumerate(specs):
        vals=[]
        for seed in SEEDS:
            if i==0:
                base,new=x0(seed),x5(seed);w='1' if seed==1234 else '1b';bk='X0' if seed==1234 else f'X0_s{seed}';nk='X5' if seed==1234 else f'X5_s{seed}'
            else:
                prefix='P' if i==1 else 'V';b='P02' if i==1 else 'V07';w='2' if seed==1234 else '3';bk=f'{b}r_s{seed}' if seed==1234 else f'{b}_s{seed}';nk=f'{prefix}F6_s{seed}';base,new=arm(w,bk),arm(w,nk)
            delta=new['pa_r2_cb']-base['pa_r2_cb'];vals.append(delta)
            rows.append(dict(target=target,comparison=comparison,seed=seed,baseline_r2_cb=base['pa_r2_cb'],new_r2_cb=new['pa_r2_cb'],paired_delta_r2_cb=delta,source_file=str(wave_paths[w].relative_to(ROOT)),baseline_json_pointer=f'arms.{bk}.checkpoints.best.pa_r2_cb',new_json_pointer=f'arms.{nk}.checkpoints.best.pa_r2_cb'))
        assert all(v>0 for v in vals)
        series.append(vals)
    p=export_csv(stem,rows)
    fig=page('Murray 几何先验：三个目标、三个配对 seed 均同向受益',
        '只比较同 seed 的同期底座与 +F6；观察跨目标复现，尚不把同向结果写成机制证明。','07  |  模块证据',
        'test34 · peak 1162 · ckpt_best（原选模规则）  |  点 = 训练 seed；菱形 = 配对均值 ± 样本 SD（n=3，非 CI）；未做显著性检验。')
    axs=fig.subplots(1,3,gridspec_kw={'left':.075,'right':.96,'bottom':.20,'top':.70,'wspace':.35})
    for i,(ax,vals,(target,comparison,color)) in enumerate(zip(axs,series,specs)):
        ax.set_gid(chr(97+i));panel_label(ax,chr(97+i));delta_panel(ax,vals,f'{target}  |  {comparison}',color,limits=(-.012,.137))
    fig.text(.5,.105,'WSS：+0.069    压力：+0.045    速度：+0.027   →   支持“流量分配先验提供了可迁移信息”这一工作假设。',ha='center',fontsize=16,color=INK)
    return save(fig,stem,claim='F6对WSS/压力/速度三目标的9个配对seed比较均为正；不声明显著性或机制已获证明。',sources=[*wave_paths.values(),p],axes=list(axs),note='Three same-size quantitative panels; physical R2_cb, not arithmetic per-case R2. Pressure/velocity seed1234 controls are contemporaneous P02r/V07r. Seeds 7/2025 use paired P02/V07 controls. Full n=3 points and sample SD retained.')

def fig08():
    stem='08_x11_tradeoff';rows=[]
    metric_specs=[('pa_r2_cb','整体拟合','ΔR²_cb ↑',TEAL),('high_wss_r2','高 WSS 区拟合','Δ高 WSS R² ↑',TEAL),('p99_ratio','p99 幅值（比值趋近 1）','Δp99 幅值比',BLUE),('top10_iou','热点定位','Δtop10 IoU ↑',RED)]
    allmetrics=[x[0] for x in metric_specs]+['pa_mae'];data={m:[] for m in allmetrics}
    for seed in SEEDS:
        a,b=x5(seed),arm('2',f'X5X11_s{seed}')
        for metric in allmetrics:
            delta=b[metric]-a[metric];data[metric].append(delta)
            rows.append(dict(seed=seed,metric=metric,X5=a[metric],X5X11=b[metric],paired_delta=delta,baseline_source=str(wave_paths['1' if seed==1234 else '1b'].relative_to(ROOT)),new_source=str(wave_paths['2'].relative_to(ROOT)),checkpoint='best',baseline_key='X5' if seed==1234 else f'X5_s{seed}',new_key=f'X5X11_s{seed}'))
    p=export_csv(stem,rows)
    fig=page('X11 改善高值幅度，但热点定位与 MAE 三个 seed 均回退',
        '把 X5X11 与同 seed 的 X5 配对：增量集中在哪里？哪些代价阻止我们把它写成全面提升？','08  |  收益与代价',
        'test34 · peak 1162 · best  |  Δ = X5X11 − X5；点 = seed，菱形 = 均值 ± 样本 SD（n=3）；各面板使用独立纵轴。')
    axs=fig.subplots(2,2,gridspec_kw={'left':.095,'right':.96,'bottom':.19,'top':.71,'wspace':.28,'hspace':.73})
    for i,(ax,(metric,title,label,color)) in enumerate(zip(axs.flat,metric_specs)):
        ax.set_gid(chr(97+i));panel_label(ax,chr(97+i));delta_panel(ax,data[metric],title,color,label);ax.tick_params(labelsize=11)
    mae=np.mean(data['pa_mae']);fig.text(.5,.105,f'MAE：{mae:+.4f} Pa（越低越好，3/3 seed 增大）  |  p99 比仍小于 1：幅值变近，热点位置却未同步改善。',ha='center',fontsize=15,color=RED)
    return save(fig,stem,claim='X11一致改善高WSS拟合和p99幅值，但IoU与MAE均在3个seed回退；应保留为尾部候选。',sources=[wave_paths[w] for w in ['1','1b','2']]+[p],axes=list(axs.flat),note='Primary panels separate aggregate fitting, high-tail fitting, amplitude and localization. P99 is a pooled amplitude ratio with optimum 1, all source ratios below 1; its positive difference improves amplitude here, not a universal higher-is-better rule. MAE is physical pooled Pa. Full five metrics exported without rounded-document reconstruction.')

# Stable IDs from the delivered Best/Median/Worst package. Median is not histogram mode.
SELECTED={
 'A_X5X11':{'B':'AG/fast/ZHANG_LIANG','M':'AG/fast/YAO_CUN_HONG','W':'AAA/unruputer/SUN_SHU_MING'},
 'B_VF6':{'B':'AG/fast/ZHANG_LIANG','M':'AG/slow/LI_HUAN_GE','W':'AAA/unruputer/SUN_SHU_MING'},
 'B_PF6':{'B':'AG/fast/YAO_CUN_HONG','M':'AG/fast/ZHANG_CHUN','W':'AAA/unruputer/SUN_SHU_MING'}}
ROLE_COLORS={'B':TEAL,'M':ORANGE,'W':RED}

def case_source_check(keys):
    """Check summary cases against source records and repeat arithmetic directly."""
    sources=[];checks=[]
    for key in keys:
        d=dists[key];cases=d['cases'];n=34 if key[0]!='C' else 35
        assert len(cases)==n and len({c['case_id'] for c in cases})==n
        for c in cases:
            a=np.asarray(c['seed_values']);assert np.isfinite(a).all()
            assert abs(np.mean(a)-c['r2'])<1e-12
            if len(a)>1:assert abs(np.std(a,ddof=1)-c['r2_sd'])<1e-12
        assert sum(d['histogram']['counts'])==n
        raw_sources=[json.loads(Path(path).read_text()) for path in d['source_paths']]
        for c in cases:
            cid=c['case_id']; values=[]
            for raw in raw_sources:
                if key.startswith('C'):
                    rc=next(v for v in raw['cases'] if v['case_id']==cid)
                    value=rc['wss_r2'] if key=='C_V4_wss' else rc['metrics'][key.removeprefix('C_V4_')]['r2']
                else:
                    block='oracle' if key=='B_CFD_oracle' else 'physics_only' if key=='B_R5V_physics' else 'test'
                    value=raw[block]['per_case'][cid]['overall']['r2']
                values.append(value)
            assert np.allclose(values,c['seed_values'],rtol=0,atol=1e-12),(key,cid,values,c['seed_values'])
        for path in d['source_paths']:
            p=Path(path);assert p.is_file();sources.append(p)
            audit=next((x for x in summary['source_audit'] if x['path']==str(p)),None)
            if audit:assert sha(p)==audit['sha256']
        checks.append({'key':key,'case_count':n,'all_values_finite':True,'seed_mean_sd_recomputed':True,'all_seed_values_verified_against_raw_json':True,'histogram_retains_all':True,'negative_count':sum(c['r2']<0 for c in cases)})
    return list(dict.fromkeys(sources)),checks

def dist_figure(historical=False):
    if historical:
        stem='16_historical_case_distributions';keys=['C_V4_speed','C_V4_pressure','C_V4_wss'];names=['速度 |u|','相对压力','派生 WSS'];colors=[BLUE,PURPLE,TEAL]
        title='历史 V4：速度与派生 WSS 的失败覆盖多数病例，需逐环节诊断'
        sub='V4-SP-PN-BC-PDE-EMA · test35 · s1234 · epoch 9999 · converged: false；所有负 R² 完整保留。'
        foot='旧数据、旧输入、旧选模协议：仅用于定位问题，不与当前 V5 作同条件排名。直方图零锚定、固定箱宽 0.10；没有删点。'
        xlim=(-1.35,1.04);nrows=3;bottom=.17;top=.74;gap=.065
    else:
        stem='09_current_case_distributions';keys=['A_X5','A_X5X11','B_VF6','B_PF6'];names=['X5 · WSS','X5X11 · WSS','VF6 · |u|','PF6 · 压力'];colors=[GREY,TEAL,BLUE,PURPLE]
        title='最佳病例只代表分布的一端：低分病例与 seed 波动仍需解释'
        sub='同一 test34 · peak 1162 · best；每个点 = 一例的 3-seed R² 均值，横须 = 样本 SD。'
        foot='这些是逐病例预测 R²，并非总体 R²_cb，也不是集成预测的 R²。直方图零锚定、箱宽 0.10；保留全部 34 例及完整 SD。'
        xlim=(-.04,1.045);nrows=4;bottom=.17;top=.74;gap=.053
    sources,checks=case_source_check(keys);rows=[];histrows=[]
    fig=page(title,sub,('16  |  历史问题' if historical else '09  |  病例分布'),foot)
    height=(top-bottom-gap*(nrows-1))/nrows;axes=[]
    fig.text(.20,.777,'逐病例分数与离散度',fontsize=15,color=INK)
    fig.text(.785,.777,'病例数分布',fontsize=15,color=INK)
    for r,(key,name,color) in enumerate(zip(keys,names,colors)):
        y=top-height-r*(height+gap)
        a=fig.add_axes([.19,y,.50,height]);b=fig.add_axes([.79,y,.17,height]);axes.extend([a,b]);a.set_gid(chr(97+2*r));b.set_gid(chr(98+2*r))
        # Fixed physical label anchors preserve row/column alignment across unequal panel widths.
        for axis,letter in [(a,chr(97+2*r)),(b,chr(98+2*r))]:
            axis.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-15,0),textcoords='offset points',fontsize=16,weight='bold',va='bottom')
        d=dists[key];cases=sorted(d['cases'],key=lambda c:(c['r2'],c['case_id']))
        vals=np.array([c['r2'] for c in cases]);sd=np.array([c['r2_sd'] for c in cases])
        # Six deterministic display lanes only prevent mark overlap; y has no numerical meaning.
        yy=np.array([((i*5)%7-3)*.14 for i in range(len(cases))]);a.axvline(0,color=LINE,ls='--',lw=.8,zorder=0)
        a.errorbar(vals,yy,xerr=sd,fmt='none',ecolor=color,alpha=.50,elinewidth=.7,zorder=1)
        a.scatter(vals,yy,s=19,color=color,alpha=.90,zorder=3)
        a.set_xlim(*xlim);a.set_ylim(-.68,1.12);a.set_yticks([]);a.spines['left'].set_visible(False)
        a.tick_params(axis='x',labelsize=10,pad=8)
        if r<nrows-1:a.tick_params(labelbottom=False)
        else:a.set_xlabel('逐病例预测 R²',fontsize=12,labelpad=6)
        if historical:a.set_xticks([-1.0,-.5,0,.5,1])
        else:a.set_xticks([0,.2,.4,.6,.8,1])
        reps=SELECTED.get(key,{})
        for role,cid in reps.items():
            idx=next(i for i,c in enumerate(cases) if c['case_id']==cid);v=vals[idx]
            a.scatter([v],[yy[idx]],s=62,color=ROLE_COLORS[role],marker={'B':'^','M':'D','W':'v'}[role],edgecolor='white',lw=.6,zorder=5)
            a.text(v,.65,role,fontsize=13,weight='bold',ha='center',color=ROLE_COLORS[role])
        fig.text(.045,y+height*.68,name,fontsize=16,weight='bold',va='center',color=color if color!=GREY else INK)
        stat=f'n={len(cases)}  |  负值 {sum(vals<0)}\n中位数 {np.median(vals):.3f}'
        fig.text(.045,y+height*.16,stat,fontsize=11,va='center',color=MUTED,linespacing=1.35)
        lo=math.floor(xlim[0]*10)/10;hi=math.ceil(xlim[1]*10)/10;edges=np.arange(round(lo*10),round(hi*10)+1)/10
        counts,_=np.histogram(vals,edges);assert counts.sum()==len(cases)
        b.bar(edges[:-1],counts,width=.10,align='edge',facecolor=matplotlib.colors.to_rgba(color,.58),edgecolor='none',lw=0)
        b.axvline(0,color=LINE,ls='--',lw=.8);b.set_xlim(*xlim);b.set_ylim(0,max(counts)*1.15+.5);b.set_yticks([0,max(counts)])
        b.tick_params(labelsize=10,pad=8)
        if historical:b.set_xticks([-1,0,1])
        else:b.set_xticks([0,.5,1])
        if r<nrows-1:b.tick_params(labelbottom=False)
        else:b.set_xlabel('R²（箱宽 0.10）',fontsize=12,labelpad=6)
        for c in cases:
            role=next((k for k,v in reps.items() if v==c['case_id']),'')
            rows.append({'model':key,'case_id':c['case_id'],'r2_case_mean':c['r2'],'r2_seed_sample_sd':c['r2_sd'],'seed_values':json.dumps(c['seed_values']),'representative_role':role,'display_lane':float(yy[cases.index(c)]),'negative':c['r2']<0,'metric_source_field':d['source_field']})
        for left,count in zip(edges[:-1],counts):histrows.append({'model':key,'bin_left':left,'bin_right':left+.1,'count':int(count)})
    if not historical:fig.text(.5,.092,'B = Best（最高）   M = Median（已有病例包的中位代表）   W = Worst（最低）  |  标记共 9 个已交付病例。',ha='center',fontsize=12,color=MUTED)
    else:fig.text(.5,.076,'34/35 例速度 R² < 0；20/35 例派生 WSS R² < 0  →  优先检查 loss 分量、速度结构与近壁派生步骤。',ha='center',fontsize=14,color=RED)
    p=export_csv(stem,rows);hp=export_csv(stem+'_histogram',histrows)
    (SOURCE/(stem+'_checks.json')).write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    ids=[a.get_gid() for a in axes]
    return save(fig,stem,claim=title,sources=[SUMMARY,*sources,p,hp],axes=axes,
        row_groups=[ids[i:i+2] for i in range(0,len(ids),2)],column_groups=[ids[::2],ids[1::2]],
        exemptions=[{'panels':ids[1::2],'checks':['panel-width'],'reason':'Narrow frequency summaries accompany the wide all-case scatter; source identities and seed uncertainty need more horizontal space.'}],
        note='All cases retained; display lanes have no numerical meaning. Current means/SD independently recomputed from seed_values and source SHA256 checked. Histogram has zero-anchored width0.10. Explicit groups audit actual rows and columns; intentional scatter:histogram width 0.50:0.17. Current B/M/W refer to existing visualization package, not modal-bin cases. Historical converged:false and incompatible protocol disclosed. Editable vector data marks.')

def fig17():
    stem='17_velocity_wss_gap';keys=['B_R5V_physics','B_R5V_wss','B_CFD_oracle'];sources,checks=case_source_check(keys+['B_R5V_speed'])
    common=sorted(c['case_id'] for c in dists[keys[0]]['cases']);series=[];rows=[]
    for key in keys:
        byid={c['case_id']:c for c in dists[key]['cases']};assert set(byid)==set(common)
        series.append(np.array([byid[c]['r2'] for c in common]))
        for cid in common:rows.append({'path':key,'case_id':cid,'per_case_r2':byid[cid]['r2'],'official_r2_cb':dists[key]['official_field_cb'],'source_field':dists[key]['source_field']})
    speed=dists['B_R5V_speed']['official_field_cb'];phys=dists[keys[0]]['official_field_cb'];pred=dists[keys[1]]['official_field_cb'];oracle=dists[keys[2]]['official_field_cb'];p=export_csv(stem,rows)
    fig=page('旧 R5V：速度幅值较准，仍不保证派生 WSS 准确',
        '用 CFD 速度替换模型速度、保持壁面与冻结算子：定位输入敏感性，再分解近壁误差。','17  |  诊断证据链',
        '历史 R5V · test34 · s1234/best；本页没有 VF6 → WSS 闭环结果。CFD 参照含既有监督校准器，不是部署成绩或严格上界。')
    a=fig.add_axes([.065,.23,.50,.48]);a.set_gid('a');a.set(xlim=(0,100),ylim=(0,100));a.axis('off');panel_label(a,'a')
    b=fig.add_axes([.665,.23,.29,.48]);b.set_gid('b');panel_label(b,'b')
    text(a,0,106,'同一管线，替换速度来源',18,bold=True)
    node(a,0,62,26,21,'模型速度',f'|u| R²_cb = {speed:.3f}',BLUE,fontsize=16,body_size=12)
    node(a,36,62,26,21,'冻结 V3','几何 / 梯度 / 校准',MUTED,fontsize=16,body_size=11)
    node(a,72,62,28,21,'派生 WSS',f'R²_cb = {pred:.3f}',RED,fontsize=16,body_size=13)
    arrow(a,(27,72),(35,72));arrow(a,(63,72),(71,72))
    node(a,0,25,26,21,'CFD 速度','诊断替换输入',TEAL,fontsize=16,body_size=12)
    node(a,36,25,26,21,'同一 V3','同壁面与校准器',MUTED,fontsize=16,body_size=11)
    node(a,72,25,28,21,'诊断 WSS',f'R²_cb = {oracle:.3f}',TEAL,fontsize=16,body_size=13)
    arrow(a,(27,35),(35,35));arrow(a,(63,35),(71,35))
    text(a,50,6,'差距不能直接归因于某一个近壁环节。',14,MUTED,ha='center')
    b.set_title('34 例的同病例输出差异',loc='left',fontsize=18,pad=18)
    matrix=np.stack(series,axis=1)
    for row in matrix:b.plot([0,1,2],row,color=GREY,lw=.7,alpha=.46,zorder=1)
    for i,color in enumerate([ORANGE,RED,TEAL]):b.scatter(np.full(len(common),i),matrix[:,i],color=color,s=20,zorder=3)
    b.axhline(0,color=LINE,ls='--',lw=.8);b.set_xlim(-.3,2.3);b.set_ylim(-1.5,1.08);b.set_yticks([-1.5,-1,-.5,0,.5,1]);b.set_ylabel('逐病例预测 R²')
    b.set_xticks([0,1,2],['预测速度\n未校准核','预测速度\n冻结 V3','CFD 速度\n同一 V3']);b.tick_params(axis='x',labelsize=11,pad=8)
    fig.text(.075,.142,'待分解：① 近壁切向速度 / 法向残速    ② 邻域、深度与梯度放大    ③ 校准输入分布变化',fontsize=15,color=INK)
    fig.text(.075,.092,'旧半径映射仍有坐标尺度残差；高 CFD 参照分数不能据此证明几何映射正确。',fontsize=12,color=MUTED)
    return save(fig,stem,claim='在旧R5V冻结派生管线中CFD速度参照明显优于预测速度，支持优先诊断速度输入与误差传播，但不能识别单一故障环节。',sources=[SUMMARY,*sources,p],axes=[a,b],row_groups=[['a','b']],
        exemptions=[{'panels':['a'],'checks':['panel-width'],'reason':'Input-substitution schematic is the hero; per-case quantitative panel is subordinate.'}],note='Left labels are physical case-balanced R2, right shows every one of34 per-case R2 values, with same-patient connector lines, no summary uncertainty for one fixed checkpoint. Frozen V3 uses historical trained calibration and known legacy radius-coordinate limitations. No VF6 closed-loop inference. All34 case values retained including -1.3416.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--only',default='all');args=parser.parse_args()
    funcs={'07':fig07,'08':fig08,'09':lambda:dist_figure(False),'16':lambda:dist_figure(True),'17':fig17}
    for k,fn in funcs.items():
        if args.only in ('all',k):
            meta=fn()
            for ext in ('.pdf','.svg','.png'):
                assert (OUT/'figures'/(meta['stem']+ext)).is_file()
