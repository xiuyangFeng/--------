# Static dependency-expanded audit snapshot; run original build script.
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
    fig.savefig(str(dest)+'.pdf',dpi=300)
    fig.savefig(str(dest)+'.svg',dpi=300)
    fig.savefig(str(dest)+'.png',dpi=300)
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

"""Question-led report schematics and measured historical physics diagnostics.

Claims: distinguish prediction from derivation; freeze comparison conditions;
separate near-wall hypotheses; expose every loss term and optimization conflict.
No invented velocity profiles, model outputs, or newly evaluated results.
"""
import argparse, json, shutil
import numpy as np
import pandas as pd
from plot_style import *

REPORT=OUT.parent
OUTLINE=REPORT/'启发式汇报提纲_v2_2026-09-13.md'
PHYS=ROOT/'docs/03-汇报材料/V4汇报/V4_稳态梯度与损失_2026-09-12'
CURVES=PHYS/'individual_losses/single_pnpp_ema/source_curves.csv'

def routes():
    fig=page('先分清：WSS 是直接预测，还是由速度计算',
      '三条路线围绕同一目标，但直接输出、后处理和证据状态不同。','01 / 研究问题',
      '当前 A/B：V5；C 为历史物理诊断。PF6 是独立压力模型，不是 VF6→WSS 算子的压力输入。')
    ax=canvas(fig)
    for y,tag,name,c in [(76,'A','直接 WSS',TEAL),(44,'B','体场数据驱动',BLUE),(9,'C','加入物理约束',ORANGE)]:
        text(ax,0,y+13,tag,25,c,True);text(ax,0,y+3,name,14,c,True)
    node(ax,16,72,22,23,'壁面几何','support + wall query',TEAL)
    node(ax,45,72,24,23,'X5 / X5X11','直接回归 log_z → Pa',TEAL)
    node(ax,76,72,23,23,'WSS','网络直接预测',TEAL)
    arrow(ax,(38,83),(45,83),TEAL);arrow(ax,(69,83),(76,83),TEAL)
    node(ax,16,39,22,23,'壁面几何 + 体内 query','共享坐标，独立查询',BLUE,fontsize=13)
    node(ax,43,39,21,23,'VF6 → u, v, w','三分量速度模型',BLUE)
    node(ax,70,39,29,23,'冻结算子 → WSS','VF6 闭环结果仍待补',GREY)
    arrow(ax,(38,50),(43,50),BLUE);arrow(ax,(64,50),(70,50),GREY)
    text(ax,44,29,'PF6 另行预测相对压力 p − p_ref',13,BLUE)
    node(ax,16,3,22,23,'联合 u, v, w, p','历史网络与输入协议',ORANGE)
    node(ax,45,3,24,23,'数据 + BC + PDE','训练时约束同一个场',ORANGE)
    node(ax,76,3,23,23,'机制诊断','V5 同底座对照待建',ORANGE)
    arrow(ax,(38,14),(45,14),ORANGE);arrow(ax,(69,14),(76,14),ORANGE)
    save(fig,'01_routes',claim='直接WSS、速度派生WSS与物理训练是三种不同误差链',sources=[OUTLINE])

def protocol():
    fig=page('统一比较条件，也明确历史结果的边界',
       '把可比条件集中在一页；后面的图不再反复混写数据版本、seed 与指标。','02 / 比较合同',
       'test34 参与过开发筛选；本组图保留原三 seed 窗口，不将后续集成成绩贴到单 seed 云图。')
    ax=canvas(fig)
    cols=[(0,'比较维度',MUTED),(23,'当前 A / B',TEAL),(69,'历史 C',ORANGE)]
    for x,s,c in cols:text(ax,x,97,s,18,c,True)
    rows=[('数据与时刻','V5：172 例；train138 / test34\npeak step 1162','历史队列 / 历史帧协议\n本批分布 test35'),
          ('模型输入','5000 个壁面 support\nX5/X5X11：27D；PF6/VF6：20D','历史输入、网络和损失不同\n不能当作同底座消融'),
          ('输出查询域','WSS：壁面；速度：体内\n压力：壁面 + 体内','按历史实际输出域解释'),
          ('指标与选例','主指标 R²_cb；分布为病例 R²\n三 seed 均值选例，s1234 展示场','保留负 R²；单 seed\n只作机制诊断'),
          ('checkpoint','各 run 冻结的 best 规则\n不按 test 重新挑 checkpoint','分布所用 V4 未收敛\n不能标“已收敛最优模型”')]
    for i,(lab,a,b) in enumerate(rows):
        y=79-i*18
        text(ax,0,y,lab,16,INK,True);text(ax,23,y,a,15);text(ax,69,y,b,14,MUTED)
        if i<4:ax.plot([0,99],[y-10,y-10],color=LINE,lw=.5)
    save(fig,'02_protocol',claim='当前A/B保留统一的V5比较合同，历史C不参与同条件排名',sources=[OUTLINE])

def diagnostic_map():
    fig=page('速度大小分数较高，还不能回答近壁梯度是否正确',
       '需要区分场值、局部分量与求导误差；下列是待验证的诊断链，不是假想的 CFD 剖面。','13 / 从现象到验证',
       '结构设计只能提出机制假设；是否改善近壁梯度，必须用同病例、同坐标和同一算子测量。')
    ax=canvas(fig)
    for x,title,body,c in [(0,'预测速度','体内 u, v, w',BLUE),(35,'近壁梯度','局部分量 + 法向距离',PURPLE),(70,'壁面 WSS','梯度 / 流变 / 校准',TEAL)]:
        node(ax,x,73,28,23,title,body,c,body_size=13)
    arrow(ax,(28,84),(35,84),BLUE);arrow(ax,(63,84),(70,84),PURPLE)
    headers=[(0,'已有观察'),(35,'尚未区分的解释'),(70,'下一张证据图')]
    for x,s in headers:text(ax,x,59,s,17,INK,True)
    rows=[('VF6 的速度大小有较好拟合','模长接近，但方向或近壁分量可能有误','分量 / 近壁壳层误差'),
          ('旧 R5V 的派生 WSS 明显下降','场误差与求导步骤误差各占多少？','VF6 与 CFD 同算子对照'),
          ('WSS 难例的误差分布不均','局部幅度不足，还是热点位置偏移？','原量 + selfmax + 原生面')]
    for i,vals in enumerate(rows):
        y=42-i*18
        for j,v in enumerate(vals):
            # Deliberate two-line labels keep each inferential role legible.
            chunks={'VF6 的速度大小有较好拟合':'VF6 的速度大小\n有较好拟合',
             '模长接近，但方向或近壁分量可能有误':'模长接近，但方向或\n近壁分量可能有误',
             '旧 R5V 的派生 WSS 明显下降':'旧 R5V 的派生 WSS\n明显下降',
             '场误差与求导步骤误差各占多少？':'场误差与求导步骤误差\n各占多少？',
             'WSS 难例的误差分布不均':'WSS 难例的误差\n分布不均',
             '局部幅度不足，还是热点位置偏移？':'局部幅度不足，还是\n热点位置偏移？'}
            text(ax,j*35,y,chunks.get(v,v),14,[INK,MUTED,TEAL][j])
    save(fig,'13_diagnostic_map',claim='速度到WSS的差距需要局部分量与冻结算子对照才能定位',sources=[OUTLINE])

def loss_panels(physics=False):
    data=pd.read_csv(CURVES);assert len(data)==9985 and np.array_equal(data.epoch,np.arange(9985))
    if physics:
        keys=['mean_no_slip_raw','mean_inlet_bc_raw','mean_continuity_raw','mean_momentum_x_raw','mean_momentum_y_raw','mean_momentum_z_raw']
        labels=['壁面无滑移','入口边界','连续性','动量 x','动量 y','动量 z'];cols=3;stem='15_physics_loss_terms'
        title='边界和 PDE 六项分开看，定位持续不降的残差'
        sub='每个小图一个未加权项；纵轴为各项缩放后的无量纲 MSE，不能据量级比较梯度强弱。'
    else:
        keys=['mean_data_u_raw','mean_data_v_raw','mean_data_w_raw','mean_data_p_raw'];labels=['速度分量 u','速度分量 v','速度分量 w','相对压力 p'];cols=2;stem='14_data_loss_terms'
        title='把 u、v、w、p 分开看，确认数据拟合是否完成'
        sub='每个小图一个标准化数据 MSE；淡线保留全部逐 epoch 均值，实线为 51 轮居中移动均值。'
    fig=page(title,sub,'14–15 / 历史 V4 · PointNet++ EMA',
        'V4-SP-PNPP-BC-PDE-EMA · seed1234 · epoch 0–9984；未补齐、外推或重训。raw loss 不等于加权贡献。')
    gs=fig.add_gridspec(2,cols,left=.08,right=.97,bottom=.16,top=.74,hspace=.63,wspace=.25)
    axes=[]
    for i,(key,label) in enumerate(zip(keys,labels)):
        ax=fig.add_subplot(gs[i//cols,i%cols]);axes.append(ax);c=ORANGE if physics else BLUE
        y=data[key].to_numpy();assert np.isfinite(y).all() and np.all(y>0)
        ax.plot(data.epoch,y,color=c,alpha=.18,lw=.65,rasterized=True)
        ax.plot(data.epoch,pd.Series(y).rolling(51,center=True,min_periods=1).mean(),color=c,lw=1.5)
        ax.set_yscale('log');ax.set_xlim(0,9984);ax.set_xticks([0,5000,9984]);ax.set_xlabel('Epoch',fontsize=12)
        ax.tick_params(axis='x',pad=8)
        ax.set_title(label,loc='left',pad=14,fontsize=16,color=c)
        ax.set_ylabel('Raw loss（log scale）',fontsize=12)
        ax.grid(axis='y',color=LINE,alpha=.5,lw=.4)
        panel_label(ax,chr(97+i))
    data[['epoch']+keys].to_csv(OUT/'source_data'/f'{stem}.csv',index=False)
    save(fig,stem,claim=title,sources=[CURVES],axes=axes,
        note='All 9985 logged epochs retained. Raw values plus centered 51-epoch mean; independently scaled positive log axes. No uncertainty band or replicate inference.')

def physics_conflicts():
    gradient=PHYS/'gradient_summary_last500.csv';loss=PHYS/'loss_summary_last500.csv'
    g=pd.read_csv(gradient);l=pd.read_csv(loss)
    fig=page('数据与无滑移持续冲突，加入约束伴随拟合代价',
       '左：PointNet 的数据 / 无滑移梯度持续相反；右：两种骨干的共同数据误差随约束加入而增大。','18 / 历史机制证据',
       '仅历史单 seed 诊断。左为末500轮中位数与10%–90%分位；右为末500轮均值，不能解释为泛化或多seed显著性。')
    gs=fig.add_gridspec(1,2,left=.11,right=.96,bottom=.24,top=.74,wspace=.38)
    ax=fig.add_subplot(gs[0]);bx=fig.add_subplot(gs[1])
    modes=['BC','BC-PDE-F','BC-PDE-EMA']
    # Use actual stored labels rather than assuming their naming scheme.
    modes=list(g['mode'].drop_duplicates());assert len(modes)==3
    no=g[g.term=='no_slip'].set_index('mode').loc[modes]
    ys=np.arange(3)
    ax.errorbar(no['median'],ys,xerr=np.vstack([no['median']-no.q10,no.q90-no['median']]),fmt='o',ms=8,color=RED,elinewidth=2,capsize=4)
    ax.axvline(0,color=GREY,lw=.8,ls='--');ax.set(xlim=(-1.03,.18),ylim=(-.5,2.5),yticks=ys,yticklabels=['+ BC','+ PDE 固定','+ PDE EMA'],xlabel='cos（数据梯度，无滑移梯度）')
    ax.invert_yaxis();ax.set_title('负余弦说明优化方向存在竞争',loc='left',pad=17,fontsize=17)
    for y,v in zip(ys,no['median']):ax.annotate(f'{v:.3f}',(v,y),xytext=(0,15),textcoords='offset points',ha='center',fontsize=13,color=RED)
    for bone,c,marker in [('PN',GREY,'o'),('PNPP',BLUE,'s')]:
        names=[f'V4-SP-{bone}-DATA-s1234',f'V4-SP-{bone}-BC-s1234',f'V4-SP-{bone}-BC-PDE-F-s1234',f'V4-SP-{bone}-BC-PDE-EMA-s1234']
        vals=l.set_index('run').loc[names,'mean_data_total'].to_numpy()
        bx.plot(np.arange(4),vals,marker=marker,color=c,lw=2.2 if bone=='PN' else 1.3,
                ms=9 if bone=='PN' else 5,markerfacecolor='white' if bone=='PN' else c,
                ls='-' if bone=='PN' else '--')
        label_y=vals[-1]+(.10 if bone=='PN' else -.10)
        bx.plot([3.04,3.16],[vals[-1],label_y],lw=.6,color=c)
        bx.text(3.20,label_y,'PointNet' if bone=='PN' else 'PointNet++',fontsize=12,color=c,va='center')
    bx.set_xticks(np.arange(4),['Data','+ BC','+ PDE 固定','+ PDE EMA']);bx.set_ylabel('共同数据 MSE');bx.set_ylim(0,l.mean_data_total.max()*1.27)
    bx.set_title('比较共同数据项，避免给不同总 loss 排名',loc='left',pad=17,fontsize=16)
    bx.set_xlim(-.2,4.4);bx.tick_params(axis='x',pad=8)
    panel_label(ax,'a');panel_label(bx,'b')
    fig.text(.11,.115,'支持：当前表达、采样与权重未能兼顾两类目标。下一步应核查边界 / 标签一致性，再在 V5 数据底座上重建对照。',fontsize=14,color=MUTED)
    g.to_csv(OUT/'source_data/18_gradient_summary.csv',index=False);l.to_csv(OUT/'source_data/18_loss_summary.csv',index=False)
    save(fig,'18_physics_conflicts',claim='历史数据与无滑移优化冲突伴随共同数据拟合代价，不能据此否定物理方程',sources=[gradient,loss],axes=[ax,bx])

def discussion():
    fig=page('下一轮先区分错误来源，再决定增加哪个模块',
      '把讨论落到可观察、可比较、能改变下一步决策的验证上。','19 / 请老师帮助裁定',
      '以上是候选验证与材料需求，不代表已执行新实验；现有病例包支持在后处理软件继续取图。')
    ax=canvas(fig)
    for x,s in [(0,'待区分的问题'),(34,'需要补的证据'),(71,'结果如何改变决策')]:text(ax,x,97,s,18,INK,True)
    rows=[(TEAL,'WSS：幅度与位置','同病例原量 / selfmax\n原生面、热点位置与幅值','幅度主导 → 先校准尺度\n定位主导 → 再改局部提取'),
          (BLUE,'速度：整体与近壁','分量、近壁壳层、二次流\nVF6 / CFD 通过同一 WSS 算子','场值有误 → 修场表达\n算子有误 → 修计算步骤'),
          (ORANGE,'物理：设置与优化','先核对边界与标签一致性\n再做同 V5 底座 data / BC / PDE','约束冲突 → 查标签与采样\n一致但难学 → 查尺度与权重')]
    for i,(c,lab,evidence,decision) in enumerate(rows):
        y=73-i*29
        text(ax,0,y,lab,18,c,True);text(ax,34,y,evidence,16);text(ax,71,y,decision,15,MUTED)
        if i<2:ax.plot([0,99],[y-15,y-15],color=LINE,lw=.5)
    save(fig,'19_discussion',claim='三个方向的下一步都应由能区分错误来源的对照决定',sources=[OUTLINE])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',nargs='*');args=ap.parse_args()
    jobs={'01':routes,'02':protocol,'13':diagnostic_map,'14':lambda:loss_panels(False),'15':lambda:loss_panels(True),'18':physics_conflicts,'19':discussion}
    for key,fun in jobs.items():
        if not args.only or key in args.only:fun()

if __name__=='__main__':main()
