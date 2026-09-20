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

#!/usr/bin/env python3
"""Four source-grounded network schematics for the Chinese group meeting.

Figure contract: Python-only schematic-led composites, 16 x 9 in, editable
PDF/SVG plus 300 dpi PNG. These drawings describe implemented information flow;
function annotations are design intentions, not independent ablation effects.
03: hierarchical geometry features are restored before query-level regression.
04: X5 adds one support-feature residual and one log_z output residual.
05: X11 conditions FiLM through section context, not a separate WSS output.
06: PF6 and VF6 are independent models with distinct coarse-context paths.
No measurements are simulated and no existing cases or runs are reselected.
"""
from pathlib import Path
import json
from matplotlib.patches import Ellipse
from plot_style import (
    OUT, ROOT, INK, MUTED, BLUE, TEAL, PURPLE, ORANGE, GREY, LINE,
    page, canvas, text, node, arrow, route, panel_label, save,
)

BASE = ROOT / 'training_wss_min/baseline_models.py'
LOCAL = ROOT / 'training_wss_min/local_refinement.py'
CFG = ROOT / 'training_wss_min/configs'
WAVE2 = CFG / 'wss_local_wave2_20260912'
RESULT2 = ROOT / 'training_wss_min/experiments/wss_local_wave2_20260912/results.json'
RESULT1B = ROOT / 'training_wss_min/experiments/wss_local_wave1b_20260912/results.json'


def join(ax, x, y, color=TEAL):
    """Addition junction; ellipse compensation gives a physical circle."""
    aspect = ax.get_position().width * 16 / (ax.get_position().height * 9)
    ax.add_patch(Ellipse((x, y), 3.0, 3.0 * aspect,
                         edgecolor=color, facecolor='white', linewidth=1.2))
    # A vector plus is a schematic operator, not a text annotation in the ring.
    ax.plot([x-.5,x+.5],[y,y],color=color,lw=1.6,solid_capstyle='butt')
    ax.plot([x,x],[y-.5*aspect,y+.5*aspect],color=color,lw=1.6,solid_capstyle='butt')


def diagram_note(stem, panels, note):
    """Human review rows are completed after the actual exports are inspected."""
    obj = {
        'archetype': 'schematic-led composite', 'backend': 'python',
        'size_inches': [16, 9], 'n_and_uncertainty':
        'Architecture, not estimated effects. X11 subtitle is the original three-seed arithmetic mean difference; no significance claim.',
        'source_reuse': 'Rebuilt from verified implementation; shared typography and export helpers only.',
        'panels': panels, 'notes': note,
    }
    (OUT/'qa'/f'{stem}.review.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def backbone():
    claim = '共用主干先逐层编码几何，再恢复support特征并对query回归。'
    fig = page('共用主干：先读几何，再把特征还原到查询点',
               '以 R4 为结构参照；X5/X5X11 与体场模型的新增通路在后续图中展开。',
               '03  /  网络如何看见几何',
               '结构作用表示设计意图；模块净收益由同条件消融判断。SA 半径使用逐例归一化坐标，不是 mm。')
    ax = canvas(fig)
    specs = [
        ('几何输入', 'R4：17D\n5000 壁面点', BLUE),
        ('逐点编码', 'Stem\n形成 32D 特征', BLUE),
        ('局部聚合', 'SA1 + 残差\n125 个中心点', BLUE),
        ('邻域交互', 'SA2 + 注意力\n125 个中心点', PURPLE),
        ('粗层压缩', 'SA3\n32 个中心点', BLUE),
        ('特征回传', '三级 FP\n恢复 5000 点', TEAL),
        ('查询与回归', '3-NN 特征插值\n回归每个 query', TEAL),
    ]
    xs = [1 + i * 14.25 for i in range(7)]
    for i, (title, body, color) in enumerate(specs):
        node(ax, xs[i], 59, 12.0, 24, title, body, color,
             fontsize=15, body_size=13)
        if i < 6: arrow(ax, (xs[i]+12, 71), (xs[i+1], 71))
    route(ax, [(21.25,83),(21.25,93),(78.25,93),(78.25,83)], TEAL, True)
    text(ax, 49.75, 98, '编码器各层 skip → 对应 FP 层', 14, TEAL, ha='center')
    functions = [
        (1, '几何位置编码', 'LocalGeoPE', '相对位置、距离与几何差\n共同参与邻域消息。', BLUE),
        (35, '局部残差修正', 'PointNeXt-R', '保留原特征，再叠加\n局部聚合与逐点修正。', TEAL),
        (69, '同一邻域内交互', 'SA2 Local Transformer', '邻居之间先交换信息，\n再聚合到中心点。', PURPLE),
    ]
    for x, title, label, body, color in functions:
        ax.plot([x, x+28], [43,43], color=LINE, lw=.8)
        text(ax,x,35,title,18,color,True)
        text(ax,x,25,label,13,MUTED)
        text(ax,x,11,body,15,INK)
    save(fig,'03_backbone',claim=claim,sources=[BASE],axes=[ax],
         note='Single information-flow schematic. Lower annotations explain three different operations, not quantitative comparable panels. R4 has no bottleneck global Transformer.')
    diagram_note('03_backbone', ['hierarchical feature path','three functional annotations'],
                 'The seven nodes simplify channel widths; 3-NN interpolates features before target regression.')


def x5():
    claim = 'X5把Murray尺度先验输入几何通路，并分别在support特征和log_z输出处做局部残差修正。'
    fig = page('X5：尺度先验与局部细节，在两处修正预测',
               '27D = 基础几何 17D + 壁面微分几何 8D + Murray 2D；输入只含可由几何得到的信息。',
               '04  /  直接预测 WSS',
               '① 方向分支修正 support 特征；② 完整壁面 patch 修正 log_z 输出。FiLM 的独立净收益尚未单独消融。')
    ax = canvas(fig)
    pipeline = [(1,12,'27D 输入','含 Murray 2D',BLUE),
                (17,12,'Stem','32D 特征',BLUE),
                (33,16,'三级编码器','SA1 → SA2 → SA3',BLUE),
                (53,12,'三级 FP','support 32D',TEAL),
                (76,17,'查询 + 基础头','3-NN → log_z',TEAL)]
    for x,w,title,body,color in pipeline:
        node(ax,x,66,w,21,title,body,color,fontsize=15,body_size=13)
    for start,end in [(13,17),(29,33),(49,53),(65,68.5),(71.5,76)]:
        arrow(ax,(start,76.5),(end,76.5))
    join(ax,70,76.5)
    text(ax,30,60,'① 在 support 分辨率补充细节',15,TEAL,True)
    node(ax,30,35,32,20,'方向多尺度局部分支',
         '三尺度 · 每尺度 16 邻居',TEAL,fontsize=16,body_size=14)
    route(ax,[(23,66),(23,45),(30,45)],TEAL)
    route(ax,[(62,45),(70,45),(70,72.8)],TEAL)
    text(ax,1,46,'读取 Stem',13,MUTED)
    node(ax,76,35,22,21,'FiLM 上下文',
         'query 32D + SA3 256D',PURPLE,fontsize=16,body_size=13)
    arrow(ax,(84.5,66),(84.5,56),PURPLE)
    node(ax,1,2,22,22,'完整壁面 K16 patch',
         '27D 几何 + 相对 mm',BLUE,fontsize=15,body_size=13)
    node(ax,32,2,31,22,'Patch → FiLM → 残差',
         '几何编码 · γ/β 调制 · Δlog_z',PURPLE,fontsize=16,body_size=13)
    text(ax,32,29,'② 在 query 处修正输出',15,PURPLE,True)
    arrow(ax,(23,13),(32,13),BLUE)
    route(ax,[(76,40),(68,40),(68,31),(59,31),(59,24)],PURPLE)
    arrow(ax,(63,13),(78.5,13),PURPLE)
    join(ax,80,13,ORANGE)
    route(ax,[(93,76.5),(99.5,76.5),(99.5,28),(80,28),(80,16.8)],ORANGE)
    node(ax,86,2,13,22,'WSS（Pa）','log_z 反变换',ORANGE,fontsize=15,body_size=13)
    arrow(ax,(81.5,13),(86,13),ORANGE)
    text(ax,1,97,'Murray：分支流量份额 + 剪切尺度先验',15,ORANGE,True)
    save(fig,'04_x5',claim=claim,sources=[BASE,LOCAL,CFG/'wss_local_wave1_20260912/X5_s1234.json'],axes=[ax],
         note='Two additions are different quantities: support32D and log_z. FiLM context label references the query feature and the max-pooled SA3 case feature. The downward reference arrow is from the query block, not from its scalar output.')
    diagram_note('04_x5',['backbone path','support local residual','query patch residual'],
                 'Murray input is not a PDE loss. Patch uses atlas relative coordinates and mean pooling. Local radii .015/.03/.06 are normalized.')


def x11():
    claim = 'X11按截面聚合support并跨截面交换信息，生成的上下文残差只进入FiLM条件。'
    fig = page('X11：把截面上下文送入 FiLM，指导局部修正',
               'X5 → X5X11：物理 R² 三 seed 均值约 +0.009；高值幅值改善，同时 MAE 与热点 IoU 略差。',
               '05  /  X5X11 新增的通路',
               '三 seed 配对 ΔR²：+0.007 / +0.005 / +0.014（1234 / 7 / 2025，ckpt_best，test34）；X5 关闭截面 token 通路。')
    ax=canvas(fig)
    top=[(1,19,'support 特征','FP + 局部后的 32D\n以及原始 27D 输入'),
         (25,20,'按截面聚合','segment × 4 mm 箱\n每段最多 64 箱'),
         (50,22,'截面 token 交互','2 层 · 4 头 Transformer\n128D → Δ256D'),
         (77,22,'按 query 查表','本截面上下文\n空箱取同段最近占用箱')]
    for x,w,t,b in top:node(ax,x,68,w,26,t,b,PURPLE,fontsize=16,body_size=13)
    for a,b in [(20,25),(45,50),(72,77)]:arrow(ax,(a,81),(b,81),PURPLE)
    node(ax,1,29,22,23,'病例粗层上下文','SA3 max-pool：256D',BLUE,fontsize=16,body_size=13)
    arrow(ax,(23,40.5),(28.5,40.5),BLUE)
    join(ax,30,40.5,PURPLE)
    route(ax,[(88,68),(88,59),(30,59),(30,44.3)],PURPLE)
    text(ax,50,62.7,'只更新调制上下文',14,PURPLE,True,ha='center')
    arrow(ax,(31.5,40.5),(43,40.5),PURPLE)
    node(ax,43,29,22,23,'拼接 FiLM 条件','query 32D ⊕ 上下文',PURPLE,fontsize=16,body_size=13)
    node(ax,1,1,22,20,'query 特征','3-NN 插值得到 32D',TEAL,fontsize=16,body_size=13)
    route(ax,[(23,11),(54,11),(54,29)],TEAL)
    node(ax,71,29,28,23,'完整壁面 patch → FiLM','K16 几何 → γ/β 调制\n输出局部残差 Δlog_z',PURPLE,fontsize=15,body_size=13)
    arrow(ax,(65,40.5),(71,40.5),PURPLE)
    arrow(ax,(85,29),(85,17),ORANGE)
    text(ax,85,9,'基础 log_z + Δlog_z',15,ORANGE,True,ha='center')
    text(ax,85,1,'反变换 → WSS（Pa）',14,INK,ha='center')
    save(fig,'05_x11',claim=claim,sources=[BASE,LOCAL,WAVE2/'X5X11_s1234.json',RESULT2,RESULT1B],axes=[ax],
         note='The context residual is not a WSS residual. Subtitle is a bounded arithmetic summary of three paired seeds, not a significance result. Nearest occupied lookup is constrained to the same segment.')
    diagram_note('05_x11',['section encoding','FiLM conditioning','unchanged output residual path'],
                 'The encoded section residual modifies the case context; query32D is then concatenated. X11 has no new WSS head.')


def volume():
    claim = 'PF6与VF6是两套独立权重模型，其BT接入位置和support局部分支不同，但均用QAD几何残差回答体内query。'
    fig=page('PF6 与 VF6：两套独立模型，用不同上下文回答 query',
             '共同输入为 20D = 基础几何 17D + 到壁距离 1D + Murray 2D；5000 个壁面 support。',
             '06  /  压力与速度的结构差别',
             'QAD 是 query 几何修正，未直接监督近壁导数。VF6 的局部分支使用普通邻域；两者均未启用 X5 的 patch / FiLM。')
    rows=[]
    configs=[('a',.505,BLUE,'PF6  |  壁面 + 内部 query',
              [('壁面编码','Stem + SA1/2'),('SA3 → 全局 BT','常规 SA3 后追加\n1 层 · 8 头'),
               ('三级 FP','恢复 support\n无局部分支'),('3-NN 查询','插值到 query\n32D 特征'),
               ('基础头 + QAD','输出端残差\n1 通道'),('相对压力 p','壁面 ∪ 内部\nPa')]),
             ('b',.245,TEAL,'VF6  |  仅内部 query',
              [('壁面编码','Stem + SA1/2'),('SA3 内置 BT','单半径 r = 0.20\n1 层 · 8 头'),
               ('FP + 局部分支','普通三尺度分支\n加到 support'),('3-NN 查询','插值到 query\n32D 特征'),
               ('基础头 + QAD','输出端残差\n3 通道'),('速度 u / v / w','仅内部点\nm/s')])]
    for label,bottom,color,heading,blocks in configs:
        ax=canvas(fig,(.04,bottom,.92,.225)); rows.append(ax)
        panel_label(ax,label)
        text(ax,1,94,heading,17,color,True)
        for i,(t,b) in enumerate(blocks):
            x=1+17*i
            node(ax,x,17,14,58,t,b,color,fontsize=14.3,body_size=12.6)
            if i<5:arrow(ax,(x+14,46),(x+17,46),color)
    fig.text(.052,.161,'QAD-lite：几何差异 × 特征门控 → 输出残差',fontsize=15,color=PURPLE,weight='bold')
    fig.text(.052,.125,'Δ20D + Δxyz + 距离；MLP 24 → 32 → 32',fontsize=13,color=MUTED)
    fig.text(.60,.161,'VF6 速度 → 冻结算子 → WSS',fontsize=15,color=ORANGE,weight='bold')
    fig.text(.60,.125,'外部后处理；本批 VF6 闭环待补',fontsize=14,color=MUTED)
    save(fig,'06_volume_models',claim=claim,sources=[BASE,WAVE2/'PF6_s1234.json',WAVE2/'VF6_s1234.json'],
         axes=rows,column_groups=[['a','b']],
         note='Two equal-width equal-height rows are aligned for architectural comparison. The bottom QAD explanation and velocity-to-WSS note are shared annotations outside the comparison axes. The arrow to WSS is external postprocessing, not an implemented VF6 head.')
    diagram_note('06_volume_models',['a: PF6 pressure route','b: VF6 velocity route'],
                 'Both rows share corresponding block positions. The VF6 SA3 single radius must not be confused with its three-radius support branch. Separate weights, not joint u,p training.')


if __name__=='__main__':
    backbone(); x5(); x11(); volume()
    for stem in ['03_backbone','04_x5','05_x11','06_volume_models']:
        for suffix in ['.pdf','.svg','.png']:
            assert (OUT/'figures'/f'{stem}{suffix}').is_file()
