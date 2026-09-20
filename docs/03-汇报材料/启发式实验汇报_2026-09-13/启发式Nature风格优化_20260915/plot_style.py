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
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,**opts)
    fig.savefig(str(dest)+'.png',dpi=300)
    plt.close(fig)
    meta={'stem':stem,'claim':claim,'sources':[{'path':str(Path(p).resolve()),'sha256':sha(p)} for p in sources],
        'exports':{'png':str(dest.relative_to(OUT))+'.png'},
        'alignment':alignment.get('verdict',alignment.get('status')),'note':note}
    Path(str(qa)+'.metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(stem,alignment.get('verdict',alignment.get('status')),flush=True)
    return meta
