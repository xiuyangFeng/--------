"""Data-growth figures from existing case-balanced WSS results; no inference.

Three-seed means and sample SD; same 136 held-out case identities in the main
comparison. Dotted version bridge is descriptive, not a fitted learning curve.
"""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import NullLocator,ScalarFormatter
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
sys.path.insert(0,'/public/newhome/cy/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
DATA=json.loads((OUT/'source_data.json').read_text())
S=DATA['summary']; R=DATA['seed_rows']; SEEDS=(1234,7,2025)
OLD=[r for r in S if r['version']=='V5.1']
NEW=next(r for r in S if r['version']=='V5.2' and r['recipe']=='X5D')
CAP=next(r for r in S if r['version']=='V5.2' and r['recipe']=='X5Dcap')
font_path='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
font_manager.fontManager.addfont(font_path)
font_name=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'font.family':['DejaVu Sans',font_name,'sans-serif'],
 'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
 'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':0.8,'legend.frameon':False,'savefig.facecolor':'white'})
INK='#253442'; MUTED='#627281'; GRID='#DFE6EC'; BLUE='#356EAA'; TEAL='#168C88'; ORANGE='#CC7845'; GREY='#8A98A5'

def style(ax):
 ax.set_facecolor('white');ax.grid(False);ax.set_axisbelow(True)
 ax.tick_params(axis='both',colors=INK,length=3.5)
 for key in ('left','bottom'):ax.spines[key].set_color('#A9B6C1')

def save(fig,name):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=OUT/(name+'.alignment.json'),
  tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
 fig.savefig(OUT/(name+'.png'),dpi=300,facecolor='white')
 fig.savefig(OUT/(name+'.pdf'),facecolor='white')
 fig.savefig(OUT/(name+'.svg'),facecolor='white')
 plt.close(fig)

x=np.array([r['n_train_mean'] for r in OLD]);y=np.array([r['r2_cb'] for r in OLD]);sd=np.array([r['r2_cb_sd'] for r in OLD])
# Every training count must be strictly positive for the log2 axis.
assert np.all(x>0) and np.all(np.diff(x)>0)
old=OLD[-1];d_old=y[-1]-y[0];d_new=NEW['r2_cb']-old['r2_cb'];d_cap=CAP['r2_cb']-NEW['r2_cb'];mae_gain=1-NEW['mae']/old['mae']

# Slide-sized export: a wide primary panel plus a narrower paired-MAE control.
fig=plt.figure(figsize=(12.8,7.2),facecolor='white')
gs=fig.add_gridspec(1,5,left=.078,right=.963,bottom=.265,top=.686,wspace=.86)
a=fig.add_subplot(gs[0,:3],label='a');b=fig.add_subplot(gs[0,3:],label='b')
for ax in (a,b):style(ax)
fig.text(.078,.94,'更多训练病例，WSS 预测持续改善',fontsize=21,fontweight='bold',color=INK)
fig.text(.078,.892,'统一在同 136 例折外病例上比较  ·  三个 seed 的单模型指标  ·  病例等权、物理 Pa 空间',fontsize=11,color=MUTED)
for pos,label,value,col in [(.078,'V5.1 同协议学习曲线：约 23 → 91 例',f'R² +{d_old:.3f}',BLUE),(.409,'版本扩展：每折约 91 → 209 例',f'R² +{d_new:.3f}',TEAL),(.755,'同 136 例，X5D 版本对比',f'MAE −{mae_gain*100:.1f}%',TEAL)]:
 fig.text(pos,.822,label,fontsize=10,color=MUTED)
 fig.text(pos,.775,value,fontsize=18,fontweight='bold',color=col)

# Panel a: actual training counts on a log2 axis. Old solid curve is controlled.
a.set_xscale('log',base=2)
a.fill_between(x,y-sd,y+sd,color=BLUE,alpha=.12,lw=0)
a.plot(x,y,color=BLUE,lw=2.7,marker='o',ms=7,mec='white',mew=1.0,zorder=4)
a.errorbar(x,y,yerr=sd,fmt='none',ecolor=BLUE,elinewidth=1.1,capsize=3,zorder=3)
for s,m in zip(SEEDS,('o','s','D')):
 yy=[r['r2_cb'] for r in R if r['version']=='V5.1' and r['seed']==s]
 a.plot(x,yy,ls='none',marker=m,ms=3.8,mfc='white',mec=BLUE,mew=.8,alpha=.72,zorder=5)
# Cross-version bridge is intentionally dashed and has no interpolation band.
a.plot([x[-1],NEW['n_train_mean']],[y[-1],NEW['r2_cb']],color=GREY,lw=1.6,ls=(0,(4,3)),zorder=2)
for row,color,marker in [(NEW,TEAL,'^'),(CAP,ORANGE,'D')]:
 a.errorbar(row['n_train_mean'],row['r2_cb'],yerr=row['r2_cb_sd'],fmt=marker,color=color,
  ms=8,mec='white',mew=.9,elinewidth=1.3,capsize=4,zorder=7)
for xx,yy,ss in zip(x,y,sd):
 a.text(xx,yy+ss+.008,f'{yy:.3f}',ha='center',va='bottom',fontsize=10,color=BLUE)
a.annotate(f"X5Dcap  {CAP['r2_cb']:.3f}",xy=(CAP['n_train_mean'],CAP['r2_cb']),xytext=(245,.782),
 ha='left',va='center',fontsize=10,color=ORANGE,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.8))
a.annotate(f"X5D  {NEW['r2_cb']:.3f}",xy=(NEW['n_train_mean'],NEW['r2_cb']),xytext=(245,.738),
 ha='left',va='center',fontsize=10,color=TEAL,arrowprops=dict(arrowstyle='-',color=TEAL,lw=.8))
a.text(135,.686,'CV3 → CV5\n版本扩展',ha='center',va='center',color=MUTED,fontsize=9.5)
a.set_xlim(18,385);a.set_ylim(.615,.802)
a.set_xticks([*x,NEW['n_train_mean']]);a.set_xticklabels(['23','45','68','91','209'])
a.xaxis.set_minor_locator(NullLocator());a.set_yticks([.62,.66,.70,.74,.78])
a.set_xlabel('每折实际训练病例数（折均；对数刻度）',labelpad=10,color=INK)
a.set_ylabel('病例等权 R²（越高越好）',labelpad=9,color=INK)
a.set_title('a   X5D 学习曲线与 V5.2 扩展',loc='left',pad=14,fontweight='bold',color=INK)

# Panel b: absolute error on the same case cohort; show all matched seed IDs.
models=[('V5.1','X5D',BLUE,'o'),('V5.2','X5D',TEAL,'^'),('V5.2','X5Dcap',ORANGE,'D')]
for s,m in zip(SEEDS,('o','s','D')):
 vals=[next(r['mae'] for r in R if r['version']==v and r['recipe']==recipe and r['fraction']==100 and r['seed']==s) for v,recipe,_,_ in models]
 b.plot(range(3),vals,color='#BAC4CE',lw=.9,marker=m,ms=3.8,mfc='white',mec=GREY,mew=.7,zorder=2)
for i,(r,(_,_,c,m)) in enumerate(zip([old,NEW,CAP],models)):
 b.errorbar(i,r['mae'],yerr=r['mae_sd'],color=c,fmt=m,ms=8,mec='white',mew=1.0,capsize=4,elinewidth=1.5,zorder=5)
 b.text(i,r['mae']+r['mae_sd']+.032,f"{r['mae']:.3f}",ha='center',va='bottom',color=c,fontsize=11,fontweight='bold')
b.set_xlim(-.35,2.35);b.set_ylim(1.28,1.575);b.set_yticks([1.30,1.35,1.40,1.45,1.50,1.55])
b.set_xticks(range(3));b.set_xticklabels(['V5.1\nX5D\n训练≈91例','V5.2\nX5D\n训练≈209例','V5.2\nX5Dcap\n训练≈209例'])
b.tick_params(axis='x',length=0,pad=8)
b.set_ylabel('MAE（Pa，越低越好）',labelpad=8,color=INK)
b.set_title('b   同病例对照：绝对误差下降',loc='left',pad=14,fontweight='bold',color=INK)
fig.text(.078,.163,'实线：V5.1 固定 CV3 的嵌套训练子集；虚线：V5.1→V5.2 的版本扩展，含 CV3→CV5 与几何修复。',fontsize=9.5,color=MUTED)
fig.text(.078,.119,'阴影／误差棒：三个 seed 的均值 ± SD；细线连接同 seed 结果。所有点为单模型指标，未混入集成收益。',fontsize=9.5,color=MUTED)
fig.text(.078,.075,'总队列 170→261 例；CV3 使用原 train136，CV5 使用全261例。跨版本变化不能单独归因于数据量。',fontsize=9.5,color=MUTED)
save(fig,'WSS_数据增益_V51_V52_同136例')

# Companion retains all original strata, now with SD for every curve.
fig,ax=plt.subplots(figsize=(10.5,6.5),facecolor='white')
fig.subplots_adjust(left=.105,right=.785,bottom=.205,top=.72)
style(ax)
fig.text(.105,.925,'WSS 学习曲线：增加病例后，三个分层均改善',fontsize=18,fontweight='bold',color=INK)
fig.text(.105,.865,'V5.1 · 固定 CV3 与同一评估病例集 · 嵌套训练子集 · 三个 seed 均值 ± SD',fontsize=10.5,color=MUTED)
fig.text(.105,.792,f'全体 R²：{y[0]:.3f} → {y[-1]:.3f}    |    末段 68 → 91 例：R² 仍增加 {y[-1]-y[-2]:.3f}',fontsize=12,color=BLUE,fontweight='bold')
series=[('r2_nonjet','非射流 111例',TEAL,'v'),('r2_cb','全体 136例',BLUE,'o'),('r2_jet','射流 25例',ORANGE,'^')]
for key,name,c,m in series:
 yy=np.array([r[key] for r in OLD]);ss=np.array([r[key+'_sd'] for r in OLD])
 ax.fill_between(x,yy-ss,yy+ss,color=c,alpha=.11,lw=0)
 ax.errorbar(x,yy,yerr=ss,color=c,lw=2.6 if key=='r2_cb' else 1.8,
  ls='-' if key=='r2_cb' else (0,(5,3)),marker=m,ms=7,mec='white',mew=1,
  capsize=3,elinewidth=.9,zorder=4 if key=='r2_cb' else 3)
 ax.text(95,yy[-1],f'{name}\n{yy[-1]:.3f}',color=c,fontsize=10.5,va='center',ha='left',fontweight='bold')
ax.set_xlim(18,95);ax.set_ylim(.55,.805)
ax.set_xticks(x);ax.set_xticklabels(['23例\n25%','45例\n50%','68例\n75%','91例\n100%'])
ax.set_yticks(np.arange(.55,.801,.05));ax.set_ylabel('病例等权 R²（物理 Pa 空间）',color=INK,labelpad=10)
ax.set_xlabel('每折实际训练病例数（折均）／训练子集比例',color=INK,labelpad=10)
fig.text(.105,.071,'射流：病例真值 WSS 的 p99 > 40 Pa。误差范围描述训练 seed 波动，非置信区间；未使用 test34。',fontsize=9.2,color=MUTED)
save(fig,'WSS_学习曲线_V51_分层优化')

# Captions and machine-readable effect summary stay with the figures.
effects={'controlled_learning_r2_gain_23_to91':d_old,'version_r2_gain_same136':d_new,'version_mae_relative_reduction_same136':mae_gain,'cap_r2_gain_same136':d_cap,'v51_lc100_r2':old['r2_cb'],'v51_lc100_mae':old['mae'],'v52_x5d_same136_r2':NEW['r2_cb'],'v52_x5d_same136_mae':NEW['mae'],'v52_cap_same136_r2':CAP['r2_cb'],'v52_cap_same136_mae':CAP['mae']}
(OUT/'figure_numbers.json').write_text(json.dumps(effects,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(effects,ensure_ascii=False,indent=2))
