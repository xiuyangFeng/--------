from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
BASE=ROOT/'outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak'
GC=OUT.parent/'V4_汇报图件重绘_2026-09-03'
KEYS=['DATA','BC','BC-PDE-F','BC-PDE-EMA']
LABELS=['Data','Data + BC','Data + BC + PDE (fixed)','Data + BC + PDE (EMA)']
SHORT=['BC','PDE fixed','PDE EMA']
COL=['#3174B5','#D87542','#258C80','#8661AB']
INK='#243549'; GRAY='#617184'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#CBD4DD','xtick.color':GRAY,'ytick.color':GRAY,'axes.titleweight':'bold','axes.titlesize':12,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
curves={}; stats=[]; sources=[]
for b in ['PN','PNPP']:
 for k in KEYS:
  run=f'V4-SP-{b}-{k}-s1234'; p=BASE/run/'epoch_progress.jsonl'
  d=pd.read_json(p,lines=True); assert d.epoch.is_unique and np.all(np.diff(d.epoch)==1)
  curves[b,k]=d
  t=d[d.epoch>d.epoch.max()-500]
  stats.append({'run':run,'epochs':len(d),'first_epoch':int(d.epoch.min()),'last_epoch':int(d.epoch.max()),**{c:float(t[c].mean()) for c in ['mean_data_total','mean_total','mean_no_slip_raw','mean_continuity_raw','mean_momentum_group_raw','mean_lambda_phy']},'fraction_lambda_ge_9_99':float((d.mean_lambda_phy>=9.99).mean())})
  sources.append({'run':run,'curve':str(p),'config':str(BASE/run/'resolved_config.json')})
pd.DataFrame(stats).to_csv(OUT/'loss_summary_last500.csv',index=False)
grad={}; gs=[]
terms=['no_slip','momentum','continuity','inlet_bc']
for k in KEYS[1:]:
 p=GC/f'gradcos_V4-SP-PN-{k}-s1234.csv'; d=pd.read_csv(p)
 assert (d.global_step%50==0).all()
 grad[k]=d; t=d[d.epoch>d.epoch.max()-500]
 for term in terms:
  if k=='BC' and term in ['momentum','continuity']: continue
  y=t['cos_grad_data_'+term]
  gs.append({'mode':k,'term':term,'n':len(y),'median':y.median(),'q10':y.quantile(.1),'q90':y.quantile(.9),'negative_fraction':(y<0).mean()})
 sources.append({'gradient_csv':str(p),'original_log':str(BASE/f'V4-SP-PN-{k}-s1234/training_progress.jsonl')})
pd.DataFrame(gs).to_csv(OUT/'gradient_summary_last500.csv',index=False)
(OUT/'sources.json').write_text(json.dumps(sources,indent=2))
pdf=PdfPages(OUT/'steady_pinn_figures.pdf')
def style(ax,log=False):
 [ax.spines[t].set_visible(False) for t in ['top','right']]; ax.grid(axis='y',color='#E8EDF2',lw=.7); ax.set_axisbelow(True)
 if log: ax.set_yscale('log')
 ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:g}k' if x else '0'))
def heading(fig,title,sub,foot):
 fig.text(.065,.955,title,fontsize=20,weight='bold');fig.text(.065,.914,sub,fontsize=10.5,color=GRAY)
 fig.text(.065,.025,foot,fontsize=9,color=GRAY)
def save(fig,name):
 for ext in ['png','pdf','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=220)
 pdf.savefig(fig);plt.close(fig)
def legends(fig,indices):
 fig.legend(handles=[Line2D([],[],color=COL[i],lw=2.7,label=LABELS[i]) for i in indices],loc='upper center',bbox_to_anchor=(.51,.886),ncol=len(indices),frameon=False,fontsize=10)
def line(ax,d,key,i):
 x=d.epoch;y=d[key];ax.plot(x,y,color=COL[i],alpha=.13,lw=.5,rasterized=True)
 ax.plot(x,y.rolling(51,center=True,min_periods=1).mean(),color=COL[i],lw=2.3)
 ax.set_xlim(0,10000)
# 1: opposing-gradient schematic and measured curves; no equations
fig=plt.figure(figsize=(14,8.5))
grid=fig.add_gridspec(2,2,left=.065,right=.955,bottom=.13,top=.79,
                     width_ratios=[.85,1.4],hspace=.42,wspace=.32)
heading(fig,'Data and wall constraints pull in opposing directions',
        'Quasi-steady V4 | PointNet | seed 1234 | parameter-gradient diagnostics',
        'Curves: 151-sample rolling median. Bands: 10th–90th percentiles within 100-epoch bins; not confidence intervals.')
legends(fig,[1,2,3])
ax=fig.add_subplot(grid[:,0]);ax.set_aspect('equal');ax.axis('off')
ax.set_xlim(-1.7,1.85);ax.set_ylim(-1.55,2.2)
ax.text(-1.6,2.03,'a  Opposing gradients',fontsize=13,weight='bold')
from matplotlib.patches import Circle, Arc
origin=np.array([0.,.35]);data_vec=np.array([1.4,0.]);wall_vec=1.28*np.array([-.9,np.sqrt(1-.9**2)])
ax.add_patch(Circle(origin,.46,fill=False,edgecolor='#D1D9E1',lw=1.3,ls=':'))
for vec,color in [(data_vec,COL[0]),(wall_vec,COL[1])]:
 ax.annotate('',xy=origin+vec,xytext=origin,
             arrowprops=dict(arrowstyle='-|>',color=color,lw=3.2,mutation_scale=23))
ax.scatter(*origin,s=24,color=INK,zorder=5)
ax.add_patch(Arc(origin,.72,.72,theta1=0,theta2=np.degrees(np.arccos(-.9)),color=GRAY,lw=1.1))
ax.text(.43,.94,'Negative cosine',fontsize=11,color=GRAY,ha='center')
ax.text(1.02,.09,'Data gradient',ha='center',va='top',color=COL[0],fontsize=12,weight='bold')
ax.text(-1.12,1.35,'No-slip gradient',ha='center',color=COL[1],fontsize=12,weight='bold')
ax.text(-1.12,1.13,'Wall constraint',ha='center',color=GRAY,fontsize=10)
ax.text(0,-.7,'Competing optimization directions',ha='center',fontsize=11,color=INK)
ax.text(0,-1.04,'Schematic in parameter space',ha='center',fontsize=10,color=GRAY)
for ax,term,title in [(fig.add_subplot(grid[0,1]),'no_slip','b  Data vs. no-slip'),
                       (fig.add_subplot(grid[1,1]),'momentum','c  Data vs. momentum')]:
 style(ax);ax.axhspan(-1,0,color='#F8ECE9',alpha=.55);ax.axhline(0,color=GRAY,lw=.8,ls='--')
 for i,k in enumerate(KEYS[1:],1):
  if term=='momentum' and k=='BC': continue
  d=grad[k];y=d['cos_grad_data_'+term];group=d.assign(y=y,bin=(d.epoch//100)*100).groupby('bin')
  ax.fill_between(group.epoch.mean(),group.y.quantile(.1),group.y.quantile(.9),color=COL[i],alpha=.10,lw=0)
  ax.plot(d.epoch,y.rolling(151,center=True,min_periods=1).median(),color=COL[i],lw=2)
 ax.set_ylim(-1.04,.4);ax.set_xlim(0,10000);ax.set_ylabel('Gradient cosine similarity');ax.set_xlabel('Epoch');ax.set_title(title,loc='left')
save(fig,'fig1_gradient_conflict_steady')
# 2: data fitting and weighted objective
fig,axs=plt.subplots(2,2,figsize=(14,9));fig.subplots_adjust(left=.08,right=.96,bottom=.18,top=.79,hspace=.36,wspace=.21)
heading(fig,'Stronger constraints leave larger data-fitting errors','Quasi-steady V4 | seed 1234 | common standardized data loss across all four arms','Faint lines: logged epoch means. Solid lines: 51-epoch moving mean. Labels: raw mean over the final 500 epochs.')
legends(fig,range(4))
for col,b in enumerate(['PN','PNPP']):
 for row,key in enumerate(['mean_data_total','mean_total']):
  ax=axs[row,col];style(ax,True)
  for i,k in enumerate(KEYS):
   d=curves[b,k];line(ax,d,key,i)
   if row==0:ax.text(10200,d[key].tail(500).mean(),f'{d[key].tail(500).mean():.3f}',color=COL[i],fontsize=10,va='center')
  ax.set_xlim(0,11200 if row==0 else 10000);ax.set_ylim((.15,1.2) if row==0 else (.15,2.3))
  ax.set_title(f'{chr(97+row*2+col)}  '+('PointNet' if b=='PN' else 'PointNet++')+(' | data loss' if row==0 else ' | weighted total'),loc='left')
  ax.set_yticks([.2,.4,.6,1] if row==0 else [.2,.5,1,2]);ax.yaxis.set_major_formatter(FuncFormatter(lambda y,p:f'{y:g}'));ax.yaxis.set_minor_formatter(plt.NullFormatter());ax.set_ylabel('Standardized MSE' if row==0 else 'Training objective');ax.set_xlabel('Epoch')
fig.text(.08,.075,'Compare data loss across arms. Weighted totals contain different active terms and weights; they are not a common accuracy metric.',fontsize=10,color=GRAY)
save(fig,'fig2_training_losses_steady')
# 3: emphasize early saturation, its duration, and measured trade-offs
fig,axs=plt.subplots(2,2,figsize=(14,9))
fig.subplots_adjust(left=.085,right=.965,bottom=.16,top=.79,hspace=.43,wspace=.27)
heading(fig,'EMA weight reaches the cap early and stays there',
        'Quasi-steady V4 | seed 1234 | PointNet and PointNet++ show the same weight trajectory',
        'Weight curves: raw epoch means. Bars: raw means over each run’s final 500 epochs. No test-set accuracy is inferred.')
fig.legend(handles=[Line2D([],[],color=COL[3],lw=2.8,label='EMA | PointNet'),
                    Line2D([],[],color=COL[3],marker='o',mfc='white',ls='',label='EMA | PointNet++'),
                    Line2D([],[],color=COL[2],lw=2.5,ls='--',label='Fixed PDE weight')],
           loc='upper center',bbox_to_anchor=(.52,.885),ncol=3,frameon=False,fontsize=11)
for col,limit in enumerate([50,10000]):
 ax=axs[0,col];style(ax);ax.set_xlim(0,limit);ax.set_ylim(0,11.4)
 ax.axhspan(9.99,11.4,color=COL[3],alpha=.08)
 d=curves['PN','BC-PDE-EMA'];ax.plot(d.epoch,d.mean_lambda_phy,color=COL[3],lw=3,zorder=4)
 d=curves['PNPP','BC-PDE-EMA'];q=d[d.epoch<=limit].iloc[::(5 if col==0 else 1000)]
 ax.plot(q.epoch,q.mean_lambda_phy,ls='',marker='o',mfc='white',mec=COL[3],ms=5,zorder=5)
 ax.axhline(1,color=COL[2],lw=2,ls='--');ax.axhline(10,color=COL[3],lw=.8,ls=':')
 ax.set_yticks([0,1,5,10]);ax.set_ylabel('PDE loss weight');ax.set_xlabel('Epoch')
 if col==0:
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
  ax.set_title('a  Early rise | first 50 epochs',loc='left')
  ax.annotate('Epoch 24: weight reaches 9.99',xy=(24,9.99),xytext=(9,6.0),
              fontsize=11,color=COL[3],arrowprops=dict(arrowstyle='->',color=COL[3],lw=1.3))
  ax.text(34,10.65,'Cap = 10',color=COL[3],fontsize=10)
 else:
  ax.set_title('b  Saturation | full training',loc='left')
  ax.text(4800,6.8,'99.76%',ha='center',fontsize=29,weight='bold',color=COL[3])
  ax.text(4800,4.3,'of logged epochs at weight ≥ 9.99\nBoth backbones',ha='center',fontsize=11,color=GRAY,linespacing=1.5)
  ax.text(7200,10.6,'Cap = 10',color=COL[3],fontsize=10)
ax=axs[1,0];style(ax);ax.grid(False,axis='y');ax.grid(axis='x',color='#E8EDF2')
rows=[('PN','Continuity'),('PNPP','Continuity'),('PN','Momentum'),('PNPP','Momentum')]
for j,(b,t) in enumerate(rows):
 key='mean_continuity_raw' if t=='Continuity' else 'mean_momentum_group_raw'
 fixed=curves[b,'BC-PDE-F'][key].tail(500).mean();ema=curves[b,'BC-PDE-EMA'][key].tail(500).mean();ratio=ema/fixed
 ax.barh(j,1,color=COL[2],alpha=.12,height=.57)
 ax.barh(j,ratio,color=COL[3],height=.57)
 ax.text(ratio+.025,j,f'{ratio:.3f}  ({1/ratio:.1f}× lower)',va='center',fontsize=10,color=COL[3])
ax.set_yticks(range(4));ax.set_yticklabels(['PN\nContinuity','PN++\nContinuity','PN\nMomentum','PN++\nMomentum'],fontsize=10)
ax.invert_yaxis();ax.set_xlim(0,1.13);ax.set_xticks([0,.5,1]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
ax.axvline(1,color=COL[2],ls='--',lw=1);ax.set_xlabel('Residual loss relative to fixed PDE (fixed = 1)')
ax.set_title('c  Lower PDE residuals with EMA',loc='left');ax.text(.97,1.035,'Fixed reference',transform=ax.transAxes,ha='right',fontsize=9,color=COL[2])
ax=axs[1,1];style(ax)
for j,b in enumerate(['PN','PNPP']):
 vals=[curves[b,k].mean_data_total.tail(500).mean() for k in ['BC-PDE-F','BC-PDE-EMA']]
 for dx,v,c in zip([-.18,.18],vals,[COL[2],COL[3]]):
  ax.bar(j+dx,v,width=.29,color=c);ax.text(j+dx,v+.02,f'{v:.3f}',ha='center',fontsize=11,color=c)
ax.set_xticks([0,1]);ax.set_xticklabels(['PointNet','PointNet++']);ax.xaxis.set_major_formatter(plt.FixedFormatter(['PointNet','PointNet++']))
ax.set_ylim(0,1.05);ax.set_ylabel('Standardized data MSE');ax.set_title('d  Higher data-fitting error with EMA',loc='left')
ax.legend(handles=[Line2D([],[],color=COL[2],lw=7,label='Fixed'),Line2D([],[],color=COL[3],lw=7,label='EMA')],frameon=False,ncol=2,loc='upper center',fontsize=10)
fig.text(.085,.075,'Weight saturation does not resolve the data–constraint trade-off.',fontsize=12,weight='bold',color=INK)
save(fig,'fig3_physics_dynamics_steady');pdf.close()
print(pd.DataFrame(stats).to_string(index=False));print(pd.DataFrame(gs).to_string(index=False))
