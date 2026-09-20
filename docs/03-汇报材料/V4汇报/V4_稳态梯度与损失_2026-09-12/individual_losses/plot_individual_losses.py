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
ROOT=OUT.parents[4]
BASE=ROOT/'outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak'
MODES=['DATA','BC','BC-PDE-F','BC-PDE-EMA']
LABELS=['Data','Data + BC','Data + BC + PDE (fixed)','Data + BC + PDE (EMA)']
COL=['#3174B5','#D87542','#258C80','#8661AB'];INK='#243549';GRAY='#617184'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#CBD4DD','xtick.color':GRAY,'ytick.color':GRAY,'pdf.fonttype':42,'svg.fonttype':'none'})
curves={};sources=[];checks=[]
for b in ['PN','PNPP']:
 for m in MODES:
  run=f'V4-SP-{b}-{m}-s1234';p=BASE/run/'epoch_progress.jsonl';d=pd.read_json(p,lines=True)
  assert d.epoch.is_unique and np.all(np.diff(d.epoch)==1)
  parts=['data_u','data_v','data_w','data_p','no_slip','inlet_bc','continuity','momentum_x','momentum_y','momentum_z']
  summed=sum(d['mean_'+k+'_weighted'] for k in parts)
  err=np.max(np.abs(summed-d.mean_total));assert np.allclose(summed,d.mean_total,rtol=1e-5,atol=1e-6)
  checks.append({'run':run,'max_absolute_sum_error':float(err),'n_epochs':len(d)})
  curves[b,m]=d;sources.append(str(p))
items=[('data_u','Velocity u',0,'Standardized MSE'),('data_v','Velocity v',0,'Standardized MSE'),('data_w','Velocity w',0,'Standardized MSE'),('data_p','Pressure p',0,'Standardized MSE'),('no_slip','Wall no-slip',1,'Dimensionless scaled MSE'),('inlet_bc','Inlet boundary',1,'Dimensionless scaled MSE'),('continuity','Continuity',2,'Dimensionless scaled MSE'),('momentum_x','Momentum x',2,'Dimensionless scaled MSE'),('momentum_y','Momentum y',2,'Dimensionless scaled MSE'),('momentum_z','Momentum z',2,'Dimensionless scaled MSE'),('momentum_group','Momentum group',2,'Dimensionless scaled MSE')]
rows=[]
with PdfPages(OUT/'individual_losses_all.pdf') as pdf:
 for num,(key,title,start,unit) in enumerate(items,1):
  fig,axes=plt.subplots(2,2,figsize=(13,8.4),sharex=True)
  fig.subplots_adjust(left=.095,right=.965,bottom=.14,top=.78,hspace=.4,wspace=.23)
  fig.text(.07,.947,f'{title} | individual loss history',fontsize=21,weight='bold')
  fig.text(.07,.905,'Quasi-steady V4 | seed 1234 | raw loss and its weighted contribution',fontsize=11,color=GRAY)
  fig.legend(handles=[Line2D([],[],color=COL[i],lw=2.5,label=LABELS[i]) for i in range(start,4)],loc='upper center',bbox_to_anchor=(.52,.87),ncol=4-start,frameon=False,fontsize=10)
  for r,suffix in enumerate(['raw','weighted']):
   limits=[]
   for c,b in enumerate(['PN','PNPP']):
    ax=axes[r,c];field='mean_'+key+'_'+suffix
    for i in range(start,4):
     d=curves[b,MODES[i]];y=d[field]
     assert np.isfinite(y).all() and (y>=0).all()
     ax.plot(d.epoch,y.where(y>0),color=COL[i],alpha=.15,lw=.5,rasterized=True)
     sm=y.rolling(51,center=True,min_periods=1).mean()
     ax.plot(d.epoch,sm.where(sm>0),color=COL[i],lw=2.2)
     rows.append({'term':key,'view':suffix,'run':f'V4-SP-{b}-{MODES[i]}-s1234','field':field,'final500_mean':float(y.tail(500).mean()),'zero_epochs':int((y==0).sum())})
    ax.set_yscale('log');ax.set_xlim(0,10000);ax.grid(axis='y',color='#E8EDF2');ax.set_axisbelow(True)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:g}k' if x else '0'))
    ax.set_title(f'{chr(97+r*2+c)}  '+('PointNet' if b=='PN' else 'PointNet++')+' | '+('raw' if r==0 else 'weighted'),loc='left',fontsize=12,weight='bold')
    ax.set_ylabel(unit if r==0 else 'Contribution to total loss');ax.set_xlabel('Epoch');limits.append(ax.get_ylim())
   for ax in axes[r]:ax.set_ylim(min(t[0] for t in limits),max(t[1] for t in limits))
  fig.text(.07,.059,'Faint: raw epoch means. Solid: 51-epoch moving mean. Inactive terms omitted; zero values masked on log axes.',fontsize=9,color=GRAY)
  fig.text(.07,.025,'Weighted curves use logged step-weighted means, including the time-varying EMA weight.',fontsize=9,color=GRAY)
  name=f'{num:02d}_{key}'
  for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=200,facecolor='white')
  pdf.savefig(fig);plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'individual_loss_summary.csv',index=False)
(OUT/'verification.json').write_text(json.dumps({'sources':sources,'total_loss_reconstruction':checks},indent=2))
print('Generated 11 figures and combined PDF; total loss reconstruction passed for all 8 runs.')
