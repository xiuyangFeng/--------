from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
OUT=Path(__file__).resolve().parent/'single_pnpp_ema'
OUT.mkdir(exist_ok=True)
ROOT=OUT.parents[5]
RUN='V4-SP-PNPP-BC-PDE-EMA-s1234'
SRC=ROOT/'outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak'/RUN/'epoch_progress.jsonl'
d=pd.read_json(SRC,lines=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#243549','axes.labelcolor':'#243549','axes.edgecolor':'#CED7E0','xtick.color':'#617184','ytick.color':'#617184','pdf.fonttype':42,'svg.fonttype':'none'})
specs=[('01_uvwp','Data loss by output channel',[(f'data_{x}',x,c,'-') for x,c in zip('uvwp',['#3174B5','#D87542','#258C80','#8661AB'])],'Standardized MSE'),('02_bc_pde','Boundary and PDE loss components',[('no_slip','Wall no-slip','#3174B5','-'),('inlet_bc','Inlet','#D87542','-'),('continuity','Continuity','#258C80','-'),('momentum_x','Momentum x','#8661AB','--'),('momentum_y','Momentum y','#BC5282','--'),('momentum_z','Momentum z','#9A8532','--')],'Dimensionless scaled MSE')]
with PdfPages(OUT/'single_pnpp_losses.pdf') as pdf:
 for name,title,terms,unit in specs:
  fig,ax=plt.subplots(figsize=(12,6.6));fig.subplots_adjust(left=.105,right=.965,bottom=.17,top=.73)
  fig.text(.075,.94,title,fontsize=22,weight='bold')
  fig.text(.075,.885,'Quasi-steady | PointNet++ | Data + BC + PDE (EMA) | seed 1234',fontsize=11,color='#617184')
  for key,label,c,ls in terms:
   y=d['mean_'+key+'_raw'];assert np.isfinite(y).all()
   ax.plot(d.epoch,y.where(y>0),color=c,lw=.5,alpha=.14,rasterized=True)
   ax.plot(d.epoch,y.rolling(51,center=True,min_periods=1).mean().where(lambda y:y>0),color=c,lw=2.3,ls=ls,label=label)
  ax.set_yscale('log');ax.set_xlim(0,10000);ax.set_xlabel('Epoch');ax.set_ylabel(unit)
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:g}k' if x else '0'))
  ax.grid(axis='y',color='#E8EDF2');ax.set_axisbelow(True)
  for sp in ['top','right']:ax.spines[sp].set_visible(False)
  fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.842),ncol=len(terms),frameon=False,fontsize=10.5)
  fig.text(.075,.045,'Unweighted losses | faint: logged epoch means; solid/dashed: 51-epoch moving mean | log scale',fontsize=10,color='#617184')
  for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=220,facecolor='white')
  pdf.savefig(fig);plt.close(fig)
fields=['mean_'+key+'_raw' for _,_,terms,_ in specs for key,*_ in terms]
d[['epoch']+fields].to_csv(OUT/'source_curves.csv',index=False)
(OUT/'README.md').write_text(f'''# 单个PointNet++稳态实验：两张损失图

按用户要求仅展示 `{RUN}`，选择EMA臂以对应此前动态权重讨论，未按测试成绩筛选。

- [u/v/w/p数据损失](01_uvwp.png)：同一坐标轴4条曲线。
- [边界与PDE分项](02_bc_pde.png)：同一坐标轴6条曲线，即no-slip、inlet、continuity、momentum x/y/z。动量本身属于PDE，不再将PDE总和作为额外独立项重复展示。
- [两页合订PDF](single_pnpp_losses.pdf)。各图另有PDF/SVG。

全部是未加权的逐epoch平均损失，51轮居中移动均值，淡线保留原始值。数据项是标准化MSE，物理项是缩放后无量纲MSE；曲线大小不能直接解释为加权总损失贡献或梯度强弱。只使用现有日志，没有新增实验。横轴从epoch0至9984，不外推至10000。

来源：`{SRC}`。字段与完整原始曲线见 `source_curves.csv`。复现脚本位于上一级 `plot_single_pnpp.py`。
''')
print(OUT)
