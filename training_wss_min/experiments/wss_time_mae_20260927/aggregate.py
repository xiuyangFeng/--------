"""Aggregate already-completed temporal MAE evaluations; require all 39 fold rows."""
from pathlib import Path
import json,csv,statistics,hashlib
D=Path(__file__).resolve().parent
order=['Tnull','T0','TB8','TB16','P-mlp','P-mlp-qx','C-raw','TL-warm','TL-random','TT-warm','TT-warm-noattn','TT-raw','TT-raw-noattn']
labels=['T-null：峰值预测＋训练折时间统计','T0：几何＋相位查询，端到端训练','TB8：8维时间基系数','TB16：16维时间基系数','P-mlp：冻结隐藏特征＋时间MLP','P-mlp-qx：去掉病例级向量','C-raw：原始几何＋峰值锚定MLP','TL-warm：峰值权重初始化＋部分解冻','TL-random：随机初始化对照','TT-warm：隐藏特征＋跨帧注意力','TT-warm-noattn：隐藏特征＋对角注意力','TT-raw：原始几何＋跨帧注意力','TT-raw-noattn：原始几何＋对角注意力']
records=json.loads((D/'existing_T0_TB_metrics.json').read_text())['rows']
for p in sorted((D/'results').glob('*/*.json')):
 r=json.loads(p.read_text());assert r['validation']['passed'],p;r['result_path']=str(p);records.append(r)
keys=['cycle_r2cb_pa','trough_r2cb_pa','peak_r2cb_pa','tawss_r2cb_pa','cycle_mae_pa','trough_mae_pa','peak_mae_pa','tawss_mae_pa','cycle_mae_pooled_pa']
rows=[]
for arm,label in zip(order,labels):
 rr=sorted([r for r in records if r['arm']==arm],key=lambda r:r['fold'])
 assert [r['fold'] for r in rr]==[0,1,2],(arm,'incomplete')
 for r in rr:assert r['metrics']['n_cases'] in (44,46)
 mean={k:statistics.mean(r['metrics'][k] for r in rr) for k in keys}
 rows.append(dict(arm=arm,label=label,**mean,folds=rr))
maxdiff=max(abs(v) for r in records if 'validation' in r for v in r['validation']['differences'].values())
summary=dict(complete=True,n_methods=13,n_fold_rows=39,new_evaluation_fold_rows=30,existing_fold_rows=9,metric_contract='physical Pa; point mean within each case/frame, mean over cases and 81 frames, arithmetic mean over three folds; raw truth; original checkpoint and cached donor',max_original_metric_abs_difference=maxdiff,rows=rows)
(D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
header=['尝试','全周期 R² ↑','谷底 R² ↑','峰值 R² ↑','TAWSS R² ↑','全周期 MAE（Pa）↓']
view=[[r['label'],*[f'{r[k]:.4f}' for k in keys[:5]]] for r in rows]
with (D/'时间建模_MAE补齐.tsv').open('w') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(header);w.writerows(view)
md=['时间建模：全周期 MAE 补评估（2026-09-27）','','全部13种方法已齐。V5.1，seed1234，原三折留出集，每例81帧；不训练、不更改checkpoint。原始run、冻结代码、配置和缓存只读，结果存入本独立目录。','','| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']
md.extend('| '+' | '.join(row)+' |' for row in view)
md += ['', 'MAE与R²均为病例等权，最后对三折求算术平均。R²使用补评估复现值；T0/TB8/TB16沿用已有逐帧病例等权指标。与原run的R²及TAWSS MAE最大绝对差：'+f'{maxdiff:.3g}。', '', '旧表中T0/TB8/TB16的0.513/0.547/0.546来自点池化MAE，本表统一病例等权后为0.5242/0.5639/0.5626 Pa。这是汇总口径调整，不是重新训练后的性能变化。全周期MAE不是TAWSS MAE，时间表的TAWSS沿用历史81帧均值。', '', '完整逐折、逐帧与逐病例误差见summary.json及results/。CPU几何模型未通过复现容差的结果保留在failed_cpu/，不进入正式表。跨批次预算、选模不同，不能将该表解释为纯结构消融。']
(D/'时间建模_MAE补齐.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md));print('rows',len(records))
