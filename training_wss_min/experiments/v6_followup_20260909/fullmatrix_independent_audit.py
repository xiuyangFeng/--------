"""Read-only complete-matrix audit. Output JSON to stdout; caller writes only /tmp."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,statistics,math
import torch
from training_wss_min.tools import report_v6_followup as R
state=json.loads((R.EXP/'queue_status.json').read_text())
integrity=R.source_integrity(state)
matrix=json.loads(R.MATRIX.read_text());arms=matrix['arms']
paths={'r2':'field_casebalanced.r2','log':'normalized.field_casebalanced.r2','mean':'aggregate.r2_casemean','p10':'aggregate.r2_casep10','neg':'aggregate.r2_negative_cases','mae':'field.mae','mae_cb':'field_casebalanced.mae','rmse':'field.rmse','high':'regional_field.high_wss.r2','high_mae':'regional_field.high_wss.mae','top':'calibration.top10_pred_true_ratio','p99':'calibration.p99_pred_true_ratio','max_ratio':'calibration.max_pred_true_ratio','iou':'hotspot.top10_iou_casemean','sp':'hotspot.spearman_all_casemean','high_sp':'hotspot.spearman_high_wss_casemean','peak':'hotspot.peak_point_dist_over_bbox_casemean','centroid':'hotspot.hotspot_centroid_dist_over_bbox_casemean','fit':'field_casebalanced.r2_linear_fit','a':'field_casebalanced.linear_fit_slope','b':'field_casebalanced.linear_fit_intercept'}
def at(m,path):
 for k in path.split('.'):m=m[k]
 return float(m)
def values(m):return {k:at(m,p) for k,p in paths.items()}
def comparison(m,base):
 assert set(m['per_case'])==set(base['per_case'])
 case={}
 for c,b in base['per_case'].items():
  a=m['per_case'][c]
  case[c]={'r2':a['overall']['r2'],'base_r2':b['overall']['r2'],'delta_r2':a['overall']['r2']-b['overall']['r2']}
  for short,key in [('iou','top10_iou'),('sp','spearman_all'),('high_sp','spearman_high_wss'),('high_mae','high_wss_mae'),('peak','peak_point_dist_over_bbox'),('centroid','hotspot_centroid_dist_over_bbox')]:
   case[c][short]=a['hotspot'][key];case[c]['base_'+short]=b['hotspot'][key];case[c]['delta_'+short]=a['hotspot'][key]-b['hotspot'][key]
  for short,key in [('top','top10_pred_true_ratio'),('p99','p99_pred_true_ratio'),('max_ratio','max_pred_true_ratio')]:
   case[c][short]=a['calibration'][key];case[c]['base_'+short]=b['calibration'][key]
   case[c]['delta_'+short]=a['calibration'][key]-b['calibration'][key]
   case[c][short+'_closeness_improvement']=abs(b['calibration'][key]-1)-abs(a['calibration'][key]-1)
 rank=sorted(case,key=lambda c:case[c]['delta_r2'])
 return {'delta':{k:at(m,p)-at(base,p) for k,p in paths.items()},'r2_cases_improved':sum(v['delta_r2']>0 for v in case.values()),'r2_case_delta_median':statistics.median(v['delta_r2'] for v in case.values()),'iou_cases_improved':sum(v['delta_iou']>0 for v in case.values()),'spearman_cases_improved':sum(v['delta_sp']>0 for v in case.values()),'high_mae_cases_improved':sum(v['delta_high_mae']<0 for v in case.values()),'top_ratio_cases_closer_to_one':sum(v['top_closeness_improvement']>0 for v in case.values()),'p99_ratio_cases_closer_to_one':sum(v['p99_closeness_improvement']>0 for v in case.values()),'largest_drops':{c:case[c] for c in rank[:3]},'largest_gains':{c:case[c] for c in rank[-3:]},'group_delta':{g:v['r2']-base['group_casebalanced'][g]['r2'] for g,v in m['group_casebalanced'].items()},'per_case':case}
def artifacts(name,ck):
 d=R.RUNS/name;c=torch.load(d/f'ckpt_{ck}.pt',map_location='cpu',weights_only=True)
 files={'metric':d/'eval'/f'ckpt_{ck}'/'metrics.json','checkpoint':d/f'ckpt_{ck}.pt','run_config':d/'config.json'}
 return {'checkpoint_epoch':c['epoch'],'checkpoint_cfg_name':c.get('cfg_name'),'parameters':sum(v.numel() for k,v in c['model'].items() if not (k.endswith('running_mean') or k.endswith('running_var') or k.endswith('num_batches_tracked'))),'files':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in files.items()}}
out={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'audit_script':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'job_id':state['job_id'],'queue_status':state['status'],'integrity':integrity,'finalization':R.matrix_finalization(state,[a['id'] for a in arms],integrity),'metric_paths':paths,'metric_definitions':R.METRIC_DEFINITIONS,'scope':'All comparisons exploratory single seed1234, exposed test34, legacy vertex metrics. No uncertainty or stability claims. No training/report/workbook writes.','checkpoints':{}}
for ck in ['best','last']:
 base=R.load_metrics(R.ANCHOR,ck);loaded={'A5':base};result={'A5':{'values':values(base),'artifacts':artifacts(R.ANCHOR,ck)},'arms':{}}
 for arm in arms:
  m,e=R.run_evidence(arm,ck,state,{},integrity)
  entry={'evidence':e,'run_name':arm['run_name'],'parent':arm.get('baseline','B0'),'hypothesis':arm['hypothesis'],'changes_vs_A5':arm['changes'],'changes_vs_parent':arm['changes_vs_parent'],'stages':state['arms'][arm['id']]['stages']}
  if e['valid_scientific_result']:
   loaded[arm['id']]=m;entry.update(values=values(m),artifacts=artifacts(arm['run_name'],ck),vs_A5=comparison(m,base))
  result['arms'][arm['id']]=entry
 for arm in arms:
  entry=result['arms'][arm['id']];p=entry['parent'];p='A5' if p in ['B0','A5','B0=A5'] else p
  if arm['id'] in loaded and p in loaded:entry['vs_parent']=comparison(loaded[arm['id']],loaded[p])
 if all(k in loaded for k in ['A5','L1','L2','L3']):
  v={k:values(loaded[k]) for k in ['A5','L1','L2','L3']}
  factors={'pinball_without_huber':('A5','L1'),'pinball_with_huber':('L2','L3'),'huber_with_pinball':('L2','A5'),'huber_without_pinball':('L3','L1')}
  result['loss_factorial']={'cells':{'none':'L1','pinball':'A5','huber':'L3','both':'L2'},'effects':{k:{'child':child,'parent':parent,**comparison(loaded[child],loaded[parent])} for k,(child,parent) in factors.items()},'interaction_raw_metric':{k:v['L2'][k]-v['A5'][k]-v['L3'][k]+v['L1'][k] for k in paths},'interaction_explanation':'(L2-A5)-(L3-L1) = effect of Huber with pinball minus effect without pinball. Descriptive arithmetic, no significance inference. Errors/distances lower better; ratios judged by distance to 1.','interaction_ratio_closeness':{k:(-abs(v['L2'][k]-1)+abs(v['A5'][k]-1))-(-abs(v['L3'][k]-1)+abs(v['L1'][k]-1)) for k in ['top','p99']}}
  result['loss_factorial']['per_case_r2_interaction']={c:loaded['L2']['per_case'][c]['overall']['r2']-loaded['A5']['per_case'][c]['overall']['r2']-loaded['L3']['per_case'][c]['overall']['r2']+loaded['L1']['per_case'][c]['overall']['r2'] for c in base['per_case']}
 result['model_edges']={}
 for child,parent,meaning in [('M1','A5','Near-matched parameter pointwise control at same fusion point; tests neighborhood branch package'),('M5','A5','Three equal middle radii versus three distinct radii; same parameter and neighbor budgets'),('M2','A5','Independent query training with fixed3NN versus SAME training'),('M3','M2','Query interpolation16NN versus3NN; encoder FP unchanged'),('M4','M3','Complete trainable local decoder+geometry+residual package versus fixed16NN'),('M4','M2','Complete trainable decoder package versus independent fixed3NN')]:
  if child in loaded and parent in loaded:result['model_edges'][child+'_vs_'+parent]={'meaning':meaning,**comparison(loaded[child],loaded[parent])}
 result['completed_count']=len(loaded)-1
 result['casebalanced_r2_contributions']={}
 for child,parent in [('M2','A5'),('M4','A5'),('M4','M3'),('L2','A5')]:
  if child not in loaded or parent not in loaded:continue
  m,b=loaded[child],loaded[parent]
  denominator=b['field_casebalanced']['rmse']**2/(1-b['field_casebalanced']['r2'])
  count=len(b['per_case'])
  contribution={c:(bc['overall']['rmse']**2-m['per_case'][c]['overall']['rmse']**2)/(count*denominator) for c,bc in b['per_case'].items()}
  delta=m['field_casebalanced']['r2']-b['field_casebalanced']['r2']
  assert abs(sum(contribution.values())-delta)<1e-12
  ranked=sorted(contribution,key=contribution.get)
  result['casebalanced_r2_contributions'][child+'_vs_'+parent]={'definition':'Each case contribution=(parent case MSE-child case MSE)/(34*shared casebalanced target variance). Contributions sum exactly to delta physical R2_cb.','total_delta':delta,'sum_verified':True,'per_case':contribution,'largest_costs':{c:contribution[c] for c in ranked[:3]},'largest_benefits':{c:contribution[c] for c in ranked[-3:]}}
 out['checkpoints'][ck]=result
key_map={'r2':'physical_r2_cb','log':'normalized_r2_cb','mean':'case_mean','mae':'mae','high':'high_wss_r2','top':'top10_ratio','p99':'p99_ratio','iou':'top10_iou','sp':'spearman'}
summary_checks={}
for ck,p in out['checkpoints'].items():
 summary_path=R.EXP/f'matrix_summary_{ck}.json'
 summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
 summary_checks[ck]=False
 for aid,a in p['arms'].items():
  if 'values' not in a:continue
  met=json.loads(Path(a['artifacts']['files']['metric']['path']).read_text())['test']
  assert a['artifacts']['parameters']==met['efficiency']['parameters'],(ck,aid)
 if summary.get('completed_count')==15 and p['completed_count']==15:
  for aid,a in p['arms'].items():
   for key,skey in key_map.items():assert abs(a['values'][key]-summary['arms'][aid][skey])<1e-12,(ck,aid,key)
   assert abs(a['vs_parent']['delta']['r2']-summary['arms'][aid]['delta_parent_physical'])<1e-12,(ck,aid,'parent')
  summary_checks[ck]=True
out['independent_crosscheck']={'all_30_evaluations_valid':all(p['completed_count']==15 for p in out['checkpoints'].values()),'parameters_match_evaluation_efficiency':True,'saved_finalizer_summary_metrics_and_parent_deltas_match':summary_checks,'casebalanced_r2_contribution_sums_match':True,'case_mse_contribution_note':'Postprocessing only; does not alter evaluation metrics or select checkpoints.'}
print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
