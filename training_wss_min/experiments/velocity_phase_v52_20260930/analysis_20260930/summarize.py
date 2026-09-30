"""Summarize existing small JSON outputs only; no training/inference or raw fields."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MATRIX = ROOT / 'training_wss_min/configs/velocity_phase_v52_20260930/matrix.json'
m = json.loads(MATRIX.read_text())
paths = {a: Path(json.loads(Path(p).read_text())['out_dir']) for a, p in m['baseline_configs'].items()}
paths.update({a['arm']: Path(a['out_dir']) for a in m['arms']})
metrics = {a: json.loads((p/'metrics.json').read_text()) for a,p in paths.items()}
cases = {a: {r['unit_id']: r['tasks']['velocity'] for r in map(json.loads, (p/'per_case.jsonl').read_text().splitlines())} for a,p in paths.items()}
phases = ['cycle', 'peak', 'trough', 'decel', 'early_trough', 'late_trough']
records, history, source, aggregation_errors = [], {}, {}, []
for a,p in paths.items():
    d = metrics[a]
    v = d['tasks']['velocity']
    assert len(cases[a]) == 55 and set(cases[a]) == set(cases['U0'])
    for s in phases:
        if s not in v:
            continue
        x = v[s]
        vals = np.array([c[s]['r2'] for c in cases[a].values()])
        aggregation_errors.append(abs(float(vals.mean())-x['r2']))
        assert aggregation_errors[-1] < 1e-12
        decomp = v.get('error_decomposition', {}).get(s, {})
        r = dict(arm=a, phase=s, r2=x['r2'], component_mae_m_s=x['mae'],
                 component_rmse_m_s=x['rmse'], vector_rmse_m_s=x['rmse']*np.sqrt(3),
                 direction_cosine=x['direction_cosine'], direction_coverage=x['direction_weight_coverage'],
                 r2_median=float(np.median(vals)), r2_p10=float(np.quantile(vals,.1)),
                 negative_r2_count=int((vals<0).sum()), n_units=55,
                 speed_r2=v['speed'][s]['r2'], speed_rmse_m_s=v['speed'][s]['rmse'])
        # Exact linear algebraic decomposition using stored component and speed MSE.
        # This reproduces the existing diagnostic without reading predictions.
        r['vector_mse_m2_s2'] = 3*x['mse']
        r['speed_mse_m2_s2'] = v['speed'][s]['mse']
        r['direction_coupled_mse_m2_s2'] = r['vector_mse_m2_s2']-r['speed_mse_m2_s2']
        r['direction_energy_share'] = r['direction_coupled_mse_m2_s2']/r['vector_mse_m2_s2']
        if decomp:
            assert np.isclose(r['direction_coupled_mse_m2_s2'], decomp['direction_coupled_mse_m2_s2'])
        r['speed_bias_m_s'] = decomp.get('speed_bias_m_s')
        r['prediction_near_zero_fraction'] = decomp.get('prediction_below_0p001_given_truth_above_floor')
        r.update({'anatomy_'+k: value for k,value in v.get('anatomy',{}).get(s,{}).items()})
        r.update({'tail_'+k: value for k,value in v.get('train_threshold_tail',{}).get(s,{}).items()})
        for prefix, family in [('anatomy','anatomy'),('tail','train_threshold_tail')]:
            for k in v.get(family,{}).get(s,{}):
                r[prefix+'_'+k+'_n_units']=sum(c[family][s].get(k) is not None for c in cases[a].values())
        records.append(r)
    hh = list(map(json.loads,(p/'history.jsonl').read_text().splitlines()))
    def base(row):
        return row.get('objective',{}).get('z_mse', row['losses']['velocity'])
    history[a] = {'epochs':len(hh), 'last_epoch':hh[-1]['epoch'], 'last_step':hh[-1]['step'],
                  'first10_z_mse_mean':float(np.mean([base(h) for h in hh[:10]])),
                  'last10_z_mse_mean':float(np.mean([base(h) for h in hh[-10:]])),
                  'last_z_mse':base(hh[-1]), 'last_phase_z_mse':hh[-1].get('phase_z_mse'),
                  'max_clip_fraction':max(h.get('gradient_clip_fraction',0) for h in hh),
                  'parameter_count':d['parameter_count'], 'train_seconds':d['train_seconds'],
                  'peak_cuda_memory_bytes':d['peak_cuda_memory_bytes'],
                  'inference_median_s':d['inference']['median_s'], 'inference_p95_s':d['inference']['p95_s'],
                  'inference_ratio_to_U0':d['inference']['median_s']/metrics['U0']['inference']['median_s']}
    if a=='D1':
        ratios=[h['gradient_diagnostics']['weighted_direction_to_base'] for h in hh]
        history[a]['direction_gradient_ratio_epoch_first_batch']={'median':float(np.median(ratios)),'min':min(ratios),'max':max(ratios),'last':ratios[-1]}
        history[a]['empty_case_phases_all_epochs']=sum(h['direction_coverage']['empty_case_phases'] for h in hh)
        history[a]['last_direction_coverage']=hh[-1]['direction_coverage']
    source[a]={'run':str(p),'metrics_sha256':hashlib.sha256((p/'metrics.json').read_bytes()).hexdigest(),
               'per_case_sha256':hashlib.sha256((p/'per_case.jsonl').read_bytes()).hexdigest(),
               'history_sha256':hashlib.sha256((p/'history.jsonl').read_bytes()).hexdigest()}

pairs = [('U0','Iu')]+[(a['arm'],c) for a in m['arms'] for c in a['primary_controls']]
pairs += [(a,'U0') for a in ['A1','G00','G01','G10','G11']]
paired, gates = [], []
for arm,control in pairs:
    av,bv=metrics[arm]['tasks']['velocity'],metrics[control]['tasks']['velocity']
    for s in phases:
        if s not in av or s not in bv: continue
        ids=sorted(cases[arm])
        aa=np.array([cases[arm][i][s]['r2'] for i in ids]);bb=np.array([cases[control][i][s]['r2']for i in ids]);dd=aa-bb
        paired.append(dict(arm=arm,control=control,phase=s,n_units=55,delta_r2_mean=float(dd.mean()),
                           delta_r2_median=float(np.median(dd)),delta_r2_p10=float(np.quantile(dd,.1)),
                           improved_count=int((dd>0).sum()),improved_fraction=float((dd>0).mean()),
                           negative_r2_count=int((aa<0).sum()),control_negative_r2_count=int((bb<0).sum()),
                           delta_mae=av[s]['mae']-bv[s]['mae'],delta_vector_rmse_m_s=np.sqrt(3)*(av[s]['rmse']-bv[s]['rmse']),
                           delta_direction_cosine=av[s]['direction_cosine']-bv[s]['direction_cosine'],
                           worst_delta_unit=ids[int(dd.argmin())],worst_delta=float(dd.min()),
                           best_delta_unit=ids[int(dd.argmax())],best_delta=float(dd.max())))
    gate={}
    for s,min_delta in [('trough',.02),('decel',.01),('peak',-.005),('cycle',-.005)]:
        gate[s+'_r2']=av[s]['r2']-bv[s]['r2']>=min_delta
    for s in ['trough','decel']:
        gate[s+'_rmse']=av[s]['rmse']<=bv[s]['rmse']
        gate[s+'_direction']=av[s]['direction_cosine']>=bv[s]['direction_cosine']
    gates.append(dict(arm=arm,control=control,primary=(control in next((x['primary_controls']for x in m['arms']if x['arm']==arm),[])),
                      passed=all(gate.values()),checks=gate,failed=[k for k,v in gate.items()if not v],
                      inference_cost_preference_pass=history[arm]['inference_ratio_to_U0']<=1.5))

interaction={}
for s in ['cycle','peak','trough','decel']:
    interaction[s]={}
    for key in ['r2','mae','rmse','direction_cosine']:
        vv={a:metrics[a]['tasks']['velocity'][s][key]for a in ['G00','G01','G10','G11']}
        interaction[s][key]={'phase_without_graph':vv['G01']-vv['G00'], 'graph_without_phase':vv['G10']-vv['G00'],
                            'phase_with_graph':vv['G11']-vv['G10'], 'graph_with_phase':vv['G11']-vv['G01'],
                            'interaction':vv['G11']-vv['G10']-vv['G01']+vv['G00']}

def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w')as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
csvout('phase_metrics.csv',records);csvout('paired_metrics.csv',paired)
csvout('cost_and_training.csv',[dict(arm=a,**{k:v for k,v in h.items()if not isinstance(v,dict)})for a,h in history.items()])
out={'scope':'existing metrics/per_case/history JSON only; no raw fields, training or inference',
     'aggregation':'equal mean of 55 data-unit metrics; within-unit cell-volume and phase weighting; R2 is mean of unit field R2',
     'phases':m['phase_windows'],'source':source,'phase_metrics':records,'paired_metrics':paired,
     'aggregation_audit':{'r2_max_absolute_difference_metrics_vs_case_mean':max(aggregation_errors),'absolute_tolerance':1e-12},
     'screening_gates':gates,'graph_2x2':interaction,'training_cost':history,
     'decomposition_note':'vector MSE = 3 * stored component MSE; speed MSE = stored speed.mse; direction-coupled MSE = difference; ratio of mean energies, not pure angular error',
     'missing_baseline_diagnostics':'No early/late window R2, anatomy, train-q80 tail or speed bias filled where absent from original metrics. Per-frame R2 averages do not replace window R2.'}
(HERE/'analysis.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')

def table(head,rows):
    return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|'for r in rows])
rr={(r['arm'],r['phase']):r for r in records}
lines=['# 速度优化：8 臂最终结果矩阵（2026-09-30）','',
       '固定 fold0 / seed1234 / 206 训练、55 开发留出 / 150 epochs、7800 steps / last / 80 相位。所有数值从既有正式 metrics、per_case、history 汇总；未启动训练或新推理。',
       '', 'R² 是 55 个数据单元的体积加权场 R² 的算术均值，非 pooled R²。MAE 是 XYZ 分量绝对误差均值；向量 RMSE = √3 × 分量 RMSE，单位 m/s。峰 17–26、谷 5–9 及 43–57、减速 27–42（0-based）。',
       '', '## 主精度矩阵','']
lines += [table(['臂','周期 R²','峰 R²','谷 R²','减速 R²','谷 MAE','谷向量 RMSE','谷 cos','减速 MAE','减速向量 RMSE','减速 cos'],
 [[a]+[f'{rr[a,s]["r2"]:.6f}'for s in ['cycle','peak','trough','decel']]+[f'{rr[a,s][k]:.6f}'for s in ['trough','decel']for k in ['component_mae_m_s','vector_rmse_m_s','direction_cosine']]for a in paths])]
lines += ['', '## 全窗口 MAE / 向量 RMSE（m/s）','',table(['臂']+[s+' MAE / RMSE'for s in ['cycle','peak','trough','decel']],
 [[a]+[f'{rr[a,s]["component_mae_m_s"]:.6f} / {rr[a,s]["vector_rmse_m_s"]:.6f}'for s in ['cycle','peak','trough','decel']]for a in paths])]
lines += ['', '## 预注册开发门（所有条件同时满足）','', '固定要求：谷 R² ≥ +0.02，减速 ≥ +0.01；两窗向量 RMSE 不增、cos 不降；峰/周期 R² 下降 ≤ 0.005。1.5× U0 推理时间是成本偏好，不并入精度门。G11 必须分别对 G10、G01 判断。', '',
 table(['臂','主对照','精度门','未满足条件','成本≤1.5×'],[[g['arm'],g['control'],'通过'if g['passed']else'未通过',', '.join(g['failed'])or'—','是'if g['inference_cost_preference_pass']else'否']for g in gates if g['primary']])]
lines += ['', '## 单元分布与对 U0 的配对（谷 / 减速）','',table(['臂','谷改善单元','谷 ΔR² 中位数','谷 ΔR² P10','谷自身 R² P10','谷负 R²','减速改善单元','减速 ΔR² 中位数','减速自身 R² P10','减速负 R²'],
 [[a, f"{next(r for r in paired if r['arm']==a and r['control']=='U0'and r['phase']=='trough')['improved_count']}/55", f"{next(r for r in paired if r['arm']==a and r['control']=='U0'and r['phase']=='trough')['delta_r2_median']:+.6f}", f"{next(r for r in paired if r['arm']==a and r['control']=='U0'and r['phase']=='trough')['delta_r2_p10']:+.6f}",f"{rr[a,'trough']['r2_p10']:.6f}",rr[a,'trough']['negative_r2_count'],f"{next(r for r in paired if r['arm']==a and r['control']=='U0'and r['phase']=='decel')['improved_count']}/55",f"{next(r for r in paired if r['arm']==a and r['control']=='U0'and r['phase']=='decel')['delta_r2_median']:+.6f}",f"{rr[a,'decel']['r2_p10']:.6f}",rr[a,'decel']['negative_r2_count']]for a in paths if a not in ['Iu','U0']])]
lines += ['', '## 早谷 / 晚谷及尾部、解剖诊断','', 'Iu/U0 原 metrics 中没有早晚窗口 R²、训练 q80 尾部或解剖诊断，表中不补造；已有历史点诊断另见上一轮 analysis_velocity_20260930。方向份额为两项平均能量的比；方向耦合项同时依赖幅值。', '',
 table(['臂','早谷 R²','晚谷 R²','谷幅值 MSE','谷方向耦合 MSE','方向份额','谷轴向 RMSE','谷横向 RMSE','谷逆轴向 precision/recall','谷尾部 RMSE','谷尾部速率比','谷尾部 precision/recall'],
 [[a,f"{rr[a,'early_trough']['r2']:.6f}",f"{rr[a,'late_trough']['r2']:.6f}",f"{rr[a,'trough']['speed_mse_m2_s2']:.7f}",f"{rr[a,'trough']['direction_coupled_mse_m2_s2']:.7f}",f"{rr[a,'trough']['direction_energy_share']:.2%}",f"{rr[a,'trough']['anatomy_axial_rmse_m_s']:.6f}",f"{rr[a,'trough']['anatomy_transverse_vector_rmse_m_s']:.6f}",f"{rr[a,'trough']['anatomy_reverse_axial_precision']:.4f}/{rr[a,'trough']['anatomy_reverse_axial_recall']:.4f}",f"{rr[a,'trough']['tail_vector_rmse_m_s']:.6f}",f"{rr[a,'trough']['tail_pred_true_speed_ratio']:.4f}",f"{rr[a,'trough']['tail_tail_precision']:.4f}/{rr[a,'trough']['tail_tail_recall']:.4f}"]for a in paths if a not in ['Iu','U0']])]
lines += ['', '## 成本与训练','',table(['臂','参数','训练分钟','CUDA peak GiB','固定查询中位秒','P95 秒','相对 U0','最后10轮共享 z-MSE'],[[a,h['parameter_count'],f"{h['train_seconds']/60:.2f}",f"{h['peak_cuda_memory_bytes']/2**30:.3f}",f"{h['inference_median_s']:.4f}",f"{h['inference_p95_s']:.4f}",f"{h['inference_ratio_to_U0']:.3f}×",f"{h['last10_z_mse_mean']:.6f}"]for a,h in history.items()]),'',
 '时间包括已缓存数据读取、传输、几何编码和 16384 固定查询 × 80 相位解码，排除原始几何预处理及导出；不是完整临床端到端耗时，且来自各 run 的单次测量而非同期专门计时实验。训练 total loss 跨 D1/D2 不可直接比较，故列共同 z-MSE。','',
 '完整精度、所有窗口诊断与配对： [analysis.json](analysis.json)、[phase_metrics.csv](phase_metrics.csv)、[paired_metrics.csv](paired_metrics.csv)、[cost_and_training.csv](cost_and_training.csv)。分析解释见 [analysis.md](analysis.md)。']
(HERE/'results_matrix.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'gates':gates,'graph_interaction':interaction,'training':history},indent=2))
