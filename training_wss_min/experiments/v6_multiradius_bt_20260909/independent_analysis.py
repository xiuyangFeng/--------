#!/usr/bin/env python3
"""Read-only MS0-MS6 analysis from original artifacts; never infer or train.

Outputs only analysis artifacts beside this script. --mode final refuses to
write a final result unless the six new runs and all 12 evaluations are complete
with frozen sources, 400 epoch rows and the exact existing 34-case partition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
RUNS = ROOT / "training_wss_min/runs"
MATRIX = ROOT / "training_wss_min/configs/v6_multiradius_bt_20260909/matrix.json"
METRICS = {
    "physical_r2_cb": "field_casebalanced.r2",
    "log_z_r2_cb": "normalized.field_casebalanced.r2",
    "case_r2_mean": "aggregate.r2_casemean",
    "case_r2_p10": "aggregate.r2_casep10",
    "negative_r2_cases": "aggregate.r2_negative_cases",
    "mae_pa": "field.mae",
    "mae_pa_cb": "field_casebalanced.mae",
    "rmse_pa_cb": "field_casebalanced.rmse",
    "high_wss_r2": "regional_field.high_wss.r2",
    "high_wss_mae_pa": "regional_field.high_wss.mae",
    "top10_iou": "hotspot.top10_iou_casemean",
    "spearman": "hotspot.spearman_all_casemean",
    "spearman_high_wss": "hotspot.spearman_high_wss_casemean",
    "top10_ratio": "calibration.top10_pred_true_ratio",
    "p99_ratio": "calibration.p99_pred_true_ratio",
    "peak_distance_bbox": "hotspot.peak_point_dist_over_bbox_casemean",
    "hotspot_centroid_distance_bbox": "hotspot.hotspot_centroid_dist_over_bbox_casemean",
}
CASE_METRICS = {
    "r2": "overall.r2", "mae_pa": "overall.mae", "rmse_pa": "overall.rmse",
    "iou": "hotspot.top10_iou", "spearman": "hotspot.spearman_all",
    "high_wss_mae_pa": "hotspot.high_wss_mae",
    "top10_ratio": "calibration.top10_pred_true_ratio",
    "p99_ratio": "calibration.p99_pred_true_ratio",
    "peak_distance_bbox": "hotspot.peak_point_dist_over_bbox",
}
CONTRASTS = [
    ("MS1", "MS0", "New-SA3 bridge: nearest ordering, cap32, centre condition and replacement initialization"),
    ("MS2", "MS1", "Single-scale global BT effect on new bridge"),
    ("MS3", "MS1", "Multi-scale sets plus independent capacity and fusion without BT"),
    ("MS4", "MS0", "Full teacher proposal versus original physical-score leader M2"),
    ("MS4", "MS1", "Full proposal versus new-SA3 bridge"),
    ("MS4", "MS2", "Multi-scale proposal versus single-scale BT"),
    ("MS4", "MS3", "Global BT package on multi-scale sets"),
    ("MS4", "MS5", "Different versus repeated radii with identical parameter count"),
    ("MS4", "MS6", "Global BT versus near-matched per-token FFN capacity"),
    ("MS6", "MS3", "Per-token capacity on multi-scale sets"),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atom(path):
    return {"path": str(path), "sha256": sha(path)}


def at(obj, path):
    for key in path.split("."):
        obj = obj[key]
    value = float(obj)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite metric: {path}")
    return value


def source_integrity(state):
    start, end = state.get("source_sha256", {}), state.get("source_sha256_end")
    changed = []
    for name, digest in start.items():
        path = ROOT / name
        if not path.exists() or sha(path) != digest:
            changed.append(name)
    return {"source_count": len(start), "current_mismatches": changed,
            "final_hashes_recorded": end is not None,
            "start_end_equal": bool(start) and end == start,
            "queue_reports_changed": state.get("source_changed"),
            "passed": bool(start) and not changed and end == start and state.get("source_changed") is False}


def history_evidence(run):
    path = run / "history.jsonl"
    rows, issues = [], []
    if not path.exists():
        return {"complete": False, "issues": ["history missing"], "row_count": 0}, []
    for n, line in enumerate(path.read_text().splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            issues.append(f"history malformed row {n + 1}")
    if [r.get("epoch") for r in rows] != list(range(400)):
        issues.append("history must contain exactly one row for each epoch0..399")
    if any(not math.isfinite(float(r.get("train_loss", math.nan))) for r in rows):
        issues.append("nonfinite train_loss")
    elapsed = [float(r["elapsed_s"]) for r in rows if "elapsed_s" in r]
    epoch_times = [elapsed[0]] + [b-a for a,b in zip(elapsed, elapsed[1:])] if elapsed else []
    return {"complete": not issues, "issues": issues, "row_count": len(rows),
            "last_epoch": rows[-1].get("epoch") if rows else None,
            "history_artifact": atom(path),
            "history_cumulative_elapsed_seconds": elapsed[-1] if elapsed else None,
            "median_epoch_seconds_after_epoch9": statistics.median(epoch_times[10:]) if len(epoch_times) > 10 else None,
            "mean_epoch_seconds": statistics.mean(epoch_times) if epoch_times else None}, rows


def checkpoint_evidence(run, checkpoint, history):
    path = run / f"ckpt_{checkpoint}.pt"
    if not path.exists():
        return {"valid": False, "issues": ["checkpoint missing"]}
    saved = torch.load(path, map_location="cpu", weights_only=True)
    weights = saved["model"]
    epoch = int(saved["epoch"])
    issues = []
    if checkpoint == "last" and epoch != 399:
        issues.append("last checkpoint epoch is not399")
    if not 0 <= epoch < 400:
        issues.append("checkpoint epoch outside0..399")
    if history and checkpoint == "best":
        losses = {int(r["epoch"]):float(r["train_loss"]) for r in history}
        if epoch not in losses or not math.isclose(losses[epoch], min(losses.values()), rel_tol=1e-10, abs_tol=1e-12):
            issues.append("best checkpoint does not match minimum recorded training loss")
    params = sum(v.numel() for k,v in weights.items() if not k.endswith(("running_mean", "running_var", "num_batches_tracked")))
    gammas = {k:float(v) for k,v in weights.items() if k.startswith("sa.2.") and "gamma" in k and v.numel() == 1}
    fuse = weights.get("sa.2.fuse.weight")
    fusion = None
    if fuse is not None:
        channels = fuse.shape[0]
        blocks = list(fuse.float().split(channels, dim=1))
        norms = [float(torch.linalg.vector_norm(b)) for b in blocks]
        total = sum(norms)
        initial = torch.zeros_like(fuse)
        mid = len(blocks)//2
        initial[:,mid*channels:(mid+1)*channels] = torch.eye(channels)
        fusion = {"shape":list(fuse.shape), "block_frobenius_norms":norms,
                  "block_norm_shares": [x/total for x in norms],
                  "distance_from_initial_middle_projection":float(torch.linalg.vector_norm(fuse-initial)),
                  "bias_norm":float(torch.linalg.vector_norm(weights['sa.2.fuse.bias'])),
                  "interpretation":"Projection norms indicate learned weights, not causal contribution or physical scale importance."}
    activated = {}
    for key,value in weights.items():
        if key.startswith('sa.2.') and ('.geo_pe.3.weight' in key or '.pos.3.weight' in key):
            activated[key] = float(torch.linalg.vector_norm(value.float()))
    return {"valid":not issues, "issues":issues, "epoch":epoch,
            "parameter_count_from_state":params, "artifact":atom(path),
            "new_module_gammas":gammas, "fusion":fusion,
            "initially_zero_geo_and_condition_weight_norms":activated}


def run_evidence(identifier, run_name, checkpoint, queue, expected_cases):
    run = RUNS/run_name
    history, rows = history_evidence(run)
    cfg_path = run/'config.json'
    issues = list(history['issues'])
    cfg = read(cfg_path) if cfg_path.exists() else None
    if cfg is None:
        issues.append('run config missing')
    elif (cfg['train']['seed'],cfg['train']['epochs'],cfg['data']['timesteps'],cfg['data']['target'],cfg['model']['out_dim']) != (1234,400,'peak','wss',1):
        issues.append('run protocol differs from seed1234/400/peak/scalarWSS')
    stage_evidence = queue.get('arms',{}).get('M2' if identifier == 'MS0' else identifier,{})
    for stage in ('train','eval_'+checkpoint):
        if stage_evidence.get('stages',{}).get(stage,{}).get('returncode') != 0:
            issues.append(stage+' stage not complete0')
    metric_path = run/'eval'/f'ckpt_{checkpoint}'/'metrics.json'
    metric = None
    if metric_path.exists():
        try:
            metric = read(metric_path)['test']
            if set(metric['per_case']) != expected_cases or set(metric['normalized']['per_case']) != expected_cases:
                issues.append('metrics case identities differ from exact test34')
            if metric.get('surface_metric_mode') != 'legacy_vertex' or metric.get('metric_space') != 'physical_target':
                issues.append('metric space or surface weighting differs')
            for path in METRICS.values():
                at(metric,path)
            for case in metric['per_case'].values():
                for path in CASE_METRICS.values():
                    at(case,path)
        except (KeyError, ValueError, TypeError) as error:
            issues.append('metric malformed: '+str(error))
    else:
        issues.append('metric missing')
    ckpt = checkpoint_evidence(run,checkpoint,rows) if not issues else {'valid':False,'issues':['checkpoint inspection deferred until run/eval prerequisites pass']}
    if not ckpt['valid']:
        issues += ckpt['issues']
    if metric is not None and ckpt['valid'] and int(metric['efficiency']['parameters']) != ckpt['parameter_count_from_state']:
        issues.append('checkpoint parameter count differs from evaluator')
    evidence = {'id':identifier,'run_name':run_name,'valid_completed_result':not issues,'issues':issues,
                'history':history,'stages':stage_evidence.get('stages',{}),'checkpoint':ckpt,
                'metric_artifact':atom(metric_path) if metric_path.exists() else None,
                'config_artifact':atom(cfg_path) if cfg_path.exists() else None}
    if metric is not None:
        evidence['efficiency'] = metric.get('efficiency',{})
    return metric if not issues else None, evidence


def summary(metric):
    return {key:at(metric,path) for key,path in METRICS.items()}


def compare(child, parent):
    if set(child['per_case']) != set(parent['per_case']):
        raise ValueError('Contrasts require identical case identities')
    case_rows = {}
    n = len(parent['per_case'])
    variance = at(parent,'field_casebalanced.rmse')**2/(1-at(parent,'field_casebalanced.r2'))
    child_variance = at(child,'field_casebalanced.rmse')**2/(1-at(child,'field_casebalanced.r2'))
    if not math.isclose(variance,child_variance,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError('casebalanced target variances differ')
    for case,p in parent['per_case'].items():
        c = child['per_case'][case]
        if c['overall']['n'] != p['overall']['n']:
            raise ValueError('Per-case point count changed: '+case)
        delta = {key:at(c,path)-at(p,path) for key,path in CASE_METRICS.items()}
        contribution = (at(p,'overall.rmse')**2-at(c,'overall.rmse')**2)/(n*variance)
        case_rows[case] = {'child':{key:at(c,path) for key,path in CASE_METRICS.items()},
                           'parent':{key:at(p,path) for key,path in CASE_METRICS.items()},
                           'delta':delta,'physical_r2_cb_contribution':contribution,
                           'ratio_closeness_improvement':{key:abs(at(p,path)-1)-abs(at(c,path)-1) for key,path in CASE_METRICS.items() if key.endswith('_ratio')}}
    total = sum(row['physical_r2_cb_contribution'] for row in case_rows.values())
    expected = at(child,'field_casebalanced.r2')-at(parent,'field_casebalanced.r2')
    if not math.isclose(total,expected,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError('Per-case MSE contributions do not sum to physical R2_cb contrast')
    costs = sorted(case_rows,key=lambda c:case_rows[c]['physical_r2_cb_contribution'])
    return {'delta':{key:at(child,path)-at(parent,path) for key,path in METRICS.items()},
            'case_count':n,'per_case_improved':{key:sum(row['delta'][key]>0 for row in case_rows.values()) for key in ('r2','iou','spearman')},
            'per_case_lower_mae':sum(row['delta']['mae_pa']<0 for row in case_rows.values()),
            'per_case_median_delta':{key:statistics.median(row['delta'][key] for row in case_rows.values()) for key in ('r2','iou','spearman')},
            'physical_r2_cb_decomposition':{'definition':'(parent case MSE-child case MSE)/(34*common casebalanced target variance)',
                'common_target_variance':variance,'sum':total,'expected_delta':expected,'absolute_reconstruction_error':abs(total-expected),
                'largest_costs':{c:case_rows[c]['physical_r2_cb_contribution'] for c in costs[:5]},
                'largest_benefits':{c:case_rows[c]['physical_r2_cb_contribution'] for c in costs[-5:][::-1]}},
            'per_case':case_rows}


def factorial(loaded):
    if not all(key in loaded for key in ('MS1','MS2','MS3','MS4')):
        return None
    values = {key:summary(loaded[key]) for key in ('MS1','MS2','MS3','MS4')}
    effect = {key:(values['MS4'][key]-values['MS3'][key])-(values['MS2'][key]-values['MS1'][key]) for key in METRICS}
    cases = {case:{key:(at(loaded['MS4']['per_case'][case],path)-at(loaded['MS3']['per_case'][case],path))-(at(loaded['MS2']['per_case'][case],path)-at(loaded['MS1']['per_case'][case],path)) for key,path in CASE_METRICS.items()} for case in loaded['MS1']['per_case']}
    return {'formula':'(MS4-MS3)-(MS2-MS1)',
            'interpretation':'Descriptive single/multi-scale x noBT/BT interaction; scale factor also changes capacity and fusion. No significance inference.',
            'metric_interaction':effect,'per_case_interaction':cases}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['partial','final'], default='final')
    args=ap.parse_args()
    torch.set_num_threads(1)
    matrix=read(MATRIX); queue=read(EXP/'queue_status.json')
    legacy_queue=read(ROOT/'training_wss_min/experiments/v6_followup_20260909/queue_status.json')
    integrity=source_integrity(queue)
    legacy_attested=(legacy_queue.get('status')=='complete' and legacy_queue.get('source_changed') is False and bool(legacy_queue.get('source_sha256')) and legacy_queue.get('source_sha256')==legacy_queue.get('source_sha256_end'))
    expected_cases=set(read(ROOT/'data_wss_v5/views/wss_min_view_v1/split_V5_train138_test34.json')['test_cases'])
    if len(expected_cases)!=34:
        raise ValueError('expected split must contain34 unique cases')
    arms={'MS0':matrix['reference']['run_name'],**{a['id']:a['run_name'] for a in matrix['arms']}}
    if set(arms)!={'MS'+str(i) for i in range(7)}:
        raise ValueError('This analysis covers exactlyMS0-MS6')
    out={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'requested_mode':args.mode,
         'analysis_script':atom(Path(__file__).resolve()),'matrix':atom(MATRIX),
         'queue_job_id':queue['job_id'],'queue_status':queue['status'],'source_integrity':integrity,
         'legacy_MS0_source_start_end_attested':legacy_attested,
         'metric_paths':METRICS,'checkpoint_results':{},
         'limitations':['Single seed1234; exposed test34; all results exploratory.',
             'Efficiency timing is observed under actual concurrent jobs, not a controlled hardware benchmark.',
             'Weight and projection norms do not measure causal branch contribution.',
             'MS1 bridge includes grouping/cap/centre condition/newSA3 initialization; original M2 is the practical anchor.',
             'Geometry G/M6 and conditional MS7-MS9 are outside this analysis and remain unexecuted.']}
    for ck in ('best','last'):
        loaded={}; result={'runs':{},'contrasts':{}}
        for identifier,run_name in arms.items():
            metric,evidence=run_evidence(identifier,run_name,ck,legacy_queue if identifier=='MS0' else queue,expected_cases)
            result['runs'][identifier]=evidence
            if metric is not None:
                loaded[identifier]=metric
                evidence['metrics']=summary(metric)
        for child,parent,meaning in CONTRASTS:
            if child in loaded and parent in loaded:
                result['contrasts'][child+'_vs_'+parent]={'meaning':meaning,**compare(loaded[child],loaded[parent])}
        result['factorial_interaction']=factorial(loaded)
        result['completed_new_run_count']=len(set(loaded)-{'MS0'})
        out['checkpoint_results'][ck]=result
    stages_ok=all(queue.get('arms',{}).get('MS'+str(i),{}).get('status')=='complete' and all(queue['arms']['MS'+str(i)]['stages'].get(stage,{}).get('returncode')==0 for stage in ('train','eval_best','eval_last')) for i in range(1,7))
    final=queue.get('status')=='complete' and integrity['passed'] and legacy_attested and stages_ok and all(result['completed_new_run_count']==6 and result['runs']['MS0']['valid_completed_result'] for result in out['checkpoint_results'].values())
    out['finalization']={'all_six_runs_and_12_evaluations_valid':all(x['completed_new_run_count']==6 for x in out['checkpoint_results'].values()),'all18_stages_zero':stages_ok,'complete_and_frozen':final}
    out['scope']='final' if final and args.mode=='final' else 'partial_informational'
    refused=args.mode=='final' and not final
    name='independent_analysis_refused.json' if refused else 'independent_analysis_'+out['scope'].replace('_informational','')+'.json'
    dest=EXP/name
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(dest),'scope':out['scope'],'refused_final':refused,'completed':{ck:x['completed_new_run_count'] for ck,x in out['checkpoint_results'].items()}},ensure_ascii=False))
    return 2 if refused else 0


if __name__=='__main__':
    raise SystemExit(main())
