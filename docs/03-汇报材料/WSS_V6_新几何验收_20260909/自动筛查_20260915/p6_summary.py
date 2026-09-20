#!/usr/bin/env python3
"""Summarise existing V6 P6 stability reports without recomputation.

The current candidate reports already contain P6 variants for valid reference
sections.  This script only reads those report.json/geometry.npz files, excludes
known duplicate cases, and emits case/category plus along-segment aggregates.
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
DEFAULT_REPORTS = ROOT / 'outputs/wss_v6_geometry_candidate_20260909/cases'
DEFAULT_OUT = ROOT / 'docs/03-汇报材料/WSS_V6_新几何验收_20260909/自动筛查_20260915/深化审查_20260915'
EXCLUDE = {'LIU_WEN_QI', 'HOU_SHEN_QIAN'}
VAR_GROUPS = {
    'shift_pm1mm': ['shift_minus_1mm', 'pointcloud_shift_minus_1mm', 'shift_plus_1mm', 'pointcloud_shift_plus_1mm'],
    'tilt_pm5deg': ['tilt_minus_5deg', 'pointcloud_tilt_minus_5deg', 'tilt_plus_5deg', 'pointcloud_tilt_plus_5deg'],
    'pointcloud_full_half': ['pointcloud_full', 'pointcloud_half'],
}
VAR_SOURCE = {name: ('pointcloud' if name.startswith('pointcloud_') else 'reference_surface') for names in VAR_GROUPS.values() for name in names}
BASE = {
    'reference_surface': {'valid': 'section_valid', 'raw': 'section_raw_contour_valid', 'group': 'section_same_segment_crossing_group_count', 'remote': 'section_same_segment_nonlocal_crossing_count'},
    'pointcloud': {'valid': 'pc_section_valid', 'raw': 'pc_section_raw_contour_valid', 'group': 'pc_section_same_segment_crossing_group_count', 'remote': 'pc_section_same_segment_nonlocal_crossing_count'},
}

def finite(x):
    try: return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError): return False

def q(vals, p):
    a = np.asarray([float(x) for x in vals if finite(x)], dtype=float)
    return float(np.quantile(a, p)) if len(a) else None

def as_bool(x):
    return bool(x) if x is not None else False

def rel_value(v):
    # same-source baseline is the preferred P6 area change; old full/half entries
    # only expose relative_area_change (to the reference baseline).
    x = v.get('relative_to_same_source_baseline')
    if not finite(x): x = v.get('relative_area_change')
    return float(x) if finite(x) else None

def classify_reason(rows, overall=False):
    reasons = []
    for r in rows:
        if r['gate_flip_count'] > 0 or r['became_invalid_count'] > 0: reasons.append('gate_flip_or_became_invalid')
        if r['closed_contour_change_count'] > 0: reasons.append('closed_contour_change')
        if r['group_change_count'] > 0: reasons.append('crossing_group_change')
        if r['nonlocal_crossing_count'] > 0 or r['nonlocal_change_count'] > 0: reasons.append('nonlocal_crossing')
        p95 = r.get('abs_rel_area_q95')
        if p95 is None: reasons.append('no_compliant_area_comparison')
        elif p95 >= 0.20: reasons.append('area_p95_ge_20pct')
        elif p95 >= 0.10: reasons.append('area_p95_ge_10pct')
        if r.get('baseline_invalid_count', 0) > 0: reasons.append('source_baseline_invalid')
    # preserve order while deduplicating
    reasons = list(dict.fromkeys(reasons))
    hard = any(x in reasons for x in ('gate_flip_or_became_invalid','closed_contour_change','crossing_group_change','nonlocal_crossing','area_p95_ge_20pct','no_compliant_area_comparison','source_baseline_invalid'))
    auto = bool(rows) and not hard and all((r.get('abs_rel_area_q95') is not None and r['abs_rel_area_q95'] < 0.10) for r in rows)
    return auto, (not auto), reasons

def variant_rows(report_path):
    report = json.loads(report_path.read_text())
    cid = report.get('canonical_id', report_path.parent.name.replace('__','/'))
    leaf = cid.rsplit('/',1)[-1]
    if leaf in EXCLUDE: return None
    zpath = report.get('geometry_npz')
    if not zpath: zpath = str(report_path.parent/'geometry.npz')
    z = np.load(zpath, allow_pickle=False)
    out = []
    for e in report.get('stability', []):
        i = int(e['section_index'])
        # P6 samples are generated from valid reference sections; enforce this
        # from geometry.npz so malformed/legacy records cannot enter silently.
        if i >= len(z['section_valid']) or not bool(z['section_valid'][i]): continue
        seg = int(e['segment_id']); s = float(e['s_local_mm'])
        for v in e.get('variants', []):
            name = v.get('name','')
            if name not in VAR_SOURCE: continue
            src = VAR_SOURCE[name]; b = BASE[src]
            base_valid = bool(z[b['valid']][i]); base_raw = bool(z[b['raw']][i])
            base_group = int(z[b['group']][i]) if finite(z[b['group']][i]) else None
            base_remote = int(z[b['remote']][i]) if finite(z[b['remote']][i]) else 0
            pipeline = bool(v.get('pipeline_valid', v.get('valid', False)))
            raw = bool(v.get('raw_contour_valid', False))
            vg = int(v['same_segment_crossing_group_count']) if finite(v.get('same_segment_crossing_group_count')) else None
            vr = int(v['same_segment_nonlocal_crossing_count']) if finite(v.get('same_segment_nonlocal_crossing_count')) else 0
            out.append({'canonical_id':cid,'segment_id':seg,'s_local_mm':s,'section_index':i,'category':next(k for k,ns in VAR_GROUPS.items() if name in ns),'variant':name,'source':src,'base_valid':base_valid,'base_raw_contour_valid':base_raw,'base_group_count':base_group,'base_nonlocal_crossing_count':base_remote,'pipeline_valid':pipeline,'raw_contour_valid':raw,'group_count':vg,'nonlocal_crossing_count':vr,'gate_flip':pipeline != base_valid,'became_invalid':base_valid and not pipeline,'closed_contour_change':raw != base_raw,'group_change':vg is not None and base_group is not None and vg != base_group,'nonlocal_crossing':vr > 0,'nonlocal_change':vr != base_remote,'comparison_valid':bool(v.get('comparison_valid', False)),'abs_rel_area':abs(rel_value(v)) if rel_value(v) is not None and bool(v.get('comparison_valid', False)) else None})
    return report, out

def summarise(items, key_fields, row_type):
    groups = defaultdict(list)
    for x in items: groups[tuple(x[k] for k in key_fields)].append(x)
    rows=[]
    for key, xs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        cats = {x['category'] for x in xs}
        # each category row represents all variants and sections in that key.
        row = {'row_type':row_type, **dict(zip(key_fields,key)), 'n_base_sections': len({(x['section_index'],x['segment_id']) for x in xs}), 'n_variants':len(xs), 'n_valid':sum(x['pipeline_valid'] for x in xs), 'valid_rate':sum(x['pipeline_valid'] for x in xs)/len(xs) if xs else None, 'invalid_count':sum(not x['pipeline_valid'] for x in xs), 'became_invalid_count':sum(x['became_invalid'] for x in xs), 'baseline_invalid_count':sum(not x['base_valid'] for x in xs), 'comparison_valid_count':sum(x['comparison_valid'] for x in xs), 'gate_flip_count':sum(x['gate_flip'] for x in xs), 'closed_contour_change_count':sum(x['closed_contour_change'] for x in xs), 'group_change_count':sum(x['group_change'] for x in xs), 'nonlocal_crossing_count':sum(x['nonlocal_crossing'] for x in xs), 'nonlocal_change_count':sum(x['nonlocal_change'] for x in xs), 'abs_rel_area_q50':q([x['abs_rel_area'] for x in xs],.5), 'abs_rel_area_q95':q([x['abs_rel_area'] for x in xs],.95), 'abs_rel_area_max':q([x['abs_rel_area'] for x in xs],1), 's_min_mm':min((x['s_local_mm'] for x in xs),default=None), 's_max_mm':max((x['s_local_mm'] for x in xs),default=None)}
        auto, manual, reasons = classify_reason([row]); row['auto_release']=auto; row['manual_review']=manual; row['manual_reasons']=';'.join(reasons)
        rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reports',type=Path,default=DEFAULT_REPORTS); ap.add_argument('--out',type=Path,default=DEFAULT_OUT); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    pairs=[]; skipped=0
    for p in sorted(args.reports.glob('*/report.json')):
        x=variant_rows(p)
        if x is None: skipped+=1
        else: pairs.append(x)
    all_items=[it for _,items in pairs for it in items]
    case_cat=summarise(all_items,['canonical_id','category'],'case_category')
    seg_cat=summarise(all_items,['segment_id','category'],'segment_category')
    # Case-level decision uses all three categories, with category-level q95 and gates.
    bycase=defaultdict(list)
    for r in case_cat: bycase[r['canonical_id']].append(r)
    overall=[]
    for cid, rs in sorted(bycase.items()):
        auto,manual,reasons=classify_reason(rs,True)
        overall.append({'row_type':'case_overall','canonical_id':cid,'n_categories':len(rs),'n_base_sections':sum(r['n_base_sections'] for r in rs),'n_variants':sum(r['n_variants'] for r in rs),'n_valid':sum(r['n_valid'] for r in rs),'valid_rate':sum(r['n_valid'] for r in rs)/sum(r['n_variants'] for r in rs),'invalid_count':sum(r['invalid_count'] for r in rs),'became_invalid_count':sum(r['became_invalid_count'] for r in rs),'baseline_invalid_count':sum(r['baseline_invalid_count'] for r in rs),'comparison_valid_count':sum(r['comparison_valid_count'] for r in rs),'gate_flip_count':sum(r['gate_flip_count'] for r in rs),'closed_contour_change_count':sum(r['closed_contour_change_count'] for r in rs),'group_change_count':sum(r['group_change_count'] for r in rs),'nonlocal_crossing_count':sum(r['nonlocal_crossing_count'] for r in rs),'nonlocal_change_count':sum(r['nonlocal_change_count'] for r in rs),'abs_rel_area_q50':q([r['abs_rel_area_q50'] for r in rs],.5),'abs_rel_area_q95':max((r['abs_rel_area_q95'] for r in rs if r['abs_rel_area_q95'] is not None),default=None),'abs_rel_area_max':max((r['abs_rel_area_max'] for r in rs if r['abs_rel_area_max'] is not None),default=None),'auto_release':auto,'manual_review':manual,'manual_reasons':';'.join(reasons)})
    # mixed-type CSV with a stable union of fields.
    rows=overall+case_cat+seg_cat
    fields=['row_type','canonical_id','category','segment_id','n_categories','n_base_sections','n_variants','n_valid','valid_rate','invalid_count','became_invalid_count','baseline_invalid_count','comparison_valid_count','gate_flip_count','closed_contour_change_count','group_change_count','nonlocal_crossing_count','nonlocal_change_count','abs_rel_area_q50','abs_rel_area_q95','abs_rel_area_max','s_min_mm','s_max_mm','auto_release','manual_review','manual_reasons']
    with (args.out/'p6_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in rows)
    auto_cases=[r['canonical_id'] for r in overall if r['auto_release']]; manual_cases=[r for r in overall if r['manual_review']]
    payload={'metadata':{'source_reports':str(args.reports),'excluded_cases':sorted(EXCLUDE),'n_reports_read':len(pairs),'n_reports_excluded':skipped,'n_cases':len(overall),'n_stability_items':len({(x['canonical_id'],x['section_index']) for x in all_items}),'criteria':{'auto_release':'all categories have compliant-area p95 < 10%; zero gate flips, closed-contour/group changes, nonlocal crossing, baseline invalid or missing comparisons','manual_review':'gate flip, closed-contour/group change, nonlocal crossing, missing comparison, source baseline invalid, or area p95 >= 20%; 10-20% also not auto-released'}},'case_overall':overall,'case_category':case_cat,'segment_category':seg_cat}
    (args.out/'p6_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# P6 沿程稳定性自动审查摘要（2026-09-15）','',f"读取 {len(pairs)} 个病例，排除重复病例 {', '.join(sorted(EXCLUDE))}；P6 有效基础截面 {payload['metadata']['n_stability_items']} 个。",'', '## 自动放行候选', '', f"共 **{len(auto_cases)}** 例：" + ('、'.join(auto_cases) if auto_cases else '无') + '', '', '## 需要人工确定', '', f"共 **{len(manual_cases)}** 例。触发条件包括 gate flip、闭合轮廓/交叉组变化、非局部 crossing、源基线无效、无可比面积或面积变化 p95≥20%；p95 在 10–20% 之间也不会自动放行。", '']
    for r in sorted(manual_cases,key=lambda x:(-x['gate_flip_count']-x['nonlocal_crossing_count']-x['group_change_count'], -(x['abs_rel_area_q95'] or 0),x['canonical_id'])):
        lines.append(f"- `{r['canonical_id']}`：p95={r['abs_rel_area_q95'] if r['abs_rel_area_q95'] is not None else 'NA'}，invalid={r['became_invalid_count']}，gate_flip={r['gate_flip_count']}，contour_change={r['closed_contour_change_count']}，group_change={r['group_change_count']}，nonlocal={r['nonlocal_crossing_count']}；{r['manual_reasons']}")
    lines += ['', '## 沿程 segment 汇总', '', '|segment|category|base sections|valid rate|gate flips|group changes|nonlocal|area p95|', '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in seg_cat: lines.append(f"|{r['segment_id']}|{r['category']}|{r['n_base_sections']}|{r['valid_rate']:.3f}|{r['gate_flip_count']}|{r['group_change_count']}|{r['nonlocal_crossing_count']}|{r['abs_rel_area_q95'] if r['abs_rel_area_q95'] is not None else 'NA'}|")
    lines += ['', '明细见 `p6_summary.json` 和 `p6_summary.csv`；脚本为 `p6_summary.py`。']
    (args.out/'p6_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'cases':len(overall),'excluded':skipped,'base_sections':payload['metadata']['n_stability_items'],'auto_release':len(auto_cases),'manual_review':len(manual_cases),'out':str(args.out)},ensure_ascii=False))
if __name__=='__main__': main()
