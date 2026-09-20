"""Offline review desk: machine actions and only unresolved human decisions."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import html
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
try:
    from wss_v5.bifurcation_user_confirmations import apply_confirmations, validate_confirmations
except ModuleNotFoundError:  # direct execution from inside the wss_v5 directory
    from bifurcation_user_confirmations import apply_confirmations, validate_confirmations

def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj

def write_csv(path,rows,fieldnames=None):
    keys=fieldnames or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--resolution',type=Path,required=True)
    ap.add_argument('--confirmations',type=Path,help='Provenance-bound confirmations for the two named root positions only')
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.root/'refinement_manifest.json').read_text())
    confirmations=None
    if a.confirmations:
        confirmations=validate_confirmations(a.confirmations,a.resolution.parent/'bifurcation_route_recovery',Path(manifest['source']))
    bif_page=a.resolution.parent/'remaining_bifurcation_review.html'
    bif_review={}
    if bif_page.exists() or confirmations:
        if confirmations:
            spec=importlib.util.spec_from_file_location('confirmed_bifurcation_review',a.resolution.parent/'make_resolution_review.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            bif_info=module.render_review(a.out,confirmations,Path(manifest['source']),'../cases/')
        else:
            (a.out/'remaining_bifurcation_review.html').write_text(bif_page.read_text().replace('../../cases/','../cases/'))
            bif_info=json.loads(bif_page.with_suffix('.json').read_text())
        bif_review={case['canonical_id']:case for case in bif_info['cases']}
        (a.out/'bifurcation_route_recovery').mkdir(exist_ok=True)
        for case in bif_info['cases']:
            name=case['key']+'.json'
            shutil.copy2(a.resolution.parent/'bifurcation_route_recovery'/name,a.out/'bifurcation_route_recovery'/name)
    audit_path=a.root/'audit_refined_geometry.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    audited=audit.get('summary',{})
    if (not audited.get('all_mechanical_checks_pass')
        or audited.get('audited_cases') != len(manifest['cases'])
        or Path(audit.get('candidate','')).resolve() != a.root.resolve()
        or {c['canonical_id'] for c in audit.get('cases',[])} != {c['canonical_id'] for c in manifest['cases']}):
        ap.error('a completed passing independent audit of this exact candidate cohort is required')
    policy=manifest.get('policy',{})
    independent_final_probes=(policy.get('holdout_azimuth_offset_deg') == 45.
        and policy.get('final_shift_center') == 'original_atlas_arclength_interpolation'
        and policy.get('final_shift_profile') == 'simultaneous_monotone_masks_until_fixed_point'
        and policy.get('recovery_minimum_final_shift_probes',0) >= 1)
    resolution=json.loads(a.resolution.read_text())
    if confirmations:resolution=apply_confirmations(resolution,confirmations)
    res={r['canonical_id']:r for r in resolution['cases']}
    cases=[];actions=[];counts=Counter();pc_counts=Counter();manual=[];low=[];user_confirmed=[]
    shift_coverage=Counter()
    for row in manifest['cases']:
        cid=row['canonical_id'];key=cid.replace('/','__');dest=a.root/'cases'/key
        r=json.loads((dest/'report.json').read_text());d=json.loads((dest/'refinement.json').read_text())
        with np.load(dest/'geometry.npz') as arch:z={k:arch[k] for k in arch.files}
        for probe_row in d.get('final_verification',{}).get('sections',[]):
            k=probe_row['section_index']
            if not z['section_valid'][k]:continue
            n=len(probe_row['shift'])
            shift_coverage['verified_stations' if n else 'original_stations_without_contiguous_bracket']+=1
            shift_coverage['retained_shift_probes']+=n
            if probe_row['was_recovered']:
                shift_coverage[f'recovered_with_{n}_directions']+=1
        source=Path(r['source_candidate']['directory'])
        with np.load(source/'geometry.npz') as arch:old={k:arch[k] for k in ['section_area_mm2','section_valid','section_polygon_offsets','section_polygon_xyz_mm','section_tangent']}
        prior=res.get(cid,{})
        bif=prior.get('bifurcation_recovery') or []
        confirmed_bif=[b for b in bif if b.get('resolved_by_user_confirmation')]
        unresolved_bif=[b for b in bif if not b['resolved_by_route_consensus'] and not b.get('resolved_by_user_confirmation')]
        focus=bif_review.get(cid,{}) if unresolved_bif else {}
        branch_by_id={s['segment_id']:s for s in r['segments']}
        manual_branches=[]
        for b in unresolved_bif:
            sid=b['parent_segment'];children=branch_by_id.get(sid,{}).get('children',[])
            manual_branches.append(f"分支 {sid} → {' / '.join(map(str,children))}，沿左右终末路径延伸")
        focus_mm=focus.get('focus_mm',[None,None])
        human=[]
        if unresolved_bif:human.append(focus.get('reason','分叉位置：延伸路径未形成一致且持续的管腔分离证据'))
        if not r['P1_centerline']['closed_reference_valid']:human.append('源表面残余非流形结构仍未自动修复')
        if r['P1_centerline']['outside_core_points']:human.append('真实解剖域内中心线仍在壁面外')
        coverage=r['P4_sections']['reference_valid_fraction'];pc_cov=r['P4_sections']['pointcloud_valid_fraction']
        hold=coverage<.60 or pc_cov<.50
        if hold:low.append(cid)
        brief={'canonical_id':cid,'manual_reasons':'；'.join(human),
               'manual_branches':'；'.join(manual_branches),
               'manual_s_start_mm':focus_mm[0],'manual_s_end_mm':focus_mm[1],
               'manual_s_definition':'从待定旧图节点起，沿左右扩展路径的 s；不是分支局部 s' if unresolved_bif else '',
               'manual_inspection_target':focus.get('review_target',''),
               'manual_evidence_page':'remaining_bifurcation_review.html' if focus else '',
               'root_position_user_confirmed':bool(confirmed_bif),
               'confirmed_root_s_mm':confirmed_bif[0]['user_confirmation']['selected_event']['s_mm'] if confirmed_bif else None,
               'confirmed_root_routes':json.dumps(confirmed_bif[0]['user_confirmation']['selected_event']['routes']) if confirmed_bif else '',
               'confirmed_root_center_mm':json.dumps(confirmed_bif[0]['user_confirmation']['selected_event']['center_mm']) if confirmed_bif else '',
               'reference_stable_stations':d['final_reference_valid'],'reference_recovered':d['counts'].get('recovered_stable_plane',0),
               'reference_automatically_withheld':sum(v for k,v in d['counts'].items() if k.startswith('automatically_withheld')),
               'pc_reference_withheld':d.get('pc_counts',{}).get('automatically_withheld_reference',0),
               'pc_density_withheld':d.get('pc_counts',{}).get('automatically_withheld_density',0),
               'reference_coverage':coverage,'pointcloud_coverage':pc_cov,'wall_coverage':r['P5_wall_mapping']['reference_valid_fraction'],
               'along_path_coverage_hold':hold,'source_closed':r['P1_centerline']['closed_reference_valid'],
               'outside_core_points':r['P1_centerline']['outside_core_points'],
               'recommendation':('待分叉/源几何定位' if human else '根部首次隔离位置已由用户确认' if confirmed_bif else '自动处理完成')+('；低覆盖沿程特征暂不建议启用' if hold else '；按有效 mask 保留候选证据'),
               'training_allowed':False}
        if human:manual.append(brief)
        if confirmed_bif:user_confirmed.append(brief)
        actions.append(brief);counts.update(d['counts']);pc_counts.update(d.get('pc_counts',{}))
        events=[]
        for k in np.flatnonzero(np.char.startswith(z['refinement_status'],'recovered')|np.char.startswith(z['refinement_status'],'automatically_withheld')):
            c=z['section_center_mm'][k].astype(float)
            # Compare both section polygons in a common local 3D frame: the
            # second plane has physically changed, not just changed area.
            t=old['section_tangent'][k].astype(float);axis=np.eye(3)[np.argmin(abs(t))];u=np.cross(t,axis);u/=np.linalg.norm(u);v=np.cross(t,u)
            def poly(archive):
                off=archive['section_polygon_offsets'];p=archive['section_polygon_xyz_mm'][off[k]:off[k+1]]-c
                return np.column_stack((p@u,p@v)).round(3).tolist()
            events.append({'station':int(k),'segment':int(z['section_segment_id'][k]),'s':float(z['section_s_local_mm'][k]),
                           'status':str(z['refinement_status'][k]),'old_area':float(old['section_area_mm2'][k]),
                           'new_area':float(z['section_area_mm2'][k]) if z['section_valid'][k] else None,
                           'angle':float(z['refinement_normal_change_deg'][k]),'old_poly':poly(old),'new_poly':poly(z)})
        # Downstream viewers see NaNs for withheld values, never the raw
        # polygon area as an implicitly accepted measurement.
        area=np.where(z['section_valid'],z['section_area_mm2'],np.nan)
        pca=np.where(z['pc_section_valid'],z['pc_section_area_mm2'],np.nan)
        oldarea=np.where(old['section_valid'],old['section_area_mm2'],np.nan)
        cases.append({**brief,'key':key,'segment':z['section_segment_id'],'s':z['section_s_local_mm'],
                      'old_area':oldarea,'area':area,'pc_area':pca,'events':events,
                      'bifurcation':bif,'P1':r['P1_centerline'],'cia':prior.get('cia_slice_transition_summary',prior.get('cia_summary',[]))})
    # Report build is contingent on a completed independent audit, never just
    # successful extraction or a reduced alarm count.
    summary={'schema':'v6_refined_review_desk_v1','cases':len(cases),'source':str(a.root.resolve()),
             'reference_counts':dict(counts),'pointcloud_counts':dict(pc_counts),'manual_case_count':len(manual),
             'manual_cases':manual,'low_coverage_hold_cases':low,'independent_audit':audit.get('summary',audit.get('status')),
             'automatically_handled_cases':sum(not x['manual_reasons'] and not x['root_position_user_confirmed'] for x in actions),'case_coverage_hold_is_not_a_user_decision':True,
             'user_confirmed_case_count':len(user_confirmed),'user_confirmed_cases':user_confirmed,
             'no_remaining_human_decision_cases':len(cases)-len(manual),
             'low_coverage_hold_count':len(low),'manual_and_low_coverage_count':sum(x['along_path_coverage_hold'] for x in manual),
             'automatically_handled_with_coverage_hold_count':sum(x['along_path_coverage_hold'] and not x['manual_reasons'] for x in actions),
             'audit_scope':'mechanical_geometry_consistency_not_clinical_anatomy_or_training_approval',
             'independent_recovery_tilt_and_final_profile_shift_policy':independent_final_probes,
             'pointcloud_requires_verified_reference':bool(policy.get('pointcloud_requires_verified_reference',False)),
             'final_shift_coverage':dict(shift_coverage),
             'source_geometry_unchanged':True,'training_allowed':False}
    (a.out/'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(a.out/'case_actions.csv',actions)
    write_csv(a.out/'human_decisions.csv',manual,fieldnames=list(actions[0]) if actions else ['canonical_id','manual_reasons'])
    event_rows=[{'canonical_id':c['canonical_id'],**{k:v for k,v in e.items() if not k.endswith('_poly')}} for c in cases for e in c['events']]
    write_csv(a.out/'section_actions.csv',event_rows)
    held=sum(v for k,v in counts.items() if k.startswith('automatically_withheld'))
    resolution_summary=resolution.get('summary',{})
    recovered_bif=sum(bool(c.get('bifurcation_recovery')) and all(b['resolved_by_route_consensus'] for b in c['bifurcation_recovery']) for c in resolution['cases'])
    cia_explained=resolution_summary.get('cia_outliers_now_inside_diagnostic_bands',0)
    verification_description=(
        '恢复候选除通过搜索时的倾角检查，还必须通过未参与搜索的四个方向倾角验证（方位角错开 45°）。'
        '全部最终有效站按原 atlas 弧长插值确定 ±1 mm 位移中心，并对最终沿程面积与有效 mask 反复复核，直到 mask 不再变化；恢复站至少保留一个可比较的位移探针。'
        f"最终恢复站中 {shift_coverage['recovered_with_1_directions']} 站有单侧连续支持、{shift_coverage['recovered_with_2_directions']} 站有双侧支持；另有 {shift_coverage['original_stations_without_contiguous_bracket']} 个原稳定站缺少连续邻站，位移标为无法评估，不作通过声明。"
        if independent_final_probes else
        '本版本的扰动检查按其 manifest 中的 policy 执行；未声明独立恢复倾角与最终沿程位移复核策略。')
    pointcloud_description=(
        f"点云先与最终有效参考截面取交集，因参考未通过而屏蔽 {pc_counts.get('automatically_withheld_reference',0)} 站；"
        f"再对剩余站执行减半密度审查，另屏蔽 {pc_counts.get('automatically_withheld_density',0)} 站。"
        '两类屏蔽按先后顺序分别计数，已经在参考检查中屏蔽的站不再计入密度失败；密度失败数下降不能解释为密度标准放宽。'
        if policy.get('pointcloud_requires_verified_reference',False) else
        f"点云减半密度审查额外屏蔽 {pc_counts.get('automatically_withheld_density',0)} 站；本版本未声明点云有效性必须与最终参考取交集。")
    md=[f"# 几何算法优化与人工待办（{'2026-09-16' if confirmations else '2026-09-15'}）",'',
        f"**{len(cases)} 例已重算；{summary['automatically_handled_cases']} 例完成自动处理，{len(user_confirmed)} 例由用户确认，剩余人工待办 {len(manual)} 例。** [打开交互审查台](index.html) · [分叉定位与确认记录](remaining_bifurcation_review.html) · [病例动作](case_actions.csv) · [截面修复与屏蔽明细](section_actions.csv) · [剩余人工待办](human_decisions.csv)",'',
        f"另有 **{len(low)} 例低覆盖，沿程特征暂不建议启用**，其中 {summary['manual_and_low_coverage_count']} 例也在人工定位清单。覆盖不足由算法保留缺失，不需要你逐站审批；“完成自动处理”不代表整例特征均可用。",'',
        f"沿程审查覆盖全部 {sum(counts.values()):,} 个站点，其中原有效 {sum(counts.values())-counts.get('previously_invalid',0):,} 站接受倾角与位移稳定性审查。{counts.get('recovered_stable_plane',0)} 站经重选切面恢复稳定，{held} 站自动标无效；恢复数不是新增原无效站数。",'',
        verification_description,'',
        pointcloud_description,'',
        '屏蔽站的原始多边形和面积仅留作诊断，壁面有效字段为 NaN；坡度、上下游 10 mm 最小面积比/距离与各自 mask 均重算，不跨无效段。原有稳定截面面积保持原值，没有用平滑曲线替换真实面积。','',
        '最终位移截面与沿程预测不一致，既可能来自真实几何急变，也可能来自提取误差。算法对这些站保留缺失；mask 为无效不代表源解剖违规。','',
        '## 你需要确定的事项','',
        '| 病例 | 待看位置 | 你需要判断的内容 | 保留人工的依据 |','| --- | --- | --- | --- |']
    md += [f"| {x['canonical_id']} | {x['manual_branches']}；{x['manual_s_start_mm']:g}–{x['manual_s_end_mm']:g} mm | {x['manual_inspection_target']} | {x['manual_reasons']} |" if x['manual_s_start_mm'] is not None else f"| {x['canonical_id']} | {x['manual_branches'] or '见本例源几何证据'} | {x['manual_inspection_target'] or x['manual_reasons']} | {x['manual_reasons']} |" for x in manual]
    if not manual:
        md += ['', '本轮没有剩余人工待办。`human_decisions.csv` 仅保留表头，不含病例行。']
    if any(x['manual_s_start_mm'] is not None for x in manual):
        md += ['', '定位区间的 s 从待定旧图节点起、沿左右扩展路径计算，不能直接当作交互面积图的分支局部 s。定位页提供同一坐标系的原壁面、中心线与候选点。']
    if confirmations:
        md += ['', '## 已落实的用户确认','',
               f"用户原话（{confirmations['confirmed_at']}）：“{confirmations['user_statement_verbatim']}”",'',
               '仅对以下两例取最早有效独立轮廓的扫描事件，不要求 1 mm 持续；同最早 s 的多路坐标全部保留，展示主记录按路径字典序确定。原算法未形成共识的状态与分歧证据仍保留。','',
               '| 病例 | 确认 s（mm） | 路径 | 切平面中心 xyz（mm） |','| --- | --- | --- | --- |']
        for c in confirmations['cases']:
            e=c['selected_event']
            md.append(f"| {c['canonical_id']} | {e['s_mm']:g} | {' / '.join('→'.join(map(str,r)) for r in e['routes'])} | {', '.join(f'{x:.6f}' for x in e['center_mm'])} |")
        md += ['', '此处 s 从旧根图节点沿扩展路径计算，坐标是原几何中的扫描切平面中心。确认只解除这两例根部定位待办，不等于整例解剖批准，不修改原图节点或授予训练许可。完整记录见 [用户确认侧车](user_bifurcation_confirmations.json)。']
    md += ['', '## 已自动处理','',
           '- 开口范围：依据实际开口多边形/封帽交点，排除中心线在解剖开口外的连续端段；仍保留原中心线坐标及排除证据。',
           '- 封帽：检查闭合边与非流形边，用有几何证明的封帽修复审计副本；原始 CFD 壁点、节点号、网格和 HDF5 不变。',
           f'- 分叉：通过四种左右终末路径延伸扫描的共识，恢复 {recovered_bif} 例旧扫描未定的切片转变定位。结果保存在分叉审查侧车；切片转变中心不是壁面分隔脊，不改写原图节点。',
           f'- CIA：图分段长度与两次管腔分离之间的长度分开存储；{cia_explained} 例原“过长”由定义差异解释。短/曲折本身不判违规。',
           '- 已确认的 WANG_KUI_WU 标签修正继续有效；两对重复病例不再进入人工清单。','',
           '## 覆盖与使用边界','',
           f"{len(low)} 例仍低于原有覆盖提示线（参考 60% 或点云 50%）："+('、'.join(low) if low else '无')+'。这些病例的沿程特征暂不建议启用，无需人工逐站放行；稳定区域与入口等其他候选证据保留。',
           '点云的选切面过程参考了表面网格，只属于几何候选修复；这不证明裸点云部署可直接复现。全部训练批准状态保持 false。',
           '独立审计验证机械几何一致性、来源与派生字段，不能等同于临床解剖确认或训练批准；自动移出人工清单表示已有足够证据处理本轮提取问题。',
           'dV/ds、A/(πRmis²)、非星形和 CIA 长度区间只作诊断。已有 P6 抽样结果保留为旧候选记录，新候选以全站审查和独立审计为准。','',
           '## 验证与复现','',
           f"新候选：`{a.root}`；`refinement_manifest.json` 记录 Slurm 作业、{len(cases)} 例名单和参数，`code/` 保存执行代码。独立审计见同目录 `audit_refined_geometry.json`。"]
    (a.out/'README.md').write_text('\n'.join(md)+'\n')
    asset=Path(__file__).resolve().parents[1]/'docs/03-汇报材料/WSS_V6_新几何验收_20260909/assets/plotly-2.35.2.min.js'
    shutil.copy2(asset,a.out/'plotly.min.js')
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>几何算法优化：只看剩余决策</title>
<style>body{font:15px system-ui;margin:24px;color:#172c39;background:#f5f7fa}h1{font-size:25px}button,select{font:inherit;padding:8px;margin:4px}main{background:white;border-radius:12px;padding:18px}.cards{display:flex;gap:15px}.card{background:#e5eff5;padding:15px;border-radius:10px;flex:1}.row{display:grid;grid-template-columns:1.4fr 1fr;gap:12px}#curve,#contour{height:430px}#status{padding:15px;background:#fff4dc;white-space:pre-wrap}small{color:#546879}a{color:#086eac}#detail{white-space:pre-wrap;max-height:200px;overflow:auto}</style>
<h1>几何算法优化与人工待办</h1><p id="intro"></p><p><a href="remaining_bifurcation_review.html">打开剩余分叉定位页：局部三维壁面、候选位置与沿程状态</a></p><div class="cards" id="cards"></div><p><button id="manual">只看需人工确定</button><button id="all">查看全部病例</button><select id="case"></select><select id="branch"></select><select id="event"></select></p>
<main><div id="status"></div><div class="row"><div id="curve"></div><div id="contour"></div></div><small>截面轮廓投影到原切面坐标用于对照；新切面的物理角度另列。曲线断点代表缺失。位移不一致可来自真实急变或提取误差，屏蔽不表示源解剖违规。</small><p id="pc-policy"></p><p><a href="README.md">结果说明</a> · <a href="case_actions.csv">全部病例动作</a> · <a href="human_decisions.csv">仅人工待办CSV</a></p><details><summary>本例自动审查证据</summary><pre id="detail"></pre></details></main>
<script src="plotly.min.js"></script><script>const D=__DATA__,S=__SUMMARY__;let cur=null;
const $=id=>document.getElementById(id);$('intro').textContent=`${S.cases} 例已重算，${S.automatically_handled_cases} 例完成自动处理，${S.manual_case_count} 例需人工定位。另有 ${S.low_coverage_hold_count} 例低覆盖沿程特征暂不建议启用（其中 ${S.manual_and_low_coverage_count} 例也需人工定位），无需逐站审批。自动处理不等同于整例特征可用；机械审计不代表临床解剖或训练批准。`;$('all').textContent=`查看全部 ${S.cases} 例`;
$('pc-policy').textContent=S.pointcloud_requires_verified_reference?`点云先因参考未通过屏蔽 ${S.pointcloud_counts.automatically_withheld_reference||0} 站，再因减半密度未通过屏蔽 ${S.pointcloud_counts.automatically_withheld_density||0} 站；两类按顺序分别计数，密度标准未放宽。`:'';
$('cards').innerHTML=[['完成自动处理',S.automatically_handled_cases],['低覆盖沿程特征暂缓',S.low_coverage_hold_count],['重选切面恢复稳定',S.reference_counts.recovered_stable_plane||0],['自动屏蔽截面',Object.entries(S.reference_counts).filter(([k])=>k.startsWith('automatically_withheld')).reduce((a,[k,v])=>a+v,0)],['人工定位待办',S.manual_case_count]].map(([k,v])=>`<div class="card">${k}<h2>${v}</h2></div>`).join('');
function fill(manual){$('case').innerHTML='';D.filter(c=>!manual||c.manual_reasons).forEach(c=>{let o=new Option(c.canonical_id,c.key);$('case').add(o)});if($('case').options.length)show()}
function show(){cur=D.find(c=>c.key===$('case').value);$('branch').innerHTML='';[...new Set(cur.segment)].forEach(s=>$('branch').add(new Option('分支 '+s,s)));$('event').innerHTML='';$('event').add(new Option('选取自动修复/屏蔽站点',''));cur.events.forEach((e,i)=>$('event').add(new Option(`站 ${e.station} · seg${e.segment} · ${e.status.startsWith('recovered')?'已恢复':'已屏蔽'}`,i)));const location=cur.manual_s_start_mm===null?'':`${cur.manual_branches}\n优先查看 ${cur.manual_s_start_mm}–${cur.manual_s_end_mm} mm；${cur.manual_s_definition}\n${cur.manual_inspection_target}\n`;$('status').textContent=`${cur.canonical_id}\n${location}${cur.manual_reasons}\n${cur.recommendation}\n参考/点云覆盖 ${(cur.reference_coverage*100).toFixed(1)}% / ${(cur.pointcloud_coverage*100).toFixed(1)}%`; $('detail').textContent=JSON.stringify({P1:cur.P1,bifurcation:cur.bifurcation},null,2);draw();Plotly.purge('contour')}
function draw(){let s=Number($('branch').value),ids=cur.segment.map((v,i)=>v===s?i:-1).filter(i=>i>=0);let traces=[['原参考',cur.old_area,'#9ba9b1','dash'],['修复后参考',cur.area,'#147c93','solid'],['修复后点云',cur.pc_area,'#dc8e32','dot']].map(([name,y,color,dash])=>({x:ids.map(i=>cur.s[i]),y:ids.map(i=>y[i]),mode:'lines+markers',marker:{size:3},line:{color,dash},name,connectgaps:false}));Plotly.react('curve',traces,{title:'沿程截面积与无效区',xaxis:{title:'本分支 s（mm）'},yaxis:{title:'面积（mm²）',type:'log'},margin:{t:45,b:50,l:70,r:10},legend:{orientation:'h'}},{responsive:true})}
function event(){if($('event').value==='')return;let e=cur.events[Number($('event').value)];$('branch').value=e.segment;draw();let trace=(p,name,color)=>({x:[...p.map(x=>x[0]),p[0]?.[0]],y:[...p.map(x=>x[1]),p[0]?.[1]],mode:'lines',name,line:{color}});let traces=[trace(e.old_poly,'原轮廓','#a2aab0')];if(e.new_area!==null)traces.push(trace(e.new_poly,'修复轮廓','#147c93'));Plotly.react('contour',traces,{title:`站 ${e.station}：${e.old_area.toFixed(1)} → ${e.new_area===null?'不可用':e.new_area.toFixed(1)} mm²；转角 ${e.angle}°`,xaxis:{title:'局部x（mm）'},yaxis:{title:'局部y（mm）',scaleanchor:'x'},margin:{t:55,b:50,l:50,r:10}},{responsive:true})}
$('manual').onclick=()=>fill(true);$('all').onclick=()=>fill(false);$('case').onchange=show;$('branch').onchange=draw;$('event').onchange=event;fill(true);
</script></html>'''
    doc=template.replace('__DATA__',json.dumps(clean(cases),ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')).replace('__SUMMARY__',json.dumps(clean(summary),ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (a.out/'index.html').write_text(doc)
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
