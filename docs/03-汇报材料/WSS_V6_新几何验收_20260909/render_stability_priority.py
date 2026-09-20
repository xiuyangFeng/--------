"""Independent offline viewer of saved before/after contour diagnostics."""
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from wss_v5.render_geometry_candidate import CSS, dumps, plane_frame


def projected(points, center, normal):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    e1, e2 = plane_frame(np.asarray(normal).copy())
    return np.column_stack([(points-center) @ e1, (points-center) @ e2])


def flattened_edges(edges):
    edges = np.asarray(edges, dtype=float).reshape(-1, 2, 3)
    out = np.full((len(edges), 3, 3), np.nan)
    out[:, :2] = edges
    return out.reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/wss_v6_geometry_candidate_20260909/priority_stability_diagnostics')
    parser.add_argument('--out', type=Path, default=HERE/'priority_stability.html')
    parser.add_argument('--candidate-version', default='v1.2', help='Explicit source version of this saved diagnostic batch')
    parser.add_argument('--recheck-of', help='Historical rank version; current gate is a recheck, not a fresh top-six ranking')
    args = parser.parse_args()
    manifest = json.loads((args.source/'manifest.json').read_text())
    if manifest.get('applied_gate_schema') and not manifest['applied_gate_schema'].endswith('_'+args.candidate_version):
        raise ValueError('Requested displayed version disagrees with diagnostic gate schema')
    records = []
    for entry in manifest['diagnostics']:
        source_path = Path(entry['diagnostic_json'])
        if not source_path.is_file():
            source_path = args.source/source_path.name
        record = json.loads(source_path.read_text())
        base = record['baseline']
        center = np.asarray(base['center_mm'])
        normal = np.asarray(base['normal'])
        for kind in ['baseline', 'perturbed']:
            data = record[kind]
            data['polygon_common2'] = projected(data['polygon_xyz_mm'], center, normal)
            data['raw3'] = flattened_edges(data['intersection_segments_xyz_mm'])
            data['raw_common2'] = projected(data['raw3'], center, normal)
            groups = data.get('same_segment_plane_intersections', {}).get('inside_crossing_groups', [])
            data['crossings3'] = [point for group in groups for point in group['points_xyz_mm']]
            data['crossings2'] = projected(data['crossings3'], center, normal)
        records.append(record)
    payload = dict(candidate_version=args.candidate_version, manifest=manifest, diagnostics=records)
    js = r'''
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
function escapeHtml(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function line2(p,name,color,close=true){let q=close&&p.length?[...p,p[0]]:p;return {type:'scatter',mode:'lines',name,x:q.map(x=>x[0]),y:q.map(x=>x[1]),line:{color,width:2}}}
function line3(p,name,color,close=true){let q=close&&p.length?[...p,p[0]]:p;return {type:'scatter3d',mode:'lines',name,x:q.map(x=>x[0]),y:q.map(x=>x[1]),z:q.map(x=>x[2]),line:{color,width:4}}}
function metric(v){return v===null||v===undefined?'—':typeof v==='number'?v.toFixed(3):escapeHtml(v)}
function marks2(p,name,color){return {type:'scatter',mode:'markers',name,x:p.map(x=>x[0]),y:p.map(x=>x[1]),marker:{color,size:11,symbol:'x'}}}
function marks3(p,name,color){return {type:'scatter3d',mode:'markers',name,x:p.map(x=>x[0]),y:p.map(x=>x[1]),z:p.map(x=>x[2]),marker:{color,size:5,symbol:'diamond'}}}
function render(i){const r=D.diagnostics[i],b=r.baseline,p=r.perturbed;document.getElementById('select').value=i;document.getElementById('case').textContent=r.canonical_id+' · #'+r.section_index_zero_based+' · S'+r.segment_id+' · '+r.variant;
 const fields=[['s / mm','s_local_mm'],['诊断轮廓 A / mm²（invalid 不供输入）','area_mm2'],['闭环数','closed_contour_count'],['同段连续切面交点组','same_segment_crossing_group_count'],['同段非局部交点组','same_segment_nonlocal_crossing_count'],['raw contour valid','raw_contour_valid'],['pipeline valid','pipeline_valid'],['pipeline reason','pipeline_reason']];
 document.getElementById('metrics').innerHTML='<table><thead><tr><th>指标</th><th>扰动前（绿色）</th><th>扰动后（紫色；invalid 灰色）</th></tr></thead><tbody>'+fields.map(([label,key])=>'<tr><td>'+label+'</td><td>'+metric(b[key])+'</td><td>'+metric(p[key])+'</td></tr>').join('')+'</tbody></table>';
 document.getElementById('change').textContent='面积变化 '+(100*r.recomputed_relative_area_change).toFixed(1)+'% · 选定轮廓质心移动 '+r.selected_contour_centroid_shift_mm.toFixed(2)+' mm · 闭环数量改变 '+r.closed_contour_count_changed;
 const pc=p.pipeline_valid?'#9333ea':'#94a3b8';
 Plotly.react('selected',[line2(b.polygon_common2,'扰动前选定','#16a34a'),line2(p.polygon_common2,p.pipeline_valid?'扰动后选定':'扰动后 invalid 诊断',pc),marks2(b.crossings2,'前：连续交点','#166534'),marks2(p.crossings2,'后：连续交点','#dc2626')],{title:'选定轮廓：共同基准平面投影',height:560,xaxis:{title:'基准 e1 / mm'},yaxis:{title:'基准 e2 / mm',scaleanchor:'x',scaleratio:1},margin:{t:50,l:60,b:50}},config);
 Plotly.react('all-lines',[line2(b.raw_common2,'前：全部交线','#94a3b8',false),line2(p.raw_common2,'后：全部交线','#f59e0b',false),line2(b.polygon_common2,'前：选定','#16a34a'),line2(p.polygon_common2,p.pipeline_valid?'后：选定':'后：invalid 诊断',pc)],{title:'全部交线与选定轮廓',height:560,xaxis:{title:'基准 e1 / mm'},yaxis:{title:'基准 e2 / mm',scaleanchor:'x',scaleratio:1},margin:{t:50,l:60,b:50}},config);
 Plotly.react('space',[line3(b.raw3,'前：全部交线','#cbd5e1',false),line3(p.raw3,'后：全部交线','#fbbf24',false),line3(b.polygon_xyz_mm,'前：选定','#16a34a'),line3(p.polygon_xyz_mm,p.pipeline_valid?'后：选定':'后：invalid 诊断',pc),marks3(b.crossings3,'前：连续交点','#166534'),marks3(p.crossings3,'后：连续交点','#dc2626')],{title:'真实 3D 坐标：保留平面位置与倾角差异',height:650,scene:{aspectmode:'data',camera:{eye:{x:2,y:2,z:1.5}},xaxis:{title:'Fluent x / mm'},yaxis:{title:'Fluent y / mm'},zaxis:{title:'Fluent z / mm'}},margin:{t:50,l:0,r:0,b:0}},config);
 document.getElementById('hits').textContent=JSON.stringify({continuous_polyline_plane_intersections:{baseline:b.same_segment_plane_intersections,perturbed:p.same_segment_plane_intersections},historical_sample_hits:{baseline:b.centerline_plane_hits_inside_selected_contour,perturbed:p.centerline_plane_hits_inside_selected_contour}},null,2);
 document.getElementById('note').textContent=r.note;
}
document.getElementById('select').innerHTML=D.diagnostics.map((r,i)=>'<option value="'+i+'">rank '+r.rank+' · '+escapeHtml(r.canonical_id)+' · #'+r.section_index_zero_based+' · '+r.variant+'</option>').join('');document.getElementById('select').onchange=e=>render(+e.target.value);render(0);
'''
    recheck_note = ''
    if args.recheck_of:
        blocked = sum(not record['perturbed']['pipeline_valid'] for record in records)
        recheck_note = f'<p class="notice">这是 {html.escape(args.recheck_of)} 历史 top{len(records)} 的 {html.escape(args.candidate_version)} 门控复核，不是新版本重排后的 top{len(records)}。当前 {blocked}/{len(records)} 个扰动被拦截，另 {len(records)-blocked} 个仍通过局部门控且需人工复核；门控通过不等于截面可信。</p>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P6 实际扰动前后轮廓</title><style>{CSS}</style><script src="assets/plotly-2.35.2.min.js"></script><header><a href="priority_review.html">← 优先审阅清单</a><h1>P6 实际扰动前后轮廓 · {html.escape(args.candidate_version)}</h1><p>这些图重算自保存的几何坐标，用于识别参考截面自身的轮廓选择不稳定；不读取 CFD 场标签，不自动修改点云输入 mask。</p>{recheck_note}<p class="notice">{html.escape(args.candidate_version)} 的 raw / pipeline 状态按当时定义保留。面积由各自切平面计算；共同平面投影仅用来比较形状。密集中心线采样命中不能直接当作独立穿越次数；连续交点以 × / 菱形显示。无效轮廓面积仅作诊断。</p></header><main><div class="card"><select id="select"></select><h2 id="case"></h2><p id="change" class="bad"></p><div id="metrics"></div></div><div class="grid"><div class="card"><div id="selected"></div></div><div class="card"><div id="all-lines"></div></div></div><div class="card"><div id="space"></div></div><div class="card"><h2>连续切面交点与历史采样记录</h2><pre id="hits"></pre><p id="note"></p></div></main><script>const D={dumps(payload)};{js}</script></html>'''
    args.out.write_text(page, encoding='utf-8')
    print(json.dumps({'page':str(args.out),'candidate_version':args.candidate_version,'diagnostics':len(records)},ensure_ascii=False))


if __name__=='__main__':
    main()
