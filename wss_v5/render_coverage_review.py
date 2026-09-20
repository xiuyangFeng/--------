"""Offline report for coverage recovery with per-field missingness.

Consumes coverage_manifest/coverage.json directly. This is a report of a
geometry candidate; it does not create a training view or change permissions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

try:
    from .bifurcation_user_confirmations import apply_confirmations, validate_confirmations
except ImportError:  # direct execution from the repository's wss_v5 directory
    from bifurcation_user_confirmations import apply_confirmations, validate_confirmations


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, keys=None):
    keys = keys or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _field_coverage(z):
    # Mask names are preserved verbatim for downstream feature lookup.
    result = {}
    for key in sorted(z):
        # Include every stored boolean validity/mask field, not only wall
        # mapping masks. This keeps section, point-cloud, profile and every
        # derived along-path feature on one explicit per-field mask table.
        a = np.asarray(z[key])
        if a.dtype != np.bool_ or not (key.endswith("_valid") or key.endswith("_mask")):
            continue
        result[key] = {"valid": int(a.sum()), "total": int(a.size),
                       "fraction": float(a.mean()) if a.size else None}
    return result


def _case_payload(root, row, confirmation):
    cid = row["canonical_id"]
    key = cid.replace("/", "__")
    case = root / "cases" / key
    report = json.loads((case / "report.json").read_text())
    coverage = json.loads((case / "coverage.json").read_text())
    source = Path(report["coverage_source"]["directory"])
    with np.load(case / "geometry.npz", allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    with np.load(source / "geometry.npz", allow_pickle=False) as f:
        old = {k: f[k] for k in ["section_valid", "pc_section_valid", "section_area_mm2",
                "pc_section_area_mm2", "section_tangent", "section_polygon_offsets", "section_polygon_xyz_mm",
                "pc_section_polygon_offsets", "pc_section_polygon_xyz_mm"]}
    reference_before = int(old["section_valid"].sum())
    reference_after = int(z["section_valid"].sum())
    pc_before = int(old["pc_section_valid"].sum())
    pc_after = int(z["pc_section_valid"].sum())
    for field, actual in (("reference_before", reference_before), ("reference_after", reference_after),
                          ("pointcloud_before", pc_before), ("pointcloud_after", pc_after)):
        if coverage[field] != actual or row[field] != actual:
            raise ValueError(f"coverage count does not match arrays: {cid} {field}")
    fields = _field_coverage(z)
    n = len(z["section_valid"])
    low = reference_after/n < .6 or pc_after/n < .5
    brief = {"canonical_id": cid, "manual_reasons": "", "root_position_user_confirmed": bool(confirmation),
             "confirmed_root_s_mm": confirmation["selected_event"]["s_mm"] if confirmation else None,
             "n_sections": n, "reference_before": reference_before, "reference_after": reference_after,
             "pointcloud_before": pc_before, "pointcloud_after": pc_after,
             "recovered_reference": int((~old["section_valid"] & z["section_valid"]).sum()),
             "withheld_reference": int((old["section_valid"] & ~z["section_valid"]).sum()),
             "recovered_pointcloud": int((~old["pc_section_valid"] & z["pc_section_valid"]).sum()),
             "withheld_pointcloud": int((old["pc_section_valid"] & ~z["pc_section_valid"]).sum()),
             "reference_coverage": reference_after/n, "pointcloud_coverage": pc_after/n,
             "reference_wall_coverage": fields["wall_map_valid"]["fraction"],
             "pointcloud_wall_coverage": fields["wall_map_pc_valid"]["fraction"],
             "low_coverage_diagnostic_only": low, "case_level_feature_disable": False,
             "retain_all_case_supervision": True, "training_allowed": False,
             "recommendation": "保留病例与全部监督；几何特征逐字段使用有效 mask，缺失保留"}
    events = []
    for source_name, prefix in (("reference", "section_"), ("pointcloud", "pc_section_")):
        for k in np.flatnonzero(~old[prefix+"valid"] & z[prefix+"valid"]):
            center = z["section_center_mm"][k].astype(float)
            t = old["section_tangent"][k].astype(float)
            axis = np.eye(3)[np.argmin(abs(t))]
            u = np.cross(t, axis); u /= np.linalg.norm(u); v = np.cross(t, u)
            def polygon(arrays):
                offsets = arrays[prefix+"polygon_offsets"]
                p = arrays[prefix+"polygon_xyz_mm"][offsets[k]:offsets[k+1]].astype(float)-center
                return np.column_stack((p@u, p@v)).round(3).tolist()
            events.append({"source": source_name, "station": int(k),
                           "segment": int(z["section_segment_id"][k]), "s_mm": float(z["section_s_local_mm"][k]),
                           "before_area_diagnostic": float(old[prefix+"area_mm2"][k]),
                           "after_area": float(z[prefix+"area_mm2"][k]),
                           "before_polygon": polygon(old), "after_polygon": polygon(z),
                           "before_reason": "原截面无效，仅诊断对照"})
    payload = {**brief, "key": key, "segment": z["section_segment_id"], "s_mm": z["section_s_local_mm"],
               "reference_before_area": np.where(old["section_valid"], old["section_area_mm2"], np.nan),
               "reference_after_area": np.where(z["section_valid"], z["section_area_mm2"], np.nan),
               "pointcloud_before_area": np.where(old["pc_section_valid"], old["pc_section_area_mm2"], np.nan),
               "pointcloud_after_area": np.where(z["pc_section_valid"], z["pc_section_area_mm2"], np.nan),
               "reference_missing": ~z["section_valid"], "pointcloud_missing": ~z["pc_section_valid"],
               "reference_reason": z["section_reason"], "pointcloud_reason": z["pc_section_reason"],
               "field_coverage": fields, "events": events, "user_confirmation": confirmation,
               "coverage_counts": coverage["counts"], "pointcloud_counts": coverage["pc_counts"]}
    return brief, payload


def build_report(root, out, review_source, resolution_path):
    root, out, review_source = Path(root), Path(out), Path(review_source)
    manifest = json.loads((root / "coverage_manifest.json").read_text())
    audit_path = root / "audit_coverage_geometry.json"
    audit = json.loads(audit_path.read_text())
    expected = {r["canonical_id"] for r in manifest["cases"]}
    audit_summary = audit.get("summary", {})
    if (not audit_summary.get("all_mechanical_checks_pass")
            or audit_summary.get("audited_cases") != len(expected)
            or Path(audit.get("candidate", "")).resolve() != root.resolve()
            or {r["canonical_id"] for r in audit.get("cases", [])} != expected):
        raise ValueError("a passing independent audit of this exact coverage cohort is required")
    source = Path(manifest["source"])
    source_manifest = json.loads((source / "refinement_manifest.json").read_text())
    if sha(source / "refinement_manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("coverage source manifest changed")
    resolution_path = Path(resolution_path)
    confirmations = validate_confirmations(review_source / "user_bifurcation_confirmations.json",
            resolution_path.parent / "bifurcation_route_recovery", Path(source_manifest["source"]))
    resolution = apply_confirmations(json.loads(resolution_path.read_text()), confirmations)
    remaining = [r["canonical_id"] for r in resolution["cases"] if any(
        not b["resolved_by_route_consensus"] and not b.get("resolved_by_user_confirmation")
        for b in (r.get("bifurcation_recovery") or []))]
    if remaining:
        raise ValueError(f"remaining bifurcation decisions need explicit report support: {remaining}")
    confirmed = {r["canonical_id"]: r for r in confirmations["cases"]}
    actions, cases = [], []
    for row in manifest["cases"]:
        brief, payload = _case_payload(root, row, confirmed.get(row["canonical_id"]))
        actions.append(brief); cases.append(payload)
    totals = {key: sum(c[key] for c in actions) for key in ["n_sections", "reference_before", "reference_after",
              "pointcloud_before", "pointcloud_after", "recovered_reference", "withheld_reference",
              "recovered_pointcloud", "withheld_pointcloud"]}
    low = [c["canonical_id"] for c in actions if c["low_coverage_diagnostic_only"]]
    summary = {"schema": "v6_coverage_review_v1", "source": str(root.resolve()),
               "baseline_source": str(source.resolve()), "job_id": manifest["job_id"],
               "cases": len(cases), "manual_case_count": 0, "manual_cases": [],
               "user_confirmed_case_count": len(confirmed), "user_confirmed_cases": list(confirmed),
               "counts": totals, "low_coverage_diagnostic_cases": low,
               "case_level_feature_disable": False, "retain_all_case_supervision": True,
               "missing_value_policy": "per_field_validity_not_case_level_disable",
               "independent_audit": audit_summary, "audit_sha256": sha(audit_path),
               "training_allowed": False}
    out.mkdir(parents=True, exist_ok=True)
    for filename in ["remaining_bifurcation_review.html", "remaining_bifurcation_review.json", "user_bifurcation_confirmations.json"]:
        if (review_source/filename).resolve() != (out/filename).resolve():
            shutil.copy2(review_source/filename, out/filename)
    if review_source.resolve() != out.resolve():
        shutil.copytree(review_source/"bifurcation_route_recovery", out/"bifurcation_route_recovery", dirs_exist_ok=True)
    asset = Path(__file__).resolve().parents[1]/"docs/03-汇报材料/WSS_V6_新几何验收_20260909/assets/plotly-2.35.2.min.js"
    shutil.copy2(asset, out/"plotly.min.js")
    write_json(out/"summary.json", summary)
    write_csv(out/"case_actions.csv", actions)
    write_csv(out/"human_decisions.csv", [], ["canonical_id", "manual_reasons"])
    field_rows = [{"canonical_id": c["canonical_id"], "field_mask": k, **v}
                  for c in cases for k, v in c["field_coverage"].items()]
    write_csv(out/"field_coverage.csv", field_rows)
    doc = TEMPLATE.replace("__DATA__", json.dumps(clean(cases), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    doc = doc.replace("__SUMMARY__", json.dumps(summary, ensure_ascii=False).replace("</", "<\\/"))
    (out/"index.html").write_text(doc)
    md = ["# 几何特征覆盖优化与确认结果（2026-09-16）", "",
          f"**{len(cases)} 例保留；新增人工待办 0 例，两例首次隔离位置已按用户确认落实。** [交互审查台](index.html) · [病例动作](case_actions.csv) · [逐字段覆盖](field_coverage.csv) · [两例确认记录](remaining_bifurcation_review.html)", "",
          f"本轮以 v3 候选为起点（作业 {manifest['job_id']}），参考有效截面 {totals['reference_before']:,} → {totals['reference_after']:,}；点云 {totals['pointcloud_before']:,} → {totals['pointcloud_after']:,}，共 {totals['n_sections']:,} 个站点。",
          f"实际新增恢复参考 {totals['recovered_reference']} 站、点云 {totals['recovered_pointcloud']} 站；更严格检查同时屏蔽原有效参考 {totals['withheld_reference']} 站、点云 {totals['withheld_pointcloud']} 站。恢复与屏蔽分开计数。", "",
          "算法以连续中心线边与切平面求交，减少离散点接近切面的误判；缺失参考站经重选切面、独立倾角与最终沿程位移复核。点云增加仅由点构建的轮廓候选，再验证参考一致性、所属分支和减半密度稳定性。", "",
          "所有病例使用相同几何字段，并按每个字段各自的有效 mask 表示可用范围。截面积有效不自动代表坡度或上下游信息有效；派生字段不跨无效区间。保留病例及其全部原有监督，缺少几何特征不删除病例或监督标签。", "",
          "旧的整例低覆盖禁用建议已撤回。参考覆盖 60% / 点云 50% 只作诊断提示，不决定病例去留，也不要求用户逐站放行。低覆盖诊断病例：" + ("、".join(low) if low else "无") + "。", "",
          "两例用户确认仅涉及根部首次隔离事件：LIU_XING_GUO 71.00 mm、SUN_SHU_MING 48.75 mm；s 从旧根图节点沿扩展路径起算，精确坐标、原话及原算法分歧保存在确认侧车。训练许可仍为 false。", "",
          "位移不一致可能来自真实急变或提取误差；保留缺失不等于认定源解剖违规。独立审计验证机械几何一致性，不代替临床解剖批准。该候选尚未接入训练，保留监督是本轮候选使用约定。", "",
          f"候选：`{root}`；基线：`{source}`。来源、参数、作业与病例名单见 `coverage_manifest.json`，逐站依据见各例 `coverage.json`；独立审计 `audit_coverage_geometry.json` 已核对相同病例集合。"]
    (out/"README.md").write_text("\n".join(md)+"\n")
    return summary


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>几何覆盖优化与已确认位置</title>
<style>body{font:15px/1.65 system-ui;margin:24px;background:#f3f6fa;color:#203049}h1{font-size:25px}main,.card{background:white;border:1px solid #dce4ed;border-radius:12px;padding:18px;margin:12px 0}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{flex:1;min-width:150px}select,button{font:inherit;padding:7px;margin:4px}.row{display:grid;grid-template-columns:1.3fr 1fr;gap:15px}#curve,#contour{height:440px}#missing{height:180px}#status{white-space:pre-wrap;background:#e9f1f5;padding:12px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e1e6ec;padding:6px;text-align:left}a{color:#086eac}small{color:#596d82}@media(max-width:850px){.row{grid-template-columns:1fr}}</style>
<h1>几何覆盖优化与已确认位置</h1><p id="intro"></p><p><a href="remaining_bifurcation_review.html">查看两例用户确认与原算法分歧</a> · <a href="README.md">结果说明</a> · <a href="case_actions.csv">病例动作</a> · <a href="field_coverage.csv">逐字段覆盖</a></p><div id="cards" class="cards"></div>
<main><p><select id="case"></select><select id="branch"></select><select id="event"></select></p><div id="status"></div><div class="row"><div><div id="curve"></div><div id="missing"></div></div><div><div id="contour"></div><small>仅展示本轮从无效恢复的截面；灰色原轮廓和原面积只作诊断。缺失保留为空，不以平滑猜测替代。</small></div></div><details><summary>本例逐字段有效覆盖</summary><table><thead><tr><th>字段 mask</th><th>有效 / 总数</th><th>覆盖</th></tr></thead><tbody id="fields"></tbody></table></details></main>
<script src="plotly.min.js"></script><script>const D=__DATA__,S=__SUMMARY__;const $=id=>document.getElementById(id);let cur;
$('intro').textContent=`${S.cases} 例全部保留，剩余人工待办 ${S.manual_case_count} 例，用户已确认 ${S.user_confirmed_case_count} 例根部位置。几何特征逐字段使用 mask；低覆盖仅诊断，保留病例与全部监督。候选尚未接入训练。`;
$('cards').innerHTML=[['新增参考恢复',S.counts.recovered_reference],['新增点云恢复',S.counts.recovered_pointcloud],['低覆盖诊断病例',S.low_coverage_diagnostic_cases.length],['剩余人工待办',S.manual_case_count]].map(([k,v])=>`<div class="card">${k}<h2>${v}</h2></div>`).join('');
D.forEach(c=>$('case').add(new Option(c.canonical_id,c.key)));
function show(){cur=D.find(c=>c.key===$('case').value);$('branch').innerHTML='';[...new Set(cur.segment)].forEach(s=>$('branch').add(new Option(`分支 ${s}`,s)));$('event').innerHTML='';$('event').add(new Option('选择本轮恢复截面',''));cur.events.forEach((e,i)=>$('event').add(new Option(`${e.source==='reference'?'参考':'点云'} 站${e.station} · 分支${e.segment}`,i)));$('status').textContent=`${cur.canonical_id}
参考 ${cur.reference_before} → ${cur.reference_after}；点云 ${cur.pointcloud_before} → ${cur.pointcloud_after}；总站数 ${cur.n_sections}。
${cur.low_coverage_diagnostic_only?'覆盖低于诊断提示线，仍保留病例与各字段有效区域。':'按各字段 mask 使用几何候选。'}
${cur.root_position_user_confirmed?'根部首次隔离位置已由用户确认，s = '+cur.confirmed_root_s_mm+' mm。':''}`;$('fields').innerHTML=Object.entries(cur.field_coverage).map(([k,v])=>`<tr><td>${k}</td><td>${v.valid} / ${v.total}</td><td>${(v.fraction*100).toFixed(1)}%</td></tr>`).join('');draw();Plotly.purge('contour')}
function draw(){let sid=Number($('branch').value),ix=cur.segment.map((v,i)=>v===sid?i:-1).filter(i=>i>=0);let traces=[['参考·v3',cur.reference_before_area,'#8796a4','dash'],['参考·本轮',cur.reference_after_area,'#137f98','solid'],['点云·v3',cur.pointcloud_before_area,'#bea07c','dash'],['点云·本轮',cur.pointcloud_after_area,'#d78426','solid']].map(([name,y,color,dash])=>({name,x:ix.map(i=>cur.s_mm[i]),y:ix.map(i=>y[i]),mode:'lines+markers',marker:{size:3},line:{color,dash},connectgaps:false}));Plotly.react('curve',traces,{title:'沿程截面积：v3 与本轮覆盖优化',xaxis:{title:'分支局部 s（mm）'},yaxis:{title:'面积（mm²）',type:'log'},legend:{orientation:'h'},margin:{t:50,l:70,b:50,r:10}},{responsive:true});let mt=[['参考缺失',cur.reference_missing,cur.reference_reason,1,'#ba4b4b'],['点云缺失',cur.pointcloud_missing,cur.pointcloud_reason,0,'#c18532']].map(([name,mask,reason,y,color])=>{const j=ix.filter(i=>mask[i]);return{name,x:j.map(i=>cur.s_mm[i]),y:j.map(()=>y),text:j.map(i=>reason[i]),mode:'markers',marker:{symbol:'x',color,size:7},hovertemplate:'s=%{x:.2f} mm<br>%{text}<extra>'+name+'</extra>'}});Plotly.react('missing',mt,{title:'本轮缺失位置（不代表源解剖违规）',xaxis:{title:'分支局部 s（mm）'},yaxis:{tickvals:[0,1],ticktext:['点云','参考'],range:[-.5,1.5]},showlegend:false,margin:{t:45,l:70,b:45,r:10}},{responsive:true})}
function event(){if($('event').value==='')return;const e=cur.events[Number($('event').value)];$('branch').value=e.segment;draw();const trace=(p,name,color)=>({name,x:p.length?[...p.map(q=>q[0]),p[0][0]]:[],y:p.length?[...p.map(q=>q[1]),p[0][1]]:[],mode:'lines',line:{color}});Plotly.react('contour',[trace(e.before_polygon,'原无效轮廓（诊断）','#a6afb7'),trace(e.after_polygon,'本轮恢复','#137f98')],{title:`${e.source==='reference'?'参考':'点云'}站 ${e.station}：恢复面积 ${e.after_area.toFixed(2)} mm²`,xaxis:{title:'局部 x（mm）'},yaxis:{title:'局部 y（mm）',scaleanchor:'x'},margin:{t:60,b:45,l:55,r:10}},{responsive:true})}
$('case').onchange=show;$('branch').onchange=draw;$('event').onchange=event;show();
</script></html>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review-source", type=Path, required=True)
    ap.add_argument("--resolution", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(build_report(a.root, a.out, a.review_source, a.resolution), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
