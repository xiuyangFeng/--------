"""Add independent review triage to the completed index without changing renderer.

Consumes the separately audited disagreement/sensitivity lists, copies their exact
bytes for provenance, and edits only this task's generated review output.
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "outputs/wss_v6_geometry_candidate_20260909"


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def case_link(cid):
    return f'<a href="cases/{esc(cid.replace("/", "__"))}.html">{esc(cid)}</a>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", type=Path, help="Optional independently audited P6 list")
    args = parser.parse_args()
    path = SOURCE / "review_priority_sections.json"
    data = json.loads(path.read_text())
    sensitivity = json.loads(args.sensitivity.read_text()) if args.sensitivity else None
    rendered = json.loads((HERE/'render_manifest.json').read_text())
    rendered_schemas = {case.get('source_schema') for case in rendered['cases']}
    expected_schema = data['source_candidate_schema']
    if rendered_schemas != {expected_schema} or rendered['preview_pages']:
        raise ValueError(f'Priority list / rendered pages schema mismatch: {expected_schema} vs {rendered_schemas}')
    if sensitivity and sensitivity.get('source_candidate_schema') != expected_schema:
        raise ValueError('Disagreement and stability lists refer to different candidate versions')
    source_code_hash = data.get('algorithm_code', {}).get('combined_sha256')
    if sensitivity and sensitivity.get('algorithm_code', {}).get('combined_sha256') != source_code_hash:
        raise ValueError('Disagreement and stability lists refer to different algorithm code')
    quality_cases = {r['canonical_id'] for r in [json.loads(p.read_text()) for p in (SOURCE / 'cases').glob('*/report.json')] if r.get('warnings')}
    disagreement_cases = {r['canonical_id'] for r in data['sections']}
    sensitive_cases = set()
    (HERE / path.name).write_bytes(path.read_bytes())
    if args.sensitivity:
        (HERE / args.sensitivity.name).write_bytes(args.sensitivity.read_bytes())
    rows = []
    for r in data["sections"]:
        cid = r["canonical_id"]
        idx = r["section_index_zero_based"]
        row_id = cid.replace("/", "__") + f"__s{idx}"
        rows.append(f'<tr id="{esc(row_id)}" data-search="{esc(cid.lower())}"><td>{case_link(cid)}</td><td>#{idx}</td><td>S{r["segment_id"]} / {esc(r.get("segment_role", ""))}</td><td>{fmt(r["s_local_mm"])}</td><td>{fmt(r["reference_area_mm2"])}</td><td>{fmt(r["pointcloud_area_mm2"])}</td><td class="bad">{fmt(100*r["relative_area_difference_pc_over_reference_minus_one"])}%</td><td>{esc(r["reference_valid"])} / {esc(r["pointcloud_valid"])}</td></tr>')
    version = expected_schema.rsplit('_', 1)[-1]
    pictures = [(f"candidate_{version}_YIN_YU_RONG_section74.png", f"{version} · YIN_YU_RONG #74：检查当前门控与原严重分歧站点的全部交线、选定轮廓。v1.2 曾在两源 valid 下给出 Aref=405.24、Apc=88.42 mm²；当前版本状态以图中 mask 为准。"),
                (f"candidate_{version}_GAO_FENG_SHAN_profiles.png", f"{version} · GAO_FENG_SHAN：点云截面覆盖较低，沿程点线保留无效缺口；Req 与原 Rmis 分开展示。"),
                (f"candidate_{version}_WANG_KUI_WU_topology.png", f"{version} · WANG_KUI_WU：此图为历史修正前状态；2026-09-11动态病例页已按用户确认修正seg3/seg5左右解剖标签，原CFD命名保留。"),
                (f"candidate_{version}_LOU_YANG_overview.png", f"{version} · LOU_YANG：几何代表例，检查壁面、七段中心线、五开口、CIA 起止。")]
    gallery = "".join(f'<figure><a href="{esc(filename)}"><img src="{esc(filename)}" alt="{esc(caption)}"></a><figcaption>{esc(caption)}</figcaption></figure>' for filename, caption in pictures if (HERE / filename).is_file())
    sensitivity_block = '<p class="muted">P6 独立敏感性清单尚未发布；不把未检查写成稳定。</p>'
    sensitivity_summary = "P6 清单待发布"
    if sensitivity:
        # Preserve all audited fields visibly. A concise source-dependent table
        # can be generated as an additional view without dropping source facts.
        records = sensitivity.get("sections", sensitivity.get("records", sensitivity.get("variants", [])))
        count = len(records)
        sensitive_cases = {r['canonical_id'] for r in records}
        union_count = len(quality_cases | disagreement_cases | sensitive_cases)
        sensitivity_summary = f"P6 {len(sensitive_cases)} 例、{count} 个变体对同源基准变化 >20%；与原 {len(quality_cases)} 例质量提示合并，三类图审优先病例共 {union_count} 例"
        sensitivity_rows = []
        for r in records:
            cid = r.get("canonical_id", "")
            body = {k: v for k, v in r.items() if k not in {"canonical_id", "report_json", "geometry_npz"}}
            sensitivity_rows.append(f'<tr><td>{case_link(cid) if cid else "—"}</td><td><pre>{esc(json.dumps(body, ensure_ascii=False, indent=2))}</pre></td></tr>')
        sensitivity_block = f'<p><a href="{esc(args.sensitivity.name)}">完整独立 P6 清单与定义</a>。原始闭环有效、pipeline 有效、comparison 有效分别保留；不把它们互相替代。</p><div class="table-wrap"><table><thead><tr><th>病例</th><th>审计记录</th></tr></thead><tbody>{"".join(sensitivity_rows)}</tbody></table></div>'
    styles = '''body{margin:0;background:#f1f5f9;color:#172033;font:15px/1.6 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:28px}h1{font-size:27px}h2{font-size:20px}a{color:#1d4ed8}.card{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin:18px 0}.bad{color:#b91c1c}.muted{color:#64748b}.lead{border-left:5px solid #dc2626}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}th{background:#f8fafc;position:sticky;top:0}.table-wrap{overflow:auto;max-height:700px}pre{white-space:pre-wrap;max-width:900px}input{padding:9px;font:inherit;width:min(500px,90%)}figure{margin:24px 0}img{max-width:100%;border:1px solid #cbd5e1}figcaption{padding:10px;background:#f8fafc}'''
    diagnostic_link = '<p><a href="priority_stability.html"><strong>打开 v1.2 历史 top6 实际扰动前后轮廓</strong></a>：其中 rank1 为最大跳变，rank3 展示同段远处再次穿越切面的问题；历史 pipeline 状态不替代新版本判定。</p>' if (HERE/'priority_stability.html').is_file() else ''
    recheck_name = f'priority_stability_{version}.html'
    recheck_manifest = SOURCE / f'priority_stability_diagnostics_{version.replace(".", "p")}' / 'manifest.json'
    if (HERE/recheck_name).is_file() and recheck_manifest.is_file():
        recheck = json.loads(recheck_manifest.read_text())
        if recheck['applied_gate_schema'] != expected_schema or recheck.get('algorithm_code', {}).get('combined_sha256') != source_code_hash:
            raise ValueError('Historical top-six recheck uses a different gate schema or code')
        count_rechecked = len(recheck['diagnostics'])
        count_blocked = sum(not r['perturbed_gate'] for r in recheck['diagnostics'])
        diagnostic_link = f'<p><a href="{esc(recheck_name)}"><strong>{esc(version)} 对历史 top{count_rechecked} 的当前门控复核</strong></a>：{count_blocked}/{count_rechecked} 个扰动被拦截，另 {count_rechecked-count_blocked} 个仍通过局部门控且待审。显示连续折线与切面的真实交点，invalid 轮廓灰色保留作诊断。这是历史案例复核，不是当前版本 top{count_rechecked} 重排。</p>' + diagnostic_link
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V6 几何优先审阅：分歧与敏感性</title><style>{styles}</style><main><a href="index.html">← 全队列索引</a><h1>优先检查：局部门控通过仍存在严重截面分歧</h1>
<div class="card lead"><p><strong>{data["case_count"]} 例、{data["section_count"]} 个站点</strong>同时满足两源局部 pipeline valid，但 |Apc/Aref−1| &gt; {100*data["threshold_absolute_relative_area_difference"]:.0f}%。{esc(sensitivity_summary)}。</p><p>这些是人工审阅优先级，未用于修改点云输入 mask，也不说明面几何参考必然正确。面拓扑只用于参考审计，裸点云候选按自己的门控判断。新特征仍未接入训练。</p></div>
<div class="card"><h2>截面分歧清单</h2><p>打开病例后进入“截面逐站检查”，选择表中 <strong>#站点</strong>（与页面一致，0 起算）。对照橙色全部交线、绿色面几何轮廓、紫色点云轮廓及壁面分支映射。主状态“ready for visual review”不代表几何已通过人工验收。</p><input id="search" placeholder="搜索病例"><div class="table-wrap"><table id="section-table"><thead><tr><th>病例</th><th>站点</th><th>分支</th><th>s/mm</th><th>Aref/mm²</th><th>Apc/mm²</th><th>Apc/Aref−1</th><th>ref/PC valid</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p><a href="review_priority_sections.json">完整分歧 JSON 与阈值定义</a></p></div>
<div class="card"><h2>P6 极敏感截面</h2>{diagnostic_link}{sensitivity_block}</div><div class="card"><h2>实际浏览器图像</h2>{gallery}</div>
<div class="card muted"><p>来源版本：{esc(data['source_candidate_schema'])} · 分歧清单 SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}</p><p>页面由独立 publish_priority_review.py 后处理；不修改候选 NPZ、原算法或正在运行的 renderer。</p></div></main><script>document.getElementById('search').oninput=e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('#section-table tbody tr').forEach(r=>r.hidden=!r.dataset.search.includes(q))}}</script></html>'''
    (HERE / "priority_review.html").write_text(page, encoding="utf-8")
    panel = f'''<!-- PRIORITY_REVIEW_START --><div class="card" style="border-left:5px solid #dc2626"><h2>须先审阅：两源 valid 仍严重不一致</h2><p>{data["case_count"]} 例、{data["section_count"]} 个截面面积差 &gt; {100*data['threshold_absolute_relative_area_difference']:.0f}%；{esc(sensitivity_summary)}。这些异常未自动并入输入 mask。</p><p><a href="priority_review.html"><strong>打开严重分歧 / 敏感性清单与实际图像</strong></a> · 包含 YIN_YU_RONG #74 等明确站点。病例原有 ready 状态不等于人工验收通过。</p></div><!-- PRIORITY_REVIEW_END -->'''
    index_path = HERE / "index.html"
    index = index_path.read_text()
    index = re.sub(r'<!-- PRIORITY_REVIEW_START -->.*?<!-- PRIORITY_REVIEW_END -->', '', index, flags=re.S)
    if '<main>' not in index:
        raise ValueError('index main insertion point missing')
    index_path.write_text(index.replace('<main>', '<main>' + panel, 1), encoding="utf-8")
    summary = {"source_candidate_schema": expected_schema, "source_algorithm_code_sha256": source_code_hash,
               "section_disagreement_cases": data["case_count"], "section_disagreements": data["section_count"],
               "quality_flag_cases": len(quality_cases), "stability_sensitive_cases": len(sensitive_cases),
               "priority_union_cases": len(quality_cases | disagreement_cases | sensitive_cases),
               "sensitivity_source": str(args.sensitivity) if args.sensitivity else None,
               "candidate_data_modified": False, "renderer_modified": False, "index_priority_panel": True}
    (HERE / "priority_review_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
