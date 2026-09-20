"""Self-contained offline review page for the two unresolved root transitions."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
CAND = ROOT / "outputs/wss_v6_geometry_candidate_20260909/cases"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from wss_v5.bifurcation_user_confirmations import validate_confirmations


def render_review(output_dir, confirmations=None, source_root=None, case_page_prefix="../../cases/"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_root) if source_root else CAND.parent
    confirmed = (validate_confirmations(confirmations, BASE / "bifurcation_route_recovery", source_root)
                 if confirmations is not None else None)
    by_id = {c["canonical_id"]: c for c in confirmed["cases"]} if confirmed else {}
    cases = []
    for key in ["AAA__unruputer__LIU_XING_GUO", "AAA__unruputer__SUN_SHU_MING"]:
        source = BASE / "bifurcation_route_recovery" / (key + ".json")
        rec = json.loads(source.read_text())
        z = np.load(source_root / "cases" / key / "geometry.npz")
        variants = rec["junctions"][0]["descendant_routes"]
        paths = []
        for sid in range(7):
            mask = z["centerline_segment_id"] == sid
            p = z["centerline_xyz_mm"][mask]
            paths.append({"sid": sid, "xyz": p[::max(1, len(p)//180)].tolist()})
        wall = z["wall_xyz_mm"]
        wall = wall[np.unique(np.linspace(0, len(wall)-1, min(len(wall), 18000)).astype(int))]
        root = np.asarray(z["centerline_xyz_mm"])[np.where(z["centerline_segment_id"] == 0)[0][-1]]
        # Crop only the rendering, preserving original coordinates in evidence.
        wall = wall[np.linalg.norm(wall-root, axis=1) < 210]
        compact = []
        for v in variants:
            rr = v["scan"]["records"]
            first_two = next((r for r in rr if r["state"] == "two"), None)
            compact.append({"routes": v["routes"], "candidate": v["scan"]["candidate"], "first_two": first_two, "records": [{"s_mm": r["s_mm"], "state": r["state"], "center_mm": r["center_mm"]} for r in rr]})
        is_sun = "SUN_SHU_MING" in key
        cases.append({"canonical_id": rec["canonical_id"], "key": key, "source_sha256": rec["source_geometry_sha256"], "recovery_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "wall": np.round(wall, 3).tolist(), "paths": paths, "root": root.tolist(), "variants": compact, "spread_mm": rec["junctions"][0]["route_center_spread_mm"], "focus_mm": [48., 51.] if is_sun else [50., 75.], "reason": "四种路径在 48.75–49.50 mm 同时出现双轮廓，但沿程持续仅 0.75 mm，未满足 1 mm 连续证据。随后探针/轮廓归属长期不明确；138.25–147.75 mm 的远端转变已拒绝用作根部分叉。" if is_sun else "四条路径的首次持续分离分别为 71.00、97.75、74.75、94.00 mm，中心位置相差 23.39 mm。中间有 4–39.5 mm 的未定区间；无法用某一条终末分支的远端切平面作为共同根部分叉。", "review_target": "先看 48–51 mm 的局部壁面脊是否已经分隔左右管腔；黄色点为首次短暂双轮廓。" if is_sun else "先看 50–75 mm 的共同管腔及分隔脊；71/74.75 mm 是两个较早的候选，94/97.75 mm 候选明显更靠远端。"})
        cases[-1]["user_confirmation"] = by_id.get(rec["canonical_id"])
        z.close()
    data = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", data).replace("../../cases/", case_page_prefix)
    if confirmed:
        html = html.replace("两例分叉定位：最小人工确认包", "两例分叉定位：用户已确认")
        html = html.replace("只需确认这两例的局部分隔位置", "两例首次隔离位置已由用户确认")
        html = html.replace("这两例保留，是因为切平面上的路径归属不稳定。红点是自动算法拒绝接受的远端候选；黄色点提示优先查看的位置。", "这两例已按用户指定的首次有效独立轮廓事件确认，不再列为人工待办。绿色点为确认位置；原算法的持续分离分歧保留在下方。")
        html = html.replace("<div id=\"app\"></div>", "<p><a href=\"user_bifurcation_confirmations.json\">用户原话、确认规则与精确位置记录</a></p><div id=\"app\"></div>")
        (output_dir / "user_bifurcation_confirmations.json").write_text(json.dumps(confirmed, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "remaining_bifurcation_review.html").write_text(html)
    manifest = {"schema": "v6_remaining_bifurcation_review_v2", "html": "remaining_bifurcation_review.html", "html_sha256": hashlib.sha256(html.encode()).hexdigest(), "rendering": "offline_canvas_reference_wall_points_and_source_centerlines; rendering_only_wall_crop_210mm_from_graph_node; exact_evidence_in_json", "user_confirmed_case_count": len(by_id), "cases": [{k:v for k,v in c.items() if k not in ["wall", "paths", "root", "variants"]} for c in cases]}
    (output_dir / "remaining_bifurcation_review.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=BASE)
    ap.add_argument("--confirmations", type=Path)
    ap.add_argument("--source-root", type=Path, default=CAND.parent)
    ap.add_argument("--case-page-prefix", default="../../cases/")
    args = ap.parse_args()
    render_review(args.out, args.confirmations, args.source_root, args.case_page_prefix)
    print(args.out / "remaining_bifurcation_review.html")


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两例分叉定位：最小人工确认包</title>
<style>body{font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif;margin:0;background:#f3f6fa;color:#203049}main{max-width:1250px;margin:auto;padding:26px}h1{font-size:25px;margin:0 0 10px}h2{font-size:20px}p{margin:9px 0}.card{background:white;border:1px solid #dce4ed;border-radius:13px;padding:20px;margin:20px 0}.note{background:#fff4d6;border-left:4px solid #d58c00;padding:10px 14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}canvas{width:100%;height:440px;border:1px solid #dce4ed;border-radius:8px;background:#f8fbff;touch-action:none}.timeline{height:165px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid #e7ebf1;text-align:left}button{border:1px solid #c4d1e2;background:white;border-radius:6px;padding:6px 11px;cursor:pointer}.small{font-size:12px;color:#607189}.key{display:inline-block;margin:0 12px 0 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}code{overflow-wrap:anywhere}@media(max-width:800px){.grid{grid-template-columns:1fr}canvas{height:360px}}</style>
<main><h1>只需确认这两例的局部分隔位置</h1><p>10 例旧分叉搜索未定中，8 例已通过四条终末路径的一致性复查。这两例保留，是因为切平面上的路径归属不稳定。红点是自动算法拒绝接受的远端候选；黄色点提示优先查看的位置。</p><p class="small">下图可拖动旋转、滚轮缩放；离线可用。灰点来自原始参考壁面，彩线来自原始中心线。这里的切片转变中心既不代表壁面脊，也不会改写图节点、壁面分支字段或训练许可。</p><div id="app"></div></main>
<script>
const data=__DATA__,cols=['#3c4d64','#216cc4','#159889','#8758b8','#c34470','#dc8a27','#458338'];
const app=document.getElementById('app');
function escapeHTML(x){return String(x).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function confirmationHTML(c){if(!c.user_confirmation)return '';const q=c.user_confirmation,e=q.selected_event;return `<p class="note" style="background:#e3f5e9;border-color:#19886a"><strong>用户已确认首次隔离：</strong>${e.s_mm.toFixed(2)} mm；路径 ${e.routes.map(r=>r.join('→')).join(' / ')}；扫描步号 ${e.record_index_0based}（从 0 起）。<br>切平面中心 [${e.center_mm.map(x=>x.toFixed(6)).join(', ')}] mm。保留各路径首次事件与同最早 s 的全部坐标；表格中的持续分离候选仍为原算法证据。</p>`;}
data.forEach((c,ci)=>{const el=document.createElement('section');el.className='card';el.innerHTML=`<h2>${escapeHTML(c.canonical_id)}</h2>${confirmationHTML(c)}<p class="note">${c.user_confirmation?"保留的原算法分歧：":""}${c.reason}</p>${c.user_confirmation?"":`<p><strong>优先查看：</strong>${c.review_target}</p>`}<div class="grid"><div><canvas id="view${ci}" aria-label="原始几何三维视图"></canvas><p><button id="reset${ci}">重置视角</button> <label><input id="wall${ci}" type="checkbox" checked>显示壁面点</label></p><p class="small"><span class="key"><i class="dot" style="background:#eaa225"></i>首次出现双轮廓</span><span class="key"><i class="dot" style="background:#d73e39"></i>未接受的持续分离候选</span><span class="key"><i class="dot" style="background:#243a59"></i>旧图节点</span></p></div><div><table><thead><tr><th>终末路径</th><th>首次双轮廓</th><th>持续分离候选</th><th>one→two 间隔</th></tr></thead><tbody>${c.variants.map(v=>`<tr><td>${v.routes.map(r=>r.join('→')).join(' / ')}</td><td>${v.first_two.s_mm.toFixed(2)} mm</td><td>${v.candidate.s_mm.toFixed(2)} mm</td><td>${v.candidate.bracket_mm.map(x=>x.toFixed(2)).join('–')} mm</td></tr>`).join('')}</tbody></table><p>四条路径候选中心最大间距：<strong>${c.spread_mm.toFixed(2)} mm</strong>（共识上限 2 mm）。</p><canvas class="timeline" id="time${ci}" aria-label="四条路径沿程判定"></canvas><p class="small">沿程横轴：从旧图节点起、沿扩展路径的 s（mm）。<span style="color:#397fc0">蓝：共同单轮廓</span>；<span style="color:#19886a">绿：各处于独立轮廓</span>；<span style="color:#8b9aab">灰：无法唯一确定</span>。细红竖线为该路径的未接受候选。</p><p><a href="bifurcation_route_recovery/${c.key}.json">完整逐站数据与坐标</a> · <a href="../../cases/${c.key}.html">原始病例验收页</a></p><details><summary>来源校验</summary><p class="small">geometry.npz SHA256：<code>${c.source_sha256}</code><br>本轮分叉证据 SHA256：<code>${c.recovery_sha256}</code></p></details></div></div>`;app.appendChild(el);initView(c,ci);drawTimeline(c,ci);});
function initView(c,i){const cv=document.getElementById('view'+i),cx=cv.getContext('2d');let az=-.3,el=.3,zoom=1,drag=null,wall=true;const focus=c.variants[0].first_two.center_mm;const all=c.paths.flatMap(p=>p.xyz);const extent=Math.max(...all.map(p=>Math.hypot(p[0]-focus[0],p[1]-focus[1],p[2]-focus[2])));function project(p){let x=p[0]-focus[0],y=p[1]-focus[1],z=p[2]-focus[2],a=Math.cos(az)*x-Math.sin(az)*y,b=Math.sin(az)*x+Math.cos(az)*y,v=Math.cos(el)*z-Math.sin(el)*b,d=Math.sin(el)*z+Math.cos(el)*b;const scale=Math.min(cv.width,cv.height)*.44/extent*zoom;return [cv.width/2+a*scale,cv.height/2-v*scale,d];}function point(p,r,color){let q=project(p);cx.fillStyle=color;cx.beginPath();cx.arc(q[0],q[1],r,0,Math.PI*2);cx.fill();}function draw(){const dpr=devicePixelRatio||1;cv.width=cv.clientWidth*dpr;cv.height=cv.clientHeight*dpr;cx.clearRect(0,0,cv.width,cv.height);if(wall){cx.fillStyle='rgba(125,151,179,.18)';c.wall.forEach(p=>{let q=project(p);cx.fillRect(q[0],q[1],1.3*dpr,1.3*dpr);});}for(const p of c.paths){cx.strokeStyle=cols[p.sid];cx.lineWidth=2*dpr;cx.beginPath();p.xyz.forEach((x,j)=>{const q=project(x);j?cx.lineTo(q[0],q[1]):cx.moveTo(q[0],q[1]);});cx.stroke();}point(c.root,5*dpr,'#243a59');c.variants.forEach(v=>point(v.candidate.center_mm,5*dpr,'#d73e39'));c.variants.forEach(v=>point(v.first_two.center_mm,4*dpr,'#eaa225'));if(c.user_confirmation)point(c.user_confirmation.selected_event.center_mm,6*dpr,'#19886a');cx.fillStyle='#56708a';cx.font=`${12*dpr}px system-ui`;cx.fillText('拖动旋转 · 滚轮缩放',12*dpr,22*dpr);}cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);};cv.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.008;el+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw();};cv.onpointerup=()=>drag=null;cv.onwheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw();};document.getElementById('reset'+i).onclick=()=>{az=-.3;el=.3;zoom=1;draw();};document.getElementById('wall'+i).onchange=e=>{wall=e.target.checked;draw();};window.addEventListener('resize',draw);draw();}
function drawTimeline(c,i){const cv=document.getElementById('time'+i),cx=cv.getContext('2d');function draw(){const dpr=devicePixelRatio||1;cv.width=cv.clientWidth*dpr;cv.height=cv.clientHeight*dpr;cx.scale(dpr,dpr);let w=cv.clientWidth,h=cv.clientHeight,l=72,r=16,max=Math.max(...c.variants.map(v=>v.records.at(-1).s_mm)),scale=(w-l-r)/max;cx.clearRect(0,0,w,h);cx.font='11px system-ui';c.variants.forEach((v,j)=>{let y=22+j*28;cx.fillStyle='#566a80';cx.fillText(v.routes.map(x=>x.at(-1)).join(' / '),9,y+11);v.records.forEach(q=>{cx.fillStyle={one:'#397fc0',two:'#19886a',ambiguous:'#c9d2dc'}[q.state];cx.fillRect(l+q.s_mm*scale,y,Math.max(1,.25*scale),17);});cx.fillStyle='#d73e39';cx.fillRect(l+v.candidate.s_mm*scale,y-3,1.5,23);});cx.fillStyle='#5c7188';for(let n=0;n<=max;n+=20){let x=l+n*scale;cx.fillText(n.toString(),x-5,h-12);}cx.fillText('s / mm',w-45,h-12);}window.addEventListener('resize',draw);draw();}
</script></html>'''


if __name__ == "__main__":
    main()
