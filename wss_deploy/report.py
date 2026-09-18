"""Report export and self-contained viewer; statistics remain on prediction points."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .paths import STATIC_DIR

META_START = '<!--WSS_META_START-->'
META_END = '<!--WSS_META_END-->'


def interpolate_to_vertices(pts: np.ndarray, values: np.ndarray, vertices: np.ndarray,
                            sigma_mm: float = 0.5, k: int = 8,
                            max_dist_mm: float = 1.5) -> np.ndarray:
    """Gaussian display interpolation; unsupported vertices are NaN, never extrapolated."""
    pts = np.asarray(pts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
        raise ValueError("Interpolation points must be a non-empty finite (N, 3) array")
    if values.shape != (len(pts),) or not np.isfinite(values).all():
        raise ValueError("Interpolation requires one finite value per prediction point")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite (N, 3) coordinates")
    if not np.isfinite(sigma_mm) or sigma_mm <= 0 or not np.isfinite(max_dist_mm) or max_dist_mm <= 0:
        raise ValueError("Interpolation sigma and coverage distance must be finite and positive")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Interpolation neighbour count must be a positive integer")
    count = min(k, len(pts))
    d, i = cKDTree(pts).query(vertices, k=count)
    d, i = d.reshape(len(vertices), count), i.reshape(len(vertices), count)
    # Subtract the smallest squared distance to avoid underflow for very small sigma.
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = -0.5 * ((d / sigma_mm) ** 2 - (d[:, :1] / sigma_mm) ** 2)
    weights = np.exp(exponent)
    weights[d > max_dist_mm] = 0.0
    den = weights.sum(axis=1)
    out = np.full(len(vertices), np.nan, dtype=np.float64)
    np.divide((weights * values[i]).sum(axis=1), den, out=out, where=den > 0)
    return out.astype(np.float32)


def nearest_label(pts: np.ndarray, labels: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    _, i = cKDTree(pts).query(vertices, k=1)
    return np.asarray(labels)[i]


def write_vtp(path: Path, vertices, faces, point_data: dict) -> bool:
    try:
        import pyvista as pv
    except ImportError:
        return False
    cells = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=np.float32), cells)
    for key, value in point_data.items():
        mesh.point_data[key] = np.asarray(value)
    mesh.save(str(path))
    return True


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _script_json(value) -> str:
    """Safe inside a script element even for user-controlled case names and flags."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def branch_unroll_data(cloud: dict) -> list[dict]:
    """Each segment owns a separate plot, including segments with overlapping s ranges."""
    segment = np.asarray(cloud["segment"])
    s = np.asarray(cloud["s_from_root_mm"])
    theta = np.asarray(cloud["theta_rad"])
    if segment.ndim != 1 or s.shape != segment.shape or theta.shape != segment.shape:
        raise ValueError("Unroll coordinates and branch IDs must have matching one-dimensional shapes")
    if not np.isfinite(segment).all() or not np.isfinite(s).all() or not np.isfinite(theta).all():
        raise ValueError("Unroll coordinates and branch IDs must be finite")
    return [{"segment_id": int(sid), "n_points": int(np.sum(segment == sid)),
             "s_min_mm": float(s[segment == sid].min()), "s_max_mm": float(s[segment == sid].max())}
            for sid in np.unique(segment)]


def update_html_meta(out_path: Path, meta: dict) -> None:
    """Refresh only embedded metadata after timing the first complete HTML write.

    This final rewrite is deliberately outside first-export timings. Callers should
    describe the small finalization overhead separately in their timing protocol.
    """
    out_path = Path(out_path)
    content = out_path.read_text(encoding="utf-8")
    before, marker, tail = content.partition(META_START)
    _, end, after = tail.partition(META_END)
    if not marker or not end:
        raise ValueError("Report metadata markers were not found")
    replacement = '<script id="wss-report-meta" type="application/json">' + _script_json(meta) + '</script>'
    tmp = out_path.with_name(out_path.name + ".meta.tmp")
    tmp.write_text(before + META_START + replacement + META_END + after, encoding="utf-8")
    tmp.replace(out_path)


def build_html(out_path: Path, meta: dict, mesh: dict, cloud: dict, centerline: dict) -> None:
    three = (STATIC_DIR / "three.min.js").read_text(encoding="utf-8")
    orbit = (STATIC_DIR / "OrbitControls.js").read_text(encoding="utf-8")
    arrays = {
        "mv": _b64(mesh["vertices"], np.float32), "mf": _b64(mesh["faces"], np.uint32),
        "mw": _b64(mesh["wss"], np.float32), "ms": _b64(mesh["segment"], np.int32),
        "pv": _b64(cloud["pts"], np.float32), "pw": _b64(cloud["wss"], np.float32),
        "ps": _b64(cloud["segment"], np.int32), "p_s": _b64(cloud["s_from_root_mm"], np.float32),
        "p_th": _b64(cloud["theta_rad"], np.float32), "p_r": _b64(cloud["radius_mm"], np.float32),
        "p_dj": _b64(cloud["dist_to_junction_mm"], np.float32),
        "cv": _b64(centerline["xyz"], np.float32), "cr": _b64(centerline["radius_mm"], np.float32),
        "ce": _b64(centerline["edges"], np.uint32), "cs": _b64(centerline["segment"], np.int32),
        "unroll_branches": branch_unroll_data(cloud),
    }
    # Metadata is inserted last so user text cannot introduce another template substitution.
    html = (TEMPLATE.replace("__THREE__", three).replace("__ORBIT__", orbit)
            .replace("__ARRAYS__", _script_json(arrays)).replace("__META__", _script_json(meta)))
    Path(out_path).write_text(html, encoding="utf-8")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSS 预测报告</title>
<style>
:root{--bg:#f3f6fa;--card:#fff;--line:#dce4ed;--ink:#203049;--muted:#5f7086;--acc:#0b6bcb}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg)}
header{display:flex;align-items:center;gap:16px;padding:12px 18px;background:#fff;border-bottom:1px solid var(--line);flex-wrap:wrap}h1{font-size:18px;margin:0}.sub,small{color:var(--muted);font-size:12px}.actions{margin-left:auto;display:flex;gap:8px}
button,select,input{font:inherit}button{padding:5px 10px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}button.on{background:var(--acc);color:white}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
main{display:grid;grid-template-columns:minmax(0,1fr) 445px;height:calc(100vh - 75px);min-height:520px}#view{position:relative;background:#eef2f7;min-width:0;overflow:hidden}#view>canvas{display:block}#labels div{position:absolute;transform:translate(-50%,-120%);background:#fffe;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12px;pointer-events:none;white-space:nowrap}
#tip{position:absolute;left:12px;bottom:12px;max-width:80%;background:#fffffff0;border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;white-space:pre-line;z-index:2}#ctrl{position:absolute;top:12px;left:12px;background:#fffffff0;border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;display:grid;gap:6px;width:280px;z-index:2;box-shadow:0 3px 12px #20304918}#ctrl-head{display:flex;align-items:center;gap:8px}#ctrl-head strong{font-size:13px}#ctrl-toggle{margin-left:auto;padding:2px 7px;font-size:11px}#ctrl.collapsed{width:auto;min-width:132px}#ctrl.collapsed #ctrl-body{display:none}#ctrl.collapsed #ctrl-toggle{margin-left:0}#ctrl label{display:flex;align-items:center;gap:5px}#ctrl input[type=range]{min-width:0;flex:1}#fixedmax{width:64px}.ctrl-summary{color:var(--muted);border-top:1px solid var(--line);padding-top:5px;line-height:1.35}.highlight-control output{min-width:34px;text-align:right;font-variant-numeric:tabular-nums}.metric-bars{display:grid;gap:7px;margin:7px 0 4px}.metric-bar{display:grid;grid-template-columns:92px minmax(0,1fr) auto;align-items:center;gap:7px;font-size:11px}.metric-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.metric-track{height:9px;background:#e9eef5;border-radius:8px;overflow:hidden}.metric-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#4b83c4,#c0392b);min-width:2px}.metric-fill.low{background:#4b83c4}.metric-fill.high{background:#df8b36}.metric-fill.very-high{background:#c0392b}.branch-chart .metric-track{height:8px}.branch-chart .metric-bar{grid-template-columns:82px minmax(0,1fr) 46px}.branch-chart .metric-fill{background:#547eb5}#cbar{position:absolute;right:70px;top:45px;width:18px;height:190px;border:1px solid #999;z-index:1}#cbar span{position:absolute;left:23px;font-size:11px;white-space:nowrap}#cbar-unit{position:absolute;right:12px;top:16px;max-width:160px;font-size:12px;text-align:right;z-index:1}.legend-note{position:absolute;right:12px;top:247px;font-size:11px;z-index:1}
aside{overflow:auto;padding:12px;border-left:1px solid var(--line);background:white}.card{border:1px solid var(--line);border-radius:9px;padding:10px 12px;margin-bottom:10px}.card h3{margin:0 0 7px;font-size:13px;font-weight:600}.big{font-size:28px;font-weight:700}.kv{display:grid;grid-template-columns:auto minmax(0,1fr);gap:3px 10px;font-size:12px}.kv span{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:11px}th,td{border-bottom:1px solid var(--line);padding:4px 3px;text-align:right}th:first-child,td:first-child{text-align:left}summary{font-weight:600;cursor:pointer}details>div{margin-top:10px}p{margin:7px 0}.flag{background:#fff4d6;border:1px solid #f0c36d;border-radius:5px;padding:5px 8px;margin:5px 0;font-size:12px}.ok{background:#e6f6ea;border-color:#8fd19e}.unroll-row{margin:8px 0}.unroll-row canvas{width:100%;height:100px;border:1px solid var(--line);display:block}.process-controls{display:flex;flex-wrap:wrap;gap:6px}.process-controls label{font-size:12px}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto}.fallback{position:absolute;inset:40% 20px auto;padding:15px;background:#fff4d6;border-radius:8px;z-index:3}#snapshot{display:none}#print-note{display:none}
@media(max-width:950px){main{grid-template-columns:1fr;height:auto}#view{height:65vh;min-height:440px}aside{overflow:visible;border-left:0}#ctrl{width:230px}}
@page{size:A4 portrait;margin:10mm}
@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact}body{background:#fff;font-size:9pt}header{padding:0 0 5mm}header h1{font-size:14pt}.actions,#ctrl,#tip,.legend-note,#labels,#view>canvas,#cbar,#cbar-unit,.process-only{display:none!important}main{display:block;height:auto;min-height:0}#view{height:auto;min-height:0;background:white;overflow:visible}#snapshot{display:block;width:100%;max-height:115mm;object-fit:contain}aside{padding:3mm 0 0;overflow:visible;border:0;display:grid;grid-template-columns:1fr 1fr;gap:3mm}.card{margin:0;padding:2.5mm;border-radius:2mm;break-inside:avoid}.card h3{font-size:9pt;margin-bottom:1mm}.big{font-size:18pt}table{font-size:8pt}.branch-card{grid-column:1/-1}.kv{font-size:8pt}.sub,small{font-size:8pt}#print-note{display:block;grid-column:1/-1;font-size:8pt}}
</style></head><body>
<header><div><h1>壁面 WSS 预测报告</h1><div class="sub" id="subtitle"></div></div><div class="actions"><button id="save-image">保存截图</button><button id="print-report">打印 / 保存 PDF</button></div></header>
<main><div id="view"><div id="labels"></div><img id="snapshot" alt="WSS 壁面视图与当前色标">
<div id="ctrl"><div id="ctrl-head"><strong>显示设置与口径</strong><button type="button" id="ctrl-toggle" aria-expanded="true">收起</button></div><div id="ctrl-body">
<div class="ctrl-summary" id="display-readout">当前视图：WSS 壁面 · 统计：预测点云 · 壁面：Gaussian 插值</div>
<div class="process-controls view-mode-controls" aria-label="显示对象"><button data-v="wss" class="on">WSS 壁面</button><button data-v="stl">输入 STL</button><button data-v="cl">中心线</button><button data-v="cloud">预测点云</button></div>
<label>色标 <select id="scale-mode"><option value="case">本例 p99</option><option value="fixed">固定 Pa · 跨报告保留</option></select></label>
<label id="fixed-control" hidden>固定上限 <input id="fixedmax" type="number" min="0.01" step="0.5" value="10"> Pa</label>
<small id="scale-description"></small>
<label>分支 <select id="branch"><option value="-1">全部</option></select></label>
<label class="highlight-control">高亮阈值 <input type="range" id="highlight-pct" min="0.5" max="10" step="0.5" value="1"><output id="highlight-value">1%</output></label>
<label><input type="checkbox" id="showtop"> 启用高亮（最高 <span id="highlight-label">1%</span>）</label>
<label><input type="checkbox" id="showpeak" checked> 全场最大值标记</label>
<details><summary>视图设置</summary><label>剖切 <input type="range" id="clip" min="0" max="100" value="100"></label><button id="reset-camera">恢复视角</button><small>拖动旋转 · 滚轮缩放 · 右键平移</small></details>
</div></div><div id="cbar-unit"></div><div id="cbar"></div><div class="legend-note">■ 灰色：插值未覆盖 · 淡色：非选中分支 / 高亮阈值以下</div><div id="tip">移动鼠标到壁面读取 Gaussian 插值；统计卡片使用预测点云。</div></div>
<aside id="panel"></aside></main>
<noscript>请启用 JavaScript 查看交互报告；summary.json 保留全部统计与运行记录。</noscript>
<!--WSS_META_START--><script id="wss-report-meta" type="application/json">__META__</script><!--WSS_META_END-->
<script id="wss-report-arrays" type="application/json">__ARRAYS__</script>
<script>__THREE__</script><script>__ORBIT__</script>
<script>
'use strict';
const META=JSON.parse(document.getElementById('wss-report-meta').textContent);
const ARR=JSON.parse(document.getElementById('wss-report-arrays').textContent);
const el=id=>document.getElementById(id);
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(x,d=2){return x===null||x===undefined||!Number.isFinite(+x)?'—':(+x).toFixed(d);}
function bounds(a){let lo=Infinity,hi=-Infinity;for(const v of a){if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}return [lo,hi];}
function dec(b64,T){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer);}
function showPanel(){const m=META,w=m.wss_field_pa,p=m.peak,ic=m.input_check||{},cloud=m.cloud||{};let h='';
 const pct=v=>Math.max(0,Math.min(100,Number.isFinite(+v)?(+v*100):0));
 const bar=(label,value,detail,klass='')=>`<div class="metric-bar"><span class="metric-label">${esc(label)}</span><span class="metric-track"><span class="metric-fill ${klass}" style="width:${pct(value)}%"></span></span><span>${esc(detail)}</span></div>`;
 h+=`<section class="card"><h3>固定收缩期帧 · 空间 p99</h3><div class="big">${fmt(p.p99_pa)} Pa</div><small>预测点云第 99 百分位，不代表时间最大值。</small><div class="kv"><b>全场最大值</b><span>${fmt(p.max_pa)} Pa</span><b>最大值位置</b><span>${esc(p.branch)}，距入口 ${fmt(p.s_from_inlet_mm,0)} mm；距分叉 ${fmt(p.dist_to_junction_mm,0)} mm</span><b>黄色标记</b><span>全场最大值预测点；局部半径 ${fmt(p.local_radius_mm,1)} mm</span></div></section>`;
 h+=`<section class="card"><h3>点数占比与估计面积</h3><div class="metric-bars">${bar('低 WSS < '+fmt(w.thresholds_pa[0],1)+' Pa',w.area_frac_low,fmt(w.area_frac_low*100,1)+'% · '+fmt(w.area_low_mm2/100,1)+' cm²','low')}${bar('高 WSS > '+fmt(w.thresholds_pa[1],1)+' Pa',w.area_frac_high,fmt(w.area_frac_high*100,1)+'% · '+fmt(w.area_high_mm2/100,1)+' cm²','high')}${bar('极高 > '+fmt(w.thresholds_pa[2],1)+' Pa',w.area_frac_very_high,fmt(w.area_frac_very_high*100,1)+'%','very-high')}</div><div class="kv"><b>均值 / 中位</b><span>${fmt(w.mean)} / ${fmt(w.median)} Pa</span></div><small>估计面积 = 点数占比 × 输入壁面总面积；未按逐面片面积加权。</small></section>`;
 const branchEntries=Object.entries(m.per_branch||{}),branchMax=Math.max(...branchEntries.map(([,b])=>Number(b.wss_p99_pa)||0),.01);
 h+='<section class="card branch-card"><h3>分支统计 · 预测点云（Pa）</h3><div class="branch-chart metric-bars">'+branchEntries.map(([name,b])=>bar(name,(Number(b.wss_p99_pa)||0)/branchMax,fmt(b.wss_p99_pa)+' Pa')).join('')+'</div><table><thead><tr><th>分支</th><th>p99</th><th>均值</th><th>最大</th><th>低%</th><th>高%</th><th>估计<br>cm²</th></tr></thead><tbody>';
 for(const [name,b] of Object.entries(m.per_branch||{}))h+=`<tr><td>${esc(name)}</td><td>${fmt(b.wss_p99_pa)}</td><td>${fmt(b.wss_mean_pa)}</td><td>${fmt(b.wss_max_pa)}</td><td>${fmt(b.frac_low*100,0)}</td><td>${fmt(b.frac_high*100,0)}</td><td>${fmt(b.area_mm2/100,1)}</td></tr>`;
 h+='</tbody></table><small>仅列出 ≥ 10 个预测点的分支；全场统计包含全部预测点。</small></section>';
 const flags=[...(ic.flags||[]),...(m.flags||[])];
 if(flags.length)h+='<section class="card"><h3>需关注的输入信息</h3>'+flags.map(f=>'<div class="flag">'+esc(f)+'</div>').join('')+'</section>';
 h+='<details class="card process-only" id="unroll-details"><summary>分支展开图（s, θ）</summary><div><small>每条分支单独显示，横轴为距入口弧长（mm），纵轴为周向角 −π 至 π。不同分支不会重叠；像素汇总显示该像素的最大预测 WSS。</small><div id="unroll-plots"></div></div></details>';
 h+='<details class="card process-only"><summary>查看计算过程与显示口径</summary><div><p><label>点云特征 <select id="feat"><option value="wss">WSS (Pa)</option><option value="s">沿程弧长 (mm)</option><option value="th">周向角 (rad)</option><option value="r">半径 (mm)</option><option value="dj">到分叉距离 (mm)</option></select></label></p><div class="process-controls"><label><input type="checkbox" id="showpts">叠加点云</label><label><input type="checkbox" id="showcl">叠加中心线</label></div>';
 h+='<p><small>所有卡片、最大值位置和最高 1% 点集均来自预测点云。STL 颜色及壁面鼠标读值使用 Gaussian 插值（σ = 0.5 mm，最多 8 邻点，覆盖半径 1.5 mm），鼠标读取最近面顶点；不等于该处原始预测值。未覆盖顶点为灰色且无有效读值。最高 1% 遇同值并列时可超过 1%。</small></p>';
 h+=`<p><small>top-5% 连通簇（≥ 20 点）：${fmt(p.top5_clusters_ge20,0)} 个；最大簇 ${fmt((p.top5_cluster_sizes||[])[0],0)} 点。几何连通半径 = 2 × 点间距。</small></p>`;
 h+='<h3>出口确认与 Murray 分流</h3><div class="kv">';
 for(const e of m.endpoints||[])h+=`<b>${esc(e.name_cn)}</b><span>盖面半径 ${fmt(e.radius_mm,1)} mm${e.share!=null?' · 分流 '+fmt(e.share*100,0)+'%':''}</span>`;
 h+=`</div><small>${m.outlets_confirmed?'出口映射已人工确认':'出口映射为自动建议'}</small>`;
 h+=`<h3>输入与采样</h3><div class="kv"><b>单位</b><span>${esc(ic.unit)}</span><b>顶点 / 面</b><span>${fmt(ic.vertices,0)} / ${fmt(ic.faces,0)}</span><b>输入面积</b><span>${fmt(ic.area_mm2/100,1)} cm²</span><b>开口</b><span>${fmt(ic.openings,0)}</span><b>点云</b><span>${fmt(cloud.n_points,0)} 点 · 间距 ${fmt(cloud.spacing_mm,3)} mm · 平滑 ${fmt(cloud.smooth_mm,2)} mm</span></div></div></details>`;
 h+='<details class="card process-only"><summary>几何参数</summary><div><table><tr><th>分支</th><th>长 mm</th><th>最小 r</th><th>中位 r</th><th>最大直径</th><th>迂曲</th></tr>';
 for(const [name,g] of Object.entries(m.geometry||{}))h+=`<tr><td>${esc(name)}</td><td>${fmt(g.length_mm,0)}</td><td>${fmt(g.radius_min_mm,1)}</td><td>${fmt(g.radius_median_mm,1)}</td><td>${fmt(g.max_diameter_mm,1)}</td><td>${fmt(g.tortuosity)}</td></tr>`;
 h+='</table></div></details>';
 h+=`<details class="card process-only"><summary>耗时与运行记录</summary><div><small>${esc(m.device)}${m.gpu?' · '+esc(m.gpu):''}</small><table>`;
 const timingNames={ingest:'输入检查',centerline:'中心线',smooth_resample:'平滑与重采样',features:'构建特征',inference_5_models:'模型预测',metrics_and_export:'统计与导出',metrics_and_interpolation:'统计与插值',export:'导出',total:'累计计算时间',compute_total:'累计计算时间',queue_wait:'排队',confirmation_wait:'等待人工确认',wall_total:'任务历时',finalization:'记录回填'};
 for(const [k,v] of Object.entries(m.timing_s||{}))h+=`<tr><td>${esc(timingNames[k]||k)}</td><td>${fmt(v,2)} s</td></tr>`;
 h+='</table><p><small>计算时间、排队和人工确认分别记录；导出包含首次完整 HTML 写盘，最终记录回填有少量额外开销。</small></p>';
 h+=`<small>发布包 ${esc(m.release)} · 输入 SHA256 ${esc(m.input_sha256)} · ${esc(m.created_at)}</small>`;
 h+='<details><summary>完整参数与版本信息</summary><pre>'+esc(JSON.stringify({release_hash:m.release_hash||m.release_sha256,run_parameters:m.run_parameters||m.parameters,mapping:m.mapping,frame_transform:m.frame_transform,statistics_protocol:m.statistics_protocol,interpolation:m.interpolation,timing_protocol:m.timing_protocol,exports:m.exports||m.export_status},null,2))+'</pre></details></div></details>';
 h+=`<div id="print-note">病例 ${esc(m.case_id)} · 发布包 ${esc(m.release)} · ${esc(m.created_at)}<br>固定收缩期帧的模型预测；统计来自预测点云，壁面显示使用 Gaussian 插值。黄色标记为全场最大值。</div>`;
 el('panel').innerHTML=h;el('subtitle').textContent=m.case_id+' · 固定收缩期帧 · '+m.release;
 for(const [sid,name] of Object.entries(m.branch_names||{})){const o=document.createElement('option');o.value=sid;o.textContent=name;el('branch').appendChild(o);}
}
showPanel();
el('print-report').onclick=()=>window.print();
function initViewer(){
 const MV=dec(ARR.mv,Float32Array),MF=dec(ARR.mf,Uint32Array),MW=dec(ARR.mw,Float32Array),MS=dec(ARR.ms,Int32Array);
 const PV=dec(ARR.pv,Float32Array),PW=dec(ARR.pw,Float32Array),PS=dec(ARR.ps,Int32Array),P_S=dec(ARR.p_s,Float32Array),P_TH=dec(ARR.p_th,Float32Array),P_R=dec(ARR.p_r,Float32Array),P_DJ=dec(ARR.p_dj,Float32Array);
 const CV=dec(ARR.cv,Float32Array),CR=dec(ARR.cr,Float32Array),CE=dec(ARR.ce,Uint32Array);
 const STOPS=[[0,[31,78,156]],[.5,[247,247,247]],[1,[192,57,43]]];
 function cmap(t){t=Math.min(1,Math.max(0,t));for(let i=1;i<STOPS.length;i++)if(t<=STOPS[i][0]){const a=STOPS[i-1],b=STOPS[i],u=(t-a[0])/(b[0]-a[0]);return [0,1,2].map(k=>(a[1][k]+(b[1][k]-a[1][k])*u)/255);}return [1,1,1];}
 function vir(t){t=Math.min(1,Math.max(0,t));const c=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],x=t*4,i=Math.min(3,Math.floor(x)),u=x-i;return [0,1,2].map(k=>(c[i][k]+(c[i+1][k]-c[i][k])*u)/255);}
 const view=el('view'),renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.localClippingEnabled=true;view.appendChild(renderer.domElement);
 renderer.domElement.addEventListener('webglcontextlost',ev=>{ev.preventDefault();el('tip').textContent='3D 显示已中断，请刷新报告；统计表仍可查看。';});
 const scene=new THREE.Scene();scene.background=new THREE.Color(0xeef2f7);const camera=new THREE.PerspectiveCamera(35,1,.1,5000),controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
 scene.add(new THREE.HemisphereLight(0xffffff,0x8899aa,.9));const light=new THREE.DirectionalLight(0xffffff,.6);camera.add(light);light.position.set(0,0,1);scene.add(camera);
 let mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];for(let i=0;i<MV.length;i+=3)for(let k=0;k<3;k++){mn[k]=Math.min(mn[k],MV[i+k]);mx[k]=Math.max(mx[k],MV[i+k]);}
 const center=mn.map((v,k)=>(v+mx[k])/2),size=Math.max(...mx.map((v,k)=>v-mn[k]),1),clipPlane=new THREE.Plane(new THREE.Vector3(0,0,-1),1e6);
 const mg=new THREE.BufferGeometry();mg.setAttribute('position',new THREE.BufferAttribute(MV,3));mg.setIndex(new THREE.BufferAttribute(MF,1));mg.computeVertexNormals();const mcol=new Float32Array(MV.length);mg.setAttribute('color',new THREE.BufferAttribute(mcol,3));
 const mmat=new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide,clippingPlanes:[clipPlane]});const mesh=new THREE.Mesh(mg,mmat);scene.add(mesh);
 const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(PV,3));const pcol=new Float32Array(PV.length);pg.setAttribute('color',new THREE.BufferAttribute(pcol,3));const pts=new THREE.Points(pg,new THREE.PointsMaterial({size:.9,vertexColors:true,clippingPlanes:[clipPlane]}));scene.add(pts);
 const cg=new THREE.BufferGeometry();cg.setAttribute('position',new THREE.BufferAttribute(CV,3));cg.setIndex(new THREE.BufferAttribute(CE,1));const ccol=new Float32Array(CV.length),[rmin,rmax]=bounds(CR);for(let i=0;i<CR.length;i++)ccol.set(vir((CR[i]-rmin)/(rmax-rmin||1)),3*i);cg.setAttribute('color',new THREE.BufferAttribute(ccol,3));const cl=new THREE.LineSegments(cg,new THREE.LineBasicMaterial({vertexColors:true,clippingPlanes:[clipPlane]}));scene.add(cl);
 const labels=[];for(const e of META.endpoints||[]){const obj=new THREE.Mesh(new THREE.SphereGeometry(Math.max(1,e.radius_mm*.6),12,8),new THREE.MeshLambertMaterial({color:e.kind==='inlet'?0x1f77b4:0xff7f0e,clippingPlanes:[clipPlane]}));obj.position.set(...e.center_mm);scene.add(obj);const div=document.createElement('div');div.textContent=e.name_cn+' r='+fmt(e.radius_mm,1)+' mm';el('labels').appendChild(div);labels.push({obj,div});}
 const peak=new THREE.Mesh(new THREE.SphereGeometry(1.6,12,8),new THREE.MeshBasicMaterial({color:0xffd400,clippingPlanes:[clipPlane]}));peak.position.set(...META.peak.xyz_mm);scene.add(peak);const peakLabel=document.createElement('div');peakLabel.textContent='全场最大值 '+fmt(META.peak.max_pa)+' Pa';el('labels').appendChild(peakLabel);labels.push({obj:peak,div:peakLabel});
 let viewMode='wss',feat='wss',branchSel=-1,fixedMax=10,scaleMode='case',vmax=Math.max(META.peak.p99_pa,.01),highlightPct=1,highlightThreshold=META.peak.p99_pa,highlightEnabled=false,topIndices=[],top=null;
 function calcHighlightThreshold(){const vals=Array.from(PW).filter(Number.isFinite).sort((a,b)=>a-b);if(!vals.length)return 0;const rank=Math.max(0,Math.min(vals.length-1,Math.ceil((1-highlightPct/100)*vals.length)-1));return vals[rank];}
 function rebuildTop(){highlightThreshold=calcHighlightThreshold();topIndices=[];const topPositions=[];for(let i=0;i<PW.length;i++)if(Number.isFinite(PW[i])&&PW[i]>=highlightThreshold&&(branchSel<0||PS[i]===branchSel)){topPositions.push(PV[i*3],PV[i*3+1],PV[i*3+2]);topIndices.push(i);}if(top){scene.remove(top);top.geometry.dispose();top.material.dispose();}const tg=new THREE.BufferGeometry();tg.setAttribute('position',new THREE.Float32BufferAttribute(topPositions,3));const tc=new Float32Array(topPositions.length);for(let j=0;j<topIndices.length;j++)tc.set([.95,.2,.75],3*j);tg.setAttribute('color',new THREE.BufferAttribute(tc,3));top=new THREE.Points(tg,new THREE.PointsMaterial({size:1.8,vertexColors:true,depthTest:false,clippingPlanes:[clipPlane],transparent:true,opacity:.95}));top.renderOrder=2;top.visible=(viewMode==='wss'||viewMode==='cloud')&&el('showtop').checked;scene.add(top);el('highlight-value').textContent=fmt(highlightPct,1)+'%';el('highlight-label').textContent=fmt(highlightPct,1)+'%';}
 try{const saved=JSON.parse(localStorage.getItem('wss-report-scale-v1'));if(saved&&saved.mode==='fixed'&&Number.isFinite(saved.max)&&saved.max>0){fixedMax=saved.max;scaleMode='fixed';}if(saved&&Number.isFinite(saved.highlightPct))highlightPct=Math.max(.5,Math.min(10,saved.highlightPct));}catch(_){}
 el('scale-mode').value=scaleMode;el('fixedmax').value=fixedMax;el('highlight-pct').value=highlightPct;rebuildTop();
 function feature(){if(feat==='s')return {values:P_S,mode:'vir',lo:0,hi:bounds(P_S)[1],label:'沿程弧长 · mm'};if(feat==='th')return {values:P_TH,mode:'vir',lo:-Math.PI,hi:Math.PI,label:'周向角 · rad'};if(feat==='r')return {values:P_R,mode:'vir',lo:bounds(P_R)[0],hi:bounds(P_R)[1],label:'局部半径 · mm'};if(feat==='dj')return {values:P_DJ,mode:'vir',lo:0,hi:bounds(P_DJ)[1],label:'到分叉距离 · mm'};return {values:PW,mode:'wss',lo:0,hi:vmax,label:'WSS · Pa'};}
 function colorArray(values,segment,array,mode,lo,hi){const isWssField=values===MW||values===PW;for(let i=0;i<values.length;i++){let c=!Number.isFinite(values[i])?[.48,.51,.55]:(mode==='grey'?[.72,.75,.8]:(mode==='wss'?cmap:vir)((values[i]-lo)/(hi-lo||1)));if(highlightEnabled&&isWssField&&mode==='wss'&&Number.isFinite(values[i])&&values[i]<highlightThreshold)c=c.map(v=>v*.28+.72*.86);if(branchSel>=0&&segment[i]!==branchSel)c=c.map(v=>v*.25+.75*.86);array.set(c,3*i);}}
 function drawBar(mode,lo,hi,label){el('cbar').style.display=viewMode==='stl'?'none':'';el('cbar-unit').textContent=viewMode==='stl'?'输入 STL · mm':label;let g='linear-gradient(to top';for(let i=0;i<=10;i++)g+=',rgb('+(mode==='wss'?cmap(i/10):vir(i/10)).map(x=>Math.round(x*255)).join(',')+') '+i*10+'%';el('cbar').style.background=g+')';el('cbar').innerHTML='<span style="top:-5px">'+fmt(hi,1)+'</span><span style="top:86px">'+fmt((lo+hi)/2,1)+'</span><span style="bottom:-5px">'+fmt(lo,1)+'</span>';}
 function recolor(){vmax=scaleMode==='fixed'?fixedMax:Math.max(META.peak.p99_pa,.01);el('fixed-control').hidden=scaleMode!=='fixed';el('scale-description').textContent='0 – '+fmt(vmax,2)+' Pa · '+(scaleMode==='fixed'?'固定色标':'本例空间 p99')+'；超限按上限着色';rebuildTop();colorArray(MW,MS,mcol,viewMode==='wss'?'wss':'grey',0,vmax);const f=feature();colorArray(f.values,PS,pcol,f.mode,f.lo,f.hi);updateColors();el('display-readout').textContent='当前视图：'+({wss:'WSS 壁面',stl:'输入 STL',cl:'中心线',cloud:'预测点云'}[viewMode]||viewMode)+' · '+(viewMode==='wss'?'统计：预测点云 · 壁面：Gaussian 插值':f.label)+(branchSel>=0?' · 分支：'+(META.branch_names[branchSel]||branchSel):'');
  if(viewMode==='cloud')drawBar(f.mode,f.lo,f.hi,f.label);else if(viewMode==='cl')drawBar('vir',rmin,rmax,'中心线半径 · mm');else drawBar('wss',0,vmax,'WSS · Pa');drawUnroll();}
 function updateColors(){mg.attributes.color.needsUpdate=true;pg.attributes.color.needsUpdate=true;}
 function visibility(){mesh.visible=viewMode!=='cloud';mmat.transparent=viewMode==='cl';mmat.opacity=viewMode==='cl'?.18:1;pts.visible=viewMode==='cloud'||el('showpts').checked;cl.visible=viewMode==='cl'||el('showcl').checked;peak.visible=viewMode==='wss'&&el('showpeak').checked;top.visible=(viewMode==='wss'||viewMode==='cloud')&&el('showtop').checked;for(const item of labels)if(item.obj!==peak)item.obj.visible=viewMode==='cl';}
 function setView(mode){viewMode=mode;document.querySelectorAll('[data-v]').forEach(b=>b.classList.toggle('on',b.dataset.v===mode));visibility();recolor();el('tip').textContent=mode==='cloud'?'点云鼠标读值为原始预测/几何特征。':'壁面鼠标读值为 Gaussian 插值；统计卡片使用预测点云。';}
 document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setView(b.dataset.v));
 function saveScale(){try{localStorage.setItem('wss-report-scale-v1',JSON.stringify({mode:scaleMode,max:fixedMax,highlightPct}));}catch(_){}recolor();}
 el('scale-mode').onchange=e=>{scaleMode=e.target.value;saveScale();};el('fixedmax').onchange=e=>{const v=Number(e.target.value);if(!Number.isFinite(v)||v<=0){e.target.value=fixedMax;return;}fixedMax=v;saveScale();};
 el('branch').onchange=e=>{branchSel=Number(e.target.value);recolor();};el('feat').onchange=e=>{feat=e.target.value;if(feat!=='wss'){el('showpts').checked=false;setView('cloud');}else recolor();};
 el('highlight-pct').oninput=e=>{highlightPct=Math.max(.5,Math.min(10,Number(e.target.value)||1));highlightEnabled=el('showtop').checked;saveScale();};
 for(const id of ['showpts','showcl','showpeak'])el(id).onchange=()=>{if(el('showpts').checked&&feat!=='wss')setView('cloud');else visibility();};
 el('showtop').onchange=()=>{highlightEnabled=el('showtop').checked;recolor();visibility();};
 const ctrl=el('ctrl'),toggle=el('ctrl-toggle');try{if(localStorage.getItem('wss-report-ctrl-collapsed')==='1'){ctrl.classList.add('collapsed');toggle.textContent='展开';toggle.setAttribute('aria-expanded','false');}}catch(_){}
 toggle.onclick=()=>{const collapsed=ctrl.classList.toggle('collapsed');toggle.textContent=collapsed?'展开':'收起';toggle.setAttribute('aria-expanded',String(!collapsed));try{localStorage.setItem('wss-report-ctrl-collapsed',collapsed?'1':'0');}catch(_){};};
 el('clip').oninput=e=>{const t=Number(e.target.value)/100;clipPlane.constant=t>=1?1e6:mn[2]+(mx[2]-mn[2])*t;el('tip').textContent='剖切已更新；鼠标只读取可见部分。';};
 function resetCamera(){camera.position.set(center[0],center[1],center[2]+size*1.8);controls.target.set(...center);camera.near=size*.001;camera.far=size*30;camera.updateProjectionMatrix();controls.update();}el('reset-camera').onclick=resetCamera;
 const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();ray.params.Points.threshold=1;
 renderer.domElement.addEventListener('mousemove',ev=>{const rect=renderer.domElement.getBoundingClientRect();mouse.set((ev.clientX-rect.left)/rect.width*2-1,-(ev.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(mouse,camera);
  if(viewMode==='cloud'){const hit=ray.intersectObject(pts,false).find(h=>clipPlane.distanceToPoint(h.point)>=0);if(!hit)return;const i=hit.index,f=feature();el('tip').textContent='预测点云 · WSS '+fmt(PW[i])+' Pa\n'+(META.branch_names[PS[i]]||PS[i])+' · '+f.label+' '+fmt(f.values[i])+'\n位置 ('+[0,1,2].map(k=>fmt(PV[3*i+k],1)).join(', ')+') mm';return;}
  if(viewMode!=='wss')return;const hit=ray.intersectObject(mesh,false).find(h=>clipPlane.distanceToPoint(h.point)>=0);if(!hit){el('tip').textContent='移动鼠标到可见壁面读取 Gaussian 插值。';return;}const f=hit.face;let best=f.a,bd=Infinity;for(const vi of [f.a,f.b,f.c]){const d=hit.point.distanceToSquared(new THREE.Vector3(MV[3*vi],MV[3*vi+1],MV[3*vi+2]));if(d<bd){bd=d;best=vi;}}
  // A partially unsupported triangle is marked invalid throughout to avoid a fictitious fill.
  const valid=[f.a,f.b,f.c].every(i=>Number.isFinite(MW[i]));el('tip').textContent=(valid?'壁面 Gaussian 插值 '+fmt(MW[best])+' Pa':'插值覆盖范围外：无有效 WSS 读值')+'\n分支 '+(META.branch_names[MS[best]]||MS[best])+'\n最近面顶点 ('+[0,1,2].map(k=>fmt(MV[3*best+k],1)).join(', ')+') mm';});
 const branchIndices=new Map();for(let i=0;i<PS.length;i++){if(!branchIndices.has(PS[i]))branchIndices.set(PS[i],[]);branchIndices.get(PS[i]).push(i);}
 const plotItems=[];for(const b of ARR.unroll_branches){const row=document.createElement('div');row.className='unroll-row';const label=document.createElement('small');label.textContent=(META.branch_names[b.segment_id]||b.segment_id)+' · s '+fmt(b.s_min_mm,1)+' – '+fmt(b.s_max_mm,1)+' mm';const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',label.textContent+' 的独立展开图');row.append(label,canvas);el('unroll-plots').appendChild(row);plotItems.push({row,canvas,...b});}
 function drawUnroll(){if(!el('unroll-details').open)return;for(const p of plotItems){p.row.hidden=branchSel>=0&&p.segment_id!==branchSel;if(p.row.hidden)continue;const c=p.canvas,W=c.width=Math.max(1,Math.round(c.clientWidth*2)),H=c.height=180,ctx=c.getContext('2d');if(!ctx)continue;const pixelMax=new Float32Array(W*H);pixelMax.fill(-Infinity);for(const i of branchIndices.get(p.segment_id)||[]){const x=Math.max(0,Math.min(W-1,Math.floor((P_S[i]-p.s_min_mm)/(p.s_max_mm-p.s_min_mm||1)*(W-1)))),y=Math.max(0,Math.min(H-1,Math.floor((1-(P_TH[i]+Math.PI)/(2*Math.PI))*(H-1))));pixelMax[y*W+x]=Math.max(pixelMax[y*W+x],PW[i]);}ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);const img=ctx.getImageData(0,0,W,H);for(let i=0;i<pixelMax.length;i++)if(Number.isFinite(pixelMax[i])){const c=cmap(pixelMax[i]/vmax);for(let k=0;k<3;k++)img.data[4*i+k]=c[k]*255;img.data[4*i+3]=255;}ctx.putImageData(img,0,0);}}
 el('unroll-details').addEventListener('toggle',drawUnroll);
 function resize(){const w=view.clientWidth,h=view.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();drawUnroll();}window.addEventListener('resize',resize);
 function capture(){renderer.render(scene,camera);const source=renderer.domElement,canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');canvas.width=1200;canvas.height=700;ctx.fillStyle='#eef2f7';ctx.fillRect(0,0,1200,700);const scale=Math.min(980/source.width,610/source.height),w=source.width*scale,h=source.height*scale;ctx.drawImage(source,(980-w)/2,45+(610-h)/2,w,h);ctx.fillStyle='#203049';ctx.font='22px sans-serif';ctx.fillText('WSS · '+String(META.case_id).slice(0,70),20,28);const f=viewMode==='cloud'?feature():viewMode==='cl'?{mode:'vir',lo:rmin,hi:rmax,label:'中心线半径 · mm'}:{mode:'wss',lo:0,hi:vmax,label:'WSS · Pa'};if(viewMode!=='stl'){for(let y=0;y<300;y++){ctx.fillStyle='rgb('+(f.mode==='wss'?cmap(1-y/299):vir(1-y/299)).map(x=>Math.round(x*255)).join(',')+')';ctx.fillRect(1010,130+y,25,1);}ctx.fillStyle='#203049';ctx.font='18px sans-serif';ctx.fillText(f.label,990,100);ctx.fillText(fmt(f.hi,2),1045,140);ctx.fillText(fmt(f.lo,2),1045,435);}ctx.fillStyle='#203049';ctx.font='17px sans-serif';ctx.fillText('固定收缩期帧 · p99 '+fmt(META.peak.p99_pa)+' Pa · 最大值 '+fmt(META.peak.max_pa)+' Pa',20,675);ctx.font='14px sans-serif';ctx.fillText('统计：预测点云；壁面：Gaussian 插值；黄色：全场最大值；粉色：高亮最高 '+fmt(highlightPct,1)+'%',20,697);return canvas.toDataURL('image/png');}
 el('save-image').onclick=()=>{const a=document.createElement('a');a.href=capture();a.download='WSS-'+String(META.case_id).replace(/[^\w\u3400-\u9fff-]/g,'_')+'.png';a.click();};
 window.addEventListener('beforeprint',()=>{el('snapshot').src=capture();});window.addEventListener('afterprint',resize);
 function frame(){controls.update();renderer.render(scene,camera);const w=view.clientWidth,h=view.clientHeight;for(const it of labels){const v=it.obj.position.clone().project(camera);const show=it.obj.visible&&clipPlane.distanceToPoint(it.obj.position)>=0&&v.z>=-1&&v.z<=1;it.div.style.display=show?'':'none';if(show){it.div.style.left=(v.x+1)/2*w+'px';it.div.style.top=(1-v.y)/2*h+'px';}}requestAnimationFrame(frame);}
 resize();resetCamera();setView('wss');frame();
}
try{initViewer();}catch(error){const d=document.createElement('div');d.className='fallback';d.textContent='无法初始化 3D 视图。请使用支持 WebGL 的浏览器并启用图形加速；右侧统计和打印仍可使用。';el('view').appendChild(d);el('save-image').disabled=true;el('ctrl').hidden=true;console.error('WSS viewer:',error);}
</script></body></html>'''
# Keep the template readable without relying on a front-end build step.
TEMPLATE = TEMPLATE.replace('m gDummy();', 'updateColors();')
