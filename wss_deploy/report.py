"""Report export and self-contained viewer; statistics remain on prediction points."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .glossary import glossary_document
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
    meta = _report_schema_metadata(meta)
    content = out_path.read_text(encoding="utf-8")
    before, marker, tail = content.partition(META_START)
    _, end, after = tail.partition(META_END)
    if not marker or not end:
        raise ValueError("Report metadata markers were not found")
    replacement = '<script id="wss-report-meta" type="application/json">' + _script_json(_report_schema_metadata(meta)) + '</script>'
    tmp = out_path.with_name(out_path.name + ".meta.tmp")
    tmp.write_text(before + META_START + replacement + META_END + after, encoding="utf-8")
    tmp.replace(out_path)


def _report_schema_metadata(meta: dict) -> dict:
    """Return report metadata with a backwards-compatible field/time envelope.

    Older jobs pre-date ``fields`` and ``time_axis``.  The viewer must still
    identify those reports as a single fixed WSS frame rather than inventing
    velocity/pressure fields or additional frames.
    """
    out = dict(meta or {})
    results = out.get("results") if isinstance(out.get("results"), dict) else {}
    raw_fields = out.get("fields") or results.get("fields")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raw_fields = {}
        if out.get("wss_field_pa") is not None or out.get("peak") is not None:
            raw_fields["wss"] = {
                "schema_version": "wss-deploy.field/v1", "id": "wss",
                "label": "壁面切应力", "units": "Pa", "location": "wall",
                "kind": "scalar", "components": 1, "array_key": "wss_pa",
                "axis_order": ["point"], "time_indices": [0],
                "source": "prediction", "statistics_key": "wss",
            }
    out["fields"] = raw_fields
    raw_axis = out.get("time_axis") or results.get("time_axis")
    if not isinstance(raw_axis, list) or not raw_axis:
        frame = dict(out.get("model_frame") or {})
        frame = {"index": 0, **frame}
        frame.setdefault("label", "固定收缩期帧")
        raw_axis = [frame]
    out["time_axis"] = raw_axis
    return out


def build_html(out_path: Path, meta: dict, mesh: dict, cloud: dict, centerline: dict) -> None:
    """Write the self-contained wall WSS report.

    ``mesh`` may carry an optional ``trust`` uint8 bitmask per vertex (contract §3);
    ``meta`` may carry ``profiles`` / ``findings`` / ``trust`` / ``review`` /
    ``frame_transform.rotation``.  Every one of those is optional: the viewer hides
    the corresponding panel when the key is absent.
    """
    meta = _report_schema_metadata(meta)
    three = (STATIC_DIR / "three.min.js").read_text(encoding="utf-8")
    orbit = (STATIC_DIR / "OrbitControls.js").read_text(encoding="utf-8")
    common = (STATIC_DIR / "report_common.js").read_text(encoding="utf-8")
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
    # Extra wall scalar fields (three-head release: TAWSS / OSI), keyed by the descriptor's array_key.
    extra_m = dict(mesh.get("extra") or {}); extra_c = dict(cloud.get("extra") or {})
    if set(extra_m) != set(extra_c):
        raise ValueError("mesh.extra and cloud.extra must describe the same fields")
    if extra_c:
        n_pts, n_vertices = len(np.asarray(cloud["pts"])), len(np.asarray(mesh["vertices"]))
        xf = {}
        for key, values in extra_c.items():
            point_values = np.asarray(values, dtype=np.float32); vertex_values = np.asarray(extra_m[key], dtype=np.float32)
            if point_values.shape != (n_pts,) or vertex_values.shape != (n_vertices,):
                raise ValueError(f"extra field {key} must hold one value per prediction point and per vertex")
            xf[str(key)] = {"m": _b64(vertex_values, np.float32), "p": _b64(point_values, np.float32)}
        arrays["xf"] = xf
    trust = mesh.get("trust")
    if trust is not None:
        trust = np.asarray(trust)
        if trust.shape != (len(np.asarray(mesh["vertices"])),):
            raise ValueError("mesh.trust must hold one bitmask value per vertex")
        if trust.size and (trust.min() < 0 or trust.max() > 255):
            raise ValueError("mesh.trust values must fit an unsigned byte")
        arrays["mt"] = _b64(trust, np.uint8)
    # Metadata is inserted last so user text cannot introduce another template substitution.
    html = (TEMPLATE.replace("__CORE__", REPORT_CORE_JS).replace("__COMMON__", common).replace("__THREE__", three).replace("__ORBIT__", orbit)
            .replace("__GLOSSARY__", _script_json(glossary_document()))
            .replace("__ARRAYS__", _script_json(arrays)).replace("__META__", _script_json(meta)))
    Path(out_path).write_text(html, encoding="utf-8")


# Pure computation helpers shared by the report page and the Node unit tests
# (``tests/test_report.py`` writes this string to a file and ``require``s it).
REPORT_CORE_JS = r'''(function(root){
'use strict';
const UNITS={Pa:{factor:1,label:'Pa'},dyn:{factor:10,label:'dyn/cm²'}};
const clamp01=x=>Math.max(0,Math.min(1,x));
function unitInfo(u){return UNITS[u]||UNITS.Pa;}
function unitNames(){return Object.keys(UNITS);}
function toUnit(v,u){return Number.isFinite(+v)?+v*unitInfo(u).factor:NaN;}
function fromUnit(v,u){return Number.isFinite(+v)?+v/unitInfo(u).factor:NaN;}
function validThresholds(t){t=(Array.isArray(t)?t:[]).map(Number);return t.length===3&&t.every(Number.isFinite)&&t[0]>=0&&t[0]<t[1]&&t[1]<t[2]?t:null;}
function thresholdAreas(values,thresholds,areaMm2,mask){
  const t=validThresholds(thresholds);if(!t)throw new Error('thresholds must be three increasing non-negative numbers');
  let n=0,low=0,high=0,vhigh=0;
  for(let i=0;i<values.length;i++){if(mask&&!mask[i])continue;const v=values[i];if(!Number.isFinite(v))continue;n++;if(v<t[0])low++;if(v>t[1])high++;if(v>t[2])vhigh++;}
  const area=Number.isFinite(+areaMm2)?+areaMm2:0,f=x=>n?x/n:0;
  return {n,frac_low:f(low),frac_high:f(high),frac_very_high:f(vhigh),area_low_mm2:f(low)*area,area_high_mm2:f(high)*area,area_very_high_mm2:f(vhigh)*area,thresholds_pa:t};
}
function normalize(v,lo,hi,log,floor){
  if(!Number.isFinite(v))return NaN;floor=floor>0?floor:0.05;
  if(log){const l0=Math.log(Math.max(lo,floor)),l1=Math.log(Math.max(hi,floor*1.001));return (Math.log(Math.max(v,floor))-l0)/((l1-l0)||1);}
  return (v-lo)/((hi-lo)||1);
}
function quantize(t,bands){t=clamp01(t);const n=Math.floor(bands);if(!(n>0))return t;return (Math.min(n-1,Math.floor(t*n))+0.5)/n;}
function bandEdges(lo,hi,bands,log,floor){const n=Math.max(1,Math.floor(bands)||1);floor=floor>0?floor:0.05;const out=[];for(let i=0;i<=n;i++){const t=i/n;out.push(log?Math.exp(Math.log(Math.max(lo,floor))+t*(Math.log(Math.max(hi,floor*1.001))-Math.log(Math.max(lo,floor)))):lo+t*(hi-lo));}return out;}
function topThreshold(values,pct){const vals=[];for(let i=0;i<values.length;i++)if(Number.isFinite(values[i]))vals.push(values[i]);vals.sort((a,b)=>a-b);if(!vals.length)return 0;const rank=Math.max(0,Math.min(vals.length-1,Math.ceil((1-pct/100)*vals.length)-1));return vals[rank];}
function marchingTriangles(vertices,faces,values,level){
  const out=[];
  const cross=(a,b)=>{const va=values[a],vb=values[b];if((va>=level)===(vb>=level))return null;const s=(level-va)/(vb-va);return [vertices[3*a]+(vertices[3*b]-vertices[3*a])*s,vertices[3*a+1]+(vertices[3*b+1]-vertices[3*a+1])*s,vertices[3*a+2]+(vertices[3*b+2]-vertices[3*a+2])*s];};
  for(let t=0;t<faces.length;t+=3){const a=faces[t],b=faces[t+1],c=faces[t+2];if(!(Number.isFinite(values[a])&&Number.isFinite(values[b])&&Number.isFinite(values[c])))continue;const p=[cross(a,b),cross(b,c),cross(c,a)].filter(Boolean);if(p.length===2)out.push(p[0][0],p[0][1],p[0][2],p[1][0],p[1][1],p[1][2]);}
  return new Float32Array(out);
}
function b64url(bytes){let s='';for(let i=0;i<bytes.length;i++)s+=String.fromCharCode(bytes[i]);return btoa(s).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');}
function unb64url(str){str=String(str||'').replace(/-/g,'+').replace(/_/g,'/');while(str.length%4)str+='=';const s=atob(str),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
function encodeView(state){return b64url(new TextEncoder().encode(JSON.stringify(state)));}
function decodeView(str){const state=JSON.parse(new TextDecoder().decode(unb64url(str)));if(!state||typeof state!=='object'||Array.isArray(state))throw new Error('invalid view state');return state;}
function viewHash(state){return '#view='+encodeView(state);}
function parseViewHash(hash){const m=/(?:^|[#&])view=([A-Za-z0-9_-]+)/.exec(String(hash||''));return m?decodeView(m[1]):null;}
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
function validFrame(ft){return !!(ft&&Array.isArray(ft.rotation)&&ft.rotation.length===3&&ft.rotation.every(r=>Array.isArray(r)&&r.length===3&&r.every(Number.isFinite))&&Array.isArray(ft.origin_mm)&&ft.origin_mm.length===3&&ft.origin_mm.every(Number.isFinite));}
function alignedFromWorld(p,R,origin){const d=[p[0]-origin[0],p[1]-origin[1],p[2]-origin[2]];return [dot(R[0],d),dot(R[1],d),dot(R[2],d)];}
function worldFromAligned(q,R,origin){return [0,1,2].map(k=>origin[k]+R[0][k]*q[0]+R[1][k]*q[1]+R[2][k]*q[2]);}
function rotateToAligned(v,R){return [dot(R[0],v),dot(R[1],v),dot(R[2],v)];}
function rotateFromAligned(v,R){return [0,1,2].map(k=>R[0][k]*v[0]+R[1][k]*v[1]+R[2][k]*v[2]);}
function cameraToAligned(cam,R,origin){return {position:alignedFromWorld(cam.position,R,origin),target:alignedFromWorld(cam.target,R,origin),up:rotateToAligned(cam.up,R)};}
function cameraFromAligned(cam,R,origin){return {position:worldFromAligned(cam.position,R,origin),target:worldFromAligned(cam.target,R,origin),up:rotateFromAligned(cam.up,R)};}
const VIEW_NAMES={front:'前',back:'后',left:'左',right:'右',top:'上',bottom:'下'};
function standardViews(R,center,distance){
  const ex=R[0],ey=R[1],ez=R[2],neg=v=>v.map(x=>-x);
  const mk=(dir,up)=>({position:center.map((v,k)=>v+dir[k]*distance),target:center.slice(),up:up.slice()});
  return {front:mk(neg(ey),ez),back:mk(ey,ez),left:mk(ex,ez),right:mk(neg(ex),ez),top:mk(ez,neg(ey)),bottom:mk(neg(ez),neg(ey))};
}
function nearestIndex(points,queries,cell){
  const n=Math.floor(points.length/3),m=Math.floor(queries.length/3),out=new Int32Array(m).fill(-1);if(!n||!m)return out;
  const mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];for(let i=0;i<n;i++)for(let k=0;k<3;k++){const v=points[3*i+k];if(v<mn[k])mn[k]=v;if(v>mx[k])mx[k]=v;}
  const ext=[0,1,2].map(k=>Math.max(mx[k]-mn[k],1e-6));
  if(!(cell>0))cell=Math.max(Math.cbrt(ext[0]*ext[1]*ext[2]/n)*2,1e-3);
  const dims=[0,1,2].map(k=>Math.min(4096,Math.floor(ext[k]/cell)+1));
  const cellOf=(x,k)=>Math.max(0,Math.min(dims[k]-1,Math.floor((x-mn[k])/cell)));
  const key=(ix,iy,iz)=>ix+dims[0]*(iy+dims[1]*iz);
  const grid=new Map();for(let i=0;i<n;i++){const k=key(cellOf(points[3*i],0),cellOf(points[3*i+1],1),cellOf(points[3*i+2],2));let a=grid.get(k);if(!a){a=[];grid.set(k,a);}a.push(i);}
  const maxRing=Math.max(dims[0],dims[1],dims[2]);
  for(let q=0;q<m;q++){const x=queries[3*q],y=queries[3*q+1],z=queries[3*q+2];const cx=cellOf(x,0),cy=cellOf(y,1),cz=cellOf(z,2);let best=-1,bd=Infinity;
    for(let r=0;r<=maxRing;r++){
      for(let ix=cx-r;ix<=cx+r;ix++){if(ix<0||ix>=dims[0])continue;for(let iy=cy-r;iy<=cy+r;iy++){if(iy<0||iy>=dims[1])continue;for(let iz=cz-r;iz<=cz+r;iz++){if(iz<0||iz>=dims[2])continue;if(Math.max(Math.abs(ix-cx),Math.abs(iy-cy),Math.abs(iz-cz))!==r)continue;const a=grid.get(key(ix,iy,iz));if(!a)continue;for(const i of a){const dx=points[3*i]-x,dy=points[3*i+1]-y,dz=points[3*i+2]-z,d=dx*dx+dy*dy+dz*dz;if(d<bd){bd=d;best=i;}}}}}
      if(best>=0&&Math.sqrt(bd)<=r*cell)break;
    }
    out[q]=best;}
  return out;
}
function sphereMask(points,center,radius){const n=Math.floor(points.length/3),mask=new Uint8Array(n),r2=radius*radius;for(let i=0;i<n;i++){const dx=points[3*i]-center[0],dy=points[3*i+1]-center[1],dz=points[3*i+2]-center[2];if(dx*dx+dy*dy+dz*dz<=r2)mask[i]=1;}return mask;}
function rangeMask(s,segment,sid,s0,s1){const n=s.length,mask=new Uint8Array(n),lo=Math.min(s0,s1),hi=Math.max(s0,s1),any=sid===null||sid===undefined;for(let i=0;i<n;i++)if((any||segment[i]===sid)&&s[i]>=lo&&s[i]<=hi)mask[i]=1;return mask;}
function maskIndices(mask){const out=[];for(let i=0;i<mask.length;i++)if(mask[i])out.push(i);return out;}
function percentile(sorted,q){if(!sorted.length)return null;const at=q*(sorted.length-1),lo=Math.floor(at),hi=Math.ceil(at);return sorted[lo]+(sorted[hi]-sorted[lo])*(at-lo);}
function stats(values,indices){const v=[];for(const i of indices){const x=values[i];if(Number.isFinite(x))v.push(x);}v.sort((a,b)=>a-b);if(!v.length)return {n:0,mean:null,p99:null,max:null,min:null};let s=0;for(const x of v)s+=x;return {n:v.length,mean:s/v.length,p99:percentile(v,.99),max:v[v.length-1],min:v[0]};}
const core={UNITS,VIEW_NAMES,unitInfo,unitNames,toUnit,fromUnit,validThresholds,thresholdAreas,normalize,quantize,bandEdges,topThreshold,marchingTriangles,encodeView,decodeView,viewHash,parseViewHash,validFrame,alignedFromWorld,worldFromAligned,rotateToAligned,rotateFromAligned,cameraToAligned,cameraFromAligned,standardViews,nearestIndex,sphereMask,rangeMask,maskIndices,percentile,stats};
root.WssReportCore=core;
if(typeof module!=='undefined'&&module.exports)module.exports=core;
})(typeof globalThis!=='undefined'?globalThis:this);'''


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSS 预测报告</title>
<style>
:root{--bg:#f3f6fa;--card:#fff;--line:#dce4ed;--ink:#203049;--muted:#5f7086;--acc:#0b6bcb;--soft:#f4f7fa}
*{box-sizing:border-box}[hidden]{display:none!important}html,body{height:100%}body{margin:0;font:14px/1.5 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg);display:flex;flex-direction:column;overflow:hidden}
header{flex:0 0 auto;display:flex;align-items:center;gap:14px;padding:8px 16px;background:#fff;border-bottom:1px solid var(--line);flex-wrap:wrap}h1{font-size:17px;margin:0}.sub,small{color:var(--muted);font-size:12px}.actions{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.mode-tabs{display:flex;gap:4px;background:var(--soft);padding:4px;border-radius:9px}.mode-tabs button{border:0;background:transparent;padding:5px 12px;border-radius:6px;font-weight:600;color:#4d6478}.mode-tabs button.on{background:var(--acc);color:#fff}
button,select,input{font:inherit;max-width:100%}button{padding:5px 10px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}button.on{background:var(--acc);color:white;border-color:var(--acc)}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
main{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:350px minmax(0,1fr)}
aside.menu{overflow-y:auto;overflow-x:hidden;min-height:0;padding:10px;background:#f9fbfd;border-right:1px solid var(--line);overscroll-behavior:contain}
details.menu-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:0 12px;margin-bottom:8px}details.menu-card>summary{cursor:pointer;font-weight:650;padding:10px 0;list-style:none;display:flex;align-items:center;font-size:13px}details.menu-card>summary::before{content:"▸";display:inline-block;width:16px;color:var(--acc);transition:transform .15s}details.menu-card[open]>summary::before{transform:rotate(90deg)}details.menu-card>summary::-webkit-details-marker{display:none}details.menu-card>.body{padding:0 0 10px;display:grid;gap:6px;font-size:12px}
.body label{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.body input[type=range]{min-width:0;flex:1}.body select{max-width:100%}.num{width:66px}.row{display:flex;gap:5px;flex-wrap:wrap}.row button{font-size:12px;padding:4px 8px}.hint{color:var(--muted);line-height:1.4}.sep{border-top:1px solid var(--line);margin:4px 0}
.schema-controls{display:grid;gap:4px}.schema-label{font-weight:600}.field-tabs{display:flex;gap:4px;flex-wrap:wrap}.field-tab{font-size:11px;padding:3px 7px;min-height:25px}.field-tab[disabled]{cursor:not-allowed;opacity:.55}.field-tab.active{background:var(--acc);color:#fff;border-radius:6px}.frame-status{line-height:1.35}
.highlight-control output{min-width:34px;text-align:right;font-variant-numeric:tabular-nums}.metric-bars{display:grid;gap:7px;margin:7px 0 4px}.metric-bar{display:grid;grid-template-columns:92px minmax(0,1fr) auto;align-items:center;gap:7px;font-size:11px}.metric-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.metric-track{height:9px;background:#e9eef5;border-radius:8px;overflow:hidden}.metric-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#4b83c4,#c0392b);min-width:2px}.metric-fill.low{background:#4b83c4}.metric-fill.high{background:#df8b36}.metric-fill.very-high{background:#c0392b}.branch-chart .metric-track{height:8px}.branch-chart .metric-bar{grid-template-columns:82px minmax(0,1fr) 46px}.branch-chart .metric-fill{background:#547eb5}
#view{position:relative;background:#eef2f7;min-width:0;height:100%;overflow:hidden}#view>canvas{display:block}#labels div{position:absolute;transform:translate(-50%,-120%);background:#fffe;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12px;pointer-events:none;white-space:nowrap}
#tip{position:absolute;left:12px;bottom:12px;max-width:60%;background:#fffffff0;border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;white-space:pre-line;z-index:2}
#cbar{position:absolute;right:70px;top:45px;width:18px;height:190px;border:1px solid #999;z-index:1}#cbar span{position:absolute;left:23px;font-size:11px;white-space:nowrap}#cbar-unit{position:absolute;right:12px;top:16px;max-width:180px;font-size:12px;text-align:right;z-index:1}.legend-note{position:absolute;right:12px;top:247px;font-size:11px;z-index:1;max-width:220px;text-align:right}
#trust-legend{position:absolute;right:12px;top:290px;background:#fffffff0;border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:11px;z-index:1;display:grid;gap:3px}.sw{display:inline-block;width:12px;height:12px;border:1px solid #999;vertical-align:-2px;margin-right:4px}
.card{border:1px solid var(--line);border-radius:9px;padding:10px 12px;margin-bottom:10px}.card h3{margin:0 0 7px;font-size:13px;font-weight:600}.big{font-size:26px;font-weight:700}.kv{display:grid;grid-template-columns:auto minmax(0,1fr);gap:3px 10px;font-size:12px}.kv span{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:11px}th,td{border-bottom:1px solid var(--line);padding:4px 3px;text-align:right}th:first-child,td:first-child{text-align:left}summary{font-weight:600;cursor:pointer}details>div{margin-top:10px}p{margin:7px 0}.flag{background:#fff4d6;border:1px solid #f0c36d;border-radius:5px;padding:5px 8px;margin:5px 0;font-size:12px}.ok{background:#e6f6ea;border-color:#8fd19e}.unroll-row{margin:8px 0}.unroll-row canvas{width:100%;height:100px;border:1px solid var(--line);display:block}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto}.fallback{position:absolute;inset:40% 20px auto;padding:15px;background:#fff4d6;border-radius:8px;z-index:3}#snapshot{display:none}#print-note{display:none}
.finding{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:6px;align-items:center;text-align:left;padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:#fff;font-size:12px;width:100%}.finding.on{border-color:var(--acc);background:#eaf3fc}.finding .rank{font-weight:700;color:var(--acc)}.finding .sub{display:block}.badge{font-size:10px;padding:1px 6px;border-radius:9px;background:#e9eef5;color:#4d6478;white-space:nowrap}.badge.attention{background:#fde7e3;color:#a53a2a}.badge.note{background:#fff2d5;color:#8b6519}.badge.info{background:#e5f0fa;color:#1766a0}
.prof canvas{width:100%;display:block;border:1px solid var(--line);background:#fff;cursor:crosshair}.prof-legend{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:11px}.prof-legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:3px}
footer{flex:0 0 auto;font-size:11px;color:var(--muted);padding:5px 16px;border-top:1px solid var(--line);background:#fff;display:flex;gap:14px;flex-wrap:wrap}footer b{color:var(--ink)}
.gloss{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;padding:0;margin-left:3px;border-radius:50%;border:1px solid var(--line);background:#f4f7fa;color:var(--acc);font-size:10px;font-weight:700;line-height:1;vertical-align:1px;cursor:help}
.gloss-pop{position:fixed;z-index:20;max-width:320px;background:#fff;border:1px solid var(--line);border-radius:8px;box-shadow:0 6px 20px #0002;padding:8px 10px;font-size:12px;line-height:1.45}.gloss-pop b{display:block;margin-bottom:3px}
.item-list{display:grid;gap:4px}.item{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px;align-items:center;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:#fff;font-size:12px}.item>span,.item>button{overflow-wrap:anywhere;text-align:left}.item .tools{display:flex;gap:3px}.item .tools button{font-size:11px;padding:2px 6px}
.finding-tools{display:flex;gap:3px;flex-wrap:wrap;align-items:center;margin:2px 0 6px}.finding-tools button{font-size:11px;padding:2px 7px}.finding-tools input{flex:1;min-width:80px;font-size:11px;padding:2px 5px}.finding-wrap.rejected .finding{opacity:.5}
#probe-card{position:absolute;left:12px;bottom:118px;max-width:60%;background:#fffffff0;border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;z-index:2;white-space:pre-line;display:grid;gap:4px}
.probe-table{font-size:11px}.mode-row button.on{background:var(--acc);color:#fff}#branch-vis label{font-size:12px;margin-right:6px}
#labels div.meas{background:#e8f6f8;border-color:#5bbcc9}#labels div.annot{background:#fff7e6;border-color:#e0a84a;pointer-events:auto;cursor:pointer;max-width:240px;white-space:normal}
#labels div.flabel{pointer-events:auto;cursor:pointer;font-weight:600}#labels div.flabel.attention{background:#fde7e3f0;border-color:#e2a294;color:#8d3223}#labels div.flabel.note{background:#fff2d5f0;border-color:#e6c47a;color:#7a5814}#labels div.flabel.info{background:#e5f0faf0;border-color:#9dc3e6;color:#175b8c}
#labels div.blabel{background:#eef4fbf0;border-color:#a9c4de;color:#2a4c6b;font-weight:600}#labels div.maxd{background:#f5ecfbf0;border-color:#b98fe0;color:#6c3483;font-weight:600}
#narrative-card p{margin:4px 0}#narrative-card .edited{background:#eaf3fc;border-color:#9dc3e6}
/* compact layout (narrow window / compare-page iframe) */
.compact-only{display:none}.compact .compact-only{display:inline-flex}
.compact header{padding:5px 10px;gap:8px;flex-wrap:nowrap;overflow:hidden}.compact header>div:first-child{display:flex;align-items:baseline;gap:8px;min-width:0;flex:0 1 auto;overflow:hidden}.compact header h1{font-size:14px;white-space:nowrap}.compact #subtitle{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.compact .mode-tabs{flex:0 0 auto}.compact .mode-tabs button{padding:4px 9px;font-size:12px;white-space:nowrap}.compact .actions{flex-wrap:nowrap;gap:4px}.compact .actions button,.compact #menu-toggle{padding:4px 8px;font-size:12px;white-space:nowrap}
.compact main{display:block;position:relative}.compact aside.menu{position:absolute;left:0;top:0;bottom:0;width:min(340px,88%);z-index:6;transform:translateX(-102%);transition:transform .15s;box-shadow:4px 0 18px #16334b26}.compact aside.menu.open{transform:none}.compact #view{height:100%}
.menu-head{display:none}.compact .menu-head{display:flex;justify-content:space-between;align-items:center;margin:0 0 8px;font-weight:650}.menu-head button{font-size:12px;padding:3px 8px}
#menu-tab{display:none}.compact #menu-tab{display:block;position:absolute;left:0;top:50%;transform:translateY(-50%);z-index:3;writing-mode:vertical-rl;padding:10px 4px;border-radius:0 8px 8px 0;border:1px solid var(--line);border-left:0;background:#fffffff0;font-size:12px;letter-spacing:2px;color:#3f5a72}
.compact footer{display:none}
.compact #cbar{right:34px;top:32px;height:140px}.compact #cbar-unit{top:8px;right:8px;font-size:11px;max-width:150px}.compact .legend-note{top:180px;right:8px;max-width:170px;font-size:10px}.compact #trust-legend{top:214px;right:8px;font-size:10px}
.compact #probe-card{left:8px;right:8px;bottom:8px;max-width:none;font-size:11px;padding:4px 8px}.compact #tip{bottom:auto;top:8px;left:8px;max-width:45%;font-size:11px;padding:5px 8px}
@page{size:A4 portrait;margin:10mm}
@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact}body{background:#fff;font-size:9pt;overflow:visible;display:block}header{padding:0 0 5mm}header h1{font-size:14pt}.actions,.mode-tabs,#tip,#probe-card,.gloss,.gloss-pop,.legend-note,#labels,#view>canvas,#cbar,#cbar-unit,#trust-legend,#menu-tab,#menu-toggle,.process-only,.menu-only{display:none!important}main{display:block;height:auto;min-height:0}#view{height:auto;min-height:0;background:white;overflow:visible}#snapshot{display:block;width:100%;max-height:115mm;object-fit:contain}aside.menu{padding:3mm 0 0;overflow:visible;border:0;background:#fff}details.menu-card{border:0;padding:0;margin:0}details.menu-card:not(#menu-stats){display:none}#menu-stats>summary{display:none}#menu-stats>.body{display:block}#panel{display:grid;grid-template-columns:1fr 1fr;gap:3mm}.card{margin:0;padding:2.5mm;border-radius:2mm;break-inside:avoid}.card h3{font-size:9pt;margin-bottom:1mm}.big{font-size:18pt}table{font-size:8pt}.branch-card{grid-column:1/-1}.kv{font-size:8pt}.sub,small{font-size:8pt}#print-note{display:block;grid-column:1/-1;font-size:8pt}footer{border:0;padding:2mm 0}}
</style></head><body>
<header><button type="button" id="menu-toggle" class="compact-only" aria-expanded="false" title="打开菜单">☰ 菜单</button><div><h1>壁面 WSS 预测报告</h1><div class="sub" id="subtitle"></div></div>
<div class="mode-tabs view-mode-controls" role="tablist" aria-label="显示对象"><button type="button" data-v="wss" class="on">WSS 壁面</button><button type="button" data-v="stl">输入 STL</button><button type="button" data-v="cl">中心线</button><button type="button" data-v="cloud">预测点云</button></div>
<div class="actions"><button type="button" id="save-image">保存截图</button><button type="button" id="print-report">打印 / 保存 PDF</button><button type="button" id="layout-toggle" title="在紧凑布局与完整布局之间切换（窄窗口与并排比较时自动紧凑）">紧凑布局</button></div></header>
<main><aside class="menu" id="menu"><div class="menu-head"><span>菜单</span><button type="button" id="menu-close">◂ 收起</button></div>
<details class="menu-card" id="menu-display" open><summary>显示</summary><div class="body">
<div class="schema-controls" aria-label="结果字段与时间帧"><div class="schema-label">结果字段</div><div id="field-tabs" class="field-tabs"></div><label>时间帧 <select id="frame-select" aria-describedby="frame-status" disabled></select></label><small id="frame-status" class="frame-status"></small></div>
<div class="sep"></div>
<label>配色 <select id="cmap"><option value="rainbow">彩虹</option><option value="turbo">Turbo</option><option value="bwr">蓝白红</option></select><select id="bands" title="色带分段"><option value="0">连续</option><option value="4">4 段</option><option value="6">6 段</option><option value="8">8 段</option><option value="12">12 段</option></select></label>
<label>色标范围 <select id="scale-mode"><option value="case">本例 p99</option><option value="fixed">固定上限 · 跨报告保留</option></select></label>
<label id="fixed-control" hidden>固定上限 <input id="fixedmax" class="num" type="number" min="0.01" step="0.5" value="10"> <span id="fixed-unit">Pa</span></label>
<label><input type="checkbox" id="logscale"> 对数色标（下限 0.05 Pa）</label>
<label>单位<button type="button" class="gloss" data-gloss="units_wss">?</button> <select id="units"><option value="Pa">Pa</option><option value="dyn">dyn/cm²</option></select></label>
<small id="scale-description" class="hint"></small>
<div class="sep"></div>
<label>阈值<button type="button" class="gloss" data-gloss="thresholds">?</button> <input id="thr0" class="num" type="number" min="0" step="0.1"> / <input id="thr1" class="num" type="number" min="0" step="0.5"> / <input id="thr2" class="num" type="number" min="0" step="0.5"> <span id="thr-unit">Pa</span></label>
<small class="hint">低 / 高 / 极高阈值；改动即时重算面积占比与等值线，不写入 summary.json。</small>
<label><input type="checkbox" id="contours"> 等值线（三条阈值）</label>
<label><input type="checkbox" id="trust"> 可信区域叠加<button type="button" class="gloss" data-gloss="trust_interpolation_uncovered">?</button> <small id="trust-hint" class="hint"></small></label>
<div class="sep"></div>
<label>分支 <select id="branch"><option value="-1">全部</option></select></label>
<div id="branch-vis" class="row" aria-label="分支显隐"></div><small id="branch-vis-note" class="hint" hidden>已隐藏分支只影响显示；统计不受显隐影响。</small>
<label class="highlight-control">高亮阈值 <input type="range" id="highlight-pct" min="0.5" max="10" step="0.5" value="1"><output id="highlight-value">1%</output></label>
<label><input type="checkbox" id="showtop"> 启用高亮（最高 <span id="highlight-label">1%</span>）</label>
<label><input type="checkbox" id="showpeak" checked> 全场最大值标记</label>
<div class="sep"></div>
<label>自动标注发现 <select id="lbl-findings"><option value="0">关</option><option value="3">前 3</option><option value="5">前 5</option><option value="10">前 10</option></select></label>
<label><input type="checkbox" id="lbl-branches"> 分支名</label>
<small class="hint">发现标签按排名钉在三维上（驳回项与隐藏分支不钉）；分支名钉在各分支中点。标签随导图、拼图与一页纸配图一起输出（需勾选「含界面标签」）。</small>
<div class="sep"></div>
<label>点云特征 <select id="feat"><option value="wss">WSS</option><option value="s">沿程弧长 (mm)</option><option value="th">周向角 (rad)</option><option value="r">半径 (mm)</option><option value="dj">到分叉距离 (mm)</option></select></label>
<div class="row"><label><input type="checkbox" id="showpts"> 叠加点云</label><label><input type="checkbox" id="showcl"> 叠加中心线</label></div>
<div class="sub" id="display-readout">当前视图：WSS 壁面 · 统计：预测点云 · 壁面：Gaussian 插值</div>
<div class="row"><button type="button" id="set-defaults">把当前口径设为默认</button></div><small id="defaults-note" class="hint">默认口径：配色、分段、单位、阈值、透明度、对数、语言；新开报告时生效，视图状态里仍显式记录。</small>
</div></details>
<details class="menu-card" id="menu-findings"><summary>发现</summary><div class="body"><div id="findings-list"></div><small id="findings-note" class="hint"></small><div class="row"><button type="button" id="finding-add">新增发现（点选壁面）</button><button type="button" id="finding-save">保存审阅</button><button type="button" class="gloss" data-gloss="finding_decision">?</button></div><small id="findings-review-note" class="hint"></small></div></details>
<details class="menu-card" id="menu-profiles"><summary>沿程曲线</summary><div class="body prof">
<label>分支 <select id="prof-branch"><option value="">全部叠加</option></select></label>
<label>横轴 <select id="prof-x"><option value="s_from_root_mm">距入口弧长</option><option value="s_local_mm">分支内弧长</option></select></label>
<div class="row" id="maxd-row" hidden><span id="maxd-text"></span><button type="button" id="maxd-fly">飞到</button></div>
<label id="maxd-toggle" hidden><input type="checkbox" id="maxd-ring" checked> 显示最大直径环</label>
<small id="maxd-note" class="hint" hidden>最大直径为该站壁面截面的最大 Feret 直径，等效直径 = 2√(面积/π)；三维上画出该站轮廓环。</small>
<div class="prof-legend" id="prof-legend"></div>
<canvas id="prof-wss" height="300" aria-label="沿程 WSS 曲线"></canvas>
<canvas id="prof-r" height="140" aria-label="沿程半径曲线"></canvas>
<small id="prof-readout" class="hint">点击曲线：三维视图高亮该处 ±2 mm 的壁面并显示数值。</small>
<div class="row"><button type="button" id="prof-clear">清除高亮</button><button type="button" id="prof-svg">导出曲线 SVG</button></div>
<small id="prof-note" class="hint">导出当前分支的两张 SVG：WSS（均值 / p99 / 最低，按当前单位）与半径；三条阈值画成虚线参考。</small>
</div></details>
<details class="menu-card" id="menu-region"><summary>区域统计</summary><div class="body">
<label>方式 <select id="region-mode"><option value="range">分支 + 弧长区间</option><option value="sphere">点选球形刷子</option></select></label>
<div id="region-range"><label>分支 <select id="region-branch"></select></label><label>起点 <input type="range" id="region-s0" min="0" max="100" step="0.5" value="0"><output id="region-s0-v"></output></label><label>终点 <input type="range" id="region-s1" min="0" max="100" step="0.5" value="100"><output id="region-s1-v"></output></label></div>
<div id="region-sphere" hidden><label><input type="checkbox" id="region-pick"> 点击壁面选中心</label><label>半径 <input type="range" id="region-radius" min="1" max="40" step="0.5" value="8"><output id="region-radius-v">8 mm</output></label><small id="region-center" class="hint">尚未选点</small></div>
<div class="row"><button type="button" id="region-apply">计算并高亮</button><button type="button" id="region-clear">清除</button></div>
<div class="kv" id="region-stats"></div><small class="hint">统计基于区域内预测点；估计面积 = 点占比 × 输入壁面面积。</small>
</div></details>
<details class="menu-card" id="menu-measure"><summary>测量与探针</summary><div class="body">
<div class="row mode-row" id="measure-modes"><button type="button" id="measure-distance">距离</button><button type="button" id="measure-arc">弧长</button><button type="button" id="measure-diameter">管径</button><button type="button" id="measure-segment">分段</button><button type="button" id="measure-clear">清空</button></div>
<small id="measure-hint" class="hint">选择模式后在壁面上点击取点；再次点击模式按钮退出。管径按该处中心线切线切出真实截面轮廓，给出最大 / 最小 / 等效直径与截面面积（等效直径 = 2√(面积/π)），轮廓画在三维里；取不到闭合轮廓时回退为内切半径 × 2 并注明<button type="button" class="gloss" data-gloss="local_diameter">?</button>；弧长沿中心线树计算<button type="button" class="gloss" data-gloss="arc_distance">?</button>。</small>
<small id="measure-status" class="hint">未选择模式：点击壁面锁定探针。</small>
<div id="measure-list" class="item-list"></div>
<div class="sep"></div><b>探针记录</b><small class="hint">点击壁面锁定探针后按「记录」；数值来自最近预测点，壁面读值为 Gaussian 插值。</small>
<div id="probe-log"></div><div class="row"><button type="button" id="probe-copy">复制 TSV</button><button type="button" id="probe-csv">导出 CSV</button><button type="button" id="probe-clear">清空记录</button></div>
</div></details>
<details class="menu-card" id="menu-annot"><summary>标注</summary><div class="body">
<div class="row"><button type="button" id="annot-add">在壁面钉标注</button><button type="button" id="annot-save">保存到服务</button></div>
<small id="annot-status" class="hint"></small><div id="annot-list" class="item-list"></div>
</div></details>
<details class="menu-card" id="menu-presets"><summary>预设</summary><div class="body">
<div id="preset-builtin" class="row"></div><small class="hint">内置预设按本例的分支名与发现列表计算。</small>
<div class="sep"></div><label>名称 <input id="preset-name" maxlength="40" placeholder="我的视图"><button type="button" id="preset-save">存为预设</button></label>
<div id="preset-user" class="item-list"></div><small id="preset-note" class="hint"></small>
</div></details>
<details class="menu-card" id="menu-view"><summary>视图</summary><div class="body">
<div class="row" id="std-views"><button type="button" data-view="front">前</button><button type="button" data-view="back">后</button><button type="button" data-view="left">左</button><button type="button" data-view="right">右</button><button type="button" data-view="top">上</button><button type="button" data-view="bottom">下</button><button type="button" id="reset-camera">恢复视角</button></div>
<small id="std-note" class="hint"></small><button type="button" class="gloss" data-gloss="standard_views">?</button>
<label>剖切 <input type="range" id="clip" min="0" max="100" value="100"></label>
<label>壁面不透明度 <input type="range" id="opacity" min="0.2" max="1" step="0.05" value="1"></label>
<div class="sep"></div>
<div class="row"><button type="button" id="view-save">保存视图</button><button type="button" id="view-restore">恢复视图</button><button type="button" id="view-export">导出 JSON</button><button type="button" id="view-import">导入 JSON</button><input type="file" id="view-file" accept="application/json" hidden></div>
<div class="row"><button type="button" id="view-link">复现此图</button><button type="button" id="views-export">导出六视角</button></div>
<small id="view-note" class="hint">视图状态包含相机、色标、阈值、单位、叠加、剖切、分支显隐、测量、标注、探针记录与导图设置；「复现此图」把状态写入链接并复制。</small>
<div class="sep"></div><b>出版级导图</b>
<div class="row"><label>分辨率 <select id="exp-scale"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option></select></label><label>背景 <select id="exp-bg"><option value="white">白</option><option value="transparent">透明</option><option value="current">当前</option></select></label></div>
<div class="row"><label>色标 <select id="exp-cbar"><option value="overlay">叠加</option><option value="none">不含</option><option value="svg">单独 SVG</option></select></label><label>语言 <select id="exp-lang"><option value="zh">中文</option><option value="en">英文</option></select></label><label><input type="checkbox" id="exp-ui"> 含界面标签</label></div>
<div class="row"><button type="button" id="exp-png">导出出版级 PNG</button><button type="button" id="exp-svg">导出色标 SVG</button></div><small id="exp-note" class="hint">离屏渲染当前视图，不含菜单；测量与标注标签随图输出；「导出六视角」同样按此设置。</small>
<div class="sep"></div><b>多视角拼图</b>
<div class="row" id="montage-views" aria-label="拼图视角"></div>
<div class="row"><label>列数 <select id="montage-cols"><option value="2">2</option><option value="3" selected>3</option><option value="4">4</option></select></label><button type="button" id="montage-go">导出拼图 PNG</button></div>
<small id="montage-note" class="hint">按上面的分辨率 / 背景 / 语言设置离屏渲染每一幅，面板标 a/b/c…，右侧一条共用色标。</small>
<div class="sep"></div><b>一页纸配图</b>
<div class="row"><button type="button" id="snap-onepage">生成一页纸配图</button></div>
<small id="snap-note" class="hint"></small>
<small>拖动旋转 · 滚轮缩放 · 右键平移</small>
</div></details>
<details class="menu-card" id="menu-stats"><summary>统计与口径</summary><div class="body"><div id="panel"></div>
<details class="card process-only" id="unroll-details"><summary>分支展开图（s, θ）</summary><div><small>每条分支单独显示，横轴为距入口弧长（mm），纵轴为周向角 −π 至 π。像素汇总显示该像素的最大预测 WSS。</small><div id="unroll-plots"></div></div></details>
</div></details>
</aside>
<div id="view"><div id="labels"></div><button type="button" id="menu-tab" title="打开菜单">菜单</button><img id="snapshot" alt="WSS 壁面视图与当前色标"><div id="cbar-unit"></div><div id="cbar"></div><div class="legend-note">■ 灰色：插值未覆盖 · 淡色：非选中分支 / 高亮阈值以下 / 区域之外</div><div id="trust-legend" hidden></div><div id="probe-card" hidden></div><div id="tip">移动鼠标到壁面读取 Gaussian 插值；统计卡片使用预测点云。</div></div>
</main>
<footer id="footer"></footer>
<noscript>请启用 JavaScript 查看交互报告；summary.json 保留全部统计与运行记录。</noscript>
<!--WSS_META_START--><script id="wss-report-meta" type="application/json">__META__</script><!--WSS_META_END-->
<script id="wss-report-arrays" type="application/json">__ARRAYS__</script>
<script>__THREE__</script><script>__ORBIT__</script>
<script>__CORE__</script>
<script id="wss-glossary" type="application/json">__GLOSSARY__</script>
<script>__COMMON__</script>
<script>
'use strict';
const META=JSON.parse(document.getElementById('wss-report-meta').textContent);
const ARR=JSON.parse(document.getElementById('wss-report-arrays').textContent);
const CORE=window.WssReportCore;
const el=id=>document.getElementById(id);
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(x,d=2){return x===null||x===undefined||!Number.isFinite(+x)?'—':(+x).toFixed(d);}
let UNIT='Pa';
const uf=()=>CORE.unitInfo(UNIT).factor,ul=()=>CORE.unitInfo(UNIT).label;
function fu(x,d=2){return fmt(Number.isFinite(+x)?+x*uf():x,d);}
function bounds(a){let lo=Infinity,hi=-Infinity;for(const v of a){if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}return [lo,hi];}
function dec(b64,T){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer);}
function schemaFields(m){
  const raw=m.fields||(m.results&&m.results.fields);
  if(raw&&typeof raw==='object'&&!Array.isArray(raw)&&Object.keys(raw).length)return raw;
  if(m.wss_field_pa||m.peak)return {wss:{id:'wss',label:'壁面切应力',units:'Pa',location:'wall',kind:'scalar',components:1,array_key:'wss_pa',axis_order:['point'],time_indices:[0],statistics_key:'wss'}};
  return {};
}
function schemaFrames(m){
  const raw=Array.isArray(m.time_axis)?m.time_axis:(m.results&&Array.isArray(m.results.time_axis)?m.results.time_axis:[]);
  if(raw.length)return raw;
  const mf=m.model_frame||{};return [{index:0,label:mf.label||'固定收缩期帧',step:mf.step,time_s:mf.time_s}];
}
const FIELD_SCHEMA=schemaFields(META),FRAME_SCHEMA=schemaFrames(META);
function fieldSupported(id,d){
  // The embedded arrays hold one scalar value per point per field, never a
  // time-by-point tensor.  A familiar field ID alone cannot establish that contract.
  if(!d||typeof d!=='object')return false;
  const scalarPoint=d.kind==='scalar'&&d.components===1&&d.location==='wall'&&
    Array.isArray(d.axis_order)&&d.axis_order.length===1&&d.axis_order[0]==='point'&&
    Array.isArray(d.time_indices)&&d.time_indices.length===1&&Number.isInteger(d.time_indices[0])&&
    FRAME_SCHEMA.some((f,i)=>(f.index??i)===d.time_indices[0]);
  if(id==='wss')return d.id==='wss'&&scalarPoint&&d.units==='Pa'&&d.array_key==='wss_pa'&&!!ARR.pw&&!!ARR.mw;
  // Extra wall scalar fields (e.g. TAWSS / OSI of the three-head release) are embedded under ARR.xf[array_key].
  return scalarPoint&&typeof d.array_key==='string'&&!!(ARR.xf&&ARR.xf[d.array_key]&&ARR.xf[d.array_key].m&&ARR.xf[d.array_key].p);
}
let activeField='wss';const XF={};
// recolor() lives inside initViewer(); the viewer registers it here so the field tabs can repaint the wall.
let recolorActive=null;
function fieldArrays(id){
  if(id!=='wss'&&!(FIELD_SCHEMA[id]&&fieldSupported(id,FIELD_SCHEMA[id])))id='wss';
  if(id==='wss')return {id:'wss',m:MW,p:PW,units:'Pa',isPa:true,label:'WSS',p99:PEAK.p99_pa,max:PEAK.max_pa,log:true};
  const d=FIELD_SCHEMA[id],key=d.array_key;if(!XF[key])XF[key]={m:dec(ARR.xf[key].m,Float32Array),p:dec(ARR.xf[key].p,Float32Array)};
  const disp=d.display&&typeof d.display==='object'?d.display:{};
  return {id,m:XF[key].m,p:XF[key].p,units:d.units||'',isPa:d.units==='Pa',label:d.label||id,p99:disp.p99,max:disp.max,log:disp.log_scale!==false};
}
function af(){return fieldArrays(activeField).isPa?uf():1;}
function al(){const F=fieldArrays(activeField);return F.isPa?ul():F.units;}
function fv(x){const F=fieldArrays(activeField);return F.isPa?fu(x)+' '+ul():fmt(x,3)+' '+F.units;}
function fieldText(id,d){return (d.label||id)+' · '+(d.units||'');}
function renderSchemaControls(){
  const tabs=el('field-tabs'),select=el('frame-select'),status=el('frame-status'); if(!tabs||!select||!status)return;
  tabs.replaceChildren();
  const supportedIds=Object.keys(FIELD_SCHEMA).filter(id=>fieldSupported(id,FIELD_SCHEMA[id]));
  if(!supportedIds.includes(activeField))activeField='wss';
  for(const [id,d0] of Object.entries(FIELD_SCHEMA)){
    const d=d0&&typeof d0==='object'?d0:{}; const supported=fieldSupported(id,d);
    // A single supported field is a status label, not a fake switch; several supported fields are a real switch
    // of the wall colouring only (statistics panels stay on the peak-frame WSS field).
    const switchable=supported&&supportedIds.length>1;
    const item=document.createElement(supported&&!switchable?'span':'button');item.className='field-tab';item.dataset.field=id;item.textContent=fieldText(id,d);
    if(supported){
      if(id===activeField){item.classList.add('active');item.setAttribute('aria-current','true');}
      item.title=switchable?'点击切换壁面着色字段（统计面板仍为峰值 WSS）':'本报告显示的字段';
      if(switchable){item.type='button';item.onclick=()=>{if(activeField===id)return;activeField=id;renderSchemaControls();if(recolorActive)recolorActive();};}
    }
    else{item.type='button';item.disabled=true;item.title='字段描述或数组布局不受当前报告查看器支持；仅显示元数据';}
    tabs.appendChild(item);
  }
  if(!tabs.children.length){const span=document.createElement('span');span.textContent='未提供字段描述';tabs.appendChild(span);}
  select.replaceChildren();FRAME_SCHEMA.forEach((f,i)=>{const o=document.createElement('option');o.value=String(f.index??i);o.textContent=f.label||('frame_'+(f.index??i));select.appendChild(o);});
  // Metadata can list more frames, but this exporter embeds only one array.
  // Keep navigation disabled until arrays and statistics can really switch.
  select.disabled=true;
  const supported=fieldSupported('wss',FIELD_SCHEMA.wss);
  const index=supported?FIELD_SCHEMA.wss.time_indices[0]:null;
  select.value=index===null?'':String(index);
  const f=FRAME_SCHEMA.find((frame,i)=>(frame.index??i)===index)||{};
  const parts=[];if(f.step!=null)parts.push('step '+f.step);if(f.time_s!=null)parts.push(Number(f.time_s).toFixed(3)+' s');
  const current=supported?'当前显示 '+(f.label||('frame_'+index))+(parts.length?' · '+parts.join(' · '):''):'当前字段没有可确认的帧映射';
  status.textContent=(FRAME_SCHEMA.length===1?'单帧结果':'时间轴含 '+FRAME_SCHEMA.length+' 帧')+' · '+current+'；本报告未提供多帧切换';
}
// ---- arrays (decoded once; the statistics panel needs them even without WebGL) ----
const MV=dec(ARR.mv,Float32Array),MF=dec(ARR.mf,Uint32Array),MW=dec(ARR.mw,Float32Array),MS=dec(ARR.ms,Int32Array),MT=ARR.mt?dec(ARR.mt,Uint8Array):null;
const PV=dec(ARR.pv,Float32Array),PW=dec(ARR.pw,Float32Array),PS=dec(ARR.ps,Int32Array),P_S=dec(ARR.p_s,Float32Array),P_TH=dec(ARR.p_th,Float32Array),P_R=dec(ARR.p_r,Float32Array),P_DJ=dec(ARR.p_dj,Float32Array);
const CV=dec(ARR.cv,Float32Array),CR=dec(ARR.cr,Float32Array),CE=dec(ARR.ce,Uint32Array),CS=dec(ARR.cs,Int32Array);
const PEAK=META.peak||{},AREA=Number((META.input_check||{}).area_mm2)||0,BRANCH_NAMES=META.branch_names||{};
const PROFILES=META.profiles&&Array.isArray(META.profiles.branches)&&META.profiles.branches.length?META.profiles:null;
const FINDINGS=META.findings&&Array.isArray(META.findings.items)?META.findings.items.filter(x=>x&&typeof x==='object').slice().sort((a,b)=>(Number(a.rank)||99)-(Number(b.rank)||99)):[];
const FRAME=CORE.validFrame(META.frame_transform)?META.frame_transform:null;
const TRUST=META.trust&&typeof META.trust==='object'?META.trust:null;
let thresholdsPa=CORE.validThresholds((META.wss_field_pa||{}).thresholds_pa)||[.4,4,7];
const KIND_CN={high_wss_cluster:'高 WSS 区',low_wss_cluster:'低 WSS 区',max_wss:'全场最大值',max_diameter:'最大直径',min_radius:'最小半径',max_speed:'最大速度',min_pressure:'最低压力',pressure_drop:'压降',low_speed_region:'低速区'};
const SEV_CN={attention:'关注',note:'提示',info:'几何'};
// ---- §17 morphology / narrative: every key is optional, the panels simply stay away without them ----
const MORPH=META.morphology&&typeof META.morphology==='object'?META.morphology:null;
const AORTA=MORPH&&MORPH.aorta&&typeof MORPH.aorta==='object'?MORPH.aorta:null;
const AMAX=AORTA&&AORTA.max&&typeof AORTA.max==='object'&&Number.isFinite(+AORTA.max.max_diameter_mm)?AORTA.max:null;
const NARRATIVE=META.narrative&&typeof META.narrative==='object'?META.narrative:null;
// Trimmed stations (§17.3): only s_from_root_mm / max_diameter_mm / equivalent_diameter_mm reach the report.
const STATIONS=(()=>{const m=new Map();for(const b of (MORPH&&Array.isArray(MORPH.branches)?MORPH.branches:[])){
  const st=b&&b.stations;if(!st||!Array.isArray(st.s_from_root_mm)||!st.s_from_root_mm.length)continue;
  m.set(Number(b.segment_id),{s:st.s_from_root_mm.map(Number),max:Array.from(st.max_diameter_mm||[],Number),eq:Array.from(st.equivalent_diameter_mm||[],Number)});}
 return m;})();
// §17.3 automatic labels: short bilingual finding captions, severity colours and the count control steps.
const KIND_SHORT={high_wss_cluster:['高 WSS','High WSS'],low_wss_cluster:['低 WSS','Low WSS'],max_wss:['最大 WSS','Max WSS'],max_diameter:['最大直径','Max diameter'],
 min_radius:['最小半径','Min radius'],max_speed:['最大速度','Max speed'],min_pressure:['最低压力','Min pressure'],pressure_drop:['压降','Pressure drop'],
 low_speed_region:['低速区','Low-speed region'],manual:['人工','Manual']};
const FLABEL_FILL={attention:'#fde7e3f0',note:'#fff2d5f0',info:'#e5f0faf0'},FLABEL_STROKE={attention:'#e2a294',note:'#e6c47a',info:'#9dc3e6'};
let LBL={findings:0,branches:false,max_diameter:!!AMAX};
// The state carries the exact count (a preset may ask for any N); the menu only offers 0 / 3 / 5 / 10.
function normLabelCount(v){const n=Math.floor(Number(v));return Number.isFinite(n)&&n>0?Math.min(n,50):0;}
function labelSelectValue(n){return String(n<=0?0:n<=3?3:n<=5?5:10);}
function severityOf(f){return ['attention','note','info'].includes(f&&f.severity)?f.severity:'info';}
// 「F1 高 WSS 21.9 Pa」 — unit aware, English through the shared dictionary.
function findingLabelText(f){const en=LANG==='en',pair=KIND_SHORT[f&&f.kind];
 let head=pair?pair[en?1:0]:String((f&&f.label)||(f&&f.kind)||'');
 if(!pair&&en)head=COMMON.englishLabel(head,'en');
 const value=f&&f.units==='Pa'?fu(f.value)+' '+ul():(f&&f.value!=null&&Number.isFinite(+f.value)?fmt(f.value,2)+(f.units?' '+f.units:''):'');
 return ((f&&f.id)?f.id+' ':'')+head+(value?' '+value:'');}
const branchName=sid=>{const n=BRANCH_NAMES[String(sid)];return typeof n==='string'?n:('分支 '+sid);};
// ---- shared library (contract §12): centreline measurements, presets, glossary, parent-page protocol ----
const COMMON=window.WssReportCommon;
const CLGROUPS=(()=>{try{return COMMON.buildCenterlineGroups({xyz:CV,radius:CR,edges:CE,segment:CS},BRANCH_NAMES);}catch(_){return [];}})();
let LANG='zh';
const bn=sid=>LANG==='en'?COMMON.englishLabel(branchName(sid),'en'):branchName(sid);
const GLOSSARY=(()=>{try{return JSON.parse(document.getElementById('wss-glossary').textContent);}catch(_){return {terms:{}};}})();
const GLOSS=COMMON.glossaryPopover({glossary:GLOSSARY,lang:LANG,mount:document.body});GLOSS.bind(document);
// Online = this very page is being served by our own job service; a downloaded single file stays read-only.
const ONLINE=typeof fetch==='function'&&/^https?:$/.test(String(location.protocol||''))&&/\/api\/jobs\/[^/]+\/report$/.test(String(location.pathname||''));
const jobURL=rel=>new URL(rel,location.href);
let CSRF=null;
async function apiJSON(url,init){const r=await fetch(String(url),Object.assign({credentials:'same-origin'},init||{}));if(r.status===404)return null;if(!r.ok){const e=new Error('HTTP '+r.status);e.status=r.status;throw e;}return r.status===204?null:r.json();}
async function apiToken(){if(CSRF!==null)return CSRF;const s=await apiJSON(jobURL('/api/session'));CSRF=(s&&s.csrf_token)||'';return CSRF;}
async function apiPut(url,body){const token=await apiToken();return apiJSON(url,{method:'PUT',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});}
async function apiPost(url,body){const token=await apiToken();return apiJSON(url,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});}
// ---- view-state v1.1 payloads; they live outside the 3D scope so captureView works without WebGL ----
let MEAS=[],PROBES=[],ANNOTS=[],PRESET_NAME=null,REVIEW={items:{},added:[]},LAST_ACTIVE=-1,TEXT_HOOK=null;
const HIDDEN=new Set();
let REDRAW=()=>{};
function askText(message,def){if(typeof prompt==='function')return prompt(message,def===undefined?'':def);return TEXT_HOOK;}
function copyText(text){try{if(navigator.clipboard&&navigator.clipboard.writeText)return navigator.clipboard.writeText(String(text));}catch(_){}return null;}
function download(href,name){const a=document.createElement('a');a.href=href;a.download=name;a.click();}
function seedReview(){const r=(META.findings&&META.findings.review)||{};
 REVIEW={items:r.items&&typeof r.items==='object'?Object.assign({},r.items):{},added:Array.isArray(r.added)?r.added.slice():[]};applyAdded();}
function applyAdded(){for(const a of REVIEW.added||[]){if(!a||!a.id||FINDINGS.some(f=>f.id===a.id))continue;
  FINDINGS.push({id:a.id,kind:'manual',label:a.text||'人工发现',branch:a.branch||'',segment_id:a.segment_id,value:null,units:'',xyz_mm:a.xyz_mm,s_from_root_mm:a.s_from_root_mm,extent_mm:4,rank:90,severity:a.severity||'note',definition:'审阅人手工标记的发现。'});}}
function reviewOf(f){return (REVIEW.items||{})[f&&f.id]||{};}
function setDecision(id,decision){const cur=REVIEW.items[id]||{};REVIEW.items[id]={decision,note:cur.note||''};renderFindingsList(LAST_ACTIVE);REDRAW();saveReview(false);}
function setNote(id,note){const cur=REVIEW.items[id]||{};REVIEW.items[id]={decision:cur.decision??null,note:String(note||'').slice(0,500)};saveReview(false);}
function removeManual(id){REVIEW.added=(REVIEW.added||[]).filter(x=>x&&x.id!==id);const k=FINDINGS.findIndex(f=>f.id===id);if(k>=0)FINDINGS.splice(k,1);delete REVIEW.items[id];renderFindingsList(-1);REDRAW();saveReview(false);}
let reviewTimer=null;
function saveReview(immediate){
 if(!ONLINE){el('findings-review-note').textContent='离线只读：判定只保存在本页的视图状态里（导出 JSON 可带走）。';return;}
 if(reviewTimer)clearTimeout(reviewTimer);
 const go=()=>{el('findings-review-note').textContent='保存审阅中…';
  apiPut(jobURL('findings_review'),{items:REVIEW.items,added:REVIEW.added}).then(r=>{el('findings-review-note').textContent='已保存审阅'+(r&&r.version?' · 版本 '+r.version:'')+'。';},
   err=>{el('findings-review-note').textContent=err&&err.status===409?'已审阅锁定，发现判定只读。':'保存失败：'+(err&&err.message||err);});};
 if(immediate)go();else reviewTimer=setTimeout(go,800);}
// ---- statistics panel (re-rendered on unit / threshold change) ----
// §17.2 read-only auto conclusion: the reviewer's edit wins, otherwise the generated sentences.
function narrativeLines(){if(!NARRATIVE)return {lines:[],edited:false};
 const edited=typeof NARRATIVE.edited==='string'&&NARRATIVE.edited.trim()?NARRATIVE.edited.trim():null;
 if(edited)return {lines:edited.split(/\r?\n+/).map(x=>x.trim()).filter(Boolean),edited:true};
 const key=LANG==='en'&&Array.isArray(NARRATIVE.en)&&NARRATIVE.en.length?'en':'zh';
 const list=Array.isArray(NARRATIVE[key])?NARRATIVE[key]:[];
 return {lines:list.map(x=>String(x).trim()).filter(Boolean),edited:false};}
function renderPanel(){const m=META,p=PEAK,ic=m.input_check||{},cloud=m.cloud||{};let h='';
 const narrative=narrativeLines();
 if(narrative.lines.length)h+=`<section class="card" id="narrative-card"><h3>结论（参考）<button type="button" class="gloss" data-gloss="narrative">?</button></h3>`
  +(narrative.edited?'<div class="flag edited">审阅人已修改</div>':'')+narrative.lines.map(x=>`<p>${esc(x)}</p>`).join('')
  +`<small>${narrative.edited?'审阅人编辑的结论，自动句子保留在 narrative.json。':'由本次结果自动生成的参考描述，只读，不是诊断结论。'}</small></section>`;
 const fields=FIELD_SCHEMA,frames=FRAME_SCHEMA,model=m.model_release||{};
 const fieldRows=Object.entries(fields).map(([id,value])=>{const d=value||{};return `<tr><td>${esc(d.label||id)}</td><td>${esc(d.units||'—')}</td><td>${esc(d.location||'—')}</td><td>${esc(d.kind||'—')}</td></tr>`;}).join('');
 const modelName=model.name||model.release||m.release||'—', modelVersion=model.version||model.model_family||'—';
 const wssEmbedded=fieldSupported('wss',fields.wss);
 const viewerNote=wssEmbedded&&frames.length===1?'当前查看器显示已嵌入的单帧标量 WSS；其他字段及时间帧保留为元数据。':'字段与时间轴元数据已保留；当前导出未提供可切换的字段数组。';
 h+=`<section class="card"><h3>结果字段与模型</h3><div class="kv"><b>声明字段</b><span>${esc(Object.keys(fields).map(id=>fieldText(id,fields[id]||{})).join('、')||'—')}</span><b>时间轴</b><span>${frames.length===1?'单帧 · '+esc(frames[0].label||'固定收缩期帧'):'多帧 · '+frames.length+' 帧'}</span><b>模型发布</b><span>${esc(modelName)}${modelVersion!=='—'?' · '+esc(modelVersion):''}</span>${model.ensemble_protocol?`<b>集成协议</b><span>${esc(model.ensemble_protocol)}</span>`:''}${Array.isArray(model.weights)?`<b>权重文件</b><span>${model.weights.length} 个</span>`:''}</div><table><thead><tr><th>字段</th><th>单位</th><th>位置</th><th>类型</th></tr></thead><tbody>${fieldRows}</tbody></table><small>${viewerNote}</small></section>`;
 const pct=v=>Math.max(0,Math.min(100,Number.isFinite(+v)?(+v*100):0));
 const bar=(label,value,detail,klass='')=>`<div class="metric-bar"><span class="metric-label">${esc(label)}</span><span class="metric-track"><span class="metric-fill ${klass}" style="width:${pct(value)}%"></span></span><span>${esc(detail)}</span></div>`;
 h+=`<section class="card"><h3>固定收缩期帧 · 空间 p99<button type="button" class="gloss" data-gloss="p99">?</button></h3><div class="big">${fu(p.p99_pa)} ${ul()}</div><small>预测点云第 99 百分位，不代表时间最大值。</small><div class="kv"><b>全场最大值<button type="button" class="gloss" data-gloss="max">?</button></b><span>${fu(p.max_pa)} ${ul()}</span><b>最大值位置</b><span>${esc(p.branch)}，距入口 ${fmt(p.s_from_inlet_mm,0)} mm；距分叉 ${fmt(p.dist_to_junction_mm,0)} mm</span><b>黄色标记</b><span>全场最大值预测点；局部半径 ${fmt(p.local_radius_mm,1)} mm</span></div></section>`;
 const ss=m.surface_statistics||{};
 if(ss.p99_pa!=null) h+=`<section class="card"><h3>壁面面积加权参考<button type="button" class="gloss" data-gloss="area_weighted_p99">?</button></h3><div class="kv"><b>面积加权 p99</b><span>${fu(ss.p99_pa)} ${ul()}</span><b>最高 1% 面积均值</b><span>${fu(ss.top_area_mean_pa)} ${ul()}</span><b>有效覆盖面积</b><span>${fmt(ss.effective_area_mm2/100,1)} cm² / ${fmt(ss.covered_area_fraction*100,1)}%</span></div><small>基于 Gaussian 插值后的完整三角面；未覆盖或部分覆盖面片不外推。主指标仍是预测点云 p99。</small></section>`;
 const quality=m.quality||{};
 if(quality.level){const qclass=quality.level==='good'?'ok':'flag';h+=`<section class="card"><h3>五模型集成质量<button type="button" class="gloss" data-gloss="quality_grade">?</button></h3><div class="flag ${qclass}">${esc(quality.label||quality.level)}</div>${(quality.reasons||[]).map(x=>`<small>${esc(x)}</small>`).join('<br>')}<p><small>逐点 seed 标准差仅写入 quality_audit.json，避免把分散度误读为预测概率。</small></p></section>`;}
 const gate=m.proposal_confidence_gate||{};
 if(gate.status||gate.confirmation_required!=null){const gateText=gate.confirmation_required?'需要人工确认':'已通过自动门控';h+=`<section class="card"><h3>出口命名门控<button type="button" class="gloss" data-gloss="confidence_proxy">?</button></h3><div class="flag ${gate.confirmation_required?'':'ok'}">${esc(gateText)}</div><small>${esc(gate.calibration_status==='validated'?'已绑定独立校准 profile。':'当前几何置信度只是代理值，未作为统计概率使用。')}</small></section>`;}
 const reference=m.reference_assessment||{}, population=reference.population||{};
 if(reference.status||population.status){
   const geometryText=reference.status==='pass'?'在已声明几何参考范围内':reference.status==='review'?'超出已声明几何参考范围，请复核':'未配置几何参考范围';
   const populationText=population.status==='pass'?'已计算同协议 CV3 折外经验分位 '+fmt(population.percentile,1)+'%':population.status==='review'?'人群参照需要复核':'未配置可核验的人群参照';
   h+=`<section class="card"><h3>参考范围与人群位置</h3><div class="kv"><b>几何参考</b><span>${esc(geometryText)}</span><b>人群参照<button type="button" class="gloss" data-gloss="population_percentile">?</button></b><span>${esc(populationText)}</span></div><small>${esc(reference.note||population.note||'参考范围用于复核，不代表校准概率或临床风险。')}</small>${(reference.reasons||[]).length?'<div class="flag">'+esc(reference.reasons.join('；'))+'</div>':''}${(population.reasons||[]).length?'<div class="flag">'+esc(population.reasons.join('；'))+'</div>':''}</section>`;
 }
 const w=CORE.thresholdAreas(PW,thresholdsPa,AREA),t=thresholdsPa,stat=m.wss_field_pa||{};
 h+=`<section class="card"><h3>点数占比与估计面积<button type="button" class="gloss" data-gloss="area_fraction">?</button></h3><div class="metric-bars">${bar('低 WSS < '+fu(t[0],1)+' '+ul(),w.frac_low,fmt(w.frac_low*100,1)+'% · '+fmt(w.area_low_mm2/100,1)+' cm²','low')}${bar('高 WSS > '+fu(t[1],1)+' '+ul(),w.frac_high,fmt(w.frac_high*100,1)+'% · '+fmt(w.area_high_mm2/100,1)+' cm²','high')}${bar('极高 > '+fu(t[2],1)+' '+ul(),w.frac_very_high,fmt(w.frac_very_high*100,1)+'% · '+fmt(w.area_very_high_mm2/100,1)+' cm²','very-high')}</div><div class="kv"><b>均值 / 中位</b><span>${fu(stat.mean)} / ${fu(stat.median)} ${ul()}</span></div><small>估计面积 = 点数占比 × 输入壁面总面积；未按逐面片面积加权。阈值可在「显示」菜单修改，此处按当前阈值即时重算。</small></section>`;
 const branchEntries=Object.entries(m.per_branch||{}),branchMax=Math.max(...branchEntries.map(([,b])=>Number(b.wss_p99_pa)||0),.01);
 h+='<section class="card branch-card"><h3>分支统计 · 预测点云（'+esc(ul())+'）</h3><div class="branch-chart metric-bars">'+branchEntries.map(([name,b])=>bar(name,(Number(b.wss_p99_pa)||0)/branchMax,fu(b.wss_p99_pa)+' '+ul())).join('')+'</div><table><thead><tr><th>分支</th><th>p99</th><th>均值</th><th>最大</th><th>低%</th><th>高%</th><th>估计<br>cm²</th></tr></thead><tbody>';
 for(const [name,b] of branchEntries){let low=b.frac_low,high=b.frac_high;if(Number.isInteger(b.segment_id)){const mask=new Uint8Array(PS.length);for(let i=0;i<PS.length;i++)mask[i]=PS[i]===b.segment_id?1:0;const bw=CORE.thresholdAreas(PW,thresholdsPa,0,mask);if(bw.n){low=bw.frac_low;high=bw.frac_high;}}h+=`<tr><td>${esc(name)}</td><td>${fu(b.wss_p99_pa)}</td><td>${fu(b.wss_mean_pa)}</td><td>${fu(b.wss_max_pa)}</td><td>${fmt(low*100,0)}</td><td>${fmt(high*100,0)}</td><td>${fmt(b.area_mm2/100,1)}</td></tr>`;}
 h+='</tbody></table><small>仅列出 ≥ 10 个预测点的分支；低% / 高% 按当前阈值重算。</small></section>';
 const flags=[...(ic.flags||[]),...(m.flags||[])];
 if(flags.length)h+='<section class="card"><h3>需关注的输入信息</h3>'+flags.map(f=>'<div class="flag">'+esc(f)+'</div>').join('')+'</section>';
 h+='<details class="card process-only"><summary>查看计算过程与显示口径</summary><div>';
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
 h+='<details><summary>完整参数与版本信息</summary><pre>'+esc(JSON.stringify({schema_version:m.schema_version,model_release:m.model_release,time_axis:m.time_axis,fields:m.fields,release_hash:m.release_hash||m.release_sha256,run_parameters:m.run_parameters||m.parameters,mapping:m.mapping,frame_transform:m.frame_transform,statistics_protocol:m.statistics_protocol,interpolation:m.interpolation,feature_contract:m.feature_contract,exports:m.exports||m.export_status},null,2))+'</pre></details></div></details>';
 h+=`<div id="print-note">病例 ${esc(m.case_id)} · 发布包 ${esc(m.release)} · ${esc(m.created_at)}<br>固定收缩期帧的模型预测；统计来自预测点云，壁面显示使用 Gaussian 插值。黄色标记为全场最大值。</div>`;
 el('panel').innerHTML=h;
}
function renderFooter(){const r=META.review||null,fc=META.feature_contract||{},mr=META.model_release||{};const status=r?({reviewed:'已复核',unreviewed:'未复核',reopened:'已重开'}[r.status]||r.status):'未记录复核状态';
 el('footer').innerHTML=`<span>复核<button type="button" class="gloss" data-gloss="review_status">?</button>：<b>${esc(status)}</b>${r&&r.by?' · '+esc(r.by):''}${r&&r.at?' · '+esc(r.at):''}${r&&r.note?' · '+esc(r.note):''}</span><span>发布包<button type="button" class="gloss" data-gloss="release">?</button> <b>${esc(mr.registry_id||mr.name||META.release||'—')}</b></span><span>特征合同<button type="button" class="gloss" data-gloss="feature_contract">?</button> <b>${esc(fc.source_hash?String(fc.source_hash).slice(0,12):'—')}</b>${fc.version?' · '+esc(fc.version):''}</span><span>run_identity<button type="button" class="gloss" data-gloss="run_identity">?</button> <b>${esc(META.run_identity?String(META.run_identity).slice(0,12):'—')}</b></span><span>生成 ${esc(META.created_at||'—')}${META.audit&&META.audit.report_rebuilt_at?' · 报告重建 '+esc(META.audit.report_rebuilt_at):''}</span>`;}
function renderFindingsList(active){const box=el('findings-list');box.replaceChildren();LAST_ACTIVE=active;
 if(!FINDINGS.length){el('findings-note').textContent='本报告未包含发现列表（旧版本结果或分析层未运行）。';return;}
 // Rejected items stay visible but dimmed and sink to the end; the stored order (= index) never changes.
 const order=FINDINGS.map((f,i)=>i).sort((a,b)=>(reviewOf(FINDINGS[a]).decision==='rejected'?1:0)-(reviewOf(FINDINGS[b]).decision==='rejected'?1:0)||a-b);
 for(const i of order){const f=FINDINGS[i],rv=reviewOf(f),manual=f.kind==='manual';
  const wrap=document.createElement('div');wrap.className='finding-wrap'+(rv.decision==='rejected'?' rejected':'');
  const b=document.createElement('button');b.type='button';b.className='finding'+(i===active?' on':'');const value=f.units==='Pa'?fu(f.value)+' '+ul():(f.value!=null?fmt(f.value,2)+' '+(f.units||''):'');
  b.innerHTML=`<span class="rank">${esc(manual?'人':(f.rank??i+1))}</span><span><strong>${esc(f.label||KIND_CN[f.kind]||f.kind)}</strong><small class="sub">${esc(f.branch||'')}${value?' · '+esc(value):''}${f.area_mm2!=null?' · '+fmt(f.area_mm2/100,1)+' cm²':''}</small></span><span class="badge ${esc(manual?'info':(f.severity||'info'))}">${esc(manual?'人工':(SEV_CN[f.severity]||f.severity||''))}</span>`;
  b.title=f.definition||'';b.onclick=()=>window.__selectFinding&&window.__selectFinding(i);wrap.appendChild(b);
  const tools=document.createElement('div');tools.className='finding-tools';
  for(const [key,text] of [['confirmed','确认'],['rejected','驳回'],[null,'未判定']]){const t=document.createElement('button');t.type='button';t.textContent=text;
   if((rv.decision??null)===key)t.className='on';t.onclick=()=>setDecision(f.id,key);tools.appendChild(t);}
  const note=document.createElement('input');note.type='text';note.maxLength=500;note.placeholder='备注';note.value=rv.note||'';note.onchange=()=>setNote(f.id,note.value);tools.appendChild(note);
  if(manual){const d=document.createElement('button');d.type='button';d.textContent='删除';d.onclick=()=>removeManual(f.id);tools.appendChild(d);}
  wrap.appendChild(tools);box.appendChild(wrap);}
 el('findings-note').textContent='点击定位并高亮；再次点击取消。面积为点占比估计。确认 / 驳回写入审阅，驳回项变淡并排到末尾。';
}
function initBranchOptions(){for(const [sid,name] of Object.entries(BRANCH_NAMES)){const o=document.createElement('option');o.value=sid;o.textContent=name;el('branch').appendChild(o);}
 const rb=el('region-branch');for(const b of ARR.unroll_branches||[]){const o=document.createElement('option');o.value=String(b.segment_id);o.textContent=branchName(b.segment_id)+' · '+fmt(b.s_min_mm,0)+'–'+fmt(b.s_max_mm,0)+' mm';rb.appendChild(o);}
 if(PROFILES)for(const b of PROFILES.branches){const o=document.createElement('option');o.value=String(b.segment_id);o.textContent=b.name||branchName(b.segment_id);el('prof-branch').appendChild(o);}
}
function initThresholdInputs(){[0,1,2].forEach(k=>{el('thr'+k).value=String(+(thresholdsPa[k]*uf()).toFixed(3));});el('thr-unit').textContent=ul();el('fixed-unit').textContent=ul();}
renderPanel();renderSchemaControls();renderFooter();seedReview();renderFindingsList(-1);initBranchOptions();initThresholdInputs();
el('subtitle').textContent=META.case_id+' · '+(FRAME_SCHEMA.length===1?'单帧 · ':'多帧 · ')+META.release;
el('print-report').onclick=()=>window.print();
// The 沿程 card also carries the §17.3 maximum-diameter row, so morphology alone is enough to keep it.
if(!PROFILES&&!AMAX){el('menu-profiles').hidden=true;}
if(!PROFILES)for(const id of ['prof-wss','prof-r','prof-svg','prof-clear'])el(id).hidden=true;
if(!FINDINGS.length&&!META.findings){el('menu-findings').hidden=true;}
if(!TRUST||!MT){el('trust').disabled=true;el('trust-hint').textContent='（本报告未含可信区域数组）';}
const MENUS=['menu-display','menu-findings','menu-profiles','menu-region','menu-measure','menu-annot','menu-presets','menu-view','menu-stats'];
for(const id of MENUS){const d=el(id);d.addEventListener('toggle',()=>{if(d.open)for(const o of MENUS)if(o!==id)el(o).open=false;if(id==='menu-stats'&&d.open)window.__drawUnroll&&window.__drawUnroll();if(id==='menu-profiles'&&d.open)window.__drawProfiles&&window.__drawProfiles();});}
function initViewer(){
 const CMAPS={rainbow:[[0,0,143],[0,32,255],[0,160,255],[0,255,255],[64,255,160],[160,255,64],[255,255,0],[255,160,0],[255,64,0],[190,0,0]],turbo:[[48,18,59],[70,107,227],[36,182,213],[37,241,150],[128,254,66],[210,239,35],[253,183,31],[240,107,14],[199,40,6],[122,4,3]],bwr:[[31,78,156],[247,247,247],[192,57,43]]};
 let cmapName='rainbow',bands=0,logScale=false,opacity=1,clipT=1,showContours=false,showTrust=false;
 try{const saved=localStorage.getItem('wss-report-cmap');if(saved&&CMAPS[saved])cmapName=saved;}catch(_){}
 function cmapRaw(t){t=Math.min(1,Math.max(0,t));const stops=CMAPS[cmapName],x=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(x)),u=x-i;return [0,1,2].map(k=>(stops[i][k]+(stops[i+1][k]-stops[i][k])*u)/255);}
 function cmap(t){return cmapRaw(CORE.quantize(t,bands));}
 function vir(t){t=Math.min(1,Math.max(0,t));const c=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],x=t*4,i=Math.min(3,Math.floor(x)),u=x-i;return [0,1,2].map(k=>(c[i][k]+(c[i+1][k]-c[i][k])*u)/255);}
 const view=el('view'),renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.localClippingEnabled=true;view.appendChild(renderer.domElement);
 renderer.domElement.addEventListener('webglcontextlost',ev=>{ev.preventDefault();el('tip').textContent='3D 显示已中断，请刷新报告；统计表仍可查看。';});
 const scene=new THREE.Scene();scene.background=new THREE.Color(0xeef2f7);const camera=new THREE.PerspectiveCamera(35,1,.1,5000),controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
 scene.add(new THREE.HemisphereLight(0xffffff,0x8899aa,.9));const light=new THREE.DirectionalLight(0xffffff,.6);camera.add(light);light.position.set(0,0,1);scene.add(camera);
 let mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];for(let i=0;i<MV.length;i+=3)for(let k=0;k<3;k++){mn[k]=Math.min(mn[k],MV[i+k]);mx[k]=Math.max(mx[k],MV[i+k]);}
 const center=mn.map((v,k)=>(v+mx[k])/2),size=Math.max(...mx.map((v,k)=>v-mn[k]),1),clipPlane=new THREE.Plane(new THREE.Vector3(0,0,-1),1e6);
 const mg=new THREE.BufferGeometry();mg.setAttribute('position',new THREE.BufferAttribute(MV,3));mg.setIndex(new THREE.BufferAttribute(MF,1));mg.computeVertexNormals();const mcol=new Float32Array(MV.length);mg.setAttribute('color',new THREE.BufferAttribute(mcol,3));
 const mmat=new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide,clippingPlanes:[clipPlane],polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1});const mesh=new THREE.Mesh(mg,mmat);scene.add(mesh);
 const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(PV,3));const pcol=new Float32Array(PV.length);pg.setAttribute('color',new THREE.BufferAttribute(pcol,3));const pts=new THREE.Points(pg,new THREE.PointsMaterial({size:.9,vertexColors:true,clippingPlanes:[clipPlane]}));scene.add(pts);
 const cg=new THREE.BufferGeometry();cg.setAttribute('position',new THREE.BufferAttribute(CV,3));cg.setIndex(new THREE.BufferAttribute(CE,1));const ccol=new Float32Array(CV.length),[rmin,rmax]=bounds(CR);for(let i=0;i<CR.length;i++)ccol.set(vir((CR[i]-rmin)/(rmax-rmin||1)),3*i);cg.setAttribute('color',new THREE.BufferAttribute(ccol,3));const cl=new THREE.LineSegments(cg,new THREE.LineBasicMaterial({vertexColors:true,clippingPlanes:[clipPlane]}));scene.add(cl);
 const labels=[];for(const e of META.endpoints||[]){const obj=new THREE.Mesh(new THREE.SphereGeometry(Math.max(1,e.radius_mm*.6),12,8),new THREE.MeshLambertMaterial({color:e.kind==='inlet'?0x1f77b4:0xff7f0e,clippingPlanes:[clipPlane]}));obj.position.set(...e.center_mm);scene.add(obj);const div=document.createElement('div');div.textContent=e.name_cn+' r='+fmt(e.radius_mm,1)+' mm';el('labels').appendChild(div);labels.push({obj,div,kind:'endpoint'});}
 const peakXYZ=Array.isArray(PEAK.xyz_mm)&&PEAK.xyz_mm.length===3?PEAK.xyz_mm:center;
 const peak=new THREE.Mesh(new THREE.SphereGeometry(1.6,12,8),new THREE.MeshBasicMaterial({color:0xffd400,clippingPlanes:[clipPlane]}));peak.position.set(...peakXYZ);scene.add(peak);const peakLabel=document.createElement('div');el('labels').appendChild(peakLabel);labels.push({obj:peak,div:peakLabel,kind:'peak'});
 const marker=new THREE.Mesh(new THREE.SphereGeometry(1,12,8),new THREE.MeshBasicMaterial({color:0x00c2d8,transparent:true,opacity:.75,depthTest:false}));marker.visible=false;marker.renderOrder=3;scene.add(marker);const markerLabel=document.createElement('div');el('labels').appendChild(markerLabel);labels.push({obj:marker,div:markerLabel,kind:'marker'});
 let contourObj=null;
 let viewMode='wss',feat='wss',branchSel=-1,fixedMax=10,scaleMode='case',vmax=Math.max(Number(PEAK.p99_pa)||0,.01),highlightPct=1,highlightThreshold=Number(PEAK.p99_pa)||0,highlightEnabled=false,topIndices=[],top=null;
 let hl={vertexMask:null,pointIndices:null,label:''},hlPts=null,activeFinding=-1,NEAR=null,VS=null;
 function near(){if(!NEAR){NEAR=CORE.nearestIndex(PV,MV,Math.max(0.5,(Number((META.cloud||{}).spacing_mm)||0.5)*3));VS=new Float32Array(MV.length/3);for(let v=0;v<VS.length;v++)VS[v]=NEAR[v]>=0?P_S[NEAR[v]]:NaN;}return NEAR;}
 // ---- C10 branch visibility (display only; every statistic keeps all branches) ----
 const MESH_SEGS=Array.from(new Set(Array.from(MS))).sort((a,b)=>a-b),branchBoxes=[];
 function rebuildBranchFilter(){
  // A triangle disappears only when all three of its vertices sit on hidden branches.
  if(!HIDDEN.size){mg.setIndex(new THREE.BufferAttribute(MF,1));pg.setIndex(null);}
  else{const keep=[];for(let t=0;t<MF.length;t+=3){const a=MF[t],b=MF[t+1],c=MF[t+2];if(HIDDEN.has(MS[a])&&HIDDEN.has(MS[b])&&HIDDEN.has(MS[c]))continue;keep.push(a,b,c);}
   mg.setIndex(new THREE.BufferAttribute(new Uint32Array(keep),1));
   const pk=[];for(let i=0;i<PS.length;i++)if(!HIDDEN.has(PS[i]))pk.push(i);pg.setIndex(new THREE.BufferAttribute(new Uint32Array(pk),1));}
  el('branch-vis-note').hidden=false;}
 function setBranchHidden(sid,hide){sid=Number(sid);if(hide)HIDDEN.add(sid);else HIDDEN.delete(sid);
  for(const box of branchBoxes)if(box.sid===sid)box.input.checked=!hide;
  rebuildBranchFilter();drawProfiles();redrawOverlays();}
 function initBranchVis(){const box=el('branch-vis');box.replaceChildren();branchBoxes.length=0;
  for(const sid of MESH_SEGS){const lab=document.createElement('label'),cb=document.createElement('input');cb.type='checkbox';cb.checked=!HIDDEN.has(sid);cb.dataset.sid=String(sid);
   cb.onchange=()=>setBranchHidden(sid,!cb.checked);lab.appendChild(cb);const t=document.createElement('span');t.textContent=' '+branchName(sid);lab.appendChild(t);box.appendChild(lab);branchBoxes.push({sid,input:cb});}
  if(MESH_SEGS.length>1)el('branch-vis-note').hidden=false;}
 // ---- C7 / C8 three-dimensional overlays: measurement polylines, annotation pins and their HTML labels ----
 const overlay=new THREE.Group();scene.add(overlay);
 function makeLabel(cls,text,obj,kind){const div=document.createElement('div');div.className=cls;div.textContent=text;el('labels').appendChild(div);labels.push({obj,div,kind});return div;}
 function dropLabels(kind){for(let i=labels.length-1;i>=0;i--)if(labels[i].kind===kind){const d=labels[i].div;if(d&&d.remove)d.remove();labels.splice(i,1);}}
 function anchorAt(p){const o=new THREE.Object3D();o.position.set(p[0],p[1],p[2]);overlay.add(o);return o;}
 function sphereAt(p,color,r){const m=new THREE.Mesh(new THREE.SphereGeometry(r||Math.max(.5,size*.004),10,8),new THREE.MeshBasicMaterial({color,depthTest:false}));m.position.set(p[0],p[1],p[2]);m.renderOrder=4;overlay.add(m);return m;}
 // §15.14 closed cross-section contour; LineLoop when three.js offers it, otherwise a Line back to the first point.
 function loopLine(points,color){const Ctor=THREE.LineLoop||THREE.Line;if(!Ctor||points.length<3)return null;
  const list=THREE.LineLoop?points:points.concat([points[0]]),pos=new Float32Array(list.length*3);
  list.forEach((p,i)=>pos.set([p[0],p[1],p[2]],3*i));
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  const o=new Ctor(g,new THREE.LineBasicMaterial({color,depthTest:false}));o.renderOrder=4;overlay.add(o);return o;}
 function polyline(points,color){const n=Math.max(0,points.length-1),pos=new Float32Array(n*6);for(let i=0;i<n;i++){pos.set(points[i],6*i);pos.set(points[i+1],6*i+3);}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));const o=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color,depthTest:false}));o.renderOrder=4;overlay.add(o);return o;}
 // §15.14 diameter labels carry the section metrics, so they are written here instead of the shared library.
 function diameterLabel(m,lang){const en=lang==='en',head=COMMON.englishLabel('diameter',lang);
  const where=m.branch?' · '+(en?COMMON.englishLabel(m.branch,'en'):m.branch):'';
  if(m.method!=='contour')return head+' ≈ '+fmt(m.value_mm,1)+' mm'+(en?' (2 × inscribed radius)':'（内切半径×2）')+where;
  return (en?'Max diameter ':'最大直径 ')+fmt(m.max_diameter_mm,1)+(en?' / min ':' / 最小 ')+fmt(m.min_diameter_mm,1)
   +(en?' / equiv ':' / 等效 ')+fmt(m.equivalent_diameter_mm,1)+' mm · '+(en?'area ':'面积 ')+fmt(m.area_mm2,1)+' mm²'+where
   +(m.synthetic?(en?' · gap closed':'（轮廓有缺口，已闭合）'):'');}
 function labelFor(m,lang){if(m&&m.kind==='diameter'&&m.method)return diameterLabel(m,lang);
  return lang==='en'?COMMON.measurementLabel(m,'en'):(m.label||COMMON.measurementLabel(m,'zh'));}
 const measLabel=m=>labelFor(m,LANG);
 function redrawOverlays(){
  for(let i=overlay.children.length-1;i>=0;i--){const o=overlay.children[i];overlay.remove(o);if(o.geometry&&o.geometry.dispose)o.geometry.dispose();if(o.material&&o.material.dispose)o.material.dispose();}
  dropLabels('meas');dropLabels('annot');
  for(const m of MEAS){const path=(Array.isArray(m.path_xyz)&&m.path_xyz.length>1)?m.path_xyz:m.points;
   if(path.length>1)polyline(path,0x0f8ea0);for(const p of m.points)sphereAt(p,0x0f8ea0);
   if(m.kind==='diameter'&&Array.isArray(m.polygon_world)&&m.polygon_world.length>2)loopLine(m.polygon_world,0x0f8ea0);
   makeLabel('meas',measLabel(m),anchorAt(path[Math.floor(path.length/2)]||m.points[0]),'meas');}
  const lift=Math.max(2,size*.05);
  for(const a of ANNOTS){const p=(a.xyz_mm||[0,0,0]).map(Number),tip=[p[0],p[1],p[2]+lift];
   polyline([p,tip],0xe0a84a);sphereAt(p,0xe0a84a,Math.max(.4,size*.003));
   makeLabel('annot',a.id+' · '+a.text,anchorAt(tip),'annot').onclick=()=>editAnnot(a.id);}
  renderAutoLabels();}
 REDRAW=redrawOverlays;
 // ---- §17.3 automatic labels: top-N findings, branch names, maximum-diameter ring ----
 // AUTO_CHIPS mirrors what is pinned in 3D so the export path can composite the very same chips.
 let AUTO_CHIPS=[];
 const MAXD_COLOR=0x8e44ad;
 function rankedFindings(){return FINDINGS.map((f,i)=>({f,i}))
  .filter(x=>x.f&&reviewOf(x.f).decision!=='rejected'&&Array.isArray(x.f.xyz_mm)&&x.f.xyz_mm.length===3&&x.f.xyz_mm.every(v=>Number.isFinite(+v)))
  .sort((a,b)=>(Number(a.f.rank)||99)-(Number(b.f.rank)||99)||a.i-b.i);}
 // Branch midpoint = the centreline row nearest 50 % of that branch's own arc length.
 function branchMidpoint(g){const k=g&&g.s?g.s.length:0;if(!k)return null;const half=g.s[k-1]/2;let best=0,bd=Infinity;
  for(let j=0;j<k;j++){const d=Math.abs(g.s[j]-half);if(d<bd){bd=d;best=j;}}
  return [g.xyz[3*best],g.xyz[3*best+1],g.xyz[3*best+2]];}
 function maxDiameterRing(){if(!AMAX||!Array.isArray(AMAX.polygon_world))return [];
  return AMAX.polygon_world.filter(q=>Array.isArray(q)&&q.length===3&&q.every(v=>Number.isFinite(+v))).map(q=>q.map(Number));}
 function renderAutoLabels(){
  AUTO_CHIPS=[];dropLabels('flabel');dropLabels('blabel');dropLabels('maxd');
  const n=normLabelCount(LBL.findings);
  if(n>0)for(const item of rankedFindings().slice(0,n)){const f=item.f;
   if(Number.isFinite(+f.segment_id)&&HIDDEN.has(+f.segment_id))continue;   // C10: a hidden branch hides its labels
   const p=f.xyz_mm.map(Number),text=findingLabelText(f),sev=severityOf(f);
   makeLabel('flabel '+sev,text,anchorAt(p),'flabel').onclick=()=>window.__selectFinding(item.i);
   AUTO_CHIPS.push({kind:'flabel',p,text,fill:FLABEL_FILL[sev],stroke:FLABEL_STROKE[sev]});}
  if(LBL.branches)for(const g of CLGROUPS){if(HIDDEN.has(Number(g.segment_id)))continue;
   const p=branchMidpoint(g);if(!p)continue;const name=g.name||branchName(g.segment_id);
   const text=LANG==='en'?COMMON.englishLabel(name,'en'):name;
   makeLabel('blabel',text,anchorAt(p),'blabel');AUTO_CHIPS.push({kind:'blabel',p,text,fill:'#eef4fbf0',stroke:'#a9c4de'});}
  if(AMAX&&LBL.max_diameter){const ring=maxDiameterRing();
   if(ring.length>2)loopLine(ring,MAXD_COLOR);
   const at=Array.isArray(AMAX.xyz_mm)&&AMAX.xyz_mm.length===3&&AMAX.xyz_mm.every(v=>Number.isFinite(+v))?AMAX.xyz_mm.map(Number):(ring.length?ring[0]:null);
   if(at){const text=(LANG==='en'?'Max diameter ':'最大直径 ')+fmt(AMAX.max_diameter_mm,1)+' mm';
    makeLabel('maxd',text,anchorAt(at),'maxd');AUTO_CHIPS.push({kind:'maxd',p:at,text,fill:'#f5ecfbf0',stroke:'#b98fe0',ring:ring.length>2?ring:null});}}}
 el('lbl-findings').value=labelSelectValue(normLabelCount(LBL.findings));el('lbl-branches').checked=!!LBL.branches;
 el('lbl-findings').onchange=e=>{LBL.findings=normLabelCount(e.target.value);e.target.value=String(LBL.findings);redrawOverlays();};
 el('lbl-branches').onchange=e=>{LBL.branches=!!e.target.checked;redrawOverlays();};
 if(AMAX){const at=Array.isArray(AMAX.xyz_mm)?AMAX.xyz_mm.map(Number):null;
  const from=Number.isFinite(+AMAX.distance_from_inlet_mm)?+AMAX.distance_from_inlet_mm:+AMAX.s_from_root_mm;
  el('maxd-row').hidden=false;el('maxd-toggle').hidden=false;el('maxd-note').hidden=false;
  el('maxd-text').textContent='最大直径 '+fmt(AMAX.max_diameter_mm,1)+' mm（等效 '+fmt(AMAX.equivalent_diameter_mm,1)+' mm）· 入口下 '+fmt(from,0)+' mm ·';
  el('maxd-ring').checked=!!LBL.max_diameter;
  el('maxd-ring').onchange=e=>{LBL.max_diameter=!!e.target.checked;redrawOverlays();};
  el('maxd-fly').onclick=()=>{if(at&&at.length===3&&at.every(Number.isFinite))flyTo(at,Math.max((Number(AMAX.max_diameter_mm)||6)/2,3));
   el('tip').textContent='最大直径站 '+fmt(AMAX.max_diameter_mm,1)+' mm · 距入口 '+fmt(from,0)+' mm';};}
 // ---- C7 measurement modes ----
 const NEED={distance:2,arc:2,diameter:1,segment:2},MODE_HINT={distance:'距离：在壁面点击两点。',arc:'弧长：在壁面点击两点，沿中心线树求和。',diameter:'管径：在壁面点击一点，按中心线切线切出真实截面。',segment:'分段：在同一分支上点击两点。'};
 let measureMode='',pendingPts=[],pickTask='';
 const measStatus=t=>{el('measure-status').textContent=t;};
 function setMeasureMode(mode){measureMode=measureMode===mode?'':mode;pendingPts=[];pickTask='';
  for(const k of Object.keys(NEED))el('measure-'+k).classList.toggle('on',measureMode===k);
  measStatus(measureMode?MODE_HINT[measureMode]:'未选择模式：点击壁面锁定探针。');if(measureMode)el('menu-measure').open=true;}
 for(const k of Object.keys(NEED))el('measure-'+k).onclick=()=>setMeasureMode(k);
 el('measure-clear').onclick=()=>{MEAS=[];pendingPts=[];renderMeasList();redrawOverlays();measStatus('已清空测量。');};
 function arcPolyline(a){if(!a||!a.same_branch||!a.a||!a.b)return null;const g=CLGROUPS[a.a.group_index];if(!g)return null;
  const first=a.a.s<=a.b.s?a.a:a.b,last=first===a.a?a.b:a.a,out=[first.xyz.slice()];
  for(let j=0;j<g.s.length;j++)if(g.s[j]>first.s&&g.s[j]<last.s)out.push([g.xyz[3*j],g.xyz[3*j+1],g.xyz[3*j+2]]);
  out.push(last.xyz.slice());return out;}
 // §15.14: the pick is projected onto the centreline, the local tangent becomes the section normal.
 function centerlineTangent(pr){const g=CLGROUPS[pr.group_index];if(!g)return null;const k=g.s.length;if(k<2)return null;
  const j=Math.max(0,Math.min(k-1,Number(pr.row_local)||0)),a=Math.max(0,j-1),b=Math.min(k-1,j+1);
  const v=[g.xyz[3*b]-g.xyz[3*a],g.xyz[3*b+1]-g.xyz[3*a+1],g.xyz[3*b+2]-g.xyz[3*a+2]],n=Math.hypot(v[0],v[1],v[2]);
  return n>1e-9?v.map(x=>x/n):null;}
 // A real lumen contour when the plane closes one; otherwise the inscribed-radius fallback, always marked as such.
 function diameterAt(p){const pr=COMMON.projectToCenterline(CLGROUPS,p);if(!pr)return null;
  const r=Number(pr.radius_mm)||0,tangent=centerlineTangent(pr),lj=COMMON.localDiameter(CLGROUPS,p);
  const base={branch:pr.name,segment_id:pr.segment_id,radius_mm:r,xyz:pr.xyz,dist_to_junction_mm:lj?lj.dist_to_junction_mm:null};
  if(tangent&&MF.length){let cs=null;
   try{cs=COMMON.crossSection({vertices:MV,faces:MF,origin:pr.xyz,normal:tangent,maxDist:Math.max(4*r,1)});}catch(_){cs=null;}
   const m=cs&&cs.found&&!cs.open?cs.metrics:null;
   if(m&&m.area_mm2>0&&Number.isFinite(m.equivalent_diameter_mm))
    return Object.assign(base,{method:'contour',max_diameter_mm:m.max_diameter_mm,min_diameter_mm:m.min_diameter_mm,
     equivalent_diameter_mm:m.equivalent_diameter_mm,area_mm2:m.area_mm2,synthetic:!!cs.synthetic,
     polygon_world:(cs.polygon_world||[]).map(q=>[+q[0].toFixed(3),+q[1].toFixed(3),+q[2].toFixed(3)])});}
  const d=2*r;
  return Object.assign(base,{method:'inscribed',max_diameter_mm:d,min_diameter_mm:d,equivalent_diameter_mm:d,area_mm2:Math.PI*r*r,synthetic:false,polygon_world:null});}
 function buildMeasurement(kind,pts){const id=COMMON.newId({distance:'D',arc:'C',diameter:'R',segment:'S'}[kind]||'M',MEAS),at=new Date().toISOString();
  if(kind==='diameter'){const d=diameterAt(pts[0]);if(!d){measStatus('本报告没有可用中心线，无法计算管径。');return null;}
   const r3=x=>Number.isFinite(+x)?+(+x).toFixed(3):null;
   const m={id,kind,points:pts,value_mm:r3(d.equivalent_diameter_mm),branch:d.branch,segment_id:d.segment_id,created_at:at,
    method:d.method,max_diameter_mm:r3(d.max_diameter_mm),min_diameter_mm:r3(d.min_diameter_mm),
    equivalent_diameter_mm:r3(d.equivalent_diameter_mm),area_mm2:r3(d.area_mm2),radius_mm:r3(d.radius_mm)};
   if(d.synthetic)m.synthetic=true;
   if(Array.isArray(d.polygon_world)&&d.polygon_world.length>2)m.polygon_world=d.polygon_world;
   m.label=diameterLabel(m,'zh')+(Number.isFinite(d.dist_to_junction_mm)?' · 距分叉 '+fmt(d.dist_to_junction_mm,1)+' mm':'');
   return m;}
  if(kind==='distance'){const v=COMMON.straightDistance(pts[0],pts[1]),pr=COMMON.projectToCenterline(CLGROUPS,pts[0]);
   return {id,kind,points:pts,value_mm:v,branch:pr?pr.name:'',segment_id:pr?pr.segment_id:null,created_at:at,label:'直线距离 '+fmt(v,1)+' mm'+(pr?'（'+pr.name+'）':'')};}
  const a=COMMON.arcDistance(CLGROUPS,pts[0],pts[1]);
  if(!a||!Number.isFinite(a.value_mm)){measStatus('两点不在同一条中心线树上，无法求弧长。');return null;}
  if(kind==='segment'&&!a.same_branch){measStatus('分段长度需要同一分支上的两点；请重新取点，或改用「弧长」。');return null;}
  return {id,kind,points:pts,value_mm:a.value_mm,branch:a.a?a.a.name:'',segment_id:a.a?a.a.segment_id:null,path:(a.path||[]).slice(),same_branch:!!a.same_branch,path_xyz:arcPolyline(a),created_at:at,
   label:(kind==='segment'?'分段长度 ':'弧长距离 ')+fmt(a.value_mm,1)+' mm · '+(a.same_branch?'同分支 '+(a.a?a.a.name:''):'跨分支经 '+(a.path||[]).map(bn).join(' → '))};}
 function addMeasurePick(p){pendingPts.push(p.slice());const need=NEED[measureMode]||2;
  if(pendingPts.length<need){measStatus('已取 '+pendingPts.length+' / '+need+' 点，继续在壁面点击。');return;}
  const pts=pendingPts.slice(0,need);pendingPts=[];const m=buildMeasurement(measureMode,pts);if(!m)return;
  MEAS.push(m);renderMeasList();redrawOverlays();measStatus(m.label);}
 function renderMeasList(){const box=el('measure-list');box.replaceChildren();
  for(const m of MEAS){const row=document.createElement('div');row.className='item';const s=document.createElement('span');s.textContent=m.id+' · '+measLabel(m);row.appendChild(s);
   const tools=document.createElement('span');tools.className='tools';
   const copy=document.createElement('button');copy.type='button';copy.textContent='复制';copy.onclick=()=>{copyText(m.id+'\t'+measLabel(m)+'\t'+(+m.value_mm).toFixed(3)+'\tmm');measStatus('已复制 '+m.id+'。');};
   const del=document.createElement('button');del.type='button';del.textContent='删除';del.onclick=()=>{MEAS=MEAS.filter(x=>x!==m);renderMeasList();redrawOverlays();};
   tools.append(copy,del);row.appendChild(tools);box.appendChild(row);}}
 // ---- C11 probe log ----
 let probeSel=-1;
 function lockProbe(p){const i=CORE.nearestIndex(PV,new Float32Array([p[0],p[1],p[2]]),Math.max(.5,(Number((META.cloud||{}).spacing_mm)||.5)*3))[0],card=el('probe-card');
  if(!(i>=0)){probeSel=-1;card.hidden=true;return;}
  probeSel=i;card.replaceChildren();
  const txt=document.createElement('span');txt.textContent='探针 · '+(activeField==='wss'?'WSS '+fu(PW[i])+' '+ul():fieldArrays(activeField).label+' '+fv(fieldArrays(activeField).p[i]))+'\n'+bn(PS[i])+' · 弧长 '+fmt(P_S[i],1)+' mm · 半径 '+fmt(P_R[i],1)+' mm\n('+[0,1,2].map(k=>fmt(PV[3*i+k],1)).join(', ')+') mm';
  const row=document.createElement('div');row.className='row';
  const add=document.createElement('button');add.type='button';add.textContent='记录';add.onclick=recordProbe;
  const close=document.createElement('button');close.type='button';close.textContent='关闭';close.onclick=()=>{card.hidden=true;probeSel=-1;};
  row.append(add,close);card.append(txt,row);card.hidden=false;}
 function recordProbe(){if(!(probeSel>=0))return;const i=probeSel;
  PROBES.push({id:COMMON.newId('P',PROBES),xyz_mm:[0,1,2].map(k=>+PV[3*i+k].toFixed(3)),branch:branchName(PS[i]),segment_id:PS[i],s_from_root_mm:+P_S[i].toFixed(2),radius_mm:+P_R[i].toFixed(3),values:{wss_pa:+PW[i].toFixed(4)},created_at:new Date().toISOString()});
  renderProbeLog();measStatus('已记录 '+PROBES.length+' 行探针。');}
 function renderProbeLog(){const box=el('probe-log');box.replaceChildren();
  if(!PROBES.length){const s=document.createElement('small');s.className='hint';s.textContent='还没有探针记录。';box.appendChild(s);return;}
  const t=document.createElement('table');t.className='probe-table';
  t.innerHTML='<thead><tr><th>编号</th><th>分支</th><th>s mm</th><th>r mm</th><th>WSS '+esc(ul())+'</th><th></th></tr></thead><tbody>'+PROBES.map(r=>`<tr><td>${esc(r.id)}</td><td>${esc(r.branch)}</td><td>${fmt(r.s_from_root_mm,1)}</td><td>${fmt(r.radius_mm,1)}</td><td>${fu(r.values.wss_pa)}</td><td><button type="button" data-del="${esc(r.id)}">×</button></td></tr>`).join('')+'</tbody>';
  t.addEventListener('click',ev=>{const id=ev&&ev.target&&ev.target.dataset&&ev.target.dataset.del;if(!id)return;PROBES=PROBES.filter(x=>x.id!==id);renderProbeLog();});box.appendChild(t);}
 el('probe-copy').onclick=()=>{copyText(COMMON.probeToTSV(PROBES,LANG));measStatus('已复制 '+PROBES.length+' 行探针记录（TSV）。');};
 el('probe-csv').onclick=()=>download('data:text/csv;charset=utf-8,'+encodeURIComponent(COMMON.probeToCSV(PROBES,LANG)),fileStem()+'-probes.csv');
 el('probe-clear').onclick=()=>{PROBES=[];renderProbeLog();};
 // ---- C8 annotations ----
 const annotStatus=t=>{el('annot-status').textContent=t;};
 function renderAnnots(){const box=el('annot-list');box.replaceChildren();
  for(const a of ANNOTS){const row=document.createElement('div');row.className='item';const s=document.createElement('span');
   s.textContent=a.id+' · '+(a.branch||'')+(Number.isFinite(a.s_from_root_mm)?' s '+fmt(a.s_from_root_mm,0)+' mm':'')+' · '+a.text;row.appendChild(s);
   const tools=document.createElement('span');tools.className='tools';
   const ed=document.createElement('button');ed.type='button';ed.textContent='编辑';ed.onclick=()=>editAnnot(a.id);
   const del=document.createElement('button');del.type='button';del.textContent='删除';del.onclick=()=>removeAnnot(a.id);
   tools.append(ed,del);row.appendChild(tools);box.appendChild(row);}}
 function addAnnotation(p){if(ANNOTS.length>=50){annotStatus('标注最多 50 条。');return;}
  const raw=askText('标注文字（≤ 200 字）','');if(raw===null||raw===undefined)return;const text=String(raw).trim().slice(0,200);if(!text)return;
  const pr=COMMON.projectToCenterline(CLGROUPS,p),root=COMMON.arcFromRoot(CLGROUPS,p);
  ANNOTS.push({id:COMMON.newId('A',ANNOTS),xyz_mm:p.map(v=>+v.toFixed(3)),text,color:'#d97706',branch:pr?pr.name:'',segment_id:pr?pr.segment_id:null,s_from_root_mm:root?+root.s_from_root_mm.toFixed(2):0,created_at:new Date().toISOString()});
  renderAnnots();redrawOverlays();annotStatus(ONLINE?'已新增；点「保存到服务」写入任务目录。':'已新增（离线，只随视图状态导出）。');}
 function editAnnot(id){const a=ANNOTS.find(x=>x.id===id);if(!a)return;const raw=askText('修改标注（清空则删除）',a.text);if(raw===null||raw===undefined)return;
  const text=String(raw).trim().slice(0,200);if(!text){removeAnnot(id);return;}a.text=text;renderAnnots();redrawOverlays();}
 function removeAnnot(id){ANNOTS=ANNOTS.filter(x=>x.id!==id);renderAnnots();redrawOverlays();}
 el('annot-add').onclick=()=>{setMeasureMode('');pickTask='annot';annotStatus('请在壁面上点击要钉标注的位置。');};
 el('annot-save').onclick=()=>{if(!ONLINE){annotStatus('离线报告不能写回服务端；标注随视图状态导出。');return;}annotStatus('保存中…');
  apiPut(jobURL('annotations'),{items:ANNOTS}).then(r=>{annotStatus('已保存 '+ANNOTS.length+' 条标注'+(r&&r.version?' · 版本 '+r.version:'')+'。');},
   err=>{annotStatus(err&&err.status===409?'已审阅锁定，标注只读':'保存失败：'+(err&&err.message||err));});};
 function loadAnnotations(){
  if(!ONLINE){const src=(META.annotations&&Array.isArray(META.annotations.items))?META.annotations.items:[];ANNOTS=src.map(x=>Object.assign({},x));el('annot-save').disabled=true;
   annotStatus('离线只读（内嵌副本）· '+ANNOTS.length+' 条');renderAnnots();redrawOverlays();return;}
  apiJSON(jobURL('files/annotations.json')).then(doc=>{ANNOTS=doc&&Array.isArray(doc.items)?doc.items:[];
   annotStatus(ANNOTS.length?'已载入 '+ANNOTS.length+' 条标注。':'尚无标注；点「在壁面钉标注」新增。');renderAnnots();redrawOverlays();},
   err=>{annotStatus('标注读取失败：'+(err&&err.message||err));});}
 // ---- C16 manual findings ----
 el('finding-add').onclick=()=>{setMeasureMode('');pickTask='finding';el('findings-review-note').textContent='请在壁面上点击新发现的位置。';};
 el('finding-save').onclick=()=>saveReview(true);
 function addManualFinding(p){const raw=askText('新发现的说明（≤ 500 字）','');if(raw===null||raw===undefined)return;const text=String(raw).trim().slice(0,500);if(!text)return;
  if((REVIEW.added||[]).length>=30){el('findings-review-note').textContent='手工发现最多 30 条。';return;}
  const pr=COMMON.projectToCenterline(CLGROUPS,p),root=COMMON.arcFromRoot(CLGROUPS,p);
  REVIEW.added=(REVIEW.added||[]).concat([{id:COMMON.newId('M',REVIEW.added||[]),xyz_mm:p.map(v=>+v.toFixed(3)),branch:pr?pr.name:'',segment_id:pr?pr.segment_id:null,s_from_root_mm:root?+root.s_from_root_mm.toFixed(2):0,text,kind:'manual',severity:'note'}]);
  applyAdded();renderFindingsList(LAST_ACTIVE);saveReview(false);el('menu-findings').open=true;}
 // ---- pick router: measurement > annotation > new finding > region brush > probe ----
 function handlePick(p){
  if(measureMode){addMeasurePick(p);return 'measure';}
  if(pickTask==='annot'){pickTask='';addAnnotation(p);return 'annot';}
  if(pickTask==='finding'){pickTask='';addManualFinding(p);return 'finding';}
  if(el('region-pick').checked&&el('region-mode').value==='sphere'){regionCenter=p.slice();el('region-center').textContent='中心 ('+p.map(v=>fmt(v,1)).join(', ')+') mm';el('menu-region').open=true;applyRegion();return 'region';}
  lockProbe(p);return 'probe';}
 function rebuildTop(){highlightThreshold=CORE.topThreshold(PW,highlightPct);topIndices=[];const topPositions=[];for(let i=0;i<PW.length;i++)if(Number.isFinite(PW[i])&&PW[i]>=highlightThreshold&&(branchSel<0||PS[i]===branchSel)){topPositions.push(PV[i*3],PV[i*3+1],PV[i*3+2]);topIndices.push(i);}if(top){scene.remove(top);top.geometry.dispose();top.material.dispose();}const tg=new THREE.BufferGeometry();tg.setAttribute('position',new THREE.Float32BufferAttribute(topPositions,3));const tc=new Float32Array(topPositions.length);for(let j=0;j<topIndices.length;j++)tc.set([.95,.2,.75],3*j);tg.setAttribute('color',new THREE.BufferAttribute(tc,3));top=new THREE.Points(tg,new THREE.PointsMaterial({size:1.8,vertexColors:true,depthTest:false,clippingPlanes:[clipPlane],transparent:true,opacity:.95}));top.renderOrder=2;top.visible=(viewMode==='wss'||viewMode==='cloud')&&el('showtop').checked;scene.add(top);el('highlight-value').textContent=fmt(highlightPct,1)+'%';el('highlight-label').textContent=fmt(highlightPct,1)+'%';}
 function rebuildHlPts(){if(hlPts){scene.remove(hlPts);hlPts.geometry.dispose();hlPts.material.dispose();hlPts=null;}const idx=hl.pointIndices||[];if(!idx.length)return;const pos=new Float32Array(idx.length*3),col=new Float32Array(idx.length*3);idx.forEach((i,j)=>{pos.set([PV[3*i],PV[3*i+1],PV[3*i+2]],3*j);col.set([0,.76,.85],3*j);});const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));g.setAttribute('color',new THREE.BufferAttribute(col,3));hlPts=new THREE.Points(g,new THREE.PointsMaterial({size:1.6,vertexColors:true,depthTest:false,clippingPlanes:[clipPlane],transparent:true,opacity:.9}));hlPts.renderOrder=2;scene.add(hlPts);}
 function setHighlight(h){hl={vertexMask:h&&h.vertexMask||null,pointIndices:h&&h.pointIndices||null,label:h&&h.label||''};if(h&&h.markerAt){marker.position.set(...h.markerAt);marker.scale.setScalar(Math.max(1,(h.markerRadius||2)));marker.visible=true;markerLabel.textContent=hl.label;}else{marker.visible=false;markerLabel.textContent='';}rebuildHlPts();recolor();}
 function clearHighlight(){activeFinding=-1;profSel=null;regionSel=null;renderFindingsList(-1);el('region-stats').innerHTML='';setHighlight(null);drawProfiles();}
 try{const saved=JSON.parse(localStorage.getItem('wss-report-scale-v1'));if(saved&&saved.mode==='fixed'&&Number.isFinite(saved.max)&&saved.max>0){fixedMax=saved.max;scaleMode='fixed';}if(saved&&Number.isFinite(saved.highlightPct))highlightPct=Math.max(.5,Math.min(10,saved.highlightPct));}catch(_){}
 el('scale-mode').value=scaleMode;el('fixedmax').value=fixedMax;el('highlight-pct').value=highlightPct;el('cmap').value=cmapName;el('cmap').onchange=e=>{cmapName=CMAPS[e.target.value]?e.target.value:'rainbow';try{localStorage.setItem('wss-report-cmap',cmapName);}catch(_){}recolor();};rebuildTop();
 el('bands').onchange=e=>{bands=Number(e.target.value)||0;recolor();};el('logscale').onchange=e=>{logScale=e.target.checked;recolor();};
 el('units').onchange=e=>{UNIT=CORE.unitInfo(e.target.value)===CORE.UNITS.dyn?'dyn':'Pa';renderPanel();initThresholdInputs();renderFindingsList(activeFinding);el('fixedmax').value=String(+(fixedMax*uf()).toFixed(3));recolor();drawProfiles();renderRegionStats();redrawOverlays();};
 function feature(){const L=(k,u)=>COMMON.englishLabel(k,LANG)+' · '+u;if(feat==='s')return {values:P_S,mode:'vir',lo:0,hi:bounds(P_S)[1],label:L('arc','mm')};if(feat==='th')return {values:P_TH,mode:'vir',lo:-Math.PI,hi:Math.PI,label:L('theta','rad')};if(feat==='r')return {values:P_R,mode:'vir',lo:bounds(P_R)[0],hi:bounds(P_R)[1],label:L('radius','mm')};if(feat==='dj')return {values:P_DJ,mode:'vir',lo:0,hi:bounds(P_DJ)[1],label:L('dist_junction','mm')};const AF=fieldArrays(activeField);return {values:AF.p,mode:'wss',lo:0,hi:vmax,label:activeField==='wss'?legendTitle(LANG):AF.label+' · '+al()};}
 const TRUST_TINT={2:[.78,.72,.62],4:[.72,.66,.8]};
 function colorArray(values,segment,array,mode,lo,hi,isMesh){const isWssField=values===MW||values===PW;for(let i=0;i<values.length;i++){let c=!Number.isFinite(values[i])?[.48,.51,.55]:(mode==='grey'?[.72,.75,.8]:(mode==='wss'?cmap(CORE.normalize(values[i],lo,hi,logScale&&fieldArrays(activeField).log!==false)):vir((values[i]-lo)/(hi-lo||1))));
   if(isMesh&&showTrust&&MT){const b=MT[i];if(b&1)c=[.48,.51,.55];else if(b&4)c=c.map((v,k)=>v*.25+.75*TRUST_TINT[4][k]);else if(b&2)c=c.map((v,k)=>v*.25+.75*TRUST_TINT[2][k]);}
   if(highlightEnabled&&isWssField&&mode==='wss'&&Number.isFinite(values[i])&&values[i]<highlightThreshold)c=c.map(v=>v*.28+.72*.86);
   if(branchSel>=0&&segment[i]!==branchSel)c=c.map(v=>v*.25+.75*.86);
   if(isMesh&&hl.vertexMask&&!hl.vertexMask[i])c=c.map(v=>v*.3+.7*.88);
   array.set(c,3*i);}}
 function drawBar(mode,lo,hi,label){el('cbar').style.display=viewMode==='stl'?'none':'';el('cbar-unit').textContent=viewMode==='stl'?(LANG==='en'?'Input STL · mm':'输入 STL · mm'):label;let g='linear-gradient(to top';const n=mode==='wss'&&bands>0?bands:0;
  if(n){for(let i=0;i<n;i++){const c='rgb('+cmapRaw((i+.5)/n).map(x=>Math.round(x*255)).join(',')+')';g+=','+c+' '+(i/n*100)+'%,'+c+' '+((i+1)/n*100)+'%';}}else for(let i=0;i<=10;i++)g+=',rgb('+(mode==='wss'?cmapRaw(i/10):vir(i/10)).map(x=>Math.round(x*255)).join(',')+') '+i*10+'%';
  el('cbar').style.background=g+')';const isW=mode==='wss',LG=logScale&&fieldArrays(activeField).log!==false,f=isW?af():1,mid=isW&&LG?Math.sqrt(Math.max(lo,.05)*Math.max(hi,.05)):(lo+hi)/2;el('cbar').innerHTML='<span style="top:-5px">'+fmt(hi*f,1)+'</span><span style="top:86px">'+fmt(mid*f,1)+'</span><span style="bottom:-5px">'+fmt((isW&&LG?Math.max(lo,.05):lo)*f,1)+'</span>';}
 function rebuildContours(){if(contourObj){scene.remove(contourObj);contourObj.geometry.dispose();contourObj.material.dispose();contourObj=null;}if(!showContours)return;const parts=[];for(const level of thresholdsPa){const seg=CORE.marchingTriangles(MV,MF,MW,level);parts.push(seg);}let total=0;for(const p of parts)total+=p.length;const all=new Float32Array(total);let at=0;for(const p of parts){all.set(p,at);at+=p.length;}const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(all,3));contourObj=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0x203049,clippingPlanes:[clipPlane]}));contourObj.visible=viewMode==='wss';scene.add(contourObj);}
 function renderTrustLegend(){const box=el('trust-legend');if(!showTrust||!TRUST||!MT){box.hidden=true;return;}const fr=TRUST.fractions||{},src=Array.isArray(TRUST.sources)?TRUST.sources:[];const rows=[[1,'插值无支撑','rgb(122,130,140)','interpolation_uncovered'],[2,'表面粗糙','rgb(199,184,158)','rough_surface'],[4,'几何越界','rgb(184,168,204)','geometry_out_of_range']];box.innerHTML='<b>可信区域叠加</b>'+rows.map(([bit,label,color,key])=>{const s=src.find(x=>Number(x.bit)===bit);return `<span><i class="sw" style="background:${color}"></i>${esc((s&&s.label)||label)} ${Number.isFinite(+fr[key])?fmt(fr[key]*100,1)+'%':''}${s&&s.rule?'<br><small>'+esc(s.rule)+'</small>':''}</span>`;}).join('');box.hidden=false;}
 function recolor(){const AF=fieldArrays(activeField),LGS=logScale&&AF.log!==false;vmax=scaleMode==='fixed'?fixedMax:Math.max(Number(AF.p99)||0,.01);el('fixed-control').hidden=scaleMode!=='fixed';el('scale-description').textContent=(activeField==='wss'?'':AF.label+' · ')+(LGS?'对数 ':'')+fmt((LGS?.05:0)*af(),2)+' – '+fmt(vmax*af(),2)+' '+al()+' · '+(scaleMode==='fixed'?'固定色标':'本例空间 p99')+'；超限按上限着色'+(bands?' · '+bands+' 段':'');rebuildTop();colorArray(AF.m,MS,mcol,viewMode==='wss'?'wss':'grey',0,vmax,true);const f=feature();colorArray(f.values,PS,pcol,f.mode,f.lo,f.hi,false);updateColors();el('display-readout').textContent='当前视图：'+({wss:'WSS 壁面',stl:'输入 STL',cl:'中心线',cloud:'预测点云'}[viewMode]||viewMode)+' · '+(viewMode==='wss'?(activeField==='wss'?'统计：预测点云 · 壁面：Gaussian 插值':AF.label+'（壁面着色）· 统计面板仍为峰值 WSS'):f.label)+(branchSel>=0?' · 分支：'+branchName(branchSel):'')+(hl.label?' · '+hl.label:'');
  if(viewMode==='cloud')drawBar(f.mode,f.lo,f.hi,f.label);else if(viewMode==='cl')drawBar('vir',rmin,rmax,COMMON.englishLabel('centerline_radius',LANG)+' · mm');else drawBar('wss',0,vmax,activeField==='wss'?legendTitle(LANG):AF.label+' · '+al());peakLabel.textContent='全场最大值 '+fu(PEAK.max_pa)+' '+ul();renderTrustLegend();if(contourObj)contourObj.visible=viewMode==='wss'&&showContours;drawUnroll();}
 recolorActive=recolor;
 function updateColors(){mg.attributes.color.needsUpdate=true;pg.attributes.color.needsUpdate=true;}
 function visibility(){mesh.visible=viewMode!=='cloud';mmat.transparent=viewMode==='cl'||opacity<1;mmat.opacity=viewMode==='cl'?.18:opacity;pts.visible=viewMode==='cloud'||el('showpts').checked;cl.visible=viewMode==='cl'||el('showcl').checked;peak.visible=viewMode==='wss'&&el('showpeak').checked;top.visible=(viewMode==='wss'||viewMode==='cloud')&&el('showtop').checked;if(hlPts)hlPts.visible=viewMode==='wss'||viewMode==='cloud';if(contourObj)contourObj.visible=viewMode==='wss'&&showContours;for(const item of labels)if(item.kind==='endpoint')item.obj.visible=viewMode==='cl';}
 function setView(mode){viewMode=mode;document.querySelectorAll('[data-v]').forEach(b=>b.classList.toggle('on',b.dataset.v===mode));visibility();recolor();el('tip').textContent=mode==='cloud'?'点云鼠标读值为原始预测/几何特征。':'壁面鼠标读值为 Gaussian 插值；统计卡片使用预测点云。';}
 document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setView(b.dataset.v));
 function saveScale(){try{localStorage.setItem('wss-report-scale-v1',JSON.stringify({mode:scaleMode,max:fixedMax,highlightPct}));}catch(_){}recolor();}
 el('scale-mode').onchange=e=>{scaleMode=e.target.value;saveScale();};el('fixedmax').onchange=e=>{const v=CORE.fromUnit(Number(e.target.value),UNIT);if(!Number.isFinite(v)||v<=0){e.target.value=String(+(fixedMax*uf()).toFixed(3));return;}fixedMax=v;saveScale();};
 el('branch').onchange=e=>{branchSel=Number(e.target.value);recolor();};el('feat').onchange=e=>{feat=e.target.value;if(feat!=='wss'){el('showpts').checked=false;setView('cloud');}else recolor();};
 el('highlight-pct').oninput=e=>{highlightPct=Math.max(.5,Math.min(10,Number(e.target.value)||1));highlightEnabled=el('showtop').checked;saveScale();};
 for(const id of ['showpts','showcl','showpeak'])el(id).onchange=()=>{if(el('showpts').checked&&feat!=='wss')setView('cloud');else visibility();};
 el('showtop').onchange=()=>{highlightEnabled=el('showtop').checked;recolor();visibility();};
 el('contours').onchange=e=>{showContours=e.target.checked;rebuildContours();};
 el('trust').onchange=e=>{showTrust=e.target.checked&&!!MT;recolor();};
 function applyThresholds(){const t=CORE.validThresholds([0,1,2].map(k=>CORE.fromUnit(Number(el('thr'+k).value),UNIT)));if(!t){initThresholdInputs();el('tip').textContent='阈值必须是三个递增的非负数。';return;}thresholdsPa=t;renderPanel();if(showContours)rebuildContours();drawProfiles();}
 for(const k of [0,1,2])el('thr'+k).onchange=applyThresholds;
 el('clip').oninput=e=>{clipT=Number(e.target.value)/100;clipPlane.constant=clipT>=1?1e6:mn[2]+(mx[2]-mn[2])*clipT;el('tip').textContent='剖切已更新；鼠标只读取可见部分。';};
 el('opacity').oninput=e=>{opacity=Number(e.target.value)||1;visibility();};
 // ---- camera: reset, tween, standard views ----
 let tween=null;
 function resetCamera(){camera.position.set(center[0],center[1],center[2]+size*1.8);controls.target.set(...center);camera.up.set(0,1,0);camera.near=size*.001;camera.far=size*30;camera.updateProjectionMatrix();controls.update();}el('reset-camera').onclick=resetCamera;
 function setCamera(c,animate){if(!c||!c.position||!c.target)return;const up=c.up&&c.up.length===3?c.up:[0,1,0];if(!animate){camera.position.set(...c.position);controls.target.set(...c.target);camera.up.set(...up);controls.update();return;}tween={t0:performance.now(),ms:450,p0:camera.position.clone(),t0v:controls.target.clone(),p1:new THREE.Vector3(...c.position),t1:new THREE.Vector3(...c.target)};camera.up.set(...up);}
 function flyTo(target,ext){const dir=camera.position.clone().sub(controls.target).normalize();const dist=Math.max(ext*6,size*.12);setCamera({position:target.map((v,k)=>v+dir.getComponent(k)*dist),target,up:camera.up.toArray()},true);}
 const stdBox=el('std-views');
 if(!FRAME){stdBox.querySelectorAll('[data-view]').forEach(b=>{b.disabled=true;b.title='本报告未提供解剖坐标架';});el('std-note').textContent='本报告未提供解剖坐标架（rotation/origin），标准视角与相机联动不可用。';el('views-export').disabled=true;}
 else{el('std-note').textContent=(META.frame_transform.direction_source==='unknown_stl'?'按解剖坐标架推断，请核对左右。':'解剖坐标架来源：'+META.frame_transform.direction_source)+' 前/后/左/右/上/下按患者方向。';stdBox.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{const v=CORE.standardViews(FRAME.rotation,center,size*1.9)[b.dataset.view];setCamera(v,true);lastViewName=b.dataset.view;});}
 // ---- highlight helpers ----
 function markVertices(pointIdxSet){const n=near();const mask=new Uint8Array(MV.length/3);for(let v=0;v<mask.length;v++){const p=n[v];if(p>=0&&pointIdxSet[p])mask[v]=1;}return mask;}
 window.__selectFinding=i=>{if(activeFinding===i){clearHighlight();return;}const f=FINDINGS[i];if(!f)return;profSel=null;regionSel=null;activeFinding=i;renderFindingsList(i);const ext=Math.max(Number(f.extent_mm)||5,3);const xyz=Array.isArray(f.xyz_mm)&&f.xyz_mm.length===3?f.xyz_mm.map(Number):null;let mask=null,idx=null;
  if(Array.isArray(f.point_indices)&&f.point_indices.length){idx=f.point_indices.map(Number).filter(k=>Number.isInteger(k)&&k>=0&&k<PW.length);const set=new Uint8Array(PW.length);for(const k of idx)set[k]=1;mask=markVertices(set);}
  else if(xyz){mask=CORE.sphereMask(MV,xyz,ext*1.5);}
  const value=f.units==='Pa'?fu(f.value)+' '+ul():(f.value!=null?fmt(f.value,2)+' '+(f.units||''):'');
  setHighlight({vertexMask:mask,pointIndices:idx,markerAt:xyz,markerRadius:Math.min(ext*.35,6),label:(f.label||KIND_CN[f.kind]||f.kind)+(value?' '+value:'')});if(xyz)flyTo(xyz,ext);el('tip').textContent=(f.label||f.kind)+'\n'+(f.definition||'');};
 // ---- profiles ----
 let profSel=null;const PROF_COLORS=['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b','#17becf','#7f7f7f'];
 const profColor=(b,bi)=>HIDDEN.has(Number(b.segment_id))?'#c3ccd6':PROF_COLORS[bi%PROF_COLORS.length];
 let profHits=[];
 function profBranches(){if(!PROFILES)return [];const which=el('prof-branch').value;return PROFILES.branches.filter(b=>which===''||String(b.segment_id)===which);}
 function drawCurve(canvas,branches,xkey,series,yLabel,scaleY,marker){const c=canvas,W=c.width=Math.max(200,Math.round(c.clientWidth*2)),H=c.height,ctx=c.getContext('2d');if(!ctx)return null;ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);
  const L=62,R=14,T=14,B=36;let xmin=Infinity,xmax=-Infinity,ymax=0;
  for(const b of branches){const xs=b[xkey]||b.s_local_mm||[];for(let i=0;i<xs.length;i++){if(!(b.wss&&b.wss.n&&b.wss.n[i]>0)&&!series.some(s=>s.key==='radius_mm'))continue;xmin=Math.min(xmin,xs[i]);xmax=Math.max(xmax,xs[i]);for(const s of series){const v=s.get(b)[i];if(Number.isFinite(v))ymax=Math.max(ymax,v*scaleY);}}}
  if(!Number.isFinite(xmin)||xmax<=xmin){ctx.fillStyle='#5f7086';ctx.font='22px sans-serif';ctx.fillText('没有可绘制的分箱',L,H/2);return null;}
  ymax=ymax||1;const X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),Y=y=>T+(1-y/ymax)*(H-T-B);
  ctx.strokeStyle='#dce4ed';ctx.lineWidth=1;ctx.beginPath();for(let k=0;k<=4;k++){const y=Y(ymax*k/4);ctx.moveTo(L,y);ctx.lineTo(W-R,y);}ctx.stroke();ctx.fillStyle='#5f7086';ctx.font='20px sans-serif';ctx.textAlign='right';for(let k=0;k<=4;k++)ctx.fillText(fmt(ymax*k/4,ymax<5?2:1),L-6,Y(ymax*k/4)+7);ctx.textAlign='center';for(let k=0;k<=4;k++){const x=xmin+(xmax-xmin)*k/4;ctx.fillText(fmt(x,0),X(x),H-12);}ctx.textAlign='left';ctx.fillText(yLabel,L+4,T+18);
  const hits=[];
  branches.forEach((b,bi)=>{const xs=b[xkey]||b.s_local_mm||[],color=profColor(b,bi);
   for(const s of series){const ys=s.get(b);if(!ys)continue;if(s.band){const zs=s.band(b);ctx.fillStyle=color+'26';ctx.beginPath();let started=false;for(let i=0;i<xs.length;i++){if(!Number.isFinite(ys[i])||!Number.isFinite(zs[i])){if(started){for(let j=i-1;j>=0&&Number.isFinite(zs[j])&&Number.isFinite(ys[j]);j--)ctx.lineTo(X(xs[j]),Y(zs[j]*scaleY));ctx.closePath();ctx.fill();ctx.beginPath();started=false;}continue;}if(!started){ctx.moveTo(X(xs[i]),Y(ys[i]*scaleY));started=true;}else ctx.lineTo(X(xs[i]),Y(ys[i]*scaleY));}if(started){for(let j=xs.length-1;j>=0;j--){if(Number.isFinite(zs[j])&&Number.isFinite(ys[j]))ctx.lineTo(X(xs[j]),Y(zs[j]*scaleY));else break;}ctx.closePath();ctx.fill();}}
    ctx.strokeStyle=color;ctx.lineWidth=s.width||2.5;if(s.dash)ctx.setLineDash(s.dash);else ctx.setLineDash([]);ctx.beginPath();let pen=false;for(let i=0;i<xs.length;i++){const v=ys[i];if(!Number.isFinite(v)){pen=false;continue;}if(!pen){ctx.moveTo(X(xs[i]),Y(v*scaleY));pen=true;}else ctx.lineTo(X(xs[i]),Y(v*scaleY));}ctx.stroke();ctx.setLineDash([]);}
   for(let i=0;i<xs.length;i++)hits.push({b,i,x:X(xs[i])/2,y:Y((series[0].get(b)[i]||0)*scaleY)/2});});
  if(marker&&marker.b&&branches.includes(marker.b)){const xs=marker.b[xkey]||marker.b.s_local_mm;const x=X(xs[marker.i]);ctx.strokeStyle='#00c2d8';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,H-B);ctx.stroke();}
  return hits;}
 // §17.3 diameter series: the morphology stations (1 mm apart) are sampled onto the profile's own bins.
 const PROF_BIN=Math.max(Number((META.profiles||{}).bin_mm)||2,1);
 // Cached: drawCurve asks for a series once per plotted sample, and stations are 1 mm apart.
 const STATION_CACHE=new Map();
 function stationSeries(b,which){const key=which+':'+(b&&b.segment_id);
  if(STATION_CACHE.has(key))return STATION_CACHE.get(key);
  const out=computeStationSeries(b,which);STATION_CACHE.set(key,out);return out;}
 function computeStationSeries(b,which){const st=STATIONS.get(Number(b&&b.segment_id));if(!st||!st.s.length)return null;
  const vals=which==='max'?st.max:st.eq;if(!vals||!vals.length)return null;
  const xs=Array.from(b.s_from_root_mm||b.s_local_mm||[],Number),out=new Array(xs.length).fill(NaN);let any=false;
  for(let i=0;i<xs.length;i++){const x=xs[i];if(!Number.isFinite(x))continue;let bj=-1,bd=Infinity;
   for(let j=0;j<st.s.length;j++){const d=Math.abs(st.s[j]-x);if(d<bd){bd=d;bj=j;}}
   // A station farther than one bin away does not describe this bin.
   if(bj>=0&&bd<=PROF_BIN&&Number.isFinite(vals[bj])){out[i]=vals[bj];any=true;}}
  return any?out:null;}
 const hasStations=b=>!!stationSeries(b,'max')||!!stationSeries(b,'eq');
 function radiusSeries(){const out=[{key:'radius_mm',get:b=>b.radius_mm||[]}];
  if(!STATIONS.size)return out;
  out.push({key:'max_diameter_mm',get:b=>stationSeries(b,'max')||[],dash:[6,5],width:2});
  out.push({key:'equivalent_diameter_mm',get:b=>stationSeries(b,'eq')||[],dash:[2,4],width:2});
  return out;}
 function drawProfiles(){if(!PROFILES||!el('menu-profiles').open)return;const branches=profBranches(),xkey=el('prof-x').value;
  const diam=STATIONS.size&&branches.some(hasStations);
  el('prof-legend').innerHTML=branches.map((b,bi)=>`<span><i style="background:${profColor(b,bi)}"></i>${esc(b.name||branchName(b.segment_id))}${HIDDEN.has(Number(b.segment_id))?' · 已隐藏':''}</span>`).join('')+'<span>实线：p99 · 虚线：均值 · 阴影：均值–p99</span>'+(diam?'<span>下图 实线：半径 · 长虚线：最大直径 · 点线：等效直径</span>':'');
  const g=b=>(b.wss||{});profHits=drawCurve(el('prof-wss'),branches,xkey,[{key:'p99',get:b=>g(b).p99_pa||[],band:b=>g(b).mean_pa||[]},{key:'mean',get:b=>g(b).mean_pa||[],dash:[8,6],width:2}],'WSS · '+ul(),uf(),profSel)||[];
  drawCurve(el('prof-r'),branches,xkey,radiusSeries(),diam?'半径 / 直径 · mm':'半径 · mm',1,profSel);}
 window.__drawProfiles=drawProfiles;
 if(PROFILES){el('prof-branch').onchange=drawProfiles;el('prof-x').onchange=drawProfiles;el('prof-clear').onclick=clearHighlight;
  el('prof-wss').addEventListener('click',ev=>{if(!profHits.length)return;const r=ev.target.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;let best=null,bd=Infinity;for(const h of profHits){const d=Math.abs(h.x-x)*4+Math.abs(h.y-y);if(d<bd){bd=d;best=h;}}if(!best)return;selectProfile(best.b,best.i);});}
 function selectProfile(b,i){const s=(b.s_from_root_mm||[])[i];if(!Number.isFinite(s))return;activeFinding=-1;regionSel=null;renderFindingsList(-1);profSel={b,i};near();const sid=Number(b.segment_id),mask=new Uint8Array(MV.length/3);for(let v=0;v<mask.length;v++)if(MS[v]===sid&&Math.abs(VS[v]-s)<=2)mask[v]=1;const pm=CORE.rangeMask(P_S,PS,sid,s-2,s+2),idx=CORE.maskIndices(pm);const w=b.wss||{};const label=(b.name||branchName(sid))+' s='+fmt(s,0)+' mm · 均值 '+fu((w.mean_pa||[])[i])+' · p99 '+fu((w.p99_pa||[])[i])+' '+ul()+' · r '+fmt((b.radius_mm||[])[i],1)+' mm';
  el('prof-readout').textContent=label;let cx=[0,0,0];if(idx.length){for(const k of idx){cx[0]+=PV[3*k];cx[1]+=PV[3*k+1];cx[2]+=PV[3*k+2];}cx=cx.map(v=>v/idx.length);}setHighlight({vertexMask:mask,pointIndices:idx,markerAt:idx.length?cx:null,markerRadius:1.2,label});drawProfiles();}
 // ---- region statistics ----
 let regionSel=null,regionCenter=null;
 function regionBranch(){const sid=Number(el('region-branch').value);return (ARR.unroll_branches||[]).find(b=>b.segment_id===sid)||null;}
 function syncRegionSliders(){const b=regionBranch();if(!b)return;for(const id of ['region-s0','region-s1']){el(id).min=String(b.s_min_mm);el(id).max=String(b.s_max_mm);}if(Number(el('region-s0').value)<b.s_min_mm||Number(el('region-s0').value)>b.s_max_mm)el('region-s0').value=String(b.s_min_mm);if(Number(el('region-s1').value)>b.s_max_mm||Number(el('region-s1').value)<b.s_min_mm)el('region-s1').value=String(b.s_max_mm);el('region-s0-v').textContent=fmt(el('region-s0').value,0)+' mm';el('region-s1-v').textContent=fmt(el('region-s1').value,0)+' mm';}
 syncRegionSliders();el('region-branch').onchange=syncRegionSliders;for(const id of ['region-s0','region-s1'])el(id).oninput=()=>{el(id+'-v').textContent=fmt(el(id).value,0)+' mm';};
 el('region-mode').onchange=e=>{el('region-range').hidden=e.target.value!=='range';el('region-sphere').hidden=e.target.value!=='sphere';};
 el('region-radius').oninput=e=>{el('region-radius-v').textContent=fmt(e.target.value,1)+' mm';if(regionSel&&regionSel.kind==='sphere')applyRegion();};
 function renderRegionStats(){const box=el('region-stats');if(!regionSel){box.innerHTML='';return;}const st=CORE.stats(PW,regionSel.indices),area=PW.length?regionSel.indices.length/PW.length*AREA:0;box.innerHTML=`<b>区域</b><span>${esc(regionSel.label)}</span><b>预测点</b><span>${st.n}</span><b>均值</b><span>${fu(st.mean)} ${esc(ul())}</span><b>p99</b><span>${fu(st.p99)} ${esc(ul())}</span><b>最大</b><span>${fu(st.max)} ${esc(ul())}</span><b>估计面积</b><span>${fmt(area/100,1)} cm²</span>`;}
 function applyRegion(){const mode=el('region-mode').value;let indices,mask,label,markerAt=null,markerRadius=1;
  if(mode==='range'){const b=regionBranch();if(!b){el('tip').textContent='没有可用分支。';return;}const s0=Number(el('region-s0').value),s1=Number(el('region-s1').value);const pm=CORE.rangeMask(P_S,PS,b.segment_id,s0,s1);indices=CORE.maskIndices(pm);near();mask=new Uint8Array(MV.length/3);const lo=Math.min(s0,s1),hi=Math.max(s0,s1);for(let v=0;v<mask.length;v++)if(MS[v]===b.segment_id&&VS[v]>=lo&&VS[v]<=hi)mask[v]=1;label=branchName(b.segment_id)+' s '+fmt(lo,0)+'–'+fmt(hi,0)+' mm';regionSel={kind:'range',indices,label,segment_id:b.segment_id,s0:lo,s1:hi};}
  else{if(!regionCenter){el('tip').textContent='请先勾选「点击壁面选中心」并在壁面上点击。';return;}const r=Number(el('region-radius').value)||8;const pm=CORE.sphereMask(PV,regionCenter,r);indices=CORE.maskIndices(pm);mask=CORE.sphereMask(MV,regionCenter,r);label='球形区域 r='+fmt(r,1)+' mm';markerAt=regionCenter;markerRadius=r;regionSel={kind:'sphere',indices,label,center:regionCenter.slice(),radius:r};}
  activeFinding=-1;profSel=null;renderFindingsList(-1);renderRegionStats();setHighlight({vertexMask:mask,pointIndices:null,markerAt,markerRadius,label});}
 el('region-apply').onclick=applyRegion;el('region-clear').onclick=()=>{regionCenter=null;el('region-center').textContent='尚未选点';clearHighlight();};
 // ---- pointer: hover readout + click picking ----
 const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();ray.params.Points.threshold=1;let pressAt=null;
 function pick(ev){const rect=renderer.domElement.getBoundingClientRect();mouse.set((ev.clientX-rect.left)/rect.width*2-1,-(ev.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(mouse,camera);return ray.intersectObject(mesh,false).find(h=>clipPlane.distanceToPoint(h.point)>=0)||null;}
 renderer.domElement.addEventListener('pointerdown',ev=>{pressAt=[ev.clientX,ev.clientY];lastViewName='custom';});
 renderer.domElement.addEventListener('pointerup',ev=>{if(!pressAt||Math.hypot(ev.clientX-pressAt[0],ev.clientY-pressAt[1])>5){pressAt=null;return;}pressAt=null;if(viewMode==='cloud')return;const hit=pick(ev);if(!hit)return;handlePick([hit.point.x,hit.point.y,hit.point.z]);});
 renderer.domElement.addEventListener('mousemove',ev=>{const rect=renderer.domElement.getBoundingClientRect();mouse.set((ev.clientX-rect.left)/rect.width*2-1,-(ev.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(mouse,camera);
  if(viewMode==='cloud'){const hit=ray.intersectObject(pts,false).find(h=>clipPlane.distanceToPoint(h.point)>=0);if(!hit)return;const i=hit.index,f=feature();el('tip').textContent='预测点云 · '+(activeField==='wss'?'WSS '+fu(PW[i])+' '+ul():fieldArrays(activeField).label+' '+fv(fieldArrays(activeField).p[i]))+'\n'+branchName(PS[i])+' · '+f.label+' '+(f.mode==='wss'?fu(f.values[i]):fmt(f.values[i]))+'\n弧长 '+fmt(P_S[i],1)+' mm · 半径 '+fmt(P_R[i],1)+' mm · 位置 ('+[0,1,2].map(k=>fmt(PV[3*i+k],1)).join(', ')+') mm';return;}
  if(viewMode!=='wss')return;const hit=ray.intersectObject(mesh,false).find(h=>clipPlane.distanceToPoint(h.point)>=0);if(!hit){el('tip').textContent='移动鼠标到可见壁面读取 Gaussian 插值。';return;}const f=hit.face;let best=f.a,bd=Infinity;for(const vi of [f.a,f.b,f.c]){const d=hit.point.distanceToSquared(new THREE.Vector3(MV[3*vi],MV[3*vi+1],MV[3*vi+2]));if(d<bd){bd=d;best=vi;}}
  // A partially unsupported triangle is marked invalid throughout to avoid a fictitious fill.
  const valid=[f.a,f.b,f.c].every(i=>Number.isFinite(MW[i]));let extra='';if(MT&&MT[best]){const bits=[];if(MT[best]&1)bits.push('插值无支撑');if(MT[best]&2)bits.push('表面粗糙');if(MT[best]&4)bits.push('几何越界');if(bits.length)extra='\n可信提示：'+bits.join('、');}
  el('tip').textContent=(valid?'壁面 Gaussian 插值 '+(activeField==='wss'?'':fieldArrays(activeField).label+' ')+fv(fieldArrays(activeField).m[best]):'插值覆盖范围外：无有效 WSS 读值')+'\n分支 '+branchName(MS[best])+(VS&&Number.isFinite(VS[best])?' · 弧长 '+fmt(VS[best],1)+' mm':'')+'\n最近面顶点 ('+[0,1,2].map(k=>fmt(MV[3*best+k],1)).join(', ')+') mm'+extra;});
 // ---- unroll plots ----
 const branchIndices=new Map();for(let i=0;i<PS.length;i++){if(!branchIndices.has(PS[i]))branchIndices.set(PS[i],[]);branchIndices.get(PS[i]).push(i);}
 const plotItems=[];for(const b of ARR.unroll_branches){const row=document.createElement('div');row.className='unroll-row';const label=document.createElement('small');label.textContent=branchName(b.segment_id)+' · s '+fmt(b.s_min_mm,1)+' – '+fmt(b.s_max_mm,1)+' mm';const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',label.textContent+' 的独立展开图');row.append(label,canvas);el('unroll-plots').appendChild(row);plotItems.push({row,canvas,...b});}
 function drawUnroll(){if(!el('unroll-details').open||!el('menu-stats').open)return;for(const p of plotItems){p.row.hidden=branchSel>=0&&p.segment_id!==branchSel;if(p.row.hidden)continue;const c=p.canvas,W=c.width=Math.max(1,Math.round(c.clientWidth*2)),H=c.height=180,ctx=c.getContext('2d');if(!ctx)continue;const pixelMax=new Float32Array(W*H);pixelMax.fill(-Infinity);for(const i of branchIndices.get(p.segment_id)||[]){const x=Math.max(0,Math.min(W-1,Math.floor((P_S[i]-p.s_min_mm)/(p.s_max_mm-p.s_min_mm||1)*(W-1)))),y=Math.max(0,Math.min(H-1,Math.floor((1-(P_TH[i]+Math.PI)/(2*Math.PI))*(H-1))));pixelMax[y*W+x]=Math.max(pixelMax[y*W+x],PW[i]);}ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);const img=ctx.getImageData(0,0,W,H);for(let i=0;i<pixelMax.length;i++)if(Number.isFinite(pixelMax[i])){const c=cmap(CORE.normalize(pixelMax[i],0,vmax,logScale));for(let k=0;k<3;k++)img.data[4*i+k]=c[k]*255;img.data[4*i+3]=255;}ctx.putImageData(img,0,0);
   if(profSel&&Number(profSel.b.segment_id)===p.segment_id){const s=(profSel.b.s_from_root_mm||[])[profSel.i];if(Number.isFinite(s)){const x=(s-p.s_min_mm)/(p.s_max_mm-p.s_min_mm||1)*(W-1);ctx.strokeStyle='#00c2d8';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}}}}
 window.__drawUnroll=drawUnroll;el('unroll-details').addEventListener('toggle',drawUnroll);
 function resize(){const w=view.clientWidth,h=view.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();drawUnroll();drawProfiles();}window.addEventListener('resize',resize);
 // ---- view state ----
 const VIEW_KEY='wss-report-view:'+(META.run_identity||META.input_sha256||META.case_id||'unknown');
 // View state v1.1 records every key explicitly so a saved figure never depends on the current defaults.
 function currentViewState(){return {schema_version:'wss-deploy.view/v1',family:'wall',run_identity:META.run_identity||null,camera:{position:camera.position.toArray(),target:controls.target.toArray(),up:camera.up.toArray()},field:'wss',mode:viewMode,colormap:cmapName,range:{mode:scaleMode,min:0,max:vmax},bands,log:logScale,units:UNIT,thresholds_pa:thresholdsPa.slice(),opacity,overlay:{trust:showTrust,contours:showContours},slice:{clip:clipT},highlight:{branch:branchSel,top_pct:highlightPct,top:highlightEnabled,peak:el('showpeak').checked,feature:feat},
  labels:{findings:normLabelCount(LBL.findings),branches:!!LBL.branches,max_diameter:!!LBL.max_diameter},
  measurements:MEAS.map(m=>Object.assign({},m)),annotations:{items:ANNOTS.map(a=>Object.assign({},a))},probe_log:PROBES.map(r=>Object.assign({},r)),branches_hidden:Array.from(HIDDEN).sort((a,b)=>a-b),preset_name:PRESET_NAME,lang:LANG,export:exportOptions(),montage:montageState(),
  findings_review:ONLINE?undefined:{items:Object.assign({},REVIEW.items),added:(REVIEW.added||[]).slice()}};}
 function applyViewState(s){if(!s||typeof s!=='object')return false;if(s.colormap&&CMAPS[s.colormap]){cmapName=s.colormap;el('cmap').value=cmapName;}if(Number.isFinite(+s.bands)){bands=[0,4,6,8,12].includes(+s.bands)?+s.bands:0;el('bands').value=String(bands);}logScale=!!s.log;el('logscale').checked=logScale;if(s.units==='dyn'||s.units==='Pa'){UNIT=s.units;el('units').value=UNIT;}
  const t=CORE.validThresholds(s.thresholds_pa);if(t)thresholdsPa=t;initThresholdInputs();if(s.range&&typeof s.range==='object'){if(s.range.mode==='fixed'&&Number.isFinite(+s.range.max)&&+s.range.max>0){scaleMode='fixed';fixedMax=+s.range.max;}else if(s.range.mode==='case')scaleMode='case';el('scale-mode').value=scaleMode;el('fixedmax').value=String(+(fixedMax*uf()).toFixed(3));}
  if(Number.isFinite(+s.opacity)){opacity=Math.max(.2,Math.min(1,+s.opacity));el('opacity').value=String(opacity);}if(s.overlay&&typeof s.overlay==='object'){showTrust=!!s.overlay.trust&&!!MT;el('trust').checked=showTrust;showContours=!!s.overlay.contours;el('contours').checked=showContours;}
  if(s.slice&&Number.isFinite(+s.slice.clip)){clipT=Math.max(0,Math.min(1,+s.slice.clip));el('clip').value=String(clipT*100);clipPlane.constant=clipT>=1?1e6:mn[2]+(mx[2]-mn[2])*clipT;}
  const h=s.highlight||{};if(Number.isFinite(+h.branch)){branchSel=+h.branch;el('branch').value=String(branchSel);if(el('branch').value!==String(branchSel)){branchSel=-1;el('branch').value='-1';}}if(Number.isFinite(+h.top_pct)){highlightPct=Math.max(.5,Math.min(10,+h.top_pct));el('highlight-pct').value=String(highlightPct);}highlightEnabled=!!h.top;el('showtop').checked=highlightEnabled;if(typeof h.peak==='boolean')el('showpeak').checked=h.peak;if(h.feature&&['wss','s','th','r','dj'].includes(h.feature)){feat=h.feature;el('feat').value=feat;}
  // v1.1 keys; every one of them is optional so v1.0 states keep applying unchanged.
  if(Array.isArray(s.branches_hidden)){HIDDEN.clear();for(const sid of s.branches_hidden)if(Number.isFinite(+sid))HIDDEN.add(+sid);initBranchVis();rebuildBranchFilter();}
  if(Array.isArray(s.measurements))MEAS=s.measurements.filter(m=>m&&Array.isArray(m.points)&&m.points.length).map(m=>Object.assign({},m));
  if(s.annotations&&Array.isArray(s.annotations.items))ANNOTS=s.annotations.items.filter(a=>a&&Array.isArray(a.xyz_mm)).map(a=>Object.assign({},a));
  if(Array.isArray(s.probe_log))PROBES=s.probe_log.filter(r=>r&&r.id).map(r=>Object.assign({},r));
  // §17.3 automatic labels; a state without the key keeps the current setting (v1.0 / v1.1 states apply unchanged).
  if(s.labels&&typeof s.labels==='object'&&!Array.isArray(s.labels)){const L=s.labels;
   if(L.findings!==undefined)LBL.findings=normLabelCount(L.findings);
   if(typeof L.branches==='boolean')LBL.branches=L.branches;
   if(typeof L.max_diameter==='boolean')LBL.max_diameter=L.max_diameter;
   el('lbl-findings').value=labelSelectValue(LBL.findings);el('lbl-branches').checked=!!LBL.branches;if(AMAX)el('maxd-ring').checked=!!LBL.max_diameter;}
  if(s.findings_review&&typeof s.findings_review==='object'&&!ONLINE){REVIEW={items:Object.assign({},s.findings_review.items),added:Array.isArray(s.findings_review.added)?s.findings_review.added.slice():[]};applyAdded();}
  if(typeof s.preset_name==='string'||s.preset_name===null)PRESET_NAME=s.preset_name;
  if(s.lang==='en'||s.lang==='zh'){LANG=s.lang;el('exp-lang').value=LANG;GLOSS.setLang(LANG);}
  if(s.montage&&typeof s.montage==='object'){const mv=Array.isArray(s.montage.views)?s.montage.views.map(String):null;
   if(mv)for(const name of MONTAGE_VIEWS){const cb=montageBoxes[name];if(cb&&!cb.disabled)cb.checked=mv.includes(name);}
   if([2,3,4].includes(+s.montage.columns))el('montage-cols').value=String(+s.montage.columns);}
  if(s.export&&typeof s.export==='object'){const x=s.export;if([1,2,4].includes(+x.scale))el('exp-scale').value=String(+x.scale);if(['white','transparent','current'].includes(x.background))el('exp-bg').value=x.background;if(['overlay','none','svg'].includes(x.colorbar))el('exp-cbar').value=x.colorbar;if(typeof x.ui==='boolean')el('exp-ui').checked=x.ui;}
  renderMeasList();renderProbeLog();renderAnnots();redrawOverlays();
  renderPanel();renderFindingsList(activeFinding);rebuildContours();if(['wss','stl','cl','cloud'].includes(s.mode))setView(s.mode);else setView(viewMode);if(s.camera&&Array.isArray(s.camera.position)&&s.camera.position.length===3&&Array.isArray(s.camera.target)&&s.camera.target.length===3&&s.camera.position.every(Number.isFinite)&&s.camera.target.every(Number.isFinite))setCamera(s.camera,false);
  if(s.ui&&typeof s.ui==='object')applyUiState(s.ui);
  if(h.finding){const k=FINDINGS.findIndex(f=>f&&f.id===h.finding);if(k>=0&&k!==activeFinding)window.__selectFinding(k);}
  return true;}
 function applyUiState(u){if(u.menu&&MENUS.includes('menu-'+u.menu))for(const id of MENUS)el(id).open=(id==='menu-'+u.menu);
  const pf=u.profile;if(!pf||!PROFILES)return;
  if(pf.branch!==undefined&&pf.branch!==null)el('prof-branch').value=String(pf.branch);
  if(pf.x)el('prof-x').value=pf.x;
  drawProfiles();
  if(!Number.isFinite(+pf.s))return;
  const b=PROFILES.branches.find(x=>Number(x.segment_id)===Number(pf.branch))||PROFILES.branches[0];if(!b)return;
  let bi=-1,bd=Infinity;(b.s_from_root_mm||[]).forEach((v,i)=>{const d=Math.abs(v-+pf.s);if(Number.isFinite(d)&&d<bd){bd=d;bi=i;}});
  if(bi>=0)selectProfile(b,bi);}
 el('view-save').onclick=()=>{try{localStorage.setItem(VIEW_KEY,JSON.stringify(currentViewState()));el('view-note').textContent='视图已保存到本浏览器（按运行身份区分）。';}catch(_){el('view-note').textContent='浏览器不允许保存。';}};
 el('view-restore').onclick=()=>{try{const s=JSON.parse(localStorage.getItem(VIEW_KEY)||'null');if(!s){el('view-note').textContent='本浏览器没有该结果的已保存视图。';return;}applyViewState(s);el('view-note').textContent='已恢复保存的视图。';}catch(_){el('view-note').textContent='已保存的视图无法读取。';}};
 el('view-export').onclick=()=>{const a=document.createElement('a');a.href='data:application/json;charset=utf-8,'+encodeURIComponent(JSON.stringify(currentViewState(),null,1));a.download='WSS-view-'+String(META.case_id).replace(/[^\w\u3400-\u9fff-]/g,'_')+'.json';a.click();};
 el('view-import').onclick=()=>el('view-file').click();el('view-file').onchange=e=>{const f=e.target.files&&e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{try{const s=JSON.parse(String(r.result));if(s.run_identity&&META.run_identity&&s.run_identity!==META.run_identity)el('view-note').textContent='注意：视图来自另一次运行（run_identity 不同），已按可用项应用。';else el('view-note').textContent='已导入视图。';applyViewState(s);}catch(err){el('view-note').textContent='视图文件无法解析。';}};r.readAsText(f);e.target.value='';};
 el('view-link').onclick=()=>{const hash=CORE.viewHash(currentViewState());try{history.replaceState(null,'',hash);}catch(_){location.hash=hash;}const url=location.href.split('#')[0]+hash;const done=()=>{el('view-note').textContent='已把视图写入链接并复制；打开该链接即可复现此图。';};if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(url).then(done,()=>{el('view-note').textContent='视图已写入地址栏链接（复制失败，请手动复制地址）。';});else el('view-note').textContent='视图已写入地址栏链接，请手动复制地址。';};
 function capture(){renderer.render(scene,camera);const source=renderer.domElement,canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');canvas.width=1200;canvas.height=700;ctx.fillStyle='#eef2f7';ctx.fillRect(0,0,1200,700);const scale=Math.min(980/source.width,610/source.height),w=source.width*scale,h=source.height*scale;ctx.drawImage(source,(980-w)/2,45+(610-h)/2,w,h);ctx.fillStyle='#203049';ctx.font='22px sans-serif';ctx.fillText('WSS · '+String(META.case_id).slice(0,70),20,28);const f=viewMode==='cloud'?feature():viewMode==='cl'?{mode:'vir',lo:rmin,hi:rmax,label:'中心线半径 · mm'}:{mode:'wss',lo:0,hi:vmax,label:'WSS · '+ul()};if(viewMode!=='stl'){for(let y=0;y<300;y++){ctx.fillStyle='rgb('+(f.mode==='wss'?cmap(1-y/299):vir(1-y/299)).map(x=>Math.round(x*255)).join(',')+')';ctx.fillRect(1010,130+y,25,1);}ctx.fillStyle='#203049';ctx.font='18px sans-serif';ctx.fillText(f.label,990,100);const k=f.mode==='wss'?uf():1;ctx.fillText(fmt(f.hi*k,2),1045,140);ctx.fillText(fmt((f.mode==='wss'&&logScale?.05:f.lo)*k,2),1045,435);}ctx.fillStyle='#203049';ctx.font='17px sans-serif';ctx.fillText('固定收缩期帧 · p99 '+fu(PEAK.p99_pa)+' '+ul()+' · 最大值 '+fu(PEAK.max_pa)+' '+ul(),20,675);ctx.font='14px sans-serif';ctx.fillText('统计：预测点云；壁面：Gaussian 插值；黄色：全场最大值；粉色：高亮最高 '+fmt(highlightPct,1)+'%'+(hl.label?'；青色：'+hl.label:''),20,697);return canvas.toDataURL('image/png');}
 const fileStem=()=>'WSS-'+String(META.case_id).replace(/[^\w\u3400-\u9fff-]/g,'_');
 el('save-image').onclick=()=>{const a=document.createElement('a');a.href=capture();a.download=fileStem()+'.png';a.click();};
 // ---- C12 publication-grade export (the six-view button uses the very same path) ----
 let lastViewName='custom';
 function exportOptions(){return {scale:Number(el('exp-scale').value)||1,background:el('exp-bg').value||'white',colorbar:el('exp-cbar').value||'overlay',ui:!!el('exp-ui').checked,lang:LANG};}
 const legendTitle=lang=>COMMON.englishLabel('wss_short',lang)+' · '+ul();
 const colorbarOptions=lang=>({stops:COMMON.colormapStops(cmapName),min:(logScale?.05:0)*uf(),max:vmax*uf(),bands:bands||0,log:logScale,units:ul(),title:legendTitle(lang),lang});
 function loadImage(url){return new Promise((res,rej)=>{const img=new Image();img.onload=()=>res(img);img.onerror=()=>rej(new Error('PNG 解码失败'));img.src=url;});}
 const projectTo=(p,W,H)=>{const v=new THREE.Vector3(p[0],p[1],p[2]).project(camera);return [(v.x+1)/2*W,(1-v.y)/2*H,v.z];};
 function drawColorbar(ctx,W,H,k,lang){const w=Math.round(18*k),h=Math.round(H*.45),x=W-w-Math.round(64*k),y=Math.round(H*.12);
  for(let i=0;i<h;i++){const c=cmapRaw(CORE.quantize(1-i/Math.max(1,h-1),bands)).map(v=>Math.round(v*255));ctx.fillStyle='rgb('+c.join(',')+')';ctx.fillRect(x,y+i,w,1);}
  ctx.strokeStyle='#33414f';ctx.lineWidth=Math.max(1,k);ctx.strokeRect(x,y,w,h);ctx.fillStyle='#203049';ctx.font=Math.round(13*k)+'px sans-serif';ctx.textAlign='left';
  ctx.fillText(legendTitle(lang),x-Math.round(4*k),y-Math.round(8*k));ctx.fillText(fmt(vmax*uf(),2),x+w+Math.round(5*k),y+Math.round(11*k));ctx.fillText(fmt((logScale?.05:0)*uf(),2),x+w+Math.round(5*k),y+h);}
 async function composeExport(dataURL,W,H,o,lang){
  // WebGL cannot capture the HTML overlays, so measurement / annotation labels are redrawn on a 2-D canvas over the PNG.
  if((!o.ui&&o.colorbar!=='overlay')||typeof Image!=='function')return dataURL;
  const canvas=document.createElement('canvas');canvas.width=W;canvas.height=H;const ctx=canvas.getContext('2d');if(!ctx)return dataURL;
  if(o.background==='white'){ctx.fillStyle='#ffffff';ctx.fillRect(0,0,W,H);}
  ctx.drawImage(await loadImage(dataURL),0,0,W,H);
  const k=Math.max(1,W/Math.max(1,view.clientWidth||W));
  if(o.ui){ctx.font=Math.round(12*k)+'px sans-serif';ctx.textAlign='center';
   const chip=(p,text,fill,stroke)=>{const xyz=projectTo(p,W,H),x=xyz[0],y=xyz[1];if(!(xyz[2]>=-1&&xyz[2]<=1))return;const pad=Math.round(5*k),w=ctx.measureText(text).width+2*pad,hh=Math.round(17*k);
    ctx.fillStyle=fill;ctx.strokeStyle=stroke;ctx.lineWidth=Math.max(1,k);ctx.beginPath();ctx.rect(x-w/2,y-hh-Math.round(6*k),w,hh);ctx.fill();ctx.stroke();ctx.fillStyle='#203049';ctx.fillText(text,x,y-Math.round(10*k));};
   for(const m of MEAS){const path=(Array.isArray(m.path_xyz)&&m.path_xyz.length>1)?m.path_xyz:m.points;
    ctx.strokeStyle='#0f8ea0';ctx.lineWidth=Math.max(1.5,1.5*k);ctx.beginPath();path.forEach((q,i)=>{const at=projectTo(q,W,H);i?ctx.lineTo(at[0],at[1]):ctx.moveTo(at[0],at[1]);});ctx.stroke();
    chip(path[Math.floor(path.length/2)]||m.points[0],COMMON.measurementLabel(m,lang),'#e8f6f8f0','#5bbcc9');}
   for(const a of ANNOTS)chip((a.xyz_mm||[0,0,0]).map(Number),a.id+' · '+a.text,'#fff7e6f0','#e0a84a');
   // §17.3: the automatic labels (and the maximum-diameter ring) travel with the figure just like annotations.
   for(const c of AUTO_CHIPS){
    if(Array.isArray(c.ring)&&c.ring.length>2){ctx.strokeStyle='#8e44ad';ctx.lineWidth=Math.max(1.5,1.5*k);ctx.beginPath();
     c.ring.forEach((q,i)=>{const at=projectTo(q,W,H);i?ctx.lineTo(at[0],at[1]):ctx.moveTo(at[0],at[1]);});ctx.closePath();ctx.stroke();}
    chip(c.p,c.text,c.fill,c.stroke);}
   ctx.textAlign='left';}
  if(o.colorbar==='overlay')drawColorbar(ctx,W,H,k,lang);
  return canvas.toDataURL('image/png');}
 async function doExport(options){
  const o=Object.assign(exportOptions(),options||{}),lang=o.lang==='en'?'en':'zh';
  const r=COMMON.renderOffscreen({renderer,scene,camera,THREE,width:view.clientWidth||960,height:view.clientHeight||640,scale:o.scale,background:o.background});
  const url=await composeExport(r.dataURL,r.width,r.height,o,lang);
  const out={png_base64:String(url).replace(/^data:image\/png;base64,/,''),filename:o.filename||COMMON.exportFilename({case_id:META.case_id,view:lastViewName,field:'wss',scale:r.scale,ext:'png'}),width:r.width,height:r.height,downgraded:!!r.downgraded};
  if(o.colorbar==='svg')out.colorbar_svg=COMMON.colorbarSVG(colorbarOptions(lang));
  el('exp-note').textContent=(r.downgraded?'显卡不支持该倍率，已降级到 1×；':'')+'已导出 '+r.width+'×'+r.height+' PNG'+(o.colorbar==='svg'?' 与独立色标 SVG':'')+'。';
  return out;}
 const pngHref=r=>'data:image/png;base64,'+r.png_base64;
 el('exp-png').onclick=async()=>{el('exp-note').textContent='正在离屏渲染…';
  try{const r=await doExport();download(pngHref(r),r.filename);
   if(r.colorbar_svg)download('data:image/svg+xml;charset=utf-8,'+encodeURIComponent(r.colorbar_svg),r.filename.replace(/\.png$/,'-colorbar.svg'));}
  catch(err){el('exp-note').textContent='导出失败：'+(err&&err.message||err);}};
 el('exp-svg').onclick=()=>{const lang=el('exp-lang').value==='en'?'en':'zh';
  download('data:image/svg+xml;charset=utf-8,'+encodeURIComponent(COMMON.colorbarSVG(colorbarOptions(lang))),COMMON.exportFilename({case_id:META.case_id,view:'colorbar',field:'wss',scale:1,ext:'svg'}));
  el('exp-note').textContent='已导出当前色标 SVG（配色、范围、分段、对数与单位一致）。';};
 el('exp-lang').onchange=()=>{LANG=el('exp-lang').value==='en'?'en':'zh';GLOSS.setLang(LANG);renderMeasList();redrawOverlays();recolor();};
 el('views-export').onclick=async()=>{if(!FRAME)return;const views=CORE.standardViews(FRAME.rotation,center,size*1.9),saved=currentViewState(),savedName=lastViewName;
  el('view-note').textContent='正在导出六视角…';
  for(const entry of Object.entries(views)){setCamera(entry[1],false);tween=null;lastViewName=entry[0];
   try{const r=await doExport();download(pngHref(r),r.filename);}catch(err){el('view-note').textContent='六视角导出失败：'+(err&&err.message||err);break;}
   await new Promise(r=>setTimeout(r,220));}
  lastViewName=savedName;setCamera(saved.camera,false);el('view-note').textContent='已按当前导图设置导出前/后/左/右/上/下六张 PNG。';};
 // ---- §15.11 multi-view montage and §15.8 one-page figures: both reuse the C12 offscreen path ----
 const VIEW_CN={front:'前视',back:'后视',left:'左视',right:'右视',top:'上视',bottom:'下视',current:'当前视角'};
 const VIEW_EN={front:'Front',back:'Back',left:'Left',right:'Right',top:'Top',bottom:'Bottom',current:'Current'};
 const viewCaption=(name,lang)=>(lang==='en'?VIEW_EN[name]:VIEW_CN[name])||name;
 const MONTAGE_VIEWS=['front','back','left','right','top','bottom','current'],montageBoxes={};
 const stdViews=()=>FRAME?CORE.standardViews(FRAME.rotation,center,size*1.9):null;
 function initMontage(){const box=el('montage-views');box.replaceChildren();
  for(const name of MONTAGE_VIEWS){const lab=document.createElement('label'),cb=document.createElement('input');cb.type='checkbox';cb.dataset.view=name;
   cb.checked=['front','left','top','current'].includes(name);
   if(!FRAME&&name!=='current'){cb.checked=false;cb.disabled=true;}
   lab.appendChild(cb);const s=document.createElement('span');s.textContent=' '+VIEW_CN[name];lab.appendChild(s);box.appendChild(lab);montageBoxes[name]=cb;}
  if(!FRAME)el('montage-note').textContent='本报告未提供解剖坐标架，只能拼当前视角。';}
 const montageState=()=>({views:MONTAGE_VIEWS.filter(n=>montageBoxes[n]&&montageBoxes[n].checked),columns:Number(el('montage-cols').value)||3});
 function shotAt(name,views,saved){if(name==='current'||!views||!views[name])setCamera(saved.camera,false);else setCamera(views[name],false);tween=null;lastViewName=name;}
 async function exportMontage(){const note=x=>{el('montage-note').textContent=x;};
  const o=exportOptions(),lang=o.lang==='en'?'en':'zh',cfg=montageState();
  if(!cfg.views.length){note('请至少勾选一个视角。');return false;}
  const views=stdViews(),saved=currentViewState(),savedName=lastViewName,panels=[];
  try{
   for(const name of cfg.views){note('正在渲染 '+viewCaption(name,lang)+'（'+(panels.length+1)+' / '+cfg.views.length+'）…');shotAt(name,views,saved);
    const r=COMMON.renderOffscreen({renderer,scene,camera,THREE,width:view.clientWidth||960,height:view.clientHeight||640,scale:o.scale,background:o.background});
    // Each panel goes through the same overlay compositing as a single export; the shared colour bar is added later.
    const url=await composeExport(r.dataURL,r.width,r.height,Object.assign({},o,{colorbar:'none'}),lang);
    panels.push({image:await COMMON.loadImage(url),label:String.fromCharCode(97+panels.length),caption:viewCaption(name,lang)});}
   const bar=await COMMON.loadImage('data:image/svg+xml;charset=utf-8,'+encodeURIComponent(COMMON.colorbarSVG(Object.assign(colorbarOptions(lang),{orientation:'vertical'}))));
   const canvas=COMMON.composeMontage({panels,columns:cfg.columns,colorbar:{image:bar},title:String(META.case_id)+' · '+COMMON.englishLabel('wss_short',lang),font:Math.round(22*o.scale/2)});
   if(!canvas||typeof canvas.toDataURL!=='function')throw new Error('当前环境无法合成拼图画布');
   download(canvas.toDataURL('image/png'),COMMON.exportFilename({case_id:META.case_id,view:'montage',field:'wss',scale:o.scale,ext:'png'}));
   note('已导出 '+panels.length+' 幅 · '+cfg.columns+' 列 · 右侧共用色标。');return true;}
  catch(err){note('拼图导出失败：'+(err&&err.message||err));return false;}
  finally{shotAt('current',null,saved);lastViewName=savedName;}}
 el('montage-go').onclick=()=>{exportMontage();};
 async function buildSnapshots(){const note=x=>{el('snap-note').textContent=x;};
  if(!ONLINE){note('离线报告不能写回服务端；请在工作台打开的报告里生成配图。');return null;}
  const lang=el('exp-lang').value==='en'?'en':'zh',views=stdViews(),saved=currentViewState(),savedName=lastViewName;
  const plan=(views?['front','left','top']:[]).concat(['current']),images=[];
  try{for(const name of plan){note('正在渲染 '+viewCaption(name,lang)+'（'+(images.length+1)+' / '+plan.length+'）…');shotAt(name,views,saved);
    const r=await doExport({scale:2,background:'white',colorbar:'overlay',lang});
    images.push({name,png_base64:r.png_base64,width:r.width,height:r.height,caption:viewCaption(name,lang)+' · '+legendTitle(lang),view:name});}}
  catch(err){note('离屏渲染失败：'+(err&&err.message||err));shotAt('current',null,saved);lastViewName=savedName;return null;}
  shotAt('current',null,saved);lastViewName=savedName;
  // The service caps a single PNG at 6 MB and the whole body at 24 MB; say so before the server does.
  const bytes=images.reduce((s,i)=>s+Math.floor(i.png_base64.length*3/4),0);
  if(bytes>22*1024*1024||images.some(i=>i.png_base64.length*3/4>6*1024*1024)){note('配图过大（单张上限 6 MB、合计 24 MB）：请缩小窗口后重试。');return null;}
  note('正在上传 '+images.length+' 张配图（约 '+Math.round(bytes/1024)+' KB）…');
  try{const res=await apiPost(jobURL('snapshots'),{images,replace:true});
   const n=res&&res.snapshots&&Array.isArray(res.snapshots.items)?res.snapshots.items.length:images.length;
   const box=el('snap-note');box.replaceChildren();
   const span=document.createElement('span');span.textContent='已生成 '+n+' 张 · ';
   const a=document.createElement('a');a.href='onepage';a.target='_blank';a.rel='noopener';a.textContent='打开一页纸';
   box.append(span,a);return res;}
  catch(err){note(err&&err.status===409?'已审阅锁定，配图未写入。':'上传失败：'+(err&&err.message||err));return null;}}
 el('snap-onepage').onclick=()=>{buildSnapshots();};
 el('snap-onepage').hidden=!ONLINE;
 if(!ONLINE)el('snap-note').textContent='一页纸配图只在工作台在线打开的报告里可用。';
 // ---- §15.12 profile curves as standalone SVG (current branch; thresholds become dashed reference lines) ----
 function profileSVGs(){if(!PROFILES)return [];
  const list=profBranches(),b=list[0];if(!b)return [];
  const lang=el('exp-lang').value==='en'?'en':'zh',xkey=el('prof-x').value||'s_from_root_mm';
  const xs=Array.from(b[xkey]||b.s_local_mm||[],Number),name=b.name||branchName(b.segment_id);
  const shown=lang==='en'?COMMON.englishLabel(name,'en'):name,k=uf();
  const xLabel=lang==='en'?(xkey==='s_local_mm'?'Arc length in branch (mm)':'Arc length from inlet (mm)'):(xkey==='s_local_mm'?'分支内弧长 (mm)':'距入口弧长 (mm)');
  const w=b.wss||{},scaled=a=>Array.from(a||[],v=>Number.isFinite(+v)?+v*k:NaN);
  const series=[{name:lang==='en'?'Mean':'均值',x:xs,y:scaled(w.mean_pa),color:'#176ea2'},
   {name:'p99',x:xs,y:scaled(w.p99_pa),color:'#c0392b'},
   {name:lang==='en'?'Minimum':'最低',x:xs,y:scaled(w.min_pa),color:'#3f8f6b',dash:'7 5'}].filter(s=>s.y.some(Number.isFinite));
  const xf=xs.filter(Number.isFinite);
  if(xf.length>1)thresholdsPa.forEach((th,i)=>{const tag=(lang==='en'?['Low ','High ','Very high ']:['低 ','高 ','极高 '])[i];
   series.push({name:tag+fmt(th*k,1)+' '+ul(),x:[Math.min.apply(null,xf),Math.max.apply(null,xf)],y:[th*k,th*k],color:['#8aa0b5','#d9963c','#b03a2e'][i],dash:'5 5',width:1.2});});
  const stem=COMMON.safeName(META.case_id)+'_profile_'+COMMON.safeName(name);
  const out=[{key:'wss',branch:name,filename:stem+'_wss.svg',
   svg:COMMON.profileSVG({series,xLabel,yLabel:legendTitle(lang),title:String(META.case_id)+' · '+shown+' · '+legendTitle(lang)})}];
  const radius=Array.from(b.radius_mm||[],Number),rSeries=[];
  if(radius.some(Number.isFinite))rSeries.push({name:lang==='en'?'Radius':'半径',x:xs,y:radius,color:'#5a4fa3'});
  // §17.3: the morphology diameters ride along on the radius plot, in the same millimetre axis.
  const dmax=stationSeries(b,'max'),deq=stationSeries(b,'eq');
  if(dmax)rSeries.push({name:lang==='en'?'Max diameter':'最大直径',x:xs,y:dmax,color:'#8e44ad',dash:'6 5'});
  if(deq)rSeries.push({name:lang==='en'?'Equivalent diameter':'等效直径',x:xs,y:deq,color:'#c39bd3',dash:'2 4'});
  const rTitle=(dmax||deq)?(lang==='en'?'Radius / diameter':'半径 / 直径'):(lang==='en'?'Radius':'半径');
  if(rSeries.length)out.push({key:'radius',branch:name,filename:stem+'_radius.svg',
   svg:COMMON.profileSVG({series:rSeries,xLabel,yLabel:rTitle+' (mm)',title:String(META.case_id)+' · '+shown+' · '+rTitle})});
  return out;}
 if(PROFILES)el('prof-svg').onclick=()=>{const list=profileSVGs();
  if(!list.length){el('prof-note').textContent='没有可导出的沿程曲线。';return;}
  for(const item of list)download('data:image/svg+xml;charset=utf-8,'+encodeURIComponent(item.svg),item.filename);
  el('prof-note').textContent='已导出「'+list[0].branch+'」的 '+list.length+' 张 SVG（阈值为虚线参考）。'+(profBranches().length>1?'「全部叠加」时只导出第一条分支。':'');};
 // ---- C9 presets: built-ins are view-state generators; user presets live in preferences or localStorage ----
 const PRESET_LIST=COMMON.builtinPresets('wall',META);
 let USER_PRESETS=[];
 const presetCtx=()=>({meta:META,branchNames:BRANCH_NAMES,findings:FINDINGS,profiles:PROFILES,standardViews:FRAME?CORE.standardViews(FRAME.rotation,center,size*1.9):null});
 function applyPreset(p){const partial=p.build(presetCtx());applyViewState(Object.assign(currentViewState(),partial));PRESET_NAME=partial.preset_name||p.name;
  el('preset-note').textContent='已应用「'+PRESET_NAME+'」'+(p.description?'：'+p.description:'');return true;}
 function applyPresetByName(name){const b=PRESET_LIST.find(x=>x.name===name);if(b)return applyPreset(b);
  const u=USER_PRESETS.find(x=>x&&x.name===name);
  if(u&&u.state){applyViewState(Object.assign(currentViewState(),u.state));PRESET_NAME=name;el('preset-note').textContent='已应用用户预设「'+name+'」。';return true;}
  el('preset-note').textContent='没有名为「'+name+'」的预设。';return false;}
 function renderBuiltinPresets(){const box=el('preset-builtin');box.replaceChildren();
  for(const p of PRESET_LIST){const b=document.createElement('button');b.type='button';b.textContent=p.name;b.title=p.description||'';b.onclick=()=>applyPreset(p);box.appendChild(b);}}
 function localPresets(){try{const v=JSON.parse(localStorage.getItem('wss-report-presets:wall')||'[]');return Array.isArray(v)?v:[];}catch(_){return [];}}
 function renderUserPresets(){const box=el('preset-user');box.replaceChildren();
  for(const u of USER_PRESETS){if(!u||!u.name)continue;const row=document.createElement('div');row.className='item';const s=document.createElement('span');
   s.textContent=u.name+(u.created_at?' · '+String(u.created_at).slice(0,10):'');row.appendChild(s);
   const tools=document.createElement('span');tools.className='tools';
   const use=document.createElement('button');use.type='button';use.textContent='应用';use.onclick=()=>applyPresetByName(u.name);
   const ren=document.createElement('button');ren.type='button';ren.textContent='重命名';ren.onclick=()=>{const n=askText('新名称',u.name);if(!n)return;u.name=String(n).slice(0,40);persistPresets();};
   const exp=document.createElement('button');exp.type='button';exp.textContent='导出';exp.onclick=()=>download('data:application/json;charset=utf-8,'+encodeURIComponent(JSON.stringify(u,null,1)),fileStem()+'-preset-'+COMMON.safeName(u.name)+'.json');
   const del=document.createElement('button');del.type='button';del.textContent='删除';del.onclick=()=>{USER_PRESETS=USER_PRESETS.filter(x=>x!==u);persistPresets();};
   tools.append(use,ren,exp,del);row.appendChild(tools);box.appendChild(row);}
  if(!USER_PRESETS.length){const s=document.createElement('small');s.className='hint';s.textContent='还没有用户预设；调好视图后点「存为预设」。';box.appendChild(s);}}
 function persistPresets(){try{localStorage.setItem('wss-report-presets:wall',JSON.stringify(USER_PRESETS));}catch(_){}
  renderUserPresets();
  if(!ONLINE){el('preset-note').textContent='已存到本浏览器（离线报告）。';return;}
  el('preset-note').textContent='保存到服务端偏好中…';
  apiJSON(jobURL('/api/preferences')).then(j=>{const prefs=(j&&j.preferences&&typeof j.preferences==='object')?j.preferences:{};
   prefs.presets=Object.assign({},prefs.presets);prefs.presets.wall=USER_PRESETS;
   return apiPut(jobURL('/api/preferences'),prefs);}).then(()=>{el('preset-note').textContent='已保存到服务端偏好（其它偏好段不受影响）。';},
   err=>{el('preset-note').textContent='服务端保存失败（本地已存）：'+(err&&err.message||err);});}
 el('preset-save').onclick=()=>{const typed=String(el('preset-name').value||'').trim(),name=(typed||String(askText('预设名称','我的视图')||'')).trim().slice(0,40);
  if(!name)return;PRESET_NAME=name;const state=currentViewState();state.preset_name=name;
  USER_PRESETS=USER_PRESETS.filter(x=>!x||x.name!==name).concat([{name,state,created_at:new Date().toISOString()}]);persistPresets();};
 function loadUserPresets(){if(!ONLINE){USER_PRESETS=localPresets();renderUserPresets();return;}
  apiJSON(jobURL('/api/preferences')).then(j=>{const pr=(j&&j.preferences)||{};USER_PRESETS=Array.isArray(pr.presets&&pr.presets.wall)?pr.presets.wall:[];renderUserPresets();},
   ()=>{USER_PRESETS=localPresets();renderUserPresets();});}
 // ---- C6 report defaults: #view= hash > server preferences > localStorage > built-in ----
 const currentDefaults=()=>({colormap:cmapName,bands,units:UNIT,thresholds_pa:thresholdsPa.slice(),opacity,log:logScale,lang:LANG});
 function applyDefaults(d){if(!d||typeof d!=='object')return;
  applyViewState(Object.assign(currentViewState(),{colormap:d.colormap,bands:d.bands,units:d.units,thresholds_pa:d.thresholds_pa,opacity:d.opacity,log:d.log,lang:d.lang}));}
 el('set-defaults').onclick=()=>{const d=currentDefaults();
  try{const all=JSON.parse(localStorage.getItem('wss-report-defaults')||'{}');all.wall=d;localStorage.setItem('wss-report-defaults',JSON.stringify(all));el('defaults-note').textContent='已设为本浏览器默认口径。';}
  catch(_){el('defaults-note').textContent='浏览器不允许保存默认口径。';}
  if(!ONLINE)return;
  apiJSON(jobURL('/api/preferences')).then(j=>{const pr=(j&&j.preferences&&typeof j.preferences==='object')?j.preferences:{};
   pr.report_defaults=Object.assign({},pr.report_defaults);pr.report_defaults.wall=d;return apiPut(jobURL('/api/preferences'),pr);})
   .then(()=>{el('defaults-note').textContent='已设为默认口径（本浏览器 + 服务端偏好）。';},err=>{el('defaults-note').textContent='本地已保存；服务端保存失败：'+(err&&err.message||err);});};
 window.addEventListener('beforeprint',()=>{el('snapshot').src=capture();window.__menuOpen=MENUS.map(id=>el(id).open);el('menu-stats').open=true;});window.addEventListener('afterprint',()=>{if(window.__menuOpen)MENUS.forEach((id,i)=>{el(id).open=window.__menuOpen[i];});resize();});
 // ---- camera link protocol (contract §6) ----
 let lastPost=0,lastCam='',applyingRemote=0;
 const sameOrigin=ev=>{try{return /^https?:$/.test(String(location.protocol||''))&&!!ev&&ev.origin===location.origin;}catch(_){return false;}};
 function postToParent(msg){try{if(window.parent&&window.parent!==window)window.parent.postMessage(msg,location.origin);}catch(_){}}
 window.addEventListener('message',ev=>{const d=ev&&ev.data;if(!d||typeof d!=='object')return;
  if(d.type==='wss-view:set-camera'){if(!FRAME||!d.camera)return;try{const c=CORE.cameraFromAligned(d.camera,FRAME.rotation,FRAME.origin_mm);applyingRemote=performance.now()+150;setCamera(c,false);}catch(_){}return;}
  // §15.6: the compare page asks for the current display state; same-origin only and the camera is never touched.
  if(d.type==='wss-view:get-state'&&sameOrigin(ev))postToParent({type:'wss-view:state',request_id:d.request_id??null,family:'wall',state:currentViewState()});});
 function postCamera(){if(!FRAME||window.parent===window)return;const now=performance.now();if(now<applyingRemote||now-lastPost<50)return;const key=camera.position.toArray().map(v=>v.toFixed(2)).join(',')+'|'+controls.target.toArray().map(v=>v.toFixed(2)).join(',');if(key===lastCam)return;lastCam=key;lastPost=now;try{window.parent.postMessage({type:'wss-view:camera',family:'wall',run_identity:META.run_identity||null,camera:CORE.cameraToAligned({position:camera.position.toArray(),target:controls.target.toArray(),up:camera.up.toArray()},FRAME.rotation,FRAME.origin_mm)},'*');}catch(_){}}
 function frame(){if(tween){const u=Math.min(1,(performance.now()-tween.t0)/tween.ms),e=u<.5?2*u*u:1-Math.pow(-2*u+2,2)/2;camera.position.lerpVectors(tween.p0,tween.p1,e);controls.target.lerpVectors(tween.t0v,tween.t1,e);if(u>=1)tween=null;}controls.update();renderer.render(scene,camera);postCamera();const w=view.clientWidth,h=view.clientHeight;for(const it of labels){const v=it.obj.position.clone().project(camera);const show=it.obj.visible&&clipPlane.distanceToPoint(it.obj.position)>=0&&v.z>=-1&&v.z<=1;it.div.style.display=show?'':'none';if(show){it.div.style.left=(v.x+1)/2*w+'px';it.div.style.top=(1-v.y)/2*h+'px';}}requestAnimationFrame(frame);}
 // ---- C13 parent-page protocol (contract §12.6); the §6 camera link above stays unchanged ----
 const MSG=COMMON.viewMessaging({family:'wall',runIdentity:META.run_identity||null,caseId:META.case_id||null,
  onApplyState:s=>{s=s&&typeof s==='object'?s:{};
   // A lone preset_name means "apply that preset", not "an empty state".
   if(typeof s.preset_name==='string'&&Object.keys(s).length===1)return applyPresetByName(s.preset_name);
   return applyViewState(Object.assign(currentViewState(),s));},
  onExport:o=>doExport(o||{})});
 // ---- first paint: branch boxes, empty lists, annotations, presets, then the default / hash priority ----
 initBranchVis();initMontage();setMeasureMode('');renderMeasList();renderProbeLog();renderBuiltinPresets();loadUserPresets();loadAnnotations();redrawOverlays();
 if(!CLGROUPS.length){for(const k of Object.keys(NEED))el('measure-'+k).disabled=true;el('measure-status').textContent='本报告没有可用中心线，测量不可用。';}
 if(ONLINE)apiJSON(jobURL('files/findings_review.json')).then(doc=>{if(!doc)return;
   REVIEW={items:(doc.items&&typeof doc.items==='object')?doc.items:{},added:Array.isArray(doc.added)?doc.added:[]};applyAdded();renderFindingsList(LAST_ACTIVE);
   el('findings-review-note').textContent='已载入审阅记录（确认 / 驳回即时保存）。';},()=>{});
 else el('findings-review-note').textContent='离线只读：判定只保存在本页的视图状态里。';
 resize();resetCamera();setView('wss');frame();
 let hashApplied=false;
 try{const s=CORE.parseViewHash(location.hash);if(s){applyViewState(s);hashApplied=true;el('view-note').textContent='已从链接恢复视图。';}}catch(_){el('view-note').textContent='链接中的视图状态无法解析。';}
 if(!hashApplied){try{const all=JSON.parse(localStorage.getItem('wss-report-defaults')||'{}');if(all&&all.wall)applyDefaults(all.wall);}catch(_){}}
 if(!hashApplied&&ONLINE)apiJSON(jobURL('/api/preferences')).then(j=>{const d=j&&j.preferences&&j.preferences.report_defaults&&j.preferences.report_defaults.wall;
   if(d&&!hashApplied){applyDefaults(d);el('defaults-note').textContent='已套用服务端默认口径。';}},()=>{});
 MSG.ready();
 // Node-only hook: the DOM-stub tests drive picking and state without a real pointer or WebGL context.
 if(typeof module!=='undefined')window.__wssTest={state:currentViewState,apply:applyViewState,pick:handlePick,mode:setMeasureMode,
  hide:setBranchHidden,presets:()=>PRESET_LIST,applyPreset:applyPresetByName,gloss:GLOSS,text:t=>{TEXT_HOOK=t;},lists:()=>({MEAS,PROBES,ANNOTS,REVIEW,hidden:Array.from(HIDDEN)}),export:doExport,
  montage:exportMontage,snapshots:buildSnapshots,profileSVGs,label:labelFor,camera:()=>({position:camera.position.toArray(),target:controls.target.toArray()}),
  autoLabels:()=>AUTO_CHIPS.map(c=>({kind:c.kind,text:c.text,ring:Array.isArray(c.ring)?c.ring.length:0})),panel:()=>el('panel').innerHTML};
}
// Compact layout for narrow windows and the compare page's half-width iframes: overlay menu, slim header,
// smaller colour bar, probe card along the bottom.  Runs even when WebGL is unavailable.
(function(){
 const BREAK=1180,SHORT={'save-image':['保存截图','截图'],'print-report':['打印 / 保存 PDF','打印']};let on=false,override=null;
 try{const v=localStorage.getItem('wss-report-compact');if(v==='on'||v==='off')override=v;}catch(_){}
 const menu=()=>el('menu');
 function setOpen(open){const m=menu();if(!m||!m.classList)return;m.classList.toggle('open',Boolean(open));const b=el('menu-toggle');if(b&&b.setAttribute)b.setAttribute('aria-expanded',String(Boolean(open)));}
 function apply(force){
  const want=override==='on'?true:override==='off'?false:(Number(window.innerWidth)||1e4)<BREAK;
  if(!force&&want===on)return;on=want;
  if(document.body&&document.body.classList)document.body.classList.toggle('compact',on);
  for(const [id,[long,short]] of Object.entries(SHORT)){const b=el(id);if(!b)continue;b.textContent=on?short:long;if(b.setAttribute)b.setAttribute('title',long);}
  const lt=el('layout-toggle');if(lt)lt.textContent=on?'完整布局':'紧凑布局';
  setOpen(false);
 }
 const bind=(id,fn)=>{const b=el(id);if(b&&b.addEventListener)b.addEventListener('click',fn);};
 bind('menu-toggle',()=>{const m=menu();setOpen(!(m&&m.classList&&m.classList.contains('open')));});
 bind('menu-tab',()=>setOpen(true));bind('menu-close',()=>setOpen(false));
 bind('layout-toggle',()=>{override=on?'off':'on';try{localStorage.setItem('wss-report-compact',override);}catch(_){}apply(true);});
 {const v=el('view');if(v&&v.addEventListener)v.addEventListener('pointerdown',()=>{if(on)setOpen(false);},{capture:true});}
 window.addEventListener('resize',()=>apply(false));
 apply(true);
})();
try{initViewer();}catch(error){const d=document.createElement('div');d.className='fallback';d.textContent='无法初始化 3D 视图。请使用支持 WebGL 的浏览器并启用图形加速；右侧统计和打印仍可使用。';el('view').appendChild(d);el('save-image').disabled=true;
 for(const id of ['menu-findings','menu-profiles','menu-region','menu-measure','menu-annot','menu-presets','menu-view'])el(id).hidden=true;
 // The batch-export parent still needs to know this frame is done, even with no WebGL at all.
 try{COMMON.viewMessaging({family:'wall',runIdentity:META.run_identity||null,caseId:META.case_id||null,webgl:false}).ready({webgl:false});}catch(_){}
 console.error('WSS viewer:',error);}
</script></body></html>'''
