"""Report export and self-contained viewer; statistics remain on prediction points."""
from __future__ import annotations

import base64
import json
import os
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


def _segment_b64(values, key: str, dtypes: dict) -> str:
    """Branch ids as uint8 when they are integers in 0..255 (v0.14: a quarter of the int32 size), else int32.

    The chosen type is recorded in ``arrays["dt"][key]`` ("u8"); a page without that entry decodes int32 (all
    reports written before v0.14).
    """
    a = np.asarray(values)
    if a.size:
        as_int = a.astype(np.int64)
        small = bool(np.array_equal(as_int, a) and as_int.min() >= 0 and as_int.max() <= 255)
    else:
        small = True
    if small:
        dtypes[key] = "u8"
        return _b64(a, np.uint8)
    return _b64(a, np.int32)


def _derived_array_keys(meta: dict) -> set[str]:
    """Array keys of fields the viewer recomputes from embedded TAWSS / OSI (RRT, ECAP; ``cycle_fields``)."""
    fields = meta.get("fields") if isinstance(meta.get("fields"), dict) else {}
    out = set()
    for d in fields.values():
        if isinstance(d, dict) and d.get("source") == "derived" and list(d.get("derived_from") or []) == ["tawss", "osi"] \
                and isinstance(d.get("array_key"), str):
            out.add(d["array_key"])
    return out


def _write_atomic(path: Path, text: str) -> None:
    """Write next to the target and swap it in, so a reader never sees a half-written report."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


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


def build_html(out_path: Path, meta: dict, mesh: dict, cloud: dict, centerline: dict, *, embed_derived: bool = False) -> None:
    """Write the self-contained wall WSS report.

    ``mesh`` may carry an optional ``trust`` uint8 bitmask per vertex (contract §3);
    ``meta`` may carry ``profiles`` / ``findings`` / ``trust`` / ``review`` /
    ``frame_transform.rotation``.  Every one of those is optional: the viewer hides
    the corresponding panel when the key is absent.

    v0.14 payload: fields declared as derived from TAWSS / OSI (RRT, ECAP) are not embedded — the viewer
    recomputes them with the ``cycle_fields.derive_indices`` formula (``embed_derived=True`` keeps the v0.13
    layout); branch-id arrays are uint8 when they fit.  The file is written atomically.
    """
    meta = _report_schema_metadata(meta)
    three = (STATIC_DIR / "three.min.js").read_text(encoding="utf-8")
    orbit = (STATIC_DIR / "OrbitControls.js").read_text(encoding="utf-8")
    common = (STATIC_DIR / "report_common.js").read_text(encoding="utf-8")
    dtypes: dict = {}
    arrays = {
        "mv": _b64(mesh["vertices"], np.float32), "mf": _b64(mesh["faces"], np.uint32),
        "mw": _b64(mesh["wss"], np.float32), "ms": _segment_b64(mesh["segment"], "ms", dtypes),
        "pv": _b64(cloud["pts"], np.float32), "pw": _b64(cloud["wss"], np.float32),
        "ps": _segment_b64(cloud["segment"], "ps", dtypes), "p_s": _b64(cloud["s_from_root_mm"], np.float32),
        "p_th": _b64(cloud["theta_rad"], np.float32), "p_r": _b64(cloud["radius_mm"], np.float32),
        "p_dj": _b64(cloud["dist_to_junction_mm"], np.float32),
        "cv": _b64(centerline["xyz"], np.float32), "cr": _b64(centerline["radius_mm"], np.float32),
        "ce": _b64(centerline["edges"], np.uint32), "cs": _segment_b64(centerline["segment"], "cs", dtypes),
        "unroll_branches": branch_unroll_data(cloud),
    }
    if dtypes:
        arrays["dt"] = dtypes
    # Extra wall scalar fields (three-head release: TAWSS / OSI), keyed by the descriptor's array_key.
    extra_m = dict(mesh.get("extra") or {}); extra_c = dict(cloud.get("extra") or {})
    if set(extra_m) != set(extra_c):
        raise ValueError("mesh.extra and cloud.extra must describe the same fields")
    if extra_c:
        n_pts, n_vertices = len(np.asarray(cloud["pts"])), len(np.asarray(mesh["vertices"]))
        fields = meta.get("fields") if isinstance(meta.get("fields"), dict) else {}
        source_keys = {(fields.get(name) or {}).get("array_key") for name in ("tawss", "osi")}
        # RRT / ECAP are recomputed in the page from the embedded TAWSS / OSI (identical formula and floors).
        skip = set() if embed_derived or not source_keys <= set(map(str, extra_c)) else _derived_array_keys(meta)
        xf = {}
        for key, values in extra_c.items():
            point_values = np.asarray(values, dtype=np.float32); vertex_values = np.asarray(extra_m[key], dtype=np.float32)
            if point_values.shape != (n_pts,) or vertex_values.shape != (n_vertices,):
                raise ValueError(f"extra field {key} must hold one value per prediction point and per vertex")
            if str(key) in skip:
                continue
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
    _write_atomic(Path(out_path), html)


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
:root{--bg:#f3f6fa;--card:#fff;--line:#dce4ed;--ink:#203049;--muted:#5a6d80;--acc:#176caa;--soft:#f4f7fa;--success:#2f855a;--warning:#b7791f;--danger:#c0392b}
*{box-sizing:border-box}[hidden]{display:none!important}html,body{height:100%}body{height:100dvh;margin:0;font:14px/1.5 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg);display:flex;flex-direction:column;overflow:hidden}
header{flex:0 0 auto;display:flex;align-items:center;gap:14px;padding:8px 16px;background:#fff;border-bottom:1px solid var(--line);flex-wrap:wrap}h1{font-size:17px;margin:0}.sub,small{color:var(--muted);font-size:12px}.actions{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.back-link{flex:0 0 auto;font-size:13px;font-weight:600;color:var(--acc);text-decoration:none;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;white-space:nowrap}.back-link:hover{background:#edf4fa}.back-link:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.mode-tabs{display:flex;gap:4px;background:var(--soft);padding:4px;border-radius:9px}.mode-tabs button{border:0;background:transparent;padding:5px 12px;border-radius:6px;font-weight:600;color:#4d6478}.mode-tabs button.on{background:var(--acc);color:#fff}
button,select,input{font:inherit;max-width:100%}button{padding:5px 10px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}button.on{background:var(--acc);color:white;border-color:var(--acc)}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
main{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:350px minmax(0,1fr)}
aside.menu{overflow-y:auto;overflow-x:hidden;min-height:0;padding:10px;background:#f9fbfd;border-right:1px solid var(--line);overscroll-behavior:contain}
details.menu-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:0 12px;margin-bottom:8px}details.menu-card>summary{cursor:pointer;font-weight:650;padding:10px 0;list-style:none;display:flex;align-items:center;font-size:13px}details.menu-card>summary::before{content:"▸";display:inline-block;width:16px;color:var(--acc);transition:transform .15s}details.menu-card[open]>summary::before{transform:rotate(90deg)}details.menu-card>summary::-webkit-details-marker{display:none}details.menu-card>.body{padding:0 0 10px;display:grid;gap:6px;font-size:12px}
.body label{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.body input[type=range]{min-width:0;flex:1}.body select{max-width:100%}.num{width:66px}.row{display:flex;gap:5px;flex-wrap:wrap}.row button{font-size:12px;padding:4px 8px}.hint{color:var(--muted);line-height:1.4}.sep{border-top:1px solid var(--line);margin:4px 0}
.schema-controls{display:grid;gap:4px}.schema-label{font-weight:600}.field-tabs{display:flex;gap:4px;flex-wrap:wrap}.field-tab{font-size:12px;padding:3px 7px;min-height:25px}.field-tab[disabled]{cursor:not-allowed;opacity:.55}.field-tab.active{background:var(--acc);color:#fff;border-radius:6px}.frame-status{line-height:1.35}
.highlight-control output{min-width:34px;text-align:right;font-variant-numeric:tabular-nums}.metric-bars{display:grid;gap:7px;margin:7px 0 4px}.metric-bar{display:grid;grid-template-columns:92px minmax(0,1fr) auto;align-items:center;gap:7px;font-size:12px}.metric-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.metric-track{height:9px;background:#e9eef5;border-radius:8px;overflow:hidden}.metric-fill{display:block;height:100%;border-radius:8px;background:linear-gradient(90deg,#4b83c4,#c0392b);min-width:2px}.metric-fill.low{background:#4b83c4}.metric-fill.high{background:#df8b36}.metric-fill.very-high{background:#c0392b}.branch-chart .metric-track{height:8px}.branch-chart .metric-bar{grid-template-columns:82px minmax(0,1fr) 46px}.branch-chart .metric-fill{background:#547eb5}
#view{position:relative;background:#eef2f7;min-width:0;height:100%;overflow:hidden}#view>canvas{display:block}#label-leaders{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;overflow:visible}#labels div{position:absolute;transform:translate(-50%,-120%);background:#fffe;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12px;pointer-events:none;white-space:nowrap}
#tip{position:absolute;left:12px;bottom:12px;max-width:60%;background:#fffffff0;border:1px solid var(--line);border-radius:8px;padding:6px 8px 6px 10px;font-size:12px;white-space:pre-line;z-index:2;display:flex;gap:6px;align-items:flex-start}#tip-close{border:0;background:transparent;min-height:0!important;padding:0 2px;font-size:15px;line-height:1;color:var(--muted);cursor:pointer}
#cbar{position:absolute;right:70px;top:45px;width:18px;height:300px;border:1px solid #999;z-index:1}#cbar span{position:absolute;left:25px;font-size:12px;line-height:16px;white-space:nowrap;font-variant-numeric:tabular-nums}
#cbar i.tk{position:absolute;left:100%;width:5px;height:0;border-top:1px solid #6b7785}#cbar i.tk.minor{width:3px}
#cbar b.thr{position:absolute;left:-6px;right:0;height:0;border-top:2px solid #203049}#cbar em.thr-l{position:absolute;right:calc(100% + 8px);font-style:normal;font-size:12px;font-weight:650;line-height:16px;color:#203049;background:#ffffffd9;border-radius:3px;padding:0 3px;white-space:nowrap}#cbar-unit{position:absolute;right:12px;top:16px;max-width:180px;font-size:12px;text-align:right;z-index:1}
#legend-stack{position:absolute;right:12px;top:357px;z-index:1;display:flex;flex-direction:column;align-items:flex-end;gap:6px;max-width:240px}.legend-note{font-size:12px;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:2px 10px;color:#3f5a72}.legend-note span{white-space:nowrap}
#trust-legend{background:#fffffff0;border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;display:grid;gap:3px}.sw{display:inline-block;width:12px;height:12px;border:1px solid #999;vertical-align:-2px;margin-right:4px}.sw.hatch{background:repeating-linear-gradient(135deg,#d91a8c 0 3px,#fff 3px 6px)}
#field-seg{position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:3;display:flex;gap:2px;background:#fffffff2;border:1px solid var(--line);border-radius:10px;padding:3px;box-shadow:0 2px 8px #16334b1a}#field-seg button{border:0;background:transparent;min-height:30px;padding:3px 14px;border-radius:7px;font-weight:600;color:#4d6478;white-space:nowrap}#field-seg button.on{background:var(--acc);color:#fff}
#warn-banner{position:absolute;left:0;right:0;top:0;z-index:4;display:flex;align-items:center;gap:10px;padding:4px 12px;background:#fff4d6f5;border-bottom:1px solid #f0c36d;color:#74551b;font-size:12px}#warn-banner span{flex:1;min-width:0}#warn-banner button{min-height:24px;padding:1px 9px;font-size:12px;background:#fffdf5;border-color:#e8c889}
#view.has-banner #field-seg{top:40px}#view.has-banner #cbar-unit{top:46px}#view.has-banner #cbar{top:75px}#view.has-banner #legend-stack{top:387px}
.tech-pop{position:fixed;right:16px;bottom:40px;z-index:30;width:min(600px,calc(100vw - 32px));max-height:72vh;overflow:auto;background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 12px 36px #0003;padding:12px 14px;font-size:12px}.tech-pop .kv{grid-template-columns:auto minmax(0,1fr)}.tech-pop code{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere;user-select:all}.tech-head{display:flex;justify-content:space-between;align-items:center;font-weight:650;margin-bottom:8px}.tech-head button{min-height:26px;padding:1px 8px}
.card{border:1px solid var(--line);border-radius:9px;padding:10px 12px;margin-bottom:10px}.card h3{margin:0 0 7px;font-size:13px;font-weight:600}.big{font-size:26px;font-weight:700}.kv{display:grid;grid-template-columns:auto minmax(0,1fr);gap:3px 10px;font-size:12px}.kv span{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:4px 3px;text-align:right}th:first-child,td:first-child{text-align:left}summary{font-weight:600;cursor:pointer}details>div{margin-top:10px}p{margin:7px 0}.flag{background:#fff4d6;border:1px solid #f0c36d;border-radius:5px;padding:5px 8px;margin:5px 0;font-size:12px}.ok{background:#e6f6ea;border-color:#8fd19e}.unroll-row{margin:8px 0}.unroll-row canvas{width:100%;height:100px;border:1px solid var(--line);display:block}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto}.fallback{position:absolute;inset:40% 20px auto;padding:15px;background:#fff4d6;border-radius:8px;z-index:3}#snapshot{display:none}#print-note{display:none}
.finding{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:6px;align-items:center;text-align:left;padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:#fff;font-size:12px;width:100%}.finding.on,.finding.on:hover{border-color:var(--acc)!important;background:#eaf3fc!important;color:var(--ink)!important}.finding .rank{font-weight:700;color:var(--acc)}.finding .sub{display:block}.badge{font-size:12px;padding:1px 6px;border-radius:9px;background:#e9eef5;color:#4d6478;white-space:nowrap}.badge.attention{background:#fde7e3;color:#a53a2a}.badge.note{background:#fff2d5;color:#8b6519}.badge.info{background:#e5f0fa;color:#1766a0}
.prof canvas{width:100%;display:block;border:1px solid var(--line);background:#fff;cursor:crosshair}.prof-legend{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:12px}.prof-legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:3px}
footer{flex:0 0 auto;font-size:12px;color:var(--muted);padding:5px 16px;border-top:1px solid var(--line);background:#fff;display:flex;gap:14px;flex-wrap:wrap;align-items:center}footer b{color:var(--ink)}footer #tech-btn{margin-left:auto;min-height:26px;padding:1px 10px;font-size:12px}
/* The generic "button" rules (min-height, padding, radius) turned the term button into a tall ellipse: pin it to an 18 px circle. */
.gloss,button.gloss{display:inline-flex;align-items:center;justify-content:center;flex:0 0 18px;width:18px!important;height:18px!important;min-width:18px;min-height:0!important;padding:0!important;margin-left:3px;border-radius:50%!important;border:1px solid var(--line);background:#f4f7fa;color:var(--acc);font-size:12px!important;font-weight:700;line-height:1;vertical-align:1px;cursor:help}
.gloss-pop{position:fixed;z-index:20;max-width:320px;background:#fff;border:1px solid var(--line);border-radius:8px;box-shadow:0 6px 20px #0002;padding:8px 10px;font-size:12px;line-height:1.45}.gloss-pop b{display:block;margin-bottom:3px}
.item-list{display:grid;gap:4px}.item{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px;align-items:center;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:#fff;font-size:12px}.item>span,.item>button{overflow-wrap:anywhere;text-align:left}.item .tools{display:flex;gap:3px}.item .tools button{font-size:12px;padding:2px 6px}
.finding-tools{display:flex;gap:3px;flex-wrap:wrap;align-items:center;margin:2px 0 6px}.finding-tools button{font-size:12px;padding:2px 7px}.finding-tools input{flex:1;min-width:80px;font-size:12px;padding:2px 5px}.finding-wrap.rejected .finding{opacity:.5}
#probe-card{position:absolute;left:12px;bottom:118px;max-width:min(60%,560px);background:#fffffff5;border:1px solid var(--line);border-radius:10px;padding:8px 10px 7px;font-size:12px;z-index:2;display:grid;gap:6px;box-shadow:0 4px 16px #16334b1f}
#probe-card .pc-head{display:flex;align-items:baseline;gap:8px;min-width:0}#probe-card .pc-head b{font-size:13px;color:var(--ink);display:flex;align-items:center;gap:6px}#probe-card .pc-head b::before{content:'';width:8px;height:8px;border-radius:50%;background:#111;box-shadow:0 0 0 2px #fff,0 0 0 3px #111}
#probe-card .pc-where{color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}#probe-card .pc-actions{margin-left:auto;display:flex;gap:2px;align-items:center;align-self:center}
#probe-card .pc-rec{font-size:12px;padding:2px 10px;min-height:0;border-color:var(--acc);color:var(--acc);background:#fff}#probe-card .pc-rec:hover{background:#eaf3fc}
#probe-card .pc-x{border:0;background:transparent;min-height:0;padding:0 4px;font-size:17px;line-height:1;color:var(--muted)}#probe-card .pc-x:hover{color:var(--ink)}
#probe-card .pc-table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}#probe-card .pc-table th,#probe-card .pc-table td{padding:3px 8px;text-align:right;white-space:nowrap}
#probe-card .pc-table thead th{font-weight:500;font-size:11px;color:var(--muted);padding-bottom:4px;border-bottom:1px solid var(--line)}#probe-card .pc-table tbody th{text-align:left;font-weight:600;color:#33475b}#probe-card .pc-table tbody th small{font-weight:400;margin-left:4px;font-size:11px}
#probe-card .pc-table tbody tr{cursor:pointer}#probe-card .pc-table tbody tr:hover{background:var(--soft)}#probe-card .pc-table tbody tr.on{background:#eaf3fc}#probe-card .pc-table tbody tr.on th{color:var(--acc)}
#probe-card .pc-table td.pc-sec{font-weight:600;color:var(--ink)}#probe-card .pc-table th:first-child,#probe-card .pc-table td:first-child{padding-left:4px}
#probe-card .pc-range{position:relative;display:inline-block;width:76px;height:14px;vertical-align:middle}#probe-card .pc-range::before{content:'';position:absolute;left:0;right:0;top:6px;height:2px;border-radius:1px;background:#cfd9e4}
#probe-card .pc-mean{position:absolute;top:1px;width:2px;height:12px;margin-left:-1px;border-radius:1px;background:var(--ink)}#probe-card .pc-pt{position:absolute;top:3px;width:8px;height:8px;margin-left:-4px;border-radius:50%;background:var(--acc);box-shadow:0 0 0 1.5px #fff}#probe-card .pc-pt.out{background:#fff;box-shadow:inset 0 0 0 1.5px var(--acc)}
#probe-card .pc-foot{display:grid;gap:1px;color:var(--muted);font-size:11px;line-height:1.45;border-top:1px solid var(--line);padding-top:5px}
#probe-card .pc-legend{margin-left:10px;white-space:nowrap}#probe-card .pc-legend i{display:inline-block;vertical-align:middle;margin:0 3px 0 7px}#probe-card .pc-lg-mean{width:2px;height:10px;border-radius:1px;background:var(--ink)}#probe-card .pc-lg-pt{width:7px;height:7px;border-radius:50%;background:var(--acc)}
.probe-table{font-size:12px}.mode-row button.on{background:var(--acc);color:#fff}#branch-vis label{font-size:12px;margin-right:6px}
#labels div.meas{background:#e8f6f8;border-color:#5bbcc9}#labels div.annot{background:#fff7e6;border-color:#e0a84a;pointer-events:auto;cursor:pointer;max-width:240px;white-space:normal}
#labels div.flabel{pointer-events:auto;cursor:pointer;font-weight:600}#labels div.flabel.attention{background:#fde7e3f0;border-color:#e2a294;color:#8d3223}#labels div.flabel.note{background:#fff2d5f0;border-color:#e6c47a;color:#7a5814}#labels div.flabel.info{background:#e5f0faf0;border-color:#9dc3e6;color:#175b8c}
#labels div.blabel{background:#eef4fbf0;border-color:#a9c4de;color:#2a4c6b;font-weight:600}#labels div.maxd{background:#f5ecfbf0;border-color:#b98fe0;color:#6c3483;font-weight:600}#labels div.trough{background:#fffffff2;border-color:#5f6f86;color:#1f2a3a;font-weight:600}
#narrative-card p{margin:4px 0}#narrative-card .edited{background:#eaf3fc;border-color:#9dc3e6}
/* v0.13 menu tabs, quick statistics, grouped rows */
.menu-tabs{position:sticky;top:-14px;z-index:4;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px;margin:-14px -14px 10px;padding:10px 14px 8px;background:#f7f9fc;border-bottom:1px solid var(--line)}
.menu-tabs button{min-height:34px;padding:4px 6px;border:1px solid var(--line);background:#fff;border-radius:8px;font-weight:650;color:#4d6478;display:inline-flex;align-items:center;justify-content:center;gap:4px}
.menu-tabs button.on,.menu-tabs button.on:hover{background:var(--acc);border-color:var(--acc);color:#fff}
.tab-count{font-size:12px;min-width:18px;padding:0 5px;border-radius:9px;background:#fde7e3;color:#a53a2a;line-height:17px}.menu-tabs button.on .tab-count{background:#ffffff33;color:#fff}
.view-row{display:flex;align-items:center;gap:6px}.row-label{font-weight:600;flex:0 0 auto}.view-row .row{gap:3px;flex-wrap:nowrap}.view-row .row button{min-width:0;padding:4px 7px;font-size:12px}
.quick-stats{display:flex;flex-wrap:wrap;align-items:center;gap:3px 10px;padding:7px 9px;border-radius:8px;background:#f1f6fb;border:1px solid #dbe7f2;font-size:12px;font-variant-numeric:tabular-nums}.quick-stats b{color:var(--acc)}
.quick-stats .linkish,.linkish{border:0;background:transparent;color:var(--acc);padding:0 2px;min-height:0!important;font-size:12px;font-weight:600;margin-left:auto;cursor:pointer}.quick-stats .linkish:hover{text-decoration:underline;background:transparent}
.export-main{display:grid!important;grid-template-columns:1fr 1fr;gap:6px}.export-main button{min-height:36px}
/* v0.13.1: menu content is never wider than the menu (a table's natural width used to stretch its card past the
   clipped edge with wide system fonts); wide tables scroll sideways inside .tscroll with the first column pinned,
   and a soft shadow on the right says there is more. */
.menu-pane,details.menu-card,#panel,#panel .card,details.menu-card>.body>*{min-width:0;max-width:100%}
details.menu-card>.body{grid-template-columns:minmax(0,1fr)}
.tscroll{overflow-x:auto;max-width:100%;overscroll-behavior-x:contain;-webkit-overflow-scrolling:touch;margin:2px 0;
  background:linear-gradient(to left,#fff 30%,#fff0) right center/28px 100% no-repeat local,radial-gradient(farthest-side at 100% 50%,#0000002b,#0000) right center/12px 100% no-repeat scroll}
.tscroll table{width:100%;min-width:max-content}.tscroll th,.tscroll td{white-space:nowrap}
.tscroll th:first-child,.tscroll td:first-child{position:sticky;left:0;z-index:1;background:#fff;box-shadow:1px 0 0 var(--line)}
.flag,.kv span,#panel small,#panel p,.finding .sub{overflow-wrap:anywhere}
/* compact layout (narrow window / compare-page iframe) */
@media screen{
.compact-only{display:none}.compact .compact-only{display:inline-flex}
/* Title, display modes and actions occupy separate rows in half-width reports. */
.compact header{display:grid;grid-template-columns:auto minmax(0,1fr);gap:5px 8px;padding:6px 10px;overflow:visible}.compact header.has-back{grid-template-columns:auto auto minmax(0,1fr)}.compact .back-link{padding:4px 8px;font-size:12px}
.compact header>div:nth-of-type(1){min-width:0;overflow:hidden}
.compact header h1{font-size:14px}
.compact #subtitle{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0}
.compact header .gloss{display:none}
.compact header .mode-tabs{grid-column:1/-1;min-width:0;display:flex;flex-wrap:wrap;padding:3px}
.compact .mode-tabs button{flex:1 1 auto;padding:4px 9px;font-size:12px;white-space:nowrap}
.compact header .actions{grid-column:1/-1;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:4px;margin:0;padding-top:5px;border-top:1px solid var(--line);min-width:0}
.compact header .actions button,.compact #menu-toggle{padding:4px 8px;font-size:12px;white-space:nowrap}
header .mode-tabs{flex-wrap:wrap}header>div:nth-of-type(1){min-width:0;overflow-wrap:anywhere}
.compact main{display:block;position:relative}.compact aside.menu{position:absolute;left:0;top:0;bottom:0;width:min(340px,88%);z-index:6;transform:translateX(-102%);transition:transform .15s;box-shadow:4px 0 18px #16334b26}.compact aside.menu.open{transform:none}.compact #view{height:100%;min-width:0}.compact main.menu-open{display:grid;grid-template-columns:minmax(190px,42%) minmax(0,1fr)}.compact main.menu-open aside.menu{position:relative;left:auto;top:auto;bottom:auto;width:auto;transform:none;transition:none;box-shadow:4px 0 18px #16334b26;z-index:2}.compact main.menu-open #menu-tab{display:none}
@media(max-width:560px){.compact main.menu-open{display:block}.compact main.menu-open aside.menu{position:absolute;left:0;top:0;bottom:0;width:min(340px,88%);transform:none}}
.menu-head{display:none}.compact .menu-head{display:flex;justify-content:space-between;align-items:center;margin:0 0 8px;font-weight:650}.menu-head button{font-size:12px;padding:3px 8px}
#menu-tab{display:none}.compact #menu-tab{display:block;position:absolute;left:0;top:50%;transform:translateY(-50%);z-index:3;writing-mode:vertical-rl;padding:10px 4px;border-radius:0 8px 8px 0;border:1px solid var(--line);border-left:0;background:#fffffff0;font-size:12px;letter-spacing:2px;color:#3f5a72}
.compact footer{display:none}
.compact #cbar{right:34px;top:32px;height:210px}.compact #cbar span{font-size:12px}.compact #cbar em.thr-l{font-size:12px}.compact #cbar-unit{top:8px;right:8px;font-size:12px;max-width:150px}.compact #legend-stack{top:254px;right:8px;max-width:170px}.compact .legend-note,.compact #trust-legend{font-size:12px}
.compact #probe-card{left:8px;right:auto;bottom:8px;max-width:calc(100% - 16px);font-size:12px;padding:5px 8px;gap:3px}.compact #probe-card .pc-foot span+span{display:none}.compact #probe-card .pc-table th,.compact #probe-card .pc-table td{padding:1px 6px}.compact #tip{bottom:auto;top:8px;left:8px;max-width:45%;font-size:12px;padding:5px 8px}
.compact #field-seg{top:6px}.compact #field-seg button{min-height:26px;padding:2px 9px;font-size:12px}.compact #view.has-seg #tip{top:44px}
.compact #view.has-banner #cbar-unit{top:40px}.compact #view.has-banner #cbar{top:64px}.compact #view.has-banner #legend-stack{top:286px}.compact #view.has-banner #field-seg{top:36px}.compact #view.has-banner #tip{top:40px}.compact #view.has-banner.has-seg #tip{top:74px}
}
@page{size:A4 portrait;margin:10mm}
@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact}body{background:#fff;font-size:9pt;overflow:visible;display:block}header{padding:0 0 5mm}header h1{font-size:14pt}.back-link,.actions,.mode-tabs,#tip,#probe-card,.gloss,.gloss-pop,.legend-note,#labels,#view>canvas,#cbar,#cbar-unit,#trust-legend,#legend-stack,#label-leaders,#field-seg,#warn-banner,.tech-pop,#tech-btn,.wss-shortcuts,#menu-tab,#menu-toggle,.menu-head,.process-only,.menu-only{display:none!important}main{display:block;height:auto;min-height:0}#view{height:auto;min-height:0;background:white;overflow:visible}#snapshot{display:block;width:100%;max-height:115mm;object-fit:contain}aside.menu{padding:3mm 0 0;overflow:visible;border:0;background:#fff}details.menu-card{border:0;padding:0;margin:0}details.menu-card:not(#menu-stats){display:none}.menu-tabs{display:none!important}.menu-pane{display:block!important}#menu-stats>summary{display:none}#menu-stats>.body{display:block}#panel{display:grid;grid-template-columns:1fr 1fr;gap:3mm}.card{margin:0;padding:2.5mm;border-radius:2mm;break-inside:avoid}.card h3{font-size:9pt;margin-bottom:1mm}.big{font-size:18pt}table{font-size:8pt}.branch-card{grid-column:1/-1}.kv{font-size:8pt}.sub,small{font-size:8pt}#print-note{display:block;grid-column:1/-1;font-size:8pt}footer{border:0;padding:2mm 0}}

/* Shared workbench scale: readable controls and a restrained report palette. */
@media screen{
header{padding:12px 18px;gap:16px}header h1{font-size:18px}button,select{min-height:34px;border-radius:8px}button:hover:not(:disabled){border-color:var(--line);background:#edf4fa}button.on:hover:not(:disabled){background:#125889;color:#fff}.mode-tabs button.on:hover{background:#125889}
main{grid-template-columns:340px minmax(0,1fr)}aside.menu{padding:14px;background:#f7f9fc}details.menu-card{border-radius:12px;margin-bottom:10px}details.menu-card>summary{padding:12px 0;font-size:14px}details.menu-card>.body{font-size:13px;line-height:1.6;gap:9px}.row{gap:8px}.row button{font-size:13px;padding:6px 10px}.card h3{font-size:14px}.kv,table,.metric-bar,.finding,.item{font-size:12px}
details.sub-details{border-top:1px solid var(--line);margin-top:8px;padding-top:4px}details.sub-details>summary{font-size:13px;color:var(--muted);padding:8px 0;cursor:pointer}details.sub-details>.body{display:grid;gap:9px;padding-bottom:4px}.body input[type=number],.body input[type=text],.body select{border:1px solid var(--line);border-radius:7px;padding:5px 7px;background:#fff;color:var(--ink)}input[type=checkbox],input[type=range]{accent-color:#176caa}.badge,.sev{font-size:12px}.badge.attention,.sev.attention{background:#fde7e3;color:var(--danger)}.badge.note,.sev.note{background:#fff2d5;color:#8b6519}.badge.info,.sev.info{background:#e5f0fa;color:#176caa}.warning{background:#fff2d5;color:#74551b}#labels div.flabel.attention{background:#fde7e3f0;border-color:#e2a294;color:#8d3223}#labels div.flabel.note{background:#fff2d5f0;border-color:#e6c47a;color:#7a5814}#labels div.flabel.info{background:#e5f0faf0;border-color:#9dc3e6;color:#175b8c}.flag{background:#fff2d5;border-color:#e8c889;color:#74551b}.flag.ok{background:#e9f6ee;border-color:#a9d8bb;color:var(--success)}
.compact header h1{font-size:15px}.compact .actions button,.compact .mode-tabs button{min-height:32px}.compact aside.menu{padding:10px}.compact .menu-tabs{top:-10px;margin:0 -10px 8px;padding:6px 10px}.compact details.menu-card>.body{font-size:13px}.compact .legend-note,.compact #trust-legend,.compact #tip,.compact #view-note{font-size:12px}
.field-tab.active:hover,.mode-tabs button.on:hover,.mode-row button.on:hover{background:#125889;color:#fff}
}
/* v0.15: touch screens (iPad) get 44 px controls; the 16–18 px「?」circle keeps its size with a larger hit area. */
@media (pointer:coarse){button:not(.gloss),select,summary,.back-link{min-height:44px!important}button:not(.gloss){min-width:40px}#std-views button{min-width:40px!important;flex:1 1 40px}button.linkish,#quick-stats-more{display:inline-flex;align-items:center;min-height:40px!important}#tip-close{min-height:40px!important;min-width:40px}.gloss{position:relative}.gloss:after{content:"";position:absolute;inset:-12px}input[type=checkbox]{width:22px;height:22px}}
/* v0.15.4: the 3-D viewport's own size (not the window) drives the overlays of the full layout — in a compare-page half or
   a narrow window the field switch moves to the top-left corner and the colour bar shortens, so they no longer overlap
   each other or the vessel. */
@media screen{#view{container:wview/size}
@container wview (max-width:760px){
body:not(.compact) #field-seg{top:8px;left:8px;transform:none}
body:not(.compact) #field-seg button{min-height:26px;padding:2px 9px;font-size:12px}
body:not(.compact) #cbar-unit{top:8px;right:8px;max-width:150px}
body:not(.compact) #cbar{right:34px;top:32px;height:clamp(150px,40cqh,300px)}
body:not(.compact) #legend-stack{top:calc(44px + clamp(150px,40cqh,300px));right:8px;max-width:170px}
body:not(.compact) #view.has-banner #field-seg{top:38px}body:not(.compact) #view.has-banner #cbar-unit{top:40px}
body:not(.compact) #view.has-banner #cbar{top:64px}body:not(.compact) #view.has-banner #legend-stack{top:calc(76px + clamp(150px,40cqh,300px))}
body:not(.compact) #tip{left:8px;bottom:8px;max-width:calc(100% - 110px);padding:4px 7px}
}
@container wview (max-width:520px){
body:not(.compact) #field-seg{max-width:calc(100% - 16px);overflow-x:auto}body:not(.compact) #field-seg button{padding:2px 7px}body:not(.compact) #tip{display:none}
body:not(.compact) #cbar-unit{top:44px}body:not(.compact) #cbar{top:66px;height:clamp(120px,36cqh,260px)}body:not(.compact) #legend-stack{top:calc(78px + clamp(120px,36cqh,260px))}
body:not(.compact) #view.has-banner #cbar-unit{top:74px}body:not(.compact) #view.has-banner #cbar{top:96px}body:not(.compact) #view.has-banner #legend-stack{top:calc(108px + clamp(120px,36cqh,260px))}
}
}
</style></head><body>
<header><a id="back-to-workbench" class="back-link" href="/" title="回到工作台（本病例）" hidden>← 工作台</a><button type="button" id="menu-toggle" class="compact-only" aria-expanded="false" title="打开菜单">☰ 菜单</button><div><h1>壁面 WSS 预测报告</h1><div class="sub" id="subtitle"></div></div>
<div class="mode-tabs view-mode-controls" role="group" aria-label="显示对象"><button type="button" id="tab-wss" aria-pressed="true" aria-controls="view" data-v="wss" class="on">WSS 壁面</button><button type="button" id="tab-stl" aria-pressed="false" aria-controls="view" data-v="stl">输入 STL</button><button type="button" id="tab-cl" aria-pressed="false" aria-controls="view" data-v="cl">中心线</button><button type="button" id="tab-cloud" aria-pressed="false" aria-controls="view" data-v="cloud">预测点云</button></div>
<div class="actions"><button type="button" id="save-image">保存截图</button><button type="button" id="print-report">打印 / 保存 PDF</button><button type="button" id="layout-toggle" title="在紧凑布局与完整布局之间切换（窄窗口与并排比较时自动紧凑）">紧凑布局</button></div></header>
<main><aside class="menu" id="menu"><div class="menu-head"><span>菜单</span><button type="button" id="menu-close">◂ 收起</button></div>
<div class="menu-tabs" role="tablist" aria-label="功能分组"><button type="button" role="tab" id="mtab-view" data-pane="view" aria-controls="pane-view" aria-selected="true" class="on">查看</button><button type="button" role="tab" id="mtab-measure" data-pane="measure" aria-controls="pane-measure" aria-selected="false">测量</button><button type="button" role="tab" id="mtab-review" data-pane="review" aria-controls="pane-review" aria-selected="false">审阅<span class="tab-count" id="mtab-review-count" hidden></span></button><button type="button" role="tab" id="mtab-export" data-pane="export" aria-controls="pane-export" aria-selected="false">导出</button></div>
<div class="menu-pane" id="pane-view" role="tabpanel" aria-labelledby="mtab-view">
<details class="menu-card" id="menu-display" open><summary>显示</summary><div class="body">
<div class="schema-controls" id="schema-controls" aria-label="结果字段"><div class="schema-label">结果字段</div><div id="field-tabs" class="field-tabs"></div></div>
<label>配色 <select id="cmap"><option value="rainbow">彩虹（默认）</option><option value="viridis">Viridis</option><option value="turbo">Turbo</option><option value="bwr">蓝白红</option></select><select id="bands" title="色带分段"><option value="0">连续</option><option value="4">4 段</option><option value="6">6 段</option><option value="8">8 段</option><option value="10">10 段</option><option value="12">12 段</option><option value="16">16 段</option><option value="20">20 段</option></select></label>
<small id="scale-description" class="hint"></small>
<label>分支 <select id="branch"><option value="-1">全部</option></select></label>
<label><input type="checkbox" id="showpeak" checked> 全场最大值标记（TAWSS 另标最小值）</label>
<div class="row"><label><input type="checkbox" id="showpts"> 叠加点云</label><label><input type="checkbox" id="showcl"> 叠加中心线</label></div>
<label id="stag-row" hidden title="单周期 TAWSS &lt; 0.4 Pa 且 OSI &gt; 0.1 的壁面：血流慢且方向来回摆动；在任一字段下用品红斜纹叠加"><input type="checkbox" id="stag"> 滞留区（TAWSS &lt; 0.4 且 OSI &gt; 0.1）</label>
<div class="view-row"><span class="row-label">视角</span><div class="row" id="std-views"><button type="button" data-view="front">前</button><button type="button" data-view="back">后</button><button type="button" data-view="left">左</button><button type="button" data-view="right">右</button><button type="button" data-view="top">上</button><button type="button" data-view="bottom">下</button><button type="button" id="reset-camera" title="解剖前视并撑满视口（快捷键 0 / R）">复位</button></div></div>
<div id="quick-stats" class="quick-stats" aria-live="polite"></div>
<div class="sub" id="display-readout">当前视图：WSS 壁面 · 统计：预测点云 · 壁面：Gaussian 插值</div><small class="hint">拖动旋转 · 滚轮缩放 · 右键平移 · 按 <b>?</b> 查看快捷键</small>
<details class="sub-details" id="more-display"><summary>更多显示设置</summary><div class="body">
<label title="沿 STL 坐标的 Z 轴水平切开模型，只保留切面以下的部分；拖到最右 = 不剖切">剖切高度（STL Z 轴） <input type="range" id="clip" min="0" max="100" value="100" aria-label="剖切高度：沿 STL Z 轴保留切面以下部分"><output id="clip-value">不剖切</output></label>
<label>壁面不透明度 <input type="range" id="opacity" min="0.2" max="1" step="0.05" value="1"></label>
<small id="std-note" class="hint"></small><button type="button" class="gloss" data-gloss="standard_views">?</button>
<label>时间帧 <select id="frame-select" aria-describedby="frame-status" disabled></select></label><small id="frame-status" class="frame-status"></small>
<div class="sep"></div>
<label>色标范围 <select id="scale-mode"><option value="case">本例 p99</option><option value="fixed">固定上限 · 跨报告保留</option></select></label>
<label id="fixed-control" hidden>固定上限 <input id="fixedmax" class="num" type="number" min="0.01" step="0.5" value="10"> <span id="fixed-unit">Pa</span></label>
<label><input type="checkbox" id="logscale"> 对数色标（下限 0.05 Pa）</label>
<label>单位<button type="button" class="gloss" data-gloss="units_wss">?</button> <select id="units"><option value="Pa">Pa</option><option value="dyn">dyn/cm²</option></select></label>
<div class="sep"></div>
<label><span id="thr-label-text">阈值</span><button type="button" class="gloss" data-gloss="thresholds">?</button> <input id="thr0" class="num" type="number" min="0" step="0.1"> / <input id="thr1" class="num" type="number" min="0" step="0.5"> / <input id="thr2" class="num" type="number" min="0" step="0.5"> <span id="thr-unit">Pa</span></label>
<small class="hint" id="thr-hint">低 / 高 / 极高阈值；跟随当前着色字段，改动即时重算面积占比、等值线与色标刻线，不写入 summary.json。</small>
<label><input type="checkbox" id="contours"> 等值线（三条阈值）</label>
<label><input type="checkbox" id="trust"> 可信区域叠加<button type="button" class="gloss" data-gloss="trust_interpolation_uncovered">?</button> <small id="trust-hint" class="hint"></small></label>
<div class="sep"></div>
<div id="branch-vis" class="row" aria-label="分支显隐"></div><small id="branch-vis-note" class="hint" hidden>已隐藏分支只影响显示；统计不受显隐影响。</small>
<label class="highlight-control">高亮阈值 <input type="range" id="highlight-pct" min="0.5" max="10" step="0.5" value="1"><output id="highlight-value">1%</output></label>
<label><input type="checkbox" id="showtop"> 启用高亮（最高 <span id="highlight-label">1%</span>）</label>
<div class="sep"></div>
<label>自动标注发现 <select id="lbl-findings"><option value="0">关</option><option value="3">前 3</option><option value="5">前 5</option><option value="10">前 10</option></select></label>
<label><input type="checkbox" id="lbl-branches"> 分支名</label>
<small class="hint">发现标签按排名钉在三维上（驳回项与隐藏分支不钉）；分支名钉在各分支中点。标签随导图、拼图与一页纸配图一起输出（需勾选「含界面标签」）。</small>
<div class="sep"></div>
<label>点云特征 <select id="feat"><option value="wss">WSS</option><option value="s">沿程弧长 (mm)</option><option value="th">周向角 (rad)</option><option value="r">半径 (mm)</option><option value="dj">到分叉距离 (mm)</option></select></label>
<div class="row"><button type="button" id="set-defaults">把当前口径设为默认</button></div><small id="defaults-note" class="hint">默认口径：配色、分段、单位、阈值、透明度、对数、语言；新开报告时生效，视图状态里仍显式记录。</small>
</div></details>
</div></details>
<details class="menu-card" id="menu-stats"><summary>统计与口径</summary><div class="body"><div id="panel"></div>
<details class="card process-only" id="unroll-details"><summary>分支展开图（s, θ）</summary><div><small>每条分支单独显示，横轴为距入口弧长（mm），纵轴为周向角 −π 至 π。像素汇总显示该像素的最大预测值（当前着色字段）。</small><div id="unroll-plots"></div></div></details>
</div></details>
<details class="menu-card" id="menu-presets"><summary>预设</summary><div class="body">
<div id="preset-builtin" class="row"></div><small class="hint">内置预设按本例的分支名与发现列表计算。</small>
<div class="sep"></div><label>名称 <input id="preset-name" maxlength="40" placeholder="我的视图"><button type="button" id="preset-save">存为预设</button></label>
<div id="preset-user" class="item-list"></div><small id="preset-note" class="hint"></small>
</div></details>
</div>
<div class="menu-pane" id="pane-measure" role="tabpanel" aria-labelledby="mtab-measure" hidden>
<details class="menu-card" id="menu-measure"><summary>测量与探针</summary><div class="body">
<div class="row mode-row" id="measure-modes"><button type="button" id="measure-distance">距离</button><button type="button" id="measure-arc">弧长</button><button type="button" id="measure-diameter">管径</button><button type="button" id="measure-segment">分段</button><button type="button" id="measure-clear">清空</button></div>
<small id="measure-hint" class="hint">选择模式后在壁面上点击取点；再次点击模式按钮退出。管径按该处中心线切线切出真实截面轮廓，给出最大 / 最小 / 等效直径与截面面积（等效直径 = 2√(面积/π)），轮廓画在三维里；取不到闭合轮廓时回退为内切半径 × 2 并注明<button type="button" class="gloss" data-gloss="local_diameter">?</button>；弧长沿中心线树计算<button type="button" class="gloss" data-gloss="arc_distance">?</button>。</small>
<small id="measure-status" class="hint">未选择模式：点击壁面锁定探针。</small>
<div id="measure-list" class="item-list"></div>
<div class="sep"></div><b>探针记录</b><small class="hint">点击壁面锁定探针后按「记录」；数值来自最近预测点，壁面读值为 Gaussian 插值。</small>
<div id="probe-log"></div><div class="row"><button type="button" id="probe-copy">复制 TSV</button><button type="button" id="probe-csv">导出 CSV</button><button type="button" id="probe-clear">清空记录</button></div>
</div></details>
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
<small id="prof-note" class="hint">导出当前分支的两张 SVG：上图随当前字段（WSS 均值 / p99 / 最低，或 TAWSS / OSI / RRT / ECAP）与半径；阈值画成虚线参考。</small>
</div></details>
<details class="menu-card" id="menu-region"><summary>区域统计</summary><div class="body">
<label>方式 <select id="region-mode"><option value="range">分支 + 弧长区间</option><option value="sphere">点选球形刷子</option></select></label>
<div id="region-range"><label>分支 <select id="region-branch"></select></label><label>起点 <input type="range" id="region-s0" min="0" max="100" step="0.5" value="0"><output id="region-s0-v"></output></label><label>终点 <input type="range" id="region-s1" min="0" max="100" step="0.5" value="100"><output id="region-s1-v"></output></label></div>
<div id="region-sphere" hidden><label><input type="checkbox" id="region-pick"> 点击壁面选中心</label><label>半径 <input type="range" id="region-radius" min="1" max="40" step="0.5" value="8"><output id="region-radius-v">8 mm</output></label><small id="region-center" class="hint">尚未选点</small></div>
<div class="row"><button type="button" id="region-apply">计算并高亮</button><button type="button" id="region-clear">清除</button></div>
<div class="kv" id="region-stats"></div><small class="hint">统计基于区域内预测点；估计面积 = 点占比 × 输入壁面面积。</small>
</div></details>
</div>
<div class="menu-pane" id="pane-review" role="tabpanel" aria-labelledby="mtab-review" hidden>
<details class="menu-card" id="menu-findings"><summary>发现</summary><div class="body"><div id="findings-list"></div><small id="findings-note" class="hint"></small><div class="row"><button type="button" id="finding-add">新增发现（点选壁面）</button><button type="button" id="finding-save">保存审阅</button><button type="button" class="gloss" data-gloss="finding_decision">?</button></div><small id="findings-review-note" class="hint"></small></div></details>
<details class="menu-card" id="menu-annot"><summary>标注</summary><div class="body">
<div class="row"><button type="button" id="annot-add">在壁面钉标注</button><button type="button" id="annot-save">保存到服务</button></div>
<small id="annot-status" class="hint"></small><div id="annot-list" class="item-list"></div>
</div></details>
</div>
<div class="menu-pane" id="pane-export" role="tabpanel" aria-labelledby="mtab-export" hidden>
<details class="menu-card" id="menu-view"><summary>导出</summary><div class="body">
<div class="row export-main"><button type="button" id="exp-png">导出出版级 PNG</button><button type="button" id="views-export">导出六视角</button><button type="button" id="exp-svg">导出色标 SVG</button><button type="button" id="snap-onepage">生成一页纸配图</button></div>
<small id="exp-note" class="hint">离屏渲染当前视图，不含菜单；测量与标注标签随图输出；「导出六视角」同样按此设置。</small><small id="snap-note" class="hint"></small>
<div class="row"><button type="button" id="view-link">复制复现链接</button></div>
<details class="sub-details"><summary>视图状态文件</summary><div class="body">
<div class="row"><button type="button" id="view-save">保存视图</button><button type="button" id="view-restore">恢复视图</button><button type="button" id="view-export">导出 JSON</button><button type="button" id="view-import">导入 JSON</button><input type="file" id="view-file" accept="application/json" hidden></div>
<small id="view-note" class="hint">视图状态包含相机、色标、阈值、单位、叠加、剖切、分支显隐、测量、标注、探针记录与导图设置；「复制复现链接」把状态写入链接并复制，打开该链接即可复现此图。</small>
</div></details>
<details class="sub-details"><summary>导出设置与多视角拼图</summary><div class="body"><b>出版级导图</b>
<div class="row"><label>分辨率 <select id="exp-scale"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option></select></label><label>背景 <select id="exp-bg"><option value="white">白</option><option value="transparent">透明</option><option value="current">当前</option></select></label></div>
<div class="row"><label>色标 <select id="exp-cbar"><option value="overlay">叠加</option><option value="none">不含</option><option value="svg">单独 SVG</option></select></label><label>语言 <select id="exp-lang"><option value="zh">中文</option><option value="en">英文</option></select></label><label><input type="checkbox" id="exp-ui"> 含界面标签</label></div>

<div class="sep"></div><b>多视角拼图</b>
<div class="row" id="montage-views" aria-label="拼图视角"></div>
<div class="row"><label>列数 <select id="montage-cols"><option value="2">2</option><option value="3" selected>3</option><option value="4">4</option></select></label><button type="button" id="montage-go">导出拼图 PNG</button></div>
<small id="montage-note" class="hint">按上面的分辨率 / 背景 / 语言设置离屏渲染每一幅，面板标 a/b/c…，右侧一条共用色标。</small>
</div></details>
</div></details>


</div>
</aside>
<div id="view" role="region" aria-label="三维 WSS 视图"><svg id="label-leaders" aria-hidden="true"></svg><div id="labels"></div><button type="button" id="menu-tab" title="打开菜单">菜单</button><img id="snapshot" alt="WSS 壁面视图与当前色标"><div id="warn-banner" role="status" hidden></div><div id="field-seg" role="group" aria-label="壁面着色字段" hidden></div><div id="cbar-unit"></div><div id="cbar"></div><div id="legend-stack"><div class="legend-note" id="legend-note"></div><div id="trust-legend" hidden></div></div><div id="probe-card" hidden></div><div id="tip"><span id="tip-text">悬停读数；壁面颜色为插值显示，统计用预测点云</span><button type="button" id="tip-close" title="关闭提示" aria-label="关闭提示">×</button></div></div>
</main>
<footer id="footer"></footer><div id="tech-pop" class="tech-pop" role="dialog" aria-label="技术信息" hidden></div>
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
// §19.1 human numbers (never scientific notation): colour-bar ticks, readouts and statistic cards.
function fq(x){return window.WssReportCommon.formatValue(x);}
function fqu(x){return fq(x==null||!Number.isFinite(+x)?x:+x*uf());}
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
// A copy: the viewer may add RRT / ECAP descriptors (below) without touching the embedded metadata.
const FIELD_SCHEMA=Object.assign({},schemaFields(META)),FRAME_SCHEMA=schemaFrames(META);
function fieldSupported(id,d){
  // The embedded arrays hold one scalar value per point per field, never a
  // time-by-point tensor.  A familiar field ID alone cannot establish that contract.
  if(!d||typeof d!=='object')return false;
  const scalarPoint=d.kind==='scalar'&&d.components===1&&d.location==='wall'&&
    Array.isArray(d.axis_order)&&d.axis_order.length===1&&d.axis_order[0]==='point'&&
    Array.isArray(d.time_indices)&&d.time_indices.length===1&&Number.isInteger(d.time_indices[0])&&
    FRAME_SCHEMA.some((f,i)=>(f.index??i)===d.time_indices[0]);
  if(id==='wss')return d.id==='wss'&&scalarPoint&&d.units==='Pa'&&d.array_key==='wss_pa'&&!!ARR.pw&&!!ARR.mw;
  // Extra wall scalar fields (e.g. TAWSS / OSI of the three-head release) are embedded under ARR.xf[array_key];
  // RRT / ECAP are computed here from those arrays (deriveInViewer): v0.14 pages declare them (source 'derived')
  // without embedding them, pages older than v0.13 do not declare them at all ('derived_in_viewer').
  return scalarPoint&&typeof d.array_key==='string'&&(!!(ARR.xf&&ARR.xf[d.array_key]&&ARR.xf[d.array_key].m&&ARR.xf[d.array_key].p)||((d.source==='derived_in_viewer'||d.source==='derived')&&!!XF[d.array_key]));
}
let activeField='wss';const XF={};
// v0.13 derived indices, same formula and floors as cycle_fields.derive_indices (TAWSS ≥ 0.01 Pa, 1 − 2·OSI ≥ 0.01).
const DERIVED_DEF={rrt:{label:'相对滞留时间 RRT',key:'rrt_per_pa',thresholds:[5,10,20],f:(t,o)=>1/(Math.max(1-2*o,.01)*t)},
 ecap:{label:'内皮细胞激活势 ECAP',key:'ecap_per_pa',thresholds:[1.4,2.8,4.2],f:(t,o)=>o/t}};
(function deriveInViewer(){
  const T=FIELD_SCHEMA.tawss,O=FIELD_SCHEMA.osi;if(!fieldSupported('tawss',T)||!fieldSupported('osi',O))return;
  const get=d=>({m:dec(ARR.xf[d.array_key].m,Float32Array),p:dec(ARR.xf[d.array_key].p,Float32Array)}),tw=get(T),os=get(O);
  const q99=v=>{const at=.99*(v.length-1),lo=Math.floor(at),hi=Math.ceil(at);return v[lo]+(v[hi]-v[lo])*(at-lo);};   // = CORE.percentile
  const calc=(f,ta,oa)=>{const out=new Float32Array(ta.length);for(let i=0;i<ta.length;i++){const t=ta[i],o=oa[i];out[i]=Number.isFinite(t)&&Number.isFinite(o)?f(Math.max(t,.01),Math.min(Math.max(o,0),.5)):NaN;}return out;};
  for(const [id,def] of Object.entries(DERIVED_DEF)){const have=FIELD_SCHEMA[id];
    // A v0.13 page embeds the derived arrays: they are used as they are.  A v0.14 page declares the field (display
    // p99 / max from summary.json) but leaves the arrays out: recompute them, keep the page's descriptor.
    if(have&&!(have.source==='derived'&&typeof have.array_key==='string'&&!(ARR.xf&&ARR.xf[have.array_key])))continue;
    const p=calc(def.f,tw.p,os.p),m=calc(def.f,tw.m,os.m);
    if(have){XF[have.array_key]={m,p};continue;}
    const v=Array.from(p).filter(Number.isFinite).sort((a,b)=>a-b);
    XF[def.key]={m,p};
    FIELD_SCHEMA[id]={id,label:def.label,units:'1/Pa',location:'wall',kind:'scalar',components:1,array_key:def.key,axis_order:['point'],time_indices:T.time_indices.slice(),
      source:'derived_in_viewer',derived_from:['tawss','osi'],display:{thresholds:def.thresholds.slice(),log_scale:true,p99:v.length?q99(v):null,max:v.length?v[v.length-1]:null,
      cycle:'derived from TAWSS / OSI',threshold_direction:'above'}};}
})();
// recolor() lives inside initViewer(); the viewer registers it here so the field tabs can repaint the wall.
let recolorActive=null;
// Short names for the segmented control, colour bar and labels; the full descriptor label stays in the menu.
const BAND_STEPS=[0,4,6,8,10,12,16,20];   // 色带分段 options (v0.13 adds 10 / 16 / 20)
const FIELD_SHORT={wss:'峰值 WSS',tawss:'TAWSS',osi:'OSI',rrt:'RRT',ecap:'ECAP'};
// OSI / RRT / ECAP count the area above their thresholds; WSS / TAWSS count the low tail first.
function aboveField(id){const d=FIELD_SCHEMA[id],dir=d&&d.display&&d.display.threshold_direction;return dir?dir==='above':['osi','rrt','ecap'].includes(id);}
function isDerived(id){const d=FIELD_SCHEMA[id];return !!(d&&(d.source==='derived'||d.source==='derived_in_viewer'||Array.isArray(d.derived_from)));}
function fieldArrays(id){
  if(id!=='wss'&&!(FIELD_SCHEMA[id]&&fieldSupported(id,FIELD_SCHEMA[id])))id='wss';
  if(id==='wss')return {id:'wss',m:MW,p:PW,units:'Pa',isPa:true,label:'WSS',short:'WSS',p99:PEAK.p99_pa,max:PEAK.max_pa,log:true};
  const d=FIELD_SCHEMA[id],key=d.array_key;if(!XF[key])XF[key]={m:dec(ARR.xf[key].m,Float32Array),p:dec(ARR.xf[key].p,Float32Array)};
  const disp=d.display&&typeof d.display==='object'?d.display:{};
  return {id,m:XF[key].m,p:XF[key].p,units:d.units||'',isPa:d.units==='Pa',label:d.label||id,short:FIELD_SHORT[id]||d.label||id,p99:disp.p99,max:disp.max,log:disp.log_scale!==false,thresholds:Array.isArray(disp.thresholds)?disp.thresholds.map(Number):null};
}
const unitText=u=>u==='1/Pa'?'Pa⁻¹':(u&&u!=='1'?u:'');   // OSI is dimensionless: no "1" after the number
function af(){return fieldArrays(activeField).isPa?uf():1;}
function al(){const F=fieldArrays(activeField);return F.isPa?ul():unitText(F.units);}
function fv(x){const F=fieldArrays(activeField),u=F.isPa?ul():unitText(F.units);return (F.isPa?fqu(x):fq(x))+(u?' '+u:'');}
function fieldText(id,d){return (d.label||id)+(unitText(d.units)?' · '+unitText(d.units):'');}
function supportedFields(){return Object.keys(FIELD_SCHEMA).filter(id=>fieldSupported(id,FIELD_SCHEMA[id]));}
// Menu tabs, the segmented control above the viewport and the F key all switch through here.
function setField(id){if(activeField===id||!supportedFields().includes(id))return false;activeField=id;renderSchemaControls();if(recolorActive)recolorActive();return true;}
function cycleField(){const ids=supportedFields();return ids.length>1&&setField(ids[(ids.indexOf(activeField)+1)%ids.length]);}
function isCycleField(id){const d=FIELD_SCHEMA[id];return id!=='wss'&&!!(d&&d.display&&d.display.cycle);}
function cyclePeriod(){const d=META.cycle&&META.cycle.definition,p=d?Number(d.period_s):NaN;return Number.isFinite(p)&&p>0?+p.toFixed(3):0.8;}
// 「收缩期峰值帧（约 0.21 s）」; the raw label and solver step live in the technical-information dialog.
function humanFrame(f){f=f||{};const raw=String(f.label||''),t=Number(f.time_s);
  return (!raw||/peak|systol|收缩期/i.test(raw)?'收缩期峰值帧':raw)+(f.time_s!=null&&Number.isFinite(t)?'（约 '+t.toFixed(2)+' s）':'');}
function renderSchemaControls(){
  const tabs=el('field-tabs'),select=el('frame-select'),status=el('frame-status'); if(!tabs||!select||!status)return;
  tabs.replaceChildren();
  const supportedIds=supportedFields();
  if(!supportedIds.includes(activeField))activeField='wss';
  for(const [id,d0] of Object.entries(FIELD_SCHEMA)){
    const d=d0&&typeof d0==='object'?d0:{}; const supported=fieldSupported(id,d);
    // A single supported field is a status label, not a fake switch; several supported fields are a real switch
    // of the wall colouring (and of the field card in 统计与口径).
    const switchable=supported&&supportedIds.length>1;
    const item=document.createElement(supported&&!switchable?'span':'button');item.className='field-tab';item.dataset.field=id;item.textContent=fieldText(id,d);
    if(supported){
      if(id===activeField){item.classList.add('active');item.setAttribute('aria-current','true');}
      item.title=switchable?'点击切换壁面着色字段（快捷键 F 循环）':'本报告显示的字段';
      if(switchable){item.type='button';item.onclick=()=>setField(id);}
    }
    else{item.type='button';item.disabled=true;item.title='字段描述或数组布局不受当前报告查看器支持；仅显示元数据';}
    tabs.appendChild(item);
  }
  if(!tabs.children.length){const span=document.createElement('span');span.textContent='未提供字段描述';tabs.appendChild(span);}
  // §19.9 segmented control above the viewport (峰值 WSS / TAWSS / OSI); absent for single-field reports.
  const seg=el('field-seg'),multi=supportedIds.length>1;
  if(seg){seg.replaceChildren();seg.hidden=!multi;
    if(multi)for(const id of supportedIds){const b=document.createElement('button');b.type='button';b.dataset.field=id;b.textContent=FIELD_SHORT[id]||FIELD_SCHEMA[id].label||id;
      b.className=id===activeField?'on':'';b.setAttribute('aria-pressed',String(id===activeField));b.title=fieldText(id,FIELD_SCHEMA[id]||{})+'（F 键循环）';b.onclick=()=>setField(id);seg.appendChild(b);}
    const v=el('view');if(v&&v.classList&&v.classList.toggle)v.classList.toggle('has-seg',multi);}
  // v0.13: with the segmented control above the viewport the menu copy of the switch is redundant.
  const sc=el('schema-controls');if(sc)sc.hidden=multi;
  // Metadata can list more frames, but this exporter embeds only one array per field: navigation stays disabled.
  const cyc=isCycleField(activeField);select.replaceChildren();
  if(cyc){const o=document.createElement('option');o.value='cycle';o.textContent='单周期积分量（'+cyclePeriod()+' s）';select.appendChild(o);}
  else FRAME_SCHEMA.forEach((f,i)=>{const o=document.createElement('option');o.value=String(f.index??i);o.textContent=humanFrame(f);select.appendChild(o);});
  select.disabled=true;
  const supported=fieldSupported('wss',FIELD_SCHEMA.wss);
  const index=supported?FIELD_SCHEMA.wss.time_indices[0]:null;
  select.value=cyc?'cycle':index===null?'':String(index);
  status.textContent=cyc&&isDerived(activeField)?'由预测的 TAWSS 与 OSI 逐点计算的派生指标（'+cyclePeriod()+' s 周期）；误差继承两者，在低 TAWSS 处放大'
    :cyc?'一个心动周期（'+cyclePeriod()+' s）的积分量，直接回归预测，不是逐帧推演'
    :!supported?'当前字段没有可确认的帧映射':FRAME_SCHEMA.length===1?'单帧结果，不提供多帧切换':'时间轴含 '+FRAME_SCHEMA.length+' 帧；本报告只嵌入其中一帧';
}
// ---- arrays (decoded once; the statistics panel needs them even without WebGL) ----
// v0.14 pages store branch ids as uint8 when they fit (ARR.dt[key] === 'u8'); older pages are int32.
const SEG_T=key=>ARR.dt&&ARR.dt[key]==='u8'?Uint8Array:Int32Array;
const MV=dec(ARR.mv,Float32Array),MF=dec(ARR.mf,Uint32Array),MW=dec(ARR.mw,Float32Array),MS=dec(ARR.ms,SEG_T('ms')),MT=ARR.mt?dec(ARR.mt,Uint8Array):null;
const PV=dec(ARR.pv,Float32Array),PW=dec(ARR.pw,Float32Array),PS=dec(ARR.ps,SEG_T('ps')),P_S=dec(ARR.p_s,Float32Array),P_TH=dec(ARR.p_th,Float32Array),P_R=dec(ARR.p_r,Float32Array),P_DJ=dec(ARR.p_dj,Float32Array);
const CV=dec(ARR.cv,Float32Array),CR=dec(ARR.cr,Float32Array),CE=dec(ARR.ce,Uint32Array),CS=dec(ARR.cs,SEG_T('cs'));
const PEAK=META.peak||{},AREA=Number((META.input_check||{}).area_mm2)||0,BRANCH_NAMES=META.branch_names||{};
const PROFILES=META.profiles&&Array.isArray(META.profiles.branches)&&META.profiles.branches.length?META.profiles:null;
const FINDINGS=META.findings&&Array.isArray(META.findings.items)?META.findings.items.filter(x=>x&&typeof x==='object').slice().sort((a,b)=>(Number(a.rank)||99)-(Number(b.rank)||99)):[];
const FRAME=CORE.validFrame(META.frame_transform)?META.frame_transform:null;
const TRUST=META.trust&&typeof META.trust==='object'?META.trust:null;
let thresholdsPa=CORE.validThresholds((META.wss_field_pa||{}).thresholds_pa)||[.4,4,7];
const KIND_CN={high_wss_cluster:'高 WSS 区',low_wss_cluster:'低 WSS 区',max_wss:'全场最大值',max_diameter:'最大直径',min_radius:'最小半径',max_speed:'最大速度',min_pressure:'最低压力',pressure_drop:'压降',low_speed_region:'低速区',
 stagnation_cluster:'滞留区',high_osi_cluster:'高 OSI 区'};
// Unit-aware finding value: Pa follows the unit switch, cm² / mm / dimensionless (OSI "1") are printed as they are.
function findingValue(f){if(!f||f.value==null||!Number.isFinite(+f.value))return '';if(f.units==='Pa')return fqu(f.value)+' '+ul();const u=unitText(f.units);return fq(f.value)+(u?' '+u:'');}
const SEV_CN={attention:'关注',note:'提示',info:'几何'};
// 'info' used to mean geometry only; the §19.2 cycle clusters (and unknown kinds) read 「参考」 instead.
const sevText=f=>f&&f.severity==='info'&&!['max_diameter','min_radius'].includes(f.kind)?'参考':(SEV_CN[f&&f.severity]||(f&&f.severity)||'');
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
 low_speed_region:['低速区','Low-speed region'],stagnation_cluster:['滞留区','Stagnation'],high_osi_cluster:['高 OSI','High OSI'],manual:['人工','Manual']};
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
 const value=findingValue(f);
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
// v0.14 「← 工作台」: only when this page came from our service over http and is not one half of the compare page.
(function(){const a=el('back-to-workbench');if(!a)return;let framed=false;try{framed=window.parent&&window.parent!==window;}catch(_){framed=true;}
  const m=/\/api\/jobs\/([^/]+)\/report$/.exec(String(location.pathname||''));
  if(!/^https?:$/.test(String(location.protocol||''))||framed)return;
  a.href='/'+(m?'#job='+m[1]:'');a.hidden=false;const h=a.parentNode;if(h&&h.classList)h.classList.add('has-back');})();
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
// ---- §19.2 cycle quantities (three-head release): summary.cycle when embedded, otherwise recomputed from the point arrays ----
const CYCLE=META.cycle&&typeof META.cycle==='object'?META.cycle:null;
const CYCLE_THR={tawss:[.4,4,7],osi:[.1,.2,.3],rrt:[5,10,20],ecap:[1.4,2.8,4.2]},CYCLE_KEYS={above:['above_t0','above_t1','above_t2'],other:['low','high','very_high']};
// Reviewer overrides of the extra fields' thresholds (base units); WSS keeps its own thresholdsPa.
const FIELD_THR={};
function defaultThr(id){const src=CYCLE&&CYCLE.fields&&CYCLE.fields[id],d=FIELD_SCHEMA[id],disp=d&&d.display;
  return CORE.validThresholds(src&&src.thresholds)||CORE.validThresholds(disp&&disp.thresholds)||(CYCLE_THR[id]||[.4,4,7]).slice();}
function fieldThr(id){return id==='wss'?thresholdsPa:(FIELD_THR[id]||defaultThr(id));}
const sameThr=(a,b)=>Array.isArray(a)&&Array.isArray(b)&&a.length===3&&a.every((x,k)=>Math.abs(+x-b[k])<=1e-9*Math.max(1,Math.abs(b[k])));
function thrFracs(values,t,above,mask){let n=0,a=0,b=0,c=0;for(let i=0;i<values.length;i++){if(mask&&!mask[i])continue;const x=values[i];if(!Number.isFinite(x))continue;n++;
  if(above){if(x>t[0])a++;}else if(x<t[0])a++;if(x>t[1])b++;if(x>t[2])c++;}
  const f=k=>n?k/n:0;return {n,f0:f(a),f1:f(b),f2:f(c)};}
function cycleStats(id){
  const src=CYCLE&&CYCLE.fields&&CYCLE.fields[id],above=aboveField(id),keys=CYCLE_KEYS[above?'above':'other'],t=fieldThr(id);
  if(src&&Number.isFinite(+src.mean)&&sameThr(src.thresholds,t))return {mean:+src.mean,p99:src.p99,max:src.max,thresholds:t.slice(),frac:src.area_frac||{},keys,per_branch:src.per_branch||null,source:'summary'};
  if(!fieldSupported(id,FIELD_SCHEMA[id]))return null;
  const F=fieldArrays(id),v=Array.from(F.p).filter(Number.isFinite).sort((a,b)=>a-b),n=v.length;if(!n)return null;
  let s=0;for(const x of v)s+=x;const g=thrFracs(F.p,t,above),frac={[keys[0]]:g.f0,[keys[1]]:g.f1,[keys[2]]:g.f2};
  // Per branch (≥ 10 points, like the summary): recomputed so a threshold edit also updates the branch table.
  const per={},ids=Array.from(new Set(Array.from(PS))).sort((a,b)=>a-b);
  for(const sid of ids){const mask=new Uint8Array(PS.length);let k=0;for(let i=0;i<PS.length;i++)if(PS[i]===sid){mask[i]=1;k++;}if(k<10)continue;
    const st=CORE.stats(F.p,CORE.maskIndices(mask)),b=thrFracs(F.p,t,above,mask);per[branchName(sid)]={segment_id:sid,mean:st.mean,p99:st.p99,max:st.max,[above?'frac_above_t0':'frac_low']:b.f0};}
  return {mean:src&&Number.isFinite(+src.mean)?+src.mean:s/n,p99:src&&src.p99!=null?src.p99:CORE.percentile(v,.99),max:src&&src.max!=null?src.max:v[n-1],thresholds:t.slice(),frac,keys,per_branch:per,source:src?'summary+thresholds':'arrays'};}
// Maximum of a field over the prediction points (the yellow marker follows the coloured field).  Peak WSS keeps the
// summary's point (metrics.py); TAWSS / OSI / RRT / ECAP take the first largest finite point value (np.argmax order).
// ``low`` asks for the minimum instead (np.argmin order): TAWSS also marks its lowest point (white marker).
const PEAK_CACHE={},MIN_FIELDS=new Set(['tawss']);
function fieldPeak(id,low){const F=fieldArrays(id);
  if(F.id==='wss'&&!low){const xyz=Array.isArray(PEAK.xyz_mm)&&PEAK.xyz_mm.length===3?PEAK.xyz_mm.map(Number):null;
    return xyz?{id:'wss',value:Number(PEAK.max_pa),xyz,branch:PEAK.branch,s:PEAK.s_from_inlet_mm,dj:PEAK.dist_to_junction_mm,r:PEAK.local_radius_mm}:null;}
  const ck=F.id+(low?':min':''),hit=PEAK_CACHE[ck];if(hit&&hit.arr===F.p)return hit.peak;
  const v=F.p;let best=-1;for(let i=0;i<v.length;i++){const x=v[i];if(Number.isFinite(x)&&(best<0||(low?x<v[best]:x>v[best])))best=i;}
  const peak=best<0||3*best+2>=PV.length?null:{id:F.id,value:v[best],index:best,xyz:[PV[3*best],PV[3*best+1],PV[3*best+2]],
    branch:branchName(PS[best]),s:P_S[best],dj:P_DJ[best],r:P_R[best]};
  PEAK_CACHE[ck]={arr:v,peak};return peak;}
const HAS_STAG=()=>fieldSupported('tawss',FIELD_SCHEMA.tawss)&&fieldSupported('osi',FIELD_SCHEMA.osi);
function stagCriteria(){const c=(CYCLE&&CYCLE.stagnation&&CYCLE.stagnation.criteria)||{};return {t:Number.isFinite(+c.tawss_lt_pa)?+c.tawss_lt_pa:.4,o:Number.isFinite(+c.osi_gt)?+c.osi_gt:.1};}
// 'm' = mesh vertices (display), 'p' = prediction points (statistics).
function stagnationMask(which){if(!HAS_STAG())return null;const T=fieldArrays('tawss')[which],O=fieldArrays('osi')[which],c=stagCriteria(),m=new Uint8Array(T.length);for(let i=0;i<T.length;i++)m[i]=T[i]<c.t&&O[i]>c.o?1:0;return m;}
function stagnationStats(){const st=CYCLE&&CYCLE.stagnation;
  if(st&&Number.isFinite(+st.area_frac)){let main=null;for(const [name,b] of Object.entries(st.per_branch||{}))if(b&&Number.isFinite(+b.area_mm2)&&+b.area_mm2>0&&(!main||+b.area_mm2>main.area_mm2))main={name,area_mm2:+b.area_mm2,frac:+b.frac};
    return {frac:+st.area_frac,area_mm2:Number(st.area_mm2),main};}
  const m=stagnationMask('p');if(!m||!m.length)return null;let k=0;for(const x of m)k+=x;return {frac:k/m.length,area_mm2:k/m.length*AREA,main:null};}
function cycleCard(bar){const ids=supportedFields().filter(isCycleField);if(!ids.length)return '';
  const act=isCycleField(activeField)?activeField:null,st=HAS_STAG()?stagnationStats():null,c=stagCriteria(),per=cyclePeriod();
  const uu=id=>unitText(fieldArrays(id).units),qv=(id,x)=>fieldArrays(id).isPa?fqu(x)+' '+ul():fq(x)+(uu(id)?' '+esc(uu(id)):''),
    thr=(id,x)=>fieldArrays(id).isPa?String(+(+x*uf()).toFixed(3))+' '+ul():String(+(+x).toFixed(3))+(uu(id)?' '+esc(uu(id)):'');
  const stagLine=st?`<b>滞留区</b><span>TAWSS &lt; ${esc(thr('tawss',c.t))} 且 OSI &gt; ${esc(String(c.o))}：${fmt(st.frac*100,0)}%（约 ${fmt(st.area_mm2/100,0)} cm²）${st.main?'，主要位于'+esc(st.main.name):''}</span>`:'';
  const derivedAct=act&&isDerived(act),glossKey=act&&['rrt','ecap'].includes(act)&&GLOSSARY.terms&&GLOSSARY.terms[act]?`<button type="button" class="gloss" data-gloss="${act}">?</button>`:'';
  let h=`<section class="card" id="cycle-card"><h3>${act?esc(fieldArrays(act).short)+' · ':''}${derivedAct?'由 TAWSS / OSI 派生（'+per+' s 周期）':'单周期积分量（'+per+' s）'}${glossKey}</h3>`;
  if(act){const s=cycleStats(act);if(s){const t=s.thresholds,k=s.keys,cm=f=>fmt(f*AREA/100,1)+' cm²',sh=fieldArrays(act).short,up=aboveField(act),pk=fieldPeak(act),
     lo=MIN_FIELDS.has(act)?fieldPeak(act,true):null,
     where=(pk?`<b>最大值位置</b><span>${esc(pk.branch)}，距入口 ${fmt(pk.s,0)} mm；距分叉 ${fmt(pk.dj,0)} mm</span><b>黄色标记</b><span>${esc(sh)} 全场最大值预测点；局部半径 ${fmt(pk.r,1)} mm</span>`:'')
      +(lo?`<b>最小</b><span>${qv(act,lo.value)}</span><b>最小值位置</b><span>${esc(lo.branch)}，距入口 ${fmt(lo.s,0)} mm；距分叉 ${fmt(lo.dj,0)} mm</span><b>白色标记</b><span>${esc(sh)} 全场最小值预测点；局部半径 ${fmt(lo.r,1)} mm</span>`:''),
     lab=up?[sh+' > '+thr(act,t[0]),sh+' > '+thr(act,t[1]),sh+' > '+thr(act,t[2])]:[(act==='tawss'?'低 TAWSS < ':'< ')+thr(act,t[0]),'> '+thr(act,t[1]),'> '+thr(act,t[2])];
     h+=`<div class="big">${qv(act,s.mean)}</div><small>预测点云均值（${esc(fieldArrays(act).label)}）</small><div class="kv"><b>p99</b><span>${qv(act,s.p99)}</span><b>最大</b><span>${qv(act,s.max)}</span>${where}${stagLine}</div>`
      +`<div class="metric-bars">${bar(lab[0],s.frac[k[0]],fmt(s.frac[k[0]]*100,1)+'% · '+cm(s.frac[k[0]]||0),up?'high':'low')}${bar(lab[1],s.frac[k[1]],fmt(s.frac[k[1]]*100,1)+'% · '+cm(s.frac[k[1]]||0),'high')}${bar(lab[2],s.frac[k[2]],fmt(s.frac[k[2]]*100,1)+'% · '+cm(s.frac[k[2]]||0),'very-high')}</div>`;
     const pb=s.per_branch&&typeof s.per_branch==='object'?Object.entries(s.per_branch):[];
     if(pb.length)h+=`<table><thead><tr><th>分支</th><th>均值</th><th>p99</th><th>${up?'&gt; '+esc(thr(act,t[0])):'&lt; '+esc(thr(act,t[0]))}</th></tr></thead><tbody>`+pb.map(([name,b])=>`<tr><td>${esc(name)}</td><td>${qv(act,b.mean)}</td><td>${qv(act,b.p99)}</td><td>${fmt((up?b.frac_above_t0:b.frac_low)*100,0)}%</td></tr>`).join('')+'</tbody></table>';}}
  else{const rows=ids.map(id=>{const s=cycleStats(id);if(!s)return '';const k=s.keys[0],t=s.thresholds[0];
     return `<b>${esc(fieldArrays(id).short)} 均值</b><span>${qv(id,s.mean)} · ${aboveField(id)?'&gt; '+esc(thr(id,t)):'&lt; '+esc(thr(id,t))} 占 ${fmt((s.frac[k]||0)*100,0)}%</span>`;}).join('');
    h+=`<div class="kv">${rows}${stagLine}</div>`;}
  const derivedNote=derivedAct?(act==='rrt'?'RRT = 1 / [(1 − 2·OSI)·TAWSS]':'ECAP = OSI / TAWSS')+'，由预测的 TAWSS 与 OSI 逐点计算（TAWSS 下限 0.01 Pa），未单独对照 CFD 验证；'+(act==='rrt'?'RRT 没有公认阈值，默认值仅作分级显示；':'ECAP > 1.4 Pa⁻¹ 为腹主动脉瘤文献常用的血栓易发参考水平；')+'阈值可在「更多显示设置」修改。':'';
  return h+`<small>${derivedNote||'一个心动周期（'+per+' s）积分量的直接回归预测，不是逐帧推演；'}面积 = 点占比 × 输入壁面面积。</small></section>`;}
// ---- B4 warning banner: geometry outside the release's reference range, population review or a non-good ensemble ----
let BANNER_OFF=false;
const REF_FIELDS={length_mm:'长度',radius_min_mm:'最小半径',radius_median_mm:'中位半径',radius_max_mm:'最大半径',max_diameter_mm:'最大直径',tortuosity:'迂曲度',spacing_mm:'点间距',surface_variation_median:'表面变化度',variation:'表面变化度'};
function referenceLabel(path){const p=String(path||'');if(p.startsWith('cloud.'))return '点云'+(REF_FIELDS[p.slice(6)]||p.slice(6));
  if(p.startsWith('geometry.')){const rest=p.slice(9),at=rest.lastIndexOf('.');if(at>0){const f=rest.slice(at+1);return rest.slice(0,at)+(REF_FIELDS[f]||f);}}return p;}
function bannerInfo(){const ref=META.reference_assessment||{},pop=ref.population||{},q=META.quality||{},msgs=[];let target='';
  const out=(Array.isArray(ref.checks)?ref.checks:[]).filter(c=>c&&c.status==='review');
  if(out.length||ref.status==='review'){const c=out[0],u=c&&c.units&&c.units!=='1'?' '+c.units:'';
    msgs.push('输入几何有 '+(out.length||'部分')+' 项超出发布包参考范围'+(c?'（'+referenceLabel(c.path)+(c.value!=null?' '+fq(c.value)+u:'')+(c.min!=null&&c.max!=null?'，参考 '+fq(c.min)+'–'+fq(c.max)+u:'')+(out.length>1?' 等':'')+'）':''));target='ref-card';}
  if(pop.status==='review'){msgs.push('人群参照需要复核');target=target||'ref-card';}
  if(q.level&&q.level!=='good'){msgs.push('模型集成质量：'+(q.label||q.level));target=target||'quality-card';}
  return msgs.length?{text:'注意：'+msgs.join('；')+'。预测可信度可能下降，请结合详情复核。',target}:null;}
function openStats(target){openCard('menu-stats');if(typeof window.__openMenu==='function')window.__openMenu();
  const t=target&&document.getElementById(target);try{if(t&&t.scrollIntoView)t.scrollIntoView({block:'start'});}catch(_){}}
function renderBanner(){const b=el('warn-banner'),v=el('view'),info=bannerInfo();if(!b)return;const show=!!info&&!BANNER_OFF;
  if(v&&v.classList)v.classList.toggle('has-banner',show);b.hidden=!show;b.replaceChildren();if(!show)return;
  const s=document.createElement('span');s.textContent=info.text;
  const d=document.createElement('button');d.type='button';d.textContent='详情';d.onclick=()=>openStats(info.target);
  const x=document.createElement('button');x.type='button';x.textContent='×';x.title='关闭提示（仅本次打开）';x.setAttribute('aria-label','关闭提示');x.onclick=()=>{BANNER_OFF=true;renderBanner();};
  b.append(s,d,x);}
// ---- bottom-left readout: data readouts always show; the static hint can be closed and stays closed ----
const TIP_HINT='悬停读数；壁面颜色为插值显示，统计用预测点云';
let TIP_OFF=false;try{TIP_OFF=localStorage.getItem('wss-report-tip-off')==='1';}catch(_){}
function setTip(text,hint){el('tip-text').textContent=String(text??'');el('tip').hidden=!!hint&&TIP_OFF;}
el('tip-close').onclick=()=>{TIP_OFF=true;try{localStorage.setItem('wss-report-tip-off','1');}catch(_){}el('tip').hidden=true;};
function renderPanel(){const m=META,p=PEAK,ic=m.input_check||{},cloud=m.cloud||{};let h='';
 const narrative=narrativeLines();
 if(narrative.lines.length)h+=`<section class="card" id="narrative-card"><h3>结论（参考）<button type="button" class="gloss" data-gloss="narrative">?</button></h3>`
  +(narrative.edited?'<div class="flag edited">审阅人已修改</div>':'')+narrative.lines.map(x=>`<p>${esc(x)}</p>`).join('')
  +`<small>${narrative.edited?'审阅人编辑的结论，自动句子保留在 narrative.json。':'由本次结果自动生成的参考描述，只读，不是诊断结论。'}</small></section>`;
 const fields=FIELD_SCHEMA,frames=FRAME_SCHEMA,model=m.model_release||{};
 const pct=v=>Math.max(0,Math.min(100,Number.isFinite(+v)?(+v*100):0));
 const bar=(label,value,detail,klass='')=>`<div class="metric-bar"><span class="metric-label">${esc(label)}</span><span class="metric-track"><span class="metric-fill ${klass}" style="width:${pct(value)}%"></span></span><span>${esc(detail)}</span></div>`;
 // §19.2 the active cycle field's card comes right after the conclusion; with peak WSS active it follows the WSS area card.
 const cyc=cycleCard(bar),cycFirst=isCycleField(activeField);
 if(cycFirst)h+=cyc;
 const fieldRows=Object.entries(fields).map(([id,value])=>{const d=value||{};return `<tr><td>${esc(d.label||id)}</td><td>${esc(unitText(d.units)||(d.units==='1'?'无量纲':'—'))}</td><td>${esc(d.location||'—')}</td><td>${esc(d.kind||'—')}</td></tr>`;}).join('');
 const modelName=model.name||model.release||m.release||'—', modelVersion=model.version||model.model_family||'—';
 const wssEmbedded=fieldSupported('wss',fields.wss),nSwitch=supportedFields().length;
 const viewerNote=nSwitch>1?'壁面着色可在峰值 WSS 与周期量之间切换（视口上方按钮或 F 键）；本卡以下的 WSS 统计固定为收缩期峰值帧。':wssEmbedded&&frames.length===1?'当前查看器显示已嵌入的单帧标量 WSS；其他字段及时间帧保留为元数据。':'字段与时间轴元数据已保留；当前导出未提供可切换的字段数组。';
 h+=`<section class="card"><h3>结果字段与模型</h3><div class="kv"><b>声明字段</b><span>${esc(Object.keys(fields).map(id=>fieldText(id,fields[id]||{})).join('、')||'—')}</span><b>时间轴</b><span>${frames.length===1?'单帧 · '+esc(humanFrame(frames[0])):'多帧 · '+frames.length+' 帧'}</span><b>模型发布</b><span>${esc(modelName)}${modelVersion!=='—'?' · '+esc(modelVersion):''}</span>${model.ensemble_protocol?`<b>集成协议</b><span>${esc(model.ensemble_protocol)}</span>`:''}${Array.isArray(model.weights)?`<b>权重文件</b><span>${model.weights.length} 个</span>`:''}</div><table><thead><tr><th>字段</th><th>单位</th><th>位置</th><th>类型</th></tr></thead><tbody>${fieldRows}</tbody></table><small>${viewerNote}</small></section>`;
 h+=`<section class="card"><h3>收缩期峰值帧 · 空间 p99<button type="button" class="gloss" data-gloss="p99">?</button></h3><div class="big">${fqu(p.p99_pa)} ${ul()}</div><small>预测点云第 99 百分位，不代表时间最大值。</small><div class="kv"><b>全场最大值<button type="button" class="gloss" data-gloss="max">?</button></b><span>${fqu(p.max_pa)} ${ul()}</span><b>最大值位置</b><span>${esc(p.branch)}，距入口 ${fmt(p.s_from_inlet_mm,0)} mm；距分叉 ${fmt(p.dist_to_junction_mm,0)} mm</span><b>黄色标记</b><span>${activeField==='wss'?'全场最大值预测点':'切回峰值 WSS 时标在此处（当前标 '+esc(fieldArrays(activeField).short)+' 最大值）'}；局部半径 ${fmt(p.local_radius_mm,1)} mm</span></div></section>`;
 const ss=m.surface_statistics||{};
 if(ss.p99_pa!=null) h+=`<section class="card"><h3>壁面面积加权参考<button type="button" class="gloss" data-gloss="area_weighted_p99">?</button></h3><div class="kv"><b>面积加权 p99</b><span>${fqu(ss.p99_pa)} ${ul()}</span><b>最高 1% 面积均值</b><span>${fqu(ss.top_area_mean_pa)} ${ul()}</span><b>有效覆盖面积</b><span>${fmt(ss.effective_area_mm2/100,1)} cm² / ${fmt(ss.covered_area_fraction*100,1)}%</span></div><small>基于 Gaussian 插值后的完整三角面；未覆盖或部分覆盖面片不外推。主指标仍是预测点云 p99。</small></section>`;
 const quality=m.quality||{},nModels=Array.isArray(model.models)?model.models.length:Array.isArray(model.weights)?model.weights.length:5;
 if(quality.level){const qclass=quality.level==='good'?'ok':'flag';h+=`<section class="card" id="quality-card"><h3>${nModels===5?'五模型':nModels+' 个模型'}集成质量<button type="button" class="gloss" data-gloss="quality_grade">?</button></h3><div class="flag ${qclass}">${esc(quality.label||quality.level)}</div>${(quality.reasons||[]).map(x=>`<small>${esc(x)}</small>`).join('<br>')}<p><small>逐点 seed 标准差仅写入 quality_audit.json，避免把分散度误读为预测概率。</small></p></section>`;}
 const gate=m.proposal_confidence_gate||{};
 if(gate.status||gate.confirmation_required!=null){const gateText=gate.confirmation_required?'需要人工确认':'已通过自动门控';h+=`<section class="card"><h3>出口命名门控<button type="button" class="gloss" data-gloss="confidence_proxy">?</button></h3><div class="flag ${gate.confirmation_required?'':'ok'}">${esc(gateText)}</div><small>${esc(gate.calibration_status==='validated'?'已绑定独立校准 profile。':'当前几何置信度只是代理值，未作为统计概率使用。')}</small></section>`;}
 const reference=m.reference_assessment||{}, population=reference.population||{};
 if(reference.status||population.status){
   const geometryText=reference.status==='pass'?'在已声明几何参考范围内':reference.status==='review'?'超出已声明几何参考范围，请复核':'未配置几何参考范围';
   const populationText=population.status==='pass'?'已计算同协议 CV3 折外经验分位 '+fmt(population.percentile,1)+'%':population.status==='review'?'人群参照需要复核':'未配置可核验的人群参照';
   h+=`<section class="card" id="ref-card"><h3>参考范围与人群位置</h3><div class="kv"><b>几何参考</b><span>${esc(geometryText)}</span><b>人群参照<button type="button" class="gloss" data-gloss="population_percentile">?</button></b><span>${esc(populationText)}</span></div><small>${esc(reference.note||population.note||'参考范围用于复核，不代表校准概率或临床风险。')}</small>${(reference.reasons||[]).length?'<div class="flag">'+esc(reference.reasons.join('；'))+'</div>':''}${(population.reasons||[]).length?'<div class="flag">'+esc(population.reasons.join('；'))+'</div>':''}</section>`;
 }
 const w=CORE.thresholdAreas(PW,thresholdsPa,AREA),t=thresholdsPa,stat=m.wss_field_pa||{};
 h+=`<section class="card"><h3>点数占比与估计面积<button type="button" class="gloss" data-gloss="area_fraction">?</button></h3><div class="metric-bars">${bar('低 WSS < '+fu(t[0],1)+' '+ul(),w.frac_low,fmt(w.frac_low*100,1)+'% · '+fmt(w.area_low_mm2/100,1)+' cm²','low')}${bar('高 WSS > '+fu(t[1],1)+' '+ul(),w.frac_high,fmt(w.frac_high*100,1)+'% · '+fmt(w.area_high_mm2/100,1)+' cm²','high')}${bar('极高 > '+fu(t[2],1)+' '+ul(),w.frac_very_high,fmt(w.frac_very_high*100,1)+'% · '+fmt(w.area_very_high_mm2/100,1)+' cm²','very-high')}</div><div class="kv"><b>均值 / 中位</b><span>${fqu(stat.mean)} / ${fqu(stat.median)} ${ul()}</span></div><small>估计面积 = 点数占比 × 输入壁面总面积；未按逐面片面积加权。阈值可在「显示」菜单修改，此处按当前阈值即时重算。</small></section>`;
 if(!cycFirst)h+=cyc;
 const branchEntries=Object.entries(m.per_branch||{}),branchMax=Math.max(...branchEntries.map(([,b])=>Number(b.wss_p99_pa)||0),.01);
 h+='<section class="card branch-card"><h3>分支统计 · 峰值 WSS · 预测点云（'+esc(ul())+'）</h3><div class="branch-chart metric-bars">'+branchEntries.map(([name,b])=>bar(name,(Number(b.wss_p99_pa)||0)/branchMax,fqu(b.wss_p99_pa)+' '+ul())).join('')+'</div><table><thead><tr><th>分支</th><th>p99</th><th>均值</th><th>最大</th><th>低%</th><th>高%</th><th>估计<br>cm²</th></tr></thead><tbody>';
 for(const [name,b] of branchEntries){let low=b.frac_low,high=b.frac_high;if(Number.isInteger(b.segment_id)){const mask=new Uint8Array(PS.length);for(let i=0;i<PS.length;i++)mask[i]=PS[i]===b.segment_id?1:0;const bw=CORE.thresholdAreas(PW,thresholdsPa,0,mask);if(bw.n){low=bw.frac_low;high=bw.frac_high;}}h+=`<tr><td>${esc(name)}</td><td>${fqu(b.wss_p99_pa)}</td><td>${fqu(b.wss_mean_pa)}</td><td>${fqu(b.wss_max_pa)}</td><td>${fmt(low*100,0)}</td><td>${fmt(high*100,0)}</td><td>${fmt(b.area_mm2/100,1)}</td></tr>`;}
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
 const timingNames={ingest:'输入检查',centerline:'中心线',smooth_resample:'平滑与重采样',features:'构建特征',inference_5_models:'模型预测',metrics_and_export:'统计与导出',metrics_and_interpolation:'统计与插值',export:'导出',total:'累计计算时间',compute_total:'累计计算时间',queue_wait:'排队',confirmation_wait:'等待人工确认',wall_total:'任务历时',finalization:'记录回填',precompute:'后台几何预计算（确认出口期间）',morphology:'形态测量'};
 // v0.14: stages served from the background geometry precompute (summary.geometry_cache.reused) take ≈ 0 s here.
 const cacheStage={mesh:'smooth_resample',resample:'smooth_resample',pointgeom:'features',morph:'morphology',morphvol:'morphology'},gc=m.geometry_cache&&typeof m.geometry_cache==='object'?m.geometry_cache:{};
 const cached=new Set((Array.isArray(gc.reused)?gc.reused:[]).map(k=>cacheStage[k]).filter(Boolean));
 for(const [k,v] of Object.entries(m.timing_s||{}))h+=`<tr><td>${esc(timingNames[k]||k)}</td><td>${fmt(v,2)} s${cached.has(k)?' · 已预计算':''}</td></tr>`;
 h+='</table><p><small>计算时间、排队和人工确认分别记录；导出包含首次完整 HTML 写盘，最终记录回填有少量额外开销。</small></p>';
 h+=`<small>发布包 ${esc(m.release)} · 输入 SHA256 ${esc(m.input_sha256)} · ${esc(m.created_at)}</small>`;
 h+='<details><summary>完整参数与版本信息</summary><pre>'+esc(JSON.stringify({schema_version:m.schema_version,model_release:m.model_release,time_axis:m.time_axis,fields:m.fields,release_hash:m.release_hash||m.release_sha256,run_parameters:m.run_parameters||m.parameters,mapping:m.mapping,frame_transform:m.frame_transform,statistics_protocol:m.statistics_protocol,interpolation:m.interpolation,feature_contract:m.feature_contract,exports:m.exports||m.export_status},null,2))+'</pre></details></div></details>';
 h+=`<div id="print-note">病例 ${esc(m.case_id)} · 发布包 ${esc(m.release)} · ${esc(COMMON.localTime(m.created_at))}<br>收缩期峰值帧的模型预测${nSwitch>1?'（另含单周期 TAWSS / OSI）':''}；统计来自预测点云，壁面显示使用 Gaussian 插值。黄色标记为${activeField==='wss'?'':esc(fieldArrays(activeField).short)+' 的'}全场最大值${MIN_FIELDS.has(activeField)?'，白色标记为全场最小值':''}。</div>`;
 // Every table scrolls sideways in its own box instead of widening the card (v0.13.1).
 el('panel').innerHTML=h.replace(/<table/g,'<div class="tscroll"><table').replace(/<\/table>/g,'</table></div>');
 renderQuickStats();
}
// 查看 tab one-liner under the display controls: the coloured field's headline numbers, thresholds as currently set.
function renderQuickStats(){const box=el('quick-stats');if(!box)return;const id=activeField,F=fieldArrays(id),pct=f=>fmt((f||0)*100,1)+'%';
 const tt=v=>String(+(+v*af()).toPrecision(4))+(al()?' '+al():'');let parts=[];
 if(id==='wss'){const t=thresholdsPa,w=CORE.thresholdAreas(PW,t,AREA);parts=['p99 '+fqu(PEAK.p99_pa)+' '+ul(),'&gt; '+esc(tt(t[1]))+' 占 '+pct(w.frac_high),'&gt; '+esc(tt(t[2]))+' 占 '+pct(w.frac_very_high)];}
 else{const st=cycleStats(id);if(st){const k=st.keys,t=st.thresholds;parts=['均值 '+esc(fv(st.mean)),'p99 '+esc(fv(st.p99)),(aboveField(id)?'&gt; ':'&lt; ')+esc(tt(t[0]))+' 占 '+pct(st.frac[k[0]])];}}
 box.innerHTML='<b>'+esc(F.short)+'</b>'+parts.map(x=>'<span>'+x+'</span>').join('')+'<button type="button" class="linkish" id="quick-stats-more">完整统计 ▸</button>';
 const more=el('quick-stats-more');if(more)more.onclick=()=>openStats('panel');}
// §19.9 footer = review state · short release name · local creation time; identities and hashes live in 「技术信息」.
function releaseShort(){const mr=META.model_release||{},id=String(mr.registry_id||mr.name||META.release||'').trim();return {id,short:id.replace(/_\d+seed_\d{6,8}$/,'')||'—'};}
// A naive created_at (written in the service process' zone) borrows the offset of an ISO stamp of the same run
// (the outlet confirmation), the rule the volume report uses; otherwise it is shown as written.
function createdIso(){const s=String(META.created_at||'').trim();if(!s||/[zZ]$|[+-]\d{2}:?\d{2}$/.test(s))return s;
  const hist=((META.audit||{}).mapping_history)||[],hint=(Array.isArray(hist)?hist:[]).map(h=>h&&h.at).find(x=>/([+-]\d{2}:\d{2}|Z)$/.test(String(x||'')));
  const m=String(hint||'').match(/([+-]\d{2}:\d{2}|Z)$/);return m?s.replace(' ','T')+m[1]:s;}
function renderFooter(){const r=META.review||null,rs=releaseShort(),status=r?({reviewed:'已审阅',unreviewed:'待审阅',reopened:'已重新打开'}[r.status]||r.status):'待审阅';
 el('footer').innerHTML=`<span${r&&r.note?` title="${esc(r.note)}"`:''}>审阅<button type="button" class="gloss" data-gloss="review_status">?</button>：<b>${esc(status)}</b>${r&&r.by?' · '+esc(r.by):''}${r&&r.at?' · '+esc(COMMON.localTime(r.at)):''}</span><span>发布包<button type="button" class="gloss" data-gloss="release">?</button> <b title="${esc(rs.id)}">${esc(rs.short)}</b></span><span title="${esc(COMMON.localTime(createdIso(),{seconds:true}))}">生成 ${esc(COMMON.localTime(createdIso())||'—')}</span><button type="button" id="tech-btn" aria-haspopup="dialog" title="发布包、特征合同、run_identity、哈希与重建时间">技术信息</button>`;
 el('tech-btn').onclick=toggleTech;}
function renderTech(){const m=META,fc=m.feature_contract||{},mr=m.model_release||{},au=m.audit||{},mf=m.model_frame||{},f0=FRAME_SCHEMA[0]||{},cd=CYCLE&&CYCLE.definition;
 const row=(k,v,gloss)=>v==null||v===''?'':`<b>${esc(k)}${gloss||''}</b><span><code>${esc(v)}</code></span>`;
 const iso=x=>x?COMMON.localTime(x,{seconds:true})+'（'+x+'）':null,nModels=Array.isArray(mr.models)?mr.models.length:Array.isArray(mr.weights)?mr.weights.length:null;
 el('tech-pop').innerHTML='<div class="tech-head"><span>技术信息</span><button type="button" id="tech-close" aria-label="关闭技术信息">×</button></div><div class="kv">'
  +row('发布包',mr.registry_id||mr.name||m.release,'<button type="button" class="gloss" data-gloss="release">?</button>')+row('发布包哈希',m.release_hash||m.release_sha256)+row('模型族',[mr.model_family||mr.family,nModels!=null?nModels+' 个模型':null].filter(Boolean).join(' · '))
  +row('特征合同',[fc.version,fc.source_hash].filter(Boolean).join(' · '),'<button type="button" class="gloss" data-gloss="feature_contract">?</button>')+row('run_identity',m.run_identity,'<button type="button" class="gloss" data-gloss="run_identity">?</button>')+row('输入 SHA256',m.input_sha256)
  +row('模型帧',[f0.label||mf.label,f0.step!=null?'step '+f0.step:mf.step!=null?'step '+mf.step:null,f0.time_s!=null?f0.time_s+' s':null].filter(Boolean).join(' · '))
  +row('周期定义',cd?[cd.frames,cd.period_s!=null?'周期 '+cd.period_s+' s':null].filter(Boolean).join(' · '):null)
  +row('生成时间',m.created_at?COMMON.localTime(createdIso(),{seconds:true})+'（'+m.created_at+'）':null)+row('报告重建',iso(au.report_rebuilt_at))+row('摘要版本',m.schema_version)
  +'</div><small class="hint">标识与哈希可全选复制；完整参数见「统计与口径 → 查看计算过程」。</small>';
 el('tech-close').onclick=()=>{el('tech-pop').hidden=true;};}
function toggleTech(){const p=el('tech-pop');if(p.hidden){renderTech();p.hidden=false;}else p.hidden=true;}
if(typeof document.addEventListener==='function'){
 document.addEventListener('keydown',ev=>{if(ev&&ev.key==='Escape')el('tech-pop').hidden=true;});
 document.addEventListener('click',ev=>{const p=el('tech-pop'),t=ev&&ev.target;if(p.hidden||!t)return;if((p.contains&&p.contains(t))||(t.closest&&t.closest('#tech-btn')))return;p.hidden=true;});}
function renderFindingsList(active){renderReviewCount();const box=el('findings-list');box.replaceChildren();LAST_ACTIVE=active;
 if(!FINDINGS.length){el('findings-note').textContent='本报告未包含发现列表（旧版本结果或分析层未运行）。';return;}
 // Rejected items stay visible but dimmed and sink to the end; the stored order (= index) never changes.
 const order=FINDINGS.map((f,i)=>i).sort((a,b)=>(reviewOf(FINDINGS[a]).decision==='rejected'?1:0)-(reviewOf(FINDINGS[b]).decision==='rejected'?1:0)||a-b);
 for(const i of order){const f=FINDINGS[i],rv=reviewOf(f),manual=f.kind==='manual';
  const wrap=document.createElement('div');wrap.className='finding-wrap'+(rv.decision==='rejected'?' rejected':'');
  const b=document.createElement('button');b.type='button';b.className='finding'+(i===active?' on':'');const value=(f.kind==='high_osi_cluster'?'最大 OSI ':'')+findingValue(f);
  // Cycle clusters (§19.2) carry their mean TAWSS / OSI; the stagnation value already is the area.
  const extra=[f.area_mm2!=null&&f.units!=='cm²'?fmt(f.area_mm2/100,1)+' cm²':'',Number.isFinite(+f.tawss_mean_pa)&&f.tawss_mean_pa!==null?'TAWSS 均值 '+fqu(f.tawss_mean_pa)+' '+ul():'',f.kind==='stagnation_cluster'&&Number.isFinite(+f.osi_mean)&&f.osi_mean!==null?'OSI 均值 '+fq(f.osi_mean):''].filter(Boolean);
  b.innerHTML=`<span class="rank">${esc(manual?'人':(f.rank??i+1))}</span><span><strong>${esc(f.label||KIND_CN[f.kind]||f.kind||'发现')}</strong><small class="sub">${esc(f.branch||'')}${findingValue(f)?' · '+esc(value):''}${extra.map(x=>' · '+esc(x)).join('')}</small></span><span class="badge ${esc(manual?'info':severityOf(f))}">${esc(manual?'人工':sevText(f))}</span>`;
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
// The three threshold inputs edit the coloured field: WSS → thresholdsPa, TAWSS / OSI / RRT / ECAP → FIELD_THR.
function thrScale(id){return fieldArrays(id).isPa?uf():1;}
function initThresholdInputs(){const id=activeField,t=fieldThr(id),k0=thrScale(id),F=fieldArrays(id);
 [0,1,2].forEach(k=>{el('thr'+k).value=String(+(t[k]*k0).toFixed(3));el('thr'+k).step=String(id==='osi'?.05:k===0&&F.isPa?.1:.5);});
 el('thr-unit').textContent=F.isPa?ul():unitText(F.units);el('fixed-unit').textContent=al();
 const lab=el('thr-label-text');if(lab)lab.textContent=id==='wss'?'阈值':F.short+' 阈值';
 const hint=el('thr-hint');if(hint)hint.textContent=(aboveField(id)?'三级阈值（高于即计入）':'低 / 高 / 极高阈值')+'；跟随当前着色字段，改动即时重算面积占比、等值线与色标刻线，不写入 summary.json。';}
renderPanel();renderSchemaControls();renderFooter();renderBanner();setTip(TIP_HINT,true);seedReview();renderFindingsList(-1);initBranchOptions();initThresholdInputs();
el('subtitle').textContent=META.case_id+' · '+releaseShort().short+(supportedFields().length>1?' · '+supportedFields().map(id=>id==='wss'?'WSS':(FIELD_SHORT[id]||id)).join(' + '):'');el('subtitle').title=releaseShort().id;
el('stag-row').hidden=!HAS_STAG();
el('print-report').onclick=()=>window.print();
// The 沿程 card also carries the §17.3 maximum-diameter row, so morphology alone is enough to keep it.
if(!PROFILES&&!AMAX){el('menu-profiles').hidden=true;}
if(!PROFILES)for(const id of ['prof-wss','prof-r','prof-svg','prof-clear'])el(id).hidden=true;
if(!FINDINGS.length&&!META.findings){el('menu-findings').hidden=true;}
// 审阅 tab badge: open (not rejected) automatic findings.
function renderReviewCount(){const b=el('mtab-review-count');if(!b)return;const n=FINDINGS.filter(f=>f&&f.severity==='attention'&&!['confirmed','rejected'].includes(reviewOf(f).decision)).length;
  b.textContent=n?String(n):'';b.hidden=!n;b.title=n?'待判定的「关注」级发现 '+n+' 条':'';}
if(!TRUST||!MT){el('trust').disabled=true;el('trust-hint').textContent='（本报告未含可信区域数组）';}
const MENUS=['menu-display','menu-findings','menu-profiles','menu-region','menu-measure','menu-annot','menu-presets','menu-view','menu-stats'];
// v0.13 four tabs (查看 / 测量 / 审阅 / 导出); inside a tab one card is open at a time, the tabs keep their own open card.
// v0.14: 预设 is a way of looking at the case, so it sits under 查看 (it was under 导出).
const PANES={view:['menu-display','menu-stats','menu-presets'],measure:['menu-measure','menu-profiles','menu-region'],review:['menu-findings','menu-annot'],export:['menu-view']};
const paneOf=id=>Object.keys(PANES).find(k=>PANES[k].includes(id))||'view';
let activePane='view';
function showPane(name,{openFirst=true}={}){if(!PANES[name])return false;activePane=name;
  for(const k of Object.keys(PANES)){const on=k===name,tab=el('mtab-'+k),pane=el('pane-'+k);pane.hidden=!on;tab.classList.toggle('on',on);tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;}
  if(openFirst&&!PANES[name].some(id=>el(id).open&&!el(id).hidden)){const first=PANES[name].find(id=>!el(id).hidden);if(first)el(first).open=true;}
  if(name==='measure'&&el('menu-profiles').open)window.__drawProfiles&&window.__drawProfiles();
  if(name==='view'&&el('menu-stats').open)window.__drawUnroll&&window.__drawUnroll();
  return true;}
// Programmatic opens (a measurement mode, a sphere pick, 「详情」, view state) switch to the card's tab first.
function openCard(id){showPane(paneOf(id),{openFirst:false});el(id).open=true;}
// A tab with no visible card (e.g. no WebGL, no findings and no annotations) is hidden.
function syncTabs(){for(const k of Object.keys(PANES))el('mtab-'+k).hidden=PANES[k].every(id=>el(id).hidden);if(el('mtab-'+activePane).hidden)showPane('view');}
for(const k of Object.keys(PANES)){const tab=el('mtab-'+k);tab.onclick=()=>showPane(k);
  tab.addEventListener('keydown',ev=>{const keys=Object.keys(PANES).filter(x=>!el('mtab-'+x).hidden),i=keys.indexOf(k);let j=-1;
    if(ev.key==='ArrowRight')j=(i+1)%keys.length;else if(ev.key==='ArrowLeft')j=(i-1+keys.length)%keys.length;else return;
    ev.preventDefault();showPane(keys[j]);try{el('mtab-'+keys[j]).focus();}catch(_){}});}
for(const id of MENUS){const d=el(id);d.addEventListener('toggle',()=>{if(d.open){const pane=paneOf(id);if(pane!==activePane)showPane(pane,{openFirst:false});for(const o of PANES[pane])if(o!==id)el(o).open=false;}
  if(id==='menu-stats'&&d.open)window.__drawUnroll&&window.__drawUnroll();if(id==='menu-profiles'&&d.open)window.__drawProfiles&&window.__drawProfiles();});}
showPane('view');syncTabs();
function initViewer(){
 // v0.14 (F2): colour tables come from WssReportCommon.PALETTES (one definition for every viewer and export).
 const CMAPS=COMMON.colormapTables();
 // §19.9: rainbow is the default again (user preference 2026-09-20); saved preferences still win.
 let cmapName='rainbow',bands=0,logScale=false,opacity=1,clipT=1,showContours=false,showTrust=false,showStag=false;
 try{const saved=localStorage.getItem('wss-report-cmap');if(saved&&CMAPS[saved])cmapName=saved;}catch(_){}
 function cmapRaw(t){return COMMON.paletteRGB(cmapName,t);}
 function cmap(t){return cmapRaw(CORE.quantize(t,bands));}
 function vir(t){return COMMON.paletteRGB('viridis',t);}
 const view=el('view'),renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.localClippingEnabled=true;renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','三维 WSS 壁面视图，可旋转、缩放并读取壁面数值');view.appendChild(renderer.domElement);
 renderer.domElement.addEventListener('webglcontextlost',ev=>{ev.preventDefault();setTip('3D 显示已中断，请刷新报告；统计表仍可查看。');});
 const scene=new THREE.Scene();scene.background=new THREE.Color(0xeef2f7);const camera=new THREE.PerspectiveCamera(35,1,.1,5000);
 // OrbitControls caches camera.up as its azimuth axis when constructed; the anatomical views use up = FRAME ez, so the
 // controls are rebuilt whenever the camera's up changes (otherwise dragging would tumble about the old world-y axis).
 // v0.14 (F5) render on demand: a frame is drawn when something asks for it (controls change, damping still settling,
 // a tween, resize, field / overlay / label changes, any UI input) instead of a continuous requestAnimationFrame loop.
 // Every export path renders once itself before reading pixels (capture(), COMMON.renderOffscreen).
 let RAF=0;
 function requestRender(){if(RAF||typeof requestAnimationFrame!=='function')return;RAF=requestAnimationFrame(()=>{RAF=0;frame();})||0;}
 let controls=null,controlsUp=null;
 function ensureControls(){const u=camera.up.toArray();if(controls&&controlsUp&&u.every((v,k)=>Math.abs(v-controlsUp[k])<1e-6))return;
  const t=controls?controls.target.toArray():null;if(controls&&typeof controls.dispose==='function')try{controls.dispose();}catch(_){}
  controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;if(t)controls.target.set(t[0],t[1],t[2]);controlsUp=u;
  if(typeof controls.addEventListener==='function'){controls.addEventListener('change',requestRender);controls.addEventListener('start',requestRender);}}
 ensureControls();
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
 // The marker and its chip follow the coloured field; a field without a finite point value hides it.
 const trough=new THREE.Mesh(new THREE.SphereGeometry(1.6,12,8),new THREE.MeshBasicMaterial({color:0xffffff,clippingPlanes:[clipPlane]}));trough.visible=false;scene.add(trough);
 const troughLabel=document.createElement('div');troughLabel.className='trough';el('labels').appendChild(troughLabel);labels.push({obj:trough,div:troughLabel,kind:'trough'});
 let peakOK=true,troughOK=false;
 function placePeak(){const pk=activeField==='wss'?null:fieldPeak(activeField);peakOK=activeField==='wss'||!!pk;
  if(activeField==='wss'){peak.position.set(...peakXYZ);peakLabel.textContent='全场最大值 '+fqu(PEAK.max_pa)+' '+ul();}
  else if(pk){peak.position.set(pk.xyz[0],pk.xyz[1],pk.xyz[2]);peakLabel.textContent=fieldArrays(activeField).short+' 最大值 '+fv(pk.value);}
  else peakLabel.textContent='';
  const lo=MIN_FIELDS.has(activeField)?fieldPeak(activeField,true):null;troughOK=!!lo;
  if(lo){trough.position.set(lo.xyz[0],lo.xyz[1],lo.xyz[2]);troughLabel.textContent=fieldArrays(activeField).short+' 最小值 '+fv(lo.value);}else troughLabel.textContent='';
  peak.visible=viewMode==='wss'&&el('showpeak').checked&&peakOK;trough.visible=viewMode==='wss'&&el('showpeak').checked&&troughOK;}
 const marker=new THREE.Mesh(new THREE.SphereGeometry(1,12,8),new THREE.MeshBasicMaterial({color:0x00c2d8,transparent:true,opacity:.75,depthTest:false}));marker.visible=false;marker.renderOrder=3;scene.add(marker);const markerLabel=document.createElement('div');el('labels').appendChild(markerLabel);labels.push({obj:marker,div:markerLabel,kind:'marker'});
 let contourObj=null;
 // v0.14 (F1): a fixed range may be scoped to one field (fixedField) and marked as set by the compare page
 // (fixedShared); another field then keeps its own p99, and a shared range is never persisted as the user's choice.
 let fixedField=null,fixedShared=false,userScale=null,ANNOUNCE=false,announceTimer=null;
 let viewMode='wss',feat='wss',branchSel=-1,fixedMax=10,scaleMode='case',vmax=Math.max(Number(PEAK.p99_pa)||0,.01),highlightPct=1,highlightThreshold=Number(PEAK.p99_pa)||0,highlightEnabled=false,topIndices=[],top=null;
 let hl={vertexMask:null,pointIndices:null,label:''},hlPts=null,activeFinding=-1,NEAR=null,VS=null;
 function near(){if(!NEAR){NEAR=CORE.nearestIndex(PV,MV,Math.max(0.5,(Number((META.cloud||{}).spacing_mm)||0.5)*3));VS=new Float32Array(MV.length/3);for(let v=0;v<VS.length;v++)VS[v]=NEAR[v]>=0?P_S[NEAR[v]]:NaN;}return NEAR;}
 // ---- C10 branch visibility (display only; every statistic keeps all branches) ----
 const MESH_SEGS=Array.from(new Set(Array.from(MS))).sort((a,b)=>a-b),branchBoxes=[];
 function rebuildBranchFilter(){requestRender();
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
 let SECTION=null;const SECTION_COLOR=0x111111;   // §25 the locked probe's section (drawn by redrawOverlays)
 let probeSel=-1,lastHover=null;   // C11 locked probe (prediction-point index) and the last hovered wall point
 function makeLabel(cls,text,obj,kind,priority){const div=document.createElement('div');div.className=cls;div.textContent=text;el('labels').appendChild(div);labels.push({obj,div,kind,priority});return div;}
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
 function redrawOverlays(){requestRender();
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
  if(SECTION&&SECTION.found)sectionMarks(SECTION);
  renderAutoLabels();}
 // §25 the probe's section, legible on any colour: the outline as a black tube with a white core (open chains stay
 // open), a black / white target at the centreline point the means belong to, and a line from the probe to it.
 function sectionMarks(sec){const ring=(sec.polygon_world||[]).filter(q=>Array.isArray(q)&&q.length===3);
  const r=Math.max(.15,Math.min(size*.003,.15*(Number(sec.radius_mm)||Infinity)));
  if(ring.length>2&&THREE.TubeGeometry&&THREE.CatmullRomCurve3){const curve=new THREE.CatmullRomCurve3(ring.map(q=>new THREE.Vector3(q[0],q[1],q[2])),!sec.open);
   for(const [rr,color,order] of [[r,SECTION_COLOR,5],[r*.45,0xffffff,6]]){const m=new THREE.Mesh(new THREE.TubeGeometry(curve,Math.min(600,2*ring.length),rr,6,!sec.open),new THREE.MeshBasicMaterial({color,depthTest:false}));m.renderOrder=order;overlay.add(m);}}
  else if(ring.length>2)(sec.open?polyline:loopLine)(ring,SECTION_COLOR);
  const R=Math.max(.5,Math.min(size*.008,.4*(Number(sec.radius_mm)||Infinity)));sphereAt(sec.origin,SECTION_COLOR,R).renderOrder=5;sphereAt(sec.origin,0xffffff,R*.5).renderOrder=6;
  if(Array.isArray(sec.pick))polyline([sec.pick,sec.origin],SECTION_COLOR);}
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
 // ---- §21.4 label de-overlap: every layout participant is placed by WssReportCommon.declutterLabels ----
 // attention finding > note > info finding > maximum diameter > branch name; the peak / highlight markers take
 // part too (they sit on the very spots the findings point at) but are never pushed aside by a finding.
 const LABEL_PRIORITY={marker:60,peak:50,trough:50,attention:40,note:30,info:20,maxd:10,blabel:0};
 const LAYOUT_KINDS=new Set(['flabel','blabel','maxd','peak','trough','marker']);
 let LBL_KEY='',LBL_RUNS=0,LBL_LAST=[];
 const requestLabelLayout=()=>{LBL_KEY='';requestRender();};   // picked up by the next frame (frame() compares LBL_KEY)
 // Compact layout or a narrow viewport: a label with no free spot is hidden (the finding stays in the list).
 function labelOverflow(){return !!(document.body&&document.body.classList&&document.body.classList.contains('compact'))||(Number(view.clientWidth)||0)<900;}
 function estimateSize(text,px){let w=0;for(const ch of String(text||''))w+=/[\u2e80-\uffff]/.test(ch)?px:px*.58;return {w:Math.ceil(w+px*1.2),h:Math.round(px*1.6)};}
 function liveSize(it){if(it.size&&it.size.real)return it.size;const d=it.div,w=Number(d.offsetWidth)||0,h=Number(d.offsetHeight)||0;
  it.size=w>0&&h>0?{w,h,real:true}:estimateSize(d.textContent,12);return it.size;}
 const labelPriority=it=>Number.isFinite(it.priority)?it.priority:(LABEL_PRIORITY[it.kind]??0);
 // Leader from the edge of the moved label (centre cx, cy; w × h) to its anchor; null when the label covers the anchor.
 function leaderSegment(cx,cy,w,h,ax,ay){const dx=ax-cx,dy=ay-cy,t=Math.min(Math.abs(dx)>1e-9?(w/2)/Math.abs(dx):Infinity,Math.abs(dy)>1e-9?(h/2)/Math.abs(dy):Infinity);
  return t>=1?null:[cx+dx*t,cy+dy*t,ax,ay];}
 const declutterOpts=(W,H,k)=>({padding:3*k,maxShift:140*k,step:12*k,bounds:{width:W,height:H},hideOverflow:labelOverflow()});
 // Live layout.  #labels chips use transform translate(-50%,-120%): the chip centre sits 0.7 × height above its point.
 function layoutLiveLabels(){LBL_RUNS++;const W=Number(view.clientWidth)||0,H=Number(view.clientHeight)||0,items=[],who=[];
  for(const it of labels){if(!LAYOUT_KINDS.has(it.kind))continue;const d=it.div,v=it.obj.position.clone().project(camera);
   const show=it.obj.visible&&clipPlane.distanceToPoint(it.obj.position)>=0&&v.z>=-1&&v.z<=1&&String(d.textContent||'')!=='';
   if(!show){d.style.display='none';continue;}
   d.style.display='';const s=liveSize(it),ax=(v.x+1)/2*W,ay=(1-v.y)/2*H;
   items.push({x:ax,y:ay-.7*s.h,w:s.w,h:s.h,priority:labelPriority(it)});who.push({it,ax,ay,s});}
  const res=COMMON.declutterLabels(items,declutterOpts(W,H,1)),lines=[];LBL_LAST=[];
  res.forEach((r,j)=>{const {it,ax,ay,s}=who[j],d=it.div,moved=!!r.moved&&!r.hidden;
   d.style.display=r.hidden?'none':'';d.style.left=r.x+'px';d.style.top=(r.y+.7*s.h)+'px';if(d.classList)d.classList.toggle('moved',moved);
   const seg=moved?leaderSegment(r.x,r.y,s.w,s.h,ax,ay):null;if(seg)lines.push(seg);
   LBL_LAST.push({kind:it.kind,text:String(d.textContent||''),x:r.x,y:r.y,w:s.w,h:s.h,ax,ay,moved:!!r.moved,hidden:!!r.hidden,leader:!!seg});});
  const f=x=>(+x).toFixed(1);
  el('label-leaders').innerHTML=lines.map(l=>`<line x1="${f(l[0])}" y1="${f(l[1])}" x2="${f(l[2])}" y2="${f(l[3])}" stroke="#2b3a4a" stroke-opacity=".55" stroke-width="1"/><circle cx="${f(l[2])}" cy="${f(l[3])}" r="2" fill="#2b3a4a" fill-opacity=".55"/>`).join('');
  return LBL_LAST;}
 // Camera / size / visibility signature: the layout reruns (inside the animation frame) only when it changes.
 function labelKey(W,H){const parts=[camera.position,controls.target,camera.up].map(v=>v.toArray().map(x=>(+x).toFixed(2)).join(','));
  parts.push(W+'x'+H,String(clipPlane.constant),labelOverflow()?'o':'-',marker.position.toArray().map(x=>(+x).toFixed(2)).join(','));
  for(const it of labels)if(LAYOUT_KINDS.has(it.kind))parts.push((it.obj.visible?'1':'0')+String(it.div.textContent||''));
  return parts.join('|');}
 // Export / montage / one-page figures: the same layout recomputed at the target resolution (k = pixel scale).
 function exportLabelLayout(W,H,k,measure){const items=[],who=[];
  for(const c of AUTO_CHIPS){const xyz=projectTo(c.p,W,H);if(!(xyz[2]>=-1&&xyz[2]<=1))continue;const s=measure(c.text);
   items.push({x:xyz[0],y:xyz[1]-s.h/2-6*k,w:s.w,h:s.h,priority:Number.isFinite(c.priority)?c.priority:0});who.push({c,ax:xyz[0],ay:xyz[1],s});}
  return COMMON.declutterLabels(items,declutterOpts(W,H,k)).map((r,j)=>({c:who[j].c,ax:who[j].ax,ay:who[j].ay,w:who[j].s.w,h:who[j].s.h,x:r.x,y:r.y,moved:!!r.moved,hidden:!!r.hidden}));}
 function renderAutoLabels(){
  AUTO_CHIPS=[];dropLabels('flabel');dropLabels('blabel');dropLabels('maxd');
  const n=normLabelCount(LBL.findings);
  if(n>0)for(const item of rankedFindings().slice(0,n)){const f=item.f;
   if(Number.isFinite(+f.segment_id)&&HIDDEN.has(+f.segment_id))continue;   // C10: a hidden branch hides its labels
   const p=f.xyz_mm.map(Number),text=findingLabelText(f),sev=severityOf(f);
   const pri=LABEL_PRIORITY[sev]??LABEL_PRIORITY.info;
   makeLabel('flabel '+sev,text,anchorAt(p),'flabel',pri).onclick=()=>window.__selectFinding(item.i);
   AUTO_CHIPS.push({kind:'flabel',p,text,fill:FLABEL_FILL[sev],stroke:FLABEL_STROKE[sev],priority:pri});}
  if(LBL.branches)for(const g of CLGROUPS){if(HIDDEN.has(Number(g.segment_id)))continue;
   const p=branchMidpoint(g);if(!p)continue;const name=g.name||branchName(g.segment_id);
   const text=LANG==='en'?COMMON.englishLabel(name,'en'):name;
   makeLabel('blabel',text,anchorAt(p),'blabel',LABEL_PRIORITY.blabel);AUTO_CHIPS.push({kind:'blabel',p,text,fill:'#eef4fbf0',stroke:'#a9c4de',priority:LABEL_PRIORITY.blabel});}
  if(AMAX&&LBL.max_diameter){const ring=maxDiameterRing();
   if(ring.length>2)loopLine(ring,MAXD_COLOR);
   const at=Array.isArray(AMAX.xyz_mm)&&AMAX.xyz_mm.length===3&&AMAX.xyz_mm.every(v=>Number.isFinite(+v))?AMAX.xyz_mm.map(Number):(ring.length?ring[0]:null);
   if(at){const text=(LANG==='en'?'Max diameter ':'最大直径 ')+fmt(AMAX.max_diameter_mm,1)+' mm';
    makeLabel('maxd',text,anchorAt(at),'maxd',LABEL_PRIORITY.maxd);AUTO_CHIPS.push({kind:'maxd',p:at,text,fill:'#f5ecfbf0',stroke:'#b98fe0',ring:ring.length>2?ring:null,priority:LABEL_PRIORITY.maxd});}}
  requestLabelLayout();}
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
   setTip('最大直径站 '+fmt(AMAX.max_diameter_mm,1)+' mm · 距入口 '+fmt(from,0)+' mm');};}
 // ---- C7 measurement modes ----
 const NEED={distance:2,arc:2,diameter:1,segment:2},MODE_HINT={distance:'距离：在壁面点击两点。',arc:'弧长：在壁面点击两点，沿中心线树求和。',diameter:'管径：在壁面点击一点，按中心线切线切出真实截面。',segment:'分段：在同一分支上点击两点。'};
 let measureMode='',pendingPts=[],pickTask='';
 const measStatus=t=>{el('measure-status').textContent=t;};
 function setMeasureMode(mode){measureMode=measureMode===mode?'':mode;pendingPts=[];pickTask='';
  for(const k of Object.keys(NEED))el('measure-'+k).classList.toggle('on',measureMode===k);
  measStatus(measureMode?MODE_HINT[measureMode]:'未选择模式：点击壁面锁定探针。');if(measureMode)openCard('menu-measure');}
 for(const k of Object.keys(NEED))el('measure-'+k).onclick=()=>setMeasureMode(k);
 el('measure-clear').onclick=()=>{MEAS=[];pendingPts=[];renderMeasList();redrawOverlays();measStatus('已清空测量。');};
 function arcPolyline(a){if(!a||!a.same_branch||!a.a||!a.b)return null;const g=CLGROUPS[a.a.group_index];if(!g)return null;
  const first=a.a.s<=a.b.s?a.a:a.b,last=first===a.a?a.b:a.a,out=[first.xyz.slice()];
  for(let j=0;j<g.s.length;j++)if(g.s[j]>first.s&&g.s[j]<last.s)out.push([g.xyz[3*j],g.xyz[3*j+1],g.xyz[3*j+2]]);
  out.push(last.xyz.slice());return out;}
 // §15.14: the pick is projected onto the centreline, the local tangent becomes the section normal.
 function centerlineTangent(pr){return COMMON.centerlineTangent(CLGROUPS,pr);}
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
 // Every embedded field at prediction point i, the active one first (M1: 峰值 WSS · TAWSS · OSI).
 const probeFieldIds=()=>{const ids=supportedFields();ids.sort((a,b)=>(b===activeField)-(a===activeField));return ids;};
 function fieldValueText(id,x,ids){const F=fieldArrays(id),u=unitText(F.units);return (id==='wss'&&ids.length>1?FIELD_SHORT.wss:F.short)+' '+(F.isPa?fqu(x)+' '+ul():fq(x)+(u?' '+u:''));}
 function pointValues(i){const ids=probeFieldIds();return ids.map(id=>fieldValueText(id,fieldArrays(id).p[i],ids)).join(' · ');}
 // ---- §25 截面均值: the section perpendicular to the centreline at the probe's station, each wall field
 // integrated around the lumen outline (∮ f dl / ∮ dl) and attached to that centreline point ----
 const sectionKey=id=>'section_'+(id==='wss'?'wss_pa':FIELD_SCHEMA[id].array_key);
 function sectionAtPick(p){if(!CLGROUPS.length||!MF.length)return null;const fields={};for(const id of supportedFields())fields[id]=fieldArrays(id).p;
  const cell=Math.max(.5,(Number((META.cloud||{}).spacing_mm)||.5)*3);
  try{return Object.assign(COMMON.sectionMeans({groups:CLGROUPS,point:p,vertices:MV,faces:MF,fields,sample:q=>CORE.nearestIndex(PV,q,cell)}),{pick:p.slice()});}catch(_){return null;}}
 function sectionNote(sec){if(!sec)return '本报告没有可用中心线或壁面网格，无法取截面';
  if(!sec.found)return sec.reason==='no_contour'?'此处垂直截面未切到管腔轮廓':'此处没有可用中心线';
  return '截面：垂直中心线，过'+bn(sec.segment_id)+'弧长 '+fmt(sec.s_from_root_mm,1)+' mm 处 · 周长 '+fmt(sec.wall_length_mm,1)+' mm'
   +(sec.open?' · 轮廓不闭合，只平均切到的壁面':sec.synthetic?' · 经过开口，缺口 '+fmt(sec.gap_mm,1)+' mm 不计':'');}
 // Plain-text reading of the card (screen readers via aria-label; the tests read it too).
 function probeSummary(i,sec){const ids=probeFieldIds(),ok=sec&&sec.found;
  return '探针 · '+pointValues(i)+'\n'+bn(PS[i])+' · 弧长 '+fmt(P_S[i],1)+' mm · 半径 '+fmt(P_R[i],1)+' mm\n('+[0,1,2].map(k=>fmt(PV[3*i+k],1)).join(', ')+') mm\n'
   +(ok?'截面均值 · '+ids.filter(id=>sec.means[id]&&Number.isFinite(sec.means[id].mean)).map(id=>fieldValueText(id,sec.means[id].mean,ids)).join(' · ')+'\n':'')+sectionNote(sec);}
 // Where the probe value and the section mean sit on the ring's min → max (clamped; hollow dot when outside).
 function rangeBar(x,m,F,u){if(!m||!Number.isFinite(m.min)||!Number.isFinite(m.max)||!(m.max>m.min))return '';
  const k=F.isPa?uf():1,pos=v=>Math.max(0,Math.min(100,(v-m.min)/(m.max-m.min)*100)).toFixed(1),out=Number.isFinite(x)&&(x<m.min||x>m.max);
  const tip='截面环上 '+fq(m.min*k)+' – '+fq(m.max*k)+(u?' '+u:'')+'；竖线 = 截面均值，圆点 = 此点'+(out?'（此点在环上范围之外）':'');
  return '<span class="pc-range" title="'+esc(tip)+'"><i class="pc-mean" style="left:'+pos(m.mean)+'%"></i>'+(Number.isFinite(x)?'<i class="pc-pt'+(out?' out':'')+'" style="left:'+pos(x)+'%"></i>':'')+'</span>';}
 // v0.15.10: the probe card is a small table — one row per field, 此点 / 截面均值 / where both sit on the ring —
 // under a title bar (branch · arc · radius, 记录, ×); the active colouring field is highlighted and a row click switches to it.
 function renderProbeCard(){const card=el('probe-card'),i=probeSel;if(!(i>=0)){card.hidden=true;return;}
  const sec=SECTION,ok=!!(sec&&sec.found),ids=supportedFields();card.replaceChildren();
  const head=document.createElement('div');head.className='pc-head';
  const title=document.createElement('b');title.textContent='探针';
  const where=document.createElement('span');where.className='pc-where';where.textContent=bn(PS[i])+' · 弧长 '+fmt(P_S[i],1)+' mm · 半径 '+fmt(P_R[i],1)+' mm';
  const acts=document.createElement('span');acts.className='pc-actions';
  const add=document.createElement('button');add.type='button';add.className='pc-rec';add.textContent='记录';add.title='把此点与截面均值加入探针记录表';add.onclick=recordProbe;
  const close=document.createElement('button');close.type='button';close.className='pc-x';close.textContent='×';close.title='关闭探针（P）';close.setAttribute('aria-label','关闭探针');close.onclick=closeProbe;
  acts.append(add,close);head.append(title,where,acts);
  const rows=ids.map(id=>{const F=fieldArrays(id),u=F.isPa?ul():unitText(F.units),x=F.p[i],m=ok?sec.means[id]:null,v=y=>Number.isFinite(y)?(F.isPa?fqu(y):fq(y)):'—';
   return '<tr data-field="'+esc(id)+'"'+(id===activeField?' class="on"':'')+' title="按 '+esc(F.short)+' 着色"><th>'+esc(id==='wss'&&ids.length>1?FIELD_SHORT.wss:F.short)+(u?'<small>'+esc(u)+'</small>':'')+'</th><td>'+v(x)+'</td>'
    +(ok?'<td class="pc-sec">'+v(m&&m.mean)+'</td><td>'+rangeBar(x,m,F,u)+'</td>':'')+'</tr>';}).join('');
  const table=document.createElement('table');table.className='pc-table';
  table.innerHTML='<thead><tr><th></th><th title="离点击处最近的预测点">此点</th>'+(ok?'<th title="截面环上逐段取最近预测点，按长度加权平均（∮f dl / 周长）">截面均值</th><th title="截面环上最小 → 最大；竖线 = 截面均值，圆点 = 此点">环上范围</th>':'')+'</tr></thead><tbody>'+rows+'</tbody>';
  table.addEventListener('click',ev=>{let t=ev&&ev.target;while(t&&t!==table&&!(t.dataset&&t.dataset.field))t=t.parentNode;const id=t&&t.dataset&&t.dataset.field;if(id&&id!==activeField)setField(id);});
  const foot=document.createElement('div');foot.className='pc-foot';
  const a=document.createElement('span');a.textContent=sectionNote(sec);
  const b=document.createElement('span');b.textContent='此点 ('+[0,1,2].map(k=>fmt(PV[3*i+k],1)).join(', ')+') mm';
  if(ok){const lg=document.createElement('span');lg.className='pc-legend';lg.innerHTML='范围条<i class="pc-lg-mean"></i>截面均值<i class="pc-lg-pt"></i>此点';b.appendChild(lg);}
  foot.append(a,b);card.append(head,table,foot);
  card.setAttribute('role','group');card.setAttribute('aria-label',probeSummary(i,sec));card.hidden=false;}
 function closeProbe(){el('probe-card').hidden=true;probeSel=-1;if(SECTION){SECTION=null;redrawOverlays();}}
 function lockProbe(p){const i=CORE.nearestIndex(PV,new Float32Array([p[0],p[1],p[2]]),Math.max(.5,(Number((META.cloud||{}).spacing_mm)||.5)*3))[0];
  if(!(i>=0)){closeProbe();return;}
  probeSel=i;SECTION=sectionAtPick(p);redrawOverlays();renderProbeCard();}
 function recordProbe(){if(!(probeSel>=0))return;const i=probeSel;
  const values={wss_pa:+PW[i].toFixed(4)};for(const id of supportedFields())if(id!=='wss'){const F=fieldArrays(id);if(Number.isFinite(F.p[i]))values[FIELD_SCHEMA[id].array_key]=+F.p[i].toFixed(4);}
  if(SECTION&&SECTION.found){for(const id of supportedFields()){const m=SECTION.means[id];if(m&&Number.isFinite(m.mean))values[sectionKey(id)]=+m.mean.toFixed(4);}
   values.section_s_from_root_mm=+SECTION.s_from_root_mm.toFixed(2);values.section_perimeter_mm=+SECTION.wall_length_mm.toFixed(2);}
  PROBES.push({id:COMMON.newId('P',PROBES),xyz_mm:[0,1,2].map(k=>+PV[3*i+k].toFixed(3)),branch:branchName(PS[i]),segment_id:PS[i],s_from_root_mm:+P_S[i].toFixed(2),radius_mm:+P_R[i].toFixed(3),values,created_at:new Date().toISOString()});
  renderProbeLog();measStatus('已记录 '+PROBES.length+' 行探针。');}
 function renderProbeLog(){const box=el('probe-log');box.replaceChildren();
  if(!PROBES.length){const s=document.createElement('small');s.className='hint';s.textContent='还没有探针记录。';box.appendChild(s);return;}
  const t=document.createElement('table');t.className='probe-table';
  // Extra cycle columns (M1: tawss_pa / osi) appear only when a recorded row carries them.
  const xcols=supportedFields().filter(id=>id!=='wss').map(id=>{const F=fieldArrays(id);return [FIELD_SCHEMA[id].array_key,F.short+(F.isPa?' '+ul():unitText(F.units)?' '+unitText(F.units):''),F.isPa];})
    // §25 截面均值 columns follow the point columns, again only when a recorded row carries them.
    .concat(supportedFields().map(id=>{const F=fieldArrays(id);return [sectionKey(id),'截面 '+(id==='wss'?'WSS':F.short)+(F.isPa?' '+ul():unitText(F.units)?' '+unitText(F.units):''),F.isPa];}))
    .filter(([k])=>PROBES.some(r=>r.values&&r.values[k]!=null));
  t.innerHTML='<thead><tr><th>编号</th><th>分支</th><th>s mm</th><th>r mm</th><th>WSS '+esc(ul())+'</th>'+xcols.map(c=>'<th>'+esc(c[1])+'</th>').join('')+'<th></th></tr></thead><tbody>'+PROBES.map(r=>`<tr><td>${esc(r.id)}</td><td>${esc(r.branch)}</td><td>${fmt(r.s_from_root_mm,1)}</td><td>${fmt(r.radius_mm,1)}</td><td>${fqu(r.values.wss_pa)}</td>${xcols.map(c=>'<td>'+(c[2]?fqu(r.values[c[0]]):fq(r.values[c[0]]))+'</td>').join('')}<td><button type="button" data-del="${esc(r.id)}">×</button></td></tr>`).join('')+'</tbody>';
  t.addEventListener('click',ev=>{const id=ev&&ev.target&&ev.target.dataset&&ev.target.dataset.del;if(!id)return;PROBES=PROBES.filter(x=>x.id!==id);renderProbeLog();});
  const wrap=document.createElement('div');wrap.className='tscroll';wrap.appendChild(t);box.appendChild(wrap);}
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
  applyAdded();renderFindingsList(LAST_ACTIVE);saveReview(false);openCard('menu-findings');}
 // ---- pick router: measurement > annotation > new finding > region brush > probe ----
 function handlePick(p){
  if(measureMode){addMeasurePick(p);return 'measure';}
  if(pickTask==='annot'){pickTask='';addAnnotation(p);return 'annot';}
  if(pickTask==='finding'){pickTask='';addManualFinding(p);return 'finding';}
  if(el('region-pick').checked&&el('region-mode').value==='sphere'){regionCenter=p.slice();el('region-center').textContent='中心 ('+p.map(v=>fmt(v,1)).join(', ')+') mm';openCard('menu-region');applyRegion();return 'region';}
  lockProbe(p);return 'probe';}
 function rebuildTop(){requestRender();const TV=fieldArrays(activeField).p;highlightThreshold=CORE.topThreshold(TV,highlightPct);topIndices=[];const topPositions=[];for(let i=0;i<TV.length;i++)if(Number.isFinite(TV[i])&&TV[i]>=highlightThreshold&&(branchSel<0||PS[i]===branchSel)){topPositions.push(PV[i*3],PV[i*3+1],PV[i*3+2]);topIndices.push(i);}if(top){scene.remove(top);top.geometry.dispose();top.material.dispose();}const tg=new THREE.BufferGeometry();tg.setAttribute('position',new THREE.Float32BufferAttribute(topPositions,3));const tc=new Float32Array(topPositions.length);for(let j=0;j<topIndices.length;j++)tc.set([.95,.2,.75],3*j);tg.setAttribute('color',new THREE.BufferAttribute(tc,3));top=new THREE.Points(tg,new THREE.PointsMaterial({size:1.8,vertexColors:true,depthTest:false,clippingPlanes:[clipPlane],transparent:true,opacity:.95}));top.renderOrder=2;top.visible=(viewMode==='wss'||viewMode==='cloud')&&el('showtop').checked;scene.add(top);el('highlight-value').textContent=fmt(highlightPct,1)+'%';el('highlight-label').textContent=fmt(highlightPct,1)+'%';}
 function rebuildHlPts(){if(hlPts){scene.remove(hlPts);hlPts.geometry.dispose();hlPts.material.dispose();hlPts=null;}const idx=hl.pointIndices||[];if(!idx.length)return;const pos=new Float32Array(idx.length*3),col=new Float32Array(idx.length*3);idx.forEach((i,j)=>{pos.set([PV[3*i],PV[3*i+1],PV[3*i+2]],3*j);col.set([0,.76,.85],3*j);});const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));g.setAttribute('color',new THREE.BufferAttribute(col,3));hlPts=new THREE.Points(g,new THREE.PointsMaterial({size:1.6,vertexColors:true,depthTest:false,clippingPlanes:[clipPlane],transparent:true,opacity:.9}));hlPts.renderOrder=2;scene.add(hlPts);}
 function setHighlight(h){requestRender();hl={vertexMask:h&&h.vertexMask||null,pointIndices:h&&h.pointIndices||null,label:h&&h.label||''};if(h&&h.markerAt){marker.position.set(...h.markerAt);marker.scale.setScalar(Math.max(1,(h.markerRadius||2)));marker.visible=true;markerLabel.textContent=hl.label;}else{marker.visible=false;markerLabel.textContent='';}rebuildHlPts();recolor();}
 function clearHighlight(){activeFinding=-1;profSel=null;regionSel=null;renderFindingsList(-1);el('region-stats').innerHTML='';setHighlight(null);drawProfiles();}
 try{const saved=JSON.parse(localStorage.getItem('wss-report-scale-v1'));if(saved&&saved.mode==='fixed'&&Number.isFinite(saved.max)&&saved.max>0){fixedMax=saved.max;scaleMode='fixed';}if(saved&&Number.isFinite(saved.highlightPct))highlightPct=Math.max(.5,Math.min(10,saved.highlightPct));}catch(_){}
 userScale={mode:scaleMode,max:fixedMax};
 el('scale-mode').value=scaleMode;el('fixedmax').value=fixedMax;el('highlight-pct').value=highlightPct;el('cmap').value=cmapName;el('cmap').onchange=e=>{cmapName=CMAPS[e.target.value]?e.target.value:'rainbow';try{localStorage.setItem('wss-report-cmap',cmapName);}catch(_){}recolor();};rebuildTop();
 el('bands').onchange=e=>{bands=Number(e.target.value)||0;recolor();};el('logscale').onchange=e=>{logScale=e.target.checked;recolor();};
 el('units').onchange=e=>{UNIT=CORE.unitInfo(e.target.value)===CORE.UNITS.dyn?'dyn':'Pa';renderPanel();initThresholdInputs();renderFindingsList(activeFinding);el('fixedmax').value=String(+(fixedMax*uf()).toFixed(3));recolor();drawProfiles();renderRegionStats();redrawOverlays();};
 function fieldTitle(lang){const AF=fieldArrays(activeField);return activeField==='wss'?legendTitle(lang):AF.short+(al()?' · '+al():'');}
 function feature(){const L=(k,u)=>COMMON.englishLabel(k,LANG)+' · '+u;if(feat==='s')return {values:P_S,mode:'vir',lo:0,hi:bounds(P_S)[1],label:L('arc','mm')};if(feat==='th')return {values:P_TH,mode:'vir',lo:-Math.PI,hi:Math.PI,label:L('theta','rad')};if(feat==='r')return {values:P_R,mode:'vir',lo:bounds(P_R)[0],hi:bounds(P_R)[1],label:L('radius','mm')};if(feat==='dj')return {values:P_DJ,mode:'vir',lo:0,hi:bounds(P_DJ)[1],label:L('dist_junction','mm')};const AF=fieldArrays(activeField);return {values:AF.p,mode:'wss',lo:0,hi:vmax,label:fieldTitle(LANG)};}
 const TRUST_TINT={2:[.78,.72,.62],4:[.72,.66,.8]};
 // §19.2 stagnation overlay: magenta diagonal hatch (planes x+y+z = const) over whatever field is showing.
 const STAG_RGB=[.85,.1,.55];let STAG_M=null,STAG_BAND=null;
 // Stripe width = 3 × the median mesh edge (so the hatch survives per-vertex colour interpolation), at least 1 % of the model.
 function stagMesh(){if(!STAG_M&&HAS_STAG()){STAG_M=stagnationMask('m');const e=[];for(let t=0;t<MF.length&&e.length<4000;t+=3*Math.max(1,Math.floor(MF.length/12000))){const a=MF[t],b=MF[t+1];e.push(Math.hypot(MV[3*a]-MV[3*b],MV[3*a+1]-MV[3*b+1],MV[3*a+2]-MV[3*b+2]));}
   e.sort((x,y)=>x-y);const w=Math.max(3*(e[e.length>>1]||1),size*.01)*Math.sqrt(3);STAG_BAND=new Uint8Array(STAG_M.length);for(let v=0;v<STAG_M.length;v++)STAG_BAND[v]=Math.floor((MV[3*v]+MV[3*v+1]+MV[3*v+2])/w)&1;}return STAG_M;}
 function colorArray(values,segment,array,mode,lo,hi,isMesh){const AF=fieldArrays(activeField),isField=values===AF.m||values===AF.p,LG=logScale&&AF.log!==false,stag=isMesh&&showStag&&mode!=='grey'?stagMesh():null;
  for(let i=0;i<values.length;i++){let c=!Number.isFinite(values[i])?[.48,.51,.55]:(mode==='grey'?[.72,.75,.8]:(mode==='wss'?cmap(CORE.normalize(values[i],lo,hi,LG)):vir((values[i]-lo)/(hi-lo||1))));
   if(isMesh&&showTrust&&MT){const b=MT[i];if(b&1)c=[.48,.51,.55];else if(b&4)c=c.map((v,k)=>v*.25+.75*TRUST_TINT[4][k]);else if(b&2)c=c.map((v,k)=>v*.25+.75*TRUST_TINT[2][k]);}
   if(stag&&stag[i])c=STAG_BAND[i]?c.map((v,k)=>v*.25+.75*STAG_RGB[k]):c.map(v=>v*.45+.55);   // magenta / pale stripes
   if(highlightEnabled&&isField&&mode==='wss'&&Number.isFinite(values[i])&&values[i]<highlightThreshold)c=c.map(v=>v*.28+.72*.86);
   if(branchSel>=0&&segment[i]!==branchSel)c=c.map(v=>v*.25+.75*.86);
   if(isMesh&&hl.vertexMask&&!hl.vertexMask[i])c=c.map(v=>v*.3+.7*.88);
   array.set(c,3*i);}}
 // Colour-bar ticks go through formatValue (§19.1): no scientific notation, three significant digits.
 function drawBar(mode,lo,hi,label){el('cbar').style.display=viewMode==='stl'?'none':'';el('cbar-unit').textContent=viewMode==='stl'?(LANG==='en'?'Input STL · mm':'输入 STL · mm'):label;let g='linear-gradient(to top';const n=mode==='wss'&&bands>0?bands:0;
  if(n){for(let i=0;i<n;i++){const c='rgb('+cmapRaw((i+.5)/n).map(x=>Math.round(x*255)).join(',')+')';g+=','+c+' '+(i/n*100)+'%,'+c+' '+((i+1)/n*100)+'%';}}else for(let i=0;i<=32;i++)g+=',rgb('+(mode==='wss'?cmapRaw(i/32):vir(i/32)).map(x=>Math.round(x*255)).join(',')+') '+(i/32*100).toFixed(2)+'%';
  el('cbar').style.background=g+')';
  // v0.13: fine ticks (nice steps + exact ends, minor marks halfway; band edges when banded) and the active field's thresholds.
  const isW=mode==='wss',LG=isW&&logScale&&fieldArrays(activeField).log!==false,f=isW?af():1,H=Math.max(80,el('cbar').clientHeight||300);
  const spec=barSpec(mode,lo,hi,LG,f);let h='';
  for(const t of spec.ticks){const pos=(t.f*100).toFixed(2)+'%';h+='<i class="tk'+(t.major?'':' minor')+'" style="bottom:'+pos+'"></i>';if(t.major)h+='<span style="bottom:calc('+pos+' - 8px)">'+esc(COMMON.tickLabel(t))+'</span>';}
  const shown=COMMON.spreadLabels(spec.marks,16/H);
  for(const m of spec.marks){const pos=(m.f*100).toFixed(2)+'%';h+='<b class="thr" style="bottom:'+pos+'" title="'+esc(m.title)+'"></b>';if(shown.includes(m))h+='<em class="thr-l" style="bottom:calc('+pos+' - 8px)" title="'+esc(m.title)+'">'+esc(m.label)+'</em>';}
  el('cbar').innerHTML=h;
  // v0.15: the bar is a picture; give it a text alternative (units, range and the threshold marks).
  const majors=spec.ticks.filter(t=>t.major),cb=el('cbar');if(cb&&cb.setAttribute){cb.setAttribute('role','img');cb.setAttribute('aria-label',(LANG==='en'?'Colour bar ':'色标 ')+String(label||'')+(majors.length?(LANG==='en'?': ':'：')+COMMON.tickLabel(majors[0])+' – '+COMMON.tickLabel(majors[majors.length-1]):'')+(spec.marks.length?(LANG==='en'?'; marks ':'；阈值 ')+spec.marks.map(m=>m.label).join(' / '):''));}}
 // Ticks and threshold marks of the colour bar in display units, shared by the on-screen bar, PNG export and screenshot.
 function barSpec(mode,lo,hi,LG,f){const isW=mode==='wss',lo2=(LG?Math.max(lo,.05):lo)*f,hi2=hi*f,floor=.05*f;
  const ticks=COMMON.colorbarTicks({min:lo2,max:hi2,log:LG,floor,bands:isW?bands:0,minGap:.055});
  const names=aboveField(activeField)?['','','']:['低','高','极高'],marks=[];
  if(isW&&viewMode==='wss')fieldThr(activeField).forEach((v,k)=>{const d=v*f,fr=CORE.normalize(d,lo2,hi2,LG,floor);
    if(Number.isFinite(fr)&&fr>=0&&fr<=1)marks.push({f:fr,v:d,label:String(+(+d).toPrecision(4)),title:(names[k]?names[k]+'阈值 ':'阈值 ')+String(+(+d).toPrecision(4))+(al()?' '+al():'')});});
  return {ticks,marks,lo:lo2,hi:hi2};}
 // Canvas version for exports: bar at (x, y, w, h), colour function t → [r, g, b] in 0..1, font scale k.
 function paintCanvasBar(ctx,x,y,w,h,k,colorFn,spec){
  for(let i=0;i<h;i++){const c=colorFn(1-i/Math.max(1,h-1)).map(v=>Math.round(v*255));ctx.fillStyle='rgb('+c.join(',')+')';ctx.fillRect(x,y+i,w,1);}
  ctx.strokeStyle='#33414f';ctx.lineWidth=Math.max(1,k);ctx.strokeRect(x,y,w,h);ctx.fillStyle='#203049';ctx.font=Math.round(13*k)+'px sans-serif';ctx.textAlign='left';
  for(const t of spec.ticks){const ty=y+h*(1-t.f);ctx.beginPath();ctx.moveTo(x+w,ty);ctx.lineTo(x+w+(t.major?5:3)*k,ty);ctx.stroke();if(t.major)ctx.fillText(COMMON.tickLabel(t),x+w+Math.round(8*k),ty+Math.round(4.5*k));}
  const shown=COMMON.spreadLabels(spec.marks,16*k/Math.max(1,h));ctx.lineWidth=Math.max(1.5,2*k);ctx.textAlign='right';ctx.font='600 '+Math.round(12*k)+'px sans-serif';
  for(const m of spec.marks){const ty=y+h*(1-m.f);ctx.beginPath();ctx.moveTo(x-5*k,ty);ctx.lineTo(x+w,ty);ctx.stroke();if(shown.includes(m))ctx.fillText(m.label,x-Math.round(8*k),ty+Math.round(4.5*k));}
  ctx.textAlign='left';ctx.lineWidth=Math.max(1,k);}
 // Contours follow the coloured field: WSS uses the editable thresholds, TAWSS / OSI their descriptor thresholds.
 function contourSpec(){const AF=fieldArrays(activeField);return {values:AF.m,levels:fieldThr(activeField)};}
 function rebuildContours(){requestRender();if(contourObj){scene.remove(contourObj);contourObj.geometry.dispose();contourObj.material.dispose();contourObj=null;}if(!showContours)return;const spec=contourSpec(),parts=[];for(const level of spec.levels){const seg=CORE.marchingTriangles(MV,MF,spec.values,level);parts.push(seg);}let total=0;for(const p of parts)total+=p.length;const all=new Float32Array(total);let at=0;for(const p of parts){all.set(p,at);at+=p.length;}const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(all,3));contourObj=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0x203049,clippingPlanes:[clipPlane]}));contourObj.visible=viewMode==='wss';scene.add(contourObj);}
 function renderTrustLegend(){const box=el('trust-legend');if(!showTrust||!TRUST||!MT){box.hidden=true;return;}const fr=TRUST.fractions||{},src=Array.isArray(TRUST.sources)?TRUST.sources:[];const rows=[[1,'插值无支撑','rgb(122,130,140)','interpolation_uncovered'],[2,'表面粗糙','rgb(199,184,158)','rough_surface'],[4,'几何越界','rgb(184,168,204)','geometry_out_of_range']];box.innerHTML='<b>可信区域叠加</b>'+rows.map(([bit,label,color,key])=>{const s=src.find(x=>Number(x.bit)===bit);return `<span><i class="sw" style="background:${color}"></i>${esc((s&&s.label)||label)} ${Number.isFinite(+fr[key])?fmt(fr[key]*100,1)+'%':''}${s&&s.rule?'<br><small>'+esc(s.rule)+'</small>':''}</span>`;}).join('');box.hidden=false;}
 // Compact swatch legend under the colour bar (was one long sentence); the stagnation row appears with its overlay.
 function renderLegendNote(){const st=showStag?stagnationStats():null,c=stagCriteria();
  el('legend-note').innerHTML='<span title="插值覆盖范围之外，没有有效读值"><i class="sw" style="background:#7a828c"></i>无插值</span><span title="非选中分支 / 高亮阈值以下 / 区域统计之外"><i class="sw" style="background:#dde1e6"></i>淡化</span>'
   +(showStag?`<span title="${esc('TAWSS < '+c.t+' Pa 且 OSI > '+c.o+'（单周期）')}"><i class="sw hatch"></i>滞留区${st?' '+fmt(st.frac*100,0)+'%':''}</span>`:'');}
 // A field switch (menu tab, segmented control, F key or view state) repaints through recolor(); the
 // field-dependent panels (statistics card, contours, profile curves) follow here.
 let paintedField=activeField;
 function fieldChanged(){paintedField=activeField;initThresholdInputs();if(showContours)rebuildContours();renderPanel();drawProfiles();}
 // v0.14 (F1): tell a parent page (the compare page) that the display changed, so its sync does not wait for a camera move.
 function announceChange(){if(!ANNOUNCE||announceTimer||window.parent===window)return;announceTimer=setTimeout(()=>{announceTimer=null;postToParent({type:'wss-view:changed',family:'wall',run_identity:META.run_identity||null});},120);}
 function recolor(){requestRender();announceChange();if(paintedField!==activeField)fieldChanged();const AF=fieldArrays(activeField),LGS=logScale&&AF.log!==false;vmax=scaleMode==='fixed'&&(!fixedField||fixedField===activeField)?fixedMax:Math.max(Number(AF.p99)||0,.01);el('fixed-control').hidden=scaleMode!=='fixed';el('scale-description').textContent=(activeField==='wss'?'':AF.short+' · ')+(LGS?'对数 ':'')+fq((LGS?.05:0)*af())+' – '+fq(vmax*af())+(al()?' '+al():'')+' · '+(scaleMode==='fixed'&&(!fixedField||fixedField===activeField)?(fixedShared?'并排比较共用上限':'固定色标'):'本例空间 p99')+'；超限按上限着色'+(bands?' · '+bands+' 段':'');rebuildTop();colorArray(AF.m,MS,mcol,viewMode==='wss'?'wss':'grey',0,vmax,true);const f=feature();colorArray(f.values,PS,pcol,f.mode,f.lo,f.hi,false);updateColors();el('display-readout').textContent='当前视图：'+({wss:'WSS 壁面',stl:'输入 STL',cl:'中心线',cloud:'预测点云'}[viewMode]||viewMode)+' · '+(viewMode==='wss'?(activeField==='wss'?'统计：预测点云 · 壁面：Gaussian 插值':AF.short+'（壁面着色）· 对应统计卡在「统计与口径」顶部'):f.label)+(branchSel>=0?' · 分支：'+branchName(branchSel):'')+(hl.label?' · '+hl.label:'');
  if(viewMode==='cloud')drawBar(f.mode,f.lo,f.hi,f.label);else if(viewMode==='cl')drawBar('vir',rmin,rmax,COMMON.englishLabel('centerline_radius',LANG)+' · mm');else drawBar('wss',0,vmax,fieldTitle(LANG));placePeak();renderTrustLegend();renderLegendNote();if(contourObj)contourObj.visible=viewMode==='wss'&&showContours;drawUnroll();
  if(probeSel>=0&&!el('probe-card').hidden)renderProbeCard();}
 recolorActive=recolor;
 function updateColors(){mg.attributes.color.needsUpdate=true;pg.attributes.color.needsUpdate=true;}
 function visibility(){requestRender();announceChange();mesh.visible=viewMode!=='cloud';mmat.transparent=viewMode==='cl'||opacity<1;mmat.opacity=viewMode==='cl'?.18:opacity;pts.visible=viewMode==='cloud'||el('showpts').checked;cl.visible=viewMode==='cl'||el('showcl').checked;peak.visible=viewMode==='wss'&&el('showpeak').checked&&peakOK;trough.visible=viewMode==='wss'&&el('showpeak').checked&&troughOK;top.visible=(viewMode==='wss'||viewMode==='cloud')&&el('showtop').checked;if(hlPts)hlPts.visible=viewMode==='wss'||viewMode==='cloud';if(contourObj)contourObj.visible=viewMode==='wss'&&showContours;for(const item of labels)if(item.kind==='endpoint')item.obj.visible=viewMode==='cl';}
 function setView(mode){viewMode=mode;document.querySelectorAll('[data-v]').forEach(b=>{const active=b.dataset.v===mode;b.classList.toggle('on',active);b.setAttribute('aria-pressed',String(active));});visibility();recolor();setTip(mode==='cloud'?'悬停读取预测点原值与几何特征':TIP_HINT,true);}
 document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setView(b.dataset.v));
 function saveScale(){try{localStorage.setItem('wss-report-scale-v1',JSON.stringify({mode:userScale.mode,max:userScale.max,highlightPct}));}catch(_){}recolor();}
 const userRange=()=>{fixedField=null;fixedShared=false;userScale={mode:scaleMode,max:fixedMax};};
 el('scale-mode').onchange=e=>{scaleMode=e.target.value;userRange();saveScale();};el('fixedmax').onchange=e=>{const v=CORE.fromUnit(Number(e.target.value),UNIT);if(!Number.isFinite(v)||v<=0){e.target.value=String(+(fixedMax*uf()).toFixed(3));return;}fixedMax=v;userRange();saveScale();};
 el('branch').onchange=e=>{branchSel=Number(e.target.value);recolor();};el('feat').onchange=e=>{feat=e.target.value;if(feat!=='wss'){el('showpts').checked=false;setView('cloud');}else recolor();};
 el('highlight-pct').oninput=e=>{highlightPct=Math.max(.5,Math.min(10,Number(e.target.value)||1));highlightEnabled=el('showtop').checked;saveScale();};
 for(const id of ['showpts','showcl','showpeak'])el(id).onchange=()=>{if(el('showpts').checked&&feat!=='wss')setView('cloud');else visibility();};
 el('showtop').onchange=()=>{highlightEnabled=el('showtop').checked;recolor();visibility();};
 el('contours').onchange=e=>{showContours=e.target.checked;rebuildContours();};
 el('trust').onchange=e=>{showTrust=e.target.checked&&!!MT;recolor();};
 function applyThresholds(){const id=activeField,k0=thrScale(id),t=CORE.validThresholds([0,1,2].map(k=>Number(el('thr'+k).value)/k0));if(!t){initThresholdInputs();setTip('阈值必须是三个递增的非负数。');return;}
  if(id==='wss')thresholdsPa=t;else if(sameThr(t,defaultThr(id)))delete FIELD_THR[id];else FIELD_THR[id]=t;renderPanel();if(showContours)rebuildContours();drawProfiles();recolor();}
 for(const k of [0,1,2])el('thr'+k).onchange=applyThresholds;
 // v0.14: the slider says what it cuts (world Z of the STL, keeps z ≤ cut) and where the cut is.
 function clipText(){return clipT>=1?'不剖切':'Z ≤ '+fmt(mn[2]+(mx[2]-mn[2])*clipT,0)+' mm';}
 el('clip').oninput=e=>{clipT=Number(e.target.value)/100;clipPlane.constant=clipT>=1?1e6:mn[2]+(mx[2]-mn[2])*clipT;el('clip-value').textContent=clipText();announceChange();setTip('剖切已更新；鼠标只读取可见部分。',true);};
 el('opacity').oninput=e=>{opacity=Number(e.target.value)||1;visibility();};
 el('stag').onchange=e=>{showStag=!!e.target.checked&&HAS_STAG();recolor();};
 // ---- camera (§19.9): default = anatomical front view framed by fitView; reset / 0 / R return to it, 1–6 = standard views ----
 // fitName remembers which fitted view is showing so a viewport resize re-frames it; any manual move clears it.
 let tween=null,fitName=null;
 function viewDir(name){if(!FRAME)return name==='front'?{dir:[0,0,-1],up:[0,1,0]}:null;   // no frame: legacy world +z camera
  const v=CORE.standardViews(FRAME.rotation,[0,0,0],1)[name];return v?{dir:v.target.map((t,k)=>t-v.position[k]),up:v.up}:null;}
 function fittedView(name){const d=viewDir(name);if(!d)return null;const f=COMMON.fitView(MV,{dir:d.dir,up:d.up,fov:camera.fov,aspect:camera.aspect,margin:1.12});return {position:f.position,target:f.target,up:f.up};}
 function fittedViews(){if(!FRAME)return null;const out={};for(const k of Object.keys(CORE.VIEW_NAMES))out[k]=fittedView(k);return out;}
 function goView(name,animate){const c=fittedView(name);if(!c)return false;setCamera(c,animate);fitName=name;lastViewName=FRAME?name:'custom';return true;}
 function resetCamera(animate){camera.near=size*.001;camera.far=size*30;camera.updateProjectionMatrix();goView('front',!!animate);}
 el('reset-camera').onclick=()=>resetCamera(true);
 function setCamera(c,animate){if(!c||!c.position||!c.target)return;requestRender();const up=c.up&&c.up.length===3?c.up:[0,1,0];if(!animate){tween=null;camera.position.set(...c.position);camera.up.set(...up);ensureControls();controls.target.set(...c.target);controls.update();return;}camera.up.set(...up);ensureControls();tween={t0:performance.now(),ms:450,p0:camera.position.clone(),t0v:controls.target.clone(),p1:new THREE.Vector3(...c.position),t1:new THREE.Vector3(...c.target)};}
 function flyTo(target,ext){const dir=camera.position.clone().sub(controls.target).normalize();const dist=Math.max(ext*6,size*.12);fitName=null;setCamera({position:target.map((v,k)=>v+dir.getComponent(k)*dist),target,up:camera.up.toArray()},true);}
 const stdBox=el('std-views');
 if(!FRAME){stdBox.querySelectorAll('[data-view]').forEach(b=>{b.disabled=true;b.title='本报告未提供解剖坐标架';});el('std-note').textContent='本报告未提供解剖坐标架（rotation/origin），标准视角与相机联动不可用；「复位视角」沿世界 +z 方向撑满视口。';el('views-export').disabled=true;}
 else{el('std-note').textContent=(META.frame_transform.direction_source==='unknown_stl'?'按解剖坐标架推断，请核对左右。':'解剖坐标架来源：'+META.frame_transform.direction_source)+' 前/后/左/右/上/下按患者方向并撑满视口（快捷键 1–6）。';stdBox.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>goView(b.dataset.view,true));}
 // ---- highlight helpers ----
 function markVertices(pointIdxSet){const n=near();const mask=new Uint8Array(MV.length/3);for(let v=0;v<mask.length;v++){const p=n[v];if(p>=0&&pointIdxSet[p])mask[v]=1;}return mask;}
 window.__selectFinding=i=>{if(activeFinding===i){clearHighlight();return;}const f=FINDINGS[i];if(!f)return;profSel=null;regionSel=null;activeFinding=i;renderFindingsList(i);const ext=Math.max(Number(f.extent_mm)||5,3);const xyz=Array.isArray(f.xyz_mm)&&f.xyz_mm.length===3?f.xyz_mm.map(Number):null;let mask=null,idx=null;
  if(Array.isArray(f.point_indices)&&f.point_indices.length){idx=f.point_indices.map(Number).filter(k=>Number.isInteger(k)&&k>=0&&k<PW.length);const set=new Uint8Array(PW.length);for(const k of idx)set[k]=1;mask=markVertices(set);}
  else if(xyz){mask=CORE.sphereMask(MV,xyz,ext*1.5);}
  const value=findingValue(f);
  setHighlight({vertexMask:mask,pointIndices:idx,markerAt:xyz,markerRadius:Math.min(ext*.35,6),label:(f.label||KIND_CN[f.kind]||f.kind||'发现')+(value?' '+value:'')});if(xyz)flyTo(xyz,ext);setTip((f.label||KIND_CN[f.kind]||f.kind||'发现')+(value?' · '+value:'')+(f.definition?'\n'+f.definition:''));};
 // ---- profiles ----
 let profSel=null;const PROF_COLORS=['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b','#17becf','#7f7f7f'];
 const profColor=(b,bi)=>HIDDEN.has(Number(b.segment_id))?'#c3ccd6':PROF_COLORS[bi%PROF_COLORS.length];
 let profHits=[];
 function profBranches(){if(!PROFILES)return [];const which=el('prof-branch').value;return PROFILES.branches.filter(b=>which===''||String(b.segment_id)===which);}
 function drawCurve(canvas,branches,xkey,series,yLabel,scaleY,marker,hlines){const c=canvas,W=c.width=Math.max(200,Math.round(c.clientWidth*2)),H=c.height,ctx=c.getContext('2d');if(!ctx)return null;ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);
  const L=62,R=14,T=14,B=36;let xmin=Infinity,xmax=-Infinity,ymax=0;
  for(const b of branches){const xs=b[xkey]||b.s_local_mm||[];for(let i=0;i<xs.length;i++){if(!(b.wss&&b.wss.n&&b.wss.n[i]>0)&&!series.some(s=>s.key==='radius_mm'))continue;xmin=Math.min(xmin,xs[i]);xmax=Math.max(xmax,xs[i]);for(const s of series){const v=s.get(b)[i];if(Number.isFinite(v))ymax=Math.max(ymax,v*scaleY);}}}
  if(!Number.isFinite(xmin)||xmax<=xmin){ctx.fillStyle='#5f7086';ctx.font='22px sans-serif';ctx.fillText('没有可绘制的分箱',L,H/2);return null;}
  for(const hv of hlines||[])if(Number.isFinite(+hv))ymax=Math.max(ymax,+hv*scaleY*1.08);
  ymax=ymax||1;const X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),Y=y=>T+(1-y/ymax)*(H-T-B);
  ctx.strokeStyle='#dce4ed';ctx.lineWidth=1;ctx.beginPath();for(let k=0;k<=4;k++){const y=Y(ymax*k/4);ctx.moveTo(L,y);ctx.lineTo(W-R,y);}ctx.stroke();ctx.fillStyle='#5f7086';ctx.font='20px sans-serif';ctx.textAlign='right';for(let k=0;k<=4;k++)ctx.fillText(fq(ymax*k/4),L-6,Y(ymax*k/4)+7);ctx.textAlign='center';for(let k=0;k<=4;k++){const x=xmin+(xmax-xmin)*k/4;ctx.fillText(fmt(x,0),X(x),H-12);}ctx.textAlign='left';ctx.fillText(yLabel,L+4,T+18);
  for(const hv of hlines||[]){if(!Number.isFinite(+hv))continue;const y=Y(+hv*scaleY);ctx.strokeStyle='#8a97a6';ctx.lineWidth=1.5;ctx.setLineDash([10,8]);ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(W-R,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#5f7086';ctx.textAlign='right';ctx.fillText(fq(+hv*scaleY),W-R-4,y-6);ctx.textAlign='left';}
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
 // §19.2 the upper curve follows the coloured field when the profile carries it (tawss: mean / min, osi: p90 / mean);
 // without those keys the WSS curve stays and the legend says so.
 function profSpec(){const want=activeField,has=id=>!!(PROFILES&&PROFILES.branches.some(b=>b&&b[id]&&typeof b[id]==='object'));
  if(want==='tawss'&&has('tawss')){const g=b=>b.tawss||{},t=stagCriteria().t;return {id:'tawss',names:['均值','最低'],primary:{key:'mean',get:b=>g(b).mean_pa||[],band:b=>g(b).min_pa||[]},secondary:{key:'min',get:b=>g(b).min_pa||[],dash:[8,6],width:2},
   label:'TAWSS · '+ul(),scale:uf(),pa:true,hlines:[t],legend:'实线：均值 · 虚线：最低 · 阴影：最低–均值 · 灰虚线：低 TAWSS 阈值 '+String(+(t*uf()).toFixed(3))+' '+ul()};}
  if(want==='osi'&&has('osi')){const g=b=>b.osi||{},o=stagCriteria().o;return {id:'osi',names:['p90','均值'],primary:{key:'p90',get:b=>g(b).p90||[],band:b=>g(b).mean||[]},secondary:{key:'mean',get:b=>g(b).mean||[],dash:[8,6],width:2},
   label:'OSI',scale:1,pa:false,hlines:[o],legend:'实线：p90 · 虚线：均值 · 阴影：均值–p90 · 灰虚线：OSI '+o};}
  if((want==='rrt'||want==='ecap')&&has(want)){const g=b=>b[want]||{},t=fieldThr(want)[0],sh=FIELD_SHORT[want];return {id:want,names:['p90','均值'],primary:{key:'p90',get:b=>g(b).p90||[],band:b=>g(b).mean||[]},secondary:{key:'mean',get:b=>g(b).mean||[],dash:[8,6],width:2},
   label:sh+' · Pa⁻¹',scale:1,pa:false,hlines:[t],short:sh,legend:'实线：p90 · 虚线：均值 · 阴影：均值–p90 · 灰虚线：'+sh+' '+String(+t.toFixed(3))+' Pa⁻¹'};}
  const g=b=>b.wss||{};return {id:'wss',names:['p99','均值'],primary:{key:'p99',get:b=>g(b).p99_pa||[],band:b=>g(b).mean_pa||[]},secondary:{key:'mean',get:b=>g(b).mean_pa||[],dash:[8,6],width:2},
   label:'WSS · '+ul(),scale:uf(),pa:true,hlines:null,legend:'实线：p99 · 虚线：均值 · 阴影：均值–p99',missing:isCycleField(want)?fieldArrays(want).short:null};}
 function drawProfiles(){if(!PROFILES||!el('menu-profiles').open)return;const branches=profBranches(),xkey=el('prof-x').value,sp=profSpec();
  const diam=STATIONS.size&&branches.some(hasStations);
  el('prof-legend').innerHTML=branches.map((b,bi)=>`<span><i style="background:${profColor(b,bi)}"></i>${esc(b.name||branchName(b.segment_id))}${HIDDEN.has(Number(b.segment_id))?' · 已隐藏':''}</span>`).join('')+'<span>'+esc(sp.legend)+'</span>'+(sp.missing?'<span>本报告的沿程数据不含 '+esc(sp.missing)+'，上图仍为峰值 WSS</span>':'')+(diam?'<span>下图 实线：半径 · 长虚线：最大直径 · 点线：等效直径</span>':'');
  profHits=drawCurve(el('prof-wss'),branches,xkey,[sp.primary,sp.secondary],sp.label,sp.scale,profSel,sp.hlines)||[];
  el('prof-wss').setAttribute('aria-label','沿程 '+sp.label+' 曲线');
  drawCurve(el('prof-r'),branches,xkey,radiusSeries(),diam?'半径 / 直径 · mm':'半径 · mm',1,profSel);}
 window.__drawProfiles=drawProfiles;
 if(PROFILES){el('prof-branch').onchange=drawProfiles;el('prof-x').onchange=drawProfiles;el('prof-clear').onclick=clearHighlight;
  el('prof-wss').addEventListener('click',ev=>{if(!profHits.length)return;const r=ev.target.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;let best=null,bd=Infinity;for(const h of profHits){const d=Math.abs(h.x-x)*4+Math.abs(h.y-y);if(d<bd){bd=d;best=h;}}if(!best)return;selectProfile(best.b,best.i);});}
 function selectProfile(b,i){const s=(b.s_from_root_mm||[])[i];if(!Number.isFinite(s))return;activeFinding=-1;regionSel=null;renderFindingsList(-1);profSel={b,i};near();const sid=Number(b.segment_id),mask=new Uint8Array(MV.length/3);for(let v=0;v<mask.length;v++)if(MS[v]===sid&&Math.abs(VS[v]-s)<=2)mask[v]=1;const pm=CORE.rangeMask(P_S,PS,sid,s-2,s+2),idx=CORE.maskIndices(pm);const sp=profSpec(),pv=v=>sp.pa?fqu(v)+' '+ul():fq(v);const label=(b.name||branchName(sid))+' s='+fmt(s,0)+' mm · '+(sp.id==='wss'?'WSS':sp.label.split(' ')[0])+' '+sp.names[0]+' '+pv(sp.primary.get(b)[i])+' · '+sp.names[1]+' '+pv(sp.secondary.get(b)[i])+' · r '+fmt((b.radius_mm||[])[i],1)+' mm';
  el('prof-readout').textContent=label;let cx=[0,0,0];if(idx.length){for(const k of idx){cx[0]+=PV[3*k];cx[1]+=PV[3*k+1];cx[2]+=PV[3*k+2];}cx=cx.map(v=>v/idx.length);}setHighlight({vertexMask:mask,pointIndices:idx,markerAt:idx.length?cx:null,markerRadius:1.2,label});drawProfiles();}
 // ---- region statistics ----
 let regionSel=null,regionCenter=null;
 function regionBranch(){const sid=Number(el('region-branch').value);return (ARR.unroll_branches||[]).find(b=>b.segment_id===sid)||null;}
 function syncRegionSliders(){const b=regionBranch();if(!b)return;for(const id of ['region-s0','region-s1']){el(id).min=String(b.s_min_mm);el(id).max=String(b.s_max_mm);}if(Number(el('region-s0').value)<b.s_min_mm||Number(el('region-s0').value)>b.s_max_mm)el('region-s0').value=String(b.s_min_mm);if(Number(el('region-s1').value)>b.s_max_mm||Number(el('region-s1').value)<b.s_min_mm)el('region-s1').value=String(b.s_max_mm);el('region-s0-v').textContent=fmt(el('region-s0').value,0)+' mm';el('region-s1-v').textContent=fmt(el('region-s1').value,0)+' mm';}
 syncRegionSliders();el('region-branch').onchange=syncRegionSliders;for(const id of ['region-s0','region-s1'])el(id).oninput=()=>{el(id+'-v').textContent=fmt(el(id).value,0)+' mm';};
 el('region-mode').onchange=e=>{el('region-range').hidden=e.target.value!=='range';el('region-sphere').hidden=e.target.value!=='sphere';};
 el('region-radius').oninput=e=>{el('region-radius-v').textContent=fmt(e.target.value,1)+' mm';if(regionSel&&regionSel.kind==='sphere')applyRegion();};
 function renderRegionStats(){const box=el('region-stats');if(!regionSel){box.innerHTML='';return;}const F=fieldArrays(activeField),st=CORE.stats(F.p,regionSel.indices),area=PW.length?regionSel.indices.length/PW.length*AREA:0;box.innerHTML=`<b>区域</b><span>${esc(regionSel.label)}</span><b>字段</b><span>${esc(F.short)}</span><b>预测点</b><span>${st.n}</span><b>均值</b><span>${esc(fv(st.mean))}</span><b>p99</b><span>${esc(fv(st.p99))}</span><b>最大</b><span>${esc(fv(st.max))}</span><b>估计面积</b><span>${fmt(area/100,1)} cm²</span>`;}
 function applyRegion(){const mode=el('region-mode').value;let indices,mask,label,markerAt=null,markerRadius=1;
  if(mode==='range'){const b=regionBranch();if(!b){setTip('没有可用分支。');return;}const s0=Number(el('region-s0').value),s1=Number(el('region-s1').value);const pm=CORE.rangeMask(P_S,PS,b.segment_id,s0,s1);indices=CORE.maskIndices(pm);near();mask=new Uint8Array(MV.length/3);const lo=Math.min(s0,s1),hi=Math.max(s0,s1);for(let v=0;v<mask.length;v++)if(MS[v]===b.segment_id&&VS[v]>=lo&&VS[v]<=hi)mask[v]=1;label=branchName(b.segment_id)+' s '+fmt(lo,0)+'–'+fmt(hi,0)+' mm';regionSel={kind:'range',indices,label,segment_id:b.segment_id,s0:lo,s1:hi};}
  else{if(!regionCenter){setTip('请先勾选「点击壁面选中心」并在壁面上点击。');return;}const r=Number(el('region-radius').value)||8;const pm=CORE.sphereMask(PV,regionCenter,r);indices=CORE.maskIndices(pm);mask=CORE.sphereMask(MV,regionCenter,r);label='球形区域 r='+fmt(r,1)+' mm';markerAt=regionCenter;markerRadius=r;regionSel={kind:'sphere',indices,label,center:regionCenter.slice(),radius:r};}
  activeFinding=-1;profSel=null;renderFindingsList(-1);renderRegionStats();setHighlight({vertexMask:mask,pointIndices:null,markerAt,markerRadius,label});}
 el('region-apply').onclick=applyRegion;el('region-clear').onclick=()=>{regionCenter=null;el('region-center').textContent='尚未选点';clearHighlight();};
 // ---- pointer: hover readout + click picking ----
 const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();ray.params.Points.threshold=1;let pressAt=null;
 function pick(ev){const rect=renderer.domElement.getBoundingClientRect();mouse.set((ev.clientX-rect.left)/rect.width*2-1,-(ev.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(mouse,camera);return ray.intersectObject(mesh,false).find(h=>clipPlane.distanceToPoint(h.point)>=0)||null;}
 renderer.domElement.addEventListener('pointerdown',ev=>{pressAt=[ev.clientX,ev.clientY];lastViewName='custom';});
 renderer.domElement.addEventListener('wheel',()=>{fitName=null;},{passive:true});
 renderer.domElement.addEventListener('pointerup',ev=>{if(!pressAt||Math.hypot(ev.clientX-pressAt[0],ev.clientY-pressAt[1])>5){if(pressAt)fitName=null;pressAt=null;return;}pressAt=null;if(viewMode==='cloud')return;const hit=pick(ev);if(!hit)return;handlePick([hit.point.x,hit.point.y,hit.point.z]);});
 renderer.domElement.addEventListener('mousemove',ev=>{const rect=renderer.domElement.getBoundingClientRect();mouse.set((ev.clientX-rect.left)/rect.width*2-1,-(ev.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(mouse,camera);
  if(viewMode==='cloud'){const hit=ray.intersectObject(pts,false).find(h=>clipPlane.distanceToPoint(h.point)>=0);if(!hit)return;const i=hit.index,f=feature();setTip('预测点云 · '+pointValues(i)+'\n'+branchName(PS[i])+(f.mode==='wss'?'':' · '+f.label+' '+fq(f.values[i]))+'\n弧长 '+fmt(P_S[i],1)+' mm · 半径 '+fmt(P_R[i],1)+' mm · 位置 ('+[0,1,2].map(k=>fmt(PV[3*i+k],1)).join(', ')+') mm');return;}
  if(viewMode!=='wss')return;const hit=ray.intersectObject(mesh,false).find(h=>clipPlane.distanceToPoint(h.point)>=0);if(!hit){lastHover=null;setTip('移动鼠标到可见壁面读取数值',true);return;}lastHover=[hit.point.x,hit.point.y,hit.point.z];const f=hit.face;let best=f.a,bd=Infinity;for(const vi of [f.a,f.b,f.c]){const d=hit.point.distanceToSquared(new THREE.Vector3(MV[3*vi],MV[3*vi+1],MV[3*vi+2]));if(d<bd){bd=d;best=vi;}}
  // A partially unsupported triangle is marked invalid throughout to avoid a fictitious fill.
  const valid=[f.a,f.b,f.c].every(i=>Number.isFinite(MW[i]));let extra='';if(MT&&MT[best]){const bits=[];if(MT[best]&1)bits.push('插值无支撑');if(MT[best]&2)bits.push('表面粗糙');if(MT[best]&4)bits.push('几何越界');if(bits.length)extra='\n可信提示：'+bits.join('、');}
  setTip((valid?'壁面插值 · '+fieldArrays(activeField).short+' '+fv(fieldArrays(activeField).m[best]):'插值覆盖范围外：无有效读值')+'\n分支 '+branchName(MS[best])+(VS&&Number.isFinite(VS[best])?' · 弧长 '+fmt(VS[best],1)+' mm':'')+'\n最近面顶点 ('+[0,1,2].map(k=>fmt(MV[3*best+k],1)).join(', ')+') mm'+extra);});
 // ---- unroll plots ----
 const branchIndices=new Map();for(let i=0;i<PS.length;i++){if(!branchIndices.has(PS[i]))branchIndices.set(PS[i],[]);branchIndices.get(PS[i]).push(i);}
 const plotItems=[];for(const b of ARR.unroll_branches){const row=document.createElement('div');row.className='unroll-row';const label=document.createElement('small');label.textContent=branchName(b.segment_id)+' · s '+fmt(b.s_min_mm,1)+' – '+fmt(b.s_max_mm,1)+' mm';const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',label.textContent+' 的独立展开图');row.append(label,canvas);el('unroll-plots').appendChild(row);plotItems.push({row,canvas,...b});}
 function drawUnroll(){if(!el('unroll-details').open||!el('menu-stats').open)return;const UF=fieldArrays(activeField),UV=UF.p,ULG=logScale&&UF.log!==false;for(const p of plotItems){p.row.hidden=branchSel>=0&&p.segment_id!==branchSel;if(p.row.hidden)continue;const c=p.canvas,W=c.width=Math.max(1,Math.round(c.clientWidth*2)),H=c.height=180,ctx=c.getContext('2d');if(!ctx)continue;const pixelMax=new Float32Array(W*H);pixelMax.fill(-Infinity);for(const i of branchIndices.get(p.segment_id)||[]){const x=Math.max(0,Math.min(W-1,Math.floor((P_S[i]-p.s_min_mm)/(p.s_max_mm-p.s_min_mm||1)*(W-1)))),y=Math.max(0,Math.min(H-1,Math.floor((1-(P_TH[i]+Math.PI)/(2*Math.PI))*(H-1))));pixelMax[y*W+x]=Math.max(pixelMax[y*W+x],UV[i]);}ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);const img=ctx.getImageData(0,0,W,H);for(let i=0;i<pixelMax.length;i++)if(Number.isFinite(pixelMax[i])){const c=cmap(CORE.normalize(pixelMax[i],0,vmax,ULG));for(let k=0;k<3;k++)img.data[4*i+k]=c[k]*255;img.data[4*i+3]=255;}ctx.putImageData(img,0,0);
   if(profSel&&Number(profSel.b.segment_id)===p.segment_id){const s=(profSel.b.s_from_root_mm||[])[profSel.i];if(Number.isFinite(s)){const x=(s-p.s_min_mm)/(p.s_max_mm-p.s_min_mm||1)*(W-1);ctx.strokeStyle='#00c2d8';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}}}}
 window.__drawUnroll=drawUnroll;el('unroll-details').addEventListener('toggle',drawUnroll);
 function resize(){const w=view.clientWidth,h=view.clientHeight;if(!w||!h)return;requestRender();renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();
  // A fitted view (default / reset / 1–6) keeps filling the viewport after a layout change; a hand-placed camera stays put.
  if(fitName){const c=fittedView(fitName);if(c)setCamera(c,false);}
  requestLabelLayout();
  drawUnroll();drawProfiles();}window.addEventListener('resize',resize);if(window.ResizeObserver){const viewResizeObserver=new ResizeObserver(resize);viewResizeObserver.observe(view);}
 // ---- view state ----
 const VIEW_KEY='wss-report-view:'+(META.run_identity||META.input_sha256||META.case_id||'unknown');
 // View state v1.1 records every key explicitly so a saved figure never depends on the current defaults.
 // range.case_max = the coloured field's own p99, whatever the mode (the compare page picks the larger of two).
 function rangeState(){const r={mode:scaleMode,min:0,max:scaleMode==='fixed'?fixedMax:vmax};if(fixedField)r.field=fixedField;if(fixedShared)r.shared=true;
  r.case_max=Math.max(Number(fieldArrays(activeField).p99)||0,.01);return r;}
 function currentViewState(){return {schema_version:'wss-deploy.view/v1',family:'wall',run_identity:META.run_identity||null,camera:{position:camera.position.toArray(),target:controls.target.toArray(),up:camera.up.toArray()},field:activeField,mode:viewMode,colormap:cmapName,range:rangeState(),bands,log:logScale,units:UNIT,thresholds_pa:thresholdsPa.slice(),field_thresholds:Object.assign({},FIELD_THR),opacity,overlay:{trust:showTrust,contours:showContours,stagnation:showStag},slice:{clip:clipT},highlight:{branch:branchSel,top_pct:highlightPct,top:highlightEnabled,peak:el('showpeak').checked,feature:feat},
  labels:{findings:normLabelCount(LBL.findings),branches:!!LBL.branches,max_diameter:!!LBL.max_diameter},
  measurements:MEAS.map(m=>Object.assign({},m)),annotations:{items:ANNOTS.map(a=>Object.assign({},a))},probe_log:PROBES.map(r=>Object.assign({},r)),branches_hidden:Array.from(HIDDEN).sort((a,b)=>a-b),preset_name:PRESET_NAME,lang:LANG,export:exportOptions(),montage:montageState(),
  findings_review:ONLINE?undefined:{items:Object.assign({},REVIEW.items),added:(REVIEW.added||[]).slice()}};}
 function sameCamera(c){const tol=1e-6*Math.max(1,size),near=(a,b)=>a.every((v,k)=>Math.abs(+v-b[k])<=tol);return near(c.position,camera.position.toArray())&&near(c.target,controls.target.toArray());}
 function applyViewState(s){if(!s||typeof s!=='object')return false;if(s.colormap&&CMAPS[s.colormap]){cmapName=s.colormap;el('cmap').value=cmapName;}if(Number.isFinite(+s.bands)){bands=BAND_STEPS.includes(+s.bands)?+s.bands:0;el('bands').value=String(bands);}logScale=!!s.log;el('logscale').checked=logScale;if(s.units==='dyn'||s.units==='Pa'){UNIT=s.units;el('units').value=UNIT;}
  const t=CORE.validThresholds(s.thresholds_pa);if(t)thresholdsPa=t;
  if(s.field_thresholds&&typeof s.field_thresholds==='object'){for(const k of Object.keys(FIELD_THR))delete FIELD_THR[k];
    for(const [id,v] of Object.entries(s.field_thresholds)){const tv=CORE.validThresholds(v);if(tv&&id!=='wss'&&FIELD_SCHEMA[id])FIELD_THR[id]=tv;}}
  initThresholdInputs();if(s.range&&typeof s.range==='object'){if(s.range.mode==='fixed'&&Number.isFinite(+s.range.max)&&+s.range.max>0){scaleMode='fixed';fixedMax=+s.range.max;fixedField=typeof s.range.field==='string'&&s.range.field?s.range.field:null;fixedShared=s.range.shared===true;}else if(s.range.mode==='case'){scaleMode='case';fixedField=null;fixedShared=false;}el('scale-mode').value=scaleMode;el('fixedmax').value=String(+(fixedMax*uf()).toFixed(3));}
  if(Number.isFinite(+s.opacity)){opacity=Math.max(.2,Math.min(1,+s.opacity));el('opacity').value=String(opacity);}if(s.overlay&&typeof s.overlay==='object'){showTrust=!!s.overlay.trust&&!!MT;el('trust').checked=showTrust;showContours=!!s.overlay.contours;el('contours').checked=showContours;if(typeof s.overlay.stagnation==='boolean'){showStag=s.overlay.stagnation&&HAS_STAG();el('stag').checked=showStag;}}
  // A field another report cannot show (TAWSS pushed to a peak-only report) is ignored; the wall stays on its field.
  if(typeof s.field==='string'&&s.field!==activeField&&supportedFields().includes(s.field)){activeField=s.field;renderSchemaControls();}
  if(s.slice&&Number.isFinite(+s.slice.clip)){clipT=Math.max(0,Math.min(1,+s.slice.clip));el('clip').value=String(clipT*100);clipPlane.constant=clipT>=1?1e6:mn[2]+(mx[2]-mn[2])*clipT;el('clip-value').textContent=clipText();}
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
  renderPanel();renderFindingsList(activeFinding);rebuildContours();if(['wss','stl','cl','cloud'].includes(s.mode))setView(s.mode);else setView(viewMode);
  // An explicit camera (view state, #view= link, preset, camera link) wins over the fitted default; re-applying the current one does not.
  if(s.camera&&Array.isArray(s.camera.position)&&s.camera.position.length===3&&Array.isArray(s.camera.target)&&s.camera.target.length===3&&s.camera.position.every(Number.isFinite)&&s.camera.target.every(Number.isFinite)&&!sameCamera(s.camera)){setCamera(s.camera,false);fitName=null;}
  if(s.ui&&typeof s.ui==='object')applyUiState(s.ui);
  if(h.finding){const k=FINDINGS.findIndex(f=>f&&f.id===h.finding);if(k>=0&&k!==activeFinding)window.__selectFinding(k);}
  return true;}
 function applyUiState(u){if(u.menu&&MENUS.includes('menu-'+u.menu)){const id='menu-'+u.menu;for(const o of PANES[paneOf(id)])el(o).open=o===id;openCard(id);}
  else if(u.tab&&PANES[u.tab])showPane(u.tab);
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
 function capture(){renderer.render(scene,camera);const source=renderer.domElement,canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');canvas.width=1200;canvas.height=700;ctx.fillStyle='#eef2f7';ctx.fillRect(0,0,1200,700);const scale=Math.min(980/source.width,610/source.height),w=source.width*scale,h=source.height*scale;ctx.drawImage(source,(980-w)/2,45+(610-h)/2,w,h);ctx.fillStyle='#203049';ctx.font='22px sans-serif';ctx.fillText('WSS · '+String(META.case_id).slice(0,70),20,28);const f=viewMode==='cloud'?feature():viewMode==='cl'?{mode:'vir',lo:rmin,hi:rmax,label:'中心线半径 · mm'}:{mode:'wss',lo:0,hi:vmax,label:fieldTitle('zh')};if(viewMode!=='stl'){const k=f.mode==='wss'?af():1,LG=f.mode==='wss'&&logScale&&fieldArrays(activeField).log!==false;
  paintCanvasBar(ctx,1030,130,24,420,1.15,t=>f.mode==='wss'?cmap(t):vir(t),barSpec(f.mode,f.lo,f.hi,LG,k));ctx.fillStyle='#203049';ctx.font='18px sans-serif';ctx.textAlign='left';ctx.fillText(f.label,990,100);}ctx.fillStyle='#203049';ctx.font='17px sans-serif';const cs=isCycleField(activeField)?cycleStats(activeField):null;ctx.fillText(cs?fieldArrays(activeField).short+' · 单周期积分量（'+cyclePeriod()+' s）· 均值 '+fv(cs.mean)+' · p99 '+fv(cs.p99)+' · 最大值 '+fv(cs.max):'收缩期峰值帧 · p99 '+fqu(PEAK.p99_pa)+' '+ul()+' · 最大值 '+fqu(PEAK.max_pa)+' '+ul(),20,675);ctx.font='14px sans-serif';ctx.fillText('统计：预测点云；壁面：Gaussian 插值；黄色：'+(activeField==='wss'?'':fieldArrays(activeField).short+' ')+'全场最大值；'+(MIN_FIELDS.has(activeField)?'白色：'+fieldArrays(activeField).short+' 全场最小值；':'')+'粉色：高亮最高 '+fmt(highlightPct,1)+'%'+(hl.label?'；青色：'+hl.label:''),20,697);return canvas.toDataURL('image/png');}
 const fileStem=()=>'WSS-'+String(META.case_id).replace(/[^\w\u3400-\u9fff-]/g,'_');
 el('save-image').onclick=()=>{const a=document.createElement('a');a.href=capture();a.download=fileStem()+'.png';a.click();};
 // ---- C12 publication-grade export (the six-view button uses the very same path) ----
 let lastViewName='custom';
 function exportOptions(){return {scale:Number(el('exp-scale').value)||1,background:el('exp-bg').value||'white',colorbar:el('exp-cbar').value||'overlay',ui:!!el('exp-ui').checked,lang:LANG};}
 const legendTitle=lang=>COMMON.englishLabel('wss_short',lang)+' · '+ul();
 const colorbarOptions=lang=>{const LG=logScale&&fieldArrays(activeField).log!==false,sp=barSpec('wss',0,vmax,LG,af());
  return {stops:COMMON.colormapStops(cmapName),min:sp.lo,max:sp.hi,floor:.05*af(),bands:bands||0,log:LG,units:al(),title:fieldTitle(lang),lang,ticks:'fine',thresholds:sp.marks.map(m=>({v:m.v,label:m.label}))};};
 function loadImage(url){return new Promise((res,rej)=>{const img=new Image();img.onload=()=>res(img);img.onerror=()=>rej(new Error('PNG 解码失败'));img.src=url;});}
 const projectTo=(p,W,H)=>{const v=new THREE.Vector3(p[0],p[1],p[2]).project(camera);return [(v.x+1)/2*W,(1-v.y)/2*H,v.z];};
 function drawColorbar(ctx,W,H,k,lang){const w=Math.round(18*k),h=Math.round(H*.5),x=W-w-Math.round(64*k),y=Math.round(H*.12);
  const LG=logScale&&fieldArrays(activeField).log!==false;
  paintCanvasBar(ctx,x,y,w,h,k,t=>cmapRaw(CORE.quantize(t,bands)),barSpec('wss',0,vmax,LG,af()));
  ctx.fillStyle='#203049';ctx.font=Math.round(13*k)+'px sans-serif';ctx.textAlign='left';ctx.fillText(fieldTitle(lang),x-Math.round(4*k),y-Math.round(8*k));}
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
   // §17.3 / §21.4: the automatic labels (and the maximum-diameter ring) travel with the figure, laid out again at
   // the target resolution with the live priorities; moved chips get the same thin leader back to their point.
   for(const c of AUTO_CHIPS)if(Array.isArray(c.ring)&&c.ring.length>2){ctx.strokeStyle='#8e44ad';ctx.lineWidth=Math.max(1.5,1.5*k);ctx.beginPath();
     c.ring.forEach((q,i)=>{const at=projectTo(q,W,H);i?ctx.lineTo(at[0],at[1]):ctx.moveTo(at[0],at[1]);});ctx.closePath();ctx.stroke();}
   const cpad=Math.round(5*k),chh=Math.round(17*k),lay=exportLabelLayout(W,H,k,t=>({w:ctx.measureText(t).width+2*cpad,h:chh}));
   ctx.save();ctx.globalAlpha=.55;ctx.strokeStyle='#2b3a4a';ctx.fillStyle='#2b3a4a';ctx.lineWidth=Math.max(1,k);
   for(const L of lay){if(!L.moved||L.hidden)continue;const seg=leaderSegment(L.x,L.y,L.w,L.h,L.ax,L.ay);if(!seg)continue;
    ctx.beginPath();ctx.moveTo(seg[0],seg[1]);ctx.lineTo(seg[2],seg[3]);ctx.stroke();ctx.beginPath();ctx.arc(seg[2],seg[3],2*k,0,2*Math.PI);ctx.fill();}
   ctx.restore();
   for(const L of lay){if(L.hidden)continue;ctx.fillStyle=L.c.fill;ctx.strokeStyle=L.c.stroke;ctx.lineWidth=Math.max(1,k);ctx.beginPath();ctx.rect(L.x-L.w/2,L.y-L.h/2,L.w,L.h);ctx.fill();ctx.stroke();
    ctx.fillStyle='#203049';ctx.fillText(L.c.text,L.x,L.y+L.h/2-Math.round(4*k));}
   ctx.textAlign='left';}
  if(o.colorbar==='overlay')drawColorbar(ctx,W,H,k,lang);
  return canvas.toDataURL('image/png');}
 async function doExport(options){
  const o=Object.assign(exportOptions(),options||{}),lang=o.lang==='en'?'en':'zh';
  const r=COMMON.renderOffscreen({renderer,scene,camera,THREE,width:view.clientWidth||960,height:view.clientHeight||640,scale:o.scale,background:o.background});
  const url=await composeExport(r.dataURL,r.width,r.height,o,lang);
  const out={png_base64:String(url).replace(/^data:image\/png;base64,/,''),filename:o.filename||COMMON.exportFilename({case_id:META.case_id,view:lastViewName,field:activeField,scale:r.scale,ext:'png'}),width:r.width,height:r.height,downgraded:!!r.downgraded};
  if(o.colorbar==='svg')out.colorbar_svg=COMMON.colorbarSVG(colorbarOptions(lang));
  el('exp-note').textContent=(r.downgraded?'显卡不支持该倍率，已降级到 1×；':'')+'已导出 '+r.width+'×'+r.height+' PNG'+(o.colorbar==='svg'?' 与独立色标 SVG':'')+'。';
  return out;}
 const pngHref=r=>'data:image/png;base64,'+r.png_base64;
 el('exp-png').onclick=async()=>{el('exp-note').textContent='正在离屏渲染…';
  try{const r=await doExport();download(pngHref(r),r.filename);
   if(r.colorbar_svg)download('data:image/svg+xml;charset=utf-8,'+encodeURIComponent(r.colorbar_svg),r.filename.replace(/\.png$/,'-colorbar.svg'));}
  catch(err){el('exp-note').textContent='导出失败：'+(err&&err.message||err);}};
 el('exp-svg').onclick=()=>{const lang=el('exp-lang').value==='en'?'en':'zh';
  download('data:image/svg+xml;charset=utf-8,'+encodeURIComponent(COMMON.colorbarSVG(colorbarOptions(lang))),COMMON.exportFilename({case_id:META.case_id,view:'colorbar',field:activeField,scale:1,ext:'svg'}));
  el('exp-note').textContent='已导出当前色标 SVG（配色、范围、分段、对数与单位一致）。';};
 el('exp-lang').onchange=()=>{LANG=el('exp-lang').value==='en'?'en':'zh';GLOSS.setLang(LANG);renderMeasList();redrawOverlays();recolor();};
 el('views-export').onclick=async()=>{if(!FRAME)return;const views=fittedViews(),saved=currentViewState(),savedName=lastViewName,savedFit=fitName;
  el('view-note').textContent='正在导出六视角…';
  for(const entry of Object.entries(views)){setCamera(entry[1],false);tween=null;lastViewName=entry[0];
   try{const r=await doExport();download(pngHref(r),r.filename);}catch(err){el('view-note').textContent='六视角导出失败：'+(err&&err.message||err);break;}
   await new Promise(r=>setTimeout(r,220));}
  lastViewName=savedName;setCamera(saved.camera,false);fitName=savedFit;el('view-note').textContent='已按当前导图设置导出前/后/左/右/上/下六张 PNG。';};
 // ---- §15.11 multi-view montage and §15.8 one-page figures: both reuse the C12 offscreen path ----
 const VIEW_CN={front:'前视',back:'后视',left:'左视',right:'右视',top:'上视',bottom:'下视',current:'当前视角'};
 const VIEW_EN={front:'Front',back:'Back',left:'Left',right:'Right',top:'Top',bottom:'Bottom',current:'Current'};
 const viewCaption=(name,lang)=>(lang==='en'?VIEW_EN[name]:VIEW_CN[name])||name;
 const MONTAGE_VIEWS=['front','back','left','right','top','bottom','current'],montageBoxes={};
 const stdViews=()=>fittedViews();
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
   const canvas=COMMON.composeMontage({panels,columns:cfg.columns,colorbar:{image:bar},title:String(META.case_id)+' · '+(activeField==='wss'?COMMON.englishLabel('wss_short',lang):fieldArrays(activeField).short),font:Math.round(22*o.scale/2)});
   if(!canvas||typeof canvas.toDataURL!=='function')throw new Error('当前环境无法合成拼图画布');
   download(canvas.toDataURL('image/png'),COMMON.exportFilename({case_id:META.case_id,view:'montage',field:activeField,scale:o.scale,ext:'png'}));
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
    images.push({name,png_base64:r.png_base64,width:r.width,height:r.height,caption:viewCaption(name,lang)+' · '+fieldTitle(lang),view:name});}}
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
  // §19.2: the exported curve follows the profile plot (WSS, or TAWSS / OSI when the profile carries them).
  const sp=profSpec(),en=lang==='en',kk=sp.pa?k:1,scaled=a=>Array.from(a||[],v=>Number.isFinite(+v)?+v*kk:NaN),xf=xs.filter(Number.isFinite),series=[];
  const hline=(label,th,color)=>{if(xf.length>1)series.push({name:label,x:[Math.min.apply(null,xf),Math.max.apply(null,xf)],y:[th*kk,th*kk],color,dash:'5 5',width:1.2});};
  if(sp.id==='wss'){const w=b.wss||{};
   series.push({name:en?'Mean':'均值',x:xs,y:scaled(w.mean_pa),color:'#176caa'},{name:'p99',x:xs,y:scaled(w.p99_pa),color:'#c0392b'},{name:en?'Minimum':'最低',x:xs,y:scaled(w.min_pa),color:'#3f8f6b',dash:'7 5'});
   thresholdsPa.forEach((th,i)=>hline((en?['Low ','High ','Very high ']:['低 ','高 ','极高 '])[i]+fmt(th*k,1)+' '+ul(),th,['#8aa0b5','#d9963c','#b03a2e'][i]));}
  else if(sp.id==='tawss'){const w=b.tawss||{};
   series.push({name:en?'Mean':'均值',x:xs,y:scaled(w.mean_pa),color:'#176caa'},{name:en?'Minimum':'最低',x:xs,y:scaled(w.min_pa),color:'#3f8f6b',dash:'7 5'});
   hline((en?'Low TAWSS ':'低 TAWSS ')+String(+(sp.hlines[0]*k).toFixed(3))+' '+ul(),sp.hlines[0],'#8aa0b5');}
  else{const w=b[sp.id]||{};
   series.push({name:en?'Mean':'均值',x:xs,y:scaled(w.mean),color:'#176caa'},{name:'p90',x:xs,y:scaled(w.p90),color:'#c0392b'});
   hline((sp.short||'OSI')+' '+String(+(+sp.hlines[0]).toFixed(3))+(sp.short?' Pa⁻¹':''),sp.hlines[0],'#8aa0b5');}
  const kept=series.filter(s=>s.y.some(Number.isFinite));
  const yTitle=sp.id==='wss'?legendTitle(lang):sp.label;
  const stem=COMMON.safeName(META.case_id)+'_profile_'+COMMON.safeName(name);
  const out=[{key:sp.id,branch:name,filename:stem+'_'+sp.id+'.svg',
   svg:COMMON.profileSVG({series:kept,xLabel,yLabel:yTitle,title:String(META.case_id)+' · '+shown+' · '+yTitle})}];
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
 const presetCtx=()=>({meta:META,branchNames:BRANCH_NAMES,findings:FINDINGS,profiles:PROFILES,standardViews:fittedViews()});
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
 window.addEventListener('beforeprint',()=>{el('snapshot').src=capture();window.__menuOpen=MENUS.map(id=>el(id).open);window.__paneOpen=activePane;openCard('menu-stats');});window.addEventListener('afterprint',()=>{if(window.__menuOpen)MENUS.forEach((id,i)=>{el(id).open=window.__menuOpen[i];});if(window.__paneOpen)showPane(window.__paneOpen,{openFirst:false});resize();});
 // ---- camera link protocol (contract §6) ----
 let lastPost=0,lastCam='',applyingRemote=0;
 const sameOrigin=ev=>{try{return /^https?:$/.test(String(location.protocol||''))&&!!ev&&ev.origin===location.origin;}catch(_){return false;}};
 function postToParent(msg){try{if(window.parent&&window.parent!==window)window.parent.postMessage(msg,location.origin);}catch(_){}}
 window.addEventListener('message',ev=>{const d=ev&&ev.data;if(!d||typeof d!=='object')return;requestRender();
  if(d.type==='wss-view:set-camera'){if(!FRAME||!d.camera)return;try{const c=CORE.cameraFromAligned(d.camera,FRAME.rotation,FRAME.origin_mm);applyingRemote=performance.now()+150;setCamera(c,false);fitName=null;}catch(_){}return;}
  // §15.6: the compare page asks for the current display state; same-origin only and the camera is never touched.
  if(d.type==='wss-view:get-state'&&sameOrigin(ev))postToParent({type:'wss-view:state',request_id:d.request_id??null,family:'wall',state:currentViewState()});});
 function postCamera(){if(!FRAME||window.parent===window)return;const now=performance.now();if(now<applyingRemote)return;const key=camera.position.toArray().map(v=>v.toFixed(2)).join(',')+'|'+controls.target.toArray().map(v=>v.toFixed(2)).join(',');if(key===lastCam)return;if(now-lastPost<50){requestRender();return;}lastCam=key;lastPost=now;try{window.parent.postMessage({type:'wss-view:camera',family:'wall',run_identity:META.run_identity||null,camera:CORE.cameraToAligned({position:camera.position.toArray(),target:controls.target.toArray(),up:camera.up.toArray()},FRAME.rotation,FRAME.origin_mm)},'*');}catch(_){}}
 function frame(){let moving=false;if(tween){const u=Math.min(1,(performance.now()-tween.t0)/tween.ms),e=u<.5?2*u*u:1-Math.pow(-2*u+2,2)/2;camera.position.lerpVectors(tween.p0,tween.p1,e);controls.target.lerpVectors(tween.t0v,tween.t1,e);if(u>=1)tween=null;moving=true;}if(controls.update())moving=true;renderer.render(scene,camera);postCamera();const w=view.clientWidth,h=view.clientHeight;for(const it of labels){if(LAYOUT_KINDS.has(it.kind))continue;const v=it.obj.position.clone().project(camera);const show=it.obj.visible&&clipPlane.distanceToPoint(it.obj.position)>=0&&v.z>=-1&&v.z<=1;it.div.style.display=show?'':'none';if(show){it.div.style.left=(v.x+1)/2*w+'px';it.div.style.top=(1-v.y)/2*h+'px';}}const lk=labelKey(w,h);if(lk!==LBL_KEY){LBL_KEY=lk;layoutLiveLabels();}if(moving||tween)requestRender();}
 // Catch-all: any UI input or parent-page message may change what is drawn; rAF coalesces them into one frame.
 if(typeof document.addEventListener==='function')for(const type of ['input','change','click','keydown','pointerup'])document.addEventListener(type,requestRender,true);
 // ---- C13 parent-page protocol (contract §12.6); the §6 camera link above stays unchanged ----
 const MSG=COMMON.viewMessaging({family:'wall',runIdentity:META.run_identity||null,caseId:META.case_id||null,
  onApplyState:s=>{s=s&&typeof s==='object'?s:{};
   // A lone preset_name means "apply that preset", not "an empty state".
   if(typeof s.preset_name==='string'&&Object.keys(s).length===1)return applyPresetByName(s.preset_name);
   return applyViewState(Object.assign(currentViewState(),s));},
  onExport:o=>doExport(o||{})});
 // ---- A9 keyboard shortcuts (§19.9); every report document listens on its own, so each compare-page iframe works when focused ----
 function toggleLabels(){const on=normLabelCount(LBL.findings)>0||!!LBL.branches;LBL.findings=on?0:5;LBL.branches=!on;
  el('lbl-findings').value=labelSelectValue(LBL.findings);el('lbl-branches').checked=LBL.branches;redrawOverlays();setTip(on?'自动标注已关闭':'自动标注：前 5 条发现 + 分支名');}
 function probeKey(){const card=el('probe-card');if(!card.hidden){closeProbe();return;}if(lastHover)lockProbe(lastHover);else setTip('把鼠标移到壁面上再按 P 锁定探针');}
 const KEYS=['front','back','left','right','top','bottom'].map((name,k)=>({keys:[String(k+1)],label:VIEW_CN[name],group:'视角',when:()=>!!FRAME,run:()=>goView(name,true)}))
  .concat([{keys:['0','r'],label:'复位视角（前视撑满）',group:'视角',run:()=>resetCamera(true)}],
   supportedFields().length>1?[{keys:['f'],label:'循环字段：'+supportedFields().map(id=>fieldArrays(id).short).join(' → '),group:'显示',run:()=>cycleField()}]:[],
   [{keys:['l'],label:'自动标注开 / 关',group:'显示',run:toggleLabels},{keys:['p'],label:'探针：锁定悬停点 / 关闭',group:'工具',run:probeKey},
    {keys:['s'],label:'保存截图',group:'工具',run:()=>el('save-image').onclick()}]);
 let SHORTCUTS=null;try{SHORTCUTS=COMMON.installShortcuts(KEYS,{doc:document,title:'壁面报告快捷键'});}catch(_){SHORTCUTS=null;}
 // ---- first paint: branch boxes, empty lists, annotations, presets, then the default / hash priority ----
 initBranchVis();initMontage();setMeasureMode('');renderMeasList();renderProbeLog();renderBuiltinPresets();loadUserPresets();loadAnnotations();redrawOverlays();
 if(!CLGROUPS.length){for(const k of Object.keys(NEED))el('measure-'+k).disabled=true;el('measure-status').textContent='本报告没有可用中心线，测量不可用。';}
 if(ONLINE)apiJSON(jobURL('files/findings_review.json')).then(doc=>{if(!doc)return;
   REVIEW={items:(doc.items&&typeof doc.items==='object')?doc.items:{},added:Array.isArray(doc.added)?doc.added:[]};applyAdded();renderFindingsList(LAST_ACTIVE);
   el('findings-review-note').textContent='已载入审阅记录（确认 / 驳回即时保存）。';},()=>{});
 else el('findings-review-note').textContent='离线只读：判定只保存在本页的视图状态里。';
 resize();resetCamera(false);setView('wss');frame();
 let hashApplied=false;
 try{const s=CORE.parseViewHash(location.hash);if(s){applyViewState(s);hashApplied=true;el('view-note').textContent='已从链接恢复视图。';}}catch(_){el('view-note').textContent='链接中的视图状态无法解析。';}
 if(!hashApplied){try{const all=JSON.parse(localStorage.getItem('wss-report-defaults')||'{}');if(all&&all.wall)applyDefaults(all.wall);}catch(_){}}
 if(!hashApplied&&ONLINE)apiJSON(jobURL('/api/preferences')).then(j=>{const d=j&&j.preferences&&j.preferences.report_defaults&&j.preferences.report_defaults.wall;
   if(d&&!hashApplied){applyDefaults(d);el('defaults-note').textContent='已套用服务端默认口径。';}},()=>{});
 MSG.ready();
 // Changes made from here on are the user's (or a parent's push, which the compare page's echo guard ignores).
 setTimeout(()=>{ANNOUNCE=true;},0);
 // Node-only hook: the DOM-stub tests drive picking and state without a real pointer or WebGL context.
 if(typeof module!=='undefined')window.__wssTest={state:currentViewState,apply:applyViewState,pick:handlePick,mode:setMeasureMode,
  hide:setBranchHidden,presets:()=>PRESET_LIST,applyPreset:applyPresetByName,gloss:GLOSS,text:t=>{TEXT_HOOK=t;},lists:()=>({MEAS,PROBES,ANNOTS,REVIEW,hidden:Array.from(HIDDEN)}),export:doExport,
  montage:exportMontage,snapshots:buildSnapshots,profileSVGs,label:labelFor,camera:()=>({position:camera.position.toArray(),target:controls.target.toArray()}),
  autoLabels:()=>AUTO_CHIPS.map(c=>({kind:c.kind,text:c.text,ring:Array.isArray(c.ring)?c.ring.length:0})),panel:()=>el('panel').innerHTML,
  // v0.12: finish a camera tween at once, the fitted views, the field switch and the shortcut handle.
  settle:()=>{if(tween){camera.position.copy(tween.p1);controls.target.copy(tween.t1);tween=null;}controls.update();},renderPending:()=>RAF!==0,fitted:fittedView,fitName:()=>fitName,
  field:()=>activeField,peak:()=>({xyz:peak.position.toArray(),text:String(peakLabel.textContent||''),visible:!!peak.visible}),trough:()=>({xyz:trough.position.toArray(),text:String(troughLabel.textContent||''),visible:!!trough.visible}),controlsUp:()=>controlsUp,values:(id,which)=>Array.from(fieldArrays(id)[which==='m'?'m':'p']),segTypes:()=>[MS,PS,CS].map(a=>a.constructor.name),supported:()=>supportedFields(),
  layoutLabels:()=>{LBL_KEY='';return layoutLiveLabels();},labelRuns:()=>LBL_RUNS,frame:()=>frame(),leaders:()=>el('label-leaders').innerHTML,
  exportLayout:(W,H,k)=>exportLabelLayout(W,H,k,t=>estimateSize(t,12*k)).map(L=>({kind:L.c.kind,text:L.c.text,x:L.x,y:L.y,w:L.w,h:L.h,ax:L.ax,ay:L.ay,moved:L.moved,hidden:L.hidden})),colors:()=>Array.from(mcol.slice(0,12)),stag:()=>({on:showStag,mask:STAG_M?Array.from(STAG_M):null}),shortcuts:()=>SHORTCUTS,resize};
}
// Compact layout for narrow windows and the compare page's half-width iframes: adaptive menu columns,
// slim header, smaller colour bar, probe card along the bottom.  Runs even when WebGL is unavailable.
(function(){
 const BREAK=1180,SHORT={'save-image':['保存截图','截图'],'print-report':['打印 / 保存 PDF','打印']};let on=false,override=null;
 try{const v=localStorage.getItem('wss-report-compact');if(v==='on'||v==='off')override=v;}catch(_){}
 const menu=()=>el('menu');
 window.__openMenu=()=>{if(on)setOpen(true);};
 function setOpen(open){const m=menu();if(!m||!m.classList)return;m.classList.toggle('open',Boolean(open));const main=m.closest&&m.closest('main');if(main&&main.classList)main.classList.toggle('menu-open',Boolean(open));const b=el('menu-toggle');if(b&&b.setAttribute)b.setAttribute('aria-expanded',String(Boolean(open)));if(typeof window!=='undefined'&&typeof window.dispatchEvent==='function')window.dispatchEvent(new Event('resize'));}
 function apply(force){
 const want=override==='on'?true:override==='off'?false:(Number(window.innerWidth)||1e4)<BREAK;
 if(!force&&want===on)return;on=want;
  const m=menu();if(m&&m.style){m.style.transition='none';window.setTimeout(()=>{try{m.style.removeProperty('transition');}catch(_){}},0);}
  if(document.body&&document.body.classList)document.body.classList.toggle('compact',on);
  for(const [id,[long,short]] of Object.entries(SHORT)){const b=el(id);if(!b)continue;b.textContent=on?short:long;if(b.setAttribute)b.setAttribute('title',long);}
  const lt=el('layout-toggle');if(lt)lt.textContent=on?'完整布局':'紧凑布局';
  setOpen(false);
 }
 const bind=(id,fn)=>{const b=el(id);if(b&&b.addEventListener)b.addEventListener('click',fn);};
 bind('menu-toggle',()=>{const m=menu();setOpen(!(m&&m.classList&&m.classList.contains('open')));});
 bind('menu-tab',()=>setOpen(true));bind('menu-close',()=>setOpen(false));
 bind('layout-toggle',()=>{override=on?'off':'on';try{localStorage.setItem('wss-report-compact',override);}catch(_){}apply(true);});
 window.addEventListener('keydown',event=>{if(event.key==='Escape'&&on)setOpen(false);});
 window.addEventListener('resize',()=>apply(false));
 apply(true);
})();
try{initViewer();}catch(error){const d=document.createElement('div');d.className='fallback';d.textContent='无法初始化 3D 视图。请使用支持 WebGL 的浏览器并启用图形加速；左侧菜单中的统计和打印仍可使用。';el('view').appendChild(d);el('save-image').disabled=true;
 for(const id of ['menu-findings','menu-profiles','menu-region','menu-measure','menu-annot','menu-presets','menu-view'])el(id).hidden=true;
 syncTabs();
 // The batch-export parent still needs to know this frame is done, even with no WebGL at all.
 try{COMMON.viewMessaging({family:'wall',runIdentity:META.run_identity||null,caseId:META.case_id||null,webgl:false}).ready({webgl:false});}catch(_){}
 console.error('WSS viewer:',error);}
</script></body></html>'''
