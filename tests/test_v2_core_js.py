"""Node tests for the workspace v2 viewer core (WORKSPACE_V2_CONTRACT.md §6.1 / §6.2, route 4).

The core files are classic scripts that attach ``globalThis.WSSV2`` and can be ``require``d.  Pure parts
(formatting, colour scales, histogram, data decoding, cursor stations, orientation, state sanitising) run bare;
the wall adapter is exercised with the real three.js r128 build (geometry + raycasting need no WebGL context).
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
CORE_FILES = ["core_util.js", "core_data.js", "core_colormap.js", "core_colorbar.js", "core_viewer.js", "core_orientation.js",
              "core_cursor.js", "adapter_wall.js", "adapter_volume.js"]
ALL_FILES = CORE_FILES + ["dev_viewer.js"]


def _prelude(three: bool) -> str:
    lines = []
    if three:
        lines.append("globalThis.THREE=require(" + json.dumps(str(STATIC_DIR / "three.min.js")) + ");")
    lines.append("require(" + json.dumps(str(STATIC_DIR / "report_common.js")) + ");")
    for name in CORE_FILES:
        lines.append("require(" + json.dumps(str(V2 / name)) + ");")
    lines.append("const ns=globalThis.WSSV2, U=ns.util, C=globalThis.WssReportCommon;")
    return "\n".join(lines) + "\n"


def _node(script: str, three: bool = False) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the v2 core tests")
    program = _prelude(three) + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout)


# ------------------------------------------------------------------------------------------------ files & loading
def test_core_scripts_parse():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    for name in ALL_FILES:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)


def test_bundle_lists_core_files_in_order():
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    scripts = bundle["scripts"]
    assert scripts[:len(CORE_FILES)] == CORE_FILES
    assert set(bundle["dev"]) == {"dev_viewer.html", "dev_viewer.js"}
    html = (V2 / "dev_viewer.html").read_text(encoding="utf-8")
    assert "<script>" not in html                       # CSP workbench: no inline scripts
    positions = [html.index("/static/v2/" + name) for name in CORE_FILES + ["dev_viewer.js"]]
    assert positions == sorted(positions)
    for legacy in bundle["legacy_scripts"]:
        assert html.index("/static/" + legacy) < positions[0]


def test_modules_load_without_dom_and_register_namespaces():
    out = _node("""
      const names=['util','data','colormap','colorbar','viewer','orientation','cursor'].filter(k=>ns[k]);
      console.log(JSON.stringify({names,adapters:Object.keys(ns.adapters).sort(),offline:ns.env.offline,util:ns.util===U,
        tools:ns.adapters.wall.supports.tools.includes('pick'),webgl:ns.viewer.webglAvailable()}));
    """)
    assert out["names"] == ["util", "data", "colormap", "colorbar", "viewer", "orientation", "cursor"]
    assert out["adapters"] == ["volume", "wall"]
    assert out["offline"] is False and out["util"] is True and out["tools"] is True
    assert out["webgl"] is False                        # no document under Node: reported, not thrown


def test_offline_flag_follows_embedded_manifest():
    out = _node("""
      globalThis.document={getElementById:id=>id==='wssv2-manifest'?{textContent:'{}'}:null};
      const a=ns.env.offline;globalThis.document={getElementById:()=>null};
      console.log(JSON.stringify({a,b:ns.env.offline}));
    """)
    assert out == {"a": True, "b": False}


# ------------------------------------------------------------------------------------------------ formatting
def test_number_formatting_three_significant_digits_and_percent_rules():
    out = _node("""
      const v=[17.006,0.00177,1.572,9.996,15544.8,999.6,0,-0.0004,0.1,100.4,-3.14159,52.6,4.357];
      console.log(JSON.stringify({sig:v.map(x=>U.fmtSig(x)),missing:[NaN,null,undefined,Infinity,''].map(x=>U.fmtSig(x)),
        trim:[1,0.05,4.36,0.5,10].map(x=>U.fmtTrim(x)),
        pct:[0.408,0.0056,0,0.0003,0.996,1,0.12345,0.00999,NaN].map(x=>U.fmtPct(x)),
        val:[U.fmtValue(0.598,'Pa'),U.fmtValue(0.0888,'1'),U.fmtValue(2.03,'1/Pa'),U.fmtValue(NaN,'Pa'),U.fmtValue(1.2,'m/s')],
        range:[U.fmtRange(0,4.357,'Pa'),U.fmtRange(-1295,99.1,'Pa'),U.fmtRange(0,1,'Pa',{trim:true})],
        esc:U.escapeHtml('<a href="x">&\\'</a>'),num:[U.num(null),U.num(''),U.num('2'),U.num(false)]}));
    """)
    assert out["sig"] == ["17.0", "0.00177", "1.57", "10.0", "15545", "1000", "0", "-0.000400", "0.100", "100", "-3.14", "52.6", "4.36"]
    assert out["missing"] == ["—"] * 5                  # missing is never printed as 0
    assert out["trim"] == ["1", "0.05", "4.36", "0.5", "10"]
    assert out["pct"] == ["41%", "0.6%", "0%", "<0.1%", "99%", "100%", "12%", "1%", "—"]
    assert out["val"] == ["0.598 Pa", "0.0888", "2.03 Pa⁻¹", "—", "1.20 m/s"]
    assert out["range"] == ["0–4.36 Pa", "-1295 至 99.1 Pa", "0–1 Pa"]
    assert out["esc"] == "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;"
    assert out["num"][0] is None and out["num"][1] is None and out["num"][2] == 2 and out["num"][3] is None


def test_emitter_off_and_listener_errors_route_to_error_event():
    out = _node("""
      const e=U.emitter(),seen=[];const off=e.on('x',v=>seen.push(v));e.on('x',()=>{throw new Error('boom');});e.on('error',err=>seen.push('err:'+err.message));
      e.emit('x',1);off();e.emit('x',2);
      console.log(JSON.stringify({seen,count:e.count('x')}));
    """)
    assert out == {"seen": [1, "err:boom", "err:boom"], "count": 1}


def test_grid_index_matches_brute_force_and_honours_filters():
    out = _node("""
      let s=7;const rnd=()=>{s=(s*16807)%2147483647;return s/2147483647;};
      const n=3000,P=new Float32Array(3*n);for(let i=0;i<3*n;i++)P[i]=rnd()*80-40;
      const G=U.gridIndex(P,{cell:3});let bad=0,badF=0;
      for(let q=0;q<300;q++){const x=rnd()*90-45,y=rnd()*90-45,z=rnd()*90-45;
        let b=-1,bd=Infinity,bf=-1,bfd=Infinity;for(let i=0;i<n;i++){const d=(P[3*i]-x)**2+(P[3*i+1]-y)**2+(P[3*i+2]-z)**2;if(d<bd){bd=d;b=i;}if(i%2===0&&d<bfd){bfd=d;bf=i;}}
        if(G.nearest(x,y,z)!==b)bad++;if(G.nearest(x,y,z,{accept:i=>i%2===0})!==bf)badF++;}
      const far=G.nearest(500,500,500,{maxDist:5});
      console.log(JSON.stringify({bad,badF,far}));
    """)
    assert out == {"bad": 0, "badF": 0, "far": -1}


# ------------------------------------------------------------------------------------------------ colour scales
def test_rainbow_and_other_palettes_are_identical_to_report_common():
    out = _node("""
      const res={};
      for(const cm of ['rainbow','turbo','viridis','bwr']){const sc=ns.colormap.scale({range:[0,1],cmap:cm});let maxd=0;
        const vals=new Float32Array(101);for(let i=0;i<=100;i++)vals[i]=i/100;const out=new Float32Array(303);sc.fill(vals,out);
        for(let i=0;i<=100;i++){const a=sc.color(i/100),b=C.paletteRGB(cm,i/100);for(let k=0;k<3;k++){maxd=Math.max(maxd,Math.abs(a[k]-b[k]),Math.abs(out[3*i+k]-b[k]));}}
        res[cm]=maxd;}
      console.log(JSON.stringify(res));
    """)
    assert all(v < 1e-6 for v in out.values()), out   # float32 fill ≈ float64 formula; color() is exact


def test_scale_missing_clamping_bands_and_log():
    out = _node("""
      const sc=ns.colormap.scale({range:[0,4]});
      const b=ns.colormap.scale({range:[0,4],bands:4});
      const L=ns.colormap.scale({range:[0,4.36],log:true});
      const neg=ns.colormap.scale({range:[-1300,100],log:true});
      const cross=ns.colormap.scale({range:[0,100],log:true,fieldMin:-5});
      console.log(JSON.stringify({missing:sc.color(NaN).map(x=>Math.round(x*255)),nullMissing:sc.color(null).map(x=>Math.round(x*255)),
        above:sc.color(9),top:C.paletteRGB('rainbow',1),below:sc.color(-3),bottom:C.paletteRGB('rainbow',0),norm:[sc.norm(2),sc.norm(8),sc.norm(NaN)],
        band:[b.color(0.1),b.color(0.9),b.color(1.1),C.paletteRGB('rainbow',0.125),C.paletteRGB('rainbow',0.375)],
        log:{floor:L.floor,n0:L.norm(0.05),n1:L.norm(4.36),mid:+L.norm(Math.sqrt(0.05*4.36)).toFixed(6),desc:L.describe().text,ticks:L.ticks(6).map(t=>t.label)},
        neg:{log:neg.log,why:neg.logRejected},cross:{log:cross.log,why:cross.logRejected},
        allowed:[ns.colormap.logAllowed({range:[0,1]}).ok,ns.colormap.logAllowed({range:[-1,1]}).ok,ns.colormap.logAllowed({min:0,max:1,fieldMin:-2}).reason]}));
    """)
    assert out["missing"] == [167, 173, 179] and out["nullMissing"] == [167, 173, 179]   # --missing, never the 0 colour
    assert out["above"] == out["top"] and out["below"] == out["bottom"]
    assert out["norm"][0] == 0.5 and out["norm"][1] == 2 and out["norm"][2] is None
    band = out["band"]                                   # 4 bands over 0–4: each band takes its centre colour
    assert band[0] == band[1] == band[3] and band[2] == band[4] and band[0] != band[2]
    lg = out["log"]
    assert lg["floor"] == 0.05 and lg["n0"] == 0 and abs(lg["n1"] - 1) < 1e-12 and abs(lg["mid"] - 0.5) < 1e-6
    assert lg["desc"] == "0.05–4.36 · 对数" and lg["ticks"][0] == "0.05" and "1" in lg["ticks"]
    assert out["neg"]["log"] is False and "负值" in out["neg"]["why"]
    assert out["cross"]["log"] is False and "跨零" in out["cross"]["why"]
    assert out["allowed"][:2] == [True, False] and "跨零" in out["allowed"][2]


def test_resolve_adaptive_named_fixed_windows_and_null_p99():
    out = _node("""
      const f={id:'tawss',units:'Pa',display:{p99:4.36,thresholds:[0.4,4,7]},windows:[{id:'low',label:'低剪切窗',range:[0,1],provisional:true}]};
      const a=ns.colormap.resolve(f,{window:'adaptive'},{min:0.1});
      const n=ns.colormap.resolve(f,{window:'low'},{min:0.1});
      const x=ns.colormap.resolve(f,{window:{range:[5,1]}},{min:0.1});
      const u=ns.colormap.resolve(f,{window:'nope'},{min:0.1});
      // relative pressure: display.p99 null must not become 0; crossing zero → p1..p99, no log
      const p=ns.colormap.resolve({id:'pressure',units:'Pa',display:{p99:null}},{window:'adaptive',log:true},{min:-1400,p1:-1295,p99:99.1});
      console.log(JSON.stringify({a:[a.scale.range,a.window.kind,a.window.text],n:[n.scale.range,n.window.label,n.window.provisional,n.window.text,n.scale.ticks(5).map(t=>t.label)],
        x:[x.scale.range,x.window.kind,x.spec.window],u:[u.window.kind,!!u.window.fallback],p:[p.scale.range,p.scale.log,p.crossesZero,p.window.text]}));
    """)
    assert out["a"] == [[0, 4.36], "adaptive", "本例自适应 0–4.36 Pa"]
    assert out["n"][0] == [0, 1] and out["n"][1] == "低剪切窗" and out["n"][2] is True
    assert out["n"][3] == "低剪切窗（暂定） 0–1 Pa" and out["n"][4][-1] == "1"
    assert out["x"] == [[1, 5], "fixed", {"range": [1, 5]}]
    assert out["u"] == ["adaptive", True]
    assert out["p"] == [[-1295, 99.1], False, True, "本例自适应 -1295 至 99.1 Pa"]


def test_histogram_counts_out_of_range_missing_and_log_bins():
    out = _node("""
      const h=ns.colormap.histogram(new Float32Array([0.1,0.5,1,2,NaN,5,-1,3.99]),{bins:4,range:[0,4]});
      const m=ns.colormap.histogram(new Float32Array([0.1,0.5,1,2,NaN,5]),{bins:2,range:[0,4],mask:new Uint8Array([1,1,0,0,1,1])});
      const g=ns.colormap.histogram(new Float32Array([0.1,1,10]),{bins:2,range:[0.1,10],log:true});
      console.log(JSON.stringify({h,m:[m.n,m.nMissing,m.nAbove,m.counts],g:[g.edges.map(x=>+x.toFixed(6)),g.counts]}));
    """)
    h = out["h"]
    assert h["counts"] == [2, 1, 1, 1] and h["n"] == 7 and h["nMissing"] == 1 and h["nBelow"] == 1 and h["nAbove"] == 1
    assert h["edges"] == [0, 1, 2, 3, 4]
    assert out["m"] == [3, 1, 1, [2, 0]]
    assert out["g"] == [[0.1, 1, 10], [1, 2]]


def test_colorbar_svg_marks_histogram_thresholds_window_and_clipping():
    out = _node("""
      const sc=ns.colormap.scale({range:[0,4.36],units:'Pa'});
      const hist=ns.colormap.histogram(new Float32Array([0.1,0.2,0.3,3,5,NaN]),{bins:8,range:[0,4.36]});
      const info={label:'TAWSS <x>',units:'Pa',scale:sc,histogram:hist,histogramSource:'points',thresholds:[0.4,4,7],thresholdDirection:'below',
        windowLabel:{label:'低剪切窗',provisional:true},missingFraction:hist.nMissing/(hist.n+hist.nMissing)};
      const svg=ns.colorbar.toSVG(info),notes=ns.colorbar.noteLines(info);
      console.log(JSON.stringify({svg,notes}));
    """)
    svg, notes = out["svg"], out["notes"]
    assert svg.startswith("<svg") and "TAWSS &lt;x&gt;" in svg and "<x>" not in svg
    assert "低剪切窗（暂定）" in svg
    assert svg.count('stroke-width="2"') == 2                 # 0.4 and 4 inside the range, 7 outside
    assert "高于上限的值按上端色" in svg and "低于下限" not in svg   # arrow only on the clipped end
    kinds = [n["kind"] for n in notes]
    assert kinds == ["hist", "clip", "missing", "thr"]
    assert notes[0]["text"] == "灰柱：采样点分布" and notes[0]["sub"] == "5 点"
    assert notes[1]["sub"] == "高于上限 20%" and notes[2]["text"] == "缺失（灰）17%"


# ------------------------------------------------------------------------------------------------ data sources
DATA_FIXTURE = """
const f32=a=>new Uint8Array(new Float32Array(a).buffer);
const manifest={schema:'wss-deploy.v2-manifest/v1',data_version:'dv1',job:{id:'J1',display_name:'病例'},result:{run_identity:'RID',family:'wall'},
  geometry:{branches:[{id:0,name:'主动脉',parent:null,length_mm:10}]},
  fields:[{id:'wss',units:'Pa',kind:'scalar',components:1,arrays:{display:'f.wss.d',read:'f.wss.r'}},
          {id:'velocity',units:'m/s',kind:'vector',components:3,arrays:{display:null,read:'f.v.r'}}],
  arrays:{'f.wss.r':{dtype:'float32',shape:[4],bytes:16},'f.wss.d':{dtype:'float32',shape:[3],bytes:12},'short':{dtype:'float32',shape:[4],bytes:16},
          'badbytes':{dtype:'float32',shape:[4],bytes:12},'weird':{dtype:'float16',shape:[2],bytes:4},'f.v.r':{dtype:'float32',shape:[2,3],bytes:24},
          'seg':{dtype:'uint8',shape:[4],bytes:4}}};
const bodies={'f.wss.r':f32([1.5,NaN,3,4]),'f.wss.d':f32([1,2,3]),'short':f32([1,2,3]),'badbytes':f32([1,2,3]),'weird':new Uint8Array(4),'f.v.r':f32([1,2,2,NaN,0,0]),'seg':new Uint8Array([0,1,2,255])};
const calls=[];
function fakeFetch(url,opts){calls.push([url,opts&&opts.credentials]);
  const m=/\\/api\\/v2\\/jobs\\/([^/]+)\\/(manifest|arrays\\/(.+))$/.exec(url);
  if(m&&m[1]==='BUSY')return Promise.resolve({ok:false,status:409,text:()=>Promise.resolve('{"error":"not done","status":"running"}')});
  if(m&&m[2]==='manifest')return Promise.resolve({ok:true,status:200,json:()=>Promise.resolve(JSON.parse(JSON.stringify(manifest)))});
  const key=decodeURIComponent(m[3]);const b=bodies[key];
  if(!b)return Promise.resolve({ok:false,status:404,text:()=>Promise.resolve('{}')});
  return Promise.resolve({ok:true,status:200,arrayBuffer:()=>Promise.resolve(b.slice().buffer)});}
const settle=p=>p.then(v=>({ok:true,v}),e=>({ok:false,name:e.name,code:e.code,status:e.status,msg:String(e.message)}));
"""


def test_online_source_decodes_validates_and_never_pads():
    out = _node(DATA_FIXTURE + """
      const src=ns.data.createOnlineSource('J1',{fetch:fakeFetch});
      const r=await ns.data.loadResult(src);
      await r.preload(['f.wss.r','f.wss.d','f.v.r','seg']);
      const read=r.fieldArray('wss','read');
      const res={family:r.family,run:r.runIdentity,len:read.length,type:read.constructor.name,nan:Number.isNaN(read[1]),
        valueAt:[r.valueAt('wss',0),r.valueAt('wss',1),r.valueAt('wss',9),r.valueAt('velocity',0),r.valueAt('nope',0)],
        vec:r.vectorAt('velocity',1),seg:Array.from(r.array('seg')),shape:r.array('f.v.r').shape,branch:r.branch(0).name,
        creds:calls.every(c=>c[1]==='same-origin')};
      res.short=await settle(r.preload(['short']));
      res.badbytes=await settle(r.preload(['badbytes']));
      res.weird=await settle(r.preload(['weird']));
      res.undeclared=await settle(r.preload(['nokey']));
      try{r.array('short');res.notLoaded='no throw';}catch(e){res.notLoaded=e.code;}
      res.busy=await settle(ns.data.loadResult(ns.data.createOnlineSource('BUSY',{fetch:fakeFetch})));
      const m2=JSON.parse(JSON.stringify(manifest));m2.schema='other';
      res.schema=await settle(ns.data.loadResult({manifest:()=>Promise.resolve(m2),fetchArray:()=>Promise.reject(new Error('x'))}));
      console.log(JSON.stringify(res,(k,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v));
    """)
    assert out["family"] == "wall" and out["run"] == "RID" and out["len"] == 4 and out["type"] == "Float32Array"
    assert out["nan"] is True
    assert out["valueAt"] == [1.5, "NaN", "NaN", "NaN", "NaN"]        # NaN stays missing; vectors are not a scalar
    assert out["vec"] == ["NaN", 0, 0] and out["seg"] == [0, 1, 2, 255] and out["shape"] == [2, 3]
    assert out["branch"] == "主动脉" and out["creds"] is True
    assert out["short"]["ok"] is False and out["short"]["name"] == "DataError" and out["short"]["code"] == "length"
    assert out["badbytes"]["code"] == "bytes" and out["weird"]["code"] == "dtype" and out["undeclared"]["code"] == "undeclared"
    assert out["notLoaded"] == "not_loaded"
    assert out["busy"]["ok"] is False and out["busy"]["status"] == 409 and out["busy"]["code"] == "http"
    assert out["schema"]["code"] == "schema"


def test_embedded_source_reads_base64_scripts_and_offline_block():
    out = _node(DATA_FIXTURE + """
      const b64=u=>Buffer.from(u).toString('base64');
      const arrays={'f.wss.r':b64(bodies['f.wss.r']),'f.wss.d':b64(bodies['f.wss.d']).replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=+$/,''),'short':b64(bodies['short'])};
      const doc={getElementById:id=>({'wssv2-manifest':{textContent:JSON.stringify(manifest)},'wssv2-arrays':{textContent:JSON.stringify({arrays})},
        'wssv2-offline':{textContent:'{"hide_name":true,"exported_at":"2026-09-30"}'}})[id]||null};
      const src=ns.data.createEmbeddedSource(doc);
      const r=await ns.data.loadResult(src);
      await r.preload(['f.wss.r','f.wss.d']);
      const res={job:src.jobId,read:Array.from(r.fieldArray('wss','read')).map(String),disp:Array.from(r.fieldArray('wss','display')),offline:src.offline()};
      res.short=await settle(r.preload(['short']));
      res.missing=await settle(r.preload(['f.v.r']));
      const empty=ns.data.createEmbeddedSource({getElementById:()=>null});
      res.none=await settle(empty.manifest());
      console.log(JSON.stringify(res));
    """)
    assert out["job"] == "J1" and out["read"] == ["1.5", "NaN", "3", "4"] and out["disp"] == [1, 2, 3]
    assert out["offline"] == {"hide_name": True, "exported_at": "2026-09-30"}
    assert out["short"]["code"] == "length" and out["missing"]["code"] == "missing" and out["none"]["code"] == "no_embedded"


# ------------------------------------------------------------------------------------------------ cursor (D1)
CURSOR_FIXTURE = """
// Root 0: straight along +z from z=0 to z=100 (51 samples, 2 mm); rows stored in reverse so only the edges give the order.
// Children 1 / 2 leave the root end at ±x (21 samples each, 20 mm).  Manifest lengths: root 101 mm (polyline 100 mm).
const xyz=[],rad=[],seg=[],edges=[];let idx=0;
function add(sid,pts,r,reverse){const ids=[];for(const p of pts){xyz.push(...p);rad.push(r);seg.push(sid);ids.push(idx++);}
  for(let i=0;i<ids.length-1;i++)edges.push(ids[i],ids[i+1]);}
const rootPts=Array.from({length:51},(_,i)=>[0,0,2*i]);
add(0,rootPts,10);add(1,Array.from({length:21},(_,i)=>[i,0,100+0.2*i]),4);add(2,Array.from({length:21},(_,i)=>[-i,0,100+0.2*i]),4);
// shuffle root rows: reverse their storage but keep the edge direction proximal → distal
const perm=[];for(let i=0;i<51;i++)perm.push(50-i);for(let i=51;i<idx;i++)perm.push(i);
const inv=new Array(idx);perm.forEach((p,i)=>inv[p]=i);
const X=new Float32Array(3*idx),R=new Float32Array(idx),S=new Uint8Array(idx);
for(let i=0;i<idx;i++){const p=perm[i];X.set(xyz.slice(3*p,3*p+3),3*i);R[i]=rad[p];S[i]=seg[p];}
const E=new Uint32Array(edges.map(e=>inv[e]));
const centers=Array.from({length:51},(_,i)=>1+2*i);
const profiles={bin_mm:2,branches:[{segment_id:0,name:'主动脉',parent_id:-1,length_mm:101,s_local_mm:centers,s_from_root_mm:centers.map(c=>c),
  wss:{n:centers.map(()=>10),mean_pa:centers.map((c,i)=>i),p99_pa:centers.map((c,i)=>i+0.5),min_pa:centers.map(()=>null)},osi:{mean:centers.map(()=>0.2),p90:centers.map(()=>0.3)}},
  {segment_id:1,name:'左髂总',parent_id:0,length_mm:20.1,s_local_mm:[1,3],s_from_root_mm:[102,104],wss:{n:[3,0],mean_pa:[7,null]}}]};
const tables={cv:X,cr:R,ce:E,cs:S};
const manifest={schema:'wss-deploy.v2-manifest/v1',result:{family:'wall'},
  geometry:{centerline:{xyz:'cv',radius_mm:'cr',edges:'ce',segment:'cs'},branches:[{id:0,name:'主动脉',parent:null,length_mm:101},{id:1,name:'左髂总',parent:0,length_mm:20.1},{id:2,name:'右髂总',parent:0,length_mm:20.1}]},
  fields:[{id:'wss',units:'Pa',kind:'scalar',components:1,arrays:{}},{id:'osi',units:'1',kind:'scalar',components:1,arrays:{}},{id:'tawss',units:'Pa',kind:'scalar',components:1,arrays:{}}],
  analysis:{profiles},arrays:{cv:{dtype:'float32',shape:[idx,3],bytes:12*idx},cr:{dtype:'float32',shape:[idx],bytes:4*idx},ce:{dtype:'uint32',shape:[E.length/2,2],bytes:4*E.length},cs:{dtype:'uint8',shape:[idx],bytes:idx}}};
const src={manifest:()=>Promise.resolve(manifest),fetchArray:k=>Promise.resolve(new Uint8Array(tables[k].buffer).slice().buffer)};
const r=await ns.data.loadResult(src);await r.preload(ns.cursor.requiredArrays(r));
"""


def test_cursor_stations_follow_edges_and_scale_to_manifest_length():
    out = _node(CURSOR_FIXTURE + """
      const c=ns.cursor.create(r),events=[];c.on('change',s=>events.push(s&&+s.s_mm.toFixed(3)));
      const b=c.branches();
      const st=c.set(0,50.5);
      const res={branches:b.map(x=>[x.segmentId,+x.length_mm.toFixed(3),x.parent,x.children]),
        st:{s:st.s_mm,z:+st.xyz[2].toFixed(3),t:st.tangent.map(v=>+v.toFixed(3)),r:+st.radius_mm.toFixed(3),root:+st.s_from_root_mm.toFixed(3)}};
      const a=c.step(3);res.step=[+a.state.s_mm.toFixed(3),a.atEnd,a.moved];
      const e=c.step(1000);res.end=[+e.state.s_mm.toFixed(3),e.atEnd,e.children];
      const e2=c.step(1);res.endAgain=[e2.moved,e2.atEnd,e2.children];
      c.set(1,5);const s0=c.step(-1000);res.start=[s0.state.s_mm,s0.atStart,s0.parent];
      c.set(0,0);const s1=c.step(-1);res.rootStart=[s1.atStart,s1.parent,s1.moved];
      res.proj=c.stationFromPoint([0.5,0,40]);res.proj.s_mm=+res.proj.s_mm.toFixed(3);
      res.events=events;
      console.log(JSON.stringify(res));
    """)
    assert out["branches"] == [[0, 101, None, [1, 2]], [1, 20.1, 0, []], [2, 20.1, 0, []]]
    st = out["st"]
    assert st["s"] == 50.5 and st["z"] == 50 and st["t"] == [0, 0, 1] and st["r"] == 10 and st["root"] == 50.5
    assert out["step"] == [53.5, False, True]
    assert out["end"] == [101, True, [1, 2]]                        # stops at the branch end, offers the children
    assert out["endAgain"] == [False, True, [1, 2]]                 # never jumps branch by itself
    assert out["start"] == [0, True, 0]
    assert out["rootStart"] == [True, None, False]
    assert out["proj"]["segmentId"] == 0 and out["proj"]["s_mm"] == 40.4 and abs(out["proj"]["dist_mm"] - 0.5) < 1e-6
    assert out["events"][:3] == [50.5, 53.5, 101]


def test_cursor_readout_uses_profile_bins_only():
    out = _node(CURSOR_FIXTURE + """
      const c=ns.cursor.create(r);
      const none=c.readout('wss');
      c.set(0,10.5);const a=c.readout('wss'),o=c.readout('osi'),t=c.readout('tawss');
      c.set(0,12);const edge=c.readout('wss');
      c.set(1,2.5);const empty=c.readout('wss');
      console.log(JSON.stringify({none:[none.available,none.definition],a,o:o.stats,t:[t.available,t.stats,t.definition],edge:[edge.bin,edge.stats.mean],empty:[empty.bin,empty.n,empty.available,empty.stats]}));
    """)
    assert out["none"][0] is False
    a = out["a"]
    assert a["bin"] == [10, 12] and a["binIndex"] == 5 and a["stats"] == {"mean": 5, "p99": 5.5, "min": None} and a["n"] == 10
    assert a["available"] is True and "2 mm 分箱" in a["definition"] and "等权" in a["definition"] and a["units"] == "Pa"
    assert out["o"] == {"mean": 0.2, "p90": 0.3}
    assert out["t"][0] is False and out["t"][1] is None and "不在游标处另算" in out["t"][2]
    assert out["edge"] == [[12, 14], 6]
    assert out["empty"] == [[2, 4], 0, False, {"mean": None}]


# ------------------------------------------------------------------------------------------------ frame / orientation / state
def test_standard_views_follow_the_anatomical_frame():
    out = _node("""
      const R=[[0,1,0],[0,0,1],[1,0,0]];const frame={rotation:R,origin_mm:[0,0,0],orientation:{left_right:'inferred',anterior_posterior:'inferred'}};
      const V=ns.viewer;const d={};for(const n of V.VIEW_NAMES)d[n]=V.viewDirection(n,frame);
      const pts=new Float32Array([0,0,0, 20,200,0, 10,100,5]);   // wide on screen (camera right = +y here)
      const plain=V.insetFit(pts,{dir:d.anterior.dir,up:d.anterior.up,fov:30,width:800,height:600,insets:{}});
      const shifted=V.insetFit(pts,{dir:d.anterior.dir,up:d.anterior.up,fov:30,width:800,height:600,insets:{right:200}});
      console.log(JSON.stringify({d,noFrame:V.viewDirection('anterior',null),bad:V.viewDirection('oblique',frame),
        plainD:+plain.distance.toFixed(3),shiftedD:+shifted.distance.toFixed(3),dt:shifted.target.map((v,k)=>+(v-plain.target[k]).toFixed(3))}));
    """)
    d = out["d"]
    assert d["anterior"] == {"dir": [0, 0, 1], "up": [1, 0, 0]}      # looks along +posterior (R[1]); up = superior (R[2])
    assert d["posterior"]["dir"] == [0, 0, -1]
    assert d["left"] == {"dir": [0, -1, 0], "up": [1, 0, 0]}          # camera on the patient's left (R[0])
    assert d["superior"] == {"dir": [-1, 0, 0], "up": [0, 0, -1]}    # anterior at the top of the screen
    assert out["noFrame"]["dir"] == [0, 0, -1] and out["bad"] is None
    assert out["shiftedD"] > out["plainD"]                           # the colour-bar strip is not part of the fit
    assert out["dt"][1] > 0 and out["dt"][0] == 0 and out["dt"][2] == 0   # camera moves right so the vessel sits left of the bar


def test_orientation_layout_marks_inferred_axes():
    out = _node("""
      const R=[[1,0,0],[0,1,0],[0,0,1]];
      const frame={rotation:R,origin_mm:[0,0,0],orientation:{left_right:'inferred',superior_inferior:'derived',anterior_posterior:'inferred'}};
      const cam={position:[0,-100,0],target:[0,0,0],up:[0,0,1]};        // anterior view
      const L=ns.orientation.layout(frame,cam);
      const conf=ns.orientation.layout(Object.assign({},frame,{orientation:{left_right:'confirmed',anterior_posterior:'confirmed'}}),{position:[100,0,0],target:[0,0,0],up:[0,0,1]});
      const none=ns.orientation.layout(null,cam);
      const ax=k=>L.axes.find(a=>a.key===k);
      console.log(JSON.stringify({lr:[+ax('lr').x.toFixed(3),ax('lr').inferred,ax('lr').dashed],si:[+ax('si').y.toFixed(3),ax('si').inferred],ap:[ax('ap').visible,ax('ap').dashed],
        caption:L.caption,view:L.view,conf:[conf.caption,conf.view,conf.axes.find(a=>a.key==='ap').dashed],none:[none.available,none.caption],svg:ns.orientation.svg(L).includes('stroke-dasharray')}));
    """)
    assert out["lr"] == [1, True, False]                  # patient left on screen right in the anterior view
    assert out["si"] == [1, False]
    assert out["ap"] == [False, True]                     # pointing at the viewer: no labels
    assert out["caption"] == ["从前方看", "左右为推断，前后为推断"] and out["view"] == "anterior"
    assert out["conf"] == [["从左侧看"], "left", False]
    assert out["none"] == [False, ["无解剖坐标架，方位未知"]]
    assert out["svg"] is True


def test_state_sanitising_keeps_known_parts_and_time_index_zero():
    out = _node("""
      const ctx={fields:['wss','tawss'],windows:{tawss:['low']},branches:[0,1,2],family:'wall'};
      const good={field:'tawss',scale:{window:'low',log:true,bands:6,cmap:'viridis'},camera:{position:[1,2,3],target:[0,0,0],up:[0,0,1],fov:30},
        lighting:'soft',layers:{outline:false,trust:true,bogus:true},cursor:{segmentId:1,s_mm:12.5},selection:42,branches:[0,2],time_index:0};
      const a=ns.viewer.sanitizeState(JSON.parse(JSON.stringify(good)),ctx);
      const b=ns.viewer.sanitizeState({field:'osi',scale:{window:'high'},camera:{position:[NaN,0,0],target:[0,0,0]},lighting:'neon',cursor:{segmentId:9,s_mm:1},
        selection:-3,branches:[7],time_index:5},ctx);
      const c=ns.viewer.sanitizeState({field:'tawss',scale:{window:'high',bands:999,cmap:'jet'}},ctx);
      console.log(JSON.stringify({a,b,c,round:JSON.stringify(a)===JSON.stringify(JSON.parse(JSON.stringify(a)))}));
    """)
    a = out["a"]
    assert a["field"] == "tawss" and a["scale"] == {"window": "low", "log": True, "bands": 6, "cmap": "viridis"}
    assert a["camera"] == {"position": [1, 2, 3], "target": [0, 0, 0], "up": [0, 0, 1], "fov": 30}
    assert a["lighting"] == "soft" and a["layers"] == {"outline": False, "trust": True}
    assert a["cursor"] == {"segmentId": 1, "s_mm": 12.5} and a["selection"] == 42 and a["branches"] == [0, 2] and a["time_index"] == 0
    b = out["b"]
    assert b == {"time_index": 0, "branches": None, "ignored": ["time_index"]}
    assert out["c"]["scale"] == {"window": "adaptive", "log": None, "bands": 0, "cmap": "rainbow"}
    assert out["round"] is True


# ------------------------------------------------------------------------------------------------ wall adapter (three.js, no WebGL)
WALL_FIXTURE = """
// A 40 mm tube (radius 5) along +z split into two branches (z < 20: 0, else 1); 24 × 21 vertices.
const T=globalThis.THREE,nr=24,nz=21,V=[],F=[],MS=[],MT=[];
for(let j=0;j<nz;j++)for(let i=0;i<nr;i++){const a=2*Math.PI*i/nr;V.push(5*Math.cos(a),5*Math.sin(a),2*j);MS.push(2*j<20?0:1);MT.push(j===3?2:0);}
for(let j=0;j<nz-1;j++)for(let i=0;i<nr;i++){const a=j*nr+i,b=j*nr+(i+1)%nr,c=a+nr,d=b+nr;F.push(a,b,d,a,d,c);}
// prediction points: 2× denser rings slightly inside (radius 4.9); read value = 100 + point index, display value = z / 10
const PV=[],PS=[],PSR=[],READ=[];let k=0;
for(let j=0;j<2*nz-1;j++)for(let i=0;i<2*nr;i++){const a=Math.PI*i/nr,z=j;PV.push(4.9*Math.cos(a),4.9*Math.sin(a),z);PS.push(z<20?0:1);PSR.push(z);READ.push(k===7?NaN:100+k);k++;}
const nV=V.length/3,nP=PV.length/3,DISP=[];for(let v=0;v<nV;v++)DISP.push(v===5?NaN:V[3*v+2]/10);
const tables={mv:new Float32Array(V),mf:new Uint32Array(F),ms:new Uint8Array(MS),mt:new Uint8Array(MT),pv:new Float32Array(PV),ps:new Uint8Array(PS),p_s:new Float32Array(PSR),
  'f.wss.d':new Float32Array(DISP),'f.wss.r':new Float32Array(READ)};
const arrays={};for(const [kk,a] of Object.entries(tables)){const w=kk==='mv'||kk==='pv'||kk==='mf'?3:1;arrays[kk]={dtype:{Float32Array:'float32',Uint32Array:'uint32',Uint8Array:'uint8'}[a.constructor.name],shape:w>1?[a.length/w,w]:[a.length],bytes:a.byteLength};}
const manifest={schema:'wss-deploy.v2-manifest/v1',result:{family:'wall',run_identity:'R'},
  geometry:{display_mesh:{vertices:'mv',faces:'mf',segment:'ms',trust:'mt'},points:{xyz:'pv',segment:'ps',s_from_root_mm:'p_s'},centerline:{},branches:[{id:0,name:'A',parent:null},{id:1,name:'B',parent:0}]},
  fields:[{id:'wss',units:'Pa',kind:'scalar',components:1,arrays:{display:'f.wss.d',read:'f.wss.r'},display:{p99:2}}],arrays};
const src={manifest:()=>Promise.resolve(manifest),fetchArray:kk=>Promise.resolve(new Uint8Array(tables[kk].buffer).slice().buffer)};
const r=await ns.data.loadResult(src);
const A=ns.adapters.wall;
await r.preload(A.requiredArrays(r,A.defaultField(r,null)));
const root=new T.Group();
const h=A.build(r,{THREE:T,root,gfx:ns.viewer.gfx,requestRender(){},kind:'main'});
const res=ns.colormap.resolve(r.field('wss'),{window:'adaptive'},{min:0,p99:2});
h.setField('wss',res.scale);
const ray=(x,y,z)=>{const rc=new T.Raycaster();rc.set(new T.Vector3(x,y,z),new T.Vector3(-x,-y,0).normalize());return rc;};
"""


def test_wall_adapter_picks_read_value_of_nearest_prediction_point():
    out = _node(WALL_FIXTURE + """
      const p=h.pick(ray(50,0,10.3));
      const q=h.pick(ray(0,50,30.2));
      // brute-force nearest point to the surface hit
      const hit=p.hitXyz;let best=-1,bd=Infinity;for(let i=0;i<nP;i++){const d=(PV[3*i]-hit[0])**2+(PV[3*i+1]-hit[1])**2+(PV[3*i+2]-hit[2])**2;if(d<bd){bd=d;best=i;}}
      const colors=root.children[0].children[0].geometry.attributes.color.array;
      console.log(JSON.stringify({p:{pi:p.pointIndex,best,value:p.value,expect:100+best,src:p.valueSource,seg:p.segmentId,s:p.s_from_root_mm,xyz:p.xyz.map(v=>+v.toFixed(2))},
        qseg:q.segmentId,missColour:[colors[15],colors[16],colors[17]].map(x=>Math.round(x*255)),fields:h.fields,counts:h.counts,
        trustAttr:Array.from(root.children[0].children[0].geometry.attributes.aFlag.array.slice(3*24,3*24+2))}));
    """, three=True)
    p = out["p"]
    assert p["pi"] == p["best"] and p["value"] == p["expect"] and p["src"] == "read"   # value = read[nearest point], not a colour
    assert p["seg"] == 0 and p["s"] == 10
    assert out["qseg"] == 1
    assert out["missColour"] == [167, 173, 179]                                    # NaN display value → --missing grey
    assert out["fields"] == ["wss"] and out["counts"] == {"vertices": 504, "faces": 960, "points": 1968}
    assert out["trustAttr"] == [1, 1]


def test_wall_adapter_branch_visibility_masks_picking_and_histogram():
    out = _node(WALL_FIXTURE + """
      h.setBranchVisibility(new Set([1]));
      const miss=h.pick(ray(50,0,10.3)),hit=h.pick(ray(50,0,30.3));
      const hv=h.histogramValues('wss');let visible=0;for(let i=0;i<nP;i++)if(hv.mask[i])visible++;
      const faces=root.children[0].children[0].geometry.index.count/3;
      h.setBranchVisibility(null);
      const back=root.children[0].children[0].geometry.index.count/3,hv2=h.histogramValues('wss');
      h.highlight([0,1,2,3]);const hlColors=Array.from(root.children[0].children[0].geometry.attributes.color.array.slice(0,3));
      h.highlight(null);
      h.dispose();
      console.log(JSON.stringify({miss,hitSeg:hit&&hit.segmentId,visible,expected:PS.filter(s=>s===1).length,faces,back,scope:[hv.scope,hv2.scope,hv2.mask],
        dim:hlColors.length,children:root.children.length}));
    """, three=True)
    assert out["miss"] is None and out["hitSeg"] == 1
    assert out["visible"] == out["expected"] and out["scope"] == ["visible", "all", None]
    assert out["faces"] < out["back"] == 960
    assert out["children"] == 0                                                 # dispose removes the adapter group


def test_volume_adapter_colours_interior_points_and_reads_wall_pressure_vertices():
    out = _node("""
      const T=globalThis.THREE;
      const V=new Float32Array([0,0,0, 10,0,0, 0,10,0, 0,0,10]),F=new Uint32Array([0,1,2, 0,1,3, 0,2,3, 1,2,3]);
      const P=new Float32Array([1,1,1, 2,2,2, 3,1,1, 1,3,1]),W=new Uint8Array([0,0,1,0]),SEG=new Int32Array([0,0,1,1]);
      const PR=new Float32Array([-5,NaN,3,7]),WP=new Float32Array([1,2,3,4]);
      const tables={mv:V,mf:F,vpts:P,vis_wall:W,vseg:SEG,'f.pressure.r':PR,'f.wall_pressure.d':WP};
      const arrays={};for(const [k,a] of Object.entries(tables)){const w=k==='mv'||k==='vpts'||k==='mf'?3:1;arrays[k]={dtype:{Float32Array:'float32',Uint32Array:'uint32',Uint8Array:'uint8',Int32Array:'int32'}[a.constructor.name],shape:w>1?[a.length/w,w]:[a.length],bytes:a.byteLength};}
      const manifest={schema:'wss-deploy.v2-manifest/v1',result:{family:'volume'},geometry:{display_mesh:{vertices:'mv',faces:'mf',segment:null,trust:null},points:{xyz:'vpts',segment:'vseg',is_wall:'vis_wall'},centerline:{},streamlines:null,branches:[]},
        fields:[{id:'pressure',units:'Pa',kind:'scalar',components:1,arrays:{display:null,read:'f.pressure.r'}},{id:'wall_pressure',units:'Pa',kind:'scalar',components:1,arrays:{display:'f.wall_pressure.d',read:null}},
                {id:'velocity',units:'m/s',kind:'vector',components:3,arrays:{display:null,read:'f.velocity.r'}}],arrays};
      const r=await ns.data.loadResult({manifest:()=>Promise.resolve(manifest),fetchArray:k=>Promise.resolve(new Uint8Array(tables[k].buffer).slice().buffer)});
      const A=ns.adapters.volume;await r.preload(A.requiredArrays(r,'pressure').concat(['f.wall_pressure.d']));
      const root=new T.Group(),h=A.build(r,{THREE:T,root,gfx:ns.viewer.gfx,requestRender(){}});
      const sc=ns.colormap.scale({range:[-5,7]});h.setField('pressure',sc);h.setLayers(ns.viewer.normalizeLayers('volume'));
      const cloud=root.children[0].children.find(o=>o.name==='volume-points');
      const hv=h.histogramValues('pressure');
      const rc=new T.Raycaster();rc.set(new T.Vector3(1,1,-20),new T.Vector3(0,0,1));const pk=h.pick(rc);
      h.setField('wall_pressure',ns.colormap.scale({range:[1,4]}));
      const rc2=new T.Raycaster();rc2.set(new T.Vector3(2,2,-20),new T.Vector3(0,0,1));const wk=h.pick(rc2);
      console.log(JSON.stringify({fields:h.fields,def:A.defaultField(r,null),interior:cloud.geometry.attributes.position.count,
        nan:Array.from(cloud.geometry.attributes.color.array.slice(3,6)).map(x=>Math.round(x*255)),mask:Array.from(hv.mask),
        pick:pk&&[pk.pointIndex,pk.value,pk.valueSource],wall:wk&&[wk.pointIndex,wk.vertexIndex,wk.value,wk.valueSource],cloudVisible:cloud.visible}));
    """, three=True)
    assert out["fields"] == ["pressure", "wall_pressure"] and out["def"] == "pressure"   # vectors are not coloured directly
    assert out["interior"] == 3 and out["nan"] == [167, 173, 179] and out["mask"] == [1, 1, 0, 1]
    assert out["pick"] == [0, -5, "read"]
    assert out["wall"][0] is None and out["wall"][2] == out["wall"][1] + 1 and out["wall"][3] == "display"
    assert out["cloudVisible"] is False                                         # opaque coloured wall hides the cloud


def test_adaptive_window_follows_the_release_log_hint():
    # TAWSS / RRT / ECAP declare display.log_scale: the adaptive window is logarithmic by default (as in the
    # classic report); an explicit log:false, a named window, or a field that crosses zero stays linear.
    out = _node("""
      const f={id:'tawss', units:'Pa', display:{log_scale:true, p99:4.36}, windows:[{id:'low', label:'低剪切窗', range:[0,1], provisional:true}]};
      const st={p99:4.36, min:0.09, max:11.9};
      const a=ns.colormap.resolve(f, {window:'adaptive', log:null}, st);
      const lin=ns.colormap.resolve(f, {window:'adaptive', log:false}, st);
      const named=ns.colormap.resolve(f, {window:'low', log:null}, st);
      const plain=ns.colormap.resolve({id:'wss', units:'Pa', display:{p99:17}}, {window:'adaptive', log:null}, {p99:17, min:0.1, max:52});
      const pz=ns.colormap.resolve({id:'pressure', units:'Pa', display:{log_scale:true}}, {window:'adaptive', log:null}, {p99:900, p1:-800, min:-1200, max:1300});
      console.log(JSON.stringify({a:a.scale.log, aLabel:a.window.label, lin:lin.scale.log, named:named.scale.log, plain:plain.scale.log, pz:pz.scale.log}));
    """)
    assert out == {"a": True, "aLabel": "本例自适应", "lin": False, "named": False, "plain": False, "pz": False}
