"""Node tests for phase 3 lane 4 of the workspace v2: display and readouts (PHASE3_LANES.md §3 第 4 路).

Covers the classic numerics ported to core_contour.js (marchingTriangles / topThreshold, compared value for value with
report.py REPORT_CORE_JS, on synthetic data and on a real preview case when its report is present), the velocity log
scale (classic VolumeViewerCore.scaleEnds floor) through core_colormap's hint hook, the viewer's P3 lane 4 block (iso-
lines, highest-x % highlight, wall Z cut) on a stub renderer, the wall adapter's clipped picking, the volume adapter's
greyed untrusted points (classic desaturate), the hover readout model, WSS in dyn/cm² on the probe card and through
ws_display.toDisplay, section planes on the X / Y / Z axes with typed tilt / offset and the drag mode, the finding-label
count, the classic presets / default settings (/api/preferences), the reproduction-link hooks for lane 5, and the shell
wiring on the workspace stub harness.  Display only: every numeric check compares against the stored values.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from wss_deploy.paths import PROJECT_ROOT, STATIC_DIR
from wss_deploy.report import REPORT_CORE_JS
from tests.test_v2_display import V2, WALL, SLICE_TUBE, PATCH, _files, _node
from tests.test_v2_workspace_js import _run

LANE4_JS = ["core_contour.js", "adapter_wall.js", "adapter_volume.js", "core_colormap.js", "core_colorbar.js", "core_viewer.js", "ws_display.js",
            "ws_slice.js", "ws_probe.js", "ws_region.js", "ws_annot.js", "ws_shell.js", "ws_store.js"]
CORE_JS = None


def _core_file() -> str:
    """REPORT_CORE_JS written to a temp file once (the classic page's own numerics, as tests/test_report.py does)."""
    global CORE_JS
    if CORE_JS is None:
        fh = tempfile.NamedTemporaryFile("w", suffix="_report_core.js", delete=False, encoding="utf-8")
        fh.write(REPORT_CORE_JS)
        fh.close()
        CORE_JS = fh.name
    return CORE_JS


def _with_contour(extra=()):
    return ["core_contour.js"] + list(extra)


# ------------------------------------------------------------------------------------------------ files
def test_lane4_scripts_parse_and_core_contour_is_bundled_after_the_adapters():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    for name in LANE4_JS:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    s = bundle["scripts"]
    assert s.index("adapter_volume.js") < s.index("core_contour.js") < s.index("ws_icons.js")
    assert "core_contour.js" not in bundle["offline_exclude"]
    html = (V2 / "index.html").read_text(encoding="utf-8")
    assert html.index("adapter_volume.js") < html.index("core_contour.js")
    # the classic numerics live in the v2 module now, so S7 can drop the page code
    assert "marchingTriangles" in (V2 / "core_contour.js").read_text(encoding="utf-8")
    assert "经典报告" not in (V2 / "ws_annot.js").read_text(encoding="utf-8")   # W20: the classic-report note is gone


# ------------------------------------------------------------------------------------------------ classic numerics
def test_marching_triangles_and_top_threshold_equal_the_classic_core():
    out = _node("""
      const K=require(%s), N=ns.contour;
      // a bumpy open cylinder with NaN holes, values on the vertices
      const nr=40,nz=30,V=[],F=[],val=[];
      for(let j=0;j<nz;j++)for(let i=0;i<nr;i++){const a=2*Math.PI*i/nr;V.push(5*Math.cos(a),5*Math.sin(a),j*0.7);val.push(Math.abs(Math.sin(i*0.37)*3+Math.cos(j*0.21)*4+0.3*i/nr)*(j%%11===5&&i%%7===0?NaN:1));}
      for(let j=0;j<nz-1;j++)for(let i=0;i<nr;i++){const a=j*nr+i,b=j*nr+(i+1)%%nr,c=a+nr,d=b+nr;F.push(a,b,d,a,d,c);}
      const Vf=new Float32Array(V),Fu=new Uint32Array(F),vf=new Float32Array(val);
      vf[3]=2;vf[4]=2;                                      // vertices exactly on a level
      const levels=[0.4,2,4.5];
      const same=levels.map(l=>{const a=K.marchingTriangles(Vf,Fu,vf,l),b=N.marchingTriangles(Vf,Fu,vf,l);return a.length===b.length&&a.every((x,k)=>Object.is(x,b[k]))&&a.length>0;});
      const all=N.segments(Vf,Fu,vf,levels.concat([NaN]));
      let cat=[];levels.forEach(l=>{cat=cat.concat(Array.from(K.marchingTriangles(Vf,Fu,vf,l)));});
      const tops=[0.5,1,2.5,10].map(p=>[K.topThreshold(vf,p),N.topThreshold(vf,p)]);
      const ti=N.topIndices(vf,1);let n=0;for(let i=0;i<vf.length;i++)if(Number.isFinite(vf[i])&&vf[i]>=K.topThreshold(vf,1))n++;
      out({same,cat:cat.length===all.length&&cat.every((x,k)=>Object.is(x,all[k])),tops,count:[ti.indices.length,n],empty:N.topThreshold(new Float32Array([NaN]),1),
        clip:[N.clipLevel({min:[0,0,-10],max:[0,0,30]},0.25),N.clipLevel({min:[0,0,-10],max:[0,0,30]},1),N.clipLevel({min:[0,0,-10],max:[0,0,30]},null),N.clipLevel({min:[0,0,0],max:[0,0,1]},undefined)],
        pct:[N.cleanPct(0.1),N.cleanPct(3.3),N.cleanPct(99),N.cleanPct('x')]});
    """ % json.dumps(_core_file()), extra=_with_contour())
    assert out["same"] == [True, True, True] and out["cat"]
    assert all(a == b for a, b in out["tops"])
    assert out["count"][0] == out["count"][1] > 0 and out["empty"] == 0
    assert out["clip"] == [0.0, None, None, None]                              # null / undefined = no cut (not z = min)
    assert out["pct"] == [0.5, 3.5, 10, 1]


def _real_case_arrays():
    report = PROJECT_ROOT / "outputs/wss_deploy_preview_jobs/20260929_230442_a7273f1670e4/report.html"
    if not report.is_file():
        pytest.skip("preview case report not present on this machine")
    html = report.read_text(encoding="utf-8")
    meta = json.loads(re.search(r'<script[^>]*id="wss-report-meta"[^>]*>(.*?)</script>', html, re.S).group(1))
    arrays = json.loads(re.search(r'<script[^>]*id="wss-report-arrays"[^>]*>(.*?)</script>', html, re.S).group(1))
    fields = meta.get("fields") or {}
    tawss = (fields.get("tawss") or {}).get("array_key")
    if not tawss or tawss not in (arrays.get("xf") or {}):
        pytest.skip("preview case has no TAWSS arrays")
    return {"mv": arrays["mv"], "mf": arrays["mf"], "mw": arrays["mw"], "pw": arrays["pw"], "tm": arrays["xf"][tawss]["m"], "tp": arrays["xf"][tawss]["p"],
            "thr_wss": (meta.get("thresholds_pa") or [0.4, 4, 7]), "thr_tawss": ((fields.get("tawss") or {}).get("display") or {}).get("thresholds") or [0.4, 1, 1.5]}


def test_iso_lines_and_highlight_on_a_real_case_equal_the_classic_page():
    data = _real_case_arrays()
    tmp = Path(tempfile.mkdtemp()) / "arrays.json"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    out = _node("""
      const K=require(%s), N=ns.contour, A=JSON.parse(require('fs').readFileSync(%s,'utf8'));
      const dec=(s,T)=>{const b=Buffer.from(s,'base64');return new T(b.buffer,b.byteOffset,b.byteLength/T.BYTES_PER_ELEMENT);};
      const V=dec(A.mv,Float32Array),F=dec(A.mf,Uint32Array),MW=dec(A.mw,Float32Array),PW=dec(A.pw,Float32Array),TM=dec(A.tm,Float32Array),TP=dec(A.tp,Float32Array);
      const cmp=(vals,levels)=>{let cat=[];levels.forEach(l=>{cat=cat.concat(Array.from(K.marchingTriangles(V,F,vals,l)));});const s=N.segments(V,F,vals,levels);
        return {n:s.length/6,same:cat.length===s.length&&cat.every((x,k)=>Object.is(x,s[k]))};};
      const tops=[[PW,1],[PW,2.5],[TP,1],[TP,10]].map(([v,p])=>{const t=N.topIndices(v,p),c=K.topThreshold(v,p);let n=0;for(let i=0;i<v.length;i++)if(Number.isFinite(v[i])&&v[i]>=c)n++;
        return {same:t.threshold===c,count:t.indices.length,classic:n};});
      out({wss:cmp(MW,A.thr_wss),tawss:cmp(TM,A.thr_tawss),tops});
    """ % (json.dumps(_core_file()), json.dumps(str(tmp))), extra=_with_contour())
    assert out["wss"]["same"] and out["wss"]["n"] > 100
    assert out["tawss"]["same"] and out["tawss"]["n"] > 100
    assert all(t["same"] and t["count"] == t["classic"] > 0 for t in out["tops"])


# ------------------------------------------------------------------------------------------------ colour scale: velocity log (V7)
def test_speed_log_floor_is_the_classic_scale_ends_and_only_with_the_hint():
    out = _node("""
      const C=ns.colormap, speed={id:'speed',units:'m/s',kind:'scalar',components:1}, wss={id:'wss',units:'Pa'};
      const st={p99:1.3719,min:0,max:1.57};
      const lin=C.resolve(speed,{window:'adaptive',log:null},st).scale;
      C.setLogHint(f=>f&&f.id==='speed'?true:null);
      const lg=C.resolve(speed,{window:'adaptive',log:null},st).scale, w=C.resolve(wss,{window:'adaptive',log:null},{p99:20,min:0.01,max:50}).scale;
      const fixed=C.resolve(speed,{window:{range:[0,2]},log:null},st).scale;
      const small=C.resolve(speed,{window:'adaptive',log:null},{p99:0.1,min:0,max:0.2}).scale;
      C.setLogHint(null);
      out({lin:[lin.log,lin.floor],lg:[lg.log,lg.floor,lg.range],classic:VC.scaleEnds({min:0,max:1.3719,log:true}),w:[w.log,w.floor],fixed:fixed.log,small:[small.floor,VC.scaleEnds({min:0,max:0.1,log:true})[0]],
        direct:C.speedLogFloor(0,1.3719),hintAfter:C.logHint(speed)});
    """)
    assert out["lin"] == [False, None]
    assert out["lg"][0] is True and out["lg"][1] == pytest.approx(out["classic"][0]) and out["lg"][1] == pytest.approx(1.3719 / 200)
    assert out["w"] == [True, 0.05]                                            # wall shear keeps its own floor rule
    assert out["fixed"] is False                                               # a fixed window stays linear
    assert out["small"][0] == pytest.approx(out["small"][1]) == pytest.approx(0.001)   # 0.1 / 200 < 0.001 m/s: the absolute floor
    assert out["direct"] == pytest.approx(1.3719 / 200) and out["hintAfter"] is False


# ------------------------------------------------------------------------------------------------ viewer: iso-lines, highlight, Z cut
def test_viewer_p3_block_draws_iso_lines_highlight_and_the_wall_cut():
    out = _node(WALL + """
      const {r}=await wallResult();
      const box=new El('div'), v=ns.viewer.create(box,{kind:'main'});
      await v.setResult(r);
      await v.setField('wss',{window:'adaptive',log:null});
      const off=v.p3Display();
      const opt={contours:true,top:2,clip:null,stl:false,th:null};
      ns.viewer.displayProvider={bands:()=>0,thresholds:()=>opt.th,units:()=>null,flags:()=>({stl:opt.stl}),markers:()=>[],state:()=>null,restore(){},
        contours:()=>opt.contours,top:()=>opt.top,clip:()=>opt.clip};
      v.refreshDisplay();
      const on=v.p3Display();
      const D=r.array('f.wss.d'),R=r.array('f.wss.r'),V=r.array('mv'),F=r.array('mf');
      const ref=ns.contour.segments(V,F,D,[0.4,4,7]).length/6, refTop=ns.contour.topIndices(R,2).indices.length;
      opt.th=[1.5,2.5,4]; v.refreshDisplay();
      const custom=v.p3Display().contours, refCustom=ns.contour.segments(V,F,D,[1.5,2.5,4]).length/6;
      v.setBranchVisibility([0]);
      const PS=r.array('ps');const refHidden=ns.contour.topIndices(R,2,i=>PS[i]===0).indices.length;
      const hidden=v.p3Display();
      v.setBranchVisibility(null);
      opt.clip=0.5; v.refreshDisplay();
      const clip=v.p3Display().clipZ;
      // clipped picking: a ray at z = 30 (removed) finds nothing, at z = 10 (kept) the wall
      const ray=z=>{const rc=new T.Raycaster(new T.Vector3(50,0,z),new T.Vector3(-1,0,0));return v.pickAt?null:rc;};
      opt.stl=true; v.refreshDisplay();
      const stl=v.p3Display();
      opt.stl=false; opt.contours=false; opt.top=null; opt.clip=null; v.refreshDisplay();
      out({off,on,ref,refTop,custom,refCustom,hidden,refHidden,clip,stl,end:v.p3Display()});
    """, extra=_with_contour())
    assert out["off"] == {"contours": 0, "top": None, "clipZ": None}           # no provider: nothing drawn
    assert out["on"]["contours"] == out["ref"] > 0                             # the release thresholds
    assert out["on"]["top"]["count"] == out["refTop"] and out["on"]["top"]["pct"] == 2
    assert out["custom"] == out["refCustom"]                                   # the colour bar's custom thresholds
    assert out["hidden"]["top"]["count"] == out["refHidden"] < out["refTop"]   # hidden branches left out
    assert out["clip"] == pytest.approx(0 + 40 * 0.5)                          # tube z 0 … 40: classic min + t · extent
    assert out["stl"]["contours"] == 0 and out["stl"]["top"] is None           # input-STL view: geometry only
    assert out["end"] == {"contours": 0, "top": None, "clipZ": None}


def test_wall_adapter_skips_the_cut_part_when_picking_and_reports_the_face():
    out = _node(WALL + """
      const {r}=await wallResult();
      const A=ns.adapters.wall;
      await r.preload(A.requiredArrays(r,'wss'));
      const root=new T.Group(), h=A.build(r,{THREE:T,root,gfx:ns.viewer.gfx,requestRender(){},kind:'main'});
      h.setField('wss',ns.colormap.scale({range:[0,5]}));h.setLayers(ns.viewer.normalizeLayers('wall'));
      root.updateMatrixWorld(true);
      const ray=z=>new T.Raycaster(new T.Vector3(50,0.3,z),new T.Vector3(-1,0,0));
      const before=[h.pick(ray(30)),h.pick(ray(10))].map(p=>p?{face:p.face,z:+p.hitXyz[2].toFixed(3)}:null);
      h.setPickClip({normal:[0,0,-1],origin:[0,0,20]});
      const after=[h.pick(ray(30)),h.pick(ray(10)),h.pickSurface(ray(30)),h.pickSurface(ray(10))].map(p=>p?true:null);
      h.setPickClip(null);
      const g=root.children[0], outline=g.children.find(o=>o.name==='wall-outline'), hatch=g.children.find(o=>o.name==='wall-trust'), stag=g.children.find(o=>o.name==='wall-stagnation');
      out({before,after,back:Boolean(h.pick(ray(30))),clip:[outline.material.clipping,hatch.material.clipping,stag.material.clipping],
        chunks:[/clipping_planes_vertex/.test(outline.material.vertexShader),/clipping_planes_fragment/.test(hatch.material.fragmentShader)]});
    """)
    assert out["before"][0] and out["before"][1] and len(out["before"][0]["face"]) == 3
    assert out["after"] == [None, True, None, True]                            # only the kept part answers
    assert out["back"] is True
    assert out["clip"] == [True, True, True] and out["chunks"] == [True, True]


# ------------------------------------------------------------------------------------------------ volume: greyed untrusted points (V12)
VOLUME_TRUST = r"""
function volumeTrust() {
  const V=new Float32Array([0,0,0, 10,0,0, 0,10,0, 0,0,40, 10,0,40, 0,10,40]),F=new Uint32Array([0,1,4, 0,4,3, 1,2,5, 1,5,4, 2,0,3, 2,3,5]);
  const P=[],W=[],SEG=[],SP=[],PR=[],TR=[];
  for(let z=0.5;z<40;z+=2)for(let x=1;x<4;x+=1){P.push(x,2,z);W.push(0);SEG.push(0);SP.push(0.1+0.02*z);PR.push(100-z);TR.push(z<10?8:z>30?16:z>20&&x===2?4:z>12&&z<14?1:0);}
  const tables={mv:V,mf:F,vpts:new Float32Array(P),vis_wall:new Uint8Array(W),vseg:new Int32Array(SEG),vtrust:new Uint8Array(TR),'f.pressure.r':new Float32Array(PR),'f.speed.r':new Float32Array(SP)};
  const arrays={};for(const [k,a] of Object.entries(tables)){const w=k==='mv'||k==='vpts'||k==='mf'?3:1;arrays[k]={dtype:{Float32Array:'float32',Uint32Array:'uint32',Uint8Array:'uint8',Int32Array:'int32'}[a.constructor.name],shape:w>1?[a.length/w,w]:[a.length],bytes:a.byteLength};}
  const manifest={schema:'wss-deploy.v2-manifest/v1',result:{family:'volume',run_identity:'RT'},geometry:{display_mesh:{vertices:'mv',faces:'mf'},points:{xyz:'vpts',segment:'vseg',is_wall:'vis_wall',trust:'vtrust'},centerline:{},branches:[{id:0,name:'主动脉'}]},
    fields:[{id:'pressure',units:'Pa',kind:'scalar',components:1,arrays:{display:null,read:'f.pressure.r'}},{id:'speed',units:'m/s',kind:'scalar',components:1,arrays:{display:null,read:'f.speed.r'}}],arrays,
    analysis:{trust:{bits:{'1':'interpolation_uncovered','4':'geometry_out_of_range','8':'low_sample_support','16':'near_opening'},fractions:{interpolation_uncovered:0,geometry_out_of_range:0,low_sample_support:0.0145,near_opening:0.0634},
      sources:[{bit:1,label:'插值无支撑',rule:'R1',support:'vertices'},{bit:4,label:'几何越界',rule:'R4',support:'interior_points'},{bit:8,label:'采样支撑弱',rule:'R8',support:'interior_points'},{bit:16,label:'邻近切口',rule:'R16',support:'interior_points'}]}}};
  return ns.data.loadResult({manifest:()=>Promise.resolve(manifest),fetchArray:k=>Promise.resolve(new Uint8Array(tables[k].buffer).slice().buffer)}).then(r=>({r,tables,TR}));
}
"""


def test_volume_adapter_greys_untrusted_interior_points_like_the_classic_desaturate():
    out = _node(VOLUME_TRUST + """
      const {r,TR}=await volumeTrust();
      const A=ns.adapters.volume;
      await r.preload(A.requiredArrays(r,'speed').concat(A.optionalArrays(r)));
      const root=new T.Group(), h=A.build(r,{THREE:T,root,gfx:ns.viewer.gfx,requestRender(){},fov:30});
      const sc=ns.colormap.scale({range:[0,1]});
      h.setField('speed',sc); const L=ns.viewer.normalizeLayers('volume'); h.setLayers(L);
      const cloud=root.children[0].children.find(o=>o.name==='volume-points'), col=()=>Array.from(cloud.geometry.attributes.color.array);
      const plain=col(); const ret=h.setLayers(Object.assign({},L,{trust:true})); const grey=col(), S=r.array('f.speed.r');
      let ok=true, n=0;
      for(let i=0;i<TR.length;i++){const c=sc.color(S[i]),bad=(TR[i]&28)!==0;if(bad)n++;
        for(let k=0;k<3;k++){const want=bad?c[k]*0.3+0.62*0.7:c[k];if(Math.abs(grey[3*i+k]-want)>1e-6)ok=false;}}
      h.setLayers(L);
      out({ok,n,count:h.trustGrey(),trust:ret.trust,back:col().every((x,k)=>x===plain[k]),rows:ns.display.trustRows(r.manifest)});
    """, extra=["ws_display.js"])
    assert out["ok"] and out["n"] > 0 and out["trust"] is True and out["back"]
    assert out["count"] == 0                                                   # the trust layer is off again
    rows = out["rows"]
    assert [x["label"] for x in rows] == ["插值无支撑", "几何越界", "采样支撑弱", "邻近切口"]
    assert rows[2]["fraction"] == pytest.approx(0.0145) and rows[2]["support"] == "interior_points" and rows[0]["support"] == "vertices"


# ------------------------------------------------------------------------------------------------ units: WSS in dyn/cm² (W8)
def test_wss_display_unit_is_the_classic_factor_and_reaches_the_numbers():
    out = _node(WALL + """
      require(%s); require(%s);
      const D=ns.display;
      const before={td:D.toDisplay(4.2,'Pa','wall'),find:D.toDisplay(4.2,'Pa',null),vol:D.toDisplay(4.2,'Pa','volume'),osi:D.toDisplay(0.3,'1','wall'),rrt:D.toDisplay(5,'1/Pa','wall')};
      D.setUnit('wss','dyn/cm²');
      const after={td:D.toDisplay(4.2,'Pa','wall'),find:D.toDisplay(4.2,'Pa',null),vol:D.toDisplay(4.2,'Pa','volume'),osi:D.toDisplay(0.3,'1','wall'),rrt:D.toDisplay(5,'1/Pa','wall'),
        unit:D.displayUnit('wss'),kinds:[D.displayKind({units:'Pa'},'wall'),D.displayKind({units:'Pa'},'volume'),D.displayKind({units:'m/s'},'volume'),D.displayKind({units:'1/Pa'},'wall')],
        prefs:Object.assign({},D.prefs().units),factor:D.unitFactor('wss','dyn/cm²')};
      const {r}=await wallResult(); await r.preload(ns.probe.requiredArrays(r).filter(k=>r.declared(k)).concat(['f.wss.r','f.tawss.r','f.osi.r']));
      const pb={kind:'wall',family:'wall',index:5,hit:[4.9,0,0],section:null,fields:{wss:r.array('f.wss.r'),tawss:r.array('f.tawss.r'),osi:r.array('f.osi.r')}};
      const card=ns.probe.card(r,pb,{ui:ns.ui,onRecord(){},onClose(){},field:'wss'}).textContent;
      D.setUnit('wss','Pa');
      const cardPa=ns.probe.card(r,pb,{ui:ns.ui,onRecord(){},onClose(){},field:'wss'}).textContent;
      out({before,after,card,cardPa,wss5:r.array('f.wss.r')[5]});
    """ % (json.dumps(str(V2 / "ws_ui.js")), json.dumps(str(V2 / "ws_probe.js"))), extra=["ws_display.js"])
    b, a = out["before"], out["after"]
    assert b["td"] == {"value": 4.2, "units": "Pa"} and b["find"]["units"] == "Pa"
    assert a["td"]["value"] == pytest.approx(42) and a["td"]["units"] == "dyn/cm²"     # classic CORE.UNITS dyn: × 10
    assert a["find"]["value"] == pytest.approx(42)                             # a wall finding (no family) follows too
    assert a["vol"] == {"value": 4.2, "units": "Pa"}                           # volume pressure keeps its own unit
    assert a["osi"]["units"] == "1" and a["rrt"] == {"value": 5, "units": "1/Pa"}   # classic isPa: only Pa stresses
    assert a["kinds"] == ["wss", "pressure", "velocity", None] and a["factor"] == 10
    assert a["prefs"]["wss"] == "dyn/cm²"                                      # in prefs().units: the repro link carries it (lane 5 d.units)
    assert "dyn/cm²" in out["card"] and f"{out['wss5'] * 10:.3g}" in out["card"]
    assert "dyn/cm²" not in out["cardPa"] and f"{out['wss5']:.3g}" in out["cardPa"]


# ------------------------------------------------------------------------------------------------ hover readout (W41, V17)
def test_hover_readout_reads_the_wall_colour_value_and_the_interior_point():
    out = _node(WALL + VOLUME_TRUST + """
      require(%s); require(%s);
      const {r}=await wallResult({trust:{sources:[{bit:2,label:'表面粗糙'}]}}); await r.preload(['mv','mf','pv','ps','f.wss.d','f.wss.r']);
      const D=r.array('f.wss.d');
      const e={vertexIndex:30,face:[30,31,54],segmentId:1,s_from_root_mm:12.34,trust:2,screen:{x:5,y:5}};
      const wall=ns.probe.hoverModel(r,e,'wss');
      const nan=ns.probe.hoverModel(r,Object.assign({},e,{face:[30,31,7]}),'wss');   // classic: a partly unsupported triangle has no value
      D[7]=NaN; const nan2=ns.probe.hoverModel(r,Object.assign({},e,{face:[30,31,7]}),'wss');
      const {r:rv}=await volumeTrust(); await rv.preload(['mv','mf','vpts','vseg','vis_wall','vtrust','f.speed.r','f.pressure.r']);
      const vol=ns.probe.hoverModel(rv,{pointIndex:0,segmentId:0,s_from_root_mm:3,values:{speed:0.11,pressure:99.5}},'speed');
      ns.display.setUnit('pressure','mmHg');
      const volMm=ns.probe.hoverModel(rv,{pointIndex:0,segmentId:0,s_from_root_mm:3,values:{speed:0.11,pressure:99.5}},'pressure');
      out({wall,nan,nan2,dv:D[30],vol,volMm,none:ns.probe.hoverModel(r,null,'wss'),marker:ns.probe.hoverModel(r,{marker:'F1'},'wss')});
    """ % (json.dumps(str(V2 / "ws_ui.js")), json.dumps(str(V2 / "ws_probe.js"))), extra=["ws_display.js"])
    w = out["wall"]
    assert w[0]["kind"] == "value" and w[0]["text"].startswith("WSS ") and f"{out['dv']:.3g}" in w[0]["text"]
    assert w[1] == {"text": "B · 距入口 12.3 mm", "kind": "where"}
    assert w[2] == {"text": "可信提示：表面粗糙", "kind": "trust"} and w[3]["kind"] == "xyz"
    assert out["nan"][0]["kind"] == "value" and out["nan2"][0]["text"] == "插值覆盖范围外：无读值"
    v = out["vol"]
    assert [x["kind"] for x in v[:2]] == ["value", "also"] and "0.11" in v[0]["text"] and "相对压力 99.5 Pa" == v[1]["text"]
    assert v[2]["text"] == "主动脉 · 距入口 3.00 mm" and v[3]["text"] == "可信提示：采样支撑弱"   # vtrust bit 8 of point 0
    assert out["volMm"][0]["text"].startswith("相对压力 ") and "mmHg" in out["volMm"][0]["text"]
    assert out["none"] is None and out["marker"] is None


# ------------------------------------------------------------------------------------------------ section tool (V26, V27, V28, V7)
def test_section_axis_planes_typed_tilt_drag_mode_and_speed_log():
    out = _node(SLICE_TUBE + """
      const SL=ns.slice, r=tube(), M=SL.model(r);
      // classic currentPlane for slice-basis x / y / z: bounding-box centre, the axis normal, min + fraction × extent
      const planes=['x','y','z'].map(b=>SL.planeOf({basis:b,fraction:0.25,pitch:0,yaw:0,offU:0,offV:0},M));
      const s=SL.create(null,r,{hint:{xyz:[0,0,12]}});
      s.recomputeNow();
      const o0=s.comp().plane.origin.slice();
      s.setBasis('z'); s.recomputeNow();
      const z=s.comp().plane, f=s.state().fraction;
      s.moveAlong(4); s.recomputeNow();
      const moved=s.comp().plane.origin[2];
      s.setAngles({pitch:200,yaw:-12,offU:99,offV:'x'}); s.recomputeNow();
      const st=s.state(), n=s.comp().plane.normal;
      s.setDragMode('rotate'); const dm=[s.dragMode(),s.hudText()]; s.setDragMode('bogus');
      const lin=s.comp().range;
      ns.display={prefs:()=>({opts:{speedLog:true}})};
      s.set({range:'section',quantity:'speed'}); s.recomputeNow();
      const lg=s.comp().range, ends=VC.scaleEnds(lg);
      s.setBasis('centerline'); s.recomputeNow();
      out({planes,M:{min:M.bounds.min,max:M.bounds.max},o0,z,f,moved,st,n,dm,keep:s.dragMode(),lin:lin.log,lg:[lg.log,lg.min,lg.max],ends,back:s.state().basis});
      s.dispose();
    """, extra=["ws_slice.js"])
    mn, mx = out["M"]["min"], out["M"]["max"]
    for ax, p in enumerate(out["planes"]):
        normal = [0, 0, 0]; normal[ax] = 1
        assert p["normal"] == pytest.approx(normal)
        assert p["origin"][ax] == pytest.approx(mn[ax] + 0.25 * (mx[ax] - mn[ax]))
        assert all(p["origin"][k] == pytest.approx((mn[k] + mx[k]) / 2) for k in range(3) if k != ax)
    assert out["z"]["normal"] == pytest.approx([0, 0, 1]) and out["z"]["origin"][2] == pytest.approx(out["o0"][2], abs=1e-6)   # starts where the plane was
    assert out["moved"] == pytest.approx(out["z"]["origin"][2] + 4)            # classic positionStep on an axis: mm of the extent
    assert out["st"]["pitch"] == 85 and out["st"]["yaw"] == -12 and out["st"]["offU"] == 30 and out["st"]["offV"] == 0   # classic slider ranges
    assert out["dm"][0] == "rotate" and "旋转" in out["dm"][1] and out["keep"] == "rotate"
    assert out["lin"] is False and out["lg"][0] is True
    assert out["ends"][0] == pytest.approx(max(out["lg"][1], out["lg"][2] / 200, 1e-3))   # classic log floor on the section map
    assert out["back"] == "centerline"


# ------------------------------------------------------------------------------------------------ labels, presets, defaults, link hooks
def test_finding_label_count_and_classic_presets_defaults_and_link_hooks():
    out = _node(WALL + """
      require(%s); require(%s); require(%s);
      const D=ns.display, A=ns.annot;
      let prefs={labels:{branches:false,findings:0,maxd:false,annotations:true}}, drawn=0, applied=[], layersCalled=0;
      const store={prefs:()=>prefs,setPrefs:p=>{prefs=Object.assign({},prefs,p);ns.store.setPrefs(p);}};
      const {r}=await wallResult();
      await r.preload(['mv','mf','ms','pv','ps','f.wss.d','f.wss.r','f.tawss.d','f.tawss.r','f.osi.d','f.osi.r']);
      const box=new El('div'), v=ns.viewer.create(box,{kind:'main'}); await v.setResult(r);
      const S={viewerA:v,viewerB:null,cur:{result:r,manifest:r.manifest,field:'wss',window:'adaptive',runIdentity:'RW'},layers:{trust:false},els:{}};
      ns.shell={state:()=>S};
      const api={cur:()=>S.cur,viewer:()=>v,state:()=>S,store:()=>store,ui:()=>ns.ui,h:ns.ui.h,offline:()=>false,drawLabels:()=>{drawn++;},setLayers:()=>{layersCalled++;},
        applyField:(f,w)=>{applied.push([f,JSON.stringify(w)]);S.cur.field=f;S.cur.window=w;},fieldById:(m,id)=>(m.fields||[]).find(x=>x.id===id),renderToolbar(){},renderInspector(){},currentTab:()=>'overview'};
      const item0=A.findingsItem(api); A.setLabelCount(api,10); const item1=A.findingsItem(api); A.setLabelCount(api,7);
      const counts={item0:[item0.label,item0.checked,item0.hint],item1:[item1.checked,item1.hint],stored:ns.store.prefs().labels.findings,drawn};
      // classic default settings → this browser's display settings
      const wallPlan=D.defaultsPlan('wall',{colormap:'viridis',bands:8,units:'dyn',thresholds_pa:[0.5,4,7],opacity:0.6,log:true,lang:'en'});
      const volPlan=D.defaultsPlan('volume',{colormap:'turbo',bands:6,pressure_units:'mmHg',speed_units:'cm/s',opacity:0.2});
      D.applyDefaults('wall',{colormap:'viridis',bands:8,units:'dyn',thresholds_pa:[0.5,4,7]},api);
      const afterDefaults=JSON.parse(JSON.stringify({cmap:ns.store.prefs().cmap,units:D.prefs().units,wss:D.prefs().defaults.wss,tawss:D.prefs().defaults.tawss,imported:Object.keys(D.prefs().opts.imported)}));
      // a classic preset saved on this same result, and one from another result
      const state={schema_version:'wss-deploy.view/v1',family:'wall',run_identity:'RW',field:'tawss',colormap:'turbo',bands:6,units:'Pa',range:{mode:'fixed',max:2},
        overlay:{trust:true,contours:true},highlight:{top:true,top_pct:2.5},slice:{clip:0.6},labels:{findings:3,branches:true,max_diameter:true},branches_hidden:[1],
        measurements:[{id:'M1'}],camera:{position:[1,2,300],target:[0,0,20],up:[0,1,0]}};
      D.applyPreset({name:'p',state},api);
      await new Promise(res=>setTimeout(res,500));                      // the camera moves with a short animation
      const P=D.prefs();
      const afterPreset={field:S.cur.field,win:S.cur.window,cmap:ns.store.prefs().cmap,unit:P.units.wss,contours:P.opts.contours,top:[P.opts.top,P.opts.topPct],clip:D.session('RW').clip,
        bands:D.session('RW').bands.tawss,labels:ns.store.prefs().labels,trust:S.layers.trust,branches:v.getState().branches,cam:v.getCamera().position,layersCalled};
      const vol=D.applyPreset({name:'v',state:{family:'volume'}},api);
      // reproduction-link hooks (lane 5: x.display): capture, then restore on a fresh state; junk is ignored
      const link=D.linkState(api);
      D.setOpts({contours:false,top:false,topPct:1,speedLog:false}); delete D.session('RW').clip;
      const restored=D.applyLinkState(link,api), again={contours:D.prefs().opts.contours,top:D.prefs().opts.top,pct:D.prefs().opts.topPct,clip:D.session('RW').clip};
      const junk=[D.applyLinkState(null,api),D.applyLinkState([1],api),D.applyLinkState({contours:'yes',topPct:'x',clip:'z'},api),D.applyLinkState({v:9},api)];
      const old=D.applyLinkState({contours:true},api), partial={contours:D.prefs().opts.contours,clip:D.session('RW').clip};
      const ext=(ns.ext||[]).find(e=>e.id==='display');
      out({counts,wallPlan:{text:wallPlan.text,skipped:wallPlan.skipped},volPlan:volPlan.text,afterDefaults,afterPreset,vol,link,restored,again,junk,old,partial,
        hooks:[typeof ext.linkState,typeof ext.applyLinkState],storeBwr:ns.store.sanitizePrefs({cmap:'bwr'}).cmap});
    """ % (json.dumps(str(V2 / "ws_ui.js")), json.dumps(str(V2 / "ws_store.js")), json.dumps(str(V2 / "ws_annot.js"))), extra=_with_contour(["ws_display.js"]))
    c = out["counts"]
    assert c["item0"] == ["发现标签…", False, "关"] and c["item1"] == [True, "前 10 条"] and c["stored"] == 10 and c["drawn"] == 1   # 7 is not a classic choice
    assert out["wallPlan"]["text"] == ["色表 Viridis", "8 段", "WSS 单位 dyn/cm²", "WSS 阈值 0.5 / 4 / 7 Pa"]
    assert len(out["wallPlan"]["skipped"]) == 3                                # wall opacity, log, English
    assert out["volPlan"] == ["色表 Turbo", "6 段", "压力 mmHg", "速度 cm/s", "外壁 0.20"]
    a = out["afterDefaults"]
    assert a["cmap"] == "viridis" and a["units"]["wss"] == "dyn/cm²" and a["wss"] == {"bands": 8, "thresholds": [0.5, 4, 7]} and a["tawss"] == {"bands": 8}
    assert a["imported"] == ["wall"]
    p = out["afterPreset"]
    assert p["field"] == "tawss" and p["win"] == {"range": [0, 2]} and p["cmap"] == "turbo" and p["unit"] == "Pa"
    assert p["contours"] is True and p["top"] == [True, 2.5] and p["clip"] == 0.6 and p["bands"] == 6
    assert p["labels"]["findings"] == 3 and p["labels"]["branches"] is True and p["labels"]["maxd"] is True and p["trust"] is True and p["layersCalled"] >= 1
    assert p["branches"] == [0] and p["cam"] == pytest.approx([1, 2, 300])     # the same result: its camera comes back
    assert out["vol"] is False                                                 # a volume preset on a wall result is refused
    assert out["link"] == {"v": 1, "contours": True, "top": True, "topPct": 2.5, "speedLog": False, "clip": 0.6}
    assert out["restored"] is True and out["again"] == {"contours": True, "top": True, "pct": 2.5, "clip": 0.6}
    assert out["junk"] == [False, False, False, False]
    assert out["old"] is True and out["partial"] == {"contours": True, "clip": 0.6}   # an old / partial state only changes what it names
    assert out["hooks"] == ["function", "function"] and out["storeBwr"] == "bwr"


# ------------------------------------------------------------------------------------------------ shell wiring (workspace stub harness)
def _files4():
    files = _files()
    files.insert(files.index("ws_display.js"), "ws_annot.js")
    files.insert(files.index("ws_display.js") + 1, "ws_probe.js")
    return files


def test_layers_menu_items_colour_maps_label_count_p_key_and_classic_tools_section():
    out = _run(PATCH + """
      canned['/api/preferences'] = {body: {preferences: {report_defaults: {wall: {colormap: 'viridis', bands: 8}}, presets: {wall: [{name: '瘤囊', created_at: '2026-09-21T00:00:00Z', state: {family: 'wall', field: 'wss'}}]}}}};
      await boot();
      await hashTo('#/job/A', 300);
      const layerBtn = walk(app(), e => e.getAttribute && e.getAttribute('aria-label') === '图层、色表与背景')[0];
      layerBtn.click(); await wait(20);
      const menu = byId('ws-menu'), labels = byClass(menu, 'menu-label').map(textOf), hints = byClass(menu, 'menu-hint').map(textOf);
      // 发现标签… opens the four classic choices
      byClass(menu, 'menu-item').find(e => textOf(e).indexOf('发现标签') >= 0).click(); await wait(40);
      const second = byClass(byId('ws-menu'), 'menu-label').map(textOf);
      byClass(byId('ws-menu'), 'menu-item').find(e => textOf(e) === '前 3 条').click(); await wait(20);
      const findings = JSON.parse(localStorage.getItem('wssv2:prefs:2')).labels.findings;
      const hoverBefore = ns.display.prefs().opts.hover;
      await key('p');
      const hoverAfter = ns.display.prefs().opts.hover;
      // the 工具 tab: 经典设置 from /api/preferences
      const tools = walk(app(), e => e.getAttribute && e.getAttribute('role') === 'tab' && textOf(e) === '工具')[0];
      if (tools) { tools.click(); await wait(80); }
      const sec = byClass(app(), 'wsd-classic')[0];
      done({labels, hints, second, findings, hoverBefore, hoverAfter, classic: sec ? textOf(sec) : null, calls: calls.filter(c => /preferences/.test(c))});
    """, files=_files4())
    assert out["errors"] == []
    labels = out["labels"]
    i = labels.index("滞留区斜纹")
    assert labels[i + 1:i + 5] == ["等值线", "高亮最高 1%…", "剖切…", "悬停读数"]   # after lane E's items, in the 图层 group
    assert labels.index("悬停读数") < labels.index("标注") and "发现标签…" in labels and "发现标签（前 5 条）" not in labels
    assert labels[-4:] == ["彩虹（默认）", "viridis", "turbo", "蓝白红"]
    assert "0.4 / 4 / 7 Pa" in out["hints"] and "P" in out["hints"]
    assert out["second"] == ["关", "前 3 条", "前 5 条", "前 10 条"] and out["findings"] == 3
    assert out["hoverBefore"] is True and out["hoverAfter"] is False
    assert out["classic"] and "瘤囊" in out["classic"] and "色表 viridis" in out["classic"].lower() and "8 段" in out["classic"]
    assert out["calls"] == ["GET /api/preferences"]


def test_offline_start_shows_no_classic_settings_and_no_errors():
    out = _run(PATCH + """
      await boot();
      await hashTo('#/job/A', 300);
      const tools = walk(app(), e => e.getAttribute && e.getAttribute('role') === 'tab' && textOf(e) === '工具')[0];
      if (tools) { tools.click(); await wait(60); }
      done({classic: byClass(app(), 'wsd-classic').length, pref: calls.filter(c => /preferences/.test(c)).length});
    """, offline=True, files=[f for f in _files4() if f not in ("ws_api.js", "ws_rail.js", "ws_upload.js", "ws_input.js", "ws_compare.js")])
    assert out["errors"] == [] and out["classic"] == 0 and out["pref"] == 0
