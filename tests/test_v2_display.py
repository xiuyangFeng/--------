"""Node tests for lane E of the workspace v2 second phase: display options (PHASE2_LANES.md §5 E).

Covers the pure helpers of ws_display.js (thresholds, fractions, peak markers, units, preferences), the viewer kernel's
lane E block (display provider: per-field bands, thresholds and units on the colour bar, markers, view state) with a
stub WebGL renderer, the adapters' display flags (input STL, stagnation hatch, glass opacity, streamline density and
tubes, velocity arrows with the classic addVectors formula), the colour bar in display units, the section tool's six
classic stations and units, the probe card units, and the shell wiring (layers menu, toolbar, key 0, compare push)
on the workspace stub harness.  Display only: every check that touches numbers compares against the stored values.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from wss_deploy.paths import STATIC_DIR
from tests.test_v2_workspace_js import WS_FILES, _run

V2 = STATIC_DIR / "v2"
LANE_E_JS = ["adapter_wall.js", "adapter_volume.js", "core_colormap.js", "core_colorbar.js", "core_viewer.js", "ws_compare.js", "ws_slice.js",
             "ws_display.js", "ws_probe.js", "ws_shell.js"]
CORE_FILES = ["core_util.js", "core_data.js", "core_colormap.js", "core_colorbar.js", "core_viewer.js", "core_orientation.js", "core_cursor.js",
              "adapter_wall.js", "adapter_volume.js"]

# Stub DOM + stub WebGL renderer / orbit controls: the viewer's own logic runs in Node, nothing is drawn.
STUBS = r"""
class El {
  constructor(tag) { this.tagName = String(tag || 'div').toUpperCase(); this.children = []; this.parentNode = null; this.style = {}; this.attrs = {}; this.events = {};
    this.className = ''; this.innerHTML = ''; this._text = ''; this.clientWidth = 800; this.clientHeight = 600; this.offsetWidth = 60; this.dataset = {}; this.value = ''; this.checked = false; this.disabled = false; this.hidden = false;
    const self = this, set = new Set();
    this.classList = {add: c => { set.add(c); self.className = [...set].join(' '); }, remove: c => { set.delete(c); self.className = [...set].join(' '); },
      toggle: (c, f) => { if (f === undefined) f = !set.has(c); f ? set.add(c) : set.delete(c); self.className = [...set].join(' '); return f; }, contains: c => set.has(c)}; }
  get ownerDocument() { return globalThis.document; }
  set textContent(v) { this._text = String(v); this.children = []; } get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  appendChild(x) { if (x.parentNode) x.parentNode.removeChild(x); x.parentNode = this; this.children.push(x); return x; }
  removeChild(x) { const i = this.children.indexOf(x); if (i >= 0) this.children.splice(i, 1); x.parentNode = null; return x; }
  replaceChildren(...xs) { this.children.forEach(c => { c.parentNode = null; }); this.children = []; this._text = ''; xs.forEach(x => this.appendChild(x)); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; } removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(n, f) { (this.events[n] = this.events[n] || []).push(f); } removeEventListener(n, f) { const l = this.events[n] || []; const i = l.indexOf(f); if (i >= 0) l.splice(i, 1); }
  getBoundingClientRect() { return {left: 0, top: 0, right: 800, bottom: 600, width: 800, height: 600}; }
  contains(x) { while (x) { if (x === this) return true; x = x.parentNode; } return false; }
  focus() {} blur() {}
}
class TextNode { constructor(t) { this.nodeValue = String(t); this.parentNode = null; } get textContent() { return this.nodeValue; } }
globalThis.document = {body: new El('body'), createElement: t => new El(t), createElementNS: (n, t) => new El(t), createTextNode: t => new TextNode(t),
  addEventListener() {}, removeEventListener() {}, getElementById: () => null};
globalThis.requestAnimationFrame = f => setTimeout(f, 0);
globalThis.cancelAnimationFrame = id => clearTimeout(id);
function stubRenderer(THREE) {
  THREE.WebGLRenderer = function () { this.domElement = new El('canvas'); this.info = {memory: {geometries: 0, textures: 0}, render: {calls: 0, triangles: 0}, programs: []}; };
  Object.assign(THREE.WebGLRenderer.prototype, {setPixelRatio() {}, setClearColor() {}, setSize() {}, render() {}, getPixelRatio() { return 1; }, dispose() {}, forceContextLoss() {}});
  THREE.OrbitControls = function (cam) { this.object = cam; this.target = new THREE.Vector3(); this.enabled = true; };
  Object.assign(THREE.OrbitControls.prototype, {addEventListener() {}, removeEventListener() {}, update() {}, dispose() {}});
}
"""


def _prelude(extra=()) -> str:
    lines = ["globalThis.THREE=require(" + json.dumps(str(STATIC_DIR / "three.min.js")) + ");", STUBS, "stubRenderer(globalThis.THREE);",
             "require(" + json.dumps(str(STATIC_DIR / "report_common.js")) + ");", "require(" + json.dumps(str(STATIC_DIR / "volume_viewer.js")) + ");"]
    for name in CORE_FILES + list(extra):
        lines.append("require(" + json.dumps(str(V2 / name)) + ");")
    lines.append("const ns=globalThis.WSSV2, C=globalThis.WssReportCommon, VC=globalThis.VolumeViewerCore, T=globalThis.THREE;")
    lines.append("const out=x=>console.log(JSON.stringify(x));")
    return "\n".join(lines) + "\n"


def _node(script: str, extra=()) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the lane E tests")
    program = _prelude(extra) + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{") or line.startswith("[")]
    return json.loads(lines[-1])


# A wall tube (radius 5, z 0 … 40, two branches split at z = 20) with WSS / TAWSS / OSI, and the peak in analysis.peak.
WALL = r"""
function wallResult(extra) {
  const nr=24,nz=21,V=[],F=[],MS=[];
  for(let j=0;j<nz;j++)for(let i=0;i<nr;i++){const a=2*Math.PI*i/nr;V.push(5*Math.cos(a),5*Math.sin(a),2*j);MS.push(2*j<20?0:1);}
  for(let j=0;j<nz-1;j++)for(let i=0;i<nr;i++){const a=j*nr+i,b=j*nr+(i+1)%nr,c=a+nr,d=b+nr;F.push(a,b,d,a,d,c);}
  const PV=[],PS=[],W=[],TA=[],OS=[];let k=0;
  for(let j=0;j<2*nz-1;j++)for(let i=0;i<2*nr;i++){const a=Math.PI*i/nr,z=j;PV.push(4.9*Math.cos(a),4.9*Math.sin(a),z);PS.push(z<20?0:1);W.push(k===7?NaN:1+0.1*z);TA.push(z<6?0.2:1+0.01*i);OS.push(i<10?0.3:0.02);k++;}
  TA[100]=0.05; TA[200]=0.05; TA[300]=9; TA[400]=9;           // ties: the first occurrence wins (np.argmin / np.argmax order)
  const nV=V.length/3,disp=f=>{const o=new Float32Array(nV);for(let v=0;v<nV;v++){const z=V[3*v+2],i=v%nr;o[v]=f(z,i);}return o;};
  const tables={mv:new Float32Array(V),mf:new Uint32Array(F),ms:new Uint8Array(MS),pv:new Float32Array(PV),ps:new Uint8Array(PS),
    'f.wss.d':disp(z=>1+0.1*z),'f.wss.r':new Float32Array(W),'f.tawss.d':disp(z=>z<6?0.2:1),'f.tawss.r':new Float32Array(TA),
    'f.osi.d':disp((z,i)=>i<5?0.3:0.02),'f.osi.r':new Float32Array(OS)};
  const arrays={};for(const [kk,a] of Object.entries(tables)){const w=kk==='mv'||kk==='pv'||kk==='mf'?3:1;arrays[kk]={dtype:{Float32Array:'float32',Uint32Array:'uint32',Uint8Array:'uint8'}[a.constructor.name],shape:w>1?[a.length/w,w]:[a.length],bytes:a.byteLength};}
  const F_=(id,u,th,dir)=>({id,label:id,short_label:id.toUpperCase(),units:u,kind:'scalar',components:1,arrays:{display:'f.'+id+'.d',read:'f.'+id+'.r'},display:{thresholds:th,threshold_direction:dir,p99:2}});
  const manifest={schema:'wss-deploy.v2-manifest/v1',result:{family:'wall',run_identity:'RW'},
    geometry:{display_mesh:{vertices:'mv',faces:'mf',segment:'ms'},points:{xyz:'pv',segment:'ps'},centerline:{},branches:[{id:0,name:'A',parent:null},{id:1,name:'B',parent:0}]},
    fields:[F_('wss','Pa',[0.4,4,7],'below'),F_('tawss','Pa',[0.4,4,7],'below'),F_('osi','1',[0.1,0.2,0.3],'above')],arrays,
    analysis:Object.assign({peak:{xyz_mm:[1,2,3],max_pa:4.5,segment_id:1}},extra||{})};
  return ns.data.loadResult({manifest:()=>Promise.resolve(manifest),fetchArray:kk=>Promise.resolve(new Uint8Array(tables[kk].buffer).slice().buffer)}).then(r=>({r,tables}));
}
"""

# A volume tube with interior points, velocity, speed, pressure and three streamlines.
VOLUME = r"""
function volumeResult() {
  const V=new Float32Array([0,0,0, 10,0,0, 0,10,0, 0,0,40, 10,0,40, 0,10,40]),F=new Uint32Array([0,1,4, 0,4,3, 1,2,5, 1,5,4, 2,0,3, 2,3,5]);
  const P=[],W=[],SEG=[],VEL=[],SP=[],PR=[];
  for(let z=0.5;z<40;z+=0.5)for(let x=1;x<4;x+=1)for(let y=1;y<4;y+=1){P.push(x,y,z);W.push(0);SEG.push(z<20?0:1);const vz=0.2+0.02*z+0.01*x;VEL.push(0.01*y,0,vz);SP.push(Math.hypot(0.01*y,0,vz));PR.push(100-z);}
  const L=[],LS=[],LO=[0];for(let l=0;l<7;l++){for(let z=0;z<=40;z+=2){L.push(1+l*0.3,2,z);LS.push(0.3+0.01*z);}LO.push(L.length/3);}
  const tables={mv:V,mf:F,vpts:new Float32Array(P),vis_wall:new Uint8Array(W),vseg:new Int32Array(SEG),'f.pressure.r':new Float32Array(PR),'f.speed.r':new Float32Array(SP),
    'f.velocity.r':new Float32Array(VEL),'sl.xyz':new Float32Array(L),'sl.speed':new Float32Array(LS),'sl.offsets':new Uint32Array(LO)};
  const arrays={};for(const [k,a] of Object.entries(tables)){const w=k==='mv'||k==='vpts'||k==='mf'||k==='sl.xyz'?3:1;arrays[k]={dtype:{Float32Array:'float32',Uint32Array:'uint32',Uint8Array:'uint8',Int32Array:'int32'}[a.constructor.name],shape:w>1?[a.length/w,w]:[a.length],bytes:a.byteLength};}
  const manifest={schema:'wss-deploy.v2-manifest/v1',result:{family:'volume',run_identity:'RV'},geometry:{display_mesh:{vertices:'mv',faces:'mf'},points:{xyz:'vpts',segment:'vseg',is_wall:'vis_wall'},
      centerline:{},streamlines:{xyz:'sl.xyz',speed:'sl.speed',offsets:'sl.offsets'},branches:[]},
    fields:[{id:'pressure',units:'Pa',kind:'scalar',components:1,arrays:{display:null,read:'f.pressure.r'}},{id:'speed',units:'m/s',kind:'scalar',components:1,arrays:{display:null,read:'f.speed.r'}},
            {id:'velocity',units:'m/s',kind:'vector',components:3,arrays:{display:null,read:'f.velocity.r'}}],arrays};
  return ns.data.loadResult({manifest:()=>Promise.resolve(manifest),fetchArray:k=>Promise.resolve(new Uint8Array(tables[k].buffer).slice().buffer)}).then(r=>({r,tables,P,VEL}));
}
"""


# ------------------------------------------------------------------------------------------------ files
def test_lane_e_scripts_parse_and_are_bundled():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    for name in LANE_E_JS:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    assert "ws_display.js" in bundle["scripts"] and "ws_display.js" not in bundle["offline_exclude"]
    assert bundle["scripts"].index("core_viewer.js") < bundle["scripts"].index("ws_display.js") < bundle["scripts"].index("ws_shell.js")
    assert "v2_display.css" in bundle["styles"]


# ------------------------------------------------------------------------------------------------ pure helpers
def test_pure_helpers_thresholds_fractions_units_and_prefs():
    out = _node("""
      const D=ns.display;
      const v=new Float32Array([0.1,0.3,0.5,5,8,NaN,0.39,4,7]);
      out({valid:[D.validThresholds([0.4,4,7]),D.validThresholds([0.4,0.4,7]),D.validThresholds([-1,4,7]),D.validThresholds(['','4','7']),D.validThresholds(['0.5','3','6'])],
        below:D.thresholdFractions(v,[0.4,4,7],false),above:D.thresholdFractions(v,[0.4,4,7],true),
        mmHg:D.unitFactor('pressure','mmHg'),classic:VC.convertUnit(1,'pressure','mmHg'),cms:D.unitFactor('velocity','cm/s'),pa:D.unitFactor('pressure','Pa'),bad:D.unitFactor('pressure','psi'),
        kinds:[D.unitKind({units:'Pa'},'volume'),D.unitKind({units:'m/s'},'volume'),D.unitKind({units:'Pa'},'wall'),D.unitKind({units:'1'},'volume')],
        prefs:D.sanitizePrefs({defaults:{tawss:{bands:8,thresholds:[0.5,3,6]},osi:{bands:7},'bad key':{bands:4},wss:{thresholds:[3,2,1]}},units:{pressure:'mmHg',velocity:'km/h'},
          layers:{stl:true,peaks:'yes'},volume:{opacity:0.9,density:2,width:9,thin:false}}),
        session:D.sanitizeSessionState({bands:{tawss:6,osi:5},thresholds:{tawss:[0.3,4,7],wss:[1]}})});
    """, extra=["ws_display.js"])
    assert out["valid"] == [[0.4, 4, 7], None, None, None, [0.5, 3, 6]]
    # classic thrFracs: below-fields x < t0, x > t1, x > t2; above-fields x > t0; NaN not counted (8 finite values)
    assert out["below"]["n"] == 8 and out["below"]["fractions"] == pytest.approx([3 / 8, 3 / 8, 1 / 8])
    assert out["above"]["fractions"] == pytest.approx([5 / 8, 3 / 8, 1 / 8])
    assert out["mmHg"] == out["classic"] == pytest.approx(1 / 133.322) and out["cms"] == 100 and out["pa"] == 1 and out["bad"] == 1
    assert out["kinds"] == ["pressure", "velocity", None, None]                 # wall shear keeps Pa
    p = out["prefs"]
    assert p["defaults"] == {"tawss": {"bands": 8, "thresholds": [0.5, 3, 6]}}   # invalid bands / keys / thresholds dropped
    assert p["units"] == {"pressure": "mmHg", "velocity": "m/s", "wss": "Pa"}   # phase 3 lane 4 (W8): WSS display unit, default Pa
    assert p["layers"] == {"stl": True, "peaks": True, "stagnation": False, "vectors": False}
    assert p["volume"] == {"opacity": 0.1, "density": 2, "width": 1, "thin": False}
    assert out["session"] == {"bands": {"tawss": 6}, "thresholds": {"tawss": [0.3, 4, 7]}}


def test_peak_markers_follow_the_classic_rules():
    out = _node(WALL + """
      const {r}=await wallResult();
      await r.preload(['pv','ps','f.wss.r','f.tawss.r','f.osi.r']);
      const D=ns.display, wss=D.peakMarkers(r,'wss'), ta=D.peakMarkers(r,'tawss'), osi=D.peakMarkers(r,'osi');
      const {r:r2}=await wallResult({peak:{}}); await r2.preload(['pv','ps','f.wss.r']);
      const TA=r.array('f.tawss.r');let mx=-1,mn=-1;for(let i=0;i<TA.length;i++){if(Number.isFinite(TA[i])&&(mx<0||TA[i]>TA[mx]))mx=i;if(Number.isFinite(TA[i])&&(mn<0||TA[i]<TA[mn]))mn=i;}
      out({wss,ta,osi:osi.map(m=>m.kind),mx,mn,fallback:D.peakMarkers(r2,'wss'),vol:D.peakMarkers({family:'volume'},'speed')});
    """, extra=["ws_display.js"])
    assert out["wss"] == [{"kind": "max", "index": None, "value": 4.5, "xyz": [1, 2, 3], "segmentId": 1}]   # the summary's peak point
    assert [m["kind"] for m in out["ta"]] == ["max", "min"]                    # TAWSS also marks its minimum
    assert out["ta"][0]["index"] == out["mx"] == 300 and out["ta"][1]["index"] == out["mn"] == 100   # first occurrence of ties
    assert out["osi"] == ["max"]
    fb = out["fallback"][0]                                                      # no summary point: argmax over the read array (NaN skipped)
    assert fb["kind"] == "max" and fb["value"] == pytest.approx(1 + 0.1 * 40) and fb["segmentId"] == 1
    assert out["vol"] == []


# ------------------------------------------------------------------------------------------------ viewer kernel, lane E block
def test_viewer_display_provider_bands_thresholds_units_markers_and_state():
    out = _node(WALL + """
      const {r}=await wallResult();
      const box=new El('div'), v=ns.viewer.create(box,{kind:'main'});
      // without a provider: plain pass-through (the viewer keeps the spec it is given)
      await v.setResult(r);
      await v.setField('tawss',{window:'adaptive',log:null,bands:4});
      const plain={bands:v.getState().scale.bands,th:v.colorbarInfo().thresholds,display:v.getState().display===undefined,peaks:v.peakMarkers().length};
      // a provider: per-field bands, thresholds, units, markers, state
      const store={bands:{tawss:8},thresholds:{tawss:[0.5,3,6]}};let restored=null;
      ns.viewer.displayProvider={
        bands:(vv,f)=>store.bands[f]!==undefined?store.bands[f]:0, thresholds:(vv,f)=>store.thresholds[f]||null,
        units:(vv,f)=>f&&f.id==='wss'?{units:'dyn/cm²',factor:10}:null, flags:()=>({stagnation:{tawss_lt_pa:0.4,osi_gt:0.1}}),
        markers:vv=>[{xyz:[0,0,5],kind:'max',text:'最大 1 Pa',segmentId:0},{xyz:[0,0,30],kind:'min',text:'最小',segmentId:1}],
        state:()=>({v:1,bands:Object.assign({},store.bands)}), restore:(vv,d)=>{restored=d;}};
      await v.setField('tawss',{window:'adaptive',log:null,bands:null});
      const t1=v.colorbarInfo();
      await v.setField('wss',{window:'adaptive',log:null,bands:null});
      const w1=v.colorbarInfo();
      v.setBranchVisibility([0]);
      const hidden=v.peakMarkers().length;
      v.setBranchVisibility(null);
      store.bands.wss=12; v.refreshDisplay();
      const st=v.getState(), refreshed=st.scale.bands;
      await v.applyState({field:'tawss',scale:{window:'adaptive',log:null,bands:3},display:{v:1,bands:{tawss:6}}});
      const after=v.getState().scale.bands;
      out({plain,t1:{bands:t1.scale.bands,th:t1.thresholds,units:t1.displayUnits||null},w1:{bands:w1.scale.bands,th:w1.thresholds,units:w1.displayUnits,k:w1.unitFactor},
        peaks:v.peakMarkers(),hidden,refreshed,stateDisplay:st.display,restored,after});
    """)
    assert out["plain"] == {"bands": 4, "th": [0.4, 4, 7], "display": True, "peaks": 0}
    assert out["t1"] == {"bands": 8, "th": [0.5, 3, 6], "units": None}       # per-field bands and thresholds from the provider
    assert out["w1"] == {"bands": 0, "th": [0.4, 4, 7], "units": "dyn/cm²", "k": 10}   # another field: its own bands, the release thresholds
    assert out["peaks"] == [[0, 0, 5], [0, 0, 30]] and out["hidden"] == 1  # a marker on a hidden branch is not drawn
    assert out["refreshed"] == 12
    assert out["stateDisplay"] == {"v": 1, "bands": {"tawss": 8, "wss": 12}}
    assert out["restored"] == {"v": 1, "bands": {"tawss": 6}}                   # restore runs before the state is applied
    assert out["after"] == 8                                                   # the provider decides the bands, not the stale spec


# ------------------------------------------------------------------------------------------------ adapters
def test_wall_adapter_input_stl_and_stagnation_hatch():
    out = _node(WALL + """
      const {r}=await wallResult();
      const A=ns.adapters.wall;
      await r.preload(A.requiredArrays(r,'tawss').concat(['f.osi.d','f.wss.d']));
      const root=new T.Group(), h=A.build(r,{THREE:T,root,gfx:ns.viewer.gfx,requestRender(){},kind:'main'});
      h.setField('tawss',ns.colormap.scale({range:[0,2]}));h.setLayers(ns.viewer.normalizeLayers('wall'));h.setLighting('flat');
      const g=root.children[0], mesh=g.children.find(o=>o.name==='wall-surface'), hatch=g.children.find(o=>o.name==='wall-stagnation');
      const colours=Array.from(mesh.geometry.attributes.color.array);
      const flat0=mesh.material.type, s1=h.setDisplay({stl:true});
      const stlMat=mesh.material.type, colSame=Array.from(mesh.geometry.attributes.color.array).every((x,i)=>x===colours[i]);
      const s2=h.setDisplay({stl:false,stagnation:{tawss_lt_pa:0.4,osi_gt:0.1}});
      const T_=r.array('f.tawss.d'),O_=r.array('f.osi.d');let n=0;for(let i=0;i<T_.length;i++)if(T_[i]<0.4&&O_[i]>0.1)n++;
      const mask=Array.from(mesh.geometry.attributes.aStag.array).reduce((a,b)=>a+b,0);
      const back=h.setDisplay({}).stagnation;
      out({flat0,s1,stlMat,colSame,s2,n,mask,hatchVisible:hatch.visible,back,material:mesh.material.type});
    """)
    assert out["flat0"] == "MeshBasicMaterial" and out["stlMat"] == "MeshPhongMaterial" and out["s1"]["stl"] is True
    assert out["colSame"]                                                      # the field colours stay computed underneath
    assert out["s2"] == {"stl": False, "stagnation": "on"} and out["n"] > 0 and out["mask"] == out["n"]   # classic rule on the display vertices
    assert out["back"] == "off" and out["material"] == "MeshBasicMaterial"


def test_volume_adapter_opacity_streamlines_and_classic_arrows():
    out = _node(VOLUME + """
      const {r,P,VEL}=await volumeResult();
      const A=ns.adapters.volume;
      await r.preload(A.requiredArrays(r,'speed').concat(A.optionalArrays(r),['f.velocity.r','f.pressure.r']));
      const root=new T.Group(), h=A.build(r,{THREE:T,root,gfx:ns.viewer.gfx,requestRender(){},fov:30});
      const sc=ns.colormap.scale({range:[0,1.2]});
      h.setField('speed',sc);h.setLayers(Object.assign(ns.viewer.normalizeLayers('volume'),{streamlines:true}));h.arraysLoaded();
      const g=root.children[0], wall=g.children.find(o=>o.name==='volume-wall'), lines=g.children.find(o=>o.name==='volume-streamlines');
      const gain0=wall.material.uniforms.uGain.value, idx0=lines.geometry.index.count;
      h.setDisplay({opacity:0.3,density:2,thin:true});
      const gain1=wall.material.uniforms.uGain.value, idx2=lines.geometry.index.count;
      const t=h.setDisplay({opacity:0.1,density:3,thin:false,width:2});
      const tubes=g.children.find(o=>o.name==='volume-streamtubes'), radius=tubes.children[0].geometry.parameters.radius;
      const a=h.setDisplay({vectors:true,thin:true});
      const arrows=g.children.find(o=>o.name==='volume-arrows'), pos=Array.from(arrows.geometry.attributes.position.array);
      // classic addVectors on the same points: stride ⌈n/600⌉, size = diag × 0.025 × √(|v| / top), heads from planeBasis
      const V=r.array('mv');let mn=[1e9,1e9,1e9],mx=[-1e9,-1e9,-1e9];for(let i=0;i<V.length;i++){const k=i%3;mn[k]=Math.min(mn[k],V[i]);mx[k]=Math.max(mx[k],V[i]);}
      const diag=Math.max(Math.hypot(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2]),1),n=P.length/3,stride=Math.max(1,Math.ceil(n/600)),ref=[];
      for(let j=0;j<n;j+=stride){const v=[VEL[3*j],VEL[3*j+1],VEL[3*j+2]],len=Math.hypot(...v);const d=v.map(x=>x/len),size=diag*0.025*Math.sqrt(len/1.2);
        const s=[P[3*j],P[3*j+1],P[3*j+2]],e=s.map((x,k)=>x+d[k]*size),u=VC.planeBasis(d).u,b=e.map((x,k)=>x-d[k]*size*0.24);
        ref.push(...s,...e,...e,...b.map((x,k)=>x+u[k]*size*0.1),...e,...b.map((x,k)=>x-u[k]*size*0.1));}
      const maxDiff=Math.max(...pos.map((x,i)=>Math.abs(x-ref[i])));
      h.setField('pressure',ns.colormap.scale({range:[60,100]}));
      const offOnPressure=!g.children.find(o=>o.name==='volume-arrows');
      const stl=h.setDisplay({stl:true});
      out({gain0,gain1,idx0,idx2,t,radius,diag,a,count:arrows.userData.count,n,stride,refLen:ref.length,posLen:pos.length,maxDiff,offOnPressure,stl,
        wallMat:wall.material.type,cloud:g.children.find(o=>o.name==='volume-points').visible});
    """)
    assert out["gain0"] == 1 and out["gain1"] == pytest.approx(3)             # classic default 0.10 ↔ the unchanged look
    assert out["idx2"] < out["idx0"]                                           # every second streamline
    assert out["t"]["tubes"] == 3                                              # lines 0, 3, 6 of seven (density 3)
    assert out["radius"] == pytest.approx(out["diag"] / 900 * 2)               # classic tube radius
    assert out["count"] == out["n"] // out["stride"] + (1 if out["n"] % out["stride"] else 0) and out["posLen"] == out["refLen"]
    assert out["maxDiff"] < 1e-4                                               # same arrows as the classic formula
    assert out["offOnPressure"]                                                # arrows only with the speed field
    assert out["stl"]["stl"] and out["wallMat"] == "MeshPhongMaterial" and out["cloud"] is False


# ------------------------------------------------------------------------------------------------ colour bar
def test_colour_bar_display_units_thresholds_and_input_stl():
    out = _node("""
      const sc=ns.colormap.scale({range:[-1300,100]}), k=1/133.322;
      const base={label:'压力',units:'Pa',scale:sc,histogram:null,thresholds:[],windowLabel:{label:'本例自适应'}};
      const raw=ns.colorbar.toSVG(base), mm=ns.colorbar.toSVG(Object.assign({},base,{displayUnits:'mmHg',unitFactor:k}));
      const labels=s=>[...s.matchAll(/>([^<>]+)</g)].map(m=>m[1].trim()).filter(Boolean);
      const ticks=ns.colorbar.barTicks(sc,6,k), fOk=ticks.every(t=>Math.abs(sc.norm(t.v/k)-t.f)<1e-9||t.f===0||t.f===1);
      const w=ns.colormap.scale({range:[0,10]}), th=ns.colorbar.toSVG({label:'WSS',units:'Pa',scale:w,thresholds:[0.4,4,7],displayUnits:'dyn/cm²',unitFactor:10});
      const el=new El('div'), cb=ns.colorbar.create(el); cb.update({label:'TAWSS',units:'Pa',scale:w,stl:true});
      out({raw:labels(raw),mm:labels(mm),fOk,th:labels(th),stl:cb.element.innerHTML});
    """)
    assert "Pa" in out["raw"] and "mmHg" in out["mm"] and "Pa" not in out["mm"]
    assert "-8" in out["mm"] or "−8" in out["mm"]                              # nice numbers in the display unit
    assert out["fOk"]
    assert {"4", "40", "70"} <= set(out["th"])                                 # threshold labels × factor
    assert "输入 STL" in out["stl"] and "svg" not in out["stl"]


# ------------------------------------------------------------------------------------------------ section tool
SLICE_TUBE = r"""
function tube(){
  const R=5,NA=48,NZ=41,V=[],F=[];
  for(let k=0;k<NZ;k++)for(let a=0;a<NA;a++){const t=2*Math.PI*a/NA;V.push(R*Math.cos(t),R*Math.sin(t),k);}
  for(let k=0;k+1<NZ;k++)for(let a=0;a<NA;a++){const i=k*NA+a,j=k*NA+(a+1)%NA,i2=i+NA,j2=j+NA;F.push(i,j,j2,i,j2,i2);}
  const P=[],W=[],S=[],VEL=[],PR=[];
  for(let z=0.35;z<40;z+=0.7)for(let x=-4.9;x<=4.9;x+=0.7)for(let y=-4.9;y<=4.9;y+=0.7){const r2=x*x+y*y;if(r2>=4.85*4.85)continue;P.push(x,y,z);W.push(0);S.push(0);VEL.push(0,0,1-r2/25);PR.push(100-2*z);}
  const CV=[],CS=[],CT=[],CE=[],CR=[];for(let k=0;k<NZ;k++){CV.push(0,0,k);CS.push(0);CT.push(0,0,1);CR.push(5);if(k+1<NZ)CE.push(k,k+1);}
  const speed=[];for(let i=0;i<VEL.length/3;i++)speed.push(Math.hypot(VEL[3*i],VEL[3*i+1],VEL[3*i+2]));
  const arrays={mv:new Float32Array(V),mf:new Uint32Array(F),vpts:new Float32Array(P),vis_wall:new Uint8Array(W),vseg:new Int32Array(S),
    cv:new Float32Array(CV),cs:new Int32Array(CS),ct:new Float32Array(CT),ce:new Uint32Array(CE),cr:new Float32Array(CR),
    'f.velocity.r':new Float32Array(VEL),'f.speed.r':new Float32Array(speed),'f.pressure.r':new Float32Array(PR)};
  const manifest={result:{family:'volume'},geometry:{display_mesh:{vertices:'mv',faces:'mf'},points:{xyz:'vpts',is_wall:'vis_wall',segment:'vseg'},
      centerline:{xyz:'cv',segment:'cs',tangent:'ct',edges:'ce',radius_mm:'cr'},branches:[{id:0,name:'主动脉'}]},
    fields:[{id:'pressure',arrays:{read:'f.pressure.r'}},{id:'speed',arrays:{read:'f.speed.r'}},{id:'velocity',arrays:{read:'f.velocity.r'}}]};
  return {family:'volume',manifest,has:k=>k in arrays,declared:k=>k in arrays,array:k=>arrays[k]};
}
"""


def test_section_stations_are_the_classic_automatic_planes_and_numbers_follow_units():
    out = _node(SLICE_TUBE + """
      const SL=ns.slice, r=tube(), M=SL.model(r);
      const st=SL.stations(M,0), auto=VC.automaticPlanes(M.groups);
      const s=SL.create(null,r,{hint:{xyz:[0,0,20]}});
      s.goStation(st[1]); s.recomputeNow();
      const p=s.comp().plane, pa=s.comp().integral.pressure_mean_pa;
      const raw=s.colorbarInfo();
      ns.display={displayUnit:k=>k==='pressure'?{units:'mmHg',factor:1/133.322}:{units:'cm/s',factor:100}};
      s.set({quantity:'pressure'}); s.recomputeNow();
      const mm=s.colorbarInfo();
      out({fractions:st.map(x=>x.fraction),ends:st.map(x=>x.nearEnd),s:st.map(x=>+x.s_mm.toFixed(3)),autoOrigin:auto[1].origin,autoNormal:auto[1].normal,origin:p.origin,normal:p.normal,
        state:s.state().basis,raw:[raw.units,raw.displayUnits||null,raw.adjustable],mm:[mm.displayUnits,mm.unitFactor],shown:SL.shownUnit('pressure'),pa,samePa:s.comp().integral.pressure_mean_pa});
      s.dispose();
    """, extra=["ws_slice.js"])
    assert out["fractions"] == [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]               # classic automaticPlanes
    assert out["ends"] == [True, False, False, False, False, True]
    assert out["s"] == pytest.approx([2, 8, 16, 24, 32, 38])
    assert out["origin"] == pytest.approx(out["autoOrigin"]) and out["normal"] == pytest.approx(out["autoNormal"])
    assert out["state"] == "centerline"
    assert out["raw"] == ["m/s", None, False]                                  # the section scale is set in its own panel
    assert out["mm"] == ["mmHg", pytest.approx(1 / 133.322)]
    assert out["shown"] == {"units": "mmHg", "factor": pytest.approx(1 / 133.322)}
    assert out["samePa"] == pytest.approx(out["pa"])                           # the data stay in Pa


def test_probe_card_volume_readings_follow_display_units():
    out = _node(SLICE_TUBE + """
      require(%s); require(%s);
      const PB=ns.probe, r=tube(), ui=ns.ui;
      let i=-1;const P=r.array('vpts');for(let j=0;j<P.length/3;j++)if(Math.abs(P[3*j+2]-20.65)<1e-3&&Math.abs(P[3*j])<0.2&&Math.abs(P[3*j+1])<0.2){i=j;break;}
      const pb=PB.pick(r,{pointIndex:i}), ctx={ui,onRecord(){},onClose(){},field:'speed'};
      const plain=PB.card(r,pb,ctx).textContent;
      ns.display={displayUnit:k=>k==='pressure'?{units:'mmHg',factor:1/133.322}:{units:'cm/s',factor:100}};
      const conv=PB.card(r,pb,ctx).textContent, rec=PB.record(r,i);
      out({plain,conv,speed:rec.speed,pressure:rec.pressure});
    """ % (json.dumps(str(V2 / "ws_ui.js")), json.dumps(str(V2 / "ws_probe.js"))), extra=["ws_slice.js"])
    assert " m/s" in out["plain"] and " Pa" in out["plain"]
    assert "cm/s" in out["conv"] and "mmHg" in out["conv"] and " Pa" not in out["conv"].replace("mmHg", "")
    assert f"{out['speed'] * 100:.3g}" in out["conv"]                          # 0.x m/s → x cm/s
    assert "mL/s" in out["conv"]                                               # flow keeps mL/s


# ------------------------------------------------------------------------------------------------ shell wiring (workspace stub harness)
def _files():
    files = list(WS_FILES)
    files.insert(files.index("ws_bookmarks.js"), "ws_display.js")
    return files


PATCH = r"""
const oc = ns.viewer.create;
ns.viewer.create = (c, o) => { const v = oc(c, o); v.standardView = n => { vcalls.push(v.id + ':std:' + n); return true; }; v.refreshDisplay = () => vcalls.push(v.id + ':refresh');
  v.result = () => null; return v; };
"""


def test_layers_menu_toolbar_reset_key_and_offline_start():
    out = _run(PATCH + """
      await boot();
      await hashTo('#/job/A', 300);
      const S = shellState();
      const layerBtn = walk(app(), e => e.getAttribute && e.getAttribute('aria-label') === '图层、色表与背景')[0];
      layerBtn.click(); await wait(20);
      const menu = byId('ws-menu'), labels = byClass(menu, 'menu-label').map(textOf), heads = byClass(menu, 'menu-head').map(textOf);
      const home = walk(app(), e => e.getAttribute && /复位视角/.test(e.getAttribute('aria-label') || ''))[0];
      const before = vcalls.length;
      await key('0');
      const afterKey = vcalls.slice(before);
      home.click(); await wait(20);
      // toggling a layer stores the preference and refreshes the viewers
      const stl = byClass(menu, 'menu-item').find(e => textOf(e).indexOf('输入 STL') >= 0);
      stl.click(); await wait(30);
      done({labels, heads, home: Boolean(home), afterKey, lastStd: vcalls.filter(x => /:std:/.test(x)).length,
        prefs: JSON.parse(localStorage.getItem('wssv2:display:1')), refreshed: vcalls.some(x => /:refresh/.test(x)), provider: Boolean(ns.viewer.displayProvider)});
    """, files=_files())
    assert out["errors"] == []
    labels = out["labels"]
    # extension layers sit in the 图层 group, right after 分支显隐…, before the labels group
    i = labels.index("分支显隐…")
    assert labels[i + 1:i + 4] == ["输入 STL", "最大 / 最小值标记", "滞留区斜纹"]   # TAWSS opens first: it also marks its minimum
    assert labels.index("标注") > i + 3
    assert out["home"] and any(":std:anterior" in x for x in out["afterKey"]) and out["lastStd"] >= 2
    assert out["prefs"]["layers"]["stl"] is True and out["refreshed"] and out["provider"]


def test_compare_panel_push_buttons_copy_field_and_window():
    out = _run(PATCH + """
      await boot();
      await hashTo('#/job/A?f=tawss', 250);
      await hashTo('#/job/A?f=tawss&v=compare&cmp=B', 400);
      const S = shellState();
      const push = byClass(app(), 'cmp-push')[0];
      const names = push ? byClass(push, 'btn').map(textOf) : [];
      // right side on OSI, left on TAWSS: 以右为准 puts the left on the right's field
      S.cur.compare.field = 'osi';
      byText(push, '以右为准').click(); await wait(60);
      const afterRight = {left: S.cur.field, right: S.cur.compare.field};
      done({names, afterRight, mode: S.cur.compare.mode});
    """, files=_files())
    assert out["errors"] == []
    assert out["names"] == ["以左为准", "以右为准"]
    assert out["afterRight"] == {"left": "osi", "right": "osi"}


def test_offline_package_starts_with_display_options():
    files = [f for f in _files() if f not in {"ws_api.js", "ws_rail.js", "ws_upload.js", "ws_input.js", "ws_compare.js"}]
    out = _run(PATCH + """
      const m = manifestFor('A');
      const s1 = new Element('script'); s1.id = 'wssv2-manifest'; s1.textContent = JSON.stringify(m); body.appendChild(s1);
      const appEl = new Element('div'); appEl.id = 'ws-app'; body.appendChild(appEl);
      await boot();
      await wait(150);
      const home = walk(app(), e => e.getAttribute && /复位视角/.test(e.getAttribute('aria-label') || ''))[0];
      done({display: Boolean(ns.display && ns.display.provider), ext: (ns.ext || []).map(x => x.id), home: Boolean(home), fetches: calls.filter(c => /^(GET|POST|PUT) /.test(c))});
    """, offline=True, files=files)
    assert out["errors"] == []
    assert out["display"] and "display" in out["ext"] and out["home"] and out["fetches"] == []
