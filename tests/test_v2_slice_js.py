"""Node tests for the workspace v2 section tool (ws_slice.js, second phase S1).

The numbers come from the classic volume report's numerical core (volume_viewer.js → VolumeViewerCore); these tests
check the plane state, the wiring into that core, the area integrals on a synthetic straight tube (radius 5 mm,
parabolic axial flow v = 1 − r²/25 m/s, linear pressure 100 − 2 z Pa) and that the workspace gives the same area
integrals as the classic report's probe section on the same plane.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"

PRELUDE = """
globalThis.THREE=require(@@three@@);
require(@@common@@);
require(@@vv@@);
require(@@util@@);
require(@@cmap@@);
require(@@slice@@);
const ns=globalThis.WSSV2, SL=ns.slice, VC=globalThis.VolumeViewerCore, C=globalThis.WssReportCommon;
// A straight tube along +z: radius 5 mm, 0 … 40 mm, 48 facets, no caps.
function tube(){
  const R=5,NA=48,NZ=41,V=[],F=[];
  for(let k=0;k<NZ;k++)for(let a=0;a<NA;a++){const t=2*Math.PI*a/NA;V.push(R*Math.cos(t),R*Math.sin(t),k);}
  for(let k=0;k+1<NZ;k++)for(let a=0;a<NA;a++){const i=k*NA+a,j=k*NA+(a+1)%NA,i2=i+NA,j2=j+NA;F.push(i,j,j2,i,j2,i2);}
  const P=[],W=[],S=[],VEL=[],PR=[];
  for(let z=0.35;z<40;z+=0.7)for(let x=-4.9;x<=4.9;x+=0.7)for(let y=-4.9;y<=4.9;y+=0.7){const r2=x*x+y*y;if(r2>=4.85*4.85)continue;
    P.push(x,y,z);W.push(0);S.push(0);VEL.push(0,0,1-r2/25);PR.push(100-2*z);}
  for(let a=0;a<60;a++){const t=2*Math.PI*a/60;P.push(5*Math.cos(t),5*Math.sin(t),20);W.push(1);S.push(0);VEL.push(0,0,0);PR.push(60);}
  const CV=[],CS=[],CT=[],CE=[],CR=[];
  for(let k=0;k<NZ;k++){CV.push(0,0,k);CS.push(0);CT.push(0,0,1);CR.push(5);if(k+1<NZ)CE.push(k,k+1);}
  const WP=[];for(let i=0;i<V.length/3;i++)WP.push(100-2*V[3*i+2]);
  const speed=[];for(let i=0;i<VEL.length/3;i++)speed.push(Math.hypot(VEL[3*i],VEL[3*i+1],VEL[3*i+2]));
  const arrays={mv:new Float32Array(V),mf:new Uint32Array(F),vpts:new Float32Array(P),vis_wall:new Uint8Array(W),vseg:new Int32Array(S),
    cv:new Float32Array(CV),cs:new Int32Array(CS),ct:new Float32Array(CT),ce:new Uint32Array(CE),cr:new Float32Array(CR),
    'f.velocity.r':new Float32Array(VEL),'f.speed.r':new Float32Array(speed),'f.pressure.r':new Float32Array(PR),'f.wall_pressure.d':new Float32Array(WP)};
  const manifest={result:{family:'volume'},geometry:{display_mesh:{vertices:'mv',faces:'mf'},points:{xyz:'vpts',is_wall:'vis_wall',segment:'vseg'},
      centerline:{xyz:'cv',segment:'cs',tangent:'ct',edges:'ce',radius_mm:'cr'},branches:[{id:0,name:'主动脉'}]},
    fields:[{id:'pressure',arrays:{read:'f.pressure.r'}},{id:'speed',arrays:{read:'f.speed.r'}},{id:'velocity',arrays:{read:'f.velocity.r'}},{id:'wall_pressure',arrays:{display:'f.wall_pressure.d'}}]};
  return {family:'volume',manifest,has:k=>k in arrays,declared:k=>k in arrays,array:k=>arrays[k]};
}
const out=x=>console.log(JSON.stringify(x));
"""


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the v2 section tests")
    prelude = PRELUDE
    for k, v in {"three": STATIC_DIR / "three.min.js", "common": STATIC_DIR / "report_common.js", "vv": STATIC_DIR / "volume_viewer.js",
                 "util": V2 / "core_util.js", "cmap": V2 / "core_colormap.js", "slice": V2 / "ws_slice.js"}.items():
        prelude = prelude.replace("@@" + k + "@@", json.dumps(str(v)))
    program = prelude + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout)


def test_slice_script_parses_and_is_bundled():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    subprocess.run(["node", "--check", str(V2 / "ws_slice.js")], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    assert "volume_viewer.js" in bundle["legacy_scripts"]
    assert bundle["legacy_scripts"].index("report_common.js") < bundle["legacy_scripts"].index("volume_viewer.js")
    assert "ws_slice.js" in bundle["scripts"] and "ws_slice.js" not in bundle["offline_exclude"]
    assert bundle["scripts"].index("ws_slice.js") < bundle["scripts"].index("ws_shell.js")


def test_classic_viewer_stops_before_its_page_code_without_a_classic_report():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    program = ("globalThis.document={getElementById:()=>null};"
               "require(" + json.dumps(str(STATIC_DIR / "report_common.js")) + ");"
               "require(" + json.dumps(str(STATIC_DIR / "volume_viewer.js")) + ");"
               "const c=globalThis.VolumeViewerCore;"
               "console.log(JSON.stringify({core:typeof c.sliceMapCore,grid:typeof c.sliceGrid,paint:typeof c.paintSection}));")
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout) == {"core": "function", "grid": "function", "paint": "function"}


def test_tube_section_numbers():
    out = _node("""
      const r=tube(), M=SL.model(r), st=SL.defaultState(M,{xyz:[0,0,20]});
      const c=SL.compute(st,M,{});
      const expected=0.5*48*25*Math.sin(2*Math.PI/48);
      out({basis:st.basis,segment:st.segment,fraction:st.fraction,origin:c.plane.origin,normal:c.plane.normal,area:c.section.area_mm2,expected,
        eq:c.section.equivalent_diameter_mm,outline:c.outlineState,quantity:c.quantity,count:c.stats.count,slab:c.indices.length,
        it:c.integral,range:c.range,samples:c.data.finite.length,grid:[c.grid.nx,c.grid.ny],wallInSlab:c.indices.some(i=>r.array('vis_wall')[i])});
    """)
    assert out["basis"] == "centerline" and out["segment"] == 0 and out["fraction"] == pytest.approx(0.5)
    assert out["origin"] == pytest.approx([0, 0, 20]) and out["normal"] == pytest.approx([0, 0, 1])
    assert out["outline"] == "closed" and out["quantity"] == "speed"
    assert out["area"] == pytest.approx(out["expected"], rel=1e-3)
    assert out["count"] == out["slab"] > 100 and not out["wallInSlab"]
    it = out["it"]
    assert it["integrated"] and it["samples"] > 100 and it["direct_fraction"] > 0.8
    # area means of the parabolic profile: ½ of the centre speed; flow = mean(v·n) × area; pressure at z = 20
    assert it["speed_mean_m_s"] == pytest.approx(0.5, abs=0.03)
    assert it["normal_mean_m_s"] == pytest.approx(it["speed_mean_m_s"], abs=1e-9)
    assert it["flow_ml_s"] == pytest.approx(it["normal_mean_m_s"] * out["area"], rel=1e-12)
    # the synthetic points sit in layers 0.7 mm apart (z = 19.25, 19.95, 20.65 in the slab); each cell takes its
    # nearest point in the plane, so the mean is one layer's pressure 100 − 2 z
    assert 100 - 2 * 20.65 - 1e-6 <= it["pressure_mean_pa"] <= 100 - 2 * 19.25 + 1e-6
    assert out["range"]["source"] == "section" and not out["range"]["diverging"] and 0 <= out["range"]["min"] < out["range"]["max"] <= 1
    assert out["grid"] == [120, 120]


def test_area_integrals_equal_the_classic_probe_section_on_the_same_plane():
    # The classic report's probe section (volume_viewer.js sectionIntegralAt): station on the centreline, plane from
    # report_common.stationSection, slab of the same thickness, points inside the outline, sectionIntegral on 64².
    out = _node("""
      const r=tube(), M=SL.model(r), A=M.A;
      const groups=C.buildCenterlineGroups({xyz:A.cv,radius:A.cr,edges:A.ce,segment:A.cs},{0:'主动脉'});
      const probe=[1.2,-0.8,13.3];
      const st=C.stationSection({groups,point:probe,vertices:A.V,faces:A.F,wallValues:A.wallP});
      const cs=st.section,plane=cs.plane,thickness=2,xy=[],idx=[];
      const point=(a,i)=>[a[3*i],a[3*i+1],a[3*i+2]];
      for(const i of VC.slabIndices(A.pts,A.walls,A.segs,plane,thickness)){const q=[0,1,2].map(k=>A.pts[3*i+k]-plane.origin[k]);
        const x=q[0]*plane.u[0]+q[1]*plane.u[1]+q[2]*plane.u[2],y=q[0]*plane.v[0]+q[1]*plane.v[1]+q[2]*plane.v[2];if(VC.pointInLoop(cs.segs,x,y)){xy.push([x,y]);idx.push(i);}}
      const speed=VC.speedField(A.velocity);
      const quantities={speed:idx.map(i=>speed[i]),normal:idx.map(i=>{const v=point(A.velocity,i);return v[0]*plane.normal[0]+v[1]*plane.normal[1]+v[2]*plane.normal[2];}),pressure:idx.map(i=>A.pressure[i])};
      const cl=VC.sectionIntegral(cs.segs,xy,quantities,{grid:64});
      const state=Object.assign(SL.defaultState(M,{}),{basis:'pick',pick:{origin:st.origin,normal:st.tangent,picks:1},thickness});
      const c=SL.compute(state,M,{});
      out({classic:{area:cs.metrics.area_mm2,speed:cl.quantities.speed.mean,normal:cl.quantities.normal.mean,pressure:cl.quantities.pressure.mean,n:idx.length,supported:cl.supported},
        ws:{area:c.section.area_mm2,speed:c.integral.speed_mean_m_s,normal:c.integral.normal_mean_m_s,pressure:c.integral.pressure_mean_pa,n:c.integral.samples,supported:c.integral.direct_fraction},
        planeU:plane.u,wsU:c.plane.u});
    """)
    assert out["planeU"] == pytest.approx(out["wsU"], abs=1e-12)
    for k in ("area", "speed", "normal", "pressure", "supported"):
        assert out["ws"][k] == pytest.approx(out["classic"][k], rel=1e-9, abs=1e-12), k
    assert out["ws"]["n"] == out["classic"]["n"]


def test_scale_modes_moves_and_picks():
    out = _node("""
      const r=tube(), M=SL.model(r), base=SL.defaultState(M,{xyz:[0,0,20]});
      const normal=SL.compute(Object.assign({},base,{quantity:'normal'}),M,{});
      const pressure=SL.compute(Object.assign({},base,{quantity:'pressure'}),M,{});
      const manual=SL.compute(Object.assign({},base,{range:'manual',manual:{min:0.1,max:0.9}}),M,{});
      const glob=SL.compute(Object.assign({},base,{range:'global'}),M,{global:q=>({min:0,max:2,log:false})});
      const s=SL.create(null,r,{hint:{xyz:[0,0,20]}});
      const f0=s.state().fraction; s.moveAlong(4); s.recomputeNow(); const f1=s.state().fraction;
      s.rotate(200,-300); s.nudgeThickness(40); const rot=s.state();
      s.addPick([5,0,10]); const one=s.state(); s.recomputeNow(); const p1=s.comp().plane;
      s.addPick([-5,0,14]); s.recomputeNow(); const two=s.state(), p2=s.comp().plane;
      const on=(p,x)=>Math.abs((x[0]-p.origin[0])*p.normal[0]+(x[1]-p.origin[1])*p.normal[1]+(x[2]-p.origin[2])*p.normal[2]);
      for(let i=0;i<40;i++)s.moveAlong(1); const shifted=s.state().shift;
      s.toCenterline(); const back=s.state();
      out({normal:normal.range,pressureQ:pressure.quantity,pressureRange:pressure.range,manual:manual.range,glob:glob.range,f0,f1,rot,
        one:{basis:one.basis,picks:one.pick.picks},p1n:p1.normal,two:{picks:two.pick.picks,chord:two.pick.chord_mm},on1:on(p2,[5,0,10]),on2:on(p2,[-5,0,14]),
        shifted,back:{basis:back.basis,seg:back.segment,fraction:back.fraction,pitch:back.pitch}});
    """)
    assert out["normal"]["diverging"] and out["normal"]["min"] == pytest.approx(-out["normal"]["max"])
    assert out["pressureQ"] == "pressure" and out["pressureRange"]["min"] < 60 < out["pressureRange"]["max"]
    assert out["manual"]["source"] == "manual" and (out["manual"]["min"], out["manual"]["max"]) == (0.1, 0.9)
    assert out["glob"]["source"] == "global" and out["glob"]["max"] == 2
    assert out["f1"] - out["f0"] == pytest.approx(4 / 40, rel=1e-6)          # 4 mm along a 40 mm branch
    assert out["rot"]["yaw"] == 85 and out["rot"]["pitch"] == -85 and out["rot"]["thickness"] == 12
    assert out["one"] == {"basis": "pick", "picks": 1} and out["p1n"] == pytest.approx([0, 0, 1])
    assert out["two"]["picks"] == 2 and out["two"]["chord"] == pytest.approx((100 + 16) ** 0.5)
    assert out["on1"] < 1e-9 and out["on2"] < 1e-9
    assert out["shifted"] == 25                                              # ±25 mm along a picked plane's normal
    assert out["back"]["basis"] == "centerline" and out["back"]["seg"] == 0 and out["back"]["pitch"] == 0


def test_map_paints_inside_the_outline_only():
    # A minimal 2-D context with pixel access: the filled map is painted per pixel inside the wall outline.
    out = _node("""
      function ctx2d(W,H){const data=new Uint8ClampedArray(4*W*H);let last={};
        return {data,getImageData:(x,y,w,h)=>{const d=new Uint8ClampedArray(4*w*h);for(let j=0;j<h;j++)for(let i=0;i<w;i++)for(let k=0;k<4;k++)d[4*(j*w+i)+k]=data[4*((y+j)*W+x+i)+k];return {data:d,width:w,height:h};},
          putImageData:(img,x,y)=>{for(let j=0;j<img.height;j++)for(let i=0;i<img.width;i++)for(let k=0;k<4;k++)data[4*((y+j)*W+x+i)+k]=img.data[4*(j*img.width+i)+k];},
          clearRect(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){},fillText(){},strokeRect(){},setLineDash(){},set fillStyle(v){},set strokeStyle(v){},set lineWidth(v){},set font(v){},set textAlign(v){},set lineCap(v){}};}
      const W=200,H=200,cx=ctx2d(W,H),canvas={width:W,height:H,getContext:()=>cx};
      const r=tube(), M=SL.model(r), c=SL.compute(SL.defaultState(M,{xyz:[0,0,20]}),M,{});
      const map=SL.paint(canvas,c,{arrows:true,velocity:M.A.velocity,scaleBar:true});
      const alpha=(x,y)=>cx.data[4*(y*W+x)+3];
      const centre=SL.readAt(map,c,W/2,H/2), corner=SL.readAt(map,c,2,2);
      out({mid:alpha(W/2,H/2),corner:alpha(2,2),centre,cornerRead:corner});
    """)
    assert out["mid"] == 255 and out["corner"] == 0
    assert out["centre"]["value"] == pytest.approx(1.0, abs=0.08) and not out["centre"]["fromWall"]
    assert out["cornerRead"] is None


def test_series_stations_follow_the_classic_fractions_and_share_one_range():
    out = _node("""
      const r=tube(), M=SL.model(r), st=SL.defaultState(M,{xyz:[0,0,20]});
      const S=SL.series(st,M,6,{}), N=SL.series(Object.assign({},st,{quantity:'normal'}),M,6,{}), E=SL.series(st,M,4,{});
      const own=S.stations.map(x=>SL.compute(Object.assign({},st,{fraction:x.fraction}),M,{}).range);
      out({fractions:S.stations.map(x=>x.fraction),z:S.stations.map(x=>x.comp.plane.origin[2]),seg:S.segment,range:S.range,
        lo:Math.min(...own.map(r=>r.min)),hi:Math.max(...own.map(r=>r.max)),same:S.stations.every(x=>x.comp.range===S.range),
        normal:N.range,four:E.stations.map(x=>x.fraction),q:S.stations.map(x=>x.comp.integral.flow_ml_s)});
    """)
    assert out["fractions"] == [0.05, 0.2, 0.4, 0.6, 0.8, 0.95] and out["seg"] == 0
    assert out["z"] == pytest.approx([2, 8, 16, 24, 32, 38])
    assert out["range"]["shared"] and out["same"]
    assert (out["range"]["min"], out["range"]["max"]) == pytest.approx((out["lo"], out["hi"]))
    assert out["normal"]["diverging"] and out["normal"]["min"] == pytest.approx(-out["normal"]["max"])
    assert out["four"] == pytest.approx([0.05, 0.35, 0.65, 0.95])
    assert max(out["q"]) - min(out["q"]) < 0.05 * max(out["q"])   # same flow through every station of a straight tube


def test_region_between_two_positions_and_its_pressure_drop():
    # pressure 100 − 2 z; region 25–75 % of the 40 mm tube = z 10–30 mm; the proximal / distal 10 % (2 mm) of the
    # stretch sit around z ≈ 11 and z ≈ 29, so the drop ≈ 2 × 18 = 36 Pa
    out = _node("""
      const r=tube(), M=SL.model(r), R=SL.region(M,{segment:0,lo:75,hi:25});
      const arc=SL.pointArcs(M), direct=VC.regionIndices(M.interior,M.A.segs,arc,0,10,30);
      out({lo:R.lo_mm,hi:R.hi_mm,count:R.count,direct:direct.length,drop:R.drop,speed:R.speed,pressure:R.pressure,
        ends:R.ends.map(e=>({n:e.ring?e.ring.length:0,z:e.ring?e.ring.reduce((s,q)=>s+q[2],0)/e.ring.length:null}))});
    """)
    assert (out["lo"], out["hi"]) == pytest.approx((10, 30))
    assert out["count"] == out["direct"] > 1000
    assert out["drop"]["drop"] == pytest.approx(36, abs=1.5)
    assert out["drop"]["proximal"] > out["drop"]["distal"]
    assert out["pressure"]["min"] == pytest.approx(100 - 2 * 30, abs=1.5)
    assert 0 < out["speed"]["mean"] < out["speed"]["max"] <= 1
    assert [e["z"] for e in out["ends"]] == pytest.approx([10, 30], abs=1e-3) and all(e["n"] >= 40 for e in out["ends"])


def test_session_region_and_cut_state():
    out = _node("""
      const r=tube(), s=SL.create(null,r,{hint:{xyz:[0,0,20]}});
      s.setMore('region'); const on=s.region(), res=s.regionResult();
      s.regionFromSlice('lo'); const fromSlice=s.region();
      s.setMore(null); const off=s.region(), offRes=s.regionResult();
      s.set({cut:'pos'}); s.recomputeNow(); const cut=s.cut();
      out({on:on.on,seg:on.segment,count:res&&res.count,lo:fromSlice.lo,hi:fromSlice.hi,off:off.on,offRes,cut});
    """)
    assert out["on"] and out["seg"] == 0 and out["count"] > 0
    assert out["lo"] == pytest.approx(50) and out["hi"] == 90
    assert not out["off"] and out["offRes"] is None and out["cut"] == "pos"
