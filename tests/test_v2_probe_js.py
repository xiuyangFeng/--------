"""Node tests for the workspace v2 probe card, probe log, measurement and wall regions (second phase S3).

The numbers come from the classic reports' shared library (WssReportCommon.sectionMeans, arcDistance, crossSection…)
and the classic statistics (VolumeViewerCore.statistics); these tests check the wiring on a synthetic wall result —
a straight tube along +z (radius 5 mm, 0 … 40 mm) with prediction points on the wall carrying
WSS = 1 + 0.1 z + 0.5 cos θ Pa — and the classic row / TSV formats.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
FILES = {"three": STATIC_DIR / "three.min.js", "common": STATIC_DIR / "report_common.js", "vv": STATIC_DIR / "volume_viewer.js",
         "util": V2 / "core_util.js", "cmap": V2 / "core_colormap.js", "slice": V2 / "ws_slice.js", "probe": V2 / "ws_probe.js",
         "measure": V2 / "ws_measure.js", "region": V2 / "ws_region.js"}

PRELUDE = """
globalThis.THREE=require(@@three@@);
require(@@common@@); require(@@vv@@); require(@@util@@); require(@@cmap@@); require(@@slice@@); require(@@probe@@); require(@@measure@@); require(@@region@@);
const ns=globalThis.WSSV2, PB=ns.probe, ME=ns.measure, RG=ns.region, C=globalThis.WssReportCommon, VC=globalThis.VolumeViewerCore;
function wallTube(){
  const R=5,NA=48,NZ=41,V=[],F=[],MS=[];
  for(let k=0;k<NZ;k++)for(let a=0;a<NA;a++){const t=2*Math.PI*a/NA;V.push(R*Math.cos(t),R*Math.sin(t),k);MS.push(0);}
  for(let k=0;k+1<NZ;k++)for(let a=0;a<NA;a++){const i=k*NA+a,j=k*NA+(a+1)%NA,i2=i+NA,j2=j+NA;F.push(i,j,j2,i,j2,i2);}
  const P=[],PS=[],PSR=[],PR=[],W=[],T=[];
  for(let z=0.25;z<40;z+=0.5)for(let a=0;a<36;a++){const t=2*Math.PI*(a+0.5)/36;P.push(R*Math.cos(t),R*Math.sin(t),z);PS.push(0);PSR.push(z);PR.push(R);
    W.push(1+0.1*z+0.5*Math.cos(t));T.push(0.5+0.05*z);}
  const CV=[],CS=[],CE=[],CR=[];for(let k=0;k<NZ;k++){CV.push(0,0,k);CS.push(0);CR.push(5);if(k+1<NZ)CE.push(k,k+1);}
  const disp=a=>{const out=new Float32Array(V.length/3);for(let v=0;v<out.length;v++){const z=V[3*v+2],t=Math.atan2(V[3*v+1],V[3*v]);out[v]=a(z,t);}return out;};
  const arrays={mv:new Float32Array(V),mf:new Uint32Array(F),ms:new Int32Array(MS),pv:new Float32Array(P),ps:new Int32Array(PS),p_s:new Float32Array(PSR),p_r:new Float32Array(PR),
    cv:new Float32Array(CV),cs:new Int32Array(CS),ce:new Uint32Array(CE),cr:new Float32Array(CR),
    'f.wss.r':new Float32Array(W),'f.wss.d':disp((z,t)=>1+0.1*z+0.5*Math.cos(t)),'f.tawss.r':new Float32Array(T),'f.tawss.d':disp(z=>0.5+0.05*z)};
  const manifest={result:{family:'wall'},geometry:{display_mesh:{vertices:'mv',faces:'mf',segment:'ms'},points:{xyz:'pv',segment:'ps',s_from_root_mm:'p_s',radius_mm:'p_r'},
      centerline:{xyz:'cv',radius_mm:'cr',edges:'ce',segment:'cs',tangent:null},branches:[{id:0,name:'主动脉',parent:-1,length_mm:40}]},
    fields:[{id:'wss',label:'峰值 WSS',short_label:'WSS',units:'Pa',kind:'scalar',components:1,location:'wall',arrays:{display:'f.wss.d',read:'f.wss.r'}},
            {id:'tawss',label:'TAWSS',short_label:'TAWSS',units:'Pa',kind:'scalar',components:1,location:'wall',arrays:{display:'f.tawss.d',read:'f.tawss.r'}}]};
  return {family:'wall',manifest,has:k=>k in arrays,declared:k=>k in arrays,array:k=>arrays[k]};
}
const out=x=>console.log(JSON.stringify(x));
"""


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the v2 probe tests")
    prelude = PRELUDE
    for k, v in FILES.items():
        prelude = prelude.replace("@@" + k + "@@", json.dumps(str(v)))
    program = prelude + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout)


def test_scripts_parse_and_are_bundled():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    for name in ("ws_probe.js", "ws_measure.js", "ws_region.js"):
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
        assert name in bundle["scripts"] and name not in bundle["offline_exclude"]
        assert bundle["scripts"].index("ws_slice.js") < bundle["scripts"].index(name) < bundle["scripts"].index("ws_shell.js")


def test_wall_probe_section_means_equal_the_shared_library():
    out = _node("""
      const r=wallTube(), hit=[4.99,0.2,20.1], pb=PB.pick(r,{hitXyz:hit,pointIndex:0});
      // the classic call: exact nearest prediction point for the probe and for every ring element
      const P=r.array('pv'), n=P.length/3, near=q=>{let b=-1,bd=Infinity;for(let i=0;i<n;i++){const d=(P[3*i]-q[0])**2+(P[3*i+1]-q[1])**2+(P[3*i+2]-q[2])**2;if(d<bd){bd=d;b=i;}}return b;};
      const groups=C.buildCenterlineGroups({xyz:r.array('cv'),radius:r.array('cr'),edges:r.array('ce'),segment:r.array('cs')},{0:'主动脉'});
      const ref=C.sectionMeans({groups,point:hit,vertices:r.array('mv'),faces:r.array('mf'),fields:{wss:r.array('f.wss.r'),tawss:r.array('f.tawss.r')},
        sample:q=>{const m=q.length/3,o=new Int32Array(m);for(let j=0;j<m;j++)o[j]=near([q[3*j],q[3*j+1],q[3*j+2]]);return o;}});
      out({index:pb.index,refIndex:near(hit),found:pb.section.found,means:pb.section.means,ref:ref.means,s:pb.section.s_from_root_mm,perim:pb.section.wall_length_mm,closed:pb.section.closed});
    """)
    assert out["index"] == out["refIndex"] and out["found"] and out["closed"]
    for f in ("wss", "tawss"):
        for k in ("mean", "min", "max", "length_mm", "n"):
            assert out["means"][f][k] == pytest.approx(out["ref"][f][k], rel=1e-12), (f, k)
    assert out["means"]["wss"]["mean"] == pytest.approx(1 + 0.1 * 20, abs=0.03)       # cos θ averages out around the ring
    assert out["means"]["wss"]["min"] == pytest.approx(2.5, abs=0.06) and out["means"]["wss"]["max"] == pytest.approx(3.5, abs=0.06)
    assert out["s"] == pytest.approx(20.1, abs=0.01) and out["perim"] == pytest.approx(2 * 3.14159 * 5, rel=0.01)


def test_probe_rows_keep_the_classic_schema_and_text_format():
    out = _node("""
      const r=wallTube(), pb=PB.pick(r,{hitXyz:[5,0,20]}), row1=PB.row(r,pb,[]), row2=PB.row(r,pb,[row1]);
      const rows=[row1,row2], tsv=PB.tsv(rows), csv=PB.csv(rows);
      out({row:row1,id2:row2.id,head:tsv.split('\\n')[0].split('\\t'),lines:tsv.split('\\n').length,csvBom:csv.charCodeAt(0),csvCRLF:csv.endsWith('\\r\\n'),same:tsv===C.probeToTSV(rows,'zh')});
    """)
    r = out["row"]
    assert r["id"] == "P1" and out["id2"] == "P2" and r["branch"] == "主动脉" and r["segment_id"] == 0
    assert set(r) >= {"xyz_mm", "s_from_root_mm", "radius_mm", "values", "created_at"} and r["radius_mm"] == 5
    assert list(r["values"])[:2] == ["wss_pa", "tawss_pa"]
    assert {"section_wss_pa", "section_tawss_pa", "section_s_from_root_mm", "section_perimeter_mm"} <= set(r["values"])
    assert out["head"][:8] == ["编号", "x_mm", "y_mm", "z_mm", "分支", "分支编号", "距入口弧长_mm", "半径_mm"]
    assert out["head"][8:10] == ["wss_pa", "tawss_pa"] and out["lines"] == 3 and out["same"]
    assert out["csvBom"] == 0xFEFF and out["csvCRLF"]


def test_measurements():
    out = _node("""
      const r=wallTube(), b=(k,p)=>ME.build(r,k,p,[]);
      const d=b('distance',[[5,0,10],[-5,0,10]]), a=b('arc',[[5,0,10],[5,0,30]]), s=b('segment',[[5,0,30],[0,5,5]]), R=b('diameter',[[5,0,20]]);
      const items=[d.item,a.item,s.item,R.item], tsv=ME.tsv(items);
      out({d:d.item,a:a.item,s:s.item,R:R.item,ids:items.map(x=>x.id),head:tsv.split('\\n')[0],lines:tsv.split('\\n').length,
        exp:0.5*48*25*Math.sin(2*Math.PI/48)});
    """)
    assert out["d"]["value_mm"] == pytest.approx(10) and out["d"]["label"].startswith("直线距离 10.0 mm")
    assert out["a"]["value_mm"] == pytest.approx(20) and out["a"]["same_branch"] and len(out["a"]["path_xyz"]) >= 2
    assert out["s"]["value_mm"] == pytest.approx(25) and out["s"]["kind"] == "segment"
    R = out["R"]
    assert R["method"] == "contour" and R["area_mm2"] == pytest.approx(out["exp"], rel=1e-3)
    assert R["max_diameter_mm"] == pytest.approx(10, abs=0.02) and R["value_mm"] == R["equivalent_diameter_mm"]
    assert all(x[0] == p for x, p in zip(out["ids"], "DCSR")) and out["ids"][0] == "D1"
    assert out["head"] == "id\tkind\tvalue_mm\tbranch\tx1\ty1\tz1\tx2\ty2\tz2" and out["lines"] == 5


def test_wall_regions():
    out = _node("""
      const r=wallTube(), W=r.array('f.wss.r'), P=r.array('pv'), S=r.array('p_s');
      const A=RG.compute(r,{mode:'range',segment:0,s0:20,s1:10},W), B=RG.compute(r,{mode:'sphere',center:[5,0,20],radius:3},W);
      const direct=[];for(let i=0;i<S.length;i++)if(S[i]>=10&&S[i]<=20)direct.push(i);
      const sph=[];for(let i=0;i<S.length;i++){const d=(P[3*i]-5)**2+P[3*i+1]**2+(P[3*i+2]-20)**2;if(d<=9)sph.push(i);}
      out({a:{n:A.points,stats:A.stats,area:A.area_mm2,label:A.label,vm:Array.from(A.vertexMask).reduce((s,x)=>s+x,0)},direct:direct.length,ref:VC.statistics(W,direct),
        b:{n:B.points,area:B.area_mm2},sph:sph.length,supported:RG.supported(r)});
    """)
    assert out["supported"] and out["a"]["n"] == out["direct"] == 20 * 36
    assert out["a"]["stats"] == out["ref"] and out["a"]["stats"]["mean"] == pytest.approx(1 + 0.1 * 15, abs=0.02)
    # a vertex belongs to the stretch through its nearest point's arc; the end rings (z = 10, 20) sit halfway between
    # two points (z ± 0.25) and may fall either side, so the band is 8–10 mm long
    assert 2 * 3.1416 * 5 * 8 * 0.98 <= out["a"]["area"] <= 2 * 3.1416 * 5 * 10 * 1.02 and out["a"]["label"].startswith("主动脉")
    assert out["b"]["n"] == out["sph"] > 0 and 0 < out["b"]["area"] < 3.1416 * 9 * 1.2


def test_volume_probe_row():
    # the volume tube of the section tests, reused through ws_slice's own fixtures is heavier; a small check here:
    # an interior point row carries the classic keys and the section integrals.
    out = _node("""
      const R=5,NA=48,NZ=41,V=[],F=[];
      for(let k=0;k<NZ;k++)for(let a=0;a<NA;a++){const t=2*Math.PI*a/NA;V.push(R*Math.cos(t),R*Math.sin(t),k);}
      for(let k=0;k+1<NZ;k++)for(let a=0;a<NA;a++){const i=k*NA+a,j=k*NA+(a+1)%NA,i2=i+NA,j2=j+NA;F.push(i,j,j2,i,j2,i2);}
      const P=[],W=[],S=[],VEL=[],PR=[],VS=[],VR=[],DW=[];
      for(let z=0.35;z<40;z+=0.7)for(let x=-4.9;x<=4.9;x+=0.7)for(let y=-4.9;y<=4.9;y+=0.7){const r2=x*x+y*y;if(r2>=4.85*4.85)continue;
        P.push(x,y,z);W.push(0);S.push(0);VEL.push(0,0,1-r2/25);PR.push(100-2*z);VS.push(z);VR.push(5);DW.push(5-Math.sqrt(r2));}
      const CV=[],CS=[],CT=[],CE=[],CR=[];for(let k=0;k<NZ;k++){CV.push(0,0,k);CS.push(0);CT.push(0,0,1);CR.push(5);if(k+1<NZ)CE.push(k,k+1);}
      const arrays={mv:new Float32Array(V),mf:new Uint32Array(F),vpts:new Float32Array(P),vis_wall:new Uint8Array(W),vseg:new Int32Array(S),v_s:new Float32Array(VS),v_r:new Float32Array(VR),v_dw:new Float32Array(DW),
        cv:new Float32Array(CV),cs:new Int32Array(CS),ct:new Float32Array(CT),ce:new Uint32Array(CE),cr:new Float32Array(CR),'f.velocity.r':new Float32Array(VEL),'f.pressure.r':new Float32Array(PR)};
      const manifest={result:{family:'volume'},geometry:{display_mesh:{vertices:'mv',faces:'mf'},points:{xyz:'vpts',is_wall:'vis_wall',segment:'vseg',s_from_root_mm:'v_s',radius_mm:'v_r',dist_to_wall_mm:'v_dw'},
          centerline:{xyz:'cv',segment:'cs',tangent:'ct',edges:'ce',radius_mm:'cr'},branches:[{id:0,name:'主动脉'}]},
        fields:[{id:'pressure',arrays:{read:'f.pressure.r'}},{id:'velocity',arrays:{read:'f.velocity.r'}}]};
      const r={family:'volume',manifest,has:k=>k in arrays,declared:k=>k in arrays,array:k=>arrays[k]};
      let i=-1;for(let j=0;j<P.length/3;j++)if(Math.abs(P[3*j+2]-20.65)<1e-3&&Math.abs(P[3*j])<0.2&&Math.abs(P[3*j+1])<0.2){i=j;break;}
      const pb=PB.pick(r,{pointIndex:i}), row=PB.row(r,pb,[{id:'P4'}]);
      out({kind:pb.kind,found:pb.section.found,integrated:pb.section.integrated,row});
    """)
    assert out["kind"] == "interior" and out["found"] and out["integrated"]
    r = out["row"]
    assert r["id"] == "P5" and r["kind"] == "interior" and r["branch"] == "主动脉"
    assert list(r["values"])[:6] == ["speed_m_s", "u", "v", "w", "pressure_pa", "dist_to_wall_mm"]
    assert {"section_area_mm2", "section_speed_mean_m_s", "section_normal_mean_m_s", "section_flow_ml_s", "section_pressure_mean_pa"} <= set(r["values"])
    assert r["values"]["section_speed_mean_m_s"] == pytest.approx(0.5, abs=0.04)
