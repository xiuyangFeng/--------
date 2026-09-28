"""Node tests for the shared report library ``static/report_common.js`` (contract §12)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "wss_deploy" / "static" / "report_common.js"

# Straight aorta along +x (11 samples, r = 3 mm) with two children leaving (20, 0, 0) at ±y (r = 2 mm).
BIFURCATION = """
const xyz=[],rad=[],seg=[],edges=[];let idx=0;
function add(sid,pts,r){const start=idx;for(const p of pts){xyz.push(...p);rad.push(r);seg.push(sid);idx++;}for(let i=start;i<idx-1;i++)edges.push(i,i+1);}
add(1,Array.from({length:11},(_,i)=>[2*i,0,0]),3);
add(2,Array.from({length:6},(_,i)=>[20+i,2*i,0]),2);
add(3,Array.from({length:6},(_,i)=>[20+i,-2*i,0]),2);
const G=C.buildCenterlineGroups({xyz:new Float32Array(xyz),radius:new Float32Array(rad),edges:new Uint32Array(edges),segment:new Int32Array(seg)},{1:'主动脉',2:'左髂总',3:'右髂总'});
"""


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the shared report library tests")
    program = "const C=require(" + json.dumps(str(COMMON)) + ");\n" + script
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_common_library_parses_and_exports_contract_api():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    subprocess.run(["node", "--check", str(COMMON)], check=True, capture_output=True)
    out = _node("console.log(JSON.stringify(Object.keys(C).sort()))")
    for name in ("buildCenterlineGroups", "projectToCenterline", "arcDistance", "localDiameter", "straightDistance", "newId",
                 "measurementLabel", "probeToTSV", "probeToCSV", "colorbarSVG", "englishLabel", "exportFilename", "builtinPresets",
                 "renderOffscreen", "glossaryPopover", "viewMessaging"):
        assert name in out


def test_centerline_groups_tree_and_arc_distances():
    out = _node(BIFURCATION + """
      const same=C.arcDistance(G,[2,.5,0],[12,.2,0]),cross=C.arcDistance(G,[10,0,0],[25,10,0]),sib=C.arcDistance(G,[22,4,0],[22,-4,0]);
      const viaJunction=C.arcDistance(G,[10,0,0],[20,0,0]).value_mm+C.arcDistance(G,[20,0,0],[25,10,0]).value_mm;
      console.log(JSON.stringify({tree:G.map(g=>[g.segment_id,g.name,g.parent_id,+g.parent_s.toFixed(3),+g.length_mm.toFixed(3),g.children]),
        same:same.value_mm,sameBranch:same.same_branch,cross:+cross.value_mm.toFixed(3),crossPath:cross.path,viaJunction:+viaJunction.toFixed(3),
        sib:+sib.value_mm.toFixed(3),sibPath:sib.path,straight:C.straightDistance([0,0,0],[3,4,0])}));
    """)
    assert out["tree"] == [[1, "主动脉", None, 0, 20, [2, 3]], [2, "左髂总", 1, 20, pytest.approx(11.18, abs=.01), []], [3, "右髂总", 1, 20, pytest.approx(11.18, abs=.01), []]]
    assert out["same"] == 10 and out["sameBranch"] is True
    # Across the bifurcation the arc equals the sum of the two legs through the junction.
    assert out["cross"] == pytest.approx(21.18, abs=.01) and out["cross"] == pytest.approx(out["viaJunction"], abs=1e-6) and out["crossPath"] == [1, 2]
    assert out["sib"] == pytest.approx(8.944, abs=.01) and out["sibPath"] == [2, 1, 3]
    assert out["straight"] == 5


def test_projection_diameter_and_junction_distance():
    out = _node(BIFURCATION + """
      console.log(JSON.stringify({d:C.localDiameter(G,[10,2.9,0]),p:C.projectToCenterline(G,[5,1,0]),child:C.localDiameter(G,[22,4,0]),none:C.projectToCenterline([],[0,0,0])}));
    """)
    d = out["d"]
    assert d["diameter_mm"] == 6 and d["radius_mm"] == 3 and d["segment_id"] == 1 and d["s"] == 10 and d["dist_to_junction_mm"] == 10
    assert out["p"]["s"] == 5 and out["p"]["dist_mm"] == 1 and out["p"]["xyz"] == [5, 0, 0] and out["p"]["row"] == 3
    assert out["child"]["diameter_mm"] == 4 and out["child"]["dist_to_junction_mm"] == pytest.approx(4.472, abs=.01)
    assert out["none"] is None


def test_labels_ids_filenames_and_english_dictionary():
    out = _node("""
      console.log(JSON.stringify({arcEn:C.measurementLabel({kind:'arc',value_mm:12.34,branch:'左髂外'},'en'),diamZh:C.measurementLabel({kind:'diameter',value_mm:8},'zh'),
        en:[C.englishLabel('右髂内高 WSS 区','en'),C.englishLabel('wss','en'),C.englishLabel('主动脉','zh'),C.englishLabel('主动脉','en'),C.englishLabel('左髂外','en'),C.englishLabel('view_top','en'),C.englishLabel(null,'en')],
        file:[C.exportFilename({case_id:'LIU YU/MING',view:'front',field:'wss',scale:2}),C.exportFilename({case_id:'病例-1',scale:4,ext:'svg'}),C.exportFilename({})],
        ids:[C.newId('D',[{id:'D1'},{id:'D3'}]),C.newId('A',['A1']),C.newId('M',[])]}));
    """)
    assert out["arcEn"] == "Arc distance 12.3 mm (Left EIA)" and out["diamZh"] == "管径 8.0 mm · 内切半径 × 2"
    assert out["en"] == ["Right IIA high-WSS region", "WSS · wall shear stress", "主动脉", "Aorta", "Left EIA", "Top", ""]
    assert out["file"] == ["LIU_YU_MING_front_wss_2x.png", "病例-1_custom_wss_4x.svg", "case_custom_wss_1x.png"]
    assert out["ids"] == ["D2", "A2", "M1"]


def test_colorbar_svg_bands_log_and_orientation():
    out = _node("""
      const banded=C.colorbarSVG({colormap:'turbo',min:0,max:10,bands:4,units:'Pa',title:'WSS',lang:'en'});
      const cont=C.colorbarSVG({colormap:'rainbow',min:0,max:10,log:true,units:'Pa',lang:'zh'});
      const horiz=C.colorbarSVG({stops:[[0,'#000000'],[1,'#ffffff']],min:0,max:1,orientation:'horizontal',title:'x<y'});
      console.log(JSON.stringify({banded:{rects:(banded.match(/<rect/g)||[]).length,ticks:(banded.match(/class="tick"/g)||[]).length,title:/>WSS </.test(banded)&&/\\(Pa\\)/.test(banded),svg:/^<svg xmlns="http:\\/\\/www.w3.org\\/2000\\/svg"/.test(banded)},
        cont:{grad:/linearGradient/.test(cont),ticks:(cont.match(/class="tick"/g)||[]).length,log:/对数色标/.test(cont),lowTick:/>0.050</.test(cont)||/>0.05</.test(cont)},
        horiz:{anchor:/text-anchor="middle"/.test(horiz),escaped:/x&lt;y/.test(horiz),mid:C.colorAt([[0,'#000000'],[1,'#ffffff']],.5)}}));
    """)
    assert out["banded"] == {"rects": 5, "ticks": 5, "title": True, "svg": True}
    assert out["cont"]["grad"] is True and out["cont"]["ticks"] == 5 and out["cont"]["log"] is True and out["cont"]["lowTick"] is True
    assert out["horiz"] == {"anchor": True, "escaped": True, "mid": "#808080"}


def test_probe_export_tsv_and_csv():
    out = _node("""
      const rows=[{id:'P1',xyz_mm:[1,2,3],branch:'主动脉, x',segment_id:1,s_from_root_mm:5,radius_mm:2,values:{wss_pa:1.5}},{id:'P2',xyz_mm:[4,5,6],branch:'左髂外',values:{wss_pa:2.25,extra:'a\\tb'}}];
      console.log(JSON.stringify({csv:C.probeToCSV(rows,'zh'),tsv:C.probeToTSV(rows,'en')}));
    """)
    csv = out["csv"]
    assert csv.startswith("﻿编号,x_mm,y_mm,z_mm,分支,分支编号,距入口弧长_mm,半径_mm,wss_pa,extra\r\n")
    assert 'P1,1,2,3,"主动脉, x",1,5,2,1.5,\r\n' in csv and csv.endswith("\r\n")
    lines = out["tsv"].split("\n")
    assert lines[0] == "ID\tx_mm\ty_mm\tz_mm\tbranch\tsegment_id\ts_from_root_mm\tradius_mm\twss_pa\textra"
    assert lines[2].split("\t")[-1] == "a b"  # tabs inside a cell never break the table


def test_builtin_presets_return_partial_states_from_case_context():
    out = _node("""
      const meta={branch_names:{4:'左髂外',1:'主动脉'},findings:{items:[{id:'F3',kind:'low_wss_cluster'},{id:'F7',kind:'high_wss_cluster'}]},
        profiles:{branches:[{segment_id:1,s_from_root_mm:[1,3,5],wss:{p99_pa:[2,9,5]}}]}};
      const sv={front:{position:[0,-1,0],target:[0,0,0],up:[0,0,1]}};
      const wall=C.builtinPresets('wall',meta),vol=C.builtinPresets('volume',meta);
      console.log(JSON.stringify({names:wall.map(p=>p.name),vnames:vol.map(p=>p.name),
        low:wall[0].build({branchNames:meta.branch_names,standardViews:sv}),hot:wall[1].build({findings:meta.findings.items}),along:wall[2].build({branchNames:meta.branch_names,profiles:meta.profiles}),
        pd:vol[0].build({branchNames:meta.branch_names}),sl:vol[1].build({}),cut:vol[2].build({branchNames:meta.branch_names}),
        noSv:'camera' in wall[0].build({branchNames:meta.branch_names}),empty:C.builtinPresets('other',meta).length}));
    """)
    assert out["names"] == ["瘤囊低 WSS 区", "髂分叉热点", "主动脉沿程", "临床视图"]
    assert out["vnames"] == ["沿程压降", "流线全貌", "瘤囊截面系列", "临床视图"]
    assert out["low"]["highlight"]["branch"] == 1 and out["low"]["range"] == {"mode": "fixed", "min": 0, "max": 2} and out["low"]["camera"]["position"] == [0, -1, 0]
    assert out["hot"]["highlight"]["finding"] == "F7" and out["hot"]["highlight"]["top"] is True
    assert out["along"]["ui"] == {"menu": "profiles", "profile": {"branch": 1, "x": "s_from_root_mm", "s": 3}}
    assert out["pd"]["ui"]["profile"]["branch"] == 1 and out["sl"]["opacity"] == .08 and out["cut"]["slice"]["fractions"] == [.4, .6, .8]
    assert out["noSv"] is False and out["empty"] == 0


def test_view_messaging_and_offscreen_helpers_fail_soft_outside_a_browser():
    out = _node("""
      const m=C.viewMessaging({family:'wall'});
      let err=null;try{C.renderOffscreen({});}catch(e){err=e.message;}
      const pop=C.glossaryPopover({glossary:{terms:{p99:{zh:'p99',zh_desc:'x'}}}});
      console.log(JSON.stringify({enabled:m.enabled,ready:m.ready(),err,popShow:pop.show('p99',null)}));
    """)
    assert out["enabled"] is False and out["ready"] is False and "renderer" in out["err"] and out["popShow"] is False


def test_view_messaging_protocol_roundtrip_in_fake_window():
    out = _node("""
      const posted=[];const listeners=[];
      global.window=global;global.location={protocol:'http:',origin:'http://x'};global.parent={postMessage:(m,o)=>posted.push([m,o])};
      global.addEventListener=(n,f)=>listeners.push([n,f]);
      const m=C.viewMessaging({family:'wall',runIdentity:'r1',caseId:'c1',onApplyState:s=>{if(s.bad)throw new Error('nope');},onExport:o=>({png_base64:'AAAA',filename:'f.png',width:o.scale*10,height:10})});
      m.ready({webgl:true});
      const fire=(origin,data)=>listeners.filter(l=>l[0]==='message').forEach(l=>l[1]({origin,data}));
      fire('http://evil',{type:'wss-view:apply-state',state:{},request_id:'x'});
      fire('http://x',{type:'wss-view:apply-state',state:{mode:'wss'},request_id:'a1'});
      fire('http://x',{type:'wss-view:apply-state',state:{bad:true},request_id:'a2'});
      fire('http://x',{type:'wss-view:export',options:{scale:2},request_id:'e1'});
      setTimeout(()=>console.log(JSON.stringify({enabled:m.enabled,posted})),20);
    """)
    assert out["enabled"] is True
    types = [(p[0]["type"], p[0].get("request_id"), p[1]) for p in out["posted"]]
    assert types[0] == ("wss-view:ready", None, "http://x") and out["posted"][0][0]["run_identity"] == "r1"
    assert ("wss-view:applied", "a1", "http://x") in types and ("wss-view:error", "a2", "http://x") in types
    exported = next(p[0] for p in out["posted"] if p[0]["type"] == "wss-view:exported")
    assert exported["request_id"] == "e1" and exported["png_base64"] == "AAAA" and exported["width"] == 20
    assert not any(p[0].get("request_id") == "x" for p in out["posted"])  # foreign origin ignored


def test_cross_section_and_montage_guard_degenerate_input():
    """§15 shape guards: a miss returns the same keys as a hit, and an empty montage is None, not a NaN canvas."""
    out = _node("""
      const cs=C.crossSection({vertices:new Float32Array([0,0,0, 1,0,0, 0,1,0]),faces:new Uint32Array([0,1,2]),origin:[0,0,50],normal:[0,0,1]});
      const doc={createElement:()=>({getContext:()=>({})})};
      console.log(JSON.stringify({keys:Object.keys(cs).sort(),found:cs.found,poly:cs.polygon_world,open:cs.open,
        empty:C.composeMontage({panels:[],columns:2,document:doc})}));
    """)
    assert out["found"] is False and out["poly"] == [] and out["open"] is False
    assert {"plane", "segs", "polygon", "polygon_world", "metrics", "closed", "synthetic", "open", "found"} <= set(out["keys"])
    assert out["empty"] is None


def test_cross_section_metrics_on_tube_and_open_chain():
    """§15: plane ∩ mesh → local loop → polygon metrics; an open chain through an opening is closed synthetically."""
    out = _node("""
      const verts=[],faces=[];const NA=36,NZ=6;
      function tube(cx,cy,R,rx){const base=verts.length/3;for(let iz=0;iz<NZ;iz++)for(let ia=0;ia<NA;ia++){const a=ia/NA*2*Math.PI;verts.push(cx+R*rx*Math.cos(a),cy+R*Math.sin(a),-5+10*iz/(NZ-1));}
        for(let iz=0;iz<NZ-1;iz++)for(let ia=0;ia<NA;ia++){const a0=base+iz*NA+ia,a1=base+iz*NA+(ia+1)%NA,b0=a0+NA,b1=a1+NA;faces.push(a0,a1,b0,a1,b1,b0);}}
      tube(0,0,10,1.5);tube(60,0,4,1);
      const V=new Float32Array(verts),F=new Uint32Array(faces);
      const cs=C.crossSection({vertices:V,faces:F,origin:[0,0,0.3],normal:[0,0,1]});
      const m=cs.metrics;
      // open chain: cut the near tube on one side (drop 6 columns of quads) → a chain that closeChain seals
      const faces2=[];for(let iz=0;iz<NZ-1;iz++)for(let ia=0;ia<NA;ia++){if(ia<6)continue;const a0=iz*NA+ia,a1=iz*NA+(ia+1)%NA,b0=a0+NA,b1=a1+NA;faces2.push(a0,a1,b0,a1,b1,b0);}
      const cs2=C.crossSection({vertices:new Float32Array(verts.slice(0,NA*NZ*3)),faces:new Uint32Array(faces2),origin:[0,0,0.3],normal:[0,0,1]});
      const svg=C.profileSVG({series:[{name:'r',x:[0,1,2],y:[1,2,1.5]}],xLabel:'s (mm)',yLabel:'r (mm)',title:'t'});
      const csv=C.tableToCSV(['a','b'],[[1,'x,y'],[2.5,'q"z']],['origin 1 2 3']);
      console.log(JSON.stringify({found:cs.found,closed:cs.closed,synthetic:cs.synthetic,n:cs.polygon.length,area:m.area_mm2,dmax:m.max_diameter_mm,dmin:m.min_diameter_mm,deq:m.equivalent_diameter_mm,circ:m.circularity,
        cs2:{found:cs2.found,closed:cs2.closed,synthetic:cs2.synthetic,area:cs2.metrics&&cs2.metrics.area_mm2},svgOk:svg.startsWith('<svg')&&svg.includes('s (mm)')&&svg.includes('<path'),csv}));
    """)
    import math
    assert out["found"] and out["closed"] and not out["synthetic"] and out["n"] >= 36
    assert abs(out["area"] - math.pi * 15 * 10) < math.pi * 150 * 0.02          # ellipse 15 × 10 (polygon underestimates slightly)
    assert abs(out["dmax"] - 30) < 0.3 and abs(out["dmin"] - 20) < 0.3
    assert abs(out["deq"] - 2 * math.sqrt(out["area"] / math.pi)) < 1e-6 and 0.85 < out["circ"] < 1.0
    assert out["cs2"]["found"] and out["cs2"]["closed"] and out["cs2"]["synthetic"] and out["cs2"]["area"] > 0.8 * out["area"]
    assert out["svgOk"]
    assert out["csv"].startswith("﻿# origin 1 2 3\r\na,b\r\n1,\"x,y\"\r\n2.5,\"q\"\"z\"\r\n")


def test_clinical_view_preset_exists_for_both_families():
    out = _node("""
      const meta={branch_names:{'0':'主动脉'},findings:{items:[]},profiles:{branches:[]}};
      const w=C.builtinPresets('wall',meta).find(p=>p.name==='临床视图'),v=C.builtinPresets('volume',meta).find(p=>p.name==='临床视图');
      console.log(JSON.stringify({w:w&&w.build({standardViews:{front:{position:[0,0,1],target:[0,0,0],up:[0,1,0]}}}),v:v&&v.build({})}));
    """)
    assert out["w"]["labels"] == {"findings": 5, "branches": True} and out["w"]["mode"] == "wss" and out["w"]["camera"]["position"] == [0, 0, 1]
    assert out["v"]["labels"] == {"findings": 5, "branches": True} and out["v"]["mode"] == "cloud" and out["v"]["field"] == "velocity"


def test_v012_fit_view_frames_points_and_recentres():
    out = _node("""
      const line=[];for(let i=0;i<=100;i++)line.push([0,0,-100+2*i]);
      const v=C.fitView(line,{dir:[0,1,0],up:[0,0,1],fov:35,aspect:1,margin:1});
      // Off-centre cloud: target moves to the middle of the projected extents, wide aspect fits the height.
      const flat=new Float32Array([10,0,0, 30,0,0, 10,0,40, 30,0,40]);
      const w=C.fitView(flat,{dir:[0,1,0],up:[0,0,1],fov:35,aspect:2,margin:1.1});
      console.log(JSON.stringify({v,w,expected:100/Math.tan(17.5*Math.PI/180),empty:C.fitView([], {center:[1,2,3],fallbackDistance:50})}));
    """)
    assert out["v"]["distance"] == pytest.approx(out["expected"], rel=1e-9)
    assert out["v"]["up"] == [0, 0, 1] and out["v"]["target"] == [0, 0, 0]
    assert out["w"]["target"] == pytest.approx([20, 0, 20])
    assert out["w"]["distance"] == pytest.approx(1.1 * 20 / __import__("math").tan(17.5 * __import__("math").pi / 180), rel=1e-6)
    assert out["empty"]["target"] == [1, 2, 3] and out["empty"]["distance"] == 50


def test_v012_format_value_never_uses_exponent_notation():
    out = _node("console.log(JSON.stringify([0.00177,1.572,17,52.6,15544.8,0,1e-7,null,NaN,-0.35].map(v=>C.formatValue(v))))")
    assert out == ["0.0018", "1.57", "17.0", "52.6", "15545", "0", "0", "—", "—", "-0.350"]
    assert not any("e" in s for s in out)


def test_v012_friendly_and_local_time_use_viewer_timezone():
    out = _node("""
      const now=new Date(2026,8,23,9,30).getTime(), iso=d=>d.toISOString();
      console.log(JSON.stringify({
        just:C.friendlyTime(iso(new Date(now-20e3)),now), min:C.friendlyTime(iso(new Date(now-12*60e3)),now),
        today:C.friendlyTime(iso(new Date(2026,8,23,7,5)),now), yday:C.friendlyTime(iso(new Date(2026,8,22,17,37)),now),
        month:C.friendlyTime(iso(new Date(2026,8,20,2,10)),now), year:C.friendlyTime(iso(new Date(2025,0,2,10,0)),now),
        bad:C.friendlyTime('not-a-date',now), local:C.localTime(iso(new Date(2026,8,20,2,10,5)),{seconds:true})}));
    """)
    assert out == {"just": "刚刚", "min": "12 分钟前", "today": "今天 07:05", "yday": "昨天 17:37", "month": "9月20日 02:10",
                   "year": "2025年1月2日", "bad": "not-a-date", "local": "2026-09-20 02:10:05"}


def test_v012_shortcut_matching_rules():
    out = _node("""
      const m=(b,e)=>C.shortcutMatches(b,e);
      console.log(JSON.stringify([m({keys:['1']},{key:'1'}),m({keys:['f']},{key:'F'}),m({keys:['shift+/']},{key:'/',shiftKey:false}),
        m({keys:['shift+/']},{key:'/',shiftKey:true}),m({keys:['r']},{key:'r',ctrlKey:true}),m({keys:['ArrowUp']},{key:'ArrowUp'}),
        C.shortcutRows([{keys:['ArrowUp','k'],label:'上一例',group:'列表'},{keys:['x']}])]));
    """)
    assert out[:6] == [True, True, False, True, False, True]
    assert out[6] == [{"group": "列表", "keys": ["↑", "K"], "label": "上一例"}]


def test_v0122_declutter_labels_separates_overlaps_deterministically():
    out = _node("""
      const items=[{x:100,y:100,w:80,h:20,priority:1},{x:105,y:102,w:80,h:20,priority:3},{x:300,y:100,w:40,h:20},{x:102,y:98,w:80,h:20,priority:2}];
      const a=C.declutterLabels(items,{padding:2}),b=C.declutterLabels(items,{padding:2});
      const boxes=a.map((p,i)=>({l:p.x-items[i].w/2,r:p.x+items[i].w/2,t:p.y-items[i].h/2,b:p.y+items[i].h/2}));
      let overlaps=0;for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){const A=boxes[i],B=boxes[j];if(Math.min(A.r,B.r)>Math.max(A.l,B.l)&&Math.min(A.b,B.b)>Math.max(A.t,B.t))overlaps++;}
      const hide=C.declutterLabels(Array.from({length:30},()=>({x:50,y:50,w:90,h:30})),{maxShift:20,bounds:{width:100,height:100},hideOverflow:true});
      console.log(JSON.stringify({a,same:JSON.stringify(a)===JSON.stringify(b),overlaps,hidden:hide.filter(p=>p.hidden).length,shown:hide.filter(p=>!p.hidden).length}));
    """)
    assert out["same"] and out["overlaps"] == 0
    assert out["a"][1] == {"x": 105, "y": 102, "moved": False, "hidden": False}      # highest priority keeps its anchor
    assert out["a"][2]["moved"] is False                                               # a free label never moves
    assert out["a"][0]["moved"] and out["a"][3]["moved"]
    assert out["shown"] >= 1 and out["hidden"] >= 25


def test_v014_one_palette_source_rainbow_unchanged():
    """F2: WssReportCommon.PALETTES is the only colour table; rainbow keeps its historical stops bit for bit."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "wss_deploy"
    program = r"""
      const legacy=[[0,0,143],[0,32,255],[0,160,255],[0,255,255],[64,255,160],[160,255,64],[255,255,0],[255,160,0],[255,64,0],[190,0,0]];
      const old=t=>{t=Math.min(1,Math.max(0,t));const x=t*(legacy.length-1),i=Math.min(legacy.length-2,Math.floor(x)),u=x-i;return [0,1,2].map(k=>(legacy[i][k]+(legacy[i+1][k]-legacy[i][k])*u)/255);};
      const ts=Array.from({length:101},(_,i)=>i/100);
      console.log(JSON.stringify({names:C.colormapNames(),rainbow:C.PALETTES.rainbow.stops,same:ts.every(t=>JSON.stringify(C.paletteRGB('rainbow',t))===JSON.stringify(old(t))),
        lens:Object.fromEntries(Object.entries(C.PALETTES).map(([k,v])=>[k,v.stops.length])),bwrMid:C.paletteRGB('bwr',.5),turboEnds:[C.paletteRGB('turbo',0),C.paletteRGB('turbo',1)],
        tables:Object.keys(C.colormapTables()),svgStops:C.colormapStops('turbo').length,unknown:C.paletteRGB('nope',0)}));
    """
    out = _node(program)
    assert out["names"] == ["rainbow", "turbo", "bwr", "viridis"] and out["same"] is True
    assert out["rainbow"][0] == [0, 0, 143] and out["rainbow"][-1] == [190, 0, 0]
    assert out["lens"] == {"rainbow": 10, "turbo": 33, "bwr": 7, "viridis": 33}
    assert out["bwrMid"] == [247 / 255] * 3 and out["svgStops"] == 33
    assert out["turboEnds"] == [[48 / 255, 18 / 255, 59 / 255], [122 / 255, 4 / 255, 3 / 255]]
    assert out["tables"] == ["rainbow", "turbo", "bwr", "viridis"] and out["unknown"] == [0, 0, 143 / 255]
    # nobody else carries a stop table any more (turbo's first stop is a fingerprint of a local copy)
    for name in ("report.py", "volume_report.py", "static/volume_viewer.js", "static/app.js", "static/compare.js"):
        text = (root / name).read_text(encoding="utf-8")
        assert "[48,18,59]" not in text and "[68,1,84]" not in text and "[31,78,156]" not in text, name


# §25 截面均值: a straight tube along +x (r = 5 mm, 48 around, rings every 2 mm) with a dense wall point cloud.
TUBE = """
const R=5,NA=48,NZ=21,V=[],F=[];
for(let iz=0;iz<NZ;iz++)for(let ia=0;ia<NA;ia++){const a=2*Math.PI*ia/NA;V.push(2*iz,R*Math.cos(a),R*Math.sin(a));}
for(let iz=0;iz<NZ-1;iz++)for(let ia=0;ia<NA;ia++){const a0=iz*NA+ia,a1=iz*NA+(ia+1)%NA;F.push(a0,a1,a0+NA,a1,a1+NA,a0+NA);}
const G=C.buildCenterlineGroups({xyz:new Float32Array([0,0,0,10,0,0,20,0,0,30,0,0,40,0,0]),radius:new Float32Array([R,R,R,R,R]),
  edges:new Uint32Array([0,1,1,2,2,3,3,4]),segment:new Int32Array([1,1,1,1,1])},{1:'主动脉'});
// prediction points every 0.5 mm along x; the half with cos θ > 0 is sampled four times as densely
const P=[],f=[],g=[];
for(let x=0;x<=40;x+=0.5)for(let k=0;k<160;k++){const a=2*Math.PI*k/160;if(Math.cos(a)<=0&&k%4)continue;P.push(x,R*Math.cos(a),R*Math.sin(a));f.push(2+Math.cos(a));g.push(x);}
const PV=new Float32Array(P);
const nearest=q=>{const out=new Int32Array(q.length/3);for(let j=0;j<out.length;j++){let b=-1,bd=Infinity;
  for(let i=0;i<PV.length/3;i++){const d=(PV[3*i]-q[3*j])**2+(PV[3*i+1]-q[3*j+1])**2+(PV[3*i+2]-q[3*j+2])**2;if(d<bd){bd=d;b=i;}}out[j]=b;}return out;};
const base={groups:G,vertices:new Float32Array(V),faces:new Uint32Array(F),fields:{f:new Float32Array(f),g:new Float32Array(g)},sample:nearest};
"""


def test_section_means_are_length_weighted_line_integrals_around_the_station():
    out = _node(TUBE + """
      const s=C.sectionMeans(Object.assign({point:[13.1,R,0]},base));
      let naive=0,n=0;for(let i=0;i<PV.length/3;i++)if(Math.abs(PV[3*i]-13)<.3){naive+=f[i];n++;}
      console.log(JSON.stringify({found:s.found,closed:s.closed,open:s.open,synthetic:s.synthetic,branch:s.branch,segment:s.segment_id,
        s:s.s_from_root_mm,origin:s.origin,tangent:s.tangent,wall:s.wall_length_mm,gap:s.gap_mm,elements:s.n_elements,points:s.n_points,
        f:s.means.f,g:s.means.g.mean,naive:naive/n,ring:s.polygon_world.length,area:s.metrics.area_mm2}));
    """)
    assert out["found"] and out["closed"] and not out["open"] and not out["synthetic"]
    assert out["branch"] == "主动脉" and out["segment"] == 1
    # the station is the centreline projection of the pick, the normal the downstream tangent
    assert out["s"] == pytest.approx(13.1, abs=1e-4) and out["origin"] == pytest.approx([13.1, 0, 0], abs=1e-4)
    assert out["tangent"] == pytest.approx([1, 0, 0])
    # 48-gon inscribed in r = 5: perimeter 2·48·5·sin(π/48) = 31.39 mm, all of it wall
    assert out["wall"] == pytest.approx(31.39, abs=.02) and out["gap"] == 0 and out["elements"] >= 48
    assert out["ring"] >= 48 and out["area"] == pytest.approx(78.33, abs=.1)
    # ∮ (2 + cos θ) dl / L = 2 although one half of the ring holds 4× more prediction points (their plain mean ≈ 2.38)
    assert out["f"]["mean"] == pytest.approx(2.0, abs=.02) and out["naive"] > 2.3
    assert out["f"]["min"] == pytest.approx(1.0, abs=.02) and out["f"]["max"] == pytest.approx(3.0, abs=.02)
    # values come from the nearest prediction points (0.5 mm rings): x = 13.0 at a station at 13.1
    assert out["g"] == pytest.approx(13.0, abs=1e-6) and out["points"] >= 48


def test_section_means_report_why_there_is_no_section():
    out = _node(TUBE + """
      const none=C.sectionMeans(Object.assign({},base,{groups:[],point:[5,R,0]}));
      const noMesh=C.sectionMeans(Object.assign({},base,{faces:new Uint32Array(0),point:[5,R,0]}));
      // a centreline running on past the mesh (x > 40): the plane at x = 55 cuts no wall
      const G2=C.buildCenterlineGroups({xyz:new Float32Array([0,0,0,20,0,0,40,0,0,60,0,0]),radius:new Float32Array([R,R,R,R]),edges:new Uint32Array([0,1,1,2,2,3]),segment:new Int32Array([1,1,1,1])},{1:'主动脉'});
      const beyond=C.sectionMeans(Object.assign({},base,{groups:G2,point:[55,R,0]}));
      console.log(JSON.stringify({none:[none.found,none.reason],noMesh:[noMesh.found,noMesh.reason,+noMesh.s_from_root_mm.toFixed(3)],
        beyond:[beyond.found,beyond.reason]}));
    """)
    assert out["none"] == [False, "no_centerline"]
    assert out["noMesh"] == [False, "no_mesh", 5.0]
    assert out["beyond"] == [False, "no_contour"]


def test_ring_elements_skip_the_synthetic_closing_edge_and_line_mean_weights_by_length():
    out = _node("""
      const cs={plane:C.planeFrame([0,0,0],[0,0,1]),segs:[[0,0,1,0,NaN,1,2],[1,0,1,3,NaN,2,3],[1,3,0,0,NaN,-1,-1]]};
      const r=C.ringElements(cs);
      console.log(JSON.stringify({n:r.length_mm.length,wall:r.wall_mm,gap:+r.gap_mm.toFixed(4),
        mean:C.lineMean([1,3],[1,3]),skip:C.lineMean([1,NaN,5],[1,9,0]),empty:C.lineMean([],[])}));
    """)
    assert out["n"] == 2 and out["wall"] == 4 and out["gap"] == pytest.approx(3.1623, abs=1e-4)
    assert out["mean"]["mean"] == 2.5 and out["mean"]["min"] == 1 and out["mean"]["max"] == 3 and out["mean"]["length_mm"] == 4
    assert out["skip"]["mean"] == 1 and out["skip"]["n"] == 1   # NaN value and zero length carry no weight
    assert out["empty"]["mean"] is None and out["empty"]["n"] == 0
