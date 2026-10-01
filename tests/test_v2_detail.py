"""Node tests for lane D of the workspace v2 second phase (S6d, result details — PHASE2_LANES.md §5 D).

Pure parts: the branch unrolled map (ws_unroll.js, the classic drawUnroll mapping), the profile curves and their SVG
export (ws_profile.js, the classic profSpec / profileSVGs), the overview completions (ws_overview.js: cycle metrics,
lumen diameter strip, per-branch table, follow-up curves) and the 工具-tab models (ws_detail.js).  Shell integration
runs on the workspace harness of tests/test_v2_workspace_js.py (stub DOM, canned service, stub kernel) with the lane's
files added: the 「沿程」 tab, the 工具 sections (metadata with tags / notes, outlet override, delete + undo) and the keys.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
PURE = [STATIC_DIR / "report_common.js", V2 / "core_util.js", V2 / "ws_ui.js", V2 / "ws_overview.js", V2 / "ws_lens.js", V2 / "ws_detail.js", V2 / "ws_unroll.js", V2 / "ws_profile.js"]
LANE = ["ws_detail.js", "ws_unroll.js", "ws_profile.js"]


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the lane D tests")
    prelude = "".join("require(" + json.dumps(str(f)) + ");\n" for f in PURE)
    prelude += "const ns=globalThis.WSSV2, U=ns.unroll, P=ns.profile, D=ns.detail, OV=ns.overview, L=ns.lens; const out=x=>console.log(JSON.stringify(x));\n"
    program = prelude + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout)


def _workspace():
    spec = importlib.util.spec_from_file_location("ws_harness", Path(__file__).with_name("test_v2_workspace_js.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shell(scenario: str, offline: bool = False) -> dict:
    ws = _workspace()
    files = list(ws.WS_FILES)
    at = files.index("ws_lens.js") + 1
    files[at:at] = LANE
    if offline:
        files = [f for f in files if f not in ws.OFFLINE_EXCLUDE]
    return ws._run(scenario, offline=offline, files=["../report_common.js"] + files)


def test_scripts_parse_and_sit_right_below_the_lane_placeholder():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    for name in LANE + ["ws_overview.js", "ws_lens.js", "ws_shell.js"]:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    s = bundle["scripts"]
    assert s[s.index("ws_detail.js"):s.index("ws_detail.js") + 3] == LANE
    assert not set(LANE) & set(bundle["offline_exclude"])          # the views work in the offline report too
    srcs = re.findall(r'<script src="/static/v2/([^"]+)"', (V2 / "index.html").read_text(encoding="utf-8"))
    assert srcs[srcs.index("ws_detail.js"):srcs.index("ws_detail.js") + 3] == LANE


def test_unrolled_map_is_the_classic_mapping():
    out = _node(r"""
      // synthetic points: two branches, a NaN value, a point on each edge of the range
      const s=new Float32Array([0,10,10,5,2,2,30,40,35]), th=new Float32Array([-Math.PI,Math.PI,0,0.1,1,1,0,0,-1]);
      const seg=new Int32Array([0,0,0,0,0,0,1,1,1]), v=new Float32Array([1,2,3,NaN,5,4,7,8,9]);
      const br=U.branches(seg,s,{geometry:{branches:[{id:0,name:'主动脉'},{id:1,name:'右髂总'}]}});
      const W=11,H=7, g=U.grid({s,theta:th,segment:seg,values:v,segmentId:0,sMin:br[0].sMin,sMax:br[0].sMax,width:W,height:H});
      // the classic drawUnroll loop, verbatim formula (report.py)
      const ref=new Float32Array(W*H).fill(-Infinity);
      for(let i=0;i<seg.length;i++){if(seg[i]!==0||!Number.isFinite(v[i]))continue;
        const x=Math.max(0,Math.min(W-1,Math.floor((s[i]-br[0].sMin)/(br[0].sMax-br[0].sMin||1)*(W-1)))),y=Math.max(0,Math.min(H-1,Math.floor((1-(th[i]+Math.PI)/(2*Math.PI))*(H-1))));
        ref[y*W+x]=Math.max(ref[y*W+x],v[i]);}
      const same=Array.from(g.max).every((x,k)=>x===ref[k]);
      const pk=U.pickAt(g,(Math.floor(2/10*(W-1))+0.5)/W,(Math.floor((1-(1+Math.PI)/(2*Math.PI))*(H-1))+0.5)/H,0);
      const miss=U.pickAt(g,0.3,0.3,0), near=U.pickAt(g,0.3,0.3,1);
      out({br, same, n:g.n, nMissing:g.nMissing, pk, miss, near:near&&near.index, frac:U.fractionOf(g,10,0), sAt:U.sAt(g,0.5), thAt:U.thetaAt(g,0.5),
        sizes:[U.gridSize(58000,288,84,4,1.7), U.gridSize(2867,700,64,4,0.87), U.gridSize(100,700,64,4), U.gridSize(1e6,288,84,4,3)]});
    """)
    assert out["same"] is True and out["n"] == 5 and out["nMissing"] == 1           # a missing value never paints a cell
    assert [(b["segmentId"], b["name"], b["n"], b["nS"], b["sMin"], b["sMax"]) for b in out["br"]] == [(0, "主动脉", 6, 4, 0, 10), (1, "右髂总", 3, 3, 30, 40)]
    assert out["pk"]["index"] == 4 and out["pk"]["value"] == 5                      # the cell keeps the largest value (5 over 4)
    assert out["miss"] is None and out["near"] == 4                             # the nearest filled cell within reach
    assert out["frac"] == {"fx": 1, "fy": 0.5} and out["sAt"] == 5 and abs(out["thAt"]) < 1e-12
    big, iliac, tiny, capped = out["sizes"]
    assert big["width"] <= 288 and big["height"] <= 84 and abs(big["width"] / big["height"] - 1.7) < 0.2
    assert iliac == {"width": 25, "height": 29} or abs(iliac["width"] / iliac["height"] - 0.87) < 0.1
    assert tiny["width"] >= 8 and tiny["height"] >= 8 and capped == {"width": 288, "height": 84}


def _profiles_fixture() -> str:
    return r"""
      const m={result:{family:'wall'}, job:{display_name:'CASE'},
        fields:[{id:'wss',units:'Pa',display:{thresholds:[0.4,4,7]}},{id:'tawss',units:'Pa'},{id:'osi',units:'1'},{id:'rrt',units:'1/Pa',display:{thresholds:[5,10,20]}}],
        geometry:{branches:[{id:0,name:'主动脉'},{id:2,name:'左髂总'}]},
        analysis:{cycle:{stagnation:{criteria:{tawss_lt_pa:0.4,osi_gt:0.1}}},
          morphology:{branches:[{segment_id:0,stations:{s_from_root_mm:[1,3,5],max_diameter_mm:[20,22,30],equivalent_diameter_mm:[19,21,28]}}]},
          profiles:{bin_mm:2,definition:'按弧长分箱',branches:[
            {segment_id:0,name:'主动脉',s_local_mm:[1,3,5],s_from_root_mm:[1,3,5],radius_mm:[10,11,15],
              wss:{n:[5,0,7],mean_pa:[1,2,3],p99_pa:[4,5,6],min_pa:[0.5,0.6,0.7]},tawss:{mean_pa:[0.3,0.5,0.2],min_pa:[0.1,0.2,0.1]},
              osi:{mean:[0.1,0.2,0.3],p90:[0.2,0.3,0.4]},rrt:{mean:[3,4,5],p90:[6,7,8]}},
            {segment_id:2,name:'左髂总',s_local_mm:[1,3],s_from_root_mm:[213,215],radius_mm:[5,5],wss:{n:[3,3],mean_pa:[2,2],p99_pa:[8,9],min_pa:[1,1]}}]}}};
    """


def test_profile_series_follow_the_classic_spec():
    out = _node(_profiles_fixture() + r"""
      const ids=['wss','tawss','osi','rrt','ecap'].map(f=>{const s=P.spec(m,f);return s?[s.id,s.primary.name,s.secondary.name,s.hlines.map(h=>h.v),s.missing||null]:null;});
      const md=P.model(m,'wss',0), mt=P.model(m,'tawss',0,[0,6]);
      const vol={result:{family:'volume'},analysis:{profiles:{bin_mm:2,branches:[{segment_id:0,s_from_root_mm:[1],s_local_mm:[1],radius_mm:[9],volume:{n:[4],speed_mean_m_s:[0.3],speed_max_m_s:[0.5],pressure_mean_pa:[10],pressure_min_pa:[-5]}}]}}};
      const vs=[P.spec(vol,'speed'),P.spec(vol,'wall_pressure')].map(s=>[s.id,s.primary.name,s.secondary.name,s.primary.get(vol.analysis.profiles.branches[0])[0]]);
      const svgs=P.exportSVGs(m,'wss',0,{caseName:'CASE'}), svgT=P.exportSVGs(m,'tawss',0,{caseName:'CASE'});
      out({ids, primary:md.primary.map(x=>Number.isNaN(x)?null:x), xr:mt.xRange, bin:[P.binAt(md,3.9),P.binAt(md,9)], stations:P.stationSeries(m,md.branch,'max'), vs,
        files:svgs.map(x=>x.filename), wssSvg:svgs[0].svg, rSvg:svgs[1].svg, tFiles:svgT.map(x=>x.filename), tSvg:svgT[0].svg, ticks:P.niceTicks(0,6.2,3)});
    """)
    assert out["ids"][0] == ["wss", "p99", "均值", [], None]
    assert out["ids"][1] == ["tawss", "均值", "最低", [0.4], None]
    assert out["ids"][2] == ["osi", "p90", "均值", [0.1], None]
    assert out["ids"][3] == ["rrt", "p90", "均值", [5], None]
    assert out["ids"][4] == ["wss", "p99", "均值", [], "ecap"]                    # no ECAP bins: peak WSS, said so
    assert out["primary"] == [4, None, 6]                                         # a bin without points is a gap
    assert out["xr"] == [0, 6] and out["bin"] == [1, -1]
    assert out["stations"] == [20, 22, 30]
    assert out["vs"] == [["speed", "截面平均", "最大", 0.3], ["pressure", "截面平均", "最低", 10]]
    assert out["files"] == ["CASE_profile_主动脉_wss.svg", "CASE_profile_主动脉_radius.svg"]
    for name in ("均值", "p99", "最低", "低 0.4 Pa", "高 4.0 Pa", "极高 7.0 Pa", "距入口弧长 (mm)"):
        assert name in out["wssSvg"], name
    assert "管腔最大直径" in out["rSvg"] and "等效直径" in out["rSvg"] and "半径 / 直径 (mm)" in out["rSvg"]
    assert out["tFiles"][0].endswith("_tawss.svg") and "低 TAWSS 0.4 Pa" in out["tSvg"]
    assert out["ticks"] == [0, 2.5, 5]


def test_overview_models_cycle_morphology_branches_followup():
    out = _node(_profiles_fixture() + r"""
      const cm={fields:[{id:'osi',short_label:'OSI',units:'1',tier:'model',display:{thresholds:[0.1,0.2,0.3]}},{id:'rrt',short_label:'RRT',units:'1/Pa',tier:'derived'},{id:'tawss'}],
        geometry:{branches:[{id:0,name:'主动脉'},{id:1,name:'右髂总'}]},
        analysis:{cycle:{fields:{osi:{mean:0.135,p99:0.35,thresholds:[0.1,0.2,0.3],area_frac:{above_t0:0.566},per_branch:{'右髂总':{mean:0.07,p99:0.23,max:0.27,frac_above_t0:0.23},'主动脉':{mean:0.16,p99:0.35,max:0.39,frac_above_t0:0.69}}},
            rrt:{mean:5.41,thresholds:[5,10,20],area_frac:{above_t0:0.49}},tawss:{mean:0.66,thresholds:[0.4,4,7],per_branch:{'主动脉':{mean:0.32,p99:1.2,max:2.1,frac_low:0.79}}}}},
          per_branch:{'主动脉':{segment_id:0,wss_p99_pa:4.99,wss_mean_pa:0.78,wss_max_pa:8.48,frac_low:0.53,frac_high:0.02,area_mm2:25984},'右髂总':{segment_id:1,wss_p99_pa:12.1,wss_mean_pa:2.4,wss_max_pa:14.7,frac_low:0.002,frac_high:0.21,area_mm2:1280}},
          morphology:{lumen_volume_ml:347.5,notes:[],branches:[{segment_id:0,name:'主动脉',n_reoriented:0,n_excluded:0,n_stations:208},{segment_id:1,name:'右髂总',n_reoriented:3,n_excluded:1,n_stations:27}],
            aorta:{segment_id:0,name:'主动脉',reference_diameter_mm:18.96,n_reoriented:0,n_excluded:0,n_stations:208,max:{max_diameter_mm:72.2,s_from_root_mm:176},
              sac:{present:true,s_start_mm:122,s_end_mm:209,length_mm:87,volume_ml:225.5},neck:{present:true,s_start_mm:48,s_end_mm:119,length_mm:72,diameter_mean_mm:20.04},
              stations:{s_from_root_mm:[2,3,4,176],max_diameter_mm:[19,20,null,72.2],reliable:[true,false,true,true]}}}}};
      const cyc=OV.cycleRows(cm).map(r=>[r.id,r.tier,r.mean,r.frac,r.threshold]);
      const ms=OV.morphStripModel(cm);
      const bw=OV.branchTableModel(cm,'wss'), bo=OV.branchTableModel(cm,'osi'), bt=OV.branchTableModel(cm,'tawss'), br=OV.branchTableModel(cm,'rrt');
      const fu=OV.followupModel({scans:[{date:'2024-03-15',geometry:{max_diameter_mm:29.1},models:{M1:{tawss_mean_pa:0.76}},jobs:[{job_id:'a',release_id:'M1'}]},
        {date:'2025-03-20',geometry:{max_diameter_mm:25.8},models:{M1:{tawss_mean_pa:0.82}},jobs:[{job_id:'x',release_id:'X5D'},{job_id:'b',release_id:'M1'}]}]},'M1');
      const charts=OV.followupCharts(fu,'a');
      out({cyc, facts:ms.facts.map(f=>[f.key,f.value]), pts:ms.points.length, reliable:ms.points.map(p=>p.reliable), band:[ms.sac,ms.neck], max:ms.max, rel:ms.reliability,
        bw:[bw.field,bw.columns.map(c=>c.label),bw.rows.map(r=>r.name)], bo:[bo.field,bo.columns.map(c=>c.label),bo.rows.map(r=>r.name)], bt:bt.columns.map(c=>c.label), br:br.field,
        charts:charts.map(c=>[c.label,c.points.map(p=>[p.value,p.jobId,p.current])]), rows:fu.rows.map(r=>r.jobId)});
    """)
    assert out["cyc"] == [["osi", "model", 0.135, 0.566, 0.1], ["rrt", "derived", 5.41, 0.49, 5]]   # ECAP absent → no tile
    assert out["facts"] == [["sac", 87], ["neck", 72], ["reference_diameter", 18.96], ["lumen_volume", 347.5]]
    assert out["pts"] == 3 and out["reliable"] == [True, False, True]            # a missing diameter is dropped, an unreliable one is a gap
    assert out["band"] == [[122, 209], [48, 119]] and out["max"] == {"s": 176, "d": 72.2}
    assert out["rel"]["text"] == "截面可靠性：3 站重定向，1 站不可靠（已排除）" and out["rel"]["details"] == ["右髂总：3 站重定向，1 站不可靠（共 27 站）"]
    assert out["bw"] == ["wss", ["p99", "均值", "最大", "< 0.4", "> 4", "cm²"], ["主动脉", "右髂总"]]
    assert out["bo"] == ["osi", ["均值", "p99", "最大", "> 0.1"], ["主动脉", "右髂总"]]   # manifest branch order
    assert out["bt"] == ["均值", "p99", "最大", "< 0.4"] and out["br"] == "wss"          # no per-branch RRT: the WSS table
    assert out["charts"] == [["管腔最大直径", [[29.1, "a", True], [25.8, "b", False]]], ["TAWSS 均值", [[0.76, "a", True], [0.82, "b", False]]]]
    assert out["rows"] == ["a", "b"]                                             # the job of this release opens, not the other model


def test_tools_models_timing_events_metadata_outlets_input():
    out = _node(r"""
      const job={version:4,case_id:'LV',mapping:{'3':'out-le','4':'out-li','5':'out-ri','6':'out-re'},
        mapping_history:[{source:'manual_confirmation'}],
        summary:{timing_s:{ingest:0.12,centerline:3.5,smooth_resample:0,features:0.59,precompute:5.7,total:5.9},geometry_cache:{reused:['mesh','pointgeom']},device:'cuda',gpu:'RTX'},
        timing:{compute_s:7.9,attempt_s:3.9,queue_s:0.5,confirmation_s:4.5,precompute_s:5.7,wall_s:13},
        events:[{at:'2026-09-29T23:04:42+08:00',action:'created'},{action:'progress'},{at:'x',action:'started',stage:'B'},{at:'y',action:'device_fallback',from_device:'cuda'},{at:'z',action:'companion_created'},{at:'w',action:'weird_new'}],
        a:{proposal:{confidence:0.884,mapping:{'3':'out-le','4':'out-li','5':'out-ri','6':'out-re'},endpoints:[{segment_id:0,kind:'inlet',radius_mm:13,center_mm:[0,0,0]},
          {segment_id:3,kind:'outlet',radius_mm:2.8,center_mm:[1,0,0]},{segment_id:4,kind:'outlet',radius_mm:2.8,center_mm:[2,0,0]},{segment_id:5,kind:'outlet',radius_mm:3,center_mm:[3,0,0]},{segment_id:6,kind:'outlet',radius_mm:3.5,center_mm:[4,0,0]}]}}};
      const t=D.timingModel(job);
      const ev=D.eventRows(job).map(e=>e.text);
      const p=D.metadataPayload({case_id:' LV ',patient_id:'P-1',scan_label:'',scan_date:'2025-06-01',tags:'AAA，随访, AAA、演示\n',notes:'  两行\n备注 '},4);
      const errs=[D.metadataError(Object.assign({},p,{case_id:''}),job), D.metadataError(Object.assign({},p,{tags:Array.from({length:13},(_,i)=>'t'+i)}),job),
        D.metadataError(Object.assign({},p,{tags:['x'.repeat(41)]}),job), D.metadataError(Object.assign({},p,{scan_date:'2025/6/1'}),job), D.metadataError(p,job)];
      const ids=[D.identifierIssue('张三',{label:'患者编号'}), D.identifierIssue('a\tb',{label:'病例名称'}), D.identifierIssue('x'.repeat(81),{label:'患者编号',max:80}), D.identifierIssue('P-001')];
      const om=D.outletModel(job), sw=D.swapMapping(om.mapping,'lr');
      const im=D.inputModel({selected_units:'auto',resolved_units:'mm',unit_confidence:0.999,scale_factor:1,bbox_size_mm:[96.7,113.5,260.8],area_mm2:34110,faces:23184,
        quality:{grade:'pass_with_limits',grade_label:'已检查项通过（存在未评估项）',checks:[{label:'单位',status:'pass'},{label:'自交',status:'not_checked'},{label:'方向',status:'unknown'},{label:'开口',status:'review'}]},
        flags:['f1','f1']},[{radius_mm:2.8,center_mm:[1,2,3]},{equivalent_radius_mm:13.4}]);
      out({stages:t.stages.map(s=>[s.key,s.label,s.cached]), t:[t.compute,t.attempt,t.split,t.queue,t.confirmation,t.precompute,t.wall,t.pipeline], ev, p, errs, ids:ids.map(x=>x&&x.level),
        om:[om.available,om.source,om.confidence,om.endpoints.length], sw, valid:[D.validMapping(sw,['3','4','5','6']),D.validMapping(Object.assign({},sw,{'3':'out-li'}),['3','4','5','6'])],
        same:[D.sameMapping(om.mapping,om.proposal),D.sameMapping(om.mapping,sw)], tags:D.tagsText(['a',' ','b']),
        im:[im.units,im.auto,im.areaCm2,im.openings.map(o=>[o.i,o.radius,Boolean(o.center)]),im.checks.map(c=>c.status),im.counts,im.flags]});
    """)
    assert out["stages"] == [["ingest", "输入检查", False], ["centerline", "中心线提取", False], ["smooth_resample", "平滑与重采样", True], ["features", "几何特征", True]]
    assert out["t"] == [7.9, 3.9, True, 0.5, 4.5, 5.7, 13, 5.9]                     # the background precompute is not a stage
    assert out["ev"] == ["已创建", "开始计算 · 阶段 B", "显存不足，已改用 CPU 重算（cuda → cpu）", "已建立同时预测的结果", "weird_new"]
    assert out["p"] == {"version": 4, "case_id": "LV", "patient_id": "P-1", "scan_label": "", "scan_date": "2025-06-01", "tags": ["AAA", "随访", "演示"], "notes": "两行\n备注"}
    assert out["errs"][0] == "病例名称不能为空。" and "12" in out["errs"][1] and "40" in out["errs"][2] and "YYYY-MM-DD" in out["errs"][3] and out["errs"][4] is None
    assert out["ids"] == ["warn", "error", "error", None]
    assert out["om"] == [True, "人工确认", 0.884, 5]
    assert out["sw"] == {"3": "out-re", "4": "out-ri", "5": "out-li", "6": "out-le"} and out["valid"] == [True, False] and out["same"] == [True, False]
    assert out["tags"] == "a, b"
    units, auto, area, ops, statuses, counts, flags = out["im"]
    assert units == "mm" and auto is True and abs(area - 341.1) < 1e-9
    assert ops == [[1, 2.8, True], [2, 13.4, False]]
    assert statuses == ["review", "unknown", "not_checked", "pass"] and counts == {"review": 1, "unknown": 1, "not_checked": 1, "pass": 1} and flags == ["f1"]


def test_lens_new_references():
    out = _node(r"""
      const m={fields:[{id:'wss',short_label:'WSS',units:'Pa',tier:'model'},{id:'osi',short_label:'OSI',units:'1',tier:'model',temporal:'cycle_summary'}],result:{display_name:'R'},job:{display_name:'C'},analysis:{}};
      const txt=ref=>L.chain(ref,m,{}).map(s=>s.rows.map(r=>r.join(' ')).join('|')).join(' / ');
      out({branch:txt({kind:'branch',name:'右髂总',field:'wss',label:'WSS 低于 0.4 的占比',value:0.002,pct:true,n:2867,area_mm2:1280}),
        point:txt({kind:'point',field:'wss',value:3.2,segmentId:0,s_from_root_mm:176,theta_rad:-0.733}),
        stat:txt({kind:'stat',field:'osi',stat:'mean',value:0.135,units:'1',label:'OSI 均值',also:[['> 0.1 占比','57%']]}),
        morph:txt({kind:'morph',label:'全腔体积',value:347.5,units:'mL',where:'全部管腔',definition:'散度定理'}),
        cursor:txt({kind:'cursor',field:'wss',readout:{segmentId:0,s_mm:73,s_from_root_mm:73.1,bin:[72,74],stats:{p99:5,mean:2},definition:'2 mm 分箱'}})});
    """)
    assert "右髂总" in out["branch"] and "2867 个" in out["branch"] and "12.8 cm²" in out["branch"] and "0.2%" in out["branch"] and "点占比" in out["branch"]
    assert "周向角 −42°（展开图纵轴）" in out["point"]
    assert "OSI 均值 0.135" in out["stat"] and "> 0.1 占比 57%" in out["stat"]
    assert "位置 全部管腔" in out["morph"] and "348 mL" in out["morph"]
    assert "（距入口 73.1 mm）" in out["cursor"]


# ---------------------------------------------------------------- shell integration (workspace harness + lane files)
_SETUP = r"""
  const prof = {bin_mm: 2, definition: '按弧长分箱', branches: [
    {segment_id: 0, name: '主动脉', s_local_mm: [1, 3, 5], s_from_root_mm: [1, 3, 5], radius_mm: [10, 11, 15], wss: {n: [5, 6, 7], mean_pa: [1, 2, 3], p99_pa: [4, 5, 6], min_pa: [0.5, 0.6, 0.7]},
      tawss: {mean_pa: [0.3, 0.5, 0.2], min_pa: [0.1, 0.2, 0.1]}},
    {segment_id: 2, name: '左髂总', s_local_mm: [1, 3], s_from_root_mm: [213, 215], radius_mm: [5, 5], wss: {n: [3, 3], mean_pa: [2, 2], p99_pa: [8, 9], min_pa: [1, 1]}}]};
  const withProfiles = id => { const b = canned['/api/v2/jobs/' + id + '/manifest'].body; b.analysis.profiles = prof;
    b.analysis.per_branch = {'主动脉': {segment_id: 0, wss_p99_pa: 4.99, wss_mean_pa: 0.78, wss_max_pa: 8.48, frac_low: 0.53, frac_high: 0.02, area_mm2: 25984}}; };
  const proposal = {confidence: 0.884, mapping: {'3': 'out-le', '4': 'out-li', '5': 'out-ri', '6': 'out-re'}, endpoints: [{segment_id: 0, kind: 'inlet', radius_mm: 13.2, center_mm: [0, 0, 0]},
    {segment_id: 3, kind: 'outlet', radius_mm: 2.8, center_mm: [1, 0, 0]}, {segment_id: 4, kind: 'outlet', radius_mm: 2.8, center_mm: [2, 0, 0]},
    {segment_id: 5, kind: 'outlet', radius_mm: 3.0, center_mm: [3, 0, 0]}, {segment_id: 6, kind: 'outlet', radius_mm: 3.5, center_mm: [4, 0, 0]}]};
  serve('A', {mapping: proposal.mapping, a: {proposal, input_check: {selected_units: 'mm', faces: 100, quality: {grade: 'pass', checks: [{label: '单位', status: 'pass'}]}}},
    tags: ['AAA'], notes: '旧备注', summary: {timing_s: {ingest: 0.1, centerline: 3.5, total: 3.6}}, events: [{at: '2026-09-29T08:00:00+08:00', action: 'created'}]});
  withProfiles('A');
  const sent = {};
  canned['POST /api/jobs/A/metadata'] = o => { sent.metadata = JSON.parse(o.body); return {body: {job: jobRecord('A', {version: 4})}}; };
  canned['POST /api/jobs/A/confirm'] = o => { sent.confirm = JSON.parse(o.body); return {body: {job: jobRecord('A', {status: 'queued_B', version: 4})}}; };
  canned['POST /api/jobs/A/delete'] = o => { sent.del = JSON.parse(o.body); return {body: {deleted: true, trashed: true, id: 'A'}}; };
  canned['POST /api/trash/A/restore'] = o => { sent.restore = true; return {body: {restored: true, id: 'A'}}; };
  const tab = id => walk(app(), e => e.dataset && e.dataset.tab === id)[0];
  const btn = (root, text) => walk(root, e => e.tagName === 'BUTTON' && textOf(e).trim() === text)[0];
  const dlg = () => byId('ws-dialog');
  const byCls = (root, cls) => walk(root, e => String((e.attrs && e.attrs.class) || e.className || '').split(/\s+/).includes(cls));   // SVG nodes carry class as an attribute
"""


def test_along_tab_profiles_and_bin_pin():
    out = _shell(_SETUP + r"""
      await boot();
      await hashTo('#/job/A', 150);
      const tabs = byClass(app(), 'tab').map(textOf);
      fire(tab('along'), 'click'); await wait(40);
      const body = byClass(app(), 'insp-body')[0];
      const svgs = byCls(body, 'along-svg').length, cap = byClass(body, 'along-cap-name').map(textOf), hint = textOf(byClass(body, 'along-read')[0]);
      const sel = walk(body, e => e.tagName === 'SELECT')[0];
      const opts = sel.children.map(textOf);
      // a click in the middle of the curve (stub rect 30 px wide): the 3 mm bin, pinned
      fire(byCls(body, 'along-main')[0], 'click', {clientX: 25, clientY: 20}); await wait(20);
      const pinned = textOf(byClass(byClass(app(), 'insp-body')[0], 'along-read')[0]);
      fire(btn(byClass(app(), 'insp-body')[0], '来源'), 'click'); await wait(30);
      const lens = textOf(byClass(app(), 'lens')[0]), now = shellState().tab;
      // switch branch
      fire(tab('along'), 'click'); await wait(30);
      const s2 = walk(byClass(app(), 'insp-body')[0], e => e.tagName === 'SELECT')[0]; s2.value = '2'; fire(s2, 'change'); await wait(30);
      const x2 = byClass(byClass(app(), 'insp-body')[0], 'along-xt').map(textOf);
      done({tabs, svgs, cap, hint, opts, pinned, lens, now, x2});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["tabs"] == ["概览", "沿程", "读数", "书签", "工具"]
    assert out["svgs"] == 2 and out["cap"] == ["TAWSS", "半径"]                  # the coloured field (TAWSS) and the radius
    assert "点曲线" in out["hint"] and "展开图" not in out["hint"]                 # no point arrays here: no map, no word about it
    assert out["opts"] == ["主动脉", "左髂总"]
    assert "主动脉 · 距入口 3.00 mm · 均值 0.500 · 最低 0.200 Pa · 半径 11.0 mm" in out["pinned"], out["pinned"]
    assert any(c.startswith("1:highlight:") for c in out["vcalls"])
    assert "分箱" in out["lens"] and out["now"] == "reading"
    assert out["x2"] and all(213 <= float(t) <= 215 for t in out["x2"]), out["x2"]   # the other branch's own arc range


def test_tools_sections_metadata_outlets_delete_undo():
    out = _shell(_SETUP + r"""
      localStorage.setItem('wssv2:prefs', JSON.stringify({tier: 'full'}));
      await boot();
      await hashTo('#/job/A', 150);
      fire(tab('tools'), 'click'); await wait(30);
      const body = () => byClass(app(), 'insp-body')[0];
      const titles = byClass(body(), 'sec-title').map(textOf);
      // metadata with tags and notes
      fire(btn(body(), '编辑…'), 'click'); await wait(20);
      const inputs = walk(dlg(), e => (e.tagName === 'INPUT' || e.tagName === 'TEXTAREA'));
      const f = label => inputs.find(e => e.attrs['aria-label'] === label);
      const before = [f('标签').value, f('备注').value];
      f('标签').value = 'AAA，随访, AAA'; f('备注').value = '新备注'; f('患者编号').value = 'P-7';
      fire(btn(dlg(), '保存'), 'click'); await wait(80);
      // outlet override: swap left / right, acknowledge, send
      fire(tab('tools'), 'click'); await wait(30);
      fire(btn(body(), '修改并重算…'), 'click'); await wait(20);
      fire(btn(body(), '左右互换'), 'click'); await wait(20);
      const ack = walk(body(), e => e.tagName === 'INPUT' && e.type === 'checkbox')[0]; ack.checked = true; fire(ack, 'change');
      fire(btn(body(), '采用并重算'), 'click'); await wait(80);
      // delete → trash → undo
      await hashTo('#/job/A', 150);
      fire(tab('tools'), 'click'); await wait(30);
      fire(btn(body(), '删除任务…'), 'click'); await wait(20);
      const confirmText = textOf(dlg());
      fire(btn(dlg(), '移入回收站'), 'click'); await wait(80);
      const afterDelete = global.location.hash, toast = textOf(byId('ws-toast'));
      fire(btn(byId('ws-toast'), '撤销'), 'click'); await wait(80);
      done({titles, before, sent, confirmText, afterDelete, toast, afterUndo: global.location.hash});
    """)
    assert out["errors"] == [], out["errors"]
    t = out["titles"]
    assert t.count("病例信息") == 1 and all(x in t for x in ("出口命名", "计算过程", "输入检查", "删除"))   # the shell's own section gives way
    assert out["before"] == ["AAA", "旧备注"]
    assert out["sent"]["metadata"] == {"version": 3, "case_id": "CASE_A", "patient_id": "P-7", "scan_label": "", "scan_date": "", "tags": ["AAA", "随访"], "notes": "新备注"}
    assert out["sent"]["confirm"] == {"mapping": {"3": "out-re", "4": "out-ri", "5": "out-li", "6": "out-le"}, "acknowledged": True, "override": True, "stage": "B", "version": 3}
    assert "回收站" in out["confirmText"] and out["sent"]["del"] == {"version": 3}
    assert out["afterDelete"] == "#/" and "已移入回收站" in out["toast"]
    assert out["sent"]["restore"] is True and out["afterUndo"] == "#/job/A"
    assert "POST /api/jobs/A/delete" in out["calls"] and "POST /api/trash/A/restore" in out["calls"]


def test_keys_new_search_onepage_and_offline():
    out = _shell(_SETUP + r"""
      await boot();
      await hashTo('#/job/A', 150);
      await key('o');
      const opened1 = opened.slice();
      await key('/');
      const focused = String(global.document.activeElement.className || '');
      global.document.activeElement = body;
      await key('n'); await wait(30);
      const dialogOpen = Boolean(byId('ws-dialog') && byId('ws-dialog').open), title = textOf(walk(byId('ws-dialog') || body, e => e.tagName === 'H2')[0]);
      fire(walk(byId('ws-dialog'), e => e.attrs && e.attrs['aria-label'] === '关闭')[0], 'click'); await wait(10);
      await key('?'); await wait(10);
      const keysText = textOf(byId('ws-dialog'));
      done({opened1, focused, dialogOpen, title, keysText});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["opened1"] == ["/api/jobs/A/onepage"]
    assert "rail-search" in out["focused"]
    assert out["dialogOpen"] is True and "上传" in out["title"]
    assert "新建（上传 STL）" in out["keysText"] and "搜索病例" in out["keysText"] and "打开一页纸" in out["keysText"]


def test_overview_branch_table_follows_the_field_and_no_detail_offline_tools():
    out = _shell(_SETUP + r"""
      await boot();
      await hashTo('#/job/A', 150);
      const insp = () => byClass(app(), 'ws-inspector')[0];
      fire(btn(insp(), '表格'), 'click'); await wait(20);
      const t1 = textOf(byClass(insp(), 'branch-block')[0]);
      walk(app(), e => e.dataset && e.dataset.field === 'wss' && e.className.includes('seg-btn'))[0].click(); await wait(40);
      const t2 = textOf(byClass(insp(), 'branch-block')[0]);
      done({t1, t2, secs: byClass(insp(), 'sec-title').map(textOf)});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["t1"].startswith("逐分支 · WSS")                                  # no per-branch TAWSS in the fixture: the WSS table
    assert "4.99" in out["t2"] and "53%" in out["t2"] and "260" in out["t2"]
    assert out["secs"][:3] == ["分区", "发现", "结论"]


def test_offline_report_keeps_the_along_tab_and_drops_the_tools():
    out = _shell(r"""
      const m = manifestFor('Z');
      m.analysis.profiles = {bin_mm: 2, branches: [{segment_id: 0, name: '主动脉', s_local_mm: [1, 3], s_from_root_mm: [1, 3], radius_mm: [10, 11], wss: {n: [1, 1], mean_pa: [1, 2], p99_pa: [3, 4], min_pa: [0, 1]}}]};
      const s1 = new Element('script'); s1.id = 'wssv2-manifest'; s1.textContent = JSON.stringify(m); body.appendChild(s1);
      const s2 = new Element('script'); s2.id = 'wssv2-offline'; s2.textContent = JSON.stringify({bookmarks: [], hide_name: false, exported_at: '2026-09-30T09:00:00+08:00', view: {}}); body.appendChild(s2);
      const appEl = new Element('div'); appEl.id = 'ws-app'; body.appendChild(appEl);
      await boot(); await wait(100);
      const tabs = byClass(app(), 'tab').map(textOf);
      await key('n'); await key('o');
      done({tabs, dialog: Boolean(byId('ws-dialog') && byId('ws-dialog').open)});
    """, offline=True)
    assert out["errors"] == [], out["errors"]
    assert "沿程" in out["tabs"] and "工具" not in out["tabs"]
    assert out["dialog"] is False and out["opened"] == []
