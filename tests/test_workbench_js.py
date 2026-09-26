"""Node tests for the workbench's DOM-free helpers (C1–C6, C13, C18 front-end logic).

``static/workbench_core.js`` and ``static/batch_export.js`` are plain UMD scripts; Node ``require``s them
directly.  The batch-export message protocol (contract §12.6) is exercised with a fake iframe/window so
the apply-state → export → exported handshake runs without a browser.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from wss_deploy.paths import STATIC_DIR

CORE = STATIC_DIR / "workbench_core.js"
BATCH = STATIC_DIR / "batch_export.js"


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the workbench tests")
    program = ("const WB=require(" + json.dumps(str(CORE)) + ");const BX=require(" + json.dumps(str(BATCH)) + ");\n" + script)
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_static_scripts_parse():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    for name in ("app.js", "workbench_core.js", "batch_export.js", "compare.js"):
        subprocess.run(["node", "--check", str(STATIC_DIR / name)], check=True, capture_output=True)


def test_unread_set_roundtrip_and_title():
    out = _node("""
      const s=WB.unreadParse('["a:1","b:2","bad"]');
      const added=WB.unreadAdd(s,'c',3), again=WB.unreadAdd(s,'c',3), bumped=WB.unreadAdd(s,'a',5);
      const count=WB.unreadCount(s), hasA=WB.unreadHas(s,'a'), removed=WB.unreadRemove(s,'a'), afterRemove=WB.unreadHas(s,'a');
      const pruned=WB.unreadPrune(s,['c']);
      console.log(JSON.stringify({added,again,bumped,count,hasA,removed,afterRemove,pruned,left:JSON.parse(WB.unreadSerialize(s)),title:[WB.titleWithUnread('WSS',0),WB.titleWithUnread('WSS',3)]}));
    """)
    assert out["added"] is True and out["again"] is False and out["bumped"] is True
    assert out["count"] == 3 and out["hasA"] is True and out["removed"] is True and out["afterRemove"] is False
    assert out["pruned"] == 1 and out["left"] == ["c:3"]
    assert out["title"] == ["WSS", "(3) WSS"]


def test_owner_events_notify_once_per_transition():
    out = _node("""
      const last={};
      const seq=[{job_id:'j',status:'running',action:'progress'},{job_id:'j',status:'running',action:'started'},{job_id:'j',status:'done',action:'finished',final:true,case_id:'CASE',family:'wall'},
        {job_id:'j',status:'done',action:'finished',final:true},{job_id:'k',status:'awaiting_confirmation',action:'finished'},{job_id:'k',status:'failed',action:'finished'}];
      const decisions=seq.map(e=>WB.shouldNotify(e,last));
      console.log(JSON.stringify({decisions,text:WB.notificationText(seq[2])}));
    """)
    assert out["decisions"] == [False, False, True, False, True, True]
    assert out["text"]["title"] == "CASE · 已完成" and "壁面 WSS" in out["text"]["body"]


def test_preferences_merge_keeps_other_sections():
    out = _node("""
      const base={schema_version:'wss-deploy.preferences/v1',upload:{units:'mm',release_id:'X'},report_defaults:{wall:{colormap:'turbo'}},presets:{wall:[{name:'p'}]}};
      const merged=WB.mergePreferences(base,{upload:{units:'cm'},notifications:{enabled:false}});
      const fresh=WB.mergePreferences({},null);
      console.log(JSON.stringify({merged,fresh,up:WB.uploadPreferencesFrom({units:'mm',release_id:'R',device:'cpu',seed_count:'3',threads:'8',remember_patient:true,patient_id:'P-1',scan_label:'基线',scan_date:'2026-09-21',tags:'AAA'})}));
    """)
    merged = out["merged"]
    assert merged["upload"] == {"units": "cm", "release_id": "X"}
    assert merged["report_defaults"] == {"wall": {"colormap": "turbo"}} and merged["presets"] == {"wall": [{"name": "p"}]}
    assert merged["notifications"] == {"enabled": False}
    assert out["fresh"]["schema_version"] == "wss-deploy.preferences/v1"
    assert out["up"]["last_patient"] == {"patient_id": "P-1", "scan_label": "基线", "scan_date": "2026-09-21", "tags": "AAA"}
    assert out["up"]["remember_patient"] is True and out["up"]["seed_count"] == "3"


def test_review_groups_and_case_card_model():
    out = _node("""
      const jobs=[{id:'a',status:'done',review:{status:'unreviewed'}},{id:'b',status:'done',review:{status:'reviewed'}},{id:'c',status:'running'},{id:'d',status:'done',review:{status:'reopened'}}];
      const groups=WB.splitByReview(jobs);
      const card={input_sha256:'x',case_ids:['LIU'],patient_id:'P-1',scan_label:'基线',tags:['AAA'],pending_review:1,latest_at:'t',reusable_job_id:'w1',missing_releases:['NEW'],
        latest:{X5D:'w1',PF6:'v1'},runs:[{job_id:'w1',release_id:'X5D',family:'wall',status:'done',review:'unreviewed',created_at:'t1'},{job_id:'v1',release_id:'PF6',family:'volume',status:'done',review:'reviewed',created_at:'t2'},{job_id:'f1',release_id:'X5D',family:'wall',status:'failed',review:'unreviewed'}]};
      const model=WB.caseCardModel(card,[{id:'X5D',family:'wall'},{id:'PF6',family:'volume'},{id:'NEW',family:'wall'}]);
      console.log(JSON.stringify({pending:groups.pending.map(j=>j.id),reviewed:groups.reviewed.map(j=>j.id),other:groups.other.map(j=>j.id),
        title:model.title,subtitle:model.subtitle,compare:model.compare,missing:model.missing,reusable:model.reusable,done:model.doneJobIds,pending_review:model.pendingReview,fam:WB.familyOfRelease('NEW',[{id:'NEW',family:'wall'}])}));
    """)
    assert out["pending"] == ["a", "d"] and out["reviewed"] == ["b"] and out["other"] == ["c"]
    assert out["title"] == "LIU" and out["subtitle"] == "患者 P-1 · 基线"
    assert out["compare"] == {"left": "w1", "right": "v1"}
    assert out["missing"] == ["NEW"] and out["reusable"] == "w1" and out["done"] == ["w1", "v1"] and out["pending_review"] == 1
    assert out["fam"] == "wall"


def test_export_url_and_batch_plan_by_family():
    out = _node("""
      const url=WB.exportUrl(['a','b','../x'],'xlsx');
      const plan=WB.planBatchExport([{id:'w',status:'done',family:'wall'},{id:'v',status:'done',family:'volume'},{id:'r',status:'running',family:'wall'},{id:'u',status:'done'}],WB.presetState('瘤囊低 WSS 区'));
      const any=WB.planBatchExport([{id:'w',status:'done',family:'wall'},{id:'v',status:'done',family:'volume'}],{schema_version:'wss-deploy.view/v1',camera:{}});
      let bad=null; try{WB.parseViewStateJSON('{"schema_version":"nope/v1"}');}catch(e){bad=e.message;}
      console.log(JSON.stringify({url,selected:plan.selected.map(j=>j.id),skipped:plan.skipped.map(s=>s.id),family:plan.family,anySelected:any.selected.length,bad,
        fname:WB.filenameFromDisposition('attachment; filename="wss_summary_20260921.csv"','x.csv'),fnameStar:WB.filenameFromDisposition("attachment; filename*=UTF-8''%E6%B1%87%E6%80%BB.xlsx",'x'),none:WB.filenameFromDisposition(null,'fallback.zip'),
        presets:WB.BUILTIN_PRESETS,volumePreset:WB.presetState('流线全貌')}));
    """)
    assert out["url"] == "/api/jobs/export?ids=a,b&format=xlsx"
    assert out["selected"] == ["w", "u"] and out["skipped"] == ["v", "r"] and out["family"] == "wall"
    assert out["anySelected"] == 2 and "视图状态" in out["bad"]
    assert out["fname"] == "wss_summary_20260921.csv" and out["fnameStar"] == "汇总.xlsx" and out["none"] == "fallback.zip"
    assert out["presets"] == {"wall": ["瘤囊低 WSS 区", "髂分叉热点", "主动脉沿程"], "volume": ["沿程压降", "流线全貌", "瘤囊截面系列"]}
    assert out["volumePreset"] == {"schema_version": "wss-deploy.view/v1", "family": "volume", "preset_name": "流线全貌"}


def test_metadata_payload_and_tag_parsing():
    out = _node("""
      const payload=WB.metadataPayload({case_id:'  LIU  ',patient_id:'P-1',scan_label:'基线',scan_date:'2026-09-21',tags:'AAA, 随访、AAA，新 ',notes:' 备注 '},7);
      const noVersion=WB.metadataPayload({case_id:'x'},null);
      console.log(JSON.stringify({payload,noVersion,empty:WB.parseTags('  , ,'),text:[WB.tagsText(['a','b']),WB.tagsText('a, b'),WB.tagsText(null)]}));
    """)
    assert out["payload"] == {"version": 7, "case_id": "LIU", "patient_id": "P-1", "scan_label": "基线",
                              "scan_date": "2026-09-21", "tags": ["AAA", "随访", "新"], "notes": "备注"}
    assert "version" not in out["noVersion"] and out["empty"] == []
    assert out["text"] == ["a, b", "a, b", ""]


def test_rerun_skip_reasons_and_attempt_timing():
    out = _node("""
      const reasons=[WB.rerunSkipReason(null),WB.rerunSkipReason({status:'running'}),WB.rerunSkipReason({status:'done'}),
        WB.rerunSkipReason({status:'done',mapping:{}}),WB.rerunSkipReason({status:'done',a:{proposal:{mapping:{'3':'out-le'}}}}),
        WB.rerunSkipReason({status:'done',mapping:{'3':'out-le'}}),
        WB.rerunSkipReason({status:'done',mapping:{'3':'out-le'},a:{proposal:{}}}),
        WB.rerunSkipReason({status:'done',mapping:{'3':'out-le'},stage_a:{}})];
      const timings=[WB.computeTiming({compute_s:120,attempt_s:40}),WB.computeTiming({compute_s:40,attempt_s:40}),
        WB.computeTiming({compute_s:40}),WB.computeTiming(null,12.5),WB.computeTiming(undefined),WB.computeTiming({compute_s:37,attempt_s:0})];
      console.log(JSON.stringify({reasons,timings}));
    """)
    assert out["reasons"][:5] == ["任务不存在", "任务未完成", "尚未确认出口命名", "尚未确认出口命名", "尚未确认出口命名"]
    assert out["reasons"][5] == "缺少可复用的中心线结果"   # same precondition the service enforces
    assert out["reasons"][6] is None and out["reasons"][7] is None
    assert out["timings"][0] == {"attempt": 40, "total": 120, "split": True}
    assert out["timings"][1]["split"] is False and out["timings"][2]["split"] is False
    assert out["timings"][3] == {"attempt": None, "total": 12.5, "split": False}
    assert out["timings"][4] == {"attempt": None, "total": None, "split": False}
    assert out["timings"][5]["split"] is False      # a zero attempt (record touched after finishing) shows the total only


def test_compare_display_subset_and_echo_guard():
    out = _node("""
      const full={schema_version:'wss-deploy.view/v1',family:'wall',camera:{position:[1,2,3]},field:'wss',mode:'field',colormap:'turbo',
        bands:6,log:true,range:{mode:'case',min:0,max:10},units:'Pa',thresholds_pa:[0.4,4,7],opacity:0.8,overlay:{trust:true},slice:{},
        highlight:{},measurements:[],run_identity:'r'};
      const same=WB.compareDisplaySubset(full,true), cross=WB.compareDisplaySubset(full,false);
      const sparse=WB.compareDisplaySubset({colormap:'viridis'},true), junk=WB.compareDisplaySubset(null,true);
      let now=1000; const guard=new WB.EchoGuard(400,()=>now);
      guard.mark('right'); const immediately=guard.blocked('right'), otherSide=guard.blocked('left');
      now+=399; const inside=guard.blocked('right'); now+=2; const outside=guard.blocked('right');
      console.log(JSON.stringify({same,cross,sparse,junk,immediately,otherSide,inside,outside,keys:WB.DISPLAY_KEYS}));
    """)
    assert "camera" not in out["same"] and "highlight" not in out["same"] and "family" not in out["same"]
    assert out["same"]["field"] == "wss" and out["same"]["thresholds_pa"] == [0.4, 4, 7] and out["same"]["overlay"] == {"trust": True}
    assert set(out["cross"]) == {"colormap", "bands", "log", "opacity"}
    assert out["sparse"] == {"colormap": "viridis"} and out["junk"] == {}
    assert out["immediately"] is True and out["otherSide"] is False and out["inside"] is True and out["outside"] is False


def test_cohort_binning_filtering_and_sorting():
    out = _node("""
      const edges=WB.binEdges([1,2,3,4],4), counts=WB.histogram([1,1.5,2.5,4,9,-1],edges);
      const flat=WB.binEdges([5,5,5],4), none=WB.binEdges([],4), empty=WB.histogram([1],[]);
      const rows=[
        {job_id:'a',case_id:'A',family:'wall',release_id:'X',review_status:'reviewed',wss_p99_pa:12.5,area_frac_high:0.08,area_frac_very_high:0.01,population_percentile:88.2,max_diameter_mm:63.4},
        {job_id:'b',case_id:'B',family:'wall',release_id:'X',review_status:'unreviewed',wss_p99_pa:4.2,area_frac_high:0.005,area_frac_very_high:0,population_percentile:12,max_diameter_mm:28.1},
        {job_id:'c',case_id:'C',family:'volume',release_id:'P',review_status:'unreviewed',speed_p99_m_s:1.4,max_diameter_mm:51.0},
        {job_id:'d',case_id:'D',family:'wall',release_id:'Y',review_status:null,wss_p99_pa:null}];
      const all=WB.cohortFilter(rows,{}), wall=WB.cohortFilter(rows,{family:'wall'}), rel=WB.cohortFilter(rows,{release_id:'X'});
      const rev=WB.cohortFilter(rows,{review_status:'unreviewed'}), hot=WB.cohortFilter(rows,{p99_min:'5'}), high=WB.cohortFilter(rows,{high_min:'1'});
      const fast=WB.cohortFilter(rows,{speed_min:1.0});
      const wide=WB.cohortFilter(rows,{diameter_min:'50'}), wideWall=WB.cohortFilter(rows,{family:'wall',diameter_min:50});
      const anyDiameter=WB.cohortFilter(rows,{diameter_min:''});
      const diamDesc=WB.cohortSort(rows,'max_diameter_mm','desc').map(r=>r.job_id);
      const desc=WB.cohortSort(rows,'wss_p99_pa','desc').map(r=>r.job_id), asc=WB.cohortSort(rows,'wss_p99_pa','asc').map(r=>r.job_id);
      const byCase=WB.cohortSort(rows,'case_id','desc').map(r=>r.job_id);
      const pop=WB.populationValues({X:{metric:'p99_pa',values_pa:[1,2,'x',3],case_count:136},P:{values_pa:[]}},rows,null);
      const pinned=WB.populationValues({X:{values_pa:[1]},Y:{values_pa:[9,9]}},rows,'Y');
      console.log(JSON.stringify({edges,counts,flat,none,empty,all:all.length,wall:wall.map(r=>r.job_id),rel:rel.map(r=>r.job_id),
        rev:rev.map(r=>r.job_id),hot:hot.map(r=>r.job_id),high:high.map(r=>r.job_id),fast:fast.map(r=>r.job_id),desc,asc,byCase,pop,pinned,
        wide:wide.map(r=>r.job_id),wideWall:wideWall.map(r=>r.job_id),anyDiameter:anyDiameter.length,diamDesc,
        releases:WB.cohortReleases(rows)}));
    """)
    assert out["edges"] == [1, 1.75, 2.5, 3.25, 4]
    assert out["counts"] == [2, 0, 1, 1]            # right-open bins; 9 and -1 fall outside the edges
    assert len(out["flat"]) == 5 and out["flat"][0] < 5 < out["flat"][4]
    assert out["none"] == [] and out["empty"] == []
    assert out["all"] == 4 and out["wall"] == ["a", "b", "d"] and out["rel"] == ["a", "b"]
    assert out["rev"] == ["b", "c", "d"]            # a blank review_status counts as unreviewed
    assert out["hot"] == ["a"] and out["high"] == ["a"] and out["fast"] == ["c"]
    assert out["desc"] == ["a", "b", "d"] or out["desc"][:2] == ["a", "b"]
    assert out["desc"][-1] == "d" and out["asc"][-1] == "d"   # blanks always sink
    assert out["byCase"] == ["d", "c", "b", "a"]
    assert out["pop"]["release_id"] == "X" and out["pop"]["values"] == [1, 2, 3] and out["pop"]["case_count"] == 136
    assert out["pinned"]["release_id"] == "Y" and out["pinned"]["values"] == [9, 9]
    assert out["releases"] == ["P", "X", "Y"]
    # §17.4: the maximum-diameter filter spans both families and a row without the key drops out.
    assert out["wide"] == ["a", "c"] and out["wideWall"] == ["a"] and out["anyDiameter"] == 4
    assert out["diamDesc"] == ["a", "c", "b", "d"]


def test_narrative_model_and_latest_max_diameter():
    out = _node("""
      const auto=WB.narrativeModel({zh:['一句。','两句。'],en:['one'],edited:null,edited_by:null});
      const edited=WB.narrativeModel({zh:['一句。'],edited:' 审阅人写的结论。 ',edited_by:'医生A',edited_at:'2026-09-22T10:00:00'});
      const english=WB.narrativeModel({zh:['一句。'],en:['one','two']},'en');
      const blankEdit=WB.narrativeModel({zh:['一句。'],edited:'   '});
      const nothing=[WB.narrativeModel(null),WB.narrativeModel({}),WB.narrativeModel({zh:[]}),WB.narrativeModel({zh:['',' ']}),WB.narrativeModel('x')];
      const runs=[{job_id:'r1',status:'running'},{job_id:'d1',status:'done',max_diameter_mm:63.4},{job_id:'d2',status:'done',max_diameter_mm:20}];
      const legacy=WB.latestMaxDiameter([{job_id:'d1',status:'done'},{job_id:'d2',status:'done',max_diameter_mm:'bad'}]);
      const model=WB.caseCardModel({case_ids:['LIU'],runs},[]);
      console.log(JSON.stringify({auto,edited,english,blankEdit,nothing,latest:WB.latestMaxDiameter(runs),legacy,none:WB.latestMaxDiameter(null),card:model.maxDiameterMm}));
    """)
    assert out["auto"]["edited"] is None and out["auto"]["sentences"] == ["一句。", "两句。"] and out["auto"]["text"] == "一句。\n两句。"
    assert out["edited"]["edited"] == "审阅人写的结论。" and out["edited"]["editedBy"] == "医生A" and out["edited"]["text"] == "审阅人写的结论。"
    assert out["english"]["sentences"] == ["one", "two"]
    assert out["blankEdit"]["edited"] is None and out["blankEdit"]["text"] == "一句。"   # whitespace-only edit is no edit
    assert out["nothing"] == [None, None, None, None, None]                           # older jobs draw no card
    assert out["latest"] == 63.4 and out["legacy"] is None and out["none"] is None and out["card"] == 63.4


def test_zip_store_is_readable_by_python_zipfile(tmp_path):
    target = tmp_path / "out.zip"
    out = _node("""
      const fs=require('fs');
      const zip=BX.zipStore([{name:'LIU_1/图.png',bytes:new Uint8Array([137,80,78,71,13,10,26,10,0,1,2,3])},{name:'manifest.json',bytes:'{"ok":true}'},{name:'manifest.json',bytes:'dup'},{name:'empty.txt',bytes:new Uint8Array(0)}]);
      fs.writeFileSync(""" + json.dumps(str(target)) + """,zip);
      console.log(JSON.stringify({size:zip.length,crc:BX.crc32(new TextEncoder().encode('123456789'))}));
    """)
    assert out["crc"] == 0xCBF43926  # standard CRC-32 check value
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert names == ["LIU_1/图.png", "manifest.json", "manifest_2.json", "empty.txt"]
        assert archive.read("LIU_1/图.png") == bytes([137, 80, 78, 71, 13, 10, 26, 10, 0, 1, 2, 3])
        assert json.loads(archive.read("manifest.json")) == {"ok": True}
        assert archive.read("manifest_2.json") == b"dup" and archive.read("empty.txt") == b""
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED and info.flag_bits & 0x800
    assert out["size"] == target.stat().st_size


def test_batch_run_protocol_roundtrip_with_fake_frames(tmp_path):
    """apply-state → applied → export → exported, plus one failing report and one timeout."""
    target = tmp_path / "batch.zip"
    out = _node("""
      const fs=require('fs');
      const png=Buffer.from([137,80,78,71,1,2,3]).toString('base64');
      const listeners=new Set(), log=[];
      function fakeFrame(behaviour){
        const win={postMessage(message,origin){log.push([behaviour,message.type,origin]);
          const reply=data=>setImmediate(()=>{for(const h of [...listeners])h({origin:'http://wb',source:win,data});});
          if(message.type==='wss-view:apply-state'){if(behaviour==='error')reply({type:'wss-view:error',request_id:message.request_id,message:'bad state'});else reply({type:'wss-view:applied',request_id:message.request_id});}
          if(message.type==='wss-view:export')reply({type:'wss-view:exported',request_id:message.request_id,png_base64:png,filename:'CASE_front_wss_2x.png',colorbar_svg:'<svg/>',width:2560,height:1600});
        }};
        const frame={contentWindow:win,removed:false,remove(){this.removed=true;},set src(v){this._src=v;
          if(behaviour==='silent')return;
          setImmediate(()=>{for(const h of [...listeners])h({origin:'http://wb',source:win,data:{type:'wss-view:ready',family:behaviour==='volume'?'volume':'wall',run_identity:'r',case_id:'CASE'}});});}};
        return frame;}
      const order=['ok','error','silent','volume'];
      let n=0;
      const env={origin:'http://wb',reportUrl:id=>'/api/jobs/'+id+'/report',createFrame:()=>fakeFrame(order[n++]),addListener:h=>listeners.add(h),removeListener:h=>listeners.delete(h),setTimeout:(fn,ms)=>setTimeout(fn,Math.min(ms,50)),clearTimeout};
      const progress=[];
      BX.run({jobs:[{id:'j1',case_id:'CASE A',family:'wall'},{id:'j2',case_id:'B',family:'wall'},{id:'j3',case_id:'C',family:'wall'},{id:'j4',case_id:'D',family:'wall'}],state:{family:'wall',preset_name:'x'},options:{scale:2,background:'white',colorbar:'svg',lang:'en'},timeoutMs:50,env,
        onProgress:p=>progress.push([p.phase,p.job.id,p.ok===undefined?null:p.ok])}).then(result=>{
        fs.writeFileSync(""" + json.dumps(str(target)) + """,result.zip);
        console.log(JSON.stringify({count:result.count,failed:result.failed,log,progress}));
      });
    """)
    assert out["count"] == 1
    failed = {item["id"]: item["message"] for item in out["failed"]}
    assert failed["j2"] == "bad state"
    assert "超过" in failed["j3"]
    assert "volume" in failed["j4"] and "wall" in failed["j4"]
    assert [entry[1] for entry in out["log"] if entry[0] == "ok"] == ["wss-view:apply-state", "wss-view:export"]
    assert all(entry[2] == "http://wb" for entry in out["log"])
    assert out["progress"][0] == ["start", "j1", None] and out["progress"][1] == ["done", "j1", True]
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "CASE_A_j1/CASE_front_wss_2x.png" in names and "CASE_A_j1/CASE_front_wss_2x_colorbar.svg" in names and "manifest.json" in names
        assert archive.read("CASE_A_j1/CASE_front_wss_2x.png") == bytes([137, 80, 78, 71, 1, 2, 3])
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema_version"] == "wss-deploy.batch_export/v1"
        assert [item["ok"] for item in manifest["results"]] == [True, False, False, False]


# ---------------------------------------------------------------------------
# Boot smoke test: the whole workbench script runs against a stub DOM and a canned
# service so a missing element id or a boot-time exception cannot slip through.
# ---------------------------------------------------------------------------
_BOOT = r"""
const fs=require('fs');
const html=fs.readFileSync(HTML,'utf8');
const ids=new Set([...html.matchAll(/id="([A-Za-z0-9_-]+)"/g)].map(m=>m[1]));
class Element{constructor(tag,id){this.tagName=String(tag||'div').toUpperCase();this.id=id||'';this.children=[];this.events={};this.hidden=false;this.disabled=false;this.open=false;this.checked=false;this.value='';this._text='';this.style={};this.dataset={};this.files=[];this.attrs={};this.className='';this.options=[];this.selectedIndex=0;this.readyState=0;
  const s=new Set();this.classList={add:(...c)=>c.forEach(x=>s.add(x)),remove:(...c)=>c.forEach(x=>s.delete(x)),toggle:(c,f)=>{if(f===undefined)f=!s.has(c);f?s.add(c):s.delete(c);return f;},contains:c=>s.has(c)};}
  set textContent(v){this._text=String(v);} get textContent(){return this._text;} set innerHTML(v){this._text=String(v);} get innerHTML(){return this._text;}
  appendChild(x){this.children.push(x);return x;} append(...xs){xs.forEach(x=>{if(x!==null&&x!==undefined)this.children.push(x);});} prepend(...xs){this.children.unshift(...xs);} replaceChildren(...xs){this.children=xs;} remove(){}
  addEventListener(n,f){(this.events[n]=this.events[n]||[]).push(f);} removeEventListener(){} setAttribute(k,v){this.attrs[k]=String(v);} getAttribute(k){return this.attrs[k]??null;} removeAttribute(k){delete this.attrs[k];} hasAttribute(k){return k in this.attrs;}
  querySelectorAll(){return [];} querySelector(){return null;} closest(){return null;} click(){} focus(){} blur(){} scrollIntoView(){} showModal(){this.open=true;} close(){this.open=false;} getContext(){return null;} matches(){return false;} contains(){return false;} insertBefore(x){this.children.push(x);return x;} get firstChild(){return this.children[0]||null;} get parentNode(){return null;} get parentElement(){return null;}}
const els={};const missing=[];const created=[];
global.Node=Element;
// Elements the page builds at runtime (e.g. the live status line) carry their own ids; only ids that
// exist neither in index.html nor in the rendered tree count as missing.
const docEvents={},winEvents={},opened=[],historyCalls=[];
global.document={getElementById:id=>{const live=[...created].reverse().find(el=>el.id===id);if(live)return live;if(!ids.has(id))missing.push(id);return els[id]||(els[id]=new Element('div',id));},createElement:t=>{const el=new Element(t);created.push(el);return el;},createElementNS:(ns,t)=>{const el=new Element(t);el.namespaceURI=ns;created.push(el);return el;},createTextNode:t=>({nodeValue:String(t)}),querySelectorAll:()=>[],querySelector:()=>null,addEventListener(n,f){(docEvents[n]=docEvents[n]||[]).push(f);},removeEventListener(){},hidden:false,title:'WSS',body:new Element('body'),documentElement:new Element('html')};
global.window=global;global.self=global;global.addEventListener=(n,f)=>{(winEvents[n]=winEvents[n]||[]).push(f);};global.removeEventListener=()=>{};global.focus=()=>{};global.open=(url)=>{opened.push(String(url));return null;};const nativeConfirms=[];global.confirm=m=>{nativeConfirms.push(String(m));return true;};global.prompt=()=>null;global.alert=()=>{};
global.localStorage={_m:new Map(),getItem(k){return this._m.has(k)?this._m.get(k):null;},setItem(k,v){this._m.set(k,String(v));},removeItem(k){this._m.delete(k);}};
global.location={hash:'',href:'http://localhost:8765/',origin:'http://localhost:8765',protocol:'http:',pathname:'/',search:''};
global.history={replaceState(s,t,url){historyCalls.push(['replace',String(url)]);const i=String(url).indexOf('#');global.location.hash=i<0?'':String(url).slice(i);},pushState(s,t,url){historyCalls.push(['push',String(url)]);const i=String(url).indexOf('#');global.location.hash=i<0?'':String(url).slice(i);}};global.navigator={userAgent:'node'};global.performance={now:()=>0};
global.requestAnimationFrame=()=>0;global.cancelAnimationFrame=()=>{};global.matchMedia=()=>({matches:false,addEventListener(){}});
global.setInterval=()=>0;global.clearInterval=()=>{};
global.EventSource=class{constructor(url){calls.push('SSE '+url);this.readyState=1;}addEventListener(){}close(){}static get CLOSED(){return 2;}};
global.THREE={WebGLRenderer:class{constructor(){throw new Error('No WebGL');}}};
const calls=[];
const canned={
 '/api/session':{authenticated:true,shared:false,login:'none',csrf_token:'csrf',username:null,role:null,claimable_owners:[],max_upload_bytes:1},
 '/api/releases':{releases:[{id:'X5D_v51_5seed_20260916',release:'X5D_v51_5seed_20260916',models_count:5,contract:{protocol:'single_frame_wss'}},{id:'PF6_VF6_peak_3seed_20260920',release:'PF6_VF6_peak_3seed_20260920',models_count:6,contract:{protocol:'single_frame_volume'}}]},
 '/api/preferences':{preferences:{schema_version:'wss-deploy.preferences/v1',upload:{units:'mm',release_id:'X5D_v51_5seed_20260916',device:'cpu',seed_count:'all',threads:'',remember_patient:true,last_patient:{patient_id:'P-9',scan_label:'基线',scan_date:'',tags:'AAA'}},notifications:{enabled:true}}},
 '/api/jobs':{jobs:[{id:'j6',case_id:'M',status:'done',version:5,review:{status:'unreviewed'},patient_id:'P-1',tags:[],model_release:{id:'M1_3head_3seed_20260922',contract:{protocol:'single_frame_wss_cycle_multi',fields:{wss:{},tawss:{},osi:{}}}},family:'wall',created_at:'2026-09-22T17:37:06+08:00',max_diameter_mm:72.2},
   {id:'j5',case_id:'E',status:'awaiting_confirmation',version:2,review:{status:'unreviewed'},tags:[],model_release:{id:'X5D_v51_5seed_20260916'},family:'wall',created_at:'2026-09-22T11:38:37-07:00',eta:{basis:'history',n_history:6,faces:23184,family:'wall',status:'awaiting_confirmation',segment:'B',current:null,stages:[{key:'ingest',label:'检查输入',expected_s:0.2,state:'done'},{key:'centerline',label:'提取中心线',expected_s:3.4,state:'done'},{key:'smooth_resample',label:'平滑与重采样',expected_s:4.2,state:'pending'},{key:'inference',label:'模型推理',expected_s:22.0,state:'pending'}],remaining_s:26.2,segment_remaining_s:26.2,b_total_s:26.2}},
   {id:'j7',case_id:'F',status:'awaiting_input',version:1,review:{status:'unreviewed'},tags:[],created_at:'2026-09-22T12:00:00-07:00'},
   {id:'j1',case_id:'A',status:'done',version:3,review:{status:'unreviewed'},tags:[],model_release:{id:'X5D_v51_5seed_20260916'},family:'wall',created_at:'2026-09-21T10:00:00'},{id:'j2',case_id:'B',status:'done',version:2,review:{status:'reviewed',by:'r'},model_release:{id:'PF6_VF6_peak_3seed_20260920'},family:'volume',created_at:'2026-09-21T09:00:00'},{id:'j3',case_id:'C',status:'running',version:1,review:{status:'unreviewed'},created_at:'2026-09-21T08:00:00',eta:{basis:'default',n_history:0,faces:23184,family:'wall',status:'running',segment:'B',current:'features',stages:[{key:'ingest',label:'检查输入',expected_s:0.2,state:'done',elapsed_s:0.1},{key:'centerline',label:'提取中心线',expected_s:3.4,state:'done',elapsed_s:3.3},{key:'smooth_resample',label:'平滑与重采样',expected_s:4.2,state:'done',elapsed_s:4.2},{key:'features',label:'几何特征',expected_s:2.8,state:'running',elapsed_s:1.1},{key:'inference',label:'模型推理',expected_s:7.0,state:'pending'},{key:'metrics',label:'统计汇总',expected_s:0.2,state:'pending'},{key:'morphology',label:'形态测量',expected_s:1.7,state:'pending'},{key:'export',label:'报告与导出',expected_s:0.8,state:'pending'}],remaining_s:11.4,segment_remaining_s:11.4,b_total_s:16.7}}],total:6,page:1,page_size:20},
 '/api/health':{ok:true,version:'0.12.0',started_at:'2026-09-23T01:00:00+08:00',uptime_s:3700,queue:{queued:1,running:1,awaiting_input:1,awaiting_confirmation:1},worker_alive:true,gpu:{available:true,name:'NVIDIA A100'},disk_free_gb:512.3,default_release:'X5D_v51_5seed_20260916',stale_reports:0},
 '/api/patients/P-1/timeline':{patient_id:'P-1',n_scans:2,
   scans:[{input_sha256:'a'.repeat(64),date:'2025-03-01',date_source:'scan_date',scan_label:'基线',case_id:'M0',jobs:[{job_id:'j8',release_id:'M1_3head_3seed_20260922',release_label:'WSS + TAWSS + OSI',family:'wall',review:'reviewed'},{job_id:'j9',release_id:'X5D_v51_5seed_20260916',release_label:'WSS',family:'wall',review:'reviewed'}],
     geometry:{max_diameter_mm:52.1,sac_present:true,sac_volume_ml:96.3},models:{M1_3head_3seed_20260922:{wss_p99_pa:16.0},X5D_v51_5seed_20260916:{wss_p99_pa:15.2}}},
    {input_sha256:'b'.repeat(64),date:'2026-03-01',date_source:'scan_date',scan_label:'随访 1',case_id:'M',jobs:[{job_id:'j6',release_id:'M1_3head_3seed_20260922',release_label:'WSS + TAWSS + OSI',family:'wall',review:'unreviewed'}],
     geometry:{max_diameter_mm:55.3,sac_present:true,sac_volume_ml:110.0},models:{M1_3head_3seed_20260922:{wss_p99_pa:17.0}}}],
   series:[],growth:{max_diameter_mm:{per_year:3.2,delta:3.2,days:365,from:{date:'2025-03-01',value:52.1},to:{date:'2026-03-01',value:55.3},basis:'first_last'}},
   notes:['年增长率只在两次扫描都填写了扫描日期时计算。']},
 '/api/jobs/j6':{job:{id:'j6',case_id:'M',status:'done',version:5,review:{status:'unreviewed'},patient_id:'P-1',scan_label:'随访 1',tags:[],family:'wall',created_at:'2026-09-22T17:37:06+08:00',
   model_release:{id:'M1_3head_3seed_20260922',contract:{protocol:'single_frame_wss_cycle_multi',fields:{wss:{},tawss:{},osi:{}}}},mapping:{'3':'out-le'},
   summary:{peak:{p99_pa:17.01,max_pa:52.6},wss_field_pa:{area_frac_low:0.3,area_frac_high:0.05,area_frac_very_high:0.01,thresholds_pa:[0.4,4,7]},timing_s:{total:48},
     fields:{wss:{},tawss:{},osi:{}},model_release:{release:'M1_3head_3seed_20260922',loaded_weights:[{},{},{}]},
     quality:{level:'review',label:'存在不确定性，建议复核',reasons:['五模型离散度超过常规范围，建议复核热点和分支']},
     reference_assessment:{status:'review',checks:[{path:'geometry.主动脉.radius_max_mm',units:'mm',value:55.1,min:9.2,max:50.3,status:'review'},{path:'cloud.spacing_mm',units:'mm',value:0.5,min:0.4,max:0.6,status:'pass'}],reasons:[]},
     cycle:{definition:{period_s:0.8},fields:{tawss:{mean:0.6648,p99:4.3575,thresholds:[0.4,4,7],area_frac:{low:0.6479}},osi:{mean:0.1348,thresholds:[0.1,0.2,0.3],area_frac:{above_t0:0.5663,above_t2:0.0774}}},
       stagnation:{area_frac:0.4557,area_mm2:15544.8,per_branch:{'主动脉':{area_mm2:14511.6,frac:0.558},'右髂总':{area_mm2:296.8,frac:0.23}}}},
     findings_top:[{id:'F1',kind:'high_wss_cluster',label:'左髂内高 WSS 区',branch:'左髂内',value:52.6,units:'Pa',severity:'attention'},{id:'F2',kind:'stagnation_cluster',label:'主动脉滞留区',branch:'主动脉',value:145.1,units:'cm²',severity:'attention'},{id:'F3',kind:'high_osi_cluster',label:'左髂总高 OSI 区',branch:'左髂总',value:0.39,units:'1',severity:'info'},{id:'F4',label:'第四条',value:1}],
     morphology:{aorta:{name:'主动脉',segment_id:0,reference_diameter_mm:19.0,n_reoriented:0,n_excluded:0,max:{max_diameter_mm:72.2,s_from_root_mm:154},sac:{present:true,length_mm:87.0,volume_ml:225.5,threshold_mm:28.4}},
       branches:[{segment_id:2,name:'左髂总',n_reoriented:5,n_excluded:1,n_stations:36}],lumen_volume_ml:310.2,notes:['左髂总：5 站重新定向，1 站截面不可靠未计入直径统计（共 36 站）']}}}},
 '/api/jobs/j6/review':{job:{id:'j6',version:6}},
 '/api/jobs/j3':{job:{id:'j3',case_id:'C',status:'running',stage:'B',phase:'features',version:1,review:{status:'unreviewed'},tags:[],family:'wall',created_at:'2026-09-21T08:00:00',eta:{basis:'default',n_history:0,faces:23184,family:'wall',status:'running',segment:'B',current:'features',stages:[{key:'ingest',label:'检查输入',expected_s:0.2,state:'done',elapsed_s:0.1},{key:'centerline',label:'提取中心线',expected_s:3.4,state:'done',elapsed_s:3.3},{key:'smooth_resample',label:'平滑与重采样',expected_s:4.2,state:'done',elapsed_s:4.2},{key:'features',label:'几何特征',expected_s:2.8,state:'running',elapsed_s:1.1},{key:'inference',label:'模型推理',expected_s:7.0,state:'pending'},{key:'metrics',label:'统计汇总',expected_s:0.2,state:'pending'},{key:'morphology',label:'形态测量',expected_s:1.7,state:'pending'},{key:'export',label:'报告与导出',expected_s:0.8,state:'pending'}],remaining_s:11.4,segment_remaining_s:11.4,b_total_s:16.7}}},
 '/api/jobs/j5':{job:{id:'j5',case_id:'E',status:'awaiting_confirmation',version:2,review:{status:'unreviewed'},tags:[],family:'wall',created_at:'2026-09-22T11:38:37-07:00',eta:{basis:'history',n_history:6,faces:23184,family:'wall',status:'awaiting_confirmation',segment:'B',current:null,stages:[{key:'ingest',label:'检查输入',expected_s:0.2,state:'done'},{key:'centerline',label:'提取中心线',expected_s:3.4,state:'done'},{key:'smooth_resample',label:'平滑与重采样',expected_s:4.2,state:'pending'},{key:'inference',label:'模型推理',expected_s:22.0,state:'pending'}],remaining_s:26.2,segment_remaining_s:26.2,b_total_s:26.2},
   detail:'自动命名未满足已校准的 95% 自动门控，请结合原始影像核对开口名称。 原因：几何置信度代理 88.4% 低于门槛 95.0%；命名 profile 未绑定具体模型发布版本',
   a:{input_check:{status:'pass',openings:5,components:1},proposal:{confirmation_required:true,confidence:0.884,mapping:{'3':'out-le','4':'out-li','5':'out-ri','6':'out-re'},
     confidence_reasons:['右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对','几何置信度代理 88.4% 低于门槛 95.0%'],flags:['右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对'],direction_note:'STL 未提供患者方向；世界 XYZ 不能自动解释为患者左右，请参考原始影像核对。',
     endpoints:[{segment_id:0,kind:'inlet',radius_mm:13.2,center_mm:[1,2,3]},{segment_id:3,kind:'outlet',radius_mm:2.8,center_mm:[1,2,3]},{segment_id:4,kind:'outlet',radius_mm:2.8,center_mm:[1,2,3]},{segment_id:5,kind:'outlet',radius_mm:3.0,center_mm:[1,2,3]},{segment_id:6,kind:'outlet',radius_mm:3.5,center_mm:[1,2,3]}]}}}},
 '/api/jobs/j5/confirm':{job:{id:'j5',status:'queued',version:3}},
 '/api/cases':{cases:[{group_key:'s',input_sha256:'s'.repeat(64),case_ids:['A'],patient_id:'P-1',scan_label:'基线',scan_date:'',tags:['AAA'],latest_at:'x',pending_review:1,runs:[{job_id:'j1',release_id:'X5D_v51_5seed_20260916',family:'wall',status:'done',review:'unreviewed',created_at:'x',version:3,max_diameter_mm:63.4}],latest:{X5D_v51_5seed_20260916:'j1'},missing_releases:['PF6_VF6_peak_3seed_20260920'],families:['wall'],reusable_job_id:'j1'}],total:1,page:1,page_size:20,releases:[{id:'X5D_v51_5seed_20260916',family:'wall'},{id:'PF6_VF6_peak_3seed_20260920',family:'volume'}]},
 '/api/trash':{items:[{id:'t1',case_id:'Old',status_before:'done',release_id:'X5D_v51_5seed_20260916',family:'wall',deleted_at:'2026-09-21T00:00:00',expires_at:'2026-10-21T00:00:00',days_left:29,patient_id:'',scan_label:''}]},
 '/api/jobs/export':{columns:['job_id','case_id','release_id','family','review_status','max_diameter_mm','wss_p99_pa','area_frac_high','area_frac_very_high','population_percentile','speed_p99_m_s'],
   rows:[{job_id:'j1',case_id:'A',release_id:'X5D_v51_5seed_20260916',family:'wall',review_status:'unreviewed',wss_p99_pa:9.12,area_frac_high:0.062,area_frac_very_high:0.004,population_percentile:71.5,max_diameter_mm:63.4},
         {job_id:'j2',case_id:'B',release_id:'PF6_VF6_peak_3seed_20260920',family:'volume',review_status:'reviewed',speed_p99_m_s:1.23,max_diameter_mm:51.2},
         {job_id:'j4',case_id:'D',release_id:'X5D_v51_5seed_20260916',family:'wall',review_status:'reviewed',wss_p99_pa:4.4,area_frac_high:0.011,area_frac_very_high:0,population_percentile:23,max_diameter_mm:28.0}],
   population:{X5D_v51_5seed_20260916:{metric:'p99_pa',values_pa:[3.2,4.8,5.5,7.1,9.9,12.4],case_count:136,reference_set:'train136 折外',thresholds_pa:[0.4,4,7]}}},
 '/static/glossary.json':{schema_version:'wss-deploy.glossary/v1',terms:{p99:{zh:'空间 p99',zh_desc:'…',en:'p99',en_desc:'…'}}},
 '/api/jobs/j1':{job:{id:'j1',case_id:'A',status:'done',version:3,review:{status:'unreviewed'},patient_id:'P-1',scan_label:'基线',scan_date:'',tags:['AAA'],notes:'备注',
   model_release:{id:'X5D_v51_5seed_20260916'},family:'wall',created_at:'2026-09-21T10:00:00',run_identity:'r'.repeat(64),
   mapping:{'3':'out-le','4':'out-li','5':'out-re','6':'out-ri'},
   timing:{compute_s:125.0,attempt_s:41.5,queue_s:2,confirmation_s:0,compute_seconds:125.0},
   a:{input_check:{status:'pass',openings:5,components:1,area_mm2:12000,vertices:50000,bbox_size_mm:[100,90,220],raw_bbox_size:[100,90,220],selected_units:'mm',scale_factor:1},
      proposal:{mapping:{'3':'out-le'},endpoints:[],confidence:0.99}},
   summary:{peak:{p99_pa:9.12,max_pa:14.3},wss_field_pa:{area_frac_low:0.1,area_frac_high:0.062,area_frac_very_high:0.004,thresholds_pa:[0.4,4,7]},
     timing_s:{total:118.2},model_release:{release:'X5D_v51_5seed_20260916'},
     morphology:{schema_version:'wss-deploy.morphology/v1',station_mm:1.0,
       aorta:{segment_id:0,name:'主动脉',length_mm:211.1,reference_diameter_mm:18.2,
         max:{max_diameter_mm:63.4,equivalent_diameter_mm:58.9,min_diameter_mm:52.0,area_mm2:2725.0,s_from_root_mm:120.0,xyz_mm:[1,2,3],distance_from_inlet_mm:120.0},
         sac:{present:true,s_start_mm:80.0,s_end_mm:165.0,length_mm:85.0,volume_ml:96.3,max_diameter_mm:63.4,threshold_mm:27.3},
         neck:{present:true,length_mm:80.0,diameter_mean_mm:19.1,diameter_min_mm:17.8,diameter_max_mm:21.0}},
       lumen_volume_ml:158.2,lumen_volume_method:'capped_mesh_divergence',branches:[],notes:['瘤体位于肾下主动脉段。']},
     narrative:{schema_version:'wss-deploy.narrative/v1',generated_at:'2026-09-22T09:00:00',
       zh:['主动脉最大直径 63.4 mm，位于入口下 120 mm；瘤体长 85 mm，体积约 96 mL。','低 WSS（< 0.4 Pa）区占壁面 10%。','以上为固定收缩期单帧预测的参考描述，非诊断结论。'],
       en:['Maximum aortic diameter 63.4 mm.'],edited:null,edited_by:null,edited_at:null}}}},
 '/api/jobs/j1/metadata':{job:{id:'j1',case_id:'A',version:4}},
 '/api/jobs/j1/narrative':{narrative:{schema_version:'wss-deploy.narrative/v1',zh:['自动句。'],edited:'审阅人改写的结论。',edited_by:'医生A',edited_at:'2026-09-22T10:00:00'},version:4},
 '/api/jobs/j1/rerun':{job:{id:'j9',case_id:'A',status:'queued',version:1}},
 // a reviewed (locked) volume job with a reviewer-edited narrative and no morphology at all
 '/api/jobs/j2':{job:{id:'j2',case_id:'B',status:'done',version:2,review:{status:'reviewed',by:'医生B',at:'2026-09-21T12:00:00'},family:'volume',
   model_release:{id:'PF6_VF6_peak_3seed_20260920'},created_at:'2026-09-21T09:00:00',
   a:{input_check:{status:'pass',openings:5,components:1,area_mm2:12000,vertices:50000,bbox_size_mm:[100,90,220],raw_bbox_size:[100,90,220],selected_units:'mm',scale_factor:1}},
   summary:{fields:{velocity:true},volume_statistics:{speed_m_s:{p99:1.23},pressure_interior_pa:{min:-120,max:340}},timing_s:{total:96.4},
     narrative:{schema_version:'wss-deploy.narrative/v1',zh:['自动句。'],edited:'审阅人改写的结论。',edited_by:'医生A',edited_at:'2026-09-22T10:00:00'}}}},
};
global.fetch=async(url,opts)=>{const path=String(url).split('?')[0];calls.push((opts&&opts.method||'GET')+' '+path);const body=canned[path];if(!body)return {ok:false,status:404,json:async()=>({error:{message:'接口不存在。'}}),headers:{get:()=>null}};return {ok:true,status:200,json:async()=>JSON.parse(JSON.stringify(body)),headers:{get:()=>null},blob:async()=>new Blob([])};};
const errors=[];console.error=(...a)=>errors.push(a.map(String).join(' '));process.on('unhandledRejection',e=>errors.push('unhandled: '+(e&&e.stack||e)));
global.WssWorkbenchCore=require(CORE);global.WssBatchExport=require(BATCH);global.WssReportCommon=require(COMMON);
require(APP);
const g=id=>els[id];
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const fire=(el,name)=>{ if(!el||!el.events[name])throw new Error('no '+name+' handler'); for(const f of [...el.events[name]]) f({preventDefault(){},stopPropagation(){},currentTarget:el,target:el,key:''}); };
const walk=(el,pred,out=[])=>{ if(!el||typeof el!=='object')return out; if(pred(el))out.push(el); for(const child of (el.children||[])) walk(child,pred,out); return out; };
const byText=(root,text)=>walk(root,e=>e._text===text)[0]||null;
const step=async(label,fn)=>{ try{ await fn(); }catch(e){ errors.push(label+': '+(e&&e.stack||e)); } };
const texts=root=>walk(root,e=>typeof e._text==='string'&&e._text).map(e=>e._text);
const byClass=(root,cls)=>walk(root,e=>String(e.className||'').split(/\s+/).includes(cls));
const keydown=async key=>{const ev={key,target:document.body,defaultPrevented:false,preventDefault(){this.defaultPrevented=true;},shiftKey:key==='?'};for(const f of [...(docEvents.keydown||[])])f(ev);await wait(10);};
(async()=>{
  await wait(80);
  // v0.12 C1: with no #job=… the detail pane is the overview (counts come from the latest 100 tasks)
  const overview={tiles:byClass(g('detail'),'overview-tile').length,values:byClass(g('detail'),'tile-value').map(e=>e._text),
    recent:byClass(g('detail'),'recent-item').length,health:texts(g('detail')).includes('v0.12.0')};
  // §19.9 list order, friendly times, no stray "#", one ⋯ menu per row, release chips
  const listInfo={groups:g('jobs-list').children.map(li=>li.className),
    stray:texts(g('jobs-list')).filter(t=>t==='#'||t.includes(' · #')),
    iso:walk(g('jobs-list'),e=>e.tagName==='TIME').filter(e=>/\d{4}-\d{2}-\d{2}T/.test(e._text)).length,
    titles:walk(g('jobs-list'),e=>e.tagName==='TIME').every(e=>/\d{2}:\d{2}:\d{2}$/.test(e.attrs.title||'')),
    more:byClass(g('jobs-list'),'more-button').length,deleteButtons:texts(g('jobs-list')).filter(t=>t==='删除').length,
    kinds:byClass(g('jobs-list'),'kind-chip').map(e=>e._text),
    etaRows:walk(g('jobs-list'),e=>String(e.className||'').startsWith('eta-row ')&&!e.hidden).map(e=>[e.className,walk(e,c=>c.className==='eta-row-text')[0]._text])};
  // exercise the list-mode switch and the trash panel after boot
  await step('cases click',()=>fire(g('list-mode-cases'),'click'));
  await wait(30);
  await step('trash toggle',()=>{ g('trash-panel').open=true; fire(g('trash-panel'),'toggle'); });
  // §15.13: the cohort card loads on its first open and renders three canvases plus the result table
  await step('cohort toggle',()=>{ g('cohort-panel').open=true; fire(g('cohort-panel'),'toggle'); });
  await wait(30);
  await step('cohort filter',()=>{ g('cohort-family').value='wall'; fire(g('cohort-family'),'change'); });
  const cohort={rows:g('cohort-rows').children.length,head:g('cohort-head').children.length,count:g('cohort-count').textContent,status:g('cohort-status').textContent,
    diameterCaption:g('cohort-cap-diameter').textContent};
  // §17.4: the maximum-diameter filter narrows the same table without another request
  await step('cohort diameter',()=>{ g('cohort-diameter').value='50'; fire(g('cohort-diameter'),'input'); });
  const cohortDiameter={rows:g('cohort-rows').children.length,count:g('cohort-count').textContent};
  await step('cohort diameter clear',()=>{ g('cohort-diameter').value=''; fire(g('cohort-diameter'),'input'); });
  const caseChips=walk(g('cases-list'),e=>String(e.className||'').includes('diameter-chip')).length;
  // back to the task list, open j1 and edit its identity fields (§15.2); its timing carries attempt_s (§15.5)
  await step('jobs click',()=>fire(g('list-mode-jobs'),'click'));
  await wait(30);
  await step('open job',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j1')[0],'click'));
  await wait(40);
  const timingText=walk(g('detail'),e=>typeof e._text==='string'&&e._text.includes('本次')).map(e=>e._text)[0]||null;
  // §17.1 / §17.4: morphology facts, the maximum-diameter chip and the narrative card
  const factLabels=walk(g('detail'),e=>String(e.className||'').includes('fact-label')).map(e=>(e.children||[]).map(c=>(c&&c.nodeValue)||'').join(''));
  const morphology={heading:Boolean(byText(g('detail'),'瘤体形态')),value:Boolean(byText(g('detail'),'63.4 mm')),
    labels:factLabels.filter(t=>['最大直径','瘤体长度 / 体积','瘤颈直径 / 长度','全腔体积'].includes(t)).sort(),
    note:Boolean(byText(g('detail'),'瘤体位于肾下主动脉段。')),
    chips:walk(g('detail'),e=>String(e.className||'').includes('diameter-chip')).length,
    gloss:[...new Set(walk(g('detail'),e=>e.attrs&&e.attrs['data-gloss']).map(e=>e.attrs['data-gloss']))].sort()};
  const narrative={sentences:walk(g('detail'),e=>e.tagName==='LI'&&typeof e._text==='string'&&e._text.includes('最大直径 63.4 mm')).length,
    edit:Boolean(byText(g('detail'),'编辑结论'))};
  // §17.2: the reviewer rewrites the conclusion; the dialog PUTs {version, text}
  await step('open narrative',()=>fire(byText(g('detail'),'编辑结论'),'click'));
  await wait(20);
  const narrativeForm=Boolean(byText(g('modal-body'),'保存'))&&Boolean(byText(g('modal-body'),'恢复自动'));
  await step('save narrative',()=>fire(byText(g('modal-body'),'保存'),'click'));
  await wait(60);
  // C2: editing moved into the header's ⋯ menu (no large delete button on the page any more)
  const detailDelete=Boolean(byText(g('detail'),'删除任务'));
  await step('open detail menu',()=>fire(walk(g('detail'),e=>e.attrs&&e.attrs['aria-label']==='更多操作')[0],'click'));
  const menuItems=texts(g('action-menu'));
  await step('edit metadata',()=>fire(byText(g('action-menu'),'编辑信息'),'click'));
  await wait(20);
  const metadataForm=Boolean(byText(g('modal-body'),'保存信息'));
  await step('save metadata',()=>fire(byText(g('modal-body'),'保存信息'),'click'));
  await wait(60);
  // §15.4: multi-select one finished job and rerun it with another release
  await step('select mode',()=>fire(g('select-jobs'),'click'));
  await wait(30);
  await step('tick job',()=>{ const box=walk(g('jobs-list'),e=>e.tagName==='INPUT'&&e.attrs['aria-label']==='选择 A')[0]; box.checked=true; fire(box,'change'); });
  await step('open rerun',()=>fire(g('rerun-selected'),'click'));
  await wait(20);
  const rerunForm=Boolean(byText(g('modal-body'),'开始重跑'));
  await step('start rerun',()=>fire(byText(g('modal-body'),'开始重跑'),'click'));
  await wait(80);
  // §17.4: a reviewed (locked) job shows the edited conclusion read-only, and a job without
  // morphology renders no chip and no facts block at all
  await step('open locked job',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j2')[0],'click'));
  await wait(40);
  const lockedNarrative={edit:Boolean(byText(g('detail'),'编辑结论')),lockedBadge:Boolean(byText(g('detail'),'已锁定')),
    edited:Boolean(byText(g('detail'),'审阅人已修改 · 医生A')),text:Boolean(byText(g('detail'),'审阅人改写的结论。')),
    chips:walk(g('detail'),e=>String(e.className||'').includes('diameter-chip')).length,
    morphology:Boolean(byText(g('detail'),'瘤体形态'))};
  g('modal').close();
  // §19.2 / B4 / G4 / B2: an M1 job with cycle numbers, top findings, an out-of-range check and a patient timeline
  await step('open m1',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j6')[0],'click'));
  await wait(60);
  const detailTexts=texts(g('detail'));
  const m1={title:detailTexts.includes('峰值 WSS · TAWSS · OSI'),
    cycle:walk(g('detail'),e=>String(e.className||'').includes('fact-label')).map(e=>(e.children||[]).map(c=>(c&&c.nodeValue)||'').join('')).filter(t=>['TAWSS 均值','OSI 均值','滞留区占比'].includes(t)),
    tawss:detailTexts.includes('0.665 Pa'),lowTawss:detailTexts.includes('低 TAWSS（< 0.4 Pa）占 65%'),stagnation:detailTexts.includes('主要位于主动脉'),
    findings:byClass(g('detail'),'finding-row').length,osiUnit:detailTexts.includes('0.390'),
    alert:detailTexts.find(t=>t.includes('超出模型训练范围'))||null,alertItem:detailTexts.some(t=>t.startsWith('主动脉最大半径 55.1 mm，高于模型训练范围')),
    quality:detailTexts.find(t=>t.includes('个模型离散度'))||null,
    reliability:detailTexts.includes('截面可靠性：5 站重定向，1 站不可靠（已排除）'),
    secondary:detailTexts.includes('226 mL'),
    timelineRows:walk(g('detail'),e=>e.tagName==='TR'&&e.parentTag!=='x').filter(tr=>walk(tr,c=>c.tagName==='INPUT').length).length,
    growth:detailTexts.includes('+3.2 mm/年'),charts:walk(g('detail'),e=>e.tagName==='SVG').length,
    legend:byClass(g('detail'),'spark-legend').length,reviewButton:Boolean(byText(g('detail'),'审阅签字'))};
  // review sign-off from the header, then the "next case to review" offer
  await step('open review',()=>fire(byText(g('detail'),'审阅签字'),'click'));
  await wait(10);
  const reviewModal={checklist:byClass(g('modal-body'),'review-checklist').length,approve:Boolean(byText(g('modal-body'),'签字通过并锁定'))};
  await step('approve',async()=>{const who=walk(g('modal-body'),e=>e.attrs&&e.attrs['aria-label']==='审阅人')[0];who.value='R';fire(who,'input');const ack=walk(g('modal-body'),e=>e.id==='review-ack-j6')[0];ack.checked=true;fire(ack,'change');fire(byText(g('modal-body'),'签字通过并锁定'),'click');});
  await wait(80);
  const nextReview={notice:texts(g('notice')).find(t=>t.includes('待审阅的任务'))||null,button:Boolean(byText(g('notice'),'下一例待审阅')),modalOpen:g('modal').open};
  // C3: the outlet confirmation page — one status sentence, reasons in plain words, then "处理下一例"
  await step('open confirm',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j5')[0],'click'));
  await wait(40);
  const confirmTexts=texts(g('detail'));
  const confirmEta=byClass(g('detail'),'eta-note').map(e=>e._text);
  const confirm={headline:confirmTexts.includes('自动命名置信度 88%，需要人工核对'),why:Boolean(byText(g('detail'),'为什么需要核对')),
    reasons:byClass(g('detail'),'outlet-why').flatMap(el=>walk(el,e=>e.tagName==='LI').map(e=>e._text)),
    layout:byClass(g('detail'),'outlet-layout').length,panel:byClass(g('detail'),'outlet-panel').length,
    longDetail:confirmTexts.some(t=>t.includes('命名 profile 未绑定'))};
  await step('confirm outlets',async()=>{const ack=walk(g('detail'),e=>e.id==='outlet-ack')[0];ack.checked=true;fire(ack,'change');fire(byText(g('detail'),'确认出口并开始预测'),'click');});
  await wait(80);
  const nextConfirm={button:Boolean(byText(g('notice'),'处理下一例')),text:texts(g('notice')).find(t=>t.includes('等待确认'))||null};
  // §21.3: a running job shows the stage bar and the remaining time
  await step('open running',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j3')[0],'click'));
  await wait(40);
  const bars=byClass(g('detail'),'eta-bar');
  const runningEta={bars:bars.length,segments:bars[0]?bars[0].children.map(c=>c.className):[],headline:(byClass(g('detail'),'eta-headline')[0]||{})._text,
    sub:(byClass(g('detail'),'eta-sub')[0]||{})._text,valuenow:bars[0]?bars[0].attrs['aria-valuenow']:null,stageRows:byClass(g('detail'),'eta-stages').length};
  // §19.9 routing: selections push #job=<id>; a hash change selects that job
  const pushed=historyCalls.filter(c=>c[0]==='push').map(c=>c[1]);
  const before=calls.filter(c=>c==='GET /api/jobs/j2').length;
  global.location.hash='#job=j2';
  await step('hashchange',()=>{for(const f of winEvents.hashchange||[])f({});});
  await wait(40);
  const routed=calls.filter(c=>c==='GET /api/jobs/j2').length>before;
  // shortcuts: N opens the upload dialog, ? shows the help layer
  await keydown('n');
  const shortcutUpload=g('upload-dialog').open;
  g('upload-dialog').close();
  await keydown('?');
  const help=document.body.children.some(c=>c.className==='wss-shortcuts');
  console.log(JSON.stringify({errors,missing:[...new Set(missing)],calls:[...new Set(calls)],workspaceHidden:g('workspace').hidden,loginHidden:g('login-panel').hidden,
    jobs:g('jobs-list').children.length,cases:g('cases-list').children.length,trash:g('trash-list')?g('trash-list').children.length:null,
    cohort,cohortDiameter,caseChips,morphology,narrative,narrativeForm,lockedNarrative,timingText,metadataForm,rerunForm,
    overview,listInfo,detailDelete,menuItems,m1,reviewModal,nextReview,confirm,confirmEta,runningEta,nextConfirm,pushed,routed,shortcutUpload,help,opened,
    units:g('upload-units').value,device:g('upload-device').value,patient:g('patient-id').value,connection:g('connection').textContent,title:document.title}));
})();
"""


def test_workbench_boots_against_stub_dom_and_canned_service(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node is required")
    program = ("const CORE=" + json.dumps(str(CORE)) + ";const BATCH=" + json.dumps(str(BATCH)) + ";const APP=" + json.dumps(str(STATIC_DIR / "app.js"))
               + ";const COMMON=" + json.dumps(str(STATIC_DIR / "report_common.js"))
               + ";const HTML=" + json.dumps(str(STATIC_DIR / "index.html")) + ";\n" + _BOOT)
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True)
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["errors"] == [], out["errors"]
    assert out["missing"] == [], out["missing"]          # every getElementById target exists in index.html
    assert out["workspaceHidden"] is False and out["loginHidden"] is True
    assert "GET /api/session" in out["calls"] and "GET /api/jobs" in out["calls"] and "GET /api/preferences" in out["calls"]
    assert "GET /api/cases" in out["calls"] and "GET /api/trash" in out["calls"] and "SSE /api/events" in out["calls"]
    assert out["jobs"] >= 1 and out["cases"] == 1 and out["trash"] == 1
    # preferences applied to the upload form (C6): device and last patient fields
    assert out["patient"] == "P-9"   # selects in the stub have no options, so only text inputs can be checked
    # cohort overview (§15.13): loaded on first open, filtered to the wall family by the change handler
    assert "GET /api/jobs/export" in out["calls"]
    assert out["cohort"]["head"] == 7 and out["cohort"]["rows"] == 2 and out["cohort"]["count"] == "· 2 / 3"
    assert "筛选出 2 个" in out["cohort"]["status"]
    # §17.4: maximum-diameter column, histogram caption and `最大直径 ≥` filter
    assert out["cohort"]["diameterCaption"] == "最大直径分布 · mm · 2 例"
    assert out["cohortDiameter"] == {"rows": 1, "count": "· 1 / 3"}
    assert out["caseChips"] == 1                     # case card chip from the latest done run's snapshot
    assert out["morphology"]["heading"] is True and out["morphology"]["value"] is True and out["morphology"]["note"] is True
    assert out["morphology"]["labels"] == sorted(["最大直径", "瘤体长度 / 体积", "瘤颈直径 / 长度", "全腔体积"])
    assert out["morphology"]["chips"] == 1
    # the §17 glossary keys the backend adds must be exactly these
    assert {"max_diameter", "aneurysm_sac", "aneurysm_neck", "lumen_volume", "narrative"} <= set(out["morphology"]["gloss"])
    # §17.2: the generated sentences render as a list and the edit dialog PUTs to the narrative endpoint
    assert out["narrative"] == {"sentences": 1, "edit": True} and out["narrativeForm"] is True
    assert "PUT /api/jobs/j1/narrative" in out["calls"]
    assert out["lockedNarrative"] == {"edit": False, "lockedBadge": True, "edited": True, "text": True, "chips": 0, "morphology": False}
    # §15.5: a job whose attempt_s differs from compute_s shows both numbers
    assert out["timingText"] and out["timingText"].startswith("本次 42 秒 · 累计 2 分 5 秒")
    # §15.2 / §15.4: both dialogs open and post through request() (CSRF-bearing)
    assert out["metadataForm"] is True and "POST /api/jobs/j1/metadata" in out["calls"]
    assert out["rerunForm"] is True and "POST /api/jobs/j1/rerun" in out["calls"]
    # ---- v0.12 (contract §19) ----
    # C1 overview: four count tiles from the latest-100 list; health details shown (version)
    ov = out["overview"]
    assert ov["tiles"] == 4 and ov["values"][:3] == ["2", "1", "2"] and ov["recent"] == 3 and ov["health"] is True
    # §19.9 group order, friendly times with the full local time in the tooltip, no stray "#", ⋯ per row
    li = out["listInfo"]
    assert li["groups"] == ["job-group group-action", "job-group group-active", "job-group group-pending", "job-group group-reviewed"]
    assert li["stray"] == [] and li["iso"] == 0 and li["titles"] is True
    assert li["more"] == 6 and li["deleteButtons"] == 0
    assert "WSS + TAWSS + OSI" in li["kinds"] and "压力 + 速度体场" in li["kinds"]
    # C2: no big delete button on the page; the header ⋯ menu carries rerun / bundle / edit / delete
    assert out["detailDelete"] is False
    assert {"用其他发布包重跑…", "打包下载 zip", "编辑信息", "删除任务…"} <= set(out["menuItems"])
    # §19.2 / B4 / G4: M1 cycle cards, top-3 findings, alert banner, model-count wording, reliability sentence
    m1 = out["m1"]
    assert m1["title"] and m1["cycle"] == ["TAWSS 均值", "OSI 均值", "滞留区占比"]
    assert m1["tawss"] and m1["lowTawss"] and m1["stagnation"] and m1["findings"] == 3 and m1["osiUnit"]
    assert m1["alert"] == "1 项几何测量超出模型训练范围，模型集成存在不确定性，结果需要复核。" and m1["alertItem"]
    assert m1["quality"].startswith("3 个模型")                 # not "五模型" for a three-model release
    assert m1["reliability"] and m1["secondary"]
    # B2: timeline card with two scans, growth per year and one chart per quantity (single-point lines dropped)
    assert m1["timelineRows"] == 2 and m1["growth"] and m1["charts"] == 3 and m1["legend"] == 0
    # G4: review sign-off from the header, then "下一例待审阅"
    assert out["reviewModal"] == {"checklist": 1, "approve": True}
    assert "POST /api/jobs/j6/review" in out["calls"]
    assert out["nextReview"]["button"] is True and out["nextReview"]["modalOpen"] is False and "A" in out["nextReview"]["notice"]
    # C3: one status sentence, plain-language reasons, two-column layout, then "处理下一例"
    c = out["confirm"]
    assert c["headline"] and c["why"] and c["layout"] == 1 and c["panel"] == 1 and c["longDetail"] is False
    assert "整体命名把握度 88%，低于自动放行要求的 95%。" in c["reasons"] and len(c["reasons"]) == len(set(c["reasons"]))
    assert "POST /api/jobs/j5/confirm" in out["calls"] and out["nextConfirm"]["button"] is True and "F" in out["nextConfirm"]["text"]
    # §19.9 routing and shortcuts
    assert "/#job=j1" in out["pushed"] and "/#job=j6" in out["pushed"] and out["routed"] is True
    assert out["shortcutUpload"] is True and out["help"] is True
    assert "GET /api/health" in out["calls"] and "GET /api/patients/P-1/timeline" in out["calls"]
    # §21.3: list rows (running: bar + remaining; waiting for outlets: time after confirmation), confirm note, detail bar
    assert ["eta-row running", "还需约 10 秒"] in out["listInfo"]["etaRows"] and ["eta-row waiting", "确认后约 25 秒"] in out["listInfo"]["etaRows"]
    assert out["confirmEta"] == ["确认后约 25 秒出结果"]
    r = out["runningEta"]
    assert r["bars"] == 1 and r["segments"] == ["eta-seg done"] * 3 + ["eta-seg running"] + ["eta-seg pending"] * 4
    assert r["headline"] == "预计还需约 10 秒" and r["sub"] == "正在：几何特征（第 4 / 8 步）" and r["valuenow"] == "44" and r["stageRows"] == 1


def test_routing_groups_and_next_case_helpers():
    out = _node("""
      const hashes=[WB.parseJobHash('#job=20260922_173705_27065942b465'),WB.parseJobHash('job=a_1'),WB.parseJobHash('#view=x&job=b-2'),
        WB.parseJobHash('#job=../x'),WB.parseJobHash('#job='),WB.parseJobHash(''),WB.parseJobHash(null),WB.jobHash('a_1'),WB.jobHash('a/b')];
      const jobs=[{id:'d1',status:'done',review:{status:'unreviewed'},created_at:'2026-09-22T10:00:00+08:00'},
        {id:'c1',status:'awaiting_confirmation',created_at:'2026-09-22T09:00:00+08:00'},{id:'c2',status:'awaiting_input',created_at:'2026-09-21T09:00:00-07:00'},
        {id:'f1',status:'failed'},{id:'i1',status:'interrupted'},{id:'r1',status:'running_B'},{id:'q1',status:'queued'},
        {id:'v1',status:'done',review:{status:'reviewed'}},{id:'x1',status:'cancelled'},{id:'d2',status:'done',review:{status:'reopened'},created_at:'2026-09-20T10:00:00+08:00'}];
      const g=WB.workGroups(jobs);
      const groups=Object.fromEntries(Object.entries(g).map(([k,v])=>[k,v.map(j=>j.id)]));
      const order=WB.WORK_GROUPS.map(x=>[x.key,x.label,x.open]);
      const next=[WB.nextToConfirm(jobs,'c1')?.id,WB.nextToConfirm(jobs,'zz')?.id,WB.nextToReview(jobs,'d2')?.id,WB.nextToReview(jobs,'d1')?.id,WB.nextToReview([jobs[0]],'d1')];
      const steps=[WB.stepId(['a','b','c'],'b',1),WB.stepId(['a','b','c'],'c',1),WB.stepId(['a','b','c'],'a',-1),WB.stepId(['a','b','c'],null,1),WB.stepId(['a','b','c'],null,-1),WB.stepId([],'a',1)];
      const now=Date.parse('2026-09-23T12:00:00+08:00');
      const week=[{id:'w1',status:'done',created_at:'2026-09-20T12:00:00+08:00',updated_at:'2026-09-01T00:00:00+08:00'},{id:'w2',status:'done',created_at:'2026-09-10T12:00:00+08:00',updated_at:'2026-09-23T11:00:00+08:00'},{id:'w3',status:'failed',created_at:'2026-09-22T12:00:00+08:00'}];
      const ov=WB.overviewModel(jobs.concat(week),now);
      console.log(JSON.stringify({hashes,groups,order,next,steps,week:WB.quickFilter(week,'week',now).map(j=>j.id),action:WB.quickFilter(jobs,'action').map(j=>j.id),counts:ov.counts,recent:ov.recent.map(j=>j.id)}));
    """)
    assert out["hashes"] == ["20260922_173705_27065942b465", "a_1", "b-2", None, None, None, None, "#job=a_1", ""]
    assert out["groups"] == {"action": ["c1", "c2", "f1", "i1"], "active": ["r1", "q1"], "pending": ["d1", "d2"], "reviewed": ["v1"], "other": ["x1"]}
    assert [row[0] for row in out["order"]] == ["action", "active", "pending", "reviewed", "other"]
    assert [row[2] for row in out["order"]] == [True, True, True, False, False]     # 已审阅 / 已取消 collapsed
    # oldest waiting first: c1 (09-22 09:00 +08) vs c2 (09-21 09:00 -07 = 09-22 00:00 +08) → c2 is older
    assert out["next"] == ["c2", "c2", "d1", "d2", None]
    assert out["steps"] == ["c", "c", "a", "a", "c", None]
    assert out["week"] == ["w1"] and out["action"] == ["c1", "c2", "f1", "i1"]
    # the week fixtures join the list: w1/w2 are unreviewed (pending), w3 failed (action); d1, d2, w1 finished this week
    assert out["counts"] == {"action": 5, "active": 2, "pending": 4, "week": 3}
    assert out["recent"][:3] == ["d1", "w1", "d2"]


def test_release_kind_cycle_summary_and_quality_wording():
    out = _node("""
      const kinds=[WB.releaseKind({id:'M1_3head_3seed_20260922',contract:{protocol:'single_frame_wss_cycle_multi',fields:{wss:{},tawss:{},osi:{}}}}),
        WB.releaseKind({id:'PF6_VF6_peak_3seed_20260920',contract:{protocol:'single_frame_volume'}}),WB.releaseKind({id:'X5D',contract:{protocol:'single_frame_wss'}}),
        WB.releaseKind('M1_3head_3seed_20260922'),WB.releaseKind('PF6_VF6_peak_3seed_20260920'),WB.releaseKind('whatever','volume'),WB.releaseKind(null)];
      const cycle={definition:{period_s:0.8},fields:{tawss:{mean:0.66,p99:4.36,thresholds:[0.4,4,7],area_frac:{low:0.648}},osi:{mean:0.135,thresholds:[0.1,0.2,0.3],area_frac:{above_t0:0.566,above_t2:0.077}}},
        stagnation:{area_frac:0.456,area_mm2:15544.8,per_branch:{'主动脉':{area_mm2:14511.6},'左髂总':{area_mm2:15000.1},'右髂总':{area_mm2:'bad'}}}};
      const cs=WB.cycleSummary(cycle);
      const none=[WB.cycleSummary(null),WB.cycleSummary({fields:{}}),WB.cycleSummary({fields:{tawss:{},osi:{}}})];
      const counts=[WB.modelCount({audit:{quality_audit:{seed_count:3}}}),WB.modelCount({model_release:{loaded_weights:[1,2,3,4,5]}}),WB.modelCount({},{compute:{seed_count:2}}),WB.modelCount({})];
      const notes=[WB.qualityNote({reasons:[]},5),WB.qualityNote({reasons:[]},3),WB.qualityNote({reasons:['五模型离散度较高，结果需要人工复核']},3),WB.qualityNote(null,null)];
      console.log(JSON.stringify({kinds,cs,none,counts,notes,sk:[WB.summaryKind({cycle:{}}),WB.summaryKind({fields:{velocity:{}}}),WB.summaryKind({peak:{}})]}));
    """)
    assert out["kinds"] == ["cycle", "volume", "wall", "cycle", "volume", "volume", None]
    cs = out["cs"]
    assert cs["tawss"] == {"mean": 0.66, "p99": 4.36, "lowFrac": 0.648, "lowThreshold": 0.4}
    assert cs["osi"]["highFrac"] == 0.566 and cs["osi"]["t2"] == 0.3
    assert cs["stagnation"]["mainBranch"] == "左髂总" and abs(cs["stagnation"]["areaCm2"] - 155.448) < 1e-9 and cs["periodS"] == 0.8
    assert out["none"] == [None, None, None]
    assert out["counts"] == [3, 5, 2, None]
    assert out["notes"][0].startswith("五模型集成离散度") and out["notes"][1].startswith("3 个模型集成离散度")
    assert out["notes"][2] == "3 个模型离散度较高，结果需要人工复核" and out["notes"][3].startswith("模型集成")
    assert out["sk"] == ["cycle", "volume", "wall"]


def test_alert_reliability_and_outlet_wording():
    out = _node("""
      const good=WB.alertModel({quality:{level:'good'},reference_assessment:{status:'pass',checks:[{path:'cloud.spacing_mm',status:'pass'}]}});
      const geo=WB.alertModel({reference_assessment:{status:'review',checks:[{path:'geometry.左髂总.length_mm',units:'mm',value:12.5,min:21.116,max:141.0,status:'review'},{path:'cloud.spacing_mm',units:'mm',value:0.5,min:0.4,max:0.6,status:'pass'},{path:'geometry.主动脉.radius_max_mm',units:'mm',value:null,status:'unknown'}]}});
      const poor=WB.alertModel({quality:{level:'poor',label:'不稳定，建议复核',reasons:['离散度较高']}});
      const reasonsOnly=WB.alertModel({reference_assessment:{status:'review',checks:[],reasons:['部分几何测量超出范围']}});
      const labels=[WB.referencePathLabel('geometry.右髂外.radius_median_mm'),WB.referencePathLabel('cloud.variation'),WB.referencePathLabel('x.y')];
      const morph={aorta:{name:'主动脉',segment_id:0,n_reoriented:2,n_excluded:1,n_stations:210},branches:[{segment_id:0,name:'主动脉',n_reoriented:2,n_excluded:1},{segment_id:2,name:'左髂总',n_reoriented:5,n_excluded:0,n_stations:36},{segment_id:3,name:'左髂外',n_reoriented:0,n_excluded:0,n_stations:59}],notes:['左髂总：5 站重新定向，0 站截面不可靠未计入直径统计（共 36 站）','瘤体位于肾下主动脉段。']};
      const rel=WB.reliabilitySummary(morph);
      const legacy=WB.reliabilitySummary({notes:['左髂总：5 站重新定向，0 站截面不可靠未计入直径统计（共 36 站）','右髂外：3 站重新定向，2 站截面不可靠未计入直径统计（共 65 站）']});
      const plain=WB.reliabilitySummary({notes:['只有说明。']});
      const review=WB.outletReview({confirmation_required:true,confidence:0.8838,confidence_reasons:['右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对','几何间隔置信度只是未校准的代理值','STL 未提供患者方向，左右语义必须人工核对','几何置信度代理 88.4% 低于门槛 95.0%','命名 profile 没有达到联合正确率下界 95%','命名 profile 未绑定具体模型发布版本','某条新原因'],
        flags:['右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对'],direction_note:'STL 未提供患者方向；世界 XYZ 不能自动解释为患者左右，请参考原始影像核对。'},'… 原因：几何置信度代理 88.4% 低于门槛 95.0%；STL 未提供已验证的患者方向，不能自动解释左右');
      const auto=WB.outletReview({confirmation_required:false,confidence:0.991});
      const bare=WB.outletReview(null,null);
      console.log(JSON.stringify({good,geo,poor,reasonsOnly,labels,rel,legacy,plain,review,auto,bare}));
    """)
    assert out["good"] is None
    geo = out["geo"]
    assert geo["title"] == "1 项几何测量超出模型训练范围，结果需要复核。" and geo["severity"] == "warning"
    assert geo["items"][0]["text"] == "左髂总长度 12.5 mm，低于模型训练范围 21.1–141 mm"
    assert out["poor"]["severity"] == "serious" and out["poor"]["items"][0]["text"] == "不稳定，建议复核：离散度较高"
    assert out["reasonsOnly"]["items"][0]["text"] == "部分几何测量超出范围" and out["reasonsOnly"]["title"].startswith("几何测量超出")
    assert out["labels"] == ["右髂外中位半径", "表面粗糙度", "x.y"]
    rel = out["rel"]    # the aorta row appears once even though branches repeat segment 0
    assert rel["text"] == "截面可靠性：7 站重定向，1 站不可靠（已排除）" and rel["other"] == ["瘤体位于肾下主动脉段。"]
    assert rel["details"] == ["主动脉：2 站重定向，1 站不可靠（共 210 站）", "左髂总：5 站重定向，0 站不可靠（共 36 站）"]
    assert out["legacy"]["text"] == "截面可靠性：8 站重定向，2 站不可靠（已排除）" and len(out["legacy"]["details"]) == 2
    assert out["plain"]["text"] is None and out["plain"]["other"] == ["只有说明。"]
    review = out["review"]
    assert review["headline"] == "自动命名置信度 88%，需要人工核对" and review["required"] is True
    assert review["reasons"] == ["右侧髂内与髂外的区分把握度只有 88%，请重点核对右侧的两个出口。", "把握度由几何形态估算，不是经过临床数据校准的概率。",
                                 "STL 文件不带患者方位信息，左右需要对照原始影像确认。", "整体命名把握度 88%，低于自动放行要求的 95%。",
                                 "当前模型版本尚未完成出口命名的独立验证，因此每例都请人工确认。", "某条新原因"]
    assert out["auto"]["headline"] == "自动命名置信度 99%，已自动通过" and out["auto"]["reasons"] == []
    assert out["bare"]["headline"] == "请结合原始影像核对出口命名"


def test_timeline_model_and_spark_geometry():
    out = _node("""
      const data={patient_id:'P-001',n_scans:3,
        scans:[{input_sha256:'a',date:'2025-03-01',date_source:'scan_date',scan_label:'基线',jobs:[{job_id:'x1',release_id:'X5D_v51_5seed_20260916',release_label:'壁面 WSS',family:'wall'},{job_id:'v1',release_id:'PF6_VF6_peak_3seed_20260920',family:'volume'}],
            geometry:{max_diameter_mm:52.1,sac_volume_ml:96.3},models:{X5D_v51_5seed_20260916:{wss_p99_pa:16.6},PF6_VF6_peak_3seed_20260920:{speed_p99_m_s:1.2}}},
          {input_sha256:'b',date:'2025-09-01',date_source:'scan_date',jobs:[{job_id:'x2',release_id:'X5D_v51_5seed_20260916',family:'wall'},{job_id:'m2',release_id:'M1_3head_3seed_20260922',family:'wall'}],
            geometry:{max_diameter_mm:53.0,sac_volume_ml:null},models:{X5D_v51_5seed_20260916:{wss_p99_pa:16.9},M1_3head_3seed_20260922:{wss_p99_pa:17.0}}},
          {input_sha256:'c',date:'2026-03-01',date_source:'created_at',jobs:[{job_id:'m3',release_id:'M1_3head_3seed_20260922',family:'wall'}],geometry:{max_diameter_mm:55.3},models:{M1_3head_3seed_20260922:{wss_p99_pa:17.4}}}],
        growth:{max_diameter_mm:{per_year:3.2,delta:3.2,days:365,from:{date:'2025-03-01'},to:{date:'2026-03-01'}}},growth_recent:{max_diameter_mm:{per_year:4.0,delta:2.3,days:181,from:{date:'2025-09-01'},to:{date:'2026-03-01'}}},
        notes:['n1','',3]};
      const m=WB.timelineModel(data,'M1_3head_3seed_20260922');
      const single=WB.timelineModel({patient_id:'P',n_scans:1,scans:[{input_sha256:'a',jobs:[{job_id:'j'}]}]},'');
      const empty=WB.timelineModel({patient_id:'P',n_scans:0,scans:[],series:[],growth:{},notes:[]},'');
      const geo=WB.sparkGeometry([{id:'g',label:'最大直径',color:'#000',points:[{date:'2025-01-01',value:50},{date:'2026-01-01',value:60}]}],{width:100,height:60,pad:{left:10,right:10,top:10,bottom:10}});
      const undated=WB.sparkGeometry([{id:'g',points:[{date:'',value:1},{date:'',value:1},{date:'x',value:1}]}],{width:100,height:60,pad:{left:0,right:0,top:0,bottom:0}});
      console.log(JSON.stringify({m,single:{n:single.nScans,rows:single.rows.length,charts:single.charts.length},empty:{n:empty.nScans,charts:empty.charts},geo,undated,none:WB.sparkGeometry([],{}),nul:WB.timelineModel(null)}));
    """)
    m = out["m"]
    assert m["patientId"] == "P-001" and m["nScans"] == 3 and [r["sha"] for r in m["rows"]] == ["a", "b", "c"]
    assert [r["id"] for r in m["releases"]] == ["X5D_v51_5seed_20260916", "PF6_VF6_peak_3seed_20260920", "M1_3head_3seed_20260922"]
    assert [r["kind"] for r in m["releases"]] == ["wall", "volume", "cycle"] and m["releases"][1]["metric"] == "speed_p99_m_s"
    assert m["releases"][0]["color"] != m["releases"][2]["color"]
    # the compare job prefers the current release, else the scan's first job
    assert [r["compareJobId"] for r in m["rows"]] == ["x1", "m2", "m3"]
    assert m["rows"][0]["metrics"] == {"X5D_v51_5seed_20260916": 16.6, "PF6_VF6_peak_3seed_20260920": 1.2, "M1_3head_3seed_20260922": None}
    charts = {c["key"]: c for c in m["charts"]}
    assert set(charts) == {"max_diameter_mm", "wss_p99_pa"}       # volume has one scan with a value; speed has one point
    assert [len(line["points"]) for line in charts["max_diameter_mm"]["lines"]] == [3]
    assert [line["id"] for line in charts["wss_p99_pa"]["lines"]] == ["X5D_v51_5seed_20260916", "M1_3head_3seed_20260922"]
    assert m["growth"]["diameter"] == {"perYear": 3.2, "delta": 3.2, "days": 365, "from": "2025-03-01", "to": "2026-03-01"}
    assert m["growth"]["diameterRecent"]["perYear"] == 4.0 and m["growth"]["volume"] is None
    assert m["notes"] == ["n1"] and m["rows"][2]["dateSource"] == "created_at"
    assert out["single"] == {"n": 1, "rows": 1, "charts": 0} and out["empty"] == {"n": 0, "charts": []}
    geo = out["geo"]
    line = geo["lines"][0]["points"]
    assert line[0]["x"] == 10 and line[1]["x"] == 90 and line[0]["y"] > line[1]["y"]      # time-scaled x, larger value higher
    assert geo["lines"][0]["d"].startswith("M10 ") and geo["baseline"] == 50 and geo["dated"] is True
    assert out["undated"]["dated"] is False and [p["x"] for p in out["undated"]["lines"][0]["points"]] == [0, 50, 100]
    assert out["none"] is None and out["nul"] is None


def test_findings_card_keeps_service_order_and_labels_rows():
    out = _node("""
      const items=[{id:'F1',kind:'stagnation_cluster',label:'主动脉滞留区',value:145.1,units:'cm²',severity:'attention'},
        {id:'F2',kind:'max_diameter',label:'最大直径',value:72.2,units:'mm',severity:'info'},
        {id:'F3',kind:'high_osi_cluster',label:'左髂总高 OSI 区',value:0.39,units:'1',severity:'info'},
        {id:'F4',kind:'high_wss_cluster',label:'左髂内高 WSS 区',value:52.6,units:'Pa',severity:'attention'},{kind:'x'}];
      const card=WB.findingsForCard(items,3);
      console.log(JSON.stringify({ids:card.map(i=>i.id),chips:items.slice(0,4).map(WB.findingChip),units:items.slice(0,4).map(WB.findingUnits),
        minRadius:WB.findingChip({kind:'min_radius',severity:'attention'}),note:WB.findingChip({kind:'max_speed',severity:'note'}),none:WB.findingsForCard(null)}));
    """)
    assert out["ids"] == ["F1", "F2", "F3"]                       # service order, new kinds kept, no re-sorting
    assert [c["text"] for c in out["chips"]] == ["需关注", "几何", "参考", "需关注"]
    assert out["units"] == [" cm²", " mm", "", " Pa"]             # OSI is dimensionless ("1" → no unit)
    assert out["minRadius"]["text"] == "几何" and out["note"]["text"] == "记录" and out["none"] == []


def test_eta_view_duration_and_row_summary():
    out = _node("""
      const running={basis:'default',n_history:0,faces:23184,family:'wall',status:'running',segment:'B',current:'features',
        stages:[{key:'ingest',label:'检查输入',expected_s:0.2,state:'done',elapsed_s:0.1},{key:'centerline',label:'提取中心线',expected_s:3.4,state:'done',elapsed_s:3.3},
          {key:'smooth_resample',label:'平滑与重采样',expected_s:4.2,state:'done',elapsed_s:4.2},{key:'features',label:'几何特征',expected_s:2.8,state:'running',elapsed_s:1.1},
          {key:'inference',label:'模型推理',expected_s:7.0,state:'pending'},{key:'metrics',label:'统计汇总',expected_s:0.2,state:'pending'},
          {key:'morphology',label:'形态测量',expected_s:1.7,state:'pending'},{key:'export',label:'报告与导出',expected_s:0.8,state:'pending'}],remaining_s:11.4,segment_remaining_s:11.4,b_total_s:16.7};
      const v=WB.etaView(running), later=WB.etaView(running,{ageS:1}), overdue=WB.etaView(running,{ageS:30});
      const segA={...running,segment:'A',current:'centerline',stages:running.stages.map((s,i)=>i===1?{...s,state:'running',elapsed_s:1}:i===0?s:{...s,state:'pending',elapsed_s:undefined})};
      const queued={status:'queued',segment:'A',stages:running.stages.map(s=>({...s,state:'pending'})),remaining_s:20.3,queue_ahead:2,queue_ahead_s:31};
      const q0={...queued,queue_ahead:0,queue_ahead_s:0};
      const confirm={status:'awaiting_confirmation',segment:'B',stages:running.stages,remaining_s:16.7,b_total_s:16.7,basis:'history',n_history:6};
      const input={status:'awaiting_input',segment:'A',stages:running.stages,remaining_s:20.3,segment_remaining_s:3.4};
      const waiting={...running,waiting:true};
      console.log(JSON.stringify({
        d:[null,0,5,9.9,10,12,13,57,58,61,85,119,600,3605].map(WB.formatDuration),
        tail:[WB.stageRemaining(10,5),WB.stageRemaining(10,8),WB.stageRemaining(10,16),WB.stageRemaining(10,1000),WB.stageRemaining(0,3)],
        v:{status:v.status,head:v.headline,sub:v.sub,pct:v.progress,rem:+v.remaining.toFixed(2),cur:v.currentIndex,sum:+v.segments.reduce((a,s)=>a+s.width,0).toFixed(6),
           minWidth:Math.min(...v.segments.map(s=>s.width)),fills:v.segments.map(s=>s.fill),titles:[v.segments[2].title,v.segments[3].title,v.segments[4].title],basis:v.basis},
        later:{rem:+later.remaining.toFixed(2),fill:later.segments[3].fill},overdue:{rem:+overdue.remaining.toFixed(2),flag:overdue.segments[3].overdue,sub:overdue.sub,fill:overdue.segments[3].fill},
        segA:WB.etaView(segA).headline,queued:[WB.etaView(queued).headline,WB.etaView(queued).sub,WB.etaView(queued,{ageS:25}).headline],q0:WB.etaView(q0).headline,
        confirm:[WB.etaView(confirm).headline,WB.etaView(confirm).basis],input:WB.etaView(input).headline,
        waiting:(()=>{const w=WB.etaView(waiting);return {status:w.status,head:w.headline,cur:w.currentIndex,held:w.segments[3].state,row:WB.etaRowSummary(waiting)};})(),overdueHead:overdue.headline,overdueRow:WB.etaRowSummary(running,{ageS:30}).text,
        rows:[WB.etaRowSummary(running),WB.etaRowSummary(queued),WB.etaRowSummary(q0),WB.etaRowSummary(confirm),WB.etaRowSummary(input)],
        none:[WB.etaView(null),WB.etaView({stages:[]}),WB.etaRowSummary(undefined)]}));
    """)
    assert out["d"] == ["", "几秒", "几秒", "几秒", "约 10 秒", "约 10 秒", "约 15 秒", "约 55 秒", "约 1 分钟", "约 1 分钟", "约 1 分 25 秒",
                        "约 2 分钟", "约 10 分钟", "约 60 分钟"]
    tail = out["tail"]    # same rule as eta.stage_remaining: linear to 80 %, then a shrinking 20 % tail
    assert tail[0] == 5 and abs(tail[1] - 2) < 1e-9 and abs(tail[2] - 1.0) < 1e-9 and 0 < tail[3] < 0.02 and tail[4] == 0
    v = out["v"]
    assert v["status"] == "running" and v["head"] == "预计还需约 10 秒" and v["sub"] == "正在：几何特征（第 4 / 8 步）"
    assert v["rem"] == 11.4 and v["cur"] == 3 and abs(v["sum"] - 100) < 0.01 and v["minWidth"] > 2
    assert v["fills"][:3] == [100, 100, 100] and v["fills"][3] == round(1.1 / 2.8 * 100, 1) and v["fills"][4:] == [0, 0, 0, 0]
    assert v["titles"] == ["平滑与重采样：已完成，用时 4.2 秒（预计 4.2 秒）", "几何特征：进行中，已用 1.1 秒 / 预计 2.8 秒", "模型推理：预计 7.0 秒"]
    assert v["pct"] == 44 and v["basis"] == "按默认耗时估计（同类历史少于 3 例） · 输入 23,184 个面片"
    assert out["later"] == {"rem": 10.4, "fill": 75}              # the running stage advances with the snapshot's age
    ov = out["overdue"]
    assert ov["flag"] is True and ov["fill"] == 96 and ov["sub"].endswith("比预计慢") and 9.7 < ov["rem"] < 9.8   # tail, never negative
    assert out["segA"].endswith("（不含人工确认出口）")
    assert out["queued"] == ["前面 2 个，约 30 秒后开始", "预计约 50 秒后出结果（含排队，不含人工确认出口）", "前面 2 个，几秒后开始"]
    # W2 notes: ``waiting`` (running but held by the stage-B lock) reads as queued; an overdue stage reads 「即将完成」
    assert out["waiting"] == {"status": "queued", "head": "等前一个任务算完后开始", "cur": -1, "held": "pending",
                              "row": {"pct": None, "text": "等待前一个任务", "state": "queued"}}
    assert out["overdueHead"] == "即将完成" and out["overdueRow"] == "即将完成"
    assert out["q0"] == "即将开始"
    assert out["confirm"] == ["确认后约 15 秒出结果", "按 6 例同类历史耗时估计"]
    assert out["input"] == "确认后几秒完成中心线提取"
    rows = out["rows"]
    assert rows[0] == {"pct": 44, "text": "还需约 10 秒", "state": "running"} and rows[1]["text"] == "前面 2 个" and rows[2]["text"] == "即将开始"
    assert rows[3] == {"pct": None, "text": "确认后约 15 秒", "state": "waiting"} and rows[4] is None
    assert out["none"] == [None, None, None]


def test_delete_notice_has_undo_and_close(tmp_path):
    """v0.13: the 「已移入回收站」 notice carries 撤销 (restores exactly that task) and a × close button."""
    if not shutil.which("node"):
        pytest.skip("Node is required")
    head = _BOOT.split("  // shortcuts: N opens the upload dialog")[0]
    tail = r"""
  canned['/api/jobs/j1/delete']={id:'j1',trashed:true};canned['/api/trash/j1/restore']={job:{id:'j1',case_id:'A'}};
  await step('open j1',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j1')[0],'click'));
  await wait(40);
  await step('open detail menu',()=>fire(walk(g('detail'),e=>e.attrs&&e.attrs['aria-label']==='更多操作')[0],'click'));
  await step('delete',()=>fire(byText(g('action-menu'),'删除任务…'),'click'));
  await wait(20);
  // v0.14 (F9): a styled confirmation dialog instead of window.confirm
  const dialog={open:g('modal').open,title:(walk(g('modal-body'),e=>e.id==='modal-title')[0]||{})._text||null,button:Boolean(byText(g('modal-body'),'移入回收站')),
    native:nativeConfirms.length,calledEarly:calls.includes('POST /api/jobs/j1/delete')};
  await step('confirm delete',()=>fire(byText(g('modal-body'),'移入回收站'),'click'));
  await wait(60);
  const n=g('notice'),snap=()=>({hidden:n.hidden,cls:n.className,text:(byClass(n,'notice-text')[0]||{})._text||null,
    undo:Boolean(byText(n,'撤销')),close:byClass(n,'notice-close').map(b=>[b._text,b.attrs['aria-label']])});
  const deleted=snap();
  await step('undo',()=>fire(byText(n,'撤销'),'click'));
  await wait(60);
  const undone=snap(),restoreCalls=calls.filter(c=>c==='POST /api/trash/j1/restore').length;
  await step('close',()=>fire(byClass(n,'notice-close')[0],'click'));
  const closed=n.hidden;
  console.log(JSON.stringify({errors,dialog,deleted,undone,restoreCalls,closed,deleteCall:calls.includes('POST /api/jobs/j1/delete')}));
})();
"""
    program = ("const CORE=" + json.dumps(str(CORE)) + ";const BATCH=" + json.dumps(str(BATCH)) + ";const APP=" + json.dumps(str(STATIC_DIR / "app.js"))
               + ";const COMMON=" + json.dumps(str(STATIC_DIR / "report_common.js"))
               + ";const HTML=" + json.dumps(str(STATIC_DIR / "index.html")) + ";\n" + head + tail)
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True, timeout=120)
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["errors"] == [], out["errors"]
    assert out["dialog"] == {"open": True, "title": "删除 1 个任务", "button": True, "native": 0, "calledEarly": False}
    assert out["deleteCall"] is True
    d = out["deleted"]
    assert d["hidden"] is False and d["text"] == "已移入回收站，30 天内可恢复 1 个任务。" and d["undo"] is True
    assert d["close"] == [["×", "关闭提示"]] and "error" not in d["cls"]
    assert out["restoreCalls"] == 1 and out["undone"]["text"] == "已撤销删除，恢复 1 个任务。" and out["undone"]["undo"] is False
    assert out["undone"]["close"] == [["×", "关闭提示"]] and out["closed"] is True


def _boot_program(tail, extra_canned=""):
    head = _BOOT.split("  // shortcuts: N opens the upload dialog")[0]
    return ("const CORE=" + json.dumps(str(CORE)) + ";const BATCH=" + json.dumps(str(BATCH)) + ";const APP=" + json.dumps(str(STATIC_DIR / "app.js"))
            + ";const COMMON=" + json.dumps(str(STATIC_DIR / "report_common.js"))
            + ";const HTML=" + json.dumps(str(STATIC_DIR / "index.html")) + ";\n"
            + head.replace("global.fetch=async", extra_canned + "\nglobal.fetch=async", 1) + tail)


def test_v014_detail_first_screen_error_records_and_list_patching(tmp_path):
    """F3 conclusion card on the first screen and no 「已完成」 next to 「待审阅」; F9 typed error records (category
    wording, no 重试 when retryable is false, 技术细节 only when admin_detail is present); F4 an unchanged list
    refresh keeps every row node, a changed job rebuilds only its own row."""
    if not shutil.which("node"):
        pytest.skip("Node is required")
    extra = r"""
canned['/api/jobs'].jobs.push({id:'j8',case_id:'G',status:'failed',version:4,review:{status:'unreviewed'},tags:[],created_at:'2026-09-21T07:00:00'},
  {id:'j9',case_id:'H',status:'failed',version:2,review:{status:'unreviewed'},tags:[],created_at:'2026-09-21T06:00:00'});
canned['/api/jobs/j8']={job:{id:'j8',case_id:'G',status:'failed',version:4,review:{status:'unreviewed'},tags:[],created_at:'2026-09-21T07:00:00',
  error:{message:'STL 有 3 个连通片，主体占比不足 60%。',diagnostic_id:'D-8',category:'input_geometry',retryable:false,admin_detail:'Traceback: components=3'}}};
canned['/api/jobs/j9']={job:{id:'j9',case_id:'H',status:'failed',version:2,review:{status:'unreviewed'},tags:[],created_at:'2026-09-21T06:00:00',
  error:{message:'GPU 显存不足，已改用 CPU 重试。',diagnostic_id:'D-9',category:'resource',retryable:true,retry_hint:'cpu'}}};
"""
    tail = r"""
  // F4: an identical refresh keeps every row node; a new version of one job rebuilds that row only
  const v14_rowsBefore=walk(g('jobs-list'),e=>e.dataset&&e.dataset.rowId).map(e=>[e.dataset.rowId,e]);
  await step('refresh same',()=>fire(g('refresh-jobs'),'click'));
  await wait(40);
  const v14_rowsSame=walk(g('jobs-list'),e=>e.dataset&&e.dataset.rowId);
  const v14_keptAll=v14_rowsBefore.every(([id,el])=>v14_rowsSame.includes(el));
  canned['/api/jobs'].jobs.find(j=>j.id==='j1').version=9;
  await step('refresh changed',()=>fire(g('refresh-jobs'),'click'));
  await wait(40);
  const v14_rowsAfter=walk(g('jobs-list'),e=>e.dataset&&e.dataset.rowId);
  const v14_rebuilt=v14_rowsBefore.filter(([id,el])=>!v14_rowsAfter.includes(el)).map(([id])=>id);
  // F3: the conclusion card is part of the result overview, not of the collapsed 「更多结果」
  await step('open j1',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j1')[0],'click'));
  await wait(40);
  const v14_main=byClass(g('detail'),'result-overview-main')[0],v14_more=byClass(g('detail'),'result-more')[0];
  const v14_narrative={inMain:byClass(v14_main,'narrative-card').length,inMore:byClass(v14_more,'narrative-card').length,
    beforeCycle:v14_main?v14_main.children.findIndex(c=>String(c.className||'').includes('narrative-card'))<v14_main.children.findIndex(c=>String(c.className||'').includes('cycle-host')):null};
  const v14_badges=walk(byClass(g('detail'),'job-badges')[0],e=>String(e.className||'').startsWith('badge')).map(e=>e._text);
  // F9: typed error records
  await step('open j8',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j8')[0],'click'));
  await wait(40);
  const v14_t8=texts(g('detail'));
  const v14_geometry={title:v14_t8.includes('输入几何无法计算'),hint:v14_t8.some(t=>t.includes('重试不会改变结果')),retry:Boolean(byText(g('detail'),'重试任务')),
    details:walk(g('detail'),e=>e.tagName==='DETAILS'&&String(e.className||'').includes('admin-detail')).length,detail:v14_t8.includes('Traceback: components=3'),diag:v14_t8.includes('诊断编号：D-8')};
  await step('open j9',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j9')[0],'click'));
  await wait(40);
  const v14_t9=texts(g('detail'));
  const v14_resource={title:v14_t9.includes('计算资源不足'),cpu:v14_t9.some(t=>t.includes('改用 CPU')),retry:Boolean(byText(g('detail'),'重试任务')),
    details:walk(g('detail'),e=>e.tagName==='DETAILS'&&String(e.className||'').includes('admin-detail')).length};
  console.log(JSON.stringify({errors,keptAll:v14_keptAll,rebuilt:v14_rebuilt,narrative:v14_narrative,badges:v14_badges,geometry:v14_geometry,resource:v14_resource}));
})();
"""
    result = subprocess.run(["node", "-e", _boot_program(tail, extra)], check=True, text=True, capture_output=True, timeout=120)
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["errors"] == [], out["errors"]
    assert out["keptAll"] is True and out["rebuilt"] == ["j1"]
    assert out["narrative"] == {"inMain": 1, "inMore": 0, "beforeCycle": True}
    assert out["badges"] == ["待审阅"]
    assert out["geometry"] == {"title": True, "hint": True, "retry": False, "details": 1, "detail": True, "diag": True}
    assert out["resource"] == {"title": True, "cpu": True, "retry": True, "details": 0}


def test_v014_event_labels_limit_messages_and_precomputed_stages():
    """Follow-up: new job event actions get labels (unknown ones stay as written), 429 / 413 bodies become the
    service's own sentence (the per-owner cap with its numbers), and precomputed stages are marked 已预计算."""
    out = _node("""
      const eta={status:'running',stages:[{key:'smooth_resample',label:'平滑与重采样',expected_s:0.4,state:'done',elapsed_s:0.1},
        {key:'features',label:'几何特征',expected_s:1.2,state:'running',elapsed_s:0.3},{key:'inference',label:'模型推理',expected_s:7,state:'pending'}],
        precomputed:['smooth_resample','morphology']};
      const view=WB.etaView(eta);
      const cached=WB.cachedStages({enabled:true,reused:['mesh','pointgeom','morph'],computed:['resample']});
      console.log(JSON.stringify({
        events:['restart_requeued','drain_requeued','device_fallback','precompute_done','precompute_skipped','precompute_cancelled','precompute_failed','analysis_rebuilt','finished','brand_new_action',null].map(a=>WB.eventText(a)),
        cap:WB.limitErrorText(429,{error:{message:'排队和计算中的任务已达上限（3 个，当前 3 个）'},active_jobs:3,max_active_jobs:3},'排队和计算中的任务已达上限（3 个，当前 3 个）'),
        busy:WB.limitErrorText(429,{error:{message:'当前正在接收其他上传，请稍后重试。'}},'当前正在接收其他上传，请稍后重试。'),
        big:WB.limitErrorText(413,null,''),busyNoBody:WB.limitErrorText(429,null,''),
        seg:view.segments.map(s=>[s.key,s.precomputed,s.title]),
        cached:[...cached].sort(),rows:['smooth_resample','features','morphology','inference_5_models','total'].map(k=>WB.timingPrecomputed(k,cached))}));
    """)
    assert out["events"] == ["服务重启后自动续排", "升级排空时重新排队", "显存不足，已改用 CPU 重算", "后台几何预计算完成",
                             "后台几何预计算已跳过（无需计算）", "后台几何预计算已取消", "后台几何预计算未完成（阶段 B 自行计算）",
                             "升级后重建分析", "计算结束", "brand_new_action", ""]
    assert out["cap"] == "排队与计算中的任务已达 3 个上限（当前 3 个），请等部分任务完成后再提交。"
    assert out["busy"] == "当前正在接收其他上传，请稍后重试。"
    assert "上限" in out["big"] and "稍后重试" in out["busyNoBody"]
    assert out["seg"][0][:2] == ["smooth_resample", True] and out["seg"][0][2].endswith("（已预计算）")
    assert out["seg"][1][1] is False and not out["seg"][1][2].endswith("（已预计算）")
    assert out["cached"] == ["features", "morphology", "smooth_resample"] and out["rows"] == [True, True, True, False, False]


def test_v014_process_card_run_record_and_upload_cap_message(tmp_path):
    """The process card marks precomputed stage timings and lists the job's event record with labels; 429 answers
    reach the notice with the service's own sentence (bundle busy) or 「排队与计算中的任务已达 N 个上限」."""
    if not shutil.which("node"):
        pytest.skip("Node is required")
    extra = r"""
const j1=canned['/api/jobs/j1'].job;
j1.summary.timing_s={smooth_resample:0.02,features:0.9,morphology:0.01,inference_5_models:6.8,total:9.1};
j1.summary.geometry_cache={enabled:true,reused:['mesh','pointgeom','morph'],computed:[]};
j1.timing.precompute_s=11.2;
j1.events=[{at:'2026-09-24T01:00:00+08:00',action:'started',status:'running'},{at:'2026-09-24T01:00:05+08:00',action:'progress',status:'running'},
  {at:'2026-09-24T01:00:07+08:00',action:'device_fallback',status:'running',from_device:'cuda:0',to_device:'cpu'},
  {at:'2026-09-24T01:01:00+08:00',action:'restart_requeued',status:'queued'},{at:'2026-09-24T01:02:00+08:00',action:'mystery_action',status:'done'}];
"""
    tail = r"""
  await step('open j1',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j1')[0],'click'));
  await wait(40);
  const v14_card=byClass(g('detail'),'process-card')[0];
  const v14_dd=walk(v14_card,e=>e.tagName==='DD').map(e=>e._text);
  const v14_dt=walk(v14_card,e=>e.tagName==='DT').map(e=>(e.children||[]).map(c=>(c&&c.nodeValue)||'').join(''));
  const v14_events=byClass(v14_card,'event-text').map(e=>e._text.trim());
  const v14_fact=walk(g('detail'),e=>String(e.className||'')==='fact-note').map(e=>e._text).find(t=>t.includes('后台预计算'))||null;
  // 429 answers keep the service's sentence: bundle busy, and the per-owner cap (rerun) with its numbers
  const v14_fetch=global.fetch;
  global.fetch=async(url,opts)=>{const u=String(url),m=(opts&&opts.method)||'GET';
    if(u==='/api/jobs/j1/bundle.zip')return {ok:false,status:429,json:async()=>({error:{message:'正在生成其他打包下载，请稍后重试。'}}),headers:{get:()=>null}};
    if(u==='/api/jobs/j1/rerun'&&m==='POST')return {ok:false,status:429,json:async()=>({error:{message:'排队和计算中的任务已达上限（5 个，当前 5 个）'},active_jobs:5,max_active_jobs:5}),headers:{get:()=>null}};
    return v14_fetch(url,opts);};
  await step('bundle',()=>fire(byText(g('detail'),'打包下载 zip'),'click'));
  await wait(30);
  const v14_bundle=(byClass(g('notice'),'notice-text')[0]||{})._text||null;
  await step('rerun',()=>fire(byText(g('detail'),'用发布包重跑'),'click'));
  await wait(40);
  const v14_cap=(byClass(g('notice'),'notice-text')[0]||{})._text||null;
  console.log(JSON.stringify({errors,dd:v14_dd,dt:v14_dt,events:v14_events,fact:v14_fact,bundle:v14_bundle,cap:v14_cap}));
})();
"""
    result = subprocess.run(["node", "-e", _boot_program(tail, extra)], check=True, text=True, capture_output=True, timeout=120)
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["errors"] == [], out["errors"]
    pairs = dict(zip(out["dt"], out["dd"]))
    assert pairs["平滑与重采样"] == "0.02 秒 · 已预计算" and pairs["几何特征"] == "0.90 秒 · 已预计算" and pairs["形态测量"] == "0.01 秒 · 已预计算"
    assert pairs["五模型预测"] == "6.80 秒" and pairs["后台几何预计算（确认出口期间）"] == "11.2 秒（不计入计算耗时）"
    assert out["events"] == ["开始计算", "显存不足，已改用 CPU 重算（cuda:0 → cpu）", "服务重启后自动续排", "mystery_action"]
    assert out["fact"] and "后台预计算 11 秒" in out["fact"]
    assert out["bundle"] == "正在生成其他打包下载，请稍后重试。"
    assert out["cap"] and "排队与计算中的任务已达 5 个上限（当前 5 个）" in out["cap"]


# ---------------------------------------------------------------------------
# v0.15 round 15 (front-end quick wins): login page, re-login over the page, live-stream / version notices,
# upload checks, identifier hints, cancel confirmation, long waits, first-use guide, dialog focus return.
# ---------------------------------------------------------------------------
def _bare_program(tail, extra_canned=""):
    """The boot harness up to ``require(APP)`` (the page boots against ``canned``), then ``tail`` (its own IIFE)."""
    head = _BOOT.split("(async()=>{\n  await wait(80);")[0]
    return ("const CORE=" + json.dumps(str(CORE)) + ";const BATCH=" + json.dumps(str(BATCH)) + ";const APP=" + json.dumps(str(STATIC_DIR / "app.js"))
            + ";const COMMON=" + json.dumps(str(STATIC_DIR / "report_common.js"))
            + ";const HTML=" + json.dumps(str(STATIC_DIR / "index.html")) + ";\n"
            + head.replace("global.fetch=async", extra_canned + "\nglobal.fetch=async", 1) + tail)


_FOCUS = "const focused=[];Element.prototype.focus=function(){focused.push(this.id||this.tagName);document.activeElement=this;};\n"


def _run(program):
    if not shutil.which("node"):
        pytest.skip("Node is required")
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True, timeout=120)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_v015_login_page_focus_inline_errors_reveal_caps_and_cooldown():
    """User-name mode: remembered user name filled in and the password focused; a wrong password is an inline error
    (no 「登录已过期」, no expired-session path); show / hide and Caps Lock hint; a 429 disables the button with a countdown."""
    extra = _FOCUS + r"""
canned['/api/session']={authenticated:false,shared:true,login:'password',legacy_token_allowed:false,csrf_token:null,username:null,role:null,claimable_owners:[]};
localStorage.setItem('wss-login-user','doctor');
"""
    tail = r"""
(async()=>{
  const g=id=>document.getElementById(id);
  await wait(60);
  const boot={title:g('login-title').textContent,submit:g('login-submit').textContent,user:g('login-username').value,focus:focused[focused.length-1],
    panel:g('login-panel').hidden,workspace:g('workspace').hidden,userGroup:g('login-user-group').hidden,tokenGroup:g('login-token-group').hidden};
  const origFetch=global.fetch;let answer=401;
  global.fetch=async(url,opts)=>{const u=String(url),m=(opts&&opts.method)||'GET';calls.push(m+' '+u);
    if(u==='/api/session'&&m==='POST'){const body=answer===401?{error:{message:'用户名或口令不正确。'}}:{error:{message:'口令错误次数过多，请一分钟后再试。'}};
      return {ok:false,status:answer,json:async()=>body,headers:{get:()=>null}};}
    return origFetch(url,opts);};
  g('login-password').value='wrong';
  await step('submit 401',()=>fire(g('login-form'),'submit'));
  await wait(30);
  const bad={error:g('login-error').hidden?null:g('login-error').textContent,conn:g('connection').textContent,focus:focused[focused.length-1],
    panel:g('login-panel').hidden,relogin:g('relogin-dialog').open,notice:g('notice').hidden};
  // show / hide the password, Caps Lock hint
  g('login-password').type='password';   // the stub DOM does not read attributes from index.html
  await step('reveal',()=>fire(g('reveal-password'),'click'));
  const reveal={type:g('login-password').type,pressed:g('reveal-password').attrs['aria-pressed'],label:g('reveal-password').attrs['aria-label'],text:g('reveal-password').textContent};
  await step('caps',()=>{for(const f of g('login-password').events.keydown)f({getModifierState:k=>k==='CapsLock'});});
  const caps=g('caps-hint').hidden;
  await step('caps off',()=>{for(const f of g('login-password').events.keyup)f({getModifierState:()=>false});});
  const capsOff=g('caps-hint').hidden;
  answer=429;
  await step('submit 429',()=>fire(g('login-form'),'submit'));
  await wait(30);
  const locked={disabled:g('login-submit').disabled,text:g('login-submit').textContent,error:g('login-error').textContent};
  const before=calls.filter(c=>c==='POST /api/session').length;
  await step('submit while waiting',()=>fire(g('login-form'),'submit'));
  await wait(20);
  const blocked=calls.filter(c=>c==='POST /api/session').length===before;
  console.log(JSON.stringify({errors,missing:[...new Set(missing)],boot,bad,reveal,caps,capsOff,locked,blocked}));
  process.exit(0);
})();
"""
    out = _run(_bare_program(tail, extra))
    assert out["errors"] == [] and out["missing"] == [], out
    b = out["boot"]
    assert b == {"title": "登录工作台", "submit": "登录", "user": "doctor", "focus": "login-password", "panel": False, "workspace": True,
                 "userGroup": False, "tokenGroup": True}
    bad = out["bad"]
    assert bad["error"] and "用户名或口令不正确" in bad["error"] and "区分大小写" in bad["error"]
    assert "过期" not in bad["conn"] and bad["focus"] == "login-password" and bad["panel"] is False and bad["relogin"] is False
    assert out["reveal"] == {"type": "text", "pressed": "true", "label": "隐藏口令", "text": "隐藏"}
    assert out["caps"] is False and out["capsOff"] is True
    assert out["locked"]["disabled"] is True and out["locked"]["text"].endswith("秒后可再试") and "一分钟" in out["locked"]["error"]
    assert out["blocked"] is True


def test_v015_expired_session_opens_login_over_the_page_and_resumes_the_same_case():
    """A 401 while working keeps the page (workspace, open case, #job) and opens the login form in a modal; the same
    user logging in resumes without a reload.  A wrong current password in 改口令 (401) is not an expired session."""
    extra = _FOCUS + r"""
canned['/api/session']={authenticated:true,shared:true,login:'password',legacy_token_allowed:false,csrf_token:'csrf',username:'admin',role:'admin',claimable_owners:[]};
"""
    tail = r"""
(async()=>{
  const g=id=>document.getElementById(id);
  await wait(80);
  await step('open j1',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j1')[0],'click'));
  await wait(40);
  const origFetch=global.fetch;let mode='ok';
  global.fetch=async(url,opts)=>{const u=String(url).split('?')[0],m=(opts&&opts.method)||'GET';
    if(mode==='expired'&&!(u==='/api/session')){calls.push(m+' '+u+' (401)');return {ok:false,status:401,json:async()=>({error:{message:'会话已失效，请刷新页面或重新输入访问口令。'}}),headers:{get:()=>null}};}
    if(u==='/api/session/password'){calls.push('POST /api/session/password');return {ok:false,status:401,json:async()=>({error:{message:'用户名或口令不正确。'}}),headers:{get:()=>null}};}
    if(u==='/api/session'&&m==='POST'){calls.push('POST /api/session');mode='ok';return {ok:true,status:200,json:async()=>({authenticated:true,shared:true,login:'password',csrf_token:'csrf2',username:'admin',role:'admin',claimable_owners:[]}),headers:{get:()=>null}};}
    return origFetch(url,opts);};
  // 改口令 with a wrong current password: the dialog says so, the session stays
  await step('change password',()=>fire(g('change-password'),'click'));
  const fields=walk(g('modal-body'),e=>e.tagName==='INPUT');fields[0].value='x';fields[1].value='new-password-1';fields[2].value='new-password-1';
  await step('save password',()=>fire(byText(g('modal-body'),'保存新口令'),'click'));
  await wait(30);
  const password={relogin:g('relogin-dialog').open,status:walk(g('modal-body'),e=>e.tagName==='P').map(e=>e._text).find(t=>t&&t.includes('口令'))||null};
  g('modal').close();
  // the session expires: the next request answers 401
  mode='expired';
  await step('refresh',()=>fire(g('refresh-jobs'),'click'));
  await wait(40);
  const expired={relogin:g('relogin-dialog').open,workspace:g('workspace').hidden,inDialog:g('relogin-body').children.includes(g('login-panel')),
    panel:g('login-panel').hidden,note:g('relogin-note').hidden,conn:g('connection').textContent,warn:g('connection').className,
    hash:location.hash,detail:Boolean(byText(g('detail'),'打开三维报告')),focus:focused[focused.length-1]};
  const pollsBefore=calls.filter(c=>c==='GET /api/jobs/j1').length;
  g('login-password').value='sandbox';
  await step('login again',()=>fire(g('login-form'),'submit'));
  await wait(80);
  const resumed={relogin:g('relogin-dialog').open,note:g('relogin-note').hidden,workspace:g('workspace').hidden,hash:location.hash,
    polled:calls.filter(c=>c==='GET /api/jobs/j1').length>pollsBefore,notice:(byClass(g('notice'),'notice-text')[0]||{})._text||null,
    conn:g('connection').textContent,detail:Boolean(byText(g('detail'),'打开三维报告')),remembered:localStorage.getItem('wss-login-user')};
  console.log(JSON.stringify({errors,missing:[...new Set(missing)],password,expired,resumed}));
  process.exit(0);
})();
"""
    out = _run(_bare_program(tail, extra))
    assert out["errors"] == [] and out["missing"] == [], out
    assert out["password"]["relogin"] is False and out["password"]["status"] == "当前口令不正确（区分大小写），请重新输入。"
    e = out["expired"]
    assert e["relogin"] is True and e["workspace"] is False and e["inDialog"] is True and e["panel"] is False and e["note"] is False
    assert e["conn"] == "登录已过期 · 请重新登录" and "warn" in e["warn"] and e["hash"] == "#job=j1" and e["detail"] is True
    assert e["focus"] == "login-password"          # user name remembered from the session, the password is next
    r = out["resumed"]
    assert r["relogin"] is False and r["note"] is True and r["workspace"] is False and r["hash"] == "#job=j1" and r["polled"] is True
    assert r["notice"] and r["notice"].startswith("已重新登录") and r["conn"] == "共享服务 · 已连接 · admin" and r["detail"] is True
    assert r["remembered"] == "admin"


def test_v015_stream_drop_version_change_upload_checks_cancel_guide_and_focus_return():
    extra = _FOCUS + r"""
const sources=[];global.EventSource=class{constructor(url){calls.push('SSE '+url);this.url=url;this.readyState=1;sources.push(this);}addEventListener(n,f){(this.l=this.l||{})[n]=f;}close(){this.readyState=2;}static get CLOSED(){return 2;}};
canned['/api/jobs/j5'].job.timing={confirmation_s:688222};
canned['/api/session'].max_upload_bytes=128*1024*1024;
"""
    tail = r"""
(async()=>{
  const g=id=>document.getElementById(id);
  await wait(80);
  // first-use guide on the overview and in 设置
  const guide={overview:byClass(g('detail'),'overview-guide').length,steps:walk(byClass(g('detail'),'guide-steps')[0],e=>e.tagName==='LI').length};
  await step('settings',()=>fire(g('settings-button'),'click'));
  guide.menu=texts(g('action-menu')).includes('三步上手与快捷键');
  await step('guide item',()=>fire(byText(g('action-menu'),'三步上手与快捷键'),'click'));
  guide.dialog=byClass(g('modal-body'),'guide-steps').length;
  // focus goes back to the control that opened the dialog (设置 got focus back from the menu)
  g('modal').close(); await step('close modal',()=>fire(g('modal'),'close'));
  const focusBack=focused[focused.length-1];
  // live stream drops → amber sentence; reopens → connected again
  const owner=sources.find(s=>s.url==='/api/events');
  owner.onerror();
  const drop={text:g('connection').textContent,cls:g('connection').className};
  owner.onopen();
  const back={text:g('connection').textContent,cls:g('connection').className};
  // the service reports a new version → 「服务已更新」 with a reload button
  canned['/api/health'].version='0.15.0';
  await step('health',()=>fire(g('connection'),'mouseenter'));
  await wait(30);
  const version={text:(byClass(g('notice'),'notice-text')[0]||{})._text||null,reload:Boolean(byText(g('notice'),'刷新页面'))};
  // upload: per-file reasons, identifier errors block the submit, only valid files are sent
  g('stl-file').files=[{name:'CASE_A.stl',size:2*1048576},{name:'notes.txt',size:10},{name:'EMPTY.stl',size:0}];
  await step('files',()=>fire(g('stl-file'),'change'));
  const rows=byClass(g('upload-files'),'upload-file-row').map(r=>[String(r.className),(byClass(r,'upload-file-state')[0]||{})._text,(byClass(r,'upload-file-size')[0]||{})._text]);
  const summary=(byClass(g('upload-files'),'upload-summary')[0]||{})._text||null;
  g('case-id').value='CASE\t1';
  await step('case id',()=>fire(g('case-id'),'input'));
  const issue={hidden:g('case-id-issue').hidden,text:g('case-id-issue').textContent,invalid:g('case-id').attrs['aria-invalid']};
  const posts=()=>calls.filter(c=>c==='POST /api/jobs'||c==='POST /api/jobs/batch').length;
  await step('submit blocked',()=>fire(g('upload-form'),'submit'));
  await wait(20);
  const blocked={posts:posts(),text:g('upload-progress-text').textContent};
  g('case-id').value='CASE-1'; await step('case id ok',()=>fire(g('case-id'),'input'));
  g('patient-id').value='张三'; await step('patient',()=>fire(g('patient-id'),'input'));
  const warn={hidden:g('patient-id-issue').hidden,cls:g('patient-id-issue').className,text:g('patient-id-issue').textContent};
  g('patient-id').value='';await step('patient clear',()=>fire(g('patient-id'),'input'));
  let sent=null;const origFetch=global.fetch;
  global.fetch=async(url,opts)=>{if(String(url)==='/api/jobs'&&opts&&opts.method==='POST'){calls.push('POST /api/jobs');sent=opts.body.getAll('stl').length;return {ok:true,status:200,json:async()=>({job:{id:'j1'}}),headers:{get:()=>null}};}return origFetch(url,opts);};
  await step('submit',()=>fire(g('upload-form'),'submit'));
  await wait(80);
  global.fetch=origFetch;
  const upload={posts:posts(),sent,notice:(byClass(g('notice'),'notice-text')[0]||{})._text||null};
  // a very long wait reads in days / hours
  await step('open j5',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j5')[0],'click'));
  await wait(40);
  const wait5=g('job-live-detail').textContent;
  // cancelling a running task asks first
  await step('open j3',()=>fire(walk(g('jobs-list'),e=>e.dataset&&e.dataset.jobId==='j3')[0],'click'));
  await wait(40);
  await step('cancel',()=>fire(byText(g('detail'),'取消任务'),'click'));
  await wait(10);
  const ask={open:g('modal').open,keep:Boolean(byText(g('modal-body'),'不取消，继续')),early:calls.includes('POST /api/jobs/j3/cancel')};
  canned['/api/jobs/j3/cancel']={job:{id:'j3',status:'cancelled',version:2}};
  await step('confirm cancel',()=>fire(byText(g('modal-body'),'取消任务'),'click'));
  await wait(40);
  ask.sent=calls.includes('POST /api/jobs/j3/cancel');
  console.log(JSON.stringify({errors,missing:[...new Set(missing)],guide,focusBack,drop,back,version,rows,summary,issue,blocked,warn,upload,wait5,ask}));
  process.exit(0);
})();
"""
    out = _run(_bare_program(tail, extra))
    assert out["errors"] == [] and out["missing"] == [], out
    assert out["guide"] == {"overview": 1, "steps": 3, "menu": True, "dialog": 1}
    assert out["focusBack"] == "settings-button"
    assert out["drop"] == {"text": "实时更新中断 · 正在重连…", "cls": "connection warn"}
    assert out["back"] == {"text": "本机服务 · 已连接", "cls": "connection online"}
    assert out["version"]["text"] and "服务已更新到 v0.15.0" in out["version"]["text"] and out["version"]["reload"] is True
    rows = out["rows"]
    assert rows[0] == ["upload-file-row", "待上传", "2.0 MB"]
    assert rows[1][0].endswith("invalid") and rows[1][1].startswith("不是 .stl 文件") and rows[2][1].startswith("文件是空的")
    assert out["summary"] == "2 个文件不会上传（原因见上），将只上传其余 1 个。"
    assert out["issue"]["hidden"] is False and "不可见字符" in out["issue"]["text"] and out["issue"]["invalid"] == "true"
    assert out["blocked"]["posts"] == 0 and "不可见字符" in out["blocked"]["text"]
    assert out["warn"]["hidden"] is False and out["warn"]["cls"] == "field-issue warn" and "真实姓名" in out["warn"]["text"]
    assert out["upload"]["posts"] == 1 and out["upload"]["sent"] == 1 and "2 个文件未上传" in (out["upload"]["notice"] or "")
    assert "已等待 7 天 23 小时" in out["wait5"]
    assert out["ask"] == {"open": True, "keep": True, "early": False, "sent": True}


def test_v015_wait_text_upload_checks_chunks_and_identifier_hints():
    out = _node("""
      const check=WB.checkUploadFiles([{name:'a.stl',size:10},{name:'b.txt',size:5},{name:'c.STL',size:0},{name:'A.STL',size:3},{name:'big.stl',size:200*1048576}]);
      console.log(JSON.stringify({
        waits:[null,30,90,3599,3600,7260,86400,90000,688222].map(WB.waitText),
        rows:check.rows.map(r=>[r.ok,r.reason||r.warn||'']),valid:check.valid.map(f=>f.name),invalid:check.invalid,
        small:WB.checkUploadFiles([{name:'a.stl',size:10}],{maxBytes:5}).rows[0].reason,
        chunks:[WB.uploadChunks([{size:1},{size:1},{size:1}],{maxFiles:2}).map(c=>c.length),WB.uploadChunks([{size:150},{size:150},{size:10}],{maxBytes:200}).map(c=>c.length),
                WB.uploadChunks([{size:500},{size:1}],{maxBytes:200}).map(c=>c.length),WB.uploadChunks([]).length],
        ids:[WB.identifierIssue(''),WB.identifierIssue('P-001'),WB.identifierIssue('CASE\\t1',{label:'病例编号'}),WB.identifierIssue('x'.repeat(81),{label:'患者编号',max:80}),WB.identifierIssue(' 张三 ')],
        bytes:[0,1023,2048,5*1048576,3*1073741824].map(WB.formatBytes)}));
    """)
    assert out["waits"] == ["—", "30 秒", "1 分 30 秒", "59 分 59 秒", "1 小时", "2 小时 1 分", "1 天", "1 天 1 小时", "7 天 23 小时"]
    assert [ok for ok, _ in out["rows"]] == [True, False, False, True, False]
    assert out["rows"][1][1].startswith("不是 .stl") and out["rows"][2][1].startswith("文件是空的") and "同名" in out["rows"][3][1] and "上限 128 MB" in out["rows"][4][1]
    assert out["valid"] == ["a.stl", "A.STL"] and out["invalid"] == 3 and "上限 5 B" in out["small"]
    assert out["chunks"] == [[2, 1], [1, 2], [1, 1], 0]
    ids = out["ids"]
    assert ids[0] is None and ids[1] is None and ids[2]["level"] == "error" and ids[2]["text"].startswith("病例编号含有")
    assert ids[3] == {"level": "error", "text": "患者编号最多 80 个字符（当前 81 个）。"} and ids[4]["level"] == "warn"
    assert out["bytes"] == ["0 B", "1023 B", "2.0 KB", "5.0 MB", "3.0 GB"]
