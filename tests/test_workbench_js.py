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
        WB.computeTiming({compute_s:40}),WB.computeTiming(null,12.5),WB.computeTiming(undefined)];
      console.log(JSON.stringify({reasons,timings}));
    """)
    assert out["reasons"][:5] == ["任务不存在", "任务未完成", "尚未确认出口命名", "尚未确认出口命名", "尚未确认出口命名"]
    assert out["reasons"][5] == "缺少可复用的中心线结果"   # same precondition the service enforces
    assert out["reasons"][6] is None and out["reasons"][7] is None
    assert out["timings"][0] == {"attempt": 40, "total": 120, "split": True}
    assert out["timings"][1]["split"] is False and out["timings"][2]["split"] is False
    assert out["timings"][3] == {"attempt": None, "total": 12.5, "split": False}
    assert out["timings"][4] == {"attempt": None, "total": None, "split": False}


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
global.document={getElementById:id=>{const live=created.find(el=>el.id===id);if(live)return live;if(!ids.has(id))missing.push(id);return els[id]||(els[id]=new Element('div',id));},createElement:t=>{const el=new Element(t);created.push(el);return el;},createTextNode:t=>({nodeValue:String(t)}),querySelectorAll:()=>[],querySelector:()=>null,addEventListener(){},removeEventListener(){},hidden:false,title:'WSS',body:new Element('body'),documentElement:new Element('html')};
global.window=global;global.self=global;global.addEventListener=()=>{};global.removeEventListener=()=>{};global.focus=()=>{};global.open=()=>null;global.confirm=()=>true;global.prompt=()=>null;global.alert=()=>{};
global.localStorage={_m:new Map(),getItem(k){return this._m.has(k)?this._m.get(k):null;},setItem(k,v){this._m.set(k,String(v));},removeItem(k){this._m.delete(k);}};
global.location={hash:'',href:'http://localhost:8765/',origin:'http://localhost:8765',protocol:'http:',pathname:'/',search:''};
global.history={replaceState(){},pushState(){}};global.navigator={userAgent:'node'};global.performance={now:()=>0};
global.requestAnimationFrame=()=>0;global.cancelAnimationFrame=()=>{};global.matchMedia=()=>({matches:false,addEventListener(){}});
global.setInterval=()=>0;global.clearInterval=()=>{};
global.EventSource=class{constructor(url){calls.push('SSE '+url);this.readyState=1;}addEventListener(){}close(){}static get CLOSED(){return 2;}};
global.THREE={WebGLRenderer:class{constructor(){throw new Error('No WebGL');}}};
const calls=[];
const canned={
 '/api/session':{authenticated:true,shared:false,login:'none',csrf_token:'csrf',username:null,role:null,claimable_owners:[],max_upload_bytes:1},
 '/api/releases':{releases:[{id:'X5D_v51_5seed_20260916',release:'X5D_v51_5seed_20260916',models_count:5,contract:{protocol:'single_frame_wss'}},{id:'PF6_VF6_peak_3seed_20260920',release:'PF6_VF6_peak_3seed_20260920',models_count:6,contract:{protocol:'single_frame_volume'}}]},
 '/api/preferences':{preferences:{schema_version:'wss-deploy.preferences/v1',upload:{units:'mm',release_id:'X5D_v51_5seed_20260916',device:'cpu',seed_count:'all',threads:'',remember_patient:true,last_patient:{patient_id:'P-9',scan_label:'基线',scan_date:'',tags:'AAA'}},notifications:{enabled:true}}},
 '/api/jobs':{jobs:[{id:'j1',case_id:'A',status:'done',version:3,review:{status:'unreviewed'},model_release:{id:'X5D_v51_5seed_20260916'},family:'wall',created_at:'2026-09-21T10:00:00'},{id:'j2',case_id:'B',status:'done',version:2,review:{status:'reviewed',by:'r'},model_release:{id:'PF6_VF6_peak_3seed_20260920'},family:'volume',created_at:'2026-09-21T09:00:00'},{id:'j3',case_id:'C',status:'running',version:1,review:{status:'unreviewed'},created_at:'2026-09-21T08:00:00'}],total:3,page:1,page_size:20},
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
global.WssWorkbenchCore=require(CORE);global.WssBatchExport=require(BATCH);
require(APP);
const g=id=>els[id];
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const fire=(el,name)=>{ if(!el||!el.events[name])throw new Error('no '+name+' handler'); for(const f of [...el.events[name]]) f({preventDefault(){},stopPropagation(){},currentTarget:el,target:el,key:''}); };
const walk=(el,pred,out=[])=>{ if(!el||typeof el!=='object')return out; if(pred(el))out.push(el); for(const child of (el.children||[])) walk(child,pred,out); return out; };
const byText=(root,text)=>walk(root,e=>e._text===text)[0]||null;
const step=async(label,fn)=>{ try{ await fn(); }catch(e){ errors.push(label+': '+(e&&e.stack||e)); } };
(async()=>{
  await wait(60);
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
  await step('edit metadata',()=>fire(byText(g('detail'),'编辑信息'),'click'));
  await wait(20);
  const metadataForm=Boolean(byText(g('modal-body'),'保存信息'));
  await step('save metadata',()=>fire(byText(g('modal-body'),'保存信息'),'click'));
  await wait(60);
  // §15.4: multi-select one finished job and rerun it with another release
  await step('select mode',()=>fire(g('select-jobs'),'click'));
  await wait(30);
  await step('tick job',()=>{ const box=walk(g('jobs-list'),e=>e.tagName==='INPUT'&&String(e.attrs['aria-label']||'').startsWith('选择'))[0]; box.checked=true; fire(box,'change'); });
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
  console.log(JSON.stringify({errors,missing:[...new Set(missing)],calls:[...new Set(calls)],workspaceHidden:g('workspace').hidden,loginHidden:g('login-panel').hidden,
    jobs:g('jobs-list').children.length,cases:g('cases-list').children.length,trash:g('trash-list')?g('trash-list').children.length:null,
    cohort,cohortDiameter,caseChips,morphology,narrative,narrativeForm,lockedNarrative,timingText,metadataForm,rerunForm,
    units:g('upload-units').value,device:g('upload-device').value,patient:g('patient-id').value,connection:g('connection').textContent,title:document.title}));
})();
"""


def test_workbench_boots_against_stub_dom_and_canned_service(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node is required")
    program = ("const CORE=" + json.dumps(str(CORE)) + ";const BATCH=" + json.dumps(str(BATCH)) + ";const APP=" + json.dumps(str(STATIC_DIR / "app.js"))
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
    assert out["timingText"] and out["timingText"].startswith("本次 ") and "累计" in out["timingText"]
    # §15.2 / §15.4: both dialogs open and post through request() (CSRF-bearing)
    assert out["metadataForm"] is True and "POST /api/jobs/j1/metadata" in out["calls"]
    assert out["rerunForm"] is True and "POST /api/jobs/j1/rerun" in out["calls"]
