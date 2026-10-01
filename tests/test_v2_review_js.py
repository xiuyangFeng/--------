"""Node tests for the workspace v2 findings review, manual findings and annotations (second phase S4).

The documents are the classic ones (findings_review / annotations sidecars); these tests check the editing rules
(empty entries dropped, notes kept by 「其余确认」, manual ids unique among every finding, limits), the merged
findings list (manual findings in, rejected ones last and out of the short list) and the debounced saver.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
FILES = [STATIC_DIR / "report_common.js", V2 / "core_util.js", V2 / "ws_ui.js", V2 / "ws_overview.js", V2 / "ws_probe.js", V2 / "ws_review.js", V2 / "ws_annot.js"]


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the v2 review tests")
    prelude = "".join("require(" + json.dumps(str(f)) + ");\n" for f in FILES)
    prelude += "const ns=globalThis.WSSV2, RV=ns.review, AN=ns.annot, OV=ns.overview; const out=x=>console.log(JSON.stringify(x));\n"
    program = prelude + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout)


def test_scripts_bundled():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    for name in ("ws_review.js", "ws_annot.js"):
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
        assert bundle["scripts"].index("ws_probe.js") < bundle["scripts"].index(name) < bundle["scripts"].index("ws_shell.js")


def test_decisions_notes_and_confirm_rest():
    out = _node("""
      let d={items:{F1:{decision:'rejected',note:''},F2:{decision:null,note:'看一下'}},added:[]};
      const a=RV.setDecision(d,'F1',null), b=RV.setNote(d,'F2',''), c=RV.setNote(d,'F3','x'.repeat(600)), e=RV.setDecision(d,'F3','confirmed');
      const r=RV.confirmRest(d,['F2','F4']);
      out({a:a.items,b:b.items,c:c.items.F3.note.length,e:e.items.F3,r:r.items,orig:d.items,dec:RV.decisionOf(r,'F2'),note:RV.noteOf(r,'F2')});
    """)
    assert "F1" not in out["a"]                                          # neither decision nor note → dropped
    assert "F2" not in out["b"]
    assert out["c"] == 500
    assert out["e"] == {"decision": "confirmed", "note": ""}
    assert out["r"]["F2"] == {"decision": "confirmed", "note": "看一下"}   # the note stays
    assert out["r"]["F4"] == {"decision": "confirmed", "note": ""} and out["r"]["F1"]["decision"] == "rejected"
    assert out["orig"]["F2"]["decision"] is None                         # the input document is not modified
    assert out["dec"] == "confirmed" and out["note"] == "看一下"


def test_manual_findings():
    out = _node("""
      const result={manifest:{geometry:{centerline:{}}},has:()=>false,declared:()=>false,array:()=>null};
      const items=[{id:'F1'},{id:'M1'}], d={items:{},added:[]};
      const r1=RV.addManual(d,result,items,[1.23456,2,3],'  壁上有个凸起  ','attention');
      const r2=RV.addManual(r1.doc,result,items,[0,0,0],'第二个','info');
      const empty=RV.addManual(d,result,items,[0,0,0],'   ','note');
      let full={items:{},added:[]}; for(let i=0;i<30;i++)full.added.push({id:'M'+(i+1),xyz_mm:[0,0,0],text:'t'});
      const over=RV.addManual(full,result,[],[0,0,0],'x','note');
      const gone=RV.removeManual(Object.assign({},r2.doc,{items:{M2:{decision:'confirmed',note:''}}}),'M2');
      out({a:r1.item,b:r2.item,empty:empty.error||null,over:over.error||null,gone:gone.added.map(x=>x.id),goneItems:gone.items,mi:RV.manualItem(r1.item)});
    """)
    assert out["a"]["id"] == "M2" and out["a"]["xyz_mm"] == [1.235, 2, 3] and out["a"]["text"] == "壁上有个凸起"
    assert out["a"]["kind"] == "manual" and out["a"]["severity"] == "attention" and out["a"]["created_at"]
    assert out["b"]["id"] == "M3" and out["b"]["severity"] == "note"
    assert out["empty"] and out["over"] and "30" in out["over"]
    assert out["gone"] == ["M3"] and "M2" not in out["goneItems"]
    assert out["mi"]["manual"] and out["mi"]["label"] == "壁上有个凸起" and out["mi"]["extent_mm"] == 10


def test_findings_model_merges_manual_and_puts_rejected_last():
    out = _node("""
      const m={analysis:{findings:{items:[
          {id:'F1',kind:'high_wss_cluster',severity:'note',rank:1,value:5,units:'Pa'},
          {id:'F2',kind:'high_wss_cluster',severity:'note',rank:2,value:4,units:'Pa'},
          {id:'F3',kind:'max_diameter',severity:'info',rank:3,value:70,units:'mm'}],
        review:{items:{F1:{decision:'rejected',note:''},F3:{decision:'confirmed',note:''}},added:[{id:'M1',xyz_mm:[0,0,0],text:'人工',severity:'note'}],legacy:[{id:'F9'}]}}}};
      const fm=OV.findingsModel(m);
      out({items:fm.items.map(x=>x.id),top:fm.top.map(x=>x.id),undecided:fm.undecided.map(x=>x.id),total:fm.total,manual:fm.manual,rejected:fm.rejected,legacy:fm.legacy.length,
        kind:OV.kindLabel(fm.items.filter(x=>x.manual)[0]),value:OV.findingValue(fm.items.filter(x=>x.manual)[0])});
    """)
    assert out["items"] == ["F2", "F3", "M1", "F1"]
    assert out["top"] == ["M1", "F2", "F3"]                              # F2 stands for the kind once F1 is rejected; manual after 关注 (none here)
    assert out["undecided"] == ["F2"] and out["total"] == 4 and out["manual"] == 1 and out["rejected"] == 1 and out["legacy"] == 1
    assert out["kind"] == "人工" and out["value"] == "人工"


def test_saver_coalesces_and_sends_the_version():
    out = _node("""
      const calls=[]; let version=7, saved=null;
      const s=RV.saver({delay:20,version:()=>version,put:p=>{calls.push(JSON.parse(JSON.stringify(p)));return Promise.resolve({findings_review:{items:p.items,added:p.added,updated_by:'admin'},version:8});},
        onSaved:(doc,v)=>{saved=doc;version=v;}});
      s.save({items:{F1:{decision:'confirmed',note:''}},added:[]}); s.save({items:{F1:{decision:'rejected',note:''}},added:[]});
      await new Promise(r=>setTimeout(r,80));
      const errs=[]; const f=RV.saver({delay:0,put:()=>Promise.reject(Object.assign(new Error('x'),{status:409})),onError:e=>errs.push(e.status)});
      f.save({items:{},added:[]}); await new Promise(r=>setTimeout(r,30));
      out({calls,saved,version,errs});
    """)
    assert len(out["calls"]) == 1 and out["calls"][0]["version"] == 7
    assert out["calls"][0]["items"]["F1"]["decision"] == "rejected"
    assert out["saved"]["updated_by"] == "admin" and out["version"] == 8 and out["errs"] == [409]


def test_annotations():
    out = _node("""
      const result={manifest:{geometry:{centerline:{}}},has:()=>false,declared:()=>false,array:()=>null};
      const m={analysis:{annotations:{items:[{id:'A1',xyz_mm:[1,2,3],text:'旧的'},{id:'bad',xyz_mm:[1,2],text:'x'}]}}};
      const items=AN.itemsOf(m), r=AN.add(items,result,[4,5,6.00049],'  新的  ');
      const edited=AN.edit(r.items,'A1','改过'), emptied=AN.edit(r.items,'A2','  ');
      let many=[];for(let i=0;i<50;i++)many.push({id:'A'+(i+1),xyz_mm:[0,0,0],text:'t'});
      out({n:items.length,item:r.item,edited:edited.map(a=>a.text),emptied:emptied.map(a=>a.id),full:AN.add(many,result,[0,0,0],'x').error||null,none:AN.add([],result,[0,0,0],' ').error||null});
    """)
    assert out["n"] == 1
    assert out["item"]["id"] == "A2" and out["item"]["text"] == "新的" and out["item"]["xyz_mm"] == [4, 5, 6] and out["item"]["color"] == "#d97706"
    assert out["edited"] == ["改过", "新的"] and out["emptied"] == ["A1"]
    assert out["full"] and out["none"]
