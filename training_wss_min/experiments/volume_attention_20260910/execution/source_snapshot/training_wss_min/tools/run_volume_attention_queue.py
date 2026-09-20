"""Slurm queue with source gates, phase barriers and separately saved predictions."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from training_wss_min.tools.volume_attention_common import CONFIGS, EXP, NAME, ROOT, RUNS, fingerprints, manifest, reference_dir, save_json, sha256, stamp


def verify_data_inputs():
    frozen=json.loads((EXP/"runtime_preflight.json").read_text())["data"]
    for filename,expected in frozen.items():
        path=Path(filename)
        if "bytes" in expected:
            stat=path.stat()
            if stat.st_size!=expected["bytes"] or stat.st_mtime_ns!=expected["mtime_ns"]:
                raise RuntimeError(f"Frozen data changed after preflight: {path}")
        elif sha256(path)!=expected["sha256"]:
            raise RuntimeError(f"Frozen split/statistics changed after preflight: {path}")
    return dict(passed=True,files=len(frozen),check="data size/mtime; split/statistics SHA256; full data SHA256 captured in runtime preflight")


def compare_reference(run):
    checks = []
    for ck in ("best", "last"):
        original = json.loads((run/"original_eval"/f"ckpt_{ck}"/"metrics.json").read_text())["test"]
        actual = json.loads((run/"eval"/f"ckpt_{ck}"/"metrics.json").read_text())["test"]
        def visit(old, new, path):
            if isinstance(old, dict):
                if not isinstance(new, dict):
                    checks.append(dict(checkpoint=ck,key=path,passed=False,error="missing dictionary"));return
                for k,v in old.items():
                    if k not in new:
                        checks.append(dict(checkpoint=ck,key=path+"."+k,passed=False,error="missing key"))
                    else:visit(v,new[k],path+"."+k)
            elif isinstance(old,(float,int)) and not isinstance(old,bool):
                passed = isinstance(new,(float,int)) and ((math.isnan(old) and math.isnan(new)) if math.isnan(old) else math.isclose(old,new,rel_tol=2e-5,abs_tol=2e-6))
                checks.append(dict(checkpoint=ck,key=path,original=old if math.isfinite(old) else None,
                                   actual=new if isinstance(new,(float,int)) and math.isfinite(new) else None,passed=passed))
        def core(old,new,prefix):
            for section in ("field","field_casebalanced","aggregate","group_casebalanced"):
                if section in old:visit(old[section],new.get(section),prefix+section)
            if "per_case" in old:
                oc,nc=old["per_case"],new.get("per_case",{})
                checks.append(dict(checkpoint=ck,key=prefix+"case_ids",passed=set(oc)==set(nc)))
                for case,entry in oc.items():
                    visit(entry["overall"],nc.get(case,{}).get("overall"),prefix+"per_case."+case+".overall")
        core(original,actual,"")
        if "normalized" in original:core(original["normalized"],actual.get("normalized",{}),"normalized.")
        if "vector" in original:visit(original["vector"],actual.get("vector"),"vector")
        for group,entry in original.get("query_groups",{}).items():
            new=actual.get("query_groups",{}).get(group,{})
            core(entry,new,"query_groups."+group+".")
            if "normalized" in entry:core(entry["normalized"],new.get("normalized",{}),"query_groups."+group+".normalized.")
        if original["aggregate"]["n_cases"] != actual["aggregate"]["n_cases"]:
            raise ValueError("Historical reference case count changed")
    return dict(passed=bool(checks) and all(c["passed"] for c in checks),checks=checks)


def make_gate(smoke_path):
    hashes = fingerprints()
    runtime=json.loads((EXP/"runtime_preflight.json").read_text())
    tests=json.loads((EXP/"tests.json").read_text())
    smoke=json.loads(Path(smoke_path).read_text())
    references=json.loads((EXP/"reference_reproduction.json").read_text())
    for name, result in (("runtime",runtime),("tests",tests),("references",references)):
        if not result.get("passed") or result["source_sha256"] != hashes:
            raise ValueError(f"{name} gate missing/failed/stale")
    if smoke["status"] != "complete" or smoke["source_sha256"] != hashes or smoke["source_changed"]:
        raise ValueError("Smoke gate failed or stale")
    if set(smoke["arms"]) != {a["id"] for a in manifest()["arms"]}:
        raise ValueError("Smoke must cover all 26 arms")
    data_check=verify_data_inputs()
    save_json(EXP/"training_gate.json",dict(passed=True,source_sha256=hashes,timestamp=stamp(),data_check=data_check,
              smoke_path=str(smoke_path),runtime_job=runtime["job_id"],reference_job=references["job_id"]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("smoke","references","formal"),default="formal")
    ap.add_argument("--slots-per-gpu",type=int,default=2)
    ap.add_argument("--min-free-mib",type=int,default=6500)
    ap.add_argument("--make-gate",type=Path)
    args=ap.parse_args()
    if args.make_gate:
        make_gate(args.make_gate);return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit GPU queue through Slurm")
    gpus=[x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
    if not gpus or args.slots_per_gpu<1:
        raise ValueError("No allocated GPUs or invalid slot count")
    hashes=fingerprints();job=os.environ["SLURM_JOB_ID"]
    if args.mode=="formal":
        gate=json.loads((EXP/"training_gate.json").read_text())
        if not gate.get("passed") or gate["source_sha256"]!=hashes:
            raise RuntimeError("Formal training requires passing matching gate")
        verify_data_inputs()
    folder=EXP/(f"smoke_{job}" if args.mode=="smoke" else "references_execution" if args.mode=="references" else "execution")
    state_path=folder/"queue_status.json"
    if state_path.exists():
        raise RuntimeError(f"Queue state already exists; refusing overwrite: {state_path}")
    folder.mkdir(parents=True,exist_ok=True)
    snapshot=folder/"source_snapshot"
    for relative in hashes:
        dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,dest)
    arms=manifest()["arms"] if args.mode!="references" else [dict(id=f"R5{k}",phase=0,run_dir=str(reference_dir(k))) for k in ("P","V")]
    state=dict(mode=args.mode,job_id=job,started_at=stamp(),status="running",gpus=gpus,
               slots_per_gpu=args.slots_per_gpu,source_sha256=hashes,arms={})
    stages=["train"] if args.mode=="smoke" else ["eval_best","diag_best","eval_last","diag_last"] if args.mode=="references" else ["train","eval_best","diag_best","eval_last","diag_last"]
    for arm in arms:
        aid=arm["id"]
        rec=dict(phase=arm["phase"],status="pending",stages={})
        if args.mode=="references":
            rec["run_dir"]=arm["run_dir"]
        else:
            path=CONFIGS/arm["config"];cfg=json.loads(path.read_text())
            if args.mode=="smoke":
                cfg=copy.deepcopy(cfg);cfg["name"]=f"{NAME}/smoke_{job}/{aid}_s1234"
                cfg["train"].update(epochs=2,min_epoch=2,eval_every=2)
                path=folder/"configs"/arm["config"];save_json(path,cfg)
            rec.update(config=str(path),run_dir=str(RUNS/cfg["name"]))
            if Path(rec["run_dir"]).exists():
                raise RuntimeError(f"Refusing existing training run: {rec['run_dir']}")
        state["arms"][aid]=rec
    save_json(state_path,state)
    pending=list(state["arms"]);active={};stopped=False

    def stop(sig,frame):
        nonlocal stopped
        stopped=True
        for task in active.values():
            try:os.killpg(task["proc"].pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

    def launch(aid,gpu,stage):
        if fingerprints()!=hashes:
            raise RuntimeError("Source/config changed before launching next stage")
        rec=state["arms"][aid]
        if stage=="train":
            cmd=[sys.executable,"-u","-m","training_wss_min.train","--config",rec["config"]]
        elif stage.startswith("eval_"):
            cmd=[sys.executable,"-u","-m","training_wss_min.evaluate","--run-dir",rec["run_dir"],
                 "--partitions","test","--allow-test","--checkpoint",stage[5:],"--no-plots","--save-predictions"]
        else:
            cmd=[sys.executable,"-u","-m","training_wss_min.tools.volume_attention_diagnostics",
                 "--run-dir",rec["run_dir"],"--ckpt",stage[5:]]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        logfile=folder/f"{aid}_{stage}.log";handle=logfile.open("w")
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        rec.update(status=stage,gpu=gpu,pid=proc.pid)
        rec["stages"][stage]=dict(started_at=stamp(),command=cmd,log=str(logfile))
        active[aid]=dict(proc=proc,handle=handle,gpu=gpu,stage=stage,start=time.monotonic())
        print(stamp(),aid,stage,"GPU",gpu,"pid",proc.pid,flush=True);save_json(state_path,state)

    last_launch=0.
    try:
        while (pending or active) and not stopped:
            for aid,task in list(active.items()):
                code=task["proc"].poll()
                if code is None:continue
                task["handle"].close();rec=state["arms"][aid]
                rec["stages"][task["stage"]].update(returncode=code,ended_at=stamp(),elapsed_seconds=time.monotonic()-task["start"])
                del active[aid]
                if code:
                    rec.update(status="failed",failed_stage=task["stage"])
                    print(stamp(),aid,"FAILED",task["stage"],code,flush=True)
                else:
                    index=stages.index(task["stage"])+1
                    if index<len(stages):launch(aid,task["gpu"],stages[index])
                    else:rec["status"]="complete"
                save_json(state_path,state)
            # Explicit phase barrier: all phase1 then phase2 then joint.
            incomplete=[r["phase"] for r in state["arms"].values() if r["status"] not in ("complete","failed")]
            phase=min(incomplete) if incomplete else None
            ready=[aid for aid in pending if state["arms"][aid]["phase"]==phase]
            if ready and time.monotonic()-last_launch>12:
                candidates=[]
                for gpu in gpus:
                    count=sum(t["gpu"]==gpu for t in active.values())
                    if count>=args.slots_per_gpu:continue
                    free=int(subprocess.check_output(["nvidia-smi","-i",gpu,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                    if free>=args.min_free_mib:candidates.append((count,-free,gpu))
                if candidates:
                    gpu=min(candidates)[2];aid=ready[0];pending.remove(aid)
                    launch(aid,gpu,stages[0]);last_launch=time.monotonic()
            time.sleep(3)
    except BaseException:
        stop(None,None)
        raise
    finally:
        if stopped:
            for aid,task in active.items():
                try:task["proc"].wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(task["proc"].pid,signal.SIGKILL)
                task["handle"].close();state["arms"][aid]["status"]="interrupted"
        state["source_sha256_end"]=fingerprints();state["source_changed"]=hashes!=state["source_sha256_end"]
        if args.mode=="formal":
            try:state["data_check_end"]=verify_data_inputs()
            except (OSError,RuntimeError) as exc:
                state["data_check_end"]=dict(passed=False,error=str(exc))
        success=all(r["status"]=="complete" for r in state["arms"].values()) and not state["source_changed"]
        success=success and state.get("data_check_end",{"passed":True})["passed"]
        state.update(status="complete" if success else "incomplete",ended_at=stamp());save_json(state_path,state)
    if success and args.mode=="references":
        checks={a["id"]:compare_reference(Path(a["run_dir"])) for a in arms}
        passed=all(x["passed"] for x in checks.values())
        save_json(EXP/"reference_reproduction.json",dict(passed=passed,job_id=job,source_sha256=hashes,references=checks))
        success=passed
    return 0 if success else 1


if __name__=="__main__":
    raise SystemExit(main())
