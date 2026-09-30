"""End-to-end combined request (M1 three-head + PF6/VF6 volume) through the real JobManager + pipeline, sandboxed."""
import json, sys, time
from pathlib import Path
code_root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(code_root))
stl, mapping_src, out_json = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
confirm_delay = float(sys.argv[5])
import wss_deploy
from wss_deploy.jobs import JobManager
from wss_deploy.registry import ReleaseRegistry
assert Path(wss_deploy.__file__).resolve().is_relative_to(code_root), wss_deploy.__file__
MAIN, COMP = "M1_3head_3seed_20260922", "PF6_VF6_peak_3seed_20260920"
registry = ReleaseRegistry(Path("/public/newhome/cy/Digital_twin/GNN/outputs/wss_deploy_release"), device="cuda")
for rid in (MAIN, COMP):                       # resident + warmed, as the service preloads them
    registry.load(rid).warm_up()
sandbox = out_json.with_suffix(".jobs"); sandbox.mkdir(parents=True, exist_ok=True)
manager = JobManager(sandbox, registry=registry)
manager.start()
mapping = json.loads((mapping_src / "job.json").read_text())["mapping"]
t0 = time.time()
job = manager.create("owner", content=stl.read_bytes(), filename=stl.name, case_id="E2E", release_id=MAIN,
                     companion_release_ids=COMP)
marks = {}
def status(jid):
    return manager.get(jid, "owner")
while True:
    s = status(job["id"])
    if s["status"] in {"awaiting_confirmation", "failed", "done", "queued"} and s.get("stage") == "A" and s["status"] == "awaiting_confirmation":
        break
    if s["status"] == "failed": raise SystemExit(json.dumps(s.get("error")))
    time.sleep(0.05)
marks["awaiting"] = time.time() - t0
time.sleep(confirm_delay)
s = status(job["id"])
manager.confirm(job["id"], "owner", {"version": s["version"], "stage": "A", "mapping": mapping, "acknowledged": True})
marks["confirmed"] = time.time() - t0
comp_id = status(job["id"])["companions"][0]["job_id"]
done = {}
while len(done) < 2:
    for jid in (job["id"], comp_id):
        s = status(jid)
        if s["status"] in {"done", "failed"} and jid not in done:
            done[jid] = time.time() - t0
            if s["status"] == "failed": raise SystemExit(json.dumps(s.get("error"), ensure_ascii=False))
    time.sleep(0.05)
marks["primary_done"], marks["companion_done"] = done[job["id"]], done[comp_id]
def summary(jid):
    return json.loads((sandbox / jid / "summary.json").read_text())
out = {"code_root": str(code_root), "stl": stl.name, "confirm_delay_s": confirm_delay, "marks_s": marks,
       "primary": {"timing_s": summary(job["id"]).get("timing_s"), "geometry_cache": summary(job["id"]).get("geometry_cache"),
                   "precompute": manager.jobs[job["id"]].get("precompute")},
       "companion": {"timing_s": summary(comp_id).get("timing_s"), "geometry_cache": summary(comp_id).get("geometry_cache")}}
manager.close()
out_json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({"marks": marks, "comp_cache": out["companion"]["geometry_cache"]}, ensure_ascii=False))
