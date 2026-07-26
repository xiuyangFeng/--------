#!/usr/bin/env python3
"""Submit the xyz-only LocalGeoPE control."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training_wss_min.config import ExpConfig
PREFLIGHT=ROOT/"training_wss_min/preflight"; MANIFEST=PREFLIGHT/"xyz_local_geope_20260725_prepared.json"; STATIC=PREFLIGHT/"xyz_local_geope_20260725_static_audit.json"; OUTPUT=PREFLIGHT/"xyz_local_geope_20260725_submission.json"; SBATCH=Path("/public/slurm/bin/sbatch"); GATE=ROOT/"training_wss_min/cluster/preflight_xyz_local_geope.slurm"; RUN=ROOT/"training_wss_min/cluster/run_xyz_local_geope.slurm"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def submit(*args:str)->str:
 p=subprocess.run([str(SBATCH),"--parsable",*args],cwd=ROOT,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.strip() or "sbatch failed")
 return p.stdout.strip().split(";",1)[0]
def main()->None:
 ap=argparse.ArgumentParser();ap.add_argument("--dry-run",action="store_true");a=ap.parse_args()
 if OUTPUT.exists():raise FileExistsError(f"refusing duplicate submission: {OUTPUT}")
 m=json.loads(MANIFEST.read_text(encoding="utf-8")); rows=m.get("configs",[]); s=json.loads(STATIC.read_text(encoding="utf-8"))
 if m.get("status")!="prepared" or len(rows)!=1 or s.get("status")!="passed" or s.get("manifest_sha256")!=sha(MANIFEST):raise RuntimeError("prepared/static contract failed")
 r=rows[0];cfg=Path(r["config"])
 if sha(cfg)!=r["sha256"] or ExpConfig.from_json(cfg).run_dir.exists():raise RuntimeError("config drift or run collision")
 out=OUTPUT.with_suffix(".dryrun.json") if a.dry_run else OUTPUT; payload={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"status":"submitting","manifest":str(MANIFEST.resolve()),"manifest_sha256":sha(MANIFEST),"static_audit":str(STATIC.resolve()),"policy":"S2 PointNeXt-R xyz-only parent + 4D relative-xyz/distance LocalGeoPE; gate then train/best-last test","jobs":[]}
 try:
  if a.dry_run:payload["status"]="dry_run"
  else:
   e=f"ALL,MATRIX_MANIFEST={MANIFEST}";g=submit("--job-name=xyzgeo_pf",f"--export={e}",str(GATE));j=submit("--job-name=xyzgeo",f"--export={e}",f"--dependency=afterok:{g}",str(RUN));payload["jobs"]=[{"role":"formal_gate","job_id":g},{"role":"training_and_best_last_eval","job_id":j,"dependency":f"afterok:{g}"}];payload["status"]="submitted";payload["submitted_at"]=datetime.now(timezone.utc).isoformat()
  out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 except Exception as exc:
  payload["status"]="partial_failed" if payload["jobs"] else "failed";payload["error"]=f"{type(exc).__name__}: {exc}";out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");raise
 print(json.dumps({"status":payload["status"],"jobs":payload["jobs"],"output":str(out)},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
