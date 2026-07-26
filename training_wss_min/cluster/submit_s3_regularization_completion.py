#!/usr/bin/env python3
"""Submit the 20-arm single-seed S3 regularization completion matrix."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from training_wss_min.config import ExpConfig

PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_regularization_completion_20260725_prepared.json"
STATIC = PREFLIGHT / "s3_regularization_completion_20260725_static_audit.json"
OUTPUT = PREFLIGHT / "s3_regularization_completion_20260725_submission.json"
SBATCH = Path("/public/slurm/bin/sbatch")
GATE = ROOT / "training_wss_min/cluster/preflight_s3_regularization_completion.slurm"
RUN = ROOT / "training_wss_min/cluster/run_s3_regularization_completion.slurm"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def submit(*args: str) -> str:
    proc=subprocess.run([str(SBATCH),"--parsable",*args],cwd=ROOT,text=True,capture_output=True)
    if proc.returncode: raise RuntimeError(proc.stderr.strip() or "sbatch failed")
    return proc.stdout.strip().split(";",1)[0]
def dump(path: Path, payload: dict) -> None: path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
    if OUTPUT.exists(): raise FileExistsError(f"refusing duplicate submission: {OUTPUT}")
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8")); rows=manifest.get("configs",[])
    if manifest.get("status")!="prepared" or len(rows)!=20 or {row['seed'] for row in rows}!={1234}: raise RuntimeError("expected exact 20-arm seed1234 matrix")
    static=json.loads(STATIC.read_text(encoding="utf-8"))
    if static.get("status")!="passed" or static.get("manifest_sha256")!=sha(MANIFEST): raise RuntimeError("static audit missing, failed, or stale")
    for row in rows:
        cfg=Path(row["config"])
        if sha(cfg)!=row["sha256"] or ExpConfig.from_json(cfg).run_dir.exists(): raise RuntimeError(f"drift or run collision: {cfg}")
    out=OUTPUT.with_suffix(".dryrun.json") if args.dry_run else OUTPUT
    payload={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"status":"submitting","manifest":str(MANIFEST.resolve()),"manifest_sha256":sha(MANIFEST),"static_audit":str(STATIC.resolve()),"static_audit_sha256":sha(STATIC),"policy":"S3 seed1234; 8 missing single-rate arms + 12 pairwise 0.05/0.10 crosses; GPU gate then array 0-19%4","tasks":20,"array":"0-19%4","jobs":[]}
    try:
        if args.dry_run: payload["status"]="dry_run"
        else:
            export=f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate=submit("--job-name=s3regc_pf",f"--export={export}",str(GATE))
            array=submit("--job-name=s3regc",f"--export={export}",f"--dependency=afterok:{gate}","--array=0-19%4",str(RUN))
            payload["jobs"]=[{"role":"formal_gate","job_id":gate},{"role":"training_and_best_last_eval","job_id":array,"array":"0-19%4","dependency":f"afterok:{gate}"}]; payload["status"]="submitted"; payload["submitted_at"]=datetime.now(timezone.utc).isoformat()
        dump(out,payload)
    except Exception as exc:
        payload["status"]="partial_failed" if payload["jobs"] else "failed"; payload["error"]=f"{type(exc).__name__}: {exc}"; dump(out,payload); raise
    print(json.dumps({"status":payload["status"],"jobs":payload["jobs"],"output":str(out)},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
