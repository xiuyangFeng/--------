#!/usr/bin/env python3
"""Submit the core and cached-FPS branches of the frozen 17-run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
SBATCH = Path("/public/slurm/bin/sbatch")
MANIFEST = REPO / "training_wss_min/preflight/sa_grouping_single_seed_prepared.json"
OUTPUT = REPO / "training_wss_min/preflight/sa_grouping_single_seed_submission.json"
CACHE_SCRIPT = REPO / "training_wss_min/cluster/build_sa_grouping_fps_cache.slurm"
GATE_SCRIPT = REPO / "training_wss_min/cluster/preflight_sa_grouping_single_seed.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_sa_grouping_single_seed.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(*args: str) -> str:
    proc = subprocess.run([str(SBATCH), "--parsable", *args], cwd=REPO, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "sbatch failed")
    return proc.stdout.strip().split(";", 1)[0]


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing duplicate submission manifest: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    core = [r for r in rows if not r["requires_fps_cache"]]
    fps = [r for r in rows if r["requires_fps_cache"]]
    if manifest.get("status") != "prepared" or len(rows) != 17 or len(core) != 10 or len(fps) != 7:
        raise RuntimeError("not the exact frozen 10-core + 7-FPS matrix")
    if {r["seed"] for r in rows} != {1234}:
        raise RuntimeError("matrix contains a non-1234 seed")
    for row in rows:
        path = Path(row["config"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"missing or drifted config: {path}")
        if ExpConfig.from_json(path).run_dir.exists():
            raise FileExistsError(f"run dir already exists: {row['run_dir']}")
    for script in (CACHE_SCRIPT, GATE_SCRIPT, TRAIN_SCRIPT):
        if not script.is_file(): raise FileNotFoundError(script)
    payload = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting", "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest), "jobs": [],
        "policy": "17 new jobs, seed1234 only, exact split106/0/27",
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            core_gate = submit("--job-name=sa_core_pf", "--export=ALL,MATRIX_SUBSET=core", str(GATE_SCRIPT))
            payload["jobs"].append({"role": "core_geometry_and_gpu_gate", "job_id": core_gate})
            core_train = submit("--job-name=sa_core", "--export=ALL,MATRIX_SUBSET=core",
                                f"--dependency=afterok:{core_gate}", "--array=0-9%4", str(TRAIN_SCRIPT))
            payload["jobs"].append({"role": "core_training", "job_id": core_train,
                                    "array": "0-9%4", "dependency": f"afterok:{core_gate}"})
            cache = submit("--job-name=sa_fps_cache", "--array=0-132%7", str(CACHE_SCRIPT))
            payload["jobs"].append({"role": "offline_fps5000_pool8_cache", "job_id": cache, "array": "0-132%7"})
            fps_gate = submit("--job-name=sa_fps_pf", "--export=ALL,MATRIX_SUBSET=fps",
                              f"--dependency=afterok:{cache}", str(GATE_SCRIPT))
            payload["jobs"].append({"role": "fps_geometry_and_gpu_gate", "job_id": fps_gate,
                                    "dependency": f"afterok:{cache}"})
            fps_train = submit("--job-name=sa_fps", "--export=ALL,MATRIX_SUBSET=fps",
                               f"--dependency=afterok:{fps_gate}", "--array=0-6%4", str(TRAIN_SCRIPT))
            payload["jobs"].append({"role": "fps_training", "job_id": fps_train,
                                    "array": "0-6%4", "dependency": f"afterok:{fps_gate}"})
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(args.output, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(args.output, payload)
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"],
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
