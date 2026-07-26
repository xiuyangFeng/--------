#!/usr/bin/env python3
"""Submit the frozen 17-run SA1-scale matrix (single preflight gate + one array)."""

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
MANIFEST = REPO / "training_wss_min/preflight/pointnetpp_sa1_scale_prepared.json"
OUTPUT = REPO / "training_wss_min/preflight/pointnetpp_sa1_scale_submission.json"
GATE_SCRIPT = REPO / "training_wss_min/cluster/preflight_pointnetpp_sa1_scale.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_pointnetpp_sa1_scale.slurm"

EXPECTED_FAMILIES = {"d1_fixed": 3, "d1_prop": 3, "d2_centers": 4, "d3_points": 3,
                     "d4_width": 2, "d5_stem": 1, "bridge": 1}


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
    parser.add_argument("--variant-b", action="store_true",
                        help="提交条件触发的 D5-B（独立 manifest、独立 tag、array=0-0）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest
    output_path = args.output
    tag = "sa1_scale"
    expected_jobs = 17
    array_spec = "0-16%4"
    if args.variant_b:
        manifest_path = REPO / "training_wss_min/preflight/pointnetpp_sa1_scale_variant_b_prepared.json"
        output_path = REPO / "training_wss_min/preflight/pointnetpp_sa1_scale_variant_b_submission.json"
        tag = "sa1_scale_vb"
        expected_jobs = 1
        array_spec = "0-0"
    if output_path.exists():
        raise FileExistsError(f"refusing duplicate submission manifest: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != expected_jobs:
        raise RuntimeError(f"not the exact frozen {expected_jobs}-run matrix: {len(rows)}")
    if not args.variant_b:
        families = {}
        for row in rows:
            families[row["family"]] = families.get(row["family"], 0) + 1
        if families != EXPECTED_FAMILIES:
            raise RuntimeError(f"unexpected family layout: {families}")
    if {r["seed"] for r in rows} != {1234}:
        raise RuntimeError("matrix contains a non-1234 seed")
    for row in rows:
        path = Path(row["config"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"missing or drifted config: {path}")
        if ExpConfig.from_json(path).run_dir.exists():
            raise FileExistsError(f"run dir already exists: {row['run_dir']}")
    for script in (GATE_SCRIPT, TRAIN_SCRIPT):
        if not script.is_file():
            raise FileNotFoundError(script)
    export = f"ALL,MATRIX_MANIFEST={manifest_path},MATRIX_TAG={tag}"
    payload = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting", "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha(manifest_path), "jobs": [],
        "policy": f"{expected_jobs} new jobs, seed1234 only, exact split106/0/27, tag={tag}",
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            gate = submit(f"--job-name={tag}_pf", f"--export={export}", str(GATE_SCRIPT))
            payload["jobs"].append({"role": "geometry_and_gpu_gate", "job_id": gate})
            train = submit(f"--job-name={tag}", f"--export={export}",
                           f"--dependency=afterok:{gate}", f"--array={array_spec}",
                           str(TRAIN_SCRIPT))
            payload["jobs"].append({"role": "training", "job_id": train,
                                    "array": array_spec, "dependency": f"afterok:{gate}"})
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(output_path, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(output_path, payload)
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"],
                      "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
