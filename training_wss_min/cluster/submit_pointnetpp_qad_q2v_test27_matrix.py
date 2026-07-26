#!/usr/bin/env python3
"""Hash-gated cluster submission for the exact-Q2V QAD historical-test27 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training_wss_min.config import ExpConfig, input_dim
from training_wss_min.models import build_model


SBATCH = Path("/public/slurm/bin/sbatch")
PREPARED = REPO / "training_wss_min/preflight/pointnetpp_qad_q2v_test27_prepared.json"
PREFLIGHT = REPO / "training_wss_min/preflight/pointnetpp_qad_q2v_test27_gpu_preflight.json"
OUTPUT = REPO / "training_wss_min/preflight/pointnetpp_qad_q2v_test27_submission.json"
PREFLIGHT_SCRIPT = REPO / "training_wss_min/cluster/run_pointnetpp_qad_q2v_test27_preflight.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_pointnetpp_qad_q2v_test27_matrix.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sbatch(*args: str) -> str:
    result = subprocess.run([str(SBATCH), "--parsable", *args], cwd=REPO, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    return result.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SBATCH.is_file():
        raise FileNotFoundError(SBATCH)
    if OUTPUT.exists() and not args.dry_run:
        raise FileExistsError(f"refusing duplicate/ambiguous submission manifest: {OUTPUT}")
    prepared = json.loads(PREPARED.read_text(encoding="utf-8"))
    rows = prepared.get("configs", [])
    if prepared.get("status") != "prepared" or len(rows) != 5:
        raise RuntimeError("prepared manifest is not the exact five-config QAD/Q2V matrix")

    configs = []
    by_pair = {}
    for row in rows:
        path = Path(row["config"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"config missing or changed: {path}")
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise FileExistsError(f"run directory already exists: {cfg.run_dir}")
        split = json.loads(Path(cfg.data.split_path).read_text(encoding="utf-8"))
        split_counts = tuple(len(split[f"{part}_cases"]) for part in ("train", "val", "test"))
        if split_counts != (106, 0, 27):
            raise RuntimeError(f"split drift: {path}")
        if cfg.model.sa_nsample != (16, 16, 16) or cfg.model.width != 32:
            raise RuntimeError(f"model drift: {path}")
        if cfg.data.query_mode != "independent" or cfg.train.selection_rule != "train_loss":
            raise RuntimeError(f"Q2V protocol drift: {path}")
        model = build_model(cfg.model, input_dim(cfg))
        params = sum(parameter.numel() for parameter in model.parameters())
        enriched = {**row, "params": params, "run_dir": str(cfg.run_dir.resolve())}
        configs.append(enriched)
        by_pair[(cfg.train.seed, cfg.model.query_decoder)] = (path, cfg)

    anchor_cfg = ExpConfig.from_json(prepared["anchor"]["config"])
    base_params = sum(p.numel() for p in build_model(anchor_cfg.model, input_dim(anchor_cfg)).parameters())
    qad_params = next(row["params"] for row in configs if row["decoder"] == "qad_lite")
    if qad_params - base_params != 2625:
        raise RuntimeError(f"unexpected QAD parameter delta: {qad_params - base_params}")
    for seed in (7, 2025):
        r0_path, _ = by_pair[(seed, "interpolate")]
        r1_path, _ = by_pair[(seed, "qad_lite")]
        left = json.loads(r0_path.read_text(encoding="utf-8"))
        right = json.loads(r1_path.read_text(encoding="utf-8"))
        for payload in (left, right):
            payload.pop("name", None)
            payload.pop("notes", None)
            payload["model"].setdefault("query_decoder", "interpolate")
            payload["model"].setdefault("qad_hidden", 32)
        left["model"]["query_decoder"] = "qad_lite"
        if left != right:
            raise RuntimeError(f"R0/R1 differ beyond decoder for seed {seed}")

    payload = {
        "schema_version": 1,
        "created_at": now(),
        "status": "dry_run" if args.dry_run else "submitting",
        "purpose": prepared["purpose"],
        "test27_policy": prepared["test27_policy"],
        "prepared_manifest": str(PREPARED.resolve()),
        "prepared_manifest_sha256": sha(PREPARED),
        "gpu_preflight_output": str(PREFLIGHT.resolve()),
        "base_params": base_params,
        "qad_params": qad_params,
        "qad_added_params": qad_params - base_params,
        "configs": configs,
        "jobs": [],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    try:
        preflight_job = sbatch("--job-name", "qadq2v_pf", str(PREFLIGHT_SCRIPT))
        payload["jobs"].append({
            "role": "formal_gpu_preflight", "job_id": preflight_job,
            "script": str(PREFLIGHT_SCRIPT.resolve()), "script_sha256": sha(PREFLIGHT_SCRIPT),
        })
        dump(OUTPUT, payload)
        train_job = sbatch(
            "--job-name", "qadq2v", f"--dependency=afterok:{preflight_job}",
            "--array=0-4%4", str(TRAIN_SCRIPT),
        )
        payload["jobs"].append({
            "role": "training_historical_test27_array", "job_id": train_job,
            "array": "0-4%4", "dependency": f"afterok:{preflight_job}",
            "script": str(TRAIN_SCRIPT.resolve()), "script_sha256": sha(TRAIN_SCRIPT),
        })
        payload["status"] = "submitted"
        payload["submitted_at"] = now()
        dump(OUTPUT, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["failed_at"] = now()
        dump(OUTPUT, payload)
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"], "manifest": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
