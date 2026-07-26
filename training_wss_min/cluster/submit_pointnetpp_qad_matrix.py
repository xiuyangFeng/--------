#!/usr/bin/env python3
"""Validate and submit the paired PointNet++ R0/R1 QAD val21 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig, input_dim
from training_wss_min.models import build_model


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "training_wss_min/configs/pointnetpp_qad_20260719"
ANCHOR = REPO / "training_wss_min/configs/pointnetpp_q2v_arch_20260718/q2v_arch_dev_n32_w32.json"
ANCHOR_METRICS = REPO / (
    "training_wss_min/runs/pointnetpp_q2v_arch_dev/outputs/"
    "q2v_arch_dev_n32_w32/eval/ckpt_best/metrics.json"
)
SCRIPT = REPO / "training_wss_min/cluster/run_pointnetpp_qad_matrix.slurm"
OUTPUT = REPO / "training_wss_min/preflight/pointnetpp_qad_matrix_submission.json"
SBATCH = Path("/public/slurm/bin/sbatch")

ORDER = (
    "r1_qad_n32_w32_seed1234.json",
    "r0_interpolate_n32_w32_seed7.json",
    "r1_qad_n32_w32_seed7.json",
    "r0_interpolate_n32_w32_seed2025.json",
    "r1_qad_n32_w32_seed2025.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def comparable_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("name", None)
    payload.pop("notes", None)
    payload["model"].setdefault("query_decoder", "interpolate")
    payload["model"].setdefault("qad_hidden", 32)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for path in (ANCHOR, ANCHOR_METRICS, SCRIPT, SBATCH):
        if not path.is_file():
            raise FileNotFoundError(path)

    configs = []
    by_seed = {}
    for task_id, filename in enumerate(ORDER):
        path = CONFIG_ROOT / filename
        cfg = ExpConfig.from_json(path)
        for dependency in (
            Path(cfg.data.split_path), Path(cfg.data.wss_stats_path),
            Path(cfg.data.feature_stats_path or ""),
        ):
            if not dependency.is_file():
                raise FileNotFoundError(dependency)
        if cfg.eval.surface_metric_mode != "legacy_vertex":
            raise ValueError(f"unexpected surface metric mode: {path}")
        if cfg.train.selection_rule != "train_loss" or cfg.train.epochs != 400:
            raise ValueError(f"training budget drift: {path}")
        if cfg.run_dir.exists():
            raise FileExistsError(f"refuse duplicate run dir: {cfg.run_dir}")
        model = build_model(cfg.model, input_dim(cfg))
        params = sum(p.numel() for p in model.parameters())
        row = {
            "task_id": task_id,
            "experiment_id": path.stem,
            "config": str(path.resolve()),
            "sha256": sha(path),
            "seed": cfg.train.seed,
            "decoder": cfg.model.query_decoder,
            "params": params,
            "expected_run_dir": str(cfg.run_dir.resolve()),
        }
        configs.append(row)
        by_seed[(cfg.train.seed, cfg.model.query_decoder)] = path

    base_1234 = comparable_payload(ANCHOR)
    for seed in (1234, 7, 2025):
        control_path = ANCHOR if seed == 1234 else by_seed[(seed, "interpolate")]
        qad_path = by_seed[(seed, "qad_lite")]
        control = comparable_payload(control_path)
        qad = comparable_payload(qad_path)
        control["train"]["seed"] = seed
        control["model"]["query_decoder"] = "qad_lite"
        if control != qad:
            raise ValueError(f"R0/R1 differ beyond decoder for seed {seed}")
        if seed == 1234:
            anchor = dict(base_1234)
            anchor["model"]["query_decoder"] = "qad_lite"
            if anchor != qad:
                raise ValueError("seed1234 QAD is not a single-variable child of existing anchor")

    base_params = sum(
        p.numel() for p in build_model(ExpConfig.from_json(ANCHOR).model, 6).parameters()
    )
    qad_params = next(row["params"] for row in configs if row["decoder"] == "qad_lite")
    added_params = qad_params - base_params
    if added_params <= 0 or added_params >= 50_000:
        raise ValueError(f"QAD parameter budget failed: added={added_params}")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_passed",
        "protocol": "val21 only; original test27 forbidden",
        "primary_metric": "normalized.field_casebalanced.r2",
        "guardrails": [
            "physical.field_casebalanced.r2", "physical MAE/RMSE",
            "normalized Spearman", "normalized top10 IoU", "normalized p99 ratio",
        ],
        "anchor_config": str(ANCHOR.resolve()),
        "anchor_config_sha256": sha(ANCHOR),
        "anchor_metrics": str(ANCHOR_METRICS.resolve()),
        "anchor_metrics_sha256": sha(ANCHOR_METRICS),
        "single_variable_gate": "passed",
        "base_params": base_params,
        "qad_params": qad_params,
        "qad_added_params": added_params,
        "max_array_concurrency": 3,
        "configs": configs,
        "job": None,
    }
    atomic_write(OUTPUT, payload)
    if args.dry_run:
        print(json.dumps({
            "status": payload["status"], "manifest": str(OUTPUT),
            "configs": len(configs), "qad_added_params": added_params,
        }, ensure_ascii=False, indent=2))
        return

    proc = subprocess.run(
        [str(SBATCH), "--parsable", "--array=0-4%3", str(SCRIPT)],
        cwd=REPO, text=True, capture_output=True,
    )
    if proc.returncode:
        payload["status"] = "submission_failed"
        payload["submission_error"] = proc.stderr.strip()
        atomic_write(OUTPUT, payload)
        raise RuntimeError(proc.stderr.strip())
    job_id = proc.stdout.strip().split(";", 1)[0]
    payload["status"] = "submitted"
    payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
    payload["job"] = {
        "job_id": job_id,
        "array": "0-4%3",
        "script": str(SCRIPT.resolve()),
        "script_sha256": sha(SCRIPT),
    }
    atomic_write(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"], "job_id": job_id,
        "array": payload["job"]["array"], "manifest": str(OUTPUT),
        "qad_added_params": added_params,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
