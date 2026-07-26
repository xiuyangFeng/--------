#!/usr/bin/env python3
"""Submit the frozen six-job vertex/FPS phase after a passing formal preflight.

The surface-area phase remains frozen as a separate backlog until every mapped case
passes the strict geometry audit.  This submitter never silently downgrades an area
configuration to vertex sampling or vertex-only metrics.
"""

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

from training_wss_min.config import ExpConfig  # noqa: E402


SPLIT = REPO / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
STATS = REPO / "data_wss_min/fold_stats/v4/AG_AAA_v4_stratified_train106_global_stats.json"
SLURM = REPO / "training_wss_min/cluster/run_v4_support_query.slurm"
ROWS = (
    ("P1V", "pointnet_v4/ag_aaa_v4_stratified_e2_global_random5000_same.json", "B0/9170", "FPS2000 fixed support -> epoch-resampled vertex-random5000 support"),
    ("P2V", "pointnet_v4/ag_aaa_v4_stratified_e2_global_random5000_sep.json", "P1V", "SAME -> independent vertex-random5000 query"),
    ("Q0", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_fps2000_fixed_same.json", "B1/9169", "legacy ratio/eval -> fixed centers/fixed-support full-query"),
    ("Q1V", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json", "Q0", "FPS2000 fixed support -> epoch-resampled vertex-random5000 support"),
    ("Q2V", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep.json", "Q1V", "SAME -> independent vertex-random5000 query"),
    ("Q3V", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_randomcenter_same.json", "Q1V", "FPS centers -> deterministic epoch-resampled random centers"),
)
AREA_DEFERRED_ROWS = (
    ("P1", "pointnet_v4/ag_aaa_v4_stratified_e2_global_area_random5000_same.json"),
    ("P2", "pointnet_v4/ag_aaa_v4_stratified_e2_global_area_random5000_sep.json"),
    ("Q1", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_area_random5000_fpscenter_same.json"),
    ("Q2", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_area_random5000_fpscenter_sep.json"),
    ("Q3", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_area_random5000_randomcenter_same.json"),
    ("Q4", "pointnetpp_v4/ag_aaa_v4_stratified_sa3_area_random5000_randomcenter_sep.json"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("status") != "passed" or len(preflight.get("jobs", [])) != len(ROWS):
        raise RuntimeError("refusing submission: formal six-config vertex/FPS preflight did not pass")
    checked = {Path(row["config"]).resolve(): row["config_sha256"] for row in preflight["jobs"]}
    expected = {
        (REPO / "training_wss_min/configs" / rel).resolve()
        for _, rel, _, _ in ROWS
    }
    if set(checked) != expected:
        raise RuntimeError("preflight config set does not exactly match the frozen vertex/FPS phase")
    submitted = []
    try:
        for experiment, rel, control, change in ROWS:
            path = REPO / "training_wss_min/configs" / rel
            cfg = ExpConfig.from_json(path)
            if cfg.run_dir.exists():
                raise FileExistsError(f"run path already exists: {cfg.run_dir}")
            if checked.get(path.resolve()) != digest(path):
                raise RuntimeError(f"config absent from preflight or changed after it: {path}")
            sampling = {
                "support_n_points": cfg.data.support_n_points,
                "support_sampling": cfg.data.support_sampling,
                "query_mode": cfg.data.query_mode,
                "query_n_points": cfg.data.query_n_points,
                "query_sampling": cfg.data.query_sampling,
                "sa_center_sampling": cfg.model.sa_center_sampling,
                "sa_center_counts": list(cfg.model.sa_center_counts),
            }
            job_name = f"v4sq_{experiment.lower()}"
            if args.dry_run:
                job_id, state = None, "DRY_RUN"
            else:
                proc = subprocess.run(
                    ["/public/slurm/bin/sbatch", "--parsable", "--job-name", job_name,
                     str(SLURM), str(path), experiment], check=True, text=True,
                    capture_output=True,
                )
                job_id, state = proc.stdout.strip().split(";", 1)[0], "SUBMITTED"
            submitted.append({
                "experiment_id": experiment, "job_id": job_id, "job_name": job_name,
                "state_at_submit": state, "config_path": str(path),
                "config_sha256": digest(path), "split_sha256": digest(SPLIT),
                "stats_sha256": digest(STATS), "expected_run_path": str(cfg.run_dir),
                "control_experiment": control, "single_changed_factor": change,
                "sampling_semantics": sampling,
                "surface_metric_mode": cfg.eval.surface_metric_mode,
                "submit_time": datetime.now(timezone.utc).isoformat(),
            })
    finally:
        status = "dry_run" if args.dry_run else ("submitted" if len(submitted) == len(ROWS) else "partial")
        payload = {
            "protocol_phase": "vertex_fps_phase_v1",
            "status": status,
            "preflight": str(args.preflight.resolve()),
            "surface_area_policy": (
                "legacy_vertex metrics only; strict area experiments are deferred, not downgraded"
            ),
            "deferred_area_experiments": [
                {"experiment_id": exp, "config_path": str(REPO / "training_wss_min/configs" / rel)}
                for exp, rel in AREA_DEFERRED_ROWS
            ],
            "jobs": submitted,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if len(submitted) != len(ROWS):
        raise RuntimeError(f"partial submission: only {len(submitted)}/{len(ROWS)} jobs submitted")
    print(json.dumps({"status": payload["status"], "jobs": len(submitted),
                      "manifest": str(args.manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
