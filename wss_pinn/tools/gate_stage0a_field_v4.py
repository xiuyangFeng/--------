"""Assemble the machine-readable Stage-0a Gate for field-v4."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from wss_pinn.config import ExperimentConfig
from wss_pinn.utils import ROOT, atomic_write_json, git_state, sha256_file, utc_now


OUTPUT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/report.json"
DATA_REPORT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_data/report.json"
PREFLIGHT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_preflight/report.json"
B0 = ROOT / "outputs/wss_pinn/volume_uvwp_peak_field_v4/b0/evaluation_val15.json"
MANUFACTURED = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/probes/manufactured_and_derivative_gate.json"
V3_PROBE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/probes/v3_support_latent_gradient_probe.json"
BULK = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/probes/bulk_knn_truth_oracle_val15.json"
CPU_SMOKE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_cpu_smoke/report.json"
MASTER_GPU_SMOKE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_gpu_preflight/master_gpu2_report.json"
NODE04_GPU_SMOKE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_gpu_preflight/node04_gpu0_report.json"
V4_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/manifest.json"
V4_QUERY_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/validation_queries/manifest.json"
V3_PREFLIGHT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_preflight/report.json"
V3_SUMMARY_MANIFEST = ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/manifest.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tests() -> dict[str, Any]:
    process = subprocess.run(
        [
            "/public/newhome/cy/.conda/envs/GNN/bin/python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "wss_pinn/tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    combined = process.stdout + process.stderr
    return {
        "command": "python -m unittest discover -s wss_pinn/tests -p test_*.py -v",
        "returncode": process.returncode,
        "passed": process.returncode == 0,
        "summary_tail": combined.splitlines()[-8:],
    }


def _legacy_compatibility() -> dict[str, Any]:
    roots = {
        "v1": ROOT / "wss_pinn/configs/volume_uvwp_peak_v1",
        "v2": ROOT / "wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2",
        "v3": ROOT / "wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3",
    }
    counts = {}
    for name, root in roots.items():
        configs = [
            path
            for path in sorted(root.glob("*.json"))
            if path.name != "matrix.json"
        ]
        for path in configs:
            ExperimentConfig.from_json(path)
        counts[name] = len(configs)
    v3_preflight = _load(V3_PREFLIGHT)
    config_hashes_match = all(
        sha256_file(roots["v3"] / name) == values["source_sha256"]
        for name, values in v3_preflight["configs"].items()
    )
    summary = _load(V3_SUMMARY_MANIFEST)
    historical_outputs_match = all(
        Path(row["path"]).is_file() and sha256_file(row["path"]) == row["sha256"]
        for row in summary["sources"]
    )
    return {
        "config_counts": counts,
        "expected_counts": {"v1": 8, "v2": 8, "v3": 6},
        "all_configs_parse": counts == {"v1": 8, "v2": 8, "v3": 6},
        "v3_config_source_hashes_unchanged": config_hashes_match,
        "v3_historical_summary_sources_unchanged": historical_outputs_match,
        "v3_summary_manifest": {
            "path": str(V3_SUMMARY_MANIFEST),
            "sha256": sha256_file(V3_SUMMARY_MANIFEST),
        },
    }


def run() -> dict[str, Any]:
    failures = []
    tests = _run_tests()
    if not tests["passed"]:
        failures.append("focused_tests_failed")
    data = _load(DATA_REPORT)
    preflight = _load(PREFLIGHT)
    b0 = _load(B0)
    manufactured = _load(MANUFACTURED)
    v3_probe = _load(V3_PROBE)
    bulk = _load(BULK)
    cpu_smoke = _load(CPU_SMOKE)
    master_gpu_smoke = _load(MASTER_GPU_SMOKE)
    node04_gpu_smoke = _load(NODE04_GPU_SMOKE)
    manifest = _load(V4_MANIFEST)
    queries = _load(V4_QUERY_MANIFEST)
    legacy = _legacy_compatibility()
    checks = {
        "data_contract": data.get("gate_result") == "pass",
        "static_preflight": preflight.get("gate_result") == "pass",
        "b0_predictor_evaluator_isolation": b0.get(
            "predictor_evaluator_interface_isolated"
        )
        is True,
        "b0_val15_complete": len(b0.get("cases", [])) == 15,
        "manufactured_and_derivative_gate": manufactured.get("gate_result") == "pass",
        "v3_support_probe_train_val_only": v3_probe.get("test35_cases_read") == 0,
        "direct_gradient_groups_present": all(
            key in v3_probe.get("gradient_cosines", {}).get("means", {}).get(
                "best_validation_data", {}
            )
            for key in ("cos_data_continuity", "cos_data_momentum", "cos_data_bc")
        ),
        "bulk_oracle_labeled_non_deployable": "not deployable"
        in bulk.get("oracle_warning", ""),
        "cpu_smoke": cpu_smoke.get("gate_result") == "pass",
        "master_4090_gpu_smoke": master_gpu_smoke.get("gate_result") == "pass",
        "node04_a100_gpu_smoke": node04_gpu_smoke.get("gate_result") == "pass",
        "test35_absent_from_v4_manifest": all(
            row.get("role") in {"train", "val"} for row in manifest.get("cases", [])
        ),
        "fixed_query_count": len(queries.get("cases", [])) == 15,
        "fixed_query_hashes_unique": len(
            {row["query_sha256"] for row in queries.get("cases", [])}
        )
        == 15,
        "legacy_configs_parse": legacy["all_configs_parse"],
        "v3_config_hashes_unchanged": legacy["v3_config_source_hashes_unchanged"],
        "v3_historical_outputs_unchanged": legacy[
            "v3_historical_summary_sources_unchanged"
        ],
    }
    failures.extend(key for key, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_field_v4",
        "gate_result": "pass" if not failures else "fail",
        "gate_reasons": failures,
        "checks": checks,
        "tests": tests,
        "selection_contract": {
            "metric": "validation_field_score_cb",
            "formula": "mean_case((MSE_u_std + MSE_v_std + MSE_w_std + MSE_p_std)/4)",
            "manual_unit_test": "test_case_balanced_score_matches_manual_and_exposes_legacy_double_weight",
            "legacy_overweighted_case": "ILO/ZHANG_YAN_SHAN-0/before",
            "legacy_weight_ratio": 2.0,
        },
        "b0": {
            "validation_field_score_cb": b0["validation_field_score_cb"],
            "case_balanced_regression": b0["case_balanced_regression"],
            "evaluation": {"path": str(B0), "sha256": sha256_file(B0)},
        },
        "artifacts": {
            "data_report": {"path": str(DATA_REPORT), "sha256": sha256_file(DATA_REPORT)},
            "preflight": {"path": str(PREFLIGHT), "sha256": sha256_file(PREFLIGHT)},
            "manufactured": {"path": str(MANUFACTURED), "sha256": sha256_file(MANUFACTURED)},
            "v3_probe": {"path": str(V3_PROBE), "sha256": sha256_file(V3_PROBE)},
            "bulk_oracle": {"path": str(BULK), "sha256": sha256_file(BULK)},
            "cpu_smoke": {"path": str(CPU_SMOKE), "sha256": sha256_file(CPU_SMOKE)},
            "master_gpu_smoke": {"path": str(MASTER_GPU_SMOKE), "sha256": sha256_file(MASTER_GPU_SMOKE)},
            "node04_gpu_smoke": {"path": str(NODE04_GPU_SMOKE), "sha256": sha256_file(NODE04_GPU_SMOKE)},
            "v4_manifest": {"path": str(V4_MANIFEST), "sha256": sha256_file(V4_MANIFEST)},
            "fixed_queries": {"path": str(V4_QUERY_MANIFEST), "sha256": sha256_file(V4_QUERY_MANIFEST)},
        },
        "legacy_compatibility": legacy,
        "test35_cases_read_by_v4": 0,
        "formal_training_submitted": False,
        "git": git_state(),
    }
    atomic_write_json(OUTPUT, report)
    return report


def main() -> None:
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({key: report[key] for key in ("gate_result", "gate_reasons", "checks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
