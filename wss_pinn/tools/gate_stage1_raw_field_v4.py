"""Post-training field/derivative Gate before the PE phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wss_pinn.config import ExperimentConfig
from wss_pinn.data import VolumeFieldDataset
from wss_pinn.data.dataset import _take_strict_volume
from wss_pinn.models import build_model
from wss_pinn.utils import ROOT, atomic_write_json, git_state, sha256_file, utc_now
from wss_pinn.volume_utils import tensor_state_sha256


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4"
OUTPUT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage1_raw_gate/report.json"
RAW_CONFIGS = ("g_raw_s1234.json", "l_raw_s1234.json")


def _trained_derivative_probe(
    config: ExperimentConfig, checkpoint: Path, device: torch.device
) -> dict[str, Any]:
    dataset = VolumeFieldDataset(
        config["paths"]["sidecar_manifest"],
        config["paths"]["field_stats"],
        roles=["val"],
        input_variant=config.input_variant,
        sampling=config["sampling"],
        seed=1234,
    )
    case = dataset.cases[0]
    query_idx = np.asarray(case.fixed_validation_indices[:512], dtype=np.int64)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(config).to(device=device, dtype=torch.float64).eval()
    model.load_state_dict(
        {key: value.to(dtype=torch.float64) for key, value in payload["model"].items()},
        strict=True,
    )
    predictions = []
    encoded_reference = None
    for support_seed in range(3):
        rng = np.random.default_rng(1234 + support_seed * 1009)
        support_idx = _take_strict_volume(
            rng, case.is_wall, 5000, excluded=query_idx
        )
        support_pos = torch.as_tensor(
            np.asarray(case.coords[support_idx], dtype=np.float64), device=device
        )
        support_features = torch.as_tensor(
            dataset._features(case.coords[support_idx], case.geometry[support_idx]),
            device=device,
            dtype=torch.float64,
        )
        encoded = model.encode_support(
            support_pos,
            support_features,
            torch.zeros(5000, dtype=torch.long, device=device),
            unit_ids=[case.case_id],
            epoch=0,
            global_seed=1234 + support_seed,
            evaluation=True,
        )
        query = torch.as_tensor(
            np.asarray(case.coords[query_idx], dtype=np.float64), device=device
        )
        with torch.no_grad():
            predictions.append(
                model.decode_query(
                    encoded,
                    query,
                    torch.zeros(len(query), dtype=torch.long, device=device),
                ).cpu().numpy()
            )
        if support_seed == 0:
            encoded_reference = encoded
    stack = np.stack(predictions)
    truth_velocity = np.asarray(case.velocity[query_idx], dtype=np.float64)
    truth_pressure = np.asarray(case.pressure[query_idx], dtype=np.float64)
    truth = np.column_stack(
        [
            (truth_velocity - dataset.velocity_mean) / dataset.velocity_std,
            (truth_pressure - dataset.pressure_mean) / dataset.pressure_std,
        ]
    )
    support_ratio = float(
        np.mean(np.std(stack, axis=0)) / max(float(np.std(truth)), 1e-30)
    )

    query = torch.as_tensor(
        np.asarray(case.coords[query_idx[:24]], dtype=np.float64),
        device=device,
    ).requires_grad_(True)
    query_batch = torch.zeros(len(query), dtype=torch.long, device=device)
    output = model.decode_query(encoded_reference, query, query_batch)[:, 0]
    first = torch.autograd.grad(output.sum(), query, create_graph=True)[0][:, 0]
    second = torch.autograd.grad(first.sum(), query)[0][:, 0]

    def decode(value: torch.Tensor) -> torch.Tensor:
        return model.decode_query(encoded_reference, value, query_batch)[:, 0]

    h1 = 1e-5
    plus = query.detach().clone(); plus[:, 0] += h1
    minus = query.detach().clone(); minus[:, 0] -= h1
    fd_first = (decode(plus) - decode(minus)) / (2.0 * h1)
    h2 = 2e-4
    plus2 = query.detach().clone(); plus2[:, 0] += h2
    minus2 = query.detach().clone(); minus2[:, 0] -= h2
    fd_second = (decode(plus2) - 2.0 * decode(query.detach()) + decode(minus2)) / h2**2
    first_error = float(
        torch.linalg.vector_norm(first.detach() - fd_first)
        / torch.linalg.vector_norm(fd_first).clamp_min(1e-10)
    )
    second_error = float(
        torch.linalg.vector_norm(second.detach() - fd_second)
        / torch.linalg.vector_norm(fd_second).clamp_min(1e-10)
    )
    gate = (
        bool(torch.isfinite(first).all() and torch.isfinite(second).all())
        and first_error < 5e-5
        and second_error < 5e-3
        and support_ratio < 0.10
    )
    return {
        "case_id": case.case_id,
        "first_derivative_relative_l2": first_error,
        "second_derivative_relative_l2": second_error,
        "support_resampling_std_over_truth_spatial_std": support_ratio,
        "gate_result": "pass" if gate else "fail",
    }


def run(*, device_name: str = "cuda:0") -> dict[str, Any]:
    device = torch.device(device_name)
    failures = []
    arms = []
    for name in RAW_CONFIGS:
        config = ExperimentConfig.from_json(CONFIG_ROOT / name)
        run_dir = config.run_dir
        summary_path = run_dir / "training_summary.json"
        evaluation_path = run_dir / "evaluation_val15_best_validation_field_cb.json"
        checkpoint = run_dir / "checkpoints/best_validation_field_cb.pt"
        total_checkpoint = run_dir / "checkpoints/best_validation_total.pt"
        last_checkpoint = run_dir / "checkpoints/last.pt"
        required = [summary_path, evaluation_path, checkpoint, total_checkpoint, last_checkpoint]
        if any(not path.is_file() for path in required):
            failures.append(f"missing_outputs:{config['experiment']['id']}")
            continue
        summary = json.loads(summary_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        derivative = _trained_derivative_probe(config, checkpoint, device)
        field_finite = np.isfinite(
            [
                evaluation["case_balanced_regression"][metric][stat]
                for metric in ("u", "v", "w", "speed", "pressure")
                for stat in ("r2", "mae", "rmse")
            ]
        ).all()
        field_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        total_payload = torch.load(
            total_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_match = (
            field_payload.get("epoch") == total_payload.get("epoch")
            and tensor_state_sha256(field_payload["model"])
            == tensor_state_sha256(total_payload["model"])
        )
        arm_gate = (
            summary.get("status") == "completed"
            and bool(field_finite)
            and derivative["gate_result"] == "pass"
            and checkpoint_match
        )
        if not arm_gate:
            failures.append(f"raw_arm_gate:{config['experiment']['id']}")
        arms.append(
            {
                "experiment_id": config["experiment"]["id"],
                "validation_field_score_cb": summary[
                    "best_validation_field_score_cb"
                ],
                "field_metrics_finite": bool(field_finite),
                "best_field_equals_best_total_state": checkpoint_match,
                "derivative_and_resampling": derivative,
                "gate_result": "pass" if arm_gate else "fail",
            }
        )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_field_v4",
        "phase": "raw",
        "gate_result": "pass" if not failures else "fail",
        "gate_reasons": failures,
        "arms": arms,
        "test35_cases_read": 0,
        "pe_formal_training_allowed": not failures,
        "formal_pe_training_submitted": False,
        "git": git_state(),
    }
    atomic_write_json(OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(device_name=args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
