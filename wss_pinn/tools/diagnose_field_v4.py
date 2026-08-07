"""Engineering probes required by field-v4 Stage 0-a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from wss_pinn.config import ExperimentConfig
from wss_pinn.data import VolumeFieldDataset, collate_volume_samples
from wss_pinn.data.dataset import _take_strict_volume
from wss_pinn.losses import compute_losses
from wss_pinn.metrics import RegressionAccumulator
from wss_pinn.models import build_model
from wss_pinn.physics.residuals import generalized_newtonian_residuals
from wss_pinn.physics.rheology import carreau_yasuda
from wss_pinn.train import _move_batch, seed_everything
from wss_pinn.utils import ROOT, atomic_write_json, git_state, sha256_file, utc_now


OUTPUT_ROOT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/probes"
MANUFACTURED_OUTPUT = OUTPUT_ROOT / "manufactured_and_derivative_gate.json"
V3_OUTPUT = OUTPUT_ROOT / "v3_support_latent_gradient_probe.json"
BULK_OUTPUT = OUTPUT_ROOT / "bulk_knn_truth_oracle_val15.json"
SPLIT_SHA256 = "c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9"
V4_CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4"
V4_DATA_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/manifest.json"
V3_STATS = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/field_stats.json"


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual - expected
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected).clamp_min(1e-14)
        ),
        "absolute_linf": float(torch.max(torch.abs(delta))),
    }


def _manufactured_probe() -> dict[str, Any]:
    torch.manual_seed(1234)
    coords = (torch.rand(128, 3, dtype=torch.float64) + 0.2).requires_grad_(True)
    a, b, c, d = 0.7, -0.4, 0.5, 3.0
    x, y, z = coords.unbind(dim=1)
    velocity = torch.stack([a * y**2, b * z**2, c * x**2], dim=1)
    pressure = d * x * y
    prediction = torch.cat([velocity, (pressure / 1060.0)[:, None]], dim=1)
    field_stats = {
        "velocity_m_s": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "pressure_relative_pa": {"mean": [0.0], "std": [1060.0]},
    }
    physics = {
        "density_kg_m3": 1060.0,
        "reference_velocity_m_s": 1.0,
        "shear_rate_epsilon_s_inv": 1e-6,
        "rheology": {
            "mu_inf_pa_s": 0.0035,
            "mu_zero_pa_s": 0.16,
            "lambda_s": 8.2,
            "a": 0.64,
            "n": 0.2128,
        },
    }
    residual = generalized_newtonian_residuals(
        prediction,
        coords,
        torch.ones(len(coords), dtype=torch.float64),
        field_stats=field_stats,
        physics_config=physics,
    )
    expected_gradient = torch.zeros(128, 3, 3, dtype=torch.float64)
    expected_gradient[:, 0, 1] = 2.0 * a * y
    expected_gradient[:, 1, 2] = 2.0 * b * z
    expected_gradient[:, 2, 0] = 2.0 * c * x
    expected_second = torch.stack(
        [
            torch.full_like(x, 2.0 * c),
            torch.full_like(x, 2.0 * a),
            torch.full_like(x, 2.0 * b),
        ],
        dim=1,
    )
    measured_second = []
    for component, axis in ((2, 0), (0, 1), (1, 2)):
        first = torch.autograd.grad(
            velocity[:, component].sum(), coords, create_graph=True, retain_graph=True
        )[0][:, axis]
        measured_second.append(
            torch.autograd.grad(
                first.sum(), coords, create_graph=True, retain_graph=True
            )[0][:, axis]
        )
    measured_second = torch.stack(measured_second, dim=1)

    strain = 0.5 * (expected_gradient + expected_gradient.transpose(1, 2))
    shear = torch.sqrt(
        2.0 * torch.sum(strain * strain, dim=(1, 2)) + 1e-12
    )
    viscosity = carreau_yasuda(shear, physics["rheology"])
    stress = 2.0 * viscosity[:, None, None] * strain
    divergence = []
    for component in range(3):
        value = torch.zeros_like(x)
        for axis in range(3):
            value = value + torch.autograd.grad(
                stress[:, component, axis].sum(),
                coords,
                create_graph=True,
                retain_graph=True,
            )[0][:, axis]
        divergence.append(value)
    divergence = torch.stack(divergence, dim=1)
    convective = torch.stack(
        [2.0 * a * b * y * z**2, 2.0 * b * c * x**2 * z, 2.0 * a * c * x * y**2],
        dim=1,
    )
    pressure_gradient = torch.stack([d * y, d * x, torch.zeros_like(x)], dim=1)
    expected_momentum = 1060.0 * convective + pressure_gradient - divergence
    checks = {
        "field_values": _relative_error(
            residual["velocity_phys_m_s"], velocity
        ),
        "first_derivatives": _relative_error(
            residual["velocity_gradient_hat"], expected_gradient
        ),
        "second_derivatives": _relative_error(measured_second, expected_second),
        "carreau_yasuda_viscosity": _relative_error(
            residual["viscosity_pa_s"], viscosity
        ),
        "momentum_residual": _relative_error(
            residual["momentum_phys_pa_per_m"], expected_momentum
        ),
        "continuity_linf": float(torch.max(torch.abs(residual["continuity_phys_s_inv"]))),
    }
    gate = (
        checks["field_values"]["absolute_linf"] < 1e-12
        and checks["first_derivatives"]["absolute_linf"] < 1e-12
        and checks["second_derivatives"]["absolute_linf"] < 1e-12
        and checks["carreau_yasuda_viscosity"]["absolute_linf"] < 1e-12
        and checks["momentum_residual"]["absolute_linf"] < 1e-9
        and checks["continuity_linf"] < 1e-12
    )
    return {"gate_result": "pass" if gate else "fail", "checks": checks}


def _model_derivative_gate() -> dict[str, Any]:
    reports = []
    for path in sorted(V4_CONFIG_ROOT.glob("*_s1234.json")):
        config = ExperimentConfig.from_json(path)
        torch.manual_seed(1234)
        model = build_model(config).double().eval()
        support = torch.rand(256, 3, dtype=torch.float64)
        features = torch.rand(256, 6, dtype=torch.float64)
        encoded = model.encode_support(
            support,
            features,
            torch.zeros(256, dtype=torch.long),
            unit_ids=["manufactured"],
            epoch=0,
            global_seed=1234,
            evaluation=True,
        )
        query = (torch.rand(24, 3, dtype=torch.float64) * 0.6 + 0.2).requires_grad_(True)
        output = model.decode_query(encoded, query, torch.zeros(24, dtype=torch.long))[:, 0]
        first = torch.autograd.grad(output.sum(), query, create_graph=True)[0][:, 0]
        second = torch.autograd.grad(first.sum(), query)[0][:, 0]

        def evaluate_at(value: torch.Tensor) -> torch.Tensor:
            return model.decode_query(
                encoded, value, torch.zeros(len(value), dtype=torch.long)
            )[:, 0]

        h1 = 1e-5
        plus = query.detach().clone(); plus[:, 0] += h1
        minus = query.detach().clone(); minus[:, 0] -= h1
        fd_first = (evaluate_at(plus) - evaluate_at(minus)) / (2.0 * h1)
        h2 = 2e-4
        plus2 = query.detach().clone(); plus2[:, 0] += h2
        minus2 = query.detach().clone(); minus2[:, 0] -= h2
        center = evaluate_at(query.detach())
        fd_second = (evaluate_at(plus2) - 2.0 * center + evaluate_at(minus2)) / h2**2
        first_error = float(
            torch.linalg.vector_norm(first.detach() - fd_first)
            / torch.linalg.vector_norm(fd_first).clamp_min(1e-10)
        )
        second_error = float(
            torch.linalg.vector_norm(second.detach() - fd_second)
            / torch.linalg.vector_norm(fd_second).clamp_min(1e-10)
        )
        reports.append(
            {
                "config": path.name,
                "first_derivative_relative_l2": first_error,
                "second_derivative_relative_l2": second_error,
                "finite": bool(torch.isfinite(first).all() and torch.isfinite(second).all()),
                "gate_result": (
                    "pass"
                    if first_error < 5e-5 and second_error < 5e-3
                    else "fail"
                ),
            }
        )
    return {
        "gate_result": "pass" if all(row["gate_result"] == "pass" for row in reports) else "fail",
        "arms": reports,
    }


def run_manufactured() -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "split_sha256": SPLIT_SHA256,
        "manufactured_solution": _manufactured_probe(),
        "query_derivative_gate": _model_derivative_gate(),
        "git": git_state(),
    }
    report["gate_result"] = (
        "pass"
        if report["manufactured_solution"]["gate_result"] == "pass"
        and report["query_derivative_gate"]["gate_result"] == "pass"
        else "fail"
    )
    atomic_write_json(MANUFACTURED_OUTPUT, report)
    return report


def _gradient_cosine(first, second, parameters) -> float:
    first_grad = torch.autograd.grad(
        first, parameters, retain_graph=True, allow_unused=True
    )
    second_grad = torch.autograd.grad(
        second, parameters, retain_graph=True, allow_unused=True
    )
    dot = first.new_zeros(())
    first_sq = first.new_zeros(())
    second_sq = first.new_zeros(())
    for left, right in zip(first_grad, second_grad):
        if left is None or right is None:
            continue
        dot = dot + torch.sum(left * right)
        first_sq = first_sq + torch.sum(left * left)
        second_sq = second_sq + torch.sum(right * right)
    return float((dot / torch.sqrt(first_sq * second_sq).clamp_min(1e-30)).detach().cpu())


def _gradient_probe(device: torch.device) -> dict[str, Any]:
    config_path = ROOT / "wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3/qsf_xyz_geom_bcpde.json"
    base = ExperimentConfig.from_json(config_path)
    payload = base.as_dict()
    payload["sampling"].update(
        {"support_points": 512, "query_points": 512, "physics_points": 64, "wall_points": 64, "inlet_points": 32, "outlet_points": 32}
    )
    config = ExperimentConfig(payload)
    dataset = VolumeFieldDataset(
        config["paths"]["sidecar_manifest"],
        config["paths"]["field_stats"],
        roles=["val"],
        input_variant=config.input_variant,
        sampling=config["sampling"],
        seed=1234,
    )
    dataset.set_epoch(0)
    stats = json.loads(Path(config["paths"]["field_stats"]).read_text())
    checkpoint = ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/runs/QSF-XYZG-BCPDE-s1234-v3/checkpoints/best_validation_data.pt"
    reports = []
    for stage in ("initialization", "best_validation_data"):
        seed_everything(1234)
        model = build_model(config).to(device=device, dtype=torch.float32)
        if stage != "initialization":
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"], strict=True)
        model.train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        for case_index in range(2):
            batch = collate_volume_samples([dataset[case_index]])
            batch = _move_batch(batch, device, torch.float32)
            losses = compute_losses(
                model, batch, field_stats=stats, config=config, epoch=0
            )
            momentum = (
                losses["momentum_x_raw"]
                + losses["momentum_y_raw"]
                + losses["momentum_z_raw"]
            ) / 3.0
            reports.append(
                {
                    "stage": stage,
                    "case_id": batch["case_ids"][0],
                    "cos_data_continuity": _gradient_cosine(
                        losses["data_total"], losses["continuity_raw"], parameters
                    ),
                    "cos_data_momentum": _gradient_cosine(
                        losses["data_total"], momentum, parameters
                    ),
                    "cos_data_bc": _gradient_cosine(
                        losses["data_total"], losses["bc_group_raw"], parameters
                    ),
                }
            )
            del losses, batch
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "config": str(config_path),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "cases": reports,
        "means": {
            stage: {
                key: float(np.mean([row[key] for row in reports if row["stage"] == stage]))
                for key in ("cos_data_continuity", "cos_data_momentum", "cos_data_bc")
            }
            for stage in ("initialization", "best_validation_data")
        },
    }


def _support_and_latent_probe(device: torch.device) -> dict[str, Any]:
    config_path = ROOT / "wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3/qsf_xyz_geom_bc.json"
    config = ExperimentConfig.from_json(config_path)
    checkpoint = ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/runs/QSF-XYZG-BC-s1234-v3/checkpoints/best_validation_data.pt"
    dataset = VolumeFieldDataset(
        V4_DATA_MANIFEST,
        V3_STATS,
        roles=["val"],
        input_variant="xyz_geom",
        sampling=config["sampling"],
        seed=1234,
    )
    torch.manual_seed(1234)
    model = build_model(config).to(device=device, dtype=torch.float32).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    selected_cases = []
    for cohort in ("AAA", "AG", "ILO"):
        selected_cases.extend(
            [case for case in dataset.cases if case.case_id.startswith(cohort + "/")][:2]
        )
    case_reports = []
    latent_means = []
    latent_within = []
    with torch.no_grad():
        for case in selected_cases:
            query_idx = np.asarray(case.fixed_validation_indices[:1000], dtype=np.int64)
            query = torch.as_tensor(
                np.asarray(case.coords[query_idx], dtype=np.float32), device=device
            )
            truth_velocity = np.asarray(case.velocity[query_idx], dtype=np.float32)
            truth_pressure = np.asarray(case.pressure[query_idx], dtype=np.float32)
            truth = np.column_stack(
                [
                    (truth_velocity - dataset.velocity_mean) / dataset.velocity_std,
                    (truth_pressure - dataset.pressure_mean) / dataset.pressure_std,
                ]
            )
            predictions = []
            latents = []
            for support_seed in range(5):
                rng = np.random.default_rng(1234 + support_seed * 1009 + len(case_reports))
                support_idx = _take_strict_volume(
                    rng, case.is_wall, 5000, excluded=query_idx
                )
                support_pos = torch.as_tensor(
                    np.asarray(case.coords[support_idx], dtype=np.float32), device=device
                )
                support_features = torch.as_tensor(
                    dataset._features(case.coords[support_idx], case.geometry[support_idx]),
                    device=device,
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
                prediction = model.decode_query(
                    encoded, query, torch.zeros(len(query), dtype=torch.long, device=device)
                )
                predictions.append(prediction.cpu().numpy())
                latents.append(encoded["case_latent"].cpu().numpy()[0])
            prediction_stack = np.stack(predictions)
            latent_stack = np.stack(latents)
            single_score = float(np.mean(np.square(prediction_stack[0] - truth)))
            ensemble_score = float(np.mean(np.square(prediction_stack.mean(axis=0) - truth)))
            case_reports.append(
                {
                    "case_id": case.case_id,
                    "prediction_sampling_std_over_prediction_spatial_std": float(
                        np.mean(np.std(prediction_stack, axis=0))
                        / max(float(np.std(prediction_stack.mean(axis=0))), 1e-30)
                    ),
                    "prediction_sampling_std_over_truth_spatial_std": float(
                        np.mean(np.std(prediction_stack, axis=0))
                        / max(float(np.std(truth)), 1e-30)
                    ),
                    "single_support_field_score": single_score,
                    "five_support_ensemble_field_score": ensemble_score,
                    "ensemble_delta": ensemble_score - single_score,
                }
            )
            latent_means.append(latent_stack.mean(axis=0))
            latent_within.append(latent_stack.std(axis=0))
    latent_means_array = np.stack(latent_means)
    latent_within_array = np.stack(latent_within)
    return {
        "config": str(config_path),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "case_ids": [case.case_id for case in selected_cases],
        "support_seeds": 5,
        "query_points_per_case": 1000,
        "cases": case_reports,
        "aggregate": {
            "sampling_std_over_prediction_spatial_std_mean": float(
                np.mean([row["prediction_sampling_std_over_prediction_spatial_std"] for row in case_reports])
            ),
            "sampling_std_over_truth_spatial_std_mean": float(
                np.mean([row["prediction_sampling_std_over_truth_spatial_std"] for row in case_reports])
            ),
            "ensemble_delta_mean": float(np.mean([row["ensemble_delta"] for row in case_reports])),
            "latent_between_case_std_mean": float(np.mean(np.std(latent_means_array, axis=0))),
            "latent_within_case_support_std_mean": float(np.mean(latent_within_array)),
            "latent_channels_between_case_std_gt_0p01": int(
                np.sum(np.std(latent_means_array, axis=0) > 0.01)
            ),
        },
    }


def run_v3(*, device_name: str = "cuda:0") -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the V3 gradient probe")
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "split_sha256": SPLIT_SHA256,
        "test35_cases_read": 0,
        "support_and_latent": _support_and_latent_probe(device),
        "gradient_cosines": _gradient_probe(device),
        "git": git_state(),
    }
    atomic_write_json(V3_OUTPUT, report)
    return report


def run_bulk_oracle(*, query_points_per_case: int = 1000) -> dict[str, Any]:
    config = ExperimentConfig.from_json(V4_CONFIG_ROOT / "g_raw_s1234.json")
    dataset = VolumeFieldDataset(
        config["paths"]["sidecar_manifest"],
        config["paths"]["field_stats"],
        roles=["val"],
        input_variant=config.input_variant,
        sampling=config["sampling"],
        seed=1234,
    )
    ks = [8, 32, 128, 512, 2048]
    per_k_cases = {k: [] for k in ks}
    pooled = {
        k: {name: RegressionAccumulator() for name in ("u", "v", "w", "speed", "pressure")}
        for k in ks
    }
    for case in dataset.cases:
        strict = np.flatnonzero(~np.asarray(case.is_wall, dtype=bool))
        query_idx = np.asarray(case.fixed_validation_indices[: int(query_points_per_case)], dtype=np.int64)
        coords = np.asarray(case.coords[strict], dtype=np.float32)
        tree = cKDTree(coords)
        distances, local_neighbors = tree.query(
            np.asarray(case.coords[query_idx], dtype=np.float32),
            k=max(ks) + 1,
            workers=-1,
        )
        neighbors = strict[np.asarray(local_neighbors[:, 1:], dtype=np.int64)]
        truth_velocity = np.asarray(case.velocity[query_idx], dtype=np.float64)
        truth_pressure = np.asarray(case.pressure[query_idx], dtype=np.float64)
        for k in ks:
            prediction_velocity = np.mean(
                np.asarray(case.velocity[neighbors[:, :k]], dtype=np.float64), axis=1
            )
            prediction_pressure = np.mean(
                np.asarray(case.pressure[neighbors[:, :k]], dtype=np.float64), axis=1
            )
            truth_values = {
                "u": truth_velocity[:, 0],
                "v": truth_velocity[:, 1],
                "w": truth_velocity[:, 2],
                "speed": np.linalg.norm(truth_velocity, axis=1),
                "pressure": truth_pressure,
            }
            prediction_values = {
                "u": prediction_velocity[:, 0],
                "v": prediction_velocity[:, 1],
                "w": prediction_velocity[:, 2],
                "speed": np.linalg.norm(prediction_velocity, axis=1),
                "pressure": prediction_pressure,
            }
            metrics = {}
            for name in truth_values:
                accumulator = RegressionAccumulator()
                accumulator.update(truth_values[name], prediction_values[name])
                pooled[k][name].update(truth_values[name], prediction_values[name])
                metrics[name] = accumulator.result()
            per_k_cases[k].append({"case_id": case.case_id, "metrics": metrics})
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "route": "volume_uvwp_peak_field_v4",
        "oracle_warning": "reads each target validation case truth; smoothing reference only, not deployable baseline",
        "roles": ["val"],
        "test35_cases_read": 0,
        "query_points_per_case": int(query_points_per_case),
        "results": {
            str(k): {
                "case_balanced": {
                    name: {
                        statistic: float(np.mean([row["metrics"][name][statistic] for row in per_k_cases[k]]))
                        for statistic in ("r2", "mae", "rmse")
                    }
                    for name in ("u", "v", "w", "speed", "pressure")
                },
                "pooled": {name: accumulator.result() for name, accumulator in pooled[k].items()},
                "cases": per_k_cases[k],
            }
            for k in ks
        },
        "split_sha256": SPLIT_SHA256,
        "git": git_state(),
    }
    atomic_write_json(BULK_OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=("manufactured", "v3", "bulk", "all"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    outputs = {}
    if args.section in {"manufactured", "all"}:
        outputs["manufactured"] = run_manufactured()
    if args.section in {"v3", "all"}:
        outputs["v3"] = run_v3(device_name=args.device)
    if args.section in {"bulk", "all"}:
        outputs["bulk"] = run_bulk_oracle()
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
