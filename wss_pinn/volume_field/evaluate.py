"""Full-volume physical-unit evaluation for one saved checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wss_pinn.utils import atomic_write_json, sha256_file, utc_now

from .config import VolumeExperimentConfig
from .data.dataset import VolumeFieldDataset, _take_strict_volume
from .metrics import RegressionAccumulator
from .models import build_volume_model
from .physics.residuals import denormalize_fields, generalized_newtonian_residuals


def _case_seed(case_id: str, seed: int) -> int:
    value = int.from_bytes(hashlib.sha256(case_id.encode("utf-8")).digest()[:8], "little")
    return (int(seed) + value) % (2**63 - 1)


def _to_tensor(value, device, dtype=None):
    tensor = torch.as_tensor(value, device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _case_metrics() -> dict[str, RegressionAccumulator]:
    names = ["u", "v", "w", "speed", "pressure"]
    output = {name: RegressionAccumulator() for name in names}
    for region in ("near_wall", "core"):
        output.update(
            {f"{region}_{name}": RegressionAccumulator() for name in names}
        )
    return output


def _metrics_result(accumulators: dict[str, RegressionAccumulator]) -> dict[str, Any]:
    return {name: accumulator.result() for name, accumulator in accumulators.items()}


def evaluate(
    config: VolumeExperimentConfig,
    checkpoint: str | Path,
    *,
    device_override: str | None = None,
    query_chunk_size: int = 16384,
    protocol: str = "full_volume",
) -> dict[str, Any]:
    if int(query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    if protocol not in {"full_volume", "same5k"}:
        raise ValueError("protocol must be full_volume or same5k")
    device_name = device_override or config["train"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    device = torch.device(device_name)
    dtype = (
        torch.float64
        if config["train"]["precision"] == "float64"
        else torch.float32
    )
    stats_path = Path(config["paths"]["field_stats"])
    field_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    dataset = VolumeFieldDataset(
        config["paths"]["sidecar_manifest"],
        stats_path,
        roles=config["data"]["eval_roles"],
        input_variant=config.input_variant,
        sampling=config["sampling"],
        seed=int(config["train"]["seed"]),
    )
    model = build_volume_model(config).to(device=device, dtype=dtype)
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("route") != "volume_uvwp_peak_v1":
        raise ValueError("checkpoint route mismatch")
    checkpoint_config = payload.get("resolved_config", {})
    for section, key in (
        ("experiment", "id"),
        ("experiment", "mode"),
        ("experiment", "architecture"),
        ("experiment", "input_variant"),
    ):
        if checkpoint_config.get(section, {}).get(key) != config[section][key]:
            raise ValueError(f"checkpoint/config mismatch: {section}.{key}")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    reports = []
    with torch.enable_grad():
        for case_index, case in enumerate(dataset.cases):
            rng = np.random.default_rng(
                _case_seed(case.case_id, int(config["sampling"]["seed"]))
            )
            support_idx = _take_strict_volume(
                rng,
                case.is_wall,
                int(config["sampling"]["support_points"]),
            )
            support_coords = _to_tensor(
                np.asarray(case.coords[support_idx]), device, dtype
            )
            support_features = _to_tensor(
                dataset._features(
                    case.coords[support_idx], case.geometry[support_idx]
                ),
                device,
                dtype,
            )
            support_batch = torch.zeros(
                len(support_idx), dtype=torch.long, device=device
            )
            encoded = model.encode_support(
                support_coords,
                support_features,
                support_batch,
                unit_ids=[case.case_id],
                epoch=0,
                global_seed=int(config["train"]["seed"]),
                evaluation=True,
            )
            strict_indices = np.flatnonzero(~np.asarray(case.is_wall, dtype=bool))
            evaluation_indices = (
                np.asarray(support_idx, dtype=np.int64)
                if protocol == "same5k"
                else strict_indices
            )
            accumulators = _case_metrics()
            predicted_pressure_sum = 0.0
            truth_pressure_sum = 0.0
            for start in range(0, len(evaluation_indices), int(query_chunk_size)):
                indices = evaluation_indices[start : start + int(query_chunk_size)]
                coords = _to_tensor(np.asarray(case.coords[indices]), device, dtype)
                batch = torch.zeros(len(indices), dtype=torch.long, device=device)
                with torch.no_grad():
                    normalized = model.decode_query(encoded, coords, batch)
                    velocity, pressure = denormalize_fields(normalized, field_stats)
                velocity_np = velocity.cpu().numpy()
                pressure_np = pressure.cpu().numpy()
                truth_velocity = np.asarray(case.velocity[indices], dtype=np.float64)
                truth_pressure = np.asarray(case.pressure[indices], dtype=np.float64)
                predicted_pressure_sum += float(
                    np.asarray(pressure_np, dtype=np.float64).sum()
                )
                truth_pressure_sum += float(truth_pressure.sum(dtype=np.float64))
                for component, name in enumerate(("u", "v", "w")):
                    accumulators[name].update(
                        truth_velocity[:, component], velocity_np[:, component]
                    )
                accumulators["speed"].update(
                    np.linalg.norm(truth_velocity, axis=1),
                    np.linalg.norm(velocity_np, axis=1),
                )
                accumulators["pressure"].update(truth_pressure, pressure_np)
                region = np.asarray(case.region[indices])
                for region_value, region_name in ((1, "near_wall"), (0, "core")):
                    mask = region == region_value
                    if not bool(np.any(mask)):
                        continue
                    for component, name in enumerate(("u", "v", "w")):
                        accumulators[f"{region_name}_{name}"].update(
                            truth_velocity[mask, component], velocity_np[mask, component]
                        )
                    accumulators[f"{region_name}_speed"].update(
                        np.linalg.norm(truth_velocity[mask], axis=1),
                        np.linalg.norm(velocity_np[mask], axis=1),
                    )
                    accumulators[f"{region_name}_pressure"].update(
                        truth_pressure[mask], pressure_np[mask]
                    )

            wall_speed_values = []
            for start in range(0, len(case.wall_coords), int(query_chunk_size)):
                coords = _to_tensor(
                    np.asarray(case.wall_coords[start : start + int(query_chunk_size)]),
                    device,
                    dtype,
                )
                batch = torch.zeros(len(coords), dtype=torch.long, device=device)
                with torch.no_grad():
                    normalized = model.decode_query(encoded, coords, batch)
                    velocity, _ = denormalize_fields(normalized, field_stats)
                wall_speed_values.append(
                    torch.linalg.vector_norm(velocity, dim=1).cpu().numpy()
                )
            wall_speed = np.concatenate(wall_speed_values)

            physics_count = min(
                int(config["sampling"]["physics_points"]), len(evaluation_indices)
            )
            physics_indices = np.sort(
                rng.choice(evaluation_indices, size=physics_count, replace=False)
            )
            physics_coords = _to_tensor(
                np.asarray(case.coords[physics_indices]), device, dtype
            ).requires_grad_(True)
            physics_batch = torch.zeros(
                physics_count, dtype=torch.long, device=device
            )
            physics_prediction = model.decode_query(
                encoded, physics_coords, physics_batch
            )
            residuals = generalized_newtonian_residuals(
                physics_prediction,
                physics_coords,
                physics_coords.new_full((physics_count,), case.length_m),
                field_stats=field_stats,
                physics_config=config["physics"],
            )
            metrics = _metrics_result(accumulators)
            reports.append(
                {
                    "case_id": case.case_id,
                    "role": case.role,
                    "strict_volume_points": int(len(strict_indices)),
                    "evaluated_points": int(len(evaluation_indices)),
                    "volume_wall_duplicate_points": int(np.sum(case.is_wall)),
                    "metrics": metrics,
                    "pressure_gauge_diagnostic_pa": {
                        "prediction_mean": predicted_pressure_sum
                        / len(evaluation_indices),
                        "truth_mean": truth_pressure_sum / len(evaluation_indices),
                        "prediction_minus_truth_mean": (
                            predicted_pressure_sum - truth_pressure_sum
                        )
                        / len(evaluation_indices),
                    },
                    "wall_speed_m_s": {
                        "rms": float(np.sqrt(np.mean(np.square(wall_speed)))),
                        "p95": float(np.quantile(wall_speed, 0.95)),
                        "max": float(np.max(wall_speed)),
                    },
                    "physics": {
                        "continuity_rms_dimensionless": float(
                            torch.sqrt(
                                torch.mean(torch.square(residuals["continuity_hat"]))
                            ).detach().cpu()
                        ),
                        "continuity_rms_s_inv": float(
                            torch.sqrt(
                                torch.mean(
                                    torch.square(residuals["continuity_phys_s_inv"])
                                )
                            ).detach().cpu()
                        ),
                        "momentum_rms_dimensionless": [
                            float(
                                torch.sqrt(
                                    torch.mean(
                                        torch.square(
                                            residuals["momentum_hat"][:, component]
                                        )
                                    )
                                ).detach().cpu()
                            )
                            for component in range(3)
                        ],
                        "momentum_rms_pa_per_m": float(
                            torch.sqrt(
                                torch.mean(
                                    torch.square(residuals["momentum_phys_pa_per_m"])
                                )
                            ).detach().cpu()
                        ),
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "volume_case_evaluated",
                        "index": case_index + 1,
                        "total": len(dataset.cases),
                        "case_id": case.case_id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    scalar_paths = {
        "velocity_u_r2": lambda row: row["metrics"]["u"]["r2"],
        "velocity_v_r2": lambda row: row["metrics"]["v"]["r2"],
        "velocity_w_r2": lambda row: row["metrics"]["w"]["r2"],
        "speed_r2": lambda row: row["metrics"]["speed"]["r2"],
        "pressure_r2": lambda row: row["metrics"]["pressure"]["r2"],
        "wall_speed_rms_m_s": lambda row: row["wall_speed_m_s"]["rms"],
        "continuity_rms_dimensionless": lambda row: row["physics"][
            "continuity_rms_dimensionless"
        ],
        "momentum_rms_pa_per_m": lambda row: row["physics"][
            "momentum_rms_pa_per_m"
        ],
        "momentum_x_rms_dimensionless": lambda row: row["physics"][
            "momentum_rms_dimensionless"
        ][0],
        "momentum_y_rms_dimensionless": lambda row: row["physics"][
            "momentum_rms_dimensionless"
        ][1],
        "momentum_z_rms_dimensionless": lambda row: row["physics"][
            "momentum_rms_dimensionless"
        ][2],
        "wall_speed_p95_m_s": lambda row: row["wall_speed_m_s"]["p95"],
        "wall_speed_max_m_s": lambda row: row["wall_speed_m_s"]["max"],
    }
    regression_names = list(_case_metrics())
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_v1",
        "evaluation_protocol": protocol,
        "support_query_relationship": (
            "same_fixed_5k" if protocol == "same5k" else "fixed_5k_to_full_volume"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_label": "reused_development_screen_train138_test35",
        "pressure_metric": "strict-volume case-mean-centered relative pressure in Pa",
        "cases": reports,
        "case_balanced": {
            name: float(np.mean([extract(row) for row in reports]))
            for name, extract in scalar_paths.items()
        },
        "case_balanced_regression": {
            metric_name: {
                statistic: float(
                    np.nanmean(
                        [
                            row["metrics"][metric_name][statistic]
                            for row in reports
                        ]
                    )
                )
                for statistic in ("r2", "mae", "rmse")
            }
            for metric_name in regression_names
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_total")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--query-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--protocol", choices=("full_volume", "same5k"), default="full_volume"
    )
    parser.add_argument("--output-name")
    args = parser.parse_args()
    config = VolumeExperimentConfig.from_json(args.config)
    checkpoint = Path(args.checkpoint)
    if args.checkpoint in {"best_data", "best_total", "last"}:
        checkpoint = config.run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    report = evaluate(
        config,
        checkpoint,
        device_override=args.device,
        query_chunk_size=args.query_chunk_size,
        protocol=args.protocol,
    )
    output = atomic_write_json(
        config.run_dir / (args.output_name or f"evaluation_{checkpoint.stem}.json"),
        report,
    )
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
