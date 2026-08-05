"""单 checkpoint 的物理单位评估（全体域或 SAME5K 协议）。

学习要点
--------
1. **两种协议**：
   - ``full_volume``：固定 5k support → 严格体域全部 query（副协议）；
   - ``same5k``：support 与评估点为同一组 5k（主协议，与训练采样对齐）。
2. **指标范围**：``u/v/w/speed/p`` 的 R²/MAE/RMSE，近壁/核心分区，压力
   gauge 诊断，壁面速度 RMS/p95/max，以及 continuity/momentum 残差。
3. **汇总口径**：``case_balanced`` 对每例标量先算再对病例取均值，避免大病例
   点权淹没小病例。
4. **不得只看 total loss**：物理一致性与速度泛化必须分开报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wss_pinn.utils import atomic_write_json, sha256_file, utc_now

from .config import ExperimentConfig
from .data.dataset import VolumeFieldDataset, _take_strict_volume
from .metrics import RegressionAccumulator
from .models import build_model
from .physics.residuals import denormalize_fields, generalized_newtonian_residuals


def _case_seed(case_id: str, seed: int) -> int:
    """由 case_id + 全局 seed 派生稳定的病例级 RNG 种子（SHA256 前 8 字节）。"""
    value = int.from_bytes(hashlib.sha256(case_id.encode("utf-8")).digest()[:8], "little")
    return (int(seed) + value) % (2**63 - 1)


def _to_tensor(value, device, dtype=None):
    """NumPy/标量 → 指定设备（可选精度）的 Tensor。"""
    tensor = torch.as_tensor(value, device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _case_metrics() -> dict[str, RegressionAccumulator]:
    """为单例评估创建全体域 + 近壁 + 核心 的回归累加器集合。"""
    names = ["u", "v", "w", "speed", "pressure"]
    output = {name: RegressionAccumulator() for name in names}
    for region in ("near_wall", "core"):
        output.update(
            {f"{region}_{name}": RegressionAccumulator() for name in names}
        )
    return output


def _metrics_result(accumulators: dict[str, RegressionAccumulator]) -> dict[str, Any]:
    """把累加器字典收成可 JSON 序列化的结果字典。"""
    return {name: accumulator.result() for name, accumulator in accumulators.items()}


def _estimated_outward_normal(
    surface_coords: np.ndarray,
    case_center: np.ndarray,
) -> np.ndarray:
    """Estimate a planar outlet normal and orient it away from the case centre.

    The frozen V3 boundary asset stores outlet vertices and exact pressure targets,
    but not face normals/area weights.  PCA therefore provides a reproducible
    diagnostic normal.  It is suitable for a predicted-flow/mass-balance audit,
    not for claiming an exact Fluent outlet-flow target.
    """
    coords = np.asarray(surface_coords, dtype=np.float64)
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(right[-1], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-30)
    outward_hint = coords.mean(axis=0) - np.asarray(case_center, dtype=np.float64)
    if float(np.dot(normal, outward_hint)) < 0.0:
        normal *= -1.0
    return normal


def _boundary_velocity_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    """Physical-unit inlet velocity errors without using ill-defined constant R²."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    output: dict[str, Any] = {}
    for component, name in enumerate(("u", "v", "w")):
        accumulator = RegressionAccumulator()
        accumulator.update(truth[:, component], prediction[:, component])
        output[name] = accumulator.result()
    accumulator = RegressionAccumulator()
    accumulator.update(
        np.linalg.norm(truth, axis=1), np.linalg.norm(prediction, axis=1)
    )
    output["speed"] = accumulator.result()
    vector_error = prediction - truth
    output["vector_rmse_m_s"] = float(
        np.sqrt(np.mean(np.sum(np.square(vector_error), axis=1)))
    )
    return output


def _predict_points(
    model,
    encoded: dict[str, torch.Tensor],
    coords: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    field_stats: dict[str, Any],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a point array in chunks and return physical velocity/pressure."""
    velocities = []
    pressures = []
    for start in range(0, len(coords), int(chunk_size)):
        query = _to_tensor(np.asarray(coords[start : start + chunk_size]), device, dtype)
        batch = torch.zeros(len(query), dtype=torch.long, device=device)
        with torch.no_grad():
            normalized = model.decode_query(encoded, query, batch)
            velocity, pressure = denormalize_fields(normalized, field_stats)
        velocities.append(velocity.cpu().numpy())
        pressures.append(pressure.cpu().numpy())
    return np.concatenate(velocities), np.concatenate(pressures)


def _update_metrics(
    case_metrics: dict[str, RegressionAccumulator],
    pooled_metrics: dict[str, RegressionAccumulator],
    truth_velocity: np.ndarray,
    predicted_velocity: np.ndarray,
    truth_pressure: np.ndarray,
    predicted_pressure: np.ndarray,
    region: np.ndarray,
) -> None:
    """Update full-volume and near-wall/core regression metrics."""
    truth_speed = np.linalg.norm(truth_velocity, axis=1)
    predicted_speed = np.linalg.norm(predicted_velocity, axis=1)
    pairs = (
        ("u", truth_velocity[:, 0], predicted_velocity[:, 0]),
        ("v", truth_velocity[:, 1], predicted_velocity[:, 1]),
        ("w", truth_velocity[:, 2], predicted_velocity[:, 2]),
        ("speed", truth_speed, predicted_speed),
        ("pressure", truth_pressure, predicted_pressure),
    )
    for name, truth, prediction in pairs:
        case_metrics[name].update(truth, prediction)
        pooled_metrics[name].update(truth, prediction)

    for region_value, region_name in ((1, "near_wall"), (0, "core")):
        mask = region == region_value
        if not bool(np.any(mask)):
            continue
        for name, truth, prediction in pairs:
            key = f"{region_name}_{name}"
            case_metrics[key].update(truth[mask], prediction[mask])
            pooled_metrics[key].update(truth[mask], prediction[mask])


def _evaluate_boundary(
    case,
    model,
    encoded: dict[str, torch.Tensor],
    case_center: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    field_stats: dict[str, Any],
    chunk_size: int,
) -> dict[str, Any] | None:
    """Evaluate the frozen V3 inlet/outlet boundary assets when present."""
    if case.boundary_manifest is None:
        return None

    inlet_velocity, _ = _predict_points(
        model,
        encoded,
        case.inlet_coords,
        device=device,
        dtype=dtype,
        field_stats=field_stats,
        chunk_size=chunk_size,
    )
    truth_inlet_velocity = np.asarray(case.inlet_velocity, dtype=np.float64)
    inlet_manifest = case.boundary_manifest["inlet"]
    inlet_normal = np.asarray(
        inlet_manifest["registered_inward_normal"], dtype=np.float64
    )
    inlet_area = float(inlet_manifest["area_m2"])
    target_inlet_flow = float(inlet_manifest["peak_flow_m3_s"])
    predicted_inlet_flow = inlet_area * float(np.mean(inlet_velocity @ inlet_normal))

    outlet_velocity, outlet_pressure = _predict_points(
        model,
        encoded,
        case.outlet_coords,
        device=device,
        dtype=dtype,
        field_stats=field_stats,
        chunk_size=chunk_size,
    )
    truth_outlet_pressure = np.asarray(case.outlet_pressure, dtype=np.float64)
    zones = []
    predicted_total_outlet_flow = 0.0
    for zone_index, zone_name in enumerate(case.outlet_zone_names):
        mask = np.asarray(case.outlet_zone) == zone_index
        zone_coords = np.asarray(case.outlet_coords[mask], dtype=np.float64)
        zone_prediction = np.asarray(outlet_pressure[mask], dtype=np.float64)
        zone_truth = np.asarray(truth_outlet_pressure[mask], dtype=np.float64)
        error = zone_prediction - zone_truth
        outlet_manifest = case.boundary_manifest["outlets"][zone_name]
        area = float(outlet_manifest["area_m2"])
        normal = _estimated_outward_normal(zone_coords, case_center)
        predicted_flow = area * float(np.mean(outlet_velocity[mask] @ normal))
        predicted_total_outlet_flow += predicted_flow
        zones.append(
            {
                "zone": zone_name,
                "points": int(np.sum(mask)),
                "area_m2": area,
                "target_pressure_relative_pa": float(np.mean(zone_truth)),
                "prediction_pressure_mean_pa": float(np.mean(zone_prediction)),
                "pressure_mean_error_pa": float(np.mean(error)),
                "pressure_mae_pa": float(np.mean(np.abs(error))),
                "pressure_rmse_pa": float(np.sqrt(np.mean(np.square(error)))),
                "estimated_outward_normal": normal.tolist(),
                "predicted_flow_m3_s": predicted_flow,
            }
        )

    outlet_error = outlet_pressure - truth_outlet_pressure
    flow_balance_error = predicted_total_outlet_flow - target_inlet_flow
    return {
        "inlet": {
            "points": int(len(case.inlet_coords)),
            "velocity": _boundary_velocity_metrics(
                truth_inlet_velocity, inlet_velocity
            ),
            "target_flow_m3_s": target_inlet_flow,
            "predicted_flow_m3_s": predicted_inlet_flow,
            "flow_error_m3_s": predicted_inlet_flow - target_inlet_flow,
            "flow_absolute_relative_error": abs(
                predicted_inlet_flow - target_inlet_flow
            )
            / max(abs(target_inlet_flow), 1e-30),
        },
        "outlet": {
            "pressure_mae_pa": float(np.mean(np.abs(outlet_error))),
            "pressure_rmse_pa": float(np.sqrt(np.mean(np.square(outlet_error)))),
            "predicted_total_flow_m3_s": predicted_total_outlet_flow,
            "target_total_flow_proxy_m3_s": target_inlet_flow,
            "mass_balance_error_m3_s": flow_balance_error,
            "mass_balance_absolute_relative_error": abs(flow_balance_error)
            / max(abs(target_inlet_flow), 1e-30),
            "flow_method": (
                "PCA outlet normal oriented away from strict-volume case centre; "
                "diagnostic only because exact outlet flow targets and face area "
                "weights were not frozen"
            ),
            "zones": zones,
        },
    }


def _evaluate_physics(
    case,
    model,
    encoded: dict[str, torch.Tensor],
    evaluation_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    device: torch.device,
    dtype: torch.dtype,
    field_stats: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    count = min(int(config["sampling"]["physics_points"]), len(evaluation_indices))
    indices = np.sort(rng.choice(evaluation_indices, size=count, replace=False))
    coords = _to_tensor(np.asarray(case.coords[indices]), device, dtype).requires_grad_(True)
    batch = torch.zeros(count, dtype=torch.long, device=device)
    prediction = model.decode_query(encoded, coords, batch)
    residuals = generalized_newtonian_residuals(
        prediction,
        coords,
        coords.new_full((count,), case.length_m),
        field_stats=field_stats,
        physics_config=config["physics"],
    )

    def rms(value: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(torch.square(value))).detach().cpu())

    return {
        "continuity_rms_dimensionless": rms(residuals["continuity_hat"]),
        "continuity_rms_s_inv": rms(residuals["continuity_phys_s_inv"]),
        "momentum_rms_dimensionless": [
            rms(residuals["momentum_hat"][:, component]) for component in range(3)
        ],
        "momentum_rms_pa_per_m": rms(residuals["momentum_phys_pa_per_m"]),
    }


def evaluate(
    config: ExperimentConfig,
    checkpoint: str | Path,
    *,
    device_override: str | None = None,
    query_chunk_size: int = 16384,
    protocol: str = "full_volume",
) -> dict[str, Any]:
    """对单个 checkpoint 跑完整评估并返回报告字典。

    Parameters
    ----------
    config :
        须与 checkpoint 内嵌的 ``resolved_config`` 在 id/mode/架构/输入上一致。
    checkpoint :
        ``.pt`` 路径。
    query_chunk_size :
        解码分块大小，控制显存峰值。
    protocol :
        ``full_volume`` 或 ``same5k``。
    """
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

    # ---- 加载统计、评估集、模型与 checkpoint ----
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
    model = build_model(config).to(device=device, dtype=dtype)
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("route") != config.route:
        raise ValueError("checkpoint route mismatch")
    # 防止拿错臂的权重评估到当前配置上
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
    pooled_accumulators = _case_metrics()
    # enable_grad：物理残差需要对 query 坐标求导；场值解码本身用 no_grad
    with torch.enable_grad():
        for case_index, case in enumerate(dataset.cases):
            rng = np.random.default_rng(
                _case_seed(case.case_id, int(config["sampling"]["seed"]))
            )
            # 支撑点：严格体域（排除壁面重复行）均匀抽取
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
            # evaluation=True：编码器侧采样走评估确定性路径
            encoded = model.encode_support(
                support_coords,
                support_features,
                support_batch,
                unit_ids=[case.case_id],
                epoch=0,
                global_seed=int(config["train"]["seed"]),
                evaluation=True,
            )

            # same5k：评估点 = support；full_volume：全部严格体域点
            strict_indices = np.flatnonzero(~np.asarray(case.is_wall, dtype=bool))
            evaluation_indices = (
                np.asarray(support_idx, dtype=np.int64)
                if protocol == "same5k"
                else strict_indices
            )
            accumulators = _case_metrics()
            predicted_pressure_sum = 0.0
            truth_pressure_sum = 0.0
            evaluated_coord_sum = np.zeros(3, dtype=np.float64)

            # ---- 分块解码场量，累计回归指标与压力均值诊断 ----
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
                evaluated_coord_sum += np.asarray(
                    case.coords[indices], dtype=np.float64
                ).sum(axis=0)
                predicted_pressure_sum += float(
                    np.asarray(pressure_np, dtype=np.float64).sum()
                )
                truth_pressure_sum += float(truth_pressure.sum(dtype=np.float64))
                _update_metrics(
                    accumulators,
                    pooled_accumulators,
                    truth_velocity,
                    velocity_np,
                    truth_pressure,
                    pressure_np,
                    np.asarray(case.region[indices]),
                )

            # ---- 独立壁面云上的速度幅值（no-slip 诊断）----
            wall_velocity, _ = _predict_points(
                model,
                encoded,
                case.wall_coords,
                device=device,
                dtype=dtype,
                field_stats=field_stats,
                chunk_size=query_chunk_size,
            )
            wall_speed = np.linalg.norm(wall_velocity, axis=1)

            # ---- V3 exact peak boundary diagnostics ----
            boundary_report = _evaluate_boundary(
                case,
                model,
                encoded,
                evaluated_coord_sum / len(evaluation_indices),
                device=device,
                dtype=dtype,
                field_stats=field_stats,
                chunk_size=query_chunk_size,
            )

            # ---- 物理残差：从评估点中再抽 physics_points 个配点 ----
            physics_report = _evaluate_physics(
                case,
                model,
                encoded,
                evaluation_indices,
                rng,
                device=device,
                dtype=dtype,
                field_stats=field_stats,
                config=config,
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
                    # 预测与真值压力均值差：检查 gauge 是否漂
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
                    "boundary": boundary_report,
                    "physics": physics_report,
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

    # ---- 病例均衡汇总：每例一个标量 → 对病例取均值 ----
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
    if config.route == "volume_uvwp_peak_qs_smooth_v3":
        scalar_paths.update(
            {
                "inlet_velocity_vector_rmse_m_s": lambda row: row["boundary"][
                    "inlet"
                ]["velocity"]["vector_rmse_m_s"],
                "inlet_flow_absolute_relative_error": lambda row: row["boundary"][
                    "inlet"
                ]["flow_absolute_relative_error"],
                "outlet_pressure_mae_pa": lambda row: row["boundary"]["outlet"][
                    "pressure_mae_pa"
                ],
                "outlet_pressure_rmse_pa": lambda row: row["boundary"]["outlet"][
                    "pressure_rmse_pa"
                ],
                "outlet_mass_balance_absolute_relative_error": lambda row: row[
                    "boundary"
                ]["outlet"]["mass_balance_absolute_relative_error"],
            }
        )
    regression_names = list(_case_metrics())
    speed_variance_ratios = np.asarray(
        [
            row["metrics"]["speed"]["prediction_to_truth_variance_ratio"]
            for row in reports
        ],
        dtype=np.float64,
    )
    pressure_variance_ratios = np.asarray(
        [
            row["metrics"]["pressure"]["prediction_to_truth_variance_ratio"]
            for row in reports
        ],
        dtype=np.float64,
    )
    truth_case_speed_means = np.asarray(
        [row["metrics"]["speed"]["truth_mean"] for row in reports],
        dtype=np.float64,
    )
    predicted_case_speed_means = np.asarray(
        [row["metrics"]["speed"]["prediction_mean"] for row in reports],
        dtype=np.float64,
    )
    report = {
        "schema_version": 2,
        "created_at": utc_now(),
        "status": "completed",
        "route": config.route,
        "experiment_id": config["experiment"]["id"],
        "mode": config.mode,
        "input_variant": config.input_variant,
        "evaluation_protocol": protocol,
        "support_query_relationship": (
            "same_fixed_5k" if protocol == "same5k" else "fixed_5k_to_full_volume"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_kind": checkpoint.stem,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_label": (
            "frozen_train123_val15_test35_s1234"
            if config.route == "volume_uvwp_peak_qs_smooth_v3"
            else "reused_development_screen_train138_test35"
        ),
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
        "pooled_regression": _metrics_result(pooled_accumulators),
        "collapse_diagnostic": {
            "speed_cases_prediction_variance_below_1pct_truth": int(
                np.sum(speed_variance_ratios < 0.01)
            ),
            "pressure_cases_prediction_variance_below_1pct_truth": int(
                np.sum(pressure_variance_ratios < 0.01)
            ),
            "speed_prediction_to_truth_variance_ratio_median": float(
                np.median(speed_variance_ratios)
            ),
            "pressure_prediction_to_truth_variance_ratio_median": float(
                np.median(pressure_variance_ratios)
            ),
            "between_case_speed_mean_variance_ratio": float(
                np.var(predicted_case_speed_means)
                / max(float(np.var(truth_case_speed_means)), 1e-30)
            ),
        },
    }
    return report


def main() -> None:
    """CLI：评估指定 checkpoint，结果写到 run_dir 下 JSON。

    ``--checkpoint`` 可为路径，或简写 ``best_data`` / ``best_total`` / ``last``
    （自动解析为 ``run_dir/checkpoints/<name>.pt``）。
    """
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
    config = ExperimentConfig.from_json(args.config)
    checkpoint = Path(args.checkpoint)
    aliases = {"best_data", "best_total", "best_validation_data", "best_validation_total", "last"}
    if args.checkpoint in aliases:
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
