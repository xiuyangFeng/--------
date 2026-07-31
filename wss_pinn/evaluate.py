"""评估入口：加载 checkpoint，按病例报告 WSS / 速度 / 压力 / 物理残差指标。

学习要点
--------
评估与训练共用同一模型与 sidecar 数据，但：

- 病例角色用 ``data.eval_roles``（常含 test）；
- WSS 从网络的 ``log_wss_direct`` 用 ``expm1`` 还原到 Pa；
- 速度/压力乘回病例特征尺度，再与真值比；
- 可选读取 ``control_run_dir`` 的父评估报告，输出配对 Δ（F0-UP/F1 科学门禁用）。

命令行示例::

    python -m wss_pinn.evaluate --config path/to/config.json --checkpoint best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data.dataset import PhysicsDataset, as_tensor
from .physics.residuals import continuity_residual
from .train import make_model, resolve_device
from .utils import atomic_write_json, sha256_file, utc_now


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """通用回归指标：R²、MAE、RMSE、相对 MAE 归一的 NRMSE。"""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    # 总平方和；常数真值时分母用 1e-12 防止除零
    denominator = float(np.sum(np.square(truth - truth.mean())))
    rmse = float(np.sqrt(np.mean(np.square(truth - prediction))))
    return {
        "r2": 1.0 - float(np.sum(np.square(truth - prediction))) / max(denominator, 1e-12),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "rmse": rmse,
        "nrmse": rmse / max(float(np.mean(np.abs(truth))), 1e-12),
    }


def top10_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """高 WSS 区域指标：真值/预测各自取 ≥P90 的掩膜。

    - ``top10_iou``：两掩膜交并比，衡量「热点位置」是否对齐；
    - ``high_wss_nrmse``：只在真值热点上算 RMSE，再除以该区 |WSS| 均值。
    """
    threshold_true = np.quantile(truth, 0.9)
    threshold_pred = np.quantile(prediction, 0.9)
    true_mask = truth >= threshold_true
    pred_mask = prediction >= threshold_pred
    intersection = int(np.sum(true_mask & pred_mask))
    union = int(np.sum(true_mask | pred_mask))
    high_rmse = float(np.sqrt(np.mean(np.square(truth[true_mask] - prediction[true_mask]))))
    return {
        "top10_iou": intersection / max(union, 1),
        "high_wss_nrmse": high_rmse / max(float(np.mean(np.abs(truth[true_mask]))), 1e-12),
    }


def evaluate(config: ExperimentConfig, checkpoint: Path, device_name: str | None) -> dict:
    """对 eval 角色下所有病例跑一遍推理，组装评估报告 dict。"""
    device = resolve_device(config["train"]["device"], device_name)
    dtype = torch.float64 if config["train"]["precision"] == "float64" else torch.float32
    model = make_model(config, device, dtype)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    dataset = PhysicsDataset(
        config["sampling"]["manifest_path"],
        verify=True,
        roles=config["data"]["eval_roles"],
    )
    cases = []
    for case in dataset.cases:
        descriptor = as_tensor(case.static["geometry_descriptor"], device, dtype)
        wall_coords = as_tensor(case.static["wall_coords"], device, dtype)

        # 近壁 + 核心拼成体域评估点（与训练 interior 定义一致）
        interior_coords = np.concatenate(
            [case.static["near_wall_coords"], case.static["core_coords"]]
        )
        velocity_truth = np.concatenate(
            [case.fields["near_wall_velocity"], case.fields["core_velocity"]]
        )
        pressure_truth = np.concatenate(
            [case.fields["near_wall_pressure"], case.fields["core_pressure"]]
        )

        with torch.no_grad():
            wall_output = model(wall_coords, descriptor)
            velocity_output = model(
                as_tensor(interior_coords, device, dtype), descriptor
            )

        # log1p 训练 → expm1 还原；clamp 防止数值噪声出负 WSS
        wss_prediction = torch.expm1(wall_output["log_wss_direct"]).clamp_min(0)
        wss_prediction_np = wss_prediction.cpu().numpy()
        wss_truth = case.fields["wall_wss"]

        # 网络输出是无量纲场，乘病例尺度回到 m/s、Pa
        velocity_prediction = velocity_output["velocity"].cpu().numpy() * case.velocity_scale
        pressure_prediction = velocity_output["pressure"].cpu().numpy() * case.pressure_scale
        # 规范不变：两侧各自去均值再比
        pressure_prediction -= pressure_prediction.mean()
        pressure_truth = pressure_truth - pressure_truth.mean()

        # 连续性残差只需子集（autograd 贵）；最多 1024 个体点
        derivative_count = min(1024, len(interior_coords))
        derivative_coords = as_tensor(
            interior_coords[:derivative_count], device, dtype
        ).requires_grad_(True)
        derivative_output = model(derivative_coords, descriptor)
        divergence = continuity_residual(
            derivative_output["velocity"], derivative_coords, create_graph=False
        )

        velocity_metrics = regression_metrics(velocity_truth, velocity_prediction)
        # 分量级与速度模长，便于诊断某一方向是否崩
        velocity_metrics["components"] = {
            name: regression_metrics(velocity_truth[:, index], velocity_prediction[:, index])
            for index, name in enumerate(("u", "v", "w"))
        }
        velocity_metrics["components"]["speed"] = regression_metrics(
            np.linalg.norm(velocity_truth, axis=1),
            np.linalg.norm(velocity_prediction, axis=1),
        )

        case_report = {
            "case_id": case.case_id,
            "cohort": case.cohort,
            "role": case.role,
            "wss": {
                **regression_metrics(wss_truth, wss_prediction_np),
                **top10_metrics(wss_truth, wss_prediction_np),
            },
            "velocity": velocity_metrics,
            "pressure_gauge_invariant": regression_metrics(
                pressure_truth, pressure_prediction
            ),
            # 无量纲连续性 RMS、壁面速度 RMS（无滑移诊断）
            "continuity_rms_dimensionless": float(
                torch.sqrt(torch.mean(torch.square(divergence))).detach().cpu()
            ),
            "no_slip_rms_dimensionless": float(
                torch.sqrt(torch.mean(torch.square(wall_output["velocity"]))).cpu()
            ),
        }
        cases.append(case_report)

    # 病例均权汇总（不是按点数加权）
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "stage": config.stage,
        "split_label": config["data"]["split_label"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cases": cases,
        "case_balanced": {
            metric: float(np.mean([case["wss"][metric] for case in cases]))
            for metric in ("r2", "mae", "rmse", "high_wss_nrmse", "top10_iou")
        },
    }
    # 按 train/test 角色再各算一份均值
    report["by_role"] = {
        role: {
            metric: float(
                np.mean(
                    [
                        case["wss"][metric]
                        for case in cases
                        if case["role"] == role
                    ]
                )
            )
            for metric in ("r2", "mae", "rmse", "high_wss_nrmse", "top10_iou")
        }
        for role in sorted({case["role"] for case in cases})
    }

    # 若配置了对照 run，按 case_id 对齐算 Δ（正负含义：本 run − 对照）
    control_dir = config["experiment"].get("control_run_dir")
    if control_dir:
        control_path = Path(control_dir)
        if not control_path.is_absolute():
            control_path = Path(__file__).resolve().parents[1] / control_path
        control_report_path = control_path / f"evaluation_{checkpoint.stem}.json"
        if control_report_path.exists():
            control = json.loads(control_report_path.read_text(encoding="utf-8"))
            control_cases = {row["case_id"]: row for row in control["cases"]}
            paired = []
            for row in cases:
                parent = control_cases.get(row["case_id"])
                if parent is None:
                    continue
                paired.append(
                    {
                        "case_id": row["case_id"],
                        "delta_wss_r2": row["wss"]["r2"] - parent["wss"]["r2"],
                        "delta_wss_mae": row["wss"]["mae"] - parent["wss"]["mae"],
                        "delta_high_wss_nrmse": (
                            row["wss"]["high_wss_nrmse"]
                            - parent["wss"]["high_wss_nrmse"]
                        ),
                        "delta_top10_iou": (
                            row["wss"]["top10_iou"] - parent["wss"]["top10_iou"]
                        ),
                        "delta_velocity_r2": (
                            row["velocity"]["r2"] - parent["velocity"]["r2"]
                        ),
                        "delta_pressure_r2": (
                            row["pressure_gauge_invariant"]["r2"]
                            - parent["pressure_gauge_invariant"]["r2"]
                        ),
                        "delta_continuity_rms": (
                            row["continuity_rms_dimensionless"]
                            - parent["continuity_rms_dimensionless"]
                        ),
                        "delta_no_slip_rms": (
                            row["no_slip_rms_dimensionless"]
                            - parent["no_slip_rms_dimensionless"]
                        ),
                    }
                )
            report["paired_control"] = {
                "path": str(control_report_path.resolve()),
                "cases": paired,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    checkpoint = Path(args.checkpoint)
    # 允许传 best/last 别名，落到 run_dir/checkpoints/
    if args.checkpoint in {"best", "last"}:
        checkpoint = config.run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    report = evaluate(config, checkpoint, args.device)
    output = atomic_write_json(
        config.run_dir / f"evaluation_{Path(args.checkpoint).stem}.json", report
    )
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
