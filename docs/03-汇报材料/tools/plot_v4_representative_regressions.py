#!/usr/bin/env python3
"""Generate V4 representative speed/WSS regression triptychs.

The script mirrors the V2/V3 ``speed_representative_regressions.png`` and
``wss_representative_regressions.png`` products.  It reads the eight V4
BC+PDE-FIXED / BC+PDE-EMA arms selected from the current workbook, chooses
worst/most-frequent/best cases using case-wise fitted-line R², reruns only
those three speed cases from each checkpoint, and reuses the existing V4
Profile-Secant WSS caches for WSS plots.

It is intentionally report-only: no workbook, checkpoint, training output,
or cached evaluation result is modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
V4_BASE = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4"
V4_AUDIT = ROOT / "outputs/wss_pinn/audits/v4_workbook_0_14_20260823/arms"
PLOT_ROOT = ROOT / "docs/03-汇报材料/V4_BC-PDE_FIX_EMA_横向R2散点图_2026-09-01"
ROI_DIR = ROOT / "outputs/wss_pinn/audits/v2_v3_linear_regression_fullpoints_20260807/anatomical_roi"
BASE_V4_WORKBOOK_TOOL = ROOT / "docs/03-汇报材料/tools/update_wss_pinn_v4_workbook.py"
BASE_V2_WORKBOOK_TOOL = ROOT / "docs/03-汇报材料/tools/update_wss_pinn_v2_v3_workbook.py"
BASE_LINEAR_TOOL = ROOT / "docs/03-汇报材料/tools/add_wss_pinn_v2_v3_linear_regression.py"

TARGET_INDICES = (2, 3, 6, 7, 10, 11, 14, 15)
TIME_SHORT = {"steady_peak": "SP", "transient_81": "TR"}
BACKBONE_SHORT = {"pointnet": "PN", "pointnetpp": "PNPP"}
MODE_SHORT = {"bc_pde_fixed": "BC-PDE-FIXED", "bc_pde_ema": "BC-PDE-EMA"}
ROLE_LABELS = {"worst": "Worst", "most_frequent": "Most Frequent", "best": "Best"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4WB = _load_module("v4_workbook_tool_for_representatives", BASE_V4_WORKBOOK_TOOL)
V2WB = _load_module("v2_workbook_tool_for_representatives", BASE_V2_WORKBOOK_TOOL)
LINEAR = _load_module("linear_tool_for_representatives", BASE_LINEAR_TOOL)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_case(case_id: str) -> str:
    return case_id.replace("/", "__")


def _arm_name(config: dict[str, Any]) -> str:
    temporal = TIME_SHORT[str(config["experiment"]["temporal_mode"])]
    backbone = BACKBONE_SHORT[str(config["experiment"]["backbone"])]
    mode = MODE_SHORT[str(config["experiment"]["training_mode"])]
    return f"V4-{temporal}-{backbone}-{mode}"


def _speed_fit_r2(speed: dict[str, Any]) -> float:
    vx = float(speed["truth_variance"])
    vy = float(speed["prediction_variance"])
    delta = float(speed["prediction_mean"]) - float(speed["truth_mean"])
    raw_r2 = float(speed["r2"])
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    mse = (1.0 - raw_r2) * vx
    covariance = (vx + vy + delta * delta - mse) / 2.0
    return float(np.clip(covariance * covariance / (vx * vy), 0.0, 1.0))


def _representative_speed_cases(run: str, temporal: str) -> dict[str, str]:
    path = V4_BASE / ("steady_peak" if temporal == "steady_peak" else "transient_autograd") / run / "evaluation_official_last_converged_full.json"
    report = _json(path)
    rows = []
    for case in report["cases"]:
        speed = case["metrics"]["speed"]
        rows.append({"case_id": str(case["case_id"]), "speed": {"regression_r2": _speed_fit_r2(speed)}})
    return LINEAR._representatives(rows, "speed")


def _representative_wss_cases(run: str) -> tuple[dict[str, str], dict[str, dict[str, np.ndarray]]]:
    cache_dir = V4_AUDIT / run / "point_cache_s1200"
    _, arrays_by_case = V2WB._apply_calibrator(cache_dir)
    rows = []
    for case_id, arrays in arrays_by_case.items():
        fit = LINEAR.linear_fit(arrays["truth_mag"], arrays["pred_mag"])
        rows.append({"case_id": case_id, "wss": fit.as_dict()})
    return LINEAR._representatives(rows, "wss"), arrays_by_case


def _encode_case(model: Any, dataset: Any, config: dict[str, Any], arrays: Any, device: torch.device):
    support_index = arrays.sample_indices("eval_support", int(config["sampling"]["support_points"]))
    support_coords = torch.as_tensor(np.asarray(arrays.coords[support_index], dtype=np.float32), device=device)
    support_features = torch.as_tensor(arrays.support_features(support_index), device=device)
    support_batch = torch.zeros(len(support_index), dtype=torch.long, device=device)
    bc_vector = torch.as_tensor(arrays.bc_vector, dtype=torch.float32, device=device).unsqueeze(0)
    return model.encode_support(
        support_coords,
        support_features,
        support_batch,
        bc_vector,
        [arrays.case_id],
        int(config["train"]["seed"]),
    )


def _predict_peak_speed(
    model: Any,
    dataset: Any,
    config: dict[str, Any],
    case_index: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from wss_pinn.v4.evaluate import _CaseArrays, denormalize_prediction

    arrays = _CaseArrays(dataset, case_index)
    v4_case = dataset._case(case_index)
    volume = V4WB._volume_case(v4_case)
    query_time = None
    if str(config["experiment"]["temporal_mode"]) != "steady_peak":
        query_time = V4WB._peak_time_s(v4_case, volume.peak_step)
    encoded = _encode_case(model, dataset, config, arrays, device)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(volume.coords), int(chunk_size)):
            coords = np.asarray(volume.coords[start : start + chunk_size], dtype=np.float32)
            query = torch.as_tensor(coords, device=device)
            batch = torch.zeros(len(query), dtype=torch.long, device=device)
            time_tensor = None
            if query_time is not None:
                time_tensor = torch.full((len(query), 1), float(query_time), dtype=torch.float32, device=device)
            normalized = model.decode_query(encoded, query, batch, time_tensor)
            velocity, _ = denormalize_prediction(normalized.detach().cpu().numpy(), dataset.field_stats)
            chunks.append(np.asarray(velocity, dtype=np.float32))
    predicted_velocity = np.concatenate(chunks, axis=0)
    with np.load(ROI_DIR / f"{_safe_case(arrays.case_id)}.npz", allow_pickle=False) as source:
        anatomical = np.asarray(source["anatomical_indices"], dtype=np.int64)
    truth_speed = np.linalg.norm(np.asarray(volume.velocity[anatomical], dtype=np.float64), axis=1)
    predicted_speed = np.linalg.norm(predicted_velocity[anatomical], axis=1)
    return truth_speed, predicted_speed


def _write_density(path: Path, truth: np.ndarray, prediction: np.ndarray, *, case_id: str, arm: str, metric: str, unit: str, scope: str) -> dict[str, Any]:
    fit = LINEAR.linear_fit(truth, prediction)
    hist = LINEAR.density_histogram(truth, prediction)
    LINEAR._plot_density(
        path,
        *hist,
        fit,
        case_id=case_id,
        arm_key=arm,
        metric_label=metric,
        unit=unit,
        scope_label=scope,
    )
    return fit.as_dict()


def _triptych(path: Path, arm: str, metric: str, case_paths: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    for axis, role in zip(axes, ("worst", "most_frequent", "best")):
        image = plt.imread(case_paths[role])
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(ROLE_LABELS[role], fontsize=12, fontweight="bold")
        axis.text(
            0.5,
            -0.012,
            f"{arm} | {case_paths[role].parent.name.replace('__', '/')}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    fig.suptitle(f"{arm} | {metric.upper()} representative regressions", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--output-dir", type=Path, default=PLOT_ROOT)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest: dict[str, Any] = {"schema_version": 1, "arms": {}}
    for index in TARGET_INDICES:
        config_path = V4WB.matrix_config_path(index)
        config = V4WB.load_config(config_path, require_assets=True)
        run = str(config["experiment"]["id"])
        arm = _arm_name(config)
        arm_dir = args.output_dir / arm
        rep_speed = _representative_speed_cases(run, str(config["experiment"]["temporal_mode"]))
        rep_wss, wss_arrays = _representative_wss_cases(run)
        speed_paths: dict[str, Path] = {}
        wss_paths: dict[str, Path] = {}
        speed_fits: dict[str, Any] = {}
        wss_fits: dict[str, Any] = {}

        dataset = V4WB.V4Dataset(config, roles=("test",))
        case_indices = {str(dataset._case(i)["canonical_id"]): i for i in range(len(dataset))}
        checkpoint = V4WB.load_arm(index).checkpoint_path
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = V4WB.build_model(config).to(device=device, dtype=torch.float32)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        for role in ("worst", "most_frequent", "best"):
            case_id = rep_speed[role]
            if case_id not in case_indices:
                raise KeyError(f"{run}: representative case is not in test35: {case_id}")
            case_dir = arm_dir / "representative_cases" / _safe_case(case_id)
            truth_speed, predicted_speed = _predict_peak_speed(
                model, dataset, config, case_indices[case_id], device, args.chunk_size
            )
            speed_path = case_dir / "speed_regression.png"
            speed_fits[role] = _write_density(
                speed_path,
                truth_speed,
                predicted_speed,
                case_id=case_id,
                arm=arm,
                metric="speed magnitude",
                unit="m/s",
                scope="all anatomical interior points; CFD extensions excluded",
            )
            speed_paths[role] = speed_path

            wss_case_id = rep_wss[role]
            wss = wss_arrays[wss_case_id]
            wss_dir = arm_dir / "representative_cases" / _safe_case(wss_case_id)
            wss_path = wss_dir / "wss_regression.png"
            wss_fits[role] = _write_density(
                wss_path,
                np.asarray(wss["truth_mag"], dtype=np.float64),
                np.asarray(wss["pred_mag"], dtype=np.float64),
                case_id=wss_case_id,
                arm=arm,
                metric="WSS magnitude",
                unit="Pa",
                scope="all frozen wall nodes; Profile-Secant V3",
            )
            wss_paths[role] = wss_path
            print(json.dumps({"event": "representative_completed", "arm": arm, "role": role, "speed_case": case_id, "wss_case": wss_case_id}, ensure_ascii=False), flush=True)

        speed_triptych = arm_dir / "speed_representative_regressions.png"
        wss_triptych = arm_dir / "wss_representative_regressions.png"
        _triptych(speed_triptych, arm, "speed", speed_paths)
        _triptych(wss_triptych, arm, "wss", wss_paths)
        manifest["arms"][arm] = {
            "index": index,
            "experiment_id": run,
            "temporal_mode": config["experiment"]["temporal_mode"],
            "backbone": config["experiment"]["backbone"],
            "training_mode": config["experiment"]["training_mode"],
            "speed_representatives": rep_speed,
            "wss_representatives": rep_wss,
            "speed_fits": speed_fits,
            "wss_fits": wss_fits,
            "speed_triptych": str(speed_triptych),
            "wss_triptych": str(wss_triptych),
            "wss_cache": str(V4_AUDIT / run / "point_cache_s1200"),
        }
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "V4_representative_regressions_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "completed", "output_dir": str(args.output_dir), "arms": list(manifest["arms"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
