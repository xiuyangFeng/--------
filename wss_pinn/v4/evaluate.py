"""Frozen-checkpoint test35 evaluation for ``volume_uvwp_bc_rcr_v4``.

This module never trains, never reads test35 for checkpoint selection, and
never computes WSS.  Primary endpoint follows the v1.2 contract: case-equal
velocity-vector relative L2 on the evaluation universe of each temporal mode.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wss_pinn.metrics import RegressionAccumulator
from wss_pinn.utils import atomic_write_json, sha256_file, utc_now
from wss_pinn.v4.config import CONFIG_ROOT, load_config
from wss_pinn.v4.data import (
    V4Dataset,
    _bc_transform,
    _geometry_transform,
    _seed,
    _take,
)
from wss_pinn.v4.models import build_model

PROTOCOLS = ("official", "same5k")
CHECKPOINT_NAMES = ("last_converged", "last", "epoch_07500", "epoch_05000")
REGION_NAMES = {0: "core", 1: "near_wall"}
SCALAR_NAMES = ("u", "v", "w", "speed", "pressure")


def matrix_config_path(index: int) -> Path:
    lines = [
        line.strip()
        for line in (CONFIG_ROOT / "matrix_configs.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if index < 0 or index >= len(lines):
        raise IndexError(f"matrix index {index} out of range 0-{len(lines) - 1}")
    return Path(lines[index])


def resolve_checkpoint(run_dir: Path, name: str) -> Path:
    preferred = run_dir / "checkpoints" / f"{name}.pt"
    if preferred.is_file():
        return preferred
    fallback = run_dir / "checkpoints" / "last.pt"
    if name == "last_converged" and fallback.is_file():
        return fallback
    raise FileNotFoundError(f"missing checkpoint {preferred}")


def vector_relative_l2(pred_u: np.ndarray, true_u: np.ndarray) -> float:
    pred = np.asarray(pred_u, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(true_u, dtype=np.float64).reshape(-1, 3)
    if pred.shape != truth.shape:
        raise ValueError("velocity fields must share shape")
    numerator = float(np.sqrt(np.square(pred - truth).sum()))
    denominator = float(np.sqrt(np.square(truth).sum()))
    return numerator / max(denominator, 1e-30)


def denormalize_prediction(
    prediction: np.ndarray, field_stats: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    vel = pred[:, :3] * field_stats["velocity_std"].reshape(1, 3) + field_stats[
        "velocity_mean"
    ].reshape(1, 3)
    pressure = pred[:, 3] * float(np.asarray(field_stats["pressure_std"]).reshape(-1)[0]) + float(
        np.asarray(field_stats["pressure_mean"]).reshape(-1)[0]
    )
    return vel.astype(np.float32), pressure.astype(np.float32)


def _empty_accumulators() -> dict[str, RegressionAccumulator]:
    output = {name: RegressionAccumulator() for name in SCALAR_NAMES}
    for region in REGION_NAMES.values():
        output.update({f"{region}_{name}": RegressionAccumulator() for name in SCALAR_NAMES})
    return output


def _nmae(accumulator: RegressionAccumulator) -> dict[str, float]:
    if accumulator.count <= 0:
        return {
            "nmae": float("nan"),
            "nmae_denominator": "RMS(y)=sqrt(mean(y^2))",
            "nmae_denominator_value": float("nan"),
        }
    rms_truth = float(np.sqrt(accumulator.true_sq_sum / accumulator.count))
    mae = accumulator.error_abs_sum / accumulator.count
    return {
        "nmae": mae / max(rms_truth, 1e-30),
        "nmae_denominator": "RMS(y)=sqrt(mean(y^2))",
        "nmae_denominator_value": rms_truth,
    }


def _finalize_metrics(accumulators: dict[str, RegressionAccumulator]) -> dict[str, Any]:
    output = {}
    for name, accumulator in accumulators.items():
        result = accumulator.result()
        output[name] = {**result, **_nmae(accumulator)}
    return output


def _mean_case_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    names = [key for key in cases[0]["metrics"] if not key.startswith("_")]
    output: dict[str, Any] = {}
    for name in names:
        keys = ("r2", "mae", "rmse", "nmae", "count")
        bucket = {key: [] for key in keys}
        for case in cases:
            item = case["metrics"][name]
            for key in keys:
                if item["count"] and np.isfinite(item.get(key, float("nan"))):
                    bucket[key].append(float(item[key]))
        output[name] = {
            key: (float(np.mean(values)) if values else float("nan"))
            for key, values in bucket.items()
            if key != "count"
        }
        output[name]["n_cases_with_metric"] = len(bucket["r2"])
        output[name]["nmae_denominator"] = "per-case MAE/RMS(y), then case-equal mean"
    return output


class _CaseArrays:
    def __init__(self, dataset: V4Dataset, index: int):
        self.dataset = dataset
        self.case = dataset._case(index)
        self.case_id = str(self.case["canonical_id"])
        self.temporal_mode = dataset.temporal_mode
        transient = self.case["files"]["transient"]
        if self.temporal_mode == "steady_peak":
            if "steady" in self.case["files"]:
                files = self.case["files"]["steady"]
                self.coords = np.load(files["coords"]["path"], mmap_mode="r")
                self.geometry_raw = np.load(files["geometry_raw"]["path"], mmap_mode="r")
                is_wall = np.asarray(np.load(files["is_wall"]["path"], mmap_mode="r"), dtype=bool)
                self.velocity = np.load(files["velocity_m_s"]["path"], mmap_mode="r")
                self.pressure = np.load(files["pressure_relative_pa"]["path"], mmap_mode="r")
                self.geometry_is_pretransformed = False
                if dataset.centerline_v2_schema and "region" not in files:
                    raise ValueError("Centerline V2 steady evaluation requires region")
                self.region = (
                    np.asarray(np.load(files["region"]["path"], mmap_mode="r"))
                    if "region" in files
                    else None
                )
            else:
                volume_manifest = json.loads(
                    Path(self.case["files"]["steady_volume_manifest"]["path"]).read_text(
                        encoding="utf-8"
                    )
                )
                files = volume_manifest["files"]
                self.coords = np.load(files["interior_coords"]["path"], mmap_mode="r")
                self.geometry_raw = np.load(files["interior_geometry"]["path"], mmap_mode="r")
                is_wall = np.asarray(
                    np.load(files["interior_is_wall"]["path"], mmap_mode="r"), dtype=bool
                )
                self.velocity = np.load(files["velocity_m_s"]["path"], mmap_mode="r")
                self.pressure = np.load(files["pressure_relative_pa"]["path"], mmap_mode="r")
                self.geometry_is_pretransformed = True
                if "interior_region" in files:
                    self.region = np.asarray(
                        np.load(files["interior_region"]["path"], mmap_mode="r")
                    )
                else:
                    self.region = None
            self.population = np.flatnonzero(~is_wall)
            self.times_s = None
        else:
            self.coords = np.load(transient["coords"]["path"], mmap_mode="r")
            self.geometry_raw = np.load(transient["geometry_raw"]["path"], mmap_mode="r")
            eligible = np.asarray(
                np.load(transient["eligible_interior"]["path"], mmap_mode="r"), dtype=bool
            )
            self.population = np.flatnonzero(eligible)
            self.velocity = np.load(transient["velocity_m_s"]["path"], mmap_mode="r")
            self.pressure = np.load(transient["pressure_raw_pa"]["path"], mmap_mode="r")
            self.times_s = np.asarray(np.load(transient["times_s"]["path"], mmap_mode="r"))
            self.geometry_is_pretransformed = False
            if dataset.centerline_v2_schema and "region" not in transient:
                raise ValueError("Centerline V2 transient evaluation requires region")
            self.region = (
                np.asarray(np.load(transient["region"]["path"], mmap_mode="r"))
                if "region" in transient
                else None
            )

        self.bc_vector = _bc_transform(
            np.asarray(self.case["conditions"]["bc_vector_raw"], dtype=np.float32),
            dataset.stats["bc"],
        )

    def sample_indices(self, stream: str, count: int) -> np.ndarray:
        rng = np.random.default_rng(_seed(self.dataset.seed, 0, self.case_id, stream))
        return _take(rng, self.population, count)

    def support_features(self, support_index: np.ndarray) -> np.ndarray:
        coords = np.asarray(self.coords[support_index], dtype=np.float32)
        geometry = np.asarray(self.geometry_raw[support_index], dtype=np.float32)
        if self.geometry_is_pretransformed:
            geometry = (
                geometry - np.asarray(self.dataset.stats["geometry"]["mean"], dtype=np.float32)
            ) / np.asarray(self.dataset.stats["geometry"]["std"], dtype=np.float32)
        else:
            geometry = _geometry_transform(geometry, self.dataset.stats["geometry"])
        return np.concatenate([coords, geometry], axis=1).astype(np.float32)

    def query_truth(
        self, query_index: np.ndarray, frame_index: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.temporal_mode == "steady_peak":
            velocity = np.asarray(self.velocity[query_index], dtype=np.float32)
            pressure = np.asarray(self.pressure[query_index], dtype=np.float32)
        else:
            if frame_index is None:
                raise ValueError("transient query requires a frame index")
            velocity = np.asarray(self.velocity[frame_index][query_index], dtype=np.float32)
            pressure = np.asarray(self.pressure[frame_index][query_index], dtype=np.float32)
        return velocity, pressure


def _protocol_query_index(arrays: _CaseArrays, protocol: str, support_count: int) -> np.ndarray:
    if protocol == "official":
        return np.asarray(arrays.population, dtype=np.int64)
    return arrays.sample_indices("eval_query", support_count)


def _update_scalars(
    accumulators: dict[str, RegressionAccumulator],
    pred_u: np.ndarray,
    true_u: np.ndarray,
    pred_p: np.ndarray,
    true_p: np.ndarray,
    region: np.ndarray | None,
) -> None:
    pred_speed = np.linalg.norm(pred_u, axis=1)
    true_speed = np.linalg.norm(true_u, axis=1)
    values = {
        "u": (true_u[:, 0], pred_u[:, 0]),
        "v": (true_u[:, 1], pred_u[:, 1]),
        "w": (true_u[:, 2], pred_u[:, 2]),
        "speed": (true_speed, pred_speed),
        "pressure": (true_p, pred_p),
    }
    for name, (truth, pred) in values.items():
        accumulators[name].update(truth, pred)
    if region is None:
        return
    for code, region_name in REGION_NAMES.items():
        mask = region == code
        if not np.any(mask):
            continue
        for name, (truth, pred) in values.items():
            accumulators[f"{region_name}_{name}"].update(truth[mask], pred[mask])


@torch.no_grad()
def _decode_chunk(
    model: Any,
    encoded: dict[str, torch.Tensor],
    coords: np.ndarray,
    query_time_s: float | None,
    device: torch.device,
    field_stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    query_coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
    query_batch = torch.zeros(query_coords.size(0), dtype=torch.long, device=device)
    time_tensor = None
    if query_time_s is not None:
        time_tensor = torch.full(
            (query_coords.size(0), 1), float(query_time_s), dtype=torch.float32, device=device
        )
    prediction = model.decode_query(encoded, query_coords, query_batch, time_tensor)
    return denormalize_prediction(prediction.detach().cpu().numpy(), field_stats)


def evaluate_run(
    config_path: str | Path,
    *,
    protocol: str = "official",
    checkpoint_name: str = "last_converged",
    device_override: str | None = None,
    limit_cases: int | None = None,
    chunk_size: int = 16384,
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol={protocol!r}")
    if checkpoint_name not in CHECKPOINT_NAMES:
        raise ValueError(f"unsupported checkpoint={checkpoint_name!r}")

    config = load_config(config_path, require_assets=True)
    device_name = device_override or str(config["train"]["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dataset = V4Dataset(config, roles=("test",))
    if len(dataset) != 35:
        raise ValueError(f"formal V4 test35 evaluation must see 35 cases, got {len(dataset)}")
    if any(row["role"] != "test" for row in dataset.rows):
        raise ValueError("non-test rows entered evaluation")

    run_dir = config.run_dir
    checkpoint_path = resolve_checkpoint(run_dir, checkpoint_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("route") != config["route"]:
        raise ValueError("checkpoint route mismatch")
    if payload.get("resolved_config_sha256") != config.resolved_sha256:
        raise ValueError("checkpoint/resolved config SHA256 mismatch")
    saved_snapshot = payload.get("dataset_snapshot")
    if saved_snapshot is not None and saved_snapshot != dataset.dataset_snapshot:
        raise ValueError("checkpoint/dataset snapshot mismatch")
    if dataset.centerline_v2_schema and saved_snapshot is None:
        raise ValueError("Centerline V2 checkpoint lacks a dataset snapshot")
    model = build_model(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    support_count = int(config["sampling"]["support_points"])
    n_cases = len(dataset) if limit_cases is None else min(int(limit_cases), len(dataset))
    cases: list[dict[str, Any]] = []
    started = time.time()

    print(
        json.dumps(
            {
                "event": "v4_eval_started",
                "experiment_id": config["experiment"]["id"],
                "protocol": protocol,
                "checkpoint": str(checkpoint_path),
                "n_cases": n_cases,
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for index in range(n_cases):
        case_started = time.time()
        arrays = _CaseArrays(dataset, index)
        support_index = arrays.sample_indices("eval_support", support_count)
        query_index = _protocol_query_index(arrays, protocol, support_count)
        support_coords = torch.as_tensor(
            np.asarray(arrays.coords[support_index], dtype=np.float32),
            device=device,
        )
        support_features = torch.as_tensor(arrays.support_features(support_index), device=device)
        support_batch = torch.zeros(support_coords.size(0), dtype=torch.long, device=device)
        bc_vector = torch.as_tensor(arrays.bc_vector, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = model.encode_support(
            support_coords,
            support_features,
            support_batch,
            bc_vector,
            [arrays.case_id],
            int(config["train"]["seed"]),
        )

        accumulators = _empty_accumulators()
        error_sq = 0.0
        truth_sq = 0.0
        frames: list[int | None]
        if arrays.times_s is None:
            frames = [None]
        else:
            frames = list(range(len(arrays.times_s)))

        print(
            json.dumps(
                {
                    "event": "v4_eval_case_started",
                    "experiment_id": config["experiment"]["id"],
                    "case_id": arrays.case_id,
                    "case_index": index,
                    "n_support": int(len(support_index)),
                    "n_query": int(len(query_index)),
                    "n_frames": len(frames),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        for frame_index in frames:
            query_time = None if frame_index is None else float(arrays.times_s[frame_index])
            for start in range(0, len(query_index), int(chunk_size)):
                chunk_index = query_index[start : start + int(chunk_size)]
                chunk_coords = np.asarray(arrays.coords[chunk_index], dtype=np.float32)
                pred_u, pred_p = _decode_chunk(
                    model,
                    encoded,
                    chunk_coords,
                    query_time,
                    device,
                    dataset.field_stats,
                )
                true_u, true_p = arrays.query_truth(chunk_index, frame_index)
                delta = pred_u.astype(np.float64) - true_u.astype(np.float64)
                error_sq += float(np.square(delta).sum())
                truth_sq += float(np.square(true_u.astype(np.float64)).sum())
                region = None
                if arrays.region is not None:
                    region = np.asarray(arrays.region[chunk_index])
                _update_scalars(accumulators, pred_u, true_u, pred_p, true_p, region)

        e_case = float(np.sqrt(error_sq) / max(np.sqrt(truth_sq), 1e-30))
        case_record = {
            "case_id": arrays.case_id,
            "e_rel_l2": e_case,
            "n_query_points": int(len(query_index)),
            "n_frames": len(frames),
            "n_evaluated_vectors": int(len(query_index) * len(frames)),
            "seconds": round(time.time() - case_started, 3),
            "metrics": _finalize_metrics(accumulators),
        }
        cases.append(case_record)
        print(
            json.dumps(
                {
                    "event": "v4_eval_case_completed",
                    "experiment_id": config["experiment"]["id"],
                    "case_id": arrays.case_id,
                    "case_index": index,
                    "e_rel_l2": e_case,
                    "seconds": case_record["seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    e_values = [float(case["e_rel_l2"]) for case in cases]
    protocol_notes = {
        "official": (
            "5k eval_support; steady query = all non-wall interior points; "
            "transient query = all eligible cache points x all 81 frames. "
            "This is the V4 evaluation universe, not a 5k spatial subsample."
        ),
        "same5k": (
            "5k eval_support and 5k eval_query; transient repeats the same 5k "
            "spatial points across all 81 frames. Screen protocol, not §12.0 "
            "all-points primary."
        ),
    }
    used_name = checkpoint_path.name.replace(".pt", "")
    result = {
        "schema_version": 1,
        "route": config["route"],
        "experiment": {
            "id": config["experiment"]["id"],
            "temporal_mode": config["experiment"]["temporal_mode"],
            "backbone": config["experiment"]["backbone"],
            "training_mode": config["experiment"]["training_mode"],
            "seed": int(config["experiment"]["seed"]),
        },
        "created_at": utc_now(),
        "protocol": protocol,
        "protocol_note": protocol_notes[protocol],
        "checkpoint": {
            "requested": checkpoint_name,
            "used": used_name,
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(payload.get("epoch", -1)),
            "converged": bool(payload.get("train_only_converged", False)),
        },
        "sampling": {
            "support_points": support_count,
            "support_stream": "eval_support",
            "query_stream": "all_population" if protocol == "official" else "eval_query",
            "epoch_for_rng": 0,
            "chunk_size": int(chunk_size),
        },
        "n_cases": len(cases),
        "limit_cases": limit_cases,
        "primary": {
            "name": "velocity_vector_relative_l2",
            "definition": "E_case = ||u_pred-u_true||_2 / ||u_true||_2; transient frames pooled",
            "e_rel_l2_case_balanced": float(np.mean(e_values)),
            "e_rel_l2_median": float(np.median(e_values)),
            "e_rel_l2_min": float(np.min(e_values)),
            "e_rel_l2_max": float(np.max(e_values)),
            "e_rel_l2_cases": e_values,
        },
        "secondary_case_balanced": _mean_case_metrics(cases),
        "cases": cases,
        "seconds": round(time.time() - started, 3),
        "limitations": [
            "test35 is a development-exposed screen; conclusions are exploratory",
            "single-seed numbers cannot replace Holm-corrected 3-seed means",
            "this evaluation does not compute WSS",
            "checkpoint was not selected on test35",
        ],
    }
    suffix = "partial" if limit_cases is not None else "full"
    output_path = run_dir / f"evaluation_{protocol}_{used_name}_{suffix}.json"
    atomic_write_json(output_path, result)
    result["output_path"] = str(output_path)
    print(
        json.dumps(
            {
                "event": "v4_eval_completed",
                "experiment_id": config["experiment"]["id"],
                "protocol": protocol,
                "e_rel_l2_case_balanced": result["primary"]["e_rel_l2_case_balanced"],
                "output_path": str(output_path),
                "seconds": result["seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def summarize_evaluations(paths: list[Path], output_path: Path) -> dict[str, Any]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment = payload["experiment"]
        secondary = payload["secondary_case_balanced"]
        rows.append(
            {
                "experiment_id": experiment["id"],
                "temporal_mode": experiment["temporal_mode"],
                "backbone": experiment["backbone"],
                "training_mode": experiment["training_mode"],
                "seed": experiment["seed"],
                "protocol": payload["protocol"],
                "checkpoint": payload["checkpoint"]["used"],
                "n_cases": payload["n_cases"],
                "e_rel_l2_case_balanced": payload["primary"]["e_rel_l2_case_balanced"],
                "e_rel_l2_median": payload["primary"]["e_rel_l2_median"],
                "speed_r2": secondary["speed"]["r2"],
                "speed_mae": secondary["speed"]["mae"],
                "speed_rmse": secondary["speed"]["rmse"],
                "u_r2": secondary["u"]["r2"],
                "v_r2": secondary["v"]["r2"],
                "w_r2": secondary["w"]["r2"],
                "pressure_r2": secondary["pressure"]["r2"],
                "pressure_mae": secondary["pressure"]["mae"],
                "path": str(path),
            }
        )
    rows.sort(
        key=lambda row: (
            row["temporal_mode"],
            row["backbone"],
            row["training_mode"],
            row["seed"],
        )
    )
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "n_runs": len(rows),
        "runs": rows,
        "note": (
            "Case-equal velocity-vector relative L2 is the pre-registered primary. "
            "Do not promote a winner from a single seed or a partial 0-14 slice."
        ),
    }
    atomic_write_json(output_path, summary)
    summary["output_path"] = str(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--matrix-index", type=int)
    parser.add_argument("--protocol", default="official", choices=PROTOCOLS)
    parser.add_argument("--checkpoint", default="last_converged", choices=CHECKPOINT_NAMES)
    parser.add_argument("--device")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--summarize", nargs="*")
    parser.add_argument("--summarize-output")
    args = parser.parse_args()
    if args.summarize:
        if not args.summarize_output:
            raise ValueError("--summarize-output is required with --summarize")
        result = summarize_evaluations(
            [Path(path) for path in args.summarize],
            Path(args.summarize_output),
        )
    else:
        config_path = args.config
        if config_path is None:
            if args.matrix_index is None:
                raise ValueError("provide --config or --matrix-index")
            config_path = matrix_config_path(args.matrix_index)
        result = evaluate_run(
            config_path,
            protocol=args.protocol,
            checkpoint_name=args.checkpoint,
            device_override=args.device,
            limit_cases=args.limit_cases,
            chunk_size=args.chunk_size,
        )
    print(json.dumps({k: result[k] for k in result if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
