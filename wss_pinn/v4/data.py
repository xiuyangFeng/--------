"""Deterministic case-balanced sampling for V4 steady/transient training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .waveform import PERIOD_S, q_nom_numpy


def _seed(global_seed: int, epoch: int, case_id: str, stream: str) -> int:
    digest = hashlib.sha256(
        f"{global_seed}|{epoch}|{case_id}|{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _take(rng: np.random.Generator, values: np.ndarray, count: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("cannot sample an empty population")
    replace = int(count) > len(values)
    return np.asarray(rng.choice(values, size=int(count), replace=replace), dtype=np.int64)


def _geometry_transform(raw: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32).copy()
    clip = float(stats["curvature_clip_abs"])
    value[:, 2] = np.sign(value[:, 2]) * np.log1p(
        np.minimum(np.abs(value[:, 2]), clip)
    )
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (value - mean) / std


def _bc_transform(raw: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32).copy()
    log_indices = [0]
    for outlet in range(4):
        base = 2 + 4 * outlet
        log_indices.extend([base, base + 1, base + 2, base + 3])
    value[log_indices] = np.log10(value[log_indices])
    return (
        value - np.asarray(stats["mean"], dtype=np.float32)
    ) / np.asarray(stats["std"], dtype=np.float32)


def field_stats(stats: dict[str, Any], temporal_mode: str) -> dict[str, np.ndarray]:
    group = stats["steady"] if temporal_mode == "steady_peak" else stats["transient"]
    return {
        "velocity_mean": np.asarray(group["velocity_m_s"]["mean"], dtype=np.float32),
        "velocity_std": np.asarray(group["velocity_m_s"]["std"], dtype=np.float32),
        "pressure_mean": np.asarray(group["pressure_pa"]["mean"], dtype=np.float32),
        "pressure_std": np.asarray(group["pressure_pa"]["std"], dtype=np.float32),
    }


def _normalize_target(
    velocity: np.ndarray,
    pressure: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        [
            (np.asarray(velocity, dtype=np.float32) - stats["velocity_mean"])
            / stats["velocity_std"],
            (
                np.asarray(pressure, dtype=np.float32).reshape(-1, 1)
                - stats["pressure_mean"].reshape(1, 1)
            )
            / stats["pressure_std"].reshape(1, 1),
        ],
        axis=1,
    ).astype(np.float32)


class V4Dataset(Dataset):
    def __init__(self, config: Any, *, roles: tuple[str, ...] = ("train",)):
        self.config = config
        self.temporal_mode = str(config["experiment"]["temporal_mode"])
        self.seed = int(config["train"]["seed"])
        self.epoch = 0
        manifest_path = Path(config["paths"]["manifest"])
        stats_path = Path(config["paths"]["stats"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.field_stats = field_stats(self.stats, self.temporal_mode)
        selected = [row for row in manifest["cases"] if row["role"] in set(roles)]
        if not selected:
            raise ValueError(f"no cases for roles={roles}")
        if roles == ("train",) and any(row["role"] == "test" for row in selected):
            raise ValueError("test35 entered the training dataset")
        self.rows = selected
        self._case_manifest_cache: dict[int, dict[str, Any]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _case(self, index: int) -> dict[str, Any]:
        if index not in self._case_manifest_cache:
            row = self.rows[index]
            payload = json.loads(Path(row["manifest"]).read_text(encoding="utf-8"))
            if payload["canonical_id"] != row["canonical_id"]:
                raise ValueError("aggregate/case manifest identity mismatch")
            self._case_manifest_cache[index] = payload
        return self._case_manifest_cache[index]

    def _sample_indices(self, case_id: str, strict: np.ndarray, stream: str, count: int) -> np.ndarray:
        rng = np.random.default_rng(_seed(self.seed, self.epoch, case_id, stream))
        return _take(rng, strict, count)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self._case(index)
        case_id = case["canonical_id"]
        sampling = self.config["sampling"]
        transient = case["files"]["transient"]
        boundary_path = Path(case["files"]["boundary_faces"]["path"])
        with np.load(boundary_path, allow_pickle=False) as payload:
            boundary = {key: np.asarray(payload[key]) for key in payload.files}

        if self.temporal_mode == "steady_peak":
            volume_manifest = json.loads(
                Path(case["files"]["steady_volume_manifest"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            files = volume_manifest["files"]
            coords = np.load(files["interior_coords"]["path"], mmap_mode="r")
            geometry_raw = np.load(files["interior_geometry"]["path"], mmap_mode="r")
            is_wall = np.load(files["interior_is_wall"]["path"], mmap_mode="r")
            velocity = np.load(files["velocity_m_s"]["path"], mmap_mode="r")
            pressure = np.load(files["pressure_relative_pa"]["path"], mmap_mode="r")
            strict = np.flatnonzero(~np.asarray(is_wall, dtype=bool))
            geometry_is_pretransformed = True
            data_time = None
            bc_time = None
            frame_index = None
        else:
            coords = np.load(transient["coords"]["path"], mmap_mode="r")
            geometry_raw = np.load(transient["geometry_raw"]["path"], mmap_mode="r")
            is_wall = np.load(transient["is_wall"]["path"], mmap_mode="r")
            eligible = np.load(
                transient["eligible_interior"]["path"], mmap_mode="r"
            )
            velocity_all = np.load(transient["velocity_m_s"]["path"], mmap_mode="r")
            pressure_all = np.load(transient["pressure_raw_pa"]["path"], mmap_mode="r")
            times = np.load(transient["times_s"]["path"], mmap_mode="r")
            strict = np.flatnonzero(np.asarray(eligible, dtype=bool))
            geometry_is_pretransformed = False
            time_rng = np.random.default_rng(
                _seed(self.seed, self.epoch, case_id, "label_time")
            )
            frame_index = int(time_rng.integers(0, len(times)))
            data_time = float(times[frame_index])
            velocity = velocity_all[frame_index]
            pressure = pressure_all[frame_index]
            buffer_s = float(sampling["time_reset_buffer_s"])
            bc_rng = np.random.default_rng(
                _seed(self.seed, self.epoch, case_id, "bc_time")
            )
            bc_time = float(bc_rng.uniform(buffer_s, PERIOD_S - buffer_s))

        support_index = self._sample_indices(
            case_id, strict, "support", int(sampling["support_points"])
        )
        query_index = self._sample_indices(
            case_id, strict, "query", int(sampling["query_points"])
        )
        physics_index = self._sample_indices(
            case_id, strict, "physics", int(sampling["physics_points"])
        )
        support_coords = np.asarray(coords[support_index], dtype=np.float32)
        sampled_geometry = np.asarray(
            geometry_raw[support_index], dtype=np.float32
        )
        if geometry_is_pretransformed:
            support_geometry = (
                sampled_geometry
                - np.asarray(self.stats["geometry"]["mean"], dtype=np.float32)
            ) / np.asarray(self.stats["geometry"]["std"], dtype=np.float32)
        else:
            support_geometry = _geometry_transform(
                sampled_geometry,
                self.stats["geometry"],
            )
        support_features = np.concatenate([support_coords, support_geometry], axis=1)
        query_target = _normalize_target(
            np.asarray(velocity[query_index], dtype=np.float32),
            np.asarray(pressure[query_index], dtype=np.float32),
            self.field_stats,
        )

        wall_population = np.arange(len(boundary["wall_coords"]), dtype=np.int64)
        inlet_population = np.arange(len(boundary["inlet_coords"]), dtype=np.int64)
        wall_index = self._sample_indices(
            case_id, wall_population, "wall", int(sampling["wall_points"])
        )
        inlet_index = self._sample_indices(
            case_id, inlet_population, "inlet", int(sampling["inlet_points"])
        )
        inlet_normal = np.asarray(boundary["inlet_normals"][inlet_index], dtype=np.float32)
        if self.temporal_mode == "steady_peak":
            inlet_speed = float(case["physics_scales"]["U_c_m_s"])
        else:
            inlet_speed = float(q_nom_numpy(bc_time) / case["conditions"]["a_in_udf_m2"])
        inlet_target = inlet_normal * inlet_speed

        if self.temporal_mode == "transient_81":
            physics_rng = np.random.default_rng(
                _seed(self.seed, self.epoch, case_id, "physics_time")
            )
            buffer_s = float(sampling["time_reset_buffer_s"])
            physics_time = float(
                physics_rng.uniform(buffer_s, PERIOD_S - buffer_s)
            )
        else:
            physics_time = None

        return {
            "case_id": case_id,
            "frame_index": frame_index,
            "support_coords": support_coords,
            "support_features": support_features.astype(np.float32),
            "query_coords": np.asarray(coords[query_index], dtype=np.float32),
            "query_target": query_target,
            "query_time_s": data_time,
            "physics_coords": np.asarray(coords[physics_index], dtype=np.float32),
            "physics_time_s": physics_time,
            "wall_coords": np.asarray(boundary["wall_coords"][wall_index], dtype=np.float32),
            "wall_time_s": bc_time,
            "inlet_coords": np.asarray(boundary["inlet_coords"][inlet_index], dtype=np.float32),
            "inlet_target_velocity_m_s": inlet_target.astype(np.float32),
            "inlet_time_s": bc_time,
            "outlet_coords": np.asarray(boundary["outlet_coords"], dtype=np.float32),
            "outlet_normals": np.asarray(boundary["outlet_normals"], dtype=np.float32),
            "outlet_area_weights_m2": np.asarray(
                boundary["outlet_area_weights_m2"], dtype=np.float32
            ),
            "outlet_zone": np.asarray(boundary["outlet_zone"], dtype=np.int64),
            "outlet_time_s": bc_time,
            "bc_vector": _bc_transform(
                np.asarray(case["conditions"]["bc_vector_raw"], dtype=np.float32),
                self.stats["bc"],
            ),
            "length_m": float(case["physics_scales"]["coordinate_length_m"]),
            "U_c_m_s": float(case["physics_scales"]["U_c_m_s"]),
            "L_c_m": float(case["physics_scales"]["L_c_m"]),
            "R1": np.asarray(
                [item["R1"] for item in case["conditions"]["rcr_mass_flow_basis"]],
                dtype=np.float32,
            ),
            "R2": np.asarray(
                [item["R2"] for item in case["conditions"]["rcr_mass_flow_basis"]],
                dtype=np.float32,
            ),
            "C": np.asarray(
                [item["C"] for item in case["conditions"]["rcr_mass_flow_basis"]],
                dtype=np.float32,
            ),
            "rcr_pressure_scale_pa": np.asarray(
                case["physics_scales"]["rcr_pressure_scale_pa"], dtype=np.float32
            ),
        }


def _cat_points(
    samples: list[dict[str, Any]], key: str, *, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [torch.as_tensor(sample[key], dtype=dtype) for sample in samples]
    batches = [
        torch.full((len(value),), index, dtype=torch.long)
        for index, value in enumerate(values)
    ]
    return torch.cat(values, dim=0), torch.cat(batches, dim=0)


def _point_times(
    samples: list[dict[str, Any]], point_key: str, time_key: str
) -> torch.Tensor | None:
    if samples[0][time_key] is None:
        return None
    return torch.cat(
        [
            torch.full(
                (len(sample[point_key]), 1),
                float(sample[time_key]),
                dtype=torch.float32,
            )
            for sample in samples
        ],
        dim=0,
    )


def collate_v4(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "case_ids": [sample["case_id"] for sample in samples],
        "frame_indices": [sample["frame_index"] for sample in samples],
        "bc_vector": torch.stack(
            [torch.as_tensor(sample["bc_vector"], dtype=torch.float32) for sample in samples]
        ),
    }
    for key in (
        "support_coords",
        "support_features",
        "query_coords",
        "query_target",
        "physics_coords",
        "wall_coords",
        "inlet_coords",
        "inlet_target_velocity_m_s",
        "outlet_coords",
        "outlet_normals",
        "outlet_area_weights_m2",
        "outlet_zone",
    ):
        dtype = torch.long if key == "outlet_zone" else torch.float32
        output[key], output[f"{key.rsplit('_', 1)[0]}_batch"] = _cat_points(
            samples, key, dtype=dtype
        )
    # The generic batch-key names above are ambiguous for target/normals;
    # replace them with the coordinate population batches used by the model.
    output["support_batch"] = _cat_points(samples, "support_coords")[1]
    output["query_batch"] = _cat_points(samples, "query_coords")[1]
    output["physics_batch"] = _cat_points(samples, "physics_coords")[1]
    output["wall_batch"] = _cat_points(samples, "wall_coords")[1]
    output["inlet_batch"] = _cat_points(samples, "inlet_coords")[1]
    output["outlet_batch"] = _cat_points(samples, "outlet_coords")[1]
    output["query_time_s"] = _point_times(samples, "query_coords", "query_time_s")
    output["physics_time_s"] = _point_times(samples, "physics_coords", "physics_time_s")
    output["wall_time_s"] = _point_times(samples, "wall_coords", "wall_time_s")
    output["inlet_time_s"] = _point_times(samples, "inlet_coords", "inlet_time_s")
    output["outlet_time_s"] = _point_times(samples, "outlet_coords", "outlet_time_s")
    for key in ("length_m", "U_c_m_s", "L_c_m"):
        output[key] = torch.as_tensor([sample[key] for sample in samples], dtype=torch.float32)
    for key in ("R1", "R2", "C", "rcr_pressure_scale_pa"):
        output[key] = torch.stack(
            [torch.as_tensor(sample[key], dtype=torch.float32) for sample in samples]
        )
    return output
