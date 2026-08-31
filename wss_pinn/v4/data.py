"""Deterministic case-balanced sampling for V4 steady/transient training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from wss_pinn.utils import sha256_file, sha256_json

from .geometry_v2 import transform_geometry
from .waveform import PERIOD_S, q_nom_numpy


CENTERLINE_V2_STAGING_ROUTE = "volume_uvwp_bc_rcr_v4_centerline_v2_rawfull_v2_staging"
V2_REQUIRED_STEADY = {
    "coords",
    "geometry_raw",
    "geometry_aux",
    "is_wall",
    "region",
    "distance_to_wall_mm",
    "velocity_m_s",
    "pressure_relative_pa",
}
V2_REQUIRED_TRANSIENT = {
    "coords",
    "geometry_raw",
    "geometry_aux",
    "is_wall",
    "eligible_interior",
    "region",
    "distance_to_wall_mm",
    "velocity_m_s",
    "pressure_raw_pa",
    "cell_ids",
    "steps",
    "times_s",
    "q_actual_m3_s",
}
V2_REQUIRED_BOUNDARY = {
    "inlet_coords",
    "inlet_normals",
    "wall_coords",
    "outlet_coords",
    "outlet_normals",
    "outlet_area_weights_m2",
    "outlet_zone",
}


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
    if "raw_names" in stats:
        return transform_geometry(raw, stats)
    # Historical pre-Centerline-V2 fallback.  It is retained only so old runs
    # remain reproducible; new manifests must provide the explicit raw schema.
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
    def __init__(
        self,
        config: Any,
        *,
        roles: tuple[str, ...] = ("train",),
        allow_staging_for_audit: bool = False,
    ):
        self.config = config
        self.temporal_mode = str(config["experiment"]["temporal_mode"])
        self.seed = int(config["train"]["seed"])
        self.epoch = 0
        manifest_path = Path(config["paths"]["manifest"])
        stats_path = Path(config["paths"]["stats"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        self.centerline_v2_schema = (
            manifest.get("route") == CENTERLINE_V2_STAGING_ROUTE
            and int(manifest.get("schema_version", 0)) == 2
        )
        if "centerline_v2" in str(manifest.get("route", "")) and not self.centerline_v2_schema:
            raise ValueError("unsupported Centerline V2 route/schema combination")
        if manifest.get("stats", {}).get("sha256"):
            if sha256_file(stats_path) != manifest["stats"]["sha256"]:
                raise ValueError("aggregate manifest/stats SHA256 mismatch")
        if self.centerline_v2_schema:
            if self.stats.get("route") != manifest["route"] or int(
                self.stats.get("schema_version", 0)
            ) != 2:
                raise ValueError("Centerline V2 manifest/stats route or schema mismatch")
            split_path = Path(config["paths"]["split"])
            split_record = manifest.get("split", {})
            if not split_record.get("sha256") or sha256_file(split_path) != split_record["sha256"]:
                raise ValueError("Centerline V2 manifest/split SHA256 mismatch")
            split = json.loads(split_path.read_text(encoding="utf-8"))
            train_ids = [row["canonical_id"] for row in manifest["cases"] if row["role"] == "train"]
            test_ids = [row["canonical_id"] for row in manifest["cases"] if row["role"] == "test"]
            if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
                raise ValueError("Centerline V2 manifest contains duplicate case IDs")
            if set(train_ids) != set(split["train_cases"]) or set(test_ids) != set(
                split["test_cases"]
            ):
                raise ValueError("Centerline V2 manifest roles do not match the frozen split")
            if set(self.stats.get("train_case_ids", [])) != set(split["train_cases"]):
                raise ValueError("Centerline V2 stats membership does not match train138")
            audit_record = manifest.get("array_audit")
            if audit_record:
                audit_path = Path(audit_record["path"])
                if sha256_file(audit_path) != audit_record["sha256"]:
                    raise ValueError("Centerline V2 array audit SHA256 mismatch")
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                expected_case_sha = {
                    row["canonical_id"]: row["manifest_sha256"] for row in manifest["cases"]
                }
                if (
                    audit.get("status") != "pass"
                    or audit.get("errors")
                    or int(audit.get("case_count", -1)) != len(manifest["cases"])
                    or int(audit.get("array_count", -1)) != int(
                        audit_record.get("array_count", -2)
                    )
                    or audit.get("case_manifest_sha256") != expected_case_sha
                ):
                    raise ValueError("Centerline V2 array audit content mismatch")
            if "train" in set(roles) and not allow_staging_for_audit:
                if manifest.get("training_ready") is not True:
                    raise ValueError("Centerline V2 staging is not training-ready")
                if manifest.get("gate_result") != "pass":
                    raise ValueError("formal Centerline V2 training requires gate_result=pass")
                if not audit_record:
                    raise ValueError("formal Centerline V2 training requires a bound array audit")
        self.dataset_snapshot = {
            "manifest_sha256": sha256_file(manifest_path),
            "stats_sha256": sha256_file(stats_path),
            "split_sha256": sha256_file(config["paths"]["split"]),
            "array_audit_sha256": (
                manifest.get("array_audit", {}).get("sha256")
            ),
            "case_manifest_root_sha256": sha256_json(
                {
                    row["canonical_id"]: row.get("manifest_sha256")
                    for row in manifest["cases"]
                }
            ),
        }
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
            case_path = Path(row["manifest"])
            if row.get("manifest_sha256") and sha256_file(case_path) != row["manifest_sha256"]:
                raise ValueError("aggregate/case manifest SHA256 mismatch")
            payload = json.loads(case_path.read_text(encoding="utf-8"))
            if payload["canonical_id"] != row["canonical_id"]:
                raise ValueError("aggregate/case manifest identity mismatch")
            if self.centerline_v2_schema:
                if payload.get("route") != CENTERLINE_V2_STAGING_ROUTE or int(
                    payload.get("schema_version", 0)
                ) != 2:
                    raise ValueError("Centerline V2 aggregate/case route or schema mismatch")
                schema = payload.get("geometry_schema", {})
                if schema.get("raw_names") != self.stats["geometry"].get("raw_names"):
                    raise ValueError("Centerline V2 case/stats raw geometry schema mismatch")
                if schema.get("model_names") != self.stats["geometry"].get("names"):
                    raise ValueError("Centerline V2 case/stats model geometry schema mismatch")
                files = payload.get("files", {})
                required = {
                    "steady": V2_REQUIRED_STEADY,
                    "transient": V2_REQUIRED_TRANSIENT,
                    "boundary_faces": V2_REQUIRED_BOUNDARY,
                }
                for group, names in required.items():
                    missing = sorted(names.difference(files.get(group, {})))
                    if missing:
                        raise ValueError(
                            f"Centerline V2 case is missing required {group} arrays: {missing}"
                        )
            self._case_manifest_cache[index] = payload
        return self._case_manifest_cache[index]

    def _sample_indices(self, case_id: str, strict: np.ndarray, stream: str, count: int) -> np.ndarray:
        rng = np.random.default_rng(_seed(self.seed, self.epoch, case_id, stream))
        return _take(rng, strict, count)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self._case(index)
        case_id = case["canonical_id"]
        sampling = self.config["sampling"]
        transient = case["files"].get("transient", {})
        boundary_group = case["files"]["boundary_faces"]
        if "path" in boundary_group:
            boundary_path = Path(boundary_group["path"])
            with np.load(boundary_path, allow_pickle=False) as payload:
                boundary = {key: np.asarray(payload[key]) for key in payload.files}
        else:
            boundary = {
                key: np.asarray(np.load(record["path"], mmap_mode="r"))
                for key, record in boundary_group.items()
            }

        if self.temporal_mode == "steady_peak":
            if "steady" in case["files"]:
                if not self.centerline_v2_schema:
                    raise ValueError("direct steady arrays require the explicit Centerline V2 schema")
                files = case["files"]["steady"]
                coords = np.load(files["coords"]["path"], mmap_mode="r")
                geometry_raw = np.load(files["geometry_raw"]["path"], mmap_mode="r")
                is_wall = np.load(files["is_wall"]["path"], mmap_mode="r")
                velocity = np.load(files["velocity_m_s"]["path"], mmap_mode="r")
                pressure = np.load(files["pressure_relative_pa"]["path"], mmap_mode="r")
                geometry_is_pretransformed = False
            else:
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
                geometry_is_pretransformed = True
            strict = np.flatnonzero(~np.asarray(is_wall, dtype=bool))
            data_time = None
            bc_time = None
            frame_index = None
        else:
            if self.centerline_v2_schema and "geometry_schema" not in case:
                raise ValueError("Centerline V2 transient arrays require explicit geometry_schema")
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
