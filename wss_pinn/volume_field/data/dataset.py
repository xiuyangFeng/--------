"""Epoch-resampled full-volume dataset backed by mmap sidecars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from wss_pinn.utils import sha256_file

from .builder import SCHEMA_VERSION


def _take(rng: np.random.Generator, length: int, count: int, excluded=None) -> np.ndarray:
    if count > length:
        raise ValueError(f"cannot sample {count} distinct rows from {length}")
    if excluded is None or len(excluded) == 0:
        return np.sort(rng.choice(length, size=count, replace=False))
    allowed = np.ones(length, dtype=bool)
    allowed[np.asarray(excluded, dtype=np.int64)] = False
    candidates = np.flatnonzero(allowed)
    if count > len(candidates):
        raise ValueError("independent query sampling pool is too small")
    return np.sort(rng.choice(candidates, size=count, replace=False))


def _take_strict_volume(
    rng: np.random.Generator,
    is_wall: np.ndarray,
    count: int,
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Uniform rejection sampling over non-wall volume rows."""
    length = int(len(is_wall))
    excluded_set = set() if excluded is None else set(map(int, excluded))
    selected: set[int] = set()
    attempts = 0
    while len(selected) < int(count):
        attempts += 1
        if attempts > 100:
            raise RuntimeError("strict-volume rejection sampling did not converge")
        need = int(count) - len(selected)
        candidates = rng.integers(0, length, size=max(need * 2, 1024), endpoint=False)
        mask = np.asarray(is_wall[candidates], dtype=bool)
        for value in candidates[~mask]:
            integer = int(value)
            if integer not in excluded_set:
                selected.add(integer)
                if len(selected) == int(count):
                    break
    return np.asarray(sorted(selected), dtype=np.int64)


class VolumeCase:
    def __init__(self, row: dict[str, Any]):
        self.row = dict(row)
        manifest_path = Path(row["manifest"])
        if sha256_file(manifest_path) != row.get("manifest_sha256"):
            raise ValueError(f"case manifest hash drift: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema_version") != SCHEMA_VERSION
            or self.manifest.get("status") != "completed"
            or self.manifest.get("route") != "volume_uvwp_peak_v1"
        ):
            raise ValueError(f"case manifest contract drift: {manifest_path}")
        if self.manifest.get("canonical_id") != row.get("canonical_id"):
            raise ValueError(f"case identity drift: {manifest_path}")
        if self.manifest.get("role") != row.get("role"):
            raise ValueError(f"case role drift: {manifest_path}")
        if (
            self.manifest.get("pressure_target")
            != "case_strict_volume_mean_centered_pa"
        ):
            raise ValueError(f"case pressure target drift: {manifest_path}")
        self.case_id = str(self.manifest["canonical_id"])
        self.role = str(self.manifest["role"])
        self.length_m = float(self.manifest["scales"]["length_m"])
        if (
            not np.isfinite(self.length_m)
            or self.length_m <= 0
            or self.manifest["scales"].get("density_kg_m3") != 1060.0
        ):
            raise ValueError(f"case physical scale drift: {manifest_path}")
        files = self.manifest["files"]
        self.coords = np.load(files["interior_coords"]["path"], mmap_mode="r")
        self.geometry = np.load(files["interior_geometry"]["path"], mmap_mode="r")
        self.region = np.load(files["interior_region"]["path"], mmap_mode="r")
        self.is_wall = np.load(files["interior_is_wall"]["path"], mmap_mode="r")
        self.velocity = np.load(files["velocity_m_s"]["path"], mmap_mode="r")
        self.pressure = np.load(files["pressure_relative_pa"]["path"], mmap_mode="r")
        self.wall_coords = np.load(files["wall_coords"]["path"], mmap_mode="r")
        self.wall_geometry = np.load(files["wall_geometry"]["path"], mmap_mode="r")


class VolumeFieldDataset(Dataset):
    """One item is one case with fresh support/query/wall indices for an epoch."""

    def __init__(
        self,
        aggregate_manifest: str | Path,
        field_stats: str | Path,
        *,
        roles: list[str] | tuple[str, ...],
        input_variant: str,
        sampling: dict[str, Any],
        seed: int,
    ):
        aggregate_path = Path(aggregate_manifest)
        stats_path = Path(field_stats)
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if (
            aggregate.get("schema_version") != SCHEMA_VERSION
            or aggregate.get("status") != "completed"
            or aggregate.get("route") != "volume_uvwp_peak_v1"
        ):
            raise ValueError("aggregate sidecar manifest contract drift")
        aggregate_stats = aggregate.get("field_stats", {})
        if (
            Path(aggregate_stats.get("path", "")).resolve() != stats_path.resolve()
            or aggregate_stats.get("sha256") != sha256_file(stats_path)
        ):
            raise ValueError("aggregate-to-field-stats binding drift")
        rows = aggregate.get("cases", [])
        case_ids = [row.get("canonical_id") for row in rows]
        if len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
            raise ValueError("aggregate case IDs must be present and unique")
        allowed = set(roles)
        self.cases = [VolumeCase(row) for row in rows if row["role"] in allowed]
        if not self.cases:
            raise ValueError(f"no volume cases for roles={sorted(allowed)}")
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if (
            self.stats.get("schema_version") != SCHEMA_VERSION
            or self.stats.get("status") != "completed"
            or self.stats.get("route") != "volume_uvwp_peak_v1"
            or self.stats.get("scope") != "train138 strict-volume only"
        ):
            raise ValueError("field stats contract drift")
        if (
            aggregate.get("split", {}).get("sha256")
            != self.stats.get("split", {}).get("sha256")
        ):
            raise ValueError("aggregate/stats split binding drift")
        expected_train_hashes = {
            row["canonical_id"]: row.get("manifest_sha256")
            for row in rows
            if row.get("role") == "train"
        }
        if self.stats.get("case_manifest_sha256") != expected_train_hashes:
            raise ValueError("train-only field-stat membership/hash drift")
        self.input_variant = str(input_variant)
        if self.input_variant not in {"xyz", "xyz_geom"}:
            raise ValueError(f"unsupported input_variant={self.input_variant!r}")
        self.sampling = dict(sampling)
        self.seed = int(seed)
        self.epoch = 0
        self.geometry_mean = np.asarray(self.stats["geometry"]["mean"], dtype=np.float32)
        self.geometry_std = np.asarray(self.stats["geometry"]["std"], dtype=np.float32)
        self.curvature_clip = float(self.stats["geometry"]["curvature_clip_abs"])
        self.velocity_mean = np.asarray(
            self.stats["velocity_m_s"]["mean"], dtype=np.float32
        )
        self.velocity_std = np.asarray(
            self.stats["velocity_m_s"]["std"], dtype=np.float32
        )
        self.pressure_mean = float(self.stats["pressure_relative_pa"]["mean"][0])
        self.pressure_std = float(self.stats["pressure_relative_pa"]["std"][0])
        numeric = (
            self.geometry_mean,
            self.geometry_std,
            self.velocity_mean,
            self.velocity_std,
            np.asarray([self.curvature_clip, self.pressure_mean, self.pressure_std]),
        )
        if (
            self.geometry_mean.shape != (3,)
            or self.geometry_std.shape != (3,)
            or self.velocity_mean.shape != (3,)
            or self.velocity_std.shape != (3,)
            or not all(np.isfinite(value).all() for value in numeric)
            or bool(np.any(self.geometry_std <= 0))
            or bool(np.any(self.velocity_std <= 0))
            or self.pressure_std <= 0
            or self.curvature_clip <= 0
        ):
            raise ValueError("field stats numeric contract drift")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.cases)

    def _features(self, coords: np.ndarray, geometry: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coords, dtype=np.float32)
        if self.input_variant == "xyz":
            return xyz.copy()
        geom = np.asarray(geometry, dtype=np.float32).copy()
        geom[:, 2] = np.clip(geom[:, 2], -self.curvature_clip, self.curvature_clip)
        geom = (geom - self.geometry_mean) / self.geometry_std
        return np.concatenate([xyz, geom], axis=1).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[int(index)]
        identity_seed = int.from_bytes(
            hashlib.sha256(case.case_id.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(
            (self.seed + self.epoch * 1_000_003 + identity_seed) % (2**63 - 1)
        )
        support_n = int(self.sampling["support_points"])
        query_n = int(self.sampling["query_points"])
        physics_n = int(self.sampling["physics_points"])
        wall_n = int(self.sampling["wall_points"])
        support_idx = _take_strict_volume(rng, case.is_wall, support_n)
        if self.sampling.get("query_mode") == "same":
            if query_n != support_n:
                raise ValueError(
                    "same query_mode requires equal support/query point counts"
                )
            query_idx = support_idx.copy()
        else:
            query_idx = _take_strict_volume(
                rng, case.is_wall, query_n, excluded=support_idx
            )
        physics_local = np.sort(rng.choice(query_n, size=physics_n, replace=False))
        wall_idx = _take(rng, len(case.wall_coords), wall_n)
        velocity = np.asarray(case.velocity[query_idx], dtype=np.float32)
        pressure = np.asarray(case.pressure[query_idx], dtype=np.float32)
        target = np.column_stack(
            [
                (velocity - self.velocity_mean) / self.velocity_std,
                (pressure - self.pressure_mean) / self.pressure_std,
            ]
        ).astype(np.float32)
        return {
            "case_id": case.case_id,
            "length_m": np.float32(case.length_m),
            "support_coords": np.asarray(case.coords[support_idx], dtype=np.float32),
            "support_features": self._features(
                case.coords[support_idx], case.geometry[support_idx]
            ),
            "query_coords": np.asarray(case.coords[query_idx], dtype=np.float32),
            "query_target": target,
            "query_region": np.asarray(case.region[query_idx], dtype=np.int8),
            "physics_local_indices": physics_local.astype(np.int64),
            "wall_coords": np.asarray(case.wall_coords[wall_idx], dtype=np.float32),
        }


def collate_volume_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate fixed-size case samples and create graph batch vectors."""
    if not samples:
        raise ValueError("cannot collate an empty case batch")
    support_coords = []
    support_features = []
    support_batch = []
    query_coords = []
    query_target = []
    query_region = []
    query_batch = []
    physics_coords = []
    physics_batch = []
    physics_length = []
    wall_coords = []
    wall_batch = []
    for graph_id, sample in enumerate(samples):
        ns = len(sample["support_coords"])
        nq = len(sample["query_coords"])
        npy = len(sample["physics_local_indices"])
        nw = len(sample["wall_coords"])
        support_coords.append(sample["support_coords"])
        support_features.append(sample["support_features"])
        support_batch.append(np.full(ns, graph_id, dtype=np.int64))
        query_coords.append(sample["query_coords"])
        query_target.append(sample["query_target"])
        query_region.append(sample["query_region"])
        query_batch.append(np.full(nq, graph_id, dtype=np.int64))
        local = sample["physics_local_indices"]
        physics_coords.append(sample["query_coords"][local])
        physics_batch.append(np.full(npy, graph_id, dtype=np.int64))
        physics_length.append(np.full(npy, sample["length_m"], dtype=np.float32))
        wall_coords.append(sample["wall_coords"])
        wall_batch.append(np.full(nw, graph_id, dtype=np.int64))
    return {
        "case_ids": [sample["case_id"] for sample in samples],
        "support_coords": torch.from_numpy(np.concatenate(support_coords)),
        "support_features": torch.from_numpy(np.concatenate(support_features)),
        "support_batch": torch.from_numpy(np.concatenate(support_batch)),
        "query_coords": torch.from_numpy(np.concatenate(query_coords)),
        "query_target": torch.from_numpy(np.concatenate(query_target)),
        "query_region": torch.from_numpy(np.concatenate(query_region)),
        "query_batch": torch.from_numpy(np.concatenate(query_batch)),
        "physics_coords": torch.from_numpy(np.concatenate(physics_coords)),
        "physics_batch": torch.from_numpy(np.concatenate(physics_batch)),
        "physics_length_m": torch.from_numpy(np.concatenate(physics_length)),
        "wall_coords": torch.from_numpy(np.concatenate(wall_coords)),
        "wall_batch": torch.from_numpy(np.concatenate(wall_batch)),
    }
