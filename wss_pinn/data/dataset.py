from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .sidecar import load_sidecar_manifest


class PhysicsCase:
    def __init__(
        self,
        manifest_path: str | Path,
        verify: bool = True,
        aggregate_row: dict[str, Any] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest = load_sidecar_manifest(self.manifest_path, verify=verify)
        static_path = Path(self.manifest["files"]["static"]["path"])
        field_path = Path(self.manifest["files"]["peak_fields"]["path"])
        with np.load(static_path, allow_pickle=False) as source:
            self.static = {key: np.asarray(source[key]) for key in source.files}
        with np.load(field_path, allow_pickle=False) as source:
            self.fields = {key: np.asarray(source[key]) for key in source.files}
        row = aggregate_row or {}
        self.case_id = str(
            self.manifest.get(
                "canonical_id",
                row.get("canonical_id", self.manifest["case_id"]),
            )
        )
        self.cohort = str(self.manifest["cohort"])
        self.role = str(self.manifest.get("role", row.get("role", "unspecified")))

    @property
    def velocity_scale(self) -> float:
        return float(self.manifest["characteristic_scales"]["velocity_m_s"])

    @property
    def pressure_scale(self) -> float:
        rho = float(self.manifest["characteristic_scales"]["density_kg_m3"])
        return rho * self.velocity_scale**2

    def sample(
        self,
        *,
        wall: int,
        near_wall: int,
        core: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(int(seed))

        def choose(length: int, count: int) -> np.ndarray:
            if length <= 0:
                raise ValueError("empty sidecar slot")
            return rng.choice(length, size=min(int(count), length), replace=False)

        wi = choose(len(self.static["wall_coords"]), wall)
        ni = choose(len(self.static["near_wall_coords"]), near_wall)
        ci = choose(len(self.static["core_coords"]), core)
        pressure = np.concatenate(
            [
                self.fields["near_wall_pressure"][ni],
                self.fields["core_pressure"][ci],
            ]
        ).astype(np.float32)
        pressure = (pressure - pressure.mean()) / max(self.pressure_scale, 1e-8)
        return {
            "descriptor": self.static["geometry_descriptor"].astype(np.float32),
            "wall_coords": self.static["wall_coords"][wi].astype(np.float32),
            "wall_wss": self.fields["wall_wss"][wi].astype(np.float32),
            "interior_coords": np.concatenate(
                [
                    self.static["near_wall_coords"][ni],
                    self.static["core_coords"][ci],
                ]
            ).astype(np.float32),
            "velocity": (
                np.concatenate(
                    [
                        self.fields["near_wall_velocity"][ni],
                        self.fields["core_velocity"][ci],
                    ]
                )
                / self.velocity_scale
            ).astype(np.float32),
            "pressure": pressure,
        }


class PhysicsDataset:
    def __init__(
        self,
        aggregate_manifest: str | Path,
        verify: bool = True,
        roles: list[str] | tuple[str, ...] | None = None,
    ):
        payload = json.loads(Path(aggregate_manifest).read_text(encoding="utf-8"))
        self.split_label = str(payload["split_label"])
        allowed = set(roles) if roles else None
        self.cases = []
        for row in payload["cases"]:
            case = PhysicsCase(
                row["manifest"], verify=verify, aggregate_row=row
            )
            if allowed is None or case.role in allowed:
                self.cases.append(case)
        if not self.cases:
            raise ValueError(f"physics dataset has no cases for roles={roles}")

    def epoch_samples(
        self, train_config: dict[str, Any], epoch: int
    ) -> list[dict[str, np.ndarray]]:
        base_seed = int(train_config["seed"]) + int(epoch) * 10007
        cases = self.cases
        batch_cases = int(train_config.get("batch_cases") or 0)
        if 0 < batch_cases < len(cases):
            rng = np.random.default_rng(base_seed + 53)
            selected = np.sort(
                rng.choice(len(cases), size=batch_cases, replace=False)
            )
            cases = [cases[int(index)] for index in selected]
        samples = []
        for index, case in enumerate(cases):
            sample = case.sample(
                wall=int(train_config["batch_wall"]),
                near_wall=int(train_config["batch_near_wall"]),
                core=int(train_config["batch_core"]),
                seed=base_seed + index * 97,
            )
            sample["case_id"] = getattr(case, "case_id", getattr(case, "name", "unknown"))
            samples.append(sample)
        return samples


def as_tensor(
    value: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)
