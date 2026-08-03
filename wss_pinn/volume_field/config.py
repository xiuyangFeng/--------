"""Configuration contract for the peak volume-field experiment matrix."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from wss_pinn.utils import ROOT, guard_write_path, sha256_file, sha256_json


EXPECTED_SPLIT_SHA256 = (
    "964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b"
)
VALID_ARCHITECTURES = {"pointnet", "pointnetpp"}
VALID_INPUTS = {"xyz", "xyz_geom"}
VALID_MODES = {"data_only", "pinn"}


DEFAULTS: dict[str, Any] = {
    "experiment": {
        "id": "",
        "mode": "",
        "architecture": "",
        "input_variant": "",
        "description": "",
    },
    "paths": {
        "split": (
            "wss_pinn/configs/splits/"
            "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_"
            "train138_test35_exclude_SHI_YUN_XI_v1.json"
        ),
        "sidecar_manifest": (
            "data_wss_pinn/volume_uvwp_peak_v1_train138_test35/manifest.json"
        ),
        "field_stats": (
            "data_wss_pinn/volume_uvwp_peak_v1_train138_test35/field_stats.json"
        ),
        "output_root": "outputs/wss_pinn/volume_uvwp_peak_v1",
        "run_dir": "",
    },
    "data": {
        "timestep": "peak",
        "train_roles": ["train"],
        "eval_roles": ["test"],
        "geometry_features": [
            "abscissa_norm",
            "local_radius",
            "curvature_signed_log1p",
        ],
        "pressure_target": "case_strict_volume_mean_centered_pa",
    },
    "sampling": {
        "support_points": 5000,
        "query_points": 5000,
        "physics_points": 512,
        "wall_points": 1024,
        "strategy": "uniform_volume_random",
        "query_mode": "independent",
        "resample_each_epoch": True,
        "seed": 1234,
    },
    "model": {
        "out_dim": 4,
        "activation": "silu",
        "pointnet": {
            "support_channels": [256, 512],
            "query_channels": [256, 512],
            "decoder_channels": [512, 256],
        },
        "pointnetpp": {
            "width": 32,
            "sa_ratios": [0.25, 0.25, 0.25],
            "sa_radius": [0.05, 0.10, 0.20],
            "sa_nsample": [128, 16, 16],
            "sa_center_counts": [125, 125, 32],
            "sa_grouping": ["knn_cover", "ball", "ball"],
            "sa_center_sampling": "fps",
            "fp_knn": 3,
            "head_hidden": 64,
        },
    },
    "physics": {
        "enabled": False,
        "density_kg_m3": 1060.0,
        "reference_velocity_m_s": 1.0,
        "shear_rate_epsilon_s_inv": 1e-6,
        "rheology": {
            "model": "carreau_yasuda",
            "mu_inf_pa_s": 0.0035,
            "mu_zero_pa_s": 0.16,
            "lambda_s": 8.2,
            "a": 0.64,
            "n": 0.2128,
        },
        "residual_epsilon": 1e-12,
    },
    "loss": {
        "velocity_component_weights": [1.0, 1.0, 1.0],
        "pressure_weight": 1.0,
        "continuity_weight": 1.0,
        "momentum_weight": 1.0,
        "no_slip_weight": 1.0,
    },
    "train": {
        "seed": 1234,
        "epochs": 400,
        "batch_cases": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 10,
        "min_learning_rate": 1e-5,
        "grad_clip_norm": 1.0,
        "device": "cuda",
        "precision": "float32",
        "num_workers": 0,
        "log_every_steps": 10,
        "checkpoint_every_epochs": 10,
        "milestone_epochs": [],
        "resume": None,
        # Deliberately unsupported: a PINN run must never load data-only weights.
        "init_checkpoint": None,
    },
    "cluster": {
        "partition": "GPU",
        "time": "24:00:00",
        "memory": "64G",
        "cpus": 8,
        "gpus": 1,
        "python": "/public/newhome/cy/.conda/envs/GNN/bin/python",
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class VolumeExperimentConfig:
    """Resolved configuration with hard scientific and path guards."""

    def __init__(self, payload: dict[str, Any], source: Path | None = None):
        self.payload = _deep_merge(DEFAULTS, payload)
        self.source = source
        self._resolve_paths()
        self.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "VolumeExperimentConfig":
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = ROOT / source
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(payload, source=source.resolve())

    def _resolve_paths(self) -> None:
        for key in (
            "split",
            "sidecar_manifest",
            "field_stats",
            "output_root",
            "run_dir",
        ):
            value = self.payload["paths"].get(key)
            if value:
                path = Path(value)
                self.payload["paths"][key] = str(
                    (path if path.is_absolute() else ROOT / path).resolve()
                )
        for key in ("resume", "init_checkpoint"):
            value = self.payload["train"].get(key)
            if value:
                path = Path(value)
                self.payload["train"][key] = str(
                    (path if path.is_absolute() else ROOT / path).resolve()
                )

    def validate(self) -> None:
        exp = self.payload["experiment"]
        mode = str(exp.get("mode", ""))
        architecture = str(exp.get("architecture", ""))
        input_variant = str(exp.get("input_variant", ""))
        if not exp.get("id"):
            raise ValueError("experiment.id is required")
        if mode not in VALID_MODES:
            raise ValueError(f"invalid experiment.mode={mode!r}")
        if architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"invalid architecture={architecture!r}")
        if input_variant not in VALID_INPUTS:
            raise ValueError(f"invalid input_variant={input_variant!r}")

        paths = self.payload["paths"]
        split = Path(paths["split"])
        expected_split = (
            ROOT
            / "wss_pinn/configs/splits/"
            "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_"
            "train138_test35_exclude_SHI_YUN_XI_v1.json"
        ).resolve()
        if split != expected_split:
            raise ValueError(f"volume-field split must remain frozen: {expected_split}")
        if split.exists() and sha256_file(split) != EXPECTED_SPLIT_SHA256:
            raise ValueError("frozen train138/test35 split hash drift")
        if paths.get("run_dir"):
            guard_write_path(paths["run_dir"])
        guard_write_path(paths["output_root"])
        guard_write_path(paths["sidecar_manifest"])
        guard_write_path(paths["field_stats"])

        if self.payload["data"]["timestep"] != "peak":
            raise ValueError("volume-field v1 is peak-only")
        if self.payload["data"]["geometry_features"] != [
            "abscissa_norm",
            "local_radius",
            "curvature_signed_log1p",
        ]:
            raise ValueError("geometry feature contract must remain the frozen three fields")
        if (
            self.payload["data"]["pressure_target"]
            != "case_strict_volume_mean_centered_pa"
        ):
            raise ValueError("pressure target must use the fixed strict-volume case gauge")
        sampling = self.payload["sampling"]
        for key in ("support_points", "query_points", "physics_points", "wall_points"):
            if int(sampling[key]) <= 0:
                raise ValueError(f"sampling.{key} must be positive")
        if int(sampling["physics_points"]) > int(sampling["query_points"]):
            raise ValueError("physics_points cannot exceed query_points")
        if sampling["strategy"] != "uniform_volume_random":
            raise ValueError("v1 sampling must be uniform over the full interior volume")
        if sampling["query_mode"] not in {"independent", "same"}:
            raise ValueError("sampling.query_mode must be independent or same")
        if (
            sampling["query_mode"] == "same"
            and int(sampling["support_points"]) != int(sampling["query_points"])
        ):
            raise ValueError("same query_mode requires support_points == query_points")
        if not bool(sampling["resample_each_epoch"]):
            raise ValueError("volume-field v1 must resample support/query each epoch")

        if int(self.payload["model"]["out_dim"]) != 4:
            raise ValueError("volume-field output must be four channels (u,v,w,p)")
        if self.payload["model"]["activation"] != "silu":
            raise ValueError("volume-field v1 query decoder is frozen to smooth SiLU")

        physics = self.payload["physics"]
        if float(physics["density_kg_m3"]) != 1060.0:
            raise ValueError("volume-field v1 density is frozen at 1060 kg/m^3")
        if float(physics["reference_velocity_m_s"]) != 1.0:
            raise ValueError("volume-field v1 reference velocity is frozen at 1 m/s")
        expected_rheology = {
            "model": "carreau_yasuda",
            "mu_inf_pa_s": 0.0035,
            "mu_zero_pa_s": 0.16,
            "lambda_s": 8.2,
            "a": 0.64,
            "n": 0.2128,
        }
        if any(physics["rheology"].get(key) != value for key, value in expected_rheology.items()):
            raise ValueError(f"volume-field v1 rheology is frozen: {expected_rheology}")

        enabled = bool(physics["enabled"])
        physics_weights = [
            float(self.payload["loss"][key])
            for key in ("continuity_weight", "momentum_weight", "no_slip_weight")
        ]
        if mode == "data_only":
            if enabled or any(weight != 0.0 for weight in physics_weights):
                raise ValueError("data_only must disable all physics terms")
        else:
            if not enabled or any(weight <= 0.0 for weight in physics_weights):
                raise ValueError(
                    "pinn must enable continuity, momentum and no-slip from step one"
                )
        if self.payload["train"].get("init_checkpoint"):
            raise ValueError(
                "warm-start is forbidden: use the same seed, never a data-only checkpoint"
            )
        if self.payload["train"]["precision"] not in {"float32", "float64"}:
            raise ValueError("train.precision must be float32 or float64")
        if int(self.payload["train"]["batch_cases"]) <= 0:
            raise ValueError("train.batch_cases must be positive")
        total_epochs = int(self.payload["train"]["epochs"])
        if total_epochs <= 0:
            raise ValueError("train.epochs must be positive")
        milestones = [
            int(value) for value in self.payload["train"].get("milestone_epochs", [])
        ]
        if milestones != sorted(set(milestones)):
            raise ValueError("train.milestone_epochs must be sorted and unique")
        if any(value <= 0 or value > total_epochs for value in milestones):
            raise ValueError("train.milestone_epochs must lie within train.epochs")
        if any(
            float(value) <= 0
            for value in self.payload["loss"]["velocity_component_weights"]
        ) or float(self.payload["loss"]["pressure_weight"]) <= 0:
            raise ValueError("all four supervised output weights must be positive")
        if not paths.get("run_dir"):
            raise ValueError("paths.run_dir is required")

    @property
    def mode(self) -> str:
        return str(self.payload["experiment"]["mode"])

    @property
    def architecture(self) -> str:
        return str(self.payload["experiment"]["architecture"])

    @property
    def input_variant(self) -> str:
        return str(self.payload["experiment"]["input_variant"])

    @property
    def run_dir(self) -> Path:
        return Path(self.payload["paths"]["run_dir"])

    @property
    def resolved_sha256(self) -> str:
        return sha256_json(self.payload)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]
