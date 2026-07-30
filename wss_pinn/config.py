from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .utils import ROOT, guard_write_path, sha256_json


VALID_STAGES = {"p0a", "p0b", "p0c", "p0d", "p1", "f0u", "f0up", "f1"}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


DEFAULTS: dict[str, Any] = {
    "experiment": {"id": "", "stage": "", "diagnostic_only": False},
    "paths": {
        "raw_root": "data_new",
        "wss_bundle_root": "data_wss_min",
        "sidecar_root": "data_wss_pinn/pilot_v1",
        "output_root": "outputs/wss_pinn",
    },
    "data": {
        "split_path": "wss_pinn/configs/pilot_cases.json",
        "split_label": "reused_development_screen",
        "cohorts": ["AG", "AAA", "ILO"],
        "timesteps": ["peak"],
        "train_roles": ["train", "unspecified"],
        "eval_roles": ["train", "test", "unspecified"],
    },
    "sampling": {
        "wall_points": 5000,
        "near_wall_points": 8000,
        "core_points": 8000,
        "seed": 1234,
        "manifest_path": "data_wss_pinn/pilot_v1/sampling_manifest.json",
    },
    "model": {
        "predict_direct_wss": True,
        "predict_velocity": True,
        "predict_pressure": False,
        "geometry_hidden": 64,
        "geometry_latent": 32,
        "field_hidden": 128,
        "field_layers": 4,
        "fourier_frequencies": 6,
        "architecture_version": "shared_v1",
    },
    "loss": {
        "direct_wss_weight": 1.0,
        "velocity_data_weight": 1.0,
        "pressure_data_weight": 0.0,
        "continuity_weight": 0.0,
        "no_slip_weight": 0.0,
        "wss_physics_weight": 0.0,
        "momentum_weight": 0.0,
    },
    "physics": {
        "density_kg_m3": 1060.0,
        "rheology": {
            "model": "carreau_yasuda",
            "mu_inf_pa_s": 0.0035,
            "mu_zero_pa_s": 0.16,
            "lambda_s": 8.2,
            "a": 0.64,
            "n": 0.2128,
            "source": "data_new/AAA/ruputer/DING_JUN_FENG/udf-inlet4.c",
        },
        "characteristic_scales": {
            "length": "per_case_coord_scale_m",
            "velocity": "per_case_peak_p95_m_s",
            "pressure": "rho_u2",
        },
    },
    "train": {
        "seed": 1234,
        "epochs": 2000,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "precision": "float32",
        "device": "cuda",
        "batch_wall": 512,
        "batch_near_wall": 512,
        "batch_core": 512,
        "checkpoint_every": 100,
        "log_every": 10,
        "batch_cases": 0,
        "init_checkpoint": None,
        "resume": None,
        "run_dir": "",
    },
    "cluster": {
        "partition": "GPU",
        "qos": "",
        "time": "24:00:00",
        "memory": "32G",
        "cpus": 8,
        "gpus": 1,
        "python": "/public/newhome/cy/.conda/envs/GNN/bin/python",
    },
}


class ExperimentConfig:
    def __init__(self, payload: dict[str, Any], source: Path | None = None):
        self.payload = _deep_merge(DEFAULTS, payload)
        self.source = source
        self._resolve_paths()
        self.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = ROOT / source
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload, source=source.resolve())

    def _resolve_paths(self) -> None:
        for key in ("raw_root", "wss_bundle_root", "sidecar_root", "output_root"):
            value = Path(self.payload["paths"][key])
            self.payload["paths"][key] = str(
                (value if value.is_absolute() else ROOT / value).resolve()
            )
        for section, key in (
            ("data", "split_path"),
            ("sampling", "manifest_path"),
            ("train", "run_dir"),
            ("train", "init_checkpoint"),
            ("train", "resume"),
        ):
            value = self.payload[section].get(key)
            if value:
                path = Path(value)
                self.payload[section][key] = str(
                    (path if path.is_absolute() else ROOT / path).resolve()
                )

    def validate(self) -> None:
        exp = self.payload["experiment"]
        stage = str(exp.get("stage", "")).lower()
        if stage not in VALID_STAGES:
            raise ValueError(f"invalid experiment.stage: {stage!r}")
        if not exp.get("id"):
            raise ValueError("experiment.id is required")
        if self.payload["data"]["split_label"] != "reused_development_screen":
            raise ValueError("pilot split must be labelled reused_development_screen")

        paths = self.payload["paths"]
        if Path(paths["raw_root"]) != (ROOT / "data_new").resolve():
            raise ValueError("paths.raw_root must remain the frozen data_new root")
        if Path(paths["wss_bundle_root"]) != (ROOT / "data_wss_min").resolve():
            raise ValueError("paths.wss_bundle_root must remain the frozen data_wss_min root")
        guard_write_path(paths["sidecar_root"])
        guard_write_path(paths["output_root"])
        if self.payload["train"].get("run_dir"):
            guard_write_path(self.payload["train"]["run_dir"])
        guard_write_path(self.payload["sampling"]["manifest_path"])

        sampling = self.payload["sampling"]
        for key in ("wall_points", "near_wall_points", "core_points"):
            if int(sampling[key]) <= 0:
                raise ValueError(f"sampling.{key} must be positive")

        model = self.payload["model"]
        loss = self.payload["loss"]
        if not model["predict_direct_wss"] or float(loss["direct_wss_weight"]) <= 0:
            raise ValueError("all F stages retain a supervised WSS_direct output")
        if float(loss["wss_physics_weight"]) != 0:
            raise ValueError("WSS physics is outside the F1 implementation scope")
        if float(loss["momentum_weight"]) != 0:
            raise ValueError("momentum must remain zero through F1")

        if stage == "f0u":
            expected = (
                bool(model["predict_velocity"])
                and not bool(model["predict_pressure"])
                and float(loss["velocity_data_weight"]) > 0
                and float(loss["pressure_data_weight"]) == 0
                and float(loss["continuity_weight"]) == 0
                and float(loss["no_slip_weight"]) == 0
            )
            if not expected:
                raise ValueError("F0-U must be velocity data-only with pressure/physics off")
        elif stage == "f0up":
            expected = (
                bool(model["predict_velocity"])
                and bool(model["predict_pressure"])
                and float(loss["velocity_data_weight"]) > 0
                and float(loss["pressure_data_weight"]) > 0
                and float(loss["continuity_weight"]) == 0
                and float(loss["no_slip_weight"]) == 0
            )
            if not expected:
                raise ValueError("F0-UP must add gauge pressure supervision only")
            if not (
                exp.get("scientific_gate_control_run_dir")
                or exp.get("control_run_dir")
            ):
                raise ValueError(
                    "F0-UP requires scientific_gate_control_run_dir or control_run_dir"
                )
        elif stage == "f1":
            common = (
                bool(model["predict_velocity"])
                and bool(model["predict_pressure"])
                and float(loss["pressure_data_weight"]) > 0
            )
            continuity = float(loss["continuity_weight"])
            no_slip = float(loss["no_slip_weight"])
            if bool(exp.get("diagnostic_only")):
                expected = common and (continuity > 0 or no_slip > 0)
                message = "diagnostic F1 requires at least one first-order loss"
            else:
                expected = common and continuity > 0 and no_slip > 0
                message = "F1 requires pressure, continuity and no-slip"
            if not expected:
                raise ValueError(message)
            if not exp.get("control_run_dir"):
                raise ValueError("F1 requires a paired F0-UP control_run_dir")

        if stage.startswith("f") and not self.payload["train"].get("run_dir"):
            raise ValueError("train.run_dir is required for F stages")
        if self.payload["train"]["precision"] not in {"float32", "float64"}:
            raise ValueError("train.precision must be float32 or float64")
        if int(self.payload["train"].get("batch_cases") or 0) < 0:
            raise ValueError("train.batch_cases must be non-negative")
        if self.payload["train"].get("resume") and self.payload["train"].get(
            "init_checkpoint"
        ):
            raise ValueError("train.resume and train.init_checkpoint are mutually exclusive")
        for key in ("train_roles", "eval_roles"):
            roles = self.payload["data"].get(key)
            if not isinstance(roles, list) or not roles:
                raise ValueError(f"data.{key} must be a non-empty list")

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)

    @property
    def stage(self) -> str:
        return self.payload["experiment"]["stage"].lower()

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment"]["id"])

    @property
    def run_dir(self) -> Path:
        return Path(self.payload["train"]["run_dir"])

    @property
    def resolved_sha256(self) -> str:
        return sha256_json(self.payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]
