"""Strict configuration contract for ``volume_uvwp_bc_rcr_v4``.

This route is intentionally isolated from the completed V1/V2/V3/field-v4
implementations.  The JSON files are fully resolved experiment records rather
than a loose bag of overrides: rejecting drift before allocating a GPU is much
cheaper than discovering a scientifically incomparable run later.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wss_pinn.utils import ROOT, guard_write_path, sha256_json


ROUTE = "volume_uvwp_bc_rcr_v4"
TEMPORAL_MODES = {"steady_peak", "transient_81"}
BACKBONES = {"pointnet", "pointnetpp"}
TRAINING_MODES = {"data", "data_bc", "bc_pde_fixed", "bc_pde_ema"}
OUTLET_ORDER = ("out-le", "out-li", "out-ri", "out-re")
SEEDS = (1234, 2345, 3456)

DATA_ROOT = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35"
OUTPUT_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4"
CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_bc_rcr_v4"
SPLIT_PATH = (
    ROOT
    / "wss_pinn/configs/splits/"
    "split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_train138_test35_cv5_s1234.json"
)
MANIFEST_PATH = DATA_ROOT / "manifest.json"
STATS_PATH = DATA_ROOT / "train138_stats.json"


DEFAULTS: dict[str, Any] = {
    "route": ROUTE,
    "schema_version": 1,
    "experiment": {
        "id": "",
        "temporal_mode": "steady_peak",
        "momentum_mode": "quasi_steady",
        "time_input": False,
        "backbone": "pointnet",
        "training_mode": "data",
        "seed": 1234,
    },
    "paths": {
        "split": str(SPLIT_PATH.relative_to(ROOT)),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "stats": str(STATS_PATH.relative_to(ROOT)),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "run_dir": "",
    },
    "data": {
        "train_roles": ["train"],
        "validation_roles": [],
        "eval_roles": ["test"],
        "pressure_gauge_steady": "case_strict_volume_mean_centered_pa",
        "pressure_gauge_transient": "raw_fluent_gauge_pa_distal_pressure_zero",
        "outlet_order": list(OUTLET_ORDER),
        "transient_phases_per_case_epoch": 1,
    },
    "sampling": {
        "support_points": 5000,
        "query_points": 5000,
        "physics_points": 512,
        "wall_points": 512,
        "inlet_points": 256,
        "resample_each_epoch": True,
        "time_reset_buffer_s": 0.005,
    },
    "model": {
        "activation": "tanh",
        "support_in_dim": 6,
        "bc_in_dim": 18,
        "bc_channels": [64, 128],
        "latent_dim": 256,
        "decoder_channels": [256, 256, 128],
        "pointnet_channels": [64, 128, 256],
        "pointnetpp": {
            "centroids": 128,
            "neighbors": 32,
            "local_channels": [64, 128, 256],
        },
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
        },
        "shear_rate_epsilon_s_inv": 1.0e-6,
        "bc_enabled": False,
        "pde_enabled": False,
        "lambda_bc": 0.0,
        "lambda_pde": 0.0,
        "rcr_pressure_definition": "predicted_face_area_mean_raw_gauge",
        "rcr_flow_basis": "mass_flow_kg_s",
    },
    "loss_balancing": {
        "mode": "disabled",
        "detach_ratio": True,
        "update_interval_steps": 50,
        "ema_beta": 0.99,
        "weight_smoothing_beta": 0.90,
        "epsilon": 1.0e-12,
        "alpha_start": 0.10,
        "alpha_end": 1.00,
        "physics_ramp_epochs": 200,
        "lambda_min": 1.0e-4,
        "lambda_max": 10.0,
        "gradient_check_interval_steps": 50,
    },
    "train": {
        "seed": 1234,
        "batch_cases": 1,
        "max_epochs": 10000,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 100,
        "min_learning_rate": 1.0e-5,
        "grad_clip_norm": 1.0,
        "checkpoint_every_epochs": 100,
        "log_every_steps": 10,
        "num_workers": 0,
        "device": "cuda",
        "precision": "float32",
        "milestone_epochs": [1000, 2500, 5000, 7500, 10000],
        "resume": None,
        "init_checkpoint": None,
    },
    "convergence": {
        "min_epochs": 1000,
        "max_epochs": 10000,
        "window_epochs": 200,
        "consecutive_windows": 3,
        "relative_change_threshold": 0.005,
        "require_min_lr": True,
        "lambda_bound_fraction_limit": 0.50,
        "gradient_clip_fraction_limit": 0.50,
    },
    "cluster": {
        "partition": "GPU",
        "gpus": 1,
        "cpus": 8,
        "memory": "64G",
        "time": "0",
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


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


@dataclass(frozen=True)
class V4Config:
    payload: dict[str, Any]
    source: Path | None = None

    @property
    def experiment(self) -> dict[str, Any]:
        return self.payload["experiment"]

    @property
    def temporal_mode(self) -> str:
        return str(self.experiment["temporal_mode"])

    @property
    def backbone(self) -> str:
        return str(self.experiment["backbone"])

    @property
    def training_mode(self) -> str:
        return str(self.experiment["training_mode"])

    @property
    def seed(self) -> int:
        return int(self.experiment["seed"])

    @property
    def run_dir(self) -> Path:
        return Path(self.payload["paths"]["run_dir"])

    @property
    def resolved_sha256(self) -> str:
        return sha256_json(self.payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def _apply_mode_contract(payload: dict[str, Any]) -> None:
    mode = str(payload["experiment"]["training_mode"])
    physics = payload["physics"]
    balance = payload["loss_balancing"]
    if mode == "data":
        expected = (False, False, 0.0, 0.0, "disabled")
    elif mode == "data_bc":
        expected = (True, False, 1.0, 0.0, "disabled")
    elif mode == "bc_pde_fixed":
        expected = (True, True, 1.0, 1.0, "fixed")
    else:
        expected = (True, True, 1.0, 1.0, "ema_loss_ratio")
    actual = (
        bool(physics["bc_enabled"]),
        bool(physics["pde_enabled"]),
        float(physics["lambda_bc"]),
        float(physics["lambda_pde"]),
        str(balance["mode"]),
    )
    if actual != expected:
        raise ValueError(f"training mode {mode!r} requires {expected}, got {actual}")


def validate_config(payload: dict[str, Any], *, require_assets: bool = False) -> None:
    if payload.get("route") != ROUTE:
        raise ValueError(f"route must be {ROUTE!r}")
    experiment = payload["experiment"]
    if not str(experiment.get("id", "")):
        raise ValueError("experiment.id is required")
    temporal = str(experiment["temporal_mode"])
    backbone = str(experiment["backbone"])
    mode = str(experiment["training_mode"])
    seed = int(experiment["seed"])
    if temporal not in TEMPORAL_MODES or backbone not in BACKBONES:
        raise ValueError("unsupported temporal mode or backbone")
    if mode not in TRAINING_MODES or seed not in SEEDS:
        raise ValueError("unsupported training mode or seed")
    expected_time = temporal == "transient_81"
    if bool(experiment["time_input"]) != expected_time:
        raise ValueError("time_input must match temporal_mode")
    expected_momentum = "transient_autograd" if expected_time else "quasi_steady"
    if experiment["momentum_mode"] != expected_momentum:
        raise ValueError("momentum_mode must match temporal_mode")
    if payload["data"]["validation_roles"]:
        raise ValueError("V4 has no validation role")
    if tuple(payload["data"]["outlet_order"]) != OUTLET_ORDER:
        raise ValueError(f"outlet order must remain {OUTLET_ORDER}")
    if payload["model"]["activation"] not in {"tanh", "silu"}:
        raise ValueError("strong-form decoder requires a smooth activation")
    if int(payload["model"]["support_in_dim"]) != 6:
        raise ValueError("support input is xyz + three geometry features")
    if int(payload["model"]["bc_in_dim"]) != 18:
        raise ValueError("BC input is A_in,Q_actual_peak + 4x(A,R1,R2,C)")
    if payload["train"].get("init_checkpoint"):
        raise ValueError("cross-run warm start is forbidden")
    if payload["train"]["precision"] != "float32":
        raise ValueError("V4 strong-form training is float32 only; AMP is disabled")
    if int(payload["sampling"]["physics_points"]) <= 0:
        raise ValueError("physics_points must be positive even for paired DATA configs")
    if not bool(payload["sampling"]["resample_each_epoch"]):
        raise ValueError("support/query/time sampling must resample each epoch")
    if not bool(payload["loss_balancing"]["detach_ratio"]):
        raise ValueError("EMA loss ratio must be detached")
    if int(payload["convergence"]["max_epochs"]) != int(payload["train"]["max_epochs"]):
        raise ValueError("train/convergence max_epochs mismatch")
    _apply_mode_contract(payload)

    paths = payload["paths"]
    expected = {
        "split": SPLIT_PATH.resolve(),
        "manifest": MANIFEST_PATH.resolve(),
        "stats": STATS_PATH.resolve(),
        "output_root": OUTPUT_ROOT.resolve(),
    }
    for key, frozen in expected.items():
        if Path(paths[key]).resolve() != frozen:
            raise ValueError(f"paths.{key} must remain isolated at {frozen}")
        if require_assets and key in {"split", "manifest", "stats"} and not frozen.is_file():
            raise FileNotFoundError(frozen)
    run_dir = guard_write_path(paths["run_dir"])
    if not run_dir.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("run_dir must be inside the V4 output root")
    resume = payload["train"].get("resume")
    if resume:
        resume_path = _resolve(resume)
        if resume_path.parent != (run_dir / "checkpoints").resolve():
            raise ValueError("resume is restricted to the same run checkpoints directory")


def load_config(path: str | Path, *, require_assets: bool = False) -> V4Config:
    source = _resolve(path)
    payload = _deep_merge(DEFAULTS, json.loads(source.read_text(encoding="utf-8")))
    for key in ("split", "manifest", "stats", "output_root", "run_dir"):
        payload["paths"][key] = str(_resolve(payload["paths"][key]))
    if payload["train"].get("resume"):
        payload["train"]["resume"] = str(_resolve(payload["train"]["resume"]))
    validate_config(payload, require_assets=require_assets)
    return V4Config(payload=payload, source=source)
