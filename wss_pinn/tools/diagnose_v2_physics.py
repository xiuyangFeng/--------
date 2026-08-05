"""No-retraining diagnostics for the SAME5K-E7500 V2 PINN.

The audit answers two narrowly scoped questions:

1. Does evaluating PDE derivatives exactly at PointNet++ support points make
   the residual look artificially small compared with independent queries?
2. Does the frozen peak CFD field satisfy the route's quasi-steady
   Carreau--Yasuda equation, and how does the result change after adding a
   central-difference ``du/dt`` term from adjacent exported timesteps?

This module never trains or mutates a checkpoint.  It writes machine-readable
JSON only under ``outputs/wss_pinn/audits``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from wss_pinn.config import VolumeExperimentConfig
from wss_pinn.data.alignment import align_raw_interior_to_bundle
from wss_pinn.data.builder import rotate_velocity_to_registered
from wss_pinn.data.dataset import VolumeFieldDataset, _take_strict_volume
from wss_pinn.data.raw_io import list_steps, read_interior, step_file
from wss_pinn.evaluate import _case_seed
from wss_pinn.models import build_volume_model
from wss_pinn.physics.residuals import generalized_newtonian_residuals
from wss_pinn.utils import ROOT, atomic_write_json, sha256_file, utc_now


DEFAULT_RUN = (
    ROOT
    / "outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/runs/"
    "VF-PNPP-XYZG-PINN-SAME5K-E7500-s1234-v2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/wss_pinn/audits/"
    "volume_uvwp_peak_same5k_e7500_v2_residual_diagnosis"
)
DEFAULT_CFD_CASES = (
    "AG/fast/FAN_JIAN_MING",
    "AAA/ruputer/KANG_YONG",
    "ILO/LI_YOU_ZHI-0/before",
)


def _rms_numpy(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def _finite_median(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def _network_condition_metrics(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    coords_np: np.ndarray,
    *,
    case_length_m: float,
    field_stats: dict[str, Any],
    physics_config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    coords = torch.as_tensor(coords_np, device=device, dtype=dtype).requires_grad_(True)
    batch = torch.zeros(len(coords), device=device, dtype=torch.long)
    prediction = model.decode_query(encoded, coords, batch)
    residuals = generalized_newtonian_residuals(
        prediction,
        coords,
        coords.new_full((len(coords),), float(case_length_m)),
        field_stats=field_stats,
        physics_config=physics_config,
    )
    momentum_hat = residuals["momentum_hat"]
    gradient_hat = residuals["velocity_gradient_hat"]
    metrics = {
        "continuity_rms_dimensionless": float(
            torch.sqrt(torch.mean(torch.square(residuals["continuity_hat"])))
            .detach()
            .cpu()
        ),
        "continuity_rms_s_inv": float(
            torch.sqrt(torch.mean(torch.square(residuals["continuity_phys_s_inv"])))
            .detach()
            .cpu()
        ),
        "momentum_rms_dimensionless": float(
            torch.sqrt(torch.mean(torch.square(momentum_hat))).detach().cpu()
        ),
        "momentum_rms_pa_per_m": float(
            torch.sqrt(
                torch.mean(torch.square(residuals["momentum_phys_pa_per_m"]))
            )
            .detach()
            .cpu()
        ),
        "velocity_gradient_hat_fro_rms": float(
            torch.sqrt(torch.mean(torch.sum(torch.square(gradient_hat), dim=(1, 2))))
            .detach()
            .cpu()
        ),
        "shear_rate_rms_s_inv": float(
            torch.sqrt(torch.mean(torch.square(residuals["shear_rate_s_inv"])))
            .detach()
            .cpu()
        ),
    }
    velocity = residuals["velocity_phys_m_s"].detach().cpu().numpy()
    pressure = residuals["pressure_relative_pa"].detach().cpu().numpy()
    del prediction, residuals, momentum_hat, gradient_hat, coords
    return metrics, velocity, pressure


def audit_network_queries(
    config: VolumeExperimentConfig,
    checkpoint: Path,
    *,
    device_name: str,
    physics_points: int,
    jitter_amplitudes: tuple[float, ...],
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for query audit but unavailable")
    dtype = (
        torch.float64
        if config["train"]["precision"] == "float64"
        else torch.float32
    )
    stats_path = Path(config["paths"]["field_stats"])
    field_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    dataset = VolumeFieldDataset(
        config["paths"]["sidecar_manifest"],
        stats_path,
        roles=config["data"]["eval_roles"],
        input_variant=config.input_variant,
        sampling=config["sampling"],
        seed=int(config["train"]["seed"]),
    )
    model = build_volume_model(config).to(device=device, dtype=dtype)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)

    case_reports: list[dict[str, Any]] = []
    for case_index, case in enumerate(dataset.cases, start=1):
        rng = np.random.default_rng(
            _case_seed(case.case_id, int(config["sampling"]["seed"]))
        )
        support_idx = _take_strict_volume(
            rng,
            case.is_wall,
            int(config["sampling"]["support_points"]),
        )
        count = min(int(physics_points), len(support_idx))
        exact_idx = np.sort(rng.choice(support_idx, size=count, replace=False))
        independent_idx = _take_strict_volume(
            rng, case.is_wall, count, excluded=support_idx
        )
        support_coords_np = np.asarray(case.coords[support_idx], dtype=np.float64)
        support_coords = torch.as_tensor(
            support_coords_np, device=device, dtype=dtype
        )
        support_features = torch.as_tensor(
            dataset._features(case.coords[support_idx], case.geometry[support_idx]),
            device=device,
            dtype=dtype,
        )
        support_batch = torch.zeros(len(support_idx), device=device, dtype=torch.long)
        with torch.no_grad():
            encoded = model.encode_support(
                support_coords,
                support_features,
                support_batch,
                unit_ids=[case.case_id],
                epoch=0,
                global_seed=int(config["train"]["seed"]),
                evaluation=True,
            )
        support_tree = cKDTree(support_coords_np)
        exact_coords = np.asarray(case.coords[exact_idx], dtype=np.float64)
        independent_coords = np.asarray(
            case.coords[independent_idx], dtype=np.float64
        )
        jitter_rng = np.random.default_rng(
            _case_seed(case.case_id + "/jitter", int(config["sampling"]["seed"]))
        )
        directions = jitter_rng.normal(size=exact_coords.shape)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)
        condition_coords: dict[str, np.ndarray] = {
            "exact_support": exact_coords,
            "independent_volume": independent_coords,
        }
        for amplitude in jitter_amplitudes:
            label = f"jittered_support_{amplitude:.0e}"
            condition_coords[label] = exact_coords + float(amplitude) * directions

        conditions: dict[str, Any] = {}
        exact_velocity: np.ndarray | None = None
        exact_pressure: np.ndarray | None = None
        for label, coords_np in condition_coords.items():
            metrics, velocity, pressure = _network_condition_metrics(
                model,
                encoded,
                coords_np,
                case_length_m=case.length_m,
                field_stats=field_stats,
                physics_config=config["physics"],
                device=device,
                dtype=dtype,
            )
            nearest_distance = support_tree.query(coords_np, k=1, workers=-1)[0]
            metrics["nearest_support_distance_mean_normalized"] = float(
                np.mean(nearest_distance)
            )
            metrics["nearest_support_distance_p95_normalized"] = float(
                np.quantile(nearest_distance, 0.95)
            )
            if label == "exact_support":
                exact_velocity = velocity
                exact_pressure = pressure
                metrics["matched_velocity_change_rms_m_s"] = 0.0
                metrics["matched_pressure_change_rms_pa"] = 0.0
            elif label.startswith("jittered_support"):
                assert exact_velocity is not None and exact_pressure is not None
                metrics["matched_velocity_change_rms_m_s"] = _rms_numpy(
                    velocity - exact_velocity
                )
                metrics["matched_pressure_change_rms_pa"] = _rms_numpy(
                    pressure - exact_pressure
                )
            conditions[label] = metrics

        case_reports.append(
            {
                "case_id": case.case_id,
                "strict_volume_points": int(np.sum(~np.asarray(case.is_wall))),
                "support_points": int(len(support_idx)),
                "physics_points": int(count),
                "conditions": conditions,
            }
        )
        print(
            json.dumps(
                {
                    "event": "network_query_audited",
                    "index": case_index,
                    "total": len(dataset.cases),
                    "case_id": case.case_id,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del encoded, support_coords, support_features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = list(case_reports[0]["conditions"])
    aggregate: dict[str, Any] = {"condition_case_balanced": {}}
    for label in labels:
        scalar_names = list(case_reports[0]["conditions"][label])
        aggregate["condition_case_balanced"][label] = {
            name: {
                "mean": _finite_mean(
                    [row["conditions"][label].get(name, float("nan")) for row in case_reports]
                ),
                "median": _finite_median(
                    [row["conditions"][label].get(name, float("nan")) for row in case_reports]
                ),
            }
            for name in scalar_names
        }
    ratio_metrics = (
        "continuity_rms_dimensionless",
        "momentum_rms_dimensionless",
        "momentum_rms_pa_per_m",
        "velocity_gradient_hat_fro_rms",
    )
    aggregate["condition_over_exact_case_ratio"] = {}
    for label in labels:
        if label == "exact_support":
            continue
        aggregate["condition_over_exact_case_ratio"][label] = {}
        for name in ratio_metrics:
            ratios = []
            for row in case_reports:
                denominator = float(row["conditions"]["exact_support"][name])
                numerator = float(row["conditions"][label][name])
                ratios.append(numerator / max(denominator, 1e-30))
            aggregate["condition_over_exact_case_ratio"][label][name] = {
                "mean": _finite_mean(ratios),
                "median": _finite_median(ratios),
                "min": float(np.min(ratios)),
                "max": float(np.max(ratios)),
            }

    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "purpose": "same-support versus independent/jittered PDE query diagnostic",
        "training_performed": False,
        "config": str(config.source or DEFAULT_RUN / "resolved_config.json"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device),
        "jitter_note": (
            "Jittered points are a derivative probe only and may leave the vessel; "
            "the independent-volume condition is the primary in-domain comparison."
        ),
        "aggregate": aggregate,
        "cases": case_reports,
    }


def _solver_time_step_s(case_dir: Path) -> tuple[float, list[str]]:
    pattern = re.compile(r"/solve/set/time-step\s+([0-9.eE+\-]+)")
    matches: list[tuple[str, float]] = []
    for journal in sorted(case_dir.glob("*.jou")):
        text = journal.read_text(encoding="utf-8", errors="replace")
        for value in pattern.findall(text):
            matches.append((str(journal.resolve()), float(value)))
    unique = sorted({value for _, value in matches})
    if len(unique) != 1 or unique[0] <= 0:
        raise ValueError(f"cannot establish unique positive Fluent time-step in {case_dir}")
    return unique[0], [path for path, _ in matches]


def _quadratic_derivatives(
    coords_m: np.ndarray,
    fields: np.ndarray,
    query_coords_m: np.ndarray,
    query_fields: np.ndarray,
    tree: cKDTree,
    *,
    neighbors: int,
) -> dict[str, np.ndarray]:
    distances, local_indices = tree.query(
        query_coords_m, k=int(neighbors), workers=-1
    )
    distances = np.asarray(distances, dtype=np.float64)
    local_indices = np.asarray(local_indices, dtype=np.int64)
    count = len(query_coords_m)
    grad_u = np.full((count, 3, 3), np.nan, dtype=np.float64)
    hess_u = np.full((count, 3, 3, 3), np.nan, dtype=np.float64)
    grad_p = np.full((count, 3), np.nan, dtype=np.float64)
    condition = np.full(count, np.inf, dtype=np.float64)
    rank = np.zeros(count, dtype=np.int64)
    radius = np.full(count, np.nan, dtype=np.float64)
    for row in range(count):
        neighbor = local_indices[row]
        delta = coords_m[neighbor] - query_coords_m[row]
        h = float(max(distances[row, -1], 1e-12))
        s = delta / h
        design = np.column_stack(
            [
                np.ones(len(neighbor)),
                s[:, 0],
                s[:, 1],
                s[:, 2],
                s[:, 0] ** 2,
                s[:, 1] ** 2,
                s[:, 2] ** 2,
                s[:, 0] * s[:, 1],
                s[:, 0] * s[:, 2],
                s[:, 1] * s[:, 2],
            ]
        )
        weight = np.exp(-4.0 * np.square(distances[row] / h))
        weighted_design = design * np.sqrt(weight)[:, None]
        centered_fields = fields[neighbor] - query_fields[row]
        weighted_fields = centered_fields * np.sqrt(weight)[:, None]
        coefficient, _, fitted_rank, singular = np.linalg.lstsq(
            weighted_design, weighted_fields, rcond=None
        )
        rank[row] = int(fitted_rank)
        condition[row] = (
            float(singular[0] / singular[-1])
            if len(singular) and singular[-1] > 0
            else float("inf")
        )
        radius[row] = h
        if fitted_rank < 10:
            continue
        grad_u[row] = coefficient[1:4, :3].T / h
        grad_p[row] = coefficient[1:4, 3] / h
        for component in range(3):
            hess_u[row, component, 0, 0] = 2.0 * coefficient[4, component] / h**2
            hess_u[row, component, 1, 1] = 2.0 * coefficient[5, component] / h**2
            hess_u[row, component, 2, 2] = 2.0 * coefficient[6, component] / h**2
            hess_u[row, component, 0, 1] = hess_u[row, component, 1, 0] = (
                coefficient[7, component] / h**2
            )
            hess_u[row, component, 0, 2] = hess_u[row, component, 2, 0] = (
                coefficient[8, component] / h**2
            )
            hess_u[row, component, 1, 2] = hess_u[row, component, 2, 1] = (
                coefficient[9, component] / h**2
            )
    return {
        "grad_u": grad_u,
        "hess_u": hess_u,
        "grad_p": grad_p,
        "condition": condition,
        "rank": rank,
        "radius_m": radius,
    }


def _carreau_viscosity_and_derivative(
    shear_rate: np.ndarray, rheology: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.asarray(shear_rate, dtype=np.float64).clip(1e-12)
    mu_inf = float(rheology["mu_inf_pa_s"])
    mu_zero = float(rheology["mu_zero_pa_s"])
    time_constant = float(rheology["lambda_s"])
    exponent_a = float(rheology["a"])
    exponent_n = float(rheology["n"])
    base = 1.0 + np.power(time_constant * gamma, exponent_a)
    outer = (exponent_n - 1.0) / exponent_a
    viscosity = mu_inf + (mu_zero - mu_inf) * np.power(base, outer)
    derivative = (
        (mu_zero - mu_inf)
        * outer
        * np.power(base, outer - 1.0)
        * exponent_a
        * np.power(time_constant, exponent_a)
        * np.power(gamma, exponent_a - 1.0)
    )
    return viscosity, derivative


def _cfd_residual_terms(
    velocity: np.ndarray,
    du_dt: np.ndarray,
    derivatives: dict[str, np.ndarray],
    *,
    physics_config: dict[str, Any],
    condition_limit: float,
) -> dict[str, Any]:
    grad = derivatives["grad_u"]
    hess = derivatives["hess_u"]
    grad_p = derivatives["grad_p"]
    stable = (
        np.isfinite(grad).all(axis=(1, 2))
        & np.isfinite(hess).all(axis=(1, 2, 3))
        & np.isfinite(grad_p).all(axis=1)
        & (derivatives["rank"] == 10)
        & (derivatives["condition"] <= float(condition_limit))
    )
    density = float(physics_config["density_kg_m3"])
    strain = 0.5 * (grad + np.transpose(grad, (0, 2, 1)))
    shear_epsilon = float(physics_config["shear_rate_epsilon_s_inv"])
    shear_rate = np.sqrt(
        2.0 * np.sum(np.square(strain), axis=(1, 2)) + shear_epsilon**2
    )
    viscosity, dmu_dgamma = _carreau_viscosity_and_derivative(
        shear_rate, physics_config["rheology"]
    )
    dstrain = np.empty((len(velocity), 3, 3, 3), dtype=np.float64)
    for direction in range(3):
        dstrain[:, :, :, direction] = 0.5 * (
            hess[:, :, :, direction]
            + np.transpose(hess[:, :, :, direction], (0, 2, 1))
        )
    dgamma = 2.0 * np.einsum("nab,nabj->nj", strain, dstrain) / shear_rate[:, None]
    dmu = dmu_dgamma[:, None] * dgamma
    stress_divergence = np.zeros_like(velocity, dtype=np.float64)
    for component in range(3):
        for direction in range(3):
            symmetric_gradient = grad[:, component, direction] + grad[:, direction, component]
            symmetric_gradient_derivative = (
                hess[:, component, direction, direction]
                + hess[:, direction, component, direction]
            )
            stress_divergence[:, component] += (
                dmu[:, direction] * symmetric_gradient
                + viscosity * symmetric_gradient_derivative
            )
    convective = density * np.einsum("nj,nij->ni", velocity, grad)
    pressure = grad_p
    viscous = stress_divergence
    temporal = density * du_dt
    quasi = convective + pressure - viscous
    transient = temporal + quasi
    continuity = np.trace(grad, axis1=1, axis2=2)
    gradient_scale = np.sqrt(np.sum(np.square(grad), axis=(1, 2)))
    quasi_scale = np.sqrt(
        np.sum(np.square(convective), axis=1)
        + np.sum(np.square(pressure), axis=1)
        + np.sum(np.square(viscous), axis=1)
    )
    transient_scale = np.sqrt(
        np.square(quasi_scale) + np.sum(np.square(temporal), axis=1)
    )

    def rms_stable(value: np.ndarray) -> float:
        return _rms_numpy(value[stable]) if bool(np.any(stable)) else float("nan")

    metrics = {
        "points_total": int(len(stable)),
        "points_stable": int(np.sum(stable)),
        "stable_fraction": float(np.mean(stable)),
        "condition_number_median": float(np.median(derivatives["condition"])),
        "condition_number_p95": float(np.quantile(derivatives["condition"], 0.95)),
        "fit_radius_median_m": float(np.median(derivatives["radius_m"])),
        "fit_radius_p95_m": float(np.quantile(derivatives["radius_m"], 0.95)),
        "continuity_rms_s_inv": rms_stable(continuity),
        "continuity_relative_to_gradient_rms": (
            rms_stable(continuity) / max(rms_stable(gradient_scale), 1e-30)
        ),
        "convective_rms_pa_per_m": rms_stable(convective),
        "pressure_gradient_rms_pa_per_m": rms_stable(pressure),
        "viscous_rms_pa_per_m": rms_stable(viscous),
        "temporal_rms_pa_per_m": rms_stable(temporal),
        "quasi_steady_residual_rms_pa_per_m": rms_stable(quasi),
        "transient_residual_rms_pa_per_m": rms_stable(transient),
        "quasi_steady_relative_residual": (
            rms_stable(quasi) / max(rms_stable(quasi_scale), 1e-30)
        ),
        "transient_relative_residual": (
            rms_stable(transient) / max(rms_stable(transient_scale), 1e-30)
        ),
        "transient_over_quasi_residual_ratio": (
            rms_stable(transient) / max(rms_stable(quasi), 1e-30)
        ),
        "shear_rate_rms_s_inv": rms_stable(shear_rate),
        "viscosity_mean_pa_s": (
            float(np.mean(viscosity[stable])) if bool(np.any(stable)) else float("nan")
        ),
    }
    return {
        "metrics": metrics,
        "stable": stable,
        "quasi_residual": quasi,
        "transient_residual": transient,
        "continuity": continuity,
    }


def _manifest_rows(config: VolumeExperimentConfig) -> dict[str, dict[str, Any]]:
    aggregate = json.loads(
        Path(config["paths"]["sidecar_manifest"]).read_text(encoding="utf-8")
    )
    return {str(row["canonical_id"]): row for row in aggregate["cases"]}


def audit_cfd_truth(
    config: VolumeExperimentConfig,
    *,
    case_ids: tuple[str, ...],
    sample_points: int,
    neighbors: tuple[int, ...],
    condition_limit: float,
) -> dict[str, Any]:
    rows = _manifest_rows(config)
    reports: list[dict[str, Any]] = []
    for case_index, case_id in enumerate(case_ids, start=1):
        if case_id not in rows:
            raise KeyError(f"CFD audit case not found in sidecar manifest: {case_id}")
        case_manifest_path = Path(rows[case_id]["manifest"])
        manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
        bundle_path = Path(manifest["parents"]["bundle"]["path"])
        raw_peak_path = Path(manifest["parents"]["raw_peak"]["path"])
        case_dir = raw_peak_path.parent.parent
        with np.load(bundle_path, allow_pickle=False) as source:
            bundle = {name: np.asarray(source[name]) for name in source.files}
        peak_step = int(manifest["peak_step"])
        available_steps = list_steps(case_dir, "ascii_in")
        previous_step = max(step for step in available_steps if step < peak_step)
        next_step = min(step for step in available_steps if step > peak_step)
        solver_dt_s, journals = _solver_time_step_s(case_dir)
        central_dt_s = float(next_step - previous_step) * solver_dt_s

        peak_raw = read_interior(step_file(case_dir, peak_step, "ascii_in"))
        previous_raw = read_interior(step_file(case_dir, previous_step, "ascii_in"))
        next_raw = read_interior(step_file(case_dir, next_step, "ascii_in"))
        peak_indices, peak_delta, peak_mode = align_raw_interior_to_bundle(
            peak_raw, bundle
        )
        previous_indices, previous_delta, previous_mode = align_raw_interior_to_bundle(
            previous_raw, bundle
        )
        next_indices, next_delta, next_mode = align_raw_interior_to_bundle(
            next_raw, bundle
        )
        rotation = bundle["transform_rotation"]
        peak_velocity = rotate_velocity_to_registered(
            peak_raw["velocity"][peak_indices], rotation
        ).astype(np.float64)
        previous_velocity = rotate_velocity_to_registered(
            previous_raw["velocity"][previous_indices], rotation
        ).astype(np.float64)
        next_velocity = rotate_velocity_to_registered(
            next_raw["velocity"][next_indices], rotation
        ).astype(np.float64)
        du_dt = (next_velocity - previous_velocity) / central_dt_s
        peak_pressure = np.asarray(peak_raw["pressure"][peak_indices], dtype=np.float64)
        coords_normalized = np.asarray(bundle["int_coords_norm"], dtype=np.float64)
        coords_m = coords_normalized * float(manifest["scales"]["length_m"])
        strict = np.asarray(bundle["int_dist_to_wall"], dtype=np.float64) > 1e-6
        core = np.asarray(bundle["int_type"], dtype=np.int64) == 0
        strict_global = np.flatnonzero(strict)
        candidates = np.flatnonzero(strict & core)
        if len(candidates) < int(sample_points):
            candidates = strict_global
        rng_seed = int.from_bytes(
            hashlib.sha256((case_id + "/cfd-oracle").encode("utf-8")).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(rng_seed)
        query_global = np.sort(
            rng.choice(candidates, size=min(int(sample_points), len(candidates)), replace=False)
        )
        strict_coords_m = coords_m[strict_global]
        strict_fields = np.column_stack(
            [peak_velocity[strict_global], peak_pressure[strict_global]]
        )
        query_fields = np.column_stack(
            [peak_velocity[query_global], peak_pressure[query_global]]
        )
        tree = cKDTree(strict_coords_m)
        neighborhood_reports: dict[str, Any] = {}
        raw_results: dict[int, dict[str, Any]] = {}
        for neighbor_count in neighbors:
            derivatives = _quadratic_derivatives(
                strict_coords_m,
                strict_fields,
                coords_m[query_global],
                query_fields,
                tree,
                neighbors=int(neighbor_count),
            )
            residual = _cfd_residual_terms(
                peak_velocity[query_global],
                du_dt[query_global],
                derivatives,
                physics_config=config["physics"],
                condition_limit=float(condition_limit),
            )
            raw_results[int(neighbor_count)] = residual
            neighborhood_reports[str(neighbor_count)] = residual["metrics"]

        sensitivity: dict[str, Any] = {}
        if len(neighbors) >= 2:
            first, second = int(neighbors[0]), int(neighbors[-1])
            a, b = raw_results[first], raw_results[second]
            stable = a["stable"] & b["stable"]
            sensitivity = {
                "comparison": f"k{first}_versus_k{second}",
                "stable_intersection_points": int(np.sum(stable)),
                "quasi_residual_relative_l2_difference": (
                    _rms_numpy(a["quasi_residual"][stable] - b["quasi_residual"][stable])
                    / max(_rms_numpy(b["quasi_residual"][stable]), 1e-30)
                    if bool(np.any(stable))
                    else float("nan")
                ),
                "transient_residual_relative_l2_difference": (
                    _rms_numpy(
                        a["transient_residual"][stable]
                        - b["transient_residual"][stable]
                    )
                    / max(_rms_numpy(b["transient_residual"][stable]), 1e-30)
                    if bool(np.any(stable))
                    else float("nan")
                ),
                "continuity_relative_l2_difference": (
                    _rms_numpy(a["continuity"][stable] - b["continuity"][stable])
                    / max(_rms_numpy(b["continuity"][stable]), 1e-30)
                    if bool(np.any(stable))
                    else float("nan")
                ),
            }
        reports.append(
            {
                "case_id": case_id,
                "peak_step": peak_step,
                "previous_step": previous_step,
                "next_step": next_step,
                "solver_time_step_s": solver_dt_s,
                "central_difference_interval_s": central_dt_s,
                "journal_paths": journals,
                "query_scope": "strict core volume where available",
                "query_points": int(len(query_global)),
                "strict_volume_points": int(len(strict_global)),
                "alignment": {
                    "peak": {
                        "mode": peak_mode,
                        "max_coordinate_delta_normalized": float(np.max(peak_delta)),
                    },
                    "previous": {
                        "mode": previous_mode,
                        "max_coordinate_delta_normalized": float(np.max(previous_delta)),
                    },
                    "next": {
                        "mode": next_mode,
                        "max_coordinate_delta_normalized": float(np.max(next_delta)),
                    },
                },
                "neighborhoods": neighborhood_reports,
                "sensitivity": sensitivity,
            }
        )
        print(
            json.dumps(
                {
                    "event": "cfd_truth_audited",
                    "index": case_index,
                    "total": len(case_ids),
                    "case_id": case_id,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    aggregate: dict[str, Any] = {}
    for neighbor_count in neighbors:
        key = str(int(neighbor_count))
        metric_names = list(reports[0]["neighborhoods"][key])
        aggregate[key] = {
            metric: {
                "mean": _finite_mean(
                    [float(row["neighborhoods"][key][metric]) for row in reports]
                ),
                "median": _finite_median(
                    [float(row["neighborhoods"][key][metric]) for row in reports]
                ),
            }
            for metric in metric_names
            if isinstance(reports[0]["neighborhoods"][key][metric], (int, float))
        }
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "purpose": "CFD truth residual under quasi-steady and transient route equations",
        "training_performed": False,
        "equation": "incompressible Carreau-Yasuda with full div(2 mu D)",
        "derivative_method": (
            "weighted local quadratic least squares in physical coordinates; "
            "reported for two k values with conditioning and sensitivity diagnostics"
        ),
        "condition_number_limit": float(condition_limit),
        "important_limit": (
            "This is a numerical derivative oracle on exported cell centers, not the "
            "native finite-volume residual used by Fluent. Large k-sensitivity makes "
            "the PDE-compatibility verdict inconclusive."
        ),
        "aggregate": aggregate,
        "cases": reports,
    }


def audit_paper_metric_bridge(config: VolumeExperimentConfig) -> dict[str, Any]:
    """Re-express the frozen V2 evaluation in the teacher-paper NMAE style."""
    dataset = VolumeFieldDataset(
        config["paths"]["sidecar_manifest"],
        config["paths"]["field_stats"],
        roles=config["data"]["eval_roles"],
        input_variant=config.input_variant,
        sampling=config["sampling"],
        seed=int(config["train"]["seed"]),
    )
    by_case = {case.case_id: case for case in dataset.cases}
    run_dir = Path(config["paths"]["run_dir"])
    protocols = {
        "same5k": run_dir / "evaluation_same5k_last.json",
        "full_volume": run_dir / "evaluation_fullvolume_last.json",
    }
    reports: dict[str, Any] = {}
    for protocol, evaluation_path in protocols.items():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        case_indices: dict[str, np.ndarray] = {}
        component_min: list[np.ndarray] = []
        component_max: list[np.ndarray] = []
        for case in dataset.cases:
            if protocol == "same5k":
                rng = np.random.default_rng(
                    _case_seed(case.case_id, int(config["sampling"]["seed"]))
                )
                indices = _take_strict_volume(
                    rng,
                    case.is_wall,
                    int(config["sampling"]["support_points"]),
                )
            else:
                indices = np.flatnonzero(~np.asarray(case.is_wall, dtype=bool))
            case_indices[case.case_id] = indices
            velocity = np.asarray(case.velocity[indices], dtype=np.float64)
            component_min.append(np.min(velocity, axis=0))
            component_max.append(np.max(velocity, axis=0))
        component_range = np.max(component_max, axis=0) - np.min(component_min, axis=0)
        case_rows = []
        for row in evaluation["cases"]:
            case = by_case[row["case_id"]]
            indices = case_indices[case.case_id]
            component_nmae = [
                float(row["metrics"][name]["mae"]) / max(float(component_range[j]), 1e-30)
                for j, name in enumerate(("u", "v", "w"))
            ]
            speed = np.linalg.norm(
                np.asarray(case.velocity[indices], dtype=np.float64), axis=1
            )
            speed_range = float(np.max(speed) - np.min(speed))
            case_rows.append(
                {
                    "case_id": case.case_id,
                    "component_global_range_nmae": {
                        "u": component_nmae[0],
                        "v": component_nmae[1],
                        "w": component_nmae[2],
                        "velocity_mean": float(np.mean(component_nmae)),
                    },
                    "speed_case_range_nmae_approximation": (
                        float(row["metrics"]["speed"]["mae"])
                        / max(speed_range, 1e-30)
                    ),
                }
            )
        velocity_nmae = [
            item["component_global_range_nmae"]["velocity_mean"]
            for item in case_rows
        ]
        speed_nmae = [
            item["speed_case_range_nmae_approximation"] for item in case_rows
        ]
        reports[protocol] = {
            "evaluation_path": str(evaluation_path.resolve()),
            "component_global_ranges_m_s": component_range.tolist(),
            "component_global_range_velocity_nmae": {
                "mean": float(np.mean(velocity_nmae)),
                "std_population": float(np.std(velocity_nmae)),
                "median": float(np.median(velocity_nmae)),
            },
            "speed_case_range_nmae_approximation": {
                "mean": float(np.mean(speed_nmae)),
                "std_population": float(np.std(speed_nmae)),
                "median": float(np.median(speed_nmae)),
            },
            "cases": case_rows,
        }
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "training_performed": False,
        "purpose": "metric bridge between V2 R2/MAE and teacher-paper NMAE",
        "interpretation_limit": (
            "The component-global-range metric follows the available teacher-code "
            "snapshot. Low NMAE can coexist with low pointwise R2 because a wide "
            "global range makes the denominator large. Cross-dataset values are not "
            "a controlled accuracy comparison."
        ),
        "protocols": reports,
    }


def _summary(
    network: dict[str, Any] | None,
    cfd: dict[str, Any] | None,
    metric_bridge: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "training_performed": False,
    }
    if network is not None:
        ratios = network["aggregate"]["condition_over_exact_case_ratio"]
        payload["network_query_test"] = {
            "independent_over_exact": ratios["independent_volume"],
            "jitter_over_exact": {
                key: value for key, value in ratios.items() if key.startswith("jittered")
            },
        }
    if cfd is not None:
        payload["cfd_truth_test"] = {
            "aggregate": cfd["aggregate"],
            "case_sensitivity": {
                row["case_id"]: row["sensitivity"] for row in cfd["cases"]
            },
        }
    if metric_bridge is not None:
        payload["paper_metric_bridge"] = {
            protocol: row["component_global_range_velocity_nmae"]
            for protocol, row in metric_bridge["protocols"].items()
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_RUN / "resolved_config.json"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_RUN / "checkpoints/last.pt"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--section", choices=("all", "network", "cfd", "metrics"), default="all"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--network-physics-points", type=int, default=512)
    parser.add_argument(
        "--jitter-amplitudes", type=float, nargs="+", default=(1e-4, 1e-3)
    )
    parser.add_argument("--cfd-cases", nargs="+", default=DEFAULT_CFD_CASES)
    parser.add_argument("--cfd-points", type=int, default=256)
    parser.add_argument("--neighbors", type=int, nargs="+", default=(48, 96))
    parser.add_argument("--condition-limit", type=float, default=1e6)
    args = parser.parse_args()

    config = VolumeExperimentConfig.from_json(args.config)
    output_dir = args.output_dir.resolve()
    network_report = None
    cfd_report = None
    metric_bridge = None
    if args.section in {"all", "network"}:
        network_report = audit_network_queries(
            config,
            args.checkpoint.resolve(),
            device_name=str(args.device),
            physics_points=int(args.network_physics_points),
            jitter_amplitudes=tuple(float(value) for value in args.jitter_amplitudes),
        )
        atomic_write_json(output_dir / "network_query_residuals.json", network_report)
    if args.section in {"all", "cfd"}:
        cfd_report = audit_cfd_truth(
            config,
            case_ids=tuple(str(value) for value in args.cfd_cases),
            sample_points=int(args.cfd_points),
            neighbors=tuple(int(value) for value in args.neighbors),
            condition_limit=float(args.condition_limit),
        )
        atomic_write_json(output_dir / "cfd_truth_residuals.json", cfd_report)
    if args.section in {"all", "metrics"}:
        metric_bridge = audit_paper_metric_bridge(config)
        atomic_write_json(output_dir / "paper_metric_bridge.json", metric_bridge)
    if network_report is None:
        existing = output_dir / "network_query_residuals.json"
        if existing.exists():
            network_report = json.loads(existing.read_text(encoding="utf-8"))
    if cfd_report is None:
        existing = output_dir / "cfd_truth_residuals.json"
        if existing.exists():
            cfd_report = json.loads(existing.read_text(encoding="utf-8"))
    if metric_bridge is None:
        existing = output_dir / "paper_metric_bridge.json"
        if existing.exists():
            metric_bridge = json.loads(existing.read_text(encoding="utf-8"))
    atomic_write_json(
        output_dir / "summary.json",
        _summary(network_report, cfd_report, metric_bridge),
    )
    print(
        json.dumps(
            {"event": "v2_physics_diagnosis_completed", "output_dir": str(output_dir)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
