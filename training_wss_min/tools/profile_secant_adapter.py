"""Array-only inference with the frozen Profile-Secant V3 velocity→WSS method.

Coordinates are physical millimetres; velocity vectors are physical m/s in the
same frame. Normals point into the lumen. All four gradient stages receive the
same explicit velocity array. No CFD files, WSS labels, or training routines
are read by this adapter. Calibration must receive an entire case, since the
frozen features include within-case ranks and quantiles.

The float32 cache boundaries intentionally reproduce the historical two-stage
cache builder. Original calculator sources and serialized models are untouched.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "wss_mri_calculator/src"
EXP = ROOT / "wss_mri_calculator/experiments"
V1_CONFIG = EXP / "pointcloud_adaptive_v1/config_frozen_v1.json"
V3_CONFIG = EXP / "pointcloud_normal_multiscale_v3/config_frozen_v3.json"
V4_CONFIG = EXP / "pointcloud_surface_mls_v4/config_test35_cache.json"
MODEL = EXP / "pointcloud_surface_mls_v4/calibrator_profile_secant_high_tail_anchor10_v3.joblib"
MODEL_MANIFEST = MODEL.with_name(MODEL.stem + "_manifest.json")
MODEL_SHA256 = "c1e53af5d60e17620c4e5123da76bab9f135c4125decf6a282b944f492433a02"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_sources() -> dict[str, Any]:
    """Return immutable dependency identities and the effective parameter sets."""
    names = (
        "run_frozen_wss_compat.py", "calculate_wss_cfd.py", "data_loader_cfd.py",
        "wss_multiscale.py", "wss_normal_multiscale_v3.py", "wss_surface_mls_v4.py",
        "batch_validate_cfd.py", "calibrate_v4_oof.py", "calibrate_v4_high_tail_oof.py",
        "v4_profile_features.py", "apply_v4_high_tail_calibrator.py",
        "validate_v4_high_tail_holdout.py",
    )
    paths = [Path(__file__).resolve(), *[SRC / name for name in names],
             V1_CONFIG, V3_CONFIG, V4_CONFIG, MODEL, MODEL_MANIFEST]
    records = [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
               for path in paths]
    model_sha = next(row["sha256"] for row in records if row["path"] == str(MODEL))
    if model_sha != MODEL_SHA256:
        raise RuntimeError("frozen Profile-Secant V3 model SHA256 mismatch")
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if manifest["model_sha256"] != MODEL_SHA256 or len(manifest["feature_names"]) != 84:
        raise RuntimeError("frozen Profile-Secant V3 manifest mismatch")
    return {
        "method": "surface_mls_v4_profile_secant_high_tail_anchor10",
        "files": records,
        "model_sha256": model_sha,
        "v1": json.loads(V1_CONFIG.read_text(encoding="utf-8")),
        "v3_method": json.loads(V3_CONFIG.read_text(encoding="utf-8"))["method"],
        "v4_method": json.loads(V4_CONFIG.read_text(encoding="utf-8"))["method"],
        "depth3_overrides": {"correction_max": 1.5, "parallel_transport": True,
                             "group_balance_power": 0.5, "depth_degree": 3},
        "units": {"coordinates": "mm", "velocity": "m/s", "gradient": "1/s", "wss": "Pa"},
        "truth_used_at_inference": False,
        "calibration_scope": "complete single case; no wall chunk calibration",
    }


@lru_cache(maxsize=1)
def _runtime() -> dict[str, Any]:
    # Restore only the historical NumPy compatibility API, in memory. The
    # functions themselves come from the existing UDF-audited wrapper.
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    compat = importlib.import_module("run_frozen_wss_compat")
    rheology = importlib.import_module("wss_pinn.physics.rheology")
    rheology.carreau_yasuda_numpy = compat.carreau_yasuda_numpy
    rheology.UDF_PARAMETERS = compat.UDF_PARAMETERS
    module_name = "wss_pinn.physics.wall_shear"
    try:
        wall_shear = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        wall_shear = types.ModuleType(module_name)
        sys.modules[module_name] = wall_shear
    wall_shear.stl_face_geometry = compat.stl_face_geometry
    modules = {name: importlib.import_module(name) for name in (
        "calculate_wss_cfd", "wss_normal_multiscale_v3", "wss_surface_mls_v4",
        "calibrate_v4_oof", "calibrate_v4_high_tail_oof", "v4_profile_features",
    )}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != (SRC / f"{name}.py").resolve():
            raise RuntimeError(f"unexpected module shadowing: {name}")
    return modules


@lru_cache(maxsize=1)
def _package() -> dict[str, Any]:
    import joblib
    if _sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("frozen Profile-Secant V3 model SHA256 mismatch")
    package = joblib.load(MODEL)
    if package.get("schema_version") != 2 or package.get("feature_variant") != "depth3_profile":
        raise RuntimeError("unexpected frozen calibrator schema or feature variant")
    return package


def _vectors(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite (N, 3) array")
    return array


def build_cache(
    canonical_id: str,
    wall_mm: np.ndarray,
    full_normals: np.ndarray,
    interior_mm: np.ndarray,
    velocity_raw_m_s: np.ndarray,
    local_radius_mm: np.ndarray,
    sample_index: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute all frozen gradient/diagnostic stages from explicit input arrays.

    ``wall_mm`` and ``full_normals`` describe the complete original wall.
    ``sample_index`` optionally selects target nodes, preserving their order.
    Radius may have full-wall length or selected-target length. Invalid fitted
    gradients remain NaN and are exposed by ``valid_mask`` during calibration.
    """
    from scipy.spatial import cKDTree
    modules = _runtime()
    physics = modules["calculate_wss_cfd"]
    v3_module = modules["wss_normal_multiscale_v3"]
    surface = modules["wss_surface_mls_v4"]
    wall = _vectors("wall_mm", wall_mm)
    normals = _vectors("full_normals", full_normals)
    interior = _vectors("interior_mm", interior_mm)
    velocity = _vectors("velocity_raw_m_s", velocity_raw_m_s)
    if normals.shape != wall.shape or velocity.shape != interior.shape:
        raise ValueError("wall/normal or interior/velocity shape mismatch")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("full_normals must be unit inward normals")
    if sample_index is None:
        indices = np.arange(len(wall), dtype=np.int64)
    else:
        original = np.asarray(sample_index)
        if original.ndim != 1 or original.dtype.kind not in "iu":
            raise ValueError("sample_index must be a one-dimensional integer array")
        indices = original.astype(np.int64)
        if (not len(indices) or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(wall))):
            raise ValueError("sample_index must be nonempty, unique, and in bounds")
    radius = np.asarray(local_radius_mm, dtype=np.float64)
    if radius.shape == (len(wall),):
        radius = radius[indices]
    if radius.shape != (len(indices),) or not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError("local_radius_mm must contain positive finite radii for every target")
    cohort = str(canonical_id).split("/", 1)[0]
    if cohort not in {"AG", "AAA", "ILO"}:
        raise ValueError("canonical_id must belong to the frozen AG/AAA/ILO feature schema")
    target, target_normals = wall[indices], normals[indices]
    tree = cKDTree(interior)
    v1 = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    v3 = v3_module.NormalMultiscaleV3Config.from_mapping(
        json.loads(V3_CONFIG.read_text(encoding="utf-8"))["method"])
    mapping = json.loads(V4_CONFIG.read_text(encoding="utf-8"))["method"]
    mapping["depth_fractions"] = tuple(mapping["depth_fractions"])
    v4 = surface.SurfaceMLSV4Config(**mapping)
    gradient_v1, used_v1, diag_v1 = physics.fit_wall_gradient(
        target, target_normals, interior, velocity, tree,
        degree=int(v1["degree"]), neighbor_mode="adaptive_cv",
        adaptive_neighbors=tuple(int(k) for k in v1["adaptive_neighbors"]),
        cv_tolerance=float(v1["cv_tolerance"]))
    gradient_v3, _, diag_v3 = v3_module.fit_wall_gradient_normal_multiscale(
        target, target_normals, wall, normals, interior, velocity, tree,
        local_radius_mm=radius, config=v3, fallback_gradient=gradient_v1)
    _, diag_surface = surface.fit_wall_gradient_surface_mls(
        target, target_normals, wall, normals, interior, velocity, tree,
        local_radius_mm=radius, fallback_gradient=gradient_v1, config=v4)
    v3_correction = np.linalg.norm(gradient_v3, axis=1) / np.maximum(
        np.linalg.norm(gradient_v1, axis=1), 1e-12)
    correction = np.maximum(v3_correction, diag_surface["correction"])
    gradient_v4 = gradient_v1 * correction[:, None]
    arrays = {
        "canonical_id": np.asarray(str(canonical_id)), "cohort": np.asarray(cohort),
        "sample_index": indices, "wall_mm": target.astype(np.float32),
        "normal": target_normals.astype(np.float32),
        "local_radius_mm": radius.astype(np.float32),
        "gradient_v1": gradient_v1.astype(np.float32),
        "gradient_v3": gradient_v3.astype(np.float32),
        "gradient_v4": gradient_v4.astype(np.float32),
        "v3_correction": v3_correction.astype(np.float32),
        "combined_correction": correction.astype(np.float32),
        "v1__used": np.asarray(used_v1),
    }
    for prefix, diagnostics in (("v1", diag_v1), ("v3", diag_v3), ("surface", diag_surface)):
        arrays.update({f"{prefix}__{key}": np.asarray(value) for key, value in diagnostics.items()})

    # The historical depth3 CLI reads the float32 base cache back as float64.
    # Reproduce this boundary before calling the same kernel with the SAME
    # supplied velocities; never reload CFD velocities behind the caller.
    depth3 = surface.SurfaceMLSV4Config(
        correction_max=1.5, parallel_transport=True,
        group_balance_power=0.5, depth_degree=3)
    cached_v1 = arrays["gradient_v1"].astype(np.float64)
    _, diag_depth3 = surface.fit_wall_gradient_surface_mls(
        arrays["wall_mm"].astype(np.float64), arrays["normal"].astype(np.float64),
        wall, normals, interior, velocity, tree,
        local_radius_mm=arrays["local_radius_mm"].astype(np.float64),
        fallback_gradient=cached_v1, config=depth3)
    base_correction = arrays["combined_correction"].astype(np.float64)
    fusion_correction = np.maximum(base_correction, diag_depth3["correction"])
    arrays["gradient_v4_base"] = arrays["gradient_v4"].copy()
    arrays["combined_correction_base"] = arrays["combined_correction"].copy()
    arrays["gradient_v4"] = (cached_v1 * fusion_correction[:, None]).astype(np.float32)
    arrays["combined_correction"] = fusion_correction.astype(np.float32)
    arrays["depth3_fusion_applied"] = (fusion_correction > base_correction + 1e-12).astype(np.int8)
    arrays.update({f"depth3__{key}": np.asarray(value) for key, value in diag_depth3.items()})
    return arrays


def calibrate_case(cache_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Apply the frozen 84-feature calibrator to one complete case without truth.

    ``truth*`` fields, if supplied by an evaluator, are discarded before feature
    construction. Outputs retain NaNs at invalid physical predictions instead
    of selecting only convenient nodes; callers must report their coverage.
    """
    modules = _runtime()
    package = _package()
    data = {key: np.asarray(value) for key, value in cache_arrays.items()
            if not key.startswith("truth")}
    canonical_id = str(data.pop("canonical_id"))
    cohort = str(data.pop("cohort"))
    if cohort not in {"AG", "AAA", "ILO"} or canonical_id.split("/", 1)[0] != cohort:
        raise ValueError("invalid single-case identity/cohort")
    n = len(data["gradient_v1"])
    if n == 0:
        raise ValueError("cannot calibrate an empty case")
    # Caller-supplied point_case/group arrays cannot turn this into mixed-case
    # calibration. Complete-case identity is constructed from one scalar ID.
    data["point_case"] = np.repeat(np.asarray(canonical_id), n)
    data["point_cohort"] = np.repeat(np.asarray(cohort), n)
    profile = modules["v4_profile_features"]
    x, names = profile.build_profile_features(data)
    if len(names) != 84 or names != list(package["feature_names"]):
        raise RuntimeError("frozen 84-feature calibrator contract mismatch")
    prediction_data = profile.base_v4_view(data)
    physics_vector = modules["calculate_wss_cfd"].wss_from_gradient(
        np.asarray(prediction_data["gradient_v4"], dtype=np.float64), "carreau")
    physics_mag = np.linalg.norm(physics_vector, axis=1)
    current_raw = np.clip(np.exp(package["current_model"].predict(x)), *package["current_raw_ratio_clip"])
    tail_raw = np.clip(np.exp(package["tail_model"].predict(x)), *package["tail_raw_ratio_clip"])
    groups, case_x = modules["calibrate_v4_oof"].build_case_scale_features(x, names, data["point_case"])
    if len(groups) != 1 or groups[0] != canonical_id:
        raise RuntimeError("calibration requires exactly one complete case")
    def case_alpha(model_key: str, clip_key: str) -> float:
        return float(np.clip(np.exp(package[model_key].predict(case_x)), *package[clip_key])[0])
    overall_alpha = case_alpha("overall_case_model", "case_alpha_clip")
    tail_alpha = case_alpha("tail_case_model", "tail_alpha_clip")
    peak_alpha = case_alpha("peak_case_model", "peak_alpha_clip")
    ratios = modules["calibrate_v4_high_tail_oof"]
    current_ratio = ratios.current_safe_ratio(current_raw, physics_mag, overall_alpha)
    final_ratio = ratios.tail_ratio(
        current_ratio, tail_raw, physics_mag, x[:, names.index("case_pred_rank")],
        tail_alpha, peak_alpha, gate_start=float(package["tail_gate_start"]),
        use_peak_anchor=float(package.get("peak_anchor_strength", package.get("use_peak_anchor", False))))
    final_vector = physics_vector * final_ratio[:, None]
    current_vector = physics_vector * current_ratio[:, None]
    # Match the frozen apply script's scalar calculation exactly (rather than
    # introducing an additional norm roundoff after scaling the vector).
    final_mag = physics_mag * final_ratio
    return {
        "wss_vec_pa": final_vector, "wss_mag_pa": final_mag,
        "physics_wss_vec_pa": physics_vector, "physics_wss_mag_pa": physics_mag,
        "current_wss_vec_pa": current_vector, "current_wss_mag_pa": physics_mag * current_ratio,
        "high_tail_ratio": final_ratio, "current_ratio": current_ratio,
        "valid_mask": np.isfinite(final_vector).all(axis=1) & np.isfinite(final_mag),
        "case_scales": {"overall": overall_alpha, "tail": tail_alpha, "peak": peak_alpha},
        "feature_count": len(names), "model_sha256": MODEL_SHA256,
    }


def validate_adapter_contracts() -> dict[str, Any]:
    """Low-cost replay against the original CLI plus physical/chunk contracts.

    The historical case is chosen by compressed cache size, without viewing
    metrics. CLI outputs are confined to a temporary directory. No training,
    frozen artifact modifications, or output-based parameter selection occurs.
    """
    import os
    import subprocess
    import tempfile
    import time
    started = time.monotonic()
    historical = ROOT / (
        "outputs/wss_mri_calculator/pointcloud_surface_mls_v4/"
        "profile_secant_v3_fullwall/inference/point_cache_test35_profile_secant_fullwall")
    path = min(historical.glob("*.npz"), key=lambda item: item.stat().st_size)
    with np.load(path, allow_pickle=False) as source:
        cache = {key: source[key] for key in source.files}
    result = calibrate_case(cache)
    with tempfile.TemporaryDirectory(prefix="profile_secant_adapter_") as temporary:
        temp = Path(temporary)
        (temp / "cache").mkdir()
        (temp / "cache" / path.name).symlink_to(path)
        command = [sys.executable, str(SRC / "run_frozen_wss_compat.py"),
                   str(SRC / "apply_v4_high_tail_calibrator.py"),
                   "--cache-dir", str(temp / "cache"), "--model", str(MODEL),
                   "--json-out", str(temp / "result.json"),
                   "--predictions-out", str(temp / "predictions.npz")]
        environment = os.environ.copy()
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[variable] = "1"
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(f"frozen CLI verification failed: {completed.stderr}")
        with np.load(temp / "predictions.npz", allow_pickle=False) as source:
            exact = {key: bool(np.array_equal(result[key].astype(np.float32), source[expected]))
                     for key, expected in (
                         ("wss_vec_pa", "high_tail_wss_vec"),
                         ("current_wss_vec_pa", "current_wss_vec"),
                         ("physics_wss_vec_pa", "base_wss_vec"),
                         ("high_tail_ratio", "high_tail_ratio"))}
        if not all(exact.values()):
            raise AssertionError(f"adapter differs from original frozen apply: {exact}")
    disturbed = dict(cache)
    disturbed["truth_mag"] = np.full_like(cache["truth_mag"], np.nan)
    disturbed["truth_vec"] = np.full_like(cache["truth_vec"], 1e12)
    for candidate in (disturbed, {k: v for k, v in cache.items() if not k.startswith("truth")}):
        comparison = calibrate_case(candidate)
        for key in ("wss_vec_pa", "wss_mag_pa"):
            if not np.array_equal(result[key], comparison[key]):
                raise AssertionError("WSS output depends on truth fields")

    # Analytic planar shear: u_x = 120 * z_m, so du_x/dz_m = 120 /s.
    grid = np.linspace(-1.0, 1.0, 9)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    wall = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    normals = np.tile([0.0, 0.0, 1.0], (len(wall), 1))
    depths = np.array([0.12, 0.25, 0.45, 0.70, 1.0])
    interior = np.concatenate([wall + depth * normals for depth in depths])
    velocity = np.column_stack([np.repeat(depths, len(wall)) * 1e-3 * 120.0,
                                np.zeros(len(interior)), np.zeros(len(interior))])
    index = np.array([39, 40, 41])
    radius = np.full(len(wall), 5.0)
    rotation = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
    base = build_cache("AG/synthetic/planar", wall, normals, interior, velocity, radius, index)
    rotated = build_cache("AG/synthetic/planar", wall @ rotation.T, normals @ rotation.T,
                          interior @ rotation.T, velocity @ rotation.T, radius, index)
    scaled = build_cache("AG/synthetic/planar", wall * 2, normals, interior * 2,
                         velocity * 2, radius * 2, index)
    for key in ("gradient_v1", "gradient_v3", "gradient_v4_base"):
        if not np.allclose(base[key], np.tile([120.0, 0.0, 0.0], (3, 1)), atol=0.002):
            raise AssertionError(f"millimetre/metre gradient contract failed: {key}")
        if not np.allclose(rotated[key], base[key] @ rotation.T, atol=0.002):
            raise AssertionError(f"rotation contract failed: {key}")
        if not np.allclose(scaled[key], base[key], atol=0.002):
            raise AssertionError(f"equal coordinate/velocity scaling failed: {key}")
    chunks = [build_cache("AG/synthetic/planar", wall, normals, interior, velocity,
                          radius, item) for item in (index[:1], index[1:])]
    compared = 0
    for key, value in base.items():
        if value.ndim == 0:
            continue
        merged = np.concatenate([chunk[key] for chunk in chunks], axis=0)
        if not np.array_equal(value, merged, equal_nan=True):
            raise AssertionError(f"wall target chunking changes a stage/diagnostic: {key}")
        compared += 1
    return {
        "status": "passed", "model_sha256": MODEL_SHA256,
        "historical_complete_cache": str(path), "historical_cache_sha256": _sha256(path),
        "canonical_id": str(cache["canonical_id"]), "n_wall": len(result["wss_mag_pa"]),
        "frozen_cli_float32_exact": exact,
        "truth_disturbed_and_removed_double_outputs_exact": True,
        "analytic_gradient_s_inv": 120.0, "analytic_tolerance_s_inv": 0.002,
        "rotation_and_equal_coordinate_velocity_scale_contract": True,
        "chunked_vs_unsplit_arrays_exact": compared,
        "n_invalid_historical_predictions": int((~result["valid_mask"]).sum()),
        "elapsed_seconds": time.monotonic() - started,
    }
