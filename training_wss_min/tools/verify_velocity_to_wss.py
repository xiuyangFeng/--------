"""Independently audit VELWSS1 arrays, metrics, identities and source hashes.

This script intentionally imports neither the production evaluator/dataset nor
the velocity-to-WSS adapter. Metric formulas are implemented with NumPy here.
Run through Slurm: the full audit reads original and derived archives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP = ROOT / "training_wss_min/experiments/v5_velocity_to_wss_20260909"
DATA = ROOT / "data_wss_v5/views/wss_min_view_v1"
MODEL_SHA = "c1e53af5d60e17620c4e5123da76bab9f135c4125decf6a282b944f492433a02"
ATOL, RTOL = 2e-10, 2e-10


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class Audit:
    def __init__(self):
        self.checks = []
        self.hashes = {}

    def require(self, name, condition, **evidence):
        self.checks.append(dict(check=name, passed=bool(condition), **clean(evidence)))

    def equal(self, name, actual, expected):
        self.require(name, actual == expected, actual=actual, expected=expected)

    def number(self, name, actual, expected, *, atol=ATOL, rtol=RTOL):
        if expected is None:
            passed = not np.isfinite(actual)
            difference = None
        else:
            difference = abs(float(actual) - float(expected))
            passed = np.isfinite(actual) and difference <= atol + rtol * abs(expected)
        self.require(name, passed, actual=actual, expected=expected,
                     abs_difference=difference, absolute_tolerance=atol,
                     relative_tolerance=rtol)

    def array(self, name, actual, expected, *, atol=0.0, rtol=0.0):
        same_shape = actual.shape == expected.shape
        passed = same_shape and np.allclose(actual, expected, atol=atol, rtol=rtol,
                                           equal_nan=False)
        self.require(name, passed, shape_actual=list(actual.shape),
                     shape_expected=list(expected.shape),
                     max_abs_difference=float(np.max(np.abs(actual - expected)))
                     if same_shape and actual.size and actual.dtype.kind != "b" else None,
                     absolute_tolerance=atol, relative_tolerance=rtol)

    def file(self, name, path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        identity = str(path.resolve())
        if identity not in self.hashes:
            self.hashes[identity] = sha(path)
        self.equal(name, self.hashes[identity], expected)


def basic(truth, prediction):
    """Explicit sums: no production metric function is called."""
    residual = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    square_error = float(np.dot(residual, residual))
    centered = truth - np.mean(truth, dtype=np.float64)
    centered_sum = float(np.dot(centered, centered))
    return dict(r2=1.0 - square_error / centered_sum if centered_sum >= 1e-12 else np.nan,
                mae=float(np.sum(np.abs(residual), dtype=np.float64) / len(truth)),
                rmse=float(np.sqrt(square_error / len(truth))), n=len(truth))


def balanced(truths, predictions):
    # Each point receives weight 1/(number of cases * nodes in its case).
    mean = sum(float(np.sum(y, dtype=np.float64)) / len(y) for y in truths) / len(truths)
    error, absolute, spread = 0.0, 0.0, 0.0
    for y, p in zip(truths, predictions):
        residual, centered = p - y, y - mean
        error += float(np.dot(residual, residual)) / len(y)
        absolute += float(np.sum(np.abs(residual), dtype=np.float64)) / len(y)
        spread += float(np.dot(centered, centered)) / len(y)
    count = len(truths)
    return dict(r2=1.0 - error / spread if spread / count > 1e-12 else np.nan,
                mae=absolute / count, rmse=float(np.sqrt(error / count)),
                n=sum(len(y) for y in truths), n_cases=count)


def hotspot(truth, prediction):
    true_high = truth >= np.quantile(truth, 0.9, method="linear")
    pred_high = prediction >= np.quantile(prediction, 0.9, method="linear")
    intersection = int(np.count_nonzero(true_high & pred_high))
    union = int(np.count_nonzero(true_high | pred_high))
    nt, npred = int(true_high.sum()), int(pred_high.sum())
    return dict(top10_iou=intersection / union, top10_precision=intersection / npred,
                top10_recall=intersection / nt, n_high_true=nt, n_high_pred=npred)


def normalized(values, stats):
    floor, epsilon = float(stats.get("floor", 0.0)), float(stats["eps"])
    return (np.log(np.maximum(values, floor) + epsilon) - float(stats["log"]["mean"])) / float(stats["log"]["std"])


def velocity_moments(truth, prediction, prediction_norm, mean, std):
    """Streaming component moments also detect direction errors hidden by speed."""
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    # Original dataset stores component targets normalized in float32.
    truth_norm = ((truth.astype(np.float32) - mean.astype(np.float32)) /
                  std.astype(np.float32)).astype(np.float64).ravel()
    norm_error = prediction_norm.ravel() - truth_norm
    speed_true = np.sqrt(np.sum(truth * truth, axis=1))
    speed_pred = np.sqrt(np.sum(prediction * prediction, axis=1))
    mask = speed_true > 0.05
    cosine = (np.sum(truth[mask] * prediction[mask], axis=1) /
              np.maximum(speed_true[mask] * speed_pred[mask], 1e-12))
    return dict(n=len(truth), truth_sum=truth.sum(axis=0),
                truth_square_sum=np.sum(truth * truth, axis=0),
                error_square_sum=np.sum(error * error, axis=0),
                normalized_truth_sum=float(truth_norm.sum()),
                normalized_truth_square_sum=float(np.dot(truth_norm, truth_norm)),
                normalized_error_square_sum=float(np.dot(norm_error, norm_error)),
                direction_cosine=float(cosine.mean()) if len(cosine) >= 10 else np.nan,
                direction_cosine_speedweighted=float(np.dot(cosine, speed_true[mask]) /
                                                       speed_true[mask].sum()) if len(cosine) >= 10 else np.nan)


def compare_velocity_vectors(audit, rows, reference):
    counts = np.array([r["n"] for r in rows], np.float64)
    total = counts.sum()
    sums = np.array([r["truth_sum"] for r in rows])
    squares = np.array([r["truth_square_sum"] for r in rows])
    error = np.array([r["error_square_sum"] for r in rows])
    pooled_variance_sum = squares.sum(axis=0) - sums.sum(axis=0) ** 2 / total
    balanced_mean = np.mean(sums / counts[:, None], axis=0)
    balanced_variance = np.mean(squares / counts[:, None], axis=0) - balanced_mean ** 2
    pooled_r2 = 1 - error.sum(axis=0) / pooled_variance_sum
    balanced_r2 = 1 - np.mean(error / counts[:, None], axis=0) / balanced_variance
    rmse = np.sqrt(error.sum(axis=0) / total)
    result = dict(components={})
    for i, component in enumerate(("u", "v", "w")):
        result["components"][component] = dict(r2_casebalanced=float(balanced_r2[i]),
                                                r2_pooled=float(pooled_r2[i]), rmse=float(rmse[i]))
        for key, value in result["components"][component].items():
            audit.number("original_velocity_vector.components." + component + "." + key,
                         value, reference["components"][component][key], atol=5e-6, rtol=5e-6)
    norm_sum = sum(r["normalized_truth_sum"] for r in rows)
    norm_square = sum(r["normalized_truth_square_sum"] for r in rows)
    norm_error = sum(r["normalized_error_square_sum"] for r in rows)
    result.update(vector_rmse_m_s=float(np.sqrt(error.sum() / total)),
                  component_normalized_r2_pooled=1 - norm_error / (norm_square - norm_sum ** 2 / (3 * total)),
                  direction_cosine_casemean=float(np.nanmean([r["direction_cosine"] for r in rows])),
                  direction_cosine_speedweighted_casemean=float(np.nanmean([r["direction_cosine_speedweighted"] for r in rows])),
                  speed_floor_m_s=0.05)
    for key, value in result.items():
        if key != "components":
            audit.number("original_velocity_vector." + key, value, reference[key], atol=5e-6, rtol=5e-6)
    return result


def compare_metrics(audit, label, truth_list, prediction_list, ids, reference):
    audit.equal(label + ".per_case_id_set", sorted(reference["per_case"]), sorted(ids))
    out = dict(field=basic(np.concatenate(truth_list), np.concatenate(prediction_list)),
               field_casebalanced=balanced(truth_list, prediction_list), per_case={})
    for group in ("field", "field_casebalanced"):
        for key, value in out[group].items():
            if key.startswith("n"):
                audit.equal(f"{label}.{group}.{key}", value, reference[group][key])
            else:
                audit.number(f"{label}.{group}.{key}", value, reference[group][key])
    for uid, truth, prediction in zip(ids, truth_list, prediction_list):
        row = dict(overall=basic(truth, prediction), hotspot=hotspot(truth, prediction))
        out["per_case"][uid] = row
        for group in ("overall", "hotspot"):
            for key, value in row[group].items():
                if key.startswith("n"):
                    audit.equal(f"{label}.{uid}.{group}.{key}", value, reference["per_case"][uid][group][key])
                else:
                    audit.number(f"{label}.{uid}.{group}.{key}", value, reference["per_case"][uid][group][key])
        for key, value in row["hotspot"].items():
            audit.number(f"{label}.{uid}.legacy_vertex_hotspot.{key}", value,
                         reference["per_case"][uid]["legacy_vertex_hotspot"][key])
    r2 = np.array([out["per_case"][uid]["overall"]["r2"] for uid in ids])
    aggregate = dict(r2_casemean=float(np.mean(r2)), r2_casemed=float(np.median(r2)),
                     r2_casep10=float(np.quantile(r2, 0.1)),
                     r2_negative_cases=int(np.sum(r2 < 0)),
                     r2_failure_rate=float(np.mean(r2 < 0)))
    for key, value in aggregate.items():
        audit.number(f"{label}.aggregate.{key}", value, reference["aggregate"][key])
    out["aggregate"] = aggregate
    out["hotspot"] = {}
    for key in ("top10_iou", "top10_precision", "top10_recall"):
        values = np.array([out["per_case"][uid]["hotspot"][key] for uid in ids])
        for suffix, value in (("", values.mean()), ("_casemean", values.mean()),
                              ("_casemed", np.median(values))):
            out["hotspot"][key + suffix] = float(value)
            audit.number(f"{label}.hotspot.{key}{suffix}", value, reference["hotspot"][key + suffix])
    return out


def verify(exp, audit):
    gate, provenance, reproduction, manifest, metrics, coverage, radius_audit = (
        read(exp / name) for name in ("evaluation_gate.json", "provenance.json",
            "velocity_reproduction.json", "prediction_manifest.json", "metrics.json",
            "coverage.json", "radius_split_audit.json"))
    stats_path = Path(provenance["normalization_path"])
    stats = read(stats_path)
    split_path = DATA / "split_V5_train138_test34.json"
    split = read(split_path)
    ids, train = split["test_cases"], split["train_cases"]
    audit.equal("fixed_split.test_count", len(ids), 34)
    audit.equal("fixed_split.train_count", len(train), 138)
    audit.equal("fixed_split.test_unique_count", len(set(ids)), 34)
    audit.equal("fixed_split.train_unique_count", len(set(train)), 138)
    audit.equal("fixed_split.train_test_overlap", sorted(set(train) & set(ids)), [])
    for group in split.get("duplicate_geometry_groups", []) + split.get("explicit_related_groups", []):
        audit.require("fixed_split.related_geometry_stays_within_one_partition",
                      not (set(group) & set(train) and set(group) & set(ids)), group=group)
    for name, values in (("prediction_manifest", manifest), ("coverage", coverage),
                         ("geometry", provenance["geometry"]),
                         ("velocity_case_records", reproduction["cases"])):
        audit.equal(name + ".test_set", sorted(values), sorted(ids))
    audit.equal("reproduction.fixed_test_order", reproduction["expected_cases"], ids)
    audit.equal("chunk_manifest.case_set", sorted({c["uid"] for c in provenance["chunks"]}), sorted(ids))
    audit.equal("chunk_manifest.unique_contiguous_indices",
                sorted(c["index"] for c in provenance["chunks"]),
                list(range(len(provenance["chunks"]))))
    for key, expected in (("passed", True), ("checkpoint", "best"), ("n_cases", 34),
                          ("valid_coverage", 1.0), ("velocity_reproduced", True),
                          ("no_new_training", True), ("calibrator_train_test_overlap", 0),
                          ("calibrator_train_matches_current_train138", True)):
        audit.equal("evaluation_gate." + key, gate[key], expected)
    for key, filename in (("provenance_sha256", "provenance.json"),
                          ("metrics_sha256", "metrics.json"),
                          ("prediction_manifest_sha256", "prediction_manifest.json"),
                          ("radius_split_audit_sha256", "radius_split_audit.json")):
        audit.file("gate_hash." + filename, exp / filename, gate[key])
    audit.file("provenance.velocity_reproduction_hash", exp / "velocity_reproduction.json",
               provenance["velocity_reproduction_sha256"])
    audit.file("provenance.normalization_hash", stats_path, provenance["normalization_sha256"])
    audit.file("reproduction.split_hash", split_path, reproduction["split_sha256"])
    audit.file("normalization.split_hash", split_path, stats["split_sha256"])
    audit.equal("normalization.method", stats["method"], "log_z")
    audit.equal("normalization.expected_train_cases", stats["n_cases"], 138)
    audit.equal("normalization.expected_wss_stats_path", str(stats_path),
                str(DATA / "wss_global_stats_train138.json"))
    audit.equal("normalization.train_cases", sorted(stats["train_units"]), sorted(train))
    audit.equal("provenance.no_new_training", provenance["new_training"], False)
    audit.equal("provenance.no_new_centerlines", provenance["new_centerlines"], False)
    audit.equal("reproduction.complete", reproduction["status"], "complete")
    audit.equal("reproduction.passed", reproduction["passed"], True)
    audit.equal("reproduction.checkpoint", reproduction["checkpoint"], "best")
    audit.equal("reproduction.seed", reproduction["frozen_support_seed"], 1234)
    audit.equal("reproduction.sources_unchanged_during_export", reproduction["source_sha256"],
                reproduction["source_sha256_end"])
    for name, table in (("original_velocity_source", reproduction["source_sha256"]),
                        ("evaluation_source", provenance["evaluation_source_sha256"])):
        for path, digest in table.items():
            audit.file(name + "." + path, path, digest)
    frozen = provenance["frozen_sources"]
    for row in frozen["files"]:
        audit.file("frozen_algorithm." + row["path"], row["path"], row["sha256"])
    audit.equal("frozen_algorithm.model_pinned", frozen["model_sha256"], MODEL_SHA)
    audit.equal("frozen_algorithm.inference_truth_contract", frozen["truth_used_at_inference"], False)
    audit.equal("adapter.contract_validation", provenance["adapter_contracts"]["status"], "passed")
    audit.equal("adapter.truth_independence_test", provenance["adapter_contracts"]["truth_disturbed_and_removed_double_outputs_exact"], True)

    evidence = radius_audit["split_evidence"]
    for path_key, hash_key in (("v5_split_path", "v5_split_sha256"),
                               ("legacy_split_path", "legacy_split_sha256"),
                               ("model_path", "model_sha256"),
                               ("model_manifest_path", "model_manifest_sha256")):
        audit.file("calibrator_split_evidence." + path_key, evidence[path_key], evidence[hash_key])
    model_manifest = read(evidence["model_manifest_path"])
    audit.equal("calibrator_manifest.selection_roles", model_manifest["selection_roles"], ["train"])
    audit.equal("calibrator_manifest.model_pinned", model_manifest["model_sha256"], MODEL_SHA)
    cache_dir = Path(model_manifest["cache_dir"])
    audit.equal("calibrator_manifest.actual_cache_dir", str(cache_dir), evidence["calibrator_train_cache_dir"])
    cache_records = {r["filename"]: r for r in evidence["actual_cache_identity_records"]}
    actual_cache_files = sorted(cache_dir.glob("*.npz"))
    audit.equal("calibrator_cache.file_set", [p.name for p in actual_cache_files], sorted(cache_records))
    actual_train = []
    for path in actual_cache_files:
        with zipfile.ZipFile(path) as archive:
            identity_bytes = archive.read("canonical_id.npy")
        uid = str(np.load(io.BytesIO(identity_bytes), allow_pickle=False))
        row = cache_records[path.name]
        audit.equal("calibrator_cache.identity." + path.name, uid, row["canonical_id"])
        audit.equal("calibrator_cache.identity_hash." + path.name,
                    hashlib.sha256(identity_bytes).hexdigest(), row["identity_npy_sha256"])
        actual_train.append(uid)
    audit.equal("calibrator_actual_train.equals_train138", sorted(actual_train), sorted(train))
    audit.equal("calibrator_actual_train.test_overlap", sorted(set(actual_train) & set(ids)), [])
    audit.equal("calibrator_audit.train_list", sorted(radius_audit["calibrator_train_cases"]), sorted(actual_train))

    run = Path(reproduction["run"])
    velocity_stats = read(run / "wss_global_stats.json")["velocity"]
    original_velocity_vector = read(run / "eval/ckpt_best/metrics.json")["test"]["vector"]
    velocity_mean, velocity_std = (np.asarray(velocity_stats[k], np.float64) for k in ("mean", "std"))
    velocity_rows = []
    truths, predictions = [], {key: [] for key in ("test", "oracle", "physics_only")}
    arrays_key = dict(test="pred_wss_pa", oracle="oracle_wss_pa", physics_only="physics_wss_pa")
    full_wall_total = 0
    frame_records = {}
    for position, uid in enumerate(ids):
        prefix = "case." + uid
        row, geom = reproduction["cases"][uid], provenance["geometry"][uid]
        for kind, path, digest in (("velocity_archive", row["path"], row["sha256"]),
                                    ("original_bundle", row["bundle"], row["bundle_sha256"]),
                                    ("original_volume", row["volume"], row["volume_sha256"]),
                                    ("geometry", geom["path"], geom["sha256"]),
                                    ("prediction", manifest[uid]["path"], manifest[uid]["sha256"])):
            audit.file(prefix + ".hash." + kind, path, digest)
        if "bundle" in geom["radius"]:
            audit.file(prefix + ".hash.frozen_radius_bundle", geom["radius"]["bundle"], geom["radius"]["bundle_sha256"])
        for space, verification in row["verification"].items():
            audit.equal(prefix + ".velocity_reproduction." + space, verification["passed"], True)
            audit.require(prefix + ".all_velocity_metric_checks." + space,
                          all(c["passed"] for c in verification["checks"]))
        with np.load(row["path"], allow_pickle=False) as z, \
                np.load(manifest[uid]["path"], allow_pickle=False) as result, \
                np.load(row["bundle"], allow_pickle=False) as bundle, \
                np.load(row["volume"], allow_pickle=False) as volume, \
                np.load(geom["path"], allow_pickle=False) as geometry:
            wall, truth = result["wall_mm"], result["truth_wss_pa"]
            n_wall = len(wall)
            full_wall_total += n_wall
            for identity, archive in (("source", z), ("result", result), ("geometry", geometry)):
                audit.equal(prefix + ".identity." + identity, str(archive["canonical_id"]), uid)
            audit.equal(prefix + ".peak_step", int(z["peak_step"]), 1162)
            audit.equal(prefix + ".original_peak", int(bundle["peak_step"]), 1162)
            audit.equal(prefix + ".volume_peak", int(volume["peak_step"]), 1162)
            step_index = list(bundle["steps"]).index(1162)
            audit.array(prefix + ".wall_order.source", wall, z["wall_mm"])
            audit.array(prefix + ".wall_order.original", wall, bundle["wall_coords_raw"])
            audit.array(prefix + ".truth.source", truth, z["truth_wss_pa"])
            audit.array(prefix + ".truth.original_peak", truth, bundle["wall_wss"][step_index])
            audit.array(prefix + ".truth_vector.original_peak", result["truth_wss_vec_pa"], bundle["wall_wss_vec"][step_index])
            audit.require(prefix + ".coordinates_finite", np.isfinite(wall).all())
            audit.require(prefix + ".truth_finite_nonnegative", np.isfinite(truth).all() and (truth >= 0).all())
            audit.require(prefix + ".full_valid_mask", result["valid_mask"].shape == (n_wall,) and result["valid_mask"].all())
            for key, expected in (("n_wall", n_wall), ("n_valid", n_wall), ("coverage", 1.0)):
                audit.equal(prefix + ".coverage." + key, coverage[uid][key], expected)
            for name, expected in (("export_wall", row["n_wall"]), ("geometry_wall", geom["n_wall"])):
                audit.equal(prefix + ".count." + name, n_wall, expected)
            audit.require(prefix + ".normals_finite", geometry["normals"].shape == (n_wall, 3) and np.isfinite(geometry["normals"]).all())
            audit.require(prefix + ".radius_finite_positive", geometry["radius_mm"].shape == (n_wall,) and np.isfinite(geometry["radius_mm"]).all() and (geometry["radius_mm"] > 0).all())
            rotation = z["transform_rotation"]
            audit.array(prefix + ".rotation.source", rotation, bundle["transform_rotation"])
            audit.array(prefix + ".rotation.orthonormal", rotation @ rotation.T, np.eye(3), atol=1e-7)
            audit.array(prefix + ".interior_order", z["interior_mm"], volume["vol_coords_raw"])
            raw_prediction = z["velocity_pred_raw_m_s"]
            aligned = z["velocity_prediction_norm"] * velocity_std + velocity_mean
            audit.array(prefix + ".velocity.inverse_normalization", aligned, z["velocity_pred_aligned_m_s"], atol=1e-12, rtol=1e-12)
            audit.array(prefix + ".velocity.prediction_raw_frame", aligned @ rotation, raw_prediction, atol=1e-12, rtol=1e-12)
            audit.array(prefix + ".velocity.cfd_raw_frame", np.asarray(volume["vol_velocity_aligned_peak"], np.float64) @ rotation, z["velocity_cfd_raw_m_s"], atol=1e-12, rtol=1e-12)
            velocity_rows.append(velocity_moments(volume["vol_velocity_aligned_peak"], aligned,
                                                  z["velocity_prediction_norm"], velocity_mean, velocity_std))
            audit.equal(prefix + ".interior_count", len(raw_prediction), row["n_interior"])
            for branch, array_name in arrays_key.items():
                prediction = np.asarray(result[array_name], np.float64)
                audit.equal(prefix + ".prediction_shape." + branch, list(prediction.shape), [n_wall])
                audit.require(prefix + ".prediction_finite_nonnegative." + branch,
                              np.isfinite(prediction).all() and (prediction >= 0).all())
                vector_name = array_name.replace("_pa", "_vec_pa")
                vector = result[vector_name]
                audit.require(prefix + ".vector_finite." + branch, vector.shape == (n_wall, 3) and np.isfinite(vector).all())
                audit.array(prefix + ".vector_magnitude." + branch, np.linalg.norm(vector, axis=1), prediction, atol=1e-10, rtol=1e-10)
                predictions[branch].append(prediction)
            audit.array(prefix + ".pred_calibration_scalar", result["physics_wss_pa"] * result["pred_calibration_ratio"], result["pred_wss_pa"], atol=1e-12, rtol=1e-12)
            truths.append(np.asarray(truth, np.float64))
            frame_records[uid] = dict(n_wall=n_wall, n_interior=len(raw_prediction), peak_step=1162,
                                      raw_velocity_frame_verified=True)
        shards = sorted((c for c in provenance["chunks"] if c["uid"] == uid), key=lambda c: c["start"])
        next_start = 0
        for shard in shards:
            shard_name = f"{shard['index']:04d}"
            audit.equal(prefix + ".chunk_continuity." + shard_name, shard["start"], next_start)
            next_start = shard["stop"]
            path = exp / "chunks" / (shard_name + ".npz")
            status = read(path.with_suffix(".json"))
            audit.equal(prefix + ".chunk_passed." + shard_name, status["passed"], True)
            for key in ("index", "uid", "start", "stop"):
                audit.equal(prefix + ".chunk_record." + shard_name + "." + key, status[key], shard[key])
            audit.file(prefix + ".chunk_hash." + shard_name, path, status["sha256"])
            with np.load(path, allow_pickle=False) as cache:
                audit.require(prefix + ".chunk_no_truth_arrays." + shard_name,
                              not any("truth" in key for key in cache.files))
                for branch in ("pred", "cfd"):
                    audit.array(prefix + ".chunk_target_order." + shard_name + "." + branch,
                                cache[branch + "__sample_index"],
                                np.arange(shard["start"], shard["stop"], dtype=np.int64))
        audit.equal(prefix + ".chunk_complete_full_wall", next_start, n_wall)
        print(f"verified identities/arrays/hashes {position + 1}/34: {uid}", flush=True)
    for source, n_wall in (("gate", gate["n_wall"]), ("provenance", provenance["n_wall"]),
                           ("independent_geometry_audit", radius_audit["radius_summary"]["n_v5_effective_wall_targets"])):
        audit.equal("full_valid_wall_count." + source, full_wall_total, n_wall)
    audit.equal("fixed_current_v5_wall_count", full_wall_total, 1231295)
    vector_reproduction = compare_velocity_vectors(audit, velocity_rows, original_velocity_vector)
    derived_metrics = {}
    normalized_truths = [normalized(t, stats) for t in truths]
    for branch, pred in predictions.items():
        derived_metrics[branch] = compare_metrics(audit, branch, truths, pred, ids, metrics[branch])
        derived_metrics[branch]["normalized"] = compare_metrics(
            audit, branch + ".normalized", normalized_truths,
            [normalized(p, stats) for p in pred], ids, metrics[branch]["normalized"])
    return dict(n_cases=len(ids), n_wall=full_wall_total, all_valid_wall_coverage=1.0,
                checkpoint="best", no_new_training=True, calibrator_actual_train_test_overlap=0,
                original_velocity_vector_reproduction=vector_reproduction,
                independently_recomputed_metrics=derived_metrics, cases=frame_records,
                limitations=[
                    "This audits numerical metrics and provenance; it does not retrain the model or independently refit every gradient.",
                    "The frozen historical radius mapper has a known coordinate-unit mismatch; keeping its outputs preserves the requested historical algorithm, not a claim of correct anatomical radius mapping.",
                    "CFD oracle results reuse the same frozen supervised train138 calibrator. They are diagnostics, not an uncalibrated physical upper bound.",
                    "Coverage is all 1,231,295 current V5 valid wall nodes; three invalid original raw nodes were excluded upstream by the established dataset.",
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXP)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this full data/hash audit through Slurm")
    started, audit = time.time(), Audit()
    report = dict(schema_version=1, independent_evaluator=True,
                  production_metric_functions_imported=False,
                  job_id=os.environ["SLURM_JOB_ID"],
                  created_at=datetime.now(timezone.utc).isoformat(),
                  verification_script=str(Path(__file__).resolve()),
                  verification_script_sha256=sha(__file__),
                  numpy_version=np.__version__)
    try:
        report.update(verify(args.experiment.resolve(), audit))
    except Exception:
        report["exception"] = traceback.format_exc()
    failures = [row for row in audit.checks if not row["passed"]]
    report.update(passed=not failures and "exception" not in report,
                  checks_total=len(audit.checks), failures=failures,
                  checked_file_sha256=audit.hashes, checks=audit.checks,
                  elapsed_seconds=time.time() - started)
    target = args.experiment / "independent_verification.json"
    temp = target.with_suffix(".tmp.json")
    temp.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(target)
    print(json.dumps({k: report[k] for k in ("passed", "checks_total", "elapsed_seconds")}), flush=True)
    if not report["passed"]:
        print(json.dumps(clean(failures[:10]), ensure_ascii=False, indent=2), flush=True)
        if "exception" in report:
            print(report["exception"], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
