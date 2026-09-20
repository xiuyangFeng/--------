"""Read-only final acceptance of the registered M2 experiment deliverables."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from training_wss_min import config as C, dataset as D, metrics as M
from training_wss_min.tools import report_m2_optimization as R
from training_wss_min.tools.m2_optimization_common import (
    ANCHOR_CONFIG, ANCHOR_RUN, COMBINATIONS, EXP, ROOT, RUNS,
    fingerprints, save_json, sha, stamp,
)
from training_wss_min.tools.tracker_section import get_section


def require(value, message):
    if not value:
        raise RuntimeError(message)


def input_boundary(inputs, arms):
    """Rebuild the manifest independently from M2's fixed data configuration."""
    require(inputs["new_geometry_used"] is False, "unapproved geometry used")
    require(sha(ANCHOR_CONFIG) == inputs["anchor_config_sha256"], "M2 anchor changed")
    anchor = C.ExpConfig.from_json(ANCHOR_CONFIG)
    data = anchor.data
    require(data.target == "wss" and data.timesteps == "peak" and len(data.input_features) == 25,
            "anchor no longer uses original 25D peak WSS")
    require(not any((data.feature_stats_path, data.case_features_path, data.waveform_path)),
            "anchor has additional external data dependencies")
    for arm in arms:
        require(C.ExpConfig.from_dict(arm["resolved_config"]).data == data,
                f"{arm['id']}: original M2 data configuration changed")
    split = json.loads(Path(data.split_path).read_text())
    train, test = split["train_cases"], split["test_cases"]
    cases = train + test
    require(len(train) == 138 and len(test) == 34 and len(set(cases)) == 172
            and not split.get("val_cases") and not split.get("unused_cases"),
            "original train138/test34 partition changed")
    require(all(len(Path(case).parts) == 3 and not Path(case).is_absolute()
                and ".." not in Path(case).parts for case in cases), "noncanonical split paths")
    consumed = {str(Path(data.split_path)), str(Path(data.wss_stats_path))}
    volumes = set()
    for case in cases:
        consumed.add(str(Path(data.data_root) / case / "bundle.npz"))
        consumed.add(str(Path(data.point_features_root) / case / "features.npz"))
        # WSS never loads these volume files; they were frozen as extra evidence.
        volumes.add(str(Path(data.data_root) / case / "volume.npz"))
    expected, listed = consumed | volumes, set(inputs["files"])
    require(listed == expected,
            f"input manifest boundary differs: missing={sorted(expected - listed)}, unexpected={sorted(listed - expected)}")
    return {"training_cases": len(train), "test_cases": len(test),
            "runtime_input_files": len(consumed), "additional_frozen_volume_files": len(volumes)}, set(test)


def verify_record(record, expected_path=None):
    path = Path(record["path"])
    if expected_path is not None:
        require(path == Path(expected_path), f"diagnostic artifact path differs: {path}")
    require(path.is_file() and sha(path) == record["sha256"], f"diagnostic artifact changed/missing: {path}")


BOUNDARY_CONTEXT = {"arm": "MO-P4", "checkpoint": "best", "case": "AAA/unruputer/ZHANG_YONG_ZHI", "space": "log_z"}
BOUNDARY_METRIC = "distribution.pred_above_one_fraction"
BOUNDARY_PROBE = EXP / "threshold_probe/job_13980/probe.json"
BOUNDARY_PROBE_SHA256 = "9825e4c07b704691dc9d97ebc4eebf6c449e86a5278da5ac800a16d24b6638fe"
BOUNDARY_SOURCE_HASHES = {
    "metrics": "6f4c49970ed2508920c2bac88ae8e5e057f3a41c0e66f483612e1772a9ad197e",
    "checkpoint": "a84e2faad6fc6286d8436b3e52ac3bec42eceaf62ae693c0cf678ee388187e29",
    "config": "71ac891c63d87da0ce87e8bbddc3aeefe67dc50833ec694dd6465099cae7e391",
}
CAPTURE_RECORD = {"path": str(EXP / "diagnostics_collection/provenance.json"),
                  "sha256": "95c11973694af1716564a5f2aa4a40c1c48655e55693c9e063250c79c1d9c725"}
BATCH_RECORD = {"path": str(EXP / "threshold_batch_probe/job_13983/probe.json"),
                "sha256": "92bd5b2ebeaf5a63cb22a3505db29a9edf7c0d7f497a4fe177873bfc358f7baf"}
HISTORICAL_CAPTURE_RECORD = {"path": str(EXP / "diagnostics_failed_13981/provenance.json"),
                            "sha256": "8a995b2467a6abefade8294a947abc57d5338a635f2eddafae6bba1f77760fd5"}


def load_verified_arrays(item):
    verify_record(item)
    with np.load(item["path"], allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def independent_metric_check(actual, saved):
    def leaves(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from leaves(item, prefix + "." + key if prefix else key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield prefix, value
    a, b = dict(leaves(actual)), dict(leaves(saved))
    require(a.keys() == b.keys(), "independent raw metric keys differ")
    rows = []
    for key, reference in b.items():
        value = a[key]
        tolerance = 0. if isinstance(reference, int) or not math.isfinite(reference) else 5e-6 + 5e-6 * abs(reference)
        passed = value == reference if isinstance(reference, int) else not math.isfinite(value) if not math.isfinite(reference) else math.isfinite(value) and abs(value-reference) <= tolerance
        rows.append({"metric": key, "reference": reference if math.isfinite(reference) else None,
                     "recomputed": value if math.isfinite(value) else None,
                     "absolute_difference": abs(value-reference) if math.isfinite(value) and math.isfinite(reference) else None,
                     "tolerance": tolerance, "passed": bool(passed)})
    return {"passed": all(row["passed"] for row in rows), "checks": rows}


def raw_case_metrics(case, truth, prediction):
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    actual = M.regional_metrics(case["pos"], case["local_radius"], truth, prediction)
    actual["calibration"] = M.calibration_metrics(truth, prediction)
    actual["distribution"] = M.distribution_metrics(truth, prediction)
    actual["hotspot"] = M.hotspot_localization_metrics(truth, prediction, case["pos"])
    actual["legacy_vertex_hotspot"] = dict(actual["hotspot"])
    return actual


def verify_batch_probe():
    """Recompute all 64 raw probe vectors independently of probe/finalizer helpers."""
    for item in (CAPTURE_RECORD, BATCH_RECORD, HISTORICAL_CAPTURE_RECORD):
        verify_record(item)
    capture = json.loads(Path(CAPTURE_RECORD["path"]).read_text())
    historical = json.loads(Path(HISTORICAL_CAPTURE_RECORD["path"]).read_text())
    batch = json.loads(Path(BATCH_RECORD["path"]).read_text())
    require(batch["job_id"] == "13983" and batch["completed"] is True and batch["metric_accepted"] is False
            and batch["status"] == "independent_review_required" and batch["collector"] == CAPTURE_RECORD
            and batch["repeats_per_case"] == 32 and batch["inference_count"] == 64 and batch["target_count"] == 2,
            "batch probe completion/count/identity differs")
    verify_record(batch["script"], ROOT / "training_wss_min/tools/probe_m2_boundary_batch.py")
    require(batch["script"]["sha256"] == "793e2599280507b388ce956e3aaf4da0899f99240d24fbae92ff23a142cc08af", "batch probe implementation changed")
    verify_record(batch["metric_verification_script"], ROOT / "training_wss_min/experiments/v6_multiradius_bt_20260909/plot_paired_wall_fields.py")
    require(batch["source_sha256_before"] == batch["source_sha256_after"] == capture["source_sha256_before"]
            == capture["source_sha256_after"] == {phase: fingerprints(phase) for phase in batch["source_sha256_before"]},
            "batch/capture frozen training sources differ")
    expected_contexts = {
        ("MO-S3", "last", "ILO/YU_XIANG_SHENG-1/before", "log_z", "distribution.pred_below_zero_fraction"): (capture, CAPTURE_RECORD, 13958),
        ("MO-P2", "last", "ILO/ZHANG_JIN_CHUN-1/before", "Pa", BOUNDARY_METRIC): (historical, HISTORICAL_CAPTURE_RECORD, 42963),
    }
    expected_inputs = {**capture["inputs"], CAPTURE_RECORD["path"]: CAPTURE_RECORD["sha256"],
                       HISTORICAL_CAPTURE_RECORD["path"]: HISTORICAL_CAPTURE_RECORD["sha256"]}
    final_inputs = {BATCH_RECORD["path"]: BATCH_RECORD["sha256"],
                    **{item["path"]: item["sha256"] for item in (batch["script"], batch["metric_verification_script"])}}
    targets, arrays_checked, scalar_checks = {}, 0, 0
    require(len(batch["groups"]) == 2, "batch group population differs")
    for group in batch["groups"].values():
        require(group["completed"] is True and len(group["targets"]) == 1 and len(group["repeats"]) == 32,
                "batch target/repeat population differs")
        target = group["targets"][0]
        c = target["context"]
        identity = tuple(c[key] for key in ("arm", "checkpoint", "case", "space")) + (target["metric"],)
        require(identity in expected_contexts and identity not in targets, "unexpected/duplicate batch target")
        source_capture, origin_record, known_vertex = expected_contexts[identity]
        uid, arm, checkpoint = c["case"], c["arm"], c["checkpoint"]
        source = source_capture["runs"][arm][checkpoint]
        source_case = source["cases"][uid]
        require(group["arm"] == arm and group["case"] == uid and group["checkpoint"] == checkpoint
                and group["run_name"] == target["run_name"] == source["run_name"]
                and group["sources"] == target["sources"] == source["sources"], "batch run/source identity differs")
        run = RUNS / source["run_name"]
        expected_source_paths = {str(run / name) for name in (f"ckpt_{checkpoint}.pt", "config.json", "feature_stats.json",
                                  "wss_global_stats.json", f"eval/ckpt_{checkpoint}/metrics.json")}
        require(set(group["sources"]) == expected_source_paths, "batch source file boundary differs")
        expected_inputs.update(group["sources"])
        cfg = C.ExpConfig.from_json(run / "config.json")
        stats = json.loads((run / "wss_global_stats.json").read_text())
        require(stats == group["wss_stats"], "batch normalization stats differ")
        cohort, case_name = uid.rsplit("/", 1)
        case = D.load_case(cohort, case_name, stats, target=cfg.data.target, target_normalization=cfg.data.target_normalization,
            data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version, timesteps=cfg.data.timesteps,
            extra_point_features=C.v6_point_features(cfg), point_features_root=cfg.data.point_features_root)
        origin = target["capture_origin"]
        base = EXP / ("diagnostics_collection" if source_capture is capture else "diagnostics_failed_13981")
        expected_array = {"path": str(base / "predictions" / arm / checkpoint / f"{uid.replace('/', '__')}.npz"), "sha256": source_case["arrays"]["sha256"]}
        expected_geometry = {"path": str(base / "geometry" / f"{uid.replace('/', '__')}.npz"), "sha256": source_capture["geometry"][uid]["arrays"]["sha256"]}
        require(origin == {"job_id": str(source_capture["job_id"]), "provenance": origin_record, "arrays": expected_array,
                           "primary_complete_capture": source_capture is capture}
                and target["collector_arrays"] == expected_array and target["geometry_arrays"] == group["geometry_arrays"] == expected_geometry,
                "batch capture/geometry linkage differs")
        for item in (expected_array, expected_geometry):
            expected_inputs[item["path"]] = item["sha256"]
        captured = load_verified_arrays(expected_array)
        geometry = load_verified_arrays(expected_geometry)
        require(np.array_equal(geometry["truth_norm"], case["y_norm"]) and np.array_equal(geometry["truth_pa"], case["y_raw"])
                and np.array_equal(geometry["pos_norm"], case["pos"]), "batch original truth/geometry/order differs")
        n = len(geometry["truth_norm"])
        failure = target["original_failure"]
        require(target["n"] == source_case["wall_points"] == n and failure in source_case["metric_verification"][c["space"]]["failed_checks"],
                "batch target is not the recorded original failure")
        saved = json.loads((run / f"eval/ckpt_{checkpoint}/metrics.json").read_text())["test"]
        saved_space = saved if c["space"] == "Pa" else saved["normalized"]
        field = target["metric"].split(".")[-1]
        reference_fraction = saved_space["per_case"][uid]["distribution"][field]
        reference_count = round(reference_fraction * n)
        require(reference_fraction == failure["reference"] == reference_count / n
                and target["reference_count"] == reference_count, "batch original fraction/count differs")
        pred_key = "prediction_pa" if c["space"] == "Pa" else "prediction_norm"
        below = field == "pred_below_zero_fraction"
        side = lambda values: values < 0. if below else values > 1.
        capture_count = int(np.count_nonzero(side(captured[pred_key])))
        require(capture_count == target["collector_count"] and capture_count / n == failure["recomputed"]
                and abs(capture_count-reference_count) == 1 and target["count_delta"] == capture_count-reference_count,
                "batch capture fraction/count differs")
        bundles, counts = [], []
        for index, repeat in enumerate(group["repeats"]):
            require(repeat["index"] == index, "batch repeat ordering differs")
            verify_record(repeat["arrays"], Path(BATCH_RECORD["path"]).parent / arm / checkpoint / uid.replace("/", "__") / f"repeat_{index}.npz")
            bundle = load_verified_arrays(repeat["arrays"])
            require(set(bundle) == {"prediction_norm", "prediction_pa", "truth_norm", "truth_pa", "wall_node_id_cas"}, "batch NPZ fields differ")
            for key, values in bundle.items():
                require(values.shape == (n,) and np.isfinite(values).all(), "batch NPZ finite population differs")
                if key in ("truth_norm", "truth_pa", "wall_node_id_cas"):
                    require(np.array_equal(values, geometry[key]), "batch truth/vertex IDs differ")
            require(bundle["prediction_norm"].dtype == bundle["prediction_pa"].dtype == np.dtype("float64")
                    and np.array_equal(bundle["prediction_norm"], bundle["prediction_norm"].astype(np.float32).astype(np.float64))
                    and np.array_equal(np.clip(D.denormalize_wss(bundle["prediction_norm"], stats), 0., None), bundle["prediction_pa"]),
                    "batch raw float32 promotion/Pa transform differs")
            for space, pk, tk in (("Pa", "prediction_pa", "truth_pa"), ("log_z", "prediction_norm", "truth_norm")):
                expected = (saved if space == "Pa" else saved["normalized"])["per_case"][uid]
                numeric = independent_metric_check(raw_case_metrics(case, bundle[tk], bundle[pk]), expected)
                require(numeric == repeat["metric_verification"][space], "batch raw metric metadata differs")
                require(all(row["passed"] or (space == c["space"] and row["metric"] == target["metric"]) for row in numeric["checks"]),
                        "batch has an unrelated continuous or other-space failure")
                scalar_checks += len(numeric["checks"])
            bundles.append(bundle)
            counts.append(int(np.count_nonzero(side(bundle[pred_key]))))
            arrays_checked += 1
            final_inputs[repeat["arrays"]["path"]] = repeat["arrays"]["sha256"]
        require(set(counts) == {reference_count, capture_count}, "batch did not reproduce both and only the expected counts")
        reference_index = counts.index(reference_count)
        reference = bundles[reference_index]
        reference_side = side(reference[pred_key])
        changed = np.flatnonzero(side(captured[pred_key]) != reference_side).tolist()
        require(changed == [known_vertex], "batch capture crosses an unexpected/additional vertex")
        for bundle, count in zip(bundles, counts):
            require(np.flatnonzero(side(bundle[pred_key]) != reference_side).tolist() == ([] if count == reference_count else changed),
                    "batch repeat crosses additional vertices")
        values = [float(bundle["prediction_norm"][known_vertex]) for bundle in bundles]
        captured_value = float(captured["prediction_norm"][known_vertex])
        require(min(values) <= captured_value <= max(values), "capture boundary value is outside the raw observed probe envelope")
        threshold = 0. if below else 1.
        normalized_threshold = threshold if c["space"] == "log_z" else float(D.normalize_wss(np.array([threshold], dtype=np.float64), stats)[0])
        proof = {"metric": target["metric"], "context": c, "n": n, "reference_count": reference_count,
                 "recomputed_count": capture_count, "count_delta": capture_count-reference_count,
                 "reference_fraction": reference_fraction, "recomputed_fraction": capture_count/n, "one_vertex_fraction": 1./n,
                 "vertex_index": known_vertex, "vertex_node_id": int(geometry["wall_node_id_cas"][known_vertex]),
                 "current_vertex_value": float(captured[pred_key][known_vertex]), "reference_vertex_value": float(reference[pred_key][known_vertex]),
                 "metric_threshold": threshold, "comparison": "lt" if below else "gt", "normalized_threshold": normalized_threshold,
                 "current_vertex_normalized_value": captured_value, "reference_vertex_normalized_value": float(reference["prediction_norm"][known_vertex]),
                 "probe_vertex_normalized_min": min(values), "probe_vertex_normalized_max": max(values),
                 "probe_vertex_normalized_values": values, "probe_counts": counts, "probe_repeat_count": len(bundles),
                 "max_normalized_distance_to_threshold": max(abs(value-normalized_threshold) for value in [captured_value, *values]),
                 "crossing_indices_vs_probe_reference": changed, "probe_reference_index": reference_index,
                 "max_abs_vs_probe_reference": float(np.max(np.abs(captured[pred_key]-reference[pred_key]))),
                 "finite_mask_matches_probe": True, "truth_matches_probe": True,
                 "float32_ulp": None if below else abs(float(np.spacing(np.float32(normalized_threshold)))),
                 "ulp_limit_applied": False}
        targets[identity] = {"target": target, "proof": proof, "group": group, "reference_array": group["repeats"][reference_index]["arrays"]}
    require(set(targets) == set(expected_contexts) and arrays_checked == 64 and scalar_checks == 10112, "batch independent verification coverage differs")
    require(batch["inputs"] == expected_inputs, "batch input manifest boundary differs")
    for path, digest in expected_inputs.items():
        require(sha(path) == digest, f"batch input changed: {path}")
    final_inputs.update(expected_inputs)
    return {"capture": capture, "batch": batch, "targets": targets, "expected_final_inputs": final_inputs,
            "arrays_verified": arrays_checked, "scalar_checks_verified": scalar_checks}


def verify_batch_review(review, verification, context, prediction_record, source_case, sources, batch):
    identity = tuple(context[key] for key in ("arm", "checkpoint", "case", "space")) + (review["metric"],)
    require(identity in batch["targets"], "new boundary review has no pinned batch evidence")
    evidence = batch["targets"][identity]
    target, proof, group = evidence["target"], evidence["proof"], evidence["group"]
    require(target["capture_origin"]["primary_complete_capture"] is True, "historical probe cannot grant a current metric exception")
    require(review["review_status"] == "accepted_discrete_boundary" and review["context"] == context
            and all(review[key] == value for key, value in proof.items()), "new boundary review differs from independent raw proof")
    require(prediction_record["sha256"] == source_case["arrays"]["sha256"] == target["collector_arrays"]["sha256"],
            "final reviewed prediction is not the immutable capture")
    require(verification["strict_passed"] is False and verification["strict_failed_checks"] == 1
            and verification["failed_checks"] == [target["original_failure"]], "new review removed/altered its original strict failure")
    require(review["probe"] == BATCH_RECORD and review["probe_reference_array"] == evidence["reference_array"]
            and review["probe_target_id"] == target["id"] and review["probe_source_script_sha256"] == batch["batch"]["script"]["sha256"]
            and review["capture_origin"] == {"job_id": "13982", "provenance": CAPTURE_RECORD, "arrays": target["collector_arrays"]},
            "new boundary provenance linkage differs")
    run = RUNS / group["run_name"]
    for key, name in (("source_metrics_sha256", f"eval/ckpt_{context['checkpoint']}/metrics.json"),
                      ("checkpoint_sha256", f"ckpt_{context['checkpoint']}.pt"), ("config_sha256", "config.json")):
        require(review[key] == sources[str(run / name)], "new boundary original source differs")
    expected_inputs = {BATCH_RECORD["path"]: BATCH_RECORD["sha256"],
        **{item["path"]: item["sha256"] for item in (batch["batch"]["script"], batch["batch"]["metric_verification_script"])},
        **{repeat["arrays"]["path"]: repeat["arrays"]["sha256"] for repeat in group["repeats"]}}
    require(review["evidence_inputs"] == expected_inputs, "new boundary evidence manifest differs")


def verify_finalization(diagnostics, batch):
    capture = batch["capture"]
    require(capture["job_id"] == "13982" and capture["passed"] is False and capture["completed"] is True
            and capture["full_inference_coverage_complete"] is True and capture["collect_failures"] is True
            and capture["evaluations_completed"] == 42 and capture["cases_evaluated"] == 1428
            and capture["strict_passed_numeric_checks"] == 225622 and capture["strict_failed_numeric_checks"] == 2
            and capture["boundary_review_count"] == 1 and capture["unreviewed_metric_failure_count"] == 1,
            "immutable full capture status/coverage differs")
    verify_record(capture["script"], ROOT / "training_wss_min/tools/diagnose_m2_optimization.py")
    require(diagnostics["capture"] == {"job_id": "13982", "provenance": CAPTURE_RECORD,
        "slurm": {"state": "FAILED", "exit_code": "1:0"}, "passed": False, "strict_passed": False,
        "script": capture["script"], "full_inference_coverage_complete": True}, "CPU result obscures its GPU capture status")
    final = diagnostics["finalization"]
    verify_record(final["source"], ROOT / "training_wss_min/tools/finalize_m2_diagnostic_capture.py")
    require(final["source"] == diagnostics["script"] and final["mode"] == "cpu_review_of_saved_capture"
            and final["completed"] is True and final["batch_probe"] == BATCH_RECORD and final["probe_job_id"] == "13983"
            and final["raw_metrics_recomputed"] == 225624 and final["new_boundary_review_count"] == 1
            and final["inherited_boundary_review_count"] == 1 and diagnostics["completed"] is True
            and diagnostics["figures_rendered"] is True and diagnostics["unreviewed_metric_failure_count"] == 0
            and diagnostics["unreviewed_metric_failures"] == [], "CPU finalization did not close the exact captured scope")
    ledger = diagnostics["batch_probe_reverification"]
    require(len(ledger) == len(batch["targets"]) == 2, "batch evidence ledger population differs")
    entries = {entry["target_id"]: entry for entry in ledger}
    require(set(entries) == {item["target"]["id"] for item in batch["targets"].values()}, "batch evidence ledger identities differ")
    for item in batch["targets"].values():
        target, proof = item["target"], item["proof"]
        entry = entries[target["id"]]
        require(entry["context"] == target["context"] and entry["metric"] == target["metric"]
                and entry["capture_origin"] == target["capture_origin"]
                and entry["primary_current_capture"] is target["capture_origin"]["primary_complete_capture"]
                and entry["raw_records_verified"] is True and entry["boundary_evidence_sufficient"] is True
                and entry["insufficient_reason"] is None and entry["grants_current_metric_exception"] is False
                and all(entry["boundary_evidence"][key] == value for key, value in proof.items()),
                "batch evidence ledger differs from independent raw proof")


def verify_boundary_probe():
    """Independently recount all eight pinned raw arrays; no metric tolerance changes."""
    verify_record({"path": str(BOUNDARY_PROBE), "sha256": BOUNDARY_PROBE_SHA256})
    probe = json.loads(BOUNDARY_PROBE.read_text())
    require(probe["completed"] is True and probe["job_id"] == "13980"
            and probe["reference_n"] == 57531 and probe["reference_fraction"] == 13859 / 57531
            and len(probe["repeats"]) == 8, "threshold probe scope/completion differs")
    verify_record({"path": str(ROOT / "training_wss_min/tools/probe_m2_threshold.py"), "sha256": probe["script_sha256"]})
    run = RUNS / "m2_optimization_20260909/MO-P4_s1234"
    for key, name in (("metrics", "eval/ckpt_best/metrics.json"), ("checkpoint", "ckpt_best.pt"), ("config", "config.json")):
        require(sha(run / name) == BOUNDARY_SOURCE_HASHES[key], f"boundary source changed: {name}")
    require(probe["source_metrics_sha256"] == BOUNDARY_SOURCE_HASHES["metrics"], "probe metric source differs")
    saved = json.loads((run / "eval/ckpt_best/metrics.json").read_text())["test"]["normalized"]["per_case"][BOUNDARY_CONTEXT["case"]]
    require(saved["distribution"]["n"] == 57531
            and saved["distribution"]["pred_above_one_fraction"] == 13859 / 57531,
            "original threshold metric changed")
    arrays, counts, truth, union, maxima = [], [], None, set(), []
    for index, repeat in enumerate(probe["repeats"]):
        path = BOUNDARY_PROBE.parent / f"repeat_{index}.npz"
        verify_record({"path": repeat["array_path"], "sha256": repeat["array_sha256"]}, path)
        with np.load(path, allow_pickle=False) as payload:
            require(set(payload.files) == {"prediction_norm", "truth_norm"}, "probe NPZ fields differ")
            pred, target = payload["prediction_norm"].copy(), payload["truth_norm"].copy()
        require(pred.shape == target.shape == (57531,) and pred.dtype == np.dtype("float64")
                and target.dtype == np.dtype("float32") and np.isfinite(pred).all() and np.isfinite(target).all(),
                "probe raw population/dtype/finite values differ")
        require(truth is None or np.array_equal(truth, target), "probe truth or vertex ordering differs")
        truth = target
        first = arrays[0] if arrays else pred
        count = int(np.count_nonzero(pred > 1.))
        crossings = [{"index": int(i), "first": float(first[i]), "current": float(pred[i])}
                     for i in np.flatnonzero((pred > 1.) != (first > 1.))]
        maximum = float(np.max(np.abs(pred - first)))
        nearest = [{"index": int(i), "value": float(pred[i]), "distance": float(abs(pred[i] - 1.)),
                    "float32_ulp": float(np.spacing(np.float32(pred[i])))}
                   for i in np.argsort(np.abs(pred - 1.))[:12]]
        require(repeat["index"] == index and repeat["dtype"] == str(pred.dtype) and repeat["n"] == len(pred)
                and repeat["above_one_count"] == count and repeat["above_one_fraction"] == count / len(pred)
                and repeat["threshold_crossings_vs_first"] == crossings and repeat["max_abs_vs_first"] == maximum
                and repeat["nearest_to_one"] == nearest, f"probe repeat {index}: raw recomputation differs")
        arrays.append(pred)
        counts.append(count)
        maxima.append(maximum)
        union.update(item["index"] for item in crossings)
    upper = float(np.nextafter(np.float32(1.), np.float32(np.inf)))
    require(counts == [13860, 13860, 13859, 13860, 13860, 13859, 13860, 13860]
            and probe["counts"] == sorted(set(counts)) and union == {55137}
            and max(maxima) == 8.940696716308594e-7, "probe observed counts/crossings/differences changed")
    reference = arrays[2]
    for pred, count in zip(arrays, counts):
        require(np.flatnonzero((pred > 1.) != (reference > 1.)).tolist() == ([] if count == 13859 else [55137])
                and pred[55137] == (1. if count == 13859 else upper), "probe exceeds the observed one-vertex/one-ULP boundary")
    return {"probe": probe, "reference": reference, "truth": truth, "upper": upper,
            "record": {"path": str(BOUNDARY_PROBE), "sha256": BOUNDARY_PROBE_SHA256},
            "reference_array": {"path": probe["repeats"][2]["array_path"], "sha256": probe["repeats"][2]["array_sha256"]},
            "evidence_inputs": {str(BOUNDARY_PROBE): BOUNDARY_PROBE_SHA256,
                str(ROOT / "training_wss_min/tools/probe_m2_threshold.py"): probe["script_sha256"],
                **{repeat["array_path"]: repeat["array_sha256"] for repeat in probe["repeats"]}},
            "recomputed_counts": counts, "unique_crossing": sorted(union), "maximum_abs_vs_first": max(maxima)}


def verify_boundary_review(review, verification, context, prediction_record, geometry, sources, evidence):
    """Allow only the pinned discrete failure, with the original strict failure retained."""
    require(context == BOUNDARY_CONTEXT and review["context"] == context
            and review["metric"] == BOUNDARY_METRIC and review["review_status"] == "accepted_discrete_boundary",
            "unreviewed metric/case/checkpoint boundary exception")
    require(verification["strict_passed"] is False and verification["strict_failed_checks"] == 1
            and len(verification["failed_checks"]) == 1, "boundary strict failure was discarded")
    require(review["probe"] == evidence["record"] and review["probe_reference_array"] == evidence["reference_array"]
            and review["probe_source_script_sha256"] == evidence["probe"]["script_sha256"]
            and review["evidence_inputs"] == evidence["evidence_inputs"], "boundary probe linkage differs")
    for key, source_key, name in (("source_metrics_sha256", "metrics", "eval/ckpt_best/metrics.json"),
                                  ("checkpoint_sha256", "checkpoint", "ckpt_best.pt"), ("config_sha256", "config", "config.json")):
        require(review[key] == BOUNDARY_SOURCE_HASHES[source_key]
                == sources[str(RUNS / "m2_optimization_20260909/MO-P4_s1234" / name)], "boundary source identity differs")
    with np.load(prediction_record["path"], allow_pickle=False) as payload:
        prediction = payload["prediction_norm"].copy()
    with np.load(geometry["arrays"]["path"], allow_pickle=False) as payload:
        truth = payload["truth_norm"].copy()
    require(prediction.shape == truth.shape == (57531,) and prediction.dtype == np.dtype("float64")
            and np.isfinite(prediction).all() and np.array_equal(truth, evidence["truth"]),
            "diagnostic boundary raw population/truth/order differs")
    n, count = len(prediction), int(np.count_nonzero(prediction > 1.))
    crossing = np.flatnonzero((prediction > 1.) != (evidence["reference"] > 1.)).tolist()
    require(count == 13860 and crossing == [55137] and prediction[55137] == evidence["upper"],
            "diagnostic boundary exceeds the probed vertex/count/ULP")
    expected = {"n": n, "reference_count": 13859, "recomputed_count": count, "count_delta": 1,
                "reference_fraction": 13859 / n, "recomputed_fraction": count / n, "one_vertex_fraction": 1. / n,
                "vertex_index": 55137, "current_vertex_value": evidence["upper"], "reference_vertex_value": 1.,
                "float32_ulp": evidence["upper"] - 1., "crossing_indices_vs_probe_reference": crossing,
                "max_abs_vs_probe_reference": float(np.max(np.abs(prediction - evidence["reference"]))),
                "finite_mask_matches_probe": True, "truth_matches_probe": True}
    require(all(review[key] == value for key, value in expected.items()), "boundary record differs from raw arrays")
    difference = abs(count / n - 13859 / n)
    expected_failure = {"metric": BOUNDARY_METRIC, "reference": 13859 / n, "recomputed": count / n,
                        "absolute_difference": difference, "tolerance": 5e-6 + 5e-6 * abs(13859 / n), "passed": False}
    require(verification["failed_checks"] == [expected_failure]
            and verification["max_absolute_difference"] >= difference, "original strict boundary failure/tolerance changed")


def verify_diagnostic_jobs(diagnostics, probe):
    """Keep CPU success, complete-but-failed GPU capture, and every failed attempt distinct."""
    failed_path = EXP / "diagnostics_failed_13979/provenance.json"
    failed_record = {"path": str(failed_path), "sha256": "2f6acd726b1f89913fcfdcdb4c5b7d7cfd85576a38779d3a7d53cd4173911ce9"}
    verify_record(failed_record)
    failed = json.loads(failed_path.read_text())
    require(failed["job_id"] == "13979" and failed["passed"] is False and failed["smoke"] is False
            and "MO-P4/best/AAA/unruputer/ZHANG_YONG_ZHI" in failed["error"]
            and BOUNDARY_METRIC in failed["error"], "failed diagnostic evidence differs")
    archived_source = {"path": str(failed_path.parent / "diagnose_source_failed_13979.py"), "sha256": failed["script"]["sha256"]}
    verify_record(archived_source)
    historical = json.loads(Path(HISTORICAL_CAPTURE_RECORD["path"]).read_text())
    historical_source = {"path": str(EXP / "diagnostics_failed_13981/diagnose_source_failed_13981.py"),
                         "sha256": historical["script"]["sha256"]}
    verify_record(HISTORICAL_CAPTURE_RECORD)
    verify_record(historical_source)
    require(historical["job_id"] == "13981" and historical["passed"] is False, "13981 failed capture status changed")
    job = str(diagnostics["job_id"])
    require(job == "13984" and probe["probe"]["job_id"] == "13980"
            and diagnostics["capture"]["job_id"] == "13982" and diagnostics["finalization"]["probe_job_id"] == "13983",
            "CPU finalizer/GPU capture/probe job identity differs")
    expected = {job: ("COMPLETED", "0:0"), "13980": ("COMPLETED", "0:0"), "13979": ("FAILED", "1:0"),
                "13981": ("FAILED", "1:0"), "13982": ("FAILED", "1:0"), "13983": ("COMPLETED", "0:0")}
    output = subprocess.check_output(["/public/slurm/bin/sacct", "-j", ",".join(expected), "-P",
        "--format=JobID,JobName,State,ExitCode,Start,End,Elapsed,NodeList"], text=True, timeout=60)
    rows = list(csv.DictReader(io.StringIO(output), delimiter="|"))
    top = {row["JobID"]: row for row in rows if row["JobID"] in expected}
    require(set(top) == set(expected), "missing diagnostic/probe/failed-attempt Slurm records")
    for job_id, status in expected.items():
        require((top[job_id]["State"], top[job_id]["ExitCode"]) == status,
                f"diagnostic/probe Slurm state differs: {job_id}: {top[job_id]}")
    attempts_path = EXP / "diagnostic_attempts.json"
    attempts = json.loads(attempts_path.read_text())["attempts"]
    require(len(attempts) == len(expected) and {item["job_id"] for item in attempts} == set(expected), "diagnostic attempt ledger coverage differs")
    for item in attempts:
        require((item["state"], item["exit_code"]) == expected[item["job_id"]], "attempt ledger obscures a Slurm exit")
        if "archive" in item:
            verify_record({"path": str(Path(item["archive"]) / "provenance.json"), "sha256": item["provenance_sha256"]})
        for path_key in ("provenance", "evidence"):
            if path_key in item:
                verify_record({"path": item[path_key], "sha256": item[path_key + "_sha256"]})
    return {"successful_diagnostic_job_id": job, "successful_cpu_finalizer_job_id": job,
            "gpu_capture_job_id": "13982", "gpu_capture_state": "FAILED", "gpu_capture_exit_code": "1:0",
            "probe_job_ids": ["13980", "13983"], "failed_attempt_job_ids": ["13979", "13981"],
            "attempt_ledger_sha256": sha(attempts_path), "slurm": rows,
            "failed_attempt_provenance": failed_record, "failed_attempt_source": archived_source,
            "second_failed_attempt_provenance": HISTORICAL_CAPTURE_RECORD, "second_failed_attempt_source": historical_source}


def verify_diagnostics(diagnostics, arms, cases):
    """Check every saved diagnostic, not just the producer's summary counters."""
    directory = EXP / "diagnostics"
    runs = {"M2": ANCHOR_RUN, **{arm["id"]: RUNS / arm["run_name"] for arm in arms}}
    require(diagnostics["passed"] and diagnostics["smoke"] is False, "full diagnostic pass missing")
    require(set(diagnostics["runs"]) == set(runs), "diagnostic run set differs")
    require(diagnostics["case_count"] == len(cases) and set(diagnostics["geometry"]) == cases,
            "diagnostic geometry population differs")
    before = diagnostics["source_sha256_before"]
    expected_phases = {"base"} | ({"combination"} if any(a["phase"] == "combination" for a in arms) else set())
    require(set(before) == expected_phases and before == diagnostics["source_sha256_after"]
            == {phase: fingerprints(phase) for phase in before}, "diagnostic training sources changed")
    verify_record(diagnostics["script"], ROOT / "training_wss_min/tools/finalize_m2_diagnostic_capture.py")
    require(diagnostics["script"]["sha256"] == "18ced007107501bf021d220a74c7059491c2ae6641541178a0f8bdf68305f61f",
            "CPU finalizer implementation changed")
    verify_record(diagnostics["report_code"], Path(R.__file__))
    probe = verify_boundary_probe()
    batch = verify_batch_probe()
    capture = batch["capture"]
    verify_finalization(diagnostics, batch)
    anchor = C.ExpConfig.from_json(ANCHOR_CONFIG)
    expected_inputs = {str(Path(root) / case / filename) for case in cases
                       for root, filename in ((anchor.data.data_root, "bundle.npz"),
                                              (anchor.data.point_features_root, "features.npz"))}
    expected_inputs.update(probe["evidence_inputs"])
    require(set(capture["inputs"]) == expected_inputs
            and all(capture["inputs"].get(path) == digest for path, digest in probe["evidence_inputs"].items()),
            "GPU capture original/probe input boundary differs")
    require(diagnostics["inputs"] == batch["expected_final_inputs"], "CPU final diagnostic input boundary differs")
    for path, digest in diagnostics["inputs"].items():
        require(sha(path) == digest, f"diagnostic input changed: {path}")
    for case, geometry in diagnostics["geometry"].items():
        verify_record(geometry["arrays"], directory / "geometry" / f"{case.replace('/', '__')}.npz")
        original = capture["geometry"][case]
        require(geometry["arrays"]["sha256"] == original["arrays"]["sha256"]
                and {k: v for k, v in geometry.items() if k != "arrays"} == {k: v for k, v in original.items() if k != "arrays"},
                "CPU final geometry differs from the immutable capture")
    numeric_checks, evaluations, strict_passed_checks, strict_failed_checks = 0, 0, 0, 0
    boundary_reviews = []
    for label, run in runs.items():
        checkpoints = diagnostics["runs"][label]
        require(set(checkpoints) == {"best", "last"}, f"{label}: diagnostic checkpoints differ")
        for checkpoint, item in checkpoints.items():
            source_item = capture["runs"][label][checkpoint]
            require(item["passed"] and item["run_name"] == str(run.relative_to(RUNS))
                    and set(item["cases"]) == cases, f"{label}/{checkpoint}: diagnostic coverage differs")
            require({k: v for k, v in item.items() if k not in {"cases", "passed"}}
                    == {k: v for k, v in source_item.items() if k not in {"cases", "passed"}}, "CPU final run metadata differs from capture")
            expected_sources = {str(run / name) for name in (f"ckpt_{checkpoint}.pt", "config.json",
                "feature_stats.json", "wss_global_stats.json", f"eval/ckpt_{checkpoint}/metrics.json")}
            require(set(item["sources"]) == expected_sources, f"{label}/{checkpoint}: diagnostic sources differ")
            for path, digest in item["sources"].items():
                require(sha(path) == digest, f"diagnostic source changed: {path}")
            metric = json.loads((run / f"eval/ckpt_{checkpoint}/metrics.json").read_text())["test"]
            for case, record in item["cases"].items():
                source_case = source_item["cases"][case]
                require(record["wall_points"] == diagnostics["geometry"][case]["wall_points"]
                        == metric["per_case"][case]["overall"]["n"], f"{label}/{checkpoint}/{case}: wall count differs")
                checks = record["metric_verification"]
                require(set(checks) == {"Pa", "log_z"} and all(v["passed"] and v["checks"] > 0 for v in checks.values()),
                        f"{label}/{checkpoint}/{case}: numeric verification incomplete")
                numeric_checks += sum(v["checks"] for v in checks.values())
                verify_record(record["arrays"], directory / "predictions" / label / checkpoint / f"{case.replace('/', '__')}.npz")
                require(record["arrays"]["sha256"] == source_case["arrays"]["sha256"]
                        and {k: v for k, v in record.items() if k not in {"arrays", "metric_verification", "passed"}}
                        == {k: v for k, v in source_case.items() if k not in {"arrays", "metric_verification", "passed"}},
                        "CPU final case values/mechanisms/arrays differ from capture")
                for space, verification in checks.items():
                    original = source_case["metric_verification"][space]
                    require({k: v for k, v in verification.items() if k not in {"passed", "boundary_reviews"}}
                            == {k: v for k, v in original.items() if k not in {"passed", "boundary_reviews"}},
                            "CPU final verification changed original strict failures/tolerances")
                    if original["passed"]:
                        require(verification == original, "CPU final changed an already passing capture metric")
                    failed = len(verification["failed_checks"])
                    require(verification["strict_failed_checks"] == failed
                            and verification["strict_passed_checks"] == verification["checks"] - failed
                            and verification["strict_passed"] is (failed == 0), "diagnostic strict counters differ")
                    reviews = verification["boundary_reviews"]
                    require(len(reviews) == failed and failed <= 1, "diagnostic contains unreviewed strict failures")
                    for review in reviews:
                        require(checks["Pa"]["strict_passed"] is True, "Pa must pass strictly before log_z boundary review")
                        context = {"arm": label, "checkpoint": checkpoint, "case": case, "space": space}
                        if context == BOUNDARY_CONTEXT:
                            verify_boundary_review(review, verification, context, record["arrays"], diagnostics["geometry"][case], item["sources"], probe)
                        else:
                            verify_batch_review(review, verification, context, record["arrays"], source_case, item["sources"], batch)
                    strict_passed_checks += verification["strict_passed_checks"]
                    strict_failed_checks += failed
                    boundary_reviews.extend(reviews)
            evaluations += 1
    representatives = set(diagnostics["representative_cases_preselected"])
    figures = diagnostics["figures"]
    require(len(representatives) == 3 and representatives <= cases and set(figures["cases"]) == representatives
            and set(figures["overviews"]) == {"best", "last"}, "diagnostic example figures incomplete")
    figure_records = []
    for case in representatives:
        require(set(figures["cases"][case]["checkpoints"]) == {"best", "last"}, f"{case}: figure checkpoint missing")
        for checkpoint, records in figures["cases"][case]["checkpoints"].items():
            expected = {str(directory / "wall_figures" / f"{case.replace('/', '__')}__{checkpoint}__{kind}.{ext}")
                        for kind in ("wss", "error", "hotspot") for ext in ("png", "pdf")}
            require(len(records) == 6 and {r["path"] for r in records} == expected, f"{case}/{checkpoint}: figures differ")
            figure_records.extend(records)
    for checkpoint, records in figures["overviews"].items():
        expected = {str(directory / "wall_figures" / f"wall_overview_{checkpoint}.{ext}") for ext in ("png", "pdf")}
        require(len(records) == 2 and {r["path"] for r in records} == expected, f"{checkpoint}: overview figures differ")
        figure_records.extend(records)
    for record in figure_records:
        verify_record(record)
    require(evaluations == diagnostics["evaluations_verified"] == 2 * len(runs)
            and numeric_checks == diagnostics["original_per_case_numeric_checks"], "diagnostic summary counters differ")
    require(diagnostics["strict_passed"] is (strict_failed_checks == 0)
            and diagnostics["strict_passed_numeric_checks"] == strict_passed_checks
            and diagnostics["strict_failed_numeric_checks"] == strict_failed_checks
            and strict_passed_checks + strict_failed_checks == numeric_checks
            and diagnostics["boundary_review_count"] == len(boundary_reviews) == 2
            and diagnostics["boundary_reviews"] == boundary_reviews, "diagnostic boundary/strict summary differs")
    return {"sha256": sha(directory / "provenance.json"), "evaluations": evaluations,
            "numeric_checks": numeric_checks, "figures_verified": len(figure_records),
            "strict_passed": strict_failed_checks == 0, "strict_passed_numeric_checks": strict_passed_checks,
            "strict_failed_numeric_checks": strict_failed_checks, "boundary_review_count": len(boundary_reviews),
            "boundary_reviews": boundary_reviews, "threshold_probe": {"record": probe["record"],
                "evidence_inputs": probe["evidence_inputs"],
                "arrays_verified": 8, "recomputed_counts": probe["recomputed_counts"],
                "unique_crossing": probe["unique_crossing"], "maximum_abs_vs_first": probe["maximum_abs_vs_first"]},
            "batch_probe": {"record": BATCH_RECORD, "arrays_verified": batch["arrays_verified"],
                "scalar_checks_verified": batch["scalar_checks_verified"],
                "targets": [{"target_id": item["target"]["id"], "capture_origin": item["target"]["capture_origin"],
                    "proof": item["proof"]} for item in batch["targets"].values()]},
            "gpu_capture": CAPTURE_RECORD, "cpu_finalization": diagnostics["finalization"],
            "geometry_arrays_verified": len(cases), "prediction_arrays_verified": len(cases) * evaluations,
            **verify_diagnostic_jobs(diagnostics, probe)}


def verify_workbook_backup(workbook):
    from training_wss_min.tools import update_m2_optimization_xlsx as U

    # The backup is from the first update; before_sha256 describes the latest
    # update and legitimately differs after a repeated transcription.
    backup = U.load_workbook(workbook["backup"], data_only=False)
    current = U.load_workbook(workbook["book"], data_only=False)
    try:
        starts = {name: U.locate_section(backup[name]) for name in U.WIDTHS}
        excluded = {name: (start, start + U.RECT_HEIGHT - 1, U.WIDTHS[name])
                    for name, start in starts.items() if start is not None}
        preserved = U.snapshot_cells(backup, excluded)
        merges = {ws.title: [str(r) for r in ws.merged_cells.ranges] for ws in backup.worksheets}
        U.verify_preserved(current, preserved, merges)
        for name in U.WIDTHS:
            require(U.locate_section(current[name]) == workbook["positions"][name]["first_row"],
                    f"{name}: workbook result section differs")
        return {"backup_sha256": sha(workbook["backup"]), "backup_historical_cells_verified": len(preserved)}
    finally:
        current.close()
        backup.close()


def main():
    evidence = {"passed": False, "started_at": stamp(), "tool_sha256": sha(__file__)}
    try:
        reports = R.report_all(write=False)
        require(all(reports[ck]["matrix_finalized"] for ck in ("best", "last")),
                "all registered runs, evaluations and combination decisions must be finalized")
        arms = R.matrix_arms()
        require(20 <= len(arms) <= 24, "registered training budget exceeded")
        evidence["registered_trainings"] = len(arms)
        evidence["phases"] = {}
        jobs = set()
        for phase, filename in (("base", "queue_status.json"), ("combination", "combination_queue_status.json")):
            if phase == "combination" and not json.loads(COMBINATIONS.read_text())["arms"]:
                evidence["phases"][phase] = {"trainings": 0, "status": "not_required_by_frozen_gate"}
                continue
            state = json.loads((EXP / filename).read_text())
            require(state["status"] == "complete", f"{phase}: queue incomplete")
            require(state["source_sha256"] == state["source_sha256_end"] == fingerprints(phase),
                    f"{phase}: training sources changed")
            snapshot = Path(state["source_snapshot"])
            for relative, expected in state["source_sha256"].items():
                require(sha(snapshot / relative) == expected, f"source archive mismatch: {relative}")
            stages = 0
            for aid, arm in state["arms"].items():
                require(arm["status"] == "complete", f"{aid}: unfinished")
                for stage in ("train", "eval_best", "eval_last"):
                    record = arm["stages"][stage]
                    require(record["returncode"] == 0 and Path(record["log"]).is_file(), f"{aid}/{stage}: invalid exit/log")
                    jobs.add(str(record["job_id"]))
                    stages += 1
            evidence["phases"][phase] = {"trainings": len(state["arms"]), "successful_stages": stages,
                                         "source_files_verified": len(state["source_sha256"]), "queue_sha256": sha(EXP / filename)}
        output = subprocess.check_output([
            "/public/slurm/bin/sacct", "-j", ",".join(sorted(jobs)), "-P",
            "--format=JobID,JobName,State,ExitCode,Start,End,Elapsed,NodeList",
        ], text=True, timeout=60)
        rows = list(csv.DictReader(io.StringIO(output), delimiter="|"))
        top = {row["JobID"]: row for row in rows if row["JobID"] in jobs}
        require(set(top) == jobs, "missing Slurm accounting records")
        for job, row in top.items():
            require(row["State"] == "COMPLETED" and row["ExitCode"] == "0:0", f"worker {job}: {row}")
        evidence["slurm"] = rows

        inputs = json.loads((EXP / "data_provenance.json").read_text())
        boundary, test_cases = input_boundary(inputs, arms)
        failures, byte_count = [], 0
        for name, original in inputs["files"].items():
            path = Path(name)
            if not path.is_file() or path.stat().st_size != original["bytes"] or sha(path) != original["sha256"]:
                failures.append(name)
            byte_count += original["bytes"]
        evidence["data"] = {"files_checked": len(inputs["files"]), "bytes_checked": byte_count,
                            "changed_files": failures, "manifest_sha256": sha(EXP / "data_provenance.json"), **boundary}
        require(not failures, "original input files changed")

        evidence["runs"] = {}
        reference_stats = {name: json.loads((ANCHOR_RUN / name).read_text()) for name in (
            "feature_stats.json", "wss_global_stats.json", "target_normalization.json", "weight_quantiles.json")}
        for arm in arms:
            aid, directory = arm["id"], RUNS / arm["run_name"]
            init = json.loads((directory / "initialization.json").read_text())
            require(init["shared_tensors_exact"] and init["reference_rng_restored"] and not init["trained_checkpoint_loaded"],
                    f"{aid}: invalid fresh shared initialization")
            require(init["reference_config_sha256"] == inputs["anchor_config_sha256"], f"{aid}: initializer reference changed")
            diag = json.loads((directory / "training_diagnostics.json").read_text())
            require(diag["completed_epochs"] == 400 and diag["fatal_nonfinite_events"] == 0, f"{aid}: training invalid")
            for name, expected in reference_stats.items():
                require(json.loads((directory / name).read_text()) == expected, f"{aid}: statistics changed: {name}")
            artifacts = [directory / name for name in ("config.json", "history.jsonl", "ckpt_best.pt", "ckpt_last.pt",
                         "initialization.json", "training_diagnostics.json", *reference_stats)]
            artifacts += [directory / "eval" / f"ckpt_{ck}" / "metrics.json" for ck in ("best", "last")]
            evidence["runs"][aid] = {"artifacts": {str(p.relative_to(ROOT)): sha(p) for p in artifacts},
                "parameters": diag["parameter_count"], "peak_cuda_reserved_mb": diag["peak_cuda_reserved_mb"],
                "elapsed_seconds": diag["elapsed_seconds"], "amp_overflow_steps": diag["amp_overflow_steps"],
                "cases_each_checkpoint": 34, "points_each_checkpoint": 1231295}

        diagnostics = json.loads((EXP / "diagnostics/provenance.json").read_text())
        evidence["diagnostics"] = verify_diagnostics(diagnostics, arms, test_cases)
        workbook = json.loads((EXP / "workbook_update.json").read_text())
        require(workbook["all_historical_cells_and_formulas_preserved"] and workbook["matrix_finalized"], "workbook audit failed")
        require(sha(workbook["book"]) == workbook["after_sha256"], "workbook changed after transcription audit")
        require(Path(workbook["backup"]).is_file(), "workbook backup missing")
        require(workbook["completed_best"] == workbook["completed_last"] == len(arms)
                and workbook["row_deletion"] is False, "workbook result coverage differs")
        evidence["workbook"] = {**workbook, **verify_workbook_backup(workbook)}
        tracker = ROOT / "docs/02-推进与变更/WSS_PINN/WSS_V5_训练实验跟踪.md"
        content = tracker.read_text()
        heading = "## 16. M2优化：20项完整筛选与组合结论（2026-09-09–10）"
        require(content.count(heading + "\n") == 1, "tracker section missing/duplicated")
        section = get_section(content, heading)
        require(all(sub in section for sub in ("### 16.1 全部best/last结果", "### 16.2 筛选结论", "### 16.3 同期对照与验收")),
                "tracker section truncated")
        evidence["documents"] = {str(p.relative_to(ROOT)): sha(p) for p in (tracker, EXP / "README.md")}
        evidence.update(passed=True, completed_at=stamp(),
            interpretation="单seed、已暴露test34的探索；工程验收不代表统计稳定提升。同期对照偏差见专门审计。")
    except Exception as exc:
        evidence.update(error=f"{type(exc).__name__}: {exc}", failed_at=stamp())
        raise
    finally:
        save_json(EXP / "final_acceptance.json", evidence)
    print(json.dumps({key: evidence[key] for key in ("passed", "registered_trainings", "data", "diagnostics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
