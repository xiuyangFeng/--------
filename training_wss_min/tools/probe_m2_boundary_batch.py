"""Repeated raw inference evidence for collected one-vertex threshold discrepancies.

This tool never accepts a metric discrepancy. Completed probes still require
independent review; missing threshold sides and multi-vertex changes stay visible.
GPU inference requires an existing Slurm allocation. --list-targets and --self-test
are CPU-only and do not submit jobs or modify the collector/formal run artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from training_wss_min import config as C, dataset as D, evaluate as E
from training_wss_min.experiments.v6_multiradius_bt_20260909 import plot_paired_wall_fields as V
from training_wss_min.tools.m2_optimization_common import EXP, ROOT, RUNS, fingerprints, save_json, sha, stamp


METRIC = "distribution.pred_above_one_fraction"
BELOW_ZERO = "distribution.pred_below_zero_fraction"
REPEATS = 8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify_record(item):
    require(sha(item["path"]) == item["sha256"], f"artifact changed: {item['path']}")


def target_id(context):
    return ":".join(context[key] for key in ("arm", "checkpoint", "case", "space"))


def integer_count(fraction, n):
    if type(n) is not int or n <= 0 or not isinstance(fraction, (int, float)) or not np.isfinite(fraction):
        return None
    count = round(fraction * n)
    return count if 0 <= count <= n and abs(fraction * n - count) <= 1e-8 else None


def select_targets(collector, requested=(), *, require_complete=True):
    require(collector.get("smoke") is False, "smoke capture is not admissible probe evidence")
    if require_complete:
        require(collector.get("completed") is True
            and collector.get("full_inference_coverage_complete") is True
            and collector.get("evaluations_completed") == 42
            and collector.get("cases_evaluated") == collector.get("expected_case_evaluations") == 1428
            and collector.get("case_count") == 34, "collector must finish all 42 x 34 inferences")
        require(len(collector["runs"]) == 21 and all(set(run) == {"best", "last"} for run in collector["runs"].values()),
            "collector registered run/checkpoint coverage differs")
    eligible, unresolved, failures = [], [], []
    for arm, run in collector["runs"].items():
        for checkpoint, item in run.items():
            if require_complete:
                require(item["completed"] is True and len(item["cases"]) == 34, "collector case coverage differs")
            for case, values in item["cases"].items():
                for space, check in values["metric_verification"].items():
                    reviewed = {review["metric"] for review in check["boundary_reviews"]}
                    for failure in check["failed_checks"]:
                        if failure["metric"] in reviewed:
                            continue
                        context = {"arm": arm, "checkpoint": checkpoint, "case": case, "space": space}
                        n = values["wall_points"]
                        target = {"id": target_id(context) + ":" + failure["metric"].split(".")[-1], "context": context, "metric": failure["metric"], "n": n,
                                  "original_failure": failure, "collector_arrays": values["arrays"]}
                        failures.append({"context": context, "metric": failure["metric"]})
                        reference_count = integer_count(failure["reference"], n)
                        observed_count = integer_count(failure["recomputed"], n)
                        if (failure["metric"] in {METRIC, BELOW_ZERO} and space in {"Pa", "log_z"}
                                and (failure["metric"] != BELOW_ZERO or space == "log_z") and type(n) is int and n > 0
                                and reference_count is not None and observed_count is not None
                                and abs(observed_count - reference_count) == 1):
                            target.update(reference_count=reference_count, collector_count=observed_count,
                                          count_delta=observed_count - reference_count)
                            eligible.append(target)
                        else:
                            target["reason"] = "not_a_supported_single_vertex_threshold_fraction_failure"
                            unresolved.append(target)
    if require_complete or "unreviewed_metric_failures" in collector:
        ledger = [{"context": row["context"], "metric": row["metric"]} for row in collector["unreviewed_metric_failures"]]
        key = lambda row: (target_id(row["context"]), row["metric"])
        require(sorted(failures, key=key) == sorted(ledger, key=key)
                and len(failures) == collector["unreviewed_metric_failure_count"], "collector failure ledger differs")
    require(len({target["id"] for target in eligible}) == len(eligible), "duplicate eligible target")
    requested = set(requested)
    require(not requested or requested <= {target["id"] for target in eligible}, "requested target is absent/ineligible")
    return {"eligible_targets": eligible, "selected_targets": [target for target in eligible if not requested or target["id"] in requested],
            "unresolved_noneligible_failures": unresolved, "unreviewed_failure_count": len(failures),
            "status": "evidence_collection_only_no_metric_accepted"}


def capture_artifact(item, provenance_path):
    """Resolve a relocated failed attempt without editing its original provenance."""
    path = Path(item["path"])
    archive = provenance_path.parent
    if archive.name.startswith("diagnostics_failed_") and not path.is_relative_to(archive):
        original_roots = [root for root in (EXP / "diagnostics", EXP / "diagnostics_collection") if path.is_relative_to(root)]
        require(len(original_roots) == 1, "unknown relocated capture root")
        path = archive / path.relative_to(original_roots[0])
    resolved = {"path": str(path.resolve()), "sha256": item["sha256"]}
    verify_record(resolved)
    return resolved


def prepare_selection(collector_path, previous_paths=(), requested=()):
    eligible, unresolved, captures = [], [], []
    for index, path in enumerate((collector_path, *previous_paths)):
        path = Path(path).resolve()
        capture = json.loads(path.read_text())
        selection = select_targets(capture, require_complete=index == 0)
        provenance = record(path)
        captures.append({"provenance": provenance, "job_id": str(capture["job_id"]), "primary_complete_capture": index == 0})
        for target in selection["eligible_targets"]:
            c = target["context"]
            source = capture["runs"][c["arm"]][c["checkpoint"]]
            target.update(id=target["id"] + f"@job{capture['job_id']}", capture_provenance=provenance,
                          capture_job_id=str(capture["job_id"]), run_name=source["run_name"], sources=source["sources"],
                          collector_arrays=capture_artifact(target["collector_arrays"], path),
                          geometry_arrays=capture_artifact(capture["geometry"][c["case"]]["arrays"], path))
            target["capture_origin"] = {"job_id": str(capture["job_id"]), "provenance": provenance,
                                        "arrays": target["collector_arrays"], "primary_complete_capture": index == 0}
            eligible.append(target)
        unresolved.extend({**target, "capture_provenance": provenance, "capture_job_id": str(capture["job_id"])}
                          for target in selection["unresolved_noneligible_failures"])
    require(len({target["id"] for target in eligible}) == len(eligible), "duplicate capture target")
    requested = set(requested)
    available = {key for target in eligible for key in (target["id"], target_id(target["context"]))}
    require(requested <= available, "requested capture target is absent/ineligible")
    chosen = [target for target in eligible if not requested or {target["id"], target_id(target["context"])} & requested]
    return {"eligible_targets": eligible, "selected_targets": chosen, "captures": captures,
            "unresolved_noneligible_failures": unresolved, "unreviewed_failure_count": len(eligible) + len(unresolved),
            "status": "evidence_collection_only_no_metric_accepted"}


def metric_prediction(norm, stats, space):
    return norm if space == "log_z" else np.clip(D.denormalize_wss(norm, stats), 0., None)


def threshold_norm(stats, space, metric=METRIC):
    require(stats["method"] == "log_z", "probe requires the frozen log_z normalization")
    if metric == BELOW_ZERO:
        require(space == "log_z", "physical predictions are clipped at zero and cannot be strictly negative")
        return 0.
    return 1. if space == "log_z" else float((np.log(1. + stats["eps"]) - stats["log"]["mean"]) / stats["log"]["std"])


def threshold_sides(prediction, metric):
    require(metric in {METRIC, BELOW_ZERO}, "unsupported threshold metric")
    return prediction < 0. if metric == BELOW_ZERO else prediction > 1.


def measure(norm, stats, space, node_ids, against, metric=METRIC):
    prediction = metric_prediction(norm, stats, space)
    baseline = metric_prediction(against, stats, space)
    threshold = 0. if metric == BELOW_ZERO else 1.
    require(norm.shape == against.shape == node_ids.shape and norm.ndim == 1
            and np.isfinite(norm).all() and np.isfinite(prediction).all(), "invalid probe raw population")
    crossing = np.flatnonzero(threshold_sides(prediction, metric) != threshold_sides(baseline, metric)).tolist()
    nearest = []
    for i in np.argsort(np.abs(prediction - threshold))[:12]:
        value = np.float32(norm[i])
        neighbors = np.asarray([np.nextafter(value, np.float32(-np.inf)), value,
                                np.nextafter(value, np.float32(np.inf))], dtype=np.float64)
        nearest.append({"index": int(i), "wall_node_id_cas": int(node_ids[i]), "prediction": float(prediction[i]),
                        "distance_to_metric_threshold": float(prediction[i] - threshold), "prediction_norm": float(norm[i]),
                        "distance_to_norm_threshold": float(norm[i] - threshold_norm(stats, space, metric)),
                        "norm_abs_float32_ulp": float(abs(np.spacing(value))),
                        "neighbor_norm_values": neighbors.tolist(),
                        "neighbor_metric_values": metric_prediction(neighbors, stats, space).tolist()})
    count = int(np.count_nonzero(threshold_sides(prediction, metric)))
    return {"n": len(norm), "count": count, "fraction": count / len(norm), "metric": metric,
            "metric_threshold": threshold, "comparison": "lt" if metric == BELOW_ZERO else "gt",
            "norm_threshold": threshold_norm(stats, space, metric), "crossing_indices_vs_collector": crossing,
            "crossings_vs_collector": [{"index": i, "wall_node_id_cas": int(node_ids[i]),
                "collector_norm": float(against[i]), "current_norm": float(norm[i]),
                "collector_metric_value": float(baseline[i]), "current_metric_value": float(prediction[i])} for i in crossing],
            "max_abs_norm_vs_collector": float(np.max(np.abs(norm - against))),
            "max_abs_metric_vs_collector": float(np.max(np.abs(prediction - baseline))), "nearest_to_threshold": nearest}


def summarize_target(target, repeats, collected, stats):
    """Describe replay evidence; even the positive case is not an acceptance decision."""
    space = target["context"]["space"]
    metric = target.get("metric", METRIC)
    all_norm = [collected, *repeats]
    counts = [int(np.count_nonzero(threshold_sides(metric_prediction(norm, stats, space), metric))) for norm in all_norm]
    reference_indices = [i for i, count in enumerate(counts) if count == target["reference_count"]]
    union = set()
    adjacent, vertex_values, vertex_sequence, maxima = False, [], [], []
    if reference_indices:
        reference_norm = all_norm[reference_indices[0]]
        reference = metric_prediction(reference_norm, stats, space)
        for norm in all_norm:
            union.update(np.flatnonzero(threshold_sides(metric_prediction(norm, stats, space), metric) != threshold_sides(reference, metric)).tolist())
            maxima.append(float(np.max(np.abs(norm - reference_norm))))
        if len(union) == 1:
            vertex = next(iter(union))
            vertex_values = sorted({float(norm[vertex]) for norm in all_norm})
            vertex_sequence = [float(norm[vertex]) for norm in all_norm]
            adjacent = (len(vertex_values) == 2 and all(float(np.float32(v)) == v for v in vertex_values)
                        and float(np.nextafter(np.float32(vertex_values[0]), np.float32(np.inf))) == vertex_values[1])
    observed = (bool(reference_indices) and len(set(counts)) == 2
                and set(counts) == {target["reference_count"], target["collector_count"]} and len(union) == 1)
    return {"status": "independent_review_required", "metric_accepted": False,
            "counts_including_collector_first": counts, "reference_count_replayed": bool(reference_indices),
            "reference_array_index_including_collector": reference_indices[0] if reference_indices else None,
            "crossing_union_vs_replayed_reference": sorted(union), "single_vertex_replay_observed": observed,
            "crossing_norm_values": vertex_values, "crossing_is_two_adjacent_float32_norm_values": bool(adjacent),
            "crossing_norm_sequence_including_collector": vertex_sequence,
            "crossing_norm_observed_range": [min(vertex_values), max(vertex_values)] if vertex_values else None,
            "max_abs_norm_vs_replayed_reference_including_collector": maxima,
            "zero_threshold_uses_observed_signed_values_not_a_one_ulp_rule": metric == BELOW_ZERO,
            "limit": "Original formal-evaluation raw predictions were not saved; replay evidence cannot reconstruct them. No tolerance or formal metric is changed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, default=EXP / "diagnostics_collection/provenance.json")
    parser.add_argument("--target", action="append", default=[], help="optional arm:checkpoint:case:space; repeat to select multiple")
    parser.add_argument("--include-attempt", type=Path, action="append", default=[],
                        help="additional archived capture; failed 13981 is always included for its saved P2 boundary")
    parser.add_argument("--repeats", type=int, choices=(8, 32), default=REPEATS,
                        help="eight by default; a separate targeted 32-repeat job may collect an unresolved threshold side")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    collector_record = record(args.collector)
    collector = json.loads(args.collector.read_text())
    previous_paths = [EXP / "diagnostics_failed_13981/provenance.json", *args.include_attempt]
    selection = prepare_selection(args.collector, previous_paths, args.target)
    if args.list_targets:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return
    require(bool(selection["selected_targets"]), "no eligible targets; no GPU work is needed")
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "probe needs an existing single-GPU Slurm allocation; it does not submit jobs")
    torch.set_num_threads(2)
    while torch.cuda.mem_get_info()[0] / 2**20 < 7000:
        time.sleep(15)
    output = EXP / "threshold_batch_probe" / f"job_{os.environ['SLURM_JOB_ID']}"
    output.mkdir(parents=True, exist_ok=False)
    evidence = {"schema_version": 1, "completed": False, "metric_accepted": False,
                "status": "collecting_evidence", "started_at": stamp(), "job_id": os.environ["SLURM_JOB_ID"],
                "script": record(__file__), "metric_verification_script": record(V.__file__),
                "collector": collector_record, "selection": selection, "repeats_per_case": args.repeats,
                "inputs": dict(collector["inputs"]), "source_sha256_before": collector["source_sha256_before"], "groups": {}}
    evidence["inputs"][collector_record["path"]] = collector_record["sha256"]
    for capture in selection["captures"]:
        item = capture["provenance"]
        evidence["inputs"][item["path"]] = item["sha256"]
    grouped = defaultdict(list)
    for target in selection["selected_targets"]:
        c = target["context"]
        grouped[(c["arm"], c["checkpoint"], c["case"])].append(target)
    try:
        require(evidence["source_sha256_before"] == collector["source_sha256_after"]
                == {phase: fingerprints(phase) for phase in evidence["source_sha256_before"]}, "frozen training source mismatch")
        for path, digest in evidence["inputs"].items():
            require(sha(path) == digest, f"collector input changed: {path}")
        for (arm, checkpoint_name, uid), targets in grouped.items():
            source = targets[0]
            run = RUNS / source["run_name"]
            require(run.resolve().is_relative_to(RUNS.resolve()), "run escapes registered root")
            for path, digest in source["sources"].items():
                require(sha(path) == digest, f"run source changed: {path}")
                evidence["inputs"][path] = digest
            geometry_record = source["geometry_arrays"]
            with np.load(geometry_record["path"], allow_pickle=False) as payload:
                truth_norm, truth_pa = payload["truth_norm"].copy(), payload["truth_pa"].copy()
                node_ids = payload["wall_node_id_cas"].copy()
            captures = {}
            for target in targets:
                require(target["sources"] == source["sources"] and target["run_name"] == source["run_name"],
                        "same-case capture uses different run/checkpoint/config sources")
                for item in (target["geometry_arrays"], target["collector_arrays"]):
                    verify_record(item)
                    evidence["inputs"][item["path"]] = item["sha256"]
                with np.load(target["collector_arrays"]["path"], allow_pickle=False) as payload:
                    captures[target["id"]] = (payload["prediction_norm"].copy(), payload["prediction_pa"].copy())
                with np.load(target["geometry_arrays"]["path"], allow_pickle=False) as payload:
                    require(np.array_equal(truth_norm, payload["truth_norm"]) and np.array_equal(truth_pa, payload["truth_pa"])
                            and np.array_equal(node_ids, payload["wall_node_id_cas"]), "capture geometry/truth/order differs")
            cfg, feature_stats, model, checkpoint = E.load_model_from_run(run, "cuda", checkpoint_name)
            stats = E.load_wss_stats_for_run(run)
            require((cfg.data.target, cfg.data.timesteps, cfg.data.target_normalization, cfg.model.out_dim,
                     cfg.eval.fixed_support, cfg.eval.full_cloud_query, cfg.eval.support_seed,
                     cfg.eval.query_chunk_size, cfg.eval.surface_metric_mode)
                    == ("wss", "peak", "global_stats", 1, True, True, 1234, 16384, "legacy_vertex")
                    and cfg.eval.pointwise_jensen is False, "evaluation protocol changed")
            cohort, case_name = uid.rsplit("/", 1)
            case = D.load_case(cohort, case_name, stats, target=cfg.data.target,
                target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                required_frame_version=cfg.data.required_frame_version, timesteps=cfg.data.timesteps,
                extra_point_features=C.v6_point_features(cfg), point_features_root=cfg.data.point_features_root)
            require(np.array_equal(case["y_norm"], truth_norm) and np.array_equal(case["y_raw"], truth_pa)
                    and np.isfinite(truth_norm).all() and np.isfinite(truth_pa).all(), "raw truth/order differs")
            for target in targets:
                captured_norm, captured_pa = captures[target["id"]]
                require(np.array_equal(metric_prediction(captured_norm, stats, "Pa"), captured_pa)
                        and len(captured_norm) == target["n"]
                        and np.count_nonzero(threshold_sides(metric_prediction(captured_norm, stats, target["context"]["space"]), target["metric"])) == target["collector_count"],
                        "captured raw count/Pa mapping differs from failure record")
            saved = json.loads((run / f"eval/ckpt_{checkpoint_name}/metrics.json").read_text())["test"]
            for target in targets:
                original = (saved if target["context"]["space"] == "Pa" else saved["normalized"])["per_case"][uid]["distribution"]
                require(original["n"] == target["n"]
                        and original[target["metric"].split(".")[-1]] == target["original_failure"]["reference"]
                        and integer_count(original[target["metric"].split(".")[-1]], original["n"]) == target["reference_count"],
                        "target reference differs from the untouched formal metric")
            key = ":".join((arm, checkpoint_name, uid))
            group = {"arm": arm, "checkpoint": checkpoint_name, "case": uid, "run_name": source["run_name"],
                     "checkpoint_epoch": int(checkpoint["epoch"]), "sources": source["sources"],
                     "geometry_arrays": geometry_record,
                     "wss_stats": stats, "targets": targets, "repeats": [], "summaries": {}, "completed": False}
            evidence["groups"][key] = group
            repeat_arrays = []
            for index in range(args.repeats):
                result = E._evaluate_partition_frame(model, [case], cfg, feature_stats, stats, "cuda", return_predictions=True)
                norm = np.asarray(result["_pred_norm_by_case"][0], dtype=np.float64)
                pa = metric_prediction(norm, stats, "Pa")
                require(norm.shape == truth_norm.shape == node_ids.shape and np.isfinite(norm).all() and np.isfinite(pa).all(),
                        "probe prediction population changed")
                array_path = output / arm / checkpoint_name / uid.replace("/", "__") / f"repeat_{index}.npz"
                array_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(array_path, prediction_norm=norm, prediction_pa=pa,
                                    truth_norm=truth_norm, truth_pa=truth_pa, wall_node_id_cas=node_ids)
                checks = {space: V.verify_metrics(actual, expected) for space, actual, expected in (
                    ("Pa", result["per_case"][uid], saved["per_case"][uid]),
                    ("log_z", result["normalized"]["per_case"][uid], saved["normalized"]["per_case"][uid]))}
                measurements = {target["id"]: measure(norm, stats, target["context"]["space"], node_ids, captures[target["id"]][0], target["metric"]) for target in targets}
                group["repeats"].append({"index": index, "arrays": record(array_path), "measurements": measurements,
                                         "metric_verification": checks})
                repeat_arrays.append(norm.copy())
                save_json(output / "probe.json", evidence)
            for target in targets:
                group["summaries"][target["id"]] = summarize_target(target, repeat_arrays, captures[target["id"]][0], stats)
            group["completed"] = True
            save_json(output / "probe.json", evidence)
            print(f"collected {key}: {args.repeats} repeats; independent review required", flush=True)
            del model, checkpoint, case, repeat_arrays
            gc.collect()
            torch.cuda.empty_cache()
        require(all(sha(path) == digest for path, digest in evidence["inputs"].items()), "probe inputs changed during inference")
        require(sha(__file__) == evidence["script"]["sha256"] and sha(V.__file__) == evidence["metric_verification_script"]["sha256"],
                "probe/verification implementation changed during inference")
        evidence["source_sha256_after"] = {phase: fingerprints(phase) for phase in evidence["source_sha256_before"]}
        require(evidence["source_sha256_after"] == evidence["source_sha256_before"], "frozen training sources changed")
        evidence.update(completed=True, completed_at=stamp(), status="independent_review_required",
                        inference_count=args.repeats * len(grouped), target_count=len(selection["selected_targets"]))
    except Exception as exc:
        evidence.update(error=f"{type(exc).__name__}: {exc}", failed_at=stamp(), status="evidence_collection_failed")
        raise
    finally:
        save_json(output / "probe.json", evidence)
    print(f"Saved {output / 'probe.json'}; no metric accepted", flush=True)


def self_test():
    stats = {"method": "log_z", "eps": 1e-6, "log": {"mean": .6408381995319768, "std": 1.2999484463172573}}
    upper = float(np.float32(threshold_norm(stats, "Pa")))
    lower = float(np.nextafter(np.float32(upper), np.float32(-np.inf)))
    collected = np.asarray([upper, 0., -1.])
    reference = np.asarray([lower, 0., -1.])
    target = {"context": {"space": "Pa"}, "reference_count": 1, "collector_count": 2}
    good = summarize_target(target, [reference, collected] * 4, collected, stats)
    require(good["single_vertex_replay_observed"] and good["crossing_is_two_adjacent_float32_norm_values"]
            and good["metric_accepted"] is False, "Pa boundary evidence semantics failed")
    missing = summarize_target(target, [collected] * 8, collected, stats)
    require(not missing["single_vertex_replay_observed"] and not missing["reference_count_replayed"], "missing side was accepted")
    two = reference.copy(); two[1] = -1.
    multiple = summarize_target(target, [reference, two] * 4, collected, stats)
    require(not multiple["single_vertex_replay_observed"], "multi-vertex replay was accepted")
    measurement = measure(collected, stats, "Pa", np.arange(3), reference)
    require(measurement["count"] == 2 and measurement["crossing_indices_vs_collector"] == [0]
            and measurement["nearest_to_threshold"][0]["neighbor_metric_values"][0] < 1., "raw threshold measurement failed")
    require(integer_count(.5, 3) is None and integer_count(2 / 3, 3) == 2, "noninteger fraction accepted")
    signed = {"context": {"space": "log_z"}, "metric": BELOW_ZERO, "reference_count": 1, "collector_count": 0}
    positive, negative = np.asarray([2.6e-8, 1., 2.]), np.asarray([-3e-8, 1., 2.])
    zero = summarize_target(signed, [negative, positive] * 4, positive, stats)
    require(zero["single_vertex_replay_observed"] and not zero["crossing_is_two_adjacent_float32_norm_values"]
            and zero["zero_threshold_uses_observed_signed_values_not_a_one_ulp_rule"] and not zero["metric_accepted"],
            "near-zero signed evidence incorrectly used a one-ULP acceptance rule")
    require(measure(negative, stats, "log_z", np.arange(3), positive, BELOW_ZERO)["count"] == 1,
            "strict below-zero count failed")
    print("CPU probe tests passed: Pa/norm mapping, strict >1 and <0, signed near-zero evidence, missing-side and multi-vertex rejection; no automatic acceptance")


if __name__ == "__main__":
    main()
