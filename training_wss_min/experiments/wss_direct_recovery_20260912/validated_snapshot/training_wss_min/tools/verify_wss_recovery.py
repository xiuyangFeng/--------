"""Independent acceptance of complete WSS matrix training and saved predictions.

No production dataset, evaluator, or metric helper is imported. Labels come
directly from source bundles; NumPy sums independently reconstruct the scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class Audit:
    def __init__(self):
        self.checks = []
        self.files = {}

    def require(self, name, condition, **evidence):
        self.checks.append(dict(check=name, passed=bool(condition), **clean(evidence)))
        if not condition:
            raise ValueError(f"acceptance failed: {name}: {clean(evidence)}")

    def number(self, name, value, expected, atol=1e-8, rtol=1e-8):
        valid = (np.isfinite(value) and expected is not None and np.isfinite(expected)
                 and np.isclose(value, expected, atol=atol, rtol=rtol))
        self.require(name, valid, actual=value, expected=expected, atol=atol, rtol=rtol)

    def array(self, name, value, expected, atol=0.0, rtol=0.0):
        value, expected = np.asarray(value), np.asarray(expected)
        self.require(name, value.shape == expected.shape and
                     np.allclose(value, expected, atol=atol, rtol=rtol, equal_nan=False),
                     shape=list(value.shape), expected_shape=list(expected.shape),
                     atol=atol, rtol=rtol)

    def file(self, path, expected=None):
        path = Path(path).resolve()
        self.require(f"file_exists:{path}", path.is_file())
        key = str(path)
        if key not in self.files:
            self.files[key] = sha256(path)
        if expected is not None:
            self.require(f"sha256:{path}", self.files[key] == expected,
                         actual=self.files[key], expected=expected)
        return self.files[key]


def basic(truth, prediction):
    y, p = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    error = p - y
    squared = np.sum(error * error, dtype=np.float64)
    spread = np.sum((y - y.mean()) ** 2, dtype=np.float64)
    return {"r2": float(1 - squared / spread) if spread >= 1e-12 else float("nan"),
            "mae": float(np.sum(np.abs(error)) / len(y)),
            "rmse": float(np.sqrt(squared / len(y))), "n": len(y)}


def summarize(truths, predictions, names):
    """Independent pooled, equal-case and case-distribution estimators."""
    truths = [np.asarray(y, np.float64) for y in truths]
    predictions = [np.asarray(p, np.float64) for p in predictions]
    center = sum(float(y.sum()) / len(y) for y in truths) / len(truths)
    errors = [(p - y) for y, p in zip(truths, predictions)]
    mse = sum(float(np.dot(e, e)) / len(e) for e in errors) / len(errors)
    spread = sum(float(np.sum((y - center) ** 2)) / len(y) for y in truths) / len(truths)
    cb = {"r2": 1 - mse / spread if spread > 1e-12 else float("nan"),
          "mae": sum(float(np.abs(e).sum()) / len(e) for e in errors) / len(errors),
          "rmse": math.sqrt(mse), "n_cases": len(truths), "n": sum(map(len, truths))}
    per_case = {}
    for name, y, p in zip(names, truths, predictions):
        yt, pt = y >= np.quantile(y, .9), p >= np.quantile(p, .9)
        per_case[name] = {**basic(y, p), "top10_iou": float((yt & pt).sum() / (yt | pt).sum())}
    r2s = np.asarray([case["r2"] for case in per_case.values()])
    if not np.isfinite(r2s).all():
        raise ValueError("undefined per-case R2; complete-case acceptance cannot silently drop cases")
    aggregate = {"r2_casemean": float(r2s.mean()), "r2_casemed": float(np.median(r2s)),
                 "r2_casep10": float(np.quantile(r2s, .1)),
                 "r2_negative_cases": int((r2s < 0).sum()), "n_cases": len(r2s)}
    return {"field": basic(np.concatenate(truths), np.concatenate(predictions)),
            "field_casebalanced": cb, "aggregate": aggregate, "per_case": per_case,
            "top10_iou": float(np.mean([c["top10_iou"] for c in per_case.values()])),
            "negative_cases": [name for name, c in per_case.items() if c["r2"] < 0]}


def _finite_numbers(value):
    if isinstance(value, dict):
        return all(_finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(v) for v in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


def check_training(audit, run, cfg, epochs):
    import torch

    audit.require("configured_epochs", cfg["train"]["epochs"] == epochs)
    audit.require("configured_seed", cfg["train"]["seed"] == 1234)
    history_path = run / "history.jsonl"
    audit.file(history_path)
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    audit.require("continuous_complete_history", [row["epoch"] for row in history] == list(range(epochs)))
    for row in history:
        epoch = row["epoch"]
        audit.require(f"history_finite:{epoch}", _finite_numbers(row))
        for key in ("train_loss", "train_mse_norm", "train_mae_norm", "train_rmse_norm", "lr"):
            audit.require(f"history_required_finite:{epoch}:{key}",
                          isinstance(row.get(key), (int, float)) and math.isfinite(row[key]))
    best_row = max(history, key=lambda row: row.get("selection_score", -row["train_loss"]))
    checkpoints = {}
    for which in ("best", "last"):
        path = run / f"ckpt_{which}.pt"
        audit.file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected_epoch = best_row["epoch"] if which == "best" else epochs - 1
        audit.require(f"checkpoint_epoch:{which}", checkpoint["epoch"] == expected_epoch,
                      actual=checkpoint["epoch"], expected=expected_epoch)
        audit.require(f"checkpoint_model_present:{which}", bool(checkpoint.get("model")))
        audit.require(f"checkpoint_model_finite:{which}",
                      all(bool(torch.isfinite(tensor).all()) for tensor in checkpoint["model"].values()
                          if torch.is_tensor(tensor)))
        if which == "best" and cfg["train"].get("selection_rule") == "train_loss":
            audit.number("best_is_minimum_training_loss", best_row["train_loss"],
                         min(row["train_loss"] for row in history))
        checkpoints[which] = {"epoch": checkpoint["epoch"], "sha256": audit.files[str(path.resolve())]}
    return {"epochs": epochs, "epoch_indexing": "zero-based", "checkpoints": checkpoints}


def load_source(audit, data, stats, unit_id, cache, frozen_data=None):
    if unit_id in cache:
        return cache[unit_id]
    path = Path(data["data_root"]) / unit_id / "bundle.npz"
    expected_hash = None
    if frozen_data is not None:
        frozen_record = frozen_data.get(str(path.resolve()), {})
        audit.require(f"source_in_preflight_freeze:{unit_id}", bool(frozen_record.get("sha256")))
        expected_hash = frozen_record["sha256"]
    audit.file(path, expected_hash)
    with np.load(path, allow_pickle=False) as bundle:
        peak = int(bundle["peak_step"])
        audit.require(f"source_peak1162:{unit_id}", peak == 1162)
        steps = list(np.asarray(bundle["steps"]).astype(int))
        audit.require(f"source_peak_unique:{unit_id}", steps.count(peak) == 1)
        raw = np.asarray(bundle["wall_wss"][steps.index(peak)], np.float32)
        audit.require(f"source_wall_rows:{unit_id}", raw.ndim == 1 and len(raw) > 1
                      and len(bundle["wall_coords_norm"]) == len(raw))
        if data.get("required_frame_version"):
            audit.require(f"source_frame:{unit_id}",
                          str(bundle["transform_frame_version"].item()) == data["required_frame_version"])
    audit.require(f"source_labels_finite:{unit_id}", np.isfinite(raw).all())
    # Explicit float32 operations reproduce the documented stored target precision.
    shifted = np.maximum(raw, np.float32(stats.get("floor", 0.0))) + np.float32(stats["eps"])
    norm = (np.log(shifted) - np.float32(stats["log"]["mean"])) / np.float32(stats["log"]["std"])
    cache[unit_id] = (raw, norm)
    return raw, norm


def compare_metrics(audit, prefix, recomputed, saved):
    for group in ("field", "field_casebalanced", "aggregate"):
        for key, value in recomputed[group].items():
            audit.number(f"{prefix}.{group}.{key}", value, saved[group][key])
    audit.number(f"{prefix}.hotspot.top10_iou", recomputed["top10_iou"], saved["hotspot"]["top10_iou"])
    audit.require(f"{prefix}.per_case_ids", set(recomputed["per_case"]) == set(saved["per_case"]))
    for case, metrics in recomputed["per_case"].items():
        for key in ("r2", "mae", "rmse", "n"):
            audit.number(f"{prefix}.{case}.{key}", metrics[key], saved["per_case"][case]["overall"][key])
        audit.number(f"{prefix}.{case}.top10_iou", metrics["top10_iou"],
                     saved["per_case"][case]["hotspot"]["top10_iou"])


def check_evaluation(audit, run, which, cfg, stats, expected_cases, source_cache, frozen_data=None):
    directory = run / "eval" / f"ckpt_{which}"
    metrics_path = directory / "metrics.json"
    audit.file(metrics_path)
    metrics = read_json(metrics_path)["test"]
    root = directory / "predictions"
    audit.file(root / "manifest.json")
    top_manifest = read_json(root / "manifest.json")
    audit.require(f"manifest_checkpoint:{which}", top_manifest["checkpoint"] == which)
    audit.require(f"manifest_target:{which}", top_manifest["target"] == "wss")
    for key, source in (("checkpoint_sha256", run / f"ckpt_{which}.pt"),
                        ("config_sha256", run / "config.json"),
                        ("wss_stats_sha256", run / "wss_global_stats.json")):
        audit.require(f"prediction_provenance:{which}:{key}",
                      top_manifest.get(key) == audit.file(source),
                      expected=audit.files[str(source.resolve())], actual=top_manifest.get(key))
    expected_split = str(Path(cfg["data"]["split_path"]).resolve())
    audit.require(f"manifest_split:{which}", top_manifest["evaluation_split_path"] == expected_split)
    audit.require(f"metrics_split:{which}", metrics["evaluation_split_path"] == expected_split)
    manifest_path = root / "test" / "manifest.json"
    audit.file(manifest_path)
    manifest = read_json(manifest_path)
    entries = manifest["cases"]
    ids = [item["unit_id"] for item in entries]
    audit.require(f"complete_test_case_list:{which}", len(ids) == len(expected_cases)
                  and len(set(ids)) == len(ids) and set(ids) == set(expected_cases))
    top_entries = top_manifest["partitions"]["test"]
    audit.require(f"manifest_crosscheck:{which}", top_entries ==
                  [{**entry, "file": str(Path("test") / entry["file"])} for entry in entries])
    truths, predictions = {"normalized": [], "physical": []}, {"normalized": [], "physical": []}
    for entry in entries:
        unit_id = entry["unit_id"]
        relative = Path(entry["file"])
        audit.require(f"safe_npz_path:{unit_id}", not relative.is_absolute() and ".." not in relative.parts
                      and relative == Path(unit_id) / "predictions.npz")
        npz_path = root / "test" / relative
        audit.file(npz_path, entry["sha256"])
        with np.load(npz_path, allow_pickle=False) as archive:
            keys = {"row_index", "pred_norm", "true_norm", "pred_pa", "true_pa", "pred_pa_unclipped"}
            audit.require(f"npz_schema:{unit_id}", keys <= set(archive.files))
            arrays = {key: archive[key] for key in keys}
        raw, normalized = load_source(audit, cfg["data"], stats, unit_id, source_cache, frozen_data)
        n = len(raw)
        audit.require(f"npz_n_points:{unit_id}", entry["n_points"] == n)
        for key, value in arrays.items():
            audit.require(f"npz_shape_finite:{unit_id}:{key}", value.shape == (n,) and np.isfinite(value).all())
        audit.require(f"row_index_integer:{unit_id}", arrays["row_index"].dtype.kind in "iu")
        audit.array(f"row_index_complete:{unit_id}", arrays["row_index"], np.arange(n, dtype=np.int64))
        audit.array(f"true_pa_matches_source:{unit_id}", arrays["true_pa"], raw)
        audit.array(f"true_norm_matches_source:{unit_id}", arrays["true_norm"], normalized, atol=2e-6, rtol=2e-6)
        recovered = np.exp(arrays["pred_norm"].astype(np.float64) * stats["log"]["std"]
                           + stats["log"]["mean"]) - stats["eps"]
        audit.array(f"inverse_log_unclipped:{unit_id}", arrays["pred_pa_unclipped"], recovered, atol=1e-10, rtol=1e-10)
        audit.array(f"physical_clipping:{unit_id}", arrays["pred_pa"], np.maximum(recovered, 0), atol=1e-10, rtol=1e-10)
        for space, truth_key, pred_key in (("normalized", "true_norm", "pred_norm"), ("physical", "true_pa", "pred_pa")):
            truths[space].append(arrays[truth_key])
            predictions[space].append(arrays[pred_key])
    result = {space: summarize(truths[space], predictions[space], ids) for space in truths}
    compare_metrics(audit, which + ".normalized", result["normalized"], metrics["normalized"])
    compare_metrics(audit, which + ".physical", result["physical"], metrics)
    return result


def verify_run(run_dir, checkpoint="last", expected_epochs=400, expected_cases=34, preflight_path=None):
    run, audit = Path(run_dir).resolve(), Audit()
    result = {"schema_version": 1, "run_dir": str(run), "requested_checkpoint": checkpoint,
              "created_at": datetime.now(timezone.utc).isoformat(), "passed": False,
              "metric_implementation": "independent NumPy; source bundle labels checked",
              "checkpoints": {}}
    try:
        audit.file(run / "config.json")
        cfg = read_json(run / "config.json")
        frozen_data = None
        if preflight_path is None and cfg.get("name", "").startswith("wss_direct_recovery_20260912/"):
            preflight_path = Path(__file__).resolve().parents[2] / "training_wss_min/experiments/wss_direct_recovery_20260912/runtime_preflight.json"
        if preflight_path is not None:
            audit.file(preflight_path)
            preflight = read_json(preflight_path)
            audit.require("successful_frozen_gpu_preflight", preflight.get("passed") is True
                          and bool(preflight.get("slurm_job_id")) and bool(preflight.get("data_fingerprints")))
            frozen_data = preflight["data_fingerprints"]
            result["preflight_path"] = str(Path(preflight_path).resolve())
        data = cfg["data"]
        audit.require("WSS_peak_global_stats_protocol", data["target"] == "wss"
                      and data.get("timesteps", "peak") == "peak"
                      and data["target_normalization"] == "global_stats")
        audit.require("no_jensen_transform", not cfg.get("eval", {}).get("pointwise_jensen", False))
        stats_path = run / "wss_global_stats.json"
        audit.file(stats_path)
        stats = read_json(stats_path)
        audit.require("log_z_stats", stats["method"] == "log_z" and stats["log"]["std"] > 0)
        audit.file(data["wss_stats_path"])
        audit.require("run_stats_equal_source", stats == read_json(data["wss_stats_path"]))
        audit.require("frozen_split_sha_present", bool(stats.get("split_sha256")))
        audit.file(data["split_path"], stats["split_sha256"])
        split = read_json(data["split_path"])
        cases = split["test_cases"]
        audit.require("test_partition_count", len(cases) == expected_cases and len(set(cases)) == len(cases))
        audit.require("train_test_disjoint", not set(cases) & set(split["train_cases"]))
        audit.require("safe_case_ids", all(not Path(c).is_absolute() and ".." not in Path(c).parts for c in cases))
        result["training"] = check_training(audit, run, cfg, expected_epochs)
        cache = {}
        for which in (("best",) if checkpoint == "best" else ("best", "last")):
            result["checkpoints"][which] = check_evaluation(audit, run, which, cfg, stats, cases, cache, frozen_data)
        result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    result["checks"], result["files_sha256"] = audit.checks, audit.files
    result = clean(result)
    output = run / f"acceptance_{checkpoint}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if checkpoint == "last":
        (run / "acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="last")
    parser.add_argument("--expected-epochs", type=int, default=400)
    parser.add_argument("--expected-cases", type=int, default=34)
    parser.add_argument("--preflight", help="GPU preflight evidence; recovery runs default to their frozen experiment evidence")
    args = parser.parse_args()
    result = verify_run(args.run_dir, args.checkpoint, args.expected_epochs, args.expected_cases, args.preflight)
    print(json.dumps({"passed": result["passed"], "error": result.get("error"),
                      "acceptance": str(Path(args.run_dir) / f"acceptance_{args.checkpoint}.json")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
