"""Evidence-qualified M2 optimization reports and predeclared combination selection.

Reading test34 here is exploratory model development, not independent validation.
This module never creates training configurations or submits jobs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean, median

from training_wss_min.tools.report_v6_followup import (
    METRIC_DEFINITIONS, PATHS as V6_PATHS, matrix_finalization, source_integrity,
)

ROOT = Path(__file__).resolve().parents[2]
NAME = "m2_optimization_20260909"
EXP = ROOT / "training_wss_min/experiments" / NAME
MATRIX = ROOT / "training_wss_min/configs" / NAME / "matrix.json"
COMBINATIONS = MATRIX.with_name("combinations.json")
RUNS = ROOT / "training_wss_min/runs"
ANCHOR = "v6_followup_20260909/M2_a5_independent_k3_s1234"
PATHS = {**V6_PATHS, "negative_cases": ("aggregate", "r2_negative_cases"),
         "high_wss_mae": ("regional_field", "high_wss", "mae"),
         "high_wss_rmse": ("regional_field", "high_wss", "rmse")}
BASE_IDS = ["MO0"] + [f"MO-{f}{i}" for f, n in (("L", 8), ("P", 5), ("S", 6)) for i in range(1, n + 1)]
PROTOCOL = "best按各臂自身训练总损失选模；last固定epoch399；单seed1234、已暴露test34，仅探索性比较"
RULES = {
    "min_delta_best_physical_r2_cb": 0.01, "min_delta_last_physical_r2_cb": 0.005,
    "max_relative_mae_increase": 0.02, "max_top10_iou_drop": 0.01,
    "max_case_p10_drop": 0.02, "max_negative_cases_increase": 0,
    "references": ["historical_m2", "MO0"], "guards_checkpoints": ["best", "last"],
    "ranking": ["best physical_r2_cb descending", "last physical_r2_cb descending", "best mae ascending", "arm ID ascending"],
    "interpretation": "预登记实用筛选线；不是置信带、显著性或稳定性判据",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def dig(obj, path):
    for key in path:
        obj = obj[key]
    value = float(obj)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite metric at {'.'.join(path)}")
    return value


def values(metric):
    return {name: dig(metric, path) for name, path in PATHS.items()}


def load_metrics(run_name, checkpoint, runs_dir=RUNS):
    path = Path(runs_dir) / run_name / "eval" / f"ckpt_{checkpoint}" / "metrics.json"
    payload = read_json(path)
    if payload is None:
        return None
    metric = payload["test"]
    if metric["aggregate"]["n_cases"] != 34 or len(metric["per_case"]) != 34:
        raise ValueError(f"Incomplete test34 metrics: {path}")
    values(metric)
    return metric


def matrix_arms(matrix_path=MATRIX, combinations_path=COMBINATIONS):
    arms = []
    for path, phase in ((Path(matrix_path), "base"), (Path(combinations_path), "combination")):
        manifest = read_json(path)
        if manifest is None:
            if phase == "base":
                raise FileNotFoundError(path)
            continue
        for item in manifest["arms"]:
            arm = copy.deepcopy(item)
            config_path = Path(arm["config"])
            if not config_path.is_absolute():
                config_path = path.parent / config_path
            config = read_json(config_path)
            if config is None:
                raise FileNotFoundError(config_path)
            if arm.get("resolved_config", config) != config:
                raise ValueError(f"Manifest/config content differs: {arm['id']}")
            if arm["run_name"] != config["name"]:
                raise ValueError(f"Manifest/config run name differs: {arm['id']}")
            if arm.get("phase", phase) != phase:
                raise ValueError(f"Manifest phase mismatch: {arm['id']}")
            arm.update(resolved_config=config, phase=phase, config_path=str(config_path.resolve()),
                       manifest_path=str(path.resolve()))
            arms.append(arm)
    if [a["id"] for a in arms if a["phase"] == "base"] != BASE_IDS:
        raise ValueError("Base matrix must contain the twenty predeclared MO arms in order")
    if len({a["id"] for a in arms}) != len(arms) or len({a["run_name"] for a in arms}) != len(arms):
        raise ValueError("Duplicate arm ID or run name")
    if sum(a["phase"] == "combination" for a in arms) > 4:
        raise ValueError("At most four additional combinations are authorized")
    return arms


def same_evaluation_population(metric, reference):
    issues = []
    if set(metric["per_case"]) != set(reference["per_case"]):
        return ["evaluation patient set differs from historical M2"]
    for case, ref in reference["per_case"].items():
        if metric["per_case"][case]["overall"]["n"] != ref["overall"]["n"]:
            issues.append(f"valid wall vertex count differs: {case}")
    for key in ("surface_metric_mode", "metric_space", "back_transform", "evaluation_split_path"):
        if metric.get(key) != reference.get(key):
            issues.append(f"evaluation protocol differs: {key}")
    if metric["field"]["n"] != reference["field"]["n"]:
        issues.append("full wall point total differs")
    return issues


def run_evidence(arm, checkpoint, state, reference, runs_dir=RUNS):
    directory = Path(runs_dir) / arm["run_name"]
    queue = state.get("arms", {}).get(arm["id"], {})
    integrity = source_integrity(state)
    issues, history = [], []
    hp = directory / "history.jsonl"
    try:
        history = [json.loads(line) for line in hp.read_text().splitlines() if line.strip()] if hp.exists() else []
    except json.JSONDecodeError:
        issues.append("incomplete or malformed history")
    trained = [h.get("epoch") for h in history] == list(range(400))
    raw_path = directory / "eval" / f"ckpt_{checkpoint}" / "metrics.json"
    try:
        metric = load_metrics(arm["run_name"], checkpoint, runs_dir)
    except (KeyError, ValueError, TypeError) as exc:
        metric = None
        issues.append(f"invalid raw metrics: {exc}")
    if metric is not None:
        if not trained:
            issues.append("400 complete epochs not verified")
        if not (directory / f"ckpt_{checkpoint}.pt").exists():
            issues.append("checkpoint missing")
        if queue.get("run_name") != arm["run_name"]:
            issues.append("queue run provenance differs or missing")
        saved_config = read_json(directory / "config.json")
        if saved_config is None or saved_config.get("name") != arm["run_name"]:
            issues.append("saved run configuration missing or run name differs")
        else:
            actual_flat = flat_config(saved_config)
            for keys, expected in flat_config(arm["resolved_config"]).items():
                if actual_flat.get(keys) != expected:
                    issues.append(f"saved run config differs from manifest: {'.'.join(keys)}")
        for filename in ("feature_stats.json", "wss_global_stats.json", "target_normalization.json", "weight_quantiles.json"):
            original_path = Path(runs_dir) / ANCHOR / filename
            current_path = directory / filename
            if not original_path.exists() or not current_path.exists():
                issues.append(f"frozen training statistics missing: {filename}")
            elif read_json(original_path) != read_json(current_path):
                issues.append(f"training statistics differ from historical M2: {filename}")
        for stage in ("train", f"eval_{checkpoint}"):
            if queue.get("stages", {}).get(stage, {}).get("returncode") != 0:
                issues.append(f"{stage} has not completed successfully")
        issues.extend(same_evaluation_population(metric, reference))
        if not integrity["start_hash_available"]:
            issues.append("source start hashes missing")
        if integrity["changed"]:
            issues.append("source hashes changed during execution")
        arm_start, arm_end = queue.get("source_sha256"), queue.get("source_sha256_end")
        if arm_start is not None and arm_end is not None and arm_start != arm_end:
            issues.append("per-arm source hashes changed")
        if trained and (directory / f"ckpt_{checkpoint}.pt").exists():
            import torch
            payload = torch.load(directory / f"ckpt_{checkpoint}.pt", map_location="cpu", weights_only=False)
            selected = min(history, key=lambda h: h["train_loss"])
            expected_epoch = selected["epoch"] if checkpoint == "best" else 399
            if payload.get("epoch") != expected_epoch:
                issues.append("checkpoint epoch differs from frozen selection rule")
            if checkpoint == "best" and not math.isclose(float(payload.get("metric", math.nan)), -selected["train_loss"], rel_tol=1e-9, abs_tol=1e-10):
                issues.append("best checkpoint score differs from minimum training total loss")
    qstatus = queue.get("status", "pending")
    valid = metric is not None and not issues
    status = "已评估" if valid else "证据待核验" if metric is not None else str(qstatus)
    evidence = {"valid_scientific_result": valid, "status": status, "issues": issues,
                "history_rows": len(history), "training_400_epochs": trained,
                "queue_status": qstatus, "phase": arm["phase"], "source_integrity": integrity,
                "metrics_path": str(raw_path), "metrics_sha256": sha256(raw_path) if raw_path.exists() else None,
                "history_sha256": sha256(hp) if hp.exists() else None,
                "stages": queue.get("stages", {})}
    if trained:
        selected = min(history, key=lambda h: h["train_loss"])
        evidence["expected_checkpoint_epoch"] = selected["epoch"] if checkpoint == "best" else 399
        evidence["training_seconds"] = history[-1].get("elapsed_s")
    return metric, evidence


def delta_values(metric, reference):
    a, b = values(metric), values(reference)
    return {key: a[key] - b[key] for key in a}


def case_comparison(metric, reference):
    """Pair all cases; MSE contributions exactly explain case-balanced R² delta."""
    if same_evaluation_population(metric, reference):
        raise ValueError("Cannot compare different evaluation populations/protocols")
    count = len(reference["per_case"])
    denominator = reference["field_casebalanced"]["rmse"] ** 2 / (1 - reference["field_casebalanced"]["r2"])
    out = {}
    for case, before in reference["per_case"].items():
        after = metric["per_case"][case]
        item = {"n": after["overall"]["n"], "r2_before": before["overall"]["r2"],
                "r2_after": after["overall"]["r2"],
                "delta_r2": after["overall"]["r2"] - before["overall"]["r2"],
                "delta_mae": after["overall"]["mae"] - before["overall"]["mae"],
                "mse_r2_contribution": (before["overall"]["rmse"] ** 2 - after["overall"]["rmse"] ** 2) / (count * denominator),
                "delta_high_wss_mae": after["high_wss"]["mae"] - before["high_wss"]["mae"],
                "delta_iou": after["hotspot"]["top10_iou"] - before["hotspot"]["top10_iou"],
                "delta_spearman": after["hotspot"]["spearman_all"] - before["hotspot"]["spearman_all"]}
        for prefix, case_metric, normalized_metric in (
            ("before", before, reference["normalized"]["per_case"][case]),
            ("after", after, metric["normalized"]["per_case"][case]),
        ):
            item[f"{prefix}_normalized_r2"] = normalized_metric["overall"]["r2"]
            item[f"{prefix}_high_wss_mae"] = case_metric["high_wss"]["mae"]
            item[f"{prefix}_high_wss_rmse"] = case_metric["high_wss"]["rmse"]
            item[f"{prefix}_high_wss_r2"] = case_metric["high_wss"]["r2"]
            item[f"{prefix}_top10_iou"] = case_metric["hotspot"]["top10_iou"]
            item[f"{prefix}_top10_ratio"] = case_metric["calibration"]["top10_pred_true_ratio"]
            item[f"{prefix}_p99_ratio"] = case_metric["calibration"]["p99_pred_true_ratio"]
        for key, name in (("top10_pred_true_ratio", "top10"), ("p99_pred_true_ratio", "p99")):
            item[f"{name}_closeness_improvement"] = abs(before["calibration"][key] - 1) - abs(after["calibration"][key] - 1)
        out[case] = item
    total = sum(v["mse_r2_contribution"] for v in out.values())
    expected = metric["field_casebalanced"]["r2"] - reference["field_casebalanced"]["r2"]
    if not math.isclose(total, expected, abs_tol=1e-9, rel_tol=1e-8):
        raise ValueError(f"Case MSE contributions do not sum to R² difference: {total} != {expected}")
    ranking = sorted(out, key=lambda c: out[c]["mse_r2_contribution"])
    return {"per_case": out, "contribution_sum": total, "delta_physical_r2_cb": expected,
            "contribution_sum_verified": True, "r2_cases_improved": sum(x["delta_r2"] > 0 for x in out.values()),
            "iou_cases_improved": sum(x["delta_iou"] > 0 for x in out.values()),
            "largest_costs": ranking[:3], "largest_benefits": ranking[-3:]}


def _percentile(array, q):
    x = sorted(array)
    at = (len(x) - 1) * q
    lo = int(at)
    return x[lo] + (x[min(lo + 1, len(x) - 1)] - x[lo]) * (at - lo)


def audit_metric(metric, reference):
    """Independently reaggregate all 34 case summaries in Pa and log_z.

    This verifies stored aggregate arithmetic, not a second inference run. The
    fixed target-variance denominator comes from the frozen historical metric.
    """
    checks = []
    def check(name, observed, expected):
        ok = math.isclose(float(observed), float(expected), rel_tol=1e-8, abs_tol=1e-9)
        checks.append({"name": name, "observed": observed, "expected": expected, "passed": ok})
    for space, data, ref in (("Pa", metric, reference), ("log_z", metric["normalized"], reference["normalized"])):
        cases = list(data["per_case"].values())
        r2 = [c["overall"]["r2"] for c in cases]
        n = sum(c["overall"]["n"] for c in cases)
        check(f"{space}.n", data["field"]["n"], n)
        check(f"{space}.case_mean", data["aggregate"]["r2_casemean"], mean(r2))
        check(f"{space}.case_median", data["aggregate"]["r2_casemed"], median(r2))
        check(f"{space}.case_p10", data["aggregate"]["r2_casep10"], _percentile(r2, .1))
        check(f"{space}.negative_cases", data["aggregate"]["r2_negative_cases"], sum(r < 0 for r in r2))
        for region, stored in (("overall", data["field"]), ("high_wss", data["regional_field"]["high_wss"])):
            regional = [c[region] for c in cases if c[region]["n"] > 0]
            size = sum(c["n"] for c in regional)
            for key in ("mae", "rmse"):
                power = 2 if key == "rmse" else 1
                combined = (sum(c[key] ** power * c["n"] for c in regional) / size) ** (1 / power)
                check(f"{space}.{region}.pooled_{key}", stored[key], combined)
        cb_mse = mean([c["overall"]["rmse"] ** 2 for c in cases])
        cb_var = ref["field_casebalanced"]["rmse"] ** 2 / (1 - ref["field_casebalanced"]["r2"])
        check(f"{space}.cb_rmse", data["field_casebalanced"]["rmse"], math.sqrt(cb_mse))
        check(f"{space}.cb_mae", data["field_casebalanced"]["mae"], mean([c["overall"]["mae"] for c in cases]))
        check(f"{space}.cb_r2_fixed_target_variance", data["field_casebalanced"]["r2"], 1 - cb_mse / cb_var)
        for src, dst in (("top10_iou", "top10_iou_casemean"), ("spearman_all", "spearman_all_casemean")):
            check(f"{space}.{dst}", data["hotspot"][dst], mean([c["hotspot"][src] for c in cases]))
    # Check every stored per-case scalar, including regions, amplitude and ranking.
    numeric_count = 0
    failures = []
    def walk(obj, path):
        nonlocal numeric_count
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, path + "." + key)
        elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
            numeric_count += 1
            if not math.isfinite(obj):
                failures.append(path)
    walk(metric["per_case"], "Pa.per_case")
    walk(metric["normalized"]["per_case"], "log_z.per_case")
    return {"passed": all(c["passed"] for c in checks) and not failures,
            "aggregation_checks": checks, "aggregation_check_count": len(checks),
            "per_case_numeric_fields_checked": numeric_count, "nonfinite_fields": failures,
            "boundary": "独立重算逐病例指标汇总与MSE贡献；未重推理，未从原始逐点预测重算所有分位数/排名"}


def candidate_gate(best, last, references):
    """Both checkpoint improvements and guards must pass both M2 references."""
    issues = []
    for ref_name, ref in references.items():
        for checkpoint, candidate in (("best", best), ("last", last)):
            before = ref[checkpoint]
            margin = RULES[f"min_delta_{checkpoint}_physical_r2_cb"]
            if candidate["physical_r2_cb"] - before["physical_r2_cb"] < margin - 1e-12:
                issues.append(f"{ref_name}/{checkpoint}: Pa R² gain < {margin}")
            if candidate["mae"] > before["mae"] * 1.02 + 1e-12:
                issues.append(f"{ref_name}/{checkpoint}: MAE increases > 2%")
            if candidate["top10_iou"] < before["top10_iou"] - .01 - 1e-12:
                issues.append(f"{ref_name}/{checkpoint}: IoU drops > .01")
            if candidate["case_p10"] < before["case_p10"] - .02 - 1e-12:
                issues.append(f"{ref_name}/{checkpoint}: case P10 drops > .02")
            if candidate["negative_cases"] > before["negative_cases"]:
                issues.append(f"{ref_name}/{checkpoint}: negative-R² case count increases")
    return {"qualified": not issues, "issues": issues, "rules": RULES}


def family_of(arm):
    family = str(arm.get("family", "")).upper()
    if family in {"L", "P", "S"}:
        return family
    for prefix, result in (("LOSS", "L"), ("PARAM", "P"), ("STRUCT", "S")):
        if family.startswith(prefix):
            return result
    return arm["id"].split("-")[-1][:1] if arm["id"].startswith("MO-") else None


def select_family_winners(summaries):
    best, last = summaries["best"], summaries["last"]
    baseline_ready = all(summaries[ck]["arms"].get("MO0", {}).get("evidence", {}).get("valid_scientific_result") for ck in ("best", "last"))
    complete = all(summaries[ck].get("base_matrix_finalized", False) for ck in ("best", "last"))
    selection = {"ready": complete and baseline_ready, "rules": RULES, "families": {}, "candidates": {},
                 "final_priority_candidates": [], "combination_priority": {}}
    if not selection["ready"]:
        selection["reason"] = "Wait for all twenty base runs and both evaluations, frozen source evidence and MO0"
        return selection
    refs = {"historical_m2": {ck: summaries[ck]["reference"] for ck in ("best", "last")},
            "MO0": {ck: summaries[ck]["arms"]["MO0"] for ck in ("best", "last")}}
    for aid, entry in best["arms"].items():
        if aid == "MO0":
            continue
        other = last["arms"].get(aid, {})
        if not entry.get("evidence", {}).get("valid_scientific_result") or not other.get("evidence", {}).get("valid_scientific_result"):
            selection["candidates"][aid] = {"qualified": False, "issues": ["both checkpoint results not qualified"]}
            continue
        selection["candidates"][aid] = candidate_gate(entry, other, refs)
    def rank(aid):
        return (-best["arms"][aid]["physical_r2_cb"], -last["arms"][aid]["physical_r2_cb"], best["arms"][aid]["mae"], aid)
    for family in ("L", "P", "S"):
        qualified = [aid for aid, gate in selection["candidates"].items() if gate["qualified"]
                     and best["arms"][aid].get("phase") == "base" and family_of(best["arms"][aid]) == family]
        qualified.sort(key=rank)
        selection["families"][family] = {"winner": qualified[0] if qualified else "MO0", "qualified": qualified,
                                            "fallback": not qualified}
    all_qualified = sorted([aid for aid, gate in selection["candidates"].items() if gate["qualified"]], key=rank)
    selection["ranking"] = all_qualified
    selection["highest_qualified_candidate"] = all_qualified[0] if all_qualified else "MO0"
    observed = sorted([aid for aid in selection["candidates"]
                       if best["arms"][aid].get("evidence", {}).get("valid_scientific_result")
                       and last["arms"].get(aid, {}).get("evidence", {}).get("valid_scientific_result")], key=rank)
    selection["observed_ranking"] = observed
    selection["highest_observed_candidate"] = observed[0] if observed else None
    selection["observed_ranking_note"] = "保留全部有效候选，包含未过门槛与负结果；观测最高值不等于合格或稳定替换。"
    # This final priority rule is separate from family selection and combination
    # generation. A valid single arm can block a combination even if that single
    # did not pass its own protection lines. Exact best ties never qualify merely
    # by having a larger last score.
    valid_singles = [aid for aid, entry in best["arms"].items()
                     if aid != "MO0" and entry.get("phase") == "base"
                     and entry.get("evidence", {}).get("valid_scientific_result")]
    highest_single_score = max((best["arms"][aid]["physical_r2_cb"] for aid in valid_singles), default=None)
    highest_single_ids = [aid for aid in valid_singles if best["arms"][aid]["physical_r2_cb"] == highest_single_score]
    for aid, entry in best["arms"].items():
        if entry.get("phase") != "combination":
            continue
        gate = selection["candidates"][aid]
        issues = []
        if not gate["qualified"]:
            issues.append("candidate_gate failed: " + "; ".join(gate["issues"]))
        score = entry.get("physical_r2_cb")
        if highest_single_score is None:
            issues.append("no valid base single-arm best score is available")
        elif score is None or not score > highest_single_score:
            issues.append("combination best Pa R² must strictly exceed every valid base single arm; ties do not qualify")
        selection["combination_priority"][aid] = {
            "eligible": not issues, "issues": issues, "candidate_gate_qualified": gate["qualified"],
            "best_physical_r2_cb": score, "highest_valid_single_best_physical_r2_cb": highest_single_score,
            "highest_valid_single_arm_ids": highest_single_ids,
            "delta_vs_highest_valid_single": score - highest_single_score if score is not None and highest_single_score is not None else None,
        }
    selection["final_priority_candidates"] = [aid for aid in all_qualified
        if best["arms"][aid].get("phase") != "combination" or selection["combination_priority"][aid]["eligible"]]
    selection["highest_final_priority_candidate"] = next(iter(selection["final_priority_candidates"]), None)
    selection["final_priority_rule"] = "基础单项通过原门槛即可进入最终优先列表；组合另须best Pa R²严格超过全部有效基础单项（不含MO0）的最高best。合格排名不是最终优先排名。"
    return selection


def flat_config(config):
    """Ignore output labels; compare actual training/evaluation settings."""
    output = {}
    def walk(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], path + (key,))
        else:
            output[path] = value
    for key in ("data", "model", "train", "eval"):
        walk(config.get(key, {}), (key,))
    return output


def config_fingerprint(config):
    serial = [[list(k), v] for k, v in sorted(flat_config(config).items())]
    return hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def propose_combinations(arms, summaries):
    selection = select_family_winners(summaries)
    result = {"ready": selection["ready"], "selection": selection, "arms": [], "skipped": [],
              "maximum_additional_runs": 4, "boundary": PROTOCOL}
    if not selection["ready"]:
        return result
    by_id = {arm["id"]: arm for arm in arms}
    baseline = by_id["MO0"]["resolved_config"]
    base_flat = flat_config(baseline)
    seen = {config_fingerprint(arm["resolved_config"]): arm["id"] for arm in arms if arm["phase"] == "base"}
    for families in (("L", "P"), ("L", "S"), ("P", "S"), ("L", "P", "S")):
        members = [selection["families"][f]["winner"] for f in families if selection["families"][f]["winner"] != "MO0"]
        combination_id = "MO-C" + "".join(families)
        if len(members) < 2:
            result["skipped"].append({"id": combination_id, "reason": "fewer than two qualified families", "members": members})
            continue
        changes, conflict = {}, []
        for member in members:
            for key, value in flat_config(by_id[member]["resolved_config"]).items():
                if key in base_flat and base_flat[key] == value:
                    continue
                if key in changes and changes[key] != value:
                    conflict.append(".".join(key))
                changes[key] = value
        if conflict:
            result["skipped"].append({"id": combination_id, "reason": "incompatible config changes", "keys": sorted(set(conflict)), "members": members})
            continue
        combined = copy.deepcopy(baseline)
        for keys, value in changes.items():
            node = combined
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = copy.deepcopy(value)
        fingerprint = config_fingerprint(combined)
        if fingerprint in seen:
            result["skipped"].append({"id": combination_id, "reason": "duplicate effective configuration", "duplicate_of": seen[fingerprint]})
            continue
        seen[fingerprint] = combination_id
        result["arms"].append({"id": combination_id, "members": members, "families": list(families),
                               "config_sha256": fingerprint, "changes": {".".join(k): v for k, v in changes.items()}})
    return result


def compute_interactions(summary, selection, arms):
    entries = summary["arms"]
    usable = {aid: entry for aid, entry in entries.items() if entry["evidence"]["valid_scientific_result"]}
    result = {"checkpoint": summary["checkpoint"], "reference": "MO0", "pairs": {}, "triple": None,
              "interpretation": "描述性加法交互；不作显著协同或独立机制判断"}
    if not selection.get("ready") or "MO0" not in usable:
        return result
    singles = {f: selection["families"][f]["winner"] for f in ("L", "P", "S")}
    combos = {}
    for arm in arms:
        if arm.get("phase") == "combination" and arm["id"] in usable:
            members = arm.get("members", arm.get("components", []))
            if members:
                combos[frozenset(members)] = arm["id"]
    for left, right in itertools.combinations(("L", "P", "S"), 2):
        a, b = singles[left], singles[right]
        cid = combos.get(frozenset((a, b)))
        if a == "MO0" or b == "MO0" or not cid or a not in usable or b not in usable:
            continue
        result["pairs"][left + right] = {"arm": cid, "members": [a, b], "values": {
            key: usable[cid][key] - usable[a][key] - usable[b][key] + usable["MO0"][key] for key in PATHS}}
    triple = combos.get(frozenset(singles.values()))
    if len(set(singles.values())) == 3 and "MO0" not in singles.values() and triple and len(result["pairs"]) == 3:
        pair_ids = [p["arm"] for p in result["pairs"].values()]
        result["triple"] = {"arm": triple, "members": list(singles.values()), "values": {
            key: usable[triple][key] - sum(usable[p][key] for p in pair_ids) + sum(usable[s][key] for s in singles.values()) - usable["MO0"][key] for key in PATHS},
            "marginal_vs_pairs": {p: {key: usable[triple][key] - usable[p][key] for key in PATHS} for p in pair_ids}}
    return result


def make_table(summary):
    columns = [("Pa R²_cb", "physical_r2_cb"), ("log_z R²_cb", "normalized_r2_cb"), ("case mean", "case_mean"),
               ("P10", "case_p10"), ("MAE Pa", "mae"), ("high-WSS R²", "high_wss_r2"), ("high-WSS MAE Pa", "high_wss_mae"),
               ("top10比", "top10_ratio"), ("p99比", "p99_ratio"), ("IoU", "top10_iou"), ("负R²数", "negative_cases")]
    lines = [f"# M2 优化实验（{summary['checkpoint']}）", "", PROTOCOL,
             "损失臂比较的是损失及其训练选模配方；不得在best/last中择优报数。新几何继续等待人工验收。", "",
             "| 臂 | 状态 | Δ vs 历史M2 | Δ vs MO0 | " + " | ".join(c[0] for c in columns) + " |",
             "|---|---|---:|---:|" + "---:|" * len(columns)]
    for aid, entry in [("历史M2", summary["reference"]), *summary["arms"].items()]:
        valid = aid == "历史M2" or entry["evidence"]["valid_scientific_result"]
        status = "历史参照" if aid == "历史M2" else entry["evidence"]["status"]
        dh = entry.get("deltas_vs_historical_m2", {}).get("physical_r2_cb")
        dm = entry.get("deltas_vs_mo0", {}).get("physical_r2_cb")
        row = [aid, status] + ["—" if x is None else f"{x:+.5f}" for x in (dh, dm)]
        row.extend(f"{entry[key]:.5f}" if valid else "—" for _, key in columns)
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", f"有效评估 {summary['completed_count']}/{summary['expected_count']}；基础矩阵完结核验={summary['base_matrix_finalized']}；全部已登记训练完结核验={summary['matrix_finalized']}。", "",
              "Pa R²_cb使用病例等权共享均值；MAE为壁面顶点pool；case mean为逐病例R²平均。幅值比按距1远近读取。",
              "R²_fit仅作相关性诊断；不使用测试标签拟合校准后重报准确率。完整指标、逐病例变化、贡献、门槛及交互在JSON/CSV中。"]
    control = summary.get("contemporary_control_diagnosis", {})
    if control.get("available"):
        lines += ["", control["note"], "",
                  "[同期MO0复现偏差审计](contemporary_control_diagnosis.md)；[逐病例与训练轨迹证据](contemporary_control_diagnosis.json)。"]
    selection = summary.get("selection", {})
    if selection.get("ready"):
        lines += ["", "合格排名（原门槛）：" + ("、".join(selection["ranking"]) or "无") + "。",
                  "最终优先候选：" + ("、".join(selection["final_priority_candidates"]) or "无") + "。",
                  selection["final_priority_rule"]]
        if any(not p["eligible"] for p in selection["combination_priority"].values()):
            lines.append("")
        for aid, priority in selection["combination_priority"].items():
            if not priority["eligible"]:
                lines.append(f"- {aid}不列最终优先：" + "；".join(priority["issues"]) + "。")
    return "\n".join(lines) + "\n"


def report_all(matrix_path=MATRIX, combinations_path=COMBINATIONS, experiment_dir=EXP, runs_dir=RUNS, write=True):
    experiment_dir = Path(experiment_dir)
    arms = matrix_arms(matrix_path, combinations_path)
    states = {"base": read_json(experiment_dir / "queue_status.json", {}),
              "combination": read_json(experiment_dir / "combination_queue_status.json", {})}
    phase_final = {}
    for phase, state in states.items():
        ids = [a["id"] for a in arms if a["phase"] == phase]
        phase_final[phase] = matrix_finalization(state, ids, source_integrity(state)) if ids else {"finalized": True, "issues": []}
    summaries, all_audits, csv_rows = {}, {}, []
    for ck in ("best", "last"):
        historical = load_metrics(ANCHOR, ck, runs_dir)
        if historical is None:
            raise FileNotFoundError(f"Historical M2 {ck} missing")
        loaded = {a["id"]: run_evidence(a, ck, states[a["phase"]], historical, runs_dir) for a in arms}
        mo0 = loaded["MO0"][0] if loaded["MO0"][1]["valid_scientific_result"] else None
        summary = {"checkpoint": ck, "reference": {"id": "historical_m2", "run_name": ANCHOR, **values(historical)},
                   "checkpoint_selection": PROTOCOL, "metric_definitions": METRIC_DEFINITIONS,
                   "expected_count": len(arms), "base_matrix_finalized": phase_final["base"]["finalized"],
                   "matrix_finalized": all(x["finalized"] for x in phase_final.values()), "phase_finalization": phase_final, "arms": {}}
        audits = {}
        for arm in arms:
            aid = arm["id"]
            metric, evidence = loaded[aid]
            entry = {k: arm[k] for k in ("id", "parent", "family", "phase", "run_name", "hypothesis", "config")}
            entry["evidence"] = evidence
            entry["members"] = arm.get("members", arm.get("components", []))
            if metric is not None:
                entry["raw_metrics"] = values(metric)
            if evidence["valid_scientific_result"]:
                audit = audit_metric(metric, historical)
                audits[aid] = audit
                if not audit["passed"]:
                    evidence["valid_scientific_result"] = False
                    evidence["status"] = "逐病例数值核验失败"
                    evidence["issues"].append("independent aggregate audit failed")
                    if aid == "MO0":
                        mo0 = None
                else:
                    entry.update(values(metric))
                    entry["efficiency"] = metric.get("efficiency", {})
                    for label, reference in (("historical_m2", historical), ("mo0", mo0)):
                        if reference is None:
                            continue
                        entry[f"deltas_vs_{label}"] = delta_values(metric, reference)
                        comparison = case_comparison(metric, reference)
                        entry[f"cases_vs_{label}"] = comparison
                        for case, result in comparison["per_case"].items():
                            csv_rows.append({"checkpoint": ck, "arm": aid, "reference": label, "case": case, **result})
                    parent = arm["parent"]
                    if parent in loaded and loaded[parent][1]["valid_scientific_result"]:
                        entry["deltas_vs_parent"] = delta_values(metric, loaded[parent][0])
                    entry["deltas_vs_members"] = {mid: delta_values(metric, loaded[mid][0]) for mid in entry["members"] if mid in loaded and loaded[mid][1]["valid_scientific_result"]}
            summary["arms"][aid] = entry
        summary["completed_count"] = sum(a["evidence"]["valid_scientific_result"] for a in summary["arms"].values())
        summary["base_matrix_finalized"] &= all(summary["arms"][aid]["evidence"]["valid_scientific_result"] for aid in BASE_IDS)
        summary["matrix_finalized"] &= summary["completed_count"] == len(arms)
        summaries[ck] = summary
        all_audits[ck] = audits
    selection = select_family_winners(summaries)
    recommendations = propose_combinations(arms, summaries)
    expected_combinations = {a["config_sha256"] for a in recommendations["arms"]}
    registered_combinations = {config_fingerprint(a["resolved_config"]) for a in arms if a["phase"] == "combination"}
    combination_registration = {
        "selection_ready": recommendations["ready"],
        "recommended_count": len(expected_combinations), "registered_count": len(registered_combinations),
        "missing_config_hashes": sorted(expected_combinations - registered_combinations),
        "unexpected_config_hashes": sorted(registered_combinations - expected_combinations),
        "matched": recommendations["ready"] and expected_combinations == registered_combinations,
    }
    for ck, summary in summaries.items():
        summary["selection"] = selection
        summary["interactions"] = compute_interactions(summary, selection, arms)
        summary["registered_matrix_finalized"] = summary["matrix_finalized"]
        summary["combination_registration"] = combination_registration
        summary["matrix_finalized"] &= combination_registration["matched"]
        control = summary["arms"]["MO0"]
        control_available = control["evidence"]["valid_scientific_result"]
        difference = control.get("deltas_vs_historical_m2", {}).get("physical_r2_cb")
        diagnosis = read_json(experiment_dir / "contemporary_control_diagnosis.json", {})
        summary["contemporary_control_diagnosis"] = {
            "available": control_available, "delta_physical_r2_cb": difference,
            "diagnosis_markdown": str(experiment_dir / "contemporary_control_diagnosis.md"),
            "diagnosis_json": str(experiment_dir / "contemporary_control_diagnosis.json"),
            "gpu_diagnosis_status": diagnosis.get("gpu_training_diagnosis", {}).get("status", "not recorded"),
            "note": (f"同期MO0的{ck} Pa R²_cb相对历史M2为{difference:+.8f}。"
                     "同配方复现结果存在差异；不能把差异直接归因于GPU噪声，也不能据单次运行判定稳定性。"
                     "所有19个单项与已登记组合均保留，继续同时对比历史M2和同期MO0及各自best/last；筛选门槛不变。") if difference is not None else "同期MO0尚未取得可核验结果。",
        }
    audit = {"checkpoints": all_audits, "all_available_results_passed": all(a["passed"] for x in all_audits.values() for a in x.values()),
             "evaluations_checked": sum(len(x) for x in all_audits.values()),
             "aggregation_checks": sum(a["aggregation_check_count"] for x in all_audits.values() for a in x.values()),
             "per_case_numeric_fields_checked": sum(a["per_case_numeric_fields_checked"] for x in all_audits.values() for a in x.values())}
    if write:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for ck, summary in summaries.items():
            (experiment_dir / f"matrix_summary_{ck}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            (experiment_dir / f"matrix_tables_{ck}.md").write_text(make_table(summary))
        for filename, data in (("selection.json", selection), ("combination_recommendations.json", recommendations), ("independent_metric_audit.json", audit)):
            (experiment_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        if csv_rows:
            with (experiment_dir / "per_case_comparisons.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
    return {**summaries, "selection": selection, "recommendations": recommendations, "audit": audit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--combinations", type=Path, default=COMBINATIONS)
    parser.add_argument("--experiment-dir", type=Path, default=EXP)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    args = parser.parse_args(argv)
    result = report_all(args.matrix, args.combinations, args.experiment_dir, args.runs_dir)
    print(json.dumps({ck: {"completed": result[ck]["completed_count"], "finalized": result[ck]["matrix_finalized"]} for ck in ("best", "last")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
