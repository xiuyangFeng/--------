"""Reuse frozen M2 runtime checks for the separately selected combinations."""
from __future__ import annotations
import gc
import json
import os
import time
import traceback
from pathlib import Path
import torch
from training_wss_min import config as C, dataset as D
from training_wss_min.models import build_model
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.runtime import seed_all
from training_wss_min.tools import preflight_m2_optimization as P
from training_wss_min.tools.m2_optimization_common import (
    ANCHOR_CONFIG, ANCHOR_RUN, COMBINATIONS, CONFIGS, EXP, MATRIX, fingerprints, save_json, sha, stamp,
)
from training_wss_min.tools.prepare_m2_combinations import validate_manifest


def validate_frozen_inputs(matrix):
    """Check base completion and exact frozen configs before allocating models."""
    arms = validate_manifest(matrix)
    if not arms:
        raise RuntimeError("No additional combinations were selected; no preflight is needed")
    base_state_path = EXP / "queue_status.json"
    base_state = json.loads(base_state_path.read_text())
    if (base_state.get("status") != "complete" or matrix["base_queue_sha256"] != sha(base_state_path)
            or matrix["base_matrix_sha256"] != sha(MATRIX)):
        raise RuntimeError("Combination selection does not match the completed frozen base phase")
    legacy = json.loads((EXP / "runtime_preflight.json").read_text())
    base_sources = fingerprints("base")
    if (not legacy.get("passed") or legacy.get("source_sha256") != base_sources
            or legacy.get("source_sha256_end") != base_sources):
        raise RuntimeError("A matching passing base runtime preflight is required")
    for arm in arms:
        path = CONFIGS / arm["config"]
        if not path.is_file() or json.loads(path.read_text()) != arm["resolved_config"]:
            raise RuntimeError(f"Frozen combination config mismatch: {arm['id']}")
    return arms


def execution_tool_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in
            ("prepare_m2_combinations.py", "preflight_m2_combinations.py")}


def main():
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Combination preflight requires Slurm CUDA allocation")
    torch.set_num_threads(2)
    source = fingerprints("combination")
    matrix = json.loads(COMBINATIONS.read_text())
    output = EXP / "combination_runtime_preflight.json"
    wrappers = execution_tool_hashes()
    result = dict(passed=False, job_id=os.environ["SLURM_JOB_ID"], started_at=stamp(),
        source_sha256=source, arms={}, execution_tool_sha256=wrappers,
        wrapper_sha256=wrappers["preflight_m2_combinations.py"])
    save_json(output, result)
    try:
        arms = validate_frozen_inputs(matrix)
        result["expected_arms"] = [arm["id"] for arm in arms]
        result["legacy_compatibility_reused"] = sha(EXP / "runtime_preflight.json")
        anchor = C.ExpConfig.from_json(ANCHOR_CONFIG)
        stats = D.load_wss_stats(anchor.data.wss_stats_path)
        feature_stats = json.loads((ANCHOR_RUN / "feature_stats.json").read_text())
        quantiles = json.loads((ANCHOR_RUN / "weight_quantiles.json").read_text())
        cases = P._load_cases(anchor, stats)
        inputs = P._input_fingerprints(anchor, cases)
        result["input_sha256"] = inputs
        dataset = D.WSSMinDataset(cases, anchor.data, feature_stats, training=True, base_seed=1234)
        samples = [dataset[i] for i in range(8)]
        P._assert(all(len(sample["support_pos"]) == len(sample["pos"]) == 5000 for sample in samples),
                  "expected eight cases with 5000 support and independent query points each")
        batch = D.collate(samples)
        evaluation_case = max(cases, key=lambda c: len(D.query_rows(c)))
        seed_all(1234)
        reference = build_model(anchor.model, C.input_dim(anchor))
        reference_state = {k: v.clone() for k, v in reference.state_dict().items()}
        del reference
        for arm in arms:
            model = None
            started = time.monotonic()
            try:
                cfg = C.ExpConfig.from_json(CONFIGS / arm["config"])
                P._assert(cfg.data == anchor.data and cfg.eval == anchor.eval,
                          "combination changes frozen M2 data/evaluation")
                for key, value in quantiles.items():
                    setattr(cfg.train, "_raw_p90" if key == "raw_p90" else key, value)
                model, initialization = build_paired_model(cfg)
                components = P._initialization_check(model, initialization, reference_state)
                keys = set(initialization["new_keys"] + initialization["changed_shape_keys"])
                model.cuda()
                train = P._train_check(model, cfg, batch, stats, keys)
                gc.collect(); torch.cuda.empty_cache()
                evaluation = P._evaluation_check(model, cfg, evaluation_case, feature_stats)
                result["arms"][arm["id"]] = dict(passed=True, job_id=os.environ["SLURM_JOB_ID"],
                    config_sha256=sha(CONFIGS / arm["config"]), common_initialization_exact=True,
                    parameters=sum(p.numel() for p in model.parameters()),
                    initial_component_sha256=components, initialization=initialization, train=train, eval=evaluation,
                    max_memory_reserved_mib=max(train["max_memory_reserved_mib"], evaluation["max_memory_reserved_mib"]),
                    max_memory_allocated_mib=max(train["max_memory_allocated_mib"], evaluation["max_memory_allocated_mib"]),
                    seconds=time.monotonic() - started)
                print(arm["id"], "preflight passed", flush=True)
            except Exception as exc:
                result["arms"][arm["id"]] = dict(passed=False, error=str(exc), traceback=traceback.format_exc())
            finally:
                del model
                gc.collect(); torch.cuda.empty_cache()
                save_json(output, result)
        result["source_sha256_end"] = fingerprints("combination")
        result["input_sha256_end"] = P._input_fingerprints(anchor, cases)
        result["execution_tool_sha256_end"] = execution_tool_hashes()
        result["passed"] = (source == result["source_sha256_end"] and inputs == result["input_sha256_end"]
                            and wrappers == result["execution_tool_sha256_end"]
                            and set(result["arms"]) == set(result["expected_arms"])
                            and all(r["passed"] for r in result["arms"].values()))
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc(), passed=False)
    finally:
        result["ended_at"] = stamp()
        save_json(output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
