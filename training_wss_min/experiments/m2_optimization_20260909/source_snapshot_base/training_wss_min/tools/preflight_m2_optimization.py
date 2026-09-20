"""Slurm-only, real-data numerical and memory preflight for all 20 M2 arms."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from training_wss_min import config as C, dataset as D, surface as S
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.runtime import lr_lambda_factory, seed_all
from training_wss_min.tools.m2_optimization_common import (
    ANCHOR_CONFIG, ANCHOR_RUN, CONFIGS, EXP, MATRIX, ROOT,
    fingerprints, save_json, sha, stamp,
)


LEGACY = ROOT / "training_wss_min/experiments/v6_followup_20260909/source_snapshot_13205/training_wss_min/baseline_models.py"
OUTPUT = EXP / "runtime_preflight.json"
MIB = 1024 ** 2
SUCCESSFUL_STEPS = 3
MAX_AMP_ATTEMPTS = 16


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _memory():
    torch.cuda.synchronize()
    return {
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
    }


def _to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _forward(model, batch, cfg):
    return model.forward_support_query(
        batch["support_pos"], batch["support_x"], batch["support_batch"],
        batch["pos"], batch["x"], batch["batch"], unit_ids=batch["unit_ids"],
        epoch=0, global_seed=cfg.train.seed, evaluation=not model.training,
    )


def _load_cases(cfg, stats):
    pairs = D.load_split_cases(cfg.data.split_path, "train")
    _assert(len(pairs) == 138, "M2 preflight requires the frozen train138 split")
    chosen = []
    for family in ("AG", "AAA", "ILO"):
        family_pairs = [pair for pair in pairs if pair[0].startswith(family + "/")]
        _assert(len(family_pairs) >= 2, f"missing training cohort family {family}")
        chosen.extend(family_pairs[:2])
    chosen.extend([pair for pair in pairs if pair not in chosen][:8 - len(chosen)])
    cases = [D.load_case(
        cohort, name, stats, target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version,
        timesteps=cfg.data.timesteps, extra_point_features=C.v6_point_features(cfg),
        point_features_root=cfg.data.point_features_root,
    ) for cohort, name in chosen]
    _assert(len(cases) == 8, "expected eight real training cases")
    return cases


def _input_fingerprints(cfg, cases):
    paths = [ANCHOR_RUN / "feature_stats.json", ANCHOR_RUN / "weight_quantiles.json",
             ANCHOR_RUN / "ckpt_best.pt", Path(cfg.data.wss_stats_path),
             Path(cfg.data.split_path), LEGACY]
    for case in cases:
        paths.extend([
            Path(case["bundle_path"]),
            Path(cfg.data.point_features_root) / case["cohort"] / case["case"] / "features.npz",
        ])
    return {str(path): sha(path) for path in sorted(set(paths))}


def _legacy_check(cfg, samples):
    """Compare actual historical code with current code, using actual M2 weights."""
    spec = importlib.util.spec_from_file_location("training_wss_min._m2_preflight_legacy", LEGACY)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    seed_all(cfg.train.seed)
    old = legacy.build_baseline_model(cfg.model, C.input_dim(cfg)).eval()
    seed_all(cfg.train.seed)
    new = build_model(cfg.model, C.input_dim(cfg)).eval()
    old_state, new_state = old.state_dict(), new.state_dict()
    _assert(old_state.keys() == new_state.keys(), "historical M2 state keys changed")
    _assert(all(torch.equal(value, new_state[key]) for key, value in old_state.items()),
            "historical M2 initialization changed")
    del old_state, new_state
    # CPU uses one full 5000/5000 case; GPU uses the complete real eight-case batch.
    cpu_batch = D.collate(samples[:1])
    with torch.no_grad():
        expected, actual = _forward(old, cpu_batch, cfg), _forward(new, cpu_batch, cfg)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        cpu_initial_max_abs = float((actual - expected).abs().max())
    gpu_batch = _to_device(D.collate(samples), "cuda")
    old, new = old.cuda(), new.cuda()
    checks = {}
    checkpoint = torch.load(ANCHOR_RUN / "ckpt_best.pt", map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    with torch.no_grad():
        for stage in ("initialization", "checkpoint"):
            if stage == "checkpoint":
                old.load_state_dict(state, strict=True)
                new.load_state_dict(state, strict=True)
            expected = _forward(old, gpu_batch, cfg)
            actual = _forward(new, gpu_batch, cfg)
            repeat = _forward(old, gpu_batch, cfg)
            _assert(bool(torch.isfinite(expected).all() & torch.isfinite(actual).all()),
                    "non-finite historical GPU comparison")
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            checks[stage] = {
                "max_abs": float((actual - expected).abs().max()),
                "historical_repeat_max_abs": float((repeat - expected).abs().max()),
            }
        old, new = old.cpu(), new.cpu()
        expected, actual = _forward(old, cpu_batch, cfg), _forward(new, cpu_batch, cfg)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        cpu_checkpoint_max_abs = float((actual - expected).abs().max())
    result = {
        "passed": True, "initialization_state_exact": True,
        "checkpoint_strict_load": True,
        "cpu": {"cases": 1, "support_per_case": 5000, "query_per_case": 5000,
                "rtol": 0.0, "atol": 0.0, "initialization_max_abs": cpu_initial_max_abs,
                "checkpoint_max_abs": cpu_checkpoint_max_abs},
        "gpu": {"cases": 8, "rtol": 2e-5, "atol": 2e-6, **checks},
    }
    del old, new, checkpoint, state, gpu_batch, expected, actual, repeat
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _initialization_check(model, evidence, reference_state):
    state = model.state_dict()
    differences = [key for key, value in reference_state.items()
                   if key in state and value.shape == state[key].shape
                   and not torch.equal(value, state[key])]
    _assert(not differences, f"common M2 initialization changed: {differences}")
    _assert(not evidence["trained_checkpoint_loaded"], "preflight must start from fresh weights")
    components = {}
    for prefix in ("local_query.correction.", "local_query.beta", "local_wall_branch.modulation."):
        selected = {key: value for key, value in state.items() if key.startswith(prefix)}
        if selected:
            components[prefix] = _tensor_hash(selected)
    return components


def _train_check(model, cfg, cpu_batch, stats, new_keys):
    batch = _to_device(cpu_batch, "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # LambdaLR applies the same epoch-zero warmup multiplier as the real trainer.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(cfg.train))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    attempts, successful = [], 0
    model.train()
    while successful < SUCCESSFUL_STEPS and len(attempts) < MAX_AMP_ATTEMPTS:
        attempt_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        components = {}
        with torch.amp.autocast("cuda", enabled=cfg.train.amp):
            prediction = _forward(model, batch, cfg)
            _assert(bool(torch.isfinite(prediction).all()), "non-finite AMP forward output")
            loss = compute_loss(prediction, batch, cfg.train, "cuda", stats, components=components)
        _assert(bool(torch.isfinite(loss)), "non-finite training loss")
        _assert(all(bool(torch.isfinite(value)) for value in components.values()),
                "non-finite loss component")
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad_gradients = [name for name, parameter in model.named_parameters()
                         if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
        finite = not bad_gradients
        gradient_norm, new_gradient_norm = None, None
        if finite:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip if cfg.train.grad_clip > 0 else float("inf"))
            _assert(bool(torch.isfinite(grad_norm)), "non-finite unscaled gradient norm")
            gradient_norm = float(grad_norm)
            new_gradient_norm = sum(float(parameter.grad.detach().float().norm())
                                    for name, parameter in model.named_parameters()
                                    if name in new_keys and parameter.grad is not None)
        elif not scaler.is_enabled():
            raise FloatingPointError(f"non-finite gradients without AMP scaling: {bad_gradients}")
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if not finite:
            _assert(scale_after < scale_before, "AMP overflow did not trigger loss-scale backoff")
        else:
            successful += 1
        _assert(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
                "optimizer produced non-finite parameters")
        torch.cuda.synchronize()
        attempts.append({
            "attempt": len(attempts) + 1, "optimizer_step_applied": finite,
            "loss": float(loss), "components": {key: float(value) for key, value in components.items()},
            "gradients_finite_after_unscale": finite, "grad_norm_preclip": gradient_norm,
            "new_or_resized_gradient_norm_after_clip": new_gradient_norm,
            "nonfinite_gradient_parameters": bad_gradients,
            "amp_scale_before": scale_before, "amp_scale_after": scale_after,
            "seconds": time.monotonic() - attempt_started,
        })
        del prediction, loss, components
    _assert(successful == SUCCESSFUL_STEPS,
            f"only {successful} finite optimizer steps in {len(attempts)} AMP attempts")
    if new_keys:
        _assert(any((attempt["new_or_resized_gradient_norm_after_clip"] or 0) > 0 for attempt in attempts),
                "new or resized model components received no gradient")
    result = {
        "passed": True, "successful_optimizer_steps": successful,
        "amp_overflow_steps": len(attempts) - successful,
        "learning_rate_epoch_zero": float(optimizer.param_groups[0]["lr"]),
        "seconds": time.monotonic() - started, "attempts": attempts, **_memory(),
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler, batch
    return result


@torch.no_grad()
def _evaluation_check(model, cfg, case, feature_stats):
    """Keep one encoded support and compare real complete-cloud query chunks."""
    model.eval()
    rows = D.query_rows(case)
    _assert(len(rows) >= 16384, "evaluation case must exercise the full maximum query chunk")
    _assert(cfg.eval.fixed_support and cfg.eval.full_cloud_query, "M2 evaluation protocol changed")
    _assert(cfg.eval.query_chunk_size == 16384, "expected the M2 maximum query chunk of 16384")
    seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
    support_idx = D.sample_support_indices(
        case, cfg.data, seed, n_points=5000,
        sampling=cfg.data.support_sampling or cfg.data.sampling, stream="eval_support")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    support_pos = torch.from_numpy(np.ascontiguousarray(case["pos"][support_idx])).cuda()
    support_x = torch.from_numpy(D.build_features(case, support_idx, cfg.data.input_features, feature_stats)).cuda()
    support_batch = torch.zeros(len(support_idx), dtype=torch.long, device="cuda")
    calls_before = model.support_encode_calls
    encoded = model.encode_support(support_pos, support_x, support_batch,
                                   unit_ids=[case["unit_id"]], epoch=0,
                                   global_seed=cfg.eval.support_seed, evaluation=True)
    predictions, chunk_counts = [], []
    for chunk_size in (16384, 8192):
        outputs = []
        for start in range(0, len(rows), chunk_size):
            idx = rows[start:start + chunk_size]
            pos = torch.from_numpy(np.ascontiguousarray(case["pos"][idx])).cuda()
            x = torch.from_numpy(D.build_features(case, idx, cfg.data.input_features, feature_stats)).cuda()
            batch = torch.zeros(len(idx), dtype=torch.long, device="cuda")
            output = model.decode_query(encoded, pos, x, batch)
            _assert(bool(torch.isfinite(output).all()), "non-finite complete-cloud evaluation output")
            outputs.append(output.detach().float().cpu())
            del pos, x, batch, output
        predictions.append(torch.cat(outputs))
        chunk_counts.append(len(outputs))
    _assert(model.support_encode_calls - calls_before == 1, "query chunks re-encoded support")
    torch.testing.assert_close(predictions[0], predictions[1], rtol=2e-5, atol=2e-6)
    diagnostics = getattr(model.local_wall_branch, "last_modulation_diagnostics", {})
    _assert(all(bool(torch.isfinite(value)) for value in diagnostics.values()), "non-finite modulation diagnostics")
    return {
        "passed": True, "case": case["unit_id"], "full_query_points": int(len(rows)),
        "support_points": int(len(support_idx)), "query_chunk_sizes": [16384, 8192],
        "query_chunk_counts": chunk_counts, "support_encode_calls": 1, "amp": False,
        "chunk_comparison_max_abs": float((predictions[0] - predictions[1]).abs().max()),
        "chunk_comparison_rtol": 2e-5, "chunk_comparison_atol": 2e-6,
        "modulation_diagnostics": {key: float(value) for key, value in diagnostics.items()},
        "seconds": time.monotonic() - started, **_memory(),
    }


def _paired_arm_checks(records):
    results = {}
    for left, right, prefixes in (
        ("MO-S1", "MO-S2", ("local_query.correction.", "local_query.beta")),
        ("MO-S3", "MO-S4", ("local_wall_branch.modulation.",)),
    ):
        if not all(records.get(arm, {}).get("passed") for arm in (left, right)):
            continue
        for prefix in prefixes:
            _assert(records[left]["initial_component_sha256"][prefix]
                    == records[right]["initial_component_sha256"][prefix],
                    f"initialization mismatch between {left} and {right}: {prefix}")
        results[f"{left}/{right}"] = True
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", help="Subset IDs; only same-source completed evidence may be reused")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use Slurm for the M2 GPU preflight; this tool does not submit jobs")
    if not torch.cuda.is_available():
        raise RuntimeError("M2 runtime preflight requires an allocated CUDA device")
    torch.set_num_threads(2)
    matrix = json.loads(MATRIX.read_text())
    expected_ids = [arm["id"] for arm in matrix["arms"]]
    _assert(len(expected_ids) == 20 and len(set(expected_ids)) == 20, "expected the complete 20-arm base matrix")
    requested = {part for value in args.arms or [] for part in value.split(",")}
    _assert(not (requested - set(expected_ids)), f"unknown preflight arms: {requested - set(expected_ids)}")
    selected = requested or set(expected_ids)
    start_hashes = fingerprints()
    anchor = C.ExpConfig.from_json(ANCHOR_CONFIG)
    stats = D.load_wss_stats(anchor.data.wss_stats_path)
    cases = _load_cases(anchor, stats)
    feature_stats = json.loads((ANCHOR_RUN / "feature_stats.json").read_text())
    quantiles = json.loads((ANCHOR_RUN / "weight_quantiles.json").read_text())
    _assert(np.isfinite(quantiles["raw_p90"]) and quantiles["raw_p90"] > 0, "invalid historical train138 p90")
    input_hashes = _input_fingerprints(anchor, cases)
    dataset = D.WSSMinDataset(cases, anchor.data, feature_stats, training=True, base_seed=1234)
    samples = [dataset[index] for index in range(8)]
    _assert(all(len(sample["support_pos"]) == len(sample["pos"]) == 5000 for sample in samples),
            "expected exactly 5000 independent support and query points per case")
    cpu_batch = D.collate(samples)
    evaluation_case = max(cases, key=lambda case: len(D.query_rows(case)))
    prior = json.loads(OUTPUT.read_text()) if args.arms and OUTPUT.is_file() else {}
    can_reuse = (prior.get("source_sha256") == start_hashes
                 and prior.get("source_sha256_end") == start_hashes
                 and prior.get("input_sha256") == input_hashes
                 and prior.get("input_sha256_end") == input_hashes)
    records = copy.deepcopy(prior.get("arms", {})) if can_reuse else {}
    report = {
        "passed": False, "started_at": stamp(), "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": os.uname().nodename, "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_arms": [aid for aid in expected_ids if aid in selected], "expected_arms": expected_ids,
        "reused_same_source_evidence": can_reuse, "cases": [case["unit_id"] for case in cases],
        "batch_cases": 8, "support_per_case": 5000, "query_per_case": 5000,
        "statistics_source": "unchanged full train138 M2 artifacts; never recomputed from these eight cases",
        "raw_p90": quantiles["raw_p90"], "source_sha256": start_hashes,
        "input_sha256": input_hashes, "arms": records,
    }
    save_json(OUTPUT, report)
    try:
        previous_legacy = prior.get("legacy_compatibility", {}) if can_reuse else {}
        report["legacy_compatibility"] = (previous_legacy if previous_legacy.get("passed")
                                          else _legacy_check(anchor, samples))
        seed_all(1234)
        reference = build_model(anchor.model, C.input_dim(anchor))
        reference_state = {key: value.clone() for key, value in reference.state_dict().items()}
        del reference
        save_json(OUTPUT, report)
        for arm in matrix["arms"]:
            aid = arm["id"]
            if aid not in selected:
                continue
            print(f"{stamp()} preflight {aid} starting", flush=True)
            arm_started = time.monotonic()
            model = None
            try:
                cfg = C.ExpConfig.from_json(CONFIGS / arm["config"])
                for key in ("data_root", "split_path", "wss_stats_path", "point_features_root", "input_features",
                            "support_n_points", "query_n_points", "query_mode", "target", "target_normalization"):
                    _assert(getattr(cfg.data, key) == getattr(anchor.data, key), f"{aid} changes frozen data field {key}")
                _assert(cfg.data == anchor.data,
                        f"{aid} changes M2 data configuration or train138 feature-stat recomputation")
                for key, value in quantiles.items():
                    if key != "raw_p90":
                        setattr(cfg.train, key, value)
                cfg.train._raw_p90 = quantiles["raw_p90"]
                model, initialization = build_paired_model(cfg)
                initial_components = _initialization_check(model, initialization, reference_state)
                parameters = sum(parameter.numel() for parameter in model.parameters())
                new_keys = set(initialization["new_keys"] + initialization["changed_shape_keys"])
                model.cuda()
                train = _train_check(model, cfg, cpu_batch, stats, new_keys)
                gc.collect()
                torch.cuda.empty_cache()
                evaluation = _evaluation_check(model, cfg, evaluation_case, feature_stats)
                records[aid] = {
                    "passed": True, "job_id": os.environ["SLURM_JOB_ID"], "config_sha256": sha(CONFIGS / arm["config"]),
                    "parameters": parameters, "initialization": initialization,
                    "common_initialization_exact": True, "initial_component_sha256": initial_components,
                    "train": train, "eval": evaluation,
                    "max_memory_allocated_mib": max(train["max_memory_allocated_mib"], evaluation["max_memory_allocated_mib"]),
                    "max_memory_reserved_mib": max(train["max_memory_reserved_mib"], evaluation["max_memory_reserved_mib"]),
                    "seconds": time.monotonic() - arm_started,
                }
                print(f"{aid} passed; reserved={records[aid]['max_memory_reserved_mib']:.1f} MiB; "
                      f"AMP backoffs={train['amp_overflow_steps']}", flush=True)
            except Exception as exc:
                records[aid] = {"passed": False, "job_id": os.environ["SLURM_JOB_ID"],
                                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                                "seconds": time.monotonic() - arm_started}
                print(f"{aid} FAILED: {records[aid]['error']}", flush=True)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                save_json(OUTPUT, report)
        report["paired_arm_initialization_checks"] = _paired_arm_checks(records)
        report["source_sha256_end"] = fingerprints()
        report["input_sha256_end"] = _input_fingerprints(anchor, cases)
        _assert(report["source_sha256_end"] == start_hashes, "source changed during runtime preflight")
        _assert(report["input_sha256_end"] == input_hashes, "data or historical artifacts changed during preflight")
        report["passed"] = (report["legacy_compatibility"]["passed"]
                            and set(records) == set(expected_ids)
                            and all(records[aid].get("passed") for aid in expected_ids)
                            and len(report["paired_arm_initialization_checks"]) == 2)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["completed_at"] = stamp()
        save_json(OUTPUT, report)
    if not report["passed"]:
        raise SystemExit("Preflight incomplete or failed; inspect runtime_preflight.json. Full gate requires all 20 arms.")


if __name__ == "__main__":
    main()
