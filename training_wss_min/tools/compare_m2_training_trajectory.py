"""Read-only old/repeat/new M2 training arithmetic comparison on one fixed batch.

Models/optimizers exist only in memory. Frozen training code and trained runs are
never modified. GPU execution requires Slurm; --self-test uses small CPU inputs.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from training_wss_min import baseline_models as current_models
from training_wss_min import config as C, dataset as D
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.runtime import lr_lambda_factory, seed_all
from training_wss_min.tools.m2_optimization_common import ANCHOR_CONFIG, ANCHOR_RUN, CONFIGS, EXP, ROOT, fingerprints, save_json, sha, stamp


LEGACY = ROOT / "training_wss_min/experiments/v6_followup_20260909/source_snapshot_13205/training_wss_min"
LABELS = ("old_reference", "old_repeat", "new_log_off", "new_log_on", "new_paired_log_on")


def import_old(name):
    spec = importlib.util.spec_from_file_location(f"training_wss_min._trajectory_old_{name}", LEGACY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_rng(device):
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else []}


def restore_rng(state, device, libc_seed):
    torch.set_rng_state(state["cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    else:
        # torch_cluster CPU FPS uses libc rand(), not the Torch generator.
        ctypes.CDLL(None).srand(libc_seed)


def rng_hash(state):
    return {"cpu": hashlib.sha256(state["cpu"].numpy().tobytes()).hexdigest(),
            "cuda": [hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in state["cuda"]]}


def difference(a, b):
    if tuple(a.shape) != tuple(b.shape):
        return {"same_shape": False, "shape_reference": list(a.shape), "shape_candidate": list(b.shape)}
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite].double() - b[finite].double()).abs()
    return {"same_shape": True, "exact": bool(torch.equal(a, b)),
            "max_abs": float(delta.max()) if delta.numel() else None,
            "mean_abs": float(delta.mean()) if delta.numel() else None,
            "relative_l2": float(delta.norm() / a[finite].double().norm().clamp_min(1e-30)) if delta.numel() else None,
            "different_elements": int((a != b).sum()),
            "nonfinite_reference": int((~torch.isfinite(a)).sum()),
            "nonfinite_candidate": int((~torch.isfinite(b)).sum())}


def tree_difference(a, b, limit=None):
    mismatched_keys = sorted(set(a) ^ set(b))
    rows = {key: difference(a[key], b[key]) for key in a.keys() & b.keys()}
    changed = {key: value for key, value in rows.items() if not value.get("exact", False)}
    largest = sorted(changed, key=lambda key: (changed[key].get("max_abs") or 0.), reverse=True)
    return {"key_difference": mismatched_keys, "tensors_checked": len(rows), "nonexact_tensors": len(changed),
            "largest_max_abs": max((row.get("max_abs") or 0.) for row in rows.values()) if rows else 0.,
            "differences": {key: changed[key] for key in largest[:limit] if key in changed} if limit else rows}


def synthetic_batch():
    gen = torch.Generator().manual_seed(357)
    n, q, graphs = 192, 137, 2
    return {"support_pos": torch.rand(n * graphs, 3, generator=gen) * .2,
            "support_x": torch.randn(n * graphs, 25, generator=gen),
            "support_batch": torch.arange(graphs).repeat_interleave(n),
            "pos": torch.rand(q * graphs, 3, generator=gen) * .2,
            "x": torch.randn(q * graphs, 25, generator=gen),
            "batch": torch.arange(graphs).repeat_interleave(q),
            "y": torch.randn(q * graphs, generator=gen), "y_raw": torch.rand(q * graphs, generator=gen) * 10,
            "unit_ids": [f"synthetic/case/{i}" for i in range(graphs)]}


def real_batch(cfg):
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    ids = json.loads((EXP / "runtime_preflight.json").read_text())["cases"]
    cases = []
    for uid in ids:
        cohort, name = uid.rsplit("/", 1)
        cases.append(D.load_case(cohort, name, stats, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version, extra_point_features=C.v6_point_features(cfg),
            point_features_root=cfg.data.point_features_root))
    feature_stats = json.loads((ANCHOR_RUN / "feature_stats.json").read_text())
    dataset = D.WSSMinDataset(cases, cfg.data, feature_stats, training=True, base_seed=cfg.train.seed)
    return D.collate([dataset[i] for i in range(len(cases))]), stats


def compare(device, steps, self_test=False, amp=True):
    old_models, old_objectives = import_old("baseline_models"), import_old("objectives")
    cfg = C.ExpConfig.from_json(ANCHOR_CONFIG)
    candidate_cfg = C.ExpConfig.from_json(CONFIGS / "MO0_s1234.json")
    if self_test:
        batch, stats = synthetic_batch(), D.load_wss_stats(cfg.data.wss_stats_path)
    else:
        batch, stats = real_batch(cfg)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    records, models, optimizers, scalers, constructor_rng = {}, {}, {}, {}, {}
    for label in LABELS:
        seed_all(cfg.train.seed)
        if label == "new_paired_log_on":
            model, evidence = build_paired_model(candidate_cfg)
            records["paired_initialization"] = evidence
        elif label.startswith("old_"):
            model = old_models.build_baseline_model(cfg.model, C.input_dim(cfg))
        else:
            model = build_model(cfg.model, C.input_dim(cfg))
        constructor_rng[label] = rng_hash(capture_rng(device))
        models[label] = model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(cfg.train))
        optimizers[label] = optimizer
        scalers[label] = torch.amp.GradScaler(device, enabled=amp)
    initial_state = {label: state_hash(model.state_dict()) for label, model in models.items()}
    if len(set(initial_state.values())) != 1:
        raise RuntimeError(f"initial models differ: {initial_state}")
    records.update(initial_state_sha256=initial_state, constructor_rng=constructor_rng,
                   parameters=sum(p.numel() for p in models["old_reference"].parameters()),
                   input_tensors_sha256=state_hash({key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}),
                   cases=batch["unit_ids"], steps=[], device=device, amp=amp,
                   amp_dtype="float16" if amp else "float32",
                   batch_protocol="one fixed identical real batch for all comparisons; no fit or quality estimate")
    rng = capture_rng(device)
    active = {"label": None, "fps": []}
    def trace_fps(original):
        def wrapped(*args, **kwargs):
            before = rng_hash(capture_rng(device))
            output = original(*args, **kwargs)
            active["fps"].append({"random_start": bool(kwargs.get("random_start", True)),
                "source_points": len(args[0]), "selected_points": len(output),
                "first_index": int(output[0]), "indices_sha256": state_hash({"indices": output}),
                "rng_before": before, "rng_after": rng_hash(capture_rng(device))})
            return output
        return wrapped
    with patch.object(old_models, "fps", trace_fps(old_models.fps)), patch.object(current_models, "fps", trace_fps(current_models.fps)):
        for step in range(steps):
            step_data, tensors, reference_rng = {}, {}, None
            for label in LABELS:
                model, optimizer, scaler = models[label], optimizers[label], scalers[label]
                optimizer.zero_grad(set_to_none=True)
                restore_rng(rng, device, 8719 + step)
                active.update(label=label, fps=[])
                outputs, hooks = {}, []
                def capture(name):
                    def hook(module, args, output):
                        value = output[1] if isinstance(output, tuple) else output
                        outputs[name] = value.detach().cpu()
                    return hook
                for name, module in model.named_modules():
                    if name in {"stem", "sa.0", "sa.1", "sa.2", "sa.1.local_transformer", "fp.0", "fp.1", "fp.2", "local_wall_branch", "head"}:
                        hooks.append(module.register_forward_hook(capture(name)))
                started = time.monotonic()
                with torch.autocast(device, dtype=torch.float16, enabled=amp):
                    prediction = model.forward_support_query(batch["support_pos"], batch["support_x"], batch["support_batch"],
                        batch["pos"], batch["x"], batch["batch"], unit_ids=batch["unit_ids"],
                        epoch=0, global_seed=cfg.train.seed, evaluation=False)
                    components = {} if label.endswith("log_on") else None
                    if label.startswith("old_"):
                        loss = old_objectives.compute_loss(prediction, batch, cfg.train, device, stats)
                    else:
                        loss = compute_loss(prediction, batch, cfg.train, device, stats, components=components)
                for handle in hooks:
                    handle.remove()
                if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite forward at {label}/{step}")
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                finite = bool(torch.isfinite(norm))
                # Mirror the trainer: both old and current clip before scaler.step.
                scaler.step(optimizer)
                scaler.update()
                end_rng = capture_rng(device)
                if label == "old_reference":
                    reference_rng = end_rng
                masks = {f"sa.{i}.blocks.{j}": block.last_drop_path_graph_scale.cpu().clone()
                         for i, sa in enumerate(model.sa) for j, block in enumerate(sa.blocks)}
                centers = {f"sa.{i}": sa.last_indices.cpu().clone() for i, sa in enumerate(model.sa)}
                tensors[label] = {"layers": outputs, "gradients": gradients,
                    "parameters_after": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "drop_path_masks": masks, "fps_indices": centers}
                if device == "cuda":
                    torch.cuda.synchronize()
                step_data[label] = {"loss": float(loss), "components": {key: float(value) for key, value in (components or {}).items()},
                    "grad_norm_preclip": float(norm) if finite else None, "gradient_norm_finite": finite,
                    "amp_scale_before": scale_before, "amp_scale_after": float(scaler.get_scale()),
                    "optimizer_step_applied": float(scaler.get_scale()) >= scale_before,
                    "rng_after": rng_hash(end_rng), "fps_calls": active["fps"], "seconds": time.monotonic() - started}
                del prediction, loss
            reference = tensors["old_reference"]
            comparisons = {}
            for label in LABELS[1:]:
                comparisons[label] = {group: tree_difference(reference[group], tensors[label][group],
                    limit=12 if group in {"gradients", "parameters_after"} else None) for group in reference}
                comparisons[label]["same_rng_after"] = step_data[label]["rng_after"] == step_data["old_reference"]["rng_after"]
                comparisons[label]["same_fps_rng_trace"] = step_data[label]["fps_calls"] == step_data["old_reference"]["fps_calls"]
                comparisons[label]["loss_delta"] = step_data[label]["loss"] - step_data["old_reference"]["loss"]
                comparisons[label]["same_scaler_decision"] = step_data[label]["amp_scale_after"] == step_data["old_reference"]["amp_scale_after"]
            records["steps"].append({"index": step, "rng_before": rng_hash(rng), "models": step_data, "comparisons": comparisons})
            rng = reference_rng
            print(json.dumps({"step": step, "loss": {label: value["loss"] for label, value in step_data.items()},
                "reference_scale": step_data["old_reference"]["amp_scale_after"],
                "output_max_abs": {label: comp["layers"]["differences"]["head"]["max_abs"] for label, comp in comparisons.items()},
                "same_fps": {label: comp["same_fps_rng_trace"] for label, comp in comparisons.items()}}, ensure_ascii=False), flush=True)
            del tensors
    records["successful_steps"] = {label: sum(step["models"][label]["optimizer_step_applied"] for step in records["steps"]) for label in LABELS}
    records["same_stochastic_choices_all_steps"] = all(comp["same_rng_after"] and comp["same_fps_rng_trace"]
        and comp["drop_path_masks"]["nonexact_tensors"] == 0 for step in records["steps"] for comp in step["comparisons"].values())
    records["all_layer_outputs_exact"] = all(comp["layers"]["nonexact_tensors"] == 0 for step in records["steps"] for comp in step["comparisons"].values())
    records["all_unscaled_gradients_exact"] = all(comp["gradients"]["nonexact_tensors"] == 0 for step in records["steps"] for comp in step["comparisons"].values())
    records["interpretation"] = "old_repeat quantifies repeat differences under the same RNG; old/new excess and first differing layer remain evidence, not automatic attribution of final R² differences"
    del models, optimizers, scalers
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1 if args.self_test else 2)
    if args.self_test:
        result = compare("cpu", min(args.steps, 3), self_test=True, amp=not args.fp32)
        if not result["same_stochastic_choices_all_steps"] or not result["all_layer_outputs_exact"]:
            raise RuntimeError("CPU old/repeat/new layer/RNG parity failed")
        print("CPU trajectory self-test passed", flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("GPU trajectory comparison requires Slurm and an allocated GPU")
    before = fingerprints()
    path = EXP / "training_trajectory" / f"job_{os.environ['SLURM_JOB_ID']}_{'fp32' if args.fp32 else 'amp'}.json"
    result = {"completed": False, "started_at": stamp(), "job_id": os.environ["SLURM_JOB_ID"],
              "script_sha256": sha(Path(__file__)), "source_sha256": before,
              "legacy_model_sha256": sha(LEGACY / "baseline_models.py"), "legacy_objectives_sha256": sha(LEGACY / "objectives.py")}
    save_json(path, result)
    try:
        result["comparison"] = compare("cuda", args.steps, amp=not args.fp32)
        result["source_sha256_end"] = fingerprints()
        if before != result["source_sha256_end"]:
            raise RuntimeError("frozen sources changed during trajectory comparison")
        result.update(completed=True, completed_at=stamp())
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", failed_at=stamp())
        raise
    finally:
        save_json(path, result)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
