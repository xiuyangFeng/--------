"""GPU preflight for the wave-1 matrix: every arm at batch8 x 5000, anchor re-evaluation, CLI smokes.

Runs inside a Slurm GPU allocation and writes ``runtime_preflight.json`` that the queue
verifies (fingerprints, arm coverage, anchor compatibility) before launching anything.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.evaluate import evaluate_partition, load_model_from_run, load_wss_stats_for_run, predict_case_norm
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.runtime import seed_all
from training_wss_min.tools.run_wss_local_wave1_queue import CONFIGS, EXP, ROOT, fingerprints

RUNS = ROOT / "training_wss_min/runs"
ANCHOR_RUN = RUNS / "wss_direct_recovery_20260912/C1_s1234"
SMOKE_ARMS = ("X16", "X11", "X13a")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_cases(cfg, pairs):
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    return [D.load_case(cohort, name, stats, target=cfg.data.target,
                        target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                        required_frame_version=cfg.data.required_frame_version,
                        timesteps=cfg.data.timesteps, waveform_path=cfg.data.waveform_path,
                        extra_point_features=C.v6_point_features(cfg),
                        point_features_root=cfg.data.point_features_root,
                        time_basis_path=getattr(cfg.data, "time_basis_path", None),
                        time_basis_k=int(getattr(cfg.model, "time_basis_k", 0)),
                        volume_time_sidecar_root=getattr(cfg.data, "volume_time_sidecar_root", None),
                        volume_h5_root=getattr(cfg.data, "volume_h5_root", None),
                        cycle_view_root=getattr(cfg.data, "cycle_view_root", None)) for cohort, name in pairs]


def forward(model, batch, training=True):
    return model.forward_support_query(batch["support_pos"].cuda(), batch["support_x"].cuda(),
        batch["support_batch"].cuda(), batch["pos"].cuda(), batch["x"].cuda(), batch["batch"].cuda(),
        unit_ids=batch["unit_ids"], epoch=0, global_seed=1234, evaluation=not training,
        **D.local_model_context(batch, "cuda"))


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from numeric_leaves(item, f"{prefix}[{index}]")
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        yield prefix, float(value)


def anchor_reevaluation(tolerance=5e-4, anchor_run=None):
    """Re-evaluate the anchor's best checkpoint (default C1) with the current code and compare every number."""
    anchor_run = Path(anchor_run) if anchor_run else ANCHOR_RUN
    cfg, feat_stats, model, _ = load_model_from_run(anchor_run, "cuda", "best")
    stats = load_wss_stats_for_run(anchor_run)
    cases = D.load_partition(cfg.data.split_path, "test", stats, target=cfg.data.target,
                             target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                             required_frame_version=cfg.data.required_frame_version,
                             extra_point_features=C.v6_point_features(cfg),
                             point_features_root=cfg.data.point_features_root)
    started = time.monotonic()
    result = evaluate_partition(model, cases, cfg, feat_stats, stats, "cuda", make_plots=False)
    stored = json.loads((anchor_run / "eval/ckpt_best/metrics.json").read_text())["test"]
    stored_leaves = dict(numeric_leaves(stored))
    new_leaves = dict(numeric_leaves(result))
    skipped = {key for key in stored_leaves if key.startswith("efficiency") or key.startswith("evaluation_split")}
    common = [key for key in stored_leaves if key in new_leaves and key not in skipped]
    worst = max(((abs(stored_leaves[k] - new_leaves[k]), k) for k in common
                 if math.isfinite(stored_leaves[k]) and math.isfinite(new_leaves[k])), default=(0.0, ""))
    missing = sorted(set(stored_leaves) - set(new_leaves) - skipped)
    evidence = {"anchor_run": str(anchor_run), "compared_fields": len(common), "max_abs_diff": worst[0], "max_abs_diff_field": worst[1],
                "missing_fields": missing[:20], "tolerance": tolerance,
                "pa_r2_cb_stored": stored["field_casebalanced"]["r2"],
                "pa_r2_cb_now": result["field_casebalanced"]["r2"],
                "seconds": time.monotonic() - started}
    evidence["passed"] = bool(worst[0] <= tolerance and not missing and len(common) > 1000)
    del model
    return evidence


def main(argv=None):
    global CONFIGS, EXP, SMOKE_ARMS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=CONFIGS)
    parser.add_argument("--experiment-dir", type=Path, default=EXP)
    parser.add_argument("--smoke-arms", default=",".join(SMOKE_ARMS))
    parser.add_argument("--control-arm", default="X0", help="arm config used to pick the smoke cases/split")
    parser.add_argument("--anchor-run", type=Path, default=ANCHOR_RUN,
                        help="trained run whose stored best metrics must be reproduced by the current code (default C1)")
    args = parser.parse_args(argv)
    CONFIGS, EXP = Path(args.config_dir), Path(args.experiment_dir)
    SMOKE_ARMS = tuple(a for a in args.smoke_arms.split(",") if a)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU allocation required")
    torch.set_num_threads(4)
    frozen = fingerprints(CONFIGS)
    evidence = dict(passed=False, slurm_job_id=os.environ["SLURM_JOB_ID"], fingerprints=frozen, arms={},
                    cli_smoke={}, anchor_reevaluation=None, started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save_json(EXP / "runtime_preflight_progress.json", evidence)

    # 1) anchor compatibility: C1 best must reproduce its stored metrics with the new code
    evidence["anchor_reevaluation"] = anchor_reevaluation(anchor_run=args.anchor_run)
    save_json(EXP / "runtime_preflight_progress.json", evidence)
    print("anchor re-evaluation", evidence["anchor_reevaluation"], flush=True)
    if not evidence["anchor_reevaluation"]["passed"]:
        raise RuntimeError("anchor re-evaluation differs from the stored C1 metrics")
    gc.collect(); torch.cuda.empty_cache()

    # 2) every arm: paired construction, three finite AMP steps on a real batch, full-wall inference
    control_path = CONFIGS / f"{args.control_arm}.json"
    if not control_path.is_file():
        control_path = CONFIGS / f"{args.control_arm}_s1234.json"
    base_cfg = C.ExpConfig.from_json(control_path)
    pairs = D.load_split_cases(base_cfg.data.split_path, "train")
    chosen = []
    for family in ("AG/", "AAA/", "ILO/"):
        chosen.extend([p for p in pairs if p[0].startswith(family)][:2])
    chosen.extend([p for p in pairs if p not in chosen][:8 - len(chosen)])
    case_cache = {}
    for path in sorted(CONFIGS.glob("*_s*.json")):
        cfg = C.ExpConfig.from_json(path)
        # cases depend on the target too (pressure_mixed vs velocity views share stats and features)
        cache_key = (cfg.data.target, cfg.data.timesteps, tuple(cfg.data.input_features), cfg.data.wss_stats_path,
                     str(cfg.data.point_features_root), float(cfg.data.query_wall_fraction),
                      cfg.data.target_normalization, getattr(cfg.data, "time_basis_path", None), int(getattr(cfg.model, "time_basis_k", 0)),
                      getattr(cfg.data, "volume_time_sidecar_root", None), getattr(cfg.data, "volume_h5_root", None),
                      getattr(cfg.data, "cycle_view_root", None))
        if cache_key not in case_cache:
            case_cache[cache_key] = load_cases(cfg, chosen)
        cases = case_cache[cache_key]
        feature_stats = json.loads(Path(cfg.data.feature_stats_path).read_text())
        ds = D.WSSMinDataset(cases, cfg.data, feature_stats, training=True, base_seed=1234)
        batch = D.collate([ds[i] for i in range(8)])
        region_feature = getattr(cfg.train, "loss_region_feature", None)
        if region_feature:  # T3 focus region: the trainer derives it from the standardised input column
            batch["loss_region_value"] = batch["x"][:, list(cfg.data.input_features).index(region_feature)]
        seed_all(1234)
        if cfg.train.init_reference_config:
            model, initial = build_paired_model(cfg)
        else:
            # X13b warm-starts from X13a which does not exist yet: exercise the architecture only.
            model, initial = build_model(cfg.model, C.input_dim(cfg)), {"mode": "architecture_only_preflight"}
        model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses, overflows, components = [], 0, {}
        stats = D.load_wss_stats(cfg.data.wss_stats_path)
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = forward(model, batch)
                loss = compute_loss(pred, batch, cfg.train, "cuda", stats, components=components)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all():
                raise RuntimeError(f"nonfinite forward {path.name}: {components}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                losses.append(float(loss.detach()))
            else:
                overflows += 1
            scaler.step(optimizer)
            scaler.update()
            if len(losses) == 3:
                break
        if len(losses) < 3:
            raise RuntimeError(f"AMP never achieved three finite updates: {path.name}")
        model.eval()
        p1 = predict_case_norm(model, cases[0], cfg.data.input_features, feature_stats, "cuda", cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2.eval.query_chunk_size = 4096
        p2 = predict_case_norm(model, cases[0], cfg2.data.input_features, feature_stats, "cuda", cfg2)
        np.testing.assert_allclose(p1, p2, rtol=3e-5, atol=3e-5)
        # volume targets predict on the query pool (interior rows), WSS on every wall row
        if not np.isfinite(p1).all() or len(p1) != len(D.query_rows(cases[0])):
            raise RuntimeError("invalid full-query output")
        torch.cuda.synchronize()
        evidence["arms"][path.name] = dict(
            parameters=sum(p.numel() for p in model.parameters()),
            peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20, seconds=time.monotonic() - started,
            losses=losses, amp_initial_overflows=overflows,
            loss_components={k: float(v) for k, v in components.items()},
            fullwall_points=len(p1), chunk_max_abs_diff=float(np.max(np.abs(p1 - p2))),
            initialization={k: v for k, v in initial.items()
                            if k in ("mode", "initial_state_sha256", "reference_state_sha256", "new_keys",
                                     "zero_extended_input_keys", "appended_input_features")},
            units=batch["unit_ids"])
        save_json(EXP / "runtime_preflight_progress.json", evidence)
        print(path.name, round(evidence["arms"][path.name]["peak_memory_mib"]), losses, flush=True)
        del model, optimizer, scaler, batch, pred, loss, ds
        gc.collect(); torch.cuda.empty_cache()

    # 3) real entrypoints: 2-epoch train + full-wall test evaluation on a tiny split, richest arms
    smoke = EXP / "smoke" / os.environ["SLURM_JOB_ID"]
    smoke.mkdir(parents=True, exist_ok=True)
    split_source = json.loads(Path(base_cfg.data.split_path).read_text())
    split = {"train_cases": [f"{a}/{b}" for a, b in chosen], "val_cases": [],
             "test_cases": [f"{a}/{b}" for a, b in D.load_split_cases(base_cfg.data.split_path, "test")[:2]]}
    save_json(smoke / "split.json", split)
    for aid in SMOKE_ARMS:
        config_path = CONFIGS / f"{aid}.json"
        if not config_path.is_file():
            config_path = CONFIGS / f"{aid}_s1234.json"
        raw = json.loads(config_path.read_text())
        raw["name"] = f"{EXP.name}_smoke/job_{os.environ['SLURM_JOB_ID']}_{aid}"
        raw["data"]["split_path"] = str(smoke / "split.json")
        raw["train"].update(epochs=2, min_epoch=2, eval_every=2)
        if aid == "X13a":
            raw["train"]["ema_decay"] = 0.999   # exercise the EMA checkpoint path in the CLI too
        save_json(smoke / f"{aid}_config.json", raw)
        commands = [[sys.executable, "-u", "-m", "training_wss_min.train", "--config", str(smoke / f"{aid}_config.json")]]
        checkpoints = ["best", "last"] + (["ema"] if raw["train"].get("ema_decay") else [])
        for checkpoint in checkpoints:
            command = [sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir",
                       str(RUNS / raw["name"]), "--checkpoint", checkpoint, "--partitions", "test",
                       "--allow-test", "--no-plots"]
            if raw["data"].get("timesteps", "peak") == "peak":
                command.append("--save-predictions")
            commands.append(command)
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
        metrics = json.loads((RUNS / raw["name"] / "eval/ckpt_best/metrics.json").read_text())["test"]
        evidence["cli_smoke"][aid] = {"run": str(RUNS / raw["name"]), "checkpoints": checkpoints,
                                      "pa_r2_cb": metrics["field_casebalanced"]["r2"]}
        save_json(EXP / "runtime_preflight_progress.json", evidence)
    if fingerprints(CONFIGS) != frozen:
        raise RuntimeError("sources/config changed during GPU preflight")
    evidence["passed"] = True
    evidence["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_json(EXP / "runtime_preflight.json", evidence)
    print("PREFLIGHT PASSED", flush=True)


if __name__ == "__main__":
    main()
