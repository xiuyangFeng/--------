#!/usr/bin/env python3
"""A0D one-variable restoration: fixed target-weight loss only."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import config as C
from . import dataset as D
from .a0d_simple_overfit import evaluate, load_cases, read_json, sha256
from .brep_mlp import _set_quantiles
from .point_mlp import build_point_mlp
from .train import compute_loss, seed_all


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "training_wss_min/configs/round5/a0d_target_weight_only_protocol.json"


def validate(protocol_path: Path):
    protocol = read_json(protocol_path)
    if protocol.get("task") != "A0D_TARGET_WEIGHT_ONLY":
        raise RuntimeError("wrong target-weight protocol")
    paths = {key: Path(value) for key, value in protocol["paths"].items()}
    for key in protocol["sha256"]:
        if not paths[key].is_file() or sha256(paths[key]) != protocol["sha256"][key]:
            raise RuntimeError(f"missing or changed frozen asset: {key}")
    if read_json(paths["control_metrics"]).get("gate") != "GO":
        raise RuntimeError("Job 6990 control did not pass")
    candidate = C.ExpConfig.from_json(paths["config"])
    control = C.ExpConfig.from_json(paths["control_config"])
    cand, ctrl = candidate.to_dict(), control.to_dict()
    cand.pop("name"); cand.pop("notes"); ctrl.pop("name"); ctrl.pop("notes")
    allowed = {
        "loss_weight_target": (False, True),
        "loss_weight_target_alpha": (0.0, 2.0),
        "loss_weight_fixed_quantiles": (False, True),
    }
    for section in ("data", "model"):
        if cand[section] != ctrl[section]:
            raise RuntimeError(f"single-variable violation in {section}")
    for key in set(cand["train"]) | set(ctrl["train"]):
        before, after = ctrl["train"].get(key), cand["train"].get(key)
        if before != after and allowed.get(key) != (before, after):
            raise RuntimeError(f"unexpected train difference: {key}: {before} -> {after}")
    if {key for key in cand["train"] if cand["train"].get(key) != ctrl["train"].get(key)} != set(allowed):
        raise RuntimeError("target-weight restoration differences are incomplete")
    if candidate.train.amp or candidate.train.weight_decay != 0 \
            or candidate.train.grad_clip != 0 or candidate.train.lr != 1e-3 \
            or candidate.train.epochs != 1000:
        raise RuntimeError("simplified optimizer controls changed")
    manifest = read_json(paths["locked_cases"])
    if not manifest.get("locked") or len(manifest["cases"]) != 4:
        raise RuntimeError("locked four-case manifest invalid")
    return protocol, paths, candidate, manifest


def train(protocol_path: Path) -> None:
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("target-weight-only training is cluster GPU / sbatch only")
    protocol, paths, cfg, manifest = validate(protocol_path)
    out = paths["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError("target-weight-only registered result already exists")
    seed_all(cfg.train.seed)
    stats = D.load_wss_stats(paths["stats_snapshot"])
    quantiles = read_json(paths["weight_quantiles_snapshot"])
    _set_quantiles(cfg, quantiles)
    feature_stats = read_json(paths["feature_stats_snapshot"])
    cases = load_cases(manifest["cases"], stats)
    ds = D.WSSMinDataset(cases, cfg.data, feature_stats, training=True,
                         base_seed=cfg.train.seed)
    batch = D.collate([ds[index] for index in range(4)])
    model = build_point_mlp(cfg.model, C.input_dim(cfg)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=0.0)
    (out / "config_snapshot.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out / "runtime.json").write_text(json.dumps({
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "single_variable": "fixed target-weight alpha2",
        "scheduler_created": False, "amp": False, "validation_loaded": False,
        "test_loaded": False,
    }, indent=2) + "\n")
    pos, features, groups, target = (batch["pos"].cuda(), batch["x"].cuda(),
                                     batch["batch"].cuda(), batch["y"].cuda())
    history_path = out / "history.jsonl"
    history_path.write_text("")
    started = time.time()
    last_fit = None
    weighted_loss = plain_loss = float("nan")
    for step in range(1, 1001):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(pos, features, groups)
        loss = compute_loss(prediction, batch, cfg.train, "cuda", stats)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step % 50 == 0:
            last_fit = evaluate(model, ds, stats)
            with torch.no_grad():
                model.eval()
                eval_pred = model(pos, features, groups)
                weighted_loss = float(compute_loss(eval_pred, batch, cfg.train, "cuda", stats))
                plain_loss = float(F.mse_loss(eval_pred, target))
            record = {"optimizer_step": step, "train_fixed_target_weight_mse": weighted_loss,
                      "train_plain_norm_mse": plain_loss, "constant_lr": 0.001,
                      "r2_field_raw": last_fit["r2_field_raw"],
                      "r2_casemean": last_fit["r2_casemean"]}
            with history_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    assert last_fit is not None
    torch.save({"model": model.state_dict(), "optimizer_step": 1000,
                "cfg_name": cfg.name, "selection": "last_step_train_only"},
               out / "ckpt_last.pt")
    gate = protocol["gate"]
    passed = all(last_fit[key] >= threshold for key, threshold in gate.items())
    control = read_json(paths["control_metrics"])
    result = {
        "task": protocol["task"], "control_job": protocol["control_job"],
        "job_id": os.environ["SLURM_JOB_ID"], "optimizer_steps": 1000,
        "checkpoint_rule": "last step only; no selection",
        "single_variable": "fixed target-weight normalized MSE alpha2",
        "fixed_quantiles": quantiles, "amp": False, "scheduler": "none",
        "weight_decay": 0.0, "grad_clip": 0.0, "constant_lr": 0.001,
        "final_weighted_norm_mse": weighted_loss, "final_plain_norm_mse": plain_loss,
        **last_fit, "thresholds": gate, "gate": "GO" if passed else "NO_GO",
        "delta_vs_control": {
            "r2_field_raw": last_fit["r2_field_raw"] - control["r2_field_raw"],
            "r2_casemean": last_fit["r2_casemean"] - control["r2_casemean"],
        },
        "elapsed_s": time.time() - started, "validation_loaded": False,
        "test_loaded": False,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"A0D target-weight-only completed gate={result['gate']}", flush=True)


def record(protocol_path: Path, job_id: str) -> None:
    _, paths, _, _ = validate(protocol_path)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    (paths["output_dir"] / "submission.json").write_text(json.dumps({
        "job_id": job_id, "submitted_unix": time.time(),
        "stage": "target_weight_only"
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "train", "record"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.mode == "dry-run":
        protocol, _, _, manifest = validate(args.protocol)
        print(json.dumps({"status": "DRY_RUN_OK", "task": protocol["task"],
                          "cases": [row["case"] for row in manifest["cases"]],
                          "single_variable": protocol["single_variable"],
                          "validation_loaded": False, "test_loaded": False}, indent=2))
    elif args.mode == "train":
        train(args.protocol)
    else:
        if not args.job_id:
            raise ValueError("record requires --job-id")
        record(args.protocol, args.job_id)


if __name__ == "__main__":
    main()
