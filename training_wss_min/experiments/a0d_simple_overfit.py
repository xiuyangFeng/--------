#!/usr/bin/env python3
"""A0D cluster-only stripped train-fit diagnostic.

Only the four locked A0M train cases are addressable.  No split is opened and
no validation/test partition exists in this runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M
from .point_mlp import build_point_mlp
from training_wss_min.train import seed_all


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "training_wss_min/configs/fit_lc_diagnosis/a0d_simple_overfit_protocol.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(protocol_path: Path) -> tuple[dict, dict[str, Path], C.ExpConfig, dict]:
    protocol = read_json(protocol_path)
    if protocol.get("task") != "A0D_SIMPLE_OVERFIT":
        raise RuntimeError("wrong A0D protocol")
    if protocol.get("forbidden_scope") != ["val", "test", "legacy test16"]:
        raise RuntimeError("forbidden partition declaration changed")
    paths = {key: Path(value) for key, value in protocol["paths"].items()}
    for key in ("config", "locked_cases", "stats_snapshot", "feature_stats_snapshot"):
        if not paths[key].is_file() or sha256(paths[key]) != protocol["sha256"][key]:
            raise RuntimeError(f"missing or changed frozen asset: {key}")
    cfg = C.ExpConfig.from_json(paths["config"])
    recipe = protocol["recipe"]
    checks = {
        "point_mlp": cfg.model.name == "point_mlp" and cfg.model.out_dim == 1,
        "canonical": cfg.data.wall_n_points == 2000 and cfg.data.sampling == "fps"
                     and not cfg.data.resample_each_epoch and not cfg.data.rot_aug,
        "plain_mse": cfg.train.loss == "mse" and not cfg.train.loss_weight_target
                     and not cfg.train.loss_geom_weight and cfg.train.loss_raw_huber_lambda == 0,
        "optimizer": cfg.train.lr == recipe["constant_lr"]
                     and cfg.train.weight_decay == 0 and cfg.train.grad_clip == 0,
        "no_amp": not cfg.train.amp,
        "steps": cfg.train.epochs == recipe["max_optimizer_steps"] == 1000,
        "train_only_selection": cfg.train.selection_rule == "train_fit_only_no_val_selection"
                                and cfg.train.ckpt_top_k == 0,
    }
    if not all(checks.values()) or recipe["scheduler"] != "none":
        raise RuntimeError(f"simplified recipe freeze violation: {checks}")
    manifest = read_json(paths["locked_cases"])
    labels = [row["case"] for row in manifest["cases"]]
    if not manifest.get("locked") or not manifest.get("replacement_forbidden") \
            or len(labels) != 4 or len(set(labels)) != 4:
        raise RuntimeError("four-case A0M manifest is not locked")
    return protocol, paths, cfg, manifest


def require_cluster_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A0D training is cluster-only; use sbatch")
    if not torch.cuda.is_available():
        raise RuntimeError("A0D training requires one visible cluster GPU")


def load_cases(rows: list[dict], stats: dict) -> list[dict]:
    cases = []
    for row in rows:
        subset, case_name = row["case"].split("/", 1)
        cases.append(D.load_case(f"AG/{subset}", case_name, stats))
    return cases


@torch.no_grad()
def evaluate(model, ds: D.WSSMinDataset, stats: dict) -> dict:
    model.eval()
    rows, all_true, all_pred = [], [], []
    for index in range(len(ds)):
        item = ds[index]
        n_points = len(item["pos"])
        pred_norm = model(item["pos"].cuda(), item["x"].cuda(),
                          torch.zeros(n_points, dtype=torch.long, device="cuda"))
        pred_raw = D.denormalize_wss(pred_norm.float().cpu().numpy(), stats).reshape(-1)
        pred_raw = np.clip(pred_raw, 0, None)
        true_raw = item["y_raw"].numpy().reshape(-1)
        metric = M.basic_metrics(true_raw, pred_raw)
        label = f"{ds.cases[index]['cohort'].split('/')[-1]}/{ds.cases[index]['case']}"
        rows.append({"case": label, **{key: float(value) for key, value in metric.items()}})
        all_true.append(true_raw); all_pred.append(pred_raw)
    field = M.basic_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    return {
        "r2_field_raw": float(field["r2"]),
        "r2_casemean": float(np.mean([row["r2"] for row in rows])),
        "field": {key: float(value) for key, value in field.items()},
        "per_case": rows,
    }


def train_stage(protocol_path: Path, stage: str, case_index: int | None) -> None:
    require_cluster_gpu()
    protocol, paths, cfg, manifest = validate(protocol_path)
    if stage == "single":
        if case_index is None or case_index not in range(4):
            raise ValueError("single stage requires case-index 0..3")
        selected = [manifest["cases"][case_index]]
        out = paths["output_root"] / "single" / f"case_{case_index}"
        gate_thresholds = protocol["gates"]["single_case"]
    else:
        selected = manifest["cases"]
        out = paths["output_root"] / "four_shared"
        gate_thresholds = protocol["gates"]["four_case"]
        summary_path = paths["output_root"] / "single_summary.json"
        if not summary_path.is_file() or read_json(summary_path).get("gate") != "GO":
            raise RuntimeError("serial gate closed: all four single-case jobs must pass first")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"registered result already exists: {out / 'metrics.json'}")

    seed_all(cfg.train.seed)
    stats = D.load_wss_stats(paths["stats_snapshot"])
    feature_stats = read_json(paths["feature_stats_snapshot"])
    ds = D.WSSMinDataset(load_cases(selected, stats), cfg.data, feature_stats,
                         training=True, base_seed=cfg.train.seed)
    fixed_items = [ds[index] for index in range(len(ds))]
    batch = D.collate(fixed_items)
    model = build_point_mlp(cfg.model, C.input_dim(cfg)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=0.0)
    max_steps = int(protocol["recipe"]["max_optimizer_steps"])
    eval_every = int(protocol["recipe"]["eval_every_steps"])
    (out / "config_snapshot.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out / "runtime.json").write_text(json.dumps({
        "job_id": os.environ["SLURM_JOB_ID"],
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "stage": stage, "cases": [row["case"] for row in selected],
        "validation_loaded": False, "test_loaded": False,
        "scheduler_created": False, "amp": False,
    }, indent=2) + "\n")

    pos, x, groups, target = (batch["pos"].cuda(), batch["x"].cuda(),
                              batch["batch"].cuda(), batch["y"].cuda())
    history_path = out / "history.jsonl"
    history_path.write_text("")
    started = time.time()
    last_fit = None
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pos, x, groups)
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at optimizer step {step}")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"non-finite gradient at optimizer step {step}")
        optimizer.step()
        if step % eval_every == 0 or step == max_steps:
            last_fit = evaluate(model, ds, stats)
            record = {"optimizer_step": step, "train_plain_norm_mse": float(loss.detach()),
                      "constant_lr": optimizer.param_groups[0]["lr"],
                      "r2_field_raw": last_fit["r2_field_raw"],
                      "r2_casemean": last_fit["r2_casemean"]}
            with history_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    assert last_fit is not None
    torch.save({"model": model.state_dict(), "optimizer_step": max_steps,
                "cfg_name": cfg.name, "selection": "last_step_train_only"},
               out / "ckpt_last.pt")
    passed = all(last_fit[key] >= threshold for key, threshold in gate_thresholds.items())
    result = {
        "task": protocol["task"], "stage": stage,
        "cases": [row["case"] for row in selected],
        "optimizer_steps": max_steps, "checkpoint_rule": "last step; no selection",
        "plain_normalized_mse": True, "amp": False, "scheduler": "none",
        "weight_decay": 0.0, "grad_clip": 0.0, "constant_lr": cfg.train.lr,
        **last_fit, "thresholds": gate_thresholds, "gate": "GO" if passed else "NO_GO",
        "elapsed_s": time.time() - started, "validation_loaded": False, "test_loaded": False,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "per_case_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["per_case"][0]))
        writer.writeheader(); writer.writerows(result["per_case"])
    print(f"A0D {stage} completed gate={result['gate']}", flush=True)


def summarize(protocol_path: Path) -> dict:
    protocol, paths, _, manifest = validate(protocol_path)
    rows = []
    for index, item in enumerate(manifest["cases"]):
        path = paths["output_root"] / "single" / f"case_{index}" / "metrics.json"
        if not path.is_file():
            rows.append({"case_index": index, "case": item["case"], "status": "MISSING"})
            continue
        metric = read_json(path)
        rows.append({"case_index": index, "case": item["case"], "status": metric["gate"],
                     "r2_field_raw": metric["r2_field_raw"],
                     "r2_casemean": metric["r2_casemean"],
                     "job_id": read_json(path.parent / "runtime.json")["job_id"]})
    passed = len(rows) == 4 and all(row["status"] == "GO" for row in rows)
    summary = {"task": protocol["task"], "stage": "single_case_array",
               "cases": rows, "gate": "GO" if passed else "NO_GO",
               "four_shared_allowed": passed, "validation_loaded": False,
               "test_loaded": False}
    (paths["output_root"] / "single_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# A0D simplified overfit｜single-case array", "",
             f"Gate: **{summary['gate']}**；four-case shared allowed: `{passed}`。", "",
             "| idx | case | R² field raw | Gate | Job |", "|---:|---|---:|---|---:|"]
    for row in rows:
        r2 = f"{row['r2_field_raw']:.6f}" if "r2_field_raw" in row else "—"
        lines.append(f"| {row['case_index']} | {row['case']} | {r2} | {row['status']} | {row.get('job_id', '—')} |")
    lines += ["", "Train-only; val/test/legacy test16 were not loaded."]
    (paths["output_root"] / "single_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def record_submission(protocol_path: Path, stage: str, job_id: str) -> None:
    _, paths, _, _ = validate(protocol_path)
    target = paths["output_root"] / ("single_submission.json" if stage == "single" else "four_shared/submission.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"stage": stage, "job_id": job_id,
                                  "submitted_unix": time.time()}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "single", "four", "summarize", "record"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--stage", choices=("single", "four"))
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.mode == "dry-run":
        protocol, paths, _, manifest = validate(args.protocol)
        print(json.dumps({"status": "DRY_RUN_OK", "task": protocol["task"],
                          "cases": [row["case"] for row in manifest["cases"]],
                          "output_root": str(paths["output_root"]),
                          "validation_loaded": False, "test_loaded": False}, indent=2))
    elif args.mode in ("single", "four"):
        train_stage(args.protocol, args.mode, args.case_index)
    elif args.mode == "summarize":
        summarize(args.protocol)
    else:
        if not args.stage or not args.job_id:
            raise ValueError("record mode requires --stage and --job-id")
        record_submission(args.protocol, args.stage, args.job_id)


if __name__ == "__main__":
    main()
