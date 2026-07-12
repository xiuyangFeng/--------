#!/usr/bin/env python3
"""Round-5 B-REP/C2 cluster-only runner with a hard serial micro gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import dataset as D
from .brep_mlp import (_loader, _set_quantiles, eval_canonical, protocol_paths,
                       read_json, sha256)
from .evaluate import evaluate_partition, write_reports
from .gate1_compare import gate1, load_val
from .pointnext_global_context import build_global_context_model
from .train import (compute_loss, compute_selection_score, lr_lambda_factory,
                    seed_all, setup_logger)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "training_wss_min/configs/round5/brep_c2_global_context_protocol.json"


def _assert_only_global_context(candidate: Path, control: Path) -> None:
    cand = C.ExpConfig.from_json(candidate).to_dict()
    ctrl = C.ExpConfig.from_json(control).to_dict()
    for top in ("data", "train"):
        if cand[top] != ctrl[top]:
            raise RuntimeError(f"C2 freeze violation: {top} differs from {control}")
    cm, bm = dict(cand["model"]), dict(ctrl["model"])
    if cm.pop("name") != "pointnext_global_context":
        raise RuntimeError("C2 candidate must select pointnext_global_context")
    if bm.pop("name") != "pointnext_s" or cm != bm:
        raise RuntimeError("C2 freeze violation: model config differs beyond model.name")


def validate(protocol_path: Path, stage: str):
    protocol = read_json(protocol_path)
    if protocol.get("task") != "B-REP/C2":
        raise RuntimeError("wrong task protocol")
    paths = protocol_paths(protocol)
    for key, path in paths.items():
        if key != "output_root" and not path.exists():
            raise FileNotFoundError(f"missing protocol asset {key}: {path}")
    if protocol.get("global_context") != "decoded_feature_case_mean_plus_max":
        raise RuntimeError("C2 context definition is not frozen")
    if protocol["serial_gate"] != {"r2_field_raw": 0.95, "r2_casemean": 0.95}:
        raise RuntimeError("C2 micro thresholds must remain 0.95/0.95")
    if stage == "micro":
        _assert_only_global_context(paths["micro_config"], paths["a0_micro_config"])
        cfg = C.ExpConfig.from_json(paths["micro_config"])
    else:
        _assert_only_global_context(paths["dev_config"], paths["b1_config"])
        cfg = C.ExpConfig.from_json(paths["dev_config"])
    if cfg.train.seed != 1234 or cfg.data.wall_n_points != 2000 or cfg.data.sampling != "fps":
        raise RuntimeError("C2 requires seed1234 and canonical FPS-2000")
    if cfg.model.out_dim != 1 or not cfg.train.loss_weight_target or not cfg.train.loss_weight_fixed_quantiles:
        raise RuntimeError("C2 requires scalar target-weight loss with fixed train quantiles")
    return protocol, paths, cfg


def _require_cluster_gpu(stage: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(f"formal C2 {stage} training is cluster-only")
    if not torch.cuda.is_available():
        raise RuntimeError(f"C2 {stage} requires a visible GPU")


def _runtime(out: Path, stage: str, cfg: C.ExpConfig, paths) -> None:
    (out / "slurm_runtime.json").write_text(json.dumps({
        "task": "B-REP/C2", "stage": stage, "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix": time.time(), "seed": cfg.train.seed,
        "config_sha256": sha256(paths[f"{stage}_config"]),
        "validation_loaded": stage == "dev", "test_loaded": False,
        "single_variable": "decoded per-point features plus broadcast per-case mean and max",
    }, indent=2) + "\n")


def _load_micro(protocol_path: Path):
    protocol, paths, cfg = validate(protocol_path, "micro")
    stats = D.load_wss_stats(paths["stats_snapshot"])
    feat = read_json(paths["feature_stats_snapshot"])
    _set_quantiles(cfg, read_json(paths["weight_quantiles_snapshot"]))
    manifest = read_json(paths["locked_cases"])
    if not manifest.get("locked") or not manifest.get("replacement_forbidden"):
        raise RuntimeError("A0M four-case manifest is not locked")
    cases = [D.load_case(f"AG/{row['case'].split('/', 1)[0]}",
                         row["case"].split("/", 1)[1], stats) for row in manifest["cases"]]
    ds = D.WSSMinDataset(cases, cfg.data, feat, training=True, base_seed=cfg.train.seed)
    return protocol, paths, cfg, stats, ds


def train_micro(protocol_path: Path) -> None:
    _require_cluster_gpu("micro")
    protocol, paths, cfg, stats, ds = _load_micro(protocol_path)
    out = paths["output_root"] / "micro"
    if (out / "metrics.json").exists():
        raise FileExistsError("C2 micro result already exists; unregistered retries are forbidden")
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out); seed_all(cfg.train.seed); _runtime(out, "micro", cfg, paths)
    loader = _loader(ds, cfg)
    model = build_global_context_model(cfg.model, C.input_dim(cfg)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg.train))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    (out / "config_snapshot.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    hist = out / "history.jsonl"; hist.write_text("")
    for epoch in range(cfg.train.epochs):
        model.train(); ds.set_epoch(epoch); losses = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.train.amp):
                pred = model(batch["pos"].cuda(), batch["x"].cuda(), batch["batch"].cuda())
                loss = compute_loss(pred, batch, cfg.train, "cuda", stats)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update(); losses.append(float(loss.item()))
        sched.step()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"]}
        if (epoch + 1) % protocol["micro_eval_every"] == 0 or epoch == cfg.train.epochs - 1:
            fit = eval_canonical(model, ds, stats)
            rec.update({k: fit[k] for k in ("r2_field_raw", "r2_casemean", "r2_field_casebalanced")})
            log.info("case_progress=4/4 epoch=%d field=%.5f casemean=%.5f", epoch,
                     fit["r2_field_raw"], fit["r2_casemean"])
        with hist.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    torch.save({"model": model.state_dict(), "epoch": cfg.train.epochs - 1,
                "cfg_name": cfg.name}, out / "ckpt_last.pt")
    result = eval_canonical(model, ds, stats); thr = protocol["serial_gate"]
    go = result["r2_field_raw"] >= thr["r2_field_raw"] and result["r2_casemean"] >= thr["r2_casemean"]
    result.update({"gate": "GO" if go else "NO_GO", "thresholds": thr,
                   "checkpoint_rule": "last epoch; train-only; no val selection",
                   "locked_cases_sha256": sha256(paths["locked_cases"])})
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "per_case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["per_case"][0]))
        writer.writeheader(); writer.writerows(result["per_case"])
    log.info("C2 micro complete gate=%s", result["gate"])


def train_dev(protocol_path: Path) -> None:
    _require_cluster_gpu("dev")
    _, paths, cfg = validate(protocol_path, "dev")
    micro = read_json(paths["output_root"] / "micro/metrics.json")
    if micro.get("gate") != "GO":
        raise RuntimeError("serial gate closed: C2 micro did not pass both 0.95 thresholds")
    out = paths["output_root"] / "dev1"
    if (out / "gate1.json").exists():
        raise FileExistsError("C2 dev1 result already exists; Gate-1 retries are forbidden")
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out); seed_all(cfg.train.seed); _runtime(out, "dev", cfg, paths)
    stats = D.load_wss_stats(Path(cfg.data.wss_stats_path))
    train_cases = D.load_partition(cfg.data.split_path, "train", stats, strict=True)
    val_cases = D.load_partition(cfg.data.split_path, "val", stats, strict=True)
    feat = D.compute_feature_stats(train_cases, cfg.data.input_features, cfg.data.curvature_transform)
    quant = D.compute_train_weight_quantiles(train_cases); _set_quantiles(cfg, quant)
    for name, obj in (("config.json", cfg.to_dict()), ("feature_stats.json", feat),
                      ("weight_quantiles.json", quant), ("wss_global_stats.json", stats)):
        (out / name).write_text(json.dumps(obj, indent=2) + "\n")
    ds = D.WSSMinDataset(train_cases, cfg.data, feat, training=True, base_seed=cfg.train.seed)
    loader = _loader(ds, cfg)
    model = build_global_context_model(cfg.model, C.input_dim(cfg)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg.train))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    hist = out / "history.jsonl"; hist.write_text(""); best = -math.inf; stale = 0
    for epoch in range(cfg.train.epochs):
        model.train(); ds.set_epoch(epoch); losses = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.train.amp):
                pred = model(batch["pos"].cuda(), batch["x"].cuda(), batch["batch"].cuda())
                loss = compute_loss(pred, batch, cfg.train, "cuda", stats)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update(); losses.append(float(loss.item()))
        sched.step()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"]}
        if (epoch + 1) % cfg.train.eval_every == 0 or epoch == cfg.train.epochs - 1:
            val = evaluate_partition(model, val_cases, cfg, feat, stats, "cuda", make_plots=False)
            score = compute_selection_score(cfg.train, val["aggregate"], val["field"], val["calibration"], val["hotspot"])
            rec.update({"val_r2_field": val["field"]["r2"], "val_r2_casemean": val["aggregate"]["r2_casemean"],
                        "val_r2_field_casebalanced": val["field_casebalanced"]["r2"],
                        "val_mae_field": val["field"]["mae"],
                        "val_top10_pred_true_ratio": val["calibration"].get("top10_pred_true_ratio"),
                        "val_top10_iou": val["hotspot"].get("top10_iou"), "val_selection_score": score})
            if np.isfinite(score) and score > best:
                best, stale = score, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "metric": score,
                            "cfg_name": cfg.name}, out / "ckpt_best.pt")
            else:
                stale += 1
            log.info("case_progress=%d/%d epoch=%d val_field=%.5f val_casemean=%.5f score=%.5f",
                     len(val_cases), len(val_cases), epoch, rec["val_r2_field"], rec["val_r2_casemean"], score)
        with hist.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg_name": cfg.name}, out / "ckpt_last.pt")
        if epoch + 1 >= cfg.train.min_epoch and stale >= cfg.train.early_stop_patience:
            break
    ckpt = torch.load(out / "ckpt_best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    candidate = evaluate_partition(model, val_cases, cfg, feat, stats, "cuda", make_plots=False)
    write_reports({"val": candidate}, out / "eval")
    ctrl = load_val(paths["control_run"]); cand = load_val(out); verdict = gate1(ctrl, cand)
    report = {"task": "B-REP/C2", "partition": "val-only", "seed": 1234,
              "control_run": str(paths["control_run"]), "control": ctrl,
              "candidate": cand, **verdict}
    (out / "gate1.json").write_text(json.dumps(report, indent=2) + "\n")
    log.info("C2 dev1 Gate-1=%s", verdict["verdict"])


def dry_run(protocol_path: Path) -> None:
    protocol, paths, _ = validate(protocol_path, "micro")
    validate(protocol_path, "dev")
    manifest = read_json(paths["locked_cases"])
    print(json.dumps({"status": "DRY_RUN_OK", "task": protocol["task"],
                      "locked_cases": [x["case"] for x in manifest["cases"]],
                      "micro_gate": protocol["serial_gate"], "test_loaded": False,
                      "output_root": str(paths["output_root"])}, indent=2))


def smoke(protocol_path: Path) -> None:
    """CPU preflight: freeze, scalar forward/backward, and exact case isolation."""
    protocol, paths, cfg = validate(protocol_path, "micro")
    validate(protocol_path, "dev")
    seed_all(cfg.train.seed)
    model = build_global_context_model(cfg.model, C.input_dim(cfg))
    model.train()
    n1, n2 = 96, 80
    pos = torch.cat([torch.rand(n1, 3) * 0.25, torch.rand(n2, 3) * 0.25 + 0.5])
    x = torch.randn(n1 + n2, C.input_dim(cfg))
    batch = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])
    pred = model(pos, x, batch)
    loss = pred.square().mean(); loss.backward()
    if pred.shape != (n1 + n2,) or not torch.isfinite(pred).all():
        raise RuntimeError("C2 scalar forward smoke failed")
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if not grad_ok:
        raise RuntimeError("C2 backward smoke failed")
    decoded_a = torch.randn(13, cfg.model.width)
    decoded_b1 = torch.randn(7, cfg.model.width)
    decoded_b2 = torch.randn(7, cfg.model.width) * 100.0 + 50.0
    batch_ab = torch.cat([torch.zeros(13, dtype=torch.long), torch.ones(7, dtype=torch.long)])
    ctx1 = model.append_global_context(torch.cat([decoded_a, decoded_b1]), batch_ab)[:13]
    ctx2 = model.append_global_context(torch.cat([decoded_a, decoded_b2]), batch_ab)[:13]
    isolation_delta = float((ctx1 - ctx2).abs().max())
    if isolation_delta != 0.0:
        raise RuntimeError(f"case context leakage detected: max_delta={isolation_delta}")
    out = paths["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "task": protocol["task"], "status": "SMOKE_OK", "output_shape": list(pred.shape),
        "loss": float(loss), "backward_finite": grad_ok,
        "case_isolation_max_abs_delta": isolation_delta, "out_dim": cfg.model.out_dim,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "test_loaded": False,
    }
    (out / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "c2_report.md").write_text(
        "# B-REP/C2 global context\n\n"
        "- Status: preflight complete; micro training pending/active.\n"
        "- Single variable: concatenate decoded point feature with its case-level mean and max, then use the scalar head.\n"
        "- Frozen: B1 data, loss, stats, seed 1234, canonical FPS-2000, four locked A0M cases, and out_dim=1.\n"
        f"- Forward/backward smoke: PASS (`shape={list(pred.shape)}`, finite gradients).\n"
        f"- Case isolation smoke: PASS (`max_abs_delta={isolation_delta}`).\n"
        "- Serial rule: submit dev1 only if micro last checkpoint reaches field and casemean R2 >= 0.95.\n"
        "- Evaluation scope: micro train-only; dev1 val-only if unlocked; legacy test16 never loaded.\n"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "smoke", "micro", "dev"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if args.mode == "dry-run":
        dry_run(args.protocol)
    elif args.mode == "smoke":
        smoke(args.protocol)
    elif args.mode == "micro":
        train_micro(args.protocol)
    else:
        train_dev(args.protocol)


if __name__ == "__main__":
    main()
