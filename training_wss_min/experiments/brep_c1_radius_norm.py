#!/usr/bin/env python3
"""Round-5 B-REP/C1 cluster-only runner with a hard serial micro gate."""

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

from training_wss_min import config as C
from training_wss_min import dataset as D
from .brep_mlp import (_loader, _set_quantiles, eval_canonical, protocol_paths,
                       read_json, sha256)
from training_wss_min.evaluate import evaluate_partition, write_reports
from training_wss_min.tools.gate1_compare import gate1, load_val
from .pointnext_radius_norm import build_radius_norm_model
from training_wss_min.objectives import compute_loss, compute_selection_score
from training_wss_min.runtime import lr_lambda_factory, seed_all, setup_logger


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "training_wss_min/configs/fit_lc_diagnosis/brep_c1_radius_norm_protocol.json"


def _assert_only_radius_norm(candidate: Path, control: Path) -> None:
    cand = C.ExpConfig.from_json(candidate).to_dict()
    ctrl = C.ExpConfig.from_json(control).to_dict()
    for top in ("data", "train"):
        if cand[top] != ctrl[top]:
            raise RuntimeError(f"C1 freeze violation: {top} differs from {control}")
    cm, bm = dict(cand["model"]), dict(ctrl["model"])
    if cm.pop("name") != "pointnext_radius_norm":
        raise RuntimeError("C1 candidate must select pointnext_radius_norm")
    if bm.pop("name") != "pointnext_s" or cm != bm:
        raise RuntimeError("C1 freeze violation: model config differs beyond model.name")


def validate(protocol_path: Path, stage: str) -> tuple[dict, dict[str, Path], C.ExpConfig]:
    protocol = read_json(protocol_path)
    if protocol.get("task") != "B-REP/C1":
        raise RuntimeError("wrong task protocol")
    paths = protocol_paths(protocol)
    for key, path in paths.items():
        if key != "output_root" and not path.exists():
            raise FileNotFoundError(f"missing protocol asset {key}: {path}")
    if protocol["serial_gate"] != {"r2_field_raw": 0.95, "r2_casemean": 0.95}:
        raise RuntimeError("C1 micro thresholds must remain 0.95/0.95")
    if stage == "micro":
        _assert_only_radius_norm(paths["micro_config"], paths["a0_micro_config"])
        cfg = C.ExpConfig.from_json(paths["micro_config"])
    else:
        _assert_only_radius_norm(paths["dev_config"], paths["b1_config"])
        cfg = C.ExpConfig.from_json(paths["dev_config"])
    if cfg.train.seed != 1234 or cfg.data.wall_n_points != 2000 or cfg.data.sampling != "fps":
        raise RuntimeError("C1 requires seed1234 and canonical FPS-2000")
    if cfg.model.out_dim != 1 or not cfg.train.loss_weight_target \
            or not cfg.train.loss_weight_fixed_quantiles:
        raise RuntimeError("C1 requires scalar target-weight loss with fixed train quantiles")
    return protocol, paths, cfg


def _require_cluster_gpu(stage: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(f"formal C1 {stage} training is cluster-only")
    if not torch.cuda.is_available():
        raise RuntimeError(f"C1 {stage} requires a visible GPU")


def _runtime(out: Path, stage: str, cfg: C.ExpConfig, paths: dict[str, Path]) -> None:
    (out / "slurm_runtime.json").write_text(json.dumps({
        "task": "B-REP/C1", "stage": stage, "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix": time.time(), "seed": cfg.train.seed,
        "config_sha256": sha256(paths[f"{stage}_config"]),
        "validation_loaded": stage == "dev", "test_loaded": False,
        "single_variable": "local relative xyz divided by layer radius",
    }, indent=2) + "\n")


def train_micro(protocol_path: Path) -> None:
    _require_cluster_gpu("micro")
    protocol, paths, cfg = validate(protocol_path, "micro")
    out = paths["output_root"] / "micro"
    if (out / "metrics.json").exists():
        raise FileExistsError("C1 micro result already exists; unregistered retries are forbidden")
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out); seed_all(cfg.train.seed); _runtime(out, "micro", cfg, paths)
    stats = D.load_wss_stats(paths["stats_snapshot"])
    feat = read_json(paths["feature_stats_snapshot"])
    _set_quantiles(cfg, read_json(paths["weight_quantiles_snapshot"]))
    manifest = read_json(paths["locked_cases"])
    if not manifest.get("locked") or not manifest.get("replacement_forbidden"):
        raise RuntimeError("A0M four-case manifest is not locked")
    cases = [D.load_case(f"AG/{x['case'].split('/', 1)[0]}",
                         x["case"].split("/", 1)[1], stats) for x in manifest["cases"]]
    ds = D.WSSMinDataset(cases, cfg.data, feat, training=True, base_seed=cfg.train.seed)
    loader = _loader(ds, cfg)
    model = build_radius_norm_model(cfg.model, C.input_dim(cfg)).cuda()
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
            log.info("case_progress=4/4 epoch=%d field=%.5f casemean=%.5f",
                     epoch, fit["r2_field_raw"], fit["r2_casemean"])
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
    log.info("C1 micro complete gate=%s", result["gate"])


def train_dev(protocol_path: Path) -> None:
    _require_cluster_gpu("dev")
    _, paths, cfg = validate(protocol_path, "dev")
    micro = read_json(paths["output_root"] / "micro/metrics.json")
    if micro.get("gate") != "GO":
        raise RuntimeError("serial gate closed: C1 micro did not pass both 0.95 thresholds")
    out = paths["output_root"] / "dev1"
    if (out / "gate1.json").exists():
        raise FileExistsError("C1 dev1 result already exists; Gate-1 retries are forbidden")
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
    model = build_radius_norm_model(cfg.model, C.input_dim(cfg)).cuda()
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
            score = compute_selection_score(cfg.train, val["aggregate"], val["field"],
                                            val["calibration"], val["hotspot"])
            rec.update({"val_r2_field": val["field"]["r2"],
                        "val_r2_casemean": val["aggregate"]["r2_casemean"],
                        "val_r2_field_casebalanced": val["field_casebalanced"]["r2"],
                        "val_mae_field": val["field"]["mae"],
                        "val_top10_pred_true_ratio": val["calibration"].get("top10_pred_true_ratio"),
                        "val_top10_iou": val["hotspot"].get("top10_iou"),
                        "val_selection_score": score})
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
    report = {"task": "B-REP/C1", "partition": "val-only", "seed": 1234,
              "control_run": str(paths["control_run"]), "control": ctrl,
              "candidate": cand, **verdict}
    (out / "gate1.json").write_text(json.dumps(report, indent=2) + "\n")
    log.info("C1 dev1 Gate-1=%s", verdict["verdict"])


def dry_run(protocol_path: Path) -> None:
    protocol, paths, _ = validate(protocol_path, "micro")
    validate(protocol_path, "dev")
    manifest = read_json(paths["locked_cases"])
    print(json.dumps({"status": "DRY_RUN_OK", "task": protocol["task"],
                      "locked_cases": [x["case"] for x in manifest["cases"]],
                      "micro_gate": protocol["serial_gate"], "test_loaded": False,
                      "output_root": str(paths["output_root"])}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dry-run", "micro", "dev"), required=True)
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = ap.parse_args()
    if args.mode == "dry-run":
        dry_run(args.protocol)
    elif args.mode == "micro":
        train_micro(args.protocol)
    else:
        train_dev(args.protocol)


if __name__ == "__main__":
    main()
