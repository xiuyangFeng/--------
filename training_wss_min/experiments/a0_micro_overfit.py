#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round-5 A0M: four-case canonical-2000 micro-overfit sanity check.

This entry point is intentionally isolated from the normal train/val pipeline:
it loads only the four locked train cases and never loads a validation or test
partition.  Formal training must be launched through the accompanying Slurm
file; local use is limited to ``--mode register`` and ``--mode dry-run``.
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
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M
from training_wss_min.pointnext import build_model
from training_wss_min.train import compute_loss, lr_lambda_factory, seed_all, setup_logger


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "training_wss_min/configs/fit_lc_diagnosis/a0_micro_protocol.json"
DEFAULT_OUT = ROOT / "training_wss_min/runs/_round5/a0_micro"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _paths(protocol_path: Path, protocol: dict) -> Dict[str, Path]:
    del protocol_path
    return {k: Path(v) for k, v in protocol["paths"].items()}


def register(protocol_path: Path) -> None:
    """Lock the deterministic fast/slow x low/high train-only case list."""
    protocol = _read_json(protocol_path)
    paths = _paths(protocol_path, protocol)
    out_dir = paths["output_dir"]
    manifest_path = out_dir / "locked_cases.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"locked manifest already exists and cannot be replaced: {manifest_path}"
        )

    split_path = paths["split"]
    split = _read_json(split_path)
    if split.get("test_cases"):
        raise RuntimeError("A0M split must expose no test cases")
    stats = D.load_wss_stats(paths["stats_snapshot"])
    rows: List[dict] = []
    for label in split["train_cases"]:
        subset, case_name = label.split("/", 1)
        case = D.load_case(f"AG/{subset}", case_name, stats)
        y = np.asarray(case["y_raw"], dtype=np.float64)
        rows.append({
            "case": label,
            "subset": subset,
            "wss_mean": float(np.mean(y)),
            "wss_median": float(np.median(y)),
            "wss_p90": float(np.percentile(y, 90)),
            "n_wall": int(len(y)),
        })

    selected: List[dict] = []
    for subset in ("fast", "slow"):
        group = sorted((r for r in rows if r["subset"] == subset),
                       key=lambda r: (r["wss_mean"], r["case"]))
        if len(group) < 2:
            raise RuntimeError(f"need at least two formal-train cases for {subset}")
        for level, row in (("low", group[0]), ("high", group[-1])):
            item = dict(row)
            item["wss_level"] = level
            item["selection_rule"] = f"{subset} formal-train min/max peak-WSS case mean"
            selected.append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "train_only_wss_candidates.csv"
    with candidates_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["subset"], r["wss_mean"], r["case"])))
    manifest = {
        "task": "A0M",
        "locked": True,
        "replacement_forbidden": True,
        "source_partition": "formal dev1 train only",
        "selection_axis": "fast/slow x min/max train-only peak-WSS case mean",
        "split_path": str(split_path),
        "split_sha256": _sha256(split_path),
        "stats_snapshot": str(paths["stats_snapshot"]),
        "stats_sha256": _sha256(paths["stats_snapshot"]),
        "cases": selected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def validate(protocol_path: Path) -> tuple[dict, dict, C.ExpConfig, Dict[str, Path]]:
    protocol = _read_json(protocol_path)
    paths = _paths(protocol_path, protocol)
    required = ("config", "split", "stats_snapshot", "feature_stats_snapshot",
                "weight_quantiles_snapshot", "output_dir")
    if any(k not in paths for k in required):
        raise KeyError(f"protocol paths must contain {required}")
    for key in required[:-1]:
        if not paths[key].is_file():
            raise FileNotFoundError(f"missing {key}: {paths[key]}")
    manifest_path = paths["output_dir"] / "locked_cases.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("locked_cases.json missing; run --mode register once")
    manifest = _read_json(manifest_path)
    if not manifest.get("locked") or not manifest.get("replacement_forbidden"):
        raise RuntimeError("case manifest is not locked")
    if _sha256(paths["split"]) != manifest["split_sha256"]:
        raise RuntimeError("formal train split changed after A0M pre-registration")
    labels = [x["case"] for x in manifest["cases"]]
    expected_axes = {(x["subset"], x["wss_level"]) for x in manifest["cases"]}
    if len(labels) != 4 or len(set(labels)) != 4:
        raise RuntimeError("A0M requires exactly four unique locked cases")
    if expected_axes != {(a, b) for a in ("fast", "slow") for b in ("low", "high")}:
        raise RuntimeError("locked cases do not cover fast/slow x low/high")
    formal_train = set(_read_json(paths["split"])["train_cases"])
    if not set(labels) <= formal_train:
        raise RuntimeError("locked cases must all belong to formal train")

    cfg = C.ExpConfig.from_json(paths["config"])
    if cfg.data.wall_n_points != 2000 or cfg.data.sampling != "fps":
        raise RuntimeError("A0M must use canonical FPS-2000")
    if cfg.data.resample_each_epoch or cfg.data.rot_aug:
        raise RuntimeError("A0M canonical subset must remain fixed across epochs")
    if cfg.model.out_dim != 1:
        raise RuntimeError("A0M must remain WSS scalar out_dim=1")
    if cfg.train.selection_rule != "train_fit_only_no_val_selection":
        raise RuntimeError("validation checkpoint selection must be disabled")
    if protocol["thresholds"] != {"r2_field_raw": 0.95, "r2_casemean": 0.95}:
        raise RuntimeError("A0M Go thresholds must both be exactly 0.95")
    return protocol, manifest, cfg, paths


def _load_locked_cases(manifest: dict, stats: dict) -> List[dict]:
    cases = []
    for item in manifest["cases"]:
        subset, case_name = item["case"].split("/", 1)
        cases.append(D.load_case(f"AG/{subset}", case_name, stats))
    return cases


@torch.no_grad()
def evaluate_canonical(model, ds: D.WSSMinDataset, stats: dict, device: str) -> dict:
    model.eval()
    per_case = []
    all_true, all_pred = [], []
    for i in range(len(ds)):
        item = ds[i]
        pos = item["pos"].to(device)
        x = item["x"].to(device)
        batch = torch.zeros(len(pos), dtype=torch.long, device=device)
        pred_norm = model(pos, x, batch).detach().float().cpu().numpy().reshape(-1)
        true_raw = item["y_raw"].numpy().reshape(-1)
        pred_raw = D.denormalize_wss(pred_norm, stats).reshape(-1)
        met = M.basic_metrics(true_raw, pred_raw)
        per_case.append({"case": f"{ds.cases[i]['cohort'].split('/')[-1]}/{ds.cases[i]['case']}",
                         **met, "true": true_raw, "pred": pred_raw})
        all_true.append(true_raw)
        all_pred.append(pred_raw)
    field = M.basic_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    casemean = float(np.nanmean([r["r2"] for r in per_case]))
    casebalanced = M.casebalanced_field_metrics(all_true, all_pred)
    return {"r2_field_raw": field["r2"], "r2_casemean": casemean,
            "r2_field_casebalanced": casebalanced["r2"], "field": field,
            "per_case": per_case}


def _save_plots(history: List[dict], result: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [r["train_loss"] for r in history])
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_title("train loss")
    ev = [r for r in history if "r2_field_raw" in r]
    ax[1].plot([r["epoch"] for r in ev], [r["r2_field_raw"] for r in ev], label="field")
    ax[1].plot([r["epoch"] for r in ev], [r["r2_casemean"] for r in ev], label="casemean")
    ax[1].axhline(0.95, color="k", linestyle="--"); ax[1].set_ylim(-0.2, 1.02)
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("canonical-2000 train R2")
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=150); plt.close(fig)

    plot_dir = out_dir / "per_case_plots"
    plot_dir.mkdir(exist_ok=True)
    for row in result["per_case"]:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(row["true"], row["pred"], s=4, alpha=0.35)
        lo = float(min(np.min(row["true"]), np.min(row["pred"])))
        hi = float(max(np.max(row["true"]), np.max(row["pred"])))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("true WSS"); ax.set_ylabel("predicted WSS")
        ax.set_title(f"{row['case']}  R2={row['r2']:.4f}")
        fig.tight_layout(); fig.savefig(plot_dir / f"{row['case'].replace('/', '__')}.png", dpi=150)
        plt.close(fig)


def train(protocol_path: Path) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal A0M training is cluster-only; submit the Slurm job")
    if not torch.cuda.is_available():
        raise RuntimeError("A0M Slurm job requires one visible GPU")
    protocol, manifest, cfg, paths = validate(protocol_path)
    out_dir = paths["output_dir"]
    log = setup_logger(out_dir)
    seed_all(cfg.train.seed)
    stats = D.load_wss_stats(paths["stats_snapshot"])
    feat_stats = _read_json(paths["feature_stats_snapshot"])
    quantiles = _read_json(paths["weight_quantiles_snapshot"])
    for key in ("y_norm_q02", "y_norm_q98", "curv_q02", "curv_q98",
                "invr_q02", "invr_q98"):
        setattr(cfg.train, key, quantiles[key])
    cfg.train._raw_p90 = quantiles["raw_p90"]  # type: ignore[attr-defined]
    cases = _load_locked_cases(manifest, stats)
    ds = D.WSSMinDataset(cases, cfg.data, feat_stats, training=True,
                         base_seed=cfg.train.seed)
    loader = DataLoader(ds, batch_size=cfg.train.batch_cases, shuffle=True,
                        collate_fn=D.collate, num_workers=cfg.data.num_workers,
                        drop_last=False, persistent_workers=False,
                        generator=torch.Generator().manual_seed(cfg.train.seed),
                        worker_init_fn=D.worker_init_fn if cfg.data.num_workers else None)
    device = "cuda"
    model = build_model(cfg.model, C.input_dim(cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(cfg.train))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n")
    (out_dir / "slurm_runtime.json").write_text(json.dumps({
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix": time.time(), "validation_loaded": False,
        "test_loaded": False,
    }, indent=2) + "\n")

    history: List[dict] = []
    hist_path = out_dir / "history.jsonl"
    hist_path.write_text("")
    eval_every = int(protocol["train_fit_eval_every"])
    for epoch in range(cfg.train.epochs):
        model.train(); ds.set_epoch(epoch)
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.train.amp):
                pred = model(batch["pos"].to(device), batch["x"].to(device),
                             batch["batch"].to(device))
                loss = compute_loss(pred, batch, cfg.train, device, stats)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.item()))
        scheduler.step()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "lr": optimizer.param_groups[0]["lr"]}
        if (epoch + 1) % eval_every == 0 or epoch == cfg.train.epochs - 1:
            fit = evaluate_canonical(model, ds, stats, device)
            rec.update({k: fit[k] for k in ("r2_field_raw", "r2_casemean",
                                             "r2_field_casebalanced")})
            log.info("epoch=%d loss=%.6g train R2 field=%.5f casemean=%.5f",
                     epoch, rec["train_loss"], rec["r2_field_raw"], rec["r2_casemean"])
        history.append(rec)
        with hist_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    torch.save({"model": model.state_dict(), "epoch": cfg.train.epochs - 1,
                "cfg_name": cfg.name, "selection": "last_epoch_train_fit_only"},
               out_dir / "ckpt_last.pt")
    result = evaluate_canonical(model, ds, stats, device)
    threshold = protocol["thresholds"]
    go = (result["r2_field_raw"] >= threshold["r2_field_raw"] and
          result["r2_casemean"] >= threshold["r2_casemean"])
    serializable = {k: v for k, v in result.items() if k != "per_case"}
    serializable["per_case"] = [{k: v for k, v in row.items() if k not in ("true", "pred")}
                                for row in result["per_case"]]
    serializable.update({"gate": "GO" if go else "NO_GO", "thresholds": threshold,
                         "checkpoint_rule": "last epoch; no val selection"})
    (out_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n")
    with (out_dir / "per_case_metrics.csv").open("w", newline="") as f:
        rows = serializable["per_case"]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _save_plots(history, result, out_dir)
    log.info("A0M completed gate=%s", serializable["gate"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("register", "dry-run", "train"), required=True)
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = ap.parse_args()
    if args.mode == "register":
        register(args.protocol)
    elif args.mode == "dry-run":
        protocol, manifest, cfg, paths = validate(args.protocol)
        stats = D.load_wss_stats(paths["stats_snapshot"])
        cases = _load_locked_cases(manifest, stats)
        feat_stats = _read_json(paths["feature_stats_snapshot"])
        ds = D.WSSMinDataset(cases, cfg.data, feat_stats, training=True,
                             base_seed=cfg.train.seed)
        shapes = [tuple(ds[i]["pos"].shape) for i in range(len(ds))]
        if shapes != [(2000, 3)] * 4:
            raise RuntimeError(f"canonical shapes mismatch: {shapes}")
        print(json.dumps({"status": "DRY_RUN_OK", "cases": [x["case"] for x in manifest["cases"]],
                          "sample_shapes": shapes, "val_loaded": False,
                          "test_loaded": False, "output_dir": str(paths["output_dir"])},
                         indent=2, ensure_ascii=False))
    else:
        train(args.protocol)


if __name__ == "__main__":
    main()
