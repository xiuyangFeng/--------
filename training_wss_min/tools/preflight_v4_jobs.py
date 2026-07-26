#!/usr/bin/env python3
"""Formal data, sampler, CPU/CUDA, memory, and full-query gate for v4 jobs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C, dataset as D, surface as S
from training_wss_min.evaluate import predict_case_norm
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def forward_model(model, batch, cfg, device, epoch=0):
    if cfg.data.support_n_points is not None or cfg.model.sa_center_counts:
        return model.forward_support_query(
            batch["support_pos"].to(device), batch["support_x"].to(device),
            batch["support_batch"].to(device), batch["pos"].to(device),
            batch["x"].to(device), batch["batch"].to(device),
            unit_ids=batch["unit_ids"], epoch=epoch, global_seed=cfg.train.seed,
            evaluation=False,
        )
    return model(batch["pos"].to(device), batch["x"].to(device), batch["batch"].to(device))


def protocol_gate(split: dict, stats: dict, split_path: Path, stats_path: Path) -> dict:
    expected = {
        part: len(split.get(f"{part}_cases", [])) for part in ("train", "val", "test")
    }
    if split.get("expected_counts") != expected:
        raise RuntimeError(
            f"partition counts disagree with explicit case lists: "
            f"{split.get('expected_counts')} != {expected}"
        )
    partitions = set(
        split.get("train_cases", []) + split.get("val_cases", []) + split.get("test_cases", [])
    )
    unused = set(split.get("unused_cases", []))
    if unused & partitions:
        raise RuntimeError("unused control cases leaked into an active partition")
    stats_units = list(stats["train_units"])
    stats_split_path = Path(stats["split_path"])
    if not stats_split_path.is_file() or stats["split_sha256"] != sha(stats_split_path):
        raise RuntimeError("stats source split is missing or its frozen hash drifted")
    if stats["n_cases"] != len(stats_units):
        raise RuntimeError("stats n_cases differs from its train manifest")
    active_train = set(split["train_cases"])
    if not set(stats_units) <= active_train:
        raise RuntimeError("target stats include a unit outside the active train partition")
    stats_mode = "refit_active_train"
    if stats["split_sha256"] != sha(split_path):
        stats_mode = "frozen_train_subset"
    elif stats_units != split["train_cases"] or stats["n_cases"] != expected["train"]:
        raise RuntimeError("refit stats train manifest differs from active train partition")
    if set(stats_units) & set(split.get("val_cases", []) + split.get("test_cases", [])):
        raise RuntimeError("target stats include an active val/test unit")
    if any(unit.startswith("ILO/") and not unit.endswith("/before") for unit in partitions | unused):
        raise RuntimeError("non-before ILO unit entered the protocol")
    return {"partition_counts": expected, "stratum_counts": split.get("counts", {}),
            "unused_count": len(unused), "split_sha256": sha(split_path),
            "stats_sha256": sha(stats_path), "target_stats_mode": stats_mode,
            "target_stats_train_cases": len(stats_units),
            "target_stats_source_split": str(stats_split_path.resolve()),
            "target_stats_source_split_sha256": stats["split_sha256"]}


def audit_area(cases, cache) -> list[dict]:
    reports = []
    for case in cases:
        unit = case["unit_id"]
        if unit not in cache:
            weights, report = S.area_weights_for_case(case, strict=True)
            cache[unit] = (weights, report)
        case["surface_area_weights"], case["surface_area_report"] = cache[unit]
        reports.append(case["surface_area_report"])
    return reports


def requires_surface_area(cfg: C.ExpConfig) -> bool:
    """Return whether this config genuinely consumes mapped surface area."""
    support = cfg.data.support_sampling or cfg.data.sampling
    query = cfg.data.query_sampling or support
    return (
        support == "area_random"
        or (cfg.data.query_mode == "independent" and query == "area_random")
        or cfg.eval.surface_metric_mode == "both_strict"
    )


def one(config_path: Path, area_cache: dict, protocol_cache: dict,
        probe_batch: str = "first") -> dict:
    cfg = C.ExpConfig.from_json(config_path)
    if cfg.run_dir.exists():
        raise FileExistsError(f"formal run dir already exists: {cfg.run_dir}")
    initialization = {"mode": "random_initialization"}
    if cfg.train.init_checkpoint_path:
        init_path = Path(cfg.train.init_checkpoint_path)
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint does not exist: {init_path}")
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        init_model = build_model(cfg.model, C.input_dim(cfg))
        result = init_model.load_state_dict(state, strict=cfg.train.init_checkpoint_strict)
        if cfg.train.init_checkpoint_strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError("strict warm-start unexpectedly reported key mismatch")
        initialization = {
            "mode": "warm_start_weights_only", "checkpoint_path": str(init_path.resolve()),
            "checkpoint_sha256": sha(init_path), "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_metric": payload.get("metric"), "strict": bool(cfg.train.init_checkpoint_strict),
            "optimizer_scheduler_reset": True,
        }
        del init_model
    if cfg.data.required_frame_version != "stl_landmarks_v4":
        raise RuntimeError("v4 frame gate missing")
    split_path, stats_path = Path(cfg.data.split_path), Path(cfg.data.wss_stats_path)
    cache_key = (
        str(split_path.resolve()), str(stats_path.resolve()), cfg.data.data_root,
        cfg.data.required_frame_version, cfg.data.target, cfg.data.target_normalization,
        tuple(cfg.data.input_features), cfg.data.curvature_transform,
        cfg.data.feature_stats_path,
    )
    if cache_key not in protocol_cache:
        split, stats = D.load_split(split_path), D.load_wss_stats(stats_path)
        protocol = protocol_gate(split, stats, split_path, stats_path)
        tr = D.load_partition(str(split_path), "train", stats, strict=True, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version)
        probe_partition = "val" if split.get("val_cases") else "test"
        probe_cases = D.load_partition(str(split_path), probe_partition, stats, strict=True, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version)
        for item in stats["bundle_manifest"]:
            if sha(Path(item["bundle_path"])) != item["bundle_sha256"]:
                raise RuntimeError(f"stats bundle hash drift: {item['unit_id']}")
        if cfg.data.feature_stats_path:
            feature_stats_path = Path(cfg.data.feature_stats_path)
            if not feature_stats_path.is_file():
                raise FileNotFoundError(feature_stats_path)
            feat = json.loads(feature_stats_path.read_text(encoding="utf-8"))
            feature_stats_mode = "frozen"
            feature_stats_sha256 = sha(feature_stats_path)
        else:
            feat = D.compute_feature_stats(tr, cfg.data.input_features, cfg.data.curvature_transform)
            feature_stats_mode = "train_recomputed"
            feature_stats_sha256 = None
        quant = D.compute_train_weight_quantiles(tr)
        protocol_cache[cache_key] = (
            split, stats, protocol, tr, probe_cases, probe_partition, feat, quant,
            feature_stats_mode, feature_stats_sha256,
        )
    (split, stats, protocol, tr, probe_cases, probe_partition, feat, quant,
     feature_stats_mode, feature_stats_sha256) = protocol_cache[cache_key]
    all_cases = tr + probe_cases
    area_required = requires_surface_area(cfg)
    area_reports = audit_area(all_cases, area_cache) if area_required else []

    ds = D.WSSMinDataset(tr, cfg.data, feat, training=True, base_seed=cfg.train.seed)
    same_item = ds[0]
    if cfg.data.query_mode == "same" and not torch.equal(same_item["support_idx"], same_item["query_idx"]):
        raise RuntimeError("SAME did not reuse identical support/query indices")
    if cfg.data.query_mode == "independent" and torch.equal(same_item["support_idx"], same_item["query_idx"]):
        raise RuntimeError("SEP reused the support index array")
    # 全点（support_n<=0）或欠点回退全点的病例没有可探测的重采样；找第一个真正
    # 会被下采样的病例做探针，找不到或关闭 resample 时记 not_applicable 而非误报。
    support_n = int(cfg.data.support_n_points or cfg.data.wall_n_points)
    probe_j = next((j for j in range(len(tr)) if 0 < support_n < len(tr[j]["pos"])), None)
    if probe_j is None or not cfg.data.resample_each_epoch:
        changed_epoch = "not_applicable"
    else:
        probe_item = ds[probe_j]
        ds.set_epoch(1)
        changed_epoch = not torch.equal(probe_item["support_idx"], ds[probe_j]["support_idx"])
        if (cfg.data.support_sampling in {"random", "area_random", "fps_multistart"}) and not changed_epoch:
            raise RuntimeError("random support did not change across epochs")
        ds.set_epoch(0)

    cpu_batch = D.collate([ds[i] for i in range(2)])
    cpu_model = build_model(cfg.model, C.input_dim(cfg))
    cpu_pred = forward_model(cpu_model, cpu_batch, cfg, "cpu")
    cpu_loss = compute_loss(cpu_pred, cpu_batch, cfg.train, "cpu", stats)
    if not torch.isfinite(cpu_loss):
        raise RuntimeError("non-finite CPU two-case smoke loss")
    cpu_loss.backward()
    cpu_result = {"cases": 2, "forward_backward": "passed", "loss": float(cpu_loss.detach())}
    del cpu_model, cpu_batch, cpu_pred, cpu_loss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for formal preflight")
    n_batch = min(cfg.train.batch_cases, len(ds))
    if probe_batch == "largest":
        # 全点/大点数矩阵的显存门禁必须用最坏 batch；split 前几例是小 AG 例，
        # 按首序探测会低估峰值显存 >20x。
        batch_idx = sorted(range(len(ds)), key=lambda j: len(tr[j]["pos"]), reverse=True)[:n_batch]
    else:
        batch_idx = list(range(n_batch))
    batch = D.collate([ds[j] for j in batch_idx])
    device = "cuda"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg.model, C.input_dim(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    with torch.amp.autocast("cuda", enabled=cfg.train.amp):
        pred = forward_model(model, batch, cfg, device)
        loss = compute_loss(pred, batch, cfg.train, device, stats)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite CUDA smoke loss")
    loss.backward(); opt.step(); torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    center_counts = [getattr(sa, "last_center_counts", []) for sa in getattr(model, "sa", [])]
    if cfg.model.sa_center_counts:
        expected = [[int(n)] * n_batch for n in cfg.model.sa_center_counts]
        if center_counts != expected:
            raise RuntimeError(f"wrong SA center counts: {center_counts} != {expected}")

    probe = min(probe_cases, key=lambda c: len(c["pos"]))
    before = getattr(model, "support_encode_calls", 0)
    old_chunk = cfg.eval.query_chunk_size
    cfg.eval.query_chunk_size = 0
    full = predict_case_norm(model.eval(), probe, cfg.data.input_features, feat, device, cfg=cfg)
    cfg.eval.query_chunk_size = 1024
    chunked = predict_case_norm(model, probe, cfg.data.input_features, feat, device, cfg=cfg)
    cfg.eval.query_chunk_size = old_chunk
    max_abs = float(np.max(np.abs(full - chunked)))
    if not np.allclose(full, chunked, atol=2e-5, rtol=2e-5):
        raise RuntimeError(f"chunked/full query mismatch: max_abs={max_abs}")
    calls = getattr(model, "support_encode_calls", 0) - before
    if cfg.eval.fixed_support and calls != 2:
        raise RuntimeError(f"support encoder call count expected 2 for two predictions, got {calls}")

    result = {"experiment": config_path.stem, "config": str(config_path.resolve()),
        "config_sha256": sha(config_path), "run_dir": str(cfg.run_dir),
        **protocol, "train_cases": len(tr), "probe_partition": probe_partition,
        "probe_cases": len(probe_cases), "all_loaded_frames": "stl_landmarks_v4",
        "stats_manifest_hashes": "all_match",
        "surface_area_gate": {
            "required": area_required,
            "status": "passed" if area_required else "not_requested",
            "cases": len(area_reports),
            "max_p95_over_bbox": (
                max(r["bidirectional_p95_over_bbox"] for r in area_reports)
                if area_reports else None
            ),
            "max_distance_over_bbox": (
                max(r["bidirectional_max_over_bbox"] for r in area_reports)
                if area_reports else None
            ),
            "min_positive_area_weights": (
                min(r["n_positive_wall_weights"] for r in area_reports)
                if area_reports else None
            ),
        },
        "sampling": {"support": cfg.data.support_sampling, "query_mode": cfg.data.query_mode,
                     "query": cfg.data.query_sampling, "without_replacement": True,
                     "epoch_resampling": changed_epoch},
        "cpu_two_case_smoke": cpu_result, "cuda_device": torch.cuda.get_device_name(0),
        "cuda_amp_forward_backward": "passed", "cuda_smoke_loss": float(loss.detach().cpu()),
        "peak_cuda_memory_bytes": peak, "batch_cases": cfg.train.batch_cases,
        "cuda_probe_batch": {"mode": probe_batch,
                             "case_points": [len(tr[j]["pos"]) for j in batch_idx]},
        "sa_center_counts_observed": center_counts,
        "full_query_chunk_consistency": {"case": probe["unit_id"], "max_abs": max_abs,
                                         "support_encoder_calls_for_two_predictions": calls},
        "feature_stats": {"mode": feature_stats_mode, "keys": sorted(feat),
                          "sha256": feature_stats_sha256},
        "initialization": initialization,
        "fixed_loss_quantiles": quant}
    del model, opt, ds, batch, tr, probe_cases, all_cases, pred, loss, full, chunked
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--probe-batch", choices=("first", "largest"), default="first",
                    help="CUDA 冒烟 batch 的取法；大点数矩阵应取 largest 反映最坏显存")
    args = ap.parse_args()
    cache = {}
    protocol_cache = {}
    jobs = []
    error = None
    try:
        for path in args.configs:
            jobs.append(one(path.resolve(), cache, protocol_cache,
                            probe_batch=args.probe_batch))
    except Exception as exc:  # keep an auditable failed gate instead of losing the report
        error = f"{type(exc).__name__}: {exc}"
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "status": "passed" if error is None else "failed", "jobs": jobs,
               "surface_area_mapping": [v[1] for v in cache.values()],
               "surface_area_policy": (
                   "audited only when sampling or evaluation consumes mapped area; "
                   "vertex/FPS phase records not_requested and never substitutes uniform weights"
               )}
    if error is not None:
        payload["error"] = error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "jobs": len(jobs),
                      "error": error, "output": str(args.output)}, ensure_ascii=False))
    if error is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
