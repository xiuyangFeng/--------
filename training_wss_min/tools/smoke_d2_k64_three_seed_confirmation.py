#!/usr/bin/env python3
"""GPU execution smoke for the frozen M1/S2/S3 confirmation configs."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.evaluate import predict_case_norm
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(model, batch, cfg):
    return model.forward_support_query(
        batch["support_pos"].cuda(), batch["support_x"].cuda(), batch["support_batch"].cuda(),
        batch["pos"].cuda(), batch["x"].cuda(), batch["batch"].cuda(),
        unit_ids=batch["unit_ids"], epoch=0, global_seed=cfg.train.seed, evaluation=False,
    )


def one(row: dict) -> dict:
    path = Path(row["config"])
    cfg = C.ExpConfig.from_json(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    train = D.load_partition(cfg.data.split_path, "train", stats, strict=True, target=cfg.data.target,
                             target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                             required_frame_version=cfg.data.required_frame_version)
    test = D.load_partition(cfg.data.split_path, "test", stats, strict=True, target=cfg.data.target,
                            target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                            required_frame_version=cfg.data.required_frame_version)
    # Historical D2 configs intentionally leave feature_stats_path unset; their
    # train loop derives the statistics from the frozen train partition.  Match
    # that runtime contract so this smoke can gate both historical D2 and newer
    # explicitly-frozen-stat configurations.
    if cfg.data.feature_stats_path:
        feature_stats = json.loads(Path(cfg.data.feature_stats_path).read_text())
    else:
        feature_stats = D.compute_feature_stats(
            train, cfg.data.input_features, cfg.data.curvature_transform
        )
    ds = D.WSSMinDataset(train, cfg.data, feature_stats, training=True, base_seed=cfg.train.seed)
    batch = D.collate([ds[0]])
    model = build_model(cfg.model, C.input_dim(cfg)).cuda()
    blocks_called, geope_called = [], []
    hooks = []
    for i, sa in enumerate(model.sa):
        for block in sa.blocks:
            hooks.append(block.register_forward_hook(lambda *_args, i=i: blocks_called.append(i)))
        if sa.geo_pe is not None:
            hooks.append(sa.geo_pe.register_forward_hook(lambda *_args, i=i: geope_called.append(i)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    prediction = forward(model, batch, cfg)
    loss = compute_loss(prediction, batch, cfg.train, "cuda", stats)
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise RuntimeError("NaN/Inf during GPU smoke")
    loss.backward(); optimizer.step(); torch.cuda.synchronize()
    centers = [sa.last_center_counts for sa in model.sa]
    if centers != [[125], [125], [32]]:
        raise RuntimeError(f"wrong center trace: {centers}")
    is_pnxr = any(int(n) > 0 for n in cfg.model.sa_blocks)
    is_geope = bool(cfg.model.local_geope)
    if bool(blocks_called) != is_pnxr:
        raise RuntimeError(f"PointNeXt-R execution mismatch: {blocks_called}")
    if bool(geope_called) != is_geope:
        raise RuntimeError(f"LocalGeoPE execution mismatch: {geope_called}")
    with tempfile.TemporaryDirectory(prefix="d2k64_confirm_smoke_") as tmp:
        checkpoint = Path(tmp) / "checkpoint.pt"
        torch.save({"model": model.state_dict()}, checkpoint)
        restored = build_model(cfg.model, C.input_dim(cfg)).cuda()
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        restored.load_state_dict(payload["model"], strict=True)
        restored.eval(); model.eval()
        probe = min(test, key=lambda case: len(case["pos"]))
        old_chunk = cfg.eval.query_chunk_size
        cfg.eval.query_chunk_size = 0
        full = predict_case_norm(restored, probe, cfg.data.input_features, feature_stats, "cuda", cfg=cfg)
        cfg.eval.query_chunk_size = 1024
        chunk = predict_case_norm(restored, probe, cfg.data.input_features, feature_stats, "cuda", cfg=cfg)
        cfg.eval.query_chunk_size = old_chunk
        maximum = float(np.max(np.abs(full - chunk)))
        if not np.isfinite(full).all() or not np.isfinite(chunk).all() or not np.allclose(full, chunk, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"full/chunk mismatch or nonfinite values ({maximum})")
    for hook in hooks:
        hook.remove()
    return {"experiment_id": row["experiment_id"], "config": str(path.resolve()), "config_sha256": sha(path),
            "cuda_device": torch.cuda.get_device_name(0), "loss": float(loss.detach().cpu()),
            "sa_centers": centers, "pointnext_r_executed": bool(blocks_called),
            "pointnext_r_layers_called": sorted(set(blocks_called)), "local_geope_executed": bool(geope_called),
            "local_geope_layers_called": sorted(set(geope_called)), "checkpoint_strict_reload": "passed",
            "full_chunk_max_abs": maximum, "finite": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    jobs, error = [], None
    try:
        for row in manifest["configs"]:
            jobs.append(one(row))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    out = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
           "status": "passed" if error is None else "failed", "manifest": str(args.manifest.resolve()),
           "manifest_sha256": sha(args.manifest), "jobs": jobs}
    if error:
        out["error"] = error
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "jobs": len(jobs), "error": error}, ensure_ascii=False))
    if error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
