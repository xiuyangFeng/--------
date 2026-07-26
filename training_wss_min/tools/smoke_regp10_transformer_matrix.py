#!/usr/bin/env python3
"""CUDA smoke for every REG-P10 local/global Transformer matrix arm."""

from __future__ import annotations

import argparse
import gc
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
        batch["support_pos"].cuda(),
        batch["support_x"].cuda(),
        batch["support_batch"].cuda(),
        batch["pos"].cuda(),
        batch["x"].cuda(),
        batch["batch"].cuda(),
        unit_ids=batch["unit_ids"],
        epoch=0,
        global_seed=cfg.train.seed,
        evaluation=False,
    )


def one(row: dict) -> dict:
    path = Path(row["config"])
    cfg = C.ExpConfig.from_json(path)
    expected_local = tuple(int(v) for v in row["local_transformer_stages"])
    expected_global = bool(row["coarse_attention"])
    if tuple(cfg.model.local_transformer_stages) != expected_local:
        raise RuntimeError("local stage contract drift")
    if bool(cfg.model.coarse_attention) != expected_global:
        raise RuntimeError("global attention contract drift")

    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    train = D.load_partition(
        cfg.data.split_path,
        "train",
        stats,
        strict=True,
        target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
    )
    test = D.load_partition(
        cfg.data.split_path,
        "test",
        stats,
        strict=True,
        target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
    )
    feature_stats = json.loads(
        Path(cfg.data.feature_stats_path).read_text(encoding="utf-8")
    )
    ds = D.WSSMinDataset(
        train, cfg.data, feature_stats, training=True, base_seed=cfg.train.seed
    )
    batch = D.collate([ds[0]])
    model = build_model(cfg.model, C.input_dim(cfg)).cuda().train()
    local_called: list[int] = []
    global_called: list[int] = []
    hooks = []
    for stage, sa in enumerate(model.sa, 1):
        if sa.local_transformer is not None:
            hooks.append(
                sa.local_transformer.register_forward_hook(
                    lambda *_args, stage=stage: local_called.append(stage)
                )
            )
    if model.coarse_global is not None:
        hooks.append(
            model.coarse_global.register_forward_hook(
                lambda *_args: global_called.append(3)
            )
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    prediction = forward(model, batch, cfg)
    loss = compute_loss(prediction, batch, cfg.train, "cuda", stats)
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise RuntimeError("NaN/Inf during GPU smoke")
    loss.backward()
    transformer_grads = [
        grad
        for name, parameter in model.named_parameters()
        if ("local_transformer" in name or "coarse_global" in name)
        for grad in [parameter.grad]
    ]
    if not transformer_grads or any(
        grad is None or not torch.isfinite(grad).all()
        for grad in transformer_grads
    ):
        raise RuntimeError("missing or non-finite Transformer gradients")
    optimizer.step()
    torch.cuda.synchronize()
    if tuple(sorted(set(local_called))) != expected_local:
        raise RuntimeError(
            f"local execution mismatch: {local_called} vs {expected_local}"
        )
    if bool(global_called) != expected_global:
        raise RuntimeError(
            f"global execution mismatch: {global_called} vs {expected_global}"
        )
    centers = [sa.last_center_counts for sa in model.sa]
    if centers != [[125], [125], [32]]:
        raise RuntimeError(f"wrong center trace: {centers}")

    with tempfile.TemporaryDirectory(prefix="regp10tf_smoke_") as tmp:
        checkpoint = Path(tmp) / "checkpoint.pt"
        torch.save({"model": model.state_dict()}, checkpoint)
        restored = build_model(cfg.model, C.input_dim(cfg)).cuda()
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        restored.load_state_dict(payload["model"], strict=True)
        restored.eval()
        probe = min(test, key=lambda case: len(case["pos"]))
        old_chunk = cfg.eval.query_chunk_size
        cfg.eval.query_chunk_size = 0
        full = predict_case_norm(
            restored,
            probe,
            cfg.data.input_features,
            feature_stats,
            "cuda",
            cfg=cfg,
        )
        cfg.eval.query_chunk_size = 1024
        chunk = predict_case_norm(
            restored,
            probe,
            cfg.data.input_features,
            feature_stats,
            "cuda",
            cfg=cfg,
        )
        cfg.eval.query_chunk_size = old_chunk
        maximum = float(np.max(np.abs(full - chunk)))
        if (
            not np.isfinite(full).all()
            or not np.isfinite(chunk).all()
            or not np.allclose(full, chunk, atol=2e-5, rtol=2e-5)
        ):
            raise RuntimeError(f"full/chunk mismatch or nonfinite values ({maximum})")

    for hook in hooks:
        hook.remove()
    result = {
        "experiment_id": row["experiment_id"],
        "config": str(path.resolve()),
        "config_sha256": sha(path),
        "cuda_device": torch.cuda.get_device_name(0),
        "loss": float(loss.detach().cpu()),
        "sa_centers": centers,
        "local_transformer_expected": list(expected_local),
        "local_transformer_executed": sorted(set(local_called)),
        "coarse_global_expected": expected_global,
        "coarse_global_executed": bool(global_called),
        "transformer_gradient_tensors": len(transformer_grads),
        "checkpoint_strict_reload": "passed",
        "full_chunk_max_abs": maximum,
        "finite": True,
    }
    del model, restored, optimizer, batch, ds, prediction, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs, error = [], None
    try:
        for row in manifest["configs"]:
            jobs.append(one(row))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if error is None else "failed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest),
        "jobs": jobs,
    }
    if error:
        payload["error"] = error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": payload["status"], "jobs": len(jobs), "error": error},
            ensure_ascii=False,
        )
    )
    if error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
