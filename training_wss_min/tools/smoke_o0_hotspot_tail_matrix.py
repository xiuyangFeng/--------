#!/usr/bin/env python3
"""Formal CUDA smoke test for the O0 hotspot/tail objective matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    static_path = Path(args.static_audit)
    output = Path(args.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if (
        static.get("status") != "passed"
        or static.get("manifest_sha256") != sha256(manifest_path)
    ):
        raise RuntimeError("static audit missing, failed, or stale")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal objective smoke")

    jobs = []
    for row in manifest["configs"]:
        config_path = Path(row["config"])
        if sha256(config_path) != row["config_sha256"]:
            raise RuntimeError(f"config drift: {config_path}")
        cfg = C.ExpConfig.from_json(config_path)
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
            case_features_path=cfg.data.case_features_path,
        )
        feat_stats = json.loads(
            Path(cfg.data.feature_stats_path).read_text(encoding="utf-8")
        )
        weight_quantiles = D.compute_train_weight_quantiles(train)
        cfg.train.y_norm_q02 = weight_quantiles["y_norm_q02"]
        cfg.train.y_norm_q98 = weight_quantiles["y_norm_q98"]
        cfg.train._raw_p90 = weight_quantiles["raw_p90"]  # type: ignore[attr-defined]
        dataset = D.WSSMinDataset(
            train, cfg.data, feat_stats, training=True, base_seed=cfg.train.seed
        )
        batch = D.collate([dataset[0], dataset[1]])
        model = build_model(cfg.model, C.input_dim(cfg)).cuda().train()
        prediction = model.forward_support_query(
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
        loss = compute_loss(prediction, batch, cfg.train, "cuda", stats)
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite smoke result: {config_path}")
        loss.backward()
        output_gradients = model.head[-1].weight.grad
        if output_gradients is None or not torch.isfinite(output_gradients).all():
            raise RuntimeError(f"missing/non-finite output gradients: {config_path}")
        if cfg.train.loss_hotspot_bce_lambda > 0 and (
            prediction.ndim != 2
            or prediction.shape[1] != 2
            or output_gradients[1].abs().sum() <= 0
        ):
            raise RuntimeError("hotspot arm did not train the auxiliary output")
        if cfg.train.loss_pinball_lambda > 0 and (
            prediction.ndim not in {1, 2}
            or output_gradients[0].abs().sum() <= 0
        ):
            raise RuntimeError("pinball arm did not train the regression output")
        jobs.append(
            {
                "experiment_id": row["experiment_id"],
                "config": str(config_path.resolve()),
                "config_sha256": row["config_sha256"],
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.detach().cpu()),
                "output_gradient_l1": [
                    float(value)
                    for value in output_gradients.detach().abs().sum(dim=1).cpu()
                ],
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
        )
        del model, batch, dataset, train
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "static_audit": str(static_path.resolve()),
        "static_audit_sha256": sha256(static_path),
        "jobs": jobs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "passed", "jobs": len(jobs), "output": str(output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
