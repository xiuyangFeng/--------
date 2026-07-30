#!/usr/bin/env python3
"""Formal CUDA smoke for REG-P10-LSA2 SAME/IND × MSE/H1/H2."""

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
        raise RuntimeError("CUDA is required for formal matrix smoke")

    first_cfg = C.ExpConfig.from_json(manifest["configs"][0]["config"])
    stats = D.load_wss_stats(first_cfg.data.wss_stats_path)
    train = D.load_partition(
        first_cfg.data.split_path,
        "train",
        stats,
        strict=True,
        target=first_cfg.data.target,
        target_normalization=first_cfg.data.target_normalization,
        data_root=first_cfg.data.data_root,
        required_frame_version=first_cfg.data.required_frame_version,
        case_features_path=first_cfg.data.case_features_path,
    )
    feat_stats = json.loads(
        Path(first_cfg.data.feature_stats_path).read_text(encoding="utf-8")
    )
    quantiles = D.compute_train_weight_quantiles(train)

    jobs = []
    for row in manifest["configs"]:
        path = Path(row["config"])
        if sha256(path) != row["config_sha256"]:
            raise RuntimeError(f"config drift: {path}")
        cfg = C.ExpConfig.from_json(path)
        if (
            cfg.data.split_path != first_cfg.data.split_path
            or cfg.data.data_root != first_cfg.data.data_root
            or cfg.data.target != first_cfg.data.target
            or cfg.data.target_normalization
            != first_cfg.data.target_normalization
            or cfg.data.feature_stats_path != first_cfg.data.feature_stats_path
            or cfg.data.case_features_path != first_cfg.data.case_features_path
        ):
            raise RuntimeError(f"data contract drift: {path}")
        cfg.train.y_norm_q02 = quantiles["y_norm_q02"]
        cfg.train.y_norm_q98 = quantiles["y_norm_q98"]
        cfg.train._raw_p90 = quantiles["raw_p90"]  # type: ignore[attr-defined]
        dataset = D.WSSMinDataset(
            train, cfg.data, feat_stats, training=True, base_seed=cfg.train.seed
        )
        examples = [dataset[0], dataset[1]]
        same_flags = [
            bool(torch.equal(item["support_idx"], item["query_idx"]))
            for item in examples
        ]
        if cfg.data.query_mode == "same" and not all(same_flags):
            raise RuntimeError("SAME arm did not reuse support indices")
        if cfg.data.query_mode == "independent" and all(same_flags):
            raise RuntimeError("IND arm did not produce off-support queries")
        batch = D.collate(examples)
        model = build_model(cfg.model, C.input_dim(cfg)).cuda().train()
        active_local_stages = [
            index + 1
            for index, stage in enumerate(model.sa)
            if stage.local_transformer is not None
        ]
        if active_local_stages != [2]:
            raise RuntimeError(
                f"expected only SA2 local Transformer, got {active_local_stages}"
            )
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
            raise RuntimeError(f"non-finite smoke result: {path}")
        loss.backward()
        gradients = model.head[-1].weight.grad
        if gradients is None or not torch.isfinite(gradients).all():
            raise RuntimeError(f"missing/non-finite output gradients: {path}")
        if cfg.train.loss_hotspot_bce_lambda > 0 and (
            prediction.ndim != 2
            or prediction.shape[1] != 2
            or gradients[1].abs().sum() <= 0
        ):
            raise RuntimeError("H1 did not train the auxiliary hotspot output")
        if cfg.train.loss_pinball_lambda > 0 and gradients[0].abs().sum() <= 0:
            raise RuntimeError("H2 did not train the regression output")
        jobs.append(
            {
                "experiment_id": row["experiment_id"],
                "config": str(path.resolve()),
                "config_sha256": row["config_sha256"],
                "query_mode": cfg.data.query_mode,
                "support_query_identical_per_case": same_flags,
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.detach().cpu()),
                "output_gradient_l1": [
                    float(value)
                    for value in gradients.detach().abs().sum(dim=1).cpu()
                ],
                "active_local_transformer_stages": active_local_stages,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
        )
        del model, batch, dataset
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
