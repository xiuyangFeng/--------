#!/usr/bin/env python3
"""Formal CUDA smoke for the LSA2-H2 log(local_radius) matrix."""

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

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.evaluate import predict_case_norm
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(model, batch: dict, cfg: C.ExpConfig) -> torch.Tensor:
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


def smoke_one(row: dict) -> dict:
    path = Path(row["config"])
    if sha256(path) != row["config_sha256"]:
        raise RuntimeError(f"config drift: {path}")
    cfg = C.ExpConfig.from_json(path)
    if (
        cfg.data.query_mode != "same"
        or cfg.train.seed != 1234
        or float(cfg.train.loss_pinball_lambda) != 0.20
        or float(cfg.train.pinball_quantile) != 0.90
        or tuple(cfg.model.local_transformer_stages) != (2,)
        or tuple(cfg.model.local_geope_feature_indices) != (3, 4, 5)
        or float(cfg.model.drop_path_rate) != 0.10
        or cfg.data.case_features_path is not None
    ):
        raise RuntimeError(f"frozen H2 anchor contract drift: {path}")
    expected_log = row["role"] == "logradius_treatment"
    if ("log_local_radius" in cfg.data.input_features) != expected_log:
        raise RuntimeError("log-radius role/config mismatch")
    if len(cfg.data.input_features) != row["input_dim"]:
        raise RuntimeError("input dimension drift")

    target_stats = D.load_wss_stats(cfg.data.wss_stats_path)
    train = D.load_partition(
        cfg.data.split_path,
        "train",
        target_stats,
        strict=True,
        target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
        case_features_path=cfg.data.case_features_path,
    )
    test = D.load_partition(
        cfg.data.split_path,
        "test",
        target_stats,
        strict=True,
        target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
        case_features_path=cfg.data.case_features_path,
    )
    feature_stats_path = Path(cfg.data.feature_stats_path)
    if (
        str(feature_stats_path.resolve()) != row["feature_stats"]
        or sha256(feature_stats_path) != row["feature_stats_sha256"]
    ):
        raise RuntimeError("feature-stat drift")
    feature_stats = json.loads(feature_stats_path.read_text(encoding="utf-8"))
    if expected_log and "log_local_radius" not in feature_stats:
        raise RuntimeError("treatment feature stats omit log_local_radius")
    if not expected_log and "log_local_radius" in cfg.data.input_features:
        raise RuntimeError("control unexpectedly includes log_local_radius")

    quantiles = D.compute_train_weight_quantiles(train)
    cfg.train.y_norm_q02 = quantiles["y_norm_q02"]
    cfg.train.y_norm_q98 = quantiles["y_norm_q98"]
    cfg.train._raw_p90 = quantiles["raw_p90"]  # type: ignore[attr-defined]
    dataset = D.WSSMinDataset(
        train,
        cfg.data,
        feature_stats,
        training=True,
        base_seed=cfg.train.seed,
    )
    examples = [dataset[0], dataset[1]]
    if not all(
        torch.equal(item["support_idx"], item["query_idx"])
        for item in examples
    ):
        raise RuntimeError("SAME control did not reuse support/query indices")
    if expected_log:
        feature_index = cfg.data.input_features.index("log_local_radius")
        item = examples[0]
        case = train[0]
        idx = item["support_idx"].numpy()
        stat = feature_stats["log_local_radius"]
        expected = (
            np.log(np.asarray(case["local_radius"][idx], dtype=np.float64))
            - stat["mean"]
        ) / stat["std"]
        actual = item["support_x"][:, feature_index].numpy()
        if (
            not np.isfinite(actual).all()
            or not np.allclose(actual, expected, rtol=1e-5, atol=1e-5)
        ):
            raise RuntimeError("treatment column is not standardized log radius")

    batch = D.collate(examples)
    model = build_model(cfg.model, C.input_dim(cfg)).cuda().train()
    active_local_stages = [
        index + 1
        for index, stage in enumerate(model.sa)
        if stage.local_transformer is not None
    ]
    if active_local_stages != [2]:
        raise RuntimeError(
            f"expected only SA2 Local Transformer, got {active_local_stages}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    prediction = forward(model, batch, cfg)
    loss = compute_loss(prediction, batch, cfg.train, "cuda", target_stats)
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise RuntimeError("non-finite forward/loss")
    loss.backward()
    head_gradient = model.head[-1].weight.grad
    if (
        head_gradient is None
        or not torch.isfinite(head_gradient).all()
        or head_gradient[0].abs().sum() <= 0
    ):
        raise RuntimeError("H2 regression output has no finite gradient")
    stem_gradient = next(model.stem.parameters()).grad
    if stem_gradient is None or not torch.isfinite(stem_gradient).all():
        raise RuntimeError("input stem has no finite gradient")
    optimizer.step()
    torch.cuda.synchronize()

    with tempfile.TemporaryDirectory(prefix="lsa2_h2_logradius_smoke_") as tmp:
        checkpoint = Path(tmp) / "checkpoint.pt"
        torch.save({"model": model.state_dict()}, checkpoint)
        restored = build_model(cfg.model, C.input_dim(cfg)).cuda()
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        restored.load_state_dict(payload["model"], strict=True)
        restored.eval()
        probe = min(test, key=lambda case: len(case["pos"]))
        original_chunk = cfg.eval.query_chunk_size
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
        cfg.eval.query_chunk_size = original_chunk
        maximum = float(np.max(np.abs(full - chunk)))
        if (
            not np.isfinite(full).all()
            or not np.isfinite(chunk).all()
            or not np.allclose(full, chunk, rtol=2e-5, atol=2e-5)
        ):
            raise RuntimeError(
                f"full/chunk mismatch or non-finite output ({maximum})"
            )

    result = {
        "experiment_id": row["experiment_id"],
        "role": row["role"],
        "config": str(path.resolve()),
        "config_sha256": sha256(path),
        "feature_stats": str(feature_stats_path.resolve()),
        "feature_stats_sha256": sha256(feature_stats_path),
        "cuda_device": torch.cuda.get_device_name(0),
        "input_features": list(cfg.data.input_features),
        "input_dim": C.input_dim(cfg),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "active_local_transformer_stages": active_local_stages,
        "loss": float(loss.detach().cpu()),
        "prediction_shape": list(prediction.shape),
        "checkpoint_strict_reload": "passed",
        "full_chunk_max_abs": maximum,
        "finite": True,
    }
    del (
        model,
        restored,
        optimizer,
        batch,
        dataset,
        prediction,
        loss,
        train,
        test,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal smoke")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    static = json.loads(args.static_audit.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "prepared"
        or static.get("status") != "passed"
        or static.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise RuntimeError("manifest/static audit missing, failed, or stale")
    jobs, error = [], None
    try:
        for row in manifest["configs"]:
            jobs.append(smoke_one(row))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if error is None else "failed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "static_audit": str(args.static_audit.resolve()),
        "static_audit_sha256": sha256(args.static_audit),
        "jobs": jobs,
    }
    if error is not None:
        payload["error"] = error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "jobs": len(jobs),
                "output": str(args.output),
                "error": error,
            },
            ensure_ascii=False,
        )
    )
    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
