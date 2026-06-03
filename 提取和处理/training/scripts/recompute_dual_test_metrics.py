"""对已训练完成的 run 目录：在 test 上重算 best_model.pt 与 best_wss_model.pt，并入 summary.json。

口径与 ``train_field`` 中 ``FieldTrainer.evaluate``（双域 loss、WSS meter）一致。

用法（仓库根目录、GNN 环境）::

    python -m training.scripts.recompute_dual_test_metrics \\
      --run-dir outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260508_001936
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..core.config import ExperimentConfig, resolve_wss_effective_dim, resolve_wss_runtime_names
from ..core.data import (
    FieldGraphDataset,
    build_dataloader,
    build_feature_mask,
    build_required_data_keys,
)
from ..core.models import build_field_model_from_config
from ..core.splits import SplitSpec
from ..core.trainer import FieldTrainer
from ..core.utils import dump_json, resolve_device, set_seed


def _rebuild_trainer_and_test_loader(
    config: ExperimentConfig,
    split: SplitSpec,
    device: torch.device,
) -> tuple[FieldTrainer, Any]:
    train_case_dirs = [Path(config.data.data_root) / case_name for case_name in split.train_cases]
    config.physics.resolve_scales_from_data(
        data_root=config.data.data_root,
        graphs_subdir=config.data.graphs_subdir,
        case_dirs=train_case_dirs,
    )

    set_seed(config.system.seed, deterministic=config.system.deterministic)
    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )
    eff_wss_dim = resolve_wss_effective_dim(
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    wss_target_names = resolve_wss_runtime_names(
        config.data.wss_target_frame,
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    required_data_keys = build_required_data_keys(
        config.model.name,
        wss_dim=eff_wss_dim,
        wss_target_frame=config.data.wss_target_frame,
    )

    test_dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.test_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_data_keys,
        wss_target_frame=config.data.wss_target_frame,
        wss_domain_norm=config.data.wss_domain_norm,
        wss_domain_norm_stats=config.data.wss_domain_norm_stats,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    model = build_field_model_from_config(config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    warmup_epochs = config.optim.warmup_epochs
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.optim.scheduler_factor,
        patience=config.optim.scheduler_patience,
    )

    wss_weights_tensor = (
        torch.tensor(config.optim.wss_weights, dtype=torch.float32)
        if eff_wss_dim > 0
        else None
    )
    norm_params_path = str(Path(config.data.data_root) / "normalization_params_global.json")

    trainer = FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_weights=torch.tensor(config.optim.target_weights, dtype=torch.float32),
        grad_clip_norm=config.optim.grad_clip_norm,
        physics_config=config.physics,
        interior_loss_boost=config.optim.interior_loss_boost,
        accumulate_grad_batches=config.optim.accumulate_grad_batches,
        log_dir=None,
        use_amp=config.system.amp,
        warmup_scheduler=warmup_scheduler,
        warmup_epochs=warmup_epochs,
        wss_loss_weight=config.optim.wss_loss_weight,
        wss_weights=wss_weights_tensor,
        early_stop_wss_weight=config.optim.early_stop_wss_weight,
        wss_loss_type=config.optim.wss_loss_type,
        wss_huber_beta=config.optim.wss_huber_beta,
        domain_loss_config=config.optim.domain_loss,
        norm_params_path=norm_params_path,
        early_stop_min_delta=config.optim.early_stop_min_delta,
        val_score_ema_alpha=config.optim.val_score_ema_alpha,
        val_score_wss_weights=config.optim.val_score_wss_weights,
        wss_target_names=wss_target_names or None,
        wss_target_frame=config.data.wss_target_frame,
        wss_output_mode=config.model.wss_output_mode,
        wss_metric_dim=config.model.wss_metric_dim if config.model.wss_output_mode == "vel_diff" else 0,
    )
    return trainer, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="补算 dual checkpoint 的 test 指标并写回 summary")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run 目录（含 config.snapshot.json、split.snapshot.json、*.pt）",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)

    cfg_path = run_dir / "config.snapshot.json"
    spl_path = run_dir / "split.snapshot.json"
    if not cfg_path.is_file() or not spl_path.is_file():
        raise SystemExit(f"缺少 snapshot: {cfg_path} 或 {spl_path}")

    config = ExperimentConfig.from_json(cfg_path)
    config.validate()
    split = SplitSpec.from_json(spl_path)

    device = resolve_device(config.system.device)
    trainer, test_loader = _rebuild_trainer_and_test_loader(config, split, device)

    ckpt_main = run_dir / "best_model.pt"
    ckpt_wss = run_dir / "best_wss_model.pt"
    if not ckpt_main.is_file():
        raise SystemExit(f"缺少 {ckpt_main}")

    test_metrics = trainer.evaluate(test_loader, checkpoint_path=ckpt_main)
    test_best_wss: Optional[Dict[str, float]] = None
    if ckpt_wss.is_file():
        test_best_wss = trainer.evaluate(test_loader, checkpoint_path=ckpt_wss)

    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    summary["test_metrics"] = test_metrics
    if test_best_wss is not None:
        summary["test_metrics_best_wss"] = test_best_wss
    dump_json(summary, summary_path)

    keys_p_wss = (
        "rmse_p",
        "mae_p",
        "r2_p",
        "loss_wall_pressure",
        "loss_wall_wss",
        "wss_rmse_wss",
        "wss_r2_wss",
        "wss_rmse",
    )

    def _pick(m: Dict[str, float]) -> Dict[str, float]:
        return {k: float(m[k]) for k in keys_p_wss if k in m}

    print("[recompute] 已更新", summary_path)
    print("  best_model:     ", _pick(test_metrics))
    if test_best_wss is not None:
        print("  best_wss_model: ", _pick(test_best_wss))
    else:
        print("  （无 best_wss_model.pt）")


if __name__ == "__main__":
    main()
