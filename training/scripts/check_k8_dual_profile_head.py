#!/usr/bin/env python3
"""K8/K9 落地自检（§15.1/15.2）：双分量剖面头 + L_ray + sidecar 数据链路。

检查项（全过才可提交 GPU probe）：
  1. 旧配置回归：K7 probe 配置（scalar variant）构建/前向不受影响；
  2. dual 头前向/反向：输出 (N,4) 有限、coef 双通道梯度有限非零；
  3. dataset sidecar 附加 + PyG Batch 拼接节点对齐；
  4. DualDomainLoss L_ray：lambda>0 时项非零、可反向；lambda=0 时旧口径不变；
  5. config 校验：dual 缺 sidecar / lambda_wss_ray 缺 dual / 旋转增强组合均须报错。

用法::

    python -m training.scripts.check_k8_dual_profile_head
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from ..core.config import ExperimentConfig
from ..core.data import FieldGraphDataset, build_dataloader
from ..core.losses import build_loss_plugin
from ..core.models import build_field_model_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]
K7_CONFIG = (
    REPO_ROOT
    / "training/configs/field/generated/v3_pointcloud/V3P-K7-ProfileWSS-Probe-post5463_seed1.json"
)
SIDE_SUBDIR = "processed/ray_targets_k16"
# smoke sidecar 已覆盖的病例（run_v3p_k9_ray_sidecar_gen --smoke 前 4 个 train case）
SMOKE_CASES = ["slow/LI_HUAN_GE", "slow/WANG_GUI"]


def _dual_cfg_dict() -> dict:
    raw = json.loads(K7_CONFIG.read_text())
    cfg = copy.deepcopy(raw)
    cfg["model"]["wss_profile_variant"] = "dual"
    cfg["data"]["ray_sidecar_subdir"] = SIDE_SUBDIR
    cfg["data"]["ray_target_scale"] = 22.726
    cfg["optim"]["domain_loss"]["lambda_wss_ray"] = 0.1
    return cfg


def _expect_error(tag: str, cfg_dict: dict, needle: str) -> None:
    try:
        ExperimentConfig.from_dict(copy.deepcopy(cfg_dict)).validate()
    except ValueError as e:
        assert needle in str(e), f"[{tag}] 报错内容不符: {e}"
        print(f"[5] 校验 {tag}: OK（{e}）")
        return
    raise AssertionError(f"[{tag}] 应报错未报错")


def main() -> None:
    torch.manual_seed(0)

    # --- 1. 旧配置回归（scalar） ---
    k7 = ExperimentConfig.from_json(K7_CONFIG)
    k7.validate()
    model_k7 = build_field_model_from_config(k7)
    n_params_k7 = sum(p.numel() for p in model_k7.parameters())
    print(f"[1] K7 scalar 构建 OK（参数 {n_params_k7:,}）")

    # --- 3. dataset sidecar 附加 + batch 拼接 ---
    dual_cfg = ExperimentConfig.from_dict(_dual_cfg_dict())
    dual_cfg.validate()
    ds = FieldGraphDataset(
        root=dual_cfg.data.data_root,
        case_names=SMOKE_CASES,
        graphs_subdir=dual_cfg.data.graphs_subdir,
        augment=False,
        ray_sidecar_subdir=dual_cfg.data.ray_sidecar_subdir,
        ray_target_scale=dual_cfg.data.ray_target_scale,
    )
    # smoke 只生成了每病例前 2 图；只取有 sidecar 的样本
    keep = [
        i for i, p in enumerate(ds.data_files)
        if (p.parents[2] / SIDE_SUBDIR / f"{p.stem}.pt").exists()
    ][:2]
    assert len(keep) == 2, f"缺 smoke sidecar，可先跑 run_v3p_k9_ray_sidecar_gen --smoke（found={len(keep)}）"
    ds.data_files = [ds.data_files[i] for i in keep]
    loader = build_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    n = batch.x.size(0)
    for key in ("ray_ts", "ray_tc", "ray_a1", "ray_valid"):
        t = getattr(batch, key)
        assert t.size(0) == n, f"{key} 未按节点拼接: {t.shape} vs N={n}"
    wall = batch.x[:, 9].bool()
    assert bool(batch.ray_valid[~wall].sum() == 0), "内部点不应有有效 ray 目标"
    frac = batch.ray_valid[wall].float().mean().item()
    print(f"[3] sidecar 附加/拼接 OK（N={n} · 壁面 ray_valid 占比 {frac:.3f}）")

    # --- 2. dual 前向/反向 ---
    model = build_field_model_from_config(dual_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    out = model(batch)
    pred, wss_pred = out
    assert wss_pred.shape == (n, 4), wss_pred.shape
    assert torch.isfinite(wss_pred).all(), "dual 头输出含非有限值"
    a1 = model.wss_profile_head.last_a1
    assert a1 is not None and a1.shape == (n, 2)
    print(f"[2] dual 前向 OK（参数 {n_params:,} · vs K7 {n_params_k7:,}）")

    # --- 4. L_ray ---
    plugin = build_loss_plugin(
        None,
        wss_weights=torch.tensor(dual_cfg.optim.wss_weights),
        wss_loss_type=dual_cfg.optim.wss_loss_type,
        domain_loss_config=dual_cfg.optim.domain_loss,
    )
    breakdown = plugin.build_loss(
        model=model, batch=batch, pred=pred, target=batch.y,
        data_weights=torch.ones(4), epoch=0, train=True, wss_pred=wss_pred,
    )
    ray_val = breakdown.loss_wss_ray.item()
    assert ray_val > 0, "lambda_wss_ray>0 但 L_ray=0"
    breakdown.total_loss.backward()
    coef_grad = model.wss_profile_head.coef_head[-1].weight.grad
    assert coef_grad is not None and torch.isfinite(coef_grad).all() and coef_grad.abs().sum() > 0
    sd = breakdown.scalar_dict()
    assert "loss_wss_ray" in sd and "weighted_loss_wss_ray" in sd
    print(f"[4] L_ray OK（loss_wss_ray={ray_val:.4f} · weighted={sd['weighted_loss_wss_ray']:.5f} · coef 梯度有限非零）")

    # lambda=0 → 项为零（K8-only 档口径）
    cfg0 = _dual_cfg_dict()
    cfg0["optim"]["domain_loss"]["lambda_wss_ray"] = 0.0
    dual0 = ExperimentConfig.from_dict(cfg0)
    plugin0 = build_loss_plugin(
        None,
        wss_weights=torch.tensor(dual0.optim.wss_weights),
        wss_loss_type=dual0.optim.wss_loss_type,
        domain_loss_config=dual0.optim.domain_loss,
    )
    b0 = plugin0.build_loss(
        model=model, batch=batch, pred=pred.detach(), target=batch.y,
        data_weights=torch.ones(4), epoch=0, train=True, wss_pred=wss_pred.detach(),
    )
    assert b0.loss_wss_ray.item() == 0.0 and b0.weighted_loss_wss_ray.item() == 0.0
    print("[4b] lambda_wss_ray=0 时 L_ray 项为零（K8-only 口径）OK")

    # --- 5. config 校验 ---
    bad = _dual_cfg_dict()
    bad["data"]["ray_sidecar_subdir"] = ""
    bad["optim"]["domain_loss"]["lambda_wss_ray"] = 0.0
    _expect_error("dual缺sidecar", bad, "ray_sidecar_subdir")

    bad = _dual_cfg_dict()
    bad["model"]["wss_profile_variant"] = "scalar"
    _expect_error("lambda_ray缺dual", bad, "lambda_wss_ray")

    bad = _dual_cfg_dict()
    bad["data"]["augment_config"]["rotation_prob"] = 0.5
    _expect_error("旋转增强组合", bad, "仅允许平移")

    print("\n全部自检通过 ✅")


if __name__ == "__main__":
    main()
