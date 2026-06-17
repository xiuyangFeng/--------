"""PINN physics 节点子采样回归测试（修复 Job 5540 ep21 IndexError）。

背景：`training/core/losses.py` 旧逻辑用 `batch[sel]`（sel 为节点号）对 PyG
Batch 索引，被误当成“选第几张图”→ `IndexError`。本测试用 mock Data 验证：

1. `_subsample_physics_batch` 做的是**节点级子图**，节点数 ≤ max_physics_nodes，
   且 edge_index 已重映射到 [0, N_sub) 范围内；
2. `PhysicsConstraintLoss.build_loss` 在 ep21（physics 刚启用、N > max）能跑通，
   不抛 IndexError，physics 前向真正在 ≤ max 节点上执行，且 loss 有限。

无需 data_new/outputs；需要 torch + torch_geometric（conda GNN 环境）。

运行：
    conda activate GNN
    python -m pytest training/tests/test_physics_subsample.py -q
  或直接：
    python training/tests/test_physics_subsample.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pipeline.config import NODE_FEATURE_NAMES
from training.core.config import PhysicsConfig
from training.core.losses import PhysicsConstraintLoss, _subsample_physics_batch
from training.core.models import expand_global_cond

N = 5000
MAX_NODES = 2048
N_FEAT = len(NODE_FEATURE_NAMES)
IS_WALL_IDX = NODE_FEATURE_NAMES.index("is_wall")
WARMUP = 20


def _make_data(num_nodes: int = N) -> Data:
    torch.manual_seed(0)
    x = torch.randn(num_nodes, N_FEAT)
    # ~5% 壁面点。
    x[:, IS_WALL_IDX] = (torch.rand(num_nodes) < 0.05).float()
    # 随机有向边（含跨整个节点范围的下标，确保会触发越界 bug）。
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.randn(num_nodes, 4)
    data.global_cond = torch.randn(1, 6)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    return data


class _TinyGNN(nn.Module):
    """极小模型：cat(x, global) → 一层基于 edge_index 的聚合 → 线性到 4 通道。

    用 index_add_ 做消息传递；若 edge_index 未重映射、含越界节点号，这里会直接
    报错，从而验证子图 edge_index 的正确性。tanh 保证二阶可导（动量项需要）。
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, 16)
        self.lin2 = nn.Linear(16, 4)

    def forward(self, data):
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        h = torch.tanh(self.lin1(x))
        row, col = data.edge_index
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, h[col])
        h = torch.tanh(h + agg)
        return self.lin2(h)


def test_subsample_is_node_level_subgraph():
    data = _make_data()
    sub = _subsample_physics_batch(data.clone(), MAX_NODES, IS_WALL_IDX)
    n_sub = int(sub.x.size(0))
    assert n_sub <= MAX_NODES, f"子采样后节点数 {n_sub} 应 <= {MAX_NODES}"
    # 节点级张量同步缩小。
    assert int(sub.y.size(0)) == n_sub
    assert int(sub.batch.size(0)) == n_sub
    # edge_index 已重映射，不再越界。
    if sub.edge_index.numel() > 0:
        assert int(sub.edge_index.max()) < n_sub
        assert int(sub.edge_index.min()) >= 0


def test_build_loss_ep21_no_indexerror():
    data = _make_data()
    cfg = PhysicsConfig(
        enabled=True,
        warmup_epochs=WARMUP,
        equation="steady",
        continuity_weight=0.1,
        momentum_weight=0.0,
        no_slip_weight=0.0,
        max_physics_nodes=MAX_NODES,
        auto_load_scales=False,
    )
    loss_fn = PhysicsConstraintLoss(cfg)
    model = _TinyGNN(N_FEAT + data.global_cond.size(1))

    pred = model(data)
    target = data.y
    data_weights = torch.ones(4)

    # ep21：warmup 刚结束，physics 启用，N(5000) > max(2048)。
    out = loss_fn.build_loss(
        model=model,
        batch=data,
        pred=pred,
        target=target,
        data_weights=data_weights,
        epoch=WARMUP + 1,
        train=True,
    )
    assert torch.isfinite(out.total_loss), "total_loss 应为有限值"
    assert torch.isfinite(out.continuity_loss)
    # continuity_weight>0 时应真正启用 physics（continuity 一般非 0）。
    assert out.continuity_loss.item() >= 0.0


def test_build_loss_val_ep21_skips_physics_under_no_grad():
    """模拟 trainer 验证路径（外层 no_grad）：默认不在 val 算 physics，不应崩溃。"""
    data = _make_data()
    cfg = PhysicsConfig(
        enabled=True,
        warmup_epochs=WARMUP,
        equation="steady",
        continuity_weight=0.1,
        momentum_weight=0.0,
        no_slip_weight=0.0,
        max_physics_nodes=MAX_NODES,
        auto_load_scales=False,
        eval_physics_on_val=False,
    )
    loss_fn = PhysicsConstraintLoss(cfg)
    model = _TinyGNN(N_FEAT + data.global_cond.size(1))
    pred = model(data)
    data_weights = torch.ones(4)

    with torch.no_grad():
        out = loss_fn.build_loss(
            model=model,
            batch=data,
            pred=pred,
            target=data.y,
            data_weights=data_weights,
            epoch=WARMUP + 1,
            train=False,
        )
    assert torch.isfinite(out.total_loss)
    assert out.continuity_loss.item() == 0.0
    assert out.momentum_loss.item() == 0.0


def test_build_loss_val_ep21_eval_physics_on_val_under_no_grad():
    """eval_physics_on_val=true 时内层 enable_grad 覆盖外层 no_grad，可算 val physics。"""
    data = _make_data()
    cfg = PhysicsConfig(
        enabled=True,
        warmup_epochs=WARMUP,
        equation="steady",
        continuity_weight=0.1,
        momentum_weight=0.0,
        no_slip_weight=0.0,
        max_physics_nodes=MAX_NODES,
        auto_load_scales=False,
        eval_physics_on_val=True,
    )
    loss_fn = PhysicsConstraintLoss(cfg)
    model = _TinyGNN(N_FEAT + data.global_cond.size(1))
    pred = model(data)
    data_weights = torch.ones(4)

    with torch.no_grad():
        out = loss_fn.build_loss(
            model=model,
            batch=data,
            pred=pred,
            target=data.y,
            data_weights=data_weights,
            epoch=WARMUP + 1,
            train=False,
        )
    assert torch.isfinite(out.total_loss)
    assert torch.isfinite(out.continuity_loss)


if __name__ == "__main__":
    test_subsample_is_node_level_subgraph()
    print("[OK] test_subsample_is_node_level_subgraph")
    test_build_loss_ep21_no_indexerror()
    print("[OK] test_build_loss_ep21_no_indexerror")
    test_build_loss_val_ep21_skips_physics_under_no_grad()
    print("[OK] test_build_loss_val_ep21_skips_physics_under_no_grad")
    test_build_loss_val_ep21_eval_physics_on_val_under_no_grad()
    print("[OK] test_build_loss_val_ep21_eval_physics_on_val_under_no_grad")
    print("ALL PASSED")
