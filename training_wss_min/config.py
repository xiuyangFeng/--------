#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""training_wss_min 实验配置（配置驱动 baseline）。

一个实验 = 一个 JSON 配置。用 `ExpConfig.from_json` 读、`to_json` 写；
`make_configs.py` 批量生成 sweep 配置。训练/评估都只吃 config 路径。

约定：所有相对路径都相对仓库根 `PROJECT_ROOT`；产物落在 `runs/<name>/`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data_wss_min"
BUNDLE_GLOB = "AG/*/*/bundle.npz"          # data_wss_min 下 cohort/case
GLOBAL_STATS = DATA_ROOT / "wss_global_stats.json"
DEFAULT_SPLIT = PROJECT_ROOT / "training" / "splits" / "split_AG_wss_min_v1.json"
RUNS_ROOT = Path(__file__).resolve().parent / "runs"


# 可用的逐点输入特征（bundle 里都在，除 x,y,z 外均为旋转不变几何量）
FEATURE_KEYS = (
    "x", "y", "z", "dist_to_wall", "abscissa_norm",
    "local_radius", "curvature", "coord_scale",
)


@dataclass
class DataConfig:
    split_path: str = str(DEFAULT_SPLIT)
    # 训练稀疏化：每例采样多少壁面点；<=0 或 >=全量 表示用全部点
    wall_n_points: int = 2000
    sampling: str = "fps"                 # 'fps' | 'random' | 'geom_weighted'
    # geom_weighted 采样权重：w = 1 + a*curv_norm + b*(1/local_radius)_norm + c*near_bifurcation
    geom_weight_curv: float = 1.0
    geom_weight_invradius: float = 1.0
    geom_weight_bifurcation: float = 0.0  # 近原点(=分叉)加权；0 关闭
    # 每个 epoch 重新采样（增广，仅训练集）。val/test 恒用完整点云评估。
    resample_each_epoch: bool = True
    # 训练期随机 3D 旋转增广：只影响 xyz 输入列（几何特征旋转不变、标量 WSS 标签旋转不变，
    # ball-query 分组本就用相对坐标故图结构不变）。逼模型别依赖绝对朝向，缩小 val/test gap。
    rot_aug: bool = False
    # 输入特征（网络实际吃的列）
    input_features: Tuple[str, ...] = ("x", "y", "z")
    # curvature 输入的鲁棒变换；旧 run 的 feature_stats 不含 transform 时仍按旧口径评估
    curvature_transform: str = "signed_log1p"  # 'signed_log1p' | 'none'
    # 时间步：baseline 用峰值收缩期单步
    timesteps: str = "peak"               # 'peak'（其余暂不支持，占位）
    target: str = "wss"                   # 'wss'(标量) —— 矢量留待二期
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "pointnext_s"
    width: int = 32                       # stem 通道
    # 4 个 SA stage 的下采样比、ball 半径（归一化坐标下）、每组邻居上限、残差块数
    sa_ratios: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    sa_radius: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    sa_nsample: Tuple[int, ...] = (16, 16, 16, 16)
    sa_blocks: Tuple[int, ...] = (1, 1, 1, 1)   # 每 stage InvResMLP 残差块数
    invres_radius_scale: float = 1.0      # InvResMLP 分组半径 = sa_radius * scale
    fp_knn: int = 3                       # FP 插值近邻数
    head_hidden: int = 64
    out_dim: int = 1                      # 标量=1；矢量=3（二期）
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 400
    batch_cases: int = 8                  # 每步几个病例（点云）
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 1234
    # loss：标准化空间 MSE；可几何加权（更看重狭窄/分叉/高梯度处）
    loss: str = "mse"                     # 'mse' | 'huber'
    huber_delta: float = 1.0
    loss_geom_weight: bool = False        # True 则按几何量加权 loss
    loss_weight_curv: float = 1.0
    loss_weight_invradius: float = 1.0
    # 按目标幅值加权 loss：直接补"高 WSS 区 R² 崩溃"（peak 处欠拟合）。
    # 权重 += alpha * clamp01(y_norm 分位)，用训练标签本身，非泄漏。
    loss_weight_target: bool = False
    loss_weight_target_alpha: float = 2.0
    amp: bool = True
    eval_every: int = 10                  # 每多少 epoch 在 val 完整点云上评估一次
    ckpt_metric: str = "val_r2_casemean"  # 选 best 的指标（越大越好）
    log_every_steps: int = 0              # >0 时按 step 打点；0 只按 epoch


@dataclass
class ExpConfig:
    name: str = "baseline_xyz_fps2000_peak"
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- 派生路径 ----
    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.name

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @staticmethod
    def from_dict(d: dict) -> "ExpConfig":
        def _mk(cls, sub):
            sub = dict(sub or {})
            # tuple 字段 JSON 里是 list，转回 tuple 保持类型一致
            for k, v in list(sub.items()):
                if isinstance(v, list):
                    sub[k] = tuple(v)
            return cls(**sub)
        return ExpConfig(
            name=d.get("name", "unnamed"),
            notes=d.get("notes", ""),
            data=_mk(DataConfig, d.get("data")),
            model=_mk(ModelConfig, d.get("model")),
            train=_mk(TrainConfig, d.get("train")),
        )

    @staticmethod
    def from_json(path: str | Path) -> "ExpConfig":
        return ExpConfig.from_dict(json.loads(Path(path).read_text()))


def input_dim(cfg: ExpConfig) -> int:
    return len(cfg.data.input_features)


if __name__ == "__main__":
    # 冒烟：默认配置能否读写往返
    c = ExpConfig()
    p = c.to_json(RUNS_ROOT / "_tmp_config_check.json")
    c2 = ExpConfig.from_json(p)
    assert c2.model.sa_radius == c.model.sa_radius, "tuple 往返失败"
    assert input_dim(c2) == 3
    p.unlink()
    print("config OK:", c.name, "input_dim", input_dim(c2), "run_dir", c.run_dir)
