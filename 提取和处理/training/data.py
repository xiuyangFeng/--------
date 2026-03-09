from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from pipeline.augmentation import DEFAULT_AUGMENT_CONFIG, apply_augmentations
from pipeline.config import GLOBAL_COND_NAMES, NODE_FEATURE_NAMES
from pipeline.dataset import load_graph_data


@dataclass
class FeatureMask:
    # 不直接裁掉特征维度，而是保留定长输入并按位 mask。
    # 这样做的好处是：
    # 1. 不需要因为消融而反复改模型输入维度
    # 2. checkpoint / 导出格式在不同实验间保持兼容
    node_mask: torch.Tensor
    global_mask: torch.Tensor


def build_feature_mask(
    enabled_node_features: List[str],
    enabled_global_features: List[str],
) -> FeatureMask:
    # 这里返回 0/1 mask，而不是裁剪特征维度。
    # 这样模型输入维度保持稳定，做消融时不需要反复改模型定义。
    node_mask = torch.tensor(
        [1.0 if name in enabled_node_features else 0.0 for name in NODE_FEATURE_NAMES],
        dtype=torch.float32,
    )
    global_mask = torch.tensor(
        [1.0 if name in enabled_global_features else 0.0 for name in GLOBAL_COND_NAMES],
        dtype=torch.float32,
    )
    return FeatureMask(node_mask=node_mask, global_mask=global_mask)


class FieldGraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        case_names: List[str],
        graphs_subdir: str = "processed/graphs",
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        preload: bool = False,
        feature_mask: Optional[FeatureMask] = None,
    ):
        # root/case_names/graphs_subdir 共同决定这次实验读取哪些图。
        # 这里约定“患者级 split -> 病例目录 -> 图快照文件”三层路径关系。
        self.root = Path(root)
        self.case_names = case_names
        self.graphs_subdir = graphs_subdir
        self.augment = augment
        self.augment_config = (augment_config or DEFAULT_AUGMENT_CONFIG.copy()).copy()
        self.preload = preload
        self.feature_mask = feature_mask

        self.data_files: List[Path] = []
        for case_name in case_names:
            case_dir = self.root / case_name / graphs_subdir
            if not case_dir.exists():
                continue
            self.data_files.extend(sorted(case_dir.glob("*.pt")))

        if not self.data_files:
            raise ValueError(
                f"未找到图数据文件 root={self.root}, cases={case_names}, subdir={graphs_subdir}"
            )

        self.cached_data = None
        if preload:
            # preload 适合数据量不大、I/O 较慢的场景；服务器显存/内存紧张时不要开。
            self.cached_data = [load_graph_data(path) for path in self.data_files]

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int):
        data_path = self.data_files[idx]
        if self.cached_data is not None:
            data = self.cached_data[idx].clone()
        else:
            data = load_graph_data(data_path)

        if self.augment:
            # 数据增强只在训练集启用；验证/测试必须使用原始图，避免评估口径漂移。
            data = apply_augmentations(data, self.augment_config)

        if self.feature_mask is not None:
            node_mask = self.feature_mask.node_mask.to(data.x.device)
            global_mask = self.feature_mask.global_mask.to(data.global_cond.device)
            # 使用乘法 mask 保留原始张量形状，便于和现有模型、checkpoint、导出逻辑兼容。
            data.x = data.x * node_mask.unsqueeze(0)
            data.global_cond = data.global_cond * global_mask.unsqueeze(0)

        # 保留样本级元信息，便于后续回溯和导出预测结果。
        data.sample_id = data_path.stem
        data.case_name = data_path.parent.parent.name
        data.graph_path = str(data_path)

        return data


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
