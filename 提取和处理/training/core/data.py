from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
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
    # 对每个节点特征名生成一个 0/1 开关。
    node_mask = torch.tensor(
        [1.0 if name in enabled_node_features else 0.0 for name in NODE_FEATURE_NAMES],
        dtype=torch.float32,
    )
    # 对每个全局条件名也生成一个 0/1 开关。
    global_mask = torch.tensor(
        [1.0 if name in enabled_global_features else 0.0 for name in GLOBAL_COND_NAMES],
        dtype=torch.float32,
    )
    # 打包成 FeatureMask 返回。
    return FeatureMask(node_mask=node_mask, global_mask=global_mask)


def build_required_data_keys(model_name: str) -> Set[str]:
    # 训练时只保留前向与 loss 真正会访问的字段，避免把图文件里的冗余属性反复搬运。
    # 所有模型都至少需要节点输入、监督目标和图级条件。
    required_keys: Set[str] = {"x", "y", "global_cond"}
    # 图模型额外需要 edge_index 来做消息传递。
    if model_name in {"graphsage", "transformer", "meshgraphnet", "pointnetpp"}:
        required_keys.add("edge_index")
    return required_keys


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
        required_keys: Optional[Set[str]] = None,
    ):
        # root/case_names/graphs_subdir 共同决定这次实验读取哪些图。
        # 这里约定“患者级 split -> 病例目录 -> 图快照文件”三层路径关系。
        # 根目录是全部病例所在位置。
        self.root = Path(root)
        # 当前数据集实际包含哪些病例名。
        self.case_names = case_names
        # 每个病例下面图数据子目录的相对路径。
        self.graphs_subdir = graphs_subdir
        # 记录 graphs_subdir 有几层目录，后面用于从图文件路径反推病例名。
        self._subdir_depth = len(Path(graphs_subdir).parts)
        # 是否启用训练时数据增强。
        self.augment = augment
        # 没给增强配置时，使用默认增强参数并复制一份，避免原地污染。
        self.augment_config = (augment_config or DEFAULT_AUGMENT_CONFIG.copy()).copy()
        # 是否在初始化阶段预载入图到内存。
        self.preload = preload
        # 特征屏蔽配置。
        self.feature_mask = feature_mask
        # 需要保留的数据字段集合；None 表示不裁剪。
        self.required_keys = set(required_keys) if required_keys is not None else None

        # 保存当前数据集对应的全部图文件路径。
        self.data_files: List[Path] = []
        for case_name in case_names:
            # 当前病例图数据目录。
            case_dir = self.root / case_name / graphs_subdir
            # 病例目录不存在时直接跳过。
            if not case_dir.exists():
                continue
            # 读取该病例下全部 .pt 图文件，并按文件名排序保证顺序稳定。
            self.data_files.extend(sorted(case_dir.glob("*.pt")))

        # 如果一个图文件都没找到，直接报错提醒配置有问题。
        if not self.data_files:
            raise ValueError(
                f"未找到图数据文件 root={self.root}, cases={case_names}, subdir={graphs_subdir}"
            )

        # 默认不开缓存。
        self.cached_data = None
        if preload:
            # preload 适合数据量不大、I/O 较慢的场景；服务器显存/内存紧张时不要开。
            # 预加载时先读图，再做字段裁剪，把静态部分缓存到内存中。
            self.cached_data = [self._prepare_static_data(load_graph_data(path)) for path in self.data_files]

    def __len__(self) -> int:
        # 数据集长度就是图文件个数。
        return len(self.data_files)

    def _prepare_static_data(self, data):
        # 如果没有要求裁剪字段，就原样返回。
        if self.required_keys is None:
            return data
        # 找出所有“不在必需集合里”的字段。
        drop_keys = [key for key in data.keys() if key not in self.required_keys]
        # 原地删除多余字段，减少后续 DataLoader 搬运量。
        for key in drop_keys:
            del data[key]
        return data

    def __getitem__(self, idx: int):
        # 先定位第 idx 个样本对应的图文件路径。
        data_path = self.data_files[idx]
        # 如果已预加载，则从缓存里取并 clone，避免增强或 mask 污染缓存。
        if self.cached_data is not None:
            data = self.cached_data[idx].clone()
        else:
            # 否则按需现读，并裁剪到训练真正需要的字段。
            data = self._prepare_static_data(load_graph_data(data_path))

        if self.augment:
            # 数据增强只在训练集启用；验证/测试必须使用原始图，避免评估口径漂移。
            data = apply_augmentations(data, self.augment_config)

        if self.feature_mask is not None:
            # 把节点特征 mask 挪到样本当前设备。
            node_mask = self.feature_mask.node_mask.to(data.x.device)
            # 把全局特征 mask 挪到 global_cond 所在设备。
            global_mask = self.feature_mask.global_mask.to(data.global_cond.device)
            # 使用乘法 mask 保留原始张量形状，便于和现有模型、checkpoint、导出逻辑兼容。
            # 节点特征按列屏蔽。
            data.x.mul_(node_mask.unsqueeze(0))
            # 图级条件按列屏蔽。
            data.global_cond.mul_(global_mask.unsqueeze(0))

        # 保留样本级元信息，便于后续回溯和导出预测结果。
        # 从 data_path 逆推 case 相对路径，兼容 ILO 嵌套 (case_name 含 '/')。
        # 先从图文件路径开始往上回退目录。
        case_path = data_path
        for _ in range(self._subdir_depth + 1):
            case_path = case_path.parent
        # sample_id 用图文件名（不含后缀）表示当前快照。
        data.sample_id = data_path.stem
        # case_name 是相对于数据根目录的病例路径。
        data.case_name = str(case_path.relative_to(self.root))
        # graph_path 保存原始图文件完整路径，便于追溯。
        data.graph_path = str(data_path)

        return data


def _worker_init_fn(worker_id: int):
    """Seed each DataLoader worker for reproducibility."""
    # 从 PyTorch worker 的初始种子派生本 worker 的随机种子。
    worker_seed = torch.initial_seed() % (2**32)
    # 固定 NumPy 的 worker 随机状态。
    np.random.seed(worker_seed + worker_id)
    import random
    # 固定 Python random 的 worker 随机状态。
    random.seed(worker_seed + worker_id)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None,
):
    # 默认不显式传 generator，让 DataLoader 使用内部默认行为。
    generator = None
    # 只有给了 seed 时才为 DataLoader 构造独立随机数生成器。
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    # 统一从这里构造 PyG DataLoader。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
