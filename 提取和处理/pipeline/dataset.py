#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFD 图数据集模块

提供用于 GNN 训练的 PyTorch Geometric 数据集类，支持在线数据增强。

主要类:
- CFDGraphDataset: 基础数据集，从 .pt 文件加载图数据
- CFDAugmentedDataset: 带在线增强的数据集

使用示例:
  from pipeline.dataset import CFDAugmentedDataset
  
  # 训练集（启用增强）
  train_dataset = CFDAugmentedDataset(
      root='data_new/AG/fast',
      case_names=['ZHANG_CHUN', 'LI_SI'],
      augment=True,
  )
  
  # 验证集（禁用增强）
  val_dataset = CFDAugmentedDataset(
      root='data_new/AG/fast',
      case_names=['WANG_WU'],
      augment=False,
  )
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Callable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# 导入增强函数
try:
    from .augmentation import (
        random_rotation,
        random_translation,
        small_scale_augmentation,
        mirror_augmentation,
        apply_augmentations,
        DEFAULT_AUGMENT_CONFIG,
    )
except ImportError:
    from pipeline.augmentation import (
        random_rotation,
        random_translation,
        small_scale_augmentation,
        mirror_augmentation,
        apply_augmentations,
        DEFAULT_AUGMENT_CONFIG,
    )


def load_graph_data(path: Path) -> Data:
    """兼容 PyTorch 新版本的图数据加载。"""
    return torch.load(path, weights_only=False)


def _discover_case_dirs(root: Path) -> List[Path]:
    """
    枚举 root 下所有病例目录。

    标准二层数据源直接返回 root 的一级子目录；
    ILO 这类三层嵌套会返回 <患者>-<0|1>/<before|after> 叶级目录。
    """
    case_dirs: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue

        phase_dirs = [
            leaf for leaf in sorted(child.iterdir())
            if leaf.is_dir() and not leaf.name.startswith(".") and leaf.name in {"before", "after"}
        ]
        if phase_dirs:
            case_dirs.extend(phase_dirs)
            continue

        case_dirs.append(child)

    return case_dirs


class CFDGraphDataset(Dataset):
    """
    CFD 图数据集
    
    从 processed/graphs/ 目录加载 .pt 文件。
    
    参数:
        root: 数据根目录（如 data_new/AG/fast）
        case_names: 病例名称列表，None 表示加载所有病例
        graphs_subdir: 图数据子目录，默认 'processed/graphs'
        transform: 数据变换函数（可选）
    """
    
    def __init__(
        self,
        root: str,
        case_names: List[str] = None,
        graphs_subdir: str = "processed/graphs",
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.graphs_subdir = graphs_subdir
        self.transform = transform
        
        # 收集所有 .pt 文件
        self.data_files = []
        self.case_indices = {}  # case_id -> (start, end)
        
        if case_names is None:
            case_dirs = _discover_case_dirs(self.root)
        else:
            case_dirs = [self.root / name for name in case_names]
        
        for case_dir in sorted(case_dirs):
            if not case_dir.exists():
                continue
            
            graphs_dir = case_dir / graphs_subdir
            if not graphs_dir.exists():
                continue
            
            pt_files = sorted(graphs_dir.glob("*.pt"))
            if pt_files:
                start_idx = len(self.data_files)
                self.data_files.extend(pt_files)
                end_idx = len(self.data_files)
                case_id = str(case_dir.relative_to(self.root))
                self.case_indices[case_id] = (start_idx, end_idx)
        
        if not self.data_files:
            raise ValueError(f"未找到任何 .pt 文件: {self.root}")
        
        print(f"📊 加载数据集: {len(self.data_files)} 个图文件, "
              f"{len(self.case_indices)} 个病例")
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Data:
        data = load_graph_data(self.data_files[idx])
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def get_case_data(self, case_name: str) -> List[Data]:
        """获取指定病例的所有数据"""
        if case_name not in self.case_indices:
            raise KeyError(f"病例不存在: {case_name}")
        
        start, end = self.case_indices[case_name]
        return [self[i] for i in range(start, end)]


class CFDAugmentedDataset(Dataset):
    """
    带在线增强的 CFD 图数据集
    
    在每次访问数据时动态应用数据增强，实现训练时的在线增强。
    
    参数:
        root: 数据根目录
        case_names: 病例名称列表
        graphs_subdir: 图数据子目录
        augment: 是否启用增强
        augment_config: 增强配置字典
        preload: 是否预加载所有数据到内存（加速训练，但占用更多内存）
    
    增强配置示例:
        {
            "rotation_prob": 0.5,      # 旋转概率
            "rotation_axes": "xyz",    # 可旋转的轴
            "translation_prob": 0.5,   # 平移概率
            "translation_range": 0.1,  # 平移范围
            "scale_prob": 0.0,         # 缩放概率（默认关闭）
            "scale_range": (0.98, 1.02),
            "mirror_prob": 0.0,        # 镜像概率（默认关闭）
        }
    """
    
    def __init__(
        self,
        root: str,
        case_names: List[str] = None,
        graphs_subdir: str = "processed/graphs",
        augment: bool = True,
        augment_config: Dict = None,
        preload: bool = False,
    ):
        self.root = Path(root)
        self.graphs_subdir = graphs_subdir
        self.augment = augment
        self.config = augment_config or DEFAULT_AUGMENT_CONFIG.copy()
        self.preload = preload
        
        # 收集文件
        self.data_files = []
        self.case_indices = {}
        
        if case_names is None:
            case_dirs = _discover_case_dirs(self.root)
        else:
            case_dirs = [self.root / name for name in case_names]
        
        for case_dir in sorted(case_dirs):
            if not case_dir.exists():
                continue
            
            graphs_dir = case_dir / graphs_subdir
            if not graphs_dir.exists():
                continue
            
            pt_files = sorted(graphs_dir.glob("*.pt"))
            if pt_files:
                start_idx = len(self.data_files)
                self.data_files.extend(pt_files)
                end_idx = len(self.data_files)
                case_id = str(case_dir.relative_to(self.root))
                self.case_indices[case_id] = (start_idx, end_idx)
        
        if not self.data_files:
            raise ValueError(f"未找到任何 .pt 文件: {self.root}")
        
        # 预加载数据
        self.cached_data = None
        if preload:
            print("📥 预加载数据到内存...")
            self.cached_data = [load_graph_data(f) for f in self.data_files]
        
        augment_status = "启用" if augment else "禁用"
        print(f"📊 数据集: {len(self.data_files)} 个样本, "
              f"{len(self.case_indices)} 个病例, "
              f"增强: {augment_status}")
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Data:
        # 加载数据
        if self.cached_data is not None:
            data = self.cached_data[idx].clone()
        else:
            data = load_graph_data(self.data_files[idx])
        
        # 应用增强
        if self.augment:
            data = apply_augmentations(data, self.config)
        
        return data
    
    def set_augment(self, augment: bool):
        """动态开启/关闭增强"""
        self.augment = augment
    
    def get_augment_config(self) -> Dict:
        """获取当前增强配置"""
        return self.config.copy()
    
    def update_augment_config(self, **kwargs):
        """更新增强配置"""
        self.config.update(kwargs)


class CFDDataModule:
    """
    数据模块，封装训练/验证/测试数据集的创建
    
    使用示例:
        dm = CFDDataModule(
            root='data_new/AG/fast',
            train_cases=['ZHANG_CHUN', 'LI_SI'],
            val_cases=['WANG_WU'],
            test_cases=['ZHAO_LIU'],
        )
        
        train_loader = dm.train_dataloader(batch_size=32)
        val_loader = dm.val_dataloader(batch_size=32)
    """
    
    def __init__(
        self,
        root: str,
        train_cases: List[str] = None,
        val_cases: List[str] = None,
        test_cases: List[str] = None,
        graphs_subdir: str = "processed/graphs",
        augment_config: Dict = None,
        preload: bool = False,
    ):
        self.root = root
        self.graphs_subdir = graphs_subdir
        self.augment_config = augment_config or DEFAULT_AUGMENT_CONFIG.copy()
        self.preload = preload
        
        self.train_cases = train_cases
        self.val_cases = val_cases
        self.test_cases = test_cases
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    @property
    def train_dataset(self) -> CFDAugmentedDataset:
        if self._train_dataset is None and self.train_cases:
            self._train_dataset = CFDAugmentedDataset(
                root=self.root,
                case_names=self.train_cases,
                graphs_subdir=self.graphs_subdir,
                augment=True,
                augment_config=self.augment_config,
                preload=self.preload,
            )
        return self._train_dataset
    
    @property
    def val_dataset(self) -> CFDAugmentedDataset:
        if self._val_dataset is None and self.val_cases:
            self._val_dataset = CFDAugmentedDataset(
                root=self.root,
                case_names=self.val_cases,
                graphs_subdir=self.graphs_subdir,
                augment=False,  # 验证集不增强
                preload=self.preload,
            )
        return self._val_dataset
    
    @property
    def test_dataset(self) -> CFDAugmentedDataset:
        if self._test_dataset is None and self.test_cases:
            self._test_dataset = CFDAugmentedDataset(
                root=self.root,
                case_names=self.test_cases,
                graphs_subdir=self.graphs_subdir,
                augment=False,  # 测试集不增强
                preload=self.preload,
            )
        return self._test_dataset
    
    def train_dataloader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """创建训练数据加载器"""
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
    
    def val_dataloader(self, batch_size: int = 32, **kwargs):
        """创建验证数据加载器"""
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
    
    def test_dataloader(self, batch_size: int = 32, **kwargs):
        """创建测试数据加载器"""
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )


def load_transform_params(case_dir: str) -> Optional[Dict]:
    """
    加载病例的坐标系变换参数
    
    参数:
        case_dir: 病例目录路径
    
    返回:
        变换参数字典，或 None（如果不存在）
    """
    params_path = Path(case_dir) / "processed/coord_normalized/transform_params.json"
    if params_path.exists():
        with open(params_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    # 简单测试
    print("数据集模块测试")
    print("=" * 50)
    
    # 测试创建数据集
    try:
        from config import DATA_ROOT
        
        # 假设有数据
        dataset = CFDAugmentedDataset(
            root=str(DATA_ROOT / "AG/fast"),
            augment=True,
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本 x 形状: {sample.x.shape}")
            print(f"样本 y 形状: {sample.y.shape}")
            print(f"样本边数: {sample.edge_index.shape[1]}")
        
        print("\n✅ 测试通过")
        
    except Exception as e:
        print(f"⚠️ 测试跳过（可能没有数据）: {e}")
