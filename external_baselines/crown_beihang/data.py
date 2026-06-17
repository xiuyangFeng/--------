from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import default_private_preprocessed_root, project_root


def _indices(feature_names: Sequence[str], selected: Sequence[str]) -> List[int]:
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [n for n in selected if n not in name_to_idx]
    if missing:
        raise KeyError(f"未知 input_features: {missing}; 可用={list(feature_names)}")
    return [name_to_idx[n] for n in selected]


def load_pkl_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list):
        raise ValueError(f"期望 list pkl: {path}")
    return records


def load_train_p_stats(stats_path: Path, point_filter: str) -> Dict[str, float]:
    if not stats_path.exists():
        return {"p_min": 0.0, "p_max": 1.0}
    import json

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    pf = point_filter
    if pf == "interior":
        pf = "volume"
    if pf in stats:
        return stats[pf]
    return stats.get("p_min", stats) if "p_min" in stats else {"p_min": 0.0, "p_max": 1.0}


class CrownDataset(Dataset):
    """
    返回体素化后的**全点云**；10000 点随机采样在 train.py 训练循环内完成（与源码一致）。
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        input_features: Sequence[str],
        p_min: float,
        p_max: float,
        coord_scale: float = 1.0,
    ) -> None:
        self.records = records
        self.input_features = list(input_features)
        self.p_min = p_min
        self.p_max = p_max
        self.coord_scale = coord_scale
        self.feature_names = list(self.records[0]["feature_names"])
        self.target_names = list(self.records[0]["target_names"])
        self.input_indices = _indices(self.feature_names, self.input_features)

    @property
    def input_dim(self) -> int:
        return len(self.input_indices)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        features = rec["features"].astype(np.float32).copy()
        targets = rec["targets"].astype(np.float32).copy()

        p_span = self.p_max - self.p_min + 1e-9
        targets[3] = (targets[3] - self.p_min) / p_span

        if self.coord_scale != 1.0:
            for i, name in enumerate(self.feature_names):
                if name in ("x", "y", "z"):
                    features[i] = features[i] * self.coord_scale

        return {
            "sample_id": rec["sample_id"],
            "case_name": rec["case_name"],
            "features": torch.from_numpy(features),
            "targets": torch.from_numpy(targets),
            "input_indices": self.input_indices,
        }


def collate_crown(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "features": [b["features"] for b in batch],
        "targets": [b["targets"] for b in batch],
        "input_indices": batch[0]["input_indices"],
        "sample_ids": [b["sample_id"] for b in batch],
        "case_names": [b["case_name"] for b in batch],
    }


def build_datasets(config: Mapping[str, Any]) -> Dict[str, CrownDataset]:
    root = project_root()
    data_cfg = config["data"]
    output_root = Path(data_cfg.get("preprocessed_root", str(default_private_preprocessed_root())))
    if not output_root.is_absolute():
        output_root = root / output_root

    point_filter = data_cfg.get("point_filter", "volume")
    if point_filter == "interior":
        point_filter = "volume"
    stats = load_train_p_stats(output_root / "stats" / "train_stats.json", point_filter)
    p_min = float(stats["p_min"])
    p_max = float(stats["p_max"])

    physics_cfg = config.get("physics", {})
    coord_scale = float(physics_cfg.get("coord_scale", 1.0)) if physics_cfg.get("enabled") else 1.0

    datasets: Dict[str, CrownDataset] = {}
    for split_name in ("train", "val", "test"):
        pkl_name = data_cfg.get(f"{split_name}_pkl") or f"crown_{point_filter}_{split_name}.pkl"
        pkl_path = output_root / "pkl" / pkl_name
        records = load_pkl_records(pkl_path)
        datasets[split_name] = CrownDataset(
            records=records,
            input_features=data_cfg["input_features"],
            p_min=p_min,
            p_max=p_max,
            coord_scale=coord_scale,
        )
    return datasets
