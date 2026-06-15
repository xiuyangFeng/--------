from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

from pipeline.config import GLOBAL_COND_NAMES, NODE_FEATURE_NAMES, TARGET_NAMES, WSS_TARGET_NAMES
from pipeline.dataset import load_graph_data
from training.core.splits import SplitSpec


@dataclass(frozen=True)
class PointCloudBatch:
    node_input: torch.Tensor
    target: torch.Tensor
    batch: torch.Tensor
    global_cond: torch.Tensor
    case_names: List[str]
    sample_ids: List[str]


TARGET_MODES: Dict[str, List[str]] = {
    "vp": ["u", "v", "w", "p"],
    "p_only": ["p"],
    "p_wss_vec": ["p", "wss_x", "wss_y", "wss_z"],
    "p_wss4": ["p", "wss", "wss_x", "wss_y", "wss_z"],
}


def _indices(names: Sequence[str], selected: Sequence[str]) -> List[int]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in selected if name not in name_to_idx]
    if missing:
        raise KeyError(f"未知字段: {missing}; 可用字段={list(names)}")
    return [name_to_idx[name] for name in selected]


class PointNetCFDDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        case_names: Sequence[str],
        graphs_subdir: str,
        node_features: Sequence[str],
        global_features: Sequence[str],
        target_mode: str,
    ) -> None:
        self.data_root = Path(data_root)
        self.case_names = list(case_names)
        self.graphs_subdir = graphs_subdir
        self.node_features = list(node_features)
        self.global_features = list(global_features)
        self.target_mode = target_mode
        self.node_indices = _indices(NODE_FEATURE_NAMES, self.node_features)
        self.global_indices = _indices(GLOBAL_COND_NAMES, self.global_features)
        if target_mode not in TARGET_MODES:
            raise KeyError(f"未知 target_mode={target_mode}; 可选={sorted(TARGET_MODES)}")

        self.data_files: List[Path] = []
        self._subdir_depth = len(Path(graphs_subdir).parts)
        for case_name in self.case_names:
            case_dir = self.data_root / case_name / graphs_subdir
            if case_dir.exists():
                self.data_files.extend(sorted(case_dir.glob("*.pt")))
        if not self.data_files:
            raise ValueError(
                f"未找到图数据文件: root={self.data_root}, cases={self.case_names}, subdir={graphs_subdir}"
            )

    def __len__(self) -> int:
        return len(self.data_files)

    @property
    def output_names(self) -> List[str]:
        return TARGET_MODES[self.target_mode]

    @property
    def input_dim(self) -> int:
        return len(self.node_features) + len(self.global_features)

    @property
    def global_dim(self) -> int:
        return len(self.global_features)

    def _case_name_from_path(self, data_path: Path) -> str:
        case_path = data_path
        for _ in range(self._subdir_depth + 1):
            case_path = case_path.parent
        return str(case_path.relative_to(self.data_root))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        data_path = self.data_files[idx]
        data = load_graph_data(data_path)

        x = data.x[:, self.node_indices].float()
        global_cond = data.global_cond[:, self.global_indices].float()
        if global_cond.ndim == 1:
            global_cond = global_cond.unsqueeze(0)
        global_row = global_cond[0]
        node_input = torch.cat([x, global_row.expand(x.shape[0], -1)], dim=-1)

        if self.target_mode in {"vp", "p_only"}:
            target_indices = _indices(TARGET_NAMES, TARGET_MODES[self.target_mode])
            target = data.y[:, target_indices].float()
            keep = torch.ones(target.shape[0], dtype=torch.bool)
        else:
            p_idx = TARGET_NAMES.index("p")
            wss_indices = _indices(WSS_TARGET_NAMES, [n for n in TARGET_MODES[self.target_mode] if n != "p"])
            if not hasattr(data, "y_wss"):
                raise AttributeError(f"{data_path} 缺少 y_wss，无法训练 {self.target_mode}")
            is_wall_idx = NODE_FEATURE_NAMES.index("is_wall")
            keep = data.x[:, is_wall_idx] > 0.5
            p = data.y[:, p_idx : p_idx + 1].float()
            wss = data.y_wss[:, wss_indices].float()
            target = torch.cat([p, wss], dim=-1)

        return {
            "node_input": node_input[keep],
            "target": target[keep],
            "global_cond": global_row,
            "case_name": self._case_name_from_path(data_path),
            "sample_id": data_path.stem,
        }


def collate_pointclouds(items: Iterable[Dict[str, object]]) -> PointCloudBatch:
    rows = list(items)
    node_inputs = [row["node_input"] for row in rows]
    targets = [row["target"] for row in rows]
    counts = [x.shape[0] for x in node_inputs]
    batch = torch.cat(
        [torch.full((count,), idx, dtype=torch.long) for idx, count in enumerate(counts)],
        dim=0,
    )
    return PointCloudBatch(
        node_input=torch.cat(node_inputs, dim=0),
        target=torch.cat(targets, dim=0),
        batch=batch,
        global_cond=torch.stack([row["global_cond"] for row in rows], dim=0),
        case_names=[str(row["case_name"]) for row in rows],
        sample_ids=[str(row["sample_id"]) for row in rows],
    )


def build_datasets(config: Dict[str, object]) -> Dict[str, PointNetCFDDataset]:
    data_cfg = config["data"]
    split = SplitSpec.from_json(data_cfg["split_file"])
    common = {
        "data_root": data_cfg["data_root"],
        "graphs_subdir": data_cfg.get("graphs_subdir", "processed/graphs"),
        "node_features": data_cfg["node_features"],
        "global_features": data_cfg["global_features"],
        "target_mode": config["target"]["mode"],
    }
    return {
        "train": PointNetCFDDataset(case_names=split.train_cases, **common),
        "val": PointNetCFDDataset(case_names=split.val_cases, **common),
        "test": PointNetCFDDataset(case_names=split.test_cases, **common),
    }
