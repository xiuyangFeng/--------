from __future__ import annotations

import json
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import default_private_preprocessed_root, project_root


def _safe_case_id(case_name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", case_name.replace("/", "__"))


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
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    pf = point_filter
    if pf == "interior":
        pf = "volume"
    if pf in stats:
        return stats[pf]
    return stats.get("p_min", stats) if "p_min" in stats else {"p_min": 0.0, "p_max": 1.0}


def _normalize_point_filter(point_filter: str) -> str:
    if point_filter == "interior":
        return "volume"
    return point_filter


def _preprocessed_paths(config: Mapping[str, Any]) -> tuple[Path, Path, str]:
    root = project_root()
    data_cfg = config["data"]
    output_root = Path(data_cfg.get("preprocessed_root", str(default_private_preprocessed_root())))
    if not output_root.is_absolute():
        output_root = root / output_root
    point_filter = _normalize_point_filter(data_cfg.get("point_filter", "volume"))
    return output_root, output_root / "pkl", point_filter


def _lazy_load_enabled(config: Mapping[str, Any], output_root: Path) -> bool:
    data_cfg = config["data"]
    if "lazy_load" in data_cfg:
        return bool(data_cfg["lazy_load"])
    jsonl_path = output_root / "audit" / "preprocess_cases.jsonl"
    partial_dir = output_root / "pkl" / "partial"
    return jsonl_path.is_file() and partial_dir.is_dir() and any(partial_dir.glob("*.pkl"))


class _PartialRecordCache:
    """按病例 shard 缓存 partial pkl，避免 100GB merged pkl 整包进内存。"""

    def __init__(self, max_cases: int = 2) -> None:
        self.max_cases = max(1, max_cases)
        self._cache: OrderedDict[Path, List[Dict[str, Any]]] = OrderedDict()

    def get_records(self, path: Path) -> List[Dict[str, Any]]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        records = load_pkl_records(path)
        self._cache[path] = records
        while len(self._cache) > self.max_cases:
            self._cache.popitem(last=False)
        return records


def build_sample_index_from_jsonl(
    output_root: Path,
    point_filter: str,
    split_names: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    jsonl_path = output_root / "audit" / "preprocess_cases.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"lazy 索引需要 preprocess_cases.jsonl: {jsonl_path}")

    partial_dir = output_root / "pkl" / "partial"
    split_set = set(split_names) if split_names else None
    per_case_idx: Dict[str, int] = {}
    index: List[Dict[str, Any]] = []
    feature_names: List[str] | None = None
    target_names: List[str] | None = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("point_filter") != point_filter:
                continue
            case_name = row["case_name"]
            frame_key = row["frame_key"]
            sample_id = f"{case_name}/{frame_key}"
            partial_path = partial_dir / f"{_safe_case_id(case_name)}_{point_filter}.pkl"
            record_idx = per_case_idx.get(case_name, 0)
            per_case_idx[case_name] = record_idx + 1
            if split_set is not None and row.get("split") not in split_set:
                continue
            index.append(
                {
                    "sample_id": sample_id,
                    "case_name": case_name,
                    "split": row.get("split"),
                    "partial_path": partial_path,
                    "record_idx": record_idx,
                }
            )
            if feature_names is None:
                feature_names = ["x", "y", "z"]
                target_names = ["u", "v", "w", "p"]

    if not index:
        raise ValueError(
            f"lazy 索引为空: point_filter={point_filter} splits={split_names} root={output_root}"
        )
    return index


def _record_to_item(
    rec: Mapping[str, Any],
    input_indices: List[int],
    p_min: float,
    p_max: float,
    coord_scale: float,
) -> Dict[str, Any]:
    features = rec["features"].astype(np.float32, copy=False)
    targets = rec["targets"].astype(np.float32, copy=True)

    p_span = p_max - p_min + 1e-9
    targets[3] = (targets[3] - p_min) / p_span

    if coord_scale != 1.0:
        feature_names = list(rec["feature_names"])
        features = features.copy()
        for i, name in enumerate(feature_names):
            if name in ("x", "y", "z"):
                features[i] = features[i] * coord_scale

    return {
        "sample_id": rec["sample_id"],
        "case_name": rec["case_name"],
        "features": torch.from_numpy(np.ascontiguousarray(features)),
        "targets": torch.from_numpy(np.ascontiguousarray(targets)),
        "input_indices": input_indices,
    }


class CrownLazyDataset(Dataset):
    """
    从 partial pkl 按需读取单帧；索引来自 preprocess_cases.jsonl，启动时不读 merged pkl。
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        input_features: Sequence[str],
        p_min: float,
        p_max: float,
        coord_scale: float = 1.0,
        partial_cache_cases: int = 2,
        feature_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
    ) -> None:
        self.index = index
        self.input_features = list(input_features)
        self.p_min = p_min
        self.p_max = p_max
        self.coord_scale = coord_scale
        self._cache = _PartialRecordCache(max_cases=partial_cache_cases)
        self.feature_names = list(feature_names or ["x", "y", "z"])
        self.target_names = list(target_names or ["u", "v", "w", "p"])
        self.input_indices = _indices(self.feature_names, self.input_features)

    @property
    def input_dim(self) -> int:
        return len(self.input_indices)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.index[idx]
        records = self._cache.get_records(entry["partial_path"])
        record_idx = entry["record_idx"]
        if record_idx >= len(records):
            raise IndexError(
                f"partial 索引越界: {entry['partial_path']} idx={record_idx} len={len(records)}"
            )
        rec = records[record_idx]
        if rec["sample_id"] != entry["sample_id"]:
            raise ValueError(
                f"partial 索引与 sample_id 不一致: expect={entry['sample_id']} got={rec['sample_id']}"
            )
        return _record_to_item(rec, self.input_indices, self.p_min, self.p_max, self.coord_scale)


class CrownDataset(Dataset):
    """
    返回体素化后的**全点云**；10000 点随机采样在 train.py 训练循环内完成（与源码一致）。
    merged pkl 全量驻内存，仅适合小数据或 audit。
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
        return _record_to_item(rec, self.input_indices, self.p_min, self.p_max, self.coord_scale)


def collate_crown(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "features": [b["features"] for b in batch],
        "targets": [b["targets"] for b in batch],
        "input_indices": batch[0]["input_indices"],
        "sample_ids": [b["sample_id"] for b in batch],
        "case_names": [b["case_name"] for b in batch],
    }


def build_datasets(
    config: Mapping[str, Any],
    splits: Sequence[str] | None = None,
    lazy_load: bool | None = None,
) -> Dict[str, CrownDataset | CrownLazyDataset]:
    output_root, pkl_root, point_filter = _preprocessed_paths(config)
    data_cfg = config["data"]
    stats = load_train_p_stats(output_root / "stats" / "train_stats.json", point_filter)
    p_min = float(stats["p_min"])
    p_max = float(stats["p_max"])

    physics_cfg = config.get("physics", {})
    coord_scale = float(physics_cfg.get("coord_scale", 1.0)) if physics_cfg.get("enabled") else 1.0
    use_lazy = _lazy_load_enabled(config, output_root) if lazy_load is None else bool(lazy_load)
    split_names = tuple(splits) if splits is not None else ("train", "val", "test")
    partial_cache_cases = int(data_cfg.get("partial_cache_cases", 2))

    datasets: Dict[str, CrownDataset | CrownLazyDataset] = {}
    if use_lazy:
        for split_name in split_names:
            index = build_sample_index_from_jsonl(output_root, point_filter, split_names=(split_name,))
            datasets[split_name] = CrownLazyDataset(
                index=index,
                input_features=data_cfg["input_features"],
                p_min=p_min,
                p_max=p_max,
                coord_scale=coord_scale,
                partial_cache_cases=partial_cache_cases,
            )
        return datasets

    for split_name in split_names:
        pkl_name = data_cfg.get(f"{split_name}_pkl") or f"crown_{point_filter}_{split_name}.pkl"
        pkl_path = pkl_root / pkl_name
        records = load_pkl_records(pkl_path)
        datasets[split_name] = CrownDataset(
            records=records,
            input_features=data_cfg["input_features"],
            p_min=p_min,
            p_max=p_max,
            coord_scale=coord_scale,
        )
    return datasets
