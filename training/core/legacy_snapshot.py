from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


LEGACY_SNAPSHOT_FORMAT = "field_legacy_snapshot_v1"


class LegacyFieldSnapshotDataset(Dataset):
    """Load deduplicated historical field inputs for prediction reproduction.

    A snapshot stores static ``x`` / ``edge_index`` once per case and stacks the
    per-frame supervision tensors in the corresponding case payload.  The
    dataset intentionally returns the historical model input as stored; it does
    not re-read or re-mask the mutable ``processed/graphs`` tree.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"legacy snapshot manifest 不存在: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)
        if self.manifest.get("format") != LEGACY_SNAPSHOT_FORMAT:
            raise ValueError(
                f"legacy snapshot format 不支持: {self.manifest.get('format')!r}"
            )

        self.cases: List[Dict[str, Any]] = list(self.manifest.get("cases") or [])
        if not self.cases:
            raise ValueError(f"legacy snapshot 没有 cases: {manifest_path}")

        self.items: List[Tuple[int, int]] = []
        for case_idx, case in enumerate(self.cases):
            num_frames = int(case.get("num_frames", 0))
            if num_frames < 1:
                raise ValueError(f"legacy snapshot case 无 frame: {case.get('case_name')}")
            self.items.extend((case_idx, frame_idx) for frame_idx in range(num_frames))

        expected = int(self.manifest.get("num_frames", len(self.items)))
        if expected != len(self.items):
            raise ValueError(
                f"legacy snapshot frame 数不一致: manifest={expected}, cases={len(self.items)}"
            )

        self._cached_case_idx: int | None = None
        self._cached_case: Dict[str, Any] | None = None

    def validate_config(self, config, subset: str | None = None) -> None:
        """Reject a config that does not match the historical masked inputs."""

        expected_node = list(self.manifest.get("enabled_node_features") or [])
        expected_global = list(self.manifest.get("enabled_global_features") or [])
        if expected_node and list(config.data.enabled_node_features) != expected_node:
            raise ValueError("legacy snapshot 与 config.enabled_node_features 不一致")
        if expected_global and list(config.data.enabled_global_features) != expected_global:
            raise ValueError("legacy snapshot 与 config.enabled_global_features 不一致")
        expected_frame = self.manifest.get("wss_target_frame")
        if expected_frame and config.data.wss_target_frame != expected_frame:
            raise ValueError(
                f"legacy snapshot WSS frame={expected_frame}, config={config.data.wss_target_frame}"
            )
        expected_subset = self.manifest.get("source_subset")
        if subset and expected_subset and subset != expected_subset:
            raise ValueError(
                f"legacy snapshot subset={expected_subset}, 请求 subset={subset}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def _load_case(self, case_idx: int) -> Dict[str, Any]:
        if self._cached_case_idx != case_idx or self._cached_case is None:
            rel = self.cases[case_idx]["payload"]
            path = (self.root / rel).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"legacy snapshot case payload 不存在: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format") != LEGACY_SNAPSHOT_FORMAT:
                raise ValueError(f"legacy case payload format 不支持: {path}")
            self._cached_case_idx = case_idx
            self._cached_case = payload
        return self._cached_case

    def __getitem__(self, idx: int) -> Data:
        case_idx, frame_idx = self.items[idx]
        payload = self._load_case(case_idx)
        data = Data(
            x=payload["x"].clone(),
            edge_index=payload["edge_index"].clone(),
            y=payload["y_true"][frame_idx].clone(),
            global_cond=payload["global_cond"][frame_idx].clone(),
        )
        if "y_wss_true" in payload:
            data.y_wss = payload["y_wss_true"][frame_idx].clone()
        if "y_wss_global_true" in payload:
            data.y_wss_global = payload["y_wss_global_true"][frame_idx].clone()

        data.sample_id = payload["sample_ids"][frame_idx]
        data.case_name = payload["case_name"]
        # 原 graph_path 现在指向已变更的图资产，不得让后处理误读。
        # 使用明确不存在的快照虚拟路径，促使 full payload 回退到
        # 快照中保存的历史 x / wall_mask。
        safe_case = payload["case_name"].replace("/", "__")
        data.graph_path = str(
            self.root / "_historical_graphs_unavailable" / safe_case / f"{data.sample_id}.pt"
        )
        return data
