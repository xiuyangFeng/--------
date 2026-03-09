from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SplitSpec:
    split_version: str
    # 始终按患者级划分，避免同一患者不同时间帧泄漏到不同子集。
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]
    source: Optional[str] = None
    notes: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitSpec":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            split_version=raw["split_version"],
            train_cases=raw["train_cases"],
            val_cases=raw["val_cases"],
            test_cases=raw["test_cases"],
            source=raw.get("source"),
            notes=raw.get("notes", ""),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "split_version": self.split_version,
            "train_cases": self.train_cases,
            "val_cases": self.val_cases,
            "test_cases": self.test_cases,
            "source": self.source,
            "notes": self.notes,
        }
