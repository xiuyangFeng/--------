"""训练/评估用的 physics 数据集：读 sidecar，按 epoch 再采样 mini-batch。

学习要点
--------
数据层次：

1. **汇总 manifest**（``sampling_manifest.json``）：病例列表 + 各自 manifest 路径；
2. **PhysicsCase**：加载一个病例的 static/fields npz；
3. **PhysicsCase.sample**：每个 epoch 从三槽里再抽 ``batch_*`` 点，
   并把速度/压力无量纲化后交给 ``stage_losses``；
4. **PhysicsDataset.epoch_samples**：可选 ``batch_cases`` 子集病例，返回本 epoch 列表。

``as_tensor`` 是薄封装，保证 numpy → torch 时 device/dtype 一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .sidecar import load_sidecar_manifest


class PhysicsCase:
    """单个病例的不可变 sidecar 视图。"""

    def __init__(
        self,
        manifest_path: str | Path,
        verify: bool = True,
        aggregate_row: dict[str, Any] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest = load_sidecar_manifest(self.manifest_path, verify=verify)
        static_path = Path(self.manifest["files"]["static"]["path"])
        field_path = Path(self.manifest["files"]["peak_fields"]["path"])
        with np.load(static_path, allow_pickle=False) as source:
            self.static = {key: np.asarray(source[key]) for key in source.files}
        with np.load(field_path, allow_pickle=False) as source:
            self.fields = {key: np.asarray(source[key]) for key in source.files}
        row = aggregate_row or {}
        # canonical_id 优先；兼容旧字段
        self.case_id = str(
            self.manifest.get(
                "canonical_id",
                row.get("canonical_id", self.manifest["case_id"]),
            )
        )
        self.cohort = str(self.manifest["cohort"])
        self.role = str(self.manifest.get("role", row.get("role", "unspecified")))

    @property
    def velocity_scale(self) -> float:
        """特征速度 U（m/s），评估时把无量纲速度乘回去。"""
        return float(self.manifest["characteristic_scales"]["velocity_m_s"])

    @property
    def pressure_scale(self) -> float:
        """压力尺度 ρU²。"""
        rho = float(self.manifest["characteristic_scales"]["density_kg_m3"])
        return rho * self.velocity_scale**2

    def sample(
        self,
        *,
        wall: int,
        near_wall: int,
        core: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        """从本病例三槽中无放回抽一个训练 batch，并做场无量纲化。"""
        rng = np.random.default_rng(int(seed))

        def choose(length: int, count: int) -> np.ndarray:
            if length <= 0:
                raise ValueError("empty sidecar slot")
            return rng.choice(length, size=min(int(count), length), replace=False)

        wi = choose(len(self.static["wall_coords"]), wall)
        ni = choose(len(self.static["near_wall_coords"]), near_wall)
        ci = choose(len(self.static["core_coords"]), core)

        # 压力：拼接近壁+核心后去均值，再除以 ρU²
        pressure = np.concatenate(
            [
                self.fields["near_wall_pressure"][ni],
                self.fields["core_pressure"][ci],
            ]
        ).astype(np.float32)
        pressure = (pressure - pressure.mean()) / max(self.pressure_scale, 1e-8)

        return {
            "descriptor": self.static["geometry_descriptor"].astype(np.float32),
            "wall_coords": self.static["wall_coords"][wi].astype(np.float32),
            "wall_wss": self.fields["wall_wss"][wi].astype(np.float32),  # 物理 Pa
            "interior_coords": np.concatenate(
                [
                    self.static["near_wall_coords"][ni],
                    self.static["core_coords"][ci],
                ]
            ).astype(np.float32),
            "velocity": (
                np.concatenate(
                    [
                        self.fields["near_wall_velocity"][ni],
                        self.fields["core_velocity"][ci],
                    ]
                )
                / self.velocity_scale
            ).astype(np.float32),
            "pressure": pressure,
        }


class PhysicsDataset:
    """汇总 manifest 上的多病例集合，可按 role 过滤。"""

    def __init__(
        self,
        aggregate_manifest: str | Path,
        verify: bool = True,
        roles: list[str] | tuple[str, ...] | None = None,
    ):
        payload = json.loads(Path(aggregate_manifest).read_text(encoding="utf-8"))
        self.split_label = str(payload["split_label"])
        allowed = set(roles) if roles else None
        self.cases = []
        for row in payload["cases"]:
            case = PhysicsCase(
                row["manifest"], verify=verify, aggregate_row=row
            )
            if allowed is None or case.role in allowed:
                self.cases.append(case)
        if not self.cases:
            raise ValueError(f"physics dataset has no cases for roles={roles}")

    def epoch_samples(
        self, train_config: dict[str, Any], epoch: int
    ) -> list[dict[str, np.ndarray]]:
        """生成本 epoch 的病例 batch 列表。

        - ``batch_cases > 0`` 且小于总病例数时，先随机抽病例子集；
        - 每个病例用 ``seed + epoch*10007 + index*97`` 保证可复现又不跨 epoch 重复。
        """
        base_seed = int(train_config["seed"]) + int(epoch) * 10007
        cases = self.cases
        batch_cases = int(train_config.get("batch_cases") or 0)
        if 0 < batch_cases < len(cases):
            rng = np.random.default_rng(base_seed + 53)
            selected = np.sort(
                rng.choice(len(cases), size=batch_cases, replace=False)
            )
            cases = [cases[int(index)] for index in selected]
        samples = []
        for index, case in enumerate(cases):
            sample = case.sample(
                wall=int(train_config["batch_wall"]),
                near_wall=int(train_config["batch_near_wall"]),
                core=int(train_config["batch_core"]),
                seed=base_seed + index * 97,
            )
            sample["case_id"] = getattr(case, "case_id", getattr(case, "name", "unknown"))
            samples.append(sample)
        return samples


def as_tensor(
    value: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """numpy 数组 → 指定 device/dtype 的 Tensor（共享存储当可能时）。"""
    return torch.as_tensor(value, device=device, dtype=dtype)
