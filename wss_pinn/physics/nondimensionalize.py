"""特征尺度与无量纲化约定。

学习要点
--------
PINN 在无量纲坐标/场上训练更稳。本仓库约定：

- 长度尺度 ``L``：病例 ``coord_scale``（mm）转成米；
- 速度尺度 ``U``：峰值时刻体点速度模长的 P95；
- 压力尺度 ``ρ U²``（动压）；压力还会先减去均值（规范不变）。

``CharacteristicScales`` 提供把物理量除以尺度的便捷方法。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CharacteristicScales:
    """单病例的 (L, U, ρ)；冻结 dataclass 防止训练中被意外改写。"""

    length_m: float
    velocity_m_s: float
    density_kg_m3: float = 1060.0

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.velocity_m_s <= 0 or self.density_kg_m3 <= 0:
            raise ValueError("characteristic scales must be positive")

    @property
    def time_s(self) -> float:
        """时间尺度 L/U，供未来非定常项用。"""
        return self.length_m / self.velocity_m_s

    @property
    def pressure_pa(self) -> float:
        """压力尺度 ρU²。"""
        return self.density_kg_m3 * self.velocity_m_s**2

    def coords(self, coords_m: np.ndarray) -> np.ndarray:
        """物理米坐标 → 无量纲。"""
        return np.asarray(coords_m) / self.length_m

    def velocity(self, velocity_m_s: np.ndarray) -> np.ndarray:
        """物理速度 → 无量纲。"""
        return np.asarray(velocity_m_s) / self.velocity_m_s

    def pressure(self, pressure_pa: np.ndarray) -> np.ndarray:
        """去均值后除以 ρU²。"""
        pressure = np.asarray(pressure_pa)
        return (pressure - np.mean(pressure, axis=-1, keepdims=True)) / self.pressure_pa
