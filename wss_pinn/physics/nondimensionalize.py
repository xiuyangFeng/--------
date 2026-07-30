from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CharacteristicScales:
    length_m: float
    velocity_m_s: float
    density_kg_m3: float = 1060.0

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.velocity_m_s <= 0 or self.density_kg_m3 <= 0:
            raise ValueError("characteristic scales must be positive")

    @property
    def time_s(self) -> float:
        return self.length_m / self.velocity_m_s

    @property
    def pressure_pa(self) -> float:
        return self.density_kg_m3 * self.velocity_m_s**2

    def coords(self, coords_m: np.ndarray) -> np.ndarray:
        return np.asarray(coords_m) / self.length_m

    def velocity(self, velocity_m_s: np.ndarray) -> np.ndarray:
        return np.asarray(velocity_m_s) / self.velocity_m_s

    def pressure(self, pressure_pa: np.ndarray) -> np.ndarray:
        pressure = np.asarray(pressure_pa)
        return (pressure - np.mean(pressure, axis=-1, keepdims=True)) / self.pressure_pa

