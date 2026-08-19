"""Detached fixed/EMA controller for the V4 PDE group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ControllerRecord:
    lambda_pde: float
    loss_ratio_raw: float
    ema_data: float
    ema_pde: float
    lambda_target: float
    lambda_applied: float
    lambda_clamped_low: bool
    lambda_clamped_high: bool
    lambda_update_flag: bool


class PDEWeightController:
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.mode = str(config["mode"])
        self.ema_data: float | None = None
        self.ema_pde: float | None = None
        self.applied = (
            1.0
            if self.mode == "fixed"
            else (
                float(config["alpha_start"])
                if self.mode == "ema_loss_ratio"
                else 0.0
            )
        )
        self.target = self.applied

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ema_data": self.ema_data,
            "ema_pde": self.ema_pde,
            "applied": self.applied,
            "target": self.target,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("mode") != self.mode:
            raise ValueError("PDE controller mode mismatch")
        self.ema_data = state.get("ema_data")
        self.ema_pde = state.get("ema_pde")
        self.applied = float(state.get("applied", self.applied))
        self.target = float(state.get("target", self.target))

    def update(
        self,
        data_loss: torch.Tensor,
        pde_loss: torch.Tensor,
        *,
        global_step: int,
        epoch: int,
    ) -> ControllerRecord:
        # The only values leaving the graph are Python floats.  This is the
        # hard guarantee preventing lambda=L_data/L_pde from cancelling PDE
        # gradients algebraically.
        data_value = float(data_loss.detach().cpu())
        pde_value = float(pde_loss.detach().cpu())
        beta = float(self.config["ema_beta"])
        self.ema_data = data_value if self.ema_data is None else beta * self.ema_data + (1.0 - beta) * data_value
        self.ema_pde = pde_value if self.ema_pde is None else beta * self.ema_pde + (1.0 - beta) * pde_value
        ratio = data_value / max(pde_value, float(self.config["epsilon"]))
        updated = False
        clamped_low = False
        clamped_high = False
        if self.mode == "disabled":
            self.target = self.applied = 0.0
        elif self.mode == "fixed":
            self.target = self.applied = 1.0
        elif global_step % int(self.config["update_interval_steps"]) == 0:
            ramp = min(max(epoch / max(int(self.config["physics_ramp_epochs"]), 1), 0.0), 1.0)
            alpha = float(self.config["alpha_start"]) + ramp * (
                float(self.config["alpha_end"]) - float(self.config["alpha_start"])
            )
            raw_target = alpha * float(self.ema_data) / max(float(self.ema_pde), float(self.config["epsilon"]))
            lower = float(self.config["lambda_min"])
            upper = float(self.config["lambda_max"])
            clamped_low = raw_target < lower
            clamped_high = raw_target > upper
            self.target = min(max(raw_target, lower), upper)
            weight_beta = float(self.config["weight_smoothing_beta"])
            self.applied = weight_beta * self.applied + (1.0 - weight_beta) * self.target
            updated = True
        return ControllerRecord(
            lambda_pde=float(self.applied),
            loss_ratio_raw=float(ratio),
            ema_data=float(self.ema_data),
            ema_pde=float(self.ema_pde),
            lambda_target=float(self.target),
            lambda_applied=float(self.applied),
            lambda_clamped_low=bool(clamped_low),
            lambda_clamped_high=bool(clamped_high),
            lambda_update_flag=bool(updated),
        )
