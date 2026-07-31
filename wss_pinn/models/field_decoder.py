"""坐标场解码器：Fourier 特征 + trunk → log_WSS / 速度 / 压力。

学习要点
--------
这是 F0-U / F0-UP / F1 **共用**的光滑坐标网络（architecture 在阶段间不变，
只改 loss 权重），保证「精度变化来自物理项而非换架构」。

前向数据流::

    coords [N,3]
        → FourierCoordinates  →  [N, 3+6F]   # 原坐标 + sin/cos 多频
    geometry_descriptor
        → GeometryEncoder     →  [1, L] 再 expand 到 [N, L]
    concat → trunk(MLP) → head
        → log_wss_direct [N]
        → velocity       [N,3]  （无量纲）
        → pressure       [N]    （无量纲、相对）

``architecture_version``：
- ``shared_v1``：一个 trunk，5 维头（WSS+场共享特征）；
- ``split_v2``：WSS 与场分两条 trunk（消融用）。
"""

from __future__ import annotations

import torch
from torch import nn

from .geometry_encoder import GeometryEncoder


class FourierCoordinates(nn.Module):
    """NeRF 风格位置编码：``[x, sin(2^k π x), cos(2^k π x)]``。

    高频带帮助 MLP 拟合空间上变化较快的剪切层；``bands = 2^{0..F-1}``。
    """

    def __init__(self, frequencies: int):
        super().__init__()
        bands = 2.0 ** torch.arange(int(frequencies), dtype=torch.float32)
        # register_buffer：随模型设备移动，但不作为可训练参数
        self.register_buffer("bands", bands, persistent=True)

    @property
    def output_dim(self) -> int:
        # 3 个原坐标 + 每个频率对 xyz 各 sin/cos → 6 * F
        return 3 + 6 * len(self.bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords[..., None] * bands → [..., 3, F]；再 flatten 最后两维
        phase = coords[..., None] * self.bands * torch.pi
        return torch.cat(
            [coords, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)],
            dim=-1,
        )


class WSSPINNModel(nn.Module):
    """共享光滑坐标场；F0-U / F0-UP / F1 结构不变，只改监督项。"""

    def __init__(self, model_config: dict):
        super().__init__()
        latent = int(model_config["geometry_latent"])
        self.geometry = GeometryEncoder(
            input_dim=15,
            hidden_dim=int(model_config["geometry_hidden"]),
            latent_dim=latent,
        )
        self.coordinates = FourierCoordinates(int(model_config["fourier_frequencies"]))
        hidden = int(model_config["field_hidden"])
        layers = int(model_config["field_layers"])
        input_dim = self.coordinates.output_dim + latent

        def make_trunk() -> nn.Sequential:
            """SiLU MLP trunk：input_dim → hidden × layers。"""
            modules: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.SiLU()]
            for _ in range(max(1, layers) - 1):
                modules.extend([nn.Linear(hidden, hidden), nn.SiLU()])
            return nn.Sequential(*modules)

        self.architecture_version = str(
            model_config.get("architecture_version", "shared_v1")
        )
        if self.architecture_version == "shared_v1":
            self.trunk = make_trunk()
            self.head = nn.Linear(hidden, 5)  # [log_wss, u, v, w, p]
        elif self.architecture_version == "split_v2":
            self.wss_trunk = make_trunk()
            self.field_trunk = make_trunk()
            self.wss_head = nn.Linear(hidden, 1)
            self.field_head = nn.Linear(hidden, 4)  # [u, v, w, p]
        else:
            raise ValueError(
                f"unsupported model.architecture_version: {self.architecture_version}"
            )

    def forward(
        self, coords: torch.Tensor, geometry_descriptor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """对一批坐标查询场。

        Parameters
        ----------
        coords
            ``[N, 3]`` 无量纲注册坐标（与 sidecar ``*_coords`` 一致）。
        geometry_descriptor
            ``[15]`` 或 ``[1, 15]`` / ``[N, 15]``；通常一病例一份。
        """
        latent = self.geometry(geometry_descriptor)
        # 单病例描述子广播到 N 个查询点
        if latent.shape[0] == 1 and coords.shape[0] != 1:
            latent = latent.expand(coords.shape[0], -1)
        features = torch.cat([self.coordinates(coords), latent], dim=-1)

        if self.architecture_version == "shared_v1":
            output = self.head(self.trunk(features))
            log_wss = output[:, 0]
            field = output[:, 1:5]
        else:
            log_wss = self.wss_head(self.wss_trunk(features))[:, 0]
            field = self.field_head(self.field_trunk(features))

        return {
            "log_wss_direct": log_wss,  # 训练对 log1p(WSS)；评估用 expm1 还原
            "velocity": field[:, 0:3],
            "pressure": field[:, 3],
        }
