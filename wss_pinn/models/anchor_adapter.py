"""冻结锚点适配器：未来若复用 W0 编码器，必须经此显式哈希校验包装。

学习要点
--------
AGENTS.md 要求：任何复用旧模型都必须记录父 checkpoint 与 SHA256，
不得依赖含义不明的 ``latest``。本类在加载时比对期望哈希，加载后冻结
``requires_grad``，前向全程 ``no_grad``，避免「悄悄微调上游」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from wss_pinn.utils import sha256_file


class FrozenAnchorAdapter(nn.Module):
    """显式、带哈希校验的冻结模块包装器（当前实验主路径尚未强制使用）。"""

    def __init__(self, module: nn.Module, checkpoint: str | Path, expected_sha256: str):
        super().__init__()
        source = Path(checkpoint)
        actual = sha256_file(source)
        if actual != expected_sha256:
            raise ValueError(f"anchor checkpoint hash mismatch: {source}")
        payload: Any = torch.load(source, map_location="cpu", weights_only=False)
        # 兼容直接存 state_dict 或包在 {"model": ...} 里的两种格式
        state = payload.get("model", payload)
        module.load_state_dict(state, strict=True)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        self.module = module
        self.checkpoint = str(source.resolve())
        self.checkpoint_sha256 = actual

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)
