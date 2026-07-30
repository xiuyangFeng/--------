from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from wss_pinn.utils import sha256_file


class FrozenAnchorAdapter(nn.Module):
    """Explicit, hash-checked wrapper for any future frozen W0 encoder."""

    def __init__(self, module: nn.Module, checkpoint: str | Path, expected_sha256: str):
        super().__init__()
        source = Path(checkpoint)
        actual = sha256_file(source)
        if actual != expected_sha256:
            raise ValueError(f"anchor checkpoint hash mismatch: {source}")
        payload: Any = torch.load(source, map_location="cpu", weights_only=False)
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

