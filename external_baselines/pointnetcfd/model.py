from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


def _make_mlp(dims: Iterable[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    dims = list(dims)
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def segment_max(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Max-pool point features per graph without requiring torch_scatter."""
    pooled = values.new_full((num_graphs, values.shape[1]), -torch.inf)
    pooled.scatter_reduce_(
        0,
        batch[:, None].expand(-1, values.shape[1]),
        values,
        reduce="amax",
        include_self=True,
    )
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


class PointNetCFD(nn.Module):
    """PointNetCFD-style point-cloud regressor.

    The model keeps the paper's key idea: shared point encoding, global max
    pooling, and point-wise decoding. Mesh edges are intentionally ignored.
    """

    def __init__(
        self,
        input_dim: int,
        global_dim: int,
        output_dim: int,
        point_hidden: int = 128,
        global_hidden: int = 256,
        decoder_hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.point_encoder = _make_mlp(
            [input_dim, point_hidden, point_hidden, global_hidden],
            dropout=dropout,
        )
        self.decoder = _make_mlp(
            [global_hidden + global_hidden + global_dim, decoder_hidden, decoder_hidden],
            dropout=dropout,
        )
        self.output = nn.Linear(decoder_hidden, output_dim)

    def forward(
        self,
        node_input: torch.Tensor,
        global_cond: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        point_feat = self.point_encoder(node_input)
        graph_feat = segment_max(point_feat, batch=batch, num_graphs=global_cond.shape[0])
        decoded = torch.cat([point_feat, graph_feat[batch], global_cond[batch]], dim=-1)
        hidden = self.decoder(decoded)
        return self.output(hidden)

