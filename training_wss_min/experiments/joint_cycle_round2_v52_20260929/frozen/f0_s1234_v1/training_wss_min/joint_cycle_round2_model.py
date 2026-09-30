"""Controlled second-round residuals on the frozen first-round architecture.

Only sampled support geometry and known BC/phase inputs are consumed. Temporal
residuals see the entire supplied phase sequence, regardless of decoder chunk
size. They require an ordered contiguous sequence; the experiment uses all 80
phases. Spatial residuals use the same support pool as the original encoder.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter, softmax
from torch.utils.checkpoint import checkpoint

from .baseline_models import _knn_group
from .joint_cycle_model import (JointCycleModel, TASK_CHANNELS, TaskCycleDecoder,
                                _cpu_seed, _interpolate_fp32)


ROUND2_MODULES = ("none", "temporal_ffn", "temporal_tcn", "query_mean",
                  "query_attention", "wall_pointwise", "wall_patch")


def _count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _closest_width(target, count):
    return min(range(1, 1025), key=lambda h: abs(count(h) - target))


class TemporalResidual(nn.Module):
    """Four residual hidden blocks, with a zero initial output projection."""

    def __init__(self, hidden, bottleneck, *, temporal):
        super().__init__()
        self.temporal = bool(temporal)
        self.bottleneck = int(bottleneck)
        self.dilations = (1, 2, 4, 8)
        self.input = nn.Linear(hidden, bottleneck)
        if temporal:
            self.blocks = nn.ModuleList(nn.Conv1d(bottleneck, bottleneck, 3,
                                                dilation=d) for d in self.dilations)
        else:
            self.blocks = nn.ModuleList(nn.Linear(bottleneck, bottleneck) for _ in self.dilations)
        self.out = nn.Linear(bottleneck, hidden)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def parameter_count(hidden, bottleneck, temporal):
        b = int(bottleneck)
        return 2 * hidden * b + b + hidden + 4 * (b * b * (3 if temporal else 1) + b)

    def forward(self, latent):
        z = F.gelu(self.input(latent))
        if self.temporal:
            if latent.shape[1] <= max(self.dilations):
                raise ValueError("reflect TCN needs at least 9 contiguous phases")
            z = z.transpose(1, 2)
            for conv, dilation in zip(self.blocks, self.dilations):
                z = z + F.gelu(conv(F.pad(z, (dilation, dilation), mode="reflect")))
            z = z.transpose(1, 2)
        else:
            for linear in self.blocks:
                z = z + F.gelu(linear(z))
        return latent + self.out(z)


class TemporalTaskDecoder(TaskCycleDecoder):
    """Reuse every original task parameter without advancing its seed stream."""

    def __init__(self, original, residual):
        nn.Module.__init__(self)
        self.query_dim = original.query_dim
        for name in ("adapter", "phase", "film", "hidden", "output"):
            setattr(self, name, getattr(original, name))
        self.round2_residual = residual

    def forward(self, features, query_pos, query_x, condition=None, *, modulation=None):
        frames = features.shape[1]
        raw = torch.cat((features, query_x[:, None].expand(-1, frames, -1),
                         query_pos[:, None].expand(-1, frames, -1)), dim=-1)
        local = self.adapter(raw)
        gamma, beta = self.condition_parameters(condition) if modulation is None else modulation
        latent = self.hidden(local * (1 + gamma) + beta)
        return self.output(self.round2_residual(latent))


class QueryNeighbourResidual(nn.Module):
    """Mean/attention edge aggregation with identical geometry inputs and kNN."""

    def __init__(self, width, support_dim, query_dim, hidden, heads, k, *, attention):
        super().__init__()
        self.attention, self.heads, self.k = bool(attention), int(heads), int(k)
        self.hidden = int(hidden)
        if attention and hidden % heads:
            raise ValueError("round2_hidden must be divisible by round2_heads")
        self.edge = nn.Sequential(nn.Linear(width + support_dim + query_dim + 4, hidden),
                                  nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.query = nn.Linear(width + query_dim + 3, hidden)
        if attention:
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def parameter_count(width, support_dim, query_dim, hidden, attention):
        edge_dim, query_dim = width + support_dim + query_dim + 4, width + query_dim + 3
        h = int(hidden)
        return h * h + (edge_dim + query_dim + width + 3) * h + width + \
            (2 * h * h + 2 * h if attention else 0)

    def forward(self, base, query_pos, query_x, query_batch, geo, *, return_attention=False):
        with torch.no_grad(), torch.autocast(query_pos.device.type, enabled=False):
            # The upstream pool is identical across the pair. FP32 geometry is
            # required even when feature matmuls use AMP.
            row, col = _knn_group(geo["pos"].float(), geo["batch"],
                                  query_pos.float(), query_batch, self.k)
            relative = geo["pos"][col].float() - query_pos[row].float()
            edge_geometry = torch.cat((relative, torch.linalg.vector_norm(
                relative, dim=-1, keepdim=True)), dim=-1)
        q = self.query(torch.cat((base, query_x, query_pos), dim=-1))
        edge = self.edge(torch.cat((geo["features"][col], geo["raw_x"][col],
                                   query_x[row], edge_geometry), dim=-1))
        if self.attention:
            keys = self.key(edge).view(-1, self.heads, self.hidden // self.heads)
            values = self.value(edge).view(-1, self.heads, self.hidden // self.heads)
            queries = q[row].view_as(keys)
            with torch.autocast(query_pos.device.type, enabled=False):
                logits = (queries.float() * keys.float()).sum(-1) * \
                    ((self.hidden // self.heads) ** -0.5)
                weights = softmax(logits, row, num_nodes=len(base)).to(values.dtype)
            aggregate = scatter(values * weights[..., None], row, dim=0,
                                dim_size=len(base), reduce="sum").flatten(1)
        else:
            weights = None
            aggregate = scatter(edge, row, dim=0, dim_size=len(base), reduce="mean")
        result = self.out(F.gelu(aggregate + q))
        return (result, weights, row, col) if return_attention else result


class PointwiseResidual(nn.Module):
    def __init__(self, width, query_dim, hidden):
        super().__init__()
        self.hidden = int(hidden)
        self.mlp = nn.Sequential(nn.Linear(width + query_dim + 3, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def parameter_count(width, query_dim, hidden):
        h = int(hidden)
        return h * h + (2 * width + query_dim + 5) * h + width

    def forward(self, base, query_pos, query_x, query_batch, geo):
        return self.out(self.mlp(torch.cat((base, query_x, query_pos), dim=-1)))


class Round2JointCycleModel(JointCycleModel):
    def __init__(self, variant, support_in_dim, query_dims, *, seed=1234,
                 active_tasks=None, model_options=None):
        opts = dict(model_options or {})
        module = opts.get("round2_module", "none")
        if module not in ROUND2_MODULES or module == "none":
            raise ValueError("Round2JointCycleModel requires a supported residual")
        super().__init__(variant, support_in_dim, query_dims, seed=seed,
                         active_tasks=active_tasks, model_options=opts)
        if variant != "I" or len(self.active_tasks) != 1:
            raise ValueError("round2 residuals require a single-task independent model")
        task = self.active_tasks[0]
        if task not in ("velocity", "wss") or (module.startswith("query_") and task != "velocity") \
                or (module.startswith("wall_") and task != "wss"):
            raise ValueError("round2 residual/task mismatch")
        self.round2_module = module
        self.round2_spatial = nn.ModuleDict()
        hidden = int(opts.get("round2_hidden", 64))
        k, heads = int(opts.get("round2_neighbors", 16)), int(opts.get("round2_heads", 4))
        bottleneck = int(opts.get("temporal_bottleneck", 32))
        if min(hidden, k, heads, bottleneck) < 1:
            raise ValueError("round2 dimensions and neighbour count must be positive")
        self.round2_contract = {"module": module, "task": task,
                                "support_pool": "encoder_sample_only", "zero_initial_residual": True}
        with _cpu_seed(int(seed) + 2000 + list(TASK_CHANNELS).index(task)):
            if module.startswith("temporal_"):
                dim = self.decoders[task].output.in_features
                reference_count = TemporalResidual.parameter_count(dim, bottleneck, True)
                actual_bottleneck = bottleneck if module == "temporal_tcn" else _closest_width(
                    reference_count, lambda h: TemporalResidual.parameter_count(dim, h, False))
                residual = TemporalResidual(dim, actual_bottleneck, temporal=module == "temporal_tcn")
                self.decoders[task] = TemporalTaskDecoder(self.decoders[task], residual)
                self.round2_contract.update(bottleneck=actual_bottleneck, layers=4,
                    kernel=3 if module == "temporal_tcn" else 1,
                    dilations=[1, 2, 4, 8] if module == "temporal_tcn" else [1] * 4,
                    receptive_field=31 if module == "temporal_tcn" else 1,
                    boundary="reflect" if module == "temporal_tcn" else "pointwise",
                    phase_chunk_behavior="entire_supplied_sequence")
            elif module in ("query_mean", "query_attention", "wall_patch"):
                reference_attention = module.startswith("query_")
                reference_count = QueryNeighbourResidual.parameter_count(
                    self.width, support_in_dim, query_dims[task], hidden, reference_attention)
                if module == "query_mean":
                    hidden = _closest_width(reference_count, lambda h: QueryNeighbourResidual.parameter_count(
                        self.width, support_in_dim, query_dims[task], h, False))
                residual = QueryNeighbourResidual(self.width, support_in_dim, query_dims[task],
                                                   hidden, heads, k, attention=module == "query_attention")
                self.round2_spatial[task] = residual
                self.round2_contract.update(hidden=hidden, neighbours=k,
                    aggregation="attention" if module == "query_attention" else "mean",
                    heads=heads if module == "query_attention" else 0)
            else:
                reference_count = QueryNeighbourResidual.parameter_count(
                    self.width, support_in_dim, query_dims[task], hidden, False)
                hidden = _closest_width(reference_count, lambda h: PointwiseResidual.parameter_count(
                    self.width, query_dims[task], h))
                residual = PointwiseResidual(self.width, query_dims[task], hidden)
                self.round2_spatial[task] = residual
                self.round2_contract.update(hidden=hidden, neighbours=0, aggregation="pointwise")
        actual_count = _count(residual)
        mismatch = abs(actual_count - reference_count) / reference_count
        if mismatch > .05:
            raise ValueError(f"residual parameter matching exceeds 5%: {mismatch:.3%}")
        self.round2_contract.update(residual_parameters=actual_count,
                                    paired_reference_parameters=reference_count,
                                    relative_parameter_mismatch=mismatch)

    def encode(self, support_pos, support_x, support_batch, **context):
        encoded = super().encode(support_pos, support_x, support_batch, **context)
        for geo in encoded["geometry"].values():
            # The original FP stack returns the full original support ordering.
            if len(geo["pos"]) != len(support_x):
                raise ValueError("round2 expects the original full-support FP output")
            geo["raw_x"] = support_x
        return encoded

    def decode(self, encoded, task, query_pos, query_x, query_batch, phase=None, *,
               bc=None, phase_state=None, phase_chunk_size=4):
        if int(phase_chunk_size) < 1:
            raise ValueError("phase_chunk_size must be positive")
        state = phase_state if phase_state is not None else self.prepare_phase(
            encoded, phase, bc, phase_chunk_size=phase_chunk_size)
        if self.round2_module.startswith("temporal_"):
            # Do not break the TCN receptive field at decoder memory chunks.
            # Query chunking remains supported and cannot cross query identities.
            return super().decode(encoded, task, query_pos, query_x, query_batch,
                                  phase_state=state, phase_chunk_size=state["condition"].shape[1])
        key = self._key(task)
        if key not in encoded["geometry"]:
            raise ValueError(f"encode did not include task {task}")
        if query_pos.ndim != 2 or query_pos.shape[1] != 3 or \
                query_x.shape != (len(query_pos), self.query_dims[task]) or \
                query_batch.shape != (len(query_pos),):
            raise ValueError("query positions/features/batch have incompatible shape")
        condition = state["condition"]
        if not len(query_pos):
            return query_x.new_empty((0, condition.shape[1], TASK_CHANNELS[task]))
        if int(query_batch.min()) < 0 or int(query_batch.max()) >= encoded["n_cases"]:
            raise ValueError("query case id has no support geometry")
        geo = encoded["geometry"][key]
        base = _interpolate_fp32(geo["features"], geo["pos"], query_pos,
                                 geo["batch"], query_batch, k=self.k)
        base = base + self.round2_spatial[task](base, query_pos, query_x, query_batch, geo)
        gamma, beta = self.decoders[task].condition_parameters(condition)
        outputs = []
        for begin in range(0, condition.shape[1], phase_chunk_size):
            end = min(begin + phase_chunk_size, condition.shape[1])
            features = base[:, None].expand(-1, end - begin, -1)
            modulation = gamma[query_batch, begin:end], beta[query_batch, begin:end]
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                def head(f, pos, x, g, b, decoder=self.decoders[task]):
                    return decoder(f, pos, x, modulation=(g, b))
                outputs.append(checkpoint(head, features, query_pos, query_x, *modulation,
                                          use_reentrant=False, preserve_rng_state=False))
            else:
                outputs.append(self.decoders[task](features, query_pos, query_x, modulation=modulation))
        return torch.cat(outputs, dim=1)


def build_joint_cycle_model(variant: str, support_in_dim: int, query_dims: Mapping[str, int],
                            *, seed: int = 1234, active_tasks: Sequence[str] | None = None,
                            model_options: Mapping | None = None) -> JointCycleModel:
    options = dict(model_options or {})
    module = options.get("round2_module", "none")
    if module not in ROUND2_MODULES:
        raise ValueError(f"round2_module must be one of {ROUND2_MODULES}")
    cls = JointCycleModel if module == "none" else Round2JointCycleModel
    return cls(variant, support_in_dim, query_dims, seed=seed, active_tasks=active_tasks,
               model_options=options)
