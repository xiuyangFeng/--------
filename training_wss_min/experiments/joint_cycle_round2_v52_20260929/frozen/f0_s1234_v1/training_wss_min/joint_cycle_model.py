"""PointNet++ geometry and paired parallel-cycle u/p/WSS model variants.

This module is deliberately separate from the established single-task factories.
Targets are *normalised* velocity components (3), relative pressure (1), and
scalar WSS (1). Normalisation and physical units belong to the dataset/runner.
No CFD state, wall derivative, boundary condition, or periodic closure is silently
supplied by the model. Queries may contain a sampled set of phases during training.

``encode`` runs spatial encoding once; ``prepare_phase`` computes spatial token
updates once for all tasks; ``decode`` supports query and phase chunking. Outputs
are always [query, phase, channel], including the scalar heads.
"""
from __future__ import annotations

import copy
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch_geometric.nn import fps, knn_interpolate
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .baseline_models import PointNetPlusPlusRegressor


TASK_CHANNELS = {"velocity": 3, "pressure": 1, "wss": 1}
VARIANTS = ("I", "J0", "J1c", "J1")


def _interpolate_fp32(features, source_pos, query_pos, source_batch, query_batch, *, k):
    """Geometry search and inverse-square interpolation stay outside AMP.

    The 1e-16 distance floor used by PyG is unsafe in float16. Coordinates and
    weights must not follow the half-precision feature dtype. The casts retain
    feature gradients and are inexpensive relative to the query-head layers.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        return knn_interpolate(features.float(), source_pos.float(), query_pos.float(),
                               source_batch, query_batch, k=k)


@contextmanager
def _cpu_seed(seed: int):
    """Reproducible CPU construction without changing the caller's RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        yield


def _geometry_options(options: Mapping | None) -> dict:
    options = dict(options or {})
    cfg = dict(width=32, sa_ratios=(0.25, 0.25, 0.25),
               sa_radius=(0.05, 0.1, 0.2), sa_nsample=(32, 16, 16),
               sa_center_counts=(128, 64, 32), sa_center_sampling="fps",
               fp_knn=3, head_hidden=64, query_decoder="interpolate")
    if options.get("geometry_config_path"):
        raw = json.loads(Path(options["geometry_config_path"]).read_text())
        cfg.update(raw.get("model", raw))
    cfg.update(options.get("geometry", {}))
    # Only the encoder is reused. Old task heads/query refiners are not trained
    # or counted as live parameters in these new controlled comparisons.
    cfg.update(query_decoder="interpolate", output_head="single", out_dim=1,
               case_scale_head=False, query_patch_film=False, section_context=False,
               direction_head=False, head_moe_experts=0, query_decoder_k=0,
               query_decoder_residual=False, query_decoder_feature_indices=())
    accepted = set(inspect.signature(PointNetPlusPlusRegressor).parameters)
    return {k: v for k, v in cfg.items() if k in accepted and k != "in_dim"}


class GeometryEncoder(nn.Module):
    """Reuse the existing local SA/FP network, retaining full-support features."""

    def __init__(self, input_dim: int, options: dict):
        super().__init__()
        self.backbone = PointNetPlusPlusRegressor(in_dim=input_dim, **options)
        self.backbone.head = nn.Identity()  # never used by encode_support
        self.width = int(options["width"])
        self.input_dim = int(input_dim)
        self.k = int(self.backbone.query_interpolation_k)

    def forward(self, pos, x, batch, **context):
        if pos.ndim != 2 or pos.shape[1] != 3 or x.shape != (len(pos), self.input_dim):
            raise ValueError("support positions/features have incompatible shape")
        encoded = self.backbone.encode_support(pos, x, batch, **context)
        return {"pos": encoded[0], "features": encoded[1], "batch": encoded[2]}


class PhaseSpatialInteraction(nn.Module):
    """Dense attention on spatial FPS tokens, independently at each phase.

    The condition enters a nonlinear node transform *before* Q/K attention.
    Thus conditioning can change neighbour weights, not just the final readout.
    J1c supplies zeros for the phase columns; J1 supplies the true phase. Both
    retain the same optional known-BC columns and exactly the same parameters.
    The zero output projection makes both start as the static J0 function.
    """

    def __init__(self, width: int, hidden: int, condition_dim: int, heads: int):
        super().__init__()
        if hidden % heads:
            raise ValueError("token_hidden must be divisible by token_heads")
        self.hidden, self.heads = int(hidden), int(heads)
        self.node = nn.Linear(width, hidden)
        self.position = nn.Linear(3, hidden, bias=False)
        self.condition = nn.Linear(condition_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.relative_bias = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads))
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor, condition: torch.Tensor,
                *, return_attention: bool = False):
        # tokens [M,W], position [M,3], condition [F,D]. No cross-case edges.
        z = self.node(tokens) + self.position(pos)
        h = self.norm(torch.nn.functional.gelu(z[None] + self.condition(condition)[:, None]))
        frames, count, _ = h.shape
        qkv = self.qkv(h).view(frames, count, 3, self.heads, self.hidden // self.heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        rel = pos[:, None] - pos[None, :]
        edge = torch.cat((rel, torch.linalg.vector_norm(rel, dim=-1, keepdim=True)), dim=-1)
        bias = self.relative_bias(edge).permute(2, 0, 1)
        # Cast *before* QK multiplication: casting already-overflowed half
        # logits afterwards would not repair an inf/NaN softmax.
        with torch.autocast(device_type=q.device.type, enabled=False):
            logits = (q.float() @ k.float().transpose(-1, -2)) * ((self.hidden // self.heads) ** -0.5)
            attention = torch.softmax(logits + bias[None].float(), dim=-1).to(v.dtype)
        mixed = (attention @ v).transpose(1, 2).reshape(frames, count, self.hidden)
        delta = self.out(self.mlp(mixed))
        return (delta, attention) if return_attention else delta


class TaskCycleDecoder(nn.Module):
    """Domain-input adapter and phase-conditioned task head without pooling."""

    def __init__(self, width: int, query_dim: int, condition_dim: int, hidden: int,
                 out_channels: int):
        super().__init__()
        self.query_dim = int(query_dim)
        self.adapter = nn.Sequential(nn.Linear(width + query_dim + 3, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.phase = nn.Sequential(nn.Linear(condition_dim, hidden), nn.GELU())
        self.film = nn.Linear(hidden, hidden * 2)
        self.hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.output = nn.Linear(hidden, out_channels)

    def condition_parameters(self, condition):
        """Compute phase FiLM at [case, phase], not repeatedly for every query."""
        return self.film(self.phase(condition)).chunk(2, dim=-1)

    def forward(self, features, query_pos, query_x, condition=None, *, modulation=None):
        # features [Q,F,W]; condition [Q,F,D]. The output layer is not zero:
        # this preserves geometry gradients at the first optimisation step.
        q, frames, _ = features.shape
        raw = torch.cat((features, query_x[:, None].expand(-1, frames, -1),
                         query_pos[:, None].expand(-1, frames, -1)), dim=-1)
        local = self.adapter(raw)
        gamma, beta = self.condition_parameters(condition) if modulation is None else modulation
        return self.output(self.hidden(local * (1 + gamma) + beta))


class JointCycleModel(nn.Module):
    def __init__(self, variant: str, support_in_dim: int, query_dims: Mapping[str, int],
                 *, seed: int = 1234, active_tasks: Sequence[str] | None = None,
                 model_options: Mapping | None = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.active_tasks = tuple(TASK_CHANNELS if active_tasks is None else active_tasks)
        if not self.active_tasks or len(set(self.active_tasks)) != len(self.active_tasks) or \
                any(t not in TASK_CHANNELS for t in self.active_tasks):
            raise ValueError("active_tasks must contain distinct supported task names")
        if any(t not in query_dims or int(query_dims[t]) < 1 for t in self.active_tasks):
            raise ValueError("query_dims must specify positive widths for every active task")
        opts = dict(model_options or {})
        self.phase_dim = int(opts.get("phase_dim", 4))
        self.bc_dim = int(opts.get("bc_dim", 0))
        self.token_count = int(opts.get("token_count", 128))
        # Optional recomputation of stateless interaction/decoder blocks only;
        # never checkpoint the encoder's stateful BatchNorm layers.
        self.activation_checkpointing = bool(opts.get("activation_checkpointing", False))
        if self.phase_dim < 1 or self.bc_dim < 0 or self.token_count < 1:
            raise ValueError("phase/token dimensions must be positive; bc_dim may be zero")
        self.condition_dim = self.phase_dim + self.bc_dim
        self.query_dims = {t: int(query_dims[t]) for t in self.active_tasks}
        geo_options = _geometry_options(opts)
        self.geometry_options = geo_options
        with _cpu_seed(seed):
            common = GeometryEncoder(int(support_in_dim), geo_options)
        self.width, self.k = common.width, common.k
        self.encoders = nn.ModuleDict({t: copy.deepcopy(common) for t in self.active_tasks}
                                     if variant == "I" else {"shared": common})
        self.decoders = nn.ModuleDict()
        # Stable task-specific streams: single-task I runs have the exact same
        # task initialisation as a multi-task J run, regardless of active_tasks.
        for task in self.active_tasks:
            with _cpu_seed(int(seed) + 100 + list(TASK_CHANNELS).index(task)):
                self.decoders[task] = TaskCycleDecoder(
                    self.width, self.query_dims[task], self.condition_dim,
                    int(opts.get("head_hidden", 64)), TASK_CHANNELS[task])
        self.interaction = None
        if variant in ("J1c", "J1"):
            with _cpu_seed(int(seed) + 1000):
                self.interaction = PhaseSpatialInteraction(
                    self.width, int(opts.get("token_hidden", 64)), self.condition_dim,
                    int(opts.get("token_heads", 4)))

    def _key(self, task: str):
        if task not in self.active_tasks:
            raise ValueError(f"inactive task {task!r}")
        return task if self.variant == "I" else "shared"

    def encode(self, support_pos, support_x, support_batch, *, unit_ids=None,
               epoch: int = 0, global_seed: int = 0, evaluation: bool = False,
               support_geometry=None, tasks: Sequence[str] | None = None, task: str | None = None):
        if tasks is not None and task is not None:
            raise ValueError("pass either tasks or task, not both")
        selected = tuple(tasks if tasks is not None else ([task] if task is not None else self.active_tasks))
        if not selected:
            raise ValueError("encode needs at least one task")
        keys = list(dict.fromkeys(self._key(t) for t in selected))
        if support_batch.ndim != 1 or len(support_batch) != len(support_pos) or not len(support_batch):
            raise ValueError("support_batch must identify nonempty support points")
        n_cases = int(support_batch.max().item()) + 1
        if not torch.equal(torch.unique(support_batch), torch.arange(n_cases, device=support_batch.device)):
            raise ValueError("support batch ids must be contiguous from zero")
        result = {"n_cases": n_cases, "geometry": {}}
        for key in keys:
            geo = self.encoders[key](support_pos, support_x, support_batch,
                                     unit_ids=unit_ids, epoch=epoch, global_seed=global_seed,
                                     evaluation=evaluation, geometry=support_geometry)
            if self.interaction is not None:
                token_ids = []
                for case in range(n_cases):
                    members = torch.nonzero(geo["batch"] == case, as_tuple=False).flatten()
                    if len(members) <= self.token_count:
                        ids = members
                    else:
                        # Fixed FPS initial point: tokenisation is geometry-only,
                        # reproducible, and identical between J1c and J1.
                        with torch.no_grad():
                            local = fps(geo["pos"][members], ratio=self.token_count / len(members),
                                        random_start=False)[:self.token_count]
                        ids = members[local]
                    token_ids.append(ids)
                geo["token_ids"] = token_ids
            result["geometry"][key] = geo
        return result

    def _conditions(self, encoded, phase, bc):
        if phase is None or phase.ndim not in (2, 3) or phase.shape[-1] != self.phase_dim:
            raise ValueError(f"phase must be [F,{self.phase_dim}] or [B,F,{self.phase_dim}]")
        n_cases = encoded["n_cases"]
        phase = phase.unsqueeze(0).expand(n_cases, -1, -1) if phase.ndim == 2 else phase
        if phase.shape[0] != n_cases or not phase.shape[1]:
            raise ValueError("phase case/phase dimensions do not match encoded geometry")
        if self.bc_dim:
            if bc is None:
                raise ValueError("bc_dim>0 requires explicit boundary-condition input")
            if bc.ndim == 2:
                bc = bc[:, None].expand(-1, phase.shape[1], -1)
            if bc.shape != (n_cases, phase.shape[1], self.bc_dim):
                raise ValueError("bc must have shape [B,D_bc] or [B,F,D_bc]")
            return torch.cat((phase, bc.to(device=phase.device, dtype=phase.dtype)), dim=-1)
        if bc is not None and bc.numel():
            raise ValueError("nonempty BC input supplied to a model with bc_dim=0")
        return phase

    def prepare_phase(self, encoded, phase, bc=None, *, phase_chunk_size: int = 4):
        if int(phase_chunk_size) < 1:
            raise ValueError("phase_chunk_size must be positive")
        condition = self._conditions(encoded, phase, bc)
        state = {"condition": condition, "token_delta": {}}
        if self.interaction is not None:
            spatial_condition = condition
            if self.variant == "J1c":
                spatial_condition = torch.cat((torch.zeros_like(condition[..., :self.phase_dim]),
                                               condition[..., self.phase_dim:]), dim=-1)
            for key, geo in encoded["geometry"].items():
                state["token_delta"][key] = []
                for case, ids in enumerate(geo["token_ids"]):
                    chunks = []
                    for begin in range(0, condition.shape[1], phase_chunk_size):
                        args = (geo["features"][ids], geo["pos"][ids],
                                spatial_condition[case, begin:begin + phase_chunk_size])
                        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                            chunks.append(activation_checkpoint(self.interaction, *args, use_reentrant=False,
                                                                 preserve_rng_state=False))
                        else:
                            chunks.append(self.interaction(*args))
                    state["token_delta"][key].append(torch.cat(chunks, dim=0))
        return state

    def decode(self, encoded, task, query_pos, query_x, query_batch, phase=None, *,
               bc=None, phase_state=None, phase_chunk_size: int = 4):
        key = self._key(task)
        if key not in encoded["geometry"]:
            raise ValueError(f"encode did not include task {task}")
        if query_pos.ndim != 2 or query_pos.shape[1] != 3 or \
                query_x.shape != (len(query_pos), self.query_dims[task]) or \
                query_batch.shape != (len(query_pos),):
            raise ValueError("query positions/features/batch have incompatible shape")
        if int(phase_chunk_size) < 1:
            raise ValueError("phase_chunk_size must be positive")
        state = phase_state if phase_state is not None else self.prepare_phase(
            encoded, phase, bc, phase_chunk_size=phase_chunk_size)
        condition = state["condition"]
        if not len(query_pos):
            return query_x.new_empty((0, condition.shape[1], TASK_CHANNELS[task]))
        if int(query_batch.min()) < 0 or int(query_batch.max()) >= encoded["n_cases"]:
            raise ValueError("query case id has no support geometry")
        geo = encoded["geometry"][key]
        base = _interpolate_fp32(geo["features"], geo["pos"], query_pos, geo["batch"], query_batch, k=self.k)
        gamma, beta = self.decoders[task].condition_parameters(condition)
        outputs = []
        for begin in range(0, condition.shape[1], phase_chunk_size):
            end = min(begin + phase_chunk_size, condition.shape[1])
            features = base[:, None].expand(-1, end - begin, -1)
            if self.interaction is not None:
                ids = torch.cat(geo["token_ids"])
                # Flatten phase/channel for one spatial interpolation per chunk.
                updates = torch.cat([d[begin:end].permute(1, 0, 2).reshape(len(ix), -1)
                                     for d, ix in zip(state["token_delta"][key], geo["token_ids"])])
                delta = _interpolate_fp32(updates, geo["pos"][ids], query_pos,
                                          geo["batch"][ids], query_batch, k=min(self.k, self.token_count))
                features = features + delta.reshape(len(query_pos), end - begin, self.width)
            modulation = (gamma[query_batch, begin:end], beta[query_batch, begin:end])
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                # Bind the decoder explicitly: a closure over the task loop in
                # a runner could otherwise be recomputed using the wrong head.
                def head(f, pos, x, g, b, decoder=self.decoders[task]):
                    return decoder(f, pos, x, modulation=(g, b))
                outputs.append(activation_checkpoint(head, features, query_pos, query_x, *modulation,
                                                      use_reentrant=False, preserve_rng_state=False))
            else:
                outputs.append(self.decoders[task](features, query_pos, query_x, modulation=modulation))
        return torch.cat(outputs, dim=1)

    def forward(self, support_pos, support_x, support_batch, queries: Mapping,
                phase, *, bc=None, phase_chunk_size: int = 4, **encode_context):
        encoded = self.encode(support_pos, support_x, support_batch,
                              tasks=tuple(queries), **encode_context)
        state = self.prepare_phase(encoded, phase, bc, phase_chunk_size=phase_chunk_size)
        return {task: self.decode(encoded, task, q["pos"], q["x"], q["batch"],
                                  phase_state=state, phase_chunk_size=phase_chunk_size)
                for task, q in queries.items()}


def build_joint_cycle_model(variant: str, support_in_dim: int, query_dims: Mapping[str, int],
                            *, seed: int = 1234, active_tasks: Sequence[str] | None = None,
                            model_options: Mapping | None = None) -> JointCycleModel:
    """Construct paired initialisations; never load a checkpoint implicitly."""
    return JointCycleModel(variant, support_in_dim, query_dims, seed=seed,
                           active_tasks=active_tasks, model_options=model_options)
