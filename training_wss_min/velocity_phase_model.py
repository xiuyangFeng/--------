"""Isolated velocity difficult-phase controls built on the unchanged U0 model.

This module is deliberately opt-in: importing the old factories never imports
or changes it. Geometry frames contain columns (t, n, b) in the same atlas XYZ
coordinates as the original targets. Only geometry and supplied phase/BC enter
the forward path. Target statistics set physical units, never input features.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import scatter

from .joint_cycle_model import TaskCycleDecoder, _cpu_seed, _interpolate_fp32
from .joint_cycle_round2_model import Round2JointCycleModel, _count


VELOCITY_PHASE_ARMS = ("D1", "D2", "A0", "A1", "G00", "G01", "G10", "G11")


class AnchorOperator(nn.Module):
    """Two pointwise/graph layers, with no inactive condition parameters.

    The pointwise controls receive the same local edge-geometry mean as graph
    arms. They do not receive neighbours' hidden features. All arms consume BC;
    phase is supplied only to dynamic operators, before nonlinear processing.
    """

    def __init__(self, input_dim, condition_dim, hidden, output_dim, *, graph):
        super().__init__()
        self.graph, self.hidden = bool(graph), int(hidden)
        self.node = nn.Linear(input_dim, hidden)
        self.condition = nn.Linear(condition_dim, hidden, bias=False) if condition_dim else None
        self.layers = nn.ModuleList(nn.Sequential(
            nn.Linear((2 if graph else 1) * hidden + 7, hidden), nn.GELU(),
            nn.Linear(hidden, hidden)) for _ in range(2))
        self.output = nn.Linear(hidden, output_dim)

    @staticmethod
    def parameter_count(input_dim, condition_dim, hidden, output_dim, *, graph):
        h = int(hidden)
        return (input_dim + 19 + condition_dim + output_dim) * h + \
            2 * (3 if graph else 2) * h * h + output_dim

    def forward(self, nodes, condition, batch, edge_index, edge_attr, edge_summary):
        # [anchor, phase, hidden]. Edges are target-row/source-column.
        h = self.node(nodes)[:, None].expand(-1, condition.shape[1], -1)
        if self.condition is not None:
            h = h + self.condition(condition)[batch]
        h = F.gelu(h)
        target, source = edge_index
        for layer in self.layers:
            if self.graph:
                geometry = edge_attr[:, None].expand(-1, h.shape[1], -1)
                message = layer(torch.cat((h[target], h[source], geometry), dim=-1))
                mixed = scatter(message, target, dim=0, dim_size=len(h), reduce="mean")
            else:
                geometry = edge_summary[:, None].expand(-1, h.shape[1], -1)
                mixed = layer(torch.cat((h, geometry), dim=-1))
            h = h + F.gelu(mixed)
        return self.output(h)


class VelocityPhaseModel(Round2JointCycleModel):
    """U0 plus optional zero-initialized frame and volume-anchor residuals."""

    def __init__(self, cfg, stats):
        arm = str(cfg.get("model", {}).get("velocity_arm", cfg.get("arm", "")))
        if arm not in VELOCITY_PHASE_ARMS:
            raise ValueError(f"velocity_arm must be one of {VELOCITY_PHASE_ARMS}")
        if cfg.get("variant", "I") != "I" or cfg.get("tasks", ["velocity"]) != ["velocity"]:
            raise ValueError("velocity phase arms require independent velocity only")
        options = dict(cfg["model"])
        if options.get("round2_module", "query_mean") != "query_mean":
            raise ValueError("velocity phase controls require the original U0 query_mean path")
        options.update(round2_module="query_mean", bc_dim=stats.get("bc_dims", 0),
                       phase_dim=stats.get("phase_dims", 4))
        seed = int(cfg.get("seed", 1234))
        super().__init__("I", stats["dimensions"]["support"],
                         {"velocity": stats["dimensions"]["velocity"]}, seed=seed,
                         active_tasks=["velocity"], model_options=options)
        self.velocity_arm = arm
        self.has_frame = arm not in ("D1", "D2")
        self.has_anchors = arm.startswith("G")
        self.dynamic_anchors = self.has_anchors and arm[-1] == "1"
        self.graph_anchors = self.has_anchors and arm[1] == "1"
        self.graph_activation_checkpointing = bool(options.get("graph_activation_checkpointing", True))
        std = torch.as_tensor(stats["targets"]["velocity"]["std"], dtype=torch.float32)
        if std.ndim != 2 or std.shape[1] != 3 or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("velocity std must be finite positive [phase, XYZ]")
        self.register_buffer("velocity_target_std", std)
        self.register_buffer("velocity_residual_scale", std.square().mean(-1, keepdim=True).sqrt())
        self.frame_residual = None
        self.anchor_operator = None
        self.anchor_readout = None
        base_count = _count(self)
        frame_hidden = int(options.get("frame_hidden", 64))
        if frame_hidden < 1:
            raise ValueError("frame_hidden must be positive")
        if self.has_frame:
            with _cpu_seed(seed + 3000):
                self.frame_residual = TaskCycleDecoder(
                    self.width, self.query_dims["velocity"] + 10, self.condition_dim, frame_hidden, 3)
                nn.init.zeros_(self.frame_residual.output.weight)
                nn.init.zeros_(self.frame_residual.output.bias)
        contract = dict(arm=arm, base="round2_U0_query_mean", base_parameters=base_count,
                        frame_parameters=_count(self.frame_residual) if self.has_frame else 0,
                        frame_input="atlas_XYZ_columns_t_n_b_plus_validity",
                        residual_coordinates="local_then_XYZ" if arm == "A1" else "XYZ",
                        physical_scale="sqrt(mean_XYZ(training_std_squared))",
                        zero_initial_residual=True, anchor_parameters=0,
                        graph_readout_parameters=0, graph_added_parameters=0,
                        relative_parameter_mismatch=0.)
        if self.has_anchors:
            if int(options.get("graph_layers", 2)) != 2:
                raise ValueError("the frozen graph design has exactly two layers")
            self.anchor_count = int(options.get("graph_anchor_count", 256))
            self.anchor_neighbors = int(options.get("graph_neighbors", 16))
            reference_hidden = int(options.get("graph_hidden", 64))
            if min(self.anchor_count, self.anchor_neighbors, reference_hidden) < 1:
                raise ValueError("anchor dimensions must be positive")
            node_dim = self.width + self.query_dims["velocity"] + 10 + 3 + 7
            cond_dim = self.condition_dim if self.dynamic_anchors else self.bc_dim
            target = AnchorOperator.parameter_count(node_dim, self.condition_dim,
                                                     reference_hidden, self.width, graph=True)
            hidden = min(range(1, 1025), key=lambda h: abs(AnchorOperator.parameter_count(
                node_dim, cond_dim, h, self.width, graph=self.graph_anchors) - target))
            with _cpu_seed(seed + 4000):
                self.anchor_operator = AnchorOperator(node_dim, cond_dim, hidden, self.width,
                                                       graph=self.graph_anchors)
            with _cpu_seed(seed + 5000):
                self.anchor_readout = TaskCycleDecoder(
                    self.width, self.query_dims["velocity"] + 10, self.condition_dim, frame_hidden, 3)
                nn.init.zeros_(self.anchor_readout.output.weight)
                nn.init.zeros_(self.anchor_readout.output.bias)
            actual = _count(self.anchor_operator)
            mismatch = abs(actual - target) / target
            if mismatch > .05:
                raise ValueError(f"active anchor parameter mismatch exceeds 5%: {mismatch:.3%}")
            readout_count = _count(self.anchor_readout)
            family_counts = {}
            for sibling in ("G00", "G01", "G10", "G11"):
                sibling_cond = self.condition_dim if sibling[-1] == "1" else self.bc_dim
                sibling_graph = sibling[1] == "1"
                sibling_hidden = min(range(1, 1025), key=lambda h: abs(
                    AnchorOperator.parameter_count(node_dim, sibling_cond, h, self.width,
                                                   graph=sibling_graph) - target))
                family_counts[sibling] = readout_count + AnchorOperator.parameter_count(
                    node_dim, sibling_cond, sibling_hidden, self.width, graph=sibling_graph)
            family_spread = (max(family_counts.values()) - min(family_counts.values())) / \
                max(family_counts.values())
            if family_spread > .05:
                raise ValueError(f"graph-family active parameter spread exceeds 5%: {family_spread:.3%}")
            contract.update(anchor_count=self.anchor_count, anchor_neighbors=self.anchor_neighbors,
                            graph_layers=2, graph_hidden_actual=hidden,
                            graph_aggregation="mean" if self.graph_anchors else "pointwise",
                            anchor_condition="phase_and_BC" if self.dynamic_anchors else "BC_only",
                            edge_geometry_dim=7, node_edge_geometry_summary="incoming_mean",
                            anchor_parameters=actual, paired_reference_anchor_parameters=target,
                            relative_parameter_mismatch=mismatch,
                            graph_readout_parameters=readout_count,
                            graph_added_parameters=actual + readout_count,
                            graph_family_added_parameters=family_counts,
                            graph_family_relative_parameter_spread=family_spread,
                            graph_activation_checkpointing=self.graph_activation_checkpointing,
                            constant_BC_static_cache="one_phase_then_expand",
                            phase_context="prepared_once_reused_across_query_chunks")
        contract.update(total_parameters=_count(self), added_parameters=_count(self) - base_count)
        self.velocity_phase_contract = contract

    def _validate_anchors(self, anchors, n_cases):
        if anchors is None:
            raise ValueError("G arms require deployment-geometry anchors")
        count = len(anchors["pos"])
        if anchors["pos"].shape != (count, 3) or \
                anchors["x"].shape != (count, self.query_dims["velocity"]) or \
                anchors["geom_extra"].shape != (count, 10) or anchors["batch"].shape != (count,):
            raise ValueError("anchor positions/features/frame dimensions are incompatible")
        if count != n_cases * self.anchor_count or not torch.equal(
                torch.bincount(anchors["batch"], minlength=n_cases),
                torch.full((n_cases,), self.anchor_count, device=anchors["batch"].device)):
            raise ValueError("each case must have the configured fixed anchor count")
        edges, edge_attr = anchors["edge_index"], anchors["edge_attr"]
        if edges.ndim != 2 or edges.shape[0] != 2 or edge_attr.shape != (edges.shape[1], 7) \
                or edges.dtype != torch.long or not edges.numel() or \
                int(edges.min()) < 0 or int(edges.max()) >= count:
            raise ValueError("anchor edge index/geometry shape or range is invalid")
        target, source = edges
        if not torch.equal(anchors["batch"][target], anchors["batch"][source]):
            raise ValueError("anchor edges cannot cross cases")
        if not torch.equal(torch.bincount(target, minlength=count),
                           torch.full((count,), self.anchor_neighbors, device=target.device)):
            raise ValueError("each anchor needs exactly graph_neighbors incoming edges")
        if any(not torch.isfinite(anchors[k]).all() for k in ("pos", "x", "geom_extra", "edge_attr")):
            raise ValueError("anchor inputs must be finite")

    def encode(self, support_pos, support_x, support_batch, *, anchors=None, **context):
        encoded = super().encode(support_pos, support_x, support_batch, **context)
        if self.has_anchors:
            self._validate_anchors(anchors, encoded["n_cases"])
            geo = encoded["geometry"]["velocity"]
            interpolated = _interpolate_fp32(geo["features"], geo["pos"], anchors["pos"],
                                             geo["batch"], anchors["batch"], k=self.k)
            summary = scatter(anchors["edge_attr"], anchors["edge_index"][0], dim=0,
                              dim_size=len(anchors["pos"]), reduce="mean")
            nodes = torch.cat((interpolated, anchors["x"], anchors["geom_extra"],
                               anchors["pos"], summary), dim=-1)
            encoded["anchors"] = dict(anchors, node_features=nodes, edge_summary=summary)
        return encoded

    def prepare_phase(self, encoded, phase, bc=None, *, phase_chunk_size=4, phase_indices=None):
        state = super().prepare_phase(encoded, phase, bc, phase_chunk_size=phase_chunk_size)
        frames = state["condition"].shape[1]
        if phase_indices is None:
            if frames != len(self.velocity_target_std):
                raise ValueError("phase_indices is required for a subset/reordering of target phases")
            phase_indices = torch.arange(frames, device=phase.device)
        phase_indices = torch.as_tensor(phase_indices, device=phase.device, dtype=torch.long)
        if phase_indices.shape != (frames,) or int(phase_indices.min()) < 0 or \
                int(phase_indices.max()) >= len(self.velocity_target_std):
            raise ValueError("phase_indices must identify the supplied target phases")
        state["phase_indices"] = phase_indices
        if self.has_anchors:
            a = encoded["anchors"]
            condition = state["condition"]
            if not self.dynamic_anchors:
                condition = condition[..., self.phase_dim:]
            # A static operator with case-level BC has precisely one state.
            # Retain the general time-dependent BC API: only collapse when the
            # actual supplied BC columns are identical across all phases.
            repeat_static = not self.dynamic_anchors and torch.equal(
                condition, condition[:, :1].expand_as(condition))
            if repeat_static:
                condition = condition[:, :1]
            chunks = []
            for begin in range(0, condition.shape[1], phase_chunk_size):
                args = (a["node_features"], condition[:, begin:begin + phase_chunk_size],
                        a["batch"], a["edge_index"], a["edge_attr"], a["edge_summary"])
                if self.graph_activation_checkpointing and self.training and torch.is_grad_enabled():
                    chunks.append(checkpoint(self.anchor_operator, *args, use_reentrant=False,
                                             preserve_rng_state=False))
                else:
                    chunks.append(self.anchor_operator(*args))
            context = torch.cat(chunks, dim=1)
            state["anchor_features"] = context.expand(-1, frames, -1) if repeat_static else context
        return state

    def physical_residual_to_z(self, residual, query_frame, query_geom, phase_indices):
        """Convert a dimensionless residual through isotropic physical units."""
        with torch.autocast(residual.device.type, enabled=False):
            xyz = residual.float()
            if self.velocity_arm == "A1":
                valid = query_geom[:, -1] > .5
                frame = query_frame.float()
                identity = torch.eye(3, dtype=frame.dtype, device=frame.device).expand_as(frame)
                frame = torch.where(valid[:, None, None], frame, identity)
                xyz = torch.einsum("nij,nfj->nfi", frame, xyz)
            scale = self.velocity_residual_scale[phase_indices]
            std = self.velocity_target_std[phase_indices]
            return xyz * scale[None] / std[None]

    @staticmethod
    def _check_query_geometry(query_pos, query_frame, query_geom):
        n = len(query_pos)
        if query_frame is None or query_geom is None or query_frame.shape != (n, 3, 3) \
                or query_geom.shape != (n, 10):
            raise ValueError("frame arms require query_frame[N,3,3] and query_geom[N,10]")
        if not torch.isfinite(query_frame).all() or not torch.isfinite(query_geom).all():
            raise ValueError("query frame and geometry must be finite")

    def decode(self, encoded, task, query_pos, query_x, query_batch, phase=None, *, bc=None,
               phase_state=None, phase_chunk_size=4, query_frame=None, query_geom=None):
        state = phase_state if phase_state is not None else self.prepare_phase(
            encoded, phase, bc, phase_chunk_size=phase_chunk_size)
        if not self.has_frame:
            return super().decode(encoded, task, query_pos, query_x, query_batch,
                                  phase_state=state, phase_chunk_size=phase_chunk_size)
        if int(phase_chunk_size) < 1:
            raise ValueError("phase_chunk_size must be positive")
        key = self._key(task)
        if key not in encoded["geometry"]:
            raise ValueError(f"encode did not include task {task}")
        if query_pos.ndim != 2 or query_pos.shape[1] != 3 or \
                query_x.shape != (len(query_pos), self.query_dims[task]) or \
                query_batch.shape != (len(query_pos),):
            raise ValueError("query positions/features/batch have incompatible shape")
        self._check_query_geometry(query_pos, query_frame, query_geom)
        if not len(query_pos):
            return query_x.new_empty((0, state["condition"].shape[1], 3))
        if int(query_batch.min()) < 0 or int(query_batch.max()) >= encoded["n_cases"]:
            raise ValueError("query case id has no support geometry")
        # Preserve U0's operation order, but reuse its query feature tensor for
        # all new heads rather than performing kNN/edge aggregation twice.
        geo = encoded["geometry"][key]
        features = _interpolate_fp32(geo["features"], geo["pos"], query_pos,
                                     geo["batch"], query_batch, k=self.k)
        features = features + self.round2_spatial[task](features, query_pos, query_x, query_batch, geo)
        frame_x = torch.cat((query_x, query_geom), dim=-1)
        condition, phases = state["condition"], state["phase_indices"]
        base_mod = self.decoders[task].condition_parameters(condition)
        frame_mod = self.frame_residual.condition_parameters(condition)
        anchor_mod = self.anchor_readout.condition_parameters(condition) if self.has_anchors else None
        outputs = []
        for begin in range(0, condition.shape[1], phase_chunk_size):
            end = min(begin + phase_chunk_size, condition.shape[1])
            local = features[:, None].expand(-1, end - begin, -1)
            modulation = tuple(x[query_batch, begin:end] for x in base_mod)
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                def head(f, pos, x, g, b, decoder=self.decoders[task]):
                    return decoder(f, pos, x, modulation=(g, b))
                base_prediction = checkpoint(head, local, query_pos, query_x, *modulation,
                                             use_reentrant=False, preserve_rng_state=False)
            else:
                base_prediction = self.decoders[task](local, query_pos, query_x, modulation=modulation)
            modulation = tuple(x[query_batch, begin:end] for x in frame_mod)
            residual = self.frame_residual(local, query_pos, frame_x, modulation=modulation)
            if self.has_anchors:
                a = encoded["anchors"]
                latent = state["anchor_features"][:, begin:end]
                mixed = _interpolate_fp32(latent.reshape(len(latent), -1), a["pos"], query_pos,
                                          a["batch"], query_batch, k=min(self.k, self.anchor_count))
                mixed = mixed.reshape(len(query_pos), end - begin, self.width)
                modulation = tuple(x[query_batch, begin:end] for x in anchor_mod)
                residual = residual + self.anchor_readout(mixed, query_pos, frame_x,
                                                          modulation=modulation)
            outputs.append(base_prediction + self.physical_residual_to_z(
                residual, query_frame, query_geom, phases[begin:end]))
        return torch.cat(outputs, dim=1)

    def forward(self, support_pos, support_x, support_batch, queries, phase, *, bc=None,
                phase_chunk_size=4, phase_indices=None, anchors=None, **encode_context):
        encoded = self.encode(support_pos, support_x, support_batch, anchors=anchors,
                              tasks=tuple(queries), **encode_context)
        state = self.prepare_phase(encoded, phase, bc, phase_chunk_size=phase_chunk_size,
                                   phase_indices=phase_indices)
        return {task: self.decode(encoded, task, q["pos"], q["x"], q["batch"],
                                  phase_state=state, phase_chunk_size=phase_chunk_size,
                                  query_frame=q.get("frame"), query_geom=q.get("geom_extra"))
                for task, q in queries.items()}


def build_velocity_phase_model(cfg, stats):
    """Construct an opt-in arm; never load/rewrite any existing checkpoint."""
    return VelocityPhaseModel(cfg, stats)
