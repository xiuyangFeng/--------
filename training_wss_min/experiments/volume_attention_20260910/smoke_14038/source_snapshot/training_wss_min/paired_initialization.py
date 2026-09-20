"""Pair fresh initial weights with a reference configuration, never a checkpoint."""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path
import random

import numpy as np
import torch

from . import config as C
from .models import build_model
from .runtime import seed_all


def _rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state().clone(), "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _substream(seed: int, module: str) -> int:
    payload = f"m2_optimization_20260909|{seed}|{module}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _donor_config(cfg: C.ExpConfig, reference: C.ExpConfig, root: str):
    """Isolate this module's changes so other additions cannot consume its RNG."""
    model_cfg = copy.deepcopy(reference.model)
    if root == "local_query":
        keys = [key for key in vars(cfg.model) if key.startswith("query_")]
        keys.append("sep_beta_init")
    elif root == "local_wall_branch" or root.startswith("local_wall_branch."):
        keys = [key for key in vars(cfg.model) if key.startswith("local_branch")]
        keys.append("pointwise_branch_hidden")
    elif root.startswith("sa.") and ".blocks." in root:
        stage = int(root.split(".")[1])
        blocks = list(model_cfg.sa_blocks)
        blocks[stage] = cfg.model.sa_blocks[stage]
        model_cfg.sa_blocks = tuple(blocks)
        keys = ["invres_expansion", "residual_scale_init", "invres_radius_scale"]
    elif root == f"sa.{len(cfg.model.sa_radius) - 1}" and cfg.model.multiradius_values:
        keys = [key for key in vars(cfg.model) if key.startswith(("multiradius_", "bottleneck_"))]
        keys += ["sa_nsample", "sa_center_counts", "sa_center_sampling", "sa_radius",
                 "coarse_attention", "neighbor_drop_rate"]
    elif root == "bottleneck" or root.startswith("bottleneck."):
        keys = [key for key in vars(cfg.model) if key.startswith("bottleneck_")]
        keys.append("coarse_attention")
    elif root in {"velocity_head", "pressure_head", "qad_velocity_out", "qad_pressure_out"}:
        keys = ["output_head", "out_dim", "query_decoder", "qad_hidden"]
    elif root in {"qad_local", "qad_gate", "qad_out"}:
        keys = ["query_decoder", "qad_hidden", "output_head", "out_dim"]
    else:
        raise ValueError(f"no isolated fresh-initialization factory for new module {root!r}")
    for key in keys:
        setattr(model_cfg, key, copy.deepcopy(getattr(cfg.model, key)))
    return model_cfg


def build_paired_model(cfg: C.ExpConfig, *, _reference_chain: tuple[Path, ...] = ()) -> tuple[torch.nn.Module, dict]:
    """Construct shared tensors from seed-matched M2 and new modules by name.

    A complete donor construction preserves custom initializers (including zero
    residual projections), unlike calling reset_parameters on arbitrary leaves.
    New modules with the same qualified name share their stream across arms.
    Explicit final-SA replacement and the single-to-joint head adaptation may
    remove their own reference tensors; every other shared tensor is preserved.
    A reference that itself uses paired initialization is reconstructed by the
    same fresh path, so chained references never silently change the parent.
    """
    reference_value = getattr(cfg.train, "init_reference_config", None)
    if not reference_value:
        raise ValueError("paired initialization requires init_reference_config")
    if cfg.train.init_checkpoint_path:
        raise ValueError("paired fresh initialization cannot load a trained checkpoint")
    reference_path = Path(reference_value)
    if not reference_path.is_absolute():
        reference_path = C.PROJECT_ROOT / reference_path
    reference_path = reference_path.resolve()
    if reference_path in _reference_chain:
        raise ValueError("cyclic fresh-initialization reference: " + str(reference_path))
    reference_cfg = C.ExpConfig.from_json(reference_path)
    if reference_cfg.train.init_checkpoint_path:
        raise ValueError("fresh-initialization reference cannot load a trained checkpoint")
    if int(reference_cfg.train.seed) != int(cfg.train.seed):
        raise ValueError("paired initialization requires the same reference seed")
    if tuple(cfg.data.input_features) != tuple(reference_cfg.data.input_features):
        raise ValueError("paired initialization requires the reference input channel order")
    if cfg.model.name != reference_cfg.model.name:
        raise ValueError("paired initialization requires the reference model family")

    entry_rng = _rng_state()
    final_rng = entry_rng
    try:
        seed_all(cfg.train.seed)
        reference_evidence = None
        if reference_cfg.train.init_reference_config:
            reference, reference_evidence = build_paired_model(
                reference_cfg, _reference_chain=(*_reference_chain, reference_path))
        else:
            reference = build_model(reference_cfg.model, C.input_dim(reference_cfg))
        final_rng = _rng_state()
        reference_state = reference.state_dict()
        reference_modules = dict(reference.named_modules())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_substream(cfg.train.seed, "candidate"))
            model = build_model(cfg.model, C.input_dim(cfg))
        state = model.state_dict()
        replacement_roots = set()
        discarded_roots = set()
        final_sa = f"sa.{len(cfg.model.sa_radius) - 1}"
        if cfg.model.multiradius_values and not reference_cfg.model.multiradius_values:
            if len(cfg.model.sa_radius) != len(reference_cfg.model.sa_radius):
                raise ValueError("final-SA replacement cannot change the number of stages")
            replacement_roots.add(final_sa)
            discarded_roots.add(final_sa)
        if (cfg.model.output_head == "velocity_pressure"
                and reference_cfg.model.output_head == "single"):
            replacement_roots.update({"velocity_head", "pressure_head", "qad_velocity_out", "qad_pressure_out"})
            discarded_roots.update({"head", "qad_out"})
        if (cfg.model.bottleneck_transformer and reference_cfg.model.bottleneck_transformer
                and cfg.model.bottleneck_mode != reference_cfg.model.bottleneck_mode):
            replacement_roots.add("bottleneck.layers")
            discarded_roots.add("bottleneck.layers")

        def under(key, roots):
            return any(key == root or key.startswith(root + ".") for root in roots)

        missing = sorted(set(reference_state) - set(state))
        forbidden_missing = [key for key in missing if not under(key, discarded_roots)]
        if forbidden_missing:
            raise ValueError(f"paired initialization lost reference tensors: {forbidden_missing}")
        changed = sorted(k for k in reference_state.keys() & state.keys()
                         if state[k].shape != reference_state[k].shape)
        forbidden = [k for k in changed if not k.startswith("local_wall_branch.")
                     and not under(k, replacement_roots)]
        if forbidden:
            raise ValueError(f"paired initialization has unapproved shape changes: {forbidden}")

        added = sorted(set(state) - set(reference_state))
        roots = set(replacement_roots)
        for key in added:
            parts = key.split(".")[:-1]
            root = next((".".join(parts[:n]) for n in range(1, len(parts) + 1)
                         if ".".join(parts[:n]) not in reference_modules), None)
            if root is None:
                # Direct parameters added to an existing module use that module's
                # own stream; shared reference tensors are restored below.
                root = ".".join(parts)
            roots.add(root)
        if any(key.startswith("local_wall_branch.") for key in changed):
            roots.add("local_wall_branch")
        roots = sorted(root for root in roots if not any(
            root.startswith(other + ".") for other in roots if other != root
        ))
        streams = {}
        for root in roots:
            stream = _substream(cfg.train.seed, root)
            streams[root] = stream
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stream)
                donor = build_model(_donor_config(cfg, reference_cfg, root), C.input_dim(cfg)).state_dict()
            for key in state:
                if key.startswith(root + ".") or not root:
                    state[key] = donor[key].clone()
        copied = []
        for key, tensor in reference_state.items():
            if key in state and state[key].shape == tensor.shape and not under(key, replacement_roots):
                state[key] = tensor.clone()
                copied.append(key)
        model.load_state_dict(state, strict=True)
        evidence = {
            "mode": "paired_fresh_random_initialization", "seed": cfg.train.seed,
            "reference_config": str(reference_path.resolve()),
            "reference_config_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "trained_checkpoint_loaded": False,
            "shared_tensors_exact": True, "reference_rng_restored": True,
            "reference_state_sha256": _state_hash(reference_state),
            "initial_state_sha256": _state_hash(state),
            "copied_keys": copied, "new_keys": added, "changed_shape_keys": changed,
            "replaced_module_roots": sorted(replacement_roots), "discarded_reference_keys": missing,
            "reference_initialization": reference_evidence,
            "substream_seeds": streams,
            "post_reference_cpu_rng_sha256": hashlib.sha256(final_rng["cpu"].numpy().tobytes()).hexdigest(),
            "rng_restored_to": "after_fresh_reference_model_construction",
        }
        return model, evidence
    finally:
        _restore_rng(final_rng)
