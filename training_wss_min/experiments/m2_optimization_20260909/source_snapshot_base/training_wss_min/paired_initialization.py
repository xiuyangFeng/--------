"""Pair fresh initial weights with a reference configuration, never a checkpoint."""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .models import build_model
from .runtime import seed_all


def _rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state().clone(), "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
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
    else:
        raise ValueError(f"no isolated fresh-initialization factory for new module {root!r}")
    for key in keys:
        setattr(model_cfg, key, copy.deepcopy(getattr(cfg.model, key)))
    return model_cfg


def build_paired_model(cfg: C.ExpConfig) -> tuple[torch.nn.Module, dict]:
    """Construct shared tensors from seed-matched M2 and new modules by name.

    A complete donor construction preserves custom initializers (including zero
    residual projections), unlike calling reset_parameters on arbitrary leaves.
    New modules with the same qualified name share their stream across arms.
    Only local-wall branch width changes may change existing tensor shapes.
    """
    reference_value = getattr(cfg.train, "init_reference_config", None)
    if not reference_value:
        raise ValueError("paired initialization requires init_reference_config")
    if cfg.train.init_checkpoint_path:
        raise ValueError("paired fresh initialization cannot load a trained checkpoint")
    reference_path = Path(reference_value)
    if not reference_path.is_absolute():
        reference_path = C.PROJECT_ROOT / reference_path
    reference_cfg = C.ExpConfig.from_json(reference_path)
    if tuple(cfg.data.input_features) != tuple(reference_cfg.data.input_features):
        raise ValueError("paired initialization requires the reference input channel order")
    if cfg.model.name != reference_cfg.model.name:
        raise ValueError("paired initialization requires the reference model family")

    entry_rng = _rng_state()
    final_rng = entry_rng
    try:
        seed_all(cfg.train.seed)
        reference = build_model(reference_cfg.model, C.input_dim(reference_cfg))
        final_rng = _rng_state()
        reference_state = reference.state_dict()
        reference_modules = dict(reference.named_modules())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_substream(cfg.train.seed, "candidate"))
            model = build_model(cfg.model, C.input_dim(cfg))
        state = model.state_dict()
        missing = sorted(set(reference_state) - set(state))
        if missing:
            raise ValueError(f"paired initialization lost reference tensors: {missing}")
        changed = sorted(k for k in reference_state if state[k].shape != reference_state[k].shape)
        forbidden = [k for k in changed if not k.startswith("local_wall_branch.")]
        if forbidden:
            raise ValueError(f"paired initialization has unapproved shape changes: {forbidden}")

        added = sorted(set(state) - set(reference_state))
        roots = set()
        for key in added:
            parts = key.split(".")[:-1]
            root = next((".".join(parts[:n]) for n in range(1, len(parts) + 1)
                         if ".".join(parts[:n]) not in reference_modules), None)
            if root is None:
                # Direct parameters added to an existing module use that module's
                # own stream; shared reference tensors are restored below.
                root = ".".join(parts)
            roots.add(root)
        if changed:
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
            if state[key].shape == tensor.shape:
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
            "substream_seeds": streams,
            "post_reference_cpu_rng_sha256": hashlib.sha256(final_rng["cpu"].numpy().tobytes()).hexdigest(),
            "rng_restored_to": "after_fresh_reference_model_construction",
        }
        return model, evidence
    finally:
        _restore_rng(final_rng)
