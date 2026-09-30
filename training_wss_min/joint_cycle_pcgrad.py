"""PCGrad on shared parameters only, preserving the mean-loss private heads.

GradScaler is used for each differentiation; shared task gradients are unscaled
before projection. The original mean loss is then differentiated and unscaled
through the optimizer, so private-head gradients and AMP overflow accounting
retain the original runner's semantics. This costs extra backward passes and
is intentionally reported as a distinct compute budget.
"""
from __future__ import annotations

import random

import torch


def project_conflicting_gradients(vectors, seed):
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("PCGrad needs at least two task gradient vectors")
    if not bool(torch.isfinite(vectors).all()):
        raise FloatingPointError("nonfinite unscaled task gradient before PCGrad")
    norm2 = vectors.square().sum(1)
    gram = vectors @ vectors.T
    denominator = torch.sqrt(norm2[:, None] * norm2[None, :]).clamp_min(1e-30)
    cosines = (gram / denominator).clamp(-1, 1)
    projected = vectors.clone()
    rng = random.Random(int(seed))
    for i in range(len(vectors)):
        order = [j for j in range(len(vectors)) if j != i]
        rng.shuffle(order)
        for j in order:
            dot = torch.dot(projected[i], vectors[j])
            coefficient = dot.clamp_max(0) / norm2[j].clamp_min(1e-30)
            projected[i].sub_(coefficient * vectors[j])
    merged = projected.mean(0)
    original = vectors.mean(0)
    diagnostics = {
        "norms": norm2.sqrt().detach().cpu().tolist(),
        "cosines": cosines.detach().cpu().tolist(),
        "conflict_fraction": float((gram[torch.triu(torch.ones_like(gram, dtype=torch.bool), 1)] < 0).float().mean()),
        "relative_shared_update_change": float((merged-original).norm() / original.norm().clamp_min(1e-30)),
    }
    return merged, diagnostics


def pcgrad_backward(task_losses, model, optimizer, scaler, *, seed):
    """Backward, unscale, project; caller still clips and steps the optimizer."""
    tasks = list(task_losses)
    if len(tasks) < 2 or getattr(model, "variant", None) == "I":
        raise ValueError("PCGrad is registered only for joint shared-task models")
    shared = [p for p in model.encoders.parameters() if p.requires_grad]
    if model.interaction is not None:
        shared += [p for p in model.interaction.parameters() if p.requires_grad]
    if not shared or len({id(p) for p in shared}) != len(shared):
        raise ValueError("shared parameter set must be nonempty and unique")
    vectors = []
    scale = float(scaler.get_scale())
    used = [False] * len(shared)
    for task in tasks:
        gradients = torch.autograd.grad(scaler.scale(task_losses[task]), shared,
                                        retain_graph=True, allow_unused=True)
        vector = []
        for index, (p, g) in enumerate(zip(shared, gradients)):
            used[index] |= g is not None
            vector.append(torch.zeros_like(p, dtype=torch.float32).reshape(-1) if g is None
                          else g.detach().float().reshape(-1) / scale)
        vectors.append(torch.cat(vector))
    merged, diagnostics = project_conflicting_gradients(torch.stack(vectors), seed)
    total = torch.stack(list(task_losses.values())).mean()
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    # Stop for an overflow anywhere, including a task-specific decoder. Never
    # replace a nonfinite gradient with a finite projected value and hide it.
    if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("nonfinite unscaled mean-loss gradient before PCGrad assignment")
    offset = 0
    for p, active in zip(shared, used):
        count = p.numel()
        if active:
            replacement = merged[offset:offset+count].view_as(p).to(p.dtype)
            if p.grad is None:
                p.grad = replacement.clone()
            else:
                p.grad.copy_(replacement)
        offset += count
    diagnostics["tasks"] = tasks
    diagnostics["shared_parameter_count"] = merged.numel()
    return diagnostics
