from __future__ import annotations

import numpy as np
import torch


def velocity_jacobian(
    velocity: torch.Tensor, coords: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """Return ``du_i / dx_j`` as ``[N, 3, 3]``."""
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape [N, 3]")
    rows = []
    for component in range(3):
        grad = torch.autograd.grad(
            velocity[:, component].sum(),
            coords,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )[0]
        rows.append(grad)
    return torch.stack(rows, dim=1)


def continuity_residual(
    velocity: torch.Tensor, coords: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    jacobian = velocity_jacobian(velocity, coords, create_graph=create_graph)
    return jacobian[:, 0, 0] + jacobian[:, 1, 1] + jacobian[:, 2, 2]


def local_linear_velocity_gradients(
    coords: np.ndarray,
    velocity: np.ndarray,
    query_indices: np.ndarray,
    *,
    neighbors: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted local-linear CFD gradient oracle.

    Coordinates and velocities should already be nondimensional.  The returned
    gradient is ``du_i/dx_j`` and the condition number documents local numerical
    trustworthiness.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(coords, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    query = np.asarray(query_indices, dtype=np.int64)
    if len(xyz) != len(vel):
        raise ValueError("coordinate/velocity length mismatch")
    k = min(max(int(neighbors), 8), len(xyz))
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz[query], k=k)
    gradients = np.full((len(query), 3, 3), np.nan, dtype=np.float64)
    conditions = np.full(len(query), np.inf, dtype=np.float64)
    for row, (center_index, dist, neighbor_index) in enumerate(
        zip(query, distances, indices)
    ):
        dx = xyz[neighbor_index] - xyz[center_index]
        dv = vel[neighbor_index] - vel[center_index]
        scale = max(float(np.median(dist[1:])), 1e-12)
        weights = np.exp(-np.square(dist / (2.0 * scale)))
        design = np.column_stack([np.ones(len(dx)), dx])
        weighted = design * np.sqrt(weights[:, None])
        target = dv * np.sqrt(weights[:, None])
        gram = weighted.T @ weighted
        conditions[row] = np.linalg.cond(gram)
        try:
            coefficients, *_ = np.linalg.lstsq(weighted, target, rcond=1e-10)
            gradients[row] = coefficients[1:].T
        except np.linalg.LinAlgError:
            continue
    return gradients, conditions

