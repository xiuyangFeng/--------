from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

BLOOD_VISCOSITY = 0.0035  # Pa·s


@dataclass(frozen=True)
class HemoSample:
    sample_id: str
    patient_id: str
    phase: str
    time_step: int
    source: str
    model_name: str
    split_version: str
    wall_mask: torch.Tensor
    y_field: torch.Tensor
    positions: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None


def parse_sample_id(sample_id: str) -> Tuple[str, str, int]:
    parts = sample_id.split("__")
    patient_id = parts[0]
    phase = "unknown"
    time_step = 0
    for part in parts[1:]:
        lower = part.lower()
        if lower.startswith("phase-"):
            phase = part.split("-", 1)[1]
        elif lower.startswith("t-"):
            try:
                time_step = int(part.split("-", 1)[1])
            except ValueError:
                time_step = 0
        elif part.isdigit():
            time_step = int(part)
    return patient_id, phase, time_step


def _estimate_wall_normals(
    positions: torch.Tensor,
    wall_mask: torch.Tensor,
    edge_index: torch.Tensor,
    k_internal: int = 5,
) -> torch.Tensor:
    """基于局部 wall-internal 几何关系估计壁面外法向。

    对每个壁面点，先沿图边寻找邻近内部点，再用“内部点均值指向壁面点”
    的方向作为近似外法向。如果当前壁面点周围找不到内部邻居，则回退到
    固定方向，保证后续 WSS 计算链路不中断。
    """
    wall_idx = torch.nonzero(wall_mask, as_tuple=False).flatten()
    internal_mask = ~wall_mask
    n_wall = wall_idx.size(0)
    normals = torch.zeros(n_wall, 3, device=positions.device, dtype=positions.dtype)

    src, dst = edge_index[0], edge_index[1]

    for local_i, wi in enumerate(wall_idx.tolist()):
        neighbours = dst[src == wi]
        internal_neighbours = neighbours[internal_mask[neighbours]]
        if internal_neighbours.numel() == 0:
            neighbours_rev = src[dst == wi]
            internal_neighbours = neighbours_rev[internal_mask[neighbours_rev]]

        if internal_neighbours.numel() == 0:
            normals[local_i] = torch.tensor([0.0, 0.0, 1.0], device=positions.device)
            continue

        if internal_neighbours.numel() > k_internal:
            dists = (positions[internal_neighbours] - positions[wi]).norm(dim=1)
            _, topk = dists.topk(k_internal, largest=False)
            internal_neighbours = internal_neighbours[topk]

        direction = positions[wi] - positions[internal_neighbours].mean(dim=0)
        norm = direction.norm()
        if norm > 1e-10:
            normals[local_i] = direction / norm
        else:
            normals[local_i] = torch.tensor([0.0, 0.0, 1.0], device=positions.device)

    return normals


def compute_wss(
    y_field: torch.Tensor,
    wall_mask: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    mu: float = BLOOD_VISCOSITY,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """沿壁面法向的速度梯度计算 WSS。

    计算公式：
    ``WSS_vector = mu * (du/dn - (du/dn · n) * n)``
    即先求法向方向上的速度梯度，再投影到切向平面。

    当预测产物没有保存坐标或边信息时，自动回退到旧版 proxy，
    以兼容历史导出结果。
    """
    wall_mask_bool = wall_mask.bool() if wall_mask.dtype != torch.bool else wall_mask
    wall_velocity = y_field[:, :3][wall_mask_bool]

    if wall_velocity.numel() == 0:
        empty = y_field.new_zeros((0,))
        empty_vec = y_field.new_zeros((0, 3))
        return {"wss": empty, "wss_vector": empty_vec, "method": "empty"}

    if positions is None or edge_index is None:
        return _compute_wss_proxy(y_field, wall_mask_bool, epsilon)

    wall_idx = torch.nonzero(wall_mask_bool, as_tuple=False).flatten()
    internal_mask = ~wall_mask_bool
    src, dst = edge_index[0], edge_index[1]

    normals = _estimate_wall_normals(positions, wall_mask_bool, edge_index)

    n_wall = wall_idx.size(0)
    wss_vectors = torch.zeros(n_wall, 3, device=y_field.device, dtype=y_field.dtype)

    for local_i, wi in enumerate(wall_idx.tolist()):
        neighbours = dst[src == wi]
        internal_neighbours = neighbours[internal_mask[neighbours]]
        if internal_neighbours.numel() == 0:
            neighbours_rev = src[dst == wi]
            internal_neighbours = neighbours_rev[internal_mask[neighbours_rev]]

        if internal_neighbours.numel() == 0:
            wss_vectors[local_i] = mu * wall_velocity[local_i]
            continue

        dists = (positions[internal_neighbours] - positions[wi]).norm(dim=1)
        closest_idx = internal_neighbours[dists.argmin()]
        dn = dists.min().clamp_min(epsilon)

        vel_wall = y_field[wi, :3]
        vel_internal = y_field[closest_idx, :3]
        du_dn = (vel_internal - vel_wall) / dn

        n = normals[local_i]
        du_dn_tangent = du_dn - (du_dn @ n) * n
        wss_vectors[local_i] = mu * du_dn_tangent

    wss_mag = wss_vectors.norm(dim=1)
    return {"wss": wss_mag, "wss_vector": wss_vectors, "method": "gradient"}


def _compute_wss_proxy(
    y_field: torch.Tensor,
    wall_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """WSS 的旧版 proxy，保留给历史结果和缺字段样本使用。"""
    wall_velocity = y_field[:, :3][wall_mask]
    if wall_velocity.numel() == 0:
        empty = y_field.new_zeros((0,))
        empty_vec = y_field.new_zeros((0, 3))
        return {"wss": empty, "wss_vector": empty_vec, "method": "proxy"}
    wss = wall_velocity.norm(dim=1)
    denom = wss.unsqueeze(1).clamp_min(epsilon)
    return {
        "wss": wss,
        "wss_vector": wall_velocity / denom * wss.unsqueeze(1),
        "method": "proxy",
    }


def compute_cycle_metrics(
    samples: Sequence[HemoSample],
    mu: float = BLOOD_VISCOSITY,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("样本序列不能为空")

    wall_mask = samples[0].wall_mask.bool()
    wss_stack: List[torch.Tensor] = []
    wss_vector_stack: List[torch.Tensor] = []
    for sample in sorted(samples, key=lambda item: item.time_step):
        result = compute_wss(
            sample.y_field,
            wall_mask=sample.wall_mask.bool(),
            positions=sample.positions,
            edge_index=sample.edge_index,
            mu=mu,
            epsilon=epsilon,
        )
        wss_stack.append(result["wss"])
        wss_vector_stack.append(result["wss_vector"])

    wss_t = torch.stack(wss_stack, dim=0)
    wss_vec_t = torch.stack(wss_vector_stack, dim=0)

    tawss = wss_t.mean(dim=0)
    # OSI / RRT 的写法严格遵循 docs/任务B指标计算规范.md 里的离散公式形式。
    osi_num = wss_vec_t.sum(dim=0).norm(dim=1)
    osi_den = wss_t.sum(dim=0).clamp_min(epsilon)
    osi = 0.5 * (1.0 - osi_num / osi_den)
    rrt = 1.0 / ((1.0 - 2.0 * osi).clamp_min(epsilon) * tawss.clamp_min(epsilon))

    return {
        "wss_time_series": wss_t,
        "wss_vector_time_series": wss_vec_t,
        "tawss": tawss,
        "osi": osi,
        "rrt": rrt,
    }


def summarize_region(values: torch.Tensor) -> Dict[str, float]:
    # 第一版先只做最小区域统计骨架，后续接入真实 region mask 后直接复用这里。
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "risk_area_ratio": 0.0,
        }
    risk_threshold = torch.quantile(values, 0.9)
    return {
        "mean": values.mean().item(),
        "max": values.max().item(),
        "p95": torch.quantile(values, 0.95).item(),
        "risk_area_ratio": (values >= risk_threshold).float().mean().item(),
    }


def build_per_case_region_rows(
    *,
    samples: Sequence[HemoSample],
    source: str,
    model_name: str,
    split_version: str,
    region_name: str = "wall",
) -> List[Dict[str, object]]:
    # 当前默认 region_name="wall"，相当于“全壁面区域”。
    # 等区域划分规则补齐后，这里可以按多个 region mask 输出多行。
    by_case: Dict[Tuple[str, str], List[HemoSample]] = {}
    for sample in samples:
        by_case.setdefault((sample.patient_id, sample.phase), []).append(sample)

    rows: List[Dict[str, object]] = []
    for (patient_id, phase), case_samples in sorted(by_case.items()):
        metrics = compute_cycle_metrics(case_samples)
        tawss_summary = summarize_region(metrics["tawss"])
        osi_summary = summarize_region(metrics["osi"])
        rrt_summary = summarize_region(metrics["rrt"])
        rows.append(
            {
                "patient_id": patient_id,
                "phase": phase,
                "region_name": region_name,
                "source": source,
                "model_name": model_name,
                "split_version": split_version,
                "num_time_steps": len(case_samples),
                "TAWSS_mean": tawss_summary["mean"],
                "TAWSS_max": tawss_summary["max"],
                "TAWSS_p95": tawss_summary["p95"],
                "TAWSS_risk_area_ratio": tawss_summary["risk_area_ratio"],
                "OSI_mean": osi_summary["mean"],
                "OSI_max": osi_summary["max"],
                "OSI_p95": osi_summary["p95"],
                "RRT_mean": rrt_summary["mean"],
                "RRT_max": rrt_summary["max"],
                "RRT_p95": rrt_summary["p95"],
            }
        )
    return rows


def build_risk_feature_rows(case_rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    # 任务 C 需要的是“病例级特征表”，所以这里把任务 B 汇总结果进一步整理成 tabular 形式。
    features: List[Dict[str, object]] = []
    for row in case_rows:
        features.append(
            {
                "patient_id": row["patient_id"],
                "phase": row["phase"],
                "source": row["source"],
                "region_name": row["region_name"],
                "TAWSS_pre": row["TAWSS_mean"],
                "OSI_pre": row["OSI_mean"],
                "RRT_post": row["RRT_mean"],
                "TAWSS_region_p95": row["TAWSS_p95"],
                "OSI_region_p95": row["OSI_p95"],
                "RRT_region_p95": row["RRT_p95"],
            }
        )
    return features


def build_per_node_rows(
    *,
    samples: Sequence[HemoSample],
    source: str,
    model_name: str,
    split_version: str,
) -> List[Dict[str, object]]:
    # per-node 表主要服务点级一致性分析、热图和后续 Bland-Altman / scatter 等可视化。
    rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], List[HemoSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.patient_id, sample.phase), []).append(sample)

    for (patient_id, phase), case_samples in sorted(grouped.items()):
        metrics = compute_cycle_metrics(case_samples)
        ordered_samples = sorted(case_samples, key=lambda item: item.time_step)
        wall_indices = torch.nonzero(ordered_samples[0].wall_mask.bool(), as_tuple=False).flatten()

        for time_index, sample in enumerate(ordered_samples):
            proxy = compute_wss(
                sample.y_field,
                sample.wall_mask.bool(),
                positions=sample.positions,
                edge_index=sample.edge_index,
            )
            for local_idx, node_idx in enumerate(wall_indices.tolist()):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "phase": phase,
                        "time_step": sample.time_step,
                        "node_index": node_idx,
                        "source": source,
                        "model_name": model_name,
                        "split_version": split_version,
                        "WSS": proxy["wss"][local_idx].item(),
                        "TAWSS": metrics["tawss"][local_idx].item(),
                        "OSI": metrics["osi"][local_idx].item(),
                        "RRT": metrics["rrt"][local_idx].item(),
                        "time_rank": time_index,
                    }
                )
    return rows
