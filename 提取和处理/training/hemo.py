from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class HemoSample:
    # HemoSample 是任务 B 的最小统一输入单元：
    # 一条样本 = 一个病例在一个时间步上的壁面相关场数据。
    sample_id: str
    patient_id: str
    phase: str
    time_step: int
    source: str
    model_name: str
    split_version: str
    wall_mask: torch.Tensor
    y_field: torch.Tensor


def parse_sample_id(sample_id: str) -> Tuple[str, str, int]:
    # 当前先约定 sample_id 中编码 patient / phase / time_step。
    # 如果后续 predict_field 输出更正式的 metadata，这里应该优先改成读取显式字段。
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


def compute_wss_proxy(y_field: torch.Tensor, wall_mask: torch.Tensor, epsilon: float = 1e-8) -> Dict[str, torch.Tensor]:
    # 这里不是论文最终版 WSS。
    # 目前只是用壁面速度模长构造一个 proxy，把任务 B 的输入输出链路先打通。
    wall_velocity = y_field[:, :3][wall_mask]
    if wall_velocity.numel() == 0:
        empty = y_field.new_zeros((0,))
        empty_vec = y_field.new_zeros((0, 3))
        return {"wss": empty, "wss_vector": empty_vec}
    wss = wall_velocity.norm(dim=1)
    denom = wss.unsqueeze(1).clamp_min(epsilon)
    return {
        "wss": wss,
        "wss_vector": wall_velocity / denom * wss.unsqueeze(1),
    }


def compute_cycle_metrics(samples: Sequence[HemoSample], epsilon: float = 1e-8) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("样本序列不能为空")

    wall_mask = samples[0].wall_mask.bool()
    wss_stack: List[torch.Tensor] = []
    wss_vector_stack: List[torch.Tensor] = []
    for sample in sorted(samples, key=lambda item: item.time_step):
        proxy = compute_wss_proxy(sample.y_field, wall_mask=sample.wall_mask.bool(), epsilon=epsilon)
        wss_stack.append(proxy["wss"])
        wss_vector_stack.append(proxy["wss_vector"])

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
            proxy = compute_wss_proxy(sample.y_field, sample.wall_mask.bool())
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
