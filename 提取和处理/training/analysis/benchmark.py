"""任务 A 模型的推理效率基准工具。

用于统一测量单 snapshot / 单病例推理时间、峰值显存和参数量，
支撑论文中的效率对比表。
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    sample_data: Data,
    device: torch.device,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Dict[str, float]:
    """测量单个 snapshot 的推理时延和峰值显存。"""
    model.eval()
    model = model.to(device)
    data = sample_data.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(n_warmup):
        _ = model(data)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model(data)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000)

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    import numpy as np
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "peak_memory_mb": peak_mem,
        "n_nodes": int(data.x.size(0)),
        "n_edges": int(data.edge_index.size(1)),
    }


def benchmark_full_case(
    model: nn.Module,
    case_graphs: list,
    device: torch.device,
) -> Dict[str, float]:
    """测量单个病例全时序的推理开销。"""
    model.eval()
    model = model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for g in case_graphs:
        data = g.to(device)
        _ = model(data)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "total_case_ms": total_ms,
        "n_snapshots": len(case_graphs),
        "per_snapshot_ms": total_ms / max(1, len(case_graphs)),
        "peak_memory_mb": peak_mem,
    }


def build_efficiency_table(
    model: nn.Module,
    sample_data: Data,
    device: torch.device,
    cfd_time_hours: Optional[float] = None,
) -> Dict[str, object]:
    """生成单个模型的一行效率评估结果。"""
    params = count_parameters(model)
    latency = benchmark_inference(model, sample_data, device)

    result = {**params, **latency}
    if cfd_time_hours is not None:
        ai_time_hours = latency["mean_ms"] / 1000 / 3600
        # 这里按常规定义使用 CFD / AI，值越大说明 AI 相对 CFD 越快。
        result["speedup_vs_cfd"] = cfd_time_hours / max(ai_time_hours, 1e-12)

    return result
