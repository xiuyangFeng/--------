from __future__ import annotations

"""V1 PINN 数据情况评估（只读诊断）。

针对 split_AG_v1 post-denylist 训练集，评估以下几点，为 PINN 损失模块设计提供依据：

1. 反归一化后壁面/内部节点速度模长分布 —— 确认壁面是否≈0（no-slip 目标设计）。
2. u/v/w/p 各通道 z-score mean/std —— 反归一化所需。
3. 逐病例坐标 scale_factor（mm）分布与中位数 —— autograd 还原物理尺度所需。
4. 壁面/内部节点占比。
5. （可选）GT 速度的连续性残差量级闭合审计 —— 评估 PDE 在离散点云上是否近似闭合。

用法：
  python -m training.scripts.diagnose_pinn_data \
      --split training/splits/split_AG_v1.json \
      --data-root data_new/AG

不训练、不写图资产，仅读取并输出 json + 控制台报告。
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from pipeline.config import NODE_FEATURE_NAMES
from pipeline.dataset import load_graph_data
from training.core.denylist import filter_case_names, skipped_case_names
from training.core.splits import SplitSpec

_IS_WALL_IDX = NODE_FEATURE_NAMES.index("is_wall")
_COORD_IDX = [NODE_FEATURE_NAMES.index(c) for c in ("x", "y", "z")]
_FIELD_NAMES = ["u", "v", "w", "p"]


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    """常用分位数摘要；空数组返回 NaN。"""
    if values.size == 0:
        return {k: float("nan") for k in ("min", "mean", "p50", "p95", "p99", "max")}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _load_field_stats(norm_params_path: Path) -> Dict[str, Dict[str, float]]:
    """从 normalization_params_global.json 读取 u/v/w/p 的 mean/std。"""
    with open(norm_params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    stats = params.get("statistics", {})
    out: Dict[str, Dict[str, float]] = {}
    for name in _FIELD_NAMES:
        if name not in stats:
            raise KeyError(f"normalization_params_global.json 缺少字段 {name} 的统计量")
        out[name] = {"mean": float(stats[name]["mean"]), "std": float(stats[name]["std"])}
    return out


def _denormalize_velocity(
    y: torch.Tensor, field_stats: Dict[str, Dict[str, float]]
) -> torch.Tensor:
    """把 z-score 速度标签 [N,3] 还原到物理单位 m/s。"""
    vel = y[:, :3].clone().float()
    for j, name in enumerate(("u", "v", "w")):
        vel[:, j] = vel[:, j] * field_stats[name]["std"] + field_stats[name]["mean"]
    return vel


def _case_scale_factor(case_dir: Path) -> Optional[float]:
    """读取病例坐标归一化 scale_factor（mm 量级）。"""
    p = case_dir / "processed" / "coord_normalized" / "transform_params.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            params = json.load(f)
        sf = float(params.get("scale_factor", 0.0))
        return sf if sf > 1e-6 else None
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _continuity_residual_audit(
    coords_norm: torch.Tensor,
    vel_phys: torch.Tensor,
    is_wall: torch.Tensor,
    scale_factor_mm: float,
    k: int,
    sample_nodes: int,
    length_unit_to_meter: float = 1e-3,
) -> np.ndarray:
    """对内部节点子集，用 kNN 局部线性最小二乘估计速度梯度并算连续性残差 |div(v)|。

    coords 在归一化对齐系（旋转保距），物理距离 = norm 距离 × scale_factor(mm) × m/mm。
    返回所采样内部节点的 |∂u/∂x+∂v/∂y+∂w/∂z| 数组（单位 1/s）。
    """
    interior_idx = (~is_wall.bool()).nonzero(as_tuple=True)[0]
    if interior_idx.numel() < k + 1:
        return np.array([], dtype=np.float64)

    # 物理尺度（米）下的坐标，用于得到 1/s 量纲的残差。
    coords_m = coords_norm.float() * (scale_factor_mm * length_unit_to_meter)

    n_pick = min(sample_nodes, int(interior_idx.numel()))
    perm = torch.randperm(int(interior_idx.numel()))[:n_pick]
    query_idx = interior_idx[perm]

    residuals: List[float] = []
    coords_all = coords_m
    for qi in query_idx.tolist():
        d = (coords_all - coords_all[qi]).pow(2).sum(dim=1)
        d[qi] = float("inf")
        kk = min(k, int(d.numel()) - 1)
        nn = torch.topk(d, kk, largest=False).indices
        dx = coords_all[nn] - coords_all[qi]  # [k,3]
        dv = vel_phys[nn] - vel_phys[qi]      # [k,3]
        # 解 dx @ G ≈ dv，G[3x3] 为速度雅可比；div = trace(G)。
        try:
            sol = torch.linalg.lstsq(dx, dv).solution  # [3,3]
        except Exception:
            continue
        div = sol[0, 0] + sol[1, 1] + sol[2, 2]
        if torch.isfinite(div):
            residuals.append(float(abs(div.item())))
    return np.asarray(residuals, dtype=np.float64)


def diagnose(
    split_file: str,
    data_root: str,
    graphs_subdir: str,
    subset: str,
    max_cases: Optional[int],
    max_graphs_per_case: int,
    per_graph_node_cap: int,
    closure_audit: bool,
    closure_k: int,
    closure_sample_nodes: int,
    closure_max_graphs: int,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = Path(data_root)
    split = SplitSpec.from_json(split_file)
    subset_map = {
        "train": split.train_cases,
        "val": split.val_cases,
        "test": split.test_cases,
    }
    if subset not in subset_map:
        raise ValueError(f"未知 subset: {subset}")
    raw_cases = subset_map[subset]
    cases = filter_case_names(raw_cases, data_root)
    denied = sorted(skipped_case_names(raw_cases, data_root))
    if max_cases is not None:
        cases = cases[:max_cases]

    field_stats = _load_field_stats(root / "normalization_params_global.json")

    wall_speed_samples: List[np.ndarray] = []
    interior_speed_samples: List[np.ndarray] = []
    n_wall_total = 0
    n_interior_total = 0
    scale_factors: List[float] = []
    missing_scale_cases: List[str] = []
    n_graphs_used = 0
    closure_residuals: List[np.ndarray] = []
    n_closure_graphs = 0

    rng = np.random.default_rng(seed)

    for case_name in cases:
        case_dir = root / case_name
        sf = _case_scale_factor(case_dir)
        if sf is None:
            missing_scale_cases.append(case_name)
        else:
            scale_factors.append(sf)

        graphs_dir = case_dir / graphs_subdir
        if not graphs_dir.exists():
            continue
        pt_files = sorted(graphs_dir.glob("*.pt"))[:max_graphs_per_case]
        for gi, pt in enumerate(pt_files):
            try:
                data = load_graph_data(pt)
            except Exception:
                continue
            if not hasattr(data, "y") or data.y is None:
                continue
            is_wall = data.x[:, _IS_WALL_IDX]
            vel_phys = _denormalize_velocity(data.y, field_stats)
            speed = vel_phys.norm(dim=1).detach().cpu().numpy()
            wall_mask = is_wall.bool().cpu().numpy()
            n_wall = int(wall_mask.sum())
            n_int = int((~wall_mask).sum())
            n_wall_total += n_wall
            n_interior_total += n_int
            n_graphs_used += 1

            w_sp = speed[wall_mask]
            i_sp = speed[~wall_mask]
            if w_sp.size > per_graph_node_cap:
                w_sp = rng.choice(w_sp, per_graph_node_cap, replace=False)
            if i_sp.size > per_graph_node_cap:
                i_sp = rng.choice(i_sp, per_graph_node_cap, replace=False)
            wall_speed_samples.append(w_sp)
            interior_speed_samples.append(i_sp)

            if closure_audit and n_closure_graphs < closure_max_graphs and sf is not None:
                coords_norm = data.x[:, _COORD_IDX]
                res = _continuity_residual_audit(
                    coords_norm,
                    vel_phys,
                    is_wall,
                    scale_factor_mm=sf,
                    k=closure_k,
                    sample_nodes=closure_sample_nodes,
                )
                if res.size > 0:
                    closure_residuals.append(res)
                    n_closure_graphs += 1

    wall_speeds = np.concatenate(wall_speed_samples) if wall_speed_samples else np.array([])
    interior_speeds = (
        np.concatenate(interior_speed_samples) if interior_speed_samples else np.array([])
    )

    n_total_nodes = n_wall_total + n_interior_total
    wall_frac = n_wall_total / n_total_nodes if n_total_nodes else float("nan")

    # 物理 no-slip 在归一化空间对应的目标值（u_norm = -mean/std），供 loss 设计参考。
    noslip_norm_targets = {
        name: -field_stats[name]["mean"] / field_stats[name]["std"]
        for name in ("u", "v", "w")
    }

    closure_summary: Dict[str, object] = {"enabled": bool(closure_audit)}
    if closure_audit and closure_residuals:
        all_res = np.concatenate(closure_residuals)
        closure_summary.update(
            {
                "n_graphs_audited": n_closure_graphs,
                "n_nodes_audited": int(all_res.size),
                "abs_divergence_1_per_s": _percentiles(all_res),
                "note": "kNN 局部线性最小二乘估计 div(v)；非结构点云上仅作量级参考。",
            }
        )
    elif closure_audit:
        closure_summary["note"] = "无可用残差（病例缺 scale_factor 或节点不足）。"

    report: Dict[str, object] = {
        "split_file": split_file,
        "data_root": data_root,
        "subset": subset,
        "n_cases_requested": len(raw_cases),
        "n_cases_used": len(cases),
        "denylist_skipped": denied,
        "n_graphs_used": n_graphs_used,
        "field_zscore_stats": field_stats,
        "node_counts": {
            "n_wall_total": n_wall_total,
            "n_interior_total": n_interior_total,
            "wall_fraction": wall_frac,
        },
        "wall_velocity_phys_m_per_s": _percentiles(wall_speeds),
        "interior_velocity_phys_m_per_s": _percentiles(interior_speeds),
        "noslip_normalized_targets": noslip_norm_targets,
        "coord_scale_factor_mm": {
            "n_cases_with_scale": len(scale_factors),
            "missing_scale_cases": missing_scale_cases,
            **(
                {
                    "median": float(statistics.median(scale_factors)),
                    "min": float(min(scale_factors)),
                    "max": float(max(scale_factors)),
                    "mean": float(statistics.fmean(scale_factors)),
                }
                if scale_factors
                else {}
            ),
        },
        "continuity_closure_audit": closure_summary,
    }
    return report


def _print_report(report: Dict[str, object]) -> None:
    print("=" * 70)
    print("V1 PINN 数据情况评估报告")
    print("=" * 70)
    print(f"split        : {report['split_file']}")
    print(f"data_root    : {report['data_root']}  subset={report['subset']}")
    print(
        f"病例         : 请求 {report['n_cases_requested']} / 使用 {report['n_cases_used']} "
        f"/ denylist 跳过 {len(report['denylist_skipped'])}"
    )
    print(f"图样本       : {report['n_graphs_used']}")
    nc = report["node_counts"]
    print(
        f"节点占比     : wall={nc['n_wall_total']} interior={nc['n_interior_total']} "
        f"wall_fraction={nc['wall_fraction']:.4f}"
    )
    wv = report["wall_velocity_phys_m_per_s"]
    iv = report["interior_velocity_phys_m_per_s"]
    print("\n[壁面速度模长 |v| (m/s, 反归一化)]")
    print(f"  min={wv['min']:.3e} mean={wv['mean']:.3e} p50={wv['p50']:.3e} "
          f"p95={wv['p95']:.3e} p99={wv['p99']:.3e} max={wv['max']:.3e}")
    print("[内部速度模长 |v| (m/s, 反归一化)]")
    print(f"  min={iv['min']:.3e} mean={iv['mean']:.3e} p50={iv['p50']:.3e} "
          f"p95={iv['p95']:.3e} p99={iv['p99']:.3e} max={iv['max']:.3e}")
    nt = report["noslip_normalized_targets"]
    print("\n[no-slip 归一化空间目标 u_norm = -mean/std]")
    print(f"  u={nt['u']:.4f} v={nt['v']:.4f} w={nt['w']:.4f}")
    print("  说明: 现有 no_slip=wall*(u²+v²+w²) 把归一化速度逼向 0；物理 no-slip 对应上述非零目标。")
    sf = report["coord_scale_factor_mm"]
    print("\n[坐标 scale_factor (mm)]")
    if "median" in sf:
        print(f"  median={sf['median']:.4f} mean={sf['mean']:.4f} "
              f"min={sf['min']:.4f} max={sf['max']:.4f} (n={sf['n_cases_with_scale']})")
    if sf["missing_scale_cases"]:
        print(f"  缺 scale_factor 病例: {len(sf['missing_scale_cases'])}")
    ca = report["continuity_closure_audit"]
    print("\n[连续性闭合审计]")
    if ca.get("abs_divergence_1_per_s"):
        d = ca["abs_divergence_1_per_s"]
        print(f"  审计图数={ca['n_graphs_audited']} 节点数={ca['n_nodes_audited']}")
        print(f"  |div(v)| (1/s): mean={d['mean']:.3e} p50={d['p50']:.3e} "
              f"p95={d['p95']:.3e} max={d['max']:.3e}")
    else:
        print(f"  {ca.get('note', '未启用')}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 PINN 数据情况评估（只读）")
    parser.add_argument("--split", default="training/splits/split_AG_v1.json")
    parser.add_argument("--data-root", default="data_new/AG")
    parser.add_argument("--graphs-subdir", default="processed/graphs")
    parser.add_argument("--subset", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-graphs-per-case", type=int, default=8)
    parser.add_argument("--per-graph-node-cap", type=int, default=4000)
    parser.add_argument("--closure-audit", dest="closure_audit", action="store_true", default=True)
    parser.add_argument("--no-closure-audit", dest="closure_audit", action="store_false")
    parser.add_argument("--closure-k", type=int, default=16)
    parser.add_argument("--closure-sample-nodes", type=int, default=300)
    parser.add_argument("--closure-max-graphs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default=None, help="JSON 报告输出路径")
    args = parser.parse_args()

    report = diagnose(
        split_file=args.split,
        data_root=args.data_root,
        graphs_subdir=args.graphs_subdir,
        subset=args.subset,
        max_cases=args.max_cases,
        max_graphs_per_case=args.max_graphs_per_case,
        per_graph_node_cap=args.per_graph_node_cap,
        closure_audit=args.closure_audit,
        closure_k=args.closure_k,
        closure_sample_nodes=args.closure_sample_nodes,
        closure_max_graphs=args.closure_max_graphs,
        seed=args.seed,
    )

    _print_report(report)

    out_path = args.output
    if out_path is None:
        out_dir = Path("outputs/field/diagnostics")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pinn_data_diagnosis.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已写入: {out_path}")


if __name__ == "__main__":
    main()
