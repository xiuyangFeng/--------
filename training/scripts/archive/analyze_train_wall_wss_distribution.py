"""
训练集壁面 WSS 标签分布与 3σ 异常比例（可分享、可复用）

用途（对齐 `docs/01-任务/任务A/计划讨论.md` 动作 1）：
  - 在「仅 p+WSS」等配置下，用训练集**真值** `y_wss` 在 `is_wall=1` 上的边缘分布；
  - 报告各维直方图 / 经验 CDF，以及 |v-μ|>3σ 的壁面点比例（μ、σ 在该维全体壁面点上一阶估计）。

说明：
  - 本脚本**不读取任何训练 run 或 checkpoint**；与 WSSP-05 是否跑完**无依赖**，只要
    `data_root` + `graphs_subdir` 下已有含 `y_wss` 的图即可。计划里放在 WSSP-05 之后
    仅表示流程上先于 WSSP-06 解读，并非技术前置条件。

示例（仓库根目录、GNN 环境）：
  python -m training.scripts.analyze_train_wall_wss_distribution \\
    --split-file training/splits/split_AG_v1.json \\
    --data-root data_new/AG \\
    --graphs-subdir processed/graphs \\
    --output-dir outputs/field/diagnostics/wall_wss_train_dist_split_AG_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from pipeline.config import NODE_FEATURE_NAMES, WSS_TARGET_NAMES
from training.core.splits import SplitSpec


def _load_graph(path: Path) -> object:
    """与 `pipeline.dataset.load_graph_data` 一致，避免本脚本强依赖 torch_geometric 的模块导入链。"""
    return torch.load(path, weights_only=False)


def _iter_train_graph_paths(
    data_root: Path, graphs_subdir: str, train_cases: List[str]
) -> List[Path]:
    files: List[Path] = []
    for case in train_cases:
        case_dir = data_root / case / graphs_subdir
        if not case_dir.is_dir():
            continue
        files.extend(sorted(case_dir.glob("*.pt")))
    return files


def _collect_wall_wss(
    paths: List[Path], is_wall_index: int, max_files: Optional[int] = None
) -> Tuple[np.ndarray, int, int]:
    """返回堆叠后的壁面 y_wss (M,4)、含 y_wss 的图数、因缺 y_wss 跳过的图数。"""
    if max_files is not None:
        paths = paths[: max(0, int(max_files))]
    chunks: List[np.ndarray] = []
    skipped = 0
    for p in paths:
        data = _load_graph(p)
        if not hasattr(data, "y_wss") or data.y_wss is None:
            skipped += 1
            continue
        y = data.y_wss.detach().cpu().numpy()
        x = data.x.detach().cpu().numpy()
        if y.ndim != 2 or y.shape[1] != len(WSS_TARGET_NAMES):
            raise ValueError(
                f"{p}: y_wss 形状异常 {y.shape}，预期 [N, {len(WSS_TARGET_NAMES)}]"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{p}: x 行数 {x.shape[0]} 与 y_wss 行数 {y.shape[0]} 不一致")
        wall = x[:, is_wall_index] > 0.5
        if not np.any(wall):
            continue
        chunks.append(y[wall].astype(np.float64))
    if not chunks:
        raise RuntimeError("未采到任何壁面 WSS 点：请检查图是否含 y_wss、壁面点是否存在。")
    stacked = np.vstack(chunks)
    return stacked, len(paths) - skipped, skipped


def _per_dim_stats(values_1d: np.ndarray) -> Dict[str, float]:
    v = values_1d.astype(np.float64)
    mu = float(np.mean(v))
    sig = float(np.std(v, ddof=0))
    sorted_v = np.sort(v)
    n = v.size
    return {
        "n": int(n),
        "mean": mu,
        "std": sig,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p01": float(sorted_v[max(0, int(0.01 * (n - 1)))]),
        "p50": float(np.median(v)),
        "p99": float(sorted_v[min(n - 1, int(0.99 * (n - 1)))]),
    }


def _frac_beyond_3sigma_marginal(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    X: (M, D)，对每一维独立用全体壁面点估计 μ,σ，再标出该维上 |v-μ|>3σ 的 mask。
    返回 per_dim 比例 (D,) 以及「任一新维度超出」的壁面点比例。
    """
    m, d = X.shape
    out_dims = np.zeros(d, dtype=np.float64)
    any_dim = np.zeros(m, dtype=bool)
    for j in range(d):
        col = X[:, j]
        mu = col.mean()
        sig = col.std(ddof=0)
        if sig <= 0.0 or not np.isfinite(sig):
            bad = np.abs(col - mu) > 0.0
        else:
            bad = np.abs(col - mu) > 3.0 * sig
        out_dims[j] = float(bad.mean()) if m else 0.0
        any_dim |= bad
    return out_dims, float(any_dim.mean()) if m else 0.0


def _maybe_plot(
    X: np.ndarray,
    out_dir: Path,
    report: Dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    d = X.shape[1]
    fig, axes = plt.subplots(2, d, figsize=(4.0 * d, 5.0), squeeze=False)
    for j, name in enumerate(WSS_TARGET_NAMES):
        col = X[:, j]
        mu, sig = float(col.mean()), float(col.std(ddof=0))
        axh = axes[0, j]
        axh.hist(col, bins=80, density=True, alpha=0.75, color="steelblue", edgecolor="none")
        if sig > 0 and np.isfinite(sig):
            for z in (-3, 3):
                axh.axvline(mu + z * sig, color="crimson", linestyle="--", linewidth=1.0)
        axh.set_title(f"{name} 直方图 (μ={mu:.4g}, σ={sig:.4g})")
        axh.set_xlabel("值")
        axh.set_ylabel("密度")

        axc = axes[1, j]
        sorted_v = np.sort(col)
        ycdf = (np.arange(1, len(sorted_v) + 1)) / len(sorted_v)
        axc.plot(sorted_v, ycdf, color="darkgreen")
        if sig > 0 and np.isfinite(sig):
            for z in (-3, 3):
                axc.axvline(mu + z * sig, color="crimson", linestyle="--", linewidth=1.0)
        axc.set_title(f"{name} 经验 CDF")
        axc.set_xlabel("值")
        axc.set_ylabel("CDF")
        axc.set_ylim(0, 1.02)

    fig.suptitle(
        f"{report['split_version']} train 壁面 WSS | "
        f"图={report['n_graphs_read']}, 壁面点={report['n_wall_points']}"
    )
    fig.tight_layout()
    fig_path = out_dir / "wall_wss_hist_cdf.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="训练集壁面 y_wss 分布与 3σ 异常值比例（读原始图，不依赖训练 run）"
    )
    p.add_argument(
        "--split-file",
        type=str,
        default="training/splits/split_AG_v1.json",
        help="含 train_cases 的 split JSON",
    )
    p.add_argument("--data-root", type=str, default="data_new/AG", help="病例根目录")
    p.add_argument(
        "--graphs-subdir",
        type=str,
        default="processed/graphs",
        help="每病例下图目录（相对病例路径）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="保存 summary.json 与图；默认 outputs/field/diagnostics/wall_wss_train_<split_version>",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="不尝试生成 PNG（仅打印与写 JSON）",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="仅处理前 N 个图文件（0 表示不限制，用于大库时试跑/调试）",
    )
    args = p.parse_args()

    split = SplitSpec.from_json(args.split_file)
    data_root = Path(args.data_root)
    is_wall_i = NODE_FEATURE_NAMES.index("is_wall")

    paths = _iter_train_graph_paths(data_root, args.graphs_subdir, split.train_cases)
    if not paths:
        raise SystemExit(
            f"未找到任何图：data_root={data_root} graphs_subdir={args.graphs_subdir} "
            f"与 train_cases 是否匹配？"
        )

    max_f = args.max_files if args.max_files and args.max_files > 0 else None
    X, n_ok, n_skip = _collect_wall_wss(paths, is_wall_i, max_files=max_f)
    out_dims, frac_any = _frac_beyond_3sigma_marginal(X)

    per_dim: Dict[str, Dict[str, float]] = {}
    for j, wname in enumerate(WSS_TARGET_NAMES):
        st = _per_dim_stats(X[:, j])
        st["frac_beyond_3sigma_marginal"] = float(out_dims[j])
        per_dim[wname] = st

    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        out_path = (
            Path("outputs/field/diagnostics")
            / f"wall_wss_train_{split.split_version.replace('.', '_')}"
        )
    out_path.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "split_file": str(Path(args.split_file).resolve()),
        "split_version": split.split_version,
        "subset": "train",
        "data_root": str(data_root.resolve()),
        "graphs_subdir": args.graphs_subdir,
        "n_train_cases": len(split.train_cases),
        "n_graph_files": len(paths),
        "max_files_applied": max_f,
        "n_graphs_read": n_ok,
        "n_graphs_skipped_no_y_wss": n_skip,
        "n_wall_points": int(X.shape[0]),
        "wss_target_names": list(WSS_TARGET_NAMES),
        "frac_train_wall_points_any_dim_beyond_3sigma_marginal": frac_any,
        "per_dimension": per_dim,
        "notes": (
            "frac_beyond_3sigma_marginal: 各维用全体训练壁面点估计 μ,σ 后，"
            "该维上 |v-μ|>3σ 的壁面点比例。any_dim: 至少一维超 3σ 的壁面点比例。"
        ),
    }

    json_path = out_path / "wall_wss_train_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已写入: {json_path.resolve()}")

    if not args.no_plot:
        _maybe_plot(X, out_path, report)
        p_png = out_path / "wall_wss_hist_cdf.png"
        if p_png.is_file():
            print(f"已生成图: {p_png.resolve()}")


if __name__ == "__main__":
    main()
