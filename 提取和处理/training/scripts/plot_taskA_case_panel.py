"""典型病例空间三联图（Figure A2）

从单个 ``predict_field.py`` 导出的 ``*.pt`` 读取节点坐标与流场，生成
``3×3`` 面板：行为「全局 / 壁面 / 高曲率」子区域，列为 ``CFD | Prediction | |Error|``，
默认使用 XY 平面散点投影（论文规划见 ``任务A论文可视化与指标建议.md`` §6.2 / §7.3）。

额外模式：

- ``--mode vel_mag_pressure``：6×3，|v| 与 p 各三行子区域。
- ``--mode vector``：3×3，投影平面内两分量 quiver（CFD / 预测 / 面内矢量误差）。
- ``--mode both``：一次生成上述两种图（``--output`` 为基名，自动加 ``_vel_p`` / ``_vec``）。

区域 mask 优先用 ``graph_path`` 读回完整节点特征（与 ``regional_eval`` / Fig A5 几何口径一致）。

用法示例
--------
# 标量单变量（默认）
python -m training.scripts.plot_taskA_case_panel \\
    --manifest outputs/field/.../predictions_test/manifest.json \\
    --sample-id result_features_merged-1120 \\
    --case-name slow/GUO_XI_JIANG \\
    --variable vel_mag \\
    --output outputs/field/plots/case_panels/fig_A2_vel_mag.png

# |v|+p 与矢量图一次出齐
python -m training.scripts.plot_taskA_case_panel \\
    --manifest .../manifest.json \\
    --sample-id result_features_merged-1120 \\
    --case-name slow/GUO_XI_JIANG \\
    --mode both \\
    --output outputs/field/plots/case_panels/fig_A2_case_GUO_1120.png
"""
from __future__ import annotations

import argparse

from ..core.field_plot_paths import CAT_CASE_PANELS
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pipeline.config import FEATURE_INDICES, NODE_FEATURE_NAMES

from ..analysis.regional_eval import load_node_features_for_region_masks
from ..analysis.visualization import (
    plot_case_field_panel,
    plot_case_vel_mag_pressure_panel,
    plot_case_velocity_quiver_panel,
)
from ._figure_utils import load_manifest, load_prediction_payload


_VALID_VARIABLES = ("u", "v", "w", "p", "vel_mag")
_VAR_DISPLAY = {
    "u": "u (m/s)",
    "v": "v (m/s)",
    "w": "w (m/s)",
    "p": "p (Pa)",
    "vel_mag": "|v| (m/s)",
}


def _field_values(
    y_true: np.ndarray, y_pred: np.ndarray, variable: str
) -> Tuple[np.ndarray, np.ndarray]:
    if variable == "vel_mag":
        cfd = np.linalg.norm(y_true[:, :3], axis=1)
        pred = np.linalg.norm(y_pred[:, :3], axis=1)
        return cfd, pred
    col = {"u": 0, "v": 1, "w": 2, "p": 3}[variable]
    return y_true[:, col].copy(), y_pred[:, col].copy()


def _build_masks(x: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
    n = x.shape[0]
    global_mask = np.ones(n, dtype=bool)

    is_wall_idx = NODE_FEATURE_NAMES.index("is_wall")
    wall_nodes = x[:, is_wall_idx] > 0.5

    lo, hi = FEATURE_INDICES["geom_scalar"]
    curv = x[:, lo:hi][:, 2]
    thr = float(np.percentile(curv, 75.0))
    high_curv = curv >= thr

    return (
        [global_mask, wall_nodes, high_curv],
        ["Global (all nodes)", "Wall (is_wall)", "High curvature (top 25%)"],
    )


def _gnn_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_graph_path(payload: Dict[str, Any]) -> Dict[str, Any]:
    gp = payload.get("graph_path")
    if not gp:
        return payload
    path = Path(str(gp))
    if path.is_file():
        return payload
    cand = _gnn_project_root() / path
    if cand.is_file():
        out = dict(payload)
        out["graph_path"] = str(cand.resolve())
        return out
    return payload


def _load_geometry_numpy(payload: Dict[str, Any]) -> np.ndarray:
    import torch

    p2 = _resolve_graph_path(payload)
    x_t = load_node_features_for_region_masks(p2)
    if not isinstance(x_t, torch.Tensor):
        raise TypeError("load_node_features_for_region_masks 应返回 Tensor")
    return x_t.detach().cpu().numpy()


def _find_manifest_item(items: List[Any], sample_id: str, case_name: str | None) -> Dict[str, Any]:
    matches = [it for it in items if str(it.get("sample_id")) == sample_id]
    if not matches:
        raise SystemExit(f"manifest 中未找到 sample_id={sample_id!r}")
    if case_name is None:
        if len(matches) > 1:
            raise SystemExit(
                f"sample_id={sample_id!r} 对应多条记录，请追加 --case-name 指定病例，例如："
                f" {matches[0].get('case_name')!r} 等共 {len(matches)} 条"
            )
        return matches[0]
    for it in matches:
        if str(it.get("case_name")) == case_name:
            return it
    raise SystemExit(f"未找到 sample_id={sample_id!r} 且 case_name={case_name!r} 的条目")


def main() -> None:
    parser = argparse.ArgumentParser(description="任务 A Figure A2 典型病例空间面板")
    parser.add_argument("--manifest", required=True, help="predict_field.py 导出的 manifest.json")
    parser.add_argument("--sample-id", required=True, help="manifest items 中的 sample_id")
    parser.add_argument(
        "--case-name",
        default="",
        help="若同一 sample_id 在多病例重复出现，指定 case_name（如 slow/GUO_XI_JIANG）以唯一定位",
    )
    parser.add_argument(
        "--mode",
        default="scalar",
        choices=("scalar", "vel_mag_pressure", "vector", "both"),
        help="scalar：单变量 3×3；vel_mag_pressure：|v|+p 6×3；vector：面内速度 quiver；both：后两者各出一张",
    )
    parser.add_argument(
        "--variable",
        default="vel_mag",
        choices=list(_VALID_VARIABLES),
        help="mode=scalar 时展示的物理量（默认 vel_mag）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出 PNG 路径，默认 <runs-root>/plots/case_panels/fig_A2_case_panel_<sample_id>.png",
    )
    parser.add_argument("--title", default="Figure A2 Case Panel", help="总标题")
    parser.add_argument("--max-points", type=int, default=12_000, help="每个子图最多绘制的点数")
    parser.add_argument("--seed", type=int, default=42, help="子采样随机种子")
    parser.add_argument(
        "--plane",
        default="xy",
        choices=("xy", "xz", "yz"),
        help="投影平面（默认 xy）",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("manifest.json 缺少 items 列表")

    case_name = args.case_name.strip() or None
    item = _find_manifest_item(items, args.sample_id, case_name)
    pred_rel = str(item["prediction_path"])

    pred_path = Path(pred_rel)
    if not pred_path.is_file():
        pred_path = (manifest_path.parent / pred_rel).resolve()

    payload = load_prediction_payload(pred_path)
    if not isinstance(payload, dict):
        raise SystemExit("预测文件结构非法：应为 dict")
    payload = _resolve_graph_path(payload)

    x_geo = _load_geometry_numpy(payload)
    if x_geo.shape[1] != len(NODE_FEATURE_NAMES):
        raise SystemExit(
            f"节点特征维数 {x_geo.shape[1]} 与 NODE_FEATURE_NAMES={len(NODE_FEATURE_NAMES)} 不一致"
        )

    y_true = payload["y_true"].detach().cpu().numpy()
    y_pred = payload["y_pred"].detach().cpu().numpy()
    if x_geo.shape[0] != y_true.shape[0]:
        raise SystemExit("几何 x 与 y_true 节点数不一致，请检查 graph_path 与预测文件是否匹配")

    pos = x_geo[:, 0:3]
    masks, row_titles = _build_masks(x_geo)

    plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    axes_plane = plane_map[args.plane]

    plots_dir = manifest_path.parent.parent.parent / "plots" / CAT_CASE_PANELS
    plots_dir.mkdir(parents=True, exist_ok=True)

    def default_scalar_path() -> Path:
        if args.output:
            return Path(args.output).resolve()
        return plots_dir / f"fig_A2_case_panel_{args.sample_id}.png"

    written: List[Path] = []

    try:
        if args.mode == "scalar":
            out = default_scalar_path()
            cfd, pred = _field_values(y_true, y_pred, args.variable)
            plot_case_field_panel(
                pos,
                cfd,
                pred,
                masks,
                row_titles,
                variable_name=_VAR_DISPLAY[args.variable],
                save_path=str(out),
                title=args.title,
                max_points=args.max_points,
                seed=args.seed,
                axes_plane=axes_plane,
            )
            written.append(out)

        elif args.mode == "vel_mag_pressure":
            if not args.output:
                raise SystemExit("mode=vel_mag_pressure 时请用 --output 指定 PNG 路径")
            out = Path(args.output).resolve()
            plot_case_vel_mag_pressure_panel(
                pos,
                y_true,
                y_pred,
                masks,
                row_titles,
                save_path=str(out),
                title=f"{args.title} — |v| & p",
                max_points=args.max_points,
                seed=args.seed,
                axes_plane=axes_plane,
            )
            written.append(out)

        elif args.mode == "vector":
            if not args.output:
                raise SystemExit("mode=vector 时请用 --output 指定 PNG 路径")
            out = Path(args.output).resolve()
            plot_case_velocity_quiver_panel(
                pos,
                y_true,
                y_pred,
                masks,
                row_titles,
                save_path=str(out),
                title=args.title,
                max_points=min(args.max_points, 6000),
                seed=args.seed,
                axes_plane=axes_plane,
            )
            written.append(out)

        else:  # both
            if not args.output:
                raise SystemExit("mode=both 时请用 --output 指定基路径（将生成 _vel_p 与 _vec 两个文件）")
            base = Path(args.output).resolve()
            stem, suf = base.stem, base.suffix or ".png"
            out_vp = base.with_name(f"{stem}_vel_p{suf}")
            out_vec = base.with_name(f"{stem}_vec{suf}")
            plot_case_vel_mag_pressure_panel(
                pos,
                y_true,
                y_pred,
                masks,
                row_titles,
                save_path=str(out_vp),
                title=f"{args.title} — |v| & p",
                max_points=args.max_points,
                seed=args.seed,
                axes_plane=axes_plane,
            )
            plot_case_velocity_quiver_panel(
                pos,
                y_true,
                y_pred,
                masks,
                row_titles,
                save_path=str(out_vec),
                title=args.title,
                max_points=min(args.max_points, 6000),
                seed=args.seed,
                axes_plane=axes_plane,
            )
            written.extend([out_vp, out_vec])

    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖。"
        ) from exc

    for p in written:
        print(p)


if __name__ == "__main__":
    main()
