"""典型病例空间三联图（Figure A2）

从单个 ``predict_field.py`` 导出的 ``*.pt`` 读取节点坐标与流场，生成
``3×3`` 面板：行为「全局 / 近壁 / 高曲率」子区域，列为 ``CFD | Prediction | |Error|``，
默认使用 XY 平面散点投影（论文规划见 ``任务A论文可视化与指标建议.md`` §6.2 / §7.3）。

用法示例
--------
# 使用某 run 的 manifest，指定样本 ID，输出到 plots/
python -m training.scripts.plot_taskA_case_panel \\
    --manifest outputs/field/field_transformer_coord_t_bc_geom_wall_split_AG_v1_seed1_20260322_064925/predictions_test/manifest.json \\
    --sample-id result_features_merged-1120 \\
    --variable vel_mag \\
    --output outputs/field/plots/fig_A2_case_panel_result_features_merged-1120.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pipeline.config import FEATURE_INDICES, NODE_FEATURE_NAMES

from ..analysis.visualization import plot_case_field_panel
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


def _build_masks(x: np.ndarray, wall_mask: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
    n = x.shape[0]
    global_mask = np.ones(n, dtype=bool)

    wm = np.asarray(wall_mask).reshape(-1)
    if wm.shape[0] != n:
        raise ValueError("wall_mask 长度与节点数不一致")
    near_wall = wm.astype(bool)

    lo, hi = FEATURE_INDICES["geom_scalar"]
    curv = x[:, lo:hi][:, 2]
    thr = float(np.percentile(curv, 75.0))
    high_curv = curv >= thr

    return (
        [global_mask, near_wall, high_curv],
        ["Global (all nodes)", "Near wall", "High curvature (top 25%)"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="任务 A Figure A2 典型病例空间面板")
    parser.add_argument("--manifest", required=True, help="predict_field.py 导出的 manifest.json")
    parser.add_argument("--sample-id", required=True, help="manifest items 中的 sample_id")
    parser.add_argument(
        "--variable",
        default="vel_mag",
        choices=list(_VALID_VARIABLES),
        help="展示的物理量（默认 vel_mag）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出 PNG 路径，默认 <manifest 父目录>/../../plots/fig_A2_case_panel_<sample_id>.png",
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
    pred_rel: str | None = None
    for it in items:
        if str(it.get("sample_id")) == args.sample_id:
            pred_rel = str(it["prediction_path"])
            break
    if pred_rel is None:
        raise SystemExit(f"manifest 中未找到 sample_id={args.sample_id!r}")

    pred_path = Path(pred_rel)
    if not pred_path.is_file():
        pred_path = (manifest_path.parent / pred_rel).resolve()

    payload = load_prediction_payload(pred_path)
    x = payload["x"].detach().cpu().numpy()
    y_true = payload["y_true"].detach().cpu().numpy()
    y_pred = payload["y_pred"].detach().cpu().numpy()
    wall_mask = payload["wall_mask"].detach().cpu().numpy()

    if x.shape[1] != len(NODE_FEATURE_NAMES):
        raise SystemExit(
            f"节点特征维数 {x.shape[1]} 与 NODE_FEATURE_NAMES={len(NODE_FEATURE_NAMES)} 不一致"
        )

    pos = x[:, 0:3]
    cfd, pred = _field_values(y_true, y_pred, args.variable)
    masks, row_titles = _build_masks(x, wall_mask)

    plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    axes_plane = plane_map[args.plane]

    if args.output:
        out = Path(args.output).resolve()
    else:
        plots_dir = manifest_path.parent.parent.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"fig_A2_case_panel_{args.sample_id}.png"

    try:
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
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖。"
        ) from exc

    print(out)


if __name__ == "__main__":
    main()
