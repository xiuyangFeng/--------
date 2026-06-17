"""典型病例截面 CFD vs 预测对比图（腔内压力 + 壁面剪切应力）。

截面模式（``--slice-mode``）：
- ``axis``：坐标轴对齐平面（x/y/z 薄层切割）
- ``abscissa``：沿中心线 Abscissa 取法向截面（投影到局部切平面）

渲染模式（``--render-mode``）：
- ``scatter``：散点
- ``contour``：三角网格插值 + 等值面填充
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import FEATURE_INDICES, NODE_FEATURE_NAMES

from ..analysis.visualization import plot_case_cross_section_panel
from ..core.field_plot_paths import CAT_CASE_PANELS
from ._figure_utils import load_manifest, load_prediction_payload


def _gnn_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_norm_stats(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("statistics")
    if not isinstance(stats, dict):
        raise SystemExit(f"归一化文件缺少 statistics: {path}")
    return stats


def _denorm_zscore(values: np.ndarray, stats: Dict[str, Dict[str, float]], key: str) -> np.ndarray:
    entry = stats[key]
    return values * float(entry["std"]) + float(entry["mean"])


def _find_manifest_item(items: List[Any], sample_id: str, case_name: str | None) -> Dict[str, Any]:
    matches = [it for it in items if str(it.get("sample_id")) == sample_id]
    if not matches:
        raise SystemExit(f"manifest 中未找到 sample_id={sample_id!r}")
    if case_name is None:
        if len(matches) > 1:
            raise SystemExit(
                f"sample_id={sample_id!r} 对应多条记录，请追加 --case-name，例如 {matches[0].get('case_name')!r}"
            )
        return matches[0]
    for it in matches:
        if str(it.get("case_name")) == case_name:
            return it
    raise SystemExit(f"未找到 sample_id={sample_id!r} 且 case_name={case_name!r} 的条目")


def _auto_axis_slice(
    pos: np.ndarray,
    interior: np.ndarray,
    wall: np.ndarray,
    *,
    tol: float,
    prefer_axis: Optional[int] = None,
    n_grid: int = 60,
) -> Tuple[int, float, np.ndarray]:
    axes = [prefer_axis] if prefer_axis is not None else [0, 2, 1]
    best_score = -1.0
    best_axis, best_val = 0, 0.0
    best_mask = np.zeros(len(pos), dtype=bool)
    for axis in axes:
        if axis is None:
            continue
        coord = pos[:, axis]
        lo, hi = float(coord.min()), float(coord.max())
        margin = max(tol, 0.05 * (hi - lo))
        for val in np.linspace(lo + margin, hi - margin, n_grid):
            m = np.abs(coord - val) < tol
            ni = int((m & interior).sum())
            nw = int((m & wall).sum())
            if ni < 20 or nw < 20:
                continue
            others = [a for a in (0, 1, 2) if a != axis]
            span = float(np.prod([pos[m, a].max() - pos[m, a].min() for a in others]))
            score = min(ni, nw) * span
            if score > best_score:
                best_score = score
                best_axis = axis
                best_val = float(val)
                best_mask = m
    if best_score < 0:
        axis = axes[0]
        best_val = float(np.median(pos[:, axis]))
        best_mask = np.abs(pos[:, axis] - best_val) < tol
        best_axis = axis
    return best_axis, best_val, best_mask


def _auto_abscissa_slice(
    x: np.ndarray,
    interior: np.ndarray,
    wall: np.ndarray,
    *,
    tol: float,
    n_grid: int = 40,
) -> Tuple[float, np.ndarray]:
    abs_idx = NODE_FEATURE_NAMES.index("Abscissa")
    abs_v = x[:, abs_idx]
    lo, hi = float(abs_v.min()), float(abs_v.max())
    margin = max(tol, 0.03)
    best_score = -1.0
    best_val = 0.35
    best_mask = np.abs(abs_v - best_val) < tol
    for val in np.linspace(lo + margin, hi - margin, n_grid):
        m = np.abs(abs_v - val) < tol
        ni = int((m & interior).sum())
        nw = int((m & wall).sum())
        if ni < 15 or nw < 30:
            continue
        idx = np.where(m & wall)[0]
        pos = x[idx, 0:3]
        span = float(max(pos[:, 0].ptp(), pos[:, 1].ptp(), pos[:, 2].ptp()))
        score = min(ni, nw) / (span + 0.05)
        if score > best_score:
            best_score = score
            best_val = float(val)
            best_mask = m
    return best_val, best_mask


def _tangent_plane_coords(
    pos: np.ndarray,
    tangents: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, Tuple[str, str]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((len(pos), 2), dtype=float), ("u (m)", "v (m)")

    center = np.mean(pos[idx], axis=0)
    t = np.mean(tangents[idx], axis=0)
    t_norm = float(np.linalg.norm(t))
    if t_norm < 1e-8:
        t = np.array([0.0, 0.0, 1.0])
    else:
        t = t / t_norm

    ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t, ref)
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(t, u)
    v = v / (np.linalg.norm(v) + 1e-8)

    rel = pos - center
    coords = np.column_stack([rel @ u, rel @ v])
    return coords, ("local u (m)", "local v (m)")


def main() -> None:
    parser = argparse.ArgumentParser(description="任务 A 截面 CFD vs 预测对比图")
    parser.add_argument("--manifest", required=True, help="predict_field.py 导出的 manifest.json")
    parser.add_argument("--sample-id", required=True, help="manifest items 中的 sample_id")
    parser.add_argument("--case-name", default="", help="可选，唯一定位病例")
    parser.add_argument(
        "--norm-params",
        default="data_new/AG/normalization_params_global.json",
        help="z-score 反归一化参数（默认 AG 全局）",
    )
    parser.add_argument(
        "--slice-mode",
        default="axis",
        choices=("axis", "abscissa"),
        help="axis=坐标轴对齐截面；abscissa=中心线法向截面",
    )
    parser.add_argument(
        "--render-mode",
        default="scatter",
        choices=("scatter", "contour"),
        help="scatter=散点；contour=三角插值等值面",
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=-1,
        choices=(-1, 0, 1, 2),
        help="slice-mode=axis 时法向轴：0=x, 1=y, 2=z；-1 自动搜索",
    )
    parser.add_argument("--slice-value", type=float, default=None, help="axis 截面位置；默认自动")
    parser.add_argument("--slice-tol", type=float, default=0.03, help="axis 截面厚度（m）")
    parser.add_argument("--abscissa-value", type=float, default=None, help="abscissa 截面中心；默认自动")
    parser.add_argument("--abscissa-tol", type=float, default=0.015, help="abscissa 带宽")
    parser.add_argument(
        "--plane",
        default="auto",
        choices=("auto", "xy", "xz", "yz"),
        help="axis 模式下投影平面；auto 为法向轴的垂直平面",
    )
    parser.add_argument("--output", default="", help="输出 PNG；默认写入 run/plots/case_panels/")
    parser.add_argument("--title", default="", help="图标题；默认含病例与模型信息")
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
    x = payload["x"].detach().cpu().numpy()
    y_true = payload["y_true"].detach().cpu().numpy()
    y_pred = payload["y_pred"].detach().cpu().numpy()
    if "y_wss_true" not in payload or "y_wss_pred" not in payload:
        raise SystemExit("预测文件缺少 y_wss_true / y_wss_pred，请用 field 模型重新 predict")

    is_wall_idx = NODE_FEATURE_NAMES.index("is_wall")
    wall = x[:, is_wall_idx] > 0.5
    interior = ~wall
    pos = x[:, 0:3]
    tan_lo, tan_hi = FEATURE_INDICES["tangent"]
    tangents = x[:, tan_lo:tan_hi]

    norm_path = Path(args.norm_params)
    if not norm_path.is_file():
        norm_path = _gnn_project_root() / norm_path
    stats = _load_norm_stats(norm_path)

    cfd_p = _denorm_zscore(y_true[:, 3], stats, "p")
    pred_p = _denorm_zscore(y_pred[:, 3], stats, "p")
    cfd_wss = _denorm_zscore(payload["y_wss_true"].detach().cpu().numpy()[:, 0], stats, "wss")
    pred_wss = _denorm_zscore(payload["y_wss_pred"].detach().cpu().numpy()[:, 0], stats, "wss")

    axis_names = ("x", "y", "z")
    coords_2d: Optional[np.ndarray] = None
    axis_labels: Optional[Tuple[str, str]] = None
    axes_plane = (0, 1)

    if args.slice_mode == "axis":
        prefer = None if args.slice_axis == -1 else args.slice_axis
        if args.slice_value is None:
            slice_axis, slice_val, slice_mask = _auto_axis_slice(
                pos, interior, wall, tol=args.slice_tol, prefer_axis=prefer,
            )
        else:
            slice_axis = 2 if args.slice_axis == -1 else args.slice_axis
            slice_val = float(args.slice_value)
            slice_mask = np.abs(pos[:, slice_axis] - slice_val) < args.slice_tol
        plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        if args.plane == "auto":
            remaining = [a for a in (0, 1, 2) if a != slice_axis]
            axes_plane = (remaining[0], remaining[1])
        else:
            axes_plane = plane_map[args.plane]
        slice_label = (
            f"Axis slice: {axis_names[slice_axis]} = {slice_val:.3f} ± {args.slice_tol:.3f} m "
            f"(interior={(interior & slice_mask).sum()}, wall={(wall & slice_mask).sum()} nodes)"
        )
    else:
        if args.abscissa_value is None:
            abs_val, slice_mask = _auto_abscissa_slice(
                x, interior, wall, tol=args.abscissa_tol,
            )
        else:
            abs_val = float(args.abscissa_value)
            abs_idx = NODE_FEATURE_NAMES.index("Abscissa")
            slice_mask = np.abs(x[:, abs_idx] - abs_val) < args.abscissa_tol
        coords_2d, axis_labels = _tangent_plane_coords(pos, tangents, slice_mask)
        slice_label = (
            f"Abscissa-normal slice: Abscissa = {abs_val:.3f} ± {args.abscissa_tol:.3f} "
            f"(interior={(interior & slice_mask).sum()}, wall={(wall & slice_mask).sum()} nodes)"
        )

    run_dir = manifest_path.parent.parent
    exp_id = ""
    config_path = run_dir / "config.snapshot.json"
    if config_path.is_file():
        snap = json.loads(config_path.read_text(encoding="utf-8"))
        meta = snap.get("meta") or {}
        exp_id = str(meta.get("exp_id") or snap.get("run", {}).get("experiment_name", ""))

    resolved_case = str(payload.get("case_name") or item.get("case_name") or case_name or args.sample_id)
    title = args.title.strip() or f"{resolved_case} · {exp_id or 'V3P field model'}"

    plots_dir = run_dir / "plots" / CAT_CASE_PANELS
    plots_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out = Path(args.output).resolve()
    else:
        safe_case = resolved_case.replace("/", "__")
        out = plots_dir / f"cross_section_{safe_case}_{args.sample_id}.png"

    try:
        plot_case_cross_section_panel(
            pos,
            interior,
            wall,
            cfd_p,
            pred_p,
            cfd_wss,
            pred_wss,
            slice_mask,
            slice_label=slice_label,
            save_path=str(out),
            title=title,
            axes_plane=axes_plane,
            axis_labels=axis_labels,
            coords_2d=coords_2d,
            render_mode=args.render_mode,
            row_labels=("Interior pressure (p)", "Wall shear stress (WSS)"),
        )
    except ImportError as exc:
        raise SystemExit("生成图像失败：当前环境缺少 matplotlib。") from exc

    print(out)


if __name__ == "__main__":
    main()
