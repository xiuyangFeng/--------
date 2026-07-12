#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Chinese-labelled figures for the 2026-07-12 WSS-min update."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; registers the 3-D projection


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "training_wss_min" / "runs"
OUT = ROOT / "docs" / "03-汇报材料" / "figures" / "WSS最小路线_20260712"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def setup_style() -> None:
    if FONT.exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(run: str, partition: str = "test") -> dict:
    m = load_json(RUNS / run / "eval" / "metrics.json")[partition]
    return {
        "R²_field": float(m["field"]["r2"]),
        "R²_casemean": float(m["aggregate"]["r2_casemean"]),
        "分叉区": float(m["regional_field"]["bifurcation"]["r2"]),
        "狭窄区": float(m["regional_field"]["stenosis"]["r2"]),
        "高WSS区": float(m["regional_field"]["high_wss"]["r2"]),
        "MAE": float(m["field"]["mae"]),
    }


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def plot_feature_comparison() -> None:
    runs = {
        "纯 XYZ": "pc_xyz_fps_w2000_peak",
        "XYZ + 几何": "feat_xyzgeom_fps_w2000_peak",
    }
    data = {name: metric_summary(run) for name, run in runs.items()}
    colors = ["#687A8F", "#E45756"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})

    keys = ["R²_field", "R²_casemean"]
    x = np.arange(len(keys))
    width = 0.34
    for i, (name, vals) in enumerate(data.items()):
        bars = axes[0].bar(x + (i - .5) * width, [vals[k] for k in keys], width,
                           label=name, color=colors[i])
        for bar in bars:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .008,
                         f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["全场池化 R²", "病例等权 R²"])
    axes[0].set_ylim(0, .25)
    axes[0].set_ylabel("R²（越高越好）")
    axes[0].set_title("整体预测精度")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=.2)

    regions = ["分叉区", "狭窄区", "高WSS区"]
    xr = np.arange(len(regions))
    for i, (name, vals) in enumerate(data.items()):
        bars = axes[1].bar(xr + (i - .5) * width, [vals[k] for k in regions], width,
                           label=name, color=colors[i])
        for bar in bars:
            y = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, y + (.05 if y >= 0 else -.08),
                         f"{y:.2f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=9)
    axes[1].axhline(0, color="#333333", linewidth=.8)
    axes[1].set_xticks(xr)
    axes[1].set_xticklabels(regions)
    axes[1].set_ylim(-2.25, .25)
    axes[1].set_ylabel("分区 R²")
    axes[1].set_title("困难区域精度")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("纯 XYZ 与 XYZ+几何特征对比（FPS-2000）", fontsize=16, fontweight="bold")
    fig.text(.5, -.01, "第一轮特征消融·同模型/同点数/同种子·test 完整壁面点云",
             ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, .93])
    save(fig, "01_纯XYZ与XYZ加几何_预测精度对比.png")


def history(run: str):
    rows = []
    for line in (RUNS / run / "history.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def moving_average(y: np.ndarray, window: int = 11) -> np.ndarray:
    if len(y) < window:
        return y
    left = window // 2
    padded = np.pad(y, (left, window - left - 1), mode="edge")
    return np.convolve(padded, np.ones(window)/window, mode="valid")


def plot_training_curves() -> None:
    specs = [
        ("纯 XYZ", "pc_xyz_fps_w2000_peak", "#687A8F"),
        ("XYZ + 几何", "feat_xyzgeom_fps_w2000_peak", "#E45756"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex="col")
    for col, (label, run, color) in enumerate(specs):
        rows = history(run)
        epochs = np.asarray([r["epoch"] for r in rows])
        loss = np.asarray([r["train_loss"] for r in rows])
        axes[0, col].plot(epochs, loss, color=color, alpha=.22, linewidth=.8)
        axes[0, col].plot(epochs, moving_average(loss), color=color, linewidth=2,
                          label="11 epoch 平滑")
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(label)
        axes[0, col].set_ylabel("训练损失（对数轴）")
        axes[0, col].grid(alpha=.2)

        vals = [r for r in rows if "val_r2_casemean" in r]
        ev = np.asarray([r["epoch"] for r in vals])
        field = np.asarray([r["val_r2_field"] for r in vals])
        case = np.asarray([r["val_r2_casemean"] for r in vals])
        axes[1, col].plot(ev, field, "o-", color="#2A9D8F", label="val 全场 R²", markersize=4)
        axes[1, col].plot(ev, case, "o-", color="#F4A261", label="val 病例等权 R²", markersize=4)
        best = max(vals, key=lambda r: r["val_r2_casemean"])
        axes[1, col].axvline(best["epoch"], color="#333333", linestyle="--", linewidth=1)
        axes[1, col].annotate(f"最佳 checkpoint\nepoch {best['epoch']}",
                              (best["epoch"], best["val_r2_casemean"]),
                              xytext=(best["epoch"] + 22, best["val_r2_casemean"] + .08),
                              arrowprops={"arrowstyle": "->", "color": "#333333"}, fontsize=9)
        axes[1, col].axhline(0, color="#777777", linewidth=.7)
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("验证集 R²")
        axes[1, col].set_ylim(-.45, .42)
        axes[1, col].grid(alpha=.2)
        axes[1, col].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("纯 XYZ 与 XYZ+几何的训练曲线", fontsize=16, fontweight="bold")
    fig.text(.5, .01, "同为 400 epoch、FPS-2000、seed=1234；虚线为按 val R²_casemean 选中的 checkpoint",
             ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, .95])
    save(fig, "02_纯XYZ与XYZ加几何_训练曲线.png")


def read_csv(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_pointcount_curve() -> None:
    rows = read_csv(RUNS / "_summary_round4_pointcount" / "pointcount_metrics_dev1_s1234.csv")
    n = np.asarray([int(r["wall_n_points"]) for r in rows])
    field = np.asarray([float(r["r2_field"]) for r in rows])
    case = np.asarray([float(r["r2_casemean"]) for r in rows])
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(n, field, "o-", color="#2A9D8F", linewidth=2.2, markersize=7, label="全场 R²")
    ax.plot(n, case, "o-", color="#F4A261", linewidth=2.2, markersize=7, label="病例等权 R²")
    for xs, ys in [(n, field), (n, case)]:
        for x, y in zip(xs, ys):
            ax.text(x, y + .012, f"{y:.3f}", ha="center", fontsize=9)
    ax.axvspan(900, 2100, color="#E9C46A", alpha=.13)
    ax.axvline(2000, color="#E45756", linestyle="--", linewidth=1.4)
    ax.annotate("默认锚点：2000 点\n精度与 1000 点接近，保留更高覆盖",
                (2000, field[n.tolist().index(2000)]), xytext=(2550, .37),
                arrowprops={"arrowstyle": "->", "color": "#E45756"}, fontsize=10)
    ax.set_xticks(n)
    ax.set_xlabel("每个病例的训练采样点数")
    ax.set_ylabel("val R²（完整壁面评估）")
    ax.set_ylim(.1, .41)
    ax.grid(alpha=.22)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("XYZ+几何：点数—精度选择曲线（单种子）", fontsize=15)
    fig.text(.5, .015, "第四轮 clean-data / dev1 / seed=1234；点数增加并未单调提升精度",
             ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, 1])
    save(fig, "03_XYZ加几何_点数与精度选择曲线.png")


def plot_multiseed() -> None:
    rows = read_csv(RUNS / "_summary_round4_pointcount" / "pointcount_multiseed_summary.csv")
    # The source contains one row per point count with *_mean and *_std fields.
    rows = sorted(rows, key=lambda r: int(float(r["wall_n_points"])))
    n = np.asarray([int(float(r["wall_n_points"])) for r in rows])
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for key, label, color, offset in [
        ("r2_field", "全场 R²", "#2A9D8F", -15),
        ("r2_casemean", "病例等权 R²", "#F4A261", 15),
    ]:
        mean = np.asarray([float(r[f"{key}_mean"]) for r in rows])
        std = np.asarray([float(r[f"{key}_std"]) for r in rows])
        ax.errorbar(n + offset, mean, yerr=std, fmt="o-", capsize=6, linewidth=2,
                    markersize=7, label=label, color=color)
        for x, y, s in zip(n + offset, mean, std):
            ax.text(x, y + s + .009, f"{y:.3f}", ha="center", fontsize=9)
    ax.set_xticks(n)
    ax.set_xlabel("每个病例的训练采样点数")
    ax.set_ylabel("三随机种子 val R²（均值 ± 标准差）")
    ax.set_ylim(.13, .39)
    ax.grid(alpha=.22)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("1000 点与 2000 点的三随机种子稳定性", fontsize=15)
    fig.text(.5, .015, "全场 R² 持平；1000 点的病例等权 R² 略高，但不足以改变 2000 点默认锚点",
             ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, 1])
    save(fig, "04_一千点与两千点_三随机种子稳定性.png")


def farthest_point_sample(pts: np.ndarray, k: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.RandomState(seed)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = rng.randint(len(pts))
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(pts - pts[sel[i]], axis=1))
    return sel


def equal_3d(ax, pts: np.ndarray) -> None:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid = (lo + hi) / 2
    radius = (hi - lo).max() / 2
    ax.set_xlim(mid[0]-radius, mid[0]+radius)
    ax.set_ylim(mid[1]-radius, mid[1]+radius)
    ax.set_zlim(mid[2]-radius, mid[2]+radius)


# 复用 visualize_sampling.py 已导出的 ParaView 诊断包：壁面点云与 STL 三角面在
# 同一配准坐标系内，直接读取即可保证曲面与采样点严格对齐（需 vtk，用 GNN 环境运行）。
SAMPLING_VIZ = (RUNS / "r4_dev1_b0_tgtw_batchq_s1234" / "sampling_viz" / "fast__RAN_QING_BO")


def _read_vtp(path: Path):
    """返回 (顶点 Nx3, 三角形索引 Mx3 或 None)；仅在导出图时依赖 vtk。"""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    tris = None
    if poly.GetNumberOfPolys() > 0:
        tris = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(np.int64)
    return verts, tris


def _draw_vessel_surface(ax, verts: np.ndarray, tris: np.ndarray) -> None:
    """把 STL 三角面画成半透明浅灰背景，让血管几何一眼可辨。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = Poly3DCollection(verts[tris], alpha=.10, linewidths=0)
    mesh.set_facecolor("#B9C2CC")
    mesh.set_edgecolor("none")
    mesh.set_zsort("average")
    ax.add_collection3d(mesh)


def plot_sampling() -> None:
    verts, tris = _read_vtp(SAMPLING_VIZ / "vessel_surface.vtp")
    pts, _ = _read_vtp(SAMPLING_VIZ / "wall_points_all.vtp")
    selections = {k: farthest_point_sample(pts, k) for k in (1000, 2000, 6000)}
    fig = plt.figure(figsize=(15, 5.4))
    panels = [("完整壁面\n17,682 点", None), ("FPS-1000", 1000), ("FPS-2000", 2000), ("FPS-6000", 6000)]
    for i, (title, k) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        _draw_vessel_surface(ax, verts, tris)
        if k is None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=.5, c="#8A97A6",
                       alpha=.45, depthshade=False)
        else:
            q = pts[selections[k]]
            # 三个面板用同一 marker 尺寸，让点数差异如实体现为疏密差异。
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=1.3,
                       c="#E4342F", alpha=.92, depthshade=False)
        # 沿 -Y 侧视：血管长轴（Z）竖直、Y 形分叉清晰展开，避免斜视角把前后壁叠糊。
        ax.view_init(elev=8, azim=-88)
        equal_3d(ax, verts)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
    fig.suptitle("完整壁面几何与 FPS 采样点覆盖对比", fontsize=16, fontweight="bold")
    fig.text(.5, .02, "浅灰：真实血管壁面（STL 三角面）　红色：训练时选中的最远点采样（FPS）点　"
             "病例 fast/RAN_QING_BO", ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=[0, .05, 1, .92])
    save(fig, "05_完整壁面与FPS选取点_可视化对比.png")


def plot_prediction_example() -> None:
    case_file = "AG_slow__GONG_HUI_XIA.png"
    source = [
        (RUNS / "pc_xyz_fps_w2000_peak" / "eval" / "heatmaps" / case_file, "纯 XYZ：病例 R² = 0.006"),
        (RUNS / "feat_xyzgeom_fps_w2000_peak" / "eval" / "heatmaps" / case_file, "XYZ + 几何：病例 R² = 0.352"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.2))
    for ax, (path, title) in zip(axes, source):
        image = plt.imread(str(path))
        ax.imshow(image[28:, :, :])
        ax.set_title(title, fontsize=14, pad=3)
        ax.axis("off")
    fig.suptitle("几何特征增益典型病例：真值、预测与绝对误差", fontsize=16, fontweight="bold")
    fig.text(.5, .012, "该例用于展示几何特征的典型增益，不代表所有病例都同幅改善",
             ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=[0, .03, 1, .95])
    save(fig, "06_几何特征增益典型病例_预测云图对比.png")


def plot_learning_curve() -> None:
    rows = read_csv(RUNS / "_round5" / "learning_curve" / "lc_points.csv")
    sizes = [13, 26, 40, 53]
    metrics = [("field_r2", "全场 R²"), ("casemean", "病例等权 R²")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for ax, (key, label) in zip(axes, metrics):
        shared = {r["seed"]: float(r[key]) for r in rows if int(r["size"]) == 53}
        for seed in ("1234", "7", "2025"):
            for chain in ("chainA", "chainB", "chainC"):
                vals = []
                for size in (13, 26, 40):
                    item = next(r for r in rows if r["seed"] == seed and r["chain"] == chain
                                and int(r["size"]) == size)
                    vals.append(float(item[key]))
                vals.append(shared[seed])
                ax.plot(sizes, vals, color="#AAB2BA", alpha=.45, linewidth=.9)
        means, stds = [], []
        for size in sizes:
            values = [float(r[key]) for r in rows if int(r["size"]) == size]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)))
        ax.errorbar(sizes, means, yerr=stds, fmt="o-", color="#2C7FB8", linewidth=2.4,
                    markersize=7, capsize=5, label="均值 ± 标准差")
        for x, y in zip(sizes, means):
            ax.text(x, y + .035, f"{y:.3f}", ha="center", fontsize=9)
        ax.axhline(.70, color="#E45756", linestyle="--", linewidth=1.2, label="工程目标 0.70")
        ax.set_xticks(sizes)
        ax.set_xlabel("训练病例数")
        ax.set_title(label)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylabel("dev1 val R²")
    axes[0].set_ylim(-.04, .75)
    fig.suptitle("训练病例数—泛化精度学习曲线", fontsize=16, fontweight="bold")
    fig.text(.5, .015, "3 条嵌套病例链 × 3 随机种子；全场 R² 约在 40 例后进入 ~0.31 平台",
             ha="center", color="#555555", fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save(fig, "07_训练病例数与泛化精度_学习曲线.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_feature_comparison()
    plot_training_curves()
    plot_pointcount_curve()
    plot_multiseed()
    plot_sampling()
    plot_prediction_example()
    plot_learning_curve()
    print(OUT)


if __name__ == "__main__":
    main()
