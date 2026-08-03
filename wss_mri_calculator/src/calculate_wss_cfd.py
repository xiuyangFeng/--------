"""
从 CFD 速度场估算 WSS 并与 CFD 真值对比（data_new 队列适配版）
==============================================================

与原 `calculate_wss.py` 的关系
-----------------------------
原脚本面向 4D Flow MRI：规则体素网格 → PyVista UniformGrid → 提取等值面 →
沿法向取 pc0/pc1/pc2 三层 → 三点抛物线拟合 → WSS = μ·(∂|v_t|/∂n)。

本脚本保留**同一套物理**（切向速度沿内法向的壁面梯度 × 粘度），但替换三处
与"体素影像"绑定的环节，因为本项目数据是 Fluent 非结构化点云：

  1. 网格构建：不再造 UniformGrid / threshold / extract_surface / smooth，
     壁面几何直接取 CFD 壁面节点，法向默认由壁面点云局部 PCA 估计。
  2. 采样方式：不再沿法向取固定 3 层等距点插值，改为取壁面点的 K 近邻
     内部单元，用其到壁面的**法向投影距离** eta 做加权最小二乘拟合。
     理由（不是「个别细血管才密」）：全库 CFD 近壁用的是各向异性棱柱边界层，
     速度只定义在单元中心。原 MRI 流程在规则体素上可任意插值；这里若固定
     0.6mm（甚至 0.10mm）三层探针，探针位置通常落在网格点之间，仍要插值，
     且固定物理深度无法随局部网格密度/管径自适应——深了穿对侧，浅了采样点
     不在真实速度节点上。K 近邻直接用已有单元中心，再按 eta 拟合更稳。
  3. 粘度：CFD 用 Carreau-Yasuda **非牛顿**血液模型（见 udf-inlet.c），
     μ 随剪切率在 0.16 → 0.0035 Pa·s 间变化（约 46 倍跨度）。固定
     3.5 cP = 0.0035 Pa·s 只对应高剪切极限，在低剪切区会系统性低估 WSS。

法向定向（关键修正）
--------------------
STL 面法向的朝向不保证一致。原实现用「指向几何质心」定向，对弯曲主动脉
不成立（实测 39% 的点被判错向 → 近壁点全落在负侧 → 有效点只剩 61%）。
本脚本改用**局部定向**：让法向指向最近 32 个内部单元的质心，实测翻转率 0%、
有效点覆盖 100%。

标定说明
--------
`raw` 指标是纯物理前向（无任何拟合真值的自由参数）。`scaled` 额外允许一个
全局标量 α（pred → α·pred），用于分离"空间分布对不对"与"整体幅值偏差"。

用法
----
  python calculate_wss_cfd.py --case-dir /path/to/AG/slow/LIU_JIN_LIANG
  python calculate_wss_cfd.py --case-dir ... --sweep      # 扫 K/degree 找最优
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import data_loader_cfd as dl
from wss_pinn.physics.rheology import carreau_yasuda_numpy
from wss_pinn.physics.wall_shear import stl_face_geometry
from wss_multiscale import fit_wall_gradient_multiscale

MM_TO_M = 1e-3
MU_HIGH_SHEAR = 0.0035  # Pa·s = 3.5 cP，Carreau–Yasuda 高剪切极限
DEFAULT_ADAPTIVE_NEIGHBORS = (16, 20, 24, 28, 32, 36, 40, 44, 48, 64)
DEFAULT_CV_TOLERANCE = 1.25
TRAIN_FROZEN_GLOBAL_SCALE = 1.2220224407405893
TRAIN_FROZEN_MULTISCALE_V2_SCALE = 1.157066322432233


def find_stl(case_dir: Path) -> Path:
    """
    定位病例的壁面 STL。

    AG / AAA 队列命名为 `<病例名>.stl`；ILO 队列用的是别名（如
    `0gaoshucai-sq.stl`），因此优先按病例名匹配，找不到就退回目录里唯一的
    STL；多个候选时取最大的那个（完整壁面面片数最多）。
    """
    preferred = case_dir / f"{case_dir.name}.stl"
    if preferred.exists():
        return preferred
    candidates = sorted(case_dir.glob("*.stl"), key=lambda p: p.stat().st_size)
    if not candidates:
        raise FileNotFoundError(f"no .stl found under {case_dir}")
    return candidates[-1]


def find_peak_step(case_dir: Path) -> int | None:
    """
    从 data_wss_min 对应 bundle.npz 读取 peak_step。

    病例目录若位于 data_new/<cohort>/...，则映射到
    data_wss_min/<cohort>/.../bundle.npz。找不到则返回 None。
    """
    case_dir = case_dir.resolve()
    parts = case_dir.parts
    if "data_new" not in parts:
        return None
    idx = parts.index("data_new")
    bundle = Path(*parts[:idx]) / "data_wss_min" / Path(*parts[idx + 1 :]) / "bundle.npz"
    if not bundle.exists():
        return None
    with np.load(bundle, allow_pickle=False) as source:
        return int(source["peak_step"])


def resolve_step(
    case_dir: Path, step: int | None, prefer_peak: bool = True
) -> tuple[int, str]:
    """
    解析要计算的时间步。

    优先级：显式 --step > bundle peak_step（默认）> ascii 目录第一帧。
    """
    steps = dl.list_steps(case_dir, "ascii")
    if not steps:
        raise FileNotFoundError(f"no ascii timesteps under {case_dir}")
    if step is not None:
        if step not in steps:
            raise FileNotFoundError(
                f"step {step} not found under {case_dir}/ascii；可用: {steps[0]}..{steps[-1]}"
            )
        return step, "explicit"
    if prefer_peak:
        peak = find_peak_step(case_dir)
        if peak is not None and peak in steps:
            return peak, "bundle_peak"
        if peak is not None:
            # peak 有定义但不在当前 ascii 导出里，退回首帧并标明原因
            return steps[0], f"first_fallback_missing_peak_{peak}"
    return steps[0], "first_available"


def save_scatter_plot(
    truth_mag: np.ndarray,
    pred_mag: np.ndarray,
    report: dict,
    output_path: Path,
) -> Path:
    """画出 CFD 真值 WSS vs 速度重建 WSS 的散点图。"""
    import matplotlib.pyplot as plt

    valid = np.isfinite(truth_mag) & np.isfinite(pred_mag)
    y = truth_mag[valid]
    p = pred_mag[valid]
    alpha = float(report.get("alpha", 1.0))
    lim = float(max(y.max(), p.max(), 1e-6)) * 1.05

    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=140)
    # 散点坐标：(truth, pred)。evaluate() 里 alpha 满足 truth ≈ alpha·pred，
    # 因此图上的拟合线应是 pred = truth/alpha，即 y = x/alpha。
    ax.scatter(y, p, s=8, alpha=0.35, c="#1f77b4", edgecolors="none", label="points")
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="y = x")
    ax.plot(
        [0, lim],
        [0, lim / max(alpha, 1e-12)],
        color="#d62728",
        lw=1.2,
        label=f"y = x/{alpha:.3f}  (truth≈α·pred)",
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("Velocity→WSS pred (Pa)")
    title = (
        f"{report.get('case', '')}  step={report.get('step')} ({report.get('step_source', '')})\n"
        f"raw R²={report.get('raw_r2', float('nan')):.3f}  "
        f"scaled R²={report.get('scaled_r2', float('nan')):.3f}  "
        f"Spearman={report.get('spearman', float('nan')):.3f}  "
        f"α={alpha:.3f}"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ----------------------------------------------------------------------
# 壁面法向
# ----------------------------------------------------------------------
def _orient_inward(
    normals: np.ndarray,
    wall_mm: np.ndarray,
    interior_mm: np.ndarray,
    tree: cKDTree,
    orient_k: int,
) -> np.ndarray:
    """
    把法向统一翻转到指向管腔，并单位化。

    定向依据是「壁面点 → 最近 orient_k 个内部单元的质心」这个**局部**方向。
    不能用「指向整个几何的质心」：对弯曲主动脉，实测 39% 的壁面点会被判错向，
    近壁点全落在法向负侧被丢弃，有效点只剩 61%。
    """
    _, nb = tree.query(wall_mm, k=min(orient_k, len(interior_mm)))
    inward = interior_mm[nb].mean(axis=1) - wall_mm
    sign = np.sign(np.einsum("ij,ij->i", inward, normals))
    sign[sign == 0] = 1.0
    n = normals * sign[:, None]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def pca_wall_normals(
    wall_mm: np.ndarray,
    interior_mm: np.ndarray,
    tree: cKDTree,
    neighbors: int = 24,
    orient_k: int = 32,
) -> np.ndarray:
    """
    直接从壁面点云估法向：局部 PCA 取最小主方向（**默认方案**）。

    对每个壁面点取最近 `neighbors` 个壁面点，做协方差矩阵特征分解；
    局部近似平面，因此最小特征值对应的特征向量就是法向。

    相对 STL 方案的优势：完全不依赖 STL 文件。实测有病例（AAA/ruputer/LIU_YU_MING）
    的 STL 只覆盖部分几何（STL z ∈ [-435,-189]，而壁面 z ∈ [-568,-65]），
    导致 wall→STL 最近面距离 p50 达 24.8mm（正常 0.6mm）、法向严重错配
    （与真值 WSS 矢量的 |cos| = 0.445，理论应为 0）。该例 raw R² 因此只有 0.652，
    换 PCA 法向后升到 0.858。正常病例上两者精度持平。
    """
    k = min(int(neighbors), len(wall_mm))
    _, nb = cKDTree(wall_mm).query(wall_mm, k=k)
    centered = wall_mm[nb] - wall_mm[nb].mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]  # 最小特征值对应的特征向量 = 法向
    return _orient_inward(normals, wall_mm, interior_mm, tree, orient_k)


def stl_wall_normals(
    wall_mm: np.ndarray,
    interior_mm: np.ndarray,
    stl_path: str | Path,
    tree: cKDTree,
    orient_k: int = 32,
) -> np.ndarray:
    """
    从 STL 取法向：最近面片的法向 + 局部定向。

    仅当 STL 与壁面点完全对齐时可用。结果里的 `wall_to_stl_distance_p95`
    超过约 3mm 就说明 STL 覆盖不全，应改用 `pca_wall_normals`。
    """
    centers, normals = stl_face_geometry(stl_path, 1.0)  # STL 已是 mm
    _, face_idx = cKDTree(centers).query(wall_mm, k=1)
    return _orient_inward(normals[face_idx], wall_mm, interior_mm, tree, orient_k)


# ----------------------------------------------------------------------
# 近壁剖面拟合
# ----------------------------------------------------------------------
def fit_wall_gradient(
    wall_mm: np.ndarray,
    normals: np.ndarray,
    interior_mm: np.ndarray,
    velocity: np.ndarray,
    tree: cKDTree,
    neighbors: int = 64,
    degree: int = 2,
    bandwidth_weight: bool = False,
    neighbor_mode: str = "adaptive_cv",
    adaptive_neighbors: tuple[int, ...] = DEFAULT_ADAPTIVE_NEIGHBORS,
    cv_tolerance: float = DEFAULT_CV_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    对每个壁面点拟合切向速度剖面 v_t(eta) = a1·eta + a2·eta² + ...，
    解析求导得壁面处速度梯度 a1（即剪切率矢量，单位 1/s）。

    无滑移条件隐含在基函数里：多项式无常数项 ⇒ v_t(0) = 0。

    参数
    ----
    neighbors : fixed 模式的 K 近邻个数。
    degree    : 多项式阶数。2 次最优；3 次开始过拟合噪声。
    bandwidth_weight : 是否对近壁点做高斯加权。实测轻微降低精度，默认关闭。
    neighbor_mode :
        ``fixed`` 使用固定 K；``adaptive_cv`` 对候选 K 做无真值参与的
        leave-one-out 速度剖面交叉验证，并选取接近最优的最小稳定邻域。
    adaptive_neighbors : adaptive_cv 的候选 K，必须严格递增。
    cv_tolerance :
        one-standard-error 风格的容忍系数。选择满足
        ``score <= cv_tolerance * min(score)`` 的最小 K。

    返回
    ----
    gradient : (n,3) 壁面切向速度梯度矢量，1/s；无效点为 NaN
    used     : (n,)  每点实际参与拟合的样本数
    diagnostics : 逐点选中 K、LOOCV 分数与设计矩阵条件数
    """
    if neighbor_mode not in {"fixed", "adaptive_cv"}:
        raise ValueError(f"unknown neighbor mode: {neighbor_mode}")
    candidates = tuple(sorted({int(value) for value in adaptive_neighbors if int(value) > 0}))
    if not candidates:
        raise ValueError("adaptive_neighbors must contain at least one positive integer")
    if cv_tolerance < 1.0:
        raise ValueError("cv_tolerance must be >= 1.0")

    query_k = int(neighbors) if neighbor_mode == "fixed" else max(candidates)
    k = min(query_k, len(interior_mm))
    _, nb = tree.query(wall_mm, k=k)

    gradient = np.full((len(wall_mm), 3), np.nan, dtype=np.float64)
    used = np.zeros(len(wall_mm), dtype=np.int32)
    selected_neighbors = np.zeros(len(wall_mm), dtype=np.int32)
    cv_score = np.full(len(wall_mm), np.nan, dtype=np.float64)
    design_condition = np.full(len(wall_mm), np.nan, dtype=np.float64)
    min_samples = degree + 3

    for row, (point, normal, idx) in enumerate(zip(wall_mm, normals, nb)):
        delta_m = (interior_mm[idx] - point) * MM_TO_M
        eta_all = delta_m @ normal
        velocity_all = velocity[idx]

        row_candidates = (min(int(neighbors), k),) if neighbor_mode == "fixed" else candidates
        fits: list[tuple[int, float, np.ndarray, int, float]] = []
        for candidate in row_candidates:
            candidate = min(int(candidate), k)
            eta = eta_all[:candidate]
            keep = eta > 1e-7
            if keep.sum() < min_samples:
                continue

            e = eta[keep]
            v = velocity_all[:candidate][keep]
            v = v - np.outer(v @ normal, normal)

            # 用局部深度归一化避免 [eta, eta^2] 在米制下产生 1e8 量级条件数。
            scale = max(float(np.median(e)), 1e-8)
            z = e / scale
            design = np.column_stack([z**power for power in range(1, degree + 1)])
            sqrt_weight = np.ones(len(e), dtype=np.float64)
            if bandwidth_weight:
                sqrt_weight = np.sqrt(np.exp(-((e / (2.0 * scale)) ** 2)))

            weighted_design = design * sqrt_weight[:, None]
            weighted_velocity = v * sqrt_weight[:, None]
            try:
                coefficient, *_ = np.linalg.lstsq(
                    weighted_design, weighted_velocity, rcond=1e-10
                )
                gram_inverse = np.linalg.pinv(weighted_design.T @ weighted_design)
                leverage = np.einsum(
                    "ij,jk,ik->i", weighted_design, gram_inverse, weighted_design
                )
                leverage = np.clip(leverage, 0.0, 0.999)
                residual = weighted_velocity - weighted_design @ coefficient
                loo_residual = residual / (1.0 - leverage[:, None])
                signal = float(np.mean(np.sum(weighted_velocity**2, axis=1)))
                score = float(np.mean(np.sum(loo_residual**2, axis=1))) / max(
                    signal, 1e-16
                )
                condition = float(np.linalg.cond(weighted_design.T @ weighted_design))
            except np.linalg.LinAlgError:
                continue

            # coefficient[0] 是对无量纲 z 的导数，除以 scale 才是 1/s。
            fits.append((candidate, score, coefficient[0] / scale, int(keep.sum()), condition))

        if not fits:
            continue
        if neighbor_mode == "adaptive_cv":
            finite_scores = [fit[1] for fit in fits if np.isfinite(fit[1])]
            if not finite_scores:
                continue
            threshold = cv_tolerance * min(finite_scores)
            chosen = next(fit for fit in fits if np.isfinite(fit[1]) and fit[1] <= threshold)
        else:
            chosen = fits[0]

        selected_neighbors[row], cv_score[row], gradient[row], used[row], design_condition[row] = chosen

    diagnostics = {
        "selected_neighbors": selected_neighbors,
        "cv_score": cv_score,
        "design_condition": design_condition,
    }
    return gradient, used, diagnostics


def wss_from_gradient(gradient: np.ndarray, viscosity_model: str) -> np.ndarray:
    """
    WSS 矢量 = μ · (∂v_t/∂n)|_wall。

    viscosity_model:
      'carreau' — Carreau-Yasuda 非牛顿，μ = μ(剪切率)，与 CFD 的 UDF 一致
      'newton'  — 固定 0.0035 Pa·s（= 3.5 cP，高剪切极限）
    """
    shear_rate = np.linalg.norm(gradient, axis=1)
    if viscosity_model == "carreau":
        mu = carreau_yasuda_numpy(np.nan_to_num(shear_rate))
    elif viscosity_model == "newton":
        mu = np.full_like(shear_rate, MU_HIGH_SHEAR)
    else:
        raise ValueError(f"unknown viscosity model: {viscosity_model}")
    return gradient * mu[:, None]


# ----------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------
def check_truth_quality(wall: dict, interior: dict) -> list[str]:
    """
    体检真值侧数据，返回告警列表（不抛异常，让调用方决定是否采信该病例）。

    动机：ILO/DONG_JIA_JU-1/before 的壁面导出有 169 万点（正常量级 1–1.4 万）、
    其中 94% 的 wall-shear 恰为 0，且壁面点数反而多于内部单元数。这类导出本身
    已损坏，任何算子在其上都只能得到负 R²，必须与「算子失效」区分开。
    """
    warnings = []
    n_wall = len(wall["coords_mm"])
    n_interior = len(interior["coords_mm"])
    zero_fraction = float((wall["wss_mag"] == 0).mean())

    if zero_fraction > 0.5:
        warnings.append(f"{zero_fraction:.1%} 的真值 WSS 恰为 0（壁面导出疑似损坏）")
    if n_wall > n_interior:
        warnings.append(f"壁面点数 {n_wall} > 内部单元数 {n_interior}（拓扑异常）")
    if not np.isfinite(wall["wss_mag"]).all():
        warnings.append("真值 WSS 含非有限值")
    return warnings


def evaluate(truth_mag, truth_vec, pred_vec) -> dict:
    """与 CFD 真值对比。raw = 纯物理；scaled = 允许一个全局标量 α。"""
    pred_mag = np.linalg.norm(pred_vec, axis=1)
    valid = np.isfinite(pred_mag) & np.isfinite(truth_mag)
    y, p = truth_mag[valid], pred_mag[valid]
    if len(y) < 10:
        return {"n_valid": int(len(y)), "coverage": float(valid.mean())}

    var = float(((y - y.mean()) ** 2).sum())
    alpha = float((y @ p) / max(p @ p, 1e-30))
    tv, pv = truth_vec[valid], pred_vec[valid]
    cos = np.einsum("ij,ij->i", pv, tv) / np.maximum(
        np.linalg.norm(pv, axis=1) * np.linalg.norm(tv, axis=1), 1e-12
    )
    high = y >= np.quantile(y, 0.9)

    return {
        "n_valid": int(len(y)),
        "coverage": float(valid.mean()),
        "raw_r2": 1.0 - float(((y - p) ** 2).sum()) / max(var, 1e-30),
        "scaled_r2": 1.0 - float(((y - alpha * p) ** 2).sum()) / max(var, 1e-30),
        "alpha": alpha,
        "spearman": float(spearmanr(y, p).statistic),
        "mae_pa": float(np.mean(np.abs(y - p))),
        "nrmse": float(np.sqrt(np.mean((y - p) ** 2)) / max(np.mean(np.abs(y)), 1e-12)),
        "high_wss_nrmse": float(
            np.sqrt(np.mean((y[high] - p[high]) ** 2)) / max(np.mean(np.abs(y[high])), 1e-12)
        ),
        "direction_cosine_p50": float(np.nanmedian(cos)),
        "truth_mean_pa": float(y.mean()),
        "pred_mean_pa": float(p.mean()),
    }


def run_case(
    case_dir: Path,
    step: int | None,
    sample_count: int,
    neighbors: int,
    degree: int,
    viscosity_model: str,
    bandwidth_weight: bool,
    seed: int,
    normals_mode: str = "pca",
    prefer_peak: bool = True,
    return_arrays: bool = False,
    neighbor_mode: str = "adaptive_cv",
    adaptive_neighbors: tuple[int, ...] = DEFAULT_ADAPTIVE_NEIGHBORS,
    cv_tolerance: float = DEFAULT_CV_TOLERANCE,
    prediction_scale: float = 1.0,
) -> dict:
    step, step_source = resolve_step(case_dir, step, prefer_peak=prefer_peak)

    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))

    # 真值损坏 → 该病例的对比结果无意义
    data_warnings = check_truth_quality(wall, interior)
    # 几何告警 → 真值仍可用，只是不该用 STL 法向
    geometry_warnings: list[str] = []
    tree = cKDTree(interior["coords_mm"])

    # STL 对齐诊断：即使用 PCA 法向也算一遍，好让 STL 覆盖不全的病例显形
    stl_distance_p95 = float("nan")
    try:
        stl = find_stl(case_dir)
        centers, _ = stl_face_geometry(str(stl), 1.0)
        distance, _ = cKDTree(centers).query(wall["coords_mm"], k=1)
        stl_distance_p95 = float(np.quantile(distance, 0.95))
        if stl_distance_p95 > 3.0:
            geometry_warnings.append(
                f"STL 与壁面点错配：wall→STL 距离 p95 = {stl_distance_p95:.1f}mm"
                "（STL 疑似只覆盖部分几何；PCA 法向不受影响）"
            )
    except FileNotFoundError:
        stl = None
        geometry_warnings.append("缺少 .stl（PCA 法向不受影响）")

    if normals_mode == "pca":
        normals = pca_wall_normals(wall["coords_mm"], interior["coords_mm"], tree)
    elif normals_mode == "stl":
        if stl is None:
            raise FileNotFoundError(f"normals_mode='stl' 但 {case_dir} 下没有 .stl")
        normals = stl_wall_normals(wall["coords_mm"], interior["coords_mm"], stl, tree)
    else:
        raise ValueError(f"unknown normals mode: {normals_mode}")

    n_wall = len(wall["coords_mm"])
    if 0 < sample_count < n_wall:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_wall, sample_count, replace=False))
    else:
        idx = np.arange(n_wall)

    if neighbor_mode == "multiscale_v2":
        if bandwidth_weight:
            raise ValueError("multiscale_v2 does not use bandwidth_weight")
        gradient, used, fit_diagnostics = fit_wall_gradient_multiscale(
            wall["coords_mm"][idx],
            normals[idx],
            interior["coords_mm"],
            interior["velocity"],
            tree,
            adaptive_neighbors=adaptive_neighbors,
            degree=degree,
            base_cv_tolerance=cv_tolerance,
        )
    else:
        gradient, used, fit_diagnostics = fit_wall_gradient(
            wall["coords_mm"][idx],
            normals[idx],
            interior["coords_mm"],
            interior["velocity"],
            tree,
            neighbors=neighbors,
            degree=degree,
            bandwidth_weight=bandwidth_weight,
            neighbor_mode=neighbor_mode,
            adaptive_neighbors=adaptive_neighbors,
            cv_tolerance=cv_tolerance,
        )
    if prediction_scale <= 0:
        raise ValueError("prediction_scale must be positive")
    pred = wss_from_gradient(gradient, viscosity_model) * float(prediction_scale)
    truth_mag = wall["wss_mag"][idx]
    truth_vec = wall["wss_vec"][idx]
    report = evaluate(truth_mag, truth_vec, pred)
    report.update(
        {
            "case": f"{case_dir.parent.name}/{case_dir.name}",
            "step": step,
            "step_source": step_source,
            "n_wall_total": n_wall,
            "n_interior": len(interior["coords_mm"]),
            "neighbors": neighbors,
            "neighbor_mode": neighbor_mode,
            "adaptive_neighbors": list(adaptive_neighbors),
            "cv_tolerance": cv_tolerance,
            "prediction_scale": prediction_scale,
            "degree": degree,
            "viscosity_model": viscosity_model,
            "bandwidth_weight": bandwidth_weight,
            "normals_mode": normals_mode,
            "samples_used_p50": float(np.median(used[used > 0])) if (used > 0).any() else 0.0,
            "selected_neighbors_p05": float(
                np.quantile(
                    fit_diagnostics["selected_neighbors"][
                        fit_diagnostics["selected_neighbors"] > 0
                    ],
                    0.05,
                )
            )
            if (fit_diagnostics["selected_neighbors"] > 0).any()
            else 0.0,
            "selected_neighbors_p50": float(
                np.median(
                    fit_diagnostics["selected_neighbors"][
                        fit_diagnostics["selected_neighbors"] > 0
                    ]
                )
            )
            if (fit_diagnostics["selected_neighbors"] > 0).any()
            else 0.0,
            "selected_neighbors_p95": float(
                np.quantile(
                    fit_diagnostics["selected_neighbors"][
                        fit_diagnostics["selected_neighbors"] > 0
                    ],
                    0.95,
                )
            )
            if (fit_diagnostics["selected_neighbors"] > 0).any()
            else 0.0,
            "fit_cv_score_p50": float(
                np.nanmedian(fit_diagnostics["cv_score"])
            ),
            "fit_condition_p95": float(
                np.nanquantile(fit_diagnostics["design_condition"], 0.95)
            ),
            "multiscale_correction_p50": float(
                np.nanmedian(fit_diagnostics["correction"])
            )
            if neighbor_mode == "multiscale_v2"
            else 1.0,
            "wall_to_stl_distance_p95": stl_distance_p95,
            "data_warnings": data_warnings,
            "geometry_warnings": geometry_warnings,
            "truth_usable": not data_warnings,
        }
    )
    if return_arrays:
        report["_arrays"] = {
            "truth_mag": truth_mag,
            "pred_mag": np.linalg.norm(pred, axis=1),
            "truth_vec": truth_vec,
            "pred_vec": pred,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--case-dir", required=True, help="病例目录，含 ascii/ ascii_in/ 和 .stl")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="时间步；默认优先用 data_wss_min bundle 的 peak_step",
    )
    parser.add_argument(
        "--no-peak",
        action="store_true",
        help="禁用 peak_step，退回 ascii 第一帧（仅当未指定 --step）",
    )
    parser.add_argument("--sample-count", type=int, default=2000, help="抽样壁面点数；0=全部")
    parser.add_argument("--neighbors", type=int, default=64, help="K 近邻个数")
    parser.add_argument(
        "--neighbor-mode",
        choices=["fixed", "adaptive_cv", "multiscale_v2"],
        default="adaptive_cv",
        help=(
            "fixed=固定K基线；adaptive_cv=v1局部LOOCV选K；"
            "multiscale_v2=冻结的多尺度零带宽修正"
        ),
    )
    parser.add_argument(
        "--adaptive-neighbors",
        type=str,
        default=",".join(str(value) for value in DEFAULT_ADAPTIVE_NEIGHBORS),
        help="adaptive_cv 候选K，逗号分隔",
    )
    parser.add_argument(
        "--cv-tolerance",
        type=float,
        default=DEFAULT_CV_TOLERANCE,
        help="adaptive_cv 近最优容忍系数，必须>=1",
    )
    parser.add_argument(
        "--prediction-scale",
        type=float,
        default=1.0,
        help=(
            "可选全局幅值校正；纯物理结果用1.0；adaptive_cv v1 冻结值为 "
            f"{TRAIN_FROZEN_GLOBAL_SCALE:.15f}；multiscale_v2 全壁面冻结值为 "
            f"{TRAIN_FROZEN_MULTISCALE_V2_SCALE:.15f}"
        ),
    )
    parser.add_argument("--degree", type=int, default=2, help="剖面多项式阶数")
    parser.add_argument(
        "--viscosity", choices=["carreau", "newton"], default="carreau", help="粘度模型"
    )
    parser.add_argument(
        "--normals",
        choices=["pca", "stl"],
        default="pca",
        help="法向来源：pca=壁面点云局部PCA（默认，不依赖STL）；stl=最近STL面片法向",
    )
    parser.add_argument("--bandwidth-weight", action="store_true", help="近壁高斯加权")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="扫 K × degree × 粘度")
    parser.add_argument("--json-out", type=str, default=None, help="结果写入 JSON")
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="真值 vs 预测 WSS 散点图路径（png）",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    prefer_peak = not args.no_peak
    want_arrays = args.plot_out is not None and not args.sweep
    adaptive_neighbors = tuple(
        int(value.strip()) for value in args.adaptive_neighbors.split(",") if value.strip()
    )

    if args.sweep:
        rows = []
        header = f"{'K':>4} {'deg':>4} {'visc':>8} {'cover':>6} {'rawR2':>8} {'scaledR2':>9} {'spear':>7} {'alpha':>6} {'cos':>6}"
        print(header)
        print("-" * len(header))
        for viscosity in ("newton", "carreau"):
            for neighbors in (32, 64, 128):
                for degree in (1, 2, 3):
                    r = run_case(
                        case_dir, args.step, args.sample_count, neighbors, degree,
                        viscosity, args.bandwidth_weight, args.seed, args.normals,
                        prefer_peak=prefer_peak,
                        neighbor_mode="fixed",
                        adaptive_neighbors=adaptive_neighbors,
                        cv_tolerance=args.cv_tolerance,
                        prediction_scale=args.prediction_scale,
                    )
                    rows.append(r)
                    print(
                        f"{neighbors:>4} {degree:>4} {viscosity:>8} {r['coverage']:>6.3f} "
                        f"{r['raw_r2']:>8.3f} {r['scaled_r2']:>9.3f} {r['spearman']:>7.3f} "
                        f"{r['alpha']:>6.3f} {r['direction_cosine_p50']:>6.3f}"
                    )
        best = max(rows, key=lambda r: r["raw_r2"])
        print(
            f"\n最优 raw R²: K={best['neighbors']} deg={best['degree']} "
            f"visc={best['viscosity_model']} → {best['raw_r2']:.3f}"
            f"  (step={best['step']} / {best['step_source']})"
        )
        payload = {"case": rows[0]["case"], "sweep": rows, "best": best}
    else:
        r = run_case(
            case_dir, args.step, args.sample_count, args.neighbors, args.degree,
            args.viscosity, args.bandwidth_weight, args.seed, args.normals,
            prefer_peak=prefer_peak,
            return_arrays=want_arrays,
            neighbor_mode=args.neighbor_mode,
            adaptive_neighbors=adaptive_neighbors,
            cv_tolerance=args.cv_tolerance,
            prediction_scale=args.prediction_scale,
        )
        arrays = r.pop("_arrays", None)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        if args.plot_out is not None:
            if arrays is None:
                raise RuntimeError("plot requested but arrays were not returned")
            plot_path = save_scatter_plot(
                arrays["truth_mag"], arrays["pred_mag"], r, Path(args.plot_out)
            )
            print(f"\n散点图写入 {plot_path}")
        payload = r

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
