"""Report completed velocity-derived WSS evaluation using measured artifacts only.

Writes README, full aggregate metrics, per-case CSV, PNG/PDF figures and their
provenance. Never invokes inference, training, archived reporters or workbooks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import FuncNorm, LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "training_wss_min/experiments/v5_velocity_to_wss_20260909"
RUN = ROOT / "training_wss_min/runs/v5_rerun_20260906/outputs/r5v_velocity_qad_s1234"
REFERENCES = {
    "R4": ROOT / "training_wss_min/runs/v5_rerun_20260906/outputs/r4_lsa2_h2_logradius_v5feat_s1234/eval/ckpt_best/metrics.json",
    "M2": ROOT / "training_wss_min/runs/v6_followup_20260909/M2_a5_independent_k3_s1234/eval/ckpt_best/metrics.json",
}
METHODS = ("test", "oracle", "physics_only")
LABELS = {"test": "VELWSS1 预测速度→冻结 V3", "oracle": "审计：CFD 速度→冻结 V3",
          "physics_only": "审计：预测速度→未校准物理核", "R4": "参考 R4 直接拟合 WSS", "M2": "参考 M2 直接拟合 WSS"}
ARRAY_KEYS = {"test": "pred_wss_pa", "oracle": "oracle_wss_pa", "physics_only": "physics_wss_pa"}
COLORS = {"test": "#d55e00", "oracle": "#009e73", "physics_only": "#0072b2", "R4": "#999999", "M2": "#cc79a7"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    stage = path.with_suffix(path.suffix + ".tmp")
    stage.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    stage.replace(path)


def dig(m, path, default=None):
    for key in path.split("."):
        if not isinstance(m, dict) or key not in m:
            return default
        m = m[key]
    return m


def fmt(value, digits=5):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "未定义"
    return str(value).replace("|", "\\|").replace("\n", " ")


def scalar_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in {"per_case", "case_metadata", "efficiency"}:
                yield from scalar_leaves(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def require_close(actual, expected, where):
    if not finite(expected) or not math.isclose(float(actual), float(expected), rel_tol=2e-6, abs_tol=2e-6):
        raise ValueError(f"Prediction archive/metrics mismatch {where}: {actual} != {expected}")


def configure_font():
    candidates = [ROOT / "training_wss_min/experiments/v6_followup_20260909/assets/NotoSansCJK-Regular.ttc",
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return str(path)
    raise FileNotFoundError("A CJK font is required for the Chinese report figures")


def load_inputs(exp):
    filenames = ("metrics.json", "evaluation_gate.json", "velocity_reproduction.json", "provenance.json",
                 "radius_split_audit.json", "prediction_manifest.json")
    source_hashes = {str(exp / name): sha(exp / name) for name in filenames}
    metrics, gate, velocity, provenance, radius_audit, manifest = (read(exp / name) for name in filenames)
    if gate.get("passed") is not True or gate.get("n_cases") != 34 or gate.get("checkpoint") != "best":
        raise ValueError("Report requires a completed passed/best/test34 evaluation gate")
    if velocity.get("passed") is not True or velocity.get("status") != "complete" or velocity.get("n_cases") != 34:
        raise ValueError("Report requires completed 34-case velocity reproduction")
    for name in ("metrics", "provenance", "prediction_manifest", "radius_split_audit"):
        if gate[name + "_sha256"] != source_hashes[str(exp / (name + ".json"))]:
            raise ValueError(f"Evaluation gate hash mismatch for {name}")
    if (gate.get("calibrator_train_test_overlap") != 0
            or gate.get("calibrator_train_matches_current_train138") is not True
            or radius_audit["split_evidence"]["calibrator_training_overlap_v5_test"]):
        raise ValueError("Calibrator train/test separation not established")
    ids = sorted(metrics["test"]["per_case"])
    if len(ids) != 34 or set(ids) != set(velocity["expected_cases"]):
        raise ValueError("Velocity and derived WSS case identities differ")
    for key in METHODS:
        m = metrics[key]
        if (m["aggregate"]["n_cases"] != 34 or m["normalized"]["aggregate"]["n_cases"] != 34
                or set(m["per_case"]) != set(ids) or set(m["normalized"]["per_case"]) != set(ids)):
            raise ValueError(f"Incomplete or unpaired case metrics for {key}")
    for key, path in REFERENCES.items():
        metrics[key] = read(path)["test"]
        source_hashes[str(path)] = sha(path)
        if set(metrics[key]["per_case"]) != set(ids):
            raise ValueError(f"Historical {key} case set differs; cannot form the paired reference table")
    predictions = {"truth": [], **{key: [] for key in METHODS}}
    archives = []
    for uid in ids:
        path = exp / "predictions" / (uid.replace("/", "__") + ".npz")
        with np.load(path, allow_pickle=False) as z:
            truth = np.asarray(z["truth_wss_pa"], dtype=np.float64).reshape(-1)
            wall = np.asarray(z["wall_mm"], dtype=np.float64)
            if wall.shape != (len(truth), 3) or not np.isfinite(wall).all():
                raise ValueError(f"Invalid wall coordinate shape/values: {uid}")
            if "canonical_id" in z and str(z["canonical_id"]) != uid:
                raise ValueError(f"Wrong archive identity: {uid}")
            predictions["truth"].append(truth)
            truth_sst = np.sum((truth - truth.mean()) ** 2)
            if not np.isfinite(truth).all() or np.any(truth < 0) or truth_sst <= 0:
                raise ValueError(f"Nonfinite/negative/constant WSS truth: {uid}")
            for key in METHODS:
                pred = np.asarray(z[ARRAY_KEYS[key]], dtype=np.float64).reshape(-1)
                if pred.shape != truth.shape or not np.isfinite(pred).all() or np.any(pred < 0):
                    raise ValueError(f"Unreported invalid/negative WSS prediction: {key}/{uid}")
                error = pred - truth
                computed = {"r2": float(1 - np.dot(error, error) / truth_sst),
                            "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
                recorded = metrics[key]["per_case"][uid]["overall"]
                if recorded["n"] != len(truth):
                    raise ValueError(f"Archive/metric node count differs: {key}/{uid}")
                for name, value in computed.items():
                    require_close(value, recorded[name], f"{key}/{uid}/{name}")
                predictions[key].append(pred)
            digest = sha(path)
            if manifest[uid]["sha256"] != digest:
                raise ValueError(f"Prediction archive differs from evaluated manifest: {uid}")
            archives.append({"uid": uid, "path": str(path), "sha256": digest, "n_wall": len(truth)})
    arrays = {key: np.concatenate(value) for key, value in predictions.items()}
    truth = arrays["truth"]
    sst = np.sum((truth - truth.mean()) ** 2)
    for key in METHODS:
        error = arrays[key] - truth
        for metric, computed in (("r2", 1 - np.dot(error, error) / sst),
                                 ("mae", np.mean(np.abs(error))), ("rmse", np.sqrt(np.mean(error ** 2)))):
            require_close(computed, metrics[key]["field"][metric], f"{key}/pooled/{metric}")
    return metrics, gate, velocity, provenance, radius_audit, ids, arrays, archives, source_hashes


def save_fig(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hexbin_figure(arrays, metrics, output):
    truth = arrays["truth"]
    joint = np.concatenate([truth, *[arrays[key] for key in METHODS]])
    full = max(1.0, float(np.max(joint)) * 1.025)
    zoom = min(full, max(1.0, float(np.quantile(joint, 0.995)) * 1.025))
    fig, axes = plt.subplots(2, 3, figsize=(16.6, 10.5), layout="constrained")
    fig.suptitle("速度派生 WSS：V5 有效全壁面 pooled 密度散点\n三种方法采用相同坐标范围；虚线为预测 = 真值", fontsize=16)
    details = []
    for i, limit in enumerate((full, zoom)):
        collections = []
        for j, key in enumerate(METHODS):
            ax = axes[i, j]
            pred = arrays[key]
            within = (truth <= limit) & (pred <= limit)
            collection = ax.hexbin(truth[within], pred[within], gridsize=88, mincnt=1,
                                   extent=(0, limit, 0, limit), cmap="viridis", linewidths=0)
            collections.append(collection)
            ax.plot([0, limit], [0, limit], "--", color="#d45087", linewidth=1.1)
            ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="CFD WSS 真值（Pa）", ylabel="估计 WSS（Pa）")
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(LABELS[key] + f'\nR²_pool={metrics[key]["field"]["r2"]:.4f}；MAE={metrics[key]["field"]["mae"]:.3f} Pa', fontsize=11)
            else:
                ax.set_title(f"共同 99.5% 分位视窗；显示 {within.mean():.2%} 顶点", fontsize=11)
            details.append({"method": key, "panel": "full" if i == 0 else "zoom",
                            "limit_pa": limit, "n_displayed": int(within.sum()), "n_total": len(truth)})
        maximum = max(2, max(float(c.get_array().max()) for c in collections))
        for collection in collections:
            collection.set_norm(LogNorm(vmin=1, vmax=maximum))
        fig.colorbar(collections[0], ax=list(axes[i]), shrink=0.77, pad=0.02, label="每个六边形内的顶点数（对数色标）")
    save_fig(fig, output, "velocity_wss_pooled_hexbin")
    return {"sampling": "all V5 effective full-wall vertices; no random subsampling", "zoom_quantile": 0.995,
            "zoom_definition": "pooled truth and all three predictions combined; same limit in each column",
            "metrics_use_all_vertices": True, "panels": details}


def case_figure(metrics, ids, output):
    ordered = sorted(ids, key=lambda uid: metrics["test"]["per_case"][uid]["overall"]["r2"])
    fig, ax = plt.subplots(figsize=(14.8, 15.5), layout="constrained")
    y = np.arange(len(ordered))
    all_scores = []
    for i, key in enumerate((*METHODS, "R4", "M2")):
        scores = np.array([metrics[key]["per_case"][uid]["overall"]["r2"] for uid in ordered])
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite per-case R² cannot be plotted")
        all_scores.extend(scores.tolist())
        ax.scatter(scores, y + (i - 2) * 0.12, s=27 if key in METHODS else 19,
                   marker="o" if key in METHODS else "x", label=LABELS[key], color=COLORS[key], alpha=0.88)
    symlog = min(all_scores) < -2
    if symlog:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axvline(1, color="#bbbbbb", linewidth=0.7, linestyle="--")
    ax.set_yticks(y, ordered, fontsize=9)
    ax.set_ylim(len(ordered) - 0.5, -0.7)
    ax.set_xlabel("逐病例物理 R²" + ("（[-1, 1] 线性，其外为对称对数坐标）" if symlog else ""))
    ax.set_title("34 例 V5 有效全壁面 WSS：按 VELWSS1 的 R² 从低到高排列\nR4、M2 为不同架构/训练目标的历史直接 WSS 参考", fontsize=15, pad=16)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.047), ncol=2, frameon=False, fontsize=10)
    save_fig(fig, output, "velocity_wss_case_r2")
    return {"ordering": ordered, "xscale": "symlog, linthresh=1" if symlog else "linear",
            "case_weights": "one point per case and method; no cases omitted"}


def wall_figure(exp, ids, output):
    selected = [next(uid for uid in sorted(ids) if uid.startswith(cohort + "/")) for cohort in ("AAA", "AG", "ILO")]
    fig = plt.figure(figsize=(14.6, 15.5), layout="constrained")
    fig.suptitle("固定三例壁面 WSS：同病例使用同一视角、同一色标\n病例按 AAA / AG / ILO 的 test 顺序首例预选；log1p 色标，刻度为 Pa", fontsize=16)
    evidence = []
    columns = (("truth_wss_pa", "CFD WSS 真值"), ("pred_wss_pa", "VELWSS1 预测速度派生"),
               ("oracle_wss_pa", "CFD 速度 oracle 审计"))
    for i, uid in enumerate(selected):
        path = exp / "predictions" / (uid.replace("/", "__") + ".npz")
        with np.load(path, allow_pickle=False) as z:
            wall = np.asarray(z["wall_mm"], dtype=np.float64)
            values = {key: np.asarray(z[key], dtype=np.float64).reshape(-1) for key, _ in columns}
        centered = wall - wall.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        for axis in range(3):
            if basis[axis, np.argmax(np.abs(basis[axis]))] < 0:
                basis[axis] *= -1
        rotation = basis[[1, 2, 0]].T
        if np.linalg.det(rotation) < 0:
            rotation[:, 1] *= -1
        display = centered @ rotation
        limit = max(0.1, float(np.quantile(np.concatenate(list(values.values())), 0.99)))
        norm = FuncNorm((np.log1p, np.expm1), vmin=0, vmax=limit, clip=True)
        row_axes, row_info = [], []
        for j, (key, label) in enumerate(columns):
            ax = fig.add_subplot(3, 3, i * 3 + j + 1, projection="3d")
            row_axes.append(ax)
            value = values[key]
            ax.scatter(*display.T, c=value, cmap="magma", norm=norm, s=0.7,
                       linewidths=0, depthshade=False, rasterized=True)
            ranges = np.maximum(np.ptp(display, axis=0), 1e-3)
            ax.set_box_aspect(ranges)
            ax.set_proj_type("ortho")
            ax.view_init(elev=10, azim=-72)
            ax.set_axis_off()
            clipped = float(np.mean(value > limit))
            title = label + f"\n>{limit:.2f} Pa 饱和 {clipped:.2%}"
            if j == 0:
                title = uid + "\n" + title
            ax.set_title(title, fontsize=10, pad=2)
            row_info.append({"array": key, "clip_fraction": clipped, "maximum_pa": float(value.max())})
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap="magma")
        # Space labels in the displayed transform to keep low-Pa ticks legible.
        ticks = np.expm1(np.linspace(0.0, np.log1p(limit), 6)).tolist()
        bar = fig.colorbar(mappable, ax=row_axes, shrink=0.62, pad=0.015, ticks=ticks, label="WSS（Pa；log1p 色标）", extend="max")
        bar.ax.set_yticklabels([f"{tick:.2g}" for tick in ticks])
        evidence.append({"uid": uid, "n_points": len(wall), "coordinate_unit": "mm",
            "view": "geometry-only PCA rigid rotation; principal axis vertical; elev=10, azim=-72, orthographic",
            "rotation": rotation.tolist(), "shared_upper_pa": limit,
            "upper_definition": "case pooled truth, prediction-derived, oracle q99; display clipping only",
            "norm": "log1p with Pa tick labels", "panels": row_info})
    save_fig(fig, output, "velocity_wss_wall_fields")
    return {"selection": "first sorted test case in each cohort AAA, AG, ILO; no score-based selection",
            "sampling": "all wall points, no downsampling", "cases": evidence}


def metric_table(metrics, rows, digits=5):
    keys = (*METHODS, "R4", "M2")
    lines = ["| 指标 | VELWSS1 主结果 | CFD 速度 oracle | 未校准物理核 | R4 直接 WSS | M2 直接 WSS |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, path in rows:
        lines.append("| " + label + " | " + " | ".join(fmt(dig(metrics[key], path), digits) for key in keys) + " |")
    return "\n".join(lines)


def velocity_summary(record):
    checks = []
    strict_exceeded = []
    for case in record["cases"].values():
        for group in case["verification"].values():
            checks.extend(group["checks"])
    differences = [c["abs_difference"] for c in checks if c.get("abs_difference") is not None]
    if not checks or not all(c["passed"] for c in checks):
        raise ValueError("Velocity reproduction contains missing/failed numeric checks")
    for c in checks:
        if c.get("abs_difference") is not None:
            strict = 0.0 if isinstance(c["old"], int) else 5e-6 + 5e-6 * abs(c["old"])
            if c["abs_difference"] > strict:
                strict_exceeded.append({"metric": c["metric"], "difference": c["abs_difference"], "strict_tolerance": strict})
    return {"n_checks": len(checks), "n_failed": 0,
            "strict_tolerance_exceeded_checks": len(strict_exceeded), "strict_tolerance_exceeded_details": strict_exceeded,
            "maximum_numeric_abs_difference": max(differences),
            "maximum_coordinate_roundtrip_mm": max(c["coordinate_roundtrip_max_error_mm"] for c in record["cases"].values()),
            "n_interior_total": sum(c["n_interior"] for c in record["cases"].values()),
            "checkpoint_epoch": record["checkpoint_epoch"], "elapsed_seconds": record.get("elapsed_seconds"),
            "reused_prediction_archive": record["reused_prediction_archive"],
            "source_hashes_unchanged": record["source_sha256"] == record["source_sha256_end"]}


def write_report(metrics, gate, velocity, provenance, radius_audit, ids, arrays, archives, output, figure_info):
    rows = [
        ("物理 R²_cb（病例等权）", "field_casebalanced.r2"), ("物理 R²_pool（顶点合并）", "field.r2"),
        ("逐例 R² 均值", "aggregate.r2_casemean"), ("逐例 R² 中位数", "aggregate.r2_casemed"),
        ("逐例 R² 第10百分位", "aggregate.r2_casep10"), ("R²<0 病例数 / 34", "aggregate.r2_negative_cases"),
        ("pooled MAE（Pa）", "field.mae"), ("pooled RMSE（Pa）", "field.rmse"),
        ("逐例 MAE 均值（Pa）", "aggregate.mae_casemean"),
        ("高 WSS 区域 R²", "regional_field.high_wss.r2"),
        ("高 WSS 区域 MAE（Pa）", "regional_field.high_wss.mae"),
        ("真值 top10% 区域幅值比", "calibration.top10_pred_true_ratio"),
        ("p99 预测/真值比", "calibration.p99_pred_true_ratio"),
        ("热点 top10% IoU（病例均值）", "hotspot.top10_iou_casemean"),
        ("V5 有效全壁面 Spearman（病例均值）", "hotspot.spearman_all_casemean"),
        ("log_z R²_cb", "normalized.field_casebalanced.r2"),
        ("log_z R²_pool", "normalized.field.r2"), ("log_z 逐例 R² 均值", "normalized.aggregate.r2_casemean"),
        ("log_z pooled MAE", "normalized.field.mae"), ("log_z pooled RMSE", "normalized.field.rmse"),
    ]
    vs = velocity_summary(velocity)
    if not vs["source_hashes_unchanged"]:
        raise ValueError("Velocity inference source changed during export")
    delta = {key: metrics["test"]["field_casebalanced"]["r2"] - metrics[key]["field_casebalanced"]["r2"]
             for key in ("oracle", "physics_only", "R4", "M2")}
    ordered = figure_info["case_r2"]["ordering"]
    main = metrics["test"]
    radius_summary = radius_audit["radius_summary"]
    split_evidence = radius_audit["split_evidence"]
    contracts = provenance["adapter_contracts"]
    if contracts["status"] != "passed":
        raise ValueError("Frozen algorithm contract validation did not pass")
    compact_evidence = {
        "evaluation_gate": gate,
        "frozen_method": provenance["frozen_sources"]["method"],
        "frozen_model_sha256": provenance["frozen_sources"]["model_sha256"],
        "inference_truth_used": provenance["frozen_sources"]["truth_used_at_inference"],
        "historical_calibration": provenance["calibration"],
        "adapter_contracts": contracts,
        "split": {key: split_evidence[key] for key in (
            "calibrator_train_count", "calibrator_train_unique_count", "calibrator_train_equals_v5_train138",
            "v5_test_count", "calibrator_training_overlap_v5_test")},
    }
    mapping_fraction = radius_summary["distance_above_0p1_mm_total"] / radius_summary["n_wall_targets"]
    excluded = [(case["canonical_id"], case["v5_valid_mask_excluded_count"]) for case in radius_audit["cases"]
                if case.get("v5_valid_mask_excluded_count", 0)]
    if radius_summary["n_v5_effective_wall_targets"] != len(arrays["truth"]):
        raise ValueError("Geometry audit and V5 effective wall counts differ")
    prior_failed_path = Path(archives[0]["path"]).parent.parent / "velocity_reproduction_failed.json"
    prior_failed = read(prior_failed_path) if prior_failed_path.exists() else None
    text = ["# VELWSS1：R5V 预测速度经冻结 Profile-Secant V3 求 WSS", "",
        f"本次完成同一 test34、峰值帧 1162 的 V5 有效全壁面评估，共 {len(arrays['truth']):,} 个壁面顶点。"
        f"主结果物理 R²_cb={main['field_casebalanced']['r2']:.5f}，MAE={main['field']['mae']:.5f} Pa，"
        f"log_z R²_cb={main['normalized']['field_casebalanced']['r2']:.5f}，热点 IoU={main['hotspot']['top10_iou_casemean']:.5f}。", "",
        f"原始 ASCII 共 {radius_summary['n_wall_targets']:,} 个壁面顶点；V5 既有 valid-mask 排除 "
        f"{radius_summary['v5_excluded_original_wall_nodes']} 个点后保留 {len(arrays['truth']):,} 个。"
        "原有排除病例/点数为：" + "；".join(f"{uid}（{count} 个）" for uid, count in excluded) + "。"
        "这是现有 V5 数据口径，本次未新增选点或按预测结果排除点，三种方法与 R4/M2 参考均使用该有效壁面。", "",
        "**本轮忠实保留旧算法的半径解析与映射，包括已发现的历史坐标尺度问题。主结果和 oracle 都受这一共同条件约束，"
        "不能把本次复现通过解释为几何映射正确或几何方案已验收。**", "",
        "## 实验身份与结果口径", "",
        "- **VELWSS1 主成绩**：已训练 R5V best 的预测速度 → 冻结 Profile-Secant V3 → WSS 标量幅值。",
        "- **CFD 速度 oracle**：相同算法改用 CFD 真值速度，用于检查输入速度误差影响；属于审计，不能作为部署成绩或严格数学上界。",
        "- **physics_only**：预测速度经同一冻结物理核、移除已训练幅值校准器；用于定位校准影响。",
        "- **R4 与 M2**：同一 test34、seed1234、best 的直接 WSS 历史参考。R4 是 17D 几何的 L-SA2 配方；M2 另有 25D 输入、近壁面分支和独立 query/3NN。其模型、输入及训练目标与 R5V 不同，横向差值不构成单变量消融。", "",
        "R5V 使用原有 18D 输入、QAD-lite 解码器和速度分量 MSE，输出三分量速度；本次只做冻结推理和 WSS 后处理，没有新增模型训练或 WSS 监督。"
        "冻结 V3 校准器在历史阶段使用过 train138 的 WSS 监督，因此完整流程应称为带既有校准器的速度派生方法。", "",
        "## WSS 指标", "", metric_table(metrics, rows), "",
        "所有指标来源于 metrics.json；物理空间单位为 Pa，log_z 使用既有训练集统计。"
        "R²_cb 与逐例 R² 均值不同；pooled MAE/RMSE 按所有顶点计算。高 WSS 区域按各病例真值 q90 划分，"
        "top10 幅值比与 p99 比遵循现有评估器定义；p99 比越接近 1 越好。", "",
        "[全部聚合数值（完整字段名）](aggregate_metrics_full.md) · [34例详细比较 CSV](per_case_comparison.csv) · [原始完整指标](metrics.json)", "",
        "## 观测与比较", "",
        f"主结果相对 M2 的物理 R²_cb 差值为 {delta['M2']:+.5f}，相对 R4 为 {delta['R4']:+.5f}。"
        f"相对 CFD 速度 oracle 为 {delta['oracle']:+.5f}，相对未校准物理核为 {delta['physics_only']:+.5f}。"
        "这些差值只描述本次冻结管线的实际表现。", "",
        f"主结果有 {main['aggregate']['r2_negative_cases']} / 34 例 R²<0。下表展示全部病例，按主结果 R² 从低到高排列，"
        "以便直接看到失效病例；未按成绩排除 V5 有效壁面顶点或病例。", "",
        "| 病例 | 主结果 R² | oracle R² | 物理核 R² | M2 R² | Δ主结果−M2 | 主结果 MAE Pa | 主结果 IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for uid in ordered:
        pc = main["per_case"][uid]
        score = pc["overall"]["r2"]
        m2_score = metrics["M2"]["per_case"][uid]["overall"]["r2"]
        values = [score, metrics["oracle"]["per_case"][uid]["overall"]["r2"],
                  metrics["physics_only"]["per_case"][uid]["overall"]["r2"], m2_score, score - m2_score,
                  pc["overall"]["mae"], pc["hotspot"]["top10_iou"]]
        text.append("| " + uid + " | " + " | ".join(fmt(x) for x in values) + " |")
    text += ["", "本轮只使用一个既有速度训练 seed，且 test34 已用于前期探索。速度 oracle 的差距可能包含近壁面速度误差、梯度放大与校准器输入分布变化；"
             "这组结果不能将三者单独归因，也不支持稳定性或显著性结论。", "",
        "## 原速度 checkpoint 与预测复现", "",
        f"原 run：`{RUN.relative_to(ROOT)}`，checkpoint=`best`，保存 epoch 字段为 `{vs['checkpoint_epoch']}`（保持原记录编号）。"
        f"速度推理固定 support seed1234，共 {vs['n_interior_total']:,} 个体内速度向量。", "",
        "原实验仅保留指标、未保留逐点预测，因此本次从原 best checkpoint 和冻结 support 协议重新导出预测。"
        "复现与历史指标按每病例物理/归一化完整数值逐项校验；不将这一步描述为直接复用已有逐点存档。" if not vs["reused_prediction_archive"] else
        "本次复用已有逐点速度预测存档，并对齐原 checkpoint 与评估指标。", "",
        f"最终采用 Slurm {velocity['job_id']} 的复现结果，共 {vs['n_checks']:,} 项速度数值检查，失败 0 项；"
        f"超过原严格容差的检查数为 {vs['strict_tolerance_exceeded_checks']}。数值叶最大绝对差 {vs['maximum_numeric_abs_difference']:.8g}"
        "（混合指标单位，仅作复现审计）。原浮点容差为 5e-6 + 5e-6×|原值|，整数计数严格一致；"
        f"坐标往返最大误差 {vs['maximum_coordinate_roundtrip_mm']:.8g} mm。源代码、checkpoint、配置和训练统计哈希前后一致。", "",
        (f"此前一次重放在 `{prior_failed['case']}` 的一个热点交集边界点触发严格容差检查失败，"
         "[失败证据另存](velocity_reproduction_failed.json)。最终上述复现结果单独核验通过；"
         "不把多次 GPU 推理描述为逐位一致，也不将先前失败结果混入本次 WSS 指标。" if prior_failed else
         "GPU 复现按上述数值容差检查，不声称逐位一致。"), "",
        "速度分量由原线性 z 统计还原至 m/s，再用原旋转矩阵变回物理坐标系；壁面、体内坐标使用 mm，梯度核执行 mm→m 换算。"
        "这避免把归一化速度、不同坐标系向量或毫米梯度直接代入 WSS。", "",
        "[逐例速度复现完整证据](velocity_reproduction.json)", "",
        "## 冻结算法与标签隔离证据", "",
        "数组适配器依次使用历史 V1 自适应梯度、V3 法向多尺度、V4 表面 MLS 与 depth3 profile 修正，"
        "再使用原 Carreau 流变和冻结的 84 特征 Profile-Secant V3 校准器。四个梯度阶段显式接收同一输入速度数组，"
        "主预测路径不能在中途换回 CFD 速度。校准按完整病例执行，保留病例内排名/分位特征。", "",
        "梯度构建接口不接收 WSS 标签；校准入口剔除所有 truth 字段。算法来源、固定参数、训练/测试交集审计和"
        "历史 CLI/解析剪切/旋转/分块/标签扰动验证以以下机器记录为准；本报告不把缺失的证据写成通过。", "",
        "```json", json.dumps(compact_evidence, ensure_ascii=False, indent=2), "```", "",
        "[完整算法与数据来源](provenance.json) · [评估准入记录](evaluation_gate.json)", "",
        "## 历史半径映射的已知局限", "",
        "冻结解析器沿用 `bundle_then_wall`：优先从旧 `data_wss_min` bundle 读取中心线半径，并按旧壁面坐标最近邻配到当前物理壁面。"
        "旧 bundle 的 `wall_coords_raw` 部分使用了病例自身的 `unit_factor`，其比例并非 CFD 物理毫米转换所用的 1000；"
        "解析器仅按坐标范数阈值决定是否乘 1000，没有恢复每个病例的真实比例。", "",
        f"独立几何审计覆盖 {radius_summary['n_cases']} 例原始 ASCII 的 {radius_summary['n_wall_targets']:,} 个壁面顶点："
        f"最近邻映射距离最大 {radius_summary['maximum_nearest_mapping_distance_mm']:.5f} mm；"
        f"其中 {radius_summary['distance_above_0p1_mm_total']:,} 个顶点（{mapping_fraction:.3%}）距离超过 0.1 mm。"
        "这些距离是旧半径来源坐标与目标壁面的配对残差，不能当作合理的局部插值误差忽略。", "",
        "本轮主结果、CFD 速度 oracle 和未校准物理核使用相同旧半径处理，未按预测成绩修正半径或重新计算中心线。"
        "该设置保证对旧算法的忠实复用，却保留了它的几何限制；如要修正半径映射，应另列实验并先完成用户要求的几何可视化审核。", "",
        "[逐病例半径与训练/测试划分审计](radius_split_audit.json)", "",
        "## 可视化", "",
        "壁面图固定选择每个队列按 test 顺序的首例（AAA、AG、ILO），不按预测成绩挑选。每例三个面板使用同一刚性几何 PCA 视角、"
        "同一 log1p 色标；色条刻度仍为 Pa。该例真值、主预测及 oracle 合并的 p99 用作共同上限，"
        "每面板标明超过上限的顶点比例。几何 PCA 只用于调整展示视角，不改变输入特征或计算结果。", "",
        "![固定三例同视角壁面 WSS](velocity_wss_wall_fields.png)", "",
        "下面的散点图使用 V5 有效全壁面顶点生成 pooled hexbin。上排显示三种方法共同全范围；下排使用真值与三组预测合并后的"
        "第99.5百分位确定共同视窗，并标注各面板实际显示比例。视窗裁切只影响展示，所有报告指标保留全部顶点。", "",
        "![V5 有效全壁面 pooled 密度散点](velocity_wss_pooled_hexbin.png)", "",
        "逐例图显示三条速度派生路径及 R4/M2 参考。若存在极端负 R²，横轴在 [-1,1] 外采用对称对数，"
        "所有病例和异常值仍保留，精确数值见上表与 CSV。", "",
        "![34例 R² 对照](velocity_wss_case_r2.png)", "",
        "[壁面图 PDF](velocity_wss_wall_fields.pdf) · [散点图 PDF](velocity_wss_pooled_hexbin.pdf) · [逐例图 PDF](velocity_wss_case_r2.pdf) · [图表来源及范围](figure_provenance.json)", "",
        "## 复现报告", "", "```bash", "/public/newhome/cy/.conda/envs/GNN/bin/python -m training_wss_min.tools.report_velocity_to_wss", "```", "",
        "生成器只读取已通过准入的结果并生成报告，不触发训练、重新推理、跟踪文档或工作簿修改。", ""]
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")
    fields = sorted(set(path for key in (*METHODS, "R4", "M2") for path, _ in scalar_leaves(metrics[key])))
    (output / "aggregate_metrics_full.md").write_text("# 全部聚合数值\n\n原始字段名保留；未定义数值记为未定义。\n\n" +
        metric_table(metrics, [("`" + field + "`", field) for field in fields], digits=9) + "\n", encoding="utf-8")
    csv_rows = []
    for uid in ordered:
        row = {"case": uid}
        for key in (*METHODS, "R4", "M2"):
            for space, pc in (("physical", metrics[key]["per_case"][uid]),
                              ("log_z", metrics[key]["normalized"]["per_case"][uid])):
                row.update({f"{key}.{space}.{name}": value for name, value in scalar_leaves(pc)})
        row["delta_main_minus_m2_r2"] = row["test.physical.overall.r2"] - row["M2.physical.overall.r2"]
        csv_rows.append(row)
    columns = ["case", *sorted(set().union(*(r.keys() for r in csv_rows)) - {"case"})]
    with (output / "per_case_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(output / "report_summary.json", {"experiment": "VELWSS1", "n_cases": 34,
        "n_wall": len(arrays["truth"]), "checkpoint": "best", "velocity_reproduction": vs,
        "historical_geometry_limitation": {k: v for k, v in radius_summary.items() if k != "anomalies"},
        "physical_r2_cb_difference_main_minus_reference": delta,
        "reference_metrics": {k: str(v) for k, v in REFERENCES.items()},
        "summary": {key: {path: dig(metrics[key], path) for _, path in rows} for key in (*METHODS, "R4", "M2")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXP)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    exp = args.experiment_dir.resolve()
    output = (args.output_dir or exp).resolve()
    metrics, gate, velocity, provenance, radius_audit, ids, arrays, archives, hashes = load_inputs(exp)
    output.mkdir(parents=True, exist_ok=True)
    font = configure_font()
    figure_info = {"hexbin": hexbin_figure(arrays, metrics, output), "case_r2": case_figure(metrics, ids, output),
                   "wall_fields": wall_figure(exp, ids, output)}
    write_report(metrics, gate, velocity, provenance, radius_audit, ids, arrays, archives, output, figure_info)
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("Report source changed during generation")
    files = ["README.md", "aggregate_metrics_full.md", "per_case_comparison.csv", "report_summary.json",
             "velocity_wss_pooled_hexbin.png", "velocity_wss_pooled_hexbin.pdf", "velocity_wss_case_r2.png", "velocity_wss_case_r2.pdf",
             "velocity_wss_wall_fields.png", "velocity_wss_wall_fields.pdf"]
    write_json(output / "figure_provenance.json", {"passed": True, "source_sha256": hashes,
        "predictions": archives, "n_cases": 34, "font": font, "reporter_sha256": sha(Path(__file__)),
        "figures": figure_info, "outputs_sha256": {name: sha(output / name) for name in files},
        "archive_metrics_checks": "Each case/method and pooled physical R2, MAE, RMSE recomputed from arrays; atol/rtol 2e-6"})
    print(json.dumps({"output": str(output), "n_cases": 34, "n_wall": len(arrays["truth"]),
                      "main_physical_r2_cb": metrics["test"]["field_casebalanced"]["r2"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
