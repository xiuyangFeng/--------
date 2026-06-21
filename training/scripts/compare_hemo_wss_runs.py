"""多 run 壁面 WSS（及病例级 TAWSS）一致性对比（任务 A / Line W 前置）

在已通过 ``predict_field`` 生成 ``manifest.json`` 与各 ``.pt`` 的前提下，对每个 run：

1. （可选）调用 ``export_hemo`` 生成 ``hemo_ai`` / ``hemo_cfd``；
2. 以 **第一条 run 的 hemo_cfd** 为参考真值，将各 run 的 **hemo_ai** 中壁面点级 ``WSS`` 与参考对齐；
3. 汇总 **RMSE / MAE / R²**（点级 WSS），以及 **病例级 TAWSS_mean 的 Pearson / Spearman**。

**前提**：
- 各 manifest 应对 **相同子集、相同样本列表**（例如均为 ``predictions_test``），否则脚本会报错退出。
- 所有 run 必须使用 **相同数据版本**（相同预处理参数生成的 ``.pt`` 文件），以保证各 run
  的 ``y_true`` 与第一条 run 的 ``hemo_cfd`` 一致。不同数据版本的 run 混合对比会产生
  虚假的 CFD 参考偏差，结果不可信。
- 预测产物中需包含 ``positions``（坐标）和 ``edge_index``（图结构），才能启用基于速度梯度的
  正确 WSS 计算（``method=gradient``）。若缺失，将回退到速度幅值 proxy，**物理不正确**。

用法示例
--------
::

    # 已在仓库根目录 GNN 下，且已对各 run 跑过 predict_field；首次对比可加 --export

    python -m training.scripts.compare_hemo_wss_runs \\
        --run A-Opt-03:outputs/field/<run_dir_03>/predictions_test/manifest.json \\
        --run A-Opt-04:outputs/field/<run_dir_04>/predictions_test/manifest.json \\
        --run A-Opt-05:outputs/field/<run_dir_05>/predictions_test/manifest.json \\
        --export \\
        --output-dir outputs/field/plots/optimization/wss_A_Opt03_04_05_seed1

    # 试跑（只导前 20 个样本，快很多；正式论文请去掉 --max-items）
    python -m training.scripts.compare_hemo_wss_runs ... --export --max-items 20 ...

若 hemo 已生成，可去掉 ``--export``，仅读 CSV 汇总。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover
    pearsonr = None  # type: ignore[misc, assignment]
    spearmanr = None  # type: ignore[misc, assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_export_hemo(manifest: Path, source: str, max_items: int = 0) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "training.scripts.export_hemo",
        "--manifest",
        str(manifest.resolve()),
        "--source",
        source,
    ]
    if max_items and max_items > 0:
        cmd.extend(["--max-items", str(max_items)])
    env = os.environ.copy()
    # 避免部分集群上子进程出现「mkl-service + INTEL 线程层与 libgomp 不兼容」导致 import 失败
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def _load_manifest_samples(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") or []
    return sorted(str(it["sample_id"]) for it in items)


def _ensure_same_manifests(manifests: Sequence[Path]) -> None:
    if len(manifests) < 2:
        return
    ref = _load_manifest_samples(manifests[0])
    for p in manifests[1:]:
        cur = _load_manifest_samples(p)
        if cur != ref:
            raise SystemExit(
                f"manifest 样本列表不一致:\n  基准 {manifests[0]} 共 {len(ref)} 条\n  当前 {p} 共 {len(cur)} 条\n"
                "请保证同一 split、同一 --subset 导出的 predictions。"
            )


def _read_per_node_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _read_per_case_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _node_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        row["patient_id"],
        row["phase"],
        row["time_step"],
        row["node_index"],
    )


def _build_wss_lookup(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str, str], float]:
    out: Dict[Tuple[str, str, str, str], float] = {}
    for row in rows:
        out[_node_key(row)] = float(row["WSS"])
    return out


def _warn_if_proxy_method(rows: List[Dict[str, str]], source_label: str = "") -> None:
    """若 per_node_metrics.csv 中存在 method=proxy 的行，发出物理警告。

    ``export_hemo`` 当前版本未将 ``method`` 字段写入 CSV，故此处仅在字段存在时检查。
    未来若 CSV 包含该字段，可自动检测并告警。
    """
    if not rows:
        return
    if "method" not in rows[0]:
        return
    proxy_count = sum(1 for r in rows if r.get("method") == "proxy")
    if proxy_count > 0:
        warnings.warn(
            f"{source_label}：{proxy_count}/{len(rows)} 行 WSS 使用了 proxy 模式"
            "（壁面速度幅值，物理不正确）。"
            " 请确认 predict_field 导出时保存了 positions 和 edge_index。",
            stacklevel=3,
        )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


_MISSING_REF_WARN_THRESHOLD = 0.05  # 超过 5% 节点键未匹配时发出警告


def _metrics_vs_ref(
    ai_rows: List[Dict[str, str]],
    wss_ref: Dict[Tuple[str, str, str, str], float],
    label: str = "",
) -> Dict[str, float]:
    y_true_list: List[float] = []
    y_pred_list: List[float] = []
    missing_ref = 0
    for row in ai_rows:
        k = _node_key(row)
        if k not in wss_ref:
            missing_ref += 1
            continue
        y_pred_list.append(float(row["WSS"]))
        y_true_list.append(wss_ref[k])
    if not y_true_list:
        raise RuntimeError("无可用 WSS 对：请检查 hemo_ai 与参考 hemo_cfd 是否匹配。")
    total = len(y_true_list) + missing_ref
    if total > 0 and missing_ref / total > _MISSING_REF_WARN_THRESHOLD:
        warnings.warn(
            f"[{label}] {missing_ref}/{total} ({missing_ref / total:.1%}) 个节点键未能匹配参考 CFD。"
            " 可能原因：不同数据版本导致节点索引不一致，或 wall_mask 不同。"
            " 建议确认各 run 使用相同预处理版本的 .pt 文件，当前指标基于不完整配对计算。",
            stacklevel=2,
        )
    yt = np.asarray(y_true_list, dtype=np.float64)
    yp = np.asarray(y_pred_list, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    r2 = _r2_score(yt, yp)
    return {
        "n_pairs": float(len(y_true_list)),
        "missing_ref_keys": float(missing_ref),
        "missing_ref_ratio": float(missing_ref / total) if total > 0 else 0.0,
        "rmse_wss": rmse,
        "mae_wss": mae,
        "r2_wss": r2,
    }


def _case_tawss_table(
    rows: List[Dict[str, str]],
    region_name: str = "wall",
) -> Dict[Tuple[str, str], float]:
    """从 per_case_region_metrics.csv 中提取指定 region 的 TAWSS_mean。

    只保留 ``region_name`` 匹配的行，避免未来新增分区（bifurcation / near-wall 等）后
    同一病例多行互相覆盖。
    """
    out: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if row.get("region_name", "wall") != region_name:
            continue
        key = (row["patient_id"], row["phase"])
        out[key] = float(row["TAWSS_mean"])
    return out


def _case_correlations(
    ai_cases: List[Dict[str, str]],
    ref_cases: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    xs: List[float] = []
    ys: List[float] = []
    for row in ai_cases:
        key = (row["patient_id"], row["phase"])
        if key not in ref_cases:
            continue
        xs.append(ref_cases[key])
        ys.append(float(row["TAWSS_mean"]))
    if len(xs) < 2 or pearsonr is None:
        return {
            "n_cases": float(len(xs)),
            "pearson_tawss_mean": float("nan"),
            "spearman_tawss_mean": float("nan"),
        }
    pr, _ = pearsonr(xs, ys)
    sr, _ = spearmanr(xs, ys)
    return {
        "n_cases": float(len(xs)),
        "pearson_tawss_mean": float(pr),
        "spearman_tawss_mean": float(sr),
    }


def _parse_run_arg(s: str) -> Tuple[str, Path]:
    if ":" not in s:
        raise SystemExit(f'--run 格式应为 LABEL:manifest.json 路径，收到: {s!r}')
    label, path = s.split(":", 1)
    label = label.strip()
    path = Path(path.strip()).expanduser()
    if not label:
        raise SystemExit(f'--run 缺少 LABEL: {s!r}')
    if not path.is_file():
        raise SystemExit(f'manifest 不存在: {path}')
    return label, path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="多 run WSS / TAWSS 与 CFD 参考对比")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL:MANIFEST",
        help="可重复传入；顺序决定控制台输出顺序，**第一条**的 hemo_cfd 作为全表 WSS 真值参考。",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="若缺 hemo_ai/hemo_cfd 则调用 training.scripts.export_hemo（各 manifest 各两次）。",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="写入 summary.json 与 per_run_metrics.csv；默认不写盘仅打印。",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="传给 export_hemo：只处理各 manifest 前 N 条（快速试跑）；0=全量。",
    )
    args = parser.parse_args()

    runs: List[Tuple[str, Path]] = [_parse_run_arg(x) for x in args.run]
    labels = [r[0] for r in runs]
    manifests = [r[1] for r in runs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"--run 的 LABEL 必须唯一，收到: {labels}")

    _ensure_same_manifests(manifests)

    for _label, manifest in runs:
        pred_dir = manifest.parent
        ai_csv = pred_dir / "hemo_ai" / "per_node_metrics.csv"
        cfd_csv = pred_dir / "hemo_cfd" / "per_node_metrics.csv"
        if args.export or not ai_csv.is_file():
            _run_export_hemo(manifest, "AI", max_items=args.max_items)
        if args.export or not cfd_csv.is_file():
            _run_export_hemo(manifest, "CFD", max_items=args.max_items)

    ref_manifest = manifests[0]
    ref_pred_dir = ref_manifest.parent
    ref_hemo_cfd_dir = ref_pred_dir / "hemo_cfd"
    ref_per_node = ref_hemo_cfd_dir / "per_node_metrics.csv"
    ref_per_case = ref_hemo_cfd_dir / "per_case_region_metrics.csv"

    if not ref_per_node.is_file():
        raise SystemExit(f"参考 CFD per_node 不存在: {ref_per_node}")

    ref_fieldnames, ref_rows = _read_per_node_csv(ref_per_node)
    _warn_if_proxy_method(ref_rows, source_label=f"参考 CFD [{labels[0]}]")
    wss_ref = _build_wss_lookup(ref_rows)

    ref_case_rows = _read_per_case_csv(ref_per_case) if ref_per_case.is_file() else []
    tawss_ref = _case_tawss_table(ref_case_rows) if ref_case_rows else {}

    summary: Dict[str, object] = {
        "reference": {
            "label": labels[0],
            "manifest": str(ref_manifest),
            "hemo_cfd_dir": str(ref_hemo_cfd_dir),
            "n_ref_nodes": len(ref_rows),
            "n_ref_cases": len(ref_case_rows),
        },
        "runs": [],
    }

    table_rows: List[Dict[str, object]] = []

    for label, manifest in runs:
        pred_dir = manifest.parent
        hemo_ai_dir = pred_dir / "hemo_ai"
        ai_node = hemo_ai_dir / "per_node_metrics.csv"
        ai_case = hemo_ai_dir / "per_case_region_metrics.csv"

        if not ai_node.is_file():
            raise SystemExit(f"缺少 {ai_node}，可加 --export 或先手工运行 export_hemo")

        _, ai_rows = _read_per_node_csv(ai_node)
        _warn_if_proxy_method(ai_rows, source_label=f"hemo_ai [{label}]")
        node_metrics = _metrics_vs_ref(ai_rows, wss_ref, label=label)

        case_block: Dict[str, object] = {}
        if ai_case.is_file() and tawss_ref:
            ai_case_rows = _read_per_case_csv(ai_case)
            case_block = _case_correlations(ai_case_rows, tawss_ref)
        else:
            case_block = {
                "n_cases": 0.0,
                "pearson_tawss_mean": float("nan"),
                "spearman_tawss_mean": float("nan"),
            }

        run_entry = {
            "label": label,
            "manifest": str(manifest),
            "hemo_ai_dir": str(hemo_ai_dir),
            **node_metrics,
            **case_block,
        }
        summary["runs"].append(run_entry)  # type: ignore[index]
        table_rows.append(run_entry)

        missing_ratio = node_metrics.get("missing_ref_ratio", 0.0)
        missing_flag = f"  ⚠ missing={missing_ratio:.1%}" if missing_ratio > 0 else ""
        print(
            f"[{label}] WSS  n={int(node_metrics['n_pairs'])}{missing_flag}  "
            f"RMSE={node_metrics['rmse_wss']:.6g}  MAE={node_metrics['mae_wss']:.6g}  "
            f"R²={node_metrics['r2_wss']:.6g}  |  "
            f"TAWSS_mean  cases={int(case_block['n_cases'])}  "
            f"Pearson={case_block['pearson_tawss_mean']:.6g}  "
            f"Spearman={case_block['spearman_tawss_mean']:.6g}"
        )

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "wss_compare_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        csv_path = out_dir / "per_run_wss_metrics.csv"
        if table_rows:
            keys = list(table_rows[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in table_rows:
                    w.writerow(row)
        print(f"已写入: {summary_path}\n        {csv_path}")


if __name__ == "__main__":
    main()
