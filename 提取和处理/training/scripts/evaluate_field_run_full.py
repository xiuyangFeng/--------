"""Run the full post-training evaluation pipeline for one field-model run.

The script is a thin orchestrator around the existing Task A tools plus the
direct WSS-head credibility report. It keeps direct WSS-head evaluation separate
from hemodynamic metrics derived from the predicted velocity field.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from ..core.utils import dump_json, ensure_dir


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: List[str], *, cwd: Path = REPO_ROOT) -> None:
    print("[eval-full]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 结构非法: {path}")
    return data


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "auto":
        best_wss = run_dir / "best_wss_model.pt"
        if best_wss.is_file():
            return best_wss
        return run_dir / "best_model.pt"
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = run_dir / ckpt
    return ckpt.resolve()


def _default_predictions_dir(run_dir: Path, subset: str, checkpoint_path: Path) -> Path:
    if checkpoint_path.name == "best_model.pt":
        return run_dir / f"predictions_{subset}"
    stem = checkpoint_path.stem.replace("_model", "")
    return run_dir / f"predictions_{subset}_{stem}"


def main() -> None:
    parser = argparse.ArgumentParser(description="训练完成后对单个 run 执行完整评估")
    parser.add_argument("--run-dir", required=True, type=Path, help="run 目录")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="checkpoint 文件名/路径；auto 优先 best_wss_model.pt，否则 best_model.pt",
    )
    parser.add_argument("--subset", default="test", choices=["train", "val", "test"])
    parser.add_argument("--predictions-dir", default="", help="预测输出目录；默认按 checkpoint 自动命名")
    parser.add_argument("--evaluation-dir", default="", help="完整评估输出目录")
    parser.add_argument("--force", action="store_true", help="删除并重建 predictions/evaluation 输出")
    parser.add_argument("--skip-predict", action="store_true", help="跳过 predict_field，复用已有 manifest")
    parser.add_argument("--skip-field-plots", action="store_true", help="跳过 A3/A4/error/regional 图件")
    parser.add_argument("--skip-wss-direct", action="store_true", help="跳过 WSS head 直接可信性评估")
    parser.add_argument(
        "--run-derived-hemo",
        action="store_true",
        help="额外运行 export_hemo 的 AI/CFD 派生血流指标；注意这评估的是速度场导出的 WSS",
    )
    parser.add_argument(
        "--clinical-pa",
        action="store_true",
        help="额外运行 Pa 量纲临床指标（eval_wss_clinical_metrics，纯后处理）",
    )
    parser.add_argument("--max-scatter-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "config.snapshot.json"
    if not cfg_path.is_file():
        raise SystemExit(f"缺少配置快照: {cfg_path}")
    checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"缺少 checkpoint: {checkpoint_path}")

    pred_dir = (
        Path(args.predictions_dir).resolve()
        if args.predictions_dir
        else _default_predictions_dir(run_dir, args.subset, checkpoint_path)
    )
    eval_dir = (
        Path(args.evaluation_dir).resolve()
        if args.evaluation_dir
        else run_dir / "evaluation" / f"{args.subset}_{checkpoint_path.stem}"
    )

    if args.force:
        if pred_dir.is_dir() and not args.skip_predict:
            shutil.rmtree(pred_dir)
        if eval_dir.is_dir():
            shutil.rmtree(eval_dir)
    ensure_dir(eval_dir)

    manifest_path = pred_dir / "manifest.json"
    if not args.skip_predict:
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.predict_field",
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(checkpoint_path),
                "--subset",
                args.subset,
                "--output",
                str(pred_dir),
            ]
        )
    if not manifest_path.is_file():
        raise SystemExit(f"缺少预测 manifest: {manifest_path}")

    outputs: Dict[str, object] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "subset": args.subset,
        "predictions_dir": str(pred_dir),
        "manifest": str(manifest_path),
        "evaluation_dir": str(eval_dir),
    }

    if not args.skip_field_plots:
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.plot_taskA_scatter",
                "--manifest",
                str(manifest_path),
                "--output",
                str(eval_dir / "fig_A3_scatter_interior.png"),
                "--max-points",
                str(args.max_scatter_points),
                "--seed",
                str(args.seed),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.plot_taskA_per_case_boxplot",
                "--manifest",
                str(manifest_path),
                "--output",
                str(eval_dir / "fig_A4_per_case_boxplot_interior.png"),
                "--metrics-output",
                str(eval_dir / "fig_A4_per_case_metrics_interior.csv"),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.plot_error_analysis",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(eval_dir / "error_analysis_interior"),
                "--max-scatter-points",
                str(args.max_scatter_points),
                "--seed",
                str(args.seed),
                "--wss",
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.plot_taskA_regional_bar",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(eval_dir / "regional_eval"),
                "--wss",
            ]
        )
        outputs["field_plots"] = {
            "scatter": str(eval_dir / "fig_A3_scatter_interior.png"),
            "per_case_boxplot": str(eval_dir / "fig_A4_per_case_boxplot_interior.png"),
            "error_analysis": str(eval_dir / "error_analysis_interior"),
            "regional_eval": str(eval_dir / "regional_eval"),
        }

    wss_summary_path = eval_dir / "wss_direct" / "wss_credibility_summary.json"
    if not args.skip_wss_direct:
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.evaluate_wss_credibility",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(eval_dir / "wss_direct"),
                "--max-scatter-points",
                str(args.max_scatter_points),
                "--seed",
                str(args.seed),
            ]
        )
        outputs["wss_direct"] = str(wss_summary_path)

    if args.clinical_pa:
        cfg = _load_json(cfg_path)
        data_root = Path(str(cfg.get("data", {}).get("data_root", "data_new/AG")))
        norm_path = data_root / "normalization_params_global.json"
        if not norm_path.is_file():
            raise SystemExit(f"缺少归一化参数: {norm_path}")
        frame_tag = str(cfg.get("data", {}).get("wss_target_frame", "global"))
        if frame_tag == "local":
            frame_tag = "local_v1"
        pa_dir = eval_dir / "wss_clinical_pa"
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.eval_wss_clinical_metrics",
                "--manifest",
                str(manifest_path),
                "--norm-params",
                str(norm_path),
                "--output-dir",
                str(pa_dir),
                "--frame-tag",
                frame_tag,
            ]
        )
        outputs["wss_clinical_pa"] = str(pa_dir / "wss_pa_summary.json")

    if args.run_derived_hemo:
        hemo_ai = eval_dir / "hemo_derived_ai"
        hemo_cfd = eval_dir / "hemo_derived_cfd"
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.export_hemo",
                "--manifest",
                str(manifest_path),
                "--source",
                "AI",
                "--output",
                str(hemo_ai),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "training.scripts.export_hemo",
                "--manifest",
                str(manifest_path),
                "--source",
                "CFD",
                "--output",
                str(hemo_cfd),
            ]
        )
        outputs["hemo_derived"] = {
            "ai": str(hemo_ai),
            "cfd": str(hemo_cfd),
            "note": "Derived from velocity fields; separate from direct WSS-head evaluation.",
        }

    if wss_summary_path.is_file():
        wss_summary = _load_json(wss_summary_path)
        outputs["wss_quick_view"] = wss_summary.get("quick_view", {})

    summary_path = eval_dir / "evaluation_summary.json"
    dump_json(outputs, summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()
