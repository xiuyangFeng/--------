from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch

from ..analysis.hemo import (
    HemoSample,
    build_per_case_region_rows,
    build_per_node_rows,
    build_risk_feature_rows,
    parse_sample_id,
)
from ..core.utils import ensure_dir


def load_prediction_manifest(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(manifest_path: str | Path, source_field: str) -> List[HemoSample]:
    # source=AI 时取 y_pred；source=CFD 时取同一文件里的 y_true。
    # 这样一套导出结果就能同时支撑“预测指标”和“真值指标”两条链路。
    manifest = load_prediction_manifest(manifest_path)
    samples: List[HemoSample] = []
    for item in manifest["items"]:
        prediction = torch.load(item["prediction_path"], map_location="cpu")
        patient_id, phase, time_step = parse_sample_id(prediction["sample_id"])
        y_field = prediction["y_pred"] if source_field == "AI" else prediction["y_true"]
        x = prediction.get("x")
        positions = None
        if x is not None:
            positions = x[:, :3]
        samples.append(
            HemoSample(
                sample_id=prediction["sample_id"],
                patient_id=patient_id,
                phase=phase,
                time_step=time_step,
                source=source_field,
                model_name=manifest.get("model_name", "unknown"),
                split_version=manifest.get("split_version", "unknown"),
                wall_mask=prediction["wall_mask"].bool(),
                y_field=y_field,
                positions=positions,
                edge_index=prediction.get("edge_index"),
            )
        )
    return samples


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="任务B指标导出骨架")
    parser.add_argument("--manifest", required=True, help="training.scripts.predict_field 输出的 manifest.json")
    parser.add_argument(
        "--source",
        default="AI",
        choices=["AI", "CFD"],
        help="使用预测场还是 CFD 真值场来计算指标",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出目录，默认保存到 manifest 同级 hemo_<source>",
    )
    args = parser.parse_args()

    output_dir = (
        ensure_dir(args.output)
        if args.output
        else ensure_dir(Path(args.manifest).resolve().parent / f"hemo_{args.source.lower()}")
    )

    samples = load_samples(args.manifest, args.source)
    if not samples:
        raise ValueError("manifest 中没有可用预测样本")

    model_name = samples[0].model_name
    split_version = samples[0].split_version

    per_node_rows = build_per_node_rows(
        samples=samples,
        source=args.source,
        model_name=model_name,
        split_version=split_version,
    )
    per_case_rows = build_per_case_region_rows(
        samples=samples,
        source=args.source,
        model_name=model_name,
        split_version=split_version,
    )
    risk_rows = build_risk_feature_rows(per_case_rows)

    write_csv(output_dir / "per_node_metrics.csv", per_node_rows)
    write_csv(output_dir / "per_case_region_metrics.csv", per_case_rows)
    write_csv(output_dir / "risk_features.csv", risk_rows)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": args.source,
                "input_manifest": str(Path(args.manifest).resolve()),
                "output_dir": str(output_dir),
                "model_name": model_name,
                "split_version": split_version,
                "num_samples": len(samples),
                "files": {
                    "per_node_metrics": str(output_dir / "per_node_metrics.csv"),
                    "per_case_region_metrics": str(output_dir / "per_case_region_metrics.csv"),
                    "risk_features": str(output_dir / "risk_features.csv"),
                },
                "notes": [
                    "当前版本优先使用基于壁面法向和邻近内部点速度梯度的 WSS 计算。",
                    "若预测产物缺少坐标或边信息，会自动回退到 velocity magnitude proxy 以保持兼容。",
                    "区域划分仍为骨架版，后续可继续细化 wall / bifurcation / near-wall 等区域规则。",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"任务B指标骨架导出完成: {output_dir}")


if __name__ == "__main__":
    main()
