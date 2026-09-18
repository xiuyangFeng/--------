"""Read-only audit of outlet naming evidence used by the confidence gate.

This module deliberately does not fit a calibration model.  It summarizes an
existing, labelled acceptance export and reports whether its records contain
the metadata needed for an independently validated profile.  It can therefore
be rerun when a new acceptance set is archived without changing a deployment
job or silently turning a test set into a calibration set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def calibration_metrics(confidences: Any, outcomes: Any, *, bins: int = 10) -> dict[str, Any]:
    """Compute Brier/ECE and reliability bins for an independent holdout.

    Inputs are intentionally explicit arrays: callers must provide predictions
    and binary correctness labels from a *different* labelled set than the
    one used to fit a profile.  No threshold or profile is fitted here.
    """
    import numpy as np
    p = np.asarray(confidences, dtype=float).reshape(-1)
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    if p.size == 0 or p.size != y.size or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("confidences/outcomes must be finite, non-empty arrays of equal length")
    if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("confidences and outcomes must lie in [0, 1]")
    bins = max(1, int(bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    records = []
    ece = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1])
        count = int(mask.sum())
        if not count:
            continue
        mean_p, mean_y = float(p[mask].mean()), float(y[mask].mean())
        ece += count / p.size * abs(mean_p - mean_y)
        records.append({"lower": float(edges[index]), "upper": float(edges[index + 1]),
                        "count": count, "mean_confidence": mean_p, "empirical_accuracy": mean_y})
    return {"n": int(p.size), "brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
            "bins": records}


def audit_acceptance(root: Path) -> dict[str, Any]:
    root = Path(root)
    acceptance_path = root / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.is_file() else {}
    rows = list(acceptance.get("rows") or [])
    stages = sorted(root.glob("jobs/*/stage_a.json"))
    stage_records = []
    for path in stages:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        proposal = data.get("proposal") or {}
        stage_records.append({
            "case": path.parent.name,
            "confidence": proposal.get("confidence"),
            "gate": proposal.get("confidence_gate"),
            "orientation_source": (data.get("input_check") or {}).get("orientation_source"),
            "flags": list(proposal.get("flags") or []),
        })

    names_total = names_correct = 0
    for row in rows:
        for item in row.get("names") or []:
            if len(item) >= 2:
                names_total += 1
                names_correct += int(item[0] == item[1])
    confidences = [float(item["confidence"]) for item in stage_records
                   if isinstance(item.get("confidence"), (int, float))]
    orientation_sources = sorted({item.get("orientation_source") for item in stage_records
                                  if item.get("orientation_source")})
    return {
        "schema_version": "wss-deploy.confidence-audit/v1",
        "source": str(root),
        "acceptance_rows": len(rows),
        "stage_a_records": len(stage_records),
        "naming": {"correct": names_correct, "total": names_total,
                   "accuracy": (names_correct / names_total if names_total else None)},
        "geometry_proxy": {"records": len(confidences), "min": min(confidences) if confidences else None,
                            "median": sorted(confidences)[len(confidences) // 2] if confidences else None,
                            "max": max(confidences) if confidences else None},
        "metadata_coverage": {
            "confidence_proxy": len(confidences),
            "confidence_gate": sum(item.get("gate") is not None for item in stage_records),
            "orientation_source": len(orientation_sources),
            "orientation_values": orientation_sources,
        },
        "calibration_status": "not_calibrated",
        "automatic_bypass_allowed": False,
        "limitations": [
            "现有 acceptance_test34 仅用于回顾，不能将同一数据集拟合为独立的 95% 校准 profile。",
            "历史 stage_a.json 不含 release 绑定、联合正确率下界和患者方向元数据。",
            "几何 margin 只能排序候选，不是经过校准的正确率概率。",
        ],
    }


def write_audit(root: Path, output: Path) -> dict[str, Any]:
    report = audit_acceptance(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_audit(args.root, args.out), ensure_ascii=False, indent=2))
