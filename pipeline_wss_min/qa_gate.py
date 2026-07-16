#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSS-min 数据划分范围与正式纳入病例质量门。

正式流程只允许划分中的训练、验证和测试病例进入预处理、统计与训练。已排除或
待定病例即使磁盘上存在历史 bundle，也只记录为“不参与正式流程”。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from . import config as C
from . import reporting


LIMITS = {
    "peak_zero_frac": 0.01,
    "all_zero_frac": 0.01,
    "peak_p90": 0.0,
    "n_wall": 50000,
    "wall_crop_frac": 0.05,
    "trunk_centering_offset_frac": 0.05,
}


def _load_report(cohort: str, case: str, out_root: str | Path | None = None) -> Dict:
    path = C.out_case_dir(cohort, case, out_root=out_root) / "report.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _case_label(cohort: str, case: str) -> str:
    return f"{cohort.split('/', 1)[1]}/{case}"


def _iter_partition_cases(split_name: str, partitions: Iterable[str]):
    for part in partitions:
        for label in C.split_case_labels(split_name, (part,)):
            subset, case = label.split("/", 1)
            yield part, f"AG/{subset}", case


def audit_split(
    split_name: str = C.DEFAULT_SPLIT_NAME,
    partitions: tuple[str, ...] = ("train", "val", "test"),
    strict: bool = True,
    out_root: str | Path | None = None,
) -> Dict:
    log = reporting.get_logger()
    rows: List[Dict] = []
    failures: List[str] = []
    warnings: List[str] = []

    for part, cohort, case in _iter_partition_cases(split_name, partitions):
        label = _case_label(cohort, case)
        bundle_path = C.out_case_dir(cohort, case, out_root=out_root) / "bundle.npz"
        report = _load_report(cohort, case, out_root=out_root)
        row: Dict = {
            "partition": part,
            "cohort": cohort,
            "case": case,
            "label": label,
            "bundle_exists": bundle_path.is_file(),
            "status": "ok",
            "fail_reasons": "",
            "warning_reasons": "",
        }
        fatal_reasons: List[str] = []
        warn_reasons: List[str] = []
        if not bundle_path.is_file():
            fatal_reasons.append("missing_bundle")
        else:
            with np.load(bundle_path, allow_pickle=True) as d:
                steps = d["steps"].tolist()
                peak = int(d["peak_step"])
                si = steps.index(peak)
                wss = d["wall_wss"].astype(np.float64)
                peak_wss = wss[si]
                finite = np.isfinite(wss)
                peak_finite = np.isfinite(peak_wss)
                row.update(
                    n_wall=int(wss.shape[1]),
                    n_steps=int(wss.shape[0]),
                    peak_step=peak,
                    all_zero_frac=float((wss[finite] <= 0).mean()) if finite.any() else float("nan"),
                    peak_zero_frac=float((peak_wss[peak_finite] <= 0).mean()) if peak_finite.any() else float("nan"),
                    peak_p90=float(np.nanpercentile(peak_wss, 90)),
                    wss_nonfinite=int((~finite).sum()),
                    coords_nonfinite=int((~np.isfinite(d["wall_coords_norm"])).sum()),
                )
        row.update(
            unit_anomaly=bool(report.get("unit_anomaly", False)),
            unit_extent_mismatch=bool(report.get("unit_extent_mismatch", False)),
            wall_crop_frac=float(report.get("wall_crop_frac", 0.0) or 0.0),
            trunk_centering_offset_frac=float(report.get("trunk_centering_offset_frac", 0.0) or 0.0),
            nodenumber_alignment_ok=bool(report.get("nodenumber_alignment_ok", False)),
            nodenumber_reordered_n_steps=int(report.get("nodenumber_reordered_n_steps", 0) or 0),
            wall_coord_mismatch_n_steps=int(report.get("wall_coord_mismatch_n_steps", 0) or 0),
        )

        if not row["bundle_exists"]:
            pass
        else:
            if row["peak_zero_frac"] > LIMITS["peak_zero_frac"]:
                fatal_reasons.append(f"peak_zero_frac>{LIMITS['peak_zero_frac']}")
            if row["all_zero_frac"] > LIMITS["all_zero_frac"]:
                fatal_reasons.append(f"all_zero_frac>{LIMITS['all_zero_frac']}")
            if row["peak_p90"] <= LIMITS["peak_p90"]:
                fatal_reasons.append("peak_p90<=0")
            if row["n_wall"] > LIMITS["n_wall"]:
                fatal_reasons.append(f"n_wall>{LIMITS['n_wall']}")
            if row["unit_anomaly"]:
                fatal_reasons.append("unit_anomaly")
            if row["unit_extent_mismatch"]:
                fatal_reasons.append("unit_extent_mismatch")
            if row["wall_crop_frac"] > LIMITS["wall_crop_frac"]:
                fatal_reasons.append(f"wall_crop_frac>{LIMITS['wall_crop_frac']}")
            if row["trunk_centering_offset_frac"] > LIMITS["trunk_centering_offset_frac"]:
                warn_reasons.append(
                    f"trunk_centering_offset_frac>{LIMITS['trunk_centering_offset_frac']}"
                )
            if row["wss_nonfinite"] > 0 or row["coords_nonfinite"] > 0:
                fatal_reasons.append("nonfinite_values")
            if not row["nodenumber_alignment_ok"]:
                fatal_reasons.append("nodenumber_alignment_missing_or_failed")
            if row["wall_coord_mismatch_n_steps"] > 0:
                fatal_reasons.append("wall_coord_mismatch")

        if fatal_reasons:
            row["status"] = "fail"
            row["fail_reasons"] = ";".join(fatal_reasons)
            failures.append(label)
        elif warn_reasons:
            row["status"] = "warn"
            row["fail_reasons"] = ""
            row["warning_reasons"] = ";".join(warn_reasons)
            warnings.append(label)
        rows.append(row)

    sp = C.load_split(split_name)
    summary = {
        "split_name": split_name,
        "partitions": list(partitions),
        "n_rows": len(rows),
        "n_failed": len(failures),
        "n_warn": len(warnings),
        "failed_cases": failures,
        "warn_cases": warnings,
        "excluded_cases": sp.get("excluded_cases", []),
        "pending_cases": sp.get("pending_cases", []),
        "not_participating": sorted(sp.get("excluded_cases", []) + sp.get("pending_cases", [])),
    }

    root = Path(out_root).resolve() if out_root is not None else C.OUT_ROOT
    out_dir = root / "pipeline_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"qa_gate_{ts}.csv"
    json_path = out_dir / f"qa_gate_{ts}.json"
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    payload = {"summary": summary, "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (out_dir / "qa_gate_latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("[qa_gate] split=%s included=%d failed=%d warn=%d -> %s",
             split_name, len(rows), len(failures), len(warnings), csv_path)
    if warnings:
        log.warning("[qa_gate] warning cases: %s", warnings)
    if failures:
        log.error("[qa_gate] failed cases: %s", failures)
        if strict:
            raise RuntimeError(f"QA gate failed for {len(failures)} included cases")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=C.DEFAULT_SPLIT_NAME)
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    args = ap.parse_args()
    reporting.setup_logging("qa_gate")
    audit_split(split_name=args.split, out_root=Path(args.out_root))


if __name__ == "__main__":
    main()
