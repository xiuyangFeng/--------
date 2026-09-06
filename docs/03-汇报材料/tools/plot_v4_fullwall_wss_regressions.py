#!/usr/bin/env python3
"""Replace the V4 1200-wall-point WSS figures with full-wall figures.

The full-wall audit must already exist under
``outputs/wss_pinn/audits/v4_fullwall_20260901``.  This script never reruns a
model.  It applies the frozen Profile-Secant V3 calibrator to the full-wall
cache, writes per-arm WSS ranking/distribution plots, regenerates the
``wss_representative_regressions.png`` triptych, and removes only the old V4
1200-point WSS images from the report folders.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = ROOT / "docs/03-汇报材料/V4_BC-PDE_FIX_EMA_横向R2散点图_2026-09-01"
FULLWALL_AUDIT = ROOT / "outputs/wss_pinn/audits/v4_fullwall_20260901"
LEGACY_AUDIT = ROOT / "outputs/wss_pinn/audits/v4_workbook_0_14_20260823"
V2_TOOL = ROOT / "docs/03-汇报材料/tools/update_wss_pinn_v2_v3_workbook.py"
LINEAR_TOOL = ROOT / "docs/03-汇报材料/tools/add_wss_pinn_v2_v3_linear_regression.py"
ARMS = (
    ("V4-SP-PN-BC-PDE-F-s1234", "V4-SP-PN-BC-PDE-FIXED"),
    ("V4-SP-PN-BC-PDE-EMA-s1234", "V4-SP-PN-BC-PDE-EMA"),
    ("V4-SP-PNPP-BC-PDE-F-s1234", "V4-SP-PNPP-BC-PDE-FIXED"),
    ("V4-SP-PNPP-BC-PDE-EMA-s1234", "V4-SP-PNPP-BC-PDE-EMA"),
    ("V4-TR-PN-BC-PDE-F-s1234", "V4-TR-PN-BC-PDE-FIXED"),
    ("V4-TR-PN-BC-PDE-EMA-s1234", "V4-TR-PN-BC-PDE-EMA"),
    ("V4-TR-PNPP-BC-PDE-F-s1234", "V4-TR-PNPP-BC-PDE-FIXED"),
    ("V4-TR-PNPP-BC-PDE-EMA-s1234", "V4-TR-PNPP-BC-PDE-EMA"),
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load("v2_workbook_fullwall_wss", V2_TOOL)
LINEAR = _load("linear_fullwall_wss", LINEAR_TOOL)


def _safe_case(case_id: str) -> str:
    return case_id.replace("/", "__")


def _triptych(path: Path, arm: str, case_paths: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    titles = {"worst": "Worst", "most_frequent": "Most Frequent", "best": "Best"}
    for axis, role in zip(axes, ("worst", "most_frequent", "best")):
        axis.imshow(plt.imread(case_paths[role]))
        axis.axis("off")
        axis.set_title(titles[role], fontsize=12, fontweight="bold")
        axis.text(
            0.5,
            -0.012,
            f"{arm} | {case_paths[role].parent.name.replace('__', '/')}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    fig.suptitle(f"{arm} | WSS representative regressions (full wall)", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "实验ID",
                "case ID",
                "full-wall points",
                "WSS raw R²",
                "WSS scaled R²",
                "WSS range NMAE",
                "WSS linear-fit R²",
                "WSS fit slope a",
                "WSS fit intercept b (Pa)",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["experiment_id"],
                    row["case_id"],
                    row["points"],
                    f'{row["raw_r2"]:.12g}',
                    f'{row["scaled_r2"]:.12g}',
                    f'{row["nmae"]:.12g}',
                    f'{row["fit_r2"]:.12g}',
                    f'{row["slope_a"]:.12g}',
                    f'{row["intercept_b"]:.12g}',
                ]
            )


def main() -> None:
    all_case_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"schema_version": 1, "protocol": "test35 x full frozen wall", "arms": {}}
    for experiment_id, report_name in ARMS:
        audit_arm = FULLWALL_AUDIT / "arms" / experiment_id
        cache_dir = audit_arm / "point_cache_s0"
        if not cache_dir.is_dir():
            raise FileNotFoundError(cache_dir)
        wss_report, arrays_by_case = V2._apply_calibrator(cache_dir)
        fit_rows = []
        for case_id, arrays in arrays_by_case.items():
            fit = LINEAR.linear_fit(arrays["truth_mag"], arrays["pred_mag"])
            wss_case = next(x for x in wss_report["cases"] if x["case_id"] == case_id)
            fit_rows.append({"case_id": case_id, "wss": fit.as_dict()})
            all_case_rows.append(
                {
                    "experiment_id": experiment_id,
                    "case_id": case_id,
                    "points": int(len(arrays["truth_mag"])),
                    "raw_r2": float(wss_case["wss_r2"]),
                    "scaled_r2": float(wss_case["calculator_metrics"]["scaled_r2"]),
                    "nmae": float(wss_case["wss_nmae_range"]),
                    "fit_r2": float(fit.regression_r2),
                    "slope_a": float(fit.slope_a),
                    "intercept_b": float(fit.intercept_b),
                }
            )
        reps = LINEAR._representatives(fit_rows, "wss")
        fit_summary = {
            "slope_a": V2._mean_std([row["wss"]["slope_a"] for row in fit_rows]),
            "intercept_b": V2._mean_std([row["wss"]["intercept_b"] for row in fit_rows]),
            "regression_r2": V2._mean_std(
                [row["wss"]["regression_r2"] for row in fit_rows]
            ),
            "protocol": "OLS y=a*x+b on test35 × all frozen wall nodes (full-wall)",
        }
        report_dir = REPORT_ROOT / report_name
        # Remove only old WSS products; speed products in the same folders stay.
        for old in report_dir.glob("representative_cases/*/wss_regression.png"):
            old.unlink()
        case_paths: dict[str, Path] = {}
        for role in ("worst", "most_frequent", "best"):
            case_id = reps[role]
            arrays = arrays_by_case[case_id]
            case_dir = report_dir / "representative_cases" / _safe_case(case_id)
            case_path = case_dir / "wss_regression.png"
            LINEAR._plot_density(
                case_path,
                *LINEAR.density_histogram(arrays["truth_mag"], arrays["pred_mag"]),
                LINEAR.linear_fit(arrays["truth_mag"], arrays["pred_mag"]),
                case_id=case_id,
                arm_key=experiment_id,
                metric_label="WSS magnitude",
                unit="Pa",
                scope_label="all frozen wall nodes; Profile-Secant V3; full-wall audit",
            )
            case_paths[role] = case_path
        _triptych(report_dir / "wss_representative_regressions.png", experiment_id, case_paths)
        rows_for_arm = [x for x in fit_rows]
        LINEAR._plot_ranking(
            report_dir / "wss_case_r2_ranking.png",
            rows_for_arm,
            "wss",
            reps,
            experiment_id,
        )
        LINEAR._plot_distribution(
            report_dir / "wss_r2_distribution.png",
            rows_for_arm,
            "wss",
            reps,
            experiment_id,
        )
        downstream_path = audit_arm / "downstream_metrics.json"
        # Physics and speed-fit diagnostics do not depend on the number of WSS
        # wall samples.  If a cache-sharded full-wall arm has not emitted its
        # aggregate file yet, reuse those invariant fields from the historical
        # 1200-point audit and replace every WSS-dependent field below.
        downstream_source = (
            downstream_path
            if downstream_path.is_file()
            else LEGACY_AUDIT / "arms" / experiment_id / "downstream_metrics.json"
        )
        downstream = json.loads(downstream_source.read_text(encoding="utf-8"))
        # Make the full-wall cache the authoritative source for every
        # WSS-dependent aggregate, even if the long-running arm process was
        # launched before the protocol label was updated.
        downstream["wss"] = wss_report
        downstream["wss_fit"] = fit_summary
        V2._write_json(audit_arm / "wss_metrics.json", wss_report)
        V2._write_json(downstream_path, downstream)
        manifest["arms"][experiment_id] = {
            "report_folder": report_name,
            "points_total": int(sum(x["points"] for x in all_case_rows if x["experiment_id"] == experiment_id)),
            "points_min": int(min(x["points"] for x in all_case_rows if x["experiment_id"] == experiment_id)),
            "points_max": int(max(x["points"] for x in all_case_rows if x["experiment_id"] == experiment_id)),
            "wss_representatives": reps,
            "wss_metrics": wss_report,
            "wss_fit": fit_summary,
            "triptych": str(report_dir / "wss_representative_regressions.png"),
            "ranking": str(report_dir / "wss_case_r2_ranking.png"),
            "distribution": str(report_dir / "wss_r2_distribution.png"),
        }
        print(json.dumps({"event": "fullwall_wss_plots", "arm": experiment_id, "points": manifest["arms"][experiment_id]["points_total"], "representatives": reps}, ensure_ascii=False), flush=True)

    _write_case_csv(REPORT_ROOT / "V4_fullwall_wss_per_case.csv", all_case_rows)
    (REPORT_ROOT / "V4_fullwall_wss_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "completed", "output_dir": str(REPORT_ROOT), "arms": len(manifest["arms"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
