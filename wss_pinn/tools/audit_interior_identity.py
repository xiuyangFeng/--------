from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from wss_pinn.config import ExperimentConfig
from wss_pinn.data.raw_io import list_steps, load_cases, read_interior, step_file
from wss_pinn.utils import atomic_write_json, guard_write_path, sha256_file, utc_now


def _write_csv(path: Path, rows: list[dict]) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def audit_case(case: dict) -> dict:
    raw_case_dir = Path(case["raw_case_dir"])
    bundle_path = Path(case["bundle_path"])
    with np.load(bundle_path, allow_pickle=False) as bundle:
        peak = int(bundle["peak_step"])
        wall_wss_min = float(np.min(bundle["wall_wss"]))
        wall_wss_max = float(np.max(bundle["wall_wss"]))
        unit_factor = float(bundle["unit_factor"])
        coord_scale_mm = float(bundle["coord_scale"])
    available = list_steps(raw_case_dir)
    candidates = [step for step in (peak - 2, peak, peak + 2) if step in available]
    if peak not in candidates:
        raise ValueError(f"peak step {peak} is missing: {raw_case_dir}")
    reference = None
    step_rows = []
    stable_order = True
    stable_set = True
    stable_coords = True
    for step in candidates:
        path = step_file(raw_case_dir, step)
        fields = read_interior(path)
        ids = fields["cell_id"]
        coords = fields["coords"]
        unique = len(np.unique(ids)) == len(ids)
        if reference is None:
            reference = {"ids": ids.copy(), "coords": coords.copy()}
            order_equal = set_equal = True
            coord_max_delta = 0.0
        else:
            set_equal = len(ids) == len(reference["ids"]) and np.array_equal(
                np.sort(ids), np.sort(reference["ids"])
            )
            order_equal = np.array_equal(ids, reference["ids"])
            coord_max_delta = (
                float(np.max(np.abs(coords - reference["coords"])))
                if order_equal and len(coords) == len(reference["coords"])
                else float("nan")
            )
        stable_order &= bool(order_equal)
        stable_set &= bool(set_equal and unique)
        stable_coords &= bool(np.isfinite(coord_max_delta) and coord_max_delta <= 1e-10)
        velocity = fields["velocity"]
        pressure = fields["pressure"]
        step_rows.append(
            {
                "step": int(step),
                "time_s": int(step) * 0.005,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": int(len(ids)),
                "ids_unique": bool(unique),
                "id_order_equal_reference": bool(order_equal),
                "id_set_equal_reference": bool(set_equal),
                "coord_max_abs_delta": coord_max_delta,
                "coord_min": coords.min(axis=0).tolist(),
                "coord_max": coords.max(axis=0).tolist(),
                "velocity_min": velocity.min(axis=0).tolist(),
                "velocity_max": velocity.max(axis=0).tolist(),
                "velocity_nonfinite": int((~np.isfinite(velocity)).sum()),
                "pressure_min_pa": float(np.min(pressure)),
                "pressure_max_pa": float(np.max(pressure)),
                "pressure_nonfinite": int((~np.isfinite(pressure)).sum()),
            }
        )
    passed = stable_order and stable_set and stable_coords and all(
        row["velocity_nonfinite"] == 0 and row["pressure_nonfinite"] == 0
        for row in step_rows
    )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "status": "completed",
        "gate_result": "pass" if passed else "fail",
        "identity": {
            "cell_id_set_stable": stable_set,
            "row_order_stable": stable_order,
            "coordinates_stable": stable_coords,
            "alignment_rule": "cellnumber; row use allowed only while order and coords pass",
        },
        "units": {
            "raw_coordinate": "nominal m",
            "raw_to_mm_factor": unit_factor,
            "registered_coordinate_scale_mm": coord_scale_mm,
            "velocity": "m/s",
            "pressure": "Pa; gauge arbitrary",
            "time_step": "0.005 s",
            "wss": "Pa",
            "wss_range_pa": [wall_wss_min, wall_wss_max],
        },
        "bundle": {"path": str(bundle_path.resolve()), "sha256": sha256_file(bundle_path)},
        "steps": step_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    cases = load_cases(config["data"]["split_path"])
    output_dir = Path(config["paths"]["output_root"]) / "audits" / "p0a"
    rows = []
    for case in cases:
        report = audit_case(case)
        case_name = str(case["case_id"]).replace("/", "__")
        path = atomic_write_json(output_dir / f"{case['cohort']}__{case_name}.json", report)
        rows.append(
            {
                "cohort": case["cohort"],
                "case_id": case["case_id"],
                "status": report["status"],
                "gate_result": report["gate_result"],
                **report["identity"],
                "report": str(path),
            }
        )
        print(f"{case['cohort']}/{case['case_id']}: {report['gate_result']}")
    _write_csv(output_dir / "cases.csv", rows)
    passed = all(row["gate_result"] == "pass" for row in rows)
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "stage": "P0-A",
        "status": "completed",
        "gate_result": "pass" if passed else "fail",
        "case_count": len(rows),
        "cases_csv": str((output_dir / "cases.csv").resolve()),
        "cases": rows,
    }
    output = atomic_write_json(output_dir / "summary.json", summary)
    print(f"P0-A {summary['gate_result']} {output}")


if __name__ == "__main__":
    main()

