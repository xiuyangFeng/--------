from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from wss_pinn.config import ExperimentConfig
from wss_pinn.data.raw_io import load_cases
from wss_pinn.physics.wall_shear import map_stl_normals_to_wall
from wss_pinn.utils import atomic_write_json, sha256_file, utc_now


def _boundary_metadata(raw_case_dir: Path, udf_files: list[Path]) -> dict:
    area_pattern = re.compile(
        r"sum_area_(outle|outli|outri|outre)=([-+0-9.eE]+)"
    )
    outlet_areas: dict[str, float] = {}
    evidence_log = None
    for log_path in sorted((raw_case_dir / "Global_conditions").glob("Fluent_*.out")):
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for name, value in area_pattern.findall(line):
                    numeric = float(value)
                    if numeric > 0 and name not in outlet_areas:
                        outlet_areas[name] = numeric
                if len(outlet_areas) == 4:
                    evidence_log = log_path
                    break
        if len(outlet_areas) == 4:
            break

    inlet_area = None
    zone_ids: dict[str, int] = {}
    udf_source = None
    for udf_path in udf_files:
        text = udf_path.read_text(encoding="utf-8", errors="replace")
        if "DEFINE_PROFILE(my_inlet" not in text:
            continue
        udf_source = udf_path
        profile = text.split("DEFINE_PROFILE(my_inlet", 1)[1].split(
            "DEFINE_PROPERTY", 1
        )[0]
        divisors = re.findall(r"/\s*([0-9]+\.[0-9]+)\s*;", profile, re.DOTALL)
        if divisors:
            inlet_area = float(divisors[-1])
        for thread_id, name in re.findall(
            r"Lookup_Thread\(d,\s*(\d+)\)\s*;\s*/\*\s*(out\w+)",
            text,
        ):
            zone_ids[name] = int(thread_id)
        break
    return {
        "inlet_area_m2": inlet_area,
        "outlet_areas_m2": outlet_areas,
        "area_evidence_log": str(evidence_log.resolve()) if evidence_log else None,
        "udf_source": str(udf_source.resolve()) if udf_source else None,
        "udf_zone_ids": zone_ids,
    }


def audit_case(case: dict, seed: int = 1234) -> dict:
    raw_case_dir = Path(case["raw_case_dir"])
    bundle_path = Path(case["bundle_path"])
    with np.load(bundle_path, allow_pickle=False) as source:
        wall = np.asarray(source["wall_coords_raw"])
        stl_path = Path(str(source["original_stl_path"].item()))
        stl_scale = float(source["original_stl_scale_to_mm"])
        rotation = np.asarray(source["transform_rotation"])
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(wall), min(2048, len(wall)), replace=False))
    normals, distance = map_stl_normals_to_wall(
        wall[indices], stl_path, stl_scale, rotation=rotation
    )
    norm_error = np.abs(np.linalg.norm(normals, axis=1) - 1.0)
    global_conditions = raw_case_dir / "Global_conditions"
    rfiles = sorted(global_conditions.glob("*-rfile.out")) if global_conditions.exists() else []
    case_files = list(raw_case_dir.glob("*.cas")) + list(raw_case_dir.glob("*.cas.gz"))
    journals = list(raw_case_dir.glob("*.jou"))
    udf_files = sorted(raw_case_dir.glob("udf*.c"))
    boundary_metadata = _boundary_metadata(raw_case_dir, udf_files)
    normal_pass = bool(
        np.isfinite(normals).all()
        and float(np.max(norm_error)) < 1e-4
        and float(np.quantile(distance, 0.95)) < 2.0
    )
    return {
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "status": "completed",
        "gate_result": "pass" if normal_pass else "fail",
        "normal": {
            "source": "original STL facet normal mapped by nearest face centroid",
            "stl_path": str(stl_path),
            "stl_sha256": sha256_file(stl_path),
            "orientation": (
                "facet winding retained; normalization verified; global inward/outward "
                "sign is not trusted for vector flux, while scalar WSS is flip-invariant"
            ),
            "sample_count": int(len(indices)),
            "finite_fraction": float(np.isfinite(normals).all(axis=1).mean()),
            "unit_norm_max_error": float(np.max(norm_error)),
            "mapping_distance_mm_p50": float(np.median(distance)),
            "mapping_distance_mm_p95": float(np.quantile(distance, 0.95)),
        },
        "boundary_assets": {
            "global_condition_rfiles": [str(path.resolve()) for path in rfiles],
            "case_files": [str(path.resolve()) for path in case_files],
            "journal_files": [str(path.resolve()) for path in journals],
            "udf_files": [str(path.resolve()) for path in udf_files],
            **boundary_metadata,
            "explicit_face_zone_mesh_export": False,
            "face_connectivity_export": False,
            "area_values_in_logs_available": len(
                boundary_metadata["outlet_areas_m2"]
            )
            == 4,
        },
        "blocked": [
            "hard inlet/outlet flux residual",
            "hard RCR residual",
            "connectivity-based finite-volume residual",
        ],
        "not_blocked": ["F0-U", "F0-UP", "F1 continuity autograd", "F1 no-slip"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    reports = [
        audit_case(case, seed=int(config["sampling"]["seed"]) + index)
        for index, case in enumerate(load_cases(config["data"]["split_path"]))
    ]
    passed = all(row["gate_result"] == "pass" for row in reports)
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "stage": "P0-D",
        "status": "completed",
        "gate_result": "pass" if passed else "fail",
        "zone_gate": "blocked",
        "cases": reports,
        "conclusion": (
            "STL normals are available for the pilot. Exact inlet/outlet zones and "
            "face connectivity are not exported, so hard flux/RCR remains blocked "
            "without blocking F0/F1."
        ),
    }
    output = atomic_write_json(
        Path(config["paths"]["output_root"]) / "audits/p0d/summary.json", payload
    )
    print(f"P0-D {payload['gate_result']} zone_gate=blocked {output}")


if __name__ == "__main__":
    main()
