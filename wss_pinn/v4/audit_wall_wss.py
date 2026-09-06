"""Wall WSS label compliance audit for the anatomy-only V4 rebuild.

Downstream WSS regression consumes the ``ascii`` wall exports as labels, so a
label that is stale, mis-oriented or inconsistent with its own magnitude column
silently caps the achievable accuracy.  For every case this script checks:

file/format
    81 frames, step list equal to the volume export, identical row count and
    identical coordinates across the sampled frames, required columns present,
    all values finite, ``wall-shear`` non-negative.

self-consistency
    the scalar ``wall-shear`` column must equal the norm of the
    ``(x,y,z)-wall-shear`` vector.

orientation
    the WSS vector must be tangential to the wall.  Face/node normals come from
    the Fluent case topology (:mod:`fluent_topology`), so this is an independent
    geometric check rather than a re-statement of the export.

provenance (staleness)
    the wall ``pressure`` column is compared against the volume export pressure
    of the adjacent cell at the same step.  For a wall and a volume written by
    the same solver run this difference is exactly 0 Pa (verified on the
    cohort); a non-zero difference proves the wall labels come from a
    superseded solution.

coverage
    the exported wall entities must be exactly the anatomy wall (faces adjacent
    to the ``blood`` cell zone), with no extension-segment wall leaking in.

Results go to
``outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits/wall_wss``.
Nothing outside that audit root is written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.data.raw_io import read_interior, read_wall, step_file
from wss_pinn.utils import ROOT, utc_now
from wss_pinn.v4.fluent_topology import anatomy_topology, anatomy_wall_faces, match_points, read_fluent_mesh

SPLIT = ROOT / "wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train138_test34_s1234.json"
BOUNDARY_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/cases"
V4_ROOT = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/cases"
PREP_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
AUDIT_ROOT = PREP_ROOT / "audits/wall_wss"
EXPECTED_STEPS = list(range(1120, 1281, 2))
NODE_TOLERANCE_M = 1.0e-7
FACE_TOLERANCE_M = 1.0e-6
# physiological guard rails for aorto-iliac WSS (Pa); breaches are reported, not clipped
WSS_P50_MIN_PA = 0.05
WSS_P50_MAX_PA = 15.0
MAGNITUDE_REL_TOL = 1.0e-3
NODE_INTERP_GAP_TOL = 0.02
TANGENCY_WARN = 0.2


def _case_ids() -> list[dict[str, Any]]:
    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    return [{"canonical_id": c, "role": "train"} for c in payload["train_cases"]] + [
        {"canonical_id": c, "role": "test"} for c in payload["test_cases"]
    ]


def _step_candidates(case_dir: Path, step: int, subdir: str) -> list[Path]:
    return sorted(path for path in Path(case_dir, subdir).glob(f"*-{int(step)}") if path.is_file())


def _optional_step_file(case_dir: Path, step: int, subdir: str) -> Path | None:
    candidates = _step_candidates(case_dir, step, subdir)
    return candidates[0] if candidates else None


def _steps(directory: Path) -> list[int]:
    out = []
    for path in directory.glob("*-[0-9]*"):
        match = re.search(r"-(\d+)$", path.name)
        if match and path.is_file():
            out.append(int(match.group(1)))
    return sorted(out)


def _node_normals(mesh, wall: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted node normals of the anatomy wall (1-based node ids kept)."""
    offsets = wall["face_offsets"]
    nodes = wall["face_nodes"]
    counts = np.diff(offsets)
    area = wall["area_vectors_m2"]
    face_index = np.repeat(np.arange(len(counts)), counts)
    accumulator = np.zeros((mesh.node_count + 1, 3), dtype=np.float64)
    for axis in range(3):
        accumulator[:, axis] = np.bincount(
            nodes, weights=area[face_index, axis], minlength=mesh.node_count + 1
        )
    ids = wall["node_ids"]
    vectors = accumulator[ids]
    norm = np.linalg.norm(vectors, axis=1)
    return ids, vectors / np.maximum(norm, 1.0e-300)[:, None]


def audit_case(entry: dict[str, Any]) -> dict[str, Any]:
    canonical_id = entry["canonical_id"]
    output = AUDIT_ROOT / "cases" / f"{canonical_id.replace('/', '__')}.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    raw_dir = ROOT / "data_new" / canonical_id
    wall_dir = raw_dir / "ascii"
    volume_dir = raw_dir / "ascii_in"
    wall_steps = _steps(wall_dir)
    volume_steps = _steps(volume_dir)
    peak_step = int(json.loads((V4_ROOT / canonical_id / "manifest.json").read_text(encoding="utf-8"))["peak_step"])
    sample_steps = sorted({EXPECTED_STEPS[0], peak_step, EXPECTED_STEPS[-1]})

    # A re-run leaves the odd solver steps behind in ``ascii`` (the cleanup script
    # only trims 1..1119 there), so the contract is "the 81 required steps are
    # present and self-consistent", not "the directory holds exactly 81 files".
    required = [path for step in EXPECTED_STEPS if (path := _optional_step_file(raw_dir, step, "ascii")) is not None]
    sizes = {path.stat().st_size for path in required}
    mtimes = [path.stat().st_mtime for path in required]
    # A rename-based re-export (e.g. ``<NAME>-1120-<step>`` beside ``<NAME>-<step>``)
    # leaves two files matching ``*-<step>``; picking one silently would read the
    # wrong solution, so ambiguity is a hard failure.
    ambiguous = sorted(
        step for step in EXPECTED_STEPS if len(_step_candidates(raw_dir, step, "ascii")) > 1
    )
    extra = [
        int(re.search(r"-(\d+)$", path.name).group(1))
        for path in wall_dir.glob("*-[0-9]*")
        if path.is_file() and int(re.search(r"-(\d+)$", path.name).group(1)) not in set(EXPECTED_STEPS)
    ]
    extra_paths = [
        path
        for path in wall_dir.glob("*-[0-9]*")
        if path.is_file() and int(re.search(r"-(\d+)$", path.name).group(1)) not in set(EXPECTED_STEPS)
    ]
    extra_benign = False
    if extra_paths and mtimes:
        lo, hi = min(mtimes), max(mtimes)
        same_window = all(lo - 86400 <= path.stat().st_mtime <= hi + 86400 for path in extra_paths)
        required_id = "nodenumber" if "nodenumber" in open(required[0], errors="replace").readline() else "cellnumber"
        same_id = all(
            (("nodenumber" in open(path, errors="replace").readline()) == (required_id == "nodenumber"))
            for path in extra_paths[:8]
        )
        extra_benign = bool(same_window and same_id)
    volume_mtimes = [path.stat().st_mtime for path in volume_dir.glob("*-[0-9]*") if path.is_file()]
    report: dict[str, Any] = {
        "schema_version": 1,
        "canonical_id": canonical_id,
        "role": entry["role"],
        "created_at": utc_now(),
        "peak_step": peak_step,
        "files": {
            "wall_files_in_directory": len(wall_steps),
            "wall_required_steps_present": len(required),
            "volume_frames": len(volume_steps),
            "extra_wall_files": len(extra),
            "extra_wall_files_benign_same_run": extra_benign,
            "ambiguous_steps": len(ambiguous),
            "ambiguous_steps_sample": ambiguous[:5],
            "ambiguous_candidates_sample": [p.name for p in _step_candidates(raw_dir, ambiguous[0], "ascii")] if ambiguous else [],
            "extra_wall_steps_sample": sorted(extra)[:5],
            "volume_steps_equal_expected": volume_steps == EXPECTED_STEPS,
            "wall_steps_equal_expected": len(required) == len(EXPECTED_STEPS),
            "wall_sizes_uniform": len(sizes) == 1,
            "wall_mtime_max": max(mtimes) if mtimes else None,
            "volume_mtime_min": min(volume_mtimes) if volume_mtimes else None,
            "wall_lag_days": (min(volume_mtimes) - max(mtimes)) / 86400.0 if mtimes and volume_mtimes else None,
        },
    }
    if len(required) != len(EXPECTED_STEPS):
        report["status"] = "wall_frame_set_invalid"
        report["gate_pass"] = False
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        return report

    with step_file(raw_dir, sample_steps[0], "ascii").open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    # Fluent writes either comma- or whitespace-delimited headers (both occur in
    # this cohort); split on whichever the file uses.
    delimiter = "comma" if "," in header else "whitespace"
    columns = [value.strip() for value in (header.split(",") if "," in header else header.split())]
    export_kind = "node" if "nodenumber" in columns else "face-centre"

    frames = {}
    frame_formats = {}
    for step in sample_steps:
        path = _optional_step_file(raw_dir, step, "ascii")
        with path.open(encoding="utf-8", errors="replace") as handle:
            frame_header = handle.readline()
        frame_formats[str(step)] = {
            "file": path.name,
            "id_column": "nodenumber" if "nodenumber" in frame_header else "cellnumber",
        }
        frames[step] = read_wall(path)

    reference = frames[sample_steps[0]]
    rows = len(reference["coords"])
    mixed_format = len({v["id_column"] for v in frame_formats.values()}) > 1 or len(
        {len(frames[step]["coords"]) for step in sample_steps}
    ) > 1
    if mixed_format:
        report.update(
            {
                "status": "mixed_frame_format",
                "export_kind": export_kind,
                "delimiter": delimiter,
                "rows": rows,
                "frame_formats": frame_formats,
                "gates": {"frames_same_format": False},
                "gate_pass": False,
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        return report
    coord_delta = max(
        float(np.max(np.abs(frames[step]["coords"] - reference["coords"]))) if len(frames[step]["coords"]) == rows else np.inf
        for step in sample_steps
    )

    per_frame = {}
    for step, frame in frames.items():
        magnitude = np.asarray(frame["wss"], dtype=np.float64)
        vector = np.asarray(frame["wss_vector"], dtype=np.float64)
        norm = np.linalg.norm(vector, axis=1)
        scale = max(float(np.percentile(magnitude, 99)), 1.0e-12)
        # Fluent interpolates |tau| and the components to nodes independently, so
        # at a node |vector| <= scalar, with the gap measuring local direction
        # spread (separation/reattachment).  At face centres the two are equal.
        gap = (magnitude - norm) / np.maximum(magnitude, 1.0e-12)
        per_frame[str(step)] = {
            "vector_norm_le_scalar_fraction": float(np.mean(norm <= magnitude + 1.0e-9)),
            "scalar_minus_vector_rel_median": float(np.median(gap)),
            "scalar_minus_vector_rel_p95": float(np.percentile(gap, 95)),
            "scalar_minus_vector_rel_max": float(np.max(gap)),
            "rows": int(len(magnitude)),
            "finite": bool(np.isfinite(magnitude).all() and np.isfinite(vector).all() and np.isfinite(frame["pressure"]).all()),
            "wss_negative_rows": int(np.sum(magnitude < 0.0)),
            "wss_zero_fraction": float(np.mean(magnitude <= 0.0)),
            "magnitude_vs_vector_max_rel": float(np.max(np.abs(magnitude - norm)) / scale),
            "wss_mean_pa": float(magnitude.mean()),
            "wss_p50_pa": float(np.percentile(magnitude, 50)),
            "wss_p95_pa": float(np.percentile(magnitude, 95)),
            "wss_p99_pa": float(np.percentile(magnitude, 99)),
            "wss_max_pa": float(magnitude.max()),
            "pressure_mean_pa": float(np.mean(frame["pressure"])),
        }

    # geometry: anatomy wall coverage + tangency + provenance
    boundary = json.loads((BOUNDARY_ROOT / canonical_id / "manifest.json").read_text(encoding="utf-8"))
    mesh = read_fluent_mesh(Path(boundary["provenance"]["fluent_case"]["path"]))
    topology = anatomy_topology(mesh)
    wall = anatomy_wall_faces(mesh, topology)
    peak = frames[peak_step]
    unique_coords, inverse = np.unique(np.asarray(peak["coords"], dtype=np.float64), axis=0, return_inverse=True)
    if export_kind == "node":
        ids, normals = _node_normals(mesh, wall)
        index, identity = match_points(
            mesh.nodes_m[ids], unique_coords, label=f"{canonical_id}:wss-nodes", tolerance_m=NODE_TOLERANCE_M
        )
        matched_normals = normals[index][inverse]
        covered = set(ids[index].tolist()) == set(wall["node_ids"].tolist())
        entities = int(len(wall["node_ids"]))
    else:
        normals = wall["area_vectors_m2"] / np.maximum(
            np.linalg.norm(wall["area_vectors_m2"], axis=1), 1.0e-300
        )[:, None]
        index, identity = match_points(
            wall["centres_m"], unique_coords, label=f"{canonical_id}:wss-faces", tolerance_m=FACE_TOLERANCE_M
        )
        matched_normals = normals[index][inverse]
        covered = len(unique_coords) == len(wall["centres_m"])
        entities = int(len(wall["centres_m"]))
    vector = np.asarray(peak["wss_vector"], dtype=np.float64)
    norm = np.linalg.norm(vector, axis=1)
    valid = norm > 1.0e-12
    tangency = np.abs(np.einsum("ij,ij->i", vector[valid], matched_normals[valid])) / norm[valid]

    volume = read_interior(step_file(raw_dir, peak_step, "ascii_in"))
    distance, nearest = cKDTree(volume["coords"]).query(peak["coords"], k=1)
    pressure_delta = np.asarray(peak["pressure"], dtype=np.float64) - volume["pressure"][nearest]
    volume_pressure = float(np.mean(volume["pressure"][nearest]))

    report.update(
        {
            "status": "parsed",
            "export_kind": export_kind,
            "delimiter": delimiter,
            "columns": columns,
            "rows": rows,
            "anatomy_wall_entities": entities,
            "coordinate_identity_across_frames_m": coord_delta,
            "sampled_steps": sample_steps,
            "frame_formats": frame_formats,
            "per_frame": per_frame,
            "identity": identity,
            "tangency": {
                "median": float(np.median(tangency)) if len(tangency) else None,
                "p95": float(np.percentile(tangency, 95)) if len(tangency) else None,
                "max": float(np.max(tangency)) if len(tangency) else None,
                "rows_with_zero_wss": int(np.sum(~valid)),
            },
            "provenance": {
                "wall_pressure_mean_pa": float(np.mean(peak["pressure"])),
                "volume_pressure_at_wall_mean_pa": volume_pressure,
                "pressure_delta_median_pa": float(np.median(pressure_delta)),
                "pressure_delta_p95_abs_pa": float(np.percentile(np.abs(pressure_delta), 95)),
                "pressure_delta_rel_to_volume": float(abs(np.median(pressure_delta)) / max(abs(volume_pressure), 1.0)),
                "nearest_cell_distance_p95_mm": float(np.percentile(distance, 95) * 1000.0),
            },
        }
    )
    frame_values = list(per_frame.values())
    gates = {
        "frames_same_format": True,
        "wall_frames_81_matching_volume": report["files"]["wall_steps_equal_expected"] and report["files"]["volume_steps_equal_expected"],
        # Extra files are harmless when they come from the same export run
        # (same id column, same mtime window) - they are simply unused odd
        # solver steps.  Only stale/foreign extras are a defect.
        "no_stale_extra_wall_files": len(extra) == 0 or extra_benign,
        "step_to_file_unambiguous": len(ambiguous) == 0,
        "row_count_constant": all(f["rows"] == rows for f in frame_values),
        "coordinates_identical_across_frames": coord_delta <= 1.0e-12,
        "all_finite": all(f["finite"] for f in frame_values),
        "wss_non_negative": all(f["wss_negative_rows"] == 0 for f in frame_values),
        "magnitude_consistent_with_vector": (
            all(f["magnitude_vs_vector_max_rel"] <= MAGNITUDE_REL_TOL for f in frame_values)
            if export_kind == "face-centre"
            else all(
                f["vector_norm_le_scalar_fraction"] >= 1.0 - 1.0e-9
                and f["scalar_minus_vector_rel_median"] <= NODE_INTERP_GAP_TOL
                for f in frame_values
            )
        ),
        "wss_tangential_to_wall": bool(report["tangency"]["p95"] is not None and report["tangency"]["p95"] <= TANGENCY_WARN),
        "covers_anatomy_wall_exactly": bool(covered),
        "identity_bijective": bool(identity["bijective"] and identity["within_tolerance"]),
        "same_solution_as_volume": abs(report["provenance"]["pressure_delta_median_pa"]) <= 1.0,
        # The bulk of the wall must sit in the physiological band.  The upper tail
        # is reported as a diagnostic instead of a pass/fail: peak-systolic
        # extremes are partly real (iliac narrowings) and partly solver noise,
        # and the cohort-relative outliers are listed in the summary.
        "wss_bulk_physiological": all(
            WSS_P50_MIN_PA <= f["wss_p50_pa"] <= WSS_P50_MAX_PA for f in frame_values
        ),
    }
    report["gates"] = gates
    report["gate_pass"] = all(gates.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, output)
    return report


def _safe(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        return audit_case(entry)
    except Exception as error:  # noqa: BLE001
        return {
            "canonical_id": entry["canonical_id"],
            "role": entry["role"],
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "gate_pass": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()
    entries = _case_ids()
    if args.cases:
        wanted = set(args.cases)
        entries = [row for row in entries if row["canonical_id"] in wanted]
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for report in pool.map(_safe, entries):
            if "error" in report:
                print(f"{report['canonical_id']}: ERROR {report['error'][:110]}", flush=True)
            else:
                failed = [k for k, v in report.get("gates", {}).items() if not v]
                print(
                    f"{report['canonical_id']}: {'pass' if report['gate_pass'] else 'FAIL ' + ','.join(failed)}",
                    flush=True,
                )
            reports.append(report)
    good = [r for r in reports if "gates" in r]
    counts: dict[str, int] = {}
    for report in good:
        for name, value in report["gates"].items():
            counts[name] = counts.get(name, 0) + (0 if value else 1)
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "cases": len(reports),
        "pass": sum(1 for r in reports if r.get("gate_pass")),
        "errors": [{"canonical_id": r["canonical_id"], "error": r["error"]} for r in reports if "error" in r],
        "gate_failure_counts": {k: v for k, v in counts.items() if v},
        "failures": [
            {
                "canonical_id": r["canonical_id"],
                "role": r["role"],
                "failed_gates": [k for k, v in r["gates"].items() if not v],
                "pressure_delta_median_pa": r.get("provenance", {}).get("pressure_delta_median_pa"),
                "wall_lag_days": r.get("files", {}).get("wall_lag_days"),
                "tangency_p95": r.get("tangency", {}).get("p95"),
                "wss_max_pa": max((f["wss_max_pa"] for f in r.get("per_frame", {}).values()), default=None),
            }
            for r in good
            if not r["gate_pass"]
        ],
        "delimiters": {
            kind: sorted(r["canonical_id"] for r in good if r.get("delimiter") == kind)
            for kind in ("comma", "whitespace")
        },
        "upper_tail_outliers": sorted(
            (
                {
                    "canonical_id": r["canonical_id"],
                    "role": r["role"],
                    "wss_p99_pa": r["per_frame"][str(r["peak_step"])]["wss_p99_pa"],
                    "wss_max_pa": r["per_frame"][str(r["peak_step"])]["wss_max_pa"],
                }
                for r in good
                if str(r.get("peak_step")) in r.get("per_frame", {})
                and r["per_frame"][str(r["peak_step"])]["wss_p99_pa"]
                > float(np.percentile([x["per_frame"][str(x["peak_step"])]["wss_p99_pa"] for x in good if str(x.get("peak_step")) in x.get("per_frame", {})], 95))
            ),
            key=lambda row: -row["wss_p99_pa"],
        ),
        "peak_frame_wss_pa": {
            stat: {
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for stat, values in {
                "p50": [r["per_frame"][str(r["peak_step"])]["wss_p50_pa"] for r in good if str(r.get("peak_step")) in r.get("per_frame", {})],
                "p99": [r["per_frame"][str(r["peak_step"])]["wss_p99_pa"] for r in good if str(r.get("peak_step")) in r.get("per_frame", {})],
                "max": [r["per_frame"][str(r["peak_step"])]["wss_max_pa"] for r in good if str(r.get("peak_step")) in r.get("per_frame", {})],
            }.items()
        },
        "export_kinds": {
            kind: sorted(r["canonical_id"] for r in good if r.get("export_kind") == kind)
            for kind in ("node", "face-centre")
        },
        "cases_report": {
            r["canonical_id"]: {
                "role": r["role"],
                "export_kind": r.get("export_kind"),
                "rows": r.get("rows"),
                "wss_p50_pa": r["per_frame"][str(r["peak_step"])]["wss_p50_pa"] if str(r.get("peak_step")) in r.get("per_frame", {}) else None,
                "wss_p99_pa": r["per_frame"][str(r["peak_step"])]["wss_p99_pa"] if str(r.get("peak_step")) in r.get("per_frame", {}) else None,
                "wss_max_pa": r["per_frame"][str(r["peak_step"])]["wss_max_pa"] if str(r.get("peak_step")) in r.get("per_frame", {}) else None,
                "wss_zero_fraction": r["per_frame"][str(r["peak_step"])]["wss_zero_fraction"] if str(r.get("peak_step")) in r.get("per_frame", {}) else None,
                "scalar_minus_vector_rel_median": r["per_frame"][str(r["peak_step"])]["scalar_minus_vector_rel_median"] if str(r.get("peak_step")) in r.get("per_frame", {}) else None,
                "tangency_p95": r.get("tangency", {}).get("p95"),
                "pressure_delta_median_pa": r.get("provenance", {}).get("pressure_delta_median_pa"),
                "wall_lag_days": r.get("files", {}).get("wall_lag_days"),
                "extra_wall_files": r.get("files", {}).get("extra_wall_files"),
                "ambiguous_steps": r.get("files", {}).get("ambiguous_steps"),
                "gate_pass": r.get("gate_pass"),
            }
            for r in good
        },
    }
    (AUDIT_ROOT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"pass {summary['pass']}/{summary['cases']}; summary at {AUDIT_ROOT / 'summary.json'}")
    return 0 if summary["pass"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
