"""Build a conservative max(base V4, depth-3 V4) point cache and evaluate it."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from batch_validate_cfd import summarize
from calculate_wss_cfd import evaluate, pca_wall_normals, wss_from_gradient
import data_loader_cfd as dl
from wss_surface_mls_v4 import SurfaceMLSV4Config, fit_wall_gradient_surface_mls


ROOT = Path("/public/newhome/cy/Digital_twin/GNN")


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for method in ("base_v4", "depth3_fusion"):
        result[method] = {
            metric: summarize([row["methods"][method][metric] for row in rows])
            for metric in (
                "raw_r2",
                "scaled_r2",
                "spearman",
                "alpha",
                "mae_pa",
                "nrmse",
                "high_wss_nrmse",
            )
        }
    delta = np.asarray(
        [
            row["methods"]["depth3_fusion"]["raw_r2"]
            - row["methods"]["base_v4"]["raw_r2"]
            for row in rows
        ]
    )
    result["paired"] = {
        "raw_r2_delta": summarize(delta.tolist()),
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
        "min_delta": float(np.min(delta)) if len(delta) else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache-dir", required=True)
    parser.add_argument("--output-cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--screen-result", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    started = time.time()

    input_dir = Path(args.input_cache_dir).resolve()
    output_dir = Path(args.output_cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = sorted(input_dir.glob("*.npz"))
    if args.max_cases > 0:
        cache_paths = cache_paths[: args.max_cases]
    if not cache_paths:
        raise FileNotFoundError(f"no input cache files under {input_dir}")
    screen_cases: set[str] = set()
    if args.screen_result:
        screen_payload = json.loads(
            Path(args.screen_result).read_text(encoding="utf-8")
        )
        screen_cases = {row["canonical_id"] for row in screen_payload["rows"]}

    depth3_config = SurfaceMLSV4Config(
        correction_max=1.5,
        parallel_transport=True,
        group_balance_power=0.5,
        depth_degree=3,
    )
    rows = []
    for number, cache_path in enumerate(cache_paths, start=1):
        with np.load(cache_path, allow_pickle=False) as source:
            arrays = {key: np.asarray(source[key]) for key in source.files}
        canonical_id = str(arrays["canonical_id"])
        step = int(arrays["step"])
        case_dir = ROOT / "data_new" / canonical_id
        wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
        interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
        interior_tree = cKDTree(interior["coords_mm"])
        full_normals = pca_wall_normals(
            wall["coords_mm"], interior["coords_mm"], interior_tree
        )
        v1_gradient = np.asarray(arrays["gradient_v1"], dtype=np.float64)
        base_gradient = np.asarray(arrays["gradient_v4"], dtype=np.float64)
        _, depth3_diag = fit_wall_gradient_surface_mls(
            np.asarray(arrays["wall_mm"], dtype=np.float64),
            np.asarray(arrays["normal"], dtype=np.float64),
            np.asarray(wall["coords_mm"], dtype=np.float64),
            full_normals,
            np.asarray(interior["coords_mm"], dtype=np.float64),
            np.asarray(interior["velocity"], dtype=np.float64),
            interior_tree,
            local_radius_mm=np.asarray(arrays["local_radius_mm"], dtype=np.float64),
            fallback_gradient=v1_gradient,
            config=depth3_config,
        )
        base_correction = np.asarray(arrays["combined_correction"], dtype=np.float64)
        depth3_correction = np.asarray(depth3_diag["correction"], dtype=np.float64)
        fusion_correction = np.maximum(base_correction, depth3_correction)
        fusion_gradient = v1_gradient * fusion_correction[:, None]
        truth_mag = np.asarray(arrays["truth_mag"], dtype=np.float64)
        truth_vec = np.asarray(arrays["truth_vec"], dtype=np.float64)
        methods = {
            "base_v4": evaluate(
                truth_mag, truth_vec, wss_from_gradient(base_gradient, "carreau")
            ),
            "depth3_fusion": evaluate(
                truth_mag, truth_vec, wss_from_gradient(fusion_gradient, "carreau")
            ),
        }
        rows.append(
            {
                "canonical_id": canonical_id,
                "cohort": str(arrays["cohort"]),
                "role": "screen36" if canonical_id in screen_cases else "validation",
                "methods": methods,
                "fusion_applied_fraction": float(
                    np.mean(fusion_correction > base_correction + 1e-12)
                ),
            }
        )

        arrays["gradient_v4_base"] = np.asarray(arrays["gradient_v4"])
        arrays["combined_correction_base"] = np.asarray(arrays["combined_correction"])
        arrays["gradient_v4"] = fusion_gradient.astype(np.float32)
        arrays["combined_correction"] = fusion_correction.astype(np.float32)
        arrays["depth3_fusion_applied"] = (
            fusion_correction > base_correction + 1e-12
        ).astype(np.int8)
        for name, value in depth3_diag.items():
            arrays[f"depth3__{name}"] = np.asarray(value)
        output_path = output_dir / cache_path.name
        with output_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        print(
            f"[{number:03d}/{len(cache_paths):03d}] {canonical_id} "
            f"base={methods['base_v4']['raw_r2']:.4f} "
            f"fusion={methods['depth3_fusion']['raw_r2']:.4f} "
            f"applied={rows[-1]['fusion_applied_fraction']:.1%}",
            flush=True,
        )

    validation_rows = [row for row in rows if row["role"] == "validation"]
    screen_rows = [row for row in rows if row["role"] == "screen36"]
    payload = {
        "status": "depth3_max_fusion_cache",
        "input_cache_dir": str(input_dir),
        "output_cache_dir": str(output_dir),
        "depth3_config": depth3_config.__dict__,
        "n_cases": len(rows),
        "aggregate_all": aggregate_rows(rows),
        "aggregate_screen36": aggregate_rows(screen_rows) if screen_rows else {},
        "aggregate_validation": (
            aggregate_rows(validation_rows) if validation_rows else {}
        ),
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate_all": payload["aggregate_all"],
                "aggregate_validation": payload["aggregate_validation"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(output)


if __name__ == "__main__":
    main()
