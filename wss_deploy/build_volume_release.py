"""Freeze the existing three PF6 and three VF6 peak models for deployment.

This copies immutable inference inputs only. It never opens training/CFD data,
trains a model, evaluates a partition, or changes a source checkpoint.
Run: python -m wss_deploy.build_volume_release
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .io_utils import file_sha256
from .paths import PROJECT_ROOT, RELEASE_DIR
from .registry import ReleaseRegistry, VOLUME_FIELDS, _verify_package

RELEASE_ID = "PF6_VF6_peak_3seed_20260920"
FILES = ("ckpt_best.pt", "config.json", "feature_stats.json", "wss_global_stats.json", "target_normalization.json")


def build_release(output_root: Path = RELEASE_DIR.parent, *, project_root: Path = PROJECT_ROOT) -> Path:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / RELEASE_ID
    if destination.exists():
        raise FileExistsError(f"发布包已存在；不会覆盖冻结权重：{destination}")
    staging = Path(tempfile.mkdtemp(prefix=".volume_release_", dir=output_root))
    try:
        models, sources = [], []
        for family, field in (("PF6", "pressure"), ("VF6", "velocity")):
            for seed in (1234, 7, 2025):
                wave = "wss_local_wave2_20260912" if seed == 1234 else "wss_local_wave3_20260913"
                source = Path(project_root) / "training_wss_min/runs" / wave / f"{family}_s{seed}"
                relative = Path("models") / source.name
                (staging / relative).mkdir(parents=True)
                for name in FILES:
                    shutil.copy2(source / name, staging / relative / name)
                models.append({"seed": seed, "field": field, "path": relative.as_posix()})
                sources.append({"field": field, "seed": seed, "path": str(source),
                                "checkpoint_sha256": file_sha256(source / "ckpt_best.pt")})
        (staging / "rules").mkdir()
        shutil.copy2(RELEASE_DIR / "rules/flow_split_rule_train136.json",
                     staging / "rules/flow_split_rule_train136.json")
        info = {
            "release": RELEASE_ID, "frozen_on": "2026-09-20", "model_family": "PF6_VF6",
            "target": "pressure_velocity", "models": models, "source_runs": sources,
            "fields": VOLUME_FIELDS,
            "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
            "checkpoint": "ckpt_best.pt; existing train_loss-selected peak runs",
            "ensemble_protocol": "restore physical units for each seed; arithmetic vector/scalar mean over 3 seeds per field",
            "pressure_reference": "p_rel = p - peak volume-weighted anatomy mean; no absolute reference pressure is available from STL",
            "velocity_frame": "model output in v5 atlas axes; world velocity = aligned velocity @ frame_rotation",
            "input_contract": {
                "geometry": "STL and confirmed anatomical centerline only; wall support and newly sampled interior queries",
                "features": "native V5 volume 18D plus log_q_branch_murray and log_tau0_murray",
                "frame": "v5_atlas_frame_v1", "normalization": "packaged train138 global linear statistics",
                "boundary_protocol": "fixed training flow/boundary-condition family, not patient-specific flow measurements",
            },
            "validation": "Deployment integration only; STL interior sampling has not been established as equivalent to the CFD-cell evaluation protocol.",
            "note": "Source configs retain training paths as provenance only. Inference does not read those paths or CFD labels.",
        }
        (staging / "release.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records = [f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}  {path.stat().st_size}"
                   for directory in (staging / "models", staging / "rules")
                   for path in sorted(directory.rglob("*")) if path.is_file()]
        (staging / "MANIFEST.sha256").write_text("\n".join(records) + "\n", encoding="utf-8")
        registry = ReleaseRegistry(staging)
        _verify_package(registry.default)
        staging.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RELEASE_DIR.parent)
    arguments = parser.parse_args()
    print(build_release(arguments.output_root))
