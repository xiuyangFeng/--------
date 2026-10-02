"""Freeze the v5.2d full265 PF6 / VF6 peak volume release (2026-10-03): pressure × 3 seeds + velocity × 3 seeds.

Source: the PF6/VF6 v5.2d retrain (``training_wss_min/runs/pf6vf6_v52d_retrain_20261002``, tracking doc 01 block §43):
recipes unchanged from the v5.0 release (18 native volume features + log_q_branch_murray / log_tau0_murray), trained on
full265 = 265 v5.2d units with no validation split (selection = train_loss), evaluated on recover8. Copies immutable
inference inputs only (checkpoint, config, feature statistics, target statistics, normalisation record) and the v5.2d
split rule; never trains, evaluates or changes a source checkpoint. The validation block is read from the readout that
the retrain's post-queue job wrote from saved predictions (``experiments/pf6vf6_v52d_retrain_20261002/readout_ckpt_best.json``).

    PYTHONPATH=. python -m wss_deploy.build_v52d_volume_release [--output-root outputs/wss_deploy_release]

Assembled in a staging folder, verified by the registry and only then renamed into place; an existing destination is never
overwritten. The geometry reference sidecar is built afterwards with
``python -m wss_deploy.build_reference_profiles --data v52d --geometry-only --release PF6_VF6_v52d_3seed_20261003 --write``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .io_utils import file_sha256
from .paths import PROJECT_ROOT, RELEASE_DIR
from .registry import ReleaseRegistry, VOLUME_FIELDS, _verify_package

RELEASE_ID = "PF6_VF6_v52d_3seed_20261003"
FROZEN_ON = "2026-10-03"
EXPERIMENT = "pf6vf6_v52d_retrain_20261002"
SEEDS = (1234, 7, 2025)
FILES = ("ckpt_best.pt", "config.json", "feature_stats.json", "wss_global_stats.json", "target_normalization.json")
RULE = "data_wss_v5/views_v5_2d_20261001/wss_min_flowref_v1/flow_split_rule_v52d_train136.json"
TRACKING = "docs/02-推进与变更/01-X5D主线与新数据/X5D主线_实验跟踪.md §43"
DATA = ("v5.2d (2026-10-01): 332 library units = 273 real + 59 synthetic; this release trains on full265 = 265 real units "
        "with no validation split and is evaluated on recover8 (8 cfd_auto units never trained on); volume labels come from "
        "the v5.2d volume root wss_min_volview_v1 (273 real units; existing volume views verified array-for-array against "
        "the v5.2d snapshot); 150 units whose per-step iterations exited early keep their labels")


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project_root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validation_block(project_root: Path) -> dict:
    """Validation numbers from the retrain readout (CFD-cell protocol: held-out units, all CFD wall nodes / anatomy cells)."""
    readout = Path(project_root) / "training_wss_min/experiments" / EXPERIMENT / "readout_ckpt_best.json"
    r = json.loads(readout.read_text(encoding="utf-8"))
    if r.get("consistency", {}).get("max_abs_diff", 1.0) > 1e-5:
        raise RuntimeError("readout consistency check did not pass; refusing to freeze its numbers")
    p_ens, v_ens = r["A"]["PF6"]["ensemble3"], r["A"]["VF6"]["ensemble3"]
    p_sd, v_sd = r["A"]["PF6"]["seed_mean_sd"], r["A"]["VF6"]["seed_mean_sd"]
    pick = lambda t, label: next(x for x in r["B"][t]["rows"] if x["model"] == label)
    p8, v8 = pick("PF6", "**full265 三 seed 集成**"), pick("VF6", "**full265 三 seed 集成**")
    e5p, e5v = pick("PF6", "**E5 已部署 v5.0 三 seed 集成**"), pick("VF6", "**E5 已部署 v5.0 三 seed 集成**")
    return {
        "protocol": ("CFD-cell protocol: held-out units scored on every CFD anatomy cell (+ valid wall nodes for pressure); "
                     "case-balanced R2 (training_wss_min.metrics.casebalanced_field_metrics); readout "
                     f"training_wss_min/experiments/{EXPERIMENT}/readout_ckpt_best.json (consistency max abs diff "
                     f"{r['consistency']['max_abs_diff']:.1e} over {r['consistency']['n_runs']} runs)"),
        "cv5_oof": {
            "n_units": p_ens["n_units"], "recipe": "same recipe, CV5 patient-grouped 5 folds x 3 seeds, out-of-fold",
            "pressure_r2cb_ensemble3": round(p_ens["r2cb_all"], 4), "pressure_wall_r2cb_ensemble3": round(p_ens["r2cb_wall"], 4),
            "pressure_interior_r2cb_ensemble3": round(p_ens["r2cb_interior"], 4),
            "pressure_r2cb_single_seed_mean": round(p_sd["r2cb_all"]["mean"], 4), "pressure_r2cb_single_seed_sd": round(p_sd["r2cb_all"]["sd"], 4),
            "speed_r2cb_ensemble3": round(v_ens["r2cb_speed"], 4), "speed_r2cb_single_seed_mean": round(v_sd["r2cb_speed"]["mean"], 4),
            "speed_r2cb_single_seed_sd": round(v_sd["r2cb_speed"]["sd"], 4),
            "velocity_vector_rmse_m_s_ensemble3": round(v_ens["vector_rmse_m_s"], 4),
            "velocity_direction_cosine_ensemble3": round(v_ens["direction_cosine_casemean"], 4),
        },
        "recover8": {
            "n_units": 8, "note": "descriptive only (n = 8)",
            "pressure_r2cb_ensemble3": round(p8["r2cb_all"], 4), "speed_r2cb_ensemble3": round(v8["r2cb_speed"], 4),
            "previous_release_pressure_r2cb": round(e5p["r2cb_all"], 4), "previous_release_speed_r2cb": round(e5v["r2cb_speed"], 4),
        },
        "deployment_note": ("Deployment samples new interior queries inside the STL; equivalence of that sampling to the "
                            "CFD-cell evaluation protocol has not been established, so these numbers describe the model on CFD "
                            "geometry, not a deployed accuracy."),
    }


def build_release(output_root: Path = RELEASE_DIR.parent, *, project_root: Path = PROJECT_ROOT) -> Path:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / RELEASE_ID
    if destination.exists():
        raise FileExistsError(f"发布包已存在；不会覆盖冻结权重：{destination}")
    staging = Path(tempfile.mkdtemp(prefix=".volume_release_v52d_", dir=output_root))
    try:
        models, sources = [], []
        for family, field in (("PF6", "pressure"), ("VF6", "velocity")):
            for seed in SEEDS:
                source = Path(project_root) / "training_wss_min/runs" / EXPERIMENT / f"{family}_full265_s{seed}"
                if "训练完成" not in (source / "train.log").read_text(encoding="utf-8", errors="replace"):
                    raise RuntimeError(f"training not finished: {source}")
                relative = Path("models") / f"{family}_s{seed}"
                (staging / relative).mkdir(parents=True)
                for name in FILES:
                    shutil.copy2(source / name, staging / relative / name)
                models.append({"seed": seed, "field": field, "path": relative.as_posix()})
                sources.append({"field": field, "seed": seed, "path": str(source),
                                "checkpoint_sha256": file_sha256(source / "ckpt_best.pt")})
        (staging / "rules").mkdir()
        shutil.copy2(Path(project_root) / RULE, staging / "rules/flow_split_rule_train136.json")
        info = {
            "release": RELEASE_ID, "frozen_on": FROZEN_ON, "model_family": "PF6_VF6",
            "target": "pressure_velocity", "models": models, "source_runs": sources,
            "fields": VOLUME_FIELDS,
            "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
            "checkpoint": "ckpt_best.pt; train_loss-selected peak runs (full265, no validation split)",
            "ensemble_protocol": "restore physical units for each seed; arithmetic vector/scalar mean over 3 seeds per field",
            "pressure_reference": "p_rel = p - peak volume-weighted anatomy mean; no absolute reference pressure is available from STL",
            "velocity_frame": "model output in v5 atlas axes; world velocity = aligned velocity @ frame_rotation",
            "input_contract": {
                "geometry": "STL and confirmed anatomical centerline only; wall support and newly sampled interior queries",
                "features": "native V5 volume 18D plus log_q_branch_murray and log_tau0_murray (unchanged from PF6_VF6_peak_3seed_20260920)",
                "frame": "v5_atlas_frame_v1", "normalization": "packaged full265 (train265) global linear statistics",
                "boundary_protocol": "fixed training flow/boundary-condition family, not patient-specific flow measurements",
            },
            "data": DATA,
            "training": {"n_train": 265, "split": "split_v52p4_full265_train265_test8 (test = recover8)",
                         "experiment": EXPERIMENT, "tracking": TRACKING, "recipe_sources": "configs wss_local_wave2_20260912 (seed 1234) / wss_local_wave3_20260913 (seeds 7, 2025)"},
            "validation": validation_block(project_root),
            "replaces": "PF6_VF6_peak_3seed_20260920",
            "builder": {"module": "wss_deploy.build_v52d_volume_release", "git_commit": _git_commit(Path(project_root))},
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
