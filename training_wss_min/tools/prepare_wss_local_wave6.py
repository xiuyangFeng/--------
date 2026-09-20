"""Wave 6 (2026-09-15): density augmentation for deployment robustness (route B).

Three single runs, paired with X5_s<seed> (same C1_s<seed> reference chain, identical inputs and architecture; the
only change is training-time density augmentation):
  X5D_s1234 / _s7 / _s2025   X5 + density_aug (levels 70/50/35/25 %, probability 0.6 per case per epoch; the
                             decimated clouds and their recomputed normals / curvature families come from
                             data_wss_v5/views/wss_min_density_v1, tools/build_density_sidecars)
Acceptance: test34 at full density (must stay within the X5 band) and the resample re-evaluation
(tools/deployment_resample_reeval) at 50 / 25 / 10 %, plus the STL-resampling deployment simulation.

    python -m training_wss_min.tools.prepare_wss_local_wave6
"""
from __future__ import annotations

import json

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave5 as W5

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave6_20260915"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
DENSITY_ROOT = ROOT / "data_wss_v5/views/wss_min_density_v1"
LEVELS = [70, 50, 35, 25]
PROB = 0.6
ARMS = {
    "X5D_s1234": ("X5 + density augmentation (70/50/35/25 %, p=0.6), seed 1234", 1234),
    "X5D_s7": ("X5 + density augmentation, seed 7", 7),
    "X5D_s2025": ("X5 + density augmentation, seed 2025", 2025),
}


def configure(aid: str, title: str, seed: int) -> dict:
    cfg = json.loads(W1.ANCHOR.read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"wave-6 {aid}: {title}; single run; deployment-legal inputs only; only training-time augmentation differs from X5."
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(W5.reference_chain(seed))
    cfg["train"]["log_loss_components"] = True
    cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + W1.F6
    cfg["data"]["point_features_root"] = [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)]
    stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
    W1.save(stats_path, W1.frozen_feature_stats(W1.F6))
    cfg["data"]["feature_stats_path"] = str(stats_path)
    cfg["data"].update(density_aug_root=str(DENSITY_ROOT), density_aug_levels=LEVELS, density_aug_prob=PROB)
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    manifest = json.loads((DENSITY_ROOT / "density_manifest.json").read_text())
    if manifest["cases"] != 172 or sorted(manifest["levels"]) != sorted(LEVELS):
        raise RuntimeError("density sidecars incomplete")
    anchor = json.loads(W1.ANCHOR.read_text())
    arms = []
    for aid, (title, seed) in ARMS.items():
        cfg = configure(aid, title, seed)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=["F6", "density_aug"], seed=seed, fold=None,
                         parent=f"X5_s{seed}", depends_on=[], evaluate=["best", "last"], single_change=True,
                         config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={
            "X5_s1234 (wave-1)": str(RUNS / W1.NAME / "X5_s1234" / "eval/ckpt_best/metrics.json"),
            "X5_s7 (wave-1b)": str(RUNS / "wss_local_wave1b_20260912" / "X5_s7" / "eval/ckpt_best/metrics.json"),
            "X5_s2025 (wave-1b)": str(RUNS / "wss_local_wave1b_20260912" / "X5_s2025" / "eval/ckpt_best/metrics.json"),
        },
        density={"root": str(DENSITY_ROOT), "levels": LEVELS, "prob": PROB, "manifest_version": manifest["version"]},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("X5D_s<seed> vs X5_s<seed> at full density on test34 (same inherited init, same inputs; only the "
                      "training-time density augmentation differs), then the resample re-evaluation at 50/25/10 % and the "
                      "STL-resampling simulation; acceptance = full-density delta within band and the 50 % delta within +-0.02"),
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
