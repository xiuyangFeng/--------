"""Wave 6b (2026-09-15, user decision): two more density-augmented seeds so X5D can be a five-seed deployment ensemble.

  X5D_s11 / X5D_s2026   identical to wave-6 X5D (X5 + density augmentation 70/50/35/25 %, p=0.6), seeds 11 / 2026,
                        paired with the wave-4 X5_s11 / X5_s2026 through the same C1_s<seed> reference chains.

    python -m training_wss_min.tools.prepare_wss_local_wave6b
"""
from __future__ import annotations

import json

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave4 as W4
from training_wss_min.tools import prepare_wss_local_wave6 as W6

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave6b_20260915"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
ARMS = {"X5D_s11": ("X5 + density augmentation, seed 11 (ensemble member)", 11),
        "X5D_s2026": ("X5 + density augmentation, seed 2026 (ensemble member)", 2026)}


def main() -> None:
    anchor = json.loads(W1.ANCHOR.read_text())
    arms = []
    for aid, (title, seed) in ARMS.items():
        cfg = json.loads(W1.ANCHOR.read_text())
        cfg["name"] = f"{NAME}/{aid}"
        cfg["notes"] = f"wave-6b {aid}: {title}; single run; same recipe as wave-6 X5D."
        cfg["train"]["seed"] = seed
        cfg["train"]["init_reference_config"] = str(W4.reference_chain(seed))
        cfg["train"]["log_loss_components"] = True
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + W1.F6
        cfg["data"]["point_features_root"] = [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)]
        stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
        W1.save(stats_path, W1.frozen_feature_stats(W1.F6))
        cfg["data"]["feature_stats_path"] = str(stats_path)
        cfg["data"].update(density_aug_root=str(W6.DENSITY_ROOT), density_aug_levels=W6.LEVELS, density_aug_prob=W6.PROB)
        C.ExpConfig.from_dict(cfg)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=["F6", "density_aug"], seed=seed, fold=None,
                         parent=f"X5_s{seed}", depends_on=[], evaluate=["best", "last"], single_change=True,
                         config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={
            "X5_s11 (wave-4)": str(RUNS / W4.NAME / "X5_s11" / "eval/ckpt_best/metrics.json"),
            "X5_s2026 (wave-4)": str(RUNS / W4.NAME / "X5_s2026" / "eval/ckpt_best/metrics.json"),
        },
        density={"root": str(W6.DENSITY_ROOT), "levels": W6.LEVELS, "prob": W6.PROB},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule="ensemble members: X5D five seeds (1234/7/2025 from wave 6 + 11/2026 here) vs X5 five seeds on test34; paired vs X5_s<seed>",
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
