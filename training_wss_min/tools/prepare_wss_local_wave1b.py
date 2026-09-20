"""Wave-1b follow-up: robustness and decomposition of the wave-1 stand-out (X5, Murray flow-share prior).

Eight single runs, one round of the 4-GPU x 2-slot queue:
  X0_s7 / X0_s2025      C1 control at two more seeds (calibrates the run-to-run band with X0_s1234)
  X5_s7 / X5_s2025      Murray prior at the same two seeds (paired with the controls above)
  X5q_s1234             log_q_branch_murray only        (which component carries the gain?)
  X5t_s1234             log_tau0_murray only
  X5F1_s1234            X5 + X2 bend-referenced angles  (do the two feature groups add?)
  X5S1a_s1234           X5 + X7 tangent-frame patch     (feature x structure)

Seed-7/2025 arms use a seed-matched reference chain (refs/C1_s<seed> -> refs/M2_s<seed>) so that
X0 and X5 of the same seed share every inherited initial tensor, exactly as the s1234 arms do.

    python -m training_wss_min.tools.prepare_wss_local_wave1b
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave1b_20260912"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
ANCHOR = W1.ANCHOR
WAVE1_EXP = W1.EXP

ARMS = {
    "X0_s7": ("C1 control, seed 7", [], 7),
    "X0_s2025": ("C1 control, seed 2025", [], 2025),
    "X5_s7": ("F6 Murray prior, seed 7", ["F6"], 7),
    "X5_s2025": ("F6 Murray prior, seed 2025", ["F6"], 2025),
    "X5q_s1234": ("F6 component: log_q_branch_murray only", ["F6q"], 1234),
    "X5t_s1234": ("F6 component: log_tau0_murray only", ["F6t"], 1234),
    "X5F1_s1234": ("F6 + F1 bend-referenced angles (X5 + X2)", ["F6", "F1"], 1234),
    "X5S1a_s1234": ("F6 + tangent-frame query patch (X5 + X7)", ["F6", "frame"], 1234),
}


def reference_chain(seed: int) -> Path:
    """Seed-matched copies of the C1 -> M2 reference configs (never trained)."""
    if seed == 1234:
        return ANCHOR
    c1 = json.loads(ANCHOR.read_text())
    m2_path = Path(c1["train"]["init_reference_config"])
    m2 = json.loads(m2_path.read_text())
    m2["name"] = f"{NAME}/refs/M2_s{seed}"
    m2["train"]["seed"] = seed
    m2["train"]["init_reference_config"] = None
    m2["notes"] = f"seed-{seed} reference copy of {m2_path.name}; initialization donor only, never trained"
    C.ExpConfig.from_dict(m2)
    m2_out = CONFIGS / "refs" / f"M2_s{seed}.json"
    W1.save(m2_out, m2)
    c1["name"] = f"{NAME}/refs/C1_s{seed}"
    c1["train"]["seed"] = seed
    c1["train"]["init_reference_config"] = str(m2_out)
    c1["notes"] = f"seed-{seed} reference copy of C1; initialization donor only, never trained"
    C.ExpConfig.from_dict(c1)
    c1_out = CONFIGS / "refs" / f"C1_s{seed}.json"
    W1.save(c1_out, c1)
    return c1_out


def configure(aid: str, title: str, modules: list[str], seed: int) -> dict:
    cfg = json.loads(ANCHOR.read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"wave-1b follow-up {aid}: {title}; modules={modules}; seed {seed}; single run."
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(reference_chain(seed))
    cfg["train"]["log_loss_components"] = True
    extra_keys: list[str] = []
    for module in modules:
        if module == "F6":
            extra_keys += W1.F6
        elif module == "F6q":
            extra_keys += ["log_q_branch_murray"]
        elif module == "F6t":
            extra_keys += ["log_tau0_murray"]
        elif module == "F1":
            extra_keys += W1.F1
        elif module == "frame":
            cfg["model"]["query_patch_frame"] = "tangent"
        else:
            raise ValueError(module)
    if extra_keys:
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + extra_keys
        cfg["data"]["point_features_root"] = [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)]
        stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
        W1.save(stats_path, W1.frozen_feature_stats(extra_keys))
        cfg["data"]["feature_stats_path"] = str(stats_path)
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    anchor = json.loads(ANCHOR.read_text())
    arms = []
    for aid, (title, modules, seed) in ARMS.items():
        cfg = configure(aid, title, modules, seed)
        filename = f"{aid}.json"
        W1.save(CONFIGS / filename, cfg)
        diff = W1.declared_diff(anchor, cfg)
        arms.append(dict(id=aid, title=title, config=filename, phase=0, modules=modules, seed=seed,
                         parent="X0_s1234" if seed == 1234 else f"X0_s{seed}", depends_on=[],
                         evaluate=["best", "last"], single_change=len(modules) <= 1,
                         config_diff_vs_anchor=diff))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(ANCHOR), anchor_run=str(W1.ANCHOR_RUN),
        control_id=None,
        external_reference_runs={"X0_s1234 (wave-1 control)": str(RUNS / W1.NAME / "X0_s1234" / "eval/ckpt_best/metrics.json"),
                                 "X5_s1234 (wave-1)": str(RUNS / W1.NAME / "X5_s1234" / "eval/ckpt_best/metrics.json")},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("robustness: compare X5_s<seed> with X0_s<seed> pairwise (same init); decomposition and "
                      "combinations are single seed 1234 against wave-1 X0/X5; exposed test34; no significance claims"),
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
