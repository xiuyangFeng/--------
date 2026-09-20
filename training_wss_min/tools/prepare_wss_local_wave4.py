"""Wave 4 (2026-09-13, user decision: deployment-legal inputs only): cap-area-fitted flow split + ensemble seeds.

Arms (single runs, one queue round):
  X5A_s1234 / _s7 / _s2025 / _s11 / _s2026   C1 + [log_q_branch_capfit, log_tau0_capfit]  (sidecar v1.2: train138-fitted
                                             log(Q_ext/Q_int) = a log((r_cap_ext/r_cap_int)^2) + b; replaces the Murray R^3 share)
  X5_s11 / X5_s2026                          C1 + F6 Murray prior at two more seeds (5-seed ensemble with wave 1/1b)
  X5B_s1234                                  C1 + F6 Murray + capfit (both share families; does the model want both?)

Every arm pairs to the seed-matched C1 reference chain (refs/C1_s<seed> -> refs/M2_s<seed>; seed 1234 = C1 anchor,
7/2025 = wave-1b refs, 11/2026 = new refs here), so X5A_s<seed> and X5_s<seed> share every inherited initial tensor
and differ only in the two zero-initialised input columns.

    python -m training_wss_min.tools.prepare_wss_local_wave4
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C, dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave1b as W1B

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave4_20260913"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
CAPFIT = ["log_q_branch_capfit", "log_tau0_capfit"]
F6 = W1.F6

ARMS = {
    "X5A_s1234": ("cap-area-fitted split (capfit) replacing Murray, seed 1234", ["capfit"], 1234),
    "X5A_s7": ("capfit split, seed 7", ["capfit"], 7),
    "X5A_s2025": ("capfit split, seed 2025", ["capfit"], 2025),
    "X5A_s11": ("capfit split, seed 11", ["capfit"], 11),
    "X5A_s2026": ("capfit split, seed 2026", ["capfit"], 2026),
    "X5_s11": ("F6 Murray prior, seed 11 (ensemble member)", ["F6"], 11),
    "X5_s2026": ("F6 Murray prior, seed 2026 (ensemble member)", ["F6"], 2026),
    "X5B_s1234": ("F6 Murray + capfit together, seed 1234", ["F6", "capfit"], 1234),
}


def reference_chain(seed: int) -> Path:
    if seed == 1234:
        return W1.ANCHOR
    existing = W1B.CONFIGS / "refs" / f"C1_s{seed}.json"
    if existing.is_file():
        return existing
    c1 = json.loads(W1.ANCHOR.read_text())
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


def capfit_stats() -> dict:
    """train138 statistics of the two capfit keys (same recipe as the wave-1 union file)."""
    path = EXP / "feature_stats" / "union_capfit_wall_train138.json"
    if not path.is_file():
        anchor = C.ExpConfig.from_json(W1.ANCHOR)
        stats = D.load_wss_stats(anchor.data.wss_stats_path)
        cases = D.load_partition(anchor.data.split_path, "train", stats, strict=True, target="wss",
                                 data_root=anchor.data.data_root,
                                 required_frame_version=anchor.data.required_frame_version,
                                 extra_point_features=tuple(CAPFIT),
                                 point_features_root=[str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)])
        if len(cases) != 138:
            raise RuntimeError(f"expected train138, loaded {len(cases)}")
        union = D.compute_feature_stats(cases, tuple(CAPFIT), anchor.data.curvature_transform)
        union["_provenance"] = {"train_split": anchor.data.split_path, "n_cases": len(cases),
                                "sidecar_roots": [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)],
                                "rule": str(W1.FLOW_ROOT / "flow_split_rule_train138.json")}
        W1.save(path, union)
    return json.loads(path.read_text())


def configure(aid: str, title: str, modules: list[str], seed: int) -> dict:
    cfg = json.loads(W1.ANCHOR.read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"wave-4 {aid}: {title}; modules={modules}; seed {seed}; single run; deployment-legal inputs only."
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(reference_chain(seed))
    cfg["train"]["log_loss_components"] = True
    extra_keys: list[str] = []
    for module in modules:
        if module == "F6":
            extra_keys += F6
        elif module == "capfit":
            extra_keys += CAPFIT
        else:
            raise ValueError(module)
    stats = W1.frozen_feature_stats([k for k in extra_keys if k in F6])
    if any(k in CAPFIT for k in extra_keys):
        union = capfit_stats()
        for key in CAPFIT:
            if key in extra_keys:
                stats[key] = union[key]
    cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + extra_keys
    cfg["data"]["point_features_root"] = [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)]
    stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
    W1.save(stats_path, stats)
    cfg["data"]["feature_stats_path"] = str(stats_path)
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    rule_path = W1.FLOW_ROOT / "flow_split_rule_train138.json"
    if not rule_path.is_file():
        raise FileNotFoundError("fit the cap-area split rule first (tools/fit_flow_split_rule)")
    anchor = json.loads(W1.ANCHOR.read_text())
    arms = []
    for aid, (title, modules, seed) in ARMS.items():
        cfg = configure(aid, title, modules, seed)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=modules, seed=seed,
                         parent=(f"X5_s{seed}" if modules == ["capfit"] else None), depends_on=[],
                         evaluate=["best", "last"], single_change=len(modules) <= 1,
                         config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN),
        control_id=None,
        external_reference_runs={
            "X5_s1234 (wave-1)": str(RUNS / W1.NAME / "X5_s1234" / "eval/ckpt_best/metrics.json"),
            "X5_s7 (wave-1b)": str(RUNS / W1B.NAME / "X5_s7" / "eval/ckpt_best/metrics.json"),
            "X5_s2025 (wave-1b)": str(RUNS / W1B.NAME / "X5_s2025" / "eval/ckpt_best/metrics.json"),
            "X0_s1234 (wave-1 control)": str(RUNS / W1.NAME / "X0_s1234" / "eval/ckpt_best/metrics.json"),
            "X0_s7 (wave-1b control)": str(RUNS / W1B.NAME / "X0_s7" / "eval/ckpt_best/metrics.json"),
            "X0_s2025 (wave-1b control)": str(RUNS / W1B.NAME / "X0_s2025" / "eval/ckpt_best/metrics.json"),
        },
        flow_split_rule=json.loads(rule_path.read_text()),
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("X5A_s<seed> vs X5_s<seed> (same inherited init, only the two appended columns differ); "
                      "ensembles are computed offline from saved test predictions (X5 5 seeds vs X5A 5 seeds); "
                      "X5B single seed vs X5/X5A at 1234; exposed test34; no significance claims"),
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
