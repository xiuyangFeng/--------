"""Wave 3 (2026-09-13, user request): PF6 / VF6 at seeds 7 and 2025 with seed-matched P02 / V07 controls.

Together with the existing seed-1234 members (PF6_s1234 / VF6_s1234 from wave 2, P02_s1234 / V07_s1234 from
the volume-attention matrix and the wave-2 reruns P02r / V07r) this gives three paired seeds per target, so the
F6-augmented arms can be promoted to the pressure / velocity bases on three-seed evidence.

Arms (all single runs, one queue round on 4 GPUs x 2 slots):
  PF6_s7 / PF6_s2025     P02 + F6 Murray prior          V F6_s7 / VF6_s2025   V07 + F6 Murray prior
  P02_s7 / P02_s2025     P02 same-seed controls          V07_s7 / V07_s2025    V07 same-seed controls

Seed-7/2025 arms use seed-matched reference chains (refs/P00_s<seed> -> refs/P02_s<seed>, refs/V00_s<seed> ->
refs/V07_s<seed>) so that the control and the F6 arm of the same seed share every inherited initial tensor,
exactly as PF6_s1234 / P02r_s1234 did in wave 2.

    python -m training_wss_min.tools.prepare_wss_local_wave3
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave2 as W2

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave3_20260913"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
VOLUME_CONFIGS = W2.VOLUME_CONFIGS
F6 = W1.F6
BASES = {"P": ("P02", "P00"), "V": ("V07", "V00")}

ARMS = {
    "PF6_s7": ("P02 pressure + F6 Murray prior, seed 7", "P", 7, True),
    "PF6_s2025": ("P02 pressure + F6 Murray prior, seed 2025", "P", 2025, True),
    "VF6_s7": ("V07 velocity + F6 Murray prior, seed 7", "V", 7, True),
    "VF6_s2025": ("V07 velocity + F6 Murray prior, seed 2025", "V", 2025, True),
    "P02_s7": ("P02 same-seed control, seed 7", "P", 7, False),
    "P02_s2025": ("P02 same-seed control, seed 2025", "P", 2025, False),
    "V07_s7": ("V07 same-seed control, seed 7", "V", 7, False),
    "V07_s2025": ("V07 same-seed control, seed 2025", "V", 2025, False),
}


def reference_chain(kind: str, seed: int) -> Path:
    """Seed-matched copies of the P00 -> P02 (V00 -> V07) reference configs; initialization donors, never trained."""
    base, root = BASES[kind]
    root_cfg = json.loads((VOLUME_CONFIGS / f"{root}_s1234.json").read_text())
    if root_cfg["train"].get("init_reference_config"):
        raise ValueError(f"{root} is expected to be the chain root")
    root_cfg["name"] = f"{NAME}/refs/{root}_s{seed}"
    root_cfg["train"]["seed"] = seed
    root_cfg["notes"] = f"seed-{seed} reference copy of {root}_s1234; initialization donor only, never trained"
    C.ExpConfig.from_dict(root_cfg)
    root_out = CONFIGS / "refs" / f"{root}_s{seed}.json"
    W1.save(root_out, root_cfg)
    base_cfg = json.loads((VOLUME_CONFIGS / f"{base}_s1234.json").read_text())
    if Path(base_cfg["train"]["init_reference_config"]).name != f"{root}_s1234.json":
        raise ValueError(f"{base} is expected to pair to {root}")
    base_cfg["name"] = f"{NAME}/refs/{base}_s{seed}"
    base_cfg["train"]["seed"] = seed
    base_cfg["train"]["init_reference_config"] = str(root_out)
    base_cfg["notes"] = f"seed-{seed} reference copy of {base}_s1234; initialization donor only, never trained"
    C.ExpConfig.from_dict(base_cfg)
    base_out = CONFIGS / "refs" / f"{base}_s{seed}.json"
    W1.save(base_out, base_cfg)
    return base_out


def configure(aid: str) -> dict:
    title, kind, seed, with_f6 = ARMS[aid]
    reference = reference_chain(kind, seed)
    cfg = json.loads(reference.read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"wave-3 {aid}: {title}; seed {seed}; single run; paired to {reference.name}."
    cfg["train"]["init_reference_config"] = str(reference)
    cfg["train"]["log_loss_components"] = True
    if with_f6:
        base_stats = json.loads(Path(cfg["data"]["feature_stats_path"]).read_text())
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + F6
        cfg["data"]["point_features_root"] = [str(W1.FLOW_ROOT)]
        stats = dict(base_stats)
        stats.update(W2.new_key_stats(F6, "volume"))
        stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
        W1.save(stats_path, stats)
        cfg["data"]["feature_stats_path"] = str(stats_path)
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    if not W2.VOLUME_UNION.is_file():
        raise FileNotFoundError("wave-2 volume Murray feature stats are required")
    arms = []
    for aid, (title, kind, seed, with_f6) in ARMS.items():
        cfg = configure(aid)
        filename = f"{aid}.json"
        W1.save(CONFIGS / filename, cfg)
        base = BASES[kind][0]
        anchor_path = VOLUME_CONFIGS / f"{base}_s1234.json"
        arms.append(dict(id=aid, title=title, config=filename, phase=0, modules=(["F6"] if with_f6 else []), seed=seed,
                         base=base, kind=kind, parent=(f"{base}_s{seed}" if with_f6 else None), depends_on=[],
                         evaluate=["best", "last"], single_change=True,
                         config_diff_vs_anchor=W1.declared_diff(json.loads(anchor_path.read_text()), cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN),
        control_id=None,
        external_reference_runs={
            "P02_s1234 (historical)": str(RUNS / "volume_attention_20260910/P02_s1234/eval/ckpt_best/metrics.json"),
            "V07_s1234 (historical)": str(RUNS / "volume_attention_20260910/V07_s1234/eval/ckpt_best/metrics.json"),
            "PF6_s1234 (wave-2)": str(RUNS / W2.NAME / "PF6_s1234" / "eval/ckpt_best/metrics.json"),
            "VF6_s1234 (wave-2)": str(RUNS / W2.NAME / "VF6_s1234" / "eval/ckpt_best/metrics.json"),
            "P02r_s1234 (wave-2 control)": str(RUNS / W2.NAME / "P02r_s1234" / "eval/ckpt_best/metrics.json"),
            "V07r_s1234 (wave-2 control)": str(RUNS / W2.NAME / "V07r_s1234" / "eval/ckpt_best/metrics.json"),
        },
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("compare PF6_s<seed> with P02_s<seed> and VF6_s<seed> with V07_s<seed> (same inherited init); "
                      "three paired seeds per target together with the wave-2 seed-1234 pair; exposed test34; "
                      "promotion to base requires the paired delta to keep its sign on all three seeds"),
        volume_note=W2.__doc__.split("\n")[0],
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
