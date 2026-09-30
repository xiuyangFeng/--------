"""§36 stage 2 (2026-09-27): CV5 confirmation of the screening winner T2 (loss_asym_under_weight = 2.0) + dose controls.

  X5Dcap_asym2_v52cv_f{0..4}_s{1234,7,2025}  (15)  = wss_v52_20260923/X5Dcap_v52cv_f{k}_s{seed} config + train.loss_asym_under_weight 2.0
  X5Dcap_asym15_s{seed}, X5Dcap_asym3_s{seed} (6)   = wss_v52_20260923/X5Dcap_v52ind_s{seed} config + weight 1.5 / 3.0 (IND dose controls)
No feature change -> feature z-scores and WSS stats are inherited from the source configs (same split/train partition).
Gate (pre-registered, matrix doc §2): 15 paired fold deltas vs X5Dcap_v52cv same (fold, seed): mean > +0.005 and >= 11/15 positive.

    python -m training_wss_min.tools.prepare_wss_v52_phys2
"""
from __future__ import annotations

import json

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_v52 as P

NAME = "wss_v52_phys2_20260927"
CONFIGS = P.MAIN / "training_wss_min/configs" / NAME
SOURCE = P.MAIN / "training_wss_min/configs/wss_v52_20260923"
SEEDS = (1234, 7, 2025)
FOLDS = range(5)


def make(src_name: str, aid: str, title: str, weight: float, phase: int, protocol: str, fold, seed: int, anchor: dict, arms: list) -> None:
    src = json.loads((SOURCE / f"{src_name}.json").read_text()); cfg = json.loads(json.dumps(src))
    cfg["name"] = f"{NAME}/{aid}"; cfg["notes"] = f"§36 stage 2 {aid}: {title}; base = wss_v52_20260923/{src_name}; single declared change train.loss_asym_under_weight={weight}."
    cfg["train"]["loss_asym_under_weight"] = weight
    for section, cls in {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg); P.save(CONFIGS / f"{aid}.json", cfg)
    arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=phase, modules=["F6", "density_aug", "cap_prior", "tail_loss"], seed=seed, fold=fold, protocol=protocol,
                     arm=aid.split("_v52cv")[0].rsplit("_s", 1)[0], parent=f"{src_name} (wss_v52_20260923)", depends_on=[], evaluate=["best", "last"], single_change=True,
                     config_diff_vs_anchor=P.declared_diff(anchor, cfg), config_diff_vs_parent=P.declared_diff(src, cfg)))


def main() -> None:
    anchor = json.loads(P.ANCHOR.read_text()); arms: list = []
    for seed in SEEDS:
        for k in FOLDS:
            make(f"X5Dcap_v52cv_f{k}_s{seed}", f"X5Dcap_asym2_v52cv_f{k}_s{seed}", f"T2 asym under-weight 2.0, CV5 fold {k}, seed {seed}", 2.0, 0, "CV5", k, seed, anchor, arms)
    for seed in SEEDS:
        make(f"X5Dcap_v52ind_s{seed}", f"X5Dcap_asym15_s{seed}", f"dose control: asym under-weight 1.5, IND, seed {seed}", 1.5, 1, "IND", None, seed, anchor, arms)
        make(f"X5Dcap_v52ind_s{seed}", f"X5Dcap_asym3_s{seed}", f"dose control: asym under-weight 3.0, IND, seed {seed}", 3.0, 1, "IND", None, seed, anchor, arms)
    P.save(CONFIGS / "matrix.json", dict(schema_version=1, experiment=NAME, anchor=str(P.ANCHOR), anchor_run=str(P.ANCHOR_RUN), control_id=None,
        external_reference_runs={**{f"X5Dcap_v52cv_f{k}_s{s}": str(P.RUNS / f"wss_v52_20260923/X5Dcap_v52cv_f{k}_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS for k in FOLDS},
                                 **{f"X5Dcap_v52ind_s{s}": str(P.RUNS / f"wss_v52_20260923/X5Dcap_v52ind_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS}},
        section="§36 stage 2", screening_readout=str(P.MAIN / "training_wss_min/experiments/wss_v52_phys_20260926/readout_phys_ckpt_best.md"),
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule="CV5: 15 paired (fold, seed) R2_cb deltas vs X5Dcap_v52cv: mean > +0.005 and >= 11/15 positive -> T2 confirmed; pooled OOF per seed mean +- sd; IND dose controls read as three paired deltas each (not a gate)"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
