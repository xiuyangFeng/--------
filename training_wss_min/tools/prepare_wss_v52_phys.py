"""§36 screening arms (2026-09-26): 1D-physics prior (P1-P3) and tail objectives (T1-T3) on the IND protocol
(train 170 library / test 91 recovered), seeds 1234/7/2025, paired against wss_v52_20260923/X5Dcap_v52ind_s{seed}.
Every arm = the X5Dcap_v52ind_s{seed} config with ONE declared change (P2/P3 add the feature and the residual target).
Feature z-scores are refitted on the IND train partition with the phys1d sidecar root appended.

    python -m training_wss_min.tools.prepare_wss_v52_phys
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_v52 as P

NAME = "wss_v52_phys_20260926"
CONFIGS = P.MAIN / "training_wss_min/configs" / NAME
EXP = P.MAIN / "training_wss_min/experiments" / NAME
PHYS = P.MAIN / "data_wss_v5/views_v5_2_full_20260923/wss_min_phys1d_v1"
SOURCE = P.MAIN / "training_wss_min/configs/wss_v52_20260923"
SEEDS = (1234, 7, 2025)
ARMS = {
    "X5Dcap_womfeat": dict(title="X5Dcap + 1D Womersley prior as input features (log_tau_1d_wom, log_wom_alpha)", add_features=["log_tau_1d_wom", "log_wom_alpha"], offset=None, train={}),
    "X5Dcap_womres": dict(title="X5Dcap + log_tau_1d_wom input + residual target ln(tau/tau_1D_wom)", add_features=["log_tau_1d_wom", "log_wom_alpha"], offset="log_tau_1d_wom", train={}),
    "X5Dcap_poisres": dict(title="X5Dcap + log_tau_1d_pois input + residual target ln(tau/tau_1D_pois) (Womersley ablation)", add_features=["log_tau_1d_pois"], offset="log_tau_1d_pois", train={}),
    "X5Dcap_tw": dict(title="X5Dcap + target-magnitude loss weighting (loss_weight_target, alpha 2)", add_features=[], offset=None, train={"loss_weight_target": True}),
    "X5Dcap_asym2": dict(title="X5Dcap + asymmetric under-prediction loss weight 2.0", add_features=[], offset=None, train={"loss_asym_under_weight": 2.0}),
    "X5Dcap_pin95": dict(title="X5Dcap + pinball quantile 0.95 (lambda 0.2 unchanged)", add_features=[], offset=None, train={"pinball_quantile": 0.95}),
}


def feature_stats(tag: str, split_path: Path, stats_path: Path, cfg: dict) -> Path:
    from training_wss_min import dataset as D
    path = EXP / "feature_stats" / f"{tag}_train_feature_stats.json"
    if path.is_file():
        return path
    input_features = list(cfg["data"]["input_features"]); stats = D.load_wss_stats(stats_path)
    extra = tuple(f for f in input_features if f in C.SIDECAR_FEATURE_KEYS)
    roots = list(cfg["data"]["point_features_root"])
    cases = D.load_partition(str(split_path), "train", stats, strict=True, target="wss", data_root=P.VIEW, required_frame_version="v5_atlas_frame_v1",
                             extra_point_features=extra, point_features_root=roots)
    expected = json.loads(split_path.read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"{tag}: expected {expected} train cases, loaded {len(cases)}")
    out = D.compute_feature_stats(cases, tuple(input_features), cfg["data"]["curvature_transform"])
    out["_provenance"] = {"train_split": str(split_path), "n_cases": len(cases), "wss_stats": str(stats_path), "sidecar_roots": roots,
                          "curvature_transform": cfg["data"]["curvature_transform"], "note": f"§36 {tag}: IND train-partition z-scores"}
    P.save(path, out)
    return path


def main() -> None:
    if not (PHYS / "phys1d_manifest.json").is_file():
        raise FileNotFoundError(PHYS)
    anchor = json.loads(P.ANCHOR.read_text()); arms = []
    for arm, spec in ARMS.items():
        feat_path = None
        for seed in SEEDS:
            src = json.loads((SOURCE / f"X5Dcap_v52ind_s{seed}.json").read_text())
            cfg = json.loads(json.dumps(src))
            aid = f"{arm}_s{seed}"
            cfg["name"] = f"{NAME}/{aid}"; cfg["notes"] = f"§36 {aid}: {spec['title']}; base = wss_v52_20260923/X5Dcap_v52ind_s{seed} (IND protocol); single declared change."
            feats = list(cfg["data"]["input_features"]) + [f for f in spec["add_features"] if f not in cfg["data"]["input_features"]]
            cfg["data"]["input_features"] = feats
            if spec["add_features"]:
                cfg["data"]["point_features_root"] = list(cfg["data"]["point_features_root"]) + [str(PHYS)]
            if spec["offset"]:
                cfg["data"]["target_log_offset_feature"] = spec["offset"]
            for k, v in spec["train"].items():
                cfg["train"][k] = v
            if spec["add_features"]:
                if feat_path is None:
                    feat_path = feature_stats(arm, Path(cfg["data"]["split_path"]), Path(cfg["data"]["wss_stats_path"]), cfg)
                cfg["data"]["feature_stats_path"] = str(feat_path)
            for section, cls in {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}.items():
                unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
                if unknown:
                    raise ValueError(f"unknown {section} fields: {unknown}")
            C.ExpConfig.from_dict(cfg)
            P.save(CONFIGS / f"{aid}.json", cfg)
            arms.append(dict(id=aid, title=f"{spec['title']}, seed {seed}", config=f"{aid}.json", phase=0, modules=["F6", "density_aug", "cap_prior"] + (["phys1d"] if spec["add_features"] else []) + (["tail_loss"] if spec["train"] else []),
                             seed=seed, fold=None, protocol="IND", arm=arm, parent=f"X5Dcap_v52ind_s{seed} (wss_v52_20260923)", depends_on=[], evaluate=["best", "last"],
                             single_change=True, config_diff_vs_anchor=P.declared_diff(anchor, cfg),
                             config_diff_vs_parent=P.declared_diff(src, cfg)))
    P.save(CONFIGS / "matrix.json", dict(schema_version=1, experiment=NAME, anchor=str(P.ANCHOR), anchor_run=str(P.ANCHOR_RUN), control_id=None,
        external_reference_runs={f"X5Dcap_v52ind_s{s}": str(P.RUNS / f"wss_v52_20260923/X5Dcap_v52ind_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS},
        section="§36", phys1d_manifest=str(PHYS / "phys1d_manifest.json"), arms_spec={k: {kk: vv for kk, vv in v.items()} for k, v in ARMS.items()},
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule="IND Pa R2_cb per seed paired with X5Dcap_v52ind same seed (three paired deltas: mean > +0.010 and 3/3 same sign -> CV5 confirmation); jet subset delta >= 0; tail under-prediction fraction; no single reading ranked"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
