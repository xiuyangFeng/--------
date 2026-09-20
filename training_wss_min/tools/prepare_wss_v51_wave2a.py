"""v5.1 wave 2a (2026-09-16): X5D_v51 on the patient-grouped cv3_v51 folds (stage-1 models for the T3n out-of-fold check).

  X5D_v51_f0/f1/f2_s1234   X5D recipe (C1 + F6 Murray + density augmentation) trained on two folds of train136, held-out fold
                           as 'test'; fold-train log_z WSS statistics and fold-train feature z-scores; test34 never used.
Their out-of-fold predictions become the stage-1 input log_wss_base (tools/make_cascade_sidecar --wave wss_v51_wave2a_20260916
--fold-run-pattern 'X5D_v51_f{k}_s{seed}'), after which prepare_wss_v51_wave2b builds the T3n arms.

    python -m training_wss_min.tools.prepare_wss_v51_wave2a
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_v51_wave1 as V1

ROOT = C.PROJECT_ROOT
NAME = "wss_v51_wave2a_20260916"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
CV_ROOT = V1.VIEW / "cv3_v51"
ARMS = {f"X5D_v51_f{k}_s1234": (f"X5D_v51 on cv3_v51 fold {k}, seed 1234", 1234, k) for k in range(3)}


def fold_feature_stats(fold: int, input_features: list[str]) -> Path:
    """Fold-train z-scores for every non-xyz input feature (v5.1 views)."""
    path = EXP / "feature_stats" / f"fold{fold}_train_feature_stats.json"
    if not path.is_file():
        anchor = C.ExpConfig.from_json(W1.ANCHOR)
        split = CV_ROOT / f"fold{fold}.json"
        stats = D.load_wss_stats(CV_ROOT / f"wss_global_stats_fold{fold}_train.json")
        extra = tuple(f for f in input_features if f in C.SIDECAR_FEATURE_KEYS)
        cases = D.load_partition(str(split), "train", stats, strict=True, target="wss", data_root=V1.VIEW,
                                 required_frame_version=anchor.data.required_frame_version, extra_point_features=extra,
                                 point_features_root=[str(V1.GEOM_ROOT), str(V1.FLOW_ROOT)])
        expected = json.loads(split.read_text())["counts"]["train"]
        if len(cases) != expected:
            raise RuntimeError(f"fold{fold}: expected {expected} train cases, loaded {len(cases)}")
        out = D.compute_feature_stats(cases, tuple(input_features), anchor.data.curvature_transform)
        out["_provenance"] = {"train_split": str(split), "n_cases": len(cases), "wss_stats": str(CV_ROOT / f"wss_global_stats_fold{fold}_train.json"),
                              "sidecar_roots": [str(V1.GEOM_ROOT), str(V1.FLOW_ROOT)], "curvature_transform": anchor.data.curvature_transform,
                              "note": "v5.1 fold-train statistics for every non-xyz input feature of the X5D recipe"}
        W1.save(path, out)
    return path


def configure(aid: str, title: str, seed: int, fold: int) -> dict:
    union = V1.union_feature_stats(list(json.loads(W1.ANCHOR.read_text())["data"]["input_features"]), json.loads(W1.ANCHOR.read_text())["data"]["curvature_transform"])
    cfg = V1.configure(aid.replace(f"_f{fold}", ""), title, seed, "base", union)  # X5D_v51 recipe on the new views
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"v5.1 wave-2a {aid}: {title}; fold {fold} of cv3_v51 (train = other two folds, test = held-out fold); test34 unused."
    cfg["data"]["split_path"] = str(CV_ROOT / f"fold{fold}.json")
    cfg["data"]["wss_stats_path"] = str(CV_ROOT / f"wss_global_stats_fold{fold}_train.json")
    cfg["data"]["feature_stats_path"] = str(fold_feature_stats(fold, list(cfg["data"]["input_features"])))
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    if not (CV_ROOT / "cv3_summary.json").is_file():
        raise FileNotFoundError("cv3_v51 splits missing (refresh_v5_1 step 8)")
    anchor = json.loads(W1.ANCHOR.read_text())
    arms = []
    for aid, (title, seed, fold) in ARMS.items():
        cfg = configure(aid, title, seed, fold)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=["F6", "density_aug"], seed=seed, fold=fold,
                         parent=f"X5D_v51_s{seed} (wave 1)", depends_on=[], evaluate=["best", "last"], single_change=False,
                         config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={}, cv3=json.loads((CV_ROOT / "cv3_summary.json").read_text()),
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule="fold arms are read only on their held-out fold; pooled over three folds = every train136 case once (stage-1 for T3n)"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
