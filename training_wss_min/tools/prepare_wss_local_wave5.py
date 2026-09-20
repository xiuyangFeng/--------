"""Wave 5 (2026-09-13): stenosis index on top of Murray / capfit, and the patient-grouped 3-fold check of X5 vs X5+X11.

Twelve single runs, one round of the 4-GPU x 3-slot queue (deployment-legal inputs only):
  X5I_s1234 / _s7 / _s2025     C1 + F6 Murray + [log_r_over_rdistal]      paired with X5_s<seed>  (same C1_s<seed> chain)
  X5AI_s1234 / _s7 / _s2025    C1 + capfit + [log_r_over_rdistal]         paired with X5A_s<seed>
  X5_f0/f1/f2_s1234            C1 + F6 Murray on cv3 fold k (train = other two folds, test = held-out fold)
  X5X11_f0/f1/f2_s1234         C1 + F6 + S3 section-token context on the same folds (paired with X5_f<k>: same init)

The stenosis index (sidecar v1.3, wall_flowref_v1) is ln(R_local / R_distal(segment)) with the Murray split's
distal-radius recipe; wave 4 showed Murray's log_tau0 helps ILO through this term rather than through the flow
split, and capfit weakened it.  Fold arms use fold-train log_z statistics and fold-train feature z-scores.

    python -m training_wss_min.tools.prepare_wss_local_wave5
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C, dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave1b as W1B
from training_wss_min.tools import prepare_wss_local_wave4 as W4

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave5_20260913"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
CV_ROOT = W1.VIEW / "cv3_20260913"
IDX = ["log_r_over_rdistal"]
F6 = W1.F6
CAPFIT = W4.CAPFIT
W2 = "wss_local_wave2_20260912"

ARMS = {
    "X5I_s1234": ("F6 Murray + stenosis index ln(R/R_distal), seed 1234", ["F6", "idx"], 1234, None),
    "X5I_s7": ("F6 Murray + stenosis index, seed 7", ["F6", "idx"], 7, None),
    "X5I_s2025": ("F6 Murray + stenosis index, seed 2025", ["F6", "idx"], 2025, None),
    "X5AI_s1234": ("capfit split + stenosis index, seed 1234", ["capfit", "idx"], 1234, None),
    "X5AI_s7": ("capfit split + stenosis index, seed 7", ["capfit", "idx"], 7, None),
    "X5AI_s2025": ("capfit split + stenosis index, seed 2025", ["capfit", "idx"], 2025, None),
    "X5_f0_s1234": ("X5 (F6 Murray) on cv3 fold 0", ["F6"], 1234, 0),
    "X5_f1_s1234": ("X5 (F6 Murray) on cv3 fold 1", ["F6"], 1234, 1),
    "X5_f2_s1234": ("X5 (F6 Murray) on cv3 fold 2", ["F6"], 1234, 2),
    "X5X11_f0_s1234": ("X5 + S3 section-token context on cv3 fold 0", ["F6", "section"], 1234, 0),
    "X5X11_f1_s1234": ("X5 + S3 section-token context on cv3 fold 1", ["F6", "section"], 1234, 1),
    "X5X11_f2_s1234": ("X5 + S3 section-token context on cv3 fold 2", ["F6", "section"], 1234, 2),
}


def reference_chain(seed: int) -> Path:
    if seed == 1234:
        return W1.ANCHOR
    path = W1B.CONFIGS / "refs" / f"C1_s{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"seed-{seed} reference chain missing: {path}")
    return path


def idx_stats() -> dict:
    """train138 statistics of the stenosis index (same recipe as the wave-1 union file)."""
    path = EXP / "feature_stats" / "union_idx_wall_train138.json"
    if not path.is_file():
        anchor = C.ExpConfig.from_json(W1.ANCHOR)
        stats = D.load_wss_stats(anchor.data.wss_stats_path)
        cases = D.load_partition(anchor.data.split_path, "train", stats, strict=True, target="wss",
                                 data_root=anchor.data.data_root,
                                 required_frame_version=anchor.data.required_frame_version,
                                 extra_point_features=tuple(IDX),
                                 point_features_root=[str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)])
        if len(cases) != 138:
            raise RuntimeError(f"expected train138, loaded {len(cases)}")
        union = D.compute_feature_stats(cases, tuple(IDX), anchor.data.curvature_transform)
        union["_provenance"] = {"train_split": anchor.data.split_path, "n_cases": len(cases),
                                "sidecar_roots": [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)], "sidecar_version": "v1.3"}
        W1.save(path, union)
    return json.loads(path.read_text())


def fold_feature_stats(fold: int, input_features: list[str]) -> Path:
    """Fold-train z-scores for every non-xyz input feature (replaces C1's frozen train138 stats inside the fold)."""
    path = EXP / "feature_stats" / f"fold{fold}_train_feature_stats.json"
    if not path.is_file():
        anchor = C.ExpConfig.from_json(W1.ANCHOR)
        split = CV_ROOT / f"fold{fold}.json"
        stats = D.load_wss_stats(CV_ROOT / f"wss_global_stats_fold{fold}_train.json")
        extra = tuple(f for f in input_features if f in C.SIDECAR_FEATURE_KEYS)
        cases = D.load_partition(str(split), "train", stats, strict=True, target="wss",
                                 data_root=anchor.data.data_root,
                                 required_frame_version=anchor.data.required_frame_version,
                                 extra_point_features=extra,
                                 point_features_root=[str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)])
        expected = json.loads(split.read_text())["counts"]["train"]
        if len(cases) != expected:
            raise RuntimeError(f"fold{fold}: expected {expected} train cases, loaded {len(cases)}")
        out = D.compute_feature_stats(cases, tuple(input_features), anchor.data.curvature_transform)
        out["_provenance"] = {"train_split": str(split), "n_cases": len(cases), "wss_stats": str(CV_ROOT / f"wss_global_stats_fold{fold}_train.json"),
                              "sidecar_roots": [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)],
                              "curvature_transform": anchor.data.curvature_transform,
                              "note": "fold-train statistics for every non-xyz input feature of the X5 recipe"}
        W1.save(path, out)
    return path


def configure(aid: str, title: str, modules: list[str], seed: int, fold: int | None) -> dict:
    cfg = json.loads(W1.ANCHOR.read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"wave-5 {aid}: {title}; modules={modules}; seed {seed}; fold {fold}; single run; deployment-legal inputs only."
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(reference_chain(seed))
    cfg["train"]["log_loss_components"] = True
    extra_keys: list[str] = []
    for module in modules:
        if module == "F6":
            extra_keys += F6
        elif module == "capfit":
            extra_keys += CAPFIT
        elif module == "idx":
            extra_keys += IDX
        elif module == "section":
            cfg["model"].update(section_context=True, section_context_bin_mm=4.0, section_context_max_bins=64,
                                section_context_layers=2, section_context_heads=4, section_context_hidden=128)
            cfg["data"]["section_tokens"] = True
        else:
            raise ValueError(module)
    cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + extra_keys
    cfg["data"]["point_features_root"] = [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)]
    if fold is None:
        stats = W1.frozen_feature_stats([k for k in extra_keys if k in F6])
        if any(k in CAPFIT for k in extra_keys):
            union = W4.capfit_stats()
            for key in CAPFIT:
                stats[key] = union[key]
        if any(k in IDX for k in extra_keys):
            union = idx_stats()
            for key in IDX:
                stats[key] = union[key]
        stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
        W1.save(stats_path, stats)
        cfg["data"]["feature_stats_path"] = str(stats_path)
    else:
        cfg["data"]["split_path"] = str(CV_ROOT / f"fold{fold}.json")
        cfg["data"]["wss_stats_path"] = str(CV_ROOT / f"wss_global_stats_fold{fold}_train.json")
        cfg["data"]["feature_stats_path"] = str(fold_feature_stats(fold, list(cfg["data"]["input_features"])))
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    if not (CV_ROOT / "cv3_summary.json").is_file():
        raise FileNotFoundError("build the cv3 splits first (tools/make_cv3_splits)")
    manifest = json.loads((W1.FLOW_ROOT / "flowref_manifest.json").read_text())
    if manifest["version"] != "v1.3" or "log_r_over_rdistal" not in manifest["features"]:
        raise RuntimeError("flowref sidecar v1.3 (log_r_over_rdistal) required")
    anchor = json.loads(W1.ANCHOR.read_text())
    arms = []
    for aid, (title, modules, seed, fold) in ARMS.items():
        cfg = configure(aid, title, modules, seed, fold)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        if modules == ["F6", "idx"]:
            parent = f"X5_s{seed}"
        elif modules == ["capfit", "idx"]:
            parent = f"X5A_s{seed}"
        elif "section" in modules:
            parent = f"X5_f{fold}_s{seed}"
        else:
            parent = None
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=modules, seed=seed, fold=fold,
                         parent=parent, depends_on=[], evaluate=["best", "last"],
                         single_change=("idx" in modules or "section" in modules),
                         config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={
            "X5_s1234 (wave-1)": str(RUNS / W1.NAME / "X5_s1234" / "eval/ckpt_best/metrics.json"),
            "X5_s7 (wave-1b)": str(RUNS / W1B.NAME / "X5_s7" / "eval/ckpt_best/metrics.json"),
            "X5_s2025 (wave-1b)": str(RUNS / W1B.NAME / "X5_s2025" / "eval/ckpt_best/metrics.json"),
            "X5A_s1234 (wave-4)": str(RUNS / W4.NAME / "X5A_s1234" / "eval/ckpt_best/metrics.json"),
            "X5A_s7 (wave-4)": str(RUNS / W4.NAME / "X5A_s7" / "eval/ckpt_best/metrics.json"),
            "X5A_s2025 (wave-4)": str(RUNS / W4.NAME / "X5A_s2025" / "eval/ckpt_best/metrics.json"),
            "X5X11_s1234 (wave-2, test34)": str(RUNS / W2 / "X5X11_s1234" / "eval/ckpt_best/metrics.json"),
        },
        cv3=json.loads((CV_ROOT / "cv3_summary.json").read_text()),
        sidecar={"pack": manifest["pack"], "version": manifest["version"], "new_key": IDX},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("X5I_s<seed> vs X5_s<seed> and X5AI_s<seed> vs X5A_s<seed> (same inherited init; one appended "
                      "zero-init column), three paired seeds each, exposed test34; fold arms are read only on their "
                      "held-out fold: X5X11_f<k> vs X5_f<k> paired per fold, pooled over the three folds = every "
                      "train138 case once; test34 is never used by fold arms; no significance claims"),
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
