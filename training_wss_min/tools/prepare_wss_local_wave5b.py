"""Wave 5b (2026-09-13, round 2): T3 hotspot cascade on X5, and cv3 at a second seed.

Twelve single runs, one round of the 4-GPU x 3-slot queue (deployment-legal inputs only):
  T3_s1234 / _s7 / _s2025    X5 + stage-1 prediction log_wss_base as input, residual target ln(WSS) - log_wss_base,
                             region-focused loss (top 20 % of the stage-1 prediction per case, outside weight 0.2),
                             eval gate: residual applied only inside that region (stage-1 kept elsewhere)
  T3n_s1234 / _s7 / _s2025   same input + residual target, no region focus, no gate (plain residual stacking control)
  X5_f0/f1/f2_s7             cv3 fold arms at seed 7 (paired with X5X11_f<k>_s7 through refs/C1_s7)
  X5X11_f0/f1/f2_s7
Stage-1 sidecar: tools/make_cascade_sidecar (out-of-fold cv3 X5 predictions for train138, fold models on test34).

    python -m training_wss_min.tools.prepare_wss_local_wave5b
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C, dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave5 as W5

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave5b_20260913"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
CASCADE_ROOT = ROOT / "data_wss_v5/views/wss_min_cascade_v1"
BASE = ["log_wss_base"]
F6 = W1.F6

ARMS = {
    "T3_s1234": ("T3 cascade: residual on stage-1 + region focus + gate, seed 1234", ["F6", "cascade", "focus"], 1234, None),
    "T3_s7": ("T3 cascade, seed 7", ["F6", "cascade", "focus"], 7, None),
    "T3_s2025": ("T3 cascade, seed 2025", ["F6", "cascade", "focus"], 2025, None),
    "T3n_s1234": ("residual stacking on stage-1 without focus/gate (control), seed 1234", ["F6", "cascade"], 1234, None),
    "T3n_s7": ("residual stacking control, seed 7", ["F6", "cascade"], 7, None),
    "T3n_s2025": ("residual stacking control, seed 2025", ["F6", "cascade"], 2025, None),
    "X5_f0_s7": ("X5 on cv3 fold 0, seed 7", ["F6"], 7, 0),
    "X5_f1_s7": ("X5 on cv3 fold 1, seed 7", ["F6"], 7, 1),
    "X5_f2_s7": ("X5 on cv3 fold 2, seed 7", ["F6"], 7, 2),
    "X5X11_f0_s7": ("X5 + S3 section-token context on cv3 fold 0, seed 7", ["F6", "section"], 7, 0),
    "X5X11_f1_s7": ("X5 + S3 section-token context on cv3 fold 1, seed 7", ["F6", "section"], 7, 1),
    "X5X11_f2_s7": ("X5 + S3 section-token context on cv3 fold 2, seed 7", ["F6", "section"], 7, 2),
}


def base_stats() -> dict:
    path = EXP / "feature_stats" / "union_logwssbase_wall_train138.json"
    if not path.is_file():
        anchor = C.ExpConfig.from_json(W1.ANCHOR)
        stats = D.load_wss_stats(anchor.data.wss_stats_path)
        cases = D.load_partition(anchor.data.split_path, "train", stats, strict=True, target="wss",
                                 data_root=anchor.data.data_root, required_frame_version=anchor.data.required_frame_version,
                                 extra_point_features=tuple(BASE),
                                 point_features_root=[str(W1.GEOM_ROOT), str(W1.FLOW_ROOT), str(CASCADE_ROOT)])
        if len(cases) != 138:
            raise RuntimeError(f"expected train138, loaded {len(cases)}")
        union = D.compute_feature_stats(cases, tuple(BASE), anchor.data.curvature_transform)
        union["_provenance"] = {"train_split": anchor.data.split_path, "n_cases": len(cases),
                                "sidecar_roots": [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT), str(CASCADE_ROOT)],
                                "cascade_manifest": str(CASCADE_ROOT / "cascade_manifest.json")}
        W1.save(path, union)
    return json.loads(path.read_text())


def configure(aid: str, title: str, modules: list[str], seed: int, fold: int | None) -> dict:
    cfg = json.loads(W1.ANCHOR.read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"wave-5b {aid}: {title}; modules={modules}; seed {seed}; fold {fold}; single run; deployment-legal inputs only."
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(W5.reference_chain(seed))
    cfg["train"]["log_loss_components"] = True
    extra_keys: list[str] = []
    roots = [str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)]
    for module in modules:
        if module == "F6":
            extra_keys += F6
        elif module == "section":
            cfg["model"].update(section_context=True, section_context_bin_mm=4.0, section_context_max_bins=64,
                                section_context_layers=2, section_context_heads=4, section_context_hidden=128)
            cfg["data"]["section_tokens"] = True
        elif module == "cascade":
            extra_keys += BASE
            roots.append(str(CASCADE_ROOT))
            cfg["data"]["target_log_offset_feature"] = BASE[0]
            cfg["data"]["wss_stats_path"] = str(CASCADE_ROOT / "wss_global_stats_train138_offset_logwssbase.json")
        elif module == "focus":
            cfg["train"].update(loss_region_feature=BASE[0], loss_region_quantile=0.8, loss_region_outside_weight=0.2)
            cfg["eval"]["residual_gate_quantile"] = 0.8
        else:
            raise ValueError(module)
    cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + extra_keys
    cfg["data"]["point_features_root"] = roots
    if fold is None:
        stats = W1.frozen_feature_stats([k for k in extra_keys if k in F6])
        if any(k in BASE for k in extra_keys):
            union = base_stats()
            for key in BASE:
                stats[key] = union[key]
        stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
        W1.save(stats_path, stats)
        cfg["data"]["feature_stats_path"] = str(stats_path)
    else:
        cfg["data"]["split_path"] = str(W5.CV_ROOT / f"fold{fold}.json")
        cfg["data"]["wss_stats_path"] = str(W5.CV_ROOT / f"wss_global_stats_fold{fold}_train.json")
        cfg["data"]["feature_stats_path"] = str(W5.fold_feature_stats(fold, list(cfg["data"]["input_features"])))
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    manifest_path = CASCADE_ROOT / "cascade_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("build the stage-1 sidecar first (tools/make_cascade_sidecar)")
    manifest = json.loads(manifest_path.read_text())
    anchor = json.loads(W1.ANCHOR.read_text())
    arms = []
    for aid, (title, modules, seed, fold) in ARMS.items():
        cfg = configure(aid, title, modules, seed, fold)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        if "cascade" in modules:
            parent = f"X5_s{seed}"
        elif "section" in modules:
            parent = f"X5_f{fold}_s{seed}"
        else:
            parent = None
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=modules, seed=seed, fold=fold,
                         parent=parent, depends_on=[], evaluate=["best", "last"],
                         single_change=("section" in modules or modules == ["F6", "cascade"]),
                         config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={
            "X5_s1234 (wave-1)": str(RUNS / W1.NAME / "X5_s1234" / "eval/ckpt_best/metrics.json"),
            "X5_s7 (wave-1b)": str(RUNS / "wss_local_wave1b_20260912" / "X5_s7" / "eval/ckpt_best/metrics.json"),
            "X5_s2025 (wave-1b)": str(RUNS / "wss_local_wave1b_20260912" / "X5_s2025" / "eval/ckpt_best/metrics.json"),
        },
        cascade={"manifest": str(manifest_path), "offset_mean": manifest["offset_mean"], "offset_std": manifest["offset_std"],
                 "stage1_ln_r2_train_oof_mean": manifest["stage1_ln_r2_train_oof_mean"],
                 "stage1_ln_r2_test_mean": manifest["stage1_ln_r2_test_mean"]},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("T3 / T3n vs X5_s<seed> (same inherited init; one appended zero-init column; different target "
                      "parametrisation), three paired seeds, exposed test34, stage-1 on test34 from 91-93-case fold models; "
                      "T3 vs T3n isolates the focus + gate; fold arms at seed 7 are read only on their held-out fold "
                      "and paired with the seed-1234 fold arms of wave 5; no significance claims"),
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
