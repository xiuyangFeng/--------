"""v5.1 wave 2b (2026-09-16): T3n residual stacking on X5D_v51, on test34 (three paired seeds) and out-of-fold on cv3_v51.

  T3n_s1234 / _s7 / _s2025     X5D_v51 + stage-1 log_wss_base input, residual target ln(WSS) - log_wss_base, no focus / gate;
                               paired with X5D_v51_s<seed> (same C1_s<seed> init, one appended zero-init column)
  T3n_f0/f1/f2_s1234           the same on cv3_v51 fold k, paired with X5D_v51_f<k>_s1234 (wave 2a); read only on the held-out fold
Stage-1 sidecar: data_wss_v5/views_v5_1/wss_min_cascade_v1 (tools/make_cascade_sidecar on the wave-2a fold models).
Fold arms use fold-train offset statistics (residual mean/std over the fold's own train cases) to keep the fold protocol nested.

    python -m training_wss_min.tools.prepare_wss_v51_wave2b
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_v51_wave1 as V1
from training_wss_min.tools import prepare_wss_v51_wave2a as V2A

ROOT = C.PROJECT_ROOT
NAME = "wss_v51_wave2b_20260916"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
CASCADE_ROOT = V1.V51 / "wss_min_cascade_v1"
BASE = ["log_wss_base"]
ARMS = {f"T3n_s{s}": (f"T3n residual stacking on X5D_v51, seed {s}", s, None) for s in (1234, 7, 2025)}
ARMS.update({f"T3n_f{k}_s1234": (f"T3n residual stacking on cv3_v51 fold {k}, seed 1234", 1234, k) for k in range(3)})


def base_stats(fold: int | None, input_features: list[str]) -> dict:
    """z-scores of log_wss_base (train136 or fold-train), computed on the v5.1 views + cascade sidecar."""
    tag = "train136" if fold is None else f"fold{fold}_train"
    path = EXP / "feature_stats" / f"union_logwssbase_wall_{tag}.json"
    if not path.is_file():
        anchor = C.ExpConfig.from_json(W1.ANCHOR)
        split = V1.SPLIT if fold is None else V2A.CV_ROOT / f"fold{fold}.json"
        stats = D.load_wss_stats(V1.STATS if fold is None else V2A.CV_ROOT / f"wss_global_stats_fold{fold}_train.json")
        cases = D.load_partition(str(split), "train", stats, strict=True, target="wss", data_root=V1.VIEW,
                                 required_frame_version=anchor.data.required_frame_version, extra_point_features=tuple(BASE),
                                 point_features_root=[str(V1.GEOM_ROOT), str(V1.FLOW_ROOT), str(CASCADE_ROOT)])
        union = D.compute_feature_stats(cases, tuple(BASE), anchor.data.curvature_transform)
        union["_provenance"] = {"train_split": str(split), "n_cases": len(cases), "cascade_manifest": str(CASCADE_ROOT / "cascade_manifest.json")}
        W1.save(path, union)
    return json.loads(path.read_text())


def fold_offset_stats(fold: int) -> Path:
    """Fold-train copy of the cascade offset statistics: residual ln(WSS+eps) - log_wss_base over the fold's train cases only."""
    path = EXP / "feature_stats" / f"wss_global_stats_fold{fold}_train_offset_logwssbase.json"
    if not path.is_file():
        stats = json.loads((V2A.CV_ROOT / f"wss_global_stats_fold{fold}_train.json").read_text())
        eps = float(stats["eps"]); s1 = s2 = 0.0; n = 0
        for cid in stats["train_units"]:
            with np.load(V1.VIEW / cid / "bundle.npz", allow_pickle=True) as b, np.load(CASCADE_ROOT / cid / "features.npz") as f:
                steps = b["steps"].tolist(); wss = b["wall_wss"][steps.index(int(b["peak_step"]))].astype(np.float64)
                if not np.array_equal(b["wall_node_id_cas"], f["wall_node_id_cas"]):
                    raise ValueError(f"cascade sidecar rows differ from the bundle: {cid}")
                resid = np.log(np.clip(wss, 0.0, None) + eps) - f[f"wall_{BASE[0]}"].astype(np.float64)
            s1 += resid.sum(); s2 += (resid ** 2).sum(); n += len(resid)
        mean = s1 / n; std = float(np.sqrt(max(s2 / n - mean ** 2, 1e-12)))
        out = {k: v for k, v in stats.items() if k != "train_units"}
        out["offset"] = {"feature": BASE[0], "mean": float(mean), "std": std, "n_points": int(n),
                         "note": f"residual target over cv3_v51 fold{fold} train cases only (nested); log_wss_base = out-of-fold X5D_v51 prediction"}
        out["source_stats"] = str(V2A.CV_ROOT / f"wss_global_stats_fold{fold}_train.json")
        W1.save(path, out)
    return path


def configure(aid: str, title: str, seed: int, fold: int | None) -> dict:
    anchor_raw = json.loads(W1.ANCHOR.read_text())
    union = V1.union_feature_stats(list(anchor_raw["data"]["input_features"]), anchor_raw["data"]["curvature_transform"])
    cfg = V1.configure(f"X5D_v51_s{seed}", title, seed, "base", union)
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"v5.1 wave-2b {aid}: {title}; residual stacking on the stage-1 sidecar; fold {fold}; deployment-legal inputs only."
    cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + BASE
    cfg["data"]["point_features_root"] = [str(V1.GEOM_ROOT), str(V1.FLOW_ROOT), str(CASCADE_ROOT)]
    cfg["data"]["target_log_offset_feature"] = BASE[0]
    if fold is None:
        cfg["data"]["wss_stats_path"] = str(CASCADE_ROOT / "wss_global_stats_train136_offset_logwssbase.json")
        stats = {k: union[k] for k in union if k in cfg["data"]["input_features"] or k == "_provenance"}
        stats[BASE[0]] = base_stats(None, cfg["data"]["input_features"])[BASE[0]]
    else:
        cfg["data"]["split_path"] = str(V2A.CV_ROOT / f"fold{fold}.json")
        cfg["data"]["wss_stats_path"] = str(fold_offset_stats(fold))
        stats = json.loads(V2A.fold_feature_stats(fold, [f for f in cfg["data"]["input_features"] if f not in BASE]).read_text())
        stats[BASE[0]] = base_stats(fold, cfg["data"]["input_features"])[BASE[0]]
    stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
    W1.save(stats_path, stats)
    cfg["data"]["feature_stats_path"] = str(stats_path)
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    manifest_path = CASCADE_ROOT / "cascade_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("build the stage-1 sidecar first (tools/make_cascade_sidecar on the wave-2a fold models)")
    manifest = json.loads(manifest_path.read_text())
    anchor = json.loads(W1.ANCHOR.read_text())
    arms = []
    for aid, (title, seed, fold) in ARMS.items():
        cfg = configure(aid, title, seed, fold)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        parent = f"X5D_v51_s{seed} (wave 1)" if fold is None else f"X5D_v51_f{fold}_s{seed} (wave 2a)"
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=["F6", "density_aug", "cascade"], seed=seed, fold=fold,
                         parent=parent, depends_on=[], evaluate=["best", "last"], single_change=True, config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={f"X5D_v51_s{s}": str(ROOT / "training_wss_min/runs/wss_v51_wave1_20260916" / f"X5D_v51_s{s}" / "eval/ckpt_best/metrics.json") for s in (1234, 7, 2025)},
        cascade={"manifest": str(manifest_path), "offset_mean": manifest["offset_mean"], "offset_std": manifest["offset_std"],
                 "stage1_ln_r2_train_oof_mean": manifest["stage1_ln_r2_train_oof_mean"], "stage1_ln_r2_test_mean": manifest["stage1_ln_r2_test_mean"]},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("T3n_s<seed> vs X5D_v51_s<seed> on test34 (three paired deltas); T3n_f<k> vs X5D_v51_f<k> on the held-out fold, "
                      "pooled over three folds = every train136 case once (the cv3 out-of-fold evidence §21.7 lacked)")))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
