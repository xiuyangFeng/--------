"""v5.1 wave 1 (2026-09-16): retrain on the corrected/deduplicated data (train136/test34) and the three follow-up arms.

Data: data_wss_v5/views_v5_1/* (26 cases re-solved after the RCR outlet-area fix, LIU_WEN_QI / HOU_SHEN_QIAN removed;
tracking doc §23–§25).  Every arm keeps the X5D recipe (C1 + F6 Murray prior + density augmentation) and the C1_s<seed>
paired-initialisation chain; feature standardisation is recomputed on train136 of the new views (one union file).

  X5D_v51_s{1234,7,2025,11,2026}  new deployment base (five-seed ensemble)
  X5Dcap_s{1234,7,2025}           protocol-cap variant: F6 keys replaced by log_q_branch_murray_cap / log_tau0_murray_cap
                                  (sidecar v1.4; Murray on the virtual-cap radii = the CFD outlet protocol), paired with X5D_v51
  X5Ddual_s{1234,7,2025}          dual-scale query patch: K16 offsets normalised by local spacing + a 4 mm ball of 16 coverage
                                  samples with mask (§22.6 A2), zero-initialised second branch
  X5Dnoise_s{1234,7,2025}         boundary-noise augmentation (sigma 0.1/0.2/0.3 mm, corr 2 mm, p=0.5; density aug still on)

    python -m training_wss_min.tools.prepare_wss_v51_wave1
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave4 as W4

ROOT = C.PROJECT_ROOT
NAME = "wss_v51_wave1_20260916"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
V51 = ROOT / "data_wss_v5/views_v5_1"
VIEW = V51 / "wss_min_view_v1"
GEOM_ROOT = V51 / "wss_min_geom_v2"
FLOW_ROOT = V51 / "wss_min_flowref_v1"
DENSITY_ROOT = V51 / "wss_min_density_v1"
NOISE_ROOT = V51 / "wss_min_noise_v1"
SPLIT = VIEW / "split_V5_train136_test34.json"
STATS = VIEW / "wss_global_stats_train136.json"
CAP = ["log_q_branch_murray_cap", "log_tau0_murray_cap"]
LEVELS, PROB = [70, 50, 35, 25], 0.6
NOISE_LEVELS, NOISE_PROB = ["0.1", "0.2", "0.3"], 0.5
RADIUS_MM, RADIUS_K = 4.0, 16
ARMS = {}
for seed in (1234, 7, 2025, 11, 2026):
    ARMS[f"X5D_v51_s{seed}"] = ("X5D on v5.1 data (train136), seed %d" % seed, seed, "base")
for seed in (1234, 7, 2025):
    ARMS[f"X5Dcap_s{seed}"] = ("X5D with protocol-cap Murray keys (log_q/log_tau0 on virtual-cap radii), seed %d" % seed, seed, "cap")
    ARMS[f"X5Ddual_s{seed}"] = ("X5D + dual-scale query patch (spacing-normalised K16 + 4 mm ball of 16), seed %d" % seed, seed, "dual")
    ARMS[f"X5Dnoise_s{seed}"] = ("X5D + boundary-noise augmentation (0.1/0.2/0.3 mm, p=0.5), seed %d" % seed, seed, "noise")


def check_roots() -> dict:
    for p in (SPLIT, STATS, GEOM_ROOT / "geom_manifest.json", FLOW_ROOT / "flowref_manifest.json", DENSITY_ROOT / "density_manifest.json",
              NOISE_ROOT / "noise_manifest.json"):
        if not p.is_file():
            raise FileNotFoundError(f"v5.1 views incomplete: {p} (run wss_v5/cluster/refresh_v5_1.slurm and build_noise_sidecars first)")
    split = json.loads(SPLIT.read_text())
    if len(split["train_cases"]) != 136 or len(split["test_cases"]) != 34:
        raise RuntimeError("v5.1 split must be train136/test34")
    dens = json.loads((DENSITY_ROOT / "density_manifest.json").read_text()); noise = json.loads((NOISE_ROOT / "noise_manifest.json").read_text())
    flow = json.loads((FLOW_ROOT / "flowref_manifest.json").read_text())
    if dens["cases"] != 170 or sorted(dens["levels"]) != sorted(LEVELS):
        raise RuntimeError("density sidecars incomplete for v5.1")
    if noise["cases"] != 170 or [str(s) for s in noise["sigmas"]] != NOISE_LEVELS:
        raise RuntimeError("noise sidecars incomplete for v5.1")
    if "log_tau0_murray_cap" not in flow["features"]:
        raise RuntimeError("flowref sidecar must be v1.4 (log_tau0_murray_cap)")
    return {"split": str(SPLIT), "density_version": dens["version"], "noise_version": noise["version"], "flowref_version": flow["version"]}


def union_feature_stats(base_features: list[str], curvature_transform: str) -> dict:
    """Standardisation statistics recomputed on v5.1 train136 for every input column any arm uses."""
    path = EXP / "feature_stats" / "union_all_train136.json"
    if path.is_file():
        return json.loads(path.read_text())
    keys = tuple(base_features) + tuple(W1.F6) + tuple(CAP)
    extra = tuple(k for k in keys if k in C.SIDECAR_FEATURE_KEYS)
    stats = D.load_wss_stats(STATS)
    cases = D.load_partition(str(SPLIT), "train", stats, strict=True, target="wss", data_root=VIEW,
                             required_frame_version=json.loads(W1.ANCHOR.read_text())["data"]["required_frame_version"],
                             extra_point_features=extra, point_features_root=[str(GEOM_ROOT), str(FLOW_ROOT)])
    if len(cases) != 136:
        raise RuntimeError(f"expected train136, loaded {len(cases)}")
    union = D.compute_feature_stats(cases, keys, curvature_transform)
    union["_provenance"] = {"train_split": str(SPLIT), "n_cases": len(cases), "data_root": str(VIEW),
                            "sidecar_roots": [str(GEOM_ROOT), str(FLOW_ROOT)], "curvature_transform": curvature_transform,
                            "note": "v5.1: recomputed on train136 (not C1's frozen stats); every arm of this wave selects its columns from here"}
    W1.save(path, union)
    return union


def configure(aid: str, title: str, seed: int, kind: str, union: dict) -> dict:
    cfg = json.loads(W1.ANCHOR.read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"v5.1 wave-1 {aid}: {title}; data = views_v5_1 train136/test34 (26 RCR-corrected reruns, 2 duplicates removed); C1_s{seed} paired init."
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(W4.reference_chain(seed))
    cfg["train"]["log_loss_components"] = True
    flow_keys = CAP if kind == "cap" else W1.F6
    cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + flow_keys
    cfg["data"]["data_root"] = str(VIEW)
    cfg["data"]["split_path"] = str(SPLIT)
    cfg["data"]["wss_stats_path"] = str(STATS)
    cfg["data"]["point_features_root"] = [str(GEOM_ROOT), str(FLOW_ROOT)]
    stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
    W1.save(stats_path, {k: union[k] for k in union if k in cfg["data"]["input_features"] or k == "_provenance"})
    cfg["data"]["feature_stats_path"] = str(stats_path)
    cfg["data"].update(density_aug_root=str(DENSITY_ROOT), density_aug_levels=LEVELS, density_aug_prob=PROB)
    if kind == "dual":
        cfg["data"].update(query_patch_radius_mm=RADIUS_MM, query_patch_radius_k=RADIUS_K, query_patch_offset_norm="spacing")
        cfg["model"]["query_patch_radius_k"] = RADIUS_K
    if kind == "noise":
        cfg["data"].update(noise_aug_root=str(NOISE_ROOT), noise_aug_levels=NOISE_LEVELS, noise_aug_prob=NOISE_PROB)
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    provenance = check_roots()
    anchor = json.loads(W1.ANCHOR.read_text())
    union = union_feature_stats(list(anchor["data"]["input_features"]), anchor["data"]["curvature_transform"])
    arms = []
    for aid, (title, seed, kind) in ARMS.items():
        cfg = configure(aid, title, seed, kind, union)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        modules = {"base": ["F6", "density_aug"], "cap": ["F6cap", "density_aug"], "dual": ["F6", "density_aug", "dual_patch"], "noise": ["F6", "density_aug", "noise_aug"]}[kind]
        arms.append(dict(id=aid, title=title, config=f"{aid}.json", phase=0, modules=modules, seed=seed, fold=None,
                         parent=(f"X5D_s{seed} (wave-6/6b, v5.0 data)" if kind == "base" else f"X5D_v51_s{seed}"), depends_on=[],
                         evaluate=["best", "last"], single_change=kind != "base", config_diff_vs_anchor=W1.declared_diff(anchor, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={f"X5D_s{s} (v5.0 data)": str(RUNS / ("wss_local_wave6_20260915" if s in (1234, 7, 2025) else "wss_local_wave6b_20260915") / f"X5D_s{s}" / "eval/ckpt_best/metrics.json") for s in (1234, 7, 2025, 11, 2026)},
        data_v51=provenance, density={"root": str(DENSITY_ROOT), "levels": LEVELS, "prob": PROB},
        noise={"root": str(NOISE_ROOT), "levels": NOISE_LEVELS, "prob": NOISE_PROB}, dual_patch={"radius_mm": RADIUS_MM, "radius_k": RADIUS_K, "offset_norm": "spacing"},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("X5D_v51 five seeds = new base (Pa-mean ensemble on the new test34; compare with the old X5D only as a label-change "
                      "effect, the 4 corrected test cases and 2 removed train cases make the numbers non-comparable one-to-one); "
                      "X5Dcap / X5Ddual / X5Dnoise vs X5D_v51 same seed, three paired deltas; band ±0.034 physical / ±0.009 normalised per seed"),
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
