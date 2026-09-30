"""Learning curve on cv3_v51 (2026-09-20): does the X5D_v51 recipe still gain from more training cases?

Design (frozen before launch):
  * folds   = the existing patient-grouped cv3_v51 folds (train = other two folds, test = held-out fold; test34 unused);
  * fractions 25 / 50 / 75 % of each fold's training cases, nested per (fold, seed) and stratified by cohort family
    (AG / AAA / ILO); duplicate / related groups stay together; 100 % = the fold's full training set;
  * seeds 1234 / 7 / 2025.  LC100_f{k}_s1234 is not retrained: it IS the wave-2a fold run (external reference);
  * every subset gets its own log_z WSS statistics and feature z-scores (fitted on the subset only), exactly as a
    smaller dataset would; everything else in the X5D_v51 fold config is copied verbatim.
Reading: physical Pa R2_cb on the held-out fold, pooled over three folds (= every train136 case once per fraction
and seed); slope of R2 vs log(n_train) with 9 paired readings per fraction; the jet subset (true p99 > 40 Pa) is read
separately from the per-case metrics.  Single readings are never ranked.

    python -m training_wss_min.tools.prepare_wss_learning_curve            # CPU, run from the frozen copy
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D
from wss_v5.views.wss_min_view import write_wss_stats

MAIN = Path("/public/newhome/cy/Digital_twin/GNN")           # every path written into configs points at the main tree
NAME = "wss_learning_curve_20260920"
CONFIGS = MAIN / "training_wss_min/configs" / NAME
EXP = MAIN / "training_wss_min/experiments" / NAME
RUNS = MAIN / "training_wss_min/runs"
VIEW = MAIN / "data_wss_v5/views_v5_1/wss_min_view_v1"
CV = VIEW / "cv3_v51"
LC = VIEW / "cv3_v51_lc"                                      # subset splits + subset WSS statistics
SOURCE_CONFIGS = MAIN / "training_wss_min/configs/wss_v51_wave2a_20260916"
SOURCE_RUNS = RUNS / "wss_v51_wave2a_20260916"
REFS = MAIN / "training_wss_min/configs/wss_local_wave1b_20260912/refs"
ANCHOR = MAIN / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"
FRACTIONS = (25, 50, 75)
SEEDS = (1234, 7, 2025)
FOLDS = (0, 1, 2)
FAMILIES = ("AG", "AAA", "ILO")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"refusing to change preregistered file: {path}")
    path.write_text(text)


def reference_config(seed: int) -> Path:
    path = ANCHOR if seed == 1234 else REFS / f"C1_s{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def nested_subsets(train_cases: list[str], groups: list[list[str]], seed: int, fold: int) -> dict[int, list[str]]:
    """Cohort-stratified nested subsets: 25% ⊂ 50% ⊂ 75% of the fold's training cases (grouped cases move together)."""
    unit_of = {m: tuple(sorted(g)) for g in groups for m in g}
    units, seen = [], set()
    for cid in train_cases:
        unit = unit_of.get(cid, (cid,))
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    rng = np.random.default_rng(int(np.random.SeedSequence([20260920, seed, fold]).generate_state(1)[0]))
    order: list[str] = []
    per_family = {}
    for family in FAMILIES:
        fam = [u for u in units if u[0].split("/")[0] == family]
        perm = rng.permutation(len(fam))
        per_family[family] = [fam[i] for i in perm]
    subsets = {}
    for frac in FRACTIONS:
        chosen = []
        for family in FAMILIES:
            fam = per_family[family]
            n_cases = sum(len(u) for u in fam)
            target = int(round(frac / 100.0 * n_cases))
            taken, count = [], 0
            for unit in fam:
                if count >= target:
                    break
                taken.extend(unit)
                count += len(unit)
            chosen.extend(taken)
        # keep the source ordering so the split file is deterministic and readable
        subsets[frac] = [c for c in train_cases if c in set(chosen)]
    for lo, hi in zip(FRACTIONS[:-1], FRACTIONS[1:]):
        if not set(subsets[lo]) <= set(subsets[hi]):
            raise RuntimeError(f"subsets not nested for seed {seed} fold {fold}: {lo} ⊄ {hi}")
    return subsets


def write_subset_split(fold: int, seed: int, frac: int, cases: list[str]) -> tuple[Path, Path]:
    src_path = CV / f"fold{fold}.json"
    src = json.loads(src_path.read_text())
    tag = f"LC{frac}_f{fold}_s{seed}"
    payload = {k: src[k] for k in ("schema_version", "pipeline", "data_root", "required_frame_version", "id_format", "created_at")}
    payload.update(
        split_version=f"cv3_v51_lc/{tag} of {src['split_version']}",
        source_split=str(src_path), source_split_sha256=sha256(src_path),
        duplicate_geometry_groups=src.get("duplicate_geometry_groups", []),
        explicit_related_groups=src.get("explicit_related_groups", []),
        train_cases=cases, val_cases=[], test_cases=list(src["test_cases"]), unused_cases=list(src["unused_cases"]),
        counts={"train": len(cases), "val": 0, "test": len(src["test_cases"])},
        expected_counts={"train": len(cases), "val": 0, "test": len(src["test_cases"])},
        learning_curve={"fraction_percent": frac, "fold": fold, "subset_seed": seed, "full_fold_train": src["counts"]["train"],
                        "by_family": {f: sum(1 for c in cases if c.startswith(f + "/")) for f in FAMILIES},
                        "nested": "25 ⊂ 50 ⊂ 75 within the same (fold, seed); stratified by cohort family; groups kept together"},
        note=(f"learning-curve subset: {frac}% of cv3_v51 fold{fold} training cases (nested, seed {seed}); "
              "test = the held-out fold; the frozen test34 stays unused"),
    )
    split_path = LC / f"{tag}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if split_path.exists() and split_path.read_text() != text:
        raise FileExistsError(f"refusing to change preregistered split: {split_path}")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(text)
    stats_path = LC / f"wss_global_stats_{tag}_train.json"
    if not stats_path.is_file():
        write_wss_stats(cases, VIEW, split_path, filename=f"{LC.name}/wss_global_stats_{tag}_train.json",
                        scope=f"train subset {tag} only ({len(cases)} cases, valid anatomy wall nodes)")
    return split_path, stats_path


def subset_feature_stats(tag: str, split_path: Path, stats_path: Path, cfg_src: dict) -> Path:
    """Subset-train z-scores for every non-xyz input feature (same recipe as wave-2a fold_feature_stats)."""
    path = EXP / "feature_stats" / f"{tag}_train_feature_stats.json"
    if path.is_file():
        return path
    input_features = list(cfg_src["data"]["input_features"])
    stats = D.load_wss_stats(stats_path)
    extra = tuple(f for f in input_features if f in C.SIDECAR_FEATURE_KEYS)
    cases = D.load_partition(str(split_path), "train", stats, strict=True, target="wss", data_root=VIEW,
                             required_frame_version=cfg_src["data"]["required_frame_version"], extra_point_features=extra,
                             point_features_root=list(cfg_src["data"]["point_features_root"]))
    expected = json.loads(split_path.read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"{tag}: expected {expected} train cases, loaded {len(cases)}")
    out = D.compute_feature_stats(cases, tuple(input_features), cfg_src["data"]["curvature_transform"])
    out["_provenance"] = {"train_split": str(split_path), "n_cases": len(cases), "wss_stats": str(stats_path),
                          "sidecar_roots": list(cfg_src["data"]["point_features_root"]),
                          "curvature_transform": cfg_src["data"]["curvature_transform"],
                          "note": f"learning-curve subset statistics ({tag}) for every non-xyz input feature of the X5D_v51 recipe"}
    save(path, out)
    return path


def make_config(tag: str, fold: int, seed: int, frac: int, split_path: Path, stats_path: Path, feat_path: Path) -> dict:
    cfg = json.loads((SOURCE_CONFIGS / f"X5D_v51_f{fold}_s1234.json").read_text())
    cfg["name"] = f"{NAME}/{tag}"
    cfg["notes"] = (f"learning curve {tag}: X5D_v51 recipe on {frac}% of cv3_v51 fold{fold} training cases (nested subset, seed {seed}); "
                    f"subset-fitted log_z WSS stats and feature z-scores; held-out fold as test; test34 unused; C1_s{seed} paired init.")
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(reference_config(seed))
    cfg["data"]["split_path"] = str(split_path)
    cfg["data"]["wss_stats_path"] = str(stats_path)
    cfg["data"]["feature_stats_path"] = str(feat_path)
    for section, cls in {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def declared_diff(anchor: dict, cfg: dict) -> dict:
    diff = {}
    for section in ("data", "model", "train", "eval"):
        for key in sorted(set(anchor[section]) | set(cfg[section])):
            before, after = anchor[section].get(key), cfg[section].get(key)
            if before != after:
                diff[f"{section}.{key}"] = {"anchor": before, "arm": after}
    return diff


def main() -> None:
    if not (CV / "cv3_summary.json").is_file():
        raise FileNotFoundError(CV / "cv3_summary.json")
    for fold in FOLDS:
        src_cfg = SOURCE_CONFIGS / f"X5D_v51_f{fold}_s1234.json"
        run_metrics = SOURCE_RUNS / f"X5D_v51_f{fold}_s1234/eval/ckpt_best/metrics.json"
        if not src_cfg.is_file() or not run_metrics.is_file():
            raise FileNotFoundError(f"wave-2a fold {fold} config/run missing: {src_cfg} / {run_metrics}")
    anchor = json.loads(ANCHOR.read_text())
    arms, subset_table = [], {}
    order = ([(frac, fold, seed) for seed in SEEDS for frac in FRACTIONS for fold in FOLDS]
             + [(100, fold, seed) for seed in SEEDS if seed != 1234 for fold in FOLDS])
    for frac, fold, seed in order:
        tag = f"LC{frac}_f{fold}_s{seed}"
        src = json.loads((CV / f"fold{fold}.json").read_text())
        cfg_src = json.loads((SOURCE_CONFIGS / f"X5D_v51_f{fold}_s1234.json").read_text())
        if frac == 100:
            split_path = CV / f"fold{fold}.json"
            stats_path = CV / f"wss_global_stats_fold{fold}_train.json"
            feat_path = Path(cfg_src["data"]["feature_stats_path"])
            cases = list(src["train_cases"])
        else:
            groups = [g for g in src.get("duplicate_geometry_groups", []) + src.get("explicit_related_groups", [])
                      if all(m in src["train_cases"] for m in g)]
            cases = nested_subsets(list(src["train_cases"]), groups, seed, fold)[frac]
            split_path, stats_path = write_subset_split(fold, seed, frac, cases)
            feat_path = subset_feature_stats(tag, split_path, stats_path, cfg_src)
        cfg = make_config(tag, fold, seed, frac, split_path, stats_path, feat_path)
        save(CONFIGS / f"{tag}.json", cfg)
        by_family = {f: sum(1 for c in cases if c.startswith(f + "/")) for f in FAMILIES}
        subset_table[tag] = {"fraction_percent": frac, "fold": fold, "seed": seed, "n_train": len(cases), "by_family": by_family,
                             "n_test": src["counts"]["test"], "split": str(split_path), "wss_stats": str(stats_path),
                             "feature_stats": str(feat_path)}
        arms.append(dict(id=tag, title=f"X5D_v51 recipe, {frac}% of fold{fold} train ({len(cases)} cases), seed {seed}",
                         config=f"{tag}.json", phase=0, modules=["F6", "density_aug"], seed=seed, fold=fold,
                         fraction_percent=frac, n_train=len(cases),
                         parent=f"X5D_v51_f{fold}_s1234 (wave 2a)", depends_on=[], evaluate=["best", "last"],
                         single_change=False, config_diff_vs_anchor=declared_diff(anchor, cfg)))
        print(tag, len(cases), by_family, flush=True)
    external = {f"LC100_f{k}_s1234": str(SOURCE_RUNS / f"X5D_v51_f{k}_s1234/eval/ckpt_best/metrics.json") for k in FOLDS}
    save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(ANCHOR), anchor_run=str(RUNS / "wss_direct_recovery_20260912/C1_s1234"),
        control_id=None, external_reference_runs=external,
        cv3=json.loads((CV / "cv3_summary.json").read_text()),
        learning_curve={"fractions_percent": list(FRACTIONS) + [100], "seeds": list(SEEDS), "folds": list(FOLDS),
                        "subset_rule": "nested per (fold, seed), stratified by cohort family, grouped cases kept together, "
                                       "subset-fitted WSS log_z stats and feature z-scores",
                        "subsets": subset_table,
                        "full_fold_seed1234": "LC100_f{k}_s1234 = wss_v51_wave2a_20260916/X5D_v51_f{k}_s1234 (not retrained)"},
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule=("held-out-fold Pa R2_cb pooled over three folds per (fraction, seed); slope of R2 vs log n_train from "
                      "9 paired readings per fraction; jet subset (true p99 > 40 Pa) read from per-case metrics; "
                      "no single reading is ranked; test34 never used")))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
