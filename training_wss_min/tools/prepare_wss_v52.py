"""v5.2 baseline retraining (2026-09-23): the frozen X5D_v51 recipe (and its protocol-cap Murray variant X5Dcap) on the
v5.2 data (170 library cases + 91 recovered units; 11 library cases rebuilt with the end-radius fix).

Two protocols decided by the user on 2026-09-23:
  IND  train = the 170 library cases, test = the 91 recovered units (independent set; same reading as the frozen-model
       screening that gave X5D_v51 0.749);   5 seeds x {X5D, X5Dcap}
  CV5  patient-grouped 5-fold CV over all 261 cases (ILO before/after of one patient in one fold; namesakes across
       cohorts grouped conservatively; ZHU_ZI_HAI/KANG_XI_MING duplicates already excluded); held-out fold = 'test',
       no separate validation (selection = train_loss as in the recipe); 5 folds x 3 seeds x {X5D, X5Dcap}
Every partition gets its own log_z WSS statistics and feature z-scores fitted on its training cases only; everything
else in the recipe configs is copied verbatim (paths moved to the assembled v5.2 views root).

    python -m training_wss_min.tools.prepare_wss_v52        # CPU (feature statistics load every training partition)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from training_wss_min import config as C
from training_wss_min import dataset as D
from wss_v5.views.wss_min_view import write_wss_stats

MAIN = Path("/public/newhome/cy/Digital_twin/GNN")
NAME = "wss_v52_20260923"
CONFIGS = MAIN / "training_wss_min/configs" / NAME
EXP = MAIN / "training_wss_min/experiments" / NAME
RUNS = MAIN / "training_wss_min/runs"
VIEW = MAIN / "data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1"
GEOM = MAIN / "data_wss_v5/views_v5_2_full_20260923/wss_min_geom_v2"
FLOW = MAIN / "data_wss_v5/views_v5_2_full_20260923/wss_min_flowref_v1"
DENS = MAIN / "data_wss_v5/views_v5_2_full_20260923/wss_min_density_v1"
CV = VIEW / "cv5_v52"
SOURCE = MAIN / "training_wss_min/configs/wss_v51_wave1_20260916"
ANCHOR = MAIN / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"
ANCHOR_RUN = RUNS / "wss_v51_wave1_20260916/X5D_v51_s1234"
REFS = {1234: ANCHOR, 7: MAIN / "training_wss_min/configs/wss_local_wave1b_20260912/refs/C1_s7.json",
        2025: MAIN / "training_wss_min/configs/wss_local_wave1b_20260912/refs/C1_s2025.json",
        11: MAIN / "training_wss_min/configs/wss_local_wave4_20260913/refs/C1_s11.json",
        2026: MAIN / "training_wss_min/configs/wss_local_wave4_20260913/refs/C1_s2026.json"}
LIB_SPLIT = MAIN / "data_wss_v5/views_v5_1/wss_min_view_v1/split_V5_train136_test34.json"
NEW_MANIFEST = MAIN / "docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/new_units_manifest.json"
REGISTRY_DIR = MAIN / "wss_pinn/configs/splits"
SEEDS_IND = (1234, 7, 2025, 11, 2026)
SEEDS_CV = (1234, 7, 2025)
FOLDS = 5
SPLIT_SEED = 20260923
RECIPES = {"X5D": "X5D_v51", "X5Dcap": "X5Dcap"}
DUP_GROUPS = [["AAA/unruputer/LIU_WEN_QI", "AAA/unruputer/ZHU_ZI_HAI"], ["AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"refusing to change preregistered file: {path}")
    path.write_text(text)


def patient_group(cid: str) -> str:
    """Same rule as wss_v5.sources.Registry.patient_group: ILO/<NAME>-k/<phase> -> NAME; else the case name."""
    parts = cid.split("/")
    if parts[0] == "ILO":
        return parts[1].rsplit("-", 1)[0]
    return parts[-1]


def stratum(cid: str) -> str:
    parts = cid.split("/")
    if parts[0] == "ILO":
        return "ILO/" + parts[1].rsplit("-", 1)[1]
    return parts[0] + "/" + parts[1]


def cv5_assign(cases: list[str]) -> dict[str, int]:
    """Deterministic, stratified, patient-grouped fold assignment: groups sorted by sha256(seed:group) inside each
    stratum, dealt round-robin over the folds with a rotating start so that every stratum is balanced."""
    groups: dict[str, list[str]] = {}
    for cid in cases:
        groups.setdefault(patient_group(cid), []).append(cid)
    by_stratum: dict[str, list[str]] = {}
    for key, members in groups.items():
        by_stratum.setdefault(stratum(sorted(members)[0]), []).append(key)
    fold_of: dict[str, int] = {}
    start = 0
    for s in sorted(by_stratum):
        keys = sorted(by_stratum[s], key=lambda k: hashlib.sha256(f"{SPLIT_SEED}:{k}".encode()).hexdigest())
        for i, key in enumerate(keys):
            fold = (start + i) % FOLDS
            for cid in groups[key]:
                fold_of[cid] = fold
        start = (start + len(keys)) % FOLDS
    return fold_of


def training_split(tag: str, train: list[str], test: list[str], source: Path, note: str, extra: dict) -> Path:
    payload = {"schema_version": 1, "pipeline": "wss_v5.views.wss_min_view", "data_root": str(VIEW.relative_to(MAIN)),
               "required_frame_version": "v5_atlas_frame_v1",
               "id_format": "canonical cohort/subset/case; ILO subset is patient-0|1 and case is before|after",
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "split_version": tag,
               "source_split": str(source), "source_split_sha256": sha256(source), "duplicate_geometry_groups": DUP_GROUPS,
               "explicit_related_groups": [["AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"]],
               "train_cases": sorted(train), "val_cases": [], "test_cases": sorted(test), "unused_cases": [],
               "counts": {"train": len(train), "val": 0, "test": len(test)}, "expected_counts": {"train": len(train), "val": 0, "test": len(test)},
               "note": note, **extra}
    path = (CV / f"{tag.split('/')[-1]}.json") if tag.startswith("cv5_v52/") else (VIEW / f"{tag}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        old = json.loads(path.read_text()); old.pop("created_at"); new = dict(payload); new.pop("created_at")
        if old != new:
            raise FileExistsError(f"refusing to change preregistered split: {path}")
        return path
    path.write_text(text)
    return path


def registry_split(name: str, train: list[str], test: list[str], folds: dict[str, int] | None, note: str) -> Path:
    """schema-4 registry split (wss_pinn/configs/splits) for provenance / V5 tooling."""
    base = json.loads((REGISTRY_DIR / "split_WSS_PINN_AG_AAA_ILO_v5_2_new91_train227_test34_s1234.json").read_text())
    payload = {k: base[k] for k in ("schema_version", "route", "excluded_cases", "duplicate_geometry_groups", "duplicate_geometry_note")}
    payload.update(created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seed=SPLIT_SEED, train_cases=sorted(train), test_cases=sorted(test),
                   counts={"train": len(train), "test": len(test), "cv_folds": FOLDS if folds else 0}, derivation=note)
    if folds:
        payload["cv"] = {"algorithm": "patient-grouped stratified deterministic SHA256 round-robin over all cases (2026-09-23)",
                         "folds": [{"fold": k, "train_cases": sorted(c for c in train if folds[c] != k), "validation_cases": sorted(c for c in train if folds[c] == k)} for k in range(FOLDS)]}
    payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = REGISTRY_DIR / name
    if not path.exists():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return path


def feature_stats(tag: str, split_path: Path, stats_path: Path, cfg_src: dict) -> Path:
    path = EXP / "feature_stats" / f"{tag}_train_feature_stats.json"
    if path.is_file():
        return path
    input_features = list(cfg_src["data"]["input_features"])
    stats = D.load_wss_stats(stats_path)
    extra = tuple(f for f in input_features if f in C.SIDECAR_FEATURE_KEYS)
    cases = D.load_partition(str(split_path), "train", stats, strict=True, target="wss", data_root=VIEW,
                             required_frame_version="v5_atlas_frame_v1", extra_point_features=extra, point_features_root=[str(GEOM), str(FLOW)])
    expected = json.loads(split_path.read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"{tag}: expected {expected} train cases, loaded {len(cases)}")
    out = D.compute_feature_stats(cases, tuple(input_features), cfg_src["data"]["curvature_transform"])
    out["_provenance"] = {"train_split": str(split_path), "n_cases": len(cases), "wss_stats": str(stats_path), "sidecar_roots": [str(GEOM), str(FLOW)],
                          "curvature_transform": cfg_src["data"]["curvature_transform"], "note": f"v5.2 {tag}: train-partition z-scores for every non-xyz input feature"}
    save(path, out)
    return path


def make_config(arm: str, recipe: str, seed: int, split_path: Path, stats_path: Path, feat_path: Path, title: str) -> dict:
    src_seed = seed if (SOURCE / f"{RECIPES[recipe]}_s{seed}.json").is_file() else 1234
    cfg = json.loads((SOURCE / f"{RECIPES[recipe]}_s{src_seed}.json").read_text())
    cfg["name"] = f"{NAME}/{arm}"
    cfg["notes"] = f"v5.2 {arm}: {title}; recipe copied from wss_v51_wave1 {RECIPES[recipe]}_s{src_seed}; data = views_v5_2_full_20260923 (170 library + 91 recovered; 11 library cases rebuilt with the end-radius fix); partition-fitted log_z WSS stats and feature z-scores; C1_s{seed} paired init."
    cfg["train"]["seed"] = seed
    cfg["train"]["init_reference_config"] = str(REFS[seed])
    cfg["data"]["data_root"] = str(VIEW)
    cfg["data"]["split_path"] = str(split_path)
    cfg["data"]["wss_stats_path"] = str(stats_path)
    cfg["data"]["feature_stats_path"] = str(feat_path)
    cfg["data"]["point_features_root"] = [str(GEOM), str(FLOW)]
    cfg["data"]["density_aug_root"] = str(DENS)
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
            if anchor[section].get(key) != cfg[section].get(key):
                diff[f"{section}.{key}"] = {"anchor": anchor[section].get(key), "arm": cfg[section].get(key)}
    return diff


def main() -> None:
    lib = json.loads(LIB_SPLIT.read_text()); library = sorted(lib["train_cases"] + lib["test_cases"])
    manifest = json.loads(NEW_MANIFEST.read_text()); new = sorted(manifest["new_units"])
    assert not manifest.get("hold"), manifest.get("hold")
    all_cases = sorted(library + new)
    for cid in all_cases:
        if not (VIEW / cid / "bundle.npz").is_file():
            raise FileNotFoundError(VIEW / cid / "bundle.npz")
    # ---- IND: train 170 / test 91 --------------------------------------------------------------------------
    reg_ind = registry_split("split_WSS_PINN_AG_AAA_ILO_v5_2_ind_train170_test91_s20260923.json", library, new, None,
                             "v5.2 IND protocol (2026-09-23): the 170 library cases train, the 91 recovered units are the independent test set")
    ind_split = training_split("split_v52_ind_train170_test91", library, new, reg_ind,
                               "v5.2 IND: train = 170 library cases (v5.1 train136 + test34), test = 91 recovered units (independent set)", {})
    ind_stats = VIEW / "wss_global_stats_ind_train170.json"
    if not ind_stats.is_file():
        write_wss_stats(library, VIEW, ind_split, filename="wss_global_stats_ind_train170.json", scope="v5.2 IND train partition only (170 library cases, valid anatomy wall nodes)")
    # ---- CV5: 261 cases, patient-grouped ------------------------------------------------------------------
    folds = cv5_assign(all_cases)
    reg_cv = registry_split("split_WSS_PINN_AG_AAA_ILO_v5_2_cv5_261_s20260923.json", all_cases, [], folds,
                            "v5.2 CV5 protocol (2026-09-23): patient-grouped stratified 5-fold CV over all 261 cases (170 library + 91 recovered); no fixed test set")
    fold_paths, fold_stats = {}, {}
    fold_table = {}
    for k in range(FOLDS):
        held = sorted(c for c in all_cases if folds[c] == k); train = sorted(c for c in all_cases if folds[c] != k)
        fold_paths[k] = training_split(f"cv5_v52/fold{k}", train, held, reg_cv,
                                       "v5.2 CV5: patient-grouped 5-fold CV over 261 cases; 'test' = the held-out fold; no separate validation (selection = train_loss)",
                                       {"fold": k, "n_folds": FOLDS, "split_seed": SPLIT_SEED})
        fold_stats[k] = CV / f"wss_global_stats_fold{k}_train.json"
        if not fold_stats[k].is_file():
            write_wss_stats(train, VIEW, fold_paths[k], filename=f"cv5_v52/wss_global_stats_fold{k}_train.json", scope=f"v5.2 CV5 fold{k} train partition only ({len(train)} cases)")
        fold_table[k] = {"n_train": len(train), "n_test": len(held), "test_by_stratum": {s: sum(1 for c in held if stratum(c) == s) for s in sorted({stratum(c) for c in all_cases})},
                         "test_groups": len({patient_group(c) for c in held})}
    save(CV / "cv5_summary.json", {"protocol": "cv5_v52", "cases": len(all_cases), "folds": FOLDS, "split_seed": SPLIT_SEED, "grouping": "patient (ILO before/after and -0/-1 of one name together; namesakes across cohorts grouped)",
                                   "strata": sorted({stratum(c) for c in all_cases}), "fold_table": fold_table, "fold_of": folds})
    # ---- configs + matrix ------------------------------------------------------------------------------------
    anchor = json.loads(ANCHOR.read_text()); arms = []
    for recipe in RECIPES:
        cfg_src = json.loads((SOURCE / f"{RECIPES[recipe]}_s1234.json").read_text())
        feat_ind = feature_stats(f"ind_{recipe}", ind_split, ind_stats, cfg_src)
        for seed in SEEDS_IND:
            arm = f"{recipe}_v52ind_s{seed}"; title = f"{recipe} recipe, IND protocol (train 170 library / test 91 recovered), seed {seed}"
            cfg = make_config(arm, recipe, seed, ind_split, ind_stats, feat_ind, title); save(CONFIGS / f"{arm}.json", cfg)
            arms.append(dict(id=arm, title=title, config=f"{arm}.json", phase=0, modules=["F6", "density_aug"] + (["cap_prior"] if recipe == "X5Dcap" else []), seed=seed, fold=None,
                             protocol="IND", recipe=recipe, parent=f"{RECIPES[recipe]}_s{seed if seed in (1234,7,2025) else 1234} (wss_v51_wave1)", depends_on=[], evaluate=["best", "last"],
                             single_change=False, config_diff_vs_anchor=declared_diff(anchor, cfg)))
        for k in range(FOLDS):
            feat_k = feature_stats(f"cv5_fold{k}_{recipe}", fold_paths[k], fold_stats[k], cfg_src)
            for seed in SEEDS_CV:
                arm = f"{recipe}_v52cv_f{k}_s{seed}"; title = f"{recipe} recipe, CV5 protocol fold {k} (261 cases, patient-grouped), seed {seed}"
                cfg = make_config(arm, recipe, seed, fold_paths[k], fold_stats[k], feat_k, title); save(CONFIGS / f"{arm}.json", cfg)
                arms.append(dict(id=arm, title=title, config=f"{arm}.json", phase=1, modules=["F6", "density_aug"] + (["cap_prior"] if recipe == "X5Dcap" else []), seed=seed, fold=k,
                                 protocol="CV5", recipe=recipe, parent=f"{RECIPES[recipe]}_s{seed} (wss_v51_wave1)", depends_on=[], evaluate=["best", "last"],
                                 single_change=False, config_diff_vs_anchor=declared_diff(anchor, cfg)))
    save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(ANCHOR), anchor_run=str(ANCHOR_RUN), control_id=None, external_reference_runs={},
        data={"views_root": str(VIEW.parent), "library": 170, "recovered": 91, "rebuilt_library_cases": 11, "geometry_program": "pc-v3 + end-radius-hold-2026-09-22"},
        protocols={"IND": {"split": str(ind_split), "wss_stats": str(ind_stats), "seeds": list(SEEDS_IND)},
                   "CV5": {"folds": {k: str(fold_paths[k]) for k in range(FOLDS)}, "seeds": list(SEEDS_CV), "summary": str(CV / "cv5_summary.json")}},
        recipes={"X5D": "X5D_v51 recipe (Murray prior on atlas distal radius)", "X5Dcap": "X5D with protocol-cap Murray keys (log_q/log_tau0 on virtual-cap radii)"},
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule=("IND: physical Pa R2_cb on the 91 recovered units, five-seed Pa-mean ensemble per recipe (compare with frozen X5D_v51 0.749); "
                      "CV5: held-out-fold R2_cb pooled over five folds per seed (every case once), three seeds -> mean +- sd; X5Dcap vs X5D read as paired deltas per (fold, seed); "
                      "no single reading is ranked; test34 is not a protocol set any more")))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()
