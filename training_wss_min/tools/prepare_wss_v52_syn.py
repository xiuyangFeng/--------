"""§36 direction 3 (2026-09-26): synthetic-CFD augmentation arms. Run AFTER the 60 children are ingested into the v5.2
views root (wss_min_view_v1 / geom_v2 / flowref_v1 / density_v1 / phys1d_v1 for every child).

Leakage rules (frozen):
  * a child never enters a test/held-out partition;
  * CV5: child joins fold k's TRAIN set only if its parent is in fold k's train set (parent held out -> child excluded);
  * IND: child joins the 170-library train set only if its parent is one of the 170 (children of the 91 recovered
    units are excluded because those units are the IND test set).
Arms (recipe = X5Dcap unless --recipe-config-dir points at a §36 winner):
  X5Dcap_syn_v52cv_f{k}_s{1234,7,2025}  (15)   X5Dcap_syn_v52ind_s{1234,7,2025} (3)
Partition WSS statistics and feature z-scores are refitted on the augmented train sets.

    python -m training_wss_min.tools.prepare_wss_v52_syn [--children docs/.../synth_children_all60.txt]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_v52 as P
from wss_v5.views.wss_min_view import write_wss_stats

NAME = "wss_v52_syn_20260926"
CONFIGS = P.MAIN / "training_wss_min/configs" / NAME
EXP = P.MAIN / "training_wss_min/experiments" / NAME
SOURCE = P.MAIN / "training_wss_min/configs/wss_v52_20260923"
CHILDREN = P.MAIN / "docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926/synth_children_all60.txt"
SYN_SPLITS = P.VIEW / "syn_v52"
SEEDS = (1234, 7, 2025)


def parent_of(cid: str) -> str:
    m = re.match(r"(.*)~m\d+(/.*)?$", cid); return m.group(1) + (m.group(2) or "")


def aug_split(tag: str, base_path: Path, kids: list[str], note: str) -> tuple[Path, Path]:
    base = json.loads(base_path.read_text())
    train = sorted(set(base["train_cases"]) | set(kids))
    payload = dict(base); payload.update(created_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), split_version=f"syn_v52/{tag} of {base['split_version']}",
                   source_split=str(base_path), source_split_sha256=P.sha256(base_path), train_cases=train,
                   counts={"train": len(train), "val": 0, "test": len(base["test_cases"])}, expected_counts={"train": len(train), "val": 0, "test": len(base["test_cases"])},
                   synthetic={"children": kids, "n": len(kids), "rule": note})
    path = SYN_SPLITS / f"{tag}.json"; path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        old = json.loads(path.read_text()); old.pop("created_at"); new = dict(payload); new.pop("created_at")
        if old != new:
            raise FileExistsError(f"refusing to change preregistered split: {path}")
    else:
        path.write_text(text)
    stats = SYN_SPLITS / f"wss_global_stats_{tag}_train.json"
    if not stats.is_file():
        write_wss_stats(train, P.VIEW, path, filename=f"syn_v52/wss_global_stats_{tag}_train.json", scope=f"§36 {tag} train partition ({len(train)} cases incl. {len(kids)} synthetic)")
    return path, stats


def feature_stats(tag: str, split_path: Path, stats_path: Path, cfg: dict) -> Path:
    from training_wss_min import dataset as D
    path = EXP / "feature_stats" / f"{tag}_train_feature_stats.json"
    if path.is_file():
        return path
    input_features = list(cfg["data"]["input_features"]); stats = D.load_wss_stats(stats_path)
    extra = tuple(f for f in input_features if f in C.SIDECAR_FEATURE_KEYS)
    cases = D.load_partition(str(split_path), "train", stats, strict=True, target="wss", data_root=P.VIEW, required_frame_version="v5_atlas_frame_v1",
                             extra_point_features=extra, point_features_root=list(cfg["data"]["point_features_root"]))
    expected = json.loads(split_path.read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"{tag}: expected {expected} train cases, loaded {len(cases)}")
    out = D.compute_feature_stats(cases, tuple(input_features), cfg["data"]["curvature_transform"])
    out["_provenance"] = {"train_split": str(split_path), "n_cases": len(cases), "wss_stats": str(stats_path), "sidecar_roots": list(cfg["data"]["point_features_root"]), "note": f"§36 synthetic augmentation {tag}"}
    P.save(path, out)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--children", type=Path, default=CHILDREN); ap.add_argument("--recipe-config-dir", type=Path, default=SOURCE)
    ap.add_argument("--recipe", default="X5Dcap"); a = ap.parse_args()
    kids = [l.strip() for l in a.children.read_text().splitlines() if l.strip() and not l.startswith("#")]
    for k in kids:
        if not (P.VIEW / k / "bundle.npz").is_file():
            raise FileNotFoundError(f"child not ingested: {P.VIEW / k / 'bundle.npz'}")
    lib = json.loads(P.LIB_SPLIT.read_text()); library = set(lib["train_cases"] + lib["test_cases"])
    anchor = json.loads(P.ANCHOR.read_text()); arms = []
    # ---- IND + syn (children of library parents only)
    kids_ind = [k for k in kids if parent_of(k) in library]
    ind_split, ind_stats = aug_split("ind_syn", P.VIEW / "split_v52_ind_train170_test91.json", kids_ind, "children of the 170 library parents only; the 91 recovered units stay a clean test set")
    for seed in SEEDS:
        src = json.loads((a.recipe_config_dir / f"{a.recipe}_v52ind_s{seed}.json").read_text()); cfg = json.loads(json.dumps(src))
        aid = f"{a.recipe}_syn_v52ind_s{seed}"; feat = feature_stats("ind_syn", ind_split, ind_stats, cfg)
        cfg.update(name=f"{NAME}/{aid}", notes=f"§36 {aid}: {a.recipe} recipe + {len(kids_ind)} synthetic children in the IND train set (test = 91 recovered, real only)")
        cfg["data"].update(split_path=str(ind_split), wss_stats_path=str(ind_stats), feature_stats_path=str(feat)); cfg["train"]["seed"] = seed
        C.ExpConfig.from_dict(cfg); P.save(CONFIGS / f"{aid}.json", cfg)
        arms.append(dict(id=aid, title=f"{a.recipe} + synthetic ({len(kids_ind)}) IND, seed {seed}", config=f"{aid}.json", phase=0, modules=["F6", "density_aug", "cap_prior", "synthetic"], seed=seed, fold=None, protocol="IND",
                         parent=f"{a.recipe}_v52ind_s{seed}", depends_on=[], evaluate=["best", "last"], single_change=True, config_diff_vs_anchor=P.declared_diff(anchor, cfg), config_diff_vs_parent=P.declared_diff(src, cfg)))
    # ---- CV5 + syn (child follows its parent's fold)
    cv = json.loads((P.CV / "cv5_summary.json").read_text()); fold_of = cv["fold_of"]
    for k in range(P.FOLDS):
        base_path = P.CV / f"fold{k}.json"; held = set(json.loads(base_path.read_text())["test_cases"])
        kids_k = [c for c in kids if parent_of(c) not in held]
        sp, st = aug_split(f"cv5_fold{k}_syn", base_path, kids_k, "children whose parent is NOT in the held-out fold (parent's fold inherited)")
        for seed in SEEDS:
            src = json.loads((a.recipe_config_dir / f"{a.recipe}_v52cv_f{k}_s{seed}.json").read_text()); cfg = json.loads(json.dumps(src))
            aid = f"{a.recipe}_syn_v52cv_f{k}_s{seed}"; feat = feature_stats(f"cv5_fold{k}_syn", sp, st, cfg)
            cfg.update(name=f"{NAME}/{aid}", notes=f"§36 {aid}: {a.recipe} recipe + {len(kids_k)} synthetic children in fold {k}'s train set (held-out fold real only)")
            cfg["data"].update(split_path=str(sp), wss_stats_path=str(st), feature_stats_path=str(feat)); cfg["train"]["seed"] = seed
            C.ExpConfig.from_dict(cfg); P.save(CONFIGS / f"{aid}.json", cfg)
            arms.append(dict(id=aid, title=f"{a.recipe} + synthetic ({len(kids_k)}) CV5 fold {k}, seed {seed}", config=f"{aid}.json", phase=0, modules=["F6", "density_aug", "cap_prior", "synthetic"], seed=seed, fold=k, protocol="CV5",
                             parent=f"{a.recipe}_v52cv_f{k}_s{seed}", depends_on=[], evaluate=["best", "last"], single_change=True, config_diff_vs_anchor=P.declared_diff(anchor, cfg), config_diff_vs_parent=P.declared_diff(src, cfg)))
    P.save(CONFIGS / "matrix.json", dict(schema_version=1, experiment=NAME, anchor=str(P.ANCHOR), anchor_run=str(P.ANCHOR_RUN), control_id=None, section="§36 direction 3",
        external_reference_runs={f"{a.recipe}_v52ind_s{s}": str(P.RUNS / f"wss_v52_20260923/{a.recipe}_v52ind_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS}
                                | {f"{a.recipe}_v52cv_f{k}_s{s}": str(P.RUNS / f"wss_v52_20260923/{a.recipe}_v52cv_f{k}_s{s}/eval/ckpt_best/metrics.json") for k in range(P.FOLDS) for s in SEEDS},
        synthetic={"children": kids, "children_file": str(a.children), "ind_children": kids_ind}, expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule="paired with the same (fold, seed) / seed arm of wss_v52_20260923: CV5 15 paired deltas (gate G3.1 mean > +0.010, >= 12/15 positive), IND 3 paired deltas (G3.2 > +0.010); held-out / test partitions contain real cases only"))
    print(CONFIGS / "matrix.json", len(arms), "arms; IND children", len(kids_ind))


if __name__ == "__main__":
    main()
