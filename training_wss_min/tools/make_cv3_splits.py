"""Patient-grouped 3-fold splits of V5 train138 for the X5 / X5+X11 stability check (2026-09-13).

Folds are stratified by cohort family (AG / AAA / ILO), keep the declared duplicate-geometry group together,
and never touch test34.  For fold k the legacy-schema split has train = the other two folds, test = fold k.
Per-fold log_z WSS statistics are written with the view's own recipe (train partition of that fold only).

    python -m training_wss_min.tools.make_cv3_splits
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from wss_v5.views.wss_min_view import write_wss_stats

VIEW = C.PROJECT_ROOT / "data_wss_v5/views/wss_min_view_v1"
SOURCE = VIEW / "split_V5_train138_test34.json"
OUT = VIEW / "cv3_20260913"
FOLDS = 3
SEED = 20260913


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_folds(train_cases: list[str], groups: list[list[str]], seed: int = SEED) -> dict[str, int]:
    """Balanced per-cohort round-robin over a seeded shuffle; grouped cases are placed as one unit."""
    unit_of = {}
    for members in groups:
        for m in members:
            unit_of[m] = tuple(sorted(members))
    units = []
    seen = set()
    for cid in train_cases:
        unit = unit_of.get(cid, (cid,))
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for family in ("AG", "AAA", "ILO"):
        fam = [u for u in units if u[0].split("/")[0] == family]
        order = rng.permutation(len(fam))
        start = int(rng.integers(0, FOLDS))
        for rank, index in enumerate(order):
            for cid in fam[index]:
                fold_of[cid] = (start + rank) % FOLDS
    missing = [c for c in train_cases if c not in fold_of]
    if missing:
        raise RuntimeError(f"unassigned cases: {missing}")
    return fold_of


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view-root", default=str(VIEW), help="wss_min_view root (2026-09-15: v5.1 refresh uses a new root)")
    ap.add_argument("--source-name", default=SOURCE.name)
    ap.add_argument("--out-name", default=OUT.name)
    ap.add_argument("--expect-train", type=int, default=138)
    args = ap.parse_args(argv)
    view = Path(args.view_root); source = view / args.source_name; out = view / args.out_name; expect_train = args.expect_train
    src = json.loads(source.read_text())
    train = list(src["train_cases"])
    if len(train) != expect_train or len(src["test_cases"]) != 34:
        raise RuntimeError("unexpected source split sizes")
    groups = [g for g in src.get("duplicate_geometry_groups", []) + src.get("explicit_related_groups", [])
              if all(m in train for m in g)]
    fold_of = assign_folds(train, groups)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "source_split": str(source), "source_split_sha256": sha256(source), "folds": FOLDS,
               "seed": SEED, "grouped_units": groups, "fold_of": fold_of, "files": {}}
    for k in range(FOLDS):
        held = [c for c in train if fold_of[c] == k]
        rest = [c for c in train if fold_of[c] != k]
        payload = {key: src[key] for key in ("schema_version", "pipeline", "data_root", "required_frame_version", "id_format")}
        payload.update(
            created_at=src["created_at"], split_version=f"{out.name}/fold{k} of {src['split_version']}",
            source_split=str(source), source_split_sha256=summary["source_split_sha256"],
            duplicate_geometry_groups=src.get("duplicate_geometry_groups", []),
            explicit_related_groups=src.get("explicit_related_groups", []),
            train_cases=rest, val_cases=[], test_cases=held, unused_cases=list(src["test_cases"]),
            counts={"train": len(rest), "val": 0, "test": len(held)},
            expected_counts={"train": len(rest), "val": 0, "test": len(held)},
            note=(f"patient-grouped 3-fold CV inside V5 train{len(train)}; 'test' here is the held-out fold, the frozen test34 is "
                  "listed as unused and never used for selection or evaluation in this protocol"),
        )
        split_path = out / f"fold{k}.json"
        split_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stats_path = write_wss_stats(rest, view, split_path, filename=f"{out.name}/wss_global_stats_fold{k}_train.json",
                                     scope=f"train partition of cv3 fold{k} only ({len(rest)} cases, valid anatomy wall nodes)")
        stats = json.loads(stats_path.read_text())
        by_family = {f: sum(1 for c in held if c.startswith(f + "/")) for f in ("AG", "AAA", "ILO")}
        summary["files"][f"fold{k}"] = {"split": str(split_path), "wss_stats": str(stats_path), "n_train": len(rest),
                                        "n_test": len(held), "held_by_family": by_family,
                                        "log_mean": stats["log"]["mean"], "log_std": stats["log"]["std"]}
        print(f"fold{k}: train {len(rest)} test {len(held)} {by_family} log mean/std {stats['log']['mean']:.4f}/{stats['log']['std']:.4f}")
    (out / "cv3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(out / "cv3_summary.json")


if __name__ == "__main__":
    main()
