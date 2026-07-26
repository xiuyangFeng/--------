#!/usr/bin/env python3
"""Prepare the authorized Q2V ILO/data and Point++ architecture matrices.

This is the only writer for the 2026-07-18 protocol assets.  It promotes the
already approved ILO-before41 cohort into a training-specific manifest without
rewriting the immutable audit record, creates grouped deterministic splits,
train-only target/feature statistics, and explicit JSON configs cloned from
Q2V-10477.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training_wss_min import dataset as D
from training_wss_min.tools.prepare_v4_protocol import FRAME, make_stats, sha256, write_json


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data_wss_min"
OLD_REPORT = DATA_ROOT / "pipeline_reports/v4_cutover_20260715_1921"
ILO_REPORT = DATA_ROOT / "pipeline_reports/ilo_before_final_approved_20260717"
OLD_WHITELIST = OLD_REPORT / "v4_training_whitelist.json"
ILO_WHITELIST = ILO_REPORT / "ilo_before_whitelist.json"
ILO_AUDIT = ILO_REPORT / "ilo_before_wss_geometry_audit.json"
OLD_SPLIT = REPO / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
OLD_STATS = DATA_ROOT / "fold_stats/v4/AG_AAA_v4_stratified_train106_global_stats.json"
Q2V_CONFIG = (
    REPO / "training_wss_min/configs/pointnetpp_v4/"
    "ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep.json"
)
Q2V_RUN = (
    REPO / "training_wss_min/runs/pointnetpp_v4/outputs/"
    "ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep"
)
Q2V_FEATURE_STATS = Q2V_RUN / "feature_stats.json"
PROMOTION = ILO_REPORT / "ilo_before_training_promotion_q2v_20260718.json"

SPLIT_DIR = REPO / "training/splits"
FIXED_SPLIT = SPLIT_DIR / "split_AG_AAA_ILO_q2v_fixed_test27_train147.json"
EXTENDED_SPLIT = SPLIT_DIR / "split_AG_AAA_ILO_q2v_extended_test36_seed1234.json"
POOL_SPLIT = SPLIT_DIR / "split_AG_AAA_ILO_q2v_pool2025_train138_test36.json"
POOL_CONTROL_SPLIT = SPLIT_DIR / "split_AG_AAA_ILO_q2v_pool2025_control106_test36.json"
ARCH_SPLIT = SPLIT_DIR / "split_AG_AAA_q2v_arch_dev_seed2025.json"

STATS_DIR = DATA_ROOT / "fold_stats/q2v_ilo_20260718"
FIXED_REFIT_STATS = STATS_DIR / "q2v_fixed_train147_global_stats.json"
POOL_CONTROL_STATS = STATS_DIR / "q2v_pool2025_control106_global_stats.json"
POOL_REFIT_STATS = STATS_DIR / "q2v_pool2025_train138_global_stats.json"
ARCH_STATS = STATS_DIR / "q2v_arch_dev_train85_global_stats.json"
POOL_CONTROL_FEATURE_STATS = STATS_DIR / "q2v_pool2025_control106_feature_stats.json"
ARCH_FEATURE_STATS = STATS_DIR / "q2v_arch_dev_train85_feature_stats.json"

DATA_CONFIG_DIR = REPO / "training_wss_min/configs/pointnetpp_q2v_ilo_20260718"
ARCH_CONFIG_DIR = REPO / "training_wss_min/configs/pointnetpp_q2v_arch_20260718"
MANIFEST = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_prepared.json"

STRATA = ("AG", "AAA/ruputer", "AAA/unruputer", "ILO/0", "ILO/1")
TEST_QUOTAS = {
    "AG": 15, "AAA/ruputer": 6, "AAA/unruputer": 6, "ILO/0": 6, "ILO/1": 3,
}
ARCH_VAL_QUOTAS = {"AG": 12, "AAA/ruputer": 4, "AAA/unruputer": 5}
KNOWN_RELATED = frozenset({"AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"})
INPUT_FEATURES = ("x", "y", "z", "abscissa_norm", "local_radius", "curvature")


def stable_rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"q2v-ilo:{seed}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def stratum(unit: str) -> str:
    if unit.startswith("AG/"):
        return "AG"
    if unit.startswith("AAA/ruputer/"):
        return "AAA/ruputer"
    if unit.startswith("AAA/unruputer/"):
        return "AAA/unruputer"
    if unit.startswith("ILO/") and unit.endswith("/before"):
        patient = unit.split("/", 2)[1]
        suffix = patient.rsplit("-", 1)[-1]
        if suffix in {"0", "1"}:
            return f"ILO/{suffix}"
    raise ValueError(f"unrecognized matrix unit: {unit}")


def geometry_fingerprint(bundle: Path) -> str:
    with np.load(bundle, allow_pickle=False) as z:
        frame = str(np.asarray(z["transform_frame_version"]).item())
        if frame != FRAME:
            raise RuntimeError(f"wrong frame: {bundle}={frame}")
        xyz = np.asarray(z["wall_coords_raw"], dtype=np.float64)
    xyz = np.round(xyz, decimals=5)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    h = hashlib.sha256()
    h.update(np.asarray(xyz.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(xyz[order], dtype="<f8").tobytes())
    return h.hexdigest()


def choose_exact(groups: list[list[str]], quota: int, rng: np.random.Generator) -> list[str]:
    order = rng.permutation(len(groups)).tolist()
    shuffled = [groups[i] for i in order]
    dp: dict[int, list[int]] = {0: []}
    for i, group in enumerate(shuffled):
        for total, selected in sorted(list(dp.items()), reverse=True):
            new_total = total + len(group)
            if new_total <= quota and new_total not in dp:
                dp[new_total] = selected + [i]
    if quota not in dp:
        raise RuntimeError(f"cannot satisfy grouped quota={quota}")
    return sorted(unit for i in dp[quota] for unit in shuffled[i])


def grouped_units(units: list[str], duplicate_groups: list[list[str]]) -> dict[str, list[list[str]]]:
    by = {name: [] for name in STRATA}
    assigned: set[str] = set()
    for group in duplicate_groups:
        if not set(group) <= set(units):
            continue
        names = {stratum(unit) for unit in group}
        if len(names) != 1:
            raise RuntimeError(f"related group crosses strata: {group}")
        by[names.pop()].append(sorted(group)); assigned.update(group)
    for unit in units:
        if unit not in assigned:
            by[stratum(unit)].append([unit])
    for groups in by.values():
        groups.sort(key=lambda row: row[0])
    return by


def split_all(units: list[str], duplicate_groups: list[list[str]], seed: int) -> tuple[dict, dict]:
    groups = grouped_units(units, duplicate_groups)
    train: dict[str, list[str]] = {}; test: dict[str, list[str]] = {}
    for name in STRATA:
        selected = choose_exact(groups[name], TEST_QUOTAS[name], stable_rng(seed, name))
        selected_set = set(selected)
        test[name] = selected
        train[name] = sorted(unit for group in groups[name] for unit in group
                             if unit not in selected_set)
    return train, test


def split_ilo(ilo_units: list[str], duplicate_groups: list[list[str]], seed: int) -> tuple[list[str], list[str]]:
    groups = grouped_units(ilo_units, duplicate_groups)
    train: list[str] = []; test: list[str] = []
    for name in ("ILO/0", "ILO/1"):
        selected = choose_exact(groups[name], TEST_QUOTAS[name], stable_rng(seed, name))
        selected_set = set(selected)
        test.extend(selected)
        train.extend(unit for group in groups[name] for unit in group if unit not in selected_set)
    return sorted(train), sorted(test)


def write_split(path: Path, *, version: str, protocol: str, seed: int | None,
                train: list[str], val: list[str], test: list[str],
                counts: dict, duplicate_groups: list[list[str]], unused: list[str] | None = None,
                source_split: Path | None = None) -> None:
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise RuntimeError(f"partition leakage while writing {version}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "pipeline_wss_min",
        "data_root": str(DATA_ROOT.resolve()),
        "required_frame_version": FRAME,
        "id_format": "canonical cohort/subset/case; ILO subset is patient-0|1 and case is before",
        "split_version": version,
        "protocol": protocol,
        "seed": seed,
        "algorithm": "canonical sort; 1e-5-mm exact geometry groups plus explicit related groups; independent per-stratum SHA256-derived numpy RNG; exact-quota DP",
        "stratum_order": list(STRATA),
        "test_quotas": TEST_QUOTAS,
        "source_ilo_training_promotion": str(PROMOTION.resolve()),
        "source_ilo_training_promotion_sha256": sha256(PROMOTION),
        "source_parent_split": str(source_split.resolve()) if source_split else None,
        "source_parent_split_sha256": sha256(source_split) if source_split else None,
        "duplicate_geometry_groups": duplicate_groups,
        "explicit_related_groups": [sorted(KNOWN_RELATED)],
        "train_cases": train, "val_cases": val, "test_cases": test,
        "unused_cases": sorted(unused or []),
        "counts": counts,
        "expected_counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    write_json(path, payload)


def add_stats_breakdown(stats_path: Path) -> dict:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    point_counts: dict[str, int] = {}
    case_counts: dict[str, int] = {}
    for row in stats["bundle_manifest"]:
        name = stratum(row["unit_id"])
        point_counts[name] = point_counts.get(name, 0) + int(row["n_points"])
        case_counts[name] = case_counts.get(name, 0) + 1
    stats["point_contribution_by_stratum"] = {
        name: {"n_cases": case_counts[name], "n_points": count,
               "fraction": count / stats["n_points"]}
        for name, count in point_counts.items()
    }
    write_json(stats_path, stats)
    return stats


def make_feature_stats(split_path: Path, stats_path: Path, output: Path) -> dict:
    print(f"[feature-stats] load train from {split_path.name}", flush=True)
    stats = D.load_wss_stats(stats_path)
    cases = D.load_partition(
        str(split_path), "train", stats, strict=True, data_root=DATA_ROOT,
        required_frame_version=FRAME,
    )
    for i, case in enumerate(cases, 1):
        print(f"[feature-stats {i:03d}/{len(cases):03d}] {case['unit_id']}", flush=True)
    result = D.compute_feature_stats(cases, INPUT_FEATURES, "signed_log1p")
    write_json(output, result)
    return result


def clone_config(base: dict, *, exp_id: str, note: str, split_path: Path,
                 stats_path: Path, feature_stats_path: Path | None,
                 width: int = 32, nsample: int = 16, architecture: bool = False) -> dict:
    cfg = copy.deepcopy(base)
    family = "pointnetpp_q2v_arch_dev" if architecture else "pointnetpp_q2v_ilo"
    cfg["name"] = f"{family}/outputs/{exp_id}"
    cfg["notes"] = note
    cfg["data"]["split_path"] = str(split_path.resolve())
    cfg["data"]["wss_stats_path"] = str(stats_path.resolve())
    cfg["data"]["feature_stats_path"] = (
        str(feature_stats_path.resolve()) if feature_stats_path else None
    )
    cfg["model"]["width"] = width
    cfg["model"]["sa_nsample"] = [nsample, nsample, nsample]
    if architecture:
        cfg["train"]["eval_every"] = 400
    return cfg


def run(repo: Path) -> dict:
    if repo.resolve() != REPO.resolve():
        raise RuntimeError(f"this protocol is pinned to {REPO}, got {repo}")
    for required in (OLD_WHITELIST, ILO_WHITELIST, ILO_AUDIT, OLD_SPLIT, OLD_STATS,
                     Q2V_CONFIG, Q2V_FEATURE_STATS):
        if not required.is_file():
            raise FileNotFoundError(required)

    old_wl = json.loads(OLD_WHITELIST.read_text(encoding="utf-8"))
    ilo_wl = json.loads(ILO_WHITELIST.read_text(encoding="utf-8"))
    audit = json.loads(ILO_AUDIT.read_text(encoding="utf-8"))
    old_split = json.loads(OLD_SPLIT.read_text(encoding="utf-8"))
    ag_units = sorted(old_wl["AG"]); aaa_units = sorted(old_wl["AAA"])
    ilo_units = sorted(ilo_wl["included_units"])
    all_units = sorted(ag_units + aaa_units + ilo_units)
    if (len(ag_units), len(aaa_units), len(ilo_units), len(all_units)) != (76, 57, 41, 174):
        raise RuntimeError("unexpected eligible counts; expected AG76+AAA57+ILO41=174")
    if audit["counts"]["qa_pass"] != 41 or audit["failed_units"]:
        raise RuntimeError("ILO final hard-gate audit is not 41/41 PASS")
    if any(not unit.endswith("/before") for unit in ilo_units):
        raise RuntimeError("ILO promotion attempted to include non-before data")

    promotion = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Q2V-10477 single-seed ILO-before data attribution matrix only",
        "training_promotion_authorized": True,
        "authorization_source": "user explicitly requested implementation and Slurm submission in the active Codex task",
        "immutable_audit_training_promotion_authorized": bool(ilo_wl.get("training_promotion_authorized", False)),
        "source_whitelist": str(ILO_WHITELIST.resolve()),
        "source_whitelist_sha256": sha256(ILO_WHITELIST),
        "source_hard_gate_audit": str(ILO_AUDIT.resolve()),
        "source_hard_gate_audit_sha256": sha256(ILO_AUDIT),
        "allowed_phase": "before",
        "included_units": ilo_units,
        "counts": {"ILO_before": 41, "ILO_0": 28, "ILO_1": 13, "ILO_after": 0},
        "forbidden": ["ILO/*/after", "all 20 units excluded by the final ILO audit"],
    }
    write_json(PROMOTION, promotion)

    fingerprints: dict[str, list[str]] = {}
    for i, unit in enumerate(all_units, 1):
        print(f"[fingerprint {i:03d}/{len(all_units):03d}] {unit}", flush=True)
        fp = geometry_fingerprint(DATA_ROOT / unit / "bundle.npz")
        fingerprints.setdefault(fp, []).append(unit)
    exact_groups = [sorted(group) for group in fingerprints.values() if len(group) > 1]
    duplicate_groups = list(exact_groups)
    if not any(set(group) == KNOWN_RELATED for group in duplicate_groups):
        duplicate_groups.append(sorted(KNOWN_RELATED))

    old_train = list(old_split["train_cases"]); old_test = list(old_split["test_cases"])
    ilo_train1234, ilo_test1234 = split_ilo(ilo_units, duplicate_groups, 1234)
    pool_train_by, pool_test_by = split_all(all_units, duplicate_groups, 2025)
    pool_train = [u for name in STRATA for u in pool_train_by[name]]
    pool_test = [u for name in STRATA for u in pool_test_by[name]]

    write_split(
        FIXED_SPLIT, version="split_AG_AAA_ILO_q2v_fixed_test27_train147",
        protocol="freeze Q2V AG/AAA test27; add every approved ILO-before41 to train",
        seed=None, train=old_train + ilo_units, val=[], test=old_test,
        counts={"train": 147, "val": 0, "test": 27, "train_AG_AAA": 106,
                "train_ILO_0": 28, "train_ILO_1": 13},
        duplicate_groups=duplicate_groups, source_split=OLD_SPLIT,
    )
    write_split(
        EXTENDED_SPLIT, version="split_AG_AAA_ILO_q2v_extended_test36_seed1234",
        protocol="freeze Q2V AG/AAA train106/test27; stratify ILO-before41 into train32/test9",
        seed=1234, train=old_train + ilo_train1234, val=[], test=old_test + ilo_test1234,
        counts={"train": 138, "val": 0, "test": 36, "train_AG_AAA": 106,
                "train_ILO_0": 22, "train_ILO_1": 10, "test_AG_AAA": 27,
                "test_ILO_0": 6, "test_ILO_1": 3},
        duplicate_groups=duplicate_groups, source_split=OLD_SPLIT,
    )
    write_split(
        POOL_SPLIT, version="split_AG_AAA_ILO_q2v_pool2025_train138_test36",
        protocol="independent five-stratum grouped 80/20 holdout over all 174 eligible units",
        seed=2025, train=pool_train, val=[], test=pool_test,
        counts={"train": 138, "val": 0, "test": 36,
                **{f"train_{k.replace('/', '_')}": len(v) for k, v in pool_train_by.items()},
                **{f"test_{k.replace('/', '_')}": len(v) for k, v in pool_test_by.items()}},
        duplicate_groups=duplicate_groups,
    )
    pool_control_train = [u for u in pool_train if not u.startswith("ILO/")]
    pool_unused = [u for u in pool_train if u.startswith("ILO/")]
    write_split(
        POOL_CONTROL_SPLIT, version="split_AG_AAA_ILO_q2v_pool2025_control106_test36",
        protocol="matched pool2025 control: AG/AAA train106; same AG/AAA/ILO test36; ILO train32 unused",
        seed=2025, train=pool_control_train, val=[], test=pool_test, unused=pool_unused,
        counts={"train": 106, "val": 0, "test": 36, "unused_ILO": 32},
        duplicate_groups=duplicate_groups,
    )

    old_train_groups = grouped_units(old_train, duplicate_groups)
    arch_val_by: dict[str, list[str]] = {}; arch_train_by: dict[str, list[str]] = {}
    for name in ("AG", "AAA/ruputer", "AAA/unruputer"):
        selected = choose_exact(
            old_train_groups[name], ARCH_VAL_QUOTAS[name], stable_rng(2025, f"arch:{name}")
        )
        selected_set = set(selected)
        arch_val_by[name] = selected
        arch_train_by[name] = sorted(
            unit for group in old_train_groups[name] for unit in group if unit not in selected_set
        )
    arch_train = [u for name in arch_train_by for u in arch_train_by[name]]
    arch_val = [u for name in arch_val_by for u in arch_val_by[name]]
    write_split(
        ARCH_SPLIT, version="split_AG_AAA_q2v_arch_dev_seed2025",
        protocol="Q2V train106 grouped inner development split; original test27 remains untouched",
        seed=2025, train=arch_train, val=arch_val, test=old_test,
        counts={"train": 85, "val": 21, "test": 27,
                "train_AG": 49, "train_AAA_ruputer": 17, "train_AAA_unruputer": 19,
                "val_AG": 12, "val_AAA_ruputer": 4, "val_AAA_unruputer": 5},
        duplicate_groups=duplicate_groups, source_split=OLD_SPLIT,
    )

    stats_jobs = (
        (FIXED_SPLIT, FIXED_REFIT_STATS),
        (POOL_CONTROL_SPLIT, POOL_CONTROL_STATS),
        (POOL_SPLIT, POOL_REFIT_STATS),
        (ARCH_SPLIT, ARCH_STATS),
    )
    for split_path, stats_path in stats_jobs:
        units = json.loads(split_path.read_text(encoding="utf-8"))["train_cases"]
        print(f"[target-stats] {stats_path.name}: {len(units)} cases", flush=True)
        make_stats(DATA_ROOT, units, split_path, stats_path)
        add_stats_breakdown(stats_path)
    make_feature_stats(POOL_CONTROL_SPLIT, POOL_CONTROL_STATS, POOL_CONTROL_FEATURE_STATS)
    make_feature_stats(ARCH_SPLIT, ARCH_STATS, ARCH_FEATURE_STATS)

    base = json.loads(Q2V_CONFIG.read_text(encoding="utf-8"))
    data_rows = (
        ("q2v_ilo_d1_fixed_frozen", "D1-F: add ILO41 to Q2V train; freeze Q2V target+feature stats", FIXED_SPLIT, OLD_STATS, Q2V_FEATURE_STATS),
        ("q2v_ilo_d1_fixed_refit", "D1-R: add ILO41 to Q2V train; refit train147 target+feature stats", FIXED_SPLIT, FIXED_REFIT_STATS, None),
        ("q2v_ilo_d2_extended_frozen", "D2-I: freeze Q2V AG/AAA split; ILO32 train/9 test; freeze Q2V stats", EXTENDED_SPLIT, OLD_STATS, Q2V_FEATURE_STATS),
        ("q2v_ilo_d3_pool2025_control", "D3-C: pool2025 AG/AAA train control; ILO9 zero-shot test", POOL_CONTROL_SPLIT, POOL_CONTROL_STATS, POOL_CONTROL_FEATURE_STATS),
        ("q2v_ilo_d3_pool2025_frozen", "D3-I-F: pool2025 add ILO32; freeze matched D3-C stats", POOL_SPLIT, POOL_CONTROL_STATS, POOL_CONTROL_FEATURE_STATS),
        ("q2v_ilo_d3_pool2025_refit", "D3-I-R: pool2025 add ILO32; refit train138 target+feature stats", POOL_SPLIT, POOL_REFIT_STATS, None),
    )
    config_rows = []
    for exp_id, note, split_path, stats_path, feat_path in data_rows:
        path = DATA_CONFIG_DIR / f"{exp_id}.json"
        write_json(path, clone_config(
            base, exp_id=exp_id, note=note, split_path=split_path,
            stats_path=stats_path, feature_stats_path=feat_path,
        ))
        config_rows.append({"family": "data", "experiment_id": exp_id, "config": str(path.resolve())})

    for nsample in (16, 32, 64):
        for width in (32, 64):
            exp_id = f"q2v_arch_dev_n{nsample}_w{width}"
            path = ARCH_CONFIG_DIR / f"{exp_id}.json"
            write_json(path, clone_config(
                base, exp_id=exp_id,
                note=f"Q2V architecture dev factorial: nsample={nsample}, width={width}; val21 only",
                split_path=ARCH_SPLIT, stats_path=ARCH_STATS,
                feature_stats_path=ARCH_FEATURE_STATS, width=width,
                nsample=nsample, architecture=True,
            ))
            config_rows.append({"family": "architecture", "experiment_id": exp_id,
                                "config": str(path.resolve())})

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "q2v_ilo_data6_plus_architecture6_single_seed",
        "surface_metric_mode": "legacy_vertex",
        "promotion": {"path": str(PROMOTION.resolve()), "sha256": sha256(PROMOTION)},
        "exact_duplicate_geometry_groups": exact_groups,
        "explicit_related_groups": [sorted(KNOWN_RELATED)],
        "splits": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in
                   (FIXED_SPLIT, EXTENDED_SPLIT, POOL_CONTROL_SPLIT, POOL_SPLIT, ARCH_SPLIT)],
        "stats": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in
                  (FIXED_REFIT_STATS, POOL_CONTROL_STATS, POOL_REFIT_STATS, ARCH_STATS,
                   POOL_CONTROL_FEATURE_STATS, ARCH_FEATURE_STATS)],
        "configs": [{**row, "sha256": sha256(Path(row["config"]))} for row in config_rows],
        "frozen_q2v": {
            "config": str(Q2V_CONFIG.resolve()), "config_sha256": sha256(Q2V_CONFIG),
            "target_stats": str(OLD_STATS.resolve()), "target_stats_sha256": sha256(OLD_STATS),
            "feature_stats": str(Q2V_FEATURE_STATS.resolve()),
            "feature_stats_sha256": sha256(Q2V_FEATURE_STATS),
        },
        "deferred": "AreaRandom/strict-area phase remains frozen until 133/133 mapping passes",
    }
    write_json(MANIFEST, payload)
    print(json.dumps({"status": "prepared", "configs": len(config_rows),
                      "manifest": str(MANIFEST)}, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    args = ap.parse_args()
    run(args.repo.resolve())


if __name__ == "__main__":
    main()
