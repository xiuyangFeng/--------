#!/usr/bin/env python3
"""生成 AG76+AAA57 的 seed1234 分层 train106/test27 协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training_wss_min.tools.prepare_v4_protocol import FRAME, make_stats, sha256, write_json


SEED = 1234
STRATA = ("AG", "AAA/ruputer", "AAA/unruputer")
TEST_QUOTAS = {"AG": 15, "AAA/ruputer": 6, "AAA/unruputer": 6}
KNOWN_DUPLICATE = frozenset({"AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"})


def geometry_fingerprint(bundle: Path) -> str:
    """对 v4 raw wall 坐标做顺序无关的精确量化指纹。"""
    with np.load(bundle, allow_pickle=False) as z:
        frame = str(np.asarray(z["transform_frame_version"]).item())
        if frame != FRAME:
            raise RuntimeError(f"wrong frame: {bundle}={frame}")
        xyz = np.asarray(z["wall_coords_raw"], dtype=np.float64)
    xyz = np.round(xyz, decimals=5)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    payload = np.ascontiguousarray(xyz[order], dtype="<f8").tobytes()
    h = hashlib.sha256()
    h.update(np.asarray(xyz.shape, dtype="<i8").tobytes())
    h.update(payload)
    return h.hexdigest()


def stratum(unit: str) -> str:
    if unit.startswith("AG/"):
        return "AG"
    if unit.startswith("AAA/ruputer/"):
        return "AAA/ruputer"
    if unit.startswith("AAA/unruputer/"):
        return "AAA/unruputer"
    raise ValueError(unit)


def choose_test_groups(groups: list[list[str]], quota: int, rng: np.random.Generator) -> list[list[str]]:
    order = rng.permutation(len(groups)).tolist()
    shuffled = [groups[i] for i in order]
    # Dynamic programming returns the first seed-determined exact subset.
    dp: dict[int, list[int]] = {0: []}
    for i, group in enumerate(shuffled):
        for total, selected in sorted(list(dp.items()), reverse=True):
            new_total = total + len(group)
            if new_total <= quota and new_total not in dp:
                dp[new_total] = selected + [i]
    if quota not in dp:
        raise RuntimeError(f"cannot satisfy exact test quota={quota} with group sizes")
    return [shuffled[i] for i in dp[quota]]


def run(repo: Path) -> dict:
    data_root = repo / "data_wss_min"
    report_dir = data_root / "pipeline_reports/v4_cutover_20260715_1921"
    whitelist_path = report_dir / "v4_training_whitelist.json"
    quality_path = report_dir / "v4_training_quality_exclusions.json"
    geometry_exclusions_path = report_dir / "v4_final_exclusions.json"
    whitelist = json.loads(whitelist_path.read_text())
    quality = json.loads(quality_path.read_text())
    geometry_exclusions = json.loads(geometry_exclusions_path.read_text())
    units = sorted(whitelist["AG"] + whitelist["AAA"])
    if len(whitelist["AG"]) != 76 or len(whitelist["AAA"]) != 57 or len(units) != 133:
        raise RuntimeError("unexpected eligible pool; expected AG76+AAA57=133")

    forbidden = {x["unit_id"] for x in quality["excluded_units"]}
    forbidden.update(x["unit_id"] for x in geometry_exclusions["manual_exclusions"])
    if forbidden & set(units):
        raise RuntimeError("excluded unit leaked into training whitelist")

    fingerprints: dict[str, list[str]] = {}
    for unit in units:
        fp = geometry_fingerprint(data_root / unit / "bundle.npz")
        fingerprints.setdefault(fp, []).append(unit)
    exact_duplicate_groups = [sorted(group) for group in fingerprints.values() if len(group) > 1]
    if not KNOWN_DUPLICATE.issubset(units):
        raise RuntimeError("known HOU_SHEN_QIAN/KANG_XI_MING pair is missing from eligible pool")
    # 该对病例的 v4 raw wall 坐标仅有亚毫米量化差异，不是逐字节相同；按既有
    # 项目事实显式设为原子组，同时继续报告全池精确指纹审计结果。
    duplicate_groups = list(exact_duplicate_groups)
    if not any(set(group) == KNOWN_DUPLICATE for group in duplicate_groups):
        duplicate_groups.append(sorted(KNOWN_DUPLICATE))
    if any(len({stratum(unit) for unit in group}) != 1 for group in duplicate_groups):
        raise RuntimeError("duplicate geometry crosses requested strata")

    grouped_by_stratum: dict[str, list[list[str]]] = {name: [] for name in STRATA}
    assigned: set[str] = set()
    for group in duplicate_groups:
        grouped_by_stratum[stratum(group[0])].append(sorted(group))
        assigned.update(group)
    for unit in units:
        if unit not in assigned:
            grouped_by_stratum[stratum(unit)].append([unit])
    for groups in grouped_by_stratum.values():
        groups.sort(key=lambda x: x[0])

    rng = np.random.default_rng(SEED)
    test_by_stratum: dict[str, list[str]] = {}
    train_by_stratum: dict[str, list[str]] = {}
    for name in STRATA:
        selected_groups = choose_test_groups(grouped_by_stratum[name], TEST_QUOTAS[name], rng)
        selected = {unit for group in selected_groups for unit in group}
        test_by_stratum[name] = sorted(selected)
        train_by_stratum[name] = sorted(unit for group in grouped_by_stratum[name]
                                         for unit in group if unit not in selected)

    train_cases = [unit for name in STRATA for unit in train_by_stratum[name]]
    test_cases = [unit for name in STRATA for unit in test_by_stratum[name]]
    if (len(train_cases), len(test_cases)) != (106, 27):
        raise RuntimeError("unexpected stratified split counts")
    if set(train_cases) & set(test_cases) or set(train_cases + test_cases) != set(units):
        raise RuntimeError("partition coverage/leakage failure")
    if any(bool(set(group) & set(train_cases)) and bool(set(group) & set(test_cases))
           for group in duplicate_groups):
        raise RuntimeError("duplicate geometry leaked across partitions")

    old_split_path = repo / "training/splits/split_AG_wss_min_v4_traintest.json"
    old_test = set(json.loads(old_split_path.read_text())["test_cases"])
    split_path = repo / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
    split = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "pipeline_wss_min",
        "data_root": str(data_root.resolve()),
        "required_frame_version": FRAME,
        "id_format": "canonical cohort/subset/case",
        "split_version": "split_AG_AAA_wss_min_v4_stratified_seed1234",
        "protocol": "eligible AG76+AAA57 pooled, deterministic 80/20 stratified holdout; no validation",
        "seed": SEED,
        "algorithm": "sort canonical IDs; group identical 1e-5-mm wall geometry; numpy default_rng; exact-quota DP per fixed stratum order",
        "stratum_order": list(STRATA),
        "test_quotas": TEST_QUOTAS,
        "source_training_whitelist": str(whitelist_path.resolve()),
        "source_training_whitelist_sha256": sha256(whitelist_path),
        "source_quality_exclusions": str(quality_path.resolve()),
        "source_quality_exclusions_sha256": sha256(quality_path),
        "source_geometry_exclusions": str(geometry_exclusions_path.resolve()),
        "source_geometry_exclusions_sha256": sha256(geometry_exclusions_path),
        "duplicate_geometry_groups": duplicate_groups,
        "exact_duplicate_geometry_groups": exact_duplicate_groups,
        "explicit_related_groups": [sorted(KNOWN_DUPLICATE)],
        "train_by_stratum": train_by_stratum,
        "test_by_stratum": test_by_stratum,
        "train_cases": train_cases,
        "val_cases": [],
        "test_cases": test_cases,
        "excluded_cases": sorted(forbidden),
        "counts": {
            "train": 106, "val": 0, "test": 27,
            "train_AG": 61, "train_AAA_ruputer": 21, "train_AAA_unruputer": 24,
            "test_AG": 15, "test_AAA_ruputer": 6, "test_AAA_unruputer": 6,
        },
        "expected_counts": {"train": 106, "val": 0, "test": 27},
        "legacy_AG_test15_locked": False,
        "legacy_AG_test15_overlap": sorted(old_test & set(test_cases)),
        "comparability": "primary test27 metrics are not directly comparable with prior common-test15 totals",
    }
    write_json(split_path, split)

    stats_path = data_root / "fold_stats/v4/AG_AAA_v4_stratified_train106_global_stats.json"
    stats = make_stats(data_root, train_cases, split_path, stats_path)
    n_by_cohort = {
        "AG": sum(row["n_points"] for row in stats["bundle_manifest"] if row["unit_id"].startswith("AG/")),
        "AAA": sum(row["n_points"] for row in stats["bundle_manifest"] if row["unit_id"].startswith("AAA/")),
    }
    stats["point_contribution_by_cohort"] = {
        name: {"n_points": count, "fraction": count / stats["n_points"]}
        for name, count in n_by_cohort.items()
    }
    stats["train_stratum_counts"] = {name: len(train_by_stratum[name]) for name in STRATA}
    write_json(stats_path, stats)
    return {
        "split_path": str(split_path), "split_sha256": sha256(split_path),
        "stats_path": str(stats_path), "stats_sha256": sha256(stats_path),
        "counts": split["counts"], "duplicate_geometry_groups": duplicate_groups,
        "exact_duplicate_geometry_groups": exact_duplicate_groups,
        "legacy_AG_test15_overlap": split["legacy_AG_test15_overlap"],
        "point_contribution_by_cohort": stats["point_contribution_by_cohort"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args()
    print(json.dumps(run(args.repo.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
