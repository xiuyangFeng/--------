#!/usr/bin/env python3
"""生成 v4 cutover 数据、历史 run 与日志的只读归档清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RETAIN_RUNS = {
    "pointnet_v4": "current AG/AAA v4 PointNet truth",
    "pointnet_distribution_matrix": "contains v3 E2 common-test15 saved-prediction anchor",
    "baseline_2x3_simple": "historical PointNet++ baseline",
    "pointnet_deeper": "E4 documented negative control",
    "pointnet_trainloss_e400": "pre-v4 train-loss protocol anchor",
    "pointnet_wide128": "documented width probe",
}

DATA_TARGETS = {
    "AG_legacy_v3_snapshot": ("data_wss_min/_snapshots/AG_legacy_v3_20260716_signedoff",
                              "archive_candidate", "old AG84 rollback/audit evidence"),
    "AG_v4_staging": ("data_wss_min/_staging/ag_v4_20260715_cutover",
                      "archive_candidate", "original AG77 v4 staging; most files hardlink active AG"),
    "AAA_HAN_pre_fix_snapshot": ("data_wss_min/_snapshots/AAA_HAN_JIAN_FU_pre_fix_20260716_signedoff",
                                 "archive_candidate", "HAN_JIAN_FU pre-fix rollback evidence"),
    "AAA_HAN_fix_staging": ("data_wss_min/_staging/aaa_v4_20260715_fixes",
                            "archive_candidate", "corrected HAN staging; hardlink active AAA"),
    "active_AG": ("data_wss_min/AG", "retain", "active AG76 v4 data"),
    "active_AAA": ("data_wss_min/AAA", "retain", "signed-off AAA data including audit-only exclusions"),
    "fold_stats_v4": ("data_wss_min/fold_stats/v4", "retain", "train-only statistics for active protocols"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan_tree(root: Path, *, include_hashes: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    inode_paths: dict[tuple[int, int], int] = {}
    inode_stats: dict[tuple[int, int], os.stat_result] = {}
    rows = []
    for path in files:
        st = path.stat()
        inode = (st.st_dev, st.st_ino)
        inode_paths[inode] = inode_paths.get(inode, 0) + 1
        inode_stats[inode] = st
        row = {
            "relative_path": str(path.relative_to(root)), "size_bytes": st.st_size,
            "allocated_bytes": st.st_blocks * 512, "device": st.st_dev, "inode": st.st_ino,
            "hardlink_count": st.st_nlink,
        }
        if include_hashes:
            row["sha256"] = sha256(path)
        rows.append(row)
    logical = sum(row["size_bytes"] for row in rows)
    allocated = sum(st.st_blocks * 512 for st in inode_stats.values())
    exclusive = sum(
        st.st_blocks * 512 for inode, st in inode_stats.items()
        if inode_paths[inode] >= st.st_nlink
    )
    external_hardlinks = sum(
        1 for inode, st in inode_stats.items() if inode_paths[inode] < st.st_nlink
    )
    return ({
        "path": str(root), "file_count": len(files), "logical_bytes": logical,
        "allocated_unique_inode_bytes": allocated,
        "estimated_bytes_freed_if_tree_deleted": exclusive,
        "external_hardlinked_inodes": external_hardlinks,
        "manifest_hashed": include_hashes,
    }, rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args()
    repo = args.repo.resolve()
    report_dir = repo / "data_wss_min/pipeline_reports/v4_cutover_20260715_1921"
    manifest_dir = report_dir / "archive_manifests"

    data_rows = []
    for name, (relpath, decision, reason) in DATA_TARGETS.items():
        summary, manifest = scan_tree(repo / relpath, include_hashes=True)
        summary.update({"name": name, "decision": decision, "reason": reason,
                        "manifest": str(manifest_dir / f"{name}.json")})
        write_json(manifest_dir / f"{name}.json", {"root": summary["path"], "files": manifest})
        data_rows.append(summary)

    runs_root = repo / "training_wss_min/runs"
    run_rows = []
    for path in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        summary, manifest = scan_tree(path, include_hashes=True)
        name = path.name
        summary.update({
            "name": name,
            "decision": "retain" if name in RETAIN_RUNS else "archive_candidate",
            "reason": RETAIN_RUNS.get(name, "completed historical WSS run; retain only after verified external archive"),
            "manifest": str(manifest_dir / "runs" / f"{name}.json"),
        })
        write_json(manifest_dir / "runs" / f"{name}.json", {"root": str(path), "files": manifest})
        run_rows.append(summary)

    log_rows = []
    for name, path in (("pipeline_logs", repo / "logs"),
                       ("training_cluster_logs", repo / "training_wss_min/cluster/logs")):
        summary, manifest = scan_tree(path, include_hashes=True)
        summary.update({"name": name, "decision": "archive_candidate",
                        "reason": "completed logs; conclusions retained in reports",
                        "manifest": str(manifest_dir / f"{name}.json")})
        write_json(manifest_dir / f"{name}.json", {"root": str(path), "files": manifest})
        log_rows.append(summary)

    archive_root = Path("/data/user_data/cy/Digital_twin/GNN_archive/wss_min/20260716_v4_pointnet_acceptance")
    archive_parent_usage = shutil.disk_usage(archive_root.parents[3])
    release_data = sum(x["estimated_bytes_freed_if_tree_deleted"] for x in data_rows
                       if x["decision"] == "archive_candidate")
    release_runs = sum(x["estimated_bytes_freed_if_tree_deleted"] for x in run_rows
                       if x["decision"] == "archive_candidate")
    release_logs = sum(x["estimated_bytes_freed_if_tree_deleted"] for x in log_rows)
    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "inventory_only; no archive and no deletion performed",
        "archive_recommendation": {
            "path": str(archive_root), "format": "separate tar.gz packages using tar+pigz",
            "parent_writable": os.access(archive_root.parents[3], os.W_OK),
            "archive_filesystem_free_bytes": archive_parent_usage.free,
            "required_order": [
                "create package without modifying source", "write file and package SHA-256",
                "reread and verify archive", "record destination", "request explicit deletion approval",
            ],
        },
        "data_directories": data_rows, "run_directories": run_rows, "log_directories": log_rows,
        "estimated_release_after_verified_archive_and_confirmed_deletion": {
            "data_bytes": release_data, "historical_run_bytes": release_runs,
            "log_bytes": release_logs, "total_bytes": release_data + release_runs + release_logs,
        },
        "never_delete_in_this_round": [
            "data_wss_min/AG", "data_wss_min/AAA", "data_wss_min/fold_stats/v4",
            "training_wss_min/runs/pointnet_v4",
            "training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_global",
        ],
        "out_of_scope_observation": {
            "path": "outputs/field", "approximate_size": "84G",
            "policy": "not part of WSS-min acceptance; not modified or classified for deletion",
        },
    }
    output = report_dir / "v4_archive_space_inventory.json"
    write_json(output, payload)
    print(json.dumps({"output": str(output),
                      "estimated_release_bytes": payload["estimated_release_after_verified_archive_and_confirmed_deletion"]["total_bytes"],
                      "manifests": str(manifest_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
