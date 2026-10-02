"""Point an experiment config (or a directory of configs) at the corrected data versions of 2026-09-30.

    python -m training_wss_min.tools.repoint_data_root <config.json | dir> --out <new dir> [--name-prefix <exp>] [--dry-run]

The library label fix of 2026-09-30 (9 units re-simulated with protocol-correct boundary conditions) produced two
corrected view roots whose layout and file names mirror their predecessors exactly:

    data_wss_v5/views_v5_2_full_20260923        ->  data_wss_v5/views_v5_2c_20260930   (IND / CV5 / syn splits)
    data_wss_v5/views_v5_2p4_full265_20260930   ->  data_wss_v5/views_v5_2c_20260930   (full265 + recover8; same file names)
    data_wss_v5/views_v5_2p5_full265_20260930   ->  data_wss_v5/views_v5_2c_20260930   (the interim v5.2p5 root, merged 09-30)

(v5.2c is the single unified data version: every unit, every pack, every split and statistics file in one root) so repointing is a pure root-prefix substitution in every string field (data_root, split_path, wss_stats_path,
cycle / multi statistics, point_features_root, density_aug_root, ...). Input-feature z-score files live in experiment
directories and are geometry-only; they were checked to be unchanged (wss_v52c_labelfix_20260930/feature_stats_check.json)
and stay as they are. Configs that reference no replaced root are copied unchanged and reported. Nothing is trained here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
# 2026-10-01: the unified root is v5.2d (library audit merge; views_v5_2c_20260930 was renamed, so every older root maps to it)
NEW_ROOT = str(G / "data_wss_v5/views_v5_2d_20261001")
MAP = {str(G / "data_wss_v5/views_v5_2_full_20260923"): NEW_ROOT,
       str(G / "data_wss_v5/views_v5_2p4_full265_20260930"): NEW_ROOT,
       str(G / "data_wss_v5/views_v5_2p5_full265_20260930"): NEW_ROOT,
       str(G / "data_wss_v5/views_v5_2c_20260930"): NEW_ROOT,
       str(G / "data_wss_v5/anatomy_pointcloud_v5_2c_20260930"): str(G / "data_wss_v5/anatomy_pointcloud_v5_2d_20261001"),
       str(G / "training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache"): str(G / "training_wss_min/experiments/joint_cycle_v52d_20261001/data_audit_cache")}
# relative spellings used inside some split files
MAP.update({k.replace(str(G) + "/", ""): v.replace(str(G) + "/", "") for k, v in list(MAP.items())})


def repoint(obj, hits: list):
    if isinstance(obj, dict):
        return {k: repoint(v, hits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [repoint(v, hits) for v in obj]
    if isinstance(obj, str):
        for old, new in MAP.items():
            if old in obj:
                hits.append(obj)
                return obj.replace(old, new)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name-prefix", default=None, help="replace the experiment part of cfg['name'] (before the first '/')")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = sorted(a.src.glob("*.json")) if a.src.is_dir() else [a.src]
    report = []
    for f in files:
        cfg = json.loads(f.read_text())
        hits: list = []
        new = repoint(cfg, hits)
        if isinstance(new, dict) and hits:
            if a.name_prefix and isinstance(new.get("name"), str):
                new["name"] = a.name_prefix + "/" + new["name"].split("/", 1)[-1]
            new["notes"] = (str(new.get("notes", "")) + " | 2026-09-30 label fix: data roots repointed to the corrected versions "
                            "(unified v5.2d since 2026-10-01) by training_wss_min/tools/repoint_data_root.py; recipe unchanged").strip(" |")
        report.append({"config": str(f), "repointed_fields": len(hits)})
        if not a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f.name).write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
