#!/usr/bin/env python3
"""Static proof that each xyz-only control changes nothing else."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training_wss_min.config import ExpConfig


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix="") -> dict[str, object]:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: value}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest_path, output = Path(args.manifest), Path(args.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 3:
        raise RuntimeError("expected the exact three-task xyz-only manifest")
    allowed = {"name", "notes", "data.input_features"}
    audited = []
    for row in rows:
        parent_path, child_path = Path(row["parent_config"]), Path(row["config"])
        if sha(parent_path) != row["parent_sha256"] or sha(child_path) != row["sha256"]:
            raise RuntimeError(f"parent or child hash drift: {row['experiment_id']}")
        parent = flatten(json.loads(parent_path.read_text(encoding="utf-8")))
        child = flatten(json.loads(child_path.read_text(encoding="utf-8")))
        changed = sorted({*parent, *child} - {key for key in {*parent, *child} if parent.get(key) == child.get(key)})
        if set(changed) - allowed:
            raise RuntimeError(f"unexpected config drift for {row['experiment_id']}: {changed}")
        cfg = ExpConfig.from_json(child_path)
        if tuple(cfg.data.input_features) != ("x", "y", "z"):
            raise RuntimeError(f"non-xyz input features: {row['experiment_id']}")
        if cfg.run_dir.exists():
            raise FileExistsError(f"run directory already exists: {cfg.run_dir}")
        audited.append({"experiment_id": row["experiment_id"], "changed_fields": changed,
                        "config": str(child_path.resolve()), "config_sha256": sha(child_path)})
    payload = {"status": "passed", "manifest": str(manifest_path.resolve()),
               "manifest_sha256": sha(manifest_path), "jobs": audited}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "jobs": len(audited)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
