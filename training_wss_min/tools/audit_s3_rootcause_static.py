#!/usr/bin/env python3
"""Static single-variable audit for the S3 root-cause matrix.

Pure-JSON (no torch): verifies every child config differs from the frozen S3
parent ONLY in its declared `only_allowed_differences`, that recorded shas are
current, and that no run_dir already exists.  Written so the formal gate can run
it before the geometry/GPU smoke without paying the torch import cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(obj: dict, prefix: str = "") -> dict:
    """Flatten nested dicts to dotted leaf paths; lists are treated as leaves."""
    out: dict[str, object] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        else:
            out[path] = value
    return out


def diff_paths(parent: dict, child: dict) -> set[str]:
    flat_parent, flat_child = flatten(parent), flatten(child)
    keys = set(flat_parent) | set(flat_child)
    return {k for k in keys if flat_parent.get(k) != flat_child.get(k)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["configs"]
    results, errors = [], []
    for row in rows:
        cfg_path = Path(row["config"])
        parent_path = Path(row["parent_config"])
        entry = {"experiment_id": row["experiment_id"], "variant": row.get("variant"),
                 "allowed": row["only_allowed_differences"]}
        try:
            if sha(cfg_path) != row["sha256"]:
                raise RuntimeError("config sha drift vs manifest")
            if sha(parent_path) != row["parent_config_sha256"]:
                raise RuntimeError("parent config sha drift vs manifest")
            child = json.loads(cfg_path.read_text(encoding="utf-8"))
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            observed = diff_paths(parent, child)
            allowed = set(row["only_allowed_differences"])
            extra = observed - allowed
            missing = allowed - observed - {"name"}  # name always differs; tolerate seed==1234 no-op
            if extra:
                raise RuntimeError(f"unexpected differing fields: {sorted(extra)}")
            if Path(row["run_dir"]).exists():
                raise RuntimeError(f"run_dir already exists: {row['run_dir']}")
            entry.update({"observed_differences": sorted(observed), "status": "passed"})
        except Exception as exc:  # noqa: BLE001
            entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            errors.append(entry["experiment_id"])
        results.append(entry)

    status = "passed" if not errors else "failed"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest),
        "rows": len(rows),
        "results": results,
    }
    if errors:
        payload["failed"] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "rows": len(rows), "failed": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
