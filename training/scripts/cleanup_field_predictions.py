from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..core.utils import dump_json, ensure_dir


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_recorded_path(raw: str | None, target: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    local = target.parent / path.name
    return local.resolve() if local.is_file() else None


def inspect_target(target: str | Path) -> Dict[str, Any]:
    path = Path(target).resolve()
    manifest_path = path / "manifest.json"
    if not path.is_dir():
        raise NotADirectoryError(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"清理目标缺少 manifest.json: {path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    prediction_files = sorted(path.glob("*.pt"))
    checkpoint = _resolve_recorded_path(manifest.get("checkpoint"), path)
    config = _resolve_recorded_path(manifest.get("config_path"), path)
    return {
        "target": str(path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_num_predictions": manifest.get("num_predictions"),
        "subset": manifest.get("subset"),
        "split_version": manifest.get("split_version"),
        "checkpoint": str(checkpoint) if checkpoint else manifest.get("checkpoint"),
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else None,
        "config": str(config) if config else manifest.get("config_path"),
        "config_sha256": _sha256_file(config) if config else None,
        "prediction_file_count": len(prediction_files),
        "prediction_bytes": sum(p.stat().st_size for p in prediction_files),
        "first_prediction": prediction_files[0].name if prediction_files else None,
        "last_prediction": prediction_files[-1].name if prediction_files else None,
    }


def cleanup_targets(
    targets: List[str | Path],
    record_path: str | Path,
    *,
    apply: bool = False,
) -> Dict[str, Any]:
    before = [inspect_target(target) for target in targets]
    if any(item["prediction_file_count"] == 0 for item in before):
        empty = [item["target"] for item in before if item["prediction_file_count"] == 0]
        raise ValueError(f"目标已无逐样本 .pt，拒绝重复执行: {empty}")

    deleted_files = 0
    deleted_bytes = 0
    if apply:
        for item in before:
            target = Path(item["target"])
            for prediction in sorted(target.glob("*.pt")):
                size = prediction.stat().st_size
                prediction.unlink()
                deleted_files += 1
                deleted_bytes += size

    after = []
    for item in before:
        target = Path(item["target"])
        remaining = sorted(target.glob("*.pt"))
        after.append(
            {
                "target": str(target),
                "remaining_prediction_files": len(remaining),
                "manifest_preserved": (target / "manifest.json").is_file(),
            }
        )
    if apply and any(item["remaining_prediction_files"] for item in after):
        raise RuntimeError("清理后仍有逐样本 .pt")

    record = {
        "format": "field_prediction_cleanup_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apply": apply,
        "targets": before,
        "target_count": len(before),
        "planned_files": sum(item["prediction_file_count"] for item in before),
        "planned_bytes": sum(item["prediction_bytes"] for item in before),
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "after": after,
    }
    record_path = Path(record_path).resolve()
    ensure_dir(record_path.parent)
    dump_json(record, record_path)
    if apply:
        for item in before:
            target = Path(item["target"])
            dump_json(
                {
                    "format": "field_prediction_pruned_v1",
                    "storage_status": "pruned",
                    "manifest_preserved": str(target / "manifest.json"),
                    "manifest_sha256_before_cleanup": item["manifest_sha256"],
                    "removed_prediction_files": item["prediction_file_count"],
                    "removed_prediction_bytes": item["prediction_bytes"],
                    "cleanup_record": str(record_path),
                    "note": (
                        "manifest items 仅作历史索引，prediction_path 指向的逐样本 .pt "
                        "已清理；重生前请查看 cleanup record 和 outputs/field/_reproducibility/README.md"
                    ),
                },
                target / "PRUNED.json",
            )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="只删除 field prediction 目录的逐样本 .pt")
    parser.add_argument("--target", action="append", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    record = cleanup_targets(args.target, args.record, apply=args.apply)
    print(
        f"targets={record['target_count']} planned_files={record['planned_files']} "
        f"planned_bytes={record['planned_bytes']} deleted_files={record['deleted_files']} "
        f"deleted_bytes={record['deleted_bytes']} record={Path(args.record).resolve()}"
    )


if __name__ == "__main__":
    main()
