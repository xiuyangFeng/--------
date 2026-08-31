from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from ..core.config import ExperimentConfig
from ..core.legacy_snapshot import LEGACY_SNAPSHOT_FORMAT
from ..core.utils import dump_json, ensure_dir


STATIC_KEYS = ("x", "edge_index", "wall_mask")
DYNAMIC_KEYS = ("global_cond", "time_value", "y_true")
OPTIONAL_DYNAMIC_KEYS = ("y_wss_true", "y_wss_global_true")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_case_filename(case_name: str) -> str:
    return case_name.replace("/", "__").replace(" ", "_") + ".pt"


def _resolve_prediction_path(prediction_dir: Path, item: Dict[str, Any]) -> Path:
    raw = Path(item["prediction_path"])
    if raw.is_file():
        return raw
    local = prediction_dir / raw.name
    if local.is_file():
        return local
    raise FileNotFoundError(f"预测文件不存在: {raw} / {local}")


def _require_tensor(payload: Dict[str, Any], key: str, source: Path) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"{source} 缺少 tensor {key}")
    return value


def _stack(payloads: Iterable[Dict[str, Any]], key: str, paths: List[Path]) -> torch.Tensor:
    values = [_require_tensor(payload, key, path) for payload, path in zip(payloads, paths)]
    return torch.stack(values, dim=0)


def build_snapshot(
    source_predictions: str | Path,
    source_config: str | Path,
    output: str | Path,
    *,
    force: bool = False,
) -> Path:
    prediction_dir = Path(source_predictions).resolve()
    config_path = Path(source_config).resolve()
    output_dir = Path(output).resolve()
    manifest_path = prediction_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"源 manifest 不存在: {manifest_path}")

    config = ExperimentConfig.from_json(config_path)
    config.validate()
    with manifest_path.open("r", encoding="utf-8") as f:
        source_manifest = json.load(f)
    items = list(source_manifest.get("items") or [])
    if not items:
        raise ValueError(f"源 manifest 没有 items: {manifest_path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"输出目录非空，请使用 --force: {output_dir}")
        shutil.rmtree(output_dir)
    cases_dir = ensure_dir(output_dir / "cases")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["case_name"])].append(item)

    case_records: List[Dict[str, Any]] = []
    total_tensor_bytes = 0
    for case_name in sorted(grouped):
        case_items = sorted(grouped[case_name], key=lambda x: str(x["sample_id"]))
        paths = [_resolve_prediction_path(prediction_dir, item) for item in case_items]
        payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
        first = payloads[0]

        for key in STATIC_KEYS:
            ref = _require_tensor(first, key, paths[0])
            for payload, path in zip(payloads[1:], paths[1:]):
                if not torch.equal(ref, _require_tensor(payload, key, path)):
                    raise ValueError(f"{case_name} 病例内静态张量 {key} 发生变化: {path}")

        case_payload: Dict[str, Any] = {
            "format": LEGACY_SNAPSHOT_FORMAT,
            "case_name": case_name,
            "x": first["x"].clone(),
            "edge_index": first["edge_index"].clone(),
            "wall_mask": first["wall_mask"].clone(),
            "sample_ids": [str(item["sample_id"]) for item in case_items],
            "graph_paths": [str(item["graph_path"]) for item in case_items],
        }
        for key in DYNAMIC_KEYS:
            case_payload[key] = _stack(payloads, key, paths)
        for key in OPTIONAL_DYNAMIC_KEYS:
            present = [key in payload for payload in payloads]
            if any(present) and not all(present):
                raise ValueError(f"{case_name} 的 {key} 在时间帧间不完整")
            if all(present):
                case_payload[key] = _stack(payloads, key, paths)

        for meta_key in (
            "node_feature_names",
            "target_names",
            "wss_target_names",
            "wss_target_frame",
        ):
            if meta_key in first:
                case_payload[meta_key] = first[meta_key]

        rel = Path("cases") / _safe_case_filename(case_name)
        out_path = output_dir / rel
        torch.save(case_payload, out_path)
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for value in case_payload.values()
            if torch.is_tensor(value)
        )
        total_tensor_bytes += tensor_bytes
        case_records.append(
            {
                "case_name": case_name,
                "payload": str(rel),
                "num_frames": len(case_items),
                "num_nodes": int(first["x"].size(0)),
                "tensor_bytes": tensor_bytes,
                "payload_bytes": out_path.stat().st_size,
                "payload_sha256": _sha256_file(out_path),
            }
        )
        del payloads, case_payload
        gc.collect()
        print(f"snapshot case: {case_name} frames={len(case_items)} -> {out_path}")

    snapshot_manifest = {
        "format": LEGACY_SNAPSHOT_FORMAT,
        "source_prediction_dir": str(prediction_dir),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_config": str(config_path),
        "source_config_sha256": _sha256_file(config_path),
        "source_checkpoint": source_manifest.get("checkpoint"),
        "source_subset": source_manifest.get("subset"),
        "source_split_version": source_manifest.get("split_version"),
        "enabled_node_features": list(config.data.enabled_node_features),
        "enabled_global_features": list(config.data.enabled_global_features),
        "wss_target_frame": config.data.wss_target_frame,
        "num_cases": len(case_records),
        "num_frames": len(items),
        "tensor_bytes": total_tensor_bytes,
        "cases": case_records,
    }
    dump_json(snapshot_manifest, output_dir / "manifest.json")
    print(
        f"legacy snapshot 完成: cases={len(case_records)} frames={len(items)} "
        f"tensor_bytes={total_tensor_bytes} output={output_dir}"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="从历史 field prediction 构建去重输入快照")
    parser.add_argument("--source-predictions", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_snapshot(
        args.source_predictions,
        args.source_config,
        args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
