"""Versioned metadata for deployment runs and prediction fields.

The first deployment release pre-dates a general result schema and stores WSS
values in a handful of top-level keys.  This module provides a small,
dependency-free envelope around those values.  New fields (for example
velocity or pressure) can use the same ``fields`` and ``time_axis`` objects
without changing the job/report protocol.  The old WSS keys are deliberately
kept by :func:`wss_compatibility` for existing consumers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_json, file_sha256

SCHEMA_VERSION = "wss-deploy.run-manifest/v1"
RESULT_SCHEMA_VERSION = "wss-deploy.results/v1"
FIELD_SCHEMA_VERSION = "wss-deploy.field/v1"


def _jsonable(value: Any) -> Any:
    """Convert common numpy/scalar values before writing a manifest."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _release_path(release: Any) -> Path | None:
    path = getattr(release, "dir", release if isinstance(release, (str, Path)) else None)
    return Path(path).resolve() if path else None


def _manifest_weights(root: Path) -> list[dict[str, Any]]:
    """Read checkpoint hashes from a release MANIFEST when available.

    Paths in a run manifest are relative to the release root.  This keeps
    results portable between the deployment host and an archive while still
    making every loaded checkpoint identifiable.
    """
    manifest = root / "MANIFEST.sha256"
    records: list[dict[str, Any]] = []
    if manifest.is_file():
        line_re = re.compile(r"^([0-9a-fA-F]{64})\s+(\S+)\s+(\d+)$")
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            match = line_re.match(line.strip())
            if not match or not match.group(2).startswith("models/") or "/ckpt_" not in match.group(2):
                continue
            relpath = match.group(2)
            seed_match = re.search(r"_s([^/]+)", relpath)
            records.append({"path": relpath, "sha256": match.group(1).lower(),
                            "size_bytes": int(match.group(3)),
                            "seed": seed_match.group(1) if seed_match else None})
    if not records:
        for path in sorted(root.glob("models/*/ckpt_*.pt")):
            records.append({"path": path.relative_to(root).as_posix(),
                            "sha256": file_sha256(path), "size_bytes": path.stat().st_size,
                            "seed": path.parent.name.rsplit("_s", 1)[-1] if "_s" in path.parent.name else None})
    return records


def model_release_metadata(release: Any) -> dict[str, Any]:
    """Return portable identity/provenance for a loaded model release.

    ``release`` may be the deployment :class:`~wss_deploy.infer.Release`, a
    release directory, or a release-like object exposing ``dir``, ``info``
    and ``name``.  No model object or absolute path is serialized.
    """
    root = _release_path(release)
    info = getattr(release, "info", None)
    if info is None and root and (root / "release.json").is_file():
        info = json.loads((root / "release.json").read_text(encoding="utf-8"))
    info = dict(info or {})
    name = getattr(release, "name", None)
    if callable(name):
        name = name()
    name = name or info.get("release") or (root.name if root else "unknown")
    out: dict[str, Any] = {
        "name": str(name),
        "release": str(info.get("release", name)),
        "frozen_on": info.get("frozen_on"),
        "git_commit": info.get("git_commit"),
        "target": info.get("target"),
        "input_contract": info.get("input_contract"),
        "ensemble_protocol": info.get("ensemble_protocol"),
        "source_runs": info.get("source_runs"),
        "weights": _manifest_weights(root) if root else [],
    }
    if root:
        release_json = root / "release.json"
        manifest = root / "MANIFEST.sha256"
        if release_json.is_file():
            out["release_json_sha256"] = file_sha256(release_json)
        if manifest.is_file():
            out["manifest_sha256"] = file_sha256(manifest)
    features = getattr(release, "input_features", None)
    if features is not None:
        out["input_features"] = list(features)
    specs = getattr(release, "model_specs", None)
    if specs is not None:
        out["models"] = _jsonable(specs)
    for key in ("model_family", "version"):
        if info.get(key) is not None:
            out[key] = info[key]
    # Do not emit keys with no information: old/custom releases often have a
    # deliberately minimal release.json.
    return {key: _jsonable(value) for key, value in out.items() if value is not None}


def single_frame_time_axis(model_frame: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Normalize the current fixed-frame model to the future frame protocol."""
    model_frame = dict(model_frame or {})
    frame: dict[str, Any] = {"index": 0}
    if model_frame.get("step") is not None:
        frame["step"] = int(model_frame["step"])
    if model_frame.get("time_s") is not None:
        frame["time_s"] = float(model_frame["time_s"])
    label = model_frame.get("label") or model_frame.get("target") or "frame_0"
    frame["label"] = str(label)
    return [frame]


def field_descriptor(field_id: str, *, label: str, units: str, location: str,
                     kind: str = "scalar", components: int = 1,
                     array_key: str | None = None, time_indices: Sequence[int] = (0,),
                     axis_order: Sequence[str] = ("point",),
                     statistics_key: str | None = None, source: str = "prediction") -> dict[str, Any]:
    """Describe one scalar/vector field without embedding its values."""
    if not field_id or not label or not units or components < 1:
        raise ValueError("field_id, label, units and positive components are required")
    return {"schema_version": FIELD_SCHEMA_VERSION, "id": str(field_id), "label": str(label),
            "units": str(units), "location": str(location), "kind": str(kind),
            "components": int(components), "array_key": array_key or f"{field_id}",
            "axis_order": list(axis_order), "time_indices": [int(i) for i in time_indices],
            "source": str(source), **({"statistics_key": statistics_key} if statistics_key else {})}


def wss_compatibility(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the historical WSS result keys as an explicit compatibility view."""
    keys = ("statistics_protocol", "wss_field_pa", "peak", "per_branch", "geometry", "branch_names")
    return {key: _jsonable(meta[key]) for key in keys if key in meta}


def build_results(*, time_axis: Sequence[Mapping[str, Any]], fields: Mapping[str, Mapping[str, Any]],
                  statistics: Mapping[str, Any], compatibility: Mapping[str, Any]) -> dict[str, Any]:
    """Build the generic result envelope stored in ``summary.json``."""
    return {"schema_version": RESULT_SCHEMA_VERSION, "time_axis": _jsonable(list(time_axis),),
            "fields": _jsonable(dict(fields)), "statistics": _jsonable(dict(statistics)),
            "compatibility": _jsonable(dict(compatibility))}


def _portable_path(job_dir: Path, value: Any) -> str | None:
    if not value:
        return None
    try:
        path = Path(str(value)).resolve()
        return path.relative_to(Path(job_dir).resolve()).as_posix()
    except (OSError, ValueError):
        return Path(str(value)).name


def _output_records(job_dir: Path, outputs: Sequence[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    root = Path(job_dir).resolve()
    for name in outputs:
        rel = Path(str(name))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"output path must be relative to job directory: {name}")
        path = (root / rel).resolve()
        if path.parent != root or not path.is_file():
            records[rel.as_posix()] = {"path": rel.as_posix(), "present": False}
            continue
        records[rel.as_posix()] = {"path": rel.as_posix(), "present": True,
                                   "size_bytes": int(path.stat().st_size),
                                   "sha256": file_sha256(path)}
    return records


def build_run_manifest(meta: Mapping[str, Any], job_dir: Path, *, outputs: Sequence[str] | None = None) -> dict[str, Any]:
    """Build a portable ``run_manifest.json`` from a stage-B summary."""
    job_dir = Path(job_dir)
    input_sha = meta.get("input_sha256")
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": job_dir.name,
        "case_id": meta.get("case_id", job_dir.name),
        "created_at": meta.get("created_at"),
        "input": {"stl": {"path": "input.stl", "sha256": input_sha},
                  "clean_stl": _portable_path(job_dir, meta.get("input_check", {}).get("clean_stl")),
                  "units": meta.get("input_check", {}).get("resolved_units") or meta.get("input_check", {}).get("units")},
        "model_release": _jsonable(meta.get("model_release", {})),
        "time_axis": _jsonable(meta.get("time_axis", [])),
        "fields": _jsonable(meta.get("fields", {})),
        "results": _jsonable(meta.get("results", {})),
        "mapping": _jsonable(meta.get("mapping", {})),
        "review": {key: _jsonable(meta[key]) for key in (
            "outlets_confirmed", "proposal_confidence", "proposal_side_confidence",
            "proposal_confirmation_required") if key in meta},
        "parameters": _jsonable(meta.get("run_parameters", {})),
        "runtime": {key: _jsonable(meta[key]) for key in ("device", "gpu", "timing_s", "seconds_per_model") if key in meta},
        "provenance": {"input_sha256": input_sha, "release_hash": meta.get("release_hash"),
                       "sampling_seed": meta.get("sampling_seed"), "frame_transform": meta.get("frame_transform")},
        "outputs": _output_records(job_dir, tuple(outputs or ())),
    }
    return _jsonable(result)


def write_run_manifest(job_dir: Path, meta: Mapping[str, Any], *, outputs: Sequence[str] | None = None) -> dict[str, Any]:
    manifest = build_run_manifest(meta, job_dir, outputs=outputs)
    atomic_json(Path(job_dir) / "run_manifest.json", manifest)
    return manifest


__all__ = ["SCHEMA_VERSION", "RESULT_SCHEMA_VERSION", "FIELD_SCHEMA_VERSION", "model_release_metadata",
           "single_frame_time_axis", "field_descriptor", "wss_compatibility",
           "build_results", "build_run_manifest", "write_run_manifest"]
