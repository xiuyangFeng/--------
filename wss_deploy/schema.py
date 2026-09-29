"""Versioned metadata for deployment runs and prediction fields.

The first deployment release pre-dates a general result schema and stores WSS
values in a handful of top-level keys.  This module provides a small,
dependency-free envelope around those values.  New fields (for example
velocity or pressure) can use the same ``fields`` and ``time_axis`` objects
without changing the job/report protocol.  The old WSS keys are deliberately
kept by :func:`wss_compatibility` for existing consumers.
"""
from __future__ import annotations

import ast
import json
import hashlib
import re
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .io_utils import atomic_json, file_sha256

SCHEMA_VERSION = "wss-deploy.run-manifest/v1"
RESULT_SCHEMA_VERSION = "wss-deploy.results/v1"
FIELD_SCHEMA_VERSION = "wss-deploy.field/v1"
# v0.14 (J9): version of the post-inference analysis (findings, derived indices, morphology, narrative) a
# summary was written with.  Bump it whenever that analysis changes; ``service upgrade`` then rebuilds every
# finished job whose ``summary.analysis_version`` is older (records without the field use the key probes).
# 2026-09-26 (v0.15.1): model_release provenance paths are redacted (<project> / ~).
# 2026-09-30 (C line, WORKSPACE_V2_CONTRACT.md §5): new ``zones`` block (U10 anatomical zones, wall family);
#   findings listing rules — the global maximum joins its high-WSS cluster, area-type clusters under 1 cm² are
#   not listed, high-WSS items graded against the cohort p90 instead of 7 Pa (U11/U12), reviewer decisions
#   re-matched by kind + position; narrative/one-page wording (lumen diameter, three-head order, hotspot
#   definition); summary ``display_name``.  No prediction, field.npz or golden-regression block changes.
ANALYSIS_VERSION = "2026-09-30"
PACKAGE_DIR = Path(__file__).resolve().parent
TRAINING_PACKAGE = "training_wss_min"
_PROVENANCE_TTL_S = 60.0
_provenance_cache: dict = {}
_provenance_lock = threading.Lock()


def stable_run_identity(*, input_sha256: str | None, release: Mapping[str, Any] | None,
                        mapping: Mapping[str, Any] | None, parameters: Mapping[str, Any] | None,
                        schema_version: str = "wss-deploy.summary/v1") -> str | None:
    """Return a portable identity for comparing immutable prediction runs."""
    if not input_sha256 or not release:
        return None
    payload = {"schema_version": schema_version, "input_sha256": str(input_sha256),
               "release": {"id": release.get("registry_id") or release.get("id") or release.get("release") or release.get("name"),
                           "fingerprint": release.get("fingerprint") or release.get("release_hash")},
               "mapping": dict(mapping or {}), "parameters": dict(parameters or {})}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def redact_paths(value: Any) -> Any:
    """Copy ``value`` with server file-system prefixes removed from every string (v0.15.1, user decision 2026-09-26).

    ``<project>`` replaces the project root and ``~`` the home directory, anywhere in a string, recursively through
    dicts / lists.  Result files (summary / manifest / quality audit / report META) keep the *relative* provenance of
    the training runs without the account name and directory layout of the server."""
    from .paths import PROJECT_ROOT
    prefixes = [(str(PROJECT_ROOT), "<project>")]
    try:
        home = str(Path.home())
        if home and home not in ("/", str(PROJECT_ROOT)):
            prefixes.append((home, "~"))
    except (RuntimeError, OSError):
        pass
    def visit(item):
        if isinstance(item, str):
            for prefix, short in prefixes:
                if prefix in item:
                    item = item.replace(prefix, short)
            return item
        if isinstance(item, dict):
            return {key: visit(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(inner) for inner in item]
        return item
    return visit(value)


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
    # Registry-bound identity is intentionally carried alongside the release
    # metadata.  The manifest hash identifies the bytes on disk; the registry
    # fingerprint identifies the selectable package a job was bound to.
    registry_fingerprint = getattr(release, "registry_fingerprint", None)
    registry_id = getattr(release, "registry_id", None)
    registry_contract = getattr(release, "registry_contract", None)
    if registry_id is not None:
        out["registry_id"] = str(registry_id)
    if registry_fingerprint is not None:
        out["fingerprint"] = str(registry_fingerprint)
    if registry_contract is not None:
        out["contract"] = _jsonable(registry_contract)
    if root:
        release_json = root / "release.json"
        manifest = root / "MANIFEST.sha256"
        if release_json.is_file():
            out["release_json_sha256"] = file_sha256(release_json)
        if manifest.is_file():
            out["manifest_sha256"] = file_sha256(manifest)
            out.setdefault("fingerprint", file_sha256(manifest))
    features = getattr(release, "input_features", None)
    if features is not None:
        out["input_features"] = list(features)
    specs = getattr(release, "model_specs", None)
    if specs is not None:
        out["models"] = _jsonable(specs)
    loaded = getattr(release, "weight_records", None)
    if loaded is not None:
        # Runtime hashes are kept separately from the release manifest hashes;
        # this proves the files actually loaded by the worker were checked.
        out["loaded_weights"] = _jsonable(loaded)
    feature_contract = getattr(release, "feature_contract", None)
    if isinstance(feature_contract, Mapping):
        # Version + source hash of the frozen STL -> feature program the weights ran on.
        out["feature_contract"] = _jsonable(dict(feature_contract))
    if isinstance(info.get("feature_contract"), Mapping):
        out["declared_feature_contract"] = _jsonable(dict(info["feature_contract"]))
    reference_sha = getattr(release, "reference_sha256", None)
    if reference_sha:
        out["reference_sha256"] = str(reference_sha)
        out["reference_profiles"] = [key for key in ("geometry_reference", "population_reference") if isinstance(info.get(key), Mapping)]
    for key in ("model_family", "version"):
        if info.get(key) is not None:
            out[key] = info[key]
    # Do not emit keys with no information: old/custom releases often have a
    # deliberately minimal release.json.
    return redact_paths({key: _jsonable(value) for key, value in out.items() if value is not None})   # v0.15.1: no server paths


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


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _imported_training_modules(source: str, package: str = TRAINING_PACKAGE) -> set[str]:
    """``training_wss_min`` modules a Python source imports (``from training_wss_min import a as A, b`` /
    ``import training_wss_min.x`` / ``from training_wss_min.x import y`` / relative imports inside the package)."""
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:                       # relative import inside the training package
                names.update([module.split(".")[0]] if module else [alias.name for alias in node.names])
            elif module == package:
                names.update(alias.name for alias in node.names)
            elif module.startswith(package + "."):
                names.add(module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(package + "."):
                    names.add(alias.name.split(".")[1])
    return names


def training_modules(project_root: Path | None = None) -> list[Path]:
    """The ``training_wss_min`` files the deployment actually runs: every module ``families.py`` (and the rest of
    ``wss_deploy``) imports, plus what those import inside the package (read with ``ast``; nothing is imported)."""
    project_root = Path(project_root or PACKAGE_DIR.parent)
    package = TRAINING_PACKAGE
    package_dir = project_root / package
    pending: set[str] = set()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if package in text:                      # cheap filter before parsing
            pending |= _imported_training_modules(text, package)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen or not (package_dir / f"{name}.py").is_file():
            continue
        seen.add(name)
        try:
            pending |= _imported_training_modules((package_dir / f"{name}.py").read_text(encoding="utf-8")) - seen
        except (OSError, UnicodeError):
            continue
    return [package_dir / f"{name}.py" for name in sorted(seen)]


def code_source_files(project_root: Path | None = None) -> list[Path]:
    """Files covered by ``source_hash``: ``wss_deploy/*.py`` and the training modules it imports."""
    project_root = Path(project_root or PACKAGE_DIR.parent)
    return sorted(PACKAGE_DIR.glob("*.py")) + training_modules(project_root)


def code_source_hash(project_root: Path | None = None, files: list[Path] | None = None) -> str:
    """sha256 over (relative path, bytes) of :func:`code_source_files` — the code a run actually used."""
    project_root = Path(project_root or PACKAGE_DIR.parent)
    digest = hashlib.sha256()
    for path in (files if files is not None else code_source_files(project_root)):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest.update(path.resolve().relative_to(project_root.resolve()).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def code_provenance(*, refresh: bool = False) -> dict[str, Any]:
    """Deployment code identity (J4): ``deploy_version``, ``git_commit``, ``git_describe``
    (``git describe --always --dirty --long``), ``git_dirty`` and ``source_hash``.  Cached for 60 s."""
    now = time.time()
    with _provenance_lock:
        if not refresh and _provenance_cache.get("at", 0) > now - _PROVENANCE_TTL_S:
            return dict(_provenance_cache["value"])
        root = PACKAGE_DIR.parent
        commit = _git(root, "rev-parse", "HEAD") or os.environ.get("WSS_DEPLOY_GIT_COMMIT")
        describe = _git(root, "describe", "--always", "--dirty", "--long")
        training = training_modules(root)
        value = {"deploy_version": __version__, "git_commit": commit, "git_describe": describe,
                 "git_dirty": describe.endswith("-dirty") if describe else None,
                 "source_hash": code_source_hash(root, sorted(PACKAGE_DIR.glob("*.py")) + training),
                 "training_modules": [p.stem for p in training]}
        _provenance_cache.update(at=now, value=value)
        return dict(value)


def display_name(case_id: Any, patient_id: Any = None) -> str:
    """C7 (2026-09-30): the name a result is shown under — the patient id when one was entered, else the case id.

    The case id comes from the upload's file name and may be a person's name; a patient id is what the
    operator typed on purpose.  Empty / non-text values fall through; never raises."""
    for value in (patient_id, case_id):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def summary_display_name(meta: Mapping[str, Any]) -> str:
    """``display_name`` of a summary: ``case_metadata.patient_id`` when non-empty, else ``case_id``."""
    metadata = meta.get("case_metadata") if isinstance(meta.get("case_metadata"), Mapping) else {}
    return display_name(meta.get("case_id"), metadata.get("patient_id"))


def summary_provenance() -> dict[str, Any]:
    """Keys a stage-B summary should carry (v0.14): analysis version, deployment version and whether the
    working tree was dirty.  ``pipeline`` / ``volume_pipeline`` / ``rebuild_report`` merge this into ``meta``."""
    code = code_provenance()
    return {"analysis_version": ANALYSIS_VERSION, "deploy_version": code["deploy_version"],
            "git_describe": code["git_describe"], "git_dirty": code["git_dirty"], "code_source_hash": code["source_hash"]}


def _code_runtime_metadata() -> dict[str, Any]:
    """Capture reproducibility identifiers without serialising host paths.

    v0.14: ``source_hash`` is the hash of the code actually used (it was the commit id), plus ``git_describe``,
    ``git_dirty`` and ``deploy_version``; ``git_commit`` is unchanged."""
    code = code_provenance()
    return {
        "git_commit": code["git_commit"],
        "git_describe": code["git_describe"],
        "git_dirty": code["git_dirty"],
        "deploy_version": code["deploy_version"],
        "source_hash": code["source_hash"],
        "training_modules": code["training_modules"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _module_version("torch"),
        "numpy": _module_version("numpy"),
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def build_run_manifest(meta: Mapping[str, Any], job_dir: Path, *, outputs: Sequence[str] | None = None) -> dict[str, Any]:
    """Build a portable ``run_manifest.json`` from a stage-B summary."""
    job_dir = Path(job_dir)
    input_sha = meta.get("input_sha256")
    code_runtime = _code_runtime_metadata()
    # J4: the summary's own values (written when the result was computed) win over the manifest writer's.
    from_summary = meta.get("deploy_version") is not None
    result = {
        "schema_version": SCHEMA_VERSION,
        "deploy_version": meta.get("deploy_version") if from_summary else code_runtime["deploy_version"],
        "git_dirty": meta.get("git_dirty") if from_summary else code_runtime["git_dirty"],
        "deploy_version_source": "summary" if from_summary else "manifest_writer",
        "analysis_version": meta.get("analysis_version"),
        "run_id": job_dir.name,
        "run_identity": meta.get("run_identity"),
        "case_id": meta.get("case_id", job_dir.name),
        "case_metadata": _jsonable(meta.get("case_metadata", {})),
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
            "proposal_confirmation_required", "proposal_confidence_gate") if key in meta},
        "parameters": _jsonable(meta.get("run_parameters", {})),
        "runtime": {key: _jsonable(meta[key]) for key in ("device", "gpu", "timing_s", "seconds_per_model") if key in meta},
        "provenance": {"input_sha256": input_sha, "release_hash": meta.get("release_hash"),
                       "sampling_seed": meta.get("sampling_seed"), "frame_transform": meta.get("frame_transform"),
                       "feature_contract": _jsonable(meta.get("feature_contract")),
                       "code_runtime": code_runtime},
        "audit": _jsonable(meta.get("audit", {})),
        "reference_assessment": _jsonable(meta.get("reference_assessment", {})),
        "outputs": _output_records(job_dir, tuple(outputs or ())),
    }
    return _jsonable(result)


def write_run_manifest(job_dir: Path, meta: Mapping[str, Any], *, outputs: Sequence[str] | None = None) -> dict[str, Any]:
    manifest = build_run_manifest(meta, job_dir, outputs=outputs)
    atomic_json(Path(job_dir) / "run_manifest.json", manifest)
    return manifest


__all__ = ["SCHEMA_VERSION", "RESULT_SCHEMA_VERSION", "FIELD_SCHEMA_VERSION", "ANALYSIS_VERSION", "code_provenance", "display_name",
           "summary_display_name",
           "code_source_files", "code_source_hash", "summary_provenance", "training_modules", "stable_run_identity", "model_release_metadata",
           "single_frame_time_axis", "field_descriptor", "wss_compatibility",
           "build_results", "build_run_manifest", "write_run_manifest"]
