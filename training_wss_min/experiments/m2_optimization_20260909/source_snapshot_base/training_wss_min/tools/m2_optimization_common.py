"""Frozen sources and atomic queue state for the authorised M2 experiments."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "m2_optimization_20260909"
EXP = ROOT / "training_wss_min/experiments" / PREFIX
CONFIGS = ROOT / "training_wss_min/configs" / PREFIX
RUNS = ROOT / "training_wss_min/runs"
MATRIX = CONFIGS / "matrix.json"
COMBINATIONS = CONFIGS / "combinations.json"
ANCHOR_CONFIG = ROOT / "training_wss_min/configs/v6_followup_20260909/M2_a5_independent_k3_s1234.json"
ANCHOR_RUN = RUNS / "v6_followup_20260909/M2_a5_independent_k3_s1234"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def state_path(phase="base"):
    return EXP / ("queue_status.json" if phase == "base" else "combination_queue_status.json")


@contextmanager
def locked_state(phase="base"):
    path = state_path(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else None
        try:
            yield state
            if state is not None:
                save_json(path, state)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def manifest_path(phase="base"):
    return MATRIX if phase == "base" else COMBINATIONS


def fingerprints(phase="base"):
    paths = list((ROOT / "training_wss_min").glob("*.py"))
    names = ("m2_optimization_common.py", "prepare_m2_optimization.py",
             "run_m2_optimization.py", "preflight_m2_optimization.py", "smoke_m2_optimization.py",
             "prepare_v6_singleframe_matrix.py")
    paths += [ROOT / "training_wss_min/tools" / name for name in names]
    manifest = manifest_path(phase)
    paths.append(manifest)
    for arm in json.loads(manifest.read_text())["arms"]:
        paths.append(CONFIGS / arm["config"])
    paths += [ANCHOR_CONFIG]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def configuration_identity(config):
    value = {key: val for key, val in config.items() if key not in {"name", "notes"}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
