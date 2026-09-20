"""Frozen names and provenance helpers for the 26-run volume experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
NAME = "volume_attention_20260910"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
MATRIX = CONFIGS / "matrix.json"
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
OLD = {
    "P": "v5_rerun_20260906/outputs/r5p_pressure_mixed_qad_s1234",
    "V": "v5_rerun_20260906/outputs/r5v_velocity_qad_s1234",
}


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprints():
    paths = list((ROOT / "training_wss_min").glob("*.py"))
    paths += list(CONFIGS.rglob("*.json"))
    paths += list((ROOT / "training_wss_min/tools").glob("*volume_attention*.py"))
    paths += list((ROOT / "training_wss_min/tests").glob("test_volume*.py"))
    paths += [ROOT / "training_wss_min/tools/prepare_v6_singleframe_matrix.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def manifest():
    return json.loads(MATRIX.read_text())


def reference_dir(prefix):
    return EXP / "reference_runs" / f"R5{prefix}"


def task_metrics_path(run, checkpoint, task, joint=False):
    base = Path(run) / "eval" / f"ckpt_{checkpoint}"
    return base / task / "metrics.json" if joint else base / "metrics.json"
