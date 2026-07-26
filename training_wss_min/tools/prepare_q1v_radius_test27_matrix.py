#!/usr/bin/env python3
"""Prepare the Q1V/SAME historical-test27 radius-only exploration matrix.

All arms retain the Q1V sampling, SA centers, training budget, and evaluation
contract.  The historical test27 has already informed prior decisions, so this
matrix is explicitly exploratory and cannot be used as independent confirmation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "training_wss_min/configs/pointnetpp_q1v_radius_test27_20260719"
PREFLIGHT = REPO / "training_wss_min/preflight/q1v_radius_test27_prepared.json"
Q1V = REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def write_once(path: Path, payload: dict) -> None:
    text = stable_json(payload)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a non-identical protocol asset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def radius_variant(stem: str, radii: list[float], scale: str) -> dict:
    cfg = copy.deepcopy(load(Q1V))
    cfg["name"] = f"pointnetpp_q1v_radius_test27/outputs/{stem}"
    cfg["notes"] = (
        f"Exploratory Q1V-10476/SAME historical-test27 anchor; only SA radii change "
        f"to {radii} ({scale} of Q1V). vertex-random5000 SAME and FPS 500-125-32 "
        "centres remain frozen. Not an independent confirmation result."
    )
    cfg["model"]["sa_radius"] = radii
    return cfg


def main() -> None:
    variants = (
        ("q1v_radius_r80_same", [0.04, 0.08, 0.16], "80%"),
        ("q1v_radius_r120_same", [0.06, 0.12, 0.24], "120%"),
        ("q1v_radius_r150_same", [0.075, 0.15, 0.30], "150%"),
    )
    rows = []
    for stem, radii, scale in variants:
        path = CONFIGS / f"{stem}.json"
        write_once(path, radius_variant(stem, radii, scale))
        rows.append({
            "experiment_id": stem,
            "config": str(path.resolve()),
            "sha256": sha(path),
            "expected_run_dir": str((REPO / "training_wss_min/runs/pointnetpp_q1v_radius_test27/outputs" / stem).resolve()),
            "anchor": "Q1V-10476 historical test27",
            "comparison_scope": "exploratory",
            "single_change": f"Q1V SAME radius-only: {scale} of 0.05/0.10/0.20 -> {radii}",
        })
    existing = load(PREFLIGHT) if PREFLIGHT.exists() else None
    manifest = {
        "schema_version": 1,
        "created_at": existing.get("created_at") if existing else now(),
        "status": "prepared",
        "purpose": "Q1V-10476 SAME anchored exploratory historical-test27 radius matrix",
        "test27_policy": "user-authorized historical anchor; not independent confirmation",
        "anchor": {"q1v_config": str(Q1V.resolve()), "q1v_config_sha256": sha(Q1V)},
        "configs": rows,
    }
    write_once(PREFLIGHT, manifest)
    print(json.dumps({"status": "prepared", "configs": len(rows), "manifest": str(PREFLIGHT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
