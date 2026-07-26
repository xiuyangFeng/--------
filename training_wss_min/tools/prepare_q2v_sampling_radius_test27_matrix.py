#!/usr/bin/env python3
"""Prepare the explicit Q2V/test27 exploratory sampling-radius matrix.

The source checkpoint and test partition are intentionally historical anchors
requested by the user.  The manifest therefore labels every result as
exploratory: it must not be promoted as an independent confirmation result.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "training_wss_min/configs/pointnetpp_q2v_sampling_radius_test27_20260718"
PREFLIGHT = REPO / "training_wss_min/preflight/q2v_sampling_radius_test27_prepared.json"
Q2V = REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep.json"
Q1V = REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json"
Q2V_BEST = REPO / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep/ckpt_best.pt"


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


def run_name(stem: str) -> str:
    return f"pointnetpp_q2v_sampling_radius_test27/outputs/{stem}"


def radius_variant(stem: str, radii: list[float], label: str) -> dict:
    cfg = copy.deepcopy(load(Q2V))
    cfg["name"] = run_name(stem)
    cfg["notes"] = (
        f"Exploratory Q2V-10477/test27 anchor; only SA radii change to {radii} ({label}); "
        "vertex-random5000 SEP and FPS 500-125-32 centres remain frozen. "
        "Not an independent confirmation result."
    )
    cfg["model"]["sa_radius"] = radii
    return cfg


def multistart_variant() -> dict:
    # Deterministic FPS and multi-start FPS intentionally remain a SAME route:
    # the project validation rejects deterministic FPS under SEP because it can
    # collapse support/query to the same cached subset.  Q1V is the matched
    # random5000-SAME control, so this is a clean sampling-only comparison.
    cfg = copy.deepcopy(load(Q1V))
    cfg["name"] = run_name("q1v_fpsmultistart5000_same")
    cfg["notes"] = (
        "Exploratory Q2V/Q1V test27 anchor; Q1V matched SAME control with only vertex-random5000 "
        "sampling changed to deterministic epoch-resampled fps_multistart5000 (pool=8). "
        "FPS 500-125-32 centres remain frozen. Not an independent confirmation result."
    )
    for key in ("sampling", "support_sampling", "query_sampling"):
        cfg["data"][key] = "fps_multistart"
    cfg["data"]["fps_pool_size"] = 8
    return cfg


def same_finetune_variant() -> dict:
    cfg = copy.deepcopy(load(Q1V))
    cfg["name"] = run_name("q2v_to_q1v_same_finetune")
    cfg["notes"] = (
        "Exploratory Q2V-10477/test27 anchor. Warm-start Q2V ckpt_best weights, then train "
        "under the Q1V-10476 random5000 SAME + FPS 500-125-32-centre protocol for its full "
        "400-epoch budget. Optimizer and LR schedule reset; not a resume and not an independent "
        "confirmation result."
    )
    cfg["train"]["init_checkpoint_path"] = str(Q2V_BEST)
    cfg["train"]["init_checkpoint_strict"] = True
    return cfg


def main() -> None:
    if not Q2V_BEST.is_file():
        raise FileNotFoundError(f"Q2V warm-start checkpoint is missing: {Q2V_BEST}")
    variants = (
        ("q2v_radius_r80_sep", radius_variant("q2v_radius_r80_sep", [0.04, 0.08, 0.16], "80%"),
         "Q2V radius-only: 80% of the original radius"),
        ("q2v_radius_r60_sep", radius_variant("q2v_radius_r60_sep", [0.03, 0.06, 0.12], "60%"),
         "Q2V radius-only: 60% of the original radius"),
        ("q1v_fpsmultistart5000_same", multistart_variant(),
         "Q1V sampling-only: vertex random -> fps_multistart5000; SAME retained"),
        ("q2v_to_q1v_same_finetune", same_finetune_variant(),
         "Warm-start Q2V best -> exact Q1V SAME protocol; optimizer reset"),
    )
    rows = []
    for stem, cfg, change in variants:
        path = CONFIGS / f"{stem}.json"
        write_once(path, cfg)
        rows.append({
            "experiment_id": stem, "config": str(path.resolve()), "sha256": sha(path),
            "expected_run_dir": str((REPO / "training_wss_min/runs" / cfg["name"]).resolve()),
            "anchor": "Q2V-10477 historical test27", "comparison_scope": "exploratory",
            "single_change_or_declared_transition": change,
            "source_checkpoint": str(Q2V_BEST.resolve()) if "init_checkpoint_path" in cfg["train"] else None,
            "source_checkpoint_sha256": sha(Q2V_BEST) if "init_checkpoint_path" in cfg["train"] else None,
        })
    existing_manifest = load(PREFLIGHT) if PREFLIGHT.exists() else None
    manifest = {
        "schema_version": 1,
        "created_at": existing_manifest.get("created_at") if existing_manifest else now(),
        "status": "prepared",
        "purpose": "Q2V-10477 anchored exploratory test27 sampling/radius matrix",
        "test27_policy": "historical anchor requested by user; do not present as independent confirmation",
        "anchors": {"q2v_config": str(Q2V.resolve()), "q2v_config_sha256": sha(Q2V),
                    "q1v_config": str(Q1V.resolve()), "q1v_config_sha256": sha(Q1V),
                    "q2v_best_checkpoint": str(Q2V_BEST.resolve()), "q2v_best_checkpoint_sha256": sha(Q2V_BEST)},
        "configs": rows,
    }
    write_once(PREFLIGHT, manifest)
    print(json.dumps({"status": "prepared", "configs": len(rows), "manifest": str(PREFLIGHT)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
