#!/usr/bin/env python3
"""Prepare the exact-Q2V three-seed interpolate/QAD historical-test27 matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "training_wss_min/configs/pointnetpp_qad_q2v_test27_20260719"
PREPARED = REPO / "training_wss_min/preflight/pointnetpp_qad_q2v_test27_prepared.json"
ANCHOR = REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep.json"
ANCHOR_RUN = REPO / (
    "training_wss_min/runs/pointnetpp_v4/outputs/"
    "ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep"
)

ORDER = (
    ("r1_qad_q2v_n16_w32_seed1234", 1234, "qad_lite"),
    ("r0_interpolate_q2v_n16_w32_seed7", 7, "interpolate"),
    ("r1_qad_q2v_n16_w32_seed7", 7, "qad_lite"),
    ("r0_interpolate_q2v_n16_w32_seed2025", 2025, "interpolate"),
    ("r1_qad_q2v_n16_w32_seed2025", 2025, "qad_lite"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def write_generated(path: Path, payload: dict) -> None:
    text = stable_json(payload)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite non-identical protocol asset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def comparable(payload: dict) -> dict:
    out = copy.deepcopy(payload)
    out.pop("name", None)
    out.pop("notes", None)
    out["train"]["seed"] = 1234
    out["model"].setdefault("query_decoder", "interpolate")
    out["model"].setdefault("qad_hidden", 32)
    return out


def make_variant(anchor: dict, stem: str, seed: int, decoder: str) -> dict:
    payload = copy.deepcopy(anchor)
    payload["name"] = f"pointnetpp_qad_q2v_test27/outputs/{stem}"
    payload["notes"] = (
        "Exact Q2V-10477 historical-test27 protocol; only train seed and query decoder "
        f"vary. seed={seed}, decoder={decoder}. Historical comparison, not independent confirmation."
    )
    payload["train"]["seed"] = seed
    payload["model"]["query_decoder"] = decoder
    payload["model"]["qad_hidden"] = 32
    return payload


def main() -> None:
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    split = json.loads(Path(anchor["data"]["split_path"]).read_text(encoding="utf-8"))
    counts = tuple(len(split[f"{part}_cases"]) for part in ("train", "val", "test"))
    if counts != (106, 0, 27):
        raise RuntimeError(f"Q2V split drift: {counts}")
    model = anchor["model"]
    if model["sa_nsample"] != [16, 16, 16] or model["width"] != 32:
        raise RuntimeError("Q2V n16/w32 architecture drift")
    if anchor["data"]["query_mode"] != "independent":
        raise RuntimeError("Q2V SEP contract drift")
    if anchor["train"]["selection_rule"] != "train_loss" or anchor["train"]["epochs"] != 400:
        raise RuntimeError("Q2V training contract drift")
    for required in (
        ANCHOR_RUN / "ckpt_best.pt",
        ANCHOR_RUN / "eval/ckpt_best/metrics.json",
        Path(anchor["data"]["wss_stats_path"]),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows = []
    for task_id, (stem, seed, decoder) in enumerate(ORDER):
        path = CONFIG_ROOT / f"{stem}.json"
        payload = make_variant(anchor, stem, seed, decoder)
        reference = comparable(anchor)
        candidate = comparable(payload)
        reference["model"]["query_decoder"] = decoder
        if reference != candidate:
            raise RuntimeError(f"variant differs beyond decoder/seed/run metadata: {stem}")
        write_generated(path, payload)
        rows.append({
            "task_id": task_id,
            "experiment_id": stem,
            "config": str(path.resolve()),
            "sha256": sha(path),
            "seed": seed,
            "decoder": decoder,
            "expected_run_dir": str((REPO / "training_wss_min/runs/pointnetpp_qad_q2v_test27/outputs" / stem).resolve()),
        })

    existing = json.loads(PREPARED.read_text(encoding="utf-8")) if PREPARED.exists() else None
    manifest = {
        "schema_version": 1,
        "created_at": existing["created_at"] if existing else datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "purpose": "Exact-Q2V n16/w32 interpolate-vs-QAD three-seed historical-test27 comparison",
        "test27_policy": "user-authorized historical comparison; not independent confirmation",
        "protocol": {
            "split": "106/0/27",
            "sampling": "vertex random5000 / SEP; FPS centers 500/125/32",
            "model": "PointNet++ n16/w32; only decoder changes within seed",
            "training": "400 epochs; train-loss checkpoint selection",
            "metrics": "physical and normalized legacy-vertex test27; best and last",
        },
        "anchor": {
            "experiment_id": "Q2V-10477",
            "job_id": "10477",
            "config": str(ANCHOR.resolve()),
            "config_sha256": sha(ANCHOR),
            "run_dir": str(ANCHOR_RUN.resolve()),
            "best_metrics": str((ANCHOR_RUN / "eval/ckpt_best/metrics.json").resolve()),
            "best_metrics_sha256": sha(ANCHOR_RUN / "eval/ckpt_best/metrics.json"),
        },
        "reused_runs": ["Q2V-10477 / seed1234 / interpolate"],
        "configs": rows,
    }
    write_generated(PREPARED, manifest)
    print(json.dumps({"status": "prepared", "configs": len(rows), "manifest": str(PREPARED)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
