#!/usr/bin/env python3
"""Materialize the frozen 17-run SA grouping/support matrix (seed 1234 only)."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
Q1V = REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json"
Q2V = REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep.json"
OUT_DIR = REPO / "training_wss_min/configs/pointnetpp_sa_grouping_single_seed_20260720"
MANIFEST = REPO / "training_wss_min/preflight/sa_grouping_single_seed_prepared.json"
CACHE = REPO / "training_wss_min/preflight/fps_support_cache_v4"
RUN_PREFIX = "pointnetpp_sa_grouping_single_seed/outputs"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make(base: dict, experiment_id: str, family: str, note: str, *,
         nsample=(16, 16, 16), width=32, grouping=("ball", "ball", "ball"),
         support="random") -> tuple[dict, dict]:
    cfg = copy.deepcopy(base)
    cfg["name"] = f"{RUN_PREFIX}/{experiment_id}"
    cfg["notes"] = (
        f"2026-07-20 SA grouping matrix; one seed only (1234); exact split 106/0/27. "
        f"{note}"
    )
    cfg["train"]["seed"] = 1234
    cfg["model"]["width"] = int(width)
    cfg["model"]["sa_nsample"] = list(nsample)
    cfg["model"]["sa_grouping"] = list(grouping)
    cfg["model"]["sa_adaptive_candidate_centers"] = 16
    cfg["model"]["sa_overlap_cap"] = 1.0 / 3.0
    cfg["data"]["wall_n_points"] = 5000
    cfg["data"]["sampling"] = support
    cfg["data"]["support_n_points"] = 5000
    cfg["data"]["support_sampling"] = support
    if cfg["data"]["query_mode"] == "same":
        cfg["data"]["query_n_points"] = 5000
        cfg["data"]["query_sampling"] = support
    else:
        cfg["data"]["query_n_points"] = 5000
        cfg["data"]["query_sampling"] = "random"
    cfg["data"]["fps_pool_size"] = 8
    if support in {"fps", "fps_multistart"}:
        cfg["data"]["fps_cache_dir"] = str(CACHE)
        cfg["data"]["fps_cache_required"] = True
    else:
        cfg["data"].pop("fps_cache_dir", None)
        cfg["data"].pop("fps_cache_required", None)
    row = {
        "experiment_id": experiment_id,
        "family": family,
        "anchor": "Q1V/SAME" if cfg["data"]["query_mode"] == "same" else "Q2V/SEP",
        "support_sampling": support,
        "requires_fps_cache": support in {"fps", "fps_multistart"},
        "seed": 1234,
        "sa_nsample": list(nsample),
        "width": int(width),
        "sa_grouping": list(grouping),
    }
    return cfg, row


def main() -> None:
    q1v, q2v = load(Q1V), load(Q2V)
    specs: list[tuple[dict, dict]] = []
    add = lambda *args, **kwargs: specs.append(make(*args, **kwargs))

    # nsample x width: N0 (16 x width32) is the reused Q1V anchor.
    add(q1v, "q1v_n32_w32", "capacity", "N1: only nsample 16->32.", nsample=(32, 32, 32))
    add(q1v, "q1v_n16_w64", "capacity", "N2: only width 32->64.", width=64)
    add(q1v, "q1v_n32_w64", "capacity", "N3: nsample32 x width64.", nsample=(32, 32, 32), width=64)

    add(q1v, "q1v_sa1_ball32", "grouping", "G1: SA1 ball cap32; SA2/SA3 remain ball16.", nsample=(32, 16, 16))
    add(q1v, "q1v_sa1_knn8", "grouping", "G2: raw KNN-8 negative control; coverage is diagnostic, not gated.", nsample=(8, 16, 16), grouping=("knn", "ball", "ball"))
    add(q1v, "q1v_sa1_knn10", "grouping", "G3: raw KNN-10 negative control; coverage is diagnostic, not gated.", nsample=(10, 16, 16), grouping=("knn", "ball", "ball"))
    add(q1v, "q1v_sa1_knn8_cover", "grouping", "G4: KNN-8 plus nearest-center repair; SA1 coverage hard-gated to 100%.", nsample=(8, 16, 16), grouping=("knn_cover", "ball", "ball"))
    add(q1v, "q1v_sa1_knn10_cover", "grouping", "G5: KNN-10 plus nearest-center repair; SA1 coverage hard-gated to 100%.", nsample=(10, 16, 16), grouping=("knn_cover", "ball", "ball"))
    add(q1v, "q1v_sa1_adaptive_cover", "grouping", "G6: adaptive coverage-first grouping; coverage=100%, pair overlap<=1/3 hard gates.", grouping=("adaptive_cover", "ball", "ball"))

    # Q1V/SAME support matrix. random/ball and fps-multistart/ball are reused anchors.
    add(q1v, "q1v_fixedfps5000_ball16", "support_q1v", "Fixed offline FPS5000 support, ball16 grouping.", support="fps")
    add(q1v, "q1v_fixedfps5000_adaptive", "support_q1v", "Fixed offline FPS5000 support, adaptive grouping.", grouping=("adaptive_cover", "ball", "ball"), support="fps")
    add(q1v, "q1v_fpsms5000_adaptive", "support_q1v", "Offline pool8 FPS-multistart5000 support, adaptive grouping.", grouping=("adaptive_cover", "ball", "ball"), support="fps_multistart")

    # Q2V/SEP support matrix: query always independent random5000.
    add(q2v, "q2v_random5000_adaptive", "support_q2v", "Random5000 support + independent random5000 query, adaptive grouping.", grouping=("adaptive_cover", "ball", "ball"))
    add(q2v, "q2v_fixedfps5000_ball16", "support_q2v", "Fixed offline FPS5000 support + independent random query, ball16.", support="fps")
    add(q2v, "q2v_fixedfps5000_adaptive", "support_q2v", "Fixed offline FPS5000 support + independent random query, adaptive grouping.", grouping=("adaptive_cover", "ball", "ball"), support="fps")
    add(q2v, "q2v_fpsms5000_ball16", "support_q2v", "Offline pool8 FPS-multistart support + independent random query, ball16.", support="fps_multistart")
    add(q2v, "q2v_fpsms5000_adaptive", "support_q2v", "Offline pool8 FPS-multistart support + independent random query, adaptive grouping.", grouping=("adaptive_cover", "ball", "ball"), support="fps_multistart")

    if len(specs) != 17:
        raise AssertionError(len(specs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg_dict, row in specs:
        path = OUT_DIR / f"{row['experiment_id']}.json"
        path.write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise FileExistsError(f"refusing existing run dir: {cfg.run_dir}")
        rows.append({**row, "config": str(path.resolve()), "sha256": sha(path), "run_dir": str(cfg.run_dir)})

    split = load(Path(q1v["data"]["split_path"]))
    counts = {p: len(split.get(f"{p}_cases", [])) for p in ("train", "val", "test")}
    if counts != {"train": 106, "val": 0, "test": 27}:
        raise RuntimeError(f"unexpected split counts: {counts}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "policy": {"seeds": [1234], "split_counts": counts, "new_jobs": 17,
                   "adaptive_gates": {"sa1_coverage": 1.0, "max_pair_overlap": 1.0 / 3.0}},
        "reused_controls": [
            {"id": "Q1V_N0_G0_random5000_ball16", "config": str(Q1V),
             "run_dir": str(ExpConfig.from_json(Q1V).run_dir)},
            {"id": "Q2V_random5000_ball16", "config": str(Q2V),
             "run_dir": str(ExpConfig.from_json(Q2V).run_dir)},
            {"id": "Q1V_fpsmultistart5000_ball16",
             "config": str(REPO / "training_wss_min/configs/pointnetpp_q2v_sampling_radius_test27_20260718/q1v_fpsmultistart5000_same.json"),
             "run_dir": str(REPO / "training_wss_min/runs/pointnetpp_q2v_sampling_radius_test27/outputs/q1v_fpsmultistart5000_same")},
        ],
        "configs": rows,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "configs": len(rows), "core": sum(not r["requires_fps_cache"] for r in rows),
                      "fps": sum(r["requires_fps_cache"] for r in rows), "manifest": str(MANIFEST)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
