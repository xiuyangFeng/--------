#!/usr/bin/env python3
"""Select X5D_v51 best / median / worst from finished five-seed metrics. No inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "training_wss_min").is_dir())
RUNS = ROOT / "training_wss_min/runs/wss_v51_wave1_20260916"
SEEDS = (1234, 7, 2025, 11, 2026)
ROLES = ("best", "median", "worst")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sources = []
    payloads = []
    for seed in SEEDS:
        path = RUNS / f"X5D_v51_s{seed}" / "eval/ckpt_best/metrics.json"
        sources.append({"seed": seed, "path": str(path), "sha256": sha(path)})
        payloads.append(json.loads(path.read_text())["test"])
    ids = sorted(payloads[0]["per_case"])
    assert len(ids) == 34
    for payload in payloads:
        assert sorted(payload["per_case"]) == ids
    cases = []
    for cid in ids:
        rows = [payload["per_case"][cid]["overall"] for payload in payloads]
        values = [float(row["r2"]) for row in rows]
        cases.append({
            "case_id": cid,
            "r2": float(np.mean(values)),
            "r2_sd": float(np.std(values, ddof=1)),
            "seed_values": values,
            "mae": float(np.mean([row["mae"] for row in rows])),
            "n": int(rows[0]["n"]),
            "s1234_r2": values[0],
        })
    ordered = sorted(cases, key=lambda item: (item["r2"], item["case_id"]))
    values = np.array([item["r2"] for item in cases], dtype=float)
    median = float(np.median(values))
    nearest = sorted(cases, key=lambda item: (abs(item["r2"] - median), item["case_id"]))
    selected = {
        "best": ordered[-1],
        "median": dict(nearest[0], statistical_median=median, rule=(
            "closest actual case to the five-seed mean R² median; "
            "near-tie broken by smaller distance then case_id"
        )),
        "worst": ordered[0],
    }
    selected["median"]["near_tie"] = [
        {
            "case_id": item["case_id"],
            "r2": item["r2"],
            "abs_distance": abs(item["r2"] - median),
            "r2_sd": item["r2_sd"],
        }
        for item in nearest[:2]
    ]
    payload = {
        "model": "X5D_v51",
        "protocol": (
            "v5.1 数据更新后重训完成的密度增广 X5D_v51；五 seed 1234/7/2025/11/2026，"
            "ckpt_best，物理空间逐例 R² 均值。不是正在训练的纵向几何 X5D_long，"
            "也不是 X5Dcap/dual/noise。"
        ),
        "n_cases": 34,
        "seeds": list(SEEDS),
        "stats": {
            "case_mean": float(values.mean()),
            "case_median": median,
            "case_p10": float(np.percentile(values, 10)),
            "negative_count": int((values < 0).sum()),
        },
        "sources": sources,
        "source_field": "test.per_case[case_id].overall.r2",
        **{role: selected[role] for role in ROLES},
        "all_cases": cases,
    }
    path = OUT / "X5D_v51" / "selection.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "wrote": str(path),
        "best": selected["best"]["case_id"],
        "median": selected["median"]["case_id"],
        "worst": selected["worst"]["case_id"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
