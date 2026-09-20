#!/usr/bin/env python3
"""Read-only diagnosis: why VF6→WSS lags X5 direct WSS. No training or inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
EXP = ROOT / "training_wss_min/experiments/vf6_velocity_to_wss_20260916"
OUT = EXP / "analysis_20260916"
VF6_PRED = ROOT / "training_wss_min/runs/wss_local_wave2_20260912/VF6_s1234/eval/ckpt_best/predictions/test"
VOL_ROOT = ROOT / "data_wss_v5/views/wss_min_view_v1"
CSV_PATH = EXP / "per_case_comparison.csv"

BINS = [
    ("0-0.5mm", 0.0, 0.5),
    ("0.5-1mm", 0.5, 1.0),
    ("1-2mm", 1.0, 2.0),
    ("2-4mm", 2.0, 4.0),
    (">4mm", 4.0, np.inf),
]
OPERATOR_MAX_MM = 2.5


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = float(np.sum((true - true.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((pred - true) ** 2) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_csv() -> list[dict]:
    rows = []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            out = {"canonical_id": row["canonical_id"], "cohort": row["cohort"]}
            for key, value in row.items():
                if key in out:
                    continue
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
            rows.append(out)
    return rows


def case_shells(cid: str) -> dict:
    pred_path = VF6_PRED / cid / "predictions.npz"
    vol_path = VOL_ROOT / cid / "volume.npz"
    with np.load(pred_path, allow_pickle=False) as z:
        query_idx = np.asarray(z["query_idx"], dtype=np.int64)
        n_wall = int(np.asarray(z["n_wall"]))
        true = np.asarray(z["true_raw"], dtype=np.float64)
        pred = np.asarray(z["pred_raw"], dtype=np.float64)
        kind = np.asarray(z["point_kind"])
    assert np.all(kind == 1)
    vol_idx = query_idx - n_wall
    assert vol_idx.min() >= 0
    with np.load(vol_path, allow_pickle=False) as z:
        dist = np.asarray(z["vol_dist_to_wall_mm"], dtype=np.float64)[vol_idx]
    true_speed = np.linalg.norm(true, axis=1)
    pred_speed = np.linalg.norm(pred, axis=1)
    vec_err = np.linalg.norm(pred - true, axis=1)
    out = {
        "n": int(len(dist)),
        "overall_speed_r2": r2(true_speed, pred_speed),
        "overall_vec_rmse": float(np.sqrt(np.mean(vec_err ** 2))),
        "overall_speed_mae": float(np.mean(np.abs(pred_speed - true_speed))),
        "bins": {},
    }
    for name, lo, hi in BINS:
        mask = (dist >= lo) & (dist < hi)
        n = int(mask.sum())
        rec = {"n": n, "fraction": float(n / len(dist))}
        if n >= 32:
            rec["speed_r2"] = r2(true_speed[mask], pred_speed[mask])
            rec["speed_mae"] = float(np.mean(np.abs(pred_speed[mask] - true_speed[mask])))
            rec["vec_rmse"] = float(np.sqrt(np.mean(vec_err[mask] ** 2)))
            rec["true_speed_mean"] = float(true_speed[mask].mean())
            rec["pred_speed_mean"] = float(pred_speed[mask].mean())
        out["bins"][name] = rec
    near = dist < OPERATOR_MAX_MM
    far = ~near
    out["operator_shell"] = {
        "n": int(near.sum()),
        "fraction": float(near.mean()),
        "speed_r2": r2(true_speed[near], pred_speed[near]) if near.sum() >= 32 else float("nan"),
        "speed_mae": float(np.mean(np.abs(pred_speed[near] - true_speed[near]))) if near.any() else float("nan"),
        "vec_rmse": float(np.sqrt(np.mean(vec_err[near] ** 2))) if near.any() else float("nan"),
        "true_speed_mean": float(true_speed[near].mean()) if near.any() else float("nan"),
    }
    out["core"] = {
        "n": int(far.sum()),
        "fraction": float(far.mean()),
        "speed_r2": r2(true_speed[far], pred_speed[far]) if far.sum() >= 32 else float("nan"),
        "speed_mae": float(np.mean(np.abs(pred_speed[far] - true_speed[far]))) if far.any() else float("nan"),
        "vec_rmse": float(np.sqrt(np.mean(vec_err[far] ** 2))) if far.any() else float("nan"),
        "true_speed_mean": float(true_speed[far].mean()) if far.any() else float("nan"),
    }
    sse = (pred_speed - true_speed) ** 2
    out["sse_share_operator_shell"] = float(sse[near].sum() / sse.sum()) if sse.sum() > 0 else float("nan")
    return out


def summarize_bins(per_case: dict) -> dict:
    summary = {}
    for name, _, _ in BINS + [("operator_shell", 0.0, OPERATOR_MAX_MM), ("core", OPERATOR_MAX_MM, np.inf)]:
        key = "bins" if name not in {"operator_shell", "core"} else None
        r2s, fracs, maes, rmses = [], [], [], []
        n_tot = 0
        for rec in per_case.values():
            block = rec["operator_shell"] if name == "operator_shell" else rec["core"] if name == "core" else rec["bins"][name]
            n_tot += block["n"]
            if "speed_r2" in block and np.isfinite(block["speed_r2"]):
                r2s.append(block["speed_r2"])
                fracs.append(block["fraction"])
                maes.append(block["speed_mae"])
                rmses.append(block["vec_rmse"])
        summary[name] = {
            "n_points": n_tot,
            "n_cases_valid": len(r2s),
            "speed_r2_casemean": float(np.mean(r2s)) if r2s else float("nan"),
            "speed_r2_casemed": float(np.median(r2s)) if r2s else float("nan"),
            "fraction_casemean": float(np.mean(fracs)) if fracs else float("nan"),
            "speed_mae_casemean": float(np.mean(maes)) if maes else float("nan"),
            "vec_rmse_casemean": float(np.mean(rmses)) if rmses else float("nan"),
        }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_csv()
    speed = np.array([r["speed_r2_mean"] for r in rows])
    wss = np.array([r["legacy_wss_r2_mean"] for r in rows])
    x5 = np.array([r["x5_direct_r2_mean"] for r in rows])
    oracle = np.array([r["legacy_oracle_r2"] for r in rows])
    physics = np.array([
        np.mean([r["legacy_physics_r2_s1234"], r["legacy_physics_r2_s7"], r["legacy_physics_r2_s2025"]])
        for r in rows
    ])
    gap = x5 - wss
    cal_minus_phys = wss - physics
    cohort_stats = {}
    for cohort in ("AAA", "AG", "ILO"):
        m = np.array([r["cohort"] == cohort for r in rows])
        cohort_stats[cohort] = {
            "n": int(m.sum()),
            "speed_r2_mean": float(speed[m].mean()),
            "vf6_wss_r2_mean": float(wss[m].mean()),
            "x5_r2_mean": float(x5[m].mean()),
            "gap_x5_minus_vf6wss": float(gap[m].mean()),
            "drop_speed_minus_vf6wss": float((speed[m] - wss[m]).mean()),
        }
    ranked = sorted(rows, key=lambda r: r["legacy_wss_r2_mean"])
    pairing = {
        "n_cases": len(rows),
        "x5_better_than_vf6wss": int(np.sum(x5 > wss)),
        "oracle_better_than_vf6wss": int(np.sum(oracle > wss)),
        "spearman_speed_vs_vf6wss": spearman(speed, wss),
        "spearman_x5_vs_vf6wss": spearman(x5, wss),
        "spearman_speed_vs_x5": spearman(speed, x5),
        "gap_x5_minus_vf6wss": {
            "mean": float(gap.mean()),
            "median": float(np.median(gap)),
            "min": float(gap.min()),
            "max": float(gap.max()),
        },
        "calibrated_minus_physics_casemean": float(cal_minus_phys.mean()),
        "calibrated_better_count": int(np.sum(wss > physics)),
        "cohort": cohort_stats,
        "worst5": [
            {
                "id": r["canonical_id"],
                "cohort": r["cohort"],
                "speed": r["speed_r2_mean"],
                "vf6_wss": r["legacy_wss_r2_mean"],
                "x5": r["x5_direct_r2_mean"],
                "oracle": r["legacy_oracle_r2"],
            }
            for r in ranked[:5]
        ],
        "best5": [
            {
                "id": r["canonical_id"],
                "cohort": r["cohort"],
                "speed": r["speed_r2_mean"],
                "vf6_wss": r["legacy_wss_r2_mean"],
                "x5": r["x5_direct_r2_mean"],
            }
            for r in ranked[-5:][::-1]
        ],
        "decoupled": [
            {
                "id": r["canonical_id"],
                "cohort": r["cohort"],
                "speed": r["speed_r2_mean"],
                "vf6_wss": r["legacy_wss_r2_mean"],
                "x5": r["x5_direct_r2_mean"],
                "drop": r["speed_r2_mean"] - r["legacy_wss_r2_mean"],
            }
            for r in sorted(rows, key=lambda r: r["speed_r2_mean"] - r["legacy_wss_r2_mean"], reverse=True)[:5]
        ],
    }

    shells = {}
    for i, row in enumerate(rows, 1):
        cid = row["canonical_id"]
        print(f"[{i:02d}/{len(rows)}] {cid}", flush=True)
        rec = case_shells(cid)
        rec["legacy_wss_r2_mean"] = row["legacy_wss_r2_mean"]
        rec["x5_direct_r2_mean"] = row["x5_direct_r2_mean"]
        rec["speed_r2_mean_csv"] = row["speed_r2_mean"]
        rec["cohort"] = row["cohort"]
        shells[cid] = rec

    near_r2 = np.array([v["operator_shell"]["speed_r2"] for v in shells.values()])
    far_r2 = np.array([v["core"]["speed_r2"] for v in shells.values()])
    wss_arr = np.array([v["legacy_wss_r2_mean"] for v in shells.values()])
    near_frac = np.array([v["operator_shell"]["fraction"] for v in shells.values()])
    shell_summary = summarize_bins(shells)
    shell_summary["spearman_operator_shell_speed_vs_wss"] = spearman(near_r2, wss_arr)
    shell_summary["spearman_core_speed_vs_wss"] = spearman(far_r2, wss_arr)
    shell_summary["spearman_near_fraction_vs_wss"] = spearman(near_frac, wss_arr)
    shell_summary["near_minus_core_speed_r2_casemean"] = float(np.nanmean(near_r2 - far_r2))

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    payload = {
        "contract": {
            "split": "V5 train138/test34",
            "frame": "peak 1162",
            "vf6_wss": "VELWSS2 legacy calibrated, three-seed mean per-case R2",
            "direct_wss": "X5 three-seed mean per-case R2",
            "velocity_shells": "VF6_s1234 ckpt_best same-point interior predictions",
            "operator_shell_mm": OPERATOR_MAX_MM,
        },
        "pairing": pairing,
        "shell_summary": shell_summary,
        "per_case_shells": shells,
    }
    payload = clean(payload)
    (OUT / "vel_vs_direct_wss_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"pairing": payload["pairing"], "shell_summary": payload["shell_summary"]}, indent=2))


if __name__ == "__main__":
    main()
