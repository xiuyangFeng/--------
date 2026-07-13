#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 run 的 val 指标判读 Gate-1（相对 control）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INDIFFERENCE_BAND = 0.02
HIGH_WSS_MAX_DROP = 0.05


def load_val(run_dir: Path) -> dict:
    mpath = run_dir / "eval" / "metrics.json"
    if mpath.is_file():
        m = json.loads(mpath.read_text())
        val = m.get("val", m)
        return {
            "r2_field": val["field"]["r2"],
            "r2_casemean": val["aggregate"]["r2_casemean"],
            "r2_field_casebalanced": (val.get("field_casebalanced") or {}).get("r2"),
            "mae": val["field"]["mae"],
            "top10": val["calibration"].get("top10_pred_true_ratio"),
            "max_ratio": val["calibration"].get("max_pred_true_ratio"),
            "iou": (val.get("hotspot") or {}).get("top10_iou"),
        }
    hist = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines() if l.strip()]
    best = max((h for h in hist if "val_selection_score" in h),
               key=lambda h: h["val_selection_score"], default=None)
    if best is None:
        raise SystemExit(f"no val metrics in {run_dir}")
    return {
        "r2_field": best.get("val_r2_field"),
        "r2_casemean": best.get("val_r2_casemean"),
        "r2_field_casebalanced": best.get("val_r2_field_casebalanced"),
        "mae": best.get("val_mae_field"),
        "top10": best.get("val_top10_pred_true_ratio"),
        "max_ratio": None,
        "iou": best.get("val_top10_iou"),
        "score": best.get("val_selection_score"),
        "epoch": best.get("epoch"),
    }


def gate1(ctrl: dict, cand: dict, indifference_band: float = INDIFFERENCE_BAND,
          high_wss_max_drop: float = HIGH_WSS_MAX_DROP) -> dict:
    dr2f = cand["r2_field"] - ctrl["r2_field"]
    dr2c = cand["r2_casemean"] - ctrl["r2_casemean"]
    dtop = (cand["top10"] - ctrl["top10"]) if cand["top10"] is not None and ctrl["top10"] is not None else None
    diou = (cand["iou"] - ctrl["iou"]) if cand.get("iou") is not None and ctrl.get("iou") is not None else None
    dmae = (cand["mae"] - ctrl["mae"]) / max(abs(ctrl["mae"]), 1e-6)
    common_improvement = dr2f > indifference_band and dr2c > indifference_band
    safe = True
    guard_reasons = []
    if dtop is not None and dtop < -high_wss_max_drop:
        safe = False
        guard_reasons.append("top10_ratio_degraded")
    if diou is not None and diou < -high_wss_max_drop:
        safe = False
        guard_reasons.append("top10_iou_degraded")
    if cand.get("max_ratio") is not None and cand["max_ratio"] > 1.5:
        safe = False
        guard_reasons.append("max_ratio_exploded")
    if common_improvement and safe:
        verdict = "GO"
    elif abs(dr2f) <= indifference_band and abs(dr2c) <= indifference_band:
        verdict = "INDIFFERENT"
    elif (dr2f > indifference_band and dr2c < -indifference_band) or (
        dr2c > indifference_band and dr2f < -indifference_band
    ):
        verdict = "TRADE_OFF"
    else:
        verdict = "NO_GO"
    return {
        "delta": {"r2_field": dr2f, "r2_casemean": dr2c, "top10": dtop, "iou": diou, "mae_rel": dmae},
        "indifference_band": indifference_band,
        "high_wss_max_drop": high_wss_max_drop,
        "common_improvement": common_improvement,
        "safe": safe,
        "guard_reasons": guard_reasons,
        "verdict": verdict,
        "go": verdict == "GO",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    ctrl = load_val(Path(args.control))
    cand = load_val(Path(args.candidate))
    g = gate1(ctrl, cand)
    print(json.dumps({"control": ctrl, "candidate": cand, **g}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
