#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 run 的 val 指标判读 Gate-1（相对 control）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_val(run_dir: Path) -> dict:
    mpath = run_dir / "eval" / "metrics.json"
    if mpath.is_file():
        m = json.loads(mpath.read_text())
        val = m.get("val", m)
        return {
            "r2_field": val["field"]["r2"],
            "r2_casemean": val["aggregate"]["r2_casemean"],
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
        "mae": best.get("val_mae_field"),
        "top10": best.get("val_top10_pred_true_ratio"),
        "max_ratio": None,
        "iou": best.get("val_top10_iou"),
        "score": best.get("val_selection_score"),
        "epoch": best.get("epoch"),
    }


def gate1(ctrl: dict, cand: dict) -> dict:
    dr2f = cand["r2_field"] - ctrl["r2_field"]
    dr2c = cand["r2_casemean"] - ctrl["r2_casemean"]
    dtop = (cand["top10"] - ctrl["top10"]) if cand["top10"] is not None and ctrl["top10"] is not None else None
    diou = (cand["iou"] - ctrl["iou"]) if cand.get("iou") is not None and ctrl.get("iou") is not None else None
    dmae = (cand["mae"] - ctrl["mae"]) / max(abs(ctrl["mae"]), 1e-6)
    overall = dr2f >= 0.02 and dr2c >= 0
    tail = False
    if dtop is not None and dtop >= 0.10 and dr2f >= -0.01 and dmae <= 0.03:
        tail = True
    if diou is not None and diou >= 0.05 and dr2f >= -0.01 and dmae <= 0.03:
        tail = True
    safe = True
    if cand.get("max_ratio") is not None and cand["max_ratio"] > 1.5:
        safe = False
    return {
        "delta": {"r2_field": dr2f, "r2_casemean": dr2c, "top10": dtop, "iou": diou, "mae_rel": dmae},
        "overall_path": overall,
        "tail_path": tail,
        "safe": safe,
        "go": bool((overall or tail) and safe),
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
