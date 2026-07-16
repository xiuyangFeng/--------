#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AG 正式纳入病例在 STL 关键点 v4 坐标架下的只读回归审计。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline_wss_min import config as C
from pipeline_wss_min.new_cohorts.audit_frame import audit_case


OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / "registration_v4_ag_regression"


def main():
    labels = C.split_case_labels(C.DEFAULT_SPLIT_NAME, ("train", "val", "test"))
    rows = []
    for i, label in enumerate(labels, 1):
        subset, case_name = label.split("/", 1)
        unit_id = f"AG/{label}"
        try:
            row = audit_case(unit_id, f"AG/{subset}", case_name)
        except Exception as exc:  # noqa: BLE001
            row = {
                "unit_id": unit_id, "status": "error", "frame_pass": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(f"[{i:2d}/{len(labels)}] {str(row.get('frame_pass', False)):5s} "
              f"{unit_id} {row.get('reason', '')}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ag_landmark_frame_v4_audit.csv", index=False)
    summary = {
        "split": C.DEFAULT_SPLIT_NAME,
        "n_total": len(df),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_frame_pass": int((df["frame_pass"] == True).sum()),
        "frame_fail_units": df[df["frame_pass"] != True]["unit_id"].tolist(),
        "centerline_translation_repaired_units": (
            df[df.get("centerline_translation_applied") == True]["unit_id"].tolist()
            if "centerline_translation_applied" in df else []
        ),
    }
    (OUT_DIR / "ag_landmark_frame_v4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
