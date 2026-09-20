#!/usr/bin/env python3
"""Add the VELWSS2 (VF6→WSS) R² distribution without redrawing the 2026-09-14 deck."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from matplotlib import font_manager
import matplotlib
matplotlib.use("Agg")
for _font in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"):
    _path = Path("/usr/share/fonts/opentype/noto") / _font
    if _path.exists():
        font_manager.fontManager.addfont(str(_path))

import numpy as np

from build_distributions import DIST, SOURCE_AUDIT, plot, read, representatives

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
OUT = Path(__file__).resolve().parent
EXP = ROOT / "training_wss_min/experiments/vf6_velocity_to_wss_20260916"
METRICS = EXP / "metrics/legacy.json"
PARTS = ("s1234_calibrated", "s7_calibrated", "s2025_calibrated")
KEY = "B_VF6_wss"


def build_record() -> dict:
    data = read(METRICS)
    payloads = [data[part] for part in PARTS]
    ids = sorted(payloads[0]["per_case"])
    assert len(ids) == 34
    for payload in payloads:
        assert sorted(payload["per_case"]) == ids
        vals = [payload["per_case"][k]["overall"]["r2"] for k in ids]
        assert np.all(np.isfinite(vals))
        assert abs(np.mean(vals) - payload["aggregate"]["r2_casemean"]) < 1e-12
    cases = []
    for case_id in ids:
        rows = [payload["per_case"][case_id]["overall"] for payload in payloads]
        vals = [float(row["r2"]) for row in rows]
        cases.append({
            "case_id": case_id,
            "r2": float(np.mean(vals)),
            "r2_sd": float(np.std(vals, ddof=1)),
            "seed_values": vals,
            "mae": float(np.mean([row["mae"] for row in rows])),
            "nmae": float(np.mean([row["nmae_range"] for row in rows])),
            "nmae_definition": "MAE / within-case truth range",
            "point_count": int(rows[0]["n"]),
        })
    cb = [payload["field_casebalanced"]["r2"] for payload in payloads]
    rec = dict(
        key=KEY,
        title="方向2｜VF6 预测速度 → 冻结 V3 派生 WSS",
        subtitle="VELWSS2 · test34 · 峰值1162 · VF6 best · 每例三个 seed 的预测R²均值；横须=seed间样本SD（非置信区间）",
        group="B 当前数据驱动体场",
        note=(
            "冻结 Profile-Secant V3 + 病例内校准；主口径 legacy 半径，与 VELWSS1 同。"
            "图中是病例分数均值，既非集成预测R²，也非总体R²_cb。不是直接 WSS 回归。"
        ),
        source_paths=[str(METRICS)],
        source_field="s{1234,7,2025}_calibrated.per_case[case_id].overall.r2",
        official_field_cb=float(np.mean(cb)),
        official_field_cb_sd=float(np.std(cb, ddof=1)),
        cases=cases,
        replicate_count=3,
        target="WSS · Pa",
    )
    DIST.append(rec)
    return rec


def median_case(cases: list[dict]) -> dict:
    median = float(np.median([c["r2"] for c in cases]))
    return min(cases, key=lambda c: (abs(c["r2"] - median), c["case_id"]))


def merge_outputs(rec: dict) -> None:
    ordered, reps, _ = representatives(rec["cases"])
    median = median_case(rec["cases"])
    selection = {
        "metric": "physical per-case prediction R2 (not fitted-line R2)",
        "radius_source": "legacy",
        "operator": "frozen Profile-Secant V3 + within-case calibration",
        "replicate_policy": "one patient per dot; arithmetic mean of VF6 seed 1234/7/2025 derived-WSS R2; sample SD whiskers; not ensemble predictions",
        "visualization_seed": 1234,
        "best": {"case_id": reps["best"]["case_id"], "r2": reps["best"]["r2"], "r2_sd": reps["best"]["r2_sd"]},
        "worst": {"case_id": reps["worst"]["case_id"], "r2": reps["worst"]["r2"], "r2_sd": reps["worst"]["r2_sd"]},
        "most_frequent": {
            "case_id": reps["most_frequent"]["case_id"],
            "r2": reps["most_frequent"]["r2"],
            "r2_sd": reps["most_frequent"]["r2_sd"],
        },
        "median": {
            "case_id": median["case_id"],
            "r2": median["r2"],
            "r2_sd": median["r2_sd"],
            "statistical_median": rec["stats"]["case_median"],
            "rule": "closest actual case to statistical median; ties broken by canonical ID",
        },
        "source_metrics": str(METRICS),
    }
    (OUT / "B_VF6_wss_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["distributions"] = [d for d in summary["distributions"] if d.get("key") != KEY]
    summary["distributions"].append(rec)
    extra = [item for item in SOURCE_AUDIT if item["path"] == str(METRICS)]
    existing_paths = {item["path"] for item in summary.get("source_audit", [])}
    for item in extra:
        if item["path"] not in existing_paths:
            summary.setdefault("source_audit", []).append(item)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    rows_path = OUT / "representative_cases.csv"
    kept = []
    with rows_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        for row in reader:
            if row["series"] != KEY:
                kept.append(row)
    for role in ("worst", "most_frequent", "best"):
        case = reps[role]
        kept.append({
            "series": KEY,
            "target": rec["target"],
            "role": role,
            "case_id": case["case_id"],
            "r2": case["r2"],
            "r2_sd": case["r2_sd"],
        })
    with rows_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)

    st, r = rec["stats"], rec["representatives"]
    line = (
        f"| {KEY} | {st['n_cases']} | {st['case_mean']:.4f} / {st['case_median']:.4f} / {st['case_p10']:.4f} | "
        f"{st['negative_count']} | "
        + " | ".join(f"{r[k]['case_id']} ({r[k]['r2']:.4f})" for k in ["worst", "most_frequent", "best"])
        + " |"
    )
    readme = (OUT / "README.md").read_text()
    if f"| {KEY} |" not in readme:
        anchor = "| B_VF6 |"
        assert anchor in readme
        readme = readme.replace(
            next(x for x in readme.splitlines() if x.startswith(anchor)),
            next(x for x in readme.splitlines() if x.startswith(anchor)) + "\n" + line,
            1,
        )
        note = (
            "\n## 2026-09-16 增补：B_VF6_wss\n\n"
            "VELWSS2（VF6 三 seed 预测速度 → 冻结 Profile-Secant V3 派生 WSS，legacy 半径校准路径）。"
            "选例与上表同一规则；后处理 best/median/worst 的 Median 是距分布中位最近的真实病例，"
            f"本批为 `{selection['median']['case_id']}`，与直方图橙色 Most 代表不同。"
            "场文件来自 s1234，不是三 seed 平均场。\n"
        )
        readme = readme.rstrip() + "\n" + note
        (OUT / "README.md").write_text(readme)
    print(json.dumps({
        "key": KEY,
        "n": st["n_cases"],
        "reps": rec["representatives"],
        "median": selection["median"],
        "r2_cb_mean": rec["official_field_cb"],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    rec = build_record()
    plot(rec)
    merge_outputs(rec)


if __name__ == "__main__":
    main()
