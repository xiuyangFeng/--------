#!/usr/bin/env python3
"""Add the finished v5.1 X5D_v51 R² distribution without redrawing the 2026-09-14 deck."""
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
RUNS = ROOT / "training_wss_min/runs/wss_v51_wave1_20260916"
SEEDS = (1234, 7, 2025, 11, 2026)
KEY = "A_X5D_v51"


def metric_paths() -> list[Path]:
    return [RUNS / f"X5D_v51_s{seed}" / "eval/ckpt_best/metrics.json" for seed in SEEDS]


def build_record() -> dict:
    paths = metric_paths()
    payloads = [read(path)["test"] for path in paths]
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
        title="方向1｜X5D_v51 · 密度增广直接 WSS",
        subtitle="WSS · Pa · test34 · 峰值1162 · best · 每例五个 seed 的预测R²均值；横须=seed间样本SD（非置信区间）",
        group="A 当前直接WSS",
        note=(
            "v5.1 重训完成的密度增广 X5D；seed 1234/7/2025/11/2026。"
            "图中是病例分数均值，既非集成预测R²，也非总体R²_cb。"
            "不是纵向几何 X5D_long，也不是 cap/dual/noise。"
        ),
        source_paths=[str(path) for path in paths],
        source_field="test.per_case[case_id].overall.r2",
        official_field_cb=float(np.mean(cb)),
        official_field_cb_sd=float(np.std(cb, ddof=1)),
        cases=cases,
        replicate_count=5,
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
        "data": "v5.1 train136/test34; density-augmented X5D_v51; not X5D_long",
        "replicate_policy": "one patient per dot; arithmetic mean of seed 1234/7/2025/11/2026 R2; sample SD whiskers; not ensemble predictions",
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
        "source_metrics": [str(path) for path in metric_paths()],
    }
    (OUT / "A_X5D_v51_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["distributions"] = [d for d in summary["distributions"] if d.get("key") != KEY]
    summary["distributions"].append(rec)
    extra = [item for item in SOURCE_AUDIT if any(item["path"] == str(path) for path in metric_paths())]
    existing_paths = {item["path"] for item in summary.get("source_audit", [])}
    for item in extra:
        if item["path"] not in existing_paths:
            summary.setdefault("source_audit", []).append(item)
            existing_paths.add(item["path"])
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
        anchor = "| A_X5X11 |"
        assert anchor in readme
        readme = readme.replace(
            next(x for x in readme.splitlines() if x.startswith(anchor)),
            next(x for x in readme.splitlines() if x.startswith(anchor)) + "\n" + line,
            1,
        )
        note = (
            "\n## 2026-09-17 增补：A_X5D_v51\n\n"
            "v5.1 数据更新后已跑完的密度增广 X5D（五 seed 1234/7/2025/11/2026，ckpt_best，test34）。"
            "选例与上表同一规则；后处理 best/median/worst 的 Median 是距分布中位最近的真实病例，"
            f"本批为 `{selection['median']['case_id']}`，与直方图橙色 Most 代表不同。"
            "场文件来自 s1234，不是五 seed 平均场。不是纵向几何 X5D_long。\n"
        )
        usage = "只补 VELWSS2：python3 build_velwss2_distribution.py。"
        if usage in readme and "build_x5d_v51_distribution.py" not in readme:
            readme = readme.replace(
                usage,
                usage + " 只补 X5D_v51：python3 build_x5d_v51_distribution.py。",
                1,
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
