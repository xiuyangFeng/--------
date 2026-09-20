"""VELWSS2 report: independent metric re-computation, three-seed tables, per-case CSV, figures, README.

Reads only completed assemble outputs (metrics/<radius>.json + predictions); never trains or re-infers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training_wss_min.tools.vf6_velocity_to_wss import (
    EXP, RADIUS_SOURCES, ROOT, RUNS, SEEDS, STATS, VELOCITY_SOURCES, VELWSS1, read, reproduction_path, sha, slug)
from training_wss_min.tools.export_r5v_for_wss import write_json
from training_wss_min.tools.verify_velocity_to_wss import balanced, basic, hotspot, normalized
from training_wss_min.tools.report_velocity_to_wss import configure_font
from training_wss_min.tools import profile_secant_adapter as A

X5_REFS = {
    "s1234": (ROOT / "training_wss_min/experiments/wss_local_wave1_20260912/results.json", "X5"),
    "s7": (ROOT / "training_wss_min/experiments/wss_local_wave1b_20260912/results.json", "X5_s7"),
    "s2025": (ROOT / "training_wss_min/experiments/wss_local_wave1b_20260912/results.json", "X5_s2025"),
}
COLUMNS = [("物理 R²_cb", "field_casebalanced.r2"), ("log_z R²_cb", "normalized.field_casebalanced.r2"),
           ("逐例 R² 均值", "aggregate.r2_casemean"), ("pooled MAE Pa", "field.mae"),
           ("pooled RMSE Pa", "field.rmse"), ("高 WSS R²", "regional_field.high_wss.r2"),
           ("top10 幅值比", "calibration.top10_pred_true_ratio"), ("p99 比", "calibration.p99_pred_true_ratio"),
           ("热点 IoU", "hotspot.top10_iou_casemean")]
X5_KEYS = {"field_casebalanced.r2": "pa_r2_cb", "normalized.field_casebalanced.r2": "norm_r2_cb",
           "aggregate.r2_casemean": "case_mean_r2", "field.mae": "pa_mae", "regional_field.high_wss.r2": "high_wss_r2",
           "calibration.top10_pred_true_ratio": "top10_ratio", "calibration.p99_pred_true_ratio": "p99_ratio",
           "hotspot.top10_iou_casemean": "top10_iou"}
RADIUS_LABEL = {"legacy": "legacy（VELWSS1 旧映射）", "legacy_exact": "legacy_exact（同半径场，映射按 1000/unit_factor 精确对应）",
                "v5atlas": "v5atlas（V5 bundle wall_local_radius）"}
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e",
           "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
ATOL = RTOL = 2e-6


def get(mapping, dotted):
    node = mapping
    for part in dotted.split("."):
        node = node[part]
    return node


def mean_sd(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"))


def close(a, b):
    return (math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)) or (
        not math.isfinite(a) and not math.isfinite(b))


# ------------------------------------------------------------------ loading

def load_all(exp):
    provenance = read(exp / "provenance.json")
    metrics, gates, manifests = {}, {}, {}
    for radius in RADIUS_SOURCES:
        gate = read(exp / "metrics" / f"{radius}_evaluation_gate.json")
        assert gate["passed"] and gate["n_cases"] == 34 and gate["valid_coverage"] == 1.0, radius
        assert gate["provenance_sha256"] == sha(exp / "provenance.json")
        assert gate["metrics_sha256"] == sha(exp / "metrics" / f"{radius}.json")
        metrics[radius] = read(exp / "metrics" / f"{radius}.json")
        gates[radius] = gate
        manifests[radius] = read(exp / "metrics" / f"{radius}_prediction_manifest.json")
    reproductions = {seed: read(reproduction_path(exp, seed)) for seed in SEEDS}
    for seed, rep in reproductions.items():
        assert rep["status"] == "complete" and rep["passed"] and rep["n_cases"] == 34
        assert provenance["velocity_reproduction_sha256"][seed] == sha(reproduction_path(exp, seed))
    return provenance, metrics, gates, manifests, reproductions


def load_x5():
    out = {}
    for seed, (path, arm) in X5_REFS.items():
        best = read(path)["arms"][arm]["checkpoints"]["best"]
        out[seed] = dict(source=str(path), arm=arm, per_case=best["per_case_pa_r2"],
                         **{k: best[v] for k, v in X5_KEYS.items()})
    return out


# ------------------------------------------------------------------ verification

def verify(exp, provenance, metrics, manifests):
    """Independent NumPy re-computation of the headline metrics from the prediction archives."""
    stats = read(STATS)
    report = dict(checks=0, failures=[], frozen_sources_match=(A.frozen_sources() == provenance["frozen_sources"]),
                  calibrator_sha256=provenance["frozen_sources"]["model_sha256"])
    ids = list(manifests[RADIUS_SOURCES[0]])

    def check(name, actual, expected):
        report["checks"] += 1
        if not close(float(actual), float(expected)):
            report["failures"].append(dict(check=name, actual=float(actual), expected=float(expected)))

    for radius in RADIUS_SOURCES:
        truths, preds = [], {}
        for uid in ids:
            entry = manifests[radius][uid]
            assert sha(entry["path"]) == entry["sha256"], (radius, uid)
            with np.load(entry["path"], allow_pickle=False) as z:
                assert str(z["radius_source"]) == radius and str(z["canonical_id"]) == uid
                truth = np.asarray(z["truth_wss_pa"], np.float64)
                truths.append(truth)
                for source in VELOCITY_SOURCES:
                    for kind in ("calibrated", "physics"):
                        key = f"{source}_{kind}"
                        pred = np.asarray(z[f"{key}_wss_pa"], np.float64)
                        preds.setdefault(key, []).append(pred)
                        m = metrics[radius][key]["per_case"][uid]
                        b = basic(truth, pred)
                        check(f"{radius}/{key}/{uid}/r2", b["r2"], m["overall"]["r2"])
                        check(f"{radius}/{key}/{uid}/mae", b["mae"], m["overall"]["mae"])
                        check(f"{radius}/{key}/{uid}/iou", hotspot(truth, pred)["top10_iou"], m["hotspot"]["top10_iou"])
        for key, plist in preds.items():
            m = metrics[radius][key]
            cb = balanced(truths, plist)
            check(f"{radius}/{key}/r2_cb", cb["r2"], m["field_casebalanced"]["r2"])
            check(f"{radius}/{key}/mae_cb", cb["mae"], m["field_casebalanced"]["mae"])
            pooled = basic(np.concatenate(truths), np.concatenate(plist))
            check(f"{radius}/{key}/pooled_mae", pooled["mae"], m["field"]["mae"])
            check(f"{radius}/{key}/pooled_rmse", pooled["rmse"], m["field"]["rmse"])
            check(f"{radius}/{key}/pooled_r2", pooled["r2"], m["field"]["r2"])
            nb = balanced([normalized(t, stats) for t in truths], [normalized(p, stats) for p in plist])
            check(f"{radius}/{key}/logz_r2_cb", nb["r2"], m["normalized"]["field_casebalanced"]["r2"])
            check(f"{radius}/{key}/iou_casemean",
                  float(np.mean([hotspot(t, p)["top10_iou"] for t, p in zip(truths, plist)])),
                  m["hotspot"]["top10_iou_casemean"])
            check(f"{radius}/{key}/case_mean_r2", float(np.mean([basic(t, p)["r2"] for t, p in zip(truths, plist)])),
                  m["aggregate"]["r2_casemean"])
        print(f"verified {radius}: {report['checks']} checks so far, {len(report['failures'])} failures", flush=True)
    report["passed"] = not report["failures"] and report["frozen_sources_match"]
    return report


# ------------------------------------------------------------------ summary

def row_of(m):
    return {dotted: float(get(m, dotted)) for _, dotted in COLUMNS}


def summarize(provenance, metrics, reproductions, x5):
    velwss1 = read(VELWSS1 / "metrics.json")
    out = dict(experiment="VELWSS2", generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seeds=list(SEEDS),
               radius_sources=list(RADIUS_SOURCES), columns=[c for _, c in COLUMNS], radius={}, references={})
    out["references"]["VELWSS1"] = {k: row_of(velwss1[k]) for k in ("test", "oracle", "physics_only")}
    out["references"]["X5_direct"] = {seed: {k: x5[seed][k] for k in X5_KEYS} for seed in SEEDS}
    out["references"]["X5_direct"]["mean"] = {k: mean_sd([x5[s][k] for s in SEEDS])[0] for k in X5_KEYS}
    out["references"]["X5_direct"]["sd"] = {k: mean_sd([x5[s][k] for s in SEEDS])[1] for k in X5_KEYS}
    out["velocity"] = {seed: dict(run=str(RUNS[seed]), checkpoint_epoch=reproductions[seed]["checkpoint_epoch"],
                                  speed_r2_cb=reproductions[seed]["saved_metrics"]["physical_r2_cb"],
                                  execution=reproductions[seed]["execution"], gpu=reproductions[seed]["gpu_name"],
                                  cuda_visible_devices=reproductions[seed]["cuda_visible_devices"],
                                  elapsed_seconds=reproductions[seed]["elapsed_seconds"]) for seed in SEEDS}
    out["velocity"]["speed_r2_cb_mean_sd"] = mean_sd([reproductions[s]["saved_metrics"]["physical_r2_cb"] for s in SEEDS])
    for radius in RADIUS_SOURCES:
        block = dict(rows={}, seed_mean={}, seed_sd={}, deltas={})
        for source in VELOCITY_SOURCES:
            for kind in ("calibrated", "physics"):
                block["rows"][f"{source}_{kind}"] = row_of(metrics[radius][f"{source}_{kind}"])
        for kind in ("calibrated", "physics"):
            for _, col in COLUMNS:
                mu, sd = mean_sd([block["rows"][f"{s}_{kind}"][col] for s in SEEDS])
                block["seed_mean"].setdefault(kind, {})[col] = mu
                block["seed_sd"].setdefault(kind, {})[col] = sd
        main = block["seed_mean"]["calibrated"]["field_casebalanced.r2"]
        block["deltas"] = dict(
            vf6_mean_minus_velwss1=main - velwss1["test"]["field_casebalanced"]["r2"],
            vf6_mean_minus_x5_mean=main - out["references"]["X5_direct"]["mean"]["field_casebalanced.r2"],
            oracle_minus_vf6_mean=block["rows"]["cfd_calibrated"]["field_casebalanced.r2"] - main,
            calibrated_minus_physics_per_seed={s: block["rows"][f"{s}_calibrated"]["field_casebalanced.r2"]
                                               - block["rows"][f"{s}_physics"]["field_casebalanced.r2"] for s in SEEDS})
        out["radius"][radius] = block
    base = out["radius"]["legacy"]
    for radius in RADIUS_SOURCES[1:]:
        blk = out["radius"][radius]
        blk["deltas"]["vs_legacy_paired_per_seed_r2_cb"] = {
            s: blk["rows"][f"{s}_calibrated"]["field_casebalanced.r2"] - base["rows"][f"{s}_calibrated"]["field_casebalanced.r2"]
            for s in SEEDS}
        blk["deltas"]["vs_legacy_oracle_r2_cb"] = (blk["rows"]["cfd_calibrated"]["field_casebalanced.r2"]
                                                   - base["rows"]["cfd_calibrated"]["field_casebalanced.r2"])
        blk["deltas"]["vs_legacy_physics_paired_per_seed_r2_cb"] = {
            s: blk["rows"][f"{s}_physics"]["field_casebalanced.r2"] - base["rows"][f"{s}_physics"]["field_casebalanced.r2"]
            for s in SEEDS}
    radius_evidence = {}
    for key in ("legacy_over_legacy_exact", "legacy_over_v5atlas"):
        p50 = [g["radius"][key]["ratio_p50"] for g in provenance["geometry"].values()]
        p95 = [g["radius"][key]["ratio_p95"] for g in provenance["geometry"].values()]
        p05 = [g["radius"][key]["ratio_p05"] for g in provenance["geometry"].values()]
        frac = [g["radius"][key]["fraction_off_by_20pct"] for g in provenance["geometry"].values()]
        radius_evidence[key] = dict(ratio_p50_across_cases=[min(p50), float(np.median(p50)), max(p50)],
                                    ratio_p05_min=min(p05), ratio_p95_max=max(p95),
                                    fraction_off_by_20pct_mean=float(np.mean(frac)), fraction_off_by_20pct_max=max(frac))
    radius_evidence["legacy_mapping_distance_mm_max"] = max(
        g["radius"]["legacy"]["mapping_distance_mm_max"] for g in provenance["geometry"].values())
    radius_evidence["legacy_unit_factor_range"] = [min(g["radius"]["legacy"]["legacy_unit_factor"] for g in provenance["geometry"].values()),
                                                   max(g["radius"]["legacy"]["legacy_unit_factor"] for g in provenance["geometry"].values())]
    radius_evidence["legacy_exact_mapping_distance_mm_max"] = max(
        g["radius"]["legacy_exact"]["mapping_distance_mm_max"] for g in provenance["geometry"].values())
    out["radius_evidence"] = radius_evidence
    return out


def per_case_rows(metrics, reproductions, x5):
    velwss1 = read(VELWSS1 / "metrics.json")["test"]["per_case"]
    ids = list(metrics["legacy"]["cfd_calibrated"]["per_case"])
    rows = []
    for uid in ids:
        row = dict(canonical_id=uid, cohort=uid.split("/")[0], n_wall=metrics["legacy"]["cfd_calibrated"]["per_case"][uid]["overall"]["n"])
        for seed in SEEDS:
            row[f"speed_r2_{seed}"] = reproductions[seed]["cases"][uid]["speed_r2"]
        row["speed_r2_mean"] = float(np.mean([row[f"speed_r2_{s}"] for s in SEEDS]))
        for radius in RADIUS_SOURCES:
            for seed in SEEDS:
                row[f"{radius}_wss_r2_{seed}"] = metrics[radius][f"{seed}_calibrated"]["per_case"][uid]["overall"]["r2"]
                row[f"{radius}_physics_r2_{seed}"] = metrics[radius][f"{seed}_physics"]["per_case"][uid]["overall"]["r2"]
            row[f"{radius}_wss_r2_mean"] = float(np.mean([row[f"{radius}_wss_r2_{s}"] for s in SEEDS]))
            row[f"{radius}_oracle_r2"] = metrics[radius]["cfd_calibrated"]["per_case"][uid]["overall"]["r2"]
        row["velwss1_r5v_wss_r2"] = velwss1[uid]["overall"]["r2"]
        for seed in SEEDS:
            row[f"x5_direct_r2_{seed}"] = x5[seed]["per_case"][uid]
        row["x5_direct_r2_mean"] = float(np.mean([row[f"x5_direct_r2_{s}"] for s in SEEDS]))
        rows.append(row)
    return rows


# ------------------------------------------------------------------ figures

def style_axes(ax):
    ax.set_facecolor(PALETTE["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=PALETTE["ink2"], labelsize=10)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=1)
    ax.set_axisbelow(True)


def figure_case_r2(rows, output):
    ordered = sorted(rows, key=lambda r: r["legacy_wss_r2_mean"])
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11.5, 13.5), layout="constrained")
    fig.patch.set_facecolor("white")
    style_axes(ax)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=1)
    lo = [min(r[f"legacy_wss_r2_{s}"] for s in SEEDS) for r in ordered]
    hi = [max(r[f"legacy_wss_r2_{s}"] for s in SEEDS) for r in ordered]
    ax.hlines(y, lo, hi, color=PALETTE["blue"], linewidth=2, alpha=0.55, zorder=2)
    ax.scatter([r["x5_direct_r2_mean"] for r in ordered], y, s=64, color=PALETTE["aqua"], edgecolor=PALETTE["surface"],
               linewidth=2, zorder=3, label="X5 直接 WSS（三 seed 均值）")
    ax.scatter([r["velwss1_r5v_wss_r2"] for r in ordered], y, s=64, color=PALETTE["orange"], edgecolor=PALETTE["surface"],
               linewidth=2, zorder=3, label="VELWSS1：R5V 速度 → WSS")
    ax.scatter([r["legacy_wss_r2_mean"] for r in ordered], y, s=80, color=PALETTE["blue"], edgecolor=PALETTE["surface"],
               linewidth=2, zorder=4, label="VELWSS2：VF6 速度 → WSS（三 seed 均值；横线 = seed 极差）")
    ax.set_yticks(y)
    ax.set_yticklabels([r["canonical_id"] for r in ordered], fontsize=8.5, color=PALETTE["ink2"])
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    ax.set_xlim(min(-1.5, min(r["velwss1_r5v_wss_r2"] for r in ordered) * 1.1), 1.0)
    ax.axvline(0, color=PALETTE["axis"], linewidth=1, zorder=1)
    ax.set_xlabel("逐例物理 R²（Pa，V5 有效全壁面；|R²|>1 为对称对数轴）", color=PALETTE["ink2"], fontsize=10.5)
    ax.set_title("34 例逐例 WSS R²：VF6 派生（legacy 半径）与 R5V 派生、X5 直接预测对照\n按 VF6 三 seed 均值从低到高排列",
                 fontsize=13, color=PALETTE["ink"], loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=PALETTE["ink2"])
    fig.savefig(output / "velwss2_case_r2.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_radius_seed(summary, output):
    fig, ax = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    fig.patch.set_facecolor("white")
    style_axes(ax)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=1)
    x = np.arange(len(RADIUS_SOURCES))
    offsets = {"s1234": -0.12, "s7": 0.0, "s2025": 0.12}
    for i, radius in enumerate(RADIUS_SOURCES):
        block = summary["radius"][radius]
        for seed in SEEDS:
            ax.scatter(i + offsets[seed], block["rows"][f"{seed}_calibrated"]["field_casebalanced.r2"], s=56,
                       color=PALETTE["blue"], edgecolor=PALETTE["surface"], linewidth=2, zorder=3)
        mu = block["seed_mean"]["calibrated"]["field_casebalanced.r2"]
        ax.hlines(mu, i - 0.24, i + 0.24, color=PALETTE["blue"], linewidth=2, zorder=4)
        ax.text(i + 0.27, mu, f"{mu:.3f}", va="center", ha="left", fontsize=10, color=PALETTE["ink"])
        oracle = block["rows"]["cfd_calibrated"]["field_casebalanced.r2"]
        ax.scatter(i, oracle, s=72, marker="D", color=PALETTE["muted"], edgecolor=PALETTE["surface"], linewidth=2, zorder=3)
        last = i == len(RADIUS_SOURCES) - 1
        ax.text(i + (-0.1 if last else 0.1), oracle, f"CFD 速度 → 同算子 {oracle:.3f}", va="center",
                ha="right" if last else "left", fontsize=9, color=PALETTE["ink2"])
    ref1 = summary["references"]["VELWSS1"]["test"]["field_casebalanced.r2"]
    ref2 = summary["references"]["X5_direct"]["mean"]["field_casebalanced.r2"]
    for value, label in ((ref1, f"VELWSS1：R5V 速度 → WSS {value_fmt(ref1)}"), (ref2, f"X5 直接 WSS 三 seed 均值 {value_fmt(ref2)}")):
        ax.axhline(value, color=PALETTE["axis"], linewidth=1, zorder=1)
        ax.text(len(RADIUS_SOURCES) - 0.55, value + 0.006, label, ha="right", va="bottom", fontsize=9, color=PALETTE["ink2"])
    ax.set_xticks(x)
    ax.set_xticklabels(["legacy\n旧映射（同 VELWSS1）", "legacy_exact\n同半径场，映射精确对应", "v5atlas\nV5 bundle 半径"],
                       fontsize=10, color=PALETTE["ink2"])
    ax.set_xlim(-0.6, len(RADIUS_SOURCES) - 0.4)
    ax.set_ylabel("物理 R²_cb（病例等权，Pa）", color=PALETTE["ink2"], fontsize=10.5)
    ax.set_title("VF6 预测速度 → 冻结 Profile-Secant V3 → WSS：三 seed（蓝点）与均值（横线），按半径来源\n灰色菱形 = CFD 真值速度进同一算子（诊断参照）",
                 fontsize=12.5, color=PALETTE["ink"], loc="left")
    fig.savefig(output / "velwss2_radius_seed_r2cb.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def value_fmt(v):
    return f"{v:.3f}"


# ------------------------------------------------------------------ markdown

def fmt(v, col):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.3f}" if "MAE" in col or "RMSE" in col else f"{v:.4f}"


def table(summary, radius):
    block = summary["radius"][radius]
    head = "| 路径 | " + " | ".join(name for name, _ in COLUMNS) + " |\n|---|" + "---:|" * len(COLUMNS) + "\n"
    lines = []

    def line(label, row, bold=False):
        cells = [fmt(row[c], n) for n, c in COLUMNS]
        if bold:
            label, cells = f"**{label}**", [f"**{c}**" for c in cells]
        return f"| {label} | " + " | ".join(cells) + " |"

    for seed in SEEDS:
        lines.append(line(f"VF6_{seed} 预测速度 → 冻结 V3（校准）", block["rows"][f"{seed}_calibrated"]))
    mean_row = {c: block["seed_mean"]["calibrated"][c] for _, c in COLUMNS}
    sd_row = {c: block["seed_sd"]["calibrated"][c] for _, c in COLUMNS}
    lines.append(line("VF6 三 seed 均值（主成绩）", mean_row, bold=True))
    lines.append("| VF6 三 seed sd | " + " | ".join(fmt(sd_row[c], n) for n, c in COLUMNS) + " |")
    lines.append(line("诊断：CFD 真值速度 → 冻结 V3", block["rows"]["cfd_calibrated"]))
    for seed in SEEDS:
        lines.append(line(f"诊断：VF6_{seed} 预测速度 → 未校准物理核", block["rows"][f"{seed}_physics"]))
    phys_mean = {c: block["seed_mean"]["physics"][c] for _, c in COLUMNS}
    lines.append(line("诊断：未校准物理核 三 seed 均值", phys_mean))
    lines.append(line("诊断：CFD 真值速度 → 未校准物理核", block["rows"]["cfd_physics"]))
    return head + "\n".join(lines) + "\n"


def reference_table(summary):
    ref = summary["references"]
    head = "| 参照 | " + " | ".join(name for name, _ in COLUMNS) + " |\n|---|" + "---:|" * len(COLUMNS) + "\n"
    lines = []
    for label, row in (("VELWSS1：R5V 预测速度 → 冻结 V3（2026-09-09）", ref["VELWSS1"]["test"]),
                       ("VELWSS1 诊断：CFD 速度 → 冻结 V3", ref["VELWSS1"]["oracle"]),
                       ("VELWSS1 诊断：R5V 预测速度 → 未校准物理核", ref["VELWSS1"]["physics_only"])):
        lines.append(f"| {label} | " + " | ".join(fmt(row[c], n) for n, c in COLUMNS) + " |")
    for seed in SEEDS:
        row = ref["X5_direct"][seed]
        lines.append(f"| X5_{seed} 直接 WSS（best） | " + " | ".join(fmt(row.get(c), n) for n, c in COLUMNS) + " |")
    lines.append("| **X5 直接 WSS 三 seed 均值** | " + " | ".join(f"**{fmt(ref['X5_direct']['mean'].get(c), n)}**" for n, c in COLUMNS) + " |")
    return head + "\n".join(lines) + "\n"


def write_markdown(exp, summary, verification, rows, provenance):
    L = summary["radius"]["legacy"]
    main = L["seed_mean"]["calibrated"]["field_casebalanced.r2"]
    sd = L["seed_sd"]["calibrated"]["field_casebalanced.r2"]
    oracle = L["rows"]["cfd_calibrated"]["field_casebalanced.r2"]
    r1 = summary["references"]["VELWSS1"]["test"]["field_casebalanced.r2"]
    x5 = summary["references"]["X5_direct"]["mean"]["field_casebalanced.r2"]
    ev = summary["radius_evidence"]
    vel = summary["velocity"]
    speeds = ", ".join(f"{s}: {vel[s]['speed_r2_cb']:.4f}" for s in SEEDS)
    text = ["# VELWSS2：VF6 三 seed 预测速度经冻结 Profile-Secant V3 求 WSS（2026-09-16）", "",
            f"- **主成绩（legacy 半径，与 VELWSS1 同口径）**：VF6 三 seed 预测速度 → 冻结校准 WSS 物理 R²_cb = **{main:.4f} ± {sd:.4f}**"
            f"（seed 1234 / 7 / 2025 = " + " / ".join(f"{L['rows'][f'{s}_calibrated']['field_casebalanced.r2']:.4f}" for s in SEEDS) + "）。",
            f"- 对照：VELWSS1（R5V 单 seed）{r1:.4f} → Δ = {main - r1:+.4f}；X5 直接 WSS 三 seed 均值 {x5:.4f} → Δ = {main - x5:+.4f}；"
            f"CFD 真值速度进同一算子 {oracle:.4f}（VF6 均值与之差 {oracle - main:.4f}）。",
            f"- 速度底座：VF6 best（train_loss 选模，400 epoch），速度幅值 R²_cb {speeds}；三 seed 均值 {vel['speed_r2_cb_mean_sd'][0]:.4f}。",
            f"- 不新增训练、不为直接 WSS 模型加速度输出、不重建中心线；冻结校准器 SHA256 `{provenance['frozen_sources']['model_sha256'][:16]}…`（历史上用 train138 WSS 标签监督，不是纯物理算法）。",
            f"- 独立数值验收：{verification['checks']} 项 NumPy 公式复算，失败 {len(verification['failures'])} 项；冻结来源哈希匹配 = {verification['frozen_sources_match']}。", "",
            "## 三种半径来源", "",
            "冻结算子的多尺度近壁深度用每个壁面节点的局部半径 `local_radius_mm` 设定。三种来源只改这一列，壁面、法向、速度、算子与校准器完全相同：", "",
            "| 来源 | 定义 | 目的 |", "|---|---|---|",
            f"| legacy | VELWSS1 原口径：旧 `data_wss_min` bundle 的 `wall_local_radius`，按旧 `wall_coords_raw` 最近邻配到 V5 毫米壁面。旧坐标按病例 unit_factor（{ev['legacy_unit_factor_range'][0]:.1f}–{ev['legacy_unit_factor_range'][1]:.1f}）缩放而非 1000，配对残差最大 {ev['legacy_mapping_distance_mm_max']:.2f} mm | 与 VELWSS1 逐位可比 |",
            f"| legacy_exact | 同一旧半径场，但旧坐标先乘 1000/unit_factor 再配对：逐节点精确对应（最大残差 {ev['legacy_exact_mapping_distance_mm_max']:.1e} mm）。legacy/legacy_exact 半径比：逐例 p50 中位 {ev['legacy_over_legacy_exact']['ratio_p50_across_cases'][1]:.3f}，p95 最大 {ev['legacy_over_legacy_exact']['ratio_p95_max']:.3f}，偏差 >20% 的节点均值 {ev['legacy_over_legacy_exact']['fraction_off_by_20pct_mean']:.2%} | 只去掉映射错位，量化映射误差本身的影响 |",
            f"| v5atlas | V5 bundle `wall_local_radius`（V5 atlas 帧，自适应 SG 窗），与节点一一对应、无映射。legacy/v5atlas 半径比：逐例 p50 范围 {ev['legacy_over_v5atlas']['ratio_p50_across_cases'][0]:.3f}–{ev['legacy_over_v5atlas']['ratio_p50_across_cases'][2]:.3f}，偏差 >20% 的节点均值 {ev['legacy_over_v5atlas']['fraction_off_by_20pct_mean']:.2%} | 换成当前部署输入的半径定义 |", "",
            "校准器训练时看到的是 legacy 半径；后两种来源下校准器输入分布有偏移，因此同时报告未校准物理核。", ""]
    for radius in RADIUS_SOURCES:
        blk = summary["radius"][radius]
        text += [f"## 结果：{RADIUS_LABEL[radius]}", "", table(summary, radius)]
        if radius != "legacy":
            d = blk["deltas"]
            text += [f"相对 legacy 的配对差（物理 R²_cb）：校准路径 " + ", ".join(f"{s} {d['vs_legacy_paired_per_seed_r2_cb'][s]:+.4f}" for s in SEEDS)
                     + f"；未校准物理核 " + ", ".join(f"{s} {d['vs_legacy_physics_paired_per_seed_r2_cb'][s]:+.4f}" for s in SEEDS)
                     + f"；CFD 速度参照 {d['vs_legacy_oracle_r2_cb']:+.4f}。", ""]
    text += ["## 历史参照", "", reference_table(summary),
             "R²_cb 为病例等权共享均值 R²；pooled MAE/RMSE 为顶点合并值；X5 行的 pooled RMSE 未从其结果表转录。R5V/X5 的输入、模型、训练目标不同，只作参照。", "",
             "## 逐例", "",
             "![34 例逐例 R²](velwss2_case_r2.png)", "", "![半径来源 × seed](velwss2_radius_seed_r2cb.png)", "",
             "逐例数值：[per_case_comparison.csv](per_case_comparison.csv)（含三 seed 速度幅值 R²、三种半径下的派生 WSS R²、CFD 参照、VELWSS1 与 X5 三 seed 逐例 R²）。", "",
             "## 执行与证据", "",
             f"- 速度导出：{vel['s1234']['execution']}；GPU {vel['s1234']['gpu']}（CUDA_VISIBLE_DEVICES=" + ", ".join(str(vel[s]['cuda_visible_devices']) for s in SEEDS) + "）。每 seed 34 例、逐例物理/归一化速度指标与 run 内 `eval/ckpt_best/metrics.json` 按 VELWSS1 容差（5e-6 绝对 + 5e-6 相对，整数严格）逐项复现通过。",
             "- CPU 阶段（prepare / worker ×3 半径 / assemble ×3）全部经 Slurm，作业号见 `submission.json`；分片 8192 节点，逐点分片拼回完整病例后再做病例内校准。",
             f"- 全壁面 {provenance['n_wall']} 节点、34 例、100% 覆盖，未按预测结果删点。",
             "- 复现报告：`python -m training_wss_min.tools.report_vf6_velocity_to_wss`（只读，不训练不推理）。", ""]
    (exp / "README.md").write_text("\n".join(text), encoding="utf-8")
    results = ["# VELWSS2 结果表（自动生成 " + time.strftime("%Y-%m-%d %H:%M") + "）", ""]
    for radius in RADIUS_SOURCES:
        results += [f"## {RADIUS_LABEL[radius]}", "", table(summary, radius)]
    results += ["## 参照", "", reference_table(summary)]
    (exp / "results.md").write_text("\n".join(results), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXP)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    exp = args.experiment_dir
    configure_font()
    provenance, metrics, gates, manifests, reproductions = load_all(exp)
    x5 = load_x5()
    verification = (dict(checks=0, failures=[], frozen_sources_match=None, passed=None, skipped=True)
                    if args.skip_verify else verify(exp, provenance, metrics, manifests))
    write_json(exp / "verification.json", verification)
    summary = summarize(provenance, metrics, reproductions, x5)
    summary["verification"] = {k: v for k, v in verification.items() if k != "failures"} | {"n_failures": len(verification["failures"])}
    summary["gates"] = {r: {k: g[k] for k in ("job_id", "n_wall", "evaluation_sources_stable_during_assemble")} for r, g in gates.items()}
    write_json(exp / "results.json", summary)
    rows = per_case_rows(metrics, reproductions, x5)
    with (exp / "per_case_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure_case_r2(rows, exp)
    figure_radius_seed(summary, exp)
    write_markdown(exp, summary, verification, rows, provenance)
    L = summary["radius"]["legacy"]
    print(json.dumps(dict(main_r2_cb_mean=L["seed_mean"]["calibrated"]["field_casebalanced.r2"],
                          sd=L["seed_sd"]["calibrated"]["field_casebalanced.r2"],
                          oracle=L["rows"]["cfd_calibrated"]["field_casebalanced.r2"],
                          verification_passed=verification.get("passed")), ensure_ascii=False))


if __name__ == "__main__":
    main()
