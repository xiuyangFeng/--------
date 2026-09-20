"""Evidence-based summary of the wave-1 local-information screening matrix.

Reads every arm's ``eval/ckpt_<best|last|ema>/metrics.json`` (only for stages the queue
recorded as complete), compares against the contemporaneous control X0 and the
historical C1 anchor, and writes ``results.md`` + ``results.json`` into the experiment
directory.  Single seed, exposed test34: screening evidence only.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from training_wss_min import config as C

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave1_20260912"
RUNS = ROOT / "training_wss_min/runs"
METRICS = {
    "pa_r2_cb": ("field_casebalanced", "r2"),
    "norm_r2_cb": ("normalized", "field_casebalanced", "r2"),
    "pa_mae": ("field", "mae"),
    "case_mean_r2": ("aggregate", "r2_casemean"),
    "case_p10_r2": ("aggregate", "r2_casep10"),
    "high_wss_r2": ("regional_field", "high_wss", "r2"),
    "top10_ratio": ("calibration", "top10_pred_true_ratio"),
    "p99_ratio": ("calibration", "p99_pred_true_ratio"),
    "top10_iou": ("hotspot", "top10_iou_casemean"),
}
DOMAINS = ("AG", "AAA", "ILO")
BAND_PA, BAND_NORM = 0.034, 0.009  # single-seed vs single-seed 95% bands (R4 calibration, §11.6)


def dig(document, path):
    value = document
    for key in path:
        value = value[key]
    return float(value)


def summarize(metrics_path: Path) -> dict:
    document = json.loads(metrics_path.read_text())["test"]
    out = {}
    for name, path in METRICS.items():
        try:
            out[name] = dig(document, path)
        except (KeyError, TypeError, ValueError):
            out[name] = None  # e.g. metrics that only exist for one target family
    out["target"] = document.get("target", "wss")
    out["negative_r2_cases"] = sorted(case for case, item in document["per_case"].items()
                                      if float(item["overall"]["r2"]) < 0)
    out["domains_pa_r2_cb"] = {d: float(document["group_casebalanced"][d]["r2"])
                               for d in DOMAINS if d in document["group_casebalanced"]}
    out["per_case_pa_r2"] = {case: float(item["overall"]["r2"]) for case, item in document["per_case"].items()}
    out["source"] = str(metrics_path)
    return out


def fmt(value, digits=4):
    return "—" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def delta(value, reference):
    if value is None or reference is None:
        return "—"
    return f"{value - reference:+.4f}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=ROOT / "training_wss_min/configs" / NAME)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "training_wss_min/experiments" / NAME)
    args = parser.parse_args(argv)
    matrix = json.loads((args.config_dir / "matrix.json").read_text())
    status_path = args.experiment_dir / "queue_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {"arms": {}}
    anchor = summarize(Path(matrix["anchor_run"]) / "eval/ckpt_best/metrics.json")
    rows = {}
    for arm in matrix["arms"]:
        aid = arm["id"]
        record = status["arms"].get(aid, {})
        run = RUNS / json.loads((args.config_dir / arm["config"]).read_text())["name"]
        entry = {"id": aid, "title": arm["title"], "modules": arm["modules"], "status": record.get("status", "not_started"),
                 "single_change": arm["single_change"], "checkpoints": {}}
        for checkpoint in arm["evaluate"]:
            stage = record.get("stages", {}).get(f"eval_{checkpoint}", {})
            path = run / f"eval/ckpt_{checkpoint}/metrics.json"
            if stage.get("returncode") == 0 and path.is_file():
                entry["checkpoints"][checkpoint] = summarize(path)
        rows[aid] = entry
    control_id = matrix.get("control_id", "X0")
    control = rows.get(control_id, {}).get("checkpoints", {}).get("best") if control_id else None
    external = {}
    for label, path in (matrix.get("external_reference_runs") or {}).items():
        if Path(path).is_file():
            external[label] = summarize(Path(path))
    lines = [f"# {matrix.get('experiment', NAME)} 结果（自动生成 {time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"单 run、已暴露 test34；参照 = 同期对照 {control_id or '无'}（同配置重跑）、matrix.json 的外部参照与历史 C1；"
             f"单 seed 对单 seed 的 95% 带约 ±{BAND_PA} 物理 R²_cb / ±{BAND_NORM} 归一化。只作筛选，不作显著性或泛化结论。", "",
             "| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, m in [("C1 历史", anchor)] + list(external.items()):
        unit = {"pressure_mixed": "压力 Pa", "velocity": "速度幅值 m/s"}.get(m.get("target"), "")
        same_family = m.get("target", "wss") == anchor.get("target", "wss")
        lines.append(f"| {label} | 参照{'（' + unit + '）' if unit else ''} | best | " + " | ".join([
            fmt(m["pa_r2_cb"]), "—", delta(m["pa_r2_cb"], anchor["pa_r2_cb"]) if (m is not anchor and same_family) else "—",
            fmt(m["norm_r2_cb"]), "—", fmt(m["pa_mae"], 3),
            fmt(m["case_p10_r2"]), fmt(m["high_wss_r2"], 3), fmt(m["top10_ratio"], 3),
            fmt(m["p99_ratio"], 3), fmt(m["top10_iou"]), str(len(m["negative_r2_cases"]))]) + " |")
    for aid, entry in rows.items():
        if not entry["checkpoints"]:
            lines.append(f"| {aid} | {entry['title']} | — | {entry['status']} | | | | | | | | | | | |")
            continue
        for checkpoint, m in entry["checkpoints"].items():
            unit = {"pressure_mixed": "压力 Pa", "velocity": "速度幅值 m/s"}.get(m.get("target"), "")
            title = f"{entry['title']}（{unit}）" if unit else entry["title"]
            same_family = m.get("target", "wss") == anchor.get("target", "wss")
            lines.append(f"| {aid} | {title} | {checkpoint} | " + " | ".join([
                fmt(m["pa_r2_cb"]), delta(m["pa_r2_cb"], control["pa_r2_cb"] if control else None),
                delta(m["pa_r2_cb"], anchor["pa_r2_cb"]) if same_family else "—", fmt(m["norm_r2_cb"]),
                delta(m["norm_r2_cb"], control["norm_r2_cb"] if control else None), fmt(m["pa_mae"], 3),
                fmt(m["case_p10_r2"]), fmt(m["high_wss_r2"], 3), fmt(m["top10_ratio"], 3), fmt(m["p99_ratio"], 3),
                fmt(m["top10_iou"]), str(len(m["negative_r2_cases"]))]) + " |")
    lines += ["", "## 分域物理 R²_cb（best）", "", "| 臂 | AG | AAA | ILO |", "|---|---:|---:|---:|"]
    lines.append("| C1 历史 | " + " | ".join(fmt(anchor["domains_pa_r2_cb"].get(d)) for d in DOMAINS) + " |")
    for aid, entry in rows.items():
        best = entry["checkpoints"].get("best")
        if best:
            lines.append(f"| {aid} | " + " | ".join(fmt(best["domains_pa_r2_cb"].get(d)) for d in DOMAINS) + " |")
    if control:
        lines += ["", "## 逐例配对（best，相对 X0 的 Pa R² 变化）", "", "| 臂 | 变好例数/34 | 中位 Δ | 最差 8 例（按 X0）均值 Δ |", "|---|---:|---:|---:|"]
        worst8 = sorted(control["per_case_pa_r2"], key=control["per_case_pa_r2"].get)[:8]
        for aid, entry in rows.items():
            best = entry["checkpoints"].get("best")
            if not best or aid == "X0":
                continue
            deltas = {c: best["per_case_pa_r2"][c] - control["per_case_pa_r2"][c]
                      for c in control["per_case_pa_r2"] if c in best["per_case_pa_r2"]}
            values = sorted(deltas.values())
            median = values[len(values) // 2] if values else float("nan")
            worst = sum(deltas[c] for c in worst8 if c in deltas) / max(1, len([c for c in worst8 if c in deltas]))
            lines.append(f"| {aid} | {sum(v > 0 for v in values)}/{len(values)} | {median:+.4f} | {worst:+.4f} |")
    lines += ["", f"队列状态：{status.get('status', 'unknown')}；作业 {status.get('job_id', '—')}；"
              f"跳过的候选：{'; '.join(matrix.get('skipped', []))}"]
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.experiment_dir / "results.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "anchor": anchor, "arms": rows,
         "bands": {"pa_r2_cb": BAND_PA, "norm_r2_cb": BAND_NORM}, "queue_status": status.get("status")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.experiment_dir / "results.md")


if __name__ == "__main__":
    main()
