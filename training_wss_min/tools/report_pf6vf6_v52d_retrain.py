"""Readout of the PF6 / VF6 v5.2d retrain (pf6vf6_v52d_retrain_20261002; 01 block tracking §43).

Read-only on saved predictions (evaluate --save-predictions), tolerant of unfinished arms. Definitions:
  R2_cb      case-balanced field R2 of training_wss_min.metrics.casebalanced_field_metrics (every unit weight 1, points equal
             within a unit, centred on the mean of the unit means); computed from per-unit sufficient statistics, so pooling
             over folds / cohorts / point groups is exact. Each run's own test R2_cb is recomputed and checked against its
             metrics.json (the "consistency" line).
  pressure   PF6 target pressure_mixed (Pa relative to the case volume-mean pressure at the peak step): all query rows,
             wall rows (point_kind 0) and interior rows (point_kind 1).
  velocity   VF6 three components in the atlas frame; scored as speed |u| R2_cb (as evaluate.py), plus vector RMSE (pooled
             over points, m/s), speed-floor 0.05 m/s direction cosine (unit mean, plain and speed weighted) and per-component R2_cb.
  ensemble   mean prediction per point over members (velocity: mean vector, then its speed); members must share query rows
             and labels.
  CV5        per seed: the five held-out folds pooled over the 261 units; three-seed mean +- sd (ddof 1); three-seed
             out-of-fold ensemble (unit u uses the three models whose held-out fold contains u).
  recover8   n = 8, descriptive only: full265 per seed + ensemble; E3 = CV5 fold models (per-seed five-fold ensembles and the
             15-model ensemble); E5 = deployed v5.0 PF6_VF6_peak_3seed_20260920 (train138) per seed + ensemble.
Historical v5.0 test34 and v5.1 cv3 numbers are other evaluation sets: listed side by side, never subtracted.

    python -m training_wss_min.tools.report_pf6vf6_v52d_retrain [--checkpoint best|last]
    -> experiments/pf6vf6_v52d_retrain_20261002/readout_ckpt_<ckpt>.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import metrics as M

ROOT = C.PROJECT_ROOT
NAME = "pf6vf6_v52d_retrain_20261002"
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs" / NAME
SEEDS = (1234, 7, 2025)
FOLDS = (0, 1, 2, 3, 4)
TARGETS = ("PF6", "VF6")
COHORTS = ("AG", "AAA", "ILO")
SPEED_FLOOR = 0.05
HIST_V50 = {"PF6": {1234: "wss_local_wave2_20260912/PF6_s1234", 7: "wss_local_wave3_20260913/PF6_s7",
                    2025: "wss_local_wave3_20260913/PF6_s2025"},
            "VF6": {1234: "wss_local_wave2_20260912/VF6_s1234", 7: "wss_local_wave3_20260913/VF6_s7",
                    2025: "wss_local_wave3_20260913/VF6_s2025"}}
HIST_V51 = {"PF6": "volume_time_20260919/PF6_v51_f{k}_s1234", "VF6": "volume_time_20260919/VF6_v51_f{k}_s1234"}


# ----------------------------------------------------------------------------------------------- per-unit statistics
def scalar_stats(t: np.ndarray, p: np.ndarray) -> dict:
    t = np.asarray(t, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    mt = float(t.mean())
    var = float(np.mean((t - mt) ** 2))
    mse = float(np.mean((t - p) ** 2))
    return {"mt": mt, "mt2": float(np.mean(t * t)), "mse": mse, "mae": float(np.mean(np.abs(t - p))), "n": int(t.size),
            "r2_unit": 1.0 - mse / var if var > 1e-12 else float("nan")}


def r2cb(stats: list[dict]) -> float:
    """Exactly metrics.casebalanced_field_metrics' R2 from per-unit (mean t, mean t^2, mse)."""
    if not stats:
        return float("nan")
    g = float(np.mean([s["mt"] for s in stats]))
    tot = float(np.mean([s["mt2"] - 2.0 * g * s["mt"] + g * g for s in stats]))
    res = float(np.mean([s["mse"] for s in stats]))
    return 1.0 - res / tot if tot > 1e-12 else float("nan")


def unit_record(target: str, true: np.ndarray, pred: np.ndarray, kind: np.ndarray) -> dict:
    if target == "PF6":
        return {"all": scalar_stats(true, pred), "wall": scalar_stats(true[kind == 0], pred[kind == 0]),
                "interior": scalar_stats(true[kind == 1], pred[kind == 1])}
    st, sp = np.linalg.norm(true, axis=1), np.linalg.norm(pred, axis=1)
    rec = {"speed": scalar_stats(st, sp)}
    for i, c in enumerate("uvw"):
        rec[c] = scalar_stats(true[:, i], pred[:, i])
    m = st > SPEED_FLOOR
    if m.sum() >= 10:
        cos = np.sum(true[m] * pred[m], axis=1) / np.clip(st[m] * sp[m], 1e-12, None)
        rec["cos"] = float(np.mean(cos))
        rec["cos_w"] = float(np.sum(cos * st[m]) / np.sum(st[m]))
    rec["sse_vec"] = float(np.sum((np.asarray(true, np.float64) - np.asarray(pred, np.float64)) ** 2))
    rec["n_vec"] = int(len(true))
    return rec


def load_unit(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return z["query_idx"], z["point_kind"], z["true_raw"].astype(np.float64), z["pred_raw"].astype(np.float64)


def manifest_units(pred_root: Path) -> dict[str, Path] | None:
    man = pred_root / "manifest.json"
    if not man.is_file():
        return None
    data = json.loads(man.read_text())   # predictions/test/manifest.json: {"cases": [{unit_id, file (relative to test/)}]}
    return {c["unit_id"]: pred_root / c["file"] for c in data["cases"]}


def member_record(target: str, files: list[Path]) -> dict:
    """One unit, one or more members (ensemble = mean prediction)."""
    q0, k0, t0, p = load_unit(files[0])
    preds = [p]
    for f in files[1:]:
        q, k, t, pm = load_unit(f)
        if not (np.array_equal(q, q0) and np.array_equal(t, t0)):
            raise ValueError(f"ensemble members disagree on rows/labels: {files[0]} vs {f}")
        preds.append(pm)
    return unit_record(target, t0, np.mean(preds, axis=0) if len(preds) > 1 else preds[0], k0)


def summarize(target: str, recs: dict[str, dict]) -> dict:
    units = sorted(recs)
    out = {"n_units": len(units)}
    if target == "PF6":
        for g in ("all", "wall", "interior"):
            out[f"r2cb_{g}"] = r2cb([recs[u][g] for u in units])
        out["unit_median_r2"] = float(np.median([recs[u]["all"]["r2_unit"] for u in units]))
        out["mae_pa"] = float(np.mean([recs[u]["all"]["mae"] for u in units]))
    else:
        out["r2cb_speed"] = r2cb([recs[u]["speed"] for u in units])
        out["unit_median_r2"] = float(np.median([recs[u]["speed"]["r2_unit"] for u in units]))
        for c in "uvw":
            out[f"r2cb_{c}"] = r2cb([recs[u][c] for u in units])
        out["vector_rmse_m_s"] = float(np.sqrt(sum(recs[u]["sse_vec"] for u in units) / sum(recs[u]["n_vec"] for u in units)))
        cos = [recs[u]["cos"] for u in units if "cos" in recs[u]]
        cosw = [recs[u]["cos_w"] for u in units if "cos_w" in recs[u]]
        out["direction_cosine_casemean"] = float(np.mean(cos)) if cos else float("nan")
        out["direction_cosine_speedweighted_casemean"] = float(np.mean(cosw)) if cosw else float("nan")
        out["mae_speed_m_s"] = float(np.mean([recs[u]["speed"]["mae"] for u in units]))
    key = "all" if target == "PF6" else "speed"
    out["cohorts"] = {c: {"n_units": len(us), "r2cb": r2cb([recs[u][key] for u in us])}
                      for c in COHORTS if (us := [u for u in units if u.startswith(c + "/")])}
    if target == "PF6":
        for c in out["cohorts"]:
            us = [u for u in units if u.startswith(c + "/")]
            out["cohorts"][c].update(r2cb_wall=r2cb([recs[u]["wall"] for u in us]),
                                     r2cb_interior=r2cb([recs[u]["interior"] for u in us]))
    return out


def primary(target: str, s: dict) -> float:
    return s["r2cb_all"] if target == "PF6" else s["r2cb_speed"]


# ------------------------------------------------------------------------------------------------------------ helpers
def pred_root(run: Path, ckpt: str) -> Path:
    return run / f"eval/ckpt_{ckpt}/predictions/test"


def stored_r2(metrics_path: Path, target: str) -> dict | None:
    if not metrics_path.is_file():
        return None
    t = json.loads(metrics_path.read_text())["test"]
    out = {"r2cb": float(t["field_casebalanced"]["r2"]), "n_cases": int(t["field_casebalanced"]["n_cases"])}
    if target == "PF6" and "query_groups" in t:
        out.update(r2cb_wall=float(t["query_groups"]["wall"]["field_casebalanced"]["r2"]),
                   r2cb_interior=float(t["query_groups"]["interior"]["field_casebalanced"]["r2"]))
    if target == "VF6" and "vector" in t:
        out.update(vector_rmse_m_s=float(t["vector"]["vector_rmse_m_s"]),
                   direction_cosine_casemean=float(t["vector"]["direction_cosine_casemean"]))
    return out


def fmt(x, nd=4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def mean_sd(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, label: str, target: str, recs: dict, metrics_path: Path):
        st = stored_r2(metrics_path, target)
        if st is None or not recs:
            return
        s = summarize(target, recs)
        diff = abs(primary(target, s) - st["r2cb"])
        self.rows.append({"run": label, "recomputed": primary(target, s), "stored": st["r2cb"], "abs_diff": diff,
                          "n_units": s["n_units"], "n_stored": st["n_cases"]})

    def worst(self):
        return max((r["abs_diff"] for r in self.rows), default=float("nan"))


def run_records(target: str, run_or_eval: Path, ckpt: str | None) -> tuple[dict, Path]:
    """{unit: record} of one run's saved test predictions (ckpt None: run_or_eval is an evaluate --output-dir)."""
    root = pred_root(run_or_eval, ckpt) if ckpt else run_or_eval / "predictions/test"
    metrics = run_or_eval / f"eval/ckpt_{ckpt}/metrics.json" if ckpt else run_or_eval / "metrics.json"
    files = manifest_units(root)
    if files is None:
        return {}, metrics
    return {u: member_record(target, [f]) for u, f in files.items()}, metrics


# ------------------------------------------------------------------------------------------------------------ sections
def section_cv5(target: str, ckpt: str, lines: list, out: dict, checks: Checks):
    name = "压力 PF6" if target == "PF6" else "速度 VF6"
    lines += [f"### A-{target[0]}：{name} CV5 折外（261 例）", ""]
    per_seed, files_by_seed, fold_rows = {}, {}, []
    for s in SEEDS:
        recs, files, done = {}, {}, 0
        for k in FOLDS:
            run = RUNS / f"{target}_v52cv_f{k}_s{s}"
            r, mpath = run_records(target, run, ckpt)
            if not r:
                continue
            done += 1
            checks.add(f"{target}_v52cv_f{k}_s{s}", target, r, mpath)
            fs = summarize(target, r)
            fold_rows.append((s, k, fs))
            recs.update(r)
            files.update(manifest_units(pred_root(run, ckpt)))
        if recs:
            per_seed[s] = {"folds_done": done, "summary": summarize(target, recs)}
            files_by_seed[s] = files
    if target == "PF6":
        lines += ["| seed | 完成折 | 单元数 | R²_cb 全部 | 壁面 | 内部 | 逐例中位 | MAE (Pa) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for s, v in per_seed.items():
            x = v["summary"]
            lines.append(f"| {s} | {v['folds_done']}/5 | {x['n_units']} | {fmt(x['r2cb_all'])} | {fmt(x['r2cb_wall'])} | "
                         f"{fmt(x['r2cb_interior'])} | {fmt(x['unit_median_r2'])} | {fmt(x['mae_pa'], 1)} |")
    else:
        lines += ["| seed | 完成折 | 单元数 | 速率 R²_cb | 逐例中位 | 向量 RMSE (m/s) | 方向余弦 | 速率加权余弦 | u / v / w R²_cb |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for s, v in per_seed.items():
            x = v["summary"]
            lines.append(f"| {s} | {v['folds_done']}/5 | {x['n_units']} | {fmt(x['r2cb_speed'])} | {fmt(x['unit_median_r2'])} | "
                         f"{fmt(x['vector_rmse_m_s'])} | {fmt(x['direction_cosine_casemean'])} | "
                         f"{fmt(x['direction_cosine_speedweighted_casemean'])} | "
                         f"{fmt(x['r2cb_u'], 3)} / {fmt(x['r2cb_v'], 3)} / {fmt(x['r2cb_w'], 3)} |")
    full = [s for s, v in per_seed.items() if v["folds_done"] == 5]
    res = {"per_seed": {str(s): v for s, v in per_seed.items()},
           "per_fold": [{"seed": s, "fold": k, **{kk: vv for kk, vv in fs.items() if kk != "cohorts"}} for s, k, fs in fold_rows]}
    keys = ["r2cb_all", "r2cb_wall", "r2cb_interior"] if target == "PF6" else ["r2cb_speed", "vector_rmse_m_s",
                                                                              "direction_cosine_casemean"]
    if full:
        res["seed_mean_sd"] = {k: {"mean": float(np.mean([per_seed[s]["summary"][k] for s in full])),
                                   "sd": float(np.std([per_seed[s]["summary"][k] for s in full], ddof=1)) if len(full) > 1 else None,
                                   "n_seeds": len(full)} for k in keys}
        lines += ["", f"完成五折的 seed 数 {len(full)}：" + "；".join(
            f"{k} {mean_sd([per_seed[s]['summary'][k] for s in full])}" for k in keys)]
    if len(full) == 3:
        units = sorted(set.intersection(*(set(files_by_seed[s]) for s in SEEDS)))
        ens = {u: member_record(target, [files_by_seed[s][u] for s in SEEDS]) for u in units}
        es = summarize(target, ens)
        res["ensemble3"] = es
        if target == "PF6":
            lines.append(f"三 seed 折外集成（{es['n_units']} 例）：R²_cb **{es['r2cb_all']:.4f}**（壁面 {es['r2cb_wall']:.4f}、"
                         f"内部 {es['r2cb_interior']:.4f}），逐例中位 {es['unit_median_r2']:.4f}")
        else:
            lines.append(f"三 seed 折外集成（{es['n_units']} 例；平均向量后取速率）：速率 R²_cb **{es['r2cb_speed']:.4f}**，"
                         f"逐例中位 {es['unit_median_r2']:.4f}，向量 RMSE {es['vector_rmse_m_s']:.4f} m/s，"
                         f"方向余弦 {es['direction_cosine_casemean']:.4f}")
    # cohorts
    lines += ["", "分队列（R²_cb；单 seed 列为三 seed 均值）：", ""]
    cohort_cols = ["n", "单 seed 均值", "集成"] + (["集成 壁面", "集成 内部"] if target == "PF6" else [])
    lines += ["| 队列 | " + " | ".join(cohort_cols) + " |", "|---|" + "---:|" * len(cohort_cols)]
    res["cohorts"] = {}
    for c in COHORTS:
        seeds_c = [per_seed[s]["summary"]["cohorts"][c]["r2cb"] for s in full if c in per_seed[s]["summary"]["cohorts"]]
        ens_c = res.get("ensemble3", {}).get("cohorts", {}).get(c, {})
        n = ens_c.get("n_units") or (per_seed[full[0]]["summary"]["cohorts"].get(c, {}).get("n_units") if full else None)
        row = [str(n or "—"), fmt(float(np.mean(seeds_c)) if seeds_c else None), fmt(ens_c.get("r2cb"))]
        if target == "PF6":
            row += [fmt(ens_c.get("r2cb_wall")), fmt(ens_c.get("r2cb_interior"))]
        lines.append(f"| {c} | " + " | ".join(row) + " |")
        res["cohorts"][c] = {"seed_mean": float(np.mean(seeds_c)) if seeds_c else None, "ensemble3": ens_c}
    lines.append("")
    out["A"][target] = res


def section_recover8(target: str, ckpt: str, lines: list, out: dict, checks: Checks):
    name = "压力 PF6" if target == "PF6" else "速度 VF6"
    key = "r2cb_all" if target == "PF6" else "r2cb_speed"
    lines += [f"### B-{target[0]}：{name} full265 → recover8（8 例，只作描述）", ""]
    res = {}
    rows = []
    full_files, full_recs = {}, {}
    for s in SEEDS:
        run = RUNS / f"{target}_full265_s{s}"
        r, mpath = run_records(target, run, ckpt)
        if r:
            checks.add(f"{target}_full265_s{s}", target, r, mpath)
            full_recs[s] = r
            full_files[s] = manifest_units(pred_root(run, ckpt))
            rows.append((f"full265 s{s}", summarize(target, r)))
        else:
            rows.append((f"full265 s{s}", None))
    if len(full_files) == 3:
        units = sorted(set.intersection(*(set(full_files[s]) for s in SEEDS)))
        ens = {u: member_record(target, [full_files[s][u] for s in SEEDS]) for u in units}
        rows.append(("**full265 三 seed 集成**", summarize(target, ens)))
        res["full265_ensemble_per_unit"] = {u: (ens[u]["all"] if target == "PF6" else ens[u]["speed"])["r2_unit"] for u in units}
    if ckpt == "best":
        # E3: CV5 fold models on recover8
        e3_files = {}
        for s in SEEDS:
            for k in FOLDS:
                d = EXP / "recover8_cv5" / f"{target}_v52cv_f{k}_s{s}"
                files = manifest_units(d / "predictions/test")
                if files:
                    r = {u: member_record(target, [f]) for u, f in files.items()}
                    checks.add(f"E3 {target}_v52cv_f{k}_s{s}", target, r, d / "metrics.json")
                    e3_files[(s, k)] = files
        singles = []
        for s in SEEDS:
            members = [e3_files[(s, k)] for k in FOLDS if (s, k) in e3_files]
            for m in members:
                singles.append(primary(target, summarize(target, {u: member_record(target, [f]) for u, f in m.items()})))
            if len(members) == 5:
                units = sorted(set.intersection(*(set(m) for m in members)))
                rows.append((f"E3 s{s} 五折集成", summarize(target, {u: member_record(target, [m[u] for m in members]) for u in units})))
        if len(e3_files) == 15:
            members = list(e3_files.values())
            units = sorted(set.intersection(*(set(m) for m in members)))
            rows.append(("**E3 15 模型集成**", summarize(target, {u: member_record(target, [m[u] for m in members]) for u in units})))
        if singles:
            res["E3_single_fold_model_mean"] = float(np.mean(singles))
            res["E3_n_models"] = len(singles)
        # E5: deployed v5.0 weights
        e5_files = {}
        for s in SEEDS:
            d = EXP / "e5_deployed_v50" / f"{target}_s{s}" / "eval"
            files = manifest_units(d / "predictions/test")
            if files:
                r = {u: member_record(target, [f]) for u, f in files.items()}
                checks.add(f"E5 {target}_s{s}", target, r, d / "metrics.json")
                e5_files[s] = files
                rows.append((f"E5 已部署 v5.0 s{s}", summarize(target, r)))
        if len(e5_files) == 3:
            units = sorted(set.intersection(*(set(e5_files[s]) for s in SEEDS)))
            rows.append(("**E5 已部署 v5.0 三 seed 集成**",
                         summarize(target, {u: member_record(target, [e5_files[s][u] for s in SEEDS]) for u in units})))
    if target == "PF6":
        lines += ["| 模型 | 单元数 | R²_cb 全部 | 壁面 | 内部 | 逐例中位 |", "|---|---:|---:|---:|---:|---:|"]
        for label, x in rows:
            lines.append(f"| {label} | (未完成) | | | | |" if x is None else
                         f"| {label} | {x['n_units']} | {fmt(x['r2cb_all'])} | {fmt(x['r2cb_wall'])} | {fmt(x['r2cb_interior'])} | "
                         f"{fmt(x['unit_median_r2'])} |")
    else:
        lines += ["| 模型 | 单元数 | 速率 R²_cb | 逐例中位 | 向量 RMSE (m/s) | 方向余弦 |", "|---|---:|---:|---:|---:|---:|"]
        for label, x in rows:
            lines.append(f"| {label} | (未完成) | | | |" if x is None else
                         f"| {label} | {x['n_units']} | {fmt(x['r2cb_speed'])} | {fmt(x['unit_median_r2'])} | "
                         f"{fmt(x['vector_rmse_m_s'])} | {fmt(x['direction_cosine_casemean'])} |")
    if "E3_single_fold_model_mean" in res:
        lines.append(f"\nE3 单个折模型平均 {res['E3_single_fold_model_mean']:.4f}（{res['E3_n_models']} 个）。")
    if "full265_ensemble_per_unit" in res:
        lines.append("\nfull265 集成逐例 R²：" + "；".join(f"{u} {v:.3f}" for u, v in res["full265_ensemble_per_unit"].items()))
    lines.append("")
    res["rows"] = [{"model": label, **({} if x is None else {k: v for k, v in x.items() if k != "cohorts"})} for label, x in rows]
    out["B"][target] = res


def section_history(lines: list, out: dict):
    lines += ["## 历史读数（不同评估集，只并列、不相减）", "",
              "| 目标 | v5.0 train138 → test34（seed 1234 / 7 / 2025，三 seed 均值） | v5.1 cv3 折外 seed 1234（f0 / f1 / f2） |",
              "|---|---|---|"]
    hist = {}
    for t in TARGETS:
        v50 = [stored_r2(ROOT / "training_wss_min/runs" / HIST_V50[t][s] / "eval/ckpt_best/metrics.json", t) for s in SEEDS]
        v51 = [stored_r2(ROOT / "training_wss_min/runs" / HIST_V51[t].format(k=k) / "eval/ckpt_best/metrics.json", t) for k in range(3)]
        a = [x["r2cb"] for x in v50 if x]
        b = [x["r2cb"] for x in v51 if x]
        hist[t] = {"v50_test34": a, "v51_cv3": b}
        lines.append(f"| {t} | {' / '.join(f'{x:.4f}' for x in a)}（{np.mean(a):.4f}） | {' / '.join(f'{x:.3f}' for x in b)} |")
    lines += ["", "PF6 是压力（全部查询点）R²_cb，VF6 是速率 R²_cb。v5.0 是 RCR 面积修正前的 172 例数据，v5.1 cv3 每折只有约 113 个训练单元。", ""]
    out["history"] = hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best", choices=("best", "last"))
    a = ap.parse_args()
    out = {"checkpoint": a.checkpoint, "A": {}, "B": {}}
    checks = Checks()
    lines = [f"# PF6 / VF6 v5.2d 重训读数（ckpt_{a.checkpoint}）", "",
             "在 v5.2d 上建立体场基线，不设过门判据。recover8 只有 8 例，只作描述。定义见 "
             "`training_wss_min/tools/report_pf6vf6_v52d_retrain.py` 文件头；矩阵与读数规则见 "
             f"`training_wss_min/configs/{NAME}/matrix.json`。", "",
             "已知数据限制：150 个每步迭代提前退出的单元（AG 105、AAA 25、ILO 20）标签保持原样（用户 10-01 决定先不动）。", "",
             "## A：CV5 折外", ""]
    for t in TARGETS:
        section_cv5(t, a.checkpoint, lines, out, checks)
    lines += ["## B：full265 → recover8" + ("，E3（CV5 折模型）与 E5（已部署 v5.0 权重）" if a.checkpoint == "best" else ""), ""]
    for t in TARGETS:
        section_recover8(t, a.checkpoint, lines, out, checks)
    section_history(lines, out)
    worst = checks.worst()
    lines += ["## 一致性检查", "",
              f"每个 run 用保存的预测重算自身测试集 R²_cb，与 metrics.json 对比：{len(checks.rows)} 个 run，最大绝对差 {fmt(worst, 8)}"
              + ("（通过）" if np.isfinite(worst) and worst < 1e-5 else "（**超过 1e-5，读数不可用，先查原因**）" if np.isfinite(worst) else ""), ""]
    out["consistency"] = {"n_runs": len(checks.rows), "max_abs_diff": worst, "rows": checks.rows}
    text = "\n".join(lines) + "\n"
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / f"readout_ckpt_{a.checkpoint}.md").write_text(text)
    (EXP / f"readout_ckpt_{a.checkpoint}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(text)


if __name__ == "__main__":
    main()
