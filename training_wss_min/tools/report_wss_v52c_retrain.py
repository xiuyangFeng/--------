"""Readout of the v5.2c retrain (wss_v52c_retrain_20260930; read-only on saved predictions, tolerant of unfinished arms).

Sections (reading rules in configs/wss_v52c_retrain_20260930/matrix.json; recover8 n = 8 -> descriptive, no gates):
  B   X5Dcap_asym2 full265 -> recover8: per seed / three-seed Pa-mean ensemble, R2_cb (unit-balanced) and per-unit median,
      paired with the pre-label-fix full265 runs (job 16192, same seeds); the recover8 truth of both must be identical.
  A   X5Dcap_asym2 CV5 out-of-fold over 261: per seed pooled R2_cb, mean +- sd, three-seed OOF ensemble; references = the
      pre-label-fix CV5 runs (wss_v52_phys2m_20260927) scored on their own (old) truth and, where the wall rows match, on
      the corrected truth (old model / new labels -> isolates the effect of training on corrected labels).
  E3  the 15 CV5 fold models on recover8: per-seed five-fold ensemble and 15-model ensemble, vs B.
  C   M1cap three-head full265 -> recover8, per channel (peak WSS, TAWSS, OSI) per seed and ensemble; guard = peak channel
      normalized R2_cb vs the same-seed B run (drop <= 0.02); E5 = the deployed M1 (v5.1 train136) on the same units.

    python -m training_wss_min.tools.report_wss_v52c_retrain [--checkpoint best]
    -> experiments/wss_v52c_retrain_20260930/readout_ckpt_<ckpt>.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training_wss_min import metrics as M

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
RUNS = ROOT / "training_wss_min/runs"
EXP = ROOT / "training_wss_min/experiments/wss_v52c_retrain_20260930"
NEW = RUNS / "wss_v52c_labelfix_20260930"
M1 = RUNS / "wss_v52c_retrain_20260930"
OLD_FULL = RUNS / "wss_v52p4_full265_20260930"
OLD_CV = RUNS / "wss_v52_phys2m_20260927"
SEEDS = (1234, 7, 2025)
FOLDS = range(5)
OSI_THR = (0.1, 0.3)
TAWSS_LOW = 0.4
STAG_TAWSS, STAG_OSI = 0.4, 0.1


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def r2cb(units: dict) -> float:
    """Unit-balanced R2 (definition of experiments/wss_v52_20260923/analyze_wss_v52.py)."""
    gm = np.mean([y.mean() for y, _ in units.values()])
    res = np.mean([np.mean((y - p) ** 2) for y, p in units.values()])
    tot = np.mean([np.mean((y - gm) ** 2) for y, _ in units.values()])
    return float(1.0 - res / tot)


def load_channel(pred_root: Path, channel: str | None = None, rows: dict | None = None) -> dict | None:
    """{unit: (true_pa, pred_pa)} from an evaluate --save-predictions test directory (channel subdir for the three-head).
    ``rows`` (optional dict) receives each unit's wall row_index so that two runs can be aligned row by row."""
    base = pred_root / channel if channel else pred_root
    man = base / "manifest.json"
    if not man.is_file():
        return None
    out = {}
    for c in json.loads(man.read_text())["cases"]:
        with np.load(base / c["file"]) as z:
            out[c["unit_id"]] = (z["true_pa"].astype(np.float64), z["pred_pa"].astype(np.float64))
            if rows is not None and "row_index" in z.files:
                rows[c["unit_id"]] = z["row_index"].copy()
    return out


def pred_dir(run: Path, ckpt: str) -> Path:
    return run / f"eval/ckpt_{ckpt}/predictions/test"


def ensemble(members: list[dict]) -> dict:
    units = sorted(set.intersection(*(set(m) for m in members)))
    return {u: (members[0][u][0], np.mean([m[u][1] for m in members], axis=0)) for u in units}


def unit_median(units: dict) -> float:
    return float(np.median([r2(y, p) for y, p in units.values()]))


def norm_r2cb(run_or_eval: Path, ckpt: str | None = None) -> float | None:
    path = run_or_eval / f"eval/ckpt_{ckpt}/metrics.json" if ckpt else run_or_eval / "metrics.json"
    if not path.is_file():
        return None
    t = json.loads(path.read_text())["test"]
    return float(t.get("normalized", t)["field_casebalanced"]["r2"])


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------------------------------------------------
def section_b(ckpt, lines, out):
    lines += ["## B：X5Dcap_asym2 full265 → recover8", "",
              "| seed | Pa R²_cb | 归一化 R²_cb | 逐例中位 | 修正前同 seed Pa R²_cb | Δ |", "|---|---:|---:|---:|---:|---:|"]
    new, old = {}, {}
    for s in SEEDS:
        n = load_channel(pred_dir(NEW / f"X5Dcap_asym2_full265_s{s}", ckpt))
        o = load_channel(pred_dir(OLD_FULL / f"X5Dcap_asym2_full265_s{s}", ckpt))
        if n is None:
            lines.append(f"| {s} | (未完成) | | | | |")
            continue
        new[s] = n
        if o is not None:
            old[s] = o
            for u in n:  # recover8 labels were not touched by the label fix
                if u in o and not np.array_equal(n[u][0], o[u][0]):
                    raise RuntimeError(f"recover8 truth differs from the pre-label-fix run: {u}")
        rn, ro = r2cb(n), (r2cb(o) if o else None)
        delta = "—" if ro is None else f"{rn - ro:+.4f}"
        lines.append(f"| {s} | {rn:.4f} | {fmt(norm_r2cb(NEW / f'X5Dcap_asym2_full265_s{s}', ckpt))} | {unit_median(n):.3f} | "
                     f"{fmt(ro)} | {delta} |")
        out["B"][f"s{s}"] = dict(r2cb=rn, r2cb_pre_fix=ro, unit_median=unit_median(n))
    if len(new) == 3:
        en = ensemble(list(new.values()))
        row = dict(r2cb=r2cb(en), unit_median=unit_median(en), per_unit={u: r2(*v) for u, v in en.items()})
        if len(old) == 3:
            eo = ensemble(list(old.values()))
            row.update(r2cb_pre_fix=r2cb(eo), unit_median_pre_fix=unit_median(eo),
                       per_unit_delta={u: r2(*en[u]) - r2(*eo[u]) for u in en})
        out["B"]["ensemble3"] = row
        lines += ["", f"三 seed 集成：Pa R²_cb **{row['r2cb']:.4f}**、逐例中位 {row['unit_median']:.3f}"
                  + (f"；修正前 {row['r2cb_pre_fix']:.4f} / {row['unit_median_pre_fix']:.3f}（Δ {row['r2cb'] - row['r2cb_pre_fix']:+.4f}，"
                     f"逐例改善 {sum(d > 0 for d in row['per_unit_delta'].values())}/8）" if "r2cb_pre_fix" in row else ""), "",
                  "| 单元 | 集成 R² | 修正前 | Δ |", "|---|---:|---:|---:|"]
        for u, v in row["per_unit"].items():
            d = row.get("per_unit_delta", {}).get(u)
            lines.append(f"| {u} | {v:.3f} | {fmt(v - d if d is not None else None, 3)} | {'' if d is None else f'{d:+.3f}'} |")
        out["_B_members"] = new
    lines.append("")


def section_a(ckpt, lines, out):
    lines += ["## A：X5Dcap_asym2 CV5 折外（261 例）", "",
              "| seed | 折 | 新 Pa R²_cb | 修正前（旧标签） | 修正前模型 · 新标签 | Δ（新 − 旧模型新标签） |", "|---|---|---:|---:|---:|---:|"]
    pooled_new, rows = {}, []
    changed_units, row_mismatch = set(), set()
    for s in SEEDS:
        pn, po, pon = {}, {}, {}
        for k in FOLDS:
            rn_rows, ro_rows = {}, {}
            n = load_channel(pred_dir(NEW / f"X5Dcap_asym2_v52cv_f{k}_s{s}", ckpt), rows=rn_rows)
            o = load_channel(pred_dir(OLD_CV / f"X5Dcap_asym2_v52cv_f{k}_s{s}", ckpt), rows=ro_rows)
            if n is None:
                continue
            pn.update(n)
            if o is not None:
                po.update(o)
                for u, (yn, _) in n.items():
                    if u not in o:
                        continue
                    yo, po_pred = o[u]
                    aligned = len(yo) == len(yn) and (u not in rn_rows or u not in ro_rows
                                                      or np.array_equal(rn_rows[u], ro_rows[u]))
                    if not aligned:
                        row_mismatch.add(u)
                        continue
                    if not np.array_equal(yo, yn):
                        changed_units.add(u)
                    pon[u] = (yn, po_pred)
        done = sum((NEW / f"X5Dcap_asym2_v52cv_f{k}_s{s}/eval/ckpt_{ckpt}/predictions/test/manifest.json").is_file() for k in FOLDS)
        if not pn:
            lines.append(f"| {s} | 0/5 | (未完成) | | | |")
            continue
        pooled_new[s] = pn
        rn = r2cb(pn)
        common = sorted(set(pn) & set(pon))
        ro = r2cb({u: po[u] for u in common}) if len(common) == len(pn) else None
        ron = r2cb({u: pon[u] for u in common}) if len(common) == len(pn) else None
        lines.append(f"| {s} | {done}/5 | {rn:.4f} | {fmt(ro)} | {fmt(ron)} | {'—' if ron is None else f'{rn - ron:+.4f}'} |")
        out["A"][f"s{s}"] = dict(folds_done=done, n_units=len(pn), r2cb=rn, r2cb_pre_fix_old_truth=ro, r2cb_pre_fix_new_truth=ron)
        if done == 5:
            rows.append(out["A"][f"s{s}"])
    if rows:
        vals = [r["r2cb"] for r in rows]
        lines.append("")
        lines.append(f"完成 {len(rows)} 个 seed：均值 {np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else float('nan'):.4f}"
                     "（修正前三 seed 各自 pooled 0.7750 / 0.7707 / 0.7684，均值 0.7714，旧标签口径）")
        out["A"]["seed_mean"] = float(np.mean(vals))
    if len(pooled_new) == 3 and all(out["A"][f"s{s}"]["folds_done"] == 5 for s in SEEDS):
        en = ensemble(list(pooled_new.values()))
        out["A"]["ensemble3"] = dict(r2cb=r2cb(en), n_units=len(en))
        lines.append(f"三 seed 折外集成：Pa R²_cb **{r2cb(en):.4f}**（n = {len(en)}）")
    out["A"]["units_with_changed_truth"] = sorted(changed_units)
    out["A"]["units_with_different_wall_rows"] = sorted(row_mismatch)
    lines += ["", f"标签变化的单元（新旧真值逐点不同）：{', '.join(sorted(changed_units)) or '无'}；"
                  f"壁面点数不同、未计入「旧模型 · 新标签」：{', '.join(sorted(row_mismatch)) or '无'}", ""]


def section_e3(ckpt, lines, out):
    lines += ["## E3：CV5 的 15 个折模型 → recover8", "", "| 集成 | 模型数 | Pa R²_cb | 逐例中位 |", "|---|---:|---:|---:|"]
    by_seed = {}
    for s in SEEDS:
        members = [m for k in FOLDS if (m := load_channel(EXP / f"recover8_cv5/X5Dcap_asym2_v52cv_f{k}_s{s}/predictions/test")) is not None]
        if members:
            by_seed[s] = members
            en = ensemble(members)
            lines.append(f"| seed {s} 五折集成 | {len(members)} | {r2cb(en):.4f} | {unit_median(en):.3f} |")
            out["E3"][f"s{s}_folds"] = dict(n_models=len(members), r2cb=r2cb(en), unit_median=unit_median(en),
                                            single=[r2cb(m) for m in members])
    allm = [m for ms in by_seed.values() for m in ms]
    if allm:
        en = ensemble(allm)
        lines.append(f"| 全部 | {len(allm)} | **{r2cb(en):.4f}** | {unit_median(en):.3f} |")
        out["E3"]["all"] = dict(n_models=len(allm), r2cb=r2cb(en), unit_median=unit_median(en),
                                single_mean=float(np.mean([r2cb(m) for m in allm])))
        lines.append(f"单个折模型平均 {out['E3']['all']['single_mean']:.4f}")
        if "ensemble3" in out["B"]:
            lines.append(f"对照 B（full265 三 seed 集成）{out['B']['ensemble3']['r2cb']:.4f}")
    lines.append("")


# ---- three-head ------------------------------------------------------------------------------------------------------
def head_summary(peak: dict, tawss: dict, osi: dict) -> dict:
    units = sorted(set(peak) & set(tawss) & set(osi))
    ta_t, ta_p = [tawss[u][0] for u in units], [tawss[u][1] for u in units]
    os_t, os_p = [osi[u][0] for u in units], [np.clip(osi[u][1], 0.0, 0.5) for u in units]
    out = {"n_units": len(units), "peak_pa_r2cb": r2cb({u: peak[u] for u in units}),
           "tawss_pa_r2cb": float(M.casebalanced_field_metrics(ta_t, ta_p)["r2"]),
           "tawss_ccc_casemed": float(np.nanmedian([M.lin_ccc(t, p) for t, p in zip(ta_t, ta_p)])),
           "osi_r2cb": float(M.casebalanced_field_metrics(os_t, os_p)["r2"]),
           "osi_ccc_casemed": float(np.nanmedian([M.lin_ccc(t, p) for t, p in zip(os_t, os_p)]))}

    def iou(masks):
        v = [np.count_nonzero(tm & pm) / u for tm, pm in masks if (u := np.count_nonzero(tm | pm))]
        return float(np.mean(v)) if v else float("nan")
    out["tawss_low_iou"] = iou([(t < TAWSS_LOW, p < TAWSS_LOW) for t, p in zip(ta_t, ta_p)])
    for T in OSI_THR:
        out[f"osi_iou_{T:g}"] = iou([(t > T, p > T) for t, p in zip(os_t, os_p)])
    out["stagnation_iou"] = iou([((a < STAG_TAWSS) & (o > STAG_OSI), (ap < STAG_TAWSS) & (op > STAG_OSI))
                                 for a, o, ap, op in zip(ta_t, os_t, ta_p, os_p)])
    return out


def load_three(pred_root: Path):
    chans = {ch: load_channel(pred_root, ch) for ch in ("wss", "tawss", "osi")}
    return None if any(v is None for v in chans.values()) else chans


KEYS = ("peak_pa_r2cb", "tawss_pa_r2cb", "tawss_ccc_casemed", "tawss_low_iou", "osi_r2cb", "osi_ccc_casemed",
        "osi_iou_0.1", "osi_iou_0.3", "stagnation_iou")
HEAD = "| 模型 | 峰值 Pa R²_cb | TAWSS Pa R²_cb | TAWSS CCC 中位 | TAWSS<0.4 IoU | OSI R²_cb | OSI CCC 中位 | OSI>0.1 IoU | OSI>0.3 IoU | 滞留区 IoU |"


def section_c(ckpt, lines, out):
    lines += ["## C：三头 M1cap full265 → recover8（E5 = 已部署三头 v5.1）", "", HEAD, "|---|" + "---:|" * len(KEYS)]
    groups = {"C": {s: load_three(pred_dir(M1 / f"M1cap_full265_s{s}", ckpt)) for s in SEEDS},
              "E5": {s: load_three(EXP / f"e5_deployed_m1/M1_s{s}/eval/predictions/test") if ckpt == "best" else None
                     for s in SEEDS}}
    for g, members in groups.items():
        done = {s: m for s, m in members.items() if m is not None}
        for s, m in done.items():
            row = head_summary(m["wss"], m["tawss"], m["osi"])
            out[g][f"s{s}"] = row
            lines.append(f"| {g} s{s} | " + " | ".join(fmt(row[k], 3) for k in KEYS) + " |")
        if len(done) == 3:
            ens = {ch: ensemble([m[ch] for m in done.values()]) for ch in ("wss", "tawss", "osi")}
            row = head_summary(ens["wss"], ens["tawss"], ens["osi"])
            out[g]["ensemble3"] = row
            lines.append(f"| **{g} 三 seed 集成** | " + " | ".join(fmt(row[k], 3) for k in KEYS) + " |")
    if "_B_members" in out:
        eb = ensemble(list(out["_B_members"].values()))
        lines.append(f"| 参照：B 三 seed 集成（峰值） | {r2cb(eb):.3f} |" + " |" * (len(KEYS) - 1))
    lines += ["", "护栏（峰值通道归一化 R²_cb 对同 seed B，掉 ≤ 0.02）："]
    for s in SEEDS:
        a, b = norm_r2cb(M1 / f"M1cap_full265_s{s}", ckpt), norm_r2cb(NEW / f"X5Dcap_asym2_full265_s{s}", ckpt)
        if a is None or b is None:
            lines.append(f"- s{s}：未完成")
            continue
        out["C"].setdefault("guard", {})[f"s{s}"] = dict(peak_norm=a, peak_norm_B=b, delta=a - b, ok=a - b >= -0.02)
        lines.append(f"- s{s}：{a:.4f} 对 {b:.4f}（Δ {a - b:+.4f}，{'过' if a - b >= -0.02 else '不过'}）")
    lines.append("")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="best"); a = ap.parse_args()
    out = {"checkpoint": a.checkpoint, "B": {}, "A": {}, "E3": {}, "C": {}, "E5": {}}
    lines = [f"# v5.2c 重训读数（ckpt_{a.checkpoint}）", "",
             "recover8 只有 8 例，只作描述、不设门。修正前数字来自旧标签训练的同配方同 seed 模型。", ""]
    section_b(a.checkpoint, lines, out)
    section_a(a.checkpoint, lines, out)
    if a.checkpoint == "best":  # E3 / E5 were evaluated at the best checkpoint only
        section_e3(a.checkpoint, lines, out)
    section_c(a.checkpoint, lines, out)
    out.pop("_B_members", None)
    text = "\n".join(lines) + "\n"
    (EXP / f"readout_ckpt_{a.checkpoint}.md").write_text(text)
    (EXP / f"readout_ckpt_{a.checkpoint}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(text)


if __name__ == "__main__":
    main()
