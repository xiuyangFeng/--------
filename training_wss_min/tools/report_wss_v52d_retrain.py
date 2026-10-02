"""Readout of the v5.2d retrain (wss_v52d_retrain_20261001; read-only on saved predictions, tolerant of unfinished arms).

Same sections and definitions as report_wss_v52c_retrain (B, A, E3, C / E5; reading rules in
configs/wss_v52d_retrain_20261001/matrix.json), reused with the v5.2d run directories. Section A differs in one point:
ILO/WANG_TIAN_QING-1/after is a new geometry in v5.2d (rebuilt from its own STL), so the v5.2 models' predictions cannot be
re-scored on its new labels. "Old model / new labels" is therefore computed on the units whose wall rows match in both runs,
the new model is also scored on that same unit set for the paired difference, and the excluded units are listed.

    python -m training_wss_min.tools.report_wss_v52d_retrain [--checkpoint best]
    -> experiments/wss_v52d_retrain_20261001/readout_ckpt_<ckpt>.{md,json}
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from training_wss_min.tools import report_wss_v52c_retrain as R

NAME = "wss_v52d_retrain_20261001"
EXP = R.ROOT / "training_wss_min/experiments" / NAME
# the v5.2c sections read these module globals at call time; all 21 v5.2d runs live in one run directory
R.EXP, R.NEW, R.M1 = EXP, R.RUNS / NAME, R.RUNS / NAME
SEEDS, FOLDS = R.SEEDS, R.FOLDS


def section_a(ckpt, lines, out):
    lines += ["## A：X5Dcap_asym2 CV5 折外（261 例）", "",
              "| seed | 折 | 新 Pa R²_cb（全部单元） | 可配对单元数 | 新（可配对） | v5.2 模型 · 旧标签 | v5.2 模型 · 新标签 | Δ（新 − 旧模型新标签） |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    pooled_new, rows = {}, []
    changed_units, row_mismatch = set(), set()
    for s in SEEDS:
        pn, po, pon = {}, {}, {}
        for k in FOLDS:
            rn_rows, ro_rows = {}, {}
            n = R.load_channel(R.pred_dir(R.NEW / f"X5Dcap_asym2_v52cv_f{k}_s{s}", ckpt), rows=rn_rows)
            o = R.load_channel(R.pred_dir(R.OLD_CV / f"X5Dcap_asym2_v52cv_f{k}_s{s}", ckpt), rows=ro_rows)
            if n is None:
                continue
            pn.update(n)
            if o is None:
                continue
            for u, (yn, _) in n.items():
                if u not in o:
                    continue
                yo, po_pred = o[u]
                aligned = len(yo) == len(yn) and (u not in rn_rows or u not in ro_rows or np.array_equal(rn_rows[u], ro_rows[u]))
                if not aligned:
                    row_mismatch.add(u)
                    continue
                if not np.array_equal(yo, yn):
                    changed_units.add(u)
                po[u] = o[u]
                pon[u] = (yn, po_pred)
        done = sum((R.NEW / f"X5Dcap_asym2_v52cv_f{k}_s{s}/eval/ckpt_{ckpt}/predictions/test/manifest.json").is_file() for k in FOLDS)
        if not pn:
            lines.append(f"| {s} | 0/5 | (未完成) | | | | | |")
            continue
        pooled_new[s] = pn
        rn = R.r2cb(pn)
        common = sorted(set(pn) & set(pon))
        rnc = R.r2cb({u: pn[u] for u in common}) if common else None
        ro = R.r2cb({u: po[u] for u in common}) if common else None
        ron = R.r2cb({u: pon[u] for u in common}) if common else None
        lines.append(f"| {s} | {done}/5 | {rn:.4f} | {len(common)} | {R.fmt(rnc)} | {R.fmt(ro)} | {R.fmt(ron)} | "
                     f"{'—' if ron is None else f'{rnc - ron:+.4f}'} |")
        out["A"][f"s{s}"] = dict(folds_done=done, n_units=len(pn), r2cb=rn, n_paired_units=len(common), r2cb_paired_units=rnc,
                                 r2cb_v52_model_old_truth=ro, r2cb_v52_model_new_truth=ron)
        if done == 5:
            rows.append(out["A"][f"s{s}"])
    if rows:
        vals = [r["r2cb"] for r in rows]
        lines.append("")
        lines.append(f"完成 {len(rows)} 个 seed：均值 {np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else float('nan'):.4f}"
                     "（v5.2 三 seed 各自 pooled 0.7750 / 0.7707 / 0.7684，均值 0.7714，v5.2 标签、261 个单元口径）")
        out["A"]["seed_mean"] = float(np.mean(vals))
    if len(pooled_new) == 3 and all(out["A"][f"s{s}"]["folds_done"] == 5 for s in SEEDS):
        en = R.ensemble(list(pooled_new.values()))
        out["A"]["ensemble3"] = dict(r2cb=R.r2cb(en), n_units=len(en))
        lines.append(f"三 seed 折外集成：Pa R²_cb **{R.r2cb(en):.4f}**（n = {len(en)}）")
    out["A"]["units_with_changed_truth"] = sorted(changed_units)
    out["A"]["units_with_different_wall_rows"] = sorted(row_mismatch)
    lines += ["", f"标签变化的单元（新旧真值逐点不同）：{', '.join(sorted(changed_units)) or '无'}；"
                  f"壁面点不同、不在可配对单元里：{', '.join(sorted(row_mismatch)) or '无'}", ""]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="best"); a = ap.parse_args()
    out = {"checkpoint": a.checkpoint, "B": {}, "A": {}, "E3": {}, "C": {}, "E5": {}}
    lines = [f"# v5.2d 重训读数（ckpt_{a.checkpoint}）", "",
             "recover8 只有 8 例，只作描述、不设门。表里的「修正前」指 v5.2 / v5.2p4 数据上同配方同 seed 的模型。", ""]
    R.section_b(a.checkpoint, lines, out)
    section_a(a.checkpoint, lines, out)
    if a.checkpoint == "best":  # E3 / E5 are evaluated at the best checkpoint only
        R.section_e3(a.checkpoint, lines, out)
    R.section_c(a.checkpoint, lines, out)
    out.pop("_B_members", None)
    text = "\n".join(lines) + "\n"
    (EXP / f"readout_ckpt_{a.checkpoint}.md").write_text(text)
    (EXP / f"readout_ckpt_{a.checkpoint}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(text)


if __name__ == "__main__":
    main()
