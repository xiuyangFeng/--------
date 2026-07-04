#!/usr/bin/env python3
"""从 M-E oracle JSON 生成一页汇总 Markdown（G5 汇报包 · 0 重训）。"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WAVE = REPO / "outputs/field/f0_decision/v3p_me_oracle_wave_20260626.json"
DEFAULT_FOLLOWUP = REPO / "outputs/field/f0_decision/v3p_me_oracle_followup_20260626.json"
DEFAULT_EJ = REPO / "outputs/field/f0_decision/v3p_me_ej_oracle_20260629.json"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verdict_emoji(v: str) -> str:
    v = (v or "").lower()
    if v in {"go", "pass", "weak_go"}:
        return "⚠️" if "weak" in v else "✅"
    return "❌"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=Path, default=DEFAULT_WAVE)
    ap.add_argument("--followup", type=Path, default=DEFAULT_FOLLOWUP)
    ap.add_argument("--ej", type=Path, default=DEFAULT_EJ)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    wave = _load(args.wave)
    follow = _load(args.followup) if args.followup.is_file() else {}
    ej = _load(args.ej) if args.ej.is_file() else {}

    baseline = wave.get("baseline_wss_r2_wss", 0.429)
    oracles: Dict[str, Any] = wave.get("oracles", {})
    rows: List[str] = []
    for oid in sorted(oracles, key=lambda x: (len(x), x)):
        o = oracles[oid]
        direction = o.get("direction", "")
        verdict = o.get("verdict", "n/a")
        key_metric = ""
        for k in (
            "pattern_r2_pooled_probe",
            "pooled_r2",
            "test_r2",
            "mean_topk_dice",
            "holdout_r2",
            "delta_r2",
        ):
            if k in o and o[k] is not None:
                key_metric = f"{k}={o[k]}"
                break
        rows.append(
            f"| {oid} | {direction} | {key_metric or '—'} | {_verdict_emoji(verdict)} {verdict} |"
        )

    follow_lines = []
    if follow:
        for key in ("O9_full", "O3_input_coeff", "O11_calib"):
            if key not in follow.get("oracles", {}):
                continue
            b = follow["oracles"][key]
            follow_lines.append(
                f"- **{key}** ({b.get('direction', '')}): verdict={b.get('verdict')} · "
                f"holdout/metric={b.get('mean_holdout_r2', b.get('coef_r2', b.get('delta_r2', '—')))}"
            )

    ej_block = ""
    if ej:
        km = ej.get("key_metrics", {})
        pc = ej.get("pressure_gradient_corr", {})
        ej_block = f"""
## E-J 压力梯度桥接（2026-06-29 · weak_go）

| 指标 | 值 |
| --- | ---: |
| \\|∇p\\| Spearman | {pc.get('mean_spearman_gradmag', '—')} |
| \\|∇p\\|+geom GBDT test R² | {km.get('gradp_mag_geom_test_r2', '—')} |
| \\|∇p\\|+geom pooled R² | {km.get('gradp_mag_geom_pooled_r2', '—')} |
| dp/ds Spearman | {pc.get('mean_spearman_neg_dpds', '—')} |
| ΔR² vs 几何 | {km.get('delta_r2_gradp_vs_geom', '—')} |
| 判读 | **{ej.get('verdict', 'weak_go')}** → GPU probe Job **5854** |
"""

    out_path = args.output or (
        REPO / "outputs/field/f0_decision" / f"V3P_ME_oracle_onepager_{date.today():%Y%m%d}.md"
    )
    md = f"""# V3P M-E Oracle 一页汇总

> 口径：V3P · `split_AG_v1` · 基线 I6-diag **best_wss {baseline:.3f}** · 禁止与 V3D/4957 混表
> 数据源：`{args.wave.name}` · `{args.followup.name}` · `{args.ej.name}`

## O1–O12 波次（2026-06-26）

| Oracle | 方向 | 关键指标 | 判读 |
| --- | --- | --- | --- |
{chr(10).join(rows)}

**汇总**：Go/weak **≤2** · **No-Go ≥10** → AsymW/结构性微调 **封口**

## 补跑（5816/5817）

{chr(10).join(follow_lines) if follow_lines else '（见 followup JSON）'}

{ej_block}

## 当前队列（2026-06-29 计划重排）

1. **E-J GPU probe** 收结果 vs **0.429**（Go ≥ **0.449** + L3 不退化）
2. **G4-c Phase 0** 调参（occ ≥ **0.50** · gap ≤ **0.10**）
3. **G5 叙事收口**（本页 + postview + 路径 G §8）

**不排队**：I6-a seed2/3 · AsymW 微调 · E-B direct vel_diff · O9/POD/标定 GPU
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
