"""Report for the bottleneck-transformer matrix (teacher method adapted) against the R4 baseline on test34.

Statistics contract (fixed after the 2026-09-08 adversarial review found the first version unsound):

* The reference is the MEAN of R4's three seeds, not its best seed. Using r4_s1234 (0.5721) as the reference
  biases every arm's delta down by ~0.016, because that seed is the max of {0.5721, 0.5565, 0.5394}.
* The noise band is recomputed here from those three runs, PER METRIC, and it is a band on a DIFFERENCE:
  sd(difference of two independent means) = sd_seed * sqrt(1/n_arm + 1/n_base). The earlier version compared a
  single-seed difference against one arm's own half-range (0.016); two of the three identical-config R4 seed
  pairs already exceed that, so it would have labelled pure seed noise as a real effect roughly half the time.
* A single-seed arm gets NO verdict. Its 95% band is +/-1.96*sd*sqrt(2) (about +/-0.045 physical), wider than any
  effect this method is predicted to produce (tracking doc section 11.2: ideal ceiling +0.009..+0.056 normalized).
* The DECISION metric is normalized R2_cb, because that is the space the effect size was pre-registered in
  (tracking doc section 11.2: ideal ceiling +0.009..+0.056 normalized) and because it is far quieter
  (seed sd 0.0033 vs 0.0124). Physical R2_cb is always reported next to it and is used only as a consistency
  check: an arm whose physical delta points the other way AND clears its own band contradicts the decision.
  Requiring the noisier metric to also clear its (3-4x wider) band would let it dictate a test it cannot resolve.
  This follows the preferred option of the 2026-09-08 review finding on decision metrics; the rule was fixed
  before the extended-seed results were read.
* With 11 arms tested at 95%, expect ~0.5 false positives by chance: read a single arm's verdict together with
  whether the controls stay flat and whether sibling arms move the same way.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "v5_rerun_20260906" / "outputs"
CFGS = ROOT / "configs" / "v5_rerun_20260906"
EXP_DIR = ROOT / "experiments" / "v5_rerun_20260906"
SEEDS = (1234, 7, 2025, 3407, 7919)  # 3407/7919 仅用于 BT-3 与 R4 的配对补种；缺失的 run 自动跳过

BASE_ARM = ("R4 基准（无注意力，32 粗 token）", "r4_lsa2_h2_logradius_v5feat")
ARMS = [
    ("对照·密度：125 粗 token，无注意力", "bt0_r4_c125_ctrl"),
    ("对照·容量：125 token + SA3 局部 Transformer（无全局）", "btc_r4_c125_localattn"),
    ("对照·条件：125 token，只有条件 MLP（layers=0，无注意力）", "bt6_r4_c125_condonly"),
    ("BT-1 我们已有的 coarse_attention（32 token）", "bt1_r4_coarseattn"),
    ("BT-2 瓶颈 Transformer（32 token）", "bt2_r4_bottleneck"),
    ("BT-3 瓶颈 Transformer（125 token）★主候选", "bt3_r4_bottleneck_c125"),
    ("BT-4 瓶颈 ×2 层（125 token）", "bt4_r4_bottleneck_c125_l2"),
    ("BT-5 导师条件方式：绝对 xyz MLP", "bt5_r4_bottleneck_c125_xyzpos"),
    ("BT-5b 关闭相对几何偏置", "bt5b_r4_bottleneck_c125_nogeobias"),
    ("BT-7 导师残差强度 γ=1.0", "bt7_r4_bottleneck_c125_res1"),
    ("BT-8 导师 dropout=0.1", "bt8_r4_bottleneck_c125_drop01"),
]
PHYS = ("field_casebalanced", "r2")
NORM = ("normalized", "field_casebalanced", "r2")


def verdict_for(dp, dn, n, n_base, sd_p, sd_n, is_base=False):
    """Single implementation of the decision rule, shared by the report and the workbook backfill.

    Decision metric is normalized R2_cb (pre-registered space, seed sd ~1/4 of physical); physical is a
    consistency check that can only overturn the call when it points the other way AND clears its own band.
    """
    if is_base:
        return "参照（全部 seed 均值）", None, None
    bandp = 1.96 * sd_p * math.sqrt(1 / n + 1 / n_base)
    bandn = 1.96 * sd_n * math.sqrt(1 / n + 1 / n_base)
    if n < 2:
        return (f"单 seed，不判定（需超 ±{1.96*sd_p*math.sqrt(2):.3f}/±{1.96*sd_n*math.sqrt(2):.3f}）", bandp, bandn)
    contradicts = abs(dp) > bandp and (dp > 0) != (dn > 0)
    if abs(dn) > bandn and not contradicts:
        side = "提升" if dn > 0 else "下降"
        note = "物理同向" if (dp > 0) == (dn > 0) else "物理方向相反但未超带"
        return (f"**{side}**（归一化超带 ±{bandn:.4f}；{note} {dp:+.4f}，其带 ±{bandp:.3f}）", bandp, bandn)
    if contradicts:
        return "两指标方向相反且物理超带，判为未定", bandp, bandn
    return f"噪声内（归一化带 ±{bandn:.4f}）", bandp, bandn


def dig(d, path):
    for k in path:
        if d is None:
            return None
        d = d.get(k)
    return d


def load(stem: str, seed: int, ck: str = "best"):
    p = RUNS / f"{stem}_s{seed}" / "eval" / f"ckpt_{ck}" / "metrics.json"
    if not p.is_file():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return m.get("test", m)


def seeds_of(stem: str, ck: str = "best"):
    out = {}
    for s in SEEDS:
        m = load(stem, s, ck)
        if m is not None:
            out[s] = m
    return out


def f(x, d=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(v) else f"{v:.{d}f}"


def gammas(stem: str, seed: int):
    """Read the learned residual scales out of the checkpoint: a null result with gamma still ~1e-3 means the
    attention branch never grew, which is a different conclusion from 'global context does not help'."""
    p = RUNS / f"{stem}_s{seed}" / "ckpt_best.pt"
    if not p.is_file():
        return None
    try:
        import torch
        sd = torch.load(p, map_location="cpu")["model"]
    except Exception:
        return None
    vals = {k: float(v) for k, v in sd.items() if "bottleneck" in k and "gamma" in k}
    return vals or None


def params_of(stem: str, seed: int):
    p = RUNS / f"{stem}_s{seed}" / "train.log"
    if not p.is_file():
        return "—"
    m = re.search(r"params=([\d.]+M)", p.read_text(encoding="utf-8", errors="ignore"))
    return m.group(1) if m else "—"


def cfg_of(stem: str, seed: int):
    p = CFGS / f"{stem}_s{seed}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main() -> int:
    base = seeds_of(BASE_ARM[1])
    if not base:
        print("R4 baseline runs not found"); return 1
    bp = np.array([dig(m, PHYS) for m in base.values()], dtype=float)
    bn = np.array([dig(m, NORM) for m in base.values()], dtype=float)
    n_base = len(bp)
    sd_p = float(bp.std(ddof=1)) if n_base > 1 else float("nan")
    sd_n = float(bn.std(ddof=1)) if n_base > 1 else float("nan")
    print(f"### 噪声标定（R4 同一配置 {n_base} 个 seed {sorted(base)}）")
    print(f"物理 R²_cb：{' / '.join(f(v) for v in bp)}  → 均值 {f(bp.mean())}，seed 标准差 {f(sd_p)}")
    print(f"归一化 R²_cb：{' / '.join(f(v) for v in bn)}  → 均值 {f(bn.mean())}，seed 标准差 {f(sd_n)}")
    print(f"单 seed 对单 seed 之差的 95% 带：物理 ±{f(1.96*sd_p*math.sqrt(2))}，归一化 ±{f(1.96*sd_n*math.sqrt(2))}")
    print(f"n seed 均值 vs {n_base} seed 参照之差的 95% 带（归一化）：n=3 时 ±{f(1.96*sd_n*math.sqrt(1/3+1/n_base))}，n=5 时 ±{f(1.96*sd_n*math.sqrt(1/5+1/n_base))}")
    print(f"参照 = R4 {n_base} seed 均值（不是最好的那个 seed）。**判据指标 = 归一化 R²_cb**（预登记空间，且 seed sd 只有物理的 1/4）；")
    print("物理 R²_cb 一并报出作一致性检查：只有当它方向相反且超过自己的带时才推翻判定。11 个臂做 95% 检验，期望约 0.5 个假阳性，")
    print("因此单个臂的判定要结合“对照臂是否保持平坦”和“同族臂是否同向”一起读。\n")

    print("### 主表（vs R4 全部 seed 的均值）")
    print("| 臂 | seed 数 | 物理 R²_cb（均值±半幅） | Δ物理 | 归一化 R²_cb | Δ归一化 | 参数量 | 判定 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    rows = [(BASE_ARM[0] + "（参照）", BASE_ARM[1])] + ARMS
    detail = []
    for label, stem in rows:
        runs = base if stem == BASE_ARM[1] else seeds_of(stem)
        if not runs:
            print(f"| {label} | 0 | (未完成) | | | | | |"); continue
        ap = np.array([dig(m, PHYS) for m in runs.values()], dtype=float)
        an = np.array([dig(m, NORM) for m in runs.values()], dtype=float)
        n = len(ap)
        dp, dn = ap.mean() - bp.mean(), an.mean() - bn.mean()
        half_p = (ap.max() - ap.min()) / 2 if n > 1 else float("nan")
        if stem == BASE_ARM[1]:
            verdict = "参照"
        else:
            verdict, _, _ = verdict_for(dp, dn, n, n_base, sd_p, sd_n)
        seed0 = sorted(runs)[0]
        print(f"| {label} | {n} | {f(ap.mean())}{'' if n<2 else ' ± '+f(half_p,3)} | {dp:+.4f} | {f(an.mean())} | {dn:+.4f} | {params_of(stem, seed0)} | {verdict} |")
        detail.append((label, stem, runs, ap, an))
    print()

    print("### 诊断（best ckpt，seed 最小的一次；γ 仍停在初值说明注意力分支没被学起来，与“全局上下文没用”是两回事）")
    print("| 臂 | 粗 token | 注意力 | token 条件 | γ_attn / γ_ffn（学后） | high-WSS R² | top10 比 | top10 IoU |")
    print("| --- | ---: | --- | --- | --- | ---: | ---: | ---: |")
    for label, stem, runs, ap, an in detail:
        seed0 = sorted(runs)[0]; m = runs[seed0]; c = cfg_of(stem, seed0)
        mc = (c or {}).get("model", {})
        tok = mc.get("sa_center_counts", [None, None, None])[-1]
        if mc.get("bottleneck_transformer"):
            L = mc.get("bottleneck_layers", 1)
            attn = f"瓶颈 ×{L}（{mc.get('bottleneck_heads',8)} 头）" if L else "无（只有条件 MLP）"
            cond = {"input_features": "输入几何特征 MLP", "xyz": "绝对 xyz MLP", "none": "无"}.get(mc.get("bottleneck_pos_enc"), "?")
            if mc.get("bottleneck_geo_bias"):
                cond += " + 相对几何偏置"
        elif mc.get("coarse_attention"):
            attn, cond = f"coarse_attention（{mc.get('coarse_attention_heads',4)} 头）", "相对几何偏置"
        elif tuple(mc.get("local_transformer_stages", ())) == (2, 3):
            attn, cond = "SA2+SA3 局部 Transformer（无全局）", "—"
        else:
            attn, cond = "无", "—"
        g = gammas(stem, seed0)
        gtxt = "—"
        if g:
            ga = [v for k, v in sorted(g.items()) if "attention" in k]
            gf = [v for k, v in sorted(g.items()) if "ffn" in k]
            gtxt = f"{'/'.join(f'{v:.3f}' for v in ga)} · {'/'.join(f'{v:.3f}' for v in gf)}"
        print(f"| {label} | {tok} | {attn} | {cond} | {gtxt} | {f(dig(m,('regional_field','high_wss','r2')))} | "
              f"{f(dig(m,('calibration','top10_pred_true_ratio')),3)} | {f(dig(m,('hotspot','top10_iou_casemean')),3)} |")
    print()

    print("### 分域 R²_cb（seed 均值）")
    print("| 臂 | AG | AAA | ILO |")
    print("| --- | ---: | ---: | ---: |")
    for label, stem, runs, ap, an in detail:
        g = {k: np.mean([dig(m, ("group_casebalanced", k, "r2")) for m in runs.values()]) for k in ("AG", "AAA", "ILO")}
        print(f"| {label} | {f(g['AG'])} | {f(g['AAA'])} | {f(g['ILO'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
