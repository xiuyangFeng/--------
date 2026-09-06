#!/usr/bin/env python3
"""V4 seed1234 16 臂 milestone (epoch 1000/2500/5000/7500/10000) test35 诊断曲线。

只读各 run 目录下 evaluation_official_{epoch_0xxxx,last_converged}_full.json，
不训练、不改 checkpoint。test35 在此仅用于 epoch 诊断，不反选 checkpoint。
在本目录运行：python3 plot_v4_milestones.py
"""
from __future__ import annotations
import json, os, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = Path("/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/volume_uvwp_bc_rcr_v4")
CKS = [("epoch_01000", 1000), ("epoch_02500", 2500), ("epoch_05000", 5000), ("epoch_07500", 7500), ("last_converged", 10000)]
MODES = ["DATA", "DATA+BC", "BC+PDE-F", "BC+PDE-EMA"]
MODE_SUFFIX = {"DATA": "DATA", "DATA+BC": "BC", "BC+PDE-F": "BC-PDE-F", "BC+PDE-EMA": "BC-PDE-EMA"}
MODE_COLOR = {"DATA": "#3B7DD8", "DATA+BC": "#E8763A", "BC+PDE-F": "#2BAA82", "BC+PDE-EMA": "#E5A419"}  # validated: CVD ΔE ≥ 10
PANELS = [("SP", "PN", "准稳态 · PointNet"), ("SP", "PNPP", "准稳态 · PointNet++"), ("TR", "PN", "瞬态 · PointNet"), ("TR", "PNPP", "瞬态 · PointNet++")]
SURF, INK, INK2, GRID, BAD = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c4443c"

# CJK font fallback
import glob
for f in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
cjk = [f.name for f in font_manager.fontManager.ttflist if any(k in f.name for k in ("CJK", "Hei", "Han", "WenQuanYi", "PingFang", "Song"))]
if cjk:
    plt.rcParams["font.family"] = [cjk[0], "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(tm: str, bb: str, mode: str, ck: str):
    sub = "steady_peak" if tm == "SP" else "transient_autograd"
    p = ROOT / sub / f"V4-{tm}-{bb}-{MODE_SUFFIX[mode]}-s1234" / f"evaluation_official_{ck}_full.json"
    if not p.is_file():
        return None
    d = json.load(open(p))
    s = d["secondary_case_balanced"]
    def r2(k):
        v = s.get(k, {}).get("r2")
        return v.get("mean") if isinstance(v, dict) else v
    return {"E": d["primary"]["e_rel_l2_case_balanced"], "speed": r2("speed"), "p": r2("pressure"),
            "nw": r2("near_wall_speed"), "core": r2("core_speed")}


rows = []
for tm, bb, _ in PANELS:
    for mode in MODES:
        for ck, ep in CKS:
            r = load(tm, bb, mode, ck)
            if r:
                rows.append({"time_mode": tm, "backbone": bb, "mode": mode, "checkpoint": ck, "epoch": ep, **r})
with open(HERE / "milestone_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xticks([1000, 2500, 5000, 7500, 10000])
    ax.set_xticklabels(["1000", "2500", "5000", "7500", "10000"])
    ax.set_xlim(500, 11800)


def series(tm, bb, mode, key):
    xs, ys = [], []
    for r in rows:
        if (r["time_mode"], r["backbone"], r["mode"]) == (tm, bb, mode) and r[key] is not None:
            xs.append(r["epoch"]); ys.append(r[key])
    return xs, ys


def draw(ax, tm, bb, key, ylabel, ref, ref_label):
    style(ax)
    ax.axhline(ref, color=BAD, ls="--", lw=1.0)
    for mode in MODES:
        xs, ys = series(tm, bb, mode, key)
        if not xs:
            continue
        ax.plot(xs, ys, color=MODE_COLOR[mode], lw=1.8, marker="o", ms=4.5, mfc=SURF, mew=1.6)
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        if mode == "DATA":  # start label only for the DATA arm to avoid collisions
            ax.annotate(f"{ys[0]:.3f}", (xs[0], ys[0]), xytext=(-6, 0), textcoords="offset points", fontsize=8, color=INK, va="center", ha="right")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.text(0.99, 0.02 if key == "E" else 0.97, ref_label, transform=ax.transAxes, fontsize=8, color=BAD, ha="right", va="bottom" if key == "E" else "top")


fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.5), facecolor=SURF)
for j, (tm, bb, title) in enumerate(PANELS):
    draw(axes[j], tm, bb, "speed", "speed R²_cb（越大越好）", 0.0, "R² = 0：均值预测器")
    axes[j].set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    axes[j].set_ylim(-0.65, 0.40)
    axes[j].set_xlabel("epoch（milestone checkpoint）", color=INK2, fontsize=9)
handles = [plt.Line2D([], [], color=MODE_COLOR[m], lw=1.8, marker="o", ms=4.5, mfc=SURF, mew=1.6, label=m) for m in MODES]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=4, frameon=False, fontsize=9)
fig.suptitle("图9 · milestone 补评 speed R²_cb：DATA 臂 epoch 1000 是全矩阵最好点，之后单调退化；物理臂几乎不退化但 ceiling 更低",
             x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
fig.text(0.01, 0.005, "test35 official 协议；瞬态评估宇宙 = 每例 1609–5296 点 × 81 帧，准稳态 = 全体域，TR 与 SP 不可直接比。"
         "test35 在此只做 epoch 诊断（development-exposed），不反选 checkpoint。seed1234 单 seed。E_rel_l2 仍写在 milestone_metrics.csv。",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.90))
fig.savefig(HERE / "fig9_milestone_E_and_speedR2.png", dpi=160)

# fig10: steady near-wall speed R2 and pressure R2 by milestone
fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), facecolor=SURF)
for j, (tm, bb, title) in enumerate(PANELS[:2]):
    draw(axes[j], tm, bb, "nw", "near-wall speed R²_cb", 0.0, "R² = 0")
    axes[j].set_title(title + " · 近壁速度 R²", loc="left", fontsize=11, color=INK, pad=8)
    axes[j].set_ylim(-2.6, 0.2)
for j, (tm, bb, title) in enumerate(PANELS[:2]):
    draw(axes[2 + j], tm, bb, "p", "pressure R²_cb", 0.0, "R² = 0")
    axes[2 + j].set_title(title + " · 压力 R²", loc="left", fontsize=11, color=INK, pad=8)
    axes[2 + j].set_ylim(0.0, 0.6)
for ax in axes:
    ax.set_xlabel("epoch（milestone checkpoint）", color=INK2, fontsize=9)
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=4, frameon=False, fontsize=9)
fig.suptitle("图10 · 准稳态 milestone：DATA 臂近壁随训练持续恶化（−1.4 → −2.4），BC/PDE 臂近壁「更好」只是预测趋零；EMA 全程牺牲压力",
             x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
fig.tight_layout(rect=(0, 0.0, 1, 0.92))
fig.savefig(HERE / "fig10_milestone_nearwall_pressure.png", dpi=160)
print("saved", len(rows), "rows")
