"""逐帧 R²：T-null / T0 / TB8。

收缩期：主射血段，入口 Q 从快速上升（帧 10，步 1140）回到舒张基线（帧 41，步 1202）。
峰值期：Q ≥ 0.80 Qpeak（帧 17–27，步 1154–1174），含峰值帧 21 / 步 1162。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
EXP = ROOT / "training_wss_min/experiments/wss_time_ecc_20260918"
RUNS = ROOT / "training_wss_min/runs"
OUT = Path(__file__).resolve().parent

fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Droid Sans Fallback"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

wf = json.loads((EXP / "offline/protocol_inlet_waveform_v51.json").read_text())
steps = np.asarray(wf["steps"], int)
q = np.asarray(wf["q_norm"], float)
peak_idx = int(wf["peak_index"])

# 生理窗口（按本协议入口波形，不是 D5 的碎片加速集）
systole = np.arange(10, 42)          # 1140–1202
peak = np.flatnonzero(q >= 0.80)     # 1154–1174
trough = np.sort(np.argsort(q)[:20])
accel = np.arange(10, 17)            # 上升、尚未到 Q=0.8
late = np.arange(28, 42)             # 峰值后射血减速


def r2_from_metrics(arm: str, fold: int) -> np.ndarray:
    p = RUNS / f"wss_time_ecc_20260918/{arm}_f{fold}_s1234/eval/ckpt_best/metrics.json"
    curve = json.loads(p.read_text())["test"]["frame_curve"]
    by_step = {int(row["step"]): float(row["r2_casebalanced_physical"]) for row in curve}
    return np.array([by_step[int(s)] for s in steps])


def r2_tnull(fold: int) -> np.ndarray:
    d2 = json.loads((EXP / "offline/d2_tnull.json").read_text())
    return np.asarray(d2[f"fold{fold}"]["B_scale"]["frame_r2cb_pa"], float)


def fold_stack(fn):
    a = np.stack([fn(k) for k in range(3)], 0)
    return a.mean(0), a.std(0, ddof=1)


tnull_m, tnull_s = fold_stack(r2_tnull)
t0_m, t0_s = fold_stack(lambda k: r2_from_metrics("T0", k))
tb8_m, tb8_s = fold_stack(lambda k: r2_from_metrics("TB8", k))
series = [
    ("T-null", tnull_m, tnull_s, "#333333"),
    ("T0", t0_m, t0_s, "#1f77b4"),
    ("TB8", tb8_m, tb8_s, "#2ca02c"),
]


def shade(ax):
    ax.axvspan(steps[systole[0]] - 1, steps[systole[-1]] + 1, color="#4C78A8", alpha=0.13, zorder=0)
    ax.axvspan(steps[peak[0]] - 1, steps[peak[-1]] + 1, color="#E45756", alpha=0.22, zorder=0)
    ax.axvline(steps[peak_idx], color="#C62828", ls="--", lw=1.0, zorder=2)
    # 主谷底段（连续的低 Q）
    main_trough = trough[trough >= 43]
    ax.axvspan(steps[main_trough[0]] - 1, steps[main_trough[-1]] + 1, color="#7B6C9A", alpha=0.10, zorder=0)


def phase_mean(y, idx):
    return float(np.asarray(y)[idx].mean())


# source
hdr = "step,frame,q_norm,in_systole,in_peak,in_trough,Tnull_mean,Tnull_sd,T0_mean,T0_sd,TB8_mean,TB8_sd"
lines = [hdr]
sys_set, peak_set, tr_set = set(systole), set(peak.tolist()), set(trough.tolist())
for i in range(81):
    lines.append(
        f"{int(steps[i])},{i},{q[i]:.6f},{int(i in sys_set)},{int(i in peak_set)},{int(i in tr_set)},"
        f"{tnull_m[i]:.6f},{tnull_s[i]:.6f},{t0_m[i]:.6f},{t0_s[i]:.6f},{tb8_m[i]:.6f},{tb8_s[i]:.6f}"
    )
(OUT / "frame_r2_source.csv").write_text("\n".join(lines) + "\n")

summary = {
    "windows": {
        "systole_steps": [int(steps[i]) for i in systole],
        "peak_steps": [int(steps[i]) for i in peak],
        "trough_frames": [int(i) for i in trough],
        "peak_index": peak_idx,
        "peak_step": int(steps[peak_idx]),
    },
    "cycle_mean": {n: float(m.mean()) for n, m, *_ in series},
    "peak_frame": {n: float(m[peak_idx]) for n, m, *_ in series},
    "phase_r2": {
        n: {
            "accel": phase_mean(m, accel),
            "peak": phase_mean(m, peak),
            "late_ejection": phase_mean(m, late),
            "trough20": phase_mean(m, trough),
            "cycle": float(m.mean()),
        }
        for n, m, *_ in series
    },
}
(OUT / "phase_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

phase_handles = [
    Patch(facecolor="#4C78A8", alpha=0.35, label="Systole"),
    Patch(facecolor="#E45756", alpha=0.50, label="Peak (Q>=0.8)"),
    Patch(facecolor="#7B6C9A", alpha=0.30, label="Trough"),
]

# fig1
fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=170)
shade(ax)
for name, m, s, c in series:
    ax.plot(steps, m, color=c, lw=2.1, label=name, zorder=3)
    ax.fill_between(steps, m - s, m + s, color=c, alpha=0.16, zorder=2)
ax.set_ylim(-0.08, 0.90)
ax.set_xlim(steps[0], steps[-1])
ax.set_xlabel("Time step (1120-1280, dt=10 ms)")
ax.set_ylabel("Per-frame physical R2_cb")
ax2 = ax.twinx()
ax2.plot(steps, q, color="#888888", lw=1.0, ls=":", label="Q/Qpeak")
ax2.set_ylabel("Inlet Q / Qpeak")
ax2.set_ylim(-0.05, 1.38)
h, lab = ax.get_legend_handles_labels()
ax.legend(h + phase_handles, lab + [p.get_label() for p in phase_handles],
          loc="lower left", frameon=False, ncol=2, fontsize=8)
ax.set_title("Peak-frame R2 is high; cycle mean is pulled down by late ejection and trough")
fig.tight_layout()
fig.savefig(OUT / "fig1_r2_vs_timestep.png")
fig.savefig(OUT / "fig1_r2_vs_timestep.svg")
plt.close(fig)

# fig2
fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=170)
labels = ["Accel", "Peak", "Late ej.", "Trough", "Cycle"]
idxs = [accel, peak, late, trough, np.arange(81)]
x = np.arange(len(labels))
w = 0.24
for i, (name, m, s, c) in enumerate(series):
    vals = [phase_mean(m, idx) for idx in idxs]
    ax.bar(x + (i - 1) * w, vals, w, color=c, label=name, zorder=3)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean per-frame R2_cb")
ax.set_ylim(0, 0.85)
ax.legend(frameon=False)
ax.set_title("All three arms are OK at peak; trough is where cycle R2 dies")
fig.tight_layout()
fig.savefig(OUT / "fig2_r2_by_phase.png")
fig.savefig(OUT / "fig2_r2_by_phase.svg")
plt.close(fig)

# fig3
fig, ax = plt.subplots(figsize=(9.4, 4.4), dpi=170)
shade(ax)
ax.axhline(0, color="k", lw=0.7)
ax.plot(steps, t0_m - tnull_m, color="#1f77b4", lw=2.1, label="T0 - T-null")
ax.plot(steps, tb8_m - tnull_m, color="#2ca02c", lw=2.1, label="TB8 - T-null")
ax.set_ylim(-0.20, 0.20)
ax.set_xlim(steps[0], steps[-1])
ax.set_xlabel("Time step")
ax.set_ylabel("Delta R2_cb vs T-null")
ax.legend(frameon=False, loc="lower left")
ax.set_title("T0 gains off-peak; TB8 is at or below the no-train waveform baseline")
fig.tight_layout()
fig.savefig(OUT / "fig3_delta_vs_tnull.png")
fig.savefig(OUT / "fig3_delta_vs_tnull.svg")
plt.close(fig)

# fig4
fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=170)
for name, m, s, c in series:
    ax.scatter([m[peak_idx]], [m.mean()], s=90, color=c, zorder=3, label=name)
    ax.annotate(name, (m[peak_idx], m.mean()), textcoords="offset points", xytext=(7, 6))
ax.plot([0.55, 0.76], [0.55, 0.76], ls=":", color="#888")
ax.set_xlim(0.55, 0.76)
ax.set_ylim(0.40, 0.58)
ax.set_xlabel("Peak-frame 1162 R2_cb")
ax.set_ylabel("Cycle mean of 81 frame R2_cb")
ax.set_title("If every frame matched the peak, points would sit on the line")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig4_peak_vs_cycle.png")
fig.savefig(OUT / "fig4_peak_vs_cycle.svg")
plt.close(fig)

print(json.dumps(summary["phase_r2"], indent=2))
print("wrote", OUT)
