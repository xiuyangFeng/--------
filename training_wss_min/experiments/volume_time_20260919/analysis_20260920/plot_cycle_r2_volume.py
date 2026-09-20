"""体场（压力 / 速度）全周期逐帧 R²_cb 可视化，与 WSS 线 `wss_time_ecc_20260918/analysis_20260919` 同口径。

数据源全部是已有产物，不重新推理：
  * 模型逐帧曲线 = 各臂 `eval/ckpt_{best,last}/metrics.json` 的 `cycle.frame_r2cb`（81 帧，固定子采样点集）
  * 基线逐帧曲线 = `offline/a3_tnull.json` 的 `tnull` / `peak_freeze`（同一套子采样点）
  * 入口波形 = WSS 线冻结的 `protocol_inlet_waveform_v51.json`（本批 A0 已验证 81 帧步序一致、峰值帧索引 21）

与 WSS 线的图相比多了一条 `peak_freeze`（峰值帧直接冻住、完全不缩放）。它是判断
「这个目标的周期结构值不值得学」的关键对照：速度上它几乎追平 T-null，压力上它塌到 0。

    python -m training_wss_min.experiments.volume_time_20260919.analysis_20260920.plot_cycle_r2_volume
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
EXP = ROOT / "training_wss_min/experiments/volume_time_20260919"
RUNS = ROOT / "training_wss_min/runs/volume_time_20260919"
WAVEFORM = ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/offline/protocol_inlet_waveform_v51.json"
OUT = Path(__file__).resolve().parent

fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Droid Sans Fallback"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

wf = json.loads(WAVEFORM.read_text())
steps = np.asarray(wf["steps"], int)
q = np.asarray(wf["q_norm"], float)
peak_idx = int(wf["peak_index"])

systole = np.arange(10, 42)           # 1140–1202 主射血段
peak = np.flatnonzero(q >= 0.80)      # 1154–1174，含峰值帧 21 / 步 1162
trough = np.sort(np.argsort(q)[:20])  # 谷底四分位，与 volume_time.trough_frames 同口径
accel = np.arange(10, 17)
late = np.arange(28, 42)

a3 = json.loads((EXP / "offline/a3_tnull.json").read_text())
gate = json.loads((EXP / "gate_report.json").read_text())

# 每个目标画哪几条：T-null / peak_freeze 是不训练的基线，T0 / TB(K*) 是本批代表臂
TARGETS = {
    "pressure": dict(unit="Pa", base="PF6_v51", t0="PT0", tb="PTB8", k=8,
                     title="Pressure (p_rel, Pa)"),
    "velocity": dict(unit="m/s", base="VF6_v51", t0="VT0", tb="VTB4", k=4,
                     title="Velocity (3 components, m/s)"),
}
COLORS = {"T-null": "#333333", "peak_freeze": "#9467bd", "T0": "#1f77b4", "TB": "#2ca02c"}


def baseline_curve(target: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.stack([np.asarray(a3[target][f"fold{k}"]["arms"][arm]["frame_r2cb"], float) for k in range(3)])
    return a.mean(0), a.std(0, ddof=1)


def model_curve(arm: str, checkpoint: str = "best") -> tuple[np.ndarray, np.ndarray]:
    a = np.stack([np.asarray(json.loads(
        (RUNS / f"{arm}_f{k}_s1234/eval/ckpt_{checkpoint}/metrics.json").read_text()
    )["test"]["cycle"]["frame_r2cb"], float) for k in range(3)])
    return a.mean(0), a.std(0, ddof=1)


def series_for(target: str, checkpoint: str = "best") -> list[tuple]:
    cfg = TARGETS[target]
    tn_m, tn_s = baseline_curve(target, "tnull")
    pf_m, pf_s = baseline_curve(target, "peak_freeze")
    t0_m, t0_s = model_curve(cfg["t0"], checkpoint)
    tb_m, tb_s = model_curve(cfg["tb"], checkpoint)
    return [("T-null", tn_m, tn_s, COLORS["T-null"], "-"),
            ("peak_freeze", pf_m, pf_s, COLORS["peak_freeze"], "--"),
            (cfg["t0"], t0_m, t0_s, COLORS["T0"], "-"),
            (cfg["tb"], tb_m, tb_s, COLORS["TB"], "-")]


def shade(ax):
    ax.axvspan(steps[systole[0]] - 1, steps[systole[-1]] + 1, color="#4C78A8", alpha=0.13, zorder=0)
    ax.axvspan(steps[peak[0]] - 1, steps[peak[-1]] + 1, color="#E45756", alpha=0.22, zorder=0)
    ax.axvline(steps[peak_idx], color="#C62828", ls="--", lw=1.0, zorder=2)
    main = trough[trough >= 43]
    ax.axvspan(steps[main[0]] - 1, steps[main[-1]] + 1, color="#7B6C9A", alpha=0.10, zorder=0)


def phase_mean(y, idx) -> float:
    return float(np.asarray(y)[idx].mean())


# T-null / peak_freeze 拿的是**真值**峰值场，峰值帧上 R²=1 是构造出来的，不是预测能力
ORACLE = ("T-null", "peak_freeze")
ORACLE_NOTE = "T-null / peak_freeze start from the ground-truth peak field:\ntheir R2=1 at step 1162 is by construction, not prediction."

PHASES = [("Accel", accel), ("Peak", peak), ("Late ej.", late), ("Trough", trough), ("Cycle", np.arange(81))]
phase_handles = [Patch(facecolor="#4C78A8", alpha=0.35, label="Systole"),
                 Patch(facecolor="#E45756", alpha=0.50, label="Peak (Q>=0.8)"),
                 Patch(facecolor="#7B6C9A", alpha=0.30, label="Trough")]

summary: dict = {"windows": {"peak_index": peak_idx, "peak_step": int(steps[peak_idx]),
                             "systole_steps": [int(steps[i]) for i in systole],
                             "peak_steps": [int(steps[i]) for i in peak],
                             "trough_frames": [int(i) for i in trough]},
                 "source": "cycle.frame_r2cb (fixed subsample) / a3_tnull.json; three-fold mean",
                 "targets": {}}
rows = ["target,checkpoint,arm,step,frame,q_norm,in_systole,in_peak,in_trough,r2cb_mean,r2cb_sd"]
sys_set, peak_set, tr_set = set(systole.tolist()), set(peak.tolist()), set(trough.tolist())

# ---- fig1: 逐帧曲线（两个目标各一行）--------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9.8, 8.6), dpi=170, sharex=True)
for ax, (target, cfg) in zip(axes, TARGETS.items()):
    shade(ax)
    for name, m, s, c, ls in series_for(target):
        ax.plot(steps, m, color=c, lw=2.0, ls=ls, label=name, zorder=3)
        ax.fill_between(steps, m - s, m + s, color=c, alpha=0.14, zorder=2)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylabel("Per-frame R2_cb")
    ax.axhline(0, color="k", lw=0.6, zorder=1)
    ax2 = ax.twinx()
    ax2.plot(steps, q, color="#888888", lw=1.0, ls=":", zorder=1)
    ax2.set_ylabel("Inlet Q / Qpeak")
    ax2.set_ylim(-0.05, 1.55)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + phase_handles, lab + [p.get_label() for p in phase_handles],
              loc="lower left", frameon=False, ncol=3, fontsize=8)
    ax.set_title(cfg["title"])
axes[-1].set_xlabel("Time step (1120-1280, dt=10 ms)")
axes[0].text(0.995, 0.03, ORACLE_NOTE, transform=axes[0].transAxes, ha="right", va="bottom",
             fontsize=7.5, color="#444444")
fig.suptitle("Volume full-cycle: pressure needs the time model, velocity does not", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "fig1_r2_vs_timestep.png")
fig.savefig(OUT / "fig1_r2_vs_timestep.svg")
plt.close(fig)

# ---- fig2: 分相位柱状 -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=170)
for ax, (target, cfg) in zip(axes, TARGETS.items()):
    s = series_for(target)
    x = np.arange(len(PHASES))
    w = 0.2
    for i, (name, m, _sd, c, _ls) in enumerate(s):
        ax.bar(x + (i - 1.5) * w, [phase_mean(m, idx) for _n, idx in PHASES], w, color=c, label=name, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in PHASES])
    ax.set_ylabel("Mean per-frame R2_cb")
    ax.set_ylim(-0.55, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title(cfg["title"])
fig.suptitle("Trough is where the cycle mean is decided", y=0.99)
fig.tight_layout()
fig.savefig(OUT / "fig2_r2_by_phase.png")
fig.savefig(OUT / "fig2_r2_by_phase.svg")
plt.close(fig)

# ---- fig3: 相对 T-null 的逐帧增量 -------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9.8, 7.4), dpi=170, sharex=True)
for ax, (target, cfg) in zip(axes, TARGETS.items()):
    shade(ax)
    s = series_for(target)
    tn = s[0][1]
    for name, m, _sd, c, ls in s[1:]:
        ax.plot(steps, m - tn, color=c, lw=2.0, ls=ls, label=f"{name} - T-null", zorder=3)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylabel("Delta R2_cb vs T-null")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title(cfg["title"])
axes[-1].set_xlabel("Time step")
fig.suptitle("Where the trained time heads actually beat the no-train waveform baseline", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "fig3_delta_vs_tnull.png")
fig.savefig(OUT / "fig3_delta_vs_tnull.svg")
plt.close(fig)

# ---- fig4: 峰值帧 vs 全周期散点 ----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), dpi=170)
for ax, (target, cfg) in zip(axes, TARGETS.items()):
    for name, m, _sd, c, _ls in series_for(target):
        oracle = name in ORACLE
        ax.scatter([m[peak_idx]], [m.mean()], s=95, zorder=3,
                   facecolor="none" if oracle else c, edgecolor=c, linewidths=2.0)
        ax.annotate(name + (" (oracle peak)" if oracle else ""), (m[peak_idx], m.mean()),
                    textcoords="offset points", xytext=(-6, -14 if name == "peak_freeze" else 9),
                    fontsize=8, ha="right" if oracle else "left")
    lo, hi = 0.40, 1.06
    ax.plot([lo, hi], [lo, hi], ls=":", color="#888")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.06, 0.80)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Peak-frame 1162 R2_cb (subsample)")
    ax.set_ylabel("Cycle mean of 81 frame R2_cb")
    ax.set_title(cfg["title"])
axes[0].text(0.03, 0.03, ORACLE_NOTE, transform=axes[0].transAxes, fontsize=7.5, color="#444444")
fig.suptitle("Hollow = starts from the true peak field; distance below the diagonal = cost of the off-peak frames",
             y=0.99)
fig.tight_layout()
fig.savefig(OUT / "fig4_peak_vs_cycle.png")
fig.savefig(OUT / "fig4_peak_vs_cycle.svg")
plt.close(fig)

# ---- fig5: 体场特有——自相似性诊断 -------------------------------------------
# peak_freeze 与 T-null 的差 = 「共同时间缩放 r(t)」到底买到了多少。
# 速度上这条差几乎贴 0 → 峰值场本身就够用，周期结构没有可学的东西。
fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=170)
shade(ax)
for target, cfg in TARGETS.items():
    tn, _ = baseline_curve(target, "tnull")
    pf, _ = baseline_curve(target, "peak_freeze")
    ax.plot(steps, tn - pf, lw=2.1,
            color="#C62828" if target == "pressure" else "#1565C0",
            label=f"{target}: T-null - peak_freeze")
ax.axhline(0, color="k", lw=0.7)
ax.set_xlim(steps[0], steps[-1])
ax.set_xlabel("Time step")
ax.set_ylabel("R2_cb gained by the common time scaling r(t)")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Self-similarity check: scaling the peak field buys a lot for pressure, almost nothing for velocity")
fig.tight_layout()
fig.savefig(OUT / "fig5_self_similarity.png")
fig.savefig(OUT / "fig5_self_similarity.svg")
plt.close(fig)

# ---- fig6: PT0 的 best/last 差异是单折的 checkpoint 失稳，不是系统性差别 --------
# 三折平均会把它抹平：fold1/fold2 两个 checkpoint 逐帧几乎重合，全部 0.056 的周期差来自 fold0。
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=170, sharey=True)
for k, ax in enumerate(axes):
    shade(ax)
    tn = np.asarray(a3["pressure"][f"fold{k}"]["arms"]["tnull"]["frame_r2cb"], float)
    ax.plot(steps, tn, color=COLORS["T-null"], lw=1.4, label="T-null")
    for checkpoint, ls in (("best", "--"), ("last", "-")):
        curve = np.asarray(json.loads(
            (RUNS / f"PT0_f{k}_s1234/eval/ckpt_{checkpoint}/metrics.json").read_text()
        )["test"]["cycle"]["frame_r2cb"], float)
        ax.plot(steps, curve, color=COLORS["T0"], lw=2.0, ls=ls,
                label=f"PT0 {checkpoint} (cycle {curve.mean():.3f})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time step")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.set_title(f"fold{k}")
axes[0].set_ylabel("Per-frame R2_cb")
fig.suptitle("PT0 best vs last: folds 1/2 are identical; the whole 0.056 cycle gap is one fold blowing up "
             "at the trough", y=0.99)
fig.tight_layout()
fig.savefig(OUT / "fig6_pt0_best_vs_last.png")
fig.savefig(OUT / "fig6_pt0_best_vs_last.svg")
plt.close(fig)

# ---- 数据落盘 ---------------------------------------------------------------
for target, cfg in TARGETS.items():
    block = {"unit": cfg["unit"], "k_star": gate["families"][target]["k_star"], "checkpoints": {}}
    for checkpoint in ("best", "last"):
        s = series_for(target, checkpoint)
        block["checkpoints"][checkpoint] = {
            name: {"cycle": float(m.mean()), "peak_frame": float(m[peak_idx]),
                   **{key.lower().replace(" ", "_").replace(".", ""): phase_mean(m, idx) for key, idx in PHASES}}
            for name, m, _sd, _c, _ls in s}
        for name, m, sd, _c, _ls in s:
            for i in range(81):
                rows.append(f"{target},{checkpoint},{name},{int(steps[i])},{i},{q[i]:.6f},"
                            f"{int(i in sys_set)},{int(i in peak_set)},{int(i in tr_set)},"
                            f"{m[i]:.6f},{sd[i]:.6f}")
    summary["targets"][target] = block
(OUT / "frame_r2_source.csv").write_text("\n".join(rows) + "\n")
(OUT / "phase_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

for target in TARGETS:
    print(f"== {target} (best) ==")
    for name, block in summary["targets"][target]["checkpoints"]["best"].items():
        print(f"  {name:12s} accel {block['accel']:+.3f} peak {block['peak']:+.3f} "
              f"late {block['late_ej']:+.3f} trough {block['trough']:+.3f} cycle {block['cycle']:+.3f}")
print("wrote", OUT)
