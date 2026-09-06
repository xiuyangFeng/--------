#!/usr/bin/env python3
"""V4 汇报图件重绘（2026-09-03）：图 4 / 图 7 重排版，新增图 11（梯度冲突）与图 12（E 与 R² 的关系）。

只读训练/评估产物；所有输出写入本目录。运行前先执行 extract_grad_cos.py 生成 gradcos_*.csv。
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Circle

# ---------- 字体：统一用 Noto Sans CJK，数学用 dejavusans（保证 R² 与中文同图不缺字） ----------
for f in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc") + glob.glob(
    "/usr/share/fonts/truetype/noto/NotoSansCJK*.ttc"
):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
_avail = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((n for n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC"] if n in _avail), None)
plt.rcParams["font.family"] = [CJK, "DejaVu Sans"] if CJK else ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 100

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; INK3 = "#8a8985"; GRID = "#e6e5e0"
MODE_COLOR = {"DATA": "#2a78d6", "DATA+BC": "#eb6834", "BC+PDE-F": "#1baf7a", "BC+PDE-EMA": "#eda100"}
MODES = ["DATA", "DATA+BC", "BC+PDE-F", "BC+PDE-EMA"]
MODE_KEY = {"DATA": "DATA", "DATA+BC": "BC", "BC+PDE-F": "BC-PDE-F", "BC+PDE-EMA": "BC-PDE-EMA"}
SHORT = {"DATA": "DATA", "DATA+BC": "+BC", "BC+PDE-F": "PDE-F", "BC+PDE-EMA": "EMA"}
BAD = "#c4443c"

ROOT = "/public/newhome/cy/Digital_twin/GNN"
BASE = f"{ROOT}/outputs/wss_pinn/volume_uvwp_bc_rcr_v4"
OUT = os.path.dirname(os.path.abspath(__file__))
MILESTONE_CSV = f"{ROOT}/docs/03-汇报材料/V4汇报/V4_milestone补评_epoch曲线_2026-09-02/milestone_metrics.csv"
FM_JSON = f"{ROOT}/outputs/wss_pinn/audits/v4_workbook_0_14_20260823/field_metrics_v4_0_15.json"
TDIR = {"SP": "steady_peak", "TR": "transient_autograd"}
GROUPS = [("SP", "PN"), ("SP", "PNPP"), ("TR", "PN"), ("TR", "PNPP")]
ARMS16 = [(t, b, m) for t, b in GROUPS for m in MODES]


def run_name(t, b, m):
    return f"V4-{t}-{b}-{MODE_KEY[m]}-s1234"


def style(ax, logy=False):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="both", color=GRID, lw=0.7)
    ax.tick_params(colors=INK2, labelsize=9.5)
    if logy:
        ax.set_yscale("log")


def load_epoch_curve(t, b, m):
    p = f"{BASE}/{TDIR[t]}/{run_name(t, b, m)}/epoch_progress.jsonl"
    ep, dt = [], []
    with open(p) as fh:
        for line in fh:
            d = json.loads(line)
            ep.append(d["epoch"]); dt.append(d["mean_data_total"])
    return np.array(ep), np.array(dt, dtype=float)


def train_loss_at(ep, dt, epoch):
    """milestone 前 100 个 epoch 的 mean_data_total 均值（checkpoint 保存在 completed_epoch == epoch）。"""
    mask = (ep >= epoch - 100) & (ep < epoch)
    if not mask.any():
        mask = ep <= ep.max()
        return float(np.nanmean(dt[-100:]))
    return float(np.nanmean(dt[mask]))


# ---------- 读 milestone / last 指标 ----------
MS = {}
with open(MILESTONE_CSV) as fh:
    for row in csv.DictReader(fh):
        key = (row["time_mode"], row["backbone"], row["mode"])
        MS.setdefault(key, {})[row["checkpoint"]] = {k: float(row[k]) for k in ("E", "speed", "p", "nw", "core")}
FM = json.load(open(FM_JSON))
E_LAST = {a: FM["arms"][run_name(*a)]["e_rel_l2"]["mean"] for a in ARMS16}
R2_SPEED = {a: FM["arms"][run_name(*a)]["r2"]["speed"]["mean"] for a in ARMS16}
CK_ORDER = ["epoch_01000", "epoch_02500", "epoch_05000", "epoch_07500", "last_converged"]
CK_EPOCH = {"epoch_01000": 1000, "epoch_02500": 2500, "epoch_05000": 5000, "epoch_07500": 7500, "last_converged": 10000}

CURVES = {a: load_epoch_curve(*a) for a in ARMS16}


# =====================================================================
# 图 4（重绘）：训练拟合 vs 测试主指标，画成 epoch 1000 → 10000 的轨迹
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharey=False)
fig.patch.set_facecolor(SURF)
ptitle = {"SP": "(a) 准稳态 peak（评估宇宙：全体域 53 万–224 万点）", "TR": "(b) 瞬态 81 帧（评估宇宙：每例 1609–5296 点 × 81 帧）"}
for ax, t in zip(axes, ["SP", "TR"]):
    style(ax)
    ax.axhline(1.0, color=INK2, lw=1.0, ls="--", zorder=1)
    ax.text(0.985, 1.0, "E = 1.0：全零预测器", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=9, color=INK2)
    for b in ["PN", "PNPP"]:
        for m in MODES:
            a = (t, b, m)
            ep, dt = CURVES[a]
            xs, ys = [], []
            for ck in CK_ORDER:
                e = CK_EPOCH[ck]
                x = train_loss_at(ep, dt, e) if ck != "last_converged" else float(np.nanmean(dt[-200:]))
                y = MS[a][ck]["E"] if ck != "last_converged" else E_LAST[a]
                xs.append(x); ys.append(y)
            col = MODE_COLOR[m]
            ls = "-" if b == "PN" else (0, (3, 2))
            ax.plot(xs, ys, color=col, lw=1.6, ls=ls, alpha=0.85, zorder=2)
            ax.scatter(xs[:-1], ys[:-1], s=[14, 18, 22, 26], color=col, alpha=0.75, zorder=3,
                       marker="o" if b == "PN" else "s", edgecolors="none")
            ax.scatter(xs[-1], ys[-1], s=110 if b == "PN" else 80, color=col, zorder=4,
                       marker="o" if b == "PN" else "s", edgecolors=SURF, linewidths=1.4)
            # 起点标注 epoch 1000
            if b == "PN":
                ax.annotate("1000", xy=(xs[0], ys[0]), xytext=(-2, -11), textcoords="offset points",
                            fontsize=7.5, color=col, ha="center")
                ax.annotate(SHORT[m], xy=(xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                            fontsize=9, color=col, va="center", fontweight="bold")
    ax.set_title(ptitle[t], fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel("训练侧 data_total（标准化 MSE，越小 = 训练拟合越好）", color=INK2, fontsize=10)
axes[0].set_ylabel("official test35  $E_{\\mathrm{rel},L2}$（病例等权，越小越好）", color=INK2, fontsize=10)
axes[0].set_xlim(0.12, 0.95); axes[0].set_ylim(0.76, 1.22)
axes[1].set_xlim(0.12, 0.95); axes[1].set_ylim(0.76, 1.22)
# 读图指引
axes[0].annotate("DATA / +BC：训练 loss 一路降，\n测试 E 一路升 —— 记住了训练病例",
                 xy=(0.215, 1.09), xytext=(0.60, 1.115), fontsize=9.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f4", ec="#f0c8c4", lw=0.8))
axes[0].annotate("PDE 臂：几乎不动。物理项只是\n防记忆的正则，天花板更低",
                 xy=(0.80, 0.905), xytext=(0.50, 0.83), fontsize=9.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#eef7f3", ec="#bfe3d2", lw=0.8))
axes[1].annotate("瞬态 DATA 同样从 0.89 退到 1.04", xy=(0.235, 1.03), xytext=(0.36, 1.13), fontsize=9.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f4", ec="#f0c8c4", lw=0.8))
axes[1].annotate("EMA：训练几乎不拟合（u 通道 MSE 0.99），\nE 却 < 1 —— 幅值压到零场附近", xy=(0.83, 0.955), xytext=(0.40, 0.80),
                 fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff9e8", ec="#f2dfa6", lw=0.8))
h1 = [Line2D([], [], marker="o", ls="", color=MODE_COLOR[m], ms=8, label=m) for m in MODES]
h2 = [Line2D([], [], marker="o", ls="-", color=INK2, ms=7, label="PointNet（实线 · 圆）"),
      Line2D([], [], marker="s", ls=(0, (3, 2)), color=INK2, ms=6, label="PointNet++（虚线 · 方）"),
      Line2D([], [], marker="o", ls="", color=INK3, ms=3, label="小点 = epoch 1000 / 2500 / 5000 / 7500，大点 = 10000（last）")]
fig.legend(handles=h1 + h2, loc="lower center", ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("图4 · 训练越拟合、测试越差：16 臂从 epoch 1000 到 10000 的轨迹（无 validation 的代价）",
             fontsize=13, color=INK, x=0.01, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0.07, 1, 0.95])
fig.savefig(f"{OUT}/fig4_overfit_trajectory.png", dpi=170, facecolor=SURF)
plt.close(fig)
print("fig4 done")


# =====================================================================
# 图 7（重绘）：瞬态压力 R² 的离群病例
# =====================================================================
d = json.load(open(f"{BASE}/transient_autograd/V4-TR-PN-BC-PDE-F-s1234/evaluation_official_last_converged_full.json"))
rows = []


def short_case(case_id: str) -> str:
    parts = case_id.split("/")
    return parts[-2] if parts[-1] in ("before", "after") else parts[-1]


for c in d["cases"]:
    m = c["metrics"]["pressure"]
    rows.append(dict(case=c["case_id"], r2=m["r2"], gauge=m["truth_mean"], tvar=m["truth_variance"],
                     rmse=m["rmse"], pvar=m["prediction_variance"]))
rows.sort(key=lambda r: r["r2"])
r2 = np.array([r["r2"] for r in rows]); gauge = np.array([r["gauge"] for r in rows])
tstd = np.sqrt(np.array([r["tvar"] for r in rows])); rmse = np.array([r["rmse"] for r in rows])
outlier = gauge < 5000
mean_all = float(r2.mean()); med_all = float(np.median(r2)); mean_wo = float(r2[~outlier].mean())
n_pos = int((r2 > 0).sum())
print(f"pressure R2: mean_all={mean_all:.2f} median={med_all:.2f} mean_without_2={mean_wo:.2f} n_pos={n_pos}")
for r, o, s, e in zip(rows, outlier, tstd, rmse):
    if o:
        print("  outlier", r["case"], f"gauge={r['gauge']:.0f} Pa  r2={r['r2']:.0f}  truth_std={s:.0f} Pa  rmse={e:.0f} Pa")
print(f"  normal cases: truth_std median={np.median(tstd[~outlier]):.0f} Pa, rmse median={np.median(rmse[~outlier]):.0f} Pa")

fig = plt.figure(figsize=(13.2, 6.6)); fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
ax = fig.add_subplot(gs[0, 0]); style(ax); ax.grid(False, axis="y")
CLIP = -1.0
y = np.arange(len(rows))
vals = np.clip(r2, CLIP, 1.0)
colors = [BAD if o else "#2a78d6" for o in outlier]
ax.barh(y, vals, color=colors, height=0.72, zorder=3)
for i, (r, o) in enumerate(zip(rows, outlier)):
    short = short_case(r["case"])
    if o:
        ax.barh(i, CLIP, color="none", edgecolor=BAD, hatch="////", height=0.72, zorder=3, lw=0.8)
        ax.annotate(f"$R^2$ = {r['r2']:.0f}（真值 gauge 均值 {r['gauge']:.0f} Pa，截断显示）",
                    xy=(0.0, i), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=BAD, va="center",
                    ha="left")
    elif r["r2"] < CLIP:
        ax.annotate(f"{r['r2']:+.2f}（截断显示）", xy=(0.0, i), xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, color=INK2, va="center", ha="left")
    else:
        ax.annotate(f"{r['r2']:+.2f}", xy=(r["r2"], i), xytext=(4 if r["r2"] >= 0 else -4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK2, va="center",
                    ha="left" if r["r2"] >= 0 else "right")
ax.axvline(0, color=INK2, lw=1.0, ls="--", zorder=4)
ax.axvline(med_all, color="#1baf7a", lw=1.4, ls="-", zorder=4)
ax.text(med_all, len(rows) - 0.2, f"中位数 {med_all:+.2f}", color="#1baf7a", fontsize=9, ha="center", va="bottom", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([short_case(r["case"]) for r in rows], fontsize=7.2,
                   color=INK2)
for lab, o in zip(ax.get_yticklabels(), outlier):
    if o:
        lab.set_color(BAD); lab.set_fontweight("bold")
ax.set_xlim(CLIP - 0.05, 1.08); ax.set_ylim(-0.8, len(rows) + 0.6)
ax.set_xlabel("逐病例 pressure $R^2$（x 轴截断于 −1）", color=INK2, fontsize=10)
ax.set_title(f"(a) 35 例逐病例压力 $R^2$（TR-PN-PDE-F）：{n_pos} 例 > 0，中位 {med_all:+.2f}", fontsize=11, color=INK, loc="left", pad=8)
ax.text(0.03, 0.56,
        f"病例等权均值：\n含两例  {mean_all:+.1f}\n去掉两例  {mean_wo:+.2f}\n\n主表里的 −20 是这两例\n拉出来的均值假象",
        transform=ax.transAxes, fontsize=9.5, color=INK, va="center", ha="left", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", fc="#fff5f4", ec="#f0c8c4", lw=0.8), zorder=6)

ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
ax2.set_xscale("log"); ax2.set_yscale("log")
for g_, s_, e_ in zip(gauge, tstd, rmse):
    ax2.plot([g_, g_], [s_, e_], color=INK3, lw=0.7, alpha=0.6, zorder=2)
ax2.scatter(gauge[~outlier], tstd[~outlier], s=48, color="#2a78d6", edgecolors=SURF, linewidths=1.0, zorder=3, label="真值压力标准差 σ（其余 33 例）")
ax2.scatter(gauge[outlier], tstd[outlier], s=90, color=BAD, edgecolors=SURF, linewidths=1.0, zorder=4, label="真值压力标准差 σ（两例 gauge 口径异常）")
ax2.scatter(gauge, rmse, s=30, marker="x", color=INK, zorder=4, label="模型 RMSE（同一病例，竖线相连）")
for r, o, s, e in zip(rows, outlier, tstd, rmse):
    if o:
        left = r["gauge"] < 400
        ax2.annotate(f"{short_case(r['case'])}\nσ = {s:.0f} Pa\nRMSE = {e:.0f} Pa",
                     xy=(r["gauge"], e), xytext=(-6 if left else 6, 10), textcoords="offset points", fontsize=8.5,
                     color=BAD, ha="right" if left else "left", va="bottom")
ax2.axvspan(12500, 16500, color="#2a78d6", alpha=0.07, zorder=1)
ax2.text(14400, 130, "全库主流\n13–16 kPa", fontsize=8.5, color="#2a78d6", ha="center", va="bottom")
ax2.set_xlim(120, 40000); ax2.set_ylim(100, 60000)
ax2.set_xlabel("病例真值 gauge 压力均值（Pa，log）", color=INK2, fontsize=10)
ax2.set_ylabel("压力标准差 σ 与模型 RMSE（Pa，log）", color=INK2, fontsize=10)
ax2.set_title("(b) 为什么会 −552：$R^2 = 1 - \\mathrm{RMSE}^2/\\sigma^2$；两例的 RMSE ≈ 13 kPa\n（模型按全库 13–16 kPa 水平预测）而 σ 只有 0.5–1 kPa", fontsize=10.5, color=INK, loc="left", pad=8)
ax2.legend(frameon=False, fontsize=8.3, loc="upper left")
fig.suptitle("图7 · 瞬态压力 R̄² = −20 的真相：两例 gauge 口径异常病例拉爆均值，其余 33 例大多良好",
             fontsize=13, color=INK, x=0.01, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/fig7_pressure_outliers_v2.png", dpi=170, facecolor=SURF)
plt.close(fig)
print("fig7 done")


# =====================================================================
# 图 11：no-slip 梯度与数据梯度反向（cos ≈ −0.9）
# =====================================================================
def load_gradcos(run):
    p = f"{OUT}/gradcos_{run}.csv"
    ep, cos_ns, cos_mom, gnd, gnorm, clip, nsr = [], [], [], [], [], [], []
    with open(p) as fh:
        for row in csv.DictReader(fh):
            ep.append(int(row["epoch"])); cos_ns.append(float(row["cos_grad_data_no_slip"]))
            cos_mom.append(float(row["cos_grad_data_momentum"])); gnd.append(float(row["grad_norm_data"]))
            gnorm.append(float(row["gradient_norm"])); clip.append(row["gradient_clip_flag"] == "True")
            nsr.append(float(row["no_slip_raw"]))
    return dict(ep=np.array(ep), cos_ns=np.array(cos_ns), cos_mom=np.array(cos_mom), gnd=np.array(gnd),
                gnorm=np.array(gnorm), clip=np.array(clip), nsr=np.array(nsr))


def rolling_median(x, w=101):
    out = np.empty_like(x, dtype=float)
    h = w // 2
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - h): i + h + 1])
    return out


GC_ARMS = [("V4-SP-PN-BC-s1234", "准稳态 · DATA+BC", MODE_COLOR["DATA+BC"], "-"),
           ("V4-SP-PN-BC-PDE-F-s1234", "准稳态 · BC+PDE-F", MODE_COLOR["BC+PDE-F"], "-"),
           ("V4-SP-PN-BC-PDE-EMA-s1234", "准稳态 · BC+PDE-EMA", MODE_COLOR["BC+PDE-EMA"], "-"),
           ("V4-TR-PN-BC-s1234", "瞬态 · DATA+BC", MODE_COLOR["DATA+BC"], (0, (3, 2))),
           ("V4-TR-PN-BC-PDE-F-s1234", "瞬态 · BC+PDE-F", MODE_COLOR["BC+PDE-F"], (0, (3, 2)))]
GC = {run: load_gradcos(run) for run, *_ in GC_ARMS}
stats_rows = []
for run, lab, *_ in GC_ARMS:
    g = GC[run]
    last = g["ep"] >= g["ep"].max() - 500
    first = g["ep"] < 200
    stats_rows.append((lab, float(np.median(g["cos_ns"][first])), float(np.median(g["cos_ns"][last])),
                       float(np.mean(g["cos_ns"][last] < -0.8)), float(np.median(g["gnd"][last])),
                       float(np.mean(g["clip"][last])), float(np.median(g["nsr"][last]))))
    print(f"{lab}: cos median first200={stats_rows[-1][1]:+.3f} last500={stats_rows[-1][2]:+.3f} "
          f"frac<-0.8={stats_rows[-1][3]:.2f} |grad_data|={stats_rows[-1][4]:.2f} clip={stats_rows[-1][5]:.2f} no_slip={stats_rows[-1][6]:.3f}")

fig = plt.figure(figsize=(14.5, 9.6)); fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1], wspace=0.22, hspace=0.36)

# (a) 向量示意：cos = −0.9 时两梯度对拉，合力短，再被 clip 截断
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(SURF); ax.set_aspect("equal")
ax.set_xlim(-2.3, 2.6); ax.set_ylim(-2.35, 1.85); ax.axis("off")


def arrow(ax, start, vec, color, label, lw=2.4, offset=(0, 0), fs=9.5, ha="center", va="center"):
    end = (start[0] + vec[0], start[1] + vec[1])
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, zorder=4))
    ax.text(end[0] + offset[0], end[1] + offset[1], label, color=color, fontsize=fs, ha=ha, va=va, zorder=5)


theta = np.arccos(-0.9)
gd = np.array([1.9, 0.0])                       # ∇L_data，范数 ≈ 3 用 1.9 示意
gn = 1.45 * np.array([np.cos(theta), np.sin(theta)])  # ∇L_no-slip，与 data 夹角 154°
gsum = gd + gn
o = (-0.1, 0.15)
arrow(ax, o, gd, MODE_COLOR["DATA"], "∇L_data\n（拟合 CFD 标签：近壁速度非零）", offset=(0.05, -0.20), ha="center", va="top")
arrow(ax, o, gn, MODE_COLOR["DATA+BC"], "∇L_no-slip\n（壁面速度 → 0）", offset=(-0.05, 0.14), ha="center", va="bottom")
arrow(ax, o, gsum, INK, "合力（真正用于更新）", lw=2.8, offset=(0.12, 0.18), ha="left", va="bottom")
ax.add_patch(Circle(o, 0.55, fill=False, ls=":", ec=INK3, lw=1.0, zorder=2))
ax.text(o[0] + 0.02, o[1] - 0.66, "grad clip = 1.0（BC 臂 82% 的 step 被截断）", color=INK3, fontsize=8.5, ha="center", va="top")
ax.text(-2.2, 1.75, "(a) cos(∇L_data, ∇L_no-slip) ≈ −0.9 意味着什么", fontsize=11, color=INK, ha="left", va="top")
ax.text(-2.2, -1.15, "两个梯度几乎指向相反方向（夹角 ≈ 154°）：\n数据项要把近壁速度拉高，no-slip 项要把它压到零。\n每一步先互相抵消，剩下的短合力再被裁剪 ——\n速度场既拟合不好、壁面也压不到零（no_slip_raw 全程 0.22–0.25）。",
        fontsize=9.2, color=INK2, ha="left", va="top", linespacing=1.45)

# (b) 真实日志：cos 随 epoch
ax = fig.add_subplot(gs[0, 1]); style(ax)
for run, lab, col, ls in GC_ARMS:
    g = GC[run]
    ax.plot(g["ep"], g["cos_ns"], color=col, ls="-", lw=0.3, alpha=0.035)
    ax.plot(g["ep"], rolling_median(g["cos_ns"], 151), color=col, ls=ls, lw=1.9, label=lab)
ax.axhline(0, color=INK2, lw=0.9, ls=":"); ax.axhline(-1, color=INK3, lw=0.7, ls=":")
ax.text(9900, 0.03, "cos = 0：互不干扰", fontsize=8.5, color=INK2, ha="right", va="bottom")
ax.text(9900, -1.04, "cos = −1：完全对拉", fontsize=8.5, color=INK3, ha="right", va="top")
ax.set_ylim(-1.2, 0.62); ax.set_xlim(0, 10100)
ax.set_xlabel("epoch", color=INK2, fontsize=10); ax.set_ylabel("cos(∇L_data, ∇L_no-slip)", color=INK2, fontsize=10)
ax.set_title("(b) 训练日志实测（每 50 step 一次，151 点滑动中位；淡色为原始值）", fontsize=11, color=INK, loc="left", pad=8)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))

# (c) 一维剖面演示：为什么折中只能是"整体压幅"
ax = fig.add_subplot(gs[1, 0]); style(ax)
R = 1.0
r = np.linspace(0, R, 400)
truth = 1.0 - (r / R) ** 8                       # 钝剖面 + 薄边界层（CFD 真值示意）
cells = np.linspace(0.0, R - 0.035, 60)          # 内部 cell-center 标签（最近一层离壁 0.035R，速度仍非零）
lab_u = 1.0 - (cells / R) ** 8
# 网络假设族：平滑低阶（示意 tanh 全局解码器分辨不了边界层）—— u = a + b r² + c r⁴
Phi = lambda x: np.stack([np.ones_like(x), x ** 2, x ** 4], axis=1)
coef_data = np.linalg.lstsq(Phi(cells), lab_u, rcond=None)[0]                    # 只拟合数据
# 数据 + 强 no-slip：在 r=R 处加大权重约束 u(R)=0
W = np.concatenate([np.ones_like(cells), [np.sqrt(60.0 * 20)]])
A = np.vstack([Phi(cells), Phi(np.array([R]))]) * W[:, None]
bvec = np.concatenate([lab_u, [0.0]]) * W
coef_both = np.linalg.lstsq(A, bvec, rcond=None)[0]
u_data = Phi(r) @ coef_data; u_both = Phi(r) @ coef_both
ax.plot(r, truth, color=INK, lw=2.2, label="CFD 真值：钝剖面 + 薄边界层，壁面为 0")
ax.scatter(cells, lab_u, s=16, color=INK, zorder=3, label="cell-center 标签（最靠壁一层仍非零）")
ax.plot(r, u_data, color=MODE_COLOR["DATA"], lw=2.0, label=f"只拟合数据：内部对得上，壁面残留 u(R) = {u_data[-1]:.2f}")
ax.plot(r, u_both, color=MODE_COLOR["DATA+BC"], lw=2.0, label=f"数据 + no-slip 折中：壁面归零，但峰值只剩 {u_both[0]:.2f}")
ax.axvline(R, color=INK3, lw=1.0, ls=":"); ax.text(R - 0.01, 1.28, "壁面", color=INK3, fontsize=9, ha="right", va="top")
ax.axvspan(R - 0.12, R, color=MODE_COLOR["DATA+BC"], alpha=0.06)
ax.text(R - 0.06, -0.14, "近壁\n1.5 mm", color=MODE_COLOR["DATA+BC"], fontsize=8.5, ha="center", va="top")
ax.set_xlim(0, 1.03); ax.set_ylim(-0.32, 1.32)
ax.set_xlabel("到血管中心的距离 r / R（示意）", color=INK2, fontsize=10); ax.set_ylabel("u / U_c（示意）", color=INK2, fontsize=10)
ax.set_title("(c) 平滑低阶网络分辨不了边界层：同时满足两项的唯一折中是整体压幅", fontsize=11, color=INK, loc="left", pad=8)
ax.legend(frameon=False, fontsize=8.3, loc="lower left")

# (d) 后果：末 500 epoch 的 cos 分布 + 幅值压缩
ax = fig.add_subplot(gs[1, 1]); style(ax); ax.grid(False, axis="x")
labs = [s[0] for s in stats_rows]
data = [GC[run]["cos_ns"][GC[run]["ep"] >= GC[run]["ep"].max() - 500] for run, *_ in GC_ARMS]
vp = ax.violinplot(data, positions=np.arange(len(labs)), widths=0.8, showmedians=True, showextrema=False)
for body, (run, lab, col, ls) in zip(vp["bodies"], GC_ARMS):
    body.set_facecolor(col); body.set_alpha(0.55); body.set_edgecolor(SURF)
vp["cmedians"].set_color(INK); vp["cmedians"].set_linewidth(1.6)
for i, s in enumerate(stats_rows):
    ax.text(i, -1.06, f"中位 {s[2]:+.2f}\n{100*s[3]:.0f}% step < −0.8\n‖∇L_data‖ {s[4]:.2f}\n裁剪 {100*s[5]:.0f}%", fontsize=8.2, color=INK2, ha="center", va="top", linespacing=1.35)
ax.axhline(0, color=INK2, lw=0.9, ls=":")
ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels([l.replace(" · ", "\n") for l in labs], fontsize=8.8, color=INK)
ax.set_ylim(-1.55, 0.7); ax.set_ylabel("cos(∇L_data, ∇L_no-slip)，末 500 epoch", color=INK2, fontsize=10)
ax.set_title("(d) 末 500 epoch 的分布：准稳态三臂 −0.88～−0.97，瞬态弱得多（标签本身舒张期近零）", fontsize=10.5, color=INK, loc="left", pad=8)
ax.text(0.015, 0.97, "后果（准稳态 PN，DATA→BC→PDE-F→EMA）：\nwall RMS 0.36 → 0.23 → 0.16 → 0.13 m/s\nspeed 方差比 0.39 → 0.16 → 0.07 → 0.04\nspeed R²_cb −0.10 → −0.28 → −0.42 → −0.51",
        transform=ax.transAxes, fontsize=8.8, color=INK, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f4", ec="#f0c8c4", lw=0.8), linespacing=1.4)
fig.suptitle("图11 · no-slip 软约束与数据项梯度反向（cos ≈ −0.9）：它是什么、日志里长什么样、为什么结果是幅值压缩",
             fontsize=13, color=INK, x=0.01, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/fig11_gradient_conflict.png", dpi=170, facecolor=SURF)
plt.close(fig)
print("fig11 done")


# =====================================================================
# 图 12：E_rel_l2 与 speed R²_cb 的关系（为什么两者必须并列报告）
# =====================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.6)); fig.patch.set_facecolor(SURF); style(ax)
for a in ARMS16:
    t, b, m = a
    mk = "o" if t == "SP" else "^"
    # last
    ax.scatter(R2_SPEED[a], E_LAST[a], s=120 if b == "PN" else 80, marker=mk, color=MODE_COLOR[m],
               edgecolors=SURF, linewidths=1.4, zorder=4)
    # epoch 1000（空心）
    e1 = MS[a]["epoch_01000"]
    ax.scatter(e1["speed"], e1["E"], s=120 if b == "PN" else 80, marker=mk, facecolors="none",
               edgecolors=MODE_COLOR[m], linewidths=1.3, zorder=3)
    ax.plot([e1["speed"], R2_SPEED[a]], [e1["E"], E_LAST[a]], color=MODE_COLOR[m], lw=0.8, alpha=0.5, zorder=2)
ax.axhline(1.0, color=INK2, lw=1.0, ls="--"); ax.axvline(0.0, color=INK2, lw=1.0, ls="--")
ax.text(0.34, 1.005, "E = 1：全零预测器", fontsize=9, color=INK2, ha="right", va="bottom")
ax.text(0.005, 0.775, "R² = 0：预测病例均值", fontsize=9, color=INK2, ha="left", va="bottom", rotation=90)
ax.annotate("准稳态物理臂：E 最低（0.91–0.97）\n但 speed R² 最负（−0.42～−0.51）\n= 预测缩向零场，E 不罚、R² 罚",
            xy=(-0.50, 0.925), xytext=(-0.60, 1.10), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9),
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff9e8", ec="#f2dfa6", lw=0.8))
ax.annotate("DATA 臂 epoch 1000（空心）：\n全矩阵唯一同时 E 较低、R² 为正的点",
            xy=(0.146, 0.803), xytext=(0.19, 0.835), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9),
            bbox=dict(boxstyle="round,pad=0.35", fc="#eef4fc", ec="#bcd3f2", lw=0.8))
ax.text(0.01, 0.03,
        "E 的分母是 ‖u_true‖（不减均值），R² 的分母是 Σ(u − ū)²（减均值）。\n"
        "把预测整体缩小，E 最多退到 1.0；R² 却随缩小幅度一路变负。\n"
        "所以 E 用于 V4 内部排序，跨代（V2/V3）对比一律用同口径的 speed R²_cb / WSS fit R²。",
        transform=ax.transAxes, fontsize=9.2, color=INK, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc=SURF, ec=GRID, lw=0.8), linespacing=1.45)
h1 = [Line2D([], [], marker="s", ls="", color=MODE_COLOR[m], label=m) for m in MODES]
h2 = [Line2D([], [], marker="o", ls="", color=INK2, label="准稳态"), Line2D([], [], marker="^", ls="", color=INK2, label="瞬态"),
      Line2D([], [], marker="o", ls="", color=INK2, ms=10, label="大 = PN，小 = PNPP"),
      Line2D([], [], marker="o", ls="", markerfacecolor="none", color=INK2, label="空心 = epoch 1000，实心 = last")]
ax.legend(handles=h1 + h2, frameon=False, fontsize=8.8, loc="upper right", ncol=2)
ax.set_xlim(-0.66, 0.40); ax.set_ylim(0.76, 1.22)
ax.set_xlabel("speed $R^2_{cb}$（病例等权，越大越好）", color=INK2, fontsize=10)
ax.set_ylabel("$E_{\\mathrm{rel},L2}$（病例等权，越小越好）", color=INK2, fontsize=10)
ax.set_title("图12 · 主指标 E 与 speed R² 在 V4 里方向相反：E 对幅值压缩不敏感，R² 惩罚它", fontsize=12.5, color=INK, loc="left", pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig12_E_vs_speedR2.png", dpi=170, facecolor=SURF)
plt.close(fig)
print("fig12 done")

# 汇总数字（供文档引用）
with open(f"{OUT}/summary_numbers.json", "w") as fh:
    json.dump({
        "pressure_r2_TR_PN_PDE_F": {"mean_all": mean_all, "median": med_all, "mean_without_2_outliers": mean_wo, "n_positive": n_pos,
                                    "outliers": [dict(case=r["case"], gauge_pa=r["gauge"], r2=r["r2"], truth_std_pa=float(np.sqrt(r["tvar"])), rmse_pa=r["rmse"]) for r, o in zip(rows, outlier) if o],
                                    "normal_truth_std_median_pa": float(np.median(tstd[~outlier])), "normal_rmse_median_pa": float(np.median(rmse[~outlier]))},
        "gradcos": [dict(arm=s[0], cos_median_first200=s[1], cos_median_last500=s[2], frac_below_m0p8_last500=s[3],
                         grad_norm_data_median_last500=s[4], clip_fraction_last500=s[5], no_slip_raw_median_last500=s[6]) for s in stats_rows],
        "profile_demo": {"u_wall_data_only": float(u_data[-1]), "u_peak_compromise": float(u_both[0])},
    }, fh, ensure_ascii=False, indent=1)
print("ALL DONE")
