"""逐帧指标套件小倍图（读 frame_metric_suite_<ckpt>.csv）。五臂固定色序，越界臂裁到轴内并标注真实范围。"""
import csv, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'
ARMS = [('X5D_v51_frozen', 'X5D_v51 峰值冻结', '#2a78d6', '-'), ('Tnull_Bscale', 'T-null', '#eb6834', '-'),
        ('T0', 'T0', '#1baf7a', '-'), ('TB8', 'TB8', '#eda100', '-'), ('TB16', 'TB16', '#e87ba4', '--')]
PANELS = [('r2cb', 'R²_cb（现行门控口径）', (-0.55, 1.0)), ('r2_fit', 'Pearson r²（去幅值/偏置）', (0.0, 1.0)),
          ('spearman', 'Spearman ρ（只看排序）', (0.0, 1.0)), ('ccc', 'Lin CCC（相关 × 幅值一致）', (0.0, 1.0)),
          ('approx_disp', 'approximation disparity ↓', (0.0, 1.25)), ('nmae_case', 'NMAE 逐例 ÷ 帧 range ↓', (0.0, 0.12))]
for fam in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans Mono CJK SC'):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = fam; break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9})
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'

rows = list(csv.DictReader(open(f'frame_metric_suite_{CKPT}.csv')))
steps = sorted({int(r['step']) for r in rows}); idx = {s: i for i, s in enumerate(steps)}
q = np.array([float(next(r['q_norm'] for r in rows if int(r['step']) == s)) for s in steps])
phase = {int(r['step']): r['phase'] for r in rows}
curve = defaultdict(lambda: np.full((3, len(steps)), np.nan))
for r in rows:
    for k, _, _ in PANELS:
        curve[(r['arm'], k)][int(r['fold']), idx[int(r['step'])]] = float(r[k])

fig = plt.figure(figsize=(13.5, 8.6), facecolor=SURF)
gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 1, 1], hspace=0.42, wspace=0.22, left=0.05, right=0.985, top=0.9, bottom=0.07)
x = np.arange(len(steps))
def shade(ax):
    for i, s in enumerate(steps):
        if phase[s] == 'peak': ax.axvspan(i - .5, i + .5, color='#f0efec', lw=0, zorder=0)
        elif phase[s] == 'trough': ax.axvspan(i - .5, i + .5, color='#f7f2ea', lw=0, zorder=0)
    ax.axvline(21, color=INK2, lw=.6, ls=':', zorder=1)
# 顶部：入口波形
axw = fig.add_subplot(gs[0, :]); shade(axw)
axw.plot(x, q, color=INK, lw=1.4); axw.set_ylabel('Q/Q_max', color=INK2); axw.set_ylim(0, 1.05)
axw.text(21, 1.0, '峰值帧 1162', ha='center', va='bottom', fontsize=8, color=INK2)
axw.text(np.mean([idx[s] for s in steps if phase[s] == 'trough' and s > 1200]), 0.62, '谷底四分位（20 帧，浅橙底）', ha='center', fontsize=8, color=INK2)
axw.text(np.mean([idx[s] for s in steps if phase[s] == 'peak']), 0.15, '峰值窗\n10 帧', ha='center', fontsize=8, color=INK2)
axw.set_title(f'WSS 全周期逐帧指标 · ckpt {CKPT} · cv3 三折均值（细线 ±1 sd）· Pa 空间', loc='left', fontsize=11, color=INK, pad=8)
for ax in [axw]:
    ax.set_facecolor(SURF); ax.grid(axis='y', color=GRID, lw=.6); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0); ax.set_xlim(-.5, 80.5)
axes = []
for n, (key, title, ylim) in enumerate(PANELS):
    ax = fig.add_subplot(gs[1 + n // 3, n % 3]); axes.append(ax); shade(ax)
    for arm, label, col, ls in ARMS:
        c = curve[(arm, key)]; m = np.nanmean(c, 0); sd = np.nanstd(c, 0)
        lo, hi = ylim; clipped = (m < lo) | (m > hi)
        mm = np.where(clipped, np.nan, m)           # 越界帧留空，不画贴边假线
        ax.plot(x, mm, color=col, lw=2 if arm != 'TB16' else 1.4, ls=ls, zorder=3, label=label, solid_capstyle='round')
        ax.fill_between(x, np.clip(mm - sd, lo, hi), np.clip(mm + sd, lo, hi), color=col, alpha=.12, lw=0, zorder=2)
        if clipped.any():
            worst = np.nanmin(m) if (m < lo).any() else np.nanmax(m)
            below = (m < lo).any()
            ax.text(0.02, 0.06 if below else 0.94, f'{label}：{int(clipped.sum())} 帧越界未画（至 {worst:.0f}）',
                    transform=ax.transAxes, fontsize=7.5, color=col, va='bottom' if below else 'top', ha='left')
    ax.set_ylim(*ylim); ax.set_xlim(-.5, 80.5); ax.set_title(title, loc='left', fontsize=9.5, color=INK)
    ax.set_facecolor(SURF); ax.grid(axis='y', color=GRID, lw=.6); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    if key == 'r2cb': ax.axhline(0, color=INK2, lw=.6)
    if n // 3 == 1: ax.set_xlabel('帧序号（0–80；Fluent 步 1120–1280，步距 2）', color=INK2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.985, 0.985))
fig.text(0.05, 0.012, 'X5D_v51 峰值冻结 = 无时间输入的部署底座直接对非峰值帧真值打分；T-null = 同一峰值预测 × 训练折 μ(t)/σ(t)；T0/TB8/TB16 = 阶段 2 时间臂。'
         '灰底 = 峰值窗，浅橙底 = 谷底四分位。越界的冻结臂帧留空并标注真实范围。', fontsize=7.5, color=INK2)
for ext in ('png', 'svg'):
    fig.savefig(f'fig_frame_metric_suite_{CKPT}.{ext}', dpi=200 if ext == 'png' else None, facecolor=SURF)
print('saved fig_frame_metric_suite_' + CKPT)
