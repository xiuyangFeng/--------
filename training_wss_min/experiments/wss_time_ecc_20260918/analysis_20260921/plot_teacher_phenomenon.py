"""Figure A for the advisor briefing. Top: inlet waveform. Bottom: R²_cb / Pearson r² / Spearman for X5D_v51 frozen, T-null, T0 (TB8 dashed)."""
import csv
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 9.5})
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
ARMS = [('X5D_v51_frozen', 'X5D_v51 peak model, frozen (no time input)', '#2a78d6', '-', 2.2),
        ('Tnull_Bscale', 'T-null: peak prediction × cohort time shape (untrained)', '#eb6834', '-', 2.2),
        ('T0', 'T0: phase input, trained on all frames', '#1baf7a', '-', 2.2),
        ('TB8', 'TB8: temporal-basis head, K=8', '#eda100', '--', 1.6)]
PANELS = [('r2cb', 'R²_cb: gated metric (amplitude + pattern)', (-0.6, 1.0)),
          ('r2_fit', 'Pearson r²: pattern only (best scale and bias)', (0.0, 1.0)),
          ('spearman', 'Spearman ρ: rank only (scale-free)', (0.0, 1.0))]
rows = list(csv.DictReader(open('frame_metric_suite_best.csv')))
steps = sorted({int(r['step']) for r in rows}); idx = {s: i for i, s in enumerate(steps)}
q = np.array([float(next(r['q_norm'] for r in rows if int(r['step']) == s)) for s in steps]); phase = {int(r['step']): r['phase'] for r in rows}
C = defaultdict(lambda: np.full((3, 81), np.nan))
for r in rows:
    for k, _, _ in PANELS: C[(r['arm'], k)][int(r['fold']), idx[int(r['step'])]] = float(r[k])
x = np.arange(81)
def style(ax):
    ax.set_facecolor(SURF); ax.grid(axis='y', color=GRID, lw=.6); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0); ax.set_xlim(-.5, 80.5)
def shade(ax):
    for i, s in enumerate(steps):
        if phase[s] == 'peak': ax.axvspan(i - .5, i + .5, color='#ecebe7', lw=0, zorder=0)
        elif phase[s] == 'trough': ax.axvspan(i - .5, i + .5, color='#f7f0e6', lw=0, zorder=0)
    ax.axvline(21, color=INK2, lw=.6, ls=':')
fig = plt.figure(figsize=(14.2, 8.6), facecolor=SURF)
gs = fig.add_gridspec(2, 3, height_ratios=[0.52, 1], hspace=0.42, wspace=0.22, left=0.055, right=0.985, top=0.84, bottom=0.16)
ax = fig.add_subplot(gs[0, :]); shade(ax); style(ax)
ax.plot(x, q, color=INK, lw=1.6); ax.set_ylim(0, 1.08); ax.set_ylabel('Inlet flow Q/Q_max', color=INK2)
for i, lab, y in [(21, 'Peak frame\n(step 1162)', 0.72), (7, 'Early diastole\n5 frames', 0.42), (49, 'Trough quartile, 15 frames\n(Q ≈ 0; lowest at step 1216)', 0.62)]:
    ax.text(i, y, lab, ha='center', fontsize=8.2, color=INK2)
ax.text(13, 0.5, 'Acceleration', ha='center', fontsize=8.2, color=INK2)
ax.text(34, 0.62, 'Deceleration', ha='center', fontsize=8.2, color=INK2)
ax.text(70, 0.28, 'Diastolic plateau', ha='center', fontsize=8.2, color=INK2)
ax.set_title('Figure A. Frame-wise accuracy falls together in deceleration and the trough.\nAll three metrics drop, so the loss is spatial pattern, not only amplitude.', loc='left', fontsize=12, color=INK, pad=10)
for n, (key, title, ylim) in enumerate(PANELS):
    ax = fig.add_subplot(gs[1, n]); shade(ax); style(ax)
    for arm, label, col, ls, lw in ARMS:
        m = np.nanmean(C[(arm, key)], 0); sd = np.nanstd(C[(arm, key)], 0); lo, hi = ylim
        clipped = (m < lo) | (m > hi); mm = np.where(clipped, np.nan, m)
        ax.plot(x, mm, color=col, lw=lw, ls=ls, label=label, zorder=3)
        ax.fill_between(x, np.clip(mm - sd, lo, hi), np.clip(mm + sd, lo, hi), color=col, alpha=.10, lw=0)
        if clipped.any():
            ax.text(0.03, 0.04, f'Blue: {int(clipped.sum())} frames < −0.6 omitted (min {np.nanmin(m):.0f}).\nPeak-field amplitude is 6–10× larger at low flow.', transform=ax.transAxes, fontsize=7.6, color=col, va='bottom')
    ax.set_ylim(*ylim); ax.set_title(title, loc='left', fontsize=9.2, color=INK); ax.set_xlabel('Frame index (0–80; one cardiac cycle = 0.8 s)', color=INK2)
    if key == 'r2cb': ax.axhline(0, color=INK2, lw=.6)
h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8.4, bbox_to_anchor=(0.52, 0.012))
fig.text(0.055, 0.955, 'cv3 mean of 3 folds (shade ±1 sd), Pa space, ckpt best. Grey band = peak window (10 frames); light orange = trough quartile (20 frames).', fontsize=8.5, color=INK2)
for ext in ('png', 'svg'): fig.savefig(f'fig_teacher_A_phenomenon.{ext}', dpi=200 if ext == 'png' else None, facecolor=SURF)
print('saved A')
