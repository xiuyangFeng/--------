"""Why deceleration and the trough are hard: ground-truth structure vs frame-wise r² (why_decel_trough.json + frame_metric_suite_best.csv)."""
import json, csv
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 9})
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
J = json.load(open('why_decel_trough.json')); fm = J['frame_mean']; q = np.array(J['q_norm']); phase = J['phase']
rows = list(csv.DictReader(open('why_decel_trough_frame.csv')))
t0 = np.array([float(r['T0_r2fit']) for r in rows]); t0s = np.array([float(r['T0_spearman']) for r in rows]); tn = np.array([float(r['Tnull_r2fit']) for r in rows])
x = np.arange(81)
def style(ax):
    ax.set_facecolor(SURF); ax.grid(axis='y', color=GRID, lw=.6); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
def shade(ax):
    for i, p in enumerate(phase):
        if p == 'peak': ax.axvspan(i - .5, i + .5, color='#f0efec', lw=0, zorder=0)
        elif p == 'trough': ax.axvspan(i - .5, i + .5, color='#f7f2ea', lw=0, zorder=0)
    ax.axvline(21, color=INK2, lw=.6, ls=':')
fig = plt.figure(figsize=(14.2, 8.8), facecolor=SURF)
gs = fig.add_gridspec(3, 2, height_ratios=[0.46, 1, 1.05], hspace=0.55, wspace=0.42, left=0.11, right=0.985, top=0.88, bottom=0.13)
ax = fig.add_subplot(gs[0, :]); shade(ax); style(ax)
ax.plot(x, q, color=INK, lw=1.4); ax.set_ylim(0, 1.22); ax.set_xlim(-.5, 80.5); ax.set_ylabel('Q/Q_max', color=INK2)
ax.set_title('Figure B. Why deceleration and the trough are hard: ground-truth structure vs frame-wise skill\n(mean CFD truth of 136 training cases; model = cv3 mean of 3 folds, ckpt best)', loc='left', fontsize=11.5, color=INK, pad=8)
ax.text(21, 0.38, 'Peak frame', ha='center', va='center', fontsize=8, color=INK2); ax.text(49, 0.42, 'Trough quartile', ha='center', fontsize=8, color=INK2)
ax = fig.add_subplot(gs[1, 0]); shade(ax); style(ax)
ax.plot(x, fm['c_peak'], color=C1, lw=2, label='Truth: corr. with peak field, c_peak')
ax.plot(x, fm['c_murray'], color=C2, lw=2, label='Truth: corr. with Murray prior, c_murray')
ax.plot(x, t0, color=C3, lw=2, label='Model T0: Pearson r²')
ax.plot(x, tn, color=C4, lw=1.6, ls='--', label='T-null: Pearson r²')
ax.set_ylim(0, 1); ax.set_xlim(-.5, 80.5); ax.set_title('1  Fraction of this frame still explained by geometry / the peak field', loc='left', fontsize=9.2, color=INK)
ax.legend(frameon=False, fontsize=7.4, loc='lower left')
ax.set_xlabel('Frame index', color=INK2)
ax = fig.add_subplot(gs[1, 1]); shade(ax); style(ax)
ax.plot(x, fm['rev_frac'], color=C1, lw=2, label='rev_frac: wall WSS reversed vs the peak field')
ax.plot(x, fm['within4mm'], color=C2, lw=2, label='within 4 mm: share of ln|τ| variance (small scale)')
ax.plot(x, fm['frac_lt005'], color=C3, lw=1.6, ls='--', label='floor: fraction with |τ| < 0.05 Pa')
ax.set_ylim(0, 0.7); ax.set_xlim(-.5, 80.5); ax.set_title('2  What the field becomes: reversal, small-scale patches, floor', loc='left', fontsize=9.2, color=INK)
ax.legend(frameon=False, fontsize=7.4, loc='upper left'); ax.set_xlabel('Frame index', color=INK2)
ax = fig.add_subplot(gs[2, 0]); style(ax)
col = {'accel': C1, 'peak': C4, 'decel': C2, 'trough': C3}
lab = {'accel': 'Acceleration', 'peak': 'Peak window', 'decel': 'Deceleration', 'trough': 'Trough'}
cp = np.array(fm['c_peak'])
for p in ('accel', 'peak', 'decel', 'trough'):
    m = np.array([ph == p for ph in phase]); ax.scatter(cp[m], t0[m], s=26, color=col[p], label=lab[p], edgecolor=SURF, linewidth=.8, zorder=3)
ok = np.isfinite(cp); rho = np.corrcoef(cp[ok].argsort().argsort(), t0[ok].argsort().argsort())[0, 1]
ax.set_xlabel('Ground-truth c_peak', color=INK2); ax.set_ylabel('T0 frame-wise Pearson r²', color=INK2)
ax.set_title(f'3  Over 81 frames, model skill is nearly a function of c_peak (Spearman ρ = {rho:.2f})', loc='left', fontsize=9.2, color=INK)
ax.legend(frameon=False, fontsize=7.4, loc='upper left'); ax.set_xlim(0.1, 1); ax.set_ylim(0.1, 0.9)
ax = fig.add_subplot(gs[2, 1]); style(ax)
i, j = J['pairs'][0]
names = ['c_peak', 'c_murray', 'rev_frac', 'within4mm']
labels = ['c_peak', 'c_murray', 'Reversed fraction', 'Small-scale share']
va = [fm[k][i] for k in names] + [t0[i], t0s[i]]; vd = [fm[k][j] for k in names] + [t0[j], t0s[j]]
labels += ['T0 r²', 'T0 Spearman']
yy = np.arange(len(labels))[::-1]
ax.barh(yy + 0.18, va, height=0.34, color=C1, label=f'Acceleration, frame {i} (Q/Qmax={q[i]:.2f})')
ax.barh(yy - 0.18, vd, height=0.34, color=C2, label=f'Deceleration, frame {j} (Q/Qmax={q[j]:.2f})')
for y_, a, d in zip(yy, va, vd):
    ax.text(a + .01, y_ + .18, f'{a:.2f}', va='center', fontsize=7.5, color=INK2); ax.text(d + .01, y_ - .18, f'{d:.2f}', va='center', fontsize=7.5, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels(labels); ax.set_xlim(0, 1.12); ax.grid(axis='x', color=GRID, lw=.6); ax.grid(axis='y', visible=False)
ax.set_title('4  Same inlet flow, opposite phase: acceleration vs deceleration (hysteresis)', loc='left', fontsize=9.2, color=INK)
ax.legend(frameon=False, fontsize=7.3, loc='upper center', bbox_to_anchor=(0.42, -0.18), ncol=1, borderaxespad=0)
fig.text(0.14, 0.012, 'Grey = peak window; light orange = trough quartile. c_peak, c_murray, rev_frac and within4mm use CFD truth only (ln|τ| space, floor 0.05 Pa). Murray prior = log τ0 from wss_min_flowref_v1.', fontsize=7.4, color=INK2)
for ext in ('png', 'svg'):
    fig.savefig(f'fig_why_decel_trough.{ext}', dpi=200 if ext == 'png' else None, facecolor=SURF)
print('saved')
