"""Figure C schematic. Protocol inlet harmonics drive a straight-tube Womersley solution: velocity profiles, wall shear vs Q, and the τ–Q loop.
Rigid-tube analytic solution only (not CFD). R = 8 mm (aortic scale) and 4 mm (iliac scale), ν = 3.5e-6 m²/s, T = 0.8 s."""
import json
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import jv
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 9.5})
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
W = json.load(open('why_decel_trough.json')); q = np.array(W['q_norm']); phase = W['phase']
T, nu, NH = 0.8, 3.5e-6, 12
qq = q[:80]  # 帧 80 与帧 0 同相位
Qh = np.fft.rfft(qq) / 80.0  # 复谐波系数（n=0..40）
t = np.arange(81) / 80.0 * T; omega = 2 * np.pi / T
rr = np.linspace(0, 1, 101)

def profile_and_wss(R):
    """返回 u(r,t)（单位流量归一，∝ m/s per (m³/s)）与壁面 τ(t)/τ_peak，Q(t) 同样归一到峰值。"""
    u = np.zeros((81, len(rr))); tau = np.zeros(81)
    # n = 0：Poiseuille
    u += 2 * Qh[0].real * (1 - rr ** 2)[None, :] / (np.pi * R ** 2); tau += 4 * Qh[0].real / (np.pi * R ** 3)
    for n in range(1, NH + 1):
        alpha = R * np.sqrt(n * omega / nu); Lam = 1j ** 1.5 * alpha
        shape = (1 - jv(0, Lam * rr) / jv(0, Lam)) / (1 - 2 * jv(1, Lam) / (Lam * jv(0, Lam))) / (np.pi * R ** 2)
        dudr = (Lam / R) * jv(1, Lam) / jv(0, Lam) / (1 - 2 * jv(1, Lam) / (Lam * jv(0, Lam))) / (np.pi * R ** 2)
        e = np.exp(1j * n * omega * t)
        u += 2 * np.real(Qh[n] * e[:, None] * shape[None, :]); tau += 2 * np.real(Qh[n] * e * (-dudr))
    return u, tau

fig = plt.figure(figsize=(14.6, 8.6), facecolor=SURF)
gs = fig.add_gridspec(2, 4, hspace=0.72, wspace=0.48, left=0.05, right=0.985, top=0.82, bottom=0.08)
def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=.6); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
PH = [(10, 'Accel. frame 10', C1), (21, 'Peak frame 21', C4), (35, 'Decel. frame 35', C2), (48, 'Trough frame 48', C3)]
out = {}
for row, R in enumerate((8e-3, 4e-3)):
    u, tau = profile_and_wss(R); upk = np.abs(u[21]).max(); out[R] = tau
    alpha1 = R * np.sqrt(omega / nu)
    # 剖面
    ax = fig.add_subplot(gs[row, 0:2]); style(ax); ax.axvline(0, color=INK2, lw=.6)
    for k, lab, col in PH:
        ax.plot(u[k] / upk, rr, color=col, lw=2.2, label=f'{lab} (Q/Qmax={q[k]:.2f})'); ax.plot(u[k] / upk, -rr, color=col, lw=2.2)
    ax.set_xlim(-0.42, 1.08); ax.set_ylim(-1.05, 1.05); ax.set_ylabel('r / R', color=INK2); ax.set_xlabel('Axial velocity / peak centerline velocity', color=INK2)
    ax.set_title(f'{"1" if row == 0 else "4"}  Straight-tube Womersley profile, R = {R*1e3:.0f} mm, α = {alpha1:.0f}', loc='left', fontsize=9.3, color=INK)
    ax.legend(frameon=False, fontsize=7.2, loc='upper center', bbox_to_anchor=(0.58, -0.32), ncol=2, borderaxespad=0)
    ax.annotate('Decel.: near wall reversed,\ncore still forward', xy=(-0.08, 0.92), xytext=(-0.40, 0.48), fontsize=7.6, color=C2, arrowprops=dict(arrowstyle='->', color=C2, lw=.8))
    ax = fig.add_subplot(gs[row, 2]); style(ax)
    ax.plot(np.arange(81), q, color=INK, lw=1.4, label='Q / Q_max'); ax.plot(np.arange(81), tau / tau.max(), color=C2, lw=2.2, label='τ / τ_peak')
    ax.axhline(0, color=INK2, lw=.6)
    for k, lab, col in PH: ax.axvline(k, color=col, lw=.8, ls=':')
    neg = np.flatnonzero(tau < 0)
    ax.set_title(f'{"2" if row == 0 else "5"}  Wall shear leads flow:\nτ < 0 for {len(neg)} frames', loc='left', fontsize=9.2, color=INK)
    ax.set_xlabel('Frame index', color=INK2); ax.legend(frameon=False, fontsize=7.4, loc='lower right'); ax.set_ylim(-0.6, 1.15)
    ax = fig.add_subplot(gs[row, 3]); style(ax)
    ax.plot(q, tau / tau.max(), color=INK2, lw=1.2); ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--')
    for k, lab, col in PH: ax.scatter(q[k], tau[k] / tau.max(), color=col, s=40, zorder=3, edgecolor=SURF)
    ax.annotate('', xy=(q[35], tau[35] / tau.max()), xytext=(q[30], tau[30] / tau.max()), arrowprops=dict(arrowstyle='->', color=INK2))
    ax.annotate('', xy=(q[14], tau[14] / tau.max()), xytext=(q[11], tau[11] / tau.max()), arrowprops=dict(arrowstyle='->', color=INK2))
    ax.set_xlabel('Q / Q_max', color=INK2); ax.set_ylabel('τ / τ_peak', color=INK2)
    ax.set_title(f'{"3" if row == 0 else "6"}  τ–Q hysteresis\n(upper: accel.; lower: decel.)', loc='left', fontsize=9.2, color=INK); ax.set_ylim(-0.6, 1.15)
fig.suptitle('Figure C. Pulsatile inertia reverses wall shear before flow.\nIn deceleration and the trough, WSS follows pressure-gradient history, not instantaneous flow × geometry.', x=0.05, ha='left', fontsize=12, color=INK, y=0.98)
fig.text(0.05, 0.905, 'Rigid straight-tube Womersley solution driven by 12 Fourier harmonics of the shared protocol inlet waveform (172 cases); ν = 3.5e-6 m²/s, T = 0.8 s. Schematic only, not CFD. Real bifurcations and the aneurysm add separation and secondary flow.', fontsize=8.0, color=INK2)
for ext in ('png', 'svg'): fig.savefig(f'fig_teacher_C_womersley.{ext}', dpi=200 if ext == 'png' else None, facecolor=SURF)
json.dump({'R_m': list(out.keys()), 'tau_over_peak': {f'{R*1e3:.0f}mm': (v / v.max()).tolist() for R, v in out.items()}, 'q_norm': q.tolist(),
           'first_negative_frame': {f'{R*1e3:.0f}mm': int(np.flatnonzero(v < 0)[0]) if (v < 0).any() else None for R, v in out.items()},
           'q_at_first_negative': {f'{R*1e3:.0f}mm': float(q[np.flatnonzero(v < 0)[0]]) if (v < 0).any() else None for R, v in out.items()}},
          open('womersley_schematic.json', 'w'), indent=1)
print('saved C', {f'{R*1e3:.0f}mm': (int(np.flatnonzero(v < 0)[0]) if (v < 0).any() else None, round(float(q[np.flatnonzero(v < 0)[0]]), 3) if (v < 0).any() else None) for R, v in out.items()})
