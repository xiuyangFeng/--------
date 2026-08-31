#!/usr/bin/env python3
"""V4 seed1234 index 0-15: training curves + diagnostic comparison figures."""
import json, os, glob, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

# ---------- fonts ----------
for f in glob.glob('/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc'):
    try: font_manager.fontManager.addfont(f)
    except Exception: pass
avail = {f.name for f in font_manager.fontManager.ttflist}
cjk = next((n for n in ['Noto Sans CJK SC','Noto Sans CJK JP','Noto Sans CJK TC'] if n in avail), None)
plt.rcParams['font.family'] = [cjk, 'DejaVu Sans'] if cjk else ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- palette (validated, light mode) ----------
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e0'
MODE_COLOR = {'DATA':'#2a78d6', 'DATA+BC':'#eb6834', 'BC+PDE-F':'#1baf7a', 'BC+PDE-EMA':'#eda100'}
MODES = ['DATA','DATA+BC','BC+PDE-F','BC+PDE-EMA']
MODE_KEY = {'DATA':'DATA','DATA+BC':'BC','BC+PDE-F':'BC-PDE-F','BC+PDE-EMA':'BC-PDE-EMA'}

BASE = '/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/volume_uvwp_bc_rcr_v4'
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

GROUPS = [('SP','PN'),('SP','PNPP'),('TR','PN'),('TR','PNPP')]
TDIR = {'SP':'steady_peak','TR':'transient_autograd'}

def run_name(t,b,mode):
    return f"V4-{t}-{b}-{MODE_KEY[mode]}-s1234"

# ---------- load training curves ----------
FIELDS = ['mean_data_total','mean_data_u_raw','mean_data_p_raw','mean_no_slip_raw',
          'mean_inlet_bc_raw','mean_rcr_bc_raw','mean_continuity_raw',
          'mean_momentum_group_raw','mean_lambda_phy']
def load_curve(t,b,mode):
    p = f"{BASE}/{TDIR[t]}/{run_name(t,b,mode)}/epoch_progress.jsonl"
    out = {k: [] for k in FIELDS}; ep=[]
    with open(p) as fh:
        for line in fh:
            d = json.loads(line)
            ep.append(d['epoch'])
            for k in FIELDS: out[k].append(d.get(k, float('nan')))
    return np.array(ep), {k: np.array(v, dtype=float) for k,v in out.items()}

print('loading curves...')
CURVES = {}
for t,b in GROUPS:
    for m in MODES:
        CURVES[(t,b,m)] = load_curve(t,b,m)
print('loaded', len(CURVES))

def smooth(y, w=51):
    if len(y) < w: return y
    k = np.ones(w)/w
    ys = np.convolve(y, k, mode='same')
    h = w//2  # fix edges
    for i in range(h):
        ys[i] = y[:i+h+1].mean(); ys[-(i+1)] = y[-(i+h+1):].mean()
    return ys

def style_ax(ax, logy=True):
    ax.set_facecolor(SURF)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(GRID)
    ax.grid(True, axis='y', color=GRID, lw=0.7)
    ax.tick_params(colors=INK2, labelsize=9)
    if logy: ax.set_yscale('log')

def plot_lines(ax, keysel, arms, labels=None, colors=None, styles=None, w=51, endlab=True):
    for i,(arm) in enumerate(arms):
        ep, c = CURVES[arm]
        y = c[keysel]
        col = colors[i] if colors else MODE_COLOR[arm[2]]
        ls = styles[i] if styles else '-'
        ax.plot(ep, y, color=col, alpha=0.10, lw=0.7, ls=ls)
        ax.plot(ep, smooth(y,w), color=col, lw=1.8, ls=ls,
                label=(labels[i] if labels else arm[2]))
        if endlab:
            yl = smooth(y,w)[-1]
            ax.annotate(f'{yl:.3g}', xy=(ep[-1], yl), xytext=(4,0), textcoords='offset points',
                        fontsize=8, color=INK, va='center')

def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=8.5, **kw)
    return lg

TITLE_KW = dict(fontsize=11, color=INK, loc='left', pad=8)
figs = []

# ================= Fig 1: raw data_total =================
fig, axes = plt.subplots(2,2, figsize=(11.5,7.2), sharex=True)
fig.patch.set_facecolor(SURF)
panel = {('SP','PN'):(0,0), ('SP','PNPP'):(0,1), ('TR','PN'):(1,0), ('TR','PNPP'):(1,1)}
ptitle = {('SP','PN'):'准稳态 · PointNet', ('SP','PNPP'):'准稳态 · PointNet++',
          ('TR','PN'):'瞬态 · PointNet', ('TR','PNPP'):'瞬态 · PointNet++'}
for (t,b),(i,j) in panel.items():
    ax = axes[i,j]; style_ax(ax)
    plot_lines(ax, 'mean_data_total', [(t,b,m) for m in MODES])
    ax.set_title(ptitle[(t,b)], **TITLE_KW)
    ax.set_ylim(0.05, 3)
    if i==1: ax.set_xlabel('epoch', color=INK2, fontsize=9)
    if j==0: ax.set_ylabel('raw data_total（标准化单位）', color=INK2, fontsize=9)
axes[0,0].set_xlim(0, 11500)
legend(axes[0,0], loc='lower left', ncol=2)
fig.suptitle('图1 · 训练侧 data loss：每加一层物理（BC→PDE→EMA），训练拟合就系统性变差一档',
             fontsize=12.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=[0,0,1,0.94])
p=f'{OUT}/fig1_train_data_total.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= Fig 2: u / p channels =================
fig, axes = plt.subplots(2,2, figsize=(11.5,7.2), sharex=True)
fig.patch.set_facecolor(SURF)
spec = [(('SP','PN'),'mean_data_u_raw','准稳态 · PN · u 通道',(0,0)),
        (('TR','PN'),'mean_data_u_raw','瞬态 · PN · u 通道',(0,1)),
        (('SP','PN'),'mean_data_p_raw','准稳态 · PN · p 通道',(1,0)),
        (('TR','PN'),'mean_data_p_raw','瞬态 · PN · p 通道',(1,1))]
for (t,b),k,ti,(i,j) in spec:
    ax = axes[i,j]; style_ax(ax)
    plot_lines(ax, k, [(t,b,m) for m in MODES])
    ax.set_title(ti, **TITLE_KW)
    if i==1: ax.set_xlabel('epoch', color=INK2, fontsize=9)
axes[0,0].set_ylabel('raw loss（标准化 MSE，1.0≈完全未拟合）', color=INK2, fontsize=9)
axes[1,0].set_ylabel('raw loss', color=INK2, fontsize=9)
axes[0,0].set_xlim(0, 11500)
for ax in axes.flat: ax.axhline(1.0, color=INK2, lw=0.8, ls=':')
axes[0,1].annotate('MSE=1.0 ≈ 预测均值/零', xy=(200,1.0), xytext=(0,4), textcoords='offset points',
                   fontsize=8, color=INK2)
legend(axes[0,0], loc='lower left', ncol=2)
fig.suptitle('图2 · 分通道训练损失：瞬态 EMA 的 u 通道停在 1.0（完全未拟合）；准稳态 PDE 臂牺牲 p 通道',
             fontsize=12.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=[0,0,1,0.94])
p=f'{OUT}/fig2_train_channels.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= Fig 3: physics dynamics =================
fig, axes = plt.subplots(2,2, figsize=(11.5,7.2))
fig.patch.set_facecolor(SURF)
# (a) lambda_pde for EMA arms  — color by temporal, style by backbone
ax = axes[0,0]; style_ax(ax, logy=False)
enc = [(('TR','PN','BC+PDE-EMA'),'#eb6834','-','TR-PN'),
       (('TR','PNPP','BC+PDE-EMA'),'#eb6834','--','TR-PNPP'),
       (('SP','PN','BC+PDE-EMA'),'#2a78d6','-','SP-PN'),
       (('SP','PNPP','BC+PDE-EMA'),'#2a78d6','--','SP-PNPP')]
for arm,col,ls,lab in enc:
    ep,c = CURVES[arm]
    ax.plot(ep, c['mean_lambda_phy'], color=col, lw=1.8, ls=ls, label=lab)
ax.axhline(10, color=INK2, lw=0.9, ls=':')
ax.annotate('clamp 上限 λ_max=10（四臂长期贴死、曲线重叠）', xy=(4500,10), xytext=(0,-12),
            textcoords='offset points', fontsize=8.5, color=INK2)
ax.set_ylim(0,11); ax.set_title('(a) EMA 臂的 λ_pde 轨迹', **TITLE_KW)
ax.set_ylabel('λ_pde', color=INK2, fontsize=9); legend(ax, loc='lower right', ncol=2)
# (b)(c) PDE residuals: F vs EMA, PN only, SP solid / TR dashed
for ax,k,ti in [(axes[0,1],'mean_continuity_raw','(b) continuity 残差（raw，无量纲）'),
                (axes[1,0],'mean_momentum_group_raw','(c) momentum 残差（raw，无量纲）')]:
    style_ax(ax)
    for t,ls in [('SP','-'),('TR','--')]:
        for m in ['BC+PDE-F','BC+PDE-EMA']:
            ep,c = CURVES[(t,'PN',m)]
            ax.plot(ep, c[k], color=MODE_COLOR[m], alpha=0.10, lw=0.7, ls=ls)
            ax.plot(ep, smooth(c[k]), color=MODE_COLOR[m], lw=1.8, ls=ls,
                    label=f'{t} · {m}')
    ax.set_title(ti, **TITLE_KW)
legend(axes[0,1], loc='lower left', ncol=1)
# (d) RCR residual TR arms
ax = axes[1,1]; style_ax(ax)
for m in ['DATA+BC','BC+PDE-F','BC+PDE-EMA']:
    ep,c = CURVES[('TR','PN',m)]
    ax.plot(ep, c['mean_rcr_bc_raw'], color=MODE_COLOR[m], alpha=0.10, lw=0.7)
    ax.plot(ep, smooth(c['mean_rcr_bc_raw']), color=MODE_COLOR[m], lw=1.8, label=m)
ax.set_title('(d) 瞬态 · PN：出口 RCR 一致性残差', **TITLE_KW)
legend(ax, loc='lower left')
for ax in axes.flat: ax.set_xlabel('epoch', color=INK2, fontsize=9)
fig.suptitle('图3 · 物理项动态：EMA 的 λ_pde 贴死上限，PDE 残差被压到 PDE-F 的 1/10 以下 —— 平凡解通道',
             fontsize=12.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=[0,0,1,0.94])
p=f'{OUT}/fig3_physics_dynamics.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= load eval data =================
FM = json.load(open('/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/audits/v4_workbook_0_14_20260823/field_metrics_v4_0_15.json'))
def wss_r2cb(run):
    d = json.load(open(f'/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/audits/v4_workbook_0_14_20260823/arms/{run}/wss_metrics.json'))
    r2 = [c['wss_r2'] for c in d['cases']]
    return float(np.mean(r2))
def eval_cases(t,b,m):
    run = run_name(t,b,m)
    d = json.load(open(f'{BASE}/{TDIR[t]}/{run}/evaluation_official_last_converged_full.json'))
    return d['cases']

ARMS16 = [(t,b,m) for t,b in GROUPS for m in MODES]
E_LAST = {a: FM['arms'][run_name(*a)]['e_rel_l2']['mean'] for a in ARMS16}
E_7500 = {a: FM['milestones'][run_name(*a)]['e_rel_l2']['mean'] for a in ARMS16 if run_name(*a) in FM.get('milestones',{})}
TRAIN_FINAL = {a: float(np.nanmean(CURVES[a][1]['mean_data_total'][-200:])) for a in ARMS16}
WSS = {a: wss_r2cb(run_name(*a)) for a in ARMS16}
NEARWALL = {a: FM['arms'][run_name(*a)]['r2']['near_wall_speed']['mean'] for a in ARMS16}

# ================= Fig 4: train fit vs test primary =================
fig, ax = plt.subplots(figsize=(9.5,6.6)); fig.patch.set_facecolor(SURF)
style_ax(ax, logy=False)
for a in ARMS16:
    t,b,m = a
    mk = 'o' if t=='SP' else '^'
    ax.scatter(TRAIN_FINAL[a], E_LAST[a], s=95 if b=='PN' else 60, marker=mk,
               color=MODE_COLOR[m], edgecolors=SURF, linewidths=1.5, zorder=3)
ax.axhline(1.0, color=INK2, lw=1.0, ls='--')
ax.annotate('E_rel=1.0：全零预测器的成绩', xy=(0.35,1.0), xytext=(0,5), textcoords='offset points',
            fontsize=9, color=INK2)
ax.annotate('训练拟合最好（DATA）\n→ 测试反而最差：过拟合', xy=(0.19,1.11), xytext=(0.24,1.14),
            fontsize=9.5, color=INK, arrowprops=dict(arrowstyle='->', color=INK2))
ax.annotate('EMA：训练几乎不拟合，\nE 靠幅值压缩贴到 0.9x', xy=(0.82,0.93), xytext=(0.52,0.965),
            fontsize=9.5, color=INK, arrowprops=dict(arrowstyle='->', color=INK2))
from matplotlib.lines import Line2D
h1 = [Line2D([],[], marker='s', ls='', color=MODE_COLOR[m], label=m) for m in MODES]
h2 = [Line2D([],[], marker='o', ls='', color=INK2, label='准稳态'),
      Line2D([],[], marker='^', ls='', color=INK2, label='瞬态'),
      Line2D([],[], marker='o', ls='', color=INK2, ms=10, label='大点=PN，小点=PNPP')]
ax.legend(handles=h1+h2, frameon=False, fontsize=8.5, loc='lower left', ncol=2)
ax.set_xlabel('训练末段 raw data_total（越小 = 训练拟合越好）', color=INK2, fontsize=10)
ax.set_ylabel('official test35 E_rel_l2（越小越好）', color=INK2, fontsize=10)
ax.set_title('图4 · 训练拟合与测试主指标完全脱钩：16 臂无一实质性优于零预测器',
             fontsize=12.5, color=INK, loc='left', pad=10)
fig.tight_layout()
p=f'{OUT}/fig4_overfit_scatter.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= Fig 5: primary bars + checkpoint sensitivity =================
fig, ax = plt.subplots(figsize=(10.5,7.6)); fig.patch.set_facecolor(SURF)
style_ax(ax, logy=False); ax.grid(True, axis='x', color=GRID, lw=0.7); ax.grid(False, axis='y')
ylabels=[]; ypos=[]; y=0
gtitle = {('SP','PN'):'准稳态 · PN', ('SP','PNPP'):'准稳态 · PNPP',
          ('TR','PN'):'瞬态 · PN', ('TR','PNPP'):'瞬态 · PNPP'}
for gi,(t,b) in enumerate(GROUPS):
    ax.annotate(gtitle[(t,b)], xy=(0.005, y), fontsize=10, color=INK, fontweight='bold',
                va='center', ha='left')
    y+=0.85
    for m in MODES:
        a=(t,b,m)
        ax.barh(y, E_LAST[a], height=0.72, color=MODE_COLOR[m], zorder=3)
        ax.annotate(f'{E_LAST[a]:.3f}', xy=(E_LAST[a], y), xytext=(11,0), textcoords='offset points',
                    fontsize=8.5, color=INK, va='center', zorder=4)
        if a in E_7500:
            ax.plot(E_7500[a], y, marker='D', ms=5.5, color=INK, zorder=5, ls='')
        ylabels.append(m); ypos.append(y); y+=1
    y+=0.55
ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=9, color=INK)
ax.invert_yaxis()
ax.axvline(1.0, color=INK2, lw=1.1, ls='--', zorder=4)
ax.annotate('零预测器 E=1.0', xy=(1.0,-1.4), fontsize=9, color=INK2, ha='center')
h = [Line2D([],[], marker='D', ls='', color=INK, ms=6, label='epoch 7500 checkpoint（黑菱形）')]
ax.legend(handles=h, frameon=False, fontsize=9, loc='lower right')
ax.set_xlim(0, 1.32)
ax.set_xlabel('E_rel_l2（case-balanced，越小越好）', color=INK2, fontsize=10)
ax.set_title('图5 · 主指标全景：全部 16 臂挤在 0.89–1.18；准稳态 8 臂在 7500 时都更好（黑菱形）→ 越训越差',
             fontsize=12, color=INK, loc='left', pad=10)
fig.tight_layout()
p=f'{OUT}/fig5_primary_bars.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= Fig 6: variance ratio boxes =================
fig, ax = plt.subplots(figsize=(11,6.2)); fig.patch.set_facecolor(SURF)
style_ax(ax, logy=True); ax.grid(True, axis='y', color=GRID, lw=0.7)
pos=1; xt=[]; xl=[]
for gi,(t,b) in enumerate(GROUPS):
    for m in MODES:
        cs = eval_cases(t,b,m)
        vr = [c['metrics']['speed']['prediction_to_truth_variance_ratio'] for c in cs]
        vr = [v for v in vr if np.isfinite(v) and v>0]
        bp = ax.boxplot([vr], positions=[pos], widths=0.66, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.4),
                        boxprops=dict(facecolor=MODE_COLOR[m], edgecolor=SURF, lw=1),
                        whiskerprops=dict(color=INK2, lw=1), capprops=dict(color=INK2, lw=1))
        xt.append(pos); xl.append(m.replace('BC+PDE-','PDE-')); pos+=1
    ax.annotate(gtitle[(t,b)], xy=(pos-3, 4.5), fontsize=10, color=INK, fontweight='bold', ha='center')
    pos+=0.8
ax.axhline(1.0, color=INK2, lw=1.1, ls='--')
ax.annotate('方差比=1：幅值正确', xy=(0.6,1.0), xytext=(0,4), textcoords='offset points', fontsize=9, color=INK2)
ax.annotate('瞬态·PN·EMA 中位数 ≈ 0.0003：近乎纯零场 ↓', xy=(11.8, 2.2e-4), fontsize=9, color=INK, ha='center')
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8.5, color=INK, rotation=25, ha='right')
ax.set_ylim(5e-6, 7)
ax.set_ylabel('逐病例 speed 方差比 pred/truth（log）', color=INK2, fontsize=10)
ax.set_title('图6 · 幅值压缩阶梯：DATA→BC→PDE-F→EMA 预测方差被逐级压向零 —— 物理臂的 E 增益主要来自“更保守”，不是“更正确”',
             fontsize=11.5, color=INK, loc='left', pad=10)
fig.tight_layout()
p=f'{OUT}/fig6_variance_ratio.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= Fig 7: transient pressure outliers =================
fig, ax = plt.subplots(figsize=(9.5,6.2)); fig.patch.set_facecolor(SURF)
style_ax(ax, logy=False)
cs = eval_cases('TR','PN','BC+PDE-F')
xs=[]; ys=[]; names=[]
for c in cs:
    m=c['metrics']['pressure']
    xs.append(m['truth_mean']); ys.append(m['r2']); names.append(c['case_id'])
ys_clip = [max(v,-8) for v in ys]
ax.scatter(xs, ys_clip, s=70, color='#2a78d6', edgecolors=SURF, linewidths=1.2, zorder=3)
ax.axhline(0, color=INK2, lw=1.0, ls='--')
med = float(np.median(ys))
ax.annotate(f'35 例中位数 R² = {med:+.2f}（29 例 > 0）', xy=(2500, 0.55), fontsize=10, color=INK)
outl = sorted([(x,v,n) for x,v,n in zip(xs,ys,names) if v < -8], key=lambda r: r[0])
for i,(x,v,n) in enumerate(outl):
    short = n.split('/')[-1] if 'before' not in n else n.split('/')[1]
    ax.annotate(f'{short}：真值 gauge≈{x:.0f} Pa，R²={v:.0f}', xy=(x,-8), xytext=(x+700,-5.4-1.5*i),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='->', color=INK2))
ax.annotate('← 全库主流 gauge 13–16 kPa', xy=(11400,-1.3), fontsize=9, color=INK2)
ax.set_ylim(-8.6, 1.4)
ax.set_xlabel('病例真值 gauge 压力均值（Pa）', color=INK2, fontsize=10)
ax.set_ylabel('逐病例 pressure R²（截断于 −8）', color=INK2, fontsize=10)
ax.set_title('图7 · 瞬态压力 R̄²=−20 的真相（TR-PN-PDE-F）：2 个 gauge 口径异常病例拉爆均值，其余 33 例大多良好',
             fontsize=11.5, color=INK, loc='left', pad=10)
fig.tight_layout()
p=f'{OUT}/fig7_pressure_outliers.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= Fig 8: WSS vs near-wall =================
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12.5,6.0), gridspec_kw={'width_ratios':[1.25,1]})
fig.patch.set_facecolor(SURF)
# left: WSS bars
style_ax(ax1, logy=False); ax1.grid(True, axis='x', color=GRID, lw=0.7); ax1.grid(False, axis='y')
ylabels=[]; ypos=[]; y=0
for gi,(t,b) in enumerate(GROUPS):
    for m in MODES:
        a=(t,b,m); v=max(WSS[a], -8.4)
        ax1.barh(y, v, height=0.72, color=MODE_COLOR[m], zorder=3)
        lab = f'{WSS[a]:.2f}'
        ax1.annotate(lab, xy=(min(v,0), y), xytext=(-4,0), textcoords='offset points',
                     fontsize=8, color=INK, va='center', ha='right')
        ylabels.append(m.replace('BC+PDE-','PDE-')); ypos.append(y); y+=1
    ax1.annotate(gtitle[(t,b)], xy=(-8.3, y-2.5), fontsize=9.5, color=INK, fontweight='bold', va='center')
    y+=0.9
ax1.set_yticks(ypos); ax1.set_yticklabels(ylabels, fontsize=8.5, color=INK)
ax1.invert_yaxis()
ax1.axvline(0, color=INK2, lw=1.0, ls='--')
ax1.set_xlim(-8.6, 1.6)
ax1.annotate('真速度输入上限 ≈ 0.96 →', xy=(1.55, 0.2), fontsize=8.5, color=INK2, ha='right')
ax1.set_xlabel('下游 WSS R²_cb（1200 壁面点 · Profile-Secant V3）', color=INK2, fontsize=9.5)
ax1.set_title('(a) 16 臂 WSS 下游结果', **TITLE_KW)
# right: coupling scatter (SP only, near-wall defined)
style_ax(ax2, logy=False)
for t,b in [('SP','PN'),('SP','PNPP')]:
    for m in MODES:
        a=(t,b,m)
        ax2.scatter(NEARWALL[a], WSS[a], s=95 if b=='PN' else 60, marker='o',
                    color=MODE_COLOR[m], edgecolors=SURF, linewidths=1.5, zorder=3)
ax2.axhline(0, color=INK2, lw=0.8, ls=':'); ax2.axvline(0, color=INK2, lw=0.8, ls=':')
ax2.set_xlabel('near-wall speed R²_cb（test35）', color=INK2, fontsize=9.5)
ax2.set_ylabel('WSS R²_cb', color=INK2, fontsize=9.5)
ax2.set_title('(b) 准稳态 8 臂：WSS 完全跟随近壁场质量', **TITLE_KW)
h1 = [Line2D([],[], marker='s', ls='', color=MODE_COLOR[m], label=m) for m in MODES]
ax2.legend(handles=h1, frameon=False, fontsize=8.5, loc='lower right')
fig.suptitle('图8 · WSS 全负不是 WSS 算法问题：排序与近壁速度 R² 同构，误差被梯度算子放大',
             fontsize=12.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=[0,0,1,0.93])
p=f'{OUT}/fig8_wss_nearwall.png'; fig.savefig(p, dpi=150, facecolor=SURF); figs.append(p); plt.close(fig)

# ================= combined PDF =================
from PIL import Image
pdf_path = f'{OUT}/V4_seed1234_0-15_训练曲线与诊断图_2026-08-31.pdf'
imgs = [Image.open(f).convert('RGB') for f in figs]
imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:], resolution=150)
print('DONE')
for f in figs: print(f)
print(pdf_path)
