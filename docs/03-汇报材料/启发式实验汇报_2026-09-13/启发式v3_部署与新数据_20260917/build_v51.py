"""图 06-09：v5.1 新底座 X5D_v51、三项后续臂，以及抽稀 / 噪声 / 重采样复评。"""
import numpy as np
from plot_style import (page, panel_label, save, INK, MUTED, BLUE, TEAL,
                        PURPLE, ORANGE, RED, GREY, LINE, LIGHT)
import datasrc as D

ARMS = (('X5Dcap', '协议版 capfit', '候选', TEAL),
        ('X5Ddual', '双尺度 patch', '不进底座', PURPLE),
        ('X5Dnoise', '噪声增广', '不进底座', ORANGE))


# --------------------------------------------------------- 06 新底座总览
def baseline_overview():
    cur = D.load(D.W1 / 'analysis_20260917' / 'current_residuals.json')
    cv3 = D.load(D.W2A / 'offline' / 'analyze_cv3.json')
    wave1 = D.load(D.W1 / 'offline' / 'analyze_wave1.json')

    seeds = {k: v['pa_r2_cb'] for k, v in wave1.items()
             if k.startswith('X5D_v51_s')}
    ens5 = wave1['X5D_v51 5-seed (Pa mean)']['pa_r2_cb']
    folds = {k: v['pa_r2_cb'] for k, v in cv3.items() if k.startswith('fold')}
    pooled = cv3['pooled']['pa_r2_cb']
    t34, oof = cur['test34_base5'], cur['cv3_oof']

    payload = {'test34_seeds': seeds, 'test34_ensemble_5seed': ens5,
               'cv3_folds': folds, 'cv3_pooled': pooled,
               'test34_summary': t34['summary'], 'cv3_summary': oof['summary'],
               'test34_cohort': {k: v['pa']['r2'] for k, v in t34['per_cohort'].items()},
               'cv3_cohort': {k: v['pa']['r2'] for k, v in oof['per_cohort'].items()}}
    src = D.dump('06_x5d_v51_baseline', payload)

    fig = page('新底座 X5D_v51：test34 集成 0.7749，患者分组折外 0.7100',
               'a 方法比较口径（test34）与泛化口径（cv3 折外）分开看｜b 逐病例 R² 的分布｜c 分队列：ILO 折外最弱',
               kicker='P24 · v5.1 新底座 · §28.1、§28.6、§28.9',
               footer='v5.1 数据，170 例 train136 / test34，best checkpoint，Pa 均值集成。0.7749 是 test34 五 seed 集成，0.7100 是 136 例折外（每折单 seed），'
                      '两者混有病例构成、训练量与集成数，不能相减当作过拟合量。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .06, 'right': .975, 'bottom': .175,
                                          'top': .66, 'wspace': .32})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    a.scatter([0] * len(seeds), list(seeds.values()), s=58, color=GREY, zorder=3)
    a.scatter([0], [ens5], s=150, marker='D', color=TEAL, zorder=4)
    a.text(.12, ens5, f'五 seed 集成\n{ens5:.4f}', fontsize=12.5, color=TEAL, va='center')
    a.scatter([1] * len(folds), list(folds.values()), s=58, color=GREY, zorder=3)
    a.scatter([1], [pooled], s=150, marker='D', color=ORANGE, zorder=4)
    a.text(1.12, pooled, f'合并折外\n{pooled:.4f}', fontsize=12.5, color=ORANGE, va='center')
    for x, vals in ((0, seeds.values()), (1, folds.values())):
        a.plot([x - .08, x + .08], [min(vals)] * 2, color=GREY, lw=.8)
        a.plot([x - .08, x + .08], [max(vals)] * 2, color=GREY, lw=.8)
        a.plot([x, x], [min(vals), max(vals)], color=GREY, lw=.8)
    a.set_xticks([0, 1], ['test34\n五 seed（方法比较）', 'cv3 折外\n三折 136 例（泛化）'],
                 fontsize=12.5)
    a.set_xlim(-.35, 1.75)
    a.set_ylim(.655, .79)
    a.set_ylabel('Pa R²$_{cb}$')
    a.set_title('两个口径要分开报，不要并排比', loc='left', fontsize=14, pad=12)

    for scope, color, name in ((t34, TEAL, 'test34 五 seed 集成（34 例）'),
                               (oof, ORANGE, 'cv3 折外（136 例）')):
        r2 = np.sort([p['r2_pa'] for p in scope['per_case']])
        q = np.linspace(0, 100, len(r2))
        b.plot(q, r2, '-', color=color, lw=2, label=name)
        p10 = scope['summary']['case_r2_p10']
        b.scatter([10], [p10], s=60, color=color, zorder=4)
        b.text(12, p10 - .012, f'P10 {p10:.3f}', fontsize=12, color=color)
    b.set_xlabel('病例分位（%）')
    b.set_ylabel('逐病例 Pa R²')
    b.set_ylim(.25, 1.0)
    b.legend(loc='lower right', fontsize=11.5)
    b.set_title('折外的差病例端更长：P10 0.680 → 0.600', loc='left', fontsize=14, pad=12)

    cohorts = ['AG', 'AAA', 'ILO']
    t_vals = [t34['per_cohort'][k]['pa']['r2'] for k in cohorts]
    o_vals = [oof['per_cohort'][k]['pa']['r2'] for k in cohorts]
    x = np.arange(3)
    c.bar(x - .19, t_vals, .36, color=TEAL, edgecolor=INK, lw=.6, label='test34 集成')
    c.bar(x + .19, o_vals, .36, color=ORANGE, edgecolor=INK, lw=.6, label='cv3 折外')
    for xi, v in zip(x - .19, t_vals):
        c.text(xi, v + .006, f'{v:.3f}', ha='center', fontsize=11.5, color=INK)
    for xi, v in zip(x + .19, o_vals):
        c.text(xi, v + .006, f'{v:.3f}', ha='center', fontsize=11.5, color=INK)
    c.set_xticks(x, [f'{k}\n({t34["per_cohort"][k]["n"]} / {oof["per_cohort"][k]["n"]} 例)'
                     for k in cohorts], fontsize=12.5)
    c.set_ylim(0, 1.06)
    c.set_ylabel('Pa R²$_{cb}$')
    c.legend(loc='upper right', fontsize=11.5)
    c.set_title('ILO 折外 0.624，仍是最弱队列（见 P31）', loc='left', fontsize=14, pad=12)

    return save(fig, '06_x5d_v51_baseline',
                claim='X5D_v51 五 seed 集成 0.7749 成为新部署底座；患者分组折外 0.7100，ILO 队列最弱。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Panel b pools per-case R2 from two different case sets (34 vs 136) and is a distribution comparison, not a paired test. Cohort counts differ between the two scopes.')


# ------------------------------------------------------ 07 三项后续臂配对
def three_arms():
    wave1 = D.load(D.W1 / 'offline' / 'analyze_wave1.json')
    base3 = wave1['X5D_v51 3-seed (Pa mean)']['pa_r2_cb']
    paired = {a: wave1[f'{a}_paired'] for a, *_ in ARMS}
    ens = {a: wave1[f'{a} 3-seed (Pa mean)']['pa_r2_cb'] for a, *_ in ARMS}
    payload = {'baseline_3seed': base3, 'paired': paired, 'ensemble_3seed': ens}
    src = D.dump('07_three_arms_paired', payload)

    fig = page('三项后续臂：只有 capfit 是小而一致的改善',
               'a 同 seed 配对的物理 R²_cb 变化｜b 归一化 R²_cb 变化｜c 三 seed 集成相对底座',
               kicker='P25 · v5.1 波 1 三臂 · §28.2、§28.3',
               footer='与同 seed 的 X5D_v51 配对（初始权重逐位相同，只改一处）。单 seed 带 ±0.034 物理 / ±0.009 归一化（§28.2 口径）；'
                      '三个点是 seed 1234 / 7 / 2025，横线为均值。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .065, 'right': .975, 'bottom': .20,
                                          'top': .66, 'wspace': .28})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    def delta_panel(ax, key, band, title, ylabel):
        for i, (arm, name, verdict, color) in enumerate(ARMS):
            vals = [p[key] for p in paired[arm]]
            ax.scatter([i] * len(vals), vals, s=62, color=color, zorder=3)
            m = float(np.mean(vals))
            ax.plot([i - .22, i + .22], [m, m], color=color, lw=2.4, zorder=4)
            ax.text(i, max(vals) + band * .10, f'均值 {m:+.4f}', ha='center',
                    fontsize=12, color=color)
            ax.text(i, -band * .93, f'{sum(v > 0 for v in vals)}/3 正', ha='center',
                    fontsize=11.5, color=MUTED)
        ax.axhspan(-band, band, color=LIGHT, zorder=0)
        ax.axhline(0, color=INK, lw=.9, ls='--')
        ax.set_xticks(range(3), [f'{n}\n{v}' for _, n, v, _ in ARMS], fontsize=12.5)
        ax.set_xlim(-.6, 2.6)
        ax.set_ylim(-band * 1.12, band * 1.12)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=14, pad=12)

    delta_panel(a, 'd_pa', .034, '物理 R²：三臂全部落在单 seed 带内', 'Δ Pa R²$_{cb}$（臂 − 底座）')
    delta_panel(b, 'd_norm', .009, '归一化：只有 capfit 三个 seed 全正', 'Δ 归一化 R²$_{cb}$')

    names = [n for _, n, _, _ in ARMS]
    vals = [ens[a] - base3 for a, *_ in ARMS]
    colors = [c for *_, c in ARMS]
    c.bar(range(3), vals, .5, color=colors, edgecolor=INK, lw=.6)
    for i, v in enumerate(vals):
        c.text(i, v + (.0004 if v > 0 else -.0004), f'{v:+.4f}',
               ha='center', va='bottom' if v > 0 else 'top', fontsize=12.5, color=INK)
    c.axhline(0, color=INK, lw=.9, ls='--')
    c.set_xticks(range(3), [f'{n}\n{ens[a]:.4f}' for (a, n, _, _) in ARMS], fontsize=12.5)
    c.set_ylabel(f'集成 Δ Pa R²$_{{cb}}$（底座 {base3:.4f}）')
    c.set_ylim(-.010, .006)
    c.set_title('集成口径：capfit +0.0035，另两臂为负', loc='left', fontsize=14, pad=12)

    return save(fig, '07_three_arms_paired',
                claim='协议版 capfit 归一化 3/3 为正、集成 +0.0035，列为候选；双尺度 patch 与噪声增广在全密度下净收益为负，不进底座。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Shaded band is the project single-seed read-out band, not a significance test; n=3 paired seeds per arm is too small for one.')


# ------------------------------------------------------------- 08 抽稀复评
def thinning_reeval():
    base = D.resample_rows(D.W1R / 'offline' / 'deployment_reeval_x5d_v51' /
                           'deployment_resample_reeval.json', pa_mean=True)
    dual = D.resample_rows(D.W1R / 'offline' / 'deployment_reeval_x5ddual' /
                           'deployment_resample_reeval.json', pa_mean=True)
    payload = {'X5D_v51': base, 'X5Ddual': dual}
    src = D.dump('08_thinning_reeval', payload)

    fig = page('双尺度 patch 确实做到了抽稀鲁棒 —— 但那不是现在的部署形态',
               'a 冻结特征：只看模型的邻域层｜b 重算特征：连曲率 / 法向一起在稀云上重算',
               kicker='P26 · 抽稀复评 · §28.4a',
               footer='三 seed Pa 均值集成，Δ 对同点全云参照。冻结特征 = 沿用全云算好的几何特征，只让模型的 support / patch 变稀；'
                      '重算 = 在稀云上重新算几何。现行部署口径是 STL 重采样 0.5 mm（≈50% 档以上），不是 10%。')
    axs = fig.subplots(1, 2, gridspec_kw={'left': .075, 'right': .975, 'bottom': .175,
                                          'top': .66, 'wspace': .22})
    fr = [1.0, .5, .25, .1]
    for ax, variant, title in zip(axs, ('frozen', 'recompute'),
                                  ('冻结特征：dual 把损失缩到 1/4 – 1/3',
                                   '重算特征：优势缩小，大头是重算的几何')):
        ax.set_gid(chr(97 + list(axs).index(ax)))
        panel_label(ax, chr(97 + list(axs).index(ax)))
        series = {}
        for rows, color, name in ((base, TEAL, 'X5D_v51 底座'), (dual, PURPLE, 'X5Ddual 双尺度')):
            look = {r['fraction']: r for r in rows if r['variant'] == variant}
            look[1.0] = [r for r in rows if r['fraction'] == 1.0][0]
            ys = [look[f]['r2'] - look[f]['ref'] for f in fr]
            ax.plot(range(len(fr)), ys, 'o-', color=color, lw=1.9, ms=7, label=name)
            series[color] = ys
        (ca, ya), (cb, yb) = series.items()
        for i in range(1, len(fr)):
            hi, lo = ((ya[i], ca), (yb[i], cb)) if ya[i] >= yb[i] else ((yb[i], cb), (ya[i], ca))
            ax.text(i, hi[0] + .007, f'{hi[0]:.3f}', ha='center', va='bottom',
                    fontsize=11.5, color=hi[1])
            ax.text(i, lo[0] - .007, f'{lo[0]:.3f}', ha='center', va='top',
                    fontsize=11.5, color=lo[1])
        ax.axhline(0, color=LINE, lw=.9)
        ax.set_xlim(-.2, 3.35)
        ax.set_xticks(range(len(fr)), ['100%', '50%', '25%', '10%'])
        ax.set_xlabel('保留的壁面点比例')
        ax.set_ylabel('Δ Pa R²$_{cb}$（对同点参照）')
        ax.set_ylim(-.225, .045)
        ax.legend(loc='lower left', fontsize=12)
        ax.set_title(title, loc='left', fontsize=14, pad=12)

    return save(fig, '08_thinning_reeval',
                claim='双尺度 patch 在 10% 抽稀下把冻结特征的损失从 −0.146 缩到 −0.056，达成设计目标，但代价是全密度 −0.007。',
                sources=src, axes=list(axs), row_groups=[['a', 'b']],
                note='Full-density rows are exactly 0 by construction; the two models have different full-density baselines (0.7713 vs 0.7633), so only the deltas are comparable here.')


# ------------------------------------------- 09 STL 重采样 + 几何噪声复评
def stl_and_noise():
    stl_base = D.stl_rows(D.W1R / 'offline' / 'stl_x5d_v51' /
                          'deployment_stl_simulation.json', pa_mean=True)
    stl_dual = D.stl_rows(D.W1R / 'offline' / 'stl_x5ddual' /
                          'deployment_stl_simulation.json', pa_mean=True)
    geo_base = D.geometry_rows(D.W1R / 'offline' / 'geometry_x5d_v51' /
                               'geometry_sensitivity.json', pa_mean=True)
    geo_noise = D.geometry_rows(D.W1R / 'offline' / 'geometry_x5dnoise' /
                                'geometry_sensitivity.json', pa_mean=True)
    payload = {'stl_X5D_v51': stl_base, 'stl_X5Ddual': stl_dual,
               'geometry_X5D_v51': geo_base, 'geometry_X5Dnoise': geo_noise}
    src = D.dump('09_stl_and_noise_reeval', payload)

    fig = page('放回现行部署口径后，两个鲁棒性臂的净收益都归零',
               'a STL 重采样 0.5–1.2 mm：dual 与底座互有胜负｜b 几何噪声：加了 1 mm 平滑后噪声增广只差 +0.002',
               kicker='P27 · 部署口径下的净效应 · §28.4b、§28.4c',
               footer='三 seed Pa 均值集成，34 例。结论：部署主防线仍是「重采样 0.5 mm + 平滑 1 mm」；'
                      'dual 留作点云粗于 1.5 mm 的备选，噪声增广只在拿不到平滑表面、噪声 ≥0.2 mm 时才值得开。')
    axs = fig.subplots(1, 2, gridspec_kw={'left': .075, 'right': .975, 'bottom': .19,
                                          'top': .66, 'wspace': .22})
    a, b = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    xs = np.arange(len(stl_base))
    labels = [f'{r["actual"]:.2f} mm' for r in stl_base]
    a.plot(xs, [r['r2'] for r in stl_base], 'o-', color=TEAL, lw=2, ms=7,
           label='X5D_v51 底座')
    a.plot(xs, [r['r2'] for r in stl_dual], 's-', color=PURPLE, lw=2, ms=7,
           label='X5Ddual 双尺度')
    for x, rb, rd in zip(xs, stl_base, stl_dual):
        hi, lo = ((rb['r2'], TEAL), (rd['r2'], PURPLE)) if rb['r2'] >= rd['r2'] \
            else ((rd['r2'], PURPLE), (rb['r2'], TEAL))
        a.text(x, hi[0] + .0028, f'{hi[0]:.4f}', ha='center', va='bottom',
               fontsize=11, color=hi[1])
        a.text(x, lo[0] - .0028, f'{lo[0]:.4f}', ha='center', va='top',
               fontsize=11, color=lo[1])
    diffs = [d['r2'] - s['r2'] for s, d in zip(stl_base, stl_dual)]
    for x, v in zip(xs, diffs):
        a.text(x, .7875, f'{v:+.4f}', ha='center', fontsize=12,
               color=TEAL if v < 0 else PURPLE)
    a.set_xticks(xs, labels, fontsize=12)
    a.set_xlim(-.45, 3.45)
    a.set_xlabel('重采样后实际中位间距')
    a.set_ylabel('集成 Pa R²$_{cb}$')
    a.set_ylim(.732, .793)
    a.legend(loc='lower left', fontsize=11.5)
    a.set_title('净差无一致符号（上排数字 = dual − 底座）', loc='left',
                fontsize=14, pad=12)

    keys = ['clean', 'smooth_s1.0', 'noise_0.2', 'noise_0.4', 'noise_0.2+smooth_s1.0']
    names = ['仅重采样', '平滑 1.0', '噪声 0.2', '噪声 0.4', '噪声 0.2\n+ 平滑 1.0']
    lb = {r['variant']: r for r in geo_base}
    ln = {r['variant']: r for r in geo_noise}
    xs = np.arange(len(keys))
    vb = [lb[k]['r2'] - lb[k]['ref'] for k in keys]
    vn = [ln[k]['r2'] - ln[k]['ref'] for k in keys]
    b.bar(xs - .19, vb, .36, color=TEAL, edgecolor=INK, lw=.6, label='X5D_v51 底座')
    b.bar(xs + .19, vn, .36, color=ORANGE, edgecolor=INK, lw=.6, label='X5Dnoise 噪声增广')
    for x, v in zip(xs - .19, vb):
        b.text(x, v - .003, f'{v:.3f}', ha='center', va='top', fontsize=10.5, color=INK)
    for x, v in zip(xs + .19, vn):
        b.text(x, v - .003, f'{v:.3f}', ha='center', va='top', fontsize=10.5, color=INK)
    for x, (p, q) in zip(xs, zip(vb, vn)):
        b.text(x, .014, f'{q - p:+.3f}', ha='center', fontsize=12,
               color=ORANGE if q > p else TEAL)
    b.axhline(0, color=LINE, lw=.9)
    b.set_xticks(xs, names, fontsize=12)
    b.set_ylabel('Δ Pa R²$_{cb}$（对同点参照）')
    b.set_ylim(-.168, .030)
    b.legend(loc='lower left', fontsize=11.5)
    b.set_title('先平滑后差距只剩 +0.002（上排数字 = 噪声增广 − 底座）', loc='left',
                fontsize=14, pad=12)

    return save(fig, '09_stl_and_noise_reeval',
                claim='在现行部署口径（重采样 0.5 mm + 平滑 1 mm）下，双尺度 patch 与噪声增广相对底座的净收益都归零，不进部署底座。',
                sources=src, axes=list(axs), row_groups=[['a', 'b']],
                note='Panel a compares absolute ensemble R2 across models at the same spacing; panel b compares same-point deltas. The two arms have different full-cloud baselines, which is why both views are shown.')


if __name__ == '__main__':
    baseline_overview()
    three_arms()
    thinning_reeval()
    stl_and_noise()
