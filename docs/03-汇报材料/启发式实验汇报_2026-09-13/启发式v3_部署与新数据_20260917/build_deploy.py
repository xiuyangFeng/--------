"""图 01-03：部署形态是怎么定下来的（只读缓存，不训练、不推理）。"""
import numpy as np
from plot_style import (page, canvas, node, arrow, text, panel_label, save,
                        INK, MUTED, BLUE, TEAL, PURPLE, ORANGE, RED, GREY, LINE, LIGHT)
import datasrc as D

W6O = D.W6 / 'offline'


def fmt(v, digits=3):
    return f'{v:+.{digits}f}'


# --------------------------------------------------------------- 01 决策链
def decision_chain():
    spacing = D.wall_spacing()
    sp = np.array([r['spacing_mm'] for r in spacing])
    p10, p50, p90 = np.percentile(sp, [10, 50, 90])
    cohort = {c: float(np.median([r['spacing_mm'] for r in spacing if r['cohort'] == c]))
              for c in ('AG', 'AAA', 'ILO')}

    stl_x5 = {r['target']: r for r in
              D.stl_rows(W6O / 'stl_x5' / 'deployment_stl_simulation.json')}
    stl_x5d = {r['target']: r for r in
               D.stl_rows(W6O / 'stl_x5d' / 'deployment_stl_simulation.json')}
    fine = D.stl_rows(W6O / 'stl_x5_fine' / 'deployment_stl_simulation.json')[0]
    dec = {(r['fraction'], r['variant']): r for r in
           D.decomposition_rows(D.FOLLOW / 'decomposition_x5d' /
                                'neighbourhood_decomposition.json')}
    res_x5 = {(r['fraction'], r['variant']): r for r in
              D.resample_rows(W6O / 'deployment_reeval_x5_3seed' /
                              'deployment_resample_reeval.json')}
    res_x5d = {(r['fraction'], r['variant']): r for r in
               D.resample_rows(W6O / 'deployment_reeval_x5d' /
                               'deployment_resample_reeval.json')}
    geom = {r['variant']: r for r in
            D.geometry_rows(D.FOLLOW / 'geometry_x5d5' / 'geometry_sensitivity.json')}
    rough = {r['variant']: r for r in
             D.geometry_rows(D.FOLLOW / 'geometry_x5d5_b3' / 'geometry_sensitivity.json')}
    tta, _ = D.tta_rows(D.FOLLOW / 'tta_x5d5' / 'sampling_tta.json')
    tta = {r['label']: r['r2'] for r in tta}
    ens5 = D.load(D.W6B / 'five_seed_ensemble.json')

    d = lambda r: r['r2'] - r['ref']
    payload = {
        'wall_spacing_mm': {'p10': p10, 'p50': p50, 'p90': p90, 'cohort_median': cohort,
                            'n_cases': len(spacing)},
        'stl_delta_X5': {'0.5mm': d(stl_x5[0.5]), '1.2mm': d(stl_x5[1.2]),
                         '0.35mm': d(fine)},
        'stl_delta_X5D_3seed': {'0.5mm': d(stl_x5d[0.5]), '1.2mm': d(stl_x5d[1.2])},
        'decomposition_25pct': {k[1]: d(v) for k, v in dec.items() if k[0] == 0.25},
        'voxel_full_density': {'X5': res_x5[(1.0, 'recompute')]['r2'],
                               'X5D': res_x5d[(1.0, 'recompute')]['r2']},
        'geometry_delta': {k: d(v) for k, v in geom.items()},
        'rough_delta': {k: d(v) for k, v in rough.items()},
        'tta': tta, 'five_seed_ensemble_v50': ens5,
    }
    src = D.dump('01_deployment_decision', payload)

    fig = page('部署形态是怎么定下来的',
               '六步只推理实验：先量训练数据自身密度，再从原始 STL 重建部署几何，定位损失，最后用密度增广与集成把曲线压平',
               kicker='P19 · 部署形态 · 跟踪文档 §21.4、§22.0–§22.6',
               footer='全部为只读推理（无训练）。Δ 一律对「同一 checkpoint 的全云预测限制到同一批点」的同点参照，不用不同点集的绝对 R² 当增益。')
    ax = canvas(fig, (.035, .155, .93, .65))

    cards = [
        (2, 54, BLUE, '① 训练数据自己的密度不是一个数',
         f'172 例壁面节点最近邻间距\np10 / p50 / p90 = {p10:.2f} / {p50:.2f} / {p90:.2f} mm\n'
         f'AG {cohort["AG"]:.2f}｜AAA {cohort["AAA"]:.2f}｜ILO {cohort["ILO"]:.2f} mm（AG 粗 2.5 倍）'),
        (35.5, 54, BLUE, '② 部署不是抽稀 CFD 网格，是从 STL 重建',
         '34 例原始 STL 面积加权重采样\n中心线投影、解剖架、PCA 法向、曲率族、\nflowref 全部从零重算；真值取最近 CFD 节点'),
        (69, 54, PURPLE, '③ 损失出在哪：邻域分解',
         f'25% 抽稀 thin/thin {d(dec[(0.25, "thin/thin")]):.3f}\n'
         f'只抽稀 patch {d(dec[(0.25, "full/thin")]):.3f}｜只抽稀 support {d(dec[(0.25, "thin/full")]):.3f}\n'
         '→ 代价几乎全在 K16 query patch'),
        (69, 8, TEAL, '④ 密度增广训练 X5D',
         f'70 / 50 / 35 / 25% 四档随机密度\nSTL 1.2 mm 档 {d(stl_x5[1.2]):.3f} → {d(stl_x5d[1.2]):.3f}\n'
         f'全密度不变（{res_x5[(1.0, "recompute")]["r2"]:.4f} → {res_x5d[(1.0, "recompute")]["r2"]:.4f}）'),
        (35.5, 8, TEAL, '⑤ 几何保真合同',
         f'重采样 0.5 mm 是甜点（X5 {d(stl_x5[0.5]):.3f}）\n'
         f'平滑 1 mm 几乎免费 {d(geom["smooth_s1.0"]):.3f}｜噪声 0.2 mm {d(geom["noise_0.2"]):.3f}\n'
         f'亚毫米粗糙 {d(rough["noise_0.2c0.5"]):.3f}，先平滑收回到 {d(rough["noise_0.2c0.5+smooth_s1.0"]):.3f}'),
        (2, 8, ORANGE, '⑥ 部署底座 = 五 seed 集成',
         f'X5D 五 seed Pa 均值 {ens5["X5D"]["ens_pa"]:.4f}（旧数据口径）\n'
         f'重采样 TTA 不值得：5 次 {tta["tta_5"] - tta["reference_full_cloud"]:.3f} vs 单次 {tta["single"] - tta["reference_full_cloud"]:.3f}\n'
         'v5.1 修正数据上重训 → 0.7749（见 P24）'),
    ]
    for x, y, color, title, body in cards:
        node(ax, x, y, 29, 34, title, body, color=color, fontsize=14.5, body_size=11.5)

    arrow(ax, (31.2, 71), (35.3, 71))
    arrow(ax, (64.7, 71), (68.8, 71))
    arrow(ax, (83.5, 53.6), (83.5, 42.6))
    arrow(ax, (68.8, 25), (64.7, 25))
    arrow(ax, (35.3, 25), (31.2, 25))

    text(ax, 50, -9,
         '部署合同：壁面点云重采样到中位间距 ≈0.5 mm → 先做 1 mm 平滑 → X5D 五 seed Pa 均值集成；'
         '髂动脉切口须与 CFD 取同一解剖层面（±5 mm 可容忍）',
         13.5, INK, True, 'center')
    return save(fig, '01_deployment_decision',
                claim='部署形态由六步只推理实验定下：密度不是单值、损失在 query patch、密度增广压平曲线、平滑几乎免费、五 seed 集成为底座。',
                sources=src,
                note='Schematic page; every number is read from the listed JSON artefacts. Deltas are against same-point full-cloud references, not across different point sets.')


# ------------------------------------------------------------- 02 密度扫描
def density_curve():
    x5 = D.stl_rows(W6O / 'stl_x5' / 'deployment_stl_simulation.json')
    fine = D.stl_rows(W6O / 'stl_x5_fine' / 'deployment_stl_simulation.json')
    x5 = sorted(x5 + fine, key=lambda r: r['actual'])
    x5d = D.stl_rows(W6O / 'stl_x5d' / 'deployment_stl_simulation.json')
    res_x5 = D.resample_rows(W6O / 'deployment_reeval_x5_3seed' /
                             'deployment_resample_reeval.json')
    res_x5d = D.resample_rows(W6O / 'deployment_reeval_x5d' /
                              'deployment_resample_reeval.json')
    dec = D.decomposition_rows(D.FOLLOW / 'decomposition_x5d' /
                               'neighbourhood_decomposition.json')

    payload = {'stl_X5_5seed': x5, 'stl_X5D_3seed': x5d,
               'voxel_X5_3seed': res_x5, 'voxel_X5D_3seed': res_x5d,
               'neighbourhood_X5D': dec}
    src = D.dump('02_density_curve', payload)

    fig = page('点云多稀才会掉分：X5 掉，X5D 不掉',
               'a 从原始 STL 重采样（部署形态）｜b 体素抽稀 CFD 节点（拆解用）｜c 把损失拆到 support 与 query patch 两层',
               kicker='P20 · 密度扫描 · §22.1、§22.3、§22.6-A',
               footer='34 例 test34，best checkpoint；a 为集成 Pa R²_cb，b/c 为对同点全云参照的 Δ。"native" = 按该例 CFD 节点数重采样（中位 0.96 mm）。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .065, 'right': .975, 'bottom': .18,
                                          'top': .66, 'wspace': .30})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    # a: absolute R2 vs actual spacing
    a.plot([r['actual'] for r in x5], [r['r2'] for r in x5], 'o-', color=PURPLE,
           lw=1.8, ms=6, label='X5（五 seed 集成）')
    a.plot([r['actual'] for r in x5d], [r['r2'] for r in x5d], 's-', color=TEAL,
           lw=1.8, ms=6, label='X5D 密度增广（三 seed）')
    a.axhline(np.mean([r['ref'] for r in x5]), color=GREY, ls='--', lw=1.0)
    a.text(1.20, np.mean([r['ref'] for r in x5]) + .006, '同点全云参照', fontsize=11,
           color=MUTED, ha='right')
    best = min(x5, key=lambda r: abs(r['actual'] - .5))
    a.annotate(f'0.5 mm 甜点\n{best["r2"] - best["ref"]:+.3f}', (best['actual'], best['r2']),
               xytext=(0, -46), textcoords='offset points', ha='center', fontsize=12,
               color=PURPLE, arrowprops=dict(arrowstyle='-', color=PURPLE, lw=.9))
    a.set_xlabel('重采样后实际中位间距（mm）')
    a.set_ylabel('集成 Pa R²$_{cb}$')
    a.set_xlim(.25, 1.30)
    a.set_ylim(.58, .79)
    a.legend(loc='lower left', fontsize=11.5)
    a.set_title('从 STL 重采样：X5 在 1.2 mm 掉 0.15，X5D 只掉 0.05',
                loc='left', fontsize=14, pad=12)

    # b: voxel thinning deltas
    fr = [1.0, .5, .25, .1]
    for rows, color, mark, name in ((res_x5, PURPLE, 'o', 'X5'), (res_x5d, TEAL, 's', 'X5D')):
        for variant, ls in (('recompute', '-'), ('frozen', '--')):
            look = {r['fraction']: r for r in rows if r['variant'] == variant}
            look.setdefault(1.0, {r['fraction']: r for r in rows}[1.0])
            ys = [look[f]['r2'] - look[f]['ref'] for f in fr]
            b.plot(range(len(fr)), ys, mark, ls=ls, color=color, lw=1.6, ms=6,
                   label=f'{name}·{"重算特征" if variant == "recompute" else "冻结特征"}')
    b.axhline(0, color=LINE, lw=.9)
    b.set_xticks(range(len(fr)), ['100%', '50%', '25%', '10%'])
    b.set_xlabel('保留的 CFD 壁面节点比例')
    b.set_ylabel('Δ Pa R²$_{cb}$（对同点参照）')
    b.set_ylim(-.43, .04)
    b.legend(loc='lower left', fontsize=11)
    b.set_title('体素抽稀：重算几何那一层是大头', loc='left', fontsize=14, pad=12)

    # c: neighbourhood decomposition
    order = ['thin/thin', 'full/thin', 'thin/full']
    names = ['support+patch\n都抽稀', '只 patch 抽稀\n(support 全云)', '只 support 抽稀\n(patch 全云)']
    width = .26
    for j, f in enumerate((.5, .25, .1)):
        look = {r['variant']: r for r in dec if r['fraction'] == f}
        vals = [look[v]['r2'] - look[v]['ref'] for v in order]
        c.bar(np.arange(3) + (j - 1) * width, vals, width * .92,
              color=[LIGHT, GREY, TEAL][j], edgecolor=INK, lw=.6,
              label=f'{int(f * 100)}%')
        for k, v in enumerate(vals):
            c.text(k + (j - 1) * width, v - .008, f'{v:.3f}', ha='center', va='top',
                   fontsize=9.5, color=INK)
    c.axhline(0, color=LINE, lw=.9)
    c.set_xticks(range(3), names, fontsize=11)
    c.set_ylabel('Δ Pa R²$_{cb}$（X5D，对同点参照）')
    c.set_ylim(-.19, .025)
    c.legend(title='保留比例', fontsize=11, title_fontsize=11, loc='lower right')
    c.set_title('代价几乎全在 query patch，不在 support', loc='left', fontsize=14, pad=12)

    return save(fig, '02_density_curve',
                claim='部署点云重采样到 0.5 mm 时 X5 几乎不掉分；密度增广 X5D 把粗点云的损失压到三分之一；损失定位在 K16 query patch 这一层。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Panel a shows absolute ensemble R2 (different point sets per spacing, reference line is the mean same-point reference); panels b and c show same-point deltas only. Full-cloud rows are exactly 0 by construction.')


# ------------------------------------------------------- 03 几何保真敏感性
def geometry_fidelity():
    geom = D.geometry_rows(D.FOLLOW / 'geometry_x5d5' / 'geometry_sensitivity.json')
    b2 = D.geometry_rows(D.FOLLOW / 'geometry_x5d5_b2' / 'geometry_sensitivity.json')
    b3 = D.geometry_rows(D.FOLLOW / 'geometry_x5d5_b3' / 'geometry_sensitivity.json')
    tta, tta_raw = D.tta_rows(D.FOLLOW / 'tta_x5d5' / 'sampling_tta.json')
    payload = {'geometry': geom, 'noise_then_smooth': b2, 'rough': b3,
               'tta': tta,
               'single_draw_case_r2_sd_median': tta_raw['single_draw_case_r2_sd_median']}
    src = D.dump('03_geometry_fidelity', payload)

    look = {r['variant']: r for r in geom + b2 + b3}
    d = lambda k: look[k]['r2'] - look[k]['ref']

    fig = page('表面几何要多准：平滑几乎免费，噪声最伤，切口可容忍',
               'a 在 0.5 mm 重采样面上单独施加每种扰动｜b 真实场景：亚毫米粗糙 + 先平滑，以及重采样 TTA 值不值得',
               kicker='P21 · 几何保真敏感性 · §22.6-B/B2/B3/C',
               footer='X5D 五 seed 集成，34 例，Δ 对同点全云参照。扰动改变的是模型输入而非真实 WSS：这是输入误差敏感性，不是"几何变了真值也该变"的验证。')
    axs = fig.subplots(1, 2, gridspec_kw={'left': .075, 'right': .975, 'bottom': .19,
                                          'top': .66, 'wspace': .22})
    a, b = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    keys = ['clean', 'smooth_s1.0', 'smooth_s1.5', 'smooth_s2.0',
            'noise_0.1', 'noise_0.2', 'noise_0.4', 'cut_5', 'cut_10']
    labels = ['仅重采样\n（基准）', '平滑 1.0', '平滑 1.5', '平滑 2.0',
              '噪声 0.1', '噪声 0.2', '噪声 0.4', '切口 ±5', '切口 ±10']
    colors = [GREY] + [TEAL] * 3 + [RED] * 3 + [ORANGE] * 2
    vals = [d(k) for k in keys]
    a.bar(range(len(keys)), vals, .66, color=colors, edgecolor=INK, lw=.6)
    for i, v in enumerate(vals):
        a.text(i, v - .004, f'{v:.3f}', ha='center', va='top', fontsize=11, color=INK)
    a.axhline(d('clean'), color=GREY, ls='--', lw=.9)
    a.axhline(0, color=LINE, lw=.9)
    a.set_xticks(range(len(keys)), labels, fontsize=11)
    a.set_ylabel('Δ Pa R²$_{cb}$（对同点参照）')
    a.set_ylim(-.155, .035)
    a.set_title('平滑到 1.5 mm 仍与基准同级；噪声 0.4 mm 掉 0.12', loc='left',
                fontsize=14, pad=12)

    keys2 = ['noise_0.2c0.5', 'noise_0.2c0.5+smooth_s1.0',
             'noise_0.2', 'noise_0.2+smooth_s1.0', 'noise_0.4', 'noise_0.4+smooth_s1.0']
    labels2 = ['亚毫米粗糙\n0.2 mm', '粗糙\n+ 平滑 1.0',
               '白噪声\n0.2 mm', '噪声 0.2\n+ 平滑 1.0', '白噪声\n0.4 mm', '噪声 0.4\n+ 平滑 1.0']
    vals2 = [d(k) for k in keys2]
    pos = [0, 1.05, 3.0, 4.05, 6.0, 7.05]
    b.bar(pos, vals2, .86, color=[RED, TEAL] * 3, edgecolor=INK, lw=.6)
    for p, v in zip(pos, vals2):
        b.text(p, v - .004, f'{v:.3f}', ha='center', va='top', fontsize=11, color=INK)
    b.axhline(0, color=LINE, lw=.9)
    b.set_xticks(pos, labels2, fontsize=11)
    b.set_xlim(-.8, 7.85)
    b.set_ylabel('Δ Pa R²$_{cb}$（对同点参照）')
    b.set_ylim(-.155, .035)
    tta_d = {r['label']: r['r2'] for r in tta}
    b.text(.015, .985,
           f'重采样 TTA（5 次平均）{tta_d["tta_5"] - tta_d["reference_full_cloud"]:+.3f}  vs  '
           f'单次采样 {tta_d["single"] - tta_d["reference_full_cloud"]:+.3f}  →  多跑 5 次只换回 '
           f'{tta_d["tta_5"] - tta_d["single"]:+.3f}，不值得',
           transform=b.transAxes, fontsize=11.5, color=MUTED, va='top')
    b.set_title('先平滑能把粗糙的损失收回七成（−0.092 → −0.038）', loc='left',
                fontsize=14, pad=12)

    return save(fig, '03_geometry_fidelity',
                claim='部署端先做 1 mm 平滑几乎不花代价并能救回大部分表面粗糙损失；边界噪声是最伤的一项；采样 TTA 不值得。',
                sources=src, axes=list(axs), row_groups=[['a', 'b']],
                note='Perturbations are applied to the resampled deployment surface only; CFD labels are unchanged, so these quantify input-error sensitivity rather than a new ground truth.')


if __name__ == '__main__':
    decision_chain()
    density_curve()
    geometry_fidelity()
