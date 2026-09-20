"""图 17：沿程通道剪枝阶梯 —— 32 → 2 列，保留 roundness 一路。"""
import numpy as np
from plot_style import (page, panel_label, save, INK, MUTED, BLUE, TEAL,
                        PURPLE, ORANGE, RED, GREY, LINE, LIGHT)
import datasrc as D

PRUNE = D.EXP / 'wss_x5d_long_prune_20260917'
LO = D.LONG / 'offline'
# 阶梯顺序：从最省的一档排到最全的一档
LADDER = [('L1_s1234', 'L1', '只 roundness', 2, TEAL),
          ('L2_s1234', 'L2', '+ 上游最窄距离', 4, GREY),
          ('L5_s1234', 'L5', 'ref 5', 10, GREY),
          ('L8_s1234', 'L8', 'ref 8', 16, BLUE),
          ('X5D_long 全32列', '全 32 列', 'ref 8 + pc 8', 32, PURPLE)]
BUCKET_CN = {'A 全 32 通道有效': 'A 全有效',
             'B ref8 有效但 pc 缺': 'B 缺 pc',
             'C 只有 roundness 有效': 'C 仅圆度',
             'D 连 roundness 也缺': 'D 圆度也缺',
             '全部': '全部'}


def parse_common_buckets(path):
    """offline/common_buckets.txt → {bucket: {'share':…, arm: pct, …}}."""
    lines = D.text(path).splitlines()
    header = lines[0].split()
    arms = header[2:]                       # 全32列 L8 L5 L2 L1
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2 + len(arms):
            continue
        pcts = parts[-len(arms):]
        share = parts[-len(arms) - 1]
        try:
            row = {'bucket': ' '.join(parts[:-len(arms) - 1]), 'share': float(share)}
            for arm, p in zip(arms, pcts):
                row[arm] = float(p.rstrip('%'))
        except ValueError:
            continue
        rows.append(row)
    return arms, rows


def prune_ladder():
    ladder = D.load(PRUNE / 'offline' / 'analyze_ladder.json')
    arms, buckets = parse_common_buckets(PRUNE / 'offline' / 'common_buckets.txt')
    sel = D.load(LO / 'select_channels_oof.json')
    red = D.load(LO / 'diagnose_redundancy.json')

    ceiling = sel['subsets']['全部 16 value']
    round_marginal = sel['ref_only_forward'][0]['cum_r2']
    cov_all32 = ladder['X5D_long 全32列']['coverage']
    cov_l1 = ladder['L1_s1234']['coverage']
    pred_round = red['channels']['geom_ref_roundness']['within_case_r2']
    pred_area = red['channels']['geom_ref_log_area']['within_case_r2']
    corr_round = red['ref_pc_corr']['roundness']
    corr_rest = float(np.median([v for k, v in red['ref_pc_corr'].items()
                                 if k != 'roundness']))

    payload = {'ladder': {k: {kk: vv for kk, vv in v.items() if kk != 'channels'}
                          for k, v in ladder.items()},
               'channels_per_arm': {k: v['channels'] for k, v in ladder.items()},
               'common_buckets': buckets,
               'criteria': {'roundness_marginal_r2': round_marginal,
                            'ceiling_all16': ceiling,
                            'coverage_L1': cov_l1, 'coverage_all32': cov_all32,
                            'predictability_roundness': pred_round,
                            'predictability_log_area': pred_area,
                            'ref_pc_corr_roundness': corr_round,
                            'ref_pc_corr_median_rest': corr_rest}}
    src = D.dump('17_prune_ladder', payload)

    fig = page('通道剪枝：32 列砍到 2 列，读数没变差，缺失区反而变好',
               'a 阶梯：归一化增益与列数无关｜b 同一组分桶的逐点改善｜c 为什么留下来的是圆度',
               kicker='P36 · 沿程通道剪枝 · §29.9、§29.12（作业 14991）',
               footer='四臂都是已审计 ref-8 集合的嵌套子集，单 seed 1234，27 → 43/37/31/29 配对零初始化（239 张量逐位不变），'
                      '协议 / split / 400 epoch / 选 ckpt 全同。通道顺序由「折外残差的边际增量」选出，选择过程不碰 test34。'
                      '单 seed 噪声带：物理 ±0.014 / 归一化 ±0.004 —— 各档之间的物理 R² 差别不可分辨。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .075, 'right': .975, 'bottom': .215,
                                          'top': .655, 'wspace': .42})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    xs = np.arange(len(LADDER))
    dnorm = [ladder[row[0]]['d_norm'] for row in LADDER]
    cov = [ladder[row[0]]['coverage'] for row in LADDER]
    a.bar(xs, dnorm, .52, color=[row[-1] for row in LADDER], edgecolor=INK, lw=.6)
    for x, v, cv in zip(xs, dnorm, cov):
        a.text(x, v + .00018, f'{v:+.4f}\n{cv:.3f}', ha='center', va='bottom',
               fontsize=11.5, color=INK)
    a.axhspan(-.004, .004, color=LIGHT, zorder=0)
    a.axhline(0, color=INK, lw=.9, ls='--')
    a.set_xticks(xs, [f'{lab}\n{n} 列' for _, lab, _sub, n, _ in LADDER], fontsize=11.5)
    a.set_ylim(0, .0098)
    a.set_ylabel('Δ 归一化 R²$_{cb}$（相对 X5D_v51 底座）')
    a.text(.03, .985, '阴影 = 归一化 seed 噪声带 ±0.004\n柱上第二行 = 逐点覆盖率',
           transform=a.transAxes, fontsize=11.5, color=MUTED, va='top')
    a.set_title('2 列 +0.0060 ≈ 32 列 +0.0065', loc='left', fontsize=13.5, pad=12)

    order = ['A 全 32 通道有效', 'B ref8 有效但 pc 缺', 'C 只有 roundness 有效',
             'D 连 roundness 也缺', '全部']
    look = {r['bucket']: r for r in buckets}
    width = .17
    for j, (key, lab, sub, n, col) in enumerate(LADDER):
        arm = 'L1' if key == 'L1_s1234' else ('全32列' if key.startswith('X5D_long')
                                              else key.split('_')[0])
        vals = [look[o][arm] for o in order]
        offs = (j - 2) * width
        bars = b.bar(np.arange(len(order)) + offs, vals, width * .9,
                     color=col if arm == 'L1' else col,
                     edgecolor=INK, lw=.5, alpha=1.0 if arm == 'L1' else .55,
                     label=f'{lab} {sub}')
    b.axhline(0, color=INK, lw=.9)
    b.set_xticks(range(len(order)),
                 [f'{BUCKET_CN[o]}\n占 {look[o]["share"]:.0%}' for o in order],
                 fontsize=10.5)
    b.set_ylim(-7.5, 10.5)
    b.set_ylabel('逐点 ln-MSE 相对改善（%）')
    b.legend(fontsize=10, ncol=2, loc='upper right')
    b.set_title('L1 是唯一在缺失区（C/D）不变差的臂', loc='left', fontsize=13.5, pad=12)

    crit = [('折外边际增量\n占 16 通道天花板', round_marginal / ceiling, 1.0,
             '只 roundness', '全部 16 value', True),
            ('逐点覆盖率\n（同时有效）', cov_l1, cov_all32, 'L1（2 列）', '全 32 列', True),
            ('被原 27 维预测的程度\n（越低越"新"）', pred_round, pred_area,
             'roundness', 'log_area', False),
            ('ref / pc 重复度\n（越低越不重复）', corr_round, corr_rest,
             'roundness', '其余 7 项中位', False)]
    ys = np.arange(len(crit))[::-1]
    for y, (lab, v, ref, nv, nr, higher_better) in zip(ys, crit):
        c.barh(y + .17, v, .3, color=TEAL, edgecolor=INK, lw=.5)
        c.barh(y - .17, ref, .3, color=GREY, edgecolor=INK, lw=.5)
        c.text(v + .015, y + .17, f'{v:.3f}  {nv}', va='center', fontsize=10.5,
               color=TEAL)
        c.text(ref + .015, y - .17, f'{ref:.3f}  {nr}', va='center', fontsize=10.5,
               color=MUTED)
    c.set_yticks(ys, [x[0] for x in crit], fontsize=10.5)
    c.set_xlim(0, 1.62)
    c.set_xticks([0, .25, .5, .75, 1.0])
    c.set_xlabel('归一化到 0–1 的判据值')
    c.set_title('圆度：最新、最不重复、覆盖率最高', loc='left', fontsize=13.5, pad=12)

    return save(fig, '17_prune_ladder',
                claim='把沿程 32 列剪到只留 ref 圆度 2 列，归一化增益不变（+0.0060 对 +0.0065），覆盖率从 0.651 升到 0.788，并且是唯一在几何缺失区不变差的配方。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Single seed 1234 throughout; differences in physical R2 between ladder rungs are inside the run-to-run band and are not read as information differences. Bucket shares are fixed across arms so the columns are comparable.')


if __name__ == '__main__':
    prune_ladder()
