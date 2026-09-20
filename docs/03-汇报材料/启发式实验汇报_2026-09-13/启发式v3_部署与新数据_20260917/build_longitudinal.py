"""图 10-11：沿程几何 32 通道接入 X5D 底座，以及"为什么没有明显提升"。"""
import re
import numpy as np
from plot_style import (page, panel_label, save, INK, MUTED, BLUE, TEAL,
                        PURPLE, ORANGE, RED, GREY, LINE, LIGHT)
import datasrc as D

LO = D.LONG / 'offline'
CN = {'log_area': '截面积 (log)', 'area_slope': '面积坡度', 'roundness': '圆度',
      'eccentricity': '偏心率', 'upstream_min_area_ratio': '上游最窄面积比',
      'downstream_min_area_ratio': '下游最窄面积比',
      'upstream_min_distance': '上游最窄距离', 'downstream_min_distance': '下游最窄距离'}


def parse_mae_iou(path):
    """Pull the per-seed MAE / IoU pairs out of the plain-text analysis dump."""
    body = D.text(path)
    seeds = {}
    for m in re.finditer(r's(\d+): MAE ([\d.]+) -> ([\d.]+).*?IoU ([\d.]+) -> ([\d.]+)', body):
        seeds[int(m.group(1))] = {'mae_base': float(m.group(2)), 'mae_long': float(m.group(3)),
                                  'iou_base': float(m.group(4)), 'iou_long': float(m.group(5))}
    ens = {}
    for m in re.finditer(r'(X5D_\S+)\s+(\d)-seed\s+MAE ([\d.]+) Pa\s+top10 IoU ([\d.]+)', body):
        ens[f'{m.group(1)} {m.group(2)}-seed'] = {'mae': float(m.group(3)),
                                                  'iou': float(m.group(4))}
    return seeds, ens


def parse_buckets(path):
    body = D.text(path)
    rows = []
    for line in body.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[-1].endswith('%'):
            continue
        try:
            share, base, long = (float(p) for p in parts[-4:-1])
            gain = float(parts[-1].rstrip('%'))
        except ValueError:
            continue
        rows.append({'group': ' '.join(parts[:-4]), 'share': share,
                     'base': base, 'long': long, 'gain_pct': gain})
    return rows


# --------------------------------------------------- 10 沿程通道接入的效果
def longitudinal_paired():
    a_json = D.load(LO / 'analyze_longitudinal.json')
    seeds_mae, ens_mae = parse_mae_iou(LO / 'analyze_mae_iou.txt')
    rows = a_json['paired_best']['rows']
    ens = a_json['ensemble_best']

    payload = {'paired_best': a_json['paired_best'], 'ensemble_best': ens,
               'mae_iou_per_seed': seeds_mae, 'mae_iou_ensemble': ens_mae,
               'coverage': {k: v for k, v in a_json['coverage_best'].items()
                            if not isinstance(v, dict)}}
    src = D.dump('10_longitudinal_paired', payload)

    fig = page('沿程几何 32 通道：尾部一致改善，整体 R² 说不清',
               'a 三 seed 同 seed 配对的整体 R²｜b 尾部与形态指标｜c 三 seed 集成对照（同 seed 数才可比）',
               kicker='P28 · 沿程 32 通道 · §29.1–§29.5',
               footer='X5D_v51 上追加 32 个沿程 value/mask 通道（27 → 59 维），配对零初始化，其余配方逐位不变。'
                      'seed 噪声带 物理 ±0.0164 / 归一化 ±0.0040（三 seed sd 口径）；n=3 不做显著性。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .065, 'right': .975, 'bottom': .185,
                                          'top': .66, 'wspace': .30})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    labels = [f'seed {r["seed"]}' for r in rows]
    x = np.arange(len(rows))
    for off, key, color, name, scale in ((-.19, 'd_pa', TEAL, '物理 R²$_{cb}$', 1),
                                         (.19, 'd_norm', BLUE, '归一化 R²$_{cb}$（×4）', 4)):
        vals = [r[key] * scale for r in rows]
        a.bar(x + off, vals, .34, color=color, edgecolor=INK, lw=.6, label=name)
        if key == 'd_pa':
            for xi, v in zip(x + off, vals):
                a.text(xi, v + (.0012 if v > 0 else -.0012), f'{v:+.4f}', ha='center',
                       va='bottom' if v > 0 else 'top', fontsize=11.5, color=color)
    a.axhline(0, color=INK, lw=.9, ls='--')
    a.axhspan(-.0164, .0164, color=LIGHT, zorder=0)
    labels = [f'{lab}\n归一化 {r["d_norm"]:+.4f}' for lab, r in zip(labels, rows)]
    a.set_xticks(x, labels, fontsize=12)
    a.set_ylim(-.032, .040)
    a.set_ylabel('Δ（沿程臂 − 底座）')
    a.legend(loc='upper right', fontsize=11.5)
    a.set_title(f'均值 {a_json["paired_best"]["mean_d_pa"]:+.4f}，sd '
                f'{a_json["paired_best"]["sd_d_pa"]:.4f}，只有 1/3 为正',
                loc='left', fontsize=14, pad=12)

    metrics = [('high-WSS\nR²', [r['d_high'] for r in rows], PURPLE, 1),
               ('top10\n幅值比', [r['d_top10'] for r in rows], ORANGE, 1),
               ('top10\nIoU', [seeds_mae[r['seed']]['iou_long'] -
                               seeds_mae[r['seed']]['iou_base'] for r in rows], TEAL, 1),
               ('MAE (Pa)\n越低越好', [seeds_mae[r['seed']]['mae_long'] -
                                  seeds_mae[r['seed']]['mae_base'] for r in rows], RED, 1)]
    for i, (name, vals, color, _s) in enumerate(metrics):
        b.scatter([i] * 3, vals, s=58, color=color, zorder=3)
        m = float(np.mean(vals))
        b.plot([i - .24, i + .24], [m, m], color=color, lw=2.4, zorder=4)
        pos = sum(v > 0 for v in vals)
        b.text(i, .098, f'{m:+.4f}\n{pos}/3 正', ha='center', fontsize=11.5, color=color)
    b.axhline(0, color=INK, lw=.9, ls='--')
    b.set_xticks(range(4), [m[0] for m in metrics], fontsize=12)
    b.set_xlim(-.6, 3.6)
    b.set_ylim(-.075, .135)
    b.set_ylabel('配对 Δ（三 seed）')
    b.set_title('IoU 3/3 升，high-WSS 与幅值比 2/3 升', loc='left', fontsize=14, pad=12)

    names = ['底座 3-seed', '底座 5-seed\n（现部署）', '沿程 3-seed']
    vals = [ens['base3']['pa_r2_cb'], ens['base5']['pa_r2_cb'], ens['long3']['pa_r2_cb']]
    colors = [GREY, GREY, TEAL]
    c.bar(range(3), vals, .5, color=colors, edgecolor=INK, lw=.6)
    for i, v in enumerate(vals):
        c.text(i, v + .0006, f'{v:.4f}', ha='center', va='bottom', fontsize=13, color=INK)
    c.annotate('', (2, .7815), xytext=(0, .7815),
               arrowprops=dict(arrowstyle='<|-|>', color=TEAL, lw=1.2))
    c.text(1, .7822, f'同 seed 数对比 {ens["d_pa_vs_base3"]:+.4f}', ha='center',
           fontsize=12.5, color=TEAL)
    c.set_xticks(range(3), names, fontsize=12.5)
    c.set_ylim(.765, .786)
    c.set_ylabel('集成 Pa R²$_{cb}$')
    c.set_title('3-seed 对 5-seed 口径不齐，不能当底座提升', loc='left', fontsize=14, pad=12)

    return save(fig, '10_longitudinal_paired',
                claim='沿程 32 通道在尾部与热点定位上一致改善（IoU 3/3 升），但整体 R²_cb 的三 seed 均值落在 seed 噪声带内，结论为候选而非底座替换。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Panel a scales the normalised delta by 4 so both bars share one axis; printed numbers are the unscaled values. Panel c compares a 3-seed ensemble with a 5-seed one, which is deliberately flagged as an unequal comparison.')


# ------------------------------------------------- 11 为什么没有明显提升
def longitudinal_why():
    red = D.load(LO / 'diagnose_redundancy.json')
    rank = D.load(LO / 'diagnose_effective_rank.json')
    resid = D.load(LO / 'diagnose_residual_explainability.json')
    buckets = parse_buckets(LO / 'diagnose_where_it_helps.txt')
    payload = {'ref_pc_corr': red['ref_pc_corr'], 'channels': red['channels'],
               'target_corr': red['target_corr'], 'effective_rank': rank,
               'residual_explainability': resid, 'buckets': buckets}
    src = D.dump('11_longitudinal_why', payload)

    fig = page('为什么沿程通道没有把整体 R² 抬起来',
               'a 一半通道互为重复｜b 最强的通道是底座已有信息的重述｜c 收益真实存在，但被缺失点摊薄',
               kicker='P29 · 只读诊断 · §29.7',
               footer='a 为 ref / pc 同名通道的逐点相关；b 横轴是"用原 27 维输入预测该通道"的逐例去均值 R²（越右越旧），'
                      '纵轴是该通道与 log_z(WSS) 的相关绝对值；c 为三 seed 集成在 ln 空间的逐点 MSE。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .085, 'right': .975, 'bottom': .185,
                                          'top': .66, 'wspace': .38})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    items = sorted(red['ref_pc_corr'].items(), key=lambda kv: kv[1])
    ys = np.arange(len(items))
    a.barh(ys, [v for _, v in items], .58,
           color=[TEAL if v < .9 else GREY for _, v in items], edgecolor=INK, lw=.6)
    for y, (k, v) in zip(ys, items):
        a.text(v - .004, y, f'{v:.3f}', va='center', ha='right', fontsize=11.5,
               color='white' if v > .9 else INK)
    a.set_yticks(ys, [CN[k] for k, _ in items], fontsize=12)
    a.set_xlim(.6, 1.0)
    a.set_xlabel('参考面通道 与 点云面通道 的逐点相关')
    a.set_title(f'除圆度外全部 ≥0.93；32 列的有效维数只有 '
                f'{rank["effective_dim"]:.1f}', loc='left', fontsize=13.5, pad=12)

    xs, ys2, cols, tags = [], [], [], []
    for key, corr in red['target_corr'].items():
        ch = red['channels'].get(key)
        if ch is None:
            continue
        xs.append(ch['within_case_r2'])
        ys2.append(abs(corr))
        is_ref = key.startswith('geom_ref_')
        cols.append(BLUE if is_ref else GREY)
        tags.append(key)
    b.scatter(xs, ys2, s=70, color=cols, lw=.5, edgecolor='white', zorder=3)
    for key, x0, y0 in zip(tags, xs, ys2):
        short = key.replace('geom_ref_', '').replace('geom_pc_', '')
        if short in ('log_area',) and key.startswith('geom_ref_'):
            b.annotate(f'{CN[short]}\n可预测度 {x0:.3f}｜与目标相关 {y0:.3f}', (x0, y0),
                       xytext=(-16, -34), textcoords='offset points', ha='right',
                       fontsize=11.5, color=BLUE,
                       arrowprops=dict(arrowstyle='-', color=BLUE, lw=.8))
        elif short == 'roundness' and key.startswith('geom_ref_'):
            b.annotate(f'{CN[short]}（真正"新"的那几个）', (x0, y0),
                       xytext=(22, 16), textcoords='offset points', fontsize=11.5,
                       color=TEAL, arrowprops=dict(arrowstyle='-', color=TEAL, lw=.8))
    b.axvspan(.8, 1.02, color=LIGHT, zorder=0)
    b.text(.90, .86, '右侧阴影 = 底座已有的信息', ha='center', fontsize=11.5, color=MUTED)
    b.set_xlim(-.03, 1.02)
    b.set_ylim(0, .92)
    b.set_xlabel('该通道能被原 27 维预测的程度（逐例 R²）')
    b.set_ylabel('|该通道与 log_z(WSS) 的相关|')
    b.set_title('越"新"的通道与目标越无关', loc='left', fontsize=13.5, pad=12)
    base = resid['X5D_v51 底座']
    arm = resid['X5D_long 沿程臂']
    b.text(.02, .035,
           f'16 个 value 对底座残差的线性解释力只有 R² {base["ols_r2_16values_pooled"]:.4f}；\n'
           f'沿程臂训练后降到 {arm["ols_r2_16values_pooled"]:.4f}，残差 sd '
           f'{base["resid_sd"]:.4f} → {arm["resid_sd"]:.4f}',
           transform=b.transAxes, fontsize=11.5, color=INK)

    order = ['全部', '沿程通道全有效', '有缺失通道', 'WSS 下 50%', '50–90%', '90–99%', 'top 1%']
    look = {r['group']: r for r in buckets}
    vals = [look[k]['gain_pct'] for k in order]
    shares = [look[k]['share'] for k in order]
    colors = [GREY] + [TEAL if v > 0 else RED for v in vals[1:]]
    ys3 = np.arange(len(order))[::-1]
    c.barh(ys3, vals, .58, color=colors, edgecolor=INK, lw=.6)
    for y, v, s in zip(ys3, vals, shares):
        c.text(v + (.22 if v > 0 else -.22), y, f'{v:+.2f}%',
               va='center', ha='left' if v > 0 else 'right', fontsize=11.5, color=INK)
    c.axvline(0, color=INK, lw=.9)
    c.set_yticks(ys3, [f'{k}\n占 {s:.0%} 点' for k, s in zip(order, shares)], fontsize=11.5)
    c.set_xlim(-4.2, 12.5)
    c.set_xlabel('沿程臂相对底座的 ln-MSE 相对改善')
    c.set_title('好处集中在 9% 的高值点，缺失区反而变差', loc='left', fontsize=13.5, pad=12)

    return save(fig, '11_longitudinal_why',
                claim='沿程通道一半互为重复、最强的一个 98% 是底座已有信息，对底座残差的线性解释力只有 R² 0.006，收益集中在 9% 的高值点而在 35% 的缺失点为负。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Panel b mixes ref and pc channels on one scatter; the two families overlap because they are near-duplicates (panel a). Bucket shares in c overlap by construction (validity split and WSS quantile split are two different partitions).')


if __name__ == '__main__':
    longitudinal_paired()
    longitudinal_why()
