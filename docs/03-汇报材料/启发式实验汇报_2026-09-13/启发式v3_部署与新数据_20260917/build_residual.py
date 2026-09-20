"""图 12-13：残差结构与尾部口径，ILO 最弱队列与"不训练就能拿到的一档"。"""
import numpy as np
from plot_style import (page, panel_label, save, INK, MUTED, BLUE, TEAL,
                        PURPLE, ORANGE, RED, GREY, LINE, LIGHT)
import datasrc as D

SCOPES = (('test34_base5', 'test34 五 seed 集成（34 例）', TEAL),
          ('cv3_oof', 'cv3 折外（136 例）', ORANGE))


# ------------------------------------------------- 12 残差结构与尾部口径
def residual_structure():
    cur = D.load(D.W1 / 'analysis_20260917' / 'current_residuals.json')
    payload = {k: {'residual_medians': cur[k]['summary']['residual_medians'],
                   'calibration_pooled': cur[k]['summary']['calibration_pooled'],
                   'high_wss_case_p90_then_pool_r2':
                       cur[k]['summary']['high_wss_case_p90_then_pool_r2'],
                   'pa_r2': cur[k]['summary']['pa']['r2']}
               for k, *_ in SCOPES}
    offline_high = D.load(D.W1 / 'offline' / 'analyze_wave1.json')[
        'X5D_v51 5-seed (Pa mean)']['high_r2']
    payload['offline_pooled_p90_high_r2'] = offline_high
    src = D.dump('12_residual_structure', payload)

    fig = page('残差主体在哪：沿程轨迹已经解决，截面内型态没有',
               'a 残差方差的三层分解（分支份额嵌套在区间均值份额里）｜b 两个尺度各自的解释度｜c 尾部的三种口径',
               kicker='P30 · 残差结构与尾部口径 · §28.9',
               footer='log_z 均值集成，"分支 × 4 mm 区间"分箱，取逐病例中位数。区间内部含轴向变化，不是纯周向误差，也不是 Pa 平方误差份额；'
                      '分支份额嵌套于区间均值份额，三项不可相加。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .065, 'right': .975, 'bottom': .175,
                                          'top': .66, 'wspace': .30})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    xs = np.arange(2)
    med = [cur[k]['summary']['residual_medians'] for k, *_ in SCOPES]
    within = [m['within_bin_share'] for m in med]
    binmean = [m['bin_mean_share'] for m in med]
    branch = [m['branch_mean_share'] for m in med]
    a.bar(xs, within, .46, color=PURPLE, edgecolor=INK, lw=.6, label='区间内型态（未解决）')
    a.bar(xs, binmean, .46, bottom=within, color=GREY, edgecolor=INK, lw=.6,
          label='区间均值（沿程轨迹）')
    for x, w, bm, br in zip(xs, within, binmean, branch):
        a.text(x, w / 2, f'{w:.1%}', ha='center', va='center', fontsize=16,
               color='white', weight='bold')
        a.text(x, w + bm / 2, f'{bm:.1%}', ha='center', va='center', fontsize=13, color=INK)
        a.plot([x - .23, x + .23], [w + br, w + br], color=RED, lw=1.6, ls='--')
    a.text(1.42, .78, '红虚线 = 其中的\n分支均值份额\n'
                      f'{branch[0]:.1%} / {branch[1]:.1%}',
           fontsize=11.5, color=RED, va='center')
    a.set_xticks(xs, ['test34\n五 seed', 'cv3 折外\n136 例'], fontsize=12.5)
    a.set_xlim(-.55, 1.95)
    a.set_ylim(0, 1.05)
    a.set_ylabel('log_z 残差方差份额（逐例中位）')
    a.legend(loc='upper right', fontsize=11.5)
    a.set_title('两个口径都是 ≈70% 在区间内部', loc='left', fontsize=14, pad=12)

    track = [m['bin_track_r2'] for m in med]
    pattern = [m['within_bin_pattern_r2'] for m in med]
    x = np.arange(2)
    b.bar(x - .19, track, .36, color=GREY, edgecolor=INK, lw=.6, label='区间均值轨迹 R²')
    b.bar(x + .19, pattern, .36, color=PURPLE, edgecolor=INK, lw=.6, label='去均值后的区间内型态 R²')
    for xi, v in zip(x - .19, track):
        b.text(xi, v + .012, f'{v:.4f}', ha='center', fontsize=12, color=INK)
    for xi, v in zip(x + .19, pattern):
        b.text(xi, v + .012, f'{v:.4f}', ha='center', fontsize=12, color=INK)
    b.set_xticks(x, ['test34 五 seed', 'cv3 折外'], fontsize=12.5)
    b.set_ylim(0, 1.18)
    b.set_ylabel('逐病例中位 R²（归一化空间）')
    b.legend(loc='upper center', fontsize=11.5)
    b.set_title('沿程轨迹 0.93+，截面内型态只有 0.52–0.61', loc='left', fontsize=14, pad=12)

    t34 = cur['test34_base5']['summary']
    oof = cur['cv3_oof']['summary']
    names = ['离线口径\n先 pool 再取 P90\n（test34）', '正式口径\n每例 P90 后 pool\n（test34）',
             '正式口径\ncv3 折外']
    vals = [offline_high, t34['high_wss_case_p90_then_pool_r2'],
            oof['high_wss_case_p90_then_pool_r2']]
    c.bar(range(3), vals, .5, color=[GREY, TEAL, ORANGE], edgecolor=INK, lw=.6)
    for i, v in enumerate(vals):
        c.text(i, v + .012, f'{v:.4f}', ha='center', fontsize=13.5, color=INK)
    c.set_xticks(range(3), names, fontsize=11.5)
    c.set_ylim(0, .78)
    c.set_ylabel('high-WSS R²')
    c.text(.03, .93,
           f'幅值仍系统性偏低：top10 幅值比 {t34["calibration_pooled"]["top10_pred_true_ratio"]:.3f}'
           f'（折外 {oof["calibration_pooled"]["top10_pred_true_ratio"]:.3f}）\n'
           f'p99 比 {t34["calibration_pooled"]["p99_pred_true_ratio"]:.3f}'
           f'（折外 {oof["calibration_pooled"]["p99_pred_true_ratio"]:.3f}），最大值只到真值的 '
           f'{t34["calibration_pooled"]["max_pred_true_ratio"]:.2f}',
           transform=c.transAxes, fontsize=11.5, color=INK, va='top')
    c.set_title('0.387 / 0.527 是两种阈值定义，不是精度变化', loc='left', fontsize=14, pad=12)

    return save(fig, '12_residual_structure',
                claim='三代底座、两套数据、test34 与折外口径一致：约 70% 的残差方差在分支×4mm 区间内部，沿程轨迹已解释到 0.93 以上，截面内型态只有 0.52–0.61。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Branch share is nested inside the bin-mean share (dashed line), so the three numbers must not be summed. Panel c compares two threshold definitions of the same predictions plus one out-of-fold scope.')


# --------------------------------------- 13 ILO 最弱队列 + 门控两模型混合
def ilo_and_gate():
    cur = D.load(D.W1 / 'analysis_20260917' / 'current_residuals.json')
    gate = D.load(D.LONG / 'offline' / 'gate_by_validity.json')
    oof = cur['cv3_oof']['per_case']
    ilo = sorted([p for p in oof if p['unit'].startswith('ILO')],
                 key=lambda p: -p['mse_pa'])
    tot_ilo = sum(p['mse_pa'] for p in ilo)
    tot_all = sum(p['mse_pa'] for p in oof)
    top2 = ilo[:2]
    payload = {'ilo_cases': [{'unit': p['unit'], 'mse_pa': p['mse_pa'],
                              'share_ilo': p['mse_pa'] / tot_ilo,
                              'r2_pa': p['r2_pa'], 'top10_ratio': p['top10_ratio'],
                              'top10_iou': p['top10_iou'],
                              'high_sse_fraction': p['high_sse_fraction']} for p in ilo],
               'top2_share_ilo': sum(p['mse_pa'] for p in top2) / tot_ilo,
               'top2_share_all136': sum(p['mse_pa'] for p in top2) / tot_all,
               'ilo_oof_r2': cur['cv3_oof']['per_cohort']['ILO']['pa']['r2'],
               'gate': gate}
    src = D.dump('13_ilo_and_gate', payload)

    fig = page('最弱的队列集中在两例；不重训也还有一档可以拿',
               'a ILO 折外 32 例的平方误差构成｜b 按沿程通道有效性做两模型门控混合',
               kicker='P31 · ILO 诊断与门控混合 · §28.9、§29.10',
               footer='a 为 cv3 折外逐例 Pa 平方误差（病例等权），只看 ILO 的 32 例；b 为 test34 三 seed Pa 均值集成，'
                      '门控只用部署侧可得的 mask，不用真值。')
    axs = fig.subplots(1, 2, gridspec_kw={'left': .075, 'right': .975, 'bottom': .175,
                                          'top': .66, 'wspace': .22})
    a, b = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    shares = [p['mse_pa'] / tot_ilo for p in ilo]
    colors = [RED, RED] + [GREY] * (len(ilo) - 2)
    a.bar(range(len(ilo)), shares, .72, color=colors, edgecolor=INK, lw=.5)
    for i, p in enumerate(top2):
        a.annotate(f'{p["unit"].split("/")[1]}\nR² {p["r2_pa"]:.3f}｜幅值比 {p["top10_ratio"]:.3f}\n'
                   f'IoU {p["top10_iou"]:.3f}｜高值区占自身误差 {p["high_sse_fraction"]:.0%}',
                   (i, shares[i]), xytext=(34 + i * 130, -18 - i * 52),
                   textcoords='offset points', fontsize=11.5, color=RED,
                   arrowprops=dict(arrowstyle='-', color=RED, lw=.9))
    a.set_xticks([0, 10, 20, 31], ['1', '11', '21', '32'])
    a.set_xlabel('ILO 32 例，按平方误差从大到小排序')
    a.set_ylabel('占 ILO 病例等权平方误差的份额')
    a.set_ylim(0, .42)
    a.text(.545, .985,
           f'前两例合占 ILO 的 {payload["top2_share_ilo"]:.2%}、\n'
           f'占全部 136 例折外的 {payload["top2_share_all136"]:.2%}\n'
           f'ILO 折外 R²$_{{cb}}$ = {payload["ilo_oof_r2"]:.4f}',
           transform=a.transAxes, fontsize=12.5, color=INK, va='top')
    a.set_title('不是整个 ILO 都差，是两例极端高幅值病例', loc='left', fontsize=14, pad=12)

    keys = ['base', 'long', 'mean', 'gate32', 'gate8']
    names = ['底座\n三 seed', '沿程臂\n三 seed', '两者\n等权平均',
             '门控：32 通道\n全有效处用沿程', '门控：ref 8\n全有效处用沿程']
    vals = [gate[k]['pa_r2_cb'] for k in keys]
    high = [gate[k]['high_wss_r2'] for k in keys]
    colors = [GREY, TEAL, BLUE, PURPLE, ORANGE]
    b.bar(range(5), vals, .5, color=colors, edgecolor=INK, lw=.6)
    for i, v in enumerate(vals):
        b.text(i, v + .0004, f'{v:.4f}', ha='center', va='bottom', fontsize=13, color=INK)
    b.set_xticks(range(5), [f'{n}\nhigh-WSS {h:.3f}' for n, h in zip(names, high)],
                 fontsize=11.5)
    b.set_ylim(.7695, .7895)
    b.set_ylabel('test34 三 seed 集成 Pa R²$_{cb}$')
    b.text(.02, .99,
           f'相对底座三 seed 共 {vals[-1] - vals[0]:+.4f}；但等权平均已经拿到 '
           f'{vals[2] - vals[0]:+.4f}，\n门控本身只再加 {vals[-1] - vals[2]:+.4f} —— 大头是两种输入配方的集成多样性。\n'
           'ref 8 门控好于 32 通道门控，说明 pc 缺失区并不是沿程臂失效的地方。',
           transform=b.transAxes, fontsize=11.5, color=INK, va='top')
    b.set_title('不重训就能到 0.7844（对照：底座 3→5 seed 只 +0.0013）', loc='left',
                fontsize=14, pad=12)

    return save(fig, '13_ilo_and_gate',
                claim='ILO 折外的误差有 52% 集中在两例高幅值病例；按沿程通道有效性门控两模型混合可在不重训的情况下把 test34 三 seed 集成从 0.7736 抬到 0.7844，但大头是集成多样性而非门控本身。',
                sources=src, axes=list(axs), row_groups=[['a', 'b']],
                note='Panel b mixes two models that differ only in inputs; the honest attribution line is printed inside the panel. Gate masks are deployment-available validity flags, never ground truth.')


if __name__ == '__main__':
    residual_structure()
    ilo_and_gate()
