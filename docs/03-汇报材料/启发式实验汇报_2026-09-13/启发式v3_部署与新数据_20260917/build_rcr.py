"""图 04-05：RCR 出口面积录入核查与 v5.0 → v5.1 的数据口径变化。"""
import collections
import numpy as np
from plot_style import (page, panel_label, save, INK, MUTED, BLUE, TEAL,
                        PURPLE, ORANGE, RED, GREY, LINE, LIGHT)
import datasrc as D

TYPES = (('小数点错一位', '小数点错一位', RED),
         ('与同侧另一出口面积互换', '左右 / 内外互换', ORANGE),
         ('面积不符', '面积不符（未归类）', PURPLE))


def kind(verdict):
    for key, label, color in TYPES:
        if verdict.startswith(key):
            return label, color
    return None, None


# ------------------------------------------------------------- 04 录入核查
def rcr_audit_figure():
    rows = D.rcr_audit()
    corrected = D.rcr_corrected()
    bad = [r for r in rows if r['verdict'] != 'ok']
    counts = collections.Counter(kind(r['verdict'])[0] for r in bad)
    bad_cases = sorted({r['case'] for r in bad})
    test_cases = sorted({r['case'] for r in bad if r['role'] == 'test'})
    rerun_cases = sorted({r['case'] for r in corrected})

    payload = {'n_outlets': len(rows), 'n_bad_outlets': len(bad),
               'n_bad_cases': len(bad_cases), 'by_type': dict(counts),
               'bad_cases': bad_cases, 'test34_cases': test_cases,
               'n_rerun_cases': len(rerun_cases), 'rerun_cases': rerun_cases,
               'ok_ratio_range': [min(float(r['area_ratio_R1']) for r in rows if r['verdict'] == 'ok'),
                                  max(float(r['area_ratio_R1']) for r in rows if r['verdict'] == 'ok')],
               'bad_ratio_range': [min(float(r['area_ratio_R1']) for r in bad),
                                   max(float(r['area_ratio_R1']) for r in bad)],
               'cohort_of_bad_outlets': dict(collections.Counter(r['cohort'] for r in bad))}
    src = D.dump('04_rcr_audit', payload)

    fig = page('CFD 边界条件的出口面积被录错了：23 例、39 个出口',
               'a 用 R1 反推每个出口的"表内隐含面积"，再除以网格实测面积｜b 三类录入错误与处理结果',
               kicker='P22 · RCR 出口面积录入核查 · §23、§24',
               footer='核查只读 UDF 与网格：R1 = 8μτ/(πr³) 反解 r，与网格出口面质心半径比较。协议本身没问题——CFD 的分流就是按出口盖面半径的 Murray；错的是把面积抄进表格这一步。')
    axs = fig.subplots(1, 2, gridspec_kw={'left': .065, 'right': .975, 'bottom': .175,
                                          'top': .66, 'wspace': .20})
    a, b = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    ratios = sorted(float(r['area_ratio_R1']) for r in rows)
    ok_x, ok_y, groups = [], [], collections.defaultdict(lambda: ([], []))
    order = {v: i for i, v in enumerate(ratios)}
    used = collections.Counter()
    for r in rows:
        v = float(r['area_ratio_R1'])
        idx = order[v] + used[v]
        used[v] += 1
        label, color = kind(r['verdict'])
        if label is None:
            ok_x.append(idx)
            ok_y.append(v)
        else:
            groups[label][0].append(idx)
            groups[label][1].append(v)
    a.scatter(ok_x, ok_y, s=9, color=GREY, alpha=.55, lw=0, label=f'录入正确（{len(ok_x)} 个出口）')
    for label, _lbl, color in TYPES:
        xs, ys = groups[_lbl]
        a.scatter(xs, ys, s=34, color=color, lw=.5, edgecolor='white',
                  label=f'{_lbl}（{len(xs)}）', zorder=3)
    a.axhline(1, color=INK, lw=.9, ls='--')
    a.axhspan(0.8, 1.25, color=LIGHT, zorder=0)
    a.set_yscale('log')
    a.set_yticks([.1, .2, .5, 1, 2, 5, 10, 40],
                 ['0.1×', '0.2×', '0.5×', '1×', '2×', '5×', '10×', '40×'])
    a.set_ylim(.07, 55)
    a.set_xlabel('全部 688 个出口，按比值排序')
    a.set_ylabel('表内隐含面积 ÷ 网格实测面积')
    a.legend(loc='upper left', fontsize=11.5)
    a.set_title('649 个出口落在 1× 附近，39 个明显不是同一个面', loc='left',
                fontsize=14, pad=12)

    names = [t[1] for t in TYPES]
    vals = [counts[n] for n in names]
    colors = [t[2] for t in TYPES]
    ypos = np.arange(len(names))[::-1]
    b.barh(ypos, vals, .52, color=colors, edgecolor=INK, lw=.6)
    for y, v in zip(ypos, vals):
        b.text(v + .5, y, f'{v} 个出口', va='center', fontsize=12.5, color=INK)
    b.set_yticks(ypos, names, fontsize=13)
    b.set_xlim(0, 24)
    b.set_xlabel('出口数')
    b.set_ylim(-3.1, 2.6)
    b.spines['left'].set_visible(False)
    b.tick_params(axis='y', length=0)
    lines = [
        f'受影响病例 {len(bad_cases)} 例：AAA 10｜AG 6｜ILO 7',
        f'其中 test34 有 {len(test_cases)} 例 → 这 4 例的 WSS 标签已经变了',
        f'另补改 3 例边缘偏差 → 共 {len(rerun_cases)} 例按修正后的 R1/R2/C 重跑 CFD',
        '重跑后 26/26 例分流复核通过；旧导出 52 个目录 248.2 GB 已删除',
        '同期扫出 2 例重复病例（LIU_WEN_QI、HOU_SHEN_QIAN）一并剔除',
    ]
    for i, s in enumerate(lines):
        b.text(-.5, -.55 - i * .48, '·  ' + s, fontsize=12.5, color=INK, va='center')
    b.set_title('处理结果：26 例重算 + 2 例去重 → v5.1 母库 170 例', loc='left',
                fontsize=14, pad=12)

    fig.text(.5, .085,
             '数据口径：172 例 → 170 例（train136 / test34）；test34 有 4 例标签修正，train 剔除 2 例重复。'
             '所有 v5.1 之后的指标都在这个口径上，不能与 v5.0 的数字直接并排。',
             ha='center', fontsize=13, color=INK, weight='bold')
    return save(fig, '04_rcr_audit',
                claim='用 R1 反推出口面积发现 23 例（39 个出口）录入错误，修正后 26 例重算 CFD，并剔除 2 例重复病例，数据口径变为 170 例。',
                sources=src, axes=list(axs), row_groups=[['a', 'b']],
                note='Panel a is one point per outlet (688 outlets over 172 cases); the shaded band is the 0.8-1.25x agreement zone used only for reading, not a pass/fail threshold. Panel b counts outlets, not cases.')


# --------------------------------------------------- 05 v5.0 → v5.1 口径变化
def data_version_figure():
    ls = D.load(D.FOLLOW / 'label_only_reeval_x5d5' / 'label_only_summary.json')
    reeval = D.load(D.FOLLOW / 'label_only_reeval_x5d5' / 'label_only_reeval.json')
    wave1 = D.load(D.W1 / 'offline' / 'analyze_wave1.json')

    old = ls['per_seed_old_weights']           # seed -> [old labels, new labels]
    new = ls['per_seed_new_weights']           # retrained, new labels
    steps = [('旧权重\n旧标签', ls['old5_old_labels'], GREY),
             ('旧权重\n新标签', ls['old5_new_labels'], ORANGE),
             ('重训权重\n新标签', wave1['X5D_v51 5-seed (Pa mean)']['pa_r2_cb'], TEAL)]
    payload = {'five_seed': {'old_weights_old_labels': ls['old5_old_labels'],
                             'old_weights_new_labels': ls['old5_new_labels'],
                             'retrained_new_labels': wave1['X5D_v51 5-seed (Pa mean)']['pa_r2_cb']},
               'three_seed': {'old_weights_new_labels': ls['old3_new_labels'],
                              'retrained_new_labels': ls['new3_new_labels']},
               'per_seed_old_weights': old, 'per_seed_retrained': new,
               'relabelled_cases': reeval['relabeled']}
    src = D.dump('05_data_v50_to_v51', payload)

    fig = page('v5.0 → v5.1：涨的那一截是标签修正，不是方法变好',
               'a 把同一批权重放到新标签上，再换成重训权重，三档拆开｜b 逐 seed：同一权重、只换标签',
               kicker='P23 · 数据口径 · §25、§28.1',
               footer='test34 五 seed Pa 均值集成，best checkpoint。a 的第三档是重训模型，前两档是同一组旧权重；'
                      '三档之间只有一个变量在动，不能把整段差读成方法收益。')
    axs = fig.subplots(1, 2, gridspec_kw={'left': .075, 'right': .975, 'bottom': .175,
                                          'top': .66, 'wspace': .22})
    a, b = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    xs = np.arange(3)
    vals = [s[1] for s in steps]
    a.bar(xs, vals, .5, color=[s[2] for s in steps], edgecolor=INK, lw=.7)
    for x, v in zip(xs, vals):
        a.text(x, v + .0012, f'{v:.4f}', ha='center', va='bottom', fontsize=14, color=INK)
    for i in range(2):
        d = vals[i + 1] - vals[i]
        a.annotate('', (xs[i + 1] - .25, vals[i]), xytext=(xs[i] + .25, vals[i]),
                   arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
        a.text((xs[i] + xs[i + 1]) / 2, vals[i] - .0035,
               ('只换标签\n' if i == 0 else '重新训练\n') + f'{d:+.4f}',
               ha='center', va='top', fontsize=12.5,
               color=ORANGE if i == 0 else TEAL)
    a.set_xticks(xs, [s[0] for s in steps], fontsize=13)
    a.set_ylim(.735, .784)
    a.set_ylabel('test34 五 seed 集成 Pa R²$_{cb}$')
    a.set_title('总共 +0.0284，其中 2/3 来自标签本身', loc='left', fontsize=14, pad=12)

    seeds = sorted(old, key=lambda s: old[s][1])
    label_y, gap = [], .0026
    for s in seeds:
        y = old[s][1]
        if label_y and y - label_y[-1] < gap:
            y = label_y[-1] + gap
        label_y.append(y)
    for s, ly in zip(seeds, label_y):
        o, n = old[s]
        b.plot([0, 1], [o, n], '-', color=GREY, lw=1.2, zorder=1)
        b.scatter([0], [o], s=54, color=GREY, zorder=3)
        b.scatter([1], [n], s=54, color=ORANGE, zorder=3)
        b.text(1.05, ly, f'{s.replace("X5D_s", "seed ")}  {n - o:+.3f}',
               fontsize=11.5, color=MUTED, va='center')
    b.set_xticks([0, 1], ['旧权重 · 旧标签', '旧权重 · 新标签'], fontsize=13)
    b.set_xlim(-.28, 1.62)
    b.set_ylabel('test34 单 seed Pa R²$_{cb}$')
    b.set_ylim(.695, .762)
    b.set_title('5/5 个 seed 在只换标签后都升，量级 0.009–0.024', loc='left',
                fontsize=14, pad=12)
    names = [c.split('/')[1] if c.startswith('ILO/') else c.split('/')[-1]
             for c in reeval['relabeled']]
    b.text(.015, .225, '标签变化的 4 例：' + '、'.join(names[:2]) + '、\n' +
           '、'.join(names[2:]),
           transform=b.transAxes, fontsize=11, color=MUTED, va='bottom')
    b.text(.015, .04,
           '重训后的三个 seed（新权重 · 新标签）= ' +
           ' / '.join(f'{v:.4f}' for v in new.values()) + '，见 P24',
           transform=b.transAxes, fontsize=11.5, color=TEAL)

    return save(fig, '05_data_v50_to_v51',
                claim='把同一批 X5D 权重放到修正后的标签上就涨 0.019，重训再涨 0.010；v5.1 相对 v5.0 的提升主要是标签修正而非方法改进。',
                sources=src, axes=list(axs), row_groups=[['a', 'b']],
                note='Old weights are the v5.0 X5D runs evaluated on the v5.1 labels of the same 34 test cases; the retrained column is a different model, so the second arrow mixes retraining with the refreshed train split (136 vs 138 cases).')


if __name__ == '__main__':
    rcr_audit_figure()
    data_version_figure()
