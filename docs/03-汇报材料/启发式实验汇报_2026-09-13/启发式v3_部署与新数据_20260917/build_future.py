"""图 14-16：后续方向 —— 为什么要转向局部形态，以及候选怎么排。"""
import re
import numpy as np
from plot_style import (page, canvas, node, arrow, text, panel_label, save,
                        INK, MUTED, BLUE, TEAL, PURPLE, ORANGE, RED, GREY, LINE, LIGHT)
import datasrc as D

# 臂的中文短名（仅标签，不是数据）
ARM_CN = {
    'X1': 'P4 权重 EMA', 'X2': 'F1 弯曲参考角', 'X3': 'F2 分叉参考角',
    'X4': 'F3 上游历史', 'X5': 'F6 Murray 分支流量先验', 'X6': 'P5 人群先验',
    'X7': 'S1a patch 切平面坐标架', 'X8': 'S1b patch 注意力池化',
    'X9': 'S1c 坐标架 + 注意力 + K32', 'X10': 'T1 方向辅助头',
    'X11': 'S3 截面 token 上下文', 'X12': 'S5 MoE 头',
    'X13a': 'P1 全帧预训练', 'X13b': 'P1 预训练 + 微调',
    'X15': 'F1+F2+F3+F6 合并', 'X16': 'S1c + F-all + T1',
}
STRUCTURAL = {'X7', 'X8', 'X9', 'X11', 'X12', 'X16'}


def parse_x5_residual(path):
    """§20.6 的纯文本残差分析：取尺度分解与轨迹 / 型态 R²。"""
    body = D.text(path)
    out = {}
    for key, pat in (('within_section_share', r'within[- ]section[^0-9]*([0-9.]+)'),
                     ('section_track_r2', r'section-mean track R2[^0-9]*([0-9.]+)'),
                     ('within_section_pattern_r2', r'within-section pattern R2[^0-9]*([0-9.]+)')):
        m = re.search(pat, body, re.I)
        if m:
            out[key] = float(m.group(1))
    return out, body


# ------------------------------------------- 14 残差尺度阶梯（四个口径）
def residual_scale_ladder():
    cached = D.load(D.W6 / 'analysis_20260915' / 'cached_residual_analysis.json')
    cur = D.load(D.W1 / 'analysis_20260917' / 'current_residuals.json')

    scopes = [
        ('X5\nv5.0 · test34', cached['models']['X5']['residual_medians'], 'section', GREY),
        ('X5D\nv5.0 · test34', cached['models']['X5D']['residual_medians'], 'section', GREY),
        ('X5D_v51\nv5.1 · test34', cur['test34_base5']['summary']['residual_medians'],
         'bin', TEAL),
        ('X5D_v51\nv5.1 · cv3 折外', cur['cv3_oof']['summary']['residual_medians'],
         'bin', ORANGE),
    ]

    def pick(med, kind, what):
        key = {('section', 'within'): 'within_section_share',
               ('section', 'track'): 'section_track_r2',
               ('section', 'pattern'): 'within_section_pattern_r2',
               ('bin', 'within'): 'within_bin_share',
               ('bin', 'track'): 'bin_track_r2',
               ('bin', 'pattern'): 'within_bin_pattern_r2'}[(kind, what)]
        return med[key]

    payload = {'scopes': [{'name': n.replace('\n', ' '),
                           'within_share': pick(m, k, 'within'),
                           'track_r2': pick(m, k, 'track'),
                           'pattern_r2': pick(m, k, 'pattern')} for n, m, k, _ in scopes],
               'decile_residuals': {m: cached['models'][m]['residual_decile_case_medians']
                                    for m in ('X5', 'X5D')},
               'calibration_test34': cur['test34_base5']['summary']['calibration_pooled'],
               'calibration_oof': cur['cv3_oof']['summary']['calibration_pooled']}
    src = D.dump('14_residual_scale_ladder', payload)

    fig = page('换了三代底座、换了数据、换到折外，瓶颈都在同一处',
               'a 残差主体的份额｜b 两个尺度各自的解释度｜c 幅值：低值高估、高值低估，一直没变',
               kicker='P33 · 剩下的误差在哪 · §20.6、§22.4、§28.9',
               footer='log_z 残差，"分支 × 4 mm 区间"（v5.0 分析里称"截面"，同一算法）分箱，逐病例中位数。'
                      '四列的数据版本与口径不同，只用来看"这个比例稳不稳"，不作为精度对比。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .065, 'right': .975, 'bottom': .20,
                                          'top': .66, 'wspace': .30})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    xs = np.arange(len(scopes))
    within = [pick(m, k, 'within') for _, m, k, _ in scopes]
    a.bar(xs, within, .52, color=[col for *_, col in scopes], edgecolor=INK, lw=.6)
    for x, v in zip(xs, within):
        a.text(x, v + .012, f'{v:.1%}', ha='center', fontsize=14, color=INK)
    a.axhline(float(np.mean(within)), color=RED, ls='--', lw=1.1)
    a.text(.02, .985, f'红虚线 = 四列均值 {np.mean(within):.1%}', transform=a.transAxes,
           fontsize=11.5, color=RED, va='top')
    a.set_xticks(xs, [n for n, *_ in scopes], fontsize=11.5)
    a.set_ylim(0, .92)
    a.set_ylabel('区间内型态占 log_z 残差方差的份额')
    a.set_title('四个口径全是 69–70%', loc='left', fontsize=14, pad=12)

    track = [pick(m, k, 'track') for _, m, k, _ in scopes]
    pattern = [pick(m, k, 'pattern') for _, m, k, _ in scopes]
    b.plot(xs, track, 'o-', color=GREY, lw=2, ms=8, label='沿程轨迹（区间均值）R²')
    b.plot(xs, pattern, 's-', color=PURPLE, lw=2, ms=8, label='截面内型态（去均值）R²')
    for x, v in zip(xs, track):
        b.text(x, v + .022, f'{v:.3f}', ha='center', fontsize=11.5, color=MUTED)
    for x, v in zip(xs, pattern):
        b.text(x, v - .022, f'{v:.3f}', ha='center', va='top', fontsize=11.5, color=PURPLE)
    b.fill_between(xs, pattern, track, color=LIGHT, zorder=0)
    b.text(1.5, .78, '这段就是还没学到的部分', ha='center', fontsize=12.5, color=INK)
    b.set_xticks(xs, [n for n, *_ in scopes], fontsize=11.5)
    b.set_xlim(-.4, 3.4)
    b.set_ylim(.40, 1.03)
    b.set_ylabel('逐病例中位 R²（归一化空间）')
    b.legend(loc='lower left', fontsize=11.5)
    b.set_title('轨迹早就 0.93+，型态卡在 0.52–0.61', loc='left', fontsize=14, pad=12)

    dec = cached['models']
    xs2 = np.arange(1, 11)
    for name, color, mark in (('X5', GREY, 'o'), ('X5D', TEAL, 's')):
        c.plot(xs2, dec[name]['residual_decile_case_medians'], mark + '-', color=color,
               lw=1.8, ms=6, label=name)
    c.axhline(0, color=INK, lw=.9, ls='--')
    c.annotate('低值高估', (1, dec['X5']['residual_decile_case_medians'][0]),
               xytext=(14, 6), textcoords='offset points', fontsize=12.5, color=RED)
    c.annotate('高值低估', (10, dec['X5']['residual_decile_case_medians'][-1]),
               xytext=(-14, -18), textcoords='offset points', ha='right',
               fontsize=12.5, color=RED)
    c.set_xticks([1, 4, 7, 10], ['最低\n十分位', '4', '7', '最高\n十分位'], fontsize=11.5)
    c.set_ylim(-.26, .34)
    c.set_ylabel('逐十分位平均残差 pred − true（log_z）')
    c.legend(loc='upper right', fontsize=11.5)
    cal = cur['test34_base5']['summary']['calibration_pooled']
    c.text(.03, .035,
           f'v5.1 新底座同样：top10 幅值比 {cal["top10_pred_true_ratio"]:.3f}、'
           f'p99 比 {cal["p99_pred_true_ratio"]:.3f}、\n最大值只到真值的 '
           f'{cal["max_pred_true_ratio"]:.2f} —— 幅值范围没打开',
           transform=c.transAxes, fontsize=11.5, color=INK)
    c.set_title('向均值收缩，是同一个局部问题的另一面', loc='left', fontsize=14, pad=12)

    return save(fig, '14_residual_scale_ladder',
                claim='从 X5 到 X5D 再到 v5.1 新底座、从 test34 到患者分组折外，残差主体始终是区间内型态（≈70%），沿程轨迹已解释到 0.93 以上而型态只有 0.52–0.61，幅值同时向均值收缩。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='The four columns come from different data versions and scopes and are shown to test the stability of a ratio, not to rank accuracy. v5.0 analyses call the bin a "section"; the binning algorithm is the same.')


# ---------------------------------- 15 已用完的杠杆 vs 还没检验过的杠杆
def levers_used_vs_unused():
    wave1 = D.load(D.WAVE1 / 'results.json')
    wave5 = D.load(D.WAVE5 / 'offline' / 'analyze_wave5.json')
    w1v51 = D.load(D.W1 / 'offline' / 'analyze_wave1.json')
    t3n = D.load(D.W2B / 'offline' / 'analyze_t3n.json')
    lon = D.load(D.LONG / 'offline' / 'analyze_longitudinal.json')
    sel = D.load(D.LONG / 'offline' / 'select_channels_oof.json')
    expl = D.load(D.LONG / 'offline' / 'diagnose_residual_explainability.json')

    x0 = wave1['arms']['X0']['checkpoints']['best']['pa_r2_cb']
    deltas = {k: v['checkpoints']['best']['pa_r2_cb'] - x0
              for k, v in wave1['arms'].items() if k != 'X0'}
    band = wave1['bands']['pa_r2_cb']

    cap = w1v51['X5Dcap_paired']
    cap_mean = float(np.mean([r['d_pa'] for r in cap]))
    steno = wave5['X5I_vs_X5']['mean_d_pa']
    long_mean = lon['paired_best']['mean_d_pa']
    t3n_test = float(np.mean([r['d_pa'] for r in t3n['test34_paired']]))
    t3n_oof = t3n['cv3_pooled']['t3n'] - t3n['cv3_pooled']['base'] \
        if isinstance(t3n.get('cv3_pooled'), dict) and 't3n' in t3n.get('cv3_pooled', {}) \
        else None

    payload = {'wave1_deltas_vs_X0': deltas, 'wave1_band': band,
               'paired_means': {'X5Dcap(v5.1, 3 seed)': cap_mean,
                                'X5I 狭窄指数(v5.0, 3 seed)': steno,
                                '沿程 32 通道(v5.1, 3 seed)': long_mean,
                                'T3n test34(v5.1, 3 seed)': t3n_test,
                                'T3n cv3 折外(v5.1)': t3n_oof},
               'ceiling': {'base27_self': sel['base27_linear_r2'],
                           'all16_on_base_residual_pooled':
                               expl['X5D_v51 底座']['ols_r2_16values_pooled'],
                           'oof_forward_all16': sel['subsets']['全部 16 value'],
                           'oof_forward_ref8': sel['subsets']['ref 8'],
                           'arm_residual_left':
                               expl['X5D_long 沿程臂']['ols_r2_16values_pooled']}}
    src = D.dump('15_levers_used_vs_unused', payload)

    fig = page('只靠"再加一个特征 / 流量量"已经到顶了',
               'a 波 1 十六个候选只有一个是大效应｜b 之后每一轮加输入都更小｜c 这类标量还剩多少可解释的残差',
               kicker='P32 · 杠杆盘点 · §20.1、§21.6、§28.2、§28.7、§29.7、§29.9',
               footer='a 为 v5.0 数据、单 seed、同期对照 X0，阴影是同配置抖动带 ±0.034（读不出来 ≠ 已被否定）；'
                      'b 为同 seed 配对的三 seed 均值，阴影是 seed 噪声带 ±0.0164；c 的四个数字口径各自标注。')
    axs = fig.subplots(1, 3, gridspec_kw={'left': .155, 'right': .985, 'bottom': .175,
                                          'top': .66, 'wspace': .62})
    a, b, c = axs
    for i, ax in enumerate(axs):
        ax.set_gid(chr(97 + i))
        panel_label(ax, chr(97 + i))

    items = sorted(deltas.items(), key=lambda kv: kv[1])
    ys = np.arange(len(items))
    colors = [TEAL if k == 'X5' else (PURPLE if k in STRUCTURAL else GREY)
              for k, _ in items]
    a.barh(ys, [v for _, v in items], .66, color=colors, edgecolor=INK, lw=.5)
    a.axvspan(-band, band, color=LIGHT, zorder=0)
    a.axvline(0, color=INK, lw=.9)
    for y, (k, v) in zip(ys, items):
        a.text(v + .004, y, f'{v:+.3f}', va='center', fontsize=10.5, color=INK)
    a.set_yticks(ys, [ARM_CN[k] for k, _ in items], fontsize=10.5)
    a.set_xlim(-.02, .135)
    a.set_xlabel('Δ Pa R²$_{cb}$（相对同期对照 X0）')
    a.set_title('紫色 = 局部结构类，全部落在抖动带内', loc='left', fontsize=13.5, pad=12)

    names = ['协议版 capfit\nv5.1 · 3 seed', '狭窄指数 X5I\nv5.0 · 3 seed',
             '沿程 32 通道\nv5.1 · 3 seed', 'T3n 残差堆叠\nv5.1 · test34',
             'T3n 残差堆叠\nv5.1 · cv3 折外']
    vals = [cap_mean, steno, long_mean, t3n_test, t3n_oof]
    cols = [TEAL, GREY, TEAL, RED, RED]
    ys = np.arange(5)[::-1]
    b.barh(ys, vals, .58, color=cols, edgecolor=INK, lw=.6)
    for y, v in zip(ys, vals):
        b.text(v + (.0009 if v > 0 else -.0009), y, f'{v:+.4f}',
               va='center', ha='left' if v > 0 else 'right', fontsize=11.5, color=INK)
    b.axvspan(-.0164, .0164, color=LIGHT, zorder=0)
    b.axvline(0, color=INK, lw=.9, ls='--')
    b.set_yticks(ys, names, fontsize=10.5)
    b.set_xlim(-.033, .016)
    b.set_xticks([-.02, -.01, 0, .01])
    b.set_xlabel('同 seed 配对 Δ Pa R²$_{cb}$')
    b.set_title('对照：波 1 的 Murray 先验是 +0.100', loc='left', fontsize=13.5, pad=12)

    cl = payload['ceiling']
    keys = ['base27_self', 'all16_on_base_residual_pooled', 'oof_forward_all16',
            'oof_forward_ref8', 'arm_residual_left']
    labels = ['原 27 维自身\n对折外残差', '16 个沿程 value\n对底座残差 pooled',
              '折外前向选择\n全部 16 value', '折外前向选择\nref 8（16 列）',
              '沿程臂训练后\n残下的部分']
    vals2 = [cl[k] for k in keys]
    cols2 = [GREY, PURPLE, PURPLE, TEAL, GREY]
    ys2 = np.arange(5)[::-1]
    c.barh(ys2, vals2, .58, color=cols2, edgecolor=INK, lw=.6)
    for y, v in zip(ys2, vals2):
        c.text(v + .00012, y, f'{v:.4f}', va='center', fontsize=11.5, color=INK)
    c.set_yticks(ys2, labels, fontsize=10.5)
    c.set_xlim(0, .0082)
    c.set_xticks([0, .002, .004, .006], ['0', '0.002', '0.004', '0.006'])
    c.set_xlabel('对底座残差的线性解释力 R²')
    c.text(.47, .99, '天花板就是千分之几 ——\n再堆同类沿程标量\n换不回可读出的整体 R²',
           transform=c.transAxes, fontsize=11.5, color=INK, va='top')
    c.set_title('这一类信息剩下的量级', loc='left', fontsize=13.5, pad=12)

    return save(fig, '15_levers_used_vs_unused',
                claim='流量 / 沿程标量这条线的杠杆已经用完：Murray 先验之后每一轮加输入的配对增益都落在 seed 噪声带内，且这类标量对底座残差的线性解释力天花板只有 R² 0.005 左右。',
                sources=src, axes=list(axs), row_groups=[['a', 'b', 'c']],
                note='Panel a is single-seed v5.0 readings against a contemporaneous control; arms inside the shaded band are unreadable at that budget, which is not the same as refuted. Panel b mixes v5.0 and v5.1 references, each labelled on its own tick.')


# --------------------------------------------- 16 后续候选与最小区分实验
def future_roadmap():
    cur = D.load(D.W1 / 'analysis_20260917' / 'current_residuals.json')
    sel = D.load(D.LONG / 'offline' / 'select_channels_oof.json')
    gate = D.load(D.LONG / 'offline' / 'gate_by_validity.json')
    wave1 = D.load(D.WAVE1 / 'results.json')
    med = cur['test34_base5']['summary']['residual_medians']
    oof = cur['cv3_oof']['summary']['residual_medians']
    cal = cur['test34_base5']['summary']['calibration_pooled']
    x0 = wave1['arms']['X0']['checkpoints']['best']['pa_r2_cb']
    x11 = wave1['arms']['X11']['checkpoints']['best']['pa_r2_cb'] - x0
    x7 = wave1['arms']['X7']['checkpoints']['best']['pa_r2_cb'] - x0

    payload = {'within_bin_share': [med['within_bin_share'], oof['within_bin_share']],
               'pattern_r2': [med['within_bin_pattern_r2'], oof['within_bin_pattern_r2']],
               'top10_ratio': cal['top10_pred_true_ratio'],
               'oof_ceiling_all16': sel['subsets']['全部 16 value'],
               'gate8': gate['gate8']['pa_r2_cb'], 'base3': gate['base']['pa_r2_cb'],
               'wave1_X11_vs_X0': x11, 'wave1_X7_vs_X0': x7,
               'wave1_band': wave1['bands']['pa_r2_cb']}
    src = D.dump('16_future_roadmap', payload)

    fig = page('后续要做的是局部形态，不是再加一个沿程标量',
               '五个候选按证据强弱排；每个都先给"最小区分实验"，不先押架构',
               kicker='P35 · 后续方向 · §19、§22.4、§28.9、§29.8',
               footer='所有候选只用壁面点云 + 中心线可得的量（2026-09-13 部署可得性裁定）；真值只进诊断与训练监督，不作部署输入。'
                      '每个候选都与同 seed 底座配对、新分支零初始化、先三 seed，再看是否上折外。')
    ax = canvas(fig, (.035, .125, .93, .675))

    text(ax, 50, 97,
         f'判据（已确认）：区间内型态占残差 {med["within_bin_share"]:.0%}（折外 {oof["within_bin_share"]:.0%}）｜'
         f'型态 R² 只有 {med["within_bin_pattern_r2"]:.2f}｜'
         f'沿程标量对残差的折外天花板 R² {sel["subsets"]["全部 16 value"]:.4f}｜'
         f'top10 幅值比 {cal["top10_pred_true_ratio"]:.3f}',
         12.5, INK, True, 'center')

    cards = [
        (0.5, 52, 31.5, TEAL, 'A  轴向 × 周向分区 token（主推）',
         '假设：K16 patch 的 mean 池化把周向位置抹平了\n'
         '证据：现行 patch 逐邻居 MLP + 共享 FiLM + mean 池化，\n无邻居间交互（local_refinement.py:240）\n'
         '先做：只读折外诊断 —— 残差是否成稳定空间斑块、\n热点是否沿流向偏移、周向低阶模态折外可否解释'),
        (34.2, 52, 31.5, BLUE, 'B  固定毫米范围的 query patch',
         '假设：K16 在稀云上尺度会漂移\n'
         '证据：25% 抽稀 thin/thin −0.061，只抽 patch −0.054，\n只抽 support −0.009（P20c）\n'
         '先做：只改 patch 取点范围，三 seed 配对；\n同时看全密度与 0.5 / 1.2 mm 两端'),
        (67.9, 52, 31.5, PURPLE, 'C  高阶壁面微分几何 / 局部隆起',
         '假设：截面内型态要的是曲面量，不是又一个截面标量\n'
         f'证据：沿程收益集中在 WSS 90–99%（+9.51%）与热点 IoU；\nV6-A5 的壁面微分几何曾是当时最好的一档\n'
         '先做：把 A5 的微分几何 sidecar 接到 X5D_v51，单 seed 起'),
        (0.5, 8, 31.5, ORANGE, 'D  保留周向相位的上游条件（更远）',
         '假设：上游几何对下游热点有方位性影响\n'
         '证据：现行上下文是逐例全局向量；θ 为 RMF 平行输运，\n不跟踪弯曲方向与分叉平面\n'
         '门槛：先证明其折外解释力超过局部信息，再谈实现'),
        (34.2, 8, 31.5, RED, 'E  有界 · 病例均衡的幅值目标（条件触发）',
         '触发：若诊断显示残差没有稳定空间结构、主要是幅值偏差\n'
         f'证据：top10 幅值比 {cal["top10_pred_true_ratio"]:.3f}、'
         f'p99 {cal["p99_pred_true_ratio"]:.3f}、最大值只到 {cal["max_pred_true_ratio"]:.2f}\n'
         '注意：Pa-MSE / 加权 / 排序 / 近邻差分已有失败记录，\n不原样重扫'),
        (67.9, 8, 31.5, GREY, '不投入清单（已有结论）',
         'T3n 原配方：折外 −0.020，3/3 负，已结案\n'
         f'继续堆 ensemble：底座 3 → 5 seed 只 +0.0013\n'
         '继续加沿程标量：折外天花板 R² 0.0047\n'
         '加深加宽 / 全局注意力：不对准本次定位的瓶颈\n'
         '速度 → WSS：0.547 对直接回归 0.717'),
    ]
    for x, y, w, color, title, body in cards:
        node(ax, x, y, w, 36, title, body, color=color, fontsize=13.5, body_size=10.5)

    text(ax, 50, -6,
         f'先做不要 GPU 的那一档：按沿程通道有效性门控两模型混合，'
         f'test34 三 seed {gate["base"]["pa_r2_cb"]:.4f} → {gate["gate8"]["pa_r2_cb"]:.4f}（P31b）；'
         '架构改动等只读折外诊断出结果再启动',
         13, INK, True, 'center')
    return save(fig, '16_future_roadmap',
                claim='后续主方向是截面内局部形态：A 轴向×周向分区 token 为主推，B/C 并行，D/E 有触发条件，另有五项已结论的不投入方向。',
                sources=src,
                note='Schematic page. Every number shown is read from the listed artefacts; the candidate list itself is a plan, not a result, and no arm here has been trained.')


if __name__ == '__main__':
    residual_scale_ladder()
    levers_used_vs_unused()
    future_roadmap()
