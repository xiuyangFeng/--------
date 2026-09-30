"""Summarize diagnose.py outputs and existing phase/normalization metadata."""
import csv
import json
from pathlib import Path
import numpy as np
from diagnose import reduce_rows, write_csv

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def table(name):
    rows = list(csv.DictReader((OUT/name).open()))
    for r in rows:
        for k,v in r.items():
            try: r[k] = float(v)
            except ValueError: pass
    return rows


def main():
    phases = table('phase_summary.csv')
    raw_bins = table('point_bins_per_unit.csv')
    for r in raw_bins:
        for k,v in r.items():
            if v == '': r[k] = None
    bins = reduce_rows(raw_bins, ('arm','phase','bin'))
    write_csv('point_bins_summary.csv',bins)
    frames = table('frames_summary.csv')
    source = ROOT/'training_wss_min/experiments/joint_cycle_v52_20260929/data_audit_cache'
    q = np.load(source/'cases/AAA/ruputer/CHEN_FU/phase.npy')[:,0]
    matches = []
    for accel in range(10,17):
        decel = 27 + int(np.argmin(abs(q[27:43]-q[accel])))
        for arm in ['Iu','U0']:
            a = next(r for r in frames if r['arm']==arm and r['frame']==accel)
            d = next(r for r in frames if r['arm']==arm and r['frame']==decel)
            matches.append({'arm':arm,'accel_frame':accel,'decel_frame':decel,
                            'q_accel':float(q[accel]),'q_decel':float(q[decel]),
                            'q_abs_difference':abs(float(q[accel]-q[decel])),
                            'accel_vector_r2':a['vector_r2'],'decel_vector_r2':d['vector_r2'],
                            'accel_vector_mse':a['vector_mse_m2_s2'],'decel_vector_mse':d['vector_mse_m2_s2'],
                            'accel_direction_cosine':a['direction_cosine_gt_0.01'],
                            'decel_direction_cosine':d['direction_cosine_gt_0.01']})
    with (OUT/'matched_inlet_q_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(matches[0]));w.writeheader();w.writerows(matches)
    lines = ['# 速度谷底与减速段诊断（2026-09-30）', '',
        '仅汇总既有物理预测及已审核的eval缓存；未训练、未推理、未读取原始HDF5。fold0，seed1234，55个开发留出数据单元，固定16384点查询池（小病例按实际池大小）。单元等权，单元内按物理cell volume和相位等权。', '',
        '所有七个run均保留55个点级预测NPZ；字段是unit_id、velocity_prediction和原始velocity_rows，真值、权重及原始几何特征从已审核cache精确按row ID匹配。Iu/U0完成存量点级配对分析；其余五组使用已存per_case指标。', '',
        '主要发现：晚谷底43–57帧比早谷底5–9帧更难；U0谷底误差集中于仍保留中等速率的少量体内区域，并非真速率接近零的点主导。该结论支持优先研究减速后残留流动结构、局部体域交互和幅值／方向联合目标。不能由此直接认定涡旋或回流的物理机制。', '',
        '## 1. 可验证的幅值／方向误差分解', '',
        r'逐点恒等式：$\|\hat{u}-u\|^2=(\|\hat{u}\|-\|u\|)^2+2\|\hat{u}\|\|u\|(1-\cos\theta)$。现有分量MSE乘3才是向量MSE。表中方向项份额是两项等单元均值的比，不是1−平均cos，也不是病人比例。方向项同时依赖两向量幅值，不能等同纯角度误差。', '',
        '| 模型 | 阶段 | 向量R² | 速率R² | 方向cos（真速率≥0.01） | 方向项能量份额 |',
        '|---|---|---:|---:|---:|---:|']
    for arm in ['Iu','U0','UT0','UT1','U1','J1','J2']:
        for phase in ['decel','trough']:
            r=next(x for x in phases if x['arm']==arm and x['phase']==phase)
            lines.append(f"| {arm} | {phase} | {r['vector_r2']:.4f} | {r['speed_r2']:.4f} | {r['direction_cosine']:.4f} | {100*r['angular_fraction_ratio_of_means']:.2f}% |")
    lines += ['', 'U0相对Iu的谷底速率MSE降低，但方向项能量增加，总向量MSE几乎不变；平均cos变好与方向能量变差不矛盾，因为权重不同。谷底的幅值与方向都需要研究，不能只优化无权角度项。', '',
              '## 2. 谷底的两个不连续子窗', '',
              '此表为存量点级误差直接汇总；没有把逐帧R²平均冒充窗口R²。early=5–9，late=43–57，decel=27–42，帧编号从0起。', '',
              '| 模型 | 阶段 | 向量MSE (m²/s²) | 平均真实速率 (m/s) | 速率偏差 (m/s) | cos | 预测近零率† |',
              '|---|---|---:|---:|---:|---:|---:|']
    for arm in ['Iu','U0']:
        for phase in ['early_trough','decel','late_trough']:
            r=next(x for x in bins if x['arm']==arm and x['phase']==phase and x['bin']=='all')
            lines.append(f"| {arm} | {phase} | {r['vector_mse_m2_s2']:.6f} | {r['truth_speed_mean_m_s']:.5f} | {r['speed_bias_m_s']:+.5f} | {r['direction_cosine_gt_0.01']:.4f} | {100*r['pred_lt_0.001_given_truth_ge_0.01']:.3f}% |")
    lines += ['', '†预测速率<0.001m/s，条件是真速率≥0.01m/s；每单元先求条件体积分数再等权。', '',
              '## 3. U0速度分箱及距壁分层', '',
              '按CFD真速率分箱仅用于误差诊断，不可把真速率分箱作为部署输入。距壁阈值1/3mm是探索性分层，不能宣称是最优临床区域界限。覆盖和“误差贡献”对全部55单元等权，空箱贡献0；各箱条件MSE/偏差仅平均有该箱的单元，n_units在CSV给出。误差贡献除以本阶段总向量MSE得贡献份额。', '',
              '| 阶段 | 分箱 | 平均体积-相位覆盖 | 向量MSE | 对本阶段误差贡献 | 速率偏差 |',
              '|---|---|---:|---:|---:|---:|']
    for phase in ['decel','trough']:
        total=next(x for x in bins if x['arm']=='U0' and x['phase']==phase and x['bin']=='all')['vector_mse_m2_s2']
        for r in bins:
            if r['arm']=='U0' and r['phase']==phase and r['bin']!='all':
                lines.append(f"| {phase} | {r['bin']} | {100*r['volume_phase_coverage']:.2f}% | {r['vector_mse_m2_s2']:.6f} | {100*r['vector_error_energy_contribution']/total:.2f}% | {r['speed_bias_m_s']:+.5f} |")
    lines += ['', '## 4. 相近入口Q配对', '',
              '对加速窗每帧，从减速窗按公开protocol的q_norm找最近邻；配对不依赖误差表现。所有7对保留在CSV，没有事后仅保留最差对。减速窗同Q仍更难支持检查历史状态表示，但不是因果证明：真实场能量及空间结构可以不同，当前模型也已有dq及sin/cos输入。', '',
              '| 加速帧→减速帧 | q差绝对值 | U0加速/减速R² | U0加速/减速cos |',
              '|---|---:|---:|---:|']
    for r in matches:
        if r['arm']=='U0':
            lines.append(f"| {r['accel_frame']}→{r['decel_frame']} | {r['q_abs_difference']:.5f} | {r['accel_vector_r2']:.4f} / {r['decel_vector_r2']:.4f} | {r['accel_direction_cosine']:.4f} / {r['decel_direction_cosine']:.4f} |")
    lines += ['', '## 5. 设计含义与证据边界', '',
        '- U0谷底真速率<0.01m/s占10.11%体积-相位，却只贡献0.66%向量误差；0.10–0.30m/s占13.08%、贡献52.95%，且条件平均速率偏差−0.0661m/s。不能把“谷底时段”理解为“所有点均接近零速率”，也不支持只做逆速率加权。',
        '- U0谷底距壁≥3mm占50.75%体积、贡献57.95%误差；距壁<1mm占19.05%体积、仅贡献9.78%。没有证据支持只堆近壁分支；应将体域中等速率残留结构纳入诊断。',
        '- U0晚谷底向量MSE约为早谷底3.36倍，cos为0.6134对0.7341。真速率≥0.01m/s但预测<0.001m/s的比例约0.02%，不支持大面积预测塌缩为严格近零这一解释。',
        '- 当前训练已经逐相位逐分量z-score（80×3）和case/phase等权；低速相位物理误差本就乘更大的1/std²，不能把低谷差直接归因于峰值绝对误差支配。优先补训练阶段分相位损失和共享梯度诊断。',
        '- 方向稳定的向量目标、局部坐标表示、体域交互及历史状态模块是可检验假说；现有存量评估不能证明是哪一个导致误差。局部坐标头要配同样几何输入的Cartesian控制，并保持物理损失口径，避免坐标改变与损失权重变化混杂。',
        '- 当前没有缓存中心线切向量，不能计算可信的轴向回流符号；cos<0只表示预测与真实速度夹角>90°，不等于真回流。当前未计算涡量、散度、流量守恒。',
        '- 无训练逐相位梯度记录，不能确认梯度竞争或某阶段欠训练；仅训练aggregate loss不足。两组存量点级诊断仍是已暴露开发集、单seed结果。',
        '- 不建议无门控地对低速点使用纯角度损失：真速率接近零方向不稳定，须固定可解释的速率门控并报告覆盖；不能以排除困难点提高cos。', '',
        '产物：[分阶段](phase_summary.csv)、[55单元配对](paired_summary.csv)、[点级分箱汇总](point_bins_summary.csv)、[逐帧汇总](frames_summary.csv)、[同Q配对](matched_inlet_q_frames.csv)、[数据与数值核对](provenance.json)。可复算：先 `python diagnose.py`，再 `python report.py`；建议OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。', '']
    (OUT/'diagnostics.md').write_text('\n'.join(lines))
    print(OUT/'diagnostics.md')


if __name__=='__main__':
    main()
