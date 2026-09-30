# WSS 最小化路线代码修改与实验推进记录

> 用途：记录 WSS 相关独立路线，包括 `pipeline_wss_min/`、`training_wss_min/`、`wss_mri_calculator/` 与 `wss_pinn/` 的代码、数据、图件和实验推进。
> V3P / 训练主线 / 通用代码修改记录见：[代码修改与实验推进记录](代码修改与实验推进记录.md)。
> 当前执行入口：**[推进与变更导航](README.md)**（2026-09-24 起 V5 按 00–05 六块组织，各块有自己的实验跟踪）/ 旧 WSS-min 路线（🧊 冻结，2026-09-24 已归档）：[PointNet baseline 矩阵](_archive/WSS最小化/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) · [训练实验跟踪](_archive/WSS最小化/WSS最小化_训练实验跟踪.md) / [体域 PINN 路线史](_archive/WSS_PINN/README.md)（🧊 2026-09-24 归档） / [velocity→WSS V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)。
> **滚动切卷**：本文件只保留 2026-08 以来的条目；2026-07 条目（PointNet 矩阵、新队列审计、WSS-PINN F0/F1）见历史卷
> [2026-07卷](_archive/WSS最小化_代码修改与实验推进记录_2026-07卷.md)。主文件超过约 1500 行或跨季度时，把最旧月份整月切入 `_archive/` 新卷并更新本索引。

## 2026-09-30｜数据统一：唯一数据版本 v5.2c 统一根，旧口径数据删除

- **本次主要修改**（用户要求所有实验的数据基础一致、旧口径数据处理掉，改库与删除清单经用户逐项批准）：
  - **库**：recover 10 个单元换入 data_new，库内旧版归档后删除 73.1 GB；3 个合成单元按协议重算后换入库。
  - **主 PREP**：12 个 recover / YANG 单元的中心线、atlas、审计从隔离根并入，重算结果与隔离根一致。
  - **统一根**：快照 `anatomy_pointcloud_v5_2c_20260930`、视图 `views_v5_2c_20260930`，332 单元，全部分区（IND、CV5、syn、full265）和统计都在其中，v5.2p5 并入。
  - **分流规则与特征包**：开口半径分流规则按修正数据重拟合（a 1.150 → 1.265，b 0.147 → 0.096，train R² 0.814 → 0.906；旧规则用了错标签和端点半径修正前的开口半径）。flowref / phys1d 对 332 单元按新规则重建，去掉 v5.1 那份由标签派生的图谱先验。
  - **删除**：库内 processed/（V1–V3 旧管线）、*_old_* / *.orig* 备份、data_wss_min 共 586.9 GB；旧快照 / 视图根、旧实验缓存、recover 变体、隔离根在统一根自检通过后删除。
- **对应代码/文档**：
  - 文档：[数据统一与旧版本清理](04-数据处理与CFD/数据统一与旧版本清理_2026-09-30.md)。
  - 脚本：`training_wss_min/experiments/wss_v52c_labelfix_20260930/`（`consolidate_v52c.py`、`refit_flow_split_rule.py`、`swap_flowref_phys1d.py`、`verify_unified.py`、`verify_after_move.py`、`make_configs.py`）、`outputs/cfd_auto_trial_20260927/_recover/`（`swap_into_data_new.py`、`merge_prep_into_main.py`）、`data_new/_archive/cleanup_20260930/cleanup_library.py`。
  - 代码：`training_wss_min/joint_cycle_data.py` 读取 `unified_snapshot_root`（旧计划文件不含该键，行为不变）；`repoint_data_root` 把 v5.2 / v5.2p4 / v5.2p5 都映射到 v5.2c。
- **推进到实验步骤**：统一根上 IND、CV5 五折、full265 严格加载通过，X5Dcap 特征 z-score 与原文件 0 差（主线输入不变）；23 个主线配置指向统一根。
- **当前状态判断**：此后只有一个数据版本（v5.2c）；用户选择暂不重训，训练结果 `training_wss_min/runs` 保留。

## 2026-09-30｜库内 9 个单元标签修正：换入 data_new，建 v5.2c / v5.2p5，统一训练和测试口径

- **本次主要修改**：
  - **换入与删除**：9 个边界条件不合协议的库内单元（RCR 挂到相邻出口、RCR 复制、入口除数 ≠ 网格入口面积）用同网格、协议正确的重算结果换入 data_new；旧文件先归档，再删掉 110.6 GB 大文件。
  - **审计与命名**：刷新共享 PREP 审计；4 例出口命名用覆盖表固定成与 v5.2 相同。
  - **快照与视图**：重建 V5 快照和全部视图包，另为 6 个在库单元重建 `volume.npz`。
  - **数据版本**：组装修正版 v5.2c（取代 v5.2）和 v5.2p5（取代 full265 v5.2p4），重算标签统计。
  - **全周期缓存**：重建 joint_cycle 缓存。
  - **配置**：生成 23 个主线 X5Dcap_asym2 重训配置。
- **对应代码/文档**：
  - 文档：[标签核查 §6](04-数据处理与CFD/库内标签问题核查_RCR挂错与入口除数_2026-09-30.md)、[数据回收 README §4.12](04-数据处理与CFD/数据回收_2026-09-20/README.md)、[01 块跟踪 §38](01-X5D主线与新数据/X5D主线_实验跟踪.md)。
  - 代码改动：`wss_pinn/v4/new_case_sources.py` 优先读出口语义覆盖表 `wss_pinn/configs/outlet_semantics_overrides_20260930.json`；`training_wss_min/joint_cycle_data.py` 加 `GNN_JOINT_VIEW_ROOT` 和 plan 标签 `v5.2c-labelfix`。两处都是不设或不登记时行为不变。
  - 新工具：`training_wss_min/tools/repoint_data_root.py`；`training_wss_min/experiments/wss_v52c_labelfix_20260930/`（`prepare_v52c.py`、`make_configs.py`）。
- **推进到实验步骤**：
  - **逐数组比对**：几何和输入特征逐位相同，只有标签变。
  - **统计**：代码路径在旧根上逐位复现原文件；WSS log 均值高 0.001–0.006；特征 z-score 0 差；严格加载通过。
  - **全周期缓存**：audit 0 错。
  - **尚未完成**：3 个合成单元还在重算（syn 分区暂不可用）；重训未提交。
- **当前状态判断**：此后训练和测试统一用 v5.2c / v5.2p5；v5.2、v5.2p4 上的旧数字（v5.2 基线、§36、full265 16192 的 0.8393）标「标签修正前」，只作历史参照。

## 2026-09-30｜时间建模两份主文档合并与训练轮次审查

- **本次主要修改**：8份阶段文档合并要点后归档，保留合并前完整跟踪快照；02根目录只留结果主文档、思路/文献主文档和短README。结果保留稳定§30/§31实验编号、三轮6＋10＋8次主表与历史失败边界；思路按“方法/文献—实验—结果—待验证问题”对齐。相对链接重定位，原run/配置/冻结源码不改。
- **对应代码/文档**：[完整实验跟踪与结果](02-时间建模/时间建模_实验跟踪.md)、[研究思路与文献](02-时间建模/时间建模_研究思路与文献.md)、[归档索引](02-时间建模/_archive/README.md)、[训练轮次审查证据](../../training_wss_min/experiments/velocity_phase_v52_20260930/analysis_20260930/training_horizon_review.md)及同目录可复算脚本/JSON/CSV；同步上层导航和根README。
- **推进到实验步骤**：仅读取已有history/config。121–140到141–150轮共同训练z-MSE下降：U0 0.182%、D1 0.436%、G10 0.331%、G11 0.294%；G11谷/减速仅0.286%/0.426%。末学习率1e-5，无逐轮留出曲线，不能由train平台确定过拟合或断言增训收益。
- **当前状态判断**：新增§30.28登记H-U0-300/H-G11-300单seed预算与退火日程敏感性候选，低学习率原轨迹续训另列控制思路；未提交新训练或推理，原150轮结果和工作簿不改。归档不表示所有后备方法已执行。

## 2026-09-30｜速度难段8臂全部完成、结果分析与工作簿同步

- **本次主要修改**：核验16265/16266全部8臂及16267报告成功，150轮/7800步/last/55单元/80相位齐全；刷新报告、生成分阶段/病例配对/成本分析，完成checkpoint、440预测和隔离指纹审计。只读已有产物，无新训练或模型推理。
- **对应代码/文档**：[实验入口](../../training_wss_min/experiments/velocity_phase_v52_20260930/README.md)、[完整分析](../../training_wss_min/experiments/velocity_phase_v52_20260930/analysis_20260930/analysis.md)、[结果矩阵](../../training_wss_min/experiments/velocity_phase_v52_20260930/analysis_20260930/results_matrix.md)、[时间跟踪§30.27](02-时间建模/时间建模_实验跟踪.md)；正式工作簿`WSS_PointNet实验矩阵与结果汇总last.xlsx`新增`V52速度周期优化`页，Iu/U0明确历史参照，备份与逐项数值核对见实验分析目录。同步02、根README、文档导航和总纲的时间维说明。
- **推进到实验步骤**：G11峰/谷/减速/周期R² 0.8167/0.3159/0.6876/0.7571，对U0谷底+0.0157/减速+0.0094；固定查询时延0.3535s（1.654×U0）。D1方向改善但精度增量小，D2/A1当前实现未获支持。58旧保护文件和70冻结输入未变，全量16192/16193正常结束。
- **当前状态判断**：本轮完训完评，8臂无一满足全部主对照预注册条件；G11保留研究候选，D1保留方向线索，不启动组合/拓扑追加或多seed。206/55、fold0、单seed开发筛选；不混旧R²_cb，不替换部署，不新增压力/WSS/OSI结论。

## 2026-09-30｜v5.2p4 full265 数据版本 + X5Dcap_asym2 三 seed 全量训练（评估集 recover8）· 完成并回填工作簿

- **用户裁定**：评估集 recover11 → recover8（3 例同病人另一期在 v5.2 → 放回训练）；YANG_BAO_KUI 入训练；训练 265 = v5.2 261 + YANG + 3，val 空，test = recover8；X5Dcap_asym2 × seed 1234/7/2025。详见 [01 块跟踪 §37](01-X5D主线与新数据/X5D主线_实验跟踪.md)、[数据回收 README §4.11](04-数据处理与CFD/数据回收_2026-09-20/README.md)、[cfd_auto §10.8](04-数据处理与CFD/STL全自动CFD工程cfd_auto_试算_2026-09-27.md)。
- **结果**：recover8 三 seed 集成 R²_cb 0.8393（同 8 例 IND 三 seed 0.8352、五 seed 0.8393），逐例中位 0.843，6/8 改善；n = 8 只作描述。权重 `training_wss_min/runs/wss_v52p4_full265_20260930/`，未打包、部署未切换。工作簿 WSS实验矩阵 603–605 行（ΔR² 相对同一 recover8 上的 IND 同 seed：+0.007 / −0.007 / +0.015）。
- **新增**：`training_wss_min/experiments/wss_v52p4_full265_20260930/`（`prepare_full265.py` 拼视图根 / 密度链接 / 代码路径复现 / split·统计 / 检查 / 配置；`ref_ind_recover8.slurm`）；`training_wss_min/cluster/{prepare,preflight,run}_wss_v52p4_full265*.slurm`；`outputs/cfd_auto_trial_20260927/_recover/eval/` 下 recover8 三件（split、EVAL_SET、`evaluate_recover8.sh`）、`stage_yang_20260930.py`、`compare_recover8.py`、`eval_recover8_full265.slurm`。
- **改动（全部向后兼容）**：`_recover/eval/chain.slurm` 加 `REC_CL/REC_SNAP/REC_VROOT/REC_SPLIT/REC_RUN` 环境覆盖（不设时路径逐字不变）；`readout_generic.py` 加 `--title`（默认 recover11，旧读数重跑逐字相同）；`training_wss_min/tools/update_wss_local_wave1_xlsx.py` 加 full265 组专用分支（协议 / 备注 / IND 同 seed 配对参照，只对 `wss_v52p4_full265*` 生效）；`tools/annotate_workbook_methods.py` 加 1 条 GLOSS。`training_wss_min/*.py` 未改；训练在冻结副本 `GNN_v52p_frozen_20260926` 里执行。
- **作业**：YANG 入库链 16180、sanity 16181、prepare 16188、预检 16191、队列 16192（4 h 09 min）、读数 16193、IND 参照评估 16300；全部 COMPLETED。v5.2 视图 / 快照、data_new、共享 PREP 本任务 `find -newermt` 核查为 0。

## 2026-09-30｜速度谷底/减速段8臂 · 隔离实现并提交Slurm

- **执行合同**：[实验入口](../../training_wss_min/experiments/velocity_phase_v52_20260930/README.md)、[提交记录](../../training_wss_min/experiments/velocity_phase_v52_20260930/submission.json)。仅D1/D2/A0/A1/G00/G01/G10/G11八个速度臂，每臂seed1234；沿用研究实验261单元缓存、206训练/55开发留出、fold0、150轮/last、80相位，不采用全量模型的数据划分/数量。
- **兼容与验证**：独立新增入口及模块，43项CPU测试通过；58个旧训练源码/全量配置指纹未变。`f0_s1234_v1`冻结33个源码传递依赖和8份配置，A0/A1及图新增有效参数配平验收通过；几何sidecar与方向λ完成后分别保存SHA门禁。正在运行的全量作业16192、其冻结源码及配置不修改。
- **队列**：09-30 03:18提交16263 node03几何→16264 GPU标定/8臂冒烟→16265第一批数组→16266第二批数组→16267 afterany报告。正式GPU任务上限4，实际按Slurm GRES排队。03:23核对几何与全部8臂GPU门禁通过，D1正式训练RUNNING，D2/A0/A1等待资源、图家族及报告等待依赖；全量作业16192继续运行。D1训练侧32batch标定λ=0.0860702960，尚无新正式成绩，冒烟不作为精度结果。
- **回填**：02入口、§30.26、[速度矩阵执行注记](02-时间建模/_archive/阶段文档_2026-09-30/速度谷底与减速段_误差机制与实验矩阵_2026-09-30.md)、实验README。峰值/谷底/减速/整周期及幅值、方向、解剖分解、逐帧和成本均纳入报告；后备或组合、压力/WSS不另启动。单折单seed开发筛选，提交不等于完成。

## 2026-09-30｜速度谷底/减速误差机制与单seed矩阵 · 诊断和设计完成

- **实测诊断**：七组已有指标、Iu/U0共110份已存预测与固定查询cache配对，未新推理。谷底0.10–0.30m/s区域贡献52.95%向量误差、覆盖13.08%；晚谷底比早谷底MSE高3.36倍；方向相关能量占谷底57.82%、减速46.86%。现有逐相位标准化已提高低流量相位物理误差权重，近零预测塌缩与单纯近壁误差不受当前证据支持。
- **设计**：[速度难段实验矩阵](02-时间建模/_archive/阶段文档_2026-09-30/速度谷底与减速段_误差机制与实验矩阵_2026-09-30.md)。8次单seed、两批：D1/D2分别改方向/速率尾部监督，A0/A1区分新增几何和局部坐标输出，G00/G01/G10/G11区分体内图交互与相位条件；四图臂BC与锚点一致，先做表面几何生成与数值验收。条件追加拓扑控制/两项组合，记忆和通量分解保留独立预算。
- **证据与记录**：[诊断报告及CSV](../../training_wss_min/experiments/joint_cycle_round2_v52_20260929/analysis_velocity_20260930/diagnostics.md)、[六篇文献核验](../../outputs/researchwrite/velocity_phase_design_20260930/literature.md)、02入口与§30.25。确认MIA2026为峰值单帧目标，不误引为全周期。无新增训练/推理、无部署改变；单折开发数据启发的设计，未声称机制已验证或达到临床精度。

## 2026-09-30｜V5.2第二轮全周期精度十臂 · 全部完成与结果回填

- **状态核验**：16165_0–9与16166均COMPLETED/0:0，最后训练01:56:04、报告01:56:06完成。每臂seed1234、fold0、150轮/7800步、last、同55开发留出单元及80相位；checkpoint身份与47项冻结指纹一致，无非有限值或指数裁剪。[完成核验](../../training_wss_min/experiments/joint_cycle_round2_v52_20260929/completion_verified.json)。
- **结果**：U0速度峰值/谷底/周期R² 0.8033/0.3003/0.7480，周期优于Iu但谷底略低；P1相对压力0.8874/0.8968/0.9033，对Ip周期MAE下降16.44%；W1标量WSS 0.7630/0.4691/0.7911，派生TAWSS R² 0.7853。保留三个研究候选；TCN三段R²未胜FFN控制，查询注意力未胜均值聚合，PCGrad未恢复三任务共同受益且训练成本增加约48%。
- **交付与回填**：[完整分析](../../training_wss_min/experiments/joint_cycle_round2_v52_20260929/analysis_20260930/analysis.md)、[全部模型成绩](../../training_wss_min/experiments/joint_cycle_round2_v52_20260929/analysis_20260930/results_matrix.md)、144行分段CSV/960行逐帧CSV；已更新实验README/report、02入口/矩阵/§30.24、03入口/§34.15及上层导航。
- **边界**：单折单seed、数据单元等权指标，不与旧R²_cb混比；P1周期R²仍略低于J1，W1相对容量控制的收益小，速度谷底未解决。三候选已有固定查询计时账面合计中位0.560s，对原独立模型0.394s，非临床全点云时延。本次仅分析既有产物，未新训练/推理、未评价OSI、未替换部署。

## 2026-09-29｜V5.2第二轮全周期精度十臂 · 配置冻结并提交

- **执行**：[第二轮入口](../../training_wss_min/experiments/joint_cycle_round2_v52_20260929/README.md)。用户授权后完成UT/WT时间卷积与控制、U查询聚合、W壁面patch、P1压力单任务交互、J2-PCGrad；每臂seed1234/fold0/150epochs/last，最多4卡。复用首轮206训练/55开发留出缓存，不重训旧基线。
- **验证**：新增隔离模型/runner/metrics/report/submit模块；42项CPU检查、47项冻结指纹通过；四对新增有效参数差均<1%。固定f0_s1234_v1快照。16164十臂GPU冒烟→16165训练数组→16166 node03报告；提交不是完成，实际状态见执行入口。
- **报告**：峰值窗/谷底窗/整周期R²与误差，另补第21帧、80帧、方向、TAWSS、时间增量和成本。第一轮冻结合同与部署不改，旧基线新增字段缺失如实保留。

## 2026-09-29｜全周期分相位精度与下一轮10次单seed矩阵 · 设计完成

- **交付**：[分相位报告](../../training_wss_min/experiments/joint_cycle_v52_20260929/analysis_round2/phase_report.md)、可复算脚本/72行CSV/来源JSON、[下一轮实验矩阵](02-时间建模/_archive/阶段文档_2026-09-30/下一轮全周期精度优化_实验矩阵_2026-09-29.md)。由已有metrics和55行逐单元记录对账，未新增模型推理。
- **依据**：Iu峰值窗/谷底窗/整周期R²为0.7941/0.3048/0.7432，Iw为0.7500/0.4630/0.7817；J1压力周期最好但谷底低于J1c。整段R²不是逐帧R²均值，峰值窗不是单帧21。J1相对独立模型周期R²改善单元分别u12/55、p41/55、WSS11/55。
- **设计**：单任务压力空间交互、联合PCGrad、速度查询聚合、壁面query patch及u/WSS局部时间TCN，共10次单seed，必要容量控制成对。全部仍为建议；尚未提交新训练或评价，首轮冻结合同不改。

## 2026-09-29｜V5.2 全周期 u/p′/WSS 单 seed 核心矩阵 · 完训完评

- **状态**：数组16151六臂Iu/Ip/Iw/J0/J1c/J1均150/150 epochs、7800 steps，统一seed1234、患者分组fold0（206/55）、80相位、last；55个相同留出单元评价齐全，history连续且无非有限值，退出码全0。北京时间19:17完成，自动报告16152完成。
- **结果**：[正式报告](../../training_wss_min/experiments/joint_cycle_v52_20260929/report.md)。I的速度向量/p′/WSS周期R²为0.7432/0.8690/0.7817，J0为0.7245/0.8551/0.7725，J1为0.7274/0.9048/0.7619。相位条件空间交互主要改善压力，尚未实现三任务全面增益。J0采样查询中位0.2168s，对I三模型总和0.3938s。
- **边界与回填**：数据单元等权均值，不同于历史R²_cb；单折单seed筛查，固定查询池计时不代表全点云临床时延，未评价OSI。本次仅读取既有产物并更新实验README、02入口与§30.21，未新增训练、推理或测试集评价。

## 2026-09-29｜全周期时序文献与后续实验矩阵 · 设计与只读审计完成

- **交付**：[文献证据与分层实验矩阵](02-时间建模/_archive/阶段文档_2026-09-30/全周期血流时序模型_文献证据与分层实验矩阵_2026-09-29.md)，含同期控制、FiLM归因、局部patch/相位相关聚合、容量配平、后续图算子及患者BC/完整体场路线；[依据与独立审查包](../../outputs/researchwrite/temporal_hemodynamics_20260929/00_scope.md)。
- **最新产物核对**：15975队列42项已完成；C-FiLM v5.2 CV5三seed，全周期/谷底Pa R²_cb 0.6108±0.0044 / 0.4569±0.0153，谷底Spearman0.6528±0.0023。只读report.json/queue_status并回填02入口、跟踪与旧讨论稿，不是新增推理。
- **设计依据**：当前时间路径绕开供体full-wall query patch；该缺口仅支持优先做消融，未证明是误差原因。现有一次几何编码、多相位解码不重复包装为新贡献。旧周期Pa表81帧与正式周期标签80帧、端点幅值差53/261、顶点/面积口径及患者簇统计一并纳入新合同。
- **证据边界**：标量WSS序列不能生成OSI方向、速度或压力；高水平期刊文献按全文/摘要层级引用，不将预印本算作一区；官方年度分区未逐刊核准。保留全周期优先及暂不做WSS向量的近期范围。
- **本轮未执行**：新增训练、测试集评估、生产代码/模型/服务变更。矩阵阈值是新研究建议，不改变旧预注册判定。

## 2026-09-29｜当前最好模型的逐例 MAE / R² / Spearman 表 · 已算出

- **口径**（用户确认）：峰值 WSS = X5Dcap_asym2 CV5 折外 261 例；压力 = PF6、速度模 = VF6，v5.1 test34；TAWSS / OSI = v5.2 M1cap 三头对应通道、CV5 261 例。均为 checkpoint best、三 seed 预测逐点平均。最好 / 最差 / 中位按逐例预测 R²，中位取距中位数最近的实际病例；平均是逐例指标的算术平均。未做后处理，未重新推理。
- **对账**：单 seed 病例等权 R² 与已发表读数一致（asym2 0.7750 / 0.7707 / 0.7684；OSI 0.5265 / 0.5314 / 0.5274，集成 0.5735；PF6 三 seed 均值 0.7894；VF6 0.7952 / 0.7936 / 0.8004）。表内「逐例平均 R²」高于这些 R²_cb，两者不是同一个数。
- **代表例**：峰值最好 AAA/unruputer/ZHU_ZI_HAI 0.926、最差 AAA/ruputer/MENG_GUANG_QIN 0.339、中位 ILO/ZHANG_HE_PING-0/before 0.802。该最差例也是 TAWSS 最差（0.301）。

## 2026-09-29｜v5.2 X5Dcap 逐例 R² 最低病例后处理包与 Spearman · 已交付

- **选例**：与 09-28 最高例同一排序（261 例、X5Dcap、checkpoint best、CV5 折外三 seed 平均）。最低为 `AAA/ruputer/MENG_GUANG_QIN`（第 4 折折外）R² **0.4220**；三 seed 0.4477 / 0.4095 / 0.3714。
- **Spearman**：同一原始壁面点、`scipy.stats.spearmanr`（并列取平均秩）。该例集成 **0.9265**（三 seed 0.9238 / 0.9282 / 0.9188）；最高例 `AG/fast/LIU_YI_BING` 集成 **0.9797**。未在面片上重算，未重新推理。
- **交付**：`outputs/field/postview/wss_v52_X5Dcap_cv5oof_worst_20260929/`。原生壁面 14527 节点、28919 三角面；selfmax 分母 CFD 87.37 Pa、预测 74.61 Pa。Gaussian 覆盖 100%，映射距离最大 2.4e-5 mm。展示用 `MENG_GUANG_QIN__cfd_wall_mesh.vtp`。

## 2026-09-28｜v5.2 X5Dcap 逐例 R² 最高病例后处理包 · 已交付

- **选例**：v5.2 全部 261 例，X5Dcap、checkpoint best，CV5 折外三 seed（1234 / 7 / 2025）`pred_pa` 平均后的逐例预测 R²。最高为 `AG/fast/LIU_YI_BING`（第 1 折折外）**0.9363**；三 seed 单独 0.9275 / 0.9239 / 0.9236。独立测试 91 例五 seed 的最高例是 `AG/slow/WANG_BAO_SHAN` 0.8958，未出图。
- **交付**：`outputs/field/postview/wss_v52_X5Dcap_cv5oof_best_20260928/`。原生 CFD 壁面、点云、Gaussian STL 三个 VTP，峰值帧 1162，原量 Pa 与 selfmax 分母固定为壁面同点最大值（CFD 32.67 Pa、预测 35.96 Pa）。正式 R² 与缓存同点真值一致；Gaussian 覆盖 100%，映射距离最大 1.1e-4 mm。未重新推理。
- **当前状态**：可直接用 `LIU_YI_BING__cfd_wall_mesh.vtp` 展示。这例在 IND 的 170 例训练集中，图上数字是折外预测，不能说成 91 例独立测试。

## 2026-09-28｜wss_deploy v0.15.11 算子与缓存提速 · 已完成并上线

- **交付**：[实施与验收](../../training_wss_min/experiments/wss_deploy_timing_20260917/analysis_20260928/implementation_20260928/README.md)（七项改动的精度证据、GPU 抖动对照、组合请求新旧端到端计时、脚本与结果），部署块活文档 §29.9、`wss_deploy/README.md` v0.15.11、契约 §26。
- **接续**：下午 codex 会话按审计开始实施，写出 `cache_handoff.py` / `volume_cache.py` / `prepared_inference.py` 后额度用尽（无测试，设备输入复用未接入）。本会话补全：设备输入块键因各成员 `feature_stats_path` 不同而从不命中（复用 0 次）→ 键去掉路径；case 哈希每成员 18–24 ms → 每作用域一次；`volume_case` 条目补进缓存清理；接入三个模型族。另加四项算子：体场采样内判改用中心线锚点安全球证书（VTK 调用少 40–60%）、封口边界一维去重、流线端点方向复用（两子代理之一）、`build_case` 只算封口记录（另一子代理）。端到端发现 M1 首调 0.7–6 s 乱跳，定位为 `input_memo` 约 300 MB 私有副本与单线程查询块构建，复用路径改为绕开副本并并行构建，首调 1.9–2.6 → 0.6–0.8 s。
- **精度**：几何 / 采样 / 特征 / 导出链逐位相同（真实任务核对：体场缓存 2 例三路、采样 5 例整例、流线 2 例 + 与生产 vtp 相同、caps 6 例 43 数组）；设备输入复用 CPU 逐位相同，GPU 与原实现差 ≤ 1.9e-5 Pa、与原实现两次运行间差（≤ 3.1e-5 Pa）同分布。744 项测试通过 / 3 跳过（+44），黄金回归 6/6。
- **速度**：M1 三头 + 体场组合请求（上传即确认，GPU 2）SHI_YUN_XI 112.5 / 126.4 → **80.1 / 82.2 s**、FAN_JIAN_MING 32.7 / 33.7 → **25.8 / 24.1 s**；体场伴随任务 B 段 SHI 48–56 → 17 s。
- **没做**：VTK 多线程后端（进程全局状态）、省掉流线里重复的 `inside()`（无法证明逐位相同，待裁定）、确认前预算体场几何、模型内同步 / FiLM / 混合精度、summary 真实墙钟分项。用户裁定流线重复 `inside()` 暂不省；提交 `0b975ab` 已推 origin，09-28 19:57 `service upgrade` 上线（PID 3496824）。

## 2026-09-28｜部署单病例速度与算子审计 · 已完成

- **交付**：[审计报告、原始计时抽取与 CPU 微基准](../../training_wss_min/experiments/wss_deploy_timing_20260917/analysis_20260928/README.md)。读取当前三个发布包配置、两组组合任务与真实源码，区分冷路径、预计算命中、重跑及上传至双报告完成的计时口径。
- **主要发现**：组合子任务在父任务预计算运行期间创建时跳过复制缓存，已有一例体场重复重采样 10.77 s、形态测量 4.97 s；后续优先共享设备输入、去诊断同步、几何图/插值计划复用、体场几何缓存，FiLM 代数重排和混合精度另作数值验证。单例 57,952 点 CPU 微基准：仅需 caps 时完整 cloud 0.208 s → caps 路径 0.091 s，元数据完全相同（复用间距则 0.022 s）。局部倍数不代表整例加速。
- **边界**：没有修改生产源码、发布包、已有任务或服务，没有运行 GPU 推理、训练或新增测试集评估；所有生产优化与端到端收益仍待实施验证，保留已有未提交工作。

## 2026-09-27｜时间建模实验与老师 Transformer 思路复核 · 已完成

- **交付**：[时间建模复核与优化顺序](../../training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/analysis_20260927/time_modeling_review.md)，复核 T0/TB、冻结/解冻适配器、正式 r3 四臂、体场比较合同与已有数据诊断。
- **解释修正**：T0 的 TAWSS 0.7164 高于 T-null 约 0.696；r3 跨帧注意力全周期增量 warm −0.0015、raw +0.0017，没有稳定主指标优势，但隐藏特征有时间形态收益。单 seed 不能宣称差异“在噪声内”；跨帧相关 0.96 与 K=8 真值投影不证明可预测性上限；出口 RCR 多由几何按协议生成，不能把“不输入”直接等同不可学习。
- **回填**：修正 [老师草图综述](02-时间建模/_archive/阶段文档_2026-09-30/老师Time_Transformer想法_已验证清单与模型框架图_2026-09-24.md)，在 [时间跟踪](02-时间建模/时间建模_实验跟踪.md) 与 [路线入口](02-时间建模/README.md) 登记。建议先查周期首尾/积分/标签合同，再作同预算锚定对照与局部时空表示检验；所有优化仍是建议。
- **范围**：仅读取已有产物并更新分析文档，未新增训练、推理、测试集评估或 CFD；未改变部署及运行队列。

## 2026-09-24｜实验设计总纲与 docs/README 按 09-15 之后的证据重写 · 已完成

- **本次主要修改**：[实验设计总纲](../实验设计总纲.md) 从 09-15 版（RCR 修复完成前、v5.0 数字 0.7465、TAWSS/OSI 列为后续研究）重写为 09-24 版，七节结构保留：§1 当前判断改为「数据已修好并扩充，缺的是外部证据」，进度表按 v5.1 / v5.2、X5D_v51 底座、v5.2 中途读数、learning curve、归因与负结果、采样与几何鲁棒性、失效三族、TAWSS/OSI、时间维、PF6/VF6、部署链路、外部证据逐行更新并给出处；§2 假设改为 H1–H5（信息 vs 容量、采样、数据量、区域描述量、跨输出）并写入部署可得性硬约束；§4.2 验证设计写明 **91 个回收单元入库时读过冻结模型预测并据此查因修数据，不是未触碰集，09-21 路线图的「73 例确认集」计划作废，目前没有未触碰集**；§5 矩阵加状态列，外部基线 / BC 变化 / 真实分割面 / 不确定度标为未开始；§6 新增 S0′ 底座选择，给出**建议规则（待用户确认）**：以 CV5 的 15 对配对折差为准，两口径同向且超过噪声带才换 X5Dcap，IND 只作旁证。
- **[docs/README](../README.md)**：文首与入口表更新（论文规划改指 09-21 顶刊路线图，汇报入口补 TAWSS/OSI 包、导师汇报提纲、全周期谷底讲解）；「当前结论」压缩为六条并与总纲对齐；任务 B 说明补周期量现状；模块表把 PF6/VF6 改到 `training_wss_min/`，新增周期量与「体域 PINN（已停）」行；归档区加 09-15 版总纲快照。
- **对应文档**：09-15 版总纲原文存为 [`_archive/实验设计总纲_2026-09-15版快照.md`](_archive/实验设计总纲_2026-09-15版快照.md)（链接已按归档位置改写）。两份文件相对链接 0 断链，无 HTML 注释或引用式链接。
- **当前状态判断**：已完成；本次只汇总已有产物与文档，未新增训练、CFD 或评估。`01-任务/` 各页与 09-15 论文规划未改（论文规划的数字是修正前的，总纲与 docs/README 已注明）。

## 2026-09-24｜文档重组第二步：Centerline_V5 进 04 块、两份冻结跟踪表归档、docs/README 当前结论重写 · 已完成

- **本次主要修改**：`Centerline_V5_点云atlas接入与出口命名修正记录_2026-09-07.md` 移入 [`04-数据处理与CFD/`](04-数据处理与CFD/README.md)（仍在推进，列入该块文件表）；2026-08-07 起冻结的 `WSS最小化_训练实验跟踪.md`、`WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md` 移入 [`_archive/WSS最小化/`](_archive/WSS最小化/README.md)（文首加移入说明，归档索引登记）。本目录根现在只剩导航 README 和两份推进记录。
- **[docs/README](../README.md) 当前结论**：由 2026-09-15 版（v5.0、RCR 修复未完成、X5D 0.7465）改写为 2026-09-24 版：v5.1 / v5.2 数据面、部署底座 X5D_v51 0.7749（历史卷 §28.1）、v5.2 中途读数（`readout_ckpt_best.md`，09-23 生成，CV5 X5Dcap 未跑完）、learning curve、时间维、TAWSS / OSI、部署工具与当前优先级；文首注明实验设计总纲仍是 09-15 版；模块表补 `wss_deploy/`，维护规则第 1 条改为写所属块跟踪。
- **对应代码/文档**：脚本改写 32 个文件 96 处路径与相对链接（含 `training_wss_min/tools/{finalize_q2v_submission_records,analyze_pointnetpp_sa1_scale_results,analyze_sa_grouping_single_seed_results,annotate_workbook_methods}.py` 的路径常量，编译通过）；`.cursor/rules/wss-min-log.mdc`、`PROJECT_PROFILE.md`（WSS-min 路线 status_doc、路线状态层）同步。另把 07 月卷、新队列审计、07-29 讨论稿里 81 条早已断开的跟踪表链接修成指向 `_archive/WSS最小化/`。
- **当前状态判断**：已完成；指向三份移动文件的断链 0。实验设计总纲（09-15）未改，数字早于 v5.1。

## 2026-09-24｜文档重组：`WSS_PINN/` 拆成六块 + 已完成文档归档 · 已完成

- **本次主要修改**：用户要求整理 `docs/02-推进与变更/WSS_PINN/`（PINN 已不再推进，不再用一个目录装全部文档）。按现行工作拆成 [`00-V5设计与历史跟踪`](00-V5设计与历史跟踪/README.md)、[`01-X5D主线与新数据`](01-X5D主线与新数据/README.md)、[`02-时间建模`](02-时间建模/README.md)、[`03-周期量TAWSS_OSI`](03-周期量TAWSS_OSI/README.md)、[`04-数据处理与CFD`](04-数据处理与CFD/README.md)、[`05-部署工具`](05-部署工具/README.md) 六块，新建 [推进与变更导航](README.md)（含旧 → 新路径对照表）与每块 README。原 `WSS_PINN/README.md`（体域 PINN 路线史）和旧 `_archive/` 整体移到 [`_archive/WSS_PINN/`](_archive/WSS_PINN/README.md)，`WSS_PINN/` 目录已删除。
- **训练跟踪拆分**：`WSS_V5_训练实验跟踪.md`（3078 行）改名为 [历史卷](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)（§0–§29、§30 局部形态、§32）；§30 偏心与全周期时间线 + §31 → [时间建模_实验跟踪](02-时间建模/时间建模_实验跟踪.md)，§33 + §35 → [X5D主线_实验跟踪](01-X5D主线与新数据/X5D主线_实验跟踪.md)，§34 → [TAWSS_OSI_实验跟踪](03-周期量TAWSS_OSI/TAWSS_OSI_实验跟踪.md)。整节逐字搬、节号不变，原处留指针；行数对账 3079 = 保留 2579 + 迁出 500（§30 286、§31 65、§33 45、§34 100、§35 4）。
- **块内归档**（文首加状态标记，各块 `_archive/README.md` 记结论与证据路径）：局部形态矩阵 ⏸️（01）、体场全周期时间矩阵 🧊（02）、RCR 出口面积核查 🧊（04）、部署工具三份已落地方案 + P0 置信度门控审计 ✅（05）。
- **对应代码/文档**：脚本化改写 112 个文件共 523 处路径与相对链接（文档、`.cursor` 规则与技能、`PROJECT_PROFILE.md`、`training_wss_min/tools/*.py` 的 `TRACKER` / `NEW_MANIFEST` 常量与 docstring、`wss_deploy/`、`wss_v5/views/wall_cycle_v1.py` docstring、`wss_pinn/{README,AGENTS}.md`、移动目录内的 Slurm / shell / Python 脚本）；指向跟踪文档的引用按上下文 § 号分到对应块。**有意未改**：`wss_pinn/configs/splits/*.json`、`training_wss_min/experiments/**` 的 JSON 与脚本、日志、冻结副本里的溯源字符串（按导航页对照表换算）。改写前整目录备份在会话草稿区。
- **推进到实验步骤**：无实验变更；运行中的 v5.2 CV5 队列 15636 在冻结副本 `GNN_v52_frozen_20260923` 内运行、不读 docs，未受影响。
- **当前状态判断**：已完成。新增相对链接断链 0（剩余断链均为改动前已存在的历史快照链接）。今后实验结果写所属块的实验跟踪；新开一条线取全局下一节号（当前用到 §35）。

## 2026-09-24｜V5.1 老师 Temporal Transformer 配置矩阵 · 12/12 完成并验收

- 用户授权 node04 单卡、每臂先 seed1234，后续明确允许共享 A100 剩余显存并要求冻结；2×2（跨帧/对角 attention × 保留/屏蔽 query_x），cv3 共12训练。新入口 `training_wss_min/time_transformer.py`，新 schema v2、新 r3 配置/产物目录；旧训练路径不导入新模块。
- 提交：09-24 13:15，北京时间；node04 共享队列 PID `298315`，GPU0（UUID `GPU-e5c48469-8122-adf2-62f0-44d68a46fc22`），首个正式 run 已训练。原 12:20 的等待队列在尚无训练产物时停止并归档。从原冻结副本复制出 `GNN_time_transformer_frozen_20260924_r3_shared`，2719 文件核对仅队列脚本不同；新增 `--allow-shared-gpu --min-free-mib 12288`，模型及12配置不变，启动脚本也冻结。批次 `submission_shared.json` / `freeze_provenance_shared.json` / `queue_status.json` 可追踪，完训后自动严格验收与汇总。
- 完成：09-24 13:25:36，全部12次正式训练与评估验收通过，均为1800 steps / 60条history，G0最大偏差8.8818e-16 ln Pa；报告已生成。14:05复核393项源码/配置指纹与36项产物哈希一致，队列/训练进程均已退出。本条为进度验收，不新增效果或部署结论。
- 18 项测试 + 4 个 node04 两步冒烟通过，12 个配置与三折输入预检通过，峰值实际最大 ln 偏差 8.9e-16；四臂 120769 参数、初始权重 hash 相同，固定 context、原始 Pa 真值、真实 steps。9 旧源码/39 旧配置逐字兼容。
- 结果分析：四臂全周期Pa R²约0.499–0.502；跨帧attention在warm组−0.0015、raw组+0.0017，未见稳定主指标增益；raw的TAWSS有+0.0080三折同向信号。隐藏特征在full/diagonal下均三折改善峰时、时序相关、ln幅值误差，未转化为Pa净优势。与C-raw/T0仅作不同预算的历史参照；不替换默认方案。分析已写入时间矩阵P0.12.6及r3 `analysis_20260924/`，未新增训练或推理。
- 初期无后缀与 `_r2` 试运行因评估与对照合同不完整已失效，原产物保留；正式协议 `wss_time_transformer_v51_20260924_r3`，不混用旧数值。矩阵和状态见[时间矩阵 P0.12](02-时间建模/_archive/阶段文档_2026-09-30/WSS_V5_偏心与全周期时间实验矩阵_2026-09-18.md)及 r3 `queue_status.json`。

## 2026-09-23｜体场 EnSight 截面：点云插到该例 Fluent 体网格（nfaced 瞬态 case）+ 后处理包整改 · 已交付

- **用户要求**：体场是点云，在 EnSight 里切出来是散点、连不成面；该例有网格和 STL，要先插值，再在 EnSight 里拖参数看不同指标。原包里的几次尝试都不合适，要整改文件夹。
- **本次主要修改**：新工具 `training_wss_min/tools/build_ensight_cfd_mesh_case.py`（`tools/` 不在矩阵指纹内）。直接用 Fluent `.cas` 面表把解剖区 `blood` 的 677082 个单元组装成 EnSight `nfaced`（面统一朝外），V5 `@R.T` 配准到毫米。体内点云 = 单元体积中心（`cell_id_cas` 身份，逐点 ≤0.002 mm）；节点值取相邻单元的 1/距离 加权平均。壁面 85304 个节点（= 壁面点云，≤1e-5 mm）：速度 0，压力取壁面点云值。写成 EnSight Gold 瞬态 case：时间步 = 4 个相位（0.13/0.21/0.35/0.48 s），13 个节点变量，变量名不带相位——速度 CFD/VT0/VTB4 及误差、矢量；压力 CFD/PT0/PTB8 及误差。
- **产物**：`training_wss_min/experiments/volume_time_20260919/postview_median_phases_20260921/AAA__unruputer__SHEN_FANG_JIN/ensight/SHEN_FANG_JIN_flow.case`（783 MB，2103605 节点）、`preview/`（VTK 回读后切的横截面 Z=24.5 mm / 纵剖面 Y=10.5 mm）、`build_report.json`、`ens_checker.log`。
- **验收**：体积 369368 mm³ 与 Fluent 一致；677082/677082 个单元封闭且面朝向一致（逐棱检查；另用单位立方体测过翻面和缺面都能检出）。本机 EnSight 2023 R1 自带的 `ens_checker231` 通过、无警告；VTK 读回的节点值与写入逐位相同，单元类型为 POLYHEDRON。节点值平均回单元中心与原点值比较：CFD R² ≥0.996；预测三相位 ≥0.985，谷底 VT0 0.92 / VTB4 0.96。
- **踩坑**：瞬态 `constant per case` 的写法两边不兼容：ens_checker 要求数值另起一行，VTK/ParaView 只认同一行。已去掉常量，相位与时间的对应写进 README。本机 EnSight `-batch` 需要 bulk 许可证、跑不起来，所以 GUI 操作没有实测；`ens_checker231` 不需要许可证，可以作格式验收。VTK 切 Fluent 非平面多面体时个别单元会报 non-manifold 并跳过（本例 1 个），逐棱检查确认文件本身无误。
- **整改**：用户确认旧方案不要了，`sliceable_tets/`（四面体 / 六面体 / 点云 EnSight）与 `velocity/anatomy_volume/` 连同旧说明已删除（3.2 GB）；生成它们的旧工具 `build_sliceable_tets.py` / `build_ensight_hexgrid.py` / `build_ensight_pointcloud.py` / `export_anatomy_volume_vtu.py` 仍在仓库里。包 README、病例 `README_打开说明.md`、`打开文件清单.csv`（UTF-8 BOM）重写，`ensight/README.md` 是可随文件夹拷走的精简版。`manifest.json` / `verification.json` 未动。
- **对之前「EnSight 装不下 477 万四面体」的判断修正**：边界三角共 225244 个，而客户端只显示 48730 个单元。更像是 EnSight Fast display / 静态快速显示把模型抽成了点，而不是容量上限；GUI 未实测，README 里写了排查项。
- **当前状态判断**：只用于显示，正式 R²/MAE 仍在同点 CSV。预测截面里的放射状纹理在同一插值下的 CFD 中不存在，来自模型输出本身。

## 2026-09-24｜导师方案第二步：后段解冻版 TL-warm / TL-random · 已完成，预训练归因过、实用门不过

- **用户安排**：跑一次解冻版（三折 × seed 1234），C-raw 等其余种子不补；ssh 到 node04，不走 Slurm。
- **本次主要修改（新文件，旧代码零改动）**：`training_wss_min/time_adapter_finetune.py`（schema `time_adapter_finetune_v1`：复用 WSSMinDataset 采样合同与 P0.10 冻结锚；X5D 同构支路只解冻 `fp` + `local_wall_branch`，冻结部分恒 eval；训练前工程检查——donor 初始化支路对缓存 query_x 1.43e-6、零初始化 Δ≡0；训练开 TF32、检查与评估用默认数值后端）；`tools/prepare_wss_time_adapter_ft_v51.py`（含 `--pilot`）；`tools/report_time_adapter_ft.py`（门 U1–U4）；`tools/run_time_adapter_probe_queue.py` 增加 `finetune` 任务类型（未写 kind 的旧矩阵行为不变）；单测新增 finetune 配置校验（5 项通过）。
- **执行**：冻结副本 `GNN_timeadapter_ft_frozen_20260923`；试跑 2 epoch 与 1 步（在 `experiments/wss_time_adapter_ft_v51_20260923/pilot/`，首版试跑发现 TF32 让初始检查偏差 6.1e-3，已改为检查/评估用默认后端后降到 1.43e-6）；正式 6 任务 09-23 22:27 → 09-24 03:18，每 GPU 3 并发。
- **结果**：TL-warm 全周期 0.499、谷底 0.296；TL-random 0.468；U4（warm − random +0.03 三折同向）过，U1/U2/U3 不过（对 C-raw −0.012、对 T0 −0.018）。详见[时间矩阵 P0.11](02-时间建模/_archive/阶段文档_2026-09-30/WSS_V5_偏心与全周期时间实验矩阵_2026-09-18.md)、[V5 跟踪 §30.13](02-时间建模/时间建模_实验跟踪.md)。
- **当前状态判断**：解冻版 v5.1 收口；部署不变；v5.2 复议保留锚定框架。

## 2026-09-23｜导师方案（峰值预训练 → 时间适配器）v5.1 冻结表示探针 · 已完成，主臂不过门

- **用户安排**：不等 v5.2，先用 v5.1 cv3 验证可行性；每臂一个 seed；ssh 到 node04 跑，不走 Slurm。
- **本次主要修改（全部新文件，旧代码零改动 → 旧配置逐位不变）**：`training_wss_min/time_adapter_probe.py`（cache：同折冻结 X5D 供体 eval 前向，实例级包装取 `query_x` / `patch_context`；fit：锚定时间头 none / ridge / mlp，配置 schema `time_adapter_probe_v1`，未知键即报错）；`tools/prepare_wss_time_adapter_v51.py`（生成 3 个 cache + 18 个臂配置 + matrix.json）；`tools/run_time_adapter_probe_queue.py`（本机多 GPU 队列，启动记指纹、派发前复核）；`tools/report_time_adapter_probe.py`（预注册门 G0/F1/F2/A）；`tests/test_time_adapter_probe.py`（4 项通过）。
- **执行**：冻结副本 `GNN_timeadapter_frozen_20260923`（`make_frozen_copy_cycle.sh`，四个代码目录逐字一致）；node04 2×A100，21:03–21:20，21 个任务退出码全 0。v5.2 CV5 队列 15636 的 72 个指纹文件在启动前复核一致，未触碰。
- **结果**：G0 过（Tnull 复现 D2 差 3.5e-8，锚偏差 8.9e-16，A100 重算供体对 4090 已存预测 1.4e-6）。导师方案主臂 P-mlp 全周期 0.433 < T-null 0.462，F1/F2/A 全不过；病例向量 `patch_context` 是主要负项；对照臂 C-raw（原始几何 + 峰值标量）0.511、谷底 0.311，与 T0 持平且峰值保持 X5D。详见[时间矩阵 P0.10](02-时间建模/_archive/阶段文档_2026-09-30/WSS_V5_偏心与全周期时间实验矩阵_2026-09-18.md)、[V5 跟踪 §30.12](02-时间建模/时间建模_实验跟踪.md)。
- **当前状态判断**：冻结表示版收口；C-raw 属事后发现，补 seed / 独立协议前不作候选；部署不变。

## 2026-09-23｜体场 EnSight 点云 case · 已留下

- **本次**：按用户要求把原来的点云写成 EnSight point 零件，不生成三角面、四面体或六面体。`training_wss_min/tools/build_ensight_pointcloud.py`。
- **产物**：`.../sliceable_tets/ensight_points/`。`velocity_points.0.case` 体内 677082 点、76 个数组；`pressure_points.0.case` 壁面 85304 + 体内 677082、57 个数组。几何关键字只有 `point`，单元类型 VTK_VERTEX。`peak_speed_cfd` 与 `velocity/VT0/peak/interior_pointcloud.vtp` 逐点差为 0。回读 `readback_ok`。约 523 MB。
- **状态**：`ensight_hex/` 仍不是查看入口。正式指标仍在同点 CSV。

## 2026-09-23｜体场 EnSight 粗六面体不是 6 月点云路线 · 撤回推荐

- **用户更正**：`ensight_hex/lumen_hex.0.case` 和 `peak_slice.0.case` 把点云换成了六面体和三角面，点没了。6 月做法见 `tools/cfdpost_cloud_export/三条对比路线.md`：点云 CSV/VTP 保留；要光滑截面时在 Fluent 里用 Cloud of Points 插到原来的 `.cas` 体网格再切（路线 B2），不是另做一套面片替换点云。
- **打开说明**：病例 README 与包 README 已去掉「EnSight 打开 lumen_hex」的推荐。体内点云仍是 `velocity/<臂>/<相位>/interior_pointcloud.vtp`。`ensight_hex/` 文件还在磁盘上，不作为查看入口。

## 2026-09-23｜体场 EnSight 截面改成 2.5 mm 六面体 · 已生成

- **原因**：`sliceable_tets/ensight/velocity_tets.0.case` 有 4774713 个四面体。EnSight Standard 2023 R1 状态栏为「48730 客户 / 4774713 服务器单元」，客户端装不下就抽成点，Clip 仍超过这个上限，所以切完还是点。`wss_deploy` 的截面是把离散点插到管腔轮廓内的格子上再填色，不是打开这份细网格。
- **本次主要修改**：新工具 `training_wss_min/tools/build_ensight_hexgrid.py`。壁面节点（速度 0）∪ 体内点云，反距离（k=8）插到 2.5 mm 六面体，质心在 `aligned_geometry.stl` 外的丢掉。
- **产物**：`.../AAA__unruputer__SHEN_FANG_JIN/sliceable_tets/ensight_hex/`。`lumen_hex.0.case`（hexa8，23959 个，节点 29320，外表面 10030 个四边形，25 MB vtu 同内容）把速度和压力放在一起；`peak_slice.0.case` 是同一预览平面上已经切好的 3608 个三角形。回读 `readback_ok`。格子体积 374359 mm³，对 Fluent 腔 369368 mm³ 约 +1.4%。最近样本距离 p50 0.68 mm、p95 1.80 mm。
- **当前状态判断**：只用于显示。EnSight 打开 `lumen_hex.0.case` 再 Clip；细四面体 case 仍留给 ParaView。正式 R²/MAE 仍在同点 CSV。

## 2026-09-23｜时间适配器 P0.9 评审意见吸收 · 复议草案修订

- **修订要点**：把峰值恒等限定为同折冻结 checkpoint；以同折 T-null(B_scale) 为锚定基底；D7 明确需训练小头且失败不证伪全部峰值表示；`TL-warm` / `TL-random` 共用冻结峰值锚，只改变时间几何支路初始化；T0 用作实用性能对照。标量输出不宣称回流方向或 OSI；IND 患者重叠，只作新单元补充。撤销旧“理想 T0 为性能上限”、v5.1 阈值和预算可直接移植的说法。
- **状态**：仅修订[时间矩阵 P0.9](02-时间建模/_archive/阶段文档_2026-09-30/WSS_V5_偏心与全周期时间实验矩阵_2026-09-18.md)与[V5 跟踪 §30.11](02-时间建模/时间建模_实验跟踪.md)。仍等待 v5.2 数据补充实验完成后由用户复议，未训练、未新增评估、未改代码或部署。

## 2026-09-23｜峰值 X5D → 全周期时间适配器 · 后续机会登记，暂缓

- **用户安排**：先等待数据补充实验（v5.2 IND + CV5 基线重训）完成，再考虑迁移学习路线；仅记候选，不自动启动。
- **记录内容**：确认旧 T0/TB8/TB16 为随机初始化、未加载峰值 checkpoint；保存独立时间编码与几何特征融合、先冻结后部分解冻、峰值保护、归一化转换及预训练患者隔离等复议要点。未来分区和判据依补充数据结果另定，不固定旧 cv3/test34。
- **入口**：[偏心与全周期时间矩阵 P0.8](02-时间建模/_archive/阶段文档_2026-09-30/WSS_V5_偏心与全周期时间实验矩阵_2026-09-18.md)；[V5 训练跟踪 §30.11](02-时间建模/时间建模_实验跟踪.md)。本次仅改文档，未新增训练、评估或部署改动。

## 2026-09-23｜导师讲解三图 · 图内文字与图注改为英文

- **本次主要修改**：`plot_teacher_phenomenon.py`、`plot_why_decel_trough.py`、`plot_teacher_womersley.py` 的标题、轴、图例、标注和脚注改为英文并重画。`build_teacher_page.py` 的三张图 alt / figcaption 同步为英文。讲解正文仍是中文。未重训、未改指标。
- **对应代码/文档**：脚本在 `training_wss_min/experiments/wss_time_ecc_20260918/analysis_20260921/`；投屏图在 `docs/03-汇报材料/WSS_全周期_减速期与谷底为何难预测_导师讲解材料_2026-09-22_图/`。Womersley 首个 τ<0 帧仍是 8 mm 帧 4（Q/Qmax 0.134）、4 mm 帧 35（0.431）。
- **当前状态判断**：只换图面语言，读数不变。

## 2026-09-22｜学习曲线 R²_cb 图 · 已出图

- **本次主要修改**：由已有 `offline/analyze_learning_curve.json` 画折外合并 136 例物理 R²_cb（ckpt best）随训练比例 25/50/75/100% 的曲线。三条线为全部病例均值±样本 SD、jet（真值 p99>40 Pa）、非 jet；灰色散点为 seed 7 / 1234 / 2025。未重训、未重评。
- **对应代码/文档**：`training_wss_min/experiments/wss_learning_curve_20260920/plot_learning_curve.py`；图 `.../learning_curve_r2cb.png`。
- **读数**：25%→100% 全部病例 R²_cb 0.647→0.713；jet 0.584→0.655，全程低于非 jet。75%→100% 每翻倍斜率仍约 +0.035，与既有预注册读法一致（数据量仍是瓶颈）。
- **当前状态判断**：只复述已有汇总，不新增结论。

## 2026-09-22｜体场连续截面 v2：壁面∪体内点云 Delaunay 四面体（点云→可切体） · 已交付

- **背景**：用户要在后处理软件里切压力/速度截面看腔内分布；`interior_pointcloud.vtp` 是散点，`anatomy_volume.vtu`（CONVEX_POINT_SET）切面三角形杂乱、579 MB、依赖 `.cas`。核对：对 `anatomy_volume.vtu` 先 CellData→PointData 再 Slice 其实也能出光滑填色，"不好看"主要来自按 Cell Data 着色 + 逐单元 Delaunay 切割；但该路子部署时没有 Fluent 网格，不通用。
- **本次主要修改**：新工具 `training_wss_min/tools/build_sliceable_tets.py`：CFD 壁面节点 85304（速度置 0 = 刚壁无滑移；压力用 `full_pointcloud.vtp` 壁面真值）∪ 体内单元中心 677082 → scipy Delaunay 5113592 四面体 → `aligned_geometry.stl` 有向距离剔除质心在腔外的 338534 个（凸包跨凹陷/髂支间隙）+ 退化 356 + 长边（>2×p95 间距）10 点复核 11 → 4774713 线性四面体；所有臂×相位数组挂 Point Data（float32）；同一网格写 EnSight Gold（per node）并回读核验。STL 法向朝内，工具按体内点样本自动判符号。全程 326 s。
- **产物**：`training_wss_min/experiments/volume_time_20260919/postview_median_phases_20260921/AAA__unruputer__SHEN_FANG_JIN/sliceable_tets/`：`velocity_tets.vtu`（442 MB，77 数组）、`pressure_tets.vtu`（255 MB，57 数组）、`lumen_tets_geometry.vtu`（60 MB）、`ensight/velocity_tets.0.case` / `pressure_tets.0.case`、`plots/preview_slice_{accel,peak,decel,trough}.png`、`paraview_slice_demo.py`、`build_report.json`。病例 README、包 README、`打开文件清单.csv` 已补。
- **验收**：单元全部 VTK_TETRA；体内点数组与源点云逐点 |Δ|=0，壁面速度 0；无未用点；腔内四面体总体积 369330 mm³ 对 Fluent anatomy `blood` 体积 369368 mm³（`wss_v5.mesh_topology.load_mesh`）差 0.01%；峰值面 Slice 0.8 s（原多面体 2.2 s）；EnSight 两个 case 回读 `readback_ok`。
- **口径**：只用于显示，正式 R²/MAE 仍在同点 CSV。壁面 u=0 是边界条件不是预测。预览里 PT0/PTB8 压力预测整体偏红（全点云均值 −13.5 / −57.5 Pa 对 CFD −47.0 Pa），是模型偏置，非插值产物。
- **打开**：ParaView Open `velocity_tets.vtu` → Slice → 着色 `VT0_peak_speed_pred`（Point Data）；透明外壳 Extract Surface + Opacity 0.15；截面矢量 Slice→Glyph。EnSight 打开 `ensight/velocity_tets.0.case` → Clip → `VT0_peak_speed_pred_n`。
- **环境限制**：本机无 X / pvpython / OSMesa，预览只能 matplotlib gouraud tripcolor（等价于 ParaView 的 Point Data 线性插值着色）。

## 2026-09-22｜体场速度连续截面：anatomy blood VTU / EnSight Gold · 已交付

- **本次主要修改**：点云 VTP 没有体单元，EnSight Clip 只能看到散点。为同一例 `AAA/unruputer/SHEN_FANG_JIN` 从 Fluent `.cas.gz` 取出 anatomy 区 `blood`（677082 cell，不含 blood1–5），做 V5 `@R.T` 毫米配准，按 `cell_id_cas` 把已导出的 VT0/VTB4 四相位速度挂到 CellData。脚本 `training_wss_min/experiments/volume_time_20260919/export_anatomy_volume_vtu.py`。未用旧 `build_sliceable_volume.py`（无 R、会合并延长段、丢掉向量）、未 Delaunay、未回插壁面。
- **产物**：`.../postview_median_phases_20260921/AAA__unruputer__SHEN_FANG_JIN/velocity/anatomy_volume/anatomy_volume.vtu`（553 MB）与 `ensight/anatomy_volume.0.case`。Slice 预览 `plots/fig_VT0_peak_slice_speed.png`（231163 填充多边形）。
- **验收**：677082 非零体单元；88 个 Cell 数组含三分量速度；`VT0_peak_speed_cfd` 与点云 |Δ|=0。单元类型 hex 8941 + CONVEX_POINT_SET 668141（Fluent 多面体）。顶点平均中心相对 Fluent 体积心中位偏移 0.58 mm，是几何定义差，不是错单元。
- **打开**：EnSight 打开 `.case` → Clip/Plane → 着色 `VT0_peak_speed_cfd`（per element）；要光滑再插到 node。不要 Glyph 旧 VTP。

## 2026-09-21｜体场时间线中位病例四相位后处理包 · 已交付

- **本次主要修改**：按 WSS `postview_median_phases_20260920` 同一例 `AAA/unruputer/SHEN_FANG_JIN`（cv3 fold2）、同一套四相位帧，为 `volume_time_20260919` 导出 ParaView 包。速度只写体内点云（含向量，不回插壁面）；压力同时保留 Gaussian/native 壁面与体内点云。脚本 `training_wss_min/experiments/volume_time_20260919/export_phase_postview.py`；产物 `.../postview_median_phases_20260921/`。
- **臂与 checkpoint**：压力 PT0/PTB8、速度 VT0/VTB4，均为 fold2 `ckpt_best`。未按体场 R² 重选病例。test34 未用。
- **相位帧**：加速 1146、峰值 1162、减速 1190、谷底 1216。全点云 query（壁面 85304 + 体内 677082）。
- **验收**：48/48 VTP 核验通过；Gaussian 覆盖率 100%（127985 顶点 / 255672 三角）；峰值同点 R² 与官方 CSV |Δ|<1e-8。压力 selfmax 分母来自该帧 wall∪interior；速度 selfmax 用体内 `|u|`。
- **本例同点 R²（accel/peak/decel/trough）**：PT0 0.954/0.654/0.718/−0.143；PTB8 0.985/0.602/0.859/0.533；VT0 0.798/0.828/0.578/−0.154；VTB4 0.765/0.770/0.556/0.041。谷底负 R² 是该例该帧的同点结果，不是插值产物。

## 2026-09-21｜X5D 与 WSSNet 的空间 Pearson 相关性对照

- 仅复算已有 `ckpt_best` 预测缓存：峰值WSS的Pa空间逐病例Pearson r，均值±样本SD。test34五seed Pa均值集成 **0.8975±0.0438**；cv3合并136例折外、每折seed1234 **0.8665±0.0480**。原R²_cb及既有拟合R²核对一致，无训练/模型推理/重选模。
- 论文WSSNet的0.92±0.05来自6例×72帧且输入近壁速度；本项目仅峰值单帧、几何与解析先验输入，不能直接排名。保留test34/CV开发暴露及聚合口径边界。
- [报告与复现产物](../../training_wss_min/experiments/wss_v51_wave1_20260916/analysis_pearson_20260921/README.md)，回填V5跟踪§28.10。

## 2026-09-20｜体场时间臂周期重评补逐帧 NMAE：作业 15286 完成

- **本次主要修改**：`VolumeTimeMetrics` 增加 pooled `frame_nmae_range`（子采样点 MAE/(true_max−true_min)，与 `field.nmae_range` 同定义）。作业 15286（master 4×4090，16:48–17:29，40 min）对 PT0/PTB8/VT0/VTB4 × 三折 `ckpt_best` 周期重评，产物 `training_wss_min/experiments/volume_time_20260919/cycle_nmae_20260920/`，未覆盖原 `metrics.json`。
- **三折均值（窗 NMAE=子采样；峰值帧 NMAE=全点云）**：压力 PT0/PTB8 全周期 NMAE 0.0228/0.0177，峰值帧 0.0129/0.0146；速度 VT0/VTB4 全周期 0.0249/0.0273，峰值帧 0.0248/0.0292。R²_cb 与原评估一致。
- **口径**：T-null/peak_freeze 的 NMAE 仍来自 A0 子采样解析重建；训练臂峰值帧列与 WSS 表一样用全点云 `field.nmae_range`。

## 2026-09-20｜WSS 时间线中位病例四相位 Gaussian 后处理包 · 已交付

- **本次主要修改**：按 postview-surface-viz 为 `wss_time_ecc_20260918` 导出 T-null / T0 / TB8 同一例中位病例、四个代表帧的 Gaussian 壁面 VTP。脚本 `training_wss_min/experiments/wss_time_ecc_20260918/export_phase_postview.py`；产物 `.../postview_median_phases_20260920/`。
- **选例**：cv3 留出折 136 例，按 T0 `ckpt_best` 峰值帧 `physical_overall_r2` 距中位数最近 → `AAA/unruputer/SHEN_FANG_JIN`（fold2；0.7159 vs 中位 0.7156）。test34 未用。
- **相位帧**：加速 1146（Q=0.41）、峰值 1162、减速 1190（Q=0.43）、谷底 1216（最低 Q）。T0/TB8 用 fold2 `ckpt_best` 全壁面推理；T-null 为 D2 B_scale。
- **验收**：12 份 `surface_gaussian.vtp` 均有三角面（255672）和 `wss_cfd`/`wss_pred`；覆盖率 100%；T0 峰值同点 R² 与 CSV 差 7e-10。正式 R² 只在同点 CSV，不在面片上重算。
- **同点 R²（本例）**：加速 0.66/0.78/0.81，峰值 0.74/0.72/0.69，减速 0.50/0.56/0.43，谷底 −0.09/0.36/0.30（T-null/T0/TB8）。谷底 T-null 为负与波形拉伸预期一致。

## 2026-09-20｜WANG / ZHAO CFD 来源审计：数据链通过，RCR 时序与求解收敛尚不能放行

- 应用户要求，先核查 `ILO/WANG_JIN_MING-0/before`、`ILO/ZHAO_CHANG_SHAN-0/before` 的 CFD 来源；未启动 CFD、GPU 训练或新增模型评估。报告与可复核脚本：[审计产物](../../outputs/wss_cfd_hardcases_audit_20260920/README.md)。
- 两例各 81 帧原始壁面导出与当前 v5.1 标签逐点匹配，数据链与出口面积核查通过。当前日志的峰值步 continuity 为 WANG `6.6531e-5`（60 次上限）、ZHAO `3.8200e-3`（20 次上限）。
- 共同 RCR 疑点：8 个出口无回流样本的压力日志贴合“滞后流量、R1 动态项消失”递推（RMS `0.00076–0.00973 Pa`），与标准三元递推（`48–353 Pa`）不符。具体 UDS 历史层机制及对 WSS 的影响量尚待 trace / 隔离对照，不能将两例定性为纯模型难例，也不能宣称 CFD 已证为主因。
- 网格/时间步独立性未验证；WANG 重跑前后 WSS P99/max 未完成原始场核查，不引用摘要中的稳定性判断。详见 V5 跟踪 §32。

## 2026-09-20｜体场全周期逐帧可视化：自相似性一图分开压力与速度，PT0 的 best/last 差是单折失稳

- **本次主要修改**：只读已有产物出图，无训练无推理。新目录 `training_wss_min/experiments/volume_time_20260919/analysis_20260920/`（`plot_cycle_r2_volume.py` + fig1–6 + `frame_r2_source.csv` + `phase_summary.json` + README），与 WSS 线 `wss_time_ecc_20260918/analysis_20260919/` 同一套口径，曲线取各臂 `cycle.frame_r2cb` 与 A3 逐帧基线（同一套固定子采样点，三折均值±标准差）。
- **fig5（体场特有）**：共同时间缩放 r(t) = `T-null − peak_freeze`，压力在减速段/谷底最多买到 **+1.9** R²，速度**全程贴 0、谷底为负**。速度场周期内近乎自相似（冻住峰值帧 0.553，T-null 仅 0.557）——这是速度三个时间臂拿不到增量的根因，补 K 无用。
- **分相位读数（best，三折均值）**：压力 T-null/PT0/PTB8 加速 0.702/0.914/0.925、峰值窗 0.939/0.747/0.672、减速 0.238/0.436/0.363、谷底 0.309/0.497/0.583、全周期 0.437/0.477/0.539；速度 T-null/VT0/VTB4 峰值窗 0.972/0.764/0.713、谷底 0.118/0.280/0.237、全周期 0.557/0.559/0.506。注意 T-null/peak_freeze 用真值峰值场，峰值帧 R²=1 是构造的。
- **新结论**：① PTB8 全周期（0.539）其实高于 PT0（0.477）、谷底也更好，时间基头「把周期学像」是有效的，No-Go 是**任务取舍**（峰值帧 −0.051）不是学不动；② 速度只在最深谷底（步 1216 附近）相对 T-null 有 +0.4 尖峰，被峰值窗和减速段抵消；③ **PT0 的 best/last 全周期差 0.056 全部来自 fold0**（0.401 vs 0.565，fold1/fold2 逐帧几乎重合），fold0 的 EMA 选点在谷底塌掉（最差帧 −0.936@49、10 帧为负，`last` 只有 3 帧为负）——修正 09-19 条的措辞：不是 EMA 选模在体场普遍不可靠，而是 PT0 在 fold0 训练末段失稳。
- **回填**：跟踪 §31.4、矩阵文档 §9，Go/No-Go 不变。下一步先查 fold0 的训练末段不稳定，再谈换选模。

## 2026-09-19｜体场（压力/速度）全周期时间矩阵：24 臂完训完评，速度全线 No-Go、压力仅 PT0(last) 有条件过门

- **起因**：导师要求把 §30 的「融入 t」在压力和速度上各做一遍；用户拍板每臂只做 seed 1234、显卡并行。底座由 X5D_v51 改为体场原生 PF6/VF6（X5D 的 27D 含壁面专属 sidecar，内部单元上没有定义）。矩阵文档 [WSS_V5_体场压力与速度全周期时间实验矩阵_2026-09-19.md](02-时间建模/_archive/WSS_V5_体场压力与速度全周期时间实验矩阵_2026-09-19.md)，跟踪 §31。
- **本次主要修改（配置驱动、旧配置逐位不变）**：`dataset.VolumeFrameSource`（`case.h5` 懒读 81 帧体场标签 + 每例 sidecar，压力逐帧减 `p_volume_mean`、速度按 `transform_rotation` 旋转）；体场目标放开 `timesteps=random_frame/time_basis`；`frame_stats` 泛化到线性空间；`config` 新增 `volume_time_sidecar_root`/`volume_h5_root`；`evaluate._evaluate_partition_volume_time`（峰值帧全云 + 周期指标固定子采样）；`time_basis` 多通道化（速度三分量共用基向量）+ `reconstruct_frame`；共享口径收进新模块 `training_wss_min/volume_time.py`。新工具 `tools/prepare_volume_time.py`、`tools/update_volume_time_xlsx.py`、`offline/{a0_scan,a1_frame_stats,a2_time_basis,a3_tnull,report}.py`。
- **回归验收**：① `X5D_v51_s1234` 重评 5622 个数值叶子最大差 3.3e-5（作业 15114）；② 旧 `PF6_s1234`/`VF6_s1234` 依赖的旧 v5 视图已随 v5.1 切换删除、无法重评，改与改动前冻结树 `GNN_time_frozen_20260918` 对拍体场峰值帧数据路径，压力/速度各 4 例、各 138 数组逐字节相同。
- **阶段 0（作业 15112/15113）**：A0 合同 170 例零错误；`p_volume_mean` 周期 DC 摆幅 2290 Pa ≫ 峰值帧空间信号 359 Pa → 逐帧去 DC 必须。**K\*(压力)=8、K\*(速度)=4**。T-null 全周期压力 0.4373 / 速度 0.5569；**速度 `peak_freeze`（峰值帧冻住、不缩放）就有 0.5463**。
- **阶段 2（预检 15122 → 队列 15123，master 4×4090，13:37–18:47，墙钟 5.16 h）**：24 训练 + 42 评估退出码全 0。三折均值（峰值帧对同折折底座 / 全周期对 T-null）：PT0 best 0.7730(+0.029)/0.4771(+0.040)、PT0 last 0.7804(+0.037)/0.5335(+0.096, 3/3)；PTB8 0.6932(−0.051)/0.5392(+0.102)；PTB16 0.6816(−0.063)/0.5395(+0.102)；VT0 best 0.7668(−0.011)/0.5589(+0.002)；VTB4 0.7158(−0.062)/0.5061(−0.051)；VTB8 0.7174(−0.061)/0.5188(−0.038)。
- **结论**：压力 TB8/TB16 No-Go（峰值三折全掉 0.05–0.06，与 §30 WSS TB 同一失败形态；TB16 对 TB8 无增益）。PT0 只有 `last` 过两门，但过门依赖 checkpoint（best/last 全周期差 0.056，说明 EMA(train_loss) 在体场逐帧目标上选模不可靠），峰值「提升」几乎全来自底座异常弱的 fold1，最差帧仍为负 → 不改部署。速度全线 No-Go：VT0 与 T-null 打平、TB 比 T-null 还差，结合 `peak_freeze` ≈ T-null 说明速度场周期内近乎自相似，全周期 R²_cb 对速度区分度不足，要谈周期得先换指标而不是加 K。
- **落账**：工作簿 `速度与压力实验矩阵` 组 `体场全周期时间阶段2｜2026-09-19（…）` **98–139 行**（42 行，6–125 列为峰值帧读数，周期指标在备注）；本批是 v5.1 + cv3 留出折，与波 2 体场行（旧 v5 train138/test34）不可直接比。
- **下一步（未启动）**：唯一有证据支持的是把 PT0 选模换成留出折周期指标或固定 epoch 重跑三折；不补 seed、不读 test34、不为速度加 K。

## 2026-09-19｜时间线逐帧 R² 图：峰值高是因为谷底把全周期均值拉低

- **本次主要修改**：只读已有 `frame_curve` / D2 `frame_r2cb_pa`，出图 `training_wss_min/experiments/wss_time_ecc_20260918/analysis_20260919/`（fig1–4）。
- **读数**：峰值帧 0.713/0.697/0.609（T-null/T0/TB8），全周期 0.461/0.517/0.450；谷底 20 帧 0.22/0.32/0.27。
- **结论**：全周期低主要是减速+谷底，不是峰值本身差。底座只训峰值能解释 T-null；T0/TB 另有「低流量场与峰值场不同构」的问题。跟踪 §30.4。

## 2026-09-19｜时间线逐帧 R² 图：峰值高、全周期被谷底拉低

- 只读作图，无新训练。T-null / T0 / TB8 三折逐帧物理 R²_cb，图在 `training_wss_min/experiments/wss_time_ecc_20260918/analysis_20260919/`。
- 峰值期（Q≥0.8）三臂 0.70 / 0.68 / 0.60；谷底 20 帧 0.22 / 0.32 / 0.27。全周期是 81 帧平均，所以会被谷底拉低。
- T-null 的形状符合「峰值场 × 波形」；T0 在离峰段相对 T-null 为正；TB8 峰值掉 0.10 且不优于套波形。见跟踪 §30.4。

## 2026-09-18｜WSS V5 时间线阶段 2：作业 15071 完训完评，TB No-Go

- **本次主要修改**：无新代码。作业 15071（master 3×4090，11:43–20:08）九臂 T0/TB8/TB16 × cv3_v51 完训完评；按预注册 G2.1–G2.6 回填跟踪 §30.3、矩阵 P0.7、PointNet 工作簿组 `WSS偏心与全周期时间阶段2｜2026-09-18（T0/TB8/TB16×cv3）`。
- **关键指标（best，留出折 81 帧，test34 未用）**：T0 峰值/全周期/谷底/TAWSS = 0.6967 / 0.5166 / 0.3218 / 0.7164；TB8 = 0.6094 / 0.4497 / 0.2682 / 0.5953；TB16 = 0.6040 / 0.4490 / 0.2696 / 0.5932。T-null 全周期 0.4613；同折 X5D 峰值 0.7130。
- **门控**：T0 过 G2.1/G2.4，G2.3 均值过、fold2 掉 0.033 不过三折硬门。TB8/TB16 不过 G2.1（低于 T-null）、G2.2（对 T0 −0.067）、G2.3（峰值 −0.10）、G2.5（amp_err）；只过 G2.4。G2.6 本轮未算。last≈best。
- **结论**：时间基头 No-Go；不补 seed、不开 2b/I1、不补 T0-800。部署仍为峰值帧 X5D_v51。证据：`training_wss_min/experiments/wss_time_ecc_20260918/stage2_report_best.json`。

## 2026-09-18｜腾盘：删 CROWN 预处理 pkl、V5.0 快照、过期 PINN bundle

- **本次主要修改**：按用户授权清理可重建/已过期产物，**未动源库 `data_new/`**（仍 3.1T）。现行训练数据只留 v5.1。
- **CROWN**：删除 `private_preprocessed_raw_ascii_v1/pkl/`（约 565GB，merge+partial）与旧 `private_preprocessed/` pkl（9.2GB）。保留导出代码/配置、`audit/manifests/stats` 与论文 `CROWN_Dataset`。重导命令写在 `external_baselines/CROWN_Beihang/private_preprocessed_raw_ascii_v1/README.md`（`submit_export_crown_array.sh`，源为 `data_new/AG` ascii_in）。
- **V5.0**：删除 `data_wss_v5/anatomy_pointcloud_v5_20260906/` 与 `views/`（约 127GB）。保留 `anatomy_pointcloud_v5_1_20260916/`（103GB）与 `views_v5_1/`（23GB）。现行 X5D_v51 配置本来就指向 v5.1。
- **过期 PINN**：清空 `data_wss_pinn/` 训练 bundle（32GB）及 `outputs/wss_pinn/volume_uvwp_*` 体场产物与 slurm 日志（约 26GB）。**未删** V5 签收 `volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/`、历史 `runs/` checkpoint、`audits/`（含 09-01 full-wall 缓存）。
- **盘面**：`/public` 39T/3.0T（93%）→ 38T/3.7T（92%），约腾出 0.7T。
- **空壳目录（同日稍后）**：撤掉只剩 README 的 `data_wss_pinn/` 与 CROWN `private_preprocessed/`。说明并入 `wss_pinn/README.md` / `crown_beihang/README.md`。保留 `private_preprocessed_raw_ascii_v1/`（重导入口 + audit）。`data_new/` 仍未动。
- **当前状态判断**：v5.1 训练数据完整；CROWN 预处理可从源 ASCII 重导；PINN V4 训练数据不再本地可训，证据与权重仍在。

## 2026-09-18｜局部形态 Wave B 扇区 token · 作业 15046 完训入账

- **本次主要修改**：核实作业 15046 四臂完训完评（S4/S6/S8/S6R2，seed 1234，400/400 epoch，8/8 评估，退出码全 0，01:26:57–07:04:25），把数字回填跟踪 §30 与工作簿 455–458 行。未做机制列、未训练、未改底座。
- **对应代码/文档**：[训练跟踪 §30](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)、[局部形态矩阵 Wave B](01-X5D主线与新数据/_archive/WSS_V5_局部形态实验矩阵_2026-09-17.md)、[`experiments/wss_local_morph_sectors_20260918/results.md`](../../training_wss_min/experiments/wss_local_morph_sectors_20260918/results.md)、工作簿 [`WSS_PointNet实验矩阵与结果汇总last.xlsx`](../03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx)。
- **关键指标**（test34，ckpt_best，对照 = 同球无扇区 W06K32 0.7560）：S4 0.7634（+0.0074）、S6 0.7560（0）、S8 0.7718（+0.0158）、S6R2 0.7565（+0.0006）；归一化 Δ 全在 +0.0027～+0.0033。物理列非单调，不用于排名。
- **Go-NoGo**：待机制验证 — 判据来源是矩阵启动前冻结的 `reading_rule`（主判据 = 区间内残差 m=1..3 净份额，尚未算）。不宣布胜出、不进底座、不补三 seed。
- **下一步**：按 Wave A 同口径算四臂 m=1..3 净份额；相对 W06K32 下降 ≥0.02 才送确认波，否则 B2 收口。

## 2026-09-17｜V4 EMA 代表病例 WSS/速度/压力后处理

- **本次主要修改**：为历史臂 `V4-SP-PN-BC-PDE-EMA-s1234`（index 3，seed1234，`last_converged` 未收敛）导出 best/median/worst 汇报包。选例按 test35 全壁面派生 WSS 物理 R²；WSS 复用 09-01 full-wall 缓存，未重新校准。速度/压力在 master GPU2 剩余显存（约 19 GB）上按历史 BC 变换重推理三例，未训练、未动冻结跟踪表/xlsx。
- **对应代码/文档**：后处理 [V4_EMA/](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/V4_EMA/README.md)；散点 [WSS](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例拟合散点_20260915/figures/V4_EMA_scatter_fit.png) / [速度](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例拟合散点_20260915/figures/V4_EMA_speed_scatter_fit.png) / [压力](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例拟合散点_20260915/figures/V4_EMA_pressure_scatter_fit.png)。脚本 `prepare_v4_ema_postview.py`、`prepare_v4_ema_volume.py`。
- **选例与指标**：Best `AG/fast/RAN_QING_BO` WSS 0.260 / 速度 −0.170 / 压力 0.145，WSS a=0.211；Median `ILO/LI_YOU_ZHI-0/before` −0.040 / −0.614 / 0.011；Worst `AG/fast/LOU_YANG` −0.889 / −0.050 / 0.256。速度/压力是同一三例，不是独立排名。Gaussian 覆盖率 100%。test35 派生 WSS 均值 −0.105，20/35 负 R²。
- **当前状态判断**：汇报用 WSS/速度/压力图已齐。历史 V4 EMA 体场与派生 WSS 均弱，不能当现行 V5 对照的正面例子。未改冻结 WSS-min 跟踪表。

## 2026-09-17｜X5D_v51 后处理并入 09-14 目录并补 R² 分布

- **本次主要修改**：把日期目录 `启发式v2_X5D_v51_代表病例后处理_20260917` 并入 `启发式v2_代表病例后处理_20260914/X5D_v51/`（与 VF6_wss 同层，不用日期命名）；按既有协议从五 seed metrics 补 A_X5D_v51 病例 R² 分布。未训练、未推理。
- **对应代码/文档**：后处理 [X5D_v51/](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/X5D_v51/README.md)；R² 图 [A_X5D_v51_distribution.png](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_R2分布_20260914/A_X5D_v51_distribution.png)；脚本 `build_x5d_v51_distribution.py`。
- **关键指标**（test34，ckpt_best，五 seed 1234/7/2025/11/2026 病例 R² 均值，非集成场）：均值/中位/P10 = 0.7619 / 0.7612 / 0.6542；负 R² 0/34。Best `AG/fast/ZHANG_LIANG` 0.9043；Most `AG/slow/HE_SHU_ZHEN` 0.7503（箱 [0.70, 0.80) 15/34）；Worst `ILO/ZHANG_YONG_SHENG-0/before` 0.6021。后处理 Median 仍是 `AAA/ruputer/KANG_YONG` 0.7608，与橙色 Most 不同。五 seed 单次 R²_cb 均值 0.7548，不是部署集成 0.7749。
- **当前状态判断**：汇报后处理与 R² 分布已并到 09-14 目录；日期文件夹已删除。

## 2026-09-17｜X5D_v51 最好/中位/最差后处理、散点与模块图

- **本次主要修改**：用 v5.1 数据更新后已跑完的密度增广 X5D_v51 五 seed 缓存，导出 best/median/worst 壁面包，并补拟合散点与第 4 页网络模块图。不是正在训练的纵向几何 X5D_long，未推理。
- **对应代码/文档**：后处理已并入 [X5D_v51/](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/X5D_v51/README.md)；散点 [X5D_v51_scatter_fit.png](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例拟合散点_20260915/figures/X5D_v51_scatter_fit.png)；模块图 [slide-4.png](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_网络模块_20260913/slide-4.png)。
- **选例**：五 seed 逐例物理 R² 均值。Best `AG/fast/ZHANG_LIANG` 0.904 / s1234 0.884，a=0.926；Median `AAA/ruputer/KANG_YONG` 0.761 / s1234 0.778（与 ZHANG_HE_PING 近并列，该例 s7=0.614）；Worst `ILO/ZHANG_YONG_SHENG-0/before` 0.602 / s1234 0.610，a=0.611。
- **验收**：Gaussian 三例覆盖率 100%；独立核验 358 项通过。最差例 STL 更密，最近距离最大 1.32 mm，热点请对照 native 面。
- **当前状态判断**：汇报用图已齐；部署底座仍是 X5D_v51 五 seed 集成，本包图画的是 s1234 单次场。test34 已暴露。

## 2026-09-17｜v5.1 后续优化：折外局部误差与 cap 优先级

- 只读当前 v5.1 五 seed/test34 与三折136例预测，新增 [CPU 诊断与建议](../../training_wss_min/experiments/wss_v51_wave1_20260916/analysis_20260917/next_optimization.md)，回填 [训练跟踪 §28.9](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)。未训练、未模型推理。
- 折外4 mm区间内部log_z残差方差份额70.13%，区间均值轨迹R² 0.9306、局部型态0.5218；两例ILO合占该队列病例等权平方误差52.16%。建议cap先做患者分组cv3再补seed，主要新结构候选为保留局部空间结构的query patch，先诊断再试验；已有部署底座不变。
- 澄清协议版cap为盖面r³ Murray、旧分流解释上限不能移用；区分全局P90与每例P90后pool的high-WSS指标。8个run主指标/正式尾部指标从缓存复现，未把test34五seed与cv单seed差值直接解释为过拟合。

## 2026-09-16｜VELWSS2 vs X5：速度→WSS 偏差拆解 · 完成

- **本次主要修改**：只读 VELWSS2 / VF6_s1234 / X5 已有指标与同点预测，按距壁分层体内速度，解释派生 WSS 为何低于直接 WSS。未训练、未推理。
- **对应代码/文档**：`training_wss_min/experiments/vf6_velocity_to_wss_20260916/analysis_20260916/`（诊断 JSON、距壁分层图、速度–WSS 散点）；跟踪 [§27.5](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)。
- **结果摘要**：算子 0.965、半径 ±0.003 均非瓶颈。体内速度与 X5 逐例 Spearman 0.886，与派生 WSS 仅 0.484。距壁 <2.5 mm 占 84% 点、速度 R² 与核心几乎相同；0–0.5 mm 层 R² 降到 0.683、相对 MAE 0.40，与派生 WSS 相关升到 0.667，仍高于派生 WSS 本身。KANG_YONG 最内层速度偏高 35%，派生 WSS −0.217，而 X5 0.754、oracle 0.976。
- **当前状态判断**：差距来自预测速度的最内层剖面/导数与监督目标不对齐，不是冻结算子坏了。速度→WSS 仍不进主线。test34 已暴露。

## 2026-09-16｜VELWSS2 汇报图：R² 分布、拟合散点、三例 Gaussian 壁面 · 完成

- **本次主要修改**：用已完成的 VF6→冻结 Profile-Secant V3 派生 WSS（legacy 校准路径）出汇报图，不训练、不推理。选例按 seed 1234/7/2025 的逐例预测 R² 均值；Median 取距分布中位最近病例；图上场值均为 s1234。
- **对应代码/文档**：分布 [B_VF6_wss_distribution.png](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_R2分布_20260914/B_VF6_wss_distribution.png)；散点 [VF6_wss_scatter_fit.png](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例拟合散点_20260915/figures/VF6_wss_scatter_fit.png)；后处理包 [VF6_wss/](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/VF6_wss/)。脚本 `build_velwss2_distribution.py`、`prepare_velwss2_postview.py`、`verify_velwss2_postview.py`。跟踪 §27.4。
- **结果摘要**：Best RAN_QING_BO 0.756 / s1234 0.793，a=0.790；Median LU_FU_SHAN-0 0.488 / s1234 0.515；Worst KANG_YONG −0.217 / s1234 −0.221，拟合 R² 0.605 不能挽救负预测 R²。三例 Gaussian 覆盖率 100%，核验 346 项通过。中位例 ILO STL 比 CFD 壁面更密（119k vs 80k），热点请对照 native 面。
- **当前状态判断**：汇报图已齐；速度→WSS 仍不进主线。test34 已暴露。

## 2026-09-16｜VELWSS2：VF6 三 seed 速度→冻结 Profile-Secant V3→WSS 工具链

- 新增 `training_wss_min/tools/vf6_velocity_to_wss.py`（export / prepare / worker / assemble 四阶段，参数化到三 seed 与三种半径来源 legacy / legacy_exact / v5atlas；只新增 tools，不改根目录 `.py`）、`tools/report_vf6_velocity_to_wss.py`（独立 NumPy 复算 2,640 项、results.md/json、逐例 CSV、两张图、README）、`tools/update_vf6_velocity_to_wss_xlsx.py`（WSS实验矩阵 追加 6 行并核验历史单元格）、`cluster/vf6_velocity_to_wss{,_export}.slurm`；`tools/annotate_workbook_methods.py` 补 VELWSS2 词条。
- 集群坑：master 四张卡被 wss_v51 队列 14428 以 gres=gpu:4 占满，不带 gres 的 GPU 分区作业在 cgroup 下看不到 GPU（探针 14448 `nvidia-smi -L` No devices found，14438 秒退）；速度导出改在登录 shell 的 GPU 3 直接跑，工具用 `--allow-login-node` 显式开关并把执行方式/主机/GPU/空闲显存写进 reproduction JSON。用户同日提示 node04 的 2×A100 可用（`docs/00-规范与记录/集群node04使用要点.md`），下次优先。
- 半径核查：旧 `data_wss_min` bundle 的 `wall_local_radius` 在 unit_factor 尺度坐标上错位、VELWSS1 的最近邻旧映射恰好补偿（V5 半径与 legacy 相关 0.985、与"精确对应"只有 0.863）；后续算子一律直接读 V5 bundle 半径。
- 结果与判读：[训练跟踪 §27](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)；实验目录 `training_wss_min/experiments/vf6_velocity_to_wss_20260916/`（17 GB，其中 chunks 11 GB 为分片诊断数组，结果固化后可删）。

## 2026-09-15｜核对 X5X11 / PF6 / VF6 查看版无 CFD 延长段

- **核查**：对三套包全部病例，用母库 `interfaces` 的最近切口平面（法向指向 blood1–blood5 延长段）判断点是否越过解剖切面。
- **结果**：壁面/体点最近平面距离最大值约 0 / −0.05～−0.13 mm，没有点落到延长段一侧。SUN 的 STL 包围盒与解剖壁面一致。图上的直管是解剖入口和髂动脉，不是网格拉伸段。
- **当前状态判断**：无需从这三套 VTP 删点。未训练、未推理。

## 2026-09-15｜VF6 管内点云去掉近壁层，不再半剖

- **本次主要修改**：按用户要求，查看版每个病例只保留一个 VTP：`dist_to_wall_mm > 1 mm` 的体单元中心，不做半剖/薄层。切开交给 ParaView Clip。
- **对应代码/文档**：[VF6 管内点云](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/VF6_体内点云查看版_20260915/README.md)。Best/Median/Worst 点数 210412 / 146581 / 263396。
- **验收**：36 项核验通过；VTP 无三角面、无壁面 query、最小距壁 > 1 mm，selfmax 分母仍为完整域。未训练、未推理。
- **当前状态判断**：近壁 1 mm 层已从文件里拿掉；完整外形仍在，内部用后处理切开查看。

## 2026-09-15｜启发式 v2 代表病例 ax+b 拟合散点

- **本次主要修改**：用已交付 9 例同点 CSV 补画 X5X11 / VF6 / PF6 的最好/中位/最差 Pred vs CFD 散点，红线 `pred = a·CFD + b`，灰虚线 y=x。未训练、未推理。
- **对应代码/文档**：[拟合散点图包](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例拟合散点_20260915/README.md)；数据来自 [09-14 代表病例包](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/README.md)。
- **结果摘要**：预测 R² 与包内 `visualization_metrics.r2` 逐例对齐。X5X11/VF6 最好例斜率 0.95/0.97；最差例斜率 0.45/0.48，拟合 R² 几乎不抬预测 R²。PF6 最好例接近恒等；中位例拟合 R² 0.934 但 a=0.693；最差例预测/拟合 0.349/0.387 一起掉。三图对齐 PASS。
- **当前状态判断**：现行三模型此前只有空间三联，没有 ax+b 散点；本包补上。拟合 R² 不能代替预测 R²。test34 已暴露。

## 2026-09-15｜启发式 v2 训练 loss history 图

- **本次主要修改**：读取 wave-2 已保存 `history.jsonl`，为汇报重绘 PF6/VF6 合图与 X5X11 三 seed 曲线。未训练、未评估、未改 checkpoint。
- **对应代码/文档**：[训练 loss 图包](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_训练loss_20260915/README.md)；源 run 为 `training_wss_min/runs/wss_local_wave2_20260912/{PF6_s1234,VF6_s1234,X5X11_s1234,X5X11_s7,X5X11_s2025}`。
- **推进到实验步骤**：只读日志出图。
- **当前状态判断**：PF6/VF6/X5X11 均写满 400 轮、无逐轮 val。ckpt_best 按 train_loss 最低选取：PF6 epoch 375（0.0283）、VF6 epoch 374（0.1208）、X5X11 三 seed 379/379/393。X5X11 的曲线是 total loss（base MSE+0.2×pinball）。两图对齐/文字/碰撞均 PASS。test34 已暴露，图注 R²_cb 未参与选模。

## 2026-09-15｜X5D 后续优化分析与同点残差诊断

- **结果**：复用X5/X5D三seed已保存的test34预测，六个单模型Pa R²复现、缓存行与真值对齐通过。X5D在log_z空间4 mm分支区间内的残差方差份额中位数69.56%、型态R²0.5999，基本同X5；Pa均值集成R²0.7382、top10幅值比0.7031，支持继续处理局部型态与高值收缩。
- **建议**：优先固定毫米范围的全壁面patch（覆盖采样＋有效mask），其次密度视图一致性；精度候选为X5D条件残差细化与保留周向位置的局部结构。均为建议，未启动训练/推理；已存在的wave6b补seed11/2026不重复启动。
- **口径核查**：区分Pa/log均值集成、三/五seed、逐例/全队列p90区域；同一Pa集成预测的归一化R²为0.8718。明确STL点采样间距不等于分割分辨率，0.5 mm未证明局部处处覆盖，毫米邻域不是已确认的唯一机制。
- **产物**：[分析报告与复现数据](../../training_wss_min/experiments/wss_local_wave6_20260915/analysis_20260915/X5D_next_directions.md)，回填[训练跟踪§22.4](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)。独立审阅确认指标与分解正确；test34开发暴露边界保留。

## 2026-09-15｜VF6 全点排壁核查与内部速度查看补充

- **核查结果**：针对整管速度图近乎全蓝的反馈，对原 Best/Median/Worst 的 1,257,254 个点逐点核对 query、母库 cell centre、坐标/向量变换，并用原生 CFD 全部壁面三角重新求最近距离。原速度 VTP 无壁面节点，也无精确零速；三例最小距壁约 0.012141/0.012159/0.011605 mm。图像外层为靠近壁面的低速体内点，不按速度值删除。
- **查看交付**：新增 [VF6 体内点云查看版](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/VF6_体内点云查看版_20260915/README.md)，各例完整/几何半剖/1 mm 薄层共 9 VTP 和 9 同点 CSV，6 张原量/selfmax 对照图，附相机/平面合同与 ParaView 辅助脚本。子集只依据几何，不插值、不补壁面点；原完整场、正式指标、角色及 selfmax 原分母保留。
- **验收**：独立核验 617 项、0 失败；1,749,941 行字段与 38 文件 SHA 匹配，6 图面板对齐/文字/碰撞检查通过。源审计发现 SUN 两个远离壁面的点，其历史近邻候选距壁特征最多高估 0.041908 mm，单列警告，不影响排壁结论，未修改源特征。
- **技能改进**：postview 的 SKILL、交付合同、V5 入口补充体内身份和真实壁面距离的独立检查，不能只看 point_kind，也不能按单元/节点整数 ID 交集或低速删点；半剖/slab 必须披露几何谓词与显示子集点数。没有训练、推理或全测试集重评。

## 2026-09-15｜安装 nature-skills，重绘模块与启发式汇报图

- **技能与交付**：按用户要求安装 `Yuan1z0825/nature-skills` 的 19 个功能技能及 `nature-shared`，共 20 个完整目录，固定提交 `9ea7330a17813a15421fe843778a776c258b9001`。新增 [Nature 风格优化包](../03-汇报材料/启发式实验汇报_2026-09-13/启发式Nature风格优化_20260915/README.md)：22 页汇报 PPT/PDF、4 页模块 PPT/PDF、22 组 PNG/PDF/SVG，附讲解备注、绘图源码和待补材料清单。
- **内容优化**：主干、X5、X11、PF6/VF6 分成四张信息流图；模块收益与设计意图分开，X11 的 R²/高值收益与 MAE/热点 IoU 代价同时展示。补齐全病例横向 R² 与直方图、9 例 Best/Median/Worst 矩阵、selfmax 附录、10 项 raw loss 及历史梯度冲突；旧 R5、V4 与当前 V5 的证据边界保留。
- **核验**：22 个 PDF 文字/碰撞检查通过，0 FAIL/0 WARN；15 个多面板图对齐 PASS，7 个单图不适用。56 个来源 SHA 匹配，66 个导出齐全，22 页/4 页两套 PPT/PDF 内容与页数核对通过。源码审计的组会尺寸、300 dpi 等 WARN 已在随包 QA 说明中解释。
- **执行范围**：只安装技能和重绘已有结果；原病例 VTP/CSV、指标和旧图保留，无新训练、推理或全 test 重评。新版 VF6 点云页绘制全部体内点，不再沿用旧预览 30,000 点抽样；PPT 为高清图件页，修改模块文字/箭头使用 SVG/PDF 或 Python 源码。

## 2026-09-14｜后处理技能统一交付合同，9 例新增各自最大值归一化

- **技能更新**：更新 `.cursor/skills/postview-surface-viz/`（`.agents/skills/` 共用软链接），新增 `delivery-fields.md`。今后完整病例包默认按变量交付：WSS 原点云＋Gaussian/native 壁面，速度仅体内点云，压力完整 wall∪interior 点云＋wall-only Gaussian/native 壁面；保留同点 CSV、预览、映射报告、manifest、打开清单和核验报告。V5 与历史 V3/CROWN 都接入新合同，病例范围仍以用户请求为准。
- **字段与实现**：新增可复用 `tools/cfdpost_cloud_export/display_fields.py`，重导 [代表病例包](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/README.md) 全部 9 例（schema 3）。原物理量和误差保留；新增 `{wss|speed|pressure}_cfd_selfmax`、`_pred_selfmax`、归一化 signed/absolute error 和 3 个有效性 mask。最大值写入 VTP FieldData、manifest 和分母清单。
- **归一化口径**：Pred/Predmax、CFD/CFDmax 各自分母取该病例所选帧原始完整同点评估域，在点云/native/Gaussian 之间固定复用，插值后不重求 max。速度对模长归一化；相对压力保持 p_ref，使用带符号 max，保留负值，不换成 maxabs/min-max。selfmax 仅供分布对照，原始 R²/MAE 不变；零/非有限分母明确 NaN＋valid=0。
- **执行与验收**：只复用原缓存并本地导出/绘图，无新训练、推理或全 test 重评。技能格式与内部引用检查通过；包内 `verify_selected_postview.py` 的 **901 项检查全部通过**，含原场、坐标/拓扑、Gaussian、selfmax 公式/分母、21 份 CSV、FieldData、64 个源/输出/验证代码 SHA 与 16 个 helper 边界检查，结果见随包 `verification.json`。原物理量/selfmax 共 18 张三联图和 4 张总览；速度仅对点云绘图抽样，每例 30,000 点，完整 VTP 不减点。

## 2026-09-14｜代表病例补齐壁面 Gaussian 插值并修正压力点云坐标

- **修正内容**：此前代表病例包只有点云，遗漏 WSS/压力壁面插值；PF6 的旧索引还将壁面压力值挂到体内坐标。本次从原始 `query_idx` 对“壁面＋体内”坐标数组重新挂载，逐点核对真值与坐标，旧包文件归入标明“勿用于汇报”的审计目录。
- **最终交付**：沿用 X5X11/VF6/PF6 各 Best、Median、Worst 共 9 个 s1234/ckpt_best 缓存病例；X5X11 = 壁面点云＋Gaussian STL 面＋原生 CFD 面；VF6 按用户要求仅保留体内速度点云；PF6 = 壁面∪体内完整压力点云＋仅壁面 Gaussian STL 面＋原生 CFD 面。另附 6 张壁面三联图与总览，入口为移动后的 [代表病例打开说明](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/README.md)。
- **核验与口径**：Gaussian r=3 mm、sharpness=2、max_dist=3 mm，6 份壁面覆盖率均 100%；STL 倍率 1.0，经 V5 刚性变换后最近壁面点最大距离小于 3.69e-5 mm。逐点核对原缓存/母库、VTP 数组读回与 R²；压力保留原 p_ref，无偏置拟合。选例三 seed R² 均值与文件 s1234 原始同点 R² 分列；不以平滑后指标代替正式评估。
- **局部展示限制**：YAO_CUN_HONG / ZHANG_CHUN 分别有 12 / 16 个高斯邻域含局部三角图不连通点，诊断与权重写入 mapping report；使用原生壁面检查局部热点和平滑影响，不以 100% 覆盖率声称无跨壁混合或面积映射通过。
- **执行范围**：重写包内缓存导出脚本，复用 9 例既有预测并本地出图；无新训练、模型推理或全 test 重评。包内输出改用相对路径，源哈希和独立核验报告随包保留；独立核验 495 项全部通过，62 个源/输出 SHA 匹配。

## 2026-09-14｜启发式汇报材料归入同一目录

- **本次主要修改**：把 09-13/09-14 启发式提纲、参考 PPT、R² 分布、代表病例后处理、R4 结构图和 R5 体场图从 `docs/03-汇报材料/` 根目录收进 [启发式实验汇报_2026-09-13](../03-汇报材料/启发式实验汇报_2026-09-13/README.md)，避免汇报材料顶层过于分散。
- **对应代码/文档**：入口 README、`docs/README.md` 汇报导航、`_archive/WSS_PINN/README.md` 提纲链接、V5 训练跟踪体场图链接、`training_wss_min/tools/export_v5_volume_postview.py` 导出目录。
- **推进到实验步骤**：只改存放位置与引用路径，未训练、重评或重导图。
- **当前状态判断**：现行讲解入口仍是提纲 v2；旧顶层路径已失效。

## 2026-09-14｜启发式汇报：病例 R² 横向分布与选例素材

- 复用 X5X11、VF6、PF6 的 s1234/ckpt_best 预测缓存和同帧 bundle/volume 坐标，整理各模型独立 Best/Median/Worst 共 9 个 ParaView 点云 VTP；未重新训练或推理。交付目录：[代表病例后处理包](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_代表病例后处理_20260914/)。
- 仅读取现有 `eval/ckpt_best/metrics.json`、V4历史评估 JSON 和 velocity→WSS 审计结果，生成 12 组横向病例预测 R² 散点＋固定宽度直方图（PNG/PDF/SVG），覆盖 X5/X5X11、PF6/VF6、R5 诊断链和 V4 三目标。
- V5 三 seed 每例取 R² 算术均值，横须为 seed 间样本 SD；Most 按 0 锚定、宽 0.10 的最高频区间选取，Best/Worst 为真实极值，负 R² 保留。`summary.json` 保存源路径/SHA256、统计和分箱敏感性；`representative_cases.csv` 保存 36 条角色选例。
- 交付[分布补充 PPT/PDF](../03-汇报材料/启发式实验汇报_2026-09-13/启发式v2_R2分布_20260914/)，共 14 页。此批不调用训练、推理或后处理导出；病例云图仍需按 CSV 中的 run/checkpoint/seed 在 ParaView 补齐。V4 源文件名虽含 `last_converged`，实际记录为 epoch9999、`converged:false`，汇报中按历史未收敛诊断页标注。

## 2026-09-13｜V4稳态独立损失项补图

- 用户澄清后另交付[单个PNPP EMA两图](../03-汇报材料/V4汇报/V4_稳态梯度与损失_2026-09-12/individual_losses/single_pnpp_ema/README.md)：同轴u/v/w/p与同轴边界/PDE六项，均为raw；挑选与此前动态权重对应的臂，不按测试成绩筛选。

- 按用户要求新增10个独立损失项及动量组共11张英文图，每张含raw/weighted与两骨干对照，输出PNG/PDF/SVG与合订PDF。
- [图件与字段口径](../03-汇报材料/V4汇报/V4_稳态梯度与损失_2026-09-12/individual_losses/README.md)；读取历史8臂日志，逐epoch校验10项加权之和复原total。关闭项不作零残差展示，稳态不画RCR/时间项。无新训练、推理或测试集评估。

## 2026-09-13｜波 2 结案、波 3 提交与文档归档（指针）

- 波 2（`wss_local_wave2_20260912`，Slurm 14185）结案：X5+X11 三 seed 只改善尾部；F6 变体无增益；F6 接体场有效（压力 +0.035、速度 +0.023）。WSS 新底座 = X5（三 seed +0.069）。详见 [V5 训练实验跟踪 §20.4](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)；工作簿 WSS 表 372–377 行、速度与压力表 74–81 行（新工具 `training_wss_min/tools/update_wss_local_wave2_volume_xlsx.py`）；工作簿另新增「方法说明对照」页与列 B 人话说明（`tools/annotate_workbook_methods.py`）。
- 波 3（`wss_local_wave3_20260913`，Slurm 14198 → 14199，1 h 10 min，已完成）：PF6/VF6 seed 7/2025 + 同 seed P02/V07 对照，seed 配对参考链 `configs/wss_local_wave3_20260913/refs/`（`tools/prepare_wss_local_wave3.py`）。三 seed 配对 Δ 压力 +0.035/+0.062/+0.039、速度 +0.023/+0.026/+0.033 全部同号 → **PF6/VF6 正式成为压力/速度底座**（[§20.5](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)）；工作簿速度与压力表 82–97 行（`update_wss_local_wave2_volume_xlsx.py --name/--group`）。用户同日裁定：部署拿不到的输入（CFD 出口分流、真值压力梯度、RCR）不做实验，含 oracle 上限探针。
- 文档归档：V5 数据重建记录、交叉审阅意见（含探针）、固定峰值快照审阅稿、局部信息头脑风暴 → `_archive/WSS_PINN/_archive/`；Centerline V2 切换记录 → `_archive/`。索引与结案说明见 [_archive/WSS_PINN/_archive/README.md](_archive/WSS_PINN/_archive/README.md)、[_archive/README.md](_archive/README.md)。

## 2026-09-12｜V4历史稳态梯度与损失英文重绘

- 复用旧V4 seed1234稳态8臂epoch日志与PN三臂梯度CSV，生成3张英文PNG/PDF/SVG及合订PDF；附中文导师短总结、统计CSV和复现脚本。
- [图件与口径](../03-汇报材料/V4汇报/V4_稳态梯度与损失_2026-09-12/README.md)：明确no-slip强负余弦、较弱动量冲突、连续性近零；关闭项标N/A，区分共同数据损失与各臂加权总损失。
- 仅历史证据重绘与精确统计，不改变路线结论；未训练、重评测试集、修改工作簿或旧图。
- 同日按用户反馈将梯度图改为反向箭头示意＋两张实测曲线，移除热图、公式和大段说明；同步更新PNG/PDF/SVG及合订PDF。随后将动态权重改为前50轮放大、99.76%饱和比例、残差比值与数据MSE四联展示，突出epoch24达到9.99及其训练拟合代价。

## 2026-09-12｜WSS 分析与可视化技能对齐现行 V5（指针）

- **本次主要修改**：项目实验分析/postview 技能改为实际 run/config 路由，补齐 V5 指标、峰值帧、坐标、物理恢复与当前导出器限制；node04 按现场资源选择执行方式并继承已有任务授权。
- **对应代码/文档**：`.cursor/skills/` 三个技能及 `.agents/skills/` 共用入口；详细改动与验证见[通用推进记录本次条目](代码修改与实验推进记录.md)。
- **推进到实验步骤**：只读已有结果做技能场景验证，未训练、重评、导出或修改工作簿。
- **当前状态判断**：本次改变 Agent 工作规范，不产生新的模型结论；后续 C1 Pa 图优先复用已验收缓存，R5V 连续 Slice 必须先满足真实体拓扑/配准合同。

## 2026-09-12｜直接WSS 14臂完成、结果回填与提交记录清理

- **本次主要修改**：14臂400轮、28份best/last整壁面test34评估完成；复核42份验收、1143个文件哈希。C1 best/last Pa R²=0.6575/0.6593，best MAE=1.8447 Pa、IoU=0.4698。删除无训练结果的旧提交记录、失败登记16行及相关冗余备份；已完成消融全部保留。
- **对应代码/文档**：[V5跟踪§18](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)、[工作簿](../03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx)、`training_wss_min/tools/report_wss_recovery.py`；旧失败登记生成器与提交脚本移除，通用队列入口改为显式配置路径。
- **推进到实验步骤**：Slurm14119于2026-09-12 15:42:02正常结束；工作簿本轮有效结果位于333–346行，保留327条历史实验及另外两页。
- **当前状态判断**：C1优先保留，C4实际为E3＋E9；E0低于历史M2的差距尚待解释。结论限于单seed、test34开发筛选。

## 2026-09-11｜压力/速度26臂完成、最终判读与工作簿回填

26臂400轮、60组目标/checkpoint和26条history验收均通过；14040耗时12:26:51，14041与14056完成退出0。压力P02通过精度和长波门槛，P03为低MAE备选；速度V07相对同期/历史基线更好，但各速度臂均未通过直接父臂完整门槛；联合J01不构成两任务共同提升。三张原结果表完整回填，新增“体场注意力结论”页，修正历史参照的待完成显示。原始科学报告不改数值；源表3000项及结论页186项转录复核通过，历史公式/合并保留。详见[跟踪§17](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)与[完整结论](../../training_wss_min/experiments/volume_attention_20260910/final_summary.md)。单seed1234、暴露test34边界保留。

## 2026-09-10｜压力/速度26臂后台监控与阶段工作簿回填

按用户要求增加每5分钟后台巡检（Slurm14056），监控14040训练和14041最终汇总。首批P00–P03的8组best/last逐点指标及4条400轮history独立验收通过；工作簿三张结果表新增独立体场分节，总览321–354、教师281–314、汇总449–512，3000项转录核对通过，原25,308个非空单元格、1,360个公式保留。阶段结果显示数值已核验，正式门槛判定待整批完成；新增结果/阶段转换时增量回填，正常巡检静默。监控与最终回填共享文件锁，11项监控检查与9项finalizer检查通过；冻结训练源码和26臂配置未改变。详情见[跟踪§17](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)。

## 2026-09-10｜压力/速度全局注意力26臂：实现、验收与正式执行

用户确认完整实施P/V各12臂及联合2臂，seed1234。新增原SA3逐token FFN容量对照、独立速度/压力heads、联合采样双mask与等权损失、递归fresh配对初始化、联合分任务评估和同点预测缓存。新增全34例分支5mm体积分箱及5/10/20/40mm残差诊断、26臂配置/队列、独立NumPy验收及工作簿幂等分节回填工具。旧默认配置/模型键保持兼容，冻结原18D统计，不扩展曲面曲率或新几何。单测104项+40子测试通过；真实GPU预检14034、26臂两轮冒烟14038、历史R5 best/last复核14036（2948检查）均通过。正式14040已启动（1GPU/4槽/48小时），自动验收与回填14041依赖其结束。原14039因分配限时调整提前中止并归档，14040全部从零训练；当前不作新体场科学结论。进展与最终结果见[唯一跟踪§17](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)。

## 2026-09-08｜V6 单帧 WSS 矩阵：局部尺度 / 解码器 / 病例幅值 / 局部微分几何 / NLL 校准

**本次主要修改**（全部配置驱动，默认值 = 旧行为）：
- `training_wss_min/baseline_models.py` 新增三个零初始化模块：`LocalWallBranch`（support 分辨率上的多尺度球邻域分支，`model.local_branch`）、
  `LocalQueryDecoder`（k 近邻几何条件凸插值 + 零初始化局部残差头，`model.query_decoder='local_attn'`）、`CaseScaleHead`（最粗层池化 → 逐例幅值，`model.case_scale_head`）。
- `evaluate.py`/`dataset.py`：高斯 NLL 的逐点 Jensen 回变换 `eval.pointwise_jensen` + 只读覆盖 `--back-transform {config,exp_mu,jensen}`；`denormalize_wss_lognormal_mean`。
- `config.py`/`dataset.py`：V6 逐点特征键（壁面主曲率族两尺度、curvedness/shape index、切向·法向、分支语义 one-hot、分支内相对弧长）与 sidecar 加载 `data.point_features_root`；曲率族沿用 curvature 的 signed_log1p + p99 截断。
- 数据侧新增 `wss_v5/views/wall_geom_v2.py`：在 PCA 法向切平面做局部二次拟合求主曲率，写 `data_wss_v5/views/wss_min_geom_v2/`（172/172 例，409 MB，40 秒），**不动冻结的 `wss_min_view_v1`**；只用部署可得输入，网格对偶面积仅作审计不进配置。
- 工具/提交：`tools/prepare_v6_singleframe_matrix.py`（含单变量自检与 `--combo`）、`tools/report_v6_matrix.py`（复用 §11 的 `verdict_for`）、`tools/update_v6_matrix_xlsx.py`、`tools/v6_module_diagnostics.py`、`cluster/run_v6_matrix_queue.slurm`（共享显存模式：预检从"卡必须空"改为"剩余显存够"）。

**前后兼容验收**：386 个历史配置全部通过校验；149 个既有单测 + 19 个新单测通过；用新代码只读重评旧 run（R4 s1234 best）与已存 `metrics.json` 逐字段比对，5624 个数值最大绝对差 1.9e-5（GPU 非确定性量级）。

**推进到实验步骤**：11 个臂（A0–A5b，每臂 seed 1234、峰值单帧、400 epoch），Slurm 作业 13164 + 13167，与另一位同学的非 Slurm 进程共享 4 张 4090。
最好配方 **A5 = 局部分支 + 局部微分几何**：物理 R²_cb 0.5567→**0.6230**、归一化 0.7830→**0.8196**（单 seed 带的 5.2 倍）、MAE 2.19→2.01 Pa、top10 IoU 0.355→0.416。
NLL 的收益全部在回变换（同一权重 exp(μ) 0.558 → 逐点 Jensen 0.599）；病例幅值头塌成常数（逐例因子 1.0325–1.0330）；只调 SA1 中心/半径远不如新增局部通路。

**当前状态判断**：单 seed 只作筛选，按既定统计合同**不判定**；A5/A4/A1 需补 seed 7/2025 做三 seed 判定。结果与判读见
[V5 训练实验跟踪 §12](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)，工作簿已回填（总览 255–265、教师视图第 ⅩⅢ 节、汇总对比 371–385）。

## 2026-09-07｜数据预处理图件汇集为老师展示目录

**本次主要修改**：把 2026-07 至 09 的几何坐标配准 QA、Centerline V2 总览/前后对照、权威表面与网格壁面、曲率端区、压力/WSS 重算、ILO 队列审核叠图、V5 特征对照，拷贝进
[数据预处理全历程展示_2026-09-07](../03-汇报材料/数据预处理全历程展示_2026-09-07/README.md)
（约 85 个文件、59 MB）。只做展示拷贝，不改 `outputs/` 真源，不改数据或训练。

**对应代码/文档**：展示入口 README；项目索引 `docs/README.md`「准备论文图和汇报」；全历程汇报准备稿 §10 增加指针。

**推进到实验步骤**：预处理证据可按 01–07 子目录给老师翻阅。

**当前状态判断**：材料齐备可展示。173 例逐例三视图仍只在 `outputs/centerline_v2_full_173_20260828/cases/`，展示包只收总览与代表病例。

## 2026-09-07｜V5：数据母库 172/172、Wave 1/2 重跑、V4 文档归档

- `wss_v5/`（contract/sources/mesh_topology/raw_frames/centerline_features/pointcloud/conditions/store/build_case/build/refresh_geometry/views）：B-Full HDF5 母库；部署侧几何程序 = PCA 法向 + kNN-MST 一致定向（跨壁边惩罚、中心线置信区还原）+ rim 平面拟合虚拟盖 + 自校准 winding 内外判定；atlas 出口命名按几何最近接口重对应（8 例）。gate 定义与 172 例结果见 [V5 数据重建记录](./_archive/WSS_PINN/_archive/WSS_V5_数据重建_Pilot与全量构建记录_2026-09-06.md)。
- `wss_v5/views/wss_min_view.py`：V5 → `training_wss_min` 旧 schema 视图（atlas 解剖坐标架、[−1,1] 归一、train138 log_z 统计、train138/test34 split）；`training_wss_min/config.py` 加 `V5_POINT_FEATURE_KEYS`，`dataset.load_case` 读 `wall_<name>`/`wall_normal_pca_aligned` 可选字段（旧 bundle 不受影响）。
- 配置 `training_wss_min/configs/v5_rerun_20260906/`、提交 `cluster/run_v5_rerun_wave{1,2}.slurm`（作业 13095/13098，GPU 0/1/3）、汇报 `tools/report_v5_rerun.py`、旧 ckpt 34 例交集重评 `experiments/v5_rerun_20260906/legacy_overlap34/`。结果与判读见 [V5 训练实验跟踪](00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)。
- 文档：V4 四份文档归档至 `_archive/WSS_PINN/_archive/`（文首结案说明），全仓库链接同步；新开 [Centerline V5 记录](04-数据处理与CFD/Centerline_V5_点云atlas接入与出口命名修正记录_2026-09-07.md)。
- 工作簿：`tools/update_v5_rerun_xlsx.py` 把 V5 Wave 1/2 的 8 个 run + 5 个旧 ckpt 交集参照回填到《WSS_PointNet实验矩阵与结果汇总last.xlsx》三张表（总览行 195–207、教师视图第 Ⅹ 节、汇总对比 2026-09-07 节），备份在 `experiments/v5_rerun_20260906/`。

## 2026-09-06｜V5 v0.3-review 局部几何输入主线 · 设计修订 / 实施待决定

**本次主要修改**：按用户最新偏好，将
[WSS V5 完整设计方案](00-V5设计与历史跟踪/WSS_V5_几何点云与中心线驱动的WSS及流体场预测完整设计方案_2026-09-06.md)
修订为 `v0.3-review`。默认输入为 xyz + 局部几何/中心线特征，保留局部物理半径、弧长等，
优先直接 WSS 与体场预测头。入口/出口面积等独立全局参数、面积分流代理及依赖面积的参考场
退出默认预测路径；RCR 不作模型输入的边界不变。

**数据与入口边界**：面积/体积积分权、接口审计元数据仍保存在 bundle，用于积分评价及审计，
不拼接为模型特征。其余数据合同及 Phase C 待决定项保持不变，B-Full + Zarr 仍为优先 pilot 候选。
同步[路线入口](_archive/WSS_PINN/README.md)、[项目索引](../README.md)与通用推进记录；
`v0.2-review` 及更早日志保留历史版本含义。

**验证与状态**：检查新增本地链接与 diff 空白。本次仅修订设计文档，未修改模型代码、训练配置、
数据或 checkpoint，未执行正式 bundle 重建、pilot 或训练。

## 2026-09-06｜V4 速度 R² 后处理补 DATA / BC+PDE-FIXED · 已完成

**本次主要修改**：在既有 `outputs/wss_pinn/audits/v4_speed_r2_postview_20260902/` 上补
seed1234 SP-PNPP 纯 DATA（index 4）与 BC+PDE-FIXED（index 6）的 official speed R²
best/worst 面片与腔内速度点云；原 index 5/7/8 未重算。导出脚本增加历史 BC z-score
兼容（旧矩阵 stats 无 `v4_bc_vector_2026-09-05` 合同），并在增量导出时合并已有臂
manifest / README。

**对应代码/文档**：`docs/03-汇报材料/tools/export_v4_speed_r2_postview.py`；
工作簿对照 `docs/03-汇报材料/WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx`
（未改表，仅按矩阵选臂）。产物见
`outputs/wss_pinn/audits/v4_speed_r2_postview_20260902/README_后处理打开说明.md`。

**推进到实验步骤**：历史 V4 seed1234 SP-PNPP 四臂（DATA / BC / FIXED / EMA）+ 瞬态 PN-DATA
的速度 R² 极值病例后处理已齐。

**当前状态判断**：四例面片覆盖率均为 100%。DATA worst/best 为 `ILO/SUN_DONG_XIN-0/before`
（R²=−0.774）/ `ILO/SUN_XU_XIA-1/before`（R²=0.227）；FIXED worst/best 为
`AG/fast/SUN_ZHI_YU`（R²=−1.228）/ `AG/fast/YAO_CUN_HONG`（R²=0.071）。
可视化产物，正式指标仍以 official eval JSON 为准。未写入工作簿新行。

## 2026-09-06｜V5 v0.2-review 综合审阅修订 · 设计完成 / 实施待决定

**本次主要修改**：综合核验交叉审阅意见与历史实验真源，修订
[WSS V5 完整设计方案](00-V5设计与历史跟踪/WSS_V5_几何点云与中心线驱动的WSS及流体场预测完整设计方案_2026-09-06.md)
至 `v0.2-review`。RCR 不作模型输入；协议探针覆盖 172 例，`(R1+R2)·C` 近常量而
`R2·C` 不恒定，左右 50/50 仅为末端 DC 电导近似，不作为实际瞬态流量硬约束。
历史小样本平台、面积回归残差与旧标签噪声估计均不作为新数据的理论精度上限。

**对应设计与入口**：物理参考场及残差学习、时间低秩表示、提前验证部署采样纳入候选；
B-Full + Zarr 为优先 pilot 候选，模块范围、物理格式和 Phase C 重建方式仍待用户决定。
[交叉审阅意见原稿](./_archive/WSS_PINN/_archive/WSS_V5_设计方案交叉审阅意见_2026-09-06.md)与
[BC 协议结构探针](./_archive/WSS_PINN/_archive/V5审阅_BC协议结构探针_2026-09-06)保留供追溯；同步
[路线入口](_archive/WSS_PINN/README.md)、[项目索引](../README.md)与通用推进记录。

**验证与边界**：核对新增本地链接与 diff 空白。此次为设计和文档修订，候选模型/实验未实现，
未修改训练代码、数据、配置、checkpoint 或作业，未执行正式 bundle 重建或训练。

## 2026-09-06｜V5 完整设计草案 · 交叉审阅后再确定 bundle

**本次主要修改**：按用户明确的部署方式编写
[WSS V5 完整设计方案](00-V5设计与历史跟踪/WSS_V5_几何点云与中心线驱动的WSS及流体场预测完整设计方案_2026-09-06.md)，
状态 `v0.1-review`。覆盖几何点云/中心线输入、共享多尺度编码、WSS幅值与空间分布、速度/相对压力、
多任务与可选物理约束、患者级CV、工程能力分级和GPU推理。数据提案为不可变物理量母库＋多种实验视图＋索引缓存，
保留完整体域/壁面、可逆压力和可选拓扑；完整时序物理量估算约95.40GiB，未含静态/缓存/格式开销。

**验证与边界**：独立审阅补充固定积分参考集的query分块不变性、非线性动量与几何-only条件平均的区别、
R²与时间聚合公式、Compact覆盖能力及缓存失效规则；检查本地引用、Markdown结构和新增diff。
本轮只写方案与入口/推进记录，未修改训练代码、数据、配置、checkpoint或作业。

**下一步**：用户交叉审阅后确定模块范围、逻辑schema和候选存储，再少例pilot冻结物理格式，最后决定正式Phase C构建。
文中的模型、精度门槛和实验参数均为提案，不代表已验证或已冻结。

## 2026-09-06｜终审版 §5.4 补压力 / WSS 前后对照表 · 已完成

**本次主要修改**：终审版 §5.4 改为两张总表。表 A 覆盖 08-30 P0-5 八例低压力族（旧 V4 `steady_reference_pa` 476–3,508 Pa → 09-01/02 后 11.7–15.9 kPa）及 ZHOU/ZUO Q 长尾；表 B 覆盖同一批病例峰值帧 WSS p50/p99/max 前后，外加 D 层 4 例 20→60 次。数字来自旧 V4 manifest、验收计划 §1.1 与 `audits/superseded_20260904|06/wall_wss`。逐例 CSV 写在 `V4汇报/V4_数据审查与中心线修复_2026-09-06/pressure_wss_before_after.csv`。单文件版已重建。

**对应代码/文档**：终审版 HTML；审查真源 08-30 审阅 P0-5、anatomy-only §17/§23/§27。未改训练产物。

**推进到实验步骤**：汇报材料整理。

**当前状态判断**：压力低值簇与壁面 WSS 错配的前后对比已可直接汇报。formal 仍 No-Go。

## 2026-09-06｜终审版补 §5 数据审查图 · 中心线错配 / 曲率 / 压力·WSS 重算 · 已完成

**本次主要修改**：终审版 HTML 新增第 5 节，把 08-28～09-06 的数据合同工作收进汇报稿（不是 16 组训练结果）。图件汇到
`docs/03-汇报材料/V4汇报/V4_数据审查与中心线修复_2026-09-06/`：`YANG_YU_QING` 错 STL、`XIE_JIN_QUAN` 漏支、
三例 STL≠CFD 壁面叠图与 `.cas` 网格重提拼图、`LI_LAO_PING` 半径自适应曲率、端点 1R 持值、
以及 10 例壁面–体域 Δp + D 层 4 例 WSS 20→60 次对照图。终审版单文件已重建（21 张图，17 MB）。

**对应代码/文档**：脚本 `V4汇报/V4_数据审查与中心线修复_2026-09-06/plot_v4_data_review_figs.py`；
终审版 `WSS_PINN_V4实验分析与汇报提纲_2026-09-02（终审版）.html` 与 `_单文件版.html`；
审查真源 `WSS_PINN_V4_anatomy-only预处理准备与173例数据核查_2026-09-03.md` §12–13 / §23 / §26–27。
未改训练产物，formal 仍 `training_ready=false`。

**推进到实验步骤**：汇报材料整理；不涉及训练/评估。

**当前状态判断**：终审版已能展示中心线错配修复、曲率 atlas 与压力/WSS 标签重算。正式重建仍 No-Go。

## 2026-09-06｜V4 汇报图件去重 · 旧双份目录删除 · 已完成

**本次主要修改**：`docs/03-汇报材料/V4汇报/` 定为 V4 汇报图件唯一存放处。删除两处旧双份目录，不再留指针夹：
`docs/02-推进与变更/WSS_PINN/V4_seed1234_0-15_训练曲线与诊断图_2026-08-31/`（fig1–8 + 合订 PDF）、
`docs/03-汇报材料/V4_BC-PDE_FIX_EMA_横向R2散点图_2026-09-01/`（8 臂横向 R² / 代表病例 / full-wall WSS，与 `V4汇报` 副本字节级相同）。
出图脚本默认输出改到 `V4汇报/`；终审版 HTML 已按改过的 `plot_v4_report_figs_v2.py` 重绘 fig4/fig7/fig11 并打包
`WSS_PINN_V4实验分析与汇报提纲_2026-09-02（终审版）_单文件版.html`（13 张图，终审稿已去掉 fig6 / fig12）。

**对应代码/文档**：图件真源 [`V4汇报/`](../03-汇报材料/V4汇报/README.md)；
脚本 `docs/03-汇报材料/tools/plot_v4_bc_pde_r2_horizontal.py`、
`plot_v4_representative_regressions.py`、`plot_v4_fullwall_wss_regressions.py`、
`update_v4_fullwall_wss_workbook.py`、`build_v4_report_singlefile.py`；
分析稿 `WSS_PINN_V4实验分析与汇报提纲_2026-09-02.md` / 终审版 HTML。未改训练产物。

**推进到实验步骤**：汇报材料整理；不涉及训练/评估。

**当前状态判断**：旧双份目录已删除，图只在 `V4汇报/`。formal Centerline-V2 训练仍是 `training_ready=false / No-Go`。

## 2026-09-04｜BC/RCR V4 与非 PINN 血统对齐 · formal backbone 选择 B · 文档完成 / 训练 No-Go

**本次主要修改**：交叉核对非 PINN 点云实验、体域 PINN V1/V2、field-v4、旧 BC/RCR
V4 代码/配置/产物和两份汇报工作簿。冻结正式口径：Centerline-V2 formal V4 按用户
选择 B 使用 PointNet P2V / 纯 PointNet++ D2 `c125-k128`，只继承骨干结构并从随机
初始化；旧 pre-Centerline-V2 V4 继续如实记录为 PN `64→128→256`、PNPP 单层
`FPS-128+32-NN` 的约 250k 容量配平历史 screen。同步澄清 P2V 父采样为 SEP、D2 父
采样为 SAME，V1/V2 只继承层宽/SA/support 概念；沿法线剖面证据属于
velocity→WSS/Profile oracle，不属于 P2V/D2 sampler。修正 DATA/DATA+BC、query geom、
`Q_actual_peak`、旧矩阵状态、真实 mm near-wall 与多处历史/当前 V4 混写，并把 formal
172 例 train138/test34 与历史 173 例 train138/test35 分开。近壁三图已重绘独立标题，
明确图 2/3 展示旧 matched-v1.2，而非尚未实现的 formal P2V/D2 sampler。

**对应代码/文档**：[WSS_PINN 路线真源](_archive/WSS_PINN/README.md)、
[V4 大重构设计](_archive/WSS_PINN/_archive/WSS_PINN_V4大重构设计方案_2026-08-08.md)、
[正式重建清单](_archive/WSS_PINN/_archive/WSS_PINN_V4正式重建前剩余整改问题与验收计划_2026-09-03.md)、
[非 PINN PointNet 矩阵](_archive/WSS最小化/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、
[训练跟踪](_archive/WSS最小化/WSS最小化_训练实验跟踪.md)、`wss_pinn/README.md`、根/`docs` README、
`docs/实验设计总纲.md`、V4 分析 Markdown/HTML/单文件版与
[`V4汇报/`](../03-汇报材料/V4汇报/README.md)。两份工作簿仅改描述性单元格：
`WSS_PointNet实验矩阵与结果汇总last.xlsx` 与
`WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx`；可复现、安全保留
OOXML drawings 的更新器为 `docs/03-汇报材料/tools/align_v4_backbone_lineage_workbooks.py`。
该更新器同时修复 PINN 主表 `A1:AO34` 打印区及两个 V4 sheet 的横向、单页宽、独立打印区；
LibreOffice 导出时两个 V4 sheet 分别成为第 192/193 页且各占一页。近壁说明、三图、CSV、
VTP 与可复现脚本位于
[`V4_近壁采样核查_2026-09-04`](../03-汇报材料/V4汇报/V4_近壁采样核查_2026-09-04/README.md)；
`build_v4_report_singlefile.py` 已在 PNG 重绘后重新打包 15 张图。

**推进到实验步骤**：完成模型血统、版本命名、历史结果与 formal 设计的文档冻结；近壁
审计已在同病例 `RAN_QING_BO` 展示旧 PN/PNPP 共用 uniform support5000 及 PNPP
FPS128/32NN。未修改训练模型、配置、数据 bundle、checkpoint 或作业；下一步是把
P2V/D2 连续 query 模型和 formal sampler 落入独立配置并重新执行导数/参数量/preflight Gate。

**当前状态判断**：旧矩阵本地 46/48 有完成摘要、38/48 有 official 评估；46/47 只有
epoch5098/4691 部分产物且远端状态未复核。formal split=train138/test34，backbone=P2V /
D2 `c125-k128`，但代码/配置尚未改造，继续 `training_ready=false / No-Go`。真实 1.5 mm
带未选完全部点，却因边界层网格加密覆盖 50.11%–88.10% cell（中位 68.49%）；正式近壁
sampler 仍须另行冻结，不能把旧 uniform 或法线后处理证据写成已实现设计。

## 2026-09-03｜WSS_PINN V4 正式重建前剩余整改合同冻结 · formal No-Go

**本次主要修改**：在八例低压力 raw 更新、ZHOU/ZUO 原 Q 长尾修复、MENG monitor 整理、
LIU_YUE_DONG UDF 归档和 LIU_ZONG_YANG 完整周期复核基础上，新增正式重建前剩余整改清单。
把“已签收压力问题”和“仍阻断的数据合同”分开，冻结七项工作：唯一分支段曲率 atlas、
全链 anatomy-only、5 个 `blood↔bloodN` 解剖切面 BC、train138 条件长尾复审、ZHOU 单步
收敛、raw 内容 SHA、正式 bundle/stats/Gate 重建。

**对应代码/文档**：
[正式重建前剩余整改问题与验收计划](_archive/WSS_PINN/_archive/WSS_PINN_V4正式重建前剩余整改问题与验收计划_2026-09-03.md)、
[WSS_PINN 当前入口](_archive/WSS_PINN/README.md)、[项目文档索引](../README.md)、根 `README.md`、
`wss_pinn/README.md`、`wss_pinn/AGENTS.md`、`docs/实验设计总纲.md`；08-30 初审文档新增
最新状态指针。本次未修改 builder、训练代码、
数据数组、配置、checkpoint 或作业。

**推进到实验步骤**：完成正式 cutover 的问题定义、实施顺序和机器/人工验收门槛；尚未执行
feature atlas、zone mask、interface BC 或正式 173 例重建。

**当前状态判断**：八例低压力为 `8/8` raw 侧通过，ZHOU/ZUO 原 Q 长尾通过；但
2026-08-31 staging 已过期。正式训练继续 `training_ready=false / No-Go`，不得先重算旧
staging stats，也不得 resume 旧 checkpoint。

## 2026-09-02｜V4 index 5/7/8 速度 R² best/worst ParaView 包 · 诊断文档归档 · 已完成

**本次主要修改**：按 V2/V3 审计同款流程，为 V4 seed1234 矩阵 index 5/7/8 各导出 official
test35 速度 R² 最好与最差病例的后处理包：源点云 VTP、Gaussian r=3 mm 壁面 STL 面片
（含 `wss_pred_over_wss_pred_max`）、解剖 ROI 速度点云。同期将已完成的 field-v4
科学合同移入 `_archive/`，不再作为待执行入口。

**对应代码/文档**：`docs/03-汇报材料/tools/export_v4_speed_r2_postview.py`；产物
`outputs/wss_pinn/audits/v4_speed_r2_postview_20260902/`（打开说明见同目录
`README_后处理打开说明.md`）；归档
[核心代码诊断与下一轮设计建议](_archive/WSS_PINN/_archive/核心代码诊断与下一轮设计建议_2026-08-05.md)。

**推进到实验步骤**：只做可视化导出与文档归档，未改 checkpoint、训练或正式指标。

**当前状态判断**：6/6 病例 STL 映射覆盖率 100%。ParaView 主文件是各例
`*__surface_wall.vtp`（WSS）和 `*__anatomical_roi_velocity_pointcloud.vtp`（速度）。
index 5=`V4-SP-PNPP-BC-s1234`（worst `AG/fast/SUN_ZHI_YU` −0.685 / best
`AAA/unruputer/WANG_MAN_TIAN` 0.130）；index 7=`V4-SP-PNPP-BC-PDE-EMA-s1234`
（worst `AG/slow/HE_SHU_ZHEN` −1.389 / best `AAA/ruputer/WANG_FU_SHUN` −0.015）；
index 8=`V4-TR-PN-DATA-s1234`（worst `ILO/LI_YOU_ZHI-0/before` −0.707 / best
`ILO/YANG_WEN_TAI-0/before` 0.291）。正式 speed R² 仍以 official evaluation JSON 为准。

## 2026-08-31｜WSS_PINN Centerline V2 几何统一与 173 例 staging 全链重建 · staging_pass / training_ready=false

**本次主要修改**：解释并修复旧 steady/transient 几何通道差异——它不是设计要求，而是
steady 复用 `local_radius_mm`、transient 复用 `NormRadius=distance/radius` 的历史上游合同
漂移。新增 `wss_pinn/v4/geometry_v2.py`，统一 raw geometry 为
`abscissa_norm/local_radius_mm/curvature_per_mm`，模型统一使用
`signed_log1p(curvature_per_mm)`；新增 `build_centerline_v2.py` 从完整 raw peak/81 帧重建。
修改 loader/evaluate/tests，增加显式 schema dispatch、split 病例集合核验、manifest/stats/
case/array-audit SHA、build-input digest、resume/pilot 防护和 `training_ready=false` 训练硬阻断。
checkpoint/resume/evaluate 同时新增 manifest/stats/split/array-audit/case-manifest-root 内容
快照绑定，避免同路径刷新数据后误用旧 checkpoint。
独立代码复核发现首版新边界漏 inlet-wall rim buffer，已恢复 shared-node + one-edge 过滤，并
补齐 steady/transient near-wall/core `region` 与 `distance_to_wall_mm`。

**对应代码/文档**：`wss_pinn/v4/geometry_v2.py`、`build_centerline_v2.py`、`data.py`、
`evaluate.py`、`wss_pinn/tests/test_volume_bc_rcr_v4.py`；数据根
`data_wss_pinn/volume_uvwp_bc_rcr_v4_centerline_v2_rawfull_v2_staging_train138_test35/`；
机器 Gate `outputs/wss_pinn/volume_uvwp_bc_rcr_v4_centerline_v2_rawfull_v2_staging/audits/gate_report.json`
与 `full_array_audit.json`；同步更新 [173 例审阅与修复计划](_archive/WSS_PINN/_archive/WSS_PINN_V4_173例训练数据数值与刚性配准审阅及修复计划_2026-08-30.md)、
[WSS_PINN 当前入口](_archive/WSS_PINN/README.md)、[代码入口](../../wss_pinn/README.md)、
[V4 设计基线](_archive/WSS_PINN/_archive/WSS_PINN_V4大重构设计方案_2026-08-08.md) 和 `wss_pinn/AGENTS.md`。

**推进到实验步骤**：Phase 1/2/3/5 的 staging 实施完成。173 例采用每例 manifest +
steady/transient/boundary direct `.npy`，不是从旧 WSS-min `.npz` bundle 派生；共 5,182 数组、
11.1222 GiB。steady 来自完整 raw peak；transient 每例 15,000 个唯一 cell × 81 帧对齐；
173/173 统一 Fluent face 边界。入口 rim filter 为 `128,437→70,016`，最小 inlet-wall 距离
`0.249 mm`。全数组 SHA/shape/dtype/NaN/Inf 审计 0 错误；最终 manifest/stats/array-audit
SHA 分别为 `ff556c95...9037fd5`、`eda1679a...ab4d7c9`、`0576854c...ba1dfef`。
最新 stats 下 steady/transient 默认 5k loader 各遍历 173 例，support/query 均为 5,000
unique，无非有限值；39/39 volume 测试通过；单例 resume pilot 不覆盖全量 manifest。

**当前状态判断**：工程 Gate 为 `staging_pass`，但 manifest 仍为 `training_ready=false`，
正式训练 No-Go。尚需关闭：8 例压力低值簇、14,013 个 raw frame content SHA256、稀疏
速度/BC 长尾与 loss 敏感性、4 例缺失 outlet monitor 标签的 optional/补建决策、正式
route/config/output 与 CPU/GPU preflight。未应用 pressure/velocity clip 或 offset；本次没有
新增训练、提交 job、修改 checkpoint，也没有改变旧 48-run 状态。

## 2026-08-31｜V4 seed1234 0–15 训练曲线与诊断图 · 已完成

**本次主要修改**：对已完训并完成 official test35 + WSS 下游的 seed1234 全部 16 臂
（index 0–15）做训练曲线与失败机制可视化：raw data_total 与 u/p 分通道曲线、
λ_pde/PDE/RCR 残差动态、训练拟合 vs 测试主指标散点、`E_rel_l2` 全景（含
epoch_07500 checkpoint 敏感性）、逐病例 speed 方差比箱线、瞬态压力 gauge 异常散点、
WSS 与近壁场耦合，共 8 图 + 合订 PDF + 可复现脚本。全程只读训练/评估产物，
未改代码、数据、配置、checkpoint 或训练状态，未回填 xlsx。

**对应代码/文档**：新增目录
[V4汇报/V4_seed1234_0-15_训练曲线与诊断图_2026-08-31/](../03-汇报材料/V4汇报/V4_seed1234_0-15_训练曲线与诊断图_2026-08-31/README.md)
（图件清单、数据真源与口径限制见其 README）；同步更新
[WSS_PINN 当前入口](_archive/WSS_PINN/README.md)。

**推进到实验步骤**：为 V4 seed1234 screen 的老师汇报与下一轮优化方向提供图证；
不改变 48-run 训练队列与评估协议。

**当前状态判断**：图证支持四个机制结论——(1) train-only plateau 停止协议下过拟合
主导（SP 8 臂 7500 均优于 last，DATA 训练最好测试最差）；(2) 物理臂 E 增益主要为
幅值压缩趋零（方差比阶梯 0.39→0.04，TR-PN-EMA 中位 ≈3e-4）；(3) EMA 控制器开局
贴死 λ_max=10 全程未调节，TR-EMA 有 continuity 1e-12 平凡解区段，RCR 残差比
DATA+BC 臂差约 300 倍；(4) 瞬态压力均值 −20 为 2 例 gauge 口径异常病例
（ZHANG_YONG_SHENG-0/before ≈196 Pa、WANG_FU_SHUN ≈688 Pa）拉爆的假象，
中位 +0.87。均为单 seed 探索性读数，非 Holm 结论。

**下一步**：候选优化方向按证据强度排序为停止/选模协议（train138 内部 CV 或
milestone 泛化曲线诊断）、173 例压力 gauge 口径审计、λ_pde 控制器重设、逐病例
无量纲化输出；与 Centerline V2 重建的先后关系以 08-30 No-Go 结论为准。

## 2026-08-30｜WSS_PINN V4 173 例训练数据数值与刚性配准审阅 · No-Go

**本次主要修改**：只读审阅实际进入 V4 的 train138/test35 共 173 例数据，核对
Centerline V2 provenance、steady/transient 几何语义、曲率、压力、刚性变换中心与左右
方向、volume scale、空间池、边界资产、stats、loader/resume Gate，并形成问题清单和分阶段
修复计划。本次未修改代码、数据、配置、checkpoint、作业或训练状态，未停止现有任务。

**对应代码/文档**：新增
[173 例训练数据数值与刚性配准审阅及修复计划](_archive/WSS_PINN/_archive/WSS_PINN_V4_173例训练数据数值与刚性配准审阅及修复计划_2026-08-30.md)；
同步更新 [WSS_PINN 当前入口](_archive/WSS_PINN/README.md)、
[V4 设计基线](_archive/WSS_PINN/_archive/WSS_PINN_V4大重构设计方案_2026-08-08.md) 与
[docs 索引](../README.md)。

**推进到实验步骤**：完成 Centerline V2 下游 cutover 前的数据审阅与 Go/No-Go 判定；
现有 48-run 统一标记为 `pre-Centerline-V2 / old-registration-frame historical matrix`，
可继续完成并保留为历史 screen，但不得作为新中心线数据的性能结论。

**当前状态判断**：**新正式训练 No-Go**。旧旋转矩阵数值正确、数组无 NaN/Inf、当前 split
无实际 test 泄漏，但 173/173 尚未绑定 Centerline V2；steady/transient 第 2 几何通道语义
不同，旧曲率系统性损坏，8 例压力低值簇未解释，左右语义、volume scale、空间池和数据快照
Gate 仍有缺口。下一步必须从完整 raw 数据建立独立版本化 root，全链重建、重算 stats 并
通过新硬 Gate；旧 checkpoint 禁止在新数据上 resume/warm-start。

## 2026-08-30｜Centerline V2 优化前后对照图 · 已完成

**本次主要修改**：用 08-27 审计包中的旧中心线与 08-28 V2 产物，按同一机位渲染
`YANG_YU_QING-1`（错 STL、中心线整体脱离）和 `XIE_JIN_QUAN`（漏支、
`Target not reached`）两例左右对照图，供汇报使用。未改提取算法或全量产物。

**对应代码/文档**：`tools/render_centerline_before_after.py`；
图 `outputs/centerline_v2_full_173_20260828/compare_before_after/`。

**推进到实验步骤**：仅补汇报对照图，不切换 `data_new` / bundle。

**当前状态判断**：两例前后差异可直接展示。YANG 旧中心线相对权威表面约偏
100 mm；XIE 由 4 端点/2 分叉补为 5 端点/3 分叉。

**下一步**：下游 cutover 仍按 08-28 切换记录，不因本图改变。

## 2026-08-30｜node04 直启 index 46/47 · 进行中

**本次主要修改**：`12210` 排队不是只剩 46/47，还有 `16/17`。用户要求把
master 上排队的 46/47 挪到 node04。已 `scancel 12210_{46,47}`，在 04 两张
空闲 A100 上全新直启（非 resume、非 Slurm）。未动 master 上
`12210_{39,42,43,45}`；`12210_{16,17}` 仍 `%4` PENDING。

**对应代码/文档**：`wss_pinn/cluster/launch_node04_v4_index46.sh`、
`launch_node04_v4_index47.sh`、`launch_node04_v4_index46_47.py`；
记录 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/node04_direct_46_47_20260830.json`；
日志 `outputs/wss_pinn/slurm/v4_train_node04_{46,47}.out`。

**方法变更**：无。node04 仍不在 GPU 分区，按 11/14/15 先例 SSH 直启。

**关键指标**：46 `V4-TR-PNPP-BC-PDE-F-s3456` GPU0 PID `2261430` 已写 epoch 0–2；
47 `V4-TR-PNPP-BC-PDE-EMA-s3456` GPU1 PID `2261834` 已写 epoch 0–1。
正式 `run_dir` 原先不存在。

**Go-NoGo**：不适用。本轮只改调度，不评 test35、不填 xlsx。

**推进到实验步骤**：矩阵训练从「master 四卡 + 排队 16/17/46/47」变为
「master 四卡 + node04 两卡 + 排队只剩 16/17」。

**当前状态判断**：04 两卡已占用。master 下一空卡会启动 16 或 17（全新开跑）。

**下一步**：保持 46/47 与 master 四臂跑完；16/17 仍走原数组。

## 2026-08-28｜index 15 official test35 + 同协议 WSS + 工作簿第 15 行 · 已完成

**本次主要修改**：index 15 `V4-TR-PNPP-BC-PDE-EMA-s1234` 已于 10:40 完训
（`max_epochs` 10000）。在 node04 GPU0 跑 official test35（`last_converged` 与
`epoch_07500`），GPU1 跑与 0–14 相同的峰值全场 + Profile-Secant V3 ×1200 WSS。
随后把主表第 15 行（工作簿行 34）和敏感性 sheet 补齐。xlsx 填前备份
`…_备份_补V4_15前.xlsx`。未按 test35 反选 checkpoint。

**对应代码/文档**：
`wss_pinn/v4/evaluate.py`；`docs/03-汇报材料/tools/update_wss_pinn_v4_workbook.py`；
场 `evaluation_official_last_converged_full.json`；
WSS `outputs/wss_pinn/audits/v4_workbook_0_14_20260823/arms/V4-TR-PNPP-BC-PDE-EMA-s1234/`；
记录 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/node04_eval_wss_15_20260828.json`。

**方法变更**：无新协议。seed1234 16 臂场+WSS 现已齐。

**关键指标**（test35 / seed1234 / last_converged）：
`E_rel_l2` = **0.951 ± 0.028**；speed R²_cb = −0.071 ± 0.077；
WSS R²_cb = **−0.198 ± 0.155**。`epoch_07500` 的 E = 0.963。
seed1234 TR-PNPP 四臂现为 DATA 1.037 / DATA+BC 0.941 / PDE-F **0.907** / EMA 0.951。
全矩阵已评最低仍约 0.894（TR-PN-BC-PDE-F-s1234 与 TR-PNPP-BC-PDE-F-s2345）。

**Go-NoGo**：
- **待定** — seed1234 16 臂 screen 已齐，但仍是单 seed，无 Holm，不能宣布正式赢家。判据：§12.0。
- **No-Go** — 用本行宣称可用场/WSS 重建。E≈0.95 仍贴近零场；WSS R² 为负。

**推进到实验步骤**：工作簿 V4 行 0–15 已填；其余 seeds 的 xlsx 行仍未写。

**当前状态判断**：场+WSS 已评 38/48。缺 16–17 / 39 / 41–47。

**下一步**：其余臂完训后再评；3-seed 齐后再做 Holm。不按本行改训练。

## 2026-08-27｜YANG_YU_QING 中心线错 STL 修复与 173 例拓扑审计

**本次主要修改**：确认 `ILO/YANG_YU_QING-1/before` 的旧中心线来自同目录错误
`YANG_YU_QING-sq-wrap.stl`，而 CFD/bundle 权威表面为 `ANG_YU_QINF-sq.stl.stl`。
在权威 STL 上重新运行 VMTK，源 `centerline.vtp/csv` 已替换，旧文件按 SHA256 归档；
为避免正确中心线较完整表面略短导致 `auto_centerline` 把 m→mm 因子误算为约 950，
新增该例固定 `unit_factor=1000` 的可审计覆盖，并把覆盖状态/原因写入 bundle 和 report。

**对应代码/文档**：`pipeline_wss_min/config.py`、`pipeline_wss_min/preprocess.py`、
`pipeline_wss_min/tests/test_unit_overrides.py`；
[VMTK 中心线修复与全队列优化方案（已执行归档）](_archive/VMTK中心线修复与全队列优化方案_已执行_2026-08-27.md)；病例内
`data_new/ILO/YANG_YU_QING-1/before/centerline/REPAIR_2026-08-27.md`；审计包
`outputs/centerline_postview_audit_173/`。

**推进到实验步骤**：修复中心线为 2495 点，图拓扑 5 端点/3 分叉节点，5/5 开口覆盖，
开口到最近端点最大 0.386 mm。固定 1000 口径下，壁面距离/局部半径 p95 从
24.233R 降为 1.459R，`>2R` 壁点从 100% 降为 0.259%。隔离 preprocess 81 帧通过，
`unit_factor=1000`、权威 STL scale=1、det(R)=1、无中心线平移修复。单位覆盖测试
`2 passed`。173 例权威 STL 均为 5 开口；172 例中心线为 5 端点，
`AAA/ruputer/XIE_JIN_QUAN` 仅 4 端点，隔离重跑仍出现 `Target not reached`。

**当前状态判断**：YANG 可修复，不做永久排除；但修复前生成的所有中心线依赖 bundle/
sidecar 均为 stale，必须重建后才能用于新实验。XIE 在完成目标级重试算法前应暂时排除
centerline-dependent 实验。尚未全量实现 Centerline V2 或重建 173 例派生产物。

## 2026-08-27｜完训臂 WSS 下游（与 0–14 同协议）· 已完成 · 未填工作簿

**本次主要修改**：用户要求对已评完的 18–38、40 补跑 WSS，口径与 0–14 相同。
node04 GPU1 从 11:45 跑到 15:35，22 臂全部完成。xlsx 仍不改。

**口径（回答「全量还是输入点」）**：
- **速度推理是峰值时刻全场体点**（V1 `interior_coords`），不是训练 2k/5k query。
- 瞬态只解 **peak frame** 到该全体域，不是 81 帧。
- **WSS 本身是 test35×1200 壁面采样**，不是全壁面；Profile-Secant V3 冻结。
- speed OLS 仍用解剖 ROI 上的全场速度。

**对应代码/文档**：
`docs/03-汇报材料/tools/update_wss_pinn_v4_workbook.py arm --output-dir …`；
产物 `outputs/wss_pinn/audits/v4_wss_18_38_40_20260827/`；
汇总 `outputs/wss_pinn/audits/v4_wss_completed_20260827.json`（含 0–14 共 37 臂）；
日志 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/gpu1_wss_18_38_40_20260827.log`。

**实验结果摘要**：WSS R²_cb 仍为负或贴零。DATA 最差（约 −3～−10）；PDE 臂约 −0.6～0。
已评最好贴零：`TR-PNPP-BC-PDE-F-s1234` ≈ 0.00、`TR-PN-BC-PDE-F-s2345` ≈ −0.06。
**禁止当正式赢家。**

| 组 | DATA | DATA+BC | BC+PDE-F | BC+PDE-EMA |
| --- | ---: | ---: | ---: | ---: |
| SP-PN s1234 | −7.97 | −1.89 | −0.36 | −0.10 |
| SP-PNPP s1234 | −7.47 | −1.54 | −0.34 | −0.45 |
| TR-PN s1234 | −4.66 | −0.81 | −0.24 | −0.37 |
| TR-PNPP s1234 | −3.88 | −0.81 | −0.00 | —（15） |
| SP-PN s2345 | —（16） | —（17） | −0.59 | −0.39 |
| SP-PNPP s2345 | −6.26 | −1.46 | −0.36 | −0.10 |
| TR-PN s2345 | −3.92 | −0.85 | −0.06 | −0.10 |
| TR-PNPP s2345 | −2.86 | −1.03 | −0.22 | −0.15 |
| SP-PN s3456 | −10.05 | −1.52 | −0.38 | −0.20 |
| SP-PNPP s3456 | −9.12 | −2.53 | −0.22 | —（39） |
| TR-PN s3456 | −2.82 | —（41） | —（42） | —（43） |
| TR-PNPP s3456 | —（44） | —（45） | —（46） | —（47） |

**遗留问题**：xlsx 未更新。缺 15 / 16–17 / 39 / 41–47。未做 full-wall fit。

**推进到实验步骤**：37/48 已有场指标 + WSS 下游；工作簿仍只有 0–14。

**当前状态判断**：GPU1 已空闲；index 15 仍在 GPU0。WSS 不能当可用重建。

**下一步**：用户点名后再填 xlsx；其余臂完训后再补评。

## 2026-08-27｜完训臂 official test35（22 个新评）· 已完成 · 未填工作簿

**本次主要修改**：用户要求用 node04 空闲卡，把已完训但未 test 的 V4 臂做
official test35，并更新文档、**先不改 xlsx**。当时 GPU0 仍在跑 index 15，
GPU1 空闲。按 0–14 同一协议评完 index **18–38、40** 共 22 臂
（`last_converged`，`eval_support`，稳态全体非壁面 / 瞬态 eligible×81）。
未跑 WSS，未按 test35 反选 checkpoint。

**对应代码/文档**：
`python -m wss_pinn.v4.evaluate --protocol official --checkpoint last_converged`；
日志 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/gpu1_completed_pending_20260827.log`；
记录 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/node04_eval_completed_pending_20260827.json`；
新 22 臂汇总 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/test35_eval_18_38_40_official_20260827.json`；
全部已评 37 臂 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/test35_eval_completed_official_20260827.json`。

**方法变更**：无。口径与 2026-08-23 的 0–14 相同。

**实验结果摘要**：primary `E_rel_l2`（病例等权，越低越好；零场预测 = 1）。
已评 37/48。方向与 0–14 一致：准稳态 DATA 仍 ≥1.10；PDE 臂略好但仍贴近零场。
目前已评最低约 **0.894**（`TR-PN-BC-PDE-F-s1234` 与 `TR-PNPP-BC-PDE-F-s2345`）。
瞬态压力 R² 仍约 −20，只作次要诊断。**禁止当正式赢家 / 3-seed Holm。**

| 组 | DATA | DATA+BC | BC+PDE-F | BC+PDE-EMA |
| --- | ---: | ---: | ---: | ---: |
| SP-PN s1234 | 1.125 | 1.176 | 0.968 | 0.935 |
| SP-PNPP s1234 | 1.105 | 1.153 | 0.964 | 0.910 |
| TR-PN s1234 | 1.041 | 0.937 | **0.894** | 0.965 |
| TR-PNPP s1234 | 1.037 | 0.941 | 0.907 | —（15 在训） |
| SP-PN s2345 | —（16） | —（17） | 0.997 | 0.929 |
| SP-PNPP s2345 | 1.128 | 1.116 | 0.979 | 0.919 |
| TR-PN s2345 | 1.017 | 0.953 | 0.919 | 0.938 |
| TR-PNPP s2345 | 1.018 | 0.950 | **0.894** | 0.951 |
| SP-PN s3456 | 1.139 | 1.138 | 0.984 | 0.926 |
| SP-PNPP s3456 | 1.129 | 1.211 | 0.950 | —（39 在训） |
| TR-PN s3456 | 1.012 | —（41） | —（42） | —（43） |
| TR-PNPP s3456 | —（44） | —（45） | —（46） | —（47） |

**遗留问题**：xlsx 未更新。缺 15 / 16–17 / 39 / 41–47。未跑 WSS。

**推进到实验步骤**：已评完训臂的场指标；工作簿仍只有 0–14。

**当前状态判断**：GPU1 评估已结束并空闲；index 15 仍在 GPU0。矩阵未齐，不能 Holm。

**下一步**：15 与其余臂完训后再评；用户点名后再填 xlsx / 跑 WSS。

## 2026-08-25｜优先直启 index 15（seed1234 末臂）· 进行中

**本次主要修改**：用户要求下一个实验先跑 **index 15**（`V4-TR-PNPP-BC-PDE-EMA-s1234`），
其余 pending 正常排队，以便先齐 seed1234 再汇总评估。未杀 master 上正在跑的
`12210_{31,37,38,39}`。node04 两张 A100 空闲且不在 GPU 分区，按 11/14 先例
SSH 直启 GPU0 全新开跑，并从数组 `scancel 12210_15`。确认正式 `run_dir` 原先不存在，
08-16 半成品仍只在 `*_cancelled_*` 目录。已写出 epoch 0–2。

**对应代码/文档**：
`wss_pinn/cluster/launch_node04_v4_index15.sh`；
记录 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/node04_direct_15_20260825.json`；
日志 `outputs/wss_pinn/slurm/v4_train_node04_15.out`。

**方法变更**：无训练/评估协议变更。15 为全新开跑，不是 resume。

**实验结果摘要**：启动后约 1 min 已完成 epoch 0–2，`mean_data_total` 约 1.17 → 0.89 → 1.00。
PID `2206976`，node04 GPU0。其余队列：`12210_[16-17,40-47]` 仍 PENDING
（`JobArrayTaskLimit`）。

**遗留问题**：15 完训前不能补工作簿第 15 行；GPU1 仍空闲，未另开别的臂。

**推进到实验步骤**：seed1234 第 16 臂已在训；0–14 评估/入账不变。

**当前状态判断**：15 已从数组抽出并在 04 上跑，不会和 master 四卡抢下一个空槽。
16–17、40–47 仍按 `%4` 排队。

**下一步**：等 15 完训后再做 official test35 + 工作簿第 15 行；不按中途 loss 改训练。

## 2026-08-23｜V4 seed1234 0–14 回填工作簿 + WSS 下游 + milestone 敏感性 · 待定（矩阵未齐）

**本次主要修改**：用户授权后，把 seed1234 index **0–14** 写入
`docs/03-汇报材料/WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx`
主表（行 19–33），**index 15 留空**。新增 `E_rel_l2_cb` 列。按 V2/V3 方法补
Profile-Secant V3、test35×1200 WSS，以及 `last_converged` vs `epoch_07500`
场敏感性。未按 test35 反选 checkpoint。备份
`…_备份_补V4_0-14前.xlsx`。

**对应代码/文档**：
`docs/03-汇报材料/tools/update_wss_pinn_v4_workbook.py`；
产物 `outputs/wss_pinn/audits/v4_workbook_0_14_20260823/`；
敏感性 sheet `V4 Checkpoint敏感性`。

**方法变更**：场指标=official 宇宙（瞬态 eligible×81）。WSS=峰值全场速度 +
冻结 Profile-Secant V3 ×1200。speed OLS 用已有解剖 ROI；WSS OLS 用同一 1200
点（V2/V3 后来的 full-wall fit 未重做）。`last` 与 `last_converged` 权值 SHA
相同，敏感性有效对照是 `epoch_07500`。

**关键指标**（test35 / seed1234 / last_converged；禁止与 V2 SAME5K、V3 val-selected 裸比）：

| 臂 | SP E_rel_l2 | SP WSS R²_cb | TR E_rel_l2 | TR WSS R²_cb |
| --- | ---: | ---: | ---: | ---: |
| DATA | 1.125 ± 0.172 | −7.97 ± 11.39 | 1.041 ± 0.143 | −4.66 ± 5.17 |
| DATA+BC | 1.176 ± 0.127 | −1.89 ± 2.65 | 0.937 ± 0.055 | −0.81 ± 1.05 |
| BC+PDE-F | 0.968 ± 0.060 | −0.36 ± 0.63 | 0.894 ± 0.052 | −0.24 ± 0.94 |
| BC+PDE-EMA | 0.935 ± 0.071 | −0.10 ± 0.26 | 0.965 ± 0.019 | −0.37 ± 0.22 |

PN++ 同向；TR-PNPP-F 的 WSS R²_cb ≈ 0.00 ± 0.37。`epoch_07500` 的 E 与终点同方向，
DATA 略好于终点、PDE 臂接近。

**Go-NoGo**：
- **待定** — 单 seed、缺 15、无 Holm，不能宣布正式赢家。判据：§12.0。
- **No-Go** — 用本表宣称可用场/WSS 重建。WSS R² 仍为负或贴零。
- 工作簿 V4 行是 **screen 草稿**，不是 Stage 4 正式入账。

**推进到实验步骤**：xlsx 已有 0–14 可对读的 V4 行；15 与其余 seeds 仍待完训。

**当前状态判断**：主表 + V4 敏感性已按现有完训臂补齐；线性回归两张附表仍只有 V2/V3。

**下一步**：等 index 15 完训后补第 15 行；48-run 齐后再做 3-seed 正式汇总。

## 2026-08-23｜V4 seed1234 index 0–14 的 test35 official 评估 · 待定（矩阵未齐）

**本次主要修改**：新增 `wss_pinn/v4/evaluate.py`。用户授权后在 node04 两张空闲
A100 上评估完训臂 **0–14**（seed1234；缺 index 15
`V4-TR-PNPP-BC-PDE-EMA-s1234`）。未跑 WSS，未按 test35 反选 checkpoint，未填
xlsx。checkpoint 固定 `last_converged.pt`。

**对应代码/文档**：`wss_pinn/v4/evaluate.py`；汇总
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/test35_eval_0_14_official_20260823.json`；
逐臂 `evaluation_official_last_converged_full.json`；日志
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/`。

**方法变更**：协议 `official` = `eval_support`（配置点数；稳态 5k / 瞬态 2k）→
稳态全体非壁面体点；瞬态全部 eligible 缓存点 × 81 帧。这是 V4 评估宇宙，不是
same5k 抽点。Primary 为病例等权速度向量相对 L2
`E_case=‖u_pred−u_true‖₂/‖u_true‖₂`。零场预测的 `E=1`。

**关键指标**（test35 / seed1234 / official / `last_converged`；禁止与 V2/V3 裸比）：

| 臂 | SP-PN | SP-PNPP | TR-PN | TR-PNPP |
| --- | ---: | ---: | ---: | ---: |
| DATA | 1.125 | 1.105 | 1.041 | 1.037 |
| DATA+BC | 1.176 | 1.153 | 0.937 | 0.941 |
| BC+PDE-F | 0.968 | 0.964 | 0.894 | 0.907 |
| BC+PDE-EMA | 0.935 | 0.910 | 0.965 | （缺 15） |

辅指标：准稳态 speed R² 仍为负；瞬态 BC+PDE-F 的 speed R² 约 0.12–0.17。
瞬态压力 R² 约 −20，不能当成功。NMAE 分母为 `RMS(y)`。

**Go-NoGo**：
- **待定** — 正式 16 臂 × 3-seed Holm 未齐；单 seed、缺 15，不能宣布架构/物理赢家。
  判据：设计 §12.0。
- **探索性方向** — 与 train-only「加物理抬高 data loss」不同：test35 上 PDE 臂
  `E` 低于 DATA（DATA 甚至差于零场）。这不是选模依据。
- **No-Go** — 用本切片宣称可用场重建。最低 `E≈0.89` 仍接近零场。
- 不回填工作簿正式 V4 行。

**推进到实验步骤**：Stage 4 的 seed1234 前 15 臂 screen 已落地；矩阵其余 run
与 index 15 仍待完训后再评。

**当前状态判断**：0–14 有可汇总的 test35 primary；结论仍是探索性 screen。

**下一步**：等 index 15 与其余 seeds 完训后再做正式汇总；不跑 WSS；不按本表砍臂。

## 2026-08-21｜归档 field-v4 Stage 0–1 执行提示词 · 🧊冻结

**本次主要修改**：将已完成的
`WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1.md` 移入
`_archive/WSS_PINN/_archive/`，按 V3 六臂先例改名为
`…_已完成_2026-08-06.md`。该提示词文首 2026-08-06 已写「已完成」；现行执行入口是
`volume_uvwp_bc_rcr_v4`，不能再当待办。未改训练代码、未动 `outputs/`。

**对应代码/文档**：
[归档后的 Stage 0–1 提示词](_archive/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1_已完成_2026-08-06.md)、
[归档索引](_archive/WSS_PINN/_archive/README.md)、[路线 README](_archive/WSS_PINN/README.md)、
[V4 设计方案](_archive/WSS_PINN/_archive/WSS_PINN_V4大重构设计方案_2026-08-08.md)。
结论仍留在路线 README 的 field-v4 结果节；证据
`outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_multiseed_and_promotion_report.json`。

**Go-NoGo**：归档不改变历史结论 — **保留 G-Raw**；G-PE / local decoder **No-Go**。
判据来源：field-v4 Stage 1 多种子报告。现行主线仍是新 V4，不据此重开 Stage 2。

**推进到实验步骤**：文档入口收口；训练矩阵状态不变。

**当前状态判断**：活动目录不再把 Stage 0–1 提示词当作当前执行合同。

**下一步**：继续 `volume_uvwp_bc_rcr_v4` 48-run；不恢复 field-v4 Stage 2。

## 2026-08-21｜V4 中期整理：22/48 完训的 train-only 分析 · 待定（primary）/ No-Go（train 侧加物理改善拟合）

**本次主要修改**：未改训练代码、未提交作业、未读 test35、未跑 WSS、未回填
`WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx`（该表仍只有
V2/V3，符合设计「无 test35 不预填 V4 行」）。对已完训 22 个 run 的
`training_summary.json` + 末 epoch `epoch_progress.jsonl` 做中期整理。

**对应代码/文档**：
- 证据：`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/midterm_train_only_20260821.json`
- 路线状态：`docs/02-推进与变更/_archive/WSS_PINN/README.md`
- 设计方案文首进度：`WSS_PINN_V4大重构设计方案_2026-08-08.md`

**背景假设**：四臂分离 DATA / DATA+BC / BC+PDE-fixed / BC+PDE-EMA 的贡献；
停止只看 train-only raw 分量；primary endpoint 仍是 test35 速度向量相对 L2。

**关键指标**（train138 末 epoch raw `mean_data_total`；split train138/test35；
对照为同 backbone×seed 的 DATA 臂；**禁止与 V2/V3 test35 R² 裸比**）：

| 臂 | SP-PN s1234 | SP-PNPP s1234 | 停止 |
| --- | ---: | ---: | --- |
| DATA | 0.1912 | 0.1932 | plateau 9859 |
| DATA+BC | 0.4902 | 0.4997 | max_epochs |
| BC+PDE-F | 0.7260 | 0.7301 | max_epochs |
| BC+PDE-EMA | 0.8329 | 0.8262 | max_epochs / plateau 9985 |

SP s1234 四臂初始化配对成立（PN `e23bcf10481f…`，PNPP `967333b84902…`）。
EMA 四个完训臂 `λ_pde` 有 99.8% epoch 贴上限 10。瞬态 DATA 出现明显 seed 差：
s1234 0.178 vs s2345 0.255。全部完训 run `test35_read_during_training=false`。

队列（2026-08-21 11:34）：完训 22；master `12210_{26,27,29,30}` running；
node04 直启 11/14 running；pending `12210_[15-17,31-47]`。

**Go-NoGo**：
- **待定** — primary endpoint（test35 相对 L2）未评估。判据：设计方案 §12.0。
- **No-Go** — train 侧「加 BC/PDE 改善场拟合」。SP s1234 完整八臂上三组预注册
  contrast 的 `data_total` 全部为正（变差）；p loss 在 PDE 两臂从 ~0.016 升到
  0.50/0.80。判据：设计 §9/§12 四臂分离 + 禁止只看 weighted total。
- **Inconclusive** — PointNet++ vs PointNet（同臂差 ~0.002）。
- 不据此砍臂或改 λ；EMA 撞限可另立案重预注册，须不读 test35。

**推进到实验步骤**：Stage 2/3 训练矩阵进行中；Stage 4 汇总与 test35 仍未启动。

**当前状态判断**：中期证据足够说明「物理臂在训练损失上系统性差于 DATA」，
但 3-seed 矩阵和 test35 主指标都未齐，不能写进工作簿或对外宣称架构/物理优劣。

**下一步**：保持 12210 与 node04 11/14 跑完；seed3456 仍走原数组；矩阵冻结后再
统一 test35。

## 2026-08-20｜node04 两张 A100 空闲，直启 `11/14`；不能同时跑 4 路

**本次主要修改**：用户要求检查 node04 GPU。UID `1006/1007`、public 树可读、两张
A100-40GB 均空闲（仅 Xorg）。**不能同时跑 4 个**：node04 只有 2 张 GPU；它在
Slurm 里属于 CPU 分区且 `DOWN+NOT_RESPONDING`、`Gres=(null)`，GPU 分区只有
master，因此无法 `sbatch` 到 04。按 V3 先例 SSH 直启两臂，一卡一个。先
`scontrol hold 12210_{11,14}`，确认进程与 epoch 写出后再 `scancel`，避免
master 腾卡后重复开跑。未 resume。

| 节点 | 作业 | 状态 |
| --- | --- | --- |
| node04 GPU0 PID `2098215` | index 11 `V4-TR-PN-BC-PDE-EMA-s1234` | 直启全新开跑，已写 epoch 1 |
| node04 GPU1 PID `2098627` | index 14 `V4-TR-PNPP-BC-PDE-F-s1234` | 直启全新开跑，已写 epoch 1 |
| master | `12210_{22,23,24}` | RUNNING |
| master 队列 | `12210_[15-17,25-47]%4` | PENDING（`Resources`） |

日志：`outputs/wss_pinn/slurm/v4_train_node04_{11,14}.out`。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/node04_direct_11_14_20260820.json`。
未评估 test35、未跑 WSS。

**当前状态判断**：集群现为 master 3 路 + node04 2 路并行。04 上没有第三、第四张
卡可再提交。

## 2026-08-20｜V4 恢复四卡并发：`12210` `ArrayTaskThrottle=2→4`，`12210_24` 已启动

**本次主要修改**：用户授权等待中的 V4 数组改回 4 GPU 并行。未新提交数组（避免重复占
index），只对已有作业执行 `scontrol update JobId=12210 ArrayTaskThrottle=4`。
未改训练代码、未 scancel 师姐 `12366`、未 resume 任何半成品。11/14/15/16/17 的
`last.pt` 目录此前已隔离，启动前核对该 5 个 run 目录均不存在。

**作业状态（操作后）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED |
| `7-10,12,13,18-21` | `12210_{7-10,12,13,18-21}` | COMPLETED |
| `22` | `12210_22` / `V4-SP-PNPP-BC-PDE-F-s2345` | RUNNING（GPU1，约 epoch 5782） |
| `23` | `12210_23` / `V4-SP-PNPP-BC-PDE-EMA-s2345` | RUNNING（GPU0，约 epoch 4249） |
| `24` | `12210_24` / `V4-TR-PN-DATA-s2345` | RUNNING（GPU3，全新开跑，epoch 0 已写出） |
| `11,14-17,25-47` | `12210_[11,14-17,25-47]%4` | PENDING（`Resources`；师姐 `12366` 仍占 GPU2） |
| 师姐 | `sunfanji` `12366` PINN_Train | RUNNING（GPU2） |

新增 completed：`18` `train_only_plateau` 9859；`19` `train_only_plateau` 9865；
`20` `train_only_plateau` 9859；`21` `max_epochs` 10000。11/14/15/16/17 再启动
仍为**全新开跑**。未评估 test35、未跑 WSS。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/throttle_restore_4gpu_20260820.json`。

**推进到实验步骤**：数组并发恢复为最多四卡；物理上当前三卡属本数组、一卡属师姐。
师姐结束或 22/23 完训后，第 4 路会立刻拉起（优先空闲 GPU，不重开作业）。

**当前状态判断**：`JobArrayTaskLimit` 已解除。剩余 pending 卡在 GPU 资源，不是
throttle。PointNet baseline 矩阵仍冻结，本条不写入该档案。

## 2026-08-17｜隔离 `12210_{11,16,17}` NODE_FAIL 半成品并放回 `%2` 队列，全新开跑

**本次主要修改**：用户确认后隔离今早 master 失联留下的 `last.pt` 目录，不 resume。
先 `scontrol hold 12210_{11,16,17}`，避免 18/19 腾卡时撞上半成品；再挪走 run 目录与
Slurm 日志。`scontrol requeuehold` 对已 pending 的 task 返回
`Job is pending execution`（它们早上已被 Slurm 自动 requeue），随后 `release`。
未改训练代码、未动 `12210_{18,19}`、未改 `ArrayTaskThrottle=2`。

**隔离位置**：

- `.../transient_autograd/V4-TR-PN-BC-PDE-EMA-s1234_cancelled_12210_20260817_epoch9401/`（epoch 9401 / step 1297370）
- `.../steady_peak/V4-SP-PN-DATA-s2345_cancelled_12210_20260817_epoch1912/`（epoch 1912 / step 263930）
- `.../steady_peak/V4-SP-PN-BC-s2345_cancelled_12210_20260817_epoch1889/`（epoch 1889 / step 260730）
- Slurm 日志：`outputs/wss_pinn/slurm/cancelled/v4_train_12210_{11,16,17}.{out,err}.cancelled_12210_20260817`

**作业状态（操作后）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED |
| `7-10,12,13` | `12210_{7-10,12,13}` | COMPLETED |
| `18,19` | `12210_{18,19}` | RUNNING |
| `11,14-17,20-47` | `12210_[11,14-17,20-47]%2` | PENDING（`JobArrayTaskLimit`） |

11/14/15/16/17 再启动均为**全新开跑**。18 或 19 结束后会先拉 index 11。未评估
test35、未跑 WSS。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/quarantine_11_16_17_20260817.json`。

## 2026-08-17｜核对 `12210_{11,16,17}`：master 失联导致 NODE_FAIL/requeue，非人工杀死

**本次主要修改**：只读核查，未改训练代码、未 scancel。用户看到 11/16 被杀死重入队、
17 只有半成品，根因是今早 **master 节点对 slurmctld 失联**，不是训练脚本自己退出。

**时间线（`/var/log/slurmctld.log` + 作业 `.err`）**：

- 04:25 / 04:51 master 多次 `not responding`，并伴随 slurmdbd `Unable to connect to database`
- 04:51:47 `requeue job JobId=12210_11(12238) due to failure of node master`
- 04:54:17 同样 requeue `12210_16(12307)`；师姐 `12240/12261` 同期被 requeue
- 05:03:19 master 被设 DOWN；作业 `.err` 写 `CANCELLED ... DUE TO NODE FAILURE`，
  随后 `STEPD TERMINATED ... JOB NOT ENDING WITH SIGNALS`（进程没被信号干净杀掉）
- 05:03:50 backfill 启动 `12210_17(12308)`
- 05:04:50 slurmd `Kill task failed`，master 被标 `DRAINING` / reason `Kill task failed`
- 05:12:20 `requeue job JobId=12210_17(12308) due to failure of node master`，master 再 DOWN
- 08:01 slurmctld 恢复并 Recovered 12210 各 task；09:13 master IDLE 后启动了
  **从未跑过的** `12210_{18,19}`，11/16/17 因 throttle=2 继续 pending

**17 为什么是半成品**：17 在 05:03 才开始，05:12 已被 Slurm requeue，但 `.err` 为空、
日志一直写到约 07:13（epoch 1888），说明 slurmd 失联后 Python 可能作为孤儿进程继续跑，
直到节点恢复前后才停。没有 `training_summary.json`，只有 `last.pt`。

**当前状态判断**：18/19 仍在正常训练。11/16/17（以及此前已隔离过的 14/15）再启动前必须
先挪走 `last.pt` 目录，否则会被 `FileExistsError` 拒绝。未评估 test35、未跑 WSS。

## 2026-08-16｜V4 让出两卡给师姐：杀死 `12210_{14,15}`，剩余改为 `%2` 排队

**本次主要修改**：用户要求空出两张 4090 给师姐，并让尚未开始的任务只使用
`12210_11`（物理 GPU0）与 `12210_13`（物理 GPU3）所在的两卡。当时四路运行是
`11/13/14/15`。先把数组 `ArrayTaskThrottle` 从 4 改为 2 并 hold 未启动的
`16-47`，再 `scancel 12210_14 12210_15`，避免空卡被自己的后续 task 立刻占回。

**14/15 半成品与日志**：训练入口遇到 `last.pt` 会拒绝覆盖。已隔离

- `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/transient_autograd/V4-TR-PNPP-BC-PDE-F-s1234_cancelled_12210_20260816_epoch3622/`（约 epoch 3622 / step 499870）
- `.../V4-TR-PNPP-BC-PDE-EMA-s1234_cancelled_12210_20260816_epoch545/`（约 epoch 545 / step 75330）
- Slurm 日志：`outputs/wss_pinn/slurm/cancelled/v4_train_12210_{14,15}.{out,err}.cancelled_12210_20260816`

随后 `scontrol requeuehold 12210_{14,15}` 再 release。14/15 **全新开跑**，非 resume。

**作业状态（操作后）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED |
| `7-10,12` | `12210_{7-10,12}` | COMPLETED |
| `11` | `12210_11` / `V4-TR-PN-BC-PDE-EMA-s1234` | RUNNING（GPU0） |
| `13` | `12210_13` / `V4-TR-PNPP-BC-s1234` | RUNNING（GPU3） |
| `14-47` | `12210_[14-47]%2` | PENDING（`JobArrayTaskLimit`，最多两卡） |
| 师姐 | `sunfanji` `12240/12261` | 释放 GPU1/2 后已启动 |

**推进到实验步骤**：矩阵未完训部分改为两卡串/并行排队；不评估 test35、不跑 WSS。

**当前状态判断**：让卡成功，我方只占 GPU0/3。后续接手先查 `squeue -j 12210` 是否仍为
`11/13` running + `14-47` pending `%2`。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/requeue_14_47_two_gpu.json`。

## 2026-08-14｜V4 补提交未完训 `7-47`：作业 `12210_[7-47%4]`

**本次主要修改**：master 四张 4090 空闲后，按 2026-08-09 约定补跑未完训数组。
不重跑 Stage 0 `11970` / preflight `11971`，也不重跑已 completed 的 `11972_{0-6}`。
index 7 的中断目录含 `last.pt`，训练入口会拒绝覆盖，因此将其挪到
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak/V4-SP-PNPP-BC-PDE-EMA-s1234_cancelled_11972_7_epoch797/`
后对 index 7 **全新开跑**（非 resume）。随后
`sbatch --array=7-47%4` 提交训练脚本 `wss_pinn/cluster/run_bc_rcr_v4.slurm`。

**作业状态（提交当时）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED（`steady_peak × seed1234` 除 PNPP BC+PDE-EMA 外的七臂） |
| `7` | `12210_7` | RUNNING（`V4-SP-PNPP-BC-PDE-EMA-s1234` 全新重跑） |
| `8-10` | `12210_{8,9,10}` | RUNNING（瞬态 seed1234 前三臂） |
| `11-47` | `12210_[11-47%4]` | PENDING（`JobArrayTaskLimit`，throttle=4） |

**对应产物 / 记录**：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/resubmission_7_47.json`；完训七臂仍在
`.../steady_peak/`。本轮不评估 test35、不跑 WSS、不回填工作簿。

**推进到实验步骤**：`steady_peak × seed1234` 七臂已完训；剩余 41 run
（index 7 + 瞬态 seed1234 + seed2345/3456）已重新入队。

**当前状态判断**：补提交成功，四卡已占用。后续接手先查 `squeue -j 12210`；
48-run 全部 completed 前不得写“矩阵完成”，也不得自动开 test35。

## 2026-08-09｜V4 二次截断：取消 `11972_7`，下次补跑从 index `7` 起

**本次主要修改**：在已取消 `11972_[8-47]` 的基础上，再 `scancel 11972_7`
（`V4-SP-PNPP-BC-PDE-EMA-s1234`）。不中断仍在跑的 `11972_{2,3,6}`。用户明确下次
补跑从 index `7` 开始（`7-47`），本轮不自动重提、不评估 test35、不跑 WSS。

**作业状态（sacct / squeue，二次截断后）**：

| array | run | 状态 |
| --- | --- | --- |
| 0 | `V4-SP-PN-DATA-s1234` | COMPLETED（`train_only_plateau`，9859 epoch） |
| 1 | `V4-SP-PN-BC-s1234` | COMPLETED（`max_epochs`，10000） |
| 2 | `V4-SP-PN-BC-PDE-F-s1234` | RUNNING（master） |
| 3 | `V4-SP-PN-BC-PDE-EMA-s1234` | RUNNING（master） |
| 4 | `V4-SP-PNPP-DATA-s1234` | COMPLETED（`train_only_plateau`，9859 epoch） |
| 5 | `V4-SP-PNPP-BC-s1234` | COMPLETED（`max_epochs`，10000） |
| 6 | `V4-SP-PNPP-BC-PDE-F-s1234` | RUNNING（master） |
| 7 | `V4-SP-PNPP-BC-PDE-EMA-s1234` | CANCELLED（约 epoch 797 / step 110120；部分产物保留） |
| 8–47 | 瞬态 seed1234 + seed2345/3456 全部臂 | CANCELLED（与 7 一并待补提交） |

**对应产物**：完训四臂在 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak/`；
index 7 目录含未完训 `history.csv` / `checkpoints/`，补跑时应按全新提交或显式
resume 处理，勿与 completed 混写。

**推进到实验步骤**：`steady_peak × seed1234` 中 DATA/DATA+BC 已完训；PN 两臂 PDE
与 PNPP BC+PDE-F 仍在训；PNPP BC+PDE-EMA 与瞬态/多种子未完成。

**当前状态判断**：当前占 3 卡（`2/3/6`）。下次接手先查 `squeue` 是否仍有
`11972_{2,3,6}`，再按用户指令补提 **`7-47`**。入口已同步：
[WSS_PINN README](_archive/WSS_PINN/README.md)、[V4 设计方案](_archive/WSS_PINN/_archive/WSS_PINN_V4大重构设计方案_2026-08-08.md)。

## 2026-08-09｜V4 正式数组截断：取消挂起 `11972_[8-47]`，仅保留 `0-7`

**本次主要修改**：用户确认四卡已被 `11972` 占满后，取消尚未开始的数组任务
`11972_[8-47]`（`scancel`），释放后续排队；当时仍保留 `0-7`。同日稍后用户进一步
取消 `11972_7`，见上一条；补跑起点改为 `7-47`。

## 2026-08-08｜WSS_PINN V4 v1.2 实现完成并提交 4 卡 Slurm 训练链

**代码与配置**：新增独立 `wss_pinn/v4/` 包，覆盖严格配置合同、UDF Fourier/RCR/物性
解析、train138/test35 + CV5 split、173 例 Stage 0 builder、跨 81 帧点身份与时间映射、
`stl_landmarks_v4` 坐标/速度注册、入口/出口面法向与面积权重、train138-only BC/几何/
瞬态场统计、`RCR~A_out` 审计、PointNet/PointNet++ + 共享 BCEncoder、准稳态/瞬态
strong-form residual、质量流量口径 RCR BC、分组无量纲化、fixed/detached-EMA `λ_pde`、
train-only 平台停止和 `last_converged` checkpoint。生成 16 臂 × 3 seeds = 48 个正式配置；
两个 backbone 参数量分别约 249.9k/250.1k，比值 `1.0008`。

**关键纠错**：真实病例 smoke 首次发现 `processed/coord_normalized` 仍是旧病例内坐标系，
不能与当前冻结 sidecar 混用；实现改读 `processed/features` 原始毫米坐标，并用 bundle 的
病例特异 `unit_factor`（不是固定 1000）、`transform_centroid/rotation/coord_scale` 转到
`stl_landmarks_v4`。DING_JUN_FENG 修正后峰值坐标/速度/压力最大偏差为
`1.56e-7 / 2.38e-7 m/s / 0.0011 Pa`。ZHOU_KE_XUN 的 `A_udf≠A_mesh` 合同复现
`Q_nom_peak=1.07274e-4`、`Q_actual_peak=1.45501e-4 m³/s`。173 例轻量审计另确认旧
`BC_Inlet` monitor 相对 UDF 解析真源最多可差 `2.01%`；因此 monitor 降为诊断，训练入口
流量只使用 `Q_nom·A_mesh/A_udf`，Stage 0 仅保留 3% gross-mismatch 护栏。test-role
KANG_YONG 构建通过但不进入训练统计。

**验证**：34/34 `test_volume_*` 通过，覆盖 UDF 重置跳变、时间特征链式法则、完整
Carreau–Yasuda 应力散度、RCR 压力形式、detached ratio 不抵消 physics 梯度、两 backbone
容量匹配、旧路线回归和 resume schema 自动重建。额外完整瞬态 BC+PDE+EMA 单步反传
finite；173 例 Stage 0 本地全量 Gate（train138/test35、每例 81 帧）通过。

**Slurm 提交**：最终依赖链为 CPU Stage 0 build/Gate `11970` → 单卡 GPU preflight
`11971`（`afterok:11970`）→ 48-run 数组 `11972_[0-47%4]`
（`afterok:11971`，`ArrayTaskThrottle=4`，每 task 1 张 GPU）；`11970/11971` 已 exit 0，
正式数组已经启动。首个 Stage 0 `11964` 因把旧
`BC_Inlet` monitor 误设为硬真值而失败；修正为 UDF 解析真源 + 3% 诊断护栏。第二个
Stage 0 `11967` 因部分 bundle 裁剪人工延长段、瞬态 feature 点未先过滤到冻结目标域而失败；
修正后全库最少保留 1569 个目标域 interior 点、最低保留率 78.45%。旧依赖作业
`11965/11966`、`11968/11969` 均已取消，未启动训练。本地全量构建时另修复 resume 对旧
schema manifest 的识别，避免缺少 `eligible_interior` 时错误跳过。任一前级失败时训练不会
启动；训练完成后不自动运行 test35、WSS、工作簿回填或汇总。

**当前状态判断**：实现与提交已完成，正式数组已启动但尚无汇总结果。按用户要求本轮不等待作业结束；
待全部实验完成后再统一评估、汇总和分析。

## 2026-08-08｜WSS_PINN V4 设计 v1.2：对抗性审核修正（入口合同/RCR 单位/四臂）

**本次主要修改**：对 [V4 大重构设计方案](_archive/WSS_PINN/_archive/WSS_PINN_V4大重构设计方案_2026-08-08.md)
完成 v1.0→v1.1→v1.2 两轮修订，仅改设计与导航文档，未动代码/数据。v1.1 曾以“入口波形
全库统一”为由移除 `Q_in` 输入并把入口/出口 BC 以残差放入物理 loss；对抗性审核证实四个
关键问题，v1.2 修正为：

1. **入口合同**：边界审计确认 6 例 `A_udf≠A_mesh`（最大约 35.6%，如 ZHOU_KE_XUN 实际
   峰值 1.45494e-4 对名义 1.0727e-4 m³/s），“入口流量零方差”不成立；冻结
   `U_in(t)=Q_nom(t)/A_udf`、`Q_actual(t)=U_in(t)·A_mesh`，恢复 `Q_actual_peak` 输入，
   `A_udf` 落库 provenance，禁止用 `Q_nom/A_mesh` 监督；入口 BC 用带 buffer 的
   face-interior 点（入口—壁面共享 32–83 顶点/例，共享带归 no-slip）。
2. **RCR 单位**：UDF 递推按 `F_FLUX`（质量流量 kg/s）标定，残差改用
   `ṁ_out=ρ∫u·n dA`，混用体积流量差约 ρ≈1060 倍；冻结法向/符号并加单出口解析单元测试。
3. **波形与 gauge**：`w=6.491≠2π/0.8`，mod-T 重置点值/导数跳变，按 UDF 分段实现、
   时间 collocation 避开 `t≈0/T`、首尾连续性降为诊断；压力锚定收紧为四条件成立的
   raw Fluent gauge 约定，`P_out` 定义三选一冻结（施加 profile 与求解 face-average
   已有约百 Pa 差异证据）。
4. **归因与统计**：训练模式改 DATA/DATA+BC/BC+PDE-fixed/BC+PDE-EMA 四臂共 16 臂，
   分离边界监督与 PDE 贡献并正式记录“BC 进入 loss”的范围变更；`λ_pde=1` 降级为
   可复现基线表述，RCR 尺度改逐出口 `(R1+R2)·ṁ_c`；“标签残差地板”拆为参考尺度
   （时间 FD/RCR 时序）与敏感性参考带（散点空间残差，k=48/96 差 1–2.2×）；primary
   endpoint 改速度向量相对 L2，预注册 4 组 contrasts + Holm，test35 结论定性为探索性；
   新增 train138 内部 CV 边界（test35 永不进 fold）。

**对应代码/文档**：设计方案 v1.2、[路线真源](_archive/WSS_PINN/README.md)（状态区已同步）、
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：Stage 0 合同新增 `A_mesh/A_udf` 双落库与 6 例处置预注册、RCR 容错
解析（`udf-inlet.c`/`udf-inlet4.c` 命名差异与无空格写法）、ρ/Carreau 冻结、`RCR~A_out`
共线性审计、81 帧离散 RCR 残差（质量流量口径）参考尺度审计。正式实现仍未开始。

## 2026-08-08｜WSS_PINN V4 大重构设计：显式 RCR 条件与稳态/瞬态双路线

**本次主要修改**：新增
[WSS_PINN V4 大重构设计方案](_archive/WSS_PINN/_archive/WSS_PINN_V4大重构设计方案_2026-08-08.md)，
将新 V4 与已完成旧 field-v4 隔离。新方案不再输入 CFD 求解后才能得到的 outlet pressure，
改用 `Q_in`、入口/四出口面积和四组 `R1/R2/C`；统一比较 PointNet/PointNet++ 的
DATA、PINN-fixed 与 PINN-EMA-ratio，共 12 臂；动态臂采用 detached EMA
`L_data/L_phy` 每 50 step 更新，固定 `lambda_phy=1` 作为消融。保留 continuity、三分量 momentum 与 no-slip，准稳态 peak 和显式
`du/dt` 瞬态路线由配置切换。入口/RCR 条件只作输入，不进入入口/出口优化 loss。

**对应代码/文档**：[路线真源](_archive/WSS_PINN/README.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、`wss_pinn/AGENTS.md` 和
[项目级推进摘要](代码修改与实验推进记录.md)。本次仅修改设计与导航文档。

**推进到实验步骤**：完成 173/173 文件级覆盖核对：每例 81 帧 `u/v/w/p`、81 帧精确
匹配 `Q_in(t)`、solver `dt=0.005 s`、场导出 `Δt=0.01 s`、四组 RCR 与正值入口/四出口
面积均存在；val15 计划并回 train，形成 train138/test35。

**当前状态判断**：瞬态路线在源数据层面可行，但尚未通过跨时相 cell/coordinate 身份、
时间索引、pressure gauge 和 train-only stats 深审计。新 route/config/data/loss/trainer
均未实现，未提交训练，工作簿仍只保留冻结 V2/V3 结果。

---

## 2026-08-07｜PointNet 全历史实验补充 `y=ax+b` 线性拟合 R²

**本次主要修改**：在 WSS-min 评估链新增 `prediction=a×truth+b` 的一元线性拟合指标，物理/归一化空间均输出 pooled 与病例等权共享拟合线的 `R²_fit`、斜率 `a` 和截距 `b`；新增独立 `linear_fit_metrics.csv`。保留原始 R²、病例等权 R²、训练损失、checkpoint 与既有 Gate，不用拟合 R²替代绝对精度。同步修复面积映射诊断入口未调用 `model.eval()` 的问题；历史上由该旧入口生成的一行结果仍按原训练态预测精确复现并在回填器中显式隔离。

**对应代码/文档**：`training_wss_min/{metrics.py,evaluate.py}`、`training_wss_min/tools/{backfill_linear_fit_r2_workbook.py,evaluate_mapping_blocked_run.py}`、`training_wss_min/tests/test_linear_fit_metrics.py`、`training_wss_min/README.md`、[跨路线评估口径](../00-规范与记录/WSS跨路线评估与横向对比口径.md)和 `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。批量审计产物位于 `outputs/wss_min/linear_fit_r2_backfill_20260807/`。

**推进到实验步骤**：从总览 180 个单实验行追溯并去重为 175 个 checkpoint，按原 split/checkpoint 重新推理；旧 test16 使用终签 legacy 快照恢复已移出活动目录的 `WANG_DENG_FENG`。175/175 个来源均通过历史 raw R²复现，最大绝对差 `1.42e-8`；另按原表规则计算 4 个三种子均值行，共回填 184 个实验行。工作簿总览新增物理/归一化 12 列，汇总对比、教师汇报视图、RCR Oracle 和指标说明已同步并经 LibreOffice 全量重算。

**当前状态判断**：全历史回填完成，无缺失或估算值。`R²_fit` 仅表示允许统一线性缩放/偏移后的趋势一致性；汇报时必须同时查看 `a、b` 与 raw/case-balanced R²、MAE/RMSE。后续新评估会自动产出该指标，但现有实验排序和晋级结论不因本次补充自动改写。

## 2026-08-06｜V2 SAME5K-E7500 补训练/物理 loss 收敛图

**本次主要修改**：新增 `wss_pinn/tools/plot_v2_loss_convergence.py`，从八臂冻结
`epoch_progress.jsonl` 生成老师汇报版 train data / physics loss 图（V2 无 val，
仅 train；50-epoch 平滑；最后 10% 阴影）。

**对应代码/文档**：`wss_pinn/tools/plot_v2_loss_convergence.py`；产物
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/summary/loss_convergence/`
（`teacher_loss_convergence.pdf`、总览、配对 `convergence_curves`、E1–E8 分臂、
物理分量、summary csv/json）。

**推进到实验步骤**：历史 V2 完训后补可视化；未改训练/评估口径，未读 test35。

**当前状态判断**：八臂 last50 vs prior50 data 变化约 `-0.07%`–`+0.06%`，与 README
平台结论一致；PINN 臂最后 10% physics 变化约 `-0.22%`–`+0.10%`。可直接把 PDF/总览
发给老师。

## 2026-08-06｜WSS_PINN field-v4 Stage 0–1 全链完成并冻结 G-Raw

**本次主要修改**：实现独立 field-v4 数据合同与写路径护栏、病例等权
`S_field^cb`、固定 val15×5000 query、B0 病例盲 atlas、global/local × raw/PE
连续查询模型、outlet 双路径与 inlet-wall/wall-zone Gate、制造解/导数/support 探针、
分阶段 Slurm 提交、单种子排名 Gate、补种子矩阵和多种子/full-volume 汇总。

**对应代码/文档**：`wss_pinn/{config.py,train.py,evaluate.py,validation.py}`、
`wss_pinn/data/dataset.py`、`wss_pinn/models/point_models.py`、
`wss_pinn/tools/{build_field_v4,b0_atlas_v4,diagnose_field_v4,audit_boundary_field_v4,gate_stage0a_field_v4,gate_stage1_raw_field_v4,gate_stage1_single_seed_field_v4,finalize_stage1_field_v4,smoke_field_v4}.py`、
`wss_pinn/cluster/{preflight,run_experiment,evaluate_field_v4_fullvolume}.slurm`、
`wss_pinn/configs/volume_uvwp_peak_field_v4*/`、[路线真源](_archive/WSS_PINN/README.md)、
[`wss_pinn` README](../../wss_pinn/README.md)和
[已完成执行合同](_archive/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1_已完成_2026-08-06.md)。

**推进到实验步骤**：Stage 0-a/0-b Gate pass；B0 val15 完成；Raw `11315_[0-1]`、
PE `11318_[0-1]`、确认种子 `11321_[0-3]` 与 full-volume `11325_[0-5]` 全部完成、
exit code 0。补种子前 GPU preflight `11320` 通过；最终 26 tests、CPU smoke、原矩阵和
补种子矩阵静态 preflight 均 pass。训练/评估只使用 train123/val15，test35 读取数为 0。

**当前状态判断**：单 seed 排名为 G-PE `0.567612`、G-Raw `0.578311`、L-Raw
`0.588856`、L-PE `0.597688`。确认后 G-Raw/G-PE 为
`0.576116±0.007477 / 0.581208±0.012599`；PE 只在 seed1234 改善，病例 bootstrap
95% CI 跨 0，并触发 pressure 与最差 ILO 队列 2% 护栏。保留 G-Raw，G-PE No-Go，
停止 PE 扫频和 local decoder；Stage 2 未启动。本轮未运行 WSS、BC loss/input 或 PDE loss。

## 2026-08-06｜WSS_PINN 当前提示词换代与六臂提示词归档

**本次主要修改**：归档已完成的准稳态平滑场六臂预注册提示词，并以冻结诊断为真源
新建 Stage 0-a/0-b → Stage 1 执行提示词。新提示词不继承旧六臂、quasi-steady
momentum、2500 epoch、`lambda=1` 或自动正式提交合同，改为病例等权选模、B0 atlas、
边界 Gate 和纯监督表示矩阵。

**对应代码/文档**：
[当时执行提示词](_archive/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1_已完成_2026-08-06.md)、
[历史六臂提示词](_archive/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[归档索引](_archive/WSS_PINN/_archive/README.md)、[路线真源](_archive/WSS_PINN/README.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、`wss_pinn/AGENTS.md`、`docs/README.md`和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：完成执行提示词换代与导航同步；未改代码、配置、数据、split、
checkpoint 或训练产物，未启动训练。

**当前状态判断**：当前智能体应从新提示词进入 Stage 0-a/0-b；旧提示词仅供 V3 六臂
历史复盘，不再具有执行授权或活动合同效力。

## 2026-08-06｜体域 `u,v,w,p` 设计最终冻结与执行入口收口

**本次主要修改**：完成冻结前最后一轮科学审查。将新增 atlas、平滑 oracle、support
稳定性、梯度余弦和 outlet/RCR 链测量保持为 B 级证据；新增病例等权 `S_field^cb` 作为
唯一 checkpoint/early-stop 标量，规定 B0 atlas 永久进入主表，修正 Stage 0 依赖、
Stage 1 的 BC/PDE 隔离、Stage 3 起点和瞬态 physics 双路线。WSS 仍严格冻结为最终
checkpoint 的 downstream audit。

**对应代码/文档**：
[冻结版核心诊断](_archive/WSS_PINN/_archive/核心代码诊断与下一轮设计建议_2026-08-05.md)、
[WSS-PINN 路线真源](_archive/WSS_PINN/README.md)、[`wss_pinn` README](../../wss_pinn/README.md)、
`wss_pinn/AGENTS.md`、根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：下一轮科学合同冻结为 `FROZEN FOR IMPLEMENTATION v1.0`，可开始
Stage 0-a/0-b 的实现准备；未改代码、配置、数据、split、checkpoint 或训练产物，未启动训练。

**当前状态判断**：设计层面无剩余阻塞问题。Stage 0-a 通过后可进入纯监督 Stage 1；
Stage 0-b 仅阻塞 Stage 2/BC。任何 WSS 指标、test35 或 physics residual 均不得反选
checkpoint 或改变本轮架构顺序。

## 2026-08-06｜冻结 velocity→WSS，转入 `u,v,w,p` 优化主线

**本次主要修改**：将 Profile-Secant V3 的角色从“继续优化的 WSS 算法路线”收束为
冻结 downstream validator。核心诊断删除 WSS loss、直接 WSS 辅助监督、WSS 选模和
WSS 算法调优建议，改为 `u/v/w/speed/p` 多任务结构、区域采样、流量、pressure gauge、
压力梯度与压降的优化和验收；同时保留全壁面 overall/high-WSS/峰值限制，避免把冻结
验证器误写成无误差算子。

**对应代码/文档**：
[核心诊断与下一轮设计](_archive/WSS_PINN/_archive/核心代码诊断与下一轮设计建议_2026-08-05.md)、
[WSS-PINN 路线真源](_archive/WSS_PINN/README.md)、
[历史六臂预注册提示词](_archive/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、`wss_pinn/AGENTS.md`、
[CFD 适配说明](../../wss_mri_calculator/README_CFD_ADAPTATION.md)、根 `README.md`、
`docs/README.md`、`docs/实验设计总纲.md`和 [项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：完成文档级范围收束和下一轮实验路线重排；未修改代码、配置、
数据、split 或模型产物，未启动新训练，也未继续调 WSS 算法。

**当前状态判断**：`u,v,w,p` 是当前唯一模型优化主线。冻结 WSS 仅在四通道主 checkpoint
按 validation 指标确定后运行一次 sanity check，不得参与 loss、早停、checkpoint、
采样比例或架构选择。

## 2026-08-05｜WSS-PINN 核心诊断的对抗性复核与下一轮因果设计

**本次主要修改**：从不可压缩瞬态流、RCR/Windkessel 边界、WSS 一阶导数可辨识性、
病例级统计和可部署性出发，重构了核心诊断文档。将准稳态算子实现正确性与
瞬态 peak 方程适定性分开；确认 val15 因 batch 等权聚合导致固定单病例双倍权重；
明确当前 outlet 目标是 monitor-derived proxy，未证明等于真实 RCR boundary profile 或部署可得输入；
撤回按 x/y/z 强行平衡 momentum 和用三例散点 residual 生成逐点 floor 的建议。新增
test35 development exposure、配对 case-bootstrap、队列异质性、边界 face 面积采样、
WSS 直接/近壁切向监督和分阶段 Go/No-Go。

**对应代码/文档**：
[对抗性修订后的核心诊断](_archive/WSS_PINN/_archive/核心代码诊断与下一轮设计建议_2026-08-05.md)、
[WSS-PINN 路线真源](_archive/WSS_PINN/README.md)、
[历史六臂预注册提示词](_archive/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、
`wss_pinn/AGENTS.md`、`docs/实验设计总纲.md`和 [项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：完成 Stage 0 前的文档级科学审查和新实验路线预设计；未改代码、
配置、split、checkpoint 或评估产物，未启动新训练。只读复核了 V3 model/loss/dataset/train、
六臂 epoch 日志、test35 逐例 CSV、V2 residual audit 和代表病例 UDF。

**当前状态判断**：当前六臂仅能支持“geom 正增量、BC 速度近中性、当前
quasi-steady PDE 组合对 speed No-Go”的历史结论；不支持全局 latent 根因已定、
物理普遍无用或已可靠恢复 WSS。新训练前必须先修边界/选模/确认集合同，然后执行
global/local × raw/PE 的 2×2 无 PDE 消融；只在最佳监督设计上再验证 continuity、瞬态或弱形式 physics。

## 2026-08-05｜准稳态平滑场 V3 六臂完训、三 checkpoint 评估与结论回填

**本次主要修改**：从 Fluent 二进制 case 恢复 173 例真实 wall/inlet/four-outlet
边界，建立 train123/val15/test35 与 train-only 统计；实现无 3NN/IDW 的条件
PointNet 平滑 query 场、DATA/BC/PDE 分组损失、统一 validation checkpoint、六臂配置、
静态/GPU Gate、双节点提交和机器可读监控；补齐 pooled 回归指标、入口/出口边界、
预测方差/塌缩诊断，并将旧八臂汇总器扩展为 V3 六臂主表、九组增量、逐病例和收敛
汇总；新增可复跑的老师汇报损失绘图器，生成六臂 data/physics loss 总览、每臂全程与
最后 10% 放大、BC/PDE 子项和 9 页 PDF；新增汇报工作簿生成器，把主指标、九组增量、
checkpoint敏感性、loss收敛和210行逐病例结果整理为7张表并嵌入关键图。入口流量使用 exact peak
`vf-in-rfile.out` 实测值除真实网格面积；6 例 UDF 陈旧分母保留 warning，不伪造成
Gate 失败。wall 坐标审计改为 sidecar→全部 Fluent wall，并允许同表面重剖分。

**对应代码/产物**：`wss_pinn/{config.py,train.py,evaluate.py,losses.py}`、
`wss_pinn/{models,data,tools,cluster}/`、
`wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3/`、
`data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/`、
`outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_*/` 与
`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/`；损失绘图器为
`wss_pinn/tools/plot_v3_loss_convergence.py`，图件位于
`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/loss_convergence/`；工作簿生成器为
`wss_pinn/tools/build_v3_teacher_workbook.py`，结果为
`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/WSS_PINN_V3_六臂核心实验与指标汇报.xlsx`。

**推进到实验步骤**：boundary Gate 173/173、25 项测试、三模式 CPU smoke、六臂静态
preflight 和 master4+node04x2 CUDA dry-run 全通过；前 30 分钟 31/31 快照 healthy。
master Slurm `11301_[0-3]` 与 node04 PID `1241185/1241335` 对应六臂均完成 2500
epoch / 155000 step，三类 checkpoint 6/6 齐全；test35 全严格体域评估 18/18 完成，
主 checkpoint 固定 `best_validation_data`，test35 未参与训练、选模或续训判断。

**当前状态判断**：geom 是稳定正增量：E4−E1、E5−E2、E6−E3 的 speed R²_cb
为 `+0.1919/+0.1977/+0.1942`。BC 对总体速度近中性（case-balanced 小幅正、pooled
小幅负），但改善入口流量与出口压力。PDE 将 continuity/momentum residual 降低约
89–90%/60–62%，却使 E3−E2、E6−E5 的 speed R²_cb 分别下降 `0.0502/0.0537`，判定
为准稳态正则与瞬态 peak 标签竞争，而非训练不足。最后 10% validation 已小幅变差，
不满足续到 5000 的条件；near-wall speed R² 六臂全负，不形成 WSS 结论。结果真源见
[`summary/README.md`](../../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/README.md)。

## 2026-08-04｜Profile-Secant V3 35 例全壁面推理、采样与穿模审计

**本次主要修改**：新增 Profile-Secant V3 全壁面可恢复执行链，按既有 test35
原始壁面点数均衡成 8 个 CPU 分片，依次生成基础 V4 缓存、depth-3 剖面/secant
诊断和冻结高尾校准结果。扩展纯模型导出器，新增 132.8 万点同点 hexbin 散点、
最佳/最差高 WSS 空间图、单点内部支撑图、全病例壁面范围穿模风险图和全量 VTP。
历史 `profile_secant_v3/` 明确降级为 test35 × 1200 预览，未覆盖原产物。

**对应代码/文档**：
`wss_mri_calculator/src/run_profile_secant_v3_fullwall.py`、
`wss_mri_calculator/src/run_frozen_wss_compat.py`、
`wss_mri_calculator/viz/export_profile_secant_v3_visualization.py`、
`wss_mri_calculator/viz/visualize_v4_{single_target_sampling,penetration_risk}.py`、
[全壁面结果与复现](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/README.md)、
[V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：Profile-Secant V3 模型 SHA256 保持
`c1e53af5…3a02`；test35 35/35、`1,328,017` 点完成，0 failure。正式结果 JSON /
predictions SHA256=`31a4bdca…d7e89` / `9aaa80ad…bdbf`；最佳/最差全量 VTP 源同点
分别为 67,760 / 18,551，STL 顶点映射覆盖率均 100%。

**当前状态判断**：full-wall overall R² 均值=`0.96037`、pooled overall
R²=`0.96790`、pooled high-WSS R²=`0.94405`、逐病例 high-WSS R² 均值
=`0.77355`。sampled 与 full-wall 的整体/平均高 WSS 结论稳定，但全壁面平均峰值
低估=`15.81%`，sampled `9.31%` 不能再作为全量峰值结论。最佳病例穿模审计拦截
2,450 个门控前跨壁风险候选，最差病例严格门控前风险为 0；两者门控后保留风险均为 0。

## 2026-08-04｜体域 PINN 下一阶段准稳态平滑场六臂执行交接

**本次主要修改**：将用户确认的下一阶段实验固化为可直接复制到新窗口的执行提示词。
路线使用 Fluent 瞬态 peak `u,v,w,p` 标签和 Carreau–Yasuda 准稳态 PDE 正则，采用
无 PointNet++/3NN/IDW 的条件 PointNet 平滑坐标场，并用
`xyz/xyz+geom × DATA/DATA+BC/DATA+BC+PDE` 六臂拆分 BC、PDE 与几何特征增量。

**对应代码/文档**：
[准稳态平滑场六臂实验交接（已归档）](_archive/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[体域 PINN 路线真源](_archive/WSS_PINN/README.md)、
[`wss_pinn` 代码入口](../../wss_pinn/README.md)、`docs/README.md` 和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：本次只冻结科学合同、边界资产 Gate、train123/val15/test35、
六臂矩阵、loss 分组、2500→5000 同预算续训条件、统一 validation checkpoint、正式
提交与监控要求；未修改训练代码、未构建新边界 sidecar、未提交 Slurm。

**当前状态判断**：handoff ready / implementation pending。下一智能体必须先证明真实
peak inlet/outlet 坐标与目标可恢复；Gate 通过后已获授权推进实现、测试、GPU
preflight、正式六臂提交、完训评估和文档收口。若 Gate 失败，禁止伪造 BC 或把只有
no-slip 的实验称为完整 DATA+BC。

## 2026-08-04｜Profile-Secant V3 纯 quicklook 与最佳/最差病例 VTP

**本次主要修改**：新增 Profile-Secant V3 专用可视化导出脚本和独立结果目录，图件
只读取 `high_tail_wss_vec`，不混入 V4 final 预测。生成 test35 指标汇总、最佳/最差
病例空间图、高 WSS 尾部图、总览联系表，以及与既有 `06_postview_vtp` 同结构的两份
ParaView 面片包。通用三联图工具增加可选预测标签，默认行为保持不变。

**对应代码/文档**：
`wss_mri_calculator/viz/export_profile_secant_v3_visualization.py`、
`tools/cfdpost_cloud_export/plot_stl_mapped_triptych.py`、
[纯模型可视化入口](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3/README.md)、
[V4/Profile 方法说明](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/README.md)、
[CFD 适配说明](../../wss_mri_calculator/README_CFD_ADAPTATION.md)。

**推进到实验步骤**：未训练或调参；直接读取冻结的 test35 × 1200 Profile-Secant V3
predictions。按逐病例 high-WSS R² 选择最佳 `ILO/LI_YOU_ZHI-0/before`
（`0.9560`）和最差 `AG/slow/ZHANG_WEI_XIAN`（`0.2381`），将同点标量 Gaussian
映射到病例 STL，并输出主 VTP、源 CSV、映射报告、预览图、色标和 manifest。

**当前状态判断**：纯 quicklook 和两份 VTP 已可直接使用。最佳病例 VTP 映射覆盖率
为 `99.51%`，最差病例为 `99.88%`；VTP 仅用于展示，正式 R² 仍以冻结同点 JSON/CSV
为准。脚本与通用三联图工具 `py_compile` 通过。

## 2026-08-04｜体域 PINN V2 零重训 residual 根因验证与老师论文对照

**本次主要修改**：新增冻结 checkpoint 的物理诊断工具并完成三项只读审计：test35
exact support / 独立体域 / jitter query residual；AG/AAA/ILO 各 1 例 CFD 真值的
准稳态与加入 `du/dt` 后 residual；当前 V2 到 Liao 2025/老师源码 NMAE 口径的指标
桥接。同步纠正路线文档中把 SAME5K 低 residual 当作连续场物理闭合证据的旧表述，
并补齐老师论文和当前代码的设计差异。

**对应代码/文档**：`wss_pinn/tools/diagnose_v2_physics.py`、
[V2 路线真源](_archive/WSS_PINN/README.md)、[`wss_pinn` 代码入口](../../wss_pinn/README.md)、
[老师论文与 V2 对照](../paper_reproduction/papers/hemodynamics_pointcloud_pinn/README.md)、
`outputs/wss_pinn/audits/volume_uvwp_peak_same5k_e7500_v2_residual_diagnosis/`、
根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`。

**推进到实验步骤**：未重新训练。固定使用
`VF-PNPP-XYZG-PINN-SAME5K-E7500-s1234-v2/last.pt`；网络审计覆盖 35/35 test，
每例 512 点；CFD 审计覆盖三域各 1 例、每例 256 个 strict-core 点，并用 `k=48/96`
检查局部二次拟合敏感性。解析三例 Fluent journal，确认 solver dt=`0.005 s`、中心
差分间隔=`0.02 s`；解析老师论文 PDF 和现有代码快照。

**当前状态判断**：SAME-IDW 根因已确认：独立 query 的 continuity/momentum/速度
梯度 RMS 约为 exact support 的 `453×/331×/187×`；`1e-4` 微扰下速度仅变化
`1.84e-5 m/s`，动量却放大 `120×`。CFD `k=96` 加时间项后 residual 均值
`1737→1494 Pa/m`，说明准稳态缺项真实但只解释部分误差，且绝对值受二阶导数邻域
敏感性限制。按老师源码 global-range NMAE，V2 可得约 `2.18%` 而 speed R² 仍仅
`0.2380`，确认原文 NMAE/FR-PD R² 不能直接证明逐点速度更准。下一轮训练前应先修
独立 collocation/连续 query、可观测 inlet/outlet BC 与瞬态方程，不再追加 epoch。

## 2026-08-04｜Profile-Secant V3 推荐模型口径纠正

**本次主要修改**：根据用户明确的选择目标“病例总体和平均高 WSS 精度”，纠正此前
把 V4 final 写成推荐版本的口径。现统一将 Profile-Secant V3 定为当前推荐结果模型，
用于主结果、可视化和后续总体/平均高 WSS 分析；V4 final 改为冻结基线和消融对照。
逐病例严格目标未达的限制保持不变。

**对应代码/文档**：[V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)、
[CFD 适配说明](../../wss_mri_calculator/README_CFD_ADAPTATION.md)、
[V4/Profile 方法说明](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/README.md)、
[完整结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/RESULTS.md)、
[Profile-Secant V3 推荐结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md)、
根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`、
[高值区域预测优化方案](_archive/WSS最小化/WSS高值区域预测优化方案.md) 和 [体域 PINN 路线](_archive/WSS_PINN/README.md)。

**推进到实验步骤**：只纠正文档中的模型选择与报告口径；未重新训练、未重新推理、
未修改模型、预测、split 或哈希。选择依据仍为已落盘的 train138 grouped OOF、固定
holdout65 和 test35 结果。

**当前状态判断**：Profile-Secant V3 的 test35 整体/pooled high-WSS R² 为
`0.9617/0.9415`，优于 V4 final 的 `0.9561/0.9282`，因此符合用户当前主目标并作为
推荐结果模型。其严格双目标仅 4/35 病例达标，推荐使用不等于逐病例可靠保证。

## 2026-08-04｜`wss_pinn` 当前主线根层化与旧 WSS-target 源码归档

**本次主要修改**：将活动峰值体域 `u,v,w,p` 路线从
`wss_pinn/volume_field/` 提升到 `wss_pinn/` 根包，统一训练、评估、数据、模型、
物理、工具和 Slurm 入口；把已停止的直接 WSS/F0–F2 实现、配置和测试移入源码
归档。保留当前路线仍复用的 raw CFD 读取、抽取后的坐标对齐和通用写盘/哈希工具，
并将这些活动依赖纳入静态 preflight 的实现哈希。

**对应代码/文档**：`wss_pinn/config.py`、`wss_pinn/train.py`、
`wss_pinn/evaluate.py`、`wss_pinn/{data,models,physics,tools,cluster}/`、
[`wss_pinn` 代码说明](../../wss_pinn/README.md)、
[`wss_pinn` 归档索引](../../wss_pinn/archive/README.md)、
[历史源码归档](../../wss_pinn/archive/wss_target_v1_20260730/README.md)、
`wss_pinn/AGENTS.md`、[体域 PINN 路线真源](_archive/WSS_PINN/README.md)及其历史文档归档。

**推进到实验步骤**：只做工程结构收敛和入口迁移；未启动训练、未改配置数值、
未改数据 split、未改模型/loss/评估口径，也未移动既有数据和输出。

**当前状态判断**：活动代码已直接位于 `wss_pinn` 根层；旧路线有可浏览源码归档，
不再与当前入口混放。现有 v1/v2 实验结论和输出路径保持不变。

## 2026-08-04｜高 WSS Profile-Secant V3、可辨识性上限与中文文档同步

**本次主要修改**：将冻结 V4 final 之后的高 WSS 增量研究统一收口为中文；明确
Profile-Secant V3 是面向总体和平均高 WSS 精度的当前推荐结果模型，V4 final 保留
为冻结基线，并新增 2026-08-04 test35 高 WSS 对比图。把 pooled/平均达标与逐病例
严格未达标分开报告，补充 oracle 上限、
近壁采样深度相关性和已否决路线，避免把整体 R² 高误写成每个病例高 WSS 可靠。

**对应代码/文档**：[V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)、
[V4 方法说明](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/README.md)、
[V4 完整结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/RESULTS.md)、
[Profile-Secant V3 结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md)、
[高 WSS quicklook](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/README.md)、
根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`、
[高值区域预测优化方案](_archive/WSS最小化/WSS高值区域预测优化方案.md) 和 [体域 PINN 路线](_archive/WSS_PINN/README.md)。

**推进到实验步骤**：未重新训练、未做 test35 调参。直接读取已冻结的
`calibrator_profile_secant_high_tail_anchor10_v3.joblib` 和 prediction archive，生成
`15_profile_secant_high_tail_comparison.png/json`。相关 12 项单元/合同测试既有记录
为通过，重复推理既有记录为逐字节一致。

**当前状态判断**：test35 整体 R²=`0.9617`、pooled high-WSS R²=`0.9415`、
逐病例 high-WSS R² 均值=`0.7721`、high-WSS NRMSE=`0.1583`、平均峰值低估
=`9.31%`。但仅 6/35 病例 high-WSS R² `≥0.90`，仅 4/35 同时满足峰值低估
`≤10%`。任意单调 oracle 仍有 13/35 不达标，现有输入上的后校准优化已接近信息上限；
严格提升需要更靠近壁面的速度层或更高近壁分辨率。当前主结果使用
Profile-Secant V3，V4 final 只作冻结对照。

## 2026-08-03｜峰值体域 SAME5K-E7500 v2 8/8 完训与速度精度审计

**本次主要修改**：完成第二轮峰值体域 `u,v,w,p` 八实验的 SAME5K 主协议、
固定 5k support→full-volume 副协议、三 checkpoint 敏感性、逐病例和分区域审计；
将路线真源、Agent 边界、代码 README、项目入口与实验总纲从 running 更新为
completed，并明确整体速度精度、近壁短板和当前模型选择。

**对应代码/文档**：[峰值体域 PINN 路线真源](_archive/WSS_PINN/README.md)、
`wss_pinn/AGENTS.md`、[`wss_pinn/README.md`](../../wss_pinn/README.md)、
根 [`README.md`](../../README.md)、
`docs/README.md`、`docs/实验设计总纲.md`、
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/runs/`。

**推进到实验步骤**：GPU preflight `11137` 与训练 array `11138_[0-7%4]` 均
8/8 completed；每臂 7500 epoch / 517,500 step，8 份 epoch 日志均为 7500 行，
SAME5K/full-volume × `best_data/best_total/last` 共 48 份 evaluation JSON 完整。
最低训练 loss 均位于 epoch 6815–7487，三 checkpoint 的 SAME5K speed R² 极差仅
`0.0006–0.0230`，训练已基本平台。

**当前状态判断**：V2 速度主锚点冻结为 PointNet++ + `xyz+geom` + PINN / `last@7500`；
其 SAME5K/full-volume speed R²=`0.2380/0.1820`、MAE=`0.1764/0.1863 m/s`，
30/35 病例 SAME5K speed R² 为正。PINN 四对聚合 `u/v/w/speed/p` R² 均提高，
但 PointNet++ 两对 speed 增益仅 `+0.0705/+0.0317` 且逐病例约一半获益；near-wall
speed R² 仍全部为负。下一步不再只延长 epoch，优先诊断 5 个负 R² 病例与近壁
标记/采样/损失，并在独立确认集或多 seed 上复核。

## 2026-08-03｜velocity→WSS V1–V4 总跟踪建立并同步 V4 冻结结论

**本次主要修改**：为 `wss_mri_calculator` 建立 V1–V4 唯一实验总跟踪页，明确
纯物理前向、train-only 全局标量和 V4 冻结诊断校准的监督边界；将 CFD 适配说明中
`alpha≈1.2` 改为 V1/V2 历史诊断，并同步根 README、docs 索引和实验总纲的当前状态。

**对应代码/文档**：
`wss_mri_calculator/experiments/README.md`、
`wss_mri_calculator/README_CFD_ADAPTATION.md`、根 `README.md`、
`docs/README.md`、`docs/实验设计总纲.md`、V1–V4 各实验目录及冻结清单。

**推进到实验步骤**：V1–V4 均为 frozen。统一 test35 × 1200 下，V4 physics
raw/scaled R² mean=`0.90788/0.93612`；V4 final 为
`0.95607/0.95844`，raw p05/min=`0.92391/0.90025`，35/35 病例优于 V3。
train138 grouped OOF 与 development73→holdout65 的 raw R² mean 分别为
`0.95493/0.95790`。

**当前状态判断**：V4 final 是当前推荐 velocity→WSS 方法；V1/V2/V3 保留为冻结
演进证据，禁止 post-test tuning。下一阶段应转向新外部队列、网格/时间步鲁棒性、
分支感知归属和不确定性，不继续通过增加射线站点或多项式阶数追分。

## 2026-08-02｜峰值体域 SAME5K-E7500 v2 实现并提交

**本次主要修改**：按用户确认的简单汇报口径新增独立八臂 v2：train138/val0/test35，
严格体域 random5000，support/query 使用相同索引，每 epoch 重采样，固定 7500 epoch。
扩展数据集和配置 Gate 以兼容 SAME，同时保留旧 SEP；新增 400/1000/2500/5000/7500
里程碑 checkpoint；评估新增 SAME5K 与固定 5k support→full-volume 双协议，并补充
case-balanced 的逐目标 R²/MAE/RMSE、三分量 momentum 与 wall P95/max。Slurm
preflight/训练均设 `--time=0`，实查 GPU 分区 `MaxTime=UNLIMITED`。

**对应代码/文档**：`wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/`、
`wss_pinn/{config.py,data/dataset.py,train.py,evaluate.py}`、
`wss_pinn/cluster/`、[路线真源](_archive/WSS_PINN/README.md)、
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/submission.json`。

**推进到实验步骤**：`test_volume_*` 20/20 通过；新矩阵静态 preflight pass；正式提交
GPU preflight `11137`，训练 array `11138_[0-7%4]` 以 `afterok:11137` 等待释放。

**当前状态判断**：GPU preflight `11137` 8/8 completed；array `11138` running。首批
四个 PointNet 臂连续观察 30:35，约 22,432 条已记录 step 全部 finite，四份 stderr
均为空，无 NaN/Inf/OOM/Traceback/CUDA error/restart；400/1000 里程碑与定期
checkpoint 写盘正常。梯度裁剪率 data-only 约 50%–54%、PINN 约 90%–95%，作为后续
诊断保留。机器可读快照：
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/monitor_30min.json`。

## 2026-08-02｜峰值体域 uvwp 八臂完训、结果图与 15,000 epoch 候选协议

**本次主要修改**：完成 array `11128_[0-7]` 的 8/8 run、checkpoint、日志与
24 份 evaluation JSON 完整性审计；新增可复现结果汇总脚本，生成 `best_total` / test35
配对 CSV/JSON、收敛统计和两张 PNG/SVG。将路线真源、代码 README、项目入口与总纲
从 running/pending 更新为 completed/audited。按用户修正把下一轮预算记录为
`max_epochs=15000`（约 1,035,000 step）的硬上限，并预注册 train138-only fixed
monitor、data fidelity/Pareto 早停边界；当前仅为候选协议，未实现、未提交训练。

**对应代码/文档**：
`wss_pinn/tools/summarize_results.py`、
`outputs/wss_pinn/volume_uvwp_peak_v1/summary/`、
[峰值体域 PINN 结果真源](_archive/WSS_PINN/README.md)、
[`wss_pinn/README.md`](../../wss_pinn/README.md)、根 `README.md`、
`docs/README.md`、`docs/实验设计总纲.md`。结果 manifest SHA256：
`e58be163…3781de8`。

**推进到实验步骤**：首轮八实验 8/8 completed；每臂 400 epoch、27,600 step，
配对初始化哈希一致、`warm_start=false`，无 NaN/Inf/OOM/traceback。四个 PINN 配对的
continuity/momentum/wall RMS 均下降，pressure R² 均提高；PointNet + `xyz+geom` 是
唯一 `u/v/w/speed/p` 全部提高的配对。静态图与机器可读表已落盘。

**当前状态判断**：首轮结果 completed / audited，但不能宣称 PINN 全面优于
data-only，也不能宣称速度场已充分收敛。15,000 epoch 是下一轮硬上限，不是本轮已经
执行的预算；早停工程实现和新配置仍 pending，正式长训练未提交。

## 2026-08-01｜schema-v2 数据 Gate 通过并启动 4-GPU 八臂训练

**本次主要修改**：按 4×RTX 4090 集群资源把训练数组并发上限改为
`0-7%4`。提交 CPU 数据构建 Job `11122`、deep-source 审计 Job `11123` 和自动
launcher `11124`；173/173 schema-v2 sidecar、注册速度、严格体域压力 gauge、
壁面重复行、split/manifest/train-only stats hash 全部通过。首次 GPU preflight
`11125` 发现 CuBLAS 确定性环境缺失后，在正式训练开始前主动取消 `11125/11126`，
补充 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，重跑 40/40 单测与静态 preflight 后提交
GPU preflight `11127` 和正式 array `11128_[0-7%4]`。

**对应代码/文档**：`wss_pinn/cluster/{preflight,run_experiment}.slurm`、
`wss_pinn/{train.py,cluster/submit_matrix.py}`、
`data_wss_pinn/volume_uvwp_peak_v1_train138_test35/`、
`outputs/wss_pinn/audits/volume_uvwp_peak_v1_train138_test35/`、
`outputs/wss_pinn/volume_uvwp_peak_v1/submission.json`、[当前路线](_archive/WSS_PINN/README.md)。

**推进到实验步骤**：build `11122` 与 deep audit `11123` completed；aggregate
manifest SHA256 `e425a657…54a1bf`，field stats SHA256 `9c15ab2b…905f46`，最终
data Gate report SHA256 `d5a86b8a…386e78`。GPU preflight `11127` completed
`0:0`；训练数组 `11128` 正在按最多 4 卡并发执行。

**当前状态判断**：正式训练前 30 分钟监控通过。无 traceback、NaN、non-finite
loss、OOM、任务重启或 GPU 过热；三条 data-only 已完成 400 epoch 与三 checkpoint
评估，PointNet 两条 PINN、PointNet++ xyz PINN 和 PointNet++ xyz+geom data-only
继续运行，最后一臂因 `%4` 上限正常等待。评估存在 mmap 只读数组转 tensor 的非致命
warning；代码没有原地写入，数值安全，待本轮结束后统一消除以保持当前运行代码口径。

## 2026-08-01｜WSS-PINN 重构为峰值体域 `u,v,w,p` 八实验

**本次主要修改**：按用户重新确认的边界，停止旧“直接 WSS 目标 + 辅助体域分支”
路线，新增独立 `volume_uvwp_peak_v1`：只做 peak 时刻，PointNet / 纯 PointNet++ ×
xyz/xyz+3 几何特征 × data-only/PINN 共 8 个实验；四通道 `u,v,w,p` 全部真值监督，
PINN 从随机初始化的第一个 step 同时加入 continuity、完整三分量 Carreau–Yasuda
momentum 和壁面 no-slip，不从 data-only 热启动。实现可微 query-coordinate 路径、
完整变黏度应力散度、逐 step/epoch data loss 与 physics loss 日志、物理单位评估和
同 run resume 保护。

数据合同升级为 schema v2：速度向量与坐标一起旋转；识别三个 bundle 的体表壁面重复
行并从 support/query/PDE、压力 gauge 和归一化统计排除；压力按病例严格体域固定均值
中心化；几何只保留 `abscissa_norm/local_radius/signed_log1p(curvature)`，曲率裁剪和
统计只来自 train138 严格体域。补充 manifest/hash/split/train-membership 数据 Gate 和
合成回归测试。旧 WSS-target 文档移入 `_archive/wss_target_v1_20260730/`，旧源码以
Git commit `bca002025d40b290d171570e6470f484fd4feec7` 追溯，不删除旧数据或输出。

**对应代码/文档**：`wss_pinn/`、
`wss_pinn/configs/volume_uvwp_peak_v1/`、`wss_pinn/tests/test_volume_*.py`、
`wss_pinn/{AGENTS.md,README.md}`、[当前体域 PINN 路线](_archive/WSS_PINN/README.md)、
[旧路线归档](_archive/WSS_PINN/_archive/wss_target_v1_20260730/README.md)、
`outputs/wss_pinn/audits/volume_uvwp_peak_v1_preflight/report.json`。

**推进到实验步骤**：`compileall` 通过，`wss_pinn/tests` 完整单元测试 40/40 通过，
八臂静态配对 preflight 为 pass；默认提交器 dry-run 返回 `jobs=[]` 和
`formal_training_submitted=false`。未构建全量 schema-v2 sidecar，未运行 GPU 八臂
dry-run，未提交正式训练。

**当前状态判断**：implementation ready / static preflight passed / new data Gate
pending / formal training not submitted。下一步是 CPU 集群构建 173 例 schema-v2
sidecar 并做 deep-source audit；数据 Gate 通过后仍需用户再次授权正式训练。

## 2026-08-01｜根 README 改以 WSS-min + wss_mri_calculator 为主入口

**本次主要修改**：重写仓库根 [`README.md`](../../README.md)：把当前主攻从旧 `pipeline/` 叙事改为 **WSS-min（预处理+训练）** 与 **`wss_mri_calculator`（CFD velocity→WSS）**；补上目录表、四阶段/训练/CFD 快速命令、v4 数据与 LSA2+H2+`log(local_radius)` 锚点、adaptive-CV v1 / multiscale v2 状态；去掉失效的本机绝对路径链接；旧 `pipeline/` / V3 / baseline 收为次级入口。

**对应代码/文档**：[`README.md`](../../README.md)；本推进记录。

**推进到实验步骤**：文档导航（无新训练/无新批量 WSS）。

**当前状态判断**：新人从根 README 可直接落到 `pipeline_wss_min`、`training_wss_min`、`README_CFD_ADAPTATION.md` 与两条实验目录；细节仍以各子 README / 矩阵文档为准。


## 2026-08-05｜WSS-PINN 核心代码维护性精简

**目标**：参考 `wss_pinn/PIPN-QN Code/` 的直接组织风格，减少当前体域 PINN 核心
入口中的重复分支和过重测试，同时保留 JSON 配置、矩阵 preflight、数据 Gate、
checkpoint/resume 与 Slurm 提交能力。

**代码修改**：

- `config.py`：新增推荐名 `ExperimentConfig`，用 route 合同表集中声明 mode、架构、
  激活和冻结 split；`VolumeExperimentConfig` 继续作为兼容别名。
- `models/point_models.py`：推荐构造入口改为 `build_model`，旧
  `build_volume_model` 保留别名。
- `losses.py`：统一 V1/V2/V3 的数据项与 PDE 计算，只在边界组合和注册权重上分 route；
  `compute_losses` 从 `(losses, diagnostics)` 简化为直接返回 `losses`。
- `train.py`：删除没有汇总脚本读取的逐 step 剪切率、黏度、残差 RMS 重复诊断；
  继续保存逐 step/epoch loss、初始化证据、best/last/milestone checkpoint 和同 run resume。
- `evaluate.py`：将重复的点云分块预测、near-wall/core 指标、V3 边界和物理 residual
  拆为小函数，主病例循环保持线性可读；输出 schema 不变。
- `cluster/submit_matrix.py`：默认从 `matrix.json` 解析并校验实验顺序，再生成 Slurm
  使用的 config list；`--config-list` 仅保留为已完成实验兼容入口。
- 测试由 7 文件缩为 `test_volume_{config,data,models,physics}.py` 四文件，共 18 项。
  删除大型 synthetic data Gate 和多轮 resume 集成 fixture；真实数据审计与逐配置 GPU
  dry-run 仍由 preflight 负责。

**验证**：18/18 核心测试通过；V1/V3 静态 preflight 分别 4/4、2/2 配对组通过；
V1 `pinn` 与 V3 `data_bc_pde` synthetic forward/backward 均得到有限 loss。没有提交
新训练，也没有改写既有实验结果。
