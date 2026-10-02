# X5D 主线与新数据 · 实验跟踪

> 2026-09-24 由 [WSS V5 训练实验跟踪（历史卷）](../00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md) 拆出：§33（learning curve）、§35（v5.2 基线重训）原文逐字搬入、节号不变，历史卷原处留指针。本线后续结果续写在本文件；新开一条实验线时取全局下一个节号（当前已用到 §35）。汇报口径同历史卷文首：归一化（log_z）空间与物理（Pa）空间并列。底座 X5D_v51 的定义与五 seed 结果在历史卷 §28；v5.2 数据构建见 [数据回收 README §4.9](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据回收记录_2026-09-20.md)。

## 33. Learning curve：数据量是否仍是瓶颈（2026-09-20 提交；预检 15301 → 队列 15302，afterok；首次预检 15299 因冻结副本旧预检工具写死 C1 锚点而 v5.0 视图已删 5 s 即败，已给副本工具加 `--anchor-run` 指向波 2a 折模型）

**起因**：用户 2026-09-20 讨论"新 STL 泛化 + 回收被排除病例"后拍板先做 learning curve，再决定投多少 CPU 重跑 CFD。配套名单见 [数据回收名单_可直接入库与需重跑CFD_2026-09-20.md](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据回收名单_可直接入库与需重跑CFD_2026-09-20.md)（A 层 86 单元可直接入库、B 层 14+1 需重跑 CFD、C 层 7 暂不投入）。

**设计（启动前冻结）**：
- 折 = 既有 cv3_v51 三折（患者分组，test34 unused）；配方 = X5D_v51 折臂配置逐字复制（`configs/wss_v51_wave2a_20260916/X5D_v51_f{k}_s1234.json`），只改 seed / 配对参考链 / split / 目标统计 / 特征统计。
- 训练例分数 25 / 50 / 75 / 100 %：每个 (fold, seed) 内**嵌套**（25 ⊂ 50 ⊂ 75），按 AG/AAA/ILO 分层，重复/相关组同进同出；每个子集自己重算 log_z WSS 统计与特征 z-score（模拟"只有这么多数据"）。子集 split 与统计在 `data_wss_v5/views_v5_1/wss_min_view_v1/cv3_v51_lc/`，特征统计在 `experiments/wss_learning_curve_20260920/feature_stats/`。
- seed 1234 / 7 / 2025；LC100_f{k}_s1234 不重训，直接引用波 2a 折模型（external_reference_runs）。共 **33 训练 + 66 评估**：n_train 22–24 / 44–46 / 68 / 90–92。
- 队列顺序：seed 1234 的 25/50/75 九臂先跑（约 4 h 后可读第一条曲线），再 seed 7、seed 2025，最后 LC100 的 s7/s2025 六臂。
- 读法：留出折 Pa R²_cb 按三折合并（每例一次），每个分数 9 个配对读数，拟合 R² 对 log(n_train) 的斜率；射流子集（真值 p99 > 40 Pa）从逐例指标单独读；单读数不排名；test34 不用。
- **判据（预注册）**：75→100 % 段斜率仍 ≥ +0.02/翻倍 → 数据量仍是瓶颈，B 层 CFD 重跑值得；射流子集斜率显著大于整体 → 优先补射流例；斜率已平 → 转模型/目标侧。

**执行**：工具 `tools/prepare_wss_learning_curve.py`（CPU，2 min，已同步进冻结副本 `GNN_v51_frozen/`）；预检 `cluster/preflight_wss_learning_curve.slurm`（1 卡，control LC50_f0_s1234，smoke LC25_f1_s7 + LC100_f2_s2025）；队列 `cluster/run_wss_learning_curve.slurm`（4 × 4090 × 2 槽）。全部 `cd` 到 `GNN_v51_frozen`（与波 2a 折模型逐位同源码，16 个 .py 指纹核对一致），主仓库可随便改。**注意**：4 张卡当前被另一用户的非 Slurm 进程占用（约 90 % 利用率、700 MB），实际墙钟会比 wave2a 的 5 h/3 臂慢。

**状态**：预检 15301 通过（8 min；锚点 = 波 2a X5D_v51_f0_s1234 重评 7518 个数值最大差 3.6e-5，Pa R²_cb 0.73658 复现；33 臂前向/AMP 检查全过；smoke LC25_f1_s7 + LC100_f2_s2025 两轮训练+全壁面评估通过）；**队列 15302 于 18:42 启动**（4 × 4090 × 2 槽），**2026-09-21 03:52 完成**（9 h 10 min；33 训练 + 66 评估退出码全 0，`queue_status=complete`；全表 `experiments/wss_learning_curve_20260920/results.md`，读数脚本与 JSON `offline/analyze_learning_curve.{py,txt,json}`）。

**33.2 结果（2026-09-21；留出折 Pa R²_cb 按三折合并为 136 例，每个分数 3 seed 均值 ± sd；射流 = 真值 p99 > 40 Pa 的 25 例；LC100 seed 1234 = 波 2a 折模型）**

| 训练例（折均） | 22.7（25 %） | 44.7（50 %） | 68.0（75 %） | 90.7（100 %） |
|---|---:|---:|---:|---:|
| 136 例折外 Pa R²_cb | 0.6468 ± 0.0098 | 0.6814 ± 0.0106 | 0.6983 ± 0.0060 | **0.7129 ± 0.0058** |
| 射流例 / 非射流例 | 0.584 / 0.696 | 0.617 / 0.736 | 0.640 / 0.747 | 0.655 / 0.762 |
| AG / AAA / ILO（seed 均值） | 0.685 / 0.685 / 0.562 | 0.724 / 0.723 / 0.589 | 0.742 / 0.728 / 0.617 | 0.756 / 0.749 / 0.626 |
| 正式 high-WSS R² | 0.292 | 0.364 | 0.396 | 0.416 |
| 逐例 R² 均值 / P10 | 0.652 / 0.524 | 0.692 / 0.575 | 0.704 / 0.590 | 0.723 / 0.609 |

同 seed 配对的**每翻倍训练例增量**（ΔR²_cb / log₂(n_hi/n_lo)，三 seed 逐个）：

| 段 | 整体 | 射流例 | 非射流例 |
|---|---|---:|---:|
| 25 → 50 % | **+0.035**（+0.050 / +0.039 / +0.017） | +0.034 | +0.040 |
| 50 → 75 % | **+0.028**（+0.011 / +0.043 / +0.029） | +0.037 | +0.019 |
| 75 → 100 % | **+0.035**（+0.023 / +0.015 / +0.067） | +0.038 | +0.035 |

**判读（按启动前冻结的判据）**

1. **数据量仍是瓶颈。** 75→100 % 段每翻倍 +0.035，三 seed 全正（最小 +0.015），高于预注册门槛 +0.02；三段斜率 +0.035 / +0.028 / +0.035 没有收窄迹象，曲线在 90 例处还在按约 +0.03/翻倍上升。按此外推，170 例扩到约 250 例（0.88 个翻倍）折外约 **+0.026**，到 0.74 上下；扩到 340 例约 +0.04。这是"同分布再加病例"的预期，回收病例若把 ILO/射流覆盖补厚，可能略高。
2. **射流例没有比整体更陡**（+0.034 / +0.037 / +0.038 对整体 +0.035 / +0.028 / +0.035），所以"优先补射流例"没有从曲线上得到额外支持；但射流例绝对水平最低（0.655 对非射流 0.762），且 ILO 最低（0.626），补数据时按队列/射流分层保证覆盖即可，不必专门加权。
3. **B 层 CFD 重跑值得**（已执行，见 §33.1）；A 层 89 单元入库值得。模型/目标侧（病例级幅值乘子、集成）与数据扩容并行，两者不互斥。
4. 噪声边界：单折单 seed 的 75→100 差值在 −0.008 ～ +0.05 之间跳（fold0 s1234 75 % 0.7389 > 100 % 0.7366），只能读三折合并 + 三 seed 均值；本节所有结论都基于合并口径。100 % 档三 seed 合并 0.7129，与 §28.6 的单 seed 0.7100 一致。

产物：`training_wss_min/runs/wss_learning_curve_20260920/<臂>/eval/ckpt_{best,last}/`（30 个新训练臂）+ 波 2a 三折。

**33.3 工作簿回填（2026-09-21）**：`update_wss_local_wave1_xlsx --name wss_learning_curve_20260920 --group "WSS learning curve（训练例数）｜2026-09-20（25/50/75/100%×cv3×3 seed，30 臂）"` → WSS 表 **477–509 行**（33 条：30 个新臂 + 波 2a 三折作 100 % seed 1234 参照），103033 个历史单元格核验不变，备份 `..._backup_before_wave1_20260921_104859.xlsx`；`annotate_workbook_methods.py` 已补 LC 代号的 GLOSS 正则（方法说明对照页 + 列 B 说明）。结果出来后回填本节 + 工作簿。


## 35. v5.2 基线重训：IND + CV5 双协议、X5D / X5Dcap 双配方、40 臂多 GPU 队列（2026-09-23 提交；作业 15616 → 15617 → 15618；首提 15613 因冻结副本缺新脚本失败）

**数据**：v5.2 = 170 在库 + 91 回收（4 例剔除、1 例豁免；11 例在库病例按端点半径修正重建），训练视图根 `data_wss_v5/views_v5_2_full_20260923`（四个视图各 261 例，符号链接拼装）。**协议**（用户拍板）：IND = 170 训 / 91 独立测（5 seed）；CV5 = 261 例患者分组分层 5 折（3 seed）。**配方**：X5D_v51 逐字复制 + X5Dcap（开口半径版 Murray 先验，对术后分流先验局限的对策）。**执行**：`prepare_wss_v52.py` 生成 split/统计/配置/矩阵 → `preflight_wss_v52.slurm`（锚点 X5D_v51_s1234 复评 + 冒烟）→ `run_wss_v52_queue.slurm`（冻结副本 GNN_v52_frozen_20260923，gres 4090×3、每卡 2 槽；单臂实测 6 h）。09-23 13:04 IND 十臂完成后按用户要求切到 4 卡：CV5 30 臂独立队列 `wss_v52_20260923_cv5`（预检 15629 → 队列 **15636**，gres 4090×4、8 臂并行），15618 取消。**读数**：IND 五 seed 集成 R²_cb 对照冻结 X5D_v51 的 0.749；CV5 折外汇总三 seed 均值±sd；X5Dcap 对 X5D 配对差；预注册在 `configs/wss_v52_20260923/matrix.json`。详见 [数据回收_2026-09-20/README.md §4.9](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据回收记录_2026-09-20.md)。

**35.1 终读与裁定（2026-09-24）**：40/40 臂完成（15618 IND 十臂 + 15636 CV5 三十臂，4 卡 8 槽，共约 40 h GPU）。IND：X5D 集成 0.751（冻结 0.749 持平）、**X5Dcap 集成 0.764**。CV5：X5D 三 seed 折外 0.754 ± 0.002、**X5Dcap 0.761 ± 0.002**，三 seed 折外集成 0.771 / 0.777，配对差 +0.0075（12/15 正）；对 v5.1 cv3 折外 0.710 升 +0.044（数据 136 → 261，与 §33 learning curve 一致）。**v5.2 基线配方 = X5Dcap**；用户裁定暂不训 261 例全量发布权重（后续还要优化，全量模型准确度由 CV5 折外 + IND 代言）。最差族仍是低 WSS 瘤腔（MENG_GUANG_QIN 0.42）与射流低估。读数 `experiments/wss_v52_20260923/readout_ckpt_best.md`、`cv5_oof_ensemble_readout.json`；详见 [数据回收 README §4.10](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据回收记录_2026-09-20.md)。

## 36. 一维物理先验 / 尾部目标 / 合成数据（2026-09-26 冻结；矩阵文档 [WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md](./WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md)）

**起点分析（X5Dcap CV5 折外集成 0.777）**：误差 64% 在每例最高 10% WSS 的点、72% 低估；78% SSE 在髂支叶段；oracle 幅值只能 +0.03 → 结构性的髂支高剪切低估。**方向一**：`wss_min_phys1d_v1` 一维 Womersley + Carreau 壁面剪切先验（零参数、部署可得）在 261 例上 ln 空间 R² 中位 **0.48**、幅值比中位 0.97（泊肃叶只有 −0.05 / 2.6×：峰值步是脉动效应）；臂 = 加特征 / 残差目标 / 泊肃叶消融。**方向二**：目标幅值加权、非对称低估权重（新字段 `loss_asym_under_weight`）、pinball 0.95。**方向三**：`tools/synth_morph_case.py` 网格径向形变合成 CFD 病例（协议 BC 不变），60 例，只进训练侧。筛选 IND 三 seed 配对，确认 CV5。


### 36.1 方向一读数（2026-09-26 22:20）：一维 Womersley/泊肃叶先验三臂 IND 配对全部不过 G1.1

- P1 特征臂 ΔR²_cb −0.001（三 seed 集成持平 0.7580）；P2 残差臂 +0.002（集成 +0.003、59/91 例改善）；P3 泊肃叶消融 ≈ P2。三臂三 seed 一致只在尾部：最高 10% 点低估比例 0.73 → 0.70–0.72（G1.3 过），射流子集 R² 反而 −0.01。
- 结论：先验信息已被现有输入覆盖，不进 CV5、不合并。细表 `training_wss_min/experiments/wss_v52_phys_20260926/readout_phys_ckpt_best.md`。方向二三臂等队列完成后同法读数。

### 36.2 方向二读数（2026-09-27 10:20）：低估加权损失 T2 四门全过 → CV5 确认；方向三 60 例合成 CFD 全部验收通过、入库中

- T2 `loss_asym_under_weight = 2.0`：IND 三 seed 配对 +0.012（3/3 正），射流子集 +0.018，最高 10% 点低估比例 0.73 → 0.66，三 seed 集成 0.771 对基线 0.758。T1 目标加权 +0.005、T3 pinball +0.001 均不过；P3 泊肃叶补齐后 +0.004 不过。
- 下一步：T2 CV5 三 seed 15 折确认（门：15 对配对 > +0.005、≥ 11/15 正）+ 剂量对照 ×1.5/×3.0 IND；合成子例 60/60 通过 RCR/壁面/帧数验收，入库链已提交，建视图后跑 S1 18 臂。

### 36.3 T2 低估加权损失 CV5 确认过门（2026-09-28 01:05）

- 15 对配对折差均值 +0.0093、13/15 正、射流 +0.016；三 seed pooled OOF 0.7714 对 X5Dcap 0.7608（+0.011）；尾部低估比例 0.71 → 0.64。与 IND 筛选一致，两级门全过。
- 待裁定：并入基线配方 / 部署候选；与合成数据（跑完后）是否做合并臂。剂量对照 ×1.5/×3.0 在 node04 排队。

### 36.4 方向三（形变合成 CFD 数据）终判：No-Go（2026-09-28 18:10）

- 59 例子例入库后 S1 臂：CV5 15 对配对 +0.0011（8/15 正）、IND 三 seed +0.004（1/3 正），两门均不过；尾部与 hotspot 持平。合成例只改几何、BC 不变，对折外精度无可测增益。
- §36 收口：方向一（一维先验）不过、方向二 T2 过两级门（pooled 0.771 对 0.761）、方向三不过。待裁定：T2 并入基线/部署候选；子例 ascii_in（约 0.5 TB）删除。
- 36.3 补：T2 五 seed 集成 91 例 0.7758 对 X5Dcap 0.7644（+0.012，五 seed 全正）；ILO +0.014、射流 +0.028、尾部低估 0.73 → 0.66；但逐例中位 −0.001、改善 43/91，AAA −0.011（n=6）→ 增益在尾部与射流病例，瘤腔低方差病例略降；部署候选裁定时需知。

### 36.5 裁定（2026-09-29，用户）：底座配方改为 X5Dcap_asym2（T2）；形变合成数据搁置

- **训练底座 = v5.2 数据 + X5Dcap_asym2 配方**（`loss_asym_under_weight = 2.0`）：CV5 三 seed pooled 0.7714（X5Dcap 0.7608）、IND 五 seed 集成 91 例 0.7758（0.7644）。后续优化与开发都以此为底座、以同 seed / 同折的 X5Dcap_asym2 run 作配对参照；配置与 run 位置见 [矩阵文档 §6](./WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md)。
- 方向三形变合成 CFD 搁置：等 cfd_auto 的 STL → 网格 → 求解链走通后，用更多形变（含边界条件 / 拓扑变化）再做；59 例子例数据保留。
- 部署发布包不变（X5D_v51 五 seed）；261 例全量 X5Dcap_asym2 权重与部署切换另行裁定。消融待办表「当前状态」已同步。
- **后续状态（2026-10-02 补记）**：配方未变，底座数字 10-02 改用 v5.2d 读数（§41）；全量权重在 v5.2d 上训练后于 10-02 上线、X5D_v51 下线（§42）；cfd_auto 形变扩充第二轮 10-02 按用户裁定暂停（04 块分析报告）；剂量对照 ×1.5 三臂在 09-29 下午中断、未完成（矩阵文档 §5 末条），用户 10-02 决定暂不补跑。
- **36.6 工作簿回填（2026-09-29 00:50）**：`update_wss_local_wave1_xlsx` 依次写入 6 组共 93 行到 `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`「WSS实验矩阵」：v5.2 基线重训 40 臂（IND 10 + CV5 30；CV5 记录来自 `experiments/wss_v52_20260923_cv5`，工具新增 `--extra-experiment` 合并多队列记录）、一维先验与尾部损失筛选 18 臂、T2 CV5 确认 15 臂、T2 补 seed 2 臂、形变合成 S1 CV5 15 臂 + IND 3 臂；`annotate_workbook_methods` 补 14 条说明正则后重跑，93 行列 B 全部带「｜说明：」。工具修正：工作簿多了周期量与方法说明两页后，页序检查改为按页名存在性判断。剂量对照（AB-06）跑完后再补一组。

## 37. v5.2p4 全量 265 例训练：X5Dcap_asym2 三 seed，评估集 recover8（2026-09-30；预检 16191 → 队列 16192 → 读数 16193，已完成）

> **口径（09-30 补记）**：本节训练集含 9 个边界条件错误的单元（标签修正前的 v5.2p4），读数只作历史参照；修正后的数据版本与重训配置见 §38。recover8 本身的标签不受影响。

**用户裁定（09-30）**：评估集由 recover11 改为 recover8（3 例同病人另一期已在 v5.2 → 放回训练）；YANG_BAO_KUI（cfd_auto 重建）入训练；训练集 = v5.2 的 261 + YANG + 3 = 265，val 空（选模按训练损失，同 IND），test = recover8；配方 X5Dcap_asym2，seed 1234 / 7 / 2025。数据版本细节见 [数据回收 README §4.11](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据回收记录_2026-09-20.md)，recover8 与 YANG 视图见 [cfd_auto 文档 §10.8](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/STL全自动CFD工程cfd_auto_试算_2026-09-27.md)。

**配置**：`training_wss_min/configs/wss_v52p4_full265_20260930/X5Dcap_asym2_full265_s{1234,7,2025}.json` 由 `wss_v52_phys_20260926/X5Dcap_asym2_s{1234,7,2025}.json` 复制，只改 `data_root / split_path / wss_stats_path / feature_stats_path / point_features_root / density_aug_root` 与 `name / notes`；model / train / eval 段（含 seed、`init_reference_config`、`loss_asym_under_weight`）逐位不变，证据 `experiments/wss_v52p4_full265_20260930/config_diff_vs_source.{txt,json}`（排序键 JSON diff）。生成与检查脚本 `experiments/wss_v52p4_full265_20260930/prepare_full265.py`（CPU 作业 16188）。

**执行**：冻结副本 `GNN_v52p_frozen_20260926`（训练 X5Dcap_asym2 的同一份代码；主树 `training_wss_min/*.py` 未改）。预检 `cluster/preflight_wss_v52p4_full265.slurm`（16191，挂 `afterany:16165:16166`；joint52r2 已于 10:56 PDT 全部 COMPLETED）：锚点 X5D_v51_s1234 复评最大差 4.1e-5（容差 5e-4）、三臂 AMP 步有限（峰值显存约 2.16 GB）、s7 冒烟 2 轮 + best/last 评估 → PREFLIGHT PASSED。队列 `cluster/run_wss_v52p4_full265_queue.slurm`（16192，`afterok:16191,afterany:16165:16166`，gres 4090×3、每卡一臂、200 GB）11:02 PDT（北京时间 09-30 02:02）三臂同时开训，每轮约 37 s（170 例时约 30 s），预计训练约 15:10 PDT（北京 06:10）结束；各臂随后在 recover8 上评 best / last。读数作业 `_recover/eval/eval_recover8_full265.slurm`（16193，`afterany:16192`）：`evaluate_recover8.sh full265_asym2 <三个 run>` + `compare_recover8.py`（与 IND 同口径配对）。作业日志打印配置与代码 sha1。

**读数规则（预写）**：recover8 物理 Pa R²_cb，三 seed Pa 均值集成；对照 X5Dcap_asym2 IND 在同 8 例上的五 seed 0.8393、seed 1234/7/2025 集成 0.8352（单 seed 0.8194 / 0.8239 / 0.8188）、逐例中位 0.830 / 0.827。n = 8，只作描述、不设门；test91 已在全量训练集内，不能再用于评估。

### 37.1 结果（2026-09-30 06:11；队列 16192 COMPLETED 4 h 09 min，读数 16193 COMPLETED）

三臂各 400 轮、约 246 min（每轮 37 s）；`nonfinite_events.jsonl` 只有 AMP 梯度溢出跳步 7 / 8 / 8 次（约 1.36 万步中，多在第 0 轮；原 IND asym2 三臂各 5 次，同类），无 NaN 损失；best（训练损失最小）0.1198 / 0.1198 / 0.1219（s1234 / s7 / s2025）。recover8 读数（`evaluate_recover8.sh`，CPU，与 IND 基线同一脚本同一口径）：

| 单元 | IND 五 seed | IND 三 seed | **full265 三 seed** | Δ full − IND3 | p99 真值 / IND3 / full3（Pa） |
|---|---:|---:|---:|---:|---:|
| AAA/ruputer/FU_GUO_JUN | 0.874 | 0.865 | 0.879 | +0.014 | 38.1 / 38.5 / 35.1 |
| AAA/ruputer/LIU_YU_MING | 0.680 | 0.668 | 0.670 | +0.002 | 19.8 / 14.1 / 14.4 |
| AAA/ruputer/SHI_YUN_XI | 0.800 | 0.791 | 0.814 | +0.023 | 21.1 / 20.5 / 19.6 |
| AAA/ruputer/WANG_SHUN_WEN | 0.878 | 0.866 | 0.889 | +0.023 | 23.8 / 26.0 / 26.2 |
| AAA/ruputer/ZHANG_ZAO_SHUAN | 0.757 | 0.761 | 0.745 | −0.017 | 36.2 / 27.0 / 24.9 |
| AAA/unruputer/LIU_WEN_QI | 0.860 | 0.855 | 0.846 | −0.009 | 19.9 / 19.6 / 18.4 |
| ILO/LI_JIE-1/after | 0.813 | 0.809 | 0.840 | +0.031 | 12.7 / 12.6 / 13.2 |
| ILO/LI_JIE-1/before | 0.847 | 0.845 | 0.855 | +0.011 | 31.7 / 28.7 / 29.5 |

- **集成 R²_cb（n = 8）：full265 三 seed 0.8393**，对同 seed IND 三 seed 0.8352（+0.004），与 IND 五 seed 0.8393 持平；逐例中位 **0.843**（IND 0.827 / 0.830）；逐例配对 6/8 改善、均值 +0.010。
- 单 seed R²_cb：full265 0.8285 / 0.8158 / 0.8307（均值 0.825），IND 0.8194 / 0.8239 / 0.8188（均值 0.821）。
- 变差的两例：ZHANG_ZAO_SHUAN（库内原 CFD，射流高值低估加重：p99 36.2 → 预测 24.9 Pa）、LIU_WEN_QI −0.009；LIU_YU_MING 仍是最差例（0.670，p99 19.8 对 14.4）。
- 队列自评（GPU、同 split）best 0.8295 / 0.8171 / 0.8311、last 0.8318 / 0.8182 / 0.8291，与 CPU 读数差 ≤ 0.0013（设备数值差）；分域 AAA 0.810–0.828、ILO 0.839–0.848（`experiments/wss_v52p4_full265_20260930/results.md`）。
- **解读**：n = 8，只作描述。全量模型比同 seed IND 模型略好（集成 +0.004、逐例中位 +0.016、6/8），幅度在噪声内（单 seed 带 ±0.045），与 170 → 265 例（约 0.64 次翻倍，learning curve 预期约 +0.02）方向一致但本集太小不能定量确认；射流高值低估这一已知失效族没有因数据增加而改善。test91 已进训练集，全量模型没有别的留出集。
- 读数文件：`outputs/cfd_auto_trial_20260927/_recover/eval/readout_full265_asym2.md`、`readout_recover8_full265_vs_ind.{md,json}`，预测 `pred_full265_asym2/`；权重 `training_wss_min/runs/wss_v52p4_full265_20260930/X5Dcap_asym2_full265_s{1234,7,2025}/`（未打包、部署未切换）。

### 37.2 工作簿回填（2026-09-30 11:32）

- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`「WSS实验矩阵」第 603–605 行，组名「WSS全量265例训练｜2026-09-30（3臂：X5Dcap_asym2 三seed，test=recover8）」（`update_wss_local_wave1_xlsx --name wss_v52p4_full265_20260930 --group …`；历史单元格 140 775 个逐一核验不变，表格范围 A5:FL605，证据 `experiments/wss_v52p4_full265_20260930/xlsx_acceptance.json`，备份 `…_backup_before_wave1_20260930_113103.xlsx`）。表内物理 R²_cb 为队列自评（GPU、best）：0.8295 / 0.8171 / 0.8311。
- **配对参照改为同一 recover8 上的 IND 同 seed**：此前 v5.2 各组的通用写法（协议「V5 train138/test34」、ΔR² 只对 C1 历史 test34）对本组不成立。新增 GPU 作业 16300（`experiments/wss_v52p4_full265_20260930/ref_ind_recover8.slurm`：冻结副本、best、full265 视图根的 test 分区）把 X5Dcap_asym2 IND s1234 / s7 / s2025 在 recover8 上重评，得 0.8222 / 0.8245 / 0.8163（CPU 读数 0.8194 / 0.8239 / 0.8188，设备差 ≤ 0.003）→ ΔR² 列 **+0.0072 / −0.0074 / +0.0148**（均值 +0.005，与 CPU 集成口径 +0.004 一致）；C1 历史保留为第二参照。
- 工具改动（只对 `NAME` 以 `wss_v52p4_full265` 开头的组生效，其他组输出不变）：`update_wss_local_wave1_xlsx.py` 加本组的中文描述、协议文字（train 265 / val 0 → test recover8）、备注（数据组成、选模、recover8 是唯一留出集、集成读数出处）与 `ref_ind_recover8/` 配对参照；`annotate_workbook_methods.py` 的 GLOSS 加一条（「方法说明对照」页 + 列 B「｜说明：」3 行，备份在 `experiments/workbook_method_annotations/`）。
- 既有缺口（与本次无关、未处理）：「速度与压力实验矩阵」页「体场全周期时间阶段2｜2026-09-19」组 42 行在 09-29 之前就没有 GLOSS 匹配，仍无「｜说明：」后缀。

## 38. 标签修正版数据 v5.2c：此后训练和测试的唯一数据（2026-09-30，用户要求）

> **09-30 晚更新**：用户要求所有实验的数据基础一致、旧口径数据删掉 → v5.2p5 并入 v5.2c，成为唯一的统一根（332 单元，快照 `anatomy_pointcloud_v5_2c_20260930`）；recover 10 单元换入库；旧根和旧缓存已删除；23 个配置重新生成，全部指向 v5.2c，特征 z-score 文件也放进了 `views_v5_2c_20260930/feature_stats/`。开口半径分流规则按修正数据重拟合（a 1.150 → 1.265），它只进 `*_capfit` 字段，主线用的 `*_murray_cap` 不受影响：IND、CV5 五折、full265 的特征 z-score 在统一根上重算，与原文件 0 差。详见 [数据统一与旧版本清理](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据统一与旧版本清理_2026-09-30.md)。下文 v5.2p5 的写法已过时。

**起因**：cfd_auto 查出库内 9 个单元的边界条件不合协议（RCR 挂到相邻出口、RCR 复制、入口除数与网格入口面积不符）。同网格、只改边界条件重算后换入 data_new，旧文件归档后删除。偏差大小和执行细节见 [标签核查](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/库内标签问题核查_RCR挂错与入口除数_2026-09-30.md) §4、§6，数据版本见 [数据回收 README §4.12](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据回收记录_2026-09-20.md)。

**数据版本**：
- **v5.2c** `data_wss_v5/views_v5_2c_20260930` 取代 v5.2 `views_v5_2_full_20260923`：IND、CV5、syn 分区。
- **v5.2p5** `data_wss_v5/views_v5_2p5_full265_20260930` 取代 v5.2p4：full265 + recover8。
- 文件名与布局和前身相同，几何与输入特征逐位相同，只有 9 例的标签不同。WSS 统计 log 均值高 0.001–0.006；特征 z-score 不变。
- 这 9 例的分布：
  - IND：train170 里 6 例，test91 里 3 例（ZHANG_WAN_ZENG-1/after、LI_YOU_ZHI-0/after、WANG_LI_MIN-0/before）；
  - CV5 测试折：fold1 1 例、fold2 1 例、fold3 3 例、fold4 4 例，fold0 没有；
  - full265：全部在训练集，recover8 里没有。

**以前数字的口径**：§35 v5.2 基线、§36（T2 / 合成）、§37 full265 都是在含 9 个错标签单元的数据上训练的，IND test91 和 CV5 fold1/3/4 的评估标签也有变化。它们作为历史参照保留；新的基线数字以 v5.2c / v5.2p5 重训为准。

**重训配置（已生成，未训练）**：`training_wss_min/configs/wss_v52c_labelfix_20260930/` 共 23 个 X5Dcap_asym2 配置，都来自现役配置、配方逐位不变，只换数据根：
- IND 5 seed，来自 `wss_v52_phys_20260926` 的 s1234/7/2025 和 `wss_v52_phys2e_20260928` 的 s11/2026；
- CV5 15 臂，来自 `wss_v52_phys2_20260927`；
- full265 3 seed，来自 `wss_v52p4_full265_20260930`。

`matrix.json` 可直接交给 `training_wss_min.tools.run_local_train_queue`。生成脚本和核验文件在 `training_wss_min/experiments/wss_v52c_labelfix_20260930/`：`make_configs.py`、`config_diff_vs_source.json`、`load_check_16372.out`（严格加载通过）。其他实验迁移到新数据用 `python -m training_wss_min.tools.repoint_data_root`。

**未完成**：3 个合成单元还在重算，完成前 v5.2c 的 syn 分区不可用（主线不用合成数据）。

## 39. v5.2c 重训：X5Dcap_asym2 CV5 × 3 seed、full265 × 3 seed，三头 M1cap full265 × 3 seed，recover8 评估（2026-09-30 20:18 提交；预检 16389 → 队列 16390 → 附加评估与读数 16391）

**用户裁定（09-30）**：
- 训练 CV5 15 臂和 full265 3 臂，用 §38 的现成配置，一个字不改；IND 5 臂这次不跑。
- 三头只做全量训练（full265 → recover8），配方照原样用最新的 M1cap_v52cv（09-25）：三通道（峰值 WSS / TAWSS / OSI）等权 MSE，q90 pinball 关，没有低估加罚。代码里低估加罚只写在单通道损失分支，三通道分支不生效，所以三头的峰值通道没有 T2 的两项峰值优化，部署时峰值仍取 B 组。
- 附加评估 E3（15 个 CV5 折模型评 recover8）和 E5（已部署三头 v5.1 评 recover8）都做。
- 全部放 master 4 × 4090，每卡一个；node04 留给后续整周期时间实验。每卡两个不划算：历史 CV5 两槽单臂中位 6.88 h，单槽折算约 3.2 h，单臂慢 2.1 倍，总吞吐反而下降。

**矩阵**（21 个训练、42 次队列内评估 + E3 15 次 + E5 3 次）：

| 组 | 臂 | 训练 → 主评估 | seed | 数量 | 参照（标签修正前） |
|---|---|---|---|---:|---|
| A | X5Dcap_asym2 CV5 | 每折约 209 例 → 折外，合并 261 例 | 1234 / 7 / 2025 | 15 | 各 seed pooled 0.7750 / 0.7707 / 0.7684（均值 0.7714） |
| B | X5Dcap_asym2 full265 | 265 → recover8 | 同上 | 3 | 单 seed 0.8295 / 0.8171 / 0.8311，集成 0.8393 |
| C | M1cap 三头 full265 | 265 → recover8，三个通道 | 同上 | 3 | 护栏：峰值通道归一化 R²_cb 对同 seed B 掉 ≤ 0.02 |
| E3 | A 组 15 个折模型 | → recover8（best） | — | 推理 | 对 B 的三 seed 集成 |
| E5 | 已部署三头 M1_3head_3seed_20260922 | → recover8（best） | — | 推理 | C 组的 TAWSS / OSI 参照 |

**执行**：
- 执行实验 `wss_v52c_retrain_20260930`。配置目录里 18 个 X5Dcap_asym2 配置是 `wss_v52c_labelfix_20260930` 的逐字副本，run 名不变，产物在 `runs/wss_v52c_labelfix_20260930/`。3 个三头配置由 `tools/prepare_wss_v52c_retrain.py` 生成，产物在 `runs/wss_v52c_retrain_20260930/`；它们和同 seed 的 B 只差 9 个声明字段（目标、周期视图、统计、out_dim、pinball、低估加罚、配对初始化参照、阈值掩膜），和 M1cap_v52cv_f0 只差数据路径。队列按 B、C、A 的顺序上卡，长臂先跑。
- 三头统计：v5.2c `wss_min_cycle_v1/stats/cycle_stats_{tawss,osi_logit,osi_linear}_full265_train265.json`（265 例、1416 万个壁面点，只新增文件），三通道合并文件 `experiments/wss_v52c_retrain_20260930/stats/multi_stats_full265_train265.json`。
- 冻结代码副本 `GNN_v52c_frozen_20260930`，和 HEAD 71422d2 逐字一致；主仓可以继续改代码，不影响队列。
- 预检锚点：旧锚点的数据根已删，改用修正前 full265 s1234（作业 16192）的副本，配置重指到 v5.2c。recover8 的标签和这个模型的输入在 v5.2c 上都没变，预检重评 1494 个字段最大差 2.1e-5，Pa R²_cb 0.82947 与存档值相同，说明统一根上的 recover8 和旧根逐位一致。
- 预检 16389 用时 3 分钟通过（21 臂 AMP、X5Dcap full265 s7 与三头 s7 两个冒烟）。队列 16390 于 20:18 开始。单臂估计 CV5 约 3.3 h、full265 约 4.2 h，全部约 21 h，预计 10-01 17:00 前后训完。作业 16391（afterany）接着跑 E3 / E5 和读数。
- E5 的评估路径已在 CPU 上试跑（s1234 峰值通道归一化 R²_cb 0.872），只用来验证能跑通，正式数字以 16391 为准。

**读数规则**：recover8 只有 8 例，只作描述、不设门。B 和修正前同 seed 比：recover8 标签没变，差值 = 9 个单元的训练标签 + 训练随机性；读数脚本会校验两边 recover8 真值逐位相同。A 的 9 个修正单元既在训练里、也在折外评估里，所以另报「修正前模型 · 新标签」：只换评估标签、不换模型，用来拆开两部分影响。C 组逐通道报：峰值、TAWSS（R²、CCC、<0.4 Pa 重合度）、OSI（R²、>0.1 / >0.3 重合度）、滞留区。

**证据**：配置与矩阵 `training_wss_min/configs/wss_v52c_retrain_20260930/`；执行、锚点副本、E5 run 目录、统计与读数 `training_wss_min/experiments/wss_v52c_retrain_20260930/`（读数 `readout_ckpt_{best,last}.{md,json}`）；工具 `tools/prepare_wss_v52c_retrain.py`、`tools/report_wss_v52c_retrain.py`；Slurm `cluster/{preflight_wss_v52c_retrain,run_wss_v52c_retrain_queue,post_wss_v52c_retrain}.slurm`。

**已取消（2026-09-30 22:55，用户要求）**：有数据需要重新处理，先停掉全部作业。队列 16390 运行 2 h 37 min 后取消，收尾作业 16391 在启动前取消；当时在跑的 B 三臂和三头 s1234 训到约 250 / 400 epoch，都没有评估，其余 17 臂未启动。`runs/wss_v52c_labelfix_20260930/X5Dcap_asym2_full265_s*`、`runs/wss_v52c_retrain_20260930/M1cap_full265_s1234` 是半截产物；实验目录留有 `queue_status.json`、`.queue.lock`、`runtime_preflight.json`、`logs/`。09-30 用户要求后已删除这些半截 run 与队列文件（两个空的 run 父目录一并删掉）；保留准备好的输入（`anchor/`、`e5_deployed_m1/`、`stats/`）和预检冒烟产物（`smoke/`、`runs/wss_v52c_retrain_20260930_smoke/`）。数据处理完后要按新数据重新生成统计和配置，并重新预检。

## 40. 数据更新为 v5.2d，重训配置换根（2026-10-01；未训练）

v5.2c 重训取消（§39）后，对母库 332 个单元做了全量审计，按结论一次合并出 v5.2d。过程、逐项验收和没做的事都在[母库全量审计 §11](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/母库全量审计_2026-10-01.md)，这里只记对主线有影响的部分。

- **数据根**：视图 `data_wss_v5/views_v5_2d_20261001`，快照 `data_wss_v5/anatomy_pointcloud_v5_2d_20261001`。v5.2c 的两个根已改名，§38、§39 里的 v5.2c 路径不再存在。分区文件名和单元清单不变（IND 170 / 91、CV5 261、full265 + recover8、syn）。
- **相对 v5.2c 变了 18 个单元**：HAN_JIAN_FU 的壁面标签（IND 训练、CV5 fold2 留出）；WANG_TIAN_QING-1/after 整例换成从本例 STL 重建的结果（IND 测试、CV5 fold0 留出，原来用的是 DONG_JIA_JU-1/after 的网格）；16 个单元的出口语义名修正（标签不变，壁面语义分段这个输入变了）。recover8 的 8 例都不在这 18 个里。
- **主线输入在未改动的 314 个单元上逐位不变**：分流规则在修正后的数据上重拟合（a 1.2645 → 1.2624，b 0.0961 → 0.1002），只改了 `*_capfit` 两个字段；X5Dcap_asym2 和三头用的是纯几何的 `log_q_branch_murray_cap`、`log_tau0_murray_cap`，不受影响。分流特征包逐数组比对：314 个单元只有 capfit 两个数组不同。
- **统计**：WSS 对数均值各分区变化 ≤ 0.0004，输入特征归一化均值移动 ≤ 0.005 个标准差。
- **配置与提交**：见 §41。
- **历史读数的口径**：v5.2、v5.2p4、v5.2c 上的数字都是这次修正之前的。影响评估集的只有两处：IND test91 和 CV5 fold0 留出里的 WANG_TIAN_QING-1/after 换了几何和标签；CV5 fold2 留出里的 HAN_JIAN_FU 换了壁面标签。recover8 的标签没变。
- **未处理的已知项**：150 个每步迭代提前退出的单元（AG 全部 105、AAA 25、ILO 20）是否重算待用户决定，一例实测标签相对迭代收敛解峰值 WSS 相对 L2 约 4 %。

用户 10-01 傍晚确认：150 个提前退出的单元先不动，按 v5.2d 重训（§41）。

## 41. v5.2d 重训：X5Dcap_asym2 CV5 × 3 seed、full265 × 3 seed，三头 M1cap full265 × 3 seed，recover8 评估（2026-10-01 16:33 开始；预检 16459 → 队列 16460 → 收尾 16461；**已完成**）

用户 10-01 要求在 v5.2d 上把 09-30 取消的重训补上。矩阵和 §39 相同，只换数据版本。

**矩阵（21 臂）**

| 组 | 臂 | 训练 / 评估 | 参照 |
| --- | --- | --- | --- |
| B | X5Dcap_asym2 full265 × seed 1234 / 7 / 2025 | 训练 265 例，评估 recover8 | v5.2p4 同 seed（作业 16192：0.8295 / 0.8171 / 0.8311，集成 0.8393） |
| C | 三头 M1cap full265 × 3 seed | 同上，三通道（峰值 WSS、TAWSS、OSI） | 同 seed 的 B（峰值通道护栏）；已部署三头（E5） |
| A | X5Dcap_asym2 CV5 5 折 × 3 seed | 261 例患者分组五折，折外评估 | v5.2 同配方三 seed 折外 pooled 0.7714 |

收尾作业另做 E3（15 个 CV5 折模型评 recover8）和 E5（已部署三头评 recover8），然后出读数。

**准备与预检**

- 配置由 `tools/prepare_wss_v52d_retrain.py` 生成：A、B 是 §39 配置换根，工具逐字段校验只有数据路径、name、notes 变了；C 用 §39 的同一个构造函数在 v5.2d 路径上重建，声明差异检查不变。全部 run 写到 `runs/wss_v52d_retrain_20261001/`，不碰 v5.2c 的目录。
- 三头合并统计按 v5.2d 的统计文件重建（265 例训练分区）。
- 冻结代码副本 `GNN_v52d_frozen_20261001`，`training_wss_min/*.py` 与 §39 的冻结副本逐字一致。
- 预检 16459 用时 3 分钟通过：21 臂配对初始化、AMP 三步、全壁面推理都正常（单臂显存峰值 ≤ 2.2 GB）；X5Dcap full265 s7 和三头 s7 两个冒烟（训练 2 轮 + 评估 best / last）通过。
- 预检锚点：v5.2p4 full265 s1234 模型（作业 16192）配置重指到 v5.2d 后重评 recover8，1494 个字段最大差 2.1e-5，Pa R²_cb 0.82947 与存档值相同。这说明 v5.2d 上 recover8 的标签和主线输入与旧根一致。

**读数规则**（预先写定，同 §39）

- recover8 只有 8 例，只作描述、不设门。
- B 对 v5.2p4 同 seed：recover8 标签没变，差值 = v5.2c 与 v5.2d 两次数据修正对训练集的影响 + 训练随机性。
- A：折外评估标签在 9 个 09-30 修正单元和 HAN_JIAN_FU 上变了；WANG_TIAN_QING-1/after 换了几何，v5.2 模型的预测无法在新标签上重评。所以「v5.2 模型 · 新标签」只在壁面点能对上的单元上算，新模型也在同一批单元上另算一个数用于配对，读数里列出被排除的单元。
- C：逐通道报峰值、TAWSS、OSI；护栏是峰值通道归一化 R²_cb 对同 seed B 掉幅 ≤ 0.02。
- 已知的数据限制：150 个每步迭代提前退出的单元标签保持原样（用户 10-01 决定先不动）。

**结果（2026-10-02；队列 16460 COMPLETED 20 h 39 min，21 臂全部完成并评估；收尾 16461 COMPLETED）**

一句话：在 v5.2d 上重训后的精度与 v5.2 持平。数据修正的作用体现在评估标签上（错标签单元的折外得分大幅回升），重训本身没有带来可分辨的增益。下面是最优检查点的数，末轮检查点结论相同（`readout_ckpt_last.md`）。

**A：CV5 折外（261 例）**

| seed | v5.2d 新模型 Pa R²_cb | v5.2 模型 · 旧标签 | v5.2 模型 · 新标签 | 新 − 旧模型新标签 |
| --- | ---: | ---: | ---: | ---: |
| 1234 | 0.7750 | 0.7749 | 0.7786 | −0.0036 |
| 7 | 0.7752 | 0.7706 | 0.7750 | +0.0001 |
| 2025 | 0.7780 | 0.7683 | 0.7719 | +0.0061 |
| 三 seed 均值 ± sd | **0.7761 ± 0.0017** | 0.7713 | 0.7752 | +0.0009 |
| 三 seed 折外集成 | **0.7926**（261 例）/ 0.7925（可配对 260 例） | 0.7892 | 0.7929 | −0.0004 |

- 后三列在可配对的 260 个单元上算。排除的 1 个是 WANG_TIAN_QING-1/after：它换了几何，v5.2 模型的预测对不上新壁面点；新模型在它上面的集成 R² 是 0.773。
- **标签修正让同一批 v5.2 模型的得分升了约 0.004**（单 seed 0.7713 → 0.7752，集成 0.7892 → 0.7929）。升幅来自 10 个标签变了的单元，这些单元在各自的留出折里，模型没见过它们的标签：

| 单元 | v5.2 模型 · 旧标签 | v5.2 模型 · 新标签 | v5.2d 模型 · 新标签 |
| --- | ---: | ---: | ---: |
| AAA/ruputer/MENG_GUANG_QIN | 0.339 | 0.804 | 0.792 |
| ILO/ZHANG_YONG_SHENG-0/before | 0.571 | 0.864 | 0.870 |
| AAA/unruputer/CHEN_SHU_LIN | 0.640 | 0.864 | 0.867 |
| ILO/LI_YOU_ZHI-0/after | 0.743 | 0.851 | 0.849 |
| ILO/WANG_LI_MIN-0/before | 0.694 | 0.728 | 0.732 |
| ILO/ZHANG_WAN_ZENG-1/after | 0.704 | 0.737 | 0.708 |
| ILO/ZHANG_YAN_SHAN-0/before | 0.833 | 0.857 | 0.852 |
| AG/slow/LIU_ZONG_YANG | 0.762 | 0.777 | 0.796 |
| AG/slow/SUN_ZONG_GE | 0.659 | 0.659 | 0.639 |
| AAA/unruputer/HAN_JIAN_FU | 0.834 | 0.820 | 0.816 |

  前三个单元是 09-30 查出的 RCR 挂错出口的病例：模型按几何预测的本来就是协议下的流场，旧标签错了才显得差。HAN_JIAN_FU 换成重算的壁面标签后得分略降 0.014。
- **重训没有带来可分辨的增益**：新模型对「v5.2 模型 · 新标签」单 seed 差 −0.004 / 0.000 / +0.006，集成差 −0.0004；逐例 131 / 260 改善，中位差 0.0001。标签没变的 250 个单元上两者集成是 0.7936 对 0.7935。这符合预期：261 个训练单元里只变了十几个。
- 分队列（集成，可配对单元；新模型 / v5.2 模型新标签 / v5.2 模型旧标签）：AG 85 例 0.823 / 0.821 / 0.821；AAA 60 例 0.819 / 0.817 / 0.804；ILO 115 例 0.756 / 0.759 / 0.757。AAA 的旧标签口径低 0.013，就是那三个错标签病例造成的。

**B：full265 → recover8（8 例，只作描述）**

| seed | v5.2d | v5.2p4 同 seed | 差 |
| --- | ---: | ---: | ---: |
| 1234 | 0.8164 | 0.8295 | −0.0131 |
| 7 | 0.8307 | 0.8171 | +0.0136 |
| 2025 | 0.8322 | 0.8311 | +0.0011 |
| 三 seed 集成 | **0.8393**（逐例中位 0.850） | 0.8401（0.842） | −0.0008 |

持平，逐例 3 / 8 改善。单 seed ±0.013 是训练随机性的量级。v5.2p4 的集成 §37.1 记的是 0.8393，本次读数工具对同一批预测重算得 0.8401，差 0.0008；不是 Pa 均值与对数均值的差别（对数均值得 0.8394），原因没有查清，两个数都留着。

**E3：15 个 CV5 折模型 → recover8**：15 模型集成 **0.8444**（逐例中位 0.848）；每个 seed 的五折集成 0.8389 / 0.8398 / 0.8408；单个折模型平均 0.8224。15 个各用约 80 % 数据训的模型集成，略高于 3 个全量模型的集成（0.8393）；n = 8，差 0.005 不作结论。

**C：三头 M1cap full265 → recover8**（E5 = 已部署三头，v5.1 train136）

| 模型 | 峰值 Pa R²_cb | TAWSS Pa R²_cb | TAWSS CCC 中位 | TAWSS<0.4 IoU | OSI R²_cb | OSI CCC 中位 | OSI>0.1 IoU | OSI>0.3 IoU | 滞留区 IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C s1234 | 0.819 | 0.841 | 0.905 | 0.680 | 0.502 | 0.673 | 0.667 | 0.233 | 0.557 |
| C s7 | 0.808 | 0.834 | 0.898 | 0.683 | 0.500 | 0.699 | 0.667 | 0.236 | 0.538 |
| C s2025 | 0.791 | 0.824 | 0.904 | 0.682 | 0.490 | 0.672 | 0.659 | 0.237 | 0.532 |
| **C 三 seed 集成** | 0.823 | 0.849 | 0.912 | 0.688 | 0.542 | 0.698 | 0.678 | 0.234 | 0.555 |
| E5 三 seed 集成 | 0.792 | 0.805 | 0.868 | 0.646 | 0.490 | 0.642 | 0.651 | 0.182 | 0.502 |

- 峰值通道护栏三个 seed 都过：归一化 R²_cb 对同 seed B 差 +0.0019 / −0.0056 / −0.0048（门限 −0.02）。
- 三头的峰值 Pa R²（集成 0.823）低于 B（0.839），是已知的：三通道损失里没有低估加罚。
- 相对已部署三头，三个通道都更高（峰值 +0.031、TAWSS +0.044、OSI +0.052、滞留区 +0.053）。两者差的是训练数据量（265 对 136 例）加数据修正，不能单独归给哪一项；n = 8。

**运行记录**：队列自带的收尾报告步骤报错退出（`report_failure.json`：`report_wss_local_wave1` 要求矩阵有 `anchor_run` 字段，本矩阵没有）。它在 21 臂训练和评估都完成之后才运行，不影响结果；正式读数由 16461 的 `report_wss_v52d_retrain` 生成。补充数字（配对集成、分队列、逐单元）由 `experiments/wss_v52d_retrain_20261001/extra_readout.py` 生成。

**裁定（2026-10-02，用户）**：底座数字改用 v5.2d 读数。训练底座 = v5.2d 数据 + X5Dcap_asym2 配方（配方不变）：CV5 折外（261 例）三 seed 各自 pooled 0.7750 / 0.7752 / 0.7780，均值 **0.7761 ± 0.0017**，三 seed 折外集成 **0.7926**；全量 265 例 → recover8 三 seed 集成 0.8393。与 v5.2 上的 0.7714 相比，约 +0.004 来自评估标签修正，其余在 seed 波动内，不是配方或重训带来的提升。IND 170 / 91 协议没有在 v5.2d 上重训，旧的 IND 五 seed 集成 0.7758 和 X5Dcap 对照（CV5 0.7608、IND 0.7644）是 v5.2 口径，只作历史参照。此后新实验从 `configs/wss_v52d_retrain_20261001/` 的配置派生，配对参照是 `runs/wss_v52d_retrain_20261001/` 里同 (fold, seed) 的 run（[§36 矩阵 §6 底座指针](./WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md)已更新）。需要 IND 协议配对的实验要先在 v5.2d 上补 IND 参照臂。

**部署切换（10-02 已完成，见 §42）**：full265 的 X5Dcap_asym2 三 seed 和三头三 seed 已打包上线，X5D_v51 五 seed 与 M1_3head_3seed_20260922 下线。

**证据**：配置与矩阵 `training_wss_min/configs/wss_v52d_retrain_20261001/`；执行、锚点、E5 目录、统计、队列状态与读数 `training_wss_min/experiments/wss_v52d_retrain_20261001/`（`runtime_preflight.json`、`queue_status.json`、`readout_ckpt_{best,last}.{md,json}`、`extra_readout.{py,json}`）；工具 `tools/prepare_wss_v52d_retrain.py`、`tools/report_wss_v52d_retrain.py`；Slurm `cluster/{preflight_wss_v52d_retrain,run_wss_v52d_retrain_queue,post_wss_v52d_retrain}.slurm`。

## 42. v5.2d 全量模型上线，v5.1 部署模型下线（2026-10-02，wss_deploy v0.16.3）

用户 10-02 要求：新三头模型和峰值 WSS 模型训练完成，重新发布上线，旧模型下线。部署侧细节在 [05 块框架文档 §29.13](../05-部署工具/WSS_部署演示工具_从STL到峰值WSS_整体框架与计时_2026-09-17.md)。

- **新发布包**：`X5Dcap_asym2_v52d_3seed_20261002`（§41 的 B 组 full265 三 seed，峰值 WSS，部署默认）；`M1cap_v52d_3seed_20261002`（§41 的 C 组三头三 seed）。体场 PF6/VF6 未重训，保留。
- **下线**：`X5D_v51_5seed_20260916`、`M1_3head_3seed_20260922` 移到 `outputs/wss_deploy_release_retired/`，服务不再列出和加载；历史结果照常查看。
- **说明卡口径**：峰值包主数字 = 同配方 CV5 折外三 seed 集成 0.793（261 例），另列 recover8 0.839；三头包 = recover8（8 例）。峰值包的人群分位参照改为 CV5 折外 p99（261 例）。
- **部署链路验收（recover8 的 STL，自动命名，同链路新旧包对比，病例等权 Pa R²_cb）**：峰值 0.830 → 0.854（7/8 例改善）；三头峰值 0.808 → 0.839、TAWSS 0.809 → 0.850（8/8）、OSI 0.445 → 0.489（7/8）。训练侧 recover8 读数是峰值 0.839、三头 TAWSS 0.849 / OSI 0.542，部署链路与之同带。部署加载器与训练评估逐点预测最大相对差 ≤ 1.1e-6。记录 `training_wss_min/experiments/wss_deploy_timing_20260917/acceptance_recover8_v52d_20261002/`。
- **上线**：提交 01efd0f，`service upgrade` 后 PID 1739065（GPU 1）。
- **口径提醒**：recover8 只有 8 例，只作描述；新病例的预期精度按 CV5 折外 0.79 说。

## 43. 体场 PF6（压力）/ VF6（速度）在 v5.2d 上复验（2026-10-02 用户确认；预检 16866 → master 队列 16867 + node04 队列（Slurm 之外）→ 收尾 16868；**已完成**，读数见 §43.2，10-03 上线见 §44）

**现状**：PF6 / VF6 从没有在 v5.2 之后的数据上训练过。

| 数据 | 协议 | 已有结果 |
| --- | --- | --- |
| v5.0（172 例，RCR 修正前） | train138 → test34，seed 1234 / 7 / 2025 | PF6 压力 R²_cb 0.7894、VF6 速率 0.7964（三 seed 均值）。部署的体场发布包 `PF6_VF6_peak_3seed_20260920` 就是这批权重（§42 上线新 WSS 模型时体场包未换） |
| v5.1（170 例） | cv3 三折，只有 seed 1234 | 折外 PF6 0.793 / 0.648 / 0.791，VF6 0.766 / 0.786 / 0.784（当时只作时间臂的配对父臂） |
| v5.2 / v5.2c / v5.2d | — | 没有。全周期线（U0 / P1 / G11）在 v5.2 fold0 上训过速度和压力，但那是另一套模型和采样，不是 PF6 / VF6 |

v5.0 之后体场标签经历了：26 例 RCR 面积修正（v5.1）、91 个回收单元（v5.2）、9 个单元按协议重算（v5.2c）、WANG_TIAN_QING-1/after 重建（v5.2d）。

**矩阵（配方逐字不变，只换数据；与 §41 的 WSS 重训同一套协议）**

| 组 | 臂 | 训练 / 评估 | 臂数 |
| --- | --- | --- | ---: |
| A-P | PF6 CV5：5 折 × seed 1234 / 7 / 2025 | 261 例患者分组五折，折外评估 | 15 |
| A-V | VF6 CV5：5 折 × 3 seed | 同上 | 15 |
| B-P | PF6 full265 × 3 seed | 训练 265 例，评估 recover8 | 3 |
| B-V | VF6 full265 × 3 seed | 同上 | 3 |

合计 36 臂。执行：六张卡（master 4 × 4090 + node04 2 × A100）每卡一个臂，见 §43.1。

收尾（只读评估）：E3 = 30 个 CV5 折模型评 recover8（每 seed 五折集成、15 模型集成）；E5 = 已部署的 v5.0 PF6 / VF6 三 seed 评 recover8，作全量权重的同口径参照。

**数据准备（训练前，CPU 作业）**

- 体场视图：v5.2d 视图根里只有 159 个单元有 `volume.npz`，全周期缓存里另有 102 个，YANG_BAO_KUI、3 个搭档单元和 recover8 共 12 个从未建过。为了不动全周期缓存的来源签名，新建一个体场数据根 `views_v5_2d_20261001/wss_min_volview_v1/`：273 个真实单元各一份体场视图（已有的 261 个与重建结果一致才链接过去，缺的 12 个从 v5.2d 快照新建），壁面视图链接到主视图。
- 已有的 159 份体场视图全部从 v5.2d 快照重建一遍逐数组比对，不一致的换成新建的。
- 统计：CV5 五折和 full265 各自的训练分区体场统计（线性 z）与输入特征归一化，共 6 套（`write_volume_stats` 按训练例数命名，fold2 / fold3 都是 211，逐分区改名）。
- 预检：36 臂初始化、三步训练、全量推理，各一个压力臂和速度臂的短程训练加评估。

**读数规则（预先写定）**

- 这是在新数据上建立基线，不设过门判据。CV5 报每个 seed 的折外 pooled R²_cb、三 seed 均值 ± sd、三 seed 折外集成；压力另分壁面 / 内部，速度报速率 R²_cb 和向量误差；分队列。
- recover8 只有 8 例，只作描述。B 对 E5（v5.0 部署权重）的差值混着训练数据量（265 对 138 例）和数据修正，不单独归因。
- 与 v5.0 test34、v5.1 cv3 的旧数字不是同一评估集，只并列、不相减。
- 已知数据限制同 §41：150 个每步迭代提前退出的单元标签保持原样（一例实测出口压力偏 1.6–2.1 %）。

**没有放进本批的（可在本批之后追加，数据准备可复用）**

- 先验换成开口半径版（`*_murray_cap`，WSS 底座和全周期缓存用的就是它；PF6 / VF6 现在用的是 `*_murray`）：单一改动的配对臂，CV5 × 3 seed 每个目标 15 臂。
- IND 170 / 91 协议。

### 43.1 用户确认与执行记录（2026-10-02）

**用户确认（10-02）**：按上表 36 臂提交，配方逐字不变、只换数据；先验对照臂（`*_murray_cap`）本批不加；收尾做 E3、E5；不设过门判据，只建基线，recover8 只作描述。cfd_auto 占着 CPU 分区，数据准备只用 node04 的 CPU；GPU 六张都可用（master 4 × 4090 + node04 2 × A100），并行方式由我判断。

**数据准备（node04 CPU，Slurm 之外，16:11–16:25）**，工具 `training_wss_min/tools/prepare_pf6vf6_v52d_retrain.py`（build / compare / assemble / stats / featstats / configs / cleanup），记录在 `experiments/pf6vf6_v52d_retrain_20261002/data_prep/`：

- **逐数组比对：没有不一致。** 273 个真实单元全部从 v5.2d 快照用 `wss_v5.views.volume_view.build_case_volume` 重建（32 进程，253 s，0 错误）。已有的 261 份体场视图（主视图根 159 + 全周期缓存 102）与重建结果逐数组（键、dtype、形状、数值）**全部完全一致**，没有需要替换的单元。其中包括 6 个 09-30 标签修正单元（MENG_GUANG_QIN、CHEN_SHU_LIN、LIU_ZONG_YANG、SUN_ZONG_GE、ZHANG_YAN_SHAN-0/before、ZHANG_YONG_SHENG-0/before），它们在主视图根里的体场视图已经是修正后的。全周期缓存 102 份的来源签名（h5 / bundle 的路径、大小、mtime）与当前文件一致，构建器 sha256 相同（`volume_view.py` 3d922d7…）。比对完的重建副本已删除。
- **体场数据根** `data_wss_v5/views_v5_2d_20261001/wss_min_volview_v1/`：每单元 `bundle.npz` 链接主视图根；`volume.npz` 有 159 个链接主视图根、102 个链接 `experiments/joint_cycle_v52d_20261001/data_audit_cache/derived_volume/`，12 个是新建文件（YANG_BAO_KUI、ILO/LI_FA_XIANG-1/before、ILO/WANG_CAI-0/before、ILO/XUE_YOU_TANG-0/after、recover8 的 8 个）。`volview_manifest.json` 记了每单元的来源和 sha256。主视图根没有写入（仍是 159 个 `volume.npz`），全周期缓存没有改动。注意：这个根依赖上述两处的 `volume.npz`，使用期间不要删除它们。
- **体场统计（线性 z）六套**，放在 `wss_min_volview_v1/stats/volume_stats_{分区}_train{n}.json`：

  | 分区 | 训练例数 | 压力 均值 ± 标准差（Pa） | 速率 均值 ± 标准差（m/s） |
  | --- | ---: | ---: | ---: |
  | cv5_fold0 | 206 | −196.3 ± 608.3 | 0.280 ± 0.315 |
  | cv5_fold1 | 204 | −177.5 ± 518.1 | 0.279 ± 0.307 |
  | cv5_fold2 | 211 | −187.4 ± 587.8 | 0.276 ± 0.311 |
  | cv5_fold3 | 211 | −187.8 ± 571.5 | 0.277 ± 0.307 |
  | cv5_fold4 | 212 | −175.0 ± 544.2 | 0.269 ± 0.299 |
  | full265 | 265 | −184.0 ± 566.7 | 0.274 ± 0.307 |

  （参照：v5.0 train138 是 −213.1 ± 594.2 Pa、0.309 ± 0.332 m/s。）
- **输入特征归一化六套**：`experiments/pf6vf6_v52d_retrain_20261002/feature_stats/{PF6,VF6}_{分区}_train.json`，与 train.py 同一个调用（`load_partition` + `compute_feature_stats`，体场点集 = 壁面节点 ∪ 内部单元）。PF6 与 VF6 在每个分区上逐值相同（特征与目标无关）。

**配置**：`configs/pf6vf6_v52d_retrain_20261002/`，36 个臂配置 + `matrix.json`（另有按主机拆分的 `matrix_master.json` 25 臂、`matrix_node04.json` 11 臂）。源配方：seed 1234 取 `configs/wss_local_wave2_20260912/{PF6,VF6}_s1234.json`（wave3 目录只有 seed 7 / 2025；三个 seed 的源配置之间只差 seed、name、notes、特征统计路径和配对初始化参照），seed 7 / 2025 取 `configs/wss_local_wave3_20260913/`。工具逐字段校验：36 个配置相对各自的源配置只改了 `data.data_root`、`data.split_path`、`data.wss_stats_path`、`data.feature_stats_path`、`data.point_features_root`（指向 v5.2d flowref）、`name`、`notes`；配对初始化参照不变。run 全部写到 `runs/pf6vf6_v52d_retrain_20261002/`。

**GPU 分配（先测速再定）**：短程测速（PF6 cv5 fold0 配方，60 个训练单元，8 轮；`experiments/<exp>/bench/bench_summary.json`）：一卡一臂每轮 4090 2.25 s、A100 2.2 s，两种卡同速；一卡两臂每臂 3.9 s / 4.07 s，总吞吐只多 15 % / 8 %，单臂慢约 1.8 倍，CPU 和内存占用翻倍（以往 X5D 长程两槽总吞吐反而降 6–13 %）。所以**六张卡每卡一个臂**，按 GPU 时等分：node04 跑 6 个 full265 臂 + PF6 CV5 seed 2025 的五折（11 臂），master 跑其余 25 臂。A100 与 4090 训练的臂只在 seed 噪声意义上可比，主机记在各队列状态文件里。正式训练实测每训练例每轮 4090 约 31 ms、A100 约 31.5 ms，与测速一致。

**冻结代码与预检**：冻结副本 `GNN_pf6vf6_v52d_frozen_20261002`，`training_wss_min/*.py` 与主树、与 §41 的 `GNN_v52d_frozen_20261001` 逐字一致。

- 锚点：v5.2 之后没有可复现的 PF6 / VF6 旧 run，所以用已部署的 v5.0 PF6_s1234 / VF6_s1234，指向新体场根，在 recover8 去掉 ILO/LI_JIE-1/after 后的 7 例上评估：先用训练 v5.1 折底座的旧代码（`GNN_voltime_frozen_20260919`）写出存档指标，再用冻结代码在同一张卡上重评、逐字段比对。压力 1798 个字段最大差 7.1e-5，速度 1352 个字段最大差 5.1e-7，R²_cb 一致到 1e-9。
- 前两次预检失败（都在训练前，没有任何 run 产生）：16853（4 s）旧代码的 split 加载器只接受 ILO before，recover8 里的 LI_JIE-1/after 被拒，于是锚点改用 7 例 split（`anchor/split_anchor_recover7_old_code_compatible.json`；E5 仍用完整 recover8）。16857（1.5 min）压力锚点有 1 个点越过热点 top10 % 阈值（FU_GUO_JUN 的 n_high_pred 41610 对 41611，4 个视图里是同一个数），超出 5e-4 的绝对容差；诊断作业 16863 让同一份新代码连跑两次，连续字段也有 5.5e-5 的抖动，属于 GPU 浮点非确定性。为此给 `tools/preflight_wss_local_wave1.py` 加了显式开关 `--anchor-pred-count-tolerance`（默认关闭，旧用法逐位不变）：只把依赖预测的点计数（n_*pred*，此锚点 28 个字段）放宽到 ±2 个点，其余字段仍是 5e-4；`check_pf6vf6_anchor` 用同一规则。
- **预检 16866 通过**（16:34–16:37）：两个锚点通过；36 臂配对初始化、AMP 三步、全量查询推理正常（单臂显存峰值 ≤ 845 MiB，分块推理最大差 3.7e-8）；压力臂 PF6_v52cv_f0_s7 与速度臂 VF6_v52cv_f0_s7 冒烟（训练 2 轮 + 评估 best / last）通过。

**提交（10-02 16:37）**：

- master 队列 **16867**（afterok 16866）：4 × 4090 每卡一个，25 臂；每臂 train → eval best → eval last（`--save-predictions`）；每次派发前复核预检指纹。
- node04 队列（Slurm 之外，`cluster/node04_pf6vf6_v52d_queue.sh`）：等预检通过后复核同一套指纹与臂覆盖，用 `run_local_train_queue --only` 跑 11 臂，状态文件 `experiments/<exp>/queue_status_node04.json`；16:37:52 开始训练。
- 收尾 **16868**（afterany 16867）：先等 node04 队列写出 finished_at（最多 16 h），再做 E3（30 个 CV5 折模型的 best 评 recover8 → `recover8_cv5/`）、E5（已部署 v5.0 PF6 / VF6 三 seed 评 recover8 → `e5_deployed_v50/*/eval`），最后出读数 `readout_ckpt_{best,last}.{md,json}`。
- 预计（按实测速度：4090 CV5 臂每轮 6.2–6.7 s，A100 full265 臂每轮 8.35 s）：node04 约 22:30 完成，master 约 23:15 完成（25 臂里有一张卡要跑 7 个），收尾与读数约 10-03 01:00（误差 ±1 h）。
- 队列自带的收尾报告 `report_wss_local_wave1` 会因矩阵没有 `anchor_run` 报错（同 §41，`report_failure.json`），不影响训练与评估；正式读数由 16868 生成。

**读数工具** `tools/report_pf6vf6_v52d_retrain.py`：R²_cb 按 `metrics.casebalanced_field_metrics` 的定义，从逐单元充分统计量合并，跨折、跨队列、分壁面 / 内部都是精确的。在 v5.1 折底座的存档预测上验证：压力（全部 / 壁面 / 内部 / 分队列 / 逐例中位）、速度（速率、向量 RMSE、方向余弦、分量 R²）都与 metrics.json 一致到 1e-10 级；正式读数里对每个 run 自动做同样的一致性检查。集成 = 逐点平均预测，速度先平均向量再取速率。历史数字（v5.0 test34、v5.1 cv3）单独成表，只并列、不相减。

**已知数据限制**：同 §41，150 个每步迭代提前退出的单元（AG 105、AAA 25、ILO 20）标签保持原样。

**证据**：数据 `data_wss_v5/views_v5_2d_20261001/wss_min_volview_v1/`（`volview_manifest.json`、`stats/`）；准备记录 `experiments/pf6vf6_v52d_retrain_20261002/data_prep/`（`build_report.json`、`compare_report.json`、`stats_report.json`、`featstats_report.json`、node04 日志）；测速 `bench/`；锚点 `anchor/`（旧代码存档指标、`VF6_anchor_check_16866.json`、诊断 16863）；预检 `runtime_preflight.json`；Slurm `cluster/{preflight,run,post}_pf6vf6_v52d*.slurm` 与 `cluster/node04_pf6vf6_v52d_queue.sh`。

### 43.2 结果（2026-10-02 22:57；36 臂全部完成，收尾 16868 COMPLETED）

一句话：PF6 / VF6 在 v5.2d 上的新基线是 CV5 折外压力 R²_cb 0.800（三 seed 集成 0.808）、速率 0.830（集成 0.843），三 seed 之间 sd 约 0.001；在同一批单元、同一套 v5.2d 标签上，新模型一致好于已部署的 v5.0 权重。

- **执行**：master 队列 16867（25 臂）16:37–22:27，node04 11 臂 16:37–22:19（node04 已释放）；E3 30/30、E5 6/6，无失败。读数一致性检查：最优检查点 72 个 run、末轮 36 个 run，用保存的预测重算 R²_cb 与 metrics.json 最大差 0。队列自带的 `report_wss_local_wave1` 因矩阵无 `anchor_run` 报错（已知，不影响结果）。

**CV5 折外（261 例，最优检查点；末轮检查点结论相同）**

| | 单 seed（1234 / 7 / 2025） | 均值 ± sd | 三 seed 折外集成 | 分队列集成 AG / AAA / ILO |
| --- | --- | ---: | ---: | --- |
| PF6 压力 R²_cb | 0.7989 / 0.8000 / 0.8003 | 0.7997 ± 0.0008 | **0.8084**（壁面 0.8061、内部 0.8087） | 0.856 / 0.862 / 0.767 |
| VF6 速率 R²_cb | 0.8290 / 0.8286 / 0.8310 | 0.8295 ± 0.0013 | **0.8425**（向量误差 0.157 m/s、方向余弦 0.921） | 0.857 / 0.844 / 0.804 |

**recover8（8 例，只作描述）**：full265 三 seed 集成压力 0.939、速率 0.903；15 个 CV5 折模型集成（E3）0.936 / 0.903；已部署 v5.0 三 seed 集成（E5）0.921 / 0.883。最难的是 LIU_YU_MING（压力 0.785、速率 0.731）。

**与旧模型同口径对比**（同一批单元、同一套 v5.2d 标签；`experiments/pf6vf6_v52d_retrain_20261002/compare_old_test34/compare_old_new.md`，旧模型重评作业 16973）：

| 集合 | 旧 v5.0（train138）三 seed 集成 | 新 v5.2d | 逐例改善 |
| --- | ---: | ---: | --- |
| test34（旧模型没见过；新模型用 CV5 折外预测）压力 | 0.821 | 0.851 | 21/34 |
| test34 速率 | 0.825 | 0.848 | 32/34 |
| recover8 压力（新 = full265） | 0.921 | 0.939 | 7/8 |
| recover8 速率 | 0.883 | 0.903 | 8/8 |

- 速度的提升稳定（几乎每例都好）；压力的提升主要来自几例原来很差的单元（SUN_SHU_MING 0.51 → 0.71、SUN_ZHI_YU 0.64 → 0.83、ZHANG_YONG_SHENG-0/before 0.74 → 0.94），且 34 例上新模型单 seed 之间差 0.83–0.86，与提升同量级。
- 差值混着训练数据量（每折约 209 例 / 265 例对 138 例）、v5.0 之后的数据修正和旧模型训练标签中的错误，不单独归因。旧模型在 v5.2d 标签上（单 seed 约 0.81）比当年 v5.0 标签下的 0.789 / 0.796 高，是修正后标签本身的作用。test34 是旧模型的开发暴露集，对旧模型略偏乐观。
- v5.0 test34、v5.1 cv3 的历史数字是另外的评估集，只并列不相减（读数文件末表）。

**证据**：`experiments/pf6vf6_v52d_retrain_20261002/readout_ckpt_{best,last}.{md,json}`、`compare_old_test34/`；run `runs/pf6vf6_v52d_retrain_20261002/`。

**工作簿回填（2026-10-03 01:33）**：「速度与压力实验矩阵」第 140–229 行，组名「体场 PF6/VF6 v5.2d 复验｜2026-10-02」：36 臂 × best / last 共 72 行（CV5 臂读留出折、full265 臂读 recover8）+ 18 行汇总（每个目标：CV5 折外三个单 seed 与三 seed 集成、recover8 的 full265 / E3 / E5 集成、test34 新旧配对；只有 test34 配对填 ΔR²）。工具 `tools/update_pf6vf6_v52d_xlsx.py`（幂等，146 220 个历史单元格逐格不变，备份 `docs/03-汇报材料/…_backup_before_pf6vf6_v52d_retrain_20261002_20261003_013212.xlsx`，证据 `experiments/pf6vf6_v52d_retrain_20261002/xlsx_acceptance.json`）；说明层 `annotate_workbook_methods.py` 加 3 条规则后重跑，90 个标签格加「｜说明」，再跑改动 0。

## 44. 体场 v5.2d 全量模型上线，v5.0 体场包下线（2026-10-03，wss_deploy v0.16.4）

用户 10-03 要求：新的体场模型部署上线，旧的下线。部署侧细节在 [05 块框架文档 §29.14](../05-部署工具/WSS_部署演示工具_从STL到峰值WSS_整体框架与计时_2026-09-17.md)。

- **新发布包** `PF6_VF6_v52d_3seed_20261003`：§43 的 full265 PF6 × 3 seed + VF6 × 3 seed，配方不变；`wss_deploy/build_v52d_volume_release.py` 冻结，验证块读自 §43.2 读数；几何参照用 full265 中心线。
- **下线**：`PF6_VF6_peak_3seed_20260920`（v5.0，train138）移到 `outputs/wss_deploy_release_retired/`，旧结果照常查看、显示为「（旧版）体内压力与速度」。
- **说明卡口径**：主数字 = 同配方 CV5 折外三 seed 集成（压力 0.81、速率 0.84，261 例），另列 recover8；写明数字是 CFD 网格口径，部署体内采样与之是否等价尚未建立。
- **部署链路验收**（recover8 的 STL，自动命名，同几何同查询点，最近 CFD 点配对，病例等权）：压力 0.913 → 0.930（6/8 例改善）、速率 0.870 → 0.889（8/8）；部署加载器对训练评估逐点最大差压力 ≤ 8.6e-4 Pa、速度 ≤ 1.5e-6 m/s；黄金回归 6/6；全套测试 1072 通过。记录 `training_wss_min/experiments/wss_deploy_timing_20260917/acceptance_recover8_volume_v52d_20261003/`。
- **上线**：10-03 01:09 `service upgrade`，PID 1161932，版本 0.16.4；启动预加载只有现役三包（X5Dcap_asym2_v52d、M1cap_v52d、PF6_VF6_v52d）。

