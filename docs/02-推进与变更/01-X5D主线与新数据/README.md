# 01 · X5D 主线与新数据

> 这块回答：峰值帧直接 WSS 的部署底座（X5D）在更多数据上能到多少，要不要换配方。数据怎么回收、怎么入库见 [04 块](../04-数据处理与CFD/README.md)。

## 当前状态（2026-10-03）

- **数据口径（2026-10-01 起）**：训练和测试只用唯一数据版本 **v5.2d**：视图 `data_wss_v5/views_v5_2d_20261001`，快照 `data_wss_v5/anatomy_pointcloud_v5_2d_20261001`，332 单元（273 真实 + 59 合成）。它是母库全量审计后在 v5.2c 上合并出来的：HAN_JIAN_FU 壁面标签换成重算结果、WANG_TIAN_QING-1/after 从本例 STL 重建、16 个单元修正出口命名、分流规则重拟合、统计重算；v5.2c 的根目录已改名，不再存在。21 个重训配置在 `training_wss_min/configs/wss_v52d_retrain_20261001/`，10-01 16:33 提交、10-02 全部训完（§41）；体场 PF6 / VF6 用独立的体场数据根 `views_v5_2d_20261001/wss_min_volview_v1/`（§43.1）（[母库全量审计 §11](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/母库全量审计_2026-10-01.md)、[跟踪 §40](./X5D主线_实验跟踪.md)）。下一句是 09-30 的 v5.2c 说明，留作历史：视图 `data_wss_v5/views_v5_2c_20260930`，快照 `data_wss_v5/anatomy_pointcloud_v5_2c_20260930`，332 单元；IND、CV5、syn、full265 + recover8 分区都在其中。v5.2 / v5.2p4 上的旧读数只作历史参照（标签修正前）；旧根已删除。23 个 X5Dcap_asym2 重训配置已生成、未训练（[跟踪 §38](./X5D主线_实验跟踪.md)、[数据统一](../04-数据处理与CFD/_archive/04块过程文档_2026-10-02/数据统一与旧版本清理_2026-09-30.md)）。
- **训练底座（配方 2026-09-29 定，数字 2026-10-02 起用 v5.2d 读数，用户确认）**：v5.2d 数据 + **X5Dcap_asym2** 配方（X5Dcap + 低估逐点损失 ×2，配方未变）。CV5 折外（261 例）三 seed 各自 pooled 0.7750 / 0.7752 / 0.7780，均值 **0.7761 ± 0.0017**，三 seed 折外集成 **0.7926**；全量 265 例 → recover8 三 seed 集成 0.8393。与 v5.2 上的 0.7714 相比，约 +0.004 来自评估标签修正，其余在 seed 波动内，不是配方或重训带来的提升。IND 170 / 91 协议没有在 v5.2d 上重训，旧的 IND 五 seed 集成 0.7758 和 X5Dcap 对照（CV5 0.7608、IND 0.7644）是 v5.2 口径，只作历史参照。新实验的源配置和配对参照改为 `wss_v52d_retrain_20261001` 的同 (fold, seed) run，见 [§36 矩阵 §6 底座指针](./WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md)；配方裁定见 [跟踪 §36.5](./X5D主线_实验跟踪.md)，读数见 [跟踪 §41](./X5D主线_实验跟踪.md)。
- **部署发布包（10-02 已切换，§42）**：默认 `X5Dcap_asym2_v52d_3seed_20261002`（§41 的 full265 三 seed，峰值 WSS），三头 `M1cap_v52d_3seed_20261002`；体场 10-03 起是 `PF6_VF6_v52d_3seed_20261003`（§44，旧 v5.0 体场包下线）。09-16 至 10-01 的部署底座 X5D_v51 五 seed（test34 Pa R²_cb 0.7749、cv3 折外 0.7100，[历史卷 §28](../00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)）已下线，发布包移到 `outputs/wss_deploy_release_retired/`。
- **§37 全量 265 例训练（已完成，2026-09-30）**：数据版本 v5.2p4 full265（v5.2 261 + YANG_BAO_KUI + 3 个同病人搭档已在 v5.2 的恢复单元），X5Dcap_asym2 三 seed（1234/7/2025）。recover8 集成 R²_cb **0.8393**（同 8 例 IND 三 seed 0.8352、五 seed 0.8393），逐例中位 0.843，6/8 例改善；n = 8 只作描述。权重在 `runs/wss_v52p4_full265_20260930/`；它们是标签修正前的训练，没有打包，部署用的是 §41 在 v5.2d 上重训的同配方权重（§42）。见 [跟踪 §37](./X5D主线_实验跟踪.md)。
- **§41 v5.2d 重训（已完成，2026-10-02）**：21 臂全部完成。CV5 折外（261 例）三 seed 均值 **0.7761 ± 0.0017**、三 seed 集成 **0.7926**；v5.2 同配方模型在旧标签上是 0.7713、换成修正后的标签是 0.7752，所以数据修正的作用在评估标签上（错标签单元折外得分回升，如 MENG_GUANG_QIN 0.34 → 0.80），重训本身无可分辨增益（集成差 −0.0004）。full265 → recover8 三 seed 集成 **0.8393**（v5.2p4 同口径 0.8401，持平）；15 个 CV5 折模型集成 0.8444。三头 full265 三 seed 集成：峰值 0.823、TAWSS 0.849、OSI 0.542，三通道都高于已部署三头（0.792 / 0.805 / 0.490），护栏全过。底座数字已按用户 10-02 的确认改用这组读数。见 [跟踪 §41](./X5D主线_实验跟踪.md)。**§42 部署切换（10-02 已上线）**：full265 峰值三 seed 与三头三 seed 打包为 `X5Dcap_asym2_v52d_3seed_20261002`（默认）和 `M1cap_v52d_3seed_20261002`，X5D_v51 与旧三头下线；recover8 的 STL 走部署链路，峰值 0.830 → 0.854、三头 TAWSS 0.809 → 0.850、OSI 0.445 → 0.489。见 [跟踪 §42](./X5D主线_实验跟踪.md)。**§40 数据已更新为 v5.2d（10-01）**；**§39 v5.2c 重训（09-30 22:55 已按用户要求取消）**：X5Dcap_asym2 CV5 × 3 seed + full265 × 3 seed，三头 M1cap full265 × 3 seed，全部 master 4 × 4090 每卡一个（预检 16389 已过 → 队列 16390 → 附加评估与读数 16391）；recover8 评估另加 E3（CV5 折模型）和 E5（已部署三头）。16390 运行 2 h 37 min 后取消，无可用结果。见 [跟踪 §39](./X5D主线_实验跟踪.md)。
- **§43 体场 PF6（压力）/ VF6（速度）v5.2d 复验（已完成，10-02）+ §44 上线（10-03）**：配方不变只换数据，PF6 / VF6 各 CV5 五折 × 3 seed + full265 × 3 seed，36 臂。CV5 折外（261 例）三 seed 均值压力 R²_cb 0.7997 ± 0.0008、速率 0.8295 ± 0.0013，三 seed 集成 **0.808** / **0.843**；recover8 full265 集成 0.939 / 0.903。同一批单元、同一套 v5.2d 标签上一致好于旧 v5.0 权重（test34 集成压力 0.821 → 0.851、速率 0.825 → 0.848；recover8 0.921 → 0.939、0.883 → 0.903）。体场数据根 `wss_min_volview_v1`（已有体场视图与快照重建逐数组一致）。10-03 全量权重上线为 `PF6_VF6_v52d_3seed_20261003`（wss_deploy v0.16.4），v5.0 体场包下线；部署链路 recover8 压力 0.913 → 0.930、速率 0.870 → 0.889。见 [跟踪 §43 / §44](./X5D主线_实验跟踪.md)。
- **§33 learning curve（已完成）**：数据量仍是瓶颈，75→100 % 段每翻倍 +0.035、三段斜率不收窄 → 触发了 04 块的数据回收。
- **§35 v5.2 基线重训（已完成，09-24）**：IND + CV5 双协议、X5D / X5Dcap 双配方 40 臂；裁定 v5.2 基线 = X5Dcap（CV5 折外 0.761 ± 0.002、IND 集成 0.764）。
- **§36 三方向（已收口，09-28）**：一维 Womersley 先验不过；低估加权损失 T2 两级门全过 → 底座；形变合成 CFD 59 例 No-Go（CV5 +0.001、IND +0.004）；用 cfd_auto 做的第二轮形变扩充也已于 10-02 按用户裁定暂停（不能提高真实病人上的整体精度，见 [04 块分析报告](../04-数据处理与CFD/合成几何数据扩充_分析报告_2026-10-02.md)）。剂量对照 ×3.0 已完成（IND 配对 +0.003，低于 ×2.0 的 +0.012，2.0 在最优附近）；×1.5 三臂在 09-29 下午停在第 392–397 / 400 轮（与 node04 当天停机重启吻合），没有完成训练和评估，只作证据的这一点缺着；用户 10-02 决定暂不补跑。

## 文件

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| [X5D主线_实验跟踪](./X5D主线_实验跟踪.md) | §33 learning curve、§35 v5.2 基线重训、§36 三方向与裁定、§37 全量 265 例训练与 recover8 读数、§38 标签修正版数据 v5.2c / v5.2p5 与重训配置、§39 v5.2c 重训（CV5 / full265 / 三头）、§40 数据更新为 v5.2d、§41 v5.2d 重训与读数、§42 v5.2d 模型上线、§43 体场 PF6 / VF6 v5.2d 复验、§44 体场模型上线 | 活跃，后续结果续写在这里 |
| [WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26](./WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md) | §36 预注册矩阵、执行记录、裁定与底座指针 | 已收口（09-29） |
| [`_archive/`](./_archive/README.md) | 局部形态实验矩阵（09-17） | ⏸️ 已暂停 |

## 证据路径

- 配置与预注册：`training_wss_min/configs/wss_v52_20260923/`（`matrix.json`）、`training_wss_min/configs/wss_v52_20260923_cv5/`
- 运行与读数：`training_wss_min/experiments/wss_v52_20260923/`、`training_wss_min/experiments/wss_v52_20260923_cv5/`
- 生成工具：`training_wss_min/tools/prepare_wss_v52.py`（读取 [04 块](../04-数据处理与CFD/README.md) 的 `new_units_manifest.json`）
- learning curve：`training_wss_min/experiments/wss_learning_curve_20260920/`
- §36：配置 `configs/wss_v52_phys_20260926/`（P1–P3、T1–T3 IND）、`configs/wss_v52_phys2m_20260927/`（T2 CV5）、`configs/wss_v52_phys2e_20260928/`（T2 seed 11/2026）、`configs/wss_v52_phys2na|nb_20260927/`（剂量对照，node04）、`configs/wss_v52_synm|synn_20260927/`（合成臂）；读数 `experiments/wss_v52_phys_20260926/readout_phys_ckpt_best.md`、`experiments/wss_v52_phys2m_20260927/readout_phys2_ckpt_best.md`、`experiments/wss_v52_phys2e_20260928/readout_t2_ensemble.md`、`experiments/wss_v52_synm_20260927/readout_syn_ckpt_best.md`
- §37：配置 `configs/wss_v52p4_full265_20260930/`、执行与证据 `experiments/wss_v52p4_full265_20260930/`（`prepare_full265.py`、`config_diff_vs_source.*`、`data_checks.json`、IND 配对参照 `ref_ind_recover8/`、工作簿证据 `xlsx_acceptance.json`；工作簿 WSS实验矩阵 603–605 行）、视图根 `data_wss_v5/views_v5_2p4_full265_20260930/`、读数 `outputs/cfd_auto_trial_20260927/_recover/eval/readout_recover8_full265_vs_ind.md`、权重 `training_wss_min/runs/wss_v52p4_full265_20260930/`
- §43：配置 `training_wss_min/configs/pf6vf6_v52d_retrain_20261002/`（`matrix.json` + `matrix_master.json` / `matrix_node04.json`）、执行与证据 `training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/`（`data_prep/`、`bench/`、`anchor/`、`runtime_preflight.json`、队列状态、读数）、体场数据根 `data_wss_v5/views_v5_2d_20261001/wss_min_volview_v1/`、run `training_wss_min/runs/pf6vf6_v52d_retrain_20261002/`、工具 `tools/prepare_pf6vf6_v52d_retrain.py` / `tools/report_pf6vf6_v52d_retrain.py` / `tools/check_pf6vf6_anchor.py`、Slurm `cluster/{preflight,run,post}_pf6vf6_v52d*.slurm` + `cluster/node04_pf6vf6_v52d_queue.sh`、冻结副本 `GNN_pf6vf6_v52d_frozen_20261002`
- 合成子例：`data_new/<母>~mNN`（壁面 81 帧 + 峰值体场 + cas）、老师包 `outputs/synth59_teacher_package_2026-09-28/`、生成与验收记录 `04 块/数据回收_2026-09-20/synth_20260926/`

## 相关的已结案探索（在历史卷里）

§20–§22（X5 / X5D 的由来）、§28（v5.1 重训）、§29（沿程 32 通道，不进主线）、§30 局部形态 Wave B（不进底座）。
