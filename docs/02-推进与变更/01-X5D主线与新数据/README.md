# 01 · X5D 主线与新数据

> 这块回答：峰值帧直接 WSS 的部署底座（X5D）在更多数据上能到多少，要不要换配方。数据怎么回收、怎么入库见 [04 块](../04-数据处理与CFD/README.md)。

## 当前状态（2026-09-30）

- **数据口径（2026-09-30 起）**：训练和测试只用唯一数据版本 **v5.2c 统一根**：视图 `data_wss_v5/views_v5_2c_20260930`，快照 `data_wss_v5/anatomy_pointcloud_v5_2c_20260930`，332 单元；IND、CV5、syn、full265 + recover8 分区都在其中。v5.2 / v5.2p4 上的旧读数只作历史参照（标签修正前）；旧根已删除。23 个 X5Dcap_asym2 重训配置已生成、未训练（[跟踪 §38](./X5D主线_实验跟踪.md)、[数据统一](../04-数据处理与CFD/数据统一与旧版本清理_2026-09-30.md)）。
- **训练底座（2026-09-29 起）**：v5.2 数据（261 例：170 在库 + 91 回收）+ **X5Dcap_asym2** 配方（X5Dcap + 低估逐点损失 ×2）。CV5 三 seed 折外 pooled 0.7714（X5Dcap 0.7608）、IND 五 seed 集成 91 例 0.7758（X5Dcap 0.7644）；尾部低估比例 0.73 → 0.66；取舍：逐例中位持平、AAA 略降。裁定见 [跟踪 §36.5](./X5D主线_实验跟踪.md)，底座指针见 [§36 矩阵 §6](./WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md)。
- **现役部署底座（未切换）**：X5D_v51 五 seed 集成，test34 Pa R²_cb 0.7749、cv3 折外 0.7100（[历史卷 §28](../00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)）；发布包 `outputs/wss_deploy_release/X5D_v51_5seed_20260916/`。261 例全量 X5Dcap_asym2 权重与部署切换另行裁定。
- **§37 全量 265 例训练（已完成，2026-09-30）**：数据版本 v5.2p4 full265（v5.2 261 + YANG_BAO_KUI + 3 个同病人搭档已在 v5.2 的恢复单元），X5Dcap_asym2 三 seed（1234/7/2025）。recover8 集成 R²_cb **0.8393**（同 8 例 IND 三 seed 0.8352、五 seed 0.8393），逐例中位 0.843，6/8 例改善；n = 8 只作描述。权重在 `runs/wss_v52p4_full265_20260930/`，未打包、部署未切换。见 [跟踪 §37](./X5D主线_实验跟踪.md)。
- **§33 learning curve（已完成）**：数据量仍是瓶颈，75→100 % 段每翻倍 +0.035、三段斜率不收窄 → 触发了 04 块的数据回收。
- **§35 v5.2 基线重训（已完成，09-24）**：IND + CV5 双协议、X5D / X5Dcap 双配方 40 臂；裁定 v5.2 基线 = X5Dcap（CV5 折外 0.761 ± 0.002、IND 集成 0.764）。
- **§36 三方向（已收口，09-28）**：一维 Womersley 先验不过；低估加权损失 T2 两级门全过 → 底座；形变合成 CFD 59 例 No-Go（CV5 +0.001、IND +0.004），搁置至 cfd_auto STL→网格链走通后用更多形变再做。剂量对照 ×1.5/×3.0 在 node04 收尾。

## 文件

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| [X5D主线_实验跟踪](./X5D主线_实验跟踪.md) | §33 learning curve、§35 v5.2 基线重训、§36 三方向与裁定、§37 全量 265 例训练与 recover8 读数、§38 标签修正版数据 v5.2c / v5.2p5 与重训配置 | 活跃，后续结果续写在这里 |
| [WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26](./WSS_V5_一维物理先验_尾部目标_合成数据_实验矩阵_2026-09-26.md) | §36 预注册矩阵、执行记录、裁定与底座指针 | 已收口（09-29） |
| [`_archive/`](./_archive/README.md) | 局部形态实验矩阵（09-17） | ⏸️ 已暂停 |

## 证据路径

- 配置与预注册：`training_wss_min/configs/wss_v52_20260923/`（`matrix.json`）、`training_wss_min/configs/wss_v52_20260923_cv5/`
- 运行与读数：`training_wss_min/experiments/wss_v52_20260923/`、`training_wss_min/experiments/wss_v52_20260923_cv5/`
- 生成工具：`training_wss_min/tools/prepare_wss_v52.py`（读取 [04 块](../04-数据处理与CFD/README.md) 的 `new_units_manifest.json`）
- learning curve：`training_wss_min/experiments/wss_learning_curve_20260920/`
- §36：配置 `configs/wss_v52_phys_20260926/`（P1–P3、T1–T3 IND）、`configs/wss_v52_phys2m_20260927/`（T2 CV5）、`configs/wss_v52_phys2e_20260928/`（T2 seed 11/2026）、`configs/wss_v52_phys2na|nb_20260927/`（剂量对照，node04）、`configs/wss_v52_synm|synn_20260927/`（合成臂）；读数 `experiments/wss_v52_phys_20260926/readout_phys_ckpt_best.md`、`experiments/wss_v52_phys2m_20260927/readout_phys2_ckpt_best.md`、`experiments/wss_v52_phys2e_20260928/readout_t2_ensemble.md`、`experiments/wss_v52_synm_20260927/readout_syn_ckpt_best.md`
- §37：配置 `configs/wss_v52p4_full265_20260930/`、执行与证据 `experiments/wss_v52p4_full265_20260930/`（`prepare_full265.py`、`config_diff_vs_source.*`、`data_checks.json`、IND 配对参照 `ref_ind_recover8/`、工作簿证据 `xlsx_acceptance.json`；工作簿 WSS实验矩阵 603–605 行）、视图根 `data_wss_v5/views_v5_2p4_full265_20260930/`、读数 `outputs/cfd_auto_trial_20260927/_recover/eval/readout_recover8_full265_vs_ind.md`、权重 `training_wss_min/runs/wss_v52p4_full265_20260930/`
- 合成子例：`data_new/<母>~mNN`（壁面 81 帧 + 峰值体场 + cas）、老师包 `outputs/synth59_teacher_package_2026-09-28/`、生成与验收记录 `04 块/数据回收_2026-09-20/synth_20260926/`

## 相关的已结案探索（在历史卷里）

§20–§22（X5 / X5D 的由来）、§28（v5.1 重训）、§29（沿程 32 通道，不进主线）、§30 局部形态 Wave B（不进底座）。
