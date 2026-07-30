# training_wss_min/configs — 主题目录对照

配置按**实验主题**分目录，不再使用 `round5` / `round6` 等轮次目录名。

**兼容约定**：JSON 文件名可语义化；文件内 `name` 字段与历史 `runs/<name>/` **保持原样**（含 `r2_`/`r3_`/`r4_`/`r5_`/`r6_`），便于复评与追溯。新实验生成器应直接写语义 `name`。

**test27采样/指标语义**：`random`=训练壁面顶点均匀无放回，`area_random`=三角面面积映射后的表面积采样；两者不可混名。`eval.surface_metric_mode=legacy_vertex`不消费面积映射且不生成area字段，`both_strict`才启用面积指标并逐例执行硬门。当前Phase-V六组为前者，Phase-A六组为后者。

| 目录 | 主题（核心问题） | 历史对照 | 状态 |
|---|---|---|---|
| `baseline_sweep/` | 点数 / 特征 / 采样正交扫 | 第 1 版 baseline | 归档 |
| `baseline_2x3/` | MLP / PointNet / PointNet++ × xyz / xyz+geom | 2026-07-14 最小矩阵 | **冻结保留** |
| `pointnet_trainloss_e400/` | PointNet × xyz / xyz+geom；train/test-only；按 train_loss 选模；400 epoch | 2026-07-14 | **完训+完评 `8968` / `8974–8975`** |
| `pointnet_wide128/` | PointNet+xyz+geom 容量探针：`width=128`/`head_hidden=256`（相对老师多一层 128 过渡）；FPS-2000；val-only | 2026-07-14 | **完训 `8970`｜补充 No-Go** |
| `pointnet_distribution_matrix/` | E2 精确导师通道 / E3 random-5000 × GLOBAL/CASE；另含 E23 宽网+5000 点交互对照 | 2026-07-14/15 | **五组完训+完评 `8976–8980`；E2 最强，CASE No-Go** |
| `pointnet_v4/` | AG-v4/混合 E2；test27 Support/Query 的 vertex `random` 与 `area_random` SAME/SEP | 2026-07-16/17 | **9170模型资产完成；P1V/P2V=`10473/10474`已提交；P1/P2面积组待修** |
| `pointnetpp_v4/` | 三层 SA；test27 fixed center、fixed-support/full-query、采样与 SAME/SEP | 2026-07-16/17 | **9167–9169已有；Q0/Q1V/Q2V/Q3V=`10475–10478`已提交；Q1–Q4面积组待修** |
| `pointnet_deeper/` | E4 导师追加深度探针：`6→64→128→256→512；1024→512→256→128→64→1`，其余严格对齐 E2 | 2026-07-15 | **Job `8999` 完训+完评｜test 退化｜No-Go** |
| `pointnetpp_sa_foundation/` | PointNet++ 三层 SA 结构审计：FPS-2000，中心 `500→125→32` | 2026-07-15 | **foundation 已审计；正式待提交配置已独立固化到 `pointnetpp_v4/`** |
| `pointnetpp_d2_k64_ilo_structure_20260723/` | D2 c125×k64 的 ILO 两协议与 PointNeXt-R/LocalGeoPE/Attention/SEP 单变量矩阵 | 2026-07-23 | **9/9 完训；S3-GEOPE 三种子通过** |
| `pointnetpp_s3_regularization_20260724/` | S3-GEOPE 的 head dropout / DropPath / 保覆盖 NeighborDrop 单变量矩阵 | 2026-07-24 | **三种子首波 12/12、seed1234 强度/交叉补齐 20/20 均完成** |
| `pointnetpp_regp10_transformer_20260726/` | REG-P10 上的 7 个局部 stage 非空子集与复用 SA3 coarse global attention 对照 | 2026-07-26 | **8/8 完训；L-SA2 `ΔR²_cb=+0.0214` 唯一过 Gate；后续不补多 seed，转入目标/IND 矩阵** |
| `pointnetpp_d2_c125_k64_pnxr_geope_transformer_20260726/` | fixed 106/0/27 D2 PNXR+7D LocalGeoPE 上的 local-SA1 / local-SA2 严格对照 | 2026-07-26 | **2/2 完训；SA1 `-0.0293` No-Go，SA2 `+0.0015` 持平不晋级** |
| `pointnetpp_regp10_edgeconv_20260727/` | REG-P10 既有几何邻域内的静态 EdgeConv 残差消息；SA1 / SA2 / SA1+SA2 | 2026-07-27 | **3/3 完训；`ΔR²_cb=-0.0053/-0.0023/-0.0338`，全部 No-Go** |
| `pointnetpp_rcr_oracle_20260728/` | S3 PNXR+LocalGeoPE 的 geometry / true RCR / shuffled RCR 信息上限矩阵 | 2026-07-28 | **3/3 完训；O1a `ΔR²_cb=+0.0805`，O2 `-0.0033`，真实 RCR Go-to-follow-up** |
| `pointnetpp_o0_hotspot_tail_20260728/` | O0 geometry-only 上的 top10 hotspot BCE / q90 pinball 单变量目标矩阵 | 2026-07-28 | **2/2 完训；H1 `ΔIoU=+0.0170`、H2 `Δhigh-WSS nRMSE=-0.00072`，主 Gate 均未达；No-Go、不组合** |
| `pointnetpp_regp10_lsa2_objective_ind_20260728/` | 历史正式 REG-P10-LSA2 锚点上的 SAME/IND × MSE/H1/H2 2×3 矩阵 | 2026-07-28/29 | **6/6 完训；IND/H1 No-Go，SAME-H2 过 Gate；后续用户决定锁定 H2 并暂缓多 seed** |
| `pointnetpp_lsa2_h2_logradius_20260729/` | 锁定 SAME-H2 后的精确同-seed control / 唯一新增 `log(local_radius)` 输入列 | 2026-07-29 | **2/2 完训；处理臂 `R²_cb=0.3506`，`ΔR²_cb=+0.0436`、`Δhigh-WSS nRMSE=-0.00295`，过 Gate并晋级新单 seed 开发锚点；无多 seed、RCR、面积或 H1 组合** |
| `loss_aug_ablation/` | loss / 采样加权 / 旋转增强 | 原 `r2_*` | 归档 |
| `clean_data/` | clean-data 主矩阵（mse / tgtw × seed） | 原 `r3_*` | 归档 |
| `protocol_gates/` | 固定阈值、Gate-1、B/C 协议锚点 | 原 `r4_dev1_b*`/`c*` | 归档；锚点 `b1_tgtw_fixedq_*` |
| `pointcount_curve/` | 点数—精度曲线（xyz+geom） | 原 `r4_dev1_pc_*` | 归档 |
| `fit_lc_diagnosis/` | 拟合链 / LC / B-REP / A0* 诊断 | 原 `round5/` | 归档（科学结案） |
| `xyz_scale_diag/` | A/B/D XYZ 尺度诊断；C/E 后续 | 原 `round6/` | **A/B/D 已完成；C/E 为当前 P0** |
| `multitarget/` | 壁面压力与多目标横向诊断 | 第六轮 Track A/B | H-PW 已完成；Track B 按 Gate 触发 |
| `sweeps/` | 提交用 manifest（`*.txt`） | 原根目录 `sweep_*.txt` | 随主题更新 |

## 常用入口

```bash
# 已完成资产：壁面 gauge pressure 横向对比
WSSMIN_MANIFEST=training_wss_min/configs/sweeps/pressure_wall.txt \
  bash training_wss_min/cluster/submit_baseline_sweep.sh

# 冻结的最小 2×3 baseline
bash training_wss_min/cluster/baseline_2x3/submit.sh

# PointNet 无 val / 按 train_loss 选模 / 400 epoch
bash training_wss_min/cluster/pointnet_trainloss_e400/submit.sh

# PointNet 加宽容量探针（xyz+geom，相对 2×3 只改 width/head）
bash training_wss_min/cluster/pointnet_wide128/submit.sh

# 已完成矩阵的复跑入口：第一阶段 E2/E3-GLOBAL
bash training_wss_min/cluster/submit_pointnet_distribution_stage.sh global

# 第二阶段 E2/E3-CASE（本轮结果已判 No-Go；仅在明确要求复跑时使用）
bash training_wss_min/cluster/submit_pointnet_distribution_stage.sh case

# 宽 PointNet + random-5000 交互对照
bash training_wss_min/cluster/submit_pointnet_distribution_stage.sh e23

# E4 导师追加 deeper PointNet（单配置完整流水线）
bash training_wss_min/cluster/pointnet_deeper/submit.sh

# PointNet++ 三层 SA 中心点与 ball-query 分组（只读结构审计，不训练）
python -m training_wss_min.tools.visualize_pointnetpp_sa

# v4 PointNet++ 三协议 + 新 split PointNet E2 对照：仅准备清单，禁止直接批量提交
training_wss_min/configs/sweeps/pointnetpp_v4_prepare_only.txt

# 新 test27 采样语义：random=vertex-uniform；area_random=triangle-area mapped。
# vertex/FPS六组已单独通过门禁并提交；面积六组保持both_strict backlog，禁止降级冒名。
python training_wss_min/cluster/submit_v4_support_query_matrix.py --help

# 正式提交前只读门禁（全量加载、哈希、双病例 CPU、RTX4090 AMP）
python -m training_wss_min.tools.preflight_v4_jobs --configs \
  training_wss_min/configs/pointnetpp_v4/ag_v4_sa3_e2_global_fps2000.json \
  training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_locked_sa3_e2_global_fps2000.json \
  training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_e2_global_fps2000.json \
  training_wss_min/configs/pointnet_v4/ag_aaa_v4_stratified_e2_global_fps2000.json \
  --output data_wss_min/pipeline_reports/v4_cutover_20260715_1921/v4_pointnetpp_matrix_preflight.json

# 协议锚点（B1 fixed target-weight）
training_wss_min/configs/protocol_gates/b1_tgtw_fixedq_s1234.json
```

配置生成器已在配置冻结后删除；后续如需新增实验，应直接新增经过审查的 JSON 与
对应 manifest，避免重新引入一次性批量生成脚本。汇总和审计入口位于
`training_wss_min/tools/`。实验指标与结论文档仍以
[`WSS最小化_训练实验跟踪.md`](../../docs/02-推进与变更/WSS最小化_训练实验跟踪.md) 为准。

> 历史兼容说明：`baseline_sweep/feat_*` 与 `loss_aug_ablation/xyzgeom_*` 共 11 份
> 早期配置仍记录壁面常量特征 `dist_to_wall`。这些 JSON 仅作为实验审计证据保留，
> 会被当前配置校验明确拒绝，禁止直接复跑或静默改写。
