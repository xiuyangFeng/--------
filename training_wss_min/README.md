# training_wss_min — WSS-min baseline 训练/评估（PointNeXt 残差版）

与既有 `training/`（V3P 图/变换器栈）**完全独立**，只读 `pipeline_wss_min` 产出的
`data_wss_min/**/bundle.npz` 与 `wss_global_stats.json`。

## 任务

- **任务**：几何点云 `(x,y,z[+几何])` → 壁面 **WSS 标量**；支持 train-only 全局 `log_z` 与逐病例 `WSS/WSSmax` 目标空间，单头。
- **模型**：`PointNeXt-S` 残差版（ball-query）；训练可稀疏采样，**评估恒在完整壁面点云**。
- **开发评估**：默认且通常只允许 `val`；`test` 需 `--allow-test`，不得用于选模。
- **v4**：配置必须显式给出 `data_root` 与 `required_frame_version=stl_landmarks_v4`；canonical AG/AAA ID 可混合加载，split 重复/泄漏直接拒绝。
- **Support/Query**：新增字段缺省时仍是历史 `wall_n_points/sampling + SAME`；新矩阵可用独立 support/query、固定 support 的完整点云分块查询。
- **采样命名**：`random` 是训练壁面顶点等概率无放回；`area_random` 是原始 STL 三角面积映射后的表面积采样。两者是不同物理口径，禁止用 `geom_weighted` 或彼此冒名替代。
- **面积指标门禁**：旧配置缺省 `eval.surface_metric_mode=legacy_vertex`，不读取面积也不输出伪 area 字段；只有显式 `both_strict` 才计算面积指标并要求每例通过严格 mapping。
- **PostView 口径**：`legacy_vertex` 的 high-risk mask 是逐壁面点 top10，导出器不会加载面积；`both_strict` 才使用 area top10。PostView Gaussian `mapping coverage=100%` 仅是可视化插值覆盖率，不代表原始 STL 三角面积严格转移已经通过。
- **线性拟合补充指标**：评估同时拟合 `prediction = a × truth + b`，输出 pooled 与病例等权共享拟合线的 `r2_linear_fit / linear_fit_slope / linear_fit_intercept`。该 R² 等价于 Pearson `r²`，只能与 `a、b` 及原始 R² 同报，不能替代绝对误差或用于改写既有 checkpoint/Gate。

## 目录地图

```text
training_wss_min/
├── config.py / dataset.py / models.py / metrics.py
├── objectives.py / runtime.py      # loss、选模与运行时公共逻辑
├── train.py / evaluate.py          # 核心训练/评估入口
├── configs/                        # 按主题分目录（见 configs/README.md）
├── tools/                          # 汇总、审计、模板、闸门、可视化、导出
├── experiments/                    # 一次性诊断（拟合链 / B-REP / density …）
├── cluster/                        # 现行 Slurm；历史 round5 脚本在 cluster/archive/
└── tests/ / runs/                  # 单测与实验产物（runs/<name>/；name 历史保留）
```

主题对照与 manifest：[`configs/README.md`](configs/README.md)。
实验结论与逐轮表：[`WSS最小化_训练实验跟踪.md`](../docs/02-推进与变更/WSS最小化_训练实验跟踪.md)。
第六轮总调度：[`WSS最小化_第六轮XYZ尺度诊断计划与执行.md`](../docs/02-推进与变更/WSS最小化_第六轮XYZ尺度诊断计划与执行.md)。

当前结构与信息筛选状态（2026-07-29）：REG-P10-LSA2 SAME/IND × MSE/H1/H2 六臂已全部完成。IND MSE 相对 SAME MSE 的 `ΔR²_cb=+0.0014`，未过 `+0.012` 主门；H1 在 SAME/IND 下的 `ΔIoU=+0.0175/-0.0018`，均 No-Go。H2 在 SAME/IND 内的 `Δhigh-WSS nRMSE=-0.00205/-0.00207` 均过门，但 IND-H2 相对 SAME-H2 的 R²、normalized R² 和 MAE更差，因此不晋级 IND。随后完成同 seed H2 精确并发 control 与唯一新增 train-only 标准化 `log(local_radius)` 输入列的两臂 Jobs `11032→11033_[0-1]`：处理臂 physical `R²_cb=0.3506`、MAE `2.5238 Pa`、high-WSS nRMSE `0.06611`，相对并发 control 分别为 `ΔR²_cb=+0.0436`、`ΔMAE=-0.0342 Pa`、`ΔnRMSE=-0.00295`，top10 IoU 仅下降 `0.0029`，全部预注册保护线通过。**LSA2 SAME-H2 + `log(local_radius)`** 现晋级为新的单 seed 开发锚点；暂不做多 seed，病例全场 R² 均值差 CI 跨零的边界必须保留。当前入口见 [`PointNet baseline 矩阵 §0Q`](../docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)。

2026-07-26 新增配置化的 SA 邻域内 Transformer：`model.local_transformer_stages` 用 1-based stage 列表控制，空列表保持历史结构；SA3 全局 Transformer 继续复用既有 `coarse_attention`。REG-P10 的 7 个局部非空 stage 子集与 1 个 SA3-global 对照已 `8/8` 完训、审计和回填；仅 `local_transformer_stages=[2]` 过单种子 Gate。后续决议不补其 seeds `7/2025`，改在该结构上完成 SAME/IND × H1/H2 矩阵。

fixed `106/0/27` 对照也已完成：以 `D2 c125×k64 + PointNeXt-R + LocalGeoPE` 为父模板，只开启 local-SA1 或 local-SA2。SA1 的物理 `ΔR²_cb=-0.0293`，No-Go；SA2 仅 `+0.0015` 且归一化/high-WSS 回退，判持平、不晋级。本组无 DropPath、无 ILO，只与同一 D2 父模型配对。

2026-07-28 完成 REG-P10 静态几何约束 EdgeConv 矩阵：`model.edgeconv_stages` 用 1-based stage 列表控制，默认空列表保持旧结构和 checkpoint key；消息仅在既有 SA 邻域内使用 `[x_i,x_j-x_i,Δp/r]`，未启用动态 KNN。SA1、SA2、SA12 三臂均未超过父模型；SA2 虽改善 MAE/IoU，但 high-WSS 与 ILO 保护线失败，三臂统一判 No-Go。

2026-07-28 完成 S3 RCR 信息上限 Oracle：O0 为 geometry-only，O1a 追加四出口真实 \(R_1/R_2/C\) 的 12 维 log 病例特征，O2 追加 train/test 内分域分层打乱的同维负对照。O1a 的 physical `R²_cb=0.3561`，相对 O0 `+0.0805`；O2 `0.2723`，相对 O0 `-0.0033`。真实 RCR 同时改善 MAE、high-WSS nRMSE、top10 IoU 和三域 R²；该信息上限证据保留，但当前不补多 seed、不做 RCR 架构。单种子、复用 test36 和 train-loss 选模仍限制其解释。

2026-07-28 O0 热点/高值首轮完成：新增 `loss_hotspot_bce_lambda/hotspot_quantile` 与 `loss_pinball_lambda/pinball_quantile`，默认均关闭。H1 用第二输出通道训练病例相对 top10 balanced BCE，但正式 WSS 评估只读取第一回归通道；H2 在原 MSE 上加入 q90 pinball。H1/H2 的 physical `ΔR²_cb=+0.0234/+0.0190` 且保护线通过，但各自主 Gate 未达，按预注册规则不组合、不扩 H2 q/λ。

## 用法

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 单实验
$PY -m training_wss_min.train    --config training_wss_min/configs/protocol_gates/b1_tgtw_fixedq_s1234.json
$PY -m training_wss_min.evaluate --config training_wss_min/configs/protocol_gates/b1_tgtw_fixedq_s1234.json

# PointNet 分布矩阵：第一阶段仅 GLOBAL；单作业自动 train→best/last eval→best test16 PostView
bash training_wss_min/cluster/submit_pointnet_distribution_stage.sh global

# v4 两项已完成配置（Jobs 9138/9140；这里只保留复现入口，不要重复提交）
/public/slurm/bin/sbatch training_wss_min/cluster/run_pointnet_distribution_matrix.slurm \
  training_wss_min/configs/pointnet_v4/ag_v4_e2_global_fps2000.json
/public/slurm/bin/sbatch training_wss_min/cluster/run_pointnet_distribution_matrix.slurm \
  training_wss_min/configs/pointnet_v4/ag_aaa_v4_e2_global_fps2000.json

# 已有矩阵：PointNet++ Jobs 9167–9169 与 PointNet E2 Job 9170 的模型资产已完成。
# 下列命令仍是只读复核入口；不要重复提交。
$PY -m training_wss_min.tools.preflight_v4_jobs --configs \
  training_wss_min/configs/pointnetpp_v4/ag_v4_sa3_e2_global_fps2000.json \
  training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_locked_sa3_e2_global_fps2000.json \
  training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_e2_global_fps2000.json \
  training_wss_min/configs/pointnet_v4/ag_aaa_v4_stratified_e2_global_fps2000.json \
  --output data_wss_min/pipeline_reports/v4_cutover_20260715_1921/v4_pointnetpp_matrix_preflight.json

# test27 两阶段矩阵：vertex/FPS 6组已由下列正式报告提交为 Jobs 10473–10478。
# submitter 会核对精确6份 config SHA/run目录；不要对同一 manifest 重复执行。
$PY -m training_wss_min.tools.preflight_v4_jobs --configs <6个Phase-V冻结config> \
  --output training_wss_min/preflight/ag_aaa_v4_vertex_phase_6_preflight.json
$PY training_wss_min/cluster/submit_v4_support_query_matrix.py \
  --preflight training_wss_min/preflight/ag_aaa_v4_vertex_phase_6_preflight.json \
  --manifest training_wss_min/preflight/ag_aaa_v4_vertex_phase_submission_manifest.json

# 面积6组仍冻结；修复6个失败病例后，必须按 backlog 重跑 both_strict 正式门禁。
training_wss_min/preflight/ag_aaa_v4_area_phase_backlog.json

# 仅用于已完训但被面积门禁拦住的 checkpoint 诊断：全27 vertex 指标，
# 面积指标只写通过映射门的子集；不会替代正常 evaluate 或放宽正式门禁。
$PY -m training_wss_min.tools.evaluate_mapping_blocked_run --help

# 导师追加 E4 deeper PointNet 复现入口；8999 已完训并判为 No-Go
bash training_wss_min/cluster/pointnet_deeper/submit.sh

# 进入 PointNet++ fine-tune 前的三层 SA 中心/分组可视化（不会训练模型）
$PY -m training_wss_min.tools.visualize_pointnetpp_sa

# 冻结的最小 2×3 baseline（MLP/PointNet/PointNet++ × xyz/xyz+geom）
bash training_wss_min/cluster/baseline_2x3/submit.sh

# 给导师查看的 PointNet 完整训练单文件（≤250 行，不导入项目内部模块）
$PY -m training_wss_min.examples.pointnet_baseline_train

# 已完成资产：横向 Track A（壁面 gauge pressure）
$PY -m training_wss_min.tools.make_pressure_stats
WSSMIN_MANIFEST=training_wss_min/configs/sweeps/pressure_wall.txt \
  bash training_wss_min/cluster/submit_baseline_sweep.sh
```

## 产物

每个实验落在 `runs/<name>/`（`name` 来自 JSON 内字段，与历史 run 目录一致）：

- `train.log` / `history.jsonl` / `history.png`
- `ckpt_best.pt` / `ckpt_last.pt` / `config.json` / `feature_stats.json`
- `eval/ckpt_best/`、`eval/ckpt_last/` 下的 `metrics.json`、`per_case_metrics.csv`、`linear_fit_metrics.csv`
- `postview/ckpt_best/test/`：每例对齐 STL、带 normalized 标量的 VTP、同点 CSV 与预览图

集群日志：`cluster/logs/wssmin_<name>_<jobid>.{out,err}`。

## 数据口径（继承 pipeline_wss_min）

- split：开发默认 `split_AG_wss_min_v2_dev1`；clean 主矩阵用 `split_AG_wss_min_v1`。
- v4 已验收协议：AG `61/0/15`；锁定 AG test15 的 AG+AAA `118/0/15`。
- v4 新独立协议：AG76 + AAA57 按 AG/AAA rupture/AAA unrupture 分层为 `106/0/27`；旧 AG test15 不锁定，因此新 test27 总指标不得与 common-test15 直接横比。
- 坐标：逐病例各向同性归一化到 `[-1,1]`。GLOBAL 保留 train-only `log_z` 并可恢复物理 WSS；CASE 直接预测 `WSS/WSSmax`，测试真值最大值不进入模型，默认不恢复物理 WSS。
- excluded/pending 不得进入训练 / stats / eval。

## 历史主题（一行索引）

| 主题目录 | 核心问题 | 状态 |
|---|---|---|
| `baseline_sweep/` | 点数/特征/采样正交扫 | 归档 |
| `baseline_2x3/` | MLP / PointNet / PointNet++ × xyz / xyz+geom | **冻结保留** |
| `loss_aug_ablation/` | 历史 loss/采样/增广；后续 L1/L2 WSS loss Gate | 历史已归档；**L1/L2 待按第六轮协议生成** |
| `clean_data/` | clean-data 主矩阵 | 归档 |
| `protocol_gates/` | 固定阈值与 Gate-1；锚点 B1 | 归档 |
| `pointcount_curve/` | 点数—精度曲线 | 归档 |
| `fit_lc_diagnosis/` | 拟合 / LC / B-REP 诊断 | 归档（科学结案） |
| `xyz_scale_diag/` | A/B/D XYZ 尺度诊断 | **当前主线承接：C/E 待执行** |
| `multitarget/` | 壁面压力与条件触发的多目标诊断 | H-PW 已完成；Track B 等 adapter/QA |
| `pointnetpp_d2_k64_ilo_structure_20260723/` | ILO 两协议与 PointNeXt/GeoPE/Attention/SEP | **9/9 完训；S3 三种子通过** |
| `pointnetpp_s3_regularization_20260724/` | S3 的 head dropout / DropPath / NeighborDrop 严格单变量三种子矩阵 | **首波 12/12 与强度/交叉补齐 20/20 均完成** |
| `pointnetpp_regp10_transformer_20260726/` | REG-P10 的 7 个 local-SA stage 子集与 SA3 global attention 对照 | **8/8 完训；仅 L-SA2 过单种子 Gate，待补 seeds 7/2025** |
| `pointnetpp_d2_c125_k64_pnxr_geope_transformer_20260726/` | fixed 106/0/27 D2 PNXR+GeoPE 上的 local-SA1 / local-SA2 对照 | **2/2 完训；SA1 No-Go，SA2 持平不晋级** |
| `pointnetpp_rcr_oracle_20260728/` | S3 geometry / true RCR / shuffled RCR 信息上限对照 | **3/3 完训；O1a `ΔR²_cb=+0.0805`，O2 `-0.0033`，Go-to-follow-up** |
| `pointnetpp_o0_hotspot_tail_20260728/` | O0 的 top10 hotspot BCE / q90 pinball 单变量目标 | **2/2 完训；保护线通过但主 Gate 均未达，No-Go、不组合** |

所有已执行 JSON 和 manifest 都作为复现资产保留；配置生成器与根目录兼容入口已退役。
正式命令只使用 `training_wss_min.train`、`training_wss_min.evaluate`、
`training_wss_min.tools.*` 或 `training_wss_min.experiments.*`。
