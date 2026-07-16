# training_wss_min/configs — 主题目录对照

配置按**实验主题**分目录，不再使用 `round5` / `round6` 等轮次目录名。

**兼容约定**：JSON 文件名可语义化；文件内 `name` 字段与历史 `runs/<name>/` **保持原样**（含 `r2_`/`r3_`/`r4_`/`r5_`/`r6_`），便于复评与追溯。新实验生成器应直接写语义 `name`。

| 目录 | 主题（核心问题） | 历史对照 | 状态 |
|---|---|---|---|
| `baseline_sweep/` | 点数 / 特征 / 采样正交扫 | 第 1 版 baseline | 归档 |
| `baseline_2x3/` | MLP / PointNet / PointNet++ × xyz / xyz+geom | 2026-07-14 最小矩阵 | **冻结保留** |
| `pointnet_trainloss_e400/` | PointNet × xyz / xyz+geom；train/test-only；按 train_loss 选模；400 epoch | 2026-07-14 | **完训+完评 `8968` / `8974–8975`** |
| `pointnet_wide128/` | PointNet+xyz+geom 容量探针：`width=128`/`head_hidden=256`（相对老师多一层 128 过渡）；FPS-2000；val-only | 2026-07-14 | **完训 `8970`｜补充 No-Go** |
| `pointnet_distribution_matrix/` | E2 精确导师通道 / E3 random-5000 × GLOBAL/CASE；另含 E23 宽网+5000 点交互对照 | 2026-07-14/15 | **五组完训+完评 `8976–8980`；E2 最强，CASE No-Go** |
| `pointnet_v4/` | AG-v4 61/15 与 AG61+AAA57 混合训练池的 E2-GLOBAL/FPS-2000 公平复跑 | 2026-07-16 | **Jobs `9138` / `9140` 完训完评；AG-v4 No-Go，AAA 有小幅整体增益但高尾退化** |
| `pointnet_deeper/` | E4 导师追加深度探针：`6→64→128→256→512；1024→512→256→128→64→1`，其余严格对齐 E2 | 2026-07-15 | **Job `8999` 完训+完评｜test 退化｜No-Go** |
| `pointnetpp_sa_foundation/` | PointNet++ 三层 SA 结构审计：FPS-2000，中心 `500→125→32` | 2026-07-15 | **仅可视化 foundation；未批准训练** |
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
