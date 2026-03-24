# 任务A 实验冻结卡

> 本文件记录任务 A 正式实验阶段的不可变配置。  
> 一旦填写完毕并开始正式训练，以下字段**不得中途修改**。  
> 上位文档：[任务A实验清单](任务A实验清单.md) / [任务A配置与启动说明](任务A配置与启动说明.md)

---

## 1. 核心冻结字段

| 字段 | 当前值 | 状态 |
|---|---|---|
| `data_version` | `AG_v1` | ✅ 已确定 |
| `data_root` | `data_new/AG` | ✅ 已确定 |
| `case_name_format` | `{group}/{patient_id}`，如 `fast/HAN_JIAN_JUN` | ✅ 已确定 |
| `graphs_subdir` | `processed/graphs` | ✅ 已确定 |
| `split_version` | `split_AG_v1` | ✅ 已确定 |
| `split_file` | `training/splits/split_AG_v1.json` | ✅ 已生成并用于全部基线训练 |
| `split_protocol` | `single_split` | ✅ 已确定 |
| `preprocess_version` | pipeline 当前版本（采样混合FPS20%，kNN图） | ✅ 已确定 |
| `normalize_source` | 仅训练集统计量 | ✅ 已确定 |
| `seed_plan` | smoke test: seed=1；正式结果: [1, 2, 3] | ✅ 已确定 |
| `primary_metric` | `RMSE_|v|`（速度模长误差） | ✅ 已确定 |
| `result_root` | `outputs/field/` | ✅ 已确定 |

---

## 2. 数据范围说明

### 2.1 数据来源

| 组别 | 路径 | 有效病例数 | 状态 |
|---|---|---|---|
| fast | `data_new/AG/fast/` | 24 | ✅ 全部完成 |
| slow | `data_new/AG/slow/` | 43（已完成）+ 19（处理中） | ⏳ 等待集群作业 1177 完成 |

> 合并后预计总病例数：**67 ~ 86 个**（取决于 slow 剩余 19 个最终结果）

### 2.2 已知排除病例

| 病例 | 组别 | 原因 |
|---|---|---|
| ZHANG_XIU_ZHEN | fast | 无 log 无 pt 文件，数据缺失 |

---

## 3. 训练通用设置（已固定）

| 参数 | 值 |
|---|---|
| `optimizer` | Adam |
| `lr` | 0.0005 |
| `scheduler` | ReduceLROnPlateau，factor=0.5，patience=10 |
| `epochs` | 200 |
| `early_stopping_patience` | 30 |
| `batch_size` | 2 |
| `grad_clip_norm` | 1.0 |
| `target_weights` | [1.0, 1.0, 1.0, 1.0]（u/v/w/p 等权） |
| `deterministic` | true |

---

## 4. 特征定义（已固定，来自 pipeline/config.py）

### 节点特征 data.x（10维）

| 索引 | 特征名 | 说明 |
|---|---|---|
| [0:3] | x, y, z | 空间坐标 |
| [3] | Abscissa | 沿轴弧长 |
| [4] | NormRadius | 归一化半径 |
| [5] | Curvature | 曲率 |
| [6:9] | Tangent_X/Y/Z | 切向量 |
| [9] | is_wall | 壁面标签（0/1） |

### 图级条件 data.global_cond（6维）

| 索引 | 特征名 | 说明 |
|---|---|---|
| [0] | t_norm | 归一化时间 |
| [1] | BC_Inlet | 入口边界条件 |
| [2:6] | BC_O1~O4 | 出口边界条件 |

### 预测目标 data.y（4维）

| 索引 | 目标名 | 说明 |
|---|---|---|
| [0:3] | u, v, w | 速度分量 |
| [3] | p | 压力 |

---

## 5. 输出目录结构（已固定）

```
outputs/field/
└── {experiment_name}_seed{seed}/
    ├── config.snapshot.json    # 配置副本
    ├── split.snapshot.json     # split 副本
    ├── history.csv             # 训练曲线
    ├── summary.json            # 最优指标
    ├── best_model.pt           # 最优权重
    ├── last_model.pt           # 末轮权重
    ├── predictions.parquet     # 测试集预测结果
    ├── fig_loss.png            # 损失曲线图
    └── fig_scatter.png         # 散点图
```

---

## 6. 执行检查清单（数据就绪后按序执行）

- [x] slow 组全部处理完成（集群作业 1177 结束）
- [x] 排查 slow 剩余 19 个病例是否全部生成 pt 文件
- [x] 生成病例清单文件 `training/splits/cases_AG_v1.txt`
- [x] 执行 `python -m training.scripts.make_split` 生成 `split_AG_v1.json`
- [x] 核对 split 中 train/val/test 病例数比例（4860 / 648 / 1458 graphs）
- [x] 核对 `data.x`、`data.global_cond`、`data.y` 维度
- [x] 执行 `python -m training.scripts.make_field_plan --groups baseline` 生成配置
- [x] 跑通 A-Base-01 smoke test（1 epoch，seed=1）
- [x] 跑通 A-Main-01 smoke test（1 epoch，seed=1）
- [x] 确认冻结卡 split_version 字段已填写
- [x] **A-Base-01 / A-Base-02 / A-Base-03 / A-Main-01 全部 3 seed 训练完成**（2026-03-22/23）

---

## 7. 后续数据就绪后的操作命令（已准备好，等待执行）

### Step 1：生成完整病例清单

```bash
# 数据就绪后，在项目根目录执行
cd /public/newhome/cy/Digital_twin/GNN
python -c "
import os
from pathlib import Path

cases = []
for group in ['fast', 'slow']:
    group_dir = Path(f'data_new/AG/{group}')
    for case_dir in sorted(group_dir.iterdir()):
        pt_files = list((case_dir / 'processed' / 'graphs').glob('*.pt'))
        if len(pt_files) > 0:
            cases.append(f'{group}/{case_dir.name}')

with open('training/splits/cases_AG_v1.txt', 'w') as f:
    f.write('\n'.join(cases))

print(f'共 {len(cases)} 个有效病例')
for c in cases:
    print(c)
"
```

### Step 2：生成患者级 split 文件

```bash
python -m training.scripts.make_split \
  --cases-file training/splits/cases_AG_v1.txt \
  --output training/splits/split_AG_v1.json \
  --split-version split_AG_v1 \
  --source AG \
  --seed 1 \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2
```

### Step 3：生成实验配置

```bash
python -m training.scripts.make_field_plan \
  --data-root data_new/AG \
  --split-file training/splits/split_AG_v1.json \
  --groups baseline \
  --output-dir training/configs/field/generated
```

### Step 4：Dry-run 确认配置无误

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --study-group baseline \
  --dry-run
```

---

## 8. 当前状态

**2026-03-16**：等待 slow 组集群作业 1177 完成（19 个病例处理中/排队中）。  
冻结卡字段除 `split_version` 和 `split_file` 外均已确定，数据就绪后立即执行第 6 节检查清单。

**2026-03-23（更新）**：所有冻结字段已全部确定，第 6 节执行清单已全部完成。  
第一批基线实验（A-Base-01 ~ A-Main-01）全部 3 seed 训练完成，结果已归档至 `outputs/field/`，实验状态详见 [任务A实验状态表](任务A实验状态表.md)。  
当前数据集规模：训练 4860 graphs / 验证 648 graphs / 测试 1458 graphs。  
测试集预测与后处理图已补齐，当前已生成：

- `fig_A3_scatter.png`
- `fig_A4_per_case_boxplot.png`
- `fig_A5_regional_bar_rmse_vel_mag.png`
- `fig_A5_regional_bar_rmse_p.png`
- `fig_error_distribution.png`
- `fig_error_cdf.png`

其中，`A-Main-01` 已具备 `wall / interior / high_curvature / low_curvature / near_wall / core_flow / bifurcation / trunk` 全套区域标签；`A-Base-02`、`A-Base-03` 当前稳定具备 `all / wall / interior`，`A-Base-01` 当前仅稳定具备 `all / interior`。这意味着现阶段可以写“总体 / 壁面 / 内部点”结论，但高曲率、近壁和分叉区域的横向对比仍需统一重导出预测资产后再做。  
效率 benchmark 已补齐，当前 `outputs/field/plots/` 下已新增：

- `fig_A7_efficiency_benchmark.json`
- `fig_A7_efficiency_bars.png`
- `fig_A7_pareto_rmse_vel_mag_vs_latency.png`

当前 benchmark 口径已升级为：4 个 baseline 的 **seed 1/2/3 共 12 个 run**，测试病例 `slow/GUO_XI_JIANG`（81 snapshots），`n_warmup=5`，`n_runs=20`。当前效率图既包含 `aggregated` 的 `mean±std`，也包含 `rows_per_seed` 的分 seed 结果。结果显示：

- `A-Base-01` 最快：`0.54 ± 0.27 ms / snapshot`，`127.34 ± 0.00 MB`
- `A-Base-02` 提供较好的折中：`2.35 ± 0.23 ms / snapshot`，`529.69 ± 0.47 MB`
- `A-Base-03` 与 `A-Main-01` 时延和显存几乎相同：`6.95 ± 0.09` vs `6.88 ± 0.02 ms / snapshot`，显存约 `2.18 GB`
- `A-Main-01` 在几乎不增加部署开销的前提下，相比 `A-Base-03` 显著降低了 `RMSE_|v|`

新增的效率图现已不止主柱图和主 Pareto 图，还包括：

- `fig_A7_efficiency_bars_mean_std.png`
- `fig_A7_latency_per_seed.png`
- `fig_A7_peak_memory_per_seed.png`
- `fig_A7_fullcase_peak_memory_per_seed.png`
- `fig_A7_pareto_per_seed_points.png`
- `fig_A7_pareto_rmse_vel_mag_vs_latency_mean_std.png`

下一步：先统一区域标签评估口径，再启动 A-Abl-01（输入特征消融）；效率证据层面当前已经具备 3-seed 版本，后续若还要继续加固，可再补多病例 benchmark 或给出 CFD 时间基线以填写 `speedup_vs_CFD`。
