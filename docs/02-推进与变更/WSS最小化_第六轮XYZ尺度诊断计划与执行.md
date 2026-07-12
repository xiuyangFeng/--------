# WSS 最小化·第六轮 XYZ 尺度诊断计划与执行

> 状态：`RUNNING`（2026-07-12 已提交 9 个 val-only 作业）  
> 目的：在同一 clean-data/dev1/B1 协议下，判断旧“纯 XYZ”较差是否主要由逐病例归一化丢失物理尺度造成。  
> 边界：本轮是 A/B/D **尺度诊断**，不是保留物理尺度的严格 XYZ 终审，也不用 61 例外推数千病例的学习上限。

## 1. 问题与口径修正

旧纯 XYZ 路线使用逐病例 `[-1,1]` 坐标，保留相对形状但丢失病例间绝对尺度。WSS 可能对血管尺寸敏感，因此“旧纯 XYZ 低于 XYZ+几何”不能直接推导为“大数据下 XYZ 不可行”。

本轮先用现有 `coord_scale` 能力做最小单变量诊断：

| 组 | 输入特征 | 用途 |
|---|---|---|
| A | `x,y,z` | 当前逐病例归一化纯 XYZ 对照 |
| B | `x,y,z,coord_scale` | 只补回病例物理尺度，诊断尺度丢失 |
| D | `x,y,z,abscissa_norm,local_radius,curvature` | 显式几何控制组 |

B 不是老师“严格只有 XYZ”的最终验证；它是一个低成本的机制诊断。若需严格验证，后续 C 组必须将保留病例间物理尺度的 XYZ 同时用于输入特征、FPS 和 ball-query 邻域。

## 2. 冻结实验协议

- split：`split_AG_wss_min_v2_dev1.json`，train 53 / val 8。
- WSS stats：`wss_stats_v2_dev1.json`，train-only、peak-only。
- 目标：入口波形峰值收缩期单帧 WSS 标量；当前 bundle 为 step 1162 / index 21。
- 采样：fixed FPS-2000；评估为 val 完整壁面点云。
- 模型：PointNeXt-S，结构、SA radius 和参数均不变。
- 训练：B1 fixed target-weight 协议，160 epoch，复合选模，early stop。
- seed：`1234 / 7 / 2025`。
- 评估：仅 `val`；本轮不读 legacy test16。
- 唯一允许变量：`input_features` 与 model seed。

机器可读预注册：`training_wss_min/configs/round6/scale_diagnostic_protocol.json`。

## 3. 判读规则

主指标：

- `R²_field_raw`
- `R²_casemean`

护栏：`top10_pred_true_ratio` / `top10_iou` / 负 R² 病例数。

1. **尺度信号 B−A**：三 seed 均值的两项主 R² 均 `> +0.02`，才认为“逐病例归一化丢失尺度”有可辨识负面影响。
2. **显式几何增量 D−B**：报告两项主 R² 的三 seed 均值、sample std 和逐 seed 方向；不用单 seed 排名。
3. **绝对精度与相对非劣效分开**：B 接近 D 只能说明显式几何可能被坐标+尺度替代；是否“精度很好”仍需单独对照 `0.70` 工程目标。
4. **外推限制**：13–53 例学习曲线只证明当前 61 例范围内 40→53 无可辨识 field 增益，不能推导几百/几千例数据无用。

## 4. 配置与提交

- 生成器：`training_wss_min/make_configs_round6_scale.py`
- 配置：`training_wss_min/configs/round6/r6_scale_{A_xyz,B_xyzscale,D_xyzgeom}_s{1234,7,2025}.json`
- manifest：`training_wss_min/configs/sweep_round6_scale_abd.txt`
- 提交记录：`training_wss_min/cluster/logs/submitted_20260712_120420.txt`

| 组 | seed | Job | 当前状态 |
|---|---:|---:|---|
| A | 1234 | 7029 | RUNNING（提交后首个启动） |
| A | 7 | 7030 | QUEUED |
| A | 2025 | 7031 | QUEUED |
| B | 1234 | 7032 | QUEUED |
| B | 7 | 7033 | QUEUED |
| B | 2025 | 7034 | QUEUED |
| D | 1234 | 7035 | QUEUED |
| D | 7 | 7036 | QUEUED |
| D | 2025 | 7037 | QUEUED |

Slurm 参数默认 `EVAL_PARTS=val`，未传入 test；GPU 分区为 `GPU`，每个作业 1×4090。

## 5. 完成条件

- 9 个 run 均生成 `ckpt_best.pt`、`history.jsonl`、`eval/metrics.json` 和 `eval/per_case_metrics.csv`。
- 汇总 A/B/D 三 seed 均值±sample std、逐 seed 方向与护栏。
- 先回答尺度丢失是否有效，再决定是否实现严格物理尺度 XYZ（C 组）。
- 不根据本轮 val 结果访问 test16。
