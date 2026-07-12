# training_wss_min — WSS-min baseline 训练/评估（PointNeXt 残差版）

与既有 `training/`（V3P 图/变换器栈）**完全独立**，不共享任何代码，只读
`pipeline_wss_min` 产出的 `data_wss_min/**/bundle.npz` 与 `wss_global_stats.json`。

## 任务与设计（第一版 baseline）
- **任务**：几何点云 `(x,y,z[+几何])` → 壁面 **WSS 标量**（`log_z` 全局归一化），单头。
- **模型**：`PointNeXt-S` 残差版（`InvResMLP` 残差块 + **ball-query** 分组）。
  ball-query 半径固定、物理尺度不随点密度变，为**训练用稀疏点、推理整条血管**提供结构基础；两种密度的输出仍可能不同，第四轮以 A5 显式诊断而非假定完全等价。
- **A 路**（部署有完整几何、缺的是 CFD 标签）：训练用稀疏子采样（可扫点数），
  **评估恒在完整壁面点云上**。稀疏是训练/科研选择，不是部署约束。
- **矢量二期预留**：`out_dim=3` 即切矢量；建议届时走"幅值(复用标量) + 内在系方向"。

## 文件
| 文件 | 作用 |
|---|---|
| `config.py` | `ExpConfig` 数据类 + JSON 读写；一个实验=一个 JSON |
| `dataset.py` | 直读 bundle；采样(fps/random/geom_weighted)、特征标准化、完整点云评估接口 |
| `pointnext.py` | PointNeXt-S 残差主干（SA + InvResMLP + FP 解码） |
| `metrics.py` | pooled / case-mean / case-balanced R²、NRMSE/MAE + 分区(分叉/狭窄/高WSS) |
| `train.py` | 训练循环：AdamW+cosine、AMP、日志、best/last ckpt、曲线 |
| `evaluate.py` | 加载 best，**完整点云**推理 → 指标 + 逐病例 CSV + 误差热力图 |
| `make_configs.py` | 生成第一版 sweep 配置 |
| `make_configs_round3_clean.py` | 生成第三轮 clean-data 主矩阵（mse/tgtw 各 3 seed） |
| `make_configs_round4.py` | 第四轮 B/C 配置（v2_dev1 + 固定阈值/选模协议） |
| `make_configs_round4_pointcount.py` | §14 点数曲线新增配置（1000/1500/3000/6000） |
| `summarize_round4_pointcount.py` | 合并 6 点曲线 → `_summary_round4_pointcount/` |
| `make_configs_round6_scale.py` | 第六轮 A/B/D XYZ 尺度诊断（3 组×3 seed） |
| `make_v2_dev_splits.py` | 生成 3× repeated-holdout 开发划分与 fold stats |
| `gate1_compare.py` | Gate-1 相对 control 判读 |
| `a0_micro_overfit.py` | 第五轮四病例 train-only micro-overfit（Slurm-only 训练） |
| `a1_density_surface.py` | 第五轮 density/cap/STL mapping/真值回插 oracle 审计 |
| `brep_mlp.py` / `point_mlp.py` | B-REP/M1 逐点 MLP 串行 micro/dev1 对照 |
| `brep_c1_radius_norm.py` / `pointnext_radius_norm.py` | B-REP/C1 相对位置 radius 归一化 |
| `brep_c2_global_context.py` / `pointnext_global_context.py` | B-REP/C2 病例内 global mean/max 单输出模型 |
| `a0d_simple_overfit.py` / `a0d_target_weight_only.py` | A0D 按 optimizer-step 的单病例/四病例拟合链与 target-weight 单变量恢复 |
| `tests/test_round4_protocol.py` | sampler / hotspot / 固定权重单测 |
| `template_baseline.py` | mean / voxel / KNN-log-template 模板基线 |
| `cluster/` | GPU Slurm 脚本与提交驱动（开发默认 `EVAL_PARTS=val`） |

## 用法
```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 1) 生成 sweep 配置（写入 configs/，清单 sweep_baseline_v1.txt）
$PY -m training_wss_min.make_configs

# 2) 提交全部作业到 GPU 队列（4×4090，自动并行 4 个）
bash training_wss_min/cluster/submit_baseline_sweep.sh

# 单个 config 本地/单卡跑
$PY -m training_wss_min.train    --config training_wss_min/configs/pc_xyz_fps_w2000_peak.json
$PY -m training_wss_min.evaluate --config training_wss_min/configs/pc_xyz_fps_w2000_peak.json
```

开发评估默认且通常只允许 `val`。`test` 被显式守卫，仅在完成 OOF 且获批执行
legacy test16 一次弱确认时使用 `--partitions test --allow-test`；不得用 test 结果返回选模。

第三轮 clean-data 入口（前置：`pipeline_wss_min` 已重跑 included bundle，`qa-gate` 通过，`wss_global_stats.json` 为 train peak-only clean stats）：

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 1) 模板基线（不占 GPU）
$PY -m training_wss_min.template_baseline --methods mean,voxel,knn --partitions val,test

# 2) 生成 clean 主矩阵：mse / target-weight loss 各 3 seed
$PY -m training_wss_min.make_configs_round3_clean

# 3) 提交 GPU 训练 + 完整点云评估；提交成功记录 job id 后即可等待后续结果分析
WSSMIN_MANIFEST=training_wss_min/configs/sweep_round3_clean_v1.txt \
  bash training_wss_min/cluster/submit_baseline_sweep.sh
```

## 第一版 sweep（3 条正交问题，均为标量 WSS + 峰值收缩期）
- **点数-精度曲线**：`pc_xyz_fps_w{1000,1500,2000,3000,6000}_peak`（训练点数，评估恒全场）。
- **特征消融**：`w2000`(xyz) vs `feat_xyzgeom_*` vs `feat_geomonly_*`。
  纯几何(旋转不变)一组能量出"配准坐标框架值多少分"。
- **采样策略**：`w2000`(fps) vs `samp_random_*` vs `samp_geomw_*`（狭窄+高曲率更密）。
共 9 个 config；`pc_xyz_fps_w2000_peak` 是三条曲线共用锚点。

## 日志与产物（便于复查）
每个实验落在 `runs/<name>/`：
- `train.log` 全程日志（控制台同步）；`history.jsonl` 每 epoch 一行(loss/lr/val 指标)；
- `ckpt_best.pt`(按 `val_r2_casemean` 选) / `ckpt_last.pt`；`history.png` 训练曲线；
- `config.json` / `feature_stats.json` 完整配置与特征标准化统计；
- `eval/metrics.json`、`eval/per_case_metrics.csv`、`eval/heatmaps/<case>.png`（test 逐病例真值/预测/误差三联图）。

集群作业日志在 `cluster/logs/wssmin_<name>_<jobid>.{out,err}`；
每次提交的作业号清单在 `cluster/logs/submitted_<ts>.txt`。

## 检查进度
```bash
/public/slurm/bin/squeue -u cy                       # 队列状态
tail -f training_wss_min/cluster/logs/wssmin_*.out   # 某作业实时日志
grep val_r2 training_wss_min/runs/*/history.jsonl     # 各实验 val 曲线
column -s, -t training_wss_min/runs/*/eval/per_case_metrics.csv | less  # 逐病例指标
```

## 数据口径（继承 pipeline_wss_min）
- split：`training/splits/split_AG_wss_min_v1.json`（第三轮 clean-data 为 train 53 / val 8 / test 16 / excluded 9 / pending 1）。
- 坐标：逐病例各向同性归一化到 [-1,1]（flow-divider 原点、刚性配准）。
- WSS：第三轮 clean-data 使用全局 `log_z`（train-only、peak-only 统计）。评估在**原始 WSS 空间**算指标。
- excluded/pending 即使磁盘上有历史 bundle，也不得进入训练、stats 或 eval；训练入口只从 split 的 train/val/test 读取病例。
- 每个 run 会保存 `wss_global_stats.json` 快照，后续复评优先使用 run 内 stats，避免全局 stats 重算后混口径。

## 第三轮 clean-data 主矩阵
- 模板基线：`template_mean_clean`、`template_voxel_clean`、`template_knn_clean`，输出与深度模型一致的 `eval/metrics.json`。
- 深度模型：`xyz + abscissa_norm + local_radius + curvature`，不含壁面常数列 `dist_to_wall`，不启用 `rot_aug`。
- 配置：`r3_clean_xyzgeom_mse_s{1234,7,2025}` 与 `r3_clean_xyzgeom_tgtw_s{1234,7,2025}`。
- 训练：240 epoch，eval_every=10，best 仍按 `val_r2_casemean` 以便和旧轮次可比，同时记录 top10/p95/p99/max 校准指标。

### 第三轮结果（2026-07-10）

- 6 个 run（Slurm 6953–6958）均已完训并完成完整点云评估。
- test 三 seed：MSE `R²_field=0.191±0.018`、`R²_casemean=0.176±0.021`；target-weight α=2 为 `0.225±0.034`、`0.212±0.027`。
- target-weight 的 high-WSS `R²=-1.531±0.167`、top10 预测/真值比 `0.357±0.037`，仍存在明显峰值低估；seed 7 未稳定优于 MSE。
- A5 只读诊断（三 seed，终审已独立复现）：8 个 val 病例、同一 FPS-2000 点上，子采样/全量推理输出的标准化 RMSE 为 `0.164–0.228`、Pearson `0.938–0.972`，三 seed 均为全量推理系统性偏高 `0.03–0.07σ`；脚本与逐病例 CSV 见 `docs/02-推进与变更/assets_第四轮/a5_density_probe.{py,csv}`。这表明密度敏感，尚不构成替换全量评估协议的依据。
- 第三轮 clean-data 组合主要改善了幅值校准，没有显著抬高第二轮 target-weight 的整体 R² 均值；跨轮还同时改变 split、curvature transform 和训练时长，不能视为 stats 单变量消融。第四轮先修 worker-safe 重采样、train-global 固定 target-weight 阈值、val-only 开发评估和 fold-specific stats，再按闸门验证 raw-space 辅助 loss 与内在局部几何。
- `runs/_summary/` 已刷新为 27 个实验的聚合结果（含第三轮深度模型和 clean 模板基线）。
- 完整表与逐病例失败清单见 [`WSS最小化_训练实验跟踪.md`](../docs/02-推进与变更/WSS最小化_训练实验跟踪.md)；第四轮已完成阶段的执行顺序和 Go/No-Go 见 [`WSS最小化_第四轮执行总结与归档`](../docs/02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。

## 第四轮（2026-07-10，Stage A→B→C 单 seed）

- 配置：`python -m training_wss_min.make_configs_round4`；开发划分 `split_AG_wss_min_v2_dev{1,2,3}`。
- 提交示例：`WSSMIN_MANIFEST=training_wss_min/configs/sweep_round4_b0_s1234.txt bash training_wss_min/cluster/submit_baseline_sweep.sh`（默认只评 val）。
- 单 seed 结论：B1 固定阈值与 B0 持平（作协议锚点）；B2/B3/C1/C4 均 Gate-1 No-Go；C2/C3/C5 条件跳过。当前锚点 `r4_dev1_b1_tgtw_fixedq_s1234`。

### §14 点数—精度曲线补充（xyz+geom，dev1）

- 单 seed 6 点曲线：`runs/_summary_round4_pointcount/`；峰值曾在 1000，曲线非单调。
- **多 seed（1000 vs 2000 × 3）**：R²_field 持平（0.337±0.017 vs 0.339±0.020）；仅 R²_casemean 稳定偏向 1000（0.221±0.019 vs 0.188±0.025）。**默认锚点仍为 2000**；1000 为更稀采样候选。
- 详见计划 §14.5 / 训练跟踪文首。

## 第五轮结案（2026-07-12）

- P0、A0R/A0D/A0E、A1、B-REP、LC、BC 资产审计和 CFD 可信性审计均完成；科学问题已结案，工程 `R²=0.70` 未达标，未访问 legacy test16。
- A0D 证明四个已见病例可拟合到 field/casemean `0.994/0.992`；当前困难不是基础拟合能力。
- LC（3 链×3 seed）在 13–53 例范围内为 field `0.258→0.297→0.312→0.310`；40→53 无可辨识增益，但该结论不外推几百/几千个高质量独立病例的上限。
- 完整证据见 `runs/_round5/`与 [`WSS最小化_训练实验跟踪.md`](../docs/02-推进与变更/WSS最小化_训练实验跟踪.md)。

## 第六轮 A/B/D XYZ 尺度诊断（2026-07-12，运行中）

目的是判断逐病例坐标归一化丢失物理尺度，是否造成旧纯 XYZ 不公平：

- A：`x,y,z`。
- B：`x,y,z,coord_scale`。
- D：`x,y,z,abscissa_norm,local_radius,curvature`。
- 共同协议：dev1 / fixed FPS-2000 / B1 fixed target-weight / val-only / `seed={1234,7,2025}`。
- Jobs `7029–7037`；提交记录 `cluster/logs/submitted_20260712_120420.txt`。

生成和提交：

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
$PY -m training_wss_min.make_configs_round6_scale
WSSMIN_MANIFEST=training_wss_min/configs/sweep_round6_scale_abd.txt \
  bash training_wss_min/cluster/submit_baseline_sweep.sh
```

B 是尺度机制诊断，不是保留物理尺度的严格纯 XYZ 终审。详细预注册、判读规则和 Job 表见 [`第六轮 XYZ 尺度诊断计划与执行`](../docs/02-推进与变更/WSS最小化_第六轮XYZ尺度诊断计划与执行.md)。
