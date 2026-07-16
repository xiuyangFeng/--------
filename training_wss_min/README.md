# training_wss_min — WSS-min baseline 训练/评估（PointNeXt 残差版）

与既有 `training/`（V3P 图/变换器栈）**完全独立**，只读 `pipeline_wss_min` 产出的
`data_wss_min/**/bundle.npz` 与 `wss_global_stats.json`。

## 任务

- **任务**：几何点云 `(x,y,z[+几何])` → 壁面 **WSS 标量**；支持 train-only 全局 `log_z` 与逐病例 `WSS/WSSmax` 目标空间，单头。
- **模型**：`PointNeXt-S` 残差版（ball-query）；训练可稀疏采样，**评估恒在完整壁面点云**。
- **开发评估**：默认且通常只允许 `val`；`test` 需 `--allow-test`，不得用于选模。
- **v4**：配置必须显式给出 `data_root` 与 `required_frame_version=stl_landmarks_v4`；canonical AG/AAA ID 可混合加载，split 重复/泄漏直接拒绝。

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

## 用法

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 单实验
$PY -m training_wss_min.train    --config training_wss_min/configs/protocol_gates/b1_tgtw_fixedq_s1234.json
$PY -m training_wss_min.evaluate --config training_wss_min/configs/protocol_gates/b1_tgtw_fixedq_s1234.json

# PointNet 分布矩阵：第一阶段仅 GLOBAL；单作业自动 train→best/last eval→best test16 PostView
bash training_wss_min/cluster/submit_pointnet_distribution_stage.sh global

# v4 两项已提交配置（AG 61/15；AG61+AAA57 train / AG15 test）
/public/slurm/bin/sbatch training_wss_min/cluster/run_pointnet_distribution_matrix.slurm \
  training_wss_min/configs/pointnet_v4/ag_v4_e2_global_fps2000.json
/public/slurm/bin/sbatch training_wss_min/cluster/run_pointnet_distribution_matrix.slurm \
  training_wss_min/configs/pointnet_v4/ag_aaa_v4_e2_global_fps2000.json

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
- `eval/ckpt_best/`、`eval/ckpt_last/` 下的 `metrics.json`、`per_case_metrics.csv`
- `postview/ckpt_best/test/`：每例对齐 STL、带 normalized 标量的 VTP、同点 CSV 与预览图

集群日志：`cluster/logs/wssmin_<name>_<jobid>.{out,err}`。

## 数据口径（继承 pipeline_wss_min）

- split：开发默认 `split_AG_wss_min_v2_dev1`；clean 主矩阵用 `split_AG_wss_min_v1`。
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

所有已执行 JSON 和 manifest 都作为复现资产保留；配置生成器与根目录兼容入口已退役。
正式命令只使用 `training_wss_min.train`、`training_wss_min.evaluate`、
`training_wss_min.tools.*` 或 `training_wss_min.experiments.*`。
