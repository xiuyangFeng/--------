# training_wss_min — WSS-min baseline 训练/评估（PointNeXt 残差版）

与既有 `training/`（V3P 图/变换器栈）**完全独立**，只读 `pipeline_wss_min` 产出的
`data_wss_min/**/bundle.npz` 与 `wss_global_stats.json`。

## 任务

- **任务**：几何点云 `(x,y,z[+几何])` → 壁面 **WSS 标量**（`log_z` 全局归一化），单头。
- **模型**：`PointNeXt-S` 残差版（ball-query）；训练可稀疏采样，**评估恒在完整壁面点云**。
- **开发评估**：默认且通常只允许 `val`；`test` 需 `--allow-test`，不得用于选模。

## 目录地图

```text
training_wss_min/
├── config.py / dataset.py / pointnext.py / metrics.py
├── train.py / evaluate.py          # 核心训练栈
├── configs/                        # 按主题分目录（见 configs/README.md）
├── tools/                          # 配置生成、汇总、模板、闸门、可视化、导出
├── experiments/                    # 一次性诊断（拟合链 / B-REP / density …）
├── cluster/                        # 现行 Slurm；历史 round5 脚本在 cluster/archive/
├── tests/ / runs/                  # 单测与实验产物（runs/<name>/；name 历史保留）
└── *.py                            # 旧入口兼容 shim（Prefer tools./experiments.）
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

# 已完成资产：横向 Track A（壁面 gauge pressure）
$PY -m training_wss_min.tools.make_pressure_stats
$PY -m training_wss_min.tools.make_configs_pressure
WSSMIN_MANIFEST=training_wss_min/configs/sweeps/pressure_wall.txt \
  bash training_wss_min/cluster/submit_baseline_sweep.sh
```

## 产物

每个实验落在 `runs/<name>/`（`name` 来自 JSON 内字段，与历史 run 目录一致）：

- `train.log` / `history.jsonl` / `history.png`
- `ckpt_best.pt` / `ckpt_last.pt` / `config.json` / `feature_stats.json`
- `eval/metrics.json`、`eval/per_case_metrics.csv`

集群日志：`cluster/logs/wssmin_<name>_<jobid>.{out,err}`。

## 数据口径（继承 pipeline_wss_min）

- split：开发默认 `split_AG_wss_min_v2_dev1`；clean 主矩阵用 `split_AG_wss_min_v1`。
- 坐标：逐病例各向同性归一化到 `[-1,1]`；WSS 为 train-only `log_z`，评估在原始空间。
- excluded/pending 不得进入训练 / stats / eval。

## 历史主题（一行索引）

| 主题目录 | 核心问题 | 状态 |
|---|---|---|
| `baseline_sweep/` | 点数/特征/采样正交扫 | 归档 |
| `loss_aug_ablation/` | 历史 loss/采样/增广；后续 L1/L2 WSS loss Gate | 历史已归档；**L1/L2 待按第六轮协议生成** |
| `clean_data/` | clean-data 主矩阵 | 归档 |
| `protocol_gates/` | 固定阈值与 Gate-1；锚点 B1 | 归档 |
| `pointcount_curve/` | 点数—精度曲线 | 归档 |
| `fit_lc_diagnosis/` | 拟合 / LC / B-REP 诊断 | 归档（科学结案） |
| `xyz_scale_diag/` | A/B/D XYZ 尺度诊断 | **当前主线承接：C/E 待执行** |
| `multitarget/` | 壁面压力与条件触发的多目标诊断 | H-PW 已完成；Track B 等 adapter/QA |
