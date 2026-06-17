# CROWN/Beihang PointNet-PINN Reproduction

> 源码快照：`external_baselines/CROWN_Beihang/source_snapshot/`  
> 预处理产物（论文 raw_ascii v1）：`external_baselines/CROWN_Beihang/private_preprocessed_raw_ascii_v1/`  
> 旧链路（无效，仅对照）：`external_baselines/CROWN_Beihang/private_preprocessed/`

## 与论文源码对齐的口径

| 环节 | 论文 `model_train_*.py` | 本实现 |
| --- | --- | --- |
| 点云来源 | Fluent volume 体点 + 体素化 | **`ascii_in` 全量体点**（不经 pipeline FPS 15000 降采样） |
| 训练采样 | 体素化后每 epoch **随机 10000** | `train.py` 训练循环内 `randperm`（非 Dataset 内采样） |
| PINN 壁面 | `\|u\|^2+\|v\|^2+\|w\|^2 \le 0.01` | `physics.wall_velocity_threshold`（默认 0.01），**不用 `is_wall`** |
| PINN 坐标 | 训练时 `×1000` | `physics.coord_scale: 1000` |
| PINN data/phy | **两套随机 idx** | `separate_phy=True` |
| 压力 | train 全局 min-max | `stats/train_stats.json` |
| 显式几何 | 论文默认无 | **第一轮 mask 掉**；后续 `crown_geom_vp` 在 **10000 点采样后** 挂中心线几何 |

`point_filter`：

- `volume`：仅 `ascii_in`（论文 volume）
- `all`：`ascii_in` + `ascii` 壁面

`export.source: features` 仅保留给几何消融，**不是**论文默认路径。

## 预处理导出

**单病例估时（先于全量）**

```bash
bash external_baselines/crown_beihang/cluster/submit_export_crown_pilot.sh
```

**全量 Array（全 CPU 节点并行）**

```bash
bash external_baselines/crown_beihang/cluster/submit_export_crown_array.sh \
  external_baselines/crown_beihang/configs/local/crown_export_split_AG_v1.json
```

日志：`private_preprocessed_raw_ascii_v1/audit/preprocess_cases.jsonl` · `pilot_timing.json`

## 训练

```bash
python -m external_baselines.crown_beihang.train \
  --config external_baselines/crown_beihang/configs/local/crown_original_vp_split_AG_v1_seed1.json
```

第一轮矩阵：`crown_original_vp`（非 PINN）· `crown_original_vp_pinn`（PINN）· 默认 `point_filter=volume`。

评估含论文 MAE/RMSE/MSE 与 **点级 R²**（`u/v/w/p_r2`、`vel_mag_r2`）。
