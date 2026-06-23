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

评估含论文 MAE/RMSE/MSE、**NMAE**（按评估 split 全局 target range 归一化）与 **点级 R²**（`u/v/w/p_r2`、`vel_mag_r2`）。

## 当前状态（2026-06-20）

| 链路 | 状态 | 口径 |
| --- | --- | --- |
| raw_ascii v1 export | ✅ 完成 | Array 5630 + 补跑 5734_80 + merge 5738；`private_preprocessed_raw_ascii_v1/` |
| `crown_original_vp` 非 PINN | ❌ Job **5751** evaluate 后 **No-Go** | best_ep=55 · `p_r2=-5.28` · `nmae=0.077`；5739 保留 OOM 截断记录 |
| `crown_original_vp_pinn` | ❌ 5740 evaluate 后 No-Go | 24h TIMEOUT checkpoint：PINN 未改善压力且速度 R2 退化；不扩 seed |
| 指标口径 | ✅ 已补齐 | `metrics_{split}.json` 同时报告 NMAE 与点级 R² |

5751 已完成充分训练 + NMAE evaluate；非 PINN raw_ascii 路线 **No-Go**（压力不可学）。5739 仍作 OOM 工程记录。

## 数据加载与评估（2026-06-19 · 已审核）

raw_ascii v1 merged pkl（train **~100GB**）全量驻内存会导致训练 OOM、evaluate 极慢。默认已切换 **lazy partial 加载**：

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `lazy_load` | `true` | 读 `pkl/partial/` + `preprocess_cases.jsonl` 索引 |
| `partial_cache_cases` | `2` | 同时缓存的病例 shard 数 |
| `eval_batch_size` | 8（非 PINN） | evaluate 批量 GPU forward |

```bash
# 默认 lazy（仅加载 test split）
python -m external_baselines.crown_beihang.evaluate \
  --checkpoint outputs/.../best_model.pt --split test

# 强制 merged pkl（回归对照）
python -m external_baselines.crown_beihang.evaluate \
  --checkpoint outputs/.../best_model.pt --split test --eager-load
```

详见 [`docs/数据加载与评估加速说明.md`](docs/数据加载与评估加速说明.md)（含审核清单）。
