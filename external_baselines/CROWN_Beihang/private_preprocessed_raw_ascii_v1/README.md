# CROWN raw_ascii 预处理产物（v1）

> 论文口径：`ascii_in` 全量体点 → 0.05 mm 体素化 → pkl；训练阶段再随机 10000 点。  
> **2026-09-18 腾盘**：`pkl/`（约 565GB，含 merge 与 partial）已删除。源场仍在 `data_new/AG/<case>/ascii_in/`（及 `ascii/` 壁面），**不要删源库**。需要时按下面命令重导。  
> 旧 Job 5625 `private_preprocessed/`（features 15000 点）空壳已撤，不再占一层目录。

## 目录约定

```text
private_preprocessed_raw_ascii_v1/
├── stats/train_stats.json          # 保留，重导协议
├── manifests/export_manifest.json  # 保留
├── audit/                          # 保留历史验收日志（~209MB）
└── pkl/                            # 2026-09-18 已删，重导后会再生成
```

## 验收

- `preprocess_cases.jsonl` 中 `n_raw` 应为 **10⁵~10⁶** 量级（非 15000）· 当前 **13122** 行（6561 帧 × volume/all）
- `export_source=raw_ascii` · `failure_count=0`（见 `preprocess_audit.json`）
- merge 产物（5738）：`crown_volume_train.pkl` **~100GB** / 4617 样本；日志 `audit/logs/merge.log` · `merge_timing.json`
- 第一轮训练 `point_filter=volume`；lazy 读 `pkl/partial/`。2026-09-18 起 pkl 已删，重训前必须先重导。
- Job **5739**（非 PINN）OOM 作废 run 见 `crown_beihang/experiments/crown_original_vp/实验分析记录.md`

## 重导（源库 `data_new` 不动）

```bash
# 试点
bash external_baselines/crown_beihang/cluster/submit_export_crown_pilot.sh

# 全量 Array：读 data_new/AG ascii_in，写回本目录 pkl/
bash external_baselines/crown_beihang/cluster/submit_export_crown_array.sh \
  external_baselines/crown_beihang/configs/local/crown_export_split_AG_v1.json
```

配置真源：`export.source=raw_ascii` · `voxel_size_mm=0.05` · `point_filters=["volume","all"]`。
代码入口：`python -m external_baselines.crown_beihang.export_pkl`。

## 训练读取方式

| 模式 | 入口 | 内存 |
| --- | --- | --- |
| **lazy（默认）** | `data.lazy_load=true` · jsonl 索引 + partial | ~2GB 索引 + 2 病例 cache |
| eager | `lazy_load=false` 或 evaluate `--eager-load` | train **~100GB**（易 OOM） |

说明：[`crown_beihang/docs/数据加载与评估加速说明.md`](../../crown_beihang/docs/数据加载与评估加速说明.md)

## 显式几何特征

第一轮 **不** 在本目录预挂几何列；后续 `crown_geom_vp` 在训练循环内对采样的 10000 点做中心线查询。
