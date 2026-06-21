# CROWN raw_ascii 预处理产物（v1）

> 论文口径：`ascii_in` 全量体点 → 0.05 mm 体素化 → pkl；训练阶段再随机 10000 点。  
> 旧目录 [`../private_preprocessed/`](../private_preprocessed/)（Job 5625 features 15000 点）**保留只读对照，勿覆盖**。

## 目录约定

```text
private_preprocessed_raw_ascii_v1/
├── pkl/
│   ├── partial/          # Array 导出按病例 shard
│   ├── crown_volume_*.pkl
│   └── crown_all_*.pkl
├── stats/train_stats.json
├── manifests/export_manifest.json
└── audit/
    ├── preprocess_cases.jsonl
    ├── pilot_timing.json
    ├── merge_timing.json
    ├── preprocess_audit.json
    ├── jsonl/            # 每病例 per-frame 行
    └── logs/             # 每病例 + merge.log
```

## 验收

- `preprocess_cases.jsonl` 中 `n_raw` 应为 **10⁵~10⁶** 量级（非 15000）· 当前 **13122** 行（6561 帧 × volume/all）
- `export_source=raw_ascii` · `failure_count=0`（见 `preprocess_audit.json`）
- merge 产物（5738）：`crown_volume_train.pkl` **~100GB** / 4617 样本；日志 `audit/logs/merge.log` · `merge_timing.json`
- 第一轮训练 `point_filter=volume`；**2026-06-19 起训练/evaluate 默认 lazy 读 `pkl/partial/`**（merged pkl 保留作 audit，勿删）
- Job **5739**（非 PINN）OOM 作废 run 见 `crown_beihang/experiments/crown_original_vp/实验分析记录.md`

## 训练读取方式

| 模式 | 入口 | 内存 |
| --- | --- | --- |
| **lazy（默认）** | `data.lazy_load=true` · jsonl 索引 + partial | ~2GB 索引 + 2 病例 cache |
| eager | `lazy_load=false` 或 evaluate `--eager-load` | train **~100GB**（易 OOM） |

说明：[`crown_beihang/docs/数据加载与评估加速说明.md`](../../crown_beihang/docs/数据加载与评估加速说明.md)

## 显式几何特征

第一轮 **不** 在本目录预挂几何列；后续 `crown_geom_vp` 在训练循环内对采样的 10000 点做中心线查询。
