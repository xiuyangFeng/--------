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
    ├── preprocess_audit.json
    ├── jsonl/            # 每病例 per-frame 行
    └── logs/             # 每病例处理日志
```

## 验收

- `preprocess_cases.jsonl` 中 `n_raw` 应为 **10⁵~10⁶** 量级（非 15000）
- `export_source=raw_ascii`
- 第一轮训练 `point_filter=volume`，读 `crown_volume_{train,val,test}.pkl`

## 显式几何特征

第一轮 **不** 在本目录预挂几何列；后续 `crown_geom_vp` 在训练循环内对采样的 10000 点做中心线查询。
