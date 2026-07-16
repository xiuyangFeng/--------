# WSS-min 代码归档

这里保存已经完成历史使命、但仍有审计价值的代码。归档模块不属于当前正式入口，
不会提供旧模块名兼容层，也不应被集群脚本或训练流程依赖。

`alignment_v3/` 保存 2026-07-07 至 2026-07-08 用于 AG v2/v3 flow-divider、
左右轴、主干居中和 STL 重叠人工检查的脚本。对应结论与图件已写入 WSS-min 推进记录
和 `outputs/wss_min/` 历史目录；归档入口会强制使用 `legacy_centerline`，不会随当前默认
配置漂移到 v4。当前正式坐标口径为原始 STL landmark v4。

如需复核历史实现，使用归档后的完整模块路径，例如：

```bash
python -m pipeline_wss_min.archive.alignment_v3.visualize_alignment --help
```

这些脚本保留的是历史诊断能力，不代表当前 v4 数据生产口径。
