# CROWN/Beihang external baseline experiments

| 实验族 | 说明 | 当前 |
| --- | --- | --- |
| `crown_original_vp` | paper-original 非 PINN · `x,y,z` → `u,v,w,p` | Job **5751/5774** · **No-Go** · [分析记录](crown_original_vp/实验分析记录.md) · [合并汇报](CROWN_非PINN与PINN复现汇报_合并.md) |
| `crown_original_vp_pinn` | paper-original PINN 对照 | Job **5757/5806** · **No-Go（较非 PINN 改善）** · [分析记录](crown_original_vp_pinn/实验分析记录.md) · [合并汇报](CROWN_非PINN与PINN复现汇报_合并.md) |

> 数据：`private_preprocessed_raw_ascii_v1`（raw_ascii 体素化 · merge 5738 完成）  
> 5625/5626 旧链路 **作废** · lazy 加载优化 **已审核通过**（[`docs/数据加载与评估加速说明.md`](../docs/数据加载与评估加速说明.md)）
