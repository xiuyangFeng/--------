# CROWN/Beihang external baseline experiments

| 实验族 | 说明 | 当前 |
| --- | --- | --- |
| `crown_original_vp` | paper-original 非 PINN · `x,y,z` → `u,v,w,p` | Job **5751** lazy 重训运行中；5739 OOM checkpoint **No-Go** · [分析记录](crown_original_vp/实验分析记录.md) |
| `crown_original_vp_pinn` | paper-original PINN 对照 | Job **5740** **TIMEOUT**（24h · 23 epoch）· evaluate **5750** 完成 · **No-Go** · [分析记录](crown_original_vp_pinn/实验分析记录.md) |

> 数据：`private_preprocessed_raw_ascii_v1`（raw_ascii 体素化 · merge 5738 完成）  
> 5625/5626 旧链路 **作废** · lazy 加载优化 **已审核通过**（[`docs/数据加载与评估加速说明.md`](../docs/数据加载与评估加速说明.md)）
