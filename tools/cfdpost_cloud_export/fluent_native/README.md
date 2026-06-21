# 路线 A：Fluent 原生 CFD 真值文件

将你在 Fluent 中导出的 **Case + Data** 放在此目录，供 CFD-Post 加载。

## 需要放置的文件

| 文件 | 说明 |
| --- | --- |
| `GUO_XI_JIANG.cas` 或 `.cas.gz` | 网格（可从 `data_new/AG/slow/GUO_XI_JIANG/` 复制） |
| `GUO_XI_JIANG.dat` | **你在 Fluent 中 File → Write → Data 导出** |

## Fluent 导出步骤（摘要）

1. Read Case → 加载对应算例  
2. 切换到与 GNN 样本一致的时间步（如 `merged-1120`）  
3. File → Write → Data → 保存为本目录下的 `.dat`  
4. CFD-Post：Load Result → 选 `.cas` + `.dat` → Wall Contour  

完整说明见 [../三条对比路线.md](../三条对比路线.md) 路线 A 一节。

## 说明

- 本仓库 **不包含** `.dat`（体积大且依赖你本地 Fluent 版本）。  
- 没有 `.dat` 时，仍可用 CSV 中的 `wss_cfd` / `p_cfd` 做 GNN 对比（路线 B/C）。
