# 后处理软件打开说明（可旋转 3D）

## 推荐：ParaView（免费，最适合旋转看面片云图）

### 你需要哪个文件？

| 文件 | 用途 | 能否旋转 |
| --- | --- | --- |
| **`xxx__surface_wall.vtp`** | **★ 主文件**：Gaussian 插值后的 STL 壁面 + 全部标量 | ✅ 面片云图，鼠标拖拽旋转 |
| `xxx__pointcloud_wall.vtp` | 原始 GNN 壁面采样点（未插值到 STL 三角面） | ✅ 点云旋转 |
| `xxx.stl` | 纯几何，无标量 | ✅ 仅看形状 |
| `xxx__surface_wall.csv` | 面片顶点坐标 + 标量（给 CFD-Post / Excel） | ❌ 需导入后处理 |
| `GNN_blue_white_red.xml` | 自定义蓝-白-红色标 | — |

**不需要单独再配 STL**：`surface_wall.vtp` 里已含三角面网格 + 点数据标量，ParaView 直接打开即可。

### ParaView 操作步骤（约 1 分钟）

1. 打开 **ParaView** → **File → Open** → 选 `xxx__surface_wall.vtp` → **Apply**
2. 左侧 **Representation** 选 **Surface**（不要 Points）
3. **Coloring** 下拉选变量：
   - `wss_cfd` — CFD 真值 WSS
   - `wss_pred` — GNN 预测 WSS
   - `err_wss` — 误差（pred − cfd）
   - `abs_err_wss` — 绝对误差
   - `p_cfd` / `p_pred` / `err_p` — 压力
4. **旋转**：鼠标左键拖拽；滚轮缩放；Shift+左键平移
5. **色标**（蓝-白-红）：
   - 点击色条 → **Edit Color Map**
   - **Load preset** → 导入同目录 `GNN_blue_white_red.xml`
   - 或手动选内置 **Blue White Red** / **Cool to Warm**
   - CFD 与 Pred 两图：**Use separate color scales 关闭**，手动设相同 Data Range
6. 导出高清图：**File → Save Screenshot**（比 Python 静态 PNG 更清晰）

### 三列对比（CFD | Pred | Error）

- **方式 A**：开 3 个 ParaView 窗口，色标范围设一致
- **方式 B**：View → **Split View** → 复制 pipeline（Ctrl+D）→ 各视图改 Color 变量

---

## 备选：ANSYS CFD-Post

CFD-Post **不能直接读 VTP**，可用：

1. **路线 B2**：`fluent_cloud/*__fluent_wss_*.csv`（若已跑 `run_all_routes.sh`）
2. **本包 CSV**：`xxx__surface_wall.csv`（x,y,z + 标量）→ **File → Import → Point Cloud**
3. 需先有 **Fluent `.cas`** 网格；真值云图还需 `.dat`

详见仓库 `tools/cfdpost_cloud_export/CFDPOST_操作指南.md`

---

## 备选：Fluent / Tecplot

- CSV 列：`x,y,z,wss_cfd,wss_pred,err_wss,...`（单位 Pa，坐标 mm）
- STL 仅几何；标量通过 CSV 点云或面片顶点表关联

---

## 变量与单位

| 变量 | 含义 | 单位 |
| --- | --- | --- |
| wss_cfd / wss_pred | 壁面剪切应力 | Pa |
| p_cfd / p_pred | 压力 | Pa |
| err_wss / err_p | pred − cfd | 同左 |
| map_dist | 映射最近点距离 | mm |
| map_valid | 是否有效插值 | 0/1 |

**指标口径**：正式 R² 仍在原始同点点云 CSV 上算；VTP/面片图仅供可视化。
