# SHEN_FANG_JIN 体场后处理：在 EnSight 里切截面看速度 / 压力

病例 `AAA/unruputer/SHEN_FANG_JIN`（cv3 fold 2 留出，中位病例）。2026-09-23 整理。

## 先回答：要不要先插值

要。体场预测是点云：只有点，没有体单元。EnSight / ParaView 的 Clip 只能在体单元内部插值，
对纯点云切出来只能是散点，连不成面。

这里把点云值插到了**这个病例自己的 Fluent 体网格**上，也就是解剖区 `blood` 的 677082 个单元，
不含 blood1–5 延长段。结果写成 EnSight Gold 瞬态 case：

| 在 EnSight 里拖动 / 切换 | 变化的是 |
| --- | --- |
| 时间条（4 个时间步） | 相位：加速 → 峰值 → 减速 → 谷底 |
| Clip 的位置数值 | 截面位置（沿血管移动） |
| Color by 选的变量 | 指标：CFD 真值 / 两个模型的预测 / 预测误差 / 压力 |

## 目录里有什么

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| **`ensight/SHEN_FANG_JIN_flow.case`** | EnSight Gold 瞬态 case（几何 + 13 个节点变量 × 4 个相位） | **主入口**，EnSight 或 ParaView 直接打开 |
| `ensight/SHEN_FANG_JIN_lumen.geo`、`ensight/vars/` | case 引用的几何与变量文件 | 与 `.case` 放在一起，别拆开 |
| `ensight/README.md` | 精简版打开要点 | 单独拷走 `ensight/` 时随身带着 |
| `ensight/preview/` | 用 VTK 回读这份 case 后切出来的预览图（横截面 Z、纵剖面 Y × 速度 / 压力） | 打开前先看效果 |
| `ensight/build_report.json`、`ensight/ens_checker.log` | 构建核对记录；EnSight 自带格式校验器的完整输出 | 查数 |
| `aligned_geometry.stl` | 同一坐标系的血管 STL | 需要外壳时一起载入 |
| `velocity/<臂>/<相位>/`、`pressure/<臂>/<相位>/` | 原始点云（`interior_pointcloud.vtp`、`full_pointcloud.vtp`）、同点 CSV（`_export/`）、压力壁面面片 | 正式 R² / MAE 在这里算；**不用于切面** |

## EnSight 打开步骤（2023 R1）

本机 EnSight 没有批处理许可证，以下操作没法在图形界面里实测，菜单名称可能略有出入。
文件格式已用 EnSight 2023 R1 自带的 `ens_checker231` 校验通过，无警告；也用 VTK 回读切过面，
见 `ensight/preview/`。

1. **拷贝**：把整个 `ensight/` 文件夹（约 0.8 GB）拷到纯英文路径下，例如
   `D:\SHEN_FANG_JIN_ensight\`。EnSight 在 Windows 上遇到中文路径可能读不进来。
2. **打开**：File → Open → 选 `SHEN_FANG_JIN_flow.case` → 全部载入。部件列表里只有一个部件
   `lumen (Fluent blood zone)`，默认显示为血管外表面。
3. **建截面**：选中 lumen 部件 → 建 Clip，类型 **XYZ**，Domain 选 Intersect。
   - 轴 **Z**、数值 **24.5**：瘤体最宽处的横截面（坐标系里 +Z 指向入口，也就是近心端）。
   - 轴 **Y**、数值 **10.5**：过瘤体中心的纵剖面。
   - 拖 Clip 面板里数值的滑块，截面就沿该轴移动。
4. **着色**：选中 Clip 部件 → Color by → 选下表的变量。
5. **切相位**：拖顶部的时间条，或逐步前进。4 个时间步分别是 0.13 s 加速、0.21 s 峰值、
   0.35 s 减速、0.48 s 谷底。Clip 和颜色会跟着自动更新。
6. **看清截面**：把 lumen 部件的透明度调到 0.1–0.2，或者直接隐藏。
7. **速度矢量**：选中 Clip 部件 → 建矢量箭头（Vector arrows），变量选 `velocity_CFD` 或 `velocity_VT0`。
8. **比较 CFD 和预测**：同一个 Clip 依次切换 `speed_CFD` / `speed_VT0` / `speed_VTB4`，
   调色板手动固定为同一个范围（见下方建议），不要让它自动缩放。

### 如果显示成点或方框

之前那份 477 万四面体的 case 在 EnSight 里只显示 48730 个客户端单元、看起来是点。
这更像是 EnSight 的快速显示（Fast display / 静态 LOD）把大模型抽成了点，而不是数据本身的问题。
这一份的规模就是直接用 EnSight 打开 Fluent `.cas` 时的那套网格（677082 个单元）。
如果仍然显示成点，依次检查：

- 工具栏的 Fast display 是否开着，关掉；
- Edit → Preferences → Performance 里的静态快速显示（static fast display）是否勾选；
- 部件的显示方式应为 `3D border, 2D full`，不要选「只载入点 / 法向」一类选项。

## 变量

13 个节点变量，4 个时间步用同一组变量名。单位：长度 mm，速度 m/s，压力 Pa。

| 变量 | 含义 |
| --- | --- |
| `speed_CFD` / `speed_VT0` / `speed_VTB4` | 速度大小 \|u\|：CFD 真值 / 预测臂 VT0 / 预测臂 VTB4 |
| `speed_err_VT0` / `speed_err_VTB4` | 预测 − CFD（带符号，m/s） |
| `velocity_CFD` / `velocity_VT0` / `velocity_VTB4` | 速度矢量；Color by 选它时 EnSight 显示的 Magnitude 就等于对应的 `speed_*` |
| `pressure_CFD` / `pressure_PT0` / `pressure_PTB8` | 相对压力 p − p_体均(t)：CFD / 预测臂 PT0 / 预测臂 PTB8 |
| `pres_err_PT0` / `pres_err_PTB8` | 预测 − CFD（带符号，Pa） |

预测臂：`VT0` / `PT0` 是时间条件直接输出头；`VTB4` / `PTB8` 是时间基输出头（K = 4 / 8）。
checkpoint 均为 fold2 `ckpt_best.pt`。

### 建议色标范围（该相位 CFD 全体网格节点的 p1–p99）

| 相位 | 时间 | Q/Qpeak | 速度 (m/s) | 压力 (Pa) |
| --- | ---: | ---: | --- | --- |
| 加速 accel | 0.13 s | 0.41 | 0 – 0.22 | −511 – 277 |
| 峰值 peak | 0.21 s | 1.00 | 0 – 0.58 | −376 – 31 |
| 减速 decel | 0.35 s | 0.43 | 0 – 0.36 | −104 – 119 |
| 谷底 trough | 0.48 s | 0.003 | 0 – 0.13 | −16 – 53 |

以上是整个腔体的范围。单个截面上的值通常更窄，可以在截面上再收紧。

## 插值怎么做的

- **网格**：直接取 Fluent `.cas` 的面表，把 `blood` 区每个单元原样组装成 EnSight `nfaced` 多面体，
  共 677082 个单元、2103605 个节点。面法向统一朝外，逐单元核对都是封闭的；体积 369368 mm³，
  和 Fluent 一致。坐标做了 V5 刚体配准（毫米），与点云、`aligned_geometry.stl` 在同一坐标系。
- **对应关系**：体内点云的 677082 个点就是这些单元的体积中心，按 `cell_id_cas` 一一对应，
  坐标逐点对上（最大偏差 0.002 mm）。壁面点云的 85304 个点就是这套网格的壁面节点
  （最大偏差 1e-5 mm）。
- **节点值**：
  - 体内节点：取相邻单元中心值，按 1/距离 加权平均，也就是 Fluent / CFD-Post 算节点值的做法；
  - 壁面节点的速度：取 0，这是无滑移边界条件，不是模型预测；
  - 壁面节点的压力：直接取壁面点云自己的值。

  EnSight 在单元内部再做线性插值，所以截面是连续填色。
- **保真度**：把节点值平均回单元中心，与原始点值比较：CFD 的 R² ≥ 0.996；预测在加速、峰值、
  减速三个相位 ≥ 0.985，谷底 VT0 为 0.92、VTB4 为 0.96。
  谷底时速度很小，预测点值本身有点级噪声，节点平均会把这部分抹平一些。

## 注意

- 只用于显示。正式 R² / MAE 仍在同点 CSV / 点云上算，不在截面上重算。
- 预测截面里能看到放射状纹理，CFD 截面里没有。两者用的是同一套插值，所以纹理来自模型输出本身，
  不是插值造成的。
- 壁面速度为 0 是边界条件，贴壁那一圈的颜色不代表预测好坏。
- 用 ParaView / VTK 切面时，个别单元可能出现针尖大小的白点：Fluent 多面体里少数面不是平面，
  VTK 三角化这些单元时会报 non-manifold 并跳过。本例两个截面里只碰到 1 个这样的单元（每个相位报一次，共 4 次），就是纵剖面预览里瘤颈处的小白点。
  这不是文件问题：677082 个单元逐一核对过都是封闭的、面朝向一致。EnSight 切多面体用的是它自己的分解方式。

## 重建 / 换病例

```bash
conda activate GNN
python training_wss_min/tools/build_ensight_cfd_mesh_case.py --case-dir <postview 病例目录>
```

输入：病例目录下的 `velocity/<臂>/<相位>/interior_pointcloud.vtp` 和
`pressure/<臂>/<相位>/full_pointcloud.vtp`，上级目录的 `manifest.json`（相位定义），
该例 Fluent `.cas`（路径取自拓扑审计）。本例约 4 分钟。输出目录已存在时工具会拒绝覆盖。
