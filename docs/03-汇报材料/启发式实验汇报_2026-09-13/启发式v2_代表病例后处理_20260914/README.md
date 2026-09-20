# X5X11 / VF6 / PF6 代表病例后处理包

> **历史 V4 EMA**：[PointNet BC+PDE-EMA 的 best / median / worst](V4_EMA/README.md)。选例按 test35 派生 WSS R²；同例补了体内速度和相对压力。不是 formal P2V/D2，不是 X5D_long。
>
> **X5D_v51 直接 WSS**：[v5.1 密度增广 X5D 的 best / median / worst Gaussian 壁面](X5D_v51/README.md)。选例按五 seed 物理 R² 均值；场来自 s1234。不是纵向几何 X5D_long。
>
> **09-16 VELWSS2 派生 WSS**：[VF6 预测速度 → 冻结 V3 的 best / median / worst Gaussian 壁面](VF6_wss/README.md)。选例按三 seed 派生 WSS R² 均值；场来自 s1234。Median 是 ILO/LU_FU_SHAN-0/before，与速度 VF6 的选例不同。
>
> **09-15 速度查看补充**：[VF6 完整体内／半剖／1 mm 薄层 VTP](VF6_体内点云查看版_20260915/README.md)。独立几何核验确认原速度文件无壁面节点；整管蓝色外层是近壁低速体点遮挡。新包增加可露出内部的几何查看子集，原完整场、病例角色和正式指标保留，归一化分母固定取原完整域。
>
> **09-15 拟合散点**：[最好 / 中位 / 最差 Pred vs CFD](../启发式v2_代表病例拟合散点_20260915/README.md)，红线 `pred = a·CFD + b`。

修订：2026-09-14，**selfmax 字段版（schema 3）**。9 个病例已重导，原始 CFD/Pred/Error 与新增的各自最大值归一化字段均在同一份场文件中。直接从下表打开 VTP，或查看 [打开文件清单.csv](打开文件清单.csv)。

## 本次交付

| 模型 | 点云 | 壁面文件 |
| --- | --- | --- |
| X5X11 · WSS | 原始壁面点云，CFD/Pred/Error（Pa）及 selfmax | `surface_gaussian.vtp`：点值高斯回插到 STL；`cfd_wall_native.vtp`：原生 CFD 三角面，同点值，无平滑；两者均含 selfmax |
| VF6 · 速度 | **仅体内点云**，速度大小与 CFD/Pred 三分量向量（m/s），以及速度大小 selfmax | 不做速度壁面插值 |
| PF6 · 压力 | **壁面∪体内完整点云**，相对压力 p−p_ref（Pa）及 selfmax | `surface_gaussian.vtp`：仅用壁面压力点高斯回插；`cfd_wall_native.vtp`：原生壁面压力，无平滑；两者均含 selfmax |

上一版遗漏了真实三角面插值；同时，PF6 旧点云存在将壁面压力挂到体内坐标的索引错误。当前文件已用原始 `query_idx` 在“壁面坐标＋体内坐标”的拼接数组中重新定位，并逐点核对真值与母库。VF6 正式交付只有体内点云，不含速度壁面投影。

## 打开哪个文件

Best / Median / Worst 沿用既有选例：按 seed 1234、7、2025 的逐病例 R² 算术均值排序；Median 为最接近分布中位数的真实病例。**文件里的场值来自 s1234 / ckpt_best 单次预测**，不是三模型平均场，因此下表单列两种 R²。各模型独立选例，Median 不一定是同一个病例。

| 模型 | 角色 | 病例 | 选例 R²：三 seed 均值 | 文件对应 R²：s1234 | 打开文件 |
| --- | --- | --- | ---: | ---: | --- |
| X5X11 | best | AG/fast/ZHANG_LIANG | 0.877710 | 0.871067 | [点云](X5X11/best/X5X11__best__AG__fast__ZHANG_LIANG__wall_wss.vtp) · [Gaussian壁面](X5X11/best/surface_gaussian.vtp) · [原生壁面](X5X11/best/cfd_wall_native.vtp) |
| X5X11 | median | AG/fast/YAO_CUN_HONG | 0.748361 | 0.720141 | [点云](X5X11/median/X5X11__median__AG__fast__YAO_CUN_HONG__wall_wss.vtp) · [Gaussian壁面](X5X11/median/surface_gaussian.vtp) · [原生壁面](X5X11/median/cfd_wall_native.vtp) |
| X5X11 | worst | AAA/unruputer/SUN_SHU_MING | 0.466580 | 0.498277 | [点云](X5X11/worst/X5X11__worst__AAA__unruputer__SUN_SHU_MING__wall_wss.vtp) · [Gaussian壁面](X5X11/worst/surface_gaussian.vtp) · [原生壁面](X5X11/worst/cfd_wall_native.vtp) |
| VF6 | best | AG/fast/ZHANG_LIANG | 0.916771 | 0.916368 | [点云](VF6/best/VF6__best__AG__fast__ZHANG_LIANG__volume_velocity.vtp) |
| VF6 | median | AG/slow/LI_HUAN_GE | 0.799679 | 0.771183 | [点云](VF6/median/VF6__median__AG__slow__LI_HUAN_GE__volume_velocity.vtp) |
| VF6 | worst | AAA/unruputer/SUN_SHU_MING | 0.481279 | 0.437689 | [点云](VF6/worst/VF6__worst__AAA__unruputer__SUN_SHU_MING__volume_velocity.vtp) |
| PF6 | best | AG/fast/YAO_CUN_HONG | 0.965040 | 0.967960 | [点云](PF6/best/PF6__best__AG__fast__YAO_CUN_HONG__volume_pressure.vtp) · [Gaussian壁面](PF6/best/surface_gaussian.vtp) · [原生壁面](PF6/best/cfd_wall_native.vtp) |
| PF6 | median | AG/fast/ZHANG_CHUN | 0.851082 | 0.860667 | [点云](PF6/median/PF6__median__AG__fast__ZHANG_CHUN__volume_pressure.vtp) · [Gaussian壁面](PF6/median/surface_gaussian.vtp) · [原生壁面](PF6/median/cfd_wall_native.vtp) |
| PF6 | worst | AAA/unruputer/SUN_SHU_MING | 0.375508 | 0.348734 | [点云](PF6/worst/PF6__worst__AAA__unruputer__SUN_SHU_MING__volume_pressure.vtp) · [Gaussian壁面](PF6/worst/surface_gaussian.vtp) · [原生壁面](PF6/worst/cfd_wall_native.vtp) |

全部为 test、peak step **1162**（`steps` 数组索引 21）、坐标系 `v5_atlas_frame_v1`、坐标单位 **mm**。WSS R² 针对原壁面点，速度 R² 针对体内速度大小，压力 R² 针对壁面∪体内相对压力。正式指标仍来自原始同点评估，**没有在高斯插值后重新计算排名或 R²**。

## ParaView 操作

1. 在上表选择文件，`File → Open → Apply`。
2. 壁面文件选 `Representation = Surface`；点云选 `Points` 或 `Point Gaussian`。ParaView 的 `Point Gaussian` 只是点的显示方式，实际壁面插值数据是单独的 `surface_gaussian.vtp`。
3. 在 Coloring 选择下表字段；复制视图分别显示 CFD、Pred、Error。CFD 与 Pred 手动设置相同色标范围，误差设置以 0 对称的色标。
4. PF6 点云中 `point_kind=0` 是壁面、`1` 是体内，可用 Threshold 分开看。VF6 可用 `dist_to_wall_mm` 筛选近壁体内点，或用 Glyph 显示速度向量。

| 模型 | CFD | Pred | 有符号误差（Pred−CFD） |
| --- | --- | --- | --- |
| X5X11 | `wss_cfd_pa` | `wss_pred_pa` | `wss_error_pred_minus_cfd_pa` |
| VF6 | `speed_cfd` | `speed_pred` | `speed_error_pred_minus_cfd` |
| PF6 | `pressure_cfd_pa` | `pressure_pred_pa` | `pressure_error_pred_minus_cfd_pa` |

VF6 向量为 `velocity_cfd_vector`、`velocity_pred_vector`，坐标与向量采用同一配准旋转。点云没有真实体单元连接，不是可直接生成连续 Slice 的体网格；本包保留用户要求的体内点云。

PF6 的压力参考为该病例 peak 时刻体内平均压力：YAO_CUN_HONG **16558.269708 Pa**、ZHANG_CHUN **17938.964798 Pa**、SUN_SHU_MING **13141.024252 Pa**。CFD 和 Pred 共用同一参考值，没有额外拟合压力偏置。

## 新增：各自最大值归一化（selfmax）

在 ParaView Coloring 中直接选择以下字段，无需手动 Calculator：

| 模型 | CFD / CFD 自身 max | Pred / Pred 自身 max | 两者有符号差值 |
| --- | --- | --- | --- |
| X5X11 | `wss_cfd_selfmax` | `wss_pred_selfmax` | `wss_selfmax_error_pred_minus_cfd` |
| VF6 | `speed_cfd_selfmax` | `speed_pred_selfmax` | `speed_selfmax_error_pred_minus_cfd` |
| PF6 | `pressure_cfd_selfmax` | `pressure_pred_selfmax` | `pressure_selfmax_error_pred_minus_cfd` |

例如 `wss_pred_selfmax = wss_pred_pa / max(原始 wss_pred_pa)`，`wss_cfd_selfmax = wss_cfd_pa / max(原始 wss_cfd_pa)`，正是 Pred/Predmax、CFD/CFDmax。另有 `{wss|speed|pressure}_selfmax_abs_error` 绝对差，以及 CFD、Pred、error 各自的有效性 mask。全部 selfmax 字段**无量纲，仅用于空间分布对照，已移除 CFD 与 Pred 之间的幅值差**；原始物理量与 R² 仍需一起查看。旧 `wss_*_norm` 是训练归一化字段，与本次 selfmax 不同。

分母来自该病例 peak 1162 的**原始完整同点域**：WSS 为 wall，速度为 interior 的速度大小 `|u|`，压力为 wall∪interior 的相对压力。原始点云、壁面专用 CSV、原生面和 Gaussian 面共享这对固定分母，不在壁面子集、截图或插值后重新取 max。因平滑降低极值，Gaussian 面片的 selfmax 最大值可以小于 1。

- WSS 和速度 selfmax 的 CFD/Pred 可统一设 **0–1** 色标；误差另用完整对称范围。
- 压力严格按带符号的 `p−p_ref` 除以各自 max，保留原参考零点，**允许负值与小于 −1**，不改为 maxabs/min-max，也不裁到 0–1；CFD/Pred 使用实际联合范围。
- 分母数值、原单位、来源域和有效状态见 [归一化分母清单.csv](归一化分母清单.csv)、每例 manifest 的 `display_normalization`，以及 VTP FieldData 中的 `{prefix}_cfd_max`、`{prefix}_pred_max`。9 例分母均为有限正数；通用工具对接近 0 或非有限分母输出 NaN 与 valid=0，不以 1 替代。

## 插值与配准核验

6 份 WSS/压力面片采用技能参数：**Gaussian radius=3 mm，sharpness=2，max_dist=3 mm，fallback=mask**。CFD 与 Pred 使用同一邻域、同一权重；有符号误差由插值后的 Pred−CFD 计算。每份面片都含真实 STL 三角面，以及 `map_valid`、`map_dist_mm`、邻域点数和几何诊断数组。

- 6 份面片有效顶点覆盖率均为 **100%**；每份含 **19,293–29,737** 个真实三角面。
- STL 与原始 CFD 壁面节点一一对应，单位倍率 **1.0**，不拟合包围盒；按 `(raw_mm−centroid) @ rotation.T` 配准后，最大最近壁面点距离 **3.69×10⁻⁵ mm**。
- 独立复核 **901 项检查全部通过**：9 例原量、坐标、selfmax 公式与分母一致，源/输出哈希核对通过，共记录 64 个文件 SHA256（含验证代码）；核验所有带场 VTP 的 FieldData 和 21 份 CSV 的行数/字段/抽样数据，另覆盖 16 个归一化工具边界检查。6 面独立抽样重算 Gaussian，数值差在 1e-9 容差内。每例 `mapping_report.json` 和 `manifest.json` 保存参数、覆盖率与来源，完整复核见 [verification.json](verification.json)。
- 欧氏距离高斯邻域在局部折叠/分叉处有混合信号：YAO_CUN_HONG 有 **12** 个顶点的邻域包含局部三角图不连通点，最大权重 **15.3%**；ZHANG_CHUN 有 **16** 个，最大 **13.8%**。其余展示壁面此项为 0；另有法向相反邻点诊断，详见报告。这些是几何诊断，不能仅凭覆盖率宣称没有跨壁混合。

因此分析局部热点时，请将 Gaussian 图与 **`cfd_wall_native.vtp` 原生面**并排检查；后者保留同点预测幅值，没有平滑。覆盖率不代表严格面积映射已验收，本包不新增面积热点指标。

## 快速预览与追溯

- 壁面：[原物理量总览](plots/surface_triptychs_contact_sheet.png) / [selfmax 总览](plots/surface_selfmax_triptychs_contact_sheet.png)。第一行 X5X11，第二行 PF6；各行 Best / Median / Worst。单张高清图在 [plots/](plots/)。VF6 场文件保持体内速度点云，预览仅画体内点，不做壁面回插。
- 速度点云：[原物理量总览](plots/VF6_volume_pointcloud_contact_sheet.png) / [selfmax 总览](plots/VF6_volume_pointcloud_selfmax_contact_sheet.png)。依次为 Best / Median / Worst。
- 本包共 **18 张三联图＋4 张总览**。速度预览每例固定抽取 30,000 个体内点，仅用于绘图；CFD/Pred/Error 与原量/selfmax 两套图共用相同索引和相机，分母及色标范围仍取完整点云。抽样索引随图保存，原始 VTP 不减点。这是体内点云正交投影视图，不是连续截面。
- 预览使用真实三角面、固定配准坐标相机；每例 CFD/Pred 共用完整值域，误差用完整对称范围。**不同病例不共用同一数值色标**，跨病例比较幅值时需再统一范围。PF6 壁面预览中的正式 R² 仍针对原始壁面∪体内点。
- 每例 `_export/same_point_fields.csv.gz` 是原始同点场；WSS/PF6 另有壁面专用 `gaussian_source.csv.gz` 和插值顶点 CSV。批次 [manifest.json](manifest.json) 汇总全部 9 例。
- 所有 VTP 均自带坐标、连接和场值；移动文件夹后仍可直接打开。包内输出链接为相对路径；`source_*.npz` 软链接及 manifest 的源绝对路径只用于本工作区追溯，复制到别的机器时不影响 VTP 使用。
- [prepare_selected_postview.py](prepare_selected_postview.py) 复用选定病例缓存完成导出；[render_surface_previews.py](render_surface_previews.py) 和 [render_velocity_previews.py](render_velocity_previews.py) 生成壁面/速度预览；[verify_selected_postview.py](verify_selected_postview.py) 独立复核文件和字段。运行环境：`/public/newhome/cy/.conda/envs/GNN/bin/python`，导出脚本需仍位于当前仓库内。可复用 selfmax 实现在 `tools/cfdpost_cloud_export/display_fields.py`；后续交付遵循仓库 `postview-surface-viz` 技能。本次没有重新训练、模型推理或新增全测试集评估。
