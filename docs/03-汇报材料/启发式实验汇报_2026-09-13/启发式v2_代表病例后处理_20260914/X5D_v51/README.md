# X5D_v51 代表病例后处理包

v5.1 数据更新后**已经跑完**的密度增广 X5D（`X5D_v51` 五 seed），不是正在训练的纵向几何 `X5D_long`，也不是 cap/dual/noise 变体。嵌在本目录 `X5D_v51/` 下，不另开日期文件夹。

Best / Median / Worst 按 seed **1234 / 7 / 2025 / 11 / 2026** 的逐病例物理 R² 算术均值排序；Median 为距分布中位数最近的真实病例。**场值来自 s1234 / ckpt_best**，不是五模型平均场。

## 打开哪个文件

| 角色 | 病例 | 选例 R²：五 seed 均值 | 文件对应 R²：s1234 | 打开文件 |
| --- | --- | ---: | ---: | --- |
| best | AG/fast/ZHANG_LIANG | 0.904264 | 0.883796 | [点云](best/X5D_v51__best__AG__fast__ZHANG_LIANG__wall_wss.vtp) · [Gaussian壁面](best/surface_gaussian.vtp) · [原生壁面](best/cfd_wall_native.vtp) |
| median | AAA/ruputer/KANG_YONG | 0.760753 | 0.778080 | [点云](median/X5D_v51__median__AAA__ruputer__KANG_YONG__wall_wss.vtp) · [Gaussian壁面](median/surface_gaussian.vtp) · [原生壁面](median/cfd_wall_native.vtp) |
| worst | ILO/ZHANG_YONG_SHENG-0/before | 0.602119 | 0.609529 | [点云](worst/X5D_v51__worst__ILO__ZHANG_YONG_SHENG-0__before__wall_wss.vtp) · [Gaussian壁面](worst/surface_gaussian.vtp) · [原生壁面](worst/cfd_wall_native.vtp) |

全部为 test、peak **1162**、坐标系 `v5_atlas_frame_v1`、坐标 **mm**。正式 R² 来自原始同点壁面评估，没有在高斯插值后重算排名。test34 已暴露。

中位例与 `ILO/ZHANG_HE_PING-0/before`（均值 0.761603）距中位数（0.761178）几乎并列；协议取距离更小的 KANG_YONG。该例 s7 为 0.614，其余四 seed 0.778–0.818，图上 s1234=0.778 不能代表它的 seed 波动。

拟合散点：[X5D_v51 Pred vs CFD](../../启发式v2_代表病例拟合散点_20260915/figures/X5D_v51_scatter_fit.png)。R² 分布：[A_X5D_v51](../../启发式v2_R2分布_20260914/A_X5D_v51_distribution.png)。

## ParaView

`File → Open → Apply`。壁面 `Representation = Surface`；点云 `Points` 或 `Point Gaussian`。

| CFD | Pred | 有符号误差（Pred−CFD） |
| --- | --- | --- |
| `wss_cfd_pa` | `wss_pred_pa` | `wss_error_pred_minus_cfd_pa` |

selfmax（无量纲，分母为该例 peak 1162 原始完整壁面最大值）：`wss_cfd_selfmax`、`wss_pred_selfmax`、`wss_selfmax_error_pred_minus_cfd`。分母见 [归一化分母清单.csv](归一化分母清单.csv)。

## 预览与核验

- 原量总览：[X5D_v51_surface_triptychs_contact_sheet.png](../plots/X5D_v51_surface_triptychs_contact_sheet.png)
- selfmax 总览：[X5D_v51_surface_selfmax_triptychs_contact_sheet.png](../plots/X5D_v51_surface_selfmax_triptychs_contact_sheet.png)
- Gaussian：r=3 mm，sharpness=2，max_dist=3 mm，三例覆盖率 **100%**
- 最好/中位 STL 与 CFD 壁面一一对应；最差例 `ZHANG YONG SHENG-sq.stl` 更密（62,851 vs 60,329 顶点），最近距离最大 1.32 mm，有 216 个法向相反邻域。看热点请并排打开 `cfd_wall_native.vtp`
- 独立核验 **358 项全部通过**，见 [verification.json](verification.json)

复用已保存 `predictions.npz`，未训练、未推理。脚本在上一级目录：`select_x5d_cases.py`、`prepare_x5d_postview.py`、`verify_x5d_postview.py`、`render_surface_previews.py --models X5D_v51`。环境：`/public/newhome/cy/.conda/envs/GNN/bin/python`。
