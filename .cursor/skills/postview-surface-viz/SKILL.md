---
name: postview-surface-viz
description: >-
  为 AAA/WSS 仓库生成可后处理的病例包：WSS/压力壁面 Gaussian 面片、原始点云、原生 CFD 壁面及原量/selfmax 对照图。
  用于 postview、点云回插、代表病例与截面展示；速度保留体内点云，按 run 适配 V5/WSS-min、历史 V3 或 CROWN 的坐标和指标口径。
---

# 病例场可视化与 ParaView 交付

交付可在后处理软件打开的场文件、预览与可追溯的数值来源。用户明确要求更窄范围时按其范围执行；完整病例包默认遵循 [交付与字段合同](delivery-fields.md)，不能只给点云或截图代替要求的壁面面片。下文仓库路径相对仓库根目录；技能参考文件相对本目录。

## 先识别数据合同

优先使用用户指定的 run、checkpoint、病例和变量；未指定时读根 `README.md`、`docs/README.md` 的当前入口，再读对应实验目录。以实际配置、预测 manifest、原始指标核对：

- `target`、数据根/版本、split、病例 canonical ID；不按目录名猜测模型输出。
- 帧号/时间模式、support/query 采样、checkpoint 选择规则、推理种子。
- 坐标单位、配准版本/旋转约定、标量单位/归一化、压力参考零点。

| 对象 | 按需阅读 / 入口 |
| --- | --- |
| 所有新建/补全病例包 | [delivery-fields.md](delivery-fields.md)：按变量交付、selfmax 分母、索引与验收合同 |
| V5 / WSS-min 直接壁面 WSS | [v5-wss-min.md](v5-wss-min.md)；`training_wss_min/tools/export_wss_postview.py` |
| V5 压力/速度体场 | [v5-wss-min.md](v5-wss-min.md) 的体场部分；检查批次工具是否固定 run 与输出 |
| 历史 V3P/GNN 或 CROWN | [legacy-v3-crown.md](legacy-v3-crown.md)；插值细节见 [reference.md](reference.md) |
| 真实连续截面 | 本文“网格选择”；再查 `tools/cfdpost_cloud_export/build_sliceable_volume.py` 与源数据坐标合同 |

2026-09-12 的 V5 直接 WSS 使用 peak 1162；历史 V3 展示通常为 merged-1146。后续仍以本次配置为准，不能给单帧图标注“81 帧 pooled”。

## 指标与展示分开核验

- 正式误差/R²来自原始同点预测和真值，按实验约定的点级、病例等权或 pooled 口径报告；不能在 Gaussian/STL 插值结果上重算后替代正式指标。
- 最好/最差病例按约定的正式逐病例预测 R²排序，中位病例取距分布中位数最近的实际病例；记录排序范围与规则。用户已指定病例则直接使用。多种子均值选例与某一个种子实际出图的 R²分别列出；回归拟合 R²不能代替预测 R²。
- 未指定病例范围时，可从已有、允许使用的评估中选最好/最差各一例。不能为了选两例暗中重新推理整队列；缺逐病例结果时先利用可用证据完成其余部分。
- CFD、Pred 在同一几何/坐标上比较，使用相同插值和 field 色标范围；signed error 为 pred−CFD，absolute error 单独标识。
- 每个可预测标量默认同时保留原量与 selfmax：CFD/CFDmax、Pred/Predmax，以及两种口径各自的有符号/绝对误差。分母取本病例、本帧原始完整同点评估域，并在点云、native、Gaussian 中固定复用；不在面片上重求最大值。selfmax 移除了幅值差，只比较分布。公式、压力负值与无效分母见 [字段合同](delivery-fields.md#selfmax-字段与固定分母)。
- 如另需 CFD 最大值共分母，作为独立标量明确命名；不能把它、训练 `true_norm/pred_norm` 或 min-max 显示缩放当作 selfmax。
- 顶点 top10% 与面积 top10% 不是同一指标；按 manifest 的 `surface_metric_mode`、面积映射状态标注。

## 网格选择

| 需求 | 合适产物 | 必查 |
| --- | --- | --- |
| WSS | 原始壁面点云 + Gaussian STL 面片；有身份一致的 CFD 壁面拓扑时另附 native 面片 | 面片必须有三角面；保留原量、selfmax 与映射质量 |
| 速度 | 体内点云，保留 CFD/Pred 速度向量和速度大小 | 以 query/单元身份及同帧真实壁面距离验域，不能只看 `point_kind`；默认不回插壁面，selfmax 使用向量模 |
| 壁面与体内均有压力 | 完整 wall∪interior 点云 + 仅壁面点回插的 Gaussian 面片；可用时附 native 面片 | 点云保留点类型，面片不混入体内压力点；所有产物共享完整原始域的 selfmax 分母 |
| 明确要求的体点云 slab/投影 | 点云 VTP、投影 PNG | slab/投影/深度均值不是连续 Slice，写明方法 |
| 连续腔内截面 | 带真实体单元连接的 VTU | 优先 Fluent 体拓扑；坐标与速度向量同帧，身份映射可核验 |

VTP 后缀不保证存在面，VTU 后缀也不保证连接正确。不要只改后缀，或把 Delaunay 凸包跨越血管分叉的补料当成原 CFD 流域。确需近似时说明局限并检查越界。

近壁低速点可能是真实体内单元中心，不能按速度值把它们当成壁面删除。完整点云遮挡内部时，可附几何半剖或薄层点集用于查看，保留完整原场、原分母及正式指标；具体身份核验和子集合同见 [体点身份与几何查看](delivery-fields.md#体点身份与几何查看)。

## 执行与验收

1. 优先复用已验收预测缓存、同点 CSV 与现有脚本；核对参数及默认输出，避免覆盖历史包。必要修改先完成并验证，再进入已授权的推理/出图。
2. 已授权的病例导出、必要本地汇总/绘图持续完成；授权包含 test 病例时，传递脚本要求的参数，不重复询问。新全量推理、批量 WSS 重评或训练不由“看看图”自动授权。
3. 大量推理/网格处理使用该路线集群入口；需要 node04 时读对应技能。独立病例/变量可并行，统一排序与色标；共享 batch manifest、README、工作簿由一个写者更新。
4. 读取 mapping report，检查覆盖率、映射距离、坐标叠合、镜像、分叉跨壁插值。覆盖率通过不等于面积映射通过；以该路线实际 Gate 为准。Gaussian 的 CFD/Pred 同一 stencil，误差在回插后的两个值之间重算。
5. 重新读取写出的 VTP/CSV：核对原点身份、数组值、向量分量、三角面与 selfmax 固定分母。不能仅用“能打开”验收。具体数值检查见 [交付与字段合同](delivery-fields.md#交付前核验)。
6. 打开代表 PNG 检查图例、单位、裁剪与色标；原量与 selfmax 分开出图。每组 CFD/Pred 共用色标，有符号误差用对称色标。VTU 另查非零体单元与场数组。
7. manifest/批次说明写清数据合同、选择/出图指标、selfmax 分母与定义、映射参数、验收结果和相对输出路径。附可直接打开的文件清单；包内路径随文件夹移动仍可用。

WSS 独立线交付写入 `docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md` 文首；历史 V3/通用交付写入 `docs/02-推进与变更/代码修改与实验推进记录.md`。单纯解释已有图时无需制造新产物或重复记账。回复给出图件预览、主包路径与影响解读的限制。
