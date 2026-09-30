# Centerline V5：点云 atlas 接入与出口命名修正记录

> 创建：2026-09-07｜状态：进行中（随 V5 数据/训练线滚动回填）｜前身：[Centerline V2 全队列修复与切换记录（2026-08-28）](../_archive/Centerline_V2全队列修复与切换记录_2026-08-28.md) 自本文起冻结为 V2 切换史，不再追加。
>
> 本文只记录 V5 如何消费中心线/atlas、V5 阶段对其发现与修正、以及尚未回灌上游的待办；不改动 V2 产物与 09-06 atlas 文件本身。相关执行文档：[V5 数据重建记录](../_archive/WSS_PINN/_archive/WSS_V5_数据重建_Pilot与全量构建记录_2026-09-06.md)、[V5 训练实验跟踪](../00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)。

## 1. V5 消费的中心线资产（不变项）

| 资产 | 位置 | V5 读法 |
| --- | --- | --- |
| Centerline V2 病例产物（`centerline_graph_mm.vtp`、`centerline_paths_mm.*`、`openings_mm.vtp`、`surface_authoritative_mm.stl`、`surface_selection.json`、`result.json`） | `outputs/centerline_v2_full_173_20260828/cases/<cid>/` | 不直接读；经 atlas |
| 三例 STL≠CFD 壁面的网格壁面重提（ZHOU_KE_XUN、LIU_WEN_QI、YU_XIANG_SHENG-1） | `outputs/centerline_v2_meshwall_20260904/` | 经 atlas（`--source-root` 混合来源） |
| 7 段唯一分支 feature atlas（schema 2，2026-09-06；半径自适应 SG 窗、端区持值、真实 mm = V2 帧 × 1000/factor） | `outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/atlas/cases/<cid>/{atlas.npz,summary.json}` | `wss_v5/centerline_features.load_atlas`；29 列原样存入 `case.h5:geometry/atlas_table` |

atlas 与 CFD 壁面处于同一真实 mm 坐标系：172 例壁面节点到最近 atlas 样本的距离与局部半径之比 median |r−R|/R 在 0.04–0.16 之间（V5 gate 阈值 0.25，全过）。

## 2. V5 在 atlas 之上的派生（存于 `case.h5`，不写回 atlas）

- **RMF 局部标架**：每段按 Wang 2008 双反射法生成旋转最小化标架，子段在分叉处继承父段法向（`geometry/atlas_frame_n|b`）。
- **逐点管道坐标**：任意点 → 最近非重复样本：`segment_id, semantic_id, s_local, s_from_root, radius, curvature, dr_ds, dist_to_junction/endpoint, junction/endpoint mask, end_zone, r, rho=r/R, theta（RMF 方位角）, axial_offset, junction_ambiguous`（次近样本属其他段且距离 ≤1.2 倍时置真）。壁面与体域同名字段。
- **分段语义**：trunk=0、左/右髂总=1/2（由叶子出口名推断）、左髂外/内=3/4、右髂外/内=5/6。
- **解剖坐标架（供 `training_wss_min` 视图）**：原点 = 主干末端（分叉），+Z = 主干末端→入口弦向，+X = 左髂总方向（由语义 1/2 样本均值差投影得到），逐例壁面 max-abs 归一到 [−1,1]。172 例中 171 例由语义确定 +X；`AAA/ruputer/WANG_KUI_WU` 回退世界 +X（其出口重命名为跨侧互换 out-le↔out-ri，使一条髂总的叶子同时含左右出口名，左右髂总语义判为不唯一，见 §3 与 §5）。解剖"左"轴在 98% 病例指向世界 −X，与旧 `stl_landmarks_v4`（世界 +X 定左右）多数互为镜像；模型从零训练不受影响，新旧权重不可互用。

## 3. 出口命名修正（8 例）

V5 构建复核盖面与真实接口平面偏移时发现，atlas 叶段 `outlet_name`（源自 V2 在归一化坐标系里的开口分配 `opening_assignment`）在 8 例与网格接口语义（求解器 BC 绑定的 zone，拓扑审计 `interfaces[].semantic_label`）互换；几何位置本身正确（端点到"正确"接口中心 ≤0.7 mm，GAO_FENG_SHAN 一处 3.0 mm）。

| case | atlas 命名 → 几何对应 |
| --- | --- |
| AG/fast/LOU_YANG、ILO/YAO_GUO_CHEN-0/before、AG/slow/LI_HUAN_GE | out-ri ↔ out-re |
| AAA/unruputer/HAN_JIAN_FU | out-le ↔ out-li |
| AG/fast/LIU_FENG_MING | out-ri ↔ out-le |
| AAA/ruputer/WANG_KUI_WU | out-le ↔ out-ri |
| AG/slow/CHEN_JING_RU | out-re ↔ out-le |
| AAA/ruputer/GAO_FENG_SHAN | out-li→out-ri、out-le→out-li、out-ri→out-le |

修正规则（`wss_v5/centerline_features.relabel_outlets`）：每个叶端点取最近的解剖出口接口中心；接受条件为距离 ≤ max(3 mm, 2R_端点) 且次近接口距离 ≥ 2 倍最近距离，且无两叶争同一标签；原名保留在 `outlet_name_atlas`，重推语义；gate `atlas_outlet_labels_resolved`。另有 6 例（LI_BING_JIANG、YANG_BEN_RUI、HE_SHU_ZHEN、LIU_FENG、LI_CHONG_ZENG、LI_GUI_YING）端点离接口 4.2–9.3 mm 但命名正确，规则放宽后通过。

影响面：V5 `semantic_id` 已修正；atlas 文件未改；旧 `geometry_v2` 的 `outlet_id` 特征与任何直接读 atlas `outlet_name` 的下游在这 8 例上带同样的错误。

**遗留不一致（1 例）**：`AAA/ruputer/WANG_KUI_WU` 重命名后仍是"同一髂总下挂左右两侧出口名"（CIA1 → out-le + out-ri，CIA2 → out-li + out-re；atlas 原名同样混侧，只是两者互换）。几何对应无歧义（端点到接口 ≤0.4 mm），说明该例**网格侧 zone 命名本身不按解剖左右**（求解器按此命名绑定 RCR，协议分析"左右 50/50"对该例是按 zone 名而非解剖侧）。当前 V5 把两条髂总都判为"左"（`semantic_of_segment` 1/1），解剖坐标架回退世界 +X。计划修正：`_semantics` 对叶子跨侧的髂总置 −1 并写 `semantic_tree_consistent=false`，单例刷新（pc-v3.1）；训练 Wave 1/2 未使用 `semantic_id`，不受影响。

## 4. 端点、开口与虚拟盖

- atlas 端点相对真实解剖切面平面的偏移大多在 ±1.4 mm 内；HAN_JIAN_FU 的 out-le/out-li 端点超出切面 4–5 mm，6 例端点缩在切面内 4–9 mm（§3 所列）。
- V5 的部署几何程序不依赖端点精确落在切面：盖面按端点附近点云 rim 的最小二乘平面拟合（24 扇区最远点，倾角 >40° 回退切线方向），半径取 rim 横向范围的 p90。斜切面（WANG_KUI_WU、CHENG_LU_LI、LIU_BAO_JUN、WANG_MAN_TIAN）由此通过域 gate。

## 5. 待办与跟踪

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 上游 V2 `opening_assignment` 改为真实 mm 帧按最近 CFD 接口中心分配 | 待做 | 目前只在 V5 侧修正；若重跑 V2/atlas 应一并修，避免下游再次继承 |
| 端点缩入切面 4–9 mm 的 6 例：核查 V2 端点截断规则 | 待做 | 不影响 V5（盖面吸附 rim），但影响以端点定义"入口/出口位置"的任何特征 |
| 部署侧：新病例 VMTK 中心线的出口命名规则 | 待定 | 部署没有 CFD 接口可对应；需在中心线流程内以左右/内外几何规则命名，并与 V5 语义一致 |
| 视图坐标架手性 | 已记录 | 保持解剖左轴 +X；如需与旧 stl_landmarks_v4 对齐可整体镜像 |
| WANG_KUI_WU 髂总语义跨侧不一致 | 待修（单例刷新 pc-v3.1） | 网格 zone 命名不按解剖左右；`_semantics` 应置 −1 并打标，不再默认判"左" |

## 6. 变更日志

- 2026-09-06：pc-v3 刷新，8 例出口命名按几何重对应，172/172 过 gate（[V5 数据重建记录 §6.4–6.5](../_archive/WSS_PINN/_archive/WSS_V5_数据重建_Pilot与全量构建记录_2026-09-06.md)）。
- 2026-09-07：`training_wss_min` 视图 v1.1 增加对齐坐标架 PCA 法向；本文创建，V2 记录冻结。
