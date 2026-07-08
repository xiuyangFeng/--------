# WSS 最小化路线代码修改与实验推进记录

> 用途：单独记录 `pipeline_wss_min/` 这条 WSS-only 最小化数据线的代码、坐标 QA、图件和实验推进。
> V3P / 训练主线 / 通用代码修改记录见：[代码修改与实验推进记录](代码修改与实验推进记录.md)。
> 上位文档：[WSS 最小化预处理交接记录](WSS最小化预处理流程_搭建与交接记录_2026-07-07.md) / [pipeline_wss_min README](../../pipeline_wss_min/README.md)。

## 2026-07-08｜P0/P1：LR 复核口径统一 + 主轴壁面 PCA 兜底 ✅

**本次主要修改**：
- **P0：复核图与正式配准口径统一。** `visualize_lr_check.py` 不再自己用单侧 `z>阈值` 选下游点，而是复用正式配准里的 `_select_wall_branch_side()`：对分叉两侧分别算横断面双峰分离度，取更像双髂支的一侧。这样消除了 `fast/LI_SHI_QIANG` 这类“配准正确、复核图全蓝”的诊断假象。
- **P0：QA 默认套用逐病例 override。** `visualize_lr_check.py`、`coord_check.py`、`visualize_alignment.py`、`visualize_stl_point_overlap.py` 均改为走 `registration_for_case()`，保证 QA 图看到的是正式 preprocess 会使用的配准配置；只有显式传 `--roll-source/--roll-sign-mode` 做旧口径对照时才关闭 override。
- **P1：主轴增加壁面 PCA 鲁棒兜底。** 默认仍用中心线入口端→分叉点 trunk 弦向；若该弦向无法清楚区分“单主干侧/双髂支侧”（两侧双峰分离度差过小），再用壁面点云 PCA 长轴，并按双髂支侧定号，使主干/分叉重新落到统一 +Z 视角。
- **P1：审计字段补全。** `RigidTransform` / `bundle.npz` / `report.json` / 批量审计新增 `main_axis_source` 与 `main_axis_wall_sep_delta`，后续能直接看到某例是 `centerline_chord`、`wall_pca_fallback` 还是 `centerline_chord_ambiguous`。

**诊断结论**：
- `fast/FAN_JIAN_MING`、`fast/LI_ZHEN_SHAN` 的中心线 trunk 弦向与壁面主方向夹角不大，属于血管自身弯曲/局部形态造成的视觉差异，不是主轴估计失败；本次保持 `centerline_chord`，避免过度旋转。
- `slow/NIE_QUAN_ZHONG` 是真实主轴问题：中心线只有局部 trunk/branch 信息，入口端→分叉弦向不能稳定代表主干方向；开启壁面 PCA 兜底后，从斜放恢复为主干近似竖直。
- 全 split 轻量诊断中，主轴来源为 `centerline_chord=76`、`wall_pca_fallback=4`、`centerline_chord_ambiguous=1`。fallback 病例进入审计清单，便于后续人工复核。
- 全量 81 例 LR QA 重跑后，待人工确认从 9 例降至 **8 例**；其中 `slow/CHENG_GUANG_SEN` 因主轴兜底后 trunk-bending 置信提高，已从 unreliable 中移出。剩余 8 例主要是 `roll_sign` 符号锚置信弱，红蓝两侧本身大多已清楚分开，属于 P2 是否加入个案 override 的范围。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/registration.py`、`config.py`、`preprocess.py`、`reporting.py`、`visualize_lr_check.py`、`coord_check.py`、`visualize_alignment.py`、`visualize_stl_point_overlap.py`
- 主轴前后对照图：`outputs/wss_min/main_axis_p1_compare.png`
- 新 81 例对齐图：`outputs/wss_min/alignment_viz_p0p1_main_axis/`
- 新全量 LR 复核图：`outputs/wss_min/lr_check/lr_side_grid_p0p1_all81_full.png`
- 剩余待确认 9→8 例复核图：`outputs/wss_min/lr_check/lr_side_grid_p0_remaining9.png`

**验证记录**：
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m compileall pipeline_wss_min`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_alignment --split split_AG_wss_min_v1 --tag p0p1_main_axis`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_lr_check --split split_AG_wss_min_v1 --n 81 --tag p0p1_all81_full`

**当前状态判断**：
- P0/P1 核心修复已完成：QA 图与正式处理逻辑一致，中心线主轴异常病例有可审计 fallback，且不会把普通弯曲病例强行拉直。
- 正式训练前仍需按新配准口径全量重跑 `preprocess -> global-stats -> build-samples`，否则旧 `bundle.npz` 仍可能混用旧坐标。
- 后续 P2 建议只处理剩余 8 例 `roll_sign_unreliable`：逐例决定是否接受 `world_axis_weakbend`，或加入 `REGISTRATION_CASE_OVERRIDES`。

## 2026-07-08｜左右轴 wall_branches 改造 + 13 例复核图诊断（待 P0/P1 修复）🔍

> 本条记录 2026-07-08 复核发现与待修项，代码收尾在该记录时尚未完成。

**背景（2026-07-08 已改）**：左右轴配准从「中心线分支 PCA + 世界轴定号」整体改为
「分叉下游**壁面点**双峰求真实 L-R + 主干弯曲(A-P)×主轴内在手性 + 世界轴两锚交叉校验」，
逐级回退 `wall_branches → branches → curvature`；并对四个 QA 脚本加 `--tag` 出到 `_new` 目录（旧图保留）。

**复核发现（诊断）**：
1. 左右轴纯度（两髂支沿 X 轴分离度）全 81 例 **≥0.97**（旧法最低 0.56），朝向抖动/左右翻转基本消除；
   **80/81 走 `wall_branches`**，`slow/WANG_BAO_SHAN` 因分叉原点落在血管轴向端点回退 `branches`（仍出正常 Y）。
2. `outputs/wss_min/lr_check/lr_unreliable_13cases_20260708.png` 复核图**结构正确**，但
   **`fast/LI_SHI_QIANG` 的「全蓝」是该图下游选取的假象**：该图用**单侧** `z>阈值`，恰好抓到只含一条髂支的
   一侧（1472 点全 x<0）；而配准里的 `_wall_branch_lr_axis` 是**两侧自适应**，选中的另一侧（8639 点）
   正常双峰（左 39% / 右 61%）→ **LI_SHI_QIANG 的左右轴其实是对的**。
3. 待复核 13 例中 **12 例可见清楚蓝左/红右**（轴对，仅主干弯曲太弱无法自证符号）；
   真正「长得歪」的是 `fast/FAN_JIAN_MING`、`fast/LI_ZHEN_SHAN`、`slow/NIE_QUAN_ZHONG`——
   属 **主轴倾斜**（主干方向估计问题），与左右轴无关。
4. 已加 4 例 override（`FAN_JIAN_MING`/`LI_SHI_QIANG`/`LI_ZHEN_SHAN`/`ZHANG_HAO` → `world_axis`），
   **待复核由 13 降至 9**：`ZHANG_XIU_WEN`、`YIN_YU_RONG`、`NIE_QUAN_ZHONG`、`ZANG_YU_SHU`、
   `LI_CHONG_ZENG`、`CHENG_GUANG_SEN`、`XU_YI_CAI`、`CHEN_SHI_MING`、`QIN_SI_FU`。

**待修复 / 待决**：
1. 把 13 例复核图的**下游选取改成与配准一致的两侧自适应**（消除 LI_SHI_QIANG 假象）；
   或直接用已是两侧逻辑的 `visualize_lr_check.py` 按当前 9 例重生成。
2. 剩余 9 例：目视基本是正常 Y，定夺是「接受 `world_axis` 回退」还是逐个加入 `REGISTRATION_CASE_OVERRIDES`。
3. 主轴倾斜的 3 例（`FAN_JIAN_MING`/`LI_ZHEN_SHAN`/`NIE_QUAN_ZHONG`）属 `main_axis` 估计问题，单独一轮处理。
4. 配准口径定稿后，对整个 split 重跑 `preprocess`（现有 bundle 仍是旧坐标），再跑 `global-stats` / `build-samples`。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/registration.py`（`_wall_branch_lr_axis` 两侧自适应 + `_trunk_bending_*` + `_sign_lr_axis` 两锚交叉校验）、
  `config.py`（`roll_source`/`roll_sign_mode`/`downstream_wall_axis_frac`/`roll_sign_min_cos`/`REGISTRATION_CASE_OVERRIDES`）、
  `preprocess.py`、`reporting.py`、`visualize_lr_check.py`（新增）、
  `visualize_alignment.py`/`coord_check.py`/`visualize_stl_point_overlap.py`（加 `--tag`；小图 ⚠ 改为「回退/待复核才告警」）
- 图件：`outputs/wss_min/lr_check/`、`alignment_viz_new/`、`coord_check_new/`、`stl_point_overlap_new/`
- 审计：`data_wss_min/pipeline_reports/preprocess_audit_latest.json`（`roll_sign_unreliable_cases`）

**当前状态判断**：
- 左右轴主问题（朝向不一致 / 偶尔翻转）已在配准层修复并有量化证据（纯度 ≥0.97、`det=+1`、可复现）。
- 剩余为三件收尾：复核图取样口径统一、9 例符号确认、3 例主轴倾斜；均不阻塞「现算配准看新朝向」。

## 2026-07-08｜4 例 roll 符号 override + 重跑 preprocess ✅

**本次主要修改**：
- 新增 `config.REGISTRATION_CASE_OVERRIDES` 与 `registration_for_case()`：对 trunk_bending 与世界轴冲突、但 `wall_branches` 左右轴仍可信的个案，显式锁定 `roll_sign_mode="world_axis"`（几何与 weakbend 回退一致，仅消除 unreliable 标记）。
- 已处理 4 例：`fast/FAN_JIAN_MING`、`fast/LI_SHI_QIANG`、`fast/LI_ZHEN_SHAN`、`fast/ZHANG_HAO`；重跑 preprocess 并刷新审计。

**对应代码/文档**：
- 代码：`pipeline_wss_min/config.py`、`preprocess.py`、`README.md`
- 审计：`data_wss_min/pipeline_reports/preprocess_audit_latest.json`（`roll_sign_unreliable` 13→9）

**推进到实验步骤**：
- split 内 4 例已纳入正式 override 清单；审计待复核由 13 例降至 9 例。

**当前状态判断**：
- 4 例 `roll_sign_reliable=True`、`roll_sign_source=world_axis`，bundle 旋转矩阵与 override 前 weakbend 结果逐元素一致。
- 剩余 9 例仍待同类处理或人工 LR 复核（含 val `XU_YI_CAI`、test `CHEN_SHI_MING` 等）。

## 2026-07-07｜随机 10 例点云/STL 同框叠加图 + 推进记录拆分 ✅

**本次主要修改**：
- `pipeline_wss_min.visualize_stl_point_overlap` 新增两张“同框叠加”图，不再只做逐病例小格子：
  - 10 例壁面点云放在同一张图、同一坐标范围、同一 3D 相机视角下叠加；
  - 同一批 10 例 STL 顶点/面片也放在同一张图中叠加。
- 两张同框图都保留原点参考线和 X-Z / Y-Z / X-Y 三投影，用来直接检查分叉原点是否对齐、主干/分叉朝向是否一致。
- 新建本文档，将 WSS-only 最小化路线的详细推进记录从 V3P 大日志中独立出来；原 `代码修改与实验推进记录.md` 后续保留 V3P / 训练主线记录和拆分说明。

**对应代码/文档**：
- 代码：`pipeline_wss_min/visualize_stl_point_overlap.py`
- 文档：`README.md`、`pipeline_wss_min/README.md`、`docs/README.md`、`docs/02-推进与变更/WSS最小化预处理流程_搭建与交接记录_2026-07-07.md`、`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`、`docs/02-推进与变更/代码修改与实验推进记录.md`
- 图件：
  - `outputs/wss_min/stl_point_overlap_20260707/04_random10_same_frame_pointcloud_overlay.png`
  - `outputs/wss_min/stl_point_overlap_20260707/05_random10_same_frame_stl_overlay.png`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的几何输入 QA：从“逐病例检查 STL/点云是否贴合”进一步推进到“多病例放入同一解剖坐标框架后是否原点对准、视角一致”。

**当前状态判断**：
- 同一随机种子 `20260707` 抽取的 10 例中，壁面点云同框叠加与 STL 同框叠加均显示：分叉附近围绕统一原点，主干整体位于 `z<0`，分叉/出口位于 `z≈0` 到正值区域。
- STL 版本与点云版本视觉结论一致，支持当前分叉原点配准和逐病例归一化可作为 `x,y,z -> wss` 简化输入的前置规整步骤。

## 2026-07-07｜随机 10 例 STL-点云逐病例重叠 QA 图 ✅

**本次主要修改**：
- 新增 `pipeline_wss_min.visualize_stl_point_overlap`：从 `split_AG_wss_min_v1` 随机抽取 10 个可匹配 STL 的病例，读取壁面 ascii 点云与 `stl_data/` 中对应 STL。
- 图件生成时不做 ICP 或额外拟合，只把 STL 按 ascii/STL 原始单位比例换到 pipeline 坐标尺度，再与壁面点云套用同一个单位换算、分叉原点刚性配准和逐病例归一化尺度。
- 输出 3D 叠加图、固定坐标范围三投影图、局部放大三投影图与 CSV 距离审计，用于确认后续 `x,y,z -> wss` 的几何输入确实处于同一处理框架。

**对应代码/文档**：
- 代码：`pipeline_wss_min/visualize_stl_point_overlap.py`
- 文档：`pipeline_wss_min/README.md`
- 图件：`outputs/wss_min/stl_point_overlap_20260707/01_random10_stl_point_overlap_3d.png`、`02_random10_stl_point_overlap_projections.png`、`03_random10_stl_point_overlap_zoomed_projections.png`
- 审计：`outputs/wss_min/stl_point_overlap_20260707/random10_stl_point_overlap_summary.csv`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的几何 QA：在解剖原点坐标系之外，补充 STL 面片与壁面训练点云的一致性检查，为只输入 `x,y,z` 的壁面 WSS 预测提供坐标规整证据。

**当前状态判断**：
- `split_AG_wss_min_v1` 中 80 例可匹配 STL，1 例未匹配到 STL；随机种子 `20260707` 抽取的 10 例全部成功出图。
- 10 例 `native_to_stl≈1000`，说明 STL 基本是 ascii 原生坐标的毫米版本；套用 pipeline 坐标尺度后，STL 到壁面点云的 p95 NN 距离约 `1.19e-05` 到 `9.66e-05 mm`，视觉上三投影完全贴合。
- 当前证据支持：壁面点云与 STL 几何源一致，且经新版分叉原点配准后可放入同一个坐标框架继续做 `x,y,z -> wss`。

## 2026-07-07｜解剖原点版配准 v2 + 全量坐标 QA 图 ✅

**本次主要修改**：
- `pipeline_wss_min` 的坐标配准从“壁面重心原点 + 中心线主轴 + 曲率 roll”升级为**分叉原点版解剖刚性配准 v2**：默认读取 `centerline.vtp` 中的 `DistToBifurcation` / `BranchId`，以分叉区域为原点，用入口端→分叉 trunk 方向对齐 +Z，用分叉后远端分支平面固定横轴；缺 VTP 拓扑时保留旧策略兜底。
- `bundle.npz`、单病例 `report.json`、批量审计 CSV 新增 `origin_kind` / `main_axis_mode` / `roll_source` 追踪字段，避免 fallback 病例静默混入。
- 重新生成 WSS-min 坐标 QA 图：`outputs/wss_min/alignment_viz_anatomical_v2/` 与 `outputs/wss_min/coord_check_20260707_anatomical_v2/`；81 例 included 全部 `origin=bifurcation`、`main=inlet_to_bifurcation`、`roll=branches`，无单位异常，`det(R)=1`，最大正交误差约 `3e-16`。
- 更新 `pipeline_wss_min` README、WSS-min 交接记录、根 README 与 docs 索引，说明新版配准口径、图件位置与旧 bundle 需重跑 preprocess 的注意事项。

**对应代码/文档**：
- 代码：`pipeline_wss_min/config.py`、`raw_io.py`、`registration.py`、`preprocess.py`、`reporting.py`、`visualize_alignment.py`、`coord_check.py`
- 文档：`pipeline_wss_min/README.md`、`docs/02-推进与变更/WSS最小化预处理流程_搭建与交接记录_2026-07-07.md`、`README.md`、`docs/README.md`
- 图件：`outputs/wss_min/alignment_viz_anatomical_v2/*.png`、`outputs/wss_min/coord_check_20260707_anatomical_v2/*.png`

**推进到实验步骤**：
- 推进到任务 A 的 WSS-only 最小化数据线：完成 `x,y,z -> wss` 前置的解剖坐标统一与全量 QA，可作为下一步重跑 `preprocess -> global-stats -> build-samples` 和壁面点数扫描的输入标准。

**当前状态判断**：
- 新版坐标系比旧壁面重心原点更符合“同一视角 / 同一框架 / 基本朝向一致”：分叉原点固定，主干统一位于 `z<0`，分叉/出口统一位于 `z≈0` 到正值区域。
- 注意旧 `data_wss_min/*/bundle.npz` 若在本次前生成，仍是旧坐标，正式训练前必须全量重跑 `pipeline_wss_min.run --stage preprocess`。
