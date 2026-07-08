# pipeline_wss_min — 最小化 WSS-only 预处理流程

与旧 `pipeline/` / `data_new/` 产物**完全独立**，不改动任何既有实验。
产物落在 `data_wss_min/`。

## 设计要点（老师口径）
- 只做一个目标：几何 → **WSS**，输入默认 `x,y,z`，输出 `wss`，**单头**。
- 点数从 13000+2000 减到可配置（壁面 1500/2000/3000 可扫）。
- **先配准 / 正交化 / 标准化，最后才稀疏化**。
- **保留**全部几何特征（dist_to_wall、abscissa、local_radius、curvature、WSS 矢量分量…），
  默认用 `input_feature_names` 掩码屏蔽，后续改配置即可解锁。
- 时间步**一次性全做**（81 步）；样本装配默认取**峰值收缩期**单样本，可切 `all` 全相位。

## 关键决策（已与用户确认）
| 项 | 选择 |
|---|---|
| 解剖配准 | **分叉原点版中心线/壁面联合刚性配准 v3.1**：`DistToBifurcation≈0` 分叉区域→原点，入口端→分叉 trunk 主轴对齐 +Z；若中心线 trunk 弦向无法清楚区分单主干侧/双髂支侧，则用壁面点云 PCA 长轴兜底并按双髂支侧定号；**左右轴(roll)从分叉下游"壁面点"双峰求真实 L-R**（中心线常只描一条髂支），符号用"主干弯曲(A-P)×主轴"内在手性 + 世界轴两锚交叉校验，冲突时记审计待复核；缺数据按 `wall_branches→branches→curvature` 逐级回退，`det(R)=+1` |
| 坐标标准化 | **逐病例**各向同性缩放到 [-1,1]（保形，视角一致） |
| WSS 标准化 | **全局** `log_z`（近似对数正态），覆盖全部时间步 |
| 数据范围 | `AG/fast` (25) + `AG/slow` (62) |
| 单位 | **逐病例自动**从中心线包围盒反推 mesh→mm 因子（正常≈1000）；异常病例标记待复核 |

> 2026-07-07 更新：旧版 `center_on="wall"` 已升级为默认 `center_on="bifurcation"`。如果 `data_wss_min/*/bundle.npz`
> 是旧版生成的，需重跑 `preprocess` 才会写入新的 `wall_coords_norm`、`transform_origin_kind`、
> `transform_main_axis_mode` 与 `transform_roll_source`。
>
> 2026-07-08 更新：左右轴(roll)默认改为 `roll_source="wall_branches"` + `roll_sign_mode="trunk_bending"`，
> 修复"每图朝向略有差异、左右髂支偶尔翻转"（旧法从只含一条髂支的中心线求 roll 轴 + 世界轴定号，既不稳又易翻）。
> **改动会改变配准结果**：需对整个 split 重跑 `preprocess`（`--stage preprocess` 或 `all`）以获得一致坐标框架，
> 混用新旧 bundle 会导致朝向不一致。先用 `visualize_lr_check` 核对，并检查审计里的
> `roll_sign_unreliable_cases`。一键回退旧配准：`roll_source="branches", roll_sign_mode="world_axis"`。
>
> 2026-07-08 P0/P1 更新：LR 复核图现在与正式配准共用同一个“分叉两侧自适应选择”函数，
> 不再出现复核图单侧取样导致的假阳性；主轴新增 `main_axis_wall_fallback`，专门处理中心线端点缺主干或落在分支、
> 造成入口端→分叉弦向不可信的病例。审计中新增 `main_axis_source` 与 `main_axis_wall_sep_delta`。

## 三个阶段
1. **preprocess**（每病例，重，一次）：读原始 CFD → 中心线配准 → 坐标正交化/标准化 →
   壁面/近壁/内部掩码（KNN 到壁面距离）→ 保留几何 → 堆叠全时间步 WSS → `bundle.npz`
2. **global-stats**（第二遍）：全病例流式累积壁面 WSS → `data_wss_min/wss_global_stats.json`
3. **build-samples**（配置驱动，轻，可反复扫）：**归一化之后**稀疏化（FPS/random）→
   选时间步（peak/all）→ 全局标准化 WSS → 每样本 `.npz`

## 用法
```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 单病例冒烟
$PY -m pipeline_wss_min.run --stage preprocess --cohort AG/fast --case CHEN_SHI_MING

# 全流程（AG fast+slow）
$PY -m pipeline_wss_min.run --stage all

# 只重扫点数（上游不重跑）
$PY -m pipeline_wss_min.run --stage build-samples --wall-n 1500 --sample-name wss_min_peak_w1500
$PY -m pipeline_wss_min.run --stage build-samples --wall-n 3000 --sample-name wss_min_peak_w3000

# 切全相位
$PY -m pipeline_wss_min.run --stage build-samples --timesteps all --sample-name wss_min_allphase_w2000

# 只做坐标 QA/出图（不写 bundle）
$PY -m pipeline_wss_min.visualize_alignment --split split_AG_wss_min_v1
$PY -m pipeline_wss_min.coord_check --split split_AG_wss_min_v1
$PY -m pipeline_wss_min.visualize_stl_point_overlap --split split_AG_wss_min_v1 --n 10 --seed 20260707

# 左右髂支朝向一致性 QA（下游髂支段按 L-R 坐标上色；⚠标记两锚冲突的待复核病例）
$PY -m pipeline_wss_min.visualize_lr_check --split split_AG_wss_min_v1 --n 20 --tag new
# 旧配准口径对照（一键回退）
$PY -m pipeline_wss_min.visualize_lr_check --roll-source branches --roll-sign-mode world_axis --tag old
```

## 配置入口
`pipeline_wss_min/config.py`：
- `RegistrationConfig` — 配准（解剖原点来源、目标主轴、左右轴/滚转锚定）
  - `roll_source`: `wall_branches`(新默认，壁面双峰求真左右轴) | `branches`(旧) | `curvature`(旧兜底)
  - `roll_sign_mode`: `trunk_bending`(新默认，主干弯曲内在手性) | `world_axis`(旧，可回退)
  - `main_axis_wall_fallback`: 默认开启；当中心线主轴无法清楚区分单主干侧/双髂支侧时，用壁面 PCA 长轴兜底
  - `main_axis_wall_fallback_sep_delta`: 触发兜底的分支侧分离度差阈值（默认 0.10，越小越保守）
  - `main_axis_wall_fallback_min_sep`: 壁面 PCA 兜底必须达到的双髂支分离度下限（默认 0.45）
  - **一键回退旧配准**：`roll_source="branches", roll_sign_mode="world_axis"`
  - `roll_sign_min_cos`: 两锚符号交叉校验的置信阈值（默认 0.2；冲突且弱→标记 `roll_sign_reliable=False`）
  - **逐病例 override**：`config.py` 的 `REGISTRATION_CASE_OVERRIDES`（键 `fast/XXX` 或 `slow/XXX`），用于 trunk/world 冲突但 wall_branches 轴仍可信的个案
- `NormalizationConfig` — 坐标/ WSS 标准化范围与方法
- `PointTaggingConfig` — 近壁阈值/上限、存储控制（矢量/内部/近壁时间序列）
- `TimestepConfig` — 时间步区间、峰值检测
- `SampleConfig` — 稀疏点数、时间步、输入特征、目标（**扫实验主要改这里**）

## bundle.npz 内容
静态几何一次算好；场值按时间步堆叠。
- 元信息：`steps, peak_step, coord_scale, transform_*`
  - 新版配准追踪字段：`transform_origin_kind, transform_main_axis_mode, transform_roll_source`
  - 主轴鲁棒性追踪：`transform_main_axis_source`（`centerline_chord`/`wall_pca_fallback`/`centerline_chord_ambiguous`）、
    `transform_main_axis_wall_sep_delta`（中心线主轴下两侧双峰分离度差）
  - 左右轴符号追踪：`transform_roll_sign_source`（`trunk_bending`/`world_axis`/`world_axis_weakbend`/`branches_internal`/`curvature_internal`）、
    `transform_roll_sign_cos`（符号置信 |cos|）、`transform_roll_sign_reliable`（False=两锚冲突，朝向可能翻转，待人工复核）
- 壁面：`wall_coords_norm/raw, wall_dist_to_wall, wall_abscissa_norm, wall_local_radius,
  wall_curvature, wall_wss(T,N), wall_pressure(T,N), wall_wss_vec(T,N,3)`
- 内部（可选）：`int_coords_norm, int_type(0内部/1近壁/2壁面), int_dist_to_wall, int_*geom`
- 近壁时间序列（可选，第二条路径）：`near_wall_idx, near_wall_vel(T,M,3), near_wall_vel_mag`

## 两条路径
- 路径 1（当前默认）：壁面 `x,y,z` → `wss`。
- 路径 2（预留）：`store_near_wall_timeseries=True` 存近壁速度，后续近壁点预测速度→算子→WSS。
  预处理产物已保留所需数据，只需改配置，不用重写。

## 日志与审计（每次运行都会产出）
- **日志文件**：`logs/wss_min_<stage>_<时间戳>.log`（控制台同步输出），逐病例记录点数/峰值步/
  缩放/行列式/单位因子/耗时，以及 SKIP/ERROR/单位异常告警。
- **每病例报告**：`data_wss_min/<cohort>/<case>/report.json`（全部处理参数与统计）。
- **批量审计**：`data_wss_min/pipeline_reports/preprocess_audit_<ts>.{csv,json}` +
  `preprocess_audit_latest.json`，含汇总：`ok / skipped / error / unit_anomaly` 分类清单。
- **坐标 QA 图**：`outputs/wss_min/alignment_viz_anatomical_v2/` 与
  `outputs/wss_min/coord_check_20260707_anatomical_v2/`，用于检查分叉原点、trunk 方向和分支横轴是否一致。
- **P0/P1 QA 图**：`outputs/wss_min/alignment_viz_p0p1_main_axis/` 与
  `outputs/wss_min/lr_check/lr_side_grid_p0p1_all81_full.png`，用于核对主轴 fallback 后的统一视角和全量 LR 符号状态。
- **STL-点云重叠 QA 图**：`outputs/wss_min/stl_point_overlap_20260707/`，随机抽 10 例，将 STL 与壁面点云套用同一个
  单位换算、分叉原点刚性配准和归一化尺度后叠加；不做 ICP 或额外拟合，用于确认 `x,y,z` 输入确实处在同一处理框架。
  - `04_random10_same_frame_pointcloud_overlay.png`：10 例壁面点云放在同一张图、同一视角、同一坐标范围内看原点和朝向。
  - `05_random10_same_frame_stl_overlay.png`：同一批 10 例 STL 顶点/面片放在同一张图中做同口径检查。
- 状态区分：`ok` 成功；`skipped` 原始数据不完整（**非报错**，如缺 wall-shear 列）；`error` 意外失败。

## 已知数据质量问题（全量跑前后重点看审计）
- **3 个病例缺 WSS 导出**（只有 x,y,z,pressure）→ 自动 `skipped`：
  `AG/slow/LIU_XI_QUAN, TE_JIN_WANG, WEI_JUN_WEN`。
- **分隔符不统一**：部分 ascii 逗号分隔、部分空白分隔 → 已按表头自动探测，两种都支持。
- **文件名前缀 ≠ 目录名**（如目录 `HOU_SHEN_QIAN`、文件 `HOU_SHEN_QIAN3-*`）→ 已按 `-<step>` 后缀回退匹配。
- **单位异常病例**（如 `AG/fast/PENG_JI_MING`，原生尺度 ~1e-4，factor≈9.7e5）→ 坐标已按中心线自动校正，
  但其 **CFD 的 WSS 物理量级可能同样异常，纳入全局统计/训练前请人工复核**（审计中 `unit_anomaly=true`）。

## 注意
- 全局 WSS 统计覆盖**全部时间步**，因此峰值样本的归一化均值为正（峰值 WSS 高于全相位均值），符合预期。
  若只训练峰值，可在 `NormalizationConfig` 另建仅峰值统计。
- `curvature` 端点附近有数值尖峰（原始中心线量），默认被掩码；如启用建议先 clip/log。
