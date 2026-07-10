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
| 解剖配准 | **分叉原点版中心线/壁面联合刚性配准 v3.2**：`DistToBifurcation≈0` 分叉区域→原点，入口端→分叉 trunk 主轴对齐 +Z；若中心线 trunk 弦向无法清楚区分单主干侧/双髂支侧，则用壁面点云 PCA 长轴兜底并按双髂支侧定号；**左右轴(roll)从分叉下游"壁面点"双峰求真实 L-R**（中心线常只描一条髂支），符号用"主干弯曲(A-P)×主轴"内在手性 + 世界轴两锚交叉校验；配准后若主干低位段在横断面仍整体偏离中心轴，则触发主干横向二次居中；缺数据按 `wall_branches→branches→curvature` 逐级回退，`det(R)=+1` |
| 坐标标准化 | **逐病例**各向同性缩放到 [-1,1]（保形，视角一致） |
| WSS 标准化 | **全局** `log_z`（近似对数正态）；第三轮 clean-data 主线默认使用 train-only peak 单步统计 |
| 数据范围 | `AG/fast` (25) + `AG/slow` (62) |
| 权威 split | `split_AG_wss_min_v1`：train 53 / val 8 / test 16 / excluded 9 / pending 1；默认流程只用 split included，WSS 全局统计只用 train |
| 单位 | **逐病例自动**从中心线包围盒反推 mesh→mm 因子（正常≈1000）；异常病例标记待复核 |

> 2026-07-07 更新：旧版 `center_on="wall"` 曾升级为 `center_on="bifurcation"`。2026-07-08 最终口径已继续升级为
> `center_on="flow_divider"`。如果 `data_wss_min/*/bundle.npz` 是旧版生成的，需重跑 `preprocess`
> 才会写入新的 `wall_coords_norm`、`transform_origin_kind`、`transform_main_axis_mode` 与 `transform_roll_source`。
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
>
> 2026-07-08 P3 更新：`slow/NIE_QUAN_ZHONG` 因中心线/STL/点云 QA 异常从 train 移入 excluded。
> `run.py`、`global_stats.py`、`build_samples.py` 默认遵守 `split_AG_wss_min_v1`；第三轮起 `--all-raw` 仅允许诊断性
> `preprocess`，正式 `qa-gate/global-stats/build-samples/all` 均拒绝 all-raw，避免 pending/excluded 历史 bundle 进入口径。
>
> 2026-07-08 最终 QA 收口：未描入口段裁剪改为入口中心线局部切平面口径，并加
> `crop_min_wall_frac=0.08` 防止正常病例擦边小裁剪；坐标缩放默认 `coord_scale_on="wall"`。
> 人工复核后将 `slow/WANG_BAO_SHAN`、`slow/SUN_WEN_QING` 从 train 移入 excluded；当时最终 QA 为 included=78，
> 只裁剪 `slow/ZHANG_HUAN_LI`（32.40%），`fast/LI_ZHEN_SHAN` 不再被擦边裁剪。该阶段图件入口为
> `outputs/wss_min/当前_78例坐标QA/`；后续 flow-divider 最终口径见下一条。
>
> 2026-07-08 flow-divider 最终原点：默认 `center_on="flow_divider"`，把
> `DistToBifurcation≈0` 附近中心线点聚成三臂（主干端 + 左右髂支起始端），对三臂中心等权平均，
> 对应“左右髂支接入主动脉的三叉连接点”。78 例对照中 |COM| 改善/变差为 61/16，
> mean |COM| 从 0.2140 降到 0.2029；最终 QA 图在
> `outputs/wss_min/flow_divider_origin_QA_78例/`。旧 `当前_78例坐标QA/` 已归档，不再作为当前口径。
>
> 2026-07-08 21:48 已在集群 `node03` 重跑 `preprocess`，Slurm 作业 `5941` 完成，退出码 `0:0`。
> 当时 78 个 included bundle 已按 flow-divider 口径刷新；审计为
> `data_wss_min/pipeline_reports/preprocess_audit_20260708_215512.csv`（`ok=78, skipped=0, error=0`）。
> `pending=1` 的 `slow/ZHAO_XIU_XUAN` 是原始目录中存在但未分配到继承 split 的病例，第三轮仍不进入 included=77 口径。
>
> 2026-07-10 第三轮 clean-data 口径：`slow/ZHANG_HUAN_LI` 因 WSS 标签近全零、
> `n_wall` 极端且多重预处理审计异常，从 train 移入 excluded；当前 included=77。
> 后续必须重跑 included bundle，使 `nodenumber` 对齐守卫和新审计字段真实落盘。
> `excluded_cases` 和 `pending_cases` 即使磁盘上仍有历史 `bundle.npz`，也不得参与
> preprocess/stats/training/eval 的正式数据集。

## 四个阶段
1. **preprocess**（默认 split included 病例，重，一次）：读原始 CFD → 中心线配准 → 坐标正交化/标准化 →
   壁面/近壁/内部掩码（KNN 到壁面距离）→ 保留几何 → 堆叠全时间步 WSS → `bundle.npz`
2. **qa-gate**（第三轮新增，默认 split included）：检查 WSS 零值、极端点数、单位/覆盖/裁剪/主干偏移、非有限值与 `nodenumber` 对齐字段；失败则不进入训练
3. **global-stats**（第二遍，默认 train-only + peak-only）：流式累积壁面 WSS → `data_wss_min/wss_global_stats.json`
4. **build-samples**（默认 split included，轻，可反复扫）：**归一化之后**稀疏化（FPS/random）→
   选时间步（peak/all）→ 全局标准化 WSS → 每样本 `.npz`

## 用法
```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 单病例冒烟
$PY -m pipeline_wss_min.run --stage preprocess --cohort AG/fast --case CHEN_SHI_MING

# 全流程（默认按 split_AG_wss_min_v1；train-only stats，included samples）
$PY -m pipeline_wss_min.run --stage all
# 第三轮 clean-data 推荐分步：preprocess 完成后先过 QA，再重算 peak-only stats
$PY -m pipeline_wss_min.run --stage preprocess
$PY -m pipeline_wss_min.run --stage qa-gate
$PY -m pipeline_wss_min.run --stage global-stats --stats-timesteps peak
# 原始目录排查才用：忽略 split，扫描全部 raw 病例；不得接 stats/build-samples
$PY -m pipeline_wss_min.run --stage preprocess --all-raw

# 集群提交（CPU/node03；脚本名沿用 preprocess，但可传 stage）
/public/slurm/bin/sbatch --parsable pipeline_wss_min/cluster/run_preprocess.slurm preprocess
/public/slurm/bin/sbatch --parsable pipeline_wss_min/cluster/run_preprocess.slurm qa-gate
/public/slurm/bin/sbatch --parsable pipeline_wss_min/cluster/run_preprocess.slurm global-stats
/public/slurm/bin/sbatch --parsable pipeline_wss_min/cluster/run_preprocess.slurm build-samples

# 只重扫点数（上游不重跑）
$PY -m pipeline_wss_min.run --stage build-samples --wall-n 1500 --sample-name wss_min_peak_w1500
$PY -m pipeline_wss_min.run --stage build-samples --wall-n 3000 --sample-name wss_min_peak_w3000

# 切全相位
$PY -m pipeline_wss_min.run --stage build-samples --timesteps all --sample-name wss_min_allphase_w2000

# 只做坐标 QA/出图（不写 bundle）
$PY -m pipeline_wss_min.visualize_alignment --split split_AG_wss_min_v1
$PY -m pipeline_wss_min.coord_check --split split_AG_wss_min_v1
$PY -m pipeline_wss_min.visualize_stl_point_overlap --split split_AG_wss_min_v1 --n 10 --seed 20260707
# flow-divider QA：点云/STL 同框每20例 + 逐病例视图 + WSS正/俯视图
$PY -m pipeline_wss_min.visualize_final_case_pages --split split_AG_wss_min_v1
# flow-divider 解剖基点新旧原点量化对照
$PY -m pipeline_wss_min.compare_flow_divider_origin --split split_AG_wss_min_v1
$PY -m pipeline_wss_min.visualize_final_case_pages --split split_AG_wss_min_v1 \
  --out-dir outputs/wss_min/flow_divider_origin_QA_78例 --center-on flow_divider
# 指定病例做 STL/点云对比
$PY -m pipeline_wss_min.visualize_stl_point_overlap --split split_AG_wss_min_v1 \
  --cases fast/ZHANG_XIU_WEN,fast/CHEN_SHI_MING --tag p3_remaining2

# 左右髂支朝向一致性 QA（下游髂支段按 L-R 坐标上色；⚠标记两锚冲突的待复核病例）
$PY -m pipeline_wss_min.visualize_lr_check --split split_AG_wss_min_v1 --n 20 --tag new
# 旧配准口径对照（一键回退）
$PY -m pipeline_wss_min.visualize_lr_check --roll-source branches --roll-sign-mode world_axis --tag old
```

## 配置入口
`pipeline_wss_min/config.py`：
- `RegistrationConfig` — 配准（解剖原点来源、目标主轴、左右轴/滚转锚定）
  - `center_on`: `flow_divider`(当前默认：三臂等权三叉连接点) | `bifurcation`(旧对照) | `wall` | `all` | `centerline`
  - `roll_source`: `wall_branches`(新默认，壁面双峰求真左右轴) | `branches`(旧) | `curvature`(旧兜底)
  - `roll_sign_mode`: `trunk_bending`(新默认，主干弯曲内在手性) | `world_axis`(旧，可回退)
  - `main_axis_wall_fallback`: 默认开启；当中心线主轴无法清楚区分单主干侧/双髂支侧时，用壁面 PCA 长轴兜底
  - `main_axis_wall_fallback_sep_delta`: 触发兜底的分支侧分离度差阈值（默认 0.10，越小越保守）
  - `main_axis_wall_fallback_min_sep`: 壁面 PCA 兜底必须达到的双髂支分离度下限（默认 0.45）
  - `trunk_centering`: 默认开启；配准后检测低位 20% 主干壁面点横向中心，若 normalized offset ≥0.08，
    只做非主轴两个方向的刚性平移，写入 `transform_trunk_centering_*` 审计字段
  - **一键回退旧配准**：`roll_source="branches", roll_sign_mode="world_axis"`
  - `roll_sign_min_cos`: 两锚符号交叉校验的置信阈值（默认 0.2；冲突且弱→标记 `roll_sign_reliable=False`）
  - **逐病例 override**：`config.py` 的 `REGISTRATION_CASE_OVERRIDES`（键 `fast/XXX` 或 `slow/XXX`），用于 trunk/world 冲突但 wall_branches 轴仍可信的个案；2026-07-08 已确认 9 例
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
  - 主干横向二次居中追踪：`transform_trunk_centering_applied`、
    `transform_trunk_centering_offset_mm`、`transform_trunk_centering_offset_frac`
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
- **2026-07-08 的 78 例 flow-divider QA 图**：`outputs/wss_min/flow_divider_origin_QA_78例/`，包含点云同框、STL 同框、逐病例 X-Z、
  峰值 WSS 正/俯视图、LR 复核图和二次居中病例复核图；每类分页图按每 20 例一页组织。
- **flow-divider 原点对照**：`outputs/wss_min/flow_divider_origin_compare/`，包含新旧原点位移、
  |COM| 改善/变差、最大变化病例和汇总图；最终 QA 图为
  `outputs/wss_min/flow_divider_origin_QA_78例/`。
- **旧口径图件归档**：`outputs/wss_min/归档_旧口径_20260708/`，保留早期 81 例、P0/P1/P2 和随机 10 例图，避免与当前口径混淆。
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
- **几何/配准/标签 QA 剔除病例**：第三轮 `excluded_cases=9`（`fast/PENG_JI_MING`、`slow/LIU_XI_QUAN`、`slow/WEI_JUN_WEN`、
  `slow/LIN_SHU_TIAN`、`slow/TE_JIN_WANG`、`slow/NIE_QUAN_ZHONG`、`slow/WANG_BAO_SHAN`、`slow/SUN_WEN_QING`、
  `slow/ZHANG_HUAN_LI`）；
  旧 bundle 若存在也不会被默认 `preprocess` / `global-stats` / `build-samples` 使用。
- **pending 病例**：`slow/ZHAO_XIU_XUAN` 存在于原始目录，但未分配到继承 split 的 train/val/test/excluded 任一名单，
  因此暂不纳入第三轮 included=77；需单独 QA 后再决定是否补入。

## 注意
- 第三轮 clean-data 的全局 WSS 统计覆盖**train peak 单步**。若后续切全相位任务，需显式用
  `--stats-timesteps all` 重算 stats，并给网络增加 `phase` / `inlet_flow` 等时间或边界条件特征。
- `curvature` 端点附近有数值尖峰（原始中心线量），默认被掩码；如启用建议先 clip/log。
