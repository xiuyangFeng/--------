# WSS-min 预处理流程

`pipeline_wss_min` 是与旧 `pipeline/`、`data_new/` 产物隔离的 WSS-only 数据线。
默认任务为患者几何 `x,y,z → wss`，正式产物写入 `data_wss_min/`。

## 当前数据口径

- 活动 AG：76例 `stl_landmarks_v4`；旧84例位于 `_snapshots/AG_legacy_v3_20260716_signedoff/`。
- v4 AG 权威划分：`training/splits/split_AG_wss_min_v4_traintest.json`（61/0/15）。
- v4 混合池：`split_AG_AAA_wss_min_v4_trainpool.json`（AG61+AAA57 train，锁定 AG test15）。
- v4 独立混合重划：`split_AG_AAA_wss_min_v4_stratified_seed1234.json`（106/0/27；按 AG/AAA rupture/AAA unrupture 分层，旧 AG test15 不锁定）。
- AG 正式范围：train 53 / val 8 / test 16 / excluded 9 / pending 1。
- 全局 WSS 统计：只读取 train、只取峰值步、使用 `log_z`，写入
  `data_wss_min/wss_global_stats.json`。
- 坐标：逐病例刚性配准后，各向同性缩放到 `[-1, 1]`；FPS 必须在归一化之后执行。
- v4 坐标架：原始 STL 自动识别近端主干、分叉和双髂支；`+Z` 指向近端主干，
  `-Z` 指向髂支，原始 STL 世界 `+X` 固定左右符号，旋转矩阵保持 `det(R)=+1`。
- AAA/ILO 只通过独立双白名单入口处理，不与 AG 扫描或产物混写。
- ILO 当前唯一活动口径为最终人工审核通过的 before41；after 活动产物为0，且 ILO 尚未进入任何 split或训练统计。

## 正式四阶段

1. `preprocess`：原始 CFD 校验、稳定节点对齐、单位换算、STL/中心线配准、裁剪、
   坐标归一化、几何特征和全时间步场堆叠，写出 `bundle.npz` 与 `report.json`。
2. `qa-gate`：检查零值、点数、单位、裁剪、非有限值、节点对齐和坐标一致性。
3. `global-stats`：基于 train bundle 计算峰值 WSS 全局统计。
4. `build-samples`：归一化坐标上执行 FPS/random，装配模型样本。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `config.py` | 路径、split、单位、配准、归一化、存储和采样配置 |
| `raw_io.py` | Fluent ASCII、中心线和入口波形读取 |
| `surface_io.py` | 原始 STL 选择、读取和尺度推断 |
| `registration.py` | v4/历史坐标架、刚性变换和入口裁剪 |
| `preprocess.py` | 单病例正式预处理与 bundle 写出 |
| `qa_gate.py` | AG split 病例硬质量门 |
| `global_stats.py` | train-only WSS 统计与标准化 |
| `build_samples.py` | 归一化后采样与样本装配 |
| `reporting.py` | 运行日志、病例报告和批量审计 |
| `run.py` | AG 四阶段统一命令行入口 |
| `v4_cutover.py` | AG v4 staging/原子发布与回滚、AAA 单病例修正发布、数据/快照清单 |
| `training_quality_v4.py` | AG/AAA v4 训练入口 WSS/波形/压力数值门禁与派生白名单 |
| `visualize_v4_cutover.py` | 直接读取 staging/活动 bundle 生成 AG–AAA 签核图 |
| `new_cohorts/` | AAA/ILO 白名单预处理、几何审计、终检和可视化 |
| `examples/` | 给导师查看的紧凑单文件示例 |
| `archive/alignment_v3/` | 已结束的 AG v2/v3 坐标 QA 历史工具，不是当前入口 |
| `cluster/` | 当前可复用的 Slurm 模板；运行日志不纳入代码库 |
| `tests/` | 配置隔离、FPS、split 和紧凑示例回归测试 |

## AG 运行命令

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 单病例预处理
$PY -m pipeline_wss_min.run \
  --stage preprocess --cohort AG/fast --case CHEN_SHI_MING

# 推荐的正式分阶段流程
$PY -m pipeline_wss_min.run --stage preprocess
$PY -m pipeline_wss_min.run --stage qa-gate
$PY -m pipeline_wss_min.run --stage global-stats --stats-timesteps peak
$PY -m pipeline_wss_min.run --stage build-samples

# 只重做采样，不重跑上游
$PY -m pipeline_wss_min.run \
  --stage build-samples --wall-n 1500 --sample-name wss_min_peak_w1500

# 仅用于原始目录诊断；不得接统计、样本或正式训练
$PY -m pipeline_wss_min.run --stage preprocess --all-raw
```

集群入口：

```bash
/public/slurm/bin/sbatch --parsable \
  pipeline_wss_min/cluster/run_preprocess.slurm preprocess
```

## AG → v4 安全迁移

AG v4 不允许直接覆盖活动目录。固定顺序是 staging 重建 → bundle 硬 QA → 生成审核图
→ 用户签核 → 同文件系统目录切换。2026-07-16 已按该合同成功切换；工具在切换后验证或记录提交失败时也会自动回滚。

```bash
RUN_ID=20260715_1921
STAGING="data_wss_min/_staging/ag_v4_20260715_cutover"
REPORT="data_wss_min/pipeline_reports/v4_cutover_${RUN_ID}"
VIS="docs/02-推进与变更/assets_新队列审计/alignment_v4_cutover_review_${RUN_ID}"

# 单例隔离 smoke；不会写 data_wss_min/AG
$PY -m pipeline_wss_min.v4_cutover preprocess --index 63 --out-root "$STAGING"

# 正式 0..76 数组；完成后再运行 QA/哈希/可视化依赖作业
/public/slurm/bin/sbatch --parsable \
  pipeline_wss_min/cluster/run_ag_v4_staging.slurm "$STAGING"

$PY -m pipeline_wss_min.v4_cutover qa \
  --ag-root "$STAGING" --aaa-root data_wss_min --report-dir "$REPORT"
$PY -m pipeline_wss_min.v4_cutover snapshot-manifest \
  --legacy-root data_wss_min --output "$REPORT/AG_legacy_v3_snapshot_manifest.json"
# --legacy-root 也可显式写为 data_wss_min/AG，清单仍统一使用 AG/... 相对路径
$PY -m pipeline_wss_min.visualize_v4_cutover \
  --ag-root "$STAGING" --aaa-root data_wss_min --legacy-root data_wss_min --out-dir "$VIS"

# 用户看图后先固化逐例决定；该命令只写最终 manifest/白名单，不切换数据
$PY -m pipeline_wss_min.v4_cutover finalize-review \
  --report-dir "$REPORT" --decisions "$REPORT/manual_review_decisions_<date>.json"

# 可选：生成终签后红/黑状态图
$PY -m pipeline_wss_min.visualize_v4_cutover \
  --ag-root "$STAGING" --aaa-root data_wss_min --legacy-root data_wss_min \
  --review-manifest "$REPORT/v4_final_dataset_manifest.json" --out-dir "$VIS"

# 仅在最终白名单 ready_for_promotion=true 后单独执行。promote 从原始
# 77 例 staging 按白名单物化候选目录，不会把人工排除病例切入活动 AG。
$PY -m pipeline_wss_min.v4_cutover promote \
  --staging-root "$STAGING" --active-root data_wss_min \
  --snapshot-root "data_wss_min/_snapshots/AG_legacy_v3_${RUN_ID}" \
  --report-dir "$REPORT" --confirm-reviewed
```

路径边界：

- staging：`data_wss_min/_staging/<run>/AG/`；
- 旧版快照：`data_wss_min/_snapshots/AG_legacy_v3_<run>/AG/`；
- 活动路径：`data_wss_min/AG/`，只有用户看图签核后才能切换；
- QA/清单：`data_wss_min/pipeline_reports/v4_cutover_<run>/`；
- 审核图：`docs/02-推进与变更/assets_新队列审计/alignment_v4_cutover_review_<run>/`。

`pipeline_wss_min.run --out-root <path>` 只对 `preprocess/qa-gate` 开放；为防止误读活动
bundle，非默认输出根不能与 `global-stats/build-samples/all` 一起使用。

## AAA/ILO v4 入口

AAA/ILO 的候选清单同时受数据层审计和 v4 坐标架审计约束。详细职责见
[`new_cohorts/README.md`](new_cohorts/README.md)。

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

$PY -m pipeline_wss_min.new_cohorts.audit_frame --workers 4
$PY -m pipeline_wss_min.new_cohorts.ilo_before --stage inventory
$PY -m pipeline_wss_min.new_cohorts.ilo_before --stage qa
$PY -m pipeline_wss_min.new_cohorts.visualize_ilo_before
$PY -m pipeline_wss_min.new_cohorts.preprocess --list
$PY -m pipeline_wss_min.new_cohorts.preprocess --unit-id AAA/ruputer/CHEN_FU
$PY -m pipeline_wss_min.new_cohorts.preprocess --summarize
$PY -m pipeline_wss_min.new_cohorts.qa
$PY -m pipeline_wss_min.new_cohorts.audit_ag
$PY -m pipeline_wss_min.new_cohorts.visualize_frame
```

批量 Slurm：

```bash
/public/slurm/bin/sbatch --parsable \
  pipeline_wss_min/cluster/run_new_cohorts_preprocess.slurm
/public/slurm/bin/sbatch --parsable \
  pipeline_wss_min/cluster/run_new_cohorts_finalize.slurm
```

## 导师展示用单文件

[`examples/preprocess_pipeline.py`](examples/preprocess_pipeline.py) 不超过 250 行，而且不导入
`pipeline_wss_min` 的任何内部模块。原始 Fluent/中心线/STL 读取、时间步与峰值选择、节点
对齐、单位换算、中心线平移修复、解剖坐标架、入口裁剪、归一化、近壁标注、全时序场
堆叠、QA、log-z 和 FPS 样本装配都直接写在该文件中，便于导师独立审阅。

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 完整重跑指定病例；只写 outputs/ 下的展示产物，不改正式 bundle
$PY -m pipeline_wss_min.examples.preprocess_pipeline \
  --cohort AG/fast --case CHEN_SHI_MING
```

## 产物与安全边界

- 正式病例：`data_wss_min/<cohort>/<case>/{bundle.npz,report.json}`。
- 批量审计：`data_wss_min/pipeline_reports/`。
- 模型样本：`data_wss_min/samples/<sample_name>/`。
- 展示产物：`outputs/wss_min/teacher_preprocess/`。
- 运行日志：`logs/wss_min_<stage>_<timestamp>.log`；已完成且结论已落入报告的日志可清理。
- 不允许把 AG v3 bundle 与 AAA/ILO v4 bundle 混合训练；需要混池时必须先整体重建 AG v4。
- excluded/pending 病例即使磁盘存在历史 bundle，也不得进入统计、训练或评估。

历史算法演进、病例剔除依据和已完成作业结论统一查阅：

- `docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`
- `docs/02-推进与变更/新队列数据可用性审计_AAA_ILO_2026-07-10.md`
- `docs/02-推进与变更/_archive/WSS最小化/`
