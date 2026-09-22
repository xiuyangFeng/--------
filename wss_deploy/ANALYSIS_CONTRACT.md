# 分析层与报告交互合同 v1（2026-09-20，四路并行开发共用）

本文件是四个并行开发任务之间唯一的数据契约。改契约必须先改这里。

## 0. 文件归属（不得改动他人文件）

| 任务 | 拥有的文件 |
|---|---|
| A 分析层（Python） | `wss_deploy/analysis.py`（新）、`wss_deploy/pipeline.py`、`wss_deploy/volume_pipeline.py`、`wss_deploy/geometry.py`（只允许新增返回键）、`wss_deploy/volume_geometry.py`（只允许新增返回键）、`wss_deploy/rebuild_report.py`、`wss_deploy/metrics.py`、`tests/test_analysis.py`（新）、`tests/test_volume_pipeline.py`（仅为适配 stub） |
| B 壁面 WSS 报告 | `wss_deploy/report.py`、`tests/test_report.py` |
| C 体场报告 | `wss_deploy/volume_report.py`、`wss_deploy/static/volume_viewer.js`、`tests/test_volume_report.py` |
| D 工作台与服务 | `wss_deploy/jobs.py`、`wss_deploy/server.py`、`wss_deploy/static/app.js`、`wss_deploy/static/app.css`、`wss_deploy/static/index.html`、`wss_deploy/static/compare.html`（新）、`wss_deploy/static/compare.js`（新）、`wss_deploy/onepager.py`（新）、`tests/test_review_lock.py`（新）、`tests/test_onepager.py`（新）、`tests/test_compare_page.py`（新） |

`README.md` 与 `docs/` 由主会话在合并后统一回填，四个任务都不要改。

`build_html` 的新参数由 A 提供；B/C 必须把新键视为**可选**：缺省时相应面板隐藏，不报错。这样 A 与 B/C 可以独立测试。

## 1. `summary["profiles"]` 沿程曲线

```json
{"schema_version": "wss-deploy.profiles/v1", "bin_mm": 2.0,
 "branches": [{"segment_id": 3, "name": "左髂外", "parent_id": 1,
   "s_local_mm": [...], "s_from_root_mm": [...], "radius_mm": [...],
   "wss":    {"mean_pa": [...], "p99_pa": [...], "min_pa": [...], "n": [...]},
   "volume": {"speed_mean_m_s": [...], "speed_max_m_s": [...], "pressure_mean_pa": [...], "pressure_min_pa": [...], "n": [...]}
 }]}
```

- 按 atlas 的 `s_local_mm` 每 `bin_mm` 一格分箱；`n=0` 的空箱其余值为 `null`。`radius_mm` 取 atlas 该弧长处的半径（对 atlas 行按 s_local 分箱取中位）。
- 壁面族只有 `wss`，体场族只有 `volume`（内部点 `point_kind=1` 分箱）。压力是相对压，曲线只用于差值。
- 分支顺序：主动脉、左髂总、左髂外、左髂内、右髂总、右髂外、右髂内（按 `summary["branch_names"]` 的中文名匹配，其余附后）。

## 2. `summary["findings"]` 发现列表

```json
{"schema_version": "wss-deploy.findings/v1",
 "items": [{"id": "F1", "kind": "high_wss_cluster", "label": "右髂内高 WSS 区", "branch": "右髂内", "segment_id": 6,
   "value": 21.9, "units": "Pa", "xyz_mm": [x, y, z], "s_from_root_mm": 255.0,
   "extent_mm": 12.0, "area_mm2": 310.0, "n_points": 1420, "rank": 1,
   "severity": "attention", "definition": "WSS ≥ 空间 p99 的连通簇，面积按点占比估计",
   "point_indices": [..] }]}
```

- `kind` 取值。壁面族：`high_wss_cluster`（≥p99 阈值的连通簇，按面积降序，最多 5）、`low_wss_cluster`（< 低阈值 0.4 Pa 的连通簇，最多 5）、`max_wss`（全场最大值点）、`max_diameter`（几何最大直径处）、`min_radius`（每条分支的最小半径处，仅当狭窄指数 ≥ 0.3 时列出）。体场族：`max_speed`、`min_pressure`、`pressure_drop`（每条分支近端 10% 与远端 10% 弧长段的平均压差，`value` = ΔP，`xyz_mm` 取分支中点）、`low_speed_region`（速度 < 全场 p10 的连通簇，最多 3）、`max_diameter`、`min_radius` 同上。
- `severity`：`high_wss_cluster`/`max_wss` 值 ≥ 7 Pa、或 `low_wss_cluster` 总面积 > 20% 壁面为 `attention`；其它预测项 `note`；几何项 `info`。
- `extent_mm` 供相机视距与高亮半径；`point_indices` 可选，最多 2000 个点云索引（壁面族指 `cloud.pts` 的索引，体场族指内部点索引）。
- 每条 `definition` 用一句中文写清口径，不夸大：面积是点占比乘输入壁面面积。

## 3. 可信区域 `trust`

```json
"trust": {"schema_version": "wss-deploy.trust/v1",
  "bits": {"1": "interpolation_uncovered", "2": "rough_surface", "4": "geometry_out_of_range", "8": "low_sample_support", "16": "near_opening"},
  "fractions": {"interpolation_uncovered": 0.02, "rough_surface": 0.05, ...},
  "sources": [{"bit": 1, "label": "插值无支撑", "rule": "Gaussian 插值 1.5 mm 内无预测点"}, ...]}
```

数组通过 `build_html` 传入并内嵌报告：

- 壁面族 `build_html(mesh={..., "trust": uint8[n_vertices]})`：bit1 未覆盖顶点（`vertex_wss_pa` 为 NaN）；bit2 粗糙（点的 PCA `surface_variation` > 0.02，取最近顶点）；bit4 几何越界（`reference_assessment.checks` 中 `status == "review"` 的分支顶点；没有 profile 时全 0）。
- 体场族 `build_html(mesh={..., "trust": uint8[n_vertices]}, cloud={..., "trust": uint8[n_interior]})`：壁面顶点 bit1 = 壁面压力插值未覆盖；内部点 bit8 = 采样支撑弱（8 近邻半径 > 全局中位数 × 2），bit4 = 几何越界分支，bit16 = 距最近切口中心 < 2 × 该切口半径。
- 体场族 `cloud` 新增可选键（探针用）：`s_from_root_mm`、`radius_mm`、`dist_to_wall_mm`，均 `float32[n_interior]`。

## 4. `summary["frame_transform"]` 补齐（壁面族与体场族一致）

```json
{"source": "...", "direction_source": "unknown_stl", "rotation": [[..],[..],[..]], "origin_mm": [x, y, z]}
```

`rotation` 的行是新轴在世界坐标中的方向：`p_aligned = R @ (p - origin)`；x = 患者左，z = 指向入口（上），y = z × x（患者后）。B/C 用它做标准视角「前 / 后 / 左 / 右 / 上 / 下」；`direction_source` 为 `unknown_stl` 时界面必须标注「按解剖坐标架推断，请核对左右」。

## 5. 视图状态（B/C 各自实现，格式统一）

```json
{"schema_version": "wss-deploy.view/v1", "family": "wall", "run_identity": "...",
 "camera": {"position": [3], "target": [3], "up": [3]},
 "field": "wss", "mode": "field", "colormap": "rainbow",
 "range": {"mode": "case", "min": 0, "max": 10}, "bands": 0, "log": false,
 "units": "Pa", "thresholds_pa": [0.4, 4, 7], "opacity": 1.0,
 "overlay": {"trust": false, "contours": false},
 "slice": {}, "highlight": {}}
```

提供：按 `run_identity` 存 localStorage、导出/导入 JSON、URL hash `#view=<base64url(json)>`、「复现此图」。`camera` 里存世界坐标。

## 6. 报告间相机联动协议（供 D 的并排比较页）

- 报告页监听 `window.message`：`{type: "wss-view:set-camera", camera: {position, target, up}}`，其中坐标为**解剖坐标架下**的相对坐标（`aligned = R @ (p - origin)`），报告页换算回自己的世界坐标后应用，并在应用期间不回发。
- 报告页相机变化时，若 `window.parent !== window`，向 parent `postMessage({type: "wss-view:camera", family, camera})`，坐标同样是解剖坐标架下的相对坐标，节流不超过 20 Hz。
- 报告页在解剖坐标架缺失时不参与联动。

## 7. 单位与阈值（显示层）

- 1 Pa = 10 dyn/cm²；压力可显示 mmHg（1 mmHg = 133.322 Pa）。换算只在显示层。
- 阈值默认 `[0.4, 4, 7]` Pa，报告内可改并即时重算面积占比（壁面族按点云 `wss` 数组、面积 = 点占比 × 输入壁面面积，与 `metrics.py` 口径一致）。改动写入视图状态，不改 `summary.json`。

## 8. `summary["review"]`（D 写，B/C 显示）

```json
{"status": "reviewed" | "unreviewed" | "reopened", "by": "匿名审阅人标识", "at": "ISO 时间", "note": "...", "version": 7}
```

B/C 在报告页脚显示：审阅状态、审阅人、时间、发布包、特征合同哈希。

## 9. 不变量

- `field.npz` 内容逐位不变；新数组只嵌入 HTML。预测数值不变。
- A 完成后必须跑 `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m wss_deploy.regress --out <scratch>/regress --reference-root outputs/wss_deploy_golden/20260920_baseline` 并 5/5 PASS。
- 旧 `summary.json`（没有 profiles / findings / trust / review）时，B/C 报告必须能打开，对应面板隐藏。
- 任何任务都**不重启服务**、不在 `outputs/wss_deploy_jobs` 上跑 `rebuild_report`；验证 rebuild 时把任务目录拷贝到自己的 scratchpad 子目录再跑。
- Python：`~/.conda/envs/GNN/bin/python`，`PYTHONPATH=.`；前端只有 `node --check` 与 node 单测（没有无头浏览器）。

---

# C 类常用功能契约 v1.1（2026-09-21，四路并行开发共用；规格见 docs/02-推进与变更/WSS_PINN/WSS_部署工具_下一轮功能与优化方案_2026-09-21.md §2–§3）

本节是 C1–C18 四个并行任务之间唯一的数据契约。改契约必须先改这里。上文 §0–§9 继续有效。

## 10. 文件归属（不得改动他人文件）

| 组 | 拥有的文件 |
|---|---|
| 后端 | `jobs.py`、`server.py`、`cases.py`（新）、`users.py`（新）、`trash.py`（新，可并入 jobs.py）、`export_table.py`（新）、`bundle.py`（新）、`glossary.py`（主会话已建，可补条目）、`static/glossary.json`（由 `python -m wss_deploy.glossary` 生成）、`onepager.py`、`rebuild_report.py`（只加标注与发现审阅内嵌）、`cli.py`（加 `user`、`jobs claim` 子命令）、`tests/test_cases.py`、`test_duplicates.py`、`test_users.py`、`test_trash.py`、`test_events_owner.py`、`test_export_table.py`、`test_bundle.py`、`test_findings_review.py`、`test_glossary.py`、`test_annotations.py`、`test_preferences.py` |
| 前端 | `static/app.js`、`static/app.css`、`static/index.html`、`static/batch_export.js`（新）、`static/compare.js`、`static/compare.html`、`tests/test_workbench_js.py`（新，node 桩测试） |
| 报告共享库 + 壁面报告 | `static/report_common.js`（新）、`report.py`（`__COMMON__` 占位接入 + 壁面报告全部 C 项）、`tests/test_report.py`、`tests/test_report_common.py`（新） |
| 体场报告 | `volume_report.py`、`static/volume_viewer.js`、`tests/test_volume_report.py` |

`README.md`、`docs/` 与本契约由主会话统一回填。`server.py` 的 `STATIC_FILES` 必须加入 `batch_export.js`、`report_common.js`、`glossary.json`（后端负责）。

通用约束：所有新接口沿用现有安全约束（会话、CSRF、owner 隔离、白名单文件、路径包含检查）；新文件名进 `OUTPUT_FILES`（`annotations.json`、`findings_review.json`）；任何写入任务目录的操作走 `atomic_json`；**不改预测数值**，黄金回归必须 5/5；旧 `summary.json` / 旧视图状态缺新键时按空处理。

## 11. 后端接口（后端实现，前端与报告消费；前端在后端完成前用本节样例做桩）

错误响应统一为 `{"error": {"message": "…"}}`，状态码见各条。所有 POST/PUT 需 `X-CSRF-Token`（`GET /api/session` 返回 `csrf_token`）；报告页在线时先 `GET /api/session` 取 `csrf_token` 再做 PUT。`PUT` 请求体为 `application/json`，上限见各条。

### 11.1 C1 病例卡 `GET /api/cases?q=&patient_id=&tag=&page=&page_size=&all=0|1`

```json
{"cases": [{"input_sha256": "…64 hex…", "case_ids": ["LIU_YU_MING"], "patient_id": "P-001", "scan_label": "基线", "scan_date": "",
   "tags": ["AAA"], "latest_at": "2026-09-20T17:41:50+08:00", "pending_review": 1,
   "runs": [{"job_id": "20260920_174150_e0502f7ea512", "release_id": "PF6_VF6_peak_3seed_20260920", "family": "volume",
             "status": "done", "review": "unreviewed", "created_at": "…", "run_identity": "…", "version": 7}],
   "latest": {"X5D_v51_5seed_20260916": "<job_id>", "PF6_VF6_peak_3seed_20260920": "<job_id>"},
   "missing_releases": [], "reusable_job_id": "<有 stage A + 已确认出口的最新任务 id 或 null>"}],
 "total": 12, "page": 1, "page_size": 20, "releases": [{"id": "X5D_v51_5seed_20260916", "family": "wall"}, {"id": "PF6_VF6_peak_3seed_20260920", "family": "volume"}]}
```

- 分组键：`job["input_sha256"]`（上传时即写入顶层，见 11.2；旧任务回退 `job["a"]["input_sha256"]`；两者都没有的任务单独成组，键为 `"job:<id>"`）。
- `family` 由 `model_release.contract.protocol` 推出：`single_frame_wss → wall`，`single_frame_volume → volume`，其它 → `null`。
- `latest[release_id]` = 该几何下该发布包最新的 `done` 任务；`missing_releases` = 注册表可用发布包中该几何没有 `done` 任务的 id。
- `runs` 按 `created_at` 倒序；`pending_review` = `done` 且 `review.status != reviewed` 的数量。
- 「补跑 <发布包>」由前端调用现有 `POST /api/jobs/<reusable_job_id>/rerun {version, release_id}`。
- `all=1` 只对 `role == admin` 生效，否则忽略。

### 11.2 C2 上传查重 `POST /api/jobs` 与 `/api/jobs/batch`，新增表单字段 `on_duplicate = ask | reuse | force`（默认 `ask`）

- `JobManager.create` 在保存前计算 `input_sha256`（上传内容的 SHA256）写入任务记录顶层 `job["input_sha256"]`；`rerun` / `reuse` 复制源任务的值。
- `find_by_input(owner, sha) -> list[dict]`：同几何任务，按创建时间倒序，每项 `{"job_id", "case_id", "status", "release_id", "family", "created_at", "review", "mapping_confirmed": bool, "reusable": bool}`。`reusable` = `_gate(a)` 通过且 `mapping` 已确认（源任务审阅锁定不影响复用，复用只读取源任务）。
- 单文件 `ask` 且存在同几何任务 → `409`：

```json
{"error": {"message": "同一几何已有任务。"}, "duplicate": true, "input_sha256": "…",
 "existing": [{"job_id": "…", "case_id": "…", "status": "done", "release_id": "…", "family": "wall", "created_at": "…", "review": "unreviewed", "mapping_confirmed": true, "reusable": true}],
 "reusable": "<job_id 或 null>"}
```

- `reuse`：要求 `reusable` 非空，否则 `409` 同上并加 `"reuse_unavailable": true`；成功走 `rerun` 同样的复制逻辑（只跑 B 段，映射沿用），但病例号、患者字段、标签、备注、发布包、设备、模型数、线程取自**本次上传表单**；新任务记录 `reused_from = <源 job_id>`；响应 `201 {"job": …, "job_id": "…", "reused_from": "…"}`。
- `force`：照常创建。
- 批量：每个文件独立判定；`ask` 遇重复时该项不建任务，`results[i] = {"index": i, "duplicate": true, "existing": [...], "reusable": "…|null"}`（不计入 `failed_count`，另加 `"duplicate_count"`）；`reuse` / `force` 逐项同单文件。
- 验收：复用产生的新任务 `summary.timing_s` 无 `centerline` 项且 `mapping` 与源任务一致。

### 11.3 C3 用户名登录

- `users.json`（jobs 根下，0600）：`{"<username>": {"password": "scrypt$<salt_b64>$<hash_b64>", "role": "user|admin", "created_at": "…", "display_name": "…", "disabled": false}}`；用户名 `^[A-Za-z0-9_.-]{1,64}$`。
- CLI：`python -m wss_deploy.cli user add <name> [--admin] [--jobs-root …]`、`user passwd <name>`、`user disable <name>`（口令从终端 `getpass` 读；非交互时读环境变量 `WSS_DEPLOY_PASSWORD`）；`python -m wss_deploy.cli jobs claim --owner <旧 owner> --user <name> [--jobs-root …]`（服务停止时使用）。
- `GET /api/session` 响应新增：`"login": "none" | "token" | "password"`（本机回环模式 `none`：免登录，行为不变；共享模式存在 `users.json` → `password`，否则 `token`）、`"username"`、`"display_name"`、`"role"`（`user|admin|null`）、`"legacy_token_allowed"`（迁移期开关：环境变量 `WSS_DEPLOY_ALLOW_LEGACY_TOKEN=1` 时即使有 `users.json` 也接受旧令牌）、`"claimable_owners": [...]`（本会话可认领的旧 owner）。
- `POST /api/session` body `{username, password}` 或旧 `{token}`。用户名登录成功后会话行 `owner = username`、`username`、`role`；若请求携带的旧 cookie 会话 owner 与之不同，把旧 owner 记入新会话行 `claimable_owners`，响应带 `"claimable": [{"owner": "…", "jobs": N}]`。响应其余字段同 `GET`。口令错误 / 用户禁用 → `401`；同一用户名 60 s 内 5 次失败 → `429`。
- `POST /api/session/logout` → `{"authenticated": false}` 并清 cookie。`POST /api/session/password` body `{old_password, new_password}`（≥ 8 字符）。
- `POST /api/jobs/claim` body `{"owner": "<旧 owner>"}`：把该 owner 名下任务改为当前用户；只允许 `claimable_owners` 里的 owner，或 `role == admin`；响应 `{"claimed": N, "job_ids": [...]}`；每个任务写事件 `owner_claimed`。
- 管理员：`GET /api/jobs?all=1`、`GET /api/cases?all=1`、`GET /api/trash?all=1` 看全部；管理员对他人任务只读（`get / geometry / report / files / onepage / export / bundle`），变更类接口仍 owner 隔离。
- 会话行新增字段对旧 `.sessions.json` 向后兼容（缺省 `role = null`）。

### 11.4 C4 回收站

- 目录 `<jobs_root>/.trash/<job_id>/`（整个任务目录 `os.rename` 进来）+ `trash.json`：`{"schema_version": "wss-deploy.trash/v1", "deleted_at", "expires_at", "deleted_by": "<owner>", "job_snapshot": <job.json 内容>}`；保留 30 天（`TRASH_DAYS = 30`）；`JobManager(..., clock=time.time)` 可注入时钟供测试。
- 现有 `POST /api/jobs/<id>/delete` 与 `/api/jobs/delete` 改为入回收站；响应在原字段上加 `"trashed": true, "expires_at": "…"`；`deleted_jobs.jsonl` 改在**彻底清除**（purge / 过期）时写入，记录多一个 `"purge_reason": "manual|expired"`。
- `GET /api/trash[?all=1]` → `{"items": [{"id", "case_id", "status_before", "release_id", "family", "deleted_at", "expires_at", "days_left", "patient_id", "scan_label"}]}`（owner 隔离，按 `deleted_at` 倒序）。
- `POST /api/trash/<id>/restore` body `{}` → `{"job": <snapshot>}`（重命名回 jobs 根，重新加载记录，状态保持删除前；id 已占用 → `409`）。`POST /api/trash/<id>/purge` body `{}` → `{"purged": true, "id": "…"}`。
- 启动时与每 24 h 扫描过期项清除；锁定任务仍不可删（沿用审阅锁）。

### 11.5 C5 owner 级事件流 `GET /api/events`

SSE。先发 `event: hello` `data: {"owner_jobs": N, "at": "…"}`，之后每次本 owner 任务状态跃迁发 `event: job`，`data` 为现有 `_live_event` 结构加 `"case_id"`、`"family"`、`"review"`（审阅状态串）；`: keepalive` 每 15 s；最长 3600 s 后关闭（浏览器自动重连）。只含本人任务（管理员也只收本人）。

### 11.6 C6 偏好 `GET/PUT /api/preferences`

- 存 `<jobs_root>/preferences/<owner_key>.json`（`owner_key` = owner 若匹配 `^[A-Za-z0-9_.-]{1,64}$`，否则 `sha256(owner)[:32]`）。
- `GET` → `{"preferences": {...}}`（无文件 → `{}`）。`PUT` body 为整个偏好对象（JSON 对象，≤ 16 KB，整体替换）→ `{"preferences": {...}}`。
- 约定键（前端与报告共用，缺省按空）：

```json
{"schema_version": "wss-deploy.preferences/v1",
 "upload": {"units": "mm", "release_id": "…", "device": "auto", "seed_count": "all", "threads": "", "remember_patient": true,
            "last_patient": {"patient_id": "", "scan_label": "", "scan_date": "", "tags": ""}},
 "report_defaults": {"wall": {"colormap": "rainbow", "bands": 0, "units": "Pa", "thresholds_pa": [0.4, 4, 7], "opacity": 1, "log": false, "lang": "zh"},
                     "volume": {"colormap": "rainbow", "bands": 0, "pressure_units": "Pa", "speed_units": "m/s", "opacity": 1, "lang": "zh"}},
 "presets": {"wall": [{"name": "…", "state": {…视图状态…}, "created_at": "…"}], "volume": [...]},
 "notifications": {"enabled": true}}
```

报告页优先级：`#view=` 链接 > 服务端 `report_defaults.<family>`（在线同源时 `fetch('/api/preferences')`）> localStorage `wss-report-defaults` > 内置默认。导出的视图状态里显式记录全部键，不受默认值影响。

### 11.7 C8 标注 `PUT /api/jobs/<id>/annotations`

- body（≤ 64 KB）：`{"items": [{"id": "A1", "xyz_mm": [x, y, z], "text": "…", "color": "#d97706", "branch": "左髂内", "segment_id": 6, "s_from_root_mm": 210.5, "created_at": "…"}], "version": <可选，给了就必须等于当前任务版本>}`；`text` ≤ 200 字符、≤ 50 条。
- 写 `<job>/annotations.json`：`{"schema_version": "wss-deploy.annotations/v1", "items": [...], "updated_at": "…", "updated_by": "<owner>"}`；同时把同一对象写入 `summary.json["annotations"]` 与报告内嵌 meta（`update_html_meta`）；任务事件 `annotations_updated`（版本 +1）。
- 只有 `done` 任务可写；审阅锁定 → `409`；响应 `{"annotations": <存储对象>, "version": <新任务版本>}`。
- 读取：报告在线时 `GET files/annotations.json`（相对报告 URL；文件进 `OUTPUT_FILES`，不存在 → 404 → 报告按空处理），离线时用 meta 内嵌副本 `META.annotations`。
- `rebuild_report` 把 `annotations.json`（若有）内嵌为 `meta["annotations"]`。
- 一页纸附「标注」表（编号、分支、弧长、文字）。

### 11.8 C14 汇总表 `GET /api/jobs/export?ids=a,b,c&format=csv|xlsx`

固定列（顺序固定，缺失填空）：`job_id, case_id, patient_id, scan_label, scan_date, tags, release_id, family, status, review_status, reviewer, created_at, input_sha256_12, run_identity_12, compute_total_s`；壁面族列：`wss_p99_pa, wss_max_pa, wss_mean_pa, area_frac_low, area_low_mm2, area_frac_high, area_high_mm2, area_frac_very_high, area_very_high_mm2, branch_p99_主动脉, branch_p99_左髂总, branch_p99_左髂外, branch_p99_左髂内, branch_p99_右髂总, branch_p99_右髂外, branch_p99_右髂内, population_percentile, quality_grade`；体场族列：`speed_p99_m_s, speed_max_m_s, pressure_min_pa, pressure_max_pa, dp_主动脉, dp_左髂总, dp_左髂外, dp_左髂内, dp_右髂总, dp_右髂外, dp_右髂内, low_speed_regions`；最后 `findings_attention, findings_note`。CSV 带 UTF-8 BOM；xlsx 用 openpyxl；≤ 200 个 id；owner 隔离（不属于本人的 id 报 404）。文件名 `wss_summary_<YYYYMMDD_HHMM>.csv|xlsx`。

### 11.9 C15 打包 `GET /api/jobs/<id>/bundle.zip`、`POST /api/jobs/bundle {"ids": [...]}`

zip 内：白名单文件（存在的）、按需生成的 `onepage.html`、`annotations.json` / `findings_review.json`（若有）、`README.txt`（病例号、发布包、run_identity、口径说明、生成时间）。`.html/.json/.csv/.txt/.vtp` 用 DEFLATED，`.npz` 用 STORED。多选 → 外层 zip 内每例一个 `<case_id>_<job_id>.zip`；≤ 50 个 id。锁定不影响下载；只有 `done` 任务可打包。POST 响应 `Content-Type: application/zip` + `Content-Disposition`。

### 11.10 C16 发现审阅 `PUT /api/jobs/<id>/findings_review`

- body（≤ 64 KB）：`{"items": {"F1": {"decision": "confirmed" | "rejected" | null, "note": "…"}}, "added": [{"id": "M1", "xyz_mm": [x, y, z], "branch": "…", "segment_id": 6, "s_from_root_mm": 0, "text": "…", "kind": "manual", "severity": "note"}], "version": <可选>}`；`note`/`text` ≤ 500 字符；`added` ≤ 30 条。
- 写 `<job>/findings_review.json`：`{"schema_version": "wss-deploy.findings_review/v1", "items": {...}, "added": [...], "updated_at": "…", "updated_by": "<owner>"}`；合并进 `summary.json["findings"]["review"]`（同一对象）与报告内嵌 meta（`update_html_meta`）；事件 `findings_review_updated`（版本 +1）；锁定 → `409`；响应 `{"findings_review": <存储对象>, "version": <新版本>}`。
- 读取：在线 `GET files/findings_review.json`；离线 `META.findings.review`。
- 一页纸：确认项 ☑、未判定项 ☐ 正文；驳回项列附录；手动发现标「人工」。`rebuild_report` 把 `findings_review.json` 合并进 `meta["findings"]["review"]`。

### 11.11 C17 术语 `GET /static/glossary.json`

`wss_deploy/glossary.py` 是单一来源：`GLOSSARY = {key: {"zh": 名称, "zh_desc": 一句解释, "en": name, "en_desc": …}}`；`python -m wss_deploy.glossary` 写出 `static/glossary.json`；`tests/test_glossary.py` 校验 JSON 与 Python 一致，并校验报告模板里出现的 `data-gloss="<key>"` 都在字典里（缺失即失败）。报告构建时把 JSON 内嵌进页面（`__GLOSSARY__` 占位，见 §12），一页纸末尾附术语表。报告与工作台里用 `<button class="gloss" data-gloss="p99">?</button>` 弹出解释。

## 12. 报告共享库 `static/report_common.js`（报告共享库组实现；壁面与体场报告通过 `__COMMON__` 占位内嵌，工作台通过 `/static/report_common.js` 引用）

UMD：`root.WssReportCommon = common; module.exports = common`（同 `REPORT_CORE_JS` 写法）。不依赖 DOM 与 three.js 的纯函数放前面，DOM/three 相关函数接收对象参数、缺省时不报错。

### 12.1 中心线（C7）

```js
buildCenterlineGroups({xyz: Float32Array(3n), radius: Float32Array(n), edges: Uint32Array(2m), segment: Int32Array(n)}, branchNames?) -> groups
// groups: [{segment_id, name, xyz: Float32Array(3k), radius: Float32Array(k), s: Float32Array(k) /* 分支内弧长 mm，从分支起点 */, parent_id: number|null, parent_s: number /* 在父分支上的挂接弧长 */, length_mm}]
// parent 推断：分支起点到其它分支折线最近点距离 < max(3 mm, 1.5×起点半径) 的那条；没有 → null（根）。
projectToCenterline(groups, p /* [x,y,z] */) -> {segment_id, name, s /* 分支内弧长 */, radius_mm, row /* 最近折线点索引 */, xyz /* 投影点 */, dist_mm}
arcDistance(groups, a, b /* [x,y,z] */) -> {value_mm, same_branch: bool, path: [segment_id, …]}   // 同分支 |s_a − s_b|；跨分支沿树到公共祖先求和
localDiameter(groups, p) -> {diameter_mm /* 2 × 内切半径 */, radius_mm, segment_id, name, s, dist_to_junction_mm}
straightDistance(a, b) -> mm
```

### 12.2 测量、标注、探针记录（C7、C8、C11）—— 视图状态键

```json
"measurements": [{"id": "D1", "kind": "distance|arc|diameter|segment", "points": [[x,y,z], [x,y,z]], "value_mm": 12.3, "branch": "左髂外", "segment_id": 3, "label": "…", "created_at": "…"}],
"annotations": {"items": [ …同 §11.7… ]},
"probe_log": [{"id": "P1", "xyz_mm": [x,y,z], "branch": "…", "segment_id": 3, "s_from_root_mm": 0, "radius_mm": 0, "values": {"wss_pa": 1.2} /* 体场：speed_m_s, pressure_pa, u,v,w, dist_to_wall_mm */, "created_at": "…"}],
"branches_hidden": [3, 4], "preset_name": "…", "lang": "zh|en",
"export": {"scale": 2, "background": "white|transparent|current", "colorbar": "overlay|none|svg", "ui": false}
```

辅助函数：`newId(prefix, existing)`、`measurementLabel(m, lang)`、`probeToTSV(rows, lang)`、`probeToCSV(rows, lang)`（BOM + CRLF）。

### 12.3 出版级导图（C12）

```js
colorbarSVG({stops: [[t, "#rrggbb"], …] 或 colormap 名, min, max, bands, log, units, title, lang, width, height, orientation: "vertical|horizontal"}) -> "<svg …>…</svg>"
englishLabel(key, lang) -> string   // 字典：图例标题、单位、分支名（Aorta, Left/Right CIA, Left/Right EIA, Left/Right IIA）、测量与标注默认词
exportFilename({case_id, view, field, scale, ext}) -> "<case>_<view>_<field>_<k>x.png"
renderOffscreen({renderer, scene, camera, width, height, scale, background /* "white"|"transparent"|"current" */, clearColor}) -> dataURL   // 临时 setSize(w×k, h×k, false) + setPixelRatio(1)，透明时 setClearColor(0,0)，渲染一帧后恢复；失败（超出显卡上限）降级到 scale 1 并在返回对象里标 downgraded
```

### 12.4 内置预设（C9）

`builtinPresets(family, meta) -> [{name, description, build(ctx) -> partialViewState}]`：壁面「瘤囊低 WSS 区」「髂分叉热点」「主动脉沿程」；体场「沿程压降」「流线全貌」「瘤囊截面系列」。`build` 按病例的分支名与发现列表计算，返回**部分**视图状态，由报告页合并到当前状态再应用。用户预设读写偏好 `presets.<family>`（在线）或 localStorage `wss-report-presets:<family>`（离线）。

### 12.5 术语弹窗（C17）

`glossaryPopover({glossary, lang, mount})` 返回 `{show(key, anchorEl), hide()}`；页面里 `data-gloss` 按钮委托绑定。

### 12.6 报告页 message 协议（C13，父页 = 工作台 `batch_export.js` 或并排页）

| 方向 | 消息 | 说明 |
|---|---|---|
| 报告 → 父 | `{type: "wss-view:ready", family, run_identity, case_id}` | 数据解码、场景建好后发一次（无 WebGL 时也发，带 `webgl: false`） |
| 父 → 报告 | `{type: "wss-view:apply-state", state, request_id}` | 应用视图状态（部分键允许缺省），完成后回 `{type: "wss-view:applied", request_id}` |
| 父 → 报告 | `{type: "wss-view:export", options: {scale, background, colorbar, ui, lang, filename?}, request_id}` | 离屏导出 |
| 报告 → 父 | `{type: "wss-view:exported", request_id, png_base64, filename, colorbar_svg?, width, height, downgraded?}` | |
| 报告 → 父 | `{type: "wss-view:error", request_id?, message}` | 任何失败 |

仅接受 `event.origin === location.origin`（离线 file:// 时不启用）。既有 `wss-view:set-camera` / `wss-view:camera` 不变。

## 13. 前端 `static/batch_export.js`（前端组）

`window.WssBatchExport = {zipStore(files: [{name, bytes: Uint8Array, mtime?}]) -> Uint8Array /* store 模式 zip，CRC32 正确 */, run({jobs: [{id, case_id, family}], state, options, onProgress}) -> Promise<{zip: Uint8Array, failed: [{id, message}]}>}`；node 可测（`zipStore` 与 CRC32 不依赖 DOM）。隐藏 iframe 逐个加载 `/api/jobs/<id>/report`，等 `wss-view:ready` → `apply-state` → `export` → 收集；每例 60 s 超时计失败。

## 14. 实现记录与偏差（2026-09-21 合并后）

- `GET /api/cases` 响应每张卡多带 `group_key`（无 `input_sha256` 的任务为 `job:<id>`）与 `families`；`runs[]` 多带 `case_id`、`reusable`、`reused_from`、`source_job_id`。
- 批量上传 `ask` 遇重复：`results[i]` 为 `{index, duplicate: true, input_sha256, existing, reusable}`，不计入 `failed_count`，另有 `duplicate_count`；`reuse` 不可用时该项进 `error` 并带 `reuse_unavailable: true`。
- `JobManager.create(on_duplicate=...)` 直接调用时默认 `force`（旧行为）；HTTP 层默认 `ask`。
- 会话：`GET/POST /api/session` 多返回 `legacy_token_allowed`；旧令牌会话的 `role = "legacy"`；`POST /api/session/logout` 清 cookie。
- 回收站：`delete` 响应保留 `deleted: true` 并加 `trashed: true, expires_at`；`trash.json` 含 `expires_ts` 与 `provenance`；每日扫描由工作线程 `maybe_purge_expired()` 触发。
- 标注 / 发现判定：任务记录顶层也保存 `annotations` / `findings_review`（与侧车同对象）；PUT 的 `version` 可选（给了才校验）。
- 视图状态：离线报告把发现判定暂存在扩展键 `findings_review`（§12.2 之外，旧状态无此键照常）。
- 壁面报告导出选项 `#exp-ui`「含界面标签」语义 = 勾选时把测量 / 标注标签与色标合成进 PNG。
- 体场报告内嵌数组：中心线半径改键 `center_radius_mm`，`radius_mm` 专指预测点半径（修复探针读错数组）；旧报告按数组长度回退。
- `preferences/<owner>.json`：`owner` 是用户名时按原样命名，随机会话 owner 取 sha256 前 32 位。


---

# 第一、二档日常功能契约 §15（2026-09-21 下午，四路并行共用）

用户裁定做"第一档 + 第二档"：并排同步口径、一页纸配图、元数据可改、截面 CSV、多选批量重跑、真实管径；截面系列拼图、多视角拼图、沿程曲线 SVG、队列总览、取消即时生效。文件归属沿用 §10（后端 / 工作台 / 壁面报告+共享库 / 体场报告）。共享库 `static/report_common.js` 已由主会话补齐 §15.0 的函数并有 node 测试，四路直接调用、不得改其签名。

## 15.0 共享库新函数（已就绪）

- `planeFrame(origin, normal) -> {origin, normal, u, v}`；`planeContour(vertices, faces, plane, wallValues|null, maxRadius) -> segs[[x0,y0,x1,y1,val,keyA,keyB]]`；`contourLoops(segs)`、`selectLoop(loops, segs, origin, maxDist)`、`closeChain(segs)`、`pointInLoop(segs, x, y)`（与体场查看器同实现）。
- `loopPolygon(segs) -> [[x,y],…]`（按端点串成有序多边形）；`sectionMetrics(poly) -> {area_mm2, perimeter_mm, centroid, max_diameter_mm, min_diameter_mm, equivalent_diameter_mm, circularity, n_points}`。
- `crossSection({vertices, faces, origin, normal, wallValues?, maxDist?}) -> {plane, segs, polygon, polygon_world, metrics|null, closed, synthetic, open, found, dmin}`：一次拿到本地管腔轮廓与几何量（面积、最大 / 最小 Feret 直径、等效直径）。
- `composeMontage({panels:[{image, label, caption}], columns, gap, pad, font, title, background, colorbar:{image}, scaleBar:{mm, px}, document?}) -> canvas`；`loadImage(dataURL) -> Promise<Image>`。
- `profileSVG({series:[{name, x[], y[], color?, dash?, width?}], xLabel, yLabel, title, width, height, bands:[{x0,x1,label,color}], yZero?}) -> "<svg…>"`。
- `tableToCSV(columns, rows, comments?) -> string`（BOM + CRLF，`comments` 变成开头的 `# …` 行）。

## 15.1 一页纸配图 `POST /api/jobs/<id>/snapshots`（后端）

- body（≤ 24 MB，`application/json`）：`{"images": [{"name": "front", "png_base64": "…", "width": 1600, "height": 1200, "caption": "前视 · WSS Pa", "view": "front"}], "replace": true}`；`name` 匹配 `^[a-z0-9_-]{1,32}$`，≤ 12 张，每张解码后 ≤ 6 MB 且必须是 PNG（魔数校验）。`replace: true`（默认）整体替换，否则按 name 覆盖合并。
- 存 `<job>/snapshot_<name>.png` 与 `<job>/snapshots.json`：`{"schema_version": "wss-deploy.snapshots/v1", "items": [{"name", "file": "snapshot_front.png", "width", "height", "caption", "view", "bytes", "created_at"}], "updated_at", "updated_by"}`；任务记录顶层 `snapshots` 存同一对象（不进 summary）；事件 `snapshots_updated`（版本 +1）；只有 `done` 任务；**审阅锁定不阻止**（不改数值）。响应 `{"snapshots": <对象>, "version": <新版本>}`。
- 文件白名单：`OUTPUT_FILES` 加 `snapshots.json`；`contained_file` 另接受 `snapshot_<name>.png`（正则），`GET files/snapshot_front.png` 返回 `image/png`。打包 zip 收录快照。
- 一页纸（`onepager.py`）：有 `snapshots.json` 时在"关键数字"之后加「配图」区，2–3 列网格，图以 data URI 内嵌（保持单文件），下方 caption；没有时给一行提示"在三维报告「视图」菜单点「生成一页纸配图」"。
- `rebuild_report` 不动快照。

## 15.2 元数据修改 `POST /api/jobs/<id>/metadata`（后端）

body `{"version": n, "case_id": "…", "patient_id": "…", "scan_label": "…", "scan_date": "YYYY-MM-DD|", "tags": ["…"], "notes": "…"}`（各键可省略 = 不改；校验沿用 `_metadata` / `_text`）。写回任务记录、`summary.json["case_metadata"]`（同名键）与报告内嵌 meta（`update_html_meta`）、run_manifest 不动；事件 `metadata_updated`；锁定 → 409；非 done 任务也允许改（元数据与计算无关）。响应 `{"job": <snapshot>}`。

## 15.3 队列数据 `GET /api/jobs/export?format=json&scope=done|ids[&ids=…][&all=1]`（后端）

- `format=json` 响应 `{"columns": [...], "rows": [{列名: 值}], "population": {"<release_id>": {"metric": "p99_pa", "values_pa": [...136], "case_count": 136, "reference_set": "…", "thresholds_pa": [0.4,4,7]}}}`；`scope=done` = 本 owner 全部 `done` 任务（≤ 500，管理员 `all=1` 全部），`scope=ids`（默认）沿用 `ids=`。人群数组来自发布包 `reference.json.population_reference.values_pa`（只对声明了的发布包给出）。CSV / xlsx 行为不变。

## 15.4 多选批量重跑（工作台，无新接口）

多选栏「用发布包重跑所选」：选发布包 → 逐个调用现有 `POST /api/jobs/<id>/rerun {version, release_id}`（只对 done 且已确认出口的任务），逐项结果汇总；跳过锁定以外的情况不需要（rerun 允许锁定）。

## 15.5 取消即时生效（后端 N1）

- `centerline.run_vessel_geom(..., cancelled=None)`：`subprocess.Popen(start_new_session=True)`，每 0.5 s 轮询 `cancelled()`；取消时对进程组 `SIGTERM`，2 s 后 `SIGKILL`，抛 `InterruptedError("任务已取消")`；超时同样杀进程组。`pipeline.stage_a` 把 `cancelled` 传下去。
- 时间语义：任务记录新增 `attempt_seconds`（本次尝试，重试时从 0 起）；`compute_seconds` 仍是累计；`_snapshot().timing` 加 `attempt_s`；中断 / 取消时写事件 `attempt_aborted`（带 `attempt_s`）。工作台显示「本次 xx s · 累计 xx s」。
- 验收：取消后 3 s 内任务进入 `cancelled` 且 `ps` 无 vessel_geom 残留；黄金回归不受影响（B 段未动）。测试用假子进程（`sleep`）验证杀进程组。

## 15.6 报告页协议新增（两份报告都实现）

- 父 → 报告 `{type: "wss-view:get-state", request_id}` → 报告回 `{type: "wss-view:state", request_id, family, state: <完整视图状态 v1.1>}`。
- `apply-state` 沿用：并排页会只发显示口径子集 `{field, mode, colormap, bands, log, range, units, pressure_units, speed_units, thresholds_pa, opacity, overlay, slice}`，报告合并后应用、不动相机。

## 15.7 并排页同步口径（工作台）

顶栏加「同步显示口径」开关（默认开）：左侧报告任何状态变化 → 并排页每 400 ms 最多一次向左侧发 `get-state`，收到后把 15.6 的子集发给右侧 `apply-state`（反向同理，以最近操作的一侧为源；用 `wss-view:camera` 与 `wss-view:applied` 判断活动侧，避免回环：应用期间 400 ms 内忽略对侧回传）。另加「以左为准 / 以右为准」两个按钮做一次性同步。族不同（wall vs volume）时只同步 `colormap/bands/log/opacity`。

## 15.8 一页纸配图按钮（两份报告，视图菜单）

「生成一页纸配图」（仅在线）：依次离屏渲染 前 / 左 / 上 三个标准视角 + 当前视图（体场再加当前截面放大图 `slice`），2×、白底、色标叠加、当前语言；`POST snapshots`（相对 URL `snapshots`，CSRF 同标注）；成功后状态行给出「打开一页纸」链接（`onepage`）。名称固定：`front, left, top, current[, slice]`；caption 形如「前视 · WSS · Pa」。

## 15.9 截面 CSV（体场报告，放大视图）

「导出 CSV」：`tableToCSV(['x_mm','y_mm','value','units','filled_from_wall'], rows, [病例, 物理量, 中心, 法向, 厚度, 网格 nx×ny, 补全模式])`，只导轮廓内 / 有值的格；文件名 `<case>_slice_<field>.csv`。

## 15.10 截面系列拼图（体场报告，截面菜单「自动截面」下）

分支下拉 + 站位数（默认 6：5/20/40/60/80/95%）+「导出系列拼图」：每站用 `sliceMapData`/`renderSliceMap` 画到 700×620 离屏 canvas（共用当前色标范围、不画色标与页脚），`composeMontage({columns:3, panels:[{image, label:'a'…, caption:'s = 12.0 mm (20%)'}], colorbar:{image: colorbarSVG→loadImage}, title})` → PNG 下载，文件名 `<case>_slices_<branch>_<field>.png`；同时把该系列写入视图状态 `slice.series = {segment_id, fractions}`。

## 15.11 多视角拼图（两份报告，视图菜单）

复选：前 / 后 / 左 / 右 / 上 / 下 / 当前 / （体场）截面；列数 2–4；分辨率沿用导图设置；每幅 `renderOffscreen`（色标不叠加），拼图右侧共用色标（`colorbarSVG` → `loadImage`），面板标 a/b/c…，caption 为视角名（按 `lang`）；文件名 `<case>_montage_<field>_<k>x.png`。

## 15.12 沿程曲线 SVG（两份报告，沿程菜单）

「导出曲线 SVG」：当前分支的全部序列（壁面：半径、WSS 均值 / p99 / 最低；体场：半径、速度均值 / 最大、压力均值 / 最低——按当前物理量导出两张或一张多轴？→ 一张一物理量，半径单独一张，共两张，或用户勾选）用 `profileSVG`，x = 距入口弧长 mm，阈值带（0.4 / 4 / 7 Pa 用 `bands`）；文件名 `<case>_profile_<branch>_<field>.svg`。

## 15.13 队列总览（工作台）

侧栏新卡片「队列总览」（默认折叠）：`GET /api/jobs/export?format=json&scope=done`；筛选：族、发布包、审阅状态、`p99 ≥`、`高 WSS 面积占比 ≥`、`极高 ≥`、`速度 p99 ≥`；直方图（canvas，`workbench_core.js` 里的分箱函数 node 可测）：壁面 p99 分布叠人群参照（`population.values_pa` 半透明）、高 WSS 面积占比分布、体场速度 p99 分布；下方结果表（病例、发布包、p99、面积占比、人群分位、审阅）点击选中任务；「导出筛选结果 CSV / xlsx」调 `export?ids=`。数字保留 2 位；缺失显示「—」。

## 15.14 真实管径（壁面报告 + 共享库）

测量菜单「管径」改为：在点选处取中心线切线为法向，`crossSection({vertices, faces, origin: 投影到中心线的点, normal})`；标签显示「最大直径 xx / 最小 xx / 等效 xx mm · 面积 xx mm²」，三维里画轮廓多边形（`polygon_world`）；找不到轮廓时回退到 2×内切半径并注明。`measurements[].kind='diameter'` 新增字段 `max_diameter_mm, min_diameter_mm, equivalent_diameter_mm, area_mm2, method: 'contour'|'inscribed'`。体场报告的探针 / 截面统计也可顺带显示 `sectionMetrics`（面积、等效直径）。

## 16. §15 实现记录与偏差（2026-09-21 晚合并后）

- 快照：`POST snapshots` 的 PNG 以 inline（无 `Content-Disposition`）方式经 `files/snapshot_<name>.png` 提供，供 `<img>` 直接引用；轻量任务快照多带 `snapshots_count`。
- 元数据：`POST metadata` 同时更新 `summary.json["case_id"]`（不只 `case_metadata`），报告与一页纸随之显示新病例号；`cancel()` 作用于排队 / 等待中的任务时不写 `attempt_aborted`（只有工作线程写）。
- 报告协议：`get-state` 并入既有 message 监听器；体场 `captureView` 顶层新增 `pressure_units / speed_units`，`applyView` 只在 `state.slice` 存在时才动截面。
- 壁面管径：`value_mm` = 等效直径，最大 / 最小 / 等效各有字段；沿程「全部叠加」时 SVG 导出取第一条分支并注明。
- 体场：`renderSliceMap` 新增 `opts.colorbar:false`；`setStats` 第 4 参数 `extra`；`sliceGridToRows(grid, bounds, units?)`；系列拼图不画各面板色标。
- 工作台：`EchoGuard` 为 ES class；启动桩测试的 stub DOM 也接受运行时创建的 id；数字筛选按 `input` 即时刷新。
- 主会话：`cases.py` 成员排序去掉随机 id 决胜，`JobManager.cases()` 按 `created_ts` 倒序喂入。


---

# 医生常用功能第一、二档契约 §17（2026-09-22，四路并行共用）

用户裁定：先做「一、瘤体大小与形态」（沿程自动最大直径 / 瘤体与瘤颈 / 体积 / 分支段级汇总表）和「二、风险区一眼看到」（自动标注发现 / 临床视图预设 / 自动结论文字）。文件归属沿用 §10；新增 Python 模块归后端组。所有新数据都是**新键**：`summary["morphology"]`、`summary["narrative"]`，不改黄金回归比对的任何键（peak / wss_field_pa / per_branch / geometry / surface_statistics / quality / volume_statistics）与 `field.npz`。

## 17.1 `summary["morphology"]`（后端：新模块 `wss_deploy/morphology.py`，两族 B 段与 `rebuild_report` 都算）

```json
{"schema_version": "wss-deploy.morphology/v1", "station_mm": 1.0,
 "method": {"section": "中心线每 1 mm 一站，以局部切线为法向取壁面网格交线的本地闭合环（穿开口时直线封口）", "max_diameter": "环上最大 Feret 直径", "equivalent_diameter": "2·sqrt(面积/π)", "reference_diameter": "主动脉等效直径的第 10 百分位（正常管径的稳健估计）", "sac": "等效直径 ≥ 1.5 × 参考直径的连续区段（AAA 常用定义）", "neck": "入口到瘤体起点之间、等效直径 < 1.2 × 参考直径的近端区段", "volume": "开口用扇形封盖后散度定理求全腔体积；瘤体体积 = 瘤体区段截面面积沿弧长积分"},
 "aorta": {"segment_id": 0, "name": "主动脉", "length_mm": 211.1,
   "stations": {"s_from_root_mm": [...], "s_local_mm": [...], "max_diameter_mm": [...], "min_diameter_mm": [...], "equivalent_diameter_mm": [...], "area_mm2": [...], "closed": [true, ...], "xyz_mm": [[x,y,z], ...]},
   "reference_diameter_mm": 18.2,
   "max": {"max_diameter_mm": 63.4, "equivalent_diameter_mm": 58.9, "min_diameter_mm": 52.0, "area_mm2": 2725.0, "s_from_root_mm": 120.0, "xyz_mm": [x,y,z], "distance_from_inlet_mm": 120.0},
   "sac": {"present": true, "s_start_mm": 80.0, "s_end_mm": 165.0, "length_mm": 85.0, "volume_ml": 96.3, "max_diameter_mm": 63.4, "threshold_mm": 27.3} ,
   "neck": {"present": true, "length_mm": 80.0, "diameter_mean_mm": 19.1, "diameter_min_mm": 17.8, "diameter_max_mm": 21.0}},
 "lumen_volume_ml": 158.2, "lumen_volume_method": "capped_mesh_divergence",
 "branches": [{"segment_id": 0, "name": "主动脉", "length_mm": 211.1, "tortuosity": 1.06, "diameter_min_mm": 15.1, "diameter_max_mm": 63.4, "diameter_mean_mm": 30.2, "area_mean_mm2": 800.0, "n_stations": 211, "n_closed": 208,
                "wss_p99_pa": 16.6, "wss_mean_pa": 2.1, "area_frac_low": 0.34, "area_frac_high": 0.02, "delta_p_pa": null, "speed_mean_m_s": null, "speed_max_m_s": null}],
 "notes": ["无瘤样扩张（主动脉最大等效直径 < 1.5 × 参考直径）", ...]}
```

- 站位：每条分支按 atlas 弧长每 `station_mm` 取一站（分支末端 2 mm 内不取，避免切到开口），法向 = 该处中心线切线；截面算法与 JS `crossSection` 相同（Python 实现：向量化求交 → 按共享边串环 → 选含原点的闭环 / 角度覆盖 > 180° 的开链 → 缺口 ≤ 0.6×链长直线封口），`closed=false` 的站不进最大值统计。`maxDist = max(4 × 局部半径, 8 mm)`。
- `aorta` 取 `branch_names` 中「主动脉」（缺省 segment 0）；没有瘤体时 `sac.present=false`、`neck.present=false`，`max` 仍给出。`branches` 顺序同 §1（主动脉、左髂总、左髂外、左髂内、右髂总、右髂外、右髂内，其余附后）；壁面族填 `wss_*`/`area_frac_*`（来自 `per_branch` 与点云按分支统计，口径与 `metrics.py` 一致），体场族填 `delta_p_pa`（来自 `findings` 的 `pressure_drop`）与 `speed_*`（内部点按分支），缺失为 `null`。
- 性能：站位截面用 numpy 向量化，LIU 例（27 万面、约 500 站）应 < 10 s；记入 `timing_s["morphology"]`。
- `findings` 的 `max_diameter` 项改用 `morphology.aorta.max`（值 = 最大 Feret 直径，`definition` 改为"壁面网格截面最大直径"），没有 morphology 时保持原实现。
- `export_table.py` 新增列：`max_diameter_mm, max_diameter_s_mm, sac_present, sac_length_mm, sac_volume_ml, neck_diameter_mm, neck_length_mm, lumen_volume_ml`（放在壁面列之前，两族都填）。
- `onepager.py` 新增「瘤体形态」（最大直径与位置 / 瘤体长度与体积 / 瘤颈 / 全腔体积 / 参考直径，含一句方法说明）与「血管分支表」（`branches`：长度、扭曲度、直径范围、p99 或 ΔP、低 WSS 占比）。
- 术语（`glossary.py`）新增：`max_diameter`, `equivalent_diameter`, `reference_diameter`, `aneurysm_sac`, `aneurysm_neck`, `lumen_volume`, `tortuosity`, `narrative`。

## 17.2 `summary["narrative"]` 自动结论（后端：`wss_deploy/narrative.py`）

```json
{"schema_version": "wss-deploy.narrative/v1", "generated_at": "…",
 "zh": ["主动脉最大直径 63.4 mm（截面最大 Feret 直径），位于入口下 120 mm；瘤体长 85 mm，体积约 96 mL；近端瘤颈长 80 mm、平均直径 19.1 mm。", "低 WSS（< 0.4 Pa）区占壁面 34%，主要位于主动脉；高 WSS（> 4 Pa）热点 2 处，最高 21.9 Pa 位于左髂内。", "壁面 WSS 空间 p99 为 16.6 Pa，处于 136 例参照人群第 32 百分位。", "以上为固定收缩期单帧预测的参考描述，非诊断结论。"],
 "en": ["…"],
 "edited": null, "edited_by": null, "edited_at": null}
```

- 只陈述存在的量：壁面族说 WSS 三句，体场族说压差 / 速度两句（「主动脉近远端压差 xx Pa，最大速度 xx m/s 位于 …」），形态一句两族共有；没有瘤体时说「主动脉最大直径 xx mm，未见瘤样扩张（< 1.5 × 参考直径）」。最后一句固定免责声明。数字：直径 1 位小数、体积整数、百分比整数、Pa 1 位。
- 审阅人修改：`PUT /api/jobs/<id>/narrative` body `{"text": "…", "version"?}`（≤ 4000 字符；空串 = 恢复自动）→ `narrative.json` `{auto: <生成对象>, edited, edited_by, edited_at}`，镜像进 `summary["narrative"].edited/edited_by/edited_at` 与报告 meta（`update_html_meta`）；事件 `narrative_updated`；锁定 → 409；响应 `{"narrative": …, "version"}`。一页纸顶部「结论（参考）」显示 `edited`（标「审阅人已修改」）否则 `zh` 自动句；打包 zip 收录 `narrative.json`；`OUTPUT_FILES` 加 `narrative.json`。
- 报告页在「统计与口径」菜单顶部显示自动结论（只读）。

## 17.3 视图状态与预设（两报告）

- 新键 `labels: {"findings": N, "branches": bool}`（N = 0 关闭，默认 0；`branches` 默认 false）。`findings: N` 时在三维上为发现列表前 N 条（按 rank，排除驳回项）钉 HTML 标签「F1 高 WSS 21.9 Pa」（class `flabel`，颜色按 severity），`branches: true` 时在每条分支中点（atlas `s_local` 50% 处）钉分支名标签（class `blabel`）。标签随导图、拼图、一页纸配图合成（复用标注标签的合成路径）。「显示」菜单加「自动标注：发现 前 N 条 / 分支名」控件。
- 最大直径标记：`morphology.aorta.max` 存在时，「沿程」菜单加「最大直径」行（数值 + 位置 + 「飞到」），三维上画该站截面轮廓环（体场同样）；沿程曲线加「直径（最大 / 等效）」序列（来自 `morphology.branches[].stations`——为省体积，报告 meta 里 `stations` 只保留 `s_from_root_mm / max_diameter_mm / equivalent_diameter_mm`，后端负责裁剪进 meta）。
- 预设「临床视图」（共享库已加，两族）：前视 + `labels:{findings:5, branches:true}` + 默认阈值 + 彩虹色标；壁面 `mode:'wss'`，体场 `mode:'cloud', field:'velocity'`。报告按 `labels` 键渲染即可，预设本身不需报告改动。

## 17.4 工作台

- 详情页「结论（参考）」卡片：显示 `summary.narrative`（edited 优先，标注状态），「编辑结论」textarea → `PUT narrative`，「恢复自动」发空串；锁定时只读。
- 病例卡 / 结果卡显示「最大直径 xx mm」芯片（有 morphology 时）。队列总览加「最大直径」直方图与 `最大直径 ≥` 筛选（列 `max_diameter_mm`）。

## 18. §17 实现记录与偏差（2026-09-22 合并后）

- `morphology.stations` 多带 `inscribed_diameter_mm / reliable / reoriented / tilt_deg / raw_max_diameter_mm`；`aorta.max` 多带 `inscribed_diameter_mm / obliquity_ratio / oblique / station_index / s_local_mm`；`sac` 多带 `equivalent_diameter_max_mm / threshold_mm / n_stations`，`neck` 多带 `s_start_mm / s_end_mm / gap_to_sac_mm / threshold_mm`。`trim_for_report` 保留 `reliable`。
- 可靠性规则（§17.1 未写、合并时加）：斜切 min/inscribed > 1.6、狭长 max/min > 2.2、实心度 < 0.8 → 可疑站锥内（15/30/45° × 8）重切取最小面积闭合截面；所有统计只用 `reliable` 站，无可靠站时回退全部闭合站并注明。
- 轻量任务快照带 `max_diameter_mm`（工作台芯片），`summary.morphology` 在任务记录里是摘要（digest）不含站位数组。
- 报告：标签顺序按发现列表顺序（严重度 → 排名）而非原始 rank；`labels.findings` 接受 0–50 任意整数，菜单只列 0/3/5/10；自动标签只在「含界面标签」勾选时合成进 PNG（与标注同规则）；直径序列画在半径子图同轴；体场英文标签用 `englishLabel` 已知的键名（如「最低压力」）。
- 结论：`narrative.json` 结构 `{auto, edited, edited_by, edited_at}`，summary 里 `narrative` 为合并后的显示对象；`PUT` 在锁定时 409。

