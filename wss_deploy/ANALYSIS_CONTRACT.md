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

# C 类常用功能契约 v1.1（2026-09-21，四路并行开发共用；规格见 docs/02-推进与变更/05-部署工具/_archive/WSS_部署工具_下一轮功能与优化方案_2026-09-21.md §2–§3）

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


---

# v0.12 三方面优化契约 §19（2026-09-23，五路并行共用）

用户 2026-09-23 凌晨裁定：从「日常使用与操作便利 / 医生功能补充 / 界面设计」三方面优化，**按主会话的判断一直做到开发完成，次日验收**。方案与审计证据见 `docs/02-推进与变更/05-部署工具/_archive/WSS_部署工具_三方面优化方案与落地_2026-09-23.md`。本节是五路并行的共同契约；文件归属见 §19.0，**不得改动他人文件**；主会话已先行落地 §19.1 共享函数、§19.4 模板模块与 `devshot.py`。

通用约束（沿用 §10）：不改预测数值，黄金回归 5/5（`regress.SUMMARY_KEYS` 中的 `peak / wss_field_pa / per_branch / geometry / surface_statistics / quality / volume_statistics / murray_shares / caps / cloud / mapping / branch_names` 与 `field.npz` 逐位不变）；新数据一律是**新键**，旧 `summary.json` / 旧任务缺新键时按空处理；所有新接口沿用会话、CSRF、owner 隔离、白名单文件、路径包含检查；写任务目录走 `atomic_json`；改报告脚本必须配 stub DOM 的 node 行为测试（`test_report_field_tab_click_repaints_wall` 是模板）；改工作台必须跑 `tests/test_workbench_js.py` 启动桩。界面改动**必须用 `python -m wss_deploy.devshot` 在自己的沙箱服务上截图自查**（§19.8）。

## 19.0 文件归属

| 组 | 拥有的文件 |
|---|---|
| W1 内容与分析（后端内容组） | `onepager.py`、`narrative.py`、`analysis.py`、`quality.py`、`export_table.py`、`comparison.py`、`glossary.py` + `static/glossary.json`（由 `python -m wss_deploy.glossary` 生成）、`pipeline.py`、`rebuild_report.py`、`build_reference_profiles.py`、`morphology.py`（只读，除非必要）、`cycle_fields.py`（可加函数，不改既有函数行为）；测试 `test_onepager.py`、`test_narrative.py`、`test_analysis.py`、`test_export_table.py`、`test_comparison.py`、`test_glossary.py`、`test_reference_profiles.py`、`test_cycle_release.py`、新建 `test_cycle_content.py` |
| W2 运维与服务（后端运维组） | `cli.py`、`server.py`、`jobs.py`、`users.py`、`bundle.py`、`cases.py`、`__init__.py`（版本号），新建 `service.py`、`doctor.py`、`timeline.py`、`report_freshness.py`；测试 `test_server_*.py`、`test_p2_jobs.py`、`test_bundle.py`、`test_cases.py`，新建 `test_service_cli.py`、`test_doctor.py`、`test_timeline.py`、`test_report_freshness.py`、`test_health.py`、`test_report_template_api.py` |
| W3 工作台（前端组） | `static/app.js`、`static/app.css`、`static/index.html`、`static/workbench_core.js`、`static/batch_export.js`、`static/compare.js`、`static/compare.html`；测试 `test_workbench_js.py`、`test_compare_page.py` |
| W4 壁面报告 | `report.py`；测试 `test_report.py` |
| W5 体场报告 | `volume_report.py`、`static/volume_viewer.js`；测试 `test_volume_report.py` |
| 主会话 | `static/report_common.js`（§19.1 已加）、`report_template.py`（§19.4 已建）、`devshot.py`、`README.md`、本契约、`docs/` |

需要共享库新函数时：先在自己文件里局部实现并在完成报告里写明，由主会话合并时决定是否上移。`server.py` 的 `STATIC_FILES` 已含 `report_common.js`；工作台若新引用静态文件须在报告里说明（W2 负责白名单）。

## 19.1 共享库新函数（`static/report_common.js`，主会话已就绪，`tests/test_report_common.py` 覆盖）

- `fitView(pts, {dir, up, fov, aspect, margin=1.12, center?, limit=6000})` → `{position, target, up, distance}`：透视相机沿 `dir`（相机→目标）看，`up` 朝上，把采样点全部框进视口且目标移到投影包围盒中心。`pts` 可以是扁平 xyz 数组或 `[[x,y,z],…]`。
- `formatValue(v, {digits=3, maxDecimals=4, missing='—'})`：永不出现科学计数法（0.00177 → `0.0018`，1.572 → `1.57`，17 → `17.0`，15544.8 → `15545`，|v| < 1e-4 → `0`）。色标刻度、读数、表格统一用它。
- `localTime(iso, {seconds})` → 浏览器本地时区的 `2026-09-22 17:37`；`friendlyTime(iso, nowMs?)` → `刚刚 / 12 分钟前 / 今天 17:37 / 昨天 17:37 / 9月20日 17:41 / 2025年1月2日`。任务记录里的时间戳带不同偏移（+08:00 与 -07:00 并存），**界面一律经这两个函数显示**，悬停 `title` 给 `localTime(…, {seconds:true})`。
- `installShortcuts(bindings, {doc, title, helpGroup})` → `{showHelp, hideHelp, dispose, bindings}`：`bindings=[{keys:['1'], label:'前视', group:'视角', run(ev), when?(ev)}]`，`keys` 为 `KeyboardEvent.key`（字母不分大小写，`shift+x` 要求 Shift）；在输入框 / 下拉 / contenteditable 中或按住 Ctrl/Meta/Alt 时不触发；自动附带 `?` 打开 / 关闭帮助层、Esc 关闭。`shortcutMatches / shortcutRows` 是可测纯函数。

## 19.2 周期量（M1 三头）全链路口径（W1 实现数据，W3 / W4 消费）

关键数字统一取自 `summary.cycle`（`cycle_fields.cycle_block`）：TAWSS 均值 = `cycle.fields.tawss.mean`、低 TAWSS 面积占比 = `cycle.fields.tawss.area_frac.low`（< 0.4 Pa）、TAWSS p99 = `.p99`；OSI 均值 = `cycle.fields.osi.mean`、OSI > 0.1 占比 = `area_frac.above_t0`、OSI > 0.3 占比 = `area_frac.above_t2`；滞留区 = `cycle.stagnation.area_frac` / `area_mm2`，主要分支 = `stagnation.per_branch` 中 `area_mm2` 最大者。没有 `summary.cycle` 的任务（X5D、PF6/VF6）一切照旧。

- **发现列表（`analysis.py`，新 kind，追加在既有条目之后，既有条目顺序与内容不变）**：
  `stagnation_cluster`（TAWSS < 0.4 ∧ OSI > 0.1 的连通簇，连接半径与最小点数同低 WSS 簇；按面积取前 3；`value` = 面积 cm²，`units:"cm²"`，附 `tawss_mean_pa / osi_mean / area_mm2 / branch / xyz_mm`（簇内 OSI 最大点或质心最近点）/ `point_indices`，`label` 如「主动脉滞留区」，面积最大者 `severity:"attention"`，其余 `"info"`）；
  `high_osi_cluster`（OSI > 0.3 的连通簇，前 3；`value` = 簇内最大 OSI，`units:"1"`，`label`「左髂总高 OSI 区」，`severity:"info"`）。`definition` 用中文一句写清口径。入口：`build_findings(..., cycle={"tawss": array, "osi": array})` 之类的可选参数，由 `pipeline.stage_b_wall` 与 `rebuild_report.rebuild_wall` 传入；缺省时输出逐位不变。
- **沿程曲线（`summary.profiles.branches[*]`）**：有周期量时每分支加 `tawss: {mean_pa:[…], min_pa:[…]}` 与 `osi: {mean:[…], p90:[…]}`（与 `wss` 同分箱，空箱 `null`）。
- **自动结论（`narrative.py`）**：有周期量时在 WSS 句之后加一句：「周期平均 TAWSS 均值 0.66 Pa，低 TAWSS（< 0.4 Pa）区占壁面 65%，以主动脉为主；OSI > 0.1 占 57%；滞留区（TAWSS < 0.4 Pa 且 OSI > 0.1）占 46%（约 155 cm²），主要位于主动脉。」英文同义；免责句改为「以上为收缩期峰值 WSS 与单周期 TAWSS / OSI 预测的参考描述，非诊断结论。」（X5D / PF6 的句子逐字不变）。
- **质量文案（`quality.py`，`quality` 是黄金比对键）**：理由文字按实际模型数：5 个模型时逐字保持「五模型…」，其他数量写「N 个模型…」。一页纸 / 工作台的「集成质量」说明同样按模型数。
- **一页纸**：关键数字卡加「TAWSS 均值 / 低 TAWSS 占比」「OSI 均值 / OSI > 0.1 占比」「滞留区面积」；局限性声明按字段生成（有周期量时不再写「没有 TAWSS、OSI 等周期量」，改为「峰值 WSS 为固定收缩期帧；TAWSS / OSI 为单周期（0.8 s、80 帧）积分量的直接回归预测，不是逐帧推演」）；分支表加 TAWSS 均值、OSI 均值、滞留区占比列。
- **汇总表 / 队列（`export_table.py`）**：新列 `tawss_mean_pa, tawss_p99_pa, tawss_low_frac, osi_mean, osi_high_frac`（> 0.1）`, osi_very_high_frac`（> 0.3）`, stagnation_frac, stagnation_area_cm2`，放在壁面列之后；`population_blocks` 不变（M1 无人群参照）。
- **比较（`comparison.py`）**：两侧都有 `cycle` 且协议一致时，标量表加 TAWSS 均值 / p99、OSI 均值、滞留区占比的差值；只一侧有时显示单侧值、差值留空并注明。
- **M1 参考侧车**：M1 与 X5D_v51 同为 v5.1 train136 → 用 `build_reference_profiles.py` 为 `M1_3head_3seed_20260922` 生成**只含几何范围**的 `reference.json`（人群分位 `population.status:"unknown"`，注明「M1 无 CV3 折外预测」），使几何越界提示对 M1 生效；不改发布包指纹（侧车按 release id 绑定，见 §7）。
- **任务记录**：W2 在 `jobs.py` 把 `cycle`（原样，小于 10 KB）与 `findings_top`（`summary.findings.items` 前 5 条，去掉 `point_indices`）加入 `job["summary"]`，并在加载时为已完成、缺这两个键的旧任务从 `summary.json` 补齐（只读 summary，不改数值）。

## 19.3 患者随访时间线 `GET /api/patients/<patient_id>/timeline[?all=1]`（W2 实现 `timeline.py` + 路由，W3 消费，W1 一页纸可选消费）

同一 owner（管理员 `all=1` 全部）、`patient_id` 完全相等（去首尾空白）、状态 `done` 的任务。**一次扫描 = 一个 `input_sha256`**（同一几何换发布包的多次运行合并为一个扫描点）。

```json
{"patient_id": "P-001", "n_scans": 2,
 "scans": [{"input_sha256": "ad23…", "date": "2025-03-01", "date_source": "scan_date", "scan_label": "基线", "case_id": "CASE-001",
            "jobs": [{"job_id": "…", "release_id": "X5D_v51_5seed_20260916", "release_label": "壁面 WSS", "family": "wall", "review": "reviewed"}],
            "geometry": {"max_diameter_mm": 52.1, "sac_present": true, "sac_length_mm": 70.2, "sac_volume_ml": 96.3, "neck_diameter_mm": 19.1, "neck_length_mm": 22.0, "lumen_volume_ml": 158.2},
            "models": {"X5D_v51_5seed_20260916": {"wss_p99_pa": 16.6, "wss_low_frac": 0.34, "wss_high_frac": 0.02},
                       "M1_3head_3seed_20260922": {"wss_p99_pa": 17.0, "tawss_mean_pa": 0.66, "tawss_low_frac": 0.65, "osi_mean": 0.13, "osi_high_frac": 0.57, "stagnation_frac": 0.46},
                       "PF6_VF6_peak_3seed_20260920": {"speed_p99_m_s": 1.2, "aorta_delta_p_pa": 850.0}}}],
 "series": [{"key": "max_diameter_mm", "label": "最大直径", "units": "mm", "group": "geometry", "points": [{"input_sha256": "…", "date": "2025-03-01", "value": 52.1}]},
            {"key": "wss_p99_pa", "label": "WSS p99", "units": "Pa", "group": "model", "release_id": "X5D_v51_5seed_20260916", "points": […]}],
 "growth": {"max_diameter_mm": {"per_year": 3.2, "delta": 3.2, "days": 365, "from": {"date": "2025-03-01", "value": 52.1}, "to": {"date": "2026-03-01", "value": 55.3}, "basis": "first_last"},
            "sac_volume_ml": {…}},
 "notes": ["年增长率只在两次扫描都填写了扫描日期时计算。", "模型量只在同一发布包内连线；几何量与发布包无关。"]}
```

- `date` = `scan_date`（`date_source:"scan_date"`）否则任务创建日期（`"created_at"`）；排序按 `date` 再按创建时间。几何量取该扫描任一完成任务的 `summary.morphology`（与发布包无关）。`growth` 只在首末两次扫描都有 `scan_date` 且相隔 ≥ 30 天时给出，否则该键缺省并在 `notes` 说明；同时给 `last_two`（最近两次）一组，键名 `growth_recent`。
- 空结果（无权限或无任务）→ `{"patient_id":…, "n_scans":0, "scans":[], "series":[], "growth":{}, "notes":[…]}`（不区分「不存在」与「无权限」）。`patient_id` 最长 80 字符，URL 编码。
- 一页纸：服务端 `GET /api/jobs/<id>/onepage` 在该任务有 `patient_id` 且时间线 ≥ 2 次扫描时调用 `onepager.render_onepage(..., timeline=<上面的对象>)`，一页纸附「随访变化」小表（日期、最大直径、瘤体体积、本发布包的主指标、年增长率）。W1 实现参数与版面，W2 负责传入。

## 19.4 机构报告模板（主会话已建 `report_template.py`；W1 消费，W2 做接口与命令行，W3 可选做设置对话框）

`<jobs_root>/report_template.json`，键：`institution`、`department`、`report_title`（空 = 按族自动）、`footer_note`、`signature_lines`（默认 `["报告人","审阅人"]`，`[]` 不印）、`show_glossary`（`used` 默认 = 只列本页出现的术语 / `all` / `none`）、`appendix`（默认 true）。函数：`load(jobs_root)`（缺失或损坏 → 默认值，不抛错）、`load_for_job(job_dir)`（兼容 `.trash/<id>`）、`validate(data)`（中文 `ValueError`）、`save(jobs_root, data)`。

- W1：`onepager.render_onepage` 读取 `report_template.load_for_job(job_dir)`（`job_dir` 为空时用默认值）。
- W2：`GET /api/report-template` → `{"template": {...}, "editable": bool}`；`PUT /api/report-template`（body = 部分或全部键，与现值合并后 `validate`）只允许回环模式或管理员，否则 403；CLI `python -m wss_deploy.cli template show | set --institution … --department … --title … --footer … --signatures 报告人,审阅人 --glossary used|all|none --appendix on|off`。
- W3（可选）：设置菜单「报告模板」对话框（`editable=false` 时只读）。

## 19.5 一页纸重排（W1）

第 1 页（A4 纵向）只放：页眉（机构 / 科室 / 标题 / 审阅状态 / 日期）→ 病例身份一行（病例号 · 患者 · 扫描 · 发布包短名 · 生成时间）→「结论（参考）」→ 关键数字卡（壁面：p99、最大值、低 / 高 WSS 占比 + 周期量三卡；体场：原有）→ 配图（有 snapshots 时 2×2）→ 瘤体形态四卡 → 「重点发现」前 5 条（一行一条：编号、名称、分支、数值、判定）→ 签字栏（模板 `signature_lines`，每项「姓名 ____ 日期 ____」）。**附录**（`@media print` 另起一页，`appendix=false` 时不印）：分支统计与血管分支表合并为一张表（去掉重复的「几何」表）、发现详情（口径列）、输入检查、可信区域、限制声明、计算耗时、身份与哈希、术语（`show_glossary`；`used` 模式只列本页 `data-gloss` 键出现过的术语）。术语表解释列左对齐。页脚：固定声明 + `footer_note` + 发布包 / 任务号 / 页码（打印 CSS `@page` 计数）。形态可靠性长句压成「截面可靠性：N 站重定向，M 站不可靠（已排除）」一句，逐分支明细进附录。

## 19.6 运维（W2）

- **`python -m wss_deploy.cli service start|stop|restart|upgrade|status|logs|token [--jobs-root …]`**（新模块 `service.py`）：配置 `<jobs_root>/service.json`（host / port / device / `env`：如 `CUDA_VISIBLE_DEVICES`、`TZ`、`WSS_DEPLOY_RELEASE`；`start` 时的命令行参数写回）；共享模式令牌存 `<jobs_root>/.service_token`（0600，首次共享启动时生成，`service token` 打印），**不再需要从 `/proc/<pid>/environ` 取令牌**；PID 文件 `<jobs_root>/service.pid`（`{pid, started_at, host, port, argv, version}`）。`status` 在没有 PID 文件时扫描 `/proc/*/cmdline` 找 `wss_deploy.cli serve` 且端口一致的进程（**必须能接管当前手工启动的 master:8765 进程**：读取其 argv 与 `/proc/<pid>/environ` 中的 `WSS_DEPLOY_* / CUDA_VISIBLE_DEVICES / TZ`，首次 `restart` 原样沿用并写入 `service.json`；令牌从旧进程环境迁移进 `.service_token`）。`stop` = SIGTERM → 等 15 s（等待当前计算的任务？不等，任务会标记中断并可重试——在输出里提示有几个任务在计算）→ SIGKILL，只杀 cmdline 匹配的进程；`restart` = stop + start + 轮询 `/api/health` 至多 60 s，失败时打印日志尾部并退出码非 0。`logs [-n 80] [--follow]`。日志：`serve --log-file` 使用 `RotatingFileHandler`（20 MB × 5），时间带时区偏移；`service start` 默认 `<jobs_root>/server.log`。
- **`python -m wss_deploy.cli doctor [--json] [--jobs-root …]`**（新模块 `doctor.py`）：逐项 ✓/⚠/✗ 与建议：GNN 环境 torch / CUDA 可用与显卡名、`GNN_vmtk` 解释器与 `vmtk` 可导入、vessel_geom 路径、每个发布包合同与 `MANIFEST.sha256`（复用 `registry`）、参考侧车、`wss_features/CONTRACT.json` 源码哈希一致、`static/glossary.json` 与 `glossary.py` 同步、任务根可写、磁盘剩余（< 20 GB 警告）、服务是否在运行 / 健康、过期报告数（§19.7）、`users.json` 是否存在（共享模式建议启用用户名登录）。退出码：有 ✗ 为 1。
- **`GET /api/health`**：无会话时只返回 `{"ok": true, "version": <wss_deploy.__version__>}`；有会话时加 `started_at, uptime_s, queue:{queued, running, awaiting_input, awaiting_confirmation}, worker_alive, gpu:{available, name}, disk_free_gb, default_release, stale_reports`。版本号取 `wss_deploy.__version__`（本节撰写时为 0.12.0，每个发布版本随之递增，以 `__init__.py` 为准）。
- **`python -m wss_deploy.cli submit <stl…> [--server http://127.0.0.1:8765] [--release-id] [--units mm] [--patient-id] [--scan-label] [--scan-date] [--tags] [--notes] [--on-duplicate ask|reuse|force] [--user name]`**：走 HTTP 调用运行中的服务（`/api/session` + `/api/jobs/batch`，每批 ≤ 20），打印任务号与工作台链接 `…/#job=<id>`；回环模式免登录；共享模式用 `--user`（口令 `WSS_DEPLOY_PASSWORD` 或终端输入）或令牌（`WSS_DEPLOY_TOKEN` / `.service_token`）；令牌模式下提示「任务属于新会话，需在网页点『认领』或启用用户名登录」。`cli jobs list [--status …] [--json]` 经 HTTP 列出任务。

## 19.7 报告模板自动刷新（W2，`report_freshness.py`）

> S7（2026-10-01）起浏览器打开 `/api/jobs/<id>/report` 会跳到新工作区，不再触发打开时刷新；程序读取这个地址（非页面访问）时仍会刷新。过期报告由 `service upgrade` 和 `cli reports refresh` 刷新。`report.html` 现在是新工作区的数据来源，不再作为页面提供。

- `ui_fingerprint()` = sha256(`report.py`、`volume_report.py`、`static/report_common.js`、`static/volume_viewer.js`、`static/three.min.js`、`static/OrbitControls.js`、`static/glossary.json` 的字节)，进程内缓存、源文件 mtime 变化时重算。任务目录旁写 `report_ui.json {"fingerprint", "refreshed_at", "source": "auto|cli"}`。
- `GET /api/jobs/<id>/report`：任务 `done`、存在 `report.html`、`report_ui.json` 缺失或指纹不同 → 在该任务的锁内调用 `rebuild_report.refresh_ui_only(job_dir)`（约 0.3 s），成功写 `report_ui.json` 后返回新报告；失败记日志、返回原报告（不 500）。`WSS_DEPLOY_AUTO_REFRESH_REPORTS=0` 关闭。锁定（已审阅）任务同样刷新（只换呈现模板，数据与嵌入 JSON 原样）。
- `python -m wss_deploy.cli reports refresh [--job ID | --all] [--check]`：`--check` 只列出过期任务。以后改完报告界面**不再需要手工 rebuild + 重启服务**（Python 模块变更仍需重启，用 `cli service upgrade`：停服务 → 重建/刷新报告 → 启动；运行中的服务发现模板模块已变更时也只提示执行它）。

## 19.8 界面自查工具 `wss_deploy/devshot.py`（主会话已建）

`python -m wss_deploy.devshot sandbox --jobs <id,…> --dir <自己的 scratch>/jobs --port <端口>` 复制正式任务到沙箱并起回环 CPU 服务；`python -m wss_deploy.devshot <url> <out.png> [--width --height --wait --full --js "…" --after 秒 --marionette-port <端口>]` 截图，页面 JS 错误打印到 stderr 并以退出码 3 返回。Python 中可 `from wss_deploy.devshot import Browser`（`go / js / shot / frame / errors / resize`）。三维报告在软件渲染下首帧约 20–30 s（`--wait 25`）。沙箱里的报告改完模板后用 `from wss_deploy.rebuild_report import refresh_ui_only; refresh_ui_only(Path(job_dir))` 刷新。**端口分配**：W3 = HTTP 8801 / Marionette 2841–2849；W4 = 8802 / 2851–2859；W5 = 8803 / 2861–2869；主会话 8799 / 2830–2839。**绝不指向正式 `outputs/wss_deploy_jobs`，也不要重启正式 8765 服务**。

## 19.9 界面约定（W3 / W4 / W5）

> 本节描述的是经典报告与经典工作台，S7（2026-10-01）起已下线，只作历史记录；新工作区的约定见 `WORKSPACE_V2_CONTRACT.md`。

- **默认配色恢复彩虹**（用户 2026-09-20 明确偏好「配色要彩虹」；v0.11.2 改成 Viridis 未见用户要求），Viridis / Turbo / 蓝白红保留可选；已保存的偏好照旧优先。
- **默认视角 = 解剖前视并撑满视口**（两报告一致）：用 `FRAME.rotation`（或体场的等价坐标架）的 `front` 方向 + `fitView(margin 1.12)`；「复位视角」与 `0` 键回到它；没有坐标架的旧报告回退到现有逻辑再 `fitView`。
- **技术信息收纳**：页脚只留「审阅状态 · 发布包短名 · 生成时间（`localTime`）」，特征合同、run_identity、哈希、重建时间收进「技术信息」按钮弹层。`?` 术语按钮必须是 16–18 px 的圆（修复页脚与菜单标题里被拉成长椭圆的样式）。时间帧文字写人话：「收缩期峰值帧（约 0.21 s）」，不直接显示 `peak_systole` / `step 1162`（技术信息里保留）。
- **快捷键**（`installShortcuts`）：两报告 `1–6` = 前 / 后 / 左 / 右 / 上 / 下、`0` / `R` = 复位、`S` = 保存截图、`P` = 点选 / 探针开关、`L` = 自动标注开关、`?` = 帮助；壁面报告 `F` = 循环字段（WSS → TAWSS → OSI）；体场报告 `V` / `B` = 速度 / 压力、`T` = 循环显示页签、`X` = 点选定位截面。工作台：`/` = 聚焦搜索、`N` = 新建预测、`J` / `K` = 下 / 上一例、`Enter` / `O` = 打开三维报告、`G` = 一页纸、`?` = 帮助（列表获得焦点时 ↑↓ 同 J/K）。
- **工作台路由**：地址 `#job=<id>`（选中即更新，刷新 / 前进后退恢复）；列表分组顺序「需要处理（待确认输入 / 待确认出口 / 失败 / 中断）→ 计算中与排队 → 待审阅 → 已审阅（折叠）」。

## 20. §19 实现记录与偏差（2026-09-23 合并后）

- **W1**：`findings_wall(..., cycle=None)` → `_cycle_findings`；有周期量时 `findings.cycle_criteria` 新键（`service.pending_rebuilds` 用它判断旧 M1 结果是否需要完整重建）。一页纸第 1 页配图改为单行（1–4 张；更多进附录），2×2 会超出 A4；页码为静态「第 1 页 / 附录」（Firefox 不支持 `@page` 计数）；severity `info` 在一页纸按 kind 显示「几何」或「参考」。`build_reference_profiles.collect(population=False)` + `--population-note`；M1 侧车 `profile_id = geometry-train136-2026-09-22`。`quality.model_count_phrase`：5 个或未知 → 「五模型」原文。
- **W2**：`growth_recent` 只在 ≥ 3 次扫描时给出（两次时与首末相同）；时间线多 `schema_version`、`growth.from/to.input_sha256`、`jobs[].created_at`，M1 模型量多 `wss_low_frac / wss_high_frac`；health 多 `queue_mine / auto_refresh_reports / shared / login`、`gpu.count / device`，`stale_reports` 缓存 60 s；报告刷新时同时持有该任务锁与 manager 全局锁（防审阅并发覆盖，代价是模板更新后每份报告首次打开时其他请求等约 0.5–3 s——`service upgrade` 在停服务期间预先刷新，避免这一等待）；stdout 进 `server.console.log`。列表 / 详情 / `cases.runs[]` 带 `family_label`（「壁面 WSS」「WSS + TAWSS + OSI」「压力 + 速度体场」）、`release_short`、`has_cycle`；主会话合并时补 `finished_at`（done 时的完成时间）与旧格式任务（仅 `peak / wss_field_pa`）归「壁面 WSS」；`findings_top` 改用 `onepager.top_findings`（每类先取一条、驳回项不出现）。主会话新增 `service upgrade`（adopt/stop → 重建 → 刷新 → start，失败也启动）与 `--rebuild / --no-auto-rebuild`。
- **W3**：概览计数与快速筛选只看最近 100 个任务；「近 7 天完成 / 最近完成」优先 `finished_at`；`computeTiming` 本次尝试 > 0 才拆分，记录为 0 的旧任务回退 summary 总时长；确认页宽屏三维 `clamp(420px, calc(100vh - 300px), 70vh)`、`fitView`、OrbitControls 创建前设 up；缩略图无网格且无三维中心线时画二维中心线示意；可选「报告模板」对话框已做。
- **W4**：P = 锁定悬停点探针 / 关闭探针（壁面报告无单独点选模式）；视图状态 `field` 记录当前字段（并排同步会带过去，对侧不支持则忽略）；等值线 / 最高 1% 高亮 / 区域统计 / 展开图随当前字段；紧凑布局下页脚隐藏，技术信息从「统计与口径 → 查看计算过程」进入；`cycle` 早已在 meta 中，`build_html` 未改。
- **W5**：OrbitControls 在创建时固定 `camera.up` 为旋转轴——先设头向再建控件、上视 / 下视时重建（W4 同法）；「壁面压力」页签在速度模式被误禁用是 bug 已修（切页签自动切物理量，真缺数据才禁用并悬停说明）；`formatSvgTicks` 让共享库 `colorbarSVG` 的刻度与 `formatValue` 一致。
- **待上移共享库**：`releaseShort`、横幅文案（`referenceLabel / warningItems`）、`createdIso / withOffset`、`humanFrame`、`unitText`、按 up 重建 OrbitControls、`formatSvgTicks`（或 `colorbarSVG` 支持 `format`）、工作台 `pillSprite`。

---

# v0.12.2 契约 §21（2026-09-23 下午：进度与剩余时间 + 标签防重叠；算子提速已由主会话完成）

主会话已完成并上线的算子提速（逐位相同或落在原有运行间抖动内，见 README v0.12.2）：`wss_features` 的重采样 / 曲率 / 缠绕数，`knn_memo.py`（集成成员共用 kNN 图），`streamlines.ball_certified_inside`，发布包预加载。本轮只做界面与估计，不改任何预测数值；文件归属同 §19.0。

## 21.1 共享库 `declutterLabels(items, {padding=4, maxShift=120, step=10, bounds?, hideOverflow?})`（主会话已加，node 测试覆盖）

`items=[{x, y, w, h, priority?}]`（锚点 = 标签中心想放的位置，CSS px）→ `[{x, y, moved, hidden}]`（按输入顺序）。高优先级先放，冲突时在锚点周围按环尝试（上 / 下 / 左右 / 对角），`moved` 为真时调用方画引线回锚点；`hideOverflow` 时放不下的低优先级标签 `hidden`。确定性（同输入同输出）。

## 21.2 剩余时间估计（W2 后端 → W3 前端）

任务快照（列表与详情）在 `queued / running / awaiting_*` 时带 `eta`：

```json
{"eta": {"basis": "history|default", "n_history": 6, "faces": 23184, "family": "wall",
         "current": "features",
         "stages": [{"key": "ingest", "label": "检查输入", "expected_s": 0.2, "state": "done", "elapsed_s": 0.1},
                    {"key": "centerline", "label": "提取中心线", "expected_s": 3.5, "state": "done", "elapsed_s": 3.3},
                    {"key": "smooth_resample", "label": "平滑与重采样", "expected_s": 4.0, "state": "done", "elapsed_s": 4.2},
                    {"key": "features", "label": "几何特征", "expected_s": 3.0, "state": "running", "elapsed_s": 1.1},
                    {"key": "inference", "label": "模型推理", "expected_s": 7.0, "state": "pending"},
                    {"key": "morphology", "label": "形态测量", "expected_s": 1.8, "state": "pending"},
                    {"key": "export", "label": "报告与导出", "expected_s": 1.0, "state": "pending"}],
         "remaining_s": 11.0, "updated_at": "…"}}
```

- 阶段按族：A 段 `ingest, centerline`；壁面 B 段 `smooth_resample, features, inference, metrics, morphology, export`；体场 B 段 `smooth_resample, volume_features, inference, streamlines, morphology, export`（`summary.timing_s` 的键 `inference_5_models / inference_volume / metrics_and_interpolation / streamlines_and_interpolation / metrics_and_export` 归并到上面的短键）。`awaiting_confirmation` 时只给 B 段预计总时长（`remaining_s` = B 段合计，用于「确认后约 N 秒出结果」）。
- 估计：输入面片数 `f`（`input_check.faces`）。历史 = 已完成、同族、且 `summary.feature_contract.source_hash` 等于当前 `wss_features.contract()` 的任务（只用提速后的真实耗时），按 `seconds / (f/10k)` 取中位（`centerline` 用 `(f/10k)^1.4` 归一，VMTK 超线性）；少于 3 例时用下列默认值在两点间按面片数线性插值、下限为小例值的一半（2026-09-23 实测，GPU 忙时）：

  | 阶段 | 2.3 万面（LV） | 27.3 万面（LIU） |
  |---|---|---|
  | ingest | 0.2 | 6.1 |
  | centerline | 3.4 | 65 |
  | smooth_resample | 4.2 | 12.8 |
  | features（壁面） | 2.8 | 8.0 |
  | volume_features | 8.9 | 14.1 |
  | inference（壁面 / 体场） | 7.0 / 3.2 | 9.0 / 3.2 |
  | metrics | 0.2 | 0.5 |
  | streamlines | 3.5 | 6.7 |
  | morphology | 1.7 | 14.0 |
  | export | 0.8 | 1.1 |

- 运行中：已完成阶段用实际耗时，当前阶段 `elapsed_s` 取自阶段开始时间（进度回调已有 phase，W2 在 `progress()` 时记录 `phase_started_ts`），`remaining_s = max(0, expected_cur − elapsed_cur) + Σ pending`；超时时当前阶段剩余按 `0.2 × expected` 递减显示，不出现负数。事件流（SSE）照常推送快照即可。
- 排队中：`queue_ahead_s` = 前面任务的剩余合计（估计）。

## 21.3 界面（W3）

详情页运行中：分阶段进度条（每阶段一格，宽度按 `expected_s` 比例，已完成实色、当前阶段按 `elapsed/expected` 填充并带动效、待做浅色），下方「预计还需约 N 秒」（`formatDuration`：< 10 s 显示「几秒」，< 60 s 按 5 s 取整，否则「约 N 分 M 秒」），阶段名悬停显示预计 / 实际秒数；待确认出口时在确认按钮旁写「确认后约 N 秒出结果」；列表行运行中显示细进度条与剩余时间；排队中显示「前面 K 个，约 N 秒后开始」。`eta` 缺失（旧服务）时保持现状。

## 21.4 报告（W4 壁面 / W5 体场）

自动标注（发现 F1…、分支名、最大直径）每帧投影后调用 `declutterLabels`（相机变化时经 `requestAnimationFrame` 节流，静止时不重复计算），优先级：attention 发现 > info 发现 > 最大直径 > 分支名；`moved` 画细引线（1 px，半透明）回锚点；紧凑布局或视口 < 900 px 时 `hideOverflow:true`（被隐藏的发现仍在发现列表里）。导图 / 拼图 / 一页纸配图合成标签时用同一布局结果（离屏渲染按目标分辨率重新计算）。

## 22. v0.13 派生周期指标 RRT / ECAP、阈值跟随字段、细分色标、报告四标签（2026-09-23 晚）

### 22.1 派生指标
- 定义（`cycle_fields.derive_indices`）：`RRT = 1 / ((1 − 2·OSI)·TAWSS)`、`ECAP = OSI / TAWSS`，单位 `1/Pa`（界面写 Pa⁻¹）；TAWSS 取 `max(TAWSS, 0.01 Pa)`，`1 − 2·OSI` 取 `max(·, 0.01)`，OSI 先截到 [0, 0.5]；输入 NaN（未覆盖顶点）输出 NaN。
- 条件：`extra_fields` 同时含 `tawss` 与 `osi` 时由 `with_derived` 追加 `rrt` / `ecap`（`derived=True`）；单头发布包输出逐位不变。
- 描述符：`fields.rrt = {units:'1/Pa', array_key:'rrt_per_pa', source:'derived', derived_from:['tawss','osi'], definition, display:{thresholds:[5,10,20], log_scale:true, p99, max, cycle, threshold_direction:'above'}}`；`ecap` 同构（`ecap_per_pa`，阈值 1.4 / 2.8 / 4.2）。`display.threshold_direction` 新增到所有周期字段（TAWSS 为 `below`，OSI / RRT / ECAP 为 `above`）。
- 统计：`cycle.fields.rrt|ecap` 与 OSI 同结构（`area_frac.above_t0..t2`、`per_branch.frac_above_t0`），同步进 `results.statistics`；`cycle.definition` 增 `rrt` / `ecap` 两条文字定义。
- 沿程：`profiles.branches[].rrt|ecap = {mean:[…], p90:[…]}`。
- 导出：`points_wss.csv` 末尾追加 `rrt_per_pa,ecap_per_pa`（`%.4f`）；`wall_wss.vtp` 增同名点数组（顶点值 = 对插值后的 TAWSS / OSI 顶点值套公式）。**field.npz 不存派生数组。**
- 旧任务：`rebuild_report` 完整重建时补齐以上各项（CSV 原有列逐字节保留，重复重建只替换派生列）；`--ui-only` 刷新不动嵌入数据，查看器在浏览器端用嵌入的 TAWSS / OSI 数组现算（`source:'derived_in_viewer'`，同公式同下限）。

### 22.2 报告查看器
- 阈值：三个输入跟随当前着色字段；WSS 仍写 `thresholds_pa`，其余字段覆盖值写视图状态可选键 `field_thresholds: {tawss|osi|rrt|ecap: [t0,t1,t2]}`（基准单位；无效或未知字段忽略；与默认值相同则不记）。阈值与 summary 不同时统计卡从点数组重算（含 ≥ 10 点的分支）。
- 色标：`WssReportCommon.colorbarTicks({min,max,log,floor,bands,maxLabels=11,target=8,minGap})` → `[{f,v,major,nice}]`；`tickLabel`（取整刻度按原样、端点三位有效数字）；`spreadLabels(items,minGap)`；`colorbarSVG` 新增可选 `ticks:'fine'` 与 `thresholds:[{v,label}]`，缺省行为不变（体场报告仍用旧刻度）。分段档位 `0,4,6,8,10,12,16,20`。
- 菜单：`PANES = {view:[menu-display,menu-stats], measure:[menu-measure,menu-profiles,menu-region], review:[menu-findings,menu-annot], export:[menu-view,menu-presets]}`；卡片 id 不变；一个标签内一次只开一张卡；视图状态 `ui.menu` 打开对应卡并切标签，新增 `ui.tab`。


## 23. v0.14 四维度优化：安全、速度、可运维、界面快赢（2026-09-24）

用户裁定范围见 `README.md` v0.14 节。五路并行的共享契约如下（合并后全部落地）。

### 23.1 错误类型 `errors.py`
- `PipelineError(user_message, *, admin_detail=None, retryable=None, category=None, retry_hint=None)`；子类 `InputGeometryError`（input_geometry，不可重试）、`ToolchainError`（toolchain，不可重试，`admin_detail` 放工具 stderr 尾）、`ResourceError`（resource，可重试；CUDA 显存不足带 `retry_hint="cpu"`）；`classify(exc)` 把 torch OOM / TimeoutExpired / MemoryError / ENOSPC / ImportError 映射为上述类型，其余返回 None。
- `job["error"]` = `{message, diagnostic_id, category, retryable, [admin_detail], [retry_hint]}`；普通 `ValueError` 仍原文显示（category internal、retryable False）；未知异常用通用文案、retryable True、原始文本进 `admin_detail`。`ingest.STLInputError` 同时继承 InputGeometryError 与 ValueError 以兼容旧处理。
- HTTP 层 `_serialize` 在共享模式对非管理员剥除任意深度的 `admin_detail`（含事件流与 `device_fallback` 事件）；本机模式保留。前端 `retryable === false` 时隐藏「重试」，`admin_detail` 存在即在「技术细节」折叠显示。

### 23.2 任务记录与预计算
- `job.json` `schema_version = "wss-deploy.job/v2"`：不再内嵌 `a.preview` / `a.proposal.preview_polylines`（只在 `stage_a.json`）；紧凑 JSON；进度事件（`version=False`）原子写不 fsync，状态变化 fsync；`MIGRATIONS=[(v1,…),(v2,…)]` 懒迁移（v1 预览与 stage_a.json 逐字相同才移除）。`JobManager.geometry()` / `stage_a(job_id)` 按需读盘。
- `pipeline.precompute_geometry_cache(job_dir, job, *, release=None, cancel_event=None) -> {ok|cancelled|skipped|error, steps:[mesh,resample,point_geometry,morphology], timing_s:{smooth_resample,features,morphology}, seconds}`：永不抛异常；缓存条目 `<job>/geometry_cache/<kind>-<key>.npz`，键 = 输入字节 + dtype/shape + 参数 + wss_features 哈希 + 代码源哈希；`stage_b` 与预计算共用每任务锁。任务管理器在 `awaiting_confirmation` 时单线程后台调度（`_release_for(record, load=False)`，不加载权重）、stage B 让路、取消 / 删除 / 停服务即取消；重跑 / 重试复制 `geometry_cache/`；`WSS_DEPLOY_PRECOMPUTE=0` 关闭。
- summary 新增 `geometry_cache = {enabled, reused:[kinds], computed:[kinds]}`、`inference_threads`、`timing_s.precompute`（任务侧 `timing.precompute_s`）；ETA 把命中缓存的运行剔出阶段历史，`apply_precompute` 覆盖率 smooth_resample / morphology 1.0、features 2/3。
- 事件动作新增 `restart_requeued`、`drain_requeued`、`device_fallback`（`reason` 只放用户文案，原始文本在 `admin_detail`）、`precompute_{done,cancelled,skipped,failed,incomplete}`、`analysis_rebuilt`；`_event(actor=)` 与 `wss_deploy.audit` 每条持久变化一行（会话 owner 仅 `h:<sha12>`）。

### 23.3 溯源、健康与服务
- `schema.code_provenance()` / `summary_provenance()`：`analysis_version`（"2026-09-24"）、`deploy_version`（`__version__`）、`git_describe`（`--always --dirty --long`）、`git_dirty`、`code_source_hash`（wss_deploy/*.py + AST 扫出的 training_wss_min 模块）。壁面 / 体场 stage B 与 `rebuild_report._finish` 写 summary 前并入；run_manifest 顶层同名键。`regress.py` 跳过这五个键（`PROVENANCE_KEYS`），`SUMMARY_KEYS` 加 `cycle`，不加 `fields`。
- `JobManager.health()` → `{ok, checks:{workers, releases, jobs_root, disk, vmtk, gpu}, cached_at}`；硬检查 = 前五项；`registry.preload_state = {status: running|done|failed|disabled, running, loaded:[ids], failed:{id: msg}}`。`/api/health` 未登录只返回 `{ok, version}`，登录后带 checks；`/api/ready` 不健康 503；`service.wait_healthy` 用 ready。
- 单写者锁 `<jobs_root>/.service.lock`（flock，JSON 记 pid / purpose / 脱敏 argv）：`serve` 持有；`jobs claim`、`reports refresh`、`python -m wss_deploy.rebuild_report`、第二个 `serve` 被拒并报 PID（`--force` 越过）；`JobManager(offline=True)` 不标中断、不清回收站、不 `start()`。
- `service upgrade`：预检（子进程 import server/jobs/pipeline/families/report + `doctor --skip service`）→ 接管 / `--drain [s]`（`.drain.json`）/ 停止 → `queue_maintenance()` → 启动 → `wait_maintenance()`；重建与报告刷新由新服务在 stage-B 锁内后台执行，期间该任务编辑返回 409，结果写 `maintenance_result.json`。`serve --token-file`；`redact_argv` 用于 service.json / status / adopt。
- 会话：行记 `token_fingerprint` 或 `credential_generation`（users.json 每用户整数，passwd / disable / set_role / API 改口令时 +1）；lookup 时不匹配、用户禁用或角色变化即失效；`legacy_token_allowed()` 为假时旧令牌行失效（`lookup(for_claim=True)` 仅供认领提示）；空闲 `WSS_DEPLOY_SESSION_IDLE_HOURS`（12）+ 7 天；每 256 次或 10 分钟清扫；本机模式无 Cookie 的会话只在内存（pending ≤ 1000、10 分钟）。认领：仅 `previous.username is None` 的 owner 可携带；API 拒绝认领已注册用户名。

### 23.4 HTTP 层
- 上传：`stream_multipart()` 逐段进 `SpooledTemporaryFile`（8 MiB 阈值，落盘在 `<jobs_root>/.tmp`），以 `content_file=` 交给 `JobManager.create / create_batch`（`_stage_upload` 分块复制、边算 sha256 与面数、暂存 `.tmp/upload_<hex>.stl` 后改名；有 `seek` 即回卷）；客户端 `metadata_json` 中的 `content` / `content_path` / `content_file` 一律丢弃。限额：并发上传 2、每 owner 排队 + 计算 20、每会话 SSE 6、登录每 IP 每分钟 10、打包并发 1 且 ≤ 2 GiB（落盘 `.tmp` 后发送）。超限 429 / 413，body `{"error":{"message":…}, active_jobs?, max_active_jobs?}`。
- 响应头：工作台 / 静态 / API `script-src 'self'`；三维报告 `'self' 'unsafe-inline'`；一页纸 `'unsafe-hashes' 'sha256-<window.print()>'`（`ONEPAGE_HANDLER_HASH`，改按钮文本须同步）；无 `Server` 头。gzip（> 8 KiB，级别 6，文件级 LRU 256 MiB）；ETag / 304：静态 `private, max-age=0, must-revalidate`，报告 / 一页纸 / 几何 `private, no-cache`，API `no-store`。
- 日志：`wss_deploy.access`（`ip method path status bytes ms`，无查询串，`/api/patients/<id>/…` 脱敏）→ `access.log`（20 MB × 5）；`wss_deploy.audit` → `server.log`。`report_freshness.ensure_fresh` 在 `<job>/.report_ui_stage_*` 暂存渲染，锁内只校验 + `os.replace`；无嵌入数组的页面直接判失败。`<jobs_root>/.tmp` 与 `.report_ui_stage_*` 不被任何扫描当作任务。

### 23.5 报告与比较页
- 视图状态 `range` 可选键：`field`（固定上限只作用于该字段）、`shared`（比较页设置，不存偏好）、`case_max`；报告在重着色 / 可见性 / 剖切变化后向父页 post `wss-view:changed {family, run_identity}`。比较页两侧就绪、勾选同步或每次显示同步后调用 `WB.sharedDisplayRange()`（两例 p99 较大者）推送到两侧；字段 / 族不同或体场报告时显示提示。
- 色表唯一来源 `report_common.js` `PALETTES` / `paletteRGB` / `colormapTables`（彩虹 10 色标不变；turbo / viridis 33 色标；bwr RdBu-7）。
- 嵌入数据：`build_html(embed_derived=False)` 默认不嵌 RRT / ECAP（描述符仍声明，查看器现算）；分支编号 `ms/ps/cs` 在 0–255 内为 uint8，`arrays.dt[key]="u8"`；`refresh_ui_only` 按 `dt` 解码，旧页面按 int32。report.html 原子写。
- 前端：`TIMER_NAMES.precompute`；`geometry_cache.reused` 命中的阶段标「已预计算」；`eta.precomputed` 阶段斜纹；`EVENT_TEXT` / `eventText(action)`（未知动作原名）；`limitErrorText(status, body)` 生成 429 / 413 文案。
- **v0.14.1**：`JobManager.claim_owner(old_owner=None, new_owner)` 迁移没有 owner 的命令行任务；`cli jobs claim --owner none`。令牌模式没有认领通道：升级或会话过期后令牌用户会拿到新 owner，正式部署应使用用户名登录（users.json 存在即启用）。

## 24. v0.15 上线前加固：安全、可运维、前端与操作性（2026-09-26）

范围与裁定见 `README.md` v0.15 节。三路并行（S 安全 / O 可运维 / F 前端）+ 主会话合并；精度链路未动。

### 24.1 网络层开关（server.py，默认全关）
- `WSS_DEPLOY_TRUST_PROXY=1`：仅当 TCP 对端是 127.0.0.1 / ::1（含 IPv4-mapped）时信任 `X-Forwarded-For`（最右一个非回环地址，遇无法解析的一跳即停）、`X-Forwarded-Proto`（最后一个值，只认 http / https）、`X-Forwarded-Host`（作为 Origin 可接受的 host）；`Handler._client_ip()` 用于登录限速、审计与 access.log；`_allowed_origins()` 只接受代理报告的协议。`service_is_shared(host)` 在该开关下恒为真（即使绑回环也要求登录；无用户无令牌时 `serve` 拒绝启动）。
- `WSS_DEPLOY_COOKIE_SECURE=auto|1|0`：`Handler._cookie_secure()`；`auto` = 经可信代理且 `X-Forwarded-Proto: https`。判定为 https 的响应（JSON / 文件 / SSE）带 `Strict-Transport-Security: max-age=31536000`；非 https 从不发。
- `WSS_DEPLOY_UMASK`（八进制）：`apply_umask_from_env()` 在 `cli serve` 拿锁前与 `server.serve` 开头执行。与之无关：`atomic_json` 以 `O_EXCL` 0600 创建临时文件；`PrivateRotatingFileHandler`、console 日志、`deleted_jobs.jsonl` 一律 0600。
- 登录限速：`LoginThrottle` 两个桶（地址 `WSS_DEPLOY_LOGIN_RATE_PER_MIN` 默认 30；用户名 `WSS_DEPLOY_LOGIN_USER_RATE_PER_MIN` 默认 10，成功退还；用户名桶拒绝时退回地址令牌）+ users.py 锁定。429 体 `{"error": {...}, "retry_after": <秒>}` 且有 `Retry-After` 头；每键每分钟一条 `login_throttled` 审计。
- 请求解析：JSON `RecursionError` → 400；令牌 / CSRF 比较用 UTF-8 字节；multipart 文件名剔除 C0/C1 控制字符。
- 审计动作新增 `logout`、`template_updated`（`keys`）、`login_throttled`；CLI `_AuditLog` 把 `user add/passwd/role/disable`、`jobs claim` 写到 service.json 的 `log_file`（actor `cli:<系统用户>`）。
- `users.password_problem(password, username)`：长度 ≥ 8 之外拒绝纯数字、等于用户名、单字符重复、30 条常见口令；`UserStore.disable/set_role(keep_admin=)`，CLI 默认 `keep_admin=True`（`--force` 越过，单账号库豁免）。
- `/api/session` 新增 `max_batch_bytes`（256 MiB）、`max_batch_files`（20）；登录后的 `/api/health` 新增 `ui_build`（`static_build_id()`：STATIC_FILES 内容 sha256 前 12 位，按 stat 签名缓存；未登录响应仍只 `{ok, version}`）；非管理员的 `checks.jobs_root` 无 `path`。

### 24.2 时钟（clock.py）
- `configure(jobs_root, tz=)`、`now_iso()`、`now_local()`、`iso(ts)`、`strftime(fmt)`、`LogFormatter`、`describe(root)`。时区优先级 `WSS_DEPLOY_TZ` → 本次 `--env TZ=` → `service.json.env.TZ` → 系统；不读 shell `TZ`。无效名逐级回落并进 `errors`（doctor ⚠）。
- 走时钟的写入点：事件 `at`、`created_at/updated_at`、审阅 / 标注 / 快照时间、回收站三个时间、维护结果、预计算时间、service.json / service.pid / 写锁 / drain、`report_ui.json.refreshed_at`、`audit.report_rebuilt_at`、`narrative.generated_at`、打包说明、`eta.finish_at`、`pipeline._now()`（**stage_a / summary / quality_audit 的 `created_at` 改为 ISO 带偏移**）、任务 id（格式不变）、导出文件名、server.log / access.log 时间戳。epoch 秒字段与浏览器端 `toISOString()` 不变。`regress.SUMMARY_KEYS` 不含时间字段，黄金 6/6。

### 24.3 运维接口
- `doctor.run_checks` 状态集合 `{ok, warn, fail, info}`；新组 `exposure`（`bind_login`、`tls`、`secret_perms`）、`cache`（`derived_cache`）、`tz`；`logs` 组加 `log_rotation`；`releases` 组加 `release_pins`（对照 `env/releases.sha256`）；`disk_thresholds()` 读 `WSS_DEPLOY_MIN_FREE_GB="<fail>[,<warn>]"`（默认 5,20），`JobManager.health()` 共用；`doctor --host` 按将要使用的绑定评判，`service upgrade` 预检传入覆盖后的 `--host` / `--env`，✗ 抛 `ServiceError`。
- `service.status(root)` / `format_status` / `--json`：`login_mode`、`maintenance_summary`（`requested_at / requested_by / version` 自 `maintenance_result.json`）、`last_started`、`tz`、`ready`；`crashed / log_tail / console_tail / start_command`。`server_summary()` 同口径给 `/api/ready` 的管理员 `summary`。
- `housekeeping.du(root, top, json)`、`housekeeping.prune_cache(root, older_than_days, yes, force, dry_run)`：只删 `geometry_cache/` 与 `.tmp/` 中早于截止的文件，跳过 `queued / running / awaiting_*`，取写锁（运行中需 `--force`），记录 `prune_cache.jsonl`。
- `rehearse.run(root, jobs, count, keep, dir, preload, timeout)`：临时目录 + 临时 users.json（管理员 `rehearsal`）+ `WSS_DEPLOY_TRUST_PROXY=1`（回环也登录）+ `--device cpu` + 不预加载；步骤 `ready_anonymous → login → list/detail → report/onepage → ready_admin`；退出码 0/1。
- 崩溃：工作线程死亡 → CRITICAL + `health.checks.worker.failures`，不自动拉起；`service.install_crash_logging()`（`threading.excepthook`）；`ServiceHTTPServer.handle_error` 记 diagnostic_id。

### 24.4 前端（F）
- 会话过期：`request()` 对 `SESSION_CALLS`（`/api/session`、`/api/session/password`）的 401 不调用 `expireSession`；其余 401 → `#relogin-dialog`（顶层、不可 Esc）；`resumeSession(result)` 同用户只重开事件流 + 刷新列表与当前病例，换用户走 `startSession`。
- 登录：`wss-login-user` / `wss-login-remember`（localStorage，只记用户名）；`loginCooldown(seconds)` 读 429 体 `retry_after`。
- 连接：事件流 `onerror` → 顶栏琥珀并每 10 s 重开；`checkServiceVersion(version, build)`：`bootVersion = version[/ui_build]`，登录后首次学到 build 不提示；版本变 →「服务已更新到 vX」，同版本 build 变 →「服务的页面文件已更新」。
- 上传：`WB.checkUploadFiles(files, {maxBytes})`、`WB.uploadChunks(files, {maxFiles, maxBytes})`（来自 session 的 `max_batch_files` 与 `max_batch_bytes × 15/16`）、`WB.identifierIssue(text)`（控制字符 / 超长 = 错误，2–4 汉字 = 真名警告）、`WB.waitText(seconds)`；XHR 上传进度，无 XHR 退回 fetch。
- 报告：壁面 `#cbar` 与体场渐变条 `role=img` + `aria-label`（随字段）；体场 `#back-to-workbench` 同壁面规则；页脚无审阅记录写「待审阅」；`pointer:coarse` 控件 ≥ 44 px。
- devshot：`sandbox / suite / 单张 --login 用户:口令`（沙箱自带 users.json，共享 / 用户名模式）。

### 24.5 v0.15.1 结果文件脱敏与权限（2026-09-26 晚，用户裁定）
- `schema.redact_paths(value)`：`str(PROJECT_ROOT)` → `<project>`、`str(Path.home())` → `~`，对字符串内任意位置替换，递归 dict / list，不改非字符串；`model_release_metadata()` 的返回值整体经过它，所以 summary / run_manifest / quality_audit / job.json / report META 的 `model_release` 不含服务器绝对路径。`rebuild_report` 读入旧 summary 时对 `model_release` 脱敏，并在写 summary 后重写 `quality_audit.json` 的 `model_release`。
- `ANALYSIS_VERSION = "2026-09-26"`：`service.needs_rebuild` 据此把所有 `analysis_version < 2026-09-26` 的已完成任务列入 `pending_rebuilds`，`service upgrade` 启动新服务后在后台完整重建（不重跑模型），历史结果文件随之脱敏。`regress.SUMMARY_KEYS` 不含 `model_release`。
- 权限：正式 jobs_root 已 `chmod -R go-rwx`；上线命令带 `--env WSS_DEPLOY_UMASK=077`。未上 TLS / 反代（用户裁定先展示使用）；工作线程死亡不自动拉起（用户裁定）。

## 25. v0.15.9 截面均值 / 截面积分：点击点所在的垂直截面，积分到中心线那一点（2026-09-27）

用户要求（2026-09-27）：点击一个点，给出沿中心线该处截面的积分均值（TAWSS 等），积分到中心线这个点上；截面 = 过中心线点、垂直于中心线的平面（用户选定），壁面与体场都做。纯显示与读数，不改任何预测、summary 或导出文件。

### 25.1 站位与截面（`report_common.js`）
- `centerlineTangent(groups, pr)`：投影行 `row_local` 两侧相邻样本的弦方向（单位向量）；组按近端 → 远端排序，所以指向下游。壁面「管径」测量（§15.14）改为调用它，算式逐字相同。
- `stationSection({groups, point, vertices, faces, wallValues?})`：`projectToCenterline` → `{segment_id, branch, s_mm, s_from_root_mm, radius_mm, origin(投影点), tangent}`；`crossSection(origin, tangent, maxDist = max(4 r, 1))` 取包住中心线点的管腔轮廓（与管径测量相同的选环规则）。`found = false` 时 `reason ∈ {no_centerline, no_tangent, no_mesh, no_contour}`。`wallValues`（每网格顶点，可选）插值到轮廓线段。
- `ringElements(cs)`：轮廓里每段真实壁面的中点（世界坐标）与长度；闭合开口的合成直线段（边键 < 0）只计入 `gap_mm`。`lineMean(values, lengths)`：∮f dl / ∮dl 与环上最小 / 最大；非有限值与零长度不计权。

### 25.2 壁面截面均值（`sectionMeans`，report.py 探针）
- 每个线元取 `o.sample(中点)` 返回的预测点（报告页传 `CORE.nearestIndex(PV, ·, 3 × 点间距)`，与探针同一查找）的值；每个字段 `means[id] = lineMean(...)`，字段 = 页面全部字段（WSS、TAWSS、OSI、RRT、ECAP，有哪个算哪个）。线长加权使结果与点云密度无关。
- 显示（v0.15.10 起）：探针卡 = `.pc-head`（「探针」、分支 · 弧长 · 半径、记录、×）+ `.pc-table`（每字段一行 `data-field`：此点 / 截面均值 / 环上范围条，当前字段 `tr.on`，点行 = `setField`，`recolor` 时重绘）+ `.pc-foot`（「截面：垂直中心线，过 分支弧长 x mm 处 · 周长 L mm[ · 经过开口，缺口 g mm 不计 | · 轮廓不闭合，只平均切到的壁面]」、坐标、图例）；整卡文字在 `aria-label`（`probeSummary`）。3D：黑色管（半径 min(0.003 × 包围盒, 0.15 × 内切半径)，下限 0.15 mm）+ 0.45 倍白芯的截面环，中心线点黑 / 白靶心，探针点 → 中心线点连线；关闭卡片或按 P 关闭即清除。
- 记录：`values.section_<array_key>`（WSS 为 `section_wss_pa`）、`section_s_from_root_mm`、`section_perimeter_mm`（壁面长度）；探针记录表按需加「截面 字段」列；TSV / CSV 通过既有 `_probeColumns` 自动带出。

### 25.3 体场截面积分（`sectionIntegral`，volume_viewer.js 探针）
- 只对**固定**的探针计算（键 = 探针类型 : 下标 : 截面厚度，变了才重算）；悬停不算。
- 样本：`slabIndices(pts, is_wall, segments, plane, 厚度)`（厚度 = 截面厚度控件，默认 2 mm）中落在轮廓内（`pointInLoop`）的体内点。
- 面积平均：轮廓外接方框 64 × 64 格，轮廓内每格取最近样本的值（Voronoi 面积光栅化），均值 = 格均值 = Σ vᵢAᵢ / A；不施加壁面值。`supported` = 离最近样本 ≤ 3.2 × 中位样本间距的格占比（截面图「直接支撑」同一半径）。
- 读数：`area_mm2`（轮廓多边形面积）、`equivalent_diameter_mm`、`speed_mean_m_s`（⟨|u|⟩）、`normal_mean_m_s`（⟨u·n⟩，n = 中心线切向，顺流为正）、`flow_ml_s = normal_mean × area`（m/s × mm² = mL/s）、`pressure_mean_pa`（相对压）。轮廓不闭合（经过开口且无法直线封闭）→ `integrated = false`，不给面积量。
- 「看截面」（`#probe-slice`，有截面时才显示）：`pick` 基准、位置 50、俯仰 / 偏航 / 面内偏移 0，`pickPlane = {origin: 中心线点, normal: 切向}`，切到截面模式——与积分同一平面、同一厚度。
- 记录：`section_s_from_root_mm`、`section_area_mm2`、`section_speed_mean_m_s`、`section_normal_mean_m_s`、`section_flow_ml_s`、`section_pressure_mean_pa`。

### 25.4 选择依据与验证
- 面积平均方法按质量守恒选：三个真实体场任务（入口波形统一）各 35 站，Voronoi 下四出口支 Q 之和 116.7 / 114.7 / 115.4 mL/s，主动脉各站 95–137 / 108–125 / 106–133，分叉处主干 ≈ 两子支之和；截面热图的「IDW + 壁面 0 值」补全用作面积平均则出口和 92 / 86 / 84、稀疏处（LV 瘤颈 15–30 点）偏低约一半，故只保留作显示。样本普通平均受采样密度影响（近壁包围盒填充），个别站可到负值。
- 壁面：截面均值与沿程 2 mm 分箱均值（周长 ≤ 1.5 × 该支中位的站）中位相对差 WSS 4–6%、TAWSS 3–4%、OSI 6–7%。
- 测试：`test_report_common.py`（线积分与密度无关、站位 / 切向、无截面原因、合成边与线长加权）、`test_report.py`（点击 → 卡片 → 记录 → 关闭）、`test_volume_report.py`（Voronoi 面积平均对 Poiseuille 1/2 与密度无关、固定探针积分 / 记录 / 看截面 / 清除）。
- 已知限制：分叉口附近垂直截面会切进相邻分支（周长突增，3D 环可见）；体场点稀疏处个别站 Q 偏 ±20%；模型速度不受质量守恒约束。

## 26. v0.15.11 算子与缓存提速：组合任务缓存接力、体场几何缓存、集成设备输入、采样 / 流线 / 封口算子（2026-09-28）

依据 `training_wss_min/experiments/wss_deploy_timing_20260917/analysis_20260928/README.md`（09-28 审计）。不改任何模型、特征定义、采样密度、种子数或精度；精度两档沿用 09-24 裁定（几何 / 采样 / 特征 / 导出链 Tier A 逐位相同，只改浮点归约顺序的 Tier B 不超过原实现运行间抖动）。summary / field.npz / 报告字段不变，`ANALYSIS_VERSION` 不变。

### 26.1 组合任务缓存接力（`cache_handoff.py`，`jobs.JobManager._inherit_geometry_cache`）
- 触发：任何带 `source_job_id` 的任务（伴随、换包重跑、查重复用）在 B 段屏障之后、B 段函数之前调用一次。源任务的后台预计算状态为 running（且未被取消）时等待，最长 `PRECOMPUTE_WAIT_S`（300 s），进度文案「等待共享几何预计算完成」；pending 不等（B 段锁在手，它不会开始）；超时不取消源任务的预计算。等待与复制期间不持有 `JobManager.lock`。
- `copy_committed(root, source_id, target_id, *, valid)` → 复制条目数：只接受任务 id `[A-Za-z0-9_-]{1,80}`、源 ≠ 目标；目录逐级 `O_DIRECTORY | O_NOFOLLOW` 打开；只复制名字符合 `<kind>-<40 hex>.npz` 的常规文件，跳过符号链接、硬链接（`st_nlink ≠ 1`）、临时文件、子目录；目标已有同名条目（包括损坏的）一律不覆盖；写临时文件（0600）→ 复制前后 `fstat` 不变 → `os.link` 发布（目标存在即失败）；每 1 MiB 与发布前调用 `valid()`（归属一致、未取消、服务未停），失败即放弃且不留临时文件。任何 `OSError` 只丢掉这次优化。读取仍由 `GeometryCache.load` 按完整键校验，复制来的旧键条目只是一次正常未命中。
- 开关：随 `WSS_DEPLOY_GEOMETRY_CACHE`。

### 26.2 体场几何缓存（`volume_cache.py`）
- `build_volume_case_cached(wall, vertices, faces, atlas, input_features, *, cache, mapping, target, case_name, n_internal, seed)`：`stage_b_volume` 在 `GC.job_lock` 内调用；条目 `volume_case-<key>.npz`。载荷 = `(case, aux)` 的类型化树（dict / list / tuple / None / str / bool / int / float / ndarray / NumPy 标量；dict 键可为任意这些类型；object / structured 数组与其他对象拒绝），JSON 树 + 数组，无 pickle；另存 `__payload_hash__` 校验。
- 键 = schema + 类型化输入（壁面点、平滑网格顶点 / 面、atlas 的 table / columns / segments / semantic_of_segment / frame_n / frame_b / tree_rows / provenance 与 KD 树数据 / 索引 / 分区、特征列表、确认映射、target、case_name、n_internal、seed）+ 源哈希（volume_cache / volume_geometry / streamlines / geometry_cache / wss_features.{atlas,cloud,flowref,frame}）+ 特征程序哈希 + numpy / scipy / pyvista / vtk 版本。键不可构造（未知对象）→ 记 `errors`、按未命中直接计算；载荷损坏 / 校验失败 → 撤销命中记录、按未命中重算；缓存写失败不影响任务。
- `prune_geometry_cache` 清理未使用的 `volume_case` 条目（与 mesh / resample / pointgeom / morph / morphvol 一起）。

### 26.3 集成设备输入（`prepared_inference.PreparedInference`）
- 用法：`with shared_knn(), shared_inputs(), PreparedInference() as prepared:` 内 `prepared.predict(E.predict_case_norm, model, case, features, feat_stats, device, cfg=…, [return_all_channels=…])`；三个模型族（`wall_wss_v1`、`pf6_vf6_volume_v1`、`wall_cycle_multi_v1`）均已接入，结果字典加 `prepared_inputs = {calls, reused, built, fallbacks, peak_bytes, output_transfers}`（不写 summary）。
- 只接管：作用域内、开关开、`cfg.eval.fixed_support`、kwargs ⊆ {cfg, return_all_channels, frame_index}、预测函数解包后是 `training_wss_min.evaluate.predict_case_norm`、模型类型恰为 `PointNetPlusPlusRegressor` / `PointNetRegressor` 且 eval 模式；否则原样调用传入的预测函数（计 `fallbacks`）。CUDA OOM → 清空、`empty_cache`、回退原评估器。
- 块键 = blake2b(case 公开条目内容〔每个 case 字典每作用域哈希一次，顶层条目被替换即重哈希〕, `cfg.data`〔除 `feature_stats_path`，统计内容另作 `feat_stats` 入键〕, `cfg.eval`, 特征列表, feat_stats, 设备, frame)；块 = 支持点块 + 每个查询块（pos / x / batch / geometry / section / patch 张量）。保留预算 `WSS_DEPLOY_PREPARED_INPUTS_MB`（默认 1024 MiB，GPU 上再限空闲显存的 1/4）；保留的块按张量 `_version` 校验，被原地改写即重建。
- 构建：块全部可保留时，构建函数调用 `input_memo` 包装之下的原函数（其私有副本无人读取）；首个成员在第一块同步构建后，其余查询块在 ≤ 8 个线程上并行构建（每块是 case 的纯函数；第一块已填好 case 的 KD 树与全壁面特征表）。某块超预算后（`_overflow`），后续构建回到 `input_memo` 包装函数、串行。
- 输出：每成员整例一次回传 CPU（单例输出 > 64 MiB 时仍逐块回传）。
- 开关：`WSS_DEPLOY_PREPARED_INPUTS=0` 回到原评估器路径。

### 26.4 几何算子（结果逐位相同）
- `geometry.cap_records(pts, atlas, normals)`：`build_case` 只读封口记录，改为 `virtual_caps(atlas, median_spacing(p), p)` + 同字段记录，不再建完整 oriented cloud；`virtual_caps` 签名不符或无开口时回到原 `build_oriented_cloud`；`diag.spacing_mm` 在点已是 float64 时复用同一次 `median_spacing`。
- `volume_geometry.sample_internal_points`：VTK 精确测试判在内部的中心线采样点（`atlas.tree_rows` 去重）作锚点，经 `streamlines.ball_certified_inside` 包装 `contains`——距锚点小于 0.999 × 锚点到封闭面距离的查询直接判内，其余仍交 VTK；没有锚点时不包装。
- `volume_geometry._unique_edges`：`_boundary_loops` 的 `np.unique(axis=0)` 改为 int64 单键 `a·M + b`（`b < M`）一维去重，值 / 顺序 / 计数 / dtype 相同；空、负数或键会溢出时回到原调用。
- `streamlines.integrate_streamlines`：接受的端点已算出的 IDW 方向 / 速度 / 支撑作为下一步起点，不再重复 kNN / IDW；`inside()` 的调用序列（及 VTK 全局随机序列位置）完全不变；逐点 list 改为逐步数组 + 稳定排序拼接。

### 26.5 VTK 射线测试的随机性（约定）
- `vtkSelectEnclosedPoints.IsInsideSurface` 的射线方向取自 VTK 全局随机序列，服务进程里每个任务的射线取决于此前跑过的任务，所以「逐位相同」对 VTK 内判只在「结果对射线方向不敏感」的前提下成立；5 个真实几何约 72 万次查询在逐点（全局随机序列）与批量（`vtkRandomPool`）两种射线来源下无一处翻转，流线在 4 个 VTK 随机种子下输出不变。改动 VTK 调用次数或顺序（26.4 的锚点证书）按此约定验收：整例 `build_volume_case` 输出与旧实现逐位比较。
- 不采用：`vtkSMPTools` STDThread 后端（批量内判快约 15 倍，但后端是进程全局状态，会影响并发的 A 段 VTK / VMTK）；流线里省掉重复的 `inside()` 调用（可再省 0.35–0.75 s，但改变全局随机序列位置、无法证明逐位相同，留待用户裁定）。

## 27. v0.15.12 补全截面：只让壁面条件作用于贴壁一层，逐像素抗锯齿绘制（2026-09-29）

用户反馈「补全截面外沿太厚、颜色不均」。只改截面平面图的显示层（侧栏、放大图、系列拼图、一页纸截面图）；截面统计、截面积分（§25 Voronoi）和数值结果不变。

### 27.1 填充（`fillSection(points, values, boundary, bounds, nx, ny, inside, options)`）
- `boundary` 元素为壁面线段 `[x0, y0, x1, y1, value]`（`sliceGrid` 传真实交线段）或点 `[x, y, value]`（零长线段）；`value` 为 NaN 的段（开口处直线封闭段、壁面压力无支撑处）不施加条件，也不计入离壁距离。
- 体内场 v_int：只用体内预测点。每点带宽 h_i = max(0.5 × 中位最近邻间距, 0.6 × 到第 5 近邻的距离)；核为截断高斯 exp(−u²/2) − exp(−4.5)（u < 3），外加 3 倍带宽、权重 0.005 的宽核，以及 1e-9 × 全局均值项（无样本角落不为 NaN）。第二遍在样本处算残差，按 0.5 × h_i 的核加回，分母为 (1 + Σw₂⁴)^¼，结果限制在样本值域内。计算方式是逐样本把核叠加到需要的格子上。
- 壁面层：d = 格子到最近条件段的距离；δ = (Σw / Σ w·d_i⁻²)^½（附近样本离壁距离的软最小），上限 `SECTION_KERNEL.wallLayer` = 0.5 mm。d < δ 时 v = v_wall + (v_int − v_wall)·t(2 − t)，t = d/δ；否则 v = v_int。
- 返回：`values`（轮廓外 NaN）、`low`（最近体内点 > `directRadius` = 3.2 × 中位间距，定义同前，CSV `filled_from_wall` 即此位）、`filled` / `direct` / `directCells`，另有 `display`（`values` 加轮廓外两格：取最近壁面值，无条件处取 v_int）、`support`（到最近体内点距离）和 `directRadius`。
- 没有体内点时退回壁面段 8 近邻 IDW，全部标 low。

### 27.2 绘制（`paintSection`）
- 仅限补全模式。在轮廓外包盒内 `getImageData` → 逐像素写 → `putImageData`；数值从 `display` 双线性读取（`bilinearGrid`，NaN 角点剔除后重新归一），经 1024 级 `colorAtT` 色表上色，色带、对数、发散色标照常。
- 覆盖率：每像素行 4 条子扫描线，对轮廓线段做奇偶填充，得到水平方向的分数覆盖率，按覆盖率与底色混合。
- 淡色：`lowFade(support, directRadius)` = 0.28 × smoothstep((s − 0.8R)/(0.6R))，向白混合，连续变化。
- 没有 `getImageData` / `putImageData` 的上下文（Node 桩）画不透明格子，low 格混白 0.28；严格模式（关补全）照旧画格子。放大图悬停读数取 `display` 的双线性值，与像素一致。
