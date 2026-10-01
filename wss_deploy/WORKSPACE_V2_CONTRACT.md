# 新工作区 v2 与 C 线实施合同（2026-09-30）

> 依据：`docs/02-推进与变更/05-部署工具/前端重构_范围说明_2026-09-30.md` §10（最终裁定）与讨论稿 §13–§15。
> 本文件是五路并行开发的**唯一接口约定**。任何一路要偏离，先在最终报告里写清楚理由，不要私下改别人的文件。
> 分支 `wss-ui-v2`，工作副本 `/public/newhome/cy/Digital_twin/GNN_wssui_v2`（下称「副本」）。

---

## 0. 总规则（所有人必读）

1. **只在副本里改。** 线上服务从 `/public/newhome/cy/Digital_twin/GNN` 运行，静态文件按请求读盘，改那里等于改线上。**绝对不要**碰主目录的 `wss_deploy/`、`tests/`、`outputs/wss_deploy_jobs/`，不要重启或 upgrade 线上服务（:8765），不要执行 `service upgrade`。
2. **数值不变。** 模型、几何、采样、特征、`field.npz`、既有统计定义冻结（Tier A 逐位）。C 线只改文案、发现规则、叙述顺序和**新增**的分区统计。黄金回归（`wss_deploy/regress.py` 的 `SUMMARY_KEYS`：peak / wss_field_pa / per_branch / geometry / surface_statistics / quality / volume_statistics / murray_shares / caps / cloud / mapping / branch_names / cycle）里的键**一个字符都不能变**（包括 `quality.label` 字符串）。
3. **不提交。** 子智能体不 `git commit`、不 `git push`；由主会话合并提交。
4. **文件归属**见 §9。只改自己名下的文件；测试文件只改和自己改动直接相关的断言。需要别人名下的改动，写进最终报告的「需要其他路配合」一节。
5. **环境**：Python `/public/newhome/cy/.conda/envs/GNN/bin/python`，在副本根目录下 `PYTHONPATH=.` 运行；Node 在 PATH 上（v24）。不要用 GPU（`CUDA_VISIBLE_DEVICES=`）。
6. **测试**：全套 `PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests`（基线 776 通过 / 9 跳过，约 3.5 分钟）。交付前你负责的测试和全套都要通过。
7. **沙箱与截图**：`python -m wss_deploy.devshot sandbox --jobs <ids> --dir <你的草稿目录>/jobs --port <HTTP> --login admin:sandbox-pass`（默认从副本的 `outputs/wss_deploy_jobs/` 复制任务），截图 `python -m wss_deploy.devshot <url> <out.png> --wait 25 --marionette-port <M> --login admin:sandbox-pass`。每路专用端口见 §9。Python 改动要重启沙箱才生效；静态文件改动即时生效。用完关掉自己的沙箱进程。
8. **开发任务目录**（副本 `outputs/wss_deploy_jobs/`，owner 都是 `admin`；不要改它们，沙箱会复制）：

| 任务 id | 病例 | 发布包 | 说明 |
|---|---|---|---|
| 20260929_230442_a7273f1670e4 | LV_GUO_YOU | M1 三头 | 当前 schema；有 TAWSS/OSI/RRT/ECAP；companions 指向下一行 |
| 20260929_230450_d1428792b846 | LV_GUO_YOU | PF6/VF6 体场 | 当前 schema |
| 20260922_173705_27065942b465 | LV_GUO_YOU | M1 三头 | 09-22 旧任务重建 |
| 20260920_113510_0aa3ba155bfa | LV_GUO_YOU | PF6/VF6 体场 | 09-20 旧任务重建 |
| 20260920_144135_673ccd0e36b1 | LV_GUO_YOU | X5D 峰值 WSS | 有人群参照 p99 |
| 20260918_152113_e2e53937b25a | WANG_SHUN_WEN | X5D | 09-18 旧任务 |
| 20260918_154058_a72784880799 | PENG_JI_MING | X5D | 09-18 旧任务 |
| 20260918_162707_f22146842b96 | PENG_JI_MING | X5D | 同一输入的第二次运行 |

9. **现有合同**：`wss_deploy/ANALYSIS_CONTRACT.md`（接口请求/响应样例、视图状态 v1.1、审阅与标注、比较合同）照旧有效；本文件只增不改它。

---

## 1. 新路径与静态文件

- 工作区入口 `GET /v2/`（`/v2` 301 到 `/v2/`）→ `static/v2/index.html`，与 `/` 相同的会话处理，CSP `workbench`（**不允许内联脚本**；内联样式允许）。
- 静态文件 `GET /static/v2/<name>`：白名单 = `static/v2/bundle.json` 里列出的全部文件名（`scripts`、`styles`、`pages`、`assets`、`dev`）。目录扁平，不允许子目录。
- 前端路由用 hash（离线 `file://` 也能用）：`#/`（待办）、`#/job/<job_id>`，查询参数 `?v=<view>&f=<field>&cmp=<job_id>&q=<question>&bm=<bookmark_id>`。`view ∈ {wall, volume, compare, input}`，缺省按结果族。
- S7（第三期，2026-10-01）：旧工作台、比较页、旧报告页下线。
  - 旧地址 302 跳到新工作区：`/` → `/v2/`；`/compare?left&right` → 比较视图；`/api/jobs/<id>/report` 的页面访问 → `/v2/?job=<id>`。`ws_legacy.js` 再把经典的 `#job=`、`#view=` 转成新工作区的路由和视图状态。
  - 运维中心 `/ops`、工单 `/support` 保留为独立页面。新工作区里，头像菜单（管理员）有「运维中心」，帮助菜单有「反馈问题」。
  - 任务目录里的 `report.html` 照常生成：它是新工作区的数据来源（`v2_data.py`），只是不再作为页面提供。下载的数据包改放新版单文件离线页 `offline_report.html`。
- `static/v2/bundle.json` 已由主会话写好，**文件名和加载顺序以它为准**。不许新增或改名文件；需要新代码就放进你名下已有的文件。
- 第二期 S1（2026-09-30）：`legacy_scripts` 加入 `volume_viewer.js`（只用它的数值核心 `VolumeViewerCore`；页面上没有经典体场报告的 `wss-report-meta` / `volume-arrays` 时它在页面代码之前返回），`scripts` 加入 `ws_slice.js`；`/static/volume_viewer.js` 进 `STATIC_FILES`。

## 2. 数据接口（第 3 路实现）

所有 `/api/v2/*` 与现有 `/api/jobs/*` 用同一套会话、owner 隔离、管理员 `any_owner` 规则；写接口要求 `X-CSRF-Token`，和现有 POST 一致。

| 方法 | 路径 | 返回 |
|---|---|---|
| GET | `/api/v2/jobs/<id>/manifest` | §3 的 manifest JSON；仅 `status == done`，否则 409 `{"error","status"}`；带 ETag（`data_version`），支持 304 |
| GET | `/api/v2/jobs/<id>/arrays/<key>` | 原始小端字节，`application/octet-stream`；`<key>` 必须出现在 manifest.arrays；ETag = `data_version + key`；不 gzip |
| GET | `/api/v2/jobs/<id>/inputcheck` | 未完成或失败任务的输入检查：`{"status","stage","error","input_check"(含 quality 与 opening_geometry，缺则 null),"openings"(中心线开口，缺则 []),"mesh": {"vertices": b64 float32, "faces": b64 uint32} 或 null, "mapping_proposal"(若有)}`；网格可用输入 STL 抽取（显示用，≤ 60k 面，可抽稀），不写任务目录以外的地方 |
| POST | `/api/v2/jobs/<id>/offline` | body `{"hide_name": bool, "bookmarks": [...], "view": {...}}` → 单文件 HTML 下载（`Content-Disposition: attachment`，文件名 `WSS_<显示名或"病例">_<yyyymmdd>.html`） |
| GET | `/api/v2/model-cards` | `{"cards": {release_id: card}}`，覆盖注册表里的全部发布包（没有卡的给 `null`） |
| GET | `/v2/example` | 示例离线报告 `static/v2/example_report.html`（由主会话最后生成），CSP `report`；文件不存在时 404 |

**数据来源（阶段一）**：manifest 与数组从任务目录现有的 `report.html` 内嵌 JSON（`wss-report-meta` + `wss-report-arrays` 或 `volume-arrays`）读出——这正是旧查看器显示的数据，保证新旧读数一致。按 `(path, size, mtime_ns)` 缓存解析结果（LRU，最多 6 个任务）。`summary.json` 里的 `review`、`narrative` 以任务记录为准（审阅、编辑后的结论不在旧 report.html 里时要以 job/summary 为准，见 §3 `job` 块）。以后可以换成独立数据文件，manifest 合同不变。

**派生量**：RRT、ECAP 在服务端用 `cycle_fields` 现有公式从 TAWSS/OSI 算（点和顶点各一份），前端不复算。体场速度模长 `speed = |v|` 同理在服务端算。

## 3. manifest（`wss-deploy.v2-manifest/v1`）

```json
{
  "schema": "wss-deploy.v2-manifest/v1",
  "data_version": "sha256 前 16 位：report.html 的 size+mtime_ns、summary.json mtime、v2_data 代码版本",
  "job": {
    "id": "...", "status": "done", "display_name": "LV_GUO_YOU",
    "case_id": "LV_GUO_YOU", "patient_id": "", "scan_label": "", "scan_date": "",
    "created_at": "...", "input_sha256": "...",
    "review": {"status": "unreviewed", "by": "", "at": "", "note": "", "version": null},
    "companions": [{"release_id": "...", "job_id": "..."}],
    "narrative_edited": false
  },
  "result": {
    "run_identity": "...", "family": "wall",
    "release_id": "M1_3head_3seed_20260922",
    "display_name": "周期指标 TAWSS · OSI",
    "analysis_version": "2026-09-30", "deploy_version": "0.15.0",
    "model_frame": {"target": "peak_systole", "time_s": 0.21}
  },
  "units": {"length": "mm"},
  "frame": {
    "rotation": [[...]], "origin_mm": [...], "convention": "...",
    "orientation": {"left_right": "confirmed|inferred", "superior_inferior": "derived", "anterior_posterior": "inferred|unknown"},
    "direction_source": "unknown_stl"
  },
  "time": {"mode": "single_frame", "axis": [{"index": 0, "time_s": 0.21, "label": "peak_systole"}], "cycle": null},
  "geometry": {
    "display_mesh": {"vertices": "mv", "faces": "mf", "segment": "ms", "trust": "mt"},
    "points": {"xyz": "pv", "segment": "ps", "s_from_root_mm": "p_s", "theta_rad": "p_th", "radius_mm": "p_r", "dist_to_junction_mm": "p_dj"},
    "centerline": {"xyz": "cv", "radius_mm": "cr", "edges": "ce", "segment": "cs", "tangent": null},
    "branches": [{"id": 0, "key": "root", "name": "主动脉", "parent": -1, "length_mm": 211.1}],
    "openings": []
  },
  "fields": [
    {
      "id": "tawss", "label": "周期平均壁面切应力", "short_label": "TAWSS", "units": "Pa",
      "location": "wall", "kind": "scalar", "components": 1,
      "temporal": "cycle_summary", "time_index": null,
      "source": "prediction", "tier": "model", "derived_from": [], "definition": "...",
      "arrays": {"display": "f.tawss.d", "read": "f.tawss.r"},
      "display": {"thresholds": [0.4, 4.0, 7.0], "threshold_direction": "below", "log_scale": true, "range": null, "p99": 4.36, "max": 11.9},
      "windows": [{"id": "low", "label": "低剪切窗", "range": [0, 1], "provisional": true}],
      "validation": {"holdout_n": 34, "r2_pa": 0.74, "note": "..."},
      "statistics": {"mean": 0.665, "p99": 4.36}
    }
  ],
  "arrays": {
    "mv": {"dtype": "float32", "shape": [11669, 3], "bytes": 140028, "url": "/api/v2/jobs/<id>/arrays/mv"},
    "f.tawss.d": {"dtype": "float32", "shape": [11669], "bytes": 46676, "url": "..."}
  },
  "mapping": {
    "display_interpolation": {"method": "Gaussian", "sigma_mm": 0.5, "max_dist_mm": 1.5},
    "trust_bits": {"1": "interpolation_uncovered", "2": "rough_surface", "4": "geometry_out_of_range"},
    "trust_sources": [], "statistics_protocol": {}
  },
  "analysis": {
    "narrative": {"zh": [], "en": [], "edited": null},
    "findings": {"items": []},
    "zones": null,
    "morphology": {}, "profiles": {}, "per_branch": {}, "cycle": null, "peak": {},
    "surface_statistics": {}, "reference_assessment": {}, "trust": {}, "quality": {},
    "volume_statistics": null, "streamlines": null, "modules": null, "pressure_reference": null
  },
  "model_card": null,
  "provenance": {"release_hash": "...", "feature_contract": {}, "git_describe": "...", "code_source_hash": "...", "timing_s": {}, "interpolation": {}},
  "reserved": {"time_series": null}
}
```

规则：

- **字段**：
  - 壁面族：`wss`（display = 旧 `mw`，read = 旧 `pw`）加上 `fields` 里声明的 TAWSS / OSI / RRT / ECAP。
  - 体场族：
    - `pressure`：read 按点，display 为 null，前端用点显示；
    - `speed`：服务端算 `|v|`，read 按点；
    - `velocity`：read 为 N×3，kind vector；
    - `wall_pressure`：display 按显示网格顶点，read 为 null。
- **`tier` 只有三档**：
  - `geometry`：几何测量；
  - `model`：模型预测；
  - `derived`：派生量。

  RRT、ECAP、滞留区属于 `derived`。体场的 `speed`、`wall_pressure` 属于 `model`。
- **`temporal`**：逐帧字段为 `"frame"`，带 `time_index`；TAWSS、OSI、RRT、ECAP 为 `"cycle_summary"`，`time_index` 为 null。多帧数组（E7/E8）以后用 `arrays.display_frames` / `arrays.read_frames` 声明，形状为「帧 × 点」。现在没有，前端遇到就忽略。
- **缺失值**：数组里的 NaN 或无支撑点保留为 NaN，前端显示为缺失灰（`#a7adb3`），**不能当 0**。
- **分区**：`analysis.zones` 来自 summary / meta 的 `zones`（第 1 路新增，§5.1）。旧任务没有就是 null，前端隐藏该块。
- **显示名**：`job.display_name` 优先用任务记录的 `display_name`（第 1 路新增），否则 `patient_id` 非空就用它，否则 `case_id`。
- **`branches[].key`**：从 `paths.BRANCH_CN` 的反查得到语义键（root、left_cia、right_cia、out-le、out-li、out-re、out-ri）。反查不到给 null。前端不能写死分支名，一律从 manifest 读。
- **`frame.orientation`**：
  - `left_right`：出口命名已确认（`outlets_confirmed` 为真）时为 `"confirmed"`，否则 `"inferred"`；
  - `superior_inferior`：固定 `"derived"`（入口 → 分叉）；
  - `anterior_posterior`：`direction_source` 为患者坐标时为 `"confirmed"`，否则 `"inferred"`。
- **模型说明卡**：`model_card` 来自 `model_cards.load(release_id)`（§4）。`fields[].windows` 与 `fields[].validation` 从卡里取。没有卡时 windows 为 `[]`，validation 为 null。
- **体场附加数组**：
  - 点：`vpts`（N×3）、`vseg`、`vis_wall`（uint8）、`vtrust`（uint8）、`v_s`、`v_r`、`v_dw`；
  - 流线：`sl.xyz`（M×3 float32）、`sl.speed`（M）、`sl.offsets`（L+1 uint32）；
  - `geometry.points` 指向 `vpts` 等；
  - `analysis.modules` 是体场分块表。

## 4. 模型说明卡与参照格式（第 1 路实现）

**位置与加载**：

- 仓库 `wss_deploy/model_cards/<release_id>.json`，由 `wss_deploy/model_cards.py` 加载：
  - `load(release_id, release_dir=None) -> dict | None`：先读 `<release_dir>/model_card.json`，再读仓库文件；
  - `all_cards(registry) -> dict`。
- 不往 `outputs/wss_deploy_release/` 写任何文件。

**卡片格式 `wss-deploy.model-card/v1`**：

```json
{
  "schema_version": "wss-deploy.model-card/v1",
  "release_id": "M1_3head_3seed_20260922",
  "display_name": "周期指标 TAWSS · OSI",
  "short_name": "周期指标",
  "version_date": "2026-09-22",
  "purpose": "一次预测峰值 WSS、TAWSS、OSI",
  "training": {"n_train": 136, "cohorts": ["AAA", "AG", "ILO"], "centers": 1, "note": "..."},
  "protocol": [
    "所有病例用同一条入口流量波形（平均 30.38 mL/s，周期 0.8 s），不是该患者实测",
    "..."
  ],
  "validation": {
    "holdout_n": 34,
    "fields": {
      "wss": {"r2_pa": 0.74, "extra": "逐例均值 0.77"},
      "tawss": {"r2_pa": 0.74, "ccc": 0.93}
    },
    "source": "release.json 的 metrics 块"
  },
  "field_tiers": {"wss": "model", "tawss": "model", "osi": "model", "rrt": "derived", "ecap": "derived", "stagnation": "derived", "max_diameter": "geometry"},
  "usage": {"tawss": "高低分布可信，绝对值有误差", "osi": "只用来定位，边界不准"},
  "weaknesses": ["小口径髂支的高值容易被低估", "..."],
  "not_applicable": ["开口数不是 5 的输入（含三出口）", "非腹主—髂动脉", "含血栓或管壁的外轮廓"],
  "display_windows": {"tawss": [{"id": "low", "label": "低剪切窗", "range": [0, 1], "provisional": true}]},
  "caveat": "研究用途，不是诊断"
}
```

**数字只能来自对应发布包的 `release.json`**（逐项核对）。没有的写 `null` 并在 `note` 里说明原因，例如体场包：「部署体内采样与 CFD 网格口径是否等价尚未建立」。CFD 协议里未核实的假设（壁面刚性、血液模型）写「待核对」，不要编。

**命名色标窗（U15，暂定）**：

| 量 | 窗一 | 窗二 |
|---|---|---|
| WSS 峰值帧 | 低值窗 0–1 Pa | 常规窗 0–10 Pa |
| TAWSS | 低剪切窗 0–1 Pa | 常规窗 0–4 Pa |
| OSI | 全范围 0–0.5 | 高振荡窗 0.1–0.5 |
| RRT | 常规窗 0–20 1/Pa | — |
| ECAP | 常规窗 0–3 1/Pa | — |
| 相对压力 | 不设固定窗，只有「本例自适应」 | — |
| 速度 | 常规窗 0–1.5 m/s | — |

每个窗都标 `"provisional": true`。

**参照文件 v2（E4）**：

- `reference.json` 可以另带 `population_references`：`[{"id","metric","field","statistic","units","n","subgroup":"all|AAA|AG|ILO","source","model_release","data_version","built_on","values"|"quantiles"}]`。
- `reference.py` 同时读 v1（`population_reference`，p99）和 v2。
- 这一轮不产生新内容，只让读取器和测试支持 v2。

## 5. C 线口径（第 1、2 路）

### 5.1 分区统计 `zones`（第 1 路，U10）

- 新函数 `analysis.zones_wall(...)`，结果写进 summary 与报告 meta 的 `zones` 键。
- **计算入口**：pipeline 壁面 B 段 + `rebuild_report.rebuild_wall` 都要调用；体场族这一轮可以不做。
- **不改动** `per_branch` 等既有块。

输出格式 `wss-deploy.zones/v1`：

```json
{
  "schema_version": "wss-deploy.zones/v1",
  "definition": "按中心线分支与弧长划分解剖分区，区内预测点等权统计；面积 = 点占比 × 输入壁面面积（估计）",
  "area_method": "point_fraction_times_input_surface_area",
  "primary_fields": ["tawss", "osi", "wss"],
  "zones": [
    {
      "id": "neck", "label": "近端瘤颈", "segment_id": 0, "branch_key": "root",
      "s_range_mm": [48.0, 119.0], "n_points": 0, "area_mm2": 0.0,
      "fields": {
        "tawss": {"mean": 0, "median": 0, "p10": 0, "p90": 0, "frac_low": 0, "frac_high": 0},
        "osi": {"mean": 0, "p90": 0, "frac_high": 0},
        "wss": {"mean": 0, "p99": 0, "frac_low": 0, "frac_high": 0}
      }
    }
  ],
  "pairs": [
    {"id": "cia", "label": "髂总", "left": "left_cia", "right": "right_cia",
     "fields": {"tawss": {"left_mean": 0, "right_mean": 0, "ratio_left_over_right": 0}}}
  ]
}
```

- **分区 id**：
  - 主动脉：有瘤时分成 `aorta_proximal`（瘤颈以上，长度大于 0 才列）、`neck`、`sac`、`aorta_distal`（瘤体以下到分叉，长度大于 0 才列）；无瘤时只有一个 `aorta`；
  - 髂动脉：`left_cia`、`right_cia`、`left_eia`（out-le）、`left_iia`（out-li）、`right_eia`、`right_iia`。
- **弧长范围**：来自 `morphology.aorta.neck/sac`。瘤颈和瘤体之间的空隙并入 `neck`。
- **阈值**：`frac_low` / `frac_high` 用字段自己的阈值与方向——WSS / TAWSS 低于 0.4 Pa 为低、高于 4 Pa 为高；OSI 高于 0.1 为高。
- **左右对**：`pairs` 为髂总、髂外、髂内三对，比值是左 / 右（右为 0 时给 null）。
- **字段范围**：只统计本结果实际有的字段。

### 5.2 发现规则（第 1 路）

- **U12 全场最大值并簇**：`max_wss` 并入包含该点的高值簇，簇上加 `"contains_global_max": true`，不再单列一条。
  - 若最大值点不在任何簇里，保留单列，但 severity 按 U11 定。
- **U12 面积下限**：面积型簇（低 WSS、低 TAWSS、高 OSI、滞留区）估计面积小于 100 mm² 的不列；高值簇保留现有的 ≥ 20 点规则。
- **审阅记录兼容**：现有 `findings_review` 边车按 id 关联。改规则后 id 会重排，要保证旧的审阅记录不会错贴到另一条发现上：按 kind + 位置匹配，匹配不上就标为「规则更新前的判定」保留而不套用。实现前先读现有代码确认它怎么关联。
- **U11 高值定级**（仅限 `high_wss_cluster` 和并入的全场最大值）：
  - 发布包有同口径人群参照（目前只有 X5D 的 p99），且本例 p99 ≥ 参照第 90 百分位 → `attention`；
  - 有参照但低于第 90 百分位 → `note`；
  - 没有参照 → `note`，并加 `"grading": "no_reference"`。
- **低值一侧规则不动。**

### 5.3 叙述与一页纸（第 1 路）

- **C1 措辞**：按讨论稿 §15.1 的表逐条改（中英文），包括 narrative / morphology / analysis / timeline / onepager / export_table 里给人看的字串。JSON 键名、CSV 列名不动。
- **C2 三头结果**：
  - 叙述顺序改为形态 → 周期量 → 峰值帧（一句）；
  - 峰值帧的「低 WSS 占比」不进叙述；
  - 热点句改成「峰值 WSS 最高的区域（不低于本例 p99，17.0 Pa）N 处」；
  - 单头结果顺序不变，但热点句同样改正定义。
- **审阅人编辑过的结论不动。**
- **C4 局限性声明**：第一条写标准血流条件，另加一条「所有直径、长度和体积都是管腔的」。
- **C5 集成质量**：一页纸显示成「多模型一致（一致不代表准确）」；`quality.label` 数据不变。
- **U13 位数**：一页纸三位有效数字。
- **C3 验证数据**：
  - 一页纸第 1 页的关键数字带来源档（几何 / 模型 / 派生）；
  - 附录加「模型验证」表（来自模型说明卡）。
- **C7 显示名**：
  - 任务公开记录与 `/api/cases`、`/api/jobs*` 返回里加 `display_name`（`patient_id` 非空用它，否则 `case_id`）；
  - 一页纸标题和 summary 的 `display_name` 用它；
  - 一页纸里把病例名称作「匿名病例编号」的说法改掉：病例名来自文件名，可能是姓名。
- **版本**：`schema.ANALYSIS_VERSION = "2026-09-30"`，注释说明改了什么。

### 5.4 旧界面文案与帮助页（第 2 路）

- **C1**：旧界面上所有「最大直径」改成「管腔最大直径」；列表上的「⌀ 72.2 mm」保留，悬停提示改成「管腔最大直径」。
  - 涉及：`app.js`、`workbench_core.js`、`report_common.js`、`volume_viewer.js`、`index.html`、`report.py` 模板、`volume_report.py` 模板、`glossary.py` / `glossary.json`。
  - 术语表按讨论稿 §15.1 加管腔说明。
- **C5**：旧界面显示「模型集成稳定」的地方改成「多模型一致」，并注明「一致不代表准确」。
- **C7**：旧工作台列表、病例卡、详情标题用 `job.display_name`（没有就回退 `case_id`）。上传时，文件名形如 `^[A-Za-z]+(?:[_ -][A-Za-z]+){1,3}$` 且全是字母的，提醒「文件名可能是姓名，建议填写患者编号」（只提醒，不拦）。
- **C8**：
  - **旧上传对话框**：加一行输入要求和「输入说明」链接（`/static/v2/help_input.html`）、「下载示例 STL」（`/static/v2/example_aaa.stl`）。
  - **帮助页**（CSP 不允许内联脚本；样式用 `help.css`，遵守 §7 视觉规格）：
    - `static/v2/help_input.html`：输入说明——要什么样的 STL、怎么在分割软件里导出、五个开口、单位、三出口不支持、常见失败与处理，并带一张内联 SVG 示意图；
    - `static/v2/help_quickstart.html`：一页操作卡——上传 → 核对单位和出口 → 看结果 → 出一页纸，每步一两句话，不配截图也要能看懂；
    - `static/v2/help_errors.html`：报错对照表——从 `errors.py`、`ingest.py`、`jobs.py` 的真实报错文案整理「报错 → 原因 → 怎么办」。
  - **示例 STL**：复制 `outputs/wss_deploy_golden/20260920_baseline/20260920_144135_673ccd0e36b1/input.stl` 到 `static/v2/example_aaa.stl`。
- **U14**：旧界面不改代号（旧界面会下线）。

## 6. 前端代码结构（第 4、5 路）

### 6.1 模块写法

每个文件都是经典脚本（不是 ES module），挂到全局命名空间 `WSSV2`，在 Node 里可以 `require`：

```js
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.util = mod;                                        // 每个文件自己的名字
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  // ... 只用 ns.xxx 访问其他模块；依赖 three 时用 root.THREE
  return { /* API */ };
});
```

- 加载顺序以 `bundle.json` 为准。模块在加载时不访问 DOM、不发请求，只在被调用时做事。
- 离线模式：页面里存在 `<script type="application/json" id="wssv2-manifest">` 时 `ns.env.offline === true`（`core_util.js` 负责判断）。离线包不含 `bundle.json` 里 `offline_exclude` 列的文件；外壳用到它们前必须判断是否存在。

### 6.2 查看器内核 API（第 4 路实现，第 5 路调用）

```text
ns.util      fmtSig(v, sig=3) / fmtPct(frac) / fmtValue(v, units) / escapeHtml / emitter() / rafThrottle(fn) / clamp
             env: {offline: bool}
ns.data      createOnlineSource(jobId, {fetch}) / createEmbeddedSource(document)
             source.manifest() -> Promise<manifest>
             loadResult(source, {signal}) -> Promise<Result>
             Result: {manifest, runIdentity, family, preload(keys) -> Promise, array(key) -> TypedArray（须已 preload）,
                      field(id) -> 描述, fieldArray(id, 'display'|'read') -> TypedArray|null,
                      branch(id) -> 描述, valueAt(fieldId, pointIndex) -> number|NaN}
             数组解码一次、校验 dtype/shape/字节数，不符则 reject（DataError），不能补 0
ns.colormap  scale({range, log, bands, cmap:'rainbow'|'turbo'|'viridis'|'bwr', missing}) -> {color(v)->[r,g,b], norm(v), ticks(n), describe()}
             histogram(values, {bins, range, log}) -> {edges, counts, n, nMissing, nBelow, nAbove}
             彩虹默认；色表优先复用 report_common.js 里已有的
ns.colorbar  create(el) -> {update(info), dispose()}
             info = {label, units, scale, histogram, thresholds, thresholdDirection, windowLabel, missingFraction}
             竖直色条 + 刻度 + 加粗阈值线 + 嵌入直方图（注明「采样点分布」）+ 窗名 + 「超出范围按端色」
ns.viewer    create(container, {kind:'main'|'thumb'}) -> Viewer；webglAvailable()；link(a, b) -> unlink()（世界坐标相机同步，防回声）
  Viewer.setResult(result) -> Promise           按 family 选适配器
  Viewer.setField(fieldId, scaleSpec)           scaleSpec = {window:'adaptive'|<窗 id>|{range:[a,b]}, log:null|bool, bands:null|int, cmap}
  Viewer.setLighting('flat'|'soft')             默认 flat + 轮廓线
  Viewer.setLayers({centerline, points, trust, outline, streamlines, wall, interior})
  Viewer.setBranchVisibility(ids|null)
  Viewer.fit() / standardView('anterior'|'posterior'|'left'|'right'|'superior'|'inferior')   按 manifest.frame 的解剖坐标架
  Viewer.select(pointIndex|null) / highlight(indices|null, {color}) / setMarkers([{id, xyz, label, kind}])
  Viewer.setCursor({segmentId, s_mm}|null)      在中心线站位画垂直环
  Viewer.getState() / applyState(state, {animate})
                                                state = {field, scale, camera:{position,target,up,fov}, lighting, layers,
                                                         cursor, selection, branches, time_index:0}
  Viewer.colorbarInfo() -> info
  Viewer.snapshot({scale:2, transparent:false}) -> Promise<Blob>
  Viewer.resize() / render() / dispose()
  事件 on(name, fn)：'pick' {pointIndex, vertexIndex, xyz, segmentId, s_from_root_mm, value, screen}
                    'hover'（同上，节流；离开时 null）/ 'camera' {camera, source} / 'marker' {id}
                    'ready' / 'contextlost' / 'contextrestored' / 'error'
ns.orientation  create(el, viewer) -> {update(), dispose()}
                方位标：左/右（未确认时标「推断」）、头/足、前/后（一律标「推断」，虚线灰字）
ns.cursor    create(result) -> Cursor
  Cursor.set(segmentId, s_mm) / step(delta_mm) -> {state, atEnd, children:[segmentId...]} / state()
  Cursor.readout(fieldId) -> {segmentId, s_mm, bin:[s0,s1], stats:{...}, definition}
                                                用 manifest.analysis.profiles 的 2 mm 分箱，不新造统计口径
  Cursor.on('change', fn)
ns.adapters.wall / ns.adapters.volume           Viewer 内部使用；接口 {supports:{fields, tools}, build(result, ctx) -> handle}
```

- **拾取**：射线命中显示网格 → 最近预测点（空间网格索引，别每次全量扫描）。`value` = read 数组在该点的值，不从颜色反推。
- **显示**：
  - 壁面：显示网格按顶点着色，缺失 NaN 着缺失灰。可选图层：trust 斜纹、中心线、预测点、轮廓线。
  - 体场：
    - `pressure` / `speed`：体内点着色，外面套半透明血管；
    - `wall_pressure`：网格着色；
    - `streamlines`：线图层，按速度着色。
- **资源**：`dispose` 要释放几何、材质、纹理、控件、监听和定时器；连续切 20 次结果内存不能单调增长。
- **按需渲染**：只在相机或状态变化时画，不开常驻 rAF 循环；不可见的视口不画。
- **调试页**：第 4 路可在 `static/v2/dev_viewer.html` + `dev_viewer.js` 做一个只加载内核的页面（`?job=<id>`），用来自己截图检查。
- **工具接口**（第二期 S1，截面工具用）：`viewer.canvasElement`；`toolOverlay()` 给工具放自己三维对象的组（换结果时随结果释放）；`rayAt(clientX, clientY)` → 已按该点设置好的 Raycaster；`projectPoint(xyz)` → 视口内 CSS 像素 `{x, y, inFront}`；`mmPerPixelAt(xyz)`；`setControlsEnabled(bool)`（工具拖动期间关掉旋转缩放）；`surfaceAt(clientX, clientY)` → 适配器的 `pickSurface`（体场：半透明管壁上的点 `{xyz, vertexIndex}`，与当前字段无关）。体场适配器 `supports.tools` 加 `slice`。
  第二期 S2：`setClipPlane({normal, origin, side, pointGap} | null)` 在一个平面处切开结果自己的对象（不切工具覆盖层），保留 `side·normal·(p − origin) ≥ 0` 一侧；点精灵的裁剪面再往保留侧让 `pointGap` mm，免得贴面的点盖住截面图；平面对象复用，移动时不重编着色器。体场半透明管壁的着色器带 three.js 裁剪片段。`highlight(indices, {color, size, opacity})`：`size` 为相对体内点的直径倍数，`opacity < 1` 时半透明。
  第二期 S3：`highlight([], {vertexMask})`（壁面：按显示顶点直接标区域，其余变淡）；`setToolLabels(key, [{xyz, text}])` 工具的三维文字标签（与发现标记一起防重叠）；壁面适配器也有 `pickSurface`（原始命中点）；体场 `setBranchVisibility` 连管壁面（顶点归最近的壁面预测点分支，无壁面点时用中心线）和流线（中段最近体内点的分支）一起藏；两个适配器 `supports.tools` 加 `measure`，壁面再加 `region`。

### 6.3 工作区（第 5 路）

**布局**：

| 区域 | 规格 |
|---|---|
| 顶栏 | 40 px。左：「WSS」字标 + 当前病例显示名；右：上传、帮助（含「反馈问题」）、用户菜单（管理员含「运维中心」） |
| 左栏 | 病例栏 240 px，可收起 |
| 中间 | 视口，一个或两个 |
| 右栏 | 检查器 360 px，可收起 |
| 视口左上 | 一行状态：结果、计算状态、复核状态、耗时 |
| 视口右侧 | 色条 |
| 视口左下 | 方位标 |
| 视口底部 | 时间栏占位，只有一帧时隐藏（E9） |

**文件与职责**：

| 文件 | 职责 |
|---|---|
| `ws_api.js` | fetch 封装：会话、CSRF、401 处理；列表、查询、任务、SSE、上传、确认、复核、发现审阅、元数据、一页纸、打包、离线导出、模型说明卡、患者时间线 |
| `ws_store.js` | A8 状态合同（§6.4） |
| `ws_ui.js` | DOM 小工具：`h()`、对话框、菜单、提示条、表格 |
| `ws_icons.js` | 约 24 个 16 px、1.5 px 描边的内联 SVG 图标：上传、搜索、左右下箭头、关闭、图层、快照、书签、书签实心、比较、探针、尺子、眼睛、眼睛关、光照、下载、打印、外链、信息、警告、对勾、刷新、播放、暂停、用户、设置 |
| `ws_rail.js` | 病例栏：病例 → 扫描 → 结果三层；搜索、状态筛选；没选病例时主区显示待办（待确认出口、失败、未复核） |
| `ws_overview.js` | 概览检查器，按顺序：<br>1. 显示名、扫描、结果切换（同输入的其他结果，含 companions）、状态行；<br>2. 结论（标注是否人工编辑）；<br>3. 分区表（U10，列按 `zones.primary_fields` 选三个量）；<br>4. 左右对比；<br>5. 形态（管腔最大直径等，来源档「几何」）；<br>6. 发现：每类最重一条，「全部 N 条」可展开，「其余按自动结果确认」按钮调用现有 findings_review；<br>7. 随访表（同一 patient_id 有两次以上扫描时，U17）；<br>8. 模型说明入口与「研究用途，非诊断」一句。<br>每个数都可点，点了进证据透镜 |
| `ws_slice.js` | 体场截面（第二期 S1）：平面定义与经典体场报告相同（中心线分支 + 位置、1 点垂直中心线 / 2 点斜截面、俯仰 / 偏航、面内偏移、厚度 0.2–12 mm）；数值全部调用 `VolumeViewerCore`（取片层、壁面轮廓、补全、按点统计、Voronoi 面积积分），同一平面与经典报告逐项一致；三维里画截面框、法向箭头和贴在平面上的补全图，拖动框移动（Shift 旋转、Alt 平移），滚轮移动（Shift 改厚度），点选模式在管壁上放点；检查器「截面」页：补全图（悬停读数）、物理量（速度 / 穿面速度 / 压力）、关键数字（面积、等效直径、面积平均、流量 Q）、色标（本截面 p2–p98 / 与三维同 / 手动）、补全与箭头开关；放大图可导出 PNG / CSV。截面打开时体内点隐藏，舞台色条换成截面色标。第二期 S2：「截面系列」（`seriesFractions` 站位、全系列共用色标、对话框并排看、点选跳转、导出拼图 PNG）、「两截面之间」（`arcAlongBranch` + `regionIndices` + `statistics` + `regionPressureDrop`，三维绿点 + 两端轮廓）、「切开」（`setClipPlane`，保留上游 / 下游，截面图盖住切口，切开时体内点重新显示） |
| `ws_probe.js` | 探针（第二期 S3）：点一下血管钉住探针。壁面：离命中点精确最近的预测点，过命中点、垂直中心线的截面上各字段环上均值（`WssReportCommon.sectionMeans`，逐段取最近预测点、按长度加权）与环上最低 → 最高；体场：体内点记录与该截面的面积积分（`stationSection` + `ws_slice.integrate`）。中心线分支组按经典报告的方式建（不用游标的）。「记录」按经典行格式存本机，`probeToTSV` / `probeToCSV` 导出；三维画截面环 |
| `ws_measure.js` | 测量（S3，M 键）：距离、弧长、管径、分段，经典 `buildMeasurement` 的算法与标签（管径两族都用真实截面，退回 2 × 内切半径时注明）；点击取管壁原始命中点；三维线、点、截面轮廓与文字标签；列表可定位 / 复制 / 删除，复制全部用经典体场报告的列；按结果存本机 |
| `ws_region.js` | 壁面区域统计（S3）：分支上一段（距入口 mm，按预测点 `s_from_root`）或球形区域；当前字段的等权均值 / p99 / 最大（`VolumeViewerCore.statistics`），顶点按经典规则归属后高亮、其余变淡；面积为显示网格面积 |
| `ws_review.js` | 发现判定（第二期 S4）：经典 findings_review 文档（items {id: {decision, note}}，added 人工发现 ≤ 30）；无判定无备注的条目去掉；人工发现编号 M<k> 避开所有发现；带任务版本号的去抖保存（0.6 s），409 读回；读数页判定条、新增人工发现对话框；锁定或离线只读 |
| `ws_annot.js` | 标注与自动标签（S4）：经典 annotations 文档（≤ 50 条，文字 ≤ 200）；三维常驻显示（点、朝头侧的竖线、`A1 · 文字`）；「标注」页钉、改、删、定位，去抖保存；自动标签：分支名、前 N 条未驳回的发现、管腔最大直径环（`prefs.labels`） |
| `ws_lens.js` | 证据透镜 D9：结果 → 位置 / 区域 → 值（预测 / 派生 / 几何）→ 映射与聚合方法（显示插值、2 mm 分箱、点等权、面积估计）→ 支撑（trust 位、几何参照、插值覆盖）→ 该量留出集一致性（卡片）→ 人群位置（只在同口径参照存在时）。未知就写未知 |
| `ws_bookmarks.js` | 书签 D10：保存（名称 + 一句备注）、列表、恢复、排序、删除、导入 / 导出 JSON；随离线导出。记录 `run_identity` / `data_version` / `viewer_version` / 字段 / 窗 / 相机 / 游标 / 选中点 / `time_index`。结果或数据版本不符时提示不兼容，不强行套用 |
| `ws_questions.js` | 按问题打开 D8，三个问题：<br>「低值区域在哪里」：TAWSS（无则 WSS）+ 低值窗 + 低值发现；<br>「同一位置的 TAWSS 与 OSI」：双视口联动；<br>「比较两次结果」：进比较。<br>显示当前套用了什么，可撤销，回到自由浏览 |
| `ws_compare.js` | 比较 D11 + 双视口 D3：选第二个结果 → 比较条件表（字段、单位、时相 / 周期、发布包、几何是否同一输入 sha、统计定义），逐条给「一致 / 不一致 / 未知」。模式二选一：「同尺度看幅值」（只对兼容字段）/「各自增强细节」（醒目标注范围不同）。相机同步只在同一输入几何时默认开，否则标「只同步视角，不代表位置对应」。**不做逐点差值** |
| `ws_upload.js` | 上传对话框：<br>- 输入要求四句话 + 帮助链接 + 示例 STL；<br>- 「想看什么」用模型说明卡的显示名和一句验证摘要（U14），不显示发布包代号；<br>- 单位；<br>- 患者编号、扫描日期标成「随访需要」；<br>- 文件名像姓名时提醒（C7）；<br>- 查重复用沿用现有流程 |
| `ws_input.js` | 失败页 U2：三维画出网格和每个开口（编号 + 半径），半径明显偏小时提示「可能是没封闭的分支残端」，并附报错对照表链接。出口确认 O4：在主视口里标出口，用现有 confirm 接口与载荷（`ANALYSIS_CONTRACT.md`），互换按钮沿用现有语义 |
| `ws_export.js` | 按用途导出 O6：<br>- 汇报图：快照 + 色条 + 窗名 + 显示名，可隐藏病例名；<br>- 复核数据：现有 CSV / VTP / zip；<br>- 离线阅读：`/api/v2/.../offline`，带书签、可隐藏病例名；<br>- 打印报告：现有一页纸 |
| `ws_shell.js` | 路由、布局、基本 / 完整两档（U22，默认基本；完整档多出：换发布包重跑、完整统计 JSON、技术信息、队列总览与回收站的经典入口）、键盘（§6.5）、状态行、离线模式外壳 |
| `ws_main.js` | 启动：会话 → 登录表单（未登录时）→ shell |
| `index.html`、`v2.css` | 页面骨架与样式 |

**离线模式**：

- 没有病例栏和写操作；检查器只有概览、读数、书签。
- 顶部写「离线报告 · 导出于 … · 导出时复核状态 …」。
- 数据来自 `createEmbeddedSource(document)`；`wssv2-offline` JSON 里带 `bookmarks`、`view`、`hide_name`、`exported_at`。

**经典页面**：第三期（S7）已下线，新工作区里不再有指向经典页面的链接。经典功能的去向见 `lanes/p3_audit_*.md` 和各路报告 `lanes/p3_*.md`。

### 6.4 状态合同 A8（`ws_store.js`）

| 层 | 真源 | 键 | 规则 |
|---|---|---|---|
| 偏好 | localStorage | `wssv2:prefs` | 档位、光照、色表、面板开合；跨结果继承前校验字段存在 |
| 阅读位置 | localStorage | `wssv2:view:<run_identity>` | viewer.getState()；同结果恢复，首次打开 fit；跨结果不迁移空间状态 |
| 书签 | localStorage | `wssv2:bm:<run_identity>` | 见 ws_bookmarks |
| 业务事实 | 服务端 | — | 状态、复核、发现判定、结论编辑；写成功后才显示已保存；409 读回新版本 |
| 暂态 | 内存 | — | 悬停、拖动、请求 |

- 所有 localStorage 读写包 try/catch，失败时照常工作。
- 快速切例：每次打开结果带递增序号，回包先核对 `run_identity` 和序号，过期的丢弃。

### 6.5 键盘（O2）

| 键 | 作用 |
|---|---|
| J / K | 下一例 / 上一例 |
| 1–9 | 按 manifest 顺序切场，不存在的序号不响应 |
| ← / → | 游标工具打开时沿血管移动 1 mm（Shift 5 mm） |
| [ / ] | 上一个 / 下一个发现 |
| L | 光照 |
| B | 存书签 |
| G | 游标开关 |
| M | 测量开关；Esc 先取消未完成的取点，再退出测量方式 |
| S | 截面开关（体场结果）；截面打开时 ↑ / ↓ 沿中心线或法向移动 1 mm（Shift 5 mm），← / → 偏航、PgUp / PgDn 俯仰 2°（Shift 10°），[ / ] 厚度 ∓0.4 mm，Esc 先结束点选再关截面 |
| Esc | 退出当前工具或关闭对话框 |
| ? | 快捷键说明 |

输入框、文本区、对话框内不触发；不抢浏览器组合键。

## 7. 视觉规格（第 2、4、5 路共同遵守）

目标气质：期刊插图式的克制，加阅片工作站的密度。检查表见讨论稿 §5.3。

**颜色**（`v2.css` 与 `help.css` 用同一组变量）：

```css
--bg:#f3f4f5; --panel:#ffffff; --ink:#1b2430; --ink-2:#465361; --ink-3:#77828e;
--line:#dfe3e7; --line-2:#c6ccd2; --accent:#1d5d95; --accent-weak:#e9f0f7;
--warn:#8a5a00; --warn-bg:#fbf3e2; --error:#9c3326; --error-bg:#faece9; --ok:#2c6a4b;
--viewport:#eceef0; --missing:#a7adb3;
```

- 界面只用一个强调色；颜色留给数据。无渐变，无装饰性阴影（只有浮层用 `0 2px 8px rgba(0,0,0,.12)`）。
- 圆角 ≤ 3 px（按钮、输入框），面板直角；分隔用 1 px `--line`。

**字体与数字**：

- 字体 `system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif`；数字 `font-variant-numeric: tabular-nums`。
- 字号三级：13 px 正文，12 px 次要，15 px 区块标题；病例显示名 18 px。
- 界面数字三位有效数字；百分比取整（小于 1% 留一位小数）；永远带单位。

**控件与状态**：

- 按钮：默认白底 1 px `--line-2` 边、`--ink` 字，28 px 高；主按钮 `--accent` 底白字，每屏最多一个；图标按钮 28×28。
- 状态 = 文字 + 6 px 圆点，不只靠颜色。状态词：排队、计算中、待确认出口、失败、已完成、已复核。
- 来源档标签：12 px 文字、1 px `--line-2` 边、2 px 圆角、无底色，「几何」「模型」「派生」。不用彩色胶囊。

**表格与视口**：

- 表格：行高 28 px，数字右对齐，表头 12 px `--ink-3`，无斑马纹，1 px 横线。
- 视口：纯色 `--viewport` 背景；血管自动适配到有效区域，不被色条和面板遮挡。

**文案**：

- 不要问候语、教程腔、营销词（智能、一键、轻松）、emoji 和感叹号。
- 「研究用途，非诊断」全局只出现一次（检查器底部；导出与离线报告里各一次）。
- 空状态说明下一步能做什么。
- 界面上不出现发布包代号、`run_identity`、Feret、Gaussian 这类内部词；这些词只放在「技术信息」里。

**其他**：

- 图标见 §6.3 `ws_icons.js`，不要用 Unicode 字符当图标。
- 支持 `prefers-reduced-motion`；相机过渡不超过 300 ms。

## 8. 验收（每路自查 + 主会话合并）

1. 你负责的测试加全套测试通过；`node --check` 过你所有 JS 文件。
2. 第 3、4、5 路：在沙箱里用 devshot 截新页面，0 个 JS 错误，并亲自看截图（Read 图片）确认没有明显错位。
3. 第 5 路外壳：Node 启动桩测试（参照 `tests/test_workbench_js.py` 的整页桩），覆盖以下几项：
   - 登录；
   - 打开结果；
   - 快速切例不串（慢回包被丢弃）；
   - 缺字段的旧任务；
   - 离线模式启动。
4. 第 3 路：manifest 对 8 个开发任务全部成功；数组解码后与 report.html 内嵌值逐字节相同；RRT/ECAP 与 `cycle_fields` 公式一致；离线 HTML 在无服务时能用 devshot `file://` 打开并渲染。
5. 最终报告写：改了哪些文件、测试结果、截图路径、已知问题、需要其他路配合的事。

## 9. 分工与端口

| 路 | 负责 | 名下文件（只改这些） | 沙箱 HTTP / Marionette |
|---|---|---|---|
| 1 C 线分析 | §4、§5.1–5.3 | `analysis.py`、`narrative.py`、`onepager.py`、`morphology.py`（仅文案）、`timeline.py`（仅文案）、`export_table.py`、`pipeline.py`、`volume_pipeline.py`、`rebuild_report.py`、`reference.py`、`schema.py`、`jobs.py`（仅 display_name）、`cases.py`（仅 display_name）、`model_cards.py`（新）、`model_cards/*.json`（新）及对应测试 | 8815 / 2915 |
| 2 C 线旧界面 | §5.4 | `static/app.js`、`static/index.html`、`static/app.css`、`static/workbench_core.js`、`static/report_common.js`（仅文案）、`static/volume_viewer.js`（仅文案）、`report.py`（模板文案）、`volume_report.py`（模板文案）、`glossary.py`、`static/glossary.json`、`static/v2/help_*.html`、`static/v2/help.css`、`static/v2/example_aaa.stl` 及对应测试 | 8814 / 2914 |
| 3 数据接口 | §2、§3、离线打包 | `v2_data.py`（新）、`v2_offline.py`（新）、`server.py`、`tests/test_v2_data.py`（新）、`tests/test_v2_offline.py`（新）、`tests/test_v2_routes.py`（新） | 8813 / 2913 |
| 4 查看器内核 | §6.1、§6.2 | `static/v2/core_*.js`、`static/v2/adapter_*.js`、`static/v2/dev_viewer.html`、`static/v2/dev_viewer.js`、`tests/test_v2_core_js.py`（新） | 8811 / 2911 |
| 5 工作区 | §6.3–6.5、§7 | `static/v2/index.html`、`static/v2/v2.css`、`static/v2/ws_*.js`、`tests/test_v2_workspace_js.py`（新） | 8812 / 2912 |

**文件冲突说明**：

- `report_common.js`：第 2 路只改文案，第 4 路只读不改；第 4 路需要改时，把需要写进报告。
- `server.py`：只有第 3 路改。旧页面上的入口链接由第 2 路改 `index.html` / `app.js`。
- `bundle.json`：由主会话维护，谁都不改。
