# wss_deploy S7 前审计：运维中心 / 工单、旧页面清单、加载速度

审计对象：`/public/newhome/cy/Digital_twin/GNN_wssui_v2`（分支 `wss-ui-v2`，HEAD `6e45be1`）。只读；未启动服务或浏览器。
行号指该工作树里的文件。"不确定"的地方单独标出。

---

## 0. 先说结论（五个意外）

1. **v2 的数据就是从 `report.html` 读的。** `v2_data.py:1-8, 396`：manifest 和全部数组都从任务目录里的经典 `report.html` 解析
   （`wss-report-meta` 加上 `wss-report-arrays` / `volume-arrays`）。所以 S7 **不能停掉 report.html 的生成**，只能停掉"把它当页面给人看"。
   想"只维护一套查看器"，要么让 report.html 继续内嵌经典查看器（这样经典查看器仍在维护），要么把它瘦身成纯数据容器（S7b，见 §2.3）。
2. **`/ops` 和 `/support` 依赖经典样式 `static/app.css`。** `ops.html:8`、`support.html:3` 都引用 `/static/app.css`；
   `.card/.primary/.text-button/.notice/.eyebrow/.overview-tiles/.brand-mark` 这些类只在 app.css 里定义（ops.css 只有 44 行增补）。
   如果 S7 连同经典工作台删掉 app.css，两页会失去基础样式。
3. **两个工作台都没有入口链接到 `/ops` 或 `/support`。** 经典 `index.html` / `app.js` / `workbench_core.js` 里都没有，v2 里也没有。
   只能手输地址，或者从文档找到（`wss_deploy/README.md:26`、`OPERATIONS.md:9-10`、`OPERATIONS_ACCEPTANCE.md:10`、`docs/02-推进与变更/05-部署工具/README.md:16`）。
   两页之间互相链接：`ops.html:20`「工单入口」指向 `/support`，两页的「返回工作台」都指向 `/`。
4. **有两项功能只有经典工作台有，删掉 app.js 就没了**（B2「功能不减」要求先处理）：
   - **认领旧会话任务**：UI 在 `app.js:517-526`，调用 `POST /api/jobs/claim`（`server.py:1696-1708`）。v2 里没有 `claim` 相关代码。备用是命令行 `cli jobs claim`（要求服务已停）。迁移矩阵里也没有这一项。
   - **换入口 / 重提中心线**：UI 在 `app.js:2257-2258`，调用 `confirm {inlet}`。v2 的 `ws_input.js:337` 只发 `mapping`，不发 `inlet`。
     这正是 v2 还保留 `ws_shell.js:509`「在经典工作台中处理」链接的原因。
5. **截图套件会在 `/` 跳转后悄悄变成未登录状态。** `devshot.py:134-140` 的 `Browser.login` 填的是经典登录表单
   （`#login-panel/#login-username`）；`suite()`（`devshot.py:259-271`）会依次打开 `/`、`/#job=`、`/api/jobs/<id>/report`、`/compare`。
   `/` 改成跳到 `/v2/` 之后，v2 的登录表单是 `#wsl-user/#wsl-pass`（`ws_shell.js:1975-1976`），`login()` 找不到表单就直接返回，不会报错。
   `PHASE2_LANES.md:80-81` 规定的沙箱截图流程也走这条路径。

另外：
- `static/v2/example_report.html` 已经过期。它是生成文件，被 gitignore 了（`.gitignore:89`）；
  文件时间 12:20，而 `ws_shell.js` 是 20:48。里面还有旧的 `CLASSIC_TOOLS`「经典报告里的工具」（第 8243 行）。S7 之后要用 `v2_examples` 重新生成。
- **改动四个遗留脚本中的任何一个，都会让所有已完成的报告被判为过期。** 这四个是 `three.min.js / OrbitControls.js / report_common.js / volume_viewer.js`，
  它们和 `glossary.json` 一起列在 `report_freshness.UI_SOURCES` 里（`report_freshness.py:33-34`）。
  判为过期后，下一次 `service upgrade` 会批量重写 report.html（`jobs.py:3114-3130`），同时 v2 的 `data_version` 和数组 ETag 全部失效。
  所以提速方案不要拆分或改写这几个文件。

---

## 1. 运维中心 `/ops` 与用户工单 `/support`

### 1.1 服务端门禁（两页共用）

| 项 | 位置 | 现状 |
|---|---|---|
| 页面外壳 | `server.py:1499-1507` | `/ops`、`/ops/`、`/support`、`/support/` 不需要登录就返回静态 HTML；外壳里没有用户数据，登录在页面里完成 |
| 运维接口门禁 | `server.py:1513-1518`（GET）、`1671-1676`（POST）；`operations_http.py:38-39, 250, 397` | 共享模式下只有 `role=admin` 能访问（403「运维接口仅限管理员访问」）；本机非共享模式下，任何回环会话都算管理员 |
| 工单接口 | `server.py:1519-1522, 1677-1680`；`operations_http.py:360-366, 453-472` | 要求登录；只能读写自己的工单（owner 不匹配时返回 404） |
| 写请求 | ops.js / support.js 的 `request()` | 带 `X-CSRF-Token`；同源检查由 server 负责 |
| 数据位置 | `OPERATIONS.md` §持久化 | `<jobs_root>/.operations/operations.sqlite3`（工单、审计、素材元数据）和 `.operations/stl/<sha256>`（归档的原始 STL）；启停账号直接写 `users.json` |

### 1.2 `/ops` 功能清单（`static/ops.html` 125 行 + `ops.js` 381 行 50.7 KB + `ops.css` 20 KB，标题「运维中心 · WSS 工作台」）

| # | 功能 | 控件 | 调用的接口 | 读/改的数据 | v2 有没有对应 | 建议 | 迁进 v2 的工作量 |
|---|---|---|---|---|---|---|---|
| O1 | 登录 / 退出 / 身份显示 / 连接状态 | 账号口令或访问令牌表单，退出按钮（`ops.js:358-359`） | `GET/POST /api/session`、`POST /api/session/logout` | 会话 | 有：v2 有登录页（`ws_shell.js:1970-1990`）和头像菜单里的「退出登录」 | 保留独立页（两边共用同一个 cookie） | — |
| O2 | 自动刷新 | 每 10 s 刷新一次，后台标签页暂停（`ops.js:379`）；「立即刷新」 | `/api/session` 加各分区接口 | — | 不适用 | 保留 | — |
| O3 | 概览：8 个指标卡 | 当前任务 / 失败任务 / 未结工单 / 待分派工单 / 待审阅 STL / STL 素材 / 7 天内到期 / 用户账号；点击跳到带筛选的分区（`ops.js:101-112`） | `GET /api/ops/overview` | 跨账号计数、`todo_counts`、最近 20 条事件、warnings | 没有（v2 首页只有本人或"看全部用户"时的任务待办） | 保留独立页 | #/ops 概览页约 0.5 d |
| O4 | 概览：提醒条 | 回收站 7 天内到期、未分派工单（`ops.js:104-106`） | 同上 | — | 没有 | 保留 | 同上 |
| O5 | 概览：服务健康 | 一行结论加可展开的 JSON（`ops.js:109-111`） | 同上（`health` 字段 = `server.health(row)`） | — | **部分有**：v2 用户菜单「服务状态…」对话框（`ws_admin.js:302-326, 359`，读 `/api/health`，显示版本、运行时长、队列、计算线程、GPU、磁盘、默认模型、待刷新报告），但不显示 `checks` 明细 | 保留；可以在 v2 对话框里给管理员加一个「运维中心」链接 | 链接约 0.5 h |
| O6 | 事件流（审计） | 关键词 / 账号 / 类别 / 级别 / 起止时间 / 操作代码 / 每页条数；暂停；回到最新；「有新动态」提示；翻历史页时不跳动；逐条展开和原始 JSON（`ops.js:171-226`） | `GET /api/ops/events` | 审计库，外加任务事件、回收站侧车、`deleted_jobs.jsonl` | 没有（v2 只在单任务的「计算过程」里有该任务的事件） | 保留独立页 | 约 1 d |
| O7 | 导出事件 JSON | 「导出筛选结果 JSON」（`ops.js:227-238`） | `GET /api/ops/events/export`（最多 1 万条，占用 bundle 槽） | 只读 | 没有 | 保留 | 含在 O6 |
| O8 | 跨账号任务检索 | 关键词 / 提交账号 / 当前或回收站 / 状态 / 回收站 7 天内到期；分页；故障详情和诊断编号（`ops.js:113-127`） | `GET /api/ops/jobs` | 跨账号任务和回收站记录 | **部分有**：v2 `#/tasks` 的「看全部用户」（`ws_admin.js:350-356, 408`，`/api/jobs?all=1`）能看跨账号列表；**看不到跨账号回收站**（`ws_admin.js:793` 的 `/api/trash` 没带 `all=1`），也不能按到期时间筛 | 保留独立页 | 约 0.5 d |
| O9 | 归档 STL（单条 / 批量） | 每行「归档 STL」；勾选本页后「归档所选 STL」，逐条报告结果（`ops.js:278-306`）；也可以按任务 ID 归档（`ops.html` `#asset-reclaim`） | `POST /api/ops/assets` `{job_id, location}` | 把原始 STL 复制进素材库（按 SHA256 去重），写审计 | 没有 | 保留 | 约 0.5 d |
| O10 | 工单处理 | 筛选（含未结、未分派）；详情；改状态 / 优先级 / 处理人 / 回复；版本冲突时保留草稿、「读取最新进展」（`ops.js:128-164`） | `GET /api/ops/tickets`、`POST /api/ops/tickets/<id>` | 工单 SQLite 和审计 | 没有 | 保留 | 约 1 d |
| O11 | STL 素材库 | 按 SHA 或备注筛选、审阅状态；来源展开；审阅状态和备注（带版本号）；刷新版本、放弃修改；下载 ZIP（`ops.js:239-262`） | `GET /api/ops/assets`、`POST /api/ops/assets/<sha>`、`GET /api/ops/assets/<sha>/download` | 素材审阅状态；下载内容是 `input.stl` 加 `manifest.json` | 没有 | 保留 | 约 0.5 d |
| O12 | 用户与权限 | 按用户名 / 显示名、状态筛选；任务数 / 失败数；暂停或恢复访问（不能停用自己或最后一个管理员）（`ops.js:263-268`） | `GET /api/ops/users`、`POST /api/ops/users/<name>` `{disabled}` | `users.json` 的 disabled 字段（会结束该用户的会话）和审计 | 没有（v2 只能改自己的口令，`ws_admin.js:328-349`；其余账号管理在 CLI `user`） | 保留 | 约 0.5 d |
| O13 | 工单入口 / 返回工作台 | `ops.html:14, 20-21` | — | — | — | 保留；「返回工作台」和 brand 链接改指 `/v2/`（算 ops 改动，要用户同意）；不改的话，经过 `/` 的 302 也能到 | 5 min |

整页迁成 v2 的 `#/ops`（工作台标签页风格，6 个子页）：大约要写 1,200–1,500 行 JS 和 CSS，还要把 33 项 Firefox 验收
（`tests/ops_browser_check.py / _edges.py / _feed.py`）改写成 v2 的 devshot 流程，并重新验证冲突和草稿保护。
**合计约 3–4 人日**，风险在于重新引入 09-29 已经验收掉的并发问题。**不建议在 S7 做。**

### 1.3 `/support` 功能清单（`static/support.html` 17 行 + `support.js` 79 行 14 KB，标题「用户工单 · WSS 工作台」）

| # | 功能 | 控件 | 调用的接口 | v2 有没有对应 | 建议 | 工作量 |
|---|---|---|---|---|---|---|
| U1 | 登录 / 退出 | 同 O1（`support.js:68-69`） | `/api/session`、`/api/session/logout` | 有（同一个 cookie，在 v2 登录过就直接进） | 保留 | — |
| U2 | 提交工单 | 主题 ≤200、描述 ≤4000、关联任务 ID（可选，只能是自己的任务）、优先级（`support.js:70-73`） | `POST /api/support/tickets` | 没有 | S7：v2 帮助菜单加「反馈问题 / 我的工单」→ `/support`（新标签页）。之后（P4）可以迁成 v2 对话框：检查器「工具」页加「就这个结果提问题」，自动带上任务 ID | 链接约 1 h；v2 对话框约 1 d（300–400 行加 node 测试） |
| U3 | 我的工单列表 | 关键词 / 状态筛选、分页 25、10 s 自动刷新（`support.js:47-65`） | `GET /api/support/tickets` | 没有 | 同 U2 | 含在上面 |
| U4 | 详情与回复 | 回复历史；发送回复（带版本号，冲突时保留草稿）；读取最新回复；放弃草稿（`support.js:35-46`） | `GET/POST /api/support/tickets/<id>` | 没有 | 同 U2 | 含在上面 |
| U5 | 带任务 ID 打开 | — | — | — | `support.js` 现在不读取 URL 参数；想从 v2 带上 `?job=<id>` 需要改 support.js（约 10 行），属于 ops/support 改动 | 0.5 h |

### 1.4 用户现在怎么到这两页

| 入口 | 位置 | 现状 |
|---|---|---|
| 经典工作台 `index.html` / `app.js` | — | **没有链接**（grep `/ops`、`/support`、`工单`、`运维` 都是 0） |
| v2 工作区 | `static/v2/*` | **没有链接**（grep `/ops`、`/support`、`api/ops` 都是 0） |
| ops ↔ support | `ops.html:20`；`ops.js:67`「用户反馈可进入工单入口」 | 互相链接 |
| 服务端响应 / CLI | `cli.py`、`service.py`、`doctor.py` | 没有（cli 里只有两行"运维审计库不可用"的告警，`cli.py:131, 163`） |
| 文档 | `wss_deploy/README.md:26`、`OPERATIONS.md:9-10`、`OPERATIONS_ACCEPTANCE.md:10`、`docs/02-推进与变更/05-部署工具/README.md:16` | 唯一的"入口" |
| 验收实例 | `tests/ops_demo.py:123` 写到 `preview.json` 的 ops / support 地址 | 仅测试用 |

### 1.5 建议（与范围说明一致：`前端重构_范围说明_2026-09-30.md:35, 110`「运维中心、工单：不动」；草案 B1 `头脑风暴…md:532`「/ops、/support 本轮不主动修改；若共享样式受影响，仅验证其可用性」）

S7 选**"保留独立页，从 v2 加链接"**：

1. **不删 `static/app.css`**（也可以另存一份 ops/support 专用的精简副本，但那就是改 ops/support 了）。在 `STATIC_FILES`（`server.py:49-51`）里保留 `app.css, ops.html, ops.js, ops.css, support.html, support.js`。
   S7 前后都跑一遍 `ops_browser_*` 的 33 项检查，外加 320 / 390 px 布局检查。
2. v2 头像菜单：`!session.shared || session.role === 'admin'` 时加「运维中心」→ `/ops`（新标签页）。
   加在 `ws_admin.js` 的 `menuItems()`（`ws_admin.js:357-363`）里，替换掉 `ws_shell.js:261` 的「经典工作台」。约 1 h，含测试。
3. v2 帮助菜单（`ws_shell.js:237-243`）加「反馈问题（工单）」→ `/support`。约 0.5 h。
4. 以后（P4，要用户裁定）：把 `/support` 迁成 v2 对话框，这是面向医生的部分，约 1 d；
   `/ops` 继续作为独立管理员控制台，同类系统（比如 Django admin）通常也这样做。
5. 不建议删掉任何功能。

---

## 2. S7 清单：所有指向经典页面或文件的地方

### 2.1 server.py 路由

| 项 | 位置 file:line | 现状（返回什么） | 建议 | 工作量 |
|---|---|---|---|---|
| `GET /` | `server.py:1499-1507` | `static/index.html`（经典工作台），会新建本机会话 | **302 → `/v2/`**（保留 query；fragment 由浏览器自动带上，RFC 7231 §7.1.2）。v2 的 `parseHash`（`ws_shell.js:84-102`）要能认旧格式 `#job=<id>` 并转成 `#/job/<id>`，否则 `cli submit` 打印的旧链接和旧书签会落到首页 | 1 h |
| `GET /compare?left=A&right=B` | 同上（`compare.html`） | 两个 iframe 各开一份 `/api/jobs/<id>/report`，加 `/api/compare` | **302 → `/v2/#/job/A?v=compare&cmp=B`**（先用 `JOB_ID_PATTERN` 校验，不合法就跳 `/v2/`）；v2 的 `cmp` 参数见 `ws_shell.js:98, 481` | 1 h |
| `GET /ops`、`/ops/`、`/support`、`/support/` | 同上 | 运维和工单外壳 | 保留 | — |
| `GET /static/<name>` | `server.py:1499-1507`，白名单 `STATIC_FILES` 在 `server.py:49-51` | 17 个经典文件；**`/static/index.html`、`/static/compare.html` 也能直接打开经典页** | 删掉 `index.html, app.js, compare.html, compare.js, batch_export.js, workbench_core.js`；保留 `three.min.js, OrbitControls.js, report_common.js, volume_viewer.js`（v2 的 `legacy_scripts`，也内嵌进 report.html）、`app.css`（给 ops/support）、ops/support 五个文件；`glossary.json` 只有 `app.js:626` 会 fetch，但 `report_freshness.py:34` 和 `doctor.py:179-186` 还在用，可以保留或同步清理 | 0.5 h |
| `GET /api/jobs/<id>/report` | `server.py:1584-1600` | 任务的 `report.html`（经典壁面或体场查看器，内嵌 three.js）；返回前 `ensure_fresh` 会按模板刷新（`1588-1593`）；可以被同源 iframe 嵌入 | **需要用户裁定**：(a) 保留，作为可离线存档的单文件页，只是去掉所有 UI 入口；或 (b) 对浏览器导航（`Accept: text/html`）302 到 `/v2/#/job/<id>`。选 (b) 的话，经典 `#view=<b64>` 视图链接（`report.py:282-284, 1707, 1955`）会丢失，而且 `rehearse.py:193` 的烟测、`devshot.py:268`、下面 §2.4 列的测试都要改 | (a) 0；(b) 2 h |
| `GET /api/jobs/<id>/files/report.html` | 同上，`OUTPUT_FILES` 在 `server.py:127-129` | 同一个文件，作为下载 | 保留（它是数据容器，也在 zip 里） | — |
| `GET /jobs/<id>/<name>`（旧式地址） | `server.py:1584`（`legacy` 正则） | 任意输出文件，包括 report.html | 删掉，或者把 `report.html` 这一项 302 到 v2。grep 不到任何代码还在生成这种链接（不确定外部书签有没有） | 0.5 h |
| `GET /api/jobs/<id>/onepage` | `server.py:1573-1580` | 服务端渲染的一页纸 | 保留（v2 用 `urls.onepage`，`ws_api.js:169`） | — |
| `GET /v2`、`/v2/`、`/static/v2/*`、`/v2/example` | `server.py:1494-1498, 1287-1306` | v2 | 保留 | — |
| `POST /api/jobs/claim` | `server.py:1696-1708` | 旧会话任务认领 | **v2 没有 UI**（意外 4）→ S7 前先补上（放头像菜单或首页提示条，约 0.5 d），或者请用户裁定只保留 CLI | 0.5 d |
| `POST /api/compare` | `server.py:1709-1718` | 经典比较数字 | 只有 `app.js:1847`、`compare.js:212` 在用；v2 不用。可以保留当兼容接口，也可以删 | — |
| `GET /api/cases` | `server.py:1543-1545` | 经典病例分组列表 | 只有 `app.js:1077` 在用；v2 不用 | — |
| health 里的 `ui_build` | `server.py:837`；`static_build_id()` 在 `101-124` | 经典静态文件的摘要，`app.js` 用它提示刷新 | 删掉，或改成只覆盖 ops/support；`test_merge_v015.py:22-52` 要跟着改 | 0.5 h |
| `EMBEDDABLE_HTML` | `server.py:53` | 允许 report / onepage 被同源 iframe 嵌入（给经典比较页和 batch_export 用） | v2 没有 iframe（grep 为 0），S7 可以收紧；不急 | — |

### 2.2 会输出经典链接或文案的 Python 代码

| 项 | 位置 | 现状 | 建议 |
|---|---|---|---|
| `cli submit` 输出 | `cli.py:313` | 打印 `{base}/#job=<id>` | 改成 `{base}/v2/#/job/<id>`；v2 认旧 hash 后，旧格式也能用 |
| 服务地址输出 | `service.py:598, 615, 1084-1085` | `http://host:port/` | 改成 `/v2/`（经过 302 也能到） |
| `rehearse` 烟测 | `rehearse.py:190-198` | GET `/api/jobs/<id>/report`，要求 200、`text/html`、正文含 `<html` | 如果 /report 改成 302，就改测 `/api/v2/jobs/<id>/manifest` 加 `/onepage` |
| devshot 截图套件 | `devshot.py:134-140`（登录）、`259-273`（suite）、`318`（`/compare` 前先开 `/`） | 全是经典页 | **必须改**：登录改填 v2 表单；suite 改开 `/v2/`、`/v2/#/job/<id>`、`…?v=compare&cmp=`、`/onepage` |
| zip README | `bundle.py:54, 58` | 「report.html 可离线打开（three.js 内嵌）」「三维报告导出的一页纸配图」 | 跟着 §2.3 的决定改文案；如果 zip 改放 v2 离线页，改 `write_job_bundle`（`bundle.py:95-118`），由 `OUTPUT_FILES` 决定成员 |
| 一页纸提示 | `onepager.py:40` | 「在三维报告『导出』菜单点『生成一页纸配图』」 | 改成 v2 的「导出 → 一页纸配图」（`ws_figure.js:953`） |
| 形态提示 | `morphology.py:1001` | 「请以三维报告中的截面环核对」 | 只是文案，改成「工作区截面」 |
| 术语表 | `glossary.py:62` | 「三维报告的顶点颜色……」 | 只是文案 |
| 发布包说明 | `build_cycle_release.py:141` | 「报告页可在 WSS / TAWSS / OSI 之间切换」 | 只是文案 |
| narrative / export_table / notifications / webhook / 邮件 | — | 没有链接；**没有邮件或 webhook**（grep 为 0）；v2 桌面通知点击后调 `go(job_id)`（`ws_admin.js:200-201`） | — |
| `v2_examples.py` | 整个模块 | 生成 `example_aaa.stl` 和 `example_report.html`（v2 离线页） | S7 后重新运行，因为当前产物已过期 |
| `v2_offline.py` | 整个模块 | 按 bundle.json 内联；v2 源码里有 `index.html:61` 的 noscript 链接和 `ws_main.js:31` 的经典链接，都会被带进离线页 | 改源文件后，离线页跟着更新 |
| 帮助页 | `static/v2/help_quickstart.html:28` | 「（经典工作台是『新建预测』）」 | 删掉括号 |
| 文档 | `wss_deploy/README.md:24`（打开 `/`）；`env/GO_LIVE.md:48`（登录后"勾『全部用户』"）、`:34`（「报告模板：刷新 N 个」）；`WORKSPACE_V2_CONTRACT.md:37, 40, 448`；`ANALYSIS_CONTRACT.md:558, 571`（`#job=` 路由） | 描述的是经典入口 | S7 合并时一起改 |

### 2.3 任务目录里生成的经典页面

| 产物 | 谁写 | 谁重写 | 谁读 | S7 能不能动 |
|---|---|---|---|---|
| `<job>/report.html`（壁面） | `pipeline.py:686-697`（`R.build_html`，之后 `update_html_meta`），列入 run_manifest 的 outputs（`pipeline.py:714`） | `rebuild_report.py:341`（完整重建）、`rebuild_report.refresh_ui_only`（`rebuild_report.py:403-450`，数据逐字节不变，只换模板和查看器）、`report_freshness.ensure_fresh`（打开 /report 时触发，`server.py:1588-1593`）、`cli reports refresh [--job/--all/--check]`（`cli.py:217-240`）、`service upgrade` 升级后的批量刷新（`jobs.py:3114-3130`）；审阅和元数据改动走 `update_html_meta`（`jobs.py:996-1000, 1041-1045`） | **v2_data.py:396**（全部 v2 数据）、`jobs.py:583`（旧任务元数据）、zip（`OUTPUT_FILES`）、`housekeeping.py:109`（占用统计）、`/api/jobs/<id>/report` | **不能停止生成**（意外 1） |
| `<job>/report.html`（体场） | `volume_pipeline.py:246-259`（outputs 在 `:262`） | `rebuild_report.py:194`，其余同上 | 同上 | 同上 |
| `<job>/report_ui.json` | `report_freshness` | 刷新时写 | 过期判断、health 里的 `stale_reports`（`server.py:804-816, 854-862`） | 跟着 report 模板走 |
| `static/v2/example_report.html` | `v2_examples.py` | 手动运行 | `/v2/example` | 重新生成 |

**两种做法（需要用户裁定）**

- **S7a（低风险，建议先做）**：report.html 照常生成，内容不变（数据容器，同时是离线可开的经典页）。
  只去掉所有 UI 入口，`/`、`/compare` 做跳转，删掉经典工作台和比较页的文件。经典查看器代码继续留在 `report.py/volume_report.py` 里，但不再有入口。
- **S7b（真正"只维护一套查看器"）**：
  1. 把 `report.py/volume_report.py` 的模板改成只放 `<script id="wss-report-meta">` 加数组的"数据页"，保持 `v2_data._Report` 需要的 id；
  2. 在 zip 里放 v2 离线页，替代 report.html（`v2_offline`，单文件约 7 MB）；
  3. 退役 `ensure_fresh`、`refresh_ui_only`、`reports refresh` 和 `stale_reports`，或者让它们改做"数据页规范化"；
  4. 删掉 test_report / test_volume_report 里的经典 UI 测试。

  代价：
  - 所有旧 report.html 会被判过期，升级时整体重写一遍（数据逐字节不变，但 mtime 变了，v2 的 `data_version` 和 ETag 全部失效一次）；
  - 每个任务能省下约 0.8–1.0 MB 的内嵌查看器代码。样本 11 个任务的 report.html 平均 6.7 MB，范围 4.3–13.5 MB。

### 2.4 覆盖经典页面的测试

| 测试文件 | 大致覆盖 | S7a 怎么处理 | S7b 追加 |
|---|---|---|---|
| `tests/test_workbench_js.py`（1347 行，31 个） | 对 `app.js / workbench_core.js / batch_export.js / compare.js` 做 node 语法检查（`:32-36`）；DOM 无关的纯函数；启动桩；批量导出协议 | **删除** | — |
| `tests/test_compare_page.py`（274 行，7 个） | `/compare` 的页头和 CSP、白名单、相机和显示状态中继、共享色标 | **删除**；新增一个 302 用例 | — |
| `tests/test_server_security.py`（737 行，24 个） | `:148` GET `/`；`:454-478` 用 `/static/app.js` 测 gzip 和 ETag、测 `/`、测 `/report` 的 gzip / ETag / 304；ensure_fresh 3 个 | 改成用 `/static/v2/core_util.js` 或 `/static/three.min.js`；`/` 断言改成 302 | ensure_fresh 3 个跟着改 |
| `tests/test_v2_routes.py:302` | 断言 `/static/app.js` 返回 200（注释写着"经典工作台不变"） | 改成 404，并加 `/` → `/v2/` 的用例 | — |
| `tests/test_v2_lane_c.py:63-108` | **拿经典 `workbench_core.js` 当对照**（`binEdges / histogram / populationValues / rerunSkipReason`） | 删掉 workbench_core.js 之前，先把对照值固化成常量 | — |
| `tests/test_v2_lane_a_export.py:313-316` | 从 `report.py` 源码里抽出 `standardViews` 当对照 | 不受影响 | 固化成常量 |
| `tests/test_report_common.py:294` | 扫描 `static/app.js`、`static/compare.js` 等源码 | 名单里去掉这两个 | — |
| `tests/test_glossary.py:11-12` | 检查各页 `data-gloss` 键，包括 `index.html / app.js / compare.js` | 名单里去掉 | — |
| `tests/test_merge_v015.py:22-52` | `static_build_id()` 的经典范围（monkeypatch 了 `STATIC_FILES`） | 跟着改 | — |
| `tests/test_report.py`（1701 行，74 个，其中 16 处调 node） | 前半是 Python 计算（保留）；后半是经典壁面查看器 UI：`test_inline_report_scripts_parse_in_node`、`test_domless_page_*`、`test_viewer_announces…`、`test_no_webgl_fallback…`、预设、测量、探针和标注、发现判定、术语嵌入等 | 不动 | 删掉 UI 部分 |
| `tests/test_volume_report.py`（2328 行，81 个） | 体场导出和数值（保留）；经典体场查看器 UI（菜单、标签页、无 WebGL 查看器、截面手势）；`VolumeViewerCore` 的数值测试要保留，v2 依赖它 | 不动 | 删掉 UI 部分 |
| `tests/test_report_freshness.py`（7 个）、`test_rebuild_report_v014.py`（4 个） | 模板刷新、只刷 UI 的重建 | 不动 | 跟着改或删 |
| `tests/test_onepager.py:162`、`tests/test_trash.py:85` | GET `/api/jobs/<id>/report` | 不动 | 如果 /report 改 302，就改断言 |
| `tests/test_ops_v015.py:458, 471`（rehearse） | `rehearse.py:193` 请求 /report | 不动 | 同上 |
| `tests/test_devshot.py` | 过滤工具和登录参数 | 如果改了 `Browser.login`，要补用例 | — |
| `tests/test_operations_http.py:79`、`test_operations_acceptance.py:52`，以及 `ops_browser_check/edges/feed.py`、`_ops_browser_viewport.py`、`ops_demo.py` | `/ops` 返回 200；Firefox 33 项 | **保留**，删 app.css 前后各跑一遍 | — |

### 2.5 黄金回归 `regress.py`

- 只比较**数字**：`summary.json` 里的 `SUMMARY_KEYS`（`regress.py:24-26`）和 `field.npz`（`compare_arrays`，`regress.py:65-92, 133-140`）。**不比较 report.html。**
- 但 `stage_b` 会顺带写 report.html（`pipeline.py:686`），计时里的 `export` 也包含这一步。所以 S7b 瘦身模板**不影响**判定；
  但如果删掉 report 生成，v2 会整体失效（意外 1），run_manifest 的 outputs 列表也会变（`pipeline.py:714`、`volume_pipeline.py:262`）。
- 结论：S7a 不影响黄金回归；S7b 的通过与否也不看回归结果，但要另外验证 v2 数据页逐字节一致（`test_v2_data.py` 的思路：manifest 解码后和内嵌值逐字节相等）。

### 2.6 v2 里仍然指向经典页的代码

| 项 | 位置 | 类型 | 建议 |
|---|---|---|---|
| `api().urls.report` | `ws_api.js:168` | 生成 `/api/jobs/<id>/report` | S7b 删掉；S7a 可以保留给「工具」页 |
| `api().urls.classic` | `ws_api.js:173` | 生成 `/#job=<id>` | 删掉 |
| 头像菜单「经典工作台」 | `ws_shell.js:245, 261` | 链接 | 换成「运维中心」（仅管理员或本机）；`ws_onepage.js:147` 的菜单顺序注释跟着改 |
| 结果加载失败时的「在经典报告中打开」 | `ws_shell.js:501` | 链接 | 改成「重试 / 导出离线报告」 |
| 未完成任务页的「在经典工作台中处理」 | `ws_shell.js:509` | 链接 | **先补上换入口 / 重提中心线**（意外 4），再删 |
| 「工具」页「经典报告」一节 | `ws_shell.js:1785-1786` | 按钮 | 删掉 |
| 登录页「也可以回到经典工作台」 | `ws_shell.js:1985` | 链接 | 删掉 |
| 致命错误页「经典工作台」 | `ws_main.js:31` | 链接 | 删掉 |
| noscript 提示 | `static/v2/index.html:61` | 链接 | 删掉 |
| 「查看器没有加载……仍不行时用经典报告」 | `ws_shell.js:386` | 文案 | 改文案 |
| 口径说明里提到"经典报告" | `ws_annot.js:164`、`ws_review.js:128`、`ws_probe.js:322`、`ws_slice.js:824, 918, 973`、`ws_display.js:376, 434, 435`、`ws_region.js:211`、`ws_figure.js:953` | 文案 | 改成「与旧版报告同口径」或删掉 |
| 过期的 `example_report.html` | `static/v2/example_report.html` | 生成文件 | 重新生成 |

---

## 3. 加载速度（静态测量）

### 3.1 `/v2/` 首屏要加载什么

加载顺序按 `static/v2/bundle.json` 和 `index.html:8-56`：6 个 CSS，4 个遗留脚本，38 个 v2 脚本，全部 `defer`。
gz 列是用 `gzip -6` 实测的，规则和服务器一样：大于 8 KB 才压缩（`server.py:182-185`）。

| 组 | 文件数 | 原始字节 | gzip 后 |
|---|---:|---:|---:|
| `index.html` | 1 | 3,598 | 3,598（小于 8 KB，不压缩） |
| 样式 `v2*.css` | 6 | 104,685 | 32,184 |
| 遗留脚本 | 4 | 996,791 | 277,640 |
| └ `three.min.js`（r128） | | 603,445 | 149,202 |
| └ `volume_viewer.js`（前约 75.5 KB 是 `VolumeViewerCore`，其余约 224 KB 是经典体场页代码，在 v2 里一加载就在 `:1024` 返回） | | 300,034 | 97,530 |
| └ `report_common.js` | | 66,937 | 25,505 |
| └ `OrbitControls.js` | | 26,375 | 5,403 |
| v2 脚本 | 38 | 1,067,103 | 354,561 |
| └ 最大的几个：`ws_shell` 127.5 K、`ws_admin` 88.6 K、`ws_slice` 80.6 K、`ws_figure` 73.2 K、`core_viewer` 73.2 K、`ws_overview` 67.1 K、`ws_detail` 63.8 K | | | |
| **合计** | **49 个请求** | **2,172,177（2.07 MiB）** | **667,983（约 652 KiB）** |

对照：经典 `/` 有 8 个请求，约 1.16 MB 原始，约 318 KB gz。v2 的字节数约为经典的 2.1 倍，请求数是 6 倍。

首屏拿到脚本之后，v2 还要调用这些接口：
- `/api/session`，然后并行 `/api/v2/model-cards` 和 `/api/releases`，然后 `/api/jobs`（管理员是 `?all=1`）；
- `/api/preferences`、`/api/health`，以及 `/api/events`（SSE，长期占着一条连接）；
- **首页病例卡片的缩略图**（`ws_shell.js:293-296`，`ws_thumbs.js:129-145`）：每个已完成的病例依次拉 manifest 和 3 个数组（显示网格的顶点、面和显示字段）。
  样本里每张卡的网格顶点加面是 408 KB–4.6 MB，数组按合同**不压缩**（`server.py:1396-1397`）。最多 48 张卡（`ws_rail.js:303`）。
  缩略图只缓存在内存里（`ws_thumbs.js:14`），每次重新加载页面都要重拉；有 ETag，所以二次访问是 304，但仍然要发请求，
  服务端也要重建 manifest（report 解析缓存只有 6 个任务，`v2_data.py:39`；解析一次 20–90 ms）。

### 3.2 服务端缓存和压缩现状

| 项 | 位置 | 现状 |
|---|---|---|
| Cache-Control（静态） | `server.py:186` `STATIC_CACHE = "private, max-age=0, must-revalidate"`；`_v2_static` 在 `:1305`，经典静态在 `:1506` | **每次访问都要把 49 个文件逐个重新验证**（返回 304） |
| ETag | `_send_open`（`server.py:967-972`），`job_etag` = inode、大小、mtime 的弱 ETag（`:378-379`），前缀 `s` | 有，304 正常 |
| gzip | `GZIP_TYPES` 覆盖 html / js / css / json（`:185`）；大于 8 KB 才压缩；级别 6；`GzipCache` LRU 256 MB（`:382-402, 756`） | 有，只压一次 |
| 连接 | `Handler` 没有设 `protocol_version`，所以是 BaseHTTPRequestHandler 默认的 **HTTP/1.0**，每个请求新建一条 TCP 连接（`server.py:866` `class Handler`；我没有抓包实测，这是按默认值推断的） | 49 个请求，每个都要握手；浏览器每主机最多 6 条并发，SSE 还长期占一条 |
| 构建摘要 | `static_build_id("v2")`（`:101-124`）已经存在；health 登录后带 `ui_build_v2`（`:838`）；v2 用它提示"页面文件已更新"（`ws_admin.js:288-300`） | 可以直接拿来做版本号 |
| 数组 | `/api/v2/jobs/<id>/arrays/<key>`：`private, no-cache` 加 ETag，不压缩 | — |

### 3.3 最便宜、最安全的改进（按性价比排序）

| # | 改进 | 做法 | 收益 | 风险 | 工作量 |
|---|---|---|---|---|---|
| 1 | **带版本号的 URL 加长缓存** | `/v2/` 返回 index.html 时，在服务端把 `/static/…js|css` 替换成 `…?v=<static_build_id('v2')>`（按 build 缓存；磁盘文件不变，`test_v2_detail.py:65` 的正则不受影响）。请求带的 `v` 等于当前 build 时，回 `Cache-Control: private, max-age=31536000, immutable`；不带 `v` 时维持现有 ETag 行为。经典静态分支也这样处理 4 个遗留脚本（`_path()` 已经会去掉 query） | 二次访问从 49 次重新验证变成 1 次（只剩 HTML）；经 SSH 隧道或跨网段访问时收益最大 | 低：版本号来自内容摘要，改了文件就换 URL；离线打包直接读磁盘，不受影响 | 约 0.5 d，含测试 |
| 2 | **缩略图持久缓存** | `ws_thumbs` 把渲染好的 PNG data URL（约 20–40 KB）存进 IndexedDB 或 localStorage，键用 `job.id + job.version`（重跑、重建会改 version）；读取失败就退回现有流程 | 首页每次重载省掉每张卡 0.4–4.6 MB 的数组下载和一次 manifest 重建；48 张卡最多能省几十 MB | 低：只影响显示；读写都包 try/catch | 2–4 h |
| 3 | **按需加载 `volume_viewer.js`** | 从 `index.html` 和 `legacy_scripts` 里拿掉这个文件，加一个 `ns.util.need('volume')` 的 Promise 加载器。在第一次打开体场结果、截面（`ws_slice.js:29-30`）、区域统计（`ws_region.js:20-21`）、`ws_figure` 截面图时等待加载；`ws_display.js:44`、`adapter_volume.js:69, 311` 已经有回退。离线包照旧内联（`v2_offline` 要改成从单独的名单读） | 首屏少 300 KB 原始、97.5 KB gz，约占 JS 解析量的 14% | 中：同步调用点要改成异步。**不要拆分或修改 volume_viewer.js 本身**，否则所有报告被判过期并重刷（`report_freshness.py:33-34`） | 0.5–1 d |

还可以做，但靠后：
- 把 42 个脚本在服务端拼成一个（和离线包内联用同一份名单），请求数从 42 变成 1，约 0.5–1 d，和 #1 叠加效果最好；
- 开 HTTP/1.1 keep-alive：SSE 和所有响应都要带正确的 Content-Length 或分块，风险中等，不建议在 S7 做；
- 首页不需要 `three.min.js`：做不到，缩略图也要用 WebGL 渲染。

---

## 4. S7 前待用户裁定的事项

1. `/api/jobs/<id>/report`：保留当存档页（S7a），还是 302 到 v2？经典 `#view=` 链接会丢失。
2. report.html 要不要瘦身成纯数据页，zip 改放 v2 离线页（S7b）？
3. 「认领旧会话任务」和「换入口 / 重提中心线」：先补进 v2，还是明确删掉、只留 CLI？
4. `/ops` 和 `/support`：同意"保留独立页，保留 app.css，从 v2 加入口"吗？「返回工作台」改指 `/v2/`（这算改动 ops/support）要不要做？
