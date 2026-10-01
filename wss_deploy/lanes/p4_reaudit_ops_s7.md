# S7 后复审：运维中心 / 工单、旧页面引用、加载速度

对象：`/public/newhome/cy/Digital_twin/GNN_wssui_v2`（分支 `wss-ui-v2`，HEAD `e7c2356`；第三期基线 `866cf02`，S7 合并 `0a6c51a`，衔接修改 `9e4501c`）。
依据：`wss_deploy/lanes/p3_audit_ops_s7.md`（下称「原审计」）、`wss_deploy/lanes/p3_s7.md`、`wss_deploy/PHASE3_LANES.md` §0/§3/§5。
方法：只读（grep / sed / git show / `node --check`）。没有启动服务、浏览器，也没有跑 pytest。行号都指当前工作树。

状态写法：**DONE**（已做，给 file:line）、**STILL-OPEN**（还缺什么）、**KEPT**（按裁定有意保留）。

> 自查：我用 `python3 -m py_compile` 检查了 6 个 Python 文件，它在 `wss_deploy/__pycache__/` 里新写了 6 个 `*.cpython-38.pyc`
> （bundle / cli / devshot / rehearse / server / service，14:16）。这些是 gitignore 的缓存，服务跑的是 3.10，不会读它们；需要时可直接删掉。除此之外没有改动任何文件。

---

## 0. 结论

- 第 1 部分共 **104 行**（有几项在原审计的几节里重复出现，这里也重复列）：**DONE 51**、**KEPT 34**、**STILL-OPEN 19**。
  STILL-OPEN 去重后是 **15 项**：
  - 要处理的：`example_report.html` 过期；33 项 ops Firefox 验收没有重跑；
  - 文档：`README.md:24`；`README.md:26` 与 `OPERATIONS*.md:3`；`GO_LIVE.md`；`ANALYSIS_CONTRACT.md`；
  - 小清理：`ui_build`；`EMBEDDABLE_HTML`；`build_cycle_release.py:141`；
  - 有意推迟或可选：S7b 数据壳；`volume_viewer.js` 按需加载；脚本拼接；HTTP/1.1；support `?job=` 预填；`/support` 迁成 v2 对话框（P4）。
- S7 **没有弄坏**任何运行路径：
  - `STATIC_FILES` 的 11 个文件都在；`bundle.json` 列的文件都在；`index.html` 的 46 个脚本和 7 个样式与 `bundle.json` 顺序一致。
  - `report_freshness.UI_SOURCES` 的 7 个文件都在，且与 `866cf02` 逐字节相同，所以不会把所有报告判为过期。
  - `doctor.check_glossary` 读的 `static/glossary.json` 还在。
  - 没有 Python 代码读取已删除的 6 个文件。
  - v2 源码里没有指向经典页的链接（`test_v2_p3_s7.py:635-655` 也会拦）。
  - `ops.js`、`support.js` 只调用 `/api/session`、`/api/ops/*`、`/api/support/*`；`node --check` 通过。
- 第 2 部分找到 **13 条遗留**。真正会被用户或运维看到、而且说法已经不对的有 4 条：
  1. `cli.py:328` 让人去网页点「认领」，但这个按钮已经没有了；
  2. 首页「示例报告」打开的 `example_report.html` 是 09-30 12:20 生成的旧版，里面还有「经典工作台」等字样；
  3. `docs/02-推进与变更/05-部署工具/前端重构_验收说明_2026-09-30.md:202` 的上线步骤还是 `git merge --ff-only`，现在会失败；
  4. `env/GO_LIVE.md:34/48/49` 的上线验证步骤还是旧说法。
- **测试覆盖的隐性退化**：第 2、3 路的几个测试原本拿经典 `workbench_core.js` / `app.js` 现场对照，文件删除后这部分**静默跳过**，没有固化成常量（`alertTitle` 已经完全没有断言）。
- `/ops`、`/support` 在没有经典工作台的情况下照常可用（第 3 部分）。**不建议现在迁进 v2。**

---

## 1. 原审计逐项状态

### 1.1 原审计 §0「五个意外」及附注

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| 意外 1：v2 数据从 `report.html` 读 | `pipeline.py` / `volume_pipeline.py` 照常生成；`report.py`、`volume_report.py` 与 `866cf02` 逐字节相同 | **KEPT**（`PHASE3_LANES.md` §0.3） | 数据壳 S7b 没做，经典查看器代码仍在模板里 | 单独一轮再做 | — |
| 意外 2：`/ops`、`/support` 依赖 `app.css` | `server.py:52-53` 白名单保留 `app.css`；`app.css` 未改 | **KEPT** | — | — | — |
| 意外 3：两个工作台都没有入口链接 | `ws_shell.js:252`「反馈问题」→ `/support`；`ws_shell.js:256-257, 272`「运维中心」→ `/ops`（管理员或本机） | **DONE** | — | — | — |
| 意外 4a：认领旧会话任务只有经典 UI | 裁定放弃（`PHASE3_LANES.md` §0.1）；路由 `server.py:1800` 仍在，无 UI | **KEPT** | `cli.py:328` 还提示「点『认领』」（见 §2 F1） | 改 CLI 提示 | 5 min |
| 意外 4b：换入口 / 重提中心线 | `ws_input.js:591-605`（第 1 路，`/confirm {inlet}`） | **DONE** | — | — | — |
| 意外 5：devshot 登录在 `/` 跳转后静默失效 | `devshot.py:135-150` 改填 `#wsl-user/#wsl-pass`，表单仍在时报错；`suite` 在 `devshot.py:250-290` | **DONE** | `suite` 不截 `/ops`、`/support` | 可选：suite 加两页 | 15 min |
| 附注：`example_report.html` 过期 | `static/v2/example_report.html`（gitignore，09-30 12:20） | **STILL-OPEN** | 含「经典工作台」×5、「在经典报告中打开」×2、`CLASSIC_TOOLS`；首页「示例报告」链接到它（`ws_rail.js:505, 652`） | 上线后按 `p3_s7.md` §9 跑 `v2_examples` | 5 min |
| 附注：遗留脚本指纹 | `report_freshness.py:33-34` | **KEPT**（7 个源文件未改） | — | 继续别动 | — |

### 1.2 原审计 §1 运维中心与工单

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| §1.1 外壳不需要登录 / 运维接口管理员门禁 / 工单 owner 范围 / CSRF / 数据位置（5 行） | `server.py:1599-1611`（外壳），`1615-1624`（门禁），`operations_http.py` 未改 | **KEPT** | — | — | — |
| O1 登录 / 退出 | `ops.js` 未改 | **KEPT**（独立页） | — | — | — |
| O2 自动刷新 | 同上 | **KEPT** | — | — | — |
| O3 概览指标卡 | 同上 | **KEPT** | — | — | — |
| O4 提醒条 | 同上 | **KEPT** | — | — | — |
| O5 服务健康 | v2 头像菜单已有「运维中心」（`ws_shell.js:256-257`） | **DONE**（入口用菜单代替对话框里的链接） | — | — | — |
| O6 事件流 | `ops.js` | **KEPT** | — | — | — |
| O7 导出事件 JSON | 同上 | **KEPT** | — | — | — |
| O8 跨账号任务检索与回收站 | v2 不做 `/api/trash?all=1`（`PHASE3_LANES.md` §0.1） | **KEPT**（只在 `/ops` 里有） | — | — | — |
| O9 归档 STL | `ops.js` | **KEPT** | — | — | — |
| O10 工单处理 | 同上 | **KEPT** | — | — | — |
| O11 STL 素材库 | 同上 | **KEPT** | — | — | — |
| O12 用户与权限 | 同上 | **KEPT** | — | — | — |
| O13「返回工作台」和 brand 链接改指 `/v2/` | `ops.html:14, 21`；`support.html:5` | **DONE** | — | — | — |
| U1–U4 工单的登录、提交、列表、详情 | `support.js` 未改 | **KEPT** | — | — | — |
| U2 的入口链接 | `ws_shell.js:252` | **DONE** | — | — | — |
| U5 `/support?job=<id>` 预填任务 | `support.js` 不读 URL 参数 | **STILL-OPEN**（可选） | v2 里没法「就这个结果提问题」 | P4；要改 support.js，需用户同意 | 0.5–1 h |
| §1.4 入口（v2 里有链接） | 同意外 3 | **DONE** | — | — | — |
| §1.4 文档里的入口说明 | `README.md:26`「现有主页保持原样」；`OPERATIONS.md:3`；`OPERATIONS_ACCEPTANCE.md:3`（「原有 index.html、app.js…保持原样」） | **STILL-OPEN** | 说法已过时；`OPERATIONS.md`「入口与权限」没写 v2 菜单入口 | 改三句文档 | 10 min |
| §1.5-1 不删 `app.css`，白名单保留 6 个文件 | `server.py:52-53` | **DONE** | — | — | — |
| §1.5-1 S7 前后各跑一遍 33 项 Firefox 验收（`tests/ops_browser_*.py`） | `lanes/p3_s7.md` §8 只有截图 `17_ops`、`18_support` | **STILL-OPEN** | 没看到跑过的记录。风险低：`app.css`、`ops.*`、`support.js` 都没改，只改了 href | 上线前在沙箱跑一次 | 10–15 min |
| §1.5-2 头像菜单「运维中心」 | `ws_shell.js:256-257, 272`；测试 `test_v2_p3_s7.py:585-633` | **DONE** | — | — | — |
| §1.5-3 帮助菜单「反馈问题」 | `ws_shell.js:252` | **DONE** | — | — | — |
| §1.5-4 把 `/support` 迁成 v2 对话框（P4） | — | **STILL-OPEN**（待裁定） | — | 见第 3 部分 | 约 1 d |
| §1.5-5 不删任何功能 | — | **DONE** | — | — | — |

### 1.3 原审计 §2.1 路由

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| `GET /` | `server.py:1390-1391`（302 到 `/v2/`，保留 query）；旧 hash 由 `ws_legacy.js:60, 358-371` 转换 | **DONE** | — | — | — |
| `GET /compare` | `server.py:1392-1399` | **DONE** | — | — | — |
| `/ops`、`/support` | `server.py:54, 1602` | **KEPT** | — | — | — |
| `/static/<name>` 白名单 | `server.py:52-53`：去掉 6 个经典文件，`/static/index.html`、`/static/compare.html` 返回 404 | **DONE** | — | — | — |
| `GET /api/jobs/<id>/report` | 页面访问 302（`server.py:1400-1402`）；程序读取和 iframe 照旧（`1688-1705`） | **DONE**（裁定 b） | — | — | — |
| `/api/jobs/<id>/files/report.html` | 只在 `Sec-Fetch-Dest: document` 时 302（`server.py:1403-1405`） | **DONE** | 老浏览器直接打开仍看到经典页（`p3_s7.md` §4 已写明） | — | — |
| `/jobs/<id>/<name>` 旧式地址 | `report.html` 的页面访问 302（`server.py:1400`）；其他文件照旧（`1688`） | **KEPT**（部分处理） | — | — | — |
| `/api/jobs/<id>/onepage` | `server.py` 一页纸路由 | **KEPT** | — | — | — |
| `/v2/`、`/static/v2/*`、`/v2/example` | `server.py:1340-1365` | **KEPT** | — | — | — |
| `POST /api/jobs/claim` | `server.py:1800` | **KEPT**（裁定放弃 UI；接口和测试保留） | 没有网页调用方 | 可留作兼容 | — |
| `POST /api/compare` | `server.py:1813`；v2 现在在用（`ws_compare.js:163`） | **KEPT** | — | — | — |
| `GET /api/cases` | `server.py:1647` | **KEPT**（裁定为经典专用接口，不迁移） | 没有调用方，只有 `test_cases.py`、`test_users.py` 在测 | 可留 | — |
| health 里的 `ui_build` | `server.py:875`；`static_build_id()` 在 `110-130` | **STILL-OPEN** | 没有消费方了（`app.js` 已删，`ops.js` 不读）。现在摘要覆盖的是 `app.css` + 4 个报告脚本 + ops/support；docstring 还写着「classic workbench」 | 删掉字段，或改名说明用途（同时改 `test_v2_routes.py:347`、`test_merge_v015.py`） | 0.5 h |
| `EMBEDDABLE_HTML` | `server.py:56-57` | **STILL-OPEN**（不急） | v2 没有 iframe；注释还写「side-by-side comparison」 | 以后把 `report.html` 改成 DENY | 0.5 h |

### 1.4 原审计 §2.2 输出经典链接或文案的 Python 代码

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| `cli submit` 链接 | `cli.py:313` → `{base}/v2/#/job/<id>` | **DONE** | 同一函数 `cli.py:328` 的「认领」提示没改（F1） | — | — |
| 服务地址 | `service.py:598, 615, 1084` → `/v2/` | **DONE** | — | — | — |
| `rehearse` 冒烟 | `rehearse.py:181-210`（`/v2/` 带 `?v=`、manifest、旧报告地址落到 `/v2/?job=`、一页纸） | **DONE** | — | — | — |
| devshot | 见 §1.1 | **DONE** | — | — | — |
| zip README | `bundle.py:63, 75-81` | **DONE** | — | — | — |
| 一页纸提示 | `onepager.py:40` | **DONE** | — | — | — |
| 形态提示 | `morphology.py:1001` | **DONE** | 旧任务 `summary.json` 里存的是旧句子。预览副本里没找到，影响很小 | — | — |
| 术语表 `glossary.py:62` | 与 `glossary.json:92` 同步 | **KEPT**（改了会让所有报告判为过期） | 一页纸术语「all」模式（`onepager.py:931-940`）会显示「三维报告的…」 | 下次动报告模板时一起改 | — |
| 发布包说明 | `build_cycle_release.py:141`「报告页可在…切换」 | **STILL-OPEN**（低） | 不在 S7 合同里；「报告页」泛指，对 v2 也成立 | 下次生成发布包时顺手改成「结果页」 | 5 min |
| `v2_examples.py` | — | **STILL-OPEN** | 见 §1.1 附注 | 上线后运行 | 5 min |
| `v2_offline.py` | 读当前源码，不含经典链接（`ws_legacy.js` 在 `offline_exclude` 里） | **DONE** | — | — | — |
| 帮助页 | `help_quickstart.html` 里已没有「经典」 | **DONE** | — | — | — |
| 文档 `WORKSPACE_V2_CONTRACT.md` | `:41, :451` 已写 S7 | **DONE** | `:37`「与 `/` 相同的会话处理」、`:441`「队列总览与回收站的经典入口」已过时 | 改两句 | 5 min |
| 文档 `README.md:24` | 「浏览器打开 `http://master:8765/`」 | **STILL-OPEN** | 经过 302 也能打开，只是不准确 | 改成 `/v2/` | 2 min |
| 文档 `env/GO_LIVE.md:34, 48, 49` | 「服务已就绪：…:8765/」、「勾『全部用户』」、「打开…三维报告」 | **STILL-OPEN** | 实际输出是 `…/v2/`，v2 叫「看全部用户」，三维报告已下线 | 改上线验证表 | 10 min |
| 文档 `ANALYSIS_CONTRACT.md:553, 558, 571` | `…/#job=<id>` 链接、`#job=` 路由 | **STILL-OPEN**（低） | 这是历史合同 | 加一行「S7 后见 WORKSPACE_V2_CONTRACT」 | 5 min |

### 1.5 原审计 §2.3 任务目录里生成的页面

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| `<job>/report.html`（壁面、体场） | 生成和重写的链路都未改 | **KEPT** | — | — | — |
| `<job>/report_ui.json` | `report_freshness` | **KEPT** | — | — | — |
| `static/v2/example_report.html` | 生成文件 | **STILL-OPEN** | 同 §1.1 附注 | — | — |
| S7a：页面下线、数据包改放新版离线页 | `bundle.py:9-12, 28-29, 118-132`；`server.py:1260-1305`（`_bundle_offline`，生成失败时退回经典页） | **DONE** | 多任务打包时每个任务都要生成约 7 MB 的离线页，最多 50 个任务时的耗时没有实测 | 上线后打一次 10 个以上任务的批量包，看耗时 | 15 min |
| S7b：数据壳 | — | **STILL-OPEN**（推迟，`p3_s7.md` §7.1） | 会让所有报告判为过期，还要删改约 60 个测试 | 单独一轮 | 1–2 d |

### 1.6 原审计 §2.4 测试，§2.5 黄金回归

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| `test_workbench_js.py`（31 个） | 已删除 | **DONE** | — | — | — |
| `test_compare_page.py`（7 个） | 已删除；302 用例在 `test_v2_p3_s7.py:46-55` | **DONE** | — | — | — |
| `test_server_security.py` | `/v2/`（约 `:148`）；gzip / ETag 改用 `/static/v2/ws_shell.js`（`:453-459`） | **DONE** | — | — | — |
| `test_v2_routes.py:302` | `:302-304`：`/static/app.js` 返回 404，`/` 返回 302 | **DONE** | — | — | — |
| `test_v2_lane_c.py` | 对照值固化为 `CLASSIC_CORE`（`:19`） | **DONE** | — | — | — |
| `test_v2_lane_a_export.py` | 不受影响 | **KEPT** | — | — | — |
| `test_report_common.py:294`、`test_glossary.py:11` | 扫描名单去掉已删文件 | **DONE** | — | — | — |
| `test_merge_v015.py:22-52` | 用的是合成的 `STATIC_FILES`（含假 `app.js`），测的是机制 | **KEPT** | — | — | — |
| `test_report.py`、`test_volume_report.py` | S7a 不动 | **KEPT** | `test_volume_report.py:2239-2256` 还在测经典报告的「← 工作台」`/#job=` 链接；这个链接经过 302 仍然可用 | — | — |
| `test_report_freshness.py`、`test_rebuild_report_v014.py` | 不动 | **KEPT** | — | — | — |
| `test_onepager.py:162`、`test_trash.py:85` | 程序请求，不带 `text/html`，不会被跳转 | **KEPT** | — | — | — |
| `test_ops_v015.py`（rehearse） | 步骤名改为「打开工作区与一页纸」 | **DONE** | — | — | — |
| `test_devshot`（`Browser.login`） | `test_v2_p3_s7.py:669` | **DONE** | — | — | — |
| ops 测试和 33 项 Firefox | `test_operations_*.py` 保留 | **KEPT** | Firefox 那部分没有重跑，见 §1.2 | — | — |
| 黄金回归 §2.5 | `PHASE3_LANES.md` §5：6/6，数值逐位不变 | **DONE** | — | — | — |

### 1.7 原审计 §2.6 v2 里指向经典页的代码

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| `urls.report` | `ws_api.js:172`（只剩注释） | **DONE** | — | — | — |
| `urls.classic` | 已删（`9e4501c`） | **DONE** | — | — | — |
| 头像菜单「经典工作台」 | `ws_shell.js:256-257, 272` 换成「运维中心」 | **DONE** | `ws_onepage.js:147` 注释还写「before 『经典工作台』」 | 改注释 | 1 min |
| 「在经典报告中打开」 | `ws_shell.js:515-523`（重试 + 下载数据包） | **DONE** | — | — | — |
| 「在经典工作台中处理」 | 已删；扫描测试不再允许任何一处（`test_v2_p3_s7.py:635-655`） | **DONE** | — | — | — |
| 工具页「经典报告」一节 | 已删 | **DONE** | — | — | — |
| 登录页「回到经典工作台」 | 已删（第 2 路） | **DONE** | — | — | — |
| `ws_main.js:31` 致命错误页 | 只剩「重试」 | **DONE** | — | — | — |
| noscript | `index.html:66` 不再有链接 | **DONE** | — | — | — |
| 「查看器没有加载…」 | `ws_shell.js:398` | **DONE** | — | — | — |
| 口径说明里的「经典报告」 | `9e4501c` 已清。v2 里用户可见的只剩 `ws_display.js:826, 830`「旧版报告的设置」 | **KEPT**（第 4 路功能：读服务端存的旧预设） | — | — | — |
| 过期的 `example_report.html` | — | **STILL-OPEN** | 同 §1.1 附注 | — | — |

### 1.8 原审计 §3 加载速度

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| 首屏加载量（现测） | `static/v2/index.html` | 53 个文件，2,508,040 B 原始，约 769,710 B gz（原审计是 49 个、2.07 MiB、652 KiB，增量来自第三期新功能） | 第一次访问（以及每次升级后的第一次）更重 | — | — |
| #1 带版本号的地址 + immutable | `server.py:133-148`（`v2_page`），`151-157`（`v2_versioned`），`219-223`，`1355-1373`，`1609` | **DONE** | 升级瞬间，旧版本号的请求会拿到新内容但不缓存；v2 会提示刷新（`ui_build_v2`） | — | — |
| #2 缩略图持久缓存 | `ws_thumbs.js:132-142`（IndexedDB，第 2 路） | **DONE** | — | — | — |
| #3 按需加载 `volume_viewer.js` | `index.html:19`，`bundle.json` 的 `legacy_scripts` | **STILL-OPEN**（可选） | 首屏多 300 KB 原始、97.5 KB gz | 第四期再做 | 0.5–1 d |
| 服务端拼接脚本 | — | **STILL-OPEN**（可选） | 现在第二次访问已经不发请求，收益只在第一次访问 | 低优先 | 0.5–1 d |
| HTTP/1.1 keep-alive | `server.py:904` `Handler` 没设 `protocol_version` | **STILL-OPEN**（可选） | 每个请求一条 TCP 连接；第一次访问 54 次握手 | 不建议现在做 | — |

### 1.9 原审计 §4 待用户裁定的事项

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| 1 `/api/jobs/<id>/report` 怎么处理 | `server.py:1400-1405` | **DONE**（选 b：页面访问跳转；经典 `#view=` 由 `ws_legacy` 转换） | — | — | — |
| 2 S7b 数据壳 | — | **STILL-OPEN**（推迟） | — | — | — |
| 3 认领 / 换入口 | 认领：裁定放弃；换入口：已在 v2 实现 | **DONE** | — | — | — |
| 4 `/ops`、`/support` 保持独立页 | `PHASE3_LANES.md` §0.2 | **DONE** | — | — | — |

---

## 2. S7 弄坏或遗留的地方

搜索范围：`wss_deploy/` 下的 Python、static、md、env；`tests/`；`docs/02-推进与变更/05-部署工具/`；CLI 帮助、路由、doctor、report_freshness、service 文案。

| # | 项 | 位置 | 现状 | 用户能看到吗 | 建议 | 工作量 |
|---|---|---|---|---|---|---|
| F1 | 「认领」按钮已不存在 | `cli.py:328`：「…或在网页登录用户后点『认领』」 | 网页认领 UI 随 `app.js` 删除（裁定放弃） | **能**（CLI 输出，只在令牌登录模式下出现；正式服务是用户名登录，很少触发） | 改为「服务停止后用 `cli jobs claim --owner … --user …`」 | 5 min |
| F2 | 示例报告过期 | `static/v2/example_report.html`（09-30 12:20）；入口 `ws_rail.js:505, 652`「示例报告」 | 是第三期之前的代码打的包：有「经典工作台」×5、「在经典报告中打开」×2、`CLASSIC_TOOLS`，也没有第三期的新功能 | **能**（首页链接） | 上线后跑 `v2_examples`（已列入 `p3_s7.md` §9）。也可以考虑在服务机上生成好以后再合并 | 5 min |
| F3 | 上线步骤写着快进合并 | `docs/02-推进与变更/05-部署工具/前端重构_验收说明_2026-09-30.md:202` `git merge --ff-only wss-ui-v2` | 第三期合并后不能快进，这条命令会报错（不会造成破坏）；同文件 `:54, :97, :102, :178` 还在描述经典工作台 | **能**（运维照着执行） | 改成普通 `git merge`，并删掉经典工作台那几行 | 15 min |
| F4 | 上线验证表过时 | `env/GO_LIVE.md:34`（实际输出 `…/v2/`）、`:48`（「全部用户」→「看全部用户」）、`:49`（「三维报告」） | 说法过时 | **能**（运维） | 改成：打开 `/v2/`，打开任一结果页和一页纸 | 10 min |
| F5 | 其他文档过时 | `README.md:24, 26`；`OPERATIONS.md:3`；`OPERATIONS_ACCEPTANCE.md:3`；`WORKSPACE_V2_CONTRACT.md:37, 441`；`ANALYSIS_CONTRACT.md:553, 558, 571` | 还说「打开 `/`」「现有主页保持原样」「`#job=` 路由」等 | 否（只在文档里） | 一次性改掉 | 20 min |
| F6 | 对照测试静默失效 | `tests/test_v2_p3_info.py:197-218`（复核清单和 `alertTitle` 对照经典 `app.js`/`workbench_core.js`，文件不在就跳过）；`tests/test_v2_p3_workbench.py:22, 56, 93-96, 139-141`（`etaView`、`etaRowSummary`、`formatDuration`、`cohortFilter`） | 全量测试仍然通过，但经典对照不再执行。`alertTitle` 已经**没有任何断言**；复核清单只有部分行有固定值 | 否 | 照 `test_v2_lane_c.py` 的 `CLASSIC_CORE` 做法：用 `git show 866cf02:wss_deploy/static/workbench_core.js` 算出对照值，固化成常量 | 1 h |
| F7 | 没有消费方的字段和过时注释 | `server.py:875` `ui_build`；`server.py:110-116` docstring（classic workbench）；`server.py:56` 注释（side-by-side comparison）；`server.py:1341` docstring（与 `/` 相同的会话处理） | 无害 | 否 | 删除或改写 | 0.5 h |
| F8 | `EMBEDDABLE_HTML` 没有收紧 | `server.py:57` | 没有页面再嵌入 `report.html` | 否 | 改为不允许嵌入（不急） | 0.5 h |
| F9 | 「打开报告时会自动刷新」不再成立 | `doctor.py:326`；`cli.py:237, 448` | 浏览器访问在 `ensure_fresh`（`server.py:1695`）之前就被 302 了（`server.py:1599-1601`）。现在只有 upgrade、CLI 和程序读取才会刷新。对 v2 没有影响：数据逐字节相同 | **能**（运维看 doctor / CLI 输出，而且只在有过期报告时才出现） | 改成「升级时或 `cli reports refresh` 刷新」 | 10 min |
| F10 | 术语表里还写「三维报告」 | `glossary.py:62` / `static/glossary.json:92`（`gaussian_interpolation`） | 有意保留：改了会让所有报告判为过期。v2 的提示没用这个键 | **偶尔**（一页纸术语「all」模式，`onepager.py:931-940`） | 下次动报告模板时一起改 | — |
| F11 | 注释过时 | `static/v2/ws_onepage.js:147` | — | 否 | 改注释 | 1 min |
| F12 | 经典报告页的「← 工作台」链接 | `report.py:777`、`static/volume_viewer.js:1299` → `/#job=<id>` | 模板冻结。点了以后经 `/` 的 302 和 `ws_legacy` 落到 v2 结果页，**能用** | 只有经典页还会显示时才看得到（老浏览器直接打开 `/files/report.html`、数据包退回经典页的情况、iframe） | 保留；做 S7b 时再处理 | — |
| F13 | 批量打包的耗时 | `server.py:1294-1300`：每个任务现场生成离线页 | 没坏，但多任务（最多 50 个）时的 CPU 时间和打包槽占用没有实测 | 可能（批量下载变慢） | 上线后实测一次 | 15 min |

**查过、没有问题的**：

- `STATIC_FILES`（`server.py:52-53`）11 个文件都在磁盘上。
- `bundle.json` 的 `scripts`、`styles`、`pages`、`assets`、`dev`、`generated`、`legacy_scripts` 都在；`index.html` 的顺序与 `bundle.json` 一致。
- `report_freshness.UI_SOURCES`、`PY_TEMPLATES` 的文件都在，未改，所以 `stale_reports` 不会突然全部变成过期；`ui_fingerprint` 遇到缺失文件会写 `<missing>`，这次不涉及。
- `doctor.check_glossary`：文件在。
- 没有 Python 读取已删除的 6 个文件，也没有 `/static/<删除文件>` 的引用。
- v2 源码里没有 `/#job=`、`/compare?`、`urls.report`、`href:'/'`。
- `ops.js`、`support.js` 不加载任何已删除的脚本，`node --check` 通过。
- `jobs.py`、`narrative`、`onepager`、`export_table`、`timeline` 都不写经典链接。
- nginx 示例只有 `location /`，不受影响。
- v2 的离线模式顶栏不显示菜单，所以离线页里不会出现指向 `/support` 的死链接。

---

## 3. `/ops` 与 `/support` 在没有经典工作台时的状态

| 项 | 位置 | 现状 | 还缺什么 / 风险 | 建议 | 工作量 |
|---|---|---|---|---|---|
| 路由 | `server.py:54, 1599-1611`：`_classic_redirect` 对 `/ops` 返回 None，按白名单返回页面；本机模式会建会话 | 可用 | — | — | — |
| 样式 | `ops.html:8-9`、`support.html:3` 加载 `app.css` 和 `ops.css`，两者都在白名单里，自 `866cf02` 未改 | 可用 | 页面上用到的类里，有 7 个只在 `app.css` 里定义：`brand-mark, card, eyebrow, overview-tiles, primary, text-button, topbar`。所以 `app.css` 不能删 | 继续保留 | — |
| 脚本 | 两页只加载 `ops.js` / `support.js`（未改），不依赖任何已删除的文件 | 可用 | — | — | — |
| 返回链接 | `ops.html:14, 21` 和 `support.html:5` 指向 `/v2/`；`ops.html:20` 指向 `/support`；测试 `test_v2_p3_s7.py:62-70` | 可用 | v2 用新标签页打开这两页，点「返回工作台」会再开一个 v2，多一条 SSE 连接（HTTP/1.0，每个主机最多 6 条） | 可接受；以后可以让「返回」先尝试关掉当前标签页 | — |
| v2 里的入口 | `ws_shell.js:252, 256-257, 272`；测试 `test_v2_p3_s7.py:585-633`（共享模式下的普通用户看不到「运维中心」） | 可用 | 运维中心的任务行没有链到 `/v2/#/job/<id>` | 可选：任务编号做成链接（改的是 ops.js） | 0.5 h |
| 缓存 | 用 `STATIC_CACHE`，每次重新验证；不在 v2 的构建摘要里（`test_v2_p3_s7.py:136`） | 正确 | — | — | — |
| 验收 | 沙箱截图 `17_ops`、`18_support` 0 个 JS 错误（`p3_s7.md` §8） | 基本可信 | 33 项 Firefox 验收没有重跑 | 上线前跑 `tests/ops_browser_check.py`、`_edges.py`、`_feed.py` | 15 min |
| 外观 | 两页是浅色的经典外观，v2 是深色工作站风格 | 只是不统一 | — | 不处理 | — |

### 要不要现在迁进 v2：**不建议**

1. **裁定**：`PHASE3_LANES.md` §0.2 定的是保持独立页。S7 后两页零改动（只改了 href），也没有坏。
2. **成本和风险**：
   - `/ops` 整页迁过来约 3–4 人日，要写 1,200–1,500 行，还要把 33 项 Firefox 验收改写成 v2 的流程；
   - 有可能把 09-29 已经验收掉的并发和草稿保护问题重新带回来；
   - 用户只有管理员一个人。管理控制台和业务界面分开，是同类系统的常见做法。
3. **`app.css` 的代价很小**：只剩这两页在用。67 KB，只在这两页加载，也不在 v2 的构建摘要里，不影响 v2 的缓存。
   精简成 ops 专用副本属于改动 ops，对用户没有收益，不建议现在做。
4. **值得做的小改动**（都算改 ops/support，要用户同意）：
   - **a.** `support.js` 读取 `?job=<id>` 并预填任务编号；v2「工具」页加「就这个结果提问题」链接到 `/support?job=<id>`（约 1 h）；
   - **b.** 运维中心任务行的编号链到 `/v2/#/job/<id>`（约 0.5 h）；
   - **c.** P4 再定：如果医生真的在用工单，把 `/support` 迁成 v2 对话框（约 1 d）。`/ops` 继续作为独立的管理员控制台。

---

## 4. 建议的收尾顺序（都很小，可以一次提交）

1. `cli.py:328` 改「认领」提示（F1）。
2. 改文档：`前端重构_验收说明_2026-09-30.md:202` 的 ff-only（F3）、`env/GO_LIVE.md:34/48/49`（F4）、`README.md:24/26`、`OPERATIONS*.md:3`、`WORKSPACE_V2_CONTRACT.md:37/441`、`ANALYSIS_CONTRACT.md`（F5）。
3. 把 F6 的经典对照值固化成常量（约 1 h）。
4. 改说法和清理：`doctor.py:326` 与 `cli.py:237/448` 的「打开报告时自动刷新」（F9）；`ui_build`、过时注释和 docstring（F7）。
5. 上线时：
   - 普通 `git merge`，然后 `service upgrade`；
   - 跑 `v2_examples`（F2）；
   - 在沙箱跑 33 项 ops Firefox 验收；
   - 实测一次 10 个以上任务的批量打包耗时（F13）。
6. 以后再做：S7b 数据壳；`volume_viewer.js` 按需加载；`EMBEDDABLE_HTML` 收紧；support 预填任务 / ops 链接 v2（要用户同意）。
