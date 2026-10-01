# 第三期第 2 路（工作台）— 交付报告

分支 `wss-ui-p3-workbench`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_b`，基线 `866cf02`。范围见 `PHASE3_LANES.md` §3「第 2 路」，六项全部做了。审计行号见 `lanes/p3_audit_workbench.md`。

## 1. 做了什么，入口在哪

| 功能（审计行） | 入口 | 代码 |
|---|---|---|
| 队列复核筛选加「已重新打开」（#60） | `#/cohort` 筛选行「全部复核状态」 | `ws_admin.js` `renderCohort` |
| 按数值上下限筛选（#61）：「指标」右边两个输入框（下限、上限），作用于当前指标；换指标后，原来的范围留在筛选行上成为一个可关掉的标签；占比按 % 输入；补上「极高 WSS 占比」一列 | `#/cohort` 筛选行 | `ws_admin.js` `inRanges` / `readRange` / `rangeText` |
| 病例卡「补跑缺少的结果」（#34）：这次扫描还没有完成结果的每个发布包，在结果标签后面加一个虚线「＋ 名称」小标签；点它确认后用这次扫描最新一个已完成、出口已确认的任务补跑 | 首页病例卡 | `ws_rail.js` `missingResults`，`ws_admin.js` `fillMissing` |
| 病例卡显示管腔最大直径和标签（#30、#37） | 首页病例卡名字右侧、名字下方 | `ws_rail.js` `caseFacts` |
| 任务列表「直径 mm」一列（#37），可排序 | `#/tasks` | `ws_admin.js` `renderTaskTable` |
| 任务行「⋯」菜单（#36）：打开、一页纸、编辑信息…、换模型重跑…、患者时间线、删除… | `#/tasks` 每行最右 | `ws_admin.js` `rowMenu` / `editInfo` / `openTimeline` |
| 实时剩余时间（#38，合同里写作 #30）：计算中 / 排队 / 待确认的任务显示「还需约 N 秒」，计算中加细进度条，每秒走字 | 任务列表状态列、病例栏结果行（行底一条细线）、首页「进行中」卡、「需要处理」卡 | `ws_rail.js` `etaView` / `etaRowSummary` / `etaBadge` |
| 多选条「批量出图…」：有第 5 路的 `ns.batch.open` 才出现，传入所选里已完成的任务号和 `shellApi` | `#/tasks` 勾选后底部操作条 | `ws_admin.js` `renderSelbar` / `openBatch` |
| 首页「最近完成」（#47）：最近 5 个完成的结果，带待复核 / 已复核；和「需要处理」并排（窄屏上下排） | 首页 | `ws_rail.js` `recentDone` |
| 首页「三步上手」（#17）：病例库为空时一直展开（带「上传 STL」）；新用户在关掉之前展开；关掉后从头像菜单「三步上手」再打开 | 首页、头像菜单 | `ws_rail.js` `guideCard`，`ws_admin.js` `openGuide` |
| 登录（#3–#5）：口令框右侧显示 / 隐藏，大写锁定提示，「记住用户名」开关，429 时按服务给的 `retry_after` 倒计时禁用按钮（#4） | 登录页 | `ws_shell.js` `showLogin` |
| 503「维护中」：登录时服务答 502 / 503 / 504，提示「服务正在启动或维护，请稍后再试」；登录后 `/api/health` 答这三种，顶部提示「服务正在启动或维护，自动重试…」，每 5 秒探一次，恢复后提示「服务已恢复」并刷新列表（这时不再同时显示「连接中断」） | 登录页、顶部横幅 | `ws_shell.js` `showLogin`，`ws_admin.js` `healthFetch` / `maintenance` |
| 登录页脚去掉「经典工作台」 | 登录页 | `ws_shell.js` `showLogin` |
| 首页缩略图持久缓存：画好的缩略图存进 IndexedDB（库 `wssv2-thumbs`），键为任务号 + 运行身份（未完成任务用记录版本）+ 绘制版本号，最多留 400 张，最旧的先删；打不开、写不进时照常从内存工作 | 首页病例卡 | `ws_thumbs.js` `keyFor` / `stored` / `store` |

少字的处理：范围筛选的口径写进队列页标题旁的 ⓘ；「补跑」「直径」「记住用户名」的说明放在悬停提示里；菜单项不可用时右侧只写一个短原因（已复核锁定 / 别人的任务 / 计算中 / 只有一次扫描）。

## 2. 与经典工作台的一致性

- **接口和载荷同经典**：补跑 `GET /api/jobs/<id>` 取版本后 `POST /api/jobs/<id>/rerun {version, release_id}`（经典 `rerunFromCase`）；编辑信息 `POST /api/jobs/<id>/metadata`，直接调用结果页自己的对话框 `ns.detail.metadataDialog`（同样的字段、检查和载荷），对话框开着时它看到的「当前任务」是这一行，关掉后恢复；一页纸 `/api/jobs/<id>/onepage`；删除走已有的 `deleteJobs`；登录 `POST /api/session`，记住用户名用经典的同两个键 `wss-login-remember`、`wss-login-user`，「三步上手」关闭状态用经典的 `wss-guide-closed`（两边关一次都算关了）。
- **规则移植并逐条对照 `workbench_core.js`**（经典文件随 S7 删除，所以把函数移到 `ws_rail.js`，测试里同时写死对照值，经典文件还在时再直接比较）：`formatDuration`、`stageRemaining`、`etaView`、`etaRowSummary`（8 种 eta、两个时间点全部相同）；队列下限筛选与经典 `cohortFilter` 的 p99 / 高占比 / 极高占比 / 速度 / 直径下限结果相同，复核状态 `reopened` 相同。
- **沙箱实测，与经典逐项比对**（`/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane2/classic_compare.json`，同一页面里加载经典 `workbench_core.js`，对照经典接口 `GET /api/cases`）：

  | 病例（扫描） | 经典 `missing_releases` | 新工作区「＋」 | 经典 `reusable_job_id` | 新工作区补跑来源 |
  |---|---|---|---|---|
  | EXAMPLE_CONFIRM | X5D | X5D（峰值 WSS） | …234331_7f97c128ed74 | 同 |
  | P-DEMO-01 第一次扫描 | PF6、X5D | PF6、X5D | …045134_6b8640d971b0 | 同 |
  | P-DEMO-01 第二次扫描 | 无 | 无 | …125023_7560290cb28c | 同 |
  | LV_GUO_YOU | 无 | 无 | …230450_d1428792b846 | 同 |
  | DEMO_BAD_INPUT（只有失败任务） | 三个 | 不显示（没有可复用来源） | 无 | 无 |

  「最近完成」5 条与经典 `overviewModel(jobs).recent` 顺序完全相同。管腔最大直径取任务列表里的 `max_diameter_mm`（服务端 `morphology.max_diameter_mm`），与队列表「管腔最大直径」列同源：P-DEMO-01 25.8 / 29.1 mm，LV_GUO_YOU、EXAMPLE_CONFIRM 72.2 mm，两处一致。

## 3. 与经典的差异和原因

1. **补跑不重复提交**：同一扫描某个发布包已有排队、计算中或待确认的任务时，不再显示它的「＋」（经典只看有没有完成结果，会让人重复提交）。补跑来源要求「已完成」且出口已确认（服务端 `rerun` 只接受已完成的任务；经典的 `reusable_job_id` 可能指向未完成任务，点了会 409）。补跑前多一个确认框（经典直接提交）；建好后留在首页，提示条里有「查看」。
2. **管理员「看全部用户」时**，补跑和行菜单里的修改类操作只对自己的任务可用（服务端规则，同第二期）。
3. **病例卡不显示「待审阅 N」**：卡片上每个结果标签已经用状态点区分待复核 / 已复核，不再加计数（少字）。
4. **病例卡直径**：经典的直径小标签读 `/api/cases` 的 `runs[].max_diameter_mm`，但这个接口现在不返回这个字段（`cases.py _run`），所以经典页上实际不显示；新工作区读任务列表里已有的值。
5. **范围筛选**比经典多了上限，并对所有指标列开放（经典只有 5 个下限）；分布图和散点保留当前指标范围外的数据作背景（变淡），表格、卡片和导出只含范围内的结果。
6. **患者时间线**：不再弹独立对话框，而是打开该结果、切到「概览」并滚到「随访」一节（随访的内容由第 3 路扩充，避免两份实现）；只有一次扫描时菜单项不可用。
7. **503**：除 503 外，网关的 502 / 504 也按「启动或维护」处理。

## 4. 改动过的公共文件

- `wss_deploy/static/v2/ws_shell.js`：只改了 `showLogin` 一个函数（上方加了一段注释）。表单仍用 `wsl-user` / `wsl-pass`（第 6 路的 `devshot` 依赖），新增 `wsl-caps`、`wsl-remember`；去掉页脚「也可以回到经典工作台」。没有改 `SHELL_API`。
- `tests/test_v2_lane_c.py`：改了一行。原来用「每行最后一个单元格」取属主，现在任务行最后一列是「⋯」菜单，改成按 `wsc-owner` 类名取属主单元格；断言不变。
- 没有改 `bundle.json`、`index.html`、Python 服务端。第 5 路的 `ws_batch.js` 只按「有就调用」使用 `ns.batch.open(jobIds, shellApi)`。

## 5. 测试结果

- 新测试 `tests/test_v2_p3_workbench.py`，11 项：剩余时间规则（写死值 + 经典对照）、补跑 / 直径 / 标签 / 最近完成 / 范围筛选的纯函数（含经典 `cohortFilter` 对照）、登录页（显示隐藏、大写锁定、记住用户名、429 倒计时期间不再提交、503）、首页（三步上手开关与头像菜单重开、最近完成、病例卡、补跑载荷、病例栏每秒走字）、空库、任务行（菜单可用性、一页纸、编辑信息载荷与版本、删除确认、批量出图）、患者时间线滚动、队列（已重新打开、范围、标签式残留范围）、维护横幅与恢复、IndexedDB 缩略图缓存（存、重载后不再取几何、没有 / 打不开 IndexedDB 时照常）。
- 改过的 js 都过 `node --check`。
- 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`：**943 通过 / 33 跳过**（基线 866cf02 在同一台机器上是 932 / 33，多出的 11 项是本路新测试；跳过项是本机没有的开发任务副本等环境条件）。
- 没有改分析、流水线、叙述或报告生成的 Python，不需要黄金回归。

## 6. 截图（沙箱 127.0.0.1:8822，预览任务副本，无头 Firefox，全程 0 个 JS 错误）

目录 `/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane2/shots/`，逐步检查值在同目录 `walk.json`：

- `01_login_reveal_caps.png` 登录页：显示口令、大写锁定提示、记住用户名，无经典链接
- `02_home_guide_recent_cards.png` 首页：三步上手、需要处理 + 最近完成并排
- `03_home_after_reload_thumbs_from_idb.png` 重载后缩略图来自 IndexedDB（3 行，manifest / arrays / geometry 请求 0）
- `04_fill_missing_confirm.png`、`05_fill_missing_created_live_eta.png` 补跑确认与建好后的剩余时间（进行中卡、需要处理卡）
- `22_rail_live_eta.png` 病例栏结果行的剩余时间与细进度条（2 秒后 70% → 72%）；`06_rail_live_eta.png` 拍到时那个任务已算完，以 22 为准
- `08_tasks_row_menu.png` 任务行「⋯」菜单、直径列、状态列剩余时间
- `09_tasks_edit_info_dialog.png`、`10_tasks_edit_info_saved.png` 行内编辑信息并保存（加标签）
- `11_tasks_selection_batch.png` 批量出图按钮（页面里临时放了一个 `ns.batch.open` 替身）
- `12_timeline_follow_section.png` 患者时间线 → 结果概览「随访」
- `13_cohort_range_p99.png`、`14_cohort_two_ranges.png` 队列范围筛选、残留范围标签
- `15_home_cards_tags_and_fill_chips.png` 病例卡直径、标签、补跑小标签
- `16_maintenance_banner.png`、`17_maintenance_recovered.png` 503 横幅与恢复（页面内模拟 `/api/health` 503）
- `18_avatar_menu_guide.png`、`19_guide_reopened.png` 头像菜单「三步上手」
- `20_home_narrow.png` 390 宽
- `21_login_cooldown_429.png` 连续输错后服务答 429，按钮倒计时

## 7. 沙箱与进程

沙箱 `devshot sandbox --port 8822`（PID 4190821）和 Marionette 3022 的 Firefox 都是本路自己启动的，用完按进程号停掉。沙箱里点「补跑」新建过几个任务，只在沙箱副本里，不影响 `outputs/wss_deploy_preview_jobs`。

## 8. 没做完 / 需要决定的

- **会话过期的原地重新登录框**（`ws_main.js openRelogin`，第 6 路的文件）还没有显示 / 隐藏、大写锁定、429 倒计时；可以照 `showLogin` 的写法补，建议第 6 路或合并时处理。
- **启动时的 503**：`ws_main.js` 的 `boot` 在 `/api/session` 答 503 时显示「暂时连不上服务：服务返回错误（HTTP 503）」，不是「维护中」的说法，同样属于第 6 路。
- 剩余时间规则移植到了 `ws_rail.js`；第 1 路如果也为「各阶段耗时」表移植了 `etaView`，合并后可以合成一份。
- 「三步上手」对已经在用、但从没关过经典引导的账号也会展开一次（经典规则就是「关过一次才不展开」）。
