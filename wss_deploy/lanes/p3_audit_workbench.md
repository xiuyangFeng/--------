# 经典工作台 + 并排比较页 → v2 迁移审计（S7 退役前）

- 范围：`wss_deploy/static/index.html`、`app.js`（3400 行，逐函数读完）、`workbench_core.js`、`batch_export.js`、`compare.html`、`compare.js`；对照 `static/v2/*.js`（worktree `GNN_wssui_v2`，分支 `wss-ui-v2`，HEAD `6e45be1`）。
- 方法：按 index.html 元素 id 与 app.js 函数逐项列出，再在 v2 里按接口、文案、行为 grep 并读实现。只读代码，没有点界面。
- 状态：HAVE = 有等价功能；PARTIAL = 有，但缺的写在「证据或缺什么」里；MISSING = 没有；DROP-CANDIDATE = 已过时或被替代，建议放弃。
- 统计（114 行）：**HAVE 65 · PARTIAL 26 · MISSING 13 · DROP-CANDIDATE 10**。

## 1. 逐项对照表

### A. 登录、会话、账户

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 1 | 登录（口令模式 / 令牌模式） | app.js `renderLoginPanel`、`#login-form` submit | HAVE | ws_shell.js `showLogin`（按 `session.login` 切换） | — | — |
| 2 | 迁移期旧令牌登录（口令模式下用户名留空、只填旧令牌，`legacy_token_allowed`） | app.js `renderLoginPanel`（`legacy`，`#login-token-group` 与用户名组同时显示） | MISSING | `showLogin` 与 ws_main.js `openRelogin` 在口令模式只发 `{username,password}`，不看 `legacy_token_allowed` | 服务端迁移窗口已关就放弃；还开着就迁移 | S |
| 3 | 显示 / 隐藏口令按钮、大写锁定提示 | app.js `#reveal-password`/`#reveal-token`、`capsCheck`/`#caps-hint` | MISSING | v2 登录框与重登框都没有 | 迁移（低优先） | S |
| 4 | 429 冷却倒计时（读 `retry_after`，按钮禁用 N 秒） | app.js `loginCooldown` | PARTIAL | v2 只显示「请一分钟后再试」，按钮不禁用，不读 `retry_after` | 迁移 | S |
| 5 | 「在这台电脑上记住用户名」开关 | app.js `#remember-username`、`wss-login-remember` | PARTIAL | v2 登录成功后总是写 `wss-login-user`，没有开关（共用电脑无法不记） | 迁移 | S |
| 6 | 认领旧会话任务横幅（登录后提示 N 个旧会话任务，「认领」/「稍后」） | app.js `applySession`（claimable）、`claimJobs`→`POST /api/jobs/claim`、`dismissClaim`；`#claim-banner` | MISSING | v2 无任何 `/api/jobs/claim` 调用；登录返回的 `claimable` 被忽略。旧令牌会话留下的任务在 v2 里无法归到用户名下 | 迁移（先查服务端还有没有未认领的 owner；有就必须迁移） | S |
| 7 | 退出登录 | app.js `#logout` | HAVE | ws_shell.js `logout`（头像菜单） | — | — |
| 8 | 修改口令 | app.js `#change-password` | HAVE | ws_admin.js `passwordDialog` | — | — |
| 9 | 会话过期原地重新登录（不丢页面；换人整页刷新） | app.js `expireSession`/`resumeSession` | HAVE | ws_main.js `onUnauthorized`/`openRelogin` | — | — |
| 10 | 管理员身份标识 | app.js `applySession`（`管理员` 标签） | HAVE | ws_shell.js `renderTop` 头像菜单标题「· 管理员」 | — | — |
| 11 | 管理员「全部用户」开关（列表、队列、回收站、时间线跟随） | app.js `#view-all` | HAVE | ws_admin.js `toggleViewAll`、任务列表「属主」列。差异：v2 `ws_shell.loadTimeline` 对管理员**总是**带 `all=1`，不看开关 | 保留；时间线口径建议改成跟随开关 | S |
| 12 | 连接状态常驻文字 + 悬停服务状态浮层（版本、运行时长、队列、计算线程、GPU、磁盘、默认发布包、待刷新报告） | app.js `setConnection`、`showHealthPop`/`healthRows`、`#connection`/`#health-pop` | HAVE | ws_admin.js `healthDialog`（头像菜单「服务状态…」，同样的行）；断线时顶部横幅 | 保留（形式不同） | — |
| 13 | 503「服务正在启动或维护 · 自动重试」 | app.js `refreshHealth` | MISSING | ws_admin.js `healthFetch` 非 2xx 一律当 null，不提示 | 可放弃；想保留就在横幅加一句 | S |
| 14 | 服务 / 页面文件更新后提示刷新 | app.js `checkServiceVersion` | HAVE | ws_admin.js `checkVersion`（`ui_build_v2`） | — | — |
| 15 | 实时流中断提示与自动重连 | app.js `openOwnerEvents` onerror、`markOnline` | HAVE | ws_admin.js `onConnection`/`startDownPoll` | — | — |

### B. 偏好、设置、帮助

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 16 | 偏好存服务端、失败时存本机，分节合并 | app.js `loadPreferences`/`savePreferences` | HAVE | ws_admin.js `loadPrefs`/`savePrefs`/`mergePrefs` | — | — |
| 17 | 「三步上手」：新用户在概览自动展开，设置菜单可再开 | app.js `renderOverview`（`overview-guide`）、`guideDialog`、设置菜单 | PARTIAL | 内容在帮助菜单「一页操作卡」`help_quickstart.html` 与空首页的链接里；不会对新用户自动展开 | 迁移「首次登录自动打开一次」或放弃 | S |
| 18 | 快捷键说明（?） | app.js 设置菜单 → `state.shortcuts.showHelp` | HAVE | ws_shell.js `shortcutsDialog` | — | — |
| 19 | 报告模板（机构、科室、标题、签字栏、术语表、附录、页脚；非管理员只读） | app.js `templateDialog` | HAVE | ws_onepage.js（userMenu「报告模板…」） | — | — |
| 20 | 设置菜单 → 队列总览 / 回收站 | app.js `#settings-button` | HAVE | 首页页签 `#/cohort`、`#/trash`；病例栏底部链接（ws_rail.js `homeHead`/`PAGES`） | — | — |
| 21 | 术语「?」弹出说明（`/static/glossary.json`，中英释义） | app.js `loadGlossary` + `[data-gloss]` 点击 | DROP-CANDIDATE | v2 不读 glossary.json，改用各区块 `infoTip` 与证据透镜里的 `definition` | 放弃（注意 glossary.json 本身仍被报告与 doctor 使用，不能删） | — |
| 22 | 浏览器桌面通知开关（首次点击自动请求权限） | app.js `#notify-toggle`、`askNotificationPermission`、`showNotification` | HAVE | ws_admin.js `toggleNotifications`/`desktopNotice`（只在打开开关时请求权限；只在页面隐藏时弹） | — | — |

### C. 新建（上传）

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 23 | 上传对话框：单 / 多文件、单位、发布包、同时预测组合、元数据、计算设置（设备、模型数、线程）、记住患者 | index.html `#upload-dialog`；app.js `refreshReleases`/`updateReleaseSettings`/`releaseCombos`/`applyUploadPreferences` | HAVE | ws_upload.js（计算设置只在完整档） | — | — |
| 24 | 整页拖放 STL | app.js document `drop` | HAVE | ws_upload.js `installDrop` | — | — |
| 25 | 逐文件预检（扩展名、空文件、大小上限、重名、「文件名可能是姓名」）+ 字节级进度 + 分批 | app.js `uploadCheck`/`renderUploadFiles`/`sendUpload`、`WB.uploadChunks` | HAVE | ws_upload.js（`api.upload` onProgress、`/api/jobs/batch` 分批） | — | — |
| 26 | 病例 / 患者编号即时校验（不可见字符、超长为错误并阻止上传，像姓名为提醒） | app.js `identifierIssues`（`WB.identifierIssue`） | PARTIAL | ws_upload.js 只有「可能是姓名」提醒和 `maxLength`；不可见字符靠服务端拒绝（编辑信息里 ws_detail.js `identifierIssue` 有） | 迁移（复用 ws_detail 的函数） | S |
| 27 | 同几何查重：打开已有 / 只跑新发布包（复用出口）/ 强制重算 / 跳过 | app.js `duplicateDialog` | HAVE | ws_upload.js `duplicate`（单个与批量逐个） | — | — |
| 28 | 上传完成后打开最后建的任务 | app.js upload submit → `selectJob` | HAVE | ws_upload.js → `onCreated`→`go` | — | — |

### D. 列表、病例卡、首页（今日概览）

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 29 | 按任务列表，按工作分组（需要处理 / 计算中与排队 / 待审阅 / 已审阅 / 已取消，折叠状态记忆） | app.js `renderJobsList`、`WB.workGroups` | HAVE | 病例栏按病例→扫描→结果（ws_rail.js `create`）+ 首页「需要处理」（`todo`/`attnPick`）+ `#/tasks`（ws_admin.js `mountTasks`） | 保留（组织方式不同） | — |
| 30 | 病例卡：管腔最大直径 chip、「待审阅 N」、#标签 | app.js `caseCard`（`/api/cases`） | PARTIAL | 首页画廊卡只有缩略图、名称、患者·扫描、结果 chip；无直径、无待审阅计数、无标签 | 迁移（直径与标签数据已在 job 列表里） | S |
| 31 | 病例卡：编辑（最近任务的病例信息） | app.js `caseCard` → `editMetadata` | HAVE | 打开结果 →「工具」→「病例信息 → 编辑…」（ws_detail.js） | 保留（多一步） | — |
| 32 | 病例卡：患者时间线 | app.js `caseCard` → `timelineDialog` | HAVE | 概览「随访」（ws_overview.js `followupModel`） | — | — |
| 33 | 病例卡：并排打开 WSS + 体场 | app.js `caseCard`（`model.compare` → `/compare?left&right`） | HAVE | 概览头部同一输入结果下拉（ws_shell.js `siblings`）+ 比较选择框「同一输入的其他结果」 | — | — |
| 34 | 病例卡：「补跑 {族}」—— 自动找出本病例缺哪个发布包，一键用可复用任务补跑 | app.js `caseCard`（`missing_releases`/`reusable_job_id`）、`rerunFromCase` | MISSING | v2 有手动「换模型重跑」（完整档工具页、任务列表重跑），但不告诉用户缺哪个结果；不调用 `/api/cases` | 迁移（按 `input_sha256` 分组在前端算缺失即可） | M |
| 35 | 病例卡：导出该病例汇总 CSV | app.js `caseCard` → `exportSummary(doneJobIds)` | PARTIAL | 导出对话框「数据」页是单个结果的统计表 CSV；多结果要去任务列表勾选 | 迁移（病例卡或概览加按钮）或接受 | S |
| 36 | 任务行「⋯」菜单：三维报告、一页纸、编辑信息、换包重跑、患者时间线、删除 | app.js `jobMenuItems`/`moreButton` | PARTIAL | `#/tasks` 行无菜单，靠多选条与结果页（lane C 报告 §7 已记） | 迁移或接受 | M |
| 37 | 任务行信息：结果类型、患者·扫描、#标签、相对时间、管腔直径 mini chip | app.js `jobRow` | PARTIAL | `#/tasks` 有结果、患者/扫描、标签、创建时间、属主；无直径 | 接受或补一列 | S |
| 38 | 计算中任务行的实时剩余时间细条 | app.js `etaRow`、`tickEta` | MISSING | 病例栏与任务列表都没有进度；只有首页「进行中」卡写阶段名 | 迁移（病例栏结果行加细条）或放弃 | S |
| 39 | 搜索（病例、扫描、备注） | index.html `#jobs-query`（服务端） | HAVE | 病例栏前端子串（不含备注）；`#/tasks` 服务端全文 | — | — |
| 40 | 状态筛选（9 个状态 + 待审阅） | index.html `#jobs-status` | HAVE | ws_rail.js `FILTERS`、ws_admin.js `STATUS_FILTERS`（合并为桶：中断算失败、排队算计算中、两类待确认合一） | — | — |
| 41 | 患者编号 / 标签精确筛选 | index.html `#jobs-patient`/`#jobs-tag` | HAVE | `#/tasks` 精确筛选 + 候选 | — | — |
| 42 | 分页、每页条数 | app.js `updatePagination`、`#jobs-page-size` | HAVE | `#/tasks` 25/50/100 | — | — |
| 43 | 「刷新」按钮 | index.html `#refresh-jobs` | DROP-CANDIDATE | v2 靠事件流 + 断线横幅，恢复时自动刷新 | 放弃 | — |
| 44 | 未读点、未读数、标题栏计数 | app.js `persistUnread`/`updateUnreadBadge` | HAVE | ws_admin.js `unreadAdd`/`updateTitle`/`repaintMarks` | — | — |
| 45 | 今日概览：问候语 + 4 个瓦片，点瓦片筛左侧列表 | app.js `renderOverview`/`setQuickFilter` | HAVE | 首页四卡（ws_rail.js `overview`）点开 `#/tasks` 带筛选；口径不同（今日上传/进行中/待处理/失败） | — | — |
| 46 | 「近 7 天完成」口径 | app.js 瓦片 `week`、`WB.QUICK_FILTERS` | DROP-CANDIDATE | v2 换成「今日上传 + 近 14 天柱」 | 放弃 | — |
| 47 | 今日概览：「最近完成」列表（带 待审阅 / 已审阅 徽章） | app.js `renderOverview`（`model.recent`） | PARTIAL | 画廊按最新排序，但不是「最近完成」，无审阅徽章 | 迁移（矩阵已记）或接受 | S |
| 48 | 今日概览：服务状态块、常用快捷键块 | app.js `renderOverviewStatus`、`keyHints` | DROP-CANDIDATE | 分别在「服务状态…」对话框和帮助菜单 | 放弃 | — |
| 49 | 空列表 / 无结果提示 | app.js `renderJobsList`（`#jobs-empty`） | HAVE | ws_rail.js `todo` 空状态（上传、输入要求、示例报告）、任务列表空状态 | — | — |
| 50 | 键盘：J/K、↑/↓（列表焦点）、`/`、N、?、Enter/O = 打开三维报告、G = 一页纸 | app.js `installWorkbenchShortcuts` | PARTIAL | v2 有 J/K、`/`、N、?、O（=一页纸）；无列表 ↑/↓；**G 在 v2 是游标、O 在经典是三维报告**，语义换了 | 保留；在快捷键表或更新说明里写明变化 | S |

### E. 多选与批量

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 51 | 进入选择、全选已结束 | index.html `#select-jobs`/`#select-finished` | HAVE | `#/tasks` 勾选列 + 本页全选 | — | — |
| 52 | 批量删除（≤100，计算中 / 已锁定排除，撤销） | app.js `deleteJobs`/`undoDelete` | HAVE | ws_admin.js `deleteJobs`/`undoDelete` | — | — |
| 53 | 批量汇总 CSV / xlsx | app.js `exportSummary` | HAVE | ws_admin.js `exportSummary` | — | — |
| 54 | 批量打包 zip | app.js `bundleJobs` | HAVE | ws_admin.js `bundleJobs` | — | — |
| 55 | 批量换发布包重跑（逐个、跳过原因） | app.js `batchRerunDialog` | HAVE | ws_admin.js `rerunDialog`（只限自己的任务） | — | — |
| 56 | 跨病例批量出图：内置预设（瘤囊低 WSS 区、髂分叉热点、主动脉沿程、沿程压降、流线全貌、瘤囊截面系列）或报告导出的视图状态 JSON；分辨率、背景、色标（叠加 / 单独 SVG / 无）、中英文 → 一个 zip | app.js `batchExportDialog` + batch_export.js `run`/`zipStore`（隐藏 iframe 加载 `/api/jobs/<id>/report`，走 `wss-view:*` 消息） | MISSING | v2 出图（ws_figure.js）只针对当前病例；没有预设的跨病例批量 | 迁移（在 v2 查看器上重写，不能再借经典报告 iframe）或明确放弃 | L |

### F. 回收站

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 57 | 回收站：列表、恢复（并打开）、彻底删除（二次确认）、剩余天数、计数 | app.js `refreshTrash`/`trashItem` | HAVE | ws_admin.js `mountTrash`/`restoreTrash`/`purgeTrash` | — | — |
| 58 | 管理员「全部用户」时看别人的回收站（服务端只读） | app.js `refreshTrash`（`?all=1`） | DROP-CANDIDATE | v2 只列自己的（ⓘ 已写明）；服务端 server.py L1737 规定只能动自己的 | 放弃 | — |

### G. 队列总览

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 59 | 关键数字表：排序、点行打开任务 | app.js `renderCohort`/`COHORT_COLUMNS` | HAVE | ws_admin.js `renderCohort`（列更多，空列自动隐藏） | — | — |
| 60 | 筛选：族 / 发布包 / 审阅（含「已重新打开」） | index.html `#cohort-family`/`#cohort-release`/`#cohort-review` | PARTIAL | 审阅筛选只有「未复核 / 已复核」，`reopened` 的行两个选项都筛不到 | 迁移 | S |
| 61 | 数值下限筛选：p99 ≥、高 WSS 占比 ≥、极高占比 ≥、速度 p99 ≥、管腔直径 ≥ | index.html `.cohort-numbers`、`WB.cohortFilter` | PARTIAL | 只能点直方图的一个柱子（一个区间），不能设「≥ 阈值」；「极高占比」列不存在 | 迁移（阈值输入或可多选柱） | M |
| 62 | 直方图（p99 叠人群参照、高占比、速度 p99、直径） | app.js `drawHistogram` | HAVE | 一个分布图（量可选）+ 与直径的散点（ws_admin.js `renderCharts`） | — | — |
| 63 | 导出筛选结果 CSV / xlsx | app.js `#cohort-export-csv`/`-xlsx` | HAVE | ws_admin.js `renderCohort` CSV / Excel | — | — |
| 64 | 宽屏展开 | app.js `#cohort-expand` | DROP-CANDIDATE | v2 是整页 | 放弃 | — |

### H. 任务详情：未完成任务（排队、计算、待确认、失败）

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 65 | 进度与剩余时间：分段条、「各阶段耗时」表（预计 / 实际）、超时标红、估计依据、每秒走字 | app.js `etaWidget`/`tickEta`/`WB.etaView` | PARTIAL | ws_input.js `etaBlock`：剩余时间 + 当前阶段 + 分段条，3 秒轮询刷新；无阶段表、超时、依据 | 迁移阶段表与超时（可复用 `WB.etaView`） | S |
| 66 | 五步步进条（上传→确认输入→确认出口→预测→结果） | app.js `renderJob`（`.stepper`） | DROP-CANDIDATE | v2 用状态点 + 输入页 | 放弃 | — |
| 67 | 取消任务（排队 / 计算中，确认对话框说明可重试） | app.js `cancelJob` | HAVE | ws_input.js `renderPanel`「取消计算」 | — | — |
| 68 | 取消任务（**待确认输入 / 待确认出口**） | app.js `renderJob`（确认页头部与 statusLine 的「取消任务」） | MISSING | ws_input.js 只在 `queued/running/cancelling` 显示取消 | 迁移 | S |
| 69 | 重试 / 恢复：**失败**任务（`failed`） | app.js `renderJob` recovery 卡「重试任务」（`failed/cancelled/interrupted`，锁定或 `retryable===false` 除外）→ `/retry` | MISSING | ws_input.js `failureCard` 只对 `interrupted`/`cancelled` 给「重试」；服务端 jobs.py `retry` 允许 `failed`。显存不足（`retry_hint:'cpu'`）等可重试错误在 v2 里无路可走 | **必须迁移** | S |
| 70 | 错误卡：按类别的标题与建议（输入几何 / 工具 / 资源 / 内部）、「改用 CPU 重算」提示、诊断编号、**管理员可见的技术细节 `admin_detail`** | app.js `ERROR_TITLES`/`ERROR_HINTS`、`renderJob` error-card | MISSING | v2 只显示 `error.message` 一句 + 报错对照表链接；`diagnostic_id`、`admin_detail`、`retryable` 全不读 | 迁移 | S |
| 71 | 输入确认：单位选择、换算后尺寸预览、删除孤立片、勾选确认 → `/input` | app.js `inputCard` | HAVE | ws_input.js `unitsCard` | — | — |
| 72 | 输入确认的细节：原始 / 换算尺寸、开口数、壁面面积与顶点数、连通片数、拟删除面积比例、`errors`/`flags` 提示条、参考范围与人群位置卡、`fail` 阻断说明 | app.js `inputFacts`/`inputCard` | PARTIAL | `unitsCard` 有预览、碎片勾选（不写比例）、质量检查列表；缺事实表、flags、参考范围卡 | 迁移（ws_detail.js `inputModel` 已算好这些） | S |
| 73 | 出口确认：三维、端点表、名称下拉、左右 / 左侧 / 右侧互换、恢复建议、规则提示、勾选确认 | app.js `outletCard` | HAVE | ws_input.js `outletCard` | — | — |
| 74 | 出口确认：「入口选错了？重新选择入口 → 重提中心线」（`/confirm {inlet}`） | app.js `outletCard`（`.inlet-details`） | MISSING | v2 不发 `inlet`；服务端 jobs.py L1983 支持。矩阵写「已有，缺重提中心线」 | 迁移 | M |
| 75 | 出口确认的细节：世界坐标列、三维里点端点选中行、复位视角、保存截图、「已修改 N 个出口」、确认按钮旁「确认后约 N 秒出结果」、人话化的核对原因（`WB.outletReview`） | app.js `outletCard`/`createViewer`/`etaNote` | PARTIAL | ws_input.js 有表、把握度、原因原文；缺上列各项（三维标记不可点） | 迁移原因人话化与 ETA 提示；其余可放弃 | S |
| 76 | 确认出口 / 签字后提示「处理下一例」「下一例待审阅」 | app.js `offerNext` | MISSING | v2 确认或复核后只有提示条 | 迁移 | S |
| 77 | 同时预测与复用来源说明：「与任务 X 同时上传」「同时预测体场：确认出口后自动创建 / 未能创建（原因）」「复用自任务 X」「由任务 X 换包重跑」 | app.js `renderJob`（`companion_of`/`companions`/`reused_from`/`source_job_id`） | PARTIAL | 只有已完成的同输入结果出现在概览头部下拉（ws_shell.js `siblings`）；未建成 / 失败的伴随任务、来源说明都不显示 | 迁移 | S |
| 78 | 待确认 / 计算中页也能展开「输入检查与计算过程」 | app.js `renderJob` → `processCard` | PARTIAL | v2 计算过程只在已完成结果的「工具」页 | 接受或迁移 | S |

### I. 任务详情：已完成结果

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 79 | 头部：名称、审阅徽章、结果类型、患者 / 扫描 / 日期、标签、创建时间、任务号 | app.js `renderJob`（`.detail-head`） | HAVE | ws_overview.js `header` + 工具页病例信息 / 技术信息 | — | — |
| 80 | 「打开三维报告」「一页纸」按钮 | app.js `renderJob` `jobLink` | HAVE | v2 本身即三维；一页纸在导出对话框「一页纸」页（ws_figure.js）与 O 键 | — | — |
| 81 | 审阅签字 / 查看签字 / 重新打开（原因必填） | app.js `reviewDialog`/`reviewSection` | HAVE | ws_shell.js `reviewDialog` | — | — |
| 82 | 签字前核对清单：出口命名来源与置信度、结果质量、几何参考范围、「需要注意」 | app.js `reviewChecklist` | PARTIAL | v2 只有三个勾选项，不列这些事实 | 迁移 | S |
| 83 | 出口已自动确认提示（最低置信度） | app.js `resultCard`（`automatic_high_confidence`） | HAVE | ws_detail.js `outletModel`（工具页「来源、把握度」） | — | — |
| 84 | 结果数字：p99、全场最大值及所在分支、计算耗时；体场 p99、相对压力最高 / 最低 | app.js `resultCard` | PARTIAL | ws_overview.js `kpiModel` 最多 4 个：无全场最大值（在分支表里）、体场只有压力跨度、耗时在状态点 tooltip | 接受（矩阵已记 PARTIAL） | S |
| 85 | 结论（可编辑、恢复自动、锁定只读、「审阅人已修改」） | app.js `narrativeCard`/`narrativeDialog`/`saveNarrative` | HAVE | ws_overview.js 结论区 + ws_shell.js `narrativeDialog` | — | — |
| 86 | 周期量（TAWSS / OSI / 滞留区 / RRT / ECAP） | app.js `cycleBlock` | HAVE | ws_overview.js `kpiModel` + `cycleSub` | — | — |
| 87 | 重点发现（前 3 条 + 去报告看全部） | app.js `findingsBlock` | HAVE | ws_overview.js 发现区 | — | — |
| 88 | 瘤体形态 + 截面可靠性 | app.js `morphologyBlock` | HAVE | ws_overview.js `morphStrip`/`reliabilityModel` | — | — |
| 89 | 黄色告警横幅（几何超出发布包声明范围 / 质量差 / 集成分歧大，带「查看详情」）+「多模型一致性与复核提示」质量卡 | app.js `alertBanner`（`WB.alertModel`）、`resultCard` quality 卡 | MISSING | v2 不读 `summary.quality`，也没有横幅；参考范围只在证据透镜与「可信度标记」图层里 | 需裁定（记忆里有「不展示离散度」的裁定）；几何超范围的告警建议迁移 | S |
| 90 | 人群位置（折外人群分位）与几何范围卡 | app.js `resultCard`（`reference_assessment`） | PARTIAL | 只在证据透镜（ws_lens.js `populationRow`/`trustRows`）里，点 p99 才看得到 | 接受或在概览加一行 | S |
| 91 | 壁面面积加权 p99（补充口径）卡 | app.js `resultCard`（`surface_statistics`） | DROP-CANDIDATE | v2 无 | 放弃 | — |
| 92 | 峰值 WSS 面积占比（低 / 高 / 极高） | app.js `resultCard`（`areaFractions`） | HAVE | KPI 低 / 高 + 色标面板三阈值占比（ws_display.js） | — | — |
| 93 | 单文件下载：壁面 VTP、点云 CSV；体场 VTP、壁面压力 VTP、体场 CSV、流线 VTP | app.js `resultCard` actions | PARTIAL | ws_export.js `dataPane` 只给「壁面数据 VTP」+ zip + 统计表 CSV（+完整档 JSON）。**缺陷**：它按 `summary.exports` 里的文件名找 `.vtp`，但 exports 是布尔值（pipeline.py L696、volume_pipeline.py L232），所以体场结果没有任何 VTP 链接，壁面结果靠写死的 `wall_wss.vtp` | 迁移（按族列出固定文件名） | S |
| 94 | 打包 zip | app.js `bundleJobs([job])` | HAVE | ws_export.js 数据页「全部文件 zip」 | — | — |
| 95 | 完整统计 JSON、可复现运行清单 | app.js `resultCard` file-links、`detailMenuItems` | HAVE | ws_shell.js `renderTools` 技术信息（完整档） | — | — |
| 96 | 用其他发布包重跑 | app.js `resultCard` inline select、`detailMenuItems` | HAVE | ws_shell.js `renderTools`（完整档）+ 任务列表重跑 | — | — |
| 97 | 「比较结果」标量差表（`POST /api/compare`） | app.js `compareWith` | DROP-CANDIDATE | 已裁定不做（矩阵 A.5） | 放弃 | — |
| 98 | 「并排比较」：从本页已加载的同族已完成任务中选一个 | app.js `resultCard`（compareSelect → `/compare`） | PARTIAL | ws_compare.js `pickDialog`：同输入、同患者全列，「其他已完成结果」只列前 20 个且不能搜索 | 迁移（加搜索或放开 20） | S |
| 99 | 检查 / 修改出口命名并重算 | app.js `openOutletOverride` | HAVE | ws_detail.js 工具页出口命名 | — | — |
| 100 | 编辑病例信息 | app.js `metadataDialog` | HAVE | ws_detail.js `metadataDialog` | — | — |
| 101 | 删除（进回收站、撤销并重新打开） | app.js `deleteJobs`/`undoDelete` | HAVE | ws_detail.js `deleteJob`/`restoreJob` | — | — |
| 102 | 计算过程：单位、换算倍数、单位置信度、各阶段耗时、预计算、本次 / 累计、排队、人工确认、设备、发布指纹、输入 SHA、运行身份、时间帧、运行记录 | app.js `processCard` | HAVE | ws_detail.js `timingModel`/`eventRows`/`inputModel` + 技术信息；只少「中心线检查通过 / 端点·分叉数」两行 | 保留 | — |
| 103 | 患者时间线：直径与瘤体体积年增长率（全程与最近两次）、每个发布包一条线、瘤体体积列、勾两次扫描「并排比较」、日期缺失注记 | app.js `timelineCard`/`renderTimeline`/`timelineTable`/`growthFact`/`sparkChart` | PARTIAL | ws_overview.js 随访：只有直径年增长、两条小曲线（直径 + 本发布包一个指标）、表格；无瘤体体积增长、无「最近两次」、无跨发布包、无勾选比较（可用比较选择框「同一患者」代替） | 迁移增长率与瘤体体积 | M |
| 104 | 血管缩略图（可拖动，点开报告） | app.js `vascularThumbnail`/`renderVascularThumbnail3D` | HAVE | 主视口 + 首页卡缩略图（ws_thumbs.js） | — | — |
| 105 | 窄屏「← 返回病例列表」 | app.js `renderJob`（`.back-to-cases`） | HAVE | 病例栏展开（ws_shell.js `toggleRail`） | — | — |

### J. 并排比较页（compare.html / compare.js）

| # | 功能 | 旧入口 (file:function/id) | v2 状态 | 证据或缺什么 | 建议 (迁移/放弃/保留) | 工作量 S/M/L |
|---|---|---|---|---|---|---|
| 106 | 同步视角 | compare.html `#sync-camera` | HAVE | ws_compare.js「同步视角」、ws_shell.js `applySync` | — | — |
| 107 | 同步显示口径 + 统一色标上限（取两例 p99 较大值） | `#sync-display`、compare.js `unifyRanges`/`WB.sharedDisplayRange` | HAVE | 「同尺度看幅值」（`commonRange`） | — | — |
| 108 | 以左为准 / 以右为准 | `#push-left`/`#push-right` | HAVE | ws_compare.js → ws_display.js `pushSide` | — | — |
| 109 | 左右互换 | `#swap-sides` | HAVE | ws_shell.js `renderComparePanel` onSwap | — | — |
| 110 | 类型 / 字段不同时的色标说明 | compare.js `setScaleNote` | HAVE | 条件表判「不一致」并禁用同尺度 | — | — |
| 111 | 标量比较表（差值 = 右 − 左） | compare.js `renderComparison`（`POST /api/compare`） | DROP-CANDIDATE | 已裁定 | 放弃 | — |
| 112 | 两侧身份：病例、患者、扫描、日期、发布包、「已审阅签字」 | compare.js `renderIdentity` | PARTIAL | v2 显示名称 + 结果名，无审阅状态与扫描日期 | 迁移或接受 | S |
| 113 | 两侧各自是完整报告（截面、测量、探针、各自色标等都能用；体场两侧都能开截面） | compare.html 两个 iframe 加载 `/api/jobs/<id>/report` | PARTIAL | v2 比较时 `toggleSlice`/`toggleMeasure` 直接拒绝（「请先退出比较」），右侧只能换字段和色标窗 | 需裁定；要支持就是大改 | L |
| 114 | 任意两任务按 URL 打开（`/compare?left=&right=`，跨病例） | compare.js `readQuery` | PARTIAL | v2 用 `#/job/<L>?v=compare&cmp=<R>` 等价；旧 URL 需服务端跳转（见 §3） | 迁移（服务端重定向） | S |

## 2. 只有经典页面调用的接口

| 接口 | 经典调用处 | v2 情况 | 建议 |
|---|---|---|---|
| `POST /api/jobs/claim` | app.js `claimJobs` | 无调用 | 迁移（#6），或确认已无可认领 owner 后放弃 |
| `GET /api/cases`（病例卡：`missing_releases`、`reusable_job_id`、`latest`、`case_ids`） | app.js `refreshCases` | 无调用（v2 在前端按 `patient_id`/`input_sha256` 分组） | 放弃接口本身；#34「补跑」的信息改前端算 |
| `POST /api/compare` | app.js `compareWith`、compare.js `start` | 无调用 | 放弃（已裁定） |
| `GET /api/jobs/<id>/events`（单任务 SSE） | app.js `openEvents` | 无调用；v2 用 `/api/events` + 输入页 3 秒轮询 | 放弃（功能已覆盖） |
| `GET /api/trash?all=1` | app.js `refreshTrash` | v2 只调 `/api/trash` | 放弃（#58） |
| `POST /api/jobs/<id>/confirm` 带 `inlet` | app.js `outletCard` 重提中心线 | v2 调 confirm 但从不带 `inlet` | 迁移（#74） |
| `POST /api/jobs/<id>/retry`（对 `failed`） | app.js `renderJob` | v2 只对 `interrupted`/`cancelled` 调 | 迁移（#69） |
| `POST /api/jobs/<id>/cancel`（对 `awaiting_*`） | app.js `cancelJob` | v2 只对排队 / 计算中调 | 迁移（#68） |
| `GET /api/jobs/<id>/files/points_wss.csv`、`volume_fields.vtp`、`wall_pressure.vtp`、`points_volume.csv`、`streamlines.vtp` | app.js `resultCard` | 无单独链接（只在 zip 里） | 迁移（#93） |
| `GET /api/jobs/<id>/report`（iframe / 新窗口）与 `wss-view:*` 消息（`get-state`/`apply-state`/`set-camera`/导出） | compare.js、batch_export.js、app.js `openReport` | v2 只在「工具 → 经典报告」和出错时作为外链 | 经典报告若也退役，#56 批量出图必须在 v2 查看器上重写 |
| `GET /static/glossary.json` | app.js `loadGlossary` | 无调用 | 放弃浏览器端用法；**文件保留**（report_freshness.py L34、doctor.py L180、报告内嵌术语依赖它） |
| 登录响应里的 `claimable` / `legacy_token_allowed`（字段而非接口） | app.js `applySession`/`renderLoginPanel` | v2 不读 | 见 #2、#6 |

其余经典调用（session、logout、password、preferences、releases、health、jobs 列表 / 详情 / 上传 / 批量上传、rerun、metadata、input、review、narrative、geometry、export（ids 与 json）、bundle.zip、bundle、delete（单个与批量）、trash restore / purge、patients timeline、report-template、onepage、`/api/events`）v2 都有调用：ws_api.js、ws_admin.js、ws_upload.js、ws_detail.js、ws_onepage.js、ws_shell.js。

## 3. S7 连带依赖（不是功能，但删文件前必须处理）

1. **旧深链接**：报告页「← 工作台」指向 `/#job=<id>`（report.py L777、static/volume_viewer.js L1299）；cli.py L313 打印 `{base}/#job=<id>`；经典列表与时间线打开 `/compare?left&right`；devshot.py L264/L271 用这两种 URL。需要服务端把 `/` 与 `/compare` 跳到 `/v2/`，并由 v2 把 `#job=<id>` 转成 `#/job/<id>`、把 `?left&right` 转成 `#/job/<left>?v=compare&cmp=<right>`（或直接改这些生成方）。
2. **v2 里指回经典的链接**（经典页一删就成死链或循环）：ws_shell.js L261 头像菜单「经典工作台」、L509 未完成任务工具栏「在经典工作台中处理」（这是 #68–#70、#74 目前唯一的退路）、L1985 登录页脚；ws_main.js L31 出错页；v2/index.html L61 noscript；ws_api.js L173 `urls.classic`。ws_shell.js L501/L1786「在经典报告中打开」指的是报告页，只有报告也退役时才要动。
3. **运维中心与工单页**：static/ops.html、static/support.html 引用 `/static/app.css`，「返回工作台」指向 `/`；v2 里没有任何入口通往 `/ops`、`/support`（经典工作台里也没有）。删 app.css 会让这两页失去样式。
4. **共享静态文件不能删**：v2/index.html 直接加载 `/static/three.min.js`、`/static/OrbitControls.js`、`/static/report_common.js`、`/static/volume_viewer.js`；`glossary.json` 见 §2。能删的只有 index.html、app.js、app.css（先处理第 3 条）、workbench_core.js、batch_export.js、compare.html、compare.js，并同步 server.py L49–50 `STATIC_FILES` 与 L1499–1501 路由。
5. **测试**：tests/ 下 9 个文件引用经典文件（test_workbench_js、test_compare_page、test_v2_lane_c（把 v2 纯函数与 workbench_core.js 逐项比对）、test_report_common、test_server_security、test_v2_routes、test_glossary、test_v2_workspace_js、test_v2_display），S7 要一并改。
6. **示例 STL**：`static/v2/example_aaa.stl` 被 .gitignore 第 88 行排除，没进版本库。经典与 v2 上传框都链到它，按 git 部署时会 404（lane C 已记）。

## 4. 退役前必须先迁移（按优先级）

**P0：没有它们，删掉经典页后有任务处理不下去（现在 v2 靠「在经典工作台中处理」兜底）**

1. 失败任务「重试」（#69），连同错误卡的类别说明、改用 CPU 提示、诊断编号、管理员技术细节（#70）。S
2. 待确认任务「取消」（#68）。S
3. 出口确认里「重新选择入口 → 重提中心线」（#74）。M
4. 旧 URL 跳转：`/#job=`、`/compare?left&right`（§3-1），以及清掉 v2 里指回经典的链接（§3-2）。S
5. 认领旧会话任务（#6）；迁移期旧令牌登录（#2）。先只读查一下服务端还有没有未认领的 owner、`legacy_token_allowed` 是否还开；都没有就两项放弃。S
6. 运维中心 / 工单页的 app.css 依赖和入口（§3-3）。S

**P1：数据或安全相关，建议在 S7 同一批做**

7. 数据页下载修正：体场结果没有 VTP（`summary.exports` 是布尔值）；补点云 CSV、体场 / 壁面压力 / 流线 VTP 链接（#93）。S
8. 告警横幅 / 结果质量（#89）：先裁定（考虑「不展示离散度」的旧裁定），几何超出参考范围的提醒建议保留。S
9. 跨病例批量出图（#56）：迁移（在 v2 查看器上重写）或明确放弃并写进矩阵。L
10. 队列：审阅筛选加「已重新打开」、数值下限筛选（#60、#61）。S + M
11. 随访：瘤体体积与最近两次增长率（#103）。M
12. 病例「补跑缺少的发布包」（#34）。M

**P2：体验差异，可在 S7 之后补**

登录细节（#3–#5）、ETA 阶段表（#65）、输入确认事实表（#72）、出口确认细节与「下一例」（#75、#76）、来源说明（#77）、签字核对清单（#82）、列表行 ETA / 行菜单 / 病例卡直径与标签（#30、#36–#38）、比较选择框放开 20 条（#98）、比较身份（#112）、编号校验（#26）、最近完成（#47）、三步上手自动展开（#17）、键位变化写进说明（#50）。#113（比较两侧完整工具）需要裁定是否接受 v2 的简化。

## 5. 与迁移矩阵（2026-09-30）不一致的地方

- 矩阵没有的行：认领旧会话任务、迁移期旧令牌登录、登录细节、失败任务重试、待确认任务取消、错误卡详情（含管理员技术细节）、确认后「下一例」、结果质量与告警横幅、单文件下载（及体场 VTP 缺陷）、病例卡「补跑」、比较两侧完整工具、旧 URL 与 ops/support 依赖。
- 矩阵写「已有」、本审计判 PARTIAL 或 MISSING 的：出口命名确认（缺重提中心线，#74 判 MISSING）、进度与剩余时间（#65）、输入检查（未完成任务那一面，#72）、患者时间线（#103）、审阅签字的核对清单（#82）。
- 矩阵「新建：元数据」「列表：按患者 / 标签筛选」「服务状态弹窗」「断线与升级提示」「会话过期不丢页」「报告模板」核对无误。
