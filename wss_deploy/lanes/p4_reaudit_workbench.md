# 经典工作台 + 比较页 → v2 复核（第三期合并后）

- 对象：`p3_audit_workbench.md` 全部 114 行。分支 `wss-ui-v2`，HEAD `e7c2356`，工作副本 `/public/newhome/cy/Digital_twin/GNN_wssui_v2`。
- 做法：
  - 经典代码用 `git show 866cf02:…` 读（`app.js`、`workbench_core.js`、`index.html`）。
  - v2 读的是当前代码，没有只看各路报告。逐行用 grep 找到函数，再读实现。几处关键数据也到服务端核对过：任务列表里的 `reusable`、`max_diameter_mm`、`eta`；`geometry` 返回的 `preview_polylines`；`_metadata` 的校验。
  - `static/v2` 下所有 js 都过了 `node --check`。v2 代码里提到已删经典文件的地方只剩注释。
- 状态说明：
  - DONE：已完成。
  - STILL-PARTIAL：只做了一部分。
  - STILL-MISSING：还没有。
  - DROPPED（§0）：`PHASE3_LANES.md` §0 明文放弃。
  - DROPPED（审计）：原审计判为 DROP-CANDIDATE，第三期 §3 没有排期，§0 也没写。下面逐条写了是什么替代了它。

**统计（114 行）：DONE 96 · STILL-PARTIAL 8 · STILL-MISSING 0 · DROPPED 10（§0 明文 4，审计放弃 6）**

和原审计相比的变化：

- 原来 MISSING 的 13 行和 PARTIAL 的 26 行，大部分已做完。
- 原判 DROP-CANDIDATE 的 #21、#97、#111，第三期反而做了：#97、#111 已完成，#21 做了一部分。
- 新发现缺口 2 处，原审计都判 HAVE：
  - #73：出口确认的三维视图不画中心线；
  - #102：计算过程里少两行中心线检查。

## 未完成的行（STILL-PARTIAL / DROPPED）

| # | 功能 | 现状 | 证据或还缺什么 | 用户损失 | 使用频率 | 工作量 | 建议 |
|---|---|---|---|---|---|---|---|
| 2 | 迁移期旧令牌登录 | DROPPED（§0） | §0-1：正式服务只有 `admin` 一个账号，两个任务都属于它，没有可认领的。`ws_shell.js` `showLogin` 和 `ws_main.js` `openRelogin` 不读 `legacy_token_allowed` | 无 | — | — | 放弃（已裁定）。以后开多账号时再评估 |
| 6 | 认领旧会话任务 | DROPPED（§0） | 理由同 #2。v2 没有调用 `/api/jobs/claim`，登录响应里的 `claimable` 被忽略 | 无（没有未认领的属主） | — | — | 放弃（已裁定） |
| 21 | 术语「?」说明读 `glossary.json` | STILL-PARTIAL | **已做**：第 3 路的 `ws_lens.js` `loadGlossary` / `termText` / `termTip`，在线读 `/static/glossary.json`，离线读内嵌的 `#wss-glossary`。已接到 `ws_overview.js`（管腔最大直径、质量、人群分位、结论、迂曲、增长率、体场术语等）、`ws_detail.js` 技术信息、`ws_notice.js`。<br>**还缺**：经典工作台上带「?」的这几处，v2 仍是内置文字：<br>• 面积占比 `area_fraction`、阈值 `thresholds`（`ws_display.js` 色标面板）；<br>• 复核状态 `review_status`（`ws_review.js`）；<br>• 队列列头 p99、高占比（`ws_admin.js` `renderCohort`）。<br>经典弹窗里的英文名和英文释义（`en` / `en_desc`），v2 哪里都不显示 | ⓘ 里有说明文字，但这几处可能和一页纸术语表说法不一致；看不到英文术语 | 很少 | S | 以后再说。把这几处换成 `ns.lens.termTip(key, 内置文字)` 即可，可以顺手做 |
| 26 | 上传时病例 / 患者编号即时校验 | STILL-PARTIAL | `ws_upload.js` 只有「可能是姓名」提醒（`looksLikeName`）和 `maxLength`，没调用 `ws_detail.js` 的 `identifierIssue`（不可见字符、超长时报错并阻止上传）。<br>不可见字符要到服务端 `jobs.py` `_metadata` / `_text` 才被拒，这时 STL 已经传完。<br>标签「最多 12 个、每个不超过 40 字」也只在服务端检查 | 从 Excel 复制的编号带换行或制表符时，大文件传完才报错，要重填重传 | 偶尔 | S | 迁移。在 `paintFiles` / `send` 前对病例名、患者编号、扫描标签、标签调用 `ns.detail.identifierIssue`，有错误就禁用「上传并检查」 |
| 35 | 病例卡「导出该病例汇总 CSV」 | STILL-PARTIAL | 首页病例卡（`ws_rail.js` `todo` 画廊）和任务行菜单 `ws_admin.js` `rowMenu` 都没有这一项。导出 →「数据」只有单个结果的统计表（`ws_export.js` `dataPane`）。<br>替代路径：`#/tasks` 按患者编号精确筛选 → 本页全选 →「汇总表 CSV / Excel」（`ws_admin.js` `exportSummary`） | 一个病例多个结果的汇总要多走 3 步 | 偶尔 | S | 放弃。替代路径够用；真要加，就在病例卡放个小按钮调 `exportSummary` |
| 43 | 列表「刷新」按钮 | DROPPED（审计） | 被事件流取代：`/api/events`，加上 `ws_admin.js` `onConnection` / `startDownPoll` 恢复后自动刷新；维护结束后也会 `refreshJobs` | 无（浏览器刷新也行） | — | — | 放弃 |
| 46 | 「近 7 天完成」口径 | DROPPED（审计） | 首页改成「今日上传」加近 14 天柱（`ws_rail.js` `homeStats`），另有「最近完成」（`recentDone`） | 无 | — | — | 放弃 |
| 48 | 概览里的服务状态块和快捷键块 | DROPPED（审计） | 改放在头像菜单「服务状态…」（`ws_admin.js` `healthDialog`）和 `?` 快捷键表（`ws_shell.js` `shortcutsDialog`） | 无 | — | — | 放弃 |
| 56 | 跨病例批量出图 | STILL-PARTIAL | **已做**：`ws_batch.js` `open` / `run` / `plan`，离屏视口逐例出图，打成 zip（带 `manifest.json`）。入口是 `ws_admin.js` `renderSelbar` → `openBatch`。<br>**还缺**：<br>① 经典的 6 个内置预设：壁面「瘤囊低 WSS 区」「髂分叉热点」「主动脉沿程」，体场「沿程压降」「流线全貌」「瘤囊截面系列」；<br>② 不接受经典报告导出的视图状态 JSON（`wss-deploy.view/v1`），只接受 v2 复现链接；<br>③ 离屏视口不画第 4 路的等值线、高亮最高 x%、剖切（`installProvider` 只替离屏视口回答分段、阈值、单位、图层、标记），所以截面系列做不出来 | 研究出图要逐项手配字段和色标；以前存的视图 JSON 不能再用；截面系列、流线全貌没有一键版 | 偶尔（研究者写论文或汇报时） | M（截面系列 L） | 以后再说。先问研究者实际用哪几个预设；「瘤囊低 WSS 区」可以先用 TAWSS / WSS 固定 0–2 Pa 代替 |
| 58 | 管理员看全部用户的回收站 | DROPPED（§0） | §0-1 放弃了经典专用接口 `/api/trash?all=1`。服务端本来就只允许动自己的 | 无 | — | — | 放弃（已裁定） |
| 64 | 队列宽屏展开 | DROPPED（审计） | v2 的 `#/cohort` 本身就是整页 | 无 | — | — | 放弃 |
| 66 | 五步步进条 | DROPPED（审计） | 被状态点和输入页取代。第 1 路报告 §3 写明「审计 #66 判放弃，没做」 | 无 | — | — | 放弃 |
| 73 | 出口确认的三维视图：中心线 | STILL-PARTIAL（本次新发现，原判 HAVE） | 经典 `app.js` `createViewer` 在出口确认三维里画 `proposal.preview_polylines`（中心线折线）。<br>v2 的 `ws_input.js` `meshView` 只画壁面和端点。`api().geometry` 其实已经取回了 `preview_polylines`（服务端 `jobs.py` `geometry`，L1421，存在 `st.geometry`），但没画。<br>`PHASE3_LANES.md` §5 也把它列为第 1 路遗留 | 核对出口、决定要不要「重新选择入口」时，看不到中心线走向，难以判断中心线有没有走错分支 | 偶尔（每个要人工确认出口的病例都会看） | S | 迁移。数据已在手，加一组 `THREE.Line`，配色同经典 |
| 78 | 待确认 / 计算中的页面也能看「输入检查与计算过程」 | STILL-PARTIAL | 失败页有「输入尺寸与面积」（`ws_input.js` `failureCard` → `factsTable`），待核对输入页有事实表（`unitsCard`）。<br>待确认出口页和计算中页没有这三样：输入检查、各阶段耗时、运行记录。`ws_detail.js` `timingModel` / `eventRows` 只在已完成结果的工具页「计算过程」里。<br>第 1 路报告 §7 写明没做 | 确认出口时不能回看单位、尺寸和运行记录；排错要等结果出来 | 很少 | S | 以后再说。在出口页右栏底部放一个可折叠的 `factsTable(ic)` 加计算过程节即可，和 #102 一起做 |
| 84 | 结果数字（全场最大值和所在分支、体场压力最高 / 最低、耗时） | STILL-PARTIAL | `ws_overview.js` `kpiModel` 最多 4 格：<br>• 壁面：p99、低 / 高占比、直径；<br>• 体场：速度 p99、最大速度、相对压力跨度。<br>还缺三样，都在别处能看到：<br>• 全场最大值和所在分支：分支表的 `wss_max_pa` 列、「全场最大 WSS」发现、最大值标记；<br>• 体场压力最高 / 最低：概览只给跨度；<br>• 计算耗时：工具页「计算过程」 | 概览上一眼看不到最大值在哪一支、压力最高 / 最低，要多点一下 | 偶尔 | S | 放弃。KPI 四格是第二期的设计裁定，迁移矩阵已记 PARTIAL，数字在别处都能看到 |
| 91 | 壁面面积加权 p99 卡 | DROPPED（审计；报告审计 W28 判法相同） | v2 manifest 已带 `surface_statistics`（`v2_data.py` L718），但 JS 不显示；完整档的「完整统计 JSON」里能看到 | 研究者对照两种口径时要去 JSON 里找 | 很少 | S（放进证据透镜一行） | 放弃。经典页自己也注明主指标是点云 p99；有人要再放进透镜 |
| 102 | 计算过程 | STILL-PARTIAL（原判 HAVE，原审计已记缺口） | `ws_detail.js` `timingModel` / `eventRows` / `inputModel` 和技术信息都在。<br>仍少经典 `processCard` 的两行：「中心线检查：通过 / 未通过」「中心线端点 / 分叉数」。v2 全部代码里没有 `hard_pass` / `junctions` | 排查中心线异常时少一个直接信号 | 很少 | S | 以后再说（和 #78 一起补） |
| 113 | 比较时两侧都是完整报告（截面、测量等都能用） | DROPPED（§0） | §0-1：新工作区的比较保持现在的简化做法。比较时开截面、测量仍会提示「请先退出比较」 | 比较时右侧不能开截面或测量 | — | — | 放弃（已裁定） |

## 已完成（DONE，96 行）

- #1 登录（口令 / 令牌模式）— `ws_shell.js`:`showLogin`
- #3 显示 / 隐藏口令、大写锁定提示 — `ws_shell.js`:`showLogin`（`wsl-caps`）；`ws_main.js`:`openRelogin`
- #4 429 冷却倒计时 — `ws_shell.js`:`showLogin`（`retry_after`）；`ws_main.js`:`openRelogin`（`cooldown`）
- #5 记住用户名开关 — `ws_shell.js`:`showLogin`（`wsl-remember`，键名同经典 `wss-login-remember` / `wss-login-user`）
- #7 退出登录 — `ws_shell.js`:`logout`
- #8 修改口令 — `ws_admin.js`:`passwordDialog`
- #9 会话过期原地重登 — `ws_main.js`:`onUnauthorized` / `openRelogin`
- #10 管理员标识 — `ws_shell.js`:`renderTop`（头像菜单「· 管理员」）
- #11 管理员「全部用户」开关 — `ws_admin.js`:`toggleViewAll`；时间线已跟随开关：`ws_shell.js`:`loadTimeline`
- #12 服务状态浮层 — `ws_admin.js`:`healthDialog` / `healthRows`
- #13 503「维护中、自动重试」— `ws_admin.js`:`healthFetch` / `maintenance`；`ws_main.js`:`boot` / `openRelogin`；`ws_shell.js`:`showLogin`
- #14 版本更新后提示刷新 — `ws_admin.js`:`checkVersion`
- #15 实时流断线重连 — `ws_admin.js`:`onConnection` / `startDownPoll`
- #16 偏好存服务端、分节合并 — `ws_admin.js`:`loadPrefs` / `savePrefs` / `mergePrefs`
- #17 三步上手 — `ws_rail.js`:`guideCard`（`todo` 内：空库或没关过时展开）；`ws_admin.js`:`openGuide`（头像菜单）
- #18 快捷键说明 — `ws_shell.js`:`shortcutsDialog`
- #19 报告模板 — `ws_onepage.js`:`openTemplate`
- #20 队列总览 / 回收站入口 — `ws_rail.js`:`PAGES`；`ws_admin.js`:`mountTrash` / `renderCohort`
- #22 桌面通知 — `ws_admin.js`:`toggleNotifications` / `desktopNotice`
- #23 上传对话框 — `ws_upload.js`:`open`
- #24 整页拖放 STL — `ws_upload.js`:`installDrop`
- #25 逐文件预检、进度、分批 — `ws_upload.js`:`checkFiles` / `sendOne`（`api.upload` onProgress、`/api/jobs/batch`）
- #27 同几何查重 — `ws_upload.js`:`duplicate`
- #28 上传后打开新任务 — `ws_upload.js`:`open` → `ctx.onCreated`（`ws_shell.js`:`openUpload` → `go`）
- #29 按工作分组的列表 — `ws_rail.js`:`create` / `todo` / `attnPick`；`ws_admin.js`:`mountTasks`
- #30 病例卡直径和标签 — `ws_rail.js`:`caseFacts`（「待审阅 N」改由结果标签上的状态点表示）
- #31 病例卡编辑信息 — `ws_detail.js`:`metadataDialog`；`ws_admin.js`:`rowMenu` → `editInfo`
- #32 患者时间线 — `ws_overview.js`:`followupModel` / `followSection`；`ws_admin.js`:`openTimeline`
- #33 并排打开 WSS 与体场 — `ws_shell.js`:`siblings`；`ws_compare.js`:`pickDialog`（同一输入）
- #34 补跑缺少的结果 — `ws_rail.js`:`missingResults`；`ws_admin.js`:`fillMissing`
- #36 任务行「⋯」菜单 — `ws_admin.js`:`rowMenu`
- #37 任务行管腔直径 — `ws_admin.js`:`renderTaskTable`（「直径 mm」列，可排序）
- #38 计算中实时剩余时间 — `ws_rail.js`:`etaView` / `etaRowSummary` / `etaBadge`（病例栏、任务列表、首页卡）
- #39 搜索 — `ws_rail.js`:`create`（病例栏搜索）；`ws_admin.js`:`mountTasks`（服务端全文）
- #40 状态筛选 — `ws_rail.js`:`FILTERS`；`ws_admin.js`:`STATUS_FILTERS`
- #41 患者 / 标签精确筛选 — `ws_admin.js`:`mountTasks`（`E.patients` / `E.tags`）
- #42 分页、每页条数 — `ws_admin.js`:`mountTasks`（25 / 50 / 100）
- #44 未读点和计数 — `ws_admin.js`:`unreadAdd` / `updateTitle` / `repaintMarks`
- #45 今日概览瓦片 — `ws_rail.js`:`homeStats` / `homeCounts`
- #47 最近完成（带复核状态）— `ws_rail.js`:`recentDone`（`todo` 内「最近完成」节）
- #49 空列表提示 — `ws_rail.js`:`todo`（`ui.empty`）；`ws_admin.js`:`mountTasks` 空状态
- #50 键盘 — `ws_shell.js`:`shortcutsDialog`（J/K、/、N、?、O；G、O、R 的键位变化已写进说明）。列表 ↑/↓ 未迁：↑/↓ 现在用来移动截面，用 J/K 代替
- #51 多选、全选 — `ws_admin.js`:`renderTaskTable`（「选择本页全部」）
- #52 批量删除、撤销 — `ws_admin.js`:`deleteJobs` / `undoDelete`
- #53 批量汇总 CSV / xlsx — `ws_admin.js`:`exportSummary`
- #54 批量打包 zip — `ws_admin.js`:`bundleJobs`
- #55 批量换发布包重跑 — `ws_admin.js`:`rerunDialog`
- #57 回收站 — `ws_admin.js`:`mountTrash` / `restoreTrash` / `purgeTrash`
- #59 队列关键数字表 — `ws_admin.js`:`renderCohort`
- #60 复核筛选含「已重新打开」— `ws_admin.js`:`renderCohort`（L1188 选项，L1025 过滤）
- #61 数值上下限筛选 — `ws_admin.js`:`inRanges` / `readRange` / `rangeText`（另补了「极高 WSS 占比」列）
- #62 直方图 — `ws_admin.js`:`renderCharts`
- #63 导出筛选结果 CSV / xlsx — `ws_admin.js`:`renderCohort` → `exportSummary`
- #65 进度、剩余时间、阶段表 — `ws_input.js`:`etaView`（`create` 内「各阶段耗时」表，超时标橙）
- #67 取消排队或计算中的任务 — `ws_input.js`:`canCancel` / `cancelLines`
- #68 取消待确认的任务 — `ws_input.js`:`canCancel`（`AWAITING` 三种状态）
- #69 失败任务重试 — `ws_input.js`:`errorModel`（`RECOVER` 含 `failed`，判 `retryable` 和是否锁定）/ `failureCard`
- #70 错误卡 — `ws_input.js`:`errorModel` / `failureCard`（类别标题、改用 CPU 提示、`diagnostic_id`、`admin_detail`）
- #71 输入确认 — `ws_input.js`:`unitsCard`
- #72 输入确认的事实表 — `ws_input.js`:`inputFacts` / `factsTable` / `checkList` / `referenceText`
- #74 重新选择入口、重提中心线 — `ws_input.js`:`outletCard`（`api().confirm({mapping, inlet, acknowledged, version})`，L932）
- #75 出口确认细节 — `ws_input.js`:`outletReview` / `humanOutletReason` / `mappingChanges`；「复位视角」「保存截图」在 L547
- #76 确认或签字后的「下一例」— `ws_input.js`:`nextJob` / `offerNext` / `watchReview`
- #77 同时预测和来源说明 — `ws_input.js`:`provenance` / `provenanceBlock`（输入页，加工具页「来源」扩展）
- #79 结果头部 — `ws_overview.js`:`header`
- #80 三维 / 一页纸入口 — v2 本身就是三维；一页纸在 `ws_figure.js`（导出 →「一页纸」）和 O 键
- #81 审阅签字、重新打开 — `ws_shell.js`:`reviewDialog`
- #82 签字核对清单 — `ws_review.js`:`checklistModel` / `checklist`（由 `ws_shell.js` `reviewDialog` 调用）
- #83 出口自动确认提示 — `ws_detail.js`:`outletModel`
- #85 结论（编辑、恢复、锁定）— `ws_overview.js`:`narrativeModel`；`ws_shell.js`:`narrativeDialog`
- #86 周期量 — `ws_overview.js`:`kpiModel` / `cycleSub`
- #87 重点发现 — `ws_overview.js`:`findingsModel`
- #88 瘤体形态和截面可靠性 — `ws_overview.js`:`morphStripModel` / `reliabilityModel`
- #89 告警横幅和质量卡 — `ws_notice.js`:`model` / `bar` / `show` / `openDetails`；`ws_overview.js`:`qualitySection`（按 §0 不显示离散度数值）
- #90 人群位置与几何范围 — `ws_overview.js`:`qualitySection`（「人群参照」「几何参考」两行）；`ws_lens.js`:`populationRow`
- #92 峰值 WSS 面积占比 — `ws_overview.js`:`kpiModel`；`ws_display.js` 色标面板的阈值占比
- #93 单文件下载 — `ws_export.js`:`dataPane` / `dataFiles`（按结果类型列出，不再把 `summary.exports` 当文件名）
- #94 单例打包 zip — `ws_export.js`:`dataPane`（`api.urls.bundle`）
- #95 完整统计 JSON、运行清单 — `ws_shell.js`:`renderTools`；`ws_detail.js`:`techRows` / `techDialog`
- #96 换发布包重跑 — `ws_shell.js`:`renderTools`（完整档）；`ws_admin.js`:`rerunDialog`
- #97 「比较结果」标量差表 — `ws_compare.js`:`scalarSection` / `fetchScalars`（`POST /api/compare`；原判放弃，第 5 路按 C5 做了）
- #98 比较选择框 — `ws_compare.js`:`pickDialog` / `candidates` / `matches`（不再截到 20 条，可搜索）
- #99 修改出口命名并重算 — `ws_detail.js` 工具页「出口命名」节（confirm override，L753–785）
- #100 编辑病例信息 — `ws_detail.js`:`metadataDialog`
- #101 删除、撤销 — `ws_detail.js`:`deleteJob` / `restoreJob`
- #103 随访 — `ws_overview.js`:`followupModel` / `followSection`（首末与最近两次增长率、瘤体体积、多发布包曲线）
- #104 血管缩略图 — `ws_thumbs.js`:`request` / `keyFor`（IndexedDB 持久缓存）
- #105 窄屏返回病例列表 — `ws_shell.js`:`toggleRail`
- #106 同步视角 — `ws_compare.js`:`render`；`ws_shell.js`:`applySync`
- #107 同步显示口径、统一色标 — `ws_compare.js`:`commonRange` / `sameScaleAllowed`
- #108 以左为准 / 以右为准 — `ws_display.js`:`pushSide`
- #109 左右互换 — `ws_shell.js`:`renderComparePanel`（`onSwap`）
- #110 类型或字段不同时的色标说明 — `ws_compare.js`:`conditions` / `sameScaleAllowed`
- #111 标量比较表 — `ws_compare.js`:`scalarSection`（同 #97）
- #112 两侧身份 — `ws_compare.js`:`identity`（复核状态、患者、病例、扫描标签和日期；按「屏幕上不出现发布包代号」的规矩不显示代号）
- #114 任意两任务按 URL 打开 — `server.py`:`_classic_redirect`（`/compare?left&right` → `#/job/A?v=compare&cmp=B`）；`ws_legacy.js`:`parseAddress` / `classicToV2`

（#73、#102 是本次从 DONE 改判出来的，已计入上表，不计入这 96 行。）
