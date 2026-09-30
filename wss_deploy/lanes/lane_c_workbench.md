# 第二期 C 路（S6c）工作台 — 交付报告

分支 `wss-ui-s6c-workbench`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_c`。范围见 `PHASE2_LANES.md` §5 C，九项全部做了（优先级 1–9）。

## 1. 做了什么，入口在哪

| 功能 | 入口 | 代码 |
|---|---|---|
| 首页四页签：病例 / 任务 / 队列 / 回收站（当前页签是大标题，旁边是计数） | 首页顶部；打开病例时病例栏底部也有「任务 · 队列 · 回收站」 | `ws_rail.js` `homeHead` |
| 首页计数：今日、进行中、待处理（待确认 + 待复核）、失败；点一下进任务列表并带上筛选 | 首页标题下四个数字块 | `ws_rail.js` `homeCounts` / `todo` |
| 「需要处理」加入待复核；超过 8 条时每一类都保留位置（轮流取），「全部 N 条」进任务列表 | 首页 | `ws_rail.js` `attnPick` |
| 平铺任务列表：点表头排序（病例、患者 / 扫描、结果、状态、创建、属主），状态筛选（待确认 / 计算中 / 待复核 / 已复核 / 失败 / 已取消），患者编号、标签精确筛选（带候选），服务端搜索，每页 25 / 50 / 100 翻页 | `#/tasks` | `ws_admin.js` `mountTasks` / `taskModel` |
| 多选与批量：删除（进回收站，提示条上可「撤销」）、重跑（选模型，逐个建任务，沿用中心线与出口确认）、汇总表 CSV / Excel、打包 zip | 任务列表勾选后底部浮出的深色操作条 | `ws_admin.js` `renderSelbar` 等 |
| 回收站：恢复（可直接打开）、彻底删除（二次确认），剩余天数 | `#/tasks` 旁的「回收站」页签，`#/trash` | `ws_admin.js` `mountTrash` |
| 队列总览：已完成任务的关键数字表（可排序，空列自动隐藏），按结果类型 / 模型 / 复核状态筛选，一个直方图（量可选；WSS p99 叠人群参照线；点柱子只看那一段），导出 CSV / Excel | `#/cohort` | `ws_admin.js` `mountCohort` / `renderHistogram` |
| 服务状态 | 头像菜单 →「服务状态…」 | `ws_admin.js` `healthDialog` |
| 断线提示：实时更新断开 2.5 秒后顶部出现「连接中断，正在重连…」，恢复后自动消失并刷新列表；流被浏览器放弃时，服务一恢复就重新打开 | 自动 | `ws_admin.js` `onConnection` |
| 服务升级提示：版本号或新工作区页面文件摘要（`ui_build_v2`）变了，顶部提示「刷新页面」 | 自动（每 30 秒、重连时各查一次） | `ws_admin.js` `checkVersion` |
| 改口令 | 头像菜单 →「修改口令…」 | `ws_admin.js` `passwordDialog` |
| 管理员「看全部用户」：列表、病例栏、首页卡片、队列都显示全部用户；任务列表多一列「属主」，别人的任务只能看（删除、重跑按钮对它们不可用） | 头像菜单勾选 | `ws_admin.js` `toggleViewAll`，服务端 `owner_name` |
| 通知：任务完成 / 失败 / 待确认时提示条（带「打开」），病例栏和首页标出未读，标题栏显示未读数；打开该任务即已读；页面在后台时可发桌面提醒（头像菜单开关，记在账户偏好里） | 自动 | `ws_admin.js` `onEvent` |
| 新建：整页拖放 STL（遮罩提示，可多个，非 STL 自动忽略）；多文件批量上传（按服务上限分批走 `/api/jobs/batch`，每个文件一行结果，重复几何逐个选择「打开已有 / 只算所选模型 / 重新计算 / 跳过」）；补齐扫描标签、标签、备注、「记住患者编号、扫描信息和标签」；完整档多出设备、集成模型数、CPU 线程 | 顶栏「上传 STL」或把文件拖进页面 | `ws_upload.js` |
| 会话过期原地重新登录：登录框浮在页面最上层（在已打开的对话框之上），已填的内容不丢；同一用户继续用，不刷新页面，实时更新重新接上；换了用户就整页刷新 | 自动 | `ws_main.js` `openRelogin` |

少字的处理：说明文字都放进 ⓘ（任务列表的搜索口径、队列数字的来源、人群参照的含义、回收站保留期、「记住患者」的用途）；上传对话框里新增的字段默认折叠在「扫描标签、标签、备注」「计算设置」两个小节里。

## 2. 与经典工作台的一致性

- **接口与文档同源**：列表 `GET /api/jobs`（服务端搜索与翻页用 `q / status / patient_id / tag / page / page_size`），删除 `POST /api/jobs/delete`，撤销与恢复 `POST /api/trash/<id>/restore`，彻底删除 `/purge`，重跑 `POST /api/jobs/<id>/rerun`，汇总表 `GET /api/jobs/export?ids&format`，打包 `/bundle.zip` 与 `POST /api/jobs/bundle`，队列 `GET /api/jobs/export?format=json&scope=done`，偏好 `GET/PUT /api/preferences`（与经典同一份文档，分节合并，不覆盖别的节），改口令 `POST /api/session/password`，上传 `/api/jobs`、`/api/jobs/batch`（同样的字段与 `metadata_json`）。
- **规则逐条对齐 `workbench_core.js`**（`tests/test_v2_lane_c.py::test_pure_helpers_match_the_classic_workbench_core` 在 Node 里把两边的结果直接比较）：直方图分箱 `binEdges / histogram`、人群参照的选取 `populationValues`、重跑跳过规则 `rerunSkipReason`、偏好合并 `mergePreferences`、上传前文件检查 `checkUploadFiles`、分批 `uploadChunks`、标签解析 `parseTags`。
- **数字逐项比对**（同一沙箱、同一时刻，经典工作台「队列总览」面板 vs `#/cohort`）：

  | 病例 | 管腔最大直径 mm | p99 Pa | 高占比 | 人群分位 |
  |---|---|---|---|---|
  | PENG_JI_MING（经典） | 38.58 | 20.65 | 29.90 % | 52.21 |
  | PENG_JI_MING（新） | 38.6 | 20.7 | 30% | 52.2 |
  | WANG_SHUN_WEN（经典） | 77.49 | 19.39 | 11.64 % | 45.59 |
  | WANG_SHUN_WEN（新） | 77.5 | 19.4 | 12% | 45.6 |

  原始值（导出 JSON）38.5779…、20.6525…、0.29903、52.206：两边来自同一行，只是位数规则不同（见下）。
- 删除进回收站、30 天、撤销恢复同一批、别人的任务 / 已复核 / 计算中不能删，与经典相同。

## 3. 与经典的差异及原因

1. **列表取数**：经典按 20 条一页向服务端要；新工作区一次取回本账号的全部任务（`GET /api/jobs` 不带分页参数，服务端本来就支持），排序与翻页在页面里做，所以任何列都能全局排序、超过 100 个任务也能看全。只要填了搜索词、患者编号或标签，就改用服务端查询并把所有页取回来（每页 100）。此前新工作区只取最新 100 个任务。
2. **首页计数**按本路范围定为「今日 / 进行中 / 待处理 / 失败」，与经典的「需要处理 / 计算中与排队 / 待审阅 / 近 7 天完成」口径不同；「待处理」= 待确认出口或输入 + 已完成待复核。
3. **首页病例卡片**先显示最新 48 个病例，「再显示 48 个」按钮往下加（每张卡都会排队画缩略图，全部历史一次铺开会一直在后台读结果）。搜索时不限。
4. **队列总览**：经典有四个直方图和五个数值下限输入框；新页面只有一个直方图（量可选），用「点柱子筛这一段」代替数值下限；表格列按结果类型自动隐藏全空的列；数字按 §7 取三位有效数字、百分比取整（经典两位小数）；人群参照画成折线（经典是浅色柱），颜色对经过配色校验（蓝柱 `#2a78d6` / 橙线 `#eb6834`，色觉异常下也分得开）。
5. **回收站**在「看全部用户」时仍只列自己的：服务端只允许恢复 / 彻底删除自己的任务，列出别人的只会得到失败的按钮（ⓘ 里写明）。
6. **通知**：经典把「状态第一次出现」都当成新消息，所以对已完成任务做复核、编辑信息或从回收站恢复时也会弹「已完成 / 失败」；新工作区用已载入列表里的状态做起点，并忽略复核、编辑、恢复这类不改状态的事件，只报真正的状态变化。取消的任务不提示（操作的人已经知道）。
7. **改口令**成功后说明「其他浏览器里的登录已退出」（服务端本来就这样做）。
8. **重跑**只对自己的已完成任务可用，一次对话框完成，逐个显示结果（与经典批量重跑相同的跳过规则）。

## 4. 改动过的公共文件

- `wss_deploy/static/v2/ws_shell.js`（都标了 `lane C`，没有挪动或改名）：
  1. `refreshJobs`：有 `ns.admin.loadJobs` 时用它取列表（全量、「看全部用户」），没有时行为不变；
  2. 事件流抽成 `openEvents()`，每个事件后 `extCall('onEvent', ev)`，流状态 `extCall('onConnection', 'up'|'down')`（原来没把 `onState` 传给 `api().events`）；
  3. `start()` 在第一次路由之后 `extCall('onStart')`；
  4. `SHELL_API` 的 `// lane C` 后加 `reopenEvents()`（重新登录后、服务重启后重新打开事件流）；
  5. 扩展点说明注释里补上这三个钩子名。
- `wss_deploy/server.py`：`GET /api/jobs` 在管理员 `all=1` 时给每个任务加 `owner_name`（注册用户名；匿名 / 旧令牌会话为空串，内部 owner 键不外露），新方法 `Handler._owner_names`。其他调用方与非管理员返回不变。测试 `test_admin_all_users_list_carries_owner_names`。
- 未改：`bundle.json`、`index.html`（没有新文件）、`ws_api.js`、`ws_icons.js`（新图标 `chevron-up / trash / list / chart` 用 `register` 加）、`v2.css`、已有测试。

合并时请注意：`ws_rail.todo()` 现在先把页面路由（`#/tasks` 之类，也包括别的路注册的页面）交给扩展的 `page` 钩子重画——外壳在每次列表刷新后都会调 `renderHome()`，原来会把任何扩展页面盖成病例画廊。这个修正放在我名下的 `ws_rail.js` 里（`pageHandled`），别的路的页面也受益；主会话若更愿意在外壳 `renderHome` 里处理，删掉 `pageHandled` 即可。

## 5. 测试结果

- `node --check`：`ws_admin.js`、`ws_rail.js`、`ws_upload.js`、`ws_main.js`、`ws_shell.js` 全过。
- 新测试 `tests/test_v2_lane_c.py`（11 项）：与 `workbench_core.js` 的纯函数对照、首页、任务列表（翻页、服务端搜索取全部页、批量删除 + 撤销、汇总表）、回收站、队列页（排序、空列隐藏、直方图筛选）、通知 / 未读 / 断线横幅 / 升级提示 / 看全部用户、原地重新登录保留对话框、上传（整页拖放、批量、元数据、完整档、重复几何的逐个处理、记住患者）、扩展页面在列表刷新后保持、服务端 `owner_name`。全部通过。
- 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`：**880 passed, 33 skipped**（3 分 51 秒）。没有改动已有测试；没改分析、流水线、叙述相关的 Python，所以没跑黄金回归。
- 沙箱浏览器（:8823，Marionette 3023，118 个合成任务 + 12 个预览任务，另建用户 bob 的 11 个任务）逐项走过，所有步骤 0 个 JS 错误（见下方截图）。

## 6. 截图

- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/01_home.png` — 首页：四页签、四个计数、需要处理（含待复核）、病例画廊
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/02_tasks.png` — 任务列表（122 个任务，第 1/3 页）
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/03_tasks_search.png` — 服务端搜索
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/04_tasks_filters.png` — 状态 + 标签精确筛选
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/05_tasks_page3.png` — 翻到第 3 页（整页截图）
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/06_tasks_selected.png` — 多选后的操作条
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/07_delete_confirm.png` — 删除确认
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/08_deleted_toast.png` — 移入回收站 + 撤销
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/09_trash.png` — 回收站
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/10_purge_confirm.png` — 彻底删除确认
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/11_cohort.png` — 队列总览：直方图（WSS p99 + 人群参照线）与关键数字表
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/12_cohort_bin.png` — 点直方图的一段筛表格
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/13_cohort_tawss.png` — 换成 TAWSS 均值、只看壁面
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/14_user_menu.png` — 头像菜单（服务状态、修改口令、看全部用户、桌面提醒）
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/15_health.png` — 服务状态
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/16_password_wrong.png` — 改口令：当前口令错误
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/17_tasks_all_users.png` — 看全部用户：属主列，别人的任务名称变淡
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/18_home_all_users.png` — 看全部用户：首页卡片带属主
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/19_rerun_dialog.png` — 重跑对话框
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/20_rerun_started.png` — 重跑已建任务
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/21_notice_toast.png` — 任务失败提示条
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/23_conn_down.png` — 服务断开：连接中断横幅
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/24_conn_back.png` — 服务恢复：横幅消失、连接已恢复
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/25_update_banner.png` — 页面文件更新后的刷新提示
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/26_relogin.png` — 会话过期：登录框浮在上传对话框之上
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/27_relogin_done.png` — 重新登录后上传对话框和已填内容还在
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/28_drop_overlay.png` — 整页拖放遮罩
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/29_upload_batch.png` — 拖入两个 STL 后的批量上传对话框
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/30_upload_duplicates.png` — 批量上传：重复几何逐个选择
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/30b_upload_skip.png` — 跳过一个
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/31_upload_done.png` — 上传完成后打开新任务
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/32_upload_fields.png` — 补齐的元数据字段与完整档计算设置
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/33_home_390.png` — 首页 390 宽
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/34_tasks_390.png` — 任务列表 390 宽
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/35_cohort_390.png` — 队列 390 宽
- `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_c/shots/36_rail_unread.png` — 病例栏未读标记（失败提示后）

## 7. 没做完 / 待定

- 任务行上的「⋯」快捷菜单（经典有：一页纸、编辑信息、患者时间线、删除）——现在点行打开任务，操作用多选条；编辑信息等在结果页（D 路）。
- 经典的「认领旧会话任务」横幅、病例卡上的「补跑缺少的结果 / 并排打开 WSS + 体场 / 导出该病例汇总」、「跨病例批量出图」没有迁（不在本路范围）。
- 经典列表的「近 7 天完成」快捷筛选没有单独做（按创建时间排序即可看到）。
- 观察到的既有问题（非本路）：`static/v2/example_aaa.stl` 被 `.gitignore` 排除，工作副本里没有这个文件，上传对话框里「下载示例 STL」在本分支是 404。
- 沙箱里的预览任务引用了正式任务目录的绝对路径（`input_clean_mm.stl`），在沙箱里「换模型重跑」会在 B 段失败；这是数据副本的问题，正式服务上不受影响（本路的重跑按钮在沙箱里验证到了建任务与失败通知）。
