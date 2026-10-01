# 第二期并行开发合同（S5–S6，五路）

2026-09-30 起。五路从分支 `wss-ui-v2` 的同一个提交分出，各在自己的工作副本和分支上开发，最后由主会话逐路合并回 `wss-ui-v2`。本文件是五路共同遵守的规则；和 `WORKSPACE_V2_CONTRACT.md` 冲突时以本文件为准。

## 1. 总规则

1. **只在自己的工作副本和分支上改**，只提交到自己的分支。不推送、不合并、不 rebase 别的分支。
2. **不碰这些**：
   - 主目录 `/public/newhome/cy/Digital_twin/GNN`（线上服务直接读那里的文件）；
   - 线上服务 :8765 和预览服务 :8766，不启停、不 upgrade；
   - 别的路的工作副本；
   - `outputs/wss_deploy_release`、`outputs/wss_deploy_golden`（只读链接）。
3. **数字和数据必须与经典页面同源**：
   - 能调用经典页面的共享函数（`WssReportCommon`、`VolumeViewerCore`）就调用，不另写一套；
   - 写服务端时用经典页面同一个接口、同一种文档格式。
4. **界面风格**：延续现在的「影像工作站」样子。深色视口、浮动工具条、关键数字卡片、少字：说明放进 ⓘ 提示，不在界面上堆解释。用户明确说过「字太多、AI 味太浓」。
5. **离线报告**（`file://`）：新功能在离线时要么能用，要么不出现，不能报错。
6. **测试环境限制**：Node 测试用的是桩 DOM，没有 innerHTML 解析，也没有 querySelectorAll。界面一律用 `h()` 或 `createElementNS` 建。

## 2. 扩展点（尽量不改 `ws_shell.js`）

外壳已经留好注册表（见 `ws_shell.js` 的「extensions」一节）：

```js
(ns.ext = ns.ext || []).push({
  id: 'figure',
  toolbar(api)            // → [按钮]，放在左侧竖排工具条（光照按钮之前）
  tabs(api)               // → [{id, label}]，检查器页签
  renderTab(tabId, body, api)  // 渲染自己的页签，处理了返回 true
  layers(api)             // → [菜单项]，图层菜单末尾（自动加分隔线）
  userMenu(api)           // → [菜单项]，头像菜单「经典工作台」之前
  tools(api)              // → [section]，检查器「工具」页末尾
  key(event, api)         // 处理了返回 true（输入框、对话框里不会调用）
  page(name, container, api)   // #/name 独立页面（首页模式），处理了返回 true
  onResult(api), onClose(api), onField(api)
})
```

`api` 是外壳的助手函数（`shellApi()`）：
- 状态：`cur()`、`viewer()`、`state()`、`offline()`、`session()`、`jobs()`、`cards()`、`releases()`；
- 工具：`ui`、`api`（接口，含通用 `api().request(path, {method, body})` 和 `api().download`）、`store`、`h`；
- 重绘：`renderInspector`、`renderToolbar`、`renderHome`、`renderTop`；
- 页签与字段：`setTab`、`currentTab`、`applyField`、`updateColorbar`、`setLayers`；
- 三维：`flyTo`、`selectFinding`、`drawLabels`、`pickOnce(message, cb)` / `cancelPickOnce`；
- 请求与导航：`conflictOr`、`reloadCurrent`、`go`、`refreshJobs`、`scheduleRefresh`、`openExport`、`openSliceAt`、`saveViewSoon`；
- 其他：`hideName`、`fieldById`、`fieldName`、`visibleFields`、`isLocked`、`editable`、`withCenterline`、`copyText`、`fileBase`、`stageMessage`、`parseHash`、`buildHash`、`replaceRoute`。

图标用 `ns.icons.register(name, spec)` 加，格式同 `ws_icons.js`：`ns.icons.P(d)`、`C(cx, cy, r, fill)`、`R(x, y, w, h, rx)`。

## 3. 文件归属

| 路 | 分支 | 独占（随便改） |
|---|---|---|
| A 导出与出图 | `wss-ui-s5a-export` | `ws_export.js`、`ws_figure.js`、`ws_bookmarks.js`、`v2_export.css` |
| B 一页纸 | `wss-ui-s5b-onepage` | `onepager.py`（含它的 HTML、CSS）、`ws_onepage.js`、`v2_onepage.css` |
| C 工作台 | `wss-ui-s6c-workbench` | `ws_rail.js`、`ws_admin.js`、`ws_upload.js`、`ws_main.js`、`v2_workbench.css` |
| D 结果详情 | `wss-ui-s6d-detail` | `ws_overview.js`、`ws_lens.js`、`ws_detail.js`、`v2_detail.css` |
| E 显示选项 | `wss-ui-s6e-display` | `adapter_wall.js`、`adapter_volume.js`、`core_colormap.js`、`core_colorbar.js`、`ws_compare.js`、`ws_slice.js`、`ws_display.js`、`v2_display.css` |

**公共文件**（改了就要在报告里逐条列出，合并时由主会话处理冲突）：

- `ws_shell.js`：
  - 先用扩展点。确实要改时只做最小改动，不重排、不改名、不挪代码。
  - 需要新的助手函数时，加在 `SHELL_API` 对象里自己那一路的注释行后面（`// lane A` …）。
- `core_viewer.js`：只在文件里 `// ==== lane A ====`、`// ==== lane E ====` 两个标记块里加代码；要导出的方法写在 `viewer` 对象里对应的 `// lane A exports`、`// lane E exports` 注释之后。
- `bundle.json`、`index.html`：新脚本只能加在自己那路占位脚本的正下一行，两个文件同时加。样式只用自己那路的 css 文件，不改 `v2.css`。
- `ws_api.js`、`ws_icons.js`、`v2.css`、`WORKSPACE_V2_CONTRACT.md`、`README.md`：不改。接口直接用 `api().request`，图标用 `register`。文档写在自己的报告里，合并时由主会话汇总。
- Python 服务端（`server.py`、`jobs.py` 等）：
  - 尽量只用现有接口，已有的包括回收站、批量、导出、删除、重跑、出口改名重算、元数据、改口令、报告模板、配图、事件流。
  - 确实缺接口时先在报告里写明，改动要最小并配测试。
  - B 路改 `onepager.py` 以外的 Python 同样要列出。
- 测试：每路新建自己的测试文件（`tests/test_v2_<路名>.py`）。改已有测试要写明原因。

## 4. 环境与验证

- **工作副本**：`/public/newhome/cy/Digital_twin/GNN_lane_<a|b|c|d|e>`。里面的 `outputs/wss_deploy_preview_jobs` 是自己的一份可写副本，golden、release 是只读链接。
- **Python**：`/public/newhome/cy/.conda/envs/GNN/bin/python`，运行时带 `PYTHONPATH=.`。
- **自己的预览沙箱**（端口互不冲突）：
  - 端口：A 8821、B 8822、C 8823、D 8824、E 8825。Marionette 端口：A 3021、B 3022、C 3023、D 3024、E 3025，需要多个时往上加 10。
  - 启动：`python -m wss_deploy.devshot sandbox --source outputs/wss_deploy_preview_jobs --jobs <id,...> --dir <你的临时目录>/sandbox --port 88xx --login admin:<自己定的口令>`
  - 截图和查 JS 错误用 `wss_deploy.devshot.Browser`（可参考 `devshot.py` 里的 `suite`）。
  - 用完关掉。
- **可用病例**（都在 `outputs/wss_deploy_preview_jobs`）：

  | 编号 | 内容 |
  |---|---|
  | `20260929_230442_a7273f1670e4` | LV_GUO_YOU 周期指标（壁面，WSS / TAWSS / OSI / RRT / ECAP） |
  | `20260929_230450_d1428792b846` | LV_GUO_YOU 体场 |
  | `20260920_144135_673ccd0e36b1` | 峰值 WSS（X5D） |
  | `20260930_045134_6b8640d971b0` / `20260930_045135_c66b989ddd40` | 随访演示 A / B |
  | `20260930_122126_e434ae6d6df2` | 待确认出口 |
  | `20260930_045135_8412b6e4b1ac` | 失败 |

- **交付前必须做**：
  1. 所有改过的 js 都过 `node --check`。
  2. 跑全量 `pytest tests`，全部通过（有预期内的跳过可以）。
  3. 改了分析、流水线、叙述相关的 Python 时，再跑黄金回归：`python -m wss_deploy.regress --jobs-root outputs/wss_deploy_golden/20260920_baseline --out <临时目录> --device cuda`，用 `CUDA_VISIBLE_DEVICES=2`。
  4. 新功能每个都要在沙箱浏览器里走一遍，截图，确认 0 个 JS 错误。
  5. 涉及数字的，要和经典页面比对，至少一例逐项一致；不同的地方写明原因。
- **提交**：只提交到自己的分支，提交说明用英文，结尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。可以分多个提交。
- **报告**：写在 `wss_deploy/lanes/<路名>.md`（新文件，随提交），给用户看的部分用中文。内容包括：
  - 做了什么，入口在哪；
  - 与经典页面的一致性证据；
  - 与经典页面的差异和原因；
  - 改动过的公共文件清单；
  - 测试结果；
  - 截图路径（放在临时目录，写绝对路径）；
  - 没做完的项。

## 5. 各路范围（按优先级排；做不完的在报告里写明）

**A 导出与出图**
1. 出版级导图对话框：倍率 1× / 2× / 4×；背景可选视口、白、透明；标签语言中文或英文；可选带不带色条、标题、标签。在现有「汇报图」上扩展。
2. 六视角拼图（前、后、左、右、上、下，共用色条）；多视角拼图（自选视角，加当前视角，体场开着截面时加截面图）。用 `WssReportCommon.composeMontage`。
3. 色标 SVG 下载（`core_colorbar` 的 `toSVG`）。
4. 复现链接：链接能恢复字段、色标窗、相机、游标，以及体场截面的位置和状态。hash 里的新参数由自己在 `onResult` 读取，不改 `parseHash`。书签也带上截面位置。
5. 一页纸配图：新工作区生成配图，走 `POST /api/jobs/<id>/snapshots`，格式与经典页 `buildSnapshots` 相同，一页纸里要能显示出来。
6. 打印当前视图（打印友好的单页）。

**B 一页纸**
1. 服务端一页纸（`/api/jobs/<id>/onepage`）改成新界面的视觉语言。
   - 内容和 C 线口径一个都不能少：结论、关键数字、形态、发现（驳回的仍放附录、人工发现、规则更新前的判定）、签字栏、全部附录。
   - 要适合 A4 打印（打印样式浅色）。
2. 签字栏模板在新工作区里编辑：头像菜单 →「报告模板…」，接口 `GET/PUT /api/report-template`。
3. 对一页纸的文字、数字不做任何改动；已有的一页纸测试要全部通过。

**C 工作台**
1. 平铺任务列表：可以排序，按状态、患者、标签精确筛选，服务端搜索，分页（超过 100 个任务也能看全）。首页有计数（今日、进行中、待处理、失败）。「需要处理」里包含待复核。
2. 多选与批量：删除（进回收站）、重跑、导出汇总表（csv / xlsx）、打包 zip。
3. 回收站页 `#/trash`：恢复、彻底删除。
4. 队列总览页 `#/cohort`：已完成任务的关键数字表，可以排序，附一个指标的直方图。
5. 服务状态：头像菜单里看 `/api/health`。断线时提示正在重连；服务升级后提示刷新页面。
6. 改口令（头像菜单）。管理员可以切换「看全部用户」，列表显示属主。
7. 通知：任务完成时提示，病例栏标出未读。
8. 新建：
   - 整页拖放 STL；批量上传（`/api/jobs/batch`）。
   - 补上元数据字段（扫描标签、标签、备注、记住患者）。
   - 完整档可选设备、seed 数、线程。
9. 会话过期时原地重新登录，不丢对话框里的内容。

**D 结果详情与分析视图**
1. 分支展开图（弧长 s × 周向 θ），用预测点的 `p_s` / `p_th`，与经典页 `drawUnroll` 同口径。点一下飞到那里；可以导出 PNG。
2. 沿程曲线：`analysis.profiles` 的分支均值和 p99 随弧长变化；和游标联动；导出 SVG（`WssReportCommon.profileSVG`）。
3. 概览补全：
   - 逐分支表（`analysis.per_branch`）；
   - 周期量里 OSI、RRT、ECAP 的均值和占比；
   - 瘤体形态补上全腔体积、参考直径、可靠性说明；
   - 随访曲线。
4. 「工具」页：
   - 编辑信息，补上标签和备注；
   - 改出口命名后重算（`confirm {override: true}`）；
   - 删除任务，删除后进回收站；
   - 计算过程（阶段计时、事件日志）；
   - 已完成任务也能看输入检查。
5. 快捷键：N 新建，/ 搜索，O 一页纸（G 已被游标占用）。

**E 显示选项与比较**
1. 色标：色带分段、自定阈值、「设为默认」（按字段记住）。
2. 图层：显示输入 STL；全场最大值 / TAWSS 最小值的标记常驻；滞留区斜纹；复位视角按钮（加 0 键）。
3. 体场：外壁透明度、流线密度与粗细、体内点速度矢量箭头；单位切换（Pa / mmHg、m/s / cm/s），色条、读数、截面页都要跟着换。
4. 截面工具：每条分支 6 个预设站位（经典页 `automaticPlanes`）。
5. 比较：以左为准、以右为准；同步口径时一并同步分段和阈值。

## 6. 合并记录（2026-09-30 晚，主会话）

五路都已合回 `wss-ui-v2`，每路都是干净合并，没有冲突。每合一路跑一次全量测试（测试机上有开发任务副本，所以跳过项比各路自测少）。

| 顺序 | 路 | 合并提交 | 各路报告 | 合并后全量测试 |
|---|---|---|---|---|
| 0 | 合并前基线 | 5628034 | — | 908 通过 / 4 跳过 |
| 1 | B 一页纸 | 35e2344 | `lanes/lane_b_onepage.md` | （B 合并后即 5628034 的基线） |
| 2 | E 显示选项与比较 | 8f1a453 | `lanes/lane_e_display.md` | 920 / 4 |
| 3 | D 结果详情 | 89d55aa | `lanes/lane_d_detail.md` | 931 / 4 |
| 4 | A 导出与出图 | f7ebba0 | `lanes/lane_a_export.md` | 946 / 4 |
| 5 | C 工作台 | 6b0814a | `lanes/lane_c_workbench.md` | 957 / 4 |

**合并后衔接修改**（各路报告里点名要主会话处理的，加上浏览器走查发现的）：

1. **体场单位切换覆盖全部数字**：E 路的 Pa / mmHg、m/s / cm/s 原来只管色条、读数页和截面页。新增 `ws_display.toDisplay(value, units, family)`，概览关键数字、体场发现、证据透镜、「沿程」曲线、工具条色标窗名称和比较时的共用范围都按显示单位写；壁面剪切力始终是 Pa；传给透镜的引用和导出数据仍是原始单位。改单位后重画工具条、自动标签，以及概览 / 沿程 / 读数 / 截面页。
2. **截面打开时的阅读位置**：截面打开时外壳只在视口里隐藏体内点。原来会把「体内点：关」存进阅读位置，下次打开就不显示体内点；现在存的是用户自己的选择。
3. **长菜单**：窗口较矮时，图层菜单会超出屏幕。现在菜单向上挪，超过窗口高度时在菜单内滚动（`ws_ui.menu`）。
4. **工具页文案**：「这些工具还没有搬到新工作区：六视角、出版级导图、分支展开图」已不成立，改为「经典报告」一节，只留「在经典报告中打开」按钮用于对照；「导出」说明改为四页的内容。
5. 新测试 `tests/test_v2_phase2_merge.py`（单位跨路生效、壁面不换算、没有显示模块时数字不变）。合并加衔接修改后全量测试 959 通过 / 4 跳过。

**浏览器走查**（沙箱，预览任务副本，无头 Firefox）：首页四页签、任务列表、队列、回收站、壁面结果概览 / 沿程 / 工具页、导出对话框、色标面板、头像菜单（B、C 两路加的项之间没有重复分隔线）、体场截面、单位切换、矮窗口图层菜单，共 14 步，JS 错误 0。

**黄金回归没有重跑**：五路合并加衔接修改，非测试 Python 只改了 `server.py` 的任务列表接口（管理员看全部用户时加 `owner_name`），分析、流水线和叙述代码没有变化；GPU 当时全被训练占满，所以没有加跑。

**留给后面的**：

- 导出对话框开着时，如果用浏览器后退切换了结果，对话框里的预览仍是原来那个结果（只在后退、前进时出现）。
- D 路：「沿程」页签会记进偏好（与「书签」「工具」相同）；分支显隐不影响「沿程」的分支下拉；展开图 θ = 0 的参考方向在 manifest 里没有说明。
- E 路没做：阈值等值线、「设为默认」写到服务端、经典「预设」卡（用书签代替）。
- C 路没迁：任务行「⋯」快捷菜单、病例卡「补跑缺少的结果」、跨病例批量出图、「近 7 天完成」快捷筛选。`static/v2/example_aaa.stl` 被 `.gitignore` 排除，分支上「下载示例 STL」是 404，要在服务机上用 `python -m wss_deploy.v2_examples` 生成。
- A 路：拼图不能逐面板自由选视角；打印版式只在无头浏览器里核对过，真实打印 / 另存 PDF 要在桌面浏览器确认一次。
- B 路：Firefox 对 `break-after: avoid` 支持有限，附录小标题偶尔会落在页尾。
- 预览任务 `20260920_144135_673ccd0e36b1` 的阶段 A 记录里，输入 STL 是别的会话临时目录的绝对路径，在它上面做「改出口命名重算」或「换模型重跑」会在阶段 B 失败。这是数据副本的问题，正式任务不受影响。
