# 第三期并行开发合同（补齐迁移 + S7 旧页面下线，六路）

2026-09-30 晚起。用户要求：先把旧版本里还没迁到新工作区的功能补齐，再做 S7（旧页面下线）。

六路从 `wss-ui-v2` 的同一个基线提交分出，各在自己的工作副本和分支上开发，最后由主会话逐路合并。第二期的 `PHASE2_LANES.md` §1 总规则、§2 扩展点照样适用；和本文件冲突时以本文件为准。

**依据**是三份只读审计，都在 `wss_deploy/lanes/`：

| 文件 | 内容 | 编号 |
|---|---|---|
| `p3_audit_workbench.md` | 经典工作台和比较页，114 项 | `#` |
| `p3_audit_reports.md` | 经典壁面、体场报告和一页纸 | `W`、`V`、`C` |
| `p3_audit_ops_s7.md` | 运维中心、工单、S7 要处理的链接和文件、加载速度 | — |

下文括号里的编号就是审计表里的行号，先读对应行再动手。

## 0. 已经定下的事（不用再问）

1. **放弃，不迁移**：
   - **认领旧会话任务、迁移期旧令牌登录**：正式服务只有 `admin` 一个账号，两个任务都属于它，没有可认领的。
   - **经典专用接口**：`GET /api/cases`、单任务 SSE、`/api/trash?all=1`。
   - **比较页左右两份完整报告**：新工作区的比较保持现在的简化做法。
   - **离散度数值**：旧裁定「不展示离散度」不变。
2. **运维中心 `/ops` 和工单 `/support`**：
   - 保持独立页面，不迁进新工作区，继续用 `static/app.css`（所以 app.css 不删）。
   - 新工作区里加入口：头像菜单给管理员加「运维中心」，帮助菜单加「反馈问题」（→ `/support`）。两页的「返回工作台」改指 `/v2/`。
3. **经典 `report.html` 继续生成**：它是新工作区的数据来源（`v2_data.py` 从它读数据）。S7 不再把它当页面提供；下载的数据包改放新版单文件离线页。
4. **数字同源**：新功能必须调用经典页同一个接口、同一种文档格式、同一组共享函数（`WssReportCommon`、`VolumeViewerCore`）。
5. **界面**：
   - 延续现在的影像工作站样子，少字，说明放进 ⓘ。
   - 做图表先按数据可视化规范：蓝色 `#2a78d6`、灰色 `#8a94a6`、参照橙色 `#eb6834`，这三个颜色已校验；状态色只用于状态。
   - 悬停提示用 `ns.ui.chartTip`。
6. **不碰**：
   - 线上服务 :8765 和主目录 `/public/newhome/cy/Digital_twin/GNN`；
   - 预览服务 :8766；
   - 别的路的工作副本；
   - `outputs/wss_deploy_release`、`outputs/wss_deploy_golden`（只读）。

## 1. 分工

| 路 | 分支 | 工作副本 | 沙箱端口 / Marionette |
|---|---|---|---|
| 1 输入与未完成任务 | `wss-ui-p3-input` | `GNN_lane_a` | 8821 / 3021 |
| 2 工作台 | `wss-ui-p3-workbench` | `GNN_lane_b` | 8822 / 3022 |
| 3 结果信息与安全 | `wss-ui-p3-info` | `GNN_lane_c` | 8823 / 3023 |
| 4 显示与读数 | `wss-ui-p3-display` | `GNN_lane_d` | 8824 / 3024 |
| 5 导出、链接与比较 | `wss-ui-p3-export` | `GNN_lane_e` | 8825 / 3025 |
| 6 S7 旧页面下线 | `wss-ui-p3-s7` | `GNN_lane_f` | 8826 / 3026 |

工作副本都在 `/public/newhome/cy/Digital_twin/` 下。

**合并顺序**：1 → 3 → 4 → 5 → 2 → 6。S7 最后合，因为它要删文件，还依赖第 5 路的批量出图。

## 2. 文件归属

| 路 | 独占（随便改） |
|---|---|
| 1 | `ws_input.js`、`v2_input.css` |
| 2 | `ws_admin.js`、`ws_rail.js`、`ws_upload.js`、`ws_thumbs.js`、`v2_workbench.css`；`ws_shell.js` 里登录页 `showLogin` 这一个函数 |
| 3 | `ws_notice.js`、`ws_overview.js`、`ws_lens.js`、`ws_detail.js`、`ws_profile.js`、`ws_unroll.js`、`ws_review.js`、`v2_detail.css` |
| 4 | `core_contour.js`（在 adapter_volume.js 之后加载）、`adapter_wall.js`、`adapter_volume.js`、`core_colormap.js`、`core_colorbar.js`、`ws_display.js`、`ws_slice.js`、`ws_probe.js`、`ws_measure.js`、`ws_region.js`、`ws_annot.js`、`v2_display.css`；`core_viewer.js` 的 `// ==== P3 lane 4 ====` 标记块和 `// P3 lane 4 exports` |
| 5 | `ws_export.js`、`ws_figure.js`、`ws_bookmarks.js`、`ws_compare.js`、`ws_batch.js`、`v2_export.css` |
| 6 | `ws_legacy.js`、`ws_main.js`、`ws_api.js`、`static/v2/index.html`（noscript 与资源版本号）、`static/v2/help_*.html`、经典静态文件（`static/index.html`、`app.js`、`compare.*`、`batch_export.js`、`workbench_core.js`）；Python：`server.py`、`cli.py`、`service.py`、`bundle.py`、`rehearse.py`、`devshot.py`、`report.py`、`volume_report.py`；`onepager.py`、`morphology.py`、`glossary.py` 只改文案；经典页相关测试 |

**公共文件规则**：

- **`ws_shell.js`**：
  - 先用扩展点。确实要改时只做最小改动，不重排、不改名、不挪代码。
  - 新助手函数加在 `SHELL_API` 里自己那路的注释行后面（`// P3 lane N …`）。
  - 允许的改动：
    - 第 1 路：去掉未完成任务工具栏里「在经典工作台中处理」。
    - 第 2 路：只改 `showLogin`。
    - 第 6 路：头像菜单、工具页「经典报告」一节、加载失败兜底、快捷键说明。
  - 每处都在报告里列出。
- **`bundle.json`、`index.html`**：
  - 基线已为新文件放好占位：`ws_notice.js`、`core_contour.js`、`ws_batch.js`、`ws_legacy.js`、`v2_input.css`。
  - 还要新文件时，加在自己占位的正下一行，两个文件同时改。
  - 第 6 路给资源加版本号时统一改写两处，其他路不动这两个文件的已有行。
- **结果页顶部提示**：基线在 `ws_shell.js` 加了 `S.els.notices` 容器和 `shellApi.noticeHost()`，归第 3 路使用。
- **跨路的调用**，用「有就调用」的写法（`ns.x && ns.x.fn`），另一路没合并时也不报错：
  - 第 2 路在任务列表多选条里加「批量出图」按钮，调用 `ns.batch.open(jobIds, shellApi)`；由第 5 路实现。
  - 第 3 路的「压降发现定位截面」「沿程曲线点一下截面跟随」用现有的 `shellApi.openSliceAt`；需要新参数时在报告里写明。
- **Python 服务端**：
  - 只有第 6 路改 `server.py`。
  - 其他路缺接口时先在报告里写明；改动要最小，并配测试。
- **测试**：
  - 每路新建自己的测试文件，命名为 `tests/test_v2_p3_<路名>.py`。
  - 改已有测试要写明原因。
  - 第 6 路负责删改经典页面测试：`test_workbench_js.py`、`test_compare_page.py` 删掉；`test_v2_lane_c.py` 先把 `workbench_core.js` 的对照值固化进测试，再删这个文件；其余见 `p3_audit_ops_s7.md` 第 2 部分。

## 3. 各路范围（按优先级；做不完的在报告里写明）

### 第 1 路 输入与未完成任务

1. **失败任务可以重试**（#69）：
   - `failed` 也显示「重试」；服务端已经允许，载荷同经典 `app.js`。
   - 错误卡补齐（#70）：错误类别说明、「改用 CPU 重试」提示、诊断编号、管理员可见的技术细节 `admin_detail`。
2. **待确认的任务可以取消**（#68）：`awaiting_input`、`awaiting_confirmation`、`awaiting_outlets` 三种状态。
3. **重新选择入口、重提中心线**（#74）：`/confirm` 带 `inlet`，交互对照经典 `app.js` 的 `outletCard`；三维里点选入口开口。
4. **补齐几张表和提示**：
   - 进度的阶段表（#65）；
   - 输入确认的事实表（#72）；
   - 出口确认的细节（#75）；
   - 确认或签字之后提示「下一例」（#76）；
   - 同时预测 / 来源说明（#77）。
5. 这几项做完后，去掉工具栏里「在经典工作台中处理」。

### 第 2 路 工作台

1. **队列页筛选**：复核状态加「已重新打开」（#60）；按数值上下限筛选，放在筛选行（#61）。
2. **病例卡**：
   - 「补跑缺少的结果」（#34）：比对发布包列表和这个病例已有的结果，调用同经典的接口；
   - 卡片上显示管腔最大直径和标签（#37、#38）。
3. **任务列表**：
   - 每行加「⋯」快捷菜单（#36）：一页纸、编辑信息、患者时间线、删除；
   - 列表行显示实时剩余时间（#30）；
   - 多选条加「批量出图」，调用 `ns.batch.open`（见 §2）。
4. **首页**：
   - 「最近完成」（#47）；
   - 「三步上手」（#17）：空库或新用户时自动展开。
5. **登录**：
   - 显示 / 隐藏口令、大写锁定提示、记住用户名（#3–#5）；
   - 服务返回 503 时给「维护中」提示；
   - 登录页脚去掉「经典工作台」。
6. **加载速度**：首页缩略图持久缓存（IndexedDB，按任务号 + 运行身份失效；存不进去时照常工作）。

### 第 3 路 结果信息与安全

1. **结果页顶部警示条**（W65、W29、V45，`ws_notice.js`）：
   - 触发条件：几何超出参考范围、人群需复核、集成质量不是 good。数据来自 `analysis.reference_assessment` 和 `analysis.quality`。
   - 文案口径同经典页。
   - 离线也显示。
   - 不给离散度数值。
2. **概览**：
   - 集成一致性卡（W29）；
   - 几何参数表与 Murray 分流（W36、W37）。
3. **技术信息**（W62）：
   - 补特征合同、模型帧、周期定义、时间戳；
   - 基本档和离线报告里也能看；
   - 补 `pressure_reference` 说明和流线说明文字。
4. **随访**（#103）：
   - 瘤体体积增长率、最近两次之间的增长率、其他发布包的曲线；
   - 管理员的时间线只在「看全部用户」开着时才请求全部用户的数据。
5. **联动截面**：
   - 点压降发现，截面定位到 10% 处（V15）；
   - 在沿程曲线上点一下，截面跟着移过去。
6. **术语提示**：改读 `/static/glossary.json`（W63、V38）。
7. **复核签字**：签字对话框补上核对清单事实（#82），归 `ws_review.js`。

### 第 4 路 显示与读数

1. **悬停读数**（W41、V17）：查看器已经发出 `hover` 事件，给一个轻量读数。
2. **色标与单位**：
   - WSS 可切换 dyn/cm²（W8），接在第二期 E 路的单位切换上；
   - 速度对数色标（V7）；
   - 蓝白红色表（W4）；
   - 可以手输固定上限（W6）。
3. **等值线**（W10）：放在 `core_contour.js`，移植 `report.py` 里 `REPORT_CORE_JS` 的 `marchingTriangles`。
4. **高亮**：最高的 x%（W16）。
5. **剖切**：壁面结果沿 Z 轴剖切（W17）。
6. **自动标注条数** 0 / 3 / 5 / 10（W20，`ws_annot.js`）；同时把 `ws_annot.js` 第 164 行提到经典报告的文字改掉。
7. **截面工具**：
   - 按 X / Y / Z 轴放截面（V26）；
   - 倾角、偏移可以输数值（V27、V28）；
   - 拖动模式工具条，触屏也能用。
8. **可信度**：体内不可信的点变灰，加可信度图例（V12、W11）。
9. **服务端已存的预设与默认口径**（W22、W23、V44）：`/api/preferences` 里的 `presets.*`、`report_defaults.*` 读进新工作区。

### 第 5 路 导出、链接与比较

1. **复现链接**补上显示状态：配色、分段、阈值、单位、标签（W55）。
   - 旧格式（`v:1`）的链接必须照样能打开。
   - 把状态格式写进报告，第 6 路据此写经典链接的转换。
2. **下载修正**（#93）：
   - 体场结果没有 VTP 链接，因为 `summary.exports` 里是布尔值，不是文件名；
   - 补单独下载：点云 CSV、体场 VTP、壁面压力 VTP、流线 VTP。
3. **批量出图**（#56，`ws_batch.js`）：
   - 在新查看器上重写经典的 `batch_export.js`，入口是 `ns.batch.open(jobIds, shellApi)`；
   - 样式跟「导出 → 图片」一致；
   - 逐个结果离屏出图，打成 zip。
4. **六视角**：可以选分成 6 个文件导出。
5. **比较**：
   - 标量对照表（C5）：`POST /api/compare` 的左、右、差值；
   - 结果选择框放开 20 条的限制，加搜索（#98）；
   - 显示两侧的身份（#112）。

### 第 6 路 S7 旧页面下线

1. **路由**（`server.py`）：

   | 旧地址 | 处理 |
   |---|---|
   | `/` | 302 到 `/v2/`（浏览器会带上原来的 `#` 片段） |
   | `/compare?left=A&right=B` | 302 到 `/v2/#/job/A?v=compare&cmp=B` |
   | `/api/jobs/<id>/report`、`/jobs/<id>/report.html` | 顶层页面访问时 302 到 `/v2/?job=<id>` |
   | `/api/jobs/<id>/files/report.html` | 同上，只在 `Sec-Fetch-Dest: document` 时跳转；其他请求照常 |
   | `/ops`、`/support` | 保留 |

   同步更新静态白名单。
2. **`ws_legacy.js`**：启动时把 `?job=<id>`、`#job=<id>`、经典 `#view=`（`wss-deploy.view/v1`）转成新工作区的路由和状态。
   - 对照表见 `p3_audit_reports.md` §F。
   - 转不过去的设置用提示条说明。
3. **删掉经典工作台和比较页**：`static/index.html`、`app.js`、`compare.html`、`compare.js`、`batch_export.js`、`workbench_core.js`。
   - 保留 `app.css`（运维中心和工单页用）、`three.min.js`、`OrbitControls.js`、`report_common.js`、`volume_viewer.js`、`glossary.json`。
   - 注意：改这些保留文件会让所有报告被判过期，所以一个字也别动。
4. **去掉新工作区里指向经典页的链接**：
   - 头像菜单「经典工作台」→ 管理员「运维中心」，另加「反馈问题」；
   - 工具页的「经典报告」一节；
   - 结果加载失败时的兜底：改成说明、重试、下载数据包；
   - `ws_main.js` 的出错页、`index.html` 的 noscript、`ws_api` 的 `urls.classic` 和 `urls.report`。
   - 登录页脚归第 2 路，工具栏那条归第 1 路，第 6 路不动。
5. **文案**：
   - `cli.py` 第 313 行打印的链接；
   - `service.py` 打印的地址（改成 `/v2/`）；
   - `bundle.py` 的数据包说明；
   - `onepager.py` 第 40 行、`morphology.py`、`glossary.py` 里的「三维报告」；
   - `help_quickstart.html`；
   - 快捷键说明里写明改过的键：G、O、R。
6. **数据包**：`bundle.zip` 改放新版单文件离线页（`v2_offline.build_offline_html`），代替经典 `report.html` 作为可离线打开的文件。
7. **工具**：
   - `devshot.py`：登录改用新工作区表单（`wsl-user` / `wsl-pass`），更新里面的地址；
   - `rehearse.py`：冒烟检查改到新地址。
8. **测试**：见 §2 的删改清单。
9. **加载速度**：
   - 给 `index.html` 里的脚本和样式加 `?v=<构建号>`，带版本号的请求发 `Cache-Control: immutable`；`index.html` 本身照旧每次验证。
   - 可选：体场结果才加载 `volume_viewer.js`，但不许改这个文件。
10. **数据壳**（可选，最后做）：`report.py`、`volume_report.py` 只输出数据外壳，删掉经典查看器的页面代码。
    - `v2_data` 的读取不变；已有报告由 `report_freshness` 批量换壳；删改 `test_report` / `test_volume_report` 里的页面钩子测试。
    - 风险大或时间不够就不做，在报告里写明。

## 4. 环境与验证

与第二期 §4 相同，另外：

- **工作副本**：`outputs/wss_deploy_preview_jobs` 是自己的一份可写副本（基线时从 `GNN_wssui_v2` 重新拷过）。
- **可用病例**：见第二期 §4 的表。
- **交付前必须做**：
  1. 改过的 js 都过 `node --check`。
  2. 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests` 通过。
     - 只停自己启动的进程，用进程号，不用 `pkill -f` 按名字杀。
  3. 改了分析、流水线、叙述或报告生成的 Python（只有第 6 路可能），跑黄金回归：`python -m wss_deploy.regress --jobs-root outputs/wss_deploy_golden/20260920_baseline --out <临时目录> --device cuda`。
     - GPU 被占满时可以用 `--device cpu`，并在报告里写明。
  4. 每个新功能都在沙箱浏览器里走一遍，截图，确认 0 个 JS 错误。
  5. 涉及数字的，至少一例与经典页面逐项比对。
- **提交**：只提交到自己的分支，提交说明用英文，结尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
- **报告**：`wss_deploy/lanes/p3_<路名>.md`（随提交）。内容同第二期：
  - 做了什么、入口在哪；
  - 一致性证据；
  - 与经典的差异；
  - 改动过的公共文件；
  - 测试结果；
  - 截图的绝对路径；
  - 没做完的项。
