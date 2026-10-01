# 经典结果页（壁面 / 体场 / 一页纸 / 并排比较）→ v2 功能核对（S7 退役前）

- 核对对象：分支 `wss-ui-v2`，工作副本 `/public/newhome/cy/Digital_twin/GNN_wssui_v2`（HEAD 6e45be1），只读代码，没有点界面。
- 经典侧：`wss_deploy/report.py`（生成的 report.html 模板 + 内联页面代码，逐个 id / 函数走过）、`wss_deploy/volume_report.py` + `static/volume_viewer.js` 第 1024–3792 行页面代码、`onepager.py`、`static/compare.js`。
- v2 侧：`static/v2/*.js`，以及 `wss_deploy/lanes/*.md`、`PHASE2_LANES.md` 的「没做」清单。
- 状态：HAVE = v2 有同等功能；PARTIAL = 有简化版（写明缺什么）；MISSING = 没有；DROP-CANDIDATE = 建议裁定放弃（写明理由）。工作量 S ≈ 半天内，M ≈ 1–2 天，L ≈ 更多。
- 「未核实」= 只读代码没能确定，需要在浏览器里点一下。

## 0. 汇总

| 部分 | 行数 | HAVE | PARTIAL | MISSING | DROP-CANDIDATE |
|---|---|---|---|---|---|
| A 壁面报告（report.py） | 72 | 27 | 29 | 8 | 8 |
| B 体场报告（volume_report.py + volume_viewer.js 页面代码） | 53 | 27 | 18 | 4 | 4 |
| C 一页纸（onepager.py） | 5 | 3 | 2 | 0 | 0 |
| D 并排比较（compare.html / compare.js） | 8 | 6 | 1 | 1 | 0 |
| 合计 | 138 | 63 | 50 | 13 | 12 |

另有 §E「S7 本身的阻塞项」（不是用户功能，而是删页面会连带坏掉的东西），见文末。

与迁移矩阵（`前端重构_迁移矩阵_2026-09-30.md`）的主要出入：

1. 矩阵 A.2「配色 已有」把几件事合在一行：蓝白红色表、手输固定上限、dyn/cm² 单位、阈值等值线其实都没有（等值线矩阵备注里写了，但状态仍记「已有」）。
2. 矩阵 A.2「统计与口径 已有」：五模型一致性卡、面积加权参考卡、几何参数表、出口 Murray 分流、顶部警示条都没有迁（manifest 里 `analysis.quality`、`analysis.surface_statistics`、`analysis.branch_geometry` 已经带着，JS 没读）。
3. 矩阵 A.2「复现链接 已有」：v2 链接只带字段、色标窗、相机、游标、选中点、分支、图层、截面；不带配色、分段、自定阈值、单位、标签设置、测量、探针记录；而且读不了经典 `#view=` 链接。应为 PARTIAL。
4. 矩阵 A.2「技术信息 已有」：v2 只在「工具」页、只在完整档、只在线；缺特征合同、模型族 / 模型数、模型帧、周期定义、生成 / 重建时间、摘要版本。应为 PARTIAL。
5. 矩阵 A.2「打印 PDF 已有」「六视角 已有」：打印只印当前视图（经典还印统计卡）；六视角出的是一张拼图，经典是 6 个 PNG 文件。应为 PARTIAL。
6. 矩阵完全没列的：悬停读数、剖切高度、壁面不透明度、高亮最高 x%、点云特征着色、顶部警示条、术语「?」弹窗（glossary.json）、语言切换、键位差异、分支下拉「淡化其余」、父页面消息协议、离线单文件 / bundle.zip、服务端已存的预设与默认口径；体场的速度对数色标、X/Y/Z 轴截面、倾角 / 偏移的精确滑杆与拖动模式工具条、体内不可信点灰化与可信图例、压降发现定位截面、沿程点击跳截面、pressure_reference 与流线提醒文字。
7. 矩阵 A.5「标量差值 没有（裁定不做逐点差值）」：经典 compare 的表不是逐点差值，而是 `/api/compare` 返回的「声明一致的标量统计」左 / 右 / 差值表。裁定的是逐点差值，这张标量表是否也放弃需要单独确认。

---

## A. 壁面报告（report.py 生成的 report.html）

| # | 功能 | 旧入口 | v2 状态 | 证据或缺什么 | 建议 | 工作量 |
|---|---|---|---|---|---|---|
| W1 | 显示对象：WSS 壁面 / 输入 STL / 中心线 / 预测点云 | 头部 `mode-tabs`（`tab-wss/stl/cl/cloud`，`setView`） | HAVE | 图层菜单 `ws_shell.layersMenu`（中心线、预测点）+ `ws_display.layerItems`「输入 STL」 | — | — |
| W2 | 中心线模式细节：中心线按半径着色 + 色条「中心线半径 · mm」、入口 / 出口小球与「名称 r=… mm」标签 | `setView('cl')`、`labels` kind `endpoint` | PARTIAL | v2 中心线是单色线（`adapter_wall` clObj）；出口标签只在「工具 → 出口命名」或「输入检查 → 在三维标出开口」时出现（`ws_detail` setToolLabels）；无半径着色 | 裁定：半径着色可放弃；出口标签够用 | S |
| W3 | 字段切换（视口上方段控件、菜单字段页签） | `field-seg`、`field-tabs`、`setField` | HAVE | `ws_shell.renderToolbar` seg-fields + 数字键 1–9 | — | — |
| W4 | 色表：彩虹 / Viridis / Turbo / 蓝白红，本机记住 | `#cmap`，localStorage `wss-report-cmap` | PARTIAL | 图层菜单「色表」只有彩虹、viridis、turbo（`ws_shell.layersMenu`）；无蓝白红（`core_colormap` 里有 bwr，只给截面穿面速度用） | 加一项即可，或裁定放弃 | S |
| W5 | 色带分段 0/4/6/8/10/12/16/20 | `#bands` | HAVE | `ws_display.buildScale`（按字段记） | — | — |
| W6 | 色标范围：本例 p99 / 固定上限（手输，跨报告记住） | `#scale-mode`、`#fixedmax`，localStorage `wss-report-scale-v1` | PARTIAL | v2 只有「本例自适应」和模型说明里的命名窗（`windowOptions`，来自 card `display_windows`）；不能手输上限，也不跨报告保留 | 在色标面板加「自定范围…」（viewer 已支持 `{range:[lo,hi]}` 窗） | M |
| W7 | 对数色标 | `#logscale` | HAVE | 「本例自适应（对数 / 线性）」`windowOptions` | — | — |
| W8 | 单位 Pa / dyn/cm² | `#units`（`CORE.UNITS`） | MISSING | `ws_display.unitKind`：壁面一律 Pa；lane E §3.7 明确没做 | dyn/cm² 在文献里常用，建议补（色条、探针、分支表、导出一起换） | S |
| W9 | 三级阈值手改，统计卡 / 分支表 / 快速统计 / 沿程虚线 / 周期卡随之重算 | `#thr0–2`、`applyThresholds` | PARTIAL | `ws_display.buildScale` 只改色条粗线和面板里的占比一行；概览、分支表、导出按发布包阈值（lane E 裁定「数字不变」） | 保持裁定，写进说明 | — |
| W10 | 等值线（三条阈值） | `#contours`、`rebuildContours`（`WssReportCore.marchingTriangles`） | MISSING | lane E「没做」第 3 条；`marchingTriangles` 只在 report.py 的 `REPORT_CORE_JS` 里，S7 删页面代码时会一起没了 | 要迁就先把 `marchingTriangles` 搬进 report_common.js，再在 viewer 加线层 | M |
| W11 | 可信区域叠加 + 屏上图例（插值无支撑 / 表面粗糙 / 几何越界各自占比与规则） | `#trust`、`renderTrustLegend` | PARTIAL | 图层「可信度标记（斜纹）」HAVE；全局图例（三类占比 + 规则）没有，只能在证据透镜里逐点看（`ws_lens.trustRows`） | 在色条说明行加一行占比 | S |
| W12 | 滞留区叠加 + 图例 | `#stag`、`renderLegendNote` | HAVE | `ws_display` 图层「滞留区斜纹」+ 概览 KPI「滞留区」 | — | — |
| W13 | 全场最大值标记（TAWSS 另标最小值）+ 位置说明（分支、距入口、距分叉、局部半径） | `#showpeak`、`placePeak`、统计卡「最大值位置」 | PARTIAL | 标记 HAVE（`ws_display.peakMarkers`，默认常驻）；位置文字在 v2 里没有 | 标记可点开探针即可，或在透镜里补一行 | S |
| W14 | 「分支」下拉：淡化其余分支（着色、最高 x% 只取本支、展开图只画本支） | `#branch`、`branchSel` | PARTIAL | v2 只有分支显隐（隐藏，不是淡化）；概览分区悬停会标出分支 | 可裁定用显隐代替 | S |
| W15 | 分支显隐（只改显示） | `#branch-vis`、`setBranchHidden` | HAVE | `ws_shell.branchDialog` | — | — |
| W16 | 高亮最高 x%（0.5–10% 滑块，本机记住） | `#highlight-pct`、`#showtop`、`rebuildTop` | MISSING | v2 全局搜不到对应代码 | 可裁定放弃（发现列表的高 WSS 簇 + 阈值色条已覆盖），否则补一个图层 | S |
| W17 | 剖切高度（沿 STL Z 轴只留切面以下） | `#clip`、`clipPlane` | MISSING | `core_viewer.setClipPlane` 只被体场截面用（`ws_slice`） | 壁面看髂分叉内侧有用；给壁面结果开一个剖切图层 | S |
| W18 | 壁面不透明度 0.2–1 | `#opacity` | DROP-CANDIDATE | v2 只有体场「外壁不透明度」；v2 中心线 `depthTest:false` 穿透显示，经典调透明主要就是为看中心线 | 裁定放弃 | — |
| W19 | 点云特征着色（沿程弧长 / 周向角 / 半径 / 到分叉距离） | `#feat`、`feature()` | DROP-CANDIDATE | v2 无；属调试用途，展开图已给出 s–θ | 裁定放弃 | — |
| W20 | 自动标注：发现前 N（关 / 3 / 5 / 10）、分支名、管腔最大直径环 | `#lbl-findings`、`#lbl-branches`、`#maxd-ring` | PARTIAL | 图层菜单只有「发现标签（前 5 条）」开关（`ws_shell.layersMenu` → `setLabelPref`）；分支名、直径环 HAVE（`ws_annot`） | 菜单改成 0/3/5/10 | S |
| W21 | 标准视角 6 个 + 复位（撑满视口） | `#std-views`、`#reset-camera`、`goView` | HAVE | 方位立方体（`core_orientation`）+ 工具条复位（`ws_display.resetView`） | 键位见 W67 | — |
| W22 | 预设：内置 4 个（瘤囊低 WSS 区 / 髂分叉热点 / 主动脉沿程 / 临床视图）；用户预设存服务端偏好，可改名、导出 JSON | 「预设」卡、`COMMON.builtinPresets('wall')`、`/api/preferences` presets.wall | PARTIAL | 「按问题看」3 个（`ws_questions`；「低值区域在哪里」≈瘤囊低 WSS 区）；用户预设由书签代替（`ws_bookmarks`，只存本机）；服务端已存的 presets.wall 没有任何 v2 读者 | 至少给已存预设做一次导入（或在 S7 前通知用户导出） | M |
| W23 | 设为默认口径（配色、分段、单位、阈值、透明度、对数、语言；本机 + 服务端 report_defaults.wall） | `#set-defaults` | PARTIAL | `ws_display.setDefault` 只按字段存分段和阈值，只存本机（lane E 没做服务端） | 按裁定可接受；服务端旧值无人读 | M |
| W24 | 快速统计一行（当前字段 p99 / 阈值占比） | `#quick-stats` | HAVE | 概览 KPI（`ws_overview.kpiModel`） | — | — |
| W25 | 结论（只读，审阅人改写优先） | 统计卡 `narrative-card` | HAVE | `ws_overview` 结论区（可编辑） | — | — |
| W26 | 结果字段与模型卡（声明字段表 单位/位置/类型、时间轴、发布包、集成协议、权重数） | `renderPanel` 第二张卡 | PARTIAL | 概览页脚「模型说明」对话框（model card）；声明字段表、集成协议、权重数没有 | 可放进技术信息 | S |
| W27 | 峰值帧 p99 卡：p99、全场最大值、最大值位置 | `renderPanel` p99 卡 | PARTIAL | 纯 WSS 结果有 KPI「WSS p99」；三头结果（有 TAWSS）概览只给 TAWSS KPI（`kpiModel` 分支），峰值 WSS 的 p99 / 最大只在分区表和透镜里；最大值位置文字没有 | 三头结果切到 WSS 时 KPI 跟字段 | S |
| W28 | 壁面面积加权参考（面积加权 p99、最高 1% 面积均值、有效覆盖面积） | `renderPanel`（`surface_statistics`） | DROP-CANDIDATE | manifest `analysis.surface_statistics` 已带，JS 未用；经典自己注明「主指标仍是预测点云 p99」 | 裁定放弃，或放进透镜 | — |
| W29 | 五模型一致性卡（quality.level / label / reasons，「一致不代表准确」） | `renderPanel` `quality-card` | MISSING | manifest `analysis.quality` 已带，v2 JS 没有读它（`ws_lens` 的「一致性」是留出集 CFD 一致性，不是本例集成质量） | 必须迁：属安全提示，与 W65 一起做 | S |
| W30 | 出口命名门控卡（需人工确认 / 已通过；校准状态） | `renderPanel`（`proposal_confidence_gate`） | PARTIAL | 「工具 → 出口命名」有来源与把握度（`ws_detail`）；「代理值未作为概率」说明没有 | 可接受 | S |
| W31 | 参考范围与人群位置卡 | `renderPanel` `ref-card` | PARTIAL | 只在证据透镜里（`ws_lens` geoRef / `populationRow`），要点开某个数才看到 | 由 W65 警示条兜底 | S |
| W32 | 点数占比与估计面积（低 / 高 / 极高 + 均值 / 中位） | `renderPanel` | PARTIAL | KPI 有低 / 高占比；「极高 > 7 Pa」和均值 / 中位不集中显示 | 可接受 | S |
| W33 | 周期量卡（均值 / p99 / 最大 / 阈值占比 / 逐分支 / 派生说明 / 滞留区） | `cycleCard` | HAVE | `ws_overview` cycleSub + 分区表 + 透镜 | — | — |
| W34 | 分支统计表 | `renderPanel` `branch-card` | HAVE | `ws_overview.branchTable`（分区「表格」里，跟随字段） | — | — |
| W35 | 需关注的输入信息（input_check.flags + summary 顶层 flags） | `renderPanel` | PARTIAL | 「工具 → 输入检查」列 input_check 的 flags（`ws_detail`）；summary 顶层 `flags`（manifest `analysis.flags`）没显示 | 补到输入检查或概览 | S |
| W36 | 计算过程与显示口径（插值 σ / 邻点 / 覆盖半径、top-5% 簇数、各出口盖面半径与 Murray 分流 %、出口是否人工确认、输入与采样） | `renderPanel` process-only | PARTIAL | 插值口径在技术信息「显示插值」、透镜；输入与采样在输入检查；Murray 分流 %、top-5% 簇数没有 | 分流 % 放进「出口命名」 | S |
| W37 | 几何参数表（长度、最小 / 中位半径、最大内切直径、迂曲度） | `renderPanel`「几何参数」 | MISSING | manifest `analysis.branch_geometry` 已带，JS 未用 | 放进工具页或透镜 | S |
| W38 | 耗时与运行记录（含「已预计算」） | `renderPanel` | HAVE | 「工具 → 计算过程」（`ws_detail`） | — | — |
| W39 | 完整参数与版本信息 JSON | `renderPanel` details | HAVE | 技术信息里的「完整统计 JSON」「运行清单」链接（完整档） | — | — |
| W40 | 分支展开图（s, θ） | `unroll-details`、`drawUnroll` | HAVE | 「沿程」页签（`ws_unroll`、`ws_detail`，逐像素一致见 lane D §2） | — | — |
| W41 | 悬停读数：壁面插值值 + 分支 + 弧长 + 最近顶点坐标 + 可信提示；点云模式读预测点全部字段与几何特征 | `#tip`、canvas mousemove | MISSING | `core_viewer` 会发 `hover` 事件，但 v2 全局没有订阅者；只有单击探针（`ws_probe`） | 需要裁定：单击代替悬停可接受就改 DROP；否则订阅 hover 在状态行显示 | M |
| W42 | 左下提示可关闭且记住 | `#tip-close`，localStorage `wss-report-tip-off` | DROP-CANDIDATE | v2 无此提示框 | 放弃 | — |
| W43 | 探针卡（此点 / 截面均值 / 环上范围条；点行切字段；记录；×） | `#probe-card`、`renderProbeCard` | HAVE | `ws_probe` 卡片，行点击 `ctx.onField`，`sectionMeans` 同一函数 | — | — |
| W44 | 探针记录（表、删行、复制 TSV、导出 CSV、清空） | `#probe-log`、`#probe-copy/csv/clear` | HAVE | `ws_probe` 探针记录（同一 `probeToTSV/CSV`；记录存本机） | — | — |
| W45 | 测量：距离 / 弧长 / 管径（真实截面，回退内切）/ 分段；逐条复制、删除、清空 | `#measure-*`、`buildMeasurement` | HAVE | `ws_measure`（同一 WssReportCommon 函数） | — | — |
| W46 | 沿程曲线：分支或「全部叠加」、横轴「距入口 / 分支内弧长」、点击高亮 ±2 mm、SVG、最大直径行 + 飞到 + 环 | `#menu-profiles`、`drawProfiles` | PARTIAL | 「沿程」页签（`ws_profile`、`ws_detail`）只能单选分支，横轴固定距入口；无「全部叠加」、无分支内弧长；其余 HAVE（SVG 逐字节相同，lane D §2） | 可接受；需要时加叠加 | S |
| W47 | 区域统计（分支 + 弧长区间 / 球形刷子） | `#menu-region`、`applyRegion` | HAVE | `ws_region` | — | — |
| W48 | 发现：列表、飞到高亮、确认 / 驳回 / 未判定、备注、人工新增 / 删除、保存、「审阅」页签待判定计数 | `#menu-findings`、`setDecision`、`saveReview` | HAVE | `ws_overview` 发现区 + `ws_review`（同一 findings_review 文档） | 离线不能判定（经典离线存进视图状态），可接受 | — |
| W49 | 标注：钉、编辑、删除、保存到服务；离线显示内嵌副本 | `#menu-annot`、`addAnnotation` | HAVE | `ws_annot`（自动保存；离线只显示） | — | — |
| W50 | 出版级 PNG：1/2/4×、背景、色标叠加 / 不含 / 另存 SVG、中英、含界面标签、显卡降级提示 | `#exp-png`、`doExport` | HAVE | `ws_figure`「图片」页 | — | — |
| W51 | 导出六视角（6 个 PNG 文件） | `#views-export` | PARTIAL | `ws_figure` 出一张六联拼图；不再分 6 个文件 | 如需单张，加「逐张下载」选项 | S |
| W52 | 色标 SVG | `#exp-svg` | HAVE | `ws_figure` | — | — |
| W53 | 多视角拼图（视角勾选 + 当前、列数 2/3/4） | `#montage-*`、`exportMontage` | HAVE | `ws_figure`「拼图」页 | — | — |
| W54 | 生成一页纸配图（上传 snapshots，给「打开一页纸」链接） | `#snap-onepage`、`buildSnapshots` | HAVE | `ws_figure`「一页纸」页（snapshots.json 键完全相同，lane A §3） | — | — |
| W55 | 复制复现链接（`#view=` 全量视图状态） | `#view-link`、`CORE.viewHash` | PARTIAL | `ws_figure.viewState` 只含 f / w / L / cam / cu / sel / br / sl；不含配色、分段、自定阈值（`getState().display` 没进链接）、单位、标签设置、测量、标注、探针记录；读不了经典 `#view=`（`decodeState` 要求 `v === 1`） | 链接里加 display；配旧链接转换（见 §F） | M |
| W56 | 视图保存到本浏览器 / 恢复 | `#view-save/restore`，localStorage `wss-report-view:<run>` | HAVE | 自动记住阅读位置（`store().readView`）+ 书签 | — | — |
| W57 | 视图状态 JSON 导出 / 导入（`wss-deploy.view/v1`，含测量、标注、探针记录、离线判定；异 run_identity 提示） | `#view-export/import` | PARTIAL | 书签 JSON（`wssv2.bookmarks/1`，`ws_bookmarks.importJson`）格式不兼容，不含测量 / 探针记录 | 可接受；若要读旧文件需转换器 | M |
| W58 | 保存截图（头部按钮 + S 键；1200×700，带标题、色条、统计说明行） | `#save-image`、`capture()` | PARTIAL | 导出对话框「图片」页可出同类图（带标题）；没有一键截图，S 键在 v2 是截面 | 可接受 | S |
| W59 | 打印 / 保存 PDF（截图 + 统计卡两栏 + 打印说明；`beforeprint` 抓图） | `#print-report`、`@media print` | PARTIAL | `ws_figure.printView` 只印当前视图 + 色条 + 标题；统计由一页纸承担；直接 Ctrl+P 只隐藏顶栏 / 工具条（`v2.css` @media print），三维画布可能为空 | 可接受（一页纸代替） | S |
| W60 | 紧凑布局切换（窄窗 / 并排比较自动，记住） | `#layout-toggle`，localStorage `wss-report-compact` | DROP-CANDIDATE | v2 面板可收起、响应式；并排比较由 v2 自己做 | 放弃 | — |
| W61 | 页脚：审阅状态、发布包短名、生成时间 | `renderFooter` | HAVE | 检查器状态行（`ws_overview` insp-status）+ 顶栏 | — | — |
| W62 | 技术信息弹窗（发布包、哈希、模型族 / 模型数、特征合同、run_identity、输入 SHA256、模型帧、周期定义、生成 / 重建时间、摘要版本） | `#tech-btn`、`renderTech` | PARTIAL | `ws_shell.renderTools`「技术信息」只在完整档、只在线（离线不显示「工具」页）；缺特征合同、模型族 / 数、模型帧、周期定义、生成 / 重建时间、摘要版本（manifest `provenance` 里多已带） | 补字段，并让基本档与离线也能看 | S |
| W63 | 术语「?」弹窗（glossary.json，中英；p99、最大、面积加权、质量、门控、人群分位、阈值、单位、可信、标准视角、管径、弧长、判定、结论、审阅、发布包、特征合同、run_identity、RRT、ECAP…） | `.gloss`、`COMMON.glossaryPopover` | PARTIAL | v2 用 `ui.infoTip` 写死的中文说明 + 证据透镜；v2 不加载 glossary.json（一页纸术语表仍用它，两处口径会分叉） | 让 infoTip 读 glossary.json | M |
| W64 | 语言切换（导图、色条、测量标签、结论英文句、术语英文） | `#exp-lang` | PARTIAL | 只有导出图片可选英文（`ws_figure`）；界面、结论英文句不切 | 可接受 | S |
| W65 | 顶部警示条：几何超出发布包参考范围 / 人群参照需复核 / 集成质量非 good；「详情」「×」 | `#warn-banner`、`bannerInfo` | MISSING | v2 结果页没有任何对应提示（`ws_admin` 的横幅只管服务状态） | 必须迁（安全提示） | S |
| W66 | WebGL 不可用 / 上下文丢失时的回退 | `initViewer` catch、`webglcontextlost` | HAVE | `ws_shell.showResult`「数字、发现和导出仍可使用」，`contextlost/restored` | — | — |
| W67 | 快捷键：1–6 标准视角、0 / R 复位、F 循环字段、L 自动标注、P 锁定悬停点探针、S 保存截图、? | `installShortcuts(KEYS)` | PARTIAL | v2（`ws_shell.onKey`、`shortcutsDialog`）：1–9 = 切字段、0 = 复位（无 R）、L = 光照、S = 截面、B = 书签、M、G、J/K、[ ]、N、/、O、?；视角 1–6、R、F、P、截图键没有，且同键不同义 | 至少加 R；在「?」里注明经典键位变化 | S |
| W68 | 「← 工作台」返回链接 | `#back-to-workbench` | DROP-CANDIDATE | v2 自己就是工作台 | 放弃 | — |
| W69 | 时间帧下拉（禁用，只作说明；周期量显示「单周期积分量」） | `#frame-select`、`#frame-status` | DROP-CANDIDATE | v2 时间条 `renderTimebar` + 色条 / 透镜说明 | 放弃 | — |
| W70 | 离线单文件：`file://` 直接打开 report.html（three.js 内嵌，离线只读，偏好存 localStorage） | report.html 本身；bundle.zip | PARTIAL | v2「导出 → 数据 → 离线阅读」（`POST /api/v2/jobs/<id>/offline`，`v2_offline`）；但 bundle.zip 仍打包经典 report.html（`bundle.py` 第 54 行说明「report.html 可离线打开」） | S7 时 bundle 改放 v2 离线页或保留冻结版 | M |
| W71 | 父页面消息协议：`wss-view:ready / set-camera / camera / get-state / state / apply-state(+preset_name) / applied / export / exported / changed / error` | `window.message`、`COMMON.viewMessaging` | DROP-CANDIDATE | v2 不需要；但只能和 compare.html、batch_export.js 同时退役（见 §E-2） | 条件放弃 | — |
| W72 | 触屏（pointer:coarse）44 px 控件 | `@media (pointer:coarse)` | PARTIAL | v2 CSS 没有 `pointer:coarse` 规则（未核实实际点按尺寸） | 在 iPad 上点一遍 | S |

## B. 体场报告（volume_report.py + volume_viewer.js 页面代码）

| # | 功能 | 旧入口 | v2 状态 | 证据或缺什么 | 建议 | 工作量 |
|---|---|---|---|---|---|---|
| V1 | 显示页签：体内点云 / 截面 / 壁面压力 / 流线（缺数据时禁用并说明原因） | `mode-*`、`setMode`、`modeAvailable` | HAVE | 字段 speed / pressure / wall_pressure + 截面（S）+ 图层「流线」 | 键 T 见 V47 | — |
| V2 | 物理量：速度大小 / 压力 | `#volume-field`、`setField` | HAVE | 工具条字段 + 1–9 | 键 V/B 见 V47 | — |
| V3 | 色标 + 色带分段（0/4/6/8/12，本机记住） | `#colormap`、`#color-bands`，localStorage `wss-volume-*` | HAVE | 图层菜单色表 + `ws_display` 分段 | — | — |
| V4 | 血管模块（按分支筛选显示与截面统计；切割后选 A / B 侧） | `#volume-module` | PARTIAL | 分支显隐只改显示（`ws_shell.branchDialog`）；A / B 侧由截面「切开」代替；截面统计不随分支筛选 | 矩阵已记，可接受 | S |
| V5 | 血管轮廓透明度 0–0.5 | `#volume-opacity` | HAVE | `ws_display.buildVolume`「外壁」 | — | — |
| V6 | 单位 Pa / mmHg、m/s / cm/s | `#pressure-unit`、`#velocity-unit` | HAVE | `ws_display` 色标面板「单位」 | 概览 / 透镜不跟单位（lane E §7.1） | — |
| V7 | 速度对数色标（下限 = max(范围下限, 上限/200, 0.001 m/s)） | `#velocity-log` | MISSING | `core_colormap.LOG_FIELDS` 不含 speed；`ws_slice.scaleFor` 本截面色标也不取对数 | 用户试用反馈 7 的来源项，建议补 | S |
| V8 | 设为默认口径（配色、分段、单位、透明度、语言；本机 + 服务端） | `#defaults-save` | PARTIAL | 同 W23 | 同 W23 | M |
| V9 | 分支显隐 | `#branch-visibility` | HAVE | `ws_shell.branchDialog`（点、壁、流线） | — | — |
| V10 | 自动标注：发现 0/3/5/10、分支名 | `#labels-findings`、`#labels-branches` | PARTIAL | 同 W20 | 同 W20 | S |
| V11 | 速度方向箭头 | `#volume-vectors`、`addVectors` | HAVE | 图层「速度箭头」（lane E，与经典公式逐段一致） | — | — |
| V12 | 可信区域：灰化体内不可信点（bit 4/8/16）+ 壁面；可信图例（各类占比、规则、术语） | `#trust-overlay`、`renderTrustLegend` | PARTIAL | v2 体场只有壁面斜纹（`adapter_volume` hatch）；`vtrust` 被列为可选数组但不参与着色；无图例 | 补体内点灰化 + 图例 | S |
| V13 | 流线：粗细、密度（全部 / 一半 / 三分之一 / 五分之一）、细线模式；> 320 条默认一半 | `#streamline-*` | HAVE | `ws_display.buildVolume`（默认细线、密度全部） | — | — |
| V14 | 发现：列表、飞到高亮、判定、备注、人工新增（先写字再点位）、删除、取消高亮 | `#menu-findings`、`activateFinding` | HAVE | `ws_overview` + `ws_review` | — | — |
| V15 | 压降发现：点击后截面放到该支近端 10% 并提示定义（「滑到 90% 看远端」） | `activateFinding` pressure_drop 分支 | MISSING | v2 对 `pressure_drop` 无特殊处理（只在 `ws_overview` KIND 表里） | 小改：选中该发现时开截面到 10% | S |
| V16 | 测量（壁面或体内点取点；管径 = 2 × 内切半径）+ 复制列表 + 清空 | `#menu-measure` | HAVE | `ws_measure`（只在壁面取点；管径改为真实截面，更好） | — | — |
| V17 | 探针：悬停出简卡、单击固定全卡、P 键开关 | `probeAt`、`renderProbe`、`setProbeEnabled` | PARTIAL | 单击固定 HAVE（`ws_probe`）；无悬停读数、无 P 键（同 W41） | 同 W41 裁定 | M |
| V18 | 探针内容：位置、速度大小与分量、相对压力、分支、弧长、局部半径、到壁距离、可信标记；壁面顶点压力；截面积分（面积、等效径、面积平均速度、平均穿面速度、流量 Q、平均压力、取样与直接支撑 %）；「看截面」 | `renderProbe`、`sectionProbeRows`、`#probe-slice` | HAVE | `ws_probe`（`stationSection` + 经典 `sectionIntegral`）；小差：卡片不显示点坐标和可信标记两行 | 可接受 | — |
| V19 | 探针记录（含截面积分列）、复制 TSV、导出 CSV、清空 | `#probe-log-*` | HAVE | `ws_probe` | — | — |
| V20 | 标准视角 6 按钮 + 方向来源说明 | `#view-*`、`#view-direction-note` | HAVE | 方位立方体；方向来源在技术信息 | — | — |
| V21 | 视图状态：保存 / 恢复 / 导出 / 导入 JSON；复制复现链接；「复现此图（按链接）」 | `#view-*`、`captureView`、`applyView` | PARTIAL | 同 W55–W57；「按链接复现」按钮无对应（v2 打开链接即复原） | 同 W55 | M |
| V22 | 导出 PNG / 六视角 / 设置（倍率、背景、色标、语言、隐藏界面元素） | `#export-*`、`renderExport`、`exportSixViews` | HAVE | `ws_figure`（六视角为拼图，同 W51） | — | — |
| V23 | 多视角拼图（含「截面」面板） | `#montage-*`、`exportMontage` | HAVE | `ws_figure`「拼图」页（截面开着时可加） | — | — |
| V24 | 一页纸配图（前 / 左 / 上 / 当前，截面页时加截面图） | `#onepage-shots`、`makeOnepageShots` | HAVE | `ws_figure`「一页纸」页 | — | — |
| V25 | 截面点选：1 点垂直、2 点斜截面；清除选点；Esc 结束 | `#pick-toggle*`、`addPick`、`#pick-clear` | HAVE | `ws_slice` 截面区「点选」、「回到中心线」 | 键 X 见 V47 | — |
| V26 | 截面平面定位：垂直 X / Y / Z 轴 | `#slice-basis` x/y/z | MISSING | `ws_slice` 只有 centerline / pick | 可裁定放弃（冠状 / 矢状切面偶尔用） | S |
| V27 | 截面精确滑杆：位置、绕横轴 / 纵轴旋转、左右 / 上下偏移、厚度 | `#slice-position/pitch/yaw/offset-u/offset-v/thickness` | PARTIAL | v2 有厚度滑杆、位置 ±1 mm 按钮（`ws_slice.positionRow`）；倾角 / 偏移只能拖动或按键，没有数值 | 加两行数值输入 | S |
| V28 | 拖动含义工具条（沿法向移动 / 旋转 / 面内平移）+ 回正 | `#slice-tools`、`setDragMode`、`#slice-reset-angle` | PARTIAL | 回正 HAVE；拖动含义只能靠 Shift / Alt（`ws_slice` drag0.kind），触屏无法旋转或平移截面 | 加三段切换 | S |
| V29 | 滚轮沿法向移动、Shift 滚轮改厚度；↑↓ ←→ PgUp/PgDn [ ] 微调 | wheel / keydown 处理 | HAVE | `ws_slice` onWheel、`slice.key` | — | — |
| V30 | 自动截面（每支 5/20/40/60/80/95%） | `#auto-presets`、`automaticPlanes` | HAVE | `ws_slice` 站位行（逐位相同，lane E） | — | — |
| V31 | 选择并显示截面 / 按截面切割 / 取消切割 | `#slice-show/cut/clear-cut` | HAVE | `ws_slice`「切开」（不切 / 保留上游 / 保留下游） | — | — |
| V32 | 截面平面视图：补全、放大、收起；色标 本截面 / 全局 / 手动；着色 速度 / 穿面速度；面内箭头；统计（点数 / 均值 / p99 / 最大 + 轮廓面积 / 最大径 / 等效径）；中心 / 法向 / 厚度 | `#slice-panel` | HAVE | `ws_slice` 检查器截面区 | — | — |
| V33 | 放大图：色标 / 着色 / 箭头 / 显示预测点 / 补全；PNG、CSV；悬停读数；说明 | `#slice-zoom` | HAVE | `ws_slice.openZoom`（色标与着色在检查器调；说明里没有「管腔最大直径」） | — | — |
| V34 | 截面系列拼图（分支、站位数 2–12） | `#slice-series-*`、`exportSliceSeries` | HAVE | `ws_slice.openSeries`（站位数 4/6/8/12） | — | — |
| V35 | 结论（只读） | `#narrative-card` | HAVE | `ws_overview` | — | — |
| V36 | 参考范围卡（状态 + 超范围清单） | `#reference-card`、`renderWarnings` | PARTIAL | 只在证据透镜里 | 由 V45 兜底 | S |
| V37 | 体场统计（当前物理量：预测点数 / 均值 / p99 / 最大） | `#volume-statistics`、`setStats` | PARTIAL | 概览 KPI 只有速度 p99、最大速度、相对压力跨度（`kpiModel`）；速度均值、压力均值 / p99 / 最大、点数不集中显示 | 可接受或加一张小表 | S |
| V38 | 术语芯片（相对压力、ΔP、速度、采样支撑、近切口、几何越界、插值无支撑） | `#menu-stats` gloss chips | PARTIAL | 同 W63 | 同 W63 | M |
| V39 | 压力参考说明（`meta.pressure_reference`）、统计口径说明、「单个时相的流线不是粒子轨迹」提醒 | `#pressure-reference`、`#volume-protocol`、warning | PARTIAL | 相对压力定义在 KPI 透镜里；pressure_reference 文字和流线提醒没有（manifest 已带 `analysis.pressure_reference`） | 放进流线图层提示与透镜 | S |
| V40 | 沿程：最大直径行（飞到、环、术语）、分支、物理量（速度平均 / 最大、压力平均 / 最低）、半径 + 管腔直径、SVG | `#menu-profiles`、`drawProfile`、`#profile-svg` | HAVE | 「沿程」页签（`ws_profile`） | — | — |
| V41 | 沿程点击 → 截面跳到该弧长；曲线上画当前截面位置线 | `#profile-canvas` click、`sliceAtArc` | PARTIAL | v2 点曲线是移游标或高亮分箱（`ws_detail`），不移截面；无截面位置线 | 截面开着时点击改成移截面 | S |
| V42 | 两截面之间区域统计（分支、近 / 远端 %、三维高亮、速度 / 压力统计、近端−远端压差） | `#region-*`、`updateRegion` | HAVE | `ws_slice`「两截面之间」 | — | — |
| V43 | 标注（输入框或弹窗、点位、保存、清空） | `#menu-annot` | HAVE | `ws_annot`（无一键清空） | — | — |
| V44 | 预设：内置（沿程压降 / 流线全貌 / 瘤囊截面系列 / 临床视图）、用户预设（服务端、改名、导出 JSON） | `#menu-presets` | PARTIAL | 同 W22（「按问题看」不含任何体场问题） | 同 W22 | M |
| V45 | 顶部警示条（几何超参考范围 / 集成质量等；详情 / ×） | `#warn-banner`、`warningItems` | MISSING | 同 W65 | 必须迁 | S |
| V46 | 页脚 + 技术信息（含计算设备、坐标架方向来源；「复制」按钮） | `footer`、`#tech-info` | PARTIAL | 同 W62 | 同 W62 | S |
| V47 | 快捷键：1–6、0/R、V 速度、B 压力、T 循环页签、L 标注、P 探针、S 截图、X 点选、↑↓←→ PgUp/PgDn [ ]、Esc、? | `shortcutBindings` | PARTIAL | 截面微调键 HAVE；1–6、R、V、T、P、X、截图键没有；B 在 v2 是书签（经典体场是压力），S 是截面 | 同 W67 | S |
| V48 | 保存截图（带病例 / 物理量说明框）/ 打印（beforeprint 快照 + 右侧图例） | `#volume-save`、`#volume-print` | PARTIAL | 同 W58 / W59 | 同 W58 | S |
| V49 | 紧凑布局；右侧图例 / 截面面板遮挡时取景让位（setViewOffset） | `applyCompact`、`dockOcclusion` | DROP-CANDIDATE | v2 布局不同（insets 取景） | 放弃 | — |
| V50 | 无 WebGL 时仍可用截面投影与统计 | `volume-error` 说明 | PARTIAL | v2 无 WebGL 时截面按钮禁用（`renderToolbar` sliceBtn.disabled） | 低优先 | S |
| V51 | 「← 工作台」 | `#back-to-workbench` | DROP-CANDIDATE | 同 W68 | 放弃 | — |
| V52 | 父页面消息协议（同 W71，含 `wss-view:changed`） | `window.message` | DROP-CANDIDATE | 同 W71 | 条件放弃 | — |
| V53 | 「复现此图（按链接）」按钮 | `#view-replay` | DROP-CANDIDATE | v2 打开链接即复原 | 放弃 | — |

## C. 一页纸（onepager.py，服务端渲染，S7 不退役）

| # | 功能 | 旧入口 | v2 状态 | 证据或缺什么 | 建议 | 工作量 |
|---|---|---|---|---|---|---|
| O1 | 打开一页纸并打印 / 另存 PDF | `GET /api/jobs/<id>/onepage`，页内按钮 `window.print()`（CSP 只放行这个哈希） | HAVE | v2 新窗口打开同一地址：O 键（`ws_detail.onKey`）、工具页、导出「一页纸」页（`ws_figure`）、报告模板保存后的提示条（`ws_onepage`）；打印用一页纸自己的按钮 | — | — |
| O2 | 一页纸里指向经典页面的链接 | — | HAVE | onepager.py 没有任何 href 指向 `/report` 或 `/#job=`；S7 无需改链接 | — | — |
| O3 | 无配图时的提示文字 | `NO_SNAPSHOTS_HINT`（onepager.py 第 40 行）「在三维报告「导出」菜单点「生成一页纸配图」…」 | PARTIAL | 文字指向经典报告；v2 入口是「导出 → 一页纸」 | S7 时改文案 | S |
| O4 | 配图生成（snapshots.json + snapshot_*.png） | 经典 `buildSnapshots` / `makeOnepageShots` | HAVE | `ws_figure` 同格式（lane A §3.1） | — | — |
| O5 | 打包说明里提到 report.html 可离线打开、配图来自「三维报告」 | `bundle.py` 第 54、59 行 | PARTIAL | 与 W70 一起改 | S7 时改文案 | S |

## D. 并排比较（compare.html / compare.js，内嵌两个经典报告 iframe）

| # | 功能 | 旧入口 | v2 状态 | 证据或缺什么 | 建议 | 工作量 |
|---|---|---|---|---|---|---|
| C1 | 同步视角 | `#sync-camera`（`wss-view:camera` / `set-camera`） | HAVE | `ws_compare`「同步视角」，同一输入默认开（`viewer.link`） | — | — |
| C2 | 同步口径 | `#sync-display`（get-state → apply-state） | HAVE | 「同尺度看幅值」（同步分段与阈值） | — | — |
| C3 | 以左 / 右为准 | `#push-left` / `#push-right` | HAVE | `ws_display.pushSide` | — | — |
| C4 | 左右互换 | `#swap-sides` | HAVE | 比较页「左右互换」 | — | — |
| C5 | 标量统计对照表（`/api/compare`：声明一致的标量 左 / 右 / 差值；原因不一致时列原因） | `#compare-table`、`renderComparison` | MISSING | v2 只有「比较条件」表（一致 / 不一致 / 未知），不列数值 | 需要单独裁定：裁定过的是「逐点差值」，这张是标量表 | M |
| C6 | 类型不同提示 | `#scale-note` | HAVE | 条件表判「不一致」并禁用同尺度 | — | — |
| C7 | 两侧身份行（病例、患者、扫描标签、扫描日期、发布包、已审阅） | `#identity-left/right` | PARTIAL | `ws_compare` 只列病例名与结果名（cmp-who） | 可接受 | S |
| C8 | 同一输入的 WSS 与体场并排（工作台「并排打开 WSS + 体场」） | `/compare?left=<wss>&right=<vol>` | HAVE | `ws_shell.enterCompare` 可跨家族（右侧取默认字段） | — | — |

---

## E. S7 本身的阻塞项（删经典页面代码会连带坏掉的）

1. **报告文件仍是 v2 的数据源。** `v2_data.py` 从 `report.html` 读内嵌的 `wss-report-meta` 与数组（第 3、396 行）；`rebuild_report.py`、`report_freshness.py`、`pipeline.py`、`volume_pipeline.py`、`jobs.py` 都写 / 读它。所以 S7 不能删 `report.build_html` / `volume_report.build_html`，只能把模板换成「数据壳」（去掉页面脚本或换成跳转）。`report_freshness.UI_SOURCES` 按 report.py、volume_report.py、report_common.js、volume_viewer.js 等文件指纹判断旧报告，模板一改，旧报告会在下次访问或 `cli reports refresh` 时只换外壳、数据逐字节不变——可以直接用它把全部旧报告换成新壳。
2. **跨病例批量出图（经典工作台）依赖经典报告页。** `static/batch_export.js` 用隐藏 iframe 加载 `/api/jobs/<id>/report`，走 `wss-view:ready → apply-state → applied → export → exported`。矩阵 A.1 已记「跨病例批量出图没迁」；页面代码一删这项就坏。compare.html 同理（C1–C8 已在 v2 有替代）。
3. **v2 自己还链到经典报告**：`ws_shell.js` 第 501 行（结果读取出错时「在经典报告中打开」）、第 1785–1786 行（工具页「经典报告」小节）、第 386 行文案「仍不行时用经典报告」、`ws_api.js` 第 168 行 `urls.report`、`ws_annot.js` 第 164 行「经典报告里也能看到」。S7 删经典页面后第 501 行这个出错兜底就没有了，需要换成别的兜底（例如「下载离线页」）。
4. **只在经典壁面页里的数值代码**：`report.py` 的 `REPORT_CORE_JS`（`WssReportCore`：`marchingTriangles`、`thresholdAreas`、`topThreshold`、`encodeView` 等）v2 没用到。以后要迁等值线（W10）或读旧 `#view=` 链接（§F），要先把相关函数搬进 report_common.js 或 v2。
5. **服务端偏好里有经典的数据没人读**：`/api/preferences` 的 `presets.wall / presets.volume`（用户预设）和 `report_defaults.wall / volume`（默认口径）。S7 前要么导入 v2（书签 / 显示默认），要么通知用户自己导出。
6. **测试与工具**：`tests/test_report.py`、`tests/test_volume_report.py`（驱动页面里的 `__wssTest` / `module.exports.__test` 钩子）、`tests/test_glossary.py`、fixture `tests/fixtures/20260920_173929_a9ec139d6cdd/report.html`；`rehearse.py` 第 193 行冒烟检查 `GET /api/jobs/<id>/report` 要返回 `<html`；`devshot.py` 第 268 行截经典报告。都要随 S7 改。
7. **离线包**：bundle.zip 里的 report.html 是离线可看的主入口（W70、O5）。换成数据壳后，包里要放 v2 离线页（`v2_offline.build_offline_html`，目前只经 `POST /api/v2/jobs/<id>/offline` 生成）或保留一份冻结的经典页。

## F. 经典 URL → v2 跳转对照

先说限制：**`#…` 片段不会发到服务器**，服务端 302 看不到 `#view=`。按 RFC 7231 §7.1.2，Location 里不带片段时浏览器会把原片段接到新地址上；带了片段（例如 `/v2/#/job/X`）原片段就丢了。所以建议统一跳到一个**不带片段**的入口，由 v2 的一小段启动代码完成转换。v2 现在的路由只读 hash（`ws_shell.parseHash`：`#/job/<id>?v=&f=&cmp=&q=&bm=`，外加 `ws_figure` 读的 `&view=<base64url>`），不读 `location.search`，这段启动代码要新写（S）。

| 经典地址 | 怎么被寻址 | 建议跳转 | 说明 |
|---|---|---|---|
| `GET /api/jobs/<id>/report` | 工作台「打开三维报告」、compare iframe、batch_export、旧书签 / 邮件 | `302 → /v2/?job=<id>`；v2 启动代码再 `replaceState` 成 `#/job/<id>`（原 hash 有经典 `#view=` 时加 `?f=<f>&view=<v2>`） | 壁面、体场同一条；任务没完成时 v2 会显示输入 / 进度页 |
| `GET /api/jobs/<id>/files/report.html` | 文件路由（报告同一文件） | 顶层导航（`Sec-Fetch-Dest: document`）同上跳转；其他请求保持原样下载 | 否则会直接显示数据壳 |
| `GET /jobs/<id>/report.html` | 旧路由（server.py 第 1584 行 `legacy`） | 同第一行 | 同一路由也服务其他输出文件，只拦 report.html |
| `…/report#view=<base64url JSON>` | 经典「复制复现链接」 | 经上面的 302 后片段保留；启动代码把 `wss-deploy.view/v1` 转成 v2 `{v:1,…}`（见下） | v2 `decodeState` 只收 `v === 1`，不能原样传 |
| `/compare?left=<A>&right=<B>` | 工作台「并排比较」「并排打开 WSS + 体场」 | `302 → /v2/#/job/<A>?v=compare&cmp=<B>` | query 服务器看得到，可以直接拼；`v=compare` 在 `VIEWS` 里 |
| `/api/jobs/<id>/onepage` | 一页纸 | 不变 | 服务端渲染，不属于退役范围 |
| `/#job=<id>` | 经典报告「← 工作台」的目标 | `/v2/#/job/<id>`（片段，只能在 `/` 的页面里用脚本转） | 属工作台退役范围，列出供对照 |
| `file://…/report.html` | bundle.zip / 下载的离线报告 | 无法跳转 | 见 §E-7 |

经典视图状态 → v2 `view` 的对照（启动代码里转换）：

- 通用：`field` → `f`（壁面 id 相同：wss / tawss / osi / rrt / ecap；体场 `velocity` → `speed`，`pressure` → `pressure`，`mode:'wall'` → `wall_pressure`，`mode:'streamlines'` → `speed` + `L.streamlines=true`）；`camera{position,target,up}` → `cam{p,t,u,fov}`（同一世界坐标 mm；fov 壁面 35、体场 42）；`range.mode:'fixed'` + `max`（Pa 基本单位）→ `w:{range:[0,max]}`，`'case'` → `'adaptive'`（`log:false` 且字段默认对数时 → `'adaptive-linear'`）；`overlay.trust` → `L.trust`；壁面 `mode:'cl'` → `L.centerline`，`'cloud'` → `L.points`；`branches_hidden` → `br` = 全部分支 id 减去隐藏的（需要 manifest `geometry.branches`）。
- 体场截面（`mode:'slice'`）：`slice.branch` → `sl.segment`，`position/100` → `sl.fraction`，`pitch/yaw/thickness` 同名，`offset_u/v` → `offU/offV`，`picks` → `sl.picks`（v2 再算平面），`quantity`（speed / normal；压力场 → pressure）、`color_range.mode/min/max` → `range`/`manual`，`arrows`；`cut` → `sl.cut`（A / B 侧与「保留上游 / 下游」的对应要在浏览器里核对，未核实）；`basis` 为 x / y / z 时 v2 没有对应，退回中心线默认位置并提示。
- 转不过去、要提示「经典链接里的这些设置新工作区不带」：`colormap`、`bands`、`thresholds_pa` / `field_thresholds`、`units`（dyn/cm²）、`labels`、`measurements`、`annotations`（服务端本来就有）、`probe_log`、`slice.clip`（剖切）、`highlight.top*` / `feature` / `branch`、`lang`、`export`、`montage`、`preset_name`、`highlight.finding`（v2 `sel` 是预测点序号，不是发现 id）。

## G. 退役前必须先做的（按优先级）

1. **顶部警示条 + 五模型一致性（W65、W29、V45）** — S。几何超出参考范围、人群需复核、集成质量非 good 时，经典页有醒目提示，v2 结果页完全没有（数据 `analysis.reference_assessment`、`analysis.quality` 已在 manifest 里）。这是安全相关的缺口。
2. **经典链接的跳转与转换（§F）** — S（302 + 启动代码）到 M（`#view=` 转换）。否则旧邮件 / 笔记里的报告链接和复现链接全部失效。
3. **批量出图与 compare / batch_export 的去留（§E-2）** — 裁定：迁到 v2（M–L）或与经典工作台一起明确放弃；否则 S7 一删就坏。
4. **v2 内部指向经典报告的链接与出错兜底（§E-3）** — S。
5. **报告文件改成数据壳 + 离线包（§E-1、§E-7、W70、O3、O5）** — M。用 report_freshness 批量换壳；bundle.zip 改放 v2 离线页；改一页纸与打包说明文案。
6. **服务端已存的预设与默认口径（W22、W23、V44、§E-5）** — M。导入或公告。
7. **复现链接补 display（分段、阈值、单位、配色）（W55）** — S，否则 v2 的链接比经典的少复原一截。
8. 功能性小缺口，建议 S7 前补或逐条裁定：dyn/cm²（W8）、速度对数色标（V7）、压降发现定位截面（V15）、截面拖动模式工具条与倾角 / 偏移数值（V27、V28，触屏必需）、体内不可信点灰化与可信图例（V12、W11）、剖切（W17）、几何参数表与 Murray 分流（W37、W36）、技术信息补字段并放开到基本档 / 离线（W62）、自动标注 0/3/5/10（W20）、键位 R 与「?」里的变更说明（W67、V47）、标量对照表（C5）。
9. 需要裁定是否放弃的：悬停读数（W41、V17）、等值线（W10）、高亮最高 x%（W16）、X/Y/Z 轴截面（V26）、蓝白红色表（W4）、手输固定上限（W6）、术语弹窗改读 glossary.json（W63、V38）。
