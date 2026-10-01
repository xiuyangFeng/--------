# 经典结果页 → v2 复核（第三期合并后，`wss-ui-v2` HEAD e7c2356）

只读复核。逐行对照 `wss_deploy/lanes/p3_audit_reports.md` 的 138 行（W1–W72、V1–V53、O1–O5、C1–C8），每行都读了当前代码（`static/v2/*.js`、`bundle.py`、`onepager.py`），没有只信路报告。DROPPED 只用于 `PHASE3_LANES.md` §0 明确放弃、能引出处的行；原审计标 DROP-CANDIDATE 但 §0 没点名的，记为 STILL-MISSING，并在建议栏写「放弃（只差一句裁定）」。

**汇总：壁面 72 = DONE 44 · STILL-PARTIAL 20 · STILL-MISSING 7（全是原 DROP-CANDIDATE，未裁定）· DROPPED 1；体场 53 = DONE 41 · STILL-PARTIAL 8 · STILL-MISSING 3（全是原 DROP-CANDIDATE）· DROPPED 1；一页纸 5 = DONE 5；并排比较 8 = DONE 8（另补 2 行比较页观察，不计入 138）。合计 138 = DONE 98 · STILL-PARTIAL 28 · STILL-MISSING 10 · DROPPED 2。** 原审计里的 13 行「真缺」（MISSING）已经全部做完；现在剩下的 STILL-MISSING 都是原来就建议放弃的页面外壳类功能。

## 未完成的行

工作量：S ≤ 2 h，M ≤ 1 天，L > 1 天。频率按「医生 / 研究者用这个工具时」估。

| 行号 | 功能 | 现状 | 证据或还缺什么 | 用户损失 | 使用频率 | 工作量 | 建议 |
|---|---|---|---|---|---|---|---|
| W2 | 中心线按半径着色 + 端点「名称 r=… mm」 | STILL-PARTIAL | `adapter_wall.js` clObj 仍是单色 `#465361`；端点标签只在「工具 → 出口命名」（`ws_detail.outletLabels`）和输入检查里出现 | 中心线模式下不能一眼看出哪段变细 | 很少 | S | 放弃（「沿程」页签有半径曲线，另有管腔最大直径环） |
| W9 | 改阈值后统计卡、分支表、沿程虚线、周期卡一起重算 | STILL-PARTIAL | `ws_display.buildScale`：只改色条粗线、等值线和面板里的占比，ⓘ 已写明「概览、发现和导出的统计仍按发布包阈值」 | 用自定阈值时，各分支占比得自己估 | 偶尔（研究者） | M | 放弃（第二期 E 路裁定「数字不变」，界面已写明） |
| W13 | 全场最大值的位置说明（分支、距入口、距分叉、局部半径） | STILL-PARTIAL | 标记只写「最大 x Pa」（`ws_display` displayProvider.markers）；`analysis.peak` 已带位置数据，JS 没读 | 写报告描述峰值位置时，要自己点探针或用游标读 | 偶尔 | S | 迁移（标记 title 或透镜里加一行） |
| W14 | 分支下拉「淡化其余」（最高 x%、展开图只取本支） | STILL-PARTIAL | 只有显隐（`ws_shell.branchDialog`，隐藏不是淡化），概览分区悬停时临时淡化（`ws_shell` `v.highlight`） | 单看一支时，其余血管的位置参照没了 | 很少 | M | 放弃（显隐和分区悬停代替） |
| W18 | 壁面不透明度 0.2–1 | STILL-MISSING（原 DROP-CANDIDATE） | v2 没有；`ws_legacy.classicToV2` 遇到它就提示「壁面透明度」没带过来 | 不能半透明看管腔内的中心线 | 很少 | S | 放弃（中心线 depthTest:false 穿透显示；第 4 路已按放弃处理，补一句裁定即可） |
| W19 | 点云特征着色（弧长 / 周向角 / 半径 / 到分叉距离） | STILL-MISSING（原 DROP-CANDIDATE） | v2 没有 | 只影响调试 | 很少 | M | 放弃（调试用途；展开图已给出 s–θ） |
| W22 | 预设：4 个内置 + 服务端用户预设（改名、导出 JSON） | STILL-PARTIAL | 服务端旧预设能读能「应用」（`ws_display.classicSection` / `applyPreset`）；4 个内置预设里只有「瘤囊低 WSS 区」由 `ws_questions`「低值区域在哪里」近似，「髂分叉热点」「主动脉沿程」「临床视图」没有一键入口；不能新建、改名、导出服务端预设，新视图只能存本机书签（`ws_bookmarks`） | 换电脑后自己的视图不跟着走；固定口径的图每次要手动配 | 偶尔 | M | 以后再说（书签 + 复现链接可以代替；多机使用的人多了再做服务端书签） |
| W23 | 设为默认口径（本机 + 服务端 `report_defaults.wall`） | STILL-PARTIAL | `ws_display.setDefault` 只按字段存分段和阈值，色表、单位是本机偏好；不写服务端；服务端旧默认要在工具页手动「套用」（`applyDefaults`） | 换浏览器或换电脑后默认口径要重设 | 偶尔 | M | 以后再说（第 4 路已提请裁定：手动套用还是首次自动套用、要不要写服务端） |
| W26 | 结果字段与模型卡：声明字段表（单位 / 位置 / 类型）、集成协议、权重数 | STILL-PARTIAL | 模型数在 `ws_detail.techRows`「模型族」；声明字段表和集成协议文字没有（`ws_overview.modelCardBody` 只有一致性表） | 看不到各输出字段的单位和位置一览（色条上有单位） | 很少 | S | 放弃（技术信息 + 色条已覆盖） |
| W27 | 峰值帧 p99 卡（p99、全场最大、最大值位置） | STILL-PARTIAL | `ws_overview.kpiModel`：结果有 TAWSS 时 KPI 固定是 TAWSS 三项，切到 WSS 字段时没有全场 WSS p99（只在分支表、透镜里）；位置文字同 W13 | 三头结果里找峰值 WSS 全场 p99 要翻表 | 偶尔（默认模型仍是 X5D 单帧，KPI 正常） | S | 迁移（KPI 跟当前字段） |
| W28 | 壁面面积加权参考（面积加权 p99、最高 1% 面积均值、覆盖面积） | STILL-MISSING（原 DROP-CANDIDATE） | v2 JS 不读 `analysis.surface_statistics` | 少一个补充口径 | 很少 | S | 放弃（经典页自己注明主指标是点云 p99；工作台审计 #91 同样建议放弃） |
| W30 | 出口命名门控卡（需人工确认 / 已通过；校准状态；「代理值未作为概率」） | STILL-PARTIAL | `ws_detail.outletSection` 写来源和「自动命名把握度 x%」；没有「把握度不是概率」这句；概览有确认来源（`outletShareModel`） | 把握度百分比可能被误读成概率 | 偶尔 | S | 迁移（加一句 ⓘ） |
| W32 | 点数占比与估计面积（低 / 高 / 极高；均值 / 中位） | STILL-PARTIAL | KPI 有低、高占比；「极高 > 7 Pa」只在色标面板的三档占比（`buildScale`）；全场中位数和估计面积 cm² 不集中显示 | 要多点一下才看到极高占比 | 很少 | S | 放弃（色标面板 + 分支表已覆盖） |
| W35 | 需关注的输入信息（`input_check.flags` + summary 顶层 flags） | STILL-PARTIAL | `ws_detail.inputModel` 只列 `ic.errors` / `ic.flags`；manifest 的 `analysis.flags`（出口命名建议的 flags，`pipeline.py` 第 600 行）在结果页没有读者 | 自动命名的告警在结果页看不到 | 偶尔 | S | 迁移（放进「工具 → 出口命名」或输入检查） |
| W42 | 左下提示可关闭并记住 | STILL-MISSING（原 DROP-CANDIDATE） | v2 没有这个提示框 | 无 | 很少 | — | 放弃（只差一句裁定） |
| W46 | 沿程曲线「全部叠加」、横轴「分支内弧长」 | STILL-PARTIAL | `ws_profile` 只能单选分支，横轴固定为距入口；其余（SVG、最大直径行）已有 | 不能把几条分支画在同一张图上比 | 偶尔（研究者） | M | 以后再说 |
| W55 | 复现链接（全量视图状态；读经典 `#view=`） | STILL-PARTIAL | 第 5 路的 `ws_figure.viewState` 加了 `d`（色表、分段、阈值、单位、标签）和 `x`；`ws_legacy` 能读经典链接。还缺两样：① 测量、探针记录不进链接；② 经典链接里的等值线、剖切、高亮最高 x%、体场速度对数仍被 `ws_legacy.classicToV2` 当作「没带过来」丢掉（第 252–294 行）。v2 已有这些功能，`ws_display.applyLinkState` 也能收 `x.display {contours, top, topPct, clip, speedLog}`，只是转换器没映射 | 旧链接打开后少了等值线、剖切；分享出去的链接不带测量 | 偶尔 | ② S，① M | 迁移 ②（映射到 `x.display`）；① 以后再说 |
| W57 | 视图状态 JSON 导出 / 导入（`wss-deploy.view/v1`，含测量、探针记录） | STILL-PARTIAL | 只有书签 JSON（`ws_bookmarks.importJson`，`wssv2.bookmarks/1`），格式不兼容，不含测量和探针记录；旧的视图文件导不进来（`ws_legacy.classicToV2` 可以复用，但没接文件入口） | 以前导出的视图文件打不开 | 很少 | M | 以后再说（有人带着旧文件来时，复用 `classicToV2` 做「导入旧视图文件」） |
| W58 | 一键保存截图（头部按钮 + S 键） | STILL-PARTIAL | 要走「导出 → 图片」（`ws_figure`）；S 键现在是截面 | 多点两下 | 偶尔 | S | 放弃（已在快捷键说明里写明） |
| W59 | 打印 / 保存 PDF（截图 + 统计卡两栏，`beforeprint` 抓图） | STILL-PARTIAL | `ws_figure.printView` 只排当前视图、色条和标题；直接按 Ctrl+P 时没有 `beforeprint` 处理，`v2.css` @media print 只隐藏顶栏和工具条（canvas 用 `preserveDrawingBuffer:false`，三维图可能是空白）。经典页的打印样式随页面一起下线了 | 想打印「图 + 统计」要改用一页纸；Ctrl+P 可能得到空白画布 | 偶尔 | S | 放弃（一页纸承担）；可选 S：Ctrl+P 时转给 printView |
| W60 | 紧凑布局切换（窄窗 / 比较时自动） | STILL-MISSING（原 DROP-CANDIDATE） | v2 面板可收起、布局响应式 | 无 | 很少 | — | 放弃（只差一句裁定） |
| W63 | 术语「?」弹窗（`glossary.json`，中英） | STILL-PARTIAL | `ws_lens.loadGlossary` / `termText` 已经接上，但只有概览、警示、技术信息、透镜里约 15 个词用它；色标面板、截面、探针、图层菜单的 ⓘ 仍写死；p99、最大、阈值、单位、标准视角、弧长、判定、审阅等三十来个条目在界面里没有入口；没有英文释义 | 有的词查不到定义；ⓘ 和一页纸术语表的说法可能不一致 | 偶尔 | M | 以后再说（各文件逐个改调 `ns.lens.termTip`，每个文件 S） |
| W64 | 语言切换（界面、结论英文句、术语英文） | STILL-PARTIAL | 只有导出图可以选英文（`ws_figure`） | 看不到英文界面和英文结论 | 很少 | L | 放弃（界面中文；出图有英文） |
| W67 | 快捷键（1–6 视角、0 / R 复位、F、L、P、S、?） | STILL-PARTIAL | `ws_shell.shortcutsDialog`：1–9 是切字段，0 复位，R 不再复位，P 是悬停读数，S 是截面；变化已写进「?」 | 老用户按习惯键会出错 | 偶尔 | S | 放弃（方位立方体代替；变化已写明） |
| W68 | 「← 工作台」返回链接 | STILL-MISSING（原 DROP-CANDIDATE，不再适用） | v2 本身就是工作台 | 无 | — | — | 放弃（只差一句裁定） |
| W69 | 时间帧下拉（禁用，只作说明） | STILL-MISSING（原 DROP-CANDIDATE） | v2 用时间条 `ws_shell.renderTimebar` 表达 | 无 | — | — | 放弃（只差一句裁定） |
| W71 | 父页面消息协议 `wss-view:*` | DROPPED | §0.1「比较页左右两份完整报告」放弃；§3 第 6 路第 3 项删了 `compare.js`、`batch_export.js`，这个协议已经没有调用方（批量出图改由 `ws_batch.js` 离屏渲染） | 无 | — | — | 已放弃 |
| W72 | 触屏 44 px 控件（pointer:coarse） | STILL-PARTIAL | 只有 `v2_display.css` 第 97 行给截面拖动方式和数值框放大；其他控件没有；真机没测过（第 4 路报告 §7.4） | 平板上有些按钮难点 | 很少 | M | 以后再说（先在真机上点一遍） |
| V4 | 血管模块（按分支筛选显示和截面统计；切割后选 A / B 侧） | STILL-PARTIAL | 分支显隐只改显示；截面统计不随分支筛选；A / B 侧由截面「切开」代替 | 截面统计里混着被隐藏分支的点（极少见） | 很少 | M | 放弃（迁移矩阵已接受） |
| V8 | 体场「设为默认口径」（本机 + 服务端） | STILL-PARTIAL | 同 W23 | 同 W23 | 偶尔 | M | 以后再说（与 W23 一起裁定） |
| V21 | 视图状态保存 / 导出 / 导入 JSON；复现链接；「按链接复现」 | STILL-PARTIAL | 同 W55、W57；体场截面状态已进链接（`sl`），并能从经典链接转换（`ws_legacy.convertSlice`）；按坐标轴放的截面和点选截面从经典链接转换时回到中心线 | 同 W55、W57 | 偶尔 | 同 W55 | 同 W55 |
| V37 | 体场统计（当前物理量：点数 / 均值 / p99 / 最大） | STILL-PARTIAL | `kpiModel` 只有速度 p99、最大速度、相对压力跨度；速度均值、压力均值 / p99 / 最大、点数不集中显示（数据在 `analysis.volume_statistics`） | 报告里要写平均速度或压力时没有现成数字 | 偶尔 | S | 迁移（概览加一张小表） |
| V44 | 体场预设（沿程压降 / 流线全貌 / 瘤囊截面系列 / 临床视图 + 用户预设） | STILL-PARTIAL | 服务端旧预设可以「应用」，但截面状态不带（`applyPreset` 提示「截面」没带过来）；4 个内置体场预设都没有一键对应，「按问题看」里没有体场问题（`ws_questions`） | 体场常用视角每次要手动配 | 偶尔 | M | 以后再说（与 W22 一起） |
| V47 | 体场快捷键（V / B 切量、T 循环、X 点选、P、截图、1–6） | STILL-PARTIAL | 截面微调键都有；V、T、X、1–6、截图键没有；B 现在是书签（经典体场是压力） | 同 W67 | 偶尔 | S | 放弃（同 W67） |
| V48 | 体场保存截图（带说明框）/ 打印（右侧图例） | STILL-PARTIAL | 同 W58、W59 | 同 W59 | 偶尔 | S | 放弃（同 W59） |
| V49 | 紧凑布局；图例或截面面板遮挡时取景让位 | STILL-MISSING（原 DROP-CANDIDATE） | v2 用 insets 取景 | 无 | 很少 | — | 放弃（只差一句裁定） |
| V50 | 无 WebGL 时仍能用截面投影与统计 | STILL-PARTIAL | `ws_shell.renderToolbar`：`sliceBtn.disabled = !S.viewerA \|\| …`，没有 WebGL 时截面不可用 | 极少数没有 WebGL 的机器上不能看截面 | 很少 | M | 放弃（低优先） |
| V51 | 「← 工作台」 | STILL-MISSING（原 DROP-CANDIDATE，不再适用） | 同 W68 | 无 | — | — | 放弃（只差一句裁定） |
| V52 | 父页面消息协议（含 `wss-view:changed`） | DROPPED | 同 W71（§0.1 + §3 第 6 路第 3 项） | 无 | — | — | 已放弃 |
| V53 | 「复现此图（按链接）」按钮 | STILL-MISSING（原 DROP-CANDIDATE） | v2 打开链接就复原（`ws_figure` applyFromHash） | 无 | — | — | 放弃（只差一句裁定） |
| C+1（补充） | 比较时右侧结果的警示条和「质量与参照」 | DROPPED（§0.1）· 建议例外 | `ws_notice.show` 只看 `cur.manifest`（左侧）；比较页签（`ws_compare.render`）不显示右侧的参考范围或质量问题。第 3 路报告 §7.1 已记下（外壳没有「进入比较」的钩子）。经典比较页两侧各是完整报告，各自有警示条 | 右侧结果几何超出参考范围或集成质量差时，比较页上看不到提示 | 偶尔（随访对比常用比较） | S | 迁移（比较页签里给右侧加一行 `ns.notice.model(mB)` 文字。属安全提示，建议作为 §0.1 的例外） |
| C+2（补充） | 比较时的截面、测量、区域统计、标注；右侧完整探针卡；右侧概览、发现、导出 | DROPPED（§0.1「比较页左右两份完整报告：保持简化做法」） | `ws_shell.renderToolbar` 在 `cur.compare` 时禁用截面、测量、标注、区域按钮；`enterCompare` 会先关掉截面；`renderReading` 只给左侧出探针卡，右侧点选只出透镜；导出只出左视口（`ws_figure` 第 887 行） | 两次扫描不能在同一站位并排看截面；「WSS + 体场」并排时体场一侧不能开截面；右侧的发现和统计要单独打开它才看得到 | 偶尔 | L | 放弃（按 §0.1）。如果随访对比截面的需求多，再单独做「比较时同步截面」（M–L） |

### 其他发现（不在 138 行里，供 S7 收尾参考）

- 经典页面专属、现在已经访问不到的：打印样式（W59 / V48）、视图 JSON 导出 / 导入（W57）、一键截图（W58）。探针 CSV、沿程 SVG、六视角单张、截面放大图 PNG / CSV 在 v2 里都有。
- `ws_legacy.classicToV2` 没有映射第 4 路的 `x.display`（见 W55 ②）。合并记录 §5 只修了分段色带，这一处漏掉了。
- `static/v2/example_report.html` 是生成文件，里面还有「在经典报告中打开」和 `CLASSIC_TOOLS`（第 7316、7424、8243 行）。已知遗留：第 6 路 §9 要求上线后用 `v2_examples` 重新生成。
- 原审计 §E：E1 数据壳没做（第 6 路 §7.1，有意推迟）；E2 批量出图已由 `ws_batch.js` 重写，经典的批量预设没迁；E3 v2 里已经没有经典链接（`ws_api.js` 第 172 行只剩注释）；E4 `marchingTriangles` 已移到 `core_contour.js`；E5 服务端偏好能读能套用（W22 / W23）；E7 数据包改放 `offline_report.html`，生成失败时退回经典 `report.html`。

## 已完成的行（DONE，共 98 行）

- W1 显示对象：壁面 / 输入 STL / 中心线 / 预测点 — ws_shell.js:layersMenu + ws_display.js:layerItems
- W3 字段切换 — ws_shell.js:renderToolbar（seg-fields）、数字键 1–9
- W4 蓝白红色表 — ws_shell.js:layersMenu（`bwr`）+ ws_store.js CMAPS
- W5 色带分段 — ws_display.js:buildScale
- W6 手输固定上限、跨结果记住 — ws_display.js:fixedRow / applyRemembered
- W7 对数色标 — ws_shell.js:windowOptions（本例自适应 对数 / 线性）
- W8 Pa / dyn/cm² — ws_display.js:displayKind / toDisplay（概览、沿程、透镜、探针、区域共用）
- W10 等值线 — core_contour.js:marchingTriangles + core_viewer.js「P3 lane 4」块；菜单在 ws_display.js:layerItems
- W11 可信区域图例 — ws_display.js:trustRows / updateLegend
- W12 滞留区叠加 — ws_display.js:layerItems（滞留区斜纹）+ ws_overview.js:kpiModel
- W15 分支显隐 — ws_shell.js:branchDialog
- W16 高亮最高 x% — ws_display.js:buildTop + core_contour.js:topThreshold
- W17 剖切高度 — ws_display.js:buildClip / setClip + core_viewer.js P3 块 + adapter_wall.js（取点过滤）
- W20 自动标注 0 / 3 / 5 / 10 — ws_annot.js:findingsItem / chooseCount
- W21 标准视角 + 复位 — core_orientation.js + ws_display.js:resetView
- W24 快速统计 — ws_overview.js:kpiModel
- W25 结论 — ws_overview.js:narrativeModel
- W29 五模型一致性 — ws_notice.js:qualityModel + ws_overview.js:qualitySection
- W31 参考范围与人群位置 — ws_overview.js:qualityRows / qualitySection + ws_notice.js:openDetails
- W33 周期量卡 — ws_overview.js:cycleRows
- W34 分支统计表 — ws_overview.js:branchTableModel
- W36 计算过程与显示口径 — ws_overview.js:outletShareModel / geometrySection（盖面半径、Murray 分流、确认来源）+ ws_detail.js:techRows（显示插值）；top-5% 簇数由发现列表体现
- W37 几何参数表 — ws_overview.js:geometryModel / geometrySection
- W38 耗时与运行记录 — ws_detail.js:processSection
- W39 完整参数与版本 JSON — ws_detail.js:techDialog（完整统计 JSON、运行清单链接）
- W40 分支展开图 — ws_unroll.js + ws_detail.js
- W41 悬停读数 — ws_probe.js:hoverModel / showHover
- W43 探针卡 — ws_probe.js:card（sectionMeans）
- W44 探针记录 — ws_probe.js:logSection / tsv / csv
- W45 测量 — ws_measure.js:buildMeasurement
- W47 区域统计 — ws_region.js
- W48 发现审阅 — ws_overview.js 发现区 + ws_review.js
- W49 标注 — ws_annot.js:panel / add / saver
- W50 出版级 PNG — ws_figure.js:figure（「图片」页）
- W51 六视角分成 6 个文件 — ws_figure.js:downloadViews / setViewsOutput（zip）
- W52 色标 SVG — ws_figure.js:colorbarSVG / downloadColorbar
- W53 多视角拼图 — ws_figure.js:montage
- W54 一页纸配图 — ws_figure.js「一页纸」页（snapshots 上传）
- W56 视图保存到本浏览器 / 恢复 — ws_store.js:readView + ws_bookmarks.js
- W61 页脚 — ws_overview.js insp-status
- W62 技术信息 — ws_detail.js:techRows / techDialog（ws_overview.js 页脚按钮，各档和离线都有）
- W65 顶部警示条 — ws_notice.js:show / bar / openDetails
- W66 WebGL 回退 — ws_shell.js:showResult / stageMessage
- W70 离线单文件 — bundle.py:offline_html（offline_report.html）+ v2_offline.build_offline_html
- V1 显示页签 — ws_shell.js:layersMenu（流线、外壁、体内点）+ 字段 speed / pressure / wall_pressure + ws_slice.js
- V2 物理量切换 — ws_shell.js:renderToolbar
- V3 色标 + 分段 — ws_shell.js:layersMenu 色表 + ws_display.js:buildScale
- V5 轮廓透明度 — ws_display.js:buildVolume
- V6 单位 Pa / mmHg、m/s / cm/s — ws_display.js:buildScale（单位行）
- V7 速度对数色标 — ws_display.js:buildScale（对数）+ core_colormap.js:setLogHint / speedLogFloor + ws_slice.js
- V9 分支显隐 — ws_shell.js:branchDialog
- V10 自动标注 — ws_annot.js:findingsItem
- V11 速度箭头 — ws_display.js:layerItems
- V12 体内不可信点灰化 + 图例 — adapter_volume.js（desaturate 0.7）+ ws_display.js:updateLegend
- V13 流线粗细 / 密度 / 细线 — ws_display.js:buildVolume
- V14 发现 — ws_overview.js + ws_review.js
- V15 压降发现定位截面 — ws_detail.js:pressureDropSlice（ws_overview.js 行点击、ws_lens 按钮）
- V16 测量 — ws_measure.js
- V17 悬停简卡 / 单击全卡 — ws_probe.js:hoverModel + card（P = 悬停开关）
- V18 探针内容 + 截面积分 — ws_probe.js（stationSection / sectionIntegral）
- V19 探针记录 — ws_probe.js:logSection
- V20 标准视角 + 方向来源 — core_orientation.js + ws_detail.js:techRows（坐标架方向来源）
- V22 导出 PNG / 六视角 / 设置 — ws_figure.js
- V23 拼图（含截面） — ws_figure.js:montage
- V24 一页纸配图 — ws_figure.js「一页纸」页
- V25 截面点选 — ws_slice.js（点选、回到中心线）
- V26 X / Y / Z 轴截面 — ws_slice.js:axisPlane / setBasis
- V27 倾角 / 偏移数值 — ws_slice.js:setAngles
- V28 拖动方式工具条 — ws_slice.js:setDragMode（DRAG_MODES）
- V29 滚轮 / 按键微调 — ws_slice.js onWheel / key
- V30 自动截面 — ws_slice.js 站位行
- V31 按截面切割 — ws_slice.js「切开」
- V32 截面平面视图 — ws_slice.js 检查器截面区
- V33 放大图 — ws_slice.js:openZoom
- V34 截面系列拼图 — ws_slice.js:openSeries
- V35 结论 — ws_overview.js:narrativeModel
- V36 参考范围卡 — ws_overview.js:qualitySection + ws_notice.js:openDetails（超范围清单）
- V38 术语芯片 — ws_overview.js:volumeNotesModel / volumeNotesSection
- V39 压力参考、统计口径、流线提醒 — ws_overview.js:volumeNotesModel + ws_lens.js:pressureReference
- V40 沿程 — ws_profile.js
- V41 沿程点击跳截面 + 位置线 — ws_detail.js（V41 块，sliceStation）
- V42 两截面之间区域统计 — ws_slice.js「两截面之间」
- V43 标注 — ws_annot.js
- V45 顶部警示条 — ws_notice.js:show
- V46 页脚 + 技术信息 — ws_detail.js:techRows / techDialog
- O1 打开一页纸并打印 — ws_detail.js onKey（O）/ ws_figure.js「一页纸」页 / ws_onepage.js
- O2 一页纸里没有经典链接 — onepager.py
- O3 无配图提示 — onepager.py:NO_SNAPSHOTS_HINT（已改成「导出」→「一页纸」）
- O4 配图生成 — ws_figure.js（snapshots.json 同格式）
- O5 打包说明 — bundle.py README 文案（offline_report.html）
- C1 同步视角 — ws_shell.js:applySync（ns.viewer.link）+ ws_compare.js「同步视角」
- C2 同步口径 — ws_compare.js「同尺度看幅值」+ ws_shell.js:applyCompareScale
- C3 以左 / 右为准 — ws_display.js:pushSide
- C4 左右互换 — ws_shell.js:renderComparePanel（onSwap）
- C5 标量对照表 — ws_compare.js:scalarSection / fetchScalars / scalarModel（POST /api/compare）
- C6 类型不同提示 — ws_compare.js:conditions / sameScaleAllowed
- C7 两侧身份行 — ws_compare.js:identity
- C8 同一输入的 WSS 与体场并排 — ws_shell.js:enterCompare
