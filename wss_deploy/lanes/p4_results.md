# 迁移收尾轮 第 B 路 结果页补齐 · 交付报告

分支 `wss-ui-p4-results`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_c`，基线 e7c2356。2026-10-01。
依据：`PHASE3_LANES.md` §0、§4，`PHASE2_LANES.md` §1–§2；复核报告 `reaudit_reports.md` 的 W13、W27、W55、V37、C+1、W58 / W59 行。

## 1. 做了什么，入口在哪

### 1.1 经典链接带上等值线、剖切、高亮和速度对数（W55 ②，`ws_legacy.js`）

- 打开经典 `#view=` 链接时，下面几项不再列为「没有带过来」，而是转成第 4 路的链接状态 `x.display`（`ws_display` 的 `linkState` / `applyLinkState` 格式）：

  | 经典字段 | `x.display` | 读法（同经典 `applyViewState`） |
  |---|---|---|
  | `overlay.contours` | `contours` | 有 `overlay` 块就取 `!!contours` |
  | `highlight.top` | `top` | `!!top` |
  | `highlight.top_pct` | `topPct` | 夹到 0.5–10 % |
  | `slice.clip` | `clip` | 夹到 0–1，同一个 Z 比例（显示网格 z 范围）；1 = 不剖切 → `null` |
  | 体场 `log` | `speedLog` | 速度对数色标；压力不受影响 |

- 只放经典状态里写了的键（部分状态只改它写到的项）。
- 转换器仍是纯函数：新增 `convertDisplayExt(classic, family)`；`classicToV2` 的上下文多一个 `xDisplay`。页面里由 `pageContext` 判断：有 `id === 'display'` 且带 `linkState` / `applyLinkState` 的扩展才写 `x.display`。没有时行为与之前逐位相同，照旧列为「没有带过来」。

### 1.2 比较时右侧结果的警示（C+1，安全项；`ws_compare.js`、`ws_notice.js`）

- 「比较」页签：两侧身份下面，每一侧有警示时各一行，例如「右侧结果 注意：输入几何有 2 项超出发布包参考范围（…）；模型集成质量：…。」，带「详情」。
  - 句子用 `ns.notice.model`，即经典警示条的触发条件和原句。
  - 「详情」打开那一侧的「参考范围与质量」对话框。
- 视口标记：比较时，有警示的一侧状态行里出现「需复核」小标。
  - 悬停显示同一句话，点击打开那一侧的详情。
  - 切到别的页签也一直在；退出比较就消失（`ns.notice.badge`）。
- 不显示离散度数值：对话框还是只写「逐点离散度只保存在质量审计文件里」。

### 1.3 最大 / 最小值标记的位置（W13，`ws_overview.js`）

- 概览关键数字下面加一到两行，对应三维里的黄色（最大）和白色（最小，仅 TAWSS）标记，例如：
  - 「● 最大 11.9 Pa　左髂内，距入口 263 mm；距分叉 14 mm；局部半径 2.6 mm」
  - 「○ 最小 0.0932 Pa　主动脉，距入口 181 mm；距分叉 30 mm；局部半径 29.5 mm」
- 取点规则同经典 `fieldPeak`：
  - 峰值 WSS 用 `analysis.peak` 的点（分支、`s_from_inlet_mm`、`dist_to_junction_mm`、`local_radius_mm`）；
  - 其他字段取预测点上第一个最大的有限值（np.argmax 顺序），TAWSS 另取第一个最小值。
  - 这就是 `ws_display.peakMarkers` 画标记的那个点。
- 点一行，三维视角飞到那个标记。
- 预测点数组没加载时先加载一次，再重画概览。
- 文字跟当前字段走，单位跟 dyn/cm² 切换走。

### 1.4 三头结果看 WSS 时的全场 p99（W27，`ws_overview.js`）

- `kpiModel(manifest, fieldId)` 多了可选的当前字段参数。
- 有 TAWSS 的结果平时仍显示 TAWSS 那组；切到峰值帧 WSS 时，换成「WSS p99（`analysis.peak.p99_pa`，同经典 p99 卡）· 低 WSS 占比 · 高 WSS 占比 · 管腔最大直径」，最大值和位置在 §1.3 那一行。
- 单帧 X5D 结果不变。

### 1.5 体场统计小表（V37，`ws_overview.js`）

- 体场结果的概览关键数字下面加「体内统计」表：
  - 行：速度、相对压力；
  - 列：点数、均值、p99、最大。
- 数据取 `analysis.volume_statistics`（没有时退回字段自己的 statistics）。
- 单位跟色标面板的单位选择走（m/s ↔ cm/s，Pa ↔ mmHg）。
- ⓘ 写经典体场页的口径：体内预测点、点权重、相对压。

### 1.6 Ctrl+P 不再打出空白三维（W58 / W59，`ws_figure.js`）

- 有结果打开时按 Ctrl / ⌘ + P：拦下浏览器打印，改走已有的「打印当前视图」（`printView`，当前视图 + 色条 + 标题排成一页 A4）。
- 从浏览器菜单打印：
  - `beforeprint` 时，每个显示中的视口在同一个任务里重画一帧，复制到一张 2-D 画布，盖在 WebGL 画布上（`.ws-print-snap`，只在打印时显示，位置在标签之下）；
  - `afterprint` 时去掉。
  - `printView` 自己的 A4 页在打印时不做这一步。
- 首页（没有打开结果）不拦，照常是浏览器打印。
- W58「一键截图」：按复核建议不另做，仍走「导出 → 图片」。

## 2. 与经典页的一致性证据

沙箱：端口 8823 / Marionette 3023，病例 LV_GUO_YOU 周期指标（`20260929_230442_a7273f1670e4`）和体场（`20260929_230450_d1428792b846`），DEMO 随访 A / B。经典页直接用 `file://` 打开同一份生成的 `report.html`，逐项读它的 DOM。

| 项 | 经典页 | 新工作区 |
|---|---|---|
| W55 同一链接（TAWSS，等值线、最高 2 %、剖切 60 %） | 等值线 ☑，最高 ☑，2 %，剖切 60（Z ≤ -232 mm） | contours true，top true，topPct 2，clip 0.6（clipZ -231.9 mm，1170 段等值线，高亮 1529 点）；提示里不再出现「等值线、剖切、高亮」 |
| W13 TAWSS 最大 | 11.9 Pa；左髂内，距入口 263 mm；距分叉 14 mm；局部半径 2.6 mm | 同左，逐字相同 |
| W13 TAWSS 最小 | 0.0932 Pa；主动脉，距入口 181 mm；距分叉 30 mm；局部半径 29.5 mm | 同左，逐字相同 |
| W27 峰值帧 p99 卡 | p99 17.0 Pa，全场最大 52.6 Pa；左髂内，距入口 263 mm；距分叉 14 mm；局部半径 2.6 mm | KPI「WSS p99 17.0 Pa · 低 41 % · 高 17 %」；位置行「最大 52.6 Pa 左髂内，距入口 263 mm；距分叉 14 mm；局部半径 2.6 mm」 |
| V37 速度 | 预测点数 20000，均值 0.422，p99 1.37，最大 1.57 m/s | 20000 · 0.422 · 1.37 · 1.57 |
| V37 压力 | 20000，-304，99.1，131 Pa | 20000 · −304 · 99.1 · 131 |
| C+1 右侧（DEMO B） | 经典比较页右侧完整报告的警示条 | 「比较」页签一行 + 右视口「需复核」；句子等于 `ns.notice.model`（2 项超范围 + 集成质量待复核） |

自动测试（`tests/test_v2_p4_results.py`）也逐项对经典代码：

- W13：把 `report.py` 的 `fieldPeak`、`branchName`、`DERIVED_DEF` 原样切出来，在同一份嵌入数组上跑。WSS / TAWSS / OSI / RRT / ECAP 的最大值和 TAWSS 最小值，位置句逐字相同，数值相同，显示文字相同。
- V37：`VolumeViewerCore.statistics` 加 `insideIndices` 加 `speedField`，按经典 `formatValue` 格式化后逐格相同。
- W55：输入由经典编码器 `report.py viewHash` 和 `VolumeViewerCore.encodeView` 生成，用真实的 `ws_display` 走完整个打开流程。

打印：用 Marionette 的打印（会触发 `beforeprint` / `afterprint`）出 PDF。

- 有拷贝时血管完整打印出来（`12_w59_print_marionette.pdf`）。
- 拒绝 2-D 画布、做不出拷贝时，三维区是空白的（`13_w59_print_without_snapshot.pdf`），这就是原来的问题。
- Ctrl+P 走 A4 打印页（`07_w58_ctrl_p_print_view.pdf`），打印后 `ws-printing` 和打印页清干净。

## 3. 与经典的差异

1. **位置行的位置**：经典写在统计卡的「最大值位置 / 黄色标记」两行里；这里放在概览关键数字下面，一到两行，鼠标悬停说明是哪个标记。三维标记上的数值小签不变（`ws_display` 不属本路）。
2. **体场统计**：经典只显示当前物理量那一张；这里速度、压力两行一起列。数字取服务端 `volume_statistics`，与经典页在浏览器里现算的 `statistics` 相比：
   - 均值的差别在 1e-8 量级（float32 累加），显示到三位有效数字时完全相同；
   - 负号用排版负号「−」，与工作区其他数字一致。
3. **Ctrl+P**：经典是「截图 + 统计卡两栏」的打印样式。这里 Ctrl+P 走第 5 路的单页 A4 打印（视图、色条、标题）；菜单打印是整页，三维区是快照。
4. **比较时的警示**：经典是左右两份完整报告各带警示条（§0.1 已裁定不做）。这里只补警示本身，属于安全项。顶部警示条仍只说左侧，左右两侧各有「需复核」小标，不会混淆。

## 4. 改动过的公共文件

- `wss_deploy/static/v2/ws_shell.js` `renderStatusLine`：比较时，左右视口状态行各加一个子元素 `ns.notice.badge(manifest, {side})`，两处各一行，标注 `// P4 lane B (C+1)`。
  - 外壳没有「进入比较」或「状态行」的扩展钩子；状态行每次重画都会清空子元素，从外面插不进去，所以只能改这里。
- 其他都在本路文件：`ws_legacy.js`、`ws_notice.js`、`ws_compare.js`、`ws_overview.js`、`ws_figure.js`、`v2_detail.css`、`v2_export.css`，以及新测试 `tests/test_v2_p4_results.py`。
- `ws_overview.js` 注册了一个只取外壳 API 的扩展 `{id: 'overview', onStart, onResult, onField}`。原因：概览要拿到打开的结果和 `flyTo`，原来要借 `ns.detail.shell()`，那是 A 路的文件。没有 `linkState`，不影响复现链接。
- 没有改 Python，没有改 `bundle.json` / `index.html`。

## 5. 测试

- 改过的 js 都过 `node --check`。
- 新增 `tests/test_v2_p4_results.py`，13 项：W55 转换与页内落地、上下文探测、C+1 页签与视口、W13 / W27 / V37 对经典代码、概览渲染、打印快捷键与快照、打印样式。已有测试没有改动。
- 全量 `PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests`：**1000 通过，33 跳过**。
  - 跳过全部是本工作副本里没有 `outputs/wss_deploy_jobs` 开发病例或中心线任务（`test_v2_routes`、`test_v2_data`、`test_v2_offline`、`test_wss_features_equivalence` 等），与本路无关。
- 沙箱走查 0 个 JS 错误（`shots/console.log`）。
  - 另一次只做打印对照的短跑，退出时记到一条 Firefox 自身的 `ASRouter … NEWTAB_MESSAGE_REQUEST`（浏览器界面，不是页面脚本）。
- 只改前端，不涉及黄金回归。

## 6. 截图与产物

目录 `/public/newhome/cy/.claude/jobs/83844143/tmp/p4_laneB/shots/`：

- `00_classic_wall.png`、`01_classic_volume.png`：经典页对照；
- `02_w55_classic_link.png`：经典链接在新工作区打开（剖切 60 %、最高 2 % 高亮、等值线）；
- `03_w13_tawss_overview.png`、`04_w13_fly_to_min_marker.png`、`05_w27_wss_kpis.png`；
- `06_w59_page-1.png` / `.pdf`、`12_w59_page-1.png` / `.pdf`、`13_w59_page-1.png` / `.pdf`（菜单打印有快照与无快照的对照），`07_w58_page-1.png` / `.pdf`（Ctrl+P 打印页，Marionette 默认 Letter 纵向）、`14_w58_page-1.png` / `.pdf`（Ctrl+P 打印页，按 A4 纸）；
- `08_v37_volume_stats.png`；
- `09_c1_compare_tab.png`、`10_c1_badge_details.png`、`11_c1_badge_on_other_tab.png`；
- `walk.json`：走查读到的全部数值。

走查脚本：`walk.py`、`walk_print.py`、`walk_a4.py`（同目录上一级）。沙箱（两次启动）都已按进程号停掉。

## 7. 没做完 / 留给后面

- 三维标记小签本身（`ws_display` 的 `markers` 文字）仍只写「最大 x Pa」，位置在概览那一行。如要写进小签，由 `ws_display` 的负责人加一个 title 或第二行。
- W55 ①（测量、探针记录进链接）仍按复核建议「以后再说」。
- 菜单打印是整页加快照，不是经典的「图 + 统计卡」两栏。要那种版式，用一页纸或 Ctrl+P 的 A4 页。
- Marionette 打印不用页面的 `@page A4` 设置，默认按 Letter 纵向出纸。这时 Ctrl+P 打印页里偏宽的图右边会被裁掉（`07_…`，打印页版式属第 5 路 `printView`）。
  - 按 A4 纸打印完整（`14_…`）；
  - 真实浏览器按 `@page` 排版，仍建议在真实浏览器里打一次确认。
