# D 路（S6d）结果详情与分析视图 · 交付报告

分支 `wss-ui-s6d-detail`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_d`。2026-09-30。

## 1. 做了什么，入口在哪

### 1.1 检查器新页签「沿程」（`ws_detail.js` + 新模块 `ws_profile.js`、`ws_unroll.js`）

- 一列三条，共用一根横轴（距主动脉入口的弧长）：
  1. 当前字段的沿程曲线：`analysis.profiles` 的 2 mm 分箱；
     - 系列与阈值线跟经典 `profSpec` 一致：WSS p99 / 均值；TAWSS 均值 / 最低（0.4 Pa 线）；OSI、RRT、ECAP p90 / 均值（各自第一档阈值线）；体场速度截面平均 / 最大，压力截面平均 / 最低；
     - 沿程数据里没有当前字段时（如 ECAP），显示峰值 WSS 并用一行字说明。
  2. 中心线半径条。
  3. 分支展开图（只有壁面结果）：用 `geometry.points` 的 `p_s` / `p_th`，映射公式与经典 `drawUnroll` 相同，每格取落入的预测点最大值。颜色直接用三维视口当前色标（`viewer.colorbarInfo().scale`），换字段、换色标窗后自动重画。
- 选分支用下拉框；ⓘ 里写口径。
- 悬停：三条上同时出竖线，底部读数条给「距入口 · 两个统计量 · 半径」或「距入口 · θ · 值」。
- 点曲线：
  - 游标（G）打开时移动游标（三维环、HUD 跟着动）；
  - 游标关闭时把该 2 mm 箱在血管上高亮（其余变淡），箱不在画面里时相机飞过去；
  - 读数条上有「来源」（证据透镜，口径与游标相同）和「清除」。
- 点展开图：选中最近的预测点，飞到那里。「来源」打开该点的证据透镜，透镜里带周向角。
- 游标联动：游标移动时三条上的游标线跟着走；游标换分支时页签跟着换分支。离开页签时撤掉箱的高亮。
- 右上两个按钮：
  - 「放大」：对话框里逐分支显示全部展开图（经典「分支展开图」卡的样子），悬停读数，点一下关闭对话框并飞到该点，带横向色条；
  - 「导出」：
    - 沿程曲线 SVG（本分支，字段一张、半径 / 管腔直径一张），走 `WssReportCommon.profileSVG`，系列与经典 `profileSVGs` 相同；
    - 展开图 PNG（全部分支，带标题、±180° 标注、弧长范围和色条）。
- 离线报告（`file://`）里页签照常可用；体场结果只有曲线和半径，没有展开图。

### 1.2 概览补全（`ws_overview.js`）

- 周期量：关键数字下加一排小卡——OSI、RRT、ECAP 的均值和「高于第一档阈值的占比」条，RRT / ECAP 标「派生」。点击进证据透镜，透镜里同时给均值和占比。
- 形态：在分区图下面加一条主动脉管腔直径曲线。
  - 曲线用可靠站的管腔最大直径；图上有瘤颈、瘤体阴影，参考直径虚线，最大值点。
  - 曲线下四个数（瘤体长度·体积、瘤颈长度·直径、参考直径、全腔体积），每个都能点进透镜。
  - 再下一行是截面可靠性（「N 站重定向，M 站不可靠」），逐分支明细放在 ⓘ 里。
  - 体场结果放在「形态」区块里，同一个样子。
- 逐分支表：在分区图的「表格」里，跟随当前字段。
  - 峰值 WSS 用 `analysis.per_branch`，列为 p99、均值、最大、< 0.4、> 4、cm²；
  - TAWSS / OSI / RRT / ECAP 用 `cycle.fields[id].per_branch`，列为均值、p99、最大、阈值占比；
  - 每个数可点，悬停一行时血管上标出该分支。
- 随访：在「随访」区块加两条小曲线（管腔最大直径和本结果的指标），当前扫描实心。点其他扫描的点就打开那次扫描的同类结果。原表格收进「表格」。

### 1.3 「工具」页（`ws_detail.js`，通过扩展点 `tools()` 加在末尾）

- 病例信息：显示患者编号、扫描、标签、备注；「编辑…」打开对话框。
  - 可改病例名称、患者编号、扫描标签、扫描日期、标签（逗号或顿号分隔）、备注；
  - 校验与经典相同：不可见字符报错、超长报错、像真实姓名时给提示、最多 12 个标签、每个标签不超过 40 字；
  - 调 `POST /api/jobs/<id>/metadata`，409 时读回；
  - 复核锁定时不能编辑。
- 出口命名：用彩色小点列出当前命名，另有来源和把握度。「修改并重算…」在检查器里展开编辑：
  - 下拉表、左右互换、左侧内 / 外、右侧内 / 外、恢复；
  - 三维视口里用工具标签标出 `#3 左髂外` 等；
  - 勾选核对后调 `POST /confirm {mapping, acknowledged: true, override: true, stage: 'B', version}`，与经典覆盖路径的载荷相同；
  - 提交后页面进入计算进度，算完自动回到结果。
- 计算过程：流水线各阶段的分段条，已预计算的阶段用浅色斜纹表示。
  - 下面一行：本次尝试、排队、人工确认、后台预计算（不计入）、任务历时、设备；
  - 「运行记录」展开最近 12 条事件，文案沿用经典 `EVENT_TEXT`，并补了经典没有命名的动作。
- 输入检查：读已完成任务记录里的阶段 A 输入检查；旧记录缺这部分时退回 `/api/v2/jobs/<id>/inputcheck`。
  - 显示评级、单位、外包、壁面面积、面数、开口数，以及各档计数；
  - 「全部 N 项」展开检查清单；
  - 「在三维标出开口」在视口里标出每个开口和它的半径。
- 删除：「删除任务…」→ 确认 → `POST /api/jobs/<id>/delete {version}` → 回首页，提示条上有「撤销」（`POST /api/trash/<id>/restore` 后重新打开）。复核锁定或计算中时不能点。

### 1.4 快捷键（通过扩展点 `key()`）

| 键 | 作用 |
|---|---|
| N | 新建（打开上传对话框） |
| / | 聚焦病例栏搜索框（病例栏收起时先展开） |
| O | 新窗口打开一页纸 |

- 离线报告里这三个键不响应。
- 快捷键表（?）里已补上这三行。

## 2. 与经典页面的一致性证据

均在沙箱（:8824）里用同一任务对照。原始输出在 `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_d/parity/`。

1. 展开图逐像素：`unroll_parity.py`，结果在 `unroll_parity_x5d.json`。
   - 任务 `20260920_144135_673ccd0e36b1`，7 条分支，600 × 180。
   - 经典 `drawUnroll` 公式用 numpy 逐字复刻，与 `ws_unroll.grid()` 相比，每条分支的全部像素相同，例如：
     - 主动脉像素 (152, 87)：2.0314 = 2.0314；
     - 左髂外 (376, 78)：8.9165 = 8.9165。
   - 每条分支的点数、s_min、s_max 与报告内嵌的 `unroll_branches` 完全相同。
2. 沿程曲线 SVG：`classic_vs_v2.py`。
   - 任务 `20260929_230442_a7273f1670e4`：在经典报告里点「导出曲线 SVG」截下的文件，与新工作区 `exportSVGs` 生成的字符串逐字节相同；
   - 字段 WSS、TAWSS、OSI 各两张（字段 + 半径），6 / 6 相同；
   - 文件：`<job>_classic_*.svg` 与 `<job>_workspace_*.svg`。
3. 逐分支表（同一任务），结果在 `classic_vs_v2_a727*.json`：
   - WSS：经典「分支统计」表与新表逐项一致。例如主动脉 p99 4.93 / 均值 0.784 / 最大 8.25 / 低 52% / 高 2% / 面积，面积经典写 259.8 cm²，新表写 260 cm²（三位有效数字）；
   - TAWSS、OSI：经典周期卡的分支表（均值、p99、阈值占比）与新表逐项一致；
   - 占比格式有差别：经典取整（0），新工作区小于 1% 时留一位小数（0.3%）。这是 §7 的数字规则。
4. 周期量、形态、沿程读数都与经典同源，数值一致：
   - 周期量：OSI、RRT、ECAP 取自 `cycle.fields`，与工作台 `cycleBlock` 同源。重算前截图是 OSI 0.135（57% > 0.1）、RRT 5.41 1/Pa（49% > 5）、ECAP 0.526 1/Pa（5% > 1.4），与 summary 相同；
   - 形态：全腔体积 347.5 → 348 mL，参考直径 18.96 → 19.0 mm，瘤体 87 mm · 225 mL，瘤颈 72 mm · Ø20.0 mm，截面可靠性「22 站重定向，0 站不可靠」，与工作台 `morphologyBlock` / `reliabilitySummary` 同算法；
   - 沿程读数：数值直接取 `profiles` 的分箱，与游标 `readout()` 同一分箱。

## 3. 与经典页面的差异和原因

- 展开图格子数：经典一个画布像素一格。点少的分支（髂动脉 2–3 千点）在经典里成了散点，因为预测点是一圈一圈排的，列比圈多就会空列。
  - 新工作区的格子数跟点数走：平均约 4 点一格，格子在管壁上接近正方形（分支长度 ÷ 周长，周长用分箱半径的中位数），列数不多于点的圈数，也不多于屏幕像素；
  - 映射公式和取最大值的定义没变（见 §2.1），显示时按像素放大；
  - 格数写在展开图标题右侧（如「195 × 75 格」）。
- 展开图里值缺失（NaN）的点跳过。经典的 `Math.max` 会把这一格变成 NaN（空白）。目前的数据里没有 NaN，结果相同。
- 展开图颜色用三维视口当前色标（窗、对数、色表都跟着），经典用报告自己的 vmax。数值相同，颜色与工作区三维一致。
- 逐分支周期量表多一列「最大」（数据里本来就有）。
- 阶段名「inference_5_models」写成「模型预测」，经典写「五模型预测」：周期结果是 3 个模型，「五」不准确。
- 流水线分段条里不画「后台几何预计算」：它在确认出口期间运行，不是流水线的阶段，放在下面一行里注明「不计入」。
- 沿程点曲线时，箱不在画面里会把相机飞过去（经典只做标记）。
- 随访小曲线用 v2 已有的指标选择（周期结果优先 TAWSS 均值），经典工作台时间线画的是 WSS p99。

## 4. 改动过的公共文件

| 文件 | 改动 |
|---|---|
| `static/v2/ws_shell.js` | ① `SHELL_API` 的 `// lane D` 之后加 `openLens(ref)`、`openUpload()`、`focusSearch()`（病例栏收起时先展开，再聚焦搜索框） |
| | ② `metadataDialog()` 开头：存在 `ns.detail.metadataDialog` 时交给 D 路（带标签和备注） |
| | ③ `renderTools()` 里完整档自带的「病例信息」区块：D 路存在时不再显示（避免重复，D 路两档都显示） |
| | ④ `shortcutsDialog()` 在「Esc」前插入 N、/、O 三行（仅 `ns.detail` 存在时） |
| `static/v2/bundle.json`、`static/v2/index.html` | 在 `ws_detail.js` 正下方加 `ws_unroll.js`、`ws_profile.js`，两文件同时加，不进 `offline_exclude` |
| 测试 | 新文件 `tests/test_v2_detail.py`；没有改已有测试 |

- 名下文件 `ws_overview.js`、`ws_lens.js`、`ws_detail.js`、`v2_detail.css` 另改。
- `ws_lens.js` 新增：分支统计的标签 / 占比 / 点数 / 面积、点的周向角、统计量附加行、形态位置、分箱「距入口」。
- 服务端没有改动，全部用现有接口：metadata、confirm（override）、delete、trash restore、inputcheck、timeline、onepage。

## 5. 测试结果

- `node --check`：`ws_detail.js`、`ws_unroll.js`、`ws_profile.js`、`ws_overview.js`、`ws_lens.js`、`ws_shell.js` 全过。
- 全量 `pytest tests`：880 通过 / 33 跳过（最后一次改动后重跑，3 分 36 秒）。
- `tests/test_v2_detail.py` 11 项：
  - 纯逻辑：展开图与经典公式逐格一致；沿程系列与 SVG 导出；概览模型；工具页模型；证据透镜新引用；
  - 外壳集成（工作区桩 + D 路文件）：「沿程」页签与箱固定；工具页元数据、出口覆盖载荷、删除与撤销；快捷键；分支表跟随字段；离线模式有「沿程」、无「工具」、N / O 不响应。
- 相关 v2 测试子集（workspace / review / probe / slice / offline / routes / examples）在最后一次改动后重跑：47 通过 / 12 跳过。
- 沙箱浏览器（Firefox 无头，Marionette 3024）走完全部新功能，0 个页面 JS 错误。
  - 一次截图脚本里出现过 `ASRouter … NEWTAB_MESSAGE_REQUEST`，那是 Firefox 自身在关闭时报的，不是页面错误。
- 端到端出口改名重算：`20260929_230442_a7273f1670e4` 左侧内 / 外互换 → 阶段 B 在 CPU 上重算完成 → 自动回到结果，新命名生效。

## 6. 截图（绝对路径）

目录 `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_d/shots/`：

| 文件 | 内容 |
|---|---|
| `01_overview.png` | 周期结果概览：OSI / RRT / ECAP 小卡 |
| `01b_overview_scrolled.png`、`38_cycle_overview_morph_final.png` | 分区图下的管腔直径条、四个形态数、可靠性 |
| `02_along.png` | 「沿程」页签：TAWSS 曲线 / 半径 / 展开图 |
| `03_along_hover_map.png`、`04_along_click_map.png` | 展开图悬停读数，点击飞到该点 |
| `05_along_click_profile.png`、`07_profile_bin_fly.png` | 点曲线固定分箱，血管上高亮并飞入 |
| `06_lens_bin.png` | 分箱的证据透镜 |
| `08_unroll_dialog.png`、`11_unroll_dialog_phys.png` | 全部分支展开图对话框（后一张为最终格子规则） |
| `09_overview_branch_table.png`、`10_overview_branch_table_wss.png` | 逐分支表（TAWSS / WSS） |
| `12_tools.png`、`13_tools_bottom.png` | 工具页：病例信息、出口命名、计算过程、输入检查、删除 |
| `14_metadata_dialog.png`、`15_tools_case_saved.png` | 编辑病例信息（标签、备注）与保存后 |
| `16_tools_process_input.png` | 运行记录、输入检查清单、三维开口标注 |
| `17_key_n_upload.png`、`18_shortcuts.png` | N 键上传、快捷键表 |
| `19_followup.png` | 随访曲线 |
| `20_volume_overview.png`、`37_volume_overview_final.png` | 体场概览形态区块 |
| `21_volume_along.png`、`22_volume_along_bin.png`、`35_volume_overview_1024.png` | 体场沿程（1024 宽） |
| `23_x5d_along_cursor.png`、`24_x5d_along_cursor_click.png` | 游标联动 |
| `25_outlet_edit.png`、`27_override_running.png`、`28_override_done.png` | 出口改名：编辑、重算中、完成 |
| `29_delete_confirm.png`、`30_deleted_toast.png`、`31_restored.png` | 删除确认、回收站提示、撤销恢复 |
| `32_outlets_note.png` | 出口命名区块 |
| `33_along_1024.png`、`34_along_1024_osi.png` | 1024 宽下的沿程页签（OSI） |
| `36_offline_along.png` | 离线报告里的沿程页签 |
| `export_LV_GUO_YOU_unroll_tawss.png`、`export_LV_GUO_YOU_profile_主动脉_*.svg` | 导出文件 |

## 7. 没做完 / 需要主会话处理的事

1. **经典工具说明要改**：`ws_shell.js` 的常量 `CLASSIC_TOOLS`（工具页「经典报告里的工具」那句）仍写着「分支展开图」。A 路也会改这句，我没动，合并时请去掉「分支展开图」。
2. **预览任务的数据问题**（不是本路代码的问题）：预览任务 `20260920_144135_673ccd0e36b1`（X5D）的阶段 A 记录里，`input_check.clean_stl` 是另一个会话临时目录下的绝对路径（`…/8d4c62cc-…/scratchpad/regress/baseline/…/input_clean_mm.stl`）。
   - 对它做出口改名重算，会在阶段 B 报 FileNotFoundError；
   - 我沙箱里的这份副本因此停在「失败」，原始预览目录没有动；
   - 周期结果 `a7273f1670e4` 的是相对路径，重算正常。
3. 出口改名重算只重算当前结果；同时预测的体场结果不会自动跟着重算（服务端现有规则），工具页 ⓘ 里写明了。
4. 「沿程」存进偏好（`prefs.tab`），下次打开结果会停在「沿程」（与「书签」「工具」一样）；如果希望每次都回到概览，需要在外壳的暂态页签列表里加上 `along`。
5. 展开图的 θ = 0 参考方向在 manifest 里没有说明，只标了 ±180° 和 0°。
6. 分支显隐（分支显隐对话框）目前不影响「沿程」页签，下拉框里仍列出全部分支。
