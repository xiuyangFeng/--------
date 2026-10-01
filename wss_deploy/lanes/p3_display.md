# 第三期第 4 路「显示与读数」· 交付报告

分支 `wss-ui-p3-display`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_d`，基线 866cf02，2026-10-01。

一句话：`PHASE3_LANES.md` §3 第 4 路的九项都做了，覆盖审计行 W4、W6、W8、W10、W11、W16、W17、W20、W22、W23、W41、V7、V12、V17、V26、V27、V28、V44。所有新选项只改画法，不改任何存储的数。新选项全关时，三维画面与基线逐字节相同（§2.3）。

## 1. 做了什么，入口在哪

| 审计行 | 功能 | 用户从哪里用 | 代码 |
|---|---|---|---|
| W41、V17 | **悬停读数**：鼠标停在血管上时，指针旁出现小卡片。<br>壁面：当前字段在最近顶点的插值值（就是看到的颜色），三角形有一个顶点无值时写「插值覆盖范围外」；另有分支、距入口、可信提示、顶点坐标。<br>体场：体内点的速度和相对压力，以及分支、距入口、可信提示 | 默认开。P 键或图层菜单「悬停读数」开关 | `ws_probe.js`（`hoverModel`、扩展 `hover`） |
| W8 | **WSS 单位 Pa / dyn/cm²**（经典 `CORE.UNITS`，× 10）。<br>跟着换的：色条与阈值标签、色标窗下拉、概览关键数字、发现、证据透镜、最大 / 最小值标记、探针卡、悬停读数、区域统计、色标面板的阈值和上限输入。<br>只换 Pa 的量（WSS、TAWSS）；OSI、RRT、ECAP 不换，与经典 `isPa` 相同 | 色条旁「调整」→「单位」 | `ws_display.js`（`displayKind`、`toDisplay`）、`ws_probe.js`、`ws_region.js` |
| V7 | **速度对数色标**：下限 = max(范围下限, 上限 / 200, 0.001 m/s)，直接调用经典 `VolumeViewerCore.scaleEnds`。<br>打开后色标窗下拉显示「本例自适应（对数）/（线性）」。截面的速度色标（本截面、手动、与三维同、截面系列）也按对数 | 体场速度的色标面板 →「对数」 | `core_colormap.js`（`setLogHint`、`speedLogFloor`）、`ws_slice.js` |
| W4 | **蓝白红**色表 | 图层菜单「色表」→「蓝白红」 | `ws_shell.js` 一行、`ws_store.js` 一处（见 §4） |
| W6 | **手输固定上限**：色标改为 0 至上限，按显示单位输入。勾「所有结果」后按字段记住，以后打开的结果这个字段从它开始（经典「固定上限 · 跨报告保留」）。空着即回到本例自适应 | 色标面板 →「上限」（只在主视口、无负值的字段） | `ws_display.js`（`fixedRow`、`applyRemembered`） |
| W10 | **等值线**：在色条上的三个阈值处画线，阈值取自定的或发布包的。<br>算法从 `report.py` `REPORT_CORE_JS` 逐式移植（`marchingTriangles`），在显示网格、显示数组上计算，跟随分支显隐 | 图层菜单「等值线」，提示里写着当前阈值 | `core_contour.js`、`core_viewer.js` P3 lane 4 块 |
| W16 | **高亮最高 x%**（0.5–10%，步长 0.5）：经典 `topThreshold` 的门槛，品红点，隐藏的分支不画。<br>面板里写门槛和点数，如「≥ 4.36 Pa · 765 个预测点」 | 图层菜单「高亮最高 1%…」 | 同上 |
| W17 | **剖切高度**：壁面结果沿 STL 的 Z 轴只留切面以下。<br>悬停、点击探针、测量和区域取点都只读留下的部分；切面以上的标记和标签隐藏 | 图层菜单「剖切…」 | `core_viewer.js` P3 块、`adapter_wall.js`（剪裁感知的轮廓线、斜纹着色器，取点过滤） |
| W20 | **自动标注发现**：关 / 前 3 / 前 5 / 前 10 条。<br>同时把 `ws_annot.js` 空标注提示里的「经典报告」去掉 | 图层菜单「发现标签…」，弹出四个选项 | `ws_annot.js`（`findingsItem`）、`ws_shell.js` 一行 |
| V26 | 截面**垂直 X / Y / Z 轴**：经典 `currentPlane`，过显示网格包围盒中心，位置 = 最小值 + 比例 × 范围。切换时从当前截面所在坐标开始 | 截面页位置行最前面的下拉「中心线 / 垂直 X 轴 / Y / Z」 | `ws_slice.js`（`axisPlane`、`setBasis`） |
| V27 | **倾角、偏移输入数值**：绕横轴、绕纵轴 ±85°，左右、上下 ±30 mm，与经典滑杆同范围。拖动或按键时数值跟着变 | 截面页控件区「倾角」「偏移」 | `ws_slice.js`（`setAngles`） |
| V28 | **拖动方式**：移动 / 旋转 / 平移三段切换。触屏也能用，鼠标仍可用 Shift / Alt | 截面页控件区「拖动」 | `ws_slice.js`（`setDragMode`） |
| W11、V12 | **可信度图例**：视口右下角列出各类的占比，规则放在 ⓘ 里，壁面斜纹与体内灰点用不同色样。<br>**体场体内点灰化**：位 4 / 8 / 16，经典 `desaturate(c, 0.7)` | 图层菜单「可信度标记（斜纹）」 | `ws_display.js`（`trustRows`、`updateLegend`）、`adapter_volume.js` |
| W22、W23、V44 | **经典设置**：读出 `/api/preferences` 中经典报告存的 `report_defaults.wall / volume` 和 `presets.wall / volume`。<br>默认口径「套用」：写进本机显示设置（色表、分段、单位、WSS 阈值、体场外壁）。<br>预设「应用」：按它显示当前结果，包括字段、色表、分段、阈值、单位、固定上限、可信度、等值线、高亮、剖切、自动标注和分支显隐；视角只在预设存自同一份结果时带上。带不过去的项写在提示条里 | 工具页「经典设置」（在线时） | `ws_display.js`（`defaultsPlan`、`applyDefaults`、`applyPreset`、`classicSection`） |

**与第 5 路复现链接的衔接**（应第 5 路要求）：

- `ws_display` 扩展（id `display`）实现 `linkState(api)` 和 `applyLinkState(state, api)`。
  - 状态为 `x.display = {v: 1, contours, top, topPct, speedLog, clip}`。
  - 旧状态、不完整的状态只改它写到的项；格式不对的整个忽略。
- WSS 单位存在 `prefs().units.wss`，所以会自动进第 5 路的 `d.units`，打开时第 5 路调用 `setUnit('wss', …)`。
- 固定上限本身就是色标窗，由第 5 路的 `w` 带上。
- 第 6 路转换经典 `#view=` 时的对照：
  - `units: 'dyn'` → `d.units.wss = 'dyn/cm²'`
  - `overlay.contours` → `x.display.contours`
  - `highlight.top` / `top_pct` → `x.display.top` / `topPct`
  - `slice.clip`（< 1）→ `x.display.clip`
  - 体场 `log` → `x.display.speedLog`

## 2. 与经典页面的一致性证据

### 2.1 浏览器里逐项比对

沙箱 :8824，同一份任务数据；经典页在同一沙箱的 `/api/jobs/<id>/report` 打开。

| 项 | 经典页 | 新工作区 | 结论 |
|---|---|---|---|
| dyn/cm²（峰值 WSS 结果 …6ccd0e36b1） | 空间 p99 166 dyn/cm²；全场最大 486 dyn/cm²；低 WSS 43%；色标 0 – 166 dyn/cm² | 关键数字「WSS p99 166 dyn/cm²」「低 WSS 占比 43%」；最大值标记「最大 486 dyn/cm²」 | 一致 |
| 剖切 55%（同一结果） | 「Z ≤ -245 mm」 | z₀ = -244.94 mm，「Z ≤ -245 mm」 | 一致 |
| 截面：垂直 Z 轴、40%（LV 体场） | 预测点 285；均值 0.0343、p99 0.0694、最大 0.0719 m/s；轮廓面积 3703 mm²；最大径 71.1 mm；等效直径 68.7 mm | 按点统计 285 点，均值 0.0343、p99 0.0694、最大 0.0719 m/s；面积 3703.39 mm²；最大径 71.1 mm；等效直径 68.67 mm；法向 [0, 0, 1] | 逐项一致 |
| 速度对数下限 | 经典体场图例 0.0079 – 1.57 m/s（经典全局范围是 min–max，1.57 / 200 = 0.0079） | 0.00686 – 1.37 m/s（新工作区的本例自适应是 0–p99，1.3719 / 200 = 0.00686） | 公式相同（同一个 `scaleEnds`）；上端不同是第二期就有的色标窗差异，见 §3 |
| 等值线、高亮（LV 周期结果，TAWSS） | 经典页面不显示这两个数，改在 Node 里用经典 `REPORT_CORE_JS` 原样计算，见 §2.2 | 页面画出 1170 段线；最高 1% 门槛 4.36 Pa、765 个预测点 | 与 §2.2 的经典计算相同 |

### 2.2 Node 中与经典 `REPORT_CORE_JS` 逐位比对

真实病例 LV_GUO_YOU 周期结果，数组取自它的 `report.html`；测试 `test_iso_lines_and_highlight_on_a_real_case_equal_the_classic_page`。

- **WSS 等值线**（阈值 0.4 / 4 / 7）：513 / 1052 / 1166 段，坐标逐位相同（`Object.is`）。
- **TAWSS 等值线**（阈值 0.4 / 4 / 7）：645 / 425 / 100 段，合计 1170，与页面上画出的段数相同。
- **最高 x% 门槛**：
  - WSS 1%：17.0076 Pa，765 点；
  - TAWSS 1%：4.3577 Pa，765 点；
  - TAWSS 2%：3.4223 Pa，1529 点。
  - 门槛逐位相同，点数等于经典同口径计数。
- **合成网格**：含 NaN 洞和顶点恰在阈值上的情况，结果同样逐位相同。

### 2.3 选项全关时画面不变

做法：从基线 866cf02 导出一份代码，起第二个沙箱（:8834，用完已按 PID 停掉）。两边用同一窗口尺寸、标准前视，取 `viewer.snapshot()` 的 PNG 字节比较。

- **新选项全关的 7 张**：LV 周期 TAWSS / WSS / OSI，体场速度 / 压力 / 壁面压力，峰值 WSS。SHA-1 全部相同。
- **另 3 张，打开轮廓线和可信度斜纹**：剪裁感知的着色器在没有剖切时与原着色器输出相同，SHA-1 也相同。
- 快照文件：`/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane4/shots/pix/`。

## 3. 与经典页面的差异和原因

1. **悬停读数**：
   - 经典是视口左下角的固定提示框，这里是跟随指针的小卡片。
   - 壁面同经典，读的是显示插值（颜色）；点击探针仍是最近的预测点（经典同样如此）。
   - 体场的简卡只给速度、相对压力、分支、距入口和可信提示；完整卡片仍在点击后出现。
2. **dyn/cm² 没覆盖的地方**：
   - 「沿程」页的 WSS 曲线：`ws_profile.spec` 只换算体场。
   - 概览的分区图和分区表：`ws_overview` 直接写原值。
   - 这两个文件归第 3 路，只需在格式化壁面值时调用 `ns.display.toDisplay(v, units, 'wall')`，建议合并时补。
   - 探针记录、CSV / TSV、所有导出：仍是 Pa，与第二期体场单位的口径相同。
3. **速度对数色标的上端**：经典体场全局色标是 min–max，新工作区的本例自适应是 0–p99（第二期 E 路的口径）。所以同一病例的对数下限不同：0.0069 对 0.0079 m/s，公式相同。
4. **固定上限**：
   - 经典的「固定」模式对整页所有字段生效；这里按字段记，而且只在勾了「所有结果」时跨结果。
   - 固定窗口按线性显示，与第 5 路 / §F 对经典 `range.mode: 'fixed'` 的映射一致。
5. **高亮点**：经典是 1.8 mm 的世界尺寸点，这里是 4 px 的屏幕尺寸点，缩小视图时也看得见。门槛和点集与经典相同。
6. **剖切时的标记**：经典把标记、端点和标签都放在剪裁平面下。这里：
   - 叠加层材质同样剪裁；
   - 文字标签在切面以上隐藏；
   - 发现标记的点击检测不受剖切限制（只是不显示）。
7. **可信度图例**：
   - 经典是视口里的列表；这里是右下角小卡片，规则放进 ⓘ。
   - 体场壁面压力的「位 2 / 4 灰化」仍用第二期的斜纹表示。
8. **默认口径**：
   - 经典在每次打开报告时自动套用服务端默认；这里改为在工具页点「套用」，写进本机设置。这样不会在用户不知情时改掉现在的显示。
   - 不带：壁面透明度（W18 已放弃）、经典 `log`（新工作区的 WSS、TAWSS 本来就按对数显示）、英文。
   - 「设为默认」仍只存本机，不写服务端。
9. **预设**：
   - 视角只在同一 run_identity 时带上。
   - 不带：测量、探针记录、体场截面状态、英文、「淡化分支」、点云特征着色。
   - 应用时会给出提示。
10. **自动标注条数**：选项与经典一致，只是从布尔开关（前 5 条）改为四选一，入口在图层菜单的二级菜单。
11. **P 键**：经典壁面 P = 锁定悬停点的探针，经典体场 P = 探针开关；这里 P = 悬停读数开关（单击本来就固定探针）。

## 4. 改动过的公共文件

- **`ws_shell.js`**：只在 `layersMenu` 里改了两处，各一行，都带 `P3 lane 4` 注释。
  1. 「发现标签（前 5 条）」一项改为 `ns.annot && ns.annot.findingsItem ? ns.annot.findingsItem(shellApi()) : {原来的项}`。没有 `ws_annot` 时，行为与原来相同。
  2. 「色表」组在 turbo 之后加一行 `{label: '蓝白红', checked: cm === 'bwr', run: function () { setCmap('bwr'); }}`。
- **`ws_store.js`**：`CMAPS` 加 `'bwr'`，一处，让偏好能存蓝白红。
- **`core_viewer.js`**：
  - 只在 `// ==== P3 lane 4 ====` 块内加代码：等值线、高亮、Z 剖切、叠加层剪裁、标签隐藏，并按第二期 E 路的方式包装 `applyField`、`setBranchVisibility`、`setLayers`、`setResult`、`refreshDisplay`、`applyClip` 的绑定。
  - `// P3 lane 4 exports` 之后加了三项：`p3Display`、`refreshP3`、`refreshClip`。
- **`bundle.json`、`index.html`**：未改，`core_contour.js` 的占位已在基线中。
- **已有测试 `tests/test_v2_display.py`**：改了一个断言。`sanitizePrefs` 的 `units` 现在多一个 `wss: 'Pa'`，原因是 W8 新增 WSS 显示单位种类，并且要让第 5 路的 `d.units` 自动带上它。
- **自有文件**：`core_contour.js`、`adapter_wall.js`、`adapter_volume.js`、`core_colormap.js`、`ws_display.js`、`ws_slice.js`、`ws_probe.js`、`ws_region.js`、`ws_annot.js`、`v2_display.css`。`core_colorbar.js`、`ws_measure.js` 不需要改。
- **没动**：`report_common.js`、`volume_viewer.js`、`three.min.js`、`OrbitControls.js`、`glossary.json`。
- **服务端**：Python 一行未改，用的是已有接口 `GET /api/preferences`。

## 5. 测试结果

- **`node --check`**：所有改过的 js 通过。
- **新测试 `tests/test_v2_p3_display.py`**，13 项，全部通过：
  - 移植的经典算法：合成数据和真实病例，与 `REPORT_CORE_JS` 逐位比对；
  - 速度对数下限；
  - viewer P3 块：等值线、高亮、剖切、输入 STL 时不画；
  - 壁面取点按剖切过滤，着色器带剪裁片段；
  - 体场灰化，公式与经典 `desaturate` 相同；
  - dyn/cm²：换算系数、`toDisplay`、探针卡；
  - 悬停读数：壁面、体场、无值三角形、可信提示；
  - 截面：X / Y / Z 平面公式、倾角与偏移限幅、拖动方式、对数；
  - 标注条数、经典默认口径与预设、链接钩子（含旧状态和坏状态）；
  - 外壳接线（stub harness）：图层菜单顺序、蓝白红、二级菜单、P 键、工具页经典设置；
  - 离线启动：不出现经典设置，也不请求接口。
- **全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`**：945 passed，33 skipped，4 分 13 秒。跳过项都是环境原因（开发任务副本不在等）。在最后一个提交（只改了视口 B 的接线和一行 CSS）之后，又跑了相关的 5 个测试文件，61 passed。
- **浏览器**（沙箱 :8824，Marionette 3024 / 3034）：每项功能都走过一遍，所有运行 0 个 JS 错误。
- **黄金回归没跑**：这一路没有改分析、流水线、叙述或报告生成的 Python。

## 6. 截图（绝对路径）

目录：`/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane4/shots/`

| 内容 | 文件 |
|---|---|
| 悬停读数：壁面 TAWSS / 体场体内点 | `b1_hover_tawss.png`、`e3_volume_hover.png` |
| 图层菜单（新项、蓝白红、发现标签…） | `b2_layers_menu.png` |
| 等值线 | `b3_contours_tawss.png` |
| 高亮最高值面板 | `b4_top_pop.png` |
| 剖切 55% | `b5_clip55.png` |
| 可信度图例：壁面 / 体场灰化 | `b6_trust_legend.png`、`d3_trust_grey.png` |
| 色标面板：Pa / dyn/cm² | `c1_scale_panel_pa.png`、`c2_dyn_units.png` |
| 固定上限 60 dyn/cm² | `c3_fixed60.png` |
| 蓝白红 | `c4_bwr.png` |
| 发现标签二级菜单、前 10 条 | `c5_label_count_menu.png`、`c6_labels10.png` |
| 速度对数色标 | `d2_speed_log.png` |
| 截面：垂直 Z 轴；倾角 + 拖动方式 | `d4_slice_axis_z.png`、`d5_slice_tilt_drag.png` |
| 截面 Z 40%：新工作区 / 经典 | `e4_v2_slice_z40.png`、`e5_classic_slice_z40.png` |
| dyn/cm² + 剖切：经典 / 新工作区 | `e1_classic_wall_dyn_clip55.png`、`e2_v2_wall_dyn_clip55.png` |
| 工具页经典设置、套用默认、应用预设 | `f1_tools_classic.png`、`f2_defaults_applied.png`、`f3_preset_applied.png` |
| 比较模式：两侧图例与右侧悬停 | `g1_compare_trust_hover.png` |
| 逐字节对照快照（基线 / 现在） | `pix/` |

## 7. 没做完的项与需要配合的事

1. **dyn/cm²**：「沿程」WSS 曲线和概览分区图、分区表还是 Pa（第 3 路的 `ws_profile.js`、`ws_overview.js`），做法见 §3.2。
2. **快捷键说明**：「?」对话框里没有 P（悬停读数）。对话框归第 6 路，请顺带写进去。
3. **需要裁定**：
   - 服务端经典默认口径是「手动套用」还是「首次自动套用」；
   - 「设为默认」要不要同时写服务端 `report_defaults`。
4. **触屏**：拖动方式只在无头浏览器里用指针事件走过，真实平板上需要点一次。
5. **沙箱**：:8824（PID 20593）和基线对照沙箱 :8834 都已按 PID 停掉，我启动的 Firefox 都已关闭。全量测试在最终提交上又跑了一次：945 passed，33 skipped。
