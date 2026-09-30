# A 路（S5a）导出与出图 · 交付报告

分支 `wss-ui-s5a-export`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_a`。2026-09-30。

## 1. 做了什么，入口在哪

入口两个，界面上不多加东西：

- **工具条最后的「导出」图标**（已有）→ 导出对话框，现在分四页：**图片 · 拼图 · 一页纸 · 数据**；底栏左侧「复现链接」。
- **检查器「工具」页末尾新增「出图」一行**（扩展点 `tools`）：导图… · 六视角 · 色标 SVG · 复现链接 · 一页纸配图… · 打印。说明都在 ⓘ 里。

| 功能 | 做法 | 位置 |
|---|---|---|
| 出版级导图（在「汇报图」上扩展） | 倍率 1× / 2× / 4×；背景 视口 / 白 / 透明；文字 中文 / English；可选 色条、标题、标签、裁边、隐藏病例名；左侧实时预览并标出输出像素 | 对话框「图片」页 |
| 六视角拼图 | 前、后、左、右、头、足六幅，共用一条色条，a–f 标号 | 「工具」页「六视角」一键出图；「拼图」页「六视角」预设 |
| 多视角拼图 | 自选视角 + 当前视角；体场开着截面时可加「截面」图；列数 2 / 3 / 4；样式跟「图片」页一致 | 「拼图」页 |
| 色标 SVG | 就是视口右侧那条色条（`core_colorbar.toSVG`），可英文、可配深色背景 | 「图片」页、「工具」页 |
| 复现链接 | `#/job/<id>?f=<字段>&view=<base64url JSON>`，恢复字段、色标窗、相机、游标、选中点、分支显隐、图层、体场截面（位置、倾角、厚度、物理量、色标方式、切开） | 对话框底栏、「工具」页 |
| 书签带截面 | 体场结果的书签多存一个 `slice`（截面状态，关着时为 null）；打开书签时重开 / 移动 / 关闭截面，再回到书签的视角 | 「书签」页（原有），键 B |
| 一页纸配图 | 经典格式 `POST /api/jobs/<id>/snapshots`（front / left / top / current，截面开着时加 slice），白底 2×、带色条；上传后显示缩略图和「打开一页纸」 | 「一页纸」页 |
| 打印当前视图 | 当前视图 + 色条排成一张 A4（横竖按图自动选），标题、字段、研究用途一行；只有打印时显示，不开新窗口 | 「图片」页、「工具」页 |

离线报告（`file://`）里：对话框只有「图片」「拼图」两页（一页纸与数据需要服务），复现链接指向那个离线文件本身，可用；实测无报错。

## 2. 代码结构

- `static/v2/ws_figure.js`（本路主模块，约 1040 行）：选项（本机 `wssv2:export`，try/catch）、中译英、主题、色条 SVG、单图合成、裁边、标签重排、拼图、文件名、复现链接编解码与恢复、书签截面、一页纸配图、打印、对话框三页、扩展点注册（`tools`、`onResult`、`onClose`）。
- `static/v2/ws_export.js`：对话框改成四页 + 底栏复现链接；「数据」页保留原有的复核数据和离线阅读；没有 `ns.figure` 时回退到第一期的汇报图（测试、精简包）。
- `static/v2/ws_bookmarks.js`：`make()` 记截面，面板「打开」在壳的恢复之后调 `ns.figure.afterBookmark`，保存对话框的「记录：」里写「截面位置」。
- `static/v2/v2_export.css`：对话框各页、「出图」行、打印专用页（`@media print`，只在 `body.ws-printing` 时生效）。
- `core_viewer.js` lane A 块：`exportPose(name)`（标准视角按导出画幅撑满，四边留 14 px，不让 HUD）与 `renderPose({camera, scale, background, labels})`（离屏渲染任意位姿；在同一次调用里把相机摆过去、渲染、摆回来，不发 camera 事件、不改 fit、不动控件，并在浏览器显示前按实时相机重画一帧，所以屏幕不闪；轮廓线和斜纹宽度按导出倍率缩放；HTML 标签按该位姿投影返回，供 2-D 画布重画）。

数值全部来自已有代码：色条 = `ns.colorbar.toSVG`（与视口色条同一份 info）；拼图 = `WssReportCommon.composeMontage`；文件名 = `WssReportCommon.exportFilename`；离屏 = `WssReportCommon.renderOffscreen`；标签避让 = `WssReportCommon.declutterLabels`；截面图 = `ws_slice.paint` + `VolumeViewerCore.sliceGrid`；标准视角 = `core_viewer.viewDirection`。本路没有新造任何统计。

## 3. 与经典页面的一致性证据

1. **一页纸配图格式**：同一病例（LV_GUO_YOU 周期指标 `20260929_230442_a7273f1670e4`）先在经典报告点「生成一页纸配图」，再在新工作区生成，`snapshots.json` 的条目键完全相同（`bytes, caption, created_at, file, height, name, view, width`），名称顺序相同（front, left, top, current）；服务端都接受，`/api/jobs/<id>/onepage` 第 1 页「配图」都显示 4 张。体场（`20260929_230450_d1428792b846`）截面开着时为 5 张（多 slice），与经典 `makeOnepageShots` 相同。
2. **标准视角方向**：`tests/test_v2_lane_a_export.py::test_standard_views_match_the_classic_report` 直接从 `report.py` 取出经典 `standardViews`，在任意旋转的坐标架上比较六个视角的视线方向与上方向：front/back/left/right/top/bottom ↔ anterior/posterior/left/right/superior/inferior，最大差 < 1e-12。
3. **色条**：导出 SVG 的刻度与阈值文字和视口色条逐项相同（`0.4, 4, 0.05, 0.1, 0.3, 1, 3`，实测 TAWSS 本例自适应对数窗）；中文白底时导出字符串与 `toSVG(info)` 逐字节相同（测试 `test_colorbar_svg_is_the_stage_drawing`）。英文版只翻译文字，刻度数字不变（同一测试）。
4. **文件名**：单图 `LV_GUO_YOU_custom_tawss_2x.png`、拼图 `LV_GUO_YOU_montage_tawss_2x.png`、色条 `LV_GUO_YOU_colorbar_tawss_1x.svg`，与经典 `exportFilename` 同一函数同一格式。
5. **复现链接**：新开一个没有任何本地存储的浏览器，直接打开链接：字段 OSI、色标窗「高振荡窗」、相机位置与目标（4 位小数）、游标（左髂总 20.5 mm，游标面板打开）、分支显隐 [0–4] 全部复原；体场链接的截面（主动脉 0.62、厚 3 mm、俯仰 8°、偏航 −6°）逐项复原。同一结果已打开时粘贴链接（hashchange）同样复原。
6. **书签带截面**：保存（截面开着）→ 关截面、转到后视 → 打开书签：截面状态逐项相同，相机与书签记录一致。

## 4. 与经典页面的差异和原因

- **标准视角名**：新工作区用「头侧视 / 足侧视」（与视口方位标的头 / 足一致），经典叫「上视 / 下视」；一页纸配图说明因此是「头侧视 · TAWSS · Pa」。文件名和配图名称仍用经典的 top / bottom。
- **一页纸配图的字段**：按当前字段出图（经典壁面报告默认 WSS）。说明格式相同：「视角 · 字段 · 单位」。
- **裁边**（默认开）：单图去掉血管周围的空白；拼图和一页纸配图按所有面板的最大内容框统一裁剪，面板仍等大。经典不裁边。一页纸配图因此是 1326 × 1390 左右，经典是 2200 × 1318 的整视口。可在「图片」页关掉裁边（拼图与单图共用该选项；一页纸配图总是统一裁剪）。
- **拼图的色条与截面图**：色条是视口上正在显示的那条（截面打开时就是截面色标，三维里的截面板也用它），截面面板用 `ws_slice.paint` 画在同样大小的画布上，与视口截面图同一色标。经典体场在非截面页时会把截面图改画成三维色标，这里不需要。
- **拼图字号**：面板多时标题、说明和色条按拼图总高放大（最多 2 倍），经典固定按倍率。
- **英文**：v2 色条文字是中文写死在 `core_colorbar` 里的，本路在导出时逐句翻译（本路自己的词表 + 经典 `ZH2EN`），不改 `core_colorbar.js`。病例名本身不翻译。
- **标签**：导出时把视口里的发现、分支名、标注、测量标签按导出分辨率重新避让排布，移动过的画细引线（同经典 composeExport）；只画视口里当前显示的标签（与图层菜单一致）。
- **打印**：经典没有单独的「打印当前视图」（只有一页纸）；这里用页面内的打印专用容器，不开新窗口，CSP 不受影响。
- **比较 / 并排**：导出的是左侧视口（A）；截面只在单视口里有。
- 已知小差别：壁面结果「预测点」图层（屏幕固定像素大小的点）在 2× / 4× 导出里相对变细；默认不显示，未处理。

## 5. 改动过的公共文件

| 文件 | 改动 |
|---|---|
| `static/v2/core_viewer.js` | 只在 `// ==== lane A ====` 块内新增 85 行（第 900–984 行：`EXPORT_INSETS`、`exportPose`、`exportWidths`、`exportLabels`、`renderPose`）；`// lane A exports` 之后加 1 行（第 1155 行：`exportPose: exportPose, renderPose: renderPose,`）。块外一个字符都没动。 |
| `static/v2/ws_shell.js` | 只在 `SHELL_API` 的 `// lane A` 之后加 1 行（第 66 行）：`toggleCursor`、`cursorMoved`、`toggleSlice`、`exitSlice` 四个助手（复现链接和书签要按壳自己的方式开关游标与截面）。`parseHash` / `buildHash` 未改，链接参数由本路在 `onResult` 和自己的 hashchange 监听里读。 |
| `bundle.json`、`index.html`、`ws_api.js`、`ws_icons.js`、`v2.css`、Python 服务端 | 未改。图标用 `ns.icons.register`（`fig-link`、`fig-montage`、`fig-colorbar`）；接口用已有的 `POST /api/jobs/<id>/snapshots` 与 `files/snapshots.json`。 |

新测试文件：`tests/test_v2_lane_a_export.py`。没有改动已有测试。

**需要主会话合并时处理**：

1. `ws_shell.js` 第 16 行 `CLASSIC_TOOLS = '六视角、出版级导图、分支展开图'` 与「工具」页「经典报告里的工具」的提示：本路合并后「六视角、出版级导图」已迁入，应只剩 D 路负责的「分支展开图」（D 路完成后整段可去掉）。这是壳里的文案，本路没有改。「导出」小节的说明「汇报图、复核数据、离线报告、一页纸。」也可以相应精简。
2. E 路会改 `core_colorbar.js`：本路依赖 `toSVG(info, {width, graphicHeight, background})` 与 `noteLines(info)` 两个接口不变；E 路新增的分段、自定阈值会自动进入导出（导出用的就是视口色条的 info）。
3. B 路改一页纸：本路只写 `snapshots.json` 和 `snapshot_<name>.png`，与经典格式相同，一页纸读法不变即可。

## 6. 测试结果

- `node --check`：`ws_figure.js`、`ws_export.js`、`ws_bookmarks.js`、`core_viewer.js`、`ws_shell.js` 全部通过。
- 本路测试 `tests/test_v2_lane_a_export.py`：15 通过（选项、中译英、色条 SVG 与视口同源、经典文件名、图幅布局、链接编解码且 `parseHash` 不受影响、恢复顺序「字段 → 图层 → 截面 → 游标 → 相机」、书签带截面、一页纸计划与大小上限、经典标准视角、扩展点与「出图」行、对话框分页与离线 / 回退）。
- 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`：**884 通过，33 跳过**（3 分 50 秒，退出码 0）。跳过全是环境原因：工作副本里没有开发任务副本（`test_v2_data/offline/routes` 共 22 个）、需要显式开启的慢测试 / 黄金回归 / 真实浏览器测试、`example_aaa.stl` 未生成等，与本路无关。本路没有改分析、流水线或叙述相关的 Python，未跑黄金回归。
- 沙箱浏览器（端口 8821 / Marionette 3021）逐项走过：壁面周期指标、峰值 WSS（带发现标签）、体场（截面）、随访演示 A/B 比较模式、离线 HTML、390 px 窄屏；每次运行 JS 错误均为 0。导出文件都在页面里截获并落盘检查（见第 7 节）。服务端接受配图，`/api/jobs/<id>/onepage` 显示。

## 7. 截图与导出样张（绝对路径）

目录 `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_a/shots/`：

- 对话框「图片」页：`final_01_figure_tab.png`；窄屏：`i2_narrow_export.png`；离线报告里：`g2_offline_export.png`
- 「拼图」页与完成后缩略图：`final_03_montage_tab.png`；体场选视角：`d3_volume_montage_tab.png`
- 「一页纸」页：`final_04_onepage_tab.png`，体场 5 张：`d4_volume_onepage.png`；一页纸页面：`b2_onepager_page.png`
- 「数据」页：`g1_data_tab.png`；「工具」页「出图」行：`final_05_tools_tab.png`
- 复现链接恢复：`c2_link_restored.png`（壁面，游标）、`e1_vol_link_restored.png`（体场截面）
- 书签带截面：`d5_bookmark_dialog.png`、`d6_bookmark_restored.png`
- 打印页（屏幕上模拟打印样式）：`e2_print_page.png`
- 导出样张：`final_fig_white_zh_0_LV_GUO_YOU_custom_tawss_2x.png`（白底裁边）、`x5d_en_viewport_0_LV_GUO_YOU_custom_wss_2x.png`（英文、深色视口背景、发现标签）、`fig_transparent_4x_0_LV_GUO_YOU_custom_tawss_4x.png`（透明 4×）、`final_montage_0_LV_GUO_YOU_montage_tawss_2x.png`（六视角）、`x5d_montage_en_dark_0_LV_GUO_YOU_montage_wss_2x.png`（英文深色六视角）、`vol_fig_0_LV_GUO_YOU_custom_slice_2x.png`（体场截面单图）、`vol_montage_0_LV_GUO_YOU_montage_slice_2x.png`（含截面面板的拼图）、`fig_viewport_en_1_LV_GUO_YOU_colorbar_tawss_1x.svg`（英文深色色标 SVG）

## 8. 没做完 / 留给后面

- 拼图不提供逐面板自由视角（「当前」只有一个）；需要时可以先存几个书签再各自出图。
- 打印只在沙箱里用屏幕模拟打印样式核对过版面（无头浏览器不能弹打印对话框），真实打印机 / 另存 PDF 请在桌面浏览器确认一次。
- 「预测点」图层在高倍率导出时点变细（见第 4 节）。
