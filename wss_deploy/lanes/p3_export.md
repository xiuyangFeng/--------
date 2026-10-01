# 第三期第 5 路：导出、链接与比较 · 交付报告

分支 `wss-ui-p3-export`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_e`，基线 866cf02。2026-10-01。
合同：`PHASE3_LANES.md` §3「第 5 路」；审计行：`p3_audit_workbench.md` #56、#93、#98、#112，`p3_audit_reports.md` W55、W51（六视角分文件）、C5。

## 1. 做了什么，入口在哪

| 项 | 做法 | 入口 |
|---|---|---|
| 复现链接补显示状态（W55） | 链接状态加可选 `d`（色表、光照、分段、阈值、单位、标签、显示图层、体场外壁与流线）和 `x`（扩展点状态）；仍是 `v:1`，旧链接照样打开、不动显示 | 导出对话框底栏、「工具」页「复现链接」（原入口） |
| 单独下载修正（#93） | 「数据」页按结果类型列出经典结果卡的单个文件：壁面 VTP、点云 CSV；体场 VTP、壁面压力 VTP、体场 CSV、流线 VTP（只在 `summary.exports.streamlines` 为真时）。`summary.exports` 是开关，不再当文件名用 | 导出 →「数据」 |
| 批量出图（#56） | `ws_batch.js` 在新查看器上重写经典 `batch_export.js`：每个已完成结果在后台一个离屏视口（900 × 900）里加载、出图，打成一个 zip（`<病例>_<任务号>/…` + `manifest.json`） | `ns.batch.open(jobIds, shellApi)`，由第 2 路任务列表多选条「批量出图」调用 |
| 六视角分成单张（W51） | 「拼图」页加「输出：一张拼图 / 分成单张」；分成单张时每个视角一张 PNG（同「图片」页样式，副标题带视角名），截面打开时截面图单独一张，打成一个 zip。「工具」页「六视角」按这个选择出图 | 导出 →「拼图」；「工具」页「六视角」 |
| 标量对照表（C5） | 比较页签加「标量对照」：`POST /api/compare` 的左、右、右 − 左；口径不一致时差值留空并列出原因（译成中文）；周期量说明照服务端 | 比较模式 →「比较」页签 |
| 选择框放开 20 条 + 搜索（#98） | 「其他已完成结果」不再截到 20 条；顶部搜索框（病例名、患者、病例号、扫描标签、日期、标签、结果名、任务号，空格分词全部匹配），计数「n / 总数」 | 工具条「比较两次结果」 |
| 两侧身份（#112） | 每侧名称下一行：复核状态、患者、病例号、扫描标签、扫描日期（来自 manifest 的 job 块）；隐藏病例名时两侧都只写「病例」，不出患者号和病例号 | 比较页签顶部 |

批量出图对话框（少字，说明在 ⓘ）：

- **显示**：按结果类型各一行（壁面 / 体场），字段（各例默认或指定）+ 色标（本例自适应 / 自适应（线性）/ 固定范围 下限–上限）。分段、阈值按各字段「设为默认」。
- **链接**：可粘贴一个复现链接，或点「用当前显示」（有结果打开时）；用它的字段、色标窗、分段、阈值、色表、单位、标签、图层，不用它的视角。
- **视角**：前后左右头足任选 +「六视角」；输出「每个视角一张 / 每例一张拼图」；可选「色标 SVG」每例另存一份。
- **样式**：直接用「图片」页的同一组控件和存储（倍率、背景、文字、色条、标题、标签、裁边、隐藏病例名）。
- 运行时有进度条和逐例记录（渲染中 / 已导出 n 个文件（附说明）/ 失败 · 原因 / 跳过 · 原因），「停止」在当前这例完成后停下，已完成的照样打包；关掉对话框也会停下且不下载。完成后自动下载，「再次下载」备用。
- 未完成、读不到的任务列为「跳过」；一次最多 200 个。

## 2. 扩展后的复现链接状态格式（给第 6 路写经典 `#view=` 转换）

地址：`#/job/<id>?f=<字段>&view=<base64url(UTF-8 JSON)>`（不变）。JSON：

```
{ v: 1,                       // 仍是 1：旧读取端忽略新键；新读取端遇到没有 d 的旧链接不动显示
  run, f, w, L, cam, cu, sel, br, sl,          // 第二期 A 路原有（字段、色标窗、图层、相机、游标、选中、分支、截面）
  d: {                        // 可选：显示
    cmap:   'rainbow' | 'turbo' | 'viridis' | …（ws_store 认得的色表；不认得的给提示、不改偏好）
    light:  'soft' | 'flat',
    labels: {branches: bool, findings: 0|3|5|10, maxd: bool, annotations: bool},
    bands:  {<字段 id>: n},                    // 每个可见字段的实际分段（0 = 连续）
    thr:    {<字段 id>: [t0, t1, t2]},         // 每个带阈值字段的实际观察阈值
    units:  {pressure: 'Pa'|'mmHg', velocity: 'm/s'|'cm/s', …},   // ws_display.UNITS 的种类（第 4 路加的种类自动带上）
    layers: {stl, peaks, stagnation, vectors, …},                   // ws_display 的显示图层
    volume: {opacity, density, width, thin}    // 只在体场结果
  },
  x: {<扩展 id>: 状态}        // 可选：ns.ext 条目的 linkState(api) 返回值；打开时调用 applyLinkState(状态, api)
}
```

- 「实际」= 这份结果自己的选择，否则该字段的「设为默认」，否则发布包阈值 / 连续。所以接收方自己的默认不同也看到同一幅图。
- 套用顺序：显示（偏好 + 本结果的分段阈值，经 `viewer.applyState({display})`）→ 字段与色标窗（有 `d` 时强制重画）→ 图层 → `x` → 截面 → 游标 → 相机。
- 色表、光照、标签、单位、显示图层、外壁与流线是本机偏好：打开链接等于在菜单里选了一次，之后保持；提示条写「已按链接设置色表、单位…」。
- **经典 `wss-deploy.view/v1` → `d` 的对照**（补 `p3_audit_reports.md` §F）：`colormap`→`d.cmap`；`bands`（当前字段一个数）→`d.bands[f]`；`field_thresholds` / `thresholds_pa`→`d.thr`；`labels {findings, branches, max_diameter}`→`d.labels {findings, branches, maxd}`；`overlay.stagnation`→`d.layers.stagnation`；`highlight.peak`→`d.layers.peaks`；体场 `opacity`→`d.volume.opacity`；`units:'dyn'`→`d.units` 里第 4 路 WSS 单位种类的键（第 4 路合并后看 `ns.display.UNITS`；没有就提示）；`overlay.contours`、`highlight.top*`、`slice.clip` 等第 4 路功能→该路扩展的 `x.<id>`（第 4 路实现 `linkState` / `applyLinkState` 后自动生效）。

扩展点（新增、可选）：`ns.ext` 条目可加 `linkState(api) → JSON | undefined` 与 `applyLinkState(state, api)`。链接里有、但本页没有对应扩展的 `x` 项会提示「已跳过」。

## 3. 一致性证据

1. **标量对照表**（LV 随访演示 A `20260930_045134_6b8640d971b0` vs B `20260930_045135_c66b989ddd40`）：新工作区与经典 `/compare?left=A&right=B` 调的是同一个 `POST /api/compare`，16 行同序同值：峰值 p99 11.09 / 10.57 / −0.53，峰值最大 27.52 / 25.72 / −1.80，壁面均值 2.41 / 2.42 / +0.006，六条分支 p99（主动脉 4.54 / 5.21 / +0.67 … 左髂总 8.61 / 8.10 / −0.51），TAWSS 均值 0.759 / 0.821，TAWSS p99 2.86 / 2.88，OSI 均值 0.160 / 0.160，RRT 3.54 / 2.91，ECAP 0.376 / 0.312，滞留区 0.248 / 0.111。差别只在显示：经典固定 2 位小数，新工作区 3 位有效数字（滞留区按百分比，差值写「个百分点」）。原始响应在 `/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane5/cmp_raw.json`，经典表文字在同目录 `cmp_classic.txt`。
2. **比较条件表「统计定义」**：原来两例判「不一致」，原因是 `statistics_protocol` 里随几何变化的 `top5_connectivity_radius_mm`（0.987 / 0.997）；服务端 `comparison.py` 早就把它和 `connectivity_radius_mm` 排除在外（`_VOLATILE_PROTOCOL_KEYS`）。现在两边同口径，判「一致」，与 `/api/compare` 的 `compatible: true` 一致。
3. **单独下载**：体场 `20260929_230450_d1428792b846` 四个文件请求都是 200，字节数与任务目录一致（volume_fields.vtp 882593、wall_pressure.vtp 411812、points_volume.csv 1646498、streamlines.vtp 1484811）；壁面 `20260929_230442_a7273f1670e4`：wall_wss.vtp 643195、points_wss.csv 6608487。文件名与经典 `app.js resultCard` 相同。
4. **复现链接**：在一台浏览器里把周期指标 TAWSS 设成 viridis、6 段、阈值 0.3 / 3 / 6、滞留区斜纹、分支名与前 5 条发现标签，复制链接（1015 字符）；在没有任何本地存储的新浏览器里打开：色表 viridis、色条 6 段、阈值 [0.3, 3, 6]、斜纹、标签、字段、相机全部复原（截图 `l2_link_restored.png`）。去掉 `d` 的旧格式链接打开后字段照改、色表偏好不动。
5. **批量出图**：沙箱 7 个任务（周期指标、体场、峰值 WSS、随访 A/B、待确认演示、失败）→ 6 例导出、1 例跳过（失败），3 秒；每张图的色条、标题、视角名、最大 / 最小值标记与单张「图片」页同一套代码（`ws_figure.figure` / `montage` / `fileName` / `colorbarSVG`）。第二批用复现链接（TAWSS 固定 0–2 Pa、8 段、turbo）+ 六视角拼图 + 色标 SVG + 隐藏病例名 + English：4 例 8 个文件，文件夹 `case_<任务号>`，`manifest.json` 记下设置、链接与逐例字段 / 色标窗 / 说明。「停止」测试：1 例完成、5 例「未开始（已停止）」，zip 只含完成的一例。
6. **六视角分成单张**：`LV_GUO_YOU_views_tawss_2x.zip` 含 `LV_GUO_YOU_{front,back,left,right,top,bottom}_tawss_2x.png`（经典 `exportFilename` 命名）；离线单文件页（file://）里同样能导出（zip 内 1 张，只选了前视）。

## 4. 与经典的差异和原因

- **批量出图的视图来源**：经典有内置预设（瘤囊低 WSS 区、髂分叉热点、主动脉沿程、沿程压降、流线全貌、瘤囊截面系列）和「报告导出的视图状态 JSON」。新版用「按类型选字段 + 色标范围 + 标准视角」，再加「粘贴复现链接 / 用当前显示」代替视图状态 JSON。预设里依赖等值线、高亮最高 1%、沿程菜单、截面系列的部分新工作区没有对应（等值线 / 高亮是第 4 路的活，沿程曲线不是图片），所以不做预设；「瘤囊低 WSS 区」可以用 TAWSS/WSS 固定 0–2 Pa + 分段默认来出。
- **视角**：经典按链接里的相机（各例世界坐标不同，常常对不准）；新版只用每例自己的解剖坐标架标准视角，撑满同样画幅。没有坐标架的结果用默认视角出一张并在记录里写明。
- **离屏显示设置**：`ws_display` 只给工作台上的两个视口答话。批量运行期间把它的 `ns.viewer.displayProvider` 包一层（`Object.create(原 provider)`，只替离屏视口回答分段、阈值、单位、显示图层、最大 / 最小值标记；其他视口和其他方法照旧由原 provider 回答），结束时还原。输入 STL 叠加和速度箭头在批量图里固定不画。没改 `ws_display.js`（第 4 路文件）。
- **zip**：经典 `batch_export.js` 的存储式 zip 写法搬进 `ws_figure.zipStore`（同一算法：CRC-32、UTF-8 文件名、重名加 _2），因为第 6 路会删 `batch_export.js`，而六视角分文件在离线页也要用。zip 的 `manifest.json` 版本改为 `wss-deploy.batch_export/v2`，多了每例字段、色标窗、说明、耗时；隐藏病例名时不写病例名。
- **六视角**：经典是 6 次单独下载；这里打成一个 zip（浏览器常拦连续多次下载）。
- **标量表数字格式**：3 位有效数字代替固定 2 位小数（OSI 0.160 不会显示成 0.16 / 0.00）。行名译成中文，分支名照服务端。
- **两侧身份**：不显示发布包代号（新工作区「不在屏幕上出现发布包代号」的规矩），结果名已在名称行、版本日期在「比较条件 → 模型与协议」。

## 5. 改动过的公共文件

无。`ws_shell.js`、`core_viewer.js`、`bundle.json`、`index.html`、`ws_api.js`、`ws_display.js`、`v2.css`、Python 服务端都没动（`ws_batch.js` 的占位在基线里已经放好）。只改了本路独占的 `ws_export.js`、`ws_figure.js`、`ws_compare.js`、`ws_batch.js`、`v2_export.css`，新加 `tests/test_v2_p3_export.py`。

跨路说明：

- **第 2 路**：多选条「批量出图」调 `ns.batch && ns.batch.open(jobIds, shellApi)`；`jobIds` 可含未完成任务（会列为跳过）；`shellApi.jobs()` 里找不到的任务号会单独 `GET /api/jobs/<id>`。离线页没有 `ns.batch`。
- **第 4 路**：想让自己的显示状态（等值线、高亮最高 x%、剖切、可信灰化等）进复现链接，在扩展条目上加 `linkState` / `applyLinkState` 即可；新单位种类、新色表（如蓝白红）只要进了 `ns.display.UNITS` / `ws_store` 的色表清单就自动进链接。批量离屏视口不会出现第 4 路新增的 provider 方法效果（原 provider 对离屏视口返回空），需要时可在 `ws_batch.installProvider` 里补。
- **第 6 路**：经典 `#view=` 转换见 §2；删 `batch_export.js` 不影响本路（zip 写法已搬进 `ws_figure`）。

## 6. 测试结果

- `node --check`：`ws_batch.js`、`ws_figure.js`、`ws_export.js`、`ws_compare.js` 通过。
- 本路测试 `tests/test_v2_p3_export.py`：15 通过（链接显示状态与扩展状态、旧链接不动显示、不认识的色表 / 单位只提示、zip 用 Python `zipfile` 校验、拼图页输出选项、按类型的单独下载、标量表模型与身份、比较页签请求与缓存与失败、选择框不截断与搜索、批量的计划 / 设置 / 链接 / 字段与色标窗 / 显示 provider / 运行（文件夹、manifest、失败、停止、清理）/ 对话框）。
- 改了已有测试：没有。
- 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`：**947 通过，33 跳过**（4 分 49 秒，退出码 0）。跳过都是环境原因（工作副本里没有开发任务副本、需要显式开启的慢测试 / 黄金回归 / 真实浏览器测试、`example_aaa.stl` 未生成），与本路无关。
- 没改分析、流水线、叙述、报告生成的 Python，未跑黄金回归。

## 7. 截图与样张（绝对路径）

目录 `/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane5/shots/`：

- 批量出图：`b1_batch_dialog.png`（对话框）、`b2_batch_running.png`、`b3_batch_done.png`、`b4_batch_contact.png`（6 例前视样张拼在一起）、`b5_batch_link_montage.png`（粘贴链接后）、`b6_batch_link_done.png`、`b7_montage_link_contact.png`（链接 + 六视角拼图 + English 样张）、`b8_batch_stopped.png`（停止）、`b9_batch_narrow.png`（390 px）
- 复现链接：`l1_source_view.png`（来源）、`l2_link_restored.png`（新浏览器复原）
- 比较：`c1_compare_panel.png`、`c3_compare_identity.png`、`c4_compare_scalars.png`、`c2_classic_compare.png`（经典对照）、`p1_picker.png`、`p2_picker_search.png`
- 下载与六视角：`d1_volume_data_pane.png`、`v1_montage_files_option.png`、`v2_montage_files_done.png`、`v3_six_files_contact.png`（六张单图）、`v4_montage_page_files.png`、`o1_offline_views_files.png`（离线页）

导出文件（zip / PNG，含病例数据，不入 git）在 `/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane5/downloads/`。每一步浏览器 JS 错误为 0（唯一一条出现过的是 Firefox 自己的 `ASRouter` 内部消息，与页面无关）。

## 8. 没做完 / 需要决定的

- 经典批量出图的内置预设没有迁（理由见 §4）；如需要，可以在第 4 路的等值线、高亮合并后加「预设」下拉，填的就是本对话框的字段与链接状态。
- 复现链接不带测量、探针记录、标注（标注本来存在服务端）——与审计 W55 的建议一致，只补了显示。
- 链接里的色表 / 单位 / 标签会改接收方的本机偏好（与在菜单里选一次相同），没有做「只对这次生效」。
- 批量图的离屏画幅固定 900 × 900 CSS px（裁边默认开，所以输出大小随血管形状变化）；没有做可选画幅。
