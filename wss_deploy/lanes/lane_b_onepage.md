# B 路（S5b）一页纸：交付报告

分支 `wss-ui-s5b-onepage`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_b`，2026-09-30。

## 1. 做了什么，入口在哪

### 1.1 一页纸换成新工作区的样子（`/api/jobs/<id>/onepage`）

只改了表现层，文字和数字一个都没动（证据见 §2）。

- **颜色与字体**：用 v2 的颜色变量（墨色 `#101828`，强调色青 `#0e6e8a`，警示橙 `#c4570d`，通过绿 `#0f8a5f`）和 v2 的字体栈。只有一个强调色，不用渐变。纸面上唯一的阴影是屏幕上纸张的 1 px 轻阴影。打印时 `--ink-3`、`--line` 比屏幕深一档，8 px 的注释和细线印出来还能看清。
- **页眉**：左边是 v2 的「WSS」图形标（装饰用，`aria-hidden`，不带文字），中间是标题，右边是审阅状态。审阅状态改成「圆点 + 文字」，色调和 `ws_ui.reviewInfo` 一样：已审阅签字是绿色实心点，已重新打开是橙色实心点，未审阅是空心点。
- **身份行**改成浅底的「病例条」：标签浅、值深，字段之间用细竖线隔开。原来的「 · 」分隔符还在文字里，只在视觉上画成竖线。
- **关键数字**：原来是浅底圆角卡片，现在改成上下有细线、格子之间有竖线的「数字条」，大数字 16 px。
  - 列数随数量定：5 个及以下排一行（单头壁面 5 格，体场 3–5 格）；6 个排 3 列；再多排 4 列（三头 8 格为 2 × 4）。
  - 「多模型一致性」卡显示成状态词：一致是绿点，分歧偏大、建议复核是橙点。
- **结论**：左侧一条细青线，不再用浅蓝底框。「审阅人已修改」「自动生成」改成细边框标签。
- **重点发现 / 发现详情**：
  - 级别由彩色胶囊改成「圆点 + 字」：关注是橙色实心点，提示是空心点，参考是灰点，几何类是灰色小方块。
  - 判定标记沿用原来的 ☑ / ✕ / ☐ 文字，并按状态着色：已确认绿色加粗，已驳回灰色，未判定浅灰。
  - 【人工】发现整行名称用强调色。
  - 附录里「已驳回的自动发现」整行变灰。
  - 「规则更新前的判定」表的说明列可以换行（原来不换行，会被截断）。
- **表格**：用 v2 的表格样式——表头 8 px 浅灰、无底色、1 px 横线、数字列右对齐、等宽数字。
- **签字栏、页脚**：保留原来的结构，间距放宽。电子审阅记录前加状态圆点。
- **配图**：细边、2 px 圆角，图注浅灰。附录里的「更多配图」标题和图片包在一起，打印时不会出现标题留在上一页、图片跑到下一页。
- **打印（A4）**：
  - 打印样式为浅色，保留颜色（`print-color-adjust:exact`）。
  - 第 1 页在纸上保持完整一页，页脚固定在第 1 页底部（`.page.first{display:flex;min-height:272mm}`）。
  - 附录另起一页。没有配图时附录接在第 1 页后面，这是原来的 F6 规则，照旧。
  - 标题尽量不与后文分开（`break-after:avoid`），表格行、卡片、签字栏不跨页。
- **屏幕**：
  - 顶部是吸顶工具条，右侧放 v2 主按钮「打印 / 保存 PDF」，带打印图标；按钮的 `onclick="window.print()"` 一字未改（一页纸的 CSP 只允许这个处理函数的哈希）。
  - 窗口窄于 840 px 时（嵌入或手机），纸张改为自适应宽度：数字条排 2 列，配图排 2 列，宽表格可以横向滚动，不会出现整页横向滚动（420 px 宽下实测 scrollWidth 438 < 窗口 450）。
- **顺带修掉一个老问题**：「可信区域」表原本应该只占约半宽（`table.narrow`），但纸张是纵向 flex 容器，表格被拉满整宽。现在加了 `align-self:flex-start`，按原意显示。

### 1.2 在新工作区里编辑签字栏模板

- **入口**：头像菜单 →「报告模板…」（在「经典工作台」上面）。
- **注册方式**：通过外壳扩展点 `ns.ext.push({id:'onepage', userMenu})` 注册，没有改 `ws_shell.js`。离线时不出现这一项：`menuItems` 在 `offline()` 时返回空列表，离线外壳本来也不显示头像菜单。
- **接口**与经典工作台一样：`GET /api/report-template` 读取，`PUT /api/report-template` 保存。载荷的键和解析方式与经典 `templateDialog` 相同：
  - 键：`institution / department / report_title / footer_note / signature_lines / show_glossary / appendix`；
  - 文字首尾去空白；
  - 签字栏按「, ， 、 换行」切分，去掉空项和重复项，规则同 `WB.parseTags`。
- **对话框**：
  - 左边是七个字段：机构名称、科室、报告标题、签字栏、页脚声明、术语表、打印附录页；
  - 右边是一张 A4 缩略示意，边改边显示机构名、标题、签字栏、页脚落在纸上的位置；勾选附录时，示意纸后面叠一张附录页；
  - 说明放进 ⓘ 提示，界面上只有字段名。
- **规则**：
  - 签字栏超过 4 项或某项超过 20 字时，在本地直接提示，文案与服务端相同，不发请求。
  - 服务端报错显示在对话框内：403 提示「没有修改权限：只有管理员可以修改。」
  - 没有修改权限（`editable:false`）时所有字段只读，只有「关闭」按钮。
  - 「恢复默认」把表单填回出厂值，点「保存」后才生效。
  - 保存成功后关闭对话框并弹出提示；如果当前打开着一个已完成的结果，提示上带「打开一页纸」。
- **代码**：`static/v2/ws_onepage.js`（逻辑）、`static/v2/v2_onepage.css`（样式，前缀 `.otp-`，对话框类 `.dlg-onepage`）。

## 2. 与经典页面的一致性证据（文字、数字零改动）

### 方法

1. 在改动前（HEAD `2df6f7e`）渲染一次，改动后再渲染一次。
2. 去掉 `<style>`、`<title>` 和全部标签，只留可见文字，然后比较：
   - 可见文字拼接后逐字符比较，一次去掉全部空白、一次把空白压成单个空格，两种都做；
   - `alt`、`title`、`aria-label` 属性逐条比较；
   - `<title>` 逐字符比较。

### 用例（共 20 个直接渲染 + 6 个服务端返回）

- **沙箱副本里全部 11 个已完成任务的 summary**：
  - 峰值 WSS X5D 旧 / 新任务；
  - M1 三头（周期量）；
  - PF6/VF6 体场；
  - 随访演示 A / B；
  - 待确认出口例。
- **构造的审阅场景**：
  - 已确认、已驳回、带备注未判定的发现；
  - 两条【人工】发现；
  - 两条「规则更新前的判定」；
  - 两条标注（其中一条含 `<b>`，用来查转义）；
  - 审阅人改过的结论；
  - 已审阅签字；
  - 机构模板：三个签字栏、页脚两行、术语表的 used / all / none、不打印附录；
  - 配图 1 / 3 / 4 / 6 张；
  - 3 次扫描的随访表。
- **测试夹具**：`wall_summary` + 形态、`m1_summary`、空 summary。
- **沙箱服务**真实返回的 `/api/jobs/<id>/onepage`，共 6 例。其中 M1 例通过接口写入了发现判定、人工发现和标注，另外直接往文件里写了一条规则更新前的判定，还改了结论并配了 6 张图；体场例有判定和标注；X5D 例已签字并配图；随访 A / B 带随访表。

### 结果

- 26 个用例全部 `text=same spaced=same attrs=same`。
- 唯一的结构差别是文本节点数多 1–2 个：身份行的「 · 」被包进了 `<span class="sep">`，文字本身没变。
- 汇总文件：
  - `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_b/text_diff_summary.txt`（20 个直接渲染）
  - `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_b/text_diff_http.txt`（6 个服务端页面）
- 逐例去标签文字在同目录的 `before/`、`after/`、`before_http/`、`after_http/` 里，比较脚本是同目录的 `compare.py` 和 `render_variants.py`。

### 其他证据

- 已有的一页纸测试一个都没改，全部通过：`test_onepager`、`test_cycle_content`、`test_findings_review`、`test_snapshots`、`test_cycle_derived`、`test_timeline`、`test_server_security`、`test_bundle`、`test_glossary`。
- 卡片构造函数（`_wall_cards` / `_cycle_cards` / `_volume_cards` / `_quality_card`）的输出格式没变，C 线测试直接断言的 `OP._cycle_cards` 也照旧。
- 打印页数与改动前相同（Firefox WebDriver:Print，A4）：M1 4 页、体场 4 页、X5D 3 页、随访 B 4 页。第 1 页都在一张纸内。

## 3. 与经典页面的差异和原因

| 差异 | 原因 |
|---|---|
| 关键数字的单位与数字同字号（v2 屏幕上单位更小） | 已有测试按原样断言 `"16.6 Pa"`、`"0.720 Pa"` 这类字面值。要拆成数字 + 小单位就得改这些测试，所以没拆。 |
| 页眉多了一个图形标，打印按钮多了一个打印图标 | 纯 SVG，`aria-hidden`，不含文字，与 v2 顶栏一致。 |
| 模板对话框多了缩略示意、本地校验、「恢复默认」、保存后「打开一页纸」 | 少用文字说明，改成看得见的效果。本地校验的文案与服务端完全相同；「恢复默认」不自动保存。 |
| 模板对话框「页脚声明」最多 400 字（经典为 500） | 服务端上限是 400（`report_template._TEXT_LIMITS`）；经典超过 400 会被服务端拒绝。 |
| 「可信区域」表不再拉满整宽 | 修掉 flex 拉伸的老问题，恢复 `table.narrow` 的原意。 |

## 4. 改动过的公共文件

**无。**

- `ws_shell.js`、`core_viewer.js`、`bundle.json`、`index.html`、`ws_api.js`、`ws_icons.js`、`v2.css`、`server.py`、合同与 README 都没改。
- 除 `onepager.py` 外没有改任何 Python 文件。

改动的文件：
- `wss_deploy/onepager.py`（本路独占）：
  - `_table` 加可选参数 `row_classes`；
  - 新增 `_columns`、`_card_tone`、`_decision`、`_row_class`；
  - 页眉、身份行、判定列、电子审阅记录加了 class；
  - 重写 CSS；
  - 打印按钮加图标；
  - 「更多配图」外面包了一层 `<section class="more-shots">`。
- `wss_deploy/static/v2/ws_onepage.js`、`wss_deploy/static/v2/v2_onepage.css`（本路独占，替换了占位内容）。
- 新测试 `tests/test_v2_onepage.py`：
  - 复用 `tests/test_v2_workspace_js.py` 的 `_run` / `WS_FILES` 桩环境，只读引用，没有改那个文件，在真实外壳里走头像菜单 → 对话框 → 保存；
  - 如果合并时那个文件的桩环境有变化，这里要跟着核对。

## 5. 测试结果

- `node --check wss_deploy/static/v2/ws_onepage.js`：通过。
- `tests/test_v2_onepage.py`：10 个全部通过，覆盖以下几点：
  - 页面上除打印处理函数外没有脚本，且处理函数的哈希等于 `server.ONEPAGE_HANDLER_HASH`；
  - 审阅状态色调、数字条列数、一致性卡色调；
  - 判定标记、行状态、驳回项只在附录、规则更新前判定表；
  - 病例条文字与分隔符；
  - 打印规则与 v2 颜色变量；
  - 数字条不添字；
  - 纯函数：签字栏切分、载荷、本地校验、离线不出菜单项；
  - 通过真实外壳的菜单打开对话框、字段回填、示意图联动、PUT 载荷与 CSRF；
  - 本地校验不发请求、403 报错、只读模式。
- 全量 `CUDA_VISIBLE_DEVICES= PYTHONPATH=. pytest -q -p no:cacheprovider tests`：**879 passed, 33 skipped**，用时 3 分 38 秒。
  - 跳过项都和环境有关：本工作副本没有 `outputs/wss_deploy_jobs` 下的开发任务、没有生成 `example_aaa.stl`，还有几项要设了环境开关才跑（`WSS_DEPLOY_SLOW_TESTS`、`WSS_DEPLOY_GOLDEN_REFERENCE`、`WSS_DEPLOY_DEVSHOT`）。
  - 用 `-rs` 逐条核对过，没有一项跳过来自本路的测试。
- 黄金回归：没有跑。`regress.py` 不涉及一页纸，本路也没有改分析、流水线或叙述代码。

## 6. 截图与打印件（沙箱 :8822，Firefox 无头，全部 0 个 JS 错误）

目录前缀 `/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/c691191d-1f19-45dc-96f9-b187bc2eee9d/scratchpad/lane_b/`：

- **改后，整页，默认模板**：
  - `final/onepage_m1.png`：周期量 M1，含已确认 / 已驳回、【人工】、规则更新前的判定、标注、改过的结论、6 张配图；
  - `final/onepage_vol.png`：体场，含判定和标注；
  - `final/onepage_x5d.png`：壁面峰值 WSS，已审阅签字；
  - `final/onepage_fuB.png`：随访表；没有配图，附录接在第 1 页后面。
- **改后，A4 打印**：
  - `final/onepage_{m1,vol,x5d}.pdf`；
  - 每页转成的图：`final/onepage_*_p-N.png`。
- **改前对照**：
  - `shots_before/onepage_{m1,vol,x5d}.png`；
  - `pdf_before/onepage_{m1,vol,x5d,fuB}.pdf`。
- **带机构模板**（三个签字栏、页脚附注）：`shots_after/onepage_m1_with_template.png`。
- **窄屏 420 px**：`shots_after/onepage_m1_narrow.png`。
- **模板对话框**：
  - 头像菜单：`shots_after/v2_usermenu.png`；
  - 打开：`shots_after/v2_template_dialog.png`；
  - 编辑中，示意图联动：`shots_after/v2_template_dialog_edited.png`；
  - 本地校验：`shots_after/v2_template_dialog_error.png`；
  - 保存后的提示：`shots_after/v2_template_saved.png`。

## 7. 没做完 / 需要主会话处理的事

1. **外壳头像菜单出现两条相邻分隔线**（「浅色视口」和「报告模板…」之间）。
   - 原因：`ws_shell.js` 在「浅色视口」后面已经放了一个 `{separator:true}`，`extMenuItems('userMenu')` 又在扩展项前面加了一个。
   - C 路往头像菜单加项时也会遇到。
   - 合并时改一行即可：去掉 userMenu 那处的固定分隔线，或者让 `extMenuItems` 在前一项已是分隔线时不再加。本路按规则没有改外壳。
2. **Firefox 对 `break-after:avoid` 支持有限。** 附录里某个小标题偶尔可能落在一页的最后一行，下面的表格从下一页开始。「更多配图」已经用包裹的方式解决，其余是尽力而为。
3. **`@page` 的页码角标（`@bottom-right`）主流浏览器不支持。** 这是原来就有的写法，保留了，没有新做。
4. **A 路（一页纸配图）接入后**：本路已测过 1 / 3 / 4 / 6 张配图的版式（第 1 页最多 4 张，其余进附录「更多配图」），A 路生成的图走同一个 `snapshots.json`，不需要再改一页纸。
5. **本路运行时的一个操作失误。** 我曾用 `pkill -f "pytest -q -p no:cacheprovider tests"` 停掉自己误用 GPU 设置启动的全量测试。这个模式可能匹配到同一用户下其它路同时在跑的同名 pytest 命令。事后查看时已经没有任何 pytest 进程在跑，所以无法确认有没有误伤。如果其它路报告全量测试莫名中断，原因可能在这里，重跑即可。
