# 第三期第 3 路 结果信息与安全 · 交付报告

分支 `wss-ui-p3-info`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_c`，基线 866cf02。2026-10-01。
依据：`PHASE3_LANES.md` §3 第 3 路；审计行 W29、W36、W37、W62、W63、W65、V15、V38、V39、V41、V45、#82、#103 和 #11 的时间线备注。

## 1. 做了什么，入口在哪

### 1.1 结果页顶部警示条（优先级 1，`ws_notice.js`，W65 / V45）

- 位置：视口上方居中（工具条下面），用外壳的 `shellApi.noticeHost()`。「按问题看」条出现时自动下移。
- 触发条件和经典页一样：
  - 几何超出发布包参考范围（`analysis.reference_assessment` 有一项 `review`，或整体是 `review`）；
  - 人群参照需要复核（`population.status === 'review'`）；
  - 集成质量不是 `good`（`analysis.quality.level`）。
- 文字：照搬经典壁面报告 `bannerInfo` 的句子，例如「注意：输入几何有 1 项超出发布包参考范围（右髂外中位半径 2.14 mm，参考 2.17–6.28 mm）；模型集成质量：存在不确定性，建议复核。预测可信度可能下降，请结合详情复核。」数字用两页共用的 `WssReportCommon.formatValue`。
- 「详情」：对话框「参考范围与质量」。
  - 内容：几何参考状态句和超范围清单（体场页的写法）；人群参照（经典 ref-card 的句子）；N 模型一致性：等级、理由和「一致不代表准确」。
  - 只用 `quality.label / reasons`，不读 `quality_audit.json`，不出现任何离散度数值。对话框里写明「逐点离散度只保存在质量审计文件里，这里不显示」。
- 「×」：本次打开内关闭；换到别的结果再回来会重新显示（同经典「仅本次打开」）。
- 离线报告：同样显示（manifest 已内嵌）。

### 1.2 概览补全（`ws_overview.js`）

- 「质量与参照」小节（W29、W31）：三行，五模型 / N 个模型一致性、几何参考、人群参照，文字用经典卡片的原句。点任一行或「详情」打开同一个对话框。没有这些数据的结果（如体场的质量）不显示对应行。
- 「几何参数」小节（W37、W36），默认收起，点「展开」：
  - 分支表：长 mm、最小 r、中位 r、最大内切直径、迂曲，取整位数同经典 `fmt`；
  - 「出口与 Murray 分流」：开口、盖面半径、分流 %（体场结果没有分流，只列开口半径）；
  - 下面一行注明出口映射是人工确认还是自动（见 §3 差异 2）。
- 「术语与口径」小节（只在体场结果，V38、V39）：
  - 术语：相对压力、ΔP 压差、速度大小、采样支撑、近切口、几何越界、插值无支撑；
  - 口径：压力参考、统计口径、流线 N 条；
  - 每个词悬停看说明，文字来自术语表和经典体场页（压力参考、统计口径、流线说明、「单个预测时相的流线不是粒子随时间运动的轨迹」）。
- 页脚「技术信息」按钮（W62）：在「模型说明」旁边，基本档、完整档、离线报告都有。

### 1.3 技术信息（`ws_detail.js` 的 `techRows` / `techDialog`，W62）

- 行：发布包、发布包哈希、模型族（含模型数）、特征合同、运行身份、输入 SHA256、模型帧、周期定义、生成时间、报告重建、摘要版本（以上同经典 `renderTech`），另加数据版本、分析版本、部署版本、代码、显示插值、计算设备、坐标架方向来源。
- 时间：naive 的生成时间借同一次运行里带时区的时间戳（经典规则，`VolumeViewerCore.withOffset`），用 `WssReportCommon.localTime` 显示，括号里是原文。
- 对话框有「复制」（制表符分隔，同经典体场页）；在线时另有「完整统计 JSON」「运行清单」链接。
- 发布包、特征合同、运行身份三行带术语说明 ⓘ。
- 工具页（完整档）原来的「技术信息」表改用同一组行，两处内容一致。

### 1.4 随访（`ws_overview.js`，#103）

- 年增长率：管腔最大直径和瘤体体积两项。
  - 大字是首末两次；三次以上且最近两次不同时，另起一行「最近两次 +x /年」；
  - 悬停看起止日期、天数和总变化量（同经典 `growthFact`）。
- 小图：管腔最大直径、瘤体体积，本发布包的指标，以及经典的跨发布包指标（WSS p99 / 速度 p99）。
  - 每个发布包一条线：本结果蓝色 `#2a78d6`，其他发布包灰色 `#8a94a6`，两条以上时有图例；
  - 横轴按扫描日期等比例（有日期缺失时按顺序）；
  - 悬停用 `ui.chartTip`，点其他扫描的点打开那个结果；末端数值标签互不重叠。
- 「表格」：扫描、管腔最大直径、瘤体体积、本发布包指标；未填扫描日期的行标 *，表下说明。
- 管理员：时间线只在「看全部用户」开着时才带 `?all=1`（审计 #11）。

### 1.5 截面联动（`ws_detail.js`、`ws_overview.js`、`ws_lens.js`）

- 压降发现（V15）：
  - 在概览里点一条「沿程压降」，截面放到该分支弧长的 10%（中心线基准，倾角、偏移归零），切到「截面」页签；
  - 提示条用经典原句：「压降定义：左髂总 近端 10% 与远端 10% 弧长段的平均压差。截面已放在近端 10%；把位置滑到 90% 查看远端。」
  - 这条发现的证据透镜里有「截面：近端 10%」「远端 90%」两个按钮，从三维标记或 [ ] 键选中时也能用。
- 沿程曲线（V41）：截面开着时，在「沿程」页签点曲线，截面移到那个弧长（同经典 `sliceAtArc`：分支局部弧长 ÷ 中心线长度）。
  - 曲线上画一条橙色 `#eb6834` 竖线标出当前截面位置；用键盘、拖动或截面页移动截面时，竖线跟着走；
  - 截面落到画面外时，镜头转过去一次。
- 透镜：体场压力的值（统计、点、压力类发现）多一行「压力参考」，就是经典体场页的「压力参考：…」。

### 1.6 复核签字核对清单（`ws_review.js`，#82）

- 「技术复核」对话框在三个勾选项上面列出事实，逐项同经典工作台 `reviewChecklist`：
  - 出口命名：自动确认（置信度 x%）/ 已人工确认；
  - 结果质量；
  - 几何参考范围；
  - 需要注意：经典 `WssWorkbenchCore.alertModel` 的标题，这段逻辑已移植到 `ws_review.js`，S7 删掉 workbench_core.js 后照样可用。
- 有问题的行用警示色。

### 1.7 术语提示（`ws_lens.js`，W63 / V38）

- `ns.lens.loadGlossary / termText / termTip`：在线时读 `/static/glossary.json`（一页纸术语表也用这份文件），离线报告读内嵌的 `#wss-glossary`。读不到时用原来的内置文字。
- 本路文件里有对应术语的提示都改读术语表：管腔最大直径、瘤体、瘤颈、参考直径、全腔体积、OSI / RRT / ECAP、自动结论、随访与年增长率、多模型一致性、人群分位、迂曲度、体场术语、发布包 / 特征合同 / 运行身份。
- 术语表在结果打开前没读到时，读到后重画一次检查器。
- 其他路的文件仍用各自的内置文字，可以直接调用 `ns.lens.termTip(key, 内置文字)` 接入。

## 2. 与经典页面的一致性证据

沙箱 http://127.0.0.1:8823（预览任务副本），在同一个浏览器里先看新工作区，再打开同一结果的经典报告，逐项对照。脚本：`/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane3/walk.py`，结果：`…/walk_*.json`。

| 项目 | 病例 | 结果 |
|---|---|---|
| 警示条句子 | WANG_SHUN_WEN（1 项超范围 + 质量 review） | 与经典 `#warn-banner` 逐字相同 |
| 警示条句子 | DEMO_FOLLOWUP_B（2 项超范围 + 质量 review） | 逐字相同（「… 等」） |
| 无警示 | LV_GUO_YOU X5D | 两边都没有 |
| 技术信息 9 行 | WANG_SHUN_WEN | 发布包、哈希、模型族、run_identity、输入 SHA256、模型帧、生成时间、报告重建、摘要版本：9/9 逐字相同 |
| 几何参数表 | WANG_SHUN_WEN | 7 个分支 × 5 列逐格相同 |
| Murray 分流 | WANG_SHUN_WEN | 5 个开口的半径与分流 % 相同；「出口映射已人工确认」相同 |
| 质量 / 参照 | WANG_SHUN_WEN | 「存在不确定性，建议复核」「超出已声明几何参考范围，请复核」「已计算同协议 CV3 折外经验分位 45.6%」与经典卡片相同 |
| 体场口径文字 | LV_GUO_YOU 体场 | 压力参考、统计口径、「已载入 527 条…」与经典 `#pressure-reference` `#volume-protocol` `#streamline-note` 相同 |
| 压降定位 | LV_GUO_YOU 体场 F7 | 截面 {centerline, 左髂总, 0.10}，提示条文字同经典 |

自动测试里的对照（`tests/test_v2_p3_info.py`）：

- 经典壁面报告的 `bannerInfo` 原代码从 report.py 里截出来，在 Node 里跑，11 组构造数据逐句比对；同时比对经典体场页的 `VolumeViewerCore.warningText`。
- 6 个真实任务经 v2 manifest 后，警示句与经典相同。
- 经典 `renderTech` 原代码在 Node 里跑，3 个真实任务逐行比对。
- 经典工作台 `reviewChecklist`（app.js）与 `WssWorkbenchCore.alertModel` 逐项比对。这两个文件被第 6 路删除后，这一项自动跳过，固定的期望值仍会检查。

## 3. 与经典页面的差异和原因

1. **警示条比经典体场页多一种情况**：参考评估整体为 `review` 但没有单项 `review` 时，经典体场页不出提示（漏报），经典壁面页出「部分」。新工作区按壁面页口径，两种结果都会提示。
2. **出口确认的说法**：经典壁面页只看 `outlets_confirmed`，自动高把握度确认也写「已人工确认」。新工作区按出口命名记录的来源写「出口映射为自动确认（高把握度，未经人工核对）」；离线时看 manifest 的左右来源。
3. **技术信息「计算设备」**：设备和 GPU 写法相同时（如都是 cpu），只写一次。经典体场页写「cpu · cpu」。
4. **警示条的「详情」**：经典页跳到统计卡，新工作区打开对话框。概览「质量与参照」小节也打开同一个对话框。
5. **随访图**：
   - 经典每个发布包按家族配色；这里本结果蓝、其他发布包灰，按已校验的三色方案；
   - 经典的「勾选两次扫描并排比较」没有迁，用比较选择框里的「同一患者」代替。
6. **沿程点曲线移截面**：经典不动镜头；新工作区在截面落到画面外时转一次镜头。

## 4. 改动过的公共文件

- `static/v2/ws_shell.js`，四处，各一行或两行，都带 `P3 lane 3` 注释：
  1. `loadTimeline`：`api().timeline(pid, …)` 的第二个参数改为「管理员且 `ns.admin.viewAll()` 为真」（原来管理员总是 `all=1`）。
  2. `openSliceAt(plane)`：多接受一种参数 `{segment, fraction}`，按中心线站位放截面；原来的 `{origin, normal}` 照旧。按合同 §2 写明：这是新增的参数。
  3. `reviewDialog`：正文里在说明和三个勾选项之间插入 `ns.review.checklist(ui(), cur.manifest, cur.job)`（有才调用）。
  4. `renderTools`：完整档「技术信息」表的行改为 `ns.detail.techRows(m, cur.job)`（有才调用，否则用原来的行）。离第 6 路要删的「经典报告」一节隔开 3 行。
- `static/v2/v2_detail.css`（第二期 D 路文件，本期归第 3 路）：
  - 改了随访小图的线和点的颜色规则，颜色改由每条线自己给；
  - 新增本路的样式，在文件末尾的标记块里。
- Python，不在默认归属里，改动尽量小，都有测试：
  - `wss_deploy/v2_data.py`：
    - manifest 的 `provenance` 多几个字段：`model_family`、`n_models`、`created_at`（summary 的生成时间）、`time_hint`（mapping history 里第一个带时区的时间，同经典壁面页）、`report_rebuilt_at`、`gpu`；
    - `provenance.model_release` 多取 `registry_id`；
    - 新函数 `_tech_provenance`，新增 `import re`。
    - 只加字段，不改已有字段。manifest 的 ETag 会因此变化，旧缓存会重新验证一次。
  - `wss_deploy/v2_offline.py`：离线页多嵌一个 `<script type="application/json" id="wss-glossary">`，内容是 `glossary.glossary_document()`；文件头说明改为「四个 JSON 块」。
  - 没有改分析、流水线、叙述或经典报告生成，所以没跑黄金回归。
- `bundle.json`、`index.html` 没动：`ws_notice.js` 用的是基线占位。

## 5. 测试结果

- 新测试 `tests/test_v2_p3_info.py`，18 项，全部通过。
- 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`：
  - 第一次提交 cf174a1 后：949 通过 / 33 跳过；
  - 走查修正后（6599f34）：950 通过 / 33 跳过。
  - 跳过的是开发任务副本 `outputs/wss_deploy_jobs` 不在本工作副本的项。
- 改过的 js 都过了 `node --check`：`ws_notice.js`、`ws_overview.js`、`ws_lens.js`、`ws_detail.js`、`ws_review.js`、`ws_shell.js`。
- 已有测试没有改动。

## 6. 截图（沙箱，无头 Firefox，1440×900）

目录 `/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane3/shots/`：

- `01_banner_wall_wang.png` 警示条；`02_banner_details.png` 详情对话框；`02b_classic_banner_wang.png` 经典页对照；`03_banner_demo_b.png` 两项超范围
- `04_overview_quality.png` 质量与参照；`05_overview_geometry_murray.png` 几何参数与 Murray 分流
- `06_tech_info_dialog.png` 技术信息；`06b_classic_tech.png` 经典对照；`07_tools_tech_full_tier.png` 完整档工具页
- `08_review_checklist.png` 复核核对清单
- `09_followup_growth_releases.png` 随访增长率与多发布包曲线；`10_followup_table.png` 随访表格
- `11_volume_terms.png` 体场术语与口径；`12_pressure_drop_slice_10pct.png` 压降定位截面；`13_pressure_drop_lens.png` 压降透镜按钮；`14_profile_click_moves_slice.png` 沿程点曲线移截面
- `15_offline_banner.png` 离线警示条；`16_offline_tech.png` 离线技术信息

JS 错误：页面错误 0。最后一轮的控制台里有一条 Firefox 自身新标签页的报错（`ASRouter … NEWTAB_MESSAGE_REQUEST`，文件为空、行号 0），不是页面代码。

**沙箱数据说明**：
- 为了演示随访的「最近两次」和多发布包曲线，只在本路沙箱的任务副本里，把 4 个任务的 `job.json` 改为同一患者 P-DEMO-01 和演示用扫描日期：WANG_SHUN_WEN 2023-09-01，LV_GUO_YOU 的三个结果 2025-09-20。所以截图里的直径增长率数字没有临床意义。
- 人群参照「需复核」在预览任务里没有实例，只在自动测试里覆盖，没有在浏览器里展示。
- 预览任务本身、`outputs/wss_deploy_preview_jobs` 都没改。

## 7. 没做完 / 需要决定的

1. 比较模式下右侧结果的警示不显示：外壳没有「进入比较」的扩展钩子。警示条只说左侧（当前）结果；右侧结果的问题要单独打开它才看到。
2. 管理员切换「看全部用户」后，已打开结果的随访不会自动重读，要重新打开结果。
3. 「看全部用户」的状态读自第 2 路 `ws_admin.js` 的 `ns.admin.viewAll()`；没有 ws_admin 时按「关」处理。
4. 术语表只接到了本路的文件；`ws_display`、`ws_slice`、`ws_probe` 等其他路文件的 ⓘ 仍是内置文字（可调用 `ns.lens.termTip`）。
5. 工具页基本档的一句话「换模型重跑、技术信息和完整统计在「完整」档」（`ws_shell.renderTools`）现在不完全准确，技术信息在页脚各档都有。这句在外壳里，本路没改，建议第 6 路改「工具页」时顺手改掉。
6. 警示条在视口顶部居中，个别病例会挡住血管顶端一小块；可以关掉。没有改视口的取景留白。

## 8. 提交

- cf174a1 主体功能与测试
- 6599f34 走查修正：标签不重叠、说明只出一次、截面移出画面时转镜头、核对清单警示色
- 本报告单独提交
