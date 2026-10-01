# 第三期第 1 路：输入与未完成任务 — 交付报告

提交：bbea34b（功能 + 测试）、bedd511（补测试）、本报告。

分支 `wss-ui-p3-input`，工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_a`，基线 866cf02。范围见 `PHASE3_LANES.md` §3「第 1 路」，审计行 #65、#68、#69、#70、#72、#74、#75、#76、#77 全部做了；做完后去掉了工具栏里「在经典工作台中处理」（§3 第 5 项）。

代码：`static/v2/ws_input.js`（主文件）、`static/v2/v2_input.css`；`ws_shell.js` 只删了一行链接（见 §4）。测试：`tests/test_v2_p3_input.py`。

## 1. 做了什么，入口在哪

入口都是未完成任务的输入页（病例栏或任务列表点一个未完成任务，`#/job/<id>` 自动进入）。

| 审计行 | 功能 | 位置 |
|---|---|---|
| #69 | **失败任务可以重试**：`failed / cancelled / interrupted` 都有「重试」（中断的叫「恢复任务」）；已复核锁定或 `retryable: false` 时不给按钮，并说明原因。重试后页面自己跟着轮询，直到任务再次停下或完成（外壳只轮询打开时就在计算的任务） | 输入页右栏，错误卡第一行按钮 |
| #70 | **错误卡补齐**：按类别的标题（输入几何无法计算 / 计算工具出错 / 计算资源不足 / 本次计算未完成 / 任务中断，可恢复），类别说明放在标题旁 ⓘ；`retry_hint: cpu` 时一句「重试时改用 CPU 计算（较慢），不必重新上传」；诊断编号（等宽字，带复制按钮）；`admin_detail` 折叠在「技术细节」里（服务端只给管理员） | 同上 |
| #68 | **取消任务**：排队、计算中，以及待核对输入、待确认出口都能取消；确认对话框说明保留什么、怎么恢复（经典 `cancelJob` 的两句话）；已经在取消中的任务不再给按钮，进度标题显示「正在取消」 | 右栏头部状态行右侧「取消任务」 |
| #65 | **进度**：剩余时间标题 + 「正在：X（第 i / n 步）」，分段条按各阶段预计秒数分宽，进行中的一段按浏览器时钟每秒推进，超时段变橙；估计依据（几例历史、面片数）放 ⓘ；「各阶段耗时」展开表：阶段 / 预计 / 已用 / 状态（比预计慢标橙，已预计算的阶段注明）。同版本的新估计只重画进度，不重建右栏 | 计算中 / 排队任务的「进度」节 |
| #72 | **输入确认的事实表**：原始尺寸、换算后尺寸、开口数（≠ 5 标橙）、壁面面积与顶点数、连通片、拟删除面积比例；`errors / flags` 提示条；有 `summary.reference_assessment` 时一行参考范围与人群位置；`status: fail` 时说明不能继续并只给「修改后重新上传」；碎片勾选项写明面积比例；默认单位也参考 `suggested_units`；确认按钮旁「确认后约 N 秒完成中心线提取」。失败任务的同一张表收在「输入尺寸与面积」里；失败但输入仍待确认（`needs_confirmation`）时直接给单位卡（服务端允许这种恢复） | 待核对输入页；失败页 |
| #74 | **重新选择入口、重提中心线**：「入口」一节显示当前入口开口；「入口选错了？」展开后可从列表选，或**在三维里直接点开口**（三维切换为显示全部开口，当前入口、选中开口分色）；「重提中心线」发 `/confirm {mapping, inlet, acknowledged, version}`，与经典 `outletCard` 相同 | 待确认出口页右栏底部 |
| #75 | **出口确认细节**：核对原因改成人话（经典 `WB.outletReview` 的规则），第一条常显、其余收在「更多原因 · N」；表头旁是「自动命名置信度 88%，需要人工核对」；行悬停显示世界坐标；三维端点可点，点中选中对应行（反过来点行也会在三维放大该端点）；「已修改 N 个出口：#5 → 右髂外；…」；确认按钮旁「确认后约 N 秒出结果」；视口右上角「复位视角」「保存截图」 | 待确认出口页 |
| #76 | **下一例**：确认出口成功后提示「出口已确认，开始预测。还有等待确认的任务：X。」并给「处理下一例」；技术复核通过后提示「技术复核已记录，结果已锁定。还有未复核的结果：X。」并给「下一例待复核」。选法同经典（最早上传的那个优先；管理员开着「看全部用户」时一并算上） | 提示条 |
| #77 | **同时预测 / 来源说明**：「与「周期指标 TAWSS · OSI」同时上传，沿用它的中心线和出口确认 · 打开」「同时预测「体内压力与速度」：确认出口后自动创建 / 未能创建（原因）/ 打开」「复用已有结果的中心线和出口确认，只重新预测 · 打开来源」「换模型重跑，沿用来源结果的中心线和出口确认 · 打开来源」 | 未完成任务：右栏头部下方；已完成结果：「工具」页末尾「来源」一节（扩展点 `tools`） |
| §3-5 | 去掉工具栏「在经典工作台中处理」 | `ws_shell.js` `showInput` |

另外顺手修的（都在自己的文件里）：

- 确认出口后任务转入计算，视口左上角的状态胶囊原来一直停在「待确认出口」，现在跟着改（`syncTitle`）。
- 视口提示「计算完成后这里显示结果。」原来压在状态胶囊上，改为顶部居中的小胶囊。
- 已取消、开口正常的任务不再列开口表；记录写着已完成而结果文件还读不到（与 worker 的竞态）时，给中性的「结果文件还在准备」和「打开结果」，不再显示「本次计算未完成」。

少字的处理：类别说明、估计依据、「为什么要核对出口」（STL 没有患者方向、XYZ 不是患者方位）、单位卡的服务端门限说明都在 ⓘ；核对原因只常显一条。

## 2. 与经典页面的一致性

**接口与载荷同源**（经典 `app.js` `mutate` = `{...payload, version}`）：

| 动作 | 请求 | 测试 |
|---|---|---|
| 重试 / 恢复 | `POST /api/jobs/<id>/retry {version}` | `test_failed_retryable_error_card_retries_with_classic_payload` |
| 取消 | `POST /api/jobs/<id>/cancel {version}` | `test_cancel_while_waiting_and_while_running` |
| 确认输入 | `POST /api/jobs/<id>/input {units, remove_fragments, acknowledged, version}` | `test_units_card_facts_flags_reference_and_fail_block` |
| 确认出口 | `POST /api/jobs/<id>/confirm {mapping, acknowledged, version}` | `test_outlet_details_confirm_and_next_case` |
| 重提中心线 | `POST /api/jobs/<id>/confirm {mapping, inlet, acknowledged, version}` | `test_repick_inlet_from_list_or_3d_sends_inlet` |
| 下一例 | `GET /api/jobs?page=1&page_size=100[&all=1]` | 同上、`test_review_approval_offers_the_next_unreviewed_result` |

**规则同源**：经典 `workbench_core.js` 的 `etaView / formatDuration / stageRemaining`、`outletReview / humanOutletReason`、`nextToConfirm / nextToReview` 移植进 `ws_input.js`（S7 要删 `workbench_core.js`，所以不能运行时依赖它）。核对办法：

1. 2026-09-30 在 Node 里把两边对 6 组 ETA 快照（运行中超时、排队、待确认、待输入、等前一任务、严重超时）、一组完整出口提案（含 `原因：` 的 detail）、6 个任务的「下一例」、11 个时长逐项比较，**完全相同**（JSON 全等）。
2. 这些值固化在 `test_ported_rules_equal_the_classic_workbench_core` 里，测试不读 `workbench_core.js`，第 6 路删掉它后仍然有效。

**数字逐项比对**（同一沙箱、同一任务 `20260930_045135_8412b6e4b1ac`，经典工作台详情「查看输入检查与计算过程」vs 新工作区「输入尺寸与面积」）：

| 项 | 经典 | 新工作区 | 原始值 |
|---|---|---|---|
| 原始尺寸 | 96.7 × 113.5 × 260.8 | 96.7 × 113 × 261 | 96.672 × 113.499 × 260.776 |
| 换算后 mm | 96.7 × 113.5 × 260.8 | 96.7 × 113 × 261 mm | 同上（× 1） |
| 开口数 | 6 | 6 | 6 |
| 壁面面积 | 340.6 cm²（11,638 个顶点） | 341 cm² · 11638 个顶点 | 34060.34 mm² |
| 连通片 | 1 | 1 | 1 |
| 拟删除面积 | 0.000% | 0.000% | 0 |

错误卡标题与说明逐字相同（「输入几何无法计算」「问题出在上传的 STL 本身，重试不会改变结果…」）；`retryable: false` 时两边都不给重试。ETA、出口原因、下一例见上一条（全等）。

## 3. 与经典页面的差异及原因

- **数字位数**：新工作区统一三位有效数字（合同 §7），经典是固定一位小数；同一原始值。拟删除面积保留经典的三位小数百分比（门限是 1%）。
- **开口编号从 1 数**：失败页一直是「开口 1…n」，入口选择也改成 `opening_id + 1`，两处一致；发给服务端的仍是原始 `opening_id`。经典入口下拉显示的是 0 起的 `opening_id`。
- **「重提中心线」在选中当前入口时不可用**：经典会照发（服务端只比较 `params.inlet`，`null` ≠ 当前编号，会白算一次中心线）。
- **取消按钮位置**：统一放在右栏头部状态行（经典在确认页头部和状态行两处）；文案「取消任务」，对话框按钮「不取消 / 取消任务」。
- **世界坐标**：经典是表格第四列；右栏窄，改为行悬停提示。
- **核对原因**：经典整段收在「为什么需要核对」里；这里第一条（通常是哪一侧要重点核对）常显，其余「更多原因 · N」。
- **下一例的措辞**：新工作区把「签字」叫「技术复核」，提示与按钮相应改为「未复核的结果」「下一例待复核」。
- **五步步进条**：审计 #66 判放弃，没做。
- **重试的位置**：按钮放在错误信息下面、开口表之前（经典是单独的「继续处理」卡片在错误卡之后）。

## 4. 改动过的公共文件

- `static/v2/ws_shell.js`：**只删一行**（合同允许的那一项）。`showInput` 里

  ```js
  ui().fill(S.els.toolbar, h('span', {'class': 'tb-title', …}), h('span', {'class': 'sec-fill'}),
    api() ? ui().link('在经典工作台中处理', api().urls.classic(job.id), {newTab: true}) : null);
  ```

  改为去掉最后一个参数。没有加 `SHELL_API` 助手。
- `bundle.json`、`index.html`：没动（`ws_input.js`、`v2_input.css` 基线已在）。
- 没有改 Python。

**运行时接缝**（不改别人的文件，但合并时需要知道）：

1. `ns.api.review` 被 `ws_input.js` 包了一层（`watchReview`，幂等）：同一请求、同一返回，只是复核「通过」成功后多发一次任务列表请求并给「下一例待复核」提示。这样第 3 路不管把复核对话框留在外壳还是搬进 `ws_review.js`，都不用再接线。主会话若更愿意显式调用，可以在复核成功处加 `ns.input && ns.input.offerNext('review', jobId)` 并删掉 `watchReview()` 一行。
2. 读外壳状态：`ns.shell.state().pollTimer`（判断外壳是否已在轮询，避免两边同时轮询）、`.jobs`（来源说明里给另一份结果起名）、`.els.toolbar` 里的 `.tb-title`（状态词跟着改）。外壳的 `onChanged` 本身不更新这个胶囊，若主会话在外壳里修，`syncTitle` 可以删。
3. 读 `ns.admin.viewAll()`（有就调用）决定「下一例」是否带 `all=1`。
4. 注册扩展 `ns.ext {id: 'input-provenance', tools}`：只在已完成结果有来源信息时，在「工具」页末尾加「来源」一节。
5. 导出给别的路复用：`ns.input.etaView / formatDuration`（第 2 路 #30 列表行剩余时间可以直接用，与经典 `etaRowSummary` 同一规则）、`nextJob / offerNext`、`errorModel`、`outletReview`、`provenance`。

## 5. 测试结果

- `tests/test_v2_p3_input.py`：12 个，全过。覆盖：移植规则与经典固化值全等；资源类错误卡（标题、CPU 提示、诊断编号、技术细节、ⓘ）与重试载荷、重试后自行跟踪到再次失败；不可重试 / 已锁定 / 中断 / 已取消四种恢复卡；四种状态的取消与「正在取消」；进度分段、阶段表、同版本只重画进度；单位卡事实表、提示、参考范围、阻断；出口确认细节（原因、坐标、三维点选、已修改、ETA）+ 确认载荷 + 「处理下一例」跳转；重新选择入口（列表 / 三维）载荷；来源说明（输入页与工具页）；复核通过后的「下一例待复核」；已取消隐藏开口表与已完成竞态。
- 已有测试没有改。
- 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`（最终代码 bedd511 之后）：**944 通过 / 33 跳过**（基线工作副本上跳过项是缺少 GPU / 正式任务副本的那些，与本路无关）。
- 改过的 js 都过了 `node --check`。
- 没有改分析、流水线、叙述或报告生成的 Python，按合同不需要黄金回归。

## 6. 浏览器走查

沙箱：`python -m wss_deploy.devshot sandbox --port 8821 --login admin:…`，任务根 `/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane1/sandbox`，无头 Firefox（Marionette 3021，比对用 3031）。除预览任务副本外，为了复现各种状态，沙箱里另做了五份改了状态的副本（只在沙箱里）：

| 副本 | 来源 | 改成 |
|---|---|---|
| `20260930_200001_p3confirm001` | 待确认出口演示（已被确认过） | 退回待确认出口，带一个待创建的体场伴随任务 |
| `20260930_200002_p3inlet00002` | 同上 | 退回待确认出口（用来改入口） |
| `20260930_200005_p3cancel0005` | 同上 | 退回待确认出口，伴随任务「未能创建」 |
| `20260930_200003_p3retry00003` | 失败演示 | 错误改成可重试的显存不足（CPU 提示、诊断编号、技术细节） |
| `20260930_200004_p3units00004` | 待确认出口演示 | 退回待核对输入（2 个连通片、0.041% 拟删除、单位把握度 62%） |

走查脚本 `/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane1/walk.py`，全部是真实点击和真实服务请求；三维点选用页面算出的端点屏幕坐标发真实 `pointerdown / pointerup`（经过 three.js 射线拾取）。沙箱 CPU 上 B 段只要几秒，看不到长时间的计算中页，所以「计算中 / 排队」两张进度截图是在页面里把 `api.job` 对两个虚构编号答成带经典 `eta` 块的快照（其余都是真的）。

最终一轮在全新的沙箱副本上跑，26 张截图，**页面 JS 错误 0**；数字比对那一轮的浏览器日志里只有一条 Firefox 自己新标签页的 `ASRouter … NEWTAB_MESSAGE_REQUEST`（浏览器内部，不是页面脚本）。走过的步骤：失败页（不可重试）与「输入尺寸与面积」→ 可重试失败页 → 点「重试」→ 自动跟到再次失败 → 待核对输入事实表 → 取消待确认任务（对话框 → 已取消 → 有「重试」）→ 改入口（三维点开口 4 → 下拉跟着变 → 重提中心线 → 计算中 → 新的待确认出口）→ 出口确认（更多原因、三维点 #5 选中行、右侧内 / 外互换、已修改 2 个出口、确认 → 「处理下一例」提示 → 点击跳到 P3_INLET）→ 确认单位 → 计算中 → 待确认出口 → 计算中 / 排队进度与阶段表 → 技术复核通过 → 「下一例待复核」→ 点击打开下一例 → 1024 宽。

截图目录：`/public/newhome/cy/.claude/jobs/83844143/tmp/p3_lane1/shots/`

| 文件 | 内容 |
|---|---|
| `01_failed_input_geometry.png`、`01b_failed_facts.png` | 不可重试的失败页、展开输入事实 |
| `02_failed_resource_card.png`、`02b_after_retry.png`、`02c_retry_result.png` | 资源类错误卡（技术细节展开）、重试后排队、自动跟到结果 |
| `03_units_facts.png`、`03b_units_confirmed_progress.png`、`03c_units_next_state.png` | 单位卡事实表、确认后进度、下一状态 |
| `08_cancel_waiting.png`、`08b_cancel_dialog.png`、`08c_cancelled.png` | 待确认任务取消 |
| `09_inlet_pick.png`、`09b_inlet_redo.png`、`09c_inlet_after.png` | 三维点选入口、重提、新的出口提案 |
| `04_outlets.png`、`04b_outlets_pick_more.png`、`04c_outlets_changed.png`、`04d_confirmed_toast_progress.png`、`05_next_case.png` | 出口确认细节、确认后「处理下一例」、跳转 |
| `12_running_progress.png`、`12b_running_stage_table.png`、`12c_queued_progress.png` | 计算中进度、阶段表、排队 |
| `10_review_dialog.png`、`10b_review_next_toast.png`、`10c_review_next_opened.png` | 技术复核后的下一例 |
| `11_narrow_1024.png` | 1024 × 768 |
| `parity_classic_8412b6e4b1ac.png`、`parity_v2_8412b6e4b1ac.png` | §2 数字比对的两边 |

## 7. 没做的、要决定的

- **#78**（待确认 / 计算中页也能展开「计算过程」）不在本路范围，没做；失败页的输入事实已有。
- **外壳的状态胶囊**：外壳 `showInput` 只写一次胶囊，`ws_input` 用 `syncTitle` 补；更干净的做法是外壳在 `onChanged` 里自己更新（第 6 路或主会话）。
- **外壳轮询**：外壳只轮询打开时就在计算 / 等待的任务；重试把失败任务变回排队后，由输入页自己轮询（`ensurePoll`，外壳一开始轮询就停）。要不要改成外壳统一处理，由主会话决定。
- **复核后的下一例**是包 `api.review` 实现的（§4 接缝 1），合第 3 路时请确认复核仍走 `api().review`。
- 经典出口确认里的「复位视角」「保存截图」已迁；三维中心线折线（经典 `preview_polylines`）没有画，三维仍只画壁面和端点。
