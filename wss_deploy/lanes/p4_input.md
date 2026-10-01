# 最后一轮迁移 · A 路：输入页、上传、工具页小项 — 交付报告

分支 `wss-ui-p4-input`（基线 e7c2356），工作副本 `/public/newhome/cy/Digital_twin/GNN_lane_a`。依据：复核表 `reaudit_workbench.md`（#26、#73、#78、#102）、`reaudit_reports.md`（W30、W35）、`reaudit_ops_s7.md`（F6）。

提交：e17f0b3（#73 / #78 / #102 / W30 / W35）、599d55f（#26）、a19df6b（新测试）、d3d8443（F6）、9749862、65dcd6e（走查后的小修：上传错误在收起区也展开、过程卡两行对齐、单阶段耗时条填满、上传表格两列顶端对齐）、本报告。

## 1. 做了什么，入口在哪

| 项 | 内容 | 入口 |
|---|---|---|
| #73 | 出口确认的三维里画中心线：取 `/geometry` 的 `preview_polylines`，没有时退回阶段 A 提案里的；只画三维坐标有限、至少两点的线。颜色和画法同经典 `createViewer`（`0x567891`，不透明度 0.9，不做深度测试，压在壁面上、在端点标记下）。没有显示网格的老任务按中心线取景，不再显示「没有可显示的网格」。「入口选错了？」模式下中心线保留 | 待确认出口页三维（`ws_input.meshView.setLines`） |
| #26 | 上传对话框即时校验病例名称（多文件时为「名称前缀」）、患者编号、扫描标签、标签：规则调用结果页的 `ns.detail.identifierIssue`，没有该文件时用同规则的本地副本；标签用结果页 `metadataError` 的两条（最多 12 个、每个 ≤ 40 字），另拒绝标签里的制表符（服务端也拒）。错误写在字段下方、输入框标红、「上传并检查」禁用，脚本调用 `send` 也会拦下并说明；「看起来像真实姓名」只提醒不拦。错误在收起的「扫描标签、标签、备注」里时自动展开 | 上传对话框 |
| #102 | 「计算过程」补经典 `processCard` 的两行：「中心线检查 · 通过 / 未通过」（带状态点）、「中心线端点 / 分叉 · 5 / 3」（`hard_pass`、`topology.endpoints / junctions`） | 工具页「计算过程」；输入页同一节 |
| #78 | 排队、计算中、待确认出口页底部加「输入检查与计算过程」（收起，点开），内容就是工具页的「输入检查」「计算过程」两节（从 `ws_detail` 复用：`inputSectionFor` / `processSectionFor`）。失败 / 取消 / 中断页已有自己的输入事实和检查表，只加「计算过程」。同经典：待核对单位时不显示；还没有输入检查（刚排队）时不显示 | 输入页右栏底部 |
| W30 | 出口命名的「把握度 x%」旁加 ⓘ：「把握度由几何形态估算，不是经过临床数据校准的概率。」（经典工作台原句）；门控已绑定校准时改为经典报告的「已绑定独立校准 profile。」。输入页的「自动命名置信度 x%」把这句放在原 ⓘ 的最前面 | 工具页「出口命名」；待确认出口页标题旁 ⓘ |
| W35 | 出口命名建议自身的告警（manifest `analysis.flags` = summary 顶层 flags，另并入阶段 A 提案的 flags，去重）用经典工作台的说法（`humanOutletReason`）列在「命名时的核对提示」下；修改命名时也显示。输入页：summary 的 flags 并入「更多原因」（提案的 flags 原来就在） | 工具页「出口命名」；待确认出口页 |
| F6 | `test_v2_p3_info.py` 的复核清单与 `alertTitle`、`test_v2_p3_workbench.py` 的 `etaView` / `etaRowSummary` / `formatDuration` / `cohortFilter`：用 866cf02 的 `app.js`、`workbench_core.js` 算出对照值，固化成常量（`CLASSIC_REVIEW`、`CLASSIC_ETA` 等），无条件断言；`alertTitle` 重新有断言 | 测试 |

## 2. 与经典的一致性

- **中心线**：数据源和取法同经典 `outletCard` → `createViewer`（`geometry.preview_polylines` 非空就用，否则 `a.proposal.preview_polylines`），材质参数逐项相同（测试断言 `[0x567891, 0.9, depthTest false, transparent]`）。沙箱 `20260930_122126_e434ae6d6df2`：7 条中心线（424 / 58 / 77 / 123 / 122 / 138 / 135 点）全部画出。
- **编号校验**：本地副本与 `ws_detail.identifierIssue` 对 9 组输入（空、正常、制表符、换行、超长、首尾空格、两种姓名、U+0085）结果逐项相同；`ws_detail.identifierIssue` 本身是经典 `WssWorkbenchCore.identifierIssue` 的移植（字句相同）。标签两条与结果页 `metadataError` 文案相同。
- **中心线两行**：取值同经典 `processCard`（`hard_pass ? '通过' : '未通过'`，`endpoints ?? '—' / junctions ?? '—'`）。沙箱 `20260920_144135_673ccd0e36b1`：job.json `hard_pass: true`、`endpoints 5`、`junctions 3` → 显示「通过」「5 / 3」。
- **把握度说明**：两句话分别取自经典工作台 `humanOutletReason`（代理值）和经典报告门控卡（validated profile）。
- **flags**：`673ccd` 的 `summary.flags` = 「右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对」→ 显示「右侧髂内与髂外的区分把握度只有 88%，请重点核对右侧的两个出口。」（经典工作台规则）。
- **F6**：固化前先确认现行代码与经典文件逐项相同（6 组复核清单 + 标题、16 组 ETA、11 个时长、7 组队列筛选），再写成常量。

## 3. 与经典的差异

- 经典 `processCard` 在中心线记录为空对象（这一步没跑）时也写「未通过」；这里不显示这两行，避免把「没跑」说成「没通过」。
- 经典 `processCard` 是一张卡里的长列表；这里复用新工作区工具页的两节（分段条、图例、运行记录收起），字段相同、排版不同。
- 经典的出口三维壁面半透明（0.3），新工作区壁面不透明，中心线靠不做深度测试透出来；颜色相同。
- 上传标签多拦一种情况（标签里有制表符），结果页没有这条；服务端本来就拒绝。

## 4. 改动过的公共文件

无。只改了本路文件：`ws_input.js`、`ws_upload.js`、`ws_detail.js`、`v2_input.css`、`v2_workbench.css`（只加上传提示的三条规则）；测试 `test_v2_p3_info.py`、`test_v2_p3_workbench.py`（F6，原因见上）、新建 `test_v2_p4_input.py`。`ws_shell.js`、`bundle.json`、`index.html`、Python 都没动。

## 5. 测试

- `node --check`：`ws_input.js`、`ws_upload.js`、`ws_detail.js` 通过。
- 新测试 `tests/test_v2_p4_input.py` 8 项：中心线（三维桩记录线条和材质）、编号规则对照、上传对话框拦截与放行、未完成页的过程卡、中心线两行、输入页 ⓘ 与 flags、工具页 ⓘ / flags / 两行。
- 全量 `PYTHONPATH=. pytest -q -p no:cacheprovider tests`（最终提交上）：**995 通过，33 跳过**，0 失败。跳过全部是环境原因（本工作副本没有 `outputs/wss_deploy_jobs` 开发任务副本 26 项、`WSS_DEPLOY_DEVSHOT` / 黄金回归 / 慢测试开关、示例 STL 未生成），与本路改动无关；合计 1028 = 合并时的 1020 + 本路新增 8。
- 没改 Python，不需要黄金回归。

## 6. 沙箱浏览器走查

沙箱 8821 / Marionette 3021，预览任务副本（`p4_laneA/sandbox`），脚本 `/public/newhome/cy/.claude/jobs/83844143/tmp/p4_laneA/shots_p4a.py`，结果 `shots.json`。待确认 / 计算中 / 排队状态是在页面里包一层 `WSSV2.api.job` 改状态字段得到的（几何和输入检查仍由沙箱服务返回），没有改磁盘上的任务。全部步骤 **0 个 JS 错误**。

截图（`/public/newhome/cy/.claude/jobs/83844143/tmp/p4_laneA/shots/`）：

| 文件 | 内容 |
|---|---|
| `01_outlet_centreline.png` | 待确认出口：壁面上的中心线；标题 ⓘ 以把握度说明开头 |
| `02_outlet_process.png` | 同页「输入检查与计算过程」展开，含中心线两行 |
| `03_running_process.png` | 计算中页：进度 + 过程卡 |
| `04_queued.png` | 排队页：过程卡入口 |
| `05_failed_process.png` | 失败任务：只有「计算过程」；无中心线 |
| `06_tools_outlets.png` | 工具页出口命名：把握度 ⓘ、「命名时的核对提示」、中心线两行 |
| `07_tools_process.png` | 工具页计算过程 |
| `08_upload_hints.png`、`08b_upload_tags.png` | 上传：患者编号含制表符（红）、姓名提醒（橙）、标签 13 个（红，自动展开），按钮禁用 |
| `09_upload_clear.png` | 改正后提示消失，按钮可用 |

## 7. 没做 / 留给后面

- W30 还有一处不在本路文件：复核签字对话框的核对清单「出口命名 · 自动确认（置信度 98.8%）」（`ws_review.js`），按 F6 要与经典逐字相同，没加 ⓘ。概览的「高把握度」（`ws_overview.js`，B 路）不带数字，没动。
- 过程卡在计算中页不会随轮询自动刷新阶段耗时（随状态 / 版本变化重画，同其他节）。
