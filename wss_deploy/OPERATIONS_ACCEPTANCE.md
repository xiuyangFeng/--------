# 运维系统验收

本次新增 `/ops` 运维中心和 `/support` 用户工单。原有 `index.html`、`app.js`、`app.css` 与主页面管理员入口保持原样。使用规则、接口和留存说明见 [OPERATIONS.md](OPERATIONS.md)。

## 打开验收实例

独立实例使用合成账号、算例、STL 和工单，数据位于 `outputs/wss_ops_acceptance_20260929/`。它与正式 `outputs/wss_deploy_jobs/` 分开，既不读取正式病例，也不启动模型推理、训练或算例计算工作线程。

- 运维页面：<http://127.0.0.1:18765/ops>
- 用户工单：<http://127.0.0.1:18765/support>
- 本次随机生成的验收账号与口令：`outputs/wss_ops_acceptance_20260929/credentials.json`（权限 0600）。`ops_admin` 是管理员，`researcher_a` / `researcher_b` 是普通用户；这些账号不能用于正式服务。
- 进程与地址：同目录 `preview.json`；启动日志：`preview.log`。

浏览器若在自己的电脑而服务运行在远端，通过现有 SSH 连接转发后打开上述地址：

```bash
ssh -N -L 18765:127.0.0.1:18765 <现有SSH登录目标>
```

建议用两个浏览器配置文件分别登录管理员和普通用户。同一浏览器的不同标签页共用登录会话；切换账号后页面会清空旧身份的数据和草稿。验收实例使用独立 cookie，避免覆盖同一主机正式服务的登录。

验收页面顶部有「独立验收实例 · 合成数据」提示。由于不启动计算工作线程，服务健康区提示计算服务未就绪属于预期，不代表工单和审计功能不可用。合成 STL 仅测试留存字节与追溯，不用于医学计算或模型优化。

## 检查步骤

| 步骤 | 操作 | 预期结果 |
|---|---|---|
| 1 | 用 `ops_admin` 登录 `/ops` | 按概览、事件流、任务与回收、工单、STL 素材、用户切换；概览指标可跳转到失败任务、即将到期回收站、未分配工单、待审阅素材 |
| 2 | 用 `researcher_a` 登录 `/support` | 只看到 A 的工单；访问 `/ops` 不能取得全站数据 |
| 3 | A 创建新工单并回复 | 自动出现自己的工单，回复保留；另一浏览器的管理员页面约 10 秒内更新 |
| 4 | 管理员分配处理人、改优先级、回复并解决工单 | A 页面能读到处理进度；B 看不到 A 的工单 |
| 5 | 在两个管理员窗口同时处理同一工单 | 旧版本保存返回冲突，页面允许保留草稿并加载最新版本，不静默覆盖 |
| 6 | 在任务或回收站勾选多条记录，点「归档所选 STL」，再去素材库展开来源、审阅并下载 | 逐项报告结果，成功项取消勾选、失败项保留；重复归档去重；下载包含 `input.stl` 与 `manifest.json` |
| 7 | 查看事件流，展开单条记录，再展开原始 JSON；组合账号、类别、级别、时间筛选并导出 | 默认按日期分组只展示摘要；详情逐级展开，JSON 导出使用已经应用的筛选条件 |
| 8 | 管理员停用 B，再用 B 原会话刷新；随后重新启用 B | 停用使会话失效；启用后需重新登录；不能停用当前管理员或最后一位管理员 |
| 9 | 展开事件详情、暂停更新或翻到历史页，再从另一窗口提交工单 | 约 10 秒后提示「有新动态」，正在阅读的列表不跳动；点击提示或「回到最新」读取新记录 |
| 10 | 输入回复草稿，等待自动刷新，快速筛选/翻页，再切换页面宽度 | 草稿不被轮询清空，列表最终与筛选一致；320/390 px 六个视图没有整体横向溢出 |
| 11 | 重启独立验收实例后再次检查 | 工单、素材、来源和审计持久保留；不重新生成账号或覆盖已有数据 |

自动测试另覆盖：原算例移入回收站并永久清理后，已归档 STL 仍能下载；篡改摘要、路径穿越、符号链接、跨用户请求、CSRF 与来源检查；写审计失败时工单/素材事务回滚；CLI 账号维护审计。

## 重启与自动退出

独立实例默认 48 小时后自动退出。需要重开时在项目根目录运行以下命令；同一目录已有进程会被写锁阻止，不能同时启动两个实例：

```bash
PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m tests.ops_demo \
  --root outputs/wss_ops_acceptance_20260929 --port 18765 --hours 48
```

工具只接受空目录或带有本验收标记的专用目录，不接受已有正式任务目录。Ctrl+C 可停止前台启动的实例。后台实例 PID 记录于 `preview.json`，停止前应核对它仍是上述 `tests.ops_demo` 进程。

## 可重复验证

```bash
PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m pytest -q \
  tests/test_operations_store.py tests/test_operations_http.py tests/test_operations_cli.py tests/test_operations_acceptance.py tests/test_operations_feed.py \
  tests/test_users.py tests/test_trash.py tests/test_delete_jobs.py \
  tests/test_server_security.py tests/test_server_hardening_v015.py tests/test_server_p2.py

PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m tests.ops_browser_check \
  --out outputs/wss_ops_browser_check

PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m tests.ops_browser_edges \
  --out outputs/wss_ops_browser_edges

PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m tests.ops_browser_feed \
  --out outputs/wss_ops_browser_feed
```

浏览器验证需要已有的系统 Firefox，通过仓库 `devshot` 工具生成桌面/手机尺寸截图及 `browser-check.json`；使用临时合成数据，结束清理测试服务。它不要求 Selenium、外部数据库或前端构建工具。

窄屏检查通过测试辅助模块校准浏览器缩放，严格断言 `window.innerWidth/innerHeight` 为 320×844 或 390×844，避免 Firefox 最小窗口宽度造成误判。报告保存实际视口、内容宽度和滚动宽度。

正式服务加载新路由仍应按 [上线手册](env/GO_LIVE.md) 进行演练、排空和升级；独立验收实例不能代替正式模型计算的上线烟测。

## 本轮优化与验证记录（2026-09-29）

本轮将单页长列表拆为六个视图；审计改为按日期分组的事件流，默认显示操作、账号和对象摘要，点击展开详情及原始 JSON。补充高级筛选、筛选导出、暂停更新与新动态提示、概览待办跳转、批量 STL 归档和素材来源展开。

- 存储、HTTP、CLI、事件流接口及既有安全/任务回归：**101 passed in 55.82s**。
- Firefox 主流程：8 项通过，包含登录、跨账号任务、STL 归档、用户工单和管理员处理、自动刷新、身份切换；报告为 `outputs/wss_ops_stream_20260929/main/browser-check.json`。
- Firefox 原有边界场景：12 项通过，包括冲突后保留草稿、编辑焦点、快速筛选、会话清理和手机布局；报告为 `outputs/wss_ops_stream_20260929/edges/browser-edges.json`。
- Firefox 新事件流和管理流程：13 项通过，检查逐级展开、新动态不打断阅读、暂停与恢复、历史页保持、按已应用条件导出、概览跳转、用户筛选、批量归档部分成功与失败重试、素材来源、身份切换清理及延迟写响应不会恢复旧账号内容。报告为 `outputs/wss_ops_stream_20260929/feed/browser-feed.json`。
- 桌面及真实 320/390 px 视口覆盖六个视图，包含展开事件详情；JavaScript 错误为 0。
- 更新后的运行实例截图和源文件摘要位于 `outputs/wss_ops_stream_20260929/preview/`、`outputs/wss_ops_stream_20260929/delivery.json`。之前的验收证据保留在 `outputs/wss_ops_acceptance_20260929/`。
- 独立验收实例继续使用 `127.0.0.1:18765` 与已有合成账号、工单和素材；原工作台页面未改，正式 `8765` 服务未重启。
