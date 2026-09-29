# 独立运维中心

现有预测工作台及其管理员功能保持原样。运维中心与工作台共用同一服务、具名账号和会话，不启动第二个任务管理器，也不需要安装前端框架或额外数据库服务。

首次检查可按 [验收说明](OPERATIONS_ACCEPTANCE.md) 打开独立合成数据实例，或重复运行浏览器验收。

## 入口与权限

- `/ops`：管理员运维中心，可直接登录；共享服务只有 `admin` 能读取和修改运维数据。普通用户和旧共享令牌不能获取全站数据。本机非共享服务沿用现有本地操作员权限。
- `/support`：用户提交问题、关联自己的任务、查看处理进度并回复工单。各用户只看到自己的工单；管理员在 `/ops` 集中处理。
- 两个页面每 10 秒刷新一次，浏览器切到后台时暂停轮询。断线、登录失效和版本冲突会在页面提示，不会凭空展示成功。

运维中心按「概览、事件流、任务与回收、工单、STL 素材、用户」分开显示。概览提供失败任务、待处理工单、待审阅素材等快捷入口。

首次使用共享服务时，沿用现有命令创建管理员（交互输入口令）：

```bash
~/.conda/envs/GNN/bin/python -m wss_deploy.cli user add ops_admin --admin
```

已运行的服务需按 [上线手册](env/GO_LIVE.md) 演练、排空并升级后才会加载新接口。新增源码不会自动重启服务。

## 日常使用

1. 查看服务健康、进行中与失败任务、账号、工单和素材数量。
2. 按账号、状态、关键词筛选任务，切换当前任务和回收站；回收站记录显示到期时间。
3. 在事件流中按时间查看任务提交与状态变化、登录、删除、恢复、永久清理，以及工单、账号、素材操作。默认只显示摘要；展开单条事件查看操作者、对象及详细字段，原始 JSON 再单独展开。
4. 处理工单：分配处理人、调整优先级和状态、回复用户。并发修改以版本校验保护；遇到冲突先刷新再保存。
5. 必要时停用或重新启用账号。停用会使原会话失效；界面不能停用自己或最后一位启用的管理员。

事件流可按关键词、所属账号、类别、级别、操作代码及时间范围组合筛选。时间输入使用浏览器本地时区，发送到服务端时转为含时区的 ISO 时间。页面的「导出筛选结果 JSON」使用已经应用的筛选条件，导出最多最近 10,000 条匹配事件，包含筛选口径和截断标记；输入后尚未点击「应用筛选」的草稿不改变导出范围。导出仅管理员可用，审计内容保留运维所需的账号和操作信息。

事件流仍每 10 秒检查一次新记录。暂停更新、查看历史页或展开详情时，不用后台响应替换正在阅读的内容；有新记录时显示「有新动态」，点击后读取最新列表。该提示不把单页差异伪装成精确新增总数。

跨用户任务与回收站在运维中心供检查及 STL 归档使用，原有任务删除、恢复与永久删除仍遵守工作台的 owner 权限和审阅锁。

CLI 的账号创建、口令修改、角色调整、停用及任务认领也同步进入运维审计库，保留现有服务日志，不记录口令。工单和素材状态变更与其审计事件在同一数据库事务中提交；账号启停写入既有 `users.json`，若独立审计库故障，页面会明确提示「账号状态已保存但审计失败」，服务日志保留操作记录。CLI 账号恢复在审计库损坏时仍可执行，并在终端提示审计问题。

## STL 留存与模型优化

从任务或回收站明确选择「归档 STL」，把原始上传复制到独立素材库。系统按文件 SHA256 去重，同时保存每次来源的任务、账号、发布包与归档时间。归档失败会返回错误，不会只生成一个空的素材记录。

任务列表支持勾选当前页后批量归档。页面逐项执行同一受保护接口，并列出成功和失败结果；成功项取消勾选，失败项保留以供重试。切换筛选或账号后清空选择，会话失效或权限不足时停止余下操作。素材来源和审阅信息可展开查看。

素材状态包括「待筛选」「可用」「排除」。修改状态不会启动训练，也不会自动把素材写入任何训练集、验证集或测试集。任务永久删除、回收站到期不会删除已归档素材；尚未归档且已永久删除的 STL 无法通过此功能找回。

单个归档输入最大 128 MiB；每份素材的页面和导出包最多列出 100 条来源，并提供 `sources_total`，完整来源仍保存在数据库中。每个工单最多保留 100 条回复，达到上限后可继续改变状态或新建后续工单。

每份下载包含原始 `input.stl` 和 `manifest.json`，下载前校验 SHA256。清单去掉用户名、病例编号与原文件名；原始 STL 字节保持不变，文件头或几何本身仍可能含需人工核对的信息，不能把下载包视为已完成医学去标识审查。用于优化模型前仍需确认数据使用范围、单位、质量与训练/验证划分。

## 持久化与备份

数据都在当前服务的 `<jobs_root>` 下：

```text
.operations/
  operations.sqlite3       # 工单、审计、素材元数据及来源
  operations.sqlite3-wal   # SQLite 运行时写前日志（可能存在）
  operations.sqlite3-shm   # SQLite 运行时共享索引（可能存在）
  stl/<sha256>             # 显式归档的原始 STL
```

目录仅服务系统账号可访问。将整个 `.operations` 纳入备份，连同既有 `users.json`、任务目录和回收站一起保管。停服务后复制整个目录最简单；在线备份数据库应使用 SQLite backup API，不能仅复制活跃的主数据库文件而遗漏 WAL。素材文件独立留存，会持续占用磁盘，运维健康区显示服务所在磁盘的剩余空间。

启用后，服务把任务状态及 HTTP 登录/删除等事件持久写入独立审计库，任务删除后仍可追踪。旧任务事件、回收站侧车及 `deleted_jobs.jsonl` 提供历史补充；升级前已经丢失的日志和文件不会被重建，历史事件可能没有实际操作者信息。审计页面每次筛选最多展示最近 10,000 条匹配事件，达到上限时应缩小账号、动作或关键词范围；数据库记录不会随这个显示窗口删除。旧清除日志补充读取文件尾部最多 2 MiB。该审计库面向运维追溯，不是防篡改外部审计系统。

## 接口

所有写接口使用现有同源检查与 `X-CSRF-Token`；分页参数为 `page`、`page_size`（最多 100）。返回列表统一为 `items/total/page/page_size`。

| 接口 | 用途 |
|---|---|
| `GET /api/ops/overview` | 全局计数、服务健康、近期事件 |
| `GET /api/ops/jobs` | `owner/status/q/location=active\|trash` 筛选任务；回收站支持 `expires_within_days=7` |
| `GET /api/ops/users` | 用户公开信息与任务统计，支持 `q/disabled=true\|false`，无密码或会话令牌 |
| `POST /api/ops/users/<username>` | `{disabled, reason}` 启停账号 |
| `GET /api/ops/events` | `owner/action/q/category/severity/since/until` 筛选审计；类别为 `job/auth/ticket/asset/user`，级别为 `info/warning/error` |
| `GET /api/ops/events/export` | 同一组筛选条件的 JSON 下载，最多 10,000 条匹配记录，附筛选条件、数量和截断标记 |
| `GET /api/ops/tickets` | `owner/status/q` 筛选工单；`status=unresolved` 包含待处理与处理中，`unassigned=true` 筛选尚未分配的未解决工单 |
| `POST /api/ops/tickets/<id>` | `{version,status,priority,assignee,message}` 处理工单 |
| `GET /api/ops/assets` | `status/q` 筛选素材 |
| `POST /api/ops/assets` | `{job_id,location,note}` 归档原始 STL |
| `POST /api/ops/assets/<sha256>` | `{version,review_status,note}` 审阅素材 |
| `GET /api/ops/assets/<sha256>/download` | 下载 STL 与来源清单 ZIP |
| `GET/POST /api/support/tickets` | 查询自己的工单 / `{title,description,job_id,priority}` 新建工单 |
| `GET /api/support/tickets/<id>` | 查询自己的工单详情 |
| `POST /api/support/tickets/<id>` | `{version,message}` 回复自己的工单 |

## 验证

```bash
PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m pytest -q \
  tests/test_operations_store.py tests/test_operations_http.py tests/test_operations_cli.py tests/test_operations_acceptance.py tests/test_operations_feed.py \
  tests/test_users.py tests/test_trash.py tests/test_delete_jobs.py \
  tests/test_server_security.py tests/test_server_hardening_v015.py tests/test_server_p2.py
node --check wss_deploy/static/ops.js
node --check wss_deploy/static/support.js
```

HTTP 测试使用临时目录与模拟计算阶段，不访问正式任务，不启动 GPU 推理或训练。

2026-09-29 本轮验证：存储、HTTP、CLI、事件筛选与导出、验收目录隔离及既有账号/回收站/删除/服务安全/任务 API 回归共 **101 项通过**。Firefox 共 **33 项流程检查通过**，涵盖主流程、冲突与草稿保护、事件流的逐级展开/暂停/历史阅读保护、筛选导出、批量归档部分失败及账号切换，并检查六个视图的 320/390 px 布局。入口、截图和复测命令见 [验收说明](OPERATIONS_ACCEPTANCE.md)。正式服务加载需另按上线手册升级。
