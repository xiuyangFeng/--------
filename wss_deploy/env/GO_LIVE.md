# WSS 部署服务上线手册

只列命令与判定标准。所有命令在项目根目录执行：

```bash
cd /public/newhome/cy/Digital_twin/GNN
PY=~/.conda/envs/GNN/bin/python; export PYTHONPATH=.
```

正式任务目录 `outputs/wss_deploy_jobs/`（下文省略 `--jobs-root`）。服务固定 GPU 1，端口 8765。

## 1. 上线前检查

| 项 | 命令 | 通过标准 |
|---|---|---|
| 预检 | `$PY -m wss_deploy.cli doctor` | 末行 `✗ 0`。⚠ 只允许「代码版本（工作树有未提交改动）」；ℹ 逐条读过 |
| 按将要使用的绑定预检 | `$PY -m wss_deploy.cli doctor --host 0.0.0.0` | 「共享绑定与登录」为 ✓（启用的管理员 ≥ 1） |
| 新代码演练 | `$PY -m wss_deploy.cli service rehearse` | 末行「演练通过」，退出码 0（约 5 s；复制 2 个已完成任务到临时目录，回环随机端口 + CPU，结束即删除） |
| 磁盘 | `$PY -m wss_deploy.cli jobs du --top 10`；`df -h /public` | 任务目录所在盘剩余 ≥ 20 GB（< 5 GB 为硬错，会阻止升级；阈值 `WSS_DEPLOY_MIN_FREE_GB`） |
| 账号 | `$PY -m wss_deploy.cli user list` | 至少一行 `"role": "admin"` 且 `"disabled": false` |
| 凭据文件权限 | doctor「凭据文件权限」 | ✓（否则执行 doctor 给出的 `chmod 600 …`） |
| 时区 | doctor「时区」 | `Asia/Shanghai（UTC+08:00）`；来源不是 service.json 时，上线命令里带 `--env TZ=Asia/Shanghai` |
| GPU | `nvidia-smi -i 1 --query-gpu=memory.used,memory.total --format=csv` | 已用 < 20 GB（占满时推理自动回退 CPU，不阻止上线） |
| 回滚目标 | `git status --short wss_deploy \| head`；`git log --oneline -3` | 当前运行版本已提交（见第 6 节）；否则先提交或导出补丁 |

## 2. 上线

```bash
$PY -m wss_deploy.cli service upgrade --drain 600 --env CUDA_VISIBLE_DEVICES=1 --env TZ=Asia/Shanghai
```

判定：依次出现「升级预检通过」→「排空完成」→「服务已就绪：http://0.0.0.0:8765/」→「报告模板：刷新 N 个」，退出码 0。

- 预检 ✗：旧服务未停、未改任何东西；按 ✗ 行修复后重跑。
- 「新服务未能就绪」：旧服务已停。看输出里的日志尾 30 行，修复后 `$PY -m wss_deploy.cli service start`（维护请求保留，启动后执行）。
- 退出码 1 且就绪：有任务重建或报告刷新失败，见输出的 ✗ 行与 `maintenance_result.json`；服务照常可用。

## 3. 上线后验证

| 项 | 命令 | 标准 |
|---|---|---|
| health | `curl -s --noproxy '*' http://127.0.0.1:8765/api/health` | `{"ok": true, "version": "0.15.0"}`（只有这两个字段） |
| ready | `curl -s --noproxy '*' -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8765/api/ready` | `200` |
| status | `$PY -m wss_deploy.cli service status` | 「运行中（由 service 管理）」；登录 = 用户名登录；写锁 = 服务持有；GPU = `CUDA_VISIBLE_DEVICES='1'`；时区 = Asia/Shanghai（service.json）；最近升级 = 刚才的时间 |
| 日志时区 | `$PY -m wss_deploy.cli service logs -n 5` | 时间带 `+0800` |
| 登录 | 浏览器打开 `http://<服务器 IP>:8765/`，admin 登录 | 进入工作台；勾「全部用户」能看到全部任务 |
| 一例烟测 | 打开任一已完成任务的三维报告与一页纸 | 三维模型与色标正常显示 |
| 可选：一例计算 | 网页上传一个已知 STL（或 `$PY -m wss_deploy.cli submit x.stl --units mm --user admin`） | 完成，GPU 端到端约 1 min |

## 4. 日常

| 做什么 | 命令 |
|---|---|
| 看状态（含队列、磁盘、最近升级、维护摘要） | `$PY -m wss_deploy.cli service status`（机读加 `--json`） |
| 看日志 | `$PY -m wss_deploy.cli service logs -n 200`；跟随 `service logs -f` |
| 查占用 | `$PY -m wss_deploy.cli jobs du --top 10` |
| 清可重算缓存 | `$PY -m wss_deploy.cli jobs prune-cache --older-than 30`（只列清单）→ 确认后同一命令加 `--yes`；服务运行时还需 `--force`（排队 / 计算中 / 待确认的任务自动跳过；只删 `geometry_cache/` 与 `.tmp/`，记录写 `prune_cache.jsonl`） |
| 回收站 | 自动：删除 30 天后清除，每次清除写 `deleted_jobs.jsonl` 一行 + server.log「Purged job」 |
| 预检 | `$PY -m wss_deploy.cli doctor`（随时可跑，不写任何文件） |
| 改了代码 | `service rehearse` → `service upgrade --drain 600`（不必重复 `--env`，已存在 service.json） |

## 5. 故障处置

| 现象 | 判定 | 处置 |
|---|---|---|
| 服务进程消失 | `service status` 显示「service.pid 存在但进程 … 已不在」并列出日志尾 30 行 | 从日志尾找 `Traceback` / `diagnostic_id` → `$PY -m wss_deploy.cli service start`（参数取自 service.json；排队任务自动续排，计算中的标「中断」，网页点「重试」） |
| `/api/ready` 返回 503 | `service status` 显示「就绪检查 /api/ready 返回 503」 | `$PY -m wss_deploy.cli service logs -n 300 \| grep -E 'CRITICAL\|died\|diagnostic_id'`；工作线程死亡（`Worker thread … died`）→ `service restart --drain 600` |
| 用户报「诊断编号 xxx」 | — | `grep xxx outputs/wss_deploy_jobs/server.log` |
| GPU 被占满 | 任务「查看计算过程」出现显存回退（device_fallback）；`nvidia-smi -i 1` | 无需处置：该任务自动在 CPU 上重算一次（单例约 1–2 min） |
| 磁盘满 | doctor「磁盘空间」✗（< 5 GB）或 ready 503（disk） | `jobs du --top 20` → `jobs prune-cache --older-than 7 --yes --force` → 网页删除旧任务 / 回收站永久删除 |
| 忘记口令 | — | `$PY -m wss_deploy.cli user passwd admin`（终端输入；非交互：`WSS_DEPLOY_PASSWORD='…' $PY -m wss_deploy.cli user passwd admin`）。该用户已登录会话全部失效，重新登录 |
| 没有可用管理员 | doctor「共享绑定与登录」✗ | `$PY -m wss_deploy.cli user add <名字> --admin` 或 `user role <名字> --admin` |
| 端口被占 | `service start` 报「端口 8765 已被其他程序占用」 | `ss -ltnp \| grep 8765` 找占用进程 |
| 写锁被占 | 「写锁被进程 … 持有」 | 等该维护命令结束；确认已无用后结束该 PID |
| 时间戳时区不对 | doctor「时区」不是 Asia/Shanghai | `service upgrade --env TZ=Asia/Shanghai`（此后服务与离线命令都按它；shell 的 `TZ` 不参与） |

## 6. 回滚

回滚目标必须是一个提交。**当前 HEAD（e8d3b69）是 v0.11.1，v0.14 的改动还没有提交**：直接回到 HEAD 会退回 v0.11。上线前先固定当前运行版本：

```bash
git add wss_deploy tests && git commit -m "wss_deploy 上线版 $(date +%F)"   # 或只导出补丁：
git diff HEAD -- wss_deploy tests > ~/wss_deploy_$(date +%F).patch
```

回滚：

```bash
git log --oneline -5 -- wss_deploy                        # 找到上一个好的提交 <commit>
git stash push -m "rollback-$(date +%F)" -- wss_deploy    # 暂存当前改动（可恢复）
git checkout <commit> -- wss_deploy                       # 需要回到更早的提交时
$PY -m wss_deploy.cli service rehearse
$PY -m wss_deploy.cli service upgrade --drain 600 --env CUDA_VISIBLE_DEVICES=1 --env TZ=Asia/Shanghai
```

判定同第 3 节。恢复新版：`git checkout HEAD -- wss_deploy && git stash pop` → rehearse → upgrade。

兼容性：

- `job.json`：本版仍是 `wss-deploy.job/v2`，回到 0.14 完全兼容；回到 0.13 会丢失 v2 任务在「待确认出口」阶段的预览（已完成任务不受影响），令牌会话需重新登录。
- `summary.json` / `stage_a.json` 的 `created_at` 本版起带时区偏移（原为无偏移的本地时间）；0.14 的报告页两种写法都能显示。
- `service.json` 的 `env.TZ`、`maintenance_result.json` 新增的 `requested_at / requested_by / version`、新文件 `prune_cache.jsonl`：旧代码忽略，无需清理。
