# wss_deploy — 从壁面 STL 到峰值 WSS 的部署/演示包（v0.2，2026-09-18）

只做推理。链路：`ingest`（单位/拓扑检查）→ `centerline`（vessel_geom/VMTK 子进程 + 出口自动命名 + 人工确认）→ `geometry`（1 mm Taubin 平滑 → 0.5 mm 重采样 → 27 维特征）→ `infer`（发布包 X5D_v51 五 seed，Pa 均值集成）→ `metrics` + `report`（summary.json / report.html / wall_wss.vtp / points_wss.csv / field.npz）。

```bash
export PYTHONPATH=/public/newhome/cy/Digital_twin/GNN
PY=~/.conda/envs/GNN/bin/python
# 单例：先检查并提取中心线；命令行必须显式确认出口映射
CUDA_VISIBLE_DEVICES=1 $PY -m wss_deploy.cli run case.stl --out outputs/wss_deploy_jobs/case --case-id CASE --stage-a-only --units mm
# 只做 A 段（中心线 + 命名建议，CPU），看 proposal 再决定
$PY -m wss_deploy.cli run case.stl --out jobs/case --stage-a-only --units mm
# 指定出口命名（叶段 id=名字）
$PY -m wss_deploy.cli run case.stl --out jobs/case --outlets "3=out-li,4=out-le,5=out-ri,6=out-re"
# 演示服务（默认仅本机；远程使用 SSH 隧道）
CUDA_VISIBLE_DEVICES=1 nohup $PY -m wss_deploy.cli serve --host 127.0.0.1 --port 8765 > outputs/wss_deploy_jobs/server.log 2>&1 &
# 局域网共享（必须令牌；推荐放在反向代理/TLS 后）
export WSS_DEPLOY_TOKEN='change-me'; CUDA_VISIBLE_DEVICES=1 nohup $PY -m wss_deploy.cli serve --host 0.0.0.0 --port 8765 --token "$WSS_DEPLOY_TOKEN" > outputs/wss_deploy_jobs/server.log 2>&1 &
```

浏览器打开 `http://master:8765/`（校园网内直接访问 master 的地址，或本机 `ssh -L 8765:localhost:8765 <user>@master` 后打开 `http://localhost:8765/`）。

| 模块 | 作用 | 备注 |
|---|---|---|
| `paths.py` | 发布包、vessel_geom、VMTK 解释器路径（可用环境变量 `WSS_DEPLOY_RELEASE / WSS_DEPLOY_VESSEL_GEOM / WSS_DEPLOY_VMTK_PYTHON` 覆盖） | 权重只从 `outputs/wss_deploy_release/<release>/` 读 |
| `ingest.py` | 显式单位确认、文件/面片上限、连通片、流形和开口计数、写干净的二进制 STL | 单位不再静默猜测；开口 ≠ 5、非流形、异常尺寸 → 阻断 |
| `centerline.py` | vessel_geom `--preset frozen-aortoiliac` 子进程（GNN_vmtk 环境）；`propose_outlets` 自动命名（左右按 x，髂内外四项加权，170 例验证 166/169 与 325/340）；`validate_mapping` / `apply_mapping` 把确认后的命名写回 atlas | `--inlet` 可改入口 |
| `geometry.py` | `deployment_stl_simulation.build_deployment_case` 的无真值版；capfit 规则显式指向发布包内 v5.1 文件 | 不读 bundle / case.h5 |
| `infer.py` | `Release`：加载五 seed，`predict(case)` → Pa 均值 + 逐 seed | 设备 auto/cuda/cpu |
| `metrics.py` | 峰值（p99 主、最大值参考、位置、热点簇）、低/高 WSS 面积、分支表、几何表 | 阈值 0.4 / 4 / 7 Pa |
| `report.py` | WSS 插值到 STL 顶点（高斯 σ 0.5 mm）、VTP、单文件 HTML（three.js 内嵌，离线可开） | 统计来自预测点云；未覆盖顶点不静默外推 |
| `pipeline.py` | `stage_a` / `stage_b` / `run_all`，每段计时写入 summary | |
| `server.py` / `jobs.py` | 本地优先 HTTP 服务：上传 → 输入确认 → 三维出口确认 → B 段 → 报告；状态机、事件、取消、重试和重启恢复 | 默认回环；共享需 token；任务落盘 `outputs/wss_deploy_jobs/<job>/job.json` |

验收与计时：`training_wss_min/experiments/wss_deploy_timing_20260917/`（分段计时、指标演示、`acceptance_test34/` 34 例回归）。设计与讨论：`docs/02-推进与变更/WSS_PINN/WSS_部署演示工具_从STL到峰值WSS_整体框架与计时_2026-09-17.md`。

## 使用口径

- STL 不携带单位。服务会先展示原始尺寸和换算尺寸；只有明确确认单位后才提取中心线。
- 预测对象是固定收缩期帧（step 1162，约 0.21 s）的壁面 WSS。报告主指标是预测点云的空间 p99；黄色标记是全场最大值位置，二者不是同一个统计量。
- 五个 seed 在 Pa 空间取均值。报告不默认展示 seed 离散度，但发布包哈希、输入 SHA256、采样种子、参数和时间账单会写入 `summary.json`。
- 三维报告中的壁面色彩和鼠标读值是 Gaussian 插值；覆盖范围外显示为灰色/无有效值。面积为点数占比乘输入壁面面积，标记为估计值。

## 服务 API

页面使用 `/api/session`、`/api/jobs`、`/api/jobs/<id>/input`、`confirm`、`cancel`、`retry`。任务状态包括 `queued`、`running`、`awaiting_input`、`awaiting_confirmation`、`done`、`failed`、`cancelled` 和 `interrupted`。所有变更请求带会话 CSRF token 和任务 `version`，重复确认会返回 409 并要求刷新。

服务会限制 STL 为 128 MiB、2,000,000 面片；只开放已完成任务的报告和导出白名单。任务重启后会标记中断，已有输入检查和中心线可以通过“重试”复用。
