# wss_deploy — 从 STL 到 WSS / TAWSS / OSI、压力与速度体场（v0.11，2026-09-22）

只做推理。链路：`ingest`（单位/拓扑检查）→ `centerline`（vessel_geom/VMTK 子进程 + 出口自动命名 + 人工确认）→ `geometry`（1 mm Taubin 平滑 → 0.5 mm 重采样 → 27 维特征，全部来自冻结特征库 `wss_features/`）→ `infer`（可替换的发布包，按模型族适配器 `families.py` 推理与集成）→ `metrics` + `report`（summary.json / run_manifest.json / report.html / wall_wss.vtp / points_wss.csv / field.npz）。

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
| `geometry.py` | `build_deployment_case` 的无真值版，全部几何调用冻结特征库 `wss_features/`；capfit 规则从发布包内文件显式传入 | 不读 bundle / case.h5，不改任何全局状态 |
| `infer.py` | `Release`：按 `release.json` 的 `models` 清单加载权重（旧 X5D_v51 目录仍兼容），`predict(case)` 交给模型族适配器 | 设备 auto/cuda/cpu；记录并可校验 `feature_contract` |
| `families.py` | 模型族注册表：合同、权重选择、逐模型配置检查、集成预测、B 段流程；现有 `wall_wss_v1`、`pf6_vf6_volume_v1` | 新增模型族只需注册一条，不改 pipeline/registry/infer |
| `regress.py` | 黄金回归：重跑已完成任务的 B 段并与参照逐键逐数组比对 | 参照 `outputs/wss_deploy_golden/20260920_baseline/` |
| `analysis.py` | 沿程曲线 / 发现列表 / 可信区域 / 解剖坐标架（合同 `ANALYSIS_CONTRACT.md` §1–§4） | 两族 B 段与 rebuild 共用 |
| `build_reference_profiles.py` | 生成发布包参考侧车 `reference.json`（几何范围 + CV3 折外 p99 人群） | 不改发布包指纹 |
| `onepager.py` | 一页纸 A4 报告（无三维） | `GET /api/jobs/<id>/onepage` |
| `rebuild_report.py` | 从 field.npz 重建报告与分析产物；`--no-streamlines` 只重建页面 | 参考评估用发布包侧车重算 |
| `registry.py` | 扫描并校验发布包合同、清单哈希和模型文件；为任务提供稳定发布包指纹 | 支持单帧 WSS 和显式 PF6/VF6 压力＋速度合同；多帧合同仍拒绝 |
| `metrics.py` | 峰值（p99 主、最大值参考、位置、热点簇）、低/高 WSS 面积、分支表、几何表 | 阈值 0.4 / 4 / 7 Pa |
| `report.py` | WSS 插值到 STL 顶点（高斯 σ 0.5 mm）、VTP、单文件 HTML（three.js 内嵌，离线可开） | 统计来自预测点云；未覆盖顶点不静默外推 |
| `pipeline.py` | `stage_a` / `stage_b` / `run_all`，每段计时写入 summary；输出通用 `fields`、`time_axis`、`results` 和 `run_manifest.json` | 旧 WSS 字段保留为兼容视图 |
| `schema.py` | 版本化的模型发布元数据、字段描述、时间轴、运行清单和 WSS 兼容层 | 为速度、压力和多时间帧预留接口 |
| `comparison.py` | 按声明合同比较两个完成任务的标量统计 | 不做不同点云的点对点差值；口径不一致时不输出差值 |
| `server.py` / `jobs.py` | 本地优先 HTTP 服务：上传 → 输入确认 → 三维出口确认 → B 段 → 报告；状态机、事件、取消、重试和重启恢复 | 默认回环；共享需 token；任务落盘 `outputs/wss_deploy_jobs/<job>/job.json` |

验收与计时：`training_wss_min/experiments/wss_deploy_timing_20260917/`（分段计时、指标演示、`acceptance_test34/` 34 例回归）。设计与讨论：`docs/02-推进与变更/WSS_PINN/WSS_部署演示工具_从STL到峰值WSS_整体框架与计时_2026-09-17.md`。

## v0.11（2026-09-22 晚）：M1 三头发布包——一次前向出峰值 WSS + TAWSS + OSI

用户裁定部署 M1 三 seed 集成单模型三输出（矩阵 §14.4）。全部为加法：旧发布包路径逐位不变（黄金回归 5/5），359 测试。

| 项 | 做了什么 | 落点 / 接口 | 测试 |
|---|---|---|---|
| 发布包 `M1_3head_3seed_20260922` | `python -m wss_deploy.build_cycle_release`：三 seed × 5 个不可变文件 + rules + test34 读数 + `release.json`（字段 / 周期定义 / 阈值 / 输入合同 / 已知代价）+ `MANIFEST.sha256`，建成即自校验；`/api/releases` 列出，默认仍 `X5D_v51_5seed_20260916` | `outputs/wss_deploy_release/M1_3head_3seed_20260922/` | `test_cycle_release.py` |
| 模型族 `wall_cycle_multi_v1` | 合同 fail-closed（`model_family` M1_3head、字段恰为 wss/tawss/osi、cycle 块帧 0–79 / 0.8 s、显式整数 seed）；逐模型校验 out_dim 3、stats method=multi（log_z / log_z / logit_z，OSI 尺度 0.5）；一次前向 (N,3) → 逐通道回物理量 → 三 seed 均值 → OSI 裁 [0,0.5]；TAWSS/OSI 以 `extra_fields` 交给管线 | `families.py` | 同上 |
| 通用额外壁面标量场 | `cycle_fields.py`（阈值 TAWSS 0.4/4/7 Pa、OSI 0.1/0.2/0.3、滞留区 TAWSS < 0.4 ∧ OSI > 0.1、逐分支统计）；`stage_b_wall`：`summary.fields` 描述符（display 块）、`summary.cycle`、`statistics.tawss/osi/stagnation`、`field.npz` 加 `tawss_pa`/`osi`（+ vertex_/seed_/seed_sd_）、`points_wss.csv` 加列、`wall_wss.vtp` 加点数组；`rebuild_report` 透传 | `pipeline.py`、`rebuild_report.py` | 同上 |
| 报告页字段切换 | 壁面着色 WSS / TAWSS / OSI（TAWSS 对数色标、OSI 线性 0–0.5，阈值线与悬停读数随字段），额外数组 base64 内嵌，页面自包含 | `report.py` | 同上（`node --check`） |

用法：单例 `python -m wss_deploy.cli run <stl> --release-id M1_3head_3seed_20260922 …`；服务切默认 `WSS_DEPLOY_RELEASE=outputs/wss_deploy_release/M1_3head_3seed_20260922 … serve`；网页在发布包下拉里选。
黄金回归请用自包含模式（源任务已被工作台删除）：`WSS_DEPLOY_GOLDEN_REFERENCE=$PWD/outputs/wss_deploy_golden/20260920_baseline WSS_DEPLOY_GOLDEN_JOBS_ROOT=$PWD/outputs/wss_deploy_golden/20260920_baseline PYTHONPATH=. $PY -m pytest -q tests/test_golden_regression.py`。
LV_GUO_YOU 端到端 59.5 s（三模型推理 4.9 s）：峰值通道对 X5D 集成同采样逐点 Pearson 0.994 / R² 0.988；TAWSS 均值 0.665 Pa（< 0.4 Pa 面积 64.8%）、OSI 均值 0.135（> 0.1 面积 56.6%）、滞留区 45.6%。
默认暂留 X5D 的原因：参考侧车 / 人群分位按 X5D 建，先为 M1 重建 profile 再切。未做：M1 参考 profile 与黄金参照、M1p 臂、seed 离散度展示。

**v0.11.1（同日 17:46）修正**：用户在网页点 TAWSS / OSI 页签后壁面不变色——`recolor()` 声明在 `initViewer()` 内部，页签的 `typeof recolor` 守卫在顶层永远看不到它，只换了高亮。现由查看器注册 `recolorActive` 钩子，页签调用钩子重绘（色标、范围描述、悬停读数、探针卡都随字段走）；新增 node 行为测试 `test_report_field_tab_click_repaints_wall`（stub DOM 里点击 TAWSS 页签必须触发一次重绘并解出逐点数组）。已重建该任务报告并重启服务（第十三次）。TAWSS / OSI 都是逐点场（与 WSS 同 76 426 点），不是单个数值。

## v0.10（2026-09-22）：医生常用功能第一、二档——瘤体形态测量、自动结论、自动标注与临床视图

用户选做「瘤体大小与形态」与「风险区一眼看到」两档（患者时间线、上传即全套、掩膜上传、机构模板、分享链接未开）。契约 `ANALYSIS_CONTRACT.md` §17，四路 opus 通用代理并行，合并后 354 项测试、黄金回归 5/5（新键 `morphology / narrative` 不进比对，比对键与 `field.npz` 逐位不变）、6 个历史报告重建、服务重启（第十一次）。

| 项 | 做了什么 | 落点 / 接口 | 测试 |
|---|---|---|---|
| 沿程自动最大直径（`summary.morphology`） | `morphology.py`：中心线每 1 mm 一站，以切线为法向取壁面网格截面（与 JS `crossSection` 同算法的 numpy 版，逐位一致到 6 位），每站记最大 / 最小 Feret 直径、等效直径、面积、闭合标志、位置；主动脉给参考直径（等效直径第 10 百分位）、最大直径站（含世界坐标轮廓环）、瘤体区段（等效直径 ≥ 1.5 × 参考，含最大站的连续段；长度、体积 = 面积沿弧长积分）、近端瘤颈（瘤体前 < 1.2 × 参考的连续段）；全腔体积用开口扇形封盖后散度定理；七条分支的段级表（长度、扭曲度、直径范围、p99 / 低 WSS 占比 或 ΔP / 速度） | 两族 B 段与 `rebuild_report`，`timing_s["morphology"]`（LIU 27 万面 12.5 s、LV 1.7 s）；`findings` 的最大直径项改用壁面截面值 | `test_morphology.py` |
| 截面可靠性（合并时发现并修） | 首版在瘤体肩部与分叉近端斜切：LIU 例主动脉"最大直径 118.6 mm"实为斜切（截面最小宽度 64 mm 对内切直径 17 mm），LV 例左髂总 93.8 mm 是分叉处切进瘤囊的钥匙孔形。现每站三条判据：斜切（最小宽度 > 1.6 × 内切直径）、狭长（最大 / 最小 > 2.2）、低实心度（面积 / 凸包 < 0.8）；可疑站在切线周围 15/30/45° × 8 方位的锥内重切，取**面积最小的闭合截面**（最接近局部垂直面）替换，记 `reoriented / tilt_deg / raw_max_diameter_mm`；最大值 / 瘤体 / 瘤颈 / 参考直径 / 分支直径统计只用 `reliable` 站。修正后 LIU 主动脉 77.8 mm@154 mm（瘤体 112–222 mm、334 mL、瘤颈 107 mm ⌀20.2）、LV 左髂总 42.7 mm；合成弯管 + 椭球瘤测试肩部站重定向后等于真实颈径、瘤体最大值误差 < 5% | 同上 | 同上 |
| 自动结论（`summary.narrative`） | `narrative.py` 按存在的量生成中英各 3–4 句（形态一句；壁面：低 WSS 占比与位置、高 WSS 热点数与最高值、p99 与人群分位；体场：主动脉近远端压差、最大速度与分支、滞留区数），末句固定"参考描述，非诊断结论"；审阅人可改：`PUT /api/jobs/<id>/narrative {text}`（空串恢复自动）写 `narrative.json` 并镜像进 summary 与报告 meta，锁定 → 409；工作台详情页「结论（参考）」卡片可编辑 / 恢复；报告「统计与口径」顶部只读显示；一页纸顶部「结论（参考）」 | `jobs.narrative`、`onepager.py` | `test_narrative.py` |
| 一页纸与汇总表 | 一页纸新增「瘤体形态」（最大直径与位置 / 瘤体长度与体积 / 瘤颈 / 全腔体积 / 参考直径 + 方法一句）与「血管分支表」；汇总表 8 列 `max_diameter_mm, max_diameter_s_mm, sac_present, sac_length_mm, sac_volume_ml, neck_diameter_mm, neck_length_mm, lumen_volume_ml`；术语表 +8 条 | `onepager.py`、`export_table.py`、`glossary.py` | `test_onepager.py`、`test_export_table.py`、`test_glossary.py` |
| 自动标注 | 两报告「显示」菜单：「自动标注发现 关 / 前 3 / 前 5 / 前 10」+「分支名」：三维上为发现列表前 N 条（按严重度与排名，排除驳回项、隐藏分支）钉「F1 高 WSS 21.9 Pa」标签，分支中点钉分支名（英文模式换英文）；点标签飞到该发现；标签随导图 / 拼图 / 一页纸配图合成；视图状态 `labels:{findings, branches, max_diameter}` | `report.py`、`volume_viewer.js` | 两报告测试 |
| 最大直径标记 | 「沿程」菜单「最大直径 xx mm（等效 xx）· 入口下 xx mm · 飞到」；三维画该站截面轮廓环（紫色）+ 标签，可关；沿程曲线加「最大直径 / 等效直径」序列并进曲线 SVG | 同上 | 同上 |
| 临床视图预设 | 共享库内置预设「临床视图」（两族）：前视 + 彩虹色标 + 默认阈值 + 前 5 条发现标注 + 分支名；壁面 WSS 模式，体场速度点云 | `report_common.builtinPresets` | `test_report_common.py` |
| 工作台 | 结果卡与病例卡「最大直径 xx mm」芯片、详情页「瘤体形态」事实块；队列总览加最大直径直方图与 `最大直径 ≥` 筛选 | `app.js`、`workbench_core.js` | `test_workbench_js.py` |

已知边界：分叉近端仍可能有单条判据都不触发的膨大读数（LIU 右髂总 47.7 mm，最大 / 最小 2.09 刚低于 2.2），分支表直径在分叉附近要结合三维轮廓环看；瘤体 / 瘤颈判据（1.5× / 1.2× 参考直径）是常用经验规则，不是指南阈值；结论文字全部标注"参考、非诊断"；报告里可靠性数组已内嵌（`stations.reliable`）但曲线暂未灰显不可靠站。

## v0.9（2026-09-21 晚）：日常功能第一、二档——并排同步口径、一页纸配图、元数据可改、队列总览、拼图与曲线导出、真实管径、取消即时生效

用户在试用 v0.8 后裁定做"第一档 + 第二档"（提速 N2、方向侧车 N3、无头浏览器 N5、随访配准 N11 未开）。契约 `ANALYSIS_CONTRACT.md` §15，共享库先由主会话补齐截面几何 / 拼图 / 曲线 SVG / CSV 函数，再四路并行（opus 通用代理）。合并后全套 315 项测试通过、黄金回归 5/5（预测数值逐位不变）、6 个历史报告重建、服务重启（第十次）。真实浏览器验收清单见各项。

| 项 | 做了什么 | 落点 / 接口 | 测试 |
|---|---|---|---|
| 并排同步口径 | 并排页顶栏「同步显示口径」（默认开）+「以左为准 / 以右为准」：活动侧变化后 400 ms 内向其发 `wss-view:get-state`，把色标 / 分段 / 对数 / 范围 / 单位 / 阈值 / 透明度 / 叠加 / 截面子集用 `apply-state` 推给另一侧，相机不动；壁面 + 体场混合对只同步色标类；回声保护 400 ms | `compare.js`、`workbench_core.js`（`compareDisplaySubset / EchoGuard`）；两报告实现 `get-state` | `test_compare_page.py` 5、`test_workbench_js.py` |
| 一页纸配图 | 报告「视图」菜单「生成一页纸配图」（仅在线）：离屏渲染 前 / 左 / 上 + 当前视图（体场再加当前截面放大图）2× 白底带色标，`POST /api/jobs/<id>/snapshots` 存 `snapshot_<name>.png` + `snapshots.json`（≤ 12 张、每张 ≤ 6 MB、PNG 魔数校验；审阅锁定不阻止）；一页纸「关键数字」后加「配图」区（data URI 内嵌，单文件不变），打包 zip 收录 | `jobs.snapshots`、`onepager.render_onepage(job_dir=)` | `test_snapshots.py`、`test_onepager.py` |
| 元数据可改 | 详情页「编辑信息」、病例卡「编辑」：病例号 / 患者 / 扫描标签 / 日期 / 标签 / 备注；`POST /api/jobs/<id>/metadata` 写任务记录、`summary.case_metadata` 与 `case_id`、报告内嵌 meta；锁定 → 409 | `jobs.update_metadata` | `test_metadata_edit.py` |
| 队列总览 | 侧栏「队列总览」卡片：`GET /api/jobs/export?format=json&scope=done` 取全部已完成任务的固定列 + 发布包人群参照（`reference.json` 的 136 例折外 p99）；筛选族 / 发布包 / 审阅 / p99 / 高、极高面积占比 / 速度 p99；三张直方图（壁面 p99 叠人群参照、高 WSS 面积占比、速度 p99）；可排序结果表点击选任务；「导出筛选结果 CSV / xlsx」 | `export_table.population_blocks`、`workbench_core.js`（`binEdges / histogram / cohortFilter / cohortSort`） | `test_export_table.py`、`test_workbench_js.py` |
| 多选批量重跑 | 多选栏「用发布包重跑所选」：选发布包后逐个调用现有 `rerun`，未完成 / 未确认出口的跳过并给原因，逐行日志与汇总 | `app.js` | 启动桩测试 |
| 取消即时生效（N1） | 中心线子进程改 `Popen(start_new_session)`，每 0.5 s 查取消，取消 / 超时对进程组 SIGTERM → 2 s → SIGKILL；任务新增 `attempt_seconds`（本次尝试，重试从 0 起），`timing.attempt_s`，事件 `attempt_aborted`；工作台显示「本次 · 累计」 | `centerline.run_vessel_geom(cancelled=)`、`jobs.py` | `test_cancel_subprocess.py`（假子进程 0.5 s 内杀死） |
| 真实管径 | 壁面报告「管径」改为：点选点投影到中心线，以切线为法向取壁面网格截面轮廓（`crossSection`），标签「最大 / 最小 / 等效直径 · 面积」，三维画轮廓环；找不到闭合轮廓回退 2×内切半径并注明；体场截面统计与放大图标题也显示轮廓面积 / 最大 / 等效直径 | `report_common.crossSection / sectionMetrics`（§15.0） | `test_report.py`、`test_report_common.py` |
| 截面 CSV | 放大视图「导出 CSV」：轮廓内每格 `x_mm, y_mm, value, units, filled_from_wall`，头部注释病例 / 物理量 / 中心 / 法向 / 厚度 / 网格 / 补全模式 | `volume_viewer.sliceGridToRows`、`tableToCSV` | `test_volume_report.py` |
| 截面系列拼图 | 自动截面下选分支 + 站位数（默认 6：5/20/40/60/80/95%）「导出系列拼图」：每站 700×620 截面图，a–f 编号、`s = xx mm (yy%)` 说明、共用色标一张 PNG；`slice.series` 进视图状态 | `composeMontage` | 同上 |
| 多视角拼图 | 两报告「视图」菜单：勾选 前 / 后 / 左 / 右 / 上 / 下 / 当前（体场 + 截面），2–4 列，沿用导图分辨率 / 背景 / 语言，面板 a/b/c 编号 + 视角名，右侧共用矢量色标；文件名 `<case>_montage_<field>_<k>x.png` | `renderOffscreen + composeMontage` | 两报告测试 |
| 沿程曲线 SVG | 「沿程」菜单「导出曲线 SVG」：当前分支物理量曲线（壁面：均值 / p99 / 最低 + 阈值虚线；体场：均值 / 最大或最低）与半径曲线各一张，坐标轴 / 图例 / 刻度齐全 | `profileSVG` | 同上 |

共享库 §15.0 新函数：`planeFrame / planeContour / contourLoops / selectLoop / closeChain / pointInLoop / loopPolygon / sectionMetrics / crossSection / composeMontage / loadImage / profileSVG / tableToCSV`（node 测试：椭圆管截面面积 / Feret 直径、穿开口链合成封口、SVG、CSV）。顺带修正：`cases.py` 病例卡成员排序不再用随机任务号决胜（同秒创建的任务顺序曾偶发变化）；体场 `sliceGrid` 在切片内 ≤ 1 个采样点时哈希格过小导致页面卡死。

未做 / 已知边界：拼图与配图的真实像素效果、并排同步的手感只能在浏览器验收；`profileSVG` 的阈值用虚线常量序列表示（无水平色带）；截面系列只取当前厚度；一页纸配图需在线打开报告（离线单文件没有上传通道）；N2 / N3 / N5 / N11 未开。

## v0.8（2026-09-21）：常用功能波 C1–C18——病例卡、查重复用、登录、回收站、通知、测量标注、出版级导图、汇总与打包

按[下一轮功能与优化方案](../docs/02-推进与变更/WSS_PINN/WSS_部署工具_下一轮功能与优化方案_2026-09-21.md) §2 的 18 项常用功能开发；四路并行（后端 / 工作台 / 报告共享库 + 壁面报告 / 体场报告），契约在 `ANALYSIS_CONTRACT.md` §10–§14。合并后单元测试 262 通过（+ 黄金回归门 1 项），`node --check` 全过，黄金回归 5/5（预测数值逐位不变），8 个历史报告已重建，服务已重启。真实浏览器渲染（导图分辨率 / 透明底 / 通知弹窗）待用户验收。

| 项 | 做了什么 | 落点 / 接口 | 测试 |
|---|---|---|---|
| C1 病例卡归组 | 列表可切「按任务 / 按病例」；同一输入几何（`input_sha256`）下的全部运行合成一张卡，每个发布包一个芯片（状态 + 审阅徽标），「并排打开 WSS + 体场」、「补跑 <缺少的发布包>」（调 `rerun`，复用中心线与出口）、「导出该病例汇总」 | `cases.py`、`GET /api/cases?q&patient_id&tag&page&page_size[&all=1]` | `test_cases.py` |
| C2 上传查重与几何复用 | 上传时即计算并写入 `job.input_sha256`；表单字段 `on_duplicate=ask|reuse|force`（默认 ask）：同几何已有任务 → 409 `{duplicate, existing[], reusable}`，网页弹出「打开已有 / 只跑新发布包（复用中心线与出口确认）/ 强制重算」；`reuse` 只跑 B 段并记 `reused_from`；批量逐项判定（`duplicate_count`） | `jobs.create/_clone_for_stage_b/find_by_input`、`server._upload` | `test_duplicates.py` |
| C3 用户名登录 | 共享模式存在 `users.json`（jobs 根，0600，scrypt）即启用用户名口令登录，`owner = username`，换浏览器任务不丢；旧令牌在无 `users.json` 或 `WSS_DEPLOY_ALLOW_LEGACY_TOKEN=1` 时仍可用；错误口令每用户 60 s 内 5 次 → 429；「认领本会话任务」把旧会话 owner 的任务改归用户；管理员 `?all=1` 只读全览；本机回环模式仍免登录 | `users.py`、`POST /api/session {username,password}`、`/api/session/logout`、`/api/session/password`、`POST /api/jobs/claim`；CLI `python -m wss_deploy.cli user add|passwd|disable|list <name> [--admin]`（口令从终端读，非交互用 `WSS_DEPLOY_PASSWORD`）、`jobs claim --owner <旧 owner> --user <name>`（停服务后执行） | `test_users.py` |
| C4 回收站 | 删除改为移入 `<jobs>/.trash/<id>/`（`trash.json` 记删除人 / 到期 / 快照），30 天可恢复（id 与状态不变、报告可开），到期由工作线程每日扫描清除；`deleted_jobs.jsonl` 改在彻底清除时写（`purge_reason=manual|expired`）；锁定任务仍不可删 | `GET /api/trash`、`POST /api/trash/<id>/restore|purge`；网页「回收站」面板 | `test_trash.py`、`test_delete_jobs.py` |
| C5 完成通知与未读 | owner 级 SSE `GET /api/events`（`hello` + 每次状态跃迁，含 `case_id/family/review`）；浏览器通知（首次点击后请求权限，点通知跳任务）；未读集合存 localStorage，列表标题徽标与标签页标题 `(N)` | `jobs.subscribe_owner`、`static/app.js` | `test_events_owner.py`、`test_workbench_js.py` |
| C6 记住习惯 | `GET/PUT /api/preferences`（`preferences/<owner>.json`，≤ 16 KB，整体替换、前端分节合并）：上传表单（单位 / 发布包 / 设备 / 模型数 / 线程 /「沿用上次的患者与标签」）与报告默认口径（`report_defaults.wall|volume`，报告页「把当前口径设为默认」；优先级 `#view=` > 服务端 > localStorage > 内置） | `server.PreferenceStore` | `test_preferences.py` |
| C7 测量 | 两报告共用 `static/report_common.js`（中心线树、投影、弧长跨分支经公共祖先、管径 = 2×内切半径）；「测量与探针」菜单：距离 / 弧长 / 管径 / 分段，壁面点选，三维线段 + 端点球 + 标签，列表可删可复制；进视图状态 `measurements` | `report_common.js`、`report.py`、`volume_viewer.js` | `test_report_common.py`、`test_report.py`、`test_volume_report.py` |
| C8 标注 | 壁面钉文字标签（标签 + 引线，点标签可编辑 / 删除）；在线 `PUT /api/jobs/<id>/annotations` 写 `annotations.json` 并镜像进 `summary.json` 与报告内嵌 meta（锁定 → 409），离线用内嵌副本只读；一页纸附「标注」表 | `jobs.annotations`、`rebuild_report.embed_sidecars` | `test_annotations.py` |
| C9 快捷预设 | 内置（壁面：瘤囊低 WSS 区 / 髂分叉热点 / 主动脉沿程；体场：沿程压降 / 流线全貌 / 瘤囊截面系列）按病例分支与发现即时生成；用户预设存偏好 `presets.<family>`（离线 localStorage），可应用 / 重命名 / 导出 / 删除 | `report_common.builtinPresets` | 报告测试 |
| C10 分支显隐 | 显示菜单分支勾选：壁面按顶点分支过滤三角面（三顶点全隐才隐）与点云，体场过滤内部点 / 箭头 / 流线 / 截面；沿程曲线变灰；状态 `branches_hidden`；统计不受影响 | 两报告 | 报告测试 |
| C11 探针记录 | 点选锁定探针 →「记录」攒成表，复制 TSV / 导出 CSV（BOM + CRLF）；状态 `probe_log` | 两报告 | 报告测试 |
| C12 出版级导图 | 离屏渲染 1×/2×/4×，背景白 / 透明 / 当前，色标叠加 / 不含 / 单独 SVG（矢量色标），中 / 英标签（Aorta、Left/Right CIA/EIA/IIA…），测量与标注标签合成进 PNG；文件名 `<case>_<view>_<field>_<k>x.png`；六视角导出走同一路径；显卡上限时降级到 1× 并提示 | `report_common.renderOffscreen/colorbarSVG/englishLabel/exportFilename` | 报告测试 |
| C13 跨病例批量出图 | 工作台多选 →「批量出图」：选视图状态（粘贴报告导出的 JSON 或内置预设名）+ 导出选项 → 隐藏 iframe 逐例加载报告，按 message 协议 `wss-view:ready → apply-state → export → exported` 收集 PNG，store 模式 zip 打包下载（无外部库，CRC 正确，Python `zipfile` 可解）；族不匹配跳过 | `static/batch_export.js`；协议见契约 §12.6 | `test_workbench_js.py` |
| C14 汇总表导出 | `GET /api/jobs/export?ids=a,b&format=csv|xlsx`（≤ 200 个 id，固定列：身份 / 审阅 / 时长 / 壁面 p99·最大·均值·面积占比·七分支 p99·人群分位·质量等级 / 体场速度 p99·最大·压力 min/max·七分支 ΔP·低速区数 / 发现计数；缺失留空不猜）；多选栏与病例卡按钮 | `export_table.py` | `test_export_table.py` |
| C15 打包下载 | `GET /api/jobs/<id>/bundle.zip`（白名单文件 + 按需生成的 `onepage.html` + 标注 / 发现判定侧车 + `README.txt` 口径说明；`.npz` 存储、文本压缩）；`POST /api/jobs/bundle {ids}` 返回 zip 的 zip（≤ 50） | `bundle.py` | `test_bundle.py` |
| C16 发现列表可编辑 | 报告里每条发现「确认 / 驳回 / 未判定」+ 备注，「新增发现（点选壁面）」加人工项；在线 `PUT /api/jobs/<id>/findings_review` 写 `findings_review.json` 并合并进 `summary.findings.review`（锁定 → 409）；一页纸：确认 ☑ / 未判定 ☐ 正文、驳回项附录、人工项标「人工」 | `jobs.findings_review`、`onepager.py` | `test_findings_review.py`、`test_onepager.py` |
| C17 术语说明 | `glossary.py` 单一来源（31 条：p99 / 最大值 / 面积占比 / 面积加权 p99 / 阈值 / 单位 / 相对压 / ΔP / 五种可信位 / 置信度代理 / 人群分位 / 质量等级 / 发布包 / 特征合同 / run_identity / 标准视角 / 局部管径 / 弧长 / 审阅状态 / 发现判定 / 流线 / 截面 / 固定帧…）→ `python -m wss_deploy.glossary` 生成 `static/glossary.json`，两报告内嵌、工作台按需拉取，「?」弹出解释；一页纸末尾附术语表；`test_glossary.py` 校验 JSON 同步且页面里每个 `data-gloss` 键都存在 | `glossary.py` | `test_glossary.py` |
| C18 审阅提醒 | 列表顶部「待审阅 N」分区默认展开、已审阅折叠；状态筛选加「待审阅」；病例卡显示待审阅计数 | `static/app.js` | `test_workbench_js.py` |

工作台脚本改为三份：`app.js`（界面）、`workbench_core.js`（无 DOM 的纯逻辑：未读集合、偏好合并、病例卡模型、批量出图计划）、`batch_export.js`（zip 写入器 + iframe 编排）；`tests/test_workbench_js.py` 用桩 DOM + 罐头服务把整个 `app.js` 启动一遍（零异常、所有 `getElementById` 目标都在 `index.html`、七个接口都被调用）。视图状态 v1.1 新增可选键 `measurements / annotations / probe_log / branches_hidden / preset_name / lang / export`，旧状态照常应用。

启用用户名登录（共享模式）：

```bash
PY=~/.conda/envs/GNN/bin/python; export PYTHONPATH=/public/newhome/cy/Digital_twin/GNN
$PY -m wss_deploy.cli user add reviewer01 --admin --display-name "审阅人一"   # 交互输入口令；非交互：WSS_DEPLOY_PASSWORD=… 
$PY -m wss_deploy.cli user list
# 旧会话任务归属到用户（停服务后；旧 owner 见网页「认领」横幅或 .sessions.json）
$PY -m wss_deploy.cli jobs claim --owner <旧 owner> --user reviewer01
```

存在 `users.json` 后 `GET /api/session` 返回 `login: "password"`，网页登录框改为用户名 + 口令；旧令牌只在设置 `WSS_DEPLOY_ALLOW_LEGACY_TOKEN=1` 时继续接受。当前 master:8765 未建用户，仍是令牌模式。

**v0.8.1 追加（2026-09-21，用户试用反馈）**：体场报告截面改为三维里直接操作——蓝色截面缩小为局部大小（8×局部半径）并带轮廓与法向箭头；悬停变色，拖动它沿法向上下移动（默认）、Shift 拖动或工具条切「旋转」后拖动旋转、Alt 拖动 / 「面内平移」平移；滚轮悬停在截面上时沿法向移动（Shift 滚轮改厚度，Ctrl 5 mm 一档）、空白处滚轮仍缩放；↑↓ ←→ PgUp/PgDn `[` `]` 键盘微调；空白处拖动恢复为旋转视图（此前截面模式下整个视图不能旋转）。点选改名「点选定位截面」，视图顶部有工具条与红色分步提示（第 1 点 / 第 2 点 / Esc），两点选完自动结束点选；菜单里写明三种放置方式（拖动 / 点选 / 滑杆）。顺带修正点选基准下"沿法向微调"读数（原显示值是实际位移的 2 倍）。核心换算 `dragAlong / positionStep` 有 node 测试；真实拖动手感待用户在浏览器里确认。

**v0.8.2 追加（2026-09-21，用户试用反馈 2）**：截面平面视图默认「补全截面」——瘤囊里预测点稀，原来只画有邻点支撑的格子、其余留灰，截面中间成片空洞。现在用截面与壁面网格的交线（marching triangles，`planeContour`）当管腔边界，扫描线奇偶判定格子在腔内（`scanlineInside`），再把壁面当边界条件——速度按无滑移取 0、压力取插值到交线上的壁面压力——与厚度内预测点一起做 IDW 填满整个腔内（`fillSection`，网格哈希近邻）；离预测点超过 3.2 倍中位间距、主要靠壁面边界补出来的格子画淡色，底部标注"邻点直接支撑 X%"；平面图上叠画深色管腔轮廓。面板头部勾选框可关掉，回到严格模式；`slice.fill` 进视图状态。node 测试：直管切面轮廓半径 10 ± 0.2、腔内格数 ≈ π r²、中心速度 ≈ 1 而近壁 < 0.2。这是显示层插值，不改 summary 里任何统计。

**v0.8.3 追加（2026-09-21，用户试用反馈 3）**：(1) 补全只取"本截面"的管腔环——交线段按共享网格边串成环（`contourLoops`），选包含截面原点的环（嵌套取最内），没有包含的取最近的（`selectLoop`）；截到邻近血管的环和它的预测点不再画进来，平面图按选中环的包围盒自动放大到截面本身。(2) 「放大」按钮打开大图视图（1400×1200 画布、240 格插值、白底、大字号、比例尺、色标带单位标题），可隐藏预测点、悬停读格值、一键导出 PNG，Esc 关闭；侧栏小图同样加了比例尺，补全统计移到画布下方的文字行不再压色标。渲染拆成 `sliceMapData` → `sliceGrid` → `renderSliceMap`，侧栏与大图共用。

**v0.8.4 追加（2026-09-21，用户试用反馈 4：小血管末端只剩点）**：根因是选环逻辑两处漏洞——交线段先按半径截断再串环，把远处闭环切成开口弧；"是否包住原点"用射线奇偶判定，开口弧被射线穿过一次就误判为包住，包围盒又更小，赢过本地闭环。修正：不再截断，先串环再按"到原点距离 ≤ max(4×局部半径, 8 mm)"筛选非包围候选；只有闭环才用奇偶判定，开链改用"绕原点的角度覆盖 > 180°"判定；截面穿过血管开口时本地轮廓是开链，两端缺口 ≤ 0.6×链长就用一条无壁面值的合成边封口（画成虚线、不做边界条件），否则退回严格模式并注明；越过血管末端找不到轮廓时注明"截面可能已越过血管末端"。真实网格 LV_GUO_YOU 七条分支 × 六站 × 倾斜/偏移 504 个平面：440 填满、24 严格（陡角穿开口）、40 无轮廓（越过末端）、0 次选错远环。

**v0.8.5 追加（2026-09-21，用户试用反馈 5：并排比较被面板遮挡）**：两个报告都加了「紧凑布局」——窗口（或并排页的半宽 iframe）宽度 < 1180 px 时自动启用，头部「紧凑布局 / 完整布局」按钮可强制（记在 localStorage `wss-report-compact`）。紧凑时：左侧菜单收成覆盖层（头部「☰ 菜单」或视图左缘「菜单」竖标打开，点视图即收起，不再占 300 px）；头部一行（标题小字、按钮短标签「点选 / 复位 / 保存 / 打印」）；页脚隐藏；体场报告右侧停靠区改为 200 px 小色标 + 底部一条探针横排（原来 330 px 宽的探针卡挡住血管），截面面板缩到 230 px 且默认只留标题行（「放大」仍在）、提示文字移到左上角、截面工具条并入右上；壁面报告色标缩短、探针卡改为底部横条。并排页：顶栏 48 px，左右身份信息并入各自窗格标题条（病例 · 患者 · 扫描 · 发布包 · 审阅），「标量比较」折叠条只占一行，页面总高固定为视口，报告区拿到剩余全部高度。探针/统计的键值对改为 `.kv` 包裹（普通布局用 `display:contents` 保持原网格）。

未做 / 已知边界：真实浏览器里的 2×/4× 分辨率、透明底、标签合成与通知弹窗只能由用户验收（无头浏览器未装）；壁面标注引线是固定 +z 偏移不朝向相机；体场「瘤囊截面系列」预设停在 40% 站并提示其余两站；报告侧栏文字保持中文（`lang` 只切换导出图的图例与三维标签）；标注 / 发现判定 PUT 不带 `version`，并发编辑后写者胜；迭代二（N1 取消即时生效、N3 方向侧车、N4 命名飞轮、N5 无头验证）与提速 N2 未开。

## v0.7（2026-09-21）：从"看图"到"阅片"——分析层、发现列表、审阅签字、并排比较

四路并行开发（契约 `ANALYSIS_CONTRACT.md`），合并后全套单元测试 185 通过、黄金回归 5/5（参照目录已自包含），8 个历史报告已重建，服务已重启。

| 层 | 内容 | 落点 |
|---|---|---|
| 分析层 | `analysis.py`：沿程曲线 `summary.profiles`（每 2 mm 分箱：半径、WSS 均值/p99/最低 或 速度均值/最大/压力均值/最低）、发现列表 `summary.findings`（高/低 WSS 簇、最大值、最大直径、狭窄最小半径；体场：最大速度、最低压、各分支 ΔP、低速区）、可信区域 `summary.trust`（位掩码：插值无支撑 / 粗糙 / 几何越界 / 采样支撑弱 / 近切口）、壁面族 `frame_transform` 补齐解剖坐标架；两族 B 段与 `rebuild_report` 共用；`field.npz` 不变 | `tests/test_analysis.py` |
| 参考侧车 | `build_reference_profiles.py` 生成 `<release>/reference.json`：几何范围（train136 atlas 逐分支半径/长度 min×0.9–max×1.1，29 条）+ 人群参照（train136 CV3 折外 p99，136 例，与 CFD Spearman 0.94）；侧车不进发布包指纹，`Release` 加载时按 release id 绑定并记录哈希；PF6/VF6 包只有几何范围。`reference.py` 的协议比较忽略逐例变化的连通半径键 | `tests/test_reference_profiles.py` |
| 壁面报告 | 折叠菜单（显示 / 发现 / 沿程 / 区域 / 视图 / 统计）；发现列表点击飞到并高亮；沿程曲线点击高亮 s 带；标准视角前/后/左/右/上/下、视图状态（localStorage / JSON / `#view=` 链接 / 复现此图 / 六视角导出）；可信区域叠加与图例；离散色带 0–12 段、对数色标、0.4/4/7 Pa 等值线；单位 Pa / dyn·cm⁻²、阈值可改即时重算面积；区域统计（分支 s 范围 / 球形刷）；相机联动协议；页脚显示审阅与哈希 | `tests/test_report.py`（18 项，含 node 核心测试与 DOM 桩） |
| 体场报告 | 探针（悬停/点击：速度分量、压力、分支、s、半径、到壁距离）；发现列表；沿程曲线点击 → 截面联动；标准视角与视图状态；可信叠加；离散色带、Pa/mmHg、m/s / cm/s；区域统计与近远端压差；相机联动；页脚 | `tests/test_volume_report.py`（25 项） |
| 工作台 | 审阅签字 `POST /api/jobs/<id>/review`（approve 锁定：retry / 出口改算 / 删除 → 409，需 reopen；rerun 仍允许并从未审阅开始）；一页纸 `GET /api/jobs/<id>/onepage`（按需生成，A4 打印）；并排比较 `GET /compare?left&right`（两 iframe 相机联动 + 标量表）；报告与一页纸响应 `frame-ancestors 'self'` | `tests/test_review_lock.py`、`test_onepager.py`、`test_compare_page.py` |

未做（依赖模型或需串行）：TAWSS/OSI/向量 WSS/流量/绝对压；患者方向 sidecar（跨 ingest/centerline/server，下一轮串行做）；等值线只在壁面报告实现。

## v0.6（2026-09-20）：冻结特征库、模型族适配器、实时事件流

按[架构评审](../docs/02-推进与变更/WSS_PINN/WSS_部署工具_架构评审与优化路线_2026-09-20.md)的 R0/R1 落地，数值验收以 `python -m wss_deploy.regress` 为准（5 个已完成任务重跑 B 段与 `outputs/wss_deploy_golden/20260920_baseline/` 逐键逐元素一致）。

| 项 | 做了什么 | 验收 |
|---|---|---|
| 冻结特征库 `wss_features/` | STL→特征的全部几何程序（STL 读写、atlas/RMF、法向/虚拟盖/内判、曲率、流量参考特征、平滑/重采样）从 `wss_v5` / `training_wss_min.tools` 逐字拷贝为无训练依赖、无全局状态的包；capfit 规则改为显式参数，`WF._CAPFIT_RULE_OVERRIDE` 全局改写已删除 | `tests/test_wss_features_equivalence.py` 在真实病例上与训练实现逐位一致（5 项）；`CONTRACT.json` 记录源码哈希，`tests/test_wss_features_contract.py` 防止无意改动；`python -m wss_features record` 有意 bump |
| 特征合同进来源链 | `summary.json` / `run_manifest.provenance.feature_contract` / `model_release.feature_contract` 记录特征程序版本与哈希；发布包可在 `release.json` 声明 `feature_contract.source_hash`，不一致时拒绝加载 | 现有发布包未声明（只记录）；新发布包应声明 |
| 模型族适配器 `families.py` | 合同校验、权重选择、逐模型配置检查、集成预测、B 段流程按族注册（`wall_wss_v1`、`pf6_vf6_volume_v1`）；`pipeline.stage_b` / `infer.Release.predict` / `registry._contract` 只做分发，不再有 `if target == "pressure_velocity"` | 注册表/体场/schema 测试不变通过；黄金回归 5/5 |
| 体场内判提速 | `winding_number` 去掉长度为 3 的 `einsum`（逐位相同，快 3 倍）；VTK 批量内判实测与逐点等价但不更快，保持原样 | 体场 `volume_features` 34 s → 21 s，单例总时长 68 s → 56 s；`internal_pts` 逐元素相等 |
| 任务记录可搬迁 | `stage_a.json` / `input_check` 内的文件改为相对任务目录（旧记录的绝对路径仍可读，`io_utils.resolve_job_path`）；`rerun` 不再做字符串替换 | 旧任务重跑通过 |
| 轻量状态接口 | `GET /api/jobs/<id>` 不再携带预览网格与中心线折线（每次轮询少约 1–2 MB，深拷贝不再占锁）；新增 `GET /api/jobs/<id>/geometry`（带 ETag / 304）供确认页取一次 | `tests/test_geometry_and_events.py`、`tests/test_server_geometry_events.py` |
| 实时事件流 | `GET /api/jobs/<id>/events`（SSE）：先发快照，再逐事件推送，任务终态后关闭；网页用 EventSource，流不可用时才回退 2 s 轮询；任务列表刷新改为 10 s | 同上 HTTP 测试 |
| 批量上传 | 网页多选文件时走 `/api/jobs/batch`（每批 ≤ 20，超过自动分批） | 手工 |
| 体场报告改版 | 菜单式界面（顶部四个显示页签：体内点云 / 截面 / 壁面压力 / 流线；左侧「显示 / 截面 / 统计与口径」三个折叠菜单，一次只展开一个；截面投影改为视图右下角浮动面板，仅截面模式显示）；色标可选 彩虹（默认）/ Turbo / 蓝白红，按浏览器记忆；流线改为带光照的管状线，可调粗细、密度，弱显卡可切细线；**点选截面**：点「开始点选」后在壁面上点 1 个点 → 垂直局部中心线的横截面，点第 2 个点 → 通过两点的（斜）截面，仍可用手动滑杆微调 | `tests/test_volume_report.py`（+2 项） |
| 流线密度与覆盖 | 种子改为每 12 mm 一站、每站 1+6+10 个（上限 420）+ 120 个体内最远点种子；覆盖判据改为局部 k-NN 半径（瘤囊内采样稀，全局 2.8 mm 截断曾让瘤囊里的线几乎立即终止）；顶点按需抽稀保持报告 ≤ 约 10 MB | LIU_YU_MING：147 条 → 521 条，瘤囊内 7.4 万个线顶点 |
| WSS 报告配色 | 新增「配色」下拉：彩虹（默认）/ Turbo / 蓝白红，按浏览器记忆 | — |
| 报告重建工具 | `python -m wss_deploy.rebuild_report [--job ID] [--no-streamlines]`：从 `field.npz` 重建 report.html；默认对体场任务重新积分流线并更新 streamlines.vtp 与 job.json（需停服务或随后重启）；`--no-streamlines` 只重建页面、从 streamlines.vtp 读回已有流线、不动 job.json，可在服务运行中执行 | 已对 8 个历史任务执行 |
| 体场报告布局 | 固定高度应用式布局：左侧菜单独立滚动、页面不滚动；截面平面视图固定在右上角色标正下方 | 手工 |
| 删除任务 | 列表每行「删除」按钮、详情页「删除任务」、「选择」进入多选后「全选已结束 / 删除选中」；确认后永久删除任务目录并写 `deleted_jobs.jsonl`；计算中的任务不可删；删除先原子重命名目录再清理，未完成的清理在下次启动时补做 | `tests/test_delete_jobs.py`（6 项） |
| 发布包缓存 LRU | 常驻 ensemble 上限 `WSS_DEPLOY_MODEL_CACHE`（默认 2），驱逐时释放显存 | `tests/test_registry.py` |
| CSV 导出 | 逐行 f-string 改为列表化写出，输出逐字节相同 | 回归 |
| 回归工具 | `python -m wss_deploy.regress --out DIR [--reference-root REF]` 重跑已完成任务的 B 段并逐键逐数组比对；`tests/test_golden_regression.py` 以 `WSS_DEPLOY_GOLDEN_REFERENCE` 作 CI 门 | 见 `outputs/wss_deploy_golden/20260920_baseline/README.md` |

未做（留给下一轮，见评审稿 §7）：FastAPI（GNN 环境未装 fastapi/uvicorn/pydantic，是否装进共享训练环境需拍板）、SQLite 任务库与 API/worker 分进程、内容寻址缓存、三套 three.js 查看器合并、用户表、PDF。

## 使用口径

- STL 不携带单位。服务会先展示原始尺寸和换算尺寸；只有明确确认单位后才提取中心线。
- 预测对象是固定收缩期帧（step 1162，约 0.21 s）的壁面 WSS。报告主指标是预测点云的空间 p99；黄色标记是全场最大值位置，二者不是同一个统计量。
- 五个 seed 在 Pa 空间取均值。报告不默认展示 seed 离散度；模型发布身份、权重哈希、输入 SHA256、采样种子、参数、时间轴和导出文件哈希写入 `summary.json` 与 `run_manifest.json`。
- 当前固定收缩期帧也通过 `time_axis` 保存；字段通过 `fields` 描述（单位、位置、标量/向量、数组键），现有 `wss_pa` 和 `peak` 等键保留为兼容视图。以后新增速度、压力或多帧结果不应覆盖旧字段。
- 三维报告中的壁面色彩和鼠标读值是 Gaussian 插值；覆盖范围外显示为灰色/无有效值。面积为点数占比乘输入壁面面积，标记为估计值。

## 服务 API

v0.8 新增接口（病例卡、查重、登录、回收站、owner 级事件、偏好、标注、发现判定、汇总表、打包）见上节表格与 `ANALYSIS_CONTRACT.md` §11。页面使用 `/api/session`、`/api/releases`、`/api/jobs`、`/api/jobs/batch`、`/api/jobs/<id>`（轻量状态，不含预览网格）、`/api/jobs/<id>/geometry`（预览网格与中心线折线，ETag）、`/api/jobs/<id>/events`（SSE 实时状态）、`/api/jobs/<id>/input`、`confirm`、`cancel`、`retry`、`rerun` 和 `/api/compare`。 删除：`POST /api/jobs/<id>/delete`（body `{version}`）与 `POST /api/jobs/delete`（body `{jobs:[{id,version},…]}`，每次 ≤ 100，逐项返回结果）。删除会移除整个任务目录（输入 STL、中心线、报告、导出、history），只拒绝正在计算的任务；每次删除在 `outputs/wss_deploy_jobs/deleted_jobs.jsonl` 追加一行来源记录（任务 id、病例号、删除前状态、发布包、输入 SHA256、run_identity、删除时间）。`/api/jobs/batch` 接受重复 `stl` 字段和与文件顺序对应的 `metadata_json`，最多 20 项并逐项返回成功/错误结果；总请求体上限 256 MiB。任务历史查询可传 `q/status/patient_id/tag/page/page_size`。任务状态包括 `queued`、`running`、`awaiting_input`、`awaiting_confirmation`、`done`、`failed`、`cancelled` 和 `interrupted`。所有变更请求带会话 CSRF token 和任务 `version`，重复确认会返回 409 并要求刷新。比较接口只返回声明一致的 p99、最大值、均值和分支 p99 标量；字段、帧、阈值或预处理参数不一致时会明确拒绝差值。

服务会限制 STL 为 128 MiB、2,000,000 面片；只开放已完成任务的报告和导出白名单。任务重启后会标记中断，已有输入检查和中心线可以通过“重试”复用。

## P1：发布包选择、结果比较和通用字段浏览

当前 P1 以设计文档 §15–16 为准。服务启动时只扫描 `outputs/wss_deploy_release/` 下的 `release.json`，通过 `GET /api/releases` 返回可用发布包、指纹和推理合同；上传任务可用表单字段 `release_id` 选择发布包。任务会持久保存发布包 id、指纹和合同，重试继续使用原绑定，不能因默认发布包变化而换权重。

已完成任务可以调用 `POST /api/jobs/<id>/rerun`，请求体为 `{ "version": <当前版本>, "release_id": "另一个发布包" }`。该接口新建任务并复用原始 STL、中心线和已确认出口映射，旧任务的报告与导出文件保持不变。每个结果含稳定的 `run_identity`，由输入 SHA256、发布包指纹、映射和推理参数共同决定，便于同病例比较。

报告读取 `fields` 和 `time_axis`。当前正式发布包 `X5D_v51_5seed_20260916` 只有固定收缩期单帧壁面 WSS（step 1162，约 0.21 s）；报告会显示字段单位/位置/类型，时间帧选择器保持禁用，并明确没有速度、压力或多帧数据。未具备真实训练和验证合同的字段不会通过界面伪造。新增 `PF6_VF6_peak_3seed_20260920` 已提供真实压力＋速度单帧合同；多时间帧仍未启用。

发布包目录可通过 `WSS_DEPLOY_RELEASE_ROOT` 覆盖；命令行也支持 `--release-root` 与 `--release-id`。不满足已支持的显式单帧合同、路径不安全或元数据损坏的发布包会被隐藏并拒绝执行。

## P2：批处理、历史检索与运行复核（2026-09-19）

网页现在支持一次选择多个 STL，按文件逐个创建任务并显示部分失败；每个任务可保存匿名 `patient_id`、`scan_label`、`scan_date`、标签和备注。任务历史支持检索词、状态、患者编号、标签筛选和分页，接口返回 `jobs/total/page/page_size`，仍按会话 owner 隔离。底层 `JobManager.create_batch()` 还提供最多 20 项的原子保存边界和逐项错误结果，便于后续接入专用批量上传客户端。

计算设置随任务保存：`device=auto|cpu|cuda`、`seed_count`（默认发布包全部模型）和 CPU `threads`。发布包缓存按发布包指纹、设备和集成模型数区分，避免重用错误的模型子集；运行摘要和 `run_identity` 记录这些参数。相同 STL 的特征采样使用输入 SHA256 派生的稳定标识，不随任务目录变化。

完成任务仍只比较声明一致的标量统计；界面显示两侧病例、患者、扫描标签和发布包身份，并明确差值方向为“右侧 − 左侧”。没有坐标注册时不生成点对点差值。发布包可声明 `geometry_reference` 和基于 CV3 折外预测的 `population_reference`，未声明或合同不匹配时显示“未配置”，不会读取 CFD 训练分布或伪造百分位。结果摘要包含参考评估和匿名病例元数据，报告会显示复核状态。

当前 P2 尚未把重复文件合并为单个 multipart `/api/jobs/batch` 请求，也没有引入 PDF 渲染依赖；网页的逐文件上传、HTML 打印和标准库 HTTP 服务保持兼容。真实人群百分位需要在发布包内提供经过核验的 CV3 OOF 参考声明后才会启用。


## PF6 / VF6 体场（2026-09-20）

在上传区选择 `PF6_VF6_peak_3seed_20260920`；已有完成病例可以选该包点击“用发布包重跑”，复用 STL、中心线和已确认出口，生成新任务。切换发布包会恢复新包的全部模型；设备和 CPU 线程设置保留。默认 `X5D_v51_5seed_20260916` 仍提供原 WSS 功能。

- **压力 PF6**：3 seed（1234 / 7 / 2025），分别查询壁面与内部，在 Pa 空间集成。输出是 `p − 当前峰值帧体积平均压力`，不能恢复绝对血压，也不会用当前采样点的均值再次校正。
- **速度 VF6**：相同 3 seed，仅查询内部点。各分量线性反归一化后恢复 STL 世界坐标，再做向量均值；速度大小是集成向量的模长。壁面零速度不作为展示对象。
- 两者仍是固定收缩期 `step 1162 / 0.21 s`，不是全周期预测。新包共六份真实权重；页面模型数按“每个字段”计算。
- 原壁面重采样提供 support；`volume_geometry.py` 构建训练对应的 20 维输入（18 基础维＋2 Murray 特征），生成 20,000 个内部 query：70% 按分支长度与局部半径自适应分配，并给每个中心线分支最低配额；至少 30% 是只经闭合三角面检测的独立均匀采样，以补充瘤囊覆盖。切口仅用于封闭管腔测试，不作为壁面 support。每个内部点通过闭合三角面的包含检查；距壁距离不含人工盖子。流程无需 CFD 文件或训练真值。

体场报告支持：速度内部点云、方向箭头、稳态流线；压力内部／壁面视图；沿中心线移动或按 XYZ 轴定位的手动截面（位置、双轴旋转、厚度）；每分支弧长 5%、20%、40%、60%、80%、95% 的自动截面。右侧提供平面投影与所选点统计。压力内部与壁面共用色标。

截面目前是**有限厚度预测点云切片**；二维图使用局部 IDW 色带插值，只在凸包、邻域数量、距离和角度覆盖都满足时绘制，不支持区域保留灰色，并叠加原始点，不会填充成连续 CFD 网格。流线使用预测世界坐标速度的局部 IDW 插值和双向中点积分，在管腔外、采样覆盖不足或低速处停止；它们是固定帧流线，不是随时间演化的粒子轨迹。局部插值是展示近似，不能据此声称精确流量或守恒。统计是采样点等权口径，不是体积积分。

新增导出：`volume_fields.vtp`（内部压力、速度向量及模长）、`wall_pressure.vtp`（独立壁面查询后 Gaussian 插值，未覆盖顶点标记无效）、`points_volume.csv`、`streamlines.vtp`（存在有效流线时）。`field.npz` 内 `pts/pressure_pa/point_kind` 对齐全部壁面＋内部点，`internal_pts/velocity_m_s/speed_m_s` 只对齐内部点；`point_kind=0/1` 分别代表壁面／内部。`run_manifest.json` 记录两个字段、参考压力、坐标旋转、实际权重哈希和全部导出哈希。

发布包重建工具：`PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m wss_deploy.build_volume_release --help`；仅复制冻结推理文件，已有包拒绝覆盖。新包的模型源为 `wss_local_wave2_20260912` 和 `wss_local_wave3_20260913` 的 PF6/VF6 三个 seed。新增模块：`volume_geometry.py`、`volume_pipeline.py`、`streamlines.py`、`volume_report.py`、`static/volume_viewer.js`。

部署烟测使用既有病例 STL 副本（不读取 CFD），输出在 `outputs/wss_deploy_volume_smoke_20260920/`：53,283 壁面点、20,000 内部点、六模型 GPU 推理约 7.2 s，完整流程约 58 s（含继承的输入／中心线计时）。这验证部署链路可运行，不代替体场科学精度评估。测试入口为 `PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m pytest -q tests`。
