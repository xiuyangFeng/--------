# wss_deploy — 从 STL 到 WSS / TAWSS / OSI / RRT / ECAP、压力与速度体场（v0.15，2026-09-26）

只做推理。链路：`ingest`（单位/拓扑检查）→ `centerline`（vessel_geom/VMTK 子进程 + 出口自动命名 + 人工确认）→ `geometry`（1 mm Taubin 平滑 → 0.5 mm 重采样 → 27 维特征，全部来自冻结特征库 `wss_features/`）→ `infer`（可替换的发布包，按模型族适配器 `families.py` 推理与集成）→ `metrics` + `report`（summary.json / run_manifest.json / report.html / wall_wss.vtp / points_wss.csv / field.npz）。

```bash
# 新机器：两个 conda 环境与 git 忽略的发布包 / vessel_geom 工具包怎么准备、怎么核对，见 env/README.md
export PYTHONPATH=/public/newhome/cy/Digital_twin/GNN
PY=~/.conda/envs/GNN/bin/python
# 单例：先检查并提取中心线；命令行必须显式确认出口映射
CUDA_VISIBLE_DEVICES=1 $PY -m wss_deploy.cli run case.stl --out outputs/wss_deploy_jobs/case --case-id CASE --stage-a-only --units mm
# 只做 A 段（中心线 + 命名建议，CPU），看 proposal 再决定
$PY -m wss_deploy.cli run case.stl --out jobs/case --stage-a-only --units mm
# 指定出口命名（叶段 id=名字）
$PY -m wss_deploy.cli run case.stl --out jobs/case --outlets "3=out-li,4=out-le,5=out-ri,6=out-re"
# 演示服务（默认仅本机；远程使用 SSH 隧道）：后台托管启动，参数写入 <jobs_root>/service.json，日志 <jobs_root>/server.log
$PY -m wss_deploy.cli service start --host 127.0.0.1 --port 8765 --env CUDA_VISIBLE_DEVICES=1
$PY -m wss_deploy.cli service status        # 之后改代码用 service upgrade，停止用 service stop
# 局域网共享（必须令牌；推荐放在反向代理/TLS 后）：令牌自动生成在 <jobs_root>/.service_token（0600），service token 打印
$PY -m wss_deploy.cli service start --host 0.0.0.0 --port 8765 --env CUDA_VISIBLE_DEVICES=1
# 前台调试时令牌用文件传入（--token 仍可用，但会告警且令牌出现在进程命令行里）
CUDA_VISIBLE_DEVICES=1 $PY -m wss_deploy.cli serve --host 0.0.0.0 --port 8765 --token-file ~/.wss_deploy_token
```

浏览器打开 `http://master:8765/`（校园网内直接访问 master 的地址，或本机 `ssh -L 8765:localhost:8765 <user>@master` 后打开 `http://localhost:8765/`）。

| 模块 | 作用 | 备注 |
|---|---|---|
| `paths.py` | 发布包、vessel_geom、VMTK 解释器路径（可用环境变量 `WSS_DEPLOY_RELEASE / WSS_DEPLOY_VESSEL_GEOM / WSS_DEPLOY_VMTK_PYTHON` 覆盖） | 权重只从 `outputs/wss_deploy_release/<release>/` 读 |
| `ingest.py` | 显式单位确认、文件/面片上限、连通片、流形和开口计数、写干净的二进制 STL | 单位不再静默猜测；开口 ≠ 5、非流形、异常尺寸 → 阻断 |
| `centerline.py` | vessel_geom `--preset frozen-aortoiliac` 子进程（GNN_vmtk 环境）；`propose_outlets` 自动命名（左右按 x，髂内外四项加权，170 例验证 166/169 与 325/340）；`validate_mapping` / `apply_mapping` 把确认后的命名写回 atlas | `--inlet` 可改入口 |
| `geometry.py` | `build_deployment_case` 的无真值版，全部几何调用冻结特征库 `wss_features/`；capfit 规则从发布包内文件显式传入 | 不读 bundle / case.h5，不改任何全局状态 |
| `infer.py` | `Release`：按 `release.json` 的 `models` 清单加载权重（旧 X5D_v51 目录仍兼容），`predict(case)` 交给模型族适配器 | 设备 auto/cuda/cpu；记录并可校验 `feature_contract` |
| `families.py` | 模型族注册表：合同、权重选择、逐模型配置检查、集成预测、B 段流程；现有 `wall_wss_v1`、`pf6_vf6_volume_v1`、`wall_cycle_multi_v1`（M1 三头：峰值 WSS + TAWSS + OSI） | 新增模型族只需注册一条，不改 pipeline/registry/infer |
| `regress.py` | 黄金回归：重跑已完成任务的 B 段并与参照逐键逐数组比对 | 参照 `outputs/wss_deploy_golden/20260920_baseline/` |
| `errors.py` | 分类错误：InputGeometryError / ToolchainError / ResourceError，`classify()` 映射 OOM / 超时 / 缺模块；记录带 category / retryable / admin_detail | v0.14；非管理员看不到 admin_detail |
| `input_memo.py` | 集成成员共用支持点 / 特征 / 16 邻域 patch（参数完全相等才复用，返回副本；训练代码不改） | v0.14；`WSS_DEPLOY_PATCH_MEMO=0` 关闭 |
| `geometry_cache.py` | 几何缓存 `<job>/geometry_cache/<kind>-<key>.npz`（键含输入字节、参数、特征程序与代码哈希）；确认出口期间后台预计算，B 段 / 重跑 / 重建命中即原值 | v0.14；`WSS_DEPLOY_GEOMETRY_CACHE=0` 关闭 |
| `analysis.py` | 沿程曲线 / 发现列表 / 可信区域 / 解剖坐标架（合同 `ANALYSIS_CONTRACT.md` §1–§4） | 两族 B 段与 rebuild 共用 |
| `build_reference_profiles.py` | 生成发布包参考侧车 `reference.json`（几何范围 + CV3 折外 p99 人群） | 不改发布包指纹 |
| `onepager.py` | 一页纸 A4 报告（无三维） | `GET /api/jobs/<id>/onepage` |
| `rebuild_report.py` | 从 field.npz 重建报告与分析产物；`--no-streamlines` 只重建页面；`--ui-only` 只替换历史报告模板与查看器 | `--ui-only` 不触碰 summary、预测数组或分析产物，仅更新报告清单哈希 |
| `registry.py` | 扫描并校验发布包合同、清单哈希和模型文件；为任务提供稳定发布包指纹 | 支持单帧 WSS、显式 PF6/VF6 压力＋速度合同和 M1 三头周期量合同；多帧合同仍拒绝 |
| `metrics.py` | 峰值（p99 主、最大值参考、位置、热点簇）、低/高 WSS 面积、分支表、几何表 | 阈值 0.4 / 4 / 7 Pa |
| `report.py` | WSS 插值到 STL 顶点（高斯 σ 0.5 mm）、VTP、单文件 HTML（three.js 内嵌，离线可开） | 统计来自预测点云；未覆盖顶点不静默外推 |
| `pipeline.py` | `stage_a` / `stage_b` / `run_all`，每段计时写入 summary；输出通用 `fields`、`time_axis`、`results` 和 `run_manifest.json` | 旧 WSS 字段保留为兼容视图 |
| `schema.py` | 版本化的模型发布元数据、字段描述、时间轴、运行清单和 WSS 兼容层 | 为速度、压力和多时间帧预留接口 |
| `comparison.py` | 按声明合同比较两个完成任务的标量统计 | 不做不同点云的点对点差值；口径不一致时不输出差值 |
| `service.py` | 服务管理（start / stop / restart / upgrade / status / logs / token）、接管手工进程、`ServiceClient`（命令行提交） | `service.json` / `service.pid` / `.service_token` 在任务目录（v0.12） |
| `doctor.py` | 环境与部署自检（`cli doctor`） | ✓/⚠/✗ + 建议，有 ✗ 退出码 1 |
| `timeline.py` | 患者随访时间线（按输入几何合并扫描、年增长率） | `GET /api/patients/<id>/timeline` |
| `report_freshness.py` | 报告模板指纹与按需 UI-only 刷新 | 打开报告时自动；`cli reports refresh` |
| `report_template.py` | 机构报告模板 `report_template.json` | 一页纸页眉 / 签字栏 / 术语范围 |
| `devshot.py` | 无头 Firefox 截图、页面 JS 错误收集、沙箱服务、标准页面套件 | 开发自查用，不参与服务 |
| `server.py` / `jobs.py` | 本地优先 HTTP 服务：上传 → 输入确认 → 三维出口确认 → B 段 → 报告；状态机、事件、取消、重试和重启恢复 | 默认回环；共享需 token；任务落盘 `outputs/wss_deploy_jobs/<job>/job.json` |

验收与计时：`training_wss_min/experiments/wss_deploy_timing_20260917/`（分段计时、指标演示、`acceptance_test34/` 34 例回归）。设计与讨论：`docs/02-推进与变更/05-部署工具/WSS_部署演示工具_从STL到峰值WSS_整体框架与计时_2026-09-17.md`。

## v0.15.10（2026-09-28）：壁面探针卡片改成对照表

用户反馈 v0.15.9 的探针卡片是几行连在一起的文字，要求更美观。改为三段：

| 部分 | 内容 |
|---|---|
| 标题栏 | 与 3D 靶心同款的黑白圆点 +「探针」、分支 · 弧长 · 半径（过长省略），右侧「记录」（描边主色按钮）与 × 关闭 |
| 对照表 | 每个字段一行（峰值 WSS / TAWSS / OSI / RRT / ECAP，单位小字跟在名称后），三列：此点、截面均值（加粗）、环上范围条（灰线 = 截面环上最小 → 最大，竖线 = 截面均值，圆点 = 此点，超出范围为空心点，悬停给出数值）。当前着色字段那一行高亮；点任一行即切换着色字段；切字段 / 单位时卡片随之刷新。没有截面时只留「此点」一列 |
| 脚注 | 截面说明（垂直中心线，过哪支弧长多少 · 周长；开口 / 不闭合注明）、此点坐标、范围条图例；紧凑布局只留截面说明，卡片按内容宽度贴左下角 |

卡片的完整文字写在 `aria-label`（读屏软件可读）。纯界面：695 项测试通过 / 3 跳过，devshot 套件 14 页 0 错误，完整与紧凑布局截图核对。文件：report.py、tests/test_report.py。

## v0.15.9（2026-09-27）：点一下，给出该处垂直截面的积分均值（挂到中心线那一点上）

用户要求：在报告上点击一个点，就给出沿中心线该处截面的积分均值（TAWSS 等指标），积分到中心线这个点上；截面取「过中心线点、垂直于中心线」的平面，壁面报告与体场报告都要。契约 §25。

| 改动 | 做法 | 文件 |
|---|---|---|
| 站位与截面（共享） | 点击点投影到中心线 → 分支、弧长、内切半径；法向 = 中心线切向（相邻样本弦，近端 → 远端即顺流）；截面 = 壁面网格 ∩ 该平面、取包住中心线点的管腔轮廓（4 × 内切半径内搜，与「管径」测量同一套）。`stationSection` / `centerlineTangent`；「管径」测量改用同一个切向函数（算式不变） | static/report_common.js, report.py |
| 壁面：截面均值 | 轮廓每段壁面线元取离它最近的预测点的值，截面均值 = ∮f dl / ∮dl（按线长加权，与点密度无关：一半环点密 4 倍时普通点平均偏到 2.38，线积分仍是 2.00）。探针卡新增一段「截面均值（垂直中心线 · 分支 弧长 · 周长）」+ 该页全部字段（WSS / TAWSS / OSI / RRT / ECAP）；开口处的直线封闭段不计、轮廓不闭合时注明；3D 画黑边白芯截面环、中心线点靶心和探针到中心线点的连线；「记录」多出 `section_<字段>`、`section_s_from_root_mm`、`section_perimeter_mm` 列（TSV / CSV 同步） | static/report_common.js（`sectionMeans` / `ringElements` / `lineMean`）, report.py |
| 体场：截面积分 | 固定探针时计算：截面厚度控件（默认 2 mm）内、落在管腔轮廓里的体内点，每点代表离它最近的那块截面面积（Voronoi，64 × 64 格），面积平均 = Σ vᵢAᵢ / A，不施加壁面值；截面面积 A = 轮廓多边形面积，流量 Q = A · ⟨u·n⟩（m/s × mm² = mL/s，顺流为正）。探针卡新增：截面面积 / 等效直径、面积平均速度、平均穿面速度、截面流量 Q、面积平均相对压力、截面取样（点数 + 直接支撑比例）；新按钮「看截面」把截面视图放到同一平面；3D 同样画截面环与中心线点；记录多 `section_area_mm2` / `section_speed_mean_m_s` / `section_normal_mean_m_s` / `section_flow_ml_s` / `section_pressure_mean_pa` / `section_s_from_root_mm` | static/volume_viewer.js（`sectionIntegral`）, volume_report.py |

面积平均方法按质量守恒选定。三个真实体场任务（LV_GUO_YOU / FAN_JIAN_MING / SHI_YUN_XI，入口波形统一）各取 35 站：Voronoi 下四个出口支流量之和为 116.7 / 114.7 / 115.4 mL/s，主动脉各站 95–137 / 108–125 / 106–133 mL/s，分叉处主干 ≈ 两子支之和（如 FAN 右髂总 57 ≈ 22 + 35）。截面热图原用的「IDW + 壁面无滑移 0 值」做面积平均会系统偏低（出口和 92 / 86 / 84，LV 瘤颈每截面只有 15–30 个体内点，偏低约一半），所以只有热图显示保留它。壁面截面均值与沿程 2 mm 分箱均值（分叉口站除外）中位相对差：WSS 4–6%、TAWSS 3–4%、OSI 6–7%；分箱按投影归点，大瘤囊里两者会明显不同，这正是改用垂直截面的原因。每次点击：壁面约 3–20 ms，体场约 30 ms。

已知限制：分叉口附近的垂直截面会切进相邻分支（周长突增，3D 截面环可直接看到）；体场点稀疏处个别站 Q 可偏 ±20%；模型预测的速度本身不受质量守恒约束。纯显示与读数，精度链路不变。

验收：新增 6 项测试（共享库 3、壁面报告 1、体场 2），全量 695 项通过 / 3 跳过，黄金回归 6/6（自包含，GPU 2），devshot 套件 14 页 0 错误；无头 Firefox 在 FAN_JIAN_MING 真实三头与体场报告上点击 → 卡片 → 截面环 →「看截面」截图核对。

## v0.15.8（2026-09-27）：预览网格改为连续简化面，主页缩略图能看到血管

用户反馈：新病例 SHI_YUN_XI 主页缩略图仍只有中心线、看不到血管。根因不是缺数据（`stage_a.json` 有 18 000 面预览），而是预览的做法：从清洗后网格（本例 192 388 面）里按 `linspace` 每约 11 个三角形取 1 个，得到的是互不相连的碎片（45 655 顶点 / 18 000 面，顶点数是面数 2.5 倍）；缩略图再按 12 000 上限隔一个丢一个，透明度 0.22，几乎不可见。出口确认页的血管呈斑点状也是同一原因。小网格病例（≤ 18 000 面，如 LV_GUO_YOU 的 23 184 面也只被抽掉少量）不明显。

| 改动 | 做法 | 文件 |
|---|---|---|
| 连续简化预览 | 新模块 `preview.py`：均匀网格顶点聚类（格内顶点取均值、去退化与重复三角形、格子逐步放大直到 ≤ 18 000 面）；192 k 面 → 15 697 面 / 7 864 顶点，97.8% 边为两面共享，包围盒偏差 ≤ 0.6 mm，0.57 s。≤ 18 000 面的网格原样保留（与旧版逐位相同，只多 `method` / `source_faces` 两个键）。只用于显示，预测始终用完整清洗 STL | preview.py, pipeline.py |
| 老任务现场补建 | `JobManager.geometry`：旧式抽样预览（无 `method` 且达 18 000 面上限）读时按任务目录里的清洗 STL 重建一次，缓存为 `geometry_cache/preview-<STL sha16>.json`；`job.json` 与 `stage_a.json` 不改；失败时退回旧预览。`/geometry` 的 ETag 加入预览版本，浏览器不会继续用缓存的碎片 | jobs.py, server.py |
| 缩略图不再丢面 | 三维缩略图面数上限 12 000 → 20 000（18 000 面全部绘制）；无 WebGL 时的二维示意图也绘制全部面 | static/app.js |

验收：新增 `tests/test_preview_mesh.py`（小网格原样、大网格连续且在预算内、老任务读时补建并命中缓存且记录不变）；全量 689 项通过 / 3 跳过，黄金回归 6/6，devshot 套件 10 页 0 错误；沙箱截图 SHI_YUN_XI 三头与体场两个任务的主页缩略图均显示半透明血管 + 中心线。

## v0.15.7（2026-09-27）：一次上传同时预测三头模型与体场

用户要求：上传一个病例时可以选择同时预测体场和三头模型（WSS + TAWSS + OSI）；单独的壁面 WSS 暂不参与组合（后续可能并入三头模型）。

| 改动 | 做法 | 文件 |
|---|---|---|
| 上传选项 | 发布包下拉在单个发布包之外，为每个「三头 × 体场」组合追加一项「WSS + TAWSS + OSI ＋ 压力 + 速度体场（同时预测，两个任务）」，取值 `主id+伴随id`；选中时集成模型数固定为「全部模型（各发布包）」，说明改为两个任务的解释；记住上次选择时保存组合本身 | static/app.js |
| 接口 | `POST /api/jobs` 与 `/api/jobs/batch` 新增表单字段 `companion_release_ids`（逗号分隔，最多 2 个）；`JobManager.create(..., companion_release_ids=)` 校验发布包存在、不与主发布包重复，写入 `job.companions = [{release_id, job_id}]` | server.py, jobs.py |
| 伴随任务 | `_spawn_companions`：出口一经确认（手动 `confirm`、自动高置信门控、查重复用上传）即按「换发布包重跑」同一路径（`_clone_for_stage_b`）为每个伴随发布包建一个只跑 B 段的新任务，复用输入、中心线与已确认出口；新任务带 `companion_of`，主任务记下其编号。每个伴随只创建一次（之后改出口重算不再创建）；创建失败写在主任务 `companions[].error` 与 `companion_failed` 事件里，不影响主任务。伴随事件不增主任务版本号，已排队的 B 段条目不会因此失效；自动门控路径下伴随任务按自己的发布包重新评估门控，不过则单独等待确认 | jobs.py |
| 详情页 | 主任务显示「同时预测 压力 + 速度体场：确认出口后自动创建 / 任务 xxx 〔打开〕」；伴随任务显示「与任务 xxx 同时上传的压力 + 速度体场预测，复用其中心线与出口确认〔打开该任务〕」；两个结果按病例并列 | static/app.js, app.css |

验收：新增 `tests/test_companions.py`（手动确认、自动门控、校验、创建失败隔离、批量上传 5 项）与 `test_v0157_upload_offers_three_head_plus_volume_and_sends_the_companion`；全量 687 项通过 / 3 跳过，黄金回归 6/6，devshot 套件 10 页 0 错误；沙箱端到端：LV_GUO_YOU 的 STL 按组合上传 → A 段后等待确认（此时未建体场任务）→ 确认出口 → 自动创建体场任务 → 两个任务都完成（CPU，体场 B 段 22 s），病例卡片并列显示两个结果。

## v0.15.6（2026-09-27）：体场探针卡片让位给截面平面视图

用户反馈：完整布局下没有固定探针时，探针卡片把 9 行读数全部铺开（标签还折成两行），把下面的截面平面视图挤到只剩标题。

| 改动 | 做法 | 文件 |
|---|---|---|
| 未固定探针 = 一行要点 | 悬停读数时卡片加 `brief` 类，只显示当前物理量（速度大小或相对压力；壁面点为壁面压力）与分支，提示改为「悬停读数 · 单击血管固定并查看全部读数」；单击固定后才列出全部读数。其余行仍在 DOM 里（CSS 隐藏），记录 / 导出不变 | static/volume_viewer.js, volume_report.py |
| 截面优先 | 截面模式下右侧栏顺序改为 色标 → 截面平面视图 → 探针（CSS `order`，DOM 顺序不变）；空间不够时探针卡片先收缩并滚动（收缩权重 20 : 1），保底高度未固定 78 px、固定 `min(128px, 24%)`，截面画布保持完整 | volume_report.py |
| 标签不折行 | 探针读数表改为 `auto / minmax(0,1fr)` 两列，标签不换行，长数值在值列内换行 | volume_report.py |

纯 UI。验收：新增 `test_v0155_unpinned_probe_is_one_line_of_key_readings_and_pinned_lists_all`（测试钩子 `setProbe`）；全量 681 项通过 / 3 跳过，黄金回归 6/6，devshot 套件 10 页 0 错误；截图核对：1300 × 820 完整布局截面模式，悬停时探针为底部一行、截面画布完整，固定后显示 3–4 行读数可滚动、截面画布仍完整。

## v0.15.5（2026-09-27）：TAWSS 最小值标记 + 并排比较时视口内元素自适应

用户要求：① TAWSS 也要标出最小值；② 并排比较时截面和体场那边的指标卡、截面卡要自适应变小，确保血管能看到。

| 改动 | 做法 | 文件 |
|---|---|---|
| TAWSS 最小值 | `fieldPeak(id, low)` 取预测点最小值（`np.argmin` 同序）；TAWSS 时另加白色标记与标签「TAWSS 最小值 0.0932 Pa」，统计卡加「最小 / 最小值位置 / 白色标记」，打印脚注与截图说明同步；复选框改为「全场最大值标记（TAWSS 另标最小值）」，同一开关控制两个标记 | report.py |
| 体场视口自适应 | `#volume-view` 设为尺寸容器（仅屏幕），按**三维视口自身**宽高（不是窗口）分三档：≤ 1000 px 右侧栏宽 `clamp(210px, 36cqw, 330px)`、截面平面视图画布高度按视口高缩放、截面工具条移到左上角并隐藏长提示；≤ 680 px 再收窄并隐藏色标注释和视图说明；≤ 460 px 工具条单行可横向滚动 | volume_report.py |
| 体场相机避让 | 右侧栏（色标 + 截面平面视图）竖向占到视口 45% 以上时，标准视角按左侧空白区的宽高适配，并用 `camera.setViewOffset` 把画面中心移到空白区中央；左上角的截面工具条同样把空白区往下推。拾取与标签用同一投影；只有图例时不移动 | static/volume_viewer.js |
| 紧凑布局截面模式 | 视口加 `mode-slice` 类；截面模式下隐藏会被工具条盖住的一行视图说明 | volume_viewer.js, volume_report.py |
| 壁面视口自适应 | `#view` 同样设为尺寸容器：≤ 760 px 字段切换条移到左上角并缩小、色标变短、底部提示缩小；≤ 520 px 字段条可横向滚动、色标下移、底部提示隐藏 | report.py |

纯报告页 UI 改动，精度链路不变。验收：新增 `test_v0154_camera_is_framed_to_the_area_the_right_dock_and_slice_toolbar_leave_free`，扩展 `test_peak_marker_follows_the_coloured_field`（最小值）；全量 680 项测试通过 / 3 跳过，黄金回归 6/6，devshot 标准套件 10 页 0 个 JS 错误；并排比较在窗口 2000 / 1500 / 1200 宽（强制完整布局与自动紧凑布局）下截图：血管均完整可见，改前 1500 宽时截面工具条被挤成竖排文字、血管完全被遮挡。

## v0.15.4（2026-09-27）：最大值标记跟随着色字段

用户要求：报告切到 TAWSS / OSI 等指标时，三维图里也要标出该指标的最大值，而不是始终只标峰值 WSS 最大值。

| 改动 | 做法 | 文件 |
|---|---|---|
| 黄色标记跟随字段 | 新增 `fieldPeak(id)`：峰值 WSS 仍用 summary 的 `peak.xyz_mm / max_pa`（位置、文字与旧版逐字相同）；TAWSS / OSI / RRT / ECAP 在页面内对预测点数组取第一个最大有限值（与 `np.argmax` 同序），标记移到该点，标签为「TAWSS 最大值 11.9 Pa」「OSI 最大值 0.394」；某字段无有限值时隐藏标记 | report.py |
| 统计卡 | 周期量卡在「最大」后加「最大值位置」（分支、距入口、距分叉）与「黄色标记」行；峰值 WSS 卡在其他字段下注明「切回峰值 WSS 时标在此处」；打印脚注、截图说明随字段改写，周期量截图说明补「最大值」 | report.py |

纯报告页 UI 改动，精度链路与 summary 不变；已有报告打开时由 `report_freshness` 自动刷新。验收：新增 `test_peak_marker_follows_the_coloured_field`，全量 679 项测试通过 / 3 跳过，黄金回归 6/6，沙箱 devshot（LV_GUO_YOU M1 报告）TAWSS 标在分叉、OSI 标在近端主动脉。

## v0.15（2026-09-26）：上线前加固——安全、可运维、前端与操作性

用户要求在正式上线前再做一轮「前端 / 操作性 / 安全」优化，允许子智能体并行、由用户验收。三路 opus 通用代理并行（S 安全审计与加固 / O 可运维与上线就绪 / F 前端与操作快赢），主会话做合并胶水、全量验证与文档。**精度链路（pipeline / geometry / morphology / centerline / ingest / infer / families / analysis / wss_features）一行未改**，黄金回归 6/6。用户此前的裁定继续有效：彩虹默认色标、「计算耗时」卡、菜单式布局保留；不拆 app.js、不做设计令牌 / 暗色；不做 systemd / cron / 备份、不做 tag 部署、不做中心线抽稀。**网络层开关全部默认关闭**：不设新环境变量时行为与 v0.14 相同（登录限速按地址默认值 10 → 30 除外）。

### ① 安全（S）：审计 18 项，无 XSS / 路径穿越 / 头注入实锤；修实锤 + 反代就绪

| 改动 | 做法 | 验收 | 文件 |
|---|---|---|---|
| 反代 / TLS 就绪（opt-in） | `WSS_DEPLOY_TRUST_PROXY=1` 且 TCP 对端是回环时才信任 `X-Forwarded-For`（取最右一个非回环地址）/ `-Proto`（只认 http/https）/ `-Host`；客户端 IP 用于限速、审计与 access.log，协议用于 Origin 校验；**同时强制共享模式**（审计发现：nginx 反代到绑 127.0.0.1 的服务会落进免登录的本机模式）。`WSS_DEPLOY_COOKIE_SECURE=auto/1/0`：判定为 https 时会话 Cookie 加 `Secure` 并发 `Strict-Transport-Security: max-age=31536000`。示例 `env/nginx.example.conf`；步骤见 `env/README.md` §4 | test_server_hardening_v015 四项：默认下伪造头一律忽略；可信代理下 IP / 协议 / Secure / HSTS；无账号时拒绝启动 | server.py, env/ |
| 请求解析 | 深层嵌套 JSON（`RecursionError`）由 500 + traceback 改为 400（未登录即可刷日志冲掉审计轨迹）；非 ASCII 令牌 / CSRF 头按字节比较不再 500；上传文件名剔除控制字符 | 三项测试 | server.py |
| 登录限速双桶 | 按地址默认 **30**/分（原 10，课题组共用出口 IP 时一人输错不再连累全组）+ 按用户名 10/分（`WSS_DEPLOY_LOGIN_USER_RATE_PER_MIN`）+ 原有 5 次/60 s 锁定；三种 429 都带 `retry_after`（响应体）与 `Retry-After` 头；改口令走同一套 | `test_login_budgets_per_address_and_per_user_with_retry_after` | server.py, users.py |
| umask 与 0600 | `WSS_DEPLOY_UMASK`（默认不设）；`users.json / .sessions.json / .service_token / service.json`、所有 JSON 原子写（`O_EXCL` 0600，无权限窗口）、`server.log / access.log`（含轮转）、`server.console.log`、`deleted_jobs.jsonl` 在任何 umask 下都 0600，已有日志打开时收窄 | `test_secret_and_log_files_are_0600_under_any_umask` | jobs.py, cli.py, service.py, server.py |
| 口令底线与最后管理员 | `password_problem`：拒绝纯数字、等于用户名、单字符重复、30 条常见口令；CLI `disable / role` 默认不允许去掉最后一位启用的管理员（`--force` 越过）；禁用后已无启用用户时 stderr 警告会退回令牌登录 | test_users / hardening | users.py, cli.py |
| 审计补全 | HTTP 新增 `logout`、`template_updated`、`login_throttled`；CLI `user add/passwd/role/disable`、`jobs claim` 经 `_AuditLog` 写进服务日志（0600、无口令、无患者编号） | 两项测试 | server.py, cli.py |
| 健康端点 | 非管理员的 `checks.jobs_root` 去掉服务器路径；未登录仍只 `{ok, version}` | `test_health_details_hide_server_paths_from_non_admins` | server.py |

已核实无问题：Content-Disposition 头注入、路径穿越（白名单 + 父目录校验）、zip 成员名、multipart 上限、SSE 注入、owner 隔离与认领、患者时间线跨 owner、元数据白名单、快照 PNG 校验、令牌 / 口令不进 argv。报告 / 一页纸 / 工作台的每个 HTML sink 都经 `esc()` / `_e()` / `_script_json`；本轮顺手把报告周期卡的单位文本也加了 `esc`。

**记录未修（见 §「待拍板」）**：结果文件内嵌训练运行的绝对路径（溯源契约）；禁用最后一个用户会退回令牌登录；存量任务目录 0775；每连接一线程（前置 nginx 缓解）。

### ② 可运维（O）：统一时钟、上线预检、状态与就绪、磁盘卫生、演练、崩溃可见、上线手册

| 改动 | 做法 | 验收 | 文件 |
|---|---|---|---|
| 统一时钟 | 新模块 `clock.py`：`now_iso()`（ISO-8601 秒级带偏移）、`now_local()`、`iso()`、`strftime()`、`LogFormatter`。时区优先级 `WSS_DEPLOY_TZ` → 本次 `--env TZ=` → `service.json` 的 `env.TZ` → 系统时区；**刻意不看 shell 的 `TZ`**（本机 shell 是 America/Los_Angeles、系统是 Asia/Shanghai，这就是 -07:00 与 +08:00 混写的来源）。§28 列出的全部调用点 + grep 到的 3 处改走它；server.log / access.log 用 `LogFormatter`；任务 id 格式长度不变；`service start` 先写 service.json 再起子进程。行为变化：summary / stage_a 的 `created_at` 由无偏移本地时间改为 ISO 带偏移（报告 / 体场 / 一页纸兼容；黄金回归不比较该字段） | test_ops_v015 四项；黄金回归两种模式 6/6 | clock.py + 12 个调用文件 |
| doctor 三级与新检查 | 新增 ℹ 信息级；`bind_login`（共享绑定或 TRUST_PROXY 时必须有启用的管理员，否则 ✗）、`tls`（直接暴露 HTTP → ℹ）、`secret_perms`（凭据文件非 0600 → ⚠ + chmod 命令）、磁盘阈值 `WSS_DEPLOY_MIN_FREE_GB`（默认 < 20 GB ⚠、< 5 GB ✗，`/api/ready` 同阈值）、`log_rotation`、`derived_cache`（ℹ）、`release_pins`（对照 `env/releases.sha256`）、VMTK 可执行位、`tz`；全部只读；`service upgrade` 预检按**覆盖参数后的配置**评判（`doctor --host`），新硬错阻断升级 | test_ops_v015 / test_doctor；正式目录实跑 ✓21 ℹ3 ⚠1（git dirty）✗0，前后 find 快照一致 | doctor.py, service.py, jobs.py |
| `service status` / `/api/ready` | status 新增登录模式、最近升级 + `maintenance_result.json` 摘要、最近启动、时区、ready 探测，`--json`；进程消失时写明「service.pid 存在但进程已不在」+ 日志尾 30 行 + 启动命令；`/api/ready` 只对管理员附 `summary`，未登录仍只 `{ok, version}` | `test_status_reports_login_maintenance_and_a_crashed_service`、`test_ready_summary_only_for_administrators` | service.py, server.py |
| 磁盘卫生 | `cli jobs du [--top N] [--json]` 只读按任务列占用（field.npz / report / geometry_cache / history 分项 + 回收站 / .tmp / 日志）；`cli jobs prune-cache --older-than DAYS`（默认只列清单，`--yes` 才删，服务运行中需 `--force`，跳过未结束任务，只清 `geometry_cache/` 与 `.tmp/`，记录 `prune_cache.jsonl`） | `test_jobs_du_…`、`test_prune_cache_…`（含另一进程持锁被拒） | housekeeping.py, cli.py |
| 上线演练 | `cli service rehearse`：复制最新 2 个已完成任务到临时目录，用**盘上的新代码**在 127.0.0.1 随机端口 + CPU 起服务，烟测 未登录 ready → 口令登录（admin）→ 列表 / 详情 → 报告 / 一页纸 → 管理员 ready，结束清理；`--keep / --json / --preload`；约 4 s | 两项真子进程测试；对正式任务副本实跑 2 次通过 | rehearse.py, cli.py |
| 崩溃可见性 | 工作线程死亡（含 `BaseException`）写 CRITICAL + diagnostic_id，`/api/ready` 503，**不自动拉起**（任务可能半写）；`serve` 装 `threading.excepthook`；请求 / SSE 线程异常经 `handle_error` 记 diagnostic_id；预计算线程由下一次调度重建 | 三项测试 | jobs.py, service.py, server.py, cli.py |
| 上线手册 | `env/GO_LIVE.md`：上线前检查表 / 上线命令 / 上线后验证 / 日常 / 故障处置 / 回滚（含 job.json 与 `created_at` 兼容性） | 人工审读 | env/GO_LIVE.md |

### ③ 前端与操作性（F）：34 条真实截图审计，27 条本轮修

真沙箱三视口（1440×900 / 1024×768 / 390×844）改前 81 张、改后 84 张、live 3 张、两次 suite 39 页，JS 错误均 0；iPad 宽度模拟触屏扫描 10 页：触控目标 < 40 px、对比度 < 4.5:1、无名按钮均由每页 21–40 个降到 0。

| 改动 | 做法 | 验收 | 文件 |
|---|---|---|---|
| 会话过期不丢页面 | 401 → 登录框 `<dialog id="relogin-dialog">` 盖在当前页面上（不可 Esc），保留病例、`#job=`、已填内容；同一用户登录后 `resumeSession` 只重开事件流并刷新，不刷新页面；换用户则干净启动。登录 POST 与改口令的 401 不再当会话过期 | `test_v015_expired_session_opens_login_over_the_page_and_resumes_the_same_case` | index.html, app.js, app.css |
| 登录页 | 用户名自动聚焦、记住用户名（只记用户名）、显示 / 隐藏口令、Caps Lock 提示、错误就地显示、429 按钮倒计时（读服务 `retry_after`，缺省 60 s） | `test_v015_login_page_…` | index.html, app.js |
| 断线与升级 | 事件流掉线顶栏琥珀「实时更新中断 · 正在重连…」并每 10 s 重开；健康检查 503 →「服务正在启动或维护」；`/api/health` 的 `version` 或 `ui_build`（登录后静态文件指纹）变化 → 常驻提示「服务已更新…请刷新页面」 | 真浏览器：停服务 / 伪版本重启两张截图 | app.js, server.py |
| 引导与上传 | 概览与设置菜单「三步上手」；单位 / 出口各一句「为什么要确认」；上传前逐文件检查（扩展名 / 空文件 / 大小 / 同名）只传合格的；XHR 字节级进度；按会话给出的 `max_batch_bytes / max_batch_files` 切批（缺省 240 MB / 20）；病例 / 患者编号即时校验控制字符与长度、疑似真名警告 | `test_v015_wait_text_upload_checks_chunks_and_identifier_hints` | workbench_core.js, app.js |
| 列表与详情 | 取消任务先确认（说明可重试）；长等待按小时 / 天；长名折两行再省略（1024 / 390 不再横向滚动）；选择模式选中行高亮、禁用按钮写原因；出口表 5 个端点 + 确认按钮在 1440×900 同屏（侧栏 685 → 511 px）；对话框关闭焦点回到触发按钮 | 截图对照 + 桩测试 | app.js, app.css |
| 可及性 | `pointer:coarse` 下控件 ≥ 44 px（「?」圆保持 16 px 只扩点击区）；5 处对比度修到 ≥ 4.5:1；详情区程序焦点不画框 | a11y 扫描 0 项 | app.css, report.py, volume_report.py, onepager.py |
| 报告页 | 体场报告补「← 工作台」；两份报告色标加读屏文字（随字段）；页脚「未记录审阅状态」→「待审阅」；一页纸窄屏打印按钮可见 | 两项模板测试 | report.py, volume_report.py, volume_viewer.js, onepager.py |
| devshot | `sandbox / suite / 单张 --login 用户:口令`：用户名登录模式沙箱（与正式一致），suite 先截登录页 | `test_login_spec_and_sandbox_users_file` | devshot.py |

**合并胶水（主会话）**：`/api/session` 增加 `max_batch_bytes / max_batch_files`；登录后的 `/api/health` 增加 `ui_build`（静态文件内容摘要，按 stat 缓存；未登录仍只 `{ok, version}`）；一页纸附录「生成时间」把 ISO 显示为 `YYYY-MM-DD HH:MM:SS`；报告周期卡单位 `esc`；版本 0.15.0（`tests/test_merge_v015.py`）。

### 新环境变量与命令（默认值）

| 变量 / 命令 | 默认 | 作用 |
|---|---|---|
| `WSS_DEPLOY_TRUST_PROXY` | 0 | 信任回环对端的 `X-Forwarded-*`，并强制共享模式（要求登录） |
| `WSS_DEPLOY_COOKIE_SECURE` | auto | https（经可信代理）时 Cookie `Secure` + HSTS；1 总是；0 从不 |
| `WSS_DEPLOY_UMASK` | 未设 | 八进制，如 077；`serve` 启动时 `os.umask` |
| `WSS_DEPLOY_LOGIN_RATE_PER_MIN` | **30**（原 10） | 每地址每分钟登录 / 改口令尝试 |
| `WSS_DEPLOY_LOGIN_USER_RATE_PER_MIN` | 10 | 每用户名每分钟尝试 |
| `WSS_DEPLOY_TZ` | 未设 | 时区最高优先级；通常用 `service upgrade --env TZ=Asia/Shanghai` 写进 service.json |
| `WSS_DEPLOY_MIN_FREE_GB` | 5（硬错），20（警告） | `<硬错>[,<警告>]`，doctor 与 `/api/ready` 共用 |
| `cli doctor --host HOST`、`--skip exposure/cache/tz` | — | 按将要使用的绑定评判；新检查组 |
| `cli service status [--json]`、`service rehearse`、`jobs du`、`jobs prune-cache` | — | 见 ② |
| `devshot … --login 用户:口令` | — | 用户名登录模式沙箱与截图 |

### 待拍板（本轮记录、未替用户决定）

1. **是否上 nginx + TLS**：要上则 `service restart --host 127.0.0.1 --env WSS_DEPLOY_TRUST_PROXY=1` + `env/nginx.example.conf`（需证书与域名）；不上则口令与 Cookie 仍是明文 HTTP，且**不要**只加 nginx 而不开 `TRUST_PROXY`（免登录陷阱）。
2. **是否设 `WSS_DEPLOY_UMASK=077`**（home 已是 0700，属纵深防御），以及存量目录 / 日志收窄：`chmod 700 outputs/wss_deploy_jobs outputs/wss_deploy_jobs/20260917_* outputs/wss_deploy_jobs/test_KANG_YONG*`、`chmod 600 outputs/wss_deploy_jobs/{server.log*,access.log*,server.console.log,deleted_jobs.jsonl}`（新服务打开日志时也会自动收窄）。
3. **回滚目标**：HEAD（e8d3b69）仍是 v0.11.1，v0.14 与本轮均未提交；上线前先提交或导出补丁，否则 GO_LIVE §6 的回滚没有目标。
4. 结果文件内嵌服务器绝对路径是否在打包时脱敏；禁用最后一个用户退回令牌登录怎么处理；工作线程死亡是否自动拉起；`created_at` 新外观（ISO 带偏移）是否保留。

### 验收（合并后，主会话）

全套 **676 项测试通过、3 项跳过**（基线 629 + 本轮 47；4 分 26 秒）、**黄金回归 6/6**（自包含模式，GPU 2，精度链路未动）、合并后新沙箱（用户名登录模式，5 例）**devshot suite 20 页 0 个 JS 错误**、版本 **0.15.0**。正式 8765 服务全程未动（仍 v0.14.0，PID 293808），`outputs/wss_deploy_jobs/` 只读。一次偶发：`test_audit_covers_logout_template_review_metadata_without_patient_ids` 在 7 文件批跑中失败一次，单跑与两次复跑均通过，未定位。**未上线**：上线由用户按 `env/GO_LIVE.md` 执行 `service rehearse` → `service upgrade --drain 600 --env CUDA_VISIBLE_DEVICES=1 --env TZ=Asia/Shanghai`；升级后 `created_at` 新写入带 +08:00，历史报告因模板变化会在打开或升级时自动 UI 刷新。

### v0.15.1 用户裁定与追加（2026-09-26 晚）

用户对「待拍板」的裁定：**① 先提交保证可回滚**（已提交 `061713f`：部署线 v0.12–v0.15 全部改动，含 wss_features / tests / 05 文档；本节追加另成一提交）；**② 暂不完全上线，先展示使用**（不上 nginx / TLS，保持 0.0.0.0 + 用户名登录）；**③ 目录权限收窄**（已执行：`outputs/wss_deploy_jobs` 整棵树 `chmod -R go-rwx`，0 个组 / 他人可读项，服务不受影响；上线命令加 `--env WSS_DEPLOY_UMASK=077` 让新文件保持只属本账号）；**④ 结果文件脱敏**（已做，见下）；**⑤ 工作线程死亡不自动拉起**（保持现状）。

| 改动 | 做法 | 验收 | 文件 |
|---|---|---|---|
| 结果文件脱敏 | `schema.redact_paths()`：把项目根替换为 `<project>`、home 替换为 `~`，递归作用于字符串；`model_release_metadata()` 返回前统一脱敏（summary / run_manifest / quality_audit / job.json / report META 中 `model_release.source_runs[*].path / test34_metrics` 不再含 `/public/newhome/...`）；`rebuild_report` 读入旧 summary 时脱敏并顺带重写 `quality_audit.json`；`ANALYSIS_VERSION` 升到 `2026-09-26`，因此 **下一次 `service upgrade` 会自动完整重建全部已完成任务**（从 field.npz 重建、不重跑模型，单例秒级），历史文件随之脱敏；**v0.15.2 补**：v0.14 之前的记录没有 `analysis_version`，首次上线时 5 例未被列入重建，故 `needs_rebuild` 增加探针「`model_release` 含服务器绝对路径即需重建」（`_has_server_paths`），再次 `service upgrade` 后自动重建；三例 v0.3 时代的老任务（09-17 / 09-18）没有 `model_release`、路径在 `input_check`，所以探针改为看**整份 summary**，重建写出前对整份 summary / run_manifest / report META / quality_audit 脱敏（只改字符串，数值不动）；job.json 的 `a.input_check.clean_stl` 是服务重跑要用的功能路径，保持原样（job.json 不打包、不经 API 下发） | `test_redact_paths_strips_project_root_and_home_everywhere`；沙箱副本对真实任务离线 `rebuild_report` 后五个文件 0 处绝对路径；黄金回归 6/6（`model_release` 不在比对键内） | schema.py, rebuild_report.py |

未脱敏（服务器内部文件，不打包、不经 API 下发）：`centerline/run.json`（vessel_geom 子进程记录）。`/api/releases` 只下发 `Release.public()`，本就不含路径。

验收：全套 **677 项测试通过、3 项跳过**（+1）、黄金回归 6/6、沙箱副本真实任务离线重建后 summary / run_manifest / quality_audit / job.json / report.html 均 0 处绝对路径。提交并推送 origin：`061713f`（v0.15）→ `c111ca3`（v0.15.1）→ `e6630b1`（v0.15.2 探针）→ `5fd4a45`（v0.15.3 整份脱敏）。**已上线（2026-09-26 21:55，用户要求后由本会话执行 `service rehearse` + `service upgrade --drain 600 --env CUDA_VISIBLE_DEVICES=1 --env TZ=Asia/Shanghai --env WSS_DEPLOY_UMASK=077`）**：PID 3072069、0.15.0、时区 +08:00、umask 077；三次升级后 5 例历史任务全部重建脱敏（result 文件 0 处绝对路径，`analysis_version` 2026-09-26）；全套 678 测试通过、3 跳过。

### 没做

TLS 由用户拍板；systemd / cron / 备份；tag 部署；中心线抽稀；app.js 拆模块 / 设计令牌 / 暗色；「待审阅 / 未审阅」术语统一（涉及一页纸与导出表列值）；一页纸手机端缩放（A4 打印版式）；比较页同步视角出框（预期行为）；后端配合项中未做的：401 区分码、报告生成时写 `review` 字段、批量上传逐文件错误 `code`（前端均已兜底）。

## v0.14（2026-09-24）：四维度优化——安全、速度、可运维、界面快赢

用户要求对部署做一次「中肯的优化审读」并讨论边界。四路只读审读（安全 / 性能 / 操作性 / 前端）+ 沙箱真实截图后，用户裁定：目前只用于课题组内部展示，**网络层安全（TLS、绑定地址、cookie Secure）与运维基建（systemd、备份）后置**；**精度验收分两档**：几何 / 采样 / 特征 / 形态 / 中心线 / 导出链上的改动一律「逐位相同」（Tier A），只有改变浮点归约顺序的改动（CPU 线程数、GPU kNN 复用）允许「不超过原实现运行间抖动」（Tier B）且要给实测抖动；**中心线抽稀不做**、vessel_geom 内部不动；**保持改完即 `service upgrade` 的节奏**；前端只做快赢，**保留彩虹默认色标和「计算耗时」卡**。其余按审读清单全做。五路 opus 通用代理并行（HTTP/安全、任务管理/服务、流水线/计算、前端、测试/打包/文档），共享契约见 `ANALYSIS_CONTRACT.md` §23。

### ① 安全（代码层原本无 XSS / 路径穿越 / 命令注入；本轮补的是授权、注入与资源上限）

| 改动 | 做法 | 验收 | 文件 |
|---|---|---|---|
| 会话随凭据失效 | 会话行记录令牌指纹 / 用户 `credential_generation`；改口令、禁用、改角色即失效；令牌窗口关闭后旧令牌会话失效；空闲 12 h（`WSS_DEPLOY_SESSION_IDLE_HOURS`）+ 7 天上限；运行期清理过期行；`last_seen` 每分钟最多落盘一次；本机模式无 Cookie 的访问不再写盘 | test_server_security S1 五项；test_users | server.py, users.py |
| 认领防接管 | 仅匿名（旧令牌）会话的任务可随登录认领；已注册用户名的任务 API 一律拒绝（管理员也只能用命令行）。审读发现原逻辑只判「上个 owner ≠ 自己」，共用工作站上可接管他人任务 | 具名会话不再出现在认领列表；旧令牌 → 用户名迁移流程保留 | server.py |
| 回收站管理员只读 + 审计 | restore / purge 不再带 any_owner；删除 / 恢复 / 清除 / 认领 / 登录 / 改口令写 `wss_deploy.audit`（执行人、IP、任务号；会话 owner 仅哈希） | test_admin_trash_view_is_read_only | server.py, jobs.py |
| 表格公式注入 | CSV 以 `= + - @ TAB CR` 开头的文本加 `'`；XLSX 文本单元格强制字符串 | `=1+1` 读回为字符串、无 `<f>` | export_table.py |
| 上传流式化与资源上限 | multipart 逐段写临时文件（内存 ≤ 8 MiB）并以文件对象交给任务管理器分块复制（边算 sha256 与面数）；同时上传 ≤ 2（`WSS_DEPLOY_MAX_CONCURRENT_UPLOADS`，超出 429）；每 owner 排队 + 计算 ≤ 20（`WSS_DEPLOY_MAX_ACTIVE_JOBS_PER_OWNER`）；每会话事件流 ≤ 6；登录按 IP 每分钟 10 次；未知用户名同样执行一次 scrypt；客户端 `metadata_json` 不能指定 STL 来源 | 114 MiB 上传：峰值内存 +1505 → +9 MiB，5.7 → 0.27 s；沙箱上传原字节一致 | server.py, users.py, jobs.py |
| 打包落盘 | zip 写在 `<jobs_root>/.tmp`，同时只打 1 个包，≤ 2 GiB（超出 413），带 Content-Length 发送后删除 | 13.3 / 23.1 MB 包，服务内存峰值不变 | server.py, bundle.py |
| 响应头 | 工作台 `script-src 'self'`；三维报告保留 inline；一页纸只放行打印按钮（哈希）；去掉 Server 头 | devshot 10 页 0 错误；注入的内联脚本与事件被拦截 | server.py |
| 访问日志去标识 | 请求行写 `wss_deploy.access` → `<jobs_root>/access.log`（20 MB × 5），只记路径、不含查询串，患者编号改为 `<id>`；`server.log` 只留应用行 | access.log 无查询串、无患者编号 | server.py, cli.py |
| 报告刷新不占任务锁 | 在任务目录内暂存副本渲染，加锁只做校验和原子替换；期间报告或模板变化则丢弃重来；无嵌入数组的页面直接判失败不导入渲染器 | 渲染时锁可用；高负载下不再卡 5 s | report_freshness.py |
| 错误细节分级 | 非管理员（共享模式）响应与事件流剥除 `admin_detail`；发布包错误去掉服务器路径；显存回退事件原始 CUDA 文本只给管理员 | test_unknown_exception_admin_detail_is_stripped… | server.py, jobs.py |
| gzip + ETag | > 8 KiB 的 JSON、静态文件、报告按需 gzip（级别 6，文件级 LRU 缓存 256 MiB）；静态 / 报告 / 一页纸 / 几何支持 ETag 与 304；API 仍 no-store | 报告 7.5 → 4.2 MB、6.4 → 3.3 MB；工作台每次加载 1.11 MB → 首次 0.30 MB、刷新 0 字节 | server.py |
| 令牌不进命令行 | `serve --token-file PATH`（`--token` 仍可用但告警，并在 service.json / status 中隐去）；`user role <名> --admin/--user` | test_service_v014 | cli.py, service.py |

**没做（用户裁定后置）**：绑定 127.0.0.1 / TLS 反代 / cookie Secure；umask 收紧。三维报告仍需 `'unsafe-inline'`（自包含页面）。

### ② 速度（B 段；A 段 VMTK 不动）

| 改动 | 做法 | 验收（精度档） | 文件 |
|---|---|---|---|
| CPU 推理线程默认值 | device=cpu 且未指定 `compute.threads` 时，推理期间 torch 线程设为 min(32, 核数)（本机默认 192 线程是 CPU 慢的主因：单模型 13 s → 0.7–1.4 s）；结束后恢复；GPU 不变；summary 新增 `inference_threads`，run_identity 不变 | Tier B 申报、实测为 A：192 线程连跑两次与 32 线程结果逐位相同（LV·M1、LV·X5D、KANG_YONG、LIU、体场）；LV·X5D CPU B 段 76.9 → 15.2 s；LIU 106 → 35 s | pipeline.py, volume_pipeline.py, infer.py |
| 集成输入只建一次 + 预热 | `input_memo.py`：集成循环内参数完全相等时复用支持点、特征、16 邻域 patch（返回副本，训练代码不改）；knn_memo 追加包装 `knn_interpolate` 内的 knn；预加载后每个发布包跑一次合成小管道预热 | CPU 开 / 关逐位相同（A）；GPU 与原实现差 7.6e-6 Pa ≤ 原实现两次运行差 1.1e-5 Pa，p99 前 7 位相同（B）；LV·X5D GPU 后 4 模型 0.38 → 0.08 s/个；首例首模型 2.0 → 0.8 s | input_memo.py, knn_memo.py, families.py, infer.py, registry.py |
| 几何缓存与后台预计算 | `pipeline.precompute_geometry_cache` 在**确认出口期间**把清洁网格、平滑重采样、法向 / 主曲率、形态截面表和全腔体积写入 `<job>/geometry_cache/`；键 = 精确输入字节 + 参数 + 特征程序哈希 + 代码源哈希，命中即原值；与发布包无关（X5D → M1 重跑也命中）；任务管理器单线程后台调度、stage B 运行时让路、可取消、失败不影响任务；ETA 扣除已预计算阶段且缓存命中的运行不进阶段历史 | Tier A：10 个任务 × 冷 / 预计算 20/20 与原实现逐位相同；黄金回归含预计算 6/6；LV·M1 B 段 GPU 13.8 → **5.6 s**、CPU 50.5 → **6.4 s**；LIU（27 万面）GPU 34.3 → **6.9 s**（预计算 25.8 s 在确认出口时完成） | pipeline.py, geometry_cache.py, geometry.py, morphology.py, jobs.py, eta.py, regress.py |
| 形态截面提速 | 面片按 Morton 序每 64 个一簇带包围球，只对平面能碰到的簇逐面判侧；串环改用 Python 列表（算法与顺序不变）；按半径预筛实测无收益未采用 | Tier A：截面与形态块逐位相同（含过顶点平面、LV、KANG_YONG、LIU）；LIU 形态 11.1 → 8.6 s | morphology.py |
| job.json 瘦身 | 预览网格与折线只留在 `stage_a.json`，`geometry()` / `stage_a()` 按需读；紧凑 JSON；进度事件原子写不 fsync，状态变化仍 fsync；`wss-deploy.job/v2`，v1 懒迁移 | 单次进度写入 111 → 0.9 ms、2.05 MB → 67 KB（原本每次都在全局锁内写 2 MB） | jobs.py |
| 小项 | 清洁网格 A 段读一次后缓存；多字段插值共用一棵 KD 树；发布包校验按文件 stat 签名缓存 10 分钟（变化即全量复核，失败不缓存）；重建报告的表面粗糙度与形态截面按精确输入缓存、field.npz 每数组只解压一次；`nvidia-smi` 30 s 缓存 | Tier A；第二次重建 LV 2.8 → 0.35 s | pipeline.py, registry.py, rebuild_report.py, service.py |
| 报告瘦身 | RRT / ECAP 不再内嵌（页面按同一公式现算）；分支编号 0–255 内存 uint8（`arrays.dt`）；report.html 原子写入；UI 刷新按 `dt` 解码 | LV·M1 报告 7.52 → 6.23 MB，现算值与原内嵌逐位相同；uint8 / int32 / 混合三种页面刷新后数组一致 | report.py, rebuild_report.py |

**没做**：中心线抽稀（用户裁定）；VMTK Voronoi 复用与常驻进程（在验证过的 vessel_geom 工具箱内，不动）；形态剩余的 `loop_polygon` / `section_metrics`（LIU 约 5 s）。

### ③ 可运维

| 改动 | 做法 | 验收 | 文件 |
|---|---|---|---|
| 任务目录单写者锁 | `serve` 持有 `<jobs_root>/.service.lock`（flock，记 pid / 用途）；`jobs claim`、`reports refresh`、`python -m wss_deploy.rebuild_report`、第二个 `serve` 在服务运行时拒绝并给出 PID（`--force` 越过）；离线 JobManager 不标中断、不清回收站 | 沙箱中四种情况均被拒 | service.py, cli.py, jobs.py, rebuild_report.py |
| 结果可追溯 | run_manifest 与 summary 记 `git describe --dirty`、`git_dirty`、`deploy_version`、`analysis_version`、源码哈希（wss_deploy + 实际 import 的 training_wss_min 模块）。审读发现原 manifest 只记干净 HEAD，而工作树有 43 个改动文件 | regress 跳过这五个键，黄金 6/6；doctor 显示 dirty 警告 | schema.py, pipeline.py, volume_pipeline.py, rebuild_report.py, doctor.py |
| 真实健康检查 | `JobManager.health()`：工作线程、发布包加载（`registry.preload_state`）、目录可写、磁盘、VMTK 为硬检查，GPU 信息 30 s 缓存；`/api/health` 合并（未登录只返回 ok 与 version）；新增 `/api/ready`（不健康 503）；`service` 用 ready 等待。原实现 `ok` 硬编码 True | 沙箱 /api/ready 200 | jobs.py, server.py, service.py |
| 升级预检与排空 | `upgrade` 先在子进程 import 新代码并跑 doctor，失败即中止（旧服务照常）；`--drain [s]` 停止接新计算并等待；重建与报告刷新改由新服务启动后在后台执行（期间编辑该任务返回 409），结果写 `maintenance_result.json`；启动失败打印日志尾 30 行 | 沙箱 upgrade 全流程 19 s，field.npz 不变 | service.py, cli.py, jobs.py |
| 错误分类与恢复 | `errors.py`：InputGeometryError / ToolchainError / ResourceError；失败记录带 category / retryable / admin_detail；VMTK 失败 / 超时 → 工具链错误（管理员看 stderr 尾）；CUDA 显存不足自动在 CPU 重算一次并记 `device_fallback`；重启后排队任务自动续排，计算中的仍标中断；B 段全部产物先写临时文件再 os.replace | 类型错误 / OOM → CPU / 重启续排 / 原子写测试 | errors.py, jobs.py, pipeline.py, centerline.py, ingest.py |
| 有序迁移表 | `MIGRATIONS=[(v1,…),(v2,…)]` 取代零散 setdefault；`pending_rebuilds` 先比 `analysis_version`，旧记录退回键探测 | test_jobs_v014 | jobs.py, service.py |
| 测试与打包 | 8 个曾因任务被删而静默跳过的测试改读 `tests/fixtures`（缺失即 FAIL）；黄金集加入 M1 三头任务（6/6）；`env/GNN.yml`、`env/GNN_vmtk.yml`、`env/README.md`（两环境分工、重建、git 外目录与 sha256 核对清单）；快速上手与契约陈旧引用修正 | 12 passed 0 skipped | tests/, env/, README.md, ANALYSIS_CONTRACT.md |

**没做（用户裁定后置）**：systemd / cron 拉起、夜间 rsync 备份、GPU 固定（升级时用 `service upgrade --env CUDA_VISIBLE_DEVICES=1` 即可）；tag 快照部署；时间戳统一时区（调用点清单见 05 部署工具活文档 §28）。

### ④ 界面快赢（保留彩虹默认与「计算耗时」卡）

| 改动 | 做法 | 验收 | 文件 |
|---|---|---|---|
| 并排比较同色同值 | 审读发现「同步显示口径」只同步 `range.mode:'case'` 规则不同步绝对值，两侧色标 17.0 与 5.24 Pa 并存。现在两侧就绪 / 勾选同步 / 每次显示同步后取两例 p99 较大者作共用固定上限推到两侧（只作用于该字段，不存为偏好）；字段或族不同时提示 | compare 截图两侧均 0–17.0 Pa；切 TAWSS 出现提示 | static/compare.js, compare.html, workbench_core.js, report.py, volume_viewer.js |
| 配色单一来源 | `report_common.js` 的 `PALETTES` 为唯一色表（原本三处定义且不一致）：彩虹原 10 色标逐位不变，turbo / viridis 33 色标正式表，bwr 为 RdBu-7 白心 | 彩虹 101 点逐位相同 | static/report_common.js, report.py, volume_viewer.js |
| 详情首屏 | 结论卡移到核心数字下方（计算耗时卡保留）；完成任务只显示审阅徽章；缩略图标签去重叠；左髂外 / 左髂内配色拉开 | detail 截图 | static/app.js, app.css |
| 列表不再每 10 秒重建 | 事件流在线时定时器不刷新列表（掉线兜底）；按 id + 版本只补丁变化行，保留焦点与滚动；概览仅在打开 / 刷新 / 完成类事件时重取 | 真实浏览器 12 秒后焦点与行节点不变 | static/app.js |
| 按需渲染 | 三个查看器去掉常驻 rAF，改为控件 / 阻尼 / 尺寸 / 字段变化时渲染；截图与导出前先渲染 | 闲置 3 秒后截图与导出 PNG 均非空白 | report.py, static/volume_viewer.js, static/app.js |
| 一页纸首页 | 无配图时配图区收成一行（不打印），附录直接接排；提示改指「导出」 | onepage 截图无半页空白 | onepager.py, static/app.js |
| 报告导航 | 预设移到「查看」；剖切滑块标注 STL Z 轴并显示位置；http 访问时显示「← 工作台」；两份报告统一「保存截图」「复制复现链接」；无 WebGL 提示改为统计在左 | report 截图 | report.py, volume_report.py |
| CSS 小修 | 次要文字色 #5a6d80（过 4.5:1）；最小字号 12 px；顶栏一个高度；全高布局 100dvh；小按钮 ≥ 32 px；直方图按设备像素比绘制 | 1024 / 390 截图 | static/app.css, report.py, volume_report.py |
| 错误与确认 | 删除改用样式确认框；按错误类别显示标题；不可重试时隐藏「重试」；「技术细节」折叠显示 admin_detail；429 / 413 显示服务原话与单用户上限 | 桩页面测试 | static/app.js, workbench_core.js |
| 预计算与运行记录可读 | 计时新增「后台几何预计算」；命中缓存的阶段标「已预计算」（ETA 斜纹）；「查看输入检查与计算过程」新增运行记录（重启续排 / 排空 / 显存回退 / 预计算 / 重建等中文标签） | test_workbench_js 新增用例 | static/app.js, workbench_core.js, report.py, onepager.py |

**没做**：app.js 拆模块、设计令牌统一、暗色模式、一页纸自动配图（用户裁定只做快赢）。

### 新环境变量（默认值）

| 变量 | 默认 | 作用 |
|---|---|---|
| `WSS_DEPLOY_CPU_THREADS` | min(32, 核数) | CPU 推理 torch 线程数；显式 `compute.threads` 优先 |
| `WSS_DEPLOY_PATCH_MEMO` / `_MB` | 1 / 3072 | 集成成员复用输入 / 内存上限 |
| `WSS_DEPLOY_WARMUP` | 1 | 预加载后预热推理 |
| `WSS_DEPLOY_GEOMETRY_CACHE` | 1 | 几何缓存读写 |
| `WSS_DEPLOY_PRECOMPUTE` | 1 | 确认出口期间后台预计算 |
| `WSS_DEPLOY_MORPH_PREFILTER` | 1 | 形态截面按簇预筛 |
| `WSS_DEPLOY_RELEASE_VERIFY_TTL` | 600 s | 发布包全量校验信任期 |
| `WSS_DEPLOY_SESSION_IDLE_HOURS` | 12 | 会话空闲超时 |
| `WSS_DEPLOY_MAX_CONCURRENT_UPLOADS` / `_BUNDLES` | 2 / 1 | 同时上传 / 打包数 |
| `WSS_DEPLOY_MAX_ACTIVE_JOBS_PER_OWNER` | 20 | 每 owner 排队 + 计算上限 |
| `WSS_DEPLOY_MAX_SSE_PER_SESSION` | 6 | 每会话事件流数 |
| `WSS_DEPLOY_LOGIN_RATE_PER_MIN` | 10 | 每 IP 登录尝试 |
| `WSS_DEPLOY_MAX_BUNDLE_BYTES` | 2 GiB | 打包上限 |
| `WSS_DEPLOY_GZIP_CACHE_MB` | 256 | gzip 文件缓存 |
| `WSS_DEPLOY_GOLDEN_PRECOMPUTE` / `WSS_DEPLOY_SLOW_TESTS` | 关 | 测试开关 |

### 验收与上线

全套 **628 项测试通过**（2 项为需显式开关的真浏览器 / 慢测试跳过）、**黄金回归 6/6**（自包含模式，GPU 1，147 s；含 M1 三头与预计算路径）、合并后新沙箱 **devshot 10 页 0 个 JS 错误**（截图核对：比较页两侧同为 0–17.0 Pa、详情页结论上移且耗时卡保留、报告页「← 工作台」、一页纸无半页空白；模板刷新路径走通）、版本号升到 **0.14.0**（`test_health` / `test_service_cli` 改读 `__version__`）。**已上线**（2026-09-24 17:40，用户授权执行 `service upgrade --drain 600 --env CUDA_VISIBLE_DEVICES=1`：预检 18 项通过 → 排空 0 s → 新进程 PID 32849、GPU 固定 1、写锁持有 → `/api/ready` 200 → 后台刷新 5 份报告模板 2.3 s；两个发布包预热 6.5 / 2.5 s；doctor 18 ✓ 2 ⚠）。

**升级注意**：① 旧令牌会话没有指纹，升级后需重新输入一次令牌（口令会话不受影响）；② 首次从 0.13 升级时旧进程不认识 `.drain.json`，`--drain` 只等计算中的任务；③ 报告模板变了，历史报告会在打开或升级时自动 UI 刷新；④ 回退到 0.13 会丢失 v2 任务在待确认阶段的预览（已完成任务不受影响）。

**v0.14.1（2026-09-24 晚，上线后第一条反馈：「之前做过的病例呢」）**：v0.14 的会话改动让升级前的令牌会话失效（旧行没有令牌指纹），而令牌模式下任务按会话 owner 隔离、又没有认领通道，重新登录后拿到新 owner，旧任务在界面上「消失」（磁盘上完好）。我先写的补丁（过期令牌会话保留 30 天只可认领 + 令牌登录带上上个会话 owner + 认领接口放开给令牌会话）被自动模式的安全分类器判为放宽会话安全而拦下，未采用。按用户随后的裁定改为**管理员视角**：服务切到**用户名登录**（`cli user add admin --admin`，users.json 存在后令牌登录自动关闭），停服务后用 `cli jobs claim --owner <旧 owner> --user admin` 把 5 例（含两例 owner 为空的命令行任务，新增 `--owner none`）全部归到 admin，再启动。口令模式下会话过期后同名重登仍是同一 owner，任务不会再丢；管理员勾选「全部用户」可只读查看其他用户的任务。测试与优化阶段大家都用 admin，后续再分配用户与视角。

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

## v0.13（2026-09-23 晚）：提示条可关可撤销、RRT / ECAP、细分色标、报告菜单四标签

用户三项要求（导师要求加 RRT、ECAP），边界经提问确认：提示条加 × 与「撤销」；RRT / ECAP 做全链路；色标四项全做；报告左侧菜单改四个标签页。

| 项 | 做法 | 代码 |
|---|---|---|
| 提示条 | 所有提示条右侧 ×；删除后的「已移入回收站」带「撤销」（只恢复这次删掉的任务，删的是当前病例则重新打开）；成功提示 8 s 后消失（带操作 15 s，鼠标停留不消失），错误提示保留到手动关闭。顺带修掉旧竞态：删除瞬间在途的详情轮询 404 会把提示条换成「任务不存在」并误报「连接中断」 | `static/app.js`（`showNotice` / `undoDelete` / `pollJob`）、`app.css` |
| RRT / ECAP | RRT = 1/[(1−2·OSI)·TAWSS]、ECAP = OSI/TAWSS（Pa⁻¹），由预测的 TAWSS / OSI 逐点计算（TAWSS 下限 0.01 Pa、1−2·OSI 下限 0.01，LV 实际不触发）。走通用额外字段通道：`summary.fields` / `cycle.fields` / `results.statistics`、沿程曲线（均值 / p90）、`points_wss.csv` 两列 `rrt_per_pa,ecap_per_pa`、VTP 两个数组、报告字段切换、统计卡、探针、一页纸一张合卡、比较两行、工作台详情两张卡；**不写入 field.npz**（它只存模型输出）。阈值：ECAP 1.4 / 2.8 / 4.2（1.4 为 AAA 文献血栓易发参考水平），RRT 5 / 10 / 20（无公认值，仅分级显示）；报告里可改 | `cycle_fields.py`（`derive_indices` / `with_derived` / `derived_display_arrays`）、`pipeline.py`、`analysis.py`、`onepager.py`、`comparison.py`、`glossary.py` |
| 旧三头任务 | UI 刷新后的旧报告在浏览器里用已嵌入的 TAWSS / OSI 数组现算 RRT / ECAP（与 Python 同公式同下限）；完整重建 `rebuild_report` 另把它们补进 summary、追加 CSV 两列（原有列逐字节不变）、重写 VTP；`service upgrade` 的待重建规则加了「有 TAWSS/OSI 缺 RRT/ECAP」，升级时自动完整重建 | `report.py`（`deriveInViewer`）、`rebuild_report.py`（`_add_derived_indices` / `_append_csv_columns`） |
| 阈值跟随字段 | 「更多显示设置」里的三个阈值输入跟随当前着色字段（WSS 仍是 `thresholds_pa`，其余字段存 `FIELD_THR`），改动即时重算面积占比、分支表、等值线、色标刻线；视图状态新增可选键 `field_thresholds` | `report.py` |
| 色标细分 | 刻度改为 1 / 2 / 2.5 / 5 取整步长（约 8 段）+ 精确的上下限 + 半格短刻度；对数色标每十倍 1-2-5；分段时刻度就是段边界（最多标 11 个）；当前字段的阈值画成横线并在左侧标数值；色标高 190 → 300 px（紧凑 140 → 210）；分段新增 10 / 16 / 20；出版级 PNG、色标 SVG、保存截图同一套刻度 | `static/report_common.js`（`colorbarTicks` / `tickLabel` / `spreadLabels`，`colorbarSVG` 的 `ticks:'fine'` / `thresholds` 为可选项，体场报告不受影响）、`report.py` |
| 报告菜单 | 九张折叠卡 → 四个标签页：查看（显示 + 统计与口径）/ 测量（测量探针、沿程、区域）/ 审阅（发现、标注；角标 = 待判定的「关注」级发现数）/ 导出（导出、预设）。每页常用项在上：查看页新增视角行与一行统计摘要（当前字段 p99 / 均值 / 阈值占比 + 「完整统计」），剖切、不透明度、时间帧、阈值等收进「更多显示设置」；导出页四个常用按钮在上，视图状态文件与导出设置折叠；多字段报告隐藏菜单里重复的字段按钮（视口上方已有切换）。程序化打开（测量模式、球形刷、横幅「详情」、视图状态 `ui.menu` / `ui.tab`）自动切到对应标签；无 WebGL 时只剩「查看」 | `report.py` |

验收：485 项测试（+15）、黄金回归 5/5（自包含）；LV_GUO_YOU 用 M1 真跑一遍（CPU 54 s）summary / 沿程 / CSV / VTP 均带 RRT、ECAP，field.npz 无派生数组；沙箱完整重建与浏览器现算一致（RRT 均值 5.41 / ECAP 0.526 Pa⁻¹；RRT > 5 占 49%、> 10 占 11%，ECAP > 1.4 占 5%）；devshot 截图：报告四标签 / RRT / ECAP 16 段 / 紧凑布局、工作台详情卡、一页纸、删除 → 撤销链路。**RRT / ECAP 的误差未单独对照 CFD 验证**（继承 TAWSS / OSI，低 TAWSS 处放大）。

**v0.13.1（2026-09-24，试用反馈：左侧详情看不全、不能右划）**：macOS 系统字体较宽时，统计卡里表格按不折行的自然宽度排版，把卡片撑出菜单（菜单横向裁切），右侧列和提示文字被截掉。改为菜单内容一律不超过菜单宽度（卡片网格列 `minmax(0,1fr)`），统计与探针表格各自放进可左右滑动的框（首列分支名固定，右侧阴影提示还有列），提示文字自动折行；486 测试，已 `service upgrade` 上线。

## v0.12.2（2026-09-23）：算子提速——结果不变，B 段快 1.4–2.9 倍

用户问「算子上有没有能提速、又不影响精度的地方」。先用 cProfile 实测 LV_GUO_YOU（2.3 万面）与 LIU_YU_MING（27.3 万面）三类发布包的 B 段，再逐个改写热点；每一项都用「与原实现逐位相同」或「差异不超过原本的运行间抖动」验收。

| 算子 | 做法 | 验收 | 耗时 |
|---|---|---|---|
| 表面重采样 `wss_features.sampling`（体素二分） | `VoxelSearch`：KD 最近邻间距并行查询、各轴偏移只算一次、二分时只数占用体素个数（按体素尺寸记忆、跨三轮校准复用）、最终分组只做一次 | 采样点与原实现逐位相同（LV、LIU）；特征库等价测试 5/5 | LV 14.8 → 3.5 s，LIU 18.3 → 4.9 s |
| 主曲率 `wss_features.curvature`（Monge 拟合） | 1024 点一块、线程池并行、设计矩阵原位填充（替代 `np.stack` 交错拷贝） | 全部输出逐位相同 | k=128：LV 3.0 → 0.3 s |
| 缠绕数内判 `wss_features.cloud.winding_number` | 每次 8 行、每线程预分配 float32 缓冲、`out=` 原位运算、线程池并行（每个元素同样的 float32 运算顺序、每行单独求和） | 逐位相同 | 6000 查询：14.4 → 0.49 s；体场特征 LV 21–38 → 8.9 s |
| 集成推理 kNN（`knn_memo.py`） | 集成成员在同一病例上建同一张近邻图（`torch_geometric.nn.knn`，确定性核）；输入逐元素相等（`torch.equal`）时复用结果，训练代码不改、只在部署进程的集成循环内生效（`WSS_DEPLOY_KNN_MEMO=0` 关闭） | 15 次调用逐次确定；复用 14/15；与不复用相比最大差 5.0e-6 Pa，原本连跑两次就差 3.95e-6 Pa（GPU 散射累加的正常抖动），p99 到小数点后 6 位相同 | 单模型 2.6 → 1.1 s（GPU 被占满时） |
| 流线内判（`streamlines.ball_certified_inside`） | 内部样本到封闭表面的距离构成「必在腔内」的球；落在球里的查询点不再调用 VTK，其余仍由 VTK 判定 | 流线逐条逐点完全相同（LV 527 条、LIU 524 条） | LV 11.0 → 3.3 s，LIU 20.0 → 6.0 s |
| 发布包加载 | 缓存默认能放下全部发布包；服务启动后后台预加载（`WSS_DEPLOY_PRELOAD=0` 关闭） | — | 首例 / 切换发布包省 4–13 s |

端到端 B 段（副本实测，GPU 同时被其他训练占满）：LV · M1 三头 38.3 → 13.3 s；LV · X5D 27.0 → 19.5 s（CPU 段 18.8 → 7.3 s，推理受 GPU 争用影响）；LV · 体场约 48 → 21.9 s；LIU · 体场约 150 → 53.6 s。全套 456 项测试、特征库与训练实现等价 5/5、黄金回归 5/5（本身也从 210 s 降到 130 s）。已 `service upgrade` 上线。

**同轮界面（契约 §21）**：
- **分阶段进度与剩余时间**：`eta.py` 按族、按面片数估计各阶段耗时；只用提速后（特征库源码哈希一致）的同族历史，不足 3 例时用默认表插值。工作台显示分阶段进度条和「预计还需约 N 秒」，确认出口页显示「确认后约 N 秒出结果」，排队时显示「前面 K 个，约 N 秒后开始」。沙箱实测 B 段估计偏差 −8% 到 +5%。
- **自动标注防重叠**：共享库 `declutterLabels` 按优先级错开标签，移动后的标签画引线回锚点，窄屏时隐藏低优先级标签；页面和导出图都生效。
- 以上 479 项测试通过、黄金回归 5/5，已上线。
- 已知：纯 CPU 模式下前几次推理每个模型要 20–27 s，之后稳定在约 12.7 s（冷启动）；GPU 上只有每个进程第一次多约 2 s。

**试过但没采用**：
- 形态测量按站位多线程：结果相同但更慢（13.8 → 21.5 s，轮廓串环是纯 Python，被 GIL 卡住），维持串行。
- VTK 逐点内判多线程或批量模式：VTK 持有 GIL，多线程反而更慢，批量模式也一样快。
- Taubin 平滑的权重矩阵构造：能做到逐位相同，但只快约 15%，常规病例这一步本来就只要 0.06 s。
- **大网格中心线先抽稀**（A 段的最大耗时，LIU 65 s）：抽到 3 万面时 VMTK 只要 3.2 s，但结果会变——分叉点移动约 9 mm（主动脉 224 → 215 mm）、瘤囊长度 110 → 103 mm、逐点预测 R² 0.985。**不满足「不影响精度」**，未上线。候选做法是只对 > 10 万面的网格启用，并先做 34 例 CFD 复核。

## v0.12.1（2026-09-23）：截面看得清（试用反馈 7）+ 报告刷新代码一致性保护

用户反馈：体场报告瘤囊截面整片深蓝、看不出变化（截面与三维都按全场范围 0.0018–1.57 m/s 着色，最大值在髂支，瘤囊 0.01–0.1 m/s 只占色标底部约 5%）。体场报告新增：

- **截面色标「本截面（默认）/ 全局 / 手动」**：本截面 = 该截面样本的稳健 p2–p98（不含壁面补全值，样本 < 8 回退全局）；作用于截面平面图、放大图与导出、截面模式下三维切片点与右上图例（注明全局范围）；系列拼图用各站合并范围、共用一个色标。LV_GUO_YOU 瘤囊 s≈195 mm 截面：0.0279–0.0872 m/s，射流核心与向心汇聚清楚可见。
- **对数色标（仅速度）**：下限 = max(范围下限, 上限/200, 1e-3 m/s)，对点云、流线、截面、图例、导出同时生效。
- **面内流向箭头**（速度默认开）：样本速度投影到截面 (u,v)，网格抽样（面板 ≤ 100、放大 ≤ 289），长度按本截面面内速度 p95 归一化。
- **穿面速度（带正负）**：v·n̂，蓝白红对称色标；中心线定位时法向沿弧长增大方向（两例 7 条分支平均 v·t 全为正已核实）→ 标注「顺流为正、负值 = 回流」；LIU 瘤囊可见明显回流区。
- 视图状态新键 `slice.color_range / slice.quantity / slice.arrows` 与顶层 `log`，旧状态按原全局口径回放。统计卡与 CSV 数值不变。

**报告刷新代码一致性保护**：服务进程启动时记录报告模板 Python 模块（`report.py / volume_report.py / rebuild_report.py`）的指纹；运行中若磁盘上的模块已改，打开报告时不再用内存里的旧模板做自动刷新（否则会把旧模板与新查看器脚本拼在一起并标记为最新），继续提供原报告并记日志；`/api/health` 带 `report_code_changed`，`cli doctor` 检出「服务启动后有 N 个 Python 模块更新」并提示 `service upgrade`。

测试 451 项通过；已 `service upgrade` 上线（令牌不变，7 份报告刷新）。

## v0.12（2026-09-23）：三方面优化——日常操作、医生功能、界面

用户要求从「日常使用与操作方便 / 功能补充方便医生 / UI 设计」三方面优化并授权连夜做完。方案与审计证据（无头浏览器真实截图）：[三方面优化方案与落地](../docs/02-推进与变更/05-部署工具/_archive/WSS_部署工具_三方面优化方案与落地_2026-09-23.md)；契约 `ANALYSIS_CONTRACT.md` §19（实现记录 §20）。五路并行（W1 内容与分析 / W2 运维 / W3 工作台 / W4 壁面报告 / W5 体场报告），主会话先落地共享函数、机构模板模块与截图工具，合并后集成验收。**全套 446 项测试通过（1 项真浏览器测试需 `WSS_DEPLOY_DEVSHOT=1`）、黄金回归 5/5（预测数值逐位不变）、16 页集成截图 0 个 JS 错误、新前端在旧后端下平稳降级。**

**正式服务尚未切换**（重启正式服务的命令被自动模式拦截，留给用户确认）。切换只需一条命令（约 10 s，令牌与已登录会话保持，队列为空时执行）：

```bash
cd /public/newhome/cy/Digital_twin/GNN && PYTHONPATH=. ~/.conda/envs/GNN/bin/python -m wss_deploy.cli service upgrade
# = 接管手工启动的 8765 进程（沿用参数、环境、令牌 → service.json / .service_token）→ 停止 → 完整重建缺周期量发现的 M1 任务
#   → 刷新全部过期报告 → 用新代码启动并等健康检查。之后日常：service status / logs / restart / upgrade、doctor
```

### ① 日常使用与操作

| 项 | 做了什么 | 落点 |
|---|---|---|
| 服务管理 | `cli service start / stop / restart / upgrade / status / logs / token`：PID 文件、`service.json`、令牌文件 `.service_token`（0600）；能识别并接管手工启动、没有 PID 文件的进程（读取其 argv 与环境）；只向命令行匹配 `wss_deploy.cli serve` 的进程发信号（防 PID 复用）；启动后等 `/api/health`，失败打印日志尾部；新服务收到 SIGTERM 正常退出（任务标记中断、VMTK 进程组一起结束） | `service.py`、`cli.py` |
| 一条命令升级 | `service upgrade`：停服务 → 完整重建需要新分析结果的任务（`pending_rebuilds`：有周期量却没有 v0.12 周期发现的任务；也可 `--rebuild <id>`）→ 刷新全部过期报告 → 启动；重建失败也会照常启动 | `service.upgrade` |
| 环境自检 | `cli doctor [--json]`：torch/CUDA、VMTK 解释器与 vessel_geom、每个发布包合同与 MANIFEST 逐文件、参考侧车、特征合同哈希、术语表同步、任务目录可写、磁盘、服务版本与健康、过期报告数、登录方式；✓/⚠/✗ + 一句建议 | `doctor.py` |
| 健康接口 | `GET /api/health`：无会话只给 `{ok, version}`；有会话加运行时长、队列、计算线程、GPU、磁盘、默认发布包、过期报告数；工作台连接状态点悬停显示 | `server.py` |
| 报告自动刷新 | 报告模板指纹（`report.py`、`volume_report.py`、共享库、查看器、three.js、术语表），打开报告时发现过期就按需 UI-only 刷新（约 0.3 s，数据与嵌入 JSON 原样），`report_ui.json` 记录；`cli reports refresh [--job/--all] [--check]`。以后改报告界面**不必再手工重建 + 重启** | `report_freshness.py` |
| 命令行提交 | `cli submit <stl…>` 经 HTTP 批量提交到运行中的服务（每批 ≤ 20，打印 `#job=<id>` 链接）；`cli jobs list` | `service.ServiceClient` |
| 日志 | `serve --log-file` 轮转 20 MB × 5，时间带时区；`service` 启动的服务写 `server.log`（stdout 进 `server.console.log`） | `server.configure_logging` |
| 工作台操作 | 地址栏 `#job=<id>` 深链接（刷新、前进后退恢复）；拖放 STL 到页面任意处上传；快捷键 `/ N J K ↑↓ Enter/O G ?`；列表「需要处理 → 计算中 → 待审阅 → 已审阅（折叠）」；确认出口 / 审阅通过后「处理下一例 / 下一例待审阅」；友好时间（今天 17:37 / 昨天 / 9月20日，悬停精确到秒，统一浏览器时区）；去掉多余「#」（空标签数组）；行内大红「删除」改「⋯」菜单 | `app.js`、`workbench_core.js` |
| 报告快捷键 | 两报告 `1–6` 标准视角、`0/R` 复位、`S` 截图、`P` 探针、`L` 自动标注、`?` 帮助；壁面 `F` 切 WSS / TAWSS / OSI；体场 `V/B` 速度 / 压力、`T` 页签、`X` 点选截面（截面方向键不冲突） | `report.py`、`volume_viewer.js` |
| 界面自查工具 | `devshot.py`：系统 Firefox + llvmpipe 软件 WebGL 无头截图（Marionette 直连，无需 selenium），收集页面 JS 错误；`devshot sandbox` 复制任务起回环沙箱；`devshot suite` 一条命令截 16 个标准页面 + `index.json` | `devshot.py` |

### ② 医生功能

| 项 | 做了什么 | 落点 |
|---|---|---|
| 周期量全链路（修正 M1 下游） | 一页纸不再对 M1 写「没有 TAWSS、OSI」；「五模型」按实际模型数（5 个时逐字不变，保黄金比对）；自动结论加周期量句与新免责句；发现列表新增「滞留区」（TAWSS < 0.4 ∧ OSI > 0.1 连通簇，前 3，值 = 面积 cm²）与「高 OSI 区」（OSI > 0.3，前 3），追加在原条目后；沿程曲线加 TAWSS / OSI；汇总表 8 列；并排比较加周期量标量；工作台详情周期量卡片；任务记录带 `cycle` 与 `findings_top`（与一页纸同一挑选规则：每类先取一条） | `analysis.py`、`narrative.py`、`quality.py`、`onepager.py`、`export_table.py`、`comparison.py`、`jobs.py`、`app.js` |
| M1 几何参考侧车 | M1 与 X5D 同为 v5.1 train136 → 生成几何-only `reference.json`（29 条，人群分位 unknown 并注明），几何越界检查对 M1 生效；发布包指纹不变 | `build_reference_profiles.py --population-note` |
| 患者随访时间线 | `GET /api/patients/<id>/timeline`：一次扫描 = 一个输入几何（多发布包运行合并）；几何量跨发布包可比、模型量同发布包连线；最大直径与瘤体体积年增长率（两次都有扫描日期且相隔 ≥ 30 天才算）；工作台「患者时间线」卡片（小折线 + 表 + 勾两次扫描并排比较）；一页纸附「随访变化」 | `timeline.py`、`server.py`、`app.js`、`onepager.py` |
| 一页纸重排 | 第 1 页 = 页眉（机构模板）→ 身份一行 → 结论 → 关键数字（M1 加三卡）→ 配图 → 瘤体形态 → 随访 → 重点发现前 5 → 签字栏；附录另起一页：合并后的分支表、发现详情、输入与参考范围、形态方法、可信区域、按字段生成的局限性、耗时、身份哈希、只列本页用到的术语（左对齐）。M1 例第 1 页实测约 272 mm < A4 可用 277 mm | `onepager.py` |
| 机构模板 | `<jobs_root>/report_template.json`（机构 / 科室 / 标题 / 页脚声明 / 签字栏 / 术语范围 / 附录开关）；`GET/PUT /api/report-template`（回环或管理员可写）；`cli template show / set`；工作台「设置 → 报告模板」 | `report_template.py`、`server.py`、`cli.py`、`app.js` |
| 告警醒目 | 几何越界 / 人群需复核 / 质量非良好时，详情页与两份报告顶部黄色横幅，一句话 + 「详情」 | `app.js`、`report.py`、`volume_viewer.js` |
| 审阅闭环 | 详情页头部「审阅签字」按钮（对话框含核对清单）、「重点发现」前 3 条 | `app.js` |

### ③ 界面设计

| 项 | 做了什么 |
|---|---|
| 工作台 | 大标题并入顶栏（新建预测、连接状态、设置）；列表行：病例名、结果类型芯片、状态 / 审阅、患者与扫描、友好时间、⌀ 最大直径；空状态改「今日概览」（需要处理 / 计算中 / 待审阅 / 近 7 天完成，可点击筛选；最近完成；服务状态；快捷键） |
| 详情页 | 头部动作区（打开三维报告 / 一页纸 / 审阅签字 / ⋯）；形态数值主副两行不再断行；可靠性压成一句可展开；缩略图出口胶囊标签（旧任务没有预览网格时画中心线示意） |
| 确认出口 | 1440×900 一屏可见整段血管与确认按钮：任务卡压成一行、三维高度随视口自适应并用 `fitView` 框住整段血管、右侧操作栏 sticky；原因改人话可展开；端点胶囊标签 |
| 报告 | 默认 = 解剖前视（入口朝上、患者左侧在屏幕右侧）并撑满视口（`fitView`，纵向约 85%），标准视角同样撑满；OrbitControls 按解剖 up 重建（旋转轴正确）；「?」修回圆形（`min-height:34px` 通用按钮样式覆盖）；页脚只留审阅 · 发布包短名 · 生成时间 +「技术信息」弹层；「收缩期峰值帧（约 0.21 s）」替代 `peak_systole / step 1162`；数值统一 `formatValue`（不再出现 1.77e-3）；**默认配色恢复彩虹**（用户 09-20 偏好；v0.11.2 改的 Viridis 保留可选）；壁面报告视口上方「峰值 WSS / TAWSS / OSI」分段按钮、滞留区斜纹叠加、周期量统计卡；体场「壁面压力」页签在速度模式下被误禁用的 bug 已修 |

共享库新函数（`report_common.js` §19.1）：`fitView / formatValue / localTime / friendlyTime / installShortcuts / shortcutMatches / shortcutRows`。旧任务兼容：09-17 的两个旧任务（无 `fields`、无预览网格、无计算时长）按「壁面 WSS」显示、缩略图画中心线示意、不显示「计算 0 秒」。

已知边界：确认页三维仍用世界坐标方向（出口命名确认前没有解剖坐标架）；自动标注在髂分叉附近仍会重叠；`cli submit` 在令牌 / 回环模式下提交的任务属于命令行会话，需网页「认领」或启用用户名登录；「近 7 天完成」基于最近 100 个任务；一页纸页码为静态「第 1 页 / 附录」（Firefox 不支持 `@page` 计数）；`releaseShort / 横幅文案 / createdIso` 两份报告各有一份局部实现，待上移共享库。

## v0.11.3（2026-09-23）：病例详情聚焦与血管形态预览

- 完成病例详情页首屏只保留病例标题、血管形态缩略图和核心指标；质量、参考范围、审阅签字、下载与重跑等次级内容收进默认折叠的「质量、参考范围与审阅」区域。
- 血管缩略图复用 `/api/jobs/<id>/geometry` 的同一份预览网格、中心线和端点数据，优先按中心线 PCA 平面固定为“入口朝上、分支向下”，再按确认后的左右命名校正镜像；四个出口使用独立颜色并显示中文标签，旧任务没有二维平面时按入口到分支方向做三维语义回退。支持 WebGL 时优先显示可拖动旋转的轻量三维预览，不支持时自动回退到二维示意；点击缩略图可打开完整三维报告。缩略图不替代报告中的完整可交互视图。
- 回归验证：工作台与报告相关测试 `148 passed`，并通过 JavaScript / Python 语法检查和 `git diff --check`。

## v0.11.2（2026-09-22）：病例优先工作台与报告界面

- 工作台首屏先显示病例与历史；「新建预测」打开独立上传对话框，批量低频操作收进「更多操作」，报告 / 一页纸入口移到结果标题旁。窄屏选中病例会定位到详情，并提供返回列表操作。
- 报告页统一主色和字号，常用的显示、发现、测量、导出与视图菜单前置；高级阈值、单位、分支和导出设置默认折叠。默认连续色图为 Viridis，原有彩虹、Turbo、蓝白红和已保存偏好继续可用；三头发布包显示为「WSS + TAWSS + OSI」。
- `rebuild_report.py --ui-only` 只替换历史报告的呈现模板和查看器，原样保留内嵌元数据、预测数组、`summary.json` 及其它清单字段，只更新报告文件的清单哈希。已用于现有已完成任务。
- 回归验证：`143 passed`，并通过 Python / JavaScript 语法检查和 `git diff --check`。

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

按[下一轮功能与优化方案](../docs/02-推进与变更/05-部署工具/_archive/WSS_部署工具_下一轮功能与优化方案_2026-09-21.md) §2 的 18 项常用功能开发；四路并行（后端 / 工作台 / 报告共享库 + 壁面报告 / 体场报告），契约在 `ANALYSIS_CONTRACT.md` §10–§14。合并后单元测试 262 通过（+ 黄金回归门 1 项），`node --check` 全过，黄金回归 5/5（预测数值逐位不变），8 个历史报告已重建，服务已重启。真实浏览器渲染（导图分辨率 / 透明底 / 通知弹窗）待用户验收。

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

按[架构评审](../docs/02-推进与变更/05-部署工具/_archive/WSS_部署工具_架构评审与优化路线_2026-09-20.md)的 R0/R1 落地，数值验收以 `python -m wss_deploy.regress` 为准（5 个已完成任务重跑 B 段与 `outputs/wss_deploy_golden/20260920_baseline/` 逐键逐元素一致）。

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
