# WSS V6 新几何候选验收

> **2026-09-15 算法修复与全站审查：** [当前交互审查台](算法优化_20260915/index.html) · [只需确认的两例分叉](算法优化_20260915/remaining_bifurcation_review.html) · [算法/结果说明](算法优化_20260915/README.md)。新候选为独立目录，170例重建和独立审计完成；旧v1.3页面保留作前后对照，不再作为当前人工队列。

> **2026-09-11 最新更新：**WANG_KUI_WU 的 seg3/seg5 解剖左右标签已按用户确认修正，全部172例新增“分叉快速定位”。[定位结果与10处未确定清单](bifurcation_review.html) · [本地下载、使用和更新说明](UPDATE_20260911.md)。下文v1.3截图及先前审计描述保留作历史记录；当前动态病例页以此次更新为准。未扩大整例/全队列几何批准范围。

打开 [全队列交互索引](index.html)。页面仅用于几何候选验收；未批准新的输入特征、解剖命名或 M6 固定毫米邻域进入训练。页面和生成器只读取几何，不读取 WSS、速度或压力标签。

截至2026-09-10的发布为 **v1.3、172/172 例候选详页、0 例预览替代**；[全页审计通过](page_audit.json)，四例实际浏览器检查通过。当时优先复核为 **118 例**：原质量提示 17 例、面积分歧 27 例 / 39 个截面、P6 同源变化 >20% 的 106 例 / 326 个扰动变体，三类有重叠。20% 用于人工检查排序，不是输入 mask 或训练验收阈值；±1 mm 同时改变实际解剖位置，大变化不能一概断言为算法失败。

页面状态由 [render_manifest.json](render_manifest.json) 记录。`preview_only=true` 仅表示既有 V5 HDF5 的几何预览，历史预览图片不属于本次最终验证；只有源 `geometry.npz` 与 `report.json` 均存在时，才渲染候选截面与稳定性详页。缺失不填为零、无效截面不强填面积。

## 审阅与核验入口

- [优先审阅清单](priority_review.html)：同时通过两套局部门控却明显分歧的截面、P6 极敏感变体，与原质量提示合并排序。门控通过不等于几何已完成验收；这些排序不自动修改点云输入 mask。
- [v1.3 对历史 top6 的门控复核](priority_stability_v1.3.html)：其中 rank3 的 +1 mm 切面与同段中心线在 s≈94.173 和 136.934 mm 两处相交，当前扰动门控为 false；其余 5 个仍需复核。这是历史案例复核，不是 v1.3 重新排名的 top6。真实连续交点用 × / 菱形显示，不能把附近采样点数量当作穿越次数。
- [v1.2 历史实际扰动前后轮廓](priority_stability.html)：六例保留前后选定轮廓、全部交线、真实 3D 位置与原始门控记录。rank1 是最大面积跳变，rank3 用于排查同段远处再次穿过切平面的机制。这里的旧版本状态不冒充最终版本的门控结果。
- [全队列页面核验](page_audit.json)：核查 172 例身份、最终 source schema/config hash、页面状态与来源报告指标一致。预期版本默认取最终候选 `manifest.json`，也可给 `audit_pages.py --expected-schema ...`。

全队列渲染和索引完成后，运行 `publish_priority_review.py --sensitivity <最终review_priority_stability.json>` 生成优先页并置顶索引。它要求分歧清单、敏感清单和所有已渲染页面的候选版本相同，避免旧统计混入新版本入口；不修改原算法或候选数据。

本次实图：[LOU_YANG 整体](candidate_v1.3_LOU_YANG_overview.png)、[WANG_KUI_WU 拓扑语义](candidate_v1.3_WANG_KUI_WU_topology.png)、[GAO_FENG_SHAN 沿程曲线](candidate_v1.3_GAO_FENG_SHAN_profiles.png)、[YIN_YU_RONG 第 74 站](candidate_v1.3_YIN_YU_RONG_section74.png)。整体默认视角能显示五个开口，截图未额外改变相机。YIN 第 74 站在 v1.3 仍为两源 valid，但面积约 405.24 vs 88.42 mm²，说明现有门控仍不足以确定截面语义。[v1.3 历史 rank3 复核图](diagnostic_v1.3_stability_rank3.png) 显示新门控实际拦截的轮廓。

[索引与优先页浏览器核验](browser_smoke_priority_index_v1.3.json) 检查首页置顶优先提示、172 个唯一病例链接、优先页跳转、39 条截面记录、四张图加载、历史/当前诊断链接与病例搜索；结果通过。[首页实际截图](priority_index_v1.3.png) 可直接预览当前入口。

## 查看顺序

2026-09-10 交互更新：在“截面逐站检查”页使用顶部固定的站点滑块、前/后一站按钮，或在“站点编号”输入数字后按回车。左右两图及状态栏会同步更新；编号从0开始。右侧图内拖动只平移视图，表格横向滚动只查看字段，都不会切换站点。第0站通常位于入口排除区，灰色轮廓表示无效候选，不表示零面积。几何候选仍为v1.3，交互更新保留原嵌入数据；已下载的旧HTML需要重新获取新版。

CIA = Common Iliac Artery，髂总动脉：从主动脉分叉至同侧髂内/髂外分叉之间的一段。当前图上的分叉点与CIA端点继承中心线图的分段位置，不能直接视作独立定位的壁面分流嵴。

1. **整体与拓扑**：透明解剖壁面、七段中心线、五个开口、三个分叉、CIA 起止与拓扑树。颜色仅编码 segment ID；画面左右与 CFD 区域名不等于解剖左右已人工确认。
2. **截面逐站检查**：切换全部站点，叠加全部交线、面几何选定轮廓、点云候选轮廓。检查多闭环、开轮廓、偏心、中心线出界、分叉歧义及 invalid 原因。
3. **沿程几何曲线**：A、Req = √(A/π)、原 Rmis、dlog(A)/ds、圆度及中心线偏心度（截面质心至中心线距离 / Req）；偏心度不是椭圆轴长定义的 eccentricity。实线是面几何参考，点线是点云候选，无效处断开。
4. **壁面映射与 mask**：分别查看 segment、映射 A、有效性与歧义。可视化抽样不参与数值计算。
5. **扰动与密度稳定性**：±1 mm 位移、±5° 倾角、点云 full/half。点云/面几何差异与点云自身密度敏感性分别解释，不把两者混称密度误差。
6. **M6 邻域覆盖候选**：同一随机 support 比较归一化半径 `0.015 / 0.03 / 0.06` 和固定 `2.5 / 5 / 10 mm` 候选。固定半径值可在 CLI 改；当前未训练、未确定最终值。

## 生成与更新

生成器：[wss_v5/render_geometry_candidate.py](../../../wss_v5/render_geometry_candidate.py)。默认候选源是 `outputs/wss_v6_geometry_candidate_20260909/cases/<ID>/{geometry.npz,report.json}`，默认输出本目录。

```bash
/public/newhome/cy/.conda/envs/GNN/bin/python -m wss_v5.render_geometry_candidate
```

只更新某几例，使用 `--cases AG/fast/CHEN_SHI_MING AAA/ruputer/WANG_KUI_WU`。候选尚未生成时，允许明确标记的旧几何预览：

```bash
/public/newhome/cy/.conda/envs/GNN/bin/python -m wss_v5.render_geometry_candidate \
  --preview-existing --cases AG/fast/CHEN_SHI_MING AAA/ruputer/WANG_KUI_WU ILO/GONG_HAI_ZENG-1/before
```

CPU 并行渲染支持 `--shards N --shard-index i`；各 shard 只写独立病例页面与 summary，全部完成后运行 `--index-only` 统一更新索引。Slurm 由主任务统一提交。本模块不提交计算或训练任务。

## 浏览器与依赖

Python 仅用 NumPy、SciPy、h5py；无须安装 Python Plotly。页面使用本地 [Plotly 2.35.2](assets/plotly-2.35.2.min.js)，下载源 `https://cdn.plot.ly/plotly-2.35.2.min.js`，SHA-256 在 render manifest 中记录。完整绘图数据以 gzip 嵌入 HTML，通过浏览器原生 `DecompressionStream` 解压（现代 Chrome/Edge/Firefox/Safari 支持）；打开页面不需服务器或网络，复制目录时须保留 `assets/` 和 `cases/`。

[check_browser.mjs](check_browser.mjs) 可连接本地 Chromium CDP，依次切换六个页签，检查脚本异常、首尾截面并截屏。它需要 Node.js 内置 WebSocket 与已启动的 Chromium 调试端口；只检查页面，不改原始数据。

```bash
node check_browser.mjs file:///absolute/path/cases/CASE.html screenshot.png 9231
```

附加第四参数指定截屏页签（如 `sections:74` / `overview` / `profiles`），第五参数可要求候选版本（如 `v1.3`）。浏览器检查包括实际候选/有效轮廓/交线非空、invalid 轮廓灰色、选择器一致、映射颜色无非法值。全量渲染后运行 [audit_pages.py](audit_pages.py)，核对全部 172 例身份、页面状态和指标与最终来源 JSON 一致，并写入 `page_audit.json`。四例六页签检查在 `browser_smoke_v1.3_*.json`；[历史六例 v1.3 复核浏览器检查](browser_smoke_stability_recheck_v1.3.json) 核对真实交点与 mask。

## 边界

- surface mesh 限制显示到最多 60,000 个面片、壁面映射最多 15,000 点；源算法仍使用其完整数据。稀疏显示可能有视觉孔隙，不作为截面计算证据。
- M6 使用 seed 1234 的最多 5,000 个随机壁面 support 作同点比较，邻居数含自身；训练层上限为 16。它不是正式训练采样重放，表中原始邻居统计未裁剪，箱线图仅为显示裁剪到 64。
- 当前坐标保留 Fluent mm，不旋转推定患者左右。`anatomy_label` 原样作为候选语义展示，人工确认状态不由 renderer 改写。
- 点云截面必须与面几何参考分开验收。`pointcloud_portability_proven=false` 不能因图片形状合理而自动改为 true。
- 截面与壁面图按各自参考/点云 mask 显示；无效轮廓保留灰色用于诊断，可信面积留空。P6 曲线按 `comparison_valid` 控制可信对比，旧 schema 才回退 `pipeline_valid`；`raw_contour_valid` 仅说明原交线可闭合，各层状态分别保留。
