# WSS 最小化路线代码修改与实验推进记录

> 用途：单独记录 `pipeline_wss_min/` 这条 WSS-only 最小化数据线的代码、坐标 QA、图件和实验推进。
> V3P / 训练主线 / 通用代码修改记录见：[代码修改与实验推进记录](代码修改与实验推进记录.md)。
> 当前执行入口：[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) / [第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md) / [WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md) / [BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)。

## 2026-07-16｜AG/AAA v4 E2-GLOBAL 完训公平评估 ✅DONE｜v4 No-Go｜AAA 小幅增益但高尾未解

**本次主要修改**：只读收敛 Job `9138` / `9140` 的400 epoch、best/last 全云 train+test15 评估和 best PostView；用预注册的 `ckpt_best(train_loss)` 作主结果，`last` 仅作敏感性。公平对照仅使用旧 E2 已保存预测纯后处得到的 common-test15，没有重训、重推理或用 test 反选 checkpoint。

**对应产物**：`training_wss_min/runs/pointnet_v4/outputs/{ag_v4_e2_global_fps2000,ag_aaa_v4_e2_global_fps2000}/`；结果真源 `data_wss_min/pipeline_reports/v4_cutover_20260715_1921/v4_e2_global_result_analysis.json`；旧锚点为 `training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_global/eval/ckpt_best/common_test15_from_saved_predictions/` 。

**推进到实验步骤**：Job `9138` / `9140` / 延迟监控 `9141` 均 `COMPLETED (0:0)`；AG 用时 `01:18:25`、best epoch `364`、loss `0.1703`，混合用时 `06:14:49`、best epoch `395`、loss `0.1353`。两套 best/last 评估齐全，PostView 均为15/15病例、15/15 manifest bundle、60份 VTP，mapping coverage 最小100%；日志无 NaN/OOM/Traceback/路径或 frame 失败。

**当前状态判断**：旧 v3 E2 common-test15 锚点为 field/case-mean `R²=0.2597/0.1928`、MAE/RMSE `2.754/4.902 Pa`；AG-v4 best 为 `0.1852/0.1465`、`2.847/5.143 Pa`，同病例口径明显回退，v4 单独重跑判 **No-Go**。混合 best 为 `0.2086/0.1572`、`2.772/5.068 Pa`，相对 AG-v4 的 field/case-mean R² 增加 `+0.0234/+0.0106`，15例中8例改善，Spearman `0.6788→0.7041`、top10 IoU `0.1314→0.1575`；但 high-WSS R² `-1.8279→-1.8473`，p99 幅值比 `0.4873→0.4263`、动态范围比 `0.2147→0.1540`，且仍未追平旧锚点。AAA 密网格还贡献了混合 target stats 约 `78.8%` 的全云点，使 log-WSS 统计从 `1.122±1.096` 变为 `0.553±1.373`，这是高幅值压缩的一个待单变量验证混杂因素。结论是 AAA 带来小幅整体/排序/热点定位增益，但高 WSS 幅值压缩更重；`LI_SHU_KUN` 在混合后病例 R² 下降约 `0.293`，下一步优先做病例/域平衡统计、高尾加权与困难病例诊断，不因 last 的小幅 test 改善改变主结论，也不优先切 random-5000。ILO 未处理；旧快照/staging 仍保留，待用户验收并指定归档位置后再清理。

## 2026-07-16｜AG/AAA v4 发布、数值门禁与 E2-GLOBAL GPU 提交 ✅DONE｜后续状态见上条

**数据发布**：发布前复核旧活动 AG 84 bundle + 84 report 共168个 SHA-256 全部匹配终签清单；事务切换后活动 AG 为76例且全部 `stl_landmarks_v4`，`WANG_DENG_FENG` 缺席、`LI_ZHEN_SHAN` 存在。旧84例完整保存在 `data_wss_min/_snapshots/AG_legacy_v3_20260716_signedoff/`，原77例 staging 未删除，`promotion_record.json` 已提交。`AAA/unruputer/HAN_JIAN_FU` 的旧 bundle/report 已做病例级 SHA-256 快照，修正版已原子换入；从原始81个时间步复核 82417 节点，step1120 精确坐标重映射后节点ID、坐标、WSS、pressure 逐点误差均为0。

**训练门禁与协议**：几何终签真源仍为 AG76/AAA63。对139个签核候选扫描 frame、shape、节点、NaN/Inf、负值、全零、非正比例、peak/all 分位数、时间突峰、入口波形、压力和哈希，hard failure=0。派生训练白名单额外排除 AAA 六例：既有 denylist `CHEN_FU`、`SU_KAI_LI`、`ZHANG_GUI_HUA`，以及入口波形量级异常 `ZHANG_ZAO_SHUAN`、`GUO_YU_YING`、`WANG_SHUN_WEN`；AAA 入训57。高 WSS 稀疏尾部病例保留观察，pressure gauge offset 只记录不自动排除。新 split 为 AG `61/0/15`、混合 `118/0/15`（AG61+AAA57），统计只写 `data_wss_min/fold_stats/v4/`。旧 E2 best 已从保存的逐点预测纯后处理重汇总 common-test15，没有重新推理/训练。

**代码/验证**：`DataConfig` 新增向后兼容的显式 `data_root`/`required_frame_version`，split loader 同时支持旧 AG 短 ID 与 canonical AG/AAA ID，并拒绝非法 ID、重复和 partition 泄漏；train/evaluate/PostView 传递显式路径与 frame。AG promotion 增加切换后验证/记录失败自动回滚测试。base pytest 的 cutover 11/11 通过；base 环境因没有 `torch_geometric` 不能收集训练测试，GNN 环境 direct unittest 42/42 通过；Python 编译、Slurm `bash -n`、全量加载、统计病例哈希与 RTX4090 AMP 前后向通过。

**GPU 作业**：两份 E2-GLOBAL/FPS-2000 配置已独立提交：AG-v4 Job `9138` 已连续监控10.18分钟，日志确认 train61、RTX4090、推进至 epoch50，当前/期间最优 loss=`0.3539/0.3333`，无 NaN/OOM/Traceback/path/frame 错误；AG+AAA-v4 Job `9140` 已提交，等待 GPU 资源，Slurm 自行调度。初次 pending Job `9139` 在观察名单路径勘误后、尚未启动前安全取消并由重新预检通过的 `9140` 替代。由于 `9140` 尚未得到 GPU，另挂只读监控 Job `9141`（`after:9140+10`），在其实际启动10分钟后自动核对 CUDA、train118、epoch推进与异常关键字。正式作业均配置训练后自动运行 best/last 全云 train+test15 评估及 best PostView。本轮已启动训练但未等待最终结果，也未分析临时指标；ILO 未处理。

**保留/清理边界**：本轮保留旧 AG84 快照、AG77 staging、AAA/HAN 病例快照与 fixes staging。只有次日结果验收并指定归档位置后，才允许按保留清单打包、复核归档 SHA 再删除旧路径。

## 2026-07-16｜AG/AAA v4 人工终签、最终白名单与发布合同 ✅READY｜未切换

**本次主要修改**：固化用户逐图审核结论：排除 `AAA/unruputer/CAO_DIAN_HE`、`AAA/ruputer/LIU_YU_MING`、`AG/slow/WANG_DENG_FENG`；`AG/fast/LI_ZHEN_SHAN` 的 `CROP` 明确放行，其余软标记病例全部放行。新增 `finalize-review` 将人工决定合并到最终 manifest/白名单/排除清单；发布命令改为从原 77 例 staging 按终签白名单物化 76 例 AG 候选目录，避免误发布 `WANG_DENG_FENG`。新增终签图，只把 3 个排除病例标红，所有已放行病例标黑。

**对应代码/产物**：`pipeline_wss_min/{v4_cutover.py,visualize_v4_cutover.py,tests/test_v4_cutover.py}`；`data_wss_min/pipeline_reports/v4_cutover_20260715_1921/{manual_review_decisions_20260716.json,v4_final_dataset_manifest.*,v4_final_whitelist.json,v4_final_exclusions.json,v4_manual_review_summary.json,v4_soft_flag_definitions.md}`；`assets_新队列审计/alignment_v4_cutover_signedoff_20260716/`。

**推进到实验步骤**：最终 AG=`76`、AAA=`63`，待审=`0`，硬 QA 失败=`0`，`ready_for_promotion=true`。10 项 v4 cutover 回归测试、Python 编译和终签图检查通过；其中已在临时目录完整演练“旧 84 例快照 + 原 staging 77 例保留 + 白名单 76 例原子提升”。

**当前状态判断**：人工终签已完成，但活动 `data_wss_min/AG` 仍为旧 84 例，`promotion_authorized=false`，本条未执行原子切换。`WANG_DENG_FENG` 原属历史 test16；排除后 v4 为 train61/test15，配对比较必须把旧 E2 也重汇总到 common-test15。

## 2026-07-15｜v4 人工审核图按 20 例分组与红黑姓名标记 ✅DONE

**本次主要修改**：在 bundle 直读的 cutover 可视化中新增 AAA/AG 每 20 例分组审核。每组同时生成固定毫米坐标的 X–Z 逐例小图和仿 `05_AG_AAA_v4_common_mm_ortho_overlay.png` 的 X–Z/Y–Z/X–Y 三视图叠加图。按用户反馈，待审与过审病例改为按组容量分层随机混排，固定 seed 保证可复现；待审姓名标红，过审姓名标黑，三视图图例保留逐例颜色与软标记。

**对应代码/产物**：`pipeline_wss_min/visualize_v4_cutover.py`；`assets_新队列审计/alignment_v4_cutover_review_20260715_1921/{grouped20_AAA,grouped20_AG}/` 共 16 张图，`grouped20_review_index.csv` 记录 142 例的分组、顺序、审核状态和图件路径。

**推进到实验步骤**：AAA 65 例分 4 组（20/20/20/5），21 个红名待审病例按容量比例分散到各组，再与 44 个黑名过审病例组内随机交错；AG 77 例同样分层随机混排为 4 组。逐图检查红黑标色、姓名、共同坐标范围和图例均正常。

**当前状态判断**：仅扩充人工审核材料，未修改 bundle、QA 结论或发布状态；`promotion_authorized=false`，仍等待用户逐组签核。

## 2026-07-15｜E4 deeper PointNet Job 8999 结果审核与配对回填 ✅DONE｜No-Go

**本次主要修改**：只读审核 Job `8999` 的 Slurm 状态、训练日志、checkpoint、best/last 全点评估和 best test16 PostView；将 `E4-DEEP-GLOBAL` 与同协议 `E2-GLOBAL` 做单变量配对，回填物理/归一化 R²、train−test gap、MAE/RMSE、high-WSS、top10 幅值比/IoU、Spearman 和双 self-max。本条同时保留前置实现事实：E4 结构为 `6→64→128→256→512；1024→512→256→128→64→1`，PointNet++ 三层 SA foundation 为 `500/125/32`、radius `0.05/0.10/0.20`、nsample=16。本次未修改训练/评估代码，未提交新作业，未启动 PointNet++ fine-tune。

**对应代码/文档**：`training_wss_min/{baseline_models.py,tools/visualize_pointnetpp_sa.py}`、`training_wss_min/configs/{pointnet_deeper/,pointnetpp_sa_foundation/,sweeps/pointnet_deeper_e4.txt}`、`training_wss_min/cluster/pointnet_deeper/`、`training_wss_min/tests/test_pointnet_deeper_sa.py`、`例子/06_PointNet++_SA三层采样与分组/`；[PointNet baseline 实验矩阵 §4.4](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#44-导师追加深度探针-e4-deep-global2026-07-15-完训完评no-go)、[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md)、`training_wss_min/configs/README.md`、`docs/README.md`；结果位于 `training_wss_min/runs/pointnet_deeper/outputs/e4_deep_global/`，日志为 `training_wss_min/cluster/logs/wsspn_e4_deep_8999.{out,err}`。

**推进到实验步骤**：Job `8999` 已 `COMPLETED (0:0)`，用时 `01:37:11`；400 epoch 齐全，best=第 379 epoch / train loss `0.121590`。`ckpt_best/last`、best/last train61+test16 评估均齐全，best PostView 16/16 病例 manifest 引用文件无缺失，mapping coverage 全为 100%，self-max/top10 图均为 16/16；日志无 Traceback/OOM/NaN。

**当前状态判断**：E4 相对 E2 将 best train loss 降低 29.5%，train 物理 `R²_cb` 从 0.6124 升至 0.6966；但 test 物理 `R²_cb` 从 0.2140 降至 0.1629，gap 从 0.3985 扩至 0.5337。test 归一化 `R²_cb` 从 0.4606 降至 0.4476，物理 MAE/RMSE 从 `2.8005/5.1141` 升至 `2.8510/5.2778`，high-WSS R² 从 −1.486 降至 −1.680，top10 幅值比从 0.378 降至 0.333。16 例物理 R² 为 6 例改善、10 例退化。虽然 top10 IoU `0.146→0.153` 和双 self-max `R²_cb −4.857→−4.208` 小幅好转，self-max 仍 16/16 负例，不足以抵消整体泛化恶化。结论为 **深度探针 No-Go**：保留 E2 锚点，停止纯 PointNet 深度扫描。best/last 同结论；test16 已被反复使用，只作导师驱动配对证据，不表述为无偏最终测试。

## 2026-07-15｜AG/AAA v4 staging、硬 QA 与可视化签核包 ⏳等待人工签核

**本次主要修改**：在预处理、批量报告、QA 和可视化入口增加显式 `--out-root`，bundle/report 改为临时文件后原子替换；新增 `v4_cutover.py` 的 staging 重建、bundle 硬门、旧版 SHA-256 清单和签核后原子提升命令，以及直接读 bundle 的 v4 对齐审核图。为处理 AAA `HAN_JIAN_FU` 唯一时间步的节点顺序循环错位，增加严格的坐标集一对一重映射；只有坐标集完全一致时才启用，真正移动网格仍会拒绝。

**对应代码/文档**：`pipeline_wss_min/{config.py,preprocess.py,reporting.py,run.py,qa_gate.py,v4_cutover.py,visualize_v4_cutover.py}`、`pipeline_wss_min/cluster/run_ag_v4_{staging,finalize}.slurm`、`pipeline_wss_min/tests/test_v4_cutover.py`、[pipeline README](../../pipeline_wss_min/README.md)、[新队列审计](新队列数据可用性审计_AAA_ILO_2026-07-10.md)、[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)。数据与审核产物位于 `data_wss_min/_staging/ag_v4_20260715_cutover/`、`data_wss_min/_staging/aaa_v4_20260715_fixes/`、`data_wss_min/pipeline_reports/v4_cutover_20260715_1921/` 和 `assets_新队列审计/alignment_v4_cutover_review_20260715_1921/`。

**推进到实验步骤**：Slurm `9007` 完成 AG 正式 77 例 v4 staging 重建，`9085` 完成 AG+AAA 终检与图件。AG `77/77`、AAA `65/65` 通过 bundle 硬门；旧 AG 84 例共 168 个 bundle/report 已生成路径、大小和 SHA-256 快照清单。可视化包含新旧 AG 对照、AG/AAA 共同毫米坐标与归一化三视图、`07_frame_metrics.csv/json` 以及所有软标记专项页。本轮没有训练、没有生成 v4 统计。

**当前状态判断**：已到达可视化人工签核门，不是已发布状态。硬 QA 零失败，但仍有 AG 1 例（`LI_ZHEN_SHAN`）和 AAA 21 例需人工确认软标记；`HAN_JIAN_FU` 修正 bundle 仍仅在 AAA fix staging。活动 `data_wss_min/AG` 仍为旧 84 例，`promotion_authorized=false`；未获用户明确签核前禁止执行 `promote`。签核后才能切换为 AG 77 例 v4，随后用独立 v4 实验 ID 配对复跑 `AG-v4 E2-GLOBAL`。

## 2026-07-15｜PointNet self-max 指标回填与下一轮优化讨论清单 ✅DONE

**本次主要修改**：按导师补充口径新增 `WSScfd/WSScfd,max`、`WSSpred/WSSpred,max` 及二者误差字段；扩展 PostView 的 CSV/VTP/manifest 和共享色标三联图。利用既有完整壁面 true/pred 对五组共 80 个 test case 做纯后处理回填，没有重新训练或运行模型前向。同步把 self-max 完整指标、负 R²/负预测点解释和下一阶段 P0–P5 优化方向写入 PointNet 矩阵状态真源。

**对应代码/文档**：`training_wss_min/tools/export_wss_postview.py`、`training_wss_min/tools/backfill_postview_selfmax.py`、`training_wss_min/tests/test_pointnet_distribution_matrix.py`；[PointNet baseline 实验矩阵 §4.3/§9](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。汇总产物：`training_wss_min/runs/pointnet_distribution_matrix/outputs/selfmax_test_summary.json`；每例图：`postview/ckpt_best/test/<case>/plots/fig_wss_selfmax_triptych.png`。

**推进到实验步骤**：五组 80/80 图、80/80 manifest、240 个 CSV 和 320 个 VTP 字段校验通过；14 项 PointNet 矩阵单元测试通过。当前完成结果归档与下一轮讨论预备，未生成新配置、未提交新 Slurm 作业。

**当前状态判断**：五组 self-max 的 16/16 逐病例 R² 均为负；E2-GLOBAL pooled R² 相对最好但仍为 −4.802，说明主要瓶颈不只是绝对幅值低估，病例内空间型态也明显失配。CASE 的线性输出产生负点（E2/E3 病例平均 7.34%/6.03%），应保留负值比例作为物理有效性护栏。下一轮先冻结 train61 内开发协议并审计推理期可得 BC/病例级信息，再讨论 shape/scale 双头、热点 loss 和局部拓扑表示；不自动重跑 random-5000 或原样 case-max。

## 2026-07-15｜给老师补齐 NMAE 表 + R² regression 图 ✅DONE

**本次主要修改**：原汇总 xlsx 只有 MAE/RMSE、无 NMAE，也无 true–pred 回归图。新增脚本 `docs/03-汇报材料/tools/build_pointnet_matrix_nmae_r2_report.py`，按 `NMAE=MAE/(max−min)` 汇总正式矩阵 6 组（物理 Pa + 归一化），并生成 hexbin 回归图；回填 `WSS_PointNet实验矩阵与结果汇总.xlsx`（新工作表 `NMAE与R2`，主表插入 NMAE 列）。E0 无 PostView，用 `model.eval()` 重推理补齐。

**对应代码/文档**：图与 CSV → `docs/03-汇报材料/figures/WSS_PointNet矩阵_NMAE与R2_20260715/`；xlsx 同上。

**推进到实验步骤**：汇报材料可直接发给老师（表 + 单 run 图 + 汇总网格）。

**当前状态判断**：物理 NMAE（range）因高峰点分母大而数值偏小，须与 R²/热点图一起看；CASE 的 NMAE≈MAE（分母≈1），不可与 GLOBAL log-z 的 NMAE 横比。

## 2026-07-15｜PointNet 分布矩阵五组结果、PostView 与归一化结论回填 ✅DONE

**本次主要修改**：只读核验 Slurm Jobs `8976–8980`、训练日志、checkpoint、best/last 全点评估和全部 test16 PostView 后，完成 E2/E3/E23 与 GLOBAL/CASE 的配对分析。五个 Job 均 `COMPLETED (0:0)`，无 traceback/OOM；5/5 run 均有 best/last train61+test16 指标，best 共 80/80 个病例包完整，VTP surface mapping coverage 100%。抽查 `WANG_DENG_FENG` 与 `GONG_HUI_XIA` 的 top10 overlay，确认定量热点结论与空间图一致。

**对应代码/文档**：[PointNet baseline 实验矩阵与进度跟踪 §4.2](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#42-正式矩阵结果2026-07-15)、[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md)、`training_wss_min/configs/README.md`、`docs/README.md`；产物位于 `training_wss_min/runs/pointnet_distribution_matrix/outputs/`。本次未修改训练/评估代码，未提交新作业。

**推进到实验步骤**：E2-GLOBAL、E3-GLOBAL、E23-GLOBAL、E2-CASE、E3-CASE 全部完成训练 → best/last eval → best test16 可视化。预注册主模型仍为 `ckpt_best(train_loss)`，没有根据 test 指标反选 last。

**当前状态判断**：导师宽网 E2 是本轮最强相对改进，test 物理 `R²_cb=0.2140`，比 E0 的 0.1414 增加 0.0725；random-5000 单独增益弱，E23 没有超过 E2，说明容量是主要因素且未见点数协同。`WSS/WSSmax` 在 E2/E3 下将 normalized `R²_cb` 分别从 0.4606/0.4218 降至 0.1724/0.1551；E3-CASE 虽提高 top10 IoU，但整体 R²、Spearman 和动态范围退化，CASE 主结论为 **No-Go**。E2 的 high-WSS R² 仍为 −1.486、top10 幅值比仅 0.378，因此本矩阵不判为可部署 Go；test16 已被连续用于探索性比较，后续也不能把它表述为无偏最终泛化估计。

## 2026-07-14｜PointNet baseline 矩阵文档同步 `E0`/`8970` 结案 ✅DONE

**本次主要修改**：更新 [PointNet baseline 实验矩阵与进度跟踪](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)：`E0-GLOBAL` 标为完训+完评并写入 §4.1 指标表；`8970` 标为补充 No-Go；进度表/执行顺序去掉「排队/运行中」；注明 §6 完整可视化合同从 `E2/E3` 起补齐。

**推进到实验步骤**：矩阵真源已与实测一致；下一跳仍是实现/提交 `E2-GLOBAL`。

**当前状态判断**：文档状态与推进记录文首分析对齐。

## 2026-07-14｜`8968` train+test16 与 `8970` wide128 结果分析 ✅DONE

**本次主要修改**：只读复盘完训/完评：`8968` 事后 eval `8974`/`8975`（`train,test`），以及宽网探针 `8970`（val-only）。产物：`runs/pointnet_trainloss_e400/outputs/{pointnet_xyz,pointnet_xyzgeom}/eval/`、`runs/pointnet_wide128/outputs/pointnet_xyzgeom/eval/`。

**关键指标**（主指标 `R²_field_cb`；**禁止**把 test16 与 2×3 的 val8 当成同一列比）：

| Run | 分区 | R²_cb | R²_raw | case mean/med/P10 | 负例 | RMSE/MAE | high-WSS R² | top10 比/IoU |
|---|---|---:|---:|---|---|---:|---:|---|
| 8968 PN·xyz | train61 | 0.5704 | 0.5677 | 0.505/0.522/0.388 | 0/61 | 4.32/2.09 | −0.04 | 0.635/0.435 |
| 8968 PN·xyz | **test16** | **0.0825** | 0.0832 | 0.071/0.033/−0.143 | **5/16** | 5.60/3.10 | **−1.88** | 0.284/0.095 |
| 8968 PN·xyz+geom | train61 | 0.5303 | 0.5277 | 0.454/0.462/0.326 | 0/61 | 4.51/2.19 | −0.20 | 0.581/0.415 |
| 8968 PN·xyz+geom | **test16** | **0.1414** | 0.1433 | 0.106/0.074/−0.062 | **5/16** | 5.42/2.95 | **−1.76** | 0.328/0.174 |
| 8970 wide128·xyz+geom | val8 | 0.3071 | 0.3167 | 0.173/0.169/−0.007 | 1/8 | 4.10/2.30 | −1.17 | 0.406/0.224 |
| 2×3 PN·xyz+geom（锚） | val8 | 0.3015 | 0.3122 | 0.184/0.140/+0.014 | 1/8 | 4.11/2.29 | −1.17 | 0.457/0.235 |

**判读**：
1. **`8968` 过拟合**：train≈0.53–0.57，test16 仅 0.08–0.14（gap≈0.39–0.49）；无 val + train_loss 选模不适合作精度主线。
2. **几何略优泛化**：test 上 xyz+geom 比纯 xyz **+0.059**；后续保留 xyz+geom。
3. **高 WSS 仍崩**：test high-WSS R²≈−1.8，top10 比~0.3。
4. **`8970` 加宽 No-Go**：val 0.3071 vs 锚点 0.3015（Δ≈+0.006），容量单变量无实质增益。

**Go-NoGo**：`8968` 作可部署精度 **No-Go**（test 过低）；作 train-fit 容量信号有。`8970` 相对锚点 **No-Go**。

**下一步**：按 PointNet baseline 矩阵推老师通道 / 5k random；不把 8968 test16 当新追分基线。

**推进到实验步骤**：8968/8970 数值已结案；正式 `E2/E3` 尚未提交。

**当前状态判断**：瓶颈在泛化与高 WSS，不在能否压低 train loss。

## 2026-07-14｜PointNet baseline 实验矩阵与归一化讨论跟踪建立 📄

**本次主要修改**：新建 PointNet baseline 独立跟踪文档，冻结后续只跑 `PointNet+xyzgeom`、无 val/无早停、5000 点 random 不放回且每 epoch 重采样；将容量/采样两条父实验各自拆成全局 log-z 与逐病例归一化配对。记录逐病例 `WSS/WSSmax` 与逐例 log-z 的待冻结差异，并定义归一化空间分布、top10 high-risk 定位和逐 test case 产物口径。

**对应代码/文档**：[PointNet baseline 实验矩阵与进度跟踪](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md)、`docs/README.md`、根 `README.md`及本记录。本次未改训练代码或配置。

**推进到实验步骤**：`8968`/`8970`/`8974`/`8975` 数值已出（见文首）；正式 `E2-GLOBAL` / `E3-GLOBAL` 尚未实现或提交。

**当前状态判断**：实验矩阵合同已建立；旧探针结案后进入矩阵正式父实验。逐病例公式未冻结前不启动 `E2-CASE` / `E3-CASE`。

## 2026-07-14｜`8968` train+test16 事后 eval 已提交 `8974`/`8975` ✅COMPLETED

**本次主要修改**：取消仅 train 的 `8972`/`8973`；改为一次评 `train,test` 并解锁 `--allow-test`。作业 `8974`/`8975` 均 `COMPLETED (0:0)`；数值见文首分析条。

**当前状态判断**：eval 产物齐全（`metrics.json` 含 train+test）。

## 2026-07-14｜`8968` train-fit 事后 eval 已提交 `8972`/`8973` ↪已取消，改 `8974`/`8975`

**本次主要修改**：对 `pointnet_trainloss_e400` 两臂提交完整壁面 **train** 分区评估（不读 val/test）。后因需保留指标并做 test16，已 `scancel`，由上条 `train,test` 作业替代。

**当前状态判断**：已取消，不作为有效产物。

## 2026-07-14｜PointNet train_loss×400 完训分析 `8968[0-1]` ✅完训｜待 eval

**本次主要修改**：只读复盘作业 `8968[0-1]`（无 val / `selection_rule=train_loss` / 400 epoch / split `61/0/16`）。两枪均 `COMPLETED`：`pointnet_xyz` ~84.1 min，`pointnet_xyzgeom` ~85.6 min；均跑满 400 epoch，无早停。产物：`training_wss_min/runs/pointnet_trainloss_e400/outputs/{pointnet_xyz,pointnet_xyzgeom}/`（`ckpt_best.pt`、`history.jsonl`、`history.png`）。

**关键指标**（仅 train_loss；本协议训练期不写 R²，test16 未读）：

| 臂 | best train_loss | best epoch | last loss | late50 mean±std |
|---|---:|---:|---:|---|
| PointNet · xyz | 0.2164 | 379 | 0.2242 | 0.2261±0.0060 |
| PointNet · xyz+geom | **0.2105** | 383 | 0.2167 | 0.2206±0.0051 |

- 几何臂在 400 epoch 中 **91.75%** 步 train_loss 低于纯 xyz；全程均值差约 `+0.010`（xyz−geom），末 100 epoch 约 `+0.006`。
- 曲线：两臂均前 ~20 epoch 陡降，其后缓慢下行至 ~0.21–0.23；末段仍有小幅下降与 batch 抖动，未完全平台。
- 对照冻结 2×3（**不可直接比 R²**）：同架构 baseline 在 val 选模下 `xyz` best train_loss≈0.257（150 ep）、`xyz+geom` 早停 70 ep 且 best val 落在 e9（当时 train_loss≈0.43）。8968 把原 val8 并入 train 并强制训满 400，train fit 更深属协议预期，不构成泛化 Go。

**Go-NoGo**：待定 — 缺泛化指标。本枪只能证明「无 val + train_loss 选模 + 400 ep」可稳定完训，且几何输入在 train fit 上仍略优；**不能**相对 baseline `R²_field_cb=0.3015` 裁决升降。

**下一步**：
1. 开发侧：`evaluate --partitions train` 报 train-fit 天花板（不碰 test）。
2. 获批后再 `--partitions test --allow-test` 做一次 test16 终评。
3. 勿用原 val8 当 held-out（已并入 train61）。

**对应代码/文档**：本条；[训练实验跟踪](WSS最小化_训练实验跟踪.md)；日志 `training_wss_min/cluster/logs/wss_pn_tl_8968_{0,1}.{out,err}`。

**推进到实验步骤**：训练完成；正式 R² / 物理误差尚未写出。

**当前状态判断**：完训健康；结论卡在事后 eval。

## 2026-07-14｜导师建库脚本改为读 bundle.npz（去掉 ascii 全流程） ✅DONE

**本次主要修改**：`teacher/build_wss_dataset.py` 改为老师式「读数据→pkl」：从 `data_wss_min/AG/*/bundle.npz` 取 xyz+几何与峰值 WSS，按正式 split 写出 pkl；不再包含 ascii/ascii_in/STL 预处理。交付说明同步。

**对应代码/文档**：`teacher/{build_wss_dataset.py,交付说明_WSS最小化适配老师接口.md}` 及本记录。

**推进到实验步骤**：建库脚本可直接在有 bundle 的环境运行；原始 Fluent 预处理仍走 `pipeline_wss_min`。

**当前状态判断**：与老师「读已有点云再打 pkl」的职责更一致。

## 2026-07-14｜导师交付脚本按老师源码风格重写 ✅DONE

**本次主要修改**：按用户澄清「外形跟老师、内容用我们项目」，重写 `teacher/build_wss_dataset.py` / `model_train_wss.py`：函数骨架与老师 `sampling_buid_dataset` / `model_train_1` 对齐（process_*、get_model/train_step/my_collate、Adam+Plateau、train-loss 存 best）；内容改为 WSS-min 全流程建库、6 维 xyz+几何、log_z WSS、FPS-2000。去掉 argparse 工程壳。交付说明改为「风格对齐」表述。

**对应代码/文档**：`teacher/{build_wss_dataset.py,model_train_wss.py,交付说明_WSS最小化适配老师接口.md}` 及本记录。

**推进到实验步骤**：可与老师原版并排审阅；尚未实跑建库/训练。

**当前状态判断**：交付形态符合「老师风格模板 + 本项目适配」。

## 2026-07-14｜PointNet 加宽容量探针（width=128，xyz+geom）⏳已提交 `8970`

**本次主要修改**：相对冻结 2×3 最佳格 `PointNet+xyz+geom`，仅加宽通道（`width=32→128`、`head_hidden=64→256`，局部 `in→128→256→512`，相对老师 `256→512` 多一层 128 过渡；`dropout=0`），其余协议不变（FPS-2000、val-only、seed=1234、未加权 MSE）。不重跑 preprocess。后续计划：代码严格对齐老师通道表后再提交两枪——(A) 老师对齐 + FPS-2000；(B) 老师对齐 + 5k random / 每 epoch 重采。

**对应代码/文档**：`training_wss_min/configs/pointnet_wide128/pointnet_xyzgeom.json`、`training_wss_min/configs/sweeps/pointnet_wide128.txt`、`training_wss_min/cluster/pointnet_wide128/`、`training_wss_min/configs/README.md` 及本记录。

**推进到实验步骤**：作业 `8970` 已提交；产物将写入 `training_wss_min/runs/pointnet_wide128/outputs/pointnet_xyzgeom/`；对照基线 `baseline_2x3_simple/outputs/pointnet_xyzgeom`（`R²_field_cb=0.3015`）。

**当前状态判断**：第一枪只隔离「加宽」杠杆；老师严格对齐与 5k 随机采留待第二批，避免与容量效应混杂。

## 2026-07-14｜导师两文件交付代码落地（建库全流程 + PointNet 读 pkl） ✅DONE

**本次主要修改**：用户确认交付口径后，在 `teacher/` 落地 `build_wss_dataset.py`（Fluent/STL/中心线全流程 → `wss_{train,val,test}.pkl` + log-z 统计，特征 6 维 xyz+几何）与 `model_train_wss.py`（老师式 get_model/collate/采点循环；FPS-2000；AdamW+warmup+cosine；完整壁面 val；case-balanced R² 或 train_loss 选模）。同步更新交付说明为已实现状态。`py_compile` 通过。未跑全量建库/训练（需集群与 conda）。

**对应代码/文档**：`teacher/{build_wss_dataset.py,model_train_wss.py,交付说明_WSS最小化适配老师接口.md}` 及本记录。

**推进到实验步骤**：交付脚本可发给老师审阅；试跑可用 `--limit 1`（建库需 `GNN_vmtk`/vtk，训练需 `GNN`）。

**当前状态判断**：两文件接口与文档口径一致；全量 77 例建库应走集群，勿在登录节点直接跑。

## 2026-07-14｜导师两文件交付口径说明（pkl 建库 + 训练）已定稿 📄待实现代码

**本次主要修改**：在 `teacher/` 写清发给导师的交付说明：保留老师「建库脚本 + 训练脚本」两文件形态；建库须 WSS-min **全流程**（原始 Fluent/STL/中心线 → 配准/归一化/几何/QA → pkl）；特征为 **xyz+几何（6 维）**，标签为峰值壁面 WSS；训练脚本模仿老师单文件读 pkl / PointNet / 采点循环，协议采用本项目 log-z、FPS-2000、case-balanced R² 等。老师原版 `sampling_buid_dataset.py` / `model_train_1.py` 保留对照；适配代码 `build_wss_dataset.py` / `model_train_wss.py` 标为待实现。

**对应代码/文档**：`teacher/交付说明_WSS最小化适配老师接口.md`、`teacher/{sampling_buid_dataset.py,model_train_1.py}` 及本记录。

**推进到实验步骤**：交付口径与 pkl 协议已文档化，可直接发给老师确认；尚未落地两份适配脚本、未重跑 preprocess/训练。

**当前状态判断**：用户确认输入 xyz+几何、建库全流程、训练内容用本项目协议后，下一步是在 `teacher/` 实现上述两脚本。↪ 已由上条落地代码替代。

## 2026-07-14｜导师展示 train / pipeline 改为真正自包含单文件 ✅DONE

**本次主要修改**：按导师“全部功能压缩到一个文件”的要求，重写两份展示代码并删除对项目内部模块的调用。PointNet 文件内直接实现 JSON 配置解析、bundle/split 读取、train-only 几何统计、log-z、确定性 FPS、DataLoader、PointNet、AMP、MSE/AdamW、warmup+cosine、梯度裁剪、完整壁面验证、case-balanced R²、早停和 best/last checkpoint。预处理文件内直接实现 Fluent ASCII/中心线/STL 读取、入口峰值选择、节点 ID 对齐、单位换算、中心线平移修复、STL 解剖坐标架、未描入口裁剪、坐标归一化、近壁标注、中心线几何特征、全时间步 WSS/压力/矢量堆叠、QA、冻结 log-z 与 FPS 峰值样本；展示产物只写 `outputs/`，不覆盖正式 bundle。两份代码的说明性注释均使用中文。

**对应代码/文档**：`training_wss_min/examples/pointnet_baseline_train.py`、`pipeline_wss_min/examples/preprocess_pipeline.py`、两侧相关测试与 README，以及本记录。

**推进到实验步骤**：训练文件 224/250 行，使用冻结 baseline JSON 完成 1 epoch 的 53 例训练和 8 例完整壁面 val 实跑，成功写出 loss、指标和 best/last checkpoint；预处理文件 248/250 行，使用 included 病例 `AG/fast/ZHANG_HAO` 从 81 个原始时间步实跑，生成 12,521 壁面点、770,914 内部点的 bundle、QA 报告和 FPS-64 峰值样本。训练/预处理共 29 项单元测试、v4 配准合成测试、编译、CLI、内部 import 禁止项和 PointNet 冻结实现输出一致性均通过；临时产物已清理，未改正式数据、未访问 test16。

**当前状态判断**：两份文件现在都可脱离仓库内部 Python 包独立审阅和执行，只依赖通用第三方库、JSON 配置/split/stats 与原始/预处理数据。生产级模块化入口继续保留用于正式实验，但老师看到的文件不再把核心逻辑藏在调用后面。

## 2026-07-14｜`pipeline_wss_min` 目录重构、历史归档与中文注释收口 ✅DONE

**本次主要修改**：将根目录收敛为 10 个正式核心模块；AAA/ILO v4 的白名单、预处理、只读几何审计、bundle 终检、AG 回归和共同坐标可视化统一迁入 `new_cohorts/`，并把重复的白名单常量与 `unit_id` 解析合并到 `common.py`。2026-07-07 至 2026-07-08 的 AG flow-divider/LR/STL/居中 QA 脚本迁入 `archive/alignment_v3/`，统一冻结为 `legacy_centerline` 且不保留旧根模块兼容层。`run.py` 改为基于 dataclass 复制构造命令行配置，避免采样覆盖项原地污染全局 `DEFAULT`；保留源码的说明性注释/docstring 和用户可见提示统一为中文，技术字段名与文件格式名保持不变。删除 494 个已完成 Slurm `.out/.err`、275 个已完成本地 WSS-min 日志和全部 Python 缓存；所有 bundle、JSON/CSV 审计、图件、split 和训练产物均保留。

**对应代码/文档**：`pipeline_wss_min/{README.md,run.py,config.py,registration.py,raw_io.py,preprocess.py,reporting.py,examples/,tests/}`、`pipeline_wss_min/new_cohorts/`、`pipeline_wss_min/archive/`、`pipeline_wss_min/cluster/`、根 `README.md`、`.gitignore`、归档交接记录及本记录。

**推进到实验步骤**：工程重构与回归验证完成；`compileall` 通过，7 个 WSS-min 单元测试和 5 个 v4 配准合成回归测试通过，3 份 Slurm 脚本通过 `bash -n`，新队列入口仍解析出 171 个双白名单单元。当次展示入口以复用正式模块方式验证；其后已由本文首条记录中的 248 行自包含实现替代。未改写正式 bundle，也未访问 test16。

**当前状态判断**：当前入口、目录职责和代码事实已经一致；AG 正式四阶段继续使用 `pipeline_wss_min.run`，AAA/ILO 使用 `pipeline_wss_min.new_cohorts.*`，历史 AG 坐标 QA 只从归档路径复核。旧命令会直接失败，避免调用者误以为仍在执行当前 v4 口径。

## 2026-07-14｜导师展示用单文件 WSS-min 预处理旧实现 ↪ 已由自包含版替代

**本次主要修改**：新增 229 行的单文件预处理展示入口。默认直接调用正式 `preprocess_case`，完整保留原始 CFD 校验、稳定节点对齐、单位换算、解剖坐标架/配准、裁剪、坐标归一化、壁面标记、几何特征及全时间步 WSS/压力/矢量堆叠；随后在同一文件中显式展示病例 QA、冻结的 train-only peak WSS log-z 统计校验、归一化坐标 FPS 和峰值训练样本构建。入口只接受当前 split 中 included 的 AG train/val/test 病例，并提供 `--skip-preprocess` 安全复用既有 bundle；展示产物仅写入 `outputs/wss_min/teacher_preprocess/`。

**对应代码/文档**：`pipeline_wss_min/examples/{preprocess_pipeline.py,__init__.py}`、`pipeline_wss_min/tests/{test_preprocess_example.py,__init__.py}`、`pipeline_wss_min/README.md` 及本记录。

**推进到实验步骤**：展示代码实现与验证完成；行数门限为 229/250，4 个展示入口单元测试、CLI/编译检查通过。使用既有 included 病例 `AG/fast/ZHANG_HAO` 以 `--skip-preprocess --wall-n 128` 跑通 QA、53 例冻结统计加载和样本构建；v4 配准合成回归测试通过。未重跑正式病例预处理、未重算全局统计、未改写正式 bundle，也未访问 test16。

**当前状态判断**：本条记录的是最初“紧凑入口调用正式模块”的版本，已不符合导师对自包含文件的要求；当前事实以本文首条 248 行自包含实现为准，且展示运行不会更新正式 bundle。

## 2026-07-14｜展示用预处理示例注释改为中文 ✅DONE

**本次主要修改**：将 `pipeline_wss_min/examples/preprocess_pipeline.py` 的模块说明、函数 docstring 与 argparse help 改为中文；逻辑与对外行为不变。

**对应代码/文档**：`pipeline_wss_min/examples/preprocess_pipeline.py` 及本记录。

**推进到实验步骤**：文档可读性调整完成；行数仍 ≤250，相关单元测试通过。

**当前状态判断**：仅注释语言变更，不影响正式 preprocess 路径与训练作业。

## 2026-07-14｜PointNet 无 val / train_loss 选模 / 400 epoch ✅完训 `8968[0-1]`（分析见文首）

**本次主要修改**：新增 train/test-only 划分（原 val8 并入 train→61/0/16），按新 train 重算 WSS 全局统计（不覆盖默认 `wss_global_stats.json`）；`train.py` 支持 `selection_rule=train_loss`（按 epoch 训练损失选 best，`early_stop_patience<=0` 关早停，不加载 val）；冻结 PointNet `xyz` / `xyz+geom` 两份 400-epoch 配置与 Slurm 入口。未重跑 preprocess，未访问 test16。作业 `8968` 已提交（array 0–1）并完训。

**对应代码/文档**：`training/splits/split_AG_wss_min_v1_traintest.json`、`data_wss_min/fold_stats/wss_stats_v1_traintest.json`、`training_wss_min/{train.py,config.py,objectives.py}`、`training_wss_min/configs/pointnet_trainloss_e400/`、`training_wss_min/configs/sweeps/pointnet_trainloss_e400.txt`、`training_wss_min/cluster/pointnet_trainloss_e400/`、`training_wss_min/configs/README.md` 及本记录。

**推进到实验步骤**：训练完成；复盘见文首分析条。

**当前状态判断**：协议与冻结 2×3（val-only）隔离；待 train-fit / 获批 test16 事后 eval。

## 2026-07-14｜导师展示用单文件 PointNet baseline 训练旧实现 ↪ 已由自包含版替代

**本次主要修改**：新增 245 行的单文件 PointNet WSS 训练入口，默认复现 2×3 baseline 中 `PointNet + xyz+geom` 配置，也可通过 `--config` 切换冻结的 `PointNet + xyz`。文件内完整展示 PointNet shared MLP、病例级 max-pool、WSS decoder、FPS-2000 数据加载、MSE/AdamW、warmup+cosine、AMP、梯度裁剪、完整壁面 val、case-balanced R² 选模、早停和 best/last checkpoint；只复用已审计的数据解析与指标公式，明确不读取 test16。

**对应代码/文档**：`training_wss_min/examples/pointnet_baseline_train.py`、`training_wss_min/examples/__init__.py`、`training_wss_min/tests/test_core_refactor.py`、`training_wss_min/README.md` 及本记录。

**推进到实验步骤**：展示代码实现与等价性验证完成；行数门限为 245/250，模型可直接加载正式 `PointNetRegressor` state dict，固定随机输入下输出逐值一致；全套 22 个单元测试与 CLI/编译检查通过。未启动训练、未重评、未访问 test16。

**当前状态判断**：本条记录的是最初仍复用项目数据/指标模块的版本，已不符合导师对自包含文件的要求；当前事实以本文首条 224 行自包含实现为准。冻结 JSON 协议和正式模块化训练入口继续保留。

## 2026-07-14｜`training_wss_min` 核心代码重构与历史入口清理 ✅DONE

**本次主要修改**：按用户确认的保留边界清理训练库：删除全部配置生成器和根目录旧命令兼容 shim，删除 Python 缓存与已完成 Slurm 的原始 `.out/.err`，保留所有 run checkpoint、指标、训练日志、可视化结果、JSON 配置和 manifest。将模型工厂、loss/选模、日志/随机种子从 `train.py` 拆到独立核心模块；训练协议、指标口径和 test 锁未改变。2×3 baseline 的 6 份 JSON 从被忽略的 `runs/` 迁到正式配置目录并新增冻结 manifest/Slurm 入口，不再依赖生成器。

**对应代码/文档**：`training_wss_min/{models.py,objectives.py,runtime.py,train.py,evaluate.py,pointnext.py,README.md}`、`training_wss_min/configs/{README.md,baseline_2x3/,sweeps/baseline_2x3_simple.txt}`、`training_wss_min/cluster/baseline_2x3/`、`training_wss_min/{tools,experiments,tests}/` 及本记录。历史 `dist_to_wall` 配置在 `configs/README.md` 明确标为仅审计保留。

**推进到实验步骤**：工程清理与等价性验证完成；20 个单元测试通过，四种模型合成点云前向/反向通过，17 份 manifest 的 98 个 JSON 引用全部存在。未重训、未重评、未访问 test16。

**当前状态判断**：核心根目录只保留 11 个正式 Python 模块；2×3 baseline、历史 JSON 和现有实验结果均可追溯。后续新增实验需直接提交审查后的 JSON/manifest，不再恢复一次性配置生成脚本或旧入口兼容层。

## 2026-07-14｜2×3 baseline 后处理收窄为最佳/最差两例 ✅DONE，`8966/8967` 均 `0:0`

**本次主要修改**：按用户要求取消全量 `8961[0-5]`（及其依赖汇总 `8962`），清理全量/冒烟输出；postview 作业改为只导出最佳 baseline `PointNet+xyz+geom` 的最佳与最差 val 病例。排序依据是同点 `metrics.json` 的逐病例 R²：`slow/CHENG_LU_LI=0.3959`、`slow/XU_YI_CAI=-0.0577`。同步更新 `postview-surface-viz` skill：已完成 baseline 默认只做 best/worst；全量必须由用户明确要求；`wss/wss(max)` 的 CFD、预测和误差强制共用 CFD max 分母。

**对应代码/文档**：`.cursor/skills/postview-surface-viz/SKILL.md`、`training_wss_min/runs/baseline_2x3_simple/postview/{README.md,run_dirs.txt,val_cases.txt,export_array.slurm,submit.sh}`、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：单例端到端 VTP/mapping QC 冒烟已通过；两例最终导出 `8966` 与自动汇总 `8967` 均完成。两个 VTP 均含 10 个规定数组，Gaussian mapping coverage 均为 100%；`comparison.csv` / `comparison_by_run.csv` / `batch_manifest.json` 已写入。未访问 test16。

**当前状态判断**：最终交付只含两例和 PointNet+xyz+geom 一个模型，避免无意生成 48 个病例包；最佳例 `CHENG_LU_LI` 同点 R²=0.3959，最差例 `XU_YI_CAI`=−0.0577。VTP 标量及同点指标口径保持不变。

## 2026-07-14｜2×3 baseline 壁面 ParaView 可视化与比较 ⏳已提交 `8961[0-5]`

**本次主要修改**：扩展 WSS-min 后处理器，使同一 STL 面片 VTP 同时包含 CFD 真值、预测、signed/absolute error 与 `wss÷wss_max` 归一化显示字段；归一化一律使用同一病例 CFD 壁面最大值。面片回插器同步改为由归一化后的 CFD/Pred 字段重算误差，避免误差被独立二次插值。新增 baseline 6 run × val8 的 array 导出、mapping/QC 汇总器和交付说明。

**对应代码/文档**：`training_wss_min/tools/export_wss_postview.py`、`tools/cfdpost_cloud_export/map_to_stl_surface.py`、`training_wss_min/runs/baseline_2x3_simple/postview/{README.md,export_array.slurm,summarize.py,summarize.slurm,submit.sh}`、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：一例 CPU 端到端冒烟已通过：面片 VTP 含 10 个必须数组，Gaussian mapping coverage=100%。完整导出 array `8961[0-5]` 已提交（最多并发 4），依赖汇总 job `8962` 会在 48 个 VTP 都成功后写入比较 CSV；只读取既有 val run 和其 bundle，未访问 test16。

**当前状态判断**：可视化结果严格用于病例云图和软件检查；R²/Pa 误差继续以同点 `_export/*__wall.csv` 与既有 `eval/metrics.json` 为唯一口径，不在插值 STL 上重算。

## 2026-07-14｜最小 2×3 baseline 全部完成 ✅（Job `8700[0-5]`，6/6 `0:0`）

**本次主要修改**：无新增训练代码；完成并汇总既有 `MLP / PointNet / PointNet++` × `xyz / xyz+geom` 六格的完整壁面 val 评估。模型、输入、AG v1 split、FPS-2000、seed、MSE 和 val-only 协议均保持冻结。

**对应代码/文档**：作业与产物 `training_wss_min/runs/baseline_2x3_simple/`；完整指标、逐病例 CSV、checkpoint 和日志均在各 `outputs/<run>/`；结果表与判读写入[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：Slurm array `8700[0-5]` 全部 `COMPLETED (0:0)`，每格完成训练与完整壁面 val 评估；未访问 test16。

**当前状态判断**：PointNet+xyz+geom 为本单 seed 最佳单格（`R²_field_cb=0.3015`，逐病例 P10=+0.014，1/8 负例）；所有 xyz+geom 组都优于相应的 xyz 组。六格 high-WSS R² 均为负且 top10 幅值明显偏低，故它是“最简单形式”的下限基线，非最终模型或可推广的架构排名。

## 2026-07-14｜AAA/ILO 原始 STL 自动解剖坐标架 v4 与 centerline 错位修复 ✅

**本次主要修改**：从第一性原理重建有符号解剖坐标架：由原始 STL 自动选取近端主干、分叉中心和双髂支端点，固定近端主干为 `+Z`、髂支为 `-Z`，用原始 STL 世界 `+X` 对左右轴定号，并保持 `det(R)=+1`。新增 centerline↔壁面纯平移守卫：只在偏移跨过一条血管尺度且平移后最近邻残差通过时修复。壁面拓扑参考改为首步/峰值步/末步三点共识；只允许丢弃极少量非稳定额外节点，稳定节点缺失仍硬失败。

**对应代码/文档**：`pipeline_wss_min/{surface_io.py,registration.py,config.py,preprocess.py,reporting.py,audit_new_cohort_frame.py,audit_ag_frame_v4.py,preprocess_new_cohorts.py,qa_new_cohorts_v4.py,visualize_new_cohort_frame_v4.py}`、`pipeline_wss_min/cluster/{run_new_cohorts_preprocess.slurm,run_new_cohorts_finalize.slurm}`、`tests/test_wss_min_registration_v4.py`、[新队列数据审计](新队列数据可用性审计_AAA_ILO_2026-07-10.md)、`assets_新队列审计/alignment_v4/`。

**推进到实验步骤**：AAA/ILO 数据层干净的 171 单元全量几何门 `171/171` 通过；用户指定的 6 个平移错位单元已修复，并额外发现/修复同类 `ILO/YU_XIANG_SHENG-1/after`，共 7 个。AG included=77 只读回归 `77/77` 通过且无误修复。已修复 ILO 单段 cohort 路径兼容；`ILO/ZHAO_JIAN_PING-0/after` 确认为首步多 1 个瞬态节点，稳定参考选择峰值步 `1162`，只从 `1120` 丢弃 node `111167`。最终作业 `8956` 与终检 `8957` 均 `COMPLETED (0:0)`；汇总 `n_ok=171, n_missing=0, n_other=0`，bundle QA `n_qa_pass=171, n_failed=0`，7 个修复单元与几何审计名单完全一致。

**当前状态判断**：新队列 171 个 bundle/report 已全部入库，仅写 `data_wss_min/AAA/**` 和 `data_wss_min/ILO/**`；未重跑 AG preprocess/global-stats/build-samples，`data_wss_min/AG/**` 当日改写数为 0。已提交的 AG 2×3 实验不受影响。如后续要混合 AG+AAA/ILO，须将 AG 整体重建为 v4，禁止混用 v3/v4 bundle。

## 2026-07-14｜第五轮正式计划与执行计划归档 ✅

**本次主要修改**：确认第五轮已完成诊断性科学结案，但未达到内部工程目标；将正式计划与历史执行计划集中移入 `_archive/WSS最小化/`，不再占用当前推进目录。

**对应代码/文档**：[第五轮正式计划](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)、[第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)、[第五轮结案说明](_archive/WSS最小化/WSS最小化_第五轮结案与归档说明_2026-07-12.md)、归档 README。

**推进到实验步骤**：第五轮停止新增任务；L0/OOF/T16 按止损决策保持未触发，后续优化统一进入第六轮。

**当前状态判断**：第五轮历史证据和产物路径保留，当前状态、指标和待办以第六轮文档及训练跟踪为准；同步修复了归档后相互引用的相对链接。

## 2026-07-14｜合并 `main` 最新推进与变更文档 ✅

**本次主要修改**：将 GitHub `main` 最新提交 `9254c11`（2026-07-13）中 `docs/02-推进与变更` 的路线计划、审计报告、训练跟踪、历史归档和 CSV 证据合并到本分支；保留本分支已有的 2×3 baseline 等补充记录。

**对应代码/文档**：仅修改 `docs/02-推进与变更/**`；未修改训练代码、配置、数据及其他目录。

**推进到实验步骤**：文档路线统一到 WSS 主线、横向最小诊断矩阵与条件路线的最新状态；新增/更新文件路径已逐项核对，未启动新实验。

**当前状态判断**：文档内容以 `main` 最新审计口径为基线，同时保留本分支实验事实；当前等待既有作业结果，`test16` 仍保持未读。

## 2026-07-13｜最小 2×3 基线（MLP / PointNet / PointNet++ × xyz / xyz+geom）⏳已提交 `8700[0-5]`

**本次主要修改**：训练统一模型工厂新增最基础的逐点 `MLP`、全局 max-pool `PointNet` 和经典 SA+FP `PointNet++`；后者不含 PointNeXt 的残差或倒置瓶颈。新增冻结的 2×3 配置生成器与 Slurm array：统一 AG v1 `53/8/16`、peak WSS 单标量、FPS-2000、seed=1234、未加权 MSE、val-only；禁用旋转增强、采样/几何/目标加权、多任务和 raw-space 辅助项。

**对应代码/文档**：`training_wss_min/{config.py,baseline_models.py,pointnext.py,train.py}`、`training_wss_min/runs/baseline_2x3_simple/{README.md,make_configs.py,train_array.slurm,submit.sh}`、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：6 份配置已生成并通过 JSON 解析；MLP、PointNet、PointNet++ 均完成合成双病例点云的 forward/backward 冒烟，Python 编译与 Slurm shell 语法检查通过。已通过 `submit.sh` 提交 Slurm array `8700[0-5]`（最多并发 4）；每格训练后只做完整壁面 val 评估，产物固定写入该 runs 子目录。

**当前状态判断**：矩阵只改变模型与输入（`xyz` 或 `xyz+abscissa_norm+local_radius+curvature`），可作为后续复杂模型/模块的可解释下限；`test16` 保持未读。当前等待队列调度和六格完成，完成后回填指标。

## 2026-07-12｜第六轮 W0 审计 + W1 因子补全(E) + C 邻域预审计 + W2-L1 loss Gate + W3 组合 ✅DONE
## 2026-07-13｜第六轮最终审查后文档收敛整理 ✅DONE（文档）

**本次主要修改**：
- 将四份第六轮执行文档改写为当前状态页，删除旧版 F3、旧优先级、已完成 W0–W3 的重复计划和 36-run 理论矩阵。
- 冻结唯一口径：固定 peak/WSS-only；`P0-Metric → P0-Density → 2×2 F3 → 分支优化 → repeated validation`。
- 统一指标语义：level、pattern、hotspot、病例等权 Pa 误差分列；旧 `R²_casemean` 不再解释为病例 level R²。
- 保留实验证据在训练跟踪、变更历史在本记录；当前计划只保留有效结论、Gate、暂停项和待办。
- 在 `docs/README.md` 增加 WSS-only 当前入口，避免从 V3P 文档误入本路线。

**对应代码/文档**：[第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)、[WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md)、[横向多目标计划](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、[文档索引](../README.md)与本记录。

**推进到实验步骤**：文档已收敛，下一项唯一 P0 是指标/选模合同代码闭环与已有 checkpoint 的只读重评；本次未修改代码、配置或数据，未启动作业。

**当前状态判断**：没有通过新版指标和病例划分确认的最终候选；E/D+raw-Huber 保持冻结，E×raw-Huber No-Go，test16 未读。

## 2026-07-12｜第六轮 W0 审计 + W1 因子补全(E) + C 邻域预审计 + W2-L1 loss Gate + W3 组合 ✅DONE（历史实验裁决；E 排名已由上文降级）

**本次主要修改（代码/配置/审计工具）**：
- `training_wss_min/tools/make_configs_xyz_scale.py`：`GROUPS` 增 **E 组**（`xyz+coord_scale+geom`），补齐 A/B/D/E 2×2 嵌套因子；协议 JSON 升级为 `xyz_scale_abde_v1`，写入 `factor_design`（B−A/E−D/D−A/E−B/交互 E−D−B+A）；新增增量 manifest `configs/sweeps/xyz_scale_e.txt`（只列 E，避免重跑 A/B/D）。
- `training_wss_min/tools/make_configs_loss_l1.py`（新增）：W2-L1 raw-Huber λ∈{0.1,0.3,1.0} 单 seed 配置，基于 `r6_scale_D_xyzgeom_s1234`（固定 D 输入），只改 `loss_raw_huber_lambda`；预注册协议 `loss_l1_rawhuber_v1`。
- `training_wss_min/tools/make_configs_round6_w3.py`（新增）：W3 确认阶段——L1 λ=1.0 补 s7/s2025、W3 组合 E×raw-Huber λ=1.0 三 seed。
- `training_wss_min/tools/config_paths.py`：新增路由 `r6_l1_`→`loss_aug_ablation/`、`r6_w3_`→`round6_w3/`。
- `training_wss_min/tools/audit_coord_scale_w0.py`（新增）：W0 只读尺度审计（coord_scale 范围/OOD、与几何/WSS 的 Spearman、fast-slow 与裁剪分层）。
- `training_wss_min/tools/audit_c_neighborhood.py`（新增）：W1 C 邻域预审计，复现 SA 级联对比 A/C 每层邻居中位/孤立率/截断率；限线程、探测上限 64。
- `training_wss_min/tools/summarize_round6_w1w2.py`（新增）：只读汇总 A/B/D/E 因子（配对 seed 差+交互项）、L1 λ Gate、W3 组合 Gate。

**作业与结果**：`7557–7559`(E×3)、`7560–7562`(L1 λ-gate)、`7563–7567`(L1 λ1 confirm s7/s2025 + W3×3) 全部 `sbatch`，**11/11 COMPLETED**。产物 `runs/_audits/{w0_coord_scale,w1_c_neighborhood,round6_w1w2_summary}/report.md`。

**核心结论**：① **E 成为最佳可部署输入**（E−D field_cb +0.021 过 Gate-1，负例最少）但**尺度/几何冗余**（交互 E−D−B+A<0，E−D casemean≈0）；② W0 印证 coord_scale 编码尺寸非幅值；③ C 邻域不退化但被 nsample=16 截断抹平预期收益，判低优先级；④ **L1 raw-Huber 单 seed 全面正、λ=1.0 最佳、无爆峰**，但**三 seed 确认后收益缩水**（单 seed field 0.385→三 seed 0.347±0.033）、seed 脆弱；⑤ **W3 组合阴性**——E×raw-Huber 未过 §6 组合 Gate（全 3 seed 低于 D+rawHuber、2/3 seed 低于 E），尺度特征与 raw-space Huber 冗余互斥，**正式停止组合线**。待用户裁决 W5 候选（E vs D+rawHuber）或启动 L2。详见[训练实验跟踪：第六轮 W0–W3 节](WSS最小化_训练实验跟踪.md)。

## 2026-07-13｜横向指标与 V3P 可比性重构，WSS 恢复为 P0 主线 ✅DONE（文档/决策）

**本次主要修改**：
- 新增跨路线共享指标口径，将对比分为 A（直接对比）/B（协议化参考）/C（叙事背景）三级，并冻结 WSS 的病例等权点级、逐病例稳健性、热点和下游四层指标。
- 修正原“V3P vs wss_min 是公平 WSS 对比、gap 约 0.1、已证明共同信息上限”的过强结论：压力定为 C 级，WSS 定为 B 级；只有统一 split/时相/点集/输入/选模的 Bridge 协议才能给精确 gap。
- 将第六轮调度从“先补齐 36-run 压力/速度横向表”改为“WSS W0–W3 是 P0，横向是最小诊断矩阵”。Track B 先做 adapter/QA，默认只跑 `|v|` 和联合 `u,v,w` 单 seed sanity；内部压力、独立速度分量与三 seed 按 Gate 触发。
- 将 `R²_field_casebalanced` + 物理单位误差提为 WSS 首要点级指标；`R²_field_raw` 保留为历史衔接，`R²_casemean` 不再单独作部署主指标。`0.70` 保留为长期理想目标，不作跨物理目标通用工程门槛。

**对应代码/文档**：[跨路线评估与横向对比口径](../00-规范与记录/WSS跨路线评估与横向对比口径.md)、[第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)、[横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md)、[BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md)、`docs/README.md`、`实验设计总纲.md`。本轮未改训练代码/配置，未启动新作业。

**推进到实验步骤**：第六轮路线重排完成；WSS W0–W3 可独立推进，Track B 仍等 adapter/QA，V3P Bridge 仅完成协议定义。

**当前状态判断**：原路线的主要风险是“不同协议的头条 R² 直接相减”和“为横向补表延误 WSS 主线”，而不是 gauge pressure 或 WSS-only 目标本身选错。当前 WSS 结果只能证明学到部分信号且稳健性/热点仍不足，不证明绝对信息上限已锁定。`test16` 保持未读。

## 2026-07-12｜V3P vs 最小路线：压力/WSS 数值可比性分析落档（已由 2026-07-13 口径修订取代）

> 历史记录保留；其中“WSS 公平对比、gap 约 0.1、已锁定信息上限”不再作当前结论，以本日志顶部 2026-07-13 条目和共享指标口径为准。

**本次主要修改**：向[横向多目标对比 §9](WSS最小化_第六轮_横向多目标对比计划与执行.md) 写入 V3P 数值可比性分析（§9.1 压力·重点、§9.2 WSS·公平对比、§9.3 部署指标口径），并在 §4.1 加交叉引用。

**核查与直算结论**：
- **压力方差分解**：本仓直算，壁面 bundle 与 `data_new` 全场 `y[:,3]` 一致——单峰帧 95.9% 是病例间"水平"、仅 4.1% 空间型态（跨全周期口径 ~90% 水平、其中 96.7% 是心动时间波）。"只预测每例均值"pooled R²=**0.959**，V3P R²_p 0.92–0.96 卡在此送分基线上。
- **BC = 压力水平**：V3P `global_cond={t_norm,BC_Inlet,BC_O1..O4}`，出口压力 BC 均值/方差≈压力目标本身；OLS 仅 BC→场均压力 R²=0.925，no-geom V3P 压力仍 0.936。出口 BC 是第五轮判定的 `oracle_non_deployable`。**故压力 0.92 与本路线 gauge 0.53 不可直接比。**
- **WSS 方差分解**：**11.5% 病例间 / 88.5% 空间型态**（与压力相反），"只预测每例均值"pooled R²=0.115——无水平可白送。V3P 最好 WSS ~**0.43**（平台 0.40–0.45，Go 线 0.459 未过），本路线纯几何 field **0.31–0.36**，**差距仅 ~0.1、都远低于 0.70**。V3P WSS 模型参数 0.25–1.3M（比本路线 4.4M 还小）→ 瓶颈是信息/表示上限，非模型容量。
- **§9.3 工程验收口径扩写**：明确"能不能用"要看物理单位误差（MAE/RMSE/分位 + NRMSE + 逐病例 casemean），不用 pooled 绝对 R²；附工程可用性小表——压力 gauge `xyz+geom` MAE **225 Pa**/RMSE 374 Pa（≈空间信号 531 Pa 的 0.42/0.70），WSS `xyz+geom` MAE **3.11 Pa**/RMSE 6.13 Pa（≈信号 6.2 Pa 的 0.50/0.99）；两者逐点误差都约为空间信号一半量级，趋势可见但离直接工程可用仍有距离。

**对应代码/文档**：[横向多目标对比 §9](WSS最小化_第六轮_横向多目标对比计划与执行.md)；证据源 `training/core/{models,metrics,losses}.py`、`pipeline/config.py`、`data_new/normalization_params_global.json`、V3 路线文档与 `任务A实验状态表.md`。

**当前状态判断**：分析性结论，未训练/未改训练代码。部署主指标口径确定为**逐病例 casemean**（压力用 gauge，WSS 用 raw+hotspot 护栏）；绝对压力/oracle BC 仅作上限探针。`test16` 保持未读。
**核心结论（当时口径）**：① E 在旧 dev1/Gate 下暂列最佳输入（E−D field_cb `+0.021`，负例最少），但已被顶部对抗性审查降级为临时候选；② W0 印证 coord_scale 编码尺寸非幅值；③ C 邻域不退化但被 `nsample=16` 截断抹平预期收益；④ L1 raw-Huber 单 seed 全面正、λ=1.0 最佳、无爆峰，但三 seed 确认后收益缩水、seed 脆弱；⑤ W3 组合阴性并停止组合线。历史数值详见[训练实验跟踪：第六轮 W0–W3 节](WSS最小化_训练实验跟踪.md)。

## 2026-07-12｜WSS 最小化预处理—训练全链路基础检查 ✅DONE

**本次主要修改**：
- 新增全链路基础检查报告，从原始读取、单位、配准/正交旋转、`[-1,1]` 各向同性缩放、节点 ID 对齐、无损压缩/FPS、训练特征/标签同索引、PointNeXt/激活/loss/超参数、完整点云评估逐项审计。
- 对 77 个 included bundle 做实际数值复核；同步在训练实验跟踪顶部加入结论入口。
- 本轮仅修改文档，不改预处理/训练代码与配置，不启动新训练，不读取新的 test16 指标。

**对应代码/文档**：
- 检查对象：`pipeline_wss_min/`、`training_wss_min/`、`data_wss_min/`、`training/splits/split_AG_wss_min_v1.json`
- 新报告：[WSS最小化_全链路基础检查报告_2026-07-12](WSS最小化_全链路基础检查报告_2026-07-12.md)
- 状态摘要：[训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：完成下一轮 C/E 严格尺度与 L1/L2 loss 前的基础审计 Gate；17 个现有单元测试通过，`compileall` 通过。

**当前状态判断**：当前 WSS 标量主线无致命坐标—标签错位；77/77 bundle 的正交旋转、逆变换、数组同长、ID/坐标守卫和采样同索引均通过。下一轮前必须优先处理/加固：3 个 included roll-sign 不可靠病例、毫米尺度真正进入 FPS/ball-query、速度路径 cell ID/裁剪同步守卫。GELU + 线性输出合理，不把激活替换作为首要提分项。

## 2026-07-12｜第六轮文档拆分与优先级重排（优先级已由 2026-07-13 再修订）

**本次主要修改**：
- 将原单篇“XYZ 尺度诊断+后续所有路线”重构为第六轮总入口和三份独立执行文档，分开横向多目标对比、WSS 精度突破、BC/速度条件路线。
- 冻结新优先级：先补齐壁面/内部压力与近壁 `u/v/w/|v|` 横向主表，再集中执行 WSS C/E、L1/L2、M1/M2；空余 GPU 可并行已预注册且互不依赖的 run。
- 保留 BC I0/I1/I2、入口裁剪分层和速度→WSS V0/V1/V2；明确横向速度基线不受 V0 oracle Gate 阻断，但 V2 仍必须过门。
- 吸收计划审核结论：C 改为 train-only 全局共享尺度及邻域 QA；L1/L2 固定 D 输入；压力/速度不再以 WSS top10 惩罚作通用选模规则；候选增加 grouped repeated validation。

**对应代码/文档**：
- [第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)
- [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md) / [WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md) / [BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md)
- `docs/README.md`、`training_wss_min/README.md`、`training_wss_min/configs/README.md`

**推进到实验步骤**：文档和执行调度重构；未修改代码/配置，未提交新作业，不中止已提交 Jobs `7551–7556`。

**当前状态判断**：横向 Track A 继续运行；Track B 必须先通过 adapter/坐标帧/指标 QA；WSS 训练主矩阵排在横向主表之后。test16 保持未读。

## 2026-07-12｜第六轮 横向对比 H-PW 压力-壁面：目标可切换 + 6 run 完训回填 ✅DONE

**本次主要修改**：
- 训练/评估管线支持 `target` 切换（此前 `target` 字段是摆设、硬编码 `wall_wss`）：
  - `dataset.load_case`/`load_partition` 增 `target` 形参；`target='pressure'` 读 `wall_pressure[peak]` 并**逐例去均值**（gauge/相对压力，可为负），复用 `normalize_wss`/`denormalize_wss` 的 `linear` 分支。
  - `train.py` 两处 `load_partition` 透传 `cfg.data.target`。
  - `evaluate.py`：`load_partition` 透传 target；**去掉对非 WSS 目标的 `clip(…,0,None)`**（gauge 压力可为负，仅 `method==log_z` 时裁剪）。
  - `tools/config_paths.py` 增 `r6_press_*` → `configs/multitarget/`。
- 新增 gauge-pressure 归一化 stats 生成器 `tools/make_pressure_stats.py` → `data_wss_min/fold_stats/pressure_gauge_stats_v2_dev1.json`（train-only/peak-only；53 例/707705 点；`method=linear`，mean≈0、std=529.2 Pa，gauge∈[-1880,1164]）。
- 新增配置生成器 `tools/make_configs_pressure.py` → `configs/multitarget/press_wall_{xyz,xyzgeom}_s{1234,7,2025}.json`（共 6）+ manifest `configs/sweeps/pressure_wall.txt` + protocol；相对 B1 control 仅改 `target=pressure`、stats 路径、`input_features`、`loss_weight_target=false`（纯 MSE，关掉 WSS 长尾加权）、name/seed。
- CPU 冒烟通过：gauge 目标逐例均值≈0、~40% 负值；normalize/denormalize 往返误差 6e-5；clip-gate 保留负值；负值下 metrics 不崩且可 JSON 序列化（`nrmse_mean` 因均值≈0 而巨大，属预期次要指标伪影，主指标 R² 正常）。

**对应代码/文档**：
- `training_wss_min/{dataset.py,train.py,evaluate.py}`、`tools/{config_paths.py,make_pressure_stats.py,make_configs_pressure.py}`
- `training_wss_min/configs/multitarget/`、`configs/sweeps/pressure_wall.txt`、`data_wss_min/fold_stats/pressure_gauge_stats_v2_dev1.json`
- [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[训练实验跟踪·第六轮](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：Track A 压力-壁面 6 个作业 `7551–7556` 全部 `COMPLETED (0:0)`，val-only 评估齐全。三 seed 均值——压力 `xyz` field/casemean `0.426±0.021 / 0.254±0.042`；压力 `xyz+geom` `0.531±0.053 / 0.509±0.015`（xyz+geom 下 0/8 失败例）。

**当前状态判断**：**同最小协议下压力比 WSS 好学得多**（xyz+geom 压力 0.531/0.509 vs WSS 0.311/0.197；xyz 压力 0.426/0.254 vs WSS 0.164/−0.085），压力 casemean 全程为正、xyz+geom 无失败例——空间压力型态比近壁剪切更由几何决定。但仍 `<0.70`，不外推其他目标。选模沿用 WSS 复合规则→列**探索性**；只读复核显示复合最优 epoch 与压力 R² 最优一致，数值应接近压力专用选模，正式复选待用户批准（预计不改数值）。Track B（压力-内部 + 近壁速度）待 data_new adapter。`test16` 保持未读。

## 2026-07-12｜第六轮 §14 多目标扩展（压力/速度）方案落地 ✅DONE（方案+口径已定，未执行）

**本次主要修改**：
- 按与老师讨论，向第六轮计划新增 §14：同最小协议（FPS-2000/PointNeXt-S/dev1/B1）下把目标从 WSS 换成压力+速度，做 `xyz`/`xyz+geom` 矩阵。
- 核对数据事实并写入方案：`wall_pressure` 已在 wss_min bundle（壁面即刻可跑，仅需 `load_case` 加 `target='pressure'` 分支）；**速度不在壁面（无滑移≈0）、wss_min bundle 无速度标签**，须走 `data_new/AG/**/result_features_merged-1162.pt` 的 `y=[u,v,w,p]`（15000 节点，已核对覆盖全部 84 个 AG 例）。
- **老师确认口径（已回填 §14）**：压力做**壁面+内部**；速度**只做近壁**；速度目标 = `u,v,w` 三分量 + `|v|` 幅值旋转不变对照。→ 6 目标 × 2 输入 × 3 seed = **36 runs**，分 Track A（压力-壁面，即刻可跑，6）+ Track B（data_new adapter：压力-内部 6 + 近壁速度 24）。
- 预注册逐目标归一化（压力不 log、建议逐例去均值；速度线性 z-norm + `|v|` 旋转不变对照）、target-weight 默认关、adapter 双采样域（内部全场 / 近壁带）要求与执行 Gate；**近壁速度与 §11 速度→WSS oracle 复用同一 adapter**。

**对应代码/文档**：
- [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[训练实验跟踪·第六轮](WSS最小化_训练实验跟踪.md)
- 关联既有：`training_wss_min/dataset.py:load_case`（现 `target` 字段未启用、硬编码 `wall_wss`）；`data_new/AG/**/result_features_merged-1162.pt`（`y=[u,v,w,p]`）

**推进到实验步骤**：仅方案落地，未提交作业。Track A（压力-壁面）无新数据即可实现；Track B 前置 data_new adapter（双采样域）。

**当前状态判断**：口径已定，等指令开工。建议先实现 Track A 压力-壁面（`target='pressure'` 分支 + 逐例去均值归一化）跑单 seed gate，再落地 adapter。`test16` 保持未读。
**当前状态判断**：**同最小协议下压力比 WSS 好学得多**（xyz+geom 压力 0.531/0.509 vs WSS 0.311/0.197；xyz 压力 0.426/0.254 vs WSS 0.164/−0.085），但不能外推其他目标。该批次沿用 WSS 复合选模，仅列为探索性历史基线；如需对外确认，只做压力专用 val-only 只读重评。Track B 已在最终审查后暂停，`test16` 保持未读。

## 2026-07-12｜第六轮 A/B/D 尺度诊断收口 + 新思路交叉验证 ✅DONE

**本次主要修改**：
- 复核首批 Jobs `7029–7037`：`7029–7032`(A×3+B_s1234) eval 完整；`7033`(B_s7) 训练成功但 eval 崩溃；`7034–7037`(B_s2025+D×3) 训练即失败。
- 定位根因：首批提交清单引用旧目录 `configs/round6/r6_scale_*.json`，而同日“目录规整”已把 config 迁至 `configs/xyz_scale_diag/scale_*.json` 并删 `round6/`；早启动作业赶在删除前解析成功，晚启动作业 `FileNotFoundError`。与训练协议/数据/模型无关，未污染已完成 run。
- 新增 `training_wss_min/cluster/run_eval_only.slurm`（仅对已有 `ckpt_best.pt` 的 run 重评，不重训）。
- 按正确 manifest `configs/sweeps/xyz_scale_abd.txt` 重跑 `7039–7043`，5 个均 `COMPLETED (0:0)`，**A/B/D 9/9 eval 齐全**，回填完整三 seed 结果与终裁。
- 对第六轮 §6–§12 各新思路做证据交叉核对，写入第六轮计划新增 §13（含已核实引用表与需重新掂量的 7 条先验/张力）。

**对应代码/文档**：
- `training_wss_min/cluster/run_eval_only.slurm`、`cluster/logs/resubmit_20260712_001011.txt`
- [第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md)（尺度结论 + 后续优化路由）
- [训练实验跟踪·第六轮](WSS最小化_训练实验跟踪.md)（完整 A/B/D 结果表 + 终裁）

**推进到实验步骤**：三 seed 均值——A(`xyz`) field/casemean `0.164±0.070 / −0.085±0.055`；B(`xyz+coord_scale`) `0.208±0.038 / +0.008±0.099`；D(`xyz+geom`) `0.311±0.014 / +0.197±0.020`。B−A field `+0.044`/casemean `+0.093` 均过 `+0.02` 门槛且逐 seed 方向一致（3/3）。

**当前状态判断**：**尺度信号 B−A 成立**——逐病例归一化确丢失有用物理尺度，`coord_scale` 稳定回收一部分（casemean 由负转正）；但只补回约三成缺口，**几何仍是压倒性主杠杆**（D−B field +0.103）。与第四轮 C4「coord_scale No-Go」不矛盾（C4 是在已含 `local_radius` 的 geom 上加、冗余）。绝对精度仍全面未达 0.70。用户批准下一步：`L1/L2 loss` 与 `C/E 严格尺度`并列先行，`M1/M2/M3` 次之，速度路线仅做 V0/V1 oracle。`test16` 保持未读。
**当前状态判断**：旧开发口径下 B−A 的尺度信号成立，但旧 `casemean` 不是病例 level R²；几何仍是主要增量（D−B field +0.103）。本条只保留当轮实验裁决，后续顺序已由 2026-07-13 顶部最终审查覆盖；`test16` 保持未读。

## 2026-07-12｜第五轮止损归档 + 第六轮后续路线预注册 ✅DONE

**本次主要修改**：
- 将第五轮口径冻结为“诊断性科学结案 / 内部工程 No-Go”；用户确认无达标候选时不强行运行 L0/15-run OOF/T16，临床/生产验证后置。
- 新增第五轮归档说明，明确原正式计划与实际止损的偏差；保留原文档/产物路径作审计证据。
- 扩展第六轮计划：补 C 严格物理 XYZ、E 嵌套因子组、BC 可辨识性、global-local/density-robust/残差/小模型矩阵、raw/case-balanced loss 矩阵、生成模型边界和速度→WSS oracle Gate。

**对应代码/文档**：
- [第五轮正式计划](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md) / [历史执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [结案归档说明](_archive/WSS最小化/WSS最小化_第五轮结案与归档说明_2026-07-12.md)
- [第六轮 XYZ 尺度诊断计划与执行](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md) / [文档索引](../README.md)

**推进到实验步骤**：文档归档与后续预注册；不改当前 A/B/D Jobs `7029–7037` 的任何 config/Gate，不提交 C/E/架构/loss 新作业。

**当前状态判断**：第五轮已按止损口径归档，`test16` 保持未读；第六轮先等 A/B/D 九个 run 完整返回并交叉审查，再由用户批准下一阶段。

## 2026-07-12｜training_wss_min 目录规整（语义 configs + tools/experiments） ✅DONE

**本次主要修改**：
- `configs/` 按主题分目录：`baseline_sweep` / `loss_aug_ablation` / `clean_data` / `protocol_gates` / `pointcount_curve` / `fit_lc_diagnosis` / `xyz_scale_diag` / `sweeps`；去掉 `round5`/`round6` 目录名与文件名前缀。
- JSON 内 `name` 与既有 `runs/<name>/` **未改**（含 `r5_*`/`r6_*`），保证在跑/历史实验可复评。
- 根目录脚本分层：`tools/`（生成器/汇总/导出）与 `experiments/`（诊断）；旧 `-m` 入口保留兼容 shim；历史 `run_round5_*.slurm` 移入 `cluster/archive/`。

**对应代码/文档**：
- `training_wss_min/configs/README.md`、`training_wss_min/README.md`
- `training_wss_min/tools/`、`training_wss_min/experiments/`、`training_wss_min/tools/config_paths.py`
- [第六轮计划](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md) 活跃路径已同步

**推进到实验步骤**：工程整理；不改训练协议、不重跑作业。

**当前状态判断**：活跃入口为 `configs/sweeps/xyz_scale_abd.txt` 与 `python -m training_wss_min.tools.make_configs_xyz_scale`；验收：锚点 `name` 与 `runs/` 一致，`python -m training_wss_min.config` 通过。

## 2026-07-12｜A0E-ctrl 两例 postview 面片包（CFD|Pred|Error） ✅DONE

**本次主要修改**：
- 按 `postview-surface-viz` 默认口径，用第五轮最新最佳锚点 `r5_a0e_b1_ctrl_s1234` 对 `slow/WU_FENG_YAN`、`fast/RAN_QING_BO` 做完整壁面推理与 STL 回插。
- 新增导出脚本 `training_wss_min/export_wss_postview.py`（wall CSV → Gaussian r=3/sharpness=2 → `*__surface_wall.vtp` + 三联图）。

**对应代码/文档**：
- `training_wss_min/export_wss_postview.py`
- `training_wss_min/cluster/run_postview_a0e_ctrl.slurm`
- `docs/03-汇报材料/figures/WSS最小路线_20260712/postview_a0e_ctrl/`
- [图表说明与汇报口径](../03-汇报材料/figures/WSS最小路线_20260712/图表说明与汇报口径.md)

**推进到实验步骤**：汇报可视化交付；映射覆盖率两例均为 100%；同点 wall R²：`WU_FENG_YAN=0.3225`、`RAN_QING_BO=0.4959`（均为 train，不作泛化结论）。

**当前状态判断**：ParaView 主入口为各例 `*__surface_wall.vtp`（含 `wss_cfd/wss_pred/err_wss/abs_err_wss`）；正式指标只读 `_export/*__wall.csv`。

## 2026-07-12｜第六轮 A/B/D XYZ 尺度诊断已预注册并提交 ⏳RUNNING

**本次主要修改**：
- 新增第六轮配置生成器和机器可读预注册；A=`xyz`、B=`xyz+coord_scale`、D=`xyz+abscissa/local_radius/curvature`。
- 9 个配置逐字段继承 B1/dev1/FPS-2000 协议，除输入特征和 `seed={1234,7,2025}` 外无其他变量。
- 同步修正 LC 外推口径：当前证据只支持“61 例范围内 40→53 无可辨识 field 增益”，不支持“几百/几千例数据无用”。

**对应代码/文档**：
- `training_wss_min/make_configs_round6_scale.py`
- `training_wss_min/configs/round6/` / `configs/sweep_round6_scale_abd.txt`
- [第六轮 XYZ 尺度诊断计划与执行](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md) / [training README](../../training_wss_min/README.md)

**推进到实验步骤**：配置不变量审计和 B 组特征构造通过；Jobs `7029–7037` 已提交，提交记录 `submitted_20260712_120420.txt`；默认 val-only，未访问 test16。

**当前状态判断**：本轮只允许判断 B−A 的尺度信号和 D−B 的显式几何增量；B 不是严格物理尺度纯 XYZ 终审。待 9 个 run 完成后按三 seed 均值±sample std 裁决是否实现 C 组。

## 2026-07-12｜第五轮 CFD 审计 + F0 结案 ✅DONE

**本次主要修改**：
- 子智能体完成 §6 CFD 可信性审计（read-only，dev61）；据全轮证据写 F0 结案报告。

**对应代码/文档**：
- `training_wss_min/runs/_round5/cfd_audit/{cfd_audit.py,cfd_per_case.csv,cfd_summary.json,cfd_audit_report.md}`
- `training_wss_min/runs/_round5/final_report/round5_final_report.md`（F0）、`round5_status_synthesis.md`（SUPERSEDED）
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：CFD 审计——peak 相位固定步统一、近壁 QA 全清、高 WSS 为真实几何热点、复现 floor ~2%（R²_cap ~0.92–0.96）；独立证实 `HOU_SHEN_QIAN=KANG_XI_MING` 同一几何且均 dev1 train。

**当前状态判断**：当前 61 例、当前几何-only 输入与 PointNeXt 协议下的 ~0.31 平台主要不是 CFD 标签噪声。§8.1 机制问题全部有可复核裁决 → **第五轮科学结案**；dev1 ~0.31/0.21 ≪ 0.70 → **未达内部工程目标**（§8.2），未跑 OOF。该结论不外推到几百/几千个高质量独立病例的极限。

## 2026-07-12｜第五轮 B-BC 资产审计：可部署 BC 杠杆关闭 ✅DONE

**本次主要修改**：
- 子智能体完成 read-only BC 资产审计（未训练、未改 data_new）：解析 61 dev 病例的 udf-inlet.c、vf-in、5 路压力监测、RCR。

**对应代码/文档**：
- `training_wss_min/runs/_round5/bc_audit/{audit_ag_bc.py,bc_per_case.csv,bc_summary.json,bc_audit_report.md}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)；G2 产物原记录路径为 `training_wss_min/runs/_round5/branch_experiments/branch_decision_g2.md`，当前工作区未保留该文件。

**推进到实验步骤**：61/61 dev 覆盖完整；Fourier 入口模板跨病例逐字节相同、`Q` 与面积无关（入口流量 CoV≈1.85e-4）；top-3 方差全为 oracle RCR（CoV 0.55–0.67）。

**当前状态判断**：**入口流量是共享人群模板、非病人特异；唯一 per-case 变化的 BC（出口 RCR/压力/流量分配）全部 `oracle_non_deployable`。可部署 B-BC 杠杆关闭。** RCR 仅是尚待病例外负对照验证的信息候选，不能从资产方差直接推断为跨病例 WSS 差异的根因；后续只允许按顶部最终审查的 2×2 oracle 探针量化。

## 2026-07-12｜第五轮 LC 收口：当前 61 例范围内暂时平台 ✅DONE

**本次主要修改**：
- 完成 LC stage-2（seed 7/2025）并用 `summarize_round5_lc.py` 汇总 30 个 run，产出曲线/方差分解/图/verdict。
- 新增 LC 汇总器 `training_wss_min/summarize_round5_lc.py` 与报告 `learning_curve_report.md`。

**对应代码/文档**：
- `training_wss_min/summarize_round5_lc.py`
- `training_wss_min/runs/_round5/learning_curve/{learning_curve_report.md,lc_points.csv,lc_curve.png,lc_verdict.json}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：field R² 均值曲线 `13→0.258 / 26→0.297 / 40→0.312 / 53→0.310`；配对增量 `13→26 +0.039`、`26→40 +0.014`、`40→53 −0.002±0.049`。端点 53 逐 seed `0.359/0.252/0.319`（std 0.044）。

**当前状态判断**：**field R² 在当前观测范围内约 40 例后进入 ~0.31 暂时平台，40→53 增量与 0 不可分**。stage-1 单 seed"仍在上升"被证明是 s1234 端点偏高伪影（多 seed 纠正）。该 LC 只覆盖 13–53 例，不能用于否定几百/几千例高质量数据的潜在收益；它仅支持在现有 61 例池中不再继续小步扩展同配方 LC。

## 2026-07-12｜第五轮 A0E 重锚 + G1 裁决 + LC 启动 ✅DONE / ⏳LC 运行中

**本次主要修改**：
- 新增 A0E dev1 control 重锚：两支 val-only 训练（`ctrl` B1 逐字 min_lr=1e-5、`nsl` 仅抬 LR 下限到 2e-4），判定历史 `0.34` 是否为训练预算伪影。
- 完成 G1 第二次分支裁决：合并 A0D/A0E/A0R/A1 证据，批准 LC、关闭 B-REP、维持 B-DEN/B-BC BLOCKED。
- 新增确定性 LC 生成器 `make_configs_round5_lc.py`：3 链嵌套分层子集 + per-subset 划分/WSS stats + B1 配方 config；提交 stage-1 seed1234 的 9 个子集训练。

**对应代码/文档**：
- `training_wss_min/make_configs_round5_lc.py`、`training_wss_min/configs/round5/a0e_b1_{ctrl,nsl}_s1234.json`、`training_wss_min/configs/round5/lc/`
- `training/splits/split_AG_wss_min_v2_dev1_lc_*.json`、`data_wss_min/fold_stats/wss_stats_v2_dev1_lc_*.json`
- `training_wss_min/runs/_round5/a0e_control/a0e_control_report.md`、`training_wss_min/runs/_round5/branch_experiments/branch_decision_g1.md`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A0E `ctrl` Job `6992` field/casemean `0.3587/0.2300`（复现 anchor），`nsl` Job `6993` `0.3419/0.2125`（更差、2 例负 R²）；G1 `DONE`；LC stage-1 Job `6994–7002` 运行中，LC53 端点复用 ctrl。

**当前状态判断**：`0.34/0.23` 经 A0E 确认为真实泛化上限、非预算伪影；非饿死 LR 假设被证据推翻，LC 改用标准 B1 schedule（`如果结果不合理则修改` 的一次实际更正）。瓶颈定性维持"泛化受限 + 目标函数与 raw-R² 错位"。下一步等 LC 曲线判读增加同分布 AG 数据的边际收益。

## 2026-07-12｜第五轮 A0D 基础拟合链分解 ✅GO

**本次主要修改**：
- 并行完成 normalization/loss、点—标签对齐/可辨识性、optimizer/AMP/BatchNorm 三路只读审计。
- 新增 A0D cluster-only 简化 overfit runner/config/Slurm，按四个 single-case→four-case shared→target-weight-only 串行执行。
- 将训练预算从易受病例数影响的 epoch 口径明确为 optimizer steps，识别旧 micro 仅 160 updates 的口径缺陷。

**对应代码/文档**：
- `training_wss_min/{a0d_simple_overfit.py,a0d_target_weight_only.py}`
- `training_wss_min/configs/round5/a0d_*.json`
- `training_wss_min/cluster/run_round5_a0d_*.slurm`
- `training_wss_min/runs/_round5/a0d_fit_chain/`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：Jobs `6986/6990/6991` 均完成。single-case R² `0.991–0.999`，four-case shared plain MSE field/casemean `0.979526/0.982533`，target-weight-only `0.993723/0.992423`。

**当前状态判断**：A0D `GO`。旧 `R²_casemean=0.844` 不是拟合上限；target-weight 单独不是缺口原因。下一步应冻结 optimizer-step 预算后做一个 dev1 val-only 候选，不启动 LC/B-DEN/test16。

## 2026-07-11｜第五轮 B-REP/C2 global context ⛔NO_GO / 已列候选止损

**本次主要修改**：
- 新增 B1 PointNeXt 解码特征的病例内 mean+max global context，广播拼接后仍由单输出 head 预测 WSS。
- 新增 C2 cluster-only 串行 runner/config/Slurm，以及前后向和病例隔离 smoke test。

**对应代码/文档**：
- `training_wss_min/{pointnext_global_context.py,brep_c2_global_context.py}`
- `training_wss_min/configs/round5/brep_c2_global_context_*.json`
- `training_wss_min/cluster/run_round5_brep_c2_global_context_{micro,dev1}.slurm`
- `training_wss_min/runs/_round5/branch_experiments/global_context/`

**推进到实验步骤**：Job `6984` `COMPLETED (0:0)`，last field/casemean `0.740682/0.695389`，未达 `0.95/0.95`，dev1 未提交。

**当前状态判断**：C2 `NO_GO`；M1/C1/C2 均未过 micro Gate，STL 特征又被 A1 mapping 阻塞。暂停 B-REP 训练扩展、LC 和 B-DEN，下一步返回 normalization/loss/标签对齐与几何可辨识性审计。

## 2026-07-11｜第五轮 B-REP/C1 radius-normalized relative position ⛔NO_GO

**本次主要修改**：
- 新增与 B1 同拓扑/同初始权重的 PointNeXt C1 候选，唯一改动是 SA/InvRes 局部 MLP 中的相对 xyz 除以对应层 radius。
- 新增 cluster-only 串行 micro/dev1 runner、config、Slurm 和唯一变量/前后向预检。

**对应代码/文档**：
- `training_wss_min/{pointnext_radius_norm.py,brep_c1_radius_norm.py}`
- `training_wss_min/configs/round5/brep_c1_radius_norm_*.json`
- `training_wss_min/cluster/run_round5_brep_c1_radius_norm_{micro,dev1}.slurm`
- `training_wss_min/runs/_round5/branch_experiments/radius_norm/`

**推进到实验步骤**：Job `6983` `COMPLETED (0:0)`，last epoch 159 的 field/casemean 为 `0.803401/0.760602`，均未达 `0.95`，dev1 未提交。

**当前状态判断**：C1 `NO_GO`。B-REP 进入正式计划中最后一个已列候选 C2 单输出 global context，仍先跑四病例 micro。

## 2026-07-11｜第五轮 B-REP/M1 逐点 MLP micro ⛔NO_GO

**本次主要修改**：
- 新增不读取 `pos/batch` 的逐点 MLP 对照、cluster-only 串行 runner、micro/dev1 config 与 Slurm 脚本。
- 先执行锁定四病例 micro；预注册规则要求两项 train R² 均 `>=0.95` 才允许 dev1。

**对应代码/文档**：
- `training_wss_min/{point_mlp.py,brep_mlp.py}`
- `training_wss_min/configs/round5/brep_m1_mlp_*.json`
- `training_wss_min/cluster/run_round5_brep_m1_mlp_{micro,dev1}.slurm`
- `training_wss_min/runs/_round5/branch_experiments/mlp/`

**推进到实验步骤**：Job `6982` `COMPLETED (0:0)`，last epoch 159 的 field/casemean 为 `0.880817/0.844018`，均未达 `0.95`，dev1 未提交。

**当前状态判断**：M1 `NO_GO`。候选 2 STL/表面特征受 A1 mapping `0/8` 阻塞；B-REP 按顺序进入 C1 相对位置 radius 归一化的四病例 micro。

## 2026-07-11｜第五轮 A1 + G0 密度/映射证据与首次分支裁决 ✅DONE

**本次主要修改**：
- 新增 A1 可复跑审计入口，完成三 seed 同索引 density probe、逐层邻域 cap、8 例 STL mapping、FPS-2000 真值 IDW3 oracle 与推理重复性检查。
- G0 合并 A0R/A0M/A1 证据，只批准 B-REP 的第一个单变量候选（逐点 MLP），拒绝当前 B-DEN/LC/B-BC。

**对应代码/文档**：
- `training_wss_min/a1_density_surface.py`
- `training_wss_min/runs/_round5/a1_density_surface/`
- `training_wss_min/runs/_round5/branch_experiments/branch_decision.md`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A1 `NO_GO`、G0 `DONE`；IDW3 oracle 通过，但 mapping 总 Gate `0/8`，因此 D1/D2 被硬阻断。

**当前状态判断**：基础拟合不足是首要可执行问题；密度敏感同时存在，但映射协议未过，不能先做回插/ensemble。B-REP MLP 候选先跑四病例 micro sanity，未过则不进入 dev1 Gate-1。

## 2026-07-11｜第五轮 A0R B1 多 checkpoint 只读诊断 ✅DONE

**本次主要修改**：
- 新增 A0R 可复跑脚本，对 B1 三 seed 的 best/last/已有 candidate 共 15 个 checkpoint 执行 train/val × canonical-2000/full 四象限只读评估。
- 产出 60 行汇总、1830 行逐病例、42 行对比和机器可读摘要；未训练、未访问 test16。

**对应代码/文档**：
- `training_wss_min/runs/_round5/a0_readonly/{run_a0_readonly.py,rerun.sh,*.csv,summary.json,a0_readonly_report.md}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A0R 已完成。best canonical train field/casemean 仅 `0.538–0.573 / 0.494–0.528`；last 将 train field 平均提升 `0.070` 时，canonical val field 平均下降 `0.061`。

**当前状态判断**：继续训练不能同时修复 train-fit 与 val 泛化；full 相对 canonical 在三 seed 中一致劣化，但密度分支是否可行仍需 A1 oracle/mapping 裁决。

## 2026-07-11｜第五轮 A0M 四病例 micro-overfit ⛔NO_GO

**本次主要修改**：
- 新增 train-only 四病例锁定/验证/训练入口、独立 config/协议与 Slurm 脚本；正式训练只能在 Slurm 环境中启动。
- 从 dev1 正式 train 内按 fast/slow × WSS 均值最低/最高锁定 `SUN_ZHI_YU`、`WANG_DAO_CHUN`、`ZANG_YU_SHU`、`MA_TIAN_YI`；不允许看结果后更换。
- Job `6981` 用 B1 / fixed FPS-2000 / seed1234 训练 160 epoch，仅使用 last checkpoint 计算 train-fit，未加载 val/test。

**对应代码/文档**：
- `training_wss_min/a0_micro_overfit.py`
- `training_wss_min/configs/round5/a0_micro_{protocol,b1_s1234}.json`
- `training_wss_min/cluster/run_round5_a0_micro.slurm`
- `training_wss_min/runs/_round5/a0_micro/`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A0M 已完成；Job `6981` `COMPLETED (0:0)`，`R²_field_raw=0.76548`、`R²_casemean=0.72381`，均低于 `0.95`。

**当前状态判断**：A0M `NO_GO`；按硬停止条件暂停 LC 和大规模 sweep，待 A0R/A1 完成后由 G0 裁决是否进入 B-REP。

## 2026-07-11｜第五轮 P0 评价协议与 Gate 实现 ✅DONE

**本次主要修改**：
- 新增病例等权全场 `R²_field_casebalanced`，并补充逐病例 R² median/P10/负 R² 数/失败率。
- 将第五轮 Gate 固化为 field/casemean 两项主 R² 同时超过 `0.02` 才可 Go，取消旧高 WSS 单路径 Go；top10 ratio/IoU 下降超过 `0.05` 会阻断 Go。
- 对评估 CLI 增加 test guard；开发默认 val-only，读取 test 必须显式传入 `--allow-test`。
- 新增 case-balanced 手算、常数标签、单病例、点数悬殊、共同 Gate 和 test guard 回归测试。

**对应代码/文档**：
- `training_wss_min/{metrics.py,evaluate.py,gate1_compare.py,tests/test_round4_protocol.py,README.md}`
- `training_wss_min/runs/_round5/protocol/{protocol_report.md,protocol_regression.json}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：P0 Gate-0 已通过；17 项聚焦测试通过，旧 B1 seed1234 val-only best checkpoint 回归的 field/casemean/MAE 最大偏差 `3.24e-9 < 1e-8`。

**当前状态判断**：P0 `DONE`，未访问 test16，未提交新训练。A0R/A0M/A1 可按任务卡并行；G0 仍被三者联合产物阻塞。

## 2026-07-11｜第五轮优化计划收敛为正式版 ✅已定稿

**本次主要修改**：
- 将 1088 行终审与历轮交叉审查轨迹收敛为正式优化计划，删除审查者过程、重复提案、已被覆盖口径和过度详细的实现草案。
- 保留最终科学边界、当前基线、共同主指标、容量/密度/learning curve/BC 分支、CFD 可信性审计、全 61 例 OOF 和完成标准。
- 明确文档分工：正式优化计划负责科学与技术口径，独立执行计划负责智能体任务 ID、依赖和产物。
- 同步修复执行计划中指向旧 §18 的链接和文档索引。本次未修改训练代码、split、stats 或 manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)
- [WSS最小化_第五轮执行计划.md](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)
- [docs/README.md](../README.md)

**推进到实验步骤**：优化计划已正式定稿；执行状态仍从 P0 `NOT_STARTED` 开始。

**当前状态判断**：项目现在有一份简洁的正式方案和一份独立的智能体执行计划，两者职责分离，后续不需要从历轮审查文本中重新解释最终口径。

## 2026-07-11｜第五轮终审稿转为智能体执行计划 ✅已定稿 / 未启动

**本次主要修改**：
- 保留第五轮优化计划为决策与审查轨迹，新建第五轮执行计划作为后续智能体的唯一交接入口。
- 将终审 §18.9 拆为 P0、A0R、A0M、A1、G0、B-REP/B-DEN/B-BC、LC、L0、OOF、T16 和 F0 任务卡，为每张卡固定依赖、允许改动、产物、Gate 和停止条件。
- 增加共享工作区的领取/回填规则、统一 `_round5/` 证据目录、最小报告模板和并行边界，避免多智能体同时改动同一文件或跳过裁决。
- 同步更新文档索引和终审稿状态指针。本次未修改训练代码、split、stats 或 manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮执行计划.md](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)
- [docs/README.md](../README.md)

**推进到实验步骤**：第五轮已从终审阶段转为可分派任务卡；第一个可领取任务为 P0，本次未开始 P0。

**当前状态判断**：文档已具备多智能体顺序交接条件。P0 通过后可分派 A0R/A0M/A1；G0 之前不允许启动条件性模型分支或 learning curve。

## 2026-07-11｜第五轮计划 v1.0：Codex 最终审定 ✅有条件通过 / 待执行批准

**本次主要修改**：
- 完成第五轮最终审定，新增 §18 作为唯一执行入口；旧 §0–§17 保留审查轨迹，但执行时不得自行择取冲突口径。
- 发现 dev1/dev2/dev3 的 val 并集仅 20 个独立病例、41/61 从未作 val；终审否决用该并集声明 0.70，改为配置锁定后的全 61 例 5-fold OOF ×3 seeds 内部工程评估。
- A0 增加 best/last/top-k train-fit 对照与 4 病例 micro-overfit sanity，避免把早停/选模问题误判为模型容量不足。
- A0.5 BC probe 降为条件候选：入口 UDF 抽查发现固定分母，与 §17 的逐病例 `Q/A_inlet` 表述不一致；RCR/流量分配统计需先补可复跑脚本、CSV 和元数据；`B3−A` 只解释为预测增量，不作因果/全部方差表述。
- 区分第五轮科学结案、内部工程达标和未来前瞻性确认；R² 未达 0.70 可科学结案，但不得标记工程达标。
- 本次仅更新计划与推进记录，未修改训练代码、split、manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)（当时终审结论现已并入正式版）
- 当前训练证据：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：终审有条件通过，仍处于 No-Run；等待用户明确批准后从 §18.9 第 1 步开始。

**当前状态判断**：计划已从多轮审查草案收敛为唯一执行顺序。当前首要任务仍是能力边界诊断而非扩大 sweep；工程 0.70 需全 61 例 OOF 内部评估，未来临床/生产声明仍需新 AG 前瞻性确认。

## 2026-07-11｜第五轮计划 v0.2：第二轮交叉审查 + 工程精度目标裁决 ⏳待执行批准

**本次主要修改**：
- 对第五轮 v0.1 与第一轮外部审查做第二轮代码/统计/密度/医工交叉核对；纠正“第四轮 B1 为纯 casemean 选模”的误读，实际为 `r4_composite_v1` 复合规则。
- 根据用户工程目标冻结：第五轮只做 AG 生长队列、peak-WSS 单标量；AAA 破裂队列独立；不要求每例 R²≥0.7，但工程目标要求逐病例完整壁面 `R²_casemean` 与 pooled `R²_field_raw` 向 0.70 验收。
- 将 A0 train-fit 提升为最高优先级；当前 B1/v2_dev1 重做 density probe 后才允许 D1/D2；全部 STL 用于表面连接/插值前 QA。
- learning curve 改为三条独立嵌套链，显式估计病例选择、模型 seed 和 dev split 三类噪声；不自动触发 AAA/ILO 混入 AG。
- AG 科学结案与 AAA/ILO 数据治理解耦；第五轮仍为 No-Run，未修改训练代码、split、manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)（历史审查口径现已并入正式版）
- [WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md) / [新队列数据可用性审计_AAA_ILO_2026-07-10.md](新队列数据可用性审计_AAA_ILO_2026-07-10.md)

**推进到实验步骤**：推进到 v0.2 审查裁决与工程验收定义；未进入 A0/A1，等待用户明确批准执行。

**当前状态判断**：目标 `R²≈0.70` 相对当前 field≈0.34 / casemean≈0.19 属结构性提升，必须先用 train-fit 与当前 B1 density 四象限判断能力边界，再决定 learning curve 或模型修复；不适合直接扩大 sweep。

## 2026-07-10｜第五轮优化计划 v0.1：冻结 AG 单队列、WSS 标量与 field-R² 主目标 ⏳待交叉审查

**本次主要修改**：
- 新建第五轮优化计划讨论稿，明确状态为“待其他智能体交叉审查、未授权执行”；本次未修改训练代码、split、数据 manifest，未提交任何实验。
- 冻结第五轮边界：AG 为唯一主实验队列；AAA/ILO 分队列治理且异常单元先隔离；任务保持峰值收缩期 WSS 单标量、单输出头；完整壁面 `R²_field` 为主指标。
- 把第五轮执行顺序收敛为：三队列清单治理 → AG train-fit/density 诊断 → AG 13/26/40/53 learning curve → 条件性单任务结构优化 → 医工交叉审计。
- 纳入新队列讨论中发现的待隔离项：`ILO/LIU_BAO_JUN-0/after` 近零 WSS、7 个 `vf-in` 数量级/口径异常单元、方向 watch 和跨队列病人分组风险。

**对应代码/文档**：
- 新计划：[WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)（后续收敛为正式版）
- 证据：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md) / [新队列数据可用性审计_AAA_ILO_2026-07-10.md](新队列数据可用性审计_AAA_ILO_2026-07-10.md)

**推进到实验步骤**：仅推进到第五轮 v0.1 计划冻结与交叉审查入口；未进入 Stage A，未授权执行。

**当前状态判断**：第五轮的首要目标不是扩大模型或混入新队列，而是先在 AG 上区分病例数、训练/推理密度、局部表示和 CFD 标签上限。交叉审查完成并经用户批准前，本文只作为讨论基线。

## 2026-07-10｜第四轮执行计划归档 ✅已完成

**本次主要修改**：
- 核对第四轮实际产物后确认：Stage A–C 与 §14 点数首轮曲线已完成；Stage D 的多 seed、重复 dev split 和最终 legacy test 尚未执行。
- 将计划从当前目录移入 `_archive/WSS最小化/`，更名为 `WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md`，并在文首明确归档边界，避免将“计划阶段完成”误写成“最终模型确认完成”。
- 同步更新 `docs/README.md`、`training_wss_min/README.md`、WSS 训练跟踪和归档索引；实验数据、run 目录与 Slurm 日志未移动。

**对应代码/文档**：
- 归档：[WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)
- 当前状态源：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md) / 本记录

**推进到实验步骤**：第四轮计划执行阶段结案；如继续推进，从 Stage D 的 1000/2000 多 seed确认开始，不再回到已判 No-Go 的单 seed配置。

**当前状态判断**：第四轮基础几何、协议修复和点数首轮探索基本完成；尚未形成跨 split、跨 seed、legacy test 的最终确认性结论。

## 2026-07-10｜点数曲线多 seed 确认完训（1000 vs 2000）✅已分析

**本次主要修改**：
- Jobs 6976–6979 完训；与 seed1234（6969/6960）合并为三 seed 对照。
- 产出 `runs/_summary_round4_pointcount/pointcount_multiseed_{1000_vs_2000.csv,summary.csv,1000_vs_2000.png,verdict.json}`。

**三 seed 均值 ± sample std（val）**：

| n | R²_field | R²_casemean | top10 | IoU | score |
|---:|---|---|---|---|---|
| 1000 | 0.337±0.017 | **0.221±0.019** | 0.413±0.032 | 0.327±0.011 | 0.276±0.025 |
| 2000 | 0.339±0.020 | 0.188±0.025 | 0.416±0.025 | 0.324±0.016 | 0.259±0.026 |

**判读**：
- 仅 **R²_casemean** 三 seed 一致且幅度超过 seed 噪声（均值Δ=+0.033）偏向 1000。
- R²_field / top10 / IoU / score：**方向不一致或持平** → 不能把默认点数改为 1000。
- **协议锚点维持 2000**；1000 保留为更稀采样候选（病例均衡更好）。稳健「最佳点数」若需要，再上 dev2/dev3。

**对应代码/文档**：计划 §14.5；训练跟踪文首；`training_wss_min/README`；方案快照完成定义；本记录。

**推进到实验步骤**：§14 第二阶段完成；第三阶段（dev2/dev3）按需，非必须。

**当前状态判断**：单 seed「1000 全面领先」已证伪；点数选择上 1000≈2000（field），1000 略优（casemean）。

## 2026-07-10｜点数曲线多 seed 确认已提交（1000 vs 2000）✅已完训

**本次主要修改**：
- 生成 `r4_dev1_pc_xyzgeom_fps1000_s{7,2025}`；2000 复用已有 B1 `s7/s2025` 配置。
- 提交 Jobs **6976–6979**（val-only）：1000×{7,2025} + 2000×{7,2025}；seed1234 已有 6969/6960。

**对应代码/文档**：`make_configs_round4_pointcount.py`（multiseed manifest）；`configs/sweep_round4_pointcount_multiseed.txt`；计划 §14.5；训练跟踪。

**推进到实验步骤**：已完训并转入上一条分析。

**当前状态判断**：见上一条。

## 2026-07-10｜第四轮 xyz+geom 点数—精度曲线完训汇总 ✅已分析

**本次主要修改**：
- Jobs 6969–6972 完训；与 B1-2000 / B3-4000 合并为 6 点曲线；覆盖审计 Job 6968 落盘。
- 新增汇总脚本 `summarize_round4_pointcount.py`，产出 `runs/_summary_round4_pointcount/`（metrics CSV、R²/尾部图、density audit、verdict JSON）。

**关键数字（val / seed1234 / B1 配方）**：

| n | R²_f | R²_c | top10 | IoU | score |
|---:|---:|---:|---:|---:|---:|
| **1000** | 0.351 | **0.241** | **0.448** | 0.330 | **0.301** |
| 1500 | 0.327 | 0.208 | 0.408 | 0.296 | 0.262 |
| 2000 | 0.351 | 0.214 | 0.436 | 0.330 | 0.283 |
| 3000 | 0.298 | 0.150 | 0.365 | 0.315 | 0.205 |
| 4000 | 0.259 | 0.207 | 0.346 | 0.296 | 0.227 |
| 6000 | 0.335 | 0.207 | 0.412 | 0.343 | 0.268 |

**判读**：
- 精度峰值（composite）= **1000**；近似平台仅含 1000。2000 与 1000 的 R²_field 持平，但 casemean/top10/score 更低。
- 曲线**非单调**：3000/4000 下凹，6000 回升仍不及 1000/2000 → 「堆点数」在本配方下无稳定收益。
- 覆盖随 n 升（fixed high-hit 约 5%→25%），精度峰值却在最稀端 → 更像密度/邻域错配，而非单纯标签覆盖不足。
- **单 seed 边界**：不得写成「1000 已是最优」；下一步对 1000 vs 2000 补 seed7/2025。

**对应代码/文档**：`summarize_round4_pointcount.py`；`runs/_summary_round4_pointcount/`；`runs/_audit/coverage_audit_v2_dev1.*`；计划 §14.3–14.4；训练跟踪文首；`training_wss_min/README.md`。

**推进到实验步骤**：§14 第一阶段（单 seed 完整曲线）完成；进入多 seed 确认（峰值 1000 × 锚点 2000）。

**当前状态判断**：已回答「geom 下是否越多点越好」——至少在 dev1/seed1234 上**不是**；2000 仍是历史协议锚点，但单 seed 峰值在 1000。

## 2026-07-10｜第四轮点数—精度曲线最小矩阵已提交 ✅已完训

**本次主要修改**：
- 生成 B1 配方点数横扫配置：`r4_dev1_pc_xyzgeom_fps{1000,1500,3000,6000}_s1234`（仅改 `wall_n_points`）。
- 提交 GPU Jobs **6969–6972**（val-only）；复用 B1-2000 Job 6960、B3-4000 Job 6962。
- 提交 node03 覆盖审计 Job **6968**：`v2_dev1 + fold stats`，k=`1000…6000` → `runs/_audit/coverage_audit_v2_dev1.*`。

**对应代码/文档**：
- `training_wss_min/make_configs_round4_pointcount.py`
- `training_wss_min/configs/sweep_round4_pointcount_s1234.txt` / `sweep_round4_pointcount_map_s1234.txt`
- `training_wss_min/cluster/run_coverage_audit_dev1.slurm`
- 计划 §14.3、训练跟踪文首、本记录。

**推进到实验步骤**：已完训并转入上一条汇总分析。

**当前状态判断**：见上一条。

## 2026-07-10｜第四轮点数—精度曲线补充方案归档 + 审计边界修正 ✅已转执行

**本次主要修改**：
- 将导师复核后的问题收敛为一个单变量补充实验：固定第四轮 B1 单头 `xyz+基础几何` 配方，横扫 `1000/1500/2000/3000/4000/6000` 点，画 `R²_field/R²_casemean`—点数曲线。
- 修正原计划的结论边界：B3 只证明 4000 在 dev1/seed1234 下不如 2000，不证明 2000 已是最佳点数；原“4000 No-Go 则不开 6000”只是算力 Gate。
- 明确本组不加辅助头、新几何特征、新 loss 或新采样器；先建立简单单头模型的点数平台。
- 记录覆盖审计的实际口径缺口：旧审计使用 v1 split/全局 stats，补充曲线执行前必须按 `v2_dev1 + fold stats` 重跑。

**对应代码/文档**：
- 归档新增：[`_archive/WSS最小化/WSS最小化_第四轮补充实验_点数精度曲线方案快照_2026-07-10.md`](_archive/WSS最小化/WSS最小化_第四轮补充实验_点数精度曲线方案快照_2026-07-10.md)。
- 更新：[第四轮执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)（新增 §14 的执行结果）、归档 README、本记录。
- 后续已生成 config 并提交作业（见上一条）。

**推进到实验步骤**：方案归档后已转入 §14.3 执行。

**当前状态判断**：见上一条「最小矩阵已提交」。

## 2026-07-10｜第四轮 Stage A→B→C 执行：协议落地 + 单 seed 矩阵 ✅已完成

**本次主要修改**：
- **Stage A**：审计脚本参数化；`load_partition` 缺失 bundle 报错；移除 `dist_to_wall`；`persistent_workers=False` + DataLoader `generator`/`worker_init_fn`；`fps_multistart` pool；hotspot 定位指标 + `r4_composite_v1` 选模/top-3/early-stop；3× `v2_dev` split + fold stats；A3 Gate-0 残差校准；A5 补 raw top10 与邻域 cap。
- **Stage B（val-only，Jobs 6959–6962）**：B0 batch-quantile 锚点 → B1 固定阈值（持平）→ B2 multi-start / B3 FPS4000。相对 B1：B2/B3 均 **Gate-1 No-Go**（B3 `ΔR²_field≈-0.09` 明确退化，不开 6000）。
- **Stage C（Jobs 6963–6964）**：C1 raw-Huber、C4 `coord_scale` 均为 Gate-1 No-Go。按条件 **跳过** C2（B1 未达增益 Go）、C3（A3 smearing 未通过）、C5（CPU Ridge 增量≈0）。
- 开发默认 `EVAL_PARTS=val`；未跑 legacy test（Stage D 未做）。

**对应代码/文档**：
- 训练侧：`training_wss_min/{config,dataset,train,metrics,evaluate}.py`、`make_configs_round4.py`、`make_v2_dev_splits.py`、`coverage_audit.py`、`a3_residual_calibration.py`、`c5_feature_probe.py`、`gate1_compare.py`、`tests/test_round4_protocol.py`、`cluster/run_train.slurm`。
- 资产：`assets_第四轮/{audit_bundles.py,a5_density_probe.py,a3_residual_calibration.json,c5_radius_gradient_probe.json,a5_density_probe_cap.csv}`；`training/splits/split_AG_wss_min_v2_dev{1,2,3}.json`；`data_wss_min/fold_stats/wss_stats_v2_dev*.json`；`runs/r4_dev1_*`。
- 单测 8/8 通过。

**关键数字（dev1 val）**：

| run | Job | R²_c | R²_f | top10 | IoU | Gate-1 |
|---|---:|---:|---:|---:|---:|---|
| B0 batchq | 6959 | 0.213 | 0.351 | 0.429 | 0.328 | 锚点 |
| B1 fixedq | 6960 | 0.214 | 0.351 | 0.436 | 0.330 | 持平（协议胜者） |
| B2 fpsms | 6961 | 0.211 | 0.352 | 0.425 | 0.341 | No-Go |
| B3 fps4000 | 6962 | 0.207 | 0.259 | 0.346 | 0.296 | No-Go |
| C1 rawhuber | 6963 | 0.200 | 0.329 | 0.413 | 0.362 | No-Go |
| C4 coordscale | 6964 | 0.215 | 0.313 | 0.396 | 0.325 | No-Go |

**其他闸门**：覆盖审计 fixed2000 high-WSS hit 均值仅 ~10%（worst ~5%），multi-start 40ep union ~34%；A5 全量推理邻域 cap 均值 ~0.97 vs 子采样 ~0.37；A3 smearing 三 seed 均使 top10 变差 → 不支持 NLL。

**推进到实验步骤**：第四轮单 seed 主矩阵已跑完；无配置达到 Gate-1 增益阈值，故不扩 3 seed / 不开 6000 / 不进 Stage D。后续若继续，应围绕密度错配（cap/train-eval 密度）或信息上限另立假设，而非重复采样/点数/简单辅助损失。

**当前状态判断**：协议与可复现性已修好；在新协议下，固定阈值≈原 batch 分位，multi-start/4000/raw-Huber/coord_scale 均无稳定收益。当前最优开发锚点为 **B1**（`r4_dev1_b1_tgtw_fixedq_s1234`）。

## 2026-07-10｜第四轮计划终审 v2.3：A5 复现落盘 + rot_aug 证据修正 ✅已完成

**本次主要修改**：
- 终审对 v2.2 的两个可检验事实主张做独立验证。A5 预检数字此前只存在于文档、无落盘脚本/CSV（不符合计划 §2.4 自定证据标准）；终审编写复现脚本并运行，`r3_clean_xyzgeom_tgtw_s1234` 的 MAE/RMSE/Pearson=`0.1645/0.2278/0.9381` **逐位复现**，并补齐 s7（`0.1413/0.1884/0.9605`）与 s2025（`0.1191/0.1638/0.9717`）两个 seed。
- 三 seed 新结论：24 个病例-seed 对中 20 个 mean-shift 为负，即**全量推理系统性高于训练密度推理约 0.03–0.07σ**；点数最多的病例（17682 点）在三 seed 中漂移均最大，支持漂移随密度差增大。A5 标准化口径至此完成，剩余 raw top10 差与邻域 cap 比例两项。
- rot_aug 历史证据核实：`r2_xyzgeom_rotaug` test `R²_field=0.186` 确为第二轮最低，但与 mse 控制组仅差 0.004（在 seed 噪声内），且 val `R²_field=0.281` 反而高于控制组 0.236。No-Go 决定维持，但依据由“历史证据最低”改写为“无正收益证据 + 单 seed 不可判 + 优先级”。
- 计划升版 v2.3：新增 §13 终审结论（含对 v2.2 各项裁决的逐项判定表和 Stage A 前遗留清单），文档冻结为第四轮执行基线，后续改动只能以新增小节/附录记录。

**对应代码/文档**：
- 新增证据资产：`docs/02-推进与变更/assets_第四轮/a5_density_probe.py`（复现脚本，FPS seed 与训练完全一致 `train.seed+7919*i`）、`a5_density_probe.csv`（24 行逐病例）、`a5_density_probe_meta.json`。
- 更新：[WSS最小化_第四轮优化计划_执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)（v2.3，§12.2-1/§12.3-3/§12.6/§11 修订 + 新增 §13）、[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、`training_wss_min/README.md`、本文。
- 复核依据：`training_wss_min/runs/_summary/summary.csv`（rot_aug 数字）、`training_wss_min/runs/r3_clean_xyzgeom_tgtw_s{1234,7,2025}/` checkpoint（只读推理）。
- 本轮仅审核、复现诊断与文档修订；未改训练/评估代码、未生成第四轮配置、未提交训练。

**推进到实验步骤**：第四轮计划审核链闭合（v1 提案 → v2 交叉审核 → v2.2 裁决 → v2.3 终审），进入 Stage A 执行：A0/A0+（审计脚本参数化、split 严格化、常量列防护）、A1/A1+（worker-safe 采样与单测）、A2–A4，以及 A5 剩余两项。

**当前状态判断**：密度敏感性已由三 seed 落盘证据确立为第四轮最强新线索——方向为全量推理整体偏高，恰与尾部欠估方向相反，说明"下采样+插值"不是修复而只是诊断基线；主控制仍是 raw B0，一切结构/特征改动按预注册条件走独立 Go/No-Go。

## 2026-07-10｜第四轮计划第三轮代码/实测交叉审核 v2.2 ✅已完成

**本次主要修改**：
- 在既有逐行核对基础上，使用 `r3_clean_xyzgeom_tgtw_s1234` 完成 A5 首个只读诊断：8 个 val 病例、同一批 FPS-2000 点上，子采样推理与全量推理取同索引的标准化输出平均 MAE=`0.1645`、RMSE=`0.2278`、Pearson=`0.9381`。密度会实质改变输出，但不能由此宣称全量评估错误或直接替换为“下采样+插值”。
- 收紧新增建议的证据等级：保留缺失 bundle 报错、固定 train-only 分位、显式 DataLoader RNG、常量列防护、A5 三 seed 和邻域 cap 审计；取消“radius 一定返回先找到的邻居”“rel 未归一化必为 bug”等过度断言。
- 将 EMA/SWA 从 B0 默认搭载降为 B0 稳定后的预注册候选；将 `rot_aug` 保持为历史 No-Go（r2 field R²=0.186），不再列为 C8 probe。
- 修正 multi-start FPS 方案：预计算多个 FPS-2000 子集而非全 FPS order，且 pool 不能自动解决 persistent-worker 的 epoch 同步；`dist_to_throat` / `stenosis_ratio` 因缺少 branch-aware 定义和病例级 QA，撤出 C5 首批特征训练。

**对应代码/文档**：
- 更新：[WSS最小化_第四轮优化计划_执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)（v2.2；重写 §12，Stage A 新增 A5 正式口径）。
- 复核依据：`training_wss_min/{dataset,train,pointnext,evaluate,metrics,config}.py`、`training_wss_min/cluster/run_train.slurm`、`training_wss_min/runs/r3_clean_*/`、`pipeline_wss_min/preprocess.py`；A5 使用既有 checkpoint，仅推理、未产生训练产物。
- 本轮仅审核/修改文档，未实现计划代码、未生成第四轮配置、未提交训练。

**推进到实验步骤**：下一步仍是 Stage A：先让 split/worker/weight/RNG 口径可验证，再补齐 A5 其余两个 seed、A3 残差校准和 A4 重复开发划分；不得先开 NLL、法向或大点数矩阵。

**当前状态判断**：最强的新线索是密度敏感性，但它还是诊断结果，不是修复结论。第四轮的主控制仍应是 raw B0；任何 EMA、结构改动或新特征都必须在预注册条件和独立 Go/No-Go 下进入。

## 2026-07-10｜第四轮优化计划交叉审核定稿 v2 ✅已完成

**本次主要修改**：
- 交叉核对 77 例 bundle 审计、第三轮 6 个 run、`dataset/train/evaluate/metrics/pointnext` 与 Slurm 提交链，重写第四轮计划的因果判断、执行顺序、实验矩阵和 Go/No-Go。
- 将“log-MSE 是唯一第一性根因、NLL+Jensen 为最高优先级”降级为需先过残差/病例留一校准 Gate-0 的假设；raw-space Huber 辅助改为默认目标函数 probe，NLL 改为条件触发项。
- 新确认 persistent worker 下 epoch 状态同步风险：`resample_each_epoch` 不仅对 FPS 无效，random/geom-weighted 也可能复用 epoch 0 子集；同时固定 batch 分位 target-weight、val-only 开发评估和 fold-specific WSS stats 被提升为 P0/P1。
- 实测 `vf-in` 77/77 可读但 train peak-flow 变异系数仅约 0.02%，与 mean/p99 WSS 相关约 0.13/0.05，因此入口流量从当前第四轮主矩阵移除。

**对应代码/文档**：
- 重写：[WSS最小化_第四轮优化计划_执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。
- 同步：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、`docs/README.md`、`training_wss_min/README.md`、本文和全项目推进记录。
- 复核依据：`docs/02-推进与变更/assets_第四轮/{audit_bundles.py,audit_bundles.csv}`、`data_new/AG/*/*/Global_conditions/vf-in-rfile.out` 和 `training_wss_min/` 当前实现。
- 本轮仅审核/修改文档，未实现计划代码、未生成第四轮配置、未提交训练。

**推进到实验步骤**：第四轮从“单方建议、待交叉验证”推进到“Stage A–D 可执行计划已定稿”；下一步从审计脚本可复现化、sampler 单测、hotspot 指标和 v2 dev split/fold stats 开始。

**当前状态判断**：当前最高优先级是实验正确性和开发协议，不是直接扩模型或启动 NLL/法向/6000 点矩阵。legacy test16 已被前三轮反复使用，第四轮开发阶段必须停止逐配置查看 test。

## 2026-07-10｜第三轮 clean-data 完训判读 + WSS 文档收口归档 ✅已完成

**本次主要修改**：
- 读取 Slurm 6953–6958 的 6 份完整 `eval/metrics.json`、训练曲线、逐病例指标与代表性热力图，完成 MSE/target-weight 各 3 seed 的均值、样本标准差、区域指标和 high-WSS 校准分析。
- 第三轮 target-weight 相对 MSE 的三 seed 均值：test `R²_field 0.225±0.034 vs 0.191±0.018`，`R²_casemean 0.212±0.027 vs 0.176±0.021`，MAE `2.788±0.040 vs 2.847±0.038`；但 seed 7 未稳定获益。
- 第三轮 clean-data 组合将 target-weight 的 top10 预测/真值均值由第二轮约 23.4% 提高到 35.7%，p99 比由约 35.1% 提高到 43.4%；整体 `R²_field` 却与第二轮 tgtw 均值 0.222 基本持平，说明数据修复改善幅值校准但未突破 geometry-only 信息/选模上限。两轮还同时改变 split、curvature transform 和训练时长，因此不把差异归因于 stats 单一变量。
- 将已执行完的预处理交接、工程建议和两版问题诊断移入 `_archive/WSS最小化/`；当前目录只保留 WSS 训练跟踪与推进记录两个活跃事实源，并新增归档索引。

**对应代码/文档**：
- 回填：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、本文、`training_wss_min/README.md`、`docs/README.md`、根 `README.md`。
- 归档：`docs/02-推进与变更/_archive/WSS最小化/`，其中原诊断结论与交接历史保留，不删除失败证据。
- 实验依据：`training_wss_min/runs/r3_clean_xyzgeom_{mse,tgtw}_s*/eval/{metrics.json,per_case_metrics.csv,heatmaps/}` 与对应 `history.jsonl`、Slurm 6953–6958 日志。
- 汇总产物：已重跑 `training_wss_min.summarize`，`training_wss_min/runs/_summary/` 当前聚合 27 个实验并刷新 `summary.csv`、点数曲线与区域柱状图。
- 本轮未修改训练、评估或预处理代码，也未重新训练模型。

**推进到实验步骤**：
- 第三轮 clean-data 已从“产物待分析”推进到“完训、完整评估、三 seed 判读完成”。
- 下一轮优先级固定为：P0 复合/平滑选模与 early stopping；P1 `coord_scale` 和入口流量/可部署边界条件信息上限探针；P2 稳健尾部 loss；P3 重复 split/病例 bootstrap 置信区间。

**当前状态判断**：
- target-weight 是均值正收益方向，但目前仅 2/3 seed 明显获益，不能宣称稳定突破；单 run 最高 test `R²_field=0.251` 只作结果描述，不作为按 test 选出的最终模型。
- high-WSS 仍 No-Go：target-weight 三 seed high-WSS `R²=-1.531±0.167`、top10 比 `0.357±0.037`，热点大致定位但幅值过度平滑。继续只调 α 或堆模型的优先级低于补边界条件/尺度信息和修 checkpoint 选择。

## 2026-07-10｜第三轮 clean-data 口径落地 + bundle 重跑 + 训练矩阵提交 ✅已提交

**背景与目标**：用户确认第三轮不能只改 split，必须在移除污染病例后重跑 included bundle，并保证 pending/excluded 历史 bundle 不进入 stats/training；预处理阶段需要等待完成，训练阶段只提交 Slurm job，结果后续再分析。

**本次主要修改**：
- split 修正：`slow/ZHANG_HUAN_LI` 已从 `train_cases` 移入 `excluded_cases`，理由写明 WSS 近全零、`n_wall` 极端和多重审计异常；当前 split 为 train 53 / val 8 / test 16 / excluded 9 / pending 1，`slow/ZHAO_XIU_XUAN` 保持 pending。
- 数据守卫：`raw_io.read_wall_fields` 补读 ID 与坐标；预处理按首步 `nodenumber/cellnumber` 对齐每个时间步 WSS/pressure/vector，集合或长度异常即失败；兼容少数壁面文件把 ID 列命名为 `cellnumber` 的历史导出口径。
- QA gate：新增 split included QA，excluded/pending 只报告为 not participating；WSS 零值、极端点数、单位/覆盖/裁剪、非有限值、nodenumber/坐标错位为 fatal，单独 `trunk_centering_offset_frac>0.05` 按最终诊断作为 warning 复核项，不默认剔除。
- scope guard：`--all-raw` 仅允许诊断性 `preprocess`；正式 `qa-gate/global-stats/build-samples/all` 在 CLI 和函数入口均拒绝 all-raw，防止 pending/excluded 历史 bundle 被读入正式口径。
- stats 与训练：`global-stats` 默认支持 train peak-only；训练 run 保存 `wss_global_stats.json` 快照；dataset 支持 `coord_scale` 输入，curvature 新 run 默认 `signed_log1p` robust 统计；eval/summarize 固化 top10/p95/p99/max 校准指标。
- 第三轮实验：新增 template mean/voxel/KNN 基线脚本，新增 `make_configs_round3_clean.py` 生成 clean `mse/tgtw` 各 3 seed；提交训练 Job 6953–6958。

**对应代码/文档**：
- 更新：`training/splits/split_AG_wss_min_v1.json`、`pipeline_wss_min/{raw_io.py,preprocess.py,qa_gate.py,global_stats.py,run.py,reporting.py,config.py}`、`training_wss_min/{config.py,dataset.py,train.py,evaluate.py,metrics.py,summarize.py,template_baseline.py,make_configs_round3_clean.py}`、`training_wss_min/cluster/submit_baseline_sweep.sh`。
- 新增/生成：`training_wss_min/configs/r3_clean_xyzgeom_{mse,tgtw}_s*.json`、`training_wss_min/configs/sweep_round3_clean_v1.txt`、`training_wss_min/runs/template_{mean,voxel,knn}_clean/`。
- 文档同步：`pipeline_wss_min/README.md`、`training_wss_min/README.md`、[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、[WSS最小化_数据与训练问题最终诊断_2026-07-10.md](_archive/WSS最小化/WSS最小化_数据与训练问题最终诊断_2026-07-10.md)。

**推进到实验步骤**：
- 预处理：Slurm Job **6952** 完成，`ok=77 / skipped=0 / error=0`；`nodenumber_reordered_cases=0`，`wall_coord_mismatch_cases=0`。
- QA gate：fatal=0，warning=3（`fast/ZHANG_QING_WANG`、`slow/GUAN_TONG_XIANG`、`fast/RAN_QING_BO`，均为 trunk offset 复核项）。
- clean stats：train peak-only 53 cases / 711412 wall points，zero_frac=0，log mean/std=`1.1595 / 1.0907`，raw p90/p99/max=`12.8847 / 32.6039 / 220.7968`。
- 模板基线（test）：mean R²_field=-0.114 / top10=0.172；voxel R²_field=0.035 / top10=0.260；KNN R²_field=-0.039 / top10=0.289。
- GPU 训练：`r3_clean_xyzgeom_mse_s{1234,7,2025}` 与 `r3_clean_xyzgeom_tgtw_s{1234,7,2025}` 已提交 Job **6953–6958**；终检时 6 个 run 均已写出 `eval/metrics.json`，本轮只记录产物入口，不做结果判读。

**当前状态判断**：第三轮 clean-data 已从“计划”推进到“训练产物已生成、结果待分析”阶段。数据污染源已移除，旧 stats 的 `std≈5.58` 已被 clean peak stats 替代；下一步不再改口径，按三 seed 均值/方差和 high-WSS 校准指标判读。

## 2026-07-10｜WSS 数据与训练问题最终诊断文档合并 ✅已完成

**背景与目标**：用户已进入第二轮实验，但第三轮是否需要重跑、怎么改，需要先把 `WSS最小化_训练结果差诊断_Claude与Codex联合审查_2026-07-09.md` 与本文 2026-07-09 数据 QA 复盘中的数据问题、训练症状和修改清单合并为一个后续智能体可直接执行的问题源。

**本次主要修改（文档合并，不改代码）**：
- 新增最终问题文档：[WSS最小化_数据与训练问题最终诊断_2026-07-10.md](_archive/WSS最小化/WSS最小化_数据与训练问题最终诊断_2026-07-10.md)（2026-07-10 执行完成后归档）。
- 明确第三轮前的最终判断：`AG/slow/ZHANG_HUAN_LI` 是当前 included 中唯一必须立即从 train 统计移除的重大污染源；`AG/slow/WANG_BAO_SHAN` 已 excluded，但必须防止直接 glob 历史 bundle 时误纳入。
- 合并训练问题链条：旧 all-time log stats 被坏病例和全时序共同压扁、high-WSS 系统性低估、peak 单步样本少、FPS 无有效重采样、val 选模噪声大、`nodenumber` 行序对齐仍需作为 P0 完整性守卫。
- 合并下一轮修改要求：修 split/denylist、增加 QA gate、重算 clean peak WSS stats、重算 feature stats、补 `nodenumber` 对齐、补 KNN/voxel/template 基线、clean-data `mse` 与 `target-weight loss` 多 seed 复跑，并记录 high-WSS 分位校准指标。
- 明确不要误删真实高 WSS 长尾病例：`LI_ZHI_LIN`、`MA_TIAN_YI`、`SUN_ZONG_GE`、`LIU_ZONG_YANG`、`SHEN_ZHI_GANG` 等只作为长尾样本保留，不按高值剔除。

**对应代码/文档**：
- 新增：原 `docs/02-推进与变更/WSS最小化_数据与训练问题最终诊断_2026-07-10.md`，执行完成后移至 `docs/02-推进与变更/_archive/WSS最小化/`。
- 更新：本文档顶部推进记录；`docs/README.md` 的 WSS-only 入口增加最终诊断文档链接。
- 本次未修改 `training_wss_min/`、`pipeline_wss_min/`、`training/splits/` 或任何实验配置。

**推进到实验步骤**：第二轮实验可以继续作为旧 stats 口径下的方向性证据；第三轮 clean-data 实验启动前，应以最终诊断文档为准完成数据口径修复与重新统计。

**当前状态判断**：问题源已合并完成；后续智能体无需再从两份源文档交叉抽取结论，可直接按最终诊断文档的 P0/P1/P2 顺序实施修改。

## 2026-07-09｜第二轮 sweep 收官汇总 + 高 WSS 分位校准复核 ✅已完成

**背景与目标**：第二轮 9 个 config（Slurm 5961–5969）全部跑完并完成 eval。本次做收官汇总（`summarize` 重跑聚合全部 18 run）、补做高 WSS 分位校准探针（与 2026-07-09 数据 QA 复盘中第一轮探针同口径），并回填跟踪文档与最终诊断文档。不改训练代码。

**第二轮最终结果（test 完整点云）**：
- 最优：`geomw_tgtw` R²_field **0.249** / `tgtwloss` **0.246**（R²_casemean 0.233、sten −0.068、hiW −1.377 均为全场最优）。tgtw 家族(0.235–0.249)全面领先 mse 参考(0.189)、huber(0.203)、rotaug(0.186，全场最差)。
- **seed 方差**：`tgtwloss` 三种子(1234/7/2025) R²_field = 0.246/0.217/0.203，极差 0.043，吞掉家族内配置排序；只有"tgtw 家族 > 非加权"这档结论可信。
- **分位校准探针（新证据，test 全场 pool）**：真值 top10% 均值 18.51 / p99 26.98 / max 126.74。mse 预测 top10% 比 21.1%；tgtwloss 仅 22.1%（几乎没动）；geomw_tgtw 26.2% 但 max 溢出至 138%；geomwsamp_tgtw 52.7% 但 max 爆到 2849(22 倍)。**结论：目标加权的尾部 R² 回拉主要来自中段，真峰值在旧 stats(log std≈5.58)下修不回来，clean-data 重算 stats 是第三轮前置硬条件。**
- best epoch 集中在 59–159（400 epoch 过长）；val(8例)→test R²_field 落差约 0.05–0.11，选模噪声大。

**对应代码/文档**：
- 更新：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)（第二轮完整表 + 分位校准表 + 5 条结论 + 待办改为第三轮 clean-data）、[WSS最小化_数据与训练问题最终诊断_2026-07-10.md](_archive/WSS最小化/WSS最小化_数据与训练问题最终诊断_2026-07-10.md)（第二轮收官证据回填 §0/§1/§3/§5/§6；执行完成后归档）。
- 产物：`training_wss_min/runs/_summary/{summary.csv,pointcount_curve.png,regional_bar.png}`（已含全部 18 run）。
- 本次未修改 `training_wss_min/` 代码与 split。

**推进到实验步骤**：第一/二轮（旧 stats 口径）到此收官，全部结论已固化为方向性证据；下一步严格按最终诊断文档 §7 启动第三轮 clean-data。

**当前状态判断**：loss 方向（目标幅值加权）已验证、数据问题已定位、旧口径上限已探明（R²_field≈0.25 / top10% 校准≈22%）。继续在旧 stats 上调参没有收益，第三轮必须先完成 split 修正 + QA gate + stats 重算。

## 2026-07-09｜第一/二轮训练后数据 QA 复盘：确认 ZHANG_HUAN_LI 为训练统计污染源 + 第三轮修改清单 ⚠️待执行

**背景与目标**：第一轮 `training_wss_min` baseline 暴露出系统性问题：整体 R² 已有弱信号，但 high-WSS / stenosis 区域 R² 全负，最佳模型在 test 高 WSS 区明显低估。为避免继续在被污染数据统计上做二轮/三轮调参，本次只做**数据侧复查与后续修改清单记录**，不改训练代码、不停止已进入第二轮的作业。

**本次主要修改（文档记录 + 数据 QA 结论）**：
- 复查范围：`training/splits/split_AG_wss_min_v1.json` 当前 included=78（train54/val8/test16）、2026-07-08 21:55:12 预处理审计 `data_wss_min/pipeline_reports/preprocess_audit_20260708_215512.csv`、以及 `data_wss_min/AG/*/*/bundle.npz` 中的全时序 WSS、peak WSS、曲率、局部半径、坐标范围和审计字段。
- **必须处理的训练污染病例：`AG/slow/ZHANG_HUAN_LI`**。
  - 当前仍在 `train_cases`，但 peak/all-time WSS 零值比例均为 `0.9166`，peak `p90=0`、peak mean `0.45`。
  - 壁面点数 `209299`，而剔除该例后的 included 正常范围约为 `9338-21452`，该例大约是正常最大值的 9.8 倍。
  - 审计字段同时触发 `unit_anomaly=True`、`unit_extent_mismatch=True`、`wall_crop_frac=0.324`、`trunk_centering_offset_frac=0.0999`。
  - 曲率 99 分位约 `3.13e6`，显著高于其它训练病例；它把第一轮几何特征标准化中的 curvature clip 拉到 `3.13e6`。
  - 结论：该例“坐标视觉回正”不等于“WSS 标签/几何统计可用”；继续留在 train 会污染 WSS global stats、feature stats 和训练 loss。
- **已排除但需防误读的同类风险：`AG/slow/WANG_BAO_SHAN`**。
  - 该例已在 `excluded_cases`，磁盘上仍有历史 bundle，peak/all-time WSS 零值比例同样约 `0.9166`，`n_wall=238116`。
  - 后续脚本必须严格按 split 读取 train/val/test，不允许直接 glob 全部 bundle 参与统计或训练。
- **未发现第二个 included 的同等级 WSS 标签污染病例**。
  - 除 `ZHANG_HUAN_LI` 外，train/val/test 的 peak_zero_frac 与 all_zero_frac 均为 0，未见 peak `p90=0`、非有限值、坐标超界或半径非正等致命问题。
  - `LI_ZHI_LIN`、`MA_TIAN_YI`、`SUN_ZONG_GE`、`LIU_ZONG_YANG`、`SHEN_ZHI_GANG` 等 peak 高值属于长尾高 WSS 样本，不能当作坏数据剔除。
- **需要复核/robust 处理但暂不建议剔除的病例**：
  - 曲率 per-case p99 较高：train `CHENG_GUANG_SEN`(`~7.99e5`)、`ZHANG_SONG_TIAN`(`~7.88e5`)；val `XU_YI_CAI`(`~6.45e5`)；test `WANG_YONG_FAN`(`~6.41e5`)、`GUO_XI_JIANG`(`~6.23e5`)、`QIN_SI_FU`(`~5.23e5`)。
  - 这些病例没有 WSS 零值污染或预处理致命审计字段；建议进入人工图件复核和 curvature robust clip，不作为第三轮默认剔除名单。
  - 坐标/方向复核清单：`LI_SHI_QIANG` 为 `bifurcation_fallback`；`ZHANG_HAO`、`ZHANG_QING_WANG`、`RAN_QING_BO`、`CHENG_LU_LI`、`GUAN_TONG_XIANG` 为 `wall_pca_fallback`；`ZHANG_XIU_WEN`、`CHENG_GUANG_SEN`、`CHEN_SHI_MING` 的 roll sign 置信较弱。当前只作为复核项，不直接判坏。
- **WSS global stats 被污染的量级**：
  - 当前 `wss_global_stats.json`：log mean/std = `-3.161 / 5.582`，train all-time zero_frac = `0.208`。
  - 若从 train 中排除 `ZHANG_HUAN_LI`：all-time log mean/std 约 `-0.353 / 1.270`，zero_frac 变为 0；若只用 clean train peak 统计：log mean/std 约 `1.159 / 1.091`。
  - 当前 stats 会把 raw WSS `20-200` 压到 log_z 约 `1.10-1.52`，峰值差异被严重压扁；这解释了 high-WSS 系统性低估。
- **第一轮模型低估证据**：
  - `feat_xyzgeom_fps_w2000_peak` 在 test：真值 p99/max = `26.98 / 126.74`，预测 p99/max = `9.37 / 16.71`；top10% 高 WSS 均值真值/预测 = `18.51 / 4.08`，只到 `22%`。
  - `feat_geomonly_fps_w2000_peak` 同样只到约 `23%`。因此 high-WSS R² 大负主要是峰值被系统压低，不是单个热力图显示问题。

**第三轮前必须安排的修改清单（给后续智能体）**：
1. **修 split / denylist**：将 `slow/ZHANG_HUAN_LI` 从 `train_cases` 移入 `excluded_cases`，理由写清楚为“WSS 标签近全零 + n_wall 极端 + unit/coverage/crop/trunk 多重异常，污染训练统计”。保持 `WANG_BAO_SHAN`、`SUN_WEN_QING`、`NIE_QUAN_ZHONG` 等既有 excluded 不自动回纳。
2. **增加数据 QA gate**：在 global-stats / build-samples / training 入口加入 split 内病例检查；遇到以下任一条件应 hard fail 或进入 denylist：`peak_zero_frac>0.01`、`all_zero_frac>0.01`、`peak_p90<=0`、`n_wall>50000`、`unit_anomaly=True`、`unit_extent_mismatch=True`、`wall_crop_frac>0.05`、`trunk_centering_offset_frac>0.05`、非有限值。曲率 p99 `>5e5` 先 warning + 图件复核，不默认剔除。
3. **重算 WSS stats**：排除 `ZHANG_HUAN_LI` 后重跑 `global-stats`；第三轮 peak 单步任务优先使用 clean train peak 的 log stats，至少不能继续使用当前 `std≈5.58` 的 all-time stats。训练与评估必须使用同一份新 stats。
4. **重算 feature stats**：所有 run 的 `feature_stats.json` 都不能复用旧值。排除 `ZHANG_HUAN_LI` 后 curvature 的全训练集 99% clip 会从 `~3.13e6` 降到约 `~1.0e5`；同时考虑对 curvature 做 `sign(x)*log1p(abs(x))` 或分位裁剪后 z-score。
5. **第三轮最小实验矩阵**：以 `xyz+geom / fps2000` 为主，重跑 `mse` 与 `target-weight loss`，各至少 3 个 seed；第二轮已跑结果只作为“loss 方向有效”的旁证，不作为 clean-data 最终指标。
6. **补高 WSS 监控指标**：每个 eval 除 R²/NRMSE/MAE 外，记录 test top10% high-WSS 的 `mean_pred/mean_true`、p99 预测/真值比、max 预测/真值比；checkpoint 选择可用 `val_r2_casemean + high_wss/stenosis` 的组合指标，避免只优化平滑低值背景。
7. **复核但不阻塞的病例清单**：对上述曲率尖峰和方向 fallback 病例补一页 QA 图；只有出现形态/方向明显错误或标签异常，才进入下一版 excluded。

**对应代码/文档**：
- 本次只修改本文档。
- 复查依据：`training/splits/split_AG_wss_min_v1.json`、`data_wss_min/pipeline_reports/preprocess_audit_20260708_215512.csv`、`data_wss_min/AG/*/*/bundle.npz`、`training_wss_min/runs/_summary/summary.csv`、第一轮 best checkpoint 完整 test 推理统计。

**推进到实验步骤**：第二轮实验可继续跑完，用于验证 loss/采样/增广方向；但第三轮 clean-data 实验启动前，必须先完成 split 修正、QA gate、WSS stats 和 feature stats 重算。

**当前状态判断**：当前 included 数据中，`ZHANG_HUAN_LI` 是唯一需要立即从训练统计中移除的重大污染源；其余 included 病例没有发现同等级 WSS 标签坏例。第三轮应视作“数据口径修正版”重新起跑，不能把第一/二轮在旧 stats 上得到的指标作为最终可发表结论。

## 2026-07-08｜training_wss_min：PointNeXt 残差 baseline 训练/评估框架 + 第一/二轮 sweep ✅完成

> 训练侧实验跟踪主文档：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)（逐轮 sweep 设计、完整指标表、结论、待办）。

**背景与目标**：flow-divider 预处理口径落地后推进到训练侧。目标：把
`几何点云 (x,y,z[+几何]) → 壁面 WSS 标量`（全局 log_z 归一化，单头）的 baseline 搭起来并集群并行提交；
最终服务工程部署（**A 路**：部署有完整几何、缺 CFD 标签，故训练用稀疏子采样、**评估恒在完整壁面点云上**）。
要求与既有 V3P `training/` **完全独立**。

**本次主要修改（新建独立包 `training_wss_min/`）**：
- `pointnext.py`：PointNeXt-S **残差版**（InvResMLP 残差块 + **ball-query** 分组，密度鲁棒 → 训练稀疏点、推理整条血管可迁移）；`out_dim=3` 即切矢量（二期）。
- `dataset.py`：直读 `data_wss_min/**/bundle.npz`（不经 build_samples）；采样 fps/random/几何加权；特征标准化；训练期随机旋转增广（第二轮加）；完整点云评估接口。
- `metrics.py`：R²/NRMSE/MAE + 分区（分叉/狭窄/高WSS）。
- `train.py`：AdamW+cosine+AMP；日志 `train.log`+`history.jsonl`；best/last ckpt；loss 支持 MSE/Huber + 几何加权 + 目标幅值加权（第二轮加）。
- `evaluate.py`：加载 best，**完整点云**推理 → 指标 + 逐病例 CSV + 真值/预测/误差热力图。
- `make_configs.py` / `make_configs_round2.py`：两轮 sweep 配置生成。
- `cluster/run_train.slurm` + `submit_baseline_sweep.sh`：GPU 分区(master 4×4090) 训练+评估一条龙，多 config 自动排队并行。
- `summarize.py`：聚合 `runs/*/eval/metrics.json` → 对比表 + 点数曲线 + 分区柱状。

**数据口径**：split `split_AG_wss_min_v1`（train54/val8/test16）；坐标逐病例各向同性归一化 [-1,1]；WSS 全局 log_z（train-only 统计，std≈5.6）；评估在原始 WSS 空间。

**第一轮 sweep（9 config，Slurm 5945–5953）✅完成**（详见跟踪文档）：
- 几何特征是压倒性杠杆：xyz+几何(test R²_field=0.200) ≈ 纯几何(0.191) ≫ 纯 xyz@2000(0.053)。纯几何≈xyz+几何 → **标量任务里绝对坐标/配准框架几乎不值分**（正面印证前期讨论口径）。
- 几何特征更省点：2000 点几何 > 6000 点纯 xyz。几何加权采样(0.108) > random(0.084) > fps(0.053)。
- **所有配置狭窄区/高 WSS 区 R² 全负**（最好也 sten −0.20 / hiW −1.63），为主瓶颈。

**第二轮 sweep（9 config，Slurm 5961–5969）✅完成**（收官汇总见 2026-07-09 条目与跟踪文档）：以 xyz+几何为默认，专打尾部崩溃；新增旋转增广 + 目标幅值加权 loss。最终：tgtw 家族 R²_field 0.235–0.249 全面领先（最优 geomw_tgtw 0.249 / tgtwloss 0.246，sten −0.19→−0.07、hiW −1.63→−1.38）；但分位校准显示真峰值仅恢复 21%→22~26%，旧 stats 是硬上限；huber/rotaug 无收益；tgtw 三种子极差 0.043。

**对应代码/文档**：
- 代码：`training_wss_min/`（整目录，独立于 `training/`）。
- 文档：新增 `docs/02-推进与变更/WSS最小化_训练实验跟踪.md`（实验跟踪主文档）、`training_wss_min/README.md`。

**推进到实验步骤**：从预处理/坐标 QA 正式推进到 `x,y,z → wss` 训练与评估；建立点数-精度、特征消融、采样、loss 加权、旋转增广的可复现 sweep 与完整点云评估基线。

**当前状态判断**：baseline 已跑通、可复现、集群并行；最优 test R²_field 目前 **0.246**（第二轮目标加权 loss，仍在跑完最后几例）。主攻方向明确——几何特征 >> 坐标 >> 点数；尾部（狭窄/高 WSS）靠目标/几何加权可回拉但仍为负，需继续。矢量三分量为二期（幅值 + 内在系方向）。

## 2026-07-08｜flow-divider 解剖基点算法化：三叉连接点最终口径 + 78例对照报告 ✅已采纳

**用户新想法**：不再只取 `DistToBifurcation≈0` 区域普通均值，而是取“左右髂支连接到中间主动脉的三叉连接点”作为解剖基点；希望所有血管在同一框架下更规整，方便后续数据增强和训练。

**本次算法化实现**：
- `RegistrationConfig.center_on` 已切换为默认 `flow_divider`，作为当前最终预处理口径。
- `registration.py` 新增 `_flow_divider_origin()`：
  1. 读取 VMTK `centerline.vtp` 中的 `DistToBifurcation`。
  2. 取 `DistToBifurcation≈0` 的分叉近邻中心线点。
  3. 对近邻点做确定性 K-means，聚成 3 臂：主干端 + 左髂支起始端 + 右髂支起始端。
  4. 对 3 个臂中心做**等权平均**作为 flow-divider 原点，避免某一臂采样点更多时把普通均值拉偏。
  5. 若三臂分离不足或数据不完整，自动回退到原 `bifurcation` 原点，并写 `origin_kind=bifurcation_fallback`。
- `visualize_final_case_pages.py` 默认输出到 `outputs/wss_min/flow_divider_origin_QA_78例/`，可用 `--center-on` 做旧口径对照。
- 新增 `compare_flow_divider_origin.py`，批量比较旧 `bifurcation` 原点与当前 `flow_divider` 原点。

**对照报告（当前 included=78）**：
- 报告目录：`outputs/wss_min/flow_divider_origin_compare/`
- CSV：`outputs/wss_min/flow_divider_origin_compare/flow_divider_origin_compare_78cases.csv`
- 汇总图：`outputs/wss_min/flow_divider_origin_compare/flow_divider_origin_summary.png`
- 说明：`outputs/wss_min/flow_divider_origin_compare/说明.md`
- 原点位移 median/mean/p95/max：`4.78 / 5.21 / 10.23 / 20.08 mm`。
- 当前原点 |COM| mean/median：`0.2140 / 0.2228`。
- flow-divider |COM| mean/median：`0.2029 / 0.2133`。
- |COM| 改善/变差：`61 / 16`；其中明确改善（delta≤-0.005）`50` 例，轻微变化（|delta|<0.005）`17` 例，明确变差（delta≥0.005）`11` 例。
- flow-divider 直接成功 `77/78`，仅 `fast/LI_SHI_QIANG` 回退到 `bifurcation_fallback`。

**最终 QA 图**：
- 输出目录：`outputs/wss_min/flow_divider_origin_QA_78例/`
- 汇总表：`outputs/wss_min/flow_divider_origin_QA_78例/当前78例_坐标QA汇总.csv`
- 关键结果：
  - `center_on=flow_divider`：78/78。
  - `origin_kind=flow_divider`：77/78；`bifurcation_fallback`：1/78。
  - `roll_sign_reliable=True`：75/78；弱置信 3 例仍为 `fast/ZHANG_XIU_WEN`、`slow/CHENG_GUANG_SEN`、`fast/CHEN_SHI_MING`。
  - `wall_crop_applied=True`：1/78，仍仅为既有裁剪病例。

**17:47 人工复核后追加修正**：
- 发现 `fast/ZHANG_HAO` 在 flow-divider 口径下触发 `wall_pca_fallback` 后整例翻转。
- 根因：`wall_pca_fallback` 原本用壁面双峰分离度判断“哪侧是髂支侧”，但该例主干弯曲/截面展布更强，导致主干侧被误判成髂支侧，主轴正负号翻转。
- 修正：fallback 仍用壁面 PCA 修正主轴倾斜，但主轴**正负号继承中心线 trunk→bifurcation 方向**；即壁面 PCA 只替换方向，不再单独决定符号。
- 重扫 5 个 `wall_pca_fallback` 病例，只有 `fast/ZHANG_HAO` 存在“PCA 符号与中心线方向冲突且侧判误导”的情况；其他 fallback 病例不受影响。
- 已重画 `outputs/wss_min/flow_divider_origin_QA_78例/` 并重算 `flow_divider_origin_compare/`；总体统计不变，`fast/ZHANG_HAO` 已回到正常朝向。

**STL page03 复核后追加修正**：
- 发现 `02_STL同框_每20例/page03` 中浅橙色 STL 大幅偏离，经颜色/顺序定位为 `slow/ZHANG_HUAN_LI`。
- 该例 CFD wall 点云本身在 `01_点云同框` 和逐病例 X-Z 图中未飞出；异常只发生在 STL 可视化。
- 根因：该例 `unit_extent_mismatch=True`，wall 原始网格含未描入口尾巴，bbox 比例法会把 STL 过度放大。其 STL 原始坐标实际已接近 mm 尺度并与 centerline 覆盖段匹配。
- 修正：`visualize_stl_point_overlap.py` 新增 `_stl_scale_to_pipeline()`，在 `bbox_ratio`、`stl_mm`、`stl_native` 三种尺度假设中打分选择；覆盖不一致病例优先匹配 centerline 尺度。
- `visualize_final_case_pages.py` 复用该函数，并在最终 QA 汇总表新增 `stl_scale_kind` 字段。
- 修正后 `slow/ZHANG_HUAN_LI` 的 STL/wall 重心偏差从 `1.65` 降至 `0.025`，STL 顶点出框比例从 `71.8%` 降至 `0%`；该例使用 `stl_scale_kind=stl_mm`，其余有 STL 的 76 例仍为 `bbox_ratio`。
- 已重画 `outputs/wss_min/flow_divider_origin_QA_78例/`；新增单例复核图：
  `outputs/wss_min/flow_divider_origin_QA_78例/单例复核_slow_ZHANG_HUAN_LI_STL尺度修正.png`。

**最终图件归档与默认口径确认**：
- 当前最终 QA 入口：`outputs/wss_min/flow_divider_origin_QA_78例/`。
- 当前新旧原点对照报告：`outputs/wss_min/flow_divider_origin_compare/`。
- `RegistrationConfig.center_on` 默认已切到 `flow_divider`；`visualize_final_case_pages.py` 默认输出目录也已切到 `flow_divider_origin_QA_78例`。
- 旧的 `当前_78例坐标QA/`、`单位修复前后对比_点云同框/`、`stl_point_overlap_20260708_trunk_centering_cases/` 已归档到：
  `outputs/wss_min/归档_旧口径_20260708/flow_divider定稿前旧图_20260708/`。

**集群 preprocess 重跑（21:48）**：
- 已通过 Slurm 提交到 `node03`：`/public/slurm/bin/sbatch --parsable pipeline_wss_min/cluster/run_preprocess.slurm preprocess`，作业号 `5941`，状态 `COMPLETED`，退出码 `0:0`，耗时 `00:06:39`。
- 运行日志：`logs/wss_min_preprocess_20260708_214835.log`；Slurm 日志：`pipeline_wss_min/cluster/logs/wss_min_pre_5941.out` / `.err`。
- 新审计：`data_wss_min/pipeline_reports/preprocess_audit_20260708_215512.csv` / `.json`。
- 结果核对：split 为 train 54 / val 8 / test 16，included=78；78 个 included bundle 均在本次作业时间窗内更新，`missing=0`、`bad_load=0`、`stale=0`。
- 审计结果：`ok=78`、`skipped=0`、`error=0`；`origin_kind=flow_divider` 77 例，`bifurcation_fallback` 1 例；`coord_scale_on=wall` 78 例。
- `excluded_cases` 和 `pending_cases` 即便磁盘上存在历史 `bundle.npz`，本次作业没有更新，默认 `preprocess/global-stats/build-samples` 也不会读取它们。
- `pending=1` 的 `slow/ZHAO_XIU_XUAN` 含义：原始目录中存在该病例，但不在继承的 `split_AG_v1` train/val/test/excluded 任一名单中，因此暂挂起，不纳入当前 78 例训练口径，待后续单独评估后再决定是否补入。

**当前判断**：
- 这个口径更符合“解剖锚点”表述：把三叉连接点作为同一坐标框架的原点，比普通分叉近邻均值更不受局部采样密度影响。
- 数值上多数病例重心更接近统一框架，但仍有 16 例 |COM| 变差，且 1 例回退；经人工复核后采用该口径作为当前最终 QA 和后续预处理口径。
- `preprocess` 已按该口径在集群完成并生成 78 例新 bundle；后续还需继续跑 `global-stats -> build-samples`，保证 WSS 统计和训练样本也与当前 QA 图同口径。

## 2026-07-08｜人工复核再剔除 2 例：最终 included=78 + 重出 QA ✅

**用户复核意见**：`当前_80例坐标QA/page02` 同框与逐病例 XZ 中仍有 2 例肉眼偏差较大，第一版 baseline 先保守剔除。

**本次 split 修改**：
- 从 `train_cases` 移入 `excluded_cases`：
  - `slow/WANG_BAO_SHAN`：虽经单位兜底与未描入口段裁剪后回到统计正常簇，但 page02 同框/逐病例视图仍呈明显形态与朝向离群。
  - `slow/SUN_WEN_QING`：`coord_scale≈93.24mm`，为当前 80 例最小，归一化后视觉占比/上下位置离群。
- split 计数更新：train 54 / val 8 / test 16 / excluded 8 / pending 1 / total_found 87，当前 included=78。

**图件刷新**：
- 旧 80 例最终图归档到：
  `outputs/wss_min/归档_旧口径_20260708/旧版_最终80例坐标QA_剔除WANG和SUN前/`
- 新 78 例最终图：
  `outputs/wss_min/当前_78例坐标QA/`
- 汇总表：
  `outputs/wss_min/当前_78例坐标QA/当前78例_坐标QA汇总.csv`

**刷新后 QA 摘要**：
- included rows=78；`WANG_BAO_SHAN`、`SUN_WEN_QING` 不在 included，已在 excluded。
- `coord_scale` min/median/max = `128.30 / 192.16 / 266.18`，最低尺度离群被移除。
- 当前只剩 `slow/ZHANG_HUAN_LI` 触发入口裁剪（32.40%）和主干二次居中。

**关于“有的偏上/偏下”**：
- 当前坐标原点固定在分叉点，缩放用壁面 `max_abs`；入口端最低点常被压到接近 -1，但分叉上方能到多高取决于髂支/出口截断长度、分叉角度、主干弯曲和真实血管大小。
- 因此剩余轻微上下差异主要是解剖/截断范围差异，不是单纯归一化失败；只有像 `SUN_WEN_QING` 这种尺度极端小、归一化后视觉占比明显离群的病例才进入剔除。

## 2026-07-08｜最终 QA 收口：旧图归档 + 入口切平面裁剪保险 + 壁面归一化最终图 ✅

**本次收口动作**：
- 归档两个容易误读的旧图目录：
  - `outputs/wss_min/归档_旧口径_20260708/旧版_80例坐标QA_旧单位未裁剪all归一化/`
  - `outputs/wss_min/归档_旧口径_20260708/旧版_新口径QA_轴向裁剪壁面归一化_保险前/`
- `RegistrationConfig` 新增：
  - `crop_use_inlet_tangent=True`：入口裁剪改用中心线入口端局部切平面，避免单纯主轴一刀切。
  - `crop_min_wall_frac=0.08`：裁剪比例低于 8% 视为端点/正常解剖擦边，不执行裁剪。
- `untraced_inlet_crop_masks()`：优先用入口端局部中心线切向判断“中心线覆盖外”的未描入口段；切向不可用时回退旧主轴逻辑。
- `visualize_final_case_pages.py`：最终 QA 汇总 CSV 新增 `unit_extent_mismatch`、`wall_crop_applied/wall_crop_frac`、`coord_scale_on`；STL 视图也按同一入口裁剪口径过滤显示。
- `preprocess.py` / `reporting.py`：bundle/report/audit 补写 `coord_scale_on`。

**最终验证（当前代码直接复算 80 例）**：
- `coord_scale_on=wall`：80/80。
- 只裁剪 2 例：`slow/WANG_BAO_SHAN` 裁 `30.21%`，`slow/ZHANG_HUAN_LI` 裁 `32.40%`。
- 旧版擦边触发的 `fast/LI_ZHEN_SHAN` 不再裁剪（保留完整壁面）。
- |COM| 分布：mean `0.213` / median `0.220` / max `0.362`；两例回到正常簇。
- 主干二次居中仅 `slow/ZHANG_HUAN_LI` 触发；`slow/WANG_BAO_SHAN` 裁剪后无需二次居中。

**最终图件**：
- 主入口：`outputs/wss_min/当前_80例坐标QA/`
- 汇总表：`outputs/wss_min/当前_80例坐标QA/当前80例_坐标QA汇总.csv`
- 重点看：
  - `01_点云同框_每20例/page02_点云同框_病例21-40.png`
  - `01_点云同框_每20例/page03_点云同框_病例41-60.png`
  - `03_逐病例XZ主轴视图_每20例/page02_逐病例XZ_病例21-40.png`
  - `03_逐病例XZ主轴视图_每20例/page03_逐病例XZ_病例41-60.png`

**注意**：
- `slow/WANG_BAO_SHAN` / `slow/ZHANG_HUAN_LI` 只能匹配到 `stl_data/name_data` 中的 STL，STL 与 CFD wall 点云不是逐点同源；专项 STL 图只看形状方向，不作为剔除依据。
- 正式训练前仍需全量重跑 `preprocess -> global-stats -> build-samples`，让 bundle 与样本完全落到最终口径。

## 2026-07-08｜同框偏位三段式收口：单位因子 + 未描主动脉尾巴裁剪 + 壁面归一化 ⏳待全量重跑

问题同下（少数病例同框整体偏位）。逐层定位后确认需要三段修复叠加，缺一不可：

**① 单位因子鲁棒兜底**（详见本条下半部分）：3 例 `unit_factor` 被“壁面/中心线覆盖不一致”压到约 1/2 → 改 `_resolve_unit_factor` 整十次幂兜底，factor→1000。修复后 |COM| 0.50/0.66→0.32，但仍偏。

**② 未描主动脉尾巴裁剪**：实测分叉点其实已在原点（bif_norm≈0），偏位真因是这 2 例壁面网格含一段中心线没描到的近端主动脉（占 31–34% 壁面点），在 per-case 归一化里撑大尺度。
- `config.py`：`RegistrationConfig` 新增 `crop_untraced_inlet=True / crop_axial_margin_frac=0.12 / crop_trigger_overshoot_frac=0.20`。
- `registration.py`：新增 `untraced_inlet_crop_masks()`——配准后（基于中心线，不受尾巴影响）、归一化前，只裁“入口(主动脉)侧”超出中心线轴向覆盖且超出量>触发阈值的壁面/内部点；对覆盖一致的正常病例是零剪裁 no-op。
- `preprocess.py`：配准后应用裁剪 mask，壁面时间步场（WSS/压力/矢量）按同 mask 过滤；新增审计 `wall_crop_applied/wall_crop_frac`（report/bundle/批量审计/日志）。
- 裁剪后 aorta/transv 比 1.13–1.14 ≈ 正常例 1.07–1.15；但发现残余仍在。

**③ 坐标改按壁面归一化**：残余真因是尺度 `max_abs(壁面+内部)` 被内部点带偏——正常例内部含 CFD 流动延伸段（≈2.2×壁面），壁面只填约 0.46 框；这 2 例无延伸段，壁面填满。
- `config.py`：`NormalizationConfig` 新增 `coord_scale_on='wall'`（默认；旧口径 `'all'`）。
- `preprocess.py` / `visualize_final_case_pages.py`：缩放参照改为只按壁面 `max_abs`。

**三段叠加后验证（GNN_vmtk 已实跑这 2 例 + 正常例）**：
- 壁面归一化下全 80 例 |COM| 分布 mean 0.213 / median 0.220；`ZHANG_HUAN_LI` 0.226（**rank 38/80，正好中位**）、`WANG_BAO_SHAN` 0.190（**rank 57/80，中位偏下**）——两例彻底回到正常簇、不再离群。最大偏位反而是普通例（SUN_ZONG_GE 0.36）。
- 同框叠加图目视：两例落进灰色主簇、同轴同心同尺度。
- 裁剪对正常例（如 LI_HUAN_GE）为 no-op（0% 裁剪）。
- 对比图/机制图：`outputs/wss_min/单位修复前后对比_点云同框/`。

**决策记录**：用户拍板“裁掉未描主动脉尾巴 + 改壁面归一化”，不删这 2 例。

**待办**：
- 壁面归一化改动**波及全部 80 例**，需在 GNN_vmtk 全量重跑 `--stage preprocess`（80 例）后再 `global-stats`。
- 用户要求**先审查预处理再做训练样本**，故 `build-samples` 暂缓；先重出点云类 QA（`outputs/wss_min/新口径QA_裁剪加壁面归一化/`）供人工复核。
- STL 同框页需 `trimesh`（GNN 环境未装），本轮 QA 跳过 STL，只出点云/主轴/WSS 视图。
- `NIE_QUAN_ZHONG`（已 excluded）根因同属①②，如需可一并重跑评估是否回纳。
- 代码尚未 `git commit`，待用户审完 QA 一起提。

---
以下为 ① 单位因子修复的原始定位与验证细节（保留）：

## 2026-07-08｜单位因子鲁棒兜底：修复“中心线覆盖不全”导致的整体偏位（①，详情） ⏳待重跑

**问题定位（坐标 QA）**：
- 同框图中少数病例整体偏位，根因不是居中策略，而是 **单位因子 `mesh->mm` 反推被壁面/中心线覆盖范围不一致带偏**。
- 逐病例扫描 `unit_factor`：80 例正常（932–984≈米制）、`PENG_JI_MING` 967789（已知异常，自动处理），
  仅 3 例落在 492–622（≈真实值一半）：`slow/WANG_BAO_SHAN`、`slow/ZHANG_HUAN_LI`、`slow/NIE_QUAN_ZHONG`。
- 复核这 3 例原始几何：中心线只描了远端一段（含分叉），壁面网格却含整条近端主动脉，
  只有约 55–65% 壁面点落在中心线包围盒内；`cl_diag/wall_diag` 因此把 factor 压到约 1/2，
  坐标缩放/配准/二次居中全线偏移（居中偏移量高达 73–77mm），即同框图所见的“整体偏位”。

**本次主要修改**：
- `pipeline_wss_min/config.py`：`UnitConfig` 新增 `phys_diag_mm=250.0`、`ratio_trust=1.5`。
- `pipeline_wss_min/preprocess.py`：`_resolve_unit_factor` 改鲁棒版——比值法结果与
  “米制整十次幂（使壁面对角线≈生理尺度的 10^k）”偏离超过 `ratio_trust` 倍时，
  判定壁面/中心线覆盖不一致，改用整十次幂并返回 `extent_mismatch=True`；正常 81 例比值≈整十次幂，行为不变。
- 新增审计字段 `unit_extent_mismatch`（report.json / bundle / 批量审计 / run 日志）。
- `pipeline_wss_min/reporting.py`、`run.py`：批量审计与日志新增 `unit_extent_mismatch_cases` 待复核清单。

**验证（未跑 sklearn 依赖步骤，仅几何/配准层）**：
- 全量重算 `_resolve_unit_factor`：**只有** 上述 3 例被标记 `extent_mismatch`，factor→1000；其余 84 例（含 PENG_JI_MING 967789）变化<1%。
- 用 `compute_transform` 模拟 factor→1000 后归一化重心 |COM|：`WANG_BAO_SHAN` 0.50→0.14、`NIE_QUAN_ZHONG` 0.52→0.14（回到正常簇 ≈0.13），
  `ZHANG_HUAN_LI` 0.66→0.27（残余为其未被中心线覆盖的近端主动脉横向弯曲）。`center_on` 由 bifurcation 改 wall 无差异；改 `all` 反而把正常例推离原点，不采用。

**对应代码/文档**：`pipeline_wss_min/{config,preprocess,reporting,run}.py`。

**当前状态判断 / 待办**：
- 代码改完并通过语法/几何层验证，但 **未重跑** `preprocess`（本会话环境缺 sklearn）。需在原环境执行
  `python -m pipeline_wss_min.run --stage all`（或先 `--stage preprocess --cohort AG/slow --case ZHANG_HUAN_LI/WANG_BAO_SHAN` 冒烟），再重出 `当前_80例坐标QA`。
- 训练集内两例 `ZHANG_HUAN_LI`、`WANG_BAO_SHAN` 预计随重跑自动回正，无需剔除。
- `NIE_QUAN_ZHONG` 当初因“整体偏位”被 excluded；根因已修，可评估是否重新纳入（决策待定，本次未改 split）。
- `ZHANG_HUAN_LI` 残余 0.27 与 3 例中心线覆盖不全是更深层问题；如需完全同口径，可选：裁剪近端主动脉至中心线段 / 重抽全长中心线（VMTK），二选一待定。

## 2026-07-08｜当前 80 例最终 QA 分页图 + 旧口径图件归档 ✅

**本次主要修改**：
- 新增 `pipeline_wss_min.visualize_final_case_pages`，基于当前 `split_AG_wss_min_v1` included=80 直接生成最终人工复核图。
- 新增中文输出目录 `outputs/wss_min/当前_80例坐标QA/`，子目录按图件用途命名：
  - `01_点云同框_每20例/`
  - `02_STL同框_每20例/`
  - `03_逐病例XZ主轴视图_每20例/`
  - `04_峰值WSS正视图YZ_每20例/`
  - `05_峰值WSS俯视图XY_每20例/`
  - `06_LR左右轴复核/`
- 每类分页图按 20 例一页输出，共 4 页；同时写出 `当前80例_坐标QA汇总.csv` 和 `说明.md`。
- 将 `outputs/wss_min/` 下旧口径英文目录归档到 `outputs/wss_min/归档_旧口径_20260708/`，并把归档子目录改成中文描述名，保留旧图不删除。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/visualize_final_case_pages.py`
- 文档：`pipeline_wss_min/README.md`、`outputs/wss_min/当前_80例坐标QA/说明.md`、`outputs/wss_min/归档_旧口径_20260708/说明.md`
- 当前主看目录：`outputs/wss_min/当前_80例坐标QA/`
- 旧图归档目录：`outputs/wss_min/归档_旧口径_20260708/`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的最终坐标 QA 阶段：当前 80 例已能按点云/STL 同框、逐病例主轴视图、峰值 WSS 正/俯视图成套人工复核。

**当前状态判断**：
- 当前输出顶层只保留两个入口：`当前_80例坐标QA/` 与 `归档_旧口径_20260708/`，降低误读旧图风险。
- 汇总 CSV 显示：STL 缺失 1 例（`slow/ZHANG_JUN_HUA`）；LR 符号继续观察 2 例（`fast/ZHANG_XIU_WEN`、`fast/CHEN_SHI_MING`）；非默认主轴来源 4 例（其中 `ZHANG_XIU_WEN` 为 `centerline_chord_ambiguous`）。
- 图件已生成但未重跑 `preprocess/global-stats/build-samples`；正式训练前仍需基于当前 split 执行 `python -m pipeline_wss_min.run --stage all`。

## 2026-07-08｜P3：剔除 `slow/NIE_QUAN_ZHONG` + split-aware 流程过滤 ✅

**本次主要修改**：
- 按人工复核决策，将 `slow/NIE_QUAN_ZHONG` 从 `training/splits/split_AG_wss_min_v1.json` 的 `train_cases` 移入 `excluded_cases`。
- split 计数更新为：train 56 / val 8 / test 16 / excluded 6 / pending 1 / total_found 87，当前 included=80。
- `pipeline_wss_min/config.py` 新增 split 读取工具：`load_split()`、`split_case_labels()`、`list_split_cases()`。
- `run.py` 默认按 `split_AG_wss_min_v1` 的 train/val/test included 病例运行；新增 `--all-raw` 仅用于原始目录排查。
- `global_stats.py` 默认只用 `train_cases` 计算 WSS 全局统计，避免 val/test/excluded 旧 bundle 混入。
- `build_samples.py` 默认只装配 split included 病例；manifest 写入 `split_name` 与 `partitions`。
- `visualize_stl_point_overlap.py --cases` 改为可指定 split 外病例做 QA，便于后续复核 excluded 个案。

**剔除原因**：
- `NIE_QUAN_ZHONG` 的主轴需要 `wall_pca_fallback`，STL→壁面点云 p95 约 `0.155 mm`，同框图显示整体偏位；这类问题更接近几何域/中心线质量异常，不适合第一版 `x,y,z -> wss` baseline 训练集。

**验证记录**：
- split 检查：included 80、train 56，`slow/NIE_QUAN_ZHONG` 不在 included/train，已在 excluded。
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m compileall pipeline_wss_min`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_alignment --split split_AG_wss_min_v1 --tag p3_exclude_nie`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_lr_check --split split_AG_wss_min_v1 --n 80 --tag p3_exclude_nie`

**图件**：
- `outputs/wss_min/alignment_viz_p3_exclude_nie/`
- `outputs/wss_min/lr_check/lr_side_grid_p3_exclude_nie.png`

**当前状态判断**：
- `NIE_QUAN_ZHONG` 已不会进入后续默认 preprocess/global-stats/build-samples；若旧 bundle 留在磁盘，也不会被默认统计和样本装配读取。
- 当前 split included=80；LR QA 剩余 `unreliable=2`，对应继续观察的 `fast/ZHANG_XIU_WEN` 与 `fast/CHEN_SHI_MING`。
- 正式训练前建议重新执行 `python -m pipeline_wss_min.run --stage all`，用新 split 与 split-aware 代码刷新 bundle、train-only WSS stats 和样本 manifest。

## 2026-07-08｜保留 3 例 STL/点云指定病例 QA 图 ✅

**本次主要修改**：
- `pipeline_wss_min.visualize_stl_point_overlap` 新增 `--cases` 参数，可直接指定病例列表出图，不再只能随机抽样。
- 对 P2 后保留继续复核的 3 例生成 STL/点云对比图：
  `fast/ZHANG_XIU_WEN`、`slow/NIE_QUAN_ZHONG`、`fast/CHEN_SHI_MING`。

**图件与审计**：
- 输出目录：`outputs/wss_min/stl_point_overlap_p2_keep3/`
- 逐例 3D 叠加：`01_selected3_stl_point_overlap_3d.png`
- 三投影叠加：`02_selected3_stl_point_overlap_projections.png`
- 局部放大：`03_selected3_stl_point_overlap_zoomed_projections.png`
- 同框点云：`04_selected3_same_frame_pointcloud_overlay.png`
- 同框 STL：`05_selected3_same_frame_stl_overlay.png`
- CSV：`selected3_stl_point_overlap_summary.csv`

**当前状态判断**：
- `fast/ZHANG_XIU_WEN` 与 `fast/CHEN_SHI_MING` 的 STL/壁面点云几乎逐点贴合，STL→点云 p95 分别约 `8.01e-05 mm`、`3.23e-05 mm`。
- `slow/NIE_QUAN_ZHONG` 的中位距离很小，但 STL→点云 p95 约 `0.155 mm`，明显高于另外两例；结合其 `main_axis_source=wall_pca_fallback` 与同框图中的整体偏位，继续保留复核是合理的。

## 2026-07-08｜P2：5 例 LR 符号加入 override，保留 3 例继续复核 ✅

**本次主要修改**：
- 按人工复核决策，将剩余 `roll_sign_unreliable` 中除 `ZHANG_XIU_WEN`、`CHEN_SHI_MING`、`NIE_QUAN_ZHONG` 以外的 5 例加入 `REGISTRATION_CASE_OVERRIDES`，统一使用 `roll_sign_mode="world_axis"`。
- 新增 override 病例：
  `slow/YIN_YU_RONG`、`slow/ZANG_YU_SHU`、`slow/LI_CHONG_ZENG`、`slow/XU_YI_CAI`、`slow/QIN_SI_FU`。
- 当前 override 总数为 9 例：此前 4 例 + 本次 5 例。

**验证结果**：
- 全量 81 例 LR QA：`unreliable=3`，输出图：
  `outputs/wss_min/lr_check/lr_side_grid_p2_override_keep3.png`
- 只读 transform 诊断确认剩余 3 例为：
  `fast/ZHANG_XIU_WEN`（`centerline_chord_ambiguous`）、
  `slow/NIE_QUAN_ZHONG`（`wall_pca_fallback`）、
  `fast/CHEN_SHI_MING`（`centerline_chord`）。

**当前状态判断**：
- 这 5 例不再作为待复核阻塞项；后续全量 preprocess 时会直接按 override 生效。
- 剩余 3 例继续保守保留，建议后续单独看 STL/点云同框与 LR 细图后再决定是否剔除、override 或保留审计标记。

## 2026-07-08｜P0/P1：LR 复核口径统一 + 主轴壁面 PCA 兜底 ✅

**本次主要修改**：
- **P0：复核图与正式配准口径统一。** `visualize_lr_check.py` 不再自己用单侧 `z>阈值` 选下游点，而是复用正式配准里的 `_select_wall_branch_side()`：对分叉两侧分别算横断面双峰分离度，取更像双髂支的一侧。这样消除了 `fast/LI_SHI_QIANG` 这类“配准正确、复核图全蓝”的诊断假象。
- **P0：QA 默认套用逐病例 override。** `visualize_lr_check.py`、`coord_check.py`、`visualize_alignment.py`、`visualize_stl_point_overlap.py` 均改为走 `registration_for_case()`，保证 QA 图看到的是正式 preprocess 会使用的配准配置；只有显式传 `--roll-source/--roll-sign-mode` 做旧口径对照时才关闭 override。
- **P1：主轴增加壁面 PCA 鲁棒兜底。** 默认仍用中心线入口端→分叉点 trunk 弦向；若该弦向无法清楚区分“单主干侧/双髂支侧”（两侧双峰分离度差过小），再用壁面点云 PCA 长轴，并按双髂支侧定号，使主干/分叉重新落到统一 +Z 视角。
- **P1：审计字段补全。** `RigidTransform` / `bundle.npz` / `report.json` / 批量审计新增 `main_axis_source` 与 `main_axis_wall_sep_delta`，后续能直接看到某例是 `centerline_chord`、`wall_pca_fallback` 还是 `centerline_chord_ambiguous`。

**诊断结论**：
- `fast/FAN_JIAN_MING`、`fast/LI_ZHEN_SHAN` 的中心线 trunk 弦向与壁面主方向夹角不大，属于血管自身弯曲/局部形态造成的视觉差异，不是主轴估计失败；本次保持 `centerline_chord`，避免过度旋转。
- `slow/NIE_QUAN_ZHONG` 是真实主轴问题：中心线只有局部 trunk/branch 信息，入口端→分叉弦向不能稳定代表主干方向；开启壁面 PCA 兜底后，从斜放恢复为主干近似竖直。
- 全 split 轻量诊断中，主轴来源为 `centerline_chord=76`、`wall_pca_fallback=4`、`centerline_chord_ambiguous=1`。fallback 病例进入审计清单，便于后续人工复核。
- 全量 81 例 LR QA 重跑后，待人工确认从 9 例降至 **8 例**；其中 `slow/CHENG_GUANG_SEN` 因主轴兜底后 trunk-bending 置信提高，已从 unreliable 中移出。剩余 8 例主要是 `roll_sign` 符号锚置信弱，红蓝两侧本身大多已清楚分开，属于 P2 是否加入个案 override 的范围。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/registration.py`、`config.py`、`preprocess.py`、`reporting.py`、`visualize_lr_check.py`、`coord_check.py`、`visualize_alignment.py`、`visualize_stl_point_overlap.py`
- 主轴前后对照图：`outputs/wss_min/main_axis_p1_compare.png`
- 新 81 例对齐图：`outputs/wss_min/alignment_viz_p0p1_main_axis/`
- 新全量 LR 复核图：`outputs/wss_min/lr_check/lr_side_grid_p0p1_all81_full.png`
- 剩余待确认 9→8 例复核图：`outputs/wss_min/lr_check/lr_side_grid_p0_remaining9.png`

**验证记录**：
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m compileall pipeline_wss_min`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_alignment --split split_AG_wss_min_v1 --tag p0p1_main_axis`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_lr_check --split split_AG_wss_min_v1 --n 81 --tag p0p1_all81_full`

**当前状态判断**：
- P0/P1 核心修复已完成：QA 图与正式处理逻辑一致，中心线主轴异常病例有可审计 fallback，且不会把普通弯曲病例强行拉直。
- 正式训练前仍需按新配准口径全量重跑 `preprocess -> global-stats -> build-samples`，否则旧 `bundle.npz` 仍可能混用旧坐标。
- 后续 P2 建议只处理剩余 8 例 `roll_sign_unreliable`：逐例决定是否接受 `world_axis_weakbend`，或加入 `REGISTRATION_CASE_OVERRIDES`。

## 2026-07-08｜左右轴 wall_branches 改造 + 13 例复核图诊断（待 P0/P1 修复）🔍

> 本条记录 2026-07-08 复核发现与待修项，代码收尾在该记录时尚未完成。

**背景（2026-07-08 已改）**：左右轴配准从「中心线分支 PCA + 世界轴定号」整体改为
「分叉下游**壁面点**双峰求真实 L-R + 主干弯曲(A-P)×主轴内在手性 + 世界轴两锚交叉校验」，
逐级回退 `wall_branches → branches → curvature`；并对四个 QA 脚本加 `--tag` 出到 `_new` 目录（旧图保留）。

**复核发现（诊断）**：
1. 左右轴纯度（两髂支沿 X 轴分离度）全 81 例 **≥0.97**（旧法最低 0.56），朝向抖动/左右翻转基本消除；
   **80/81 走 `wall_branches`**，`slow/WANG_BAO_SHAN` 因分叉原点落在血管轴向端点回退 `branches`（仍出正常 Y）。
2. `outputs/wss_min/lr_check/lr_unreliable_13cases_20260708.png` 复核图**结构正确**，但
   **`fast/LI_SHI_QIANG` 的「全蓝」是该图下游选取的假象**：该图用**单侧** `z>阈值`，恰好抓到只含一条髂支的
   一侧（1472 点全 x<0）；而配准里的 `_wall_branch_lr_axis` 是**两侧自适应**，选中的另一侧（8639 点）
   正常双峰（左 39% / 右 61%）→ **LI_SHI_QIANG 的左右轴其实是对的**。
3. 待复核 13 例中 **12 例可见清楚蓝左/红右**（轴对，仅主干弯曲太弱无法自证符号）；
   真正「长得歪」的是 `fast/FAN_JIAN_MING`、`fast/LI_ZHEN_SHAN`、`slow/NIE_QUAN_ZHONG`——
   属 **主轴倾斜**（主干方向估计问题），与左右轴无关。
4. 已加 4 例 override（`FAN_JIAN_MING`/`LI_SHI_QIANG`/`LI_ZHEN_SHAN`/`ZHANG_HAO` → `world_axis`），
   **待复核由 13 降至 9**：`ZHANG_XIU_WEN`、`YIN_YU_RONG`、`NIE_QUAN_ZHONG`、`ZANG_YU_SHU`、
   `LI_CHONG_ZENG`、`CHENG_GUANG_SEN`、`XU_YI_CAI`、`CHEN_SHI_MING`、`QIN_SI_FU`。

**待修复 / 待决**：
1. 把 13 例复核图的**下游选取改成与配准一致的两侧自适应**（消除 LI_SHI_QIANG 假象）；
   或直接用已是两侧逻辑的 `visualize_lr_check.py` 按当前 9 例重生成。
2. 剩余 9 例：目视基本是正常 Y，定夺是「接受 `world_axis` 回退」还是逐个加入 `REGISTRATION_CASE_OVERRIDES`。
3. 主轴倾斜的 3 例（`FAN_JIAN_MING`/`LI_ZHEN_SHAN`/`NIE_QUAN_ZHONG`）属 `main_axis` 估计问题，单独一轮处理。
4. 配准口径定稿后，对整个 split 重跑 `preprocess`（现有 bundle 仍是旧坐标），再跑 `global-stats` / `build-samples`。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/registration.py`（`_wall_branch_lr_axis` 两侧自适应 + `_trunk_bending_*` + `_sign_lr_axis` 两锚交叉校验）、
  `config.py`（`roll_source`/`roll_sign_mode`/`downstream_wall_axis_frac`/`roll_sign_min_cos`/`REGISTRATION_CASE_OVERRIDES`）、
  `preprocess.py`、`reporting.py`、`visualize_lr_check.py`（新增）、
  `visualize_alignment.py`/`coord_check.py`/`visualize_stl_point_overlap.py`（加 `--tag`；小图 ⚠ 改为「回退/待复核才告警」）
- 图件：`outputs/wss_min/lr_check/`、`alignment_viz_new/`、`coord_check_new/`、`stl_point_overlap_new/`
- 审计：`data_wss_min/pipeline_reports/preprocess_audit_latest.json`（`roll_sign_unreliable_cases`）

**当前状态判断**：
- 左右轴主问题（朝向不一致 / 偶尔翻转）已在配准层修复并有量化证据（纯度 ≥0.97、`det=+1`、可复现）。
- 剩余为三件收尾：复核图取样口径统一、9 例符号确认、3 例主轴倾斜；均不阻塞「现算配准看新朝向」。

## 2026-07-08｜4 例 roll 符号 override + 重跑 preprocess ✅

**本次主要修改**：
- 新增 `config.REGISTRATION_CASE_OVERRIDES` 与 `registration_for_case()`：对 trunk_bending 与世界轴冲突、但 `wall_branches` 左右轴仍可信的个案，显式锁定 `roll_sign_mode="world_axis"`（几何与 weakbend 回退一致，仅消除 unreliable 标记）。
- 已处理 4 例：`fast/FAN_JIAN_MING`、`fast/LI_SHI_QIANG`、`fast/LI_ZHEN_SHAN`、`fast/ZHANG_HAO`；重跑 preprocess 并刷新审计。

**对应代码/文档**：
- 代码：`pipeline_wss_min/config.py`、`preprocess.py`、`README.md`
- 审计：`data_wss_min/pipeline_reports/preprocess_audit_latest.json`（`roll_sign_unreliable` 13→9）

**推进到实验步骤**：
- split 内 4 例已纳入正式 override 清单；审计待复核由 13 例降至 9 例。

**当前状态判断**：
- 4 例 `roll_sign_reliable=True`、`roll_sign_source=world_axis`，bundle 旋转矩阵与 override 前 weakbend 结果逐元素一致。
- 剩余 9 例仍待同类处理或人工 LR 复核（含 val `XU_YI_CAI`、test `CHEN_SHI_MING` 等）。

## 2026-07-08｜P4 主干横向二次居中 + 当前 80 例 QA 刷新 ✅

**背景问题**：
- 用户复核 `当前_80例坐标QA/01_点云同框_每20例/page02/page03` 时发现两例整体偏离中心轴。
- 量化排查确认不是已剔除的 `slow/NIE_QUAN_ZHONG`，而是当前 split 内的两例 train 病例：
  - `slow/WANG_BAO_SHAN`：page02 第 24 例，低位主干横向 offset_frac≈`0.278`。
  - `slow/ZHANG_HUAN_LI`：page03 第 46 例，低位主干横向 offset_frac≈`0.236`。

**本次主要修改**：
- `RegistrationConfig` 新增 `trunk_centering=True`、`trunk_centering_quantile=0.20`、`trunk_centering_min_offset_frac=0.08`、`trunk_centering_stat="median"`。
- `registration.py` 在主轴/roll 旋转确定后，检测低位 20% 主干壁面点的横向中心；若 normalized offset 超阈值，则只做非主轴两个方向的刚性平移，并折算进 `RigidTransform.centroid`。
- 新增审计字段：
  - `transform_trunk_centering_applied`
  - `transform_trunk_centering_offset_mm`
  - `transform_trunk_centering_offset_frac`
- `preprocess.py`、`reporting.py`、`coord_check.py`、`visualize_stl_point_overlap.py`、`visualize_final_case_pages.py` 同步写入/展示该字段。
- `visualize_final_case_pages.py` 新增 `07_二次居中病例复核/二次居中病例_STL点云三视图.png`。

**验证结果**：
- 全 80 例复测中，仅 `slow/WANG_BAO_SHAN` 和 `slow/ZHANG_HUAN_LI` 触发二次居中。
- 修正后两例低位主干残余横向偏移接近 0：
  - `WANG_BAO_SHAN`：mean_r≈`0.0016`，median_r≈`0.0000`。
  - `ZHANG_HUAN_LI`：mean_r≈`0.0004`，median_r≈`0.0000`。
- 已重跑两例 bundle：
  - `data_wss_min/AG/slow/WANG_BAO_SHAN/bundle.npz`
  - `data_wss_min/AG/slow/ZHANG_HUAN_LI/bundle.npz`
- 已刷新当前 80 例 QA：
  - `outputs/wss_min/当前_80例坐标QA/01_点云同框_每20例/page02_点云同框_病例21-40.png`
  - `outputs/wss_min/当前_80例坐标QA/01_点云同框_每20例/page03_点云同框_病例41-60.png`
  - `outputs/wss_min/当前_80例坐标QA/07_二次居中病例复核/二次居中病例_STL点云三视图.png`
  - `outputs/wss_min/当前_80例坐标QA/当前80例_坐标QA汇总.csv`
- 二次居中前的旧 QA 图已归档到：
  - `outputs/wss_min/归档_旧口径_20260708/旧版_80例坐标QA_二次居中前/`

**注意事项**：
- 两例只能匹配到 `stl_data/name_data/` 中的 STL，STL 与 CFD wall 点云不是逐点贴合口径；专项图中 STL-点云距离较大，不能作为本轮剔除依据。
- 当前仓库没有 `data_wss_min/wss_global_stats.json` 与默认 samples 目录，因此本轮未强行生成训练样本；正式训练前仍需按当前配准口径全量跑 `preprocess -> global-stats -> build-samples`。

## 2026-07-07｜随机 10 例点云/STL 同框叠加图 + 推进记录拆分 ✅

**本次主要修改**：
- `pipeline_wss_min.visualize_stl_point_overlap` 新增两张“同框叠加”图，不再只做逐病例小格子：
  - 10 例壁面点云放在同一张图、同一坐标范围、同一 3D 相机视角下叠加；
  - 同一批 10 例 STL 顶点/面片也放在同一张图中叠加。
- 两张同框图都保留原点参考线和 X-Z / Y-Z / X-Y 三投影，用来直接检查分叉原点是否对齐、主干/分叉朝向是否一致。
- 新建本文档，将 WSS-only 最小化路线的详细推进记录从 V3P 大日志中独立出来；原 `代码修改与实验推进记录.md` 后续保留 V3P / 训练主线记录和拆分说明。

**对应代码/文档**：
- 代码：`pipeline_wss_min/visualize_stl_point_overlap.py`
- 文档：`README.md`、`pipeline_wss_min/README.md`、`docs/README.md`、`docs/02-推进与变更/_archive/WSS最小化/WSS最小化预处理流程_搭建与交接记录_2026-07-07.md`、`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`、`docs/02-推进与变更/代码修改与实验推进记录.md`
- 图件：
  - `outputs/wss_min/stl_point_overlap_20260707/04_random10_same_frame_pointcloud_overlay.png`
  - `outputs/wss_min/stl_point_overlap_20260707/05_random10_same_frame_stl_overlay.png`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的几何输入 QA：从“逐病例检查 STL/点云是否贴合”进一步推进到“多病例放入同一解剖坐标框架后是否原点对准、视角一致”。

**当前状态判断**：
- 同一随机种子 `20260707` 抽取的 10 例中，壁面点云同框叠加与 STL 同框叠加均显示：分叉附近围绕统一原点，主干整体位于 `z<0`，分叉/出口位于 `z≈0` 到正值区域。
- STL 版本与点云版本视觉结论一致，支持当前分叉原点配准和逐病例归一化可作为 `x,y,z -> wss` 简化输入的前置规整步骤。

## 2026-07-07｜随机 10 例 STL-点云逐病例重叠 QA 图 ✅

**本次主要修改**：
- 新增 `pipeline_wss_min.visualize_stl_point_overlap`：从 `split_AG_wss_min_v1` 随机抽取 10 个可匹配 STL 的病例，读取壁面 ascii 点云与 `stl_data/` 中对应 STL。
- 图件生成时不做 ICP 或额外拟合，只把 STL 按 ascii/STL 原始单位比例换到 pipeline 坐标尺度，再与壁面点云套用同一个单位换算、分叉原点刚性配准和逐病例归一化尺度。
- 输出 3D 叠加图、固定坐标范围三投影图、局部放大三投影图与 CSV 距离审计，用于确认后续 `x,y,z -> wss` 的几何输入确实处于同一处理框架。

**对应代码/文档**：
- 代码：`pipeline_wss_min/visualize_stl_point_overlap.py`
- 文档：`pipeline_wss_min/README.md`
- 图件：`outputs/wss_min/stl_point_overlap_20260707/01_random10_stl_point_overlap_3d.png`、`02_random10_stl_point_overlap_projections.png`、`03_random10_stl_point_overlap_zoomed_projections.png`
- 审计：`outputs/wss_min/stl_point_overlap_20260707/random10_stl_point_overlap_summary.csv`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的几何 QA：在解剖原点坐标系之外，补充 STL 面片与壁面训练点云的一致性检查，为只输入 `x,y,z` 的壁面 WSS 预测提供坐标规整证据。

**当前状态判断**：
- `split_AG_wss_min_v1` 中 80 例可匹配 STL，1 例未匹配到 STL；随机种子 `20260707` 抽取的 10 例全部成功出图。
- 10 例 `native_to_stl≈1000`，说明 STL 基本是 ascii 原生坐标的毫米版本；套用 pipeline 坐标尺度后，STL 到壁面点云的 p95 NN 距离约 `1.19e-05` 到 `9.66e-05 mm`，视觉上三投影完全贴合。
- 当前证据支持：壁面点云与 STL 几何源一致，且经新版分叉原点配准后可放入同一个坐标框架继续做 `x,y,z -> wss`。

## 2026-07-07｜解剖原点版配准 v2 + 全量坐标 QA 图 ✅

**本次主要修改**：
- `pipeline_wss_min` 的坐标配准从“壁面重心原点 + 中心线主轴 + 曲率 roll”升级为**分叉原点版解剖刚性配准 v2**：默认读取 `centerline.vtp` 中的 `DistToBifurcation` / `BranchId`，以分叉区域为原点，用入口端→分叉 trunk 方向对齐 +Z，用分叉后远端分支平面固定横轴；缺 VTP 拓扑时保留旧策略兜底。
- `bundle.npz`、单病例 `report.json`、批量审计 CSV 新增 `origin_kind` / `main_axis_mode` / `roll_source` 追踪字段，避免 fallback 病例静默混入。
- 重新生成 WSS-min 坐标 QA 图：`outputs/wss_min/alignment_viz_anatomical_v2/` 与 `outputs/wss_min/coord_check_20260707_anatomical_v2/`；81 例 included 全部 `origin=bifurcation`、`main=inlet_to_bifurcation`、`roll=branches`，无单位异常，`det(R)=1`，最大正交误差约 `3e-16`。
- 更新 `pipeline_wss_min` README、WSS-min 交接记录、根 README 与 docs 索引，说明新版配准口径、图件位置与旧 bundle 需重跑 preprocess 的注意事项。

**对应代码/文档**：
- 代码：`pipeline_wss_min/config.py`、`raw_io.py`、`registration.py`、`preprocess.py`、`reporting.py`、`visualize_alignment.py`、`coord_check.py`
- 文档：`pipeline_wss_min/README.md`、`docs/02-推进与变更/_archive/WSS最小化/WSS最小化预处理流程_搭建与交接记录_2026-07-07.md`、`README.md`、`docs/README.md`
- 图件：`outputs/wss_min/alignment_viz_anatomical_v2/*.png`、`outputs/wss_min/coord_check_20260707_anatomical_v2/*.png`

**推进到实验步骤**：
- 推进到任务 A 的 WSS-only 最小化数据线：完成 `x,y,z -> wss` 前置的解剖坐标统一与全量 QA，可作为下一步重跑 `preprocess -> global-stats -> build-samples` 和壁面点数扫描的输入标准。

**当前状态判断**：
- 新版坐标系比旧壁面重心原点更符合“同一视角 / 同一框架 / 基本朝向一致”：分叉原点固定，主干统一位于 `z<0`，分叉/出口统一位于 `z≈0` 到正值区域。
- 注意旧 `data_wss_min/*/bundle.npz` 若在本次前生成，仍是旧坐标，正式训练前必须全量重跑 `pipeline_wss_min.run --stage preprocess`。
