# V5 / WSS-min 实验分析参考

用于 `training_wss_min` 的真实 run。**代码路径不决定数据版本**：V5 视图与旧 WSS-min 共用训练器，必须读 config。以下路径及字段已于 2026-09-12 用实际产物核对；后续 schema 变化以原文件为准。

## 定位与读取

V5 入口：`docs/02-推进与变更/WSS_PINN/README.md` 与其中的 `WSS_V5_训练实验跟踪.md`。先搜索批次名/臂名，读对应章节；设计合同只读与当前问题有关的部分。

| 产物 | 作用 |
| --- | --- |
| `training_wss_min/configs/<批次>/matrix.json` 或该批次 manifest | 臂定义、配置与同期对照；不是每批都有相同文件名 |
| `training_wss_min/experiments/<批次>/README.md`、报告、queue/submission/验收记录 | 本轮协议、决策与运行状态证据 |
| `training_wss_min/runs/<批次>/<run>/config.json` | 实际数据根、目标、split、模型/损失、采样、seed 和选模规则 |
| `history.jsonl`、`train.log`、`training_diagnostics.json` | 轮次、loss 分量、学习率、选择信号、异常；文件按实际存在读取 |
| `initialization.json`、`feature_stats_source.json`、`target_normalization.json` | 初始化、train-only 统计与归一化来源；可用时纳入可比性审查 |
| `eval/ckpt_best/metrics.json`、`eval/ckpt_last/metrics.json` | 两类 checkpoint 的既有评估 |
| 对应评估目录下 `per_case_metrics.csv`、`linear_fit_metrics.csv`、`predictions/` | 病例、拟合、配对诊断与必要的原始同点证据 |
| `acceptance*.json` 或批次验收报告 | 检查 passed、病例集合、checkpoint/指标匹配；已有 hash 可用于追溯，不为普通分析无故重哈希全部大资产 |

历史 run 也可能直接位于 `runs/<name>/`；优先用 manifest/config 定位，不猜目录。训练状态与评估状态分别报告。

## 直接 WSS 的指标字段

先选实际 JSON 中的分区键（如 `test` 或 `val`，下表记为 `P`）。V5 直接 WSS 的典型 `log_z` run 中，`P` 下的这些统计对应恢复后的 Pa 空间，`P.normalized` 是目标空间；需结合 `data.target` 与归一化配置确认。

| 对外名称 | `metrics.json` 相对字段 |
| --- | --- |
| 物理 Pa 病例等权场 R²（R²_cb） | `P.field_casebalanced.r2` |
| 物理 pooled 场 R² / MAE / RMSE | `P.field.r2` / `mae` / `rmse` |
| 物理病例 R² 均值 / P10 / 负例数 | `P.aggregate.r2_casemean` / `r2_casep10` / `r2_negative_cases` |
| 归一化 R²_cb | `P.normalized.field_casebalanced.r2` |
| top10 热点 IoU（病例均值） | `P.hotspot.top10_iou_casemean` |
| high-WSS 区域指标 | `P.regional_field.high_wss.*` |
| top10 / p99 预测幅值比 | `P.calibration.top10_pred_true_ratio` / `p99_pred_true_ratio` |
| 分域病例等权统计 | `P.group_casebalanced.*` |
| 线性拟合诊断 | `P.field` 或 `P.field_casebalanced` 下的 `r2_linear_fit`、`linear_fit_slope`、`linear_fit_intercept` |

注意 `per_case_metrics.csv` 的历史 `overall_*` 等兼容列可能指向归一化空间；分析 Pa 病例差异时优先选明确的 `physical_overall_*`，并与 JSON 交叉核对，不能按列名猜单位。

不要将 `R²_cb` 写成“逐病例 R² 平均值”。`legacy_vertex` 与 `both_strict` 是不同的顶点/面积口径；检查 `eval.surface_metric_mode`，不把插值 coverage 当作面积指标验收。CASE/WSSmax 目标没有物理恢复合同则不报告伪 Pa 指标。

## 体场目标与冻结验证器

同一 V5 训练器也用于压力、速度或其他目标；先检查 `data.target`、输出字段、压力 gauge、速度单位与归一化恢复规则，再读跟踪记录对应章节。不能把上表 WSS 字段机械套到所有体场 run。

直接 WSS 回归与 velocity→WSS 是两条链路。后者须注明上游体场 checkpoint、冻结算法版本、点级缓存、校准模型和评估协议；其数字不能冒充网络直接输出 WSS 的精度。冻结算法细节只从 `wss_mri_calculator/experiments/README.md` 对应版本读取。

## 比较和选模

- V5 当前批次常用 train138/test34；每次以 split 的实际病例 ID 和来源为准。它与历史 test35/test36 不相等；旧模型在交集上重评才可形成相同病例集合的对照，但仍需说明数据版本不同。
- test34 已参与多轮开发筛选，不得称作未触碰的独立确认集；用户要求比较既有 test34 结果可直接读取，不需因读取动作再次申请评估授权。
- `ckpt_best` 的选择可能是 train-loss/top-k 等具体协议；以 config/训练产物确认，不能称作 validation-best。`ckpt_last` 是该次训练末轮；两者不是两次独立重复。
- 损失不同的臂不能直接凭 total train loss 排名。查看共享 data loss、选模分量和正式评估指标；出现 nonfinite/AMP overflow 时区分可恢复事件与训练失败，按实际验收记录判断。
- 比较结构消融时列出 support/query 的 SAME/independent、点数、完整壁面 query patch、特征统计和输入条件等真实差异；不能只比较 config 中一个开关。

**2026-09-12 批次的定位示例**：`training_wss_min/experiments/wss_direct_recovery_20260912/`。分析该轮时读 `final_results_20260912.json` 的原始指标路径与 `queue_status.json` 的选择证据，再读相关 run。E0 为同期 independent-query 控制、E1 为 SAME-query 控制；需解释组合机制或协同增益时，再与各组成单臂比较。C4 包含自适应结构选择；历史 M2 与同期 E0 的差距是需要报告的控制漂移。这里不固定最新冠军或成绩。

## 产物与回填

分析报告和新图件沿用该实验目录的组织；没有约定时用 `training_wss_min/experiments/<批次>/analysis_<日期>/`，保留图表使用的数据字段、run/checkpoint/分区与原始路径。`history.png` 可直接复核；新增对比图使用 `history.jsonl`，不调用要求 V3 `summary.json` 的脚本。

V5 新结论更新 `WSS_V5_训练实验跟踪.md` 对应章节并在 `WSS最小化_代码修改与实验推进记录.md` 文首记录；07 月的 `WSS最小化_训练实验跟踪.md` / `WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md` 是旧档案，不承接 V5 新结论。xlsx 如需更新，先读 `docs/00-规范与记录/实验记录填写规范.md`，保留 best/last、空间和聚合方式。
