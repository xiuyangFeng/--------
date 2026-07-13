# WSS 最小化·第六轮边界条件与速度条件路线

> 状态：`CONDITIONAL / I1 等待 P0-Metric 与 P0-Density；V2 被 V0/V1 锁定`
> 更新：2026-07-13（最终审查收敛版）
> 定位：只维护 BC 与速度→WSS 的触发条件，不重复 WSS 主计划。

## 1. 已知数据边界

| 字段 | 覆盖 | 部署定位 |
|---|---:|---|
| 入口面积/模板流量派生速度 | 61/61 | 可计算，但与几何冗余 |
| 入口体积流量波形 | 61/61 | 全病例共享模板，无病人间信息量 |
| 入口/四出口压力 | 61/61 | CFD 产物，`oracle_non_deployable` |
| 四出口 RCR | 61/61 | CFD 设定，跨病例变异大，`oracle_non_deployable` |
| 出口分流 | 不完整 | CFD 解，当前不可部署 |

CFD 数值不得重命名为临床实测 BC。若新增临床数据，必须记录 `source=clinical_measured|estimated|cfd_oracle`、时间、单位、缺失规则和 CFD 时相对齐。

## 2. BC 路线

### I0｜只读残差关联

按 WSS 新合同检查：

- 病例 mean/p95/p99 bias；
- 每例中心化 spatial R²；
- 热点幅值/定位误差；
- 与 RCR、出口压力、分流、入口面积和 `coord_scale` 的 Spearman/排名关联。

I0 只用于生成假设，不作因果结论，不据此修改预处理阈值。

### I1｜oracle-BC 2×2 探针

前置条件：WSS 主线已完成 P0-Metric 和 P0-Density，产生冻结 control。

矩阵：架构 `{冻结 control/高容量或全局上下文}` × 输入 `{geometry/geometry+RCR}`，并加入：

- shuffled-RCR；
- RCR-only 或等维随机病例特征；
- 固定的出口/分支 permutation QA；
- duplicate-grouped 病例外 validation。

**Gate**：真实 RCR 必须在病例外稳定优于 shuffled-RCR/随机病例特征，才认定存在可泛化 BC 信息效应；只在 train 改善按病例记忆/身份编码处理。所有 I1 产物强制 `oracle_non_deployable=true`。

### I2｜临床可获得性

只核查真实来源：血压、超声/多普勒、4D-flow MRI、心率或出口分流。没有可部署且有病人间变异的来源，不启动“可部署 BC 输入”模型矩阵。

## 3. 入口裁剪/坐标边界

1. 继续按 `wall_crop_applied`、`wall_crop_frac`、fast/slow 和 `coord_scale` OOD 分层报告。
2. 核对裁剪后 wall/int 坐标范围、入口覆盖和中心线端点一致性。
3. 只有 train-only QA 支持时才建立新裁剪版本，并全量 preprocess、重做 split/stats/覆盖 QA。
4. 不在原 bundle 局部补丁，不混用新旧 bundle，不根据 val 反复调裁剪阈值。

## 4. 速度→WSS 路线

### 当前证据

- 单点近壁差分 oracle 约 `R²=0.166`，不作硬门槛。
- 法向速度剖面光滑拟合 cross-calibrated oracle 约 `R²=0.632`，说明存在信息，但尚未证明能稳定优于 direct-WSS。
- 历史有限差分会放大速度误差；全速度辅助监督曾与 WSS loss 竞争。
- 当前 `data_wss_min` 没有内部 `u/v/w`，需 Track B adapter 才能启动。

### V0/V1/V2 Gate

1. **V0 CFD-velocity oracle**：用 GT `u/v/w`、壁面法向和真实近壁采样重建 WSS，报告 A level、B spatial、C hotspot、病例等权 Pa 误差和失败率。
2. **V1 分辨率上限**：GT 速度降采样到实际预测密度后再重建 WSS，分离后处理上限与网络误差。
3. **V2 最小学习结构**：仅在 V0/V1 相对新版 direct-WSS control 稳定改善病例等权 Pa 误差与 B spatial 指标、且 C 不明显退化时启动。优先预测固定法向深度的有序近壁切向速度剖面，保留 direct-WSS head 与梯度一致性 loss。

速度横向 R² 高不代表 WSS 可由简单差分恢复。V1 必须复用横向速度的相同近壁点集和 FPS 索引；联合速度模型必须报方向夹角/余弦、模长和符号错误。

## 5. 当前顺序与待办

1. WSS 主线先完成 P0-Metric、P0-Density。
2. I0 可只读执行，但不改变主线。
3. I1 与主计划 2×2 F3 统一执行，不另建口径。
4. 只有明确触发速度→WSS 时才验收 Track B adapter 并执行 V0/V1。
5. V0/V1 No-Go 时停止，不启动 V2。

- [ ] P0 control 冻结后执行 I1。
- [ ] 仅在有真实临床来源时执行 I2。
- [ ] 仅在用户重新立项速度→WSS 时执行 V0/V1。
