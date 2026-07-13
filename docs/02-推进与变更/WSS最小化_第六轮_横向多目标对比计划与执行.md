# WSS 最小化·第六轮横向多目标对比计划与执行

> 状态：`H-PW DONE / 其余 Track B 暂停`
> 更新：2026-07-13（最终审查收敛版）
> 定位：压力是独立辅助诊断；速度只在明确触发速度→WSS 问题时恢复。

## 1. 当前结论

1. 固定 peak 的 WSS-min 是唯一 P0 训练主线；压力/速度不与 WSS 共训，也不作为 WSS 准入条件。
2. 壁面 gauge pressure H-PW 已完成，说明同一最小协议下压力空间型态比 WSS 更易拟合；该结论不能外推为 WSS 的信息上限。
3. H-PW 使用了旧 WSS 复合 checkpoint 规则，因此结果只作探索性基线；正式对外报告前需做压力专用 val-only 只读重评。
4. H-PI、H-U/H-V/H-W/H-M 和 Track B adapter 全部暂停，不再为了补齐 36-run 表消耗资源。
5. V3P 压力和 WSS 与 wss_min 协议不同，不能直接用头条 R² 排名或计算精确 gap。

## 2. 冻结比较协议

- split：`split_AG_wss_min_v2_dev1.json`，train 53 / val 8。
- 时相：`peak step 1162 / WSS-min index 21`。
- 采样：每例 FPS-2000；评估在各目标冻结域进行。
- 模型：PointNeXt-S，单目标独立训练。
- 输入消融：`xyz` vs `xyz+abscissa_norm+local_radius+curvature`。
- seed：`1234/7/2025`。
- 数据范围：val-only，不读 test16。
- 目标分别使用独立 train-only 归一化和 checkpoint 规则。

“同一协议”只表示 split、点数、主干规模和输入消融可比，不代表压力、速度与 WSS 共用定义域、统计或护栏。

## 3. 已完成 H-PW 结果

任务：逐病例去均值的壁面 gauge pressure，线性 train-only z-normalization；MAE 单位 Pa。

| 输入 | 三 seed `R²_field` | 旧 `R²_casemean` | MAE | 当前判读 |
|---|---:|---:|---:|---|
| `xyz` | `0.426±0.021` | `0.254±0.042` | `261` | 坐标已有可辨识空间信号 |
| `xyz+geom` | `0.531±0.053` | `0.509±0.015` | `225` | 显式几何稳定改善压力空间型态 |

与同协议 WSS 历史结果相比，`xyz+geom` 压力 `0.531/0.509` 高于 WSS `0.311/0.197`。这只支持“当前协议下 gauge pressure 更易学”，不能证明差距完全来自物理内禀难度，也不能据此外推 WSS 可达上限。

### H-PW 对外确认前必须补齐

1. 对现有 top-k/last checkpoint 做压力专用 val-only 只读重评。
2. 主报病例等权/每例 gauge spatial R²、RMSE/MAE、Spearman。
3. 补压差误差、压力梯度误差和失败病例。
4. 不再使用 WSS top10/IoU 作为压力选模或护栏。
5. 若压力专用最优 epoch 未保存，现有结果继续标记探索性，不重训改写历史。

## 4. 目标专用指标合同

| 目标 | 主指标 | 护栏 |
|---|---|---|
| gauge pressure | 病例等权/每例 spatial R²、RMSE/MAE、Spearman | 压差、梯度、失败病例 |
| `|v|` | R²、RMSE/NRMSE、Spearman | 低/中/高速分层误差 |
| 联合 `u,v,w` | 分量 R²、RMSE、方向夹角/余弦、模长误差 | 坐标帧与符号错误 |
| WSS | 以[WSS 主计划 §3](WSS最小化_第六轮_WSS精度突破计划与执行.md)为准 | level/pattern/hotspot 分列 |

未来任何横向新训练启动前，目标专用指标必须同时进入训练 history、checkpoint 规则和最终 Gate；不得训练时使用 WSS 复合分数、报告时再切换指标。

## 5. 与 V3P 的可比性

### 压力：只作任务背景

V3P `r2_p≈0.92–0.96` 使用绝对压力、多时相和 CFD BC；wss_min H-PW 使用单 peak、逐例去均值 gauge pressure且不输入 BC。二者目标、R² 分母和输入信息不同，当前不得定量直比。

唯一允许的补充动作是：对冻结 V3P checkpoint 做逐 `case × time` 去均值只读重评，同时报告 absolute pooled R²、gauge spatial R²、gauge MAE/RMSE、level MAE 和压差误差；不重训、不用 test 选模。

### WSS：仅为协议化参考

V3P 历史 `wss_r2_wss≈0.429` 与 wss_min `R²_field_raw≈0.31–0.36` 虽同为 raw wall WSS magnitude 的点级 R²，但仍存在以下差异：

- split/报告集：V3P test16 vs wss_min dev1-val8；
- 时相：多时相 pooled vs peak 单帧；
- 点集：wall13000+near2000 vs wall FPS-2000；
- 输入：V3P 含 `t_norm` 和 CFD BC，wss_min 只含可部署几何；
- 目标/训练：P+WSS 多任务 vs WSS magnitude 单头；
- checkpoint 规则不同。

因此只能说两条路线都处于中低点级 WSS R² 区间，不能报告精确 gap。真正比较前必须统一以上六项，并按 WSS 新 level/pattern/hotspot 合同重评。

## 6. 当前需要做与不做

### 需要做

- [ ] 仅在需要对外确认 H-PW 时，执行压力专用只读重评。
- [ ] 仅在明确要解释 V3P 压力组成时，执行冻结 checkpoint 的 gauge/level 分解。

### 暂不做

- H-PI 内部压力矩阵。
- 独立 H-U/H-V/H-W 分量矩阵。
- H-M `|v|` 与联合 `u,v,w` sanity。
- 压力+WSS 联训、共享 backbone 或压力辅助 loss。
- 为补齐表格而启动 Track B adapter。

速度→WSS 的 V0/V1/V2 触发条件只在[边界条件与速度路线](WSS最小化_第六轮_边界条件与速度路线.md)维护。
