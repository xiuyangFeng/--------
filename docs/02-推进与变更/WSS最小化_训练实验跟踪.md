# WSS 最小化路线 · 训练实验跟踪

> 用途：跟踪 `training_wss_min/`（PointNeXt 残差 baseline，独立于 V3P `training/`）的逐轮实验：
> 设计、完整指标表、结论、待办。**每完成一轮/一次任务，回填本文档。**
> 上位：[WSS最小化_代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) / [training_wss_min/README](../../training_wss_min/README.md)。

## PointNet E4 deeper 追加探针与三层 SA foundation（2026-07-15 ✅完训完评｜No-Go）

- `E4-DEEP-GLOBAL` 严格复用 E2-GLOBAL 的 train61 / 固定 FPS-2000 / global log-z / 400 epoch / train-loss 选模协议，仅将网络改为 `6→64→128→256→512；1024→512→256→128→64→1`；参数量 `874,561`，相对 E2 增加 10.3%。
- Job `8999` 已 `COMPLETED (0:0)`，用时 `01:37:11`；400 epoch 完整，best=第 379 epoch / train loss `0.121590`，比 E2 的 `0.172548` 降低 29.5%。best/last train61+test16 全点评估齐全，best 的 16/16 PostView 包完整，mapping coverage 100%，无 Traceback/OOM/NaN。
- PointNet++ fine-tune 本轮未启动。前置三层 SA 结构审计已完成：FPS-2000 后中心数 `500/125/32`，radius `0.05/0.10/0.20`，nsample=16；`fast/RAN_QING_BO` 的覆盖率 `99.90%/100%/100%`。完整 PNG、VTP、assignment CSV 与 manifest 位于 `例子/06_PointNet++_SA三层采样与分组/`。

| best 配对指标 | E2 | E4 | E4−E2 |
| --- | ---: | ---: | ---: |
| train / test 物理 `R²_cb` | 0.6124 / **0.2140** | **0.6966** / 0.1629 | +0.0841 / **−0.0511** |
| 物理 train−test gap | 0.3985 | 0.5337 | +0.1352（变差） |
| train / test 归一化 `R²_cb` | 0.7717 / **0.4606** | **0.8375** / 0.4476 | +0.0658 / −0.0130 |
| test 物理 MAE / RMSE | **2.8005 / 5.1141** | 2.8510 / 5.2778 | +0.0505 / +0.1637 |
| test high-WSS R² / top10 幅值比 | **−1.486 / 0.378** | −1.680 / 0.333 | 均变差 |
| test Spearman / top10 IoU | **0.661** / 0.146 | 0.657 / **0.153** | −0.004 / +0.007 |
| test 双 self-max `R²_cb` / 负例 | −4.857 / 16 | **−4.208** / 16 | 小幅改善，仍 No-Go |

**结论**：E4 把 train-fit 做得更好，却使物理/归一化 test R² 下降、gap 扩大、MAE/RMSE 和 high-WSS 幅值恢复变差；16 例中物理 R² 仅 6 例改善、10 例退化。top10 IoU 和 self-max 的小幅好转不足以抵消整体泛化恶化，因此导师追加深度探针判为 **No-Go**，保留 E2 为 PointNet 锚点，不继续纯深度扫描。`ckpt_last` 的 test 物理/归一化 R²=`0.1618/0.4451`，与 best 同结论。test16 已被反复使用，本结果只作为导师驱动的同协议配对证据，不表述为无偏最终测试。完整表见[PointNet 矩阵 §4.4](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#44-导师追加深度探针-e4-deep-global2026-07-15-完训完评no-go)。

## PointNet baseline 新矩阵（2026-07-15 ✅五组完训+完评）

- 当前只保留 `PointNet+xyzgeom`；共同协议为无 val/无早停/400 epoch。
- 正式父实验分为导师通道对齐的容量组和 5000 点 random 不放回/每 epoch 重采样组；两组各做全局与逐病例归一化配对。
- 该矩阵的公式讨论、Job 状态、high-risk 评价与每例可视化产物统一转到[PointNet baseline 实验矩阵与进度跟踪](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)；本文仅保留已完成的原 baseline 数值与其他路线证据。
- **2026-07-14 实现回填**：`N-CASE` 冻结为 `WSS/WSSmax`；E2/E3 的 GLOBAL/CASE 四份主配置、`E23-GLOBAL`（精确导师宽网 + random-5000）交互配置、random-5000 协议校验、best/last 隔离评估和 test16 STL/VTP 流水线均已写入。
- 训练阶段新增 sampled normalized MSE/MAE/RMSE；完整点云 R²、Spearman、top10 与热点位置仍由训练后的 eval 计算。`ckpt_best(train_loss)` 为主报告和 PostView，`ckpt_last` 只作指标审计。
- **Slurm 结案（2026-07-15）**：`8976–8980` 五个 Job 均 `COMPLETED (0:0)`；5/5 run 的 best/last train61+test16 全点评估齐全，best 共导出 80/80 个 test case PostView 包，surface mapping coverage 为 100%。

| Run | 变量 | test 主空间 `R²_cb` | Spearman | top10 IoU | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `E0-GLOBAL` | 原 PointNet + FPS-2000 | 物理 0.1414 / norm 0.3964 | 0.635 | 0.128 | 共同对照 |
| **`E2-GLOBAL`** | **导师宽网** | **物理 0.2140 / norm 0.4606** | 0.661 | 0.146 | 本轮最强；容量有效，但热点仍 No-Go |
| `E3-GLOBAL` | random-5000 | 物理 0.1637 / norm 0.4218 | 0.645 | 0.124 | 点数/重采样单独增益弱 |
| `E23-GLOBAL` | 宽网 + random-5000 | 物理 0.1988 / norm 0.4598 | **0.674** | 0.148 | 未超过 E2，无明确协同 |
| `E2-CASE` | E2 + `WSS/WSSmax` | norm 0.1724 | 0.574 | 0.140 | **No-Go** |
| `E3-CASE` | E3 + `WSS/WSSmax` | norm 0.1551 | 0.559 | **0.159** | 局部 IoU 改善但整体分布 **No-Go** |

**本轮结论**：导师宽通道是主要有效因素；random-5000 不是稳定主增量，且与加宽没有明确协同。逐病例 `WSS/WSSmax` 使 R²、Spearman、p99 与动态范围明显退化，虽在 E3 上出现局部 hotspot IoU 改善，仍不足以支持替换全局 log-z。E2 的 physical high-WSS R² 仍为 −1.486、top10 幅值比仅 0.378，所以只判为相对改进，不判为可部署 Go。完整配对、best/last 审计和失败病例可视化见[独立矩阵文档 §4.2](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#42-正式矩阵结果2026-07-15)。

**导师 self-max 补充（2026-07-15）**：五组 80 个既有 test PostView 已无推理回填 `WSScfd/WSScfd,max` 与 `WSSpred/WSSpred,max`。五组的 16/16 逐病例 self-max R² 均为负；相对较好的 E2-GLOBAL pooled/case-balanced R² 仍为 −4.802/−4.857，说明去掉绝对幅值后空间型态仍未学准。CASE 线性输出还产生负点：E2/E3-CASE 病例平均占 7.34%/6.03%。定义、完整表、负值解释和 VTP/图件路径见[独立矩阵文档 §4.3](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#43-导师补充指标cfdcfdmax-对-predpredmax2026-07-15)。

**下一轮暂不自动执行**：以 E2-GLOBAL 为锚点，讨论顺序冻结为开发协议（train61 内 group-dev/repeated holdout）→ 推理期可得 BC/病例级信息审计 → shape/scale 拆分与热点 loss → 局部拓扑表示 → 几何分层采样。random-5000、原样 `WSS/WSSmax` 和继续盲目加宽不列为优先项；详见[独立矩阵文档 §9](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#9-下一阶段优化方向待讨论不自动开跑)。

## PointNet 无 val / train_loss 选模 / 400 epoch（2026-07-14 ✅完训+完评｜`8968` / eval `8974`/`8975`）

- 协议：`split_AG_wss_min_v1_traintest`（61/0/16）、train_loss 选模、400 epoch、FPS-2000、seed=1234。
- **主结果 `R²_field_cb`**：

| 臂 | train61 | test16 | gap | test 负例 | test high-WSS R² |
|---|---:|---:|---:|---|---:|
| xyz | 0.5704 | 0.0825 | 0.488 | 5/16 | −1.88 |
| **xyz+geom** | 0.5303 | **0.1414** | 0.389 | 5/16 | −1.76 |

- **结论**：No-Go（泛化）；几何臂 test 略好，保留 xyz+geom；train-fit 有容量信号但选模协议不适合作精度主线。产物：`runs/pointnet_trainloss_e400/outputs/*/eval/metrics.json`。

## PointNet 加宽容量探针（2026-07-14 ✅完训｜Job `8970`｜No-Go）

- 相对 2×3 最佳格只改容量：`PointNet + xyz+geom`，`width=128` / `head_hidden=256`，其余同冻结 val 协议。
- **val8 `R²_field_cb=0.3071`** vs 锚点 **0.3015**（Δ≈+0.006）；RMSE/MAE/热点几乎持平。
- **结论**：单变量加宽无实质增益；下一步按矩阵做老师通道对齐与 5k random，不再扩宽本结构。

## 最小 2×3 baseline 矩阵（2026-07-14 ✅DONE｜Job `8700[0-5]` 全部 `0:0`）

- 重新从最基础形式起步：`MLP / PointNet / PointNet++` × `xyz / xyz+geom`，共 6 个单 seed（1234）作业；`xyz+geom=xyz+abscissa_norm+local_radius+curvature`。
- 统一使用 `split_AG_wss_min_v1` 的 AG 划分（train/val/test=`53/8/16`），只训练和选择 `val`，暂不读取 test；峰值收缩期、WSS 单标量、FPS-2000、完整壁面 val 推理。
- 训练仅用未加权标准化空间 MSE，关闭旋转增强、采样/几何/目标加权、多任务和 raw-space 辅助项；PointNet++ 为经典 SA+FP，无 PointNeXt 残差/倒置瓶颈。
- 实验包位于 `training_wss_min/runs/baseline_2x3_simple/`：`make_configs.py` 固化 6 份配置，`submit.sh` 提交最多 4 并发的 Slurm array，所有产物落在同目录 `outputs/`。`8700[0-5]` 于 2026-07-14 依次完成，6/6 均 `COMPLETED (0:0)`；全程只做 val，`test16` 未读。

| 模型·输入 | `R²_field_cb` | `R²_field_raw` | case mean / median / P10 R² | 负例 | RMSE / MAE (Pa) | high-WSS R² | top10 比 / IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP · xyz | 0.1980 | 0.2043 | 0.0080 / 0.1114 / −0.3414 | 1/8 | 4.424 / 2.615 | −1.3788 | 0.361 / 0.189 |
| MLP · xyz+geom | 0.2442 | 0.2632 | 0.0669 / 0.0496 / −0.2142 | 3/8 | 4.257 / 2.476 | −1.0972 | 0.447 / 0.182 |
| PointNet · xyz | 0.1727 | 0.1704 | 0.1194 / 0.0699 / −0.0554 | 2/8 | 4.517 / 2.454 | −1.6736 | 0.313 / 0.150 |
| **PointNet · xyz+geom** | **0.3015** | **0.3122** | **0.1842 / 0.1404 / +0.0143** | **1/8** | 4.113 / 2.291 | −1.1690 | 0.457 / 0.235 |
| PointNet++ · xyz | 0.2169 | 0.2367 | 0.0156 / 0.0694 / −0.3146 | 2/8 | 4.333 / 2.583 | −1.0329 | 0.461 / 0.207 |
| PointNet++ · xyz+geom | 0.2587 | 0.2668 | 0.1287 / 0.1427 / −0.0950 | 1/8 | **4.247 / 2.281** | −1.4213 | 0.387 / 0.226 |

**判读**：几何特征对三种架构的 `R²_field_cb` 均有增益（MLP `+0.046`、PointNet `+0.129`、PointNet++ `+0.042`）。本矩阵的最强且最稳单格是 **PointNet + xyz+geom**：主指标/pooled R² 最高，逐病例 P10 唯一为正，且只 1/8 负例。PointNet++ 的 `xyz+geom` MAE 最低但主 R² 较 PointNet 低 `0.043`；在纯 xyz 条件下 PointNet++ 的主指标最好（0.217），但仍明显低于加入几何后的 PointNet。六格 high-WSS R² 全为负、top10 幅值比仅 `0.313–0.461`，说明基础网络能学到中低 WSS 空间趋势，却仍严重低估高 WSS，不能把单 seed 结果视为最终结论或架构终裁。

**后续边界**：这轮完成“最简单形式”的单 seed baseline，不扩展模块、不访问 test；若要把 PointNet+xyz+geom 作为正式比较锚点，需要另行决定是否补多 seed。

### baseline 壁面 ParaView 包（2026-07-14 ✅DONE｜`8966` / `8967` 均 `0:0`）

- 原 `8961[0-5]` 的全量 6×8 导出在用户收窄范围后已取消，已生成的全量/冒烟产物已清理；不产生腔内切片，不读取 test。
- 当前范围只保留当前最佳 baseline `PointNet+xyz+geom` 的两个 val 病例，按同点逐病例 R² 排序：最佳 `slow/CHENG_LU_LI=0.3959`，最差 `slow/XU_YI_CAI=-0.0577`。单作业 `8966` 与依赖汇总 `8967` 均 `COMPLETED (0:0)`。
- 每个 `surface_wall.vtp` 同时挂载 `wss_cfd`、`wss_pred`、`err_wss`、`abs_err_wss`（Pa），以及四个 `*_over_cfd_max` 字段。归一化的分母固定为**同一病例 CFD 壁面最大 WSS**，所以 CFD、预测与误差可在 ParaView 中用同一 0–1 标尺比较。
- CFD 和预测同用 Gaussian `r=3 mm, sharpness=2, max_dist=3 mm` 回插到同一经配准变换的 STL；另写 `map_dist` / `map_valid`、mapping report、同点 wall CSV、CFD/Pred/signed-error 三联预览。正式 R² 始终只读同点 CSV。
- 两个最终 VTP 均验证 10 个字段齐全、mapping coverage=100%。`CHENG_LU_LI` 的 CFD max=48.595 Pa、同点 R²=0.3959；`XU_YI_CAI` 的 CFD max=36.417 Pa、同点 R²=−0.0577。最终批次输出与打开说明见 `training_wss_min/runs/baseline_2x3_simple/postview/README.md`；`comparison.csv` 与 `comparison_by_run.csv` 已生成。

## 第六轮指标/调度修订（2026-07-13 ✅DONE｜覆盖下方旧的 36-run 默认调度）

- WSS W0–W3 恢复为 P0 主线；横向 36 runs 仅为理论上限，不再是必做表。
- H-PW 已完成并收口。Track B 先完成 adapter/QA，默认只执行 `|v|+geom` 和联合 `u,v,w+geom` 单 seed sanity；其他目标/三 seed 按 Gate 触发。
- WSS 首要点级指标改为 `R²_field_casebalanced` + 物理单位 RMSE/MAE，并强制同报逐病例中位数/P10/失败数和热点护栏。`R²_field_raw` 与 `R²_casemean` 保留为历史衔接。
- V3P 0.429 与 wss_min 0.31–0.36 只作 B 级协议化参考，不报精确 gap。具体见[跨路线评估口径](../00-规范与记录/WSS跨路线评估与横向对比口径.md)。
## 第六轮对抗性审查回写（2026-07-13 ✅DONE｜覆盖旧版 F3 调度）

- **P0 先修指标合同**：现有 `R²_casemean` 是逐病例 spatial R² 的平均，不是病例 mean-WSS 的跨病例 R²。新合同拆成 A 病例 level、B 病例内 pattern、C hotspot，以及病例等权 Pa 误差；文档、`evaluate/train/gate/checkpoint` 完全一致并补单元测试前，不启动新训练。
- **E 结论降级**：E−D `field_cb +0.021` 只是在 dev1-val8/旧选模合同下通过开发筛选；在新版指标只读重评和 duplicate-grouped repeated validation 前，E 是临时候选，不称为已确认最佳可部署输入。
- **P0 密度修复前置**：`nsample=16`/完整点云评估的密度错配先做同 seed 单变量实验，形成新的冻结 control；旧版受混杂的 `~0.54` train-fit 不直接进入信息天花板裁决。
- **F3 改为 2×2**：架构 `{冻结 control/高容量或无下采样}` × 输入 `{geometry/geometry+RCR}`，加入 shuffled-RCR、RCR-only/随机病例特征负对照，同时报告 train-fit 与病例外 validation。信息效应和架构效应允许并存；取消单一 `train R²≥0.85` 二分。
- **当前顺序**：`P0-Metric → P0-Density → 2×2 F3 → 分支优化 → repeated validation`。oracle 特征继续强制 `oracle_non_deployable=true`；压力/速度保持条件路线。

## 全链路基础检查（2026-07-12 ✅DONE｜当前标量主线无致命错位，下一轮先补 3 项基础 Gate）

- 已从预处理、配准/正交旋转、逐病例缩放、节点 ID 对齐、FPS 稀疏化、训练特征/标签索引、模型/激活/loss/选模和完整点云评估逐项检查。
- 77/77 included bundle 齐全；`det(R)≈1`、最大正交误差 `4.44e-16`、逆变换最大误差约 `7.17e-05 mm`；数组同长，跨时间步 `nodenumber`/坐标守卫全过；FPS-2000 后坐标与标签同索引误差为 0；train-only peak stats 与 split 无泄漏。
- 结论：既有 `R²≈0.31` 不能归因于“基础坐标/标签整体错位”。优先风险是 3 个 included 病例 roll-sign 不可靠但 QA gate 未告警、ball-query 半径固定在逐病例归一化尺度而非毫米尺度、预留速度路径尚缺 cell ID/裁剪同步守卫。
- 激活函数 GELU + 线性输出合理，不是当前首要提分项；下一轮优先级为严格尺度进入 FPS/ball-query、log+raw/case-balanced loss、密度鲁棒邻域，再做 global-local 和网络宽度/激活微调。
- 完整报告：[WSS最小化_全链路基础检查报告_2026-07-12](WSS最小化_全链路基础检查报告_2026-07-12.md)。本轮不读新 test16、不改代码/配置、不启动新训练。

## 第六轮｜A/B/D XYZ 尺度诊断（2026-07-12 ✅DONE｜尺度信号 B−A 成立，几何仍是主杠杆）

- 目的：判断旧纯 XYZ 较差是否主要由“坐标逐病例归一化到 `[-1,1]`”丢失物理尺度造成。
- A=`xyz`；B=`xyz+coord_scale`；D=`xyz+abscissa_norm+local_radius+curvature`。三组均为 dev1 / fixed FPS-2000 / B1 fixed target-weight / val-only / `seed={1234,7,2025}`。
- 预注册：B−A 的三 seed 均值在 field/casemean 均 `>0.02` 才判为可辨识尺度信号；D−B 报告显式几何增量和 seed 方差。
- 范围声明：B 是尺度诊断，不是严格物理尺度纯 XYZ 终审；本轮不读 test16，不外推几百/几千例数据上限。
- 详细协议与 Job 表：[WSS最小化_第六轮XYZ尺度诊断计划与执行](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)。

### ⚠ 首批提交（7029–7037）5/9 因 config 路径失效而失败，已重跑补齐

- 首批提交记录 `submitted_20260712_120420.txt` 用的是**旧目录名** `configs/round6/r6_scale_*.json`；但同日“目录规整”已把这批 config 迁到 `configs/xyz_scale_diag/scale_*.json` 并删除 `round6/`。
- 早启动的 `7029–7031`(A×3) 与 `7032`(B_s1234) 赶在删除前解析成功；`7033`(B_s7) 训练成功但评估阶段目录已删 → 崩溃；`7034`(B_s2025)、`7035–7037`(D×3) 载 config 即 `FileNotFoundError`，**从未训练**。根因为提交清单路径与实际目录不一致，与协议/数据/模型无关，未污染任何已完成 run。
- 修复：按正确 manifest `configs/sweeps/xyz_scale_abd.txt` 重跑——`7039`(B_s2025)、`7040–7042`(D×3) 完整 train+eval，`7043`(B_s7) 复用 ckpt 仅重评（新增 `cluster/run_eval_only.slurm`）。记录 `cluster/logs/resubmit_20260712_001011.txt`。**5 个重跑均 `COMPLETED (0:0)`，9/9 eval 齐全。**

### 完整结果（val 完整壁面点云；MAE 单位 Pa）

| 组·seed | Job | `R²_field` | `R²_casemean` | `R²_casemed` | 负例 | top10 比 | IoU | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A·1234 | 7029 | +0.089 | −0.032 | +0.042 | 3/8 | 0.236 | 0.226 | 3.591 |
| A·7 | 7030 | +0.178 | −0.083 | +0.063 | 2/8 | 0.287 | 0.156 | 3.519 |
| A·2025 | 7031 | +0.227 | −0.141 | +0.024 | 4/8 | 0.333 | 0.241 | 3.497 |
| **A `xyz` 三 seed** | | **+0.164±0.070** | **−0.085±0.055** | | | 0.285±0.048 | 0.208±0.045 | 3.536 |
| B·1234 | 7032 | +0.173 | −0.010 | +0.068 | 3/8 | 0.309 | 0.148 | 3.710 |
| B·7 | 7043 | +0.204 | −0.080 | +0.147 | 2/8 | 0.335 | 0.159 | 3.563 |
| B·2025 | 7039 | +0.248 | +0.115 | +0.114 | 1/8 | 0.352 | 0.230 | 3.342 |
| **B `xyz+coord_scale` 三 seed** | | **+0.208±0.038** | **+0.008±0.099** | | | 0.332±0.022 | 0.179±0.044 | 3.538 |
| D·1234 | 7040 | +0.326 | +0.208 | +0.242 | 1/8 | 0.406 | 0.286 | 3.113 |
| D·7 | 7041 | +0.308 | +0.208 | +0.206 | 1/8 | 0.385 | 0.302 | 3.087 |
| D·2025 | 7042 | +0.298 | +0.174 | +0.143 | 1/8 | 0.362 | 0.375 | 3.129 |
| **D `xyz+geom` 三 seed** | | **+0.311±0.014** | **+0.197±0.020** | | | 0.384±0.022 | 0.321±0.048 | 3.110 |

**对照（三 seed 均值差）**：B−A field `+0.044` / casemean `+0.093`；D−B field `+0.103` / casemean `+0.189`；D−A field `+0.146` / casemean `+0.282`。

### 判读（终裁）

1. **尺度信号 B−A 成立。** B−A field `+0.044`、casemean `+0.093`，两项三 seed 均值均过预注册 `+0.02` 门槛；且**逐 seed 方向一致**（matched-seed field 增量 +0.084/+0.026/+0.021，casemean +0.022/+0.003/+0.256，3/3 seed 为正）。**结论：逐病例 `[-1,1]` 归一化确实丢失了对 WSS 有用的病例物理尺度，补回 `coord_scale` 标量能稳定回收一部分——主要体现在跨病例可辨识性（casemean 由 −0.085 转正到 +0.008）。**
1. **旧开发口径下尺度信号 B−A 成立。** B−A field `+0.044`、旧 per-case mean R² `+0.093`，三 seed 方向一致。该结果说明 `coord_scale` 提供了可用上下文，但旧 `casemean` 不是病例 level R²，不能据此宣称“跨病例整体水平被恢复”；病例 level 结论等待新版指标只读重评。
2. **但尺度标量只补回约三成缺口，几何仍是压倒性主杠杆。** 从 A(0.164)→B(0.208)→D(0.311)：coord_scale 把 field 抬 `+0.044`，而显式几何（D）再抬 `+0.103`（D−B）、相对 A 共 `+0.146`。与第一轮“纯几何≈xyz+几何≫纯 xyz”的结论一致。
3. **与第四轮 C4「coord_scale No-Go」不矛盾，反而互补澄清。** C4 是在 `xyz+geom` 之上再加 coord_scale（`local_radius` 已带尺度 → 冗余无增量）；本轮 B 是在**无 geom 的裸 xyz** 上加 coord_scale（非冗余 → 有增量）。两者一起说明：**尺度信息本身有用，但一旦有 `local_radius` 等局部几何，标量尺度基本被覆盖**。这提示 C（严格物理 mm-XYZ）要想跑赢 D，必须靠“物理尺度同时进 FPS/ball-query 邻域”带来的、局部几何特征无法替代的增量，而非仅把 mm 尺度塞进 feature。
4. **绝对精度和稳健性仍不足。** 本轮最好的 D（`xyz+geom`，三 seed field `0.311±0.014`、casemean `0.197±0.020`）的高 WSS 护栏（top10 比 0.384 / IoU 0.321）延续“定位有信号、幅值系统性低估”。尺度诊断解释了旧纯 XYZ 差的一部分成因，但未证明当前协议的绝对信息上限；`0.70` 仅作长期理想参考。

### 交叉验证与下一步（详见[第六轮总路线](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)与其拆分文档）

- 已对第六轮各新思路做证据交叉核对；内容随文档重构分入[WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md)与[边界条件与速度路线](WSS最小化_第六轮_边界条件与速度路线.md)。
- **用户批准的下一步优先级（2026-07-12）**：`L1/L2 loss` 与 `C/E 严格尺度` **并列第一**；架构 `M1/M2/M3` 次之；速度→WSS 只做 `V0/V1` oracle 复核 0.632 天花板，**oracle 未过门前不建 data_new adapter、不训速度 surrogate**。
- **多目标扩展（2026-07-13 修订）**：原 36 runs 保留为理论上限，不再是必做主表。压力-壁面 H-PW 已完成；Track B 的 data_new adapter 完成 QA 后，默认只跑近壁 `|v|+geom` 和联合 `u,v,w+geom` 单 seed sanity，其余实验按机制/论文需要触发。近壁速度与速度→WSS oracle 复用同一 adapter。完整方案与结果见[横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)。

### 结果返回后的预注册后续（待用户批准）

- 当前 A/B/D 九个作业的 config、Gate 和评估不变；不在运行中修改协议。
- 尺度补充：C=严格 mm-XYZ（输入/FPS/ball-query 同时使用物理尺度）；E=`XYZ+coord_scale+geom`，与 A/B/D 构成嵌套因子对照。
- 输入信息：先做残差↔RCR/压力/分流只读诊断和 oracle-BC 上限；临床实测、估计与 CFD oracle 强制分开。
- 架构：global-local/FiLM、density-robust neighborhood、解析尺度 residual 和小模型对照；不先扩大参数量。
- loss：当前 B1 vs `log + raw scaled-Huber` vs case-balanced raw robust 的三 seed 小矩阵。
- 速度→WSS：先做 CFD-velocity oracle 和降采样上限；当前光滑剖面 oracle 约 `R²=0.632`，未过 `0.70`，不直接训全速度 surrogate。
- 详细 Gate、数据边界和分阶段顺序见[第六轮总路线与执行入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)。

## 第六轮｜W0 审计 + W1 因子(E) + C 邻域预审计 + W2-L1 + W3 组合（2026-07-12 ✅DONE｜E 险胜、尺度/几何冗余、raw-Huber seed 脆弱、W3 组合阴性）

> 承接上节 A/B/D，本节补 **W0 只读审计**、**E 组**（补齐 A/B/D/E 2×2 嵌套因子）、**C 邻域预审计** 与 **W2-L1 raw-Huber λ Gate**。协议全程冻结 B1/dev1/FPS-2000/val-only，seed=1234/7/2025。作业 `7557–7562`。产物：`runs/_audits/{w0_coord_scale,w1_c_neighborhood,round6_w1w2_summary}/report.md`。

### W0｜coord_scale 只读审计（`runs/_audits/w0_coord_scale/report.md`）

- **val 无尺度外插**：val `coord_scale ∈ [147, 214]` 全落在 train `[128, 262]` 内；dev1 **无任何壁面裁剪**（`wall_crop_applied` 全 False）。
- **coord_scale 是血管尺寸代理，但与 WSS 幅值无关**：Spearman(coord_scale, bbox 对角线)=+0.74、(z 向 extent)=+0.79、(入口面积代理)=+0.61、(local_radius 中位)=+0.65；而 (WSS case mean)=−0.15、(p95)=−0.03、(p99)=+0.02。**结论：B 组 coord_scale 的增益来自跨病例可辨识性/归一化上下文，不是幅值定标**——这与下面 casemean 的抬升方向一致，也解释了为何 B 组抬 casemean 多于 field。
- **coord_scale 是血管尺寸代理，但与 WSS 幅值无关**：Spearman(coord_scale, bbox 对角线)=+0.74、(z 向 extent)=+0.79、(入口面积代理)=+0.61、(local_radius 中位)=+0.65；而与 WSS case mean/p95/p99 的相关约为 −0.15/−0.03/+0.02。结论仅限于“它提供尺寸/归一化上下文”，不再用旧 `casemean` 推导病例 level 机制。

### W1｜A/B/D/E 2×2 因子（E 补齐，三 seed 均值；MAE 单位 Pa）

| 组 | 输入 | R²_field | R²_field_cb | R²_casemean | R²_casemed | 负例(∑/24) | top10 比 | IoU | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | xyz | +0.164±0.057 | +0.164±0.050 | −0.085±0.044 | +0.043 | 9 | 0.285 | 0.208 | 3.536 |
| B | xyz+scale | +0.208±0.031 | +0.213±0.036 | +0.008±0.081 | +0.110 | 6 | 0.332 | 0.179 | 3.538 |
| D | xyz+geom | +0.311±0.012 | +0.309±0.016 | +0.197±0.016 | +0.197 | 3 | 0.384 | 0.321 | 3.110 |
| **E** | **xyz+scale+geom** | **+0.331±0.014** | **+0.330±0.014** | **+0.199±0.018** | **+0.204** | **2** | **0.404** | **0.341** | **3.094** |

**严格归因（逐 seed 配对差，mean±std）**：

| 对比 | 含义 | ΔR²_field | ΔR²_field_cb | ΔR²_casemean |
|---|---|---:|---:|---:|
| B−A | 尺度主效应 | +0.044±0.028 | +0.049±0.019 | +0.093±0.115 |
| D−A | 几何主效应 | +0.146±0.069 | +0.145±0.066 | +0.282±0.031 |
| E−B | 几何｜有尺度 | +0.123±0.044 | +0.118±0.049 | +0.191±0.098 |
| E−D | 尺度｜有几何 | +0.020±0.009 | +0.021±0.012 | +0.002±0.011 |
| E−D−B+A | 尺度×几何交互 | −0.023±0.033 | −0.027±0.031 | −0.091±0.115 |

**判读**：

1. **E 是当前最佳可部署输入，但仅险胜 D。** E 三 seed field `0.331`、field_cb `0.330`、casemean `0.199`、**负例仅 2/24（全组最少）**、top10/IoU 均最高。E−D 在 field_cb `+0.021`（过预注册 Gate-1 `+0.02`），负例不增、top10/IoU 反升，**E 干净通过 Gate-1，取代 D 成为最佳输入**。
2. **尺度与几何冗余、非协同（关键新结论）。** 交互项 E−D−B+A 三项**全为负**（field −0.023 / casemean −0.091）；尺度的边际价值从"无几何"的 B−A(+0.044 field/+0.093 casemean) 坍缩到"有几何"的 E−D(+0.020 field/**+0.002 casemean**)。**一旦有 `local_radius` 等局部几何，coord_scale 标量的贡献基本被吸收**——量化印证上节 §判读3 与 C4 No-Go 的猜想。W0 也从另一侧印证：coord_scale 编码的是尺寸而非幅值，几何特征已覆盖其可用信息。
3. **E 的增益虽小但一致**：E−D field 逐 seed +0.009~+0.031 全正，主要抬 field/pooled 与热点护栏，对 casemean 无增量。

1. **E 是 dev1 旧口径下的临时候选。** E 三 seed field `0.331`、field_cb `0.330`、旧 per-case mean R² `0.199`、负例 2/24；E−D field_cb `+0.021` 通过旧开发筛选，但尚未通过新版指标与 grouped repeated validation，不再称为最佳可部署输入。
2. **尺度与几何无协同证据。** 交互项 E−D−B+A 为负；在已有 `local_radius` 等显式几何时，`coord_scale` 的边际收益很小。该结论限于 dev1/旧指标，不外推为普遍机制。
3. **E 的增益虽小但一致**：E−D field 逐 seed +0.009~+0.031 全正，主要抬 field/pooled 与热点护栏，对 casemean 无增量。

> ⚠ **2026-07-13 代码核验更正**：上文"E 干净通过 Gate-1、取代 D 成为最佳输入"依据的是文档 `field_cb` 口径；但提交的 `gate1_compare.py` 判 GO 需 `Δr2_field>0.02 且 Δr2_casemean>0.02`，`field_cb` 被加载却不参与判定。E−D 的 `Δr2_casemean=+0.002` 使 `common_improvement=False` → 代码实际判为 **INDIFFERENT/NO_GO**；且产出 E 的 checkpoint 由 composite（`0.6·casemean+0.4·field`）选出。故"E>D"仅为探索性，**不作冻结 control/最佳输入结论**；须先统一"文档 Gate=代码 gate1=选模 rule"口径。详见 [WSS 精度突破 §2026-07-13（第二轮·本地代码核验）](WSS最小化_第六轮_WSS精度突破计划与执行.md)。

### W1｜C 组邻域预审计（`runs/_audits/w1_c_neighborhood/report.md`，只读未训练）

- C 定义：train 拟合全局共享常数 `s_global=max(train coord_scale)=261.9` 缩放所有病例，固定物理半径。复现 PointNeXt-S 的 SA 级联对比 A/C 每层邻居。
- **Gate 结论**：① **C 不产生退化邻域**（A/C 所有层孤立率=0.000，无空邻域）→ 邻域结构上可训；② **C 更密而非更稀**（训练分辨率邻居中位约 A 的 1.5–1.9×）；③ **`nsample=16` 截断已主导**（评估分辨率 A/C 截断率≈1.00，训练细层≈0.93–1.00），把 C"物理尺度进邻域"的预期收益大部分抹平，真正分化只存活到最粗层。④ 结合 W1 已证尺度/几何冗余，**C 先验跑赢 D 的理由弱，判为低优先级（低于 E/L1）**；若仍训 C 需先调 radius 协议（提高 nsample 或改 median 参考的 s_global）。

### W2-L1｜raw-Huber λ Gate（固定 D 输入，单 seed 1234，`7560–7562`）

log-z MSE 主损失 + raw-space scaled Huber 辅助。L0=`r6_scale_D_xyzgeom_s1234`（λ=0）。

| run | λ | R²_field | R²_field_cb | R²_casemean | top10 比 | p99 比 | IoU | high_wss_MAE | max 比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.0 | +0.326 | +0.331 | +0.208 | 0.406 | 0.369 | 0.286 | 13.85 | 0.14 |
| L1 | 0.1 | +0.374 | +0.376 | +0.245 | 0.447 | 0.478 | 0.347 | 13.03 | 0.17 |
| L1 | 0.3 | +0.362 | +0.362 | +0.235 | 0.421 | 0.370 | 0.330 | 13.48 | 0.14 |
| **L1** | **1.0** | **+0.385** | **+0.384** | **+0.253** | **0.463** | 0.450 | **0.360** | **12.61** | 0.16 |

**判读**：**三个 λ 全面优于 L0**（field/casemean/top10/IoU/high_wss_MAE 无一退化），且**无爆峰**（max 比 0.14–0.17，远低于 1.5 阈值）。λ=1.0 最佳（vs L0 同 seed：field +0.059、casemean +0.045、top10 +0.057、high_wss_MAE 13.85→12.61）。单 seed 仅筛选，**选中 λ=1.0 补三 seed 确认**。注意 high_wss_MAE 只降 ~9%、max 比仍 ~0.16——raw-Huber 抬中高段与整体 R²/定位，但**极端峰值坍缩仍未解决**（延续第二轮"峰值压扁是硬上限"判断）。

### W2-L1 λ=1.0 三 seed 确认（`7563–7564`+s1234）+ W3 唯一组合（`7565–7567`，2026-07-12 ✅DONE）

**L1 λ=1.0 三 seed（D 输入，loss-control）**：单 seed s1234 的 field 0.385 属"幸运高 seed"，三 seed 均值回落：

| 配置 | 输入 | λ | R²_field | R²_field_cb | R²_casemean | 负例(∑) | top10 比 | IoU | high_wss_MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D (baseline) | xyz+geom | 0.0 | +0.311±0.012 | +0.309±0.016 | +0.197±0.016 | 3 | 0.384 | 0.321 | 14.36 |
| D+rawHuber | xyz+geom | 1.0 | +0.347±0.033 | +0.346±0.035 | +0.199±0.054 | 4 | **0.429** | 0.310 | **13.37** |
| E (input-control) | xyz+scale+geom | 0.0 | +0.331±0.014 | +0.330±0.014 | +0.199±0.018 | **2** | 0.404 | 0.341 | 13.90 |
| **W3 = E+rawHuber** | xyz+scale+geom | 1.0 | +0.316±0.037 | +0.315±0.036 | +0.189±0.009 | **2** | 0.400 | **0.351** | 14.08 |

**W3 Gate（配对 seed 差）**：W3−E field `−0.015±0.032` / top10 `−0.004`；W3−(D+rawHuber) field `−0.030±0.022` / top10 `−0.029`；两侧仅 IoU 微升（+0.011 / +0.042）。

**判读（W3 = 阴性，按 §6 停止组合）**：

1. **W3 未通过组合 Gate。** 逐 seed 看，W3 在**全部 3 seed 低于 D+rawHuber**（field_cb 0.366/0.294/0.286 vs 0.384/0.300/0.353），在 **2/3 seed 低于 E**（仅幸运 seed s1234 反超）。W3 在主指标（field/field_cb/casemean/top10）不优于任一控制，只在热点 IoU 微升。**按 §6："组合未同时优于两个 control → 不再扩展交互组合"，正式停止 scale-feature × raw-Huber 的组合线。**
2. **机制：尺度特征与 raw-space Huber 冗余/轻微互斥。** 两者都在推高 WSS/物理尺度表示，叠加不增益反而略降——与 W1 尺度×几何交互为负一致。
3. **L1 raw-Huber 收益真实但 seed 脆弱。** D+rawHuber 三 seed 抬 field +0.036、top10 +0.045、high_wss_MAE −7%（幅值定向有效），但 casemean 持平、+1 负例、方差大（field_cb ±0.035，主要靠 s1234）。**单 seed 0.385 不可外推**——印证 §5 W2"单 seed 只用于廉价筛选"。
4. **两条互斥候选，均未跨越稳健性瓶颈。** E（最稳、负例最少、IoU 最高）与 D+rawHuber（field/top10 幅值最好但脆弱）**不可叠加**，且都未改善 casemean/负例。**下一步交由用户裁决 W5 确认候选（E vs D+rawHuber），或启动尚未尝试的 L2 case-balanced robust loss 直击病例稳健性**。汇总真源：`runs/_audits/round6_w1w2_summary/report.md`。
4. **两条互斥候选，均未跨越稳健性瓶颈。** E（最稳、负例最少、IoU 最高）与 D+rawHuber（field/top10 幅值最好但脆弱）**不可叠加**，且都未改善旧 per-case mean R²/负例。最终审查后两者均冻结，不再直接进入 W5 或 L2；先完成 P0-Metric、P0-Density 与 2×2 F3。汇总真源：`runs/_audits/round6_w1w2_summary/report.md`。

## 第六轮 横向对比 H-PW Track A｜压力-壁面（2026-07-12 ✅DONE｜压力空间型态比 WSS 易学）

- 目标：同最小协议（FPS-2000/PointNeXt-S/dev1/B1 schedule）预测壁面 **gauge 压力**（peak-step `wall_pressure` 逐例去均值，隔离 ~1.5e4 Pa 的 DC 偏置），`xyz` vs `xyz+geom` 矩阵，纯 MSE（关掉 WSS 长尾加权）。
- 数据/代码：管线支持 `target` 切换（原硬编码 `wall_wss`）；gauge stats `pressure_gauge_stats_v2_dev1.json`（53 例/707705 点，std=529.2 Pa，gauge∈[-1880,1164]）；配置 `configs/multitarget/press_wall_{xyz,xyzgeom}_s{1234,7,2025}.json`（6 个）。
- Jobs `7551–7556` 全部 `COMPLETED (0:0)`（记录 `submitted_20260712_050410.txt`）；CPU 冒烟先行通过。

| 组·seed | Job | `R²_field` | `R²_casemean` | 负例 | top10 比 | IoU | MAE(Pa) |
|---|---:|---:|---:|---:|---:|---:|---:|
| press·xyz·{1234,7,2025} | 7551–3 | +0.406/+0.424/+0.448 | +0.286/+0.207/+0.271 | 1/2/3 | — | — | ~261 |
| **press `xyz` 三 seed** | | **+0.426±0.021** | **+0.254±0.042** | | 0.601±0.058 | 0.298±0.021 | 261 |
| press·xyzgeom·{1234,7,2025} | 7554–6 | +0.506/+0.495/+0.592 | +0.493/+0.522/+0.513 | 0/0/0 | — | — | ~225 |
| **press `xyz+geom` 三 seed** | | **+0.531±0.053** | **+0.509±0.015** | | 0.575±0.046 | 0.389±0.097 | 225 |

- **判读**：同协议下 gauge pressure 空间型态比 WSS 易学——`xyz+geom` 压力 `0.531/0.509` vs WSS `0.311/0.197`；`xyz` 压力 `0.426/0.254` vs WSS `0.164/−0.085`。压力 casemean 全程为正、`xyz+geom` 下 **0/8 失败例**（WSS 每 seed 都有负例）。几何增益 field `+0.105`/casemean `+0.255`。该结果不使用跨目标通用 `0.70` 门槛，不外推其他目标。
- **选模**：沿用 WSS 复合选模，列为**探索性**；只读复核显示复合最优 epoch 与压力 R² 最优 epoch 一致，数值应接近压力专用选模；正式复选待用户批准（预计不改数值）。
- 完整表与读法：[横向多目标对比 §8](WSS最小化_第六轮_横向多目标对比计划与执行.md#8-h-pw-结果壁面-gauge-pressure2026-07-12-done探索性选模)。Track B（压力-内部 + 近壁速度）待 data_new adapter。
- **选模**：沿用 WSS 复合选模，列为**探索性**；只读复核显示复合最优 epoch 与压力 R² 最优 epoch 一致。仅在需要对外确认 H-PW 时，按压力专用指标做 val-only 只读重评；Track B 保持暂停。
- 完整表与当前读法：[横向多目标对比 §3](WSS最小化_第六轮_横向多目标对比计划与执行.md#3-已完成-h-pw-结果)。

## 汇报｜A0E-ctrl 两例 postview（2026-07-12 ✅DONE）

- 模型 `r5_a0e_b1_ctrl_s1234`；病例 `slow/WU_FENG_YAN`、`fast/RAN_QING_BO`（均为 train）。
- 产物：`docs/03-汇报材料/figures/WSS最小路线_20260712/postview_a0e_ctrl/`（`*__surface_wall.vtp` 含 CFD/Pred/Error；映射覆盖率 100%）。
- 同点 wall R²：`0.3225` / `0.4959`；脚本 `training_wss_min/tools/export_wss_postview.py`。

## 第五轮｜F0 结案（2026-07-12 ✅DONE / 科学结案·工程未达标）

- 全部 §8.1 机制问题均有可复核裁决 → **第五轮科学结案**；dev1 field/casemean ~0.31/0.21 ≪ 0.70 → **未达内部工程目标**（§8.2），未跑 OOF（无达标候选）。
- 机制链：拟合足（A0D）→ 当前协议泛化锚点 ~0.34（A0E）→ 现有 61 例范围内 ~0.31 暂时平台（LC）→ 可部署 BC 无新杠杆（B-BC）→ 标签噪声小（CFD，R²_cap ~0.92–0.96）。该结论限于当前数据池、输入和模型协议，不外推数千个高质量独立病例的上限。
- 报告：`training_wss_min/runs/_round5/final_report/round5_final_report.md`。可选后续（非部署路径、待用户定）：oracle RCR 增量探针 / P2 loss 探针。
- 归档口径：用户确认无达标候选时不强行运行 15-run OOF；L0/OOF/T16 经记录未触发，`test16` 保持未读；P2 转交第六轮 loss 小矩阵。见[第五轮归档说明](_archive/WSS最小化/WSS最小化_第五轮结案与归档说明_2026-07-12.md)。

## 第五轮｜CFD 可信性审计 §6（2026-07-12 ✅DONE）

- read-only；dev61。产物 `training_wss_min/runs/_round5/cfd_audit/{cfd_audit.py,cfd_per_case.csv,cfd_summary.json,cfd_audit_report.md}`。
- peak 相位按**固定步**（1162/idx21）统一取，结构上无跨病例相位错配；仅 3/61 真峰晚 ≥12 步（含 2 val：`LIU_JUN_FENG` +13.8%、`CHENG_GUANG_SEN` +12.2%）→ n=8 val 数值脆弱。
  - ⚠ **2026-07-13 与固定 peak 的交互**：这 2 个 val 病例（占 dev1-val8 的 25%）在固定步与真峰之间有约 12–14% 的幅值差；`CHENG_GUANG_SEN` 又是 A0R 常见失败例。固定 peak 协议保持不变，但新版病例级 level 报告必须把两例的真峰敏感性单列，不得据 val 结果重新选择时相。见 [WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md)。
- 近壁 QA 全清（normal_invalid=0、basis≤0.066、radial≤0.066、mesh≥786k）；高 WSS 尖峰为真实几何热点（不与 QA 旗标共现），但单节点 max 受尖峰主导，报告以 p95/p99 为准。
- **重复几何**：独立证实 `HOU_SHEN_QIAN`=`KANG_XI_MING` 同一几何（且均 dev1 train）→ OOF 须整组；`LIU_XI_QUAN`（excluded）peak 全零损坏。
- **裁决：~0.31 平台主要不能由简单 CFD 复现噪声解释。** 复现 floor ~2%，即便按 10–15% per-case 噪声估算，理论 cap 仍约 0.92–0.96；但该审计不能区分模型容量、密度协议、缺失条件信息和目标定义，不再写成“信息上限已确认”。

## 第五轮｜B-BC 资产审计（2026-07-12 ✅DONE / 可部署 B-BC 关闭）

- read-only 审计 61/61 dev 病例 BC 完整（udf-inlet.c + vf-in + 5 压力监测）。产物 `training_wss_min/runs/_round5/bc_audit/{audit_ag_bc.py,bc_per_case.csv,bc_summary.json,bc_audit_report.md}`。
- **决定性发现：入口流量不是病人特异的。** Fourier 流量模板（a0..b8,w,T,A1/B/D/E/n）在所有病例逐字节相同；入口 BC 为平速 `v=1e-6·template/area`，故 `Q=v·area=1e-6·template` 与面积无关 → 各病例入口流量本质相同（dev CoV≈1.85e-4，Fourier↔实测比 0.999–1.000）。唯一 per-case 入口标量是入口**面积**（几何可得）。
- **可部署 vs oracle**：可部署 = 入口面积/速度/流量（但零方差或与几何冗余，非信息量）；**唯一有跨病例方差的是出口 RCR（CoV 0.55–0.67）与出口压力/流量分配，全部 `oracle_non_deployable`**（CFD 设定/解产物，新病人不可知）。
- 数据质量旗标：两对拷贝 BC（`HOU_SHEN_QIAN=KANG_XI_MING` 均 dev1 train、`LIU_XI_QUAN=LI_BING_YI`）→ 其面积/RCR 不独立；`vf-outri` 20 例退化（15 在 dev1）；两例异常入口监测均已 excluded。
- **裁决：可部署 B-BC 关闭**——现有资产中没有可部署、有信息量且病人特异的 BC 输入。出口 RCR 只有方差证据，尚未证明能在病例外解释 WSS 残差；最终审查后仅允许把它放入带 shuffled/RCR-only 负对照的 2×2 oracle 探针，不能写成已确认根因。

## 第五轮｜LC 三链 learning curve（2026-07-12 ✅DONE）

- G1 批准后启动；A0E-ctrl（标准 B1，min_lr=1e-5）配方，val 恒为 dev1 固定 8 例；3 链嵌套 `LC13⊂26⊂40⊂53`，硬分层 cohort×train-only WSS 三分位；case-drop（chain salt）与 model seed 分离。生成器 `make_configs_round5_lc.py`，per-subset 划分/WSS stats，feature stats 与 loss 分位运行时重算。
- 30 个 run（3 链×3 seed×{13,26,40} + 3 端点）Job `6994–7002`/`7003–7022` 全部 `COMPLETED`。
- **field R² 均值±std：13 `0.258±0.023` / 26 `0.297±0.025` / 40 `0.312±0.033` / 53 `0.310±0.044`**；casemean `0.087/0.158/0.177/0.208`。
- **配对 field 增量：13→26 +0.039±0.037、26→40 +0.014±0.027、40→53 −0.002±0.049**——**field 在当前 13–53 例范围内约 40 例后出现 ~0.31 暂时平台**，40→53 增量与 0 不可分。
- 端点 53 逐 seed field `0.359/0.252/0.319`（std 0.044，超过 26→53 整段增量）；stage-1 单 seed 的"53 仍在上升"是 s1234 偏高伪影，补 seed 后纠正——多 seed 必要性再次印证。
- **裁决（修正后口径）：现有 61 例池内的小步扩展没有显示可将 field R² 从 ~0.31 推到 0.70 的证据**。剩余差距与输入信息、坐标/尺度表示和 loss 目标错位有关；13–53 例 LC 不能否定几百/几千例高质量独立数据的潜在收益。高 WSS 护栏全程未改善（端点 top10 ratio 0.466 / IoU 0.302）。
- 产物：`training_wss_min/runs/_round5/learning_curve/{learning_curve_report.md,lc_points.csv,lc_curve.png,lc_verdict.json}`。下一步见 G2 路由：B-BC 输入信息（首选，先 read-only 审计）+ P2 loss 探针（并行）；不启动 L0/OOF（当前配置远低于 0.70）。

## 第五轮｜A0E dev1 control 重锚（2026-07-12 ✅DONE）

- 目的：判定历史 `0.34` 是真实泛化上限还是训练预算伪影，并冻结 LC 训练协议。control=历史 B1 anchor（field/casemean `0.3511/0.2138`）。
- `ctrl`（B1 逐字，min_lr=1e-5）Job `6992`：field/casemean **`0.3587/0.2300`**，best epoch 29，负 R² 病例 0——复现 anchor（±0.02 内），确认 `0.34/0.23` 是真实泛化上限。
- `nsl`（仅抬 LR 下限到 2e-4）Job `6993`：field/casemean `0.3419/0.2125`，2 例负 R²（失败率 0.25）——非饿死 LR 对 dev1 无益且略有害。
- 裁决：LC 用标准 B1 schedule（min_lr=1e-5，即 ctrl）；更正预注册时"dev1 需非饿死 LR"的假设。A0D"按 optimizer step 计预算"规则仍成立，但 dev1（batch8、约 1120 step、best-val 时 LR 健康）本就满足。产物 `training_wss_min/runs/_round5/a0e_control/a0e_control_report.md`。

## 第五轮｜G1 第二次分支裁决（2026-07-12 ✅DONE）

- 合并 A0D（拟合能力 GO）+ A0E（`0.34` 真实泛化上限）+ A0R（53 例即过拟合）+ A1（密度受 mapping 阻断）证据。
- 裁决：**批准 LC**（泛化/病例数主导）；**关闭 B-REP**（表示坏了前提被证伪）；维持 B-DEN/B-BC `BLOCKED`；normalized-MSE↔raw-R² 目标错位（79.75×）列 P2 并行探针。
- 产物：`training_wss_min/runs/_round5/branch_experiments/branch_decision_g1.md`。

## 第五轮｜A0D 基础拟合链（2026-07-12 ✅GO）

- 只读审计：stats/逐点对齐无异常；AMP 首步跳过不足以解释缺口；旧四病例 micro 只有 160 次 optimizer updates。
- Job `6986`：四个单病例简化 MLP 全部达 R² `0.991–0.999`。
- Job `6990`：four-case shared plain MSE，field/casemean `0.979526/0.982533`，逐例最低 `0.975705`。
- Job `6991`：只恢复 fixed target-weight alpha2，field/casemean `0.993723/0.992423`，相对 6990 `+0.014197/+0.009890`。
- 结论：`0.844` 不是四个已见病例的拟合上限，target-weight 单独不是原缺口的充分原因。这些仍是 train-only/canonical-2000 结果，不代表新病例精度。
- 产物：`training_wss_min/runs/_round5/a0d_fit_chain/`。下一步为预注册的 dev1 val-only 单变量候选。

## 第五轮｜B-REP/C2 global context（2026-07-11 ⛔NO_GO）

- Job `6984`：四病例 train-only、fixed FPS-2000、seed1234、last checkpoint，`COMPLETED (0:0)`，未读 val/test16。
- train `R²_field_raw=0.740682`、`R²_casemean=0.695389`，未达 `0.95/0.95`，dev1 未提交。
- C2 `NO_GO`；已列 B-REP 候选均未通过 micro Gate，暂停新候选训练、LC 和 B-DEN，转入 normalization/loss/标签对齐与几何可辨识性审计。

## 第五轮｜B-REP/C1 radius normalization（2026-07-11 ⛔NO_GO）

- Job `6983`：四病例 train-only、fixed FPS-2000、seed1234、last checkpoint，`COMPLETED (0:0)`，未读 val/test16。
- train `R²_field_raw=0.803401`、`R²_casemean=0.760602`，未达 `0.95/0.95`，dev1 未提交。
- C1 `NO_GO`；下一个仅执行 C2 单输出 global context 的四病例 micro。

## 第五轮｜B-REP/M1 逐点 MLP（2026-07-11 ⛔NO_GO）

- Job `6982`：四病例 train-only、fixed FPS-2000、seed1234、last checkpoint，`COMPLETED (0:0)`，未读 val/test16。
- train `R²_field_raw=0.880817`、`R²_casemean=0.844018`，虽高于 B1 micro，仍未达 `0.95/0.95`，所以 dev1 未提交。
- M1 `NO_GO`；STL 表面特征受 mapping Gate 阻塞，下一个只执行 C1 radius-normalized relative position 的四病例 micro。

## 第五轮｜A1 与 G0（2026-07-11 ✅已裁决）

- density probe 确认同索引预测会随 canonical/full 密度变化（三 seed 预测间 R² `0.865–0.882`），但相对真值的两项主指标方向不跨 seed 一致。
- full density 的 SA L1–L4 query cap 截断率约 `0.9998/1/1/1`，密度敏感有明确结构证据。
- 真值 IDW3 oracle 通过：field `0.9293`、casemean `0.9174`、top10 ratio `0.9353`、IoU `0.8003`。
- 但预注册 STL mapping 总 Gate 为 `0/8`；失败集中在连续三角面采样到离散 CFD wall 节点的 1 mm 覆盖门槛，因此 A1 `NO_GO`并阻断 D1/D2。
- G0 只批准 B-REP 的逐点 MLP 候选；先用已锁定四病例重跑 micro-overfit，通过后才能提交 dev1 Gate-1。

## 第五轮｜A0R 多 checkpoint 只读诊断（2026-07-11 ✅DONE）

- 范围：B1 `s{1234,7,2025}` 的 15 个 best/last/candidate checkpoint，train/val × canonical-2000/full；无训练、无 test16。
- best canonical train `R²_field_raw=0.538–0.573`、`R²_casemean=0.494–0.528`；容量/基础拟合不足信号明确。
- last 相对 best 的 canonical train field 平均 `+0.070`，canonical val field 平均 `-0.061`，继续训练不是同时修复拟合与泛化的答案。
- best full-canonical：train field/casemean 平均 `-0.159/-0.171`，val 平均 `-0.055/-0.080`，top10 ratio/IoU 也在三 seed 中全部下降；密度迁移方向一致。
- 最集中的 val 失败病例是 `slow/XU_YI_CAI`，其次为部分 seed/checkpoint 的 `slow/CHENG_GUANG_SEN`。产物见 `training_wss_min/runs/_round5/a0_readonly/`。

## 第五轮｜A0M 四病例 micro-overfit（2026-07-11 ⛔NO_GO）

- 四例固定为 fast/low `SUN_ZHI_YU`、fast/high `WANG_DAO_CHUN`、slow/low `ZANG_YU_SHU`、slow/high `MA_TIAN_YI`；全部来自 dev1 正式 train。
- Slurm Job `6981`：B1 配方、seed1234、fixed canonical FPS-2000、160 epoch、last checkpoint、无 val 选模、未读 test16；状态 `COMPLETED (0:0)`。
- train `R²_field_raw=0.76548`、`R²_casemean=0.72381`，未达同时 `>=0.95` 的 Go 阈值，因此 A0M `NO_GO`。
- 产物：`training_wss_min/runs/_round5/a0_micro/`；按停止规则暂停 LC/大规模 sweep，待 A0R/A1 后进入 G0。

## 第五轮｜P0 评价协议（2026-07-11 ✅DONE）

- 评价现同时输出 pooled `R²_field_raw`、逐病例等权 `R²_casemean` 与病例总权重相等的 `R²_field_casebalanced`，并固化 median/P10/负 R² 数/失败率。
- Gate-1 仅在 field/casemean 相对 control 都改善 `>0.02` 且 top10 ratio/IoU 均未下降 `>0.05` 时为 Go；不再允许高 WSS 单指标旁路 Go。
- `evaluate` 默认 val-only；test 必须额外显式 `--allow-test`。P0 未访问 legacy test16。
- 旧 control `r4_dev1_b1_tgtw_fixedq_s1234` / best epoch 49 的 val-only 复评：`R²_field_raw=0.3510821344`、`R²_casemean=0.2138313493`、`MAE=3.0635919684`，与历史值最大偏差 `3.24e-9`；新 `R²_field_casebalanced=0.3451643087`。
- 产物：`training_wss_min/runs/_round5/protocol/protocol_report.md`、`protocol_regression.json`。下一步可并行 A0R/A0M/A1。

## 第四轮补充｜点数—精度曲线（§14，2026-07-10 ✅单 seed + 多 seed）

- **单 seed 曲线**：见下表；峰值曾在 1000，但属单点。
- **多 seed 确认（1000 vs 2000 × {1234,7,2025}）**：

| n | R²_field | R²_casemean | top10 | IoU | score |
|---:|---|---|---|---|---|
| 1000 | 0.337±0.017 | **0.221±0.019** | 0.413±0.032 | 0.327±0.011 | 0.276±0.025 |
| 2000 | 0.339±0.020 | 0.188±0.025 | 0.416±0.025 | 0.324±0.016 | 0.259±0.026 |

- **裁决**：仅 R²_casemean 三 seed 一致偏向 1000；R²_field/top10/IoU/score 持平或不一致 → **默认锚点仍为 2000**；1000 为更稀采样候选。Jobs 6976–6979 + 既有 6969/6960。
- **产物**：`runs/_summary_round4_pointcount/`（含 `pointcount_multiseed_*`）。
- 归档计划：[第四轮执行总结与归档 §14.5](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。

### 单 seed 6 点曲线（历史，seed1234）

| n | R²_field | R²_casemean | top10 | IoU | score | 来源 |
|---:|---:|---:|---:|---:|---:|---|
| 1000 | **0.351** | **0.241** | **0.448** | 0.330 | **0.301** | Job 6969 |
| 1500 | 0.327 | 0.208 | 0.408 | 0.296 | 0.262 | 6970 |
| 2000 | **0.351** | 0.214 | 0.436 | 0.330 | 0.283 | B1 6960 |
| 3000 | 0.298 | 0.150 | 0.365 | 0.315 | 0.205 | 6971 |
| 4000 | 0.259 | 0.207 | 0.346 | 0.296 | 0.227 | B3 6962 |
| 6000 | 0.335 | 0.207 | 0.412 | **0.343** | 0.268 | 6972 |

## 第四轮状态（Stage A→B→C 单 seed 已执行，2026-07-10）

- **协议**：v2_dev1 开发划分 + fold stats；val-only；`persistent_workers=False`；固定 train 分位权重；复合选模 + early stop（160 / patience=6）。
- **B 组**：B0/B1 持平（B1 为协议胜者）；B2 multi-start、B3 FPS4000 均 Gate-1 No-Go（B3 `R²_field` 大跌；§14 后改为补全曲线而非永久不开 6000）。
- **C 组**：C1 raw-Huber、C4 coord_scale No-Go；C2/C3/C5 按条件跳过（A3 smearing 失败；C5 Ridge 增量≈0）。
- **A5 补齐**：raw top10 差 + 邻域 cap（全量 cap≈0.97，子采样≈0.37）。
- **覆盖**：旧 v1 审计 fixed2000 high-WSS hit ~10%、multi-start 40ep union ~34%；dev1 对齐审计见 Job 6968。
- **当前锚点**：`r4_dev1_b1_tgtw_fixedq_s1234`（val R²_c=0.214 / R²_f=0.351 / top10=0.436）。未扩 3 seed、未跑 legacy test。
- 归档计划：[WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。

## 第四轮状态（计划 v2.3 终审通过，历史快照）

- 开发阶段仍须 val-only；`test16` 仅保留为最后一次的 legacy benchmark。
- A5 只读诊断已完成**三 seed 标准化口径**（终审独立复现，seed1234 与原引用逐位一致）：8 个 val 病例、同一 FPS-2000 点上比较子采样/全量推理同索引输出，MAE/RMSE/Pearson/mean-shift：s1234 `0.1645/0.2278/0.9381/-0.050`、s7 `0.1413/0.1884/0.9605/-0.033`、s2025 `0.1191/0.1638/0.9717/-0.066`。三 seed 方向一致：全量推理系统性偏高 `0.03–0.07σ`，且点数最多的病例漂移最大。证据：`docs/02-推进与变更/assets_第四轮/a5_density_probe.{py,csv}`。这证明推理密度敏感，尚不能推出全量评估不正确或采用下采样插值作为修复。
- 下一步固定为 Stage A：split/bundle 完整性、worker-safe sampler、固定 train-only loss 阈值、A5 剩余项（同索引 raw top10 差 + 邻域 cap 审计）、残差校准、repeated-holdout；不直接启动 NLL、法向、rot_aug 或 6000 点训练。
- 现行状态入口：本跟踪文档与 [WSS最小化代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md)；第四轮计划已归档为[执行总结](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。

## 任务定义与通用设置
- **任务**：几何点云 `(x,y,z[+几何]) → 壁面 WSS 标量`，全局 `log_z` 归一化，单头。矢量三分量为二期。
- **路线 A（部署导向）**：部署有完整几何、缺 CFD 标签 → 训练用稀疏子采样，**评估恒在完整壁面点云上**。
- **模型**：PointNeXt-S 残差版（InvResMLP + ball-query，密度鲁棒）。约 4.4M 参数。
- **当前数据口径**：split `split_AG_wss_min_v1`（train 53 / val 8 / test 16 / excluded 9 / pending 1）；坐标逐病例 [-1,1]；WSS 为 train-only、peak-only 全局 `log_z`（std=1.0907）。第一/二轮旧 stats 仅作历史对照。
- **第三轮训练**：AdamW + cosine（warmup 10）+ AMP，240 epoch，batch 8 病例，eval_every 10；best 按 `val_r2_casemean`。第一/二轮 400 epoch 设置见各轮记录。
- **集群**：GPU 分区 `master`（4×RTX4090），`submit_baseline_sweep.sh` 提交，自动排队 4 并行；第三轮每 config 实测约 3–4 min 训练，随后做完整点云评估。
- **产物**：`training_wss_min/runs/<name>/`（`train.log`、`history.jsonl`、`ckpt_best/last.pt`、`config.json`、`feature_stats.json`、`eval/metrics.json`、`eval/per_case_metrics.csv`、`eval/heatmaps/`）。
- **汇总**：`python -m training_wss_min.summarize` → `runs/_summary/{summary.csv,pointcount_curve.png,regional_bar.png}`（聚合全部轮次）。

## 指标口径
- 均在**原始 WSS 空间**（denormalize 后）计算。
- `R²_field`：所有点 pool 起来算（受高 WSS 病例主导）；旧 `R²_casemean`：逐病例分别计算 spatial R² 后再平均，虽为病例等权，但不是病例 mean-WSS 的跨病例 level R²。
- 分区：`bifurcation`(距原点≤0.25) / `stenosis`(local_radius 最小 20%) / `high_wss`(原始 WSS 前 10%)。

---

## 第三轮 clean-data ✅完训并完成判读（2026-07-10，Slurm 6952–6958）

**目的**：移除污染病例、重算 peak-only 统计后，以相同 `xyz+geom / FPS 2000 / PointNeXt-S` 配方比较 MSE 与 target-weight（α=2），各跑 3 个 seed，判断数据清理和尾部加权是否真正改善完整点云预测。

**数据侧已完成**：
- split 已更新为 train 53 / val 8 / test 16 / excluded 9 / pending 1；`slow/ZHANG_HUAN_LI` 已移入 excluded，`slow/ZHAO_XIU_XUAN` 保持 pending，excluded/pending 不参与正式数据集。
- scope guard 已加硬：`--all-raw` 仅允许诊断性 `preprocess`，正式 `qa-gate/global-stats/build-samples/all` 均按 split，pending/excluded 历史 bundle 不会进入 stats、训练或评估口径。
- `preprocess` 已重跑 77 个 included bundle：Slurm Job **6952**，`ok=77 / skipped=0 / error=0`；`nodenumber/cellnumber` 对齐守卫落盘，`nodenumber_reordered_cases=0`，`wall_coord_mismatch_cases=0`。
- QA gate：fatal=0，warning=3（`fast/ZHANG_QING_WANG`、`slow/GUAN_TONG_XIANG`、`fast/RAN_QING_BO` 仅 `trunk_centering_offset_frac>0.05`，按最终诊断 P2 作为复核项，不默认剔除）。
- 新 stats：train peak-only，53 cases / 711412 wall points，zero_frac=0，log mean/std=`1.1595 / 1.0907`，raw p90/p99/max=`12.8847 / 32.6039 / 220.7968`。

**模板基线（test，完整点云）**：

| 配置 | R²_field | R²_casemean | high_wss R² | top10 pred/true | p99 比 |
|---|---:|---:|---:|---:|---:|
| template_mean_clean | -0.114 | -0.139 | -2.533 | 0.172 | 0.118 |
| template_voxel_clean | 0.035 | -0.023 | -2.058 | 0.260 | 0.392 |
| template_knn_clean | -0.039 | -0.225 | -1.994 | 0.289 | 0.576 |

**单 run 结果（test，完整点云；best 仍由 val `R²_casemean` 选择）**：

| Job / 配置 | R²_field | R²_casemean | bif | stenosis | high_wss | top10 比 | p99 比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6953 / `mse_s1234` | 0.210 | 0.200 | 0.040 | -0.186 | -1.591 | 0.359 | 0.437 |
| 6954 / `mse_s7` | 0.187 | 0.163 | 0.012 | -0.168 | -1.669 | 0.332 | 0.425 |
| 6955 / `mse_s2025` | 0.175 | 0.165 | -0.003 | -0.255 | -1.673 | 0.319 | 0.421 |
| 6956 / `tgtw_s1234` | **0.251** | 0.220 | **0.097** | -0.129 | **-1.366** | **0.393** | **0.486** |
| 6957 / `tgtw_s7` | 0.186 | 0.182 | 0.002 | -0.273 | -1.699 | 0.320 | 0.410 |
| 6958 / `tgtw_s2025` | 0.239 | **0.234** | 0.064 | **-0.097** | -1.528 | 0.357 | 0.406 |

**三 seed 汇总（mean ± sample std）**：

| loss | R²_field | R²_casemean | bif | stenosis | high_wss | top10 比 | p99 比 | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE | 0.191±0.018 | 0.176±0.021 | 0.016±0.022 | -0.203±0.046 | -1.644±0.046 | 0.337±0.020 | 0.428±0.008 | 2.847±0.038 |
| target-weight α=2 | **0.225±0.034** | **0.212±0.027** | **0.055±0.048** | **-0.166±0.094** | **-1.531±0.167** | **0.357±0.037** | **0.434±0.045** | **2.788±0.040** |

**结果判读**：

1. **target-weight 方向仍成立，但不是稳定获胜。** 三 seed 均值相对 MSE：`R²_field +0.035`、`R²_casemean +0.036`、MAE `-0.059`，区域指标也平均回升；但 seed 7 的 field 持平、stenosis/high-WSS 反而退化。当前只能下“2/3 seed 有效、均值正收益”的结论，不能把单 seed 最优 0.251 当作稳定水平。
2. **第三轮 clean-data 组合改善了幅值刻度，但没有突破整体 R² 平台。** 第二轮旧口径 tgtw 三 seed `R²_field=0.222±0.022`，第三轮为 `0.225±0.034`，几乎持平；但 top10 预测/真值均值约从 `23.4%` 提高到 `35.7%`，p99 比约从 `35.1%` 提高到 `43.4%`。这说明清理数据/peak stats 是必要修复，主要收益是减少峰值压扁，而非自动提高空间拟合上限。注意两轮还同时改变了 split、curvature transform、epoch/eval 设置，故该跨轮比较不是“只改 stats”的严格因果消融。
3. **高 WSS 仍是明确 No-Go 项。** 6 个 run 的 high-WSS R² 全为负，tgtw 均值仍为 `-1.531`；top10 真值只恢复约 36%，max 比均值仅约 13%。典型热力图显示模型大致知道热点在分叉/髂支附近，但输出过度平滑、峰值幅值系统性偏低。
4. **checkpoint 选择噪声仍大。** tgtw 三个 best epoch 为 39/19/19，MSE 为 149/59/99；240 epoch 对 tgtw 明显冗余。val 只有 8 例，单一 `val_r2_casemean` 会偏向很早的偶然峰值，也无法约束 high-WSS 校准。
5. **病例级失败不是单个坏例造成。** tgtw 三 seed 平均最差病例为 `WANG_DENG_FENG`、`LU_ZHEN_QING`、`LV_FU_LONG`、`QIN_SI_FU`；fast/slow 两组病例均值接近，没有证据把问题归因于某一个 cohort。QA warning 3 例都在训练集，也不能解释 test 的系统性尾部低估。

**汇总产物**：已重跑 `python -m training_wss_min.summarize`，`training_wss_min/runs/_summary/` 现聚合 27 个实验（含第三轮 6 个深度模型与 3 个 clean 模板基线）。

**下一轮优化优先级**（详见已归档的 [第四轮执行总结与归档 v2](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)）：

- **P0｜协议与正确性**：修复 persistent worker 下 epoch 采样状态、train-global 固定 target-weight 阈值、val-only 开发评估、top-k checkpoint 与 early stopping；不采用跨 checkpoint 指标滑窗直接选当前权重。
- **P1｜采样覆盖**：先做 train-only 覆盖审计，再依次比较 fixed FPS 2000、multi-start FPS 2000、fixed FPS 4000；4000 Go 后才开 6000。
- **P2｜尾部目标假设**：先做 3 seed 残差分位与病例留一校准 Gate-0；raw-space Huber 辅助为主 probe，高斯 NLL 仅在 Gate-0 支持时探索，expectile/quantile 不进入默认均值回归矩阵。
- **P3｜内在局部几何**：先 `radius_gradient`，再 radial/surface rotation-invariant 特征；不直接输入未经符号/局部 frame QA 的全局法向。
- **P4｜统计固化**：train+val 61 例做版本化 repeated holdout 与 fold-specific stats；legacy test16 停止逐配置查看，最终锁定后只运行一次，并做病例级 bootstrap。

## 第一轮 sweep ✅完成（2026-07-08，Slurm 5945–5953）
**目的**：三条正交问题——点数-精度曲线、特征消融、采样策略。均标量 WSS + 峰值收缩期 + 400 epoch。

**结果（test，完整点云，按 R²_field 降序）**：

| 配置 | 点数 | 采样 | 特征 | R²_field | R²_casemean | bif | stenosis | high_wss |
|---|---|---|---|---|---|---|---|---|
| feat_xyzgeom | 2000 | fps | xyz+几何 | **0.200** | 0.188 | 0.017 | −0.196 | −1.634 |
| feat_geomonly | 2000 | fps | 纯几何 | 0.191 | 0.189 | 0.014 | −0.263 | −1.712 |
| pc_xyz_w6000 | 6000 | fps | xyz | 0.141 | 0.119 | −0.074 | −0.277 | −1.805 |
| pc_xyz_w3000 | 3000 | fps | xyz | 0.118 | 0.108 | −0.072 | −0.345 | −1.842 |
| samp_geomw | 2000 | 几何加权 | xyz | 0.108 | 0.100 | −0.110 | −0.349 | −1.934 |
| samp_random | 2000 | random | xyz | 0.084 | 0.062 | −0.139 | −0.326 | −1.956 |
| pc_xyz_w1000 | 1000 | fps | xyz | 0.058 | 0.070 | −0.151 | −0.474 | −2.037 |
| pc_xyz_w2000 | 2000 | fps | xyz | 0.053 | 0.047 | −0.160 | −0.482 | −2.062 |
| pc_xyz_w1500 | 1500 | fps | xyz | −0.072 | −0.051 | −0.315 | −0.749 | −2.446 |

**结论**：
1. **几何特征是压倒性杠杆**：xyz+几何(0.200) ≈ 纯几何(0.191) ≫ 纯 xyz@2000(0.053)，约 4×。纯几何≈xyz+几何 → 绝对坐标几乎不贡献，**标量任务里那套 flow-divider 配准/左右轴框架不值分**（模型学局部几何而非绝对位置）。
2. **几何特征更省点**：2000 点几何 > 6000 点纯 xyz；纯 xyz 才靠堆点数往上爬。`w1500` 负值是单种子方差异常。
3. **几何加权采样有用**：0.108 > random 0.084 > fps 0.053（xyz@2000）。
4. **主瓶颈**：所有配置在 stenosis / high_wss 区 R² 全负（最好也 −0.20 / −1.63），val/test gap 大（w1000 val 0.29 vs test 0.06，小样本过拟合）。

---

## 第二轮 sweep ✅完成（2026-07-08 提交，Slurm 5961–5969；2026-07-09 收官汇总）
**目的**：以第一轮最优方向为默认（**xyz+几何 / fps2000**），专打尾部（狭窄/高 WSS）崩溃，并稳方差、缩小 gap。

**新增代码能力（配置驱动，向后兼容）**：
- 训练期**随机 3D 旋转增广**（`DataConfig.rot_aug`）：只扰动 xyz 输入列（几何/标量标签旋转不变、ball-query 图结构不变），逼模型别背绝对朝向。
- **目标幅值加权 loss**（`TrainConfig.loss_weight_target`）：`weight += α·clamp01(y_norm)`，用训练标签给高 WSS 点更大权重（非泄漏），直接补 high_wss 欠拟合。
- Huber loss 选项、几何加权 loss（`loss_geom_weight`）。

**配置矩阵（9）**：`mse`(参考) / `huber` / `geomwloss`(几何加权 loss) / `tgtwloss`(目标加权 loss) / `geomw_tgtw`(双加权) / `geomwsamp_tgtw`(几何加权采样+目标加权) / `rotaug`(旋转增广) / `tgtw_s2025` `tgtw_s7`(目标加权多种子)。

**最终结果（test，完整点云，按 R²_field 降序；best epoch 为 `val_r2_casemean` 选中的 ckpt）**：

| 配置 | R²_field | R²_casemean | bif | stenosis | high_wss | best epoch |
|---|---|---|---|---|---|---|
| r2_xyzgeom_geomw_tgtw | **0.249** | 0.222 | **0.086** | −0.111 | −1.401 | 119 |
| r2_xyzgeom_tgtwloss (s1234) | 0.246 | **0.233** | 0.074 | **−0.068** | **−1.377** | 79 |
| r2_xyzgeom_geomwsamp_tgtw | 0.235 | 0.226 | 0.064 | −0.128 | −1.524 | 59 |
| r2_xyzgeom_geomwloss | 0.229 | 0.196 | 0.049 | −0.154 | −1.398 | 279 |
| r2_xyzgeom_tgtw_s7 | 0.217 | 0.220 | 0.040 | −0.139 | −1.587 | 99 |
| r2_xyzgeom_tgtw_s2025 | 0.203 | 0.165 | 0.048 | −0.209 | −1.460 | 59 |
| r2_xyzgeom_huber | 0.203 | 0.180 | 0.017 | −0.169 | −1.574 | 339 |
| r2_xyzgeom_mse (参考) | 0.189 | 0.156 | −0.005 | −0.191 | −1.634 | 159 |
| r2_xyzgeom_rotaug | 0.186 | 0.173 | −0.015 | −0.094 | −1.577 | 59 |

**高 WSS 分位校准探针（test 全场 pool，ckpt_best 完整推理；真值 top10% 均值 18.51，p99 26.98，max 126.74）**：

| 配置 | top10% pred/true | p99 比 | max 比 |
|---|---|---|---|
| r2_xyzgeom_mse | 3.90/18.51 = **21.1%** | 34.4% | 33.5% |
| r2_xyzgeom_tgtwloss | 4.10/18.51 = 22.1% | 34.3% | 31.1% |
| r2_xyzgeom_tgtw_s2025 / _s7 | 22.0% / 26.0% | 31.2% / 39.9% | 41.9% / 34.4% |
| r2_xyzgeom_geomw_tgtw | 4.86/18.51 = 26.2% | 40.4% | **138%**（max 溢出） |
| r2_xyzgeom_geomwsamp_tgtw | 9.76/18.51 = **52.7%** | 78.5% | **2248%**（max 2849，爆炸） |

**结论**：
1. **目标幅值加权方向确认有效，且是唯一稳定正收益**：tgtw 家族（tgtwloss/geomw_tgtw/geomwsamp_tgtw）R²_field 0.235–0.249 全面领先 mse 参考(0.189)/huber(0.203)/rotaug(0.186)，stenosis 从 −0.19 回拉到 −0.07~−0.13、high_wss 从 −1.63 回拉到 −1.38~−1.52。
2. **但分位校准揭示：R² 回拉主要来自中段，真峰值几乎没恢复**。top10% 高 WSS 校准 mse 21.1% → tgtwloss 仅 22.1%（p99/max 比甚至持平或更低）；`geomwsamp_tgtw` 可推到 52.7% 但 max 溢出 22 倍，不可用。**印证最终诊断的 P1 判断：旧 all-time stats（log std≈5.58）把峰值压扁是硬上限，loss 加权修不了，必须 clean-data 重算 stats。**
3. **seed 方差大到吞掉多数配置间差异**：tgtwloss 三种子 R²_field = 0.246/0.217/0.203（均值 0.222，极差 0.043）。除"tgtw 家族 > 非加权"这一档间隔外，家族内部排序（如 geomw_tgtw 0.249 vs tgtwloss 0.246）不可作数；第三轮必须多 seed。
4. **Huber 无收益**（0.203，在 seed 噪声内）；**rotaug 全场最差**（0.186）且与 canonical flow-divider 坐标框架假设冲突——第三轮 `xyz+geom` 主线不启用（与最终诊断 6.2 一致）。
5. **400 epoch 明显过长**：best epoch 集中在 59–159（仅 huber 339 / geomwloss 279 例外），后期都是过拟合区；val(8例)→test 落差 R²_field 约 0.05–0.11。支持最终诊断 5.1/5.2：缩短训练/early stopping + 复合选模指标。

**产物**：`runs/_summary/{summary.csv,pointcount_curve.png,regional_bar.png}`（已含全部 18 run）；分位探针脚本口径见最终诊断文档 §9 交付物要求（top10%/p95/p99/max 比值）。

---

历史轮次到此结束。当前待办不在历史段落重复维护，以文首最终审查回写和[WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md)为唯一准绳。
