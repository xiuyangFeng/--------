# WSS 最小化·PointNet baseline 实验矩阵与进度跟踪

> 建立日期：2026-07-14
>
> 更新日期：2026-07-16（AG/AAA v4 E2-GLOBAL 完训公平评估）
>
> 用途：跟踪导师提出的 PointNet baseline 容量、采样点数与 WSS 归一化实验；本文是这一小矩阵的状态真源。
>
> 相关入口：[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md) · [WSS 代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) · [`training_wss_min`](../../training_wss_min/README.md)

> **v4 当前状态（2026-07-16）**：活动 AG76、AAA 几何终签63/入训57；AG `61/0/15` 与混合 `118/0/15` 的 E2-GLOBAL/FPS-2000 已全部完训完评（Jobs `9138` / `9140`）。主结果严格使用 `ckpt_best(train_loss)`：AG-v4 比旧 E2 common-test15 锚点回退，判 No-Go；AAA 混入相对 AG-v4 小幅提升整体 R²、排序和热点定位，但高 WSS 幅值压缩更重，仍未追平旧锚点。本轮未切 random-5000，ILO 未处理。

## 0. v4 结果结论（2026-07-16）

三列均是同一 AG common-test15 的物理 WSS。旧锚点是从已保存预测纯后处重汇总，没有重训/重推理；两套 v4 取预注册的 `ckpt_best(train_loss)`。

| 口径 | train | best epoch | field R² | case-mean R² | MAE / RMSE (Pa) | high-WSS R² | top10 幅值比 / IoU | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 v3 E2 common-test15 锚点 | AG61 | 364 | **0.2597** | **0.1928** | **2.754 / 4.902** | — | — / 0.1528 | 0.6756 |
| AG-v4 E2 | AG61 | 364 | 0.1852 | 0.1465 | 2.847 / 5.143 | -1.8279 | 0.3508 / 0.1314 | 0.6788 |
| AG+AAA-v4 E2 | AG61+AAA57 | 395 | 0.2086 | 0.1572 | 2.772 / 5.068 | -1.8473 | 0.3405 / **0.1575** | **0.7041** |

1. **AG-v4 单独重跑 No-Go**：对同一 test15，相对旧锚点 field/case-mean R² 下降 `0.0744/0.0463`，MAE/RMSE 增加 `0.093/0.241 Pa`，15例仅3例 R² 改善。因此不能把 v4 坐标/数据发布解读为 E2 精度提升。
2. **AAA 扩容是小幅、非均匀的增益**：相对 AG-v4，混合 field/case-mean R² 增加 `0.0234/0.0106`，MAE/RMSE 下降 `0.075/0.074 Pa`，15例中8例改善；Spearman 提高 `0.0254`，top10 IoU 提高 `0.0261`，热点质心距离/bbox 由 `0.0895` 降到 `0.0785`。
3. **但幅值与高尾更差**：high-WSS R² `-1.8279→-1.8473`、MAE `10.919→11.195 Pa`，top10/p99 幅值比 `0.3508/0.4873→0.3405/0.4263`，动态范围比 `0.2147→0.1540`。混合模型仍比旧锚点低 `0.0510` field R²，只是 MAE 已接近（高 `0.018 Pa`）。
4. **病例异质性值得优先处理**：混合相对 AG-v4 对 `BAI_WEN_JIE`、`WANG_YONG_FAN`、`LU_ZHEN_QING` 改善最大；对高 WSS 观察病例 `LI_SHU_KUN` 恶化最大（R² 约 `-0.293`），`GUO_XI_JIANG` 约 `-0.149`。下一步应优先检查 AG/AAA 域比例、高尾损失/采样和困难病例，不建议先切 random-5000。
5. `last` 对 AG-v4 略差；混合 last 的 field R² 为 `0.2193`，比 best 高 `0.0106`，但不能用 test 指标改写预注册选模。这只说明 best/last 敏感性低且主结论不变。
6. **归一化有一个需要先隔离的混杂因素**：混合训练按病例是 AG:AAA=`61:57`，但 train-only global stats 按全云点汇总；AAA 密网格贡献了混合统计约 `78.8%` 的点。因此 log-WSS 均值/标准差从 AG 的 `1.122/1.096` 变为混合的 `0.553/1.373`，p50 从 `3.088` 降到 `1.683 Pa`。这与 AG test 高幅值进一步压缩一致，但目前只是机制性推断；下一个最干净的单变量应是“病例平衡/域平衡 target stats”，而不是同时改采样点数。
7. 混合 best 的物理 train `R²_cb` 从 AG-v4 的 `0.6041` 降到 `0.4922`，而test 从 `0.1866` 升到 `0.2054`，train−test gap 由 `0.4174` 缩到 `0.2867`。这更像“AAA 提供了正则化/空间排序信号，但幅值和域匹配仍未解决”；两套的 train loss 因 target stats 不同不能直接比较。

产物完整性：Jobs `9138` / `9140` 均 `COMPLETED (0:0)`；两套 best/last metrics 齐全，best PostView 均 15/15 病例、mapping coverage 100%。结果结构化摘要见 `data_wss_min/pipeline_reports/v4_cutover_20260715_1921/v4_e2_global_result_analysis.json`。

## 1. 当前冻结决策

1. 后续实验只跑 **PointNet + xyzgeom**，不再扩展 MLP / PointNet++ 或 xyz-only 矩阵。
2. 历史正式协议为 AG `61/0/16`；v4 公平协议为 `61/0/15`，只删除 `WANG_DENG_FENG`，不补病例。训练期不读 test，关闭早停，训练400 epoch，按 train loss 保存 best。
3. 容量实验按导师结构对齐：局部编码 `6→256→512`，全局拼接后解码 `1024→512→256→1`，`dropout=0`。
4. 点数实验每个 case 每个 epoch **随机不放回采样 5000 点**；不再跑有放回对照。
5. 归一化实验只研究无量纲的 WSS 空间分布和 high-risk position，不要求恢复测试病例的物理 WSS。
6. Job `8970` 已完训（旧 `53/8/16 + val-only + 160 epoch` 协议的 width=128 探针），**仅作补充证据，不纳入下述正式配对矩阵，也不代替 `E2-GLOBAL`**。
7. `N-CASE` 已冻结为 **`WSS/WSSmax`**；不再把逐病例 log-z 混入本矩阵。
8. `ckpt_best(train_loss)` 为预注册主模型并导出 test16 PostView；`ckpt_last` 只做完整指标审计。连通性指标本轮暂缓。
9. 导师追加的 `E4-DEEP-GLOBAL` 是矩阵结案后的单次深度探针：严格对齐 E2，只改网络深度；Job `8999` 已完训并判为 **No-Go**，不并入已完成的五组正式矩阵，也不恢复 random-5000 或 CASE 路线。
10. v4 发布后的 AG 协议为 `61/0/15`；旧 `61/0/16` 结果保留为历史事实，公平配对列使用 common-test15。

## 2. 要回答的科学问题

### Q1｜模型容量

在同一输入、点数、损失和训练协议下，将 PointNet 对齐到导师的宽通道结构，是否能改善 WSS 非线性空间分布与高 WSS 位置识别。

### Q2｜单例点数与每 epoch 重采样

在不改模型的情况下，将每例训练点从固定 FPS-2000 改为随机不放回 5000 点，并在每个 epoch 重采样，是否能让网络看到更完整的病例内 WSS 空间型态。

### Q3｜跨病例幅值是否遮蔽空间型态

对照全局归一化与逐病例归一化，重点检验：当各 case 的目标数值被压到近似尺度后，网络是否更能预测病例内的 WSS 相对分布、高风险区域与峰值位置，而不再主要受病例间绝对幅值差异影响。

## 3. 归一化两套实验

> 术语说明：z-score 严格由均值和标准差定义，不是由 range 定义。导师口头所说的“全局 range”，在本文按“全局共享统计量”理解。

### N-GLOBAL｜全局 train-only log-z

\[
y_G=\frac{\log(\mathrm{WSS}+\epsilon)-\mu_{\mathrm{train}}}
{\sigma_{\mathrm{train}}}
\]

- `mu_train` / `sigma_train` 由 61 个 train case 的峰值时相壁面点共同计算，所有病例共享。
- 当前统计：`eps=1e-6`、`log mean=1.12116`、`log std=1.09615`。
- 保留病例间幅值差异；模型同时承担“跨病例水平 + 病例内空间型态”。
- `E2-GLOBAL` / `E3-GLOBAL` 就是容量实验（2）和采样实验（3）本身，不重复训第二份同配置。

### N-CASE｜逐病例相对尺度

正式冻结为 case-max 口径：

\[
y_C=\frac{\mathrm{WSS}}
{\max_j\mathrm{WSS}_j}
\]

- 每个 case 的真值最大值都为 1，直接移除病例间绝对幅值差异。
- 测试时模型只输出相对场；真实 `WSSmax` 只用于构造归一化评价真值，不作模型输入，也不用于恢复物理 WSS。
- 该任务回答“哪里相对更高”，不回答“绝对值是多少 Pa”。

#### 已排除的本轮替代口径

如果“第二个实验使用逐病例 z-score”指的是每例独立均值/标准差，则另一个公式是：

\[
y_{CZ}=\frac{\log(\mathrm{WSS}+\epsilon)-\mu_i}{\sigma_i}
\]

| 口径 | 对齐效果 | high-risk 解释 | 主要风险 |
| --- | --- | --- | --- |
| `WSS/WSSmax` | 每例 max=1，保留非负相对幅值 | 适合直观画 0–1 分布和 top-q% | max 对单点尖峰/异常值敏感 |
| 逐例 log-z | 每例 mean≈0、std≈1，分布对齐更强 | 须用病例内分位数定义高风险 | 输出有正负，不再是直观的 WSS 比例 |

**冻结结论**：本矩阵只实现 `WSS/WSSmax`。逐例 log-z 若未来获批，必须使用新实验 ID 和独立汇总，不复用 `E2-CASE/E3-CASE`。

## 4. 正式实验矩阵

| ID | 模型/采样变量 | 目标归一化 | 与父实验的唯一差异 | 状态 |
| --- | --- | --- | --- | --- |
| `E0-GLOBAL` | 原 PointNet，FPS-2000 | 全局 log-z | 无 val/无早停共同对照 | ✅ 完训+完评（`8968_1` / eval `8975`）；见 §4.1 |
| `E2-GLOBAL` | 导师宽 PointNet，FPS-2000 | 全局 log-z | 相对 E0 只改容量 | ✅ 完训+完评，Job `8976` |
| `E3-GLOBAL` | 原 PointNet，random-5000/不放回/每 epoch 重采 | 全局 log-z | 相对 E0 只改采样 | ✅ 完训+完评，Job `8977` |
| `E2-CASE` | 与 E2-GLOBAL 相同 | `WSS/WSSmax` | 相对 E2-GLOBAL 只改目标归一化 | ✅ 完训+完评，Job `8979` |
| `E3-CASE` | 与 E3-GLOBAL 相同 | `WSS/WSSmax` | 相对 E3-GLOBAL 只改目标归一化 | ✅ 完训+完评，Job `8980` |
| `E23-GLOBAL` | 导师宽 PointNet，random-5000/不放回/每 epoch 重采 | 全局 log-z | 同时合并容量（2）与采样（3），用于检查交互效应 | ✅ 完训+完评，Job `8978` |

主矩阵对应“2+4”和“3+4”。全局列是原始容量/采样实验，逐病例列是各自的配对归一化实验。`E23-GLOBAL` 是后补的交互对照：只在 E2/E3 单变量结果之外解释“加宽与 5000 点是否协同”，不能替代 E2 或 E3，也不用于单独归因容量/采样贡献。

### 4.1 `E0-GLOBAL` 结果摘要（2026-07-14）

对照臂为 Job `8968` 的 **PointNet + xyz+geom**（`pointnet_trainloss_e400/outputs/pointnet_xyzgeom`）；纯 xyz 仅作输入消融，不进矩阵主线。

| 分区 | `R²_field_cb` | `R²_field_raw` | case mean/med/P10 | 负例 | RMSE/MAE (Pa) | high-WSS R² | top10 比/IoU |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| train61 | 0.5303 | 0.5277 | 0.454 / 0.462 / 0.326 | 0/61 | 4.51 / 2.19 | −0.20 | 0.581 / 0.415 |
| **test16** | **0.1414** | 0.1433 | 0.106 / 0.074 / −0.062 | **5/16** | 5.42 / 2.95 | **−1.76** | 0.328 / 0.174 |

- **判读**：train-fit 有容量信号，但 test16 泛化弱、热点失败；本协议下 `E0` 作为「可部署精度」**No-Go**，仍保留为后续 `E2/E3` 的共同对照锚点（同 split / 同选模）。
- 同协议纯 xyz test16 `R²_cb=0.0825`（更差）→ 确认只保留 xyz+geom。
- 证据：`eval/metrics.json`（`8975`）；详细分析见推进记录文首 `2026-07-14｜8968 train+test16 与 8970 …`。
- **注意**：不把本 test16 数与 2×3 的 val8 `0.3015` 混表。

### 补充探针（不进正式配对矩阵）

| Job | 协议 | 当前作用 |
| --- | --- | --- |
| `8700[0-5]` | `53/8/16`、val-only、160 epoch、MLP/PointNet/PointNet++ × xyz/xyzgeom | 已完成的最小 2×3 架构下限；PointNet+xyzgeom val `R²_cb=0.3015` 为当轮最佳单格 |
| `8968[0-1]` | `61/0/16`、无早停、400 epoch、train-loss 选模 | 已完成；**xyzgeom = `E0-GLOBAL`**；xyz 仅消融 |
| `8970` | `53/8/16`、val-only、160 epoch、`6→128→256→512` | ✅ 完训；val `R²_cb=0.3071` vs 锚点 0.3015（Δ≈+0.006）→ **旧协议加宽 No-Go**；不代替 `E2-GLOBAL` |

### 4.2 正式矩阵结果（2026-07-15）

五个 Job 均为 `COMPLETED (0:0)`，训练、best/last 全点评估和 best 的 test16 PostView 已全部结束。以下均使用预注册的 `ckpt_best(train_loss)` 作主结果；`last` 只用于敏感性审计。

#### GLOBAL：物理 WSS 对照

| Run | best epoch | test `R²_field_cb` | MAE / RMSE (Pa) | high-WSS R² | top10 幅值比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `E0-GLOBAL` | 383 | 0.1414 | 2.913 / 5.345 | −1.76 | 0.328 |
| **`E2-GLOBAL`** | **364** | **0.2140** | **2.801 / 5.114** | **−1.486** | **0.378** |
| `E3-GLOBAL` | 354 | 0.1637 | 2.876 / 5.275 | −1.665 | 0.343 |
| `E23-GLOBAL` | 354 | 0.1988 | 2.806 / 5.163 | −1.525 | 0.372 |

- `E2-GLOBAL` 相对 E0 的主 R² 增加 **0.0725**，是本轮最强的物理 WSS 配置；说明导师宽通道确有容量增益。
- `E3-GLOBAL` 相对 E0 只增加 **0.0223**；单独把 2000 点改为每 epoch random-5000 不是主要突破口。
- `E23-GLOBAL` 没有超过 E2（`−0.0152`），因此没有“加宽 × 5000 点”的明确协同收益；它比 E3 高 `0.0351`，进一步说明本轮主要增量来自容量。
- 即使最优 E2 的 high-WSS R² 仍为 **−1.486**、top10 幅值比仅 **0.378**，所以它是“相对改进”，仍不是可部署精度的 Go。

#### 归一化空间分布与 high-risk position

GLOBAL 行在 train61 共享 log-z 空间评价，CASE 行在逐病例 `WSS/WSSmax` 空间评价。两种目标变换下的 MAE/RMSE 数值尺度不同，不直接横比；R²、排序、top10 和归一化位置指标用于回答本轮空间分布问题。

| Run | test `R²_cb` | case med / P10 | 负例 | Spearman | top10 IoU | 峰值距 / bbox | 质心距 / bbox | p99 比 | 动态范围比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `E0-GLOBAL` | 0.3964 | 0.319 / 0.118 | 0/16 | 0.635 | 0.128 | 0.175 | 0.090 | 0.659 | 0.569 |
| **`E2-GLOBAL`** | **0.4606** | **0.431 / 0.202** | 1/16 | 0.661 | 0.146 | 0.172 | 0.085 | 0.684 | 0.577 |
| `E3-GLOBAL` | 0.4218 | 0.372 / 0.234 | 0/16 | 0.645 | 0.124 | 0.178 | **0.073** | 0.642 | 0.581 |
| `E23-GLOBAL` | 0.4598 | 0.394 / **0.282** | 0/16 | **0.674** | 0.148 | **0.158** | 0.090 | **0.693** | **0.605** |
| `E2-CASE` | 0.1724 | 0.122 / −0.328 | 4/16 | 0.574 | 0.140 | 0.172 | 0.094 | 0.486 | 0.456 |
| `E3-CASE` | 0.1551 | 0.077 / −0.406 | 5/16 | 0.559 | **0.159** | 0.162 | 0.078 | 0.547 | 0.454 |

配对判读：

1. **容量是本轮主增量**：E2 相对 E0 的 normalized `R²_cb` 增加 `0.0642`，Spearman 增加 `0.0261`，top10 IoU 增加 `0.0177`。E23 相对 E3 也全面更好；但 E23 相对 E2 基本持平，5000 点没有稳定附加收益。
2. **`WSS/WSSmax` 主结论为 No-Go**：E2-CASE 相对 E2-GLOBAL 的 `R²_cb` 从 `0.4606` 降至 `0.1724`，Spearman 从 `0.661` 降至 `0.574`，top10 IoU 也没有改善。E3-CASE 的 IoU 从 `0.124` 升至 `0.159`，但 `R²_cb` 从 `0.4218` 降至 `0.1551`、负例从 0 增至 5，且 p99/动态范围明显压缩；局部热点重叠的单项改善不足以支持替换 GLOBAL。
3. 抽查 `WANG_DENG_FENG`（E2）和 `GONG_HUI_XIA`（E3）的 top10 overlay 与指标一致：CASE 偶尔收窄或移动部分假阳性，但仍存在大块漏检和假阳性，没有恢复真实高风险带。
4. 这回答了本轮核心问题：当病例目标被压到相近尺度后，网络**没有更可靠地学会完整 WSS 分布和 high-risk position**；相反，case-max 的尖峰尺度使大部分分布被压缩，尤其不利于 p99 与动态范围。

#### best / last 敏感性审计

| Run | best normalized `R²_cb` | last normalized `R²_cb` | last − best |
| --- | ---: | ---: | ---: |
| `E2-GLOBAL` | 0.4606 | 0.4455 | −0.0151 |
| `E3-GLOBAL` | 0.4218 | 0.4188 | −0.0030 |
| `E23-GLOBAL` | 0.4598 | 0.4591 | −0.0006 |
| `E2-CASE` | 0.1724 | 0.1748 | +0.0025 |
| `E3-CASE` | 0.1551 | 0.1556 | +0.0004 |

best/last 不改变容量、采样或归一化结论。虽然个别 last 的某项 test 指标略高，仍严格保留 `ckpt_best(train_loss)` 为主模型，不根据 test16 反选 checkpoint。

#### 产物完整性

- 5/5 run 均具有 `ckpt_best.pt`、`ckpt_last.pt`、`eval/ckpt_best/` 和 `eval/ckpt_last/`；日志未发现 traceback、OOM 或失败退出。
- 5 × 16 = **80/80** 个 best-test PostView case 包完整；每例具有对齐 STL、同点 CSV、点云/表面 VTP、共享色标三联图、top10 overlay、指标 JSON 与 normalization manifest。
- 80/80 manifest 的必需标量字段齐全，引用文件全部存在；表面 Gaussian 回插 mapping coverage 的最小值和均值均为 **100%**。
- 结果根目录：`training_wss_min/runs/pointnet_distribution_matrix/outputs/`。

### 4.3 导师补充指标：`CFD/CFDmax` 对 `Pred/Predmax`（2026-07-15）

导师补充要求比较每个病例内两个场各自除以自身最大值后的空间型态：

\[
y_{\mathrm{CFD,self}}=\frac{WSS_{\mathrm{CFD}}}{\max(WSS_{\mathrm{CFD}})},
\qquad
y_{\mathrm{Pred,self}}=\frac{WSS_{\mathrm{Pred}}}{\max(WSS_{\mathrm{Pred}})}
\]

这与原先已经保存的 `WSSpred/WSScfd,max` 不是同一个量：

- `WSSpred/WSScfd,max` 使用 CFD 真值尺度，保留预测幅值是否正确的信息；
- `WSSpred/WSSpred,max` 使用预测自身尺度，主动移除绝对幅值，只检查相对空间分布；
- 两个分母均为该病例峰值时相的**完整壁面点场最大值**，先算最大值再映射到 STL；Gaussian 表面回插只用于展示，不参与正式同点指标；
- 正比例缩放不改变点排序，因此 top10 high-risk 位置集合不会因 self-max 操作本身改变；新指标是对既有空间排序/定位结果的补充，不替代 top10 IoU 与距离指标。

既有 PostView 已保存完整壁面 true/pred，无需重新训练或重新跑模型前向。五组共 80 个 test case 已用纯后处理回填：

| Run | pooled R² | case-balanced R² | case R² med / P10 | R²<0 病例 | MAE / RMSE | 负预测点占比（病例均值） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`E2-GLOBAL`** | **−4.802** | **−4.857** | −7.083 / −14.812 | 16/16 | **0.185 / 0.247** | 0% |
| `E3-GLOBAL` | −6.557 | −6.605 | −6.915 / −17.654 | 16/16 | 0.215 / 0.281 | 0% |
| `E23-GLOBAL` | −5.454 | −5.422 | **−6.283** / −14.369 | 16/16 | 0.190 / 0.259 | 0% |
| `E2-CASE` | −8.814 | −8.922 | −12.088 / −22.508 | 16/16 | 0.250 / 0.321 | 7.34% |
| `E3-CASE` | −8.868 | −9.079 | −11.177 / −35.641 | 16/16 | 0.252 / 0.324 | 6.03% |

判读：

1. self-max 后五组的 16/16 逐病例 R² 均为负，说明主要问题不只是整体幅值偏小；分别把两张场都缩放到峰值 1 后，预测的相对空间型态仍明显错误。负 R² 表示误差大于“预测为 CFD 归一化场均值”的基线，**不等于所有归一化点值都为负**。
2. `E2-GLOBAL` 的 pooled/case-balanced R² 与 MAE/RMSE 相对最好，仍支持“导师宽网是本轮较优锚点”；但它的 self-max R² 远低于 0，因此不能据此宣称学会高 WSS 空间分布。
3. CASE 网络使用线性输出层，没有非负约束。`E2-CASE` 为 15/16 个病例出现至少一个负预测点，病例平均占 7.34%、最差 40.48%；`E3-CASE` 为 14/16，平均 6.03%、最差 20.83%。除以正的 `Predmax` 不会消除负数。正式指标保留未裁剪结果，并报告负值比例；如做裁剪图，必须标注为可视化/敏感性分析，不能覆盖正式结果。
4. GLOBAL 的物理 WSS 由 log 目标逆变换得到非负场，因此本批物理 self-max 字段无负点；这与 CASE 的线性 0–1 目标输出机制不同。

新增字段为 `wss_cfd_over_cfd_max`、`wss_pred_over_pred_max`、`err_wss_selfmax`、`abs_err_wss_selfmax`，已写入同点 CSV、点云 VTP 和表面 VTP；每例新增 `plots/fig_wss_selfmax_triptych.png`。五组汇总为 `outputs/selfmax_test_summary.json`，逐病例表为各 run 的 `postview/ckpt_best/test/selfmax_per_case_metrics.csv`。

### 4.4 导师追加深度探针 `E4-DEEP-GLOBAL`（2026-07-15 完训完评｜No-Go）

该探针不改写 §4.2 已结案的五组正式矩阵，只回答“在 E2 宽度锚点上增加编码/解码深度是否继续有收益”。

| Run | 局部编码 | global 拼接后解码 | 参数量 | 其余协议 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `E2-GLOBAL` | `6→256→512` | `1024→512→256→1` | 792,833 | train61 / FPS-2000 / global log-z / 400ep / train-loss | ✅ 已完成，配对锚点 |
| `E4-DEEP-GLOBAL` | `6→64→128→256→512` | `1024→512→256→128→64→1` | 874,561 | 与 E2 完全相同 | ✅ Job `8999` `COMPLETED (0:0)`｜**No-Go** |

E4 相对 E2 增加 `81,728` 个参数（10.3%），因此它主要是深度探针而非再次大幅加宽。Job `8999` 用时 `01:37:11`；400 epoch 齐全，best=第 379 epoch，train loss `0.121590`（E2 为第 364 epoch / `0.172548`，下降 29.5%）。

| best 全点指标 | `E2-GLOBAL` | `E4-DEEP-GLOBAL` | E4−E2 |
| --- | ---: | ---: | ---: |
| train 物理 `R²_field_cb` | 0.6124 | **0.6966** | +0.0841 |
| test 物理 `R²_field_cb` | **0.2140** | 0.1629 | **−0.0511** |
| 物理 train−test gap | 0.3985 | **0.5337** | +0.1352（变差） |
| train 归一化 `R²_field_cb` | 0.7717 | **0.8375** | +0.0658 |
| test 归一化 `R²_field_cb` | **0.4606** | 0.4476 | −0.0130 |
| 归一化 train−test gap | 0.3111 | **0.3899** | +0.0788（变差） |
| test 物理 MAE / RMSE（Pa，病例等权） | **2.8005 / 5.1141** | 2.8510 / 5.2778 | +0.0505 / +0.1637 |
| test 归一化 MAE / RMSE | **0.5732 / 0.7354** | 0.5799 / 0.7442 | +0.0067 / +0.0088 |
| test 物理 high-WSS R² | **−1.4861** | −1.6798 | −0.1937 |
| test 物理 top10 幅值比 | **0.3781** | 0.3328 | −0.0452 |
| test Spearman / top10 IoU | **0.6613** / 0.1457 | 0.6570 / **0.1531** | −0.0043 / +0.0073 |
| test 病例 R² mean / median / P10 | **0.1717 / 0.2010 / −0.0075** | 0.1390 / 0.0962 / −0.0288 | 全部变差 |
| test R²<0 病例 | **2/16** | 4/16 | +2 |
| test 双 self-max `R²_cb` / 负例 | −4.857 / 16 | **−4.208** / 16 | +0.649，仍 No-Go |

配对判读：

1. 更深网络确实提高了 train-fit，但物理/归一化 test R² 均下降，两种空间的 train−test gap 均扩大；这是更强的过拟合证据，不是容量突破。
2. 16 个 test 病例中，E4 的物理 R² 仅 6 例改善、10 例退化，配对中位变化 −0.0295；不是由少数离群例拉低总表。
3. top10 IoU 和双 self-max R² 有小幅改善，说明更深网络对部分相对型态/定位有弱信号；但 self-max 仍 16/16 负 R²，且高 WSS R²、幅值恢复、MAE/RMSE 和整体排序均变差，不足以改变 No-Go。
4. `ckpt_last` 的 test 物理/归一化 R² 为 `0.1618/0.4451`，top10 IoU `0.1540`，与 best 一致；结论不是 train-loss checkpoint 巧合。保留 E2 为 PointNet 锚点，不启动后续纯深度扫描。

完整产物审计：`ckpt_best/last`、best/last train61+test16 全点评估均齐全；best PostView 为 16/16 病例，manifest 引用文件无缺失，STL mapping coverage 全为 100%，self-max/top10 图均为 16/16。日志无 Traceback/OOM/NaN。上述 test16 只是与 E2 同协议的导师驱动配对证据，不是无偏最终测试。

进入 PointNet++ fine-tune 前的三层 SA foundation 已独立完成，但未启动训练。冻结结构为输入 FPS-2000，中心 `500→125→32`，radius `0.05→0.10→0.20`、nsample=16；代表病例 `fast/RAN_QING_BO` 的 source coverage 为 `99.90%→100%→100%`。总览 PNG、代表 group 图、逐层 VTP/CSV 和可复核 trace hash 位于 `例子/06_PointNet++_SA三层采样与分组/`。

## 5. 评价口径：优先检查空间分布和高风险位置

### 5.1 归一化空间分布

每个 test case 独立计算，再做病例等权汇总：

- 归一化空间 `R²`：中位数、P10、负 R² 病例数；
- 归一化 MAE / RMSE；
- Spearman 相关：检查病例内高低排序是否对齐；
- true/pred 的 p50/p90/p95/p99 和动态范围比；
- 绘制每例归一化目标的分布摘要，确认“各 case 数值更相近”确实发生，而不是只改了名称。

### 5.2 high-risk position

由于逐病例归一化后不再有可比的绝对 Pa 阈值，high-risk 主定义使用每例真值 **top 10%** 点，并固定报告：

- top10 IoU；
- top10 precision / recall；
- 峰值点欧氏距离，同时除以该病例坐标 bbox 对角线作尺度归一化；
- high-risk 区域质心距离；
- high-risk 区域连通性/主区域命中本轮暂缓，不引入未冻结邻接定义。

首要判断不是“pred max 是否等于 1”，而是高 WSS 区的位置、范围和排序是否与真值一致。

### 5.3 实验配对判读

- `E2-CASE` 只与 `E2-GLOBAL` 比，`E3-CASE` 只与 `E3-GLOBAL` 比。
- `E23-GLOBAL` 分别与 E2-GLOBAL（隔离 5000 点在宽网下的增量）及 E3-GLOBAL（隔离宽度在 5000 点下的增量）比较，只报告交互证据。
- 主结论回答“归一化是否改善病例内 pattern / hotspot”，不使用物理 Pa 指标判定逐病例归一化组。
- 在公式冻结前不预设数值 Go 线；至少要求归一化 R²/Spearman 和 top10 IoU/峰值距离中同时有“分布 + 定位”两类证据，不以单一 pooled R² 宣布成功。
- 因本协议取消 val，且 test16 会被用于多个连续实验的比较，本矩阵属于导师驱动的探索性实验；不将 test 结果宣称为未经反复使用的无偏最终泛化估计。
- `E0` 已显示 train–test 大 gap；后续 `E2/E3` 仍报 train-fit 与 test16，但**不以压低 train loss  alone 作为成功**。

## 6. 每个 test case 必须保存的可视化与数据

每个正式 run 的 test 评估至少保存：

1. 归一化 true / pred / absolute error 三联图，true 和 pred 强制共用同一色标范围。
2. true-top10 / pred-top10 / 交集与漏检的 high-risk 位置覆盖图。
3. 逐点结果文件，至少包含 `xyz`、`true_norm`、`pred_norm`、`abs_err_norm`、`true_highrisk`、`pred_highrisk`。
4. 逐例指标 JSON/CSV，并记录 `normalization_mode`、归一化公式和当例使用的统计量。
5. 同时将归一化字段写入 VTP 供 ParaView 表面查看；定量指标仍只在同点点云上计算。
6. 导师补充 self-max 图：`CFD/CFDmax`、`Pred/Predmax` 和二者绝对误差共用可比较色标；CSV/VTP 保留未裁剪负预测值，并在指标 JSON 中报告负点比例。

> `E0` 只有标准 `evaluate` 产物（`metrics.json` / `per_case_metrics.csv` / test heatmaps）；§6 完整清单已在 `E2/E3/E23` 五个正式 run 中全部补齐，完整性见 §4.2。

## 7. 执行顺序

1. ~~旧协议 Job `8970`~~ → ✅ 已完成（补充 No-Go，不代替 `E2-GLOBAL`）。
2. ~~对齐导师精确通道表并运行 `E2-GLOBAL`~~ → ✅ 已完成。
3. ~~运行 `E3-GLOBAL`：5000 点、random、不放回、每 epoch 重采样~~ → ✅ 已完成。
4. ~~运行 `E2-CASE` / `E3-CASE` 的 `WSS/WSSmax` 配对~~ → ✅ 已完成。
5. ~~运行 `E23-GLOBAL` 检查容量×点数交互~~ → ✅ 已完成。
6. ~~生成全部 test16 归一化空间图、high-risk 位置图并完成配对分析~~ → ✅ 已完成；结论见 §4.2。

## 8. 进度表

| 日期 | 事项 | 状态 | 证据/下一步 |
| --- | --- | --- | --- |
| 2026-07-14 | 最小 2×3 baseline | ✅ 完成 | Job `8700[0-5]`；PointNet+xyzgeom val `R²_cb=0.3015` |
| 2026-07-14 | `E0-GLOBAL`（无 val/400ep PointNet+xyzgeom） | ✅ 完训+完评 | 训 `8968_1`；eval `8975`；test16 `R²_cb=0.1414`，train 0.5303；见 §4.1 |
| 2026-07-14 | 同协议 xyz 消融 | ✅ 完评（不进主线） | `8968_0` / `8974`；test16 `R²_cb=0.0825` |
| 2026-07-14 | 旧协议 width=128 探针 | ✅ 完成｜补充 No-Go | Job `8970`；val `R²_cb=0.3071` vs 0.3015 |
| 2026-07-15 | 正式 `E2-GLOBAL` | ✅ 完训+完评 | Job `8976`；test 物理 `R²_cb=0.2140`，本轮最强，但 high-WSS 仍 No-Go |
| 2026-07-15 | 正式 `E3-GLOBAL` | ✅ 完训+完评 | Job `8977`；test 物理 `R²_cb=0.1637`，random-5000 单独增益弱 |
| 2026-07-15 | `E2-CASE` / `E3-CASE` | ✅ 完训+完评｜No-Go | Jobs `8979` / `8980`；整体分布 R²/Spearman 明显下降，见 §4.2 |
| 2026-07-15 | `E23-GLOBAL` 宽网+random-5000 交互对照 | ✅ 完训+完评 | Job `8978`；未超过 E2，未见明确协同效应 |
| 2026-07-15 | 五组 best/last + test16 PostView 审计 | ✅ 完成 | 5/5 run、80/80 case 包完整，mapping coverage 100% |
| 2026-07-14 | 完整产物流水线 | ✅ 实现并通过单测/smoke | 34 项单测通过；1 epoch 端到端 smoke 完成 train → best/last eval → test STL/VTP（100% mapping）；临时产物已清理，未提交正式 GPU 作业 |
| 2026-07-15 | `E4-DEEP-GLOBAL` 深度探针 | ✅ 完训+完评｜No-Go | Job `8999` `COMPLETED (0:0)`；test 物理 `R²_cb=0.1629` < E2 `0.2140`，gap 扩大；见 §4.4 |
| 2026-07-15 | PointNet++ 三层 SA foundation | ✅ 可视化完成｜未训练 | `500/125/32` 中心；VTP/CSV/PNG/manifest 完整；40 项训练模块单测通过 |
| 2026-07-15 | §9 第一性原理重排 + 对接第五轮天花板 + 对抗性审查 | ✅ 文档更新｜No-Run | 对接 `runs/_round5` 信息天花板；核实 E2 test16 0.214<PointNeXt 0.31、AAA65+ILO106 例壁面 WSS 有效（p99≈19–23 Pa）、无多队列 split；重排 P0/P0b/P1/P2/P3，新增 §9.4 对抗性审查；待导师批 |

## 9. 下一阶段优化方向（第一性原理重排 · 待讨论 · 不自动开跑）

> 本节 2026-07-15 按第一性原理重写：先对接第五轮已结案的信息天花板结论（§9.0），再据此把「提高预测精度」的杠杆按 **物理可解释性 × 可部署性 × 成本** 重排（§9.1–§9.3），最后对本重排本身做对抗性审查（§9.4）。所有条目仍为 **No-Run**，等导师确认 §9.2 的 P0 开发协议与 P1/P2 范围后再冻结最小矩阵。

### 9.0 第一性原理定位与第五轮对接（关键）

WSS 的物理定义是 \(\tau_w=\mu\,(\partial u/\partial n)|_{\text{wall}}\)：由**几何** + **入口流量波形** + **出口流量分配**三者共同决定。第五轮（PointNeXt、AG 单队列、dev1）已对这三项做过量化结案（`runs/_round5/final_report/round5_final_report.md`），结论必须先接住，否则本矩阵会重复其已排除的路径：

1. **入口无信息**：AG 86 例入口为共享人群模板（Fourier 波形 + Carreau-Yasuda 黏度全同，实测流量 CoV≈1.85e-4），逐例唯一差异是入口面积（几何）。
2. **出口是唯一隐藏杠杆，且不可部署**：4 髂动脉出口各自 RCR 三元 Windkessel，逐例几乎全不同；单支峰值流量占比在 4%–46% 间摆动（~39pp），这是跨病例 WSS 残差的主控变量，但属 `oracle_non_deployable`，不在壁面点云里。
3. **标签可信**：CFD 复现 floor ~2%，`R²_cap≈0.92–0.96`，因此天花板是**信息/表示**上限，不是标签噪声。
4. **几何-only 天花板**：dev1 达 `R²_field≈0.31 / casemean≈0.21`，且拟合足（A0D 四例 0.98–0.99）、预算非限、AG 池内 40 例后出现暂时平台、局部表示（邻域/半径归一化/global-context）无跨病例增量。

**本矩阵与第五轮的数值对接**：本矩阵最优 `E2-GLOBAL` 物理 test16 `R²_cb=0.214`，**低于**第五轮 PointNeXt 的 dev1 `≈0.31`。二者 split、协议、主干均不同，因此当前 0.214 与 0.31 之间同时含两条缝：

- **主干/容量缝（约 +0.09，可能可回收）**：本矩阵用的是导师指定的干净 PointNet，比第五轮的 PointNeXt（残差 + ball-query）弱；这段差距**未必是信息天花板**，可能只是主干不足。
- **信息天花板缝（约 0.31 处）**：越过主干缝后，几何-only 仍受第 2 条出口流量分配的硬约束。

> 结论：**「加宽 PointNet / 加点数 / 换归一化」这类在信息天花板以下的调参已接近收口**（本矩阵 E2/E3/E23/CASE 与第五轮 B-REP 双重证据）。要真正提高精度，杠杆必须落在 **(a) 把主干补到第五轮已验证的水平以回收容量缝、(b) 引入直指出口流量分配的可部署代理输入、(c) 用真实新队列扩大数据与多样性以检验平台是否 AG 同质性造成的假象、(d) 把目标改写成几何真正能学的「型态 + 热点」而非绝对幅值**——而不是继续堆 PointNet 宽度。

### 9.1 证据约束（更新）

- 天花板以下的调参已收口：宽通道相对 E0 一次性 +0.0725 后，random-5000 单独增益弱、E23 未超 E2、`WSS/WSSmax` 与双 self-max 均 No-Go；这些与第五轮「局部表示无跨病例增量」互证，下一轮**不再盲目扩宽度/点数/换归一化**。
- **当前「型态学得好」的表象被高估**：log-z normalized `R²_cb=0.4606`、Spearman 0.661 看似不差，但同一批预测在 self-max（`Pred/Predmax` vs `CFD/CFDmax`）下 16/16 例 R² 全负（pooled −4.8）。差异来源是 log 压缩 + 保留了病例间「整体水平」这一相对容易、且部分由血管尺寸几何决定的方差；去掉水平后暴露出**病例内精细型态与峰值定位仍弱**。因此下一轮**主指标改用 self-max（p99 稳健尺度）+ Spearman + top-k IoU**，不再以 log-z normalized R² 领头。
- **主要矛盾定位**：从「能否拟合训练集」转为「主干是否补到位 + 输入信息是否够 + 目标是否可学」；而输入信息的物理主控变量（出口流量分配）已知不可部署，只能用几何代理逼近。
- **test16 已被连续复用**（E0/E2/E3/E23/E2-CASE/E3-CASE/E4），它已不是未使用的最终测试集，其 R² 对「选模」而言是乐观偏置。**当前 split 上没有一个干净的无偏泛化估计**，这是 §9.2-P0 必须先补的方法学前提。

### 9.2 重排后的优先级（第一性原理）

排序依据：先补方法学前提与最便宜的判别性实验，再上两条真正的精度杠杆（信息 + 数据规模），最后才是目标改写；纯容量/采样/归一化扫描全部降级。

| 优先级 | 方向 | 为什么现在做（第一性原理） | 最小可证伪动作 | 进入/成功判据 |
| --- | --- | --- | --- | --- |
| **P0（前提）** | 恢复无泄漏病例级开发协议 | test16 已被反复使用，任何「涨点」都含选模偏置；没有干净 dev 就无法可信判断任何杠杆 | train pool 内做 grouped repeated holdout 或 grouped K-fold/OOF；**按病人分组**（ILO before/after 同病人、重复几何 `HOU_SHEN_QIAN=KANG_XI_MING` 必须整组同 fold）；预注册 split/统计量/选模规则；test16 只在方案锁定后审计**一次** | 有 OOF 曲线可选模；不再用 test16 逐轮调参 |
| **P0b（便宜判别）** | 把第五轮 PointNeXt 主干搬到当前 split | 直接判定 0.214→0.31 的「容量缝」是否可回收；`pointnext.py` 已存在，一次作业即可再锚定，避免继续在弱主干上做结论 | 用 `pointnext.py` 在 `split_v1_traintest` 上按 E2 协议跑一臂；与 E2-GLOBAL 严格配对 | 若逼近 ~0.31 → 容量缝真实、优先补主干；若仍 ~0.21 → 天花板随 split 前移，转向 P1/P2 |
| **P1（最高真实杠杆·导师已点名·第五轮未测）** | 真实新队列扩数据 + 病例数 learning curve | 第五轮平台只在**同质 AG 池**内出现（40 例），从未测跨解剖扩容；导师已明确要求先用真实 AAA/ILO 做 learning curve 判定「数据量是否瓶颈」再谈合成数据。现有 AAA 65 + ILO 106 例壁面 WSS 有效（峰值 p99≈19–23 Pa，与 AG≈20 Pa 同量级，可比） | 前置门：复用 `bc_audit` 脚本刻画 AAA/ILO 入口/出口 BC 是否与 AG 共享模板；再分层（cohort 作特征、按病人分组、控制点预算）跑 AG-only → AG+AAA/ILO 的病例数曲线，逐队列 + 混合分别评 | 平台随数据/多样性上移，且 **AG-test16 不退化** → 数据是瓶颈；否则跨解剖再证几何-only 天花板（本身可发表） |
| **P2（可部署·直指隐藏变量）** | 出口流量分配的物理代理**空间特征** | 出口 RCR 是已证的隐藏主控变量且不可部署；唯一可部署代理是出口截面积（几何免费）。第五轮 B1 只用了**全局标量**出口面积（≈+0.03，边际），未试把 Murray 定律（流量∝r³）预测的**逐分支流量占比映射到每个壁面点** | 在 E2/PointNeXt 锚点上，仅新增：分支归属 + 上下游面积比 + 距分叉距离构成的逐点「流量占比代理场」；保持采样/target/loss 不变 | 必须**专门降低远端(髂)分支误差**；若无效则确认几何无法代理流量分配（亦为有价值结论）。Murray 在瘤体/病变段失效，预注册 null |
| **P3（目标改写·诚实指标）** | 把「尺度」与「型态」拆开 + loss 直接约束热点 | 幅值受信息天花板限，几何真正能学的是病例内相对型态与热点位置（临床初筛也主要用这个）；应把可学部分单独优化并如实汇报 | 预注册 `global log-z` 锚点，对照双头 `shape + case-level scale`（scale 必须由网络/可部署几何预测，不借测试真值）；在锚点上加**一个**可微 rank/quantile/hotspot 辅助项，保留全场 MSE；不重复既有仅调 raw-Huber/target-weight 的路线 | 同时改善 self-max/Spearman 与 top-k IoU/峰值·质心距离；不能只改 MAE、不能退化全场 R² |
| **P4（降级·已收口）** | 纯容量/深度/点数/归一化扫描 | 本矩阵 + 第五轮双重证据显示已近天花板；E4-deep(8999) 已证实 train-fit 增强但 test 退化；`WSS/WSSmax` 已 No-Go | 不主动扩；仅在 P0b 判定容量缝真实时，才把「补主干」并入 P0b，不再单独堆 PointNet 宽度/深度 | 仅作为 P0b 的附带结论存在 |
| **P5（依赖 P1 结果·非本轮）** | 合成几何补数据（GAN/扩散） | 导师提议，但前置是「真实数据 learning curve 证明数据量确是瓶颈」+「合成几何的 WSS 标签来源与配准口径先解决」 | 仅当 P1 判定数据量为瓶颈后再评估 | P1 未证瓶颈前不启动 |

### 9.3 输出约束与暂不优先项

1. 逐病例 `[0,1]` 目标若继续，`Sigmoid`/`Softplus` 只作**输出非负性敏感性实验**并同时报原始/裁剪指标；能消负点但不解型态错误，不列 P0/P1。
2. high-risk 主定义仍固定为每例 CFD 真值 top10%（top5% 次指标）；正比例 self-max 不改排序，**不得**把「归一化后 top10 不变」当作新增定位能力。
3. 下一轮主判据必须同时含两类证据——分布（self-max/Spearman/动态范围）与定位（top-k IoU、峰值/质心归一化距离）；不得以单一 pooled R²、NMAE 或一张平滑表面图宣布 Go。self-max 用 p99 稳健尺度、不用单点 max。
4. 当前仅形成讨论清单：**未生成下一轮配置、未提交新作业**。导师确认 P0 开发协议 + P0b/P1/P2 范围后再冻结最小矩阵。

### 9.4 对抗性审查（对本重排本身，2026-07-15）

对 §9.0–§9.3 逐条自证伪，避免把「看似合理的方向」写成结论：

1. **「扩数据能提精度」可能是把「数据量」与「多样性」混为一谈。** AG 池 40 例平台可能来自**同质性**而非样本不足；混入 AAA/ILO 同时改变了量与多样性，单条混合曲线无法把增益归因于「量」。→ 缓解：P1 必须同时跑逐队列曲线（AG-only 延长、AAA-only、ILO-only）与混合曲线，令「量 vs 多样性」可分离。
2. **跨队列合并可能拉低 AG，而非抬高。** 三队列峰值 WSS 量级可比（去风险），但解剖形状先验不同、网格密度差 4–6×（AG≈1.3 万 vs ILO≈8.1 万壁面点），FPS 覆盖与点预算不匹配；AAA/ILO 入口/出口 BC 是否共享 AG 模板**尚未刻画**，若不共享则混合模型面对更多隐藏 BC 方差。→ 缓解：cohort 作特征、统一点预算、逐队列评估、并把「AG-test16 不退化」设为硬验收门。
3. **P0b 可能直接推翻本节的天花板叙事。** 若 PointNeXt 在当前 split 上仍只有 ~0.21，则天花板随 split 前移；若达 ~0.31，则当前 PointNet 只是容量不足、「信息天花板」对本 split 属**过早断言**。这正是把 P0b 排在便宜前置的原因——**在 PointNeXt + OOF 复核前，不得把第五轮 dev1 的天花板直接钉在本矩阵 test16 上**。
4. **Murray 代理很可能边际甚至系统性错。** 第五轮 B1 全局面积仅 +0.03；Murray 定律在瘤体/病变/分叉段常失效，恰好在 WSS 最关键处代理偏差最大。→ 处理：预注册 null 为有价值结论（证明几何无法代理流量分配），不把 P2 写成必胜项。
5. **self-max R² 可能被单点 max 放大而过苛。** 病例 `max/p99` 中位 2.36，除以尖峰 max 对单点异常敏感，可能低估真实型态技能。→ 缓解：self-max 一律用 p99 稳健尺度，并与 Spearman/top-k 三角互证，不单独据 self-max 下结论。
6. **test16 复用使任何「涨点」含选模偏置。** 跨这些杠杆在 test16 上测得的改进都部分是「按 test 选出来的」；唯一干净信号是 P0 的 OOF + 最终对 test16 一次性审计。**严禁**在 test16 上迭代。
7. **不可无视导师给本矩阵设定的框架。** 本矩阵是导师要求的干净 PointNet baseline + 指定消融；整体转向 PointNeXt + 多队列属方向变更，必须提交导师确认，不能单方面开跑——故本节维持 No-Run，与既有 §9.3.4 及第五轮 No-Run 立场一致。

## 10. 配置与运行入口（2026-07-14）

- 五份配置：四个主矩阵配置加一个可选 `E23-GLOBAL`，位于 `training_wss_min/configs/pointnet_distribution_matrix/`。
- 第一阶段清单：`configs/sweeps/pointnet_distribution_global_stage1.txt`；第二阶段清单：`pointnet_distribution_case_stage2.txt`，不会自动串联提交。
- 可选交互清单：`configs/sweeps/pointnet_distribution_e23_optional.txt`，只包含 `E23-GLOBAL`。
- 单配置流水线：`cluster/run_pointnet_distribution_matrix.slurm`。训练日志仅报告采样目标空间 loss/MAE/RMSE；完整 R²/Spearman/top10/位置指标由训练后的全点 eval 生成。
- E4 配置：`configs/pointnet_deeper/e4_deep_global.json`；提交入口：`cluster/pointnet_deeper/submit.sh`；Job `8999` 已完训完评，复用上述完整流水线。
- 三层 SA foundation：`configs/pointnetpp_sa_foundation/sa3_xyzgeom.json`；运行 `python -m training_wss_min.tools.visualize_pointnetpp_sa` 只生成结构审计图件，不训练模型。
- evaluator 分别写 `eval/ckpt_best/` 与 `eval/ckpt_last/`；PostView 只为 best 导出全部 test16。STL 只保存几何，标量保存在 VTP/同点 CSV。

### 10.1 本轮 Slurm 提交记录

- 提交时间：2026-07-14；执行入口：`cluster/run_pointnet_distribution_matrix.slurm`。
- GLOBAL：`E2=8976`、`E3=8977`、`E23=8978`。
- CASE：`E2-CASE=8979`、`E3-CASE=8980`；依赖统一冻结为 `afterok:8976:8977`。
- 完成状态（2026-07-15 核验）：五个 Job 均 `COMPLETED (0:0)`；队列中无本矩阵残留任务。
- 用时：`8976=01:21:36`、`8977=00:08:20`、`8978=00:08:20`、`8979=01:20:40`、`8980=00:07:58`。固定 FPS-2000 宽网组明显更慢主要与该协议的数据准备/固定 FPS 路径有关，不能把墙钟时间差直接解释为容量代价。
- 每个 Job 已自动完成训练、best/last 的 train61+test16 全点评估，以及 best 的 test16 STL/VTP 导出；未根据 test 反选 checkpoint。
- E4 追加记录：`8999` 于 2026-07-15 `18:31:12–20:08:23` 运行，用时 `01:37:11`，状态 `COMPLETED (0:0)`；best/last eval 和 best test16 PostView 齐全。
