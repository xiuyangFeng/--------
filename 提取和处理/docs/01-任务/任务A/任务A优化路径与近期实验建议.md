# 任务A优化路径与近期实验建议

> 上位文档：[任务A实验清单](任务A实验清单.md) | 相关文档：[任务A实验状态表](任务A实验状态表.md) / [任务A冻结卡](任务A冻结卡.md)

---

## 1. 文档目的

本文件用于回答任务 A 基线完成后的两个现实问题：

1. 下一步应该先做消融，还是先做优化。
2. 当前主模型是否应该继续单纯加深加宽。

这里不追求一次列出所有可能改法，而是给出一条兼顾以下两个目标的执行路径：

- **短期目标**：尽快拿出一轮比当前 baseline 更好的结果，便于向老师汇报。
- **中期目标**：后续实验还能自然接到论文叙事，不把变量搅乱。

---

## 2. 当前基线结论的工作判断

基于 `A-Base-01/02/03` 与 `A-Main-01` 当前结果，可以先做三个判断：

### 2.1 当前主瓶颈不在壁面，而在内部流场

- `A-Main-01` 已经把壁面区域 `RMSE_|v|` 压到很低水平。
- 但内部区域 `RMSE_|v|` 仍显著偏高，速度分量 `R²_u / R²_v / R²_w` 也没有达到“结构已学清楚”的程度。
- 这说明模型已经学到边界条件、几何先验和局部趋势，但对内部复杂流动结构的表达仍不足。

### 2.2 当前阶段不应把“backbone 名字”当作首要矛盾

- 无 geometry 的 `Transformer` 与 `GraphSAGE` 几乎持平。
- 加入 geometry 后才出现明显增益。
- 因此，下一步的主问题不是“换一个更花的 backbone 名字”，而是：
  - 优化目标是否把学习能力放到了速度场上；
  - 当前单尺度表达是否足以覆盖内部复杂流场；
  - 模型深度增加后是否真的换来了有效感受野和有效表达。

### 2.3 当前阶段适合先做“小步优化”，不适合直接做“大改架构”

- 你已经有一组完整 baseline，可以支撑“现状诊断”。
- 但消融主线尚未完整展开，如果现在直接跳到多尺度 U-Net / 完整 PointNet++ / physics 组合，很容易失去归因。
- 所以最合理的路径是：先做一轮低耦合优化，看能否尽快压低内部速度误差；如果单尺度优化很快见顶，再进入多尺度升级。

### 2.4 端到端视角下的瓶颈重新审视（2026-03-25 补充）

上面三条判断是从"任务 A 流场重建精度"出发的。但论文最终要证明的是完整数字孪生链路：

```
流场 u,v,w → 壁面速度梯度 ∂u/∂n → WSS → TAWSS/OSI/RRT → 髂支闭塞风险预测
```

组内已有实验表明，**WSS、OSI 等壁面血流动力学指标对髂支闭塞预测的贡献远大于内部流场精度**。这意味着从端到端视角看，优化方向需要做以下修正：

#### 2.4.1 "壁面 RMSE 低 ≠ WSS 准确"

`A-Main-01` 壁面 `RMSE_|v|` = 0.0381，看起来很好。但 WSS 是**梯度量**：

$$\text{WSS} = \mu \left|\frac{\partial \mathbf{u}}{\partial n}\right|_{\text{wall}}$$

即使壁面速度本身很准，如果**近壁区域**（紧邻壁面的内部节点）速度不准，用有限差分或插值估计 ∂u/∂n 时，梯度误差仍然很大。当前近壁区域 `RMSE_|v|` = **1.5727**，这是决定 WSS 质量的真正瓶颈。

#### 2.4.2 对风险预测最关键的区域优先级

从端到端链路出发，各区域对最终临床预测的重要性排序为：

| 优先级 | 区域 | 当前 RMSE_\|v\| | 对端到端的影响 |
|--------|------|---------------|--------------|
| **最高** | 壁面 (is_wall=1) | 0.0381 | 直接决定 WSS 点值 |
| **极高** | 近壁区 (near-wall) | 1.5727 | 决定 ∂u/∂n 梯度质量，决定 WSS 精度 |
| **高** | 分叉/高曲率区 | 1.13 | OSI 异常高发区域，决定风险区定位 |
| **中** | 内部主流区 | 2.0668 | 对 WSS/OSI 贡献较小 |

#### 2.4.3 两条优化线并行推进

基于以上分析，当前优化应分两条线并行：

- **Line A（已启动）**：内部流场精度优化（A-Opt-01~10），继续按原计划推进，主要服务于任务 A 论文叙事和流场重建精度。
- **Line W（新增）**：壁面导向优化（A-Opt-W01~W05），直接服务于端到端链路质量，以 WSS/TAWSS/OSI 恢复精度为评判标准。

两条线共享同一套 **P0 组合训练配方**（`target_weights` + Pre-Norm；**`A-Opt-03w` 未更好**）。**（2026-03-31）默认起跑配置（母版 backbone）统一为 `A-Opt-05`**（`hidden_dim=256`，`num_layers=4`）：**`interior` / `near_wall` 等略优于 `A-Opt-03`**，更贴近 WSS 前置质量；**`A-Opt-03`（h128）** 作 **轻量/效率对照**。详见 [任务A实验状态表](任务A实验状态表.md)「战略锚点」。各自独立归因不变。

---

## 3. 关于“是否继续加深网络”的明确判断

### 3.1 可以继续加深，但不能把“加深”当作默认主解

可以继续加深，但我不建议把“层数继续堆高”作为下一阶段主线。

原因有三点：

- **第一，当前 `FieldTransformer` 仍是单尺度局部消息传递。**
  即使层数增加，感受野扩大也仍然依赖局部邻域层层传播，捕获大尺度分叉结构的效率有限。

- **第二，深层收益需要训练结构先跟上。**
  如果没有更规范的归一化和残差组织，直接从 3 层加到 6 层甚至更高，未必能稳定转化成有效表达。

- **第三，256 维不是问题核心，“只有 256 维表达不了复杂流场”这个判断不能直接成立。**
  真正要看的不是隐藏维本身，而是：
  - 深度上去后验证误差是否继续下降；
  - 内部区域指标是否改善；
  - 速度分量 `R²` 是否同步改善；
  - 改善是否值得额外显存和时间成本。

### 3.2 对“继续加深”的建议顺序

不建议直接跳到很深。建议按下面顺序试：

1. 先在当前 Transformer 残差块中补 `LayerNorm`，采用 Pre-Norm 风格。（**已完成**：`A-Opt-02` 三 seed，结论见下文 **P0-2** 与 [任务A实验状态表](任务A实验状态表.md)。）
2. 再把 `hidden_dim` 从 `128` 提到 `256`。
3. 再试 `num_layers = 4`。
4. 如果 `4` 层仍有稳定收益，再试 `6` 层。

不建议一开始直接做：

- `hidden_dim = 256` 且 `num_layers = 6`
- 同时叠加新损失、新调度器、新增强

这样做的风险是：一旦结果变化，你无法判断收益来自哪里。

### 3.3 怎样判断“继续加深还值不值”

若出现下面任一情况，就说明单纯加深的边际收益已经接近上限：

- `val_loss` 和内部区域 `RMSE_|v|` 基本不再下降。
- 速度分量 `R²_u / R²_v / R²_w` 没有明显改善。
- 训练时间和显存增长明显，但全局 `RMSE_|v|` 只改善很小。
- 壁面区继续改善，但内部区几乎不动。

一旦出现这些信号，就不应该继续在线数上死磕，而应转向多尺度结构。

---

## 4. 先做消融还是先做优化

### 4.1 当前建议：先做一轮“主线优化”，再回到消融

如果老师近期更需要看到“结果继续变好”，那就不建议你现在立刻把所有时间先投入完整消融。

更合适的顺序是：

1. **先做一轮小规模、低耦合优化**，目标是尽快拿到比 `A-Main-01` 更好的结果。
2. **优化出一个更强的单尺度主线后**，再围绕这个主线组织“原因解释型实验”。
3. **正式论文写作时**，baseline 负责交代起点，优化线负责交代怎么提高，消融负责交代为什么提高。

### 4.2 为什么不建议现在只做消融

如果现在直接进入完整消融，短期内你会回答很多“为什么”，但不一定能很快把主结果继续推高。  
这对论文必要，但对阶段汇报未必最优。

### 4.3 为什么也不建议现在只做大规模优化

如果只顾堆优化，不安排后续最少必要的解释实验，后面文章会出现两个问题：

- 结果变好了，但无法说清楚究竟是哪一类设计起了作用。
- 优化实验之间变量耦合过多，论文审稿时很难讲清公平性。

所以最合理的方案不是“消融 vs 优化 二选一”，而是：

- **近期先做 1 轮主线优化拿结果；**
- **随后做最小必要消融把结果解释清楚。**

---

## 5. 推荐的优化主线

下面的优化顺序按“预期收益 / 实现成本 / 归因清晰度”综合排序。

### 5.1 P0：先做低成本主线优化

这是最应该先做的一层，目标是尽快拿到一版明显优于 `A-Main-01` 的结果。

#### P0-1 调整目标损失权重

建议先试：

- `target_weights = [2.0, 2.0, 2.0, 0.5]`

理由：

- 当前压力已经不差，速度才是主瓶颈。
- 这类改动实现最简单，归因最清楚。
- 即使效果一般，也能快速判断“问题是否主要来自优化目标分配失衡”。

建议暂时不要第一轮就上特别激进的 `[3, 3, 3, 0.5]`，先看保守权重是否已经能带来稳定收益。

**具体操作方法：** 不需要改任何代码。以 `A-Main-01` 的 `config.snapshot.json` 为模板复制一份，仅修改以下字段：

```json
"optim": {
  "target_weights": [2.0, 2.0, 2.0, 0.5]
},
"meta": {
  "exp_id": "A-Opt-01",
  "study_group": "optimization",
  "question": "速度权重加大是否改善内部流场",
  "ablation_axis": "target_weights"
}
```

保存为 `training/configs/field/generated/optimization/A-Opt-01_seed1.json`，运行：

```bash
conda activate rag_venv
python -m training.scripts.train_field \
  --config training/configs/field/generated/optimization/A-Opt-01_seed1.json
```

> **结果摘要（2026-03-26，三 seed）**：`A-Opt-01` 相对 `A-Main-01` 在测试集上 **`RMSE_|v|` 与分区域 `interior / high_curvature / bifurcation / trunk` 的 `rmse_vel_mag` 均一致下降**，`near_wall` 基本持平；`RMSE p` 未退化。单 run 与多模型汇总图见各 run 下 `predictions_test/regional_eval/` 与 `outputs/field/plots/multimodel_baseline/fig_A5_multimodel_regional_bar_*.png`。台账与数值见 [任务A实验状态表](任务A实验状态表.md)「A-Opt-01」。

训练结束后请在本机或集群 **`GNN` 环境**下补全与 baseline 相同的后处理链路（与 `A-Main-01` 一致）：

```bash
python -m training.scripts.predict_field \
  --config outputs/field/<run_dir>/config.snapshot.json \
  --checkpoint outputs/field/<run_dir>/best_model.pt \
  --subset test
python -m training.scripts.plot_taskA_regional_bar \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json
```

> **权重兼容性（重要）**：若在 `FieldTransformer` 中引入了 Pre-Norm 的 `LayerNorm` 参数，则**旧 checkpoint（无对应 state_dict 键）无法加载**。当前仓库已用配置项 **`model.use_transformer_prenorm`**（默认 `false`，与 `A-Main-01` / `A-Opt-01` 对齐）/（`A-Opt-02` 配置中为 `true`）区分两种结构；导出预测前请确认 `config.snapshot.json` 与 checkpoint 训练时一致。

#### P0-2 给 Transformer 残差块补 LayerNorm

**实现状态**：代码侧已支持 **`use_transformer_prenorm`**（2026-03-26 起），不再依赖手工改 `forward` 注释。**三 seed 训练与评估已归档**（2026-03-27，见本节末「实验结论摘要」）；下列代码片段仍保留为 **Pre-Norm 设计动机与结构说明**。

建议把当前主线 Transformer 调整为 Pre-Norm 风格残差块，再观察：

- 是否更稳定；
- 在增加深度时是否更能释放收益；
- 内部区域误差是否下降。

这一项适合作为结构层面的最小增强。

**具体操作方法：** 需修改 `training/core/models.py` 中的 `FieldTransformer` 类，改动约 10 行。

1. 在 `__init__` 中新增 `LayerNorm` 列表（在 `self.post_layers` 之后添加）：

```python
self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
```

2. 将 `forward` 中的消息传递循环改为 Pre-Norm 风格：

```python
# 改前（当前代码）：
for conv, linear in zip(self.layers, self.post_layers):
    residual = x
    x = conv(x, data.edge_index)
    x = F.elu(x)
    x = linear(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x + residual

# 改后（Pre-Norm 风格）：
for conv, linear, norm in zip(self.layers, self.post_layers, self.norms):
    residual = x
    x = norm(x)
    x = conv(x, data.edge_index)
    x = F.elu(x)
    x = linear(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x + residual
```

Pre-Norm 的关键在于：先归一化再做注意力计算，残差加在未归一化的原始表示上。这一设计在标准 Transformer 文献中已被证明对深层网络的训练稳定性有显著帮助。

配置 JSON 与 `A-Main-01` 完全相同，仅修改 `meta.exp_id` 为 `A-Opt-02`。因为是代码变更而非配置变更，需在 `meta.notes` 中注明 `"models.py: FieldTransformer Pre-Norm LayerNorm"`。

> **注意：** 该修改会改变模型参数结构，旧的 `best_model.pt` 不兼容。新实验从头训练即可，已完成的 baseline 结果不受影响。

**实验结论摘要（2026-03-27，`split_AG_v1`，三 seed）**

- **Run 目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_split_AG_v1_seed{1,2,3}_20260327_*`（`A-Opt-02`，`model.use_transformer_prenorm: true`，其余与 `A-Main-01` 对齐）。
- **相对 `A-Main-01`（同 split、同特征）**：`summary.json` 上合成 **`RMSE` 与 `test loss` 三 seed 均值下降**；**`rmse_vel_mag`（`test_metrics`，全图节点口径）** 均值约 **1.161 → 1.113**；**内部点 `interior.rmse_vel_mag`（`regional_eval`）** 均值约 **2.067 → 1.973**。分区域 **`rmse_vel_mag`** 上，**内部 / 近壁 / 高曲率 / 分叉 / 主干** 较 Main 均值改善；**壁面区** 均值略升且种子方差更大（**0.038 → 0.041** 量级），报告时宜并列写出。
- **稳定性**：seed1/2 的 **`best_epoch` 约 144～148**，seed3 约 **64** 且测试指标偏弱，**随机种子敏感性仍在**。
- **归档与作图**：各 run 已具备 `predictions_test/`、`predictions_test/error_analysis_interior/`、`regional_eval/`。与 **`A-Main-01` 并排** 的内部节点 **|v|**、**p** 散点见 **`outputs/field/plots/optimization/prenorm_A_Opt02_vs_Main01/fig_A3_multimodel_scatter_{vel_mag,p}_interior_geo_only_seed{1,2,3}.png`**（`plot_taskA_multimodel_scatter` 支持 **`--tag`**；图例 **`A-Opt-02` → 「Transformer+Geom (Pre-Norm)」**）。
- **P0-3（已完成，2026-03-27）**：实验 **`A-Opt-02_warmup`** 三 seed 已训练并归档；run 目录 `outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_warmup5_split_AG_v1_seed{1,2,3}_20260327_*`。结论摘要见本节 **「P0-3 实验结论摘要」** 与 [任务A实验状态表](任务A实验状态表.md)「实验记录摘要 · A-Opt-02_warmup」。三模型对照图：**`outputs/field/plots/optimization/prenorm_Main_P02_P02w/`**（`python -m training.scripts.regenerate_p02_warmup_comparison_figures` 可重生成）。  
- **再下一步（P0-4，✅ 2026-03-28）**：**`A-Opt-03`**（`A-Opt-01` 的 `target_weights` + **`use_transformer_prenorm: true`**）与 **`A-Opt-03w`**（再叠 **`warmup_epochs=5`**）均已 **三 seed 归档**；训练期 **`val`/`best_epoch` 汇总**：`outputs/field/plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`。  
  - **`A-Opt-03` 结论（h128 P0-4）**：**速度侧主指标相对 `A-Main-01`、`A-Opt-01`、`A-Opt-02` 为当时 P0 线最优**（`summary.test_metrics.rmse_vel_mag` 三 seed 均值 **~1.031**；`regional_eval` · **`interior.rmse_vel_mag` ~1.822**；**`wall.rmse_vel_mag` 亦优于两条单改线**）；**内部点 `R²_u/v/w` 均值** 高于单独 P0-1 / P0-2。**压力侧**：**`summary.test_metrics.rmse_p` 三 seed 均值**（~**0.642**）**差于 `A-Opt-01` / `A-Opt-02`**（~**0.620 / 0.610**），与低压权一致；**`regional_eval` · `all.rmse_p`** 与 **`A-Opt-02`** 持平，**`interior.rmse_p`** 略差于 **`A-Opt-02`**。**（2026-03-31）** 均值与 **近壁等区** 上 **`A-Opt-05`** 略进一步 → **新开跑以 05 为母版**。  
  - **`A-Opt-03w`**：相对 **`A-Opt-03`** **速度未更好、`rmse_p` 未收复**——**组合线默认不必开 warmup**。  
  - **（2026-03-29 更新）P0-5 容量**：**`A-Opt-04`**（`hidden_dim=256`）**三 seed 已归档**；相对 **`A-Opt-03`**，**`interior.rmse_vel_mag` 变差（~1.822→1.849）**，**`summary.rmse_p` 略优**——**未过 §「P0-5 推进门槛」中「内部速度主指标继续改善」**。**`A-Opt-05`**（`num_layers=4`）**已跑完**：相对 04 **回补内部速度**，**均值略优于 03**（~1.816 vs 1.822），**近壁等区域略优**，**方差与成本更高**。
  - **（2026-03-31 更新）战略母版**：**后续消融 / Line G / Line W 以 `A-Opt-05` 为配置复制起点**；**`A-Opt-03`** 作 **轻量对照**（见状态表「战略锚点」）。容量线写入正文或附录时须**并列写出 trade-off**（`summary.rmse_p`、显存、方差）。
  - **（2026-03-31 更新）P0-5 后 `A-Opt-05_tune`**：在 **`A-Opt-05`** 上试了 **10-epoch warmup + 学习率 / 权重衰减 / 调度耐心** 等小步改动；**已入账 run** 与 **后处理产物**见 [任务A实验状态表](任务A实验状态表.md)「`A-Opt-05_tune`」。**seed1 横向图（对 `A-Opt-03`）**：`outputs/field/plots/optimization/A_Opt05_tune_vs_Opt03_seed1/`（**读图时以 05 为主视角**）。**结论**：**`lr3e-4` 有潜在收益、需多 seed 验证**；**`wd2e-4` 当前不可取**；**`schedpat15`** 相对默认 **`warmup10`** **未形成稳定主指标优势**——**不改变「母版仍为骨架 `A-Opt-05`、调参分支单独归因」**。**壁面 WSS 全量对比暂缓**。

#### P0-3 学习率 Warmup

除已完成的 **`A-Opt-02_warmup`（`warmup_epochs=5`）** 外，主文档基线与多数优化实验仍默认 **`warmup_epochs = 0`**，训练从第一个 epoch 即使用完整学习率 5e-4。从 `A-Main-01` 的训练曲线看，前几个 epoch loss 下降很快，但也伴随着明显的震荡。5~10 epoch 的线性 warmup 能让模型在初期以低学习率找到稳定优化方向，再逐步加速，代价几乎为零。

**这是纯配置改动，不需要任何代码修改**（`train_field.py` 已内置 `LinearLR` warmup 支持）。

**具体操作方法：** 在 JSON 配置中添加一个字段即可：

```json
"optim": {
  "warmup_epochs": 5
}
```

当前 `ReduceLROnPlateau` 调度器已与 warmup 兼容（见 `trainer.py` 的 `_step_scheduler`：warmup 阶段走 `LinearLR`，结束后自动切回 Plateau）。

**与本仓库的对齐（2026-03-27，结论延展 2026-03-31）**：P0-3 已落地为 **`A-Opt-02_warmup`**（在 **`A-Opt-02`** 上仅打开上述 warmup），与 [任务A冻结卡](任务A冻结卡.md)、[任务A实验状态表](任务A实验状态表.md) 第三批推荐顺序一致。**`A-Opt-02_warmup`（三 seed）已归档**。**`A-Opt-03` / `A-Opt-03w`（P0-4）已于 2026-03-28 归档**。**`A-Opt-04` / `A-Opt-05`（P0-5）已于 2026-03-29 归档**。**（2026-03-31）新开跑母版切换为 `A-Opt-05`**（见状态表「战略锚点」）。训练复现示例（P0-3）：

```bash
cd /public/newhome/cy/Digital_twin/GNN
python -m training.scripts.train_field \
  --config training/configs/field/generated/optimization/A-Opt-02_warmup_seed1.json
```

（seed 2、3 替换对应配置文件路径即可。）

**实验结论摘要（2026-03-27，`split_AG_v1`，`A-Opt-02_warmup` 三 seed，相对 `A-Opt-02` / `A-Main-01`）**

- **训练动态**：`best_epoch` 约 **93 / 70 / 113**，**不再出现仅 seed3 极端早停于 ~64** 的模式（与 **`A-Opt-02`** / Main 的 seed3 现象对照），**warmup 对 Pre-Norm 支线的种子稳定性为正贡献**。
- **`summary.json`（全图节点）**：**`rmse_vel_mag` 均值** 相对 **`A-Opt-02`** **略升略降并存**（约 **1.113 → 1.110**）；**合成 `RMSE` 略变差**（**0.761 → 0.766**）；**`rmse_p` 三 seed 均值明显变差**（约 **0.610 → 0.642**），仍**优于 `A-Main-01` 的 ~0.654**。
- **`regional_eval` · `interior.rmse_vel_mag`**：三 seed 均值相对 **`A-Opt-02`** **小幅下降**（约 **1.973 → 1.965**）；**seed3 单 seed 改善明显**（约 **2.06 → 1.91**），与「救差种子」归因一致。
- **壁面等指标**：与 **`A-Opt-02`** 相比 **互有小幅胜负**；汇报宜用 **`prenorm_Main_P02_P02w`** 中 Fig A5/A4 并列，避免只引用全图 `summary`。
- **叙事建议**：**5 epoch warmup 不是 P0-2 的严格占优叠加项**，而是 **以压力略差换速度与内部区域稳定性/最差 seed 修复** 的折中。**（2026-03-28）** **`A-Opt-03` 已出**：**组合线默认可不写 warmup**（**`A-Opt-03w` 未优于 `A-Opt-03`**）；**仅在 P0-2 单线**上是否保留 warmup 仍可单独决定。

#### P0-4 组合实验优先级

第一批建议只跑下面几组（均 seed=1 先做 smoke test）。**与上文衔接**：**`A-Opt-02`、P0-3 / `A-Opt-02_warmup`（2026-03-27）与 P0-4 / `A-Opt-03`、`A-Opt-03w`（2026-03-28）、P0-5 / `A-Opt-04`、`A-Opt-05`（2026-03-29）均已归档**；**下一优先为叙事收敛与消融 / Line G（见 §P0-6），非继续叠深 `A-Opt-06`。**

1. `A-Main-01` 当前基线（已完成，直接比较）
2. `A-Opt-01`：仅 `+ target_weights [2,2,2,0.5]`
3. `A-Opt-02`：仅 `+ LayerNorm`
4. `A-Opt-02_warmup`：`A-Opt-02 + optim.warmup_epochs=5`（**P0-3**，三 seed ✅ 2026-03-27）
5. `A-Opt-03`：`target_weights + LayerNorm`（**P0-4**，三 seed ✅ 2026-03-28）
6. `A-Opt-03w`：`target_weights + LayerNorm + warmup=5`（**P0-4**，三 seed ✅ 2026-03-28；**未优于 `A-Opt-03`**，与 P0-3 的 `A-Opt-02_warmup` **归因对象不同**、互不替代）

这样能非常清楚地回答：

- 主要收益来自损失重分配，还是来自网络表达稳定性；
- 两者是互补，还是其中一项基本无效；
- warmup 在更好的基础配置上是否进一步稳定训练。

> **评估要点：** 每组实验完成后，不要只看全局 `RMSE_|v|`，必须同时运行预测和区域评估：
>
> ```bash
> python -m training.scripts.predict_field --run-dir <outputs/field/xxx>
> python -m training.scripts.plot_taskA_regional_bar --run-dir <outputs/field/xxx>
> ```
>
> 重点对比 `interior.rmse_vel_mag` 和 `R²_u / R²_v / R²_w`。

#### P0-5 推进门槛

为了避免一边跑一边反复改主意，建议在进入下一阶段前使用统一推进门槛：

- **从 `A-Opt-01/02/03` 进入 `A-Opt-04` 的条件：**
  - 全局 `RMSE_|v|` 相对 `A-Main-01` 下降；
  - `interior.rmse_vel_mag` 同步下降；
  - `R²_u / R²_v / R²_w` 中至少 1 项出现明确改善；
  - 若只看到压力更好、壁面更好，而内部区不动，则不进入容量扩展。

- **从 `A-Opt-04` 进入 `A-Opt-05/06` 的条件：**
  - `hidden_dim = 256` 后，全局与内部主指标仍继续改善；
  - 训练时间和显存上涨仍在可接受范围内；
  - 改善不是仅来自壁面区，而是内部区也同步变好。

- **（2026-03-29）本仓库对已跑结果的执行建议（与上条对照）**：**`A-Opt-04` 未满足「内部继续改善」**，严格按条应 **停止于 04**；**`A-Opt-05` 已作为探索性加深跑完**，可视为「04 的补救实验」——**不必机械删数**，但 **不默认启动 `A-Opt-06`**。

- **停止继续堆深度并转向多尺度的条件：**
  - `A-Opt-04/05` 相对前一组只带来很小改善；
  - `val_loss`、`RMSE_|v|` 与 `interior.rmse_vel_mag` 已接近平台；
  - 速度分量 `R²` 基本不再提升；
  - 额外深度主要带来训练成本上涨，而不是有效精度收益。

#### P0-6 显式几何特征增强建议（Line G，2026-03-26 新增）

既然当前 baseline 已经表明 geometry 是主增益来源，下一步确实可以继续深挖几何/局部血流相关先验；但**不建议一次性向 `data.x` 塞入很多新维度**。对当前任务来说，真正稳妥的做法是：

- 先完成 `A-Abl-02`，确认现有 `Abscissa / NormRadius / Curvature / Tangent` 的主贡献项；
- 再按“**每次只新增 1 类特征**”的小步方式推进；
- 每组新特征与 **`A-Opt-05` 母版**做一对一对照（论文中与 **`A-Main-01`** 的叙事对比可保留，但控制变量以 05 为准）；
- 判断标准不能只看全局 `RMSE_|v|`，还必须看 `near_wall / bifurcation / high_curvature` 是否改善。

下面这张表给出当前最值得优先尝试的显式几何增强候选：

| Line G 实验 | 候选特征 | 物理含义 | 预期改善区域 | 提取难度 | 过拟合/冗余风险 | 当前建议 |
| --- | --- | --- | --- | --- | --- | --- |
| `A-Opt-G01` | 距分叉点距离 + 分叉拓扑标记 | 显式编码局部拓扑转折，告诉模型“这里离分叉还有多远” | `bifurcation`、`trunk->branch` 过渡区 | 低 | 低 | **最优先** |
| `A-Opt-G02` | 半径变化率 / 截面积变化率 | 比 `NormRadius` 更进一步，描述扩张、收缩和局部几何突变 | `bifurcation`、瘤腔扩张区、近壁剪切变化区 | 中 | 中 | **高优先** |
| `A-Opt-G03` | 扭率 `torsion` | 补充曲率无法表达的三维空间扭转信息 | 三维弯扭段、高曲率区 | 中 | 中 | **高优先** |
| `A-Opt-G04` | 壁面距离相关特征 | 直接告诉模型节点离壁面有多远，服务近壁速度剖面学习 | `near_wall`、WSS 相关区域 | 中 | 低 | **高优先** |
| `A-Opt-G05` | 中心线方向变化率 | 比单点 `Tangent` 更强调局部流向转折强度 | 高曲率段、入口下游扰动区 | 中 | 中 | 值得尝试 |

> **不建议优先做的事情**：一口气增加 3~5 类新特征；加入定义口径不稳定的高阶导数几何量；在样本量不大的情况下堆很多彼此强相关的几何列。

**Line G 推荐执行顺序：**

1. `A-Abl-02` 先跑，先证明当前几何分量里谁最关键；
2. `A-Opt-G01`：先补拓扑显式先验；
3. `A-Opt-G02`：再补局部尺度变化信息；
4. `A-Opt-G03 / G04`：视分区域误差走向二选一或顺序推进；
5. `A-Opt-G05`：作为第五优先级补充项。

**Line G 推进门槛：**

- 训练集和验证集都改善，才视为正向；
- 至少一个复杂区域（`near_wall / bifurcation / high_curvature`）出现明确改善；
- 若训练更好、验证不动或变差，则判定为“特征加噪声”，停止继续堆这一路；
- 若 `A-Opt-G01/G02` 都没有带来稳定收益，优先回头检查当前几何提取口径，而不是继续扩维。

### 5.2 P1：在 P0 有正收益后，做容量扩展和区域加权

P1 包含两个独立方向，分别解决不同层面的问题。

#### P1-1 容量扩展

如果 P0 的结果是正向的，再考虑增大模型容量。

注意：

- 这里推荐“先加宽，再适度加深”，不是反过来。
- 每次只改一个核心维度，不要同时改很多项。

建议的顺序：

1. `128 x 3` + P0-4（已归档为 **`A-Opt-03`**；**轻量对照**）
2. `256 x 3`（已归档为 **`A-Opt-04`**）
3. `256 x 4`（已归档为 **`A-Opt-05`**，**当前消融/Line G/W 母版**）
4. **`256 x 6`（`A-Opt-06`）默认不启动**，除非要写「容量极限」附录

**具体操作方法：** 纯配置改动，修改 JSON 中的 `model` 部分：

```json
"model": {
  "name": "transformer",
  "hidden_dim": 256,
  "num_layers": 3,
  "dropout": 0.1,
  "heads": 4
}
```

`hidden_dim=256` 时 `heads=4` 仍然合理（每头 64 维）。参数量变化：`128 x 3` 约 130K，`256 x 3` 约 500K，`256 x 6` 约 1M。若单卡显存紧张，可将 `accumulate_grad_batches` 从 1 提到 2。

#### P1-2 内部点区域加权损失

区域加权直接针对当前最大瓶颈（内部流场速度误差高），改动集中在 `losses.py` 一个文件，归因清晰。

核心思路：利用已有的 `is_wall` 标记，让内部点（`is_wall=0`）的损失权重大于壁面点（`is_wall=1`）。

> **为什么从原 P2 提升到 P1：** 当前内部点 `RMSE_|v|` = 2.07 vs 壁面 = 0.038，差距超过 50 倍。全局均匀 MSE 被大量低误差壁面点稀释，优化器在内部高误差区投入的梯度信号不足。区域加权是对此最直接的干预，且与容量扩展可独立验证。

**具体操作方法：** 需修改 `training/core/losses.py`。

1. 新增区域加权 MSE 函数（在 `weighted_mse_loss` 后面添加）：

```python
def region_weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    is_wall: torch.Tensor,
    interior_boost: float = 3.0,
) -> torch.Tensor:
    mse_per_node = F.mse_loss(pred, target, reduction="none")  # [N, 4]
    node_weight = torch.where(
        is_wall.squeeze(-1).bool(),
        torch.ones(pred.size(0), device=pred.device),
        torch.full((pred.size(0),), interior_boost, device=pred.device),
    )
    weighted = (mse_per_node * weights.unsqueeze(0)) * node_weight.unsqueeze(-1)
    return weighted.mean()
```

2. 在 `NullPhysicsLoss.build_loss` 和 `PhysicsConstraintLoss.build_loss` 中，需要从 `batch` 取 `is_wall` 列传入。`is_wall` 在节点特征 `data.x` 的第 9 列（即 `NODE_FEATURE_NAMES.index("is_wall")`）。当 `interior_boost > 1.0` 时启用区域加权，否则退化为原有的均匀加权。

3. 建议在 `OptimConfig` 新增一个字段 `interior_loss_boost: float = 1.0`（默认不加权），通过 JSON 配置对比不同权重值（如 2.0、3.0、5.0），而不需要反复改代码。

### 5.3 P2：训练策略精调与物理约束

P2 的项在 P0/P1 效果确认后再逐步叠加，每项改动都很小但需要验证交叉效果。

#### P2-1 增大有效 batch

当前 `batch_size=2`，有效梯度估计噪声较大。

**具体操作方法：** 纯配置改动，在 JSON 中设置：

```json
"optim": {
  "accumulate_grad_batches": 4
}
```

等效 `batch_size=8`，不增加显存峰值，但每 4 步才更新一次参数，梯度估计更稳定。

#### P2-2 启用连续性物理约束

代码中已完整实现 Navier-Stokes 物理约束（`losses.py` 中的 `PhysicsConstraintLoss`），当前全部关闭。建议先只开连续性方程（不可压条件 `div(v)=0`），它是零阶 PDE 约束，计算最稳定。

**具体操作方法：** 纯配置改动，在 JSON 中设置：

```json
"physics": {
  "enabled": true,
  "warmup_epochs": 20,
  "continuity_weight": 0.01,
  "momentum_weight": 0.0,
  "no_slip_weight": 0.0,
  "auto_load_scales": true
}
```

`warmup_epochs=20` 让模型先学到基本场分布再引入 PDE 约束。`continuity_weight` 从 0.01 起步。

> **注意：** 物理约束需要对输入坐标做 autograd，训练速度降低约 2~3 倍。只在 P0/P1 的最优配置上试。

#### P2-3 数据增强强化

当前增强只有旋转+平移。可以启用微小缩放扰动。

**具体操作方法：** 修改 JSON 的 `augment_config`，加入 `"scale_prob": 0.3`。

### 5.4 P3：单尺度见顶后，再做多尺度架构升级

如果前面步骤都做了，内部区域速度仍是硬瓶颈，就可以把多尺度结构正式立项为下一阶段主线。

推荐优先考虑 **图 U-Net / 层次化池化-上采样结构**，而不是单纯更深的单尺度 Transformer。

理由：当前问题更像"缺少大尺度结构感知"，而不是"少几层非线性"。

**设计思路与具体操作方法：**

图 U-Net 的核心是在图数据上构造编码器-解码器结构：

```
编码器：细图 → (GNN + Pool) → 中图 → (GNN + Pool) → 粗图
解码器：粗图 → (Unpool + GNN + Skip) → 中图 → (Unpool + GNN + Skip) → 细图
```

关键技术选型：

1. **图池化（Encoder 下采样）**：推荐 PyG 的 `TopKPooling`（按节点得分保留 top-k 比例节点）或 `voxel_grid`（按空间栅格合并）。建议保留比例 0.5，即每层减半。
2. **图反池化（Decoder 上采样）**：使用 `knn_interpolate` 将粗图特征插值回细图节点位置。
3. **Skip Connection**：编码器每一层的输出与解码器对应层做 `torch.cat` 后再过一层 GNN，保留细粒度几何细节。
4. **每层 GNN 模块**：可复用已有的 `TransformerConv` 或 `SAGEConv` 残差块。

实现规模约 200~300 行（新增一个 `FieldGraphUNet` 类），需要在 `MODEL_REGISTRY` 中注册。建议以独立文件 `training/core/models_unet.py` 实现，避免打乱现有模型代码。

> **PyG 已有 `GraphUNet` 参考实现**（`torch_geometric.nn.models.GraphUNet`），可以作为起点，但它默认用 `TopKPooling`，需要根据血管网格的特性调优池化比例和层数。

### 5.5 壁面导向优化线 Line W（2026-03-25 新增）

> **设计动机**：组内实验已表明壁面血流动力学指标（WSS/OSI/RRT）对髂支闭塞的预测贡献远大于内部流场精度。Line W 直接面向端到端链路质量优化，与 Line A（内部精度优化）并行推进、独立归因。
>
> **前置条件**：Line W 的所有实验均基于 **`A-Opt-05` 母版**起跑（**`A-Opt-03w` 不作为母版**；若以 **`A-Opt-03`** 做低开销对照须单独声明）。
>
> **评估标准差异**：Line W 的评估不只看 `RMSE_|v|`，而是必须同时运行 WSS 后处理脚本，以 WSS/TAWSS/OSI 的恢复质量（点级 R²、病例级 Pearson/Spearman）作为核心判定指标。

#### 5.5.1 PW0：近壁区域加权损失

**研究问题**：将损失函数的区域加权从"内部 boost"翻转为"近壁 boost"，是否能改善 WSS 相关的速度梯度质量。

**核心思路**：利用已有的壁面距离信息（近壁区域 mask），让近壁点（`distance_to_wall < threshold`）和分叉/高曲率点的损失权重大于内部主流区点。

**建议权重方案**：

| 区域 | 权重 | 理由 |
|------|------|------|
| 壁面 (is_wall=1) | 1.0~2.0 | 壁面速度已很准，适度维持 |
| 近壁区 (near-wall) | **3.0~5.0** | WSS 梯度估计的关键区域 |
| 分叉/高曲率区 | **2.0~3.0** | OSI 异常高发区域 |
| 内部主流区 | **0.5~1.0** | 对 WSS/OSI 贡献较小 |

**具体操作方法**：需修改 `training/core/losses.py`。

1. 新增 `wall_oriented_mse_loss` 函数：

```python
def wall_oriented_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    is_wall: torch.Tensor,
    near_wall_mask: torch.Tensor,
    wall_boost: float = 1.5,
    near_wall_boost: float = 3.0,
    interior_weight: float = 0.5,
) -> torch.Tensor:
    mse_per_node = F.mse_loss(pred, target, reduction="none")  # [N, 4]
    is_w = is_wall.squeeze(-1).bool()
    is_nw = near_wall_mask.squeeze(-1).bool() & ~is_w
    node_weight = torch.full((pred.size(0),), interior_weight, device=pred.device)
    node_weight[is_nw] = near_wall_boost
    node_weight[is_w] = wall_boost
    weighted = (mse_per_node * weights.unsqueeze(0)) * node_weight.unsqueeze(-1)
    return weighted.mean()
```

2. `near_wall_mask` 的来源：在 `pipeline/preprocess.py` 阶段，对每个内部节点计算到最近壁面节点的欧氏距离，距离低于阈值的标记为近壁区（建议阈值 = 壁面距离分布的 25th percentile）。将该标记作为 `data.x` 的新列或存入 `data` 的额外属性。

3. 在 `OptimConfig` 中新增字段：
   - `wall_boost: float = 1.0`
   - `near_wall_boost: float = 1.0`
   - `interior_weight: float = 1.0`

所有默认值为 1.0，退化为原有均匀加权行为，向后兼容。

**实验编号**：`A-Opt-W01`

#### 5.5.2 PW1：壁面法向速度梯度监督

**研究问题**：在标准数据 MSE 之上，增加壁面法向速度梯度的直接监督，是否能提升 WSS 精度。

**核心思路**：WSS = μ × |∂u/∂n|\_wall，如果直接监督 ∂u/∂n，等价于端到端优化 WSS。当前 `losses.py` 已有 `PhysicsConstraintLoss._grad()` autograd 基础设施，可复用。

**具体做法**：

对每个壁面节点 i：
1. 找到其在图中通过 `edge_index` 连接的最近**内部**邻居 j。
2. 计算壁面法向方向的有限差分：`grad_n ≈ (v_j - v_i) / |pos_j - pos_i|`。
3. 对所有壁面节点计算上述近似梯度，与 CFD 真值的对应梯度做 MSE。

也可以使用 autograd 方式（与现有物理约束的实现一致）：对预测输出 u,v,w 在壁面节点处求对坐标的梯度，再投影到壁面法向。

**损失函数形式**：

```
L_total = L_data + λ_wall_grad × L_wall_gradient
```

建议 `λ_wall_grad` 从 0.01 起步，逐步增大。

**前置依赖**：
- 需要壁面法向量（pipeline 阶段已有 `Tangent`，可由此推导或直接从网格文件提取法向）
- 或使用邻居有限差分替代法向梯度，不依赖额外数据

**实验编号**：`A-Opt-W02`

**改动量**：代码约 40~60 行（`losses.py` 新增函数 + `trainer.py` 接入）

#### 5.5.3 PW2：直接 WSS 监督

**研究问题**：如果直接用 WSS 作为额外监督目标（而不仅仅监督 u,v,w,p），是否能最大程度提升端到端链路质量。

**核心思路**：从 CFD 后处理中获取每个壁面节点每个时间步的 WSS 标量真值，在训练时把 WSS 作为第 5 个输出维度（或作为辅助损失项）。

**方案 A：辅助损失项**

```
L_total = L_data(u,v,w,p) + λ_wss × MSE(WSS_pred, WSS_cfd)
```

其中 `WSS_pred` 通过对预测速度场在壁面法向方向的可微分有限差分计算得到。

**方案 B：额外输出头**

模型输出维度从 4 扩展为 5（u, v, w, p, WSS），WSS 只在壁面节点上有真值和梯度。

推荐先用方案 A，因为它不改变模型输出结构，且 WSS 的物理定义保证了与速度场的一致性。

**前置依赖**：
- CFD 后处理中每个壁面节点的 WSS 真值（需预计算并存入 `.pt` 图数据）
- 可微分的 WSS 计算函数

**实验编号**：`A-Opt-W03`

**改动量**：代码约 80~120 行（WSS 计算函数 + 损失集成 + 数据管线增加 WSS 目标）

#### 5.5.4 PW3：两阶段训练——全局场 → 壁面精调

**研究问题**：先用标准 MSE 学到全局流场分布，再用壁面加权损失做精调，是否比一开始就加权更有效。

**核心思路**：

- 阶段 1（epoch 1~N1）：使用标准均匀 MSE，让模型先学到全局分布。
- 阶段 2（epoch N1+1~N2）：冻结大部分参数或降低学习率，切换为近壁加权 + 壁面梯度损失，精调近壁区精度。

**优势**：避免一开始就偏向壁面导致内部场崩坏，先建立整体场的合理先验，再定向优化端到端关键区域。

**实验编号**：`A-Opt-W04`

**改动量**：代码约 30~50 行（`trainer.py` 中增加阶段切换逻辑）

#### 5.5.5 PW4：OSI 敏感区域加权

**研究问题**：在损失函数中对分叉区域和高曲率区域（OSI 异常高发区）施加更大权重，是否改善 OSI 恢复质量。

**核心思路**：`pipeline` 阶段已生成 `Curvature` 和分叉区域标签。高曲率节点和分叉附近节点的损失权重乘以 boost 系数。与 PW0 可叠加。

**实验编号**：`A-Opt-W05`

**改动量**：代码约 20 行（在 `wall_oriented_mse_loss` 基础上增加曲率/分叉区权重）

#### 5.5.6 Line W 组合优先级

推荐执行顺序（均基于 **`A-Opt-05` 母版**，seed=1 先做 smoke test）：

1. `A-Opt-W01`：近壁区域加权（实现最简单，归因最清晰）
2. `A-Opt-W02`：壁面梯度监督（中等复杂度，对 WSS 最直接）
3. `A-Opt-W03`：直接 WSS 监督（需要数据管线改动，但收益预期最大）
4. `A-Opt-W04`：两阶段训练（可与 W01/W02 叠加）
5. `A-Opt-W05`：OSI 敏感区域加权（可与 W01 叠加）

> **评估要点**：Line W 的所有实验，除标准区域评估外，**必须额外运行 WSS 后处理对比**：
>
> ```bash
> python -m training.scripts.predict_field --run-dir <outputs/field/xxx>
> python -m training.scripts.compute_wss --run-dir <outputs/field/xxx>
> python -m training.scripts.compare_wss_cfd_vs_ai --run-dir <outputs/field/xxx>
> ```
>
> 重点对比壁面 `WSS_RMSE`、`WSS_R2`、`TAWSS` 病例级 Pearson、`OSI` 区域级一致性。

#### 5.5.7 Line W 推进门槛

- **从 `A-Opt-W01` 进入 `A-Opt-W02` 的条件**：
  - 近壁区域 `RMSE_|v|` 相对基座下降；
  - WSS 点级 `R²` 或 `RMSE` 有改善（若 WSS 后处理已就绪）；
  - 全局 `RMSE_|v|` 不出现显著崩坏。

- **从 `A-Opt-W02` 进入 `A-Opt-W03` 的条件**：
  - 壁面梯度损失确实能改善 WSS 质量；
  - 但改善幅度不足以满足任务 B 通过标准时，再尝试更直接的 WSS 监督。

- **停止 Line W 并转向任务 B 精调的条件**：
  - WSS 点级 `R²` 与 CFD 真值达到较高一致性（> 0.8）；
  - TAWSS/OSI 病例级排序趋势与 CFD 基本一致。


---

## 6. 近期给老师汇报的建议路径

如果目标是尽快拿出更好的效果，建议按下面顺序推进：

### 第 1 周：快速结果轮

只围绕当前 `A-Main-01` 做 5 组小实验（均 seed=1）：

| # | Exp ID | 变化项 |
|---|---|---|
| 1 | A-Main-01 | 当前基线（已完成，直接比较） |
| 2 | A-Opt-01 | `target_weights=[2,2,2,0.5]` |
| 3 | A-Opt-02 | `+ LayerNorm`（Pre-Norm） |
| 4 | A-Opt-03 | `target_weights + LayerNorm`（✅ 2026-03-28，三 seed） |
| 5 | A-Opt-03w | `target_weights + LayerNorm + warmup=5`（✅ 2026-03-28；未优于 03） |

**具体执行流程：**

```bash
# 1. 准备配置目录
mkdir -p training/configs/field/generated/optimization

# 2. 以 A-Main-01 的 config 为模板生成各组 JSON（手动修改对应字段）

# 3. 对每组先做 smoke test（只跑 2 epoch 验证无报错）
python -m training.scripts.train_field \
  --config training/configs/field/generated/optimization/A-Opt-01_seed1.json
# 确认无报错后 Ctrl+C 中断，再正式跑

# 4. 正式训练（建议用 nohup 或 tmux 后台运行）
nohup python -m training.scripts.train_field \
  --config training/configs/field/generated/optimization/A-Opt-01_seed1.json \
  > logs/A-Opt-01_seed1.log 2>&1 &

# 5. 训练完成后运行预测和区域评估
python -m training.scripts.predict_field --run-dir outputs/field/<run_dir>
python -m training.scripts.plot_taskA_regional_bar --run-dir outputs/field/<run_dir>

# 6. 对比结果：重点看 interior.rmse_vel_mag 和 R²_u/v/w
```

只要其中有 1 组相对 baseline 在内部区域和全局 `RMSE_|v|` 上都有稳定改善，就足够形成一次阶段汇报。

### 第 2 周：确认主线候选

在第 1 周最优配置基础上，开始试容量扩展和区域加权：

1. `A-Opt-04`：`hidden_dim = 256`（基于第 1 周最优配置）（**✅ 2026-03-29：内部 `RMSE_|v|` 未继续改善**）
2. `A-Opt-05`：`hidden_dim = 256, num_layers = 4`（**✅ 2026-03-29；✅ 2026-03-31：定为消融/Line G/W 母版**；均值略优于 03，近壁等区域略优；方差与成本更高）
3. `A-Opt-07`：区域加权 `interior_boost = 3.0`（需先完成代码改动）

**（2026-03-29）** 已观测到 **加宽单步伤害内部速度、加深部分回补** 的模式——**应停止在 6L 上继续堆**，转向 **区域加权 / `A-Abl-02` / Line G / 多尺度立项**。

### 第 3 周：补最小必要解释实验

围绕“最佳优化版主线”补做最少量解释实验：

- 是否主要收益来自损失权重；
- 是否主要收益来自 LayerNorm；
- 是否主要收益来自容量扩大。

这一步不等于把完整消融全部做完，而是优先补最能支撑汇报和论文叙事的那几组。

---

## 7. 评估与停机标准

近期所有优化实验，不建议只盯一个全局指标。建议统一看四类指标：

### 7.1 主指标

- `RMSE_|v|`
- 内部区域 `RMSE_|v|`

### 7.2 辅指标

- `RMSE_p`
- `R2_p`

### 7.3 关键解释指标

- `R²_u`
- `R²_v`
- `R²_w`

### 7.4 效率约束

- 单次训练耗时
- 推理时间
- 峰值显存

若一个改动满足下面三条，可判定为“值得继续”：

1. 全局 `RMSE_|v|` 下降；
2. 内部区域 `RMSE_|v|` 同步下降；
3. 速度分量 `R²` 至少有一项明显改善，而不是仅压力继续变好。

若只出现下面情况，则不建议进入主线：

- 压力更好了，但内部速度几乎不动；
- 壁面进一步变好，但内部区无改善；
- 计算成本明显上升，但主指标改善很小。

### 7.5 端到端壁面指标（Line W 专用，2026-03-25 新增）

Line W 实验除上述四类指标外，必须额外评估以下壁面衍生指标：

- 近壁区域 `RMSE_|v|`
- `WSS` 点级 `RMSE` 和 `R²`（与 CFD 对比）
- `TAWSS` 病例级 Pearson / Spearman 相关
- `OSI` 区域级一致性
- `RRT` 病例级排序一致性

**Line W "值得继续"的判断标准**：

1. 近壁区域 `RMSE_|v|` 下降（这是 WSS 梯度质量的前置指标）；
2. `WSS` 点级或区域级指标改善；
3. 全局 `RMSE_|v|` 不出现显著崩坏（可接受轻微上升，因为内部点权重降低）。

**Line W "可进入任务 B"的标准**：

1. `WSS` 点级 `R²` > 0.7（初始目标），理想 > 0.85；
2. `TAWSS` 病例级 Pearson 相关 > 0.8；
3. `OSI` 高低排序趋势与 CFD 基本一致。

> **重要**：Line W 的评估需要 WSS 后处理管线就绪。建议在 Line W 启动前，先完成任务 B 的 WSS 计算脚本开发，至少能对 `A-Main-01` 的预测结果算出 WSS 与 CFD 的基线对比。

---

## 8. 当前建议结论

### 8.1 关于“是否继续加深”

- **可以试，但不建议盲目继续堆深度。**
- 当前更合理的路径是：先补 `LayerNorm`，再试 `256` 宽度和 `4` 层，最后才考虑更深。
- 如果这些改动后内部区仍明显卡住，就说明应把主要精力转向多尺度，而不是继续在线数上硬推。

### 8.2 关于“先做消融还是优化”

- **近期优先做一轮低耦合优化，争取先拿到更好的效果。**
- **优化出主线候选后，再补最小必要消融解释原因。**
- 不建议现在直接做完整大消融，也不建议直接跳到多尺度大改。

### 8.3 最推荐的近期执行顺序

1. `A-Opt-01`：`target_weights [2,2,2,0.5]`
2. `A-Opt-02`：`LayerNorm`（Pre-Norm）
3. `A-Opt-03`：`target_weights + LayerNorm`（✅ 2026-03-28）
4. `A-Opt-03w`：叠加 `warmup=5`（✅ 2026-03-28；**未更好**；**不作为母版**）
5. `A-Opt-04`：`hidden_dim = 256`（✅ 2026-03-29；**`interior.rmse_vel_mag` 较 `A-Opt-03` 变差**）
6. `A-Opt-05`：`hidden_dim = 256, num_layers = 4`（✅ 2026-03-29；**相对 03 均值略优、近壁略优；方差与成本更高；✅ 2026-03-31 定为后续母版**）
7. **下一优先**：`A-Abl-02`：以 **`A-Opt-05` 为母版** 补齐显式几何分量消融（或与 **`A-Opt-07`** 二选一）
8. `A-Opt-G01`：分叉距离/拓扑标记
9. `A-Opt-G02`：半径变化率/截面积变化率
10. 视区域误差走向尝试 `A-Opt-G03` 或 `A-Opt-G04`
11. 若单尺度收益趋缓，再立项多尺度结构（A-Opt-10）；**默认暂缓 `A-Opt-06`（6L）**

### 8.4 关于壁面导向优化与端到端链路（2026-03-25 补充）

- **从端到端视角看，近壁区精度比内部精度更重要。** 因为 WSS/OSI 等壁面指标直接依赖壁面速度梯度，而梯度质量取决于近壁区的速度预测精度，不是内部主流区。
- **Line A（内部精度优化）和 Line W（壁面导向优化）应并行推进，互不替代。** Line A 服务于任务 A 论文叙事（"我们的模型整体精度更高"），Line W 服务于端到端链路质量（"我们的模型产生的 WSS/OSI 更准确"）。
- **建议在任务 A 优化循环中就引入 WSS 质量作为前置评估**，不要等到任务 B 才发现壁面梯度质量不够。
- **Line W 的推荐执行顺序**：`A-Opt-W01`（近壁加权）→ `A-Opt-W02`（壁面梯度监督）→ `A-Opt-W03`（直接 WSS 监督）→ `A-Opt-W04`（两阶段训练）→ `A-Opt-W05`（OSI 敏感区加权）。

---

## 9. 建议新增的优化实验编号

为避免和现有 baseline / ablation 混淆，建议单独开一条"优化线"：

| Exp ID | 研究问题 | 唯一变化项 | 改动类型 | 建议优先级 |
| --- | --- | --- | --- | --- |
| A-Opt-01 | 速度权重是否能改善内部流场 | `target_weights=[2,2,2,0.5]` | 仅配置 | P0 |
| A-Opt-02 | LayerNorm 是否提升 Transformer 表达 | Pre-Norm 残差块 | 代码 ~10行 | P0 |
| A-Opt-03 | 损失重加权与 LayerNorm 是否互补 | `A-Opt-01 + A-Opt-02` | 仅配置 | P0 |
| A-Opt-03w | 叠加 warmup 是否进一步稳定训练 | `+ warmup_epochs=5` | 仅配置 | P0 |
| A-Opt-04 | 容量扩大是否继续有效 | `hidden_dim=256` | 仅配置 | P1（✅ 2026-03-29；**内部速度未更好**） |
| A-Opt-05 | 适度加深是否继续有效 | `hidden_dim=256, num_layers=4` | 仅配置 | P1（✅ 2026-03-29；**相对 03 边际、方差大**） |
| A-Opt-06 | 单尺度进一步加深是否还值得 | `hidden_dim=256, num_layers=6` | 仅配置 | P1（**默认暂缓**） |
| A-Opt-07 | 内部点区域加权是否改善瓶颈 | `interior_boost=3.0` | 代码 ~30行 | P1 |
| A-Opt-08 | 梯度累积是否稳定训练 | `accumulate_grad_batches=4` | 仅配置 | P2 |
| A-Opt-09 | 连续性物理约束是否有效 | `continuity_weight=0.01` | 仅配置 | P2 |
| A-Opt-10 | 多尺度结构是否带来本质提升 | graph U-Net | 代码 ~250行 | P3 |

建议先把 `A-Opt-01 ~ A-Opt-07` 作为近期主线，其余根据结果再决定是否进入正式执行。

### 显式几何增强线 Line G（2026-03-26 新增）

| Exp ID | 研究问题 | 唯一变化项 | 改动类型 | 建议优先级 |
| --- | --- | --- | --- | --- |
| A-Opt-G01 | 显式分叉拓扑先验是否改善复杂转折区建模 | `distance_to_bifurcation + branch_flag` | 预处理 + 配置 | PG0 |
| A-Opt-G02 | 局部尺度变化信息是否优于单纯半径值 | `radius_change_rate / area_change_rate` | 预处理 + 配置 | PG1 |
| A-Opt-G03 | 扭率能否补足曲率缺失的三维弯扭信息 | `torsion` | 预处理 + 配置 | PG1 |
| A-Opt-G04 | 显式壁面距离是否改善近壁速度剖面学习 | `distance_to_wall / normalized_wall_distance` | 预处理 + 配置 | PG1 |
| A-Opt-G05 | 中心线方向变化率是否提升转折区表达 | `d_tangent/ds` 或等价方向变化量 | 预处理 + 配置 | PG2 |

**Line G 统一要求：**

- 每次只新增 1 类特征，其他配置固定为 **`A-Opt-05` 母版**（与 **`A-Main-01`** 的并列叙述另说，不混为控制变量）；
- 每组先跑 `seed=1`，信号明确后再补 `seed=2,3`；
- 必须额外报告 `near_wall / bifurcation / high_curvature`；
- 如果增益只出现在训练集，不进入主线。

### 壁面导向优化线 Line W（2026-03-25 新增）

| Exp ID | 研究问题 | 唯一变化项 | 改动类型 | 建议优先级 |
| --- | --- | --- | --- | --- |
| A-Opt-W01 | 近壁区域加权是否改善 WSS 梯度质量 | `near_wall_boost=3.0, interior_weight=0.5` | 代码 ~40行 | PW0 |
| A-Opt-W02 | 壁面法向梯度监督是否提升 WSS 精度 | `wall_grad_weight=0.01` | 代码 ~60行 | PW1 |
| A-Opt-W03 | 直接 WSS 监督是否最大化端到端质量 | `wss_loss_weight=0.1` | 代码 ~120行 + 数据管线 | PW2 |
| A-Opt-W04 | 两阶段训练是否优于一开始就加权 | 阶段1:均匀MSE → 阶段2:壁面精调 | 代码 ~50行 | PW3 |
| A-Opt-W05 | OSI 敏感区加权是否改善 OSI 恢复 | 分叉/高曲率区 boost | 代码 ~20行 | PW4 |

建议 `A-Opt-W01 ~ W02` 作为 Line W 近期主线，`W03` 视数据管线就绪情况决定是否进入。

> **注意**：Line W 与 Line A（A-Opt-01~10）并行推进、各自归因。**（2026-03-31）** 两线默认 **同以 `A-Opt-05` 为母版**；**`A-Opt-03`** 仅作低开销对照。

---

## 10. 需要的代码改动清单

以下汇总所有优化实验涉及的代码修改。**仅配置改动的实验不在此列**。

### 10.1 P0-2 LayerNorm（A-Opt-02 起生效）

**文件：** `training/core/models.py`
**范围：** `FieldTransformer.__init__` 和 `FieldTransformer.forward`
**改动量：** ~10 行

- `__init__`：在 `self.post_layers` 后新增 `self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])`
- `forward`：循环签名从 `for conv, linear in zip(...)` 改为 `for conv, linear, norm in zip(...)`，循环体首行加 `x = norm(x)`

### 10.2 P1-2 区域加权损失（A-Opt-07 起生效）

**文件：** `training/core/losses.py`、`training/core/config.py`
**改动量：** ~30 行

1. `losses.py`：新增 `region_weighted_mse_loss` 函数
2. `losses.py`：`NullPhysicsLoss.build_loss` 和 `PhysicsConstraintLoss.build_loss` 中把 `weighted_mse_loss` 替换为条件调用
3. `config.py`：`OptimConfig` 新增 `interior_loss_boost: float = 1.0` 字段
4. `train_field.py`：把 `interior_loss_boost` 传入 `FieldTrainer`

### 10.3 P3 图 U-Net（A-Opt-10 起生效）

**文件：** 新建 `training/core/models_unet.py`
**改动量：** ~250 行

1. 实现 `FieldGraphUNet` 类（编码器-池化-解码器-skip connection）
2. 在 `training/core/models.py` 的 `MODEL_REGISTRY` 中注册 `"graph_unet": FieldGraphUNet`
3. `config.py` 的 `validate()` 中补充合法模型名

> 注意：所有代码改动都应保持向后兼容——现有 baseline 的配置文件和 checkpoint 不应受到影响。新增字段使用默认值退化为原有行为。

### 10.4 PW0 近壁区域加权损失（A-Opt-W01 起生效，2026-03-25 新增）

**文件：** `training/core/losses.py`、`training/core/config.py`、`pipeline/preprocess.py`（可选）
**改动量：** ~40 行

1. `losses.py`：新增 `wall_oriented_mse_loss` 函数（按壁面/近壁/内部三区域分配权重）
2. `config.py`：`OptimConfig` 新增 `wall_boost: float = 1.0`、`near_wall_boost: float = 1.0`、`interior_weight: float = 1.0` 字段
3. `losses.py`：`NullPhysicsLoss.build_loss` 和 `PhysicsConstraintLoss.build_loss` 中条件调用
4. （可选）`pipeline/preprocess.py`：为每个内部节点计算到壁面的距离，生成 `near_wall_mask` 并存入 `data`

**近壁区域标记方案**：
- 方案 A（推荐）：在 pipeline 阶段预计算每个内部节点到最近壁面节点的欧氏距离，距离 < 阈值的标记为近壁区。阈值建议取训练集壁面距离分布的 25th percentile。
- 方案 B（简化）：利用 `edge_index` 找到壁面节点的 1-hop 或 2-hop 邻居中的内部节点，标记为近壁区。不需要额外预处理。

### 10.5 PW1 壁面法向速度梯度监督（A-Opt-W02 起生效，2026-03-25 新增）

**文件：** `training/core/losses.py`、`training/core/config.py`
**改动量：** ~60 行

1. `losses.py`：新增 `wall_gradient_loss` 函数
   - 输入：预测速度、壁面节点 mask、壁面法向量（或使用 edge_index 有限差分近似）
   - 计算壁面处速度法向梯度，与 CFD 真值的对应梯度做 MSE
2. `config.py`：新增 `wall_grad_weight: float = 0.0` 字段（默认关闭）
3. `losses.py`：在 `build_loss` 中当 `wall_grad_weight > 0` 时计算并叠加

**壁面法向获取方案**：
- 方案 A：从原始网格文件提取壁面法向量，存入 `data` 的新属性
- 方案 B：使用 `Tangent` 特征推导法向（血管表面法向 ≈ 与切向正交的径向方向）
- 方案 C（最简）：用 edge_index 连接的最近内部邻居做有限差分，不显式使用法向量

### 10.6 PW2 直接 WSS 监督（A-Opt-W03 起生效，2026-03-25 新增）

**文件：** `training/core/losses.py`、`training/core/config.py`、`pipeline/` WSS 预计算脚本
**改动量：** ~120 行

1. `pipeline/` 新增 WSS 预计算：从 CFD 原始数据中提取每个壁面节点每时间步的 WSS 标量，存入 `.pt` 图数据
2. `losses.py`：新增 `wss_supervision_loss` 函数（可微分 WSS 计算 + MSE）
3. `config.py`：新增 `wss_loss_weight: float = 0.0` 字段

> **依赖**：需要 CFD 后处理中的壁面法向量和 WSS 真值数据。建议与任务 B 的 WSS 计算脚本同步开发，一次性建立统一的 WSS 计算管线。
