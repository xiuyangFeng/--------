# V4 稳态 PINN：梯度冲突与损失曲线汇报

2026-09-12 重绘。仅使用 pre-Centerline-V2 BC/RCR V4 历史 `steady_peak`、seed1234，严格名称为**峰值准稳态**。损失为 PN/PNPP 共8臂；梯度为原图已有的 PN 三个约束臂，不外推 PNPP 梯度。这里的 PN/PNPP 为历史容量配平结构，并非 formal P2V/D2。没有新增训练或测试集评估。

## 可直接给老师的短总结

在这组峰值准稳态 PINN 实验中，最明显的问题是数据拟合与壁面无滑移约束存在持续的优化冲突。PointNet 的 DATA+BC、固定 PDE、EMA PDE 三臂在最后500轮的 data/no-slip 梯度余弦中位数分别为 −0.876、−0.968、−0.969；动量项冲突较弱，连续性项接近零，不能将所有物理约束都概括为强冲突。同时，PointNet 的共同数据损失从 DATA 的0.192增至 +BC 的0.489、固定 PDE 的0.723、EMA 的0.830，PointNet++呈现相近趋势。EMA 相比固定权重降低了无量纲 PDE 残差，但其权重很快接近上限10，未消除壁面约束与数据拟合的竞争。这些现象说明当前网络、采样与损失配置未能兼顾数据和物理约束；尚不能仅凭曲线断定是物理方程不适用，也不能据此否定 PINN。建议优先核查近壁采样与连续场表达、边界和标签的一致性，以及残差尺度和权重设置。

## 三张图的讲解顺序

1. [梯度冲突](fig1_gradient_conflict_steady.png)：按用户反馈改为左侧梯度反向箭头示意，右上 data/no-slip、右下 data/momentum 实测曲线；移除旧C/D热图、公式和解释文字。箭头为参数空间示意，长度与角度不代表某个训练step的实测向量；不绘制合力冒充Adam更新。BC臂未启用的PDE项不画，完整分项统计保留在CSV。
2. [损失曲线](fig2_training_losses_steady.png)：上排为可跨臂比较的共同标准化数据误差，下排为各自训练目标。总损失包含不同的项和权重，不可用总损失高低给各臂排精度。曲线反映训练拟合，不单独证明测试泛化。
3. [物理项动态](fig3_physics_dynamics_steady.png)：按反馈改为四联图：前50轮权重增长、全训练饱和时长、末500轮残差相对固定权重的比值、共同数据MSE对比。两种骨干均在日志epoch24首次达到逐轮平均权重≥9.99，约99.76%的日志epoch保持在此阈值以上。PN为实线、PN++为空心点，重叠不是漏画。残差为缩放后的无量纲均方损失，不是 Pa/m 的物理 RMS。

所有图内标签为英文。各图提供 PNG、PDF、SVG；[三页合订PDF](steady_pinn_figures.pdf)可直接用于汇报。

## 统计与复现口径

- 数据源：[sources.json](sources.json)。8条曲线各自检查 epoch 唯一且连续；DATA终止于9858，PNPP EMA终止于9984，其余终止于9999（日志从0开始）。不延长、补齐或外推较短曲线。
- 数据损失/残差：`epoch_progress.jsonl` 的 `mean_data_total/mean_total/mean_continuity_raw/mean_momentum_group_raw/mean_lambda_phy`。51轮居中移动均值；淡线保留原始逐轮均值。端部用可用窗口，不补零。
- 梯度：复用09-03的 `gradcos_V4-SP-PN-*.csv`，来源是原始 `training_progress.jsonl` 每50 step实际计算的诊断；确认global_step均为50倍数。未重新计算模型梯度。`physics.py:gradient_diagnostics`对全部可训练参数求梯度，使用未加权 data/no-slip/momentum/continuity/inlet 子损失。
- 梯度实线为151个诊断样本居中滑动中位数；色带为每100 epoch箱内的10%–90%分位，描述训练样本波动，不是多seed置信区间。最终统计严格取 `epoch > max_epoch-500`，共500轮；PN梯度各含1380个记录。
- [梯度统计CSV](gradient_summary_last500.csv)和[损失统计CSV](loss_summary_last500.csv)均用未平滑数据计算。损失CSV中关闭项的0为日志占位，不代表实测满足方程。
- 两种骨干EMA约99.76%的已记录epoch满足 `mean_lambda_phy >= 9.99`，直接从权重轨迹计数，不使用旧clamp标志频率推算；epoch均值也不是step命中率。
- PN末500轮EMA/固定连续性损失比约0.064，动量损失比约0.056；这是该训练样本上的残差降低，不能自动转换为测试场精度提升或证明零场塌缩。

复现：在仓库根目录执行 `python3 docs/03-汇报材料/V4汇报/V4_稳态梯度与损失_2026-09-12/plot_steady.py`。

## 动态权重的准确解释

代码真源：`wss_pinn/v4/controller.py`、各run的`resolved_config.json`。这里EMA仅调整PDE权重，BC权重固定为1；它基于损失比，不是梯度方向调节。稳态损失为：

\[
L=L_{data}+\lambda_{BC}\frac{L_{wall}+L_{inlet}}{2}
+\lambda_{PDE}\frac{L_{continuity}+L_{momentum}}{2}.
\]

\(L_{data}\) 为u/v/w/p标准化MSE的均值，\(L_{momentum}\) 为三个分量缩放残差MSE的均值。稳态没有RCR残差项，即使run名称所属路线包含BC/RCR。

每step对detached损失更新：

\[
\bar L_j\leftarrow0.99\bar L_j+0.01L_j,\qquad j\in\{data,PDE\}.
\]

每50 step更新目标和实际权重：

\[
\alpha(e)=0.1+0.9\min(e/200,1),\quad
\lambda_{target}=\operatorname{clip}\left(\alpha(e)\frac{\bar L_{data}}{\max(\bar L_{PDE},10^{-12})},10^{-4},10\right),
\]
\[
\lambda_{PDE}\leftarrow0.9\lambda_{PDE}+0.1\lambda_{target}.
\]

固定PDE臂为1。动态比值detached，不经过该比值反向传播。正标量改变某项梯度大小但不改变其方向，因此EMA损失比本身没有消除负余弦的机制；实际网络轨迹仍可能随权重改变，不能只靠此公式预言泛化。

梯度冲突是当前参数下的一阶局部现象。若\(\Delta\theta=-\eta\nabla L_{data}\)，则\(\Delta L_k\approx-\eta\nabla L_k\cdot\nabla L_{data}\)，负内积意味着这一小步可能增加约束损失。真实Adam更新还涉及各项加权、动量、预条件及裁剪，图中不将两条未加权梯度简单求和冒充真实更新。

本次移除旧图中未经病例剖面测量支持的边界层拟合示意，避免把假设画成实测CFD剖面。“近壁表达不足”和“标签/BC不匹配”仍是待验证机制；非零近壁cell速度与壁面零速度在物理上可以同时成立。

## 2026-09-13 独立损失项补图

[逐项图件与说明](individual_losses/README.md)：u/v/w/p、no-slip、inlet、continuity、三方向momentum共10个独立项，另附momentum group；每项单图区分raw/weighted及PN/PNPP，附11页合订PDF。
