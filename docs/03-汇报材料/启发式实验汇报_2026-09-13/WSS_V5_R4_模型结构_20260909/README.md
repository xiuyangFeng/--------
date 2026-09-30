# V5·R4 模型结构与输入表

对应工作簿「实验矩阵总览」207 行、「教师汇报视图」182 行的 **★ V5·R4 R1 + V5 十维几何特征 s1234**。

## 图件

- [Mermaid 完整版](V5_R4_模型结构_Mermaid.md)：总图、SA2 Transformer、PointNeXt-R 残差块三张图；另附同名独立 `.mmd` 源文件，可直接复制编辑。
- `01_整体网络结构.png`：适合汇报主页面的完整结构。
- `02_SA与Transformer展开.png`：SA、LocalGeoPE、四头 Transformer 与实际模块顺序。
- `03_残差解码与回归展开.png`：PointNeXt-R 双残差、DropPath、FP 和监督目标。
- 三页图件 PNG 为 4400×2475。
- `V5_R4_输入特征与通道表.xlsx`：输入定义、真实训练统计、逐层通道三张表。原实验工作簿未修改。
- `V5_R4_输入特征表.tsv/.csv`：用于复制到 Excel / Word；本文表格也可直接复制。

## 可直接用于汇报的说明

模型以 5000 个壁面 support 点的 17 维几何特征为输入，经 Stem 映射到 32 维。三级编码器的点数依次为 125、125、32，通道依次为 64、128、256。每级局部聚合均加入 LocalGeoPE；第二级在每个邻域内加入四头 Transformer。第一、二级聚合后各接一个 PointNeXt-R 倒置瓶颈残差块。解码器通过 3-NN 插值、编码器跳连拼接和 MLP，逐级恢复到 5000×32 的 support 特征，再读取或插值到 query 点，经 32→64→1 回归头预测峰值 WSS 的 log_z 值，最后还原为 Pa。

原有 7D = xyz 3D + 归一化弧长 1D + 半径 1D + 曲率 1D + 对数半径 1D。
新增 10D = 管道坐标 3D + 半径坡度 1D + 距离与端区 3D + 法向 3D。

## 输入表（严格按配置顺序）

| 输入索引（0起） | 分组 | 配置字段名 | 维数 | 几何含义 | 标准化前单位/范围 | 进入网络前处理 |
| --- | --- | --- | --- | --- | --- | --- |
| 0–2 | 原有7D | x, y, z | 3 | 对齐后的壁面三维坐标 | 无量纲，[-1,1] | 分叉为原点；+Z朝入口、+X朝左髂总；除以逐例 max-abs 尺度；不做 z-score |
| 3 | 原有7D | abscissa_norm | 1 | 从入口根节点沿中心线树到映射点的累计弧长 / 本例壁面映射弧长最大值 | 无量纲，[0,1] | train138 z-score；同时进入 LocalGeoPE 的差分属性 |
| 4 | 原有7D | local_radius | 1 | 映射中心线位置的平滑局部半径 R | mm | train138 z-score；同时进入 LocalGeoPE 的差分属性 |
| 5 | 原有7D | curvature | 1 | 映射中心线位置的曲率 κ | mm⁻¹ | signed_log1p → 训练集绝对值 P99 裁剪 → train138 z-score；同时用于 LocalGeoPE |
| 6 | 原有7D | log_local_radius | 1 | 局部半径的自然对数 ln(max(R, 10⁻⁶))，R 取 mm 数值 | 对数特征 | 先取 ln，再用 train138 z-score |
| 7 | V5新增10D | rho | 1 | 到中心线的横向径向距离 r⊥ / 局部半径 R | 无量纲 | train138 z-score；不强制限制在 [0,1] |
| 8 | V5新增10D | theta_sin | 1 | 中心线局部法平面内周向角的 sinθ | 无量纲，[-1,1] | 从 atlas 局部坐标架计算 θ，再取 sin，最后 train138 z-score |
| 9 | V5新增10D | theta_cos | 1 | 中心线局部法平面内周向角的 cosθ | 无量纲，[-1,1] | 从 atlas 局部坐标架计算 θ，再取 cos，最后 train138 z-score |
| 10 | V5新增10D | dr_ds | 1 | 局部半径沿中心线弧长的变化率 dR/ds | mm/mm，无量纲 | train138 z-score |
| 11 | V5新增10D | dist_to_junction_mm | 1 | 映射中心线点沿中心线树到最近分叉节点的距离 | mm | train138 z-score；沿树距离 |
| 12 | V5新增10D | dist_to_endpoint_mm | 1 | 映射中心线点沿中心线树到最近开口端点的距离（入口或出口） | mm | train138 z-score；沿树距离 |
| 13 | V5新增10D | end_zone | 1 | 中心线端区特征保持/修正区域标记；含开口端点与子分支起始端 | 原始值 0/1 | 该实验仍做 train138 z-score；不是单纯的入口/出口类别 |
| 14–16 | V5新增10D | nx_aligned, ny_aligned, nz_aligned | 3 | 壁面点云 PCA 估计的外向法向，旋转到与 xyz 相同的解剖坐标架 | 原始单位法向的三个分量 | 三个分量分别使用 train138 z-score；标准化后不再是单位向量 |


xyz 另作为 `pos` 传入 FPS、邻域搜索与插值；其已包含在 17 维特征中，不另计为 20 维。其余 14 个通道均用 train138 统计进行 z-score，包括角度编码、end_zone 与法向分量。统计来自全部训练病例的壁面顶点；推理使用冻结统计。原始范围是标准化前的范围。

ρ 中的 r⊥ 为点到最近 atlas 样点切线的横向距离；θ 在该样点的局部法平面架内定义。两项距离继承该映射样点的沿中心线树距离。法向由壁面点云 PCA 估计并定向，不是中心线局部坐标架的法向。

## 逐层通道

| 阶段 | 输出尺寸（每例） | 通道变换 | 模块/说明 |
| --- | --- | --- | --- |
| 输入 | 5000 × 17 | 17 | pos 同时单独作为几何坐标 |
| Stem | 5000 × 32 | 17 → 32 → 32 | 逐点 MLP |
| SA1 | 125 × 64 | 边 MLP 35→64→64；GeoPE 7→64→64 | FPS125；knn_cover64；max；InvRes×1 |
| SA1 InvRes | 125 × 64 | 局部 67→64→64；逐点 64→128→64 | ball .05 / ≤64；DropPath 0 |
| SA2 | 125 × 128 | 边 MLP 67→128→128；GeoPE 7→128→128 | FPS125；ball .10 / ≤16；L-SA2→max；InvRes×1 |
| SA2 Transformer | 每邻域 K × 128 → 128 | 4头×32；FFN 128→256→128 | Pre-LN；两处缩放残差；max后为125×128 |
| SA2 InvRes | 125 × 128 | 局部 131→128→128；逐点 128→256→128 | ball .10 / ≤16；DropPath .10 |
| SA3 | 32 × 256 | 边 MLP 131→256→256；GeoPE 7→256→256 | FPS32；ball .20 / ≤16；max；残差块0 |
| FP3 | 125 × 128 | 256+128=384 → 128 → 128 | 3-NN 特征插值 + SA2 skip |
| FP2 | 125 × 64 | 128+64=192 → 64 → 64 | 3-NN 特征插值 + SA1 skip |
| FP1 | 5000 × 32 | 64+32=96 → 32 → 32 | 3-NN 特征插值 + Stem skip |
| Query | Nq × 32 | 32 → 32 | SAME直接读取；独立query固定3-NN插值 |
| 回归头 | Nq × 1 | 32 → 64 → 1 | ReLU；最后线性输出log_z |
| 物理输出 | Nq × 1 | 1 → 1 | exp(σtrain ẑ + μtrain) − eps，单位Pa |


## 配置解释与实现要点

1. 配置字段为 `model.name=pointnetpp`，实际类是 `PointNetPlusPlusRegressor`，启用了 PointNeXt-R 残差扩展。汇报称 **PointNeXt-R + LocalGeoPE + L-SA2**，不能用独立 `pointnext.py` 中的默认模型代替。
2. `sa_center_counts=[125,125,32]` 优先于 ratios，因此点数为 5000→125→125→32。SA2 为同点数重采样与邻域特征提升。FP2 同样保持 125 点数。
3. SA1 是 KNN-64 加未覆盖点最近中心补边，邻域可能超过 64；SA2/3 分别是半径 0.10/0.20 的 ball-query，最多 16 邻居。所有 radius 均在逐例归一化坐标空间，不是 mm。SA1 的 0.05 仍用于 GeoPE 缩放与该级 InvRes 的 ball 邻域。
4. GeoPE 的 7 维边输入为 `[Δp/r (3), ||Δp/r|| (1), Δs̃, ΔR̃, Δκ̃ (3)]`。最后一层零初始化，可学习几何增量加到 SA 边特征；新增 10D 经 Stem 主通路输入，未额外并入 GeoPE 属性索引。
5. Transformer 位于 SA2 的逐邻居 token MLP + GeoPE 之后、max pooling 之前；它有自己的 attention/FFN 缩放残差。其后再接独立 `LocalInvResBlock`，两种残差模块应区分。注意力只在同一个中心的邻域中计算，不跨病例、也不是 32 点瓶颈全局注意力。
6. InvRes 采用局部聚合更新和 C→2C→C 逐点更新，两个 gamma 初始 0.001。两处相加后不额外激活。DropPath 在两个块间线性分配为 0 和 0.10，按病例抽取并在该块两分支共享。Transformer 的 gamma 初始也为 0.001，但其分支不使用此 DropPath。
7. FP3/2/1 的拼接通道分别为 384、192、96。3-NN 使用 PyG 的归一化逆平方距离权重；对 query 插值的是 32D 特征，之后才回归标量。SAME 点直接读取编码特征；独立 query 只使用坐标定位插值，本配置不将 query 自身的 17D 特征额外送入回归头。
8. 单输出头；H2 是实验损失配方标签，不能解释为双输出头。损失为 `mean((z-ẑ)^2) + .2 mean(max(.9(z-ẑ), -.1(z-ẑ)))`，在所有监督点上计算，q90 不表示只训练真值最高 10% 的点。
9. 标签是峰值帧 1162 的 CFD 壁面 WSS 标量。`z=[ln(WSS+1e-6)-0.6408381995]/1.2999484463`；输出恢复为 `exp(1.2999484463*ẑ+0.6408381995)-1e-6 Pa`。原实验没有 PDE/PINN 损失、速度/压力联合输出或额外 q90 输出头。
10. R4 中 bottleneck Transformer、coarse global attention、EdgeConv、SEP/QAD/local-attn query decoder、局部壁面分支与病例幅值头未启用。图中只展示实际启用模块。

## 训练与验证

train138/test34；seed1234；400 epoch；batch8；AdamW，lr=1e-3，warmup10 后 cosine 至1e-5；weight decay=1e-4；grad clip=1；AMP。每 epoch 随机5000 support=query（SAME），无旋转增强。按训练损失选 best；推理固定5000 support（seed1234）并对全壁面 query 分块（16384）预测。

已将 `ckpt_best.pt` 严格加载到配置构建的模型，全部键匹配；参数量 **573,351**。真实病例抽取5000点完成 CPU 前向核对，并检查 SAME/独立query 输出有限。源文件哈希、实际输出尺寸与检查结果见 `verification.json`。这些检查仅验证本次图件对应结构，不重新训练或改变实验指标。

## 依据

- `training_wss_min/configs/v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234.json`
- `training_wss_min/baseline_models.py`：`PointNetPlusPlusRegressor`、`PointNetSetAbstraction`、`LocalNeighborhoodTransformer`、`LocalInvResBlock`、`FeaturePropagation`
- `training_wss_min/dataset.py`：特征组装、train统计、归一化/反归一化
- `training_wss_min/objectives.py`：MSE + pinball
- `wss_v5/views/wss_min_view.py`、`wss_v5/centerline_features.py`、`wss_pinn/v4/centerline_atlas.py`：几何定义
- 该 run 的 `config.json`、`feature_stats.json`、`wss_global_stats.json` 与 `ckpt_best.pt`
- `docs/02-推进与变更/00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md` §4
- 用户提供工作簿中的 R4 s1234 对应行
