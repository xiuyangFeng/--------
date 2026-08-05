# velocity→WSS V1–V4 实验总跟踪

> 状态日期：2026-08-04
>
> 适用范围：`wss_mri_calculator` 的 Fluent 非结构化点云 velocity→WSS 路线
>
> 稳定使用入口：[CFD 适配说明](../README_CFD_ADAPTATION.md)

本文件是 V1–V4 路线状态、报告口径和实验入口的唯一总览。算法细节由各版本
`README.md` 负责，完整数值由各版本 `RESULTS.md` 和机器可读 JSON 负责，不在本页
重复维护长篇过程记录。

## 当前结论

面向用户当前确定的目标——**病例总体精度和平均高 WSS 精度**——当前推荐结果模型
是 **Profile-Secant V3**。它以 Surface-MLS V4 final 为基底，增加推理期可获得的
近壁剖面与切线梯度诊断特征，并使用 train138 拟合的冻结高尾校准模型。

**Surface-MLS V4 final 现在统一定位为冻结基线**：保留其模型、manifest、结果和
复现入口，用于消融对照和方法演进，不再称为当前数值最优模型。

在统一的 test35 × 1200 evaluation-only 口径下：

| 方法 | test35 raw R² mean | scaled R² mean | MAE (Pa) | high-WSS NRMSE |
| --- | ---: | ---: | ---: | ---: |
| V1 adaptive-CV | 0.85862 | 0.92544 | 1.07544 | 0.31295 |
| V2 multiscale | 0.86750 | 0.92350 | 1.03720 | 0.30150 |
| V3 normal multiscale | 0.89570 | 0.92973 | 0.88774 | 0.27287 |
| V4 physics | 0.90788 | 0.93612 | 0.82646 | 0.25718 |
| V4 final（冻结基线） | 0.95607 | 0.95844 | 0.53018 | 0.17017 |
| **Profile-Secant V3（当前推荐）** | **0.96170** | **0.96537** | **0.48032** | **0.15834** |

V4 final 的 raw R² p05 为 `0.92391`，最差病例为 `0.90025`；相对 V3 和
V4 physics 均为 `35/35` 病例获胜。

Profile-Secant V3 在 test35 将整体 R² 提高到 `0.9617`、pooled high-WSS R²
提高到 `0.9415`、逐病例 high-WSS R² 均值提高到 `0.7721`，high-WSS NRMSE
降至 `0.1583`。
严格逐病例目标仍未达到：6/35 病例 high-WSS R² `≥0.90`，22/35 病例峰值低估
`≤10%`，两项同时满足仅 4/35。当前推荐口径因此是：**Profile-Secant V3 用于主结果、
可视化和后续病例总体/平均高 WSS 分析；V4 final 用于冻结复现与消融对照。**

以上方法横向表使用统一的 test35 × 1200 协议。2026-08-04 已追加 Profile-Secant V3
的 `sample_count=0` 全壁面推理：35 例共 `1,328,017` 个壁面节点，病例 overall R²
均值=`0.9604`、pooled overall R²=`0.9679`、pooled high-WSS R²=`0.9440`、逐病例
high-WSS R² 均值=`0.7735`、high-WSS NRMSE 均值=`0.1623`。全壁面捕获到更极端
真实峰值，平均峰值低估为 `15.81%`，说明 sampled 的 `9.31%` 不能替代全量峰值
结论。完整同点散点、采样/穿模图和 VTP 见
[Profile-Secant V3 全壁面结果](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/README.md)。

## 路线状态矩阵

| 路线 | 核心假设 | 推理时是否依赖真值 | 选择/拟合数据 | 当前状态 | 当前用途 |
| --- | --- | --- | --- | --- | --- |
| [V1 adaptive-CV](pointcloud_adaptive_v1/) | 用速度剖面 LOOCV 逐点选择邻域尺度，替代固定 K | 否 | train138 选 K/tolerance | frozen / historical baseline | 基础物理前向与方向基线 |
| [V2 multiscale](pointcloud_multiscale_v2/) | 根据跨尺度收敛趋势进行有界正向修正 | 否 | train138 冻结门控与上限 | frozen / historical | 零真值局部多尺度证据 |
| [V3 normal multiscale](pointcloud_normal_multiscale_v3/) | 沿物理法向建立插值站点，并加入壁面归属和局部 Re 深度先验 | 否 | train138 消融与冻结 | frozen / historical comparator | 法向射线路线的最终版本 |
| [V4 physics](pointcloud_surface_mls_v4/) | 直接拟合各自最近壁面锚点下的真实内部单元，满足曲面无滑移 | 否 | train36 cap 筛选 + train138 审计 | frozen / final physics core | 无监督物理算子 |
| [V4 final](pointcloud_surface_mls_v4/) | 从推理期物理诊断量学习有限分辨率残差，并用病例尺度锚定和边界保护 | 训练时使用 train138 WSS；推理时不使用 | train138 grouped OOF、development73→holdout65、全量 train138 冻结 | frozen baseline | 复现、消融对照与方法演进基线 |
| [Profile-Secant V3](pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md) | 用近壁剖面与切线梯度诊断改善高 WSS 尾部 | 训练时使用 train138 WSS；推理时不使用 | train138 grouped OOF、development73→holdout65、冻结后 test35 | **selected / recommended result model** | 当前主结果、可视化和总体/平均高 WSS 分析 |

## V4 的证据层级

| 证据 | raw R² mean | scaled R² mean | 作用 |
| --- | ---: | ---: | --- |
| train138 grouped 5-fold OOF | 0.95493 | 0.95836 | 排除点级随机划分泄漏 |
| 固定 development73 → holdout65 | 0.95790 | 0.96338 | 检查固定病例留出泛化 |
| 冻结 train138 模型 → test35 | 0.95607 | 0.95844 | evaluation-only 外部项目基准 |

test35 是项目既有数据集，不应表述为新采集的前瞻性盲测队列。V4 校准模型没有使用
test35 拟合，且冻结模型的重复应用得到逐字节一致的 JSON 和 NPZ。

## 高 WSS 逐病例边界

| test35 指标 | V4 final | Profile-Secant V3 |
| --- | ---: | ---: |
| 整体 R² 均值 | 0.9561 | **0.9617** |
| pooled high-WSS R² | 0.9282 | **0.9415** |
| 逐病例 high-WSS R² 均值 | 0.7403 | **0.7721** |
| high-WSS NRMSE | 0.1702 | **0.1583** |
| 平均峰值低估 | 11.51% | **9.31%** |

该表是统一 test35 × 1200 横向对比。Profile-Secant V3 的全壁面平均峰值低估为
`15.81%`；整体和 high-WSS R² 基本稳定，但 sampled 峰值指标偏乐观。

真实峰值在 holdout65 与 test35 中均全部落在预测 top10% 内，但任意单调 oracle 在
test35 仍有 13/35 病例不能达到 high-WSS R² 0.90。峰值处内部单元深度与病例高 WSS
精度呈稳定负相关。现有输入下，继续做病例缩放、尾部拉伸、局部回归或模型堆叠，
没有证据能实现逐病例保证；需要更靠近壁面的速度样本或更高近壁空间分辨率。

## 版本演进逻辑

```text
fixed K=64
  → V1：速度重建误差选择局部邻域
  → V2：跨尺度零带宽正向修正
  → V3：法向射线插值 + 归属门控 + 局部 Re 深度
  → V4 physics：最近曲面真实单元 + 曲面无滑移 robust MLS
  → V4 final：train-only 冻结诊断残差校准
  → Profile-Secant V3：近壁剖面 + 切线梯度高尾校准（当前推荐结果模型）
```

方向估计在 V1 后已接近饱和，test35 的方向余弦中位数平均约为 `0.9985`。V2–V4
的主要收益来自减少近壁导数幅值衰减，而不是重新学习 WSS 方向。

## 报告口径

1. `raw_r2` 用于报告实际前向输出，不能与逐病例 oracle `scaled_r2` 混写。
2. V1/V2/V3/V4 physics 属于不读取 WSS 真值的物理前向。
3. V1/V2 的 train-only 全局标量和 V4 diagnostic calibrator 都属于监督校准结果，
   必须明确写出训练数据和冻结协议。
4. sampled test35 × 1200 与 full-wall test35 是不同点数协议，只能在协议一致时横向
   比较。
5. 同点 quantitative 指标优先；回插面片和 quicklook 图只用于空间解释与展示。
6. 当前主结果使用 Profile-Secant V3；V4 final 必须标注为冻结基线，不能因名称含
   `final` 而写成当前数值最优模型。

## 模型产物与入口

| 路线 | 方法说明 | 完整结果 | 冻结清单 | 可视化 |
| --- | --- | --- | --- | --- |
| V1 | [README](pointcloud_adaptive_v1/README.md) | [RESULTS](pointcloud_adaptive_v1/RESULTS.md) | [manifest](pointcloud_adaptive_v1/freeze_manifest.json) | 版本目录内图件 |
| V2 | [README](pointcloud_multiscale_v2/README.md) | [RESULTS](pointcloud_multiscale_v2/RESULTS.md) | [manifest](pointcloud_multiscale_v2/freeze_manifest.json) | 版本目录内图件 |
| V3 | [README](pointcloud_normal_multiscale_v3/README.md) | [RESULTS](pointcloud_normal_multiscale_v3/RESULTS.md) | [manifest](pointcloud_normal_multiscale_v3/freeze_manifest.json) | [quicklook](../../outputs/wss_mri_calculator/pointcloud_normal_multiscale_v3/00_quicklook/README.md) |
| V4 | [README](pointcloud_surface_mls_v4/README.md) | [RESULTS](pointcloud_surface_mls_v4/RESULTS.md) | [manifest](pointcloud_surface_mls_v4/freeze_manifest.json) | [quicklook](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/README.md) |
| Profile-Secant V3 | [结果与边界](pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md) | [全壁面 test35 JSON](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/inference/test35_profile_secant_v3_fullwall.json) | [model](pointcloud_surface_mls_v4/calibrator_profile_secant_high_tail_anchor10_v3.joblib) | [全壁面 quicklook + VTP](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/README.md) |

V1–V4 基线统一冻结校验：

```bash
/public/newhome/cy/.conda/envs/GNN/bin/python \
  wss_mri_calculator/experiments/verify_frozen_methods.py
```

## 下一阶段

Profile-Secant V3 是当前推荐结果模型，并在 sampled 与 full-wall 协议下均保持较高
整体和 pooled 高 WSS R²；但全壁面平均峰值低估 `15.81%`，未达到 `≤10%` 目标，
逐病例严格目标也仍受输入分辨率限制。后续优先级不是继续增加射线站点、
多项式阶数或后校准复杂度，而是：

1. 新采集、预先冻结协议的外部验证队列；
2. 网格分辨率、时间步和非峰值相位鲁棒性；
3. 基于壁面拓扑的分支感知或测地距离归属；
4. 归属歧义、深度带宽与校准外推的不确定性估计；
5. 新扫描设备/中心下 cohort 特征的安全性验证；
6. 获取更靠近壁面的首层速度样本，或建立可控近壁网格加密实验，直接验证
   high-WSS 可辨识性随采样深度的变化。

在这些证据出现前，V1–V4 冻结产物和 Profile-Secant V3 当前推荐模型不得通过
post-test tuning 改写。
