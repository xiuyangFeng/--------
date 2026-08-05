# Surface-MLS V4：最近曲面物理拟合 + 冻结诊断校准

V4 final 是 `pointcloud_normal_multiscale_v3` 之后冻结的基线版本。它保留 V1/V3
已经非常准确的 WSS 方向，用最近曲面鲁棒 MLS 改进幅值估计，再通过一个仅在
训练集上拟合并冻结的残差校准器，修正有限空间分辨率造成的剩余系统误差。
当前主结果在此基线上继续使用 Profile-Secant V3 高尾校准。

> **2026-08-04 推荐口径**：面向病例总体和平均高 WSS 精度，当前推荐结果模型是
> [Profile-Secant V3](PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md)；V4 final 保留为冻结
> 基线和消融对照。Profile-Secant V3 在 test35 将整体 R² 从 `0.9561` 提高到 `0.9617`、pooled
> high-WSS R² 从 `0.9282` 提高到 `0.9415`，但逐病例 high-WSS R² 均值只有
> `0.7721`，仅 6/35 病例达到 `R²≥0.90`，仅 4/35 同时满足峰值低估 `≤10%`。
> 当前输入下继续做后校准、尾部拉伸或邻域模型，没有证据能实现逐病例保证；严格
> 提升需要更靠近壁面的速度层或更高分辨率近壁速度场。

| 高 WSS 指标 | V4 final | Profile-Secant V3 |
| --- | ---: | ---: |
| 整体 R² 均值 | 0.9561 | **0.9617** |
| pooled high-WSS R² | 0.9282 | **0.9415** |
| 逐病例 high-WSS R² 均值 | 0.7403 | **0.7721** |
| high-WSS NRMSE | 0.1702 | **0.1583** |
| 平均峰值低估 | 11.51% | **9.31%** |

2026-08-04 图件见
[Profile-Secant V3 高 WSS 对比图](../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/15_profile_secant_high_tail_comparison.png)，
原始 V4 final 的逐病例高 WSS 审计见
[test35 高 WSS 审计图](../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/13_test35_high_wss_case_audit.png)。

只看 Profile-Secant V3 的独立图集与 ParaView 面片包见
[Profile-Secant V3 全壁面可视化](../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/README.md)。
旧 [`profile_secant_v3/`](../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3/README.md)
明确保留为 test35 × 1200 历史预览。
最佳/最差病例按 test35 逐病例 high-WSS R² 选择，分别为
`ILO/LI_YOU_ZHI-0/before` 和 `AG/slow/ZHANG_WEI_XIAN`。

V4 final 冻结基线的 evaluation-only test35 结果如下：

| 指标 | V3 | 物理 V4 | 最终 V4 |
| --- | ---: | ---: | ---: |
| raw R² mean | 0.89570 | 0.90788 | **0.95607** |
| scaled R² mean | 0.92973 | 0.93612 | **0.95844** |
| Spearman mean | 0.98201 | 0.98367 | **0.99046** |
| MAE (Pa) | 0.88774 | 0.82646 | **0.53018** |
| NRMSE | 0.38434 | 0.36091 | **0.24479** |
| high-WSS NRMSE | 0.27287 | 0.25718 | **0.17017** |

最终 V4 相对物理 V4 和 V3 均为 **35/35 病例获胜**。test35 的 raw R²
p05 为 `0.92391`，最差病例仍达到 `0.90025`。

## 从第一性原理分解问题

对于广义牛顿流体，壁面剪切应力由切向速度在壁面处的单侧法向导数决定：

```text
tau_w = mu(gamma_dot) * P_t * (du/dn)|wall
```

因此，WSS 误差可以分解为方向误差与幅值误差。在 test35 上，V1/V3 的中位
方向余弦平均已经达到 `0.99852`。这意味着继续调整法向邻域，或者只增加切平面
插值密度，无法解释大部分剩余 WSS 误差。当前主要问题是有限单元中心间距和重建
算子对 `(du/dn)|wall` 的衰减。

V3 先通过三维 IDW 在目标法向射线上生成伪观测，再拟合满足无滑移条件的速度
剖面。增加射线站点并不会增加独立信息：相邻站点往往重复使用同一批内部单元，
而 IDW 会在拟合前平滑近壁速度梯度。V4 因此改为直接拟合真实内部单元观测。

## 第一层：最近曲面鲁棒 MLS 物理核

每个内部单元首先归属到其最近的壁面锚点。单元深度沿该锚点的内法向计算，锚点
位置则投影到目标壁面点的切向坐标系。V4 拟合：

```text
u_q = z * (b0 + b_eta*z + b1*s1 + b2*s2)
```

其中：

- `z` 是沿各自壁面锚点法向计算的深度；
- `s1/s2` 是锚点相对目标点的曲面切向坐标；
- `u_q` 是速度在冻结 V1 剪切方向上的投影。

所有基函数项都包含 `z`，因此无滑移条件施加在局部曲面上，而不仅仅是单个目标
切平面上。

拟合直接使用真实单元中心，并结合 Huber IRLS、各向异性空间权重、锚点法向
一致性、多深度交叉验证、条件数检查和有界正向修正。最终物理修正为：

```text
correction_physics_v4 = max(correction_v3, correction_surface_mls)
```

这样既保留稳定的 V1 方向，又只在独立证据表明梯度受到衰减时提高其幅值。采样、
拟合、门控和融合过程均不使用 CFD WSS 真值。

## 第二层：冻结诊断残差校准器

未经监督校准的物理 V4 仍有稳定的幅值低估，在 test35 上
`alpha=1.1422`。这种残差并非完全随机，而是与推理时已经可获得的物理诊断量
相关，包括预测 WSS/梯度、V3 与曲面拟合比例、局部 Reynolds 数、半径、深度、
拟合质量、病例内排序和数据队列。

校准器在推理时**不使用**壁面坐标、病例 ID 或 CFD 真值。冻结模型包含：

- 点级 `HistGradientBoostingRegressor`，用于学习局部残差结构；
- 低维病例级 Ridge 尺度锚点，用于修正整体幅值衰减；
- 原始修正比例和最终修正比例边界；
- 接近“不修正”边界时减弱局部修正强度的保护机制；
- 当预测病例尺度到达下边界时回退到物理 V4。

由于部分特征是病例内统计量和排序，模型推理时需要一次输入完整病例。若诊断特征
的名称或顺序与冻结模型不一致，`apply_v4_calibrator.py` 会在预测前直接拒绝运行。

## 验证协议

当前证据分为三个层级：

1. train138 上按病例分组的 5-fold OOF；
2. 固定 development73 模型在其余 holdout65 病例上的验证；
3. 使用完整 train138 拟合并冻结模型后，在 test35 × 1200 上进行
   evaluation-only 应用。
4. 使用同一冻结 Profile-Secant V3 模型，在 test35 全部 35 例上以
   `sample_count=0` 追加全壁面推理，共 `1,328,017` 个壁面节点。

test35 在项目中已有历史，因此不能称为新采集的前瞻性盲测队列。但它没有参与
校准器拟合：序列化模型只使用 train138，在应用 test35 前已经冻结并记录哈希，
而且重复应用结果逐字节一致。若要形成临床或外部泛化结论，仍需要一个全新、未经
查看的独立队列。

完整指标、哈希、设计决策和限制见 `RESULTS.md`。

## 主要产物

- 物理核：`../../src/wss_surface_mls_v4.py`
- 物理实验与点级缓存生成：`../../src/run_v4_experiment.py`
- OOF 校准研究：`../../src/calibrate_v4_oof.py`
- 固定 holdout 验证：`../../src/validate_v4_calibrator_holdout.py`
- 冻结模型拟合：`../../src/fit_v4_calibrator.py`
- 冻结模型推理：`../../src/apply_v4_calibrator.py`
- 冻结模型：`calibrator_frozen_v4.joblib`
- 模型元数据：`calibrator_manifest_v4.json`
- 不可变产物哈希：`freeze_manifest.json`
- 当前推荐结果模型：`calibrator_profile_secant_high_tail_anchor10_v3.joblib`
- 当前推荐 test35 结果：
  `../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_profile_secant_high_tail_anchor10_v3.json`
- 当前推荐 full-wall test35 结果：
  `../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/inference/test35_profile_secant_v3_fullwall.json`
- 全壁面分片与审计入口：`../../src/run_profile_secant_v3_fullwall.py`
- Profile-Secant V3 完整说明：`PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md`
- Profile-Secant V3 纯模型图集与 VTP：
  `../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/`
- 可视化图集：
  `../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/`

## 复现方法

使用 GNN 环境，并将计算器源码目录和仓库根目录同时加入 `PYTHONPATH`。耗时较长
的缓存生成与拟合任务在 node04 上运行。当前实现基于 NumPy、SciPy 和
scikit-learn，因此即使在 node04 设置 `CUDA_VISIBLE_DEVICES`，核心计算仍然
主要由 CPU 完成。

```bash
export PYTHONPATH=/public/newhome/cy/Digital_twin/GNN/wss_mri_calculator/src:/public/newhome/cy/Digital_twin/GNN
export PYTHON=/public/newhome/cy/.conda/envs/GNN/bin/python

$PYTHON wss_mri_calculator/src/run_v4_experiment.py \
  --config wss_mri_calculator/experiments/pointcloud_surface_mls_v4/config_train138.json

$PYTHON wss_mri_calculator/src/calibrate_v4_oof.py \
  --cache-dir outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_train138_s600 \
  --json-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/calibration_oof_train138.json

$PYTHON wss_mri_calculator/src/validate_v4_calibrator_holdout.py \
  --cache-dir outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_train138_s600 \
  --case-order-result outputs/wss_mri_calculator/pointcloud_surface_mls_v4/train138_s600.json \
  --development-cases 73 \
  --json-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/calibrator_dev73_holdout65.json

$PYTHON wss_mri_calculator/src/fit_v4_calibrator.py \
  --cache-dir outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_train138_s600 \
  --model-out wss_mri_calculator/experiments/pointcloud_surface_mls_v4/calibrator_frozen_v4.joblib \
  --manifest-out wss_mri_calculator/experiments/pointcloud_surface_mls_v4/calibrator_manifest_v4.json

$PYTHON wss_mri_calculator/src/run_v4_experiment.py \
  --config wss_mri_calculator/experiments/pointcloud_surface_mls_v4/config_test35_cache.json

$PYTHON wss_mri_calculator/src/apply_v4_calibrator.py \
  --cache-dir outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_test35_s1200 \
  --model wss_mri_calculator/experiments/pointcloud_surface_mls_v4/calibrator_frozen_v4.joblib \
  --json-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_calibrated_s1200.json \
  --predictions-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_calibrated_predictions_s1200.npz
```
