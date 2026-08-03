# Point-cloud multi-scale WSS v2 结果

## 算法

v2 从 v1 的多邻域二次无滑移剖面出发：

1. v1 adaptive-CV 梯度作为稳定基线和 WSS 方向；
2. 对 LOOCV 接近最优的多个尺度拟合 `g(h)=g0+c*h`，取 `h→0` 截距；
3. 只有梯度幅值随尺度增大而下降、且跨尺度方向一致时才允许正向修正；
4. 修正量在壁面点云 24-NN 上取局部中值，并向 1.0 收缩 50%；
5. 修正范围冻结为 `[1.0, 1.35]`，因此不会主动降低 v1 梯度。

局部选择和修正全过程只读取坐标、法向和 `u/v/w`，不读取 WSS 真值。

## Train138 冻结审计

每例抽样 600 个壁面点：

| 指标 | adaptive-CV v1 | multi-scale v2 |
| --- | ---: | ---: |
| raw R² mean | 0.8572 | **0.8622** |
| raw R² p05 | 0.7547 | **0.7661** |
| raw R² min | 0.6798 | **0.6938** |
| MAE (Pa) | 1.0296 | **1.0064** |
| high-WSS NRMSE | 0.3151 | **0.3095** |
| Spearman | **0.97946** | 0.97938 |
| direction cosine p50 | 0.99849 | 0.99849 |

成对 raw R²：126/138 病例提升，mean delta `+0.00499`，最差退化 `-0.00894`。

原始结果：[`results/stage_b2_train138_s600_gated_shrink05.json`](results/stage_b2_train138_s600_gated_shrink05.json)。

## Blind test35

冻结 train 配置后，每例抽样 1200 个壁面点，一次性运行 test35：

| 指标 | adaptive-CV v1 | multi-scale v2 | 变化 |
| --- | ---: | ---: | ---: |
| raw R² mean | 0.8586 | **0.8675** | +0.0089 |
| raw R² p05 | 0.7647 | **0.7836** | +0.0188 |
| raw R² min | 0.7568 | **0.7759** | +0.0191 |
| MAE (Pa) | 1.0754 | **1.0372** | -3.6% |
| NRMSE | 0.4375 | **0.4240** | -3.1% |
| high-WSS NRMSE | 0.3130 | **0.3015** | -3.6% |
| Spearman | **0.98122** | 0.98104 | -0.00018 |
| direction cosine p50 | 0.99852 | 0.99852 | 不变 |
| alpha mean | 1.2342 | **1.2089** | 幅值低估减小 |

成对 raw R²：33/35 病例提升，mean delta `+0.00891`，最差退化 `-0.00246`。

原始结果：[`results/stage_c_test35_s1200_frozen.json`](results/stage_c_test35_s1200_frozen.json)。

## 结论与使用边界

v2 是比 v1 更好的**零真值局部校准点云前向算子**：提高 raw R²，降低 MAE 和
高 WSS 误差，同时完全保留方向一致性。Spearman 的变化小于 0.0002，可视为基本持平。

但如果允许使用 train138 WSS 真值拟合一个全局标量，v1 的冻结标量版本在 test35 的
case-mean R² 仍略高：v1 calibrated `0.9153`，v2 使用各自 train 最优标量约 `0.9139`。
因此当前推荐口径是：

- 纯坐标+速度前向、不使用任何真值幅值标定：使用 `multiscale_v2`；
- 允许 train-only 全局标量且追求最高平均 R²：保留 v1 calibrated；
- 两种结果不可混写为同一个 raw 指标。

## 全壁面 train-only 标量（最终口径）

早期 `600/1200` 点用于快速参数筛选，并不是标量推导的物理要求。由于 v2 的 24-NN
空间正则依赖壁面点密度，随机子集还会改变局部正则半径。因此最终改为：

- train138 每例使用全部壁面点，共 **5,436,791** 点；
- 按病例等权最大化 train mean raw R²；
- test35 同样使用全部壁面点，共 **1,328,017** 点；
- test 不参与标量推导或选择。

冻结结果：

```text
prediction_scale = 1.157066322432233
```

配置见 [`config_calibrated_fullwall_v2.json`](config_calibrated_fullwall_v2.json)，train
合并结果见 [`results/fullwall_train138_merged_scale.json`](results/fullwall_train138_merged_scale.json)，
test 四方法成对合并结果见
[`results/fullwall_test35_merged_both_calibrated.json`](results/fullwall_test35_merged_both_calibrated.json)。

### 全壁面 train138 raw

| 指标 | adaptive-CV v1 | multi-scale v2 |
| --- | ---: | ---: |
| raw R² mean | 0.8521 | **0.8723** |
| raw R² p05 | 0.7581 | **0.8010** |
| raw R² min | 0.6980 | **0.7328** |
| MAE (Pa) | 1.0879 | **1.0158** |
| high-WSS NRMSE | 0.3239 | **0.2960** |
| Spearman | **0.97964** | 0.97920 |

### Blind test35 全壁面

| 指标 | v1 raw | v2 raw | v1 + train scale | v2 + train scale |
| --- | ---: | ---: | ---: | ---: |
| raw R² mean | 0.8557 | 0.8736 | **0.9123** | 0.9050 |
| raw R² p05 | 0.7643 | 0.8005 | **0.8577** | 0.8528 |
| raw R² min | 0.7270 | 0.7606 | **0.8131** | 0.7858 |
| MAE (Pa) | 1.1053 | 1.0296 | **0.7663** | 0.7799 |
| NRMSE | 0.4375 | 0.4077 | **0.3429** | 0.3541 |
| high-WSS NRMSE | 0.3162 | 0.2912 | **0.2442** | 0.2553 |
| Spearman | **0.98179** | 0.98134 | **0.98179** | 0.98134 |
| direction cosine p50 | 0.99849 | 0.99849 | 0.99849 | 0.99849 |
| alpha mean | 1.2343 | 1.1696 | **1.0102** | 1.0108 |

同一全壁面 train 口径下，v1 的最优全局标量为 `1.2219229200422976`。完整成对运行
确认 v1 calibrated 的 mean/p05/min R²、MAE、NRMSE 和 high-WSS NRMSE 均略优于
v2 calibrated。原因是标量只能修复整体幅值，无法提高空间分布上限；v2 的 per-case
oracle scaled R² mean 为 `0.9164`，低于 v1 的 `0.9227`。因此 v2 的明确优势是
**不带标量的 raw 前向**；若允许 train-only 标量并追求综合最优，当前应选择
v1 calibrated。
