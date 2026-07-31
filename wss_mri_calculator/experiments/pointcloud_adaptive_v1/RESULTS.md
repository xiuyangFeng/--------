# 最终结果：Point-cloud adaptive WSS v1

## 口径

- 病例：冻结 split 的 train138 + test35，共 173 例；
- 每例：固定随机种子抽 1200 个壁面点；
- 时间步：bundle `peak_step`；
- baseline：固定 K=64；
- candidate：train138 冻结的 adaptive-CV，候选 K 为
  `16,20,24,28,32,36,40,44,48,64`，tolerance=1.25；
- 两方法共享相同病例、时间步、壁面点、PCA 法向和 Carreau–Yasuda；
- test35 未参与任何超参数选择。

原始结果：[`results/stage_c_split173_s1200.json`](results/stage_c_split173_s1200.json)。

汇总图：[`fig_split173_comparison.png`](fig_split173_comparison.png)，生成脚本为
[`plot_results.py`](plot_results.py)。

最终模型逐点散点图（blind test35，不带/带 train-only 标量）：
[`fig_final_model_scatter_raw_vs_scaled.png`](fig_final_model_scatter_raw_vs_scaled.png)，
对应脚本和指标为 [`plot_final_scatter.py`](plot_final_scatter.py) 与
[`final_scatter_metrics.json`](final_scatter_metrics.json)。

## 全部 173 例

| 指标 | fixed K=64 | adaptive-CV | 变化 |
| --- | ---: | ---: | ---: |
| raw R² mean | 0.6943 | **0.8540** | +0.1597 |
| raw R² p05 | 0.5768 | **0.7618** | +0.1850 |
| raw R² min | 0.4722 | **0.6931** | +0.2210 |
| scaled R² mean | 0.8701 | **0.9207** | +0.0506 |
| Spearman mean | 0.9720 | **0.9798** | +0.0078 |
| MAE mean (Pa) | 1.5006 | **1.0546** | -29.7% |
| NRMSE mean | 0.6791 | **0.4580** | -32.6% |
| high-WSS NRMSE mean | 0.4963 | **0.3229** | -34.9% |
| direction cosine p50 mean | 0.9957 | **0.9985** | +0.0027 |
| alpha mean | 1.4988 | **1.2388** | 幅值低估减小 |

成对 raw R² 增益：mean `+0.1597`，p05 `+0.0829`，min `+0.0526`；
**173/173 病例均为正增益**。

## 未参与调参的 test35

| 指标 | fixed K=64 | adaptive-CV |
| --- | ---: | ---: |
| raw R² mean | 0.6994 | **0.8586** |
| raw R² p05 | 0.5793 | **0.7647** |
| raw R² min | 0.5736 | **0.7568** |
| MAE mean (Pa) | 1.5389 | **1.0754** |
| high-WSS NRMSE mean | 0.4868 | **0.3130** |
| Spearman mean | 0.9734 | **0.9812** |
| direction cosine p50 mean | 0.9958 | **0.9985** |

test35 成对 raw R² 增益 mean `+0.1593`，min `+0.0526`；**35/35 提升**。

## 可选：train-only 全局幅值校准

纯物理 adaptive 仍有 `alpha≈1.24` 的幅值低估。使用
[`derive_global_scale.py`](derive_global_scale.py) 只根据 train138 推导出冻结尺度：

```text
prediction_scale = 1.2220224407405893
```

随后在 blind test35 一次性评估：

| 指标 | adaptive raw | adaptive + train scale |
| --- | ---: | ---: |
| raw R² mean | 0.8586 | **0.9153** |
| raw R² p05 | 0.7647 | **0.8555** |
| raw R² min | 0.7568 | **0.8205** |
| MAE mean (Pa) | 1.0754 | **0.7530** |
| NRMSE mean | 0.4375 | **0.3409** |
| high-WSS NRMSE | 0.3130 | **0.2378** |

原始结果：[`results/stage_d_test35_global_scale1222.json`](results/stage_d_test35_global_scale1222.json)。
该版本显式使用 train 真值拟合一个全局常数，因此必须与零校准的 raw 物理结果分开报告。

## 分队列 raw R²

| 队列 | n | fixed mean | adaptive mean | adaptive min | 成对增益 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| AG | 76 | 0.6659 | **0.8126** | 0.6931 | +0.1467 |
| AAA | 56 | 0.7377 | **0.8927** | 0.7785 | +0.1550 |
| ILO | 41 | 0.6877 | **0.8781** | 0.7708 | +0.1904 |

病例级选中 K 的中位数也验证了固定 K 不合理：AG 多为 `44`，AAA 多为 `24`，
ILO 多为 `24`。train/test 的病例级 K 中位数分布一致，未见 test 漂移。

## 结论

在不使用 CFD 网格拓扑的约束下，主要误差源确实是固定邻域跨越不同数量的边界层
单元。用速度剖面自身的 LOOCV 自适应选择局部尺度，可以同时提高 R²、相关性、
方向一致性，并显著降低绝对误差和高 WSS 区误差。

残余的 `alpha≈1.24` 可被 train-only 全局尺度部分吸收；若继续优化，应研究基于局部
点云尺度/曲率的正值离散化修正或受约束 MLS，而不是继续扩大 K。
