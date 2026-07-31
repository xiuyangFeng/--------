# 实验日志

## 2026-07-31 · 实现与冒烟

- 新增 `adaptive_cv` 邻域模式；保留 `fixed` 作为兼容基线；
- 使用局部深度归一化降低二次设计矩阵条件数；
- 新增逐点选中 K、CV 分数、条件数诊断；
- 新增 `src/compare_pointcloud_methods.py`，同一病例只加载一次并做成对比较；
- 3 病例 × 120 点冒烟：baseline raw R² mean `0.662`，adaptive mean `0.830`，
  3/3 病例均提升。该结果仅验证程序链路，不作为冻结结论。

## 2026-07-31 · Stage A / train36 × 400 点

第一轮扫描固定 K 和 `tolerance={1.0,1.25,1.5,2.0}`：

- fixed K=64：raw R² mean `0.7170`；
- adaptive tolerance=1.25：raw R² mean `0.8604`，36/36 提升；
- MAE `1.2644 → 0.9056 Pa`；
- high-WSS NRMSE `0.4782 → 0.3104`。

第二轮去掉容易高方差的 K=12，并细化候选为
`{16,20,24,28,32,36,40,44,48,64}`：

- tolerance=1.15 / 1.25 / 1.35 的 raw R² mean 分别为
  `0.8633 / 0.8647 / 0.8654`；
- tolerance=1.25 的最差病例 raw R² 为 `0.6156`，优于 1.35 的 `0.6073`，
  且 MAE 更低，因此进入 Stage B 的主候选为 `1.25`；
- 1.15 和 1.35 作为相邻消融继续在 train138 上比较，不接触 test。

## 2026-07-31 · Stage B / train138 × 600 点 · 配置冻结

| 方法 | raw R² mean | p05 | min | MAE Pa | high-WSS NRMSE | 方向余弦 p50 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed K=64 | 0.7039 | 0.5440 | 0.5103 | 1.4500 | 0.4837 | 0.9958 |
| adaptive tol=1.15 | 0.8554 | 0.7484 | 0.6766 | **1.0209** | 0.3209 | 0.9982 |
| adaptive tol=1.25 | **0.8572** | **0.7547** | **0.6798** | 1.0296 | 0.3151 | 0.9985 |
| adaptive tol=1.35 | 0.8562 | 0.7525 | 0.6750 | 1.0468 | **0.3126** | **0.9987** |

冻结 `tolerance=1.25`：它在 raw R² mean/p05/min 上均为三者最好，138/138 相对
fixed K=64 提升；虽然 MAE 比 1.15 高 0.009 Pa、高 WSS NRMSE 比 1.35 高 0.0025，
但综合 R²、尾部稳定性和绝对误差最均衡。

冻结配置：

```text
neighbor_mode = adaptive_cv
adaptive_neighbors = 16,20,24,28,32,36,40,44,48,64
cv_tolerance = 1.25
degree = 2
normals = pca
viscosity = carreau
```

## 2026-07-31 · Stage C / split173 × 1200 点 · 最终结果

- split SHA256：`964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b`；
- 173/173 完成、173/173 真值可用、失败 0；
- adaptive 相对 fixed K=64：**173/173 raw R² 提升**；
- raw R² mean `0.6943 → 0.8540`，p05 `0.5768 → 0.7618`，min `0.4722 → 0.6931`；
- MAE `1.5006 → 1.0546 Pa`；
- high-WSS NRMSE `0.4963 → 0.3229`；
- Spearman `0.9720 → 0.9798`；
- 方向余弦 p50 的病例均值 `0.9957 → 0.9985`；
- test35 raw R² mean `0.6994 → 0.8586`，35/35 提升。

最终 JSON SHA256：
`3a2156755265563c5af5630250c824b9f335c99191f145e4d7d650b50e85dd4e`。

## 2026-07-31 · Stage D / train-only 全局幅值校准 → blind test35

根据 Stage C 的 train138 病例摘要，用解析公式最大化 train 的病例平均 raw R²，得到
全局尺度 `1.2220224407405893`。该尺度在查看 test 指标前冻结。

test35 结果：

| 指标 | fixed K=64 | adaptive raw | adaptive + train scale |
| --- | ---: | ---: | ---: |
| raw R² mean | 0.6994 | 0.8586 | **0.9153** |
| raw R² p05 | 0.5793 | 0.7647 | **0.8555** |
| raw R² min | 0.5736 | 0.7568 | **0.8205** |
| MAE Pa | 1.5389 | 1.0754 | **0.7530** |
| NRMSE | 0.6585 | 0.4375 | **0.3409** |
| high-WSS NRMSE | 0.4868 | 0.3130 | **0.2378** |

校准版相对 baseline 仍为 35/35 提升。它是显式 train-supervised 的离散化校正，
不应与无真值自由参数的 adaptive raw 指标混写。

