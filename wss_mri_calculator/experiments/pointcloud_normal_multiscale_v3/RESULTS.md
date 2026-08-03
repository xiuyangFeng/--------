# Point-cloud normal multi-scale WSS v3：冻结结果

## 数据与边界

- split：冻结的 train138 + test35，SHA256
  `964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b`；
- V3 的方法选择、消融和参数修订全部只使用 train138；
- 唯一配置先冻结为 `config_frozen_v3.json`，随后才执行一次 blind test35；
- test35 结果未用于任何调参，冻结后不允许 post-test tuning；
- V1/V2 源码、配置和正式结果由各自 `freeze_manifest.json` 保护。

## 方法演进

### 初版：直接法线管选真实单元点

初版在三个 train 病例上 raw R² mean 仅 `0.7306`，低于 V1 `0.8406`。虽然覆盖率
为 100%，但内部单元中心到目标法线的横向距离仍为法向深度的约 `1.6–1.8` 倍，
且每尺度只有约 16 个点，出现明显高方差。

因此没有进入 train138。

### 法线站点插值

在法线上设置 12 个物理深度站点，每个站点用附近 16 个内部单元做 IDW 插值，
再拟合二次无滑移剖面。相较“复制最近单元速度”，对称插值能降低非结构网格横向偏移。

三病例冒烟中，纯 ray 版本显著改善低 R² 的 CHEN_SHI_MING，但会轻微降低原本已经
很好的 AAA/ILO 病例。因此 V3 改为几何风险 hybrid：

1. `lateral_p50 / eta_p50 >= 1.4`；
2. 每点 ray 插值的非局部壁面归属拒绝数 `<=20`；
3. `|g_ray| / |g_v1| >= 1.0`，只允许修复 V1 的幅值低估；
4. 保留冻结 V1 梯度方向；
5. 最终幅值修正限制在 `[1.0, 1.2]`。

## Train36 消融

每队列取 12 个 train 病例，每例 400 个固定随机壁面点。

| 阶段 | 门控 | mean ΔR² vs V1 | wins | min ΔR² |
| --- | --- | ---: | ---: | ---: |
| A0 | lateral/eta only | +0.0248 | 18/36 | -0.0721 |
| A1 | + ownership≤20 | +0.0217 | 18/36 | -0.0713 |
| A2 | + ray/V1≥1 | +0.0332 | 31/36 | -0.0040 |
| A3 | + correction≤1.2，保留 V1 方向 | +0.0272 | **33/36** | **-0.00273** |

A2 在 full train138 暴露一个 ray/V1 幅值比最高约 4.3 的灾难例，因此最终采用 A3
的 `1.2` 上限。A3 虽然牺牲部分平均增益，但消除了高倍离群修正。

## Stage A4：full train138 × 600，geometry-only capped

结果：
`outputs/wss_mri_calculator/pointcloud_normal_multiscale_v3/stage_a4_geometry_cap120_train138_s600.json`

V3 多尺度修正保持关闭，仅验证法线插值、局部 Re 深度先验和 hybrid 门控。

| 指标 | adaptive-CV V1 | capped V3 | 变化 |
| --- | ---: | ---: | ---: |
| raw R² mean | 0.8572 | **0.8915** | +0.0343 |
| raw R² p05 | 0.7547 | **0.7978** | +0.0432 |
| raw R² min | 0.6798 | **0.7305** | +0.0507 |
| scaled R² mean | 0.9242 | **0.9270** | +0.0029 |
| MAE (Pa) | 1.0296 | **0.8734** | -15.2% |
| NRMSE | 0.4493 | **0.3995** | -11.1% |
| high-WSS NRMSE | 0.3151 | **0.2767** | -12.2% |
| Spearman | 0.97946 | **0.98057** | +0.00111 |
| alpha mean | 1.2397 | **1.1675** | 幅值低估减小 |
| direction cosine p50 | 0.99849 | 0.99849 | 完全保留 |

成对 raw R²：`127/138` 提升，mean delta `+0.03430`，p05 delta 约
`-0.000010`，最差退化仅 `-0.000379`。

分队列：

| 队列 | n | mean ΔR² | wins | min ΔR² |
| --- | ---: | ---: | ---: | ---: |
| AG | 61 | +0.06356 | 60 | 0.00000 |
| AAA | 45 | +0.01536 | 40 | -0.000293 |
| ILO | 32 | +0.00516 | 27 | -0.000379 |

## Stage B1：full train138 × 600，加入物理深度多尺度

结果：
`outputs/wss_mri_calculator/pointcloud_normal_multiscale_v3/stage_b1_multiscale_cap120_train138_s600.json`

| 指标 | adaptive-CV V1 | multiscale V2 | frozen V3 |
| --- | ---: | ---: | ---: |
| raw R² mean | 0.8572 | 0.8622 | **0.8929** |
| raw R² p05 | 0.7547 | 0.7661 | **0.7982** |
| raw R² min | 0.6798 | 0.6938 | **0.7369** |
| MAE (Pa) | 1.0296 | 1.0064 | **0.8678** |
| NRMSE | 0.4493 | 0.4421 | **0.3969** |
| high-WSS NRMSE | 0.3151 | 0.3095 | **0.2745** |
| alpha mean | 1.2397 | 1.2256 | **1.1627** |

相对 Stage A4 的 geometry-only V3，多尺度带来额外 mean ΔR² `+0.001350`，
`110/138` 病例提升；相对 V1 的 mean ΔR² 为 `+0.035652`，`129/138` 提升，
最差退化 `-0.000293`。这部分收益不大，但方向稳定，且没有破坏 capped hybrid 的
安全边界，因此被纳入唯一冻结配置。

## Stage C：blind test35 × 1200，一次性测试

结果：
`outputs/wss_mri_calculator/pointcloud_normal_multiscale_v3/stage_c_blind_test35_s1200.json`

| 指标 | adaptive-CV V1 | multiscale V2 | frozen V3 |
| --- | ---: | ---: | ---: |
| raw R² mean | 0.8586 | 0.8675 | **0.8957** |
| raw R² p05 | 0.7647 | 0.7836 | **0.8414** |
| raw R² min | 0.7568 | 0.7759 | **0.8081** |
| scaled R² mean | 0.9254 | 0.9235 | **0.9297** |
| MAE (Pa) | 1.0754 | 1.0372 | **0.8877** |
| NRMSE | 0.4375 | 0.4240 | **0.3843** |
| high-WSS NRMSE | 0.3130 | 0.3015 | **0.2729** |
| alpha mean | 1.2342 | 1.2089 | **1.1608** |

成对 raw R²：

- 相对 V1：mean delta `+0.037085`，`33/35` 提升，最差 `-0.000194`；
- 相对 V2：mean delta `+0.028175`，`29/35` 提升，最差 `-0.016458`。

## 最终判断

blind test 延续并略高于 train138 上的平均收益，同时显著改善 p05、最差病例、MAE 和
高 WSS 误差，支持老师提出的核心判断：仅按三维 KNN/adaptive-k 扩大邻域确实可能混入
相邻壁面所属的内部点并平滑 WSS；沿物理法线建立站点、用局部插值投影速度、再加入
壁面归属过滤与局部 Re 深度先验，是更稳健的点云实现。

V3 至此冻结。后续只能开展不改变该 blind 结论的新路线，例如更严格的法线插值误差
估计、弯曲血管中的测地归属，或独立命名的 V4；不能回看 test35 后继续修改 V3。
