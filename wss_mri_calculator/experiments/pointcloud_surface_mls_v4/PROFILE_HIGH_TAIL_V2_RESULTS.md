# V4 近壁剖面高尾候选 V2 结果

## 决策

冻结的 V4 final 保持不变。Profile V2 是历史研究候选；当前推荐结果模型已经升级为
Profile-Secant V3，Profile V2 不再作为主结果。
它改善了 pooled 和平均高 WSS 精度，但不能保证每个病例都同时达到
`high-WSS R² ≥ 0.90` 和峰值低估 `≤10%`。

## 方法

方法保留原 V4 预测作为物理基底，只使用 depth-3 最近曲面拟合暴露推理期可获得的
近壁速度剖面诊断量：

- 最小内部单元深度、深度分位数和深度跨度；
- 二次、三次深度曲率系数；
- 加权剖面拟合残差；
- 不同允许深度窗口下的梯度变化；
- 有效样本数、壁面锚点组数、条件数和 depth-3 修正病例内排序。

train138 病例分组 OOF 选择了 10% 峰值锚定强度。它是已筛选强度中第一个使 OOF
平均峰值低估不超过 10% 的设置。test35 没有参与特征集、模型参数或锚定强度选择。

## 证据

| 评估 | 原高尾模型 | Profile V2 | 变化 |
| --- | ---: | ---: | ---: |
| train138 grouped OOF，逐病例 high-WSS R² 均值 | 0.7517 | 0.7821 | +0.0304 |
| train138 grouped OOF，high-WSS NRMSE 均值 | 0.1659 | 0.1549 | -0.0110 |
| train138 grouped OOF，峰值低估均值 | 10.82% | 9.91% | -0.91 pp |
| holdout65 pooled high-WSS R² | 0.9223 | 0.9330 | +0.0107 |
| holdout65 逐病例 high-WSS R² 均值 | 0.7890 | 0.7762 | -0.0128 |
| holdout65 峰值低估均值 | 12.09% | 9.14% | -2.95 pp |
| test35 整体 R² 均值 | 0.9551 | 0.9608 | +0.0057 |
| test35 pooled high-WSS R² | 0.9304 | 0.9408 | +0.0104 |
| test35 逐病例 high-WSS R² 均值 | 0.7397 | 0.7669 | +0.0271 |
| test35 high-WSS NRMSE 均值 | 0.1701 | 0.1598 | -0.0103 |
| test35 峰值低估均值 | 9.58% | 9.06% | -0.52 pp |

Profile V2 在 test35 的逐病例达标数：

- `high-WSS R² ≥ 0.90`：4/35；
- 峰值低估 `≤10%`：22/35；
- 两项同时满足：3/35。

因此，pooled high-WSS R² 和平均峰值低估达到了目标数值，但更严格的逐病例保证
没有被证明，而且与当前 test35 证据不符。

## 已否决组合

- 在 depth-3 融合物理缓存上直接重训校准器，使 test35 pooled high-WSS R²
  降至 `0.9239`；
- 在原校准结果上叠加 depth-3 增益，使 test35 pooled high-WSS R² 从 `0.9304`
  降至 `0.9284`；
- 直接采用病例均衡 high-tail R² 权重目标，grouped OOF 表现低于标准 Profile 模型。

## 产物

- 模型：`calibrator_profile_high_tail_anchor10_v2.joblib`
- 模型 SHA256：`aca02647446e0dfe2fdac87defb02bd3bf9960c8437cb9ff1f20ccd8a188302a`
- test35 结果：`test35_profile_high_tail_anchor10_v2.json`
- test35 predictions SHA256：
  `edc64a9cacc961d69115954bd15f2c7421b7fb841190c5d4c3090a27647e5000`
- 对比图：`00_quicklook/14_profile_high_tail_comparison.png`

## 验证

- 12 项相关单元测试和模型合同测试通过；
- 重复 test35 推理得到逐字节一致的 prediction archive。
