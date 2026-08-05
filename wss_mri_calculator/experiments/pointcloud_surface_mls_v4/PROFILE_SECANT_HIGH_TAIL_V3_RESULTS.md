# V4 Profile-Secant V3 当前推荐结果模型

## 状态

面向病例总体和平均高 WSS 精度，这是当前推荐结果模型。它独立保存并使用自身模型
哈希和预测产物；V4 final 保留为冻结基线和消融对照。

当前结论必须分成两种口径：

- pooled high-WSS R² `≥0.90`：达到；
- sampled test35 × 1200 平均峰值低估 `≤10%`：达到；
- full-wall test35 平均峰值低估 `≤10%`：未达到（`15.81%`）；
- 每个病例 high-WSS R² `≥0.90`：未达到；
- 每个病例峰值低估 `≤10%`：未达到。

## 新增推理特征

V3 保留 V2 的全部近壁剖面诊断量，并增加由“内部速度 ÷ 局部壁面深度”计算的
稳健切线梯度特征：

- 切线梯度 p25 / p50 / p75 / p90；
- 距壁最近 25% 有效单元的切线梯度中位数；
- 切线梯度四分位距；
- 最近深度估计的有效单元数；
- 病例内切线梯度排序。

构造这些推理特征时不使用 CFD WSS 真值。

## 验证证据

| 评估 | Profile V2 | Profile-Secant V3 |
| --- | ---: | ---: |
| train138 grouped OOF high-WSS R² 均值 | 0.7821 | 0.7846 |
| train138 grouped OOF 整体 R² 均值 | 0.9607 | 0.9613 |
| holdout65 high-WSS R² 均值 | 0.7762 | 0.7813 |
| holdout65 high-WSS NRMSE 均值 | 0.1606 | 0.1586 |
| holdout65 峰值低估均值 | 9.14% | 8.89% |
| test35 整体 R² 均值 | 0.9608 | 0.9617 |
| test35 pooled high-WSS R² | 0.9408 | 0.9415 |
| test35 high-WSS R² 均值 | 0.7669 | 0.7721 |
| test35 high-WSS NRMSE 均值 | 0.1598 | 0.1583 |
| test35 峰值低估均值 | 9.06% | 9.31% |

Profile-Secant V3 在 test35 的逐病例达标数：

- `high-WSS R² ≥0.90`：6/35；
- 峰值低估 `≤10%`：22/35；
- 两项同时满足：4/35。

## 2026-08-04 全壁面追加推理

冻结模型在 test35 全部 35 例上以 `sample_count=0` 推理，共覆盖 `1,328,017` 个
CFD 壁面节点。模型未重训，split 和模型哈希不变。

| 指标 | sampled test35 × 1200 | full-wall test35 |
| --- | ---: | ---: |
| 病例 overall R² 均值 | 0.9617 | 0.9604 |
| pooled overall R² | 0.9662 | 0.9679 |
| pooled high-WSS R² | 0.9415 | 0.9440 |
| 逐病例 high-WSS R² 均值 | 0.7721 | 0.7735 |
| high-WSS NRMSE 均值 | 0.1583 | 0.1623 |
| 平均峰值低估 | 9.31% | 15.81% |

全壁面结果确认整体和平均高 WSS 结论稳定，但 sampled 峰值低估明显偏乐观。全量
最佳/最差病例仍为 `ILO/LI_YOU_ZHI-0/before` 和 `AG/slow/ZHANG_WEI_XIAN`，对应
high-WSS R²=`0.9167/0.4218`。完整图册、内部点采样、穿模风险和全量 VTP 见
[全壁面结果入口](../../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/README.md)。

## 可辨识性审计

真实峰值位于预测 top10% 的病例数为 holdout65 `65/65`、test35 `35/35`，因此
当前主要失败不是“没有找到热点位置”。

test35 上使用真值辅助的乐观 oracle 后：

- 每病例标量缩放 oracle：7/35 达到 `high-WSS R² ≥0.90`；
- 每病例仿射 oracle：10/35；
- 任意样本内单调映射 oracle：22/35。

即使允许最乐观的任意单调映射，仍有 13/35 病例不能达到 0.90。这说明病例尺度、
偏置、幂次拉伸以及其他不改变当前点排序的后校准，都无法消除剩余失败。

最稳定的失败相关因素是近壁分辨率。真实峰值位置处内部单元中位深度与逐病例
high-WSS R² 的 Spearman 相关约为：grouped OOF `-0.33`、holdout65 `-0.53`、
test35 `-0.60`。第一层速度单元越远，恢复高尾局部形状所需的排序信息丢失越严重。

## 已否决的 train-only 路线

- depth-3 直接融合后重训校准器；
- 校准后叠加 depth-3 增益；
- 病例均衡 R² 损失加权；
- 剖面特征与欧氏壁面邻域线性融合；
- 直接回归邻域 `log(WSS)`；
- 病例预测四个 rank-band 的单调高尾形状校准；
- 统一尾部幂次拉伸、病例最佳指数预测和单纯尾部尺度锚定；
- 原始内在坐标 ExtraTrees 与单独局部空间上下文。

这些方法都没有同时通过 grouped OOF 和固定 holdout 要求。

## 产物

- 模型：`calibrator_profile_secant_high_tail_anchor10_v3.joblib`
- 模型 SHA256：`c1e53af5d60e17620c4e5123da76bab9f135c4125decf6a282b944f492433a02`
- test35 结果：`test35_profile_secant_high_tail_anchor10_v3.json`
- test35 predictions SHA256：
  `b77ada6ae0dff5a9e706f3a35474489cd20cd50ae5c195aebdfb2382bf8f3e37`
- full-wall test35 结果 SHA256：
  `31a4bdca1f9cd9cc0e9c41baee7d3e844a955341144d815ca1ac64b6814d7e89`
- full-wall predictions SHA256：
  `9aaa80ada0a7dc22ddc8275286bdab4bd133ac774e8049e85d6b2f304f86bdbf`
- 可辨识性审计：`profile_identifiability_oof_train138_v1.json`、
  `profile_identifiability_holdout65_v1.json`、`profile_identifiability_test35_v1.json`
- 2026-08-04 对比图：`00_quicklook/15_profile_secant_high_tail_comparison.png`

## 验证

- 12 项相关单元测试和模型合同测试通过；
- 重复 test35 推理得到逐字节一致的 prediction archive。

## 严格逐病例目标所需的外部变化

需要增加新的局部排序信息，最可能的来源是更靠近壁面的速度采样层，或更高分辨率的
近壁速度场。继续在当前点云上调整校准器，不能证明或实现所有病例
`high-WSS R² ≥0.90` 的保证。
