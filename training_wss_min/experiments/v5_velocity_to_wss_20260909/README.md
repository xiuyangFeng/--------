# VELWSS1：R5V 预测速度经冻结 Profile-Secant V3 求 WSS

本次完成同一 test34、峰值帧 1162 的 V5 有效全壁面评估，共 1,231,295 个壁面顶点。主结果物理 R²_cb=0.49123，MAE=2.51530 Pa，log_z R²_cb=0.67685，热点 IoU=0.30571。

原始 ASCII 共 1,231,298 个壁面顶点；V5 既有 valid-mask 排除 3 个点后保留 1,231,295 个。原有排除病例/点数为：AAA/ruputer/WANG_FU_SHUN（1 个）；ILO/SUN_XU_XIA-1/before（1 个）；ILO/ZHANG_JIN_CHUN-1/before（1 个）。这是现有 V5 数据口径，本次未新增选点或按预测结果排除点，三种方法与 R4/M2 参考均使用该有效壁面。

**本轮忠实保留旧算法的半径解析与映射，包括已发现的历史坐标尺度问题。主结果和 oracle 都受这一共同条件约束，不能把本次复现通过解释为几何映射正确或几何方案已验收。**

## 实验身份与结果口径

- **VELWSS1 主成绩**：已训练 R5V best 的预测速度 → 冻结 Profile-Secant V3 → WSS 标量幅值。
- **CFD 速度 oracle**：相同算法改用 CFD 真值速度，用于检查输入速度误差影响；属于审计，不能作为部署成绩或严格数学上界。
- **physics_only**：预测速度经同一冻结物理核、移除已训练幅值校准器；用于定位校准影响。
- **R4 与 M2**：同一 test34、seed1234、best 的直接 WSS 历史参考。R4 是 17D 几何的 L-SA2 配方；M2 另有 25D 输入、近壁面分支和独立 query/3NN。其模型、输入及训练目标与 R5V 不同，横向差值不构成单变量消融。

R5V 使用原有 18D 输入、QAD-lite 解码器和速度分量 MSE，输出三分量速度；本次只做冻结推理和 WSS 后处理，没有新增模型训练或 WSS 监督。冻结 V3 校准器在历史阶段使用过 train138 的 WSS 监督，因此完整流程应称为带既有校准器的速度派生方法。

## WSS 指标

| 指标 | VELWSS1 主结果 | CFD 速度 oracle | 未校准物理核 | R4 直接 WSS | M2 直接 WSS |
|---|---:|---:|---:|---:|---:|
| 物理 R²_cb（病例等权） | 0.49123 | 0.96501 | 0.46825 | 0.57215 | 0.62841 |
| 物理 R²_pool（顶点合并） | 0.44048 | 0.96690 | 0.41755 | 0.52031 | 0.57617 |
| 逐例 R² 均值 | 0.37964 | 0.95980 | 0.42521 | 0.54005 | 0.62075 |
| 逐例 R² 中位数 | 0.42562 | 0.96486 | 0.44111 | 0.55134 | 0.64248 |
| 逐例 R² 第10百分位 | 0.17086 | 0.93337 | 0.28277 | 0.35427 | 0.45553 |
| R²<0 病例数 / 34 | 1 | 0 | 1 | 0 | 0 |
| pooled MAE（Pa） | 2.51530 | 0.47489 | 2.45152 | 2.16102 | 1.98823 |
| pooled RMSE（Pa） | 6.36700 | 1.54854 | 6.49615 | 5.89530 | 5.54141 |
| 逐例 MAE 均值（Pa） | 2.34012 | 0.49074 | 2.29410 | 2.02129 | 1.81411 |
| 高 WSS 区域 R² | -0.01971 | 0.94094 | -0.11250 | 0.09002 | 0.21946 |
| 高 WSS 区域 MAE（Pa） | 10.49061 | 2.13407 | 11.24579 | 10.03855 | 9.18284 |
| 真值 top10% 区域幅值比 | 0.53846 | 1.00351 | 0.47545 | 0.55896 | 0.61560 |
| p99 预测/真值比 | 0.53172 | 1.00888 | 0.46273 | 0.56779 | 0.60993 |
| 热点 top10% IoU（病例均值） | 0.30571 | 0.81817 | 0.30724 | 0.36178 | 0.42747 |
| V5 有效全壁面 Spearman（病例均值） | 0.80230 | 0.99217 | 0.80488 | 0.85078 | 0.88457 |
| log_z R²_cb | 0.67685 | 0.98582 | 0.68704 | 0.78635 | 0.82370 |
| log_z R²_pool | 0.63972 | 0.98845 | 0.65518 | 0.77864 | 0.81447 |
| log_z 逐例 R² 均值 | 0.59383 | 0.98180 | 0.60985 | 0.72726 | 0.77982 |
| log_z pooled MAE | 0.45115 | 0.07092 | 0.43715 | 0.34594 | 0.30601 |
| log_z pooled RMSE | 0.60459 | 0.10823 | 0.59148 | 0.47391 | 0.43386 |

所有指标来源于 metrics.json；物理空间单位为 Pa，log_z 使用既有训练集统计。R²_cb 与逐例 R² 均值不同；pooled MAE/RMSE 按所有顶点计算。高 WSS 区域按各病例真值 q90 划分，top10 幅值比与 p99 比遵循现有评估器定义；p99 比越接近 1 越好。

[全部聚合数值（完整字段名）](aggregate_metrics_full.md) · [34例详细比较 CSV](per_case_comparison.csv) · [原始完整指标](metrics.json)

## 观测与比较

主结果相对 M2 的物理 R²_cb 差值为 -0.13718，相对 R4 为 -0.08092。相对 CFD 速度 oracle 为 -0.47378，相对未校准物理核为 +0.02298。这些差值只描述本次冻结管线的实际表现。

主结果有 1 / 34 例 R²<0。下表展示全部病例，按主结果 R² 从低到高排列，以便直接看到失效病例；未按成绩排除 V5 有效壁面顶点或病例。

| 病例 | 主结果 R² | oracle R² | 物理核 R² | M2 R² | Δ主结果−M2 | 主结果 MAE Pa | 主结果 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAA/ruputer/KANG_YONG | -1.34163 | 0.97635 | -0.55928 | 0.33985 | -1.68148 | 1.39529 | 0.15682 |
| ILO/LI_YOU_ZHI-0/before | 0.12986 | 0.97755 | 0.34861 | 0.65116 | -0.52129 | 1.25066 | 0.35629 |
| ILO/SUN_DONG_XIN-0/before | 0.13435 | 0.96435 | 0.20261 | 0.49171 | -0.35736 | 1.43354 | 0.17837 |
| ILO/ZHANG_HE_PING-0/before | 0.14255 | 0.96933 | 0.37278 | 0.70882 | -0.56627 | 1.76047 | 0.27082 |
| AG/slow/ZHANG_WEI_XIAN | 0.23691 | 0.90602 | 0.30499 | 0.53054 | -0.29363 | 3.18788 | 0.27123 |
| ILO/ZHANG_JIN_CHUN-1/before | 0.27964 | 0.96225 | 0.23280 | 0.39857 | -0.11893 | 4.62808 | 0.28670 |
| ILO/ZHANG_YONG_SHENG-0/before | 0.30334 | 0.97951 | 0.51300 | 0.64835 | -0.34502 | 1.05587 | 0.38134 |
| AG/slow/MA_YU | 0.30463 | 0.96937 | 0.31370 | 0.49918 | -0.19455 | 2.16360 | 0.22581 |
| AAA/unruputer/ZHANG_YONG_ZHI | 0.32511 | 0.95092 | 0.27324 | 0.45042 | -0.12532 | 4.52350 | 0.31041 |
| AAA/unruputer/SUN_SHU_MING | 0.33137 | 0.96478 | 0.39682 | 0.55866 | -0.22728 | 0.96869 | 0.18777 |
| AG/fast/SUN_ZHI_YU | 0.33991 | 0.94682 | 0.39818 | 0.18966 | 0.15025 | 1.08912 | 0.27105 |
| AAA/ruputer/WANG_FU_SHUN | 0.35197 | 0.96494 | 0.33321 | 0.46743 | -0.11547 | 3.02999 | 0.33712 |
| AAA/unruputer/GAO_DIAN_WEN | 0.35203 | 0.96521 | 0.47803 | 0.70061 | -0.34858 | 1.59418 | 0.19464 |
| AAA/unruputer/MA_JIN_HE | 0.37309 | 0.93891 | 0.37195 | 0.48201 | -0.10892 | 2.21387 | 0.38865 |
| ILO/SUN_XU_XIA-1/before | 0.38853 | 0.97468 | 0.35591 | 0.57644 | -0.18791 | 6.82998 | 0.35697 |
| AG/slow/CHEN_JING_RU | 0.40932 | 0.96414 | 0.46058 | 0.68093 | -0.27161 | 1.49960 | 0.21342 |
| AG/slow/HE_SHU_ZHEN | 0.42370 | 0.92466 | 0.39700 | 0.62773 | -0.20403 | 1.78900 | 0.24393 |
| AG/slow/LI_HUAN_GE | 0.42753 | 0.97286 | 0.48280 | 0.69560 | -0.26806 | 3.24331 | 0.24145 |
| AG/fast/FAN_JIAN_MING | 0.43370 | 0.93151 | 0.44792 | 0.55434 | -0.12064 | 1.37382 | 0.31154 |
| AAA/unruputer/WANG_MAN_TIAN | 0.44899 | 0.96031 | 0.39887 | 0.49428 | -0.04529 | 2.21243 | 0.55534 |
| ILO/LU_FU_SHAN-0/before | 0.47271 | 0.97818 | 0.54518 | 0.68844 | -0.21573 | 2.22355 | 0.28687 |
| AG/fast/ZHANG_CHUN | 0.48659 | 0.93772 | 0.43430 | 0.63661 | -0.15002 | 3.78555 | 0.30969 |
| AG/slow/MA_TIAN_YI | 0.49616 | 0.96116 | 0.42762 | 0.59123 | -0.09506 | 5.89967 | 0.45204 |
| AAA/ruputer/YU_TIAN_HAI | 0.49739 | 0.96551 | 0.48798 | 0.66107 | -0.16368 | 2.56676 | 0.35280 |
| ILO/YU_XIANG_SHENG-1/before | 0.52464 | 0.97114 | 0.50303 | 0.60207 | -0.07742 | 1.91358 | 0.34146 |
| AAA/ruputer/LI_ZHEN_HUA | 0.56580 | 0.97250 | 0.62057 | 0.79020 | -0.22440 | 2.50247 | 0.19375 |
| AG/slow/GUO_XI_JIANG | 0.56589 | 0.95009 | 0.50494 | 0.75166 | -0.18577 | 1.69884 | 0.28321 |
| ILO/YANG_WEN_TAI-0/before | 0.57945 | 0.98420 | 0.61538 | 0.77820 | -0.19875 | 1.26688 | 0.38975 |
| AG/fast/RAN_QING_BO | 0.59385 | 0.97229 | 0.52852 | 0.76401 | -0.17016 | 1.89494 | 0.33711 |
| AG/fast/YAO_CUN_HONG | 0.61544 | 0.92244 | 0.56941 | 0.77867 | -0.16323 | 2.55133 | 0.29987 |
| AAA/unruputer/LIU_KANG_WEN | 0.65037 | 0.97452 | 0.69019 | 0.83100 | -0.18063 | 1.80076 | 0.38025 |
| AG/slow/BAI_WEN_JIE | 0.65107 | 0.94811 | 0.60084 | 0.79908 | -0.14801 | 1.23534 | 0.36247 |
| AG/fast/ZHANG_LIANG | 0.66458 | 0.96873 | 0.69917 | 0.84714 | -0.18256 | 1.49140 | 0.34141 |
| AG/fast/LOU_YANG | 0.74898 | 0.96213 | 0.70626 | 0.83971 | -0.09072 | 1.49004 | 0.32394 |

本轮只使用一个既有速度训练 seed，且 test34 已用于前期探索。速度 oracle 的差距可能包含近壁面速度误差、梯度放大与校准器输入分布变化；这组结果不能将三者单独归因，也不支持稳定性或显著性结论。

## 原速度 checkpoint 与预测复现

原 run：`training_wss_min/runs/v5_rerun_20260906/outputs/r5v_velocity_qad_s1234`，checkpoint=`best`，保存 epoch 字段为 `361`（保持原记录编号）。速度推理固定 support seed1234，共 14,122,419 个体内速度向量。

原实验仅保留指标、未保留逐点预测，因此本次从原 best checkpoint 和冻结 support 协议重新导出预测。复现与历史指标按每病例物理/归一化完整数值逐项校验；不将这一步描述为直接复用已有逐点存档。

最终采用 Slurm 13789 的复现结果，共 5,372 项速度数值检查，失败 0 项；超过原严格容差的检查数为 0。数值叶最大绝对差 1.8495228e-06（混合指标单位，仅作复现审计）。原浮点容差为 5e-6 + 5e-6×|原值|，整数计数严格一致；坐标往返最大误差 6.4984586e-05 mm。源代码、checkpoint、配置和训练统计哈希前后一致。

此前一次重放在 `AAA/ruputer/YU_TIAN_HAI` 的一个热点交集边界点触发严格容差检查失败，[失败证据另存](velocity_reproduction_failed.json)。最终上述复现结果单独核验通过；不把多次 GPU 推理描述为逐位一致，也不将先前失败结果混入本次 WSS 指标。

速度分量由原线性 z 统计还原至 m/s，再用原旋转矩阵变回物理坐标系；壁面、体内坐标使用 mm，梯度核执行 mm→m 换算。这避免把归一化速度、不同坐标系向量或毫米梯度直接代入 WSS。

[逐例速度复现完整证据](velocity_reproduction.json)

## 冻结算法与标签隔离证据

数组适配器依次使用历史 V1 自适应梯度、V3 法向多尺度、V4 表面 MLS 与 depth3 profile 修正，再使用原 Carreau 流变和冻结的 84 特征 Profile-Secant V3 校准器。四个梯度阶段显式接收同一输入速度数组，主预测路径不能在中途换回 CFD 速度。校准按完整病例执行，保留病例内排名/分位特征。

梯度构建接口不接收 WSS 标签；校准入口剔除所有 truth 字段。算法来源、固定参数、训练/测试交集审计和历史 CLI/解析剪切/旋转/分块/标签扰动验证以以下机器记录为准；本报告不把缺失的证据写成通过。

```json
{
  "evaluation_gate": {
    "passed": true,
    "checkpoint": "best",
    "n_cases": 34,
    "n_wall": 1231295,
    "valid_coverage": 1.0,
    "velocity_reproduced": true,
    "no_new_training": true,
    "calibrator_train_test_overlap": 0,
    "calibrator_train_matches_current_train138": true,
    "provenance_sha256": "0085227d0ed00624d2a39065f7d00eaaba0a7597ce3e8acf4613090752d65247",
    "metrics_sha256": "bd09a28bd5ef82ac1e84fc83ac3edf38d85ea777e0d8f484b56640ba8a6069cd",
    "prediction_manifest_sha256": "572d80c4b21dc426bb0eefffdfc5ac46b8533f9fce4cf799fe77f53082ebb3bf",
    "radius_split_audit_sha256": "358cfc61c12a31e850d5634f5ee96970f860940bbc34ce9c4b3a2b50c1034790",
    "job_id": "13858"
  },
  "frozen_method": "surface_mls_v4_profile_secant_high_tail_anchor10",
  "frozen_model_sha256": "c1e53af5d60e17620c4e5123da76bab9f135c4125decf6a282b944f492433a02",
  "inference_truth_used": false,
  "historical_calibration": "frozen; historically WSS-supervised on train138",
  "adapter_contracts": {
    "status": "passed",
    "model_sha256": "c1e53af5d60e17620c4e5123da76bab9f135c4125decf6a282b944f492433a02",
    "historical_complete_cache": "/public/newhome/cy/Digital_twin/GNN/outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/inference/point_cache_test35_profile_secant_fullwall/AG__fast__YAO_CUN_HONG.npz",
    "historical_cache_sha256": "6ef83b0acde24da0e7df98e2e02f3dc04da4e1dfbeeed2e2d150fb850f760cfc",
    "canonical_id": "AG/fast/YAO_CUN_HONG",
    "n_wall": 9704,
    "frozen_cli_float32_exact": {
      "wss_vec_pa": true,
      "current_wss_vec_pa": true,
      "physics_wss_vec_pa": true,
      "high_tail_ratio": true
    },
    "truth_disturbed_and_removed_double_outputs_exact": true,
    "analytic_gradient_s_inv": 120.0,
    "analytic_tolerance_s_inv": 0.002,
    "rotation_and_equal_coordinate_velocity_scale_contract": true,
    "chunked_vs_unsplit_arrays_exact": 103,
    "n_invalid_historical_predictions": 0,
    "elapsed_seconds": 8.396607918664813
  },
  "split": {
    "calibrator_train_count": 138,
    "calibrator_train_unique_count": 138,
    "calibrator_train_equals_v5_train138": true,
    "v5_test_count": 34,
    "calibrator_training_overlap_v5_test": []
  }
}
```

[完整算法与数据来源](provenance.json) · [评估准入记录](evaluation_gate.json)

## 历史半径映射的已知局限

冻结解析器沿用 `bundle_then_wall`：优先从旧 `data_wss_min` bundle 读取中心线半径，并按旧壁面坐标最近邻配到当前物理壁面。旧 bundle 的 `wall_coords_raw` 部分使用了病例自身的 `unit_factor`，其比例并非 CFD 物理毫米转换所用的 1000；解析器仅按坐标范数阈值决定是否乘 1000，没有恢复每个病例的真实比例。

独立几何审计覆盖 34 例原始 ASCII 的 1,231,298 个壁面顶点：最近邻映射距离最大 38.56399 mm；其中 1,230,733 个顶点（99.954%）距离超过 0.1 mm。这些距离是旧半径来源坐标与目标壁面的配对残差，不能当作合理的局部插值误差忽略。

本轮主结果、CFD 速度 oracle 和未校准物理核使用相同旧半径处理，未按预测成绩修正半径或重新计算中心线。该设置保证对旧算法的忠实复用，却保留了它的几何限制；如要修正半径映射，应另列实验并先完成用户要求的几何可视化审核。

[逐病例半径与训练/测试划分审计](radius_split_audit.json)

## 可视化

壁面图固定选择每个队列按 test 顺序的首例（AAA、AG、ILO），不按预测成绩挑选。每例三个面板使用同一刚性几何 PCA 视角、同一 log1p 色标；色条刻度仍为 Pa。该例真值、主预测及 oracle 合并的 p99 用作共同上限，每面板标明超过上限的顶点比例。几何 PCA 只用于调整展示视角，不改变输入特征或计算结果。

![固定三例同视角壁面 WSS](velocity_wss_wall_fields.png)

下面的散点图使用 V5 有效全壁面顶点生成 pooled hexbin。上排显示三种方法共同全范围；下排使用真值与三组预测合并后的第99.5百分位确定共同视窗，并标注各面板实际显示比例。视窗裁切只影响展示，所有报告指标保留全部顶点。

![V5 有效全壁面 pooled 密度散点](velocity_wss_pooled_hexbin.png)

逐例图显示三条速度派生路径及 R4/M2 参考。若存在极端负 R²，横轴在 [-1,1] 外采用对称对数，所有病例和异常值仍保留，精确数值见上表与 CSV。

![34例 R² 对照](velocity_wss_case_r2.png)

[壁面图 PDF](velocity_wss_wall_fields.pdf) · [散点图 PDF](velocity_wss_pooled_hexbin.pdf) · [逐例图 PDF](velocity_wss_case_r2.pdf) · [图表来源及范围](figure_provenance.json)

## 复现报告

```bash
/public/newhome/cy/.conda/envs/GNN/bin/python -m training_wss_min.tools.report_velocity_to_wss
```

生成器只读取已通过准入的结果并生成报告，不触发训练、重新推理、跟踪文档或工作簿修改。
