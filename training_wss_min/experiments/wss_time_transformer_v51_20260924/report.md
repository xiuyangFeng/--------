> INVALIDATED: evaluation/control contract failed. Historical audit only; do not use for formal comparisons.

# wss_time_transformer_v51_20260924

缺失：无

| 臂 | cycle Pa | trough Pa | TAWSS | cycle ln | peak p90 | ts corr | amp err |
|---|---:|---:|---:|---:|---:|---:|---:|
| TT-warm | 0.5035 | 0.2936 | 0.6906 | 0.6548 | 8.67 | 0.8542 | 0.5524 |
| TT-warm-noattn | 0.5017 | 0.2891 | 0.6893 | 0.6580 | 8.74 | 0.8535 | 0.5585 |
| TT-raw | 0.5005 | 0.2884 | 0.6932 | 0.6466 | 9.04 | 0.8490 | 0.5842 |
| TT-raw-noattn | 0.5014 | 0.2914 | 0.6925 | 0.6495 | 9.05 | 0.8490 | 0.5873 |

配对差（前者减后者，fold0/1/2；单 seed 描述）：

- **A1_warm_minus_warm_noattn** cycle Pa = +0.0069/-0.0007/-0.0010；mean=+0.0018；trough mean=+0.0045
- **A2_warm_minus_raw** cycle Pa = +0.0070/+0.0026/-0.0006；mean=+0.0030；trough mean=+0.0051
- **A3_raw_attention_minus_noattn** cycle Pa = -0.0014/-0.0011/-0.0003；mean=-0.0009；trough mean=-0.0029
- **A4_warm_representation_minus_raw** cycle Pa = -0.0013/+0.0021/+0.0001；mean=+0.0003；trough mean=-0.0023

解释边界：本矩阵使用静态 V5.1 几何和相位，没有逐帧速度/压力状态；Transformer 只在病例级几何 token 上做时间注意力。结果用于筛选结构，不改变部署。
