# 压力/速度全局注意力：26臂结果

更新时间：2026-09-11T06:44:17+0800；整批核验完成：True。

固定18D、seed1234、400epoch；best按训练损失，last固定400轮；联合对单任务以last为主。

## 压力（Pa，混合query；内部/壁面分组见metrics.json）

| ID/ckpt | 状态 | R²_cb | Δ父臂 | MAE | RMSE | 向量RMSE | 径向R² | 周向R² | 20mm去bias MSE | 40mm去bias MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R5P/best | 历史参照 | 0.7068 | — | 136.5223 | 328.5331 | — | — | — | 16775.9065 | 15534.7065 |
| R5P/last | 历史参照 | 0.7031 | — | 136.8271 | 330.8238 | — | — | — | 17212.1013 | 15983.2624 |
| P00/best | complete | 0.7182 | 0.0115 | 133.9721 | 319.7044 | — | — | — | 15128.6280 | 14043.1748 |
| P00/last | complete | 0.7217 | 0.0186 | 133.0026 | 317.8360 | — | — | — | 14910.8191 | 13843.1990 |
| P01/best | complete | 0.7228 | 0.0045 | 130.9381 | 316.9188 | — | — | — | 14343.3117 | 13255.9959 |
| P01/last | complete | 0.7206 | -0.0012 | 130.4422 | 318.4744 | — | — | — | 14484.9380 | 13400.4896 |
| P02/best | complete | 0.7491 | 0.0309 | 118.0129 | 300.8761 | — | — | — | 12881.2855 | 12322.0224 |
| P02/last | complete | 0.7528 | 0.0311 | 117.6979 | 298.8828 | — | — | — | 12850.2926 | 12304.2518 |
| P03/best | complete | 0.7351 | 0.0123 | 115.4025 | 309.9248 | — | — | — | 12645.1132 | 12082.1550 |
| P03/last | complete | 0.7456 | 0.0250 | 113.2929 | 303.8888 | — | — | — | 12298.0570 | 11764.5808 |
| P04/best | complete | 0.6989 | -0.0238 | 136.4043 | 332.9943 | — | — | — | 16475.6354 | 15421.0732 |
| P04/last | complete | 0.6974 | -0.0231 | 136.0245 | 333.8873 | — | — | — | 16638.9985 | 15611.8435 |
| P05/best | complete | 0.7191 | 0.0201 | 132.7140 | 320.4159 | — | — | — | 15766.7363 | 14642.6235 |
| P05/last | complete | 0.7167 | 0.0193 | 132.4076 | 322.0515 | — | — | — | 15935.0273 | 14824.5239 |
| P06/best | complete | 0.6971 | -0.0256 | 138.0318 | 332.4145 | — | — | — | 17036.2425 | 15898.7844 |
| P06/last | complete | 0.6949 | -0.0257 | 137.9506 | 333.6187 | — | — | — | 17399.3466 | 16287.8574 |
| P07/best | complete | 0.7160 | 0.0188 | 124.1194 | 322.0603 | — | — | — | 14587.0471 | 13909.5229 |
| P07/last | complete | 0.7178 | 0.0229 | 122.6761 | 321.3042 | — | — | — | 14597.6422 | 13974.2318 |
| P08/best | complete | 0.6904 | -0.0067 | 130.4714 | 334.1204 | — | — | — | 15279.2771 | 14389.9431 |
| P08/last | complete | 0.6873 | -0.0075 | 130.4654 | 335.9608 | — | — | — | 15523.3075 | 14657.8502 |
| P09/best | complete | 0.7166 | 0.0262 | 122.7785 | 321.4018 | — | — | — | 14465.5704 | 13721.3295 |
| P09/last | complete | 0.7166 | 0.0293 | 121.6257 | 321.3940 | — | — | — | 14660.5142 | 13912.0785 |
| P10/best | complete | 0.7230 | 0.0064 | 123.7430 | 317.5503 | — | — | — | 14908.7145 | 14161.7951 |
| P10/last | complete | 0.7207 | 0.0042 | 124.4424 | 318.8500 | — | — | — | 15156.6398 | 14459.0918 |
| P11/best | complete | 0.6928 | 0.0023 | 132.2063 | 332.4547 | — | — | — | 15578.2104 | 14662.9213 |
| P11/last | complete | 0.6954 | 0.0081 | 130.9695 | 331.5710 | — | — | — | 15798.2196 | 14885.9321 |
| J00/best | complete | 0.6887 | -0.0340 | 139.7025 | 339.1217 | — | — | — | 17538.4685 | 16422.9293 |
| J00/last | complete | 0.6881 | -0.0325 | 139.1771 | 339.8801 | — | — | — | 17666.5644 | 16583.5531 |
| J01/best | complete | 0.7408 | 0.0521 | 122.4237 | 309.9390 | — | — | — | 13431.2750 | 12612.4993 |
| J01/last | complete | 0.7398 | 0.0517 | 122.3288 | 310.4806 | — | — | — | 13808.4249 | 12972.3766 |

## 速度（m/s，幅值及向量分别评价）

| ID/ckpt | 状态 | R²_cb | Δ父臂 | MAE | RMSE | 向量RMSE | 径向R² | 周向R² | 20mm去bias MSE | 40mm去bias MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R5V/best | 历史参照 | 0.7483 | — | 0.0947 | 0.1756 | 0.2144 | 0.2332 | 0.0023 | 0.0025 | 0.0023 |
| R5V/last | 历史参照 | 0.7431 | — | 0.0953 | 0.1775 | 0.2150 | 0.2430 | 0.0182 | 0.0026 | 0.0024 |
| V00/best | complete | 0.7516 | 0.0033 | 0.0942 | 0.1739 | 0.2147 | 0.2007 | -0.0366 | 0.0025 | 0.0022 |
| V00/last | complete | 0.7488 | 0.0057 | 0.0945 | 0.1748 | 0.2147 | 0.2067 | -0.0236 | 0.0025 | 0.0023 |
| V01/best | complete | 0.7618 | 0.0102 | 0.0931 | 0.1708 | 0.2084 | 0.2603 | 0.0983 | 0.0024 | 0.0022 |
| V01/last | complete | 0.7595 | 0.0107 | 0.0930 | 0.1717 | 0.2081 | 0.2724 | 0.1071 | 0.0024 | 0.0022 |
| V02/best | complete | 0.7603 | 0.0087 | 0.0920 | 0.1712 | 0.2134 | 0.2287 | -0.0156 | 0.0026 | 0.0024 |
| V02/last | complete | 0.7597 | 0.0109 | 0.0920 | 0.1714 | 0.2131 | 0.2321 | -0.0113 | 0.0025 | 0.0024 |
| V03/best | complete | 0.7632 | 0.0013 | 0.0916 | 0.1706 | 0.2083 | 0.2563 | 0.0903 | 0.0024 | 0.0022 |
| V03/last | complete | 0.7607 | 0.0012 | 0.0919 | 0.1714 | 0.2083 | 0.2696 | 0.0973 | 0.0025 | 0.0023 |
| V04/best | complete | 0.7537 | -0.0082 | 0.0935 | 0.1740 | 0.2101 | 0.2743 | 0.1318 | 0.0026 | 0.0024 |
| V04/last | complete | 0.7505 | -0.0090 | 0.0937 | 0.1751 | 0.2107 | 0.2776 | 0.1320 | 0.0026 | 0.0024 |
| V05/best | complete | 0.7534 | -0.0003 | 0.0934 | 0.1741 | 0.2095 | 0.2729 | 0.1110 | 0.0025 | 0.0023 |
| V05/last | complete | 0.7495 | -0.0010 | 0.0936 | 0.1755 | 0.2098 | 0.2816 | 0.1150 | 0.0025 | 0.0023 |
| V06/best | complete | 0.7656 | 0.0037 | 0.0929 | 0.1698 | 0.2108 | 0.2451 | 0.0929 | 0.0025 | 0.0023 |
| V06/last | complete | 0.7647 | 0.0052 | 0.0921 | 0.1700 | 0.2102 | 0.2556 | 0.1006 | 0.0025 | 0.0023 |
| V07/best | complete | 0.7762 | 0.0107 | 0.0899 | 0.1655 | 0.2055 | 0.2440 | 0.1064 | 0.0022 | 0.0020 |
| V07/last | complete | 0.7753 | 0.0106 | 0.0894 | 0.1657 | 0.2053 | 0.2566 | 0.1136 | 0.0022 | 0.0020 |
| V08/best | complete | 0.7693 | 0.0037 | 0.0907 | 0.1677 | 0.2077 | 0.2301 | 0.0805 | 0.0021 | 0.0019 |
| V08/last | complete | 0.7686 | 0.0039 | 0.0902 | 0.1678 | 0.2070 | 0.2425 | 0.0901 | 0.0021 | 0.0019 |
| V09/best | complete | 0.7665 | -0.0028 | 0.0907 | 0.1683 | 0.2055 | 0.2584 | 0.1019 | 0.0023 | 0.0021 |
| V09/last | complete | 0.7658 | -0.0028 | 0.0900 | 0.1686 | 0.2053 | 0.2697 | 0.1079 | 0.0023 | 0.0021 |
| V10/best | complete | 0.7675 | 0.0010 | 0.0912 | 0.1689 | 0.2056 | 0.2778 | 0.1031 | 0.0022 | 0.0020 |
| V10/last | complete | 0.7666 | 0.0008 | 0.0907 | 0.1692 | 0.2056 | 0.2844 | 0.1072 | 0.0022 | 0.0020 |
| V11/best | complete | 0.7758 | 0.0066 | 0.0898 | 0.1655 | 0.2081 | 0.2588 | 0.1114 | 0.0024 | 0.0022 |
| V11/last | complete | 0.7733 | 0.0046 | 0.0900 | 0.1665 | 0.2080 | 0.2713 | 0.1162 | 0.0024 | 0.0022 |
| J00/best | complete | 0.7332 | -0.0286 | 0.0950 | 0.1810 | 0.2171 | 0.1702 | 0.0510 | 0.0027 | 0.0025 |
| J00/last | complete | 0.7346 | -0.0249 | 0.0949 | 0.1807 | 0.2167 | 0.1709 | 0.0515 | 0.0027 | 0.0025 |
| J01/best | complete | 0.7565 | 0.0233 | 0.0915 | 0.1728 | 0.2130 | 0.1642 | 0.0260 | 0.0023 | 0.0020 |
| J01/last | complete | 0.7572 | 0.0226 | 0.0914 | 0.1726 | 0.2126 | 0.1677 | 0.0284 | 0.0023 | 0.0020 |

## 预登记筛选结果

| 臂/任务 | 精度筛选 | 长波筛选 | 未通过原因 |
|---|---|---|---|
| P00/pressure | False | False | best mae reduction<3% |
| P01/pressure | False | False | best ΔR²<0.01；best mae reduction<3%；last R² not improved |
| P02/pressure | True | True |  |
| P03/pressure | True | False |  |
| P04/pressure | False | False | best ΔR²<0.01；best mae reduction<3%；last R² not improved；last mae not improved |
| P05/pressure | False | False | best mae reduction<3% |
| P06/pressure | False | False | best ΔR²<0.01；best mae reduction<3%；last R² not improved；last mae not improved |
| P07/pressure | True | True |  |
| P08/pressure | False | False | best ΔR²<0.01；last R² not improved |
| P09/pressure | True | False |  |
| P10/pressure | False | False | best ΔR²<0.01；best mae reduction<3%；last mae not improved |
| P11/pressure | False | False | best ΔR²<0.01；best mae reduction<3%；last mae not improved |
| V00/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；best radial_r2 protection failed；best circ_r2 protection failed；last radial_r2 protection failed；last circ_r2 protection failed |
| V01/velocity | False | False | best vector_rmse reduction<3% |
| V02/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3% |
| V03/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；last vector_rmse not improved |
| V04/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；last R² not improved；last vector_rmse not improved |
| V05/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；last R² not improved；best circ_r2 protection failed；last circ_r2 protection failed |
| V06/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；last vector_rmse not improved；best radial_r2 protection failed；last radial_r2 protection failed |
| V07/velocity | False | False | best vector_rmse reduction<3% |
| V08/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；best radial_r2 protection failed；best circ_r2 protection failed；last radial_r2 protection failed；last circ_r2 protection failed |
| V09/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；last R² not improved |
| V10/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；last vector_rmse not improved |
| V11/velocity | False | False | best ΔR²<0.01；best vector_rmse reduction<3%；last vector_rmse not improved |
| J00/pressure | False | False | last ΔR²<0.01；last mae reduction<3%；last R² not improved；last mae not improved |
| J00/velocity | False | False | last ΔR²<0.01；last vector_rmse reduction<3%；last R² not improved；last vector_rmse not improved；last radial_r2 protection failed；last circ_r2 protection failed |
| J01/pressure | True | True |  |
| J01/velocity | False | False | best vector_rmse reduction<3%；best circ_r2 protection failed；last circ_r2 protection failed |

单seed、暴露test34仅支持开发筛选。FFN与BT残差拓扑不同；注意力激活不等于因果有效。
