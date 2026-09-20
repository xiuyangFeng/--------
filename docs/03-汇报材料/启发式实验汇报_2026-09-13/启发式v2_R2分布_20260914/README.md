# 病例R²分布与选例清单（2026-09-14）

每组提供横向排名散点 + 同宽直方图；最好/最常见/最差从该组原始预测R²选择。负R²全部保留。
V5固定v2所用seed1234/7/2025：一例一个点，横须为三seed病例R²的样本标准差；不是102例或集成预测。
最常见：0为锚点、箱宽0.10；并列最高频箱取距中位数最近者，再取较低箱。代表取该箱内距中心最近病例；ID打破并列。
此定义不等于中位病例；分箱敏感性（0.05/0.10/0.20）保存在summary.json，不能称唯一的典型病例。

| 组 | 例数 | 病例R²均值 / 中位 / P10 | 负R² | 最差 | 最常见区间代表 | 最好 |
|---|---:|---|---:|---|---|---|
| A_X5 | 34 | 0.7339 / 0.7504 / 0.6045 | 0 | ILO/ZHANG_JIN_CHUN-1/before (0.5177) | AAA/ruputer/YU_TIAN_HAI (0.7473) | AG/fast/ZHANG_LIANG (0.8933) |
| A_X5X11 | 34 | 0.7357 / 0.7457 / 0.6369 | 0 | AAA/unruputer/SUN_SHU_MING (0.4666) | AG/fast/YAO_CUN_HONG (0.7484) | AG/fast/ZHANG_LIANG (0.8777) |
| A_X5D_v51 | 34 | 0.7619 / 0.7612 / 0.6542 | 0 | ILO/ZHANG_YONG_SHENG-0/before (0.6021) | AG/slow/HE_SHU_ZHEN (0.7503) | AG/fast/ZHANG_LIANG (0.9043) |
| B_VF6 | 34 | 0.7760 / 0.8037 / 0.6390 | 0 | AAA/unruputer/SUN_SHU_MING (0.4813) | ILO/ZHANG_YONG_SHENG-0/before (0.8436) | AG/fast/ZHANG_LIANG (0.9168) |
| B_VF6_wss | 34 | 0.4651 / 0.4879 / 0.3006 | 1 | AAA/ruputer/KANG_YONG (-0.2173) | AG/slow/LI_HUAN_GE (0.4489) | AG/fast/RAN_QING_BO (0.7562) |
| B_PF6 | 34 | 0.8130 / 0.8496 / 0.6009 | 0 | AAA/unruputer/SUN_SHU_MING (0.3755) | AG/slow/GUO_XI_JIANG (0.9544) | AG/fast/YAO_CUN_HONG (0.9650) |
| B_R5V_speed | 34 | 0.7066 / 0.7299 / 0.5178 | 0 | AAA/ruputer/KANG_YONG (0.3537) | AG/fast/YAO_CUN_HONG (0.8638) | AAA/ruputer/LI_ZHEN_HUA (0.8901) |
| B_R5P_pressure | 34 | 0.5828 / 0.7297 / 0.3364 | 3 | AAA/ruputer/KANG_YONG (-1.6748) | AAA/unruputer/ZHANG_YONG_ZHI (0.7494) | AAA/ruputer/LI_ZHEN_HUA (0.9570) |
| B_R5V_wss | 34 | 0.3796 / 0.4256 / 0.1709 | 1 | AAA/ruputer/KANG_YONG (-1.3416) | AAA/unruputer/WANG_MAN_TIAN (0.4490) | AG/fast/LOU_YANG (0.7490) |
| B_CFD_oracle | 34 | 0.9598 / 0.9649 / 0.9334 | 0 | AG/slow/ZHANG_WEI_XIAN (0.9060) | AG/slow/GUO_XI_JIANG (0.9501) | ILO/YANG_WEN_TAI-0/before (0.9842) |
| B_R5V_physics | 34 | 0.4252 / 0.4411 / 0.2828 | 1 | AAA/ruputer/KANG_YONG (-0.5593) | ILO/LI_YOU_ZHI-0/before (0.3486) | AG/fast/LOU_YANG (0.7063) |
| C_V4_speed | 35 | -0.5134 / -0.4736 / -0.9620 | 34 | AG/fast/SUN_ZHI_YU (-1.2185) | AG/fast/ZHANG_CHUN (-0.4397) | AAA/ruputer/WANG_FU_SHUN (0.0252) |
| C_V4_pressure | 35 | 0.1110 / 0.0785 / 0.0130 | 1 | AAA/ruputer/WANG_FU_SHUN (-1.1708) | AG/fast/FAN_JIAN_MING (0.0519) | ILO/LU_FU_SHAN-0/before (0.7195) |
| C_V4_wss | 35 | -0.1051 / -0.0403 / -0.4400 | 20 | AG/fast/LOU_YANG (-0.8893) | AG/slow/GUO_XI_JIANG (0.0567) | AG/fast/RAN_QING_BO (0.2599) |

## 使用方式

当前X5/X5X11/VF6/PF6、历史R5诊断链、历史V4 PINN均分组标识。test34与test35、三seed与单seed不作同条件冠军排名。
wave4已完成，X5保留为底座、五seed部署集成R²_cb=0.7354；本批仍固定v2三个seed，集成总体成绩另列。
每页CSV包括原始病例ID、各seed R²、病例均值/SD、可用MAE/NMAE和定义；跨版本NMAE分母不同，不直接合并。
代表病例云图应与所展示run/checkpoint对应。三seed均值选例时，若只画s1234云图，必须标s1234病例R²；均值不冒充单seed或集成场的分数。
已有跨目标共同病例保留用于横向比较；本批每目标自己的最好/最常见/最差用于诊断，二者是不同选例任务。
V4源文件名为last_converged，实际记录epoch9999、converged:false；不能称“已收敛模型”。V4 test35比V5 test34多AAA/ruputer/YANG_BAO_KUI。
原V4横向示例按每组min/max划5箱，本批按0锚定、宽0.10分箱，因此橙色Most代表可能变化，Best/Worst仍取真实预测R²极值。
历史目录另有拟合直线R²图；本批只读原始预测R²，未用相关系数平方或拟合线R²替代。
病例预测R²=1−Σ(Pred−CFD)²/Σ(CFD−病例CFD均值)²；横轴越右越好。负值表示在该指标上比预测该病例真值均值更差。
每组各有*_distribution.png及*_cases.csv；representative_cases.csv含模型/目标/角色选例记录，病例可重复。summary.json保存统计、分箱、源路径与SHA256。
重绘全套：python3 build_distributions.py。只补 VELWSS2：python3 build_velwss2_distribution.py。只补 X5D_v51：python3 build_x5d_v51_distribution.py。仅读取现有JSON，不调用训练或推理。

## 2026-09-16 增补：B_VF6_wss

VELWSS2（VF6 三 seed 预测速度 → 冻结 Profile-Secant V3 派生 WSS，legacy 半径校准路径）。选例与上表同一规则；后处理 best/median/worst 的 Median 是距分布中位最近的真实病例，本批为 `ILO/LU_FU_SHAN-0/before`，与直方图橙色 Most 代表不同。场文件来自 s1234，不是三 seed 平均场。

## 2026-09-17 增补：A_X5D_v51

v5.1 数据更新后已跑完的密度增广 X5D（五 seed 1234/7/2025/11/2026，ckpt_best，test34）。选例与上表同一规则；后处理 best/median/worst 的 Median 是距分布中位最近的真实病例，本批为 `AAA/ruputer/KANG_YONG`，与直方图橙色 Most 代表不同。场文件来自 s1234，不是五 seed 平均场。不是纵向几何 X5D_long。
