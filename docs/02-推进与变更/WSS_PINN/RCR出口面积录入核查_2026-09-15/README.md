# RCR 出口面积录入核查（2026-09-15，只读，不占 GPU）

## 起因

X5D/X5 在 test34 上最差的病例 `AAA/unruputer/SUN_SHU_MING`：右髂外出口（网格里最大的出口，半径 5.7 mm，无狭窄）在 CFD 里几乎零流量，
模型在该支预测 1.77 Pa 对真值 0.33 Pa。用户重填出口面积后的 RCR 表与病例 UDF 对不上，于是把 172 例全部反查了一遍。

## 协议解码（从用户的表格 + 172 例 UDF 反推，`rcr_area_audit.py`）

| 项 | 规则 | 证据 |
|---|---|---|
| 总流量 | 0.033 kg/s，左右各半 | 表头 0.033，左侧两出口 flow 之和 0.0165 |
| 髂外/髂内分配 d | **Murray r³**，r 由出口面积 s 反算 | 5.6213³/(5.6213³+4.2436³)=0.6992 = 表内 d |
| R_t（每出口） | P_mean / Q_o | 1.076e6 × 0.01154 = 12.4 kPa ≈ 93 mmHg |
| R1 | 只依赖出口半径，R1 ∝ r^−2.24～−2.30 | 逐队列幂律拟合，内点残差 sd 0.008–0.020（log） |
| C | τ / R_t，τ = 1.790 s | 172 例 × 4 出口 p1/p99 = 1.790/1.790 |

结论：**CFD 出口协议本身就是 Murray 定律（r³）按出口面积分配**，唯一逐例输入是四个出口面积。
此前记忆里"髂外/髂内分配只有 28–78% 能由面积解释"不是操作者的临床判断，而是面积录入错误造成的。

## Fluent 里编号对不上的原因

表格里的 15967–15970（leftnei/leftwai/rightnei/rightwai）是切口原始面，拉伸出口延长段后它们变成了 `interior` 面；
真正挂 `pressure_out*::libudf` 的压力出口是 `*+` 面（rightwai+ 15958、rightnei+ 15957、leftwai+ 15971、leftnei+ 15959）。
两组面面积相同（用户重填的面积与网格出口面积一致到 0.1%），编号不是错误。R1/R2/C 写死在 `udf-inlet4.c` 的四个 `DEFINE_PROFILE(pressure_out*)` 里，Fluent 面板看不到数值。

## 方法

从每例 UDF 的 R1 反推表内隐含半径 r_impl = exp((ln R1 − a)/p)（逐队列幂律），与网格出口面积比较；|log10(面积比)| > log10(1.5) 判为录入错误；
比值 ≈ 10 或 0.1 判"小数点错一位"，与同侧另一出口面积一致判"互换/复制"。同时给出每侧髂外电导份额 UDF 值与按网格 r³ 应得值。
全表 `rcr_area_audit.csv`（172 × 4 出口）；峰值出口通量份额来自 case.h5（step 1162）。

## 需要重新处理的病例：首轮按单出口面积比 > 1.5 倍圈出 23 例（39 个出口；此前按份额偏差圈出的 12 例全部在内），后按"侧份额偏差 > 0.05"补 3 例，重跑名单 26 例（见文末）

| # | 病例 | 集合 | 类型 | 出错出口 | 表内隐含面积 → 应填网格面积 (m²) | 该侧髂外份额 UDF → 应为 | 此前 12 例 |
|---|---|---|---|---|---|---|---|
| 1 | `ILO/HOU_SHI_GUO-0/before` | train | A 小数点错一位 | out-ri | out-ri: 9.24e-05 → 9.22e-06 | r: 0.16 → 0.86 | 是 |
| 2 | `ILO/ZHANG_JIN_CHUN-1/before` | **test34** | A 小数点错一位 | out-ri | out-ri: 8.19e-05 → 8.18e-06 | r: 0.14 → 0.84 | 是 |
| 3 | `AAA/unruputer/GUO_BAO_CHUN` | train | A 小数点错一位 | out-li, out-ri | out-li: 2.40e-05 → 2.40e-04; out-ri: 2.25e-05 → 2.25e-04 | l: 0.89 → 0.20; r: 0.90 → 0.21 | 是 |
| 4 | `ILO/SUN_XU_XIA-1/before` | **test34** | A 小数点错一位 | out-le | out-le: 9.54e-05 → 9.52e-06 | l: 0.89 → 0.21 | 是 |
| 5 | `ILO/YANG_WANG_QI-1/before` | train | A 小数点错一位 | out-ri | out-ri: 8.00e-05 → 7.99e-06 | r: 0.10 → 0.78 | 是 |
| 6 | `AAA/ruputer/WU_GUANG_CUN` | train | A 小数点错一位 | out-le | out-le: 6.64e-05 → 6.64e-06 | l: 0.92 → 0.28 | 是 |
| 7 | `ILO/LIU_BAO_JUN-0/before` | train | A 小数点错一位 | out-ri | out-ri: 1.23e-05 → 1.24e-04 | r: 0.92 → 0.28 | 是 |
| 8 | `AAA/unruputer/SUN_SHU_MING` | **test34** | A 小数点错一位 | out-re | out-re: 1.03e-05 → 1.03e-04 | r: 0.06 → 0.68 | 是 |
| 9 | `AAA/ruputer/ZOU_LI_SHUN` | train | A 小数点错一位 | out-ri | out-ri: 2.47e-04 → 2.47e-05 | r: 0.03 → 0.51 | 是 |
| 10 | `AAA/ruputer/MENG_GUANG_QIN` | train | B 同侧髂内/外面积互换或复制 | out-le, out-li | out-le: 2.77e-05 → 9.36e-05; out-li: 9.37e-05 → 2.77e-05 | l: 0.14 → 0.86 | 是 |
| 11 | `AG/slow/CHENG_GUANG_SEN` | train | B 同侧髂内/外面积互换或复制 | out-re, out-ri | out-re: 3.29e-05 → 6.85e-05; out-ri: 6.88e-05 → 3.30e-05 | r: 0.25 → 0.75 | 是 |
| 12 | `ILO/ZHANG_YONG_SHENG-0/before` | **test34** | B 同侧髂内/外面积互换或复制 | out-le, out-li | out-le: 7.37e-05 → 4.43e-05; out-li: 4.43e-05 → 7.36e-05 | l: 0.68 → 0.32 |  |
| 13 | `AG/fast/LIU_YI_BING` | train | B 同侧髂内/外面积互换或复制 | out-re, out-ri | out-re: 2.63e-05 → 1.64e-05; out-ri: 1.63e-05 → 2.64e-05 | r: 0.67 → 0.33 |  |
| 14 | `AAA/ruputer/LIU_YONG_LAN` | train | B 同侧髂内/外面积互换或复制 | out-le, out-ri | out-le: 2.95e-05 → 4.71e-05; out-ri: 4.88e-05 → 2.84e-05 | l: 0.57 → 0.68; r: 0.42 → 0.70 |  |
| 15 | `AAA/ruputer/ZHOU_KE_XUN` | train | C 面积不符（来源待查） | out-le, out-li | out-le: 3.92e-05 → 7.15e-06; out-li: 1.23e-05 → 2.58e-05 | l: 0.85 → 0.13 | 是 |
| 16 | `AG/fast/LIU_JUN_FENG` | train | C 面积不符（来源待查） | out-le, out-li, out-re, out-ri | out-le: 6.59e-04 → 8.11e-05; out-li: 6.09e-04 → 3.44e-05; out-re: 5.89e-04 → 6.84e-05; out-ri: 6.45e-04 → 1.67e-05 | l: 0.53 → 0.78; r: 0.47 → 0.89 |  |
| 17 | `AG/fast/LIU_LI_QUN` | train | C 面积不符（来源待查） | out-ri | out-ri: 6.41e-05 → 2.89e-05 | r: 0.41 → 0.70 |  |
| 18 | `AG/fast/ZHANG_QING_WANG` | train | C 面积不符（来源待查） | out-ri | out-ri: 3.60e-05 → 1.80e-05 | r: 0.39 → 0.64 |  |
| 19 | `AAA/unruputer/LIU_XING_GUO` | train | C 面积不符（来源待查） | out-le, out-li, out-ri | out-le: 6.17e-05 → 3.57e-05; out-li: 4.92e-05 → 2.88e-05; out-ri: 6.34e-05 → 3.30e-05 | l: 0.58 → 0.58; r: 0.43 → 0.68 |  |
| 20 | `AG/fast/WANG_CHUN_MING` | train | C 面积不符（来源待查） | out-le, out-li, out-re, out-ri | out-le: 4.76e-04 → 5.01e-05; out-li: 4.33e-04 → 2.97e-05; out-re: 4.42e-04 → 4.47e-05; out-ri: 4.36e-04 → 3.25e-05 | l: 0.53 → 0.69; r: 0.50 → 0.62 |  |
| 21 | `AAA/unruputer/QIAO_XIU_YUN` | train | C 面积不符（来源待查） | out-li | out-li: 3.87e-05 → 2.52e-05 | l: 0.65 → 0.78 |  |
| 22 | `ILO/LIU_LIAN_YOU-0/before` | train | C 面积不符（来源待查） | out-ri | out-ri: 2.20e-05 → 1.29e-05 | r: 0.81 → 0.85 |  |
| 23 | `AAA/ruputer/ZUO_DAO_SHENG` | train | C 面积不符（来源待查） | out-le, out-li | out-le: 7.15e-05 → 3.91e-05; out-li: 2.59e-05 → 1.23e-05 | l: 0.82 → 0.85 |  |

另有 14 个出口比值在 1.2–1.5 之间（LIU_YONG_LAN re、WANG_KUI_WU li/ri、ZHOU_KE_XUN ri、ZUO_DAO_SHENG ri、YANG_BEN_RUI re、LIU_ZONG_YANG re/ri、LIU_LIAN_YOU-0 li/re、YU_XIANG_SHENG-1 四个），
未列入清单，可能是表格用了另一版网格的面积，建议顺手核对（见 CSV `area_ratio_R1` 列）。

## 后果（为什么必须重算）

- **×0.1 型**（SUN_SHU_MING、GUO_BAO_CHUN、LIU_BAO_JUN-0）：把大出口几乎关掉（峰值通量份额 −0.02～−0.03），该支 WSS 真值被压到 0.3 Pa 量级，几何上毫无线索。
- **×10 型**（ZHANG_JIN_CHUN-1、SUN_XU_XIA-1、HOU_SHI_GUO-0、YANG_WANG_QI-1、WU_GUANG_CUN、ZOU_LI_SHUN）：把 16–41% 的总流量硬推进半径 1.45–2.8 mm 的出口，
  出口平均速度 1.8–2.7 m/s，该末支峰值帧 WSS p99 85–123 Pa。test34 里决定物理 R² 符号的"ILO 射流病例"ZHANG_JIN_CHUN-1（真值峰 178 Pa）和 SUN_XU_XIA-1（275 Pa）正是这一型：
  尾部不是纯几何/流体现象，一半以上来自边界条件录入错误。§20.7 波 4"capfit 在 ILO −0.042"、§21.7 T3 尾部结论都要在重算后重读。
- **互换/复制型**与**面积不符型**：该侧分配偏离 r³ 规则 0.05–0.7，是分支尺度偏置。

## 重算步骤（每例）

1. 用 CSV 的 `area_mesh` 列（或 Fluent 里 `*+` 出口面的 Area 报告）重填表格四个面积，重新生成 R1/R2/C。
2. 改 `data_new/<队列>/<子集>/<病例>/udf-inlet4.c` 四个 `DEFINE_PROFILE(pressure_out*)` 块的 R1/R2/C，删掉 `libudf/` 让 Fluent 重新编译（或本机重编译后一起拷）。
3. 先把 `ascii/` 旧导出挪走（Fluent 导出不覆盖同名文件），再按原 `2.jou` 跑 8 周期并重新导出壁面/体场。
4. 全部跑完后刷新 V5 母库（`wss_v5/` 刷新工具，见 V5 数据母库记忆）与视图 sidecar，再用现有 X5D 五 seed 在新 test34 上重评。

## 表格公式（`计算R和C-质量流量-new计算结果.xlsx`，用户 2026-09-15 提供）与修正后的 R1/R2/C

公式：r = sqrt(s/π)·1000 mm；d = (2r)³/((2r_nei)³+(2r_wai)³)；flow = A1·d·0.5；Rt = 93.33·133/flow；C = 1.79/Rt；R1 = 13.3/(2r)^0.3/s；R2 = Rt − R1。
A1（总质量流量）逐例从 UDF 精确反推（R1 反解表内隐含半径 → 表内 d → A1 = 2P/(Rt·d)，四个出口一致到 1e-5）：AAA/ILO = 0.03300 kg/s，AG = 0.02842 kg/s。
唯一例外 `AAA/ruputer/GUO_AI_JUN`（train，不在名单）：左侧反推 0.033、右侧 0.0233，即右侧总阻力比协议大 42%，左侧电导占比 0.586 而非 0.5，是另一类录入异常，**未改**，登记待定。
用网格面积代回公式，录入正确的出口能复现 UDF 的 R1/R2/C；SUN_SHU_MING 右侧复现结果与用户重填的表格逐位一致。

**修正范围（用户 2026-09-15 拍板"应该改的和建议改的都改"）**：病例入选 = 任一出口面积录错，或任一侧 |UDF 髂外份额 − 网格 r³ 份额| > 0.05；入选病例里某侧偏差 > 0.03 的两个出口都重写，另一侧原样保留。
共 26 例、35 个侧。补入的 3 例：`AG/slow/LIU_ZONG_YANG`（右侧 re/ri 互换，+0.28）、`AAA/ruputer/WANG_KUI_WU`（左右髂内跨侧互换，±0.12）、`AAA/unruputer/YANG_BEN_RUI`（右 −0.10）；顺手补齐的另一侧：LIU_LIAN_YOU-0 左（+0.09）、ZHOU_KE_XUN 右（−0.04）、ZUO_DAO_SHENG 右（+0.04）。`ILO/YU_XIANG_SHENG-1`（test34）四出口同比例偏 1.2–1.4 倍、份额只差 ±0.02–0.03，不改不重跑。
下表（`rcr_corrected_values.py` → `rcr_corrected_udf_values.csv`）："←改"= 该侧重写；"(录错)"= 该出口面积本身填错；"（不动）"= 该侧原样。

| 病例 | A1 kg/s | 出口 | 应填面积 s (m²) | R1 | R2 | C | 髂外份额 d（原 UDF → 新） |
|---|---|---|---|---|---|---|---|
| `AAA/ruputer/LIU_YONG_LAN` | 0.0330 | out-le ←改(录错) / out-li ←改 | 4.7119e-05 / 2.8381e-05 | 1.5274E+05 / 2.7361E+05 | 9.5123E+05 / 2.0880E+06 | 1.6214E-06 / 7.5795E-07 | 0.575 → 0.681 |
| `AAA/ruputer/LIU_YONG_LAN` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 4.9657e-05 / 2.8433e-05 | 1.4379E+05 / 2.7304E+05 | 9.3445E+05 / 2.2156E+06 | 1.6601E-06 / 7.1926E-07 | 0.417 → 0.698 |
| `AAA/ruputer/MENG_GUANG_QIN` | 0.0330 | out-le ←改(录错) / out-li ←改(录错) | 9.3626e-05 / 2.7651e-05 | 6.9344E+04 / 2.8193E+05 | 8.0369E+05 / 5.1575E+06 | 2.0503E-06 / 3.2908E-07 | 0.138 → 0.862 |
| `AAA/ruputer/MENG_GUANG_QIN` | 0.0330 | out-re / out-ri | 2.6741e-05 / 3.6120e-05 | 2.9300E+05 / 2.0735E+05 | 1.6403E+06 / 1.0241E+06 | 9.2588E-07 / 1.4535E-06 | 0.384 → 0.389（不动） |
| `AAA/ruputer/WANG_KUI_WU` | 0.0330 | out-le ←改 / out-li ←改 | 3.4690e-05 / 3.0593e-05 | 2.1721E+05 / 2.5098E+05 | 1.1581E+06 / 1.4097E+06 | 1.3015E-06 / 1.0779E-06 | 0.663 → 0.547 |
| `AAA/ruputer/WANG_KUI_WU` | 0.0330 | out-re ←改 / out-ri ←改 | 3.0872e-05 / 2.2117e-05 | 2.4838E+05 / 3.6448E+05 | 9.6011E+05 / 1.6284E+06 | 1.4812E-06 / 8.9819E-07 | 0.503 → 0.623 |
| `AAA/ruputer/WU_GUANG_CUN` | 0.0330 | out-le ←改(录错) / out-li ←改 | 6.6377e-06 / 1.2537e-05 | 1.4548E+06 / 7.0013E+05 | 1.2504E+06 / 3.4199E+05 | 6.6169E-07 / 1.7177E-06 | 0.924 → 0.278 |
| `AAA/ruputer/WU_GUANG_CUN` | 0.0330 | out-re / out-ri | 1.8283e-05 / 1.9637e-05 | 4.5368E+05 / 4.1791E+05 | 1.1360E+06 / 1.0103E+06 | 1.1260E-06 / 1.2533E-06 | 0.473 → 0.473（不动） |
| `AAA/ruputer/ZHOU_KE_XUN` | 0.0330 | out-le ←改(录错) / out-li ←改(录错) | 7.1478e-06 / 2.5805e-05 | 1.3360E+06 / 3.0524E+05 | 4.5769E+06 / 5.5672E+05 | 3.0273E-07 / 2.0767E-06 | 0.851 → 0.127 |
| `AAA/ruputer/ZHOU_KE_XUN` | 0.0330 | out-re ←改 / out-ri ←改 | 1.7195e-05 / 2.3666e-05 | 4.8685E+05 / 3.3719E+05 | 1.4801E+06 / 8.8104E+05 | 9.1003E-07 / 1.4693E-06 | 0.342 → 0.382 |
| `AAA/ruputer/ZOU_LI_SHUN` | 0.0330 | out-le / out-li | 3.1531e-05 / 2.2619e-05 | 2.4242E+05 / 3.5519E+05 | 9.6698E+05 / 1.6353E+06 | 1.4801E-06 / 8.9929E-07 | 0.622 → 0.622（不动） |
| `AAA/ruputer/ZOU_LI_SHUN` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 2.5437e-05 / 2.4716e-05 | 3.1033E+05 / 3.2076E+05 | 1.1625E+06 / 1.2170E+06 | 1.2153E-06 / 1.1640E-06 | 0.032 → 0.511 |
| `AAA/ruputer/ZUO_DAO_SHENG` | 0.0330 | out-le ←改(录错) / out-li ←改(录错) | 3.9132e-05 / 1.2254e-05 | 1.8911E+05 / 7.1879E+05 | 6.9502E+05 / 4.3266E+06 | 2.0246E-06 / 3.5478E-07 | 0.821 → 0.851 |
| `AAA/ruputer/ZUO_DAO_SHENG` | 0.0330 | out-re ←改 / out-ri ←改 | 1.9575e-05 / 3.0321e-05 | 4.1943E+05 / 2.5358E+05 | 1.7831E+06 / 8.8895E+05 | 8.1268E-07 / 1.5667E-06 | 0.382 → 0.342 |
| `AAA/unruputer/GUO_BAO_CHUN` | 0.0330 | out-le ←改 / out-li ←改(录错) | 9.4673e-05 / 2.4018e-04 | 6.8463E+04 / 2.3469E+04 | 3.7237E+06 / 9.1500E+05 | 4.7203E-07 / 1.9074E-06 | 0.887 → 0.198 |
| `AAA/unruputer/GUO_BAO_CHUN` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 9.4402e-05 / 2.2518e-04 | 6.8689E+04 / 2.5275E+04 | 3.4552E+06 / 9.3122E+05 | 5.0797E-07 / 1.8714E-06 | 0.896 → 0.213 |
| `AAA/unruputer/LIU_XING_GUO` | 0.0330 | out-le ←改(录错) / out-li ←改(录错) | 3.5672e-05 / 2.8835e-05 | 2.1035E+05 / 2.6866E+05 | 1.0887E+06 / 1.5188E+06 | 1.3780E-06 / 1.0014E-06 | 0.584 → 0.579 |
| `AAA/unruputer/LIU_XING_GUO` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 5.4426e-05 / 3.2966e-05 | 1.2940E+05 / 2.3032E+05 | 9.7752E+05 / 2.1178E+06 | 1.6171E-06 / 7.6230E-07 | 0.428 → 0.680 |
| `AAA/unruputer/QIAO_XIU_YUN` | 0.0330 | out-le ←改 / out-li ←改(录错) | 5.7691e-05 / 2.5181e-05 | 1.2101E+05 / 3.1397E+05 | 8.4819E+05 / 3.0471E+06 | 1.8469E-06 / 5.3256E-07 | 0.646 → 0.776 |
| `AAA/unruputer/QIAO_XIU_YUN` | 0.0330 | out-re / out-ri | 6.3207e-05 / 3.7352e-05 | 1.0895E+05 / 1.9951E+05 | 9.8506E+05 / 2.2088E+06 | 1.6362E-06 / 7.4327E-07 | 0.688 → 0.688（不动） |
| `AAA/unruputer/SUN_SHU_MING` | 0.0330 | out-le / out-li | 9.9272e-05 / 5.6575e-05 | 6.4828E+04 / 1.2377E+05 | 1.0111E+06 / 2.3771E+06 | 1.6636E-06 / 7.1574E-07 | 0.699 → 0.699（不动） |
| `AAA/unruputer/SUN_SHU_MING` | 0.0330 | out-re ←改(录错) / out-ri ←改 | 1.0264e-04 / 6.1998e-05 | 6.2390E+04 / 1.1140E+05 | 1.0431E+06 / 2.2433E+06 | 1.6192E-06 / 7.6017E-07 | 0.063 → 0.681 |
| `AAA/unruputer/YANG_BEN_RUI` | 0.0330 | out-le / out-li | 6.6451e-05 / 5.7045e-05 | 1.0286E+05 / 1.2259E+05 | 1.2478E+06 / 1.5756E+06 | 1.3253E-06 / 1.0541E-06 | 0.557 → 0.557（不动） |
| `AAA/unruputer/YANG_BEN_RUI` | 0.0330 | out-re ←改 / out-ri ←改 | 8.7776e-05 / 7.7316e-05 | 7.4685E+04 / 8.6418E+04 | 1.2995E+06 / 1.5759E+06 | 1.3026E-06 / 1.0768E-06 | 0.451 → 0.547 |
| `AG/fast/LIU_JUN_FENG` | 0.0284 | out-le ←改(录错) / out-li ←改(录错) | 8.1088e-05 / 3.4445e-05 | 8.1811E+04 / 2.1899E+05 | 1.0335E+06 / 3.8096E+06 | 1.6049E-06 / 4.4433E-07 | 0.528 → 0.783 |
| `AG/fast/LIU_JUN_FENG` | 0.0284 | out-re ←改(录错) / out-ri ←改(录错) | 6.8385e-05 / 1.6658e-05 | 9.9520E+04 / 5.0495E+05 | 8.7900E+05 / 7.6342E+06 | 1.8293E-06 / 2.1993E-07 | 0.467 → 0.893 |
| `AG/fast/LIU_LI_QUN` | 0.0284 | out-le / out-li | 4.1000e-05 / 3.4823e-05 | 1.7923E+05 / 2.1626E+05 | 1.3779E+06 / 1.7732E+06 | 1.1495E-06 / 8.9976E-07 | 0.561 → 0.561（不动） |
| `AG/fast/LIU_LI_QUN` | 0.0284 | out-re ←改 / out-ri ←改(录错) | 5.0623e-05 / 2.8913e-05 | 1.4064E+05 / 2.6783E+05 | 1.1099E+06 / 2.6293E+06 | 1.4314E-06 / 6.1785E-07 | 0.414 → 0.699 |
| `AG/fast/LIU_YI_BING` | 0.0284 | out-le / out-li | 1.7921e-05 / 3.1817e-05 | 4.6425E+05 / 2.3991E+05 | 2.4756E+06 / 1.0028E+06 | 6.0887E-07 / 1.4404E-06 | 0.307 → 0.297（不动） |
| `AG/fast/LIU_YI_BING` | 0.0284 | out-re ←改(录错) / out-ri ←改(录错) | 1.6398e-05 / 2.6382e-05 | 5.1415E+05 / 2.9759E+05 | 2.1417E+06 / 1.0040E+06 | 6.7397E-07 / 1.3753E-06 | 0.668 → 0.329 |
| `AG/fast/WANG_CHUN_MING` | 0.0284 | out-le ←改(录错) / out-li ←改(录错) | 5.0138e-05 / 2.9734e-05 | 1.4221E+05 / 2.5935E+05 | 1.1302E+06 / 2.5269E+06 | 1.4068E-06 / 6.4245E-07 | 0.534 → 0.686 |
| `AG/fast/WANG_CHUN_MING` | 0.0284 | out-re ←改(录错) / out-ri ←改(录错) | 4.4693e-05 / 3.2544e-05 | 1.6231E+05 / 2.3376E+05 | 1.2540E+06 / 2.0455E+06 | 1.2639E-06 / 7.8533E-07 | 0.505 → 0.617 |
| `AG/fast/ZHANG_QING_WANG` | 0.0284 | out-le / out-li | 2.1750e-05 / 3.3114e-05 | 3.7157E+05 / 2.2914E+05 | 2.1429E+06 / 1.1094E+06 | 7.1188E-07 / 1.3373E-06 | 0.360 → 0.347（不动） |
| `AG/fast/ZHANG_QING_WANG` | 0.0284 | out-re ←改 / out-ri ←改(录错) | 2.6463e-05 / 1.8024e-05 | 2.9654E+05 / 4.6121E+05 | 1.0680E+06 / 1.9663E+06 | 1.3118E-06 / 7.3737E-07 | 0.388 → 0.640 |
| `AG/slow/CHENG_GUANG_SEN` | 0.0284 | out-le / out-li | 6.7644e-05 / 3.6026e-05 | 1.0078E+05 / 2.0798E+05 | 1.1122E+06 / 2.9130E+06 | 1.4757E-06 / 5.7354E-07 | 0.719 → 0.720（不动） |
| `AG/slow/CHENG_GUANG_SEN` | 0.0284 | out-re ←改(录错) / out-ri ←改(录错) | 6.8463e-05 / 3.3010e-05 | 9.9390E+04 / 2.2997E+05 | 1.0666E+06 / 3.2525E+06 | 1.5352E-06 / 5.1400E-07 | 0.254 → 0.749 |
| `AG/slow/LIU_ZONG_YANG` | 0.0284 | out-le / out-li | 5.0696e-05 / 2.1198e-05 | 1.4041E+05 / 3.8273E+05 | 9.6927E+05 / 3.7215E+06 | 1.6131E-06 / 4.3614E-07 | 0.780 → 0.787（不动） |
| `AG/slow/LIU_ZONG_YANG` | 0.0284 | out-re ←改 / out-ri ←改 | 1.5831e-05 / 2.3293e-05 | 5.3542E+05 / 3.4340E+05 | 1.8971E+06 / 1.0195E+06 | 7.3585E-07 / 1.3134E-06 | 0.640 → 0.359 |
| `ILO/HOU_SHI_GUO-0/before` | 0.0330 | out-le / out-li | 2.5356e-05 / 1.8669e-05 | 3.1147E+05 / 4.4291E+05 | 9.1611E+05 / 1.5001E+06 | 1.4582E-06 / 9.2124E-07 | 0.613 → 0.613（不动） |
| `ILO/HOU_SHI_GUO-0/before` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 3.0108e-05 / 9.2236e-06 | 2.5564E+05 / 9.9651E+05 | 6.2421E+05 / 4.1925E+06 | 2.0344E-06 / 3.4496E-07 | 0.157 → 0.855 |
| `ILO/LIU_BAO_JUN-0/before` | 0.0330 | out-le / out-li | 6.1113e-05 / 6.3192e-05 | 1.1326E+05 / 1.0898E+05 | 1.4300E+06 / 1.3588E+06 | 1.1599E-06 / 1.2195E-06 | 0.487 → 0.487（不动） |
| `ILO/LIU_BAO_JUN-0/before` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 6.5750e-05 / 1.2358e-04 | 1.0412E+05 / 5.0395E+04 | 2.5866E+06 / 9.9386E+05 | 6.6525E-07 / 1.7141E-06 | 0.925 → 0.280 |
| `ILO/LIU_LIAN_YOU-0/before` | 0.0330 | out-le ←改 / out-li ←改 | 4.4514e-05 / 2.9074e-05 | 1.6306E+05 / 2.6613E+05 | 9.8633E+05 / 1.9114E+06 | 1.5573E-06 / 8.2204E-07 | 0.748 → 0.655 |
| `ILO/LIU_LIAN_YOU-0/before` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 4.0898e-05 / 1.2914e-05 | 1.7974E+05 / 6.7669E+05 | 7.0603E+05 / 4.3154E+06 | 2.0208E-06 / 3.5856E-07 | 0.806 → 0.849 |
| `ILO/SUN_XU_XIA-1/before` | 0.0330 | out-le ←改(录错) / out-li ←改 | 9.5244e-06 / 2.3336e-05 | 9.6040E+05 / 3.4268E+05 | 2.6771E+06 / 6.0578E+05 | 4.9210E-07 / 1.8873E-06 | 0.892 → 0.207 |
| `ILO/SUN_XU_XIA-1/before` | 0.0330 | out-re / out-ri | 1.0423e-05 / 1.9680e-05 | 8.6586E+05 / 4.1686E+05 | 1.8384E+06 / 6.2538E+05 | 6.6193E-07 / 1.7175E-06 | 0.278 → 0.278（不动） |
| `ILO/YANG_WANG_QI-1/before` | 0.0330 | out-le / out-li | 2.0450e-05 / 1.7162e-05 | 3.9885E+05 / 4.8793E+05 | 9.3180E+05 / 1.2429E+06 | 1.3452E-06 / 1.0342E-06 | 0.565 → 0.565（不动） |
| `ILO/YANG_WANG_QI-1/before` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 1.8308e-05 / 7.9940e-06 | 4.5298E+05 / 1.1747E+06 | 5.1637E+05 / 2.1849E+06 | 1.8466E-06 / 5.3279E-07 | 0.099 → 0.776 |
| `ILO/ZHANG_JIN_CHUN-1/before` | 0.0330 | out-le / out-li | 4.4459e-05 / 2.3683e-05 | 1.6329E+05 / 3.3690E+05 | 8.8149E+05 / 2.3503E+06 | 1.7133E-06 / 6.6611E-07 | 0.720 → 0.720（不动） |
| `ILO/ZHANG_JIN_CHUN-1/before` | 0.0330 | out-re ←改 / out-ri ←改(录错) | 2.4784e-05 / 8.1825e-06 | 3.1976E+05 / 1.1437E+06 | 5.7525E+05 / 3.5742E+06 | 2.0000E-06 / 3.7941E-07 | 0.143 → 0.841 |
| `ILO/ZHANG_YONG_SHENG-0/before` | 0.0330 | out-le ←改(录错) / out-li ←改(录错) | 4.4281e-05 / 7.3617e-05 | 1.6405E+05 / 9.1430E+04 | 2.2009E+06 / 1.0118E+06 | 7.5690E-07 / 1.6225E-06 | 0.682 → 0.318 |
| `ILO/ZHANG_YONG_SHENG-0/before` | 0.0330 | out-re / out-ri | 5.8050e-05 / 2.7845e-05 | 1.2015E+05 / 2.7968E+05 | 8.8205E+05 / 2.7372E+06 | 1.7861E-06 / 5.9334E-07 | 0.751 → 0.751（不动） |

## 批量写入结果（2026-09-15，`apply_udf_corrections.py` → `apply_log.md`；核对 `verify_udf_corrections.py`）

- **26 例全部已写入**（首轮 23 例 + 补 3 例；首轮 A1 用"正确侧"推断的病例已按精确 A1 重写 R2/C，差 0.3% 以内）各自的 `udf-inlet4.c` / `udf-inlet.c`（AG 为 `udf-inlet.c`），原文件备份为同目录 `*.orig_20260915`，换行风格（CRLF/LF）原样保留。
- 只重写有出口录错的那一侧的两个 `DEFINE_PROFILE(pressure_out*)` 块（R1/R2/C 三行 + 块首一行 ASCII 注释记录网格面积、A1 与旧值），另一侧原样不动。
- 核对：改后数值 == `rcr_corrected_udf_values.csv`；去掉数值行和注释行后与备份逐字节相同（26/26 通过，`apply_log.md`）。`libudf/src/` 里的编译副本没动，Fluent 启动时会重新拷贝源码并编译。
- gcc 语法预检跑不通（Fluent 头文件链缺 `string_safe.h`，原文件同样跑不通），所以真正的编译检查在 Fluent 启动时完成：看 transcript 里 `libudf` 编译段是否 `Done.`、四个 `pressure_out*` 是否列出。
- 两例 AG（LIU_JUN_FENG、WANG_CHUN_MING）四个出口都填成了约 4–6e-4 m²（≈入口面积），A1 由各自 UDF 精确反推为 0.02842 kg/s（四出口一致），已按此写入。

**用户还要做的**

1. 重跑前每例执行 `bash prepare_rerun.sh <病例目录>`（或 `bash prepare_rerun.sh $(cat rerun_cases.txt)`）：把 `ascii/ ascii_in/ libudf/ Global_conditions/` 挪成 `*_old_20260915`；然后 `export PATH=/public/slurm/bin:$PATH && sbatch fluent.slurm`（在病例目录内）。
2. ~~可选决定~~ **已做（2026-09-15，用户拍板）**：`AAA/ruputer/LIU_YONG_LAN` 入口 UDF 的面积除数 0.0004121751 → `0.00037587005`（网格 inlet+ 面积；旧值大 9.7%，入口速度此前偏低 9%），第 47 行，带注释，备份仍是 `.orig_20260915`。
3. ~~边缘出口待定~~ **已做（用户 2026-09-15 拍板）**：按"侧份额偏差"补改 LIU_ZONG_YANG、WANG_KUI_WU、YANG_BEN_RUI，并补齐 LIU_LIAN_YOU-0 / ZHOU_KE_XUN / ZUO_DAO_SHENG 未改的一侧；YU_XIANG_SHENG-1 不动。含义见上文"修正范围"。
4. 全部跑完后刷新 V5 母库与 sidecar，再用 X5D 五 seed 在新 test34 重评，并重读跟踪文档 §20.7 / §21.7 的尾部结论。

## 路径统一（2026-09-15，`fix_paths.py` → `fix_paths_log.md`）

26 例里 17 例的 `2.jou` read-case 路径、16 例的 `.cas.gz` 内部绝对路径（ascii 导出前缀、autosave 根名、UDF 源文件）还指向已不存在的 `GNN/data/…`（含 `data/ag/`、`data/ILO/ILO/`、`CHENG_GUAN_G_SEN`、`//public` 等旧写法）。
已全部改为当前病例目录的绝对路径，文件名不变；备份 `2.jou.orig_20260915`、`<case>.cas.gz.orig_20260915`（.cas 只替换 scheme 文本里的路径串，二进制网格段不含这些串，重压缩后回读逐字节一致）。

**.cas 里两条自动导出的语义与去向（26 例逐一核对，路径异常 0）**：

| 导出 | 内容 | `file-name` | 产物 | fluent.slurm 里的去向 |
|---|---|---|---|---|
| export-N（体场） | `(cellzones blood)`，无 surfaces；压力、速度分量等 | `<病例目录>/<case>` | 病例目录下 `<case>-NNNN`，每步一个 | 步骤 1–2 按序号删/留（留 1120 与 1121–1280 偶数号），步骤 4 `mv *-NNNN ascii_in/` |
| export-N+1（壁面） | `(surfaces wall)`；压力、WSS 及分量 | `<病例目录>/ascii/<case>` | `ascii/<case>-NNNN` | 步骤 6 在 ascii/ 内同样规则删留（AG 的 7 例脚本只删 1–1119、LIU_LI_QUN/LIU_YI_BING 无步骤 6，会多留文件，不影响后处理） |

`ascii/` 目录必须在 Fluent 启动前存在（导出不会建目录），所以 `prepare_rerun.sh` 在把旧 `ascii/` 挪成 `ascii_old_20260915` 后会重建一个空的 `ascii/`；`ascii_in/`、`Global_conditions/` 由 fluent.slurm 自己 `mkdir -p`。
fluent.slurm 全部用相对路径（`cd $SLURM_SUBMIT_DIR`、`./2.jou`、`./ascii_in/`），不需要改。

## 冒烟测试（2026-09-15，Slurm 14288，node06，`smoke_readcase.slurm` → `smoke_14288.out`）

只 read-case 再退出，不迭代：验证改过路径的 .cas.gz 能读、改过的 UDF 能被 Fluent 自动编译并挂上四个出口钩子。
- `AAA/ruputer/LIU_YONG_LAN`（.cas 路径改过 + CRLF 的 udf-inlet4.c 改出口与入口）：从 data_new 路径读入 → Copy 源码 → Compiling → `Done.` → Opening library → 列出 pressure_outle/li/ri/re → 退出码 0。
- `AG/fast/WANG_CHUN_MING`（.cas 路径改过 + LF 的 udf-inlet.c 四出口全改）：同上，退出码 0。
- 编译只有原来就有的 unused-variable 警告；`Clock skew detected` 是 node06 时钟比文件系统慢约 150 s，构建仍完成，不影响。两例的 `libudf/` 已是用修正后源码新编的（重跑前 `prepare_rerun.sh` 仍会把它挪走再让 Fluent 重编一次，无妨）。
- **注意集群状态**：node01、node04 自 2026-09-09 起 `down (Not responding)`；26 例里 17 例的 `fluent.slurm` 写死 `#SBATCH -w node01`，直接提交会一直 PD。要么等 node01 恢复，要么把 `-w node01` 改成可用节点（本次冒烟用的 node06 当时空闲）或删掉这一行让 Slurm 自选。

## 重跑启动（2026-09-15，用户拍板：核数统一 64、节点 node06、先并行 3 例检查输出）

- 26 例 `fluent.slurm` 统一改为 `#SBATCH -w node06`、`--ntasks-per-node=64`、`fluent … -t64`（原件备份 `fluent.slurm.orig_20260915`；此前 17 例写死 node01、AG 用 48 核、两例 ILO 用 40 核）。其余步骤（导出文件删留、搬 ascii_in/、搬 Global_conditions/）未动。
- 第 1 批（`prepare_rerun.sh` 已挪走旧 ascii/ ascii_in/ libudf/ Global_conditions/ → `*_old_20260915` 并重建空 ascii/）：见 `rerun_jobs.md`。选这三例是为了覆盖三个队列、两种 UDF 格式（CRLF udf-inlet4.c / LF udf-inlet.c）、两种 slurm 变体，且 SUN_SHU_MING、ZHANG_JIN_CHUN-1 都是 test34 最差病例。
- 资源：node06 共 192 核，3 × 64 正好占满，第 2 批要等第 1 批结束。磁盘：体场导出每步约 130 MB，1280 步在 fluent.slurm 清理前峰值约 170 GB/例，三例并行约 0.5 TB；`/public` 现余 3.2 TB。
- 检查输出看什么：病例目录 `Fluent_<jobid>.out`（跑完会被搬进 Global_conditions/）里 libudf 编译 `Done.` 与四个 pressure_out* 钩子；跑完后 `ascii/` 与 `ascii_in/` 各应有 1120 号 + 1122–1280 偶数号共 81 个文件；`Global_conditions/` 里 `p-out{le,li,re,ri}-rfile.out` 四个出口压力、`report-def-*` 流量；对 SUN_SHU_MING 重点看右髂外（out-re）流量份额是否从 −0.02 回到约 0.3。

## 第 1 批中途核对（2026-09-15，跑到 145–217 步时，`check_rerun_flowshares.py`）

UDF 每步打印四个出口的质量流量，按 0.2 s 后的累计份额对比旧运行与新运行（UDF 打印的出口面积与网格面积一致）：

| 病例 | 侧 | UDF 设定 旧 → 新 | CFD 运行 旧 → 新 |
|---|---|---|---|
| SUN_SHU_MING | 右侧髂外占该侧 | 0.063 → 0.681 | 0.059 → 0.699（out-re 占总流量 0.03 → 0.35） |
| ZHANG_JIN_CHUN-1 | 右侧髂外占该侧 | 0.143 → 0.841 | 0.113 → 0.843（半径 1.6 mm 的 out-ri 占总流量 0.39 → 0.07，射流消失） |
| WANG_CHUN_MING | 左 / 右侧髂外占该侧 | 0.534 → 0.686 / 0.505 → 0.617 | 0.552 → 0.691 / 0.517 → 0.618 |

左右各半核对：SUN_SHU_MING 0.501、WANG_CHUN_MING 0.501、ZHANG_JIN_CHUN-1 0.535（ILO 右侧出口很小，三维阻力本身偏大，合理）。**结论：新运行的分流与修正后的 UDF 一致。**
速度：64 核约 11–17 步/分钟，单例 1280 步约 1.3–2 小时。

**第 2 批（node03，用户拍板"6 例并行"）**：SUN_XU_XIA-1 14303、ZHANG_YONG_SHENG-0 14304、GUO_BAO_CHUN 14305（这三例 fluent.slurm 的 `-w` 改为 node03）。以后每例跑完可用 `check_rerun_flowshares.py <病例目录>` 复核。

**第 3 批（用户拍板"剩下 20 例也提交"）**：20 例已全部 `prepare_rerun.sh` 并提交，`-w` 按提交顺序交替 node06 / node03（各 10 例），Slurm 在各节点腾出 64 核时自动启动下一例，每节点始终 3 例并行；作业号 14306–14325 见 `rerun_jobs.md`。26 例全部排完约需 26 ÷ 6 × 1.5–2 h ≈ 7–9 小时。
跑完后：`python check_rerun_flowshares.py` 复核全部 26 例分流；再刷新 V5 母库 / sidecar，X5D 五 seed 重评新 test34。
