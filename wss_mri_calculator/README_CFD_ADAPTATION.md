# 适配说明：用本项目 CFD 数据跑 velocity → WSS

原仓库（`wss_mri_calculator`）面向 **4D Flow MRI 规则体素影像**。本项目 `data_new/`
是 **Fluent 非结构化点云**，因此新增了 CFD 适配核心与可视化脚本，原有 MRI 文件未改动：

| 文件 | 作用 |
| --- | --- |
| `src/data_loader_cfd.py` | 读 Fluent ASCII 导出（替代原 `data_loader.py` 的 HDF5 读取） |
| `src/calculate_wss_cfd.py` | 单病例算 WSS 并与 CFD 真值对比（替代 `calculate_wss.py`） |
| `src/batch_validate_cfd.py` | 跨队列批量验证 |
| `src/wss_multiscale.py` | 纯点云多尺度零带宽修正候选 v2 |
| `src/compare_multiscale_v2.py` | v1/v2 成对验证 |
| `viz/` | 诊断/审计可视化（pred–truth、法向、邻域锚定、采样诊断、postview 导出）；不放在 `src/` |

## 病例范围（必须先读）

`data_new/` 全库存在导出损坏/缺列/拓扑异常病例。**正式 velocity→WSS 对比
只允许使用 WSS-PINN 已通过审计并冻结的 173 例**，不得再对整个 `data_new`
做随机抽样当结论。

| 项目 | 冻结口径 |
| --- | --- |
| Split | `wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json` |
| SHA256 | `964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b` |
| 计数 | train **138** + test **35** = **173** |
| 排除 | `AAA/ruputer/SHI_YUN_XI`（体域速度全零，PINN 不可用） |
| 上位记录 | `docs/02-推进与变更/WSS_PINN/README.md`（D-014 / source audit 173/173） |

`batch_validate_cfd.py` **默认绑定上述 split**；只有显式加 `--discover-all`
才会扫描全库（仅用于找坏导出，**不能**写成正式 raw R²）。

## 快速开始

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd wss_mri_calculator/src

# 单病例（默认自动读 data_wss_min bundle 的 peak_step）
$PY calculate_wss_cfd.py \
  --case-dir /public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LIU_JIN_LIANG \
  --plot-out ../../outputs/wss_pinn/audits/cfd_velocity_wss_explore/demo_scatter.png

# 扫参数（K × 阶数 × 粘度模型）
$PY calculate_wss_cfd.py --case-dir <病例目录> --sweep

# 正式批量：PINN 冻结 173 例（峰值步，默认 adaptive-CV）
$PY batch_validate_cfd.py \
  --json-out ../../outputs/wss_pinn/audits/cfd_velocity_wss_explore/pinn173_peak.json

# 快速抽检：仍来自该 173 例 split，不是全库
$PY batch_validate_cfd.py --limit-per-cohort 3

# 仅 test35
$PY batch_validate_cfd.py --roles test

# 复现旧 baseline
$PY batch_validate_cfd.py --neighbor-mode fixed --neighbors 64
```

依赖：`numpy` / `scipy` / `matplotlib`（出图时）即可（**不需要 pyvista**）。
`GNN` conda 环境已满足。

## 探索审计图目录

单病例深度图（全量 pred/truth、法向定向、邻域锚定、采样诊断）统一放在：

`outputs/wss_pinn/audits/cfd_velocity_wss_explore/`

阅读入口：[该目录 README](../outputs/wss_pinn/audits/cfd_velocity_wss_explore/README.md)
（按 `01`–`06` 主题划分；旧 2000 点抽样散点在 `_legacy_root/`）。

出图脚本在 `viz/`（从该目录运行，会自动找到 `src/` 里的算子）：

```bash
cd wss_mri_calculator/viz
$PY plot_pred_truth_full.py --case-dir <病例目录> --sample-count 0
$PY visualize_normal_orientation.py --case-dir <病例目录>
$PY visualize_neighbor_anchor.py --case-dir <病例目录>
$PY diagnose_wss_sampling.py --case-dir <病例目录>
$PY export_velocity_wss_postview.py --case-dir <病例目录>
```

## 数据格式

```
<病例目录>/
  ascii/<前缀>-<step>       壁面节点：nodenumber,x,y,z,pressure,wall-shear,x/y/z-wall-shear
  ascii_in/<前缀>-<step>    内部单元：cellnumber,x,y,z,pressure,velocity-magnitude,x/y/z-velocity
  *.stl                     壁面几何，单位 mm，与 ascii 同坐标系
```

单位：坐标 **m**（加载后转 mm）、速度 m/s、WSS/压力 Pa。
每病例 81 个时间步，壁面 ~1.1–1.4 万点，内部 ~74–80 万单元。

## 相对原实现的五处修改

### 1. 网格构建：不再造体素网格

原流程 `UniformGrid → threshold_percent(40) → extract_surface → smooth(500)` 全部去掉。
壁面几何直接用 CFD 壁面节点，法向默认由壁面点云局部 PCA 估计。原流程的平滑迭代是为了消除
体素阶梯伪影，而 CFD 壁面本身已光滑，做平滑只会收缩几何、引入偏差。

### 2. 采样：K 近邻 + 法向投影，替代固定三层等距点

原方法沿法向取 `pc0/pc1/pc2` 三个等距点（间距 0.6mm）插值。本项目 CFD 用的是
**各向异性边界层网格**：内部单元间距 p50 仅 0.18mm，壁面点到最近内部单元 p50 0.60mm。
固定 0.6mm 三层会跨过多层单元、抹平近壁曲率。

改为：取壁面点的候选 K 近邻内部单元，算每个单元到壁面的**法向投影距离** eta，
用 `(eta_i, v_tangential_i)` 做最小二乘拟合 `v_t(eta) = a1·eta + a2·eta²`，
解析求导得壁面梯度 `a1`。无滑移条件隐含在基函数里（无常数项 ⇒ v_t(0)=0），
不需要原代码那个 `--no-slip` 开关。

### 3. 粘度：Carreau-Yasuda 非牛顿，替代固定高剪切粘度 ★

**这是"参数怎么调才能对上"的最关键一项。** CFD 的 `udf-inlet.c` 用的是非牛顿血液模型：

```c
#define A1 0.0035   // mu_inf
#define B  0.16     // mu_zero
#define D  8.2      // lambda
#define E  0.64     // a
#define n  0.2128
viscosity = A1 + (B - A1) * pow(1+pow(D*eps, E), (n-1)/E);
```

μ 在 0.0035–0.16 Pa·s 间随剪切率变化（**46 倍跨度**）。原适配的固定值
0.0035 Pa·s（**3.5 cP**）只等于高剪切极限，在低剪切区会系统性低估 WSS。

实测同一病例、同一拟合配置，仅换粘度模型：

| 粘度模型 | raw R² |
| --- | --- |
| 固定 3.5 cP（高剪切极限） | 0.763 |
| Carreau-Yasuda（对齐 CFD） | **0.893** |

代码复用项目已有的 `wss_pinn/physics/rheology.py`，参数与 UDF 完全一致。

### 4. 法向：局部定向 + 从点云自己估，不依赖 STL ★

这里有两个独立的坑。

**坑 A：定向方式。** 法向朝向不保证一致，需要统一翻转到指向管腔。项目已有实现
（`wss_pinn/physics/wall_shear.py` 及 `velocity_to_wss_oracle.py`）用的是
**指向几何质心**，这对弯曲主动脉不成立：实测 **39% 的壁面点被判错向**，
导致近壁点全落在法向负侧被丢弃，**有效点只剩 61%**。

改为**局部定向**：法向指向该壁面点最近 32 个内部单元的质心。翻转率降到 0%、
有效点覆盖 **100%**。

**坑 B：STL 不可靠。** 批量验证时 `AAA/ruputer/LIU_YU_MING` 只有 raw R² 0.652、
方向余弦 0.882（其余病例 0.998）。查出该病例的 STL **只覆盖部分几何**：
STL z ∈ [−435, −189]，而壁面点 z ∈ [−568, −65]。结果 wall→STL 最近面距离
p50 达 **24.8mm**（正常 0.6mm），法向严重错配（与真值 WSS 矢量 |cos| = 0.445，
理论应为 0）。另有 `ILO/SUN_CHUN_PU-0` 干脆没有 STL 文件。

因此**默认改为从壁面点云自己估法向**（局部 PCA 取最小主方向，`--normals pca`）：

| 病例 | STL 法向 | PCA 法向 |
| --- | --- | --- |
| AAA/ruputer/LIU_YU_MING（STL 覆盖不全） | 0.652 | **0.858** |
| ILO/SUN_CHUN_PU-0/before（无 STL） | 无法运行 | **0.877** |
| AG/slow/LIU_JIN_LIANG（STL 正常） | 0.893 | 0.890 |
| ILO/BAO_EN_YUN-0/before（STL 正常） | 0.907 | 0.907 |

STL 正常时两者持平，STL 有问题时 PCA 显著更好，且少一个外部依赖。
仍会计算 `wall_to_stl_distance_p95` 作为诊断（>3mm 触发 `geometry_warnings`），
`--normals stl` 保留用于对照。

### 5. 邻域：逐点 adaptive-CV，替代固定 K=64 ★

固定 K 在不同点云密度下代表完全不同的物理尺度。最终算法对
`K={16,20,24,28,32,36,40,44,48,64}` 分别拟合归一化二次无滑移剖面，
使用不读取 WSS 真值的 leave-one-out 速度重建误差选 K：

```text
z = eta / median(eta)
v_t(z) = b1*z + b2*z^2
gradient_wall = b1 / median(eta)
```

在 `score <= 1.25 * min(score)` 的候选中选择最小 K。`1.25` 及候选集合只在
train138 上确定，随后冻结到 test35。病例级选中 K 的中位数显示：AG 多为 44，
AAA/ILO 多为 24，证明固定 K=64 不具备跨网格合理性。

## 验证结果

### 正式口径（PINN split173，冻结配置）

最终实验：173 例全部完成、失败 0；每例用相同随机种子抽 1200 个壁面点，
baseline 与 candidate 使用相同病例、时间步、壁面点、PCA 法向和流变模型。

```bash
$PY compare_pointcloud_methods.py \
  --roles train,test --sample-count 1200 \
  --fixed-neighbors 64 --cv-tolerances 1.25 \
  --adaptive-neighbors 16,20,24,28,32,36,40,44,48,64 \
  --json-out ../experiments/pointcloud_adaptive_v1/results/stage_c_split173_s1200.json
```

| 指标 | fixed K=64 | adaptive-CV |
| --- | ---: | ---: |
| raw R² mean | 0.6943 | **0.8540** |
| raw R² p05 | 0.5768 | **0.7618** |
| raw R² min | 0.4722 | **0.6931** |
| MAE (Pa) | 1.5006 | **1.0546** |
| high-WSS NRMSE | 0.4963 | **0.3229** |
| Spearman | 0.9720 | **0.9798** |
| 方向余弦 p50 | 0.9957 | **0.9985** |

成对 raw R² 增益 mean `+0.1597`，**173/173 病例提升**。未参与调参的 test35：
raw R² mean `0.6994 → 0.8586`、min `0.5736 → 0.7568`，35/35 提升。

完整记录见 [`experiments/pointcloud_adaptive_v1/RESULTS.md`](experiments/pointcloud_adaptive_v1/RESULTS.md)。
旧 discover 抽检只保留作历史调试，不再作为正式精度结论。

## 推荐参数

```
--neighbor-mode adaptive_cv \
--adaptive-neighbors 16,20,24,28,32,36,40,44,48,64 \
--cv-tolerance 1.25 --degree 2 --viscosity carreau --normals pca
```

若目标是最低绝对误差，可选用只在 train138 上拟合并冻结的全局离散化尺度：

```bash
--prediction-scale 1.2220224407405893
```

它在 blind test35 上把 raw R² mean 从 `0.8586` 提高到 `0.9153`，MAE 从
`1.0754` 降到 `0.7530 Pa`。该结果含一个 train-supervised 全局常数，必须与
`--prediction-scale 1.0` 的纯物理 raw 结果分开报告。

- **deg=2 必需**：1 次拟合不出近壁曲率；3 次开始拟合网格噪声。
- **固定 K=64 仅作为 baseline**；正式算法逐壁面点自适应选 K。
- 近壁高斯加权（`--bandwidth-weight`）实测**略微降低**精度，默认关闭。

### 纯点云多尺度候选 v2

若不使用任何 WSS 真值幅值标定，可显式启用多尺度零带宽修正：

```bash
--neighbor-mode multiscale_v2
```

它保留 adaptive-CV v1 的梯度方向，只在多个邻域呈现可信收敛趋势时对梯度幅值做
正向、收缩后的局部修正。冻结配置在 blind test35 上把 raw R² mean 从 `0.8586`
提高到 `0.8675`，p05 从 `0.7647` 提高到 `0.7836`，MAE 从 `1.0754` 降到
`1.0372 Pa`，33/35 病例提升；方向余弦不变，Spearman `0.98122 → 0.98104`。

完整记录见 [`experiments/pointcloud_multiscale_v2/RESULTS.md`](experiments/pointcloud_multiscale_v2/RESULTS.md)。
v1 仍保持默认，以免改变已冻结复现口径；若允许 train-only 全局标量，v1 calibrated
的平均 R² 仍略高于 v2 calibrated。

v2 使用全部 train138 壁面点重新推导的冻结标量为：

```bash
--neighbor-mode multiscale_v2 --prediction-scale 1.157066322432233
```

该标量来自 5,436,791 个 train 壁面点；full-wall blind test35 共 1,328,017 点，
得到 case-mean R² `0.9050`、MAE `0.7799 Pa`、high-WSS NRMSE `0.2553`。

## 指标含义

- `raw_r2` — 纯物理前向，**无任何拟合真值的自由参数**。这是真正的"对得上吗"。
- `scaled_r2` — 额外允许一个全局标量 α（pred → α·pred）。用于区分
  "空间分布对不对" 与 "整体幅值有偏"。
- `alpha` — 最优全局标量。实测 ≈1.2，即系统性低估约 20%（见下）。
- `direction_cosine_p50` — 预测 WSS 矢量与真值矢量夹角余弦。实测 0.998。

## 已知残留偏差：α ≈ 1.2

预测的 WSS 空间分布几乎完美（Spearman 0.98、方向余弦 0.998），但幅值系统性
低估约 20%。诊断：按剪切率分箱看 `真值/预测` 比值，从低剪切区的 1.07 缓增到
高剪切区的 1.19。

这不是粘度模型错（反解出的隐含 μ 与 Carreau-Yasuda 预测值比值仅 1.14），而是
**有限差分离散化的固有低估** —— 多项式拟合在有限样本上略微低估真实壁面斜率。
Fluent 自己算 `wall-shear` 用的是单元内的形函数梯度，与外部点云拟合不完全等价。

最终 173 例的逐病例 α mean = 1.239、std = 0.099；train138 mean = 1.240，
test35 mean = 1.234。它可用于诊断残余幅值偏差，但当前 `scaled_r2` 使用逐病例真值
计算 α，不能作为部署性能，也不能用于 test 病例校准。

对下游用途的影响：
- 作为**监督信号 / 物理一致性约束**：0.98 的 Spearman 和 0.998 的方向余弦已足够，
  幅值偏差被一个全局常数吸收即可。
- 作为**绝对值预测**：若要增加全局校正，必须只在 train138 上拟合并冻结；本轮正式
  raw 结果没有使用任何真值校正。

## 真值损坏与运行期防护

历史上对 `data_new` 全库抽检时，曾出现导出本身损坏的病例（例如
`ILO/DONG_JIA_JU-1/before`：壁面点数远超内部单元、约 94% 的 `wall-shear`
恰为 0），会导致 raw R² 暴跌到负几十，与算子无关。

正式路径已通过 **PINN 173 例 split** 避开这类坏导出。运行期仍保留轻量防护：

- `data_loader_cfd.load_wall()`：缺 `wall-shear` 列时直接报错；
- `calculate_wss_cfd.check_truth_quality()`：检测「大量 WSS=0 / 壁面点数>内部点数 /
  非有限值」，写入 `truth_usable` 与 `data_warnings`；
- `batch_validate_cfd.py`：把 `truth_usable=false` 的病例从汇总统计中剔除并单独列出。

全库导出体检脚本已删除；坏导出入库审计以 WSS-PINN source audit 与既有
`docs/02-推进与变更/新队列数据可用性审计_AAA_ILO_2026-07-10.md` 为准。

## 与项目已有实现的关系

正式批量验证病例池与 WSS-PINN 一致：只使用
`train138/test35 exclude SHI_YUN_XI`（173 例，SHA256 `964d7021…9361f2b`）。
baseline 原 `train138/test36` 只读、不修改。

`wss_pinn/tools/velocity_to_wss_oracle.py` 和
`training/scripts/run_v3p_profile_wss_oracle.py` 已实现同类算子，但历史结果的
`raw_r2` 是 **-4253 到 -202772** 的量级（见
`outputs/field/f0_decision/v3p_profile_wss_oracle_20260704.json`），只有校准后
R² 才到 0.57，因此当时只能靠 `calibrated_r2` 判读。

本适配把 raw R² 提到可用量级，两处修正各自的贡献已在上文第 3、4 节量化。
如果要把增益回灌到 `wss_pinn`，需改的是：
- `wss_pinn/physics/wall_shear.py:map_stl_normals_to_wall` — 加局部定向
- `run_v3p_profile_wss_oracle.py` — 粘度换成 Carreau-Yasuda（当前 `MU_BLOOD` 是常数）
