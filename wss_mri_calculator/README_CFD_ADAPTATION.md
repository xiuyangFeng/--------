# 适配说明：用本项目 CFD 数据跑 velocity → WSS

原仓库（`wss_mri_calculator`）面向 **4D Flow MRI 规则体素影像**。本项目 `data_new/`
是 **Fluent 非结构化点云**，因此新增了三个文件，原有文件未改动：

| 文件 | 作用 |
| --- | --- |
| `src/data_loader_cfd.py` | 读 Fluent ASCII 导出（替代原 `data_loader.py` 的 HDF5 读取） |
| `src/calculate_wss_cfd.py` | 单病例算 WSS 并与 CFD 真值对比（替代 `calculate_wss.py`） |
| `src/batch_validate_cfd.py` | 跨队列批量验证 |

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

# 正式批量：PINN 冻结 173 例（峰值步）
$PY batch_validate_cfd.py \
  --json-out ../../outputs/wss_pinn/audits/cfd_velocity_wss_explore/pinn173_peak.json

# 快速抽检：仍来自该 173 例 split，不是全库
$PY batch_validate_cfd.py --limit-per-cohort 3

# 仅 test35
$PY batch_validate_cfd.py --roles test
```

依赖：`numpy` / `scipy` / `matplotlib`（出图时）即可（**不需要 pyvista**）。
`GNN` conda 环境已满足。

## 数据格式

```
<病例目录>/
  ascii/<前缀>-<step>       壁面节点：nodenumber,x,y,z,pressure,wall-shear,x/y/z-wall-shear
  ascii_in/<前缀>-<step>    内部单元：cellnumber,x,y,z,pressure,velocity-magnitude,x/y/z-velocity
  *.stl                     壁面几何，单位 mm，与 ascii 同坐标系
```

单位：坐标 **m**（加载后转 mm）、速度 m/s、WSS/压力 Pa。
每病例 81 个时间步，壁面 ~1.1–1.4 万点，内部 ~74–80 万单元。

## 相对原实现的四处修改

### 1. 网格构建：不再造体素网格

原流程 `UniformGrid → threshold_percent(40) → extract_surface → smooth(500)` 全部去掉。
壁面几何直接用 CFD 壁面节点，法向取最近 STL 面片法向。原流程的平滑迭代是为了消除
体素阶梯伪影，而 CFD 壁面本身已光滑，做平滑只会收缩几何、引入偏差。

### 2. 采样：K 近邻 + 法向投影，替代固定三层等距点

原方法沿法向取 `pc0/pc1/pc2` 三个等距点（间距 0.6mm）插值。本项目 CFD 用的是
**各向异性边界层网格**：内部单元间距 p50 仅 0.18mm，壁面点到最近内部单元 p50 0.60mm。
固定 0.6mm 三层会跨过多层单元、抹平近壁曲率。

改为：取壁面点的 K 近邻内部单元，算每个单元到壁面的**法向投影距离** eta，
用 `(eta_i, v_tangential_i)` 做最小二乘拟合 `v_t(eta) = a1·eta + a2·eta²`，
解析求导得壁面梯度 `a1`。无滑移条件隐含在基函数里（无常数项 ⇒ v_t(0)=0），
不需要原代码那个 `--no-slip` 开关。

### 3. 粘度：Carreau-Yasuda 非牛顿，替代固定 4 cP ★

**这是"参数怎么调才能对上"的最关键一项。** CFD 的 `udf-inlet.c` 用的是非牛顿血液模型：

```c
#define A1 0.0035   // mu_inf
#define B  0.16     // mu_zero
#define D  8.2      // lambda
#define E  0.64     // a
#define n  0.2128
viscosity = A1 + (B - A1) * pow(1+pow(D*eps, E), (n-1)/E);
```

μ 在 0.0035–0.16 Pa·s 间随剪切率变化（**56 倍跨度**）。原代码默认 `--viscosity 4`
（cP，即 0.0035 Pa·s）只等于**高剪切极限**，在低剪切区会系统性低估 WSS。

实测同一病例、同一拟合配置，仅换粘度模型：

| 粘度模型 | raw R² |
| --- | --- |
| 固定 4 cP（原默认） | 0.763 |
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

## 验证结果

### 正式口径（PINN 173 例 split）

批量脚本默认只跑冻结 split。重跑命令：

```bash
$PY batch_validate_cfd.py \
  --json-out ../../outputs/wss_pinn/audits/cfd_velocity_wss_explore/pinn173_peak.json
```

输出 JSON 会写入 `split.path` / `split.sha256`，便于核对是否真的绑在
`964d7021…9361f2b` 上。未带该字段的旧结果（对 `data_new` 随机抽检）
**不得**再当作正式结论。

### 历史抽检（36 例，每队列 12 例；非正式）

以下为适配开发阶段的 `discover` 抽检存档，配置 K=64 / deg=2 / Carreau；
**病例池不是 PINN 173 split**，仅作方法调试参考：

| 指标 | mean | p05 | p50 | p95 | min |
| --- | --- | --- | --- | --- | --- |
| **raw R²** | **0.871** | 0.821 | 0.871 | 0.921 | 0.652 |
| scaled R² | 0.940 | 0.902 | 0.955 | 0.973 | 0.695 |
| Spearman | 0.982 | 0.968 | 0.984 | 0.990 | 0.948 |
| α | 1.240 | 1.136 | 1.240 | 1.346 | 1.103 |
| 方向余弦 | 0.994 | 0.995 | 0.998 | 0.999 | 0.882 |
| 有效点覆盖 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

跨队列一致，无队列特异性失效：

| 队列 | n | raw R² mean | p05 | min |
| --- | --- | --- | --- | --- |
| AG | 12 | 0.862 | 0.824 | 0.823 |
| AAA | 12 | 0.873 | 0.767 | 0.652 |
| ILO | 11 | 0.880 | 0.833 | 0.816 |

（另有 1 例 ILO 因真值导出损坏被排除，见下文。）

## 推荐参数

```
--neighbors 64 --degree 2 --viscosity carreau
```

单病例扫参结果（`--sweep`，raw R²）：

| K | deg=1 | deg=2 | deg=3 |
| --- | --- | --- | --- |
| 32 | 0.783 | 0.832 | 0.580 |
| 64 | 0.651 | **0.893** | 0.778 |
| 128 | 0.548 | 0.819 | 0.750 |

- **deg=2 必需**：1 次拟合不出近壁曲率；3 次开始拟合网格噪声。
- **K=64 最优**：K 太小样本不足，太大会外扩出边界层。K 自适应匹配局部网格密度，
  比按物理距离截断（试过 0.5/1.0/1.5/2.0/3.0 mm）更好。
- 近壁高斯加权（`--bandwidth-weight`）实测**略微降低**精度，默认关闭。

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

**α 足够稳定，可以用单个全局常数校正**（35 例统计）：

- α mean = 1.240，std = 0.074，**CV 仅 5.9%**
- 用全局 α=1.240 替代逐例最优 α，相对偏差 p50 = 3.5%、p95 = 10.8%
- 队列间差异很小：AG 1.284 / AAA 1.236 / ILO 1.196

对下游用途的影响：
- 作为**监督信号 / 物理一致性约束**：0.98 的 Spearman 和 0.998 的方向余弦已足够，
  幅值偏差被一个全局常数吸收即可。
- 作为**绝对值预测**：乘 α=1.240。**务必用训练集拟合这个常数，不要逐病例校准**
  —— 后者会泄漏测试集真值。

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
