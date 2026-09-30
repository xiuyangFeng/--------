# 合成血流动力学数据集（59 例）

2026-09-28 整理。59 个由真实病例几何形变得到的合成病例，每例都有全周期 CFD 结果（81 帧），包括：壁面点云上的压力、WSS 三分量和 WSS 标量；解剖区体点上的压力和速度；STL 壁面；以及与母例完全相同的血流动力学设置。

## 1. 数据是怎么来的

- **母例**：59 个真实病例（主动脉–髂动脉，AG / AAA / ILO 三个队列），每个母例只派生 1 个子例，母例互不重复。
- **形变**：直接移动母例 Fluent 网格里解剖区节点的坐标。沿中心线加 2 个高斯型局部狭窄或扩张，半径缩放 f(s) = 1 + Σ a_k·exp(−(s−s_k)²/2σ_k²)，|a| 在 0.15–0.40 之间，σ 在 6–14 mm 之间，f 截断在 [0.55, 1.45]。
  - 开口 12 mm 以内、分叉 1.5 倍半径以内逐渐收回到 f = 1，所以入口和出口面积不变。
  - 延伸段不动。
  - 每例具体的形变参数见 `manifest.csv` 的 bump1/bump2 列，以及 h5 属性 `morph`。血管段名按本包的出口标签命名（CHEN_SHU_LIN~m25 与我们库内母例标签相反，已按本包标签改正，见第 6 节）。
- **求解**：边界条件、UDF、血液流变、求解设置一律沿用母例，逐字节相同（见第 5 节），重新做瞬态 CFD。
- **病例编号**：`<母例>~mNN`。ILO 的写法是 `ILO/<姓名-k>~mNN/<before|after>`。

| 队列 | 亚组 | 例数 |
| --- | --- | ---: |
| AG | fast 3 / slow 17 | 20 |
| AAA | ruputer 7 / unruputer 8 | 15 |
| ILO | before 15 / after 9 | 24 |
| 合计 | | **59** |

总计 281.8 万个壁面点、2640.9 万个体点、81 帧；数据包约 32 GB（每例 0.3–1.4 GB）。

## 2. 目录结构

```
synth59_teacher_package_2026-09-28/
├── README.md                  本文件
├── manifest.csv               59 行总表：编号、母例、形变参数、点数、质量标记、文件校验和
├── read_example.py            读取示例（Python）
├── verification/
│   ├── settings_identity_59.csv    与母例设置逐字节一致性核验
│   ├── package_selfcheck_59.csv    写后自检（逐位对照、STL 顶点、三角网朝向）
│   ├── raw_export_spotcheck.csv    抽 6 例 × 2 帧，对照 Fluent 原始导出
│   ├── numeric_sanity_59.csv       59 例 × 81 帧数值合理性扫描（值域、质量守恒、周期收敛、与母例对照）
│   ├── numeric_child_vs_parent_59.csv  同一组指标在母例上的对照
│   └── shape_preview_15cases.png   15 例形变预览（蓝 = 狭窄，红 = 扩张，虚线 = 母例轮廓）
├── AG/fast/PENG_JI_MING~m25/
│   ├── PENG_JI_MING~m25.h5                 点云 + 全部场量（见 3.1）
│   ├── PENG_JI_MING~m25.stl                解剖区壁面
│   ├── PENG_JI_MING~m25_hemodynamics.json  血流动力学设置 + 一致性核验
│   └── cfd_setup/                          Fluent 原始输入：*.cas.gz、udf-inlet*.c、2.jou
├── AAA/ruputer/…   AAA/unruputer/…
└── ILO/<姓名-k>~mNN/before/…   ILO/<姓名-k>~mNN/after/…
```

## 3. 每例文件

### 3.1 `<名>.h5`（HDF5；81 帧的量都按帧分块，gzip 压缩）

| 路径 | 形状 | 单位 | 说明 |
| --- | --- | --- | --- |
| `time/step`、`time/time_s`、`time/phase_s` | (81,) | –、s、s | 求解步 1120–1280（每 2 步一帧）；相位 0–0.80 s。属性 `peak_index` = 21 对应峰值帧 step 1162（相位 0.21 s） |
| `wall/xyz_mm` | (Nw, 3) | mm | 壁面点坐标（Fluent 网格壁面节点），与 STL 顶点是同一批点 |
| `wall/normal_out` | (Nw, 3) | – | 外法向（单位向量） |
| `wall/area_m2` | (Nw,) | m² | 每个点分到的壁面面积（面积加权积分用，总和 = 壁面面积） |
| `wall/triangles` | (Nt, 3) | – | 三角网（0 起始索引到 wall 行），法向朝外，与 STL 三角形一一对应 |
| `wall/fluent_node_id` | (Nw,) | – | Fluent 节点编号（对照 cfd_setup 里的 .cas 用） |
| `wall/pressure_pa` | (81, Nw) | Pa | 壁面压力（表压） |
| `wall/wss_vector_pa` | (81, Nw, 3) | Pa | WSS 三分量（x/y/z） |
| `wall/wss_scalar_pa` | (81, Nw) | Pa | WSS 标量（Fluent wall-shear 导出列） |
| `volume/xyz_mm` | (Nv, 3) | mm | 体点 = 解剖区网格单元中心（不含入口/出口延伸段） |
| `volume/volume_m3` | (Nv,) | m³ | 单元体积（体积加权积分用） |
| `volume/fluent_cell_id` | (Nv,) | – | Fluent 单元编号 |
| `volume/pressure_pa` | (81, Nv) | Pa | 压力（表压） |
| `volume/velocity_m_s` | (81, Nv, 3) | m/s | 速度三分量 |
| `boundary/<开口>/…` | | | 5 个开口：`inlet`、`out-le` 左髂外、`out-li` 左髂内、`out-re` 右髂外、`out-ri` 右髂内。属性给面积、中心、外法向、Fluent 出口边界名；数据集有 `flux_outward_m3s` (81,)（流出为正，入口为负）、`p_mean_pa` (81,)（面积平均压力）、`face_center_mm`、`face_area_vec_m2` |
| 根属性 | | | `case_id`、`parent_id`、`units`、`coordinate_frame`、`hemodynamics`（带中文注释的 JSON，与 `*_hemodynamics.json` 逐字相同）、`morph`（形变参数 JSON）、`quality_flags`（JSON） |

- **壁面没有速度**：壁面为无滑移条件，速度恒为 0，所以速度只放在体点上。
- **WSS 标量与三分量的关系**：标量与三分量合成的模长有插值差（各例中位 0.1–0.6%，总体约 0.3%）。做标量任务请直接用 `wss_scalar_pa`。
- **时间平均**：81 帧每隔 0.01 s 一帧，第 1 帧与第 81 帧是同一相位（周期首尾）。求周期平均（如 TAWSS、平均流量）时取前 80 帧再平均，即 `x[:80].mean(0)`；用全部 81 帧会把首尾相位多算一次。
- **中文说明**：h5 里每个组和数据集都有一个中文属性 `说明`，写明含义、形状和单位。用 HDFView 打开可直接看到，Python 里用 `f["wall/wss_scalar_pa"].attrs["说明"]` 读取。

### 3.2 `<名>.stl`

二进制 STL，单位 mm，坐标系与 h5 相同，法向朝外。只包含解剖区壁面，是一个开口面：入口和 4 个出口处敞开，也就是 `boundary/` 里的 5 个开口。

- 顶点就是 `wall/xyz_mm` 中的点（float32 取整，误差 < 1e-4 mm）。
- 三角形由 Fluent 壁面多边形剖分而来，没有加新顶点。

### 3.3 `<名>_hemodynamics.json`

本例的全部血流动力学设置和一致性核验结论。文件里每个参数前面都紧跟一个以 `_说明` 结尾的键，是中文注释（JSON 格式本身不支持注释，所以用这种写法）；程序读取时忽略这些键即可，原有键名和数值不受影响。文件开头有两段：

- `文件说明`：怎么读这个文件，以及键名末尾的单位后缀。
- `本例解读`：本例的具体数字，包括入口流量（平均、峰值、最小）、四个出口按 RCR 应分的流量与实测流量、平均压力、RCR 规律核对、形变概况和质量标记。

各块内容：

- `fluid`：密度、粘度模型与参数，另附几个剪切率下的粘度参考值。
- `inlet`：波形公式、Fourier 系数、周期、UDF 面积常数、逐帧标称流量。
- `outlets.per_outlet`：每个出口的 Fluent 边界名、R1 / R2 / C（同时给质量流量和体积流量两种口径）、开口面积。
- `solver`：软件、层流、步长、步数、每步迭代数、导出帧、峰值帧、残差层级。
- `identical_to_parent`：与母例的逐项比对结果。

### 3.4 `cfd_setup/`

- `*.cas.gz`：形变后的网格 + 全部 Fluent 设置。
- `udf-inlet*.c`：入口波形、Carreau-Yasuda 粘度和四个出口 RCR 的 UDF 源码。
- `2.jou`：求解 journal。

有了这三个文件就能在 ANSYS Fluent 2023 R1 里原样重算。文件里的绝对路径是我们集群的路径，重算前需要改成本地路径（2.jou 第一行的 read-case，以及 .cas 里的导出路径）。

## 4. 血流动力学设置

| 项 | 设置 | 59 例是否相同 |
| --- | --- | --- |
| 血液 | 密度 1060 kg/m³；Carreau-Yasuda 粘度 μ = μ∞ + (μ0 − μ∞)(1 + (λγ̇)^a)^((n−1)/a)，μ∞ = 0.0035 Pa·s、μ0 = 0.16 Pa·s、λ = 8.2 s、a = 0.64、n = 0.2128 | 全部相同 |
| 入口 | velocity-inlet，截面均匀速度 = Q(t)/A_udf；Q(t) = 1e-6 × (a0 + Σ_{k=1..8} a_k cos(kωt') + b_k sin(kωt')) m³/s，t' = t mod 0.8 s，ω = 6.491，系数见 json | 波形全部相同；A_udf 逐例不同。CHEN_SHU_LIN~m25 实际入口面积 / A_udf = 0.888，实际入流是标称值的 0.888 倍 |
| 出口 | pressure-outlet，三元 Windkessel（RCR）：P_n = ((R1 + R2 + R1β)Q_n + βP_{n−1} − R1βQ_{n−1})/(1 + β)，β = R2·C/Δt，Q 为出口质量流量 | R1 / R2 / C 逐例逐出口不同：总阻力 AG 约 4.37e5、AAA/ILO 约 3.76e5 Pa·s/kg，按出口面积以 Murray r³ 规则分配 |
| 壁面 | 刚性、无滑移 | 全部相同 |
| 求解 | ANSYS Fluent 2023 R1，3ddp，层流，瞬态 dual-time；Δt = 0.005 s，1280 步（8 个周期），每步最多 20 次迭代；操作压力 101325 Pa，输出为表压 | 全部相同 |
| 导出 | 最后一个周期，step 1120–1280 每 2 步一帧，共 81 帧；峰值帧 step 1162 | 全部相同 |

RCR 单位说明：UDF 按质量流量积分，所以 R 的单位是 Pa·s/kg、C 的单位是 kg/Pa。json 里另给了按密度换算的体积流量口径（Pa·s/m³、m³/Pa）。

## 5. 与母例"设置一模一样"的核验

逐例对照母例的原始 Fluent 输入（`verification/settings_identity_59.csv`），59/59 全部通过：

- **UDF**（入口波形、粘度、RCR）：逐字节相同，59/59。
- **journal**：除第一行 read-case 路径外逐字节相同，59/59。
- **.cas.gz**：解压后去掉节点坐标段，并把绝对路径归一成文件名，其余逐字节相同，59/59。形变只改了坐标，每例移动了 2.1 万–80 万个节点。
- **提交脚本 fluent.slurm**：只有作业调度不同（节点指定、并行核数 40/48 → 64），59/59；不涉及任何物理设置。
- **运行结果佐证**：左右流量分配子例与母例的差中位 0.0002（55 例有母例运行记录）；四个出口平均压力的差中位约 2 Pa。

## 5b. 数值合理性（59 例 × 81 帧全扫描，`verification/numeric_sanity_59.csv`）

- **有限性**：所有帧、所有量都没有 NaN/Inf；WSS 标量没有负值。
- **值域**：压力 9.7–19.3 kPa（表压，约 73–145 mmHg）；峰值帧 WSS 中位 2.3 Pa、p99 中位 21 Pa（最大 84 Pa）；TAWSS 中位 0.79 Pa；最大流速 0.6–3.4 m/s。最高的几例都在狭窄射流里，母例同样的位置也有这么快。与母例逐项对照（`numeric_child_vs_parent_59.csv`），压力值域和最大流速都基本相同；只有加了强髂外狭窄的两例射流明显变快，符合物理：MI_DE_XI~m32 由 2.24 升到 3.09 m/s，ZHANG_LING_GUANG~m22 由 1.44 升到 2.46 m/s。
- **质量守恒**：Fluent 统计的出入口质量守恒为 0.9997–1.0012。h5 `boundary/*/flux_outward_m3s` 是用开口截面邻近单元的速度近似积分的，逐帧会有 0.1–3.4%（相对峰值入流）的不平衡。母例用同一方法算出的数字完全一样，说明这是积分近似的误差，不是求解问题。需要精确流量时，请按出口份额使用 json 里的 RCR 结果或 Fluent 监测值。
- **周期收敛**：第 1 帧（t = 5.6 s）与第 81 帧（t = 6.4 s）相隔正好一个周期。一半病例两帧 WSS 相差 < 1%；约 10% 的病例相差 10–20%，母例也有同样量级（p90 11%，最大 43%）。差异只出现在舒张末期低流量时刻，那时 WSS 仅约 0.3 Pa，绝对差中位 0.02–0.08 Pa；对 TAWSS 的影响中位 < 0.1%。平均压力逐周期仍有约 150 Pa（约 1%）的漂移：出口 RCR 的时间常数 1.79 s 大于周期 0.8 s，8 个周期后压力水平还没完全稳定。这一点母例相同，不影响速度场和 WSS。
- **形变没碰到的区域**：未移动壁面节点上的峰值帧 WSS 与母例相差中位 1.5%。差得多的是在髂外加了强狭窄的病例，下游射流和分流改变让未形变区域的 WSS 也跟着变：ZHANG_LING_GUANG~m22 相差 22%，MI_DE_XI~m32 相差 12%。

## 6. 质量标记（`manifest.csv` 的 quality_notes 列，h5 属性 `quality_flags` 里也有）

所有病例都通过了我们入库链的全部门槛（拓扑、壁面、中心线、81 帧、RCR 运行态、点云 19 项）。以下是需要知道的例外：

- **CHEN_SHU_LIN~m25：左侧两出口的 RCR 与开口面积对调。** 这是母例 CFD 设置本身的问题，子例原样继承。
  - 原因：母网格左侧内部接口面名与出口边界名是交叉的，当初按接口面名取面积算 RCR，结果左髂外（34 mm² 开口）拿到了按左髂内（19 mm²）面积算出的高阻。
  - 后果：左髂外只分到约 15% 的流量，左髂内约 33%，与 Murray 协议相反。
  - 其余 58 例同侧两支的阻力比都与面积规则一致。
  - 本包的出口标签按出口边界名和 UDF 标注，按开口位置与右侧对称判断，解剖上是对的。
  - 另外，这一例的入口实际流量是标称值的 0.888 倍。
  - 形变段名也按本包标签：这一例的狭窄凸包在左髂内（半径约 2.5 mm 的小支），manifest 与 h5 `morph` 已同步。
- **4 例有形变引入的少量网格"左手面"**：GONG_HAI_ZENG-1~m34/after、LI_YOU_ZHI-0~m48/before、SUN_CHUN_PU-0~m53/before、WENG_ZHAO_GUANG-0~m03/before。
  - 每例 1–9 个面，都在离壁 0.5 mm 以内的边界层内部面，没有负体积单元，母例没有这种面。
  - 这 4 例都已正常求解、通过全部门槛，但局部近壁 WSS 的可信度略低于其余病例。
- **求解残差层级**：按 81 个导出步最后一次连续性残差的最大值分为 A ≤ 1e-3、B < 2e-3、C < 5e-3、D ≥ 5e-3。
  - D：SUN_ZONG_GE~m43（9.8e-3，母例 B）、ZHANG_LING_GUANG~m22（6.0e-3，母例 C）、ZHAO_CHANG_SHAN-0~m06/before（5.2e-3，母例 C）。
  - C：5 例。
  - 这一项不是门槛。我们在母库里对 D 层重跑过，WSS 变化不到 1%。
- **中心线标复核**：3 例，不影响本包数据。

## 7. 读取

Python（需要 numpy、h5py）：

```python
import h5py, json
with h5py.File("AG/fast/PENG_JI_MING~m25/PENG_JI_MING~m25.h5", "r") as f:
    k = f["time"].attrs["peak_index"]           # 峰值帧
    xyz = f["wall/xyz_mm"][...]                 # (N,3) mm
    wss = f["wall/wss_scalar_pa"][k]            # (N,)  Pa
    tau = f["wall/wss_vector_pa"][k]            # (N,3) Pa
    p   = f["wall/pressure_pa"][k]              # (N,)  Pa
    u   = f["volume/velocity_m_s"][k]           # (M,3) m/s
    hemo = json.loads(f.attrs["hemodynamics"])  # 血流动力学设置
```

完整示例见 `python read_example.py AG/fast/PENG_JI_MING~m25`。

MATLAB：`h5read(file, '/wall/wss_scalar_pa')` 返回的数组维度与 Python 顺序相反，是 N×81；属性用 `h5readatt(file, '/', 'hemodynamics')` 读。

## 8. 注意事项

- 坐标都是各病例自己的 Fluent 算例坐标系（mm），病例之间没有做配准。
- 体点只覆盖解剖区（和壁面点云同一区域），不含 CFD 为稳定边界条件而加的入口/出口延伸段。
- 形变只改变了局部管径，没有改变血管拓扑。形变后的几何没有对应的影像数据。
- 数值来源：写包时逐位对照了我们的入库文件；另外抽 6 例 × 2 帧对照 Fluent 原始导出，压力、WSS 三分量与标量、速度三分量的相对差都在 6e-8 以内（float32 精度）。
