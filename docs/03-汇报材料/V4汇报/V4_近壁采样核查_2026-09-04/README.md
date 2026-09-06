# BC/RCR V4 近壁 1.5 mm 定义与 PointNet / PointNet++ 采样核查

> 核查日期：2026-09-04  
> 同病例对照：`AG/fast/RAN_QING_BO`（V4 test35）  
> 对照臂：`V4-SP-PN-DATA-s1234` 与 `V4-SP-PNPP-DATA-s1234`  
> 重要边界：已训练 V4 是 pre-Centerline-V2 历史矩阵；当前 Centerline-V2 rawfull-v2 staging 尚未训练且 `training_ready=false`。
> 2026-09-04 用户已选择正式新骨干为 P2V / D2 `c125-k128`；本页图 2/3 仍是旧容量
> 配平模型的真实采样，不代表尚未实现的 formal V4。

## 一页结论

1. **数据中没有血管壁组织厚度。** 它只有流体管腔内点、管腔表面和局部最大内切球半径。真实 mm 的 anatomy atlas 在 173 例共 199,033 个唯一中心线样本上，半径 P5 / P50 / P95 为 **2.57 / 6.16 / 16.25 mm**，对应直径约 **5.14 / 12.33 / 32.49 mm**。
2. **当前 staging 的真实 1.5 mm 层没有把点全选完，但已选了多数 cell。** 按真实单位重算，173 例逐例覆盖 **50.11%–88.10%**，中位 **68.49%**，0/173 例达到 100%；pooled 为 92,409,142 / 134,133,743 = **68.89%**。这是近壁边界层网格加密造成的“点数占多数”，不等于管腔体积的 68.89%。
3. **旧已训练 V4 的 near-wall 并不是完整 1.5 mm 层。** 旧预处理先用 `distance<=1.5`，然后只保留距壁最近的 40,000 点。旧 V4 的 `near-wall speed R²` 实际在这个被 cap 的 `int_type` 子集上算，它是评估 region，不是训练采样。
4. **V4 训练没有近壁分层，也没有沿法线采样。** PointNet 和 PointNet++ 都从全部 strict-volume 独立均匀随机取 `support=5000` 与 `query=5000`；同病例、seed、epoch 时两个 backbone 的 5000 support **完全相同**。
5. **PointNet++ 只多了 backbone 内部的一层 `FPS-128 + 32-NN`。** 它不是早期表面 WSS 实验中的 D2 三层 `125/125/32 + k128/16/16`，也没有沿内法线组成有序速度剖面。
6. **正式新 V4 已决定恢复 P2V/D2，但近壁 sampler 尚未随之自动冻结。** 骨干选择 B
   不等于继承父实验的全部 query 协议，更不等于已经加入法线剖面采样。

**判定：** `1.5 mm` 可以保留为一个宽口径评估带，但不适合单独代表 WSS 敏感的最近壁 1–2 层 cell。旧 V4 的 cap 标签不能继续当成固定 1.5 mm 指标；当前 staging 则不需要继续粗暴“加大近壁比例”，而需要更细的距离分带与法线剖面结构。

## 为什么两个 backbone 用同一个病例

`wss_pinn/v4/data.py` 的采样 seed 只含 `global_seed | epoch | case | stream`，不含 backbone。如果 PointNet 和 PointNet++ 各换一个病例，图上差异会同时包含血管几何与网格密度差异。因此两图都用 `RAN_QING_BO`，并固定 seed1234 / epoch0：

- PointNet 图展示它直接使用的 5000 support；
- PointNet++ 图在**同一个** 5000 support 上叠加 128 个 FPS center 和 32-NN 并集。

`RAN_QING_BO` 也出现在早期 PointNet FPS 采样包和 PointNet++ SA3 trace 中，可与旧资料直接对照。

## 图 1：厚度与覆盖

![V4 近壁厚度与覆盖](./fig1_全队列近壁厚度与病例覆盖.png)

`RAN_QING_BO` 的真实 anatomy-atlas 半径 min / P5 / P50 / P95 / max 为 **1.82 / 2.35 / 8.48 / 15.04 / 16.78 mm**，直径中位约 **16.97 mm**。这个病例中：

| 口径 | 点数 | strict-volume 占比 | 含义 |
| --- | ---: | ---: | --- |
| 旧距离数组中的真实 `d<=1.5 mm` | 303,507 | 30.64% | 旧壁面节点/几何口径下的候选层 |
| 旧 V4 实际 `int_type=1` | 40,000 | 4.04% | 40k cap 后的评估子集，最远仅 0.637 mm |
| 当前 staging 的真实 `d<=1.5 mm` | 567,294 | 57.27% | 全 Fluent physical-wall nodes、无 cap |
| 当前 staging 直接读存储 `region` | 581,868 | 58.74% | 存储的伪 mm `1.5` 对本例实际为 1.550 mm |

旧/当前 staging 采用的逐例 `fluent_to_mm_unit_factor` 不是物理单位换算。本核查按
`true_mm = stored_length * 1000 / factor` 重标；`RAN_QING_BO` 的 factor 为 967.74877，重标系数为 1.033326。

## 图 2：PointNet 的实际基础采样

![PointNet 采样](./fig2_PointNet_RAN_QING_BO_epoch0采样.png)

epoch0 的 5000 support 中，在当前真实距离口径下有 2,806 点（56.12%）落在 1.5 mm 内；但旧 V4 `int_type` 只标了 207 点（4.14%）。PointNet 不会根据两种标签重采样，它对全部 5000 点做 point MLP 后 global max。

## 图 3：PointNet++ 的 backbone 内部降采样

![PointNet++ 采样](./fig3_PointNetPP_RAN_QING_BO_epoch0采样.png)

| 层次 | 点数 | 当前真实 `d<=1.5 mm` | 旧 V4 `int_type` |
| --- | ---: | ---: | ---: |
| shared support | 5,000 | 2,806（56.12%） | 207（4.14%） |
| FPS centers | 128 | 113（88.28%） | 6（4.69%） |
| 128×32-NN 的唯一并集 | 3,626 | 2,000（55.16%） | 140（3.86%） |

FPS center 在空间外围偏多，所以当前 1.5 mm 占比到了 88.28%；但 32-NN 并集又回到与基础 support 接近的 55.16%。这个空间覆盖特性不保证每个壁面锚点都有沿内法线的多深度速度点。

## 与早期无 PINN 实验的关系

- 早期 PointNet 表面 WSS 线中，`P2V vertex-random5000 / SEP` 好于 SAME；PointNet++ 的 `random5000 + FPS center` 好于 fixed-FPS2000/5000 和 multi-start FPS。全点与 10k 并未稳定胜过 5k，说明“更多点”不是单调收益。
- 体域 PINN V1/V2 实际冻结的父结构是 P2V / 纯 D2 `c125-k128`；2026-09-04 的
  formal V4 选择 B 也是指这两个 provenance 锚点。父 P2V 为 SEP、父 D2 为 SAME，
  所以“继承骨干”不等于继承同一个完整 query sampler。
- 后续 direct-WSS 开发转向 D2-k64，并依次加入 PointNeXt-R、LocalGeoPE、DropPath0.1、
  L-SA2、H2 q90 pinball 与 log-radius；这是 WSS-only 线后来的开发锚点，没有进入
  V1/V2 provenance，也不是本次选择 B 的 D2 `c125-k128`。
- 沿内法线的多点速度剖面有独立正证据：二次剖面 `u_t(η)=a1η+a2η²` 的 Profile oracle 在 73 例上 cross calibrated R²=0.632，高于单点差分 0.166；后续 K9 复审到 0.704。这支持“保留有序法线剖面”，不支持“只增加一个宽近壁 mask”。

早期数字来自 `WSS_PointNet实验矩阵与结果汇总last.xlsx`的“教师汇报视图”与“实验矩阵总览”；上述实验是壁面直接 WSS 任务，不能把它的 R² 与体域速度 V4 裸比。
沿法线剖面的证据则来自 `wss_mri_calculator` velocity→WSS/Profile oracle 后处理线，
不属于 P2V 或 D2 的训练采样；三类对象必须分开引用。

## formal V4 近壁采样提案（尚未冻结）

以下是本次可视化审计给出的设计建议，不是当前代码事实，也不因骨干选择 B 自动生效：

1. 正式数据先固定 `Fluent m × 1000 = true mm`，使用 anatomy-only 物理壁面，不再读旧 `int_type` 或 40k cap。
2. 评估同时报 `0–0.5 / 0.5–1.0 / 1.0–1.5 / >1.5 mm`，并补一个局部尺度 `d/R` 分带；不用单一 1.5 mm R² 掩盖最近壁层。
3. support 保留全域几何覆盖；query/loss 再按上述距离带固定配额采样，必要时用逆采样概率加权，避免改变原始物理分布。
4. 为 near-wall velocity 监督/query 或冻结 velocity→WSS 下游审计另建“壁面锚点 +
   同一内法线 3–5 个深度点”，并保留 stream/cross 双分量、穿模和拓扑 Gate；本条不把
   WSS 输出或 WSS loss 自动加入 formal V4，也不参与选模。
5. 在重训前做单病例 sampler dry-run：验收每个距离带的唯一点数、管腔分支覆盖、法线有效率和无放回采样。

## 产物索引

| 文件 | 用途 |
| --- | --- |
| `fig1_全队列近壁厚度与病例覆盖.png` | 半径分布、173 例覆盖率、RAN 截面和旧 cap 对照 |
| `fig2_PointNet_RAN_QING_BO_epoch0采样.png` | PointNet 的同病例 5000 support 可视化 |
| `fig3_PointNetPP_RAN_QING_BO_epoch0采样.png` | PointNet++ 的 FPS-128 / 32-NN 可视化 |
| `RAN_QING_BO_V4_sampling_audit.vtp` | ParaView 交互点云；含真实壁距、新/旧 near-wall、support、FPS、kNN membership 字段 |
| `RAN_QING_BO_epoch0_support5000.csv` | 5000 个 support 的逐点索引与标签 |
| `v4_nearwall_case_summary.csv` | 173 例单位、覆盖率、旧 cap 有效厚度和 atlas 半径汇总 |
| `audit_summary.json` | 本页核心数字的机器可读版 |
| `plot_v4_nearwall_sampling.py` | 可复现脚本 |

## 真源

- 旧 1.5 mm + 40k cap：`pipeline_wss_min/config.py:200-205`、`pipeline_wss_min/preprocess.py:370-382`。
- V4 均匀采样：`wss_pinn/v4/config.py:67-75`、`wss_pinn/v4/data.py:262,337-345`。
- PointNet / PointNet++ 编码器：`wss_pinn/v4/models.py:48-107`。
- region 只用于评估：`wss_pinn/v4/evaluate.py:238-264,411-414`。
- Centerline-V2 无 cap 定义：`wss_pinn/v4/build_centerline_v2.py:315-375`。
- 长度单位核查：`docs/02-推进与变更/WSS_PINN/WSS_PINN_V4_anatomy-only预处理准备与173例数据核查_2026-09-03.md:31-90`。
- RAN 单位机器证据：`outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits/topology/cases/AG__fast__RAN_QING_BO.json`。
- 早期 PointNet 采样：`training_wss_min/runs/r4_dev1_b0_tgtw_batchq_s1234/sampling_viz/fast__RAN_QING_BO/`。
- 早期 PointNet++ SA3：`例子/06_PointNet++_SA三层采样与分组/Q2V-10477_vertex-random5000_FPS-center_fast_RAN_QING_BO_ParaView/`。
