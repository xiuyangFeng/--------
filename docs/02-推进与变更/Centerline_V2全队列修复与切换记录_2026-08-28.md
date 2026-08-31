# Centerline V2 全队列修复与切换记录（2026-08-28）

> 2026-08-31 addendum：WSS_PINN 已在独立 staging root 完成 173/173 Centerline V2
> rawfull 重建与工程 Gate，未覆盖旧 `data_wss_pinn`，且 manifest 保持
> `training_ready=false`。`data_wss_min`、pipeline 默认输入、其他旧 bundle 和正式训练
> route 仍未 cutover；详见
> [WSS_PINN V4 173 例审阅与修复计划](WSS_PINN/WSS_PINN_V4_173例训练数据数值与刚性配准审阅及修复计划_2026-08-30.md)。

## 1. 最终结论

按照 [VMTK 中心线修复与全队列优化方案（已执行归档）](_archive/VMTK中心线修复与全队列优化方案_已执行_2026-08-27.md) 完成 173 例 Centerline V2 全量提取、自动修复、Gate、逐例渲染与完整性核验。

最终结果：

| 项目 | 结果 |
| --- | ---: |
| 全队列病例 | 173 |
| 硬 Gate 通过 | 173/173 |
| `pass` | 153 |
| `pass_review` | 20 |
| 处理错误 | 0 |
| 首轮标准提取入选 | 170 |
| 第 3 轮 clean + 种子内移入选 | 3 |
| 终末尖刺自动修复 | 37 例 / 41 条 |
| 安全长边细分 | 1 例 / 1 条 / 插入 6 点 |
| 病例级三视图 | 173 份 |
| 全队列总览图 | 9 页 |
| 路径 CSV 总行数 | 422,388 |
| 最终完整性核验错误 | 0 |

全量输出位于
[`outputs/centerline_v2_full_173_20260828/`](../../outputs/centerline_v2_full_173_20260828/README.md)。

## 2. 本次落地的 Centerline V2 合同

### 2.1 权威表面和单位先冻结

- 从既有 173 例 bundle manifest 读取不依赖新中心线的权威 STL/单位事实；
- 对每例记录权威 STL 路径、SHA256、STL→mm 比例、Fluent→mm 单位因子、候选 STL 和选择依据；
- 使用 CFD inlet / 四出口边界面的几何合同识别开口角色，不再使用世界坐标 `max Z` 作为默认入口规则；
- 新中心线不得反向参与权威 STL 或基础单位的决定。

病例级冻结证据写入 `surface_selection.json`。

### 2.2 三轮 VMTK 提取与定向重试

每例最多执行：

1. `attempt_1_standard`：单入口到四出口的标准全目标 VMTK；
2. `attempt_2_per_target`：四个出口逐目标提取并合并；
3. `attempt_3_clean_shifted`：轻度 clean 后按 `0.35 × 开口等效半径` 内移种子，再逐目标提取。

每轮保存原始中心线、VMTK 日志、Gate、尖刺修复和长边修复审计。任何失败不会静默复用不完整中心线。

### 2.3 显式图和逐路径特征

- 去除重复无向边并建立显式中心线图；
- 冻结入口根节点、5 个表面开口、4 个稳定 `OutletId`；
- 强制图连通、无环，并检查 5 个端点与 3 个分叉节点；
- 每条 root→outlet 路径以 0.5 mm 弧长步长重采样；
- 输出 `PathId`、`OutletId`、弧长、局部半径、切线、曲率、扭率、`dR/ds`、到分叉距离和 `JunctionMask`。

### 2.4 保守自动修复

终末尖刺修复仅删除“长边末端叶节点”，且要求：父节点为度 2、父节点比异常叶节点更贴近唯一匹配开口、修复后端点数不变、图仍连通且无环。

安全长边细分不通过放宽阈值实现。仅当长边：

- 两端都是度 2 节点；
- 与两侧局部方向连续，余弦不低于 0.85；
- 整条线段采样点 100% 位于闭合管腔内；
- 长度不超过较小端局部半径的 2 倍；

才允许线性插值到不超过 2 mm 的子段。任一条件不满足，原边保持不变并继续触发硬失败。

## 3. 关键病例修复记录

### 3.1 `AAA/ruputer/XIE_JIN_QUAN`

旧审计中该例只有 4 个端点，并出现 `Target not reached`。Centerline V2 在第 3 轮 clean + 种子内移逐目标提取后得到：

- 5 个端点；
- 3 个分叉节点；
- 最大开口—端点距离 4.833 mm；
- 最大开口—端点距离/开口半径 0.411；
- 壁面距离/局部半径 p95 1.576；
- 壁面点超过 2R 的比例 0.45%；
- 最终状态 `pass`。

因此旧方案中的临时排除建议已解除；但必须在下游使用本次 V2 产物或由它重建的 bundle，不能继续读取旧缺支中心线。

### 3.2 `ILO/YANG_YU_QING-1/before`

该例使用权威 STL、固定单位和 CFD 边界合同后，首轮标准提取即通过：

- 5 个端点；
- 3 个分叉节点；
- 最大开口—端点距离 0.405 mm；
- 最大距离/开口半径 0.189；
- 壁面距离/局部半径 p95 1.453；
- 最终状态 `pass`。

这确认了旧故障属于 STL 版本/单位合同串错，而不是病例本身不可提取。

### 3.3 `AAA/unruputer/SUN_SHU_MING`

初次全量运行中该例拓扑、开口、壁内性均通过，但存在一条 13.876 mm 图边，超过 10 mm 硬阈值。进一步核查发现：

- 长边两端均为度 2 节点；
- 两端方向连续余弦分别为 0.969 和 0.999；
- 整条长边 100% 位于闭合管腔内；
- 长度/局部较小半径为 1.453；
- 三视图未见跨支或穿壁连接。

该边被判定为 VMTK 稀疏采样段，按安全细分规则拆为 7 段并插入 6 点。最终最大边长 8.723 mm，图仍为 5 端点、3 分叉、连通无环，状态由 `hard_gate_failed` 转为 `pass`。

### 3.4 其他第 3 轮重试病例

| 病例 | 最终状态 | 最大开口—端点距离 | 最大距离/半径 | 说明 |
| --- | --- | ---: | ---: | --- |
| `AAA/ruputer/LIN_LIANG_XIAO` | `pass_review` | 4.859 mm | 0.519 | 硬 Gate 通过，仅轻微触发端点 >0.5R 软阈值 |
| `AAA/ruputer/MA_XIAO_DONG` | `pass` | 2.570 mm | 0.438 | 第 3 轮通过，无软标记 |

## 4. 终末尖刺修复清单

37 例共修复 41 条。`CHENG_GUANG_SEN` 修复 3 条，`DING_JUN_FENG` 和 `LIU_JI_XIN` 各修复 2 条，其余各 1 条：

- AG：`HAN_JIAN_JUN`、`CAO_FENG_CHI`、`CHENG_GUANG_SEN`、`HOU_SHEN_QIAN`、`KANG_XI_MING`、`LIU_JIN_LIANG`、`QIN_SI_FU`、`WANG_GUI`、`YANG_YOU_SHENG`、`ZANG_YU_SHU`、`ZHANG_JING_SHUN`、`BAI_WEN_JIE`、`MA_TIAN_YI`、`MA_YU`；
- AAA/ruputer：`DING_JUN_FENG`、`FENG_LI_XIN`、`LIU_JI_XIN`、`MENG_GUANG_QIN`、`TONG_XUE_LIAN`、`WANG_AN`、`ZHANG_MAO_JIN`、`ZHOU_KE_XUN`、`LI_ZHEN_HUA`、`WANG_FU_SHUN`、`YANG_BAO_KUI`；
- AAA/unruputer：`CHEN_LIANG_FU`、`FENG_ZHI_YING`、`HAN_JIAN_FU`、`LIU_JIE`、`LIU_LI_FENG`、`SHEN_CHUN_WANG`、`MA_JIN_HE`；
- ILO：`LIU_BAO_JUN-0/before`、`LI_SHENG_WEN-0/before`、`ZHAO_CHANG_SHAN-0/before`、`SUN_XU_XIA-1/before`、`YU_XIANG_SHENG-1/before`。

每个动作的原叶节点、替代端点、开口 ID、修复前后距离和边长均保存在病例 `attempts/*/gate_metrics.json`。

## 5. Gate 结果与软复核

### 5.1 硬 Gate

最终 173/173 全部满足：

- 权威 STL 哈希与提取输入一致；
- 表面预检通过且有 5 个合法开口；
- 5 个开口与 5 个图端点唯一匹配；
- 图连通、无环；
- 无未处理 `Target not reached`；
- 无未修复异常长边；
- 中心线至少 95% 位于闭合管腔内；
- 单位不由待审计中心线决定。

全队列 `inside_fraction` 最小值为 0.9872，中位数为 0.9995。

### 5.2 软 Gate

20 例触发至少一项软阈值：

- 15 例触发开口—端点距离 >0.5R；
- 7 例触发壁面距离/局部半径 p95 >2R；
- 7 例触发壁面点 >2R 比例 >5%；
- 0 例触发端点绝对距离 >10 mm；
- 0 例触发分叉数不是 3。

完整病例表和具体数值见
[输出目录 README](../../outputs/centerline_v2_full_173_20260828/README.md#20-例软复核队列) 与
[`pilot_summary.csv`](../../outputs/centerline_v2_full_173_20260828/pilot_summary.csv)。

其中 `AAA/unruputer/LIU_WEN_QI`、`ILO/YU_XIANG_SHENG-1/before`、`AAA/ruputer/ZHOU_KE_XUN` 的壁面覆盖软指标较高，应在重建下游 wall→centerline 特征前优先查看病例三视图、CFD 壁面范围和坐标平移合同。它们的中心线拓扑和管腔内硬 Gate 均通过，因此当前标为 `pass_review`，不自动排除。

## 6. 输出合同

每例输出：

- `surface_selection.json`：权威表面、哈希、单位、CFD 边界和开口分配；
- `surface_authoritative_mm.stl`：毫米制权威表面副本；
- `centerline_graph_mm.vtp`：显式图；
- `centerline_paths_mm.vtp` / `.csv`：4 条逐路径中心线和几何特征；
- `openings_mm.vtp`：入口/出口点与稳定 ID；
- `result.json`：最终状态、Gate、重试和输出索引；
- `review_front.png`、`review_side.png`、`review_iso.png`、`review_three_views.png`；
- `attempts/`：全部 VMTK 日志和自动修复审计。

队列级输出：

- `pilot_manifest.json`；
- `pilot_summary.csv` / `.json`；
- `run_result.json`；
- `contact_sheet_index.json`；
- `centerline_v2_contact_sheet_page_001.png` ～ `009.png`。

## 7. 完整性验证

最终目录完成逐例只读验证：

- 173/173 权威 STL 文件存在且 SHA256 与冻结记录一致；
- 173/173 图 VTP 可读，均为 5 端点、3 分叉；
- 173/173 开口 VTP 均有 5 点；
- 173/173 路径 VTP 均有 4 条线及全部要求的 point arrays；
- 路径特征无 NaN/Inf；
- 422,388 行 CSV 与路径 VTP 点数逐例一致，`PathId={0,1,2,3}`；
- 692 张病例级 PNG 和 9 页总览 PNG 均可读且尺寸正确；
- `tools/centerline_v2_pilot.py`、`tools/centerline_v2_batch.py` 在 `GNN_vmtk` 环境通过 `py_compile`；
- 最终核验错误数为 0。

## 8. 切换边界和下一步

本次只完成 Centerline V2 源产物和审计包，采用版本目录写入策略，没有覆盖历史 `data_new` 中心线。

当前仍未执行：

1. 把 V2 中心线切换为 `pipeline/`、`pipeline_wss_min/` 或 `wss_pinn/` 的默认输入；
2. 重建 `data_wss_min/`、`data_wss_pinn/`、旧 bundle 和 centerline-dependent sidecar；
3. 重新生成训练 split、统计量或模型输入缓存；
4. 使用新中心线启动任何训练或比较新旧模型结果。

因此旧 bundle/sidecar 继续只保留历史复现价值。下一阶段必须先制定独立 cutover manifest，明确新中心线哈希、20 例软复核处置、重建输出目录和回滚方式，再进行下游重建。模型路线状态未改变，所以本次不修改 `docs/实验设计总纲.md`；也没有新增长期项目硬规则，因此不修改任何 `AGENTS.md`。
