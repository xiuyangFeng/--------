# WSS 最小化路线：训练结果差的联合诊断记录

- 日期：2026-07-09
- 参与：Claude（初审）+ Codex gpt-5.5 xhigh（只读复核，经 Orca 终端讨论两轮）
- 关联流程：`提取和处理/pipeline_wss_min/`（预处理）、`提取和处理/training_wss_min/`（训练）
- 关联 split：`training/splits/split_AG_wss_min_v1.json`（train 54 / val 8 / test 16）
- 关联优化文档：[WSS最小化_高精度工程化注意事项与优化建议_2026-07-08.md](WSS最小化_高精度工程化注意事项与优化建议_2026-07-08.md)
- **本轮只做只读审查与讨论，未改任何代码；代码修改待回集群后进行。**

---

## 0. 一句话结论

10 条审查方向均成立，但优先级需重排：**第 1 条（跨步 nodenumber 错位）从"最高优先主因"降级为"必须先排雷、但很可能不是主因"**；当前 `high_wss_r2` 全线为负的真正主因更像是**"峰值被系统性低估"这一簇**（目标口径 log+all-phase + xyz-only 丢绝对尺度 + 54 样本过拟合 + 无模板基线定标）。下一步不是急着改代码或重训，而是**先用现有 ckpt 做一个分位校准诊断**把假设坐实。

---

## 1. 症状（实测指标）

以 `pc_xyz_fps_w3000_peak` 为例（其余 run 同型）：

- `train_loss`：0.6 → **0.026**（训练集拟合得不错）。
- `val_r2_casemean`：在 **-0.8 ~ +0.09** 剧烈震荡，best 仅 0.094（epoch 119，基本撞运气）。
- `val_r2_field`：~0.22。
- 逐病例 `high_wss_r2`：**几乎全为负**（-1.4、-4.9、-5.8、-13.4、-15.0…）。
- `stenosis_r2` / `bifurcation_r2`：大面积为负。

signature 判断：模型过拟合了一张"统一坐标系下的平均 WSS 低频热力图"，**抓不到峰值 / 高剪切 / 狭窄 / 分叉**这些真正重要的局部。`field` 明显好于 `casemean`、`high_wss` 全负，都指向"预测场过平滑 + 高端压低"。

---

## 2. 逐条评估（含证据与联合结论）

| # | 主题 | 结论 | 优先级 | 证据位置 |
|---|---|---|---|---|
| 1 | 几何读自 `steps[0]`、WSS 读自峰值步，仅按行号配对，`read_wall_fields` 不读 `nodenumber`，跨步行序不一致会静默错位污染标签 | 成立，**P0 排雷项，但很可能非主因**（见 §3） | P0 | `pipeline_wss_min/raw_io.py:91`、`preprocess.py:152`、`preprocess.py:238` |
| 2 | 全局 log_z 统计覆盖全 81 步（含舒张期近零 WSS），但只训练峰值 → 峰值被压扁 | **部分成立**：统计**已经是 train-only**（默认 `partitions=("train",)`）；真正问题是 all-phase 而非非-train-only。应改 **peak-only + train-only** | P1 | `global_stats.py:40`、`global_stats.py:54` |
| 3 | 逐病例 `max_abs` 归一化到 [-1,1] 抹掉绝对管径/长度尺度；`coord_scale` 存在 bundle 但 xyz-only 用不到 | 成立，**可能被低估** | P1 | `config.py:155`、`preprocess.py:199`、`training_wss_min/dataset.py:157` |
| 4 | 配准朝向仍有残留不一致（9 例 world_axis override、3 例 `roll_sign_reliable=False`、5 例 `wall_pca_fallback`、1 例 `centerline_chord_ambiguous`） | 成立，但 75/78 仍 reliable，**不足以单独解释 high_wss 全负** | P2 | QA 汇总、`config.py` overrides |
| 5 | 壁面 `dist_to_wall` 恒为 0，作为输入是常数列 | 成立，浪费一维，**非主因** | P3 | `preprocess.py:230` |
| 6 | 峰值单步 = 只 54 个训练样本，跑 400 epoch；FPS 下 `resample_each_epoch` 实际无效（代码仅 `sampling!="fps"` 才重采样），结构性过拟合 | 成立 | P1 | `training_wss_min/dataset.py:240`、各 `configs/*.json` |
| 7 | ckpt 选模用 `val_r2_casemean`，8 例上噪声极大，best 撞运气 | 成立，但属**选模噪声**，非表达能力主因 | P2 | run `config.json` `ckpt_metric` |
| 8 | loss 在归一化 log 空间、评估在物理线性空间，log-MSE 天然偏向相对误差 / 平滑预测 → 低估峰值 | 成立，**很符合 high_wss 全负症状** | P1 | 训练 loss 与 `evaluate` 口径 |
| 9 | `rot_aug` 对 xyz 输入会破坏已配准的 canonical frame，自相矛盾 | 成立，但**只影响启用 rot_aug 的实验**（baseline 未开，仅 `r2_xyzgeom_rotaug`） | P2 | `training_wss_min/dataset.py:246`、`configs/r2_xyzgeom_rotaug.json:12` |
| 10 | 缺 KNN/RBF/voxel 模板基线，无法判断 0.22 field R² 是模型能力还是平均热力图 | 成立，**诊断优先级高** | P1 | 训练目录无实现；优化文档 §7.3 已列为必需对照 |

---

## 3. 收敛点 A：为什么第 1 条很可能不是主因

- `train_loss` 能从 0.6 降到 0.026（log-z 空间已很低），说明训练集**标签与坐标之间存在稳定可学的空间关系**。
- 若峰值步 WSS 相对 `steps[0]` 几何**大面积**乱序，PointNeXt 即使 4.39M 参数、固定 FPS，也很难**同时**做到低 train loss 和 val `field R²≈0.22`——模型没有 case id，只靠 xyz/形状记忆 54 例的随机点级置换，泛化到 val 得正 R² 难以解释。
- 结论：**多数病例 row order 大概率稳定；风险更可能落在个别病例 / 局部时间步**。第 1 条仍需回集群后抽查验证（廉价），但不该占主要排查精力。

> 抽查方法（回集群执行）：任取若干病例，比对 `steps[0]` 与峰值步两份 ascii 的 `nodenumber` 列（或 x/y/z 三列）是否逐行相等。只要有病例不相等，则该病例标签需按 `nodenumber` 重排后重跑。

---

## 4. 收敛点 B：下一步行动排序（先诊断，后干预）

主因假设：**"峰值被系统性低估"**（`#2/#8` 目标口径 + `#3` 丢尺度 + `#6` 小样本 + `#10` 无基线 共同作用）。

1. **【最高杠杆，零重训】现有 ckpt 分位校准诊断。**
   - 在 test/val 上，按 **true-WSS 分位分箱**看 top 10% / 5% / 1% 的 `pred/true` 中位数、线性校准斜率、`p95/p99` 比值、每病例高 WSS 区均值偏差。
   - **可判定结论**（比看 R² 更直接）：
     - top 5% true WSS 中，`pred/true` 中位数是否显著 **< 1**？
     - 线性校准斜率是否 **< 1**？
     - `pred p95` 是否系统性低于 `true p95`？
   - 若"高 true 区 pred < true 且斜率 < 1" → **坐实峰值压扁**；若"高端均值不低但位置错" → 问题转向空间定位 / 配准 / 模板。
2. **【第二快，定标】KNN / voxel 平均模板基线。**
   - 回答"0.22 field R² 是不是统一坐标平均热力图"。
   - 若 KNN 模板也 `field R²≈0.2` 且 high_wss 全负 → 任务信息量 / 目标口径本身偏模板化；若 KNN 高端更好 → 问题指向网络 / loss / 采样。
3. **【干预，非诊断第一步】peak-only log_z 统计重训。**
   - 最直接的修复实验，但成本更高、会混入随机种子 / 选模 / 训练动态，应在第 1 步坐实"高端压低"后再做。
4. **【尺度验证】`coord_scale` / `local_radius` 特征消融。**
   - 主要验证"绝对尺度缺失"，更可能改善 `casemean` / 跨病例量级 / 局部狭窄区，但不是最快证伪"峰值压扁"的动作。

---

## 5. 回集群后的代码修改清单（按优先级，待执行）

### P0（数据完整性守卫 + 可复现）
- [ ] `raw_io.read_wall_fields` 补读 `nodenumber`；`preprocess.py` 堆叠 `wss_ts` 前加**逐时间步 `nodenumber` 对齐守卫**（不一致则重排；集合/长度不一致则报错并写审计）。附加廉价自检：抽查相邻步坐标列是否逐行相等（坐标本应静态）。
- [ ] 确认本地 checkout 缺失的训练源码已在集群完整存在并可复现：`training_wss_min/config.py(ExpConfig) / train.py / evaluate.py / pointnext.py / metrics.py` 与 `split_AG_wss_min_v1.json`（本地仅有 `dataset.py` + `make_configs.py`）。

### P1（最像当前结果差的主因）
- [ ] 峰值路线改用 **peak-only + train-only** 的 log_z 统计（`global_stats.py` 增 peak-only 开关；`dataset.normalize_wss` 对齐同一份统计）。
- [ ] 给 dataset 增加 `coord_scale`（及可选 `local_radius`）输入通道，做 `xyz` vs `xyz+coord_scale` vs `xyz+local_radius` 消融。
- [ ] 建立 **KNN / voxel 模板基线**脚本，作为所有深度模型的定标对照。
- [ ] 正视 54 样本天花板：要么切 all-phase（**必须给网络加 `phase` / `inlet_flow` 特征**，否则同坐标多相位多标签、任务不可辨识），要么在峰值上做有效增广 / 强正则；相应缩短 epoch 或加早停。
- [ ]（可选）loss 侧：peak-only 统计基础上对高 WSS 区加权（仅用训练集标签定义，避免泄漏），或改物理/混合空间监督。

### P2（放大波动 / 污染个别实验）
- [ ] 选模指标从 `val_r2_casemean`（8 例噪声大）改为更稳的 val 物理 RMSE，并看 per-case 分布。
- [ ] `rot_aug` 只用于旋转不变的 geom-only 组，不与 xyz 输入混用。
- [ ] 复核残留朝向不一致病例（override / `roll_sign_reliable=False` / `wall_pca_fallback`）对 casemean 的影响。

### P3
- [ ] 移除 / 屏蔽壁面点上恒为 0 的 `dist_to_wall` 输入列。

---

## 6. 备注

- 本轮无法在本地复现训练结果：`data_new/`、`data_wss_min/`、训练核心源码与 split 均在集群，本地 checkout 只保留了 `dataset.py`、`make_configs.py`、configs 与 runs 产物。所有 `file:line` 证据基于本地可见文件；回集群后以集群版本为准再次核对。
- 与 [WSS最小化_高精度工程化注意事项与优化建议_2026-07-08.md](WSS最小化_高精度工程化注意事项与优化建议_2026-07-08.md) 的关系：那份文档覆盖"工程闭环"（元数据可逆、采样、评估口径、防虚高精度），本文档补上"**为什么当前结果差**"的因果诊断与最小验证顺序，两者配合使用。
