# `training/splits` 版本区隔（V3 · `data_new` 全域）

## 必须与旧结果「分表」，禁止混读

V3 **全量实验**候选：**`data_new/AAA`、`data_new/AG`、`data_new/ILO`** 三域中，经全流程（步骤 1–5）齐套、且**不在** `pipeline/export_gap_preprocess_queue.PREPROCESS_DENYLIST`（当前 **15** 条）内的所有叶级病例。

| 产物 | 数据域 | 用途 |
|------|--------|------|
| `split_AG_v1.json` / `cases_AG_v1.txt` | 仅 **`data_new/AG`** 的患者级划分 | **历史主线**已在 AG 上报过的 PointNeXt/Transformer、`V3P-*` **旧 split** 结论；**数值与命名不得**与新 **`split_*_v3`**（三域候选池划分）无条件混表。 |
| **`cases_data_new_v3_candidate_pool.txt`**（终审作业写出） | **AAA + AG + ILO**，扣 **`PREPROCESS_DENYLIST`** 且 **`matched_frame_count` 与各步产出对齐** | **V3 `data_new` 正式全域候选**，供 **`make_split`**、manifest。**AG** 侧 OOD 剔除（如 **`AG/fast/PENG_JI_MING`**）与 **denylist**、`split_AG_v1.excluded_cases` 对齐。 |
| **`split_data_new_v3.json`** | **AAA + AG + ILO**（由上项名单 **按域 AAA→AG→ILO 分层** 切分） | **`split_version: split_data_new_v3`**；默认 **seed=1**，**0.7 / 0.1 / 0.2**（每域内套用相同比例后合并）；详情见 JSON 内 **`split_meta`**。训练 **`split_file`** 请指向本文件；**勿与 `split_AG_v1` 历史指标混表**。 |

## `PREPROCESS_DENYLIST` 与 AG 的关系

- **AG** 中已文档化、并写入 `split_AG_v1.json` **`excluded_cases`** 的病历（OOD/CFD 异常），**必须**继续在 **`PREPROCESS_DENYLIST`** 中保留，才能保证 **导出队列、`export_post_preprocess_queue`、`make_split` 候选、`run_v3_aaa_ilo_prepool_audit` 终审**共用同一剔除口径。
- 若只在 `split_AG_v1` 移除而不进 denylist，三域并行清单仍会把该 AG 路径当作合格病例——与「AG 不参与训练」的结论冲突。

详见：

- `docs/01-任务/任务A/03-V3路线/V3_后续优化待办.md` **TODO-1**
- `docs/01-任务/任务A/03-V3路线/V3_数据质量与OOD诊断记录.md`

## 如何刷新候选清单与终审摘要

在 `pipeline/archive/onetime_batch_jobs/` 下提交 `sbatch run_v3_prepool_audit.slurm`；成功后查看：

- `outputs/field/diagnostics/v3_data_new_prepool_audit_<JOB_ID>/prepool_audit_summary.json`
- 本目录 **`cases_data_new_v3_candidate_pool.txt`**（一行一条相对 **`data_root`（默认 `data_new`）** 的路径）

> **旧文件名** `cases_AAA_ILO_v3_candidate_pool.txt` 已由 **上述文件**取代（仅 AAA+ILO 的子集口径不再作为 V3 全量定义）。

## 如何生成（或重做）`split_data_new_v3.json`

```bash
cd <repo-root>
python -m training.scripts.make_split \
  --cases-file training/splits/cases_data_new_v3_candidate_pool.txt \
  --output training/splits/split_data_new_v3.json \
  --split-version split_data_new_v3 \
  --seed 1 \
  --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 \
  --stratify-by-domain
```

不传 **`--stratify-by-domain`** 时为全盘随机 shuffle（不推荐三域混训）。

## `split_data_new_v3_val15`（路径 C · TODO-3 · val 15%）

与 `split_data_new_v3` **分表对照**；**不改** `normalization_params_global.json` / 图资产（post-4901 同一套 renorm）。

```bash
cd <repo-root>
python -m training.scripts.make_split \
  --cases-file training/splits/cases_data_new_v3_candidate_pool.txt \
  --output training/splits/split_data_new_v3_val15.json \
  --split-version split_data_new_v3_val15 \
  --seed 1 \
  --train-ratio 0.65 --val-ratio 0.15 --test-ratio 0.2 \
  --stratify-by-domain
```

定稿规模（seed=1，257 例）：**train 165 / val 37 / test 55**（原 v3：**179 / 25 / 53**）；分域 val：AAA **9**（+3）/ AG **12**（+4）/ ILO **16**（+5）。

训练配置：`V3D-Probe-WSS-Val15_seed1.json`（单变量仅 `split_file`）。

## `split_data_new_v3` 定稿后的全局归一化与转图（步骤 4–5）

- **目的**：用 **`split_data_new_v3.json`** 的 **`train_cases`** 计算 **`normalization_params_global.json`**（**意图** train-only），并对 **五域数据源** 下全部叶目录应用同一套参数且重建 **`processed/graphs`**。
- **集群模板（一次性作业）**：**`pipeline/archive/onetime_batch_jobs/run_v3_split_data_new_renorm_regraph.slurm`**
  - 提交方式（**必须在仓库根目录**）：`sbatch pipeline/archive/onetime_batch_jobs/run_v3_split_data_new_renorm_regraph.slurm`
  - **`SLURM_SUBMIT_DIR`** 定位工程根（避免计算节点 spool 脚本 **`$0`** 导致 **`mkdir logs` 失败）。
- **已知主干代码问题**：**`pipeline/normalize.py`** 的 **`train_cases`** 与 **ILO** 嵌套叶目录匹配时，存在 **`after`/`before` 误匹配**，全局统计**可能混入非 train 路径**；修复后应 **重跑本节的 normalize + `convert_to_graph`**。详见 **`docs/02-推进与变更/代码修改与实验推进记录.md`** **2026-05-17** 条目。
- **已完成作业参考**：**4664**（**267/277** 成功；失败 **10** 例缺 **`coord_normalized`**，与 **`cases_data_new_v3_candidate_pool.txt`** 名单无交集）；日志：**`logs/v3_renorm_regraph_4664.out`**。
