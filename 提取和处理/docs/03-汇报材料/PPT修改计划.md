# 任务 A 基线汇报 PPT 修改计划（执行手册）

> **目标文件**：`docs/03-汇报材料/任务A基线模型对比汇报_LineG_WSS补充版.pptx`  
> **配套计划**：[`PPT补充计划.md`](./PPT补充计划.md)  
> **图件目录**：`docs/03-汇报材料/figures/`  
> **编制日期**：2026-05-23（5/24 实验闭合后更新）

**说明**：本文档为**手工改 PPT 的操作顺序**；不在此对话中直接改 `.pptx`（除 trivial 外）。改前请 **另存备份**。

---

## 1. 图件清单

### 1.0 脚本自动生成（`generate_v3_ppt_figures.py`）

| 文件名 | 内容 | 数据来源 | 插入页码 |
| --- | --- | --- | ---: |
| `v3p_wss_comparison.png` | V3P 五柱 WSS 对比（Main 0.365 / WSS-a 0.395 / **AsymW-a 0.394±0.005** + seed 散点 / AsymW+WssDO 0.398 / WssDO 0.379），0.40 参考线 | 各 run `summary.json` → `test_metrics_best_wss.wss_r2_wss` | **37** |
| `v3p_wss_components.png` | Main-PW / AsymW-a（三 seed 均值±std） / AsymW+WssDO 的 z/x/y 分量 R²；x/y 标注「≈0 横向瓶颈」 | 同上，分量字段 | **39**, **42** |
| `v3p_val_wss_components_history.png` | val wss_x/y/z R² 随 epoch（1×3：Main / AsymW seed1 / AsymW+WssDO）；每子图标 `best_wss_ep` 竖虚线 | 三个 run 的 `history.csv` | **39** |
| `v3d_per_domain_metrics.png` | AAA/AG/ILO 的 r2_p 与 wss_r2_wss（左右两子图，x tick 含病例数，含全局均值参考线） | `eval_by_domain_test/metrics_by_domain.json` | **48**（新增） |
| **`v3p_asymw_seed_consistency.png`** | **新增** AsymW-a 三 seed 一致性（左：3 seed 横条+均值红线；右：best_wss_ep 散点 24/57/109） | 4957 / 4999 / 5000 `summary.json` | **46** 或 **49** |

**重新生成**（300 DPI，16:9，淡米色底）：

```bash
conda activate GNN
python3 docs/03-汇报材料/tools/generate_v3_ppt_figures.py
```

### 1.0b 外部工具手工制作（**不由脚本生成**）

| 文件名 | 内容 | 参考文档 | 插入页码 | 状态 |
| --- | --- | --- | ---: | --- |
| `v3_path_map.png` | 路径 A–E 示意 + 执行顺序 | `V3_精度突破路径与发散方案.md` §3, §4–§8, §9 | **45** | 占位可弃；见 [`PPT补充计划.md` §7.1](./PPT补充计划.md#71-路径地图-v3_path_mappng) |
| `v3_todo_priority.png` | TODO 优先级表（节选） | `V3_后续优化待办.md` 路径地图 + 总览表 | **45** 或 **49** | 占位可弃；见 [`PPT补充计划.md` §7.2](./PPT补充计划.md#72-待办优先级表-v3_todo_prioritypng) |

> Prompt 全文、Mermaid 起点、章节索引见 **`PPT补充计划.md` §7**。脚本内 `fig_path_map_optional()` / `fig_todo_priority_optional()` 仅作应急占位，**默认不调用**。

### 1.1 数据路径明细

```
# V3P summary（seed1 单变量 + seed2/3 + 组合）
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260508_001936/summary.json
outputs/field/field_v3_pointnext_localpool_wss01a_geom_pw_lambda005_wall13000_near2000_split_AG_v1_seed1_20260507_001902/summary.json
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260522_124946/summary.json
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed2_20260523_124511/summary.json
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed3_20260523_124511/summary.json
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_wssdo_a_wall13000_near2000_split_AG_v1_seed1_20260523_124511/summary.json
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wssdo_a_wall13000_near2000_split_AG_v1_seed1_20260522_131813/summary.json

# history.csv（用于 val 曲线 1×3 与 best_wss_ep 标注）
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260508_001936/history.csv
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260522_124946/history.csv
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed2_20260523_124511/history.csv
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed3_20260523_124511/history.csv
outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_wssdo_a_wall13000_near2000_split_AG_v1_seed1_20260523_124511/history.csv

# V3D 分域
outputs/field/field_v3d_pointnext_localpool_probe_p01_geom_wall13000_near2000_split_data_new_v3_v3_seed1_20260521_103843/eval_by_domain_test/metrics_by_domain.json
outputs/field/field_v3d_pointnext_localpool_probe_wss01_geom_wall13000_near2000_split_data_new_v3_v3_seed1_20260521_101738/eval_by_domain_test/metrics_by_domain.json
```

### 1.2 未生成图件（及原因）

| 计划图 | 状态 | 说明 |
| --- | --- | --- |
| val loss 分量曲线 | **跳过** | `history.csv` 有 `val_loss_wall_wss` 等，但与分量 R² 叙事重复；若需要可从同目录二次脚本导出 |
| AsymW seed2/3 对比 | ✅ **已完成** | 作业 4999/5000 已闭合（2026-05-24），数据已纳入 `v3p_wss_comparison.png` / `v3p_asymw_seed_consistency.png` |
| 5001 组合 vs 单变量 | ✅ **已完成** | 作业 5001 已闭合（0.398），已纳入 `v3p_wss_comparison.png` / `v3p_wss_components.png` 第三组柱 |

**重新生成数据图命令**（不含 path_map / todo_priority）：

```bash
python3 docs/03-汇报材料/tools/generate_v3_ppt_figures.py
```

**外部图件**：按 `PPT补充计划.md` §7 Prompt 在 draw.io / Figma 等工具制作后覆盖 `figures/v3_path_map.png`、`v3_todo_priority.png`。

---

## 2. PPT 修改步骤（推荐顺序）

### Step 0 — 准备

1. 复制 `任务A基线模型对比汇报_LineG_WSS补充版.pptx` → `..._v20260523.pptx`
2. 打开 `PPT补充计划.md` 对照页码
3. 图件位于 `docs/03-汇报材料/figures/`

### Step 1 — 填充空页（P0）

| 顺序 | 页码 | 操作 |
| ---: | ---: | --- |
| 1 | **35** | 粘贴 §3.1 标题 + V3P/V3D 对比表 + PENG/4901 要点 |
| 2 | **39** | 插入 `v3p_val_wss_components_history.png`（宽 ~24cm，1×3 三子图）；副标题写 history 路径 |
| 3 | **39** | 可选右侧小图 `v3p_wss_components.png` |

### Step 2 — 更新过时主线页（P0）

| 顺序 | 页码 | 操作 |
| ---: | ---: | --- |
| 4 | **37** | 删除「WSS-a-PW 0.395 为最高」表述；换 §3.2 表（5 行 + 三 seed 均值行）；插入 `v3p_wss_comparison.png`（多 seed 版） |
| 5 | **37** | 页脚改为：「三 seed 已闭合：0.394±0.005（n=3）；建议升级 PW 母版默认权重为 AsymW 配方」 |
| 6 | **38** | 换 §3.3 val 过拟合表（含 seed2/3 + 5001 行）；强调带宽 ~0.40 即便组合也未突破 |
| 7 | **42** | 更新 history 数字；插入 `v3p_wss_components.png`（含三 seed 均值±std） |

### Step 3 — 更新路线页（P0）

| 顺序 | 页码 | 操作 |
| ---: | ---: | --- |
| 8 | **43** | 全文替换为 §3.6（4901 ✅、探针 ✅、TODO-6/7 ✅ 闭合、下一跳 TODO-5/8） |
| 9 | **44** | 全文替换为 §3.7（勿盲目加量、TODO-19） |
| 10 | **45** | 插入**外部制作**的 `v3_path_map.png` + `v3_todo_priority.png`（Prompt 见补充计划 §7，TODO-6 状态已更新为 ✅）；路径表 §3.8 |

### Step 4 — 新增 V3D 与汇总页（P1）

| 顺序 | 操作 |
| ---: | --- |
| 11 | 在 slide 45 后 **新建 4 页**（46–49），内容见 `PPT补充计划.md` §4 |
| 12 | Slide **46**：判读段写「三 seed 0.394±0.005 + AsymW+Dropout 0.398 同档」；插入 `v3p_asymw_seed_consistency.png` |
| 13 | Slide **48**：插入 `v3d_per_domain_metrics.png` + 分域表 |
| 14 | Slide **49**：TODO-6/7 闭合汇总表（真实数据），见 §3 占位页规范 |

### Step 5 — 微调（P2）

| 页码 | 操作 |
| ---: | --- |
| 36 | 脚注补「2026-05 新预处理验证」 |
| 40 | 脚注：AsymW 4957 待补 predict + top-k |
| 41 | 一句：AsymW 为过渡，下一跳 TODO-5 |

### Step 6 — 训练完成后回填（已完成）

| 触发 | 更新页 | 状态 |
| --- | --- | --- |
| 4999/5000/5001 `summary.json` 就绪 | **37** 表增 seed2/3/组合行；**49** 填数 | ✅ 2026-05-24 |
| AsymW 三 seed 汇总 | **46** 判读段补充 0.394±0.005 + 引用 seed_consistency 图 | ✅ 2026-05-24 |
| 重新跑 `generate_v3_ppt_figures.py` | 多 seed 版 `v3p_wss_comparison.png` 等四图 + 新图 `v3p_asymw_seed_consistency.png` | ✅ 2026-05-24 |

### Step 7 — 三 seed 汇总后定母版（新增）

| 动作 | 文件 | 说明 |
| --- | --- | --- |
| 升级 PW 母版默认 `wss_weights` | `training/configs/field/v3_pointcloud/V3P-Main-01-PW.json` 或 generator 母版 | 改为 `[1, 0.05, 0.05, 0.90]`（仍标注「全局坐标过渡方案」） |
| 同步实验台账 | `docs/00-规范与记录/实验记录表.xlsx` | 在备注列标记「自 2026-05-24 起母版默认 AsymW 配方」 |
| 路线文档 | `V3_精度突破路径与发散方案.md` §10 / `V3_后续优化待办.md` TODO-6 | 标 ✅ 闭合并写入母版升级建议 |

---

## 3. 占位页规范（4999 / 5000 / 5001）

**Slide 37 页脚 / Slide 49 正文**使用统一汇总表（已闭合）：

| Exp ID | 作业 | seed | 配置变更 | 状态 | test `wss_r2_wss` | `best_wss_ep` | 备注 |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| V3P-Main-01-PW-AsymW-a | 4957 | 1 | `wss_weights=[1,0.05,0.05,0.90]` | ✅ 200 ep | **0.399** | 24 | 三 seed 峰值 |
| V3P-Main-01-PW-AsymW-a | 4999 | 2 | 同 4957 权重 | ✅ 200 ep | 0.389 | 57 | 三 seed 下限 |
| V3P-Main-01-PW-AsymW-a | 5000 | 3 | 同 4957 权重 | ✅ 200 ep | 0.395 | 109 | — |
| **AsymW-a 三 seed 均值 ± std** | — | n=3 | — | ✅ | **0.394 ± 0.005** | — | 极差 0.010 |
| V3P-Main-01-PW-AsymW-WssDO-a | 5001 | 1 | AsymW + `wss_head_dropout=0.15` | ✅ 166 ep 早停 | 0.398 | 53 | vs 4957/4958，与 AsymW seed1 同档 |

**回填后检查**：
- 与 `实验记录表.xlsx` `taskA_field` / `experiment_master` 一致（row 124–128）
- 运行 `python3 docs/03-汇报材料/tools/update_experiment_xlsx_v3.py`（已扩展读取 4999/5000/5001 summary）

---

## 4. 实验记录表同步

| 动作 | 文件 |
| --- | --- |
| 已执行 | `docs/00-规范与记录/实验记录表.xlsx`（备份 `*.bak_2026-05-23`） |
| 脚本 | `docs/03-汇报材料/tools/update_experiment_xlsx_v3.py` |

**已写入/更新行**（2026-05-23 → 2026-05-24 闭合）：
- `V3P-Main-01-PW` seed1 row117：`wss_r2_wss=0.365`（best_wss）
- `V3P-Main-01-PW-AsymW-a` seed1 row124
- `V3P-Main-01-PW-WssDO-a` seed1 row125
- ~~pending：AsymW seed2/3、AsymW-WssDO seed1~~ → ✅ 已回填 rows 126–128（2026-05-24）
- `V3D-Probe-P-01` / `V3D-Probe-WSS-01` seed1（rows 129–130）

---

## 5. 质量检查清单

- [ ] 37 页最高值是否标注 **AsymW 三 seed 均值 0.394±0.005 / 峰值 0.399**（非 WSS-a-PW 0.395 单值）
- [ ] 37 页是否包含 **三 seed 一致性**说明（seed 间极差 0.010，标量稳）
- [ ] 38 / 49 页是否体现 **AsymW+WssDO（5001）≈ 纯 AsymW**（0.398，未提供额外带宽）
- [ ] 35 页是否包含 **4901** 与 PENG 双轨诊断
- [ ] V3P 与 V3D 指标是否**未混表**（脚注标注 split）
- [ ] pre-fix 4773 **−4.45** 是否标注「作废」
- [ ] 43 页是否删除「病例待提交」过时表述，并改为「TODO-6/7 ✅ 闭合 + 下一跳 TODO-5/8」
- [ ] 图件引用路径是否为相对仓库的 `figures/` 副本（插入 PPT 后嵌入）
- [ ] 页码脚标是否与总页数一致（新增后 49 页）

---

## 变更历史

| 日期 | 内容 |
| --- | --- |
| 2026-05-24 | TODO-6/7 闭合：§1.0 图件清单加入新生成的 `v3p_asymw_seed_consistency.png` 与多 seed 版四图；§1.1 数据路径补 seed2/3/组合三个 run；§1.2 未生成图件改 ✅ 已完成；§2 Step 1–4 改为闭合后的实际更新指令并新增 Step 7「定母版」；§3 占位表替换为真实数据；§4 pending 行划掉；§5 新增三 seed 一致性 / 5001 组合检查项 |
| 2026-05-23 | 初版：图件映射 + 六步修改顺序 + 训练占位规范 |
| 2026-05-23 | 拆分脚本生成 vs 外部图件；path_map/todo 改 §7 Prompt 指引 |
