---
name: analyze-experiment
description: >-
  分析 V3（V3P/V3D）场重建实验：读 summary/history、对比 AsymW 等基线、出 analysis_compare 图、回填 V3 文档与推进记录。
  在用户分析实验、判 Go/No-Go、解读 V3P/V3D run 时使用。非 V3 路线仅当用户明确指定。
---

# 实验结果分析（V3 优先）

## 触发

「分析实验」「对比母版」「No-Go 原因」等；或给出 `V3P-`/`V3D-` `exp_id`、作业号、`outputs/field/...`。

**默认假定 V3**。仅当 `exp_id` 为 `A-`/`V2P-` 或用户点名 V1/V2 时，改用文末「非 V3 附录」。

**禁止**：凭记忆或旧文档数字下结论；必须读 run 目录内文件。

---

## 0. 锁定对象与读文档（控制上下文）

1. **run 目录**：`experiment_index.csv` 或用户给的路径。
2. **分支**：`config.snapshot.json` → `meta.exp_id` → **V3P** / **V3D**，并判别路径前缀（见下表）。
3. **导航入口**：先扫 `docs/01-任务/任务A/03-V3路线/README.md` 文首「当前状态」+「口径对照」（防混表、知下一跳）。

### 0.1 路径判别 → Go/No-Go 文档

| `exp_id` 模式 | 路径 | Go/No-Go 读哪里 |
| --- | --- | --- |
| `V3P-G-*` | **G**（当前主线） | `00-当前主线/路径G_下一代架构与精度突破方案_2026-06-05.md` **§10** |
| `V3P-F-*` | F（历史发散） | `02-历史路线/V3_发散优化探索路线.md` **§6** |
| `V3P-*` 其他 | A–E / 母版线 | 跟踪日志块内判据；历史路径见 `02-历史路线/V3_精度突破路径与发散方案.md` |
| `V3D-*` | C / 三域 | 跟踪日志块 + 待办；**禁止与 V3P 比绝对值** |
| G0 / oracle only | G0 | 路径 G **§3** 触发判据；产物 `outputs/field/f0_decision/v3_g0_oracle_*.json` |

### 0.2 分层读文档（勿通读整本待办/发散/推进记录）

| 层 | 内容 |
| --- | --- |
| **L0** | `03-V3路线/README` 当前状态 · `01-执行与待办/V3_实验执行跟踪日志` 顶 1–2 块 · 推进记录文首 3–5 条 |
| **L1** | `01-执行与待办/V3_后续优化待办` 相关 TODO · 上表 Go/No-Go 文档 · `V3P-F-*` 再读发散 **§10** 最新一条 |
| **L2+** | 写下一步/立项时：路径 G §9/§13 · 精度突破路径 · `V3P_后续优化执行计划` **相关节**（与 G 冲突以 G 为准） |

完整清单与「明确不读」→ [v3-docs.md](v3-docs.md) · 规则 → `.cursor/rules/v3-context.mdc`。

### 0.3 对照基线

| 分支 | 基线 | 主指标 | checkpoint |
| --- | --- | --- | --- |
| **V3P** | AsymW-a **4957**（0.399；三 seed **0.394±0.005**） | `wss_r2_wss` | `best_wss` |
| **V3P 路径 F** | 立项写的中间基线（如 VelSup **5266**、PGradFeat **5277**）须在文中写明 | 同上 | `best_wss` |
| **V3P 路径 G** | **4957** + G1 阶梯时 **上一档** run | `wss_r2_wss` + 分量 `wss_x/y` | `best_wss` |
| **V3D** | **WSS-01** post-4901（**0.243**）；val15 **5331** 已强 No-Go | 分域 WSS / `r2_p` | 按待办 |

次要指标：`r2_p`、`r2_vel_mag`、`wss_z`、`wss_x`、`wss_y`；病例/区域级在路径 G §10、待办 TODO-8/20 或 F/G 门禁要求时读 `regional_eval` / full eval。

---

## 1. 收集产物（必做）

| 文件 | 用途 |
| --- | --- |
| `summary.json` | `test_metrics` / `test_metrics_best_wss` |
| `history.csv` | 曲线；V3 常看 `val_score`、`wss_*` |
| `config.snapshot.json` | 单变量 diff |
| `run_manifest.json` | loss / head / `graphs_subdir` |

路径：[paths.md](paths.md)

---

## 2. 曲线与精度

- **曲线**：收敛、过拟合、`best_model` vs `best_wss` 分叉；路径 F 辅助 loss 常伴 `r2_vel_mag`↑、`wss_r2_wss`↓；**5331** 类 val→test 脱节须单独写明。
- **test**：以 `summary.json` 为准；缺字段 → `recompute_dual_test_metrics`（见推进记录）。
- **判定**：
  - `V3P-G-*` → 路径 G **§10**（分量破 **0.10** 为结构性信号）
  - `V3P-F-*` → 发散路线 **§6**
  - `V3D` → 跟踪日志/待办分域阈值；对照 **0.243** 而非 V3P

---

## 3. 对比图（强烈建议）

目录：`outputs/field/plots/analysis_compare/<YYYYMMDD>_<exp>_vs_<baseline>/`

```bash
python -m training.scripts.plot_experiment_analysis_compare \
  --run-dir <current> --baseline-run-dir <4957_or_stated_baseline> \
  --slug <slug> --primary-metric wss_r2_wss --checkpoint best_wss

python -m training.scripts.plot_training_history \
  --run-dir <current> --run-dir <baseline> \
  --output-dir outputs/field/plots/analysis_compare/<slug> \
  --compare-metric val_loss
```

V3D：`--primary-metric` 按待办；基线用 WSS-01 run 目录；**禁止**与 V3P 画在同一「谁更好」柱图。

规范：`docs/00-规范与记录/实验分析对比图目录说明.md`

---

## 4. 原因分析

配置单变量 → 曲线 → 推进记录/待办/路径 G §13 中的**同方向先例**（如 5311 PCv2、5331 val15、路径 F 批次）→ 机制 → Go/No-Go。

路径 G 实验须分开报告：点级 R²、分量/角度、病例 p95/Pa、高 WSS 区域（§10 四层门禁）。

---

## 5. 下一步

### 5.1 清单内（必写 1–3 条）

来源：**仅** `01-执行与待办/V3_后续优化待办.md` + `README.md` 当前状态 + 推进记录/跟踪日志已写的下一跳（如 TODO-1 数据扩池、5311 clinical-pa、路径 G G0→G1-a）。**不要**从 V1 清单或实验总纲拉项。

路径 G 训练须先过 G0 oracle（§3）；H1–H14 观察池**不**直接开训。

### 5.2 新方案提议（可选，≤2 条）

清单外、有本次证据的单变量方向。须对齐 G0/oracle 与对应 Go/No-Go 表；若属 H 观察池，注明「需低成本 oracle 后才升级 TODO」。

---

## 6. 回填（V3 默认执行）

| 顺序 | 文档 |
| --- | --- |
| 1 | `docs/02-推进与变更/代码修改与实验推进记录.md` **文首**（含对比图路径、Δ、判读、下一步、新方案） |
| 2 | `01-执行与待办/V3_实验执行跟踪日志.md` **顶部** |
| 3 | `V3P-F-*`：`02-历史路线/V3_发散优化探索路线.md` **§10 顶部** |
| 4 | `01-执行与待办/V3_后续优化待办.md` 更新对应 TODO 状态行 |

**通常不回填**：路径 G 规划正文（除非门控/执行顺序变化）、`任务A实验状态表`（V1）、`实验设计总纲`，除非用户要求正式入账 V1 表或 xlsx。

**G0 only**：写 `v3_g0_oracle_*.json` + 推进记录；路线结论变化时同步 `README.md` 当前状态。

---

## 7. 回复结构

```markdown
## 实验：<exp_id>（V3P|V3D · 路径 A–G · 作业 xxx）

### 对照（基线 + 口径 + 是否混表）
### 结果（test）| 指标 | 本 run | 基线 | Δ |
### 曲线
### 对比图 · `outputs/field/plots/analysis_compare/<slug>/`
### 原因与判读（Go/No-Go · 引用 §6/§10）
### 下一步（V3 待办 / README 当前状态）
### 新方案提议（可选）
### 已回填文档
```

---

## 8. 约束

- 不训练、不 Slurm、不批量 WSS，除非用户明确要求
- 不加载 V1/V2/B/C/D 文档除非用户明确历史对照
- `_archive/` 仅用户点名或查早期 V3P 历史时定向打开
- 出图需 GNN 环境；执行前确认 conda

---

## 附录 · 非 V3（仅用户明确时）

| 路线 | 基线 | 主指标 |
| --- | --- | --- |
| V1 | `A-Opt-05` | `interior.rmse_vel_mag` |
| V2 | 待办/状态表 WSSP 锚点 | `wss_r2_wss` / `r2_p` |

此时可读 `任务A实验状态表`「战略锚点」；对比图 `--checkpoint best_model` 等按路线调整。
