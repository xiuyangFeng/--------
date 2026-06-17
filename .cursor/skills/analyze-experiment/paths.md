# 实验产物路径速查（V3 默认）

## Run 目录

- `outputs/field/field_v3_*` / `experiment_index.csv`
- V3P：`split_AG_v1` · V3D：`data_new` + `split_data_new_v3`（**分表**）

## 单 run

| 文件 | 说明 |
| --- | --- |
| `summary.json` | `test_metrics` / `test_metrics_best_wss` |
| `history.csv` | 训练曲线 |
| `config.snapshot.json` | `meta.exp_id` 判别 V3P/V3D/路径 |
| `run_manifest.json` | loss / head / `graphs_subdir` |

## 分析对比图

`outputs/field/plots/analysis_compare/<slug>/` — 见 `docs/00-规范与记录/实验分析对比图目录说明.md`

```bash
python -m training.scripts.plot_experiment_analysis_compare \
  --run-dir <current> --baseline-run-dir <4957_or_stated_baseline> \
  --slug <slug> --primary-metric wss_r2_wss --checkpoint best_wss

python -m training.scripts.plot_training_history \
  --run-dir <current> --run-dir <baseline> \
  --output-dir outputs/field/plots/analysis_compare/<slug> \
  --compare-metric val_loss
```

## Oracle 产物（0 重训）

| 类型 | 路径 |
| --- | --- |
| F0（路径 F） | `outputs/field/f0_decision/*.json` |
| G0（路径 G） | `outputs/field/f0_decision/v3_g0_oracle_<date>.json`（**不覆盖** F0） |

## V3 文档（分析时 · 2026-06-06 目录）

分层读法 → [v3-docs.md](v3-docs.md)（**勿**默认打开 V1 状态表）

| 用途 | 路径 |
| --- | --- |
| 导航入口 | `docs/01-任务/任务A/03-V3路线/README.md` |
| 路径 G 主规划 | `00-当前主线/路径G_下一代架构与精度突破方案_2026-06-05.md` |
| 待办 / 跟踪 | `01-执行与待办/V3_后续优化待办.md` · `V3_实验执行跟踪日志.md` |
| 路径 F Go/No-Go | `02-历史路线/V3_发散优化探索路线.md` §6 / §10 |
| A–E 历史 | `02-历史路线/V3_精度突破路径与发散方案.md` |
| 三域 / OOD | `03-数据与质量/V3_三域数据.md` |
| 推进 | `docs/02-推进与变更/代码修改与实验推进记录.md`（文首） |

## V3P 母版与近期对照

| 角色 | 作业 | `wss_r2_wss` |
| --- | ---: | ---: |
| AsymW-a 母版 | **4957** | **0.399** |
| 三 seed 带宽 | 4957/4999/5000 | 0.394±0.005 |
| PGradFeat | 5277 | 0.406 |
| PCv2-BLContext | 5311 | 0.399 |

`experiment_index.csv` 查 `run_dir`。

## V3D 主基线

- **WSS-01** post-4901：**0.243**（`V3D-Probe-WSS-01`）
- val15 **5331**：强 No-Go，不作新基线

## 其他 plots（非日常分析必扫）

- `plots/training_curves/`、`plots/optimization/` — V1 campaign
- `docs/03-汇报材料/figures/` — PPT 汇总
