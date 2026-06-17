# 外部 baseline 实验记录规范

> 上位文档：[实验记录填写规范](实验记录填写规范.md) · [实验设计总纲](../实验设计总纲.md)  
> 适用范围：`external_baselines/` 下所有外部论文复现（当前首批：PointNetCFD）。

## 1. 文档定位

外部 baseline 与任务 A V1/V2/V3 **分表**；不得把外部模型结果写入 V3 主线实验状态表或 AsymW 对照列，避免口径混表。

## 2. 三层记录 + 梳理（粒度不同）

| 层级 | 路径 | 更新时机 | 内容 |
| --- | --- | --- | --- |
| **单次 run** | `outputs/.../<run_dir>/analysis_report.md` | 每个 Job 结束（训练脚本自动） | 当次 test 指标快照 |
| **实验族** | `external_baselines/<baseline>/experiments/<experiment_name>/实验分析记录.md` | 每个 Job 结束自动/分析时 | 该配置族指标与判读 |
| **梳理（批次）** | `docs/paper_reproduction/papers/<paper_id>/梳理记录.md` | **一轮计划实验矩阵全部跑完后** 写 1 次 | 汇总 + gap 原因 + Go/No-Go |
| **总账** | `docs/02-推进与变更/代码修改与实验推进记录.md` 文首 | 矩阵完成且梳理写好时 | 1 条摘要 + 梳理链接 |

梳理规范：[04-梳理记录规范](../paper_reproduction/04-梳理记录规范.md) · 模板：`papers/_template/梳理记录.md`

## 3. 路径约定（实验族 / run）

```text
external_baselines/<baseline>/
├── README.md
├── configs/…
├── experiments/
│   ├── README.md                    ← 实验族索引
│   └── <experiment_name>/
│       └── 实验分析记录.md          ← 该配置族最新分析（覆盖或文首追加，由 baseline README 约定）
└── …

outputs/external_baselines/<baseline>/<run_dir>/
├── manifest.json
├── metrics_test.json
├── history.csv
└── analysis_report.md               ← 当次 run 快照（训练结束自动生成）
```

- `<experiment_name>` 与配置中 `run.experiment_name` 一致（如 `pointnetcfd_geom_vp`）。
- `<run_dir>` 命名：`<experiment_name>_<split>_seed<seed>_<timestamp>`。

## 4. 指标与口径

- 默认报告 **z-score 归一化空间** 的 MSE（`loss_mse`）、RMSE、MAE、R²；与 `metrics_test.json` 字段一致。
- 物理单位（Pa、m/s）须显式写清是否已反归一化；未反归一化时不得写物理单位表。
- split、seed、`data_root`、`split_file` 须从 `manifest.json` 或 `config.json` 抄录，禁止手填猜测。

## 5. 分析任务检查清单

**每个 Job 结束**（不写梳理）：

1. 读 `outputs/.../<run_dir>/manifest.json` 与 `metrics_test.json`。
2. 确认 `analysis_report.md` 与 `experiments/<experiment_name>/实验分析记录.md` 已更新。

**一轮计划实验矩阵全部跑完后**（写梳理）：

1. 按 [04-梳理记录规范](../paper_reproduction/04-梳理记录规范.md) 填写或追加 `docs/paper_reproduction/papers/<paper_id>/梳理记录.md`。
2. 更新 **`docs/02-推进与变更/代码修改与实验推进记录.md` 文首** 1 条（含梳理链接与 Go/No-Go 一句）。

## 6. 与任务 A 实验记录表的关系

- 外部 baseline **不**写入 `docs/00-规范与记录/实验记录表.xlsx` 的 V3 主线列，除非论文制表单开 sheet（如 `external_baselines`）。
- 若论文需要汇总表，从各 `experiments/*/实验分析记录.md` 抽取，勿从 V3P/V3D 状态表复制。

## 7. 常见错误

1. 只更新总账、不在 baseline 项目根下写 `experiments/<name>/实验分析记录.md`。
2. 把 PointNetCFD test 指标与 V3P AsymW-a **同表对比**而不注明 split/模型差异。
3. 覆盖 `实验分析记录.md` 时删除历史 run 链接，导致无法回溯 `run_dir`。
4. 未写清指标为归一化空间还是物理单位。
5. **每个 Job 就改 `梳理记录.md`** → 应等整轮矩阵完成再写（见 [04-梳理记录规范](../paper_reproduction/04-梳理记录规范.md)）。

---

**关联代码**：`external_baselines/<baseline>/reporting.py`（训练结束写 `analysis_report.md` 并同步 `experiments/`）。
