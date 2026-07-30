# WSS-PINN 独立实验线

> 建立日期：2026-07-30
>
> 当前状态：**PINN 专用 train138/test35 与 173 例 sidecar 已冻结；full F0-UP completed / audited / No-Go；full F1 被配对科学 Gate 阻断且未提交；F2 blocked**

本目录用于开发以峰值壁面 WSS 为最终目标、以体域
\(u,v,w,p\) 神经场为训练期辅助变量的 physics-informed surrogate。
它与现有 `pipeline_wss_min/`、`training_wss_min/` 隔离，现已提供可运行的
P0/P1 工具、统一训练/评估入口、阶段配置校验、测试和 Slurm 提交器。

## 入口

- [路线总入口](../docs/02-推进与变更/WSS_PINN/README.md)
- [阶梯实验矩阵与进度跟踪](../docs/02-推进与变更/WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)
- [第一性原理与物理约束方案](../docs/02-推进与变更/WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md)
- [下一智能体目标提示词：实现并推进至 F1](../docs/02-推进与变更/WSS_PINN/WSS_PINN_下一智能体目标提示词_推进至F1.md)

## 隔离原则

| 层 | 路径 | 读写规则 |
| --- | --- | --- |
| 原始 CFD | `data_new/` | 只读 |
| 已验收 WSS 几何母版 | `data_wss_min/` | 只读 |
| 既有 WSS 预处理/训练 | `pipeline_wss_min/`、`training_wss_min/` | 只读复用 |
| PINN 代码 | `wss_pinn/` | 本路线独占 |
| PINN 派生数据 | `data_wss_pinn/` | 本路线独占，不反写旧 bundle |
| PINN 结果 | `outputs/wss_pinn/` | 本路线独占 |
| PINN 记录 | `docs/02-推进与变更/WSS_PINN/` | 本路线执行真源 |

因此，新建本目录本身不会改变任何既有模型、数据或实验结果。后续即使需要复用
PointNeXt-R/LocalGeoPE，也只通过 adapter 读取冻结 checkpoint，不直接修改旧训练入口。

## 已实现的代码结构

```text
wss_pinn/
├── AGENTS.md
├── README.md
├── config.py
├── data/
│   ├── raw_io.py
│   ├── sidecar.py
│   ├── sampling.py
│   └── dataset.py
├── models/
│   ├── anchor_adapter.py
│   ├── geometry_encoder.py
│   └── field_decoder.py
├── physics/
│   ├── rheology.py
│   ├── residuals.py
│   ├── wall_shear.py
│   └── nondimensionalize.py
├── tools/
│   ├── audit_interior_identity.py
│   ├── audit_udf_units.py
│   ├── velocity_to_wss_oracle.py
│   ├── cfd_residual_oracle.py
│   ├── audit_wall_normals_zones.py
│   ├── build_physics_sidecar.py
│   ├── audit_full_dataset.py
│   ├── audit_sidecar_dataset.py
│   ├── analyze_f1_matrix.py
│   ├── prepare_full_training_configs.py
│   └── preflight.py
├── configs/
├── tests/
├── cluster/
├── train.py
└── evaluate.py
```

F0-U、F0-UP、F1 使用同一套 `train.py`、`evaluate.py` 和
`cluster/{preflight.slurm,run_experiment.slurm,submit_experiment.py}`。阶段差异只在
JSON 配置的输出头与 loss 权重中表达；配置预检拒绝 F1 中任何非零 momentum 或
WSS-physics 权重。full-data 配置只在 source/sidecar/matrix 三个报告均通过且
哈希绑定当前 split/manifest 后生成；full F1 还会逐项校验其 full F0-UP
对照的数据、采样、模型、训练协议和共享 loss。

## 快速验证

```bash
source /public/newapps/anaconda3/etc/profile.d/conda.sh
conda activate GNN
python -m unittest discover -s wss_pinn/tests -p 'test_*.py' -v
python -m wss_pinn.train \
  --config wss_pinn/configs/f1_pilot_v2.json \
  --dry-run --device cpu
python wss_pinn/cluster/submit_experiment.py \
  --config wss_pinn/configs/f1_pilot_v2.json \
  --dry-run
```

真实提交不加 `--dry-run`。提交器会检查前级科学 Gate。原 F1
`0.1/0.1` Jobs `11045→11046` 的 No-Go 结果保留；后续八臂内部诊断
Jobs `11049…11063→11064` 已选中 `continuity=1e-4`、`no-slip=10`，
best/last 均通过 pilot Gate。PINN 专用 `train138/test35` 已冻结，
173 例 source/sidecar Gate 已通过；full F0-UP Jobs `11073→11074`
已完成，但全量能力 Gate 为 No-Go；full F1 未提交。

## 不允许的捷径

- 不把 PINN loss 直接塞进 `training_wss_min/train.py`。
- 不原地扩写 `data_wss_min/**/bundle.npz`。
- 不把 `random5000` 全壁面点当成完整 physics batch。
- 不一次同时打开 continuity、no-slip、WSS consistency 和 momentum。
- 不用 `test36` 选择 loss 权重后再把它称为独立测试集。
- 不在未记录 parent/config/data SHA 的情况下启动正式训练。

## 运行状态

- P0-A/B/C/D：三个 pilot（AG/AAA/ILO 各一例）均 completed；P0-D 同时确认
  exact face zone/connectivity 不可得，hard flux/RCR residual blocked。
- P1：`data_wss_pinn/pilot_v1/sampling_manifest.json` completed，三病例均为
  5k wall / 8k near-wall / 8k core，聚合 SHA256
  `c5bdd7b7ec89b43b16d8a1eccd664caf9490b39eb4aca6e1c4e8ec05177678f8`。
- F0-U v1：Jobs `11037→11038` completed，但速度 R² 为
  AG/AAA/ILO `0.654/0.586/0.567`，未达到 0.95，No-Go。
- F0-U v2：Jobs `11039→11040` completed；case-balanced WSS R² `0.9906`，
  但严格逐病例 `u/v/w/speed` Gate 未全部达到 0.95，No-Go。
- F0-U v2ext：从 v2 `last.pt` 显式恢复并只增加 12000 epoch；Jobs
  `11041→11042` completed。逐病例全部 velocity 聚合、`u/v/w/speed`
  的最小 R² `0.9888`，通过严格 Gate。
- F0-UP v2ext：Jobs `11043→11044` completed。best 的逐病例全部 velocity
  聚合、`u/v/w/speed` 最低 R² `0.9872`，pressure R² 最低 `0.9987`；
  best/last 均通过严格 Gate。
- F1 v2ext `0.1/0.1`：Jobs `11045→11046` 均 completed，best/last 已审计。best
  mean continuity RMS `70.1526→0.3359`，但 mean no-slip RMS
  `0.12447→0.13396`，mean velocity R² `0.99502→0.82491`；No-Go。
  该失败证据保留。
- F1 内部诊断：八个 no-slip-only、continuity-only 和联合权重臂全部
  completed；唯一双 checkpoint 通过臂为 `continuity=1e-4`、
  `no-slip=10`。best/last continuity ratio `0.0769/0.0715`，
  no-slip ratio `0.1232/0.1465`，最低 velocity R² `0.9689/0.9697`。
  汇总为 `outputs/wss_pinn/matrices/f1_diagnostics_20260730/report.json`。
- PINN 专用 full split：用户确认只从 PINN test 移除
  `AAA/ruputer/SHI_YUN_XI`，train138 不变、不补位；新 split SHA256
  `964d7021…9361f2b`。baseline 原 split SHA256 仍为
  `d16fc497…bdd8f1`，未修改。
- 全量 source/sidecar：Job `11070` 为 173/173 source pass；
  Jobs `11071→11072` 为 173/173 sidecar build/audit pass；manifest
  SHA256 `81a32066…43b5e14`。
- full F0-UP：GPU preflight `11073`、train/eval `11074` 均 completed；
  best 的 train/test velocity mean R² 为 `0.6568/0.1800`，last 为
  `0.6591/0.1770`，严格能力 Gate 失败，判为 audited / No-Go。
  配置 SHA256 `d27c215c…12e073`。
- full F1：配置 SHA256 `c41cf992…11273`，dry-run code-ready；
  full F0-UP 的配对 Gate 报告 SHA256 `daccfd1d…75cfed` 为 false，
  因此 blocked / not submitted。
- F1 始终只含 continuity + no-slip，momentum/WSS-physics 均为 0；
  F2 blocked。
