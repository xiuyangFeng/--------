# WSS 最小化路线代码修改与实验推进记录

> 用途：记录 WSS 相关独立路线，包括 `pipeline_wss_min/`、`training_wss_min/`、`wss_mri_calculator/` 与 `wss_pinn/` 的代码、数据、图件和实验推进。
> V3P / 训练主线 / 通用代码修改记录见：[代码修改与实验推进记录](代码修改与实验推进记录.md)。
> 当前执行入口：[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)（🧊冻结） / [训练实验跟踪](WSS最小化_训练实验跟踪.md)（🧊冻结） / [WSS-PINN 当前入口](WSS_PINN/README.md) / [velocity→WSS V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)。
> **滚动切卷**：本文件只保留 2026-08 以来的条目；2026-07 条目（PointNet 矩阵、新队列审计、WSS-PINN F0/F1）见历史卷
> [2026-07卷](_archive/WSS最小化_代码修改与实验推进记录_2026-07卷.md)。主文件超过约 1500 行或跨季度时，把最旧月份整月切入 `_archive/` 新卷并更新本索引。

## 2026-08-27｜完训臂 WSS 下游（与 0–14 同协议）· 已完成 · 未填工作簿

**本次主要修改**：用户要求对已评完的 18–38、40 补跑 WSS，口径与 0–14 相同。
node04 GPU1 从 11:45 跑到 15:35，22 臂全部完成。xlsx 仍不改。

**口径（回答「全量还是输入点」）**：
- **速度推理是峰值时刻全场体点**（V1 `interior_coords`），不是训练 2k/5k query。
- 瞬态只解 **peak frame** 到该全体域，不是 81 帧。
- **WSS 本身是 test35×1200 壁面采样**，不是全壁面；Profile-Secant V3 冻结。
- speed OLS 仍用解剖 ROI 上的全场速度。

**对应代码/文档**：
`docs/03-汇报材料/tools/update_wss_pinn_v4_workbook.py arm --output-dir …`；
产物 `outputs/wss_pinn/audits/v4_wss_18_38_40_20260827/`；
汇总 `outputs/wss_pinn/audits/v4_wss_completed_20260827.json`（含 0–14 共 37 臂）；
日志 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/gpu1_wss_18_38_40_20260827.log`。

**实验结果摘要**：WSS R²_cb 仍为负或贴零。DATA 最差（约 −3～−10）；PDE 臂约 −0.6～0。
已评最好贴零：`TR-PNPP-BC-PDE-F-s1234` ≈ 0.00、`TR-PN-BC-PDE-F-s2345` ≈ −0.06。
**禁止当正式赢家。**

| 组 | DATA | DATA+BC | BC+PDE-F | BC+PDE-EMA |
| --- | ---: | ---: | ---: | ---: |
| SP-PN s1234 | −7.97 | −1.89 | −0.36 | −0.10 |
| SP-PNPP s1234 | −7.47 | −1.54 | −0.34 | −0.45 |
| TR-PN s1234 | −4.66 | −0.81 | −0.24 | −0.37 |
| TR-PNPP s1234 | −3.88 | −0.81 | −0.00 | —（15） |
| SP-PN s2345 | —（16） | —（17） | −0.59 | −0.39 |
| SP-PNPP s2345 | −6.26 | −1.46 | −0.36 | −0.10 |
| TR-PN s2345 | −3.92 | −0.85 | −0.06 | −0.10 |
| TR-PNPP s2345 | −2.86 | −1.03 | −0.22 | −0.15 |
| SP-PN s3456 | −10.05 | −1.52 | −0.38 | −0.20 |
| SP-PNPP s3456 | −9.12 | −2.53 | −0.22 | —（39） |
| TR-PN s3456 | −2.82 | —（41） | —（42） | —（43） |
| TR-PNPP s3456 | —（44） | —（45） | —（46） | —（47） |

**遗留问题**：xlsx 未更新。缺 15 / 16–17 / 39 / 41–47。未做 full-wall fit。

**推进到实验步骤**：37/48 已有场指标 + WSS 下游；工作簿仍只有 0–14。

**当前状态判断**：GPU1 已空闲；index 15 仍在 GPU0。WSS 不能当可用重建。

**下一步**：用户点名后再填 xlsx；其余臂完训后再补评。

## 2026-08-27｜完训臂 official test35（22 个新评）· 已完成 · 未填工作簿

**本次主要修改**：用户要求用 node04 空闲卡，把已完训但未 test 的 V4 臂做
official test35，并更新文档、**先不改 xlsx**。当时 GPU0 仍在跑 index 15，
GPU1 空闲。按 0–14 同一协议评完 index **18–38、40** 共 22 臂
（`last_converged`，`eval_support`，稳态全体非壁面 / 瞬态 eligible×81）。
未跑 WSS，未按 test35 反选 checkpoint。

**对应代码/文档**：
`python -m wss_pinn.v4.evaluate --protocol official --checkpoint last_converged`；
日志 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/gpu1_completed_pending_20260827.log`；
记录 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/node04_eval_completed_pending_20260827.json`；
新 22 臂汇总 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/test35_eval_18_38_40_official_20260827.json`；
全部已评 37 臂 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/test35_eval_completed_official_20260827.json`。

**方法变更**：无。口径与 2026-08-23 的 0–14 相同。

**实验结果摘要**：primary `E_rel_l2`（病例等权，越低越好；零场预测 = 1）。
已评 37/48。方向与 0–14 一致：准稳态 DATA 仍 ≥1.10；PDE 臂略好但仍贴近零场。
目前已评最低约 **0.894**（`TR-PN-BC-PDE-F-s1234` 与 `TR-PNPP-BC-PDE-F-s2345`）。
瞬态压力 R² 仍约 −20，只作次要诊断。**禁止当正式赢家 / 3-seed Holm。**

| 组 | DATA | DATA+BC | BC+PDE-F | BC+PDE-EMA |
| --- | ---: | ---: | ---: | ---: |
| SP-PN s1234 | 1.125 | 1.176 | 0.968 | 0.935 |
| SP-PNPP s1234 | 1.105 | 1.153 | 0.964 | 0.910 |
| TR-PN s1234 | 1.041 | 0.937 | **0.894** | 0.965 |
| TR-PNPP s1234 | 1.037 | 0.941 | 0.907 | —（15 在训） |
| SP-PN s2345 | —（16） | —（17） | 0.997 | 0.929 |
| SP-PNPP s2345 | 1.128 | 1.116 | 0.979 | 0.919 |
| TR-PN s2345 | 1.017 | 0.953 | 0.919 | 0.938 |
| TR-PNPP s2345 | 1.018 | 0.950 | **0.894** | 0.951 |
| SP-PN s3456 | 1.139 | 1.138 | 0.984 | 0.926 |
| SP-PNPP s3456 | 1.129 | 1.211 | 0.950 | —（39 在训） |
| TR-PN s3456 | 1.012 | —（41） | —（42） | —（43） |
| TR-PNPP s3456 | —（44） | —（45） | —（46） | —（47） |

**遗留问题**：xlsx 未更新。缺 15 / 16–17 / 39 / 41–47。未跑 WSS。

**推进到实验步骤**：已评完训臂的场指标；工作簿仍只有 0–14。

**当前状态判断**：GPU1 评估已结束并空闲；index 15 仍在 GPU0。矩阵未齐，不能 Holm。

**下一步**：15 与其余臂完训后再评；用户点名后再填 xlsx / 跑 WSS。

## 2026-08-25｜优先直启 index 15（seed1234 末臂）· 进行中

**本次主要修改**：用户要求下一个实验先跑 **index 15**（`V4-TR-PNPP-BC-PDE-EMA-s1234`），
其余 pending 正常排队，以便先齐 seed1234 再汇总评估。未杀 master 上正在跑的
`12210_{31,37,38,39}`。node04 两张 A100 空闲且不在 GPU 分区，按 11/14 先例
SSH 直启 GPU0 全新开跑，并从数组 `scancel 12210_15`。确认正式 `run_dir` 原先不存在，
08-16 半成品仍只在 `*_cancelled_*` 目录。已写出 epoch 0–2。

**对应代码/文档**：
`wss_pinn/cluster/launch_node04_v4_index15.sh`；
记录 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/node04_direct_15_20260825.json`；
日志 `outputs/wss_pinn/slurm/v4_train_node04_15.out`。

**方法变更**：无训练/评估协议变更。15 为全新开跑，不是 resume。

**实验结果摘要**：启动后约 1 min 已完成 epoch 0–2，`mean_data_total` 约 1.17 → 0.89 → 1.00。
PID `2206976`，node04 GPU0。其余队列：`12210_[16-17,40-47]` 仍 PENDING
（`JobArrayTaskLimit`）。

**遗留问题**：15 完训前不能补工作簿第 15 行；GPU1 仍空闲，未另开别的臂。

**推进到实验步骤**：seed1234 第 16 臂已在训；0–14 评估/入账不变。

**当前状态判断**：15 已从数组抽出并在 04 上跑，不会和 master 四卡抢下一个空槽。
16–17、40–47 仍按 `%4` 排队。

**下一步**：等 15 完训后再做 official test35 + 工作簿第 15 行；不按中途 loss 改训练。

## 2026-08-23｜V4 seed1234 0–14 回填工作簿 + WSS 下游 + milestone 敏感性 · 待定（矩阵未齐）

**本次主要修改**：用户授权后，把 seed1234 index **0–14** 写入
`docs/03-汇报材料/WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx`
主表（行 19–33），**index 15 留空**。新增 `E_rel_l2_cb` 列。按 V2/V3 方法补
Profile-Secant V3、test35×1200 WSS，以及 `last_converged` vs `epoch_07500`
场敏感性。未按 test35 反选 checkpoint。备份
`…_备份_补V4_0-14前.xlsx`。

**对应代码/文档**：
`docs/03-汇报材料/tools/update_wss_pinn_v4_workbook.py`；
产物 `outputs/wss_pinn/audits/v4_workbook_0_14_20260823/`；
敏感性 sheet `V4 Checkpoint敏感性`。

**方法变更**：场指标=official 宇宙（瞬态 eligible×81）。WSS=峰值全场速度 +
冻结 Profile-Secant V3 ×1200。speed OLS 用已有解剖 ROI；WSS OLS 用同一 1200
点（V2/V3 后来的 full-wall fit 未重做）。`last` 与 `last_converged` 权值 SHA
相同，敏感性有效对照是 `epoch_07500`。

**关键指标**（test35 / seed1234 / last_converged；禁止与 V2 SAME5K、V3 val-selected 裸比）：

| 臂 | SP E_rel_l2 | SP WSS R²_cb | TR E_rel_l2 | TR WSS R²_cb |
| --- | ---: | ---: | ---: | ---: |
| DATA | 1.125 ± 0.172 | −7.97 ± 11.39 | 1.041 ± 0.143 | −4.66 ± 5.17 |
| DATA+BC | 1.176 ± 0.127 | −1.89 ± 2.65 | 0.937 ± 0.055 | −0.81 ± 1.05 |
| BC+PDE-F | 0.968 ± 0.060 | −0.36 ± 0.63 | 0.894 ± 0.052 | −0.24 ± 0.94 |
| BC+PDE-EMA | 0.935 ± 0.071 | −0.10 ± 0.26 | 0.965 ± 0.019 | −0.37 ± 0.22 |

PN++ 同向；TR-PNPP-F 的 WSS R²_cb ≈ 0.00 ± 0.37。`epoch_07500` 的 E 与终点同方向，
DATA 略好于终点、PDE 臂接近。

**Go-NoGo**：
- **待定** — 单 seed、缺 15、无 Holm，不能宣布正式赢家。判据：§12.0。
- **No-Go** — 用本表宣称可用场/WSS 重建。WSS R² 仍为负或贴零。
- 工作簿 V4 行是 **screen 草稿**，不是 Stage 4 正式入账。

**推进到实验步骤**：xlsx 已有 0–14 可对读的 V4 行；15 与其余 seeds 仍待完训。

**当前状态判断**：主表 + V4 敏感性已按现有完训臂补齐；线性回归两张附表仍只有 V2/V3。

**下一步**：等 index 15 完训后补第 15 行；48-run 齐后再做 3-seed 正式汇总。

## 2026-08-23｜V4 seed1234 index 0–14 的 test35 official 评估 · 待定（矩阵未齐）

**本次主要修改**：新增 `wss_pinn/v4/evaluate.py`。用户授权后在 node04 两张空闲
A100 上评估完训臂 **0–14**（seed1234；缺 index 15
`V4-TR-PNPP-BC-PDE-EMA-s1234`）。未跑 WSS，未按 test35 反选 checkpoint，未填
xlsx。checkpoint 固定 `last_converged.pt`。

**对应代码/文档**：`wss_pinn/v4/evaluate.py`；汇总
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/test35_eval_0_14_official_20260823.json`；
逐臂 `evaluation_official_last_converged_full.json`；日志
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/`。

**方法变更**：协议 `official` = `eval_support`（配置点数；稳态 5k / 瞬态 2k）→
稳态全体非壁面体点；瞬态全部 eligible 缓存点 × 81 帧。这是 V4 评估宇宙，不是
same5k 抽点。Primary 为病例等权速度向量相对 L2
`E_case=‖u_pred−u_true‖₂/‖u_true‖₂`。零场预测的 `E=1`。

**关键指标**（test35 / seed1234 / official / `last_converged`；禁止与 V2/V3 裸比）：

| 臂 | SP-PN | SP-PNPP | TR-PN | TR-PNPP |
| --- | ---: | ---: | ---: | ---: |
| DATA | 1.125 | 1.105 | 1.041 | 1.037 |
| DATA+BC | 1.176 | 1.153 | 0.937 | 0.941 |
| BC+PDE-F | 0.968 | 0.964 | 0.894 | 0.907 |
| BC+PDE-EMA | 0.935 | 0.910 | 0.965 | （缺 15） |

辅指标：准稳态 speed R² 仍为负；瞬态 BC+PDE-F 的 speed R² 约 0.12–0.17。
瞬态压力 R² 约 −20，不能当成功。NMAE 分母为 `RMS(y)`。

**Go-NoGo**：
- **待定** — 正式 16 臂 × 3-seed Holm 未齐；单 seed、缺 15，不能宣布架构/物理赢家。
  判据：设计 §12.0。
- **探索性方向** — 与 train-only「加物理抬高 data loss」不同：test35 上 PDE 臂
  `E` 低于 DATA（DATA 甚至差于零场）。这不是选模依据。
- **No-Go** — 用本切片宣称可用场重建。最低 `E≈0.89` 仍接近零场。
- 不回填工作簿正式 V4 行。

**推进到实验步骤**：Stage 4 的 seed1234 前 15 臂 screen 已落地；矩阵其余 run
与 index 15 仍待完训后再评。

**当前状态判断**：0–14 有可汇总的 test35 primary；结论仍是探索性 screen。

**下一步**：等 index 15 与其余 seeds 完训后再做正式汇总；不跑 WSS；不按本表砍臂。

## 2026-08-21｜归档 field-v4 Stage 0–1 执行提示词 · 🧊冻结

**本次主要修改**：将已完成的
`WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1.md` 移入
`WSS_PINN/_archive/`，按 V3 六臂先例改名为
`…_已完成_2026-08-06.md`。该提示词文首 2026-08-06 已写「已完成」；现行执行入口是
`volume_uvwp_bc_rcr_v4`，不能再当待办。未改训练代码、未动 `outputs/`。

**对应代码/文档**：
[归档后的 Stage 0–1 提示词](WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1_已完成_2026-08-06.md)、
[归档索引](WSS_PINN/_archive/README.md)、[路线 README](WSS_PINN/README.md)、
[V4 设计方案](WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md)。
结论仍留在路线 README 的 field-v4 结果节；证据
`outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_multiseed_and_promotion_report.json`。

**Go-NoGo**：归档不改变历史结论 — **保留 G-Raw**；G-PE / local decoder **No-Go**。
判据来源：field-v4 Stage 1 多种子报告。现行主线仍是新 V4，不据此重开 Stage 2。

**推进到实验步骤**：文档入口收口；训练矩阵状态不变。

**当前状态判断**：活动目录不再把 Stage 0–1 提示词当作当前执行合同。

**下一步**：继续 `volume_uvwp_bc_rcr_v4` 48-run；不恢复 field-v4 Stage 2。

## 2026-08-21｜V4 中期整理：22/48 完训的 train-only 分析 · 待定（primary）/ No-Go（train 侧加物理改善拟合）

**本次主要修改**：未改训练代码、未提交作业、未读 test35、未跑 WSS、未回填
`WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx`（该表仍只有
V2/V3，符合设计「无 test35 不预填 V4 行」）。对已完训 22 个 run 的
`training_summary.json` + 末 epoch `epoch_progress.jsonl` 做中期整理。

**对应代码/文档**：
- 证据：`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/midterm_train_only_20260821.json`
- 路线状态：`docs/02-推进与变更/WSS_PINN/README.md`
- 设计方案文首进度：`WSS_PINN_V4大重构设计方案_2026-08-08.md`

**背景假设**：四臂分离 DATA / DATA+BC / BC+PDE-fixed / BC+PDE-EMA 的贡献；
停止只看 train-only raw 分量；primary endpoint 仍是 test35 速度向量相对 L2。

**关键指标**（train138 末 epoch raw `mean_data_total`；split train138/test35；
对照为同 backbone×seed 的 DATA 臂；**禁止与 V2/V3 test35 R² 裸比**）：

| 臂 | SP-PN s1234 | SP-PNPP s1234 | 停止 |
| --- | ---: | ---: | --- |
| DATA | 0.1912 | 0.1932 | plateau 9859 |
| DATA+BC | 0.4902 | 0.4997 | max_epochs |
| BC+PDE-F | 0.7260 | 0.7301 | max_epochs |
| BC+PDE-EMA | 0.8329 | 0.8262 | max_epochs / plateau 9985 |

SP s1234 四臂初始化配对成立（PN `e23bcf10481f…`，PNPP `967333b84902…`）。
EMA 四个完训臂 `λ_pde` 有 99.8% epoch 贴上限 10。瞬态 DATA 出现明显 seed 差：
s1234 0.178 vs s2345 0.255。全部完训 run `test35_read_during_training=false`。

队列（2026-08-21 11:34）：完训 22；master `12210_{26,27,29,30}` running；
node04 直启 11/14 running；pending `12210_[15-17,31-47]`。

**Go-NoGo**：
- **待定** — primary endpoint（test35 相对 L2）未评估。判据：设计方案 §12.0。
- **No-Go** — train 侧「加 BC/PDE 改善场拟合」。SP s1234 完整八臂上三组预注册
  contrast 的 `data_total` 全部为正（变差）；p loss 在 PDE 两臂从 ~0.016 升到
  0.50/0.80。判据：设计 §9/§12 四臂分离 + 禁止只看 weighted total。
- **Inconclusive** — PointNet++ vs PointNet（同臂差 ~0.002）。
- 不据此砍臂或改 λ；EMA 撞限可另立案重预注册，须不读 test35。

**推进到实验步骤**：Stage 2/3 训练矩阵进行中；Stage 4 汇总与 test35 仍未启动。

**当前状态判断**：中期证据足够说明「物理臂在训练损失上系统性差于 DATA」，
但 3-seed 矩阵和 test35 主指标都未齐，不能写进工作簿或对外宣称架构/物理优劣。

**下一步**：保持 12210 与 node04 11/14 跑完；seed3456 仍走原数组；矩阵冻结后再
统一 test35。

## 2026-08-20｜node04 两张 A100 空闲，直启 `11/14`；不能同时跑 4 路

**本次主要修改**：用户要求检查 node04 GPU。UID `1006/1007`、public 树可读、两张
A100-40GB 均空闲（仅 Xorg）。**不能同时跑 4 个**：node04 只有 2 张 GPU；它在
Slurm 里属于 CPU 分区且 `DOWN+NOT_RESPONDING`、`Gres=(null)`，GPU 分区只有
master，因此无法 `sbatch` 到 04。按 V3 先例 SSH 直启两臂，一卡一个。先
`scontrol hold 12210_{11,14}`，确认进程与 epoch 写出后再 `scancel`，避免
master 腾卡后重复开跑。未 resume。

| 节点 | 作业 | 状态 |
| --- | --- | --- |
| node04 GPU0 PID `2098215` | index 11 `V4-TR-PN-BC-PDE-EMA-s1234` | 直启全新开跑，已写 epoch 1 |
| node04 GPU1 PID `2098627` | index 14 `V4-TR-PNPP-BC-PDE-F-s1234` | 直启全新开跑，已写 epoch 1 |
| master | `12210_{22,23,24}` | RUNNING |
| master 队列 | `12210_[15-17,25-47]%4` | PENDING（`Resources`） |

日志：`outputs/wss_pinn/slurm/v4_train_node04_{11,14}.out`。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/node04_direct_11_14_20260820.json`。
未评估 test35、未跑 WSS。

**当前状态判断**：集群现为 master 3 路 + node04 2 路并行。04 上没有第三、第四张
卡可再提交。

## 2026-08-20｜V4 恢复四卡并发：`12210` `ArrayTaskThrottle=2→4`，`12210_24` 已启动

**本次主要修改**：用户授权等待中的 V4 数组改回 4 GPU 并行。未新提交数组（避免重复占
index），只对已有作业执行 `scontrol update JobId=12210 ArrayTaskThrottle=4`。
未改训练代码、未 scancel 师姐 `12366`、未 resume 任何半成品。11/14/15/16/17 的
`last.pt` 目录此前已隔离，启动前核对该 5 个 run 目录均不存在。

**作业状态（操作后）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED |
| `7-10,12,13,18-21` | `12210_{7-10,12,13,18-21}` | COMPLETED |
| `22` | `12210_22` / `V4-SP-PNPP-BC-PDE-F-s2345` | RUNNING（GPU1，约 epoch 5782） |
| `23` | `12210_23` / `V4-SP-PNPP-BC-PDE-EMA-s2345` | RUNNING（GPU0，约 epoch 4249） |
| `24` | `12210_24` / `V4-TR-PN-DATA-s2345` | RUNNING（GPU3，全新开跑，epoch 0 已写出） |
| `11,14-17,25-47` | `12210_[11,14-17,25-47]%4` | PENDING（`Resources`；师姐 `12366` 仍占 GPU2） |
| 师姐 | `sunfanji` `12366` PINN_Train | RUNNING（GPU2） |

新增 completed：`18` `train_only_plateau` 9859；`19` `train_only_plateau` 9865；
`20` `train_only_plateau` 9859；`21` `max_epochs` 10000。11/14/15/16/17 再启动
仍为**全新开跑**。未评估 test35、未跑 WSS。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/throttle_restore_4gpu_20260820.json`。

**推进到实验步骤**：数组并发恢复为最多四卡；物理上当前三卡属本数组、一卡属师姐。
师姐结束或 22/23 完训后，第 4 路会立刻拉起（优先空闲 GPU，不重开作业）。

**当前状态判断**：`JobArrayTaskLimit` 已解除。剩余 pending 卡在 GPU 资源，不是
throttle。PointNet baseline 矩阵仍冻结，本条不写入该档案。

## 2026-08-17｜隔离 `12210_{11,16,17}` NODE_FAIL 半成品并放回 `%2` 队列，全新开跑

**本次主要修改**：用户确认后隔离今早 master 失联留下的 `last.pt` 目录，不 resume。
先 `scontrol hold 12210_{11,16,17}`，避免 18/19 腾卡时撞上半成品；再挪走 run 目录与
Slurm 日志。`scontrol requeuehold` 对已 pending 的 task 返回
`Job is pending execution`（它们早上已被 Slurm 自动 requeue），随后 `release`。
未改训练代码、未动 `12210_{18,19}`、未改 `ArrayTaskThrottle=2`。

**隔离位置**：

- `.../transient_autograd/V4-TR-PN-BC-PDE-EMA-s1234_cancelled_12210_20260817_epoch9401/`（epoch 9401 / step 1297370）
- `.../steady_peak/V4-SP-PN-DATA-s2345_cancelled_12210_20260817_epoch1912/`（epoch 1912 / step 263930）
- `.../steady_peak/V4-SP-PN-BC-s2345_cancelled_12210_20260817_epoch1889/`（epoch 1889 / step 260730）
- Slurm 日志：`outputs/wss_pinn/slurm/cancelled/v4_train_12210_{11,16,17}.{out,err}.cancelled_12210_20260817`

**作业状态（操作后）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED |
| `7-10,12,13` | `12210_{7-10,12,13}` | COMPLETED |
| `18,19` | `12210_{18,19}` | RUNNING |
| `11,14-17,20-47` | `12210_[11,14-17,20-47]%2` | PENDING（`JobArrayTaskLimit`） |

11/14/15/16/17 再启动均为**全新开跑**。18 或 19 结束后会先拉 index 11。未评估
test35、未跑 WSS。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/quarantine_11_16_17_20260817.json`。

## 2026-08-17｜核对 `12210_{11,16,17}`：master 失联导致 NODE_FAIL/requeue，非人工杀死

**本次主要修改**：只读核查，未改训练代码、未 scancel。用户看到 11/16 被杀死重入队、
17 只有半成品，根因是今早 **master 节点对 slurmctld 失联**，不是训练脚本自己退出。

**时间线（`/var/log/slurmctld.log` + 作业 `.err`）**：

- 04:25 / 04:51 master 多次 `not responding`，并伴随 slurmdbd `Unable to connect to database`
- 04:51:47 `requeue job JobId=12210_11(12238) due to failure of node master`
- 04:54:17 同样 requeue `12210_16(12307)`；师姐 `12240/12261` 同期被 requeue
- 05:03:19 master 被设 DOWN；作业 `.err` 写 `CANCELLED ... DUE TO NODE FAILURE`，
  随后 `STEPD TERMINATED ... JOB NOT ENDING WITH SIGNALS`（进程没被信号干净杀掉）
- 05:03:50 backfill 启动 `12210_17(12308)`
- 05:04:50 slurmd `Kill task failed`，master 被标 `DRAINING` / reason `Kill task failed`
- 05:12:20 `requeue job JobId=12210_17(12308) due to failure of node master`，master 再 DOWN
- 08:01 slurmctld 恢复并 Recovered 12210 各 task；09:13 master IDLE 后启动了
  **从未跑过的** `12210_{18,19}`，11/16/17 因 throttle=2 继续 pending

**17 为什么是半成品**：17 在 05:03 才开始，05:12 已被 Slurm requeue，但 `.err` 为空、
日志一直写到约 07:13（epoch 1888），说明 slurmd 失联后 Python 可能作为孤儿进程继续跑，
直到节点恢复前后才停。没有 `training_summary.json`，只有 `last.pt`。

**当前状态判断**：18/19 仍在正常训练。11/16/17（以及此前已隔离过的 14/15）再启动前必须
先挪走 `last.pt` 目录，否则会被 `FileExistsError` 拒绝。未评估 test35、未跑 WSS。

## 2026-08-16｜V4 让出两卡给师姐：杀死 `12210_{14,15}`，剩余改为 `%2` 排队

**本次主要修改**：用户要求空出两张 4090 给师姐，并让尚未开始的任务只使用
`12210_11`（物理 GPU0）与 `12210_13`（物理 GPU3）所在的两卡。当时四路运行是
`11/13/14/15`。先把数组 `ArrayTaskThrottle` 从 4 改为 2 并 hold 未启动的
`16-47`，再 `scancel 12210_14 12210_15`，避免空卡被自己的后续 task 立刻占回。

**14/15 半成品与日志**：训练入口遇到 `last.pt` 会拒绝覆盖。已隔离

- `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/transient_autograd/V4-TR-PNPP-BC-PDE-F-s1234_cancelled_12210_20260816_epoch3622/`（约 epoch 3622 / step 499870）
- `.../V4-TR-PNPP-BC-PDE-EMA-s1234_cancelled_12210_20260816_epoch545/`（约 epoch 545 / step 75330）
- Slurm 日志：`outputs/wss_pinn/slurm/cancelled/v4_train_12210_{14,15}.{out,err}.cancelled_12210_20260816`

随后 `scontrol requeuehold 12210_{14,15}` 再 release。14/15 **全新开跑**，非 resume。

**作业状态（操作后）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED |
| `7-10,12` | `12210_{7-10,12}` | COMPLETED |
| `11` | `12210_11` / `V4-TR-PN-BC-PDE-EMA-s1234` | RUNNING（GPU0） |
| `13` | `12210_13` / `V4-TR-PNPP-BC-s1234` | RUNNING（GPU3） |
| `14-47` | `12210_[14-47]%2` | PENDING（`JobArrayTaskLimit`，最多两卡） |
| 师姐 | `sunfanji` `12240/12261` | 释放 GPU1/2 后已启动 |

**推进到实验步骤**：矩阵未完训部分改为两卡串/并行排队；不评估 test35、不跑 WSS。

**当前状态判断**：让卡成功，我方只占 GPU0/3。后续接手先查 `squeue -j 12210` 是否仍为
`11/13` running + `14-47` pending `%2`。记录：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/requeue_14_47_two_gpu.json`。

## 2026-08-14｜V4 补提交未完训 `7-47`：作业 `12210_[7-47%4]`

**本次主要修改**：master 四张 4090 空闲后，按 2026-08-09 约定补跑未完训数组。
不重跑 Stage 0 `11970` / preflight `11971`，也不重跑已 completed 的 `11972_{0-6}`。
index 7 的中断目录含 `last.pt`，训练入口会拒绝覆盖，因此将其挪到
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak/V4-SP-PNPP-BC-PDE-EMA-s1234_cancelled_11972_7_epoch797/`
后对 index 7 **全新开跑**（非 resume）。随后
`sbatch --array=7-47%4` 提交训练脚本 `wss_pinn/cluster/run_bc_rcr_v4.slurm`。

**作业状态（提交当时）**：

| 范围 | 作业 | 状态 |
| --- | --- | --- |
| `0-6` | `11972_{0-6}` | COMPLETED（`steady_peak × seed1234` 除 PNPP BC+PDE-EMA 外的七臂） |
| `7` | `12210_7` | RUNNING（`V4-SP-PNPP-BC-PDE-EMA-s1234` 全新重跑） |
| `8-10` | `12210_{8,9,10}` | RUNNING（瞬态 seed1234 前三臂） |
| `11-47` | `12210_[11-47%4]` | PENDING（`JobArrayTaskLimit`，throttle=4） |

**对应产物 / 记录**：
`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/resubmission_7_47.json`；完训七臂仍在
`.../steady_peak/`。本轮不评估 test35、不跑 WSS、不回填工作簿。

**推进到实验步骤**：`steady_peak × seed1234` 七臂已完训；剩余 41 run
（index 7 + 瞬态 seed1234 + seed2345/3456）已重新入队。

**当前状态判断**：补提交成功，四卡已占用。后续接手先查 `squeue -j 12210`；
48-run 全部 completed 前不得写“矩阵完成”，也不得自动开 test35。

## 2026-08-09｜V4 二次截断：取消 `11972_7`，下次补跑从 index `7` 起

**本次主要修改**：在已取消 `11972_[8-47]` 的基础上，再 `scancel 11972_7`
（`V4-SP-PNPP-BC-PDE-EMA-s1234`）。不中断仍在跑的 `11972_{2,3,6}`。用户明确下次
补跑从 index `7` 开始（`7-47`），本轮不自动重提、不评估 test35、不跑 WSS。

**作业状态（sacct / squeue，二次截断后）**：

| array | run | 状态 |
| --- | --- | --- |
| 0 | `V4-SP-PN-DATA-s1234` | COMPLETED（`train_only_plateau`，9859 epoch） |
| 1 | `V4-SP-PN-BC-s1234` | COMPLETED（`max_epochs`，10000） |
| 2 | `V4-SP-PN-BC-PDE-F-s1234` | RUNNING（master） |
| 3 | `V4-SP-PN-BC-PDE-EMA-s1234` | RUNNING（master） |
| 4 | `V4-SP-PNPP-DATA-s1234` | COMPLETED（`train_only_plateau`，9859 epoch） |
| 5 | `V4-SP-PNPP-BC-s1234` | COMPLETED（`max_epochs`，10000） |
| 6 | `V4-SP-PNPP-BC-PDE-F-s1234` | RUNNING（master） |
| 7 | `V4-SP-PNPP-BC-PDE-EMA-s1234` | CANCELLED（约 epoch 797 / step 110120；部分产物保留） |
| 8–47 | 瞬态 seed1234 + seed2345/3456 全部臂 | CANCELLED（与 7 一并待补提交） |

**对应产物**：完训四臂在 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/steady_peak/`；
index 7 目录含未完训 `history.csv` / `checkpoints/`，补跑时应按全新提交或显式
resume 处理，勿与 completed 混写。

**推进到实验步骤**：`steady_peak × seed1234` 中 DATA/DATA+BC 已完训；PN 两臂 PDE
与 PNPP BC+PDE-F 仍在训；PNPP BC+PDE-EMA 与瞬态/多种子未完成。

**当前状态判断**：当前占 3 卡（`2/3/6`）。下次接手先查 `squeue` 是否仍有
`11972_{2,3,6}`，再按用户指令补提 **`7-47`**。入口已同步：
[WSS_PINN README](WSS_PINN/README.md)、[V4 设计方案](WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md)。

## 2026-08-09｜V4 正式数组截断：取消挂起 `11972_[8-47]`，仅保留 `0-7`

**本次主要修改**：用户确认四卡已被 `11972` 占满后，取消尚未开始的数组任务
`11972_[8-47]`（`scancel`），释放后续排队；当时仍保留 `0-7`。同日稍后用户进一步
取消 `11972_7`，见上一条；补跑起点改为 `7-47`。

## 2026-08-08｜WSS_PINN V4 v1.2 实现完成并提交 4 卡 Slurm 训练链

**代码与配置**：新增独立 `wss_pinn/v4/` 包，覆盖严格配置合同、UDF Fourier/RCR/物性
解析、train138/test35 + CV5 split、173 例 Stage 0 builder、跨 81 帧点身份与时间映射、
`stl_landmarks_v4` 坐标/速度注册、入口/出口面法向与面积权重、train138-only BC/几何/
瞬态场统计、`RCR~A_out` 审计、PointNet/PointNet++ + 共享 BCEncoder、准稳态/瞬态
strong-form residual、质量流量口径 RCR BC、分组无量纲化、fixed/detached-EMA `λ_pde`、
train-only 平台停止和 `last_converged` checkpoint。生成 16 臂 × 3 seeds = 48 个正式配置；
两个 backbone 参数量分别约 249.9k/250.1k，比值 `1.0008`。

**关键纠错**：真实病例 smoke 首次发现 `processed/coord_normalized` 仍是旧病例内坐标系，
不能与当前冻结 sidecar 混用；实现改读 `processed/features` 原始毫米坐标，并用 bundle 的
病例特异 `unit_factor`（不是固定 1000）、`transform_centroid/rotation/coord_scale` 转到
`stl_landmarks_v4`。DING_JUN_FENG 修正后峰值坐标/速度/压力最大偏差为
`1.56e-7 / 2.38e-7 m/s / 0.0011 Pa`。ZHOU_KE_XUN 的 `A_udf≠A_mesh` 合同复现
`Q_nom_peak=1.07274e-4`、`Q_actual_peak=1.45501e-4 m³/s`。173 例轻量审计另确认旧
`BC_Inlet` monitor 相对 UDF 解析真源最多可差 `2.01%`；因此 monitor 降为诊断，训练入口
流量只使用 `Q_nom·A_mesh/A_udf`，Stage 0 仅保留 3% gross-mismatch 护栏。test-role
KANG_YONG 构建通过但不进入训练统计。

**验证**：34/34 `test_volume_*` 通过，覆盖 UDF 重置跳变、时间特征链式法则、完整
Carreau–Yasuda 应力散度、RCR 压力形式、detached ratio 不抵消 physics 梯度、两 backbone
容量匹配、旧路线回归和 resume schema 自动重建。额外完整瞬态 BC+PDE+EMA 单步反传
finite；173 例 Stage 0 本地全量 Gate（train138/test35、每例 81 帧）通过。

**Slurm 提交**：最终依赖链为 CPU Stage 0 build/Gate `11970` → 单卡 GPU preflight
`11971`（`afterok:11970`）→ 48-run 数组 `11972_[0-47%4]`
（`afterok:11971`，`ArrayTaskThrottle=4`，每 task 1 张 GPU）；`11970/11971` 已 exit 0，
正式数组已经启动。首个 Stage 0 `11964` 因把旧
`BC_Inlet` monitor 误设为硬真值而失败；修正为 UDF 解析真源 + 3% 诊断护栏。第二个
Stage 0 `11967` 因部分 bundle 裁剪人工延长段、瞬态 feature 点未先过滤到冻结目标域而失败；
修正后全库最少保留 1569 个目标域 interior 点、最低保留率 78.45%。旧依赖作业
`11965/11966`、`11968/11969` 均已取消，未启动训练。本地全量构建时另修复 resume 对旧
schema manifest 的识别，避免缺少 `eligible_interior` 时错误跳过。任一前级失败时训练不会
启动；训练完成后不自动运行 test35、WSS、工作簿回填或汇总。

**当前状态判断**：实现与提交已完成，正式数组已启动但尚无汇总结果。按用户要求本轮不等待作业结束；
待全部实验完成后再统一评估、汇总和分析。

## 2026-08-08｜WSS_PINN V4 设计 v1.2：对抗性审核修正（入口合同/RCR 单位/四臂）

**本次主要修改**：对 [V4 大重构设计方案](WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md)
完成 v1.0→v1.1→v1.2 两轮修订，仅改设计与导航文档，未动代码/数据。v1.1 曾以“入口波形
全库统一”为由移除 `Q_in` 输入并把入口/出口 BC 以残差放入物理 loss；对抗性审核证实四个
关键问题，v1.2 修正为：

1. **入口合同**：边界审计确认 6 例 `A_udf≠A_mesh`（最大约 35.6%，如 ZHOU_KE_XUN 实际
   峰值 1.45494e-4 对名义 1.0727e-4 m³/s），“入口流量零方差”不成立；冻结
   `U_in(t)=Q_nom(t)/A_udf`、`Q_actual(t)=U_in(t)·A_mesh`，恢复 `Q_actual_peak` 输入，
   `A_udf` 落库 provenance，禁止用 `Q_nom/A_mesh` 监督；入口 BC 用带 buffer 的
   face-interior 点（入口—壁面共享 32–83 顶点/例，共享带归 no-slip）。
2. **RCR 单位**：UDF 递推按 `F_FLUX`（质量流量 kg/s）标定，残差改用
   `ṁ_out=ρ∫u·n dA`，混用体积流量差约 ρ≈1060 倍；冻结法向/符号并加单出口解析单元测试。
3. **波形与 gauge**：`w=6.491≠2π/0.8`，mod-T 重置点值/导数跳变，按 UDF 分段实现、
   时间 collocation 避开 `t≈0/T`、首尾连续性降为诊断；压力锚定收紧为四条件成立的
   raw Fluent gauge 约定，`P_out` 定义三选一冻结（施加 profile 与求解 face-average
   已有约百 Pa 差异证据）。
4. **归因与统计**：训练模式改 DATA/DATA+BC/BC+PDE-fixed/BC+PDE-EMA 四臂共 16 臂，
   分离边界监督与 PDE 贡献并正式记录“BC 进入 loss”的范围变更；`λ_pde=1` 降级为
   可复现基线表述，RCR 尺度改逐出口 `(R1+R2)·ṁ_c`；“标签残差地板”拆为参考尺度
   （时间 FD/RCR 时序）与敏感性参考带（散点空间残差，k=48/96 差 1–2.2×）；primary
   endpoint 改速度向量相对 L2，预注册 4 组 contrasts + Holm，test35 结论定性为探索性；
   新增 train138 内部 CV 边界（test35 永不进 fold）。

**对应代码/文档**：设计方案 v1.2、[路线真源](WSS_PINN/README.md)（状态区已同步）、
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：Stage 0 合同新增 `A_mesh/A_udf` 双落库与 6 例处置预注册、RCR 容错
解析（`udf-inlet.c`/`udf-inlet4.c` 命名差异与无空格写法）、ρ/Carreau 冻结、`RCR~A_out`
共线性审计、81 帧离散 RCR 残差（质量流量口径）参考尺度审计。正式实现仍未开始。

## 2026-08-08｜WSS_PINN V4 大重构设计：显式 RCR 条件与稳态/瞬态双路线

**本次主要修改**：新增
[WSS_PINN V4 大重构设计方案](WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md)，
将新 V4 与已完成旧 field-v4 隔离。新方案不再输入 CFD 求解后才能得到的 outlet pressure，
改用 `Q_in`、入口/四出口面积和四组 `R1/R2/C`；统一比较 PointNet/PointNet++ 的
DATA、PINN-fixed 与 PINN-EMA-ratio，共 12 臂；动态臂采用 detached EMA
`L_data/L_phy` 每 50 step 更新，固定 `lambda_phy=1` 作为消融。保留 continuity、三分量 momentum 与 no-slip，准稳态 peak 和显式
`du/dt` 瞬态路线由配置切换。入口/RCR 条件只作输入，不进入入口/出口优化 loss。

**对应代码/文档**：[路线真源](WSS_PINN/README.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、`wss_pinn/AGENTS.md` 和
[项目级推进摘要](代码修改与实验推进记录.md)。本次仅修改设计与导航文档。

**推进到实验步骤**：完成 173/173 文件级覆盖核对：每例 81 帧 `u/v/w/p`、81 帧精确
匹配 `Q_in(t)`、solver `dt=0.005 s`、场导出 `Δt=0.01 s`、四组 RCR 与正值入口/四出口
面积均存在；val15 计划并回 train，形成 train138/test35。

**当前状态判断**：瞬态路线在源数据层面可行，但尚未通过跨时相 cell/coordinate 身份、
时间索引、pressure gauge 和 train-only stats 深审计。新 route/config/data/loss/trainer
均未实现，未提交训练，工作簿仍只保留冻结 V2/V3 结果。

---

## 2026-08-07｜PointNet 全历史实验补充 `y=ax+b` 线性拟合 R²

**本次主要修改**：在 WSS-min 评估链新增 `prediction=a×truth+b` 的一元线性拟合指标，物理/归一化空间均输出 pooled 与病例等权共享拟合线的 `R²_fit`、斜率 `a` 和截距 `b`；新增独立 `linear_fit_metrics.csv`。保留原始 R²、病例等权 R²、训练损失、checkpoint 与既有 Gate，不用拟合 R²替代绝对精度。同步修复面积映射诊断入口未调用 `model.eval()` 的问题；历史上由该旧入口生成的一行结果仍按原训练态预测精确复现并在回填器中显式隔离。

**对应代码/文档**：`training_wss_min/{metrics.py,evaluate.py}`、`training_wss_min/tools/{backfill_linear_fit_r2_workbook.py,evaluate_mapping_blocked_run.py}`、`training_wss_min/tests/test_linear_fit_metrics.py`、`training_wss_min/README.md`、[跨路线评估口径](../00-规范与记录/WSS跨路线评估与横向对比口径.md)和 `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。批量审计产物位于 `outputs/wss_min/linear_fit_r2_backfill_20260807/`。

**推进到实验步骤**：从总览 180 个单实验行追溯并去重为 175 个 checkpoint，按原 split/checkpoint 重新推理；旧 test16 使用终签 legacy 快照恢复已移出活动目录的 `WANG_DENG_FENG`。175/175 个来源均通过历史 raw R²复现，最大绝对差 `1.42e-8`；另按原表规则计算 4 个三种子均值行，共回填 184 个实验行。工作簿总览新增物理/归一化 12 列，汇总对比、教师汇报视图、RCR Oracle 和指标说明已同步并经 LibreOffice 全量重算。

**当前状态判断**：全历史回填完成，无缺失或估算值。`R²_fit` 仅表示允许统一线性缩放/偏移后的趋势一致性；汇报时必须同时查看 `a、b` 与 raw/case-balanced R²、MAE/RMSE。后续新评估会自动产出该指标，但现有实验排序和晋级结论不因本次补充自动改写。

## 2026-08-06｜V2 SAME5K-E7500 补训练/物理 loss 收敛图

**本次主要修改**：新增 `wss_pinn/tools/plot_v2_loss_convergence.py`，从八臂冻结
`epoch_progress.jsonl` 生成老师汇报版 train data / physics loss 图（V2 无 val，
仅 train；50-epoch 平滑；最后 10% 阴影）。

**对应代码/文档**：`wss_pinn/tools/plot_v2_loss_convergence.py`；产物
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/summary/loss_convergence/`
（`teacher_loss_convergence.pdf`、总览、配对 `convergence_curves`、E1–E8 分臂、
物理分量、summary csv/json）。

**推进到实验步骤**：历史 V2 完训后补可视化；未改训练/评估口径，未读 test35。

**当前状态判断**：八臂 last50 vs prior50 data 变化约 `-0.07%`–`+0.06%`，与 README
平台结论一致；PINN 臂最后 10% physics 变化约 `-0.22%`–`+0.10%`。可直接把 PDF/总览
发给老师。

## 2026-08-06｜WSS_PINN field-v4 Stage 0–1 全链完成并冻结 G-Raw

**本次主要修改**：实现独立 field-v4 数据合同与写路径护栏、病例等权
`S_field^cb`、固定 val15×5000 query、B0 病例盲 atlas、global/local × raw/PE
连续查询模型、outlet 双路径与 inlet-wall/wall-zone Gate、制造解/导数/support 探针、
分阶段 Slurm 提交、单种子排名 Gate、补种子矩阵和多种子/full-volume 汇总。

**对应代码/文档**：`wss_pinn/{config.py,train.py,evaluate.py,validation.py}`、
`wss_pinn/data/dataset.py`、`wss_pinn/models/point_models.py`、
`wss_pinn/tools/{build_field_v4,b0_atlas_v4,diagnose_field_v4,audit_boundary_field_v4,gate_stage0a_field_v4,gate_stage1_raw_field_v4,gate_stage1_single_seed_field_v4,finalize_stage1_field_v4,smoke_field_v4}.py`、
`wss_pinn/cluster/{preflight,run_experiment,evaluate_field_v4_fullvolume}.slurm`、
`wss_pinn/configs/volume_uvwp_peak_field_v4*/`、[路线真源](WSS_PINN/README.md)、
[`wss_pinn` README](../../wss_pinn/README.md)和
[已完成执行合同](WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1_已完成_2026-08-06.md)。

**推进到实验步骤**：Stage 0-a/0-b Gate pass；B0 val15 完成；Raw `11315_[0-1]`、
PE `11318_[0-1]`、确认种子 `11321_[0-3]` 与 full-volume `11325_[0-5]` 全部完成、
exit code 0。补种子前 GPU preflight `11320` 通过；最终 26 tests、CPU smoke、原矩阵和
补种子矩阵静态 preflight 均 pass。训练/评估只使用 train123/val15，test35 读取数为 0。

**当前状态判断**：单 seed 排名为 G-PE `0.567612`、G-Raw `0.578311`、L-Raw
`0.588856`、L-PE `0.597688`。确认后 G-Raw/G-PE 为
`0.576116±0.007477 / 0.581208±0.012599`；PE 只在 seed1234 改善，病例 bootstrap
95% CI 跨 0，并触发 pressure 与最差 ILO 队列 2% 护栏。保留 G-Raw，G-PE No-Go，
停止 PE 扫频和 local decoder；Stage 2 未启动。本轮未运行 WSS、BC loss/input 或 PDE loss。

## 2026-08-06｜WSS_PINN 当前提示词换代与六臂提示词归档

**本次主要修改**：归档已完成的准稳态平滑场六臂预注册提示词，并以冻结诊断为真源
新建 Stage 0-a/0-b → Stage 1 执行提示词。新提示词不继承旧六臂、quasi-steady
momentum、2500 epoch、`lambda=1` 或自动正式提交合同，改为病例等权选模、B0 atlas、
边界 Gate 和纯监督表示矩阵。

**对应代码/文档**：
[当时执行提示词](WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1_已完成_2026-08-06.md)、
[历史六臂提示词](WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[归档索引](WSS_PINN/_archive/README.md)、[路线真源](WSS_PINN/README.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、`wss_pinn/AGENTS.md`、`docs/README.md`和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：完成执行提示词换代与导航同步；未改代码、配置、数据、split、
checkpoint 或训练产物，未启动训练。

**当前状态判断**：当前智能体应从新提示词进入 Stage 0-a/0-b；旧提示词仅供 V3 六臂
历史复盘，不再具有执行授权或活动合同效力。

## 2026-08-06｜体域 `u,v,w,p` 设计最终冻结与执行入口收口

**本次主要修改**：完成冻结前最后一轮科学审查。将新增 atlas、平滑 oracle、support
稳定性、梯度余弦和 outlet/RCR 链测量保持为 B 级证据；新增病例等权 `S_field^cb` 作为
唯一 checkpoint/early-stop 标量，规定 B0 atlas 永久进入主表，修正 Stage 0 依赖、
Stage 1 的 BC/PDE 隔离、Stage 3 起点和瞬态 physics 双路线。WSS 仍严格冻结为最终
checkpoint 的 downstream audit。

**对应代码/文档**：
[冻结版核心诊断](WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md)、
[WSS-PINN 路线真源](WSS_PINN/README.md)、[`wss_pinn` README](../../wss_pinn/README.md)、
`wss_pinn/AGENTS.md`、根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：下一轮科学合同冻结为 `FROZEN FOR IMPLEMENTATION v1.0`，可开始
Stage 0-a/0-b 的实现准备；未改代码、配置、数据、split、checkpoint 或训练产物，未启动训练。

**当前状态判断**：设计层面无剩余阻塞问题。Stage 0-a 通过后可进入纯监督 Stage 1；
Stage 0-b 仅阻塞 Stage 2/BC。任何 WSS 指标、test35 或 physics residual 均不得反选
checkpoint 或改变本轮架构顺序。

## 2026-08-06｜冻结 velocity→WSS，转入 `u,v,w,p` 优化主线

**本次主要修改**：将 Profile-Secant V3 的角色从“继续优化的 WSS 算法路线”收束为
冻结 downstream validator。核心诊断删除 WSS loss、直接 WSS 辅助监督、WSS 选模和
WSS 算法调优建议，改为 `u/v/w/speed/p` 多任务结构、区域采样、流量、pressure gauge、
压力梯度与压降的优化和验收；同时保留全壁面 overall/high-WSS/峰值限制，避免把冻结
验证器误写成无误差算子。

**对应代码/文档**：
[核心诊断与下一轮设计](WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md)、
[WSS-PINN 路线真源](WSS_PINN/README.md)、
[历史六臂预注册提示词](WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、`wss_pinn/AGENTS.md`、
[CFD 适配说明](../../wss_mri_calculator/README_CFD_ADAPTATION.md)、根 `README.md`、
`docs/README.md`、`docs/实验设计总纲.md`和 [项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：完成文档级范围收束和下一轮实验路线重排；未修改代码、配置、
数据、split 或模型产物，未启动新训练，也未继续调 WSS 算法。

**当前状态判断**：`u,v,w,p` 是当前唯一模型优化主线。冻结 WSS 仅在四通道主 checkpoint
按 validation 指标确定后运行一次 sanity check，不得参与 loss、早停、checkpoint、
采样比例或架构选择。

## 2026-08-05｜WSS-PINN 核心诊断的对抗性复核与下一轮因果设计

**本次主要修改**：从不可压缩瞬态流、RCR/Windkessel 边界、WSS 一阶导数可辨识性、
病例级统计和可部署性出发，重构了核心诊断文档。将准稳态算子实现正确性与
瞬态 peak 方程适定性分开；确认 val15 因 batch 等权聚合导致固定单病例双倍权重；
明确当前 outlet 目标是 monitor-derived proxy，未证明等于真实 RCR boundary profile 或部署可得输入；
撤回按 x/y/z 强行平衡 momentum 和用三例散点 residual 生成逐点 floor 的建议。新增
test35 development exposure、配对 case-bootstrap、队列异质性、边界 face 面积采样、
WSS 直接/近壁切向监督和分阶段 Go/No-Go。

**对应代码/文档**：
[对抗性修订后的核心诊断](WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md)、
[WSS-PINN 路线真源](WSS_PINN/README.md)、
[历史六臂预注册提示词](WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[`wss_pinn` README](../../wss_pinn/README.md)、
`wss_pinn/AGENTS.md`、`docs/实验设计总纲.md`和 [项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：完成 Stage 0 前的文档级科学审查和新实验路线预设计；未改代码、
配置、split、checkpoint 或评估产物，未启动新训练。只读复核了 V3 model/loss/dataset/train、
六臂 epoch 日志、test35 逐例 CSV、V2 residual audit 和代表病例 UDF。

**当前状态判断**：当前六臂仅能支持“geom 正增量、BC 速度近中性、当前
quasi-steady PDE 组合对 speed No-Go”的历史结论；不支持全局 latent 根因已定、
物理普遍无用或已可靠恢复 WSS。新训练前必须先修边界/选模/确认集合同，然后执行
global/local × raw/PE 的 2×2 无 PDE 消融；只在最佳监督设计上再验证 continuity、瞬态或弱形式 physics。

## 2026-08-05｜准稳态平滑场 V3 六臂完训、三 checkpoint 评估与结论回填

**本次主要修改**：从 Fluent 二进制 case 恢复 173 例真实 wall/inlet/four-outlet
边界，建立 train123/val15/test35 与 train-only 统计；实现无 3NN/IDW 的条件
PointNet 平滑 query 场、DATA/BC/PDE 分组损失、统一 validation checkpoint、六臂配置、
静态/GPU Gate、双节点提交和机器可读监控；补齐 pooled 回归指标、入口/出口边界、
预测方差/塌缩诊断，并将旧八臂汇总器扩展为 V3 六臂主表、九组增量、逐病例和收敛
汇总；新增可复跑的老师汇报损失绘图器，生成六臂 data/physics loss 总览、每臂全程与
最后 10% 放大、BC/PDE 子项和 9 页 PDF；新增汇报工作簿生成器，把主指标、九组增量、
checkpoint敏感性、loss收敛和210行逐病例结果整理为7张表并嵌入关键图。入口流量使用 exact peak
`vf-in-rfile.out` 实测值除真实网格面积；6 例 UDF 陈旧分母保留 warning，不伪造成
Gate 失败。wall 坐标审计改为 sidecar→全部 Fluent wall，并允许同表面重剖分。

**对应代码/产物**：`wss_pinn/{config.py,train.py,evaluate.py,losses.py}`、
`wss_pinn/{models,data,tools,cluster}/`、
`wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3/`、
`data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/`、
`outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_*/` 与
`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/`；损失绘图器为
`wss_pinn/tools/plot_v3_loss_convergence.py`，图件位于
`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/loss_convergence/`；工作簿生成器为
`wss_pinn/tools/build_v3_teacher_workbook.py`，结果为
`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/WSS_PINN_V3_六臂核心实验与指标汇报.xlsx`。

**推进到实验步骤**：boundary Gate 173/173、25 项测试、三模式 CPU smoke、六臂静态
preflight 和 master4+node04x2 CUDA dry-run 全通过；前 30 分钟 31/31 快照 healthy。
master Slurm `11301_[0-3]` 与 node04 PID `1241185/1241335` 对应六臂均完成 2500
epoch / 155000 step，三类 checkpoint 6/6 齐全；test35 全严格体域评估 18/18 完成，
主 checkpoint 固定 `best_validation_data`，test35 未参与训练、选模或续训判断。

**当前状态判断**：geom 是稳定正增量：E4−E1、E5−E2、E6−E3 的 speed R²_cb
为 `+0.1919/+0.1977/+0.1942`。BC 对总体速度近中性（case-balanced 小幅正、pooled
小幅负），但改善入口流量与出口压力。PDE 将 continuity/momentum residual 降低约
89–90%/60–62%，却使 E3−E2、E6−E5 的 speed R²_cb 分别下降 `0.0502/0.0537`，判定
为准稳态正则与瞬态 peak 标签竞争，而非训练不足。最后 10% validation 已小幅变差，
不满足续到 5000 的条件；near-wall speed R² 六臂全负，不形成 WSS 结论。结果真源见
[`summary/README.md`](../../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/README.md)。

## 2026-08-04｜Profile-Secant V3 35 例全壁面推理、采样与穿模审计

**本次主要修改**：新增 Profile-Secant V3 全壁面可恢复执行链，按既有 test35
原始壁面点数均衡成 8 个 CPU 分片，依次生成基础 V4 缓存、depth-3 剖面/secant
诊断和冻结高尾校准结果。扩展纯模型导出器，新增 132.8 万点同点 hexbin 散点、
最佳/最差高 WSS 空间图、单点内部支撑图、全病例壁面范围穿模风险图和全量 VTP。
历史 `profile_secant_v3/` 明确降级为 test35 × 1200 预览，未覆盖原产物。

**对应代码/文档**：
`wss_mri_calculator/src/run_profile_secant_v3_fullwall.py`、
`wss_mri_calculator/src/run_frozen_wss_compat.py`、
`wss_mri_calculator/viz/export_profile_secant_v3_visualization.py`、
`wss_mri_calculator/viz/visualize_v4_{single_target_sampling,penetration_risk}.py`、
[全壁面结果与复现](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/README.md)、
[V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：Profile-Secant V3 模型 SHA256 保持
`c1e53af5…3a02`；test35 35/35、`1,328,017` 点完成，0 failure。正式结果 JSON /
predictions SHA256=`31a4bdca…d7e89` / `9aaa80ad…bdbf`；最佳/最差全量 VTP 源同点
分别为 67,760 / 18,551，STL 顶点映射覆盖率均 100%。

**当前状态判断**：full-wall overall R² 均值=`0.96037`、pooled overall
R²=`0.96790`、pooled high-WSS R²=`0.94405`、逐病例 high-WSS R² 均值
=`0.77355`。sampled 与 full-wall 的整体/平均高 WSS 结论稳定，但全壁面平均峰值
低估=`15.81%`，sampled `9.31%` 不能再作为全量峰值结论。最佳病例穿模审计拦截
2,450 个门控前跨壁风险候选，最差病例严格门控前风险为 0；两者门控后保留风险均为 0。

## 2026-08-04｜体域 PINN 下一阶段准稳态平滑场六臂执行交接

**本次主要修改**：将用户确认的下一阶段实验固化为可直接复制到新窗口的执行提示词。
路线使用 Fluent 瞬态 peak `u,v,w,p` 标签和 Carreau–Yasuda 准稳态 PDE 正则，采用
无 PointNet++/3NN/IDW 的条件 PointNet 平滑坐标场，并用
`xyz/xyz+geom × DATA/DATA+BC/DATA+BC+PDE` 六臂拆分 BC、PDE 与几何特征增量。

**对应代码/文档**：
[准稳态平滑场六臂实验交接（已归档）](WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)、
[体域 PINN 路线真源](WSS_PINN/README.md)、
[`wss_pinn` 代码入口](../../wss_pinn/README.md)、`docs/README.md` 和
[项目级推进摘要](代码修改与实验推进记录.md)。

**推进到实验步骤**：本次只冻结科学合同、边界资产 Gate、train123/val15/test35、
六臂矩阵、loss 分组、2500→5000 同预算续训条件、统一 validation checkpoint、正式
提交与监控要求；未修改训练代码、未构建新边界 sidecar、未提交 Slurm。

**当前状态判断**：handoff ready / implementation pending。下一智能体必须先证明真实
peak inlet/outlet 坐标与目标可恢复；Gate 通过后已获授权推进实现、测试、GPU
preflight、正式六臂提交、完训评估和文档收口。若 Gate 失败，禁止伪造 BC 或把只有
no-slip 的实验称为完整 DATA+BC。

## 2026-08-04｜Profile-Secant V3 纯 quicklook 与最佳/最差病例 VTP

**本次主要修改**：新增 Profile-Secant V3 专用可视化导出脚本和独立结果目录，图件
只读取 `high_tail_wss_vec`，不混入 V4 final 预测。生成 test35 指标汇总、最佳/最差
病例空间图、高 WSS 尾部图、总览联系表，以及与既有 `06_postview_vtp` 同结构的两份
ParaView 面片包。通用三联图工具增加可选预测标签，默认行为保持不变。

**对应代码/文档**：
`wss_mri_calculator/viz/export_profile_secant_v3_visualization.py`、
`tools/cfdpost_cloud_export/plot_stl_mapped_triptych.py`、
[纯模型可视化入口](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3/README.md)、
[V4/Profile 方法说明](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/README.md)、
[CFD 适配说明](../../wss_mri_calculator/README_CFD_ADAPTATION.md)。

**推进到实验步骤**：未训练或调参；直接读取冻结的 test35 × 1200 Profile-Secant V3
predictions。按逐病例 high-WSS R² 选择最佳 `ILO/LI_YOU_ZHI-0/before`
（`0.9560`）和最差 `AG/slow/ZHANG_WEI_XIAN`（`0.2381`），将同点标量 Gaussian
映射到病例 STL，并输出主 VTP、源 CSV、映射报告、预览图、色标和 manifest。

**当前状态判断**：纯 quicklook 和两份 VTP 已可直接使用。最佳病例 VTP 映射覆盖率
为 `99.51%`，最差病例为 `99.88%`；VTP 仅用于展示，正式 R² 仍以冻结同点 JSON/CSV
为准。脚本与通用三联图工具 `py_compile` 通过。

## 2026-08-04｜体域 PINN V2 零重训 residual 根因验证与老师论文对照

**本次主要修改**：新增冻结 checkpoint 的物理诊断工具并完成三项只读审计：test35
exact support / 独立体域 / jitter query residual；AG/AAA/ILO 各 1 例 CFD 真值的
准稳态与加入 `du/dt` 后 residual；当前 V2 到 Liao 2025/老师源码 NMAE 口径的指标
桥接。同步纠正路线文档中把 SAME5K 低 residual 当作连续场物理闭合证据的旧表述，
并补齐老师论文和当前代码的设计差异。

**对应代码/文档**：`wss_pinn/tools/diagnose_v2_physics.py`、
[V2 路线真源](WSS_PINN/README.md)、[`wss_pinn` 代码入口](../../wss_pinn/README.md)、
[老师论文与 V2 对照](../paper_reproduction/papers/hemodynamics_pointcloud_pinn/README.md)、
`outputs/wss_pinn/audits/volume_uvwp_peak_same5k_e7500_v2_residual_diagnosis/`、
根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`。

**推进到实验步骤**：未重新训练。固定使用
`VF-PNPP-XYZG-PINN-SAME5K-E7500-s1234-v2/last.pt`；网络审计覆盖 35/35 test，
每例 512 点；CFD 审计覆盖三域各 1 例、每例 256 个 strict-core 点，并用 `k=48/96`
检查局部二次拟合敏感性。解析三例 Fluent journal，确认 solver dt=`0.005 s`、中心
差分间隔=`0.02 s`；解析老师论文 PDF 和现有代码快照。

**当前状态判断**：SAME-IDW 根因已确认：独立 query 的 continuity/momentum/速度
梯度 RMS 约为 exact support 的 `453×/331×/187×`；`1e-4` 微扰下速度仅变化
`1.84e-5 m/s`，动量却放大 `120×`。CFD `k=96` 加时间项后 residual 均值
`1737→1494 Pa/m`，说明准稳态缺项真实但只解释部分误差，且绝对值受二阶导数邻域
敏感性限制。按老师源码 global-range NMAE，V2 可得约 `2.18%` 而 speed R² 仍仅
`0.2380`，确认原文 NMAE/FR-PD R² 不能直接证明逐点速度更准。下一轮训练前应先修
独立 collocation/连续 query、可观测 inlet/outlet BC 与瞬态方程，不再追加 epoch。

## 2026-08-04｜Profile-Secant V3 推荐模型口径纠正

**本次主要修改**：根据用户明确的选择目标“病例总体和平均高 WSS 精度”，纠正此前
把 V4 final 写成推荐版本的口径。现统一将 Profile-Secant V3 定为当前推荐结果模型，
用于主结果、可视化和后续总体/平均高 WSS 分析；V4 final 改为冻结基线和消融对照。
逐病例严格目标未达的限制保持不变。

**对应代码/文档**：[V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)、
[CFD 适配说明](../../wss_mri_calculator/README_CFD_ADAPTATION.md)、
[V4/Profile 方法说明](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/README.md)、
[完整结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/RESULTS.md)、
[Profile-Secant V3 推荐结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md)、
根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`、
[高值区域预测优化方案](_archive/WSS最小化/WSS高值区域预测优化方案.md) 和 [体域 PINN 路线](WSS_PINN/README.md)。

**推进到实验步骤**：只纠正文档中的模型选择与报告口径；未重新训练、未重新推理、
未修改模型、预测、split 或哈希。选择依据仍为已落盘的 train138 grouped OOF、固定
holdout65 和 test35 结果。

**当前状态判断**：Profile-Secant V3 的 test35 整体/pooled high-WSS R² 为
`0.9617/0.9415`，优于 V4 final 的 `0.9561/0.9282`，因此符合用户当前主目标并作为
推荐结果模型。其严格双目标仅 4/35 病例达标，推荐使用不等于逐病例可靠保证。

## 2026-08-04｜`wss_pinn` 当前主线根层化与旧 WSS-target 源码归档

**本次主要修改**：将活动峰值体域 `u,v,w,p` 路线从
`wss_pinn/volume_field/` 提升到 `wss_pinn/` 根包，统一训练、评估、数据、模型、
物理、工具和 Slurm 入口；把已停止的直接 WSS/F0–F2 实现、配置和测试移入源码
归档。保留当前路线仍复用的 raw CFD 读取、抽取后的坐标对齐和通用写盘/哈希工具，
并将这些活动依赖纳入静态 preflight 的实现哈希。

**对应代码/文档**：`wss_pinn/config.py`、`wss_pinn/train.py`、
`wss_pinn/evaluate.py`、`wss_pinn/{data,models,physics,tools,cluster}/`、
[`wss_pinn` 代码说明](../../wss_pinn/README.md)、
[`wss_pinn` 归档索引](../../wss_pinn/archive/README.md)、
[历史源码归档](../../wss_pinn/archive/wss_target_v1_20260730/README.md)、
`wss_pinn/AGENTS.md`、[体域 PINN 路线真源](WSS_PINN/README.md)及其历史文档归档。

**推进到实验步骤**：只做工程结构收敛和入口迁移；未启动训练、未改配置数值、
未改数据 split、未改模型/loss/评估口径，也未移动既有数据和输出。

**当前状态判断**：活动代码已直接位于 `wss_pinn` 根层；旧路线有可浏览源码归档，
不再与当前入口混放。现有 v1/v2 实验结论和输出路径保持不变。

## 2026-08-04｜高 WSS Profile-Secant V3、可辨识性上限与中文文档同步

**本次主要修改**：将冻结 V4 final 之后的高 WSS 增量研究统一收口为中文；明确
Profile-Secant V3 是面向总体和平均高 WSS 精度的当前推荐结果模型，V4 final 保留
为冻结基线，并新增 2026-08-04 test35 高 WSS 对比图。把 pooled/平均达标与逐病例
严格未达标分开报告，补充 oracle 上限、
近壁采样深度相关性和已否决路线，避免把整体 R² 高误写成每个病例高 WSS 可靠。

**对应代码/文档**：[V1–V4 总跟踪](../../wss_mri_calculator/experiments/README.md)、
[V4 方法说明](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/README.md)、
[V4 完整结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/RESULTS.md)、
[Profile-Secant V3 结果](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md)、
[高 WSS quicklook](../../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/00_quicklook/README.md)、
根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`、
[高值区域预测优化方案](_archive/WSS最小化/WSS高值区域预测优化方案.md) 和 [体域 PINN 路线](WSS_PINN/README.md)。

**推进到实验步骤**：未重新训练、未做 test35 调参。直接读取已冻结的
`calibrator_profile_secant_high_tail_anchor10_v3.joblib` 和 prediction archive，生成
`15_profile_secant_high_tail_comparison.png/json`。相关 12 项单元/合同测试既有记录
为通过，重复推理既有记录为逐字节一致。

**当前状态判断**：test35 整体 R²=`0.9617`、pooled high-WSS R²=`0.9415`、
逐病例 high-WSS R² 均值=`0.7721`、high-WSS NRMSE=`0.1583`、平均峰值低估
=`9.31%`。但仅 6/35 病例 high-WSS R² `≥0.90`，仅 4/35 同时满足峰值低估
`≤10%`。任意单调 oracle 仍有 13/35 不达标，现有输入上的后校准优化已接近信息上限；
严格提升需要更靠近壁面的速度层或更高近壁分辨率。当前主结果使用
Profile-Secant V3，V4 final 只作冻结对照。

## 2026-08-03｜峰值体域 SAME5K-E7500 v2 8/8 完训与速度精度审计

**本次主要修改**：完成第二轮峰值体域 `u,v,w,p` 八实验的 SAME5K 主协议、
固定 5k support→full-volume 副协议、三 checkpoint 敏感性、逐病例和分区域审计；
将路线真源、Agent 边界、代码 README、项目入口与实验总纲从 running 更新为
completed，并明确整体速度精度、近壁短板和当前模型选择。

**对应代码/文档**：[峰值体域 PINN 路线真源](WSS_PINN/README.md)、
`wss_pinn/AGENTS.md`、[`wss_pinn/README.md`](../../wss_pinn/README.md)、
根 [`README.md`](../../README.md)、
`docs/README.md`、`docs/实验设计总纲.md`、
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/runs/`。

**推进到实验步骤**：GPU preflight `11137` 与训练 array `11138_[0-7%4]` 均
8/8 completed；每臂 7500 epoch / 517,500 step，8 份 epoch 日志均为 7500 行，
SAME5K/full-volume × `best_data/best_total/last` 共 48 份 evaluation JSON 完整。
最低训练 loss 均位于 epoch 6815–7487，三 checkpoint 的 SAME5K speed R² 极差仅
`0.0006–0.0230`，训练已基本平台。

**当前状态判断**：V2 速度主锚点冻结为 PointNet++ + `xyz+geom` + PINN / `last@7500`；
其 SAME5K/full-volume speed R²=`0.2380/0.1820`、MAE=`0.1764/0.1863 m/s`，
30/35 病例 SAME5K speed R² 为正。PINN 四对聚合 `u/v/w/speed/p` R² 均提高，
但 PointNet++ 两对 speed 增益仅 `+0.0705/+0.0317` 且逐病例约一半获益；near-wall
speed R² 仍全部为负。下一步不再只延长 epoch，优先诊断 5 个负 R² 病例与近壁
标记/采样/损失，并在独立确认集或多 seed 上复核。

## 2026-08-03｜velocity→WSS V1–V4 总跟踪建立并同步 V4 冻结结论

**本次主要修改**：为 `wss_mri_calculator` 建立 V1–V4 唯一实验总跟踪页，明确
纯物理前向、train-only 全局标量和 V4 冻结诊断校准的监督边界；将 CFD 适配说明中
`alpha≈1.2` 改为 V1/V2 历史诊断，并同步根 README、docs 索引和实验总纲的当前状态。

**对应代码/文档**：
`wss_mri_calculator/experiments/README.md`、
`wss_mri_calculator/README_CFD_ADAPTATION.md`、根 `README.md`、
`docs/README.md`、`docs/实验设计总纲.md`、V1–V4 各实验目录及冻结清单。

**推进到实验步骤**：V1–V4 均为 frozen。统一 test35 × 1200 下，V4 physics
raw/scaled R² mean=`0.90788/0.93612`；V4 final 为
`0.95607/0.95844`，raw p05/min=`0.92391/0.90025`，35/35 病例优于 V3。
train138 grouped OOF 与 development73→holdout65 的 raw R² mean 分别为
`0.95493/0.95790`。

**当前状态判断**：V4 final 是当前推荐 velocity→WSS 方法；V1/V2/V3 保留为冻结
演进证据，禁止 post-test tuning。下一阶段应转向新外部队列、网格/时间步鲁棒性、
分支感知归属和不确定性，不继续通过增加射线站点或多项式阶数追分。

## 2026-08-02｜峰值体域 SAME5K-E7500 v2 实现并提交

**本次主要修改**：按用户确认的简单汇报口径新增独立八臂 v2：train138/val0/test35，
严格体域 random5000，support/query 使用相同索引，每 epoch 重采样，固定 7500 epoch。
扩展数据集和配置 Gate 以兼容 SAME，同时保留旧 SEP；新增 400/1000/2500/5000/7500
里程碑 checkpoint；评估新增 SAME5K 与固定 5k support→full-volume 双协议，并补充
case-balanced 的逐目标 R²/MAE/RMSE、三分量 momentum 与 wall P95/max。Slurm
preflight/训练均设 `--time=0`，实查 GPU 分区 `MaxTime=UNLIMITED`。

**对应代码/文档**：`wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/`、
`wss_pinn/{config.py,data/dataset.py,train.py,evaluate.py}`、
`wss_pinn/cluster/`、[路线真源](WSS_PINN/README.md)、
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/submission.json`。

**推进到实验步骤**：`test_volume_*` 20/20 通过；新矩阵静态 preflight pass；正式提交
GPU preflight `11137`，训练 array `11138_[0-7%4]` 以 `afterok:11137` 等待释放。

**当前状态判断**：GPU preflight `11137` 8/8 completed；array `11138` running。首批
四个 PointNet 臂连续观察 30:35，约 22,432 条已记录 step 全部 finite，四份 stderr
均为空，无 NaN/Inf/OOM/Traceback/CUDA error/restart；400/1000 里程碑与定期
checkpoint 写盘正常。梯度裁剪率 data-only 约 50%–54%、PINN 约 90%–95%，作为后续
诊断保留。机器可读快照：
`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/monitor_30min.json`。

## 2026-08-02｜峰值体域 uvwp 八臂完训、结果图与 15,000 epoch 候选协议

**本次主要修改**：完成 array `11128_[0-7]` 的 8/8 run、checkpoint、日志与
24 份 evaluation JSON 完整性审计；新增可复现结果汇总脚本，生成 `best_total` / test35
配对 CSV/JSON、收敛统计和两张 PNG/SVG。将路线真源、代码 README、项目入口与总纲
从 running/pending 更新为 completed/audited。按用户修正把下一轮预算记录为
`max_epochs=15000`（约 1,035,000 step）的硬上限，并预注册 train138-only fixed
monitor、data fidelity/Pareto 早停边界；当前仅为候选协议，未实现、未提交训练。

**对应代码/文档**：
`wss_pinn/tools/summarize_results.py`、
`outputs/wss_pinn/volume_uvwp_peak_v1/summary/`、
[峰值体域 PINN 结果真源](WSS_PINN/README.md)、
[`wss_pinn/README.md`](../../wss_pinn/README.md)、根 `README.md`、
`docs/README.md`、`docs/实验设计总纲.md`。结果 manifest SHA256：
`e58be163…3781de8`。

**推进到实验步骤**：首轮八实验 8/8 completed；每臂 400 epoch、27,600 step，
配对初始化哈希一致、`warm_start=false`，无 NaN/Inf/OOM/traceback。四个 PINN 配对的
continuity/momentum/wall RMS 均下降，pressure R² 均提高；PointNet + `xyz+geom` 是
唯一 `u/v/w/speed/p` 全部提高的配对。静态图与机器可读表已落盘。

**当前状态判断**：首轮结果 completed / audited，但不能宣称 PINN 全面优于
data-only，也不能宣称速度场已充分收敛。15,000 epoch 是下一轮硬上限，不是本轮已经
执行的预算；早停工程实现和新配置仍 pending，正式长训练未提交。

## 2026-08-01｜schema-v2 数据 Gate 通过并启动 4-GPU 八臂训练

**本次主要修改**：按 4×RTX 4090 集群资源把训练数组并发上限改为
`0-7%4`。提交 CPU 数据构建 Job `11122`、deep-source 审计 Job `11123` 和自动
launcher `11124`；173/173 schema-v2 sidecar、注册速度、严格体域压力 gauge、
壁面重复行、split/manifest/train-only stats hash 全部通过。首次 GPU preflight
`11125` 发现 CuBLAS 确定性环境缺失后，在正式训练开始前主动取消 `11125/11126`，
补充 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，重跑 40/40 单测与静态 preflight 后提交
GPU preflight `11127` 和正式 array `11128_[0-7%4]`。

**对应代码/文档**：`wss_pinn/cluster/{preflight,run_experiment}.slurm`、
`wss_pinn/{train.py,cluster/submit_matrix.py}`、
`data_wss_pinn/volume_uvwp_peak_v1_train138_test35/`、
`outputs/wss_pinn/audits/volume_uvwp_peak_v1_train138_test35/`、
`outputs/wss_pinn/volume_uvwp_peak_v1/submission.json`、[当前路线](WSS_PINN/README.md)。

**推进到实验步骤**：build `11122` 与 deep audit `11123` completed；aggregate
manifest SHA256 `e425a657…54a1bf`，field stats SHA256 `9c15ab2b…905f46`，最终
data Gate report SHA256 `d5a86b8a…386e78`。GPU preflight `11127` completed
`0:0`；训练数组 `11128` 正在按最多 4 卡并发执行。

**当前状态判断**：正式训练前 30 分钟监控通过。无 traceback、NaN、non-finite
loss、OOM、任务重启或 GPU 过热；三条 data-only 已完成 400 epoch 与三 checkpoint
评估，PointNet 两条 PINN、PointNet++ xyz PINN 和 PointNet++ xyz+geom data-only
继续运行，最后一臂因 `%4` 上限正常等待。评估存在 mmap 只读数组转 tensor 的非致命
warning；代码没有原地写入，数值安全，待本轮结束后统一消除以保持当前运行代码口径。

## 2026-08-01｜WSS-PINN 重构为峰值体域 `u,v,w,p` 八实验

**本次主要修改**：按用户重新确认的边界，停止旧“直接 WSS 目标 + 辅助体域分支”
路线，新增独立 `volume_uvwp_peak_v1`：只做 peak 时刻，PointNet / 纯 PointNet++ ×
xyz/xyz+3 几何特征 × data-only/PINN 共 8 个实验；四通道 `u,v,w,p` 全部真值监督，
PINN 从随机初始化的第一个 step 同时加入 continuity、完整三分量 Carreau–Yasuda
momentum 和壁面 no-slip，不从 data-only 热启动。实现可微 query-coordinate 路径、
完整变黏度应力散度、逐 step/epoch data loss 与 physics loss 日志、物理单位评估和
同 run resume 保护。

数据合同升级为 schema v2：速度向量与坐标一起旋转；识别三个 bundle 的体表壁面重复
行并从 support/query/PDE、压力 gauge 和归一化统计排除；压力按病例严格体域固定均值
中心化；几何只保留 `abscissa_norm/local_radius/signed_log1p(curvature)`，曲率裁剪和
统计只来自 train138 严格体域。补充 manifest/hash/split/train-membership 数据 Gate 和
合成回归测试。旧 WSS-target 文档移入 `_archive/wss_target_v1_20260730/`，旧源码以
Git commit `bca002025d40b290d171570e6470f484fd4feec7` 追溯，不删除旧数据或输出。

**对应代码/文档**：`wss_pinn/`、
`wss_pinn/configs/volume_uvwp_peak_v1/`、`wss_pinn/tests/test_volume_*.py`、
`wss_pinn/{AGENTS.md,README.md}`、[当前体域 PINN 路线](WSS_PINN/README.md)、
[旧路线归档](WSS_PINN/_archive/wss_target_v1_20260730/README.md)、
`outputs/wss_pinn/audits/volume_uvwp_peak_v1_preflight/report.json`。

**推进到实验步骤**：`compileall` 通过，`wss_pinn/tests` 完整单元测试 40/40 通过，
八臂静态配对 preflight 为 pass；默认提交器 dry-run 返回 `jobs=[]` 和
`formal_training_submitted=false`。未构建全量 schema-v2 sidecar，未运行 GPU 八臂
dry-run，未提交正式训练。

**当前状态判断**：implementation ready / static preflight passed / new data Gate
pending / formal training not submitted。下一步是 CPU 集群构建 173 例 schema-v2
sidecar 并做 deep-source audit；数据 Gate 通过后仍需用户再次授权正式训练。

## 2026-08-01｜根 README 改以 WSS-min + wss_mri_calculator 为主入口

**本次主要修改**：重写仓库根 [`README.md`](../../README.md)：把当前主攻从旧 `pipeline/` 叙事改为 **WSS-min（预处理+训练）** 与 **`wss_mri_calculator`（CFD velocity→WSS）**；补上目录表、四阶段/训练/CFD 快速命令、v4 数据与 LSA2+H2+`log(local_radius)` 锚点、adaptive-CV v1 / multiscale v2 状态；去掉失效的本机绝对路径链接；旧 `pipeline/` / V3 / baseline 收为次级入口。

**对应代码/文档**：[`README.md`](../../README.md)；本推进记录。

**推进到实验步骤**：文档导航（无新训练/无新批量 WSS）。

**当前状态判断**：新人从根 README 可直接落到 `pipeline_wss_min`、`training_wss_min`、`README_CFD_ADAPTATION.md` 与两条实验目录；细节仍以各子 README / 矩阵文档为准。


## 2026-08-05｜WSS-PINN 核心代码维护性精简

**目标**：参考 `wss_pinn/PIPN-QN Code/` 的直接组织风格，减少当前体域 PINN 核心
入口中的重复分支和过重测试，同时保留 JSON 配置、矩阵 preflight、数据 Gate、
checkpoint/resume 与 Slurm 提交能力。

**代码修改**：

- `config.py`：新增推荐名 `ExperimentConfig`，用 route 合同表集中声明 mode、架构、
  激活和冻结 split；`VolumeExperimentConfig` 继续作为兼容别名。
- `models/point_models.py`：推荐构造入口改为 `build_model`，旧
  `build_volume_model` 保留别名。
- `losses.py`：统一 V1/V2/V3 的数据项与 PDE 计算，只在边界组合和注册权重上分 route；
  `compute_losses` 从 `(losses, diagnostics)` 简化为直接返回 `losses`。
- `train.py`：删除没有汇总脚本读取的逐 step 剪切率、黏度、残差 RMS 重复诊断；
  继续保存逐 step/epoch loss、初始化证据、best/last/milestone checkpoint 和同 run resume。
- `evaluate.py`：将重复的点云分块预测、near-wall/core 指标、V3 边界和物理 residual
  拆为小函数，主病例循环保持线性可读；输出 schema 不变。
- `cluster/submit_matrix.py`：默认从 `matrix.json` 解析并校验实验顺序，再生成 Slurm
  使用的 config list；`--config-list` 仅保留为已完成实验兼容入口。
- 测试由 7 文件缩为 `test_volume_{config,data,models,physics}.py` 四文件，共 18 项。
  删除大型 synthetic data Gate 和多轮 resume 集成 fixture；真实数据审计与逐配置 GPU
  dry-run 仍由 preflight 负责。

**验证**：18/18 核心测试通过；V1/V3 静态 preflight 分别 4/4、2/2 配对组通过；
V1 `pinn` 与 V3 `data_bc_pde` synthetic forward/backward 均得到有限 loss。没有提交
新训练，也没有改写既有实验结果。
