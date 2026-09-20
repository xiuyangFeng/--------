# 波 1b：Murray 先验的 seed 复核、拆分与组合（wss_local_wave1b_20260912）

波 1（`../wss_local_wave1_20260912/README.md`）第一轮发现 X5（Murray 分支流量先验）是唯一超带的臂，随即追加本批 8 次单 run：X0/X5 各补 seed 7、2025（同 seed 配对参考链 `configs/wss_local_wave1b_20260912/refs/C1_s<seed>.json → refs/M2_s<seed>.json`，同 seed 的 X0 与 X5 共享全部继承张量的初始权重）；X5q（只 log_q_branch_murray）、X5t（只 log_tau0_murray）；X5F1（X5+X2 弯曲角）、X5S1a（X5+X7 切平面 patch）。

- 生成：`python -m training_wss_min.tools.prepare_wss_local_wave1b`；预检 `cluster/preflight_wss_local_wave1b.slurm`（Slurm 14168，afterany:14160）；矩阵 `cluster/run_wss_local_wave1b.slurm`（14169，afterok:14168，4 卡 × 2 槽）。
- 汇总：`python -m training_wss_min.tools.report_wss_local_wave1 --config-dir configs/wss_local_wave1b_20260912 --experiment-dir experiments/wss_local_wave1b_20260912` → `results.md` / `results.json`；工作簿总览第 364–371 行（`tools/update_wss_local_wave1_xlsx.py --name wss_local_wave1b_20260912 --group ...`，回填前备份在本目录）。
- 结果与判读：跟踪文档 §20.3。要点：X5 三 seed 配对 Δ 物理 +0.069（sd 0.031）/ 归一化 +0.017（sd 0.003），三 seed 均值 0.7173（sd 0.004）对 C1 配置四次运行均值 0.6506（sd 0.027）；增益来自 log Q 份额（X5q ≈ X5，X5t 物理低 0.017）；X5+X2 / X5+X7 无物理增益（归一化 +0.004 / +0.003 带内）。
