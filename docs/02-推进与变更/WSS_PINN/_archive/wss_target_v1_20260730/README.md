# WSS-target PINN v1 历史归档

> 归档日期：2026-08-01
>
> 状态：historical / frozen / not active

本目录保留 2026-07-30 前后的旧 WSS-PINN 方案：最终目标仍为直接 WSS，体域
`u,v,w,p` 作为辅助分支，并按 F0/F1/F2 阶梯推进。用户在 2026-08-01 重新收敛边界后，
该方案被新的“峰值四通道体域场 + PointNet/PointNet++ 八实验”路线替代。

归档内容：

- `WSS_PINN_阶梯实验矩阵与进度跟踪.md`
- `WSS_PINN_下一智能体目标提示词_推进至F1.md`

旧源码已在 2026-08-04 从活动包根层移入：

[`wss_pinn/archive/wss_target_v1_20260730/`](../../../../../wss_pinn/archive/wss_target_v1_20260730/README.md)

其中保存旧训练入口、模型、物理项、工具、Slurm、测试与 F0/F1/F2 配置。旧
`data/sidecar.py` 已随路线归档；当前体域路线只保留 `data/raw_io.py`、从旧 sidecar
提取的 `data/alignment.py` 和 `utils.py`。旧 sidecar 产物与输出不移动，避免破坏
历史结果引用。完全复原原 import surface 时使用 Git 快照：

`bca002025d40b290d171570e6470f484fd4feec7`

注意：

- 旧 8k near-wall + 8k core sidecar 不符合新路线完整体域均匀随机合同；
- 旧 WSS checkpoint 不得作为新 PINN 热启动；
- 旧 F0/F1/F2 状态不得写成新路线已完成；
- 历史文档中的路径、Job ID 和结论仅解释当时实验。

当前活动入口：

- [`wss_pinn/README.md`](../../../../../wss_pinn/README.md)
- [峰值体域 `u,v,w,p` PINN 路线](../../README.md)
