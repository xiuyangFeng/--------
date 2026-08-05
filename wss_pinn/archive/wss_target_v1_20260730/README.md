# 直接 WSS PINN v1 源码归档

> 归档日期：2026-08-04
>
> 对应历史阶段：2026-07-30 前后的 WSS-target F0/F1/F2 阶梯
>
> 状态：historical / frozen / not executable as the active package

本目录保存被峰值体域 `u,v,w,p` 路线替代的旧实现，包括：

- 根训练入口：`config.py`、`train.py`、`evaluate.py`、`objectives.py`；
- 旧模型与物理项：`models/`、`physics/`；
- 旧 F0/F1/F2 数据采样、审计、构建和 Slurm 入口：`data/`、`tools/`、`cluster/`；
- 旧配置与测试：`configs/`、`tests/`。

以下共享模块仍被当前体域路线使用，因此保留在活动区，没有复制到归档：

- `wss_pinn/data/raw_io.py`
- `wss_pinn/data/alignment.py`（从旧 sidecar 提取的 raw/bundle 坐标对齐）
- `wss_pinn/utils.py`

本归档用于代码审阅和历史追溯，不保证从当前包路径直接运行。需要完全复原当时的
import surface 时，应检出冻结提交：

`bca002025d40b290d171570e6470f484fd4feec7`

相关文档归档：

- [`docs/02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/`](../../../docs/02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/README.md)

当前活动入口：[`wss_pinn/README.md`](../../README.md)。
