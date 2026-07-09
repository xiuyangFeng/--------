"""training_wss_min — 独立的 WSS-min baseline 训练/评估包。

与既有 `training/`（V3P 图/变换器栈）**完全独立**，不共享任何代码，
只读 `pipeline_wss_min` 产出的 `data_wss_min/**/bundle.npz` 与全局 WSS 统计。

设计目标（见 README.md）：
- 任务：几何点云 (x,y,z[+几何]) -> 壁面 WSS 标量（log_z 归一化），单头。
- 模型：PointNeXt-S 残差版（InvResMLP + ball-query 分组），密度鲁棒。
- 训练用稀疏子采样（可扫点数），评估恒在**完整壁面点云**上（A 路：部署有完整几何）。
- 一切配置驱动，集群并行提交，日志全程落盘便于复查。
"""

__all__ = []
