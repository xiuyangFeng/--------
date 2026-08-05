"""峰值体域 ``(u, v, w, p)`` 物理信息神经网络（PINN）活动路线包。

活动训练、评估、模型、数据与物理模块均直接位于 :mod:`wss_pinn` 根层。
已被取代的「直接预测 WSS + F0/F1/F2 阶梯」路线冻结在
``wss_pinn/archive/wss_target_v1_20260730``，仅供追溯，不可作为执行入口。

对外导出简短的 :class:`ExperimentConfig`；旧名称继续保留以兼容历史脚本。
拿到的是带科学合同护栏的解析后配置，而不是原始 JSON dict。
"""

from .config import ExperimentConfig, VolumeExperimentConfig

__all__ = ["ExperimentConfig", "VolumeExperimentConfig"]
