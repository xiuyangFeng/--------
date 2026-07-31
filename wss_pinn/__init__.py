"""独立的 WSS-PINN 实验线包入口。

最终目标仍是预测峰值壁面剪切应力（WSS）；
速度/压力场 (u,v,w,p) 只作为训练期的体域辅助量与物理约束变量。

隔离约定（重要）：
- ``data_new`` / ``data_wss_min`` / ``pipeline_wss_min`` / ``training_wss_min``
  视为只读上游，不得在此写入 PINN 专用产物。
- PINN 派生数据 → ``data_wss_pinn/``
- 训练与评估输出 → ``outputs/wss_pinn/``
- 本包代码与配置 → ``wss_pinn/``
"""

__version__ = "0.1.0"
