"""pipeline_wss_min

最小化 WSS-only 数据预处理流程（与旧 `pipeline/` 完全独立，互不影响）。

设计目标（见 README.md）:
- 从原始 CFD (ascii 壁面 + ascii_in 内部 + centerline) 出发
- 先做解剖学刚性配准（中心线主导）+ 坐标正交化/标准化，再做稀疏化
- 保留 壁面/近壁/内部 的 point_type 掩码，几何特征全部保留供后续 mask
- 一次性处理全部时间步；样本装配阶段默认取峰值收缩期单样本，可切换全相位
- 稀疏化与样本装配都由配置驱动，方便扫点数
"""

__all__ = ["config"]
