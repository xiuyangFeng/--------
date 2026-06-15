# External Baselines

> 定位：存放外部论文 baseline 的最小复现代码。这里的代码用于在本项目私有数据上复现公开论文方法，不作为任务 A V1/V2/V3 内部训练代码的一部分。

## 当前内容

| baseline | 目录 | 状态 |
| --- | --- | --- |
| PointNetCFD | `pointnetcfd/` | 已建立 PyG 图快照读取、PointNetCFD-style 训练入口、四组初始配置和 Slurm 模板 |

## 使用原则

- 外部 baseline 必须使用与任务 A 一致的患者级 split。
- 第一轮先跑 paper-original 或最接近 paper-original 的输入输出，再做显式几何特征增强。
- 输出表必须和 V3 主线分开记录，避免把外部模型适配写成 V3 内部优化。
- 若外部论文只预测速度/压力，不应直接写成 WSS baseline，除非完整走通 WSS 后处理或直接 WSS head。
