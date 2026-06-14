# MultiViewUNet AAA TAWSS

## 基本信息

- 论文：Rapid wall shear stress prediction for aortic aneurysms using deep learning: a fast alternative to CFD
- 代码：`https://github.com/atick-faisal/MultiViewUNet-Aneurysm`
- 任务：AAA TAWSS prediction
- 本项目优先级：P0

## 方法要点

该工作将复杂 3D 主动脉几何通过 domain transformation 转成 2D 多视图表示，再用 U-Net 预测 TAWSS。它不预测完整 3D 时序场，而是直接面向壁面时间平均 WSS。

## 与本项目的关系

适合成为非图 baseline，并且能直接回应本项目 G4 2D 展开路线的成败：

- 若 2D 展开在外部方法中有效，本项目需要认真比较展开方式；
- 若在本项目数据上 2D grid 指标高但回映射 3D 差，必须把反映射 gap 写入结论。

## 适配清单

- [ ] 审计仓库输入格式
- [ ] 用本项目 wall points / STL / centerline 生成 2D 展开图
- [ ] 训练输出 TAWSS grid
- [ ] 回映射到 3D wall points
- [ ] 同时报告 2D grid 指标和 3D wall 指标

## 风险

- 它只预测 TAWSS，不覆盖瞬态 WSS 和 `u/v/w/p`。
- 展开会在分叉处产生拓扑和度量扭曲。
