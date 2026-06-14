# LaB-GATr

## 基本信息

- 论文：LaB-GATr / large biomedical geometric algebra transformer
- 代码：`https://github.com/sukjulian/lab-gatr`
- 任务：biomedical surface / volume mesh 上的逐点回归
- 本项目优先级：P0

## 方法要点

LaB-GATr 使用 geometric algebra transformer 处理 biomedical mesh / point tokens，并支持从低分辨率 token 回插到原分辨率点。它可作为等变 AAA WSS 论文的底座，也可作为本项目替代 PointNeXt 的等变 surface/point baseline。

## 与本项目的关系

适合回答：

- 等变 transformer 是否比 PointNeXt-style local pooling 更适合壁面 WSS；
- 是否能通过 tokenisation 降低大点云/面片成本；
- 是否能缓解 `wss_x/y` 在全局坐标下长期接近 0 的问题。

## 适配清单

- [ ] 审计 PyG / xFormers / GATr 依赖
- [ ] 确认 point-cloud pooling 和原分辨率插值接口
- [ ] 建立本项目 wall surface / near-wall 点输入
- [ ] 输出 WSS vector 或 TAWSS
- [ ] 与 PointNeXt、coronary mesh convolution 分表比较

## 风险

- 依赖较重，训练成本可能高。
- 如果 AAA WSS 目标仓库后续补齐代码，应优先复现 paper-specific repo，再用 LaB-GATr 作底座解释。
