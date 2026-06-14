# AneuG-Flow / IA WSS benchmark

## 基本信息

- 方向：颅内动脉瘤 synthetic CFD / WSS benchmark
- 代码/数据：`https://github.com/WenHaoDing/AneuG-Flow`，另有 HuggingFace dataset / OpenReview 入口
- 本项目优先级：P1

## 方法要点

AneuG-Flow 提供颅内动脉瘤几何和 CFD/hemodynamics 数据，可用于测试医学血管 WSS / velocity / pressure surrogate。它不是 AAA 数据，但对“公开医学血管 WSS benchmark”很有参考价值。

## 与本项目的关系

适合：

- 作为外部 WSS benchmark；
- 对齐社区中的 hemodynamics 数据字段和指标；
- 用来检验某些模型是否只对本项目私有数据有效。

不适合：

- 直接替代 AAA 私有数据 baseline；
- 把 IA 上的 WSS 结论写成 AAA 结论。

## 适配清单

- [ ] 下载小样本数据并检查字段
- [ ] 对齐 velocity / pressure / WSS 命名和单位
- [ ] 记录 IA 与 AAA 几何、BC、mesh 差异
- [ ] 若用于模型预训练，必须严格避免本项目 test leakage
