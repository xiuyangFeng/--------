# CROWN private preprocessing outputs

该目录用于保存 CROWN/Beihang 脑血管 PointNet-PINN 源码适配本项目私有数据集后产生的专用预处理结果。

## 目录约定

```text
private_preprocessed/
├── csv_full/       # 可选：每个样本保留完整列的 CSV 快照
├── pkl/            # CROWN 训练/测试直接读取的 pkl
├── manifests/      # split、病例、时间步、输入列、目标列清单
├── stats/          # train-only 归一化统计量
└── audit/          # 点数、列缺失、mask 分布、单位检查报告
```

## 口径

- 不覆盖本项目原有 `processed/`、`graphs/` 或 `outputs/field/`。
- 预处理阶段始终保留全量显式几何特征；训练阶段通过配置中的 `input_features` mask 决定实际输入列。
- 不强制使用 PyG。该目录优先保存 CROWN 源码可直接读取的 numpy/pkl 中间层。
- 大体积 `*.pkl`、`*.npz`、`*.csv` 和集群输出不建议纳入 Git；提交前只保留 README、配置、manifest/audit 的小型文本摘要。
