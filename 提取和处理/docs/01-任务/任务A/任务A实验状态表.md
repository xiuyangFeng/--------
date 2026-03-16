# 任务A 实验状态总表

> 本表是任务 A 所有实验的唯一执行状态追踪文件。  
> 每次启动、完成或失败一组实验，**必须更新此表**。  
> 上位文档：[任务A实验清单](任务A实验清单.md) / [任务A冻结卡](任务A冻结卡.md)

---

## 状态说明

| 状态标记 | 含义 |
|---|---|
| 🔒 未开始 | 尚未生成配置或提交 |
| 🔬 待 smoke test | 配置已就绪，等待最小闭环验证 |
| 🚀 进行中 | 至少一个 seed 已启动 |
| 🌱 待补 seed | seed=1 已通过，等待补 seed=2,3 |
| 📋 待汇总 | 训练完成，等待写入记录表 |
| ✅ 已完成 | 结果、图表、实验记录均已归档 |
| ❌ 失败待重跑 | 出现错误或配置问题，需修复后重跑 |

---

## 第一批：基线实验

| Exp ID | 研究问题 | split_version | seeds | 当前状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|---|
| A-Base-01 | 点模型下限（无图结构） | split_AG_v1 | [1,2,3] | 🔒 未开始 | outputs/field/field_mlp_coord_t_bc_seed{seed}/ | 等待数据处理完成 |
| A-Base-02 | 图结构是否必要 | split_AG_v1 | [1,2,3] | 🔒 未开始 | outputs/field/field_graphsage_coord_t_bc_wall_seed{seed}/ | |
| A-Base-03 | Transformer 无 geometry 对照 | split_AG_v1 | [1,2,3] | 🔒 未开始 | outputs/field/field_transformer_coord_t_bc_wall_seed{seed}/ | |
| A-Main-01 | Transformer + geometry 主模型 | split_AG_v1 | [1,2,3] | 🔒 未开始 | outputs/field/field_transformer_coord_t_bc_geom_wall_seed{seed}/ | |

---

## 第二批：必做消融

| Exp ID | 研究问题 | 唯一变化项 | split_version | seeds | 当前状态 | 备注 |
|---|---|---|---|---|---|---|
| A-Abl-01-01 | 输入特征消融 | coords + t 仅坐标+时间 | split_AG_v1 | [1] | 🔒 未开始 | 依赖 A-Main-01 完成 |
| A-Abl-01-02 | 输入特征消融 | + BC | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-01-03 | 输入特征消融 | + is_wall | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-01-04 | 输入特征消融 | + geometry（无 wall） | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-02-01 | 几何分量消融 | 去掉 Abscissa | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-02-02 | 几何分量消融 | 去掉 NormRadius | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-02-03 | 几何分量消融 | 去掉 Curvature | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-02-04 | 几何分量消融 | 去掉 Tangent | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-03-01 | 坐标归一化消融 | 原始坐标 | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-03-02 | 坐标归一化消融 | 仅中心化 | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-03-03 | 坐标归一化消融 | 中心化+PCA对齐 | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-03-04 | 坐标归一化消融 | 中心化+PCA+缩放（当前版本） | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-04-01 | 增强消融 | 无增强 | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-04-02 | 增强消融 | 仅旋转 | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-04-03 | 增强消融 | 旋转+平移（默认） | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-04-04 | 增强消融 | 旋转+平移+微扰 | split_AG_v1 | [1] | 🔒 未开始 | |
| A-Abl-05-01 | 物理约束消融 | 仅数据损失 | split_AG_v1 | [1] | 🔒 未开始 | 依赖主线稳定后 |
| A-Abl-05-02 | 物理约束消融 | + continuity | split_AG_v1 | [1] | 🔒 未开始 | |

---

## 主结果表（持续更新）

> 每完成一组 seed=1 后填入此表，多 seed 后更新为 mean±std。

| Exp ID | Model | Geom | BC | is_wall | Physics | RMSE_u | RMSE_v | RMSE_w | RMSE_\|v\| | RMSE_p | R2_p | Infer(ms) | Mem(MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-Base-01 | MLP | ✗ | ✓ | ✗ | ✗ | - | - | - | - | - | - | - | - |
| A-Base-02 | GraphSAGE | ✗ | ✓ | ✓ | ✗ | - | - | - | - | - | - | - | - |
| A-Base-03 | Transformer | ✗ | ✓ | ✓ | ✗ | - | - | - | - | - | - | - | - |
| A-Main-01 | Transformer | ✓ | ✓ | ✓ | ✗ | - | - | - | - | - | - | - | - |

---

## 实验记录摘要（每完成一组填写）

### A-Base-01

- **完成日期**：-
- **seed**：-
- **best epoch**：-
- **RMSE_|v|**：-
- **RMSE_p**：-
- **壁面点误差**：-
- **内部点误差**：-
- **推理时间**：-
- **峰值显存**：-
- **一句话结论**：-
- **下一步动作**：-

---

### A-Base-02

- **完成日期**：-
- **seed**：-
- **best epoch**：-
- **RMSE_|v|**：-
- **RMSE_p**：-
- **壁面点误差**：-
- **内部点误差**：-
- **推理时间**：-
- **峰值显存**：-
- **一句话结论**：-
- **下一步动作**：-

---

### A-Base-03

- **完成日期**：-
- **seed**：-
- **best epoch**：-
- **RMSE_|v|**：-
- **RMSE_p**：-
- **壁面点误差**：-
- **内部点误差**：-
- **推理时间**：-
- **峰值显存**：-
- **一句话结论**：-
- **下一步动作**：-

---

### A-Main-01

- **完成日期**：-
- **seed**：-
- **best epoch**：-
- **RMSE_|v|**：-
- **RMSE_p**：-
- **壁面点误差**：-
- **内部点误差**：-
- **推理时间**：-
- **峰值显存**：-
- **一句话结论**：-
- **下一步动作**：-
