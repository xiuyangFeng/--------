# PointNetCFD

## 基本信息

- 论文：A point-cloud deep learning framework for prediction of fluid flow fields on irregular geometries
- 代码：`https://github.com/Ali-Stanford/PointNetCFD`
- 本项目优先级：P0

## 方法要点

PointNetCFD 使用 shared MLP + global max pooling + point-wise decoder 在不规则几何点云上预测 CFD 流场。它不依赖 mesh connectivity，是建立点云外部 baseline 的最低成本入口。

## 与本项目的关系

适合回答：在不使用图结构时，公开 PointNet-style 方法能否在本项目私有数据上达到可接受的 `u/v/w/p` 或近壁质量。

## 最小复现路线

1. 将本项目采样点导出为 PointNetCFD 可读格式。
2. 输入先只用 `x,y,z,t,BC,is_wall`。
3. 输出先对齐 `u/v/w/p`。
4. 再加入显式几何特征做 ablation。
5. 最后才尝试 WSS head。

## 本项目代码落地（2026-06-15）

已在 `external_baselines/pointnetcfd/` 建立独立复现代码，不改任务 A V1/V2/V3 主训练入口。当前实现直接读取本项目 PyG `.pt` 图快照，但忽略 `edge_index`，保留 PointNetCFD 的核心结构：shared point MLP、global max pooling、point-wise decoder。

代码不只是单个 trainer，当前包括：

- `data.py`：私有 PyG 图快照到 PointNet 点云样本的 adapter；
- `model.py`：PointNetCFD-style 网络；
- `train.py`：训练与 checkpoint；
- `evaluate.py`：独立评估、按病例/按 sample 指标和预测导出；
- `audit_dataset.py`：集群提交前的数据兼容性审计；
- `metrics.py` / `utils.py`：指标与通用工具；
- `configs/` 与 `cluster/`：四组实验模板和 Slurm 提交脚本。

### 已提供配置

| 配置 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| `pointnetcfd_original_vp.json` | `x,y,z + t,BC` | `u,v,w,p` | 最接近 paper-original 的点云速度/压力 baseline |
| `pointnetcfd_wall_vp.json` | `x,y,z,is_wall + t,BC` | `u,v,w,p` | 检查 wall mask 对近壁质量的影响 |
| `pointnetcfd_geom_vp.json` | `x,y,z,is_wall,Abscissa,NormRadius,Curvature,Tangent + t,BC` | `u,v,w,p` | 验证显式几何特征对速度/压力的增益 |
| `pointnetcfd_geom_pwss.json` | 同上 | wall-only `p,wss_x,wss_y,wss_z` | 验证压力 + WSS 直接预测路线 |

### 集群使用入口

```bash
bash external_baselines/pointnetcfd/cluster/submit_pointnetcfd.sh \
  external_baselines/pointnetcfd/configs/pointnetcfd_original_vp.json
```

提交前需要把配置中的 `data_root` 与 `split_file` 改为集群私有数据路径。建议复制配置后修改，保留原始模板。

训练前建议先运行：

```bash
python -m external_baselines.pointnetcfd.audit_dataset \
  --config external_baselines/pointnetcfd/configs/pointnetcfd_geom_pwss.json
```

训练后可运行：

```bash
python -m external_baselines.pointnetcfd.evaluate \
  --checkpoint outputs/external_baselines/pointnetcfd/<run_dir>/best_model.pt \
  --split test \
  --save-predictions
```

## 风险

- PointNet 的全局 pooling 对局部边界层梯度可能不足。
- 只要速度/压力改善，不代表 WSS 后处理一定改善。
