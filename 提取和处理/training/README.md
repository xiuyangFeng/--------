# Training

这个目录是独立于 `pipeline/` 的训练脚手架，优先服务任务 A 的场重建实验。

## 目录说明

- `config.py`：实验配置 dataclass
- `splits.py`：患者级数据划分读取
- `data.py`：图数据集、增强和特征掩码
- `models.py`：`MLP / GraphSAGE / Transformer` 模型注册
- `losses.py`：训练损失
- `metrics.py`：回归指标
- `trainer.py`：训练循环和早停
- `train_field.py`：任务 A 训练入口
- `eval_field.py`：独立评估入口
- `predict_field.py`：批量预测导出入口
- `make_split.py`：根据病例名单生成患者级 split
- `io.py`：checkpoint 与实验索引落盘
- `configs/field/`：第一批基线实验模板
- `configs/field/ablations/`：第一批消融模板
- `splits/`：split 样例模板

## 设计原则

1. 不修改现有 `pipeline/` 数据处理流程。
2. 所有特征消融通过“特征掩码”实现，不改图文件结构。
3. 第一阶段只服务任务 A，先把 `MLP / GraphSAGE / Transformer` 跑通。
4. 后续加 physics loss、任务 B、任务 C 时直接扩展这套骨架。

## 使用方式

先准备一个患者级 split JSON，再选择一个配置模板：

```bash
python -m training.train_field --config training/configs/field/transformer_geometry.json
```

如果本机没有数据、只是在本地准备脚手架，可以先在服务器上根据病例名单生成 split：

```bash
python -m training.make_split \
  --cases-file /path/to/case_names.txt \
  --output training/splits/split_v1.json \
  --split-version split_v1 \
  --source AG/fast
```

独立评估：

```bash
python -m training.eval_field \
  --config training/configs/field/transformer_geometry.json \
  --checkpoint outputs/field/<run_dir>/best_model.pt \
  --subset test
```

批量导出预测结果：

```bash
python -m training.predict_field \
  --config training/configs/field/transformer_geometry.json \
  --checkpoint outputs/field/<run_dir>/best_model.pt \
  --subset test
```

安装依赖可参考：

```bash
pip install -r training/requirements.txt
```

如果服务器使用 GPU，请按实际 CUDA 版本单独安装匹配的 `torch` 和 `torch_geometric` 轮子，不要机械照搬 CPU 默认安装。

## split 文件示例

见：

- `training/splits/split_example.json`

注意：

1. 当前仓库本机目录里没有实际训练数据，因此这里没有生成真实的患者划分文件。
2. 真实 split 必须在服务器上根据实际病例名单生成或手工整理。
3. `data_root` 需要改成服务器上与 `pipeline/README.md` 一致的数据路径。

## 第一批模板配置

- `mlp_baseline.json`
- `graphsage_baseline.json`
- `transformer_no_geometry.json`
- `transformer_geometry.json`

第一批常用消融模板：

- `ablations/transformer_coords_t_bc.json`
- `ablations/transformer_no_bc.json`
- `ablations/transformer_no_is_wall.json`
