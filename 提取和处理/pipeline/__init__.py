"""
Pipeline - 新数据格式完整处理流程

包含以下模块:
- config: 配置参数
- preprocess: 数据清洗+合并+降采样
- extract_features: 几何特征提取+边界条件
- vmtk_core: 主线几何中心线提取与特征计算核心
- coord_normalize: 坐标系归一化（中心化+PCA对齐+缩放）【新增】
- normalize: 特征归一化
- convert_to_graph: 转换为图数据
- augmentation: 数据增强函数（旋转、平移等）【新增】
- dataset: 数据集类（含在线增强）【新增】
- run_all: 一键运行全流程

处理流程（5步）:
1. preprocess: 清洗 + 合并 + 降采样 → processed/merged/
2. extract_features: 几何特征 + 边界条件 → processed/features/
3. coord_normalize: 坐标系归一化 → processed/coord_normalized/
4. normalize: 特征归一化 → processed/normalized/
5. convert_to_graph: 图数据转换 → processed/graphs/

数据增强:
- 在线增强: 使用 pipeline.dataset.CFDAugmentedDataset 在训练时动态增强
- 支持: 随机旋转、随机平移（矢量特征同步变换）

数据源: data_new/
- AG/fast:       动脉移植物-快速增长
- AG/slow:       动脉移植物-慢速增长
- AAA/ruputer:   腹主动脉瘤-破裂
- AAA/unruputer: 腹主动脉瘤-未破裂
- ILO:           髂支闭塞（三层嵌套: ILO/<患者>-<0|1>/<before|after>/）
"""

__version__ = "2.1.0"
