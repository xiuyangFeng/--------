"""
Pipeline - 新数据格式完整处理流程

包含以下模块:
- config: 配置参数
- preprocess: 数据清洗+合并+降采样
- extract_features: 几何特征提取+边界条件
- normalize: 特征归一化
- convert_to_graph: 转换为图数据
- run_all: 一键运行全流程

数据源: data_new/
- AG/fast: 动脉移植物快速增长
- AG/slow: 动脉移植物慢速增长 (待放入)
- AAA/rupture: 腹主动脉瘤破裂 (待放入)
- AAA/unrupture: 腹主动脉瘤未破裂 (待放入)
- ILO/sq, ILO/sh: 髂支闭塞 (待放入)
"""

__version__ = "1.0.0"
