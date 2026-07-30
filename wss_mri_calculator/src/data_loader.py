"""
数据加载模块
============
负责从 HDF5 文件中读取：
  1. 速度场三分量 (u, v, w) —— 对应 4D Flow MRI 的相位速度图像
  2. 分割掩膜 (mask) —— 标记血管内腔体素

说明：
  - 当前实现只支持 HDF5 格式；若输入是 NIfTI / DICOM / VTK 等，
    需要自行改写本模块中的加载函数。
  - HDF5 中每个字段通常是一个时间序列数组，形状类似 (T, X, Y, Z)；
    通过 idx 选取某一个时间帧。
"""

import numpy as np
import h5py


def load_segmentation(input_filepath, column, idx):
    """
    从 HDF5 文件中加载分割掩膜（某一时间帧）。

    参数
    ----
    input_filepath : str
        HDF5 文件路径。
    column : str
        掩膜在文件中的数据集名称，例如 'mask'。
    idx : int
        时间帧索引。例如 idx=0 表示取第一帧。

    返回
    ----
    m : np.ndarray
        三维掩膜数组，形状 (X, Y, Z)。
        值通常是非二值的概率/软分割（示例数据如此），
        后续会用 threshold_percent 做阈值分割。
    """
    with h5py.File(input_filepath, 'r') as hf:
        # hf.get(column) 取出数据集；[idx] 选取第 idx 个时间帧
        m = np.asarray(hf.get(column)[idx])
    return m


def load_vector_fields(input_filepath, columns, idx):
    """
    从 HDF5 文件中加载速度场三分量（某一时间帧）。

    参数
    ----
    input_filepath : str
        HDF5 文件路径。
    columns : list[str]
        三个数据集名称，依次对应 x/y/z 方向速度，例如 ['u', 'v', 'w']。
        约定：速度单位为 m/s（与粘度单位搭配后得到 Pa 量级的 WSS）。
    idx : int
        时间帧索引。

    返回
    ----
    u, v, w : np.ndarray
        三个同形状的三维数组，分别是 x、y、z 方向的速度分量。
    """
    with h5py.File(input_filepath, 'r') as hf:
        u = np.asarray(hf.get(columns[0])[idx])  # x 方向速度
        v = np.asarray(hf.get(columns[1])[idx])  # y 方向速度
        w = np.asarray(hf.get(columns[2])[idx])  # z 方向速度
    return u, v, w
