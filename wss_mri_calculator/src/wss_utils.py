"""
WSS 计算工具函数
================
本模块实现壁面剪切应力（Wall Shear Stress, WSS）计算中的核心数学操作：

  1. 从 PolyData 提取速度向量
  2. 将速度分解为法向分量与切向分量
  3. 计算向量模长
  4. 对壁面附近三点的切向速度做多项式拟合，求壁面处速度梯度

物理背景（简要）
----------------
壁面剪切应力 τ ≈ μ * (∂v_tangential / ∂n)|_wall
其中：
  - μ 是动力粘度（血液约 4 cP）
  - v_tangential 是沿壁面切向的速度
  - n 是壁面法向（指向管内）

本仓库沿着法向取 3 个等距点（壁面 pc0、向内 1 步 pc1、向内 2 步 pc2），
拟合 v_t(n)，再在壁面处求导得到速度梯度，最后乘粘度得到 WSS。
"""

import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger('wss_utils')


def extract_vectors(polydata):
    """
    从 PyVista PolyData 中提取速度三分量，并堆叠成 (n, 3) 向量数组。

    前置条件：
      polydata 上已存在标量数组 "u", "v", "w"（通常由 sample(velocity) 插值得到）。

    参数
    ----
    polydata : pv.PolyData
        带有 u/v/w 点数据的点云或表面。

    返回
    ----
    vector : np.ndarray, shape (n, 3)
        第 i 个点的速度向量 [u_i, v_i, w_i]。
    """
    u = polydata["u"]
    v = polydata["v"]
    w = polydata["w"]
    # axis=-1：把三个长度为 n 的一维数组堆成 (n, 3)
    vector = np.stack((u, v, w), axis=-1)
    return vector


def get_orthogonal_vectors(vectors, point_normals):
    """
    将速度向量分解为「沿壁面法向」与「沿壁面切向」两部分。

    数学过程（对每个点）：
      设速度 v、单位法向 n：
        v_normal  = (v · n) * n          # 法向投影
        v_tangent = v - v_normal         # 切向剩余（垂直于 n）

    WSS 关心的是切向速度沿法向的变化率，因此后续只用 tangent 分量。

    参数
    ----
    vectors : np.ndarray, shape (n, 3)
        各点处的速度向量。
    point_normals : np.ndarray, shape (n, 3)
        各点处的壁面法向（一般已单位化；方向约定：本脚本中 flip 后指向管内）。

    返回
    ----
    normal_vectors : np.ndarray, shape (n, 3)
        速度的法向分量。
    tangent_vectors : np.ndarray, shape (n, 3)
        速度的切向分量（用于后续梯度 / WSS 计算）。
    """
    logger.debug("Get orthogonal vectors")
    logger.debug('Actual vector {}'.format(vectors[0:2]))

    # ---- 计算标量投影系数 c = v · n ----
    # 先逐元素相乘再按分量求和，等价于逐点点积
    c = vectors * point_normals
    c = np.sum(c, axis=1)  # shape: (n,)

    # ---- 法向分量 = c * n；切向分量 = v - 法向分量 ----
    # c[:, np.newaxis] 把 (n,) 扩成 (n, 1)，便于与 (n, 3) 广播相乘
    normal_vectors = c[:, np.newaxis] * point_normals
    tangent_vectors = vectors - normal_vectors

    logger.debug('Normal vector {}'.format(normal_vectors[0:2]))
    logger.debug('Tangent vector {}'.format(tangent_vectors[0:2]))

    return normal_vectors, tangent_vectors


def get_vector_magnitude(vectors):
    """
    计算一组三维向量的模长 |v|。

    参数
    ----
    vectors : np.ndarray, shape (n, 3)

    返回
    ----
    c : np.ndarray, shape (n,)
        每个向量的欧氏范数 sqrt(vx^2 + vy^2 + vz^2)。
    """
    c = vectors * vectors       # 逐元素平方
    c = np.sum(c, axis=1)       # 三个分量相加
    c = c ** 0.5                # 开方得到模长
    return c


def _calculate_gradient_with_values(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag,
                                    inward_distance, use_parabolic):
    """
    对每个壁面点，用沿法向的 3 个切向速度样本做多项式拟合，并求壁面处梯度。

    采样布局（沿壁面内法向）：
      距离 n = 0               : pc0（壁面）
      距离 n = inward_distance : pc1
      距离 n = 2*inward_distance : pc2

    拟合策略：
      - 始终用 2 次多项式（3 点恰好定一个抛物线，degree = len(x)-1 = 2）
      - use_parabolic=True 时：在更密的 x_new 上求值，再用 np.gradient 求导
        （对应论文中的抛物线 / Vel-Parabolic 方法）
      - use_parabolic=False 时：只在原始 3 个 x 上求值再求梯度
        （更接近线性插值 / Vel-Wall 方法的数值行为）

    参数
    ----
    pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag : np.ndarray, shape (n,)
        三层点云各自的切向速度模长。
        若启用 no-slip，pc0 通常全为 0。
    inward_distance : float
        相邻采样层之间的法向间距（mm）。
    use_parabolic : bool
        True → 加密采样后再求导；False → 直接在 3 点上求导。

    返回
    ----
    gradients : np.ndarray, shape (n,)
        每个壁面点在 n=0 处的 ∂|v_t|/∂n（壁面速度梯度）。
    x_new : np.ndarray
        用于求值的法向坐标网格（调试 / 测试绘图用）。
    y_new : np.ndarray, shape (n, len(x_new))
        拟合曲线在 x_new 上的速度值（调试 / 测试绘图用）。

    参考
    ----
    多行同时 polyfit 的写法来自：
    https://stackoverflow.com/questions/20202710/numpy-polyfit-and-polyval-in-multiple-dimensions
    """
    logger.info("Calculating gradient for {} points".format(pc1_tangent_mag.shape))

    # ---- 构造法向坐标 x：三层点距离壁面分别为 0, d, 2d ----
    x = np.array([0, 1, 2])          # 无量纲层号
    x = x * inward_distance          # 乘间距得到真实距离（mm）

    # ---- 把三层速度排成 (n, 3)，每行是一个壁面点的 (v0, v1, v2) ----
    y = np.stack((pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag), axis=1)

    # polyfit 期望 y 的形状为 (len(x), n_samples)，因此需要转置
    y = np.transpose(y)  # 现在是 (3, n)
    # 对每个样本拟合 degree=2 的多项式；z 的形状为 (3, n)，即各阶系数
    z = np.polynomial.polynomial.polyfit(x, y, len(x) - 1)

    # ---- 选择求值网格 ----
    if use_parabolic:
        # 抛物线模式：在 [0, 2d] 上加密到 15 个点，曲线更光滑，数值梯度更稳
        x_new = np.linspace(x[0], x[-1], len(x) * 5)
    else:
        # 线性模式：仍用原始 3 个采样点
        x_new = x

    # ---- 用拟合系数在 x_new 上求值 ----
    # y_new 形状约为 (n, len(x_new))
    y_new = np.polynomial.polynomial.polyval(x_new, z)

    # ---- 沿法向坐标求数值梯度 ∂y/∂x ----
    # axis=1：对每个壁面点各自的速度曲线求导
    gg = np.gradient(y_new, x_new, axis=1)

    # 只要壁面处（n=0，即第 0 列）的梯度
    return gg[:, 0], x_new, y_new


def calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag,
                       inward_distance, use_parabolic=True):
    """
    计算壁面切向速度梯度的对外接口（正式计算只用梯度，不返回曲线）。

    参数含义同 `_calculate_gradient_with_values`。

    返回
    ----
    gradients : np.ndarray, shape (n,)
        壁面处 ∂|v_t|/∂n。后续在主程序中乘以粘度 μ 即得 WSS。
    """
    gradients, x_, y_ = _calculate_gradient_with_values(
        pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag,
        inward_distance, use_parabolic
    )
    # x_, y_ 仅供测试脚本画拟合曲线；正式流程丢弃它们
    return gradients
