"""
wss_utils 单元测试 / 可视化检查脚本
==================================
用途：
  1. 验证向量模长、正交分解是否与手工真值一致
  2. 画出三点拟合曲线与壁面梯度，直观对比 parabolic / linear

运行：
  cd src/
  python wss_utils_test.py

注意：
  test_calculate_gradient 会弹出 matplotlib 窗口（共 4 组图），
  关闭当前窗口后才会进入下一组。
"""

import logging
import wss_utils
import numpy as np
import matplotlib.pyplot as plt


def test_vector_magnitude():
    """
    测试 get_vector_magnitude：给定两个三维向量，模长应与手工结果一致。
    """
    v1 = np.asarray([[0.370491361, -0.167037462, -0.225748788],
                     [0.585235848, -0.099175084, -0.116312987]])
    v1_mag = wss_utils.get_vector_magnitude(v1)

    # 预先算好的真值（欧氏范数）
    true_mag = [0.464895555, 0.60486809]
    print('calculated mag', v1_mag)
    # 原代码这里第二行也打印了 v1_mag（应为 true_mag），保留原行为
    print('actual mag', v1_mag)
    np.testing.assert_almost_equal(v1_mag, true_mag)

    print('test_vector_magnitude PASS')


def test_orthogonal_vectors():
    """
    测试 get_orthogonal_vectors：速度分解为法向 / 切向后，应与真值一致。

    验证关系：
      v = v_normal + v_tangent
      v_normal // n
      v_tangent · n ≈ 0
    """
    # 输入速度（两行相同，便于对照不同法向）
    v1 = np.asarray([[0.64604717, 0.0196206, 0.08837089],
                     [0.64604717, 0.0196206, 0.08837089]])
    # 两个不同的壁面法向
    normals = np.asarray([[0.6021105, 0.40786213, 0.6863755],
                          [0.24887183, 0.48617426, 0.8376738]])

    normal_vectors, tangent_vectors = wss_utils.get_orthogonal_vectors(v1, normals)

    # 手工 / 预先计算的分解结果
    true_normal = np.asarray([
        [0.275555809, 0.186658062, 0.314119678],
        [0.060811322, 0.118795684, 0.204683877]
    ])
    true_tangent = np.asarray([
        [0.370491361, -0.167037462, -0.225748788],
        [0.585235848, -0.099175084, -0.116312987]
    ])

    np.testing.assert_almost_equal(normal_vectors, true_normal)
    np.testing.assert_almost_equal(tangent_vectors, true_tangent)

    print('test_orthogonal_vectors PASS')


def test_calculate_gradient(inward_distance=1, parabolic=True):
    """
    测试 / 可视化壁面速度梯度拟合。

    对 6 个「假想壁面点」，给定 pc0/pc1/pc2 切向速度模长，
    调用内部函数得到拟合曲线，并在 2×3 子图中画出：
      - 圆点：原始三点样本
      - 曲线：多项式拟合结果
      - 标题：壁面处数值梯度

    参数
    ----
    inward_distance : float
        采样层间距（测试里试 1.0 与 0.7）。
    parabolic : bool
        True → 加密网格抛物线风格；False → 三点线性风格。
    """
    # pc0 全 0：模拟 no-slip（壁面速度为 0）
    pc0_tangent_mag = np.asarray([0, 0, 0, 0, 0, 0])
    # pc1 / pc2：人为设定的内层切向速度，最后一列来自上面正交分解测试的模长
    pc1_tangent_mag = np.asarray([1, 4, 3, 1, 2, 0.46489556])
    pc2_tangent_mag = np.asarray([2, 5, 5, 4, 5, 0.60486809])

    # 直接调用带曲线返回值的内部函数，便于画图
    g, xx, yy = wss_utils._calculate_gradient_with_values(
        pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag,
        inward_distance, use_parabolic=parabolic
    )

    # 原始采样点的 x / y，用于散点叠加
    x = np.array([0, 1, 2]) * inward_distance
    y = np.stack((pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag), axis=1)

    # ---- 画图：每个「壁面点」一个子图 ----
    fig = plt.figure(1)
    main_title = "Parabolic" if parabolic else "Linear"
    main_title = "{} x={}".format(main_title, inward_distance)
    fig.suptitle(main_title)

    for i in range(0, len(y)):
        y_new = yy[i]
        # 再算一遍该点曲线的梯度，取壁面处（索引 0）写进标题
        g = np.gradient(y_new, xx)
        # print('wall_gradient', g[0])

        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(x, y[i], 'o', xx, y_new)  # 'o'=原始点；曲线=拟合
        ax.title.set_text("Wall gradient {:.2f}".format(g[0]))

    plt.show()
    plt.clf()


if __name__ == "__main__":
    # 打开工具模块 DEBUG 日志，便于观察分解过程
    logging.getLogger("wss_utils").setLevel(logging.DEBUG)

    test_vector_magnitude()
    test_orthogonal_vectors()

    # 四组可视化：抛物线 / 线性 × 两种间距
    test_calculate_gradient(inward_distance=1, parabolic=True)
    test_calculate_gradient(inward_distance=0.7, parabolic=True)

    test_calculate_gradient(inward_distance=1, parabolic=False)
    test_calculate_gradient(inward_distance=0.7, parabolic=False)
