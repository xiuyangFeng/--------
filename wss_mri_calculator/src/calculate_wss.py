"""
从 4D Flow MRI 速度场估算壁面剪切应力（WSS）——主程序
====================================================

整体流程（对应 README 中的 workflow）：
  1. 读入速度三分量 (u,v,w) 与血管掩膜
  2. 建成 PyVista 均匀网格
  3. 提取血管表面并做 Laplacian 平滑
  4. 计算表面点法向（翻转向内，指向管腔）
  5. 沿法向向内取两层等距点 pc1、pc2，与壁面点 pc0 一起采样速度
  6. 把速度分解为法向 / 切向，取切向模长
  7. 对三点切向速度做多项式拟合，求壁面梯度
  8. WSS = 粘度 × 壁面梯度；可视化或导出 VTK

对应论文方法：
  - Vel-LE：      --parabolic False --no-slip False
  - Vel-Wall：    --parabolic False（默认 no-slip=True）
  - Vel-Parabolic：默认（parabolic=True, no-slip=True）
"""

import pyvista as pv
import numpy as np
import argparse
import logging
import data_loader
import wss_utils


def create_uniform_grid(mask, spacing):
    """
    用分割掩膜创建 PyVista 均匀体网格（UniformGrid）。

    注意：
      - UniformGrid 的 dimensions 是「节点数」，而 mask 是按「体素/单元」存的，
        所以 dimensions = mask.shape + 1。
      - 数据按 Fortran 顺序 (order="F") 展平后写入 cell_arrays，
        这与 VTK / PyVista 的单元排列约定一致。

    参数
    ----
    mask : np.ndarray, shape (nx, ny, nz)
        分割掩膜（可为软分割概率）。
    spacing : tuple(float, float, float)
        各轴体素尺寸，单位 mm。

    返回
    ----
    mesh : pv.UniformGrid
        带有 cell 数据 "mask" 的体网格。
    """
    mesh = pv.UniformGrid()
    # 节点数 = 体素数 + 1
    mesh.dimensions = np.array(mask.shape) + 1

    # 如需对齐真实世界坐标，可在此设置原点，例如：
    # mesh.origin = (100, 33, 55.6)
    mesh.spacing = spacing  # 各轴体素边长 (mm)
    # 按列优先展平，写入单元（cell）标量
    mesh.cell_arrays["mask"] = mask.flatten(order="F")
    return mesh


def create_uniform_vector(u, v, w, spacing):
    """
    用速度三分量创建均匀体网格，并附加速度标量与分量。

    参数
    ----
    u, v, w : np.ndarray
        同形状的速度分量（m/s）。
    spacing : tuple
        体素尺寸 (mm)。

    返回
    ----
    mesh : pv.UniformGrid
        含 cell 数据：u, v, w, Velocity（速度模长）。
        当前活动标量设为 "Velocity"，便于着色显示。
    """
    # 速度模长 |v| = sqrt(u^2 + v^2 + w^2)
    vel = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(u.shape) + 1
    # mesh.origin = (100, 33, 55.6)  # 如需真实坐标原点可打开
    mesh.spacing = spacing

    # 三个速度分量分别写入 cell 数组
    mesh.cell_arrays["u"] = u.flatten(order="F")
    mesh.cell_arrays["v"] = v.flatten(order="F")
    mesh.cell_arrays["w"] = w.flatten(order="F")

    # 速度模长，用作默认显示标量
    mesh.cell_arrays["Velocity"] = vel.flatten(order="F")
    mesh.set_active_scalars("Velocity")
    return mesh


def boolean_string(s):
    """
    argparse 辅助：把命令行字符串 'True'/'False' 转成真正的 bool。

    直接用 type=bool 是不行的，因为 bool("False") == True（非空字符串）。
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main():
    # =========================
    # 命令行参数
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='../data/aorta03_sample.h5',
                        help='速度场 HDF5（含 u,v,w）')
    parser.add_argument('--mask-file', type=str, default='../data/aorta03_sample.h5',
                        help='分割掩膜 HDF5（可与速度同文件）；换 STL 需另写 loader')
    parser.add_argument('--voxel-size', type=float, default=0.6,
                        help='体素尺寸，假定各向同性 (mm)')
    parser.add_argument('--inward-distance', type=float, default=0.6,
                        help='沿内法向采样点间距 (mm)；通常取 ≈ 1 个体素')
    parser.add_argument('--smoothing', type=int, default=500,
                        help='表面 Laplacian 平滑迭代次数')
    parser.add_argument('--parabolic', type=boolean_string, default=True,
                        help='True=抛物线拟合求斜率；False=线性风格')
    parser.add_argument('--no-slip', type=boolean_string, default=True,
                        help='True=壁面速度强制为 0（无滑移）；False=用表面插值速度')
    parser.add_argument('--viscosity', type=float, default=4,
                        help='动力粘度 (centiPoise)；血液默认 4 cP')
    parser.add_argument('--show-plot', type=boolean_string, default=True,
                        help='是否弹出 PyVista 可视化窗口')
    parser.add_argument('--show-wss-contours', type=boolean_string, default=False,
                        help='是否在 WSS 表面上叠加等值线')
    # 注意：原仓库此处 type=float，实际当作布尔用（0/1 或 False）；保持原行为不改
    parser.add_argument('--save-to-vtk', type=float, default=False,
                        help='是否保存体积与表面 VTK（非 0 即保存）')
    parser.add_argument('--vtk-filename', type=str, default='result',
                        help='VTK 输出文件名前缀')
    parser.add_argument('--loglevel', type=int, default=logging.INFO,
                        help='日志级别，如 logging.INFO=20, DEBUG=10')

    args = parser.parse_args()

    print('\nArguments:', args)
    # 同步设置工具模块的日志级别
    logging.getLogger("wss_utils").setLevel(args.loglevel)

    logging.basicConfig()
    logger = logging.getLogger('calculate_wss')
    logger.setLevel(args.loglevel)

    # =========================
    # 运行参数整理
    # =========================
    # 若体素非各向同性，可在此手动改成 (sx, sy, sz)
    spacing = (args.voxel_size, args.voxel_size, args.voxel_size)  # mm
    inward_distance = args.inward_distance
    smoothing_iteration = args.smoothing

    no_slip_condition = args.no_slip
    parabolic_fitting = args.parabolic
    # 粘度单位 cP；速度单位 m/s；两者搭配使 WSS 落在常用力学量级
    viscosity = args.viscosity

    input_filepath = args.input_file
    mask_filepath = args.mask_file

    # ------------------------------------------------------------------
    # 1. 加载速度场与分割掩膜
    #    若数据格式不是示例 HDF5，请替换 data_loader 中的函数。
    # ------------------------------------------------------------------
    logger.info("Loading velocity vectors")
    # 取时间帧 idx=0；返回同形状的 u,v,w
    u, v, w = data_loader.load_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
    logger.debug("Image shape {}".format(u.shape))

    logger.info("Load segmentation")
    mask = data_loader.load_segmentation(mask_filepath, 'mask', 0)

    # ------------------------------------------------------------------
    # 2. 构建均匀网格；对掩膜做百分位阈值，得到血管区域网格
    # ------------------------------------------------------------------
    logger.info("Constructing uniform grids for vectors and mask")
    velocity = create_uniform_vector(u, v, w, spacing)
    mesh = create_uniform_grid(mask, spacing)
    # threshold_percent(40)：保留 mask 值高于约 40% 分位的单元，抽出血管体
    # 也可改用绝对阈值，例如：mesh = mesh.threshold([0.3, 0.8])
    mesh = mesh.threshold_percent(40)

    # ------------------------------------------------------------------
    # 3. 在掩膜区域内采样速度 —— 得到带速度的体积（主要用于可视化）
    # ------------------------------------------------------------------
    logger.info("Get volume")
    volume = mesh.sample(velocity)
    volume.set_active_scalars("Velocity")

    # ------------------------------------------------------------------
    # 4. 提取表面并平滑
    #    注意：不要用 extract_geometry（官网示例容易误导）；
    #    extract_surface() 才是从体网格得到表面网格的正确方式。
    # ------------------------------------------------------------------
    logger.info("Extracting surface")
    surf = mesh.extract_surface()
    # Laplacian 平滑：减少阶梯状体素表面噪声，但次数过多会收缩/抹平细节
    surf = surf.smooth(n_iter=smoothing_iteration)

    # ------------------------------------------------------------------
    # 5. 计算表面法向
    #    flip_normals=True：把法向翻转向内（指向管腔），后续「向内」采样才正确。
    #    点法向与单元法向数组名可能相同，这里明确要求算点法向（WSS 按点算）。
    # ------------------------------------------------------------------
    logger.info("Computing surface normals")
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=True)
    logger.debug("{} {} normal points".format(surf.point_normals[0:2], len(surf.point_normals)))
    logger.debug("{} {} points coords".format(surf.points[0:2], len(surf.points)))

    # ------------------------------------------------------------------
    # 6. 构造三层点云并采样速度
    #    pc0：壁面点（表面顶点）
    #    pc1：沿内法向移动 1 * inward_distance
    #    pc2：沿内法向移动 2 * inward_distance
    #    sample(velocity)：在体速度场上对这些点做插值
    # ------------------------------------------------------------------
    logger.info("Probing velocity from surface and equidistant points {} nodes".format(len(surf.points)))

    # --- pc0：壁面 ---
    pc0 = pv.PolyData(surf.points)
    pc0 = pc0.sample(velocity)
    pc0.set_active_scalars("Velocity")

    # --- pc1：向内第 1 层 ---
    loc1 = pc0.points + (inward_distance * surf.point_normals)
    pc1 = pv.PolyData(loc1)
    pc1 = pc1.sample(velocity)

    # --- pc2：向内第 2 层 ---
    loc2 = pc0.points + (2 * inward_distance * surf.point_normals)
    pc2 = pv.PolyData(loc2)
    pc2 = pc2.sample(velocity)

    # ------------------------------------------------------------------
    # 7. 计算各层切向速度模长
    # ------------------------------------------------------------------
    logger.info("Calculate normal and tangential velocity vectors")

    # --- 壁面层 pc0 ---
    if no_slip_condition:
        # 无滑移假设：固体壁面处流体速度 = 0
        # （推荐；不依赖分割边界上的插值速度，更稳健）
        pc0_tangent_mag = np.zeros(len(pc0.points))
    else:
        # Vel-LE 风格：直接用表面插值得到的速度（对分割误差很敏感）
        pc0_vectors = wss_utils.extract_vectors(pc0)
        pc0_normals, pc0_tangent = wss_utils.get_orthogonal_vectors(pc0_vectors, surf.point_normals)
        pc0_tangent_mag = wss_utils.get_vector_magnitude(pc0_tangent)
    logger.debug("Tangent vector pc0 {}".format(pc0_tangent_mag[0:2]))

    # --- 内层 pc1 ---
    pc1_vectors = wss_utils.extract_vectors(pc1)
    pc1_normals, pc1_tangent = wss_utils.get_orthogonal_vectors(pc1_vectors, surf.point_normals)
    pc1_tangent_mag = wss_utils.get_vector_magnitude(pc1_tangent)
    # 把切向向量存回点云，后面可视化用箭头显示方向
    pc1['vectors'] = pc1_tangent

    # --- 内层 pc2 ---
    pc2_vectors = wss_utils.extract_vectors(pc2)
    pc2_normals, pc2_tangent = wss_utils.get_orthogonal_vectors(pc2_vectors, surf.point_normals)
    pc2_tangent_mag = wss_utils.get_vector_magnitude(pc2_tangent)
    pc2['vectors'] = pc2_tangent

    # === 实验性处理：统一 pc1 / pc2 切向方向的符号 ===
    # 模长本身非负，但若 pc2 切向与 pc1 反向，拟合时曲线会折返，
    # 导致梯度符号/大小失真。这里用点积符号把 pc2 模长改成可正可负：
    #   同向 → 正；反向 → 负。
    # 不用 pc0 做参考，因为 no-slip 时 pc0=0，且非 no-slip 时方向不可靠。
    c = pc1_tangent * pc2_tangent
    c = np.sum(c, axis=1)       # 逐点点积
    c = np.sign(c)              # +1 / -1 / 0
    pc2_tangent_mag = c * pc2_tangent_mag
    # === 结束 ===

    logger.debug("Tangent vector pc1 {}".format(pc1_tangent_mag[0:2]))
    logger.debug("Tangent vector pc2 {}".format(pc2_tangent_mag[0:2]))

    # ------------------------------------------------------------------
    # 8. 三点拟合，求壁面处切向速度梯度 ∂|v_t|/∂n
    # ------------------------------------------------------------------
    logger.info("Spline fitting and calculating gradients ...")
    gradients = wss_utils.calculate_gradient(
        pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag,
        inward_distance, use_parabolic=parabolic_fitting
    )
    logger.debug("gradients {}".format(gradients[0:2]))

    # ------------------------------------------------------------------
    # 9. WSS = μ * (∂v_t / ∂n)|_wall
    #    并把结果写回表面网格，供显示 / 导出
    # ------------------------------------------------------------------
    logger.info("Calculating WSS")
    surf["wss"] = gradients * args.viscosity
    # 用 pc1 的切向向量作为 WSS「方向」示意（仅可视化，不是严格矢量 WSS）
    surf['wss_vectors'] = pc1_tangent

    # ------------------------------------------------------------------
    # 10. 可选：保存体积与表面 VTK
    # ------------------------------------------------------------------
    # TODO（原作者）：希望将来把 volume 与 surface 合并进同一个 vtk
    if args.save_to_vtk:
        logger.info("Saving to {}_surface.vtk".format(args.vtk_filename))
        volume.save("{}_volume.vtk".format(args.vtk_filename))
        surf.save("{}_surface.vtk".format(args.vtk_filename))

    # ------------------------------------------------------------------
    # 11. 可选：2×2 可视化面板
    # ------------------------------------------------------------------
    if args.show_plot:
        logger.info("Preparing plot...")
        # notebook=0：独立窗口；shape=(2,2)：四个子图
        p = pv.Plotter(notebook=0, shape=(2, 2), border=False)

        # --- 左上：血管体素 / 速度体积 ---
        p.subplot(0, 0)
        p.add_text("Voxels\n{} mm".format(args.voxel_size), font_size=20)
        p.add_mesh(volume, show_edges=True, cmap='jet')

        # --- 右上：平滑后表面 + 法向箭头 ---
        p.subplot(0, 1)
        p.add_text("Surface (n={})".format(smoothing_iteration), font_size=20)
        result = surf.sample(velocity)
        p.add_mesh(result, scalars="Velocity", opacity=0.7, cmap='viridis')
        # glyph：在表面稀疏位置画法向箭头，检查法向是否朝内
        arrows = surf.glyph(scale="Normals", orient="Normals", tolerance=0.05)
        p.add_mesh(arrows, color="black")
        p.show_bounds(all_edges=True)

        # --- 左下：三层采样点；pc1/pc2 用切向箭头 ---
        p.subplot(1, 0)
        p.add_text("Wall points\n(inward {})".format(inward_distance), font_size=20)
        p.add_mesh(pc0, cmap='jet', opacity=0.5)
        pc1_arrows = pc1.glyph(orient='vectors', scale=False, factor=0.6)
        pc2_arrows = pc2.glyph(orient='vectors', scale=False, factor=0.6)
        p.add_mesh(pc1_arrows, color='black')  # 第 1 内层
        p.add_mesh(pc2_arrows, color='red')    # 第 2 内层

        # --- 右下：最终 WSS 表面图 ---
        p.subplot(1, 1)
        fitting_functext = "parabolic" if parabolic_fitting else "linear"
        p.add_text("WSS ({})".format(fitting_functext), font_size=20)
        p.add_mesh(surf, scalars="wss", cmap='jet')

        if args.show_wss_contours:
            # 叠加 WSS 等值线，便于看空间分布结构
            contours = surf.contour()
            p.add_mesh(contours, color='black', line_width=1)

        # 四个视图相机联动，方便对比同一视角
        p.link_views()
        p.show(full_screen=True)


if __name__ == "__main__":
    main()
